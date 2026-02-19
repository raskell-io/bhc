//! Expression lowering from HIR to Core.
//!
//! This module handles the transformation of HIR expressions to Core
//! expressions. Key transformations include:
//!
//! - `If` expressions become `Case` on booleans
//! - `Lam` with multiple patterns becomes nested lambdas with case
//! - `Tuple` and `List` become constructor applications

use bhc_core::{self as core, Alt, AltCon, Bind, DataCon, Literal, Var, VarId};
use bhc_hir::{self as hir, DefRef, Expr, Lit};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Constraint, Kind, Ty, TyCon};

use crate::context::{has_type_variables, LowerContext};
use crate::pattern::lower_pat_to_alt;
use crate::{LowerError, LowerResult};

/// Format a constraint for error messages.
fn format_constraint(constraint: &Constraint) -> String {
    if constraint.args.is_empty() {
        constraint.class.as_str().to_string()
    } else {
        format!(
            "{} {}",
            constraint.class.as_str(),
            constraint
                .args
                .iter()
                .map(format_type)
                .collect::<Vec<_>>()
                .join(" ")
        )
    }
}

/// Format a type for error messages.
fn format_type(ty: &Ty) -> String {
    match ty {
        Ty::Con(c) => c.name.as_str().to_string(),
        Ty::Var(v) => format!("t{}", v.id),
        Ty::App(f, a) => format!("({} {})", format_type(f), format_type(a)),
        Ty::Fun(a, r) => format!("({} -> {})", format_type(a), format_type(r)),
        Ty::Tuple(ts) => format!(
            "({})",
            ts.iter().map(format_type).collect::<Vec<_>>().join(", ")
        ),
        Ty::List(e) => format!("[{}]", format_type(e)),
        _ => "?".to_string(),
    }
}

/// Create an error expression that will fail at runtime with a message.
fn make_error_expr(msg: &str, span: Span) -> core::Expr {
    let error_var = Var {
        name: Symbol::intern("error"),
        id: VarId::new(0),
        ty: Ty::Error,
    };
    let msg_lit = core::Expr::Lit(Literal::String(Symbol::intern(msg)), Ty::Error, span);
    core::Expr::App(
        Box::new(core::Expr::Var(error_var, span)),
        Box::new(msg_lit),
        span,
    )
}

/// Lower a HIR expression to Core.
pub fn lower_expr(ctx: &mut LowerContext, expr: &hir::Expr) -> LowerResult<core::Expr> {
    match expr {
        Expr::Lit(lit, span) => lower_lit(lit, *span),

        Expr::Var(def_ref) => lower_var(ctx, def_ref),

        Expr::Con(def_ref) => lower_con(ctx, def_ref),

        Expr::App(f, x, span) => lower_app(ctx, f, x, *span),

        Expr::Lam(pats, body, span) => lower_lambda(ctx, pats, body, *span),

        Expr::Let(bindings, body, span) => lower_let(ctx, bindings, body, *span),

        Expr::Case(scrutinee, alts, span) => lower_case(ctx, scrutinee, alts, *span),

        Expr::If(cond, then_br, else_br, span) => lower_if(ctx, cond, then_br, else_br, *span),

        Expr::Tuple(elems, span) => lower_tuple(ctx, elems, *span),

        Expr::List(elems, span) => lower_list(ctx, elems, *span),

        Expr::Record(con_ref, fields, span) => lower_record(ctx, con_ref, fields, *span),

        Expr::FieldAccess(expr, field, span) => lower_field_access(ctx, expr, *field, *span),

        Expr::RecordUpdate(expr, fields, span) => lower_record_update(ctx, expr, fields, *span),

        Expr::Ann(expr, ty, span) => {
            // Check if this is an Integer-annotated literal — if so, create Literal::Integer
            if let Expr::Lit(Lit::Int(n), _) = expr.as_ref() {
                if matches!(ty, Ty::Con(tc) if tc.name.as_str() == "Integer") {
                    return Ok(core::Expr::Lit(
                        Literal::Integer(*n as i128),
                        Ty::Con(TyCon::new(Symbol::intern("Integer"), Kind::Star)),
                        *span,
                    ));
                }
            }
            // Check for negated Integer literal: negate (Int n) :: Integer
            if let Expr::App(f, arg, _) = expr.as_ref() {
                if let Expr::Var(def_ref) = f.as_ref() {
                    let is_negate = ctx
                        .lookup_var(def_ref.def_id)
                        .map(|v| v.name.as_str() == "negate")
                        .unwrap_or(false);
                    if is_negate {
                        if let Expr::Lit(Lit::Int(n), _) = arg.as_ref() {
                            if matches!(ty, Ty::Con(tc) if tc.name.as_str() == "Integer") {
                                return Ok(core::Expr::Lit(
                                    Literal::Integer(-(*n as i128)),
                                    Ty::Con(TyCon::new(Symbol::intern("Integer"), Kind::Star)),
                                    *span,
                                ));
                            }
                        }
                    }
                }
            }
            // Type annotations are erased in Core (types are tracked separately)
            lower_expr(ctx, expr)
        }

        Expr::TypeApp(expr, ty, span) => lower_type_app(ctx, expr, ty, *span),

        Expr::Error(span) => {
            // Generate a runtime error expression
            let error_name = Symbol::intern("error");
            let error_var = Var {
                name: error_name,
                id: VarId::new(0),
                ty: Ty::Error,
            };
            let msg = core::Expr::Lit(
                Literal::String(Symbol::intern("pattern match error")),
                Ty::Error,
                *span,
            );
            Ok(core::Expr::App(
                Box::new(core::Expr::Var(error_var, *span)),
                Box::new(msg),
                *span,
            ))
        }
    }
}

/// Lower a literal to Core.
fn lower_lit(lit: &Lit, span: Span) -> LowerResult<core::Expr> {
    let core_lit = match lit {
        Lit::Int(n) => Literal::Int(*n as i64),
        Lit::Float(f) => Literal::Float(*f as f32),
        Lit::Char(c) => Literal::Char(*c),
        Lit::String(s) => Literal::String(*s),
    };
    Ok(core::Expr::Lit(core_lit, Ty::Error, span))
}

/// Lower a variable reference to Core.
///
/// This handles several cases:
///
/// 1. **Class methods**: If the variable is a class method (like `==` from `Eq`),
///    and we're inside a constrained function, select the method from the
///    appropriate dictionary.
///
/// 2. **Constrained functions**: If the referenced function has type class
///    constraints, apply dictionary arguments from the current scope or
///    resolve instances for concrete types.
///
/// 3. **Regular variables**: Just return a variable reference.
fn lower_var(ctx: &mut LowerContext, def_ref: &DefRef) -> LowerResult<core::Expr> {
    // First, check if this is a class method reference
    let var_name = ctx.lookup_var(def_ref.def_id).map(|v| v.name);

    if let Some(name) = var_name {
        // Check if this is a class method
        let is_method = ctx.is_class_method(name);
        if let Some(class_name) = is_method {
            // This is a class method - we need to select it from a dictionary
            // Look for an in-scope dictionary for this class
            if let Some(dict_var) = ctx.lookup_dict(class_name) {
                // Select the method from the dictionary
                if let Some(method_expr) =
                    ctx.select_method_from_dict(dict_var, class_name, name, def_ref.span)
                {
                    return Ok(method_expr);
                }
            } else if ctx.is_user_class(class_name) {
                // No direct dict in scope — try superclass extraction.
                // If we have MyOrd in scope and need MyEq, extract MyEq from MyOrd.
                if let Some(method_expr) =
                    ctx.select_method_via_superclass(class_name, name, def_ref.span)
                {
                    return Ok(method_expr);
                }
                // Still no dict — don't try to resolve here.
                // The App case in lower_app will handle resolution
                // when the argument type is known.
                let var = ctx.lookup_var(def_ref.def_id).cloned().unwrap();
                return Ok(core::Expr::Var(var, def_ref.span));
            }
            // Builtin class method with no dict — fall through to regular handling
        }
    }

    // Regular variable handling
    let base_expr = if let Some(var) = ctx.lookup_var(def_ref.def_id) {
        core::Expr::Var(var.clone(), def_ref.span)
    } else {
        // Variable not found - this could be a builtin or external reference
        // Create a placeholder variable
        let placeholder = Var {
            name: Symbol::intern("unknown"),
            id: VarId::new(def_ref.def_id.index()),
            ty: Ty::Error,
        };
        core::Expr::Var(placeholder, def_ref.span)
    };

    // Check if the referenced function has user-defined class constraints
    // that need dictionary arguments.
    // IMPORTANT: Only apply dictionary-passing for USER-DEFINED classes.
    // Builtin classes (Eq, Ord, Num, Show, etc.) are handled by codegen's
    // hardcoded dispatch and must NOT go through dictionary construction,
    // because the builtin class registry uses DefIds that don't match the
    // lowering context's actual DefId assignments.
    if let Some(scheme) = ctx.lookup_scheme(def_ref.def_id) {
        // Filter to only user-defined class constraints
        let user_constraints: Vec<_> = scheme
            .constraints
            .iter()
            .filter(|c| ctx.is_user_class(c.class))
            .cloned()
            .collect();

        if !user_constraints.is_empty() {
            // Check if ALL user-class constraints have type variables
            // (meaning they can't be resolved yet — defer to App-level)
            let all_deferred = user_constraints
                .iter()
                .all(|c| c.args.iter().any(has_type_variables));
            if all_deferred {
                return Ok(base_expr);
            }

            // Apply dictionaries for each user-defined class constraint
            let mut result = base_expr;
            for constraint in &user_constraints {
                // Skip constraints with type variables (deferred to App)
                if constraint.args.iter().any(has_type_variables) {
                    continue;
                }

                // Try to resolve the dictionary
                if let Some(dict_expr) = ctx.resolve_dictionary(constraint, def_ref.span) {
                    result =
                        core::Expr::App(Box::new(result), Box::new(dict_expr), def_ref.span);
                } else {
                    // Dictionary not available - generate an error expression
                    let error_msg = format!(
                        "No {} dictionary available for constraint {}",
                        constraint.class.as_str(),
                        format_constraint(constraint)
                    );
                    let error_expr = make_error_expr(&error_msg, def_ref.span);
                    result =
                        core::Expr::App(Box::new(result), Box::new(error_expr), def_ref.span);
                }
            }
            return Ok(result);
        }
    }

    Ok(base_expr)
}

/// Try to infer the concrete type of an HIR expression.
///
/// Returns `Some(Ty)` for expressions with obvious types:
/// - Constructors: look up the constructor's type name
/// - Int/Float/Char/String literals: return the corresponding type
/// - Other expressions: return None (type not inferrable without type checker)
fn try_infer_arg_type(ctx: &LowerContext, expr: &hir::Expr) -> Option<Ty> {
    match expr {
        Expr::Con(def_ref) => {
            // Look up the constructor's data type
            if let Some(con_info) = ctx.lookup_constructor(def_ref.def_id) {
                Some(Ty::Con(TyCon::new(con_info.type_name, Kind::Star)))
            } else {
                None
            }
        }
        Expr::Var(def_ref) => {
            // Look up the type of this variable from the type checker.
            // Only return concrete (non-polymorphic) types.
            let ty = ctx.lookup_type(def_ref.def_id);
            match &ty {
                Ty::Con(_) => Some(ty),
                Ty::Error => {
                    // Core IR params often have ty: Error in type_schemes.
                    // Check the Core Var's type as a fallback (populated from
                    // the function's type signature in compile_equations).
                    if let Some(var) = ctx.lookup_var(def_ref.def_id) {
                        match &var.ty {
                            Ty::Con(_) => Some(var.ty.clone()),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        Expr::Lit(lit, _) => match lit {
            Lit::Int(_) => Some(Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star))),
            Lit::Float(_) => Some(Ty::Con(TyCon::new(Symbol::intern("Double"), Kind::Star))),
            Lit::Char(_) => Some(Ty::Con(TyCon::new(Symbol::intern("Char"), Kind::Star))),
            Lit::String(_) => Some(Ty::List(Box::new(Ty::Con(TyCon::new(
                Symbol::intern("Char"),
                Kind::Star,
            ))))),
        },
        Expr::App(f, x, _) => {
            // Peel off App layers to find head constructor and collect args:
            // App(App(Con(Pair), x), y) → (Con(Pair), [x, y])
            let mut head = f.as_ref();
            let mut con_args = vec![x.as_ref()];
            while let Expr::App(inner_f, inner_x, _) = head {
                con_args.push(inner_x.as_ref());
                head = inner_f.as_ref();
            }
            con_args.reverse();
            if let Expr::Con(def_ref) = head {
                if let Some(con_info) = ctx.lookup_constructor(def_ref.def_id) {
                    let base = Ty::Con(TyCon::new(con_info.type_name, Kind::Star));
                    // Try to determine the result type using the constructor's type
                    // signature. If the type checker has the constructor's type, use it.
                    let con_ty = ctx.lookup_type(def_ref.def_id);
                    if let Some(result_ty) =
                        extract_constructor_result_type(&con_ty, &con_args, ctx)
                    {
                        return Some(result_ty);
                    }
                    // Return bare type name. The caller (lower_app) will try
                    // applied-type resolution as a second attempt if this fails.
                    return Some(base);
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract the result type from a constructor's type signature, substituting
/// type variables based on inferred argument types.
///
/// For example, given `MkPair :: a -> a -> Pair a` and args `[Red, Blue]`:
/// 1. Peel Fun layers: `a -> a -> Pair a` → params `[a, a]`, result `Pair a`
/// 2. Infer arg types: `Red :: Color`, `Blue :: Color`
/// 3. Build substitution: `{a -> Color}`
/// 4. Apply to result: `Pair Color` = `App(Con("Pair"), Con("Color"))`
fn extract_constructor_result_type(
    con_ty: &Ty,
    con_args: &[&hir::Expr],
    ctx: &LowerContext,
) -> Option<Ty> {
    // Peel off Fun layers to get parameter types and result type
    let mut param_tys = Vec::new();
    let mut current = con_ty;
    while let Ty::Fun(arg, ret) = current {
        param_tys.push(arg.as_ref().clone());
        current = ret;
    }

    // If the constructor type is Error or we have no params, fall back
    if matches!(current, Ty::Error) || param_tys.is_empty() {
        return None;
    }

    let result_ty = current;

    // Build a substitution by matching param types against inferred arg types
    let mut subst = bhc_types::Subst::new();
    for (param_ty, arg_expr) in param_tys.iter().zip(con_args.iter()) {
        if let Ty::Var(tv) = param_ty {
            if subst.get(tv).is_none() {
                if let Some(arg_ty) = try_infer_arg_type(ctx, arg_expr) {
                    subst.insert(tv, arg_ty);
                }
            }
        }
    }

    // Apply substitution to the result type
    let concrete_result = subst.apply(result_ty);
    // Only return if we got something more specific than the original
    if concrete_result != *result_ty || !matches!(result_ty, Ty::App(_, _)) {
        Some(concrete_result)
    } else {
        Some(concrete_result)
    }
}

/// Try to infer an applied type from a constructor application expression.
///
/// For bare constructors like `Red`, returns None (use `try_infer_arg_type`).
/// For applied constructors like `Wrap Green`:
/// - Head: Wrap → type name "Wrapper"
/// - Arg: Green → type Color
/// - Returns: `App(Con("Wrapper"), Con("Color"))`
///
/// This enables resolution of parameterized instance types like
/// `instance Describable a => Describable (Wrapper a)`.
fn try_infer_applied_type(ctx: &LowerContext, expr: &hir::Expr) -> Option<Ty> {
    let Expr::App(_, _, _) = expr else {
        return None;
    };

    let mut head = expr;
    let mut con_args = Vec::new();
    while let Expr::App(f, x, _) = head {
        con_args.push(x.as_ref());
        head = f.as_ref();
    }
    con_args.reverse();

    let def_ref = match head {
        Expr::Con(dr) => dr,
        _ => return None,
    };

    let con_info = ctx.lookup_constructor(def_ref.def_id)?;
    let base = Ty::Con(TyCon::new(con_info.type_name, Kind::Star));

    let mut arg_types = Vec::new();
    for arg in &con_args {
        if let Some(ty) = try_infer_arg_type(ctx, arg) {
            arg_types.push(ty);
        }
    }

    if arg_types.is_empty() {
        return None;
    }

    // Deduplicate: for `Pair a a`, both value args map to the same type param
    let mut unique_types = Vec::new();
    for ty in &arg_types {
        if !unique_types.contains(ty) {
            unique_types.push(ty.clone());
        }
    }

    // Build applied type: App(App(base, ty1), ty2)
    let mut result = base;
    for ty in unique_types {
        result = Ty::App(Box::new(result), Box::new(ty));
    }

    Some(result)
}

/// Peel an application chain to find the head variable and collected arguments.
///
/// Given `App(App(Var(f), a1), a2)`, returns `Some((f_def_ref, [a1, a2]))`.
/// Arguments are returned in application order (inside-out).
fn peel_app_chain<'a>(expr: &'a hir::Expr) -> Option<(&'a DefRef, Vec<&'a hir::Expr>)> {
    let mut args = Vec::new();
    let mut current = expr;

    // Walk the App chain collecting arguments
    while let Expr::App(f, x, _) = current {
        args.push(x.as_ref());
        current = f.as_ref();
    }

    // The head must be a Var
    if let Expr::Var(def_ref) = current {
        // Reverse so args are in application order (innermost first)
        args.reverse();
        Some((def_ref, args))
    } else {
        None
    }
}

/// Lower a function application, handling dictionary-passing for class methods
/// and constrained functions when the argument type is known.
///
/// When `f` is a class method or constrained function and we can infer the
/// argument type, we resolve dictionaries at this concrete type. This handles
/// cases like `describe Red` where `describe` is a class method of `Describable`
/// and `Red` is a `Color` constructor.
fn lower_app(
    ctx: &mut LowerContext,
    f: &hir::Expr,
    x: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    // Check if f is a Var referencing a class method or constrained function
    if let Expr::Var(def_ref) = f {
        if let Some(var) = ctx.lookup_var(def_ref.def_id).cloned() {
            let method_name = var.name;

            // Case 1: User-defined class method with no dict in scope
            // Only apply dictionary resolution for user-defined classes.
            // Builtin classes (Eq, Ord, Num, Show, etc.) are handled by codegen.
            if let Some(class_name) = ctx.is_class_method(method_name) {
                if ctx.is_user_class(class_name) && ctx.lookup_dict(class_name).is_none() {
                    let param_count = ctx.class_param_count(class_name);

                    if param_count > 1 {
                        // Multi-param class with just one argument.
                        // Try to infer the arg type and complete remaining
                        // types from matching instances (fundep-style).
                        if let Some(arg_ty) = try_infer_arg_type(ctx, x) {
                            let mut types_for_resolution = vec![arg_ty];
                            // Search instances to complete the type list
                            if let Some(instances) =
                                ctx.class_registry().instances.get(&class_name)
                            {
                                for inst in &*instances {
                                    if inst.instance_types.len() >= param_count {
                                        let all_match =
                                            types_for_resolution.iter().enumerate().all(
                                                |(i, ty)| {
                                                    inst.instance_types
                                                        .get(i)
                                                        .map_or(false, |it| it == ty)
                                                },
                                            );
                                        if all_match {
                                            types_for_resolution =
                                                inst.instance_types[..param_count].to_vec();
                                            break;
                                        }
                                    }
                                }
                            }
                            if types_for_resolution.len() >= param_count {
                                if let Some(method_expr) =
                                    ctx.resolve_method_at_concrete_types(
                                        method_name,
                                        class_name,
                                        &types_for_resolution,
                                        span,
                                    )
                                {
                                    let x_core = lower_expr(ctx, x)?;
                                    return Ok(core::Expr::App(
                                        Box::new(method_expr),
                                        Box::new(x_core),
                                        span,
                                    ));
                                }
                            }
                        }
                        // Fall through to Case 3 if we couldn't resolve
                    } else {
                        // Single-param class: resolve at concrete type from argument
                        let inferred = try_infer_arg_type(ctx, x);
                        if let Some(concrete_ty) = inferred {
                            let resolved = ctx.resolve_method_at_concrete_type(
                                method_name,
                                class_name,
                                &concrete_ty,
                                span,
                            );
                            if let Some(method_expr) = resolved {
                                let x_core = lower_expr(ctx, x)?;
                                return Ok(core::Expr::App(
                                    Box::new(method_expr),
                                    Box::new(x_core),
                                    span,
                                ));
                            }
                            // Fallback: bare type didn't match instance head.
                            // Try applied type for parameterized instances.
                            if let Some(applied_ty) = try_infer_applied_type(ctx, x) {
                                if let Some(method_expr) =
                                    ctx.resolve_method_at_concrete_type(
                                        method_name,
                                        class_name,
                                        &applied_ty,
                                        span,
                                    )
                                {
                                    let x_core = lower_expr(ctx, x)?;
                                    return Ok(core::Expr::App(
                                        Box::new(method_expr),
                                        Box::new(x_core),
                                        span,
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            // Case 2: Constrained function with unresolved type-variable constraints
            if let Some(scheme) = ctx.lookup_scheme(def_ref.def_id) {
                let has_unresolved = scheme.constraints.iter().any(|c| {
                    // Only user-defined classes (not builtins like Show/Eq/Ord/Num)
                    ctx.is_user_class(c.class)
                        && c.args.iter().any(has_type_variables)
                });
                if has_unresolved {
                    if let Some(concrete_ty) = try_infer_arg_type(ctx, x) {
                        // Resolve dictionaries with the concrete type substituted in
                        let constraints = scheme.constraints.clone();
                        let mut result =
                            core::Expr::Var(var.clone(), def_ref.span);

                        for constraint in &constraints {
                            if ctx.is_user_class(constraint.class)
                                && constraint.args.iter().any(has_type_variables)
                            {
                                // Replace type variables with the concrete type
                                let concrete_constraint = Constraint::new(
                                    constraint.class,
                                    concrete_ty.clone(),
                                    constraint.span,
                                );
                                if let Some(dict_expr) =
                                    ctx.resolve_dictionary(&concrete_constraint, span)
                                {
                                    result = core::Expr::App(
                                        Box::new(result),
                                        Box::new(dict_expr),
                                        span,
                                    );
                                }
                            }
                        }

                        // Now apply the argument
                        let x_core = lower_expr(ctx, x)?;
                        return Ok(core::Expr::App(Box::new(result), Box::new(x_core), span));
                    }
                }
            }
        }
    }

    // Case 3: f is an App chain whose head is a class method or constrained function
    // e.g. myMap (+1) (Box 42) → f = App(Var(myMap), (+1)), x = App(Con(Box), Lit(42))
    // We peel the chain to find the head Var, then resolve dictionaries from x's type.
    if let Some((head_def_ref, collected_args)) = peel_app_chain(f) {
        if let Some(var) = ctx.lookup_var(head_def_ref.def_id).cloned() {
            let method_name = var.name;

            // Case 3a: Head is a user-defined class method
            if let Some(class_name) = ctx.is_class_method(method_name) {
                if ctx.is_user_class(class_name) && ctx.lookup_dict(class_name).is_none() {
                    let param_count = ctx.class_param_count(class_name);

                    if param_count > 1 {
                        // Multi-param class: collect types from all arguments
                        // For `combine Red Circle`, collected_args=[Red], x=Circle
                        // We need types from each arg: [Color, Shape]
                        let mut all_args: Vec<&hir::Expr> = collected_args.to_vec();
                        all_args.push(x);

                        let mut concrete_types: Vec<Ty> = Vec::new();
                        for arg in &all_args {
                            if let Some(ty) = try_infer_arg_type(ctx, arg) {
                                concrete_types.push(ty);
                            }
                        }

                        // If we have fewer types than params, try completing
                        // from instance declarations (fundep-style resolution).
                        // E.g., for `class Extract a b | a -> b` with `extract :: a -> b`,
                        // calling `extract w` gives us only type `a` from the value arg.
                        // We search instances to find the matching `b`.
                        // If we have fewer types than params, try completing
                        // from instance declarations (fundep-style resolution).
                        let mut types_for_resolution = concrete_types.clone();
                        if types_for_resolution.len() < param_count
                            && !types_for_resolution.is_empty()
                        {
                            if let Some(instances) =
                                ctx.class_registry().instances.get(&class_name)
                            {
                                for inst in instances {
                                    if inst.instance_types.len() >= param_count {
                                        let all_match =
                                            types_for_resolution.iter().enumerate().all(
                                                |(i, ty)| {
                                                    inst.instance_types
                                                        .get(i)
                                                        .map_or(false, |it| it == ty)
                                                },
                                            );
                                        if all_match {
                                            types_for_resolution =
                                                inst.instance_types[..param_count].to_vec();
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        if types_for_resolution.len() >= param_count {
                            if let Some(method_expr) = ctx.resolve_method_at_concrete_types(
                                method_name,
                                class_name,
                                &types_for_resolution,
                                span,
                            ) {
                                // Apply all arguments after dictionary resolution
                                let mut result = method_expr;
                                for arg in &collected_args {
                                    let arg_core = lower_expr(ctx, arg)?;
                                    result = core::Expr::App(
                                        Box::new(result),
                                        Box::new(arg_core),
                                        span,
                                    );
                                }
                                let x_core = lower_expr(ctx, x)?;
                                return Ok(core::Expr::App(
                                    Box::new(result),
                                    Box::new(x_core),
                                    span,
                                ));
                            }
                        }
                    } else {
                        // Single-param class: original logic
                        let inferred = try_infer_arg_type(ctx, x)
                            .or_else(|| {
                                collected_args
                                    .iter()
                                    .find_map(|arg| try_infer_arg_type(ctx, arg))
                            });
                        if let Some(concrete_ty) = inferred {
                            let resolved = ctx.resolve_method_at_concrete_type(
                                method_name,
                                class_name,
                                &concrete_ty,
                                span,
                            );
                            if let Some(method_expr) = resolved {
                                let mut result = method_expr;
                                for arg in &collected_args {
                                    let arg_core = lower_expr(ctx, arg)?;
                                    result = core::Expr::App(
                                        Box::new(result),
                                        Box::new(arg_core),
                                        span,
                                    );
                                }
                                let x_core = lower_expr(ctx, x)?;
                                return Ok(core::Expr::App(
                                    Box::new(result),
                                    Box::new(x_core),
                                    span,
                                ));
                            }
                            // Fallback: try applied type for parameterized instances
                            let applied = try_infer_applied_type(ctx, x)
                                .or_else(|| {
                                    collected_args
                                        .iter()
                                        .find_map(|arg| try_infer_applied_type(ctx, arg))
                                });
                            if let Some(applied_ty) = applied {
                                if let Some(method_expr) =
                                    ctx.resolve_method_at_concrete_type(
                                        method_name,
                                        class_name,
                                        &applied_ty,
                                        span,
                                    )
                                {
                                    let mut result = method_expr;
                                    for arg in &collected_args {
                                        let arg_core = lower_expr(ctx, arg)?;
                                        result = core::Expr::App(
                                            Box::new(result),
                                            Box::new(arg_core),
                                            span,
                                        );
                                    }
                                    let x_core = lower_expr(ctx, x)?;
                                    return Ok(core::Expr::App(
                                        Box::new(result),
                                        Box::new(x_core),
                                        span,
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            // Case 3b: Head is a constrained function with unresolved type-variable constraints
            if let Some(scheme) = ctx.lookup_scheme(head_def_ref.def_id) {
                let has_unresolved = scheme.constraints.iter().any(|c| {
                    ctx.is_user_class(c.class)
                        && c.args.iter().any(has_type_variables)
                });
                if has_unresolved {
                    // Try to infer concrete type from x or collected args
                    let inferred = try_infer_arg_type(ctx, x)
                        .or_else(|| {
                            collected_args.iter().find_map(|arg| try_infer_arg_type(ctx, arg))
                        });
                    if let Some(concrete_ty) = inferred {
                        let constraints = scheme.constraints.clone();
                        // Build from the head var (not lowered f, to avoid double resolution)
                        let mut result = core::Expr::Var(var.clone(), head_def_ref.span);

                        for constraint in &constraints {
                            if ctx.is_user_class(constraint.class)
                                && constraint.args.iter().any(has_type_variables)
                            {
                                let concrete_constraint = Constraint::new(
                                    constraint.class,
                                    concrete_ty.clone(),
                                    constraint.span,
                                );
                                if let Some(dict_expr) =
                                    ctx.resolve_dictionary(&concrete_constraint, span)
                                {
                                    result = core::Expr::App(
                                        Box::new(result),
                                        Box::new(dict_expr),
                                        span,
                                    );
                                }
                            }
                        }

                        // Apply collected args, then x
                        for arg in &collected_args {
                            let arg_core = lower_expr(ctx, arg)?;
                            result = core::Expr::App(
                                Box::new(result),
                                Box::new(arg_core),
                                span,
                            );
                        }
                        let x_core = lower_expr(ctx, x)?;
                        return Ok(core::Expr::App(Box::new(result), Box::new(x_core), span));
                    }
                }
            }
        }
    }

    // Default: lower f and x normally
    let f_core = lower_expr(ctx, f)?;
    let x_core = lower_expr(ctx, x)?;
    Ok(core::Expr::App(Box::new(f_core), Box::new(x_core), span))
}

/// Lower a type application expression.
///
/// Type applications like `f @Int` are used to instantiate polymorphic functions
/// at specific types. For class methods, this is the key mechanism for resolving
/// which instance to use at a monomorphic call site.
///
/// For example, `(+) @Int` should resolve to the `(+)` method from the `Num Int` instance.
fn lower_type_app(
    ctx: &mut LowerContext,
    expr: &hir::Expr,
    ty: &Ty,
    span: Span,
) -> LowerResult<core::Expr> {
    // Check if this is a type application to a class method
    if let Expr::Var(def_ref) = expr {
        if let Some(var) = ctx.lookup_var(def_ref.def_id) {
            let method_name = var.name;

            // Check if this is a user-defined class method
            // Only apply dictionary resolution for user-defined classes.
            // Builtin classes (Eq, Ord, Num, Show, etc.) are handled by codegen.
            if let Some(class_name) = ctx.is_class_method(method_name) {
                if ctx.is_user_class(class_name) {
                    // This is a user-defined class method being instantiated at a concrete type
                    // Construct the dictionary and select the method from it
                    if let Some(method_expr) =
                        ctx.resolve_method_at_concrete_type(method_name, class_name, ty, span)
                    {
                        return Ok(method_expr);
                    }
                }
                // Fall through to regular handling if resolution fails
            }
        }
    }

    // Regular type application handling
    let expr_core = lower_expr(ctx, expr)?;
    Ok(core::Expr::TyApp(Box::new(expr_core), ty.clone(), span))
}

/// Lower a constructor reference to Core.
fn lower_con(ctx: &mut LowerContext, def_ref: &DefRef) -> LowerResult<core::Expr> {
    // Constructors are represented as variables in Core
    // (they get special treatment during optimization)
    if let Some(var) = ctx.lookup_var(def_ref.def_id) {
        Ok(core::Expr::Var(var.clone(), def_ref.span))
    } else {
        let placeholder = Var {
            name: Symbol::intern("Con"),
            id: VarId::new(def_ref.def_id.index()),
            ty: Ty::Error,
        };
        Ok(core::Expr::Var(placeholder, def_ref.span))
    }
}

/// Lower a lambda expression to Core.
///
/// HIR lambdas can have multiple patterns: `\x y -> body`
/// Core lambdas take a single variable, so we need to:
/// 1. Create nested lambdas for each argument
/// 2. Compile patterns into case expressions
fn lower_lambda(
    ctx: &mut LowerContext,
    pats: &[hir::Pat],
    body: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    if pats.is_empty() {
        // No patterns - just lower the body
        return lower_expr(ctx, body);
    }

    // First pass: register all pattern variables so they're available in the body
    // We need to do this before lowering the body because the body may reference them
    let mut pat_vars: Vec<(hir::DefId, Var)> = Vec::new();
    for pat in pats {
        register_pattern_vars(ctx, pat, &mut pat_vars);
    }

    // Now lower the body (pattern vars are registered)
    let body_core = lower_expr(ctx, body)?;

    // Build nested lambdas from right to left
    let mut result = body_core;

    for pat in pats.iter().rev() {
        // Check if the pattern is simple (just a variable)
        match pat {
            hir::Pat::Var(name, def_id, _) => {
                // Simple case: pattern is just a variable
                // Look up the var we registered earlier
                let var = ctx.lookup_var(*def_id).cloned().unwrap_or_else(|| Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                });
                result = core::Expr::Lam(var, Box::new(result), span);
            }
            hir::Pat::Wild(_) => {
                // Wildcard: just use a fresh variable that's not referenced
                let arg_var = ctx.fresh_var("lam", Ty::Error, span);
                result = core::Expr::Lam(arg_var, Box::new(result), span);
            }
            _ => {
                // Complex pattern: need a case expression
                let arg_var = ctx.fresh_var("lam", Ty::Error, span);
                let alt = lower_pat_to_alt(ctx, pat, result.clone(), span)?;
                let default_alt = Alt {
                    con: AltCon::Default,
                    binders: vec![],
                    rhs: make_pattern_error(span),
                };

                let case_expr = core::Expr::Case(
                    Box::new(core::Expr::Var(arg_var.clone(), span)),
                    vec![alt, default_alt],
                    Ty::Error,
                    span,
                );

                result = core::Expr::Lam(arg_var, Box::new(case_expr), span);
            }
        }
    }

    Ok(result)
}

/// Register all variables bound by a pattern into the context.
fn register_pattern_vars(
    ctx: &mut LowerContext,
    pat: &hir::Pat,
    vars: &mut Vec<(hir::DefId, Var)>,
) {
    match pat {
        hir::Pat::Var(name, def_id, _) => {
            let var = Var {
                name: *name,
                id: VarId::new(def_id.index()),
                ty: Ty::Error,
            };
            ctx.register_var(*def_id, var.clone());
            vars.push((*def_id, var));
        }
        hir::Pat::As(name, def_id, inner, _) => {
            let var = Var {
                name: *name,
                id: VarId::new(def_id.index()),
                ty: Ty::Error,
            };
            ctx.register_var(*def_id, var.clone());
            vars.push((*def_id, var));
            register_pattern_vars(ctx, inner, vars);
        }
        hir::Pat::Con(_, sub_pats, _) => {
            for sub in sub_pats {
                register_pattern_vars(ctx, sub, vars);
            }
        }
        hir::Pat::RecordCon(_, field_pats, _) => {
            for fp in field_pats {
                register_pattern_vars(ctx, &fp.pat, vars);
            }
        }
        hir::Pat::Or(left, right, _) => {
            register_pattern_vars(ctx, left, vars);
            register_pattern_vars(ctx, right, vars);
        }
        hir::Pat::Ann(inner, _, _) | hir::Pat::View(_, inner, _) => {
            register_pattern_vars(ctx, inner, vars);
        }
        hir::Pat::Wild(_) | hir::Pat::Lit(_, _) | hir::Pat::Error(_) => {}
    }
}

/// Lower a let expression to Core.
fn lower_let(
    ctx: &mut LowerContext,
    bindings: &[hir::Binding],
    body: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    use crate::binding::{lower_bindings, preregister_bindings};

    // First, pre-register all binding variables so they're available
    // when lowering the body (and for recursive references in RHSes)
    let _vars = preregister_bindings(ctx, bindings)?;

    // Now lower the body - it can reference the bound variables
    let body_core = lower_expr(ctx, body)?;

    // Check if we have pattern bindings that need case expressions
    // For simple `let x = e in body`, we just create a let binding.
    // For pattern bindings like `let (x, y) = e in body`, we generate
    // `case e of (x, y) -> body` instead.
    lower_let_bindings(ctx, bindings, body_core, span)
}

/// Lower let bindings, handling pattern bindings with case expressions.
fn lower_let_bindings(
    ctx: &mut LowerContext,
    bindings: &[hir::Binding],
    body: core::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    use crate::binding::lower_bindings;

    // Process bindings from right to left, wrapping the body
    let mut result = body;

    for binding in bindings.iter().rev() {
        result = lower_single_let_binding(ctx, binding, result, span)?;
    }

    Ok(result)
}

/// Lower a single let binding.
/// For simple variable patterns, creates a let binding.
/// For complex patterns, creates a case expression.
fn lower_single_let_binding(
    ctx: &mut LowerContext,
    binding: &hir::Binding,
    body: core::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    use crate::binding::collect_free_vars;

    match &binding.pat {
        // Simple variable pattern: let x = e in body
        hir::Pat::Var(name, def_id, _) => {
            let rhs = lower_expr(ctx, &binding.rhs)?;
            let var = ctx.lookup_var(*def_id).cloned().unwrap_or_else(|| Var {
                name: *name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            });

            // Check if the binding is self-recursive
            let free_vars = collect_free_vars(&rhs);
            let is_recursive = free_vars.contains(name);

            let bind = if is_recursive {
                Bind::Rec(vec![(var, Box::new(rhs))])
            } else {
                Bind::NonRec(var, Box::new(rhs))
            };

            Ok(core::Expr::Let(Box::new(bind), Box::new(body), span))
        }

        // Complex pattern: let pat = e in body -> case e of pat -> body
        _ => {
            let scrutinee = lower_expr(ctx, &binding.rhs)?;
            let alt = lower_pat_to_alt(ctx, &binding.pat, body, span)?;
            Ok(core::Expr::Case(
                Box::new(scrutinee),
                vec![alt],
                Ty::Error,
                span,
            ))
        }
    }
}

/// Lower a case expression to Core.
fn lower_case(
    ctx: &mut LowerContext,
    scrutinee: &hir::Expr,
    alts: &[hir::CaseAlt],
    span: Span,
) -> LowerResult<core::Expr> {
    use crate::pattern::{bind_pattern_vars, lower_pat_to_alt_with_fallthrough};

    let scrutinee_core = lower_expr(ctx, scrutinee)?;

    // Check if any alternative has a nested/complex sub-pattern that needs
    // fallthrough support (e.g., `Lit 0` where the literal match may fail).
    let needs_fallthrough = alts.iter().any(|alt| has_complex_subpatterns(&alt.pat));

    if !needs_fallthrough {
        // Simple case: no nested patterns, use the fast path
        let mut core_alts = Vec::with_capacity(alts.len());
        for alt in alts {
            bind_pattern_vars(ctx, &alt.pat, None);
            let rhs = if alt.guards.is_empty() {
                lower_expr(ctx, &alt.rhs)?
            } else {
                lower_guarded_rhs(ctx, &alt.guards, &alt.rhs, span)?
            };
            let core_alt = lower_pat_to_alt(ctx, &alt.pat, rhs, span)?;
            core_alts.push(core_alt);
        }
        return Ok(core::Expr::Case(
            Box::new(scrutinee_core),
            core_alts,
            Ty::Error,
            span,
        ));
    }

    // Complex case: some alternatives have nested sub-patterns.
    // Bind the scrutinee to a variable so fallthrough can re-case on it.
    let scrut_var = ctx.fresh_var("scrut", Ty::Error, span);

    // First, lower all alternatives' RHS and patterns (we need them all
    // to build fallthrough expressions).
    let mut lowered_alts: Vec<(hir::Pat, core::Expr)> = Vec::with_capacity(alts.len());
    for alt in alts {
        bind_pattern_vars(ctx, &alt.pat, None);
        let rhs = if alt.guards.is_empty() {
            lower_expr(ctx, &alt.rhs)?
        } else {
            lower_guarded_rhs(ctx, &alt.guards, &alt.rhs, span)?
        };
        lowered_alts.push((alt.pat.clone(), rhs));
    }

    // Build the core alternatives with fallthrough.
    // For alternative i, the fallthrough is a case on scrut_var with alts [i+1..].
    // We build from the end backwards so each fallthrough can be computed.
    let mut core_alts_reversed: Vec<core::Alt> = Vec::with_capacity(lowered_alts.len());

    for i in (0..lowered_alts.len()).rev() {
        let (ref pat, ref rhs) = lowered_alts[i];

        // Build fallthrough: a case on scrut_var with the remaining (already-built) alternatives
        let fallthrough = if core_alts_reversed.is_empty() {
            None // Last alternative: no fallthrough
        } else {
            // The remaining alternatives (in correct order)
            let remaining: Vec<core::Alt> = core_alts_reversed.iter().rev().cloned().collect();
            Some(core::Expr::Case(
                Box::new(core::Expr::Var(scrut_var.clone(), span)),
                remaining,
                Ty::Error,
                span,
            ))
        };

        let core_alt = lower_pat_to_alt_with_fallthrough(ctx, pat, rhs.clone(), span, fallthrough)?;
        core_alts_reversed.push(core_alt);
    }

    // Reverse to get correct order
    core_alts_reversed.reverse();

    // Wrap in let-binding for the scrutinee variable
    let case_expr = core::Expr::Case(
        Box::new(core::Expr::Var(scrut_var.clone(), span)),
        core_alts_reversed,
        Ty::Error,
        span,
    );

    let bind = core::Bind::NonRec(scrut_var, Box::new(scrutinee_core));
    Ok(core::Expr::Let(
        Box::new(bind),
        Box::new(case_expr),
        span,
    ))
}

/// Check if a pattern has complex (non-variable, non-wildcard) sub-patterns
/// within a constructor pattern. These require fallthrough support.
fn has_complex_subpatterns(pat: &hir::Pat) -> bool {
    match pat {
        hir::Pat::Con(_, sub_pats, _) => {
            sub_pats.iter().any(|p| !matches!(p, hir::Pat::Var(..) | hir::Pat::Wild(_)))
        }
        hir::Pat::RecordCon(_, fields, _) => {
            fields.iter().any(|fp| !matches!(&fp.pat, hir::Pat::Var(..) | hir::Pat::Wild(_)))
        }
        _ => false,
    }
}

/// Lower guarded RHS to nested if expressions.
fn lower_guarded_rhs(
    ctx: &mut LowerContext,
    guards: &[hir::Guard],
    rhs: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    let rhs_core = lower_expr(ctx, rhs)?;

    // Build nested ifs from right to left
    let mut result = make_pattern_error(span); // Default if no guard matches

    for guard in guards.iter().rev() {
        let cond = lower_expr(ctx, &guard.cond)?;
        result = make_if_expr(cond, rhs_core.clone(), result, span);
    }

    Ok(result)
}

/// Lower an if expression to a case on Bool.
fn lower_if(
    ctx: &mut LowerContext,
    cond: &hir::Expr,
    then_br: &hir::Expr,
    else_br: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    let cond_core = lower_expr(ctx, cond)?;
    let then_core = lower_expr(ctx, then_br)?;
    let else_core = lower_expr(ctx, else_br)?;

    Ok(make_if_expr(cond_core, then_core, else_core, span))
}

/// Create a Core if expression (case on Bool).
fn make_if_expr(
    cond: core::Expr,
    then_br: core::Expr,
    else_br: core::Expr,
    span: Span,
) -> core::Expr {
    let bool_tycon = TyCon::new(Symbol::intern("Bool"), Kind::Star);
    let true_con = DataCon {
        name: Symbol::intern("True"),
        ty_con: bool_tycon.clone(),
        tag: 1,
        arity: 0,
    };
    let false_con = DataCon {
        name: Symbol::intern("False"),
        ty_con: bool_tycon,
        tag: 0,
        arity: 0,
    };

    let true_alt = Alt {
        con: AltCon::DataCon(true_con),
        binders: vec![],
        rhs: then_br,
    };

    let false_alt = Alt {
        con: AltCon::DataCon(false_con),
        binders: vec![],
        rhs: else_br,
    };

    core::Expr::Case(Box::new(cond), vec![true_alt, false_alt], Ty::Error, span)
}

/// Lower a tuple expression to Core.
fn lower_tuple(ctx: &mut LowerContext, elems: &[hir::Expr], span: Span) -> LowerResult<core::Expr> {
    if elems.is_empty() {
        // Unit: ()
        let unit_var = Var {
            name: Symbol::intern("()"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        return Ok(core::Expr::Var(unit_var, span));
    }

    // Build tuple constructor application
    let tuple_name = Symbol::intern(&format!("({})", ",".repeat(elems.len() - 1)));
    let tuple_var = Var {
        name: tuple_name,
        id: VarId::new(0),
        ty: Ty::Error,
    };

    let mut result = core::Expr::Var(tuple_var, span);

    for elem in elems {
        let elem_core = lower_expr(ctx, elem)?;
        result = core::Expr::App(Box::new(result), Box::new(elem_core), span);
    }

    Ok(result)
}

/// Lower a list expression to Core.
fn lower_list(ctx: &mut LowerContext, elems: &[hir::Expr], span: Span) -> LowerResult<core::Expr> {
    // Build list from right to left: [a,b,c] = a : (b : (c : []))
    let nil_var = Var {
        name: Symbol::intern("[]"),
        id: VarId::new(0),
        ty: Ty::Error,
    };
    let cons_var = Var {
        name: Symbol::intern(":"),
        id: VarId::new(0),
        ty: Ty::Error,
    };

    let mut result = core::Expr::Var(nil_var, span);

    for elem in elems.iter().rev() {
        let elem_core = lower_expr(ctx, elem)?;
        // Apply (:) to elem and result
        let cons_app = core::Expr::App(
            Box::new(core::Expr::Var(cons_var.clone(), span)),
            Box::new(elem_core),
            span,
        );
        result = core::Expr::App(Box::new(cons_app), Box::new(result), span);
    }

    Ok(result)
}

/// Lower a record construction to Core.
fn lower_record(
    ctx: &mut LowerContext,
    con_ref: &DefRef,
    fields: &[hir::FieldExpr],
    span: Span,
) -> LowerResult<core::Expr> {
    // Record construction becomes constructor application
    // The fields must be in the correct order for the constructor
    let con_core = lower_con(ctx, con_ref)?;

    let mut result = con_core;
    for field in fields {
        let value_core = lower_expr(ctx, &field.value)?;
        result = core::Expr::App(Box::new(result), Box::new(value_core), span);
    }

    Ok(result)
}

/// Lower field access to Core.
///
/// Field access `r.field` is compiled to a case expression that extracts
/// the appropriate field from the constructor.
fn lower_field_access(
    ctx: &mut LowerContext,
    expr: &hir::Expr,
    field: Symbol,
    span: Span,
) -> LowerResult<core::Expr> {
    let expr_core = lower_expr(ctx, expr)?;

    // Try to find the field selector information (clone to avoid borrow issues)
    let field_info = ctx.lookup_field_selector(field).cloned();

    if let Some(info) = field_info {
        // Generate a case expression to extract the field
        // case r of Con x0 x1 ... xn -> xi (where xi is the field we want)

        // Create binder variables for all fields
        let mut binders = Vec::with_capacity(info.total_fields);
        let mut result_var = None;

        for i in 0..info.total_fields {
            let var_name = format!("$field_{}", i);
            let var = ctx.fresh_var(&var_name, Ty::Error, span);
            if i == info.field_index {
                result_var = Some(var.clone());
            }
            binders.push(var);
        }

        let result_var = result_var.unwrap_or_else(|| {
            // Shouldn't happen, but handle it gracefully
            binders
                .first()
                .cloned()
                .unwrap_or_else(|| ctx.fresh_var("$error", Ty::Error, span))
        });

        // Look up constructor info for the data constructor
        let con_info = ctx.lookup_constructor(info.con_id).cloned();
        let (con_name, tycon, tag) = if let Some(ci) = con_info {
            (ci.name, TyCon::new(ci.type_name, Kind::Star), ci.tag)
        } else {
            (info.con_name, TyCon::new(info.type_name, Kind::Star), 0)
        };

        let data_con = core::DataCon {
            name: con_name,
            ty_con: tycon,
            tag,
            arity: info.total_fields as u32,
        };

        // Create the case alternative
        let alt = core::Alt {
            con: core::AltCon::DataCon(data_con),
            binders,
            rhs: core::Expr::Var(result_var, span),
        };

        // Add a default case for safety
        let default_alt = core::Alt {
            con: core::AltCon::Default,
            binders: vec![],
            rhs: make_pattern_error(span),
        };

        Ok(core::Expr::Case(
            Box::new(expr_core),
            vec![alt, default_alt],
            Ty::Error,
            span,
        ))
    } else {
        // Fallback: use selector function (works for imported types where we don't have full info)
        let selector_var = Var {
            name: field,
            id: VarId::new(0),
            ty: Ty::Error,
        };

        Ok(core::Expr::App(
            Box::new(core::Expr::Var(selector_var, span)),
            Box::new(expr_core),
            span,
        ))
    }
}

/// Lower record update to Core.
///
/// Record update `r { field1 = e1, field2 = e2 }` is compiled to a case expression
/// that extracts all fields, applies updates, and reconstructs the record.
fn lower_record_update(
    ctx: &mut LowerContext,
    expr: &hir::Expr,
    fields: &[hir::FieldExpr],
    span: Span,
) -> LowerResult<core::Expr> {
    if fields.is_empty() {
        // No fields to update - just return the original expression
        return lower_expr(ctx, expr);
    }

    let expr_core = lower_expr(ctx, expr)?;

    // Try to find the field selector information for the first field (clone to avoid borrow issues)
    let first_field = &fields[0];
    let field_info = ctx.lookup_field_selector(first_field.name).cloned();

    if let Some(info) = field_info {
        // Build a map of field name -> new value expression
        let mut updates: std::collections::HashMap<Symbol, core::Expr> =
            std::collections::HashMap::new();
        for field in fields {
            let value_core = lower_expr(ctx, &field.value)?;
            updates.insert(field.name, value_core);
        }

        // Create binder variables for all fields
        let mut binders = Vec::with_capacity(info.total_fields);
        for i in 0..info.total_fields {
            let var_name = format!("$old_{}", i);
            let var = ctx.fresh_var(&var_name, Ty::Error, span);
            binders.push(var);
        }

        // Look up constructor info (clone to avoid borrow issues)
        let con_info = ctx.lookup_constructor(info.con_id).cloned();
        let field_names: Vec<Symbol> = con_info
            .as_ref()
            .map(|ci| ci.field_names.clone())
            .unwrap_or_default();

        let (con_name, tycon, tag, arity) = if let Some(ref ci) = con_info {
            (
                ci.name,
                TyCon::new(ci.type_name, Kind::Star),
                ci.tag,
                ci.arity,
            )
        } else {
            (
                info.con_name,
                TyCon::new(info.type_name, Kind::Star),
                0,
                info.total_fields as u32,
            )
        };

        // Build the constructor application with updated fields
        let data_con = core::DataCon {
            name: con_name,
            ty_con: tycon.clone(),
            tag,
            arity,
        };

        let con_var = Var {
            name: con_name,
            id: VarId::new(0),
            ty: Ty::Error,
        };
        let mut result = core::Expr::Var(con_var, span);

        // Apply each field (using updated value if present, otherwise old value)
        for (i, binder) in binders.iter().enumerate() {
            let field_name = field_names.get(i).copied();
            let field_value = if let Some(fname) = field_name {
                if let Some(new_val) = updates.get(&fname) {
                    new_val.clone()
                } else {
                    core::Expr::Var(binder.clone(), span)
                }
            } else {
                core::Expr::Var(binder.clone(), span)
            };

            result = core::Expr::App(Box::new(result), Box::new(field_value), span);
        }

        // Create the case alternative
        let alt = core::Alt {
            con: core::AltCon::DataCon(data_con),
            binders,
            rhs: result,
        };

        // Add a default case for safety
        let default_alt = core::Alt {
            con: core::AltCon::Default,
            binders: vec![],
            rhs: make_pattern_error(span),
        };

        Ok(core::Expr::Case(
            Box::new(expr_core),
            vec![alt, default_alt],
            Ty::Error,
            span,
        ))
    } else {
        // No field info available - cannot compile record update
        ctx.error(LowerError::Internal(format!(
            "cannot compile record update: no information for field '{}'",
            first_field.name.as_str()
        )));
        Ok(make_pattern_error(span))
    }
}

/// Create a pattern match error expression.
fn make_pattern_error(span: Span) -> core::Expr {
    let error_var = Var {
        name: Symbol::intern("error"),
        id: VarId::new(0),
        ty: Ty::Error,
    };
    let msg = core::Expr::Lit(
        Literal::String(Symbol::intern("Non-exhaustive patterns")),
        Ty::Error,
        span,
    );
    core::Expr::App(
        Box::new(core::Expr::Var(error_var, span)),
        Box::new(msg),
        span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_hir::DefId;
    use bhc_index::Idx;

    fn make_def_ref(id: usize) -> DefRef {
        DefRef {
            def_id: DefId::new(id),
            span: Span::default(),
        }
    }

    #[test]
    fn test_lower_literal() {
        let lit = Lit::Int(42);
        let result = lower_lit(&lit, Span::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_lower_tuple() {
        let mut ctx = LowerContext::new();
        let elems = vec![
            hir::Expr::Lit(Lit::Int(1), Span::default()),
            hir::Expr::Lit(Lit::Int(2), Span::default()),
        ];
        let result = lower_tuple(&mut ctx, &elems, Span::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_lower_list() {
        let mut ctx = LowerContext::new();
        let elems = vec![
            hir::Expr::Lit(Lit::Int(1), Span::default()),
            hir::Expr::Lit(Lit::Int(2), Span::default()),
        ];
        let result = lower_list(&mut ctx, &elems, Span::default());
        assert!(result.is_ok());
    }
}
