//! Type inference for expressions (Algorithm W).
//!
//! This module implements the core type inference algorithm for HIR
//! expressions. It follows Algorithm W from Hindley-Milner type inference
//! with extensions for Haskell-specific constructs.
//!
//! ## Algorithm Overview
//!
//! 1. **Literals**: Return the literal's type
//! 2. **Variables**: Instantiate the variable's type scheme
//! 3. **Application**: Infer function and argument, unify with fresh result
//! 4. **Lambda**: Infer body with parameter bindings, build function type
//! 5. **Let**: Infer binding, generalize, then infer body

use bhc_hir::{Binding, CaseAlt, Expr, Lit};
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Scheme, Ty};

use crate::context::TyCtxt;
use crate::diagnostics;

/// Infer the type of an expression.
///
/// This is the main entry point for expression type inference.
/// Returns the inferred type (which may contain type variables).
#[allow(clippy::too_many_lines)]
pub fn infer_expr(ctx: &mut TyCtxt, expr: &Expr) -> Ty {
    match expr {
        Expr::Lit(lit, span) => infer_lit(ctx, lit, *span),

        Expr::Var(def_ref) => {
            // Look up variable and instantiate its type scheme
            let scheme = ctx.env.lookup_def_id(def_ref.def_id).cloned();
            if let Some(s) = scheme {
                ctx.instantiate(&s)
            } else {
                diagnostics::emit_unbound_var(ctx, def_ref.def_id, def_ref.span);
                Ty::Error
            }
        }

        Expr::Con(def_ref) => {
            // Look up data constructor
            let scheme = ctx
                .env
                .lookup_data_con_by_id(def_ref.def_id)
                .map(|i| i.scheme.clone());
            if let Some(s) = scheme {
                ctx.instantiate(&s)
            } else {
                diagnostics::emit_unbound_constructor(ctx, def_ref.def_id, def_ref.span);
                Ty::Error
            }
        }

        Expr::App(func, arg, span) => {
            let func_ty = infer_expr(ctx, func);
            let arg_ty = infer_expr(ctx, arg);
            let result_ty = ctx.fresh_ty();

            // func : arg_ty -> result_ty
            let expected_func_ty = Ty::fun(arg_ty, result_ty.clone());
            ctx.unify(&func_ty, &expected_func_ty, *span);

            result_ty
        }

        Expr::Lam(pats, body, _span) => {
            // Enter scope for lambda bindings
            ctx.env.push_scope();

            // Infer types for each pattern (left to right)
            let mut arg_types = Vec::new();
            for pat in pats {
                let pat_ty = ctx.infer_pattern(pat);
                arg_types.push(pat_ty);
            }

            // Infer body type
            let body_ty = infer_expr(ctx, body);

            // Exit lambda scope
            ctx.env.pop_scope();

            // Build function type: arg1 -> arg2 -> ... -> body
            arg_types
                .into_iter()
                .rev()
                .fold(body_ty, |acc, arg_ty| Ty::fun(arg_ty, acc))
        }

        Expr::Let(bindings, body, _span) => {
            use bhc_hir::{DefId, Pat};

            // Enter scope for let bindings
            ctx.env.push_scope();

            // For recursive let bindings (Haskell's default), we need to:
            // 1. Pre-register all binding variables with fresh type variables
            // 2. Infer the RHS types (which can now reference other bindings)
            // 3. Unify and generalize

            // Helper to extract (name, def_id) pairs from a pattern
            fn extract_var_ids(pat: &Pat, out: &mut Vec<(Symbol, DefId)>) {
                match pat {
                    Pat::Var(name, def_id, _) => out.push((*name, *def_id)),
                    Pat::As(name, def_id, inner, _) => {
                        out.push((*name, *def_id));
                        extract_var_ids(inner, out);
                    }
                    Pat::Con(_, sub_pats, _) => {
                        for p in sub_pats {
                            extract_var_ids(p, out);
                        }
                    }
                    Pat::RecordCon(_, field_pats, _) => {
                        for fp in field_pats {
                            extract_var_ids(&fp.pat, out);
                        }
                    }
                    Pat::Ann(inner, _, _) | Pat::View(_, inner, _) => {
                        extract_var_ids(inner, out);
                    }
                    Pat::Or(left, _, _) => extract_var_ids(left, out),
                    Pat::Wild(_) | Pat::Lit(_, _) | Pat::Error(_) => {}
                }
            }

            // Step 1: Pre-register all bound variables with fresh types
            let mut binding_types: Vec<(Vec<(Symbol, DefId)>, Ty)> = Vec::new();
            for binding in bindings {
                let fresh_ty = ctx.fresh_ty();
                let mut var_ids = Vec::new();
                extract_var_ids(&binding.pat, &mut var_ids);

                // Register by both name and DefId (like pattern checking does)
                let scheme = Scheme::mono(fresh_ty.clone());
                for (name, def_id) in &var_ids {
                    ctx.env.insert_local(*name, scheme.clone());
                    ctx.env.insert_global(*def_id, scheme.clone());
                }
                binding_types.push((var_ids, fresh_ty));
            }

            // Step 2: Check each binding's RHS and unify with pre-registered type
            for (binding, (_var_ids, expected_ty)) in bindings.iter().zip(binding_types.iter()) {
                let rhs_ty = infer_expr(ctx, &binding.rhs);

                // If there's a signature, unify with it
                if let Some(sig) = &binding.sig {
                    ctx.unify(&rhs_ty, &sig.ty, binding.span);
                }

                // Unify with the pre-registered type
                ctx.unify(&rhs_ty, expected_ty, binding.span);

                // Check pattern
                ctx.check_pattern(&binding.pat, &rhs_ty);
            }

            // Step 3: Generalize binding types (update the environment)
            // For pattern bindings, check_pattern has already set the correct types.
            // We need to look up each variable's actual type and generalize it,
            // rather than using the whole binding type.
            for (binding, (var_ids, _ty)) in bindings.iter().zip(binding_types.iter()) {
                // For simple variable patterns, the var's type is the binding type
                // For complex patterns (Con, As, etc.), each var has its own type
                for (name, def_id) in var_ids {
                    // Look up the current type for this variable (set by check_pattern)
                    let var_ty = ctx
                        .env
                        .lookup_def_id(*def_id)
                        .map(|s| ctx.apply_subst(&s.ty))
                        .unwrap_or_else(|| ctx.fresh_ty());
                    let scheme = ctx.generalize(&var_ty);
                    ctx.env.insert_local(*name, scheme.clone());
                    ctx.env.insert_global(*def_id, scheme);
                }
                // For simple variable bindings, also add the generalized type annotation if present
                if let Some(sig) = &binding.sig {
                    if var_ids.len() == 1 {
                        let (_name, def_id) = &var_ids[0];
                        ctx.env.insert_global(*def_id, sig.clone());
                    }
                }
            }

            // Infer body type
            let body_ty = infer_expr(ctx, body);

            // Exit let scope
            ctx.env.pop_scope();

            body_ty
        }

        Expr::Case(scrutinee, alts, span) => {
            // Infer scrutinee type
            let scrut_ty = infer_expr(ctx, scrutinee);

            // Check if the scrutinee type involves a GADT
            let is_gadt_case = is_gadt_scrutinee(ctx, &scrut_ty);

            // All alternatives must produce the same type
            let result_ty = ctx.fresh_ty();

            for alt in alts {
                if is_gadt_case {
                    // Save substitution before each GADT alternative
                    let saved_subst = ctx.subst.clone();

                    let alt_ty = infer_case_alt(ctx, alt, &scrut_ty);
                    ctx.unify(&alt_ty, &result_ty, *span);

                    // Restore substitution after each alternative
                    // This keeps GADT type refinements local to each branch
                    ctx.subst = saved_subst;
                } else {
                    let alt_ty = infer_case_alt(ctx, alt, &scrut_ty);
                    ctx.unify(&alt_ty, &result_ty, *span);
                }
            }

            result_ty
        }

        Expr::If(cond, then_branch, else_branch, span) => {
            // Condition must be Bool
            let cond_ty = infer_expr(ctx, cond);
            ctx.unify(&cond_ty, &ctx.builtins.bool_ty.clone(), *span);

            // Both branches must have the same type
            let then_ty = infer_expr(ctx, then_branch);
            let else_ty = infer_expr(ctx, else_branch);
            ctx.unify(&then_ty, &else_ty, *span);

            then_ty
        }

        Expr::Tuple(elems, _span) => {
            let elem_types: Vec<Ty> = elems.iter().map(|e| infer_expr(ctx, e)).collect();
            Ty::Tuple(elem_types)
        }

        Expr::List(elems, span) => {
            let elem_ty = ctx.fresh_ty();

            // All elements must have the same type
            for elem in elems {
                let ty = infer_expr(ctx, elem);
                ctx.unify(&ty, &elem_ty, *span);
            }

            Ty::List(Box::new(elem_ty))
        }

        Expr::Record(con_ref, fields, span) => {
            // Look up constructor
            let scheme = ctx
                .env
                .lookup_data_con_by_id(con_ref.def_id)
                .map(|i| i.scheme.clone());
            if let Some(s) = scheme {
                let con_ty = ctx.instantiate(&s);

                // Extract the result type from the constructor type
                // Con type is: T1 -> T2 -> ... -> Tn -> Result
                let mut current = &con_ty;
                while let Ty::Fun(_, ret) = current {
                    current = ret.as_ref();
                }
                let result_ty = current.clone();

                // Try to use named field definitions for proper field matching
                if let Some(field_defs) = ctx.get_con_fields(con_ref.def_id) {
                    // Build a map from field name to expected type
                    // We need to instantiate the field types with the same substitution
                    // as the constructor type. Since we already instantiated the con_ty,
                    // the field types from field_defs need to be instantiated consistently.
                    // For now, we extract types from the instantiated constructor function type.
                    let mut expected_field_types = Vec::new();
                    let mut current = &con_ty;
                    while let Ty::Fun(arg, ret) = current {
                        expected_field_types.push(arg.as_ref().clone());
                        current = ret.as_ref();
                    }

                    // Build name -> type map using field definitions for names
                    // and instantiated types from expected_field_types
                    let field_type_map: std::collections::HashMap<Symbol, Ty> = field_defs
                        .iter()
                        .zip(expected_field_types.iter())
                        .map(|((name, _), ty): (&(Symbol, Ty), &Ty)| (*name, ty.clone()))
                        .collect();

                    // For each field in the record construction, look up by name
                    for field in fields {
                        let field_val_ty = infer_expr(ctx, &field.value);
                        if let Some(expected) = field_type_map.get(&field.name) {
                            ctx.unify(&field_val_ty, expected, field.span);
                        }
                        // If field name not found, that's an error but we continue
                    }
                } else {
                    // Fallback: positional matching for constructors without named fields
                    let mut expected_field_types = Vec::new();
                    let mut current = &con_ty;
                    while let Ty::Fun(arg, ret) = current {
                        expected_field_types.push(arg.as_ref().clone());
                        current = ret.as_ref();
                    }

                    for (i, field) in fields.iter().enumerate() {
                        let field_val_ty = infer_expr(ctx, &field.value);
                        if let Some(expected) = expected_field_types.get(i) {
                            ctx.unify(&field_val_ty, expected, field.span);
                        }
                    }
                }

                result_ty
            } else {
                diagnostics::emit_unbound_constructor(ctx, con_ref.def_id, *span);
                Ty::Error
            }
        }

        Expr::FieldAccess(record, _field_name, _span) => {
            // Infer record type
            let _record_ty = infer_expr(ctx, record);

            // Field access requires knowing the record type
            // For now, return a fresh variable (full implementation would resolve field)
            // TODO: Resolve field type from record type
            ctx.fresh_ty()
        }

        Expr::RecordUpdate(record, fields, _span) => {
            // Record update produces same type as input record
            let record_ty = infer_expr(ctx, record);

            // Check field types (simplified)
            for field in fields {
                let _field_ty = infer_expr(ctx, &field.value);
                // TODO: Verify field exists and types match
            }

            record_ty
        }

        Expr::Ann(inner, ty, span) => {
            // Type annotation: check inner against declared type.
            // If ScopedTypeVariables is enabled, resolve any scoped type variables
            // in the annotation type so they refer to the same types as the
            // enclosing function's forall-bound variables.
            let resolved_ty = if ctx.scoped_type_variables {
                ctx.resolve_scoped_type_vars(ty)
            } else {
                ty.clone()
            };
            let inner_ty = infer_expr(ctx, inner);
            ctx.unify(&inner_ty, &resolved_ty, *span);
            resolved_ty
        }

        Expr::TypeApp(inner, _ty_arg, _span) => {
            // Explicit type application: f @Int
            // If inner has a forall type, instantiate with the argument
            // For now, we just return the inner type
            // TODO: Handle explicit type application properly
            infer_expr(ctx, inner)
        }

        Expr::Error(_) => Ty::Error,
    }
}

/// Infer the type of a literal.
///
/// For numeric literals, this now generates type class constraints:
/// - Integer literals: `Num a => a` (but defaults to Int)
/// - Float literals: `Fractional a => a` (but defaults to Float)
///
/// The constraints are collected during inference and solved after,
/// potentially defaulting ambiguous type variables to standard types.
fn infer_lit(ctx: &mut TyCtxt, lit: &Lit, span: bhc_span::Span) -> Ty {
    match lit {
        Lit::Int(_) => {
            // Integer literals have type `Num a => a`
            // For now, we generate the constraint but default to Int
            // This allows overloaded numeric literals
            let ty = ctx.fresh_ty();
            let num_class = Symbol::intern("Num");
            ctx.emit_constraint(num_class, ty.clone(), span);
            ty
        }
        Lit::Float(_) => {
            // Float literals have type `Fractional a => a`
            let ty = ctx.fresh_ty();
            let fractional_class = Symbol::intern("Fractional");
            ctx.emit_constraint(fractional_class, ty.clone(), span);
            ty
        }
        Lit::Char(_) => ctx.builtins.char_ty.clone(),
        Lit::String(_) => {
            if ctx.overloaded_strings {
                // With OverloadedStrings, string literals have type `IsString a => a`
                let ty = ctx.fresh_ty();
                let is_string_class = Symbol::intern("IsString");
                ctx.emit_constraint(is_string_class, ty.clone(), span);
                ty
            } else {
                ctx.builtins.string_ty.clone()
            }
        }
    }
}

/// Check a binding (pattern = expression).
pub fn check_binding(ctx: &mut TyCtxt, binding: &Binding) {
    // Infer RHS type
    let rhs_ty = infer_expr(ctx, &binding.rhs);

    // If there's a signature, unify with it
    if let Some(sig) = &binding.sig {
        ctx.unify(&rhs_ty, &sig.ty, binding.span);
    }

    // Check pattern and bind variables
    ctx.check_pattern(&binding.pat, &rhs_ty);

    // Generalize and add to environment for each bound variable
    for var in binding.pat.bound_vars() {
        let scheme = ctx.generalize(&rhs_ty);
        ctx.env.insert_local(var, scheme);
    }
}

/// Infer the type of a case alternative.
fn infer_case_alt(ctx: &mut TyCtxt, alt: &CaseAlt, scrut_ty: &Ty) -> Ty {
    // Enter scope for pattern bindings
    ctx.env.push_scope();

    // Check pattern against scrutinee type
    ctx.check_pattern(&alt.pat, scrut_ty);

    // Check guards (must all be Bool)
    for guard in &alt.guards {
        let guard_ty = infer_expr(ctx, &guard.cond);
        ctx.unify(&guard_ty, &ctx.builtins.bool_ty.clone(), guard.span);
    }

    // Infer RHS type
    let rhs_ty = infer_expr(ctx, &alt.rhs);

    // Exit scope
    ctx.env.pop_scope();

    rhs_ty
}

/// Check if a scrutinee type involves a GADT.
///
/// Applies the current substitution to resolve type variables, then extracts
/// the type constructor head and checks if it's registered as a GADT.
fn is_gadt_scrutinee(ctx: &TyCtxt, scrut_ty: &Ty) -> bool {
    if ctx.gadt_types.is_empty() {
        return false;
    }
    let resolved = ctx.subst.apply(scrut_ty);
    let head = extract_type_head(&resolved);
    if let Ty::Con(tc) = head {
        ctx.gadt_types.contains(&tc.name)
    } else {
        false
    }
}

/// Extract the head type constructor from a type.
///
/// Given `T a b`, returns `T`. Given `a -> b`, returns `->`.
fn extract_type_head(ty: &Ty) -> Ty {
    match ty {
        Ty::App(f, _) => extract_type_head(f),
        _ => ty.clone(),
    }
}

/// Extract the result type from a constructor type.
///
/// Given `a -> b -> T a b`, returns `T a b`.
fn extract_result_type(ty: &Ty) -> Ty {
    match ty {
        Ty::Fun(_, result) => extract_result_type(result),
        _ => ty.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_hir::{DefId, Pat};
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::{FileId, Span};

    fn test_context() -> TyCtxt {
        let mut ctx = TyCtxt::new(FileId::new(0));
        ctx.register_builtins();
        ctx
    }

    #[test]
    fn test_infer_int_lit() {
        let mut ctx = test_context();
        let expr = Expr::Lit(Lit::Int(42), Span::DUMMY);

        let ty = infer_expr(&mut ctx, &expr);

        // Integer literals now have type `Num a => a` (fresh type variable with constraint)
        // After solving constraints, it should default to Int
        ctx.solve_constraints();
        let resolved_ty = ctx.apply_subst(&ty);
        assert_eq!(resolved_ty, ctx.builtins.int_ty);
    }

    #[test]
    fn test_infer_tuple() {
        let mut ctx = test_context();
        let expr = Expr::Tuple(
            vec![
                Expr::Lit(Lit::Int(1), Span::DUMMY),
                Expr::Lit(Lit::Char('a'), Span::DUMMY),
            ],
            Span::DUMMY,
        );

        let ty = infer_expr(&mut ctx, &expr);

        // Solve constraints first to resolve numeric literals
        ctx.solve_constraints();

        match ty {
            Ty::Tuple(elems) => {
                assert_eq!(elems.len(), 2);
                // Apply substitution to resolve defaulted types
                let elem0 = ctx.apply_subst(&elems[0]);
                let elem1 = ctx.apply_subst(&elems[1]);
                assert_eq!(elem0, ctx.builtins.int_ty);
                assert_eq!(elem1, ctx.builtins.char_ty);
            }
            _ => panic!("expected tuple type"),
        }
    }

    #[test]
    fn test_infer_list() {
        let mut ctx = test_context();
        let expr = Expr::List(
            vec![
                Expr::Lit(Lit::Int(1), Span::DUMMY),
                Expr::Lit(Lit::Int(2), Span::DUMMY),
            ],
            Span::DUMMY,
        );

        let ty = infer_expr(&mut ctx, &expr);

        // Solve constraints to resolve numeric literals
        ctx.solve_constraints();

        match ty {
            Ty::List(elem) => {
                let elem_ty = ctx.apply_subst(&elem);
                assert_eq!(elem_ty, ctx.builtins.int_ty);
            }
            _ => panic!("expected list type"),
        }
    }

    #[test]
    fn test_infer_lambda() {
        let mut ctx = test_context();
        let x = Symbol::intern("x");
        let expr = Expr::Lam(
            vec![Pat::Var(x, DefId::new(100), Span::DUMMY)],
            Box::new(Expr::Lit(Lit::Int(42), Span::DUMMY)),
            Span::DUMMY,
        );

        let ty = infer_expr(&mut ctx, &expr);

        // Solve constraints first
        ctx.solve_constraints();

        // Should be a -> Int for some a (after constraint solving, body is Int)
        match ty {
            Ty::Fun(from, to) => {
                assert!(matches!(*from, Ty::Var(_)));
                let body_ty = ctx.apply_subst(&to);
                assert_eq!(body_ty, ctx.builtins.int_ty);
            }
            _ => panic!("expected function type"),
        }
    }

    #[test]
    fn test_infer_if() {
        let mut ctx = test_context();
        let expr = Expr::If(
            Box::new(Expr::Lit(Lit::Int(1), Span::DUMMY)), // Wrong, but tests unification
            Box::new(Expr::Lit(Lit::Int(1), Span::DUMMY)),
            Box::new(Expr::Lit(Lit::Int(2), Span::DUMMY)),
            Span::DUMMY,
        );

        let _ty = infer_expr(&mut ctx, &expr);

        // Solve constraints - this will detect that the condition's type
        // (which was unified with Bool) doesn't have a Num instance
        ctx.solve_constraints();

        // The condition has a numeric type variable that gets unified with Bool
        // This causes a "no instance for Num Bool" error during constraint solving
        assert!(ctx.has_errors());

        // Note: With overloaded numeric literals, the exact resolved type
        // when there are errors is implementation-dependent. The key thing
        // is that we detect the error.
    }

    #[test]
    fn test_infer_char_lit() {
        let mut ctx = test_context();
        let expr = Expr::Lit(Lit::Char('a'), Span::DUMMY);

        let ty = infer_expr(&mut ctx, &expr);

        // Char literals have a fixed type (no constraint)
        assert_eq!(ty, ctx.builtins.char_ty);
    }

    #[test]
    fn test_infer_string_lit() {
        let mut ctx = test_context();
        let expr = Expr::Lit(Lit::String(Symbol::intern("hello")), Span::DUMMY);

        let ty = infer_expr(&mut ctx, &expr);

        // String literals have a fixed type (no constraint)
        assert_eq!(ty, ctx.builtins.string_ty);
    }
}
