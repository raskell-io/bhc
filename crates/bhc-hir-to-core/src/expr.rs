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
use bhc_types::{Kind, Ty, TyCon};

use crate::context::LowerContext;
use crate::pattern::lower_pat_to_alt;
use crate::{LowerError, LowerResult};

/// Lower a HIR expression to Core.
pub fn lower_expr(ctx: &mut LowerContext, expr: &hir::Expr) -> LowerResult<core::Expr> {
    match expr {
        Expr::Lit(lit, span) => lower_lit(lit, *span),

        Expr::Var(def_ref) => lower_var(ctx, def_ref),

        Expr::Con(def_ref) => lower_con(ctx, def_ref),

        Expr::App(f, x, span) => {
            let f_core = lower_expr(ctx, f)?;
            let x_core = lower_expr(ctx, x)?;
            Ok(core::Expr::App(Box::new(f_core), Box::new(x_core), *span))
        }

        Expr::Lam(pats, body, span) => lower_lambda(ctx, pats, body, *span),

        Expr::Let(bindings, body, span) => lower_let(ctx, bindings, body, *span),

        Expr::Case(scrutinee, alts, span) => lower_case(ctx, scrutinee, alts, *span),

        Expr::If(cond, then_br, else_br, span) => lower_if(ctx, cond, then_br, else_br, *span),

        Expr::Tuple(elems, span) => lower_tuple(ctx, elems, *span),

        Expr::List(elems, span) => lower_list(ctx, elems, *span),

        Expr::Record(con_ref, fields, span) => lower_record(ctx, con_ref, fields, *span),

        Expr::FieldAccess(expr, field, span) => lower_field_access(ctx, expr, *field, *span),

        Expr::RecordUpdate(expr, fields, span) => lower_record_update(ctx, expr, fields, *span),

        Expr::Ann(expr, _ty, _span) => {
            // Type annotations are erased in Core (types are tracked separately)
            lower_expr(ctx, expr)
        }

        Expr::TypeApp(expr, ty, span) => {
            let expr_core = lower_expr(ctx, expr)?;
            Ok(core::Expr::TyApp(Box::new(expr_core), ty.clone(), *span))
        }

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
fn lower_var(ctx: &mut LowerContext, def_ref: &DefRef) -> LowerResult<core::Expr> {
    if let Some(var) = ctx.lookup_var(def_ref.def_id) {
        Ok(core::Expr::Var(var.clone(), def_ref.span))
    } else {
        // Variable not found - this could be a builtin or external reference
        // Create a placeholder variable
        eprintln!("DEBUG: lower_var could not find DefId({:?})", def_ref.def_id);
        let placeholder = Var {
            name: Symbol::intern("unknown"),
            id: VarId::new(def_ref.def_id.index()),
            ty: Ty::Error,
        };
        Ok(core::Expr::Var(placeholder, def_ref.span))
    }
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
fn register_pattern_vars(ctx: &mut LowerContext, pat: &hir::Pat, vars: &mut Vec<(hir::DefId, Var)>) {
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
        hir::Pat::Ann(inner, _, _) => {
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
            let var = ctx.lookup_var(*def_id).cloned().unwrap_or_else(|| {
                Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                }
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
    use crate::pattern::bind_pattern_vars;

    let scrutinee_core = lower_expr(ctx, scrutinee)?;

    let mut core_alts = Vec::with_capacity(alts.len());

    for alt in alts {
        // Pre-bind pattern variables so guards can reference them
        // The variables will be bound to the scrutinee when the pattern matches
        bind_pattern_vars(ctx, &alt.pat, None);

        // Handle guards by wrapping RHS in nested ifs
        let rhs = if alt.guards.is_empty() {
            lower_expr(ctx, &alt.rhs)?
        } else {
            lower_guarded_rhs(ctx, &alt.guards, &alt.rhs, span)?
        };

        let core_alt = lower_pat_to_alt(ctx, &alt.pat, rhs, span)?;
        core_alts.push(core_alt);
    }

    Ok(core::Expr::Case(
        Box::new(scrutinee_core),
        core_alts,
        Ty::Error,
        span,
    ))
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
fn make_if_expr(cond: core::Expr, then_br: core::Expr, else_br: core::Expr, span: Span) -> core::Expr {
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

    core::Expr::Case(
        Box::new(cond),
        vec![true_alt, false_alt],
        Ty::Error,
        span,
    )
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
fn lower_field_access(
    ctx: &mut LowerContext,
    expr: &hir::Expr,
    field: Symbol,
    span: Span,
) -> LowerResult<core::Expr> {
    // Field access becomes selector function application
    let expr_core = lower_expr(ctx, expr)?;

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

/// Lower record update to Core.
fn lower_record_update(
    ctx: &mut LowerContext,
    expr: &hir::Expr,
    fields: &[hir::FieldExpr],
    span: Span,
) -> LowerResult<core::Expr> {
    // Record update is complex - it needs to know the record type
    // For now, generate a placeholder that will be filled in during type checking
    let _ = lower_expr(ctx, expr)?;

    // TODO: Implement proper record update
    // This requires knowing the record type to generate the correct case expression
    ctx.error(LowerError::Internal("record update not yet implemented".into()));

    Ok(make_pattern_error(span))
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
