//! Pattern compilation for HIR to Core lowering.
//!
//! This module handles the compilation of HIR patterns into Core case
//! alternatives. The main challenge is compiling multi-equation function
//! definitions with overlapping patterns into efficient decision trees.
//!
//! ## Pattern Compilation Strategy
//!
//! We use a simplified version of the algorithm from:
//! "The Implementation of Functional Programming Languages" by SPJ
//!
//! For now, we use a straightforward approach that generates one alternative
//! per equation. This is correct but may generate redundant checks.

use bhc_core::{self as core, Alt, AltCon, DataCon, Literal, Var, VarId};
use bhc_hir::{self as hir, Pat, ValueDef};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Kind, Ty, TyCon};

use crate::context::LowerContext;
use crate::expr::lower_expr;
use crate::{LowerError, LowerResult};

/// Lower a HIR pattern to a Core case alternative.
///
/// This creates a single alternative that matches the given pattern and
/// executes the given RHS expression.
pub fn lower_pat_to_alt(
    ctx: &mut LowerContext,
    pat: &hir::Pat,
    rhs: core::Expr,
    span: Span,
) -> LowerResult<Alt> {
    match pat {
        Pat::Wild(_) => {
            // Wildcard matches anything
            Ok(Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs,
            })
        }

        Pat::Var(name, def_id, _) => {
            // Variable pattern: bind the value to the variable
            // Use pre-registered variable if available (for guards),
            // otherwise create a fresh one
            let var = ctx.lookup_var(*def_id).cloned().unwrap_or_else(|| {
                Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                }
            });
            // Default pattern with a binder
            Ok(Alt {
                con: AltCon::Default,
                binders: vec![var],
                rhs,
            })
        }

        Pat::Lit(lit, _) => {
            // Literal pattern
            let core_lit = match lit {
                hir::Lit::Int(n) => Literal::Int(*n as i64),
                hir::Lit::Float(f) => Literal::Float(*f as f32),
                hir::Lit::Char(c) => Literal::Char(*c),
                hir::Lit::String(s) => Literal::String(*s),
            };
            Ok(Alt {
                con: AltCon::Lit(core_lit),
                binders: vec![],
                rhs,
            })
        }

        Pat::Con(def_ref, sub_pats, _) => {
            // Constructor pattern
            // We need to create binders for all sub-patterns
            let mut binders = Vec::with_capacity(sub_pats.len());
            let mut inner_rhs = rhs;

            // Process sub-patterns from right to left
            // For complex sub-patterns, we need to wrap the RHS in nested cases
            for (i, sub_pat) in sub_pats.iter().enumerate().rev() {
                let binder_name = format!("pat_{}", i);
                let binder = ctx.fresh_var(&binder_name, Ty::Error, span);

                match sub_pat {
                    Pat::Var(name, def_id, _) => {
                        // Look up the registered variable (from bind_pattern_vars)
                        // Fall back to creating a fresh one if not found
                        let var = if let Some(registered) = ctx.lookup_var(*def_id) {
                            registered.clone()
                        } else {
                            Var {
                                name: *name,
                                id: ctx.fresh_id(),
                                ty: Ty::Error,
                            }
                        };
                        binders.push(var);
                    }
                    Pat::Wild(_) => {
                        // Wildcard: use a fresh variable
                        binders.push(binder);
                    }
                    _ => {
                        // Complex sub-pattern: need nested case
                        let sub_alt = lower_pat_to_alt(ctx, sub_pat, inner_rhs.clone(), span)?;
                        let default_alt = Alt {
                            con: AltCon::Default,
                            binders: vec![],
                            rhs: make_pattern_error(span),
                        };

                        inner_rhs = core::Expr::Case(
                            Box::new(core::Expr::Var(binder.clone(), span)),
                            vec![sub_alt, default_alt],
                            Ty::Error,
                            span,
                        );
                        binders.push(binder);
                    }
                }
            }

            // Reverse binders to match left-to-right order
            binders.reverse();

            // Create the data constructor
            // Look up the constructor name from the context
            let con_name = if let Some(var) = ctx.lookup_var(def_ref.def_id) {
                var.name
            } else {
                // Fallback - shouldn't happen if constructor was registered
                Symbol::intern("Con")
            };
            let placeholder_tycon = TyCon::new(Symbol::intern("DataType"), Kind::Star);
            let con = DataCon {
                name: con_name,
                ty_con: placeholder_tycon,
                tag: def_ref.def_id.index() as u32,
                arity: sub_pats.len() as u32,
            };

            Ok(Alt {
                con: AltCon::DataCon(con),
                binders,
                rhs: inner_rhs,
            })
        }

        Pat::RecordCon(def_ref, field_pats, _) => {
            // Record constructor pattern
            // For now, treat field patterns similarly to sub-patterns
            // TODO: Properly handle out-of-order fields by looking up field indices
            let mut binders = Vec::with_capacity(field_pats.len());
            let mut inner_rhs = rhs;

            // Process field patterns from right to left
            for (i, fp) in field_pats.iter().enumerate().rev() {
                let binder_name = format!("field_{}", i);
                let binder = ctx.fresh_var(&binder_name, Ty::Error, span);

                match &fp.pat {
                    Pat::Var(name, def_id, _) => {
                        let var = if let Some(registered) = ctx.lookup_var(*def_id) {
                            registered.clone()
                        } else {
                            Var {
                                name: *name,
                                id: ctx.fresh_id(),
                                ty: Ty::Error,
                            }
                        };
                        binders.push(var);
                    }
                    Pat::Wild(_) => {
                        binders.push(binder);
                    }
                    _ => {
                        let sub_alt = lower_pat_to_alt(ctx, &fp.pat, inner_rhs.clone(), span)?;
                        let default_alt = Alt {
                            con: AltCon::Default,
                            binders: vec![],
                            rhs: make_pattern_error(span),
                        };
                        inner_rhs = core::Expr::Case(
                            Box::new(core::Expr::Var(binder.clone(), span)),
                            vec![sub_alt, default_alt],
                            Ty::Error,
                            span,
                        );
                        binders.push(binder);
                    }
                }
            }

            binders.reverse();

            // Create the data constructor
            let con_name = if let Some(var) = ctx.lookup_var(def_ref.def_id) {
                var.name
            } else {
                Symbol::intern("Con")
            };
            let placeholder_tycon = TyCon::new(Symbol::intern("DataType"), Kind::Star);
            let con = DataCon {
                name: con_name,
                ty_con: placeholder_tycon,
                tag: def_ref.def_id.index() as u32,
                arity: field_pats.len() as u32,
            };

            Ok(Alt {
                con: AltCon::DataCon(con),
                binders,
                rhs: inner_rhs,
            })
        }

        Pat::As(name, _def_id, inner_pat, inner_span) => {
            // As-pattern: bind the value and also match the inner pattern
            let var = Var {
                name: *name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            };

            // Create a nested case for the inner pattern
            let inner_alt = lower_pat_to_alt(ctx, inner_pat, rhs, span)?;
            let default_alt = Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: make_pattern_error(span),
            };

            // The outer pattern binds the variable, then checks the inner
            let inner_case = core::Expr::Case(
                Box::new(core::Expr::Var(var.clone(), span)),
                vec![inner_alt, default_alt],
                Ty::Error,
                *inner_span,
            );

            Ok(Alt {
                con: AltCon::Default,
                binders: vec![var],
                rhs: inner_case,
            })
        }

        Pat::Or(left, right, _) => {
            // Or-pattern: try left, then right
            // This is complex because both branches must bind the same variables
            // For now, we generate a nested case structure
            // TODO: Proper or-pattern compilation
            ctx.error(LowerError::PatternError {
                message: "or-patterns not yet supported".into(),
                span,
            });
            Ok(Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: make_pattern_error(span),
            })
        }

        Pat::Ann(inner, _ty, _) => {
            // Type-annotated pattern: just use the inner pattern
            lower_pat_to_alt(ctx, inner, rhs, span)
        }

        Pat::Error(_) => {
            // Error pattern: generate error
            Ok(Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: make_pattern_error(span),
            })
        }
    }
}

/// Compile pattern matching for multiple equations.
///
/// This takes a function definition with multiple equations and compiles
/// them into a list of Core case alternatives.
pub fn compile_match(
    ctx: &mut LowerContext,
    value_def: &ValueDef,
    args: &[Var],
) -> LowerResult<Vec<Alt>> {
    let mut alts = Vec::with_capacity(value_def.equations.len());

    for eq in &value_def.equations {
        // IMPORTANT: Register pattern variables BEFORE lowering the RHS
        // This allows the RHS to reference variables bound by patterns
        for (i, pat) in eq.pats.iter().enumerate() {
            let arg_var = args.get(i).cloned();
            bind_pattern_vars(ctx, pat, arg_var.as_ref());
        }

        // Lower the RHS (pattern vars are now registered)
        let rhs = if eq.guards.is_empty() {
            lower_expr(ctx, &eq.rhs)?
        } else {
            // Handle guards
            compile_guarded_rhs(ctx, &eq.guards, &eq.rhs, eq.span)?
        };

        // Handle pattern matching
        if eq.pats.is_empty() {
            // No patterns - this is a simple value definition
            alts.push(Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs,
            });
        } else if eq.pats.len() == 1 {
            // Single pattern
            let alt = lower_pat_to_alt(ctx, &eq.pats[0], rhs, eq.span)?;
            alts.push(alt);
        } else {
            // Multiple patterns - need tuple pattern
            let tuple_alt = compile_tuple_pattern(ctx, &eq.pats, rhs, eq.span)?;
            alts.push(tuple_alt);
        }
    }

    // Add a default case for non-exhaustive patterns
    alts.push(Alt {
        con: AltCon::Default,
        binders: vec![],
        rhs: make_pattern_error(value_def.span),
    });

    Ok(alts)
}

/// Bind pattern variables in the context so they can be referenced in the RHS.
///
/// For a variable pattern like `x`, this registers the HIR DefId to map to
/// the corresponding Core variable (either the arg variable or a fresh one).
pub fn bind_pattern_vars(ctx: &mut LowerContext, pat: &hir::Pat, arg_var: Option<&Var>) {
    match pat {
        Pat::Var(_name, def_id, _span) => {
            // For a simple variable pattern, bind it to the arg variable
            // The arg variable comes from the scrutinee of the case expression
            if let Some(arg) = arg_var {
                ctx.register_var(*def_id, arg.clone());
            } else {
                // No arg var provided - create a fresh one
                let var = Var {
                    name: *_name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                };
                ctx.register_var(*def_id, var);
            }
        }
        Pat::Wild(_) => {
            // Wildcards don't bind any variables
        }
        Pat::Lit(_, _) => {
            // Literals don't bind any variables
        }
        Pat::Con(_, sub_pats, _) => {
            // Recursively bind sub-pattern variables
            // Each sub-pattern will need its own fresh variable
            for sub_pat in sub_pats {
                bind_pattern_vars(ctx, sub_pat, None);
            }
        }
        Pat::RecordCon(_, field_pats, _) => {
            // Recursively bind field pattern variables
            for fp in field_pats {
                bind_pattern_vars(ctx, &fp.pat, None);
            }
        }
        Pat::As(name, def_id, inner_pat, _) => {
            // As-pattern binds the name and also any variables in the inner pattern
            if let Some(arg) = arg_var {
                ctx.register_var(*def_id, arg.clone());
            } else {
                let var = Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                };
                ctx.register_var(*def_id, var);
            }
            bind_pattern_vars(ctx, inner_pat, None);
        }
        Pat::Or(left, right, _) => {
            // Or-patterns: both branches should bind the same variables
            bind_pattern_vars(ctx, left, arg_var);
            // Note: right should bind the same vars, but we skip for now
        }
        Pat::Ann(inner, _, _) => {
            bind_pattern_vars(ctx, inner, arg_var);
        }
        Pat::Error(_) => {
            // Error patterns don't bind variables
        }
    }
}

/// Compile a guarded RHS into nested conditionals.
fn compile_guarded_rhs(
    ctx: &mut LowerContext,
    guards: &[hir::Guard],
    rhs: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    let rhs_core = lower_expr(ctx, rhs)?;

    // Build nested ifs from right to left
    let mut result = make_pattern_error(span);

    for guard in guards.iter().rev() {
        let cond = lower_expr(ctx, &guard.cond)?;
        result = make_if(cond, rhs_core.clone(), result, span);
    }

    Ok(result)
}

/// Compile a tuple of patterns into a single alternative.
fn compile_tuple_pattern(
    ctx: &mut LowerContext,
    pats: &[hir::Pat],
    rhs: core::Expr,
    span: Span,
) -> LowerResult<Alt> {
    // Create a tuple pattern that matches all the individual patterns
    let tuple_name = Symbol::intern(&format!("({})", ",".repeat(pats.len() - 1)));

    // Create binders and potentially nested cases for each sub-pattern
    let mut binders = Vec::with_capacity(pats.len());
    let mut inner_rhs = rhs;

    for (i, pat) in pats.iter().enumerate().rev() {
        let binder_name = format!("arg{}", i);
        let binder = ctx.fresh_var(&binder_name, Ty::Error, span);

        match pat {
            Pat::Var(name, _, _) => {
                let var = Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                };
                binders.push(var);
            }
            Pat::Wild(_) => {
                binders.push(binder);
            }
            _ => {
                // Complex sub-pattern: need nested case
                let sub_alt = lower_pat_to_alt(ctx, pat, inner_rhs.clone(), span)?;
                let default_alt = Alt {
                    con: AltCon::Default,
                    binders: vec![],
                    rhs: make_pattern_error(span),
                };

                inner_rhs = core::Expr::Case(
                    Box::new(core::Expr::Var(binder.clone(), span)),
                    vec![sub_alt, default_alt],
                    Ty::Error,
                    span,
                );
                binders.push(binder);
            }
        }
    }

    binders.reverse();

    let tuple_tycon = TyCon::new(tuple_name, Kind::Star);
    let tuple_con = DataCon {
        name: tuple_name,
        ty_con: tuple_tycon,
        tag: 0,
        arity: pats.len() as u32,
    };

    Ok(Alt {
        con: AltCon::DataCon(tuple_con),
        binders,
        rhs: inner_rhs,
    })
}

/// Create an if expression (case on Bool).
fn make_if(cond: core::Expr, then_br: core::Expr, else_br: core::Expr, span: Span) -> core::Expr {
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

    core::Expr::Case(
        Box::new(cond),
        vec![
            Alt {
                con: AltCon::DataCon(true_con),
                binders: vec![],
                rhs: then_br,
            },
            Alt {
                con: AltCon::DataCon(false_con),
                binders: vec![],
                rhs: else_br,
            },
        ],
        Ty::Error,
        span,
    )
}

/// Create a pattern match error expression.
pub fn make_pattern_error(span: Span) -> core::Expr {
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
    use bhc_hir::{DefId, DefRef};
    use bhc_index::Idx;

    #[test]
    fn test_lower_wildcard_pattern() {
        let mut ctx = LowerContext::new();
        let pat = Pat::Wild(Span::default());
        let rhs = core::Expr::Lit(Literal::Int(42), Ty::Error, Span::default());

        let result = lower_pat_to_alt(&mut ctx, &pat, rhs, Span::default());
        assert!(result.is_ok());

        let alt = result.unwrap();
        assert!(matches!(alt.con, AltCon::Default));
    }

    #[test]
    fn test_lower_literal_pattern() {
        let mut ctx = LowerContext::new();
        let pat = Pat::Lit(hir::Lit::Int(42), Span::default());
        let rhs = core::Expr::Lit(Literal::Int(1), Ty::Error, Span::default());

        let result = lower_pat_to_alt(&mut ctx, &pat, rhs, Span::default());
        assert!(result.is_ok());

        let alt = result.unwrap();
        assert!(matches!(alt.con, AltCon::Lit(Literal::Int(42))));
    }

    #[test]
    fn test_lower_var_pattern() {
        let mut ctx = LowerContext::new();
        let x = Symbol::intern("x");
        let def_id = DefId::new(0);
        let pat = Pat::Var(x, def_id, Span::default());
        let rhs = core::Expr::Lit(Literal::Int(1), Ty::Error, Span::default());

        let result = lower_pat_to_alt(&mut ctx, &pat, rhs, Span::default());
        assert!(result.is_ok());

        let alt = result.unwrap();
        assert!(matches!(alt.con, AltCon::Default));
        assert_eq!(alt.binders.len(), 1);
        assert_eq!(alt.binders[0].name, x);
    }
}
