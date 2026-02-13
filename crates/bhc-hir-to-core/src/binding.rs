//! Binding group analysis for HIR to Core lowering.
//!
//! This module handles the analysis and lowering of let bindings.
//! The main task is to determine which bindings are mutually recursive
//! and group them appropriately.
//!
//! ## Binding Types in Core
//!
//! - `NonRec(var, expr)`: A non-recursive binding (the variable cannot appear
//!   free in the expression)
//! - `Rec([(var, expr)])`: A group of mutually recursive bindings

use bhc_core::{self as core, Bind, Var};
use bhc_hir::{Binding, Pat};
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::Ty;
use rustc_hash::FxHashSet;

use crate::context::LowerContext;
use crate::expr::lower_expr;
use crate::{LowerError, LowerResult};

/// Pre-register binding variables in the context.
///
/// This registers all variables bound by the bindings so they can be
/// referenced in the body before the binding RHSes are lowered.
/// Returns the variables in order corresponding to the bindings.
pub fn preregister_bindings(ctx: &mut LowerContext, bindings: &[Binding]) -> LowerResult<Vec<Var>> {
    let mut vars = Vec::with_capacity(bindings.len());

    for binding in bindings {
        let var = preregister_pattern(ctx, &binding.pat)?;
        vars.push(var);
    }

    Ok(vars)
}

/// Pre-register a pattern's bound variable in the context.
/// Returns the top-level variable for the pattern (used for the binding).
fn preregister_pattern(ctx: &mut LowerContext, pat: &Pat) -> LowerResult<Var> {
    match pat {
        Pat::Var(name, def_id, _span) => {
            let var = Var {
                name: *name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            };
            // Register so it can be looked up later
            ctx.register_var(*def_id, var.clone());
            Ok(var)
        }
        Pat::Wild(span) => {
            let var = ctx.fresh_var("_wild", Ty::Error, *span);
            Ok(var)
        }
        Pat::Con(_, sub_pats, span) => {
            // Also pre-register all sub-pattern variables
            for sub_pat in sub_pats {
                preregister_pattern_nested(ctx, sub_pat)?;
            }
            let var = ctx.fresh_var("_pat", Ty::Error, *span);
            Ok(var)
        }
        Pat::RecordCon(_, field_pats, span) => {
            // Pre-register all field pattern variables
            for fp in field_pats {
                preregister_pattern_nested(ctx, &fp.pat)?;
            }
            let var = ctx.fresh_var("_rec", Ty::Error, *span);
            Ok(var)
        }
        Pat::As(name, def_id, inner, _span) => {
            // Also pre-register inner pattern variables
            preregister_pattern_nested(ctx, inner)?;
            let var = Var {
                name: *name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            };
            ctx.register_var(*def_id, var.clone());
            Ok(var)
        }
        Pat::Lit(_, span) => {
            let var = ctx.fresh_var("_lit", Ty::Error, *span);
            Ok(var)
        }
        Pat::Ann(inner, _, _) => preregister_pattern(ctx, inner),
        Pat::Or(left, _, _) => preregister_pattern(ctx, left),
        Pat::View(_, result_pat, span) => {
            // View pattern: pre-register variables from the result pattern
            preregister_pattern_nested(ctx, result_pat)?;
            let var = ctx.fresh_var("_view", Ty::Error, *span);
            Ok(var)
        }
        Pat::Error(span) => {
            let var = ctx.fresh_var("_err", Ty::Error, *span);
            Ok(var)
        }
    }
}

/// Pre-register all variables in a nested pattern.
/// This is used for sub-patterns inside constructor patterns.
fn preregister_pattern_nested(ctx: &mut LowerContext, pat: &Pat) -> LowerResult<()> {
    match pat {
        Pat::Var(name, def_id, _span) => {
            let var = Var {
                name: *name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            };
            ctx.register_var(*def_id, var);
            Ok(())
        }
        Pat::Wild(_) | Pat::Lit(_, _) | Pat::Error(_) => Ok(()),
        Pat::Con(_, sub_pats, _) => {
            for sub_pat in sub_pats {
                preregister_pattern_nested(ctx, sub_pat)?;
            }
            Ok(())
        }
        Pat::RecordCon(_, field_pats, _) => {
            for fp in field_pats {
                preregister_pattern_nested(ctx, &fp.pat)?;
            }
            Ok(())
        }
        Pat::As(name, def_id, inner, _span) => {
            preregister_pattern_nested(ctx, inner)?;
            let var = Var {
                name: *name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            };
            ctx.register_var(*def_id, var);
            Ok(())
        }
        Pat::Ann(inner, _, _) | Pat::Or(inner, _, _) | Pat::View(_, inner, _) => {
            preregister_pattern_nested(ctx, inner)
        }
    }
}

/// Lower a group of HIR bindings to Core.
///
/// This analyzes the bindings for mutual recursion and creates appropriate
/// `Rec` or `NonRec` binding groups.
pub fn lower_bindings(
    ctx: &mut LowerContext,
    bindings: &[Binding],
    span: Span,
) -> LowerResult<Bind> {
    if bindings.is_empty() {
        // Empty binding group - shouldn't happen, but handle gracefully
        return Err(LowerError::Internal("empty binding group".into()));
    }

    if bindings.len() == 1 {
        // Single binding - check if it's self-recursive
        return lower_single_binding(ctx, &bindings[0]);
    }

    // Multiple bindings - analyze for mutual recursion
    lower_binding_group(ctx, bindings, span)
}

/// Lower a single binding, checking for self-recursion.
fn lower_single_binding(ctx: &mut LowerContext, binding: &Binding) -> LowerResult<Bind> {
    let (var, names) = extract_binding_info(ctx, &binding.pat)?;

    // Lower the RHS
    let rhs = lower_expr(ctx, &binding.rhs)?;

    // Check if the binding refers to itself
    let free_vars = collect_free_vars(&rhs);
    let is_recursive = names.iter().any(|n| free_vars.contains(n));

    if is_recursive {
        Ok(Bind::Rec(vec![(var, Box::new(rhs))]))
    } else {
        Ok(Bind::NonRec(var, Box::new(rhs)))
    }
}

/// Lower a group of bindings, grouping mutually recursive ones.
fn lower_binding_group(
    ctx: &mut LowerContext,
    bindings: &[Binding],
    span: Span,
) -> LowerResult<Bind> {
    // Extract all binding names and their variables
    let mut all_vars = Vec::with_capacity(bindings.len());
    let mut all_names = FxHashSet::default();

    for binding in bindings {
        let (var, names) = extract_binding_info(ctx, &binding.pat)?;
        all_vars.push(var);
        all_names.extend(names);
    }

    // Lower all RHS expressions
    let mut rhs_exprs = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let rhs = lower_expr(ctx, &binding.rhs)?;
        rhs_exprs.push(rhs);
    }

    // Check if any binding refers to any other binding in the group
    let mut is_recursive = false;
    for rhs in &rhs_exprs {
        let free = collect_free_vars(rhs);
        if all_names.iter().any(|n| free.contains(n)) {
            is_recursive = true;
            break;
        }
    }

    if is_recursive {
        // All bindings form a recursive group
        let pairs: Vec<_> = all_vars
            .into_iter()
            .zip(rhs_exprs)
            .map(|(v, e)| (v, Box::new(e)))
            .collect();
        Ok(Bind::Rec(pairs))
    } else {
        // For simplicity, still treat as a single Rec group
        // A more sophisticated analysis would split into SCCs
        let pairs: Vec<_> = all_vars
            .into_iter()
            .zip(rhs_exprs)
            .map(|(v, e)| (v, Box::new(e)))
            .collect();
        if pairs.len() == 1 {
            let (var, rhs) = pairs.into_iter().next().unwrap();
            Ok(Bind::NonRec(var, rhs))
        } else {
            Ok(Bind::Rec(pairs))
        }
    }
}

/// Extract binding information from a pattern.
///
/// Returns the main variable and all names bound by the pattern.
/// If the variable was pre-registered (via `preregister_bindings`), reuses that.
fn extract_binding_info(ctx: &mut LowerContext, pat: &Pat) -> LowerResult<(Var, Vec<Symbol>)> {
    match pat {
        Pat::Var(name, def_id, _span) => {
            // Try to use pre-registered variable if available
            let var = if let Some(v) = ctx.lookup_var(*def_id) {
                v.clone()
            } else {
                Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                }
            };
            Ok((var, vec![*name]))
        }

        Pat::Wild(span) => {
            let var = ctx.fresh_var("_wild", Ty::Error, *span);
            Ok((var, vec![]))
        }

        Pat::Con(_, _sub_pats, span) => {
            // Pattern binding: let (x, y) = ...
            // We need to generate code to destructure the tuple/constructor
            // For now, create a single variable and let pattern compilation handle it
            let var = ctx.fresh_var("_pat", Ty::Error, *span);
            let names = pat.bound_vars();
            Ok((var, names))
        }

        Pat::As(name, def_id, inner, _span) => {
            // Try to use pre-registered variable if available
            let var = if let Some(v) = ctx.lookup_var(*def_id) {
                v.clone()
            } else {
                Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                }
            };
            let mut names = vec![*name];
            names.extend(inner.bound_vars());
            Ok((var, names))
        }

        Pat::Ann(inner, _, _) => extract_binding_info(ctx, inner),

        _ => {
            // Other patterns (Lit, Or, Error)
            let var = ctx.fresh_var("_pat", Ty::Error, pat.span());
            let names = pat.bound_vars();
            Ok((var, names))
        }
    }
}

/// Collect free variables from a Core expression.
///
/// This is a simple free variable analysis that collects all variable
/// names referenced in the expression.
pub fn collect_free_vars(expr: &core::Expr) -> FxHashSet<Symbol> {
    let mut free = FxHashSet::default();
    collect_free_vars_impl(expr, &mut FxHashSet::default(), &mut free);
    free
}

fn collect_free_vars_impl(
    expr: &core::Expr,
    bound: &mut FxHashSet<Symbol>,
    free: &mut FxHashSet<Symbol>,
) {
    match expr {
        core::Expr::Var(var, _) => {
            if !bound.contains(&var.name) {
                free.insert(var.name);
            }
        }

        core::Expr::Lit(_, _, _) => {}

        core::Expr::App(f, x, _) => {
            collect_free_vars_impl(f, bound, free);
            collect_free_vars_impl(x, bound, free);
        }

        core::Expr::TyApp(e, _, _) => {
            collect_free_vars_impl(e, bound, free);
        }

        core::Expr::Lam(var, body, _) => {
            let was_bound = bound.insert(var.name);
            collect_free_vars_impl(body, bound, free);
            if !was_bound {
                bound.remove(&var.name);
            }
        }

        core::Expr::TyLam(_, body, _) => {
            collect_free_vars_impl(body, bound, free);
        }

        core::Expr::Let(bind, body, _) => {
            // Collect bound names from the binding
            let bound_names: Vec<_> = match bind.as_ref() {
                Bind::NonRec(var, rhs) => {
                    collect_free_vars_impl(rhs, bound, free);
                    vec![var.name]
                }
                Bind::Rec(pairs) => {
                    let names: Vec<_> = pairs.iter().map(|(v, _)| v.name).collect();
                    // For Rec, all names are in scope for all RHS
                    for name in &names {
                        bound.insert(*name);
                    }
                    for (_, rhs) in pairs {
                        collect_free_vars_impl(rhs, bound, free);
                    }
                    names
                }
            };

            // Add bound names and process body
            for name in &bound_names {
                bound.insert(*name);
            }
            collect_free_vars_impl(body, bound, free);
            for name in bound_names {
                bound.remove(&name);
            }
        }

        core::Expr::Case(scrutinee, alts, _, _) => {
            collect_free_vars_impl(scrutinee, bound, free);
            for alt in alts {
                let alt_bound: Vec<_> = alt.binders.iter().map(|v| v.name).collect();
                for name in &alt_bound {
                    bound.insert(*name);
                }
                collect_free_vars_impl(&alt.rhs, bound, free);
                for name in alt_bound {
                    bound.remove(&name);
                }
            }
        }

        core::Expr::Lazy(e, _) => {
            collect_free_vars_impl(e, bound, free);
        }

        core::Expr::Cast(e, _, _) => {
            collect_free_vars_impl(e, bound, free);
        }

        core::Expr::Tick(_, e, _) => {
            collect_free_vars_impl(e, bound, free);
        }

        core::Expr::Type(_, _) | core::Expr::Coercion(_, _) => {}
    }
}

/// Analyze binding groups for mutual recursion.
///
/// This performs dependency analysis on a set of bindings and groups them
/// into strongly connected components (SCCs). Each SCC becomes a `Rec`
/// binding if it has more than one element or if it's self-recursive,
/// otherwise it becomes a `NonRec` binding.
pub fn analyze_bindings(bindings: &[Binding]) -> Vec<Vec<usize>> {
    // For simplicity, treat all bindings as one group
    // A proper implementation would use Tarjan's SCC algorithm
    if bindings.is_empty() {
        return vec![];
    }

    vec![(0..bindings.len()).collect()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_core::VarId;
    use bhc_hir as hir;
    use bhc_index::Idx;

    #[test]
    fn test_collect_free_vars_var() {
        let x = Symbol::intern("x");
        let expr = core::Expr::Var(
            Var {
                name: x,
                id: VarId::new(0),
                ty: Ty::Error,
            },
            Span::default(),
        );

        let free = collect_free_vars(&expr);
        assert!(free.contains(&x));
    }

    #[test]
    fn test_collect_free_vars_lambda() {
        let x = Symbol::intern("x");
        let y = Symbol::intern("y");

        // \x -> y (y is free, x is bound)
        let expr = core::Expr::Lam(
            Var {
                name: x,
                id: VarId::new(0),
                ty: Ty::Error,
            },
            Box::new(core::Expr::Var(
                Var {
                    name: y,
                    id: VarId::new(1),
                    ty: Ty::Error,
                },
                Span::default(),
            )),
            Span::default(),
        );

        let free = collect_free_vars(&expr);
        assert!(free.contains(&y));
        assert!(!free.contains(&x));
    }

    #[test]
    fn test_single_binding_non_recursive() {
        use bhc_hir::DefId;

        let mut ctx = LowerContext::new();
        let x = Symbol::intern("x");
        let def_id = DefId::new(0);

        let binding = Binding {
            pat: Pat::Var(x, def_id, Span::default()),
            sig: None,
            rhs: hir::Expr::Lit(hir::Lit::Int(42), Span::default()),
            span: Span::default(),
        };

        let result = lower_single_binding(&mut ctx, &binding);
        assert!(result.is_ok());

        let bind = result.unwrap();
        assert!(matches!(bind, Bind::NonRec(_, _)));
    }
}
