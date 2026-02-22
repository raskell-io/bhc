//! Shared utility functions for Core IR simplification passes.

use std::sync::atomic::{AtomicU32, Ordering};

use bhc_index::Idx;
use rustc_hash::FxHashSet;

use crate::{Expr, VarId};

/// Counter for generating fresh variable IDs.
/// Starts at 2,000,000 to avoid collision with all existing VarId ranges:
/// builtins 0-95, LowerContext 100+, fixed DefIds 10000-11273,
/// deriving 50000+, RTS 1000000+.
static FRESH_COUNTER: AtomicU32 = AtomicU32::new(2_000_000);

/// Generate a fresh `VarId` that is guaranteed unique across all simplifier runs.
pub fn fresh_var_id() -> VarId {
    let id = FRESH_COUNTER.fetch_add(1, Ordering::Relaxed);
    VarId::new(id as usize)
}

/// Count the number of AST nodes in an expression (for inlining budget).
pub fn expr_size(expr: &Expr) -> usize {
    match expr {
        Expr::Var(_, _) | Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => 1,
        Expr::App(f, a, _) => 1 + expr_size(f) + expr_size(a),
        Expr::TyApp(f, _, _) => 1 + expr_size(f),
        Expr::Lam(_, body, _) | Expr::TyLam(_, body, _) => 1 + expr_size(body),
        Expr::Let(bind, body, _) => {
            let bind_size = match bind.as_ref() {
                crate::Bind::NonRec(_, rhs) => 1 + expr_size(rhs),
                crate::Bind::Rec(binds) => {
                    binds.iter().map(|(_, rhs)| 1 + expr_size(rhs)).sum::<usize>()
                }
            };
            bind_size + expr_size(body)
        }
        Expr::Case(scrut, alts, _, _) => {
            1 + expr_size(scrut)
                + alts
                    .iter()
                    .map(|alt| 1 + expr_size(&alt.rhs))
                    .sum::<usize>()
        }
        Expr::Lazy(e, _) | Expr::Cast(e, _, _) | Expr::Tick(_, e, _) => 1 + expr_size(e),
    }
}

/// Collect the set of free variable IDs in an expression.
pub fn free_var_ids(expr: &Expr) -> FxHashSet<VarId> {
    let mut free = FxHashSet::default();
    let mut bound = FxHashSet::default();
    collect_free_ids(expr, &mut free, &mut bound);
    free
}

fn collect_free_ids(expr: &Expr, free: &mut FxHashSet<VarId>, bound: &mut FxHashSet<VarId>) {
    match expr {
        Expr::Var(v, _) => {
            if !bound.contains(&v.id) {
                free.insert(v.id);
            }
        }
        Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => {}
        Expr::App(f, a, _) => {
            collect_free_ids(f, free, bound);
            collect_free_ids(a, free, bound);
        }
        Expr::TyApp(f, _, _) => collect_free_ids(f, free, bound),
        Expr::Lam(x, body, _) => {
            let was_new = bound.insert(x.id);
            collect_free_ids(body, free, bound);
            if was_new {
                bound.remove(&x.id);
            }
        }
        Expr::TyLam(_, body, _) => collect_free_ids(body, free, bound),
        Expr::Let(bind, body, _) => match bind.as_ref() {
            crate::Bind::NonRec(x, rhs) => {
                collect_free_ids(rhs, free, bound);
                let was_new = bound.insert(x.id);
                collect_free_ids(body, free, bound);
                if was_new {
                    bound.remove(&x.id);
                }
            }
            crate::Bind::Rec(binds) => {
                let mut added = Vec::new();
                for (x, _) in binds {
                    if bound.insert(x.id) {
                        added.push(x.id);
                    }
                }
                for (_, rhs) in binds {
                    collect_free_ids(rhs, free, bound);
                }
                collect_free_ids(body, free, bound);
                for id in added {
                    bound.remove(&id);
                }
            }
        },
        Expr::Case(scrut, alts, _, _) => {
            collect_free_ids(scrut, free, bound);
            for alt in alts {
                let mut added = Vec::new();
                for v in &alt.binders {
                    if bound.insert(v.id) {
                        added.push(v.id);
                    }
                }
                collect_free_ids(&alt.rhs, free, bound);
                for id in added {
                    bound.remove(&id);
                }
            }
        }
        Expr::Lazy(e, _) | Expr::Cast(e, _, _) | Expr::Tick(_, e, _) => {
            collect_free_ids(e, free, bound);
        }
    }
}

/// Returns true if the expression is "cheap" (safe to duplicate without work duplication).
/// Cheap expressions: variables, literals, types, coercions.
pub fn is_cheap(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Var(_, _) | Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _)
    )
}

/// Returns true if the expression contains a `Case` at the "top level" — i.e.,
/// not nested under a lambda or lazy. Expressions with top-level case create
/// LLVM basic block terminators when lowered, making them unsafe to inline into
/// arbitrary positions (e.g., function arguments).
pub fn contains_toplevel_case(expr: &Expr) -> bool {
    match expr {
        Expr::Case(_, _, _, _) => true,
        Expr::Var(_, _) | Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => false,
        Expr::App(f, a, _) => contains_toplevel_case(f) || contains_toplevel_case(a),
        Expr::TyApp(f, _, _) => contains_toplevel_case(f),
        // Lambda/TyLam bodies are deferred execution — Case inside doesn't
        // create immediate control flow at the call site.
        Expr::Lam(_, _, _) | Expr::TyLam(_, _, _) => false,
        Expr::Let(bind, body, _) => {
            let bind_has = match bind.as_ref() {
                crate::Bind::NonRec(_, rhs) => contains_toplevel_case(rhs),
                crate::Bind::Rec(pairs) => {
                    pairs.iter().any(|(_, rhs)| contains_toplevel_case(rhs))
                }
            };
            bind_has || contains_toplevel_case(body)
        }
        // Lazy is deferred execution, same as lambda
        Expr::Lazy(_, _) => false,
        Expr::Cast(e, _, _) | Expr::Tick(_, e, _) => contains_toplevel_case(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::Ty;
    use crate::{Bind, Var};

    fn mk_var(name: &str, id: u32) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id as usize), Ty::Error)
    }

    fn mk_var_expr(name: &str, id: u32) -> Expr {
        Expr::Var(mk_var(name, id), Span::default())
    }

    fn mk_int(n: i64) -> Expr {
        Expr::Lit(crate::Literal::Int(n), Ty::Error, Span::default())
    }

    #[test]
    fn test_fresh_var_ids_are_unique() {
        let a = fresh_var_id();
        let b = fresh_var_id();
        assert_ne!(a, b);
    }

    #[test]
    fn test_expr_size_literal() {
        assert_eq!(expr_size(&mk_int(42)), 1);
    }

    #[test]
    fn test_expr_size_app() {
        let app = Expr::App(
            Box::new(mk_var_expr("f", 1)),
            Box::new(mk_int(42)),
            Span::default(),
        );
        assert_eq!(expr_size(&app), 3);
    }

    #[test]
    fn test_free_var_ids_simple() {
        let e = mk_var_expr("x", 1);
        let fv = free_var_ids(&e);
        assert!(fv.contains(&VarId::new(1)));
        assert_eq!(fv.len(), 1);
    }

    #[test]
    fn test_free_var_ids_lambda_binds() {
        // \x -> x  (x is bound, not free)
        let lam = Expr::Lam(
            mk_var("x", 1),
            Box::new(mk_var_expr("x", 1)),
            Span::default(),
        );
        let fv = free_var_ids(&lam);
        assert!(fv.is_empty());
    }

    #[test]
    fn test_free_var_ids_let() {
        // let x = y in x  (y is free, x is bound)
        let e = Expr::Let(
            Box::new(Bind::NonRec(mk_var("x", 1), Box::new(mk_var_expr("y", 2)))),
            Box::new(mk_var_expr("x", 1)),
            Span::default(),
        );
        let fv = free_var_ids(&e);
        assert!(fv.contains(&VarId::new(2)));
        assert!(!fv.contains(&VarId::new(1)));
    }

    #[test]
    fn test_is_cheap() {
        assert!(is_cheap(&mk_int(42)));
        assert!(is_cheap(&mk_var_expr("x", 1)));
        let app = Expr::App(
            Box::new(mk_var_expr("f", 1)),
            Box::new(mk_int(42)),
            Span::default(),
        );
        assert!(!is_cheap(&app));
    }
}
