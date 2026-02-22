//! Beta reduction for Core IR.
//!
//! Transforms `(\x -> body) arg` into `body[arg/x]` when safe.

use crate::Expr;

use super::expr_util::{contains_toplevel_case, is_cheap, expr_size};
use super::occurrence::analyze_occurrences;
use super::subst::substitute_single;

/// Attempt beta reduction on an application `App(fun, arg)`.
///
/// Returns `Some(reduced)` if the reduction fires, `None` otherwise.
///
/// Reduces `(\x -> body) arg` to `body[arg/x]` when:
/// - `arg` is cheap (variable or literal), OR
/// - `x` is used at most once in `body`
pub fn try_beta_reduce(fun: &Expr, arg: &Expr, inline_threshold: usize) -> Option<Expr> {
    if let Expr::Lam(x, body, _) = fun {
        // Always safe to substitute cheap args (no work duplication)
        if is_cheap(arg) {
            return Some(substitute_single(*body.clone(), x.id, arg));
        }

        // For non-cheap args, only substitute if used at most once
        let occs = analyze_occurrences(body);
        let count = occs.get(&x.id).copied();
        match count {
            None => {
                // x is dead in body, drop arg (if pure) and return body
                // We still reduce — the arg is dropped but we eliminate the lambda
                Some(*body.clone())
            }
            Some(super::occurrence::OccCount::Once) => {
                // Used exactly once, not under lambda — safe to inline
                // unless the arg contains a Case (creates LLVM terminators)
                if !contains_toplevel_case(arg) {
                    Some(substitute_single(*body.clone(), x.id, arg))
                } else {
                    None
                }
            }
            Some(super::occurrence::OccCount::OnceInLam) => {
                // Used once but under a lambda — only inline if small
                // and doesn't contain control flow
                if !contains_toplevel_case(arg) && expr_size(arg) <= inline_threshold {
                    Some(substitute_single(*body.clone(), x.id, arg))
                } else {
                    None
                }
            }
            Some(super::occurrence::OccCount::Many) => {
                // Used multiple times — only inline cheap args (handled above)
                None
            }
            Some(super::occurrence::OccCount::Dead) => {
                Some(*body.clone())
            }
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::Ty;
    use crate::{Literal, Var, VarId};

    fn mk_var(name: &str, id: u32) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id as usize), Ty::Error)
    }

    fn mk_var_expr(name: &str, id: u32) -> Expr {
        Expr::Var(mk_var(name, id), Span::default())
    }

    fn mk_int(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), Ty::Error, Span::default())
    }

    #[test]
    fn test_beta_reduce_cheap_arg() {
        // (\x -> x) 42 => 42
        let lam = Expr::Lam(mk_var("x", 1), Box::new(mk_var_expr("x", 1)), Span::default());
        let arg = mk_int(42);
        let result = try_beta_reduce(&lam, &arg, 20);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), Expr::Lit(Literal::Int(42), _, _)));
    }

    #[test]
    fn test_beta_reduce_dead_arg() {
        // (\x -> 99) expensive => 99
        let lam = Expr::Lam(mk_var("x", 1), Box::new(mk_int(99)), Span::default());
        let expensive = Expr::App(
            Box::new(mk_var_expr("f", 2)),
            Box::new(mk_int(1)),
            Span::default(),
        );
        let result = try_beta_reduce(&lam, &expensive, 20);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), Expr::Lit(Literal::Int(99), _, _)));
    }

    #[test]
    fn test_no_beta_for_non_lambda() {
        let f = mk_var_expr("f", 1);
        let arg = mk_int(42);
        assert!(try_beta_reduce(&f, &arg, 20).is_none());
    }

    #[test]
    fn test_no_beta_for_multi_use_expensive_arg() {
        // (\x -> x + x) expensive => no reduction
        let body = Expr::App(
            Box::new(mk_var_expr("x", 1)),
            Box::new(mk_var_expr("x", 1)),
            Span::default(),
        );
        let lam = Expr::Lam(mk_var("x", 1), Box::new(body), Span::default());
        let expensive = Expr::App(
            Box::new(mk_var_expr("f", 2)),
            Box::new(mk_int(1)),
            Span::default(),
        );
        assert!(try_beta_reduce(&lam, &expensive, 20).is_none());
    }
}
