//! Inlining decisions for Core IR bindings.
//!
//! Determines which let-bound variables should be replaced by their
//! right-hand sides during simplification.

use rustc_hash::FxHashMap;

use crate::{Bind, Expr, VarId};

use super::expr_util::{contains_toplevel_case, expr_size, is_cheap};
use super::occurrence::OccCount;

/// Build an inline environment: a map from variable IDs to their definitions
/// for variables that should be inlined.
///
/// Inlining criteria:
/// - **Always inline**: single-use, non-recursive bindings
/// - **Always inline**: cheap expressions (variables, literals) regardless of use count
/// - **Consider**: multi-use when `expr_size(rhs) <= threshold`
/// - **Never**: recursive bindings
pub fn build_inline_env(
    bindings: &[Bind],
    occs: &FxHashMap<VarId, OccCount>,
    inline_threshold: usize,
) -> FxHashMap<VarId, Expr> {
    let mut env = FxHashMap::default();

    for bind in bindings {
        match bind {
            Bind::NonRec(var, rhs) => {
                if should_inline(var.id, rhs, occs, inline_threshold, false) {
                    env.insert(var.id, *rhs.clone());
                }
            }
            Bind::Rec(_) => {
                // Never inline recursive bindings
            }
        }
    }

    env
}

/// Decide whether a specific binding should be inlined.
fn should_inline(
    var_id: VarId,
    rhs: &Expr,
    occs: &FxHashMap<VarId, OccCount>,
    threshold: usize,
    is_recursive: bool,
) -> bool {
    // Never inline recursive bindings
    if is_recursive {
        return false;
    }

    // Always inline cheap expressions (no work duplication)
    if is_cheap(rhs) {
        return true;
    }

    // Never inline expressions containing Case — they create LLVM basic
    // block terminators that break codegen when placed in argument positions.
    if contains_toplevel_case(rhs) {
        return false;
    }

    let occ = occs.get(&var_id).copied().unwrap_or(OccCount::Dead);

    match occ {
        OccCount::Dead => {
            // Dead bindings will be removed by dead code elimination,
            // no need to inline
            false
        }
        OccCount::Once => {
            // Single use, not under lambda — safe to inline
            // (Case expressions already excluded above)
            true
        }
        OccCount::OnceInLam => {
            // Single use under lambda — inline if small enough
            expr_size(rhs) <= threshold
        }
        OccCount::Many => {
            // Multiple uses — only inline small expressions
            expr_size(rhs) <= threshold / 2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::Ty;
    use crate::{Literal, Var};

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
    fn test_inline_cheap_multi_use() {
        // let x = y in ... (x used Many times) => inline because y is cheap
        let bindings = vec![Bind::NonRec(
            mk_var("x", 1),
            Box::new(mk_var_expr("y", 2)),
        )];
        let mut occs = FxHashMap::default();
        occs.insert(VarId::new(1), OccCount::Many);

        let env = build_inline_env(&bindings, &occs, 20);
        assert!(env.contains_key(&VarId::new(1)));
    }

    #[test]
    fn test_inline_single_use() {
        // let x = f y in ... (x used Once) => inline
        let app = Expr::App(
            Box::new(mk_var_expr("f", 10)),
            Box::new(mk_var_expr("y", 11)),
            Span::default(),
        );
        let bindings = vec![Bind::NonRec(mk_var("x", 1), Box::new(app))];
        let mut occs = FxHashMap::default();
        occs.insert(VarId::new(1), OccCount::Once);

        let env = build_inline_env(&bindings, &occs, 20);
        assert!(env.contains_key(&VarId::new(1)));
    }

    #[test]
    fn test_no_inline_recursive() {
        // let rec f = \x -> f x  => never inline
        let body = Expr::App(
            Box::new(mk_var_expr("f", 1)),
            Box::new(mk_var_expr("x", 2)),
            Span::default(),
        );
        let lam = Expr::Lam(mk_var("x", 2), Box::new(body), Span::default());
        let bindings = vec![Bind::Rec(vec![(mk_var("f", 1), Box::new(lam))])];
        let mut occs = FxHashMap::default();
        occs.insert(VarId::new(1), OccCount::Once);

        let env = build_inline_env(&bindings, &occs, 20);
        assert!(!env.contains_key(&VarId::new(1)));
    }

    #[test]
    fn test_no_inline_large_multi_use() {
        // Build a large expression (size >> threshold)
        let mut e: Expr = mk_int(1);
        for i in 0..30 {
            e = Expr::App(
                Box::new(mk_var_expr("f", 100 + i)),
                Box::new(e),
                Span::default(),
            );
        }

        let bindings = vec![Bind::NonRec(mk_var("x", 1), Box::new(e))];
        let mut occs = FxHashMap::default();
        occs.insert(VarId::new(1), OccCount::Many);

        let env = build_inline_env(&bindings, &occs, 20);
        assert!(!env.contains_key(&VarId::new(1)));
    }
}
