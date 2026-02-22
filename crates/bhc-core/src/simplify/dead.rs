//! Dead binding elimination for Core IR.
//!
//! Removes `let x = rhs in body` when `x` is never referenced in `body`.

use rustc_hash::FxHashMap;

use crate::{Expr, VarId};

use super::occurrence::OccCount;

/// Remove a dead non-recursive binding, returning just the body.
/// Returns `Some(body)` if the binding is dead, `None` if it's live.
pub fn try_eliminate_dead_nonrec(
    var_id: VarId,
    _body: &Expr,
    occs: &FxHashMap<VarId, OccCount>,
) -> bool {
    match occs.get(&var_id) {
        None | Some(OccCount::Dead) => true,
        _ => false,
    }
}

/// Filter dead entries from a recursive binding group.
/// Returns `None` if all entries survive, `Some(filtered)` if any were removed.
/// If all entries are dead, returns `Some(vec![])`.
pub fn filter_dead_rec(
    binds: &[(crate::Var, Box<Expr>)],
    _body: &Expr,
    occs: &FxHashMap<VarId, OccCount>,
) -> Option<Vec<(crate::Var, Box<Expr>)>> {
    let live: Vec<(crate::Var, Box<Expr>)> = binds
        .iter()
        .filter(|(v, _)| {
            matches!(
                occs.get(&v.id),
                Some(OccCount::Once | OccCount::OnceInLam | OccCount::Many)
            )
        })
        .cloned()
        .collect();

    if live.len() == binds.len() {
        None // Nothing removed
    } else {
        Some(live)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::Ty;
    use crate::Var;

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
    fn test_dead_nonrec_eliminated() {
        let body = mk_int(42);
        let mut occs = FxHashMap::default();
        // x not in occs => dead
        assert!(try_eliminate_dead_nonrec(VarId::new(1), &body, &occs));

        // Explicitly dead
        occs.insert(VarId::new(2), OccCount::Dead);
        assert!(try_eliminate_dead_nonrec(VarId::new(2), &body, &occs));
    }

    #[test]
    fn test_live_nonrec_preserved() {
        let body = mk_var_expr("x", 1);
        let mut occs = FxHashMap::default();
        occs.insert(VarId::new(1), OccCount::Once);
        assert!(!try_eliminate_dead_nonrec(VarId::new(1), &body, &occs));
    }

    #[test]
    fn test_filter_dead_rec() {
        let binds = vec![
            (mk_var("x", 1), Box::new(mk_int(1))),
            (mk_var("y", 2), Box::new(mk_int(2))),
        ];
        let body = mk_var_expr("x", 1);
        let mut occs = FxHashMap::default();
        occs.insert(VarId::new(1), OccCount::Once);
        // y is not in occs => dead

        let result = filter_dead_rec(&binds, &body, &occs);
        assert!(result.is_some());
        let live = result.unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].0.id, VarId::new(1));
    }

    #[test]
    fn test_filter_all_live() {
        let binds = vec![
            (mk_var("x", 1), Box::new(mk_int(1))),
            (mk_var("y", 2), Box::new(mk_int(2))),
        ];
        let body = mk_int(0);
        let mut occs = FxHashMap::default();
        occs.insert(VarId::new(1), OccCount::Once);
        occs.insert(VarId::new(2), OccCount::Many);

        let result = filter_dead_rec(&binds, &body, &occs);
        assert!(result.is_none()); // Nothing removed
    }
}
