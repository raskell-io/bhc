//! Occurrence analysis for Core IR variables.
//!
//! Counts how many times each variable is referenced, and whether those
//! references occur under lambdas. This information drives inlining and
//! dead code elimination decisions.

use rustc_hash::FxHashMap;

use crate::{Bind, Expr, VarId};

/// Occurrence count for a variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccCount {
    /// Never referenced.
    Dead,
    /// Referenced exactly once, not under a lambda.
    Once,
    /// Referenced exactly once, but under a lambda (may be called many times).
    OnceInLam,
    /// Referenced multiple times.
    Many,
}

/// Analyze variable occurrences across a list of top-level bindings.
pub fn analyze_module_occurrences(bindings: &[Bind]) -> FxHashMap<VarId, OccCount> {
    let mut counts: FxHashMap<VarId, u32> = FxHashMap::default();
    let mut in_lam: FxHashMap<VarId, bool> = FxHashMap::default();

    for bind in bindings {
        match bind {
            Bind::NonRec(_, rhs) => {
                count_expr(rhs, &mut counts, &mut in_lam, false);
            }
            Bind::Rec(pairs) => {
                for (_, rhs) in pairs {
                    count_expr(rhs, &mut counts, &mut in_lam, false);
                }
            }
        }
    }

    counts
        .into_iter()
        .map(|(id, count)| {
            let occ = match count {
                0 => OccCount::Dead,
                1 => {
                    if in_lam.get(&id).copied().unwrap_or(false) {
                        OccCount::OnceInLam
                    } else {
                        OccCount::Once
                    }
                }
                _ => OccCount::Many,
            };
            (id, occ)
        })
        .collect()
}

/// Analyze variable occurrences within a single expression.
pub fn analyze_occurrences(expr: &Expr) -> FxHashMap<VarId, OccCount> {
    let mut counts: FxHashMap<VarId, u32> = FxHashMap::default();
    let mut in_lam: FxHashMap<VarId, bool> = FxHashMap::default();
    count_expr(expr, &mut counts, &mut in_lam, false);

    counts
        .into_iter()
        .map(|(id, count)| {
            let occ = match count {
                0 => OccCount::Dead,
                1 => {
                    if in_lam.get(&id).copied().unwrap_or(false) {
                        OccCount::OnceInLam
                    } else {
                        OccCount::Once
                    }
                }
                _ => OccCount::Many,
            };
            (id, occ)
        })
        .collect()
}

fn count_expr(
    expr: &Expr,
    counts: &mut FxHashMap<VarId, u32>,
    in_lam: &mut FxHashMap<VarId, bool>,
    inside_lam: bool,
) {
    match expr {
        Expr::Var(v, _) => {
            *counts.entry(v.id).or_insert(0) += 1;
            if inside_lam {
                in_lam.insert(v.id, true);
            }
        }
        Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => {}
        Expr::App(f, a, _) => {
            count_expr(f, counts, in_lam, inside_lam);
            count_expr(a, counts, in_lam, inside_lam);
        }
        Expr::TyApp(f, _, _) => count_expr(f, counts, in_lam, inside_lam),
        Expr::Lam(_, body, _) => {
            // References inside lambda bodies are marked as "inside lam"
            count_expr(body, counts, in_lam, true);
        }
        Expr::TyLam(_, body, _) => count_expr(body, counts, in_lam, inside_lam),
        Expr::Let(bind, body, _) => {
            match bind.as_ref() {
                Bind::NonRec(_, rhs) => {
                    count_expr(rhs, counts, in_lam, inside_lam);
                }
                Bind::Rec(pairs) => {
                    for (_, rhs) in pairs {
                        count_expr(rhs, counts, in_lam, inside_lam);
                    }
                }
            }
            count_expr(body, counts, in_lam, inside_lam);
        }
        Expr::Case(scrut, alts, _, _) => {
            count_expr(scrut, counts, in_lam, inside_lam);
            for alt in alts {
                count_expr(&alt.rhs, counts, in_lam, inside_lam);
            }
        }
        Expr::Lazy(e, _) | Expr::Cast(e, _, _) | Expr::Tick(_, e, _) => {
            count_expr(e, counts, in_lam, inside_lam);
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
    fn test_dead_variable() {
        // let x = 1 in 2  -- x is dead
        let binds = vec![Bind::NonRec(mk_var("x", 1), Box::new(mk_int(1)))];
        let body = mk_int(2);
        let full = Expr::Let(
            Box::new(binds[0].clone()),
            Box::new(body),
            Span::default(),
        );
        let occs = analyze_occurrences(&full);
        assert_eq!(occs.get(&VarId::new(1)).copied(), None);
    }

    #[test]
    fn test_once_variable() {
        // let x = 1 in x  -- x is used once
        let body = mk_var_expr("x", 1);
        let full = Expr::Let(
            Box::new(Bind::NonRec(mk_var("x", 1), Box::new(mk_int(1)))),
            Box::new(body),
            Span::default(),
        );
        let occs = analyze_occurrences(&full);
        assert_eq!(occs.get(&VarId::new(1)).copied(), Some(OccCount::Once));
    }

    #[test]
    fn test_many_variable() {
        // x + x  -- x used twice
        let e = Expr::App(
            Box::new(mk_var_expr("x", 1)),
            Box::new(mk_var_expr("x", 1)),
            Span::default(),
        );
        let occs = analyze_occurrences(&e);
        assert_eq!(occs.get(&VarId::new(1)).copied(), Some(OccCount::Many));
    }

    #[test]
    fn test_once_in_lam() {
        // \y -> x  -- x used once, under lambda
        let lam = Expr::Lam(
            mk_var("y", 2),
            Box::new(mk_var_expr("x", 1)),
            Span::default(),
        );
        let occs = analyze_occurrences(&lam);
        assert_eq!(
            occs.get(&VarId::new(1)).copied(),
            Some(OccCount::OnceInLam)
        );
    }
}
