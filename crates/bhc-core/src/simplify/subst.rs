//! Capture-avoiding substitution for Core IR expressions.

use bhc_index::Idx;
use bhc_intern::Symbol;
use rustc_hash::FxHashMap;

use crate::{Alt, Bind, Expr, Var, VarId};

use super::expr_util::{free_var_ids, fresh_var_id};

/// Substitute multiple variables simultaneously with capture avoidance.
pub fn substitute(expr: Expr, subst: &FxHashMap<VarId, Expr>) -> Expr {
    if subst.is_empty() {
        return expr;
    }
    subst_expr(expr, subst)
}

/// Substitute a single variable in an expression with capture avoidance.
pub fn substitute_single(expr: Expr, var_id: VarId, replacement: &Expr) -> Expr {
    let mut subst = FxHashMap::default();
    subst.insert(var_id, replacement.clone());
    substitute(expr, &subst)
}

fn subst_expr(expr: Expr, subst: &FxHashMap<VarId, Expr>) -> Expr {
    match expr {
        Expr::Var(ref v, _) => {
            if let Some(replacement) = subst.get(&v.id) {
                replacement.clone()
            } else {
                expr
            }
        }
        Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => expr,
        Expr::App(f, a, span) => {
            Expr::App(Box::new(subst_expr(*f, subst)), Box::new(subst_expr(*a, subst)), span)
        }
        Expr::TyApp(f, ty, span) => Expr::TyApp(Box::new(subst_expr(*f, subst)), ty, span),
        Expr::Lam(x, body, span) => {
            let (new_x, new_subst) = handle_binder(x, subst);
            Expr::Lam(new_x, Box::new(subst_expr(*body, &new_subst)), span)
        }
        Expr::TyLam(tv, body, span) => {
            Expr::TyLam(tv, Box::new(subst_expr(*body, subst)), span)
        }
        Expr::Let(bind, body, span) => {
            let (new_bind, new_subst) = subst_bind(*bind, subst);
            Expr::Let(
                Box::new(new_bind),
                Box::new(subst_expr(*body, &new_subst)),
                span,
            )
        }
        Expr::Case(scrut, alts, ty, span) => {
            let new_scrut = subst_expr(*scrut, subst);
            let new_alts = alts.into_iter().map(|alt| subst_alt(alt, subst)).collect();
            Expr::Case(Box::new(new_scrut), new_alts, ty, span)
        }
        Expr::Lazy(e, span) => Expr::Lazy(Box::new(subst_expr(*e, subst)), span),
        Expr::Cast(e, coercion, span) => {
            Expr::Cast(Box::new(subst_expr(*e, subst)), coercion, span)
        }
        Expr::Tick(tick, e, span) => Expr::Tick(tick, Box::new(subst_expr(*e, subst)), span),
    }
}

/// Handle a binder: if the binder shadows a substituted variable, remove it from subst.
/// If the binder would capture a free variable from the substitution values, alpha-rename.
fn handle_binder(binder: Var, subst: &FxHashMap<VarId, Expr>) -> (Var, FxHashMap<VarId, Expr>) {
    // Remove the binder from substitution (it shadows)
    let mut new_subst: FxHashMap<VarId, Expr> = subst
        .iter()
        .filter(|(k, _)| **k != binder.id)
        .map(|(k, v)| (*k, v.clone()))
        .collect();

    if new_subst.is_empty() {
        return (binder, new_subst);
    }

    // Check if the binder would capture any free variable in substitution values
    let subst_free: rustc_hash::FxHashSet<VarId> = new_subst
        .values()
        .flat_map(|e| free_var_ids(e))
        .collect();

    if subst_free.contains(&binder.id) {
        // Alpha-rename the binder to avoid capture
        let fresh_id = fresh_var_id();
        let fresh_name = Symbol::intern(&format!("${}", fresh_id.index()));
        let new_binder = Var::new(fresh_name, fresh_id, binder.ty.clone());

        // Add a substitution from old binder to new binder variable
        new_subst.insert(
            binder.id,
            Expr::Var(new_binder.clone(), bhc_span::Span::default()),
        );

        (new_binder, new_subst)
    } else {
        (binder, new_subst)
    }
}

fn subst_bind(bind: Bind, subst: &FxHashMap<VarId, Expr>) -> (Bind, FxHashMap<VarId, Expr>) {
    match bind {
        Bind::NonRec(x, rhs) => {
            // Substitute in RHS first (before the binder is in scope)
            let new_rhs = subst_expr(*rhs, subst);
            let (new_x, body_subst) = handle_binder(x, subst);
            (Bind::NonRec(new_x, Box::new(new_rhs)), body_subst)
        }
        Bind::Rec(binds) => {
            // For recursive bindings, all binders are in scope simultaneously
            let mut body_subst = subst.clone();
            let mut new_vars = Vec::with_capacity(binds.len());

            // First pass: handle all binders
            for (x, _) in &binds {
                body_subst.remove(&x.id);
            }

            // Check for capture
            let subst_free: rustc_hash::FxHashSet<VarId> = body_subst
                .values()
                .flat_map(|e| free_var_ids(e))
                .collect();

            let mut rename_map: FxHashMap<VarId, Var> = FxHashMap::default();
            for (x, _) in &binds {
                if subst_free.contains(&x.id) {
                    let fresh_id = fresh_var_id();
                    let fresh_name = Symbol::intern(&format!("${}", fresh_id.index()));
                    let new_x = Var::new(fresh_name, fresh_id, x.ty.clone());
                    body_subst.insert(
                        x.id,
                        Expr::Var(new_x.clone(), bhc_span::Span::default()),
                    );
                    rename_map.insert(x.id, new_x.clone());
                    new_vars.push(new_x);
                } else {
                    new_vars.push(x.clone());
                }
            }

            // Second pass: substitute in all RHSs
            let new_binds: Vec<(Var, Box<Expr>)> = new_vars
                .into_iter()
                .zip(binds.into_iter().map(|(_, rhs)| rhs))
                .map(|(v, rhs)| (v, Box::new(subst_expr(*rhs, &body_subst))))
                .collect();

            (Bind::Rec(new_binds), body_subst)
        }
    }
}

fn subst_alt(alt: Alt, subst: &FxHashMap<VarId, Expr>) -> Alt {
    if alt.binders.is_empty() {
        return Alt {
            con: alt.con,
            binders: alt.binders,
            rhs: subst_expr(alt.rhs, subst),
        };
    }

    // Remove binders from subst and check for capture
    let mut alt_subst = subst.clone();
    for b in &alt.binders {
        alt_subst.remove(&b.id);
    }

    if alt_subst.is_empty() {
        return Alt {
            con: alt.con,
            binders: alt.binders,
            rhs: alt.rhs,
        };
    }

    let subst_free: rustc_hash::FxHashSet<VarId> = alt_subst
        .values()
        .flat_map(|e| free_var_ids(e))
        .collect();

    let mut new_binders = Vec::with_capacity(alt.binders.len());
    for b in alt.binders {
        if subst_free.contains(&b.id) {
            let fresh_id = fresh_var_id();
            let fresh_name = Symbol::intern(&format!("${}", fresh_id.index()));
            let new_b = Var::new(fresh_name, fresh_id, b.ty.clone());
            alt_subst.insert(b.id, Expr::Var(new_b.clone(), bhc_span::Span::default()));
            new_binders.push(new_b);
        } else {
            new_binders.push(b);
        }
    }

    Alt {
        con: alt.con,
        binders: new_binders,
        rhs: subst_expr(alt.rhs, &alt_subst),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::Ty;

    fn mk_var(name: &str, id: u32) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id as usize), Ty::Error)
    }

    fn mk_var_expr(name: &str, id: u32) -> Expr {
        Expr::Var(mk_var(name, id), Span::default())
    }

    fn mk_int(n: i64) -> Expr {
        Expr::Lit(crate::Literal::Int(n), Ty::Error, Span::default())
    }

    fn get_var_id(expr: &Expr) -> Option<VarId> {
        if let Expr::Var(v, _) = expr {
            Some(v.id)
        } else {
            None
        }
    }

    #[test]
    fn test_substitute_var() {
        let e = mk_var_expr("x", 1);
        let result = substitute_single(e, VarId::new(1), &mk_int(42));
        assert!(matches!(result, Expr::Lit(crate::Literal::Int(42), _, _)));
    }

    #[test]
    fn test_substitute_different_var_unchanged() {
        let e = mk_var_expr("y", 2);
        let result = substitute_single(e, VarId::new(1), &mk_int(42));
        assert_eq!(get_var_id(&result), Some(VarId::new(2)));
    }

    #[test]
    fn test_empty_subst_is_identity() {
        let e = mk_var_expr("x", 1);
        let subst = FxHashMap::default();
        let result = substitute(e.clone(), &subst);
        // Should be the same expression
        assert_eq!(get_var_id(&result), Some(VarId::new(1)));
    }

    #[test]
    fn test_lambda_shadows_substitution() {
        // (\x -> x)[x := 42] should be (\x -> x), not (\x -> 42)
        let lam = Expr::Lam(
            mk_var("x", 1),
            Box::new(mk_var_expr("x", 1)),
            Span::default(),
        );
        let result = substitute_single(lam, VarId::new(1), &mk_int(42));
        if let Expr::Lam(_, body, _) = result {
            assert!(matches!(*body, Expr::Var(_, _)));
        } else {
            panic!("expected Lam");
        }
    }

    #[test]
    fn test_capture_avoidance() {
        // (\y -> x + y)[x := y] should alpha-rename y
        // to avoid capturing the free y in the replacement
        let body = Expr::App(
            Box::new(mk_var_expr("x", 1)),
            Box::new(mk_var_expr("y", 2)),
            Span::default(),
        );
        let lam = Expr::Lam(mk_var("y", 2), Box::new(body), Span::default());
        let result = substitute_single(lam, VarId::new(1), &mk_var_expr("y", 2));

        // The lambda binder should have been renamed
        if let Expr::Lam(new_binder, _, _) = &result {
            assert_ne!(new_binder.id, VarId::new(2), "binder should be renamed");
        } else {
            panic!("expected Lam");
        }
    }
}
