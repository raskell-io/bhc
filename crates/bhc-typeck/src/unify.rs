//! Type unification with occurs check.
//!
//! This module implements the unification algorithm for Hindley-Milner
//! type inference. Unification finds a substitution that makes two types
//! equal, or reports an error if no such substitution exists.
//!
//! ## Algorithm
//!
//! Unification proceeds recursively:
//!
//! 1. Apply current substitution to both types
//! 2. If both are variables with same ID, succeed
//! 3. If one is a variable, bind it (with occurs check)
//! 4. If both are constructors/functions, unify components
//! 5. Otherwise, report mismatch
//!
//! ## Occurs Check
//!
//! The occurs check prevents infinite types. Before binding a variable
//! `a` to a type `t`, we verify that `a` does not appear in `t`.

use bhc_span::Span;
use bhc_types::{Ty, TyVar};

use crate::context::TyCtxt;
use crate::diagnostics;

/// Unify two types, updating the substitution in the context.
///
/// If unification fails, an error diagnostic is emitted and the types
/// may be left partially unified.
pub fn unify(ctx: &mut TyCtxt, t1: &Ty, t2: &Ty, span: Span) {
    // Apply current substitution to both types
    let t1 = ctx.apply_subst(t1);
    let t2 = ctx.apply_subst(t2);

    unify_inner(ctx, &t1, &t2, span);
}

/// Inner unification after substitution has been applied.
fn unify_inner(ctx: &mut TyCtxt, t1: &Ty, t2: &Ty, span: Span) {
    match (t1, t2) {
        // Error types unify with anything (error recovery)
        (Ty::Error, _) | (_, Ty::Error) => {}

        // Same variable unifies trivially
        (Ty::Var(v1), Ty::Var(v2)) if v1.id == v2.id => {}

        // Variable on either side: bind it
        (Ty::Var(v), t) | (t, Ty::Var(v)) => {
            bind_var(ctx, v, t, span);
        }

        // Type constructors: must have same name
        (Ty::Con(c1), Ty::Con(c2)) => {
            if c1.name != c2.name {
                diagnostics::emit_type_mismatch(ctx, t1, t2, span);
            }
        }

        // Type applications: unify both components
        (Ty::App(f1, a1), Ty::App(f2, a2)) => {
            unify_inner(ctx, f1, f2, span);
            // Apply updated substitution before unifying arguments
            let a1 = ctx.apply_subst(a1);
            let a2 = ctx.apply_subst(a2);
            unify_inner(ctx, &a1, &a2, span);
        }

        // Function types: unify domain and codomain
        (Ty::Fun(from1, to1), Ty::Fun(from2, to2)) => {
            unify_inner(ctx, from1, from2, span);
            // Apply updated substitution before unifying codomains
            let to1 = ctx.apply_subst(to1);
            let to2 = ctx.apply_subst(to2);
            unify_inner(ctx, &to1, &to2, span);
        }

        // Tuple types: must have same length, unify components
        (Ty::Tuple(tys1), Ty::Tuple(tys2)) => {
            if tys1.len() != tys2.len() {
                diagnostics::emit_type_mismatch(ctx, t1, t2, span);
                return;
            }
            for (elem1, elem2) in tys1.iter().zip(tys2.iter()) {
                let applied1 = ctx.apply_subst(elem1);
                let applied2 = ctx.apply_subst(elem2);
                unify_inner(ctx, &applied1, &applied2, span);
            }
        }

        // List types: unify element types
        (Ty::List(elem1), Ty::List(elem2)) => {
            unify_inner(ctx, elem1, elem2, span);
        }

        // Forall types: handle carefully (should be instantiated first)
        (Ty::Forall(_, body1), t2) => {
            // This shouldn't normally happen in Algorithm W
            // For now, try to unify the body
            unify_inner(ctx, body1, t2, span);
        }

        (t1, Ty::Forall(_, body2)) => {
            unify_inner(ctx, t1, body2, span);
        }

        // Different type structures: mismatch
        _ => {
            diagnostics::emit_type_mismatch(ctx, t1, t2, span);
        }
    }
}

/// Bind a type variable to a type after occurs check.
fn bind_var(ctx: &mut TyCtxt, var: &TyVar, ty: &Ty, span: Span) {
    // Skip if binding to self
    if let Ty::Var(v) = ty {
        if v.id == var.id {
            return;
        }
    }

    // Occurs check: ensure var doesn't appear in ty
    if occurs_check(var, ty) {
        diagnostics::emit_occurs_check_error(ctx, var, ty, span);
        return;
    }

    // Add binding to substitution
    ctx.subst.insert(var.clone(), ty.clone());
}

/// Check if a type variable occurs in a type (prevents infinite types).
fn occurs_check(var: &TyVar, ty: &Ty) -> bool {
    match ty {
        Ty::Var(v) => v.id == var.id,
        Ty::Con(_) | Ty::Error => false,
        Ty::App(f, a) => occurs_check(var, f) || occurs_check(var, a),
        Ty::Fun(from, to) => occurs_check(var, from) || occurs_check(var, to),
        Ty::Tuple(tys) => tys.iter().any(|t| occurs_check(var, t)),
        Ty::List(elem) => occurs_check(var, elem),
        Ty::Forall(bound, body) => {
            // var occurring in bound vars doesn't count (it's shadowed)
            if bound.iter().any(|v| v.id == var.id) {
                false
            } else {
                occurs_check(var, body)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_intern::Symbol;
    use bhc_span::FileId;
    use bhc_types::{Kind, TyCon};

    fn test_context() -> TyCtxt {
        TyCtxt::new(FileId::new(0))
    }

    #[test]
    fn test_unify_same_var() {
        let mut ctx = test_context();
        let a = ctx.fresh_ty_var();

        unify(&mut ctx, &Ty::Var(a.clone()), &Ty::Var(a), Span::DUMMY);

        assert!(!ctx.has_errors());
    }

    #[test]
    fn test_unify_var_with_con() {
        let mut ctx = test_context();
        let a = ctx.fresh_ty_var();
        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));

        unify(&mut ctx, &Ty::Var(a.clone()), &int, Span::DUMMY);

        assert!(!ctx.has_errors());
        assert_eq!(ctx.apply_subst(&Ty::Var(a)), int);
    }

    #[test]
    fn test_unify_function_types() {
        let mut ctx = test_context();
        let a = ctx.fresh_ty_var();
        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));

        // a -> a ~ Int -> Int
        let t1 = Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()));
        let t2 = Ty::fun(int.clone(), int.clone());

        unify(&mut ctx, &t1, &t2, Span::DUMMY);

        assert!(!ctx.has_errors());
        assert_eq!(ctx.apply_subst(&Ty::Var(a)), int);
    }

    #[test]
    fn test_occurs_check_fails() {
        let mut ctx = test_context();
        let a = ctx.fresh_ty_var();

        // a ~ a -> a (infinite type)
        let t = Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()));

        unify(&mut ctx, &Ty::Var(a), &t, Span::DUMMY);

        assert!(ctx.has_errors());
    }

    #[test]
    fn test_unify_mismatch() {
        let mut ctx = test_context();
        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        unify(&mut ctx, &int, &bool_ty, Span::DUMMY);

        assert!(ctx.has_errors());
    }

    #[test]
    fn test_unify_tuples() {
        let mut ctx = test_context();
        let a = ctx.fresh_ty_var();
        let b = ctx.fresh_ty_var();
        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        let t1 = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
        let t2 = Ty::Tuple(vec![int.clone(), bool_ty.clone()]);

        unify(&mut ctx, &t1, &t2, Span::DUMMY);

        assert!(!ctx.has_errors());
        assert_eq!(ctx.apply_subst(&Ty::Var(a)), int);
        assert_eq!(ctx.apply_subst(&Ty::Var(b)), bool_ty);
    }

    #[test]
    fn test_unify_tuple_length_mismatch() {
        let mut ctx = test_context();
        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));

        let t1 = Ty::Tuple(vec![int.clone(), int.clone()]);
        let t2 = Ty::Tuple(vec![int.clone()]);

        unify(&mut ctx, &t1, &t2, Span::DUMMY);

        assert!(ctx.has_errors());
    }
}
