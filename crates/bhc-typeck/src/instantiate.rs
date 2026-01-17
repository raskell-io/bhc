//! Type scheme instantiation.
//!
//! This module implements instantiation of polymorphic type schemes.
//! When a polymorphic value is used, its type scheme is instantiated
//! with fresh type variables to produce a monomorphic type.
//!
//! ## Example
//!
//! The identity function has scheme `forall a. a -> a`.
//! When used, we instantiate it to `t1 -> t1` where `t1` is fresh.

use bhc_types::{Scheme, Ty};
use rustc_hash::FxHashMap;

use crate::context::TyCtxt;

/// Instantiate a type scheme with fresh type variables.
///
/// Replaces all bound type variables in the scheme with fresh unification
/// variables. Returns a monomorphic type suitable for unification.
///
/// # Example
///
/// ```ignore
/// // forall a b. a -> b -> a
/// let scheme = Scheme::poly(vec![a, b], Ty::fun(a, Ty::fun(b, a)));
///
/// // After instantiation: t1 -> t2 -> t1 (fresh t1, t2)
/// let ty = instantiate(ctx, &scheme);
/// ```
pub fn instantiate(ctx: &mut TyCtxt, scheme: &Scheme) -> Ty {
    // If monomorphic, no instantiation needed
    if scheme.is_mono() {
        return scheme.ty.clone();
    }

    // Create fresh type variables for each bound variable
    let mut subst: FxHashMap<u32, Ty> = FxHashMap::default();
    for var in &scheme.vars {
        let fresh = ctx.fresh_ty_var_with_kind(var.kind.clone());
        subst.insert(var.id, Ty::Var(fresh));
    }

    // Apply substitution to the type
    substitute(&scheme.ty, &subst)
}

/// Apply a substitution (mapping var IDs to types) to a type.
fn substitute(ty: &Ty, subst: &FxHashMap<u32, Ty>) -> Ty {
    match ty {
        Ty::Var(v) => subst.get(&v.id).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Con(_) => ty.clone(),
        Ty::App(f, a) => Ty::App(
            Box::new(substitute(f, subst)),
            Box::new(substitute(a, subst)),
        ),
        Ty::Fun(from, to) => Ty::Fun(
            Box::new(substitute(from, subst)),
            Box::new(substitute(to, subst)),
        ),
        Ty::Tuple(tys) => Ty::Tuple(tys.iter().map(|t| substitute(t, subst)).collect()),
        Ty::List(elem) => Ty::List(Box::new(substitute(elem, subst))),
        Ty::Forall(vars, body) => {
            // Remove bound vars from substitution to avoid capture
            let mut inner_subst = subst.clone();
            for v in vars {
                inner_subst.remove(&v.id);
            }
            Ty::Forall(vars.clone(), Box::new(substitute(body, &inner_subst)))
        }
        Ty::Error => Ty::Error,
    }
}

/// Instantiate a type scheme for a specific use with a given set of type arguments.
///
/// This is used for explicit type applications like `f @Int`.
#[allow(dead_code)]
pub fn instantiate_with(scheme: &Scheme, type_args: &[Ty]) -> Result<Ty, InstantiateError> {
    if type_args.len() != scheme.vars.len() {
        return Err(InstantiateError::ArityMismatch {
            expected: scheme.vars.len(),
            found: type_args.len(),
        });
    }

    // Create substitution from bound vars to provided type arguments
    let mut subst: FxHashMap<u32, Ty> = FxHashMap::default();
    for (var, ty) in scheme.vars.iter().zip(type_args.iter()) {
        subst.insert(var.id, ty.clone());
    }

    Ok(substitute(&scheme.ty, &subst))
}

/// Errors that can occur during instantiation.
#[derive(Debug, Clone, thiserror::Error)]
#[allow(dead_code)]
pub enum InstantiateError {
    /// Wrong number of type arguments.
    #[error("expected {expected} type arguments, found {found}")]
    ArityMismatch {
        /// Expected number of type arguments.
        expected: usize,
        /// Actual number of type arguments.
        found: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_intern::Symbol;
    use bhc_span::FileId;
    use bhc_types::{Kind, TyCon, TyVar};

    fn test_context() -> TyCtxt {
        TyCtxt::new(FileId::new(0))
    }

    #[test]
    fn test_instantiate_mono() {
        let mut ctx = test_context();
        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let scheme = Scheme::mono(int.clone());

        let result = instantiate(&mut ctx, &scheme);
        assert_eq!(result, int);
    }

    #[test]
    fn test_instantiate_poly() {
        let mut ctx = test_context();

        // Use a high ID for the original variable to ensure it's different from fresh ones
        let a = TyVar::new_star(1000);

        // forall a. a -> a
        let scheme = Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a)));

        let result = instantiate(&mut ctx, &scheme);

        // Result should be t -> t for some fresh t
        match result {
            Ty::Fun(from, to) => match (*from, *to) {
                (Ty::Var(v1), Ty::Var(v2)) => {
                    // Both should be the same fresh variable
                    assert_eq!(v1.id, v2.id);
                    // And different from the original (which was 1000)
                    assert_ne!(v1.id, 1000);
                }
                _ => panic!("expected function between type vars"),
            },
            _ => panic!("expected function type"),
        }
    }

    #[test]
    fn test_instantiate_with_explicit() {
        let a = TyVar::new_star(0);
        let b = TyVar::new_star(1);

        // forall a b. a -> b -> a
        let scheme = Scheme::poly(
            vec![a.clone(), b.clone()],
            Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b), Ty::Var(a))),
        );

        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        let result = instantiate_with(&scheme, &[int.clone(), bool_ty.clone()]).unwrap();

        // Should be Int -> Bool -> Int
        match result {
            Ty::Fun(from, rest) => {
                assert_eq!(*from, int);
                match *rest {
                    Ty::Fun(from2, to) => {
                        assert_eq!(*from2, bool_ty);
                        assert_eq!(*to, int);
                    }
                    _ => panic!("expected nested function"),
                }
            }
            _ => panic!("expected function type"),
        }
    }

    #[test]
    fn test_instantiate_with_wrong_arity() {
        let a = TyVar::new_star(0);
        let scheme = Scheme::poly(vec![a.clone()], Ty::Var(a));

        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        let result = instantiate_with(&scheme, &[int, bool_ty]);
        assert!(matches!(
            result,
            Err(InstantiateError::ArityMismatch {
                expected: 1,
                found: 2
            })
        ));
    }
}
