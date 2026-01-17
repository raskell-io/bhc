//! Type generalization for let-polymorphism.
//!
//! This module implements generalization, which converts a monomorphic type
//! into a polymorphic type scheme by quantifying over type variables that
//! are free in the type but not in the environment.
//!
//! ## Let-Polymorphism
//!
//! In Hindley-Milner, types are generalized at let-bindings:
//!
//! ```text
//! let id = \x -> x  -- id gets type forall a. a -> a
//! in (id 1, id True) -- uses different instantiations
//! ```
//!
//! Without generalization, `id` would have a monomorphic type and the
//! second use would fail to type check.

use bhc_types::{Scheme, Ty, TyVar};
use rustc_hash::FxHashSet;

use crate::context::TyCtxt;

/// Generalize a type into a type scheme.
///
/// Quantifies over all type variables that are:
/// 1. Free in the type
/// 2. Not free in the type environment
///
/// # Arguments
///
/// * `ctx` - The type checking context (for accessing the environment)
/// * `ty` - The type to generalize
///
/// # Returns
///
/// A type scheme with the appropriate bound variables.
pub fn generalize(ctx: &TyCtxt, ty: &Ty) -> Scheme {
    // Apply current substitution first
    let ty = ctx.apply_subst(ty);

    // Get free variables in the type
    let type_fvs = ty.free_vars();

    // Get free variables in the environment
    let env_fvs: FxHashSet<u32> = ctx.env.free_vars().iter().map(|v| v.id).collect();

    // Quantify over variables free in type but not in environment
    let quantified: Vec<TyVar> = type_fvs
        .into_iter()
        .filter(|v| !env_fvs.contains(&v.id))
        .collect();

    if quantified.is_empty() {
        Scheme::mono(ty)
    } else {
        Scheme::poly(quantified, ty)
    }
}

/// Generalize a type with additional constraints.
///
/// Similar to `generalize`, but also handles type class constraints
/// that apply to the quantified variables.
#[allow(dead_code)]
pub fn generalize_with_constraints(
    ctx: &TyCtxt,
    ty: &Ty,
    constraints: Vec<bhc_types::Constraint>,
) -> Scheme {
    let ty = ctx.apply_subst(ty);
    let type_fvs = ty.free_vars();
    let env_fvs: FxHashSet<u32> = ctx.env.free_vars().iter().map(|v| v.id).collect();

    let quantified: Vec<TyVar> = type_fvs
        .into_iter()
        .filter(|v| !env_fvs.contains(&v.id))
        .collect();

    // Filter constraints to only those mentioning quantified variables
    let quantified_ids: FxHashSet<u32> = quantified.iter().map(|v| v.id).collect();
    let relevant_constraints: Vec<_> = constraints
        .into_iter()
        .filter(|c| {
            c.args.iter().any(|arg| {
                arg.free_vars().iter().any(|v| quantified_ids.contains(&v.id))
            })
        })
        .collect();

    if quantified.is_empty() {
        Scheme {
            vars: Vec::new(),
            constraints: relevant_constraints,
            ty,
        }
    } else {
        Scheme::qualified(quantified, relevant_constraints, ty)
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
    fn test_generalize_monomorphic() {
        let ctx = test_context();
        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));

        let scheme = generalize(&ctx, &int);

        assert!(scheme.is_mono());
        assert_eq!(scheme.ty, int);
    }

    #[test]
    fn test_generalize_polymorphic() {
        let mut ctx = test_context();
        let a = ctx.fresh_ty_var();

        // a -> a
        let ty = Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()));

        let scheme = generalize(&ctx, &ty);

        assert!(!scheme.is_mono());
        assert_eq!(scheme.vars.len(), 1);
        assert_eq!(scheme.vars[0].id, a.id);
    }

    #[test]
    fn test_generalize_with_env_binding() {
        let mut ctx = test_context();
        let a = ctx.fresh_ty_var();
        let b = ctx.fresh_ty_var();

        // Bind 'a' in environment (it shouldn't be generalized)
        ctx.env
            .insert_local(Symbol::intern("x"), Scheme::mono(Ty::Var(a.clone())));

        // a -> b
        let ty = Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()));

        let scheme = generalize(&ctx, &ty);

        // Only 'b' should be generalized since 'a' is in environment
        assert_eq!(scheme.vars.len(), 1);
        assert_eq!(scheme.vars[0].id, b.id);
    }

    #[test]
    fn test_generalize_nested_function() {
        let mut ctx = test_context();
        let a = ctx.fresh_ty_var();
        let b = ctx.fresh_ty_var();
        let c = ctx.fresh_ty_var();

        // a -> b -> c
        let ty = Ty::fun(
            Ty::Var(a.clone()),
            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
        );

        let scheme = generalize(&ctx, &ty);

        // All three should be generalized
        assert_eq!(scheme.vars.len(), 3);
    }
}
