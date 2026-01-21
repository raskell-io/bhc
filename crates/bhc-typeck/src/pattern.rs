//! Pattern type checking.
//!
//! This module handles type inference and checking for patterns.
//! Patterns introduce bindings into scope and must be checked against
//! their expected types.
//!
//! ## Pattern Types
//!
//! - `_` (wildcard): Matches any type
//! - `x` (variable): Binds x to the matched type
//! - `42` (literal): Must match the literal's type
//! - `Con p1 p2` (constructor): Deconstructs the type
//! - `x@p` (as-pattern): Binds x and matches p
//! - `p :: T` (annotated): Constrains pattern type

use bhc_hir::{Lit, Pat};
use bhc_types::{Scheme, Ty};

use crate::context::TyCtxt;

/// Infer the type of a pattern, adding bindings to the environment.
///
/// Returns the type that values matching this pattern must have.
pub fn infer_pattern(ctx: &mut TyCtxt, pat: &Pat) -> Ty {
    match pat {
        Pat::Wild(_) => {
            // Wildcard matches anything
            ctx.fresh_ty()
        }

        Pat::Var(name, def_id, _span) => {
            // Variable pattern: create fresh type and bind
            let ty = ctx.fresh_ty();
            let scheme = Scheme::mono(ty.clone());
            // Bind by name (for nested pattern lookups)
            ctx.env.insert_local(*name, scheme.clone());
            // Also bind by DefId (for expression variable lookups)
            ctx.env.insert_global(*def_id, scheme);
            ty
        }

        Pat::Lit(lit, _span) => {
            // Literal pattern: return the literal's type
            infer_lit_type(ctx, lit)
        }

        Pat::Con(def_ref, sub_pats, span) => {
            // Constructor pattern: look up constructor type and unify
            let scheme = ctx
                .env
                .lookup_data_con_by_id(def_ref.def_id)
                .map(|i| i.scheme.clone());
            if let Some(s) = scheme {
                // Instantiate the constructor's type scheme
                let con_ty = ctx.instantiate(&s);

                // The constructor type should be: arg1 -> arg2 -> ... -> result
                // We need to unify sub-pattern types with arguments
                unify_constructor_pattern(ctx, &con_ty, sub_pats, *span)
            } else {
                // Constructor not found, return error type
                crate::diagnostics::emit_unbound_constructor(ctx, def_ref.def_id, *span);
                Ty::Error
            }
        }

        Pat::As(name, def_id, inner, _span) => {
            // As-pattern: bind name and check inner pattern
            let ty = infer_pattern(ctx, inner);
            let scheme = Scheme::mono(ty.clone());
            // Bind by name (for nested pattern lookups)
            ctx.env.insert_local(*name, scheme.clone());
            // Also bind by DefId (for expression variable lookups)
            ctx.env.insert_global(*def_id, scheme);
            ty
        }

        Pat::Or(left, right, span) => {
            // Or-pattern: both branches must have same type
            let left_ty = infer_pattern(ctx, left);
            let right_ty = infer_pattern(ctx, right);
            ctx.unify(&left_ty, &right_ty, *span);
            left_ty
        }

        Pat::Ann(inner, ty, _span) => {
            // Annotated pattern: check inner against annotation
            check_pattern(ctx, inner, ty);
            ty.clone()
        }

        Pat::Error(_) => Ty::Error,
    }
}

/// Check a pattern against an expected type.
///
/// Unifies the pattern's inferred type with the expected type and
/// adds bindings to the environment.
pub fn check_pattern(ctx: &mut TyCtxt, pat: &Pat, expected: &Ty) {
    let span = pat.span();
    let inferred = infer_pattern(ctx, pat);
    ctx.unify(&inferred, expected, span);
}

/// Unify a constructor pattern's arguments with a constructor type.
///
/// Given a constructor type like `a -> b -> T a b` and patterns `[p1, p2]`,
/// unifies `p1`'s type with `a`, `p2`'s type with `b`, and returns `T a b`.
fn unify_constructor_pattern(
    ctx: &mut TyCtxt,
    con_ty: &Ty,
    sub_pats: &[Pat],
    span: bhc_span::Span,
) -> Ty {
    let mut current_ty = con_ty.clone();
    let mut pat_iter = sub_pats.iter();
    let mut matched_any_arrows = false;

    // Peel off function arrows, matching each argument with a pattern
    while let Ty::Fun(arg_ty, result_ty) = current_ty {
        matched_any_arrows = true;
        match pat_iter.next() {
            Some(pat) => {
                check_pattern(ctx, pat, &arg_ty);
                current_ty = *result_ty;
            }
            None => {
                // More arguments in type than patterns (partial application in pattern)
                // This is unusual but we'll return the remaining function type
                return Ty::Fun(arg_ty, result_ty);
            }
        }
    }

    // Check for extra patterns
    if pat_iter.next().is_some() {
        // Only emit error if we actually matched some function arrows.
        // If the constructor had arity 0 (no function arrows), this might be a
        // record pattern where field types aren't known (e.g., external types).
        // In that case, infer fresh types for remaining patterns silently.
        if matched_any_arrows {
            crate::diagnostics::emit_too_many_pattern_args(ctx, span);
        } else {
            // Infer fresh types for all sub-patterns (record-style pattern with unknown fields)
            for pat in sub_pats {
                infer_pattern(ctx, pat);
            }
        }
    }

    current_ty
}

/// Infer the type of a literal.
fn infer_lit_type(ctx: &TyCtxt, lit: &Lit) -> Ty {
    match lit {
        Lit::Int(_) => ctx.builtins.int_ty.clone(),
        Lit::Float(_) => ctx.builtins.float_ty.clone(),
        Lit::Char(_) => ctx.builtins.char_ty.clone(),
        Lit::String(_) => ctx.builtins.string_ty.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_intern::Symbol;
    use bhc_span::{FileId, Span};

    fn test_context() -> TyCtxt {
        let mut ctx = TyCtxt::new(FileId::new(0));
        ctx.register_builtins();
        ctx
    }

    #[test]
    fn test_wildcard_pattern() {
        let mut ctx = test_context();
        let pat = Pat::Wild(Span::DUMMY);

        let ty = infer_pattern(&mut ctx, &pat);

        // Should be a fresh type variable
        assert!(matches!(ty, Ty::Var(_)));
    }

    #[test]
    fn test_var_pattern() {
        use bhc_hir::DefId;
        use bhc_index::Idx;

        let mut ctx = test_context();
        let x = Symbol::intern("x");
        let def_id = DefId::new(1000); // Dummy DefId for test
        let pat = Pat::Var(x, def_id, Span::DUMMY);

        let ty = infer_pattern(&mut ctx, &pat);

        // x should be bound in environment (by name)
        let scheme = ctx.env.lookup_local(x).unwrap();
        assert_eq!(scheme.ty, ty);

        // Also bound by DefId
        let scheme_by_id = ctx.env.lookup_global(def_id).unwrap();
        assert_eq!(scheme_by_id.ty, ty);
    }

    #[test]
    fn test_lit_pattern() {
        let mut ctx = test_context();
        let pat = Pat::Lit(Lit::Int(42), Span::DUMMY);

        let ty = infer_pattern(&mut ctx, &pat);

        assert_eq!(ty, ctx.builtins.int_ty);
    }

    #[test]
    fn test_as_pattern() {
        use bhc_hir::DefId;
        use bhc_index::Idx;

        let mut ctx = test_context();
        let x = Symbol::intern("x");
        let def_id = DefId::new(1001); // Dummy DefId for test
        let inner = Box::new(Pat::Lit(Lit::Int(42), Span::DUMMY));
        let pat = Pat::As(x, def_id, inner, Span::DUMMY);

        let ty = infer_pattern(&mut ctx, &pat);

        // Type should be Int
        assert_eq!(ty, ctx.builtins.int_ty);

        // x should be bound to Int (by name)
        let scheme = ctx.env.lookup_local(x).unwrap();
        assert_eq!(scheme.ty, ctx.builtins.int_ty);

        // Also bound by DefId
        let scheme_by_id = ctx.env.lookup_global(def_id).unwrap();
        assert_eq!(scheme_by_id.ty, ctx.builtins.int_ty);
    }
}
