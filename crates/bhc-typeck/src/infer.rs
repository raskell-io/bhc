//! Type inference for expressions (Algorithm W).
//!
//! This module implements the core type inference algorithm for HIR
//! expressions. It follows Algorithm W from Hindley-Milner type inference
//! with extensions for Haskell-specific constructs.
//!
//! ## Algorithm Overview
//!
//! 1. **Literals**: Return the literal's type
//! 2. **Variables**: Instantiate the variable's type scheme
//! 3. **Application**: Infer function and argument, unify with fresh result
//! 4. **Lambda**: Infer body with parameter bindings, build function type
//! 5. **Let**: Infer binding, generalize, then infer body

use bhc_hir::{Binding, CaseAlt, Expr, Lit};
use bhc_intern::Symbol;
use bhc_types::{Scheme, Ty};

use crate::context::TyCtxt;
use crate::diagnostics;

/// Infer the type of an expression.
///
/// This is the main entry point for expression type inference.
/// Returns the inferred type (which may contain type variables).
#[allow(clippy::too_many_lines)]
pub fn infer_expr(ctx: &mut TyCtxt, expr: &Expr) -> Ty {
    match expr {
        Expr::Lit(lit, _span) => infer_lit(ctx, lit),

        Expr::Var(def_ref) => {
            // Look up variable and instantiate its type scheme
            let scheme = ctx.env.lookup_def_id(def_ref.def_id).cloned();
            if let Some(s) = scheme {
                ctx.instantiate(&s)
            } else {
                diagnostics::emit_unbound_var(ctx, def_ref.def_id, def_ref.span);
                Ty::Error
            }
        }

        Expr::Con(def_ref) => {
            // Look up data constructor
            let scheme = ctx
                .env
                .lookup_data_con_by_id(def_ref.def_id)
                .map(|i| i.scheme.clone());
            if let Some(s) = scheme {
                ctx.instantiate(&s)
            } else {
                diagnostics::emit_unbound_constructor(ctx, def_ref.def_id, def_ref.span);
                Ty::Error
            }
        }

        Expr::App(func, arg, span) => {
            let func_ty = infer_expr(ctx, func);
            let arg_ty = infer_expr(ctx, arg);
            let result_ty = ctx.fresh_ty();

            // func : arg_ty -> result_ty
            let expected_func_ty = Ty::fun(arg_ty, result_ty.clone());
            ctx.unify(&func_ty, &expected_func_ty, *span);

            result_ty
        }

        Expr::Lam(pats, body, _span) => {
            // Enter scope for lambda bindings
            ctx.env.push_scope();

            // Infer types for each pattern (left to right)
            let mut arg_types = Vec::new();
            for pat in pats {
                let pat_ty = ctx.infer_pattern(pat);
                arg_types.push(pat_ty);
            }

            // Infer body type
            let body_ty = infer_expr(ctx, body);

            // Exit lambda scope
            ctx.env.pop_scope();

            // Build function type: arg1 -> arg2 -> ... -> body
            arg_types
                .into_iter()
                .rev()
                .fold(body_ty, |acc, arg_ty| Ty::fun(arg_ty, acc))
        }

        Expr::Let(bindings, body, _span) => {
            use bhc_hir::{DefId, Pat};

            // Enter scope for let bindings
            ctx.env.push_scope();

            // For recursive let bindings (Haskell's default), we need to:
            // 1. Pre-register all binding variables with fresh type variables
            // 2. Infer the RHS types (which can now reference other bindings)
            // 3. Unify and generalize

            // Helper to extract (name, def_id) pairs from a pattern
            fn extract_var_ids(pat: &Pat, out: &mut Vec<(Symbol, DefId)>) {
                match pat {
                    Pat::Var(name, def_id, _) => out.push((*name, *def_id)),
                    Pat::As(name, def_id, inner, _) => {
                        out.push((*name, *def_id));
                        extract_var_ids(inner, out);
                    }
                    Pat::Con(_, sub_pats, _) => {
                        for p in sub_pats {
                            extract_var_ids(p, out);
                        }
                    }
                    Pat::Ann(inner, _, _) => extract_var_ids(inner, out),
                    Pat::Or(left, _, _) => extract_var_ids(left, out),
                    Pat::Wild(_) | Pat::Lit(_, _) | Pat::Error(_) => {}
                }
            }

            // Step 1: Pre-register all bound variables with fresh types
            let mut binding_types: Vec<(Vec<(Symbol, DefId)>, Ty)> = Vec::new();
            for binding in bindings {
                let fresh_ty = ctx.fresh_ty();
                let mut var_ids = Vec::new();
                extract_var_ids(&binding.pat, &mut var_ids);

                // Register by both name and DefId (like pattern checking does)
                let scheme = Scheme::mono(fresh_ty.clone());
                for (name, def_id) in &var_ids {
                    ctx.env.insert_local(*name, scheme.clone());
                    ctx.env.insert_global(*def_id, scheme.clone());
                }
                binding_types.push((var_ids, fresh_ty));
            }

            // Step 2: Check each binding's RHS and unify with pre-registered type
            for (binding, (_var_ids, expected_ty)) in bindings.iter().zip(binding_types.iter()) {
                let rhs_ty = infer_expr(ctx, &binding.rhs);

                // If there's a signature, unify with it
                if let Some(sig) = &binding.sig {
                    ctx.unify(&rhs_ty, &sig.ty, binding.span);
                }

                // Unify with the pre-registered type
                ctx.unify(&rhs_ty, expected_ty, binding.span);

                // Check pattern
                ctx.check_pattern(&binding.pat, &rhs_ty);
            }

            // Step 3: Generalize binding types (update the environment)
            // For pattern bindings, check_pattern has already set the correct types.
            // We need to look up each variable's actual type and generalize it,
            // rather than using the whole binding type.
            for (binding, (var_ids, _ty)) in bindings.iter().zip(binding_types.iter()) {
                // For simple variable patterns, the var's type is the binding type
                // For complex patterns (Con, As, etc.), each var has its own type
                for (name, def_id) in var_ids {
                    // Look up the current type for this variable (set by check_pattern)
                    let var_ty = ctx.env.lookup_def_id(*def_id)
                        .map(|s| ctx.apply_subst(&s.ty))
                        .unwrap_or_else(|| ctx.fresh_ty());
                    let scheme = ctx.generalize(&var_ty);
                    ctx.env.insert_local(*name, scheme.clone());
                    ctx.env.insert_global(*def_id, scheme);
                }
                // For simple variable bindings, also add the generalized type annotation if present
                if let Some(sig) = &binding.sig {
                    if var_ids.len() == 1 {
                        let (_name, def_id) = &var_ids[0];
                        ctx.env.insert_global(*def_id, sig.clone());
                    }
                }
            }

            // Infer body type
            let body_ty = infer_expr(ctx, body);

            // Exit let scope
            ctx.env.pop_scope();

            body_ty
        }

        Expr::Case(scrutinee, alts, span) => {
            // Infer scrutinee type
            let scrut_ty = infer_expr(ctx, scrutinee);

            // All alternatives must produce the same type
            let result_ty = ctx.fresh_ty();

            for alt in alts {
                let alt_ty = infer_case_alt(ctx, alt, &scrut_ty);
                ctx.unify(&alt_ty, &result_ty, *span);
            }

            result_ty
        }

        Expr::If(cond, then_branch, else_branch, span) => {
            // Condition must be Bool
            let cond_ty = infer_expr(ctx, cond);
            ctx.unify(&cond_ty, &ctx.builtins.bool_ty.clone(), *span);

            // Both branches must have the same type
            let then_ty = infer_expr(ctx, then_branch);
            let else_ty = infer_expr(ctx, else_branch);
            ctx.unify(&then_ty, &else_ty, *span);

            then_ty
        }

        Expr::Tuple(elems, _span) => {
            let elem_types: Vec<Ty> = elems.iter().map(|e| infer_expr(ctx, e)).collect();
            Ty::Tuple(elem_types)
        }

        Expr::List(elems, span) => {
            let elem_ty = ctx.fresh_ty();

            // All elements must have the same type
            for elem in elems {
                let ty = infer_expr(ctx, elem);
                ctx.unify(&ty, &elem_ty, *span);
            }

            Ty::List(Box::new(elem_ty))
        }

        Expr::Record(con_ref, _fields, span) => {
            // Look up constructor
            let scheme = ctx
                .env
                .lookup_data_con_by_id(con_ref.def_id)
                .map(|i| i.scheme.clone());
            if let Some(s) = scheme {
                let con_ty = ctx.instantiate(&s);
                // For now, just return the result type
                // A full implementation would check field types
                extract_result_type(&con_ty)
            } else {
                diagnostics::emit_unbound_constructor(ctx, con_ref.def_id, *span);
                Ty::Error
            }
        }

        Expr::FieldAccess(record, _field_name, _span) => {
            // Infer record type
            let _record_ty = infer_expr(ctx, record);

            // Field access requires knowing the record type
            // For now, return a fresh variable (full implementation would resolve field)
            // TODO: Resolve field type from record type
            ctx.fresh_ty()
        }

        Expr::RecordUpdate(record, fields, _span) => {
            // Record update produces same type as input record
            let record_ty = infer_expr(ctx, record);

            // Check field types (simplified)
            for field in fields {
                let _field_ty = infer_expr(ctx, &field.value);
                // TODO: Verify field exists and types match
            }

            record_ty
        }

        Expr::Ann(inner, ty, span) => {
            // Type annotation: check inner against declared type
            let inner_ty = infer_expr(ctx, inner);
            ctx.unify(&inner_ty, ty, *span);
            ty.clone()
        }

        Expr::TypeApp(inner, _ty_arg, _span) => {
            // Explicit type application: f @Int
            // If inner has a forall type, instantiate with the argument
            // For now, we just return the inner type
            // TODO: Handle explicit type application properly
            infer_expr(ctx, inner)
        }

        Expr::Error(_) => Ty::Error,
    }
}

/// Infer the type of a literal.
fn infer_lit(ctx: &TyCtxt, lit: &Lit) -> Ty {
    match lit {
        Lit::Int(_) => ctx.builtins.int_ty.clone(),
        Lit::Float(_) => ctx.builtins.float_ty.clone(),
        Lit::Char(_) => ctx.builtins.char_ty.clone(),
        Lit::String(_) => ctx.builtins.string_ty.clone(),
    }
}

/// Check a binding (pattern = expression).
pub fn check_binding(ctx: &mut TyCtxt, binding: &Binding) {
    // Infer RHS type
    let rhs_ty = infer_expr(ctx, &binding.rhs);

    // If there's a signature, unify with it
    if let Some(sig) = &binding.sig {
        ctx.unify(&rhs_ty, &sig.ty, binding.span);
    }

    // Check pattern and bind variables
    ctx.check_pattern(&binding.pat, &rhs_ty);

    // Generalize and add to environment for each bound variable
    for var in binding.pat.bound_vars() {
        let scheme = ctx.generalize(&rhs_ty);
        ctx.env.insert_local(var, scheme);
    }
}

/// Infer the type of a case alternative.
fn infer_case_alt(ctx: &mut TyCtxt, alt: &CaseAlt, scrut_ty: &Ty) -> Ty {
    // Enter scope for pattern bindings
    ctx.env.push_scope();

    // Check pattern against scrutinee type
    ctx.check_pattern(&alt.pat, scrut_ty);

    // Check guards (must all be Bool)
    for guard in &alt.guards {
        let guard_ty = infer_expr(ctx, &guard.cond);
        ctx.unify(&guard_ty, &ctx.builtins.bool_ty.clone(), guard.span);
    }

    // Infer RHS type
    let rhs_ty = infer_expr(ctx, &alt.rhs);

    // Exit scope
    ctx.env.pop_scope();

    rhs_ty
}

/// Extract the result type from a constructor type.
///
/// Given `a -> b -> T a b`, returns `T a b`.
fn extract_result_type(ty: &Ty) -> Ty {
    match ty {
        Ty::Fun(_, result) => extract_result_type(result),
        _ => ty.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_hir::{DefId, Pat};
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::{FileId, Span};

    fn test_context() -> TyCtxt {
        let mut ctx = TyCtxt::new(FileId::new(0));
        ctx.register_builtins();
        ctx
    }

    #[test]
    fn test_infer_int_lit() {
        let mut ctx = test_context();
        let expr = Expr::Lit(Lit::Int(42), Span::DUMMY);

        let ty = infer_expr(&mut ctx, &expr);

        assert_eq!(ty, ctx.builtins.int_ty);
    }

    #[test]
    fn test_infer_tuple() {
        let mut ctx = test_context();
        let expr = Expr::Tuple(
            vec![
                Expr::Lit(Lit::Int(1), Span::DUMMY),
                Expr::Lit(Lit::Char('a'), Span::DUMMY),
            ],
            Span::DUMMY,
        );

        let ty = infer_expr(&mut ctx, &expr);

        match ty {
            Ty::Tuple(elems) => {
                assert_eq!(elems.len(), 2);
                assert_eq!(elems[0], ctx.builtins.int_ty);
                assert_eq!(elems[1], ctx.builtins.char_ty);
            }
            _ => panic!("expected tuple type"),
        }
    }

    #[test]
    fn test_infer_list() {
        let mut ctx = test_context();
        let expr = Expr::List(
            vec![
                Expr::Lit(Lit::Int(1), Span::DUMMY),
                Expr::Lit(Lit::Int(2), Span::DUMMY),
            ],
            Span::DUMMY,
        );

        let ty = infer_expr(&mut ctx, &expr);

        match ty {
            Ty::List(elem) => {
                let elem_ty = ctx.apply_subst(&elem);
                assert_eq!(elem_ty, ctx.builtins.int_ty);
            }
            _ => panic!("expected list type"),
        }
    }

    #[test]
    fn test_infer_lambda() {
        let mut ctx = test_context();
        let x = Symbol::intern("x");
        let expr = Expr::Lam(
            vec![Pat::Var(x, DefId::new(100), Span::DUMMY)],
            Box::new(Expr::Lit(Lit::Int(42), Span::DUMMY)),
            Span::DUMMY,
        );

        let ty = infer_expr(&mut ctx, &expr);

        // Should be a -> Int for some a
        match ty {
            Ty::Fun(from, to) => {
                assert!(matches!(*from, Ty::Var(_)));
                assert_eq!(*to, ctx.builtins.int_ty);
            }
            _ => panic!("expected function type"),
        }
    }

    #[test]
    fn test_infer_if() {
        let mut ctx = test_context();
        let expr = Expr::If(
            Box::new(Expr::Lit(Lit::Int(1), Span::DUMMY)), // Wrong, but tests unification
            Box::new(Expr::Lit(Lit::Int(1), Span::DUMMY)),
            Box::new(Expr::Lit(Lit::Int(2), Span::DUMMY)),
            Span::DUMMY,
        );

        let ty = infer_expr(&mut ctx, &expr);

        // Should have error because condition is Int, not Bool
        assert!(ctx.has_errors());
        // But result type should still be Int
        assert_eq!(ty, ctx.builtins.int_ty);
    }
}
