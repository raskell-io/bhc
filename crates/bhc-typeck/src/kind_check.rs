//! Kind checking for types.
//!
//! This module implements kind checking to ensure that types are well-formed.
//! In particular, it verifies that type constructors are applied to the
//! correct number of arguments with appropriate kinds.
//!
//! ## M9 Dependent Types
//!
//! Kind checking is extended to handle:
//! - `Nat` kind for type-level natural numbers
//! - `[Nat]` kind for tensor shapes
//! - The `Tensor :: [Nat] -> * -> *` type constructor

use bhc_span::Span;
use bhc_types::{Kind, Ty, TyCon, TyList, TyNat, TyVar};
use rustc_hash::FxHashMap;

use crate::context::TyCtxt;
use crate::diagnostics;

/// Kind environment mapping type variables to their kinds.
#[derive(Debug, Default, Clone)]
pub struct KindEnv {
    /// Mapping from type variable IDs to their kinds.
    vars: FxHashMap<u32, Kind>,
    /// Mapping from type constructor names to their kinds.
    cons: FxHashMap<bhc_intern::Symbol, Kind>,
}

impl KindEnv {
    /// Creates a new empty kind environment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a type variable binding.
    pub fn bind_var(&mut self, var: &TyVar, kind: Kind) {
        self.vars.insert(var.id, kind);
    }

    /// Looks up the kind of a type variable.
    #[must_use]
    pub fn lookup_var(&self, var: &TyVar) -> Option<&Kind> {
        self.vars.get(&var.id)
    }

    /// Registers a type constructor with its kind.
    pub fn register_con(&mut self, con: &TyCon) {
        self.cons.insert(con.name, con.kind.clone());
    }

    /// Looks up the kind of a type constructor.
    #[must_use]
    pub fn lookup_con(&self, name: bhc_intern::Symbol) -> Option<&Kind> {
        self.cons.get(&name)
    }
}

/// Infer the kind of a type.
///
/// Returns the kind of the type, or emits a diagnostic if the type is ill-kinded.
pub fn infer_kind(ctx: &mut TyCtxt, env: &KindEnv, ty: &Ty, span: Span) -> Kind {
    match ty {
        Ty::Var(v) => {
            // Look up the variable's kind in the environment or use its declared kind
            env.lookup_var(v).cloned().unwrap_or_else(|| v.kind.clone())
        }

        Ty::Con(c) => {
            // Type constructor kind from environment or declaration
            env.lookup_con(c.name)
                .cloned()
                .unwrap_or_else(|| c.kind.clone())
        }

        Ty::Prim(_) => {
            // Primitive types have kind *
            Kind::Star
        }

        Ty::App(fun, arg) => {
            // Infer kinds of function and argument
            let fun_kind = infer_kind(ctx, env, fun, span);
            let arg_kind = infer_kind(ctx, env, arg, span);

            // Function must have arrow kind
            match fun_kind {
                Kind::Arrow(expected_arg, result) => {
                    if !kinds_unify(&expected_arg, &arg_kind) {
                        diagnostics::emit_kind_mismatch(
                            ctx,
                            &format!("{expected_arg:?}"),
                            &format!("{arg_kind:?}"),
                            span,
                        );
                    }
                    *result
                }
                _ => {
                    diagnostics::emit_kind_mismatch(
                        ctx,
                        "arrow kind",
                        &format!("{fun_kind:?}"),
                        span,
                    );
                    Kind::Star
                }
            }
        }

        Ty::Fun(_, _) => {
            // Function types have kind *
            // (We could also check that from and to have kind *)
            Kind::Star
        }

        Ty::Tuple(_) => {
            // Tuple types have kind *
            Kind::Star
        }

        Ty::List(_) => {
            // Value-level list types have kind *
            Kind::Star
        }

        Ty::Forall(vars, body) => {
            // Extend environment with bound variables
            let mut inner_env = env.clone();
            for v in vars {
                inner_env.bind_var(v, v.kind.clone());
            }
            // Kind of forall is the kind of the body (usually *)
            infer_kind(ctx, &inner_env, body, span)
        }

        Ty::Error => Kind::Star,

        // === M9 Dependent Types ===
        Ty::Nat(n) => {
            // Type-level naturals have kind Nat
            check_nat_kind(ctx, env, n, span);
            Kind::Nat
        }

        Ty::TyList(l) => {
            // Type-level lists - infer element kind
            let elem_kind = infer_ty_list_kind(ctx, env, l, span);
            Kind::List(Box::new(elem_kind))
        }
    }
}

/// Check that a type-level natural is well-kinded.
fn check_nat_kind(ctx: &mut TyCtxt, env: &KindEnv, n: &TyNat, span: Span) {
    match n {
        TyNat::Lit(_) => {
            // Literals are always well-kinded
        }
        TyNat::Var(v) => {
            // Variable must have kind Nat
            let var_kind = env.lookup_var(v).cloned().unwrap_or_else(|| v.kind.clone());
            if !matches!(var_kind, Kind::Nat) {
                diagnostics::emit_kind_mismatch(ctx, "Nat", &format!("{var_kind:?}"), span);
            }
        }
        TyNat::Add(a, b) | TyNat::Mul(a, b) => {
            // Both operands must have kind Nat
            check_nat_kind(ctx, env, a, span);
            check_nat_kind(ctx, env, b, span);
        }
    }
}

/// Infer the element kind of a type-level list.
fn infer_ty_list_kind(ctx: &mut TyCtxt, env: &KindEnv, l: &TyList, span: Span) -> Kind {
    match l {
        TyList::Nil => {
            // Empty list could have any element kind, default to Nat for shapes
            Kind::Nat
        }
        TyList::Var(v) => {
            // Variable must have kind [k] for some k
            let var_kind = env.lookup_var(v).cloned().unwrap_or_else(|| v.kind.clone());
            match var_kind {
                Kind::List(elem) => *elem,
                _ => {
                    diagnostics::emit_kind_mismatch(ctx, "[k]", &format!("{var_kind:?}"), span);
                    Kind::Nat
                }
            }
        }
        TyList::Cons(head, tail) => {
            // Infer kind of head and check tail has same element kind
            let head_kind = infer_kind(ctx, env, head, span);
            let tail_elem_kind = infer_ty_list_kind(ctx, env, tail, span);

            if !kinds_unify(&head_kind, &tail_elem_kind) {
                diagnostics::emit_kind_mismatch(
                    ctx,
                    &format!("{head_kind:?}"),
                    &format!("{tail_elem_kind:?}"),
                    span,
                );
            }

            head_kind
        }
        TyList::Append(xs, ys) => {
            // Both lists must have the same element kind
            let xs_kind = infer_ty_list_kind(ctx, env, xs, span);
            let ys_kind = infer_ty_list_kind(ctx, env, ys, span);

            if !kinds_unify(&xs_kind, &ys_kind) {
                diagnostics::emit_kind_mismatch(
                    ctx,
                    &format!("{xs_kind:?}"),
                    &format!("{ys_kind:?}"),
                    span,
                );
            }

            xs_kind
        }
    }
}

/// Check if two kinds unify.
fn kinds_unify(k1: &Kind, k2: &Kind) -> bool {
    match (k1, k2) {
        (Kind::Star, Kind::Star) => true,
        (Kind::Constraint, Kind::Constraint) => true,
        (Kind::Nat, Kind::Nat) => true,
        (Kind::Arrow(a1, r1), Kind::Arrow(a2, r2)) => kinds_unify(a1, a2) && kinds_unify(r1, r2),
        (Kind::List(e1), Kind::List(e2)) => kinds_unify(e1, e2),
        (Kind::Var(_), _) | (_, Kind::Var(_)) => {
            // For now, variables unify with anything (simple approach)
            true
        }
        _ => false,
    }
}

/// Check that a type has kind *.
pub fn check_star_kind(ctx: &mut TyCtxt, env: &KindEnv, ty: &Ty, span: Span) {
    let kind = infer_kind(ctx, env, ty, span);
    if !matches!(kind, Kind::Star) {
        diagnostics::emit_kind_mismatch(ctx, "*", &format!("{kind:?}"), span);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_intern::Symbol;
    use bhc_span::FileId;

    fn test_context() -> TyCtxt {
        TyCtxt::new(FileId::new(0))
    }

    #[test]
    fn test_var_kind() {
        let mut ctx = test_context();
        let env = KindEnv::new();
        let var = TyVar::new_star(0);

        let kind = infer_kind(&mut ctx, &env, &Ty::Var(var), Span::DUMMY);
        assert_eq!(kind, Kind::Star);
    }

    #[test]
    fn test_nat_kind() {
        let mut ctx = test_context();
        let env = KindEnv::new();
        let n = Ty::Nat(TyNat::lit(42));

        let kind = infer_kind(&mut ctx, &env, &n, Span::DUMMY);
        assert_eq!(kind, Kind::Nat);
    }

    #[test]
    fn test_ty_list_kind() {
        let mut ctx = test_context();
        let env = KindEnv::new();
        let shape = Ty::TyList(TyList::shape_from_dims(&[1024, 768]));

        let kind = infer_kind(&mut ctx, &env, &shape, Span::DUMMY);
        assert_eq!(kind, Kind::nat_list());
    }

    #[test]
    fn test_app_kind() {
        let mut ctx = test_context();
        let env = KindEnv::new();

        // Maybe :: * -> *
        let maybe = TyCon::new(Symbol::intern("Maybe"), Kind::star_to_star());
        let int = TyCon::new(Symbol::intern("Int"), Kind::Star);

        // Maybe Int :: *
        let ty = Ty::App(Box::new(Ty::Con(maybe)), Box::new(Ty::Con(int)));
        let kind = infer_kind(&mut ctx, &env, &ty, Span::DUMMY);

        assert_eq!(kind, Kind::Star);
    }

    #[test]
    fn test_kinds_unify() {
        assert!(kinds_unify(&Kind::Star, &Kind::Star));
        assert!(kinds_unify(&Kind::Nat, &Kind::Nat));
        assert!(kinds_unify(&Kind::nat_list(), &Kind::nat_list()));
        assert!(!kinds_unify(&Kind::Star, &Kind::Nat));
    }
}
