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
//!
//! ## M9 Dependent Types
//!
//! This module extends unification to handle type-level naturals (`TyNat`)
//! and type-level lists (`TyList`) for shape-indexed tensors.
//!
//! ## Type Aliases
//!
//! String is a type alias for [Char] in Haskell, so we handle this
//! equivalence during unification.

use bhc_span::Span;
use bhc_types::{Ty, TyCon, TyList, TyNat, TyVar};

use crate::context::TyCtxt;
use crate::diagnostics;
use crate::nat_solver::{NatConstraint, NatSolver};

/// Check if a type is the String type constructor.
fn is_string_con(ty: &Ty) -> bool {
    matches!(ty, Ty::Con(c) if c.name.as_str() == "String")
}

/// Check if a type is the Dimension type constructor (XMonad type alias).
fn is_dimension_con(ty: &Ty) -> bool {
    matches!(ty, Ty::Con(c) if c.name.as_str() == "Dimension")
}

/// Check if a type is the WorkspaceId type constructor (XMonad type alias).
fn is_workspaceid_con(ty: &Ty) -> bool {
    matches!(ty, Ty::Con(c) if c.name.as_str() == "WorkspaceId")
}

/// Check if a type is the D type constructor (XMonad type alias for (Int, Int)).
fn is_d_con(ty: &Ty) -> bool {
    matches!(ty, Ty::Con(c) if c.name.as_str() == "D")
}

/// Check if a type is the Position type constructor (XMonad type alias for Int).
fn is_position_con(ty: &Ty) -> bool {
    matches!(ty, Ty::Con(c) if c.name.as_str() == "Position")
}

/// Get the element type of a list, if this is a list type.
fn get_list_elem(ty: &Ty) -> Option<&Ty> {
    match ty {
        Ty::List(elem) => Some(elem.as_ref()),
        _ => None,
    }
}

/// Get the Char type constructor.
fn char_ty() -> Ty {
    use bhc_intern::Symbol;
    use bhc_types::Kind;
    Ty::Con(TyCon::new(Symbol::intern("Char"), Kind::Star))
}

/// Get the Int type constructor.
fn int_ty() -> Ty {
    use bhc_intern::Symbol;
    use bhc_types::Kind;
    Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star))
}

/// Check if two type constructor names are compatible type aliases.
/// Returns true if they both resolve to the same underlying type.
fn are_compatible_type_aliases(name1: &str, name2: &str) -> bool {
    // Type aliases for Int
    const INT_ALIASES: &[&str] = &["Int", "Dimension", "Position", "KeyMask", "Window", "ScreenId"];

    // Type aliases for String
    const STRING_ALIASES: &[&str] = &["String", "WorkspaceId"];

    // Check if both are Int aliases
    let is_int1 = INT_ALIASES.contains(&name1);
    let is_int2 = INT_ALIASES.contains(&name2);
    if is_int1 && is_int2 {
        return true;
    }

    // Check if both are String aliases
    let is_str1 = STRING_ALIASES.contains(&name1);
    let is_str2 = STRING_ALIASES.contains(&name2);
    if is_str1 && is_str2 {
        return true;
    }

    false
}

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

        // Type constructors: must have same name OR be compatible type aliases
        (Ty::Con(c1), Ty::Con(c2)) => {
            if c1.name == c2.name {
                // Same type, OK
            } else if are_compatible_type_aliases(c1.name.as_str(), c2.name.as_str()) {
                // Type aliases that should be equivalent
            } else {
                diagnostics::emit_type_mismatch(ctx, t1, t2, span);
            }
        }

        // Primitive types: must be the same
        (Ty::Prim(p1), Ty::Prim(p2)) => {
            if p1 != p2 {
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

        // === M9 Dependent Types: Type-level naturals ===
        (Ty::Nat(n1), Ty::Nat(n2)) => {
            unify_nat(ctx, n1, n2, span);
        }

        // === M9 Dependent Types: Type-level lists ===
        (Ty::TyList(l1), Ty::TyList(l2)) => {
            unify_ty_list(ctx, l1, l2, span);
        }

        // === Type alias: String = [Char] ===
        // In Haskell, String is defined as `type String = [Char]`, so these
        // types should unify. Handle both directions.
        // When unifying String with [a], unify a with Char.
        (t1, t2) if is_string_con(t1) => {
            if let Some(elem) = get_list_elem(t2) {
                // String ~ [elem] => unify elem with Char
                let char = char_ty();
                unify_inner(ctx, elem, &char, span);
            } else if is_string_con(t2) {
                // String ~ String, ok
            } else if is_workspaceid_con(t2) {
                // String ~ WorkspaceId, ok (WorkspaceId = String)
            } else {
                diagnostics::emit_type_mismatch(ctx, t1, t2, span);
            }
        }
        (t1, t2) if is_string_con(t2) => {
            if let Some(elem) = get_list_elem(t1) {
                // [elem] ~ String => unify elem with Char
                let char = char_ty();
                unify_inner(ctx, elem, &char, span);
            } else if is_workspaceid_con(t1) {
                // WorkspaceId ~ String, ok (WorkspaceId = String)
            } else {
                diagnostics::emit_type_mismatch(ctx, t1, t2, span);
            }
        }

        // === Type alias: Dimension = Int (XMonad) ===
        (t1, t2) if is_dimension_con(t1) => {
            let int = int_ty();
            if is_dimension_con(t2) {
                // Dimension ~ Dimension, ok
            } else if is_position_con(t2) {
                // Dimension ~ Position, both are Int aliases
            } else {
                unify_inner(ctx, &int, t2, span);
            }
        }
        (t1, t2) if is_dimension_con(t2) => {
            let int = int_ty();
            if is_position_con(t1) {
                // Position ~ Dimension, both are Int aliases
            } else {
                unify_inner(ctx, t1, &int, span);
            }
        }

        // === Type alias: Position = Int (XMonad) ===
        (t1, t2) if is_position_con(t1) => {
            let int = int_ty();
            if is_position_con(t2) {
                // Position ~ Position, ok
            } else {
                unify_inner(ctx, &int, t2, span);
            }
        }
        (t1, t2) if is_position_con(t2) => {
            let int = int_ty();
            unify_inner(ctx, t1, &int, span);
        }

        // === Type alias: WorkspaceId = String (XMonad) ===
        (t1, t2) if is_workspaceid_con(t1) => {
            if is_workspaceid_con(t2) {
                // WorkspaceId ~ WorkspaceId, ok
            } else if let Some(elem) = get_list_elem(t2) {
                // WorkspaceId ~ [elem] => unify elem with Char (WorkspaceId = String = [Char])
                let char = char_ty();
                unify_inner(ctx, elem, &char, span);
            } else {
                diagnostics::emit_type_mismatch(ctx, t1, t2, span);
            }
        }
        (t1, t2) if is_workspaceid_con(t2) => {
            if let Some(elem) = get_list_elem(t1) {
                // [elem] ~ WorkspaceId => unify elem with Char
                let char = char_ty();
                unify_inner(ctx, elem, &char, span);
            } else {
                diagnostics::emit_type_mismatch(ctx, t1, t2, span);
            }
        }

        // === Type alias: D = (Int, Int) (XMonad) ===
        (t1, t2) if is_d_con(t1) => {
            let int = int_ty();
            if is_d_con(t2) {
                // D ~ D, ok
            } else if let Ty::Tuple(elems) = t2 {
                if elems.len() == 2 {
                    unify_inner(ctx, &int, &elems[0], span);
                    unify_inner(ctx, &int, &elems[1], span);
                } else {
                    diagnostics::emit_type_mismatch(ctx, t1, t2, span);
                }
            } else {
                diagnostics::emit_type_mismatch(ctx, t1, t2, span);
            }
        }
        (t1, t2) if is_d_con(t2) => {
            let int = int_ty();
            if let Ty::Tuple(elems) = t1 {
                if elems.len() == 2 {
                    unify_inner(ctx, &elems[0], &int, span);
                    unify_inner(ctx, &elems[1], &int, span);
                } else {
                    diagnostics::emit_type_mismatch(ctx, t1, t2, span);
                }
            } else {
                diagnostics::emit_type_mismatch(ctx, t1, t2, span);
            }
        }

        // Different type structures: mismatch
        _ => {
            diagnostics::emit_type_mismatch(ctx, t1, t2, span);
        }
    }
}

// === M9 Dependent Types: Type-level natural unification ===

/// Unify two type-level naturals.
///
/// This uses the `NatSolver` for complex arithmetic constraints that
/// can't be solved by simple structural unification.
fn unify_nat(ctx: &mut TyCtxt, n1: &TyNat, n2: &TyNat, span: Span) {
    match (n1, n2) {
        // Same literals unify trivially
        (TyNat::Lit(v1), TyNat::Lit(v2)) => {
            if v1 != v2 {
                diagnostics::emit_dimension_mismatch(ctx, *v1, *v2, span);
            }
        }

        // Same variable unifies trivially
        (TyNat::Var(v1), TyNat::Var(v2)) if v1.id == v2.id => {}

        // Variable on either side: bind it
        (TyNat::Var(v), n) | (n, TyNat::Var(v)) => {
            bind_nat_var(ctx, v, n, span);
        }

        // Addition: try to solve or record constraint
        (TyNat::Add(a1, b1), TyNat::Add(a2, b2)) => {
            // Structurally unify the components
            unify_nat(ctx, a1, a2, span);
            let b1_applied = ctx.subst.apply_nat(b1);
            let b2_applied = ctx.subst.apply_nat(b2);
            unify_nat(ctx, &b1_applied, &b2_applied, span);
        }

        // Multiplication: try to solve or record constraint
        (TyNat::Mul(a1, b1), TyNat::Mul(a2, b2)) => {
            // Structurally unify the components
            unify_nat(ctx, a1, a2, span);
            let b1_applied = ctx.subst.apply_nat(b1);
            let b2_applied = ctx.subst.apply_nat(b2);
            unify_nat(ctx, &b1_applied, &b2_applied, span);
        }

        // For complex cases, use the constraint solver
        _ => {
            // First try to evaluate to ground values
            if let (Some(v1), Some(v2)) = (n1.eval(), n2.eval()) {
                if v1 != v2 {
                    diagnostics::emit_dimension_mismatch(ctx, v1, v2, span);
                }
                return;
            }

            // Use NatSolver for complex arithmetic constraints
            // This handles cases like: a + 3 = 10 => a = 7
            let mut solver = NatSolver::new();
            solver.add_constraint(NatConstraint::Equal(n1.clone(), n2.clone()));

            match solver.solve() {
                Ok(subst) => {
                    // Apply the solved substitution to our type context
                    for (var_id, nat) in subst.iter() {
                        // Create a TyVar for this binding
                        let ty_var = TyVar::new(var_id, bhc_types::Kind::Nat);
                        ctx.subst.insert(&ty_var, Ty::Nat(nat.clone()));
                    }
                }
                Err(err) => {
                    // Solver couldn't find a solution
                    diagnostics::emit_nat_solver_error(ctx, &err, span);
                }
            }
        }
    }
}

/// Bind a type-level natural variable to a value.
fn bind_nat_var(ctx: &mut TyCtxt, var: &TyVar, n: &TyNat, span: Span) {
    // Skip if binding to self
    if let TyNat::Var(v) = n {
        if v.id == var.id {
            return;
        }
    }

    // Occurs check for naturals
    if occurs_check_nat(var, n) {
        diagnostics::emit_nat_occurs_check_error(ctx, var, n, span);
        return;
    }

    // Add binding to substitution (wrap in Ty::Nat)
    ctx.subst.insert(var, Ty::Nat(n.clone()));
}

/// Check if a type variable occurs in a type-level natural.
fn occurs_check_nat(var: &TyVar, n: &TyNat) -> bool {
    match n {
        TyNat::Lit(_) => false,
        TyNat::Var(v) => v.id == var.id,
        TyNat::Add(a, b) | TyNat::Mul(a, b) => {
            occurs_check_nat(var, a) || occurs_check_nat(var, b)
        }
    }
}

// === M9 Dependent Types: Type-level list unification ===

/// Unify two type-level lists.
fn unify_ty_list(ctx: &mut TyCtxt, l1: &TyList, l2: &TyList, span: Span) {
    match (l1, l2) {
        // Both empty: unify trivially
        (TyList::Nil, TyList::Nil) => {}

        // Same variable unifies trivially
        (TyList::Var(v1), TyList::Var(v2)) if v1.id == v2.id => {}

        // Variable on either side: bind it
        (TyList::Var(v), l) | (l, TyList::Var(v)) => {
            bind_ty_list_var(ctx, v, l, span);
        }

        // Both cons: unify heads and tails
        (TyList::Cons(h1, t1), TyList::Cons(h2, t2)) => {
            // Unify heads
            let h1_applied = ctx.apply_subst(h1);
            let h2_applied = ctx.apply_subst(h2);
            unify_inner(ctx, &h1_applied, &h2_applied, span);

            // Unify tails
            let t1_applied = ctx.subst.apply_ty_list(t1);
            let t2_applied = ctx.subst.apply_ty_list(t2);
            unify_ty_list(ctx, &t1_applied, &t2_applied, span);
        }

        // Length mismatch: one is empty, other is not
        (TyList::Nil, TyList::Cons(_, _)) | (TyList::Cons(_, _), TyList::Nil) => {
            diagnostics::emit_shape_length_mismatch(ctx, l1, l2, span);
        }

        // Append: handle structurally or defer
        (TyList::Append(xs1, ys1), TyList::Append(xs2, ys2)) => {
            // Try structural unification
            let xs1_applied = ctx.subst.apply_ty_list(xs1);
            let xs2_applied = ctx.subst.apply_ty_list(xs2);
            unify_ty_list(ctx, &xs1_applied, &xs2_applied, span);

            let ys1_applied = ctx.subst.apply_ty_list(ys1);
            let ys2_applied = ctx.subst.apply_ty_list(ys2);
            unify_ty_list(ctx, &ys1_applied, &ys2_applied, span);
        }

        // Other cases: cannot unify
        _ => {
            diagnostics::emit_shape_mismatch(ctx, l1, l2, span);
        }
    }
}

/// Bind a type-level list variable to a value.
fn bind_ty_list_var(ctx: &mut TyCtxt, var: &TyVar, l: &TyList, span: Span) {
    // Skip if binding to self
    if let TyList::Var(v) = l {
        if v.id == var.id {
            return;
        }
    }

    // Occurs check for lists
    if occurs_check_ty_list(var, l) {
        diagnostics::emit_ty_list_occurs_check_error(ctx, var, l, span);
        return;
    }

    // Add binding to substitution (wrap in Ty::TyList)
    ctx.subst.insert(var, Ty::TyList(l.clone()));
}

/// Check if a type variable occurs in a type-level list.
fn occurs_check_ty_list(var: &TyVar, l: &TyList) -> bool {
    match l {
        TyList::Nil => false,
        TyList::Var(v) => v.id == var.id,
        TyList::Cons(head, tail) => {
            occurs_check(var, head) || occurs_check_ty_list(var, tail)
        }
        TyList::Append(xs, ys) => {
            occurs_check_ty_list(var, xs) || occurs_check_ty_list(var, ys)
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
    ctx.subst.insert(var, ty.clone());
}

/// Check if a type variable occurs in a type (prevents infinite types).
fn occurs_check(var: &TyVar, ty: &Ty) -> bool {
    match ty {
        Ty::Var(v) => v.id == var.id,
        Ty::Con(_) | Ty::Prim(_) | Ty::Error => false,
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
        // M9: Check in type-level naturals and lists
        Ty::Nat(n) => occurs_check_nat(var, n),
        Ty::TyList(l) => occurs_check_ty_list(var, l),
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

    // === M9 Dependent Types: Nat unification tests ===

    #[test]
    fn test_unify_nat_literals() {
        let mut ctx = test_context();
        let n1 = Ty::Nat(TyNat::Lit(1024));
        let n2 = Ty::Nat(TyNat::Lit(1024));

        unify(&mut ctx, &n1, &n2, Span::DUMMY);

        assert!(!ctx.has_errors());
    }

    #[test]
    fn test_unify_nat_literal_mismatch() {
        let mut ctx = test_context();
        let n1 = Ty::Nat(TyNat::Lit(1024));
        let n2 = Ty::Nat(TyNat::Lit(768));

        unify(&mut ctx, &n1, &n2, Span::DUMMY);

        assert!(ctx.has_errors());
    }

    #[test]
    fn test_unify_nat_var_to_literal() {
        let mut ctx = test_context();
        let m = TyVar::new(1, Kind::Nat);

        let t1 = Ty::Nat(TyNat::Var(m.clone()));
        let t2 = Ty::Nat(TyNat::Lit(256));

        unify(&mut ctx, &t1, &t2, Span::DUMMY);

        assert!(!ctx.has_errors());
        let result = ctx.apply_subst(&Ty::Nat(TyNat::Var(m)));
        assert_eq!(result, Ty::Nat(TyNat::Lit(256)));
    }

    #[test]
    fn test_unify_nat_var_to_var() {
        let mut ctx = test_context();
        let m = TyVar::new(1, Kind::Nat);
        let n = TyVar::new(2, Kind::Nat);

        let t1 = Ty::Nat(TyNat::Var(m.clone()));
        let t2 = Ty::Nat(TyNat::Var(n.clone()));

        unify(&mut ctx, &t1, &t2, Span::DUMMY);

        assert!(!ctx.has_errors());
        // After unification, both should resolve to the same thing
        let r1 = ctx.apply_subst(&Ty::Nat(TyNat::Var(m.clone())));
        let r2 = ctx.apply_subst(&Ty::Nat(TyNat::Var(n.clone())));
        // One should be bound to the other
        assert!(r1 == Ty::Nat(TyNat::Var(n)) || r2 == Ty::Nat(TyNat::Var(m)));
    }

    #[test]
    fn test_unify_nat_add_with_solver() {
        let mut ctx = test_context();
        let a = TyVar::new(1, Kind::Nat);

        // a + 3 = 10, so a should be 7
        let t1 = Ty::Nat(TyNat::add(TyNat::Var(a.clone()), TyNat::Lit(3)));
        let t2 = Ty::Nat(TyNat::Lit(10));

        unify(&mut ctx, &t1, &t2, Span::DUMMY);

        assert!(!ctx.has_errors());
        let result = ctx.apply_subst(&Ty::Nat(TyNat::Var(a)));
        assert_eq!(result, Ty::Nat(TyNat::Lit(7)));
    }

    #[test]
    fn test_unify_nat_mul_with_solver() {
        let mut ctx = test_context();
        let a = TyVar::new(1, Kind::Nat);

        // a * 4 = 12, so a should be 3
        let t1 = Ty::Nat(TyNat::mul(TyNat::Var(a.clone()), TyNat::Lit(4)));
        let t2 = Ty::Nat(TyNat::Lit(12));

        unify(&mut ctx, &t1, &t2, Span::DUMMY);

        assert!(!ctx.has_errors());
        let result = ctx.apply_subst(&Ty::Nat(TyNat::Var(a)));
        assert_eq!(result, Ty::Nat(TyNat::Lit(3)));
    }

    #[test]
    fn test_unify_nat_mul_not_divisible() {
        let mut ctx = test_context();
        let a = TyVar::new(1, Kind::Nat);

        // a * 4 = 10 has no integer solution
        let t1 = Ty::Nat(TyNat::mul(TyNat::Var(a.clone()), TyNat::Lit(4)));
        let t2 = Ty::Nat(TyNat::Lit(10));

        unify(&mut ctx, &t1, &t2, Span::DUMMY);

        assert!(ctx.has_errors());
    }

    #[test]
    fn test_unify_ty_list_nil() {
        let mut ctx = test_context();
        let l1 = Ty::TyList(TyList::Nil);
        let l2 = Ty::TyList(TyList::Nil);

        unify(&mut ctx, &l1, &l2, Span::DUMMY);

        assert!(!ctx.has_errors());
    }

    #[test]
    fn test_unify_ty_list_cons() {
        let mut ctx = test_context();
        let shape1 = TyList::from_vec(vec![Ty::Nat(TyNat::Lit(1024)), Ty::Nat(TyNat::Lit(768))]);
        let shape2 = TyList::from_vec(vec![Ty::Nat(TyNat::Lit(1024)), Ty::Nat(TyNat::Lit(768))]);

        unify(&mut ctx, &Ty::TyList(shape1), &Ty::TyList(shape2), Span::DUMMY);

        assert!(!ctx.has_errors());
    }

    #[test]
    fn test_unify_ty_list_length_mismatch() {
        let mut ctx = test_context();
        let shape1 = TyList::from_vec(vec![Ty::Nat(TyNat::Lit(1024))]);
        let shape2 = TyList::from_vec(vec![Ty::Nat(TyNat::Lit(1024)), Ty::Nat(TyNat::Lit(768))]);

        unify(&mut ctx, &Ty::TyList(shape1), &Ty::TyList(shape2), Span::DUMMY);

        assert!(ctx.has_errors());
    }

    #[test]
    fn test_unify_ty_list_with_var() {
        let mut ctx = test_context();
        let m = TyVar::new(1, Kind::Nat);
        let shape1 = TyList::from_vec(vec![Ty::Nat(TyNat::Var(m.clone())), Ty::Nat(TyNat::Lit(768))]);
        let shape2 = TyList::from_vec(vec![Ty::Nat(TyNat::Lit(1024)), Ty::Nat(TyNat::Lit(768))]);

        unify(&mut ctx, &Ty::TyList(shape1), &Ty::TyList(shape2), Span::DUMMY);

        assert!(!ctx.has_errors());
        let result = ctx.apply_subst(&Ty::Nat(TyNat::Var(m)));
        assert_eq!(result, Ty::Nat(TyNat::Lit(1024)));
    }
}
