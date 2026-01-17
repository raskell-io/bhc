//! Built-in types and data constructors.
//!
//! This module defines the primitive types that are always available
//! in BHC programs: `Int`, `Float`, `Char`, `Bool`, `String`, etc.
//!
//! These types are registered into the type environment before
//! type checking user code.

use bhc_hir::DefId;
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_types::{Kind, Scheme, Ty, TyCon, TyVar};

use crate::env::TypeEnv;

/// Built-in types and their type constructors.
#[derive(Debug, Clone)]
pub struct Builtins {
    // Type constructors
    /// The `Int` type constructor.
    pub int_con: TyCon,
    /// The `Float` type constructor.
    pub float_con: TyCon,
    /// The `Char` type constructor.
    pub char_con: TyCon,
    /// The `Bool` type constructor.
    pub bool_con: TyCon,
    /// The `String` type constructor.
    pub string_con: TyCon,
    /// The `[]` (list) type constructor.
    pub list_con: TyCon,
    /// The `Maybe` type constructor.
    pub maybe_con: TyCon,
    /// The `Either` type constructor.
    pub either_con: TyCon,
    /// The `IO` type constructor.
    pub io_con: TyCon,

    // Convenient type values
    /// The `Int` type.
    pub int_ty: Ty,
    /// The `Float` type.
    pub float_ty: Ty,
    /// The `Char` type.
    pub char_ty: Ty,
    /// The `Bool` type.
    pub bool_ty: Ty,
    /// The `String` type.
    pub string_ty: Ty,
}

impl Default for Builtins {
    fn default() -> Self {
        Self::new()
    }
}

impl Builtins {
    /// Create the built-in types.
    #[must_use]
    pub fn new() -> Self {
        // Type constructors with kind *
        let int_con = TyCon::new(Symbol::intern("Int"), Kind::Star);
        let float_con = TyCon::new(Symbol::intern("Float"), Kind::Star);
        let char_con = TyCon::new(Symbol::intern("Char"), Kind::Star);
        let bool_con = TyCon::new(Symbol::intern("Bool"), Kind::Star);
        let string_con = TyCon::new(Symbol::intern("String"), Kind::Star);

        // Type constructors with kind * -> *
        let list_con = TyCon::new(Symbol::intern("[]"), Kind::star_to_star());
        let maybe_con = TyCon::new(Symbol::intern("Maybe"), Kind::star_to_star());
        let io_con = TyCon::new(Symbol::intern("IO"), Kind::star_to_star());

        // Type constructors with kind * -> * -> *
        let either_kind = Kind::Arrow(
            Box::new(Kind::Star),
            Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
        );
        let either_con = TyCon::new(Symbol::intern("Either"), either_kind);

        // Convenient types
        let int_ty = Ty::Con(int_con.clone());
        let float_ty = Ty::Con(float_con.clone());
        let char_ty = Ty::Con(char_con.clone());
        let bool_ty = Ty::Con(bool_con.clone());
        let string_ty = Ty::Con(string_con.clone());

        Self {
            int_con,
            float_con,
            char_con,
            bool_con,
            string_con,
            list_con,
            maybe_con,
            either_con,
            io_con,
            int_ty,
            float_ty,
            char_ty,
            bool_ty,
            string_ty,
        }
    }

    /// Register built-in data constructors in the environment.
    pub fn register_data_cons(&self, env: &mut TypeEnv) {
        // Bool constructors
        // True :: Bool
        // False :: Bool
        let true_id = DefId::new(BUILTIN_TRUE_ID);
        let false_id = DefId::new(BUILTIN_FALSE_ID);
        env.register_data_con(
            true_id,
            Symbol::intern("True"),
            Scheme::mono(self.bool_ty.clone()),
        );
        env.register_data_con(
            false_id,
            Symbol::intern("False"),
            Scheme::mono(self.bool_ty.clone()),
        );

        // Maybe constructors
        // Nothing :: forall a. Maybe a
        // Just :: forall a. a -> Maybe a
        let a = TyVar::new_star(BUILTIN_TYVAR_A);
        let maybe_a = Ty::App(
            Box::new(Ty::Con(self.maybe_con.clone())),
            Box::new(Ty::Var(a.clone())),
        );

        let nothing_id = DefId::new(BUILTIN_NOTHING_ID);
        let just_id = DefId::new(BUILTIN_JUST_ID);
        env.register_data_con(
            nothing_id,
            Symbol::intern("Nothing"),
            Scheme::poly(vec![a.clone()], maybe_a.clone()),
        );
        env.register_data_con(
            just_id,
            Symbol::intern("Just"),
            Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), maybe_a)),
        );

        // List constructors
        // [] :: forall a. [a]
        // (:) :: forall a. a -> [a] -> [a]
        let list_a = Ty::List(Box::new(Ty::Var(a.clone())));

        let nil_id = DefId::new(BUILTIN_NIL_ID);
        let cons_id = DefId::new(BUILTIN_CONS_ID);
        env.register_data_con(
            nil_id,
            Symbol::intern("[]"),
            Scheme::poly(vec![a.clone()], list_a.clone()),
        );
        env.register_data_con(
            cons_id,
            Symbol::intern(":"),
            Scheme::poly(
                vec![a.clone()],
                Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
            ),
        );

        // Either constructors
        // Left :: forall a b. a -> Either a b
        // Right :: forall a b. b -> Either a b
        let b = TyVar::new_star(BUILTIN_TYVAR_B);
        let either_ab = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::Con(self.either_con.clone())),
                Box::new(Ty::Var(a.clone())),
            )),
            Box::new(Ty::Var(b.clone())),
        );

        let left_id = DefId::new(BUILTIN_LEFT_ID);
        let right_id = DefId::new(BUILTIN_RIGHT_ID);
        env.register_data_con(
            left_id,
            Symbol::intern("Left"),
            Scheme::poly(
                vec![a.clone(), b.clone()],
                Ty::fun(Ty::Var(a.clone()), either_ab.clone()),
            ),
        );
        env.register_data_con(
            right_id,
            Symbol::intern("Right"),
            Scheme::poly(
                vec![a, b.clone()],
                Ty::fun(Ty::Var(b), either_ab),
            ),
        );

        // Unit constructor
        // () :: ()
        let unit_id = DefId::new(BUILTIN_UNIT_ID);
        env.register_data_con(unit_id, Symbol::intern("()"), Scheme::mono(Ty::unit()));
    }

    /// Create a list type `[a]`.
    #[must_use]
    #[allow(dead_code)]
    pub fn list_of(elem: Ty) -> Ty {
        Ty::List(Box::new(elem))
    }

    /// Create a Maybe type `Maybe a`.
    #[must_use]
    #[allow(dead_code)]
    pub fn maybe_of(&self, elem: Ty) -> Ty {
        Ty::App(Box::new(Ty::Con(self.maybe_con.clone())), Box::new(elem))
    }

    /// Create an IO type `IO a`.
    #[must_use]
    #[allow(dead_code)]
    pub fn io_of(&self, elem: Ty) -> Ty {
        Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(elem))
    }

    /// Create an Either type `Either a b`.
    #[must_use]
    #[allow(dead_code)]
    pub fn either_of(&self, left: Ty, right: Ty) -> Ty {
        Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::Con(self.either_con.clone())),
                Box::new(left),
            )),
            Box::new(right),
        )
    }
}

// Reserved DefId values for built-in constructors
// These are in a reserved range that won't conflict with user definitions
const BUILTIN_BASE: usize = 0xFFFF_0000;
const BUILTIN_TRUE_ID: usize = BUILTIN_BASE;
const BUILTIN_FALSE_ID: usize = BUILTIN_BASE + 1;
const BUILTIN_NOTHING_ID: usize = BUILTIN_BASE + 2;
const BUILTIN_JUST_ID: usize = BUILTIN_BASE + 3;
const BUILTIN_NIL_ID: usize = BUILTIN_BASE + 4;
const BUILTIN_CONS_ID: usize = BUILTIN_BASE + 5;
const BUILTIN_LEFT_ID: usize = BUILTIN_BASE + 6;
const BUILTIN_RIGHT_ID: usize = BUILTIN_BASE + 7;
const BUILTIN_UNIT_ID: usize = BUILTIN_BASE + 8;

// Reserved TyVar IDs for built-in schemes
const BUILTIN_TYVAR_A: u32 = 0xFFFF_0000;
const BUILTIN_TYVAR_B: u32 = 0xFFFF_0001;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtins_creation() {
        let builtins = Builtins::new();

        assert_eq!(builtins.int_con.name, Symbol::intern("Int"));
        assert!(builtins.int_con.kind.is_star());

        assert_eq!(builtins.maybe_con.name, Symbol::intern("Maybe"));
        assert!(!builtins.maybe_con.kind.is_star()); // * -> *
    }

    #[test]
    fn test_list_of() {
        let builtins = Builtins::new();
        let list_int = Builtins::list_of(builtins.int_ty.clone());

        match list_int {
            Ty::List(elem) => assert_eq!(*elem, builtins.int_ty),
            _ => panic!("expected list type"),
        }
    }

    #[test]
    fn test_register_data_cons() {
        let builtins = Builtins::new();
        let mut env = TypeEnv::new();
        builtins.register_data_cons(&mut env);

        // Check True is registered
        let true_info = env.lookup_data_con(Symbol::intern("True")).unwrap();
        assert_eq!(true_info.scheme.ty, builtins.bool_ty);

        // Check Just is registered with correct scheme
        let just_info = env.lookup_data_con(Symbol::intern("Just")).unwrap();
        assert!(!just_info.scheme.is_mono());
        assert_eq!(just_info.scheme.vars.len(), 1);
    }
}
