//! Built-in types and data constructors.
//!
//! This module defines the primitive types that are always available
//! in BHC programs: `Int`, `Float`, `Char`, `Bool`, `String`, etc.
//!
//! Additionally, this module defines the `Tensor` type constructor for
//! shape-indexed tensors (M9 Dependent Types Preview):
//!
//! ```text
//! Tensor :: [Nat] -> * -> *
//! ```
//!
//! Example usage:
//! ```text
//! Tensor '[1024, 768] Float  -- A 1024x768 matrix of floats
//! ```
//!
//! ## Dynamic Tensors (M9 Phase 5)
//!
//! For gradual adoption, `DynTensor` provides a runtime-shaped escape hatch:
//!
//! ```text
//! DynTensor :: * -> *
//!
//! toDynamic :: forall shape a. Tensor shape a -> DynTensor a
//! fromDynamic :: forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)
//! ```
//!
//! These types are registered into the type environment before
//! type checking user code.

use bhc_hir::DefId;
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{dyn_tensor, Constraint, Kind, Scheme, Ty, TyCon, TyList, TyVar};

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
    /// The `Text` type constructor (packed UTF-8).
    pub text_con: TyCon,
    /// The `ByteString` type constructor (packed bytes).
    pub bytestring_con: TyCon,
    /// The `Ordering` type constructor.
    pub ordering_con: TyCon,
    /// The `[]` (list) type constructor.
    pub list_con: TyCon,
    /// The `Maybe` type constructor.
    pub maybe_con: TyCon,
    /// The `Either` type constructor.
    pub either_con: TyCon,
    /// The `IO` type constructor.
    pub io_con: TyCon,
    /// The `Tensor` type constructor (M9).
    /// Kind: `[Nat] -> * -> *`
    pub tensor_con: TyCon,

    // M9 Phase 5: Dynamic tensor types
    /// The `DynTensor` type constructor.
    /// Kind: `* -> *`
    /// An existentially-quantified tensor with runtime-only shape.
    pub dyn_tensor_con: TyCon,
    /// The `ShapeWitness` type constructor.
    /// Kind: `[Nat] -> *`
    /// A singleton type for reifying shapes at runtime.
    pub shape_witness_con: TyCon,

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
    /// The `Text` type (packed UTF-8).
    pub text_ty: Ty,
    /// The `ByteString` type (packed bytes).
    pub bytestring_ty: Ty,
    /// The `Ordering` type.
    pub ordering_ty: Ty,
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
        let text_con = TyCon::new(Symbol::intern("Text"), Kind::Star);
        let bytestring_con = TyCon::new(Symbol::intern("ByteString"), Kind::Star);
        let ordering_con = TyCon::new(Symbol::intern("Ordering"), Kind::Star);

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

        // M9: Tensor type constructor with kind [Nat] -> * -> *
        // This enables shape-indexed tensors: Tensor '[1024, 768] Float
        let tensor_kind = Kind::Arrow(
            Box::new(Kind::List(Box::new(Kind::Nat))), // [Nat]
            Box::new(Kind::Arrow(
                Box::new(Kind::Star), // element type
                Box::new(Kind::Star), // result type
            )),
        );
        let tensor_con = TyCon::new(Symbol::intern("Tensor"), tensor_kind);

        // M9 Phase 5: Dynamic tensor types
        // DynTensor :: * -> *
        let dyn_tensor_con = dyn_tensor::dyn_tensor_tycon();

        // ShapeWitness :: [Nat] -> *
        let shape_witness_con = dyn_tensor::shape_witness_tycon();

        // Convenient types
        let int_ty = Ty::Con(int_con.clone());
        let float_ty = Ty::Con(float_con.clone());
        let char_ty = Ty::Con(char_con.clone());
        let bool_ty = Ty::Con(bool_con.clone());
        // String is a type alias for [Char] in Haskell
        let string_ty = Ty::List(Box::new(Ty::Con(char_con.clone())));
        // Text is a packed UTF-8 type (not [Char])
        let text_ty = Ty::Con(text_con.clone());
        // ByteString is a packed byte array type
        let bytestring_ty = Ty::Con(bytestring_con.clone());
        // Ordering is an ADT: LT | EQ | GT
        let ordering_ty = Ty::Con(ordering_con.clone());

        Self {
            int_con,
            float_con,
            char_con,
            bool_con,
            string_con,
            text_con,
            bytestring_con,
            ordering_con,
            list_con,
            maybe_con,
            either_con,
            io_con,
            tensor_con,
            dyn_tensor_con,
            shape_witness_con,
            int_ty,
            float_ty,
            char_ty,
            bool_ty,
            string_ty,
            text_ty,
            bytestring_ty,
            ordering_ty,
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
            Scheme::poly(vec![a, b.clone()], Ty::fun(Ty::Var(b), either_ab)),
        );

        // Ordering constructors
        // LT :: Ordering (tag=0)
        // EQ :: Ordering (tag=1)
        // GT :: Ordering (tag=2)
        env.register_data_con(
            DefId::new(BUILTIN_LT_ID),
            Symbol::intern("LT"),
            Scheme::mono(self.ordering_ty.clone()),
        );
        env.register_data_con(
            DefId::new(BUILTIN_EQ_ID),
            Symbol::intern("EQ"),
            Scheme::mono(self.ordering_ty.clone()),
        );
        env.register_data_con(
            DefId::new(BUILTIN_GT_ID),
            Symbol::intern("GT"),
            Scheme::mono(self.ordering_ty.clone()),
        );

        // Unit constructor
        // () :: ()
        let unit_id = DefId::new(BUILTIN_UNIT_ID);
        env.register_data_con(unit_id, Symbol::intern("()"), Scheme::mono(Ty::unit()));

        // Tuple constructors
        // (,) :: forall a b. a -> b -> (a, b)
        let pair_a = TyVar::new_star(BUILTIN_TYVAR_A);
        let pair_b = TyVar::new_star(BUILTIN_TYVAR_B);
        let pair_id = DefId::new(BUILTIN_PAIR_ID);
        let pair_ty = Ty::Tuple(vec![Ty::Var(pair_a.clone()), Ty::Var(pair_b.clone())]);
        env.register_data_con(
            pair_id,
            Symbol::intern("(,)"),
            Scheme::poly(
                vec![pair_a.clone(), pair_b.clone()],
                Ty::fun(Ty::Var(pair_a), Ty::fun(Ty::Var(pair_b), pair_ty)),
            ),
        );

        // (,,) :: forall a b c. a -> b -> c -> (a, b, c)
        let triple_a = TyVar::new_star(BUILTIN_TYVAR_A);
        let triple_b = TyVar::new_star(BUILTIN_TYVAR_B);
        let triple_c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
        let triple_id = DefId::new(BUILTIN_TRIPLE_ID);
        let triple_ty = Ty::Tuple(vec![
            Ty::Var(triple_a.clone()),
            Ty::Var(triple_b.clone()),
            Ty::Var(triple_c.clone()),
        ]);
        env.register_data_con(
            triple_id,
            Symbol::intern("(,,)"),
            Scheme::poly(
                vec![triple_a.clone(), triple_b.clone(), triple_c.clone()],
                Ty::fun(
                    Ty::Var(triple_a),
                    Ty::fun(Ty::Var(triple_b), Ty::fun(Ty::Var(triple_c), triple_ty)),
                ),
            ),
        );
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

    /// Create a Tensor type `Tensor shape elem`.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape as a type-level list (e.g., `TyList::shape_from_dims(&[1024, 768])`)
    /// * `elem` - The element type (e.g., `float_ty`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use bhc_types::TyList;
    ///
    /// let builtins = Builtins::new();
    /// let shape = TyList::shape_from_dims(&[1024, 768]);
    /// let tensor_type = builtins.tensor_of(Ty::TyList(shape), builtins.float_ty.clone());
    /// // tensor_type represents: Tensor '[1024, 768] Float
    /// ```
    #[must_use]
    #[allow(dead_code)]
    pub fn tensor_of(&self, shape: Ty, elem: Ty) -> Ty {
        Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::Con(self.tensor_con.clone())),
                Box::new(shape),
            )),
            Box::new(elem),
        )
    }

    /// Create a DynTensor type `DynTensor a`.
    ///
    /// # Arguments
    ///
    /// * `elem` - The element type (e.g., `float_ty`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builtins = Builtins::new();
    /// let dyn_float = builtins.dyn_tensor_of(builtins.float_ty.clone());
    /// // dyn_float represents: DynTensor Float
    /// ```
    #[must_use]
    #[allow(dead_code)]
    pub fn dyn_tensor_of(&self, elem: Ty) -> Ty {
        Ty::App(
            Box::new(Ty::Con(self.dyn_tensor_con.clone())),
            Box::new(elem),
        )
    }

    /// Create a ShapeWitness type `ShapeWitness shape`.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape as a type-level list
    #[must_use]
    #[allow(dead_code)]
    pub fn shape_witness_of(&self, shape: TyList) -> Ty {
        Ty::App(
            Box::new(Ty::Con(self.shape_witness_con.clone())),
            Box::new(Ty::TyList(shape)),
        )
    }

    /// Register primitive operators in the environment.
    ///
    /// This registers arithmetic, comparison, and other basic operators.
    /// The DefIds MUST match the order in bhc_lower::context::define_builtins.
    ///
    /// Operators registered:
    /// - Arithmetic: +, -, *, /, div, mod, ^, ^^, **
    /// - Comparison: ==, /=, <, <=, >, >=
    /// - Boolean: &&, ||
    /// - List: :, ++, !!
    /// - Function: ., $
    /// - Monadic: >>=, >>
    /// - Applicative: <*>, <$>, *>, <*
    /// - Alternative: <|>
    /// - And many more...
    pub fn register_primitive_ops(&self, env: &mut TypeEnv) {
        // Start after types (24) and constructors (37) = 61
        // Order MUST match bhc_lower::context::define_builtins
        let mut next_id = BUILTIN_TYPE_COUNT
            + BUILTIN_CON_COUNT
            + BUILTIN_ORDERING_COUNT
            + BUILTIN_LIST_UNIT_COUNT
            + BUILTIN_TUPLE_COUNT
            + BUILTIN_EXTRA_CON_COUNT;

        // Type variables for polymorphic types
        let a = TyVar::new_star(BUILTIN_TYVAR_A);
        let b = TyVar::new_star(BUILTIN_TYVAR_B);

        // Type class symbols for constraints
        let eq_class = Symbol::intern("Eq");
        let ord_class = Symbol::intern("Ord");
        let num_class = Symbol::intern("Num");
        let _show_class = Symbol::intern("Show");

        // Helper to create a constraint
        let eq_constraint = |ty: Ty| Constraint::new(eq_class, ty, Span::default());
        let ord_constraint = |ty: Ty| Constraint::new(ord_class, ty, Span::default());
        let _num_constraint = |ty: Ty| Constraint::new(num_class, ty, Span::default());

        // Helper to create common type schemes
        let num_binop = || {
            // a -> a -> a (for Num types, we simplify to Int for now)
            Scheme::mono(Ty::fun(
                self.int_ty.clone(),
                Ty::fun(self.int_ty.clone(), self.int_ty.clone()),
            ))
        };

        let cmp_binop = || {
            // a -> a -> Bool (for Ord types, polymorphic like eq_binop)
            Scheme::poly(
                vec![a.clone()],
                Ty::fun(
                    Ty::Var(a.clone()),
                    Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                ),
            )
        };

        let eq_binop = || {
            // a -> a -> Bool (for Eq types, polymorphic)
            Scheme::poly(
                vec![a.clone()],
                Ty::fun(
                    Ty::Var(a.clone()),
                    Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                ),
            )
        };

        let bool_binop = || {
            Scheme::mono(Ty::fun(
                self.bool_ty.clone(),
                Ty::fun(self.bool_ty.clone(), self.bool_ty.clone()),
            ))
        };

        // Register each operator with its DefId matching the lowering order
        // Order MUST match bhc_lower::context::define_builtins EXACTLY
        // Any mismatch will cause "cannot find value DefId(N)" errors
        let ops: Vec<(&str, Scheme)> = vec![
            // Arithmetic operators
            ("+", num_binop()),
            ("-", num_binop()),
            ("*", num_binop()),
            ("/", num_binop()),
            ("div", num_binop()),
            ("mod", num_binop()),
            ("^", num_binop()),
            ("^^", num_binop()),
            ("**", num_binop()),
            // Comparison operators
            ("==", eq_binop()),
            ("/=", eq_binop()),
            ("<", cmp_binop()),
            ("<=", cmp_binop()),
            (">", cmp_binop()),
            (">=", cmp_binop()),
            // Boolean operators
            ("&&", bool_binop()),
            ("||", bool_binop()),
            // List operators
            (":", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("++", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("!!", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a, Ty::fun(self.int_ty.clone(), Ty::Var(a.clone()))),
                )
            }),
            // List difference
            ("\\\\", {
                // (\\) :: Eq a => [a] -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            // Function composition
            (".", {
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                        ),
                    ),
                )
            }),
            ("$", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    ),
                )
            }),
            // Monadic operators (polymorphic - works with any monad)
            // m has kind * -> * (type constructor)
            (">>=", {
                // (>>=) :: m a -> (a -> m b) -> m b
                // We represent 'm a' as App(Var(m), Var(a))
                let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                let m = TyVar::new(BUILTIN_TYVAR_M, m_kind);
                let ma = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())));
                let mb = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![m.clone(), a.clone(), b.clone()],
                    Ty::fun(ma, Ty::fun(Ty::fun(Ty::Var(a.clone()), mb.clone()), mb)),
                )
            }),
            (">>", {
                // (>>) :: Monad m => m a -> m b -> m b
                let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                let m = TyVar::new(BUILTIN_TYVAR_M, m_kind);
                let ma = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())));
                let mb = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![m.clone(), a.clone(), b.clone()],
                    Ty::fun(ma, Ty::fun(mb.clone(), mb)),
                )
            }),
            ("=<<", {
                // (=<<) :: (a -> m b) -> m a -> m b (flipped >>=)
                let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                let m = TyVar::new(BUILTIN_TYVAR_M, m_kind);
                let ma = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())));
                let mb = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![m.clone(), a.clone(), b.clone()],
                    Ty::fun(Ty::fun(Ty::Var(a.clone()), mb.clone()), Ty::fun(ma, mb)),
                )
            }),
            // Applicative/Functor operators (list-specialized)
            // NOTE: Order MUST match bhc_lower::context::define_builtins
            ("<*>", {
                // (<*>) :: [a -> b] -> [a] -> [b] (list applicative)
                let list_fn = Ty::List(Box::new(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(list_fn, Ty::fun(list_a, list_b)),
                )
            }),
            ("<$>", {
                // (<$>) :: (a -> b) -> [a] -> [b] (same as fmap/map)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(list_a, list_b),
                    ),
                )
            }),
            ("<$", {
                // (<$) :: a -> [b] -> [a] (replace all with constant)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(list_b, list_a)),
                )
            }),
            ("*>", {
                // (*>) :: [a] -> [b] -> [b] (sequence, discarding first result)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(list_a, Ty::fun(list_b.clone(), list_b)),
                )
            }),
            ("<*", {
                // (<*) :: [a] -> [b] -> [a] (sequence, discarding second result)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(list_a.clone(), Ty::fun(list_b, list_a)),
                )
            }),
            ("fmap", {
                // fmap :: (a -> b) -> [a] -> [b] (same as map)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(list_a, list_b),
                    ),
                )
            }),
            // Alternative operator (list-specialized)
            ("<|>", {
                // (<|>) :: [a] -> [a] -> [a] (same as ++ for lists)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("empty", {
                // empty :: [a] (empty list for Alternative)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], list_a)
            }),
            // Semigroup/Monoid operators (list-specialized)
            ("<>", {
                // (<>) :: [a] -> [a] -> [a] (same as ++ for lists)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("mempty", {
                // mempty :: [a] (empty list for Monoid)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], list_a)
            }),
            ("mappend", {
                // mappend :: [a] -> [a] -> [a] (same as ++ for lists)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("mconcat", {
                // mconcat :: [[a]] -> [a] (same as concat for lists)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_list_a, list_a))
            }),
            // Monadic operations
            // return and pure are polymorphic - work with any monad/applicative
            ("return", {
                // return :: a -> m a
                let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                let m = TyVar::new(BUILTIN_TYVAR_M, m_kind);
                let ma = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![m, a.clone()], Ty::fun(Ty::Var(a.clone()), ma))
            }),
            ("pure", {
                // pure :: a -> f a (Applicative, same as return for monads)
                let f_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                let f = TyVar::new(BUILTIN_TYVAR_F, f_kind);
                let fa = Ty::App(Box::new(Ty::Var(f.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![f, a.clone()], Ty::fun(Ty::Var(a.clone()), fa))
            }),
            ("join", {
                // join :: [[a]] -> [a] (same as concat for lists)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_list_a, list_a))
            }),
            ("liftM", {
                // liftM :: (a -> b) -> [a] -> [b] (same as fmap/map)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(list_a, list_b),
                    ),
                )
            }),
            ("liftM2", {
                // liftM2 :: (a -> b -> c) -> [a] -> [b] -> [c]
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                        ),
                        Ty::fun(list_a, Ty::fun(list_b, list_c)),
                    ),
                )
            }),
            ("ap", {
                // ap :: [a -> b] -> [a] -> [b] (same as <*>)
                let list_fn = Ty::List(Box::new(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(list_fn, Ty::fun(list_a, list_b)),
                )
            }),
            ("mapM", {
                // mapM :: (a -> IO b) -> [a] -> IO [b] (IO-specialized)
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let io_list_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_b));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_b),
                        Ty::fun(list_a, io_list_b),
                    ),
                )
            }),
            ("mapM_", {
                // mapM_ :: (a -> IO b) -> [a] -> IO () (IO-specialized)
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::fun(Ty::Var(a.clone()), io_b), Ty::fun(list_a, io_unit)),
                )
            }),
            ("forM", {
                // forM :: [a] -> (a -> IO b) -> IO [b] (flipped mapM)
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let io_list_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_b));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        list_a,
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), io_b), io_list_b),
                    ),
                )
            }),
            ("forM_", {
                // forM_ :: [a] -> (a -> IO b) -> IO () (flipped mapM_)
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(list_a, Ty::fun(Ty::fun(Ty::Var(a.clone()), io_b), io_unit)),
                )
            }),
            ("sequence", {
                // sequence :: [IO a] -> IO [a] (IO-specialized)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let list_io_a = Ty::List(Box::new(io_a));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_list_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_a));
                Scheme::poly(vec![a.clone()], Ty::fun(list_io_a, io_list_a))
            }),
            ("sequence_", {
                // sequence_ :: [IO a] -> IO () (IO-specialized)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let list_io_a = Ty::List(Box::new(io_a));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_io_a, io_unit))
            }),
            ("when", {
                // when :: Bool -> IO () -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(
                    self.bool_ty.clone(),
                    Ty::fun(io_unit.clone(), io_unit),
                ))
            }),
            ("unless", {
                // unless :: Bool -> IO () -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(
                    self.bool_ty.clone(),
                    Ty::fun(io_unit.clone(), io_unit),
                ))
            }),
            ("void", {
                // void :: IO a -> IO ()
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(vec![a.clone()], Ty::fun(io_a, io_unit))
            }),
            ("filterM", {
                // filterM :: (a -> IO Bool) -> [a] -> IO [a]
                let io_bool = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.bool_ty.clone()));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_list_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_a.clone()));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_bool),
                        Ty::fun(list_a, io_list_a),
                    ),
                )
            }),
            ("foldM", {
                // foldM :: (b -> a -> IO b) -> b -> [a] -> IO b
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), io_b.clone())),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, io_b)),
                    ),
                )
            }),
            ("foldM_", {
                // foldM_ :: (b -> a -> IO b) -> b -> [a] -> IO ()
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), io_b)),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, io_unit)),
                    ),
                )
            }),
            ("replicateM", {
                // replicateM :: Int -> IO a -> IO [a]
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_list_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_a));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(io_a, io_list_a)),
                )
            }),
            ("replicateM_", {
                // replicateM_ :: Int -> IO a -> IO ()
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(io_a, io_unit)),
                )
            }),
            ("zipWithM", {
                // zipWithM :: (a -> b -> IO c) -> [a] -> [b] -> IO [c]
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let io_c = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(c.clone())));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                let io_list_c = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_c));
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), io_c)),
                        Ty::fun(list_a, Ty::fun(list_b, io_list_c)),
                    ),
                )
            }),
            ("zipWithM_", {
                // zipWithM_ :: (a -> b -> IO c) -> [a] -> [b] -> IO ()
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let io_c = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(c.clone())));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), io_c)),
                        Ty::fun(list_a, Ty::fun(list_b, io_unit)),
                    ),
                )
            }),
            ("liftIO", {
                // liftIO :: IO a -> IO a (identity for base IO)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(io_a.clone(), io_a))
            }),
            // Reader/State monad operations (simplified/polymorphic)
            ("ask", {
                // ask :: r -> r (simplified: Reader r r identity)
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                )
            }),
            ("asks", {
                // asks :: (r -> a) -> r -> a (simplified)
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    ),
                )
            }),
            ("local", {
                // local :: (r -> r) -> (r -> a) -> r -> a (simplified)
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        ),
                    ),
                )
            }),
            ("reader", {
                // reader :: (r -> a) -> r -> a (simplified)
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    ),
                )
            }),
            ("get", {
                // get :: s -> (a, s) where a = s (simplified State)
                let pair = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(a.clone())]);
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), pair))
            }),
            ("gets", {
                // gets :: (s -> a) -> s -> (a, s) (simplified)
                let pair = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(Ty::Var(a.clone()), pair),
                    ),
                )
            }),
            ("put", {
                // put :: s -> s -> ((), s) (simplified)
                let pair = Ty::Tuple(vec![Ty::unit(), Ty::Var(a.clone())]);
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), pair)),
                )
            }),
            ("modify", {
                // modify :: (s -> s) -> s -> ((), s) (simplified)
                let pair = Ty::Tuple(vec![Ty::unit(), Ty::Var(a.clone())]);
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                        Ty::fun(Ty::Var(a.clone()), pair),
                    ),
                )
            }),
            ("modify'", {
                // modify' :: (s -> s) -> s -> ((), s) (strict version)
                let pair = Ty::Tuple(vec![Ty::unit(), Ty::Var(a.clone())]);
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                        Ty::fun(Ty::Var(a.clone()), pair),
                    ),
                )
            }),
            ("state", {
                // state :: (s -> (a, s)) -> s -> (a, s) (simplified)
                let pair = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), pair.clone()),
                        Ty::fun(Ty::Var(a.clone()), pair),
                    ),
                )
            }),
            ("runReader", {
                // runReader :: (r -> a) -> r -> a (simplified)
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    ),
                )
            }),
            ("runReaderT", {
                // runReaderT :: (r -> IO a) -> r -> IO a (simplified)
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_b.clone()),
                        Ty::fun(Ty::Var(a.clone()), io_b),
                    ),
                )
            }),
            ("runState", {
                // runState :: (s -> (a, s)) -> s -> (a, s) (simplified)
                let pair = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), pair.clone()),
                        Ty::fun(Ty::Var(a.clone()), pair),
                    ),
                )
            }),
            ("runStateT", {
                // runStateT :: (s -> IO (a, s)) -> s -> IO (a, s) (simplified)
                let pair = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                let io_pair = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(pair.clone()),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_pair.clone()),
                        Ty::fun(Ty::Var(a.clone()), io_pair),
                    ),
                )
            }),
            ("evalState", {
                // evalState :: (s -> (a, s)) -> s -> a (get result, discard state)
                let pair = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), pair),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    ),
                )
            }),
            ("evalStateT", {
                // evalStateT :: (s -> IO (a, s)) -> s -> IO a
                let pair = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                let io_pair = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(pair));
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_pair),
                        Ty::fun(Ty::Var(a.clone()), io_b),
                    ),
                )
            }),
            ("execState", {
                // execState :: (s -> (a, s)) -> s -> s (get final state)
                let pair = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), pair),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                    ),
                )
            }),
            ("execStateT", {
                // execStateT :: (s -> IO (a, s)) -> s -> IO s
                let pair = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                let io_pair = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(pair));
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_pair),
                        Ty::fun(Ty::Var(a.clone()), io_a),
                    ),
                )
            }),
            ("lift", {
                // lift :: IO a -> IO a (identity for base IO monad)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(io_a.clone(), io_a))
            }),
            // Exception handling (simplified with String as exception type)
            ("catch", {
                // catch :: IO a -> (String -> IO a) -> IO a (simplified)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        io_a.clone(),
                        Ty::fun(Ty::fun(self.string_ty.clone(), io_a.clone()), io_a),
                    ),
                )
            }),
            ("try", {
                // try :: IO a -> IO (Either String a) (simplified)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let either_result = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(self.string_ty.clone()),
                    )),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_either = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(either_result),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(io_a, io_either))
            }),
            ("throw", {
                // throw :: String -> a (simplified with String exception)
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.string_ty.clone(), Ty::Var(a.clone())),
                )
            }),
            ("throwIO", {
                // throwIO :: String -> IO a (simplified)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(self.string_ty.clone(), io_a))
            }),
            ("bracket", {
                // bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let io_c = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(c.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        io_a,
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), io_b),
                            Ty::fun(Ty::fun(Ty::Var(a.clone()), io_c.clone()), io_c),
                        ),
                    ),
                )
            }),
            ("bracket_", {
                // bracket_ :: IO a -> IO b -> IO c -> IO c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let io_c = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(c.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(io_a, Ty::fun(io_b, Ty::fun(io_c.clone(), io_c))),
                )
            }),
            ("bracketOnError", {
                // bracketOnError :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let io_c = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(c.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        io_a,
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), io_b),
                            Ty::fun(Ty::fun(Ty::Var(a.clone()), io_c.clone()), io_c),
                        ),
                    ),
                )
            }),
            ("finally", {
                // finally :: IO a -> IO b -> IO a
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(io_a.clone(), Ty::fun(io_b, io_a)),
                )
            }),
            ("onException", {
                // onException :: IO a -> IO b -> IO a
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(io_a.clone(), Ty::fun(io_b, io_a)),
                )
            }),
            ("handle", {
                // handle :: (String -> IO a) -> IO a -> IO a (flipped catch)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(self.string_ty.clone(), io_a.clone()),
                        Ty::fun(io_a.clone(), io_a),
                    ),
                )
            }),
            ("handleJust", {
                // handleJust :: (String -> Maybe b) -> (b -> IO a) -> IO a -> IO a
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let maybe_b = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(self.string_ty.clone(), maybe_b),
                        Ty::fun(
                            Ty::fun(Ty::Var(b.clone()), io_a.clone()),
                            Ty::fun(io_a.clone(), io_a),
                        ),
                    ),
                )
            }),
            ("catchJust", {
                // catchJust :: (String -> Maybe b) -> IO a -> (b -> IO a) -> IO a
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let maybe_b = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(self.string_ty.clone(), maybe_b),
                        Ty::fun(
                            io_a.clone(),
                            Ty::fun(Ty::fun(Ty::Var(b.clone()), io_a.clone()), io_a),
                        ),
                    ),
                )
            }),
            ("tryJust", {
                // tryJust :: (String -> Maybe b) -> IO a -> IO (Either b a)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let maybe_b = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let either_ba = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    )),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_either =
                    Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(either_ba));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(self.string_ty.clone(), maybe_b),
                        Ty::fun(io_a, io_either),
                    ),
                )
            }),
            ("evaluate", {
                // evaluate :: a -> IO a (force evaluation)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), io_a))
            }),
            ("mask", {
                // mask :: ((IO a -> IO a) -> IO b) -> IO b (simplified)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::fun(Ty::fun(io_a.clone(), io_a), io_b.clone()), io_b),
                )
            }),
            ("mask_", {
                // mask_ :: IO a -> IO a
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(io_a.clone(), io_a))
            }),
            ("uninterruptibleMask", {
                // uninterruptibleMask :: ((IO a -> IO a) -> IO b) -> IO b
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::fun(Ty::fun(io_a.clone(), io_a), io_b.clone()), io_b),
                )
            }),
            ("uninterruptibleMask_", {
                // uninterruptibleMask_ :: IO a -> IO a
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(io_a.clone(), io_a))
            }),
            // IO operations (Handle abstracted as Int for now)
            ("hPutStr", {
                // hPutStr :: Handle -> String -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(
                    self.int_ty.clone(),
                    Ty::fun(self.string_ty.clone(), io_unit),
                ))
            }),
            ("hPutStrLn", {
                // hPutStrLn :: Handle -> String -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(
                    self.int_ty.clone(),
                    Ty::fun(self.string_ty.clone(), io_unit),
                ))
            }),
            ("hPrint", {
                // hPrint :: Show a => Handle -> a -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), io_unit)),
                )
            }),
            ("hGetLine", {
                // hGetLine :: Handle -> IO String
                let io_string = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(self.string_ty.clone()),
                );
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_string))
            }),
            ("hGetContents", {
                // hGetContents :: Handle -> IO String
                let io_string = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(self.string_ty.clone()),
                );
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_string))
            }),
            ("hClose", {
                // hClose :: Handle -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_unit))
            }),
            ("openFile", {
                // openFile :: FilePath -> IOMode -> IO Handle (FilePath = String, IOMode = Int, Handle = Int)
                let io_handle = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(self.int_ty.clone()),
                );
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::fun(self.int_ty.clone(), io_handle),
                ))
            }),
            ("withFile", {
                // withFile :: FilePath -> IOMode -> (Handle -> IO r) -> IO r
                let io_r = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        self.string_ty.clone(),
                        Ty::fun(
                            self.int_ty.clone(),
                            Ty::fun(Ty::fun(self.int_ty.clone(), io_r.clone()), io_r),
                        ),
                    ),
                )
            }),
            ("stdin", {
                // stdin :: Handle (Int)
                Scheme::mono(self.int_ty.clone())
            }),
            ("stdout", {
                // stdout :: Handle (Int)
                Scheme::mono(self.int_ty.clone())
            }),
            ("stderr", {
                // stderr :: Handle (Int)
                Scheme::mono(self.int_ty.clone())
            }),
            ("hFlush", {
                // hFlush :: Handle -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_unit))
            }),
            ("hIsEOF", {
                // hIsEOF :: Handle -> IO Bool
                let io_bool = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(self.bool_ty.clone()),
                );
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_bool))
            }),
            ("isEOF", {
                // isEOF :: IO Bool
                Scheme::mono(Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(self.bool_ty.clone()),
                ))
            }),
            ("getContents", {
                // getContents :: IO String
                Scheme::mono(Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(self.string_ty.clone()),
                ))
            }),
            ("interact", {
                // interact :: (String -> String) -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(
                    Ty::fun(self.string_ty.clone(), self.string_ty.clone()),
                    io_unit,
                ))
            }),
            ("appendFile", {
                // appendFile :: FilePath -> String -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::fun(self.string_ty.clone(), io_unit),
                ))
            }),
            // List operations (these need proper types for list comprehensions)
            ("map", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(list_a, list_b),
                    ),
                )
            }),
            ("filter", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                        Ty::fun(list_a.clone(), list_a),
                    ),
                )
            }),
            ("foldr", {
                // foldr :: (a -> b -> b) -> b -> [a] -> b
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())),
                        ),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, Ty::Var(b.clone()))),
                    ),
                )
            }),
            ("foldl", {
                // foldl :: (b -> a -> b) -> b -> [a] -> b
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(b.clone()),
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        ),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, Ty::Var(b.clone()))),
                    ),
                )
            }),
            ("foldl'", {
                // foldl' :: (b -> a -> b) -> b -> [a] -> b
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(b.clone()),
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        ),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, Ty::Var(b.clone()))),
                    ),
                )
            }),
            ("concatMap", {
                // concatMap :: (a -> [b]) -> [a] -> [b]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), list_b.clone()),
                        Ty::fun(list_a, list_b),
                    ),
                )
            }),
            ("head", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
            }),
            ("tail", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
            }),
            ("length", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, self.int_ty.clone()))
            }),
            ("null", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, self.bool_ty.clone()))
            }),
            ("reverse", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
            }),
            ("take", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("drop", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("sum", {
                let list_a = Ty::List(Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(list_a, self.int_ty.clone()))
            }),
            ("product", {
                let list_a = Ty::List(Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(list_a, self.int_ty.clone()))
            }),
            ("maximum", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
            }),
            ("minimum", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
            }),
            ("zip", {
                // zip :: [a] -> [b] -> [(a, b)]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let pair_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                let list_pair = Ty::List(Box::new(pair_ty));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(list_a, Ty::fun(list_b, list_pair)),
                )
            }),
            ("zipWith", {
                // zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                        ),
                        Ty::fun(list_a, Ty::fun(list_b, list_c)),
                    ),
                )
            }),
            ("zip3", {
                // zip3 :: [a] -> [b] -> [c] -> [(a, b, c)]
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                let triple_ty = Ty::Tuple(vec![
                    Ty::Var(a.clone()),
                    Ty::Var(b.clone()),
                    Ty::Var(c.clone()),
                ]);
                let list_triple = Ty::List(Box::new(triple_ty));
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(list_a, Ty::fun(list_b, Ty::fun(list_c, list_triple))),
                )
            }),
            ("zipWith3", {
                // zipWith3 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let d = TyVar::new_star(BUILTIN_TYVAR_B + 2);
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                let list_d = Ty::List(Box::new(Ty::Var(d.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone(), d.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(
                                Ty::Var(b.clone()),
                                Ty::fun(Ty::Var(c.clone()), Ty::Var(d.clone())),
                            ),
                        ),
                        Ty::fun(list_a, Ty::fun(list_b, Ty::fun(list_c, list_d))),
                    ),
                )
            }),
            ("unzip", {
                // unzip :: [(a, b)] -> ([a], [b])
                let pair_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                let list_pair = Ty::List(Box::new(pair_ty));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let result = Ty::Tuple(vec![list_a, list_b]);
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(list_pair, result))
            }),
            ("unzip3", {
                // unzip3 :: [(a, b, c)] -> ([a], [b], [c])
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let triple_ty = Ty::Tuple(vec![
                    Ty::Var(a.clone()),
                    Ty::Var(b.clone()),
                    Ty::Var(c.clone()),
                ]);
                let list_triple = Ty::List(Box::new(triple_ty));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                let result = Ty::Tuple(vec![list_a, list_b, list_c]);
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(list_triple, result),
                )
            }),
            ("lines", {
                // lines :: String -> [String]
                let list_string = Ty::List(Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(self.string_ty.clone(), list_string))
            }),
            ("unlines", {
                // unlines :: [String] -> String
                let list_string = Ty::List(Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(list_string, self.string_ty.clone()))
            }),
            ("words", {
                // words :: String -> [String]
                let list_string = Ty::List(Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(self.string_ty.clone(), list_string))
            }),
            ("unwords", {
                // unwords :: [String] -> String
                let list_string = Ty::List(Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(list_string, self.string_ty.clone()))
            }),
            ("concat", {
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_list_a, list_a))
            }),
            ("intercalate", {
                // intercalate :: [a] -> [[a]] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a.clone(), Ty::fun(list_list_a, list_a)),
                )
            }),
            ("intersperse", {
                // intersperse :: a -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("transpose", {
                // transpose :: [[a]] -> [[a]]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a));
                Scheme::poly(vec![a.clone()], Ty::fun(list_list_a.clone(), list_list_a))
            }),
            ("subsequences", {
                // subsequences :: [a] -> [[a]]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, list_list_a))
            }),
            ("permutations", {
                // permutations :: [a] -> [[a]]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, list_list_a))
            }),
            ("scanl", {
                // scanl :: (b -> a -> b) -> b -> [a] -> [b]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(b.clone()),
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        ),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, list_b)),
                    ),
                )
            }),
            ("scanl'", {
                // scanl' :: (b -> a -> b) -> b -> [a] -> [b]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(b.clone()),
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        ),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, list_b)),
                    ),
                )
            }),
            ("scanr", {
                // scanr :: (a -> b -> b) -> b -> [a] -> [b]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())),
                        ),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, list_b)),
                    ),
                )
            }),
            ("iterate", {
                // iterate :: (a -> a) -> a -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                        Ty::fun(Ty::Var(a.clone()), list_a),
                    ),
                )
            }),
            ("repeat", {
                // repeat :: a -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), list_a))
            }),
            ("replicate", {
                // replicate :: Int -> a -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), list_a)),
                )
            }),
            ("cycle", {
                // cycle :: [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
            }),
            ("splitAt", {
                // splitAt :: Int -> [a] -> ([a], [a])
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let pair = Ty::Tuple(vec![list_a.clone(), list_a.clone()]);
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(list_a, pair)),
                )
            }),
            ("span", {
                // span :: (a -> Bool) -> [a] -> ([a], [a])
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let pair = Ty::Tuple(vec![list_a.clone(), list_a.clone()]);
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                        Ty::fun(list_a, pair),
                    ),
                )
            }),
            ("break", {
                // break :: (a -> Bool) -> [a] -> ([a], [a])
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let pair = Ty::Tuple(vec![list_a.clone(), list_a.clone()]);
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                        Ty::fun(list_a, pair),
                    ),
                )
            }),
            ("takeWhile", {
                // takeWhile :: (a -> Bool) -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                        Ty::fun(list_a.clone(), list_a),
                    ),
                )
            }),
            ("dropWhile", {
                // dropWhile :: (a -> Bool) -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                        Ty::fun(list_a.clone(), list_a),
                    ),
                )
            }),
            ("group", {
                // group :: Eq a => [a] -> [[a]]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a, list_list_a),
                )
            }),
            ("inits", {
                // inits :: [a] -> [[a]]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, list_list_a))
            }),
            ("tails", {
                // tails :: [a] -> [[a]]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, list_list_a))
            }),
            ("isPrefixOf", {
                // isPrefixOf :: Eq a => [a] -> [a] -> Bool
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), Ty::fun(list_a, self.bool_ty.clone())),
                )
            }),
            ("isSuffixOf", {
                // isSuffixOf :: Eq a => [a] -> [a] -> Bool
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), Ty::fun(list_a, self.bool_ty.clone())),
                )
            }),
            ("isInfixOf", {
                // isInfixOf :: Eq a => [a] -> [a] -> Bool
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), Ty::fun(list_a, self.bool_ty.clone())),
                )
            }),
            ("elem", {
                // elem :: Eq a => a -> [a] -> Bool
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a, self.bool_ty.clone())),
                )
            }),
            ("notElem", {
                // notElem :: Eq a => a -> [a] -> Bool
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a, self.bool_ty.clone())),
                )
            }),
            ("lookup", {
                // lookup :: Eq a => a -> [(a, b)] -> Maybe b
                let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                let list_pair = Ty::List(Box::new(pair_ab));
                let maybe_b = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::qualified(
                    vec![a.clone(), b.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(list_pair, maybe_b)),
                )
            }),
            ("find", {
                // find :: (a -> Bool) -> [a] -> Maybe a
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                        Ty::fun(list_a, maybe_a),
                    ),
                )
            }),
            ("partition", {
                // partition :: (a -> Bool) -> [a] -> ([a], [a])
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let pair = Ty::Tuple(vec![list_a.clone(), list_a.clone()]);
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                        Ty::fun(list_a, pair),
                    ),
                )
            }),
            ("nub", {
                // nub :: Eq a => [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), list_a),
                )
            }),
            ("delete", {
                // delete :: Eq a => a -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("union", {
                // union :: Eq a => [a] -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("intersect", {
                // intersect :: Eq a => [a] -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("sort", {
                // sort :: Ord a => [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![ord_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), list_a),
                )
            }),
            ("sortBy", {
                // sortBy :: (a -> a -> Ordering) -> [a] -> [a]
                // Using Int as stand-in for Ordering (LT=-1, EQ=0, GT=1)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(a.clone()), self.int_ty.clone()),
                        ),
                        Ty::fun(list_a.clone(), list_a),
                    ),
                )
            }),
            ("sortOn", {
                // sortOn :: Ord b => (a -> b) -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone(), b.clone()],
                    vec![ord_constraint(Ty::Var(b.clone()))],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(list_a.clone(), list_a),
                    ),
                )
            }),
            ("insert", {
                // insert :: Ord a => a -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::qualified(
                    vec![a.clone()],
                    vec![ord_constraint(Ty::Var(a.clone()))],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("genericLength", {
                // genericLength :: Num i => [a] -> i
                // We'll use Int for now (no type class constraints)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, self.int_ty.clone()))
            }),
            ("genericTake", {
                // genericTake :: Integral i => i -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("genericDrop", {
                // genericDrop :: Integral i => i -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(list_a.clone(), list_a)),
                )
            }),
            ("genericSplitAt", {
                // genericSplitAt :: Integral i => i -> [a] -> ([a], [a])
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let pair = Ty::Tuple(vec![list_a.clone(), list_a.clone()]);
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(list_a, pair)),
                )
            }),
            ("genericIndex", {
                // genericIndex :: Integral i => [a] -> i -> a
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a, Ty::fun(self.int_ty.clone(), Ty::Var(a.clone()))),
                )
            }),
            ("genericReplicate", {
                // genericReplicate :: Integral i => i -> a -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), list_a)),
                )
            }),
            // Prelude functions
            (
                "id",
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                ),
            ),
            (
                "const",
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone())),
                    ),
                ),
            ),
            ("flip", {
                // flip :: (a -> b -> c) -> b -> a -> c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                        ),
                        Ty::fun(
                            Ty::Var(b.clone()),
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                        ),
                    ),
                )
            }),
            (
                "error",
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.string_ty.clone(), Ty::Var(a.clone())),
                ),
            ),
            (
                "undefined",
                Scheme::poly(vec![a.clone()], Ty::Var(a.clone())),
            ),
            (
                "seq",
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())),
                    ),
                ),
            ),
            // Numeric operations
            (
                "fromInteger",
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone())),
            ),
            ("fromRational", {
                // fromRational :: Rational -> Float (simplified)
                // Rational approximated as (Int, Int) tuple
                let rational = Ty::Tuple(vec![self.int_ty.clone(), self.int_ty.clone()]);
                Scheme::mono(Ty::fun(rational, self.float_ty.clone()))
            }),
            (
                "negate",
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone())),
            ),
            (
                "abs",
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone())),
            ),
            (
                "signum",
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone())),
            ),
            (
                "sqrt",
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone())),
            ),
            (
                "exp",
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone())),
            ),
            (
                "log",
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone())),
            ),
            (
                "sin",
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone())),
            ),
            (
                "cos",
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone())),
            ),
            (
                "tan",
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone())),
            ),
            // Comparison
            ("compare", {
                // compare :: a -> a -> Ordering (polymorphic for derived Ord)
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(a.clone()), self.ordering_ty.clone()),
                    ),
                )
            }),
            ("min", num_binop()),
            ("max", num_binop()),
            // Show
            (
                "show",
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), self.string_ty.clone()),
                ),
            ),
            // Boolean
            (
                "not",
                Scheme::mono(Ty::fun(self.bool_ty.clone(), self.bool_ty.clone())),
            ),
            ("otherwise", Scheme::mono(self.bool_ty.clone())),
            // Maybe
            ("maybe", {
                // maybe :: b -> (a -> b) -> Maybe a -> b
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Var(b.clone()),
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(maybe_a, Ty::Var(b.clone())),
                        ),
                    ),
                )
            }),
            ("fromMaybe", {
                // fromMaybe :: a -> Maybe a -> a
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(maybe_a, Ty::Var(a.clone()))),
                )
            }),
            // Either
            ("either", {
                // either :: (a -> c) -> (b -> c) -> Either a b -> c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let either_ab = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    )),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                        Ty::fun(
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                            Ty::fun(either_ab, Ty::Var(c.clone())),
                        ),
                    ),
                )
            }),
            // IO
            (
                "print",
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit())),
                    ),
                ),
            ),
            (
                "putStrLn",
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit())),
                )),
            ),
            (
                "putStr",
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit())),
                )),
            ),
            (
                "getLine",
                Scheme::mono(Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(self.string_ty.clone()),
                )),
            ),
            (
                "readFile",
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::App(
                        Box::new(Ty::Con(self.io_con.clone())),
                        Box::new(self.string_ty.clone()),
                    ),
                )),
            ),
            (
                "writeFile",
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::fun(
                        self.string_ty.clone(),
                        Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit())),
                    ),
                )),
            ),
            // Guard helper
            ("guard", {
                // guard :: Alternative f => Bool -> f ()
                // List-specialized: guard :: Bool -> [()]
                let list_unit = Ty::List(Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(self.bool_ty.clone(), list_unit))
            }),
            // Tuple functions
            ("fst", {
                // fst :: (a, b) -> a
                let pair_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(pair_ty, Ty::Var(a.clone())),
                )
            }),
            ("snd", {
                // snd :: (a, b) -> b
                let pair_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(pair_ty, Ty::Var(b.clone())),
                )
            }),
            ("curry", {
                // curry :: ((a, b) -> c) -> a -> b -> c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let pair_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(pair_ty, Ty::Var(c.clone())),
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                        ),
                    ),
                )
            }),
            ("uncurry", {
                // uncurry :: (a -> b -> c) -> (a, b) -> c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let pair_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                        ),
                        Ty::fun(pair_ty, Ty::Var(c.clone())),
                    ),
                )
            }),
            ("swap", {
                // swap :: (a, b) -> (b, a)
                let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                let pair_ba = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(pair_ab, pair_ba))
            }),
            // Character functions registered at fixed DefIds below (10200+)
            // Enum functions (Int-specialized)
            ("succ", {
                // succ :: Int -> Int (simplified for Int)
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))
            }),
            ("pred", {
                // pred :: Int -> Int
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))
            }),
            ("toEnum", {
                // toEnum :: Int -> a (polymorphic result)
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::Var(a.clone())),
                )
            }),
            ("fromEnum", {
                // fromEnum :: a -> Int (polymorphic input)
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), self.int_ty.clone()),
                )
            }),
            ("enumFrom", {
                // enumFrom :: Int -> [Int] (Int-specialized, [n..])
                let list_int = Ty::List(Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), list_int))
            }),
            ("enumFromThen", {
                // enumFromThen :: Int -> Int -> [Int] (Int-specialized, [n,m..])
                let list_int = Ty::List(Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(
                    self.int_ty.clone(),
                    Ty::fun(self.int_ty.clone(), list_int),
                ))
            }),
            ("enumFromTo", {
                // enumFromTo :: Int -> Int -> [Int] (Int-specialized, [n..m])
                let list_int = Ty::List(Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(
                    self.int_ty.clone(),
                    Ty::fun(self.int_ty.clone(), list_int),
                ))
            }),
            ("enumFromThenTo", {
                // enumFromThenTo :: Int -> Int -> Int -> [Int] (Int-specialized, [n,m..o])
                let list_int = Ty::List(Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(
                    self.int_ty.clone(),
                    Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), list_int)),
                ))
            }),
            // Bounded (Int-specialized)
            ("minBound", {
                // minBound :: Int (for Int, returns minimum Int value)
                Scheme::mono(self.int_ty.clone())
            }),
            ("maxBound", {
                // maxBound :: Int (for Int, returns maximum Int value)
                Scheme::mono(self.int_ty.clone())
            }),
            // Read functions (simplified)
            ("read", {
                // read :: String -> a (polymorphic result, may fail at runtime)
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.string_ty.clone(), Ty::Var(a.clone())),
                )
            }),
            ("reads", {
                // reads :: String -> [(a, String)] (with remaining string)
                let pair = Ty::Tuple(vec![Ty::Var(a.clone()), self.string_ty.clone()]);
                let list_pair = Ty::List(Box::new(pair));
                Scheme::poly(vec![a.clone()], Ty::fun(self.string_ty.clone(), list_pair))
            }),
            ("readMaybe", {
                // readMaybe :: String -> Maybe a
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(self.string_ty.clone(), maybe_a))
            }),
            ("readEither", {
                // readEither :: String -> Either String a
                let either_result = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(self.string_ty.clone()),
                    )),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.string_ty.clone(), either_result),
                )
            }),
            // Numeric conversion (simplified)
            ("toInteger", {
                // toInteger :: Int -> Int (identity for Int)
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))
            }),
            ("toRational", {
                // toRational :: Int -> (Int, Int) (as numerator/denominator pair)
                let rational = Ty::Tuple(vec![self.int_ty.clone(), self.int_ty.clone()]);
                Scheme::mono(Ty::fun(self.int_ty.clone(), rational))
            }),
            ("realToFrac", {
                // realToFrac :: Float -> Float (simplified, could be polymorphic)
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone()))
            }),
            ("truncate", {
                // truncate :: Float -> Int
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.int_ty.clone()))
            }),
            ("round", {
                // round :: Float -> Int
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.int_ty.clone()))
            }),
            ("ceiling", {
                // ceiling :: Float -> Int
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.int_ty.clone()))
            }),
            ("floor", {
                // floor :: Float -> Int
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.int_ty.clone()))
            }),
            // even/odd/gcd/lcm/quot/rem/quotRem/divMod moved to fixed DefIds 10500+
            // Keep placeholders to maintain sequential alignment
            ("even", Scheme::mono(Ty::fun(self.int_ty.clone(), self.bool_ty.clone()))),
            ("odd", Scheme::mono(Ty::fun(self.int_ty.clone(), self.bool_ty.clone()))),
            ("gcd", num_binop()),
            ("lcm", num_binop()),
            ("quot", num_binop()),
            ("rem", num_binop()),
            ("quotRem", {
                let pair = Ty::Tuple(vec![self.int_ty.clone(), self.int_ty.clone()]);
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), pair)))
            }),
            ("divMod", {
                let pair = Ty::Tuple(vec![self.int_ty.clone(), self.int_ty.clone()]);
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), pair)))
            }),
            ("recip", {
                // recip :: Fractional a => a -> a
                // Using Float for now (no type class constraints)
                Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone()))
            }),
            // Data.Function
            ("on", {
                // on :: (b -> b -> c) -> (a -> b) -> a -> a -> c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(b.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                        ),
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(
                                Ty::Var(a.clone()),
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                            ),
                        ),
                    ),
                )
            }),
            ("fix", {
                // fix :: (a -> a) -> a
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                        Ty::Var(a.clone()),
                    ),
                )
            }),
            // Data.Maybe (additional)
            ("isJust", {
                // isJust :: Maybe a -> Bool
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(maybe_a, self.bool_ty.clone()))
            }),
            ("isNothing", {
                // isNothing :: Maybe a -> Bool
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(maybe_a, self.bool_ty.clone()))
            }),
            ("listToMaybe", {
                // listToMaybe :: [a] -> Maybe a
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, maybe_a))
            }),
            ("maybeToList", {
                // maybeToList :: Maybe a -> [a]
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(maybe_a, list_a))
            }),
            ("catMaybes", {
                // catMaybes :: [Maybe a] -> [a]
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let list_maybe_a = Ty::List(Box::new(maybe_a));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_maybe_a, list_a))
            }),
            ("mapMaybe", {
                // mapMaybe :: (a -> Maybe b) -> [a] -> [b]
                let maybe_b = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), maybe_b),
                        Ty::fun(list_a, list_b),
                    ),
                )
            }),
            // Data.Either (additional)
            ("isLeft", {
                // isLeft :: Either a b -> Bool
                let either_ab = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    )),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(either_ab, self.bool_ty.clone()),
                )
            }),
            ("isRight", {
                // isRight :: Either a b -> Bool
                let either_ab = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    )),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(either_ab, self.bool_ty.clone()),
                )
            }),
            ("lefts", {
                // lefts :: [Either a b] -> [a]
                let either_ab = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    )),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_either = Ty::List(Box::new(either_ab));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(list_either, list_a))
            }),
            ("rights", {
                // rights :: [Either a b] -> [b]
                let either_ab = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    )),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_either = Ty::List(Box::new(either_ab));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(list_either, list_b))
            }),
            ("partitionEithers", {
                // partitionEithers :: [Either a b] -> ([a], [b])
                let either_ab = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    )),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_either = Ty::List(Box::new(either_ab));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let result = Ty::Tuple(vec![list_a, list_b]);
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(list_either, result))
            }),
            // Control.Applicative (list-specialized)
            ("optional", {
                // optional :: [a] -> [Maybe a] (list-specialized)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let list_maybe = Ty::List(Box::new(maybe_a));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, list_maybe))
            }),
            ("some", {
                // some :: [a] -> [[a]] (one or more occurrences, list-specialized)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, list_list_a))
            }),
            ("many", {
                // many :: [a] -> [[a]] (zero or more occurrences, list-specialized)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, list_list_a))
            }),
            // Data.Foldable (list-specialized versions)
            ("fold", {
                // fold :: Monoid m => [m] -> m (list-specialized)
                // Without type classes, use polymorphic version
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
            }),
            ("foldMap", {
                // foldMap :: Monoid m => (a -> m) -> [a] -> m (list-specialized)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(list_a, Ty::Var(b.clone())),
                    ),
                )
            }),
            ("toList", {
                // toList :: Foldable t => t a -> [a] (identity for lists)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
            }),
            ("any", {
                // any :: Foldable t => (a -> Bool) -> t a -> Bool (list-specialized)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                        Ty::fun(list_a, self.bool_ty.clone()),
                    ),
                )
            }),
            ("all", {
                // all :: Foldable t => (a -> Bool) -> t a -> Bool (list-specialized)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()),
                        Ty::fun(list_a, self.bool_ty.clone()),
                    ),
                )
            }),
            ("and", {
                // and :: [Bool] -> Bool
                let list_bool = Ty::List(Box::new(self.bool_ty.clone()));
                Scheme::mono(Ty::fun(list_bool, self.bool_ty.clone()))
            }),
            ("or", {
                // or :: [Bool] -> Bool
                let list_bool = Ty::List(Box::new(self.bool_ty.clone()));
                Scheme::mono(Ty::fun(list_bool, self.bool_ty.clone()))
            }),
            ("asum", {
                // asum :: (Foldable t, Alternative f) => t (f a) -> f a
                // Specialized to list of Maybe: [Maybe a] -> Maybe a
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let list_maybe = Ty::List(Box::new(maybe_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_maybe, maybe_a))
            }),
            ("msum", {
                // msum :: (Foldable t, MonadPlus m) => t (m a) -> m a
                // Same as asum for MonadPlus
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let list_maybe = Ty::List(Box::new(maybe_a.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_maybe, maybe_a))
            }),
            // Data.Traversable (IO-specialized)
            ("traverse", {
                // traverse :: (a -> IO b) -> [a] -> IO [b] (same as mapM)
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let io_list_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_b));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_b),
                        Ty::fun(list_a, io_list_b),
                    ),
                )
            }),
            ("traverse_", {
                // traverse_ :: (a -> IO b) -> [a] -> IO () (same as mapM_)
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::fun(Ty::Var(a.clone()), io_b), Ty::fun(list_a, io_unit)),
                )
            }),
            ("for", {
                // for :: [a] -> (a -> IO b) -> IO [b] (flipped traverse)
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let io_list_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_b));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        list_a,
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), io_b), io_list_b),
                    ),
                )
            }),
            ("for_", {
                // for_ :: [a] -> (a -> IO b) -> IO () (flipped traverse_)
                let io_b = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(b.clone())),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(list_a, Ty::fun(Ty::fun(Ty::Var(a.clone()), io_b), io_unit)),
                )
            }),
            ("sequenceA", {
                // sequenceA :: [IO a] -> IO [a] (same as sequence)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let list_io_a = Ty::List(Box::new(io_a));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_list_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_a));
                Scheme::poly(vec![a.clone()], Ty::fun(list_io_a, io_list_a))
            }),
            ("sequenceA_", {
                // sequenceA_ :: [IO a] -> IO () (same as sequence_)
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                let list_io_a = Ty::List(Box::new(io_a));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_io_a, io_unit))
            }),
            // Common type constructors used as functions
            ("Just", {
                // Just :: a -> Maybe a
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), maybe_a))
            }),
            ("Nothing", {
                // Nothing :: Maybe a
                let maybe_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(vec![a.clone()], maybe_a)
            }),
            ("Left", {
                // Left :: a -> Either a b
                let either_ab = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    )),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), either_ab),
                )
            }),
            ("Right", {
                // Right :: b -> Either a b
                let either_ab = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    )),
                    Box::new(Ty::Var(b.clone())),
                );
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(b.clone()), either_ab),
                )
            }),
            // ---- Entries matching context.rs after "Right" ----
            // These must be in the exact same order as context.rs
            // Monad fail
            ("fail", {
                // fail :: String -> m a
                let io_a = Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                );
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.string_ty.clone(), io_a),
                )
            }),
            // Control.Applicative.Backwards
            ("forwards", {
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                )
            }),
            // Data.Monoid
            ("appEndo", {
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                )
            }),
            ("getAny", {
                Scheme::mono(Ty::fun(self.bool_ty.clone(), self.bool_ty.clone()))
            }),
            ("getAll", {
                Scheme::mono(Ty::fun(self.bool_ty.clone(), self.bool_ty.clone()))
            }),
            // Control.Arrow
            ("first", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                )
            }),
            ("second", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                )
            }),
            ("***", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                )
            }),
            ("&&&", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                )
            }),
            ("arr", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::Var(a.clone()),
                    ),
                )
            }),
            ("returnA", {
                Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
            }),
            ("<<<", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                )
            }),
            (">>>", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                )
            }),
            // Data.Bits
            (".|.", num_binop()),
            (".&.", num_binop()),
            ("xor", num_binop()),
            ("complement", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))
            }),
            ("shiftL", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))
            }),
            ("shiftR", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))
            }),
            ("rotateL", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))
            }),
            ("rotateR", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))
            }),
            ("bit", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))
            }),
            ("setBit", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))
            }),
            ("clearBit", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))
            }),
            ("complementBit", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))
            }),
            ("testBit", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.bool_ty.clone())))
            }),
            ("popCount", {
                Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))
            }),
            ("zeroBits", {
                Scheme::mono(self.int_ty.clone())
            }),
            // Data.Typeable
            ("cast", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::App(
                        Box::new(Ty::Con(self.maybe_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    )),
                )
            }),
            ("typeOf", {
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), self.string_ty.clone()),
                )
            }),
            ("typeRep", {
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), self.string_ty.clone()),
                )
            }),
            // System.Directory
            ("getAppUserDataDirectory", {
                let io_string = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(self.string_ty.clone(), io_string))
            }),
            ("getXdgDirectory", {
                let io_string = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(self.string_ty.clone(), Ty::fun(self.string_ty.clone(), io_string)))
            }),
            ("createDirectoryIfMissing", {
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(self.bool_ty.clone(), Ty::fun(self.string_ty.clone(), io_unit)))
            }),
            ("listDirectory", {
                let list_string = Ty::List(Box::new(self.string_ty.clone()));
                let io_list = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_string));
                Scheme::mono(Ty::fun(self.string_ty.clone(), io_list))
            }),
            ("getModificationTime", {
                let io_string = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(self.string_ty.clone(), io_string))
            }),
            ("getPermissions", {
                let io_string = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(self.string_ty.clone(), io_string))
            }),
            ("executable", {
                Scheme::mono(Ty::fun(self.string_ty.clone(), self.bool_ty.clone()))
            }),
            ("canonicalizePath", {
                let io_string = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(self.string_ty.clone(), io_string))
            }),
            // System.FilePath
            ("splitExtension", {
                let pair = Ty::Tuple(vec![self.string_ty.clone(), self.string_ty.clone()]);
                Scheme::mono(Ty::fun(self.string_ty.clone(), pair))
            }),
            // System.Process
            ("createProcess_", {
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), io_unit))
            }),
            ("waitForProcess", {
                let io_int = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.int_ty.clone()));
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), io_int))
            }),
            ("proc", {
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.string_ty.clone(), Ty::fun(Ty::List(Box::new(self.string_ty.clone())), Ty::Var(a.clone()))),
                )
            }),
            // System.Exit
            ("exitSuccess", {
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], io_a)
            }),
            ("exitFailure", {
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], io_a)
            }),
            ("exitWith", {
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), io_a))
            }),
            // Data.Version
            ("showVersion", {
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.string_ty.clone()))
            }),
            ("version", {
                Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
            }),
            // System.Info
            ("compilerName", {
                Scheme::mono(self.string_ty.clone())
            }),
            ("compilerVersion", {
                Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
            }),
            ("arch", {
                Scheme::mono(self.string_ty.clone())
            }),
            ("os", {
                Scheme::mono(self.string_ty.clone())
            }),
            // Data.Ratio
            ("%", num_binop()),
            // Data.Function (additional)
            ("$!", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    ),
                )
            }),
            // Control.Exception
            ("fromException", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::App(
                        Box::new(Ty::Con(self.maybe_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    )),
                )
            }),
            ("toException", {
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                )
            }),
            ("displayException", {
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), self.string_ty.clone()),
                )
            }),
            // ---- Phase 1 new PrimOps (must also be added to context.rs) ----
            // Scans
            ("scanl1", {
                // scanl1 :: (a -> a -> a) -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                        Ty::fun(list_a.clone(), list_a),
                    ),
                )
            }),
            ("scanr1", {
                // scanr1 :: (a -> a -> a) -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                        Ty::fun(list_a.clone(), list_a),
                    ),
                )
            }),
            // Integral
            ("subtract", {
                // subtract :: Num a => a -> a -> a (simplified to Int)
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))
            }),
            // Data.List "By" variants
            ("nubBy", {
                // nubBy :: (a -> a -> Bool) -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())),
                        Ty::fun(list_a.clone(), list_a),
                    ),
                )
            }),
            ("groupBy", {
                // groupBy :: (a -> a -> Bool) -> [a] -> [[a]]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_list_a = Ty::List(Box::new(list_a.clone()));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())),
                        Ty::fun(list_a, list_list_a),
                    ),
                )
            }),
            ("deleteBy", {
                // deleteBy :: (a -> a -> Bool) -> a -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())),
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                    ),
                )
            }),
            ("unionBy", {
                // unionBy :: (a -> a -> Bool) -> [a] -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())),
                        Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                    ),
                )
            }),
            ("intersectBy", {
                // intersectBy :: (a -> a -> Bool) -> [a] -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())),
                        Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                    ),
                )
            }),
            ("stripPrefix", {
                // stripPrefix :: Eq a => [a] -> [a] -> Maybe [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let maybe_list_a = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(list_a.clone()),
                );
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a.clone(), Ty::fun(list_a, maybe_list_a)),
                )
            }),
            // Accumulating maps
            ("mapAccumL", {
                // mapAccumL :: (acc -> x -> (acc, y)) -> acc -> [x] -> (acc, [y])
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let pair_ac = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::List(Box::new(Ty::Var(b.clone())))]);
                let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(c.clone()), pair_ab)),
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_c, pair_ac)),
                    ),
                )
            }),
            ("mapAccumR", {
                // mapAccumR :: (acc -> x -> (acc, y)) -> acc -> [x] -> (acc, [y])
                let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let pair_ac = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::List(Box::new(Ty::Var(b.clone())))]);
                let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(c.clone()), pair_ab)),
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_c, pair_ac)),
                    ),
                )
            }),
            ("unfoldr", {
                // unfoldr :: (b -> Maybe (a, b)) -> b -> [a]
                let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                let maybe_pair = Ty::App(
                    Box::new(Ty::Con(self.maybe_con.clone())),
                    Box::new(pair_ab),
                );
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(b.clone()), maybe_pair),
                        Ty::fun(Ty::Var(b.clone()), list_a),
                    ),
                )
            }),
            // Data.Char
            ("toTitle", {
                Scheme::mono(Ty::fun(self.char_ty.clone(), self.char_ty.clone()))
            }),
            ("isLatin1", {
                Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))
            }),
            ("isAsciiLower", {
                Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))
            }),
            ("isAsciiUpper", {
                Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))
            }),
            // Show helpers
            ("showString", {
                // showString :: String -> ShowS (where ShowS = String -> String)
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::fun(self.string_ty.clone(), self.string_ty.clone()),
                ))
            }),
            ("showChar", {
                // showChar :: Char -> ShowS
                Scheme::mono(Ty::fun(
                    self.char_ty.clone(),
                    Ty::fun(self.string_ty.clone(), self.string_ty.clone()),
                ))
            }),
            ("showParen", {
                // showParen :: Bool -> ShowS -> ShowS
                Scheme::mono(Ty::fun(
                    self.bool_ty.clone(),
                    Ty::fun(
                        Ty::fun(self.string_ty.clone(), self.string_ty.clone()),
                        Ty::fun(self.string_ty.clone(), self.string_ty.clone()),
                    ),
                ))
            }),
            // IO
            ("getChar", {
                // getChar :: IO Char
                Scheme::mono(Ty::App(
                    Box::new(Ty::Con(self.io_con.clone())),
                    Box::new(self.char_ty.clone()),
                ))
            }),
            // Data.Function
            ("&", {
                // (&) :: a -> (a -> b) -> b
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::Var(b.clone()),
                        ),
                    ),
                )
            }),
            // Container PrimOps: Data.Map
            ("Data.Map.empty", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("Data.Map.singleton", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone()))))),
            ("Data.Map.null", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()))),
            ("Data.Map.size", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), self.int_ty.clone()))),
            ("Data.Map.member", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), self.bool_ty.clone())))),
            ("Data.Map.notMember", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), self.bool_ty.clone())))),
            ("Data.Map.lookup", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))))),
            ("Data.Map.findWithDefault", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())))))),
            ("Data.Map.!", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))))),
            ("Data.Map.insert", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))))),
            ("Data.Map.insertWith", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))))),
            ("Data.Map.delete", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))))),
            ("Data.Map.adjust", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())))))),
            ("Data.Map.update", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(b.clone()), self.maybe_of(Ty::Var(b.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())))))),
            ("Data.Map.alter", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(self.maybe_of(Ty::Var(b.clone())), self.maybe_of(Ty::Var(b.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())))))),
            ("Data.Map.union", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.Map.unionWith", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))))),
            ("Data.Map.unionWithKey", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))))),
            ("Data.Map.unions", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
            ("Data.Map.intersection", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.Map.intersectionWith", {
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                Scheme::poly(vec![a.clone(), b.clone(), c.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())))))
            }),
            ("Data.Map.difference", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.Map.differenceWith", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone())))))),
            ("Data.Map.map", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))))),
            ("Data.Map.mapWithKey", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))))),
            ("Data.Map.mapKeys", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))))),
            ("Data.Map.filter", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(b.clone()), self.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.Map.filterWithKey", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), self.bool_ty.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.Map.foldr", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))))),
            ("Data.Map.foldl", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone())))))),
            ("Data.Map.foldrWithKey", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))))),
            ("Data.Map.foldlWithKey", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone())))))),
            ("Data.Map.keys", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Map.elems", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Map.assocs", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Map.toList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Map.toAscList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Map.toDescList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Map.fromList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Map.fromListWith", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))))),
            ("Data.Map.keysSet", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Map.isSubmapOf", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())))),
            // Container PrimOps: Data.Set
            ("Data.Set.empty", Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))),
            ("Data.Set.singleton", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.null", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()))),
            ("Data.Set.size", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.int_ty.clone()))),
            ("Data.Set.member", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), self.bool_ty.clone())))),
            ("Data.Set.notMember", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), self.bool_ty.clone())))),
            ("Data.Set.insert", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))))),
            ("Data.Set.delete", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))))),
            ("Data.Set.union", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.Set.unions", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.intersection", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.Set.difference", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.Set.isSubsetOf", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())))),
            ("Data.Set.isProperSubsetOf", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())))),
            ("Data.Set.map", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))))),
            ("Data.Set.filter", Scheme::poly(vec![a.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.Set.partition", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))))),
            ("Data.Set.foldr", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))))),
            ("Data.Set.foldl", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))))),
            ("Data.Set.toList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.toAscList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.toDescList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.fromList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.elems", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.findMin", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.findMax", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.lookupMin", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.lookupMax", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.Set.deleteMin", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
            ("Data.Set.deleteMax", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
            // Container PrimOps: Data.IntMap
            ("Data.IntMap.empty", Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))),
            ("Data.IntMap.singleton", Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntMap.null", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()))),
            ("Data.IntMap.size", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.int_ty.clone()))),
            ("Data.IntMap.member", Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())))),
            ("Data.IntMap.lookup", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))))),
            ("Data.IntMap.findWithDefault", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))))),
            ("Data.IntMap.insert", Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))))),
            ("Data.IntMap.insertWith", Scheme::poly(vec![a.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())), Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))))),
            ("Data.IntMap.delete", Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntMap.adjust", Scheme::poly(vec![a.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())), Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))))),
            ("Data.IntMap.union", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntMap.unionWith", Scheme::poly(vec![a.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))))),
            ("Data.IntMap.intersection", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntMap.difference", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntMap.map", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))))),
            ("Data.IntMap.mapWithKey", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))))),
            ("Data.IntMap.filter", Scheme::poly(vec![a.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntMap.foldr", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))))),
            ("Data.IntMap.foldlWithKey", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone())))), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone())))))),
            ("Data.IntMap.keys", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
            ("Data.IntMap.elems", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.IntMap.toList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.IntMap.toAscList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            ("Data.IntMap.fromList", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
            // Container PrimOps: Data.IntSet
            ("Data.IntSet.empty", Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))),
            ("Data.IntSet.singleton", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::Var(a.clone())))),
            ("Data.IntSet.null", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone()))),
            ("Data.IntSet.size", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.int_ty.clone()))),
            ("Data.IntSet.member", Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())))),
            ("Data.IntSet.insert", Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntSet.delete", Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntSet.union", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntSet.intersection", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntSet.difference", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntSet.isSubsetOf", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.bool_ty.clone())))),
            ("Data.IntSet.filter", Scheme::poly(vec![a.clone()], Ty::fun(Ty::fun(self.int_ty.clone(), self.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
            ("Data.IntSet.foldr", Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))))),
            ("Data.IntSet.toList", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
            ("Data.IntSet.fromList", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
            // Data.Text PrimOps: packed UTF-8 text
            ("Data.Text.empty", Scheme::mono(self.text_ty.clone())),
            ("Data.Text.singleton", Scheme::mono(Ty::fun(self.char_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.pack", Scheme::mono(Ty::fun(self.string_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.unpack", Scheme::mono(Ty::fun(self.text_ty.clone(), self.string_ty.clone()))),
            ("Data.Text.null", Scheme::mono(Ty::fun(self.text_ty.clone(), self.bool_ty.clone()))),
            ("Data.Text.length", Scheme::mono(Ty::fun(self.text_ty.clone(), self.int_ty.clone()))),
            ("Data.Text.head", Scheme::mono(Ty::fun(self.text_ty.clone(), self.char_ty.clone()))),
            ("Data.Text.last", Scheme::mono(Ty::fun(self.text_ty.clone(), self.char_ty.clone()))),
            ("Data.Text.tail", Scheme::mono(Ty::fun(self.text_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.init", Scheme::mono(Ty::fun(self.text_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.append", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), self.text_ty.clone())))),
            ("Data.Text.<>", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), self.text_ty.clone())))),
            ("Data.Text.reverse", Scheme::mono(Ty::fun(self.text_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.take", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.text_ty.clone(), self.text_ty.clone())))),
            ("Data.Text.takeEnd", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.text_ty.clone(), self.text_ty.clone())))),
            ("Data.Text.drop", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.text_ty.clone(), self.text_ty.clone())))),
            ("Data.Text.dropEnd", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.text_ty.clone(), self.text_ty.clone())))),
            ("Data.Text.isPrefixOf", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), self.bool_ty.clone())))),
            ("Data.Text.isSuffixOf", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), self.bool_ty.clone())))),
            ("Data.Text.isInfixOf", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), self.bool_ty.clone())))),
            ("Data.Text.toLower", Scheme::mono(Ty::fun(self.text_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.toUpper", Scheme::mono(Ty::fun(self.text_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.toCaseFold", Scheme::mono(Ty::fun(self.text_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.toTitle", Scheme::mono(Ty::fun(self.text_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.map", Scheme::mono(Ty::fun(Ty::fun(self.char_ty.clone(), self.char_ty.clone()), Ty::fun(self.text_ty.clone(), self.text_ty.clone())))),
            ("Data.Text.eq", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), self.bool_ty.clone())))),
            ("Data.Text.==", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), self.bool_ty.clone())))),
            // compare returns Ordering tags (0=LT, 1=EQ, 2=GT) as Int
            ("Data.Text.compare", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), self.int_ty.clone())))),
            // Additional Data.Text operations
            ("Data.Text.filter", Scheme::mono(Ty::fun(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()), Ty::fun(self.text_ty.clone(), self.text_ty.clone())))),
            ("Data.Text.foldl'", Scheme::poly(vec![a.clone()], Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(self.char_ty.clone(), Ty::Var(a.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::fun(self.text_ty.clone(), Ty::Var(a.clone())))))),
            ("Data.Text.concat", Scheme::mono(Ty::fun(Ty::List(Box::new(self.text_ty.clone())), self.text_ty.clone()))),
            ("Data.Text.intercalate", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(Ty::List(Box::new(self.text_ty.clone())), self.text_ty.clone())))),
            ("Data.Text.strip", Scheme::mono(Ty::fun(self.text_ty.clone(), self.text_ty.clone()))),
            ("Data.Text.words", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::List(Box::new(self.text_ty.clone()))))),
            ("Data.Text.lines", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::List(Box::new(self.text_ty.clone()))))),
            ("Data.Text.splitOn", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), Ty::List(Box::new(self.text_ty.clone())))))),
            ("Data.Text.replace", Scheme::mono(Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), Ty::fun(self.text_ty.clone(), self.text_ty.clone()))))),
            // Data.Text.Encoding
            ("Data.Text.Encoding.encodeUtf8", Scheme::mono(Ty::fun(self.text_ty.clone(), self.bytestring_ty.clone()))),
            ("Data.Text.Encoding.decodeUtf8", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), self.text_ty.clone()))),
            // Data.ByteString PrimOps: packed byte arrays
            ("Data.ByteString.empty", Scheme::mono(self.bytestring_ty.clone())),
            ("Data.ByteString.singleton", Scheme::mono(Ty::fun(self.int_ty.clone(), self.bytestring_ty.clone()))),
            ("Data.ByteString.pack", Scheme::mono(Ty::fun(Ty::List(Box::new(self.int_ty.clone())), self.bytestring_ty.clone()))),
            ("Data.ByteString.unpack", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), Ty::List(Box::new(self.int_ty.clone()))))),
            ("Data.ByteString.null", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), self.bool_ty.clone()))),
            ("Data.ByteString.length", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), self.int_ty.clone()))),
            ("Data.ByteString.head", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), self.int_ty.clone()))),
            ("Data.ByteString.last", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), self.int_ty.clone()))),
            ("Data.ByteString.tail", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), self.bytestring_ty.clone()))),
            ("Data.ByteString.init", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), self.bytestring_ty.clone()))),
            ("Data.ByteString.append", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), Ty::fun(self.bytestring_ty.clone(), self.bytestring_ty.clone())))),
            ("Data.ByteString.cons", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.bytestring_ty.clone(), self.bytestring_ty.clone())))),
            ("Data.ByteString.snoc", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), Ty::fun(self.int_ty.clone(), self.bytestring_ty.clone())))),
            ("Data.ByteString.take", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.bytestring_ty.clone(), self.bytestring_ty.clone())))),
            ("Data.ByteString.drop", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.bytestring_ty.clone(), self.bytestring_ty.clone())))),
            ("Data.ByteString.reverse", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), self.bytestring_ty.clone()))),
            ("Data.ByteString.elem", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.bytestring_ty.clone(), self.bool_ty.clone())))),
            ("Data.ByteString.index", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))),
            ("Data.ByteString.eq", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), Ty::fun(self.bytestring_ty.clone(), self.bool_ty.clone())))),
            ("Data.ByteString.compare", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), Ty::fun(self.bytestring_ty.clone(), self.int_ty.clone())))),
            ("Data.ByteString.isPrefixOf", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), Ty::fun(self.bytestring_ty.clone(), self.bool_ty.clone())))),
            ("Data.ByteString.isSuffixOf", Scheme::mono(Ty::fun(self.bytestring_ty.clone(), Ty::fun(self.bytestring_ty.clone(), self.bool_ty.clone())))),
            ("Data.ByteString.readFile", Scheme::mono(Ty::fun(self.string_ty.clone(), Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.bytestring_ty.clone()))))),
            ("Data.ByteString.writeFile", Scheme::mono(Ty::fun(self.string_ty.clone(), Ty::fun(self.bytestring_ty.clone(), Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Tuple(vec![]))))))),
            // ---- Phase 3: IO PrimOps (genuinely new) ----
            ("hGetChar", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("hPutChar", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("hSetBuffering", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("hGetBuffering", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("hSeek", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("hTell", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("hFileSize", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("setEnv", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            // ---- Phase 4: Control.* PrimOps (genuinely new) ----
            ("liftM3", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("liftM4", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("liftM5", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("mzero", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("mplus", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("mfilter", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            (">=>", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("<=<", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("liftA", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("liftA2", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("liftA3", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("myThreadId", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("throwTo", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            // ---- Phase 5: Data.* PrimOps (genuinely new) ----
            ("comparing", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("clamp", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("foldr'", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("foldl1", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("foldr1", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("maximumBy", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("minimumBy", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("fromString", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("shift", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("rotate", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("countLeadingZeros", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("countTrailingZeros", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("asProxyTypeOf", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("absurd", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
            ("vacuous", Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))),
        ];

        for (name, scheme) in ops {
            let def_id = DefId::new(next_id);
            env.register_value(def_id, Symbol::intern(name), scheme);
            next_id += 1;
        }

        // Register type-specialized show functions at fixed DefIds (10100+)
        env.register_value(
            DefId::new(10100),
            Symbol::intern("showInt"),
            Scheme::mono(Ty::fun(self.int_ty.clone(), self.string_ty.clone())),
        );
        env.register_value(
            DefId::new(10101),
            Symbol::intern("showDouble"),
            Scheme::mono(Ty::fun(self.float_ty.clone(), self.string_ty.clone())),
        );
        env.register_value(
            DefId::new(10102),
            Symbol::intern("showFloat"),
            Scheme::mono(Ty::fun(self.float_ty.clone(), self.string_ty.clone())),
        );
        env.register_value(
            DefId::new(10103),
            Symbol::intern("showBool"),
            Scheme::mono(Ty::fun(self.bool_ty.clone(), self.string_ty.clone())),
        );
        env.register_value(
            DefId::new(10104),
            Symbol::intern("showChar"),
            Scheme::mono(Ty::fun(self.char_ty.clone(), self.string_ty.clone())),
        );
        env.register_value(
            DefId::new(10105),
            Symbol::intern("showString"),
            Scheme::mono(Ty::fun(self.string_ty.clone(), self.string_ty.clone())),
        );
        {
            let a = TyVar::new_star(BUILTIN_TYVAR_A);
            let b = TyVar::new_star(BUILTIN_TYVAR_B);
            env.register_value(
                DefId::new(10106),
                Symbol::intern("showList"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::list(Ty::Var(a.clone())), self.string_ty.clone()),
                ),
            );
            env.register_value(
                DefId::new(10107),
                Symbol::intern("showMaybe"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::App(Box::new(Ty::Con(self.maybe_con.clone())), Box::new(Ty::Var(a.clone()))),
                        self.string_ty.clone(),
                    ),
                ),
            );
            env.register_value(
                DefId::new(10108),
                Symbol::intern("showEither"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::App(
                            Box::new(Ty::App(Box::new(Ty::Con(self.either_con.clone())), Box::new(Ty::Var(a.clone())))),
                            Box::new(Ty::Var(b.clone())),
                        ),
                        self.string_ty.clone(),
                    ),
                ),
            );
            env.register_value(
                DefId::new(10109),
                Symbol::intern("showTuple2"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Tuple(vec![Ty::Var(a), Ty::Var(b)]),
                        self.string_ty.clone(),
                    ),
                ),
            );
        }
        env.register_value(
            DefId::new(10110),
            Symbol::intern("showUnit"),
            Scheme::mono(Ty::fun(Ty::unit(), self.string_ty.clone())),
        );

        // Register character functions at fixed DefIds (10200+)
        env.register_value(
            DefId::new(10200),
            Symbol::intern("ord"),
            Scheme::mono(Ty::fun(self.char_ty.clone(), self.int_ty.clone())),
        );
        env.register_value(
            DefId::new(10201),
            Symbol::intern("chr"),
            Scheme::mono(Ty::fun(self.int_ty.clone(), self.char_ty.clone())),
        );
        let char_predicates: &[(usize, &str)] = &[
            (10202, "isAlpha"),
            (10203, "isAlphaNum"),
            (10204, "isAscii"),
            (10205, "isControl"),
            (10206, "isDigit"),
            (10207, "isHexDigit"),
            (10208, "isLetter"),
            (10209, "isLower"),
            (10210, "isNumber"),
            (10211, "isPrint"),
            (10212, "isPunctuation"),
            (10213, "isSpace"),
            (10214, "isSymbol"),
            (10215, "isUpper"),
        ];
        for &(id, name) in char_predicates {
            env.register_value(
                DefId::new(id),
                Symbol::intern(name),
                Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone())),
            );
        }
        env.register_value(
            DefId::new(10216),
            Symbol::intern("toLower"),
            Scheme::mono(Ty::fun(self.char_ty.clone(), self.char_ty.clone())),
        );
        env.register_value(
            DefId::new(10217),
            Symbol::intern("toUpper"),
            Scheme::mono(Ty::fun(self.char_ty.clone(), self.char_ty.clone())),
        );
        env.register_value(
            DefId::new(10218),
            Symbol::intern("digitToInt"),
            Scheme::mono(Ty::fun(self.char_ty.clone(), self.int_ty.clone())),
        );
        env.register_value(
            DefId::new(10219),
            Symbol::intern("intToDigit"),
            Scheme::mono(Ty::fun(self.int_ty.clone(), self.char_ty.clone())),
        );

        // Register Data.Text.IO functions at fixed DefIds (10300+).
        // Handle is Int, FilePath is String, Text is text_ty.
        {
            let io_text = Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(self.text_ty.clone()),
            );
            let io_unit = Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(Ty::unit()),
            );

            // readFile :: FilePath -> IO Text
            env.register_value(
                DefId::new(10300),
                Symbol::intern("Data.Text.IO.readFile"),
                Scheme::mono(Ty::fun(self.string_ty.clone(), io_text.clone())),
            );
            // writeFile :: FilePath -> Text -> IO ()
            env.register_value(
                DefId::new(10301),
                Symbol::intern("Data.Text.IO.writeFile"),
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::fun(self.text_ty.clone(), io_unit.clone()),
                )),
            );
            // appendFile :: FilePath -> Text -> IO ()
            env.register_value(
                DefId::new(10302),
                Symbol::intern("Data.Text.IO.appendFile"),
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::fun(self.text_ty.clone(), io_unit.clone()),
                )),
            );
            // hGetContents :: Handle -> IO Text
            env.register_value(
                DefId::new(10303),
                Symbol::intern("Data.Text.IO.hGetContents"),
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_text.clone())),
            );
            // hGetLine :: Handle -> IO Text
            env.register_value(
                DefId::new(10304),
                Symbol::intern("Data.Text.IO.hGetLine"),
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_text.clone())),
            );
            // hPutStr :: Handle -> Text -> IO ()
            env.register_value(
                DefId::new(10305),
                Symbol::intern("Data.Text.IO.hPutStr"),
                Scheme::mono(Ty::fun(
                    self.int_ty.clone(),
                    Ty::fun(self.text_ty.clone(), io_unit.clone()),
                )),
            );
            // hPutStrLn :: Handle -> Text -> IO ()
            env.register_value(
                DefId::new(10306),
                Symbol::intern("Data.Text.IO.hPutStrLn"),
                Scheme::mono(Ty::fun(
                    self.int_ty.clone(),
                    Ty::fun(self.text_ty.clone(), io_unit.clone()),
                )),
            );
            // putStr :: Text -> IO ()
            env.register_value(
                DefId::new(10307),
                Symbol::intern("Data.Text.IO.putStr"),
                Scheme::mono(Ty::fun(self.text_ty.clone(), io_unit.clone())),
            );
            // putStrLn :: Text -> IO ()
            env.register_value(
                DefId::new(10308),
                Symbol::intern("Data.Text.IO.putStrLn"),
                Scheme::mono(Ty::fun(self.text_ty.clone(), io_unit.clone())),
            );
            // getLine :: IO Text
            env.register_value(
                DefId::new(10309),
                Symbol::intern("Data.Text.IO.getLine"),
                Scheme::mono(io_text.clone()),
            );
            // getContents :: IO Text
            env.register_value(
                DefId::new(10310),
                Symbol::intern("Data.Text.IO.getContents"),
                Scheme::mono(io_text),
            );
        }

        // Numeric operations at fixed DefIds 10500-10507
        // These are registered at fixed DefIds to avoid sequential alignment issues
        {
            let int_pair = Ty::Tuple(vec![self.int_ty.clone(), self.int_ty.clone()]);
            let numeric_ops: &[(usize, &str, Scheme)] = &[
                (10500, "even", Scheme::mono(Ty::fun(self.int_ty.clone(), self.bool_ty.clone()))),
                (10501, "odd", Scheme::mono(Ty::fun(self.int_ty.clone(), self.bool_ty.clone()))),
                (10502, "gcd", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))),
                (10503, "lcm", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))),
                (10504, "quot", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))),
                (10505, "rem", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), self.int_ty.clone())))),
                (10506, "quotRem", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), int_pair.clone())))),
                (10507, "divMod", Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), int_pair)))),
            ];
            for (id, name, scheme) in numeric_ops {
                env.register_value(
                    DefId::new(*id),
                    Symbol::intern(name),
                    scheme.clone(),
                );
            }
        }

        // IORef operations at fixed DefIds 10400-10404
        {
            let a = TyVar::new_star(BUILTIN_TYVAR_A);
            let a_ty = Ty::Var(a.clone());
            let io_a = Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(a_ty.clone()),
            );
            let io_unit = Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(Ty::unit()),
            );

            // newIORef :: a -> IO a
            env.register_value(
                DefId::new(10400),
                Symbol::intern("newIORef"),
                Scheme::poly(vec![a.clone()], Ty::fun(a_ty.clone(), io_a.clone())),
            );
            // readIORef :: a -> IO a (IORef is opaque ptr, treated as `a`)
            env.register_value(
                DefId::new(10401),
                Symbol::intern("readIORef"),
                Scheme::poly(vec![a.clone()], Ty::fun(a_ty.clone(), io_a.clone())),
            );
            // writeIORef :: a -> a -> IO ()
            env.register_value(
                DefId::new(10402),
                Symbol::intern("writeIORef"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(a_ty.clone(), Ty::fun(a_ty.clone(), io_unit.clone())),
                ),
            );
            // modifyIORef :: a -> (a -> a) -> IO ()
            env.register_value(
                DefId::new(10403),
                Symbol::intern("modifyIORef"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        a_ty.clone(),
                        Ty::fun(Ty::fun(a_ty.clone(), a_ty.clone()), io_unit.clone()),
                    ),
                ),
            );
            // modifyIORef' :: a -> (a -> a) -> IO ()
            env.register_value(
                DefId::new(10404),
                Symbol::intern("modifyIORef'"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        a_ty.clone(),
                        Ty::fun(Ty::fun(a_ty.clone(), a_ty.clone()), io_unit),
                    ),
                ),
            );
        }

        // Data.Either extra operations at fixed DefIds 10600-10601
        {
            let a = TyVar::new_star(BUILTIN_TYVAR_A);
            let b = TyVar::new_star(BUILTIN_TYVAR_B);
            let either_ab = Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(self.either_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                )),
                Box::new(Ty::Var(b.clone())),
            );

            // fromLeft :: a -> Either a b -> a
            env.register_value(
                DefId::new(10600),
                Symbol::intern("fromLeft"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(either_ab.clone(), Ty::Var(a.clone()))),
                ),
            );
            // fromRight :: b -> Either a b -> b
            env.register_value(
                DefId::new(10601),
                Symbol::intern("fromRight"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(b.clone()), Ty::fun(either_ab, Ty::Var(b.clone()))),
                ),
            );
        }

        // Data.Maybe / Data.Either / guard at fixed DefIds 10610-10622
        // These override the sequential array registrations to fix DefId alignment.
        {
            let a = TyVar::new_star(BUILTIN_TYVAR_A);
            let b = TyVar::new_star(BUILTIN_TYVAR_B);
            let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
            let maybe_a = Ty::App(
                Box::new(Ty::Con(self.maybe_con.clone())),
                Box::new(Ty::Var(a.clone())),
            );
            let either_ab = Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(self.either_con.clone())),
                    Box::new(Ty::Var(a.clone())),
                )),
                Box::new(Ty::Var(b.clone())),
            );

            // fromMaybe :: a -> Maybe a -> a
            env.register_value(
                DefId::new(10610),
                Symbol::intern("fromMaybe"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(maybe_a.clone(), Ty::Var(a.clone()))),
                ),
            );
            // maybe :: b -> (a -> b) -> Maybe a -> b
            env.register_value(
                DefId::new(10611),
                Symbol::intern("maybe"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Var(b.clone()),
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(maybe_a.clone(), Ty::Var(b.clone())),
                        ),
                    ),
                ),
            );
            // listToMaybe :: [a] -> Maybe a
            env.register_value(
                DefId::new(10612),
                Symbol::intern("listToMaybe"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::List(Box::new(Ty::Var(a.clone()))), maybe_a.clone()),
                ),
            );
            // maybeToList :: Maybe a -> [a]
            env.register_value(
                DefId::new(10613),
                Symbol::intern("maybeToList"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(maybe_a.clone(), Ty::List(Box::new(Ty::Var(a.clone())))),
                ),
            );
            // catMaybes :: [Maybe a] -> [a]
            env.register_value(
                DefId::new(10614),
                Symbol::intern("catMaybes"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::List(Box::new(maybe_a.clone())),
                        Ty::List(Box::new(Ty::Var(a.clone()))),
                    ),
                ),
            );
            // mapMaybe :: (a -> Maybe b) -> [a] -> [b]
            env.register_value(
                DefId::new(10615),
                Symbol::intern("mapMaybe"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::App(
                            Box::new(Ty::Con(self.maybe_con.clone())),
                            Box::new(Ty::Var(b.clone())),
                        )),
                        Ty::fun(
                            Ty::List(Box::new(Ty::Var(a.clone()))),
                            Ty::List(Box::new(Ty::Var(b.clone()))),
                        ),
                    ),
                ),
            );
            // either :: (a -> c) -> (b -> c) -> Either a b -> c
            env.register_value(
                DefId::new(10616),
                Symbol::intern("either"),
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                        Ty::fun(
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                            Ty::fun(either_ab.clone(), Ty::Var(c.clone())),
                        ),
                    ),
                ),
            );
            // isLeft :: Either a b -> Bool
            env.register_value(
                DefId::new(10617),
                Symbol::intern("isLeft"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(either_ab.clone(), self.bool_ty.clone()),
                ),
            );
            // isRight :: Either a b -> Bool
            env.register_value(
                DefId::new(10618),
                Symbol::intern("isRight"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(either_ab.clone(), self.bool_ty.clone()),
                ),
            );
            // lefts :: [Either a b] -> [a]
            env.register_value(
                DefId::new(10619),
                Symbol::intern("lefts"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::List(Box::new(either_ab.clone())),
                        Ty::List(Box::new(Ty::Var(a.clone()))),
                    ),
                ),
            );
            // rights :: [Either a b] -> [b]
            env.register_value(
                DefId::new(10620),
                Symbol::intern("rights"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::List(Box::new(either_ab.clone())),
                        Ty::List(Box::new(Ty::Var(b.clone()))),
                    ),
                ),
            );
            // partitionEithers :: [Either a b] -> ([a], [b])
            env.register_value(
                DefId::new(10621),
                Symbol::intern("partitionEithers"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::List(Box::new(either_ab)),
                        Ty::Tuple(vec![
                            Ty::List(Box::new(Ty::Var(a.clone()))),
                            Ty::List(Box::new(Ty::Var(b.clone()))),
                        ]),
                    ),
                ),
            );
            // guard :: Bool -> [()]
            env.register_value(
                DefId::new(10622),
                Symbol::intern("guard"),
                Scheme::mono(Ty::fun(
                    self.bool_ty.clone(),
                    Ty::List(Box::new(Ty::unit())),
                )),
            );
        }

        // Data.List completions at fixed DefIds 10700-10706
        {
            let a = TyVar::new_star(BUILTIN_TYVAR_A);
            let b = TyVar::new_star(BUILTIN_TYVAR_B);
            let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
            let d = TyVar::new_star(BUILTIN_TYVAR_B + 2);

            // scanr :: (a -> b -> b) -> b -> [a] -> [b]
            env.register_value(
                DefId::new(10700),
                Symbol::intern("scanr"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(
                            Ty::List(Box::new(Ty::Var(a.clone()))),
                            Ty::List(Box::new(Ty::Var(b.clone()))),
                        )),
                    ),
                ),
            );
            // scanl1 :: (a -> a -> a) -> [a] -> [a]
            env.register_value(
                DefId::new(10701),
                Symbol::intern("scanl1"),
                {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                            Ty::fun(list_a.clone(), list_a),
                        ),
                    )
                },
            );
            // scanr1 :: (a -> a -> a) -> [a] -> [a]
            env.register_value(
                DefId::new(10702),
                Symbol::intern("scanr1"),
                {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                            Ty::fun(list_a.clone(), list_a),
                        ),
                    )
                },
            );
            // unfoldr :: (b -> Maybe (a, b)) -> b -> [a]
            env.register_value(
                DefId::new(10703),
                Symbol::intern("unfoldr"),
                {
                    let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                    let maybe_pair = Ty::App(
                        Box::new(Ty::Con(self.maybe_con.clone())),
                        Box::new(pair_ab),
                    );
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(b.clone()), maybe_pair),
                            Ty::fun(Ty::Var(b.clone()), list_a),
                        ),
                    )
                },
            );
            // intersect :: Eq a => [a] -> [a] -> [a]
            env.register_value(
                DefId::new(10704),
                Symbol::intern("intersect"),
                {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::qualified(
                        vec![a.clone()],
                        vec![eq_constraint(Ty::Var(a.clone()))],
                        Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                    )
                },
            );
            // zip3 :: [a] -> [b] -> [c] -> [(a, b, c)]
            env.register_value(
                DefId::new(10705),
                Symbol::intern("zip3"),
                {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                    let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                    let triple_ty = Ty::Tuple(vec![
                        Ty::Var(a.clone()),
                        Ty::Var(b.clone()),
                        Ty::Var(c.clone()),
                    ]);
                    let list_triple = Ty::List(Box::new(triple_ty));
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(list_a, Ty::fun(list_b, Ty::fun(list_c, list_triple))),
                    )
                },
            );
            // zipWith3 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
            env.register_value(
                DefId::new(10706),
                Symbol::intern("zipWith3"),
                {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                    let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                    let list_d = Ty::List(Box::new(Ty::Var(d.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone(), d.clone()],
                        Ty::fun(
                            Ty::fun(
                                Ty::Var(a.clone()),
                                Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(c.clone()), Ty::Var(d.clone()))),
                            ),
                            Ty::fun(list_a, Ty::fun(list_b, Ty::fun(list_c, list_d))),
                        ),
                    )
                },
            );
        }

        // E.16: List operations and Foldable basics at fixed DefIds 10800-10809
        {
            let a = TyVar::new_star(BUILTIN_TYVAR_A);
            let b = TyVar::new_star(BUILTIN_TYVAR_B);
            let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
            let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
            let int_ty = self.int_ty.clone();
            let bool_ty = self.bool_ty.clone();
            let ordering_ty = self.ordering_ty.clone();
            let maybe_int = Ty::App(
                Box::new(Ty::Con(self.maybe_con.clone())),
                Box::new(int_ty.clone()),
            );

            // elemIndex :: Eq a => a -> [a] -> Maybe Int
            env.register_value(
                DefId::new(10800),
                Symbol::intern("elemIndex"),
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), maybe_int.clone())),
                ),
            );
            // findIndex :: (a -> Bool) -> [a] -> Maybe Int
            env.register_value(
                DefId::new(10801),
                Symbol::intern("findIndex"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), bool_ty.clone()),
                        Ty::fun(list_a.clone(), maybe_int.clone()),
                    ),
                ),
            );
            // isPrefixOf :: Eq a => [a] -> [a] -> Bool
            env.register_value(
                DefId::new(10802),
                Symbol::intern("isPrefixOf"),
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), bool_ty.clone())),
                ),
            );
            // isSuffixOf :: Eq a => [a] -> [a] -> Bool
            env.register_value(
                DefId::new(10803),
                Symbol::intern("isSuffixOf"),
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), bool_ty.clone())),
                ),
            );
            // isInfixOf :: Eq a => [a] -> [a] -> Bool
            env.register_value(
                DefId::new(10804),
                Symbol::intern("isInfixOf"),
                Scheme::qualified(
                    vec![a.clone()],
                    vec![eq_constraint(Ty::Var(a.clone()))],
                    Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), bool_ty.clone())),
                ),
            );
            // tails :: [a] -> [[a]]
            env.register_value(
                DefId::new(10805),
                Symbol::intern("tails"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a.clone(), Ty::List(Box::new(list_a.clone()))),
                ),
            );
            // inits :: [a] -> [[a]]
            env.register_value(
                DefId::new(10806),
                Symbol::intern("inits"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(list_a.clone(), Ty::List(Box::new(list_a.clone()))),
                ),
            );
            // maximumBy :: (a -> a -> Ordering) -> [a] -> a
            env.register_value(
                DefId::new(10807),
                Symbol::intern("maximumBy"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), ordering_ty.clone())),
                        Ty::fun(list_a.clone(), Ty::Var(a.clone())),
                    ),
                ),
            );
            // minimumBy :: (a -> a -> Ordering) -> [a] -> a
            env.register_value(
                DefId::new(10808),
                Symbol::intern("minimumBy"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), ordering_ty.clone())),
                        Ty::fun(list_a.clone(), Ty::Var(a.clone())),
                    ),
                ),
            );
            // foldMap :: (a -> [b]) -> [a] -> [b] (simplified for list Foldable)
            env.register_value(
                DefId::new(10809),
                Symbol::intern("foldMap"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), list_b.clone()),
                        Ty::fun(list_a.clone(), list_b.clone()),
                    ),
                ),
            );
        }

        // E.17: Ordering ADT - compare at fixed DefId
        {
            // compare :: a -> a -> Ordering (polymorphic for derived Ord)
            env.register_value(
                DefId::new(10900),
                Symbol::intern("compare"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(a.clone()), self.ordering_ty.clone()),
                    ),
                ),
            );
        }

        // E.18: Monadic combinators at fixed DefIds
        {
            let a = TyVar::new_star(BUILTIN_TYVAR_A);
            let b = TyVar::new_star(BUILTIN_TYVAR_B);
            let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);

            // filterM :: (a -> IO Bool) -> [a] -> IO [a]
            let io_bool = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.bool_ty.clone()));
            let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
            let io_list_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_a.clone()));
            env.register_value(
                DefId::new(11000),
                Symbol::intern("filterM"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_bool),
                        Ty::fun(list_a.clone(), io_list_a),
                    ),
                ),
            );

            // foldM :: (b -> a -> IO b) -> b -> [a] -> IO b
            let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
            let list_a2 = Ty::List(Box::new(Ty::Var(a.clone())));
            env.register_value(
                DefId::new(11001),
                Symbol::intern("foldM"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), io_b.clone())),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a2.clone(), io_b.clone())),
                    ),
                ),
            );

            // foldM_ :: (b -> a -> IO b) -> b -> [a] -> IO ()
            let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
            env.register_value(
                DefId::new(11002),
                Symbol::intern("foldM_"),
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), io_b.clone())),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a2.clone(), io_unit.clone())),
                    ),
                ),
            );

            // replicateM :: Int -> IO a -> IO [a]
            let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
            let io_list_a2 = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_a.clone()));
            env.register_value(
                DefId::new(11003),
                Symbol::intern("replicateM"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(io_a.clone(), io_list_a2)),
                ),
            );

            // replicateM_ :: Int -> IO a -> IO ()
            env.register_value(
                DefId::new(11004),
                Symbol::intern("replicateM_"),
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.int_ty.clone(), Ty::fun(io_a, io_unit.clone())),
                ),
            );

            // zipWithM :: (a -> b -> IO c) -> [a] -> [b] -> IO [c]
            let io_c = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(c.clone())));
            let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
            let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
            let io_list_c = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_c));
            env.register_value(
                DefId::new(11005),
                Symbol::intern("zipWithM"),
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), io_c.clone())),
                        Ty::fun(list_a, Ty::fun(list_b.clone(), io_list_c)),
                    ),
                ),
            );

            // zipWithM_ :: (a -> b -> IO c) -> [a] -> [b] -> IO ()
            env.register_value(
                DefId::new(11006),
                Symbol::intern("zipWithM_"),
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), io_c)),
                        Ty::fun(list_a2, Ty::fun(list_b, io_unit)),
                    ),
                ),
            );
        }

        // E.19: System.FilePath + System.Directory at fixed DefIds
        {
            let io_unit = Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(Ty::unit()),
            );
            let io_list_string = Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(Ty::List(Box::new(self.string_ty.clone()))),
            );
            let tuple2_string_string = Ty::Tuple(vec![
                self.string_ty.clone(),
                self.string_ty.clone(),
            ]);

            // FilePath: String -> String (5 functions)
            for (id, name) in [
                (11100, "takeFileName"),
                (11101, "takeDirectory"),
                (11102, "takeExtension"),
                (11103, "dropExtension"),
                (11104, "takeBaseName"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(self.string_ty.clone(), self.string_ty.clone())),
                );
            }

            // FilePath: String -> String -> String
            env.register_value(
                DefId::new(11105),
                Symbol::intern("replaceExtension"),
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::fun(self.string_ty.clone(), self.string_ty.clone()),
                )),
            );

            // FilePath: String -> Bool (3 functions)
            for (id, name) in [
                (11106, "isAbsolute"),
                (11107, "isRelative"),
                (11108, "hasExtension"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(self.string_ty.clone(), self.bool_ty.clone())),
                );
            }

            // splitExtension: String -> (String, String)
            env.register_value(
                DefId::new(11109),
                Symbol::intern("splitExtension"),
                Scheme::mono(Ty::fun(self.string_ty.clone(), tuple2_string_string)),
            );

            // Directory: String -> IO () (2 functions)
            for (id, name) in [
                (11110, "setCurrentDirectory"),
                (11111, "removeDirectory"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(self.string_ty.clone(), io_unit.clone())),
                );
            }

            // Directory: String -> String -> IO () (2 functions)
            for (id, name) in [(11112, "renameFile"), (11113, "copyFile")] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(
                        self.string_ty.clone(),
                        Ty::fun(self.string_ty.clone(), io_unit.clone()),
                    )),
                );
            }

            // listDirectory: String -> IO [String]
            env.register_value(
                DefId::new(11114),
                Symbol::intern("listDirectory"),
                Scheme::mono(Ty::fun(self.string_ty.clone(), io_list_string)),
            );

            // </> : String -> String -> String (filepath combine)
            env.register_value(
                DefId::new(11115),
                Symbol::intern("</>"),
                Scheme::mono(Ty::fun(
                    self.string_ty.clone(),
                    Ty::fun(self.string_ty.clone(), self.string_ty.clone()),
                )),
            );
        }

        // E.20: Data.Text at fixed DefIds (fixes sequential array misalignment)
        {
            let text_ty = self.text_ty.clone();
            let char_ty = self.char_ty.clone();
            let int_ty = self.int_ty.clone();
            let bool_ty = self.bool_ty.clone();
            let string_ty = self.string_ty.clone();
            let bs_ty = self.bytestring_ty.clone();

            // Text (no args)
            env.register_value(
                DefId::new(11200),
                Symbol::intern("Data.Text.empty"),
                Scheme::mono(text_ty.clone()),
            );

            // Char -> Text
            env.register_value(
                DefId::new(11201),
                Symbol::intern("Data.Text.singleton"),
                Scheme::mono(Ty::fun(char_ty.clone(), text_ty.clone())),
            );

            // String -> Text
            env.register_value(
                DefId::new(11202),
                Symbol::intern("Data.Text.pack"),
                Scheme::mono(Ty::fun(string_ty.clone(), text_ty.clone())),
            );

            // Text -> String
            env.register_value(
                DefId::new(11203),
                Symbol::intern("Data.Text.unpack"),
                Scheme::mono(Ty::fun(text_ty.clone(), string_ty.clone())),
            );

            // Text -> Bool
            env.register_value(
                DefId::new(11204),
                Symbol::intern("Data.Text.null"),
                Scheme::mono(Ty::fun(text_ty.clone(), bool_ty.clone())),
            );

            // Text -> Int
            env.register_value(
                DefId::new(11205),
                Symbol::intern("Data.Text.length"),
                Scheme::mono(Ty::fun(text_ty.clone(), int_ty.clone())),
            );

            // Text -> Char (head, last)
            for (id, name) in [
                (11206, "Data.Text.head"),
                (11207, "Data.Text.last"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(text_ty.clone(), char_ty.clone())),
                );
            }

            // Text -> Text (tail, init, reverse, strip, toLower, toUpper, toCaseFold, toTitle)
            for (id, name) in [
                (11208, "Data.Text.tail"),
                (11209, "Data.Text.init"),
                (11212, "Data.Text.reverse"),
                (11220, "Data.Text.toLower"),
                (11221, "Data.Text.toUpper"),
                (11222, "Data.Text.toCaseFold"),
                (11223, "Data.Text.toTitle"),
                (11232, "Data.Text.strip"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(text_ty.clone(), text_ty.clone())),
                );
            }

            // Text -> Text -> Text (append, <>)
            for (id, name) in [
                (11210, "Data.Text.append"),
                (11211, "Data.Text.<>"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(text_ty.clone(), Ty::fun(text_ty.clone(), text_ty.clone()))),
                );
            }

            // Int -> Text -> Text (take, takeEnd, drop, dropEnd)
            for (id, name) in [
                (11213, "Data.Text.take"),
                (11214, "Data.Text.takeEnd"),
                (11215, "Data.Text.drop"),
                (11216, "Data.Text.dropEnd"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(int_ty.clone(), Ty::fun(text_ty.clone(), text_ty.clone()))),
                );
            }

            // Text -> Text -> Bool (isPrefixOf, isSuffixOf, isInfixOf)
            for (id, name) in [
                (11217, "Data.Text.isPrefixOf"),
                (11218, "Data.Text.isSuffixOf"),
                (11219, "Data.Text.isInfixOf"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(text_ty.clone(), Ty::fun(text_ty.clone(), bool_ty.clone()))),
                );
            }

            // (Char -> Char) -> Text -> Text (map)
            env.register_value(
                DefId::new(11224),
                Symbol::intern("Data.Text.map"),
                Scheme::mono(Ty::fun(
                    Ty::fun(char_ty.clone(), char_ty.clone()),
                    Ty::fun(text_ty.clone(), text_ty.clone()),
                )),
            );

            // Text -> Text -> Bool (eq, ==)
            for (id, name) in [
                (11225, "Data.Text.eq"),
                (11226, "Data.Text.=="),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(text_ty.clone(), Ty::fun(text_ty.clone(), bool_ty.clone()))),
                );
            }

            // Text -> Text -> Int (compare)
            env.register_value(
                DefId::new(11227),
                Symbol::intern("Data.Text.compare"),
                Scheme::mono(Ty::fun(text_ty.clone(), Ty::fun(text_ty.clone(), int_ty.clone()))),
            );

            // (Char -> Bool) -> Text -> Text (filter)
            env.register_value(
                DefId::new(11228),
                Symbol::intern("Data.Text.filter"),
                Scheme::mono(Ty::fun(
                    Ty::fun(char_ty.clone(), bool_ty.clone()),
                    Ty::fun(text_ty.clone(), text_ty.clone()),
                )),
            );

            // foldl' :: (a -> Char -> a) -> a -> Text -> a
            {
                let foldl_a = TyVar::new_star(BUILTIN_TYVAR_A);
                env.register_value(
                    DefId::new(11229),
                    Symbol::intern("Data.Text.foldl'"),
                    Scheme::poly(
                        vec![foldl_a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(foldl_a.clone()), Ty::fun(char_ty.clone(), Ty::Var(foldl_a.clone()))),
                            Ty::fun(Ty::Var(foldl_a.clone()), Ty::fun(text_ty.clone(), Ty::Var(foldl_a.clone()))),
                        ),
                    ),
                );
            }

            // [Text] -> Text (concat)
            env.register_value(
                DefId::new(11230),
                Symbol::intern("Data.Text.concat"),
                Scheme::mono(Ty::fun(Ty::List(Box::new(text_ty.clone())), text_ty.clone())),
            );

            // Text -> [Text] -> Text (intercalate)
            env.register_value(
                DefId::new(11231),
                Symbol::intern("Data.Text.intercalate"),
                Scheme::mono(Ty::fun(
                    text_ty.clone(),
                    Ty::fun(Ty::List(Box::new(text_ty.clone())), text_ty.clone()),
                )),
            );

            // Text -> [Text] (words, lines)
            for (id, name) in [
                (11233, "Data.Text.words"),
                (11234, "Data.Text.lines"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(text_ty.clone(), Ty::List(Box::new(text_ty.clone())))),
                );
            }

            // Text -> Text -> [Text] (splitOn)
            env.register_value(
                DefId::new(11235),
                Symbol::intern("Data.Text.splitOn"),
                Scheme::mono(Ty::fun(
                    text_ty.clone(),
                    Ty::fun(text_ty.clone(), Ty::List(Box::new(text_ty.clone()))),
                )),
            );

            // Text -> Text -> Text -> Text (replace)
            env.register_value(
                DefId::new(11236),
                Symbol::intern("Data.Text.replace"),
                Scheme::mono(Ty::fun(
                    text_ty.clone(),
                    Ty::fun(text_ty.clone(), Ty::fun(text_ty.clone(), text_ty.clone())),
                )),
            );

            // Data.Text.Encoding: Text -> ByteString
            env.register_value(
                DefId::new(11238),
                Symbol::intern("Data.Text.Encoding.encodeUtf8"),
                Scheme::mono(Ty::fun(text_ty.clone(), bs_ty.clone())),
            );

            // Data.Text.Encoding: ByteString -> Text
            env.register_value(
                DefId::new(11239),
                Symbol::intern("Data.Text.Encoding.decodeUtf8"),
                Scheme::mono(Ty::fun(bs_ty.clone(), text_ty.clone())),
            );

            // E.20: Data.ByteString at fixed DefIds
            let io_bs = Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(bs_ty.clone()),
            );
            let io_unit = Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(Ty::Tuple(vec![])),
            );

            // ByteString (no args)
            env.register_value(
                DefId::new(11250),
                Symbol::intern("Data.ByteString.empty"),
                Scheme::mono(bs_ty.clone()),
            );

            // Int -> ByteString (singleton)
            env.register_value(
                DefId::new(11251),
                Symbol::intern("Data.ByteString.singleton"),
                Scheme::mono(Ty::fun(int_ty.clone(), bs_ty.clone())),
            );

            // [Int] -> ByteString (pack)
            env.register_value(
                DefId::new(11252),
                Symbol::intern("Data.ByteString.pack"),
                Scheme::mono(Ty::fun(Ty::List(Box::new(int_ty.clone())), bs_ty.clone())),
            );

            // ByteString -> [Int] (unpack)
            env.register_value(
                DefId::new(11253),
                Symbol::intern("Data.ByteString.unpack"),
                Scheme::mono(Ty::fun(bs_ty.clone(), Ty::List(Box::new(int_ty.clone())))),
            );

            // ByteString -> Bool (null)
            env.register_value(
                DefId::new(11254),
                Symbol::intern("Data.ByteString.null"),
                Scheme::mono(Ty::fun(bs_ty.clone(), bool_ty.clone())),
            );

            // ByteString -> Int (length, head, last)
            for (id, name) in [
                (11255, "Data.ByteString.length"),
                (11256, "Data.ByteString.head"),
                (11257, "Data.ByteString.last"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(bs_ty.clone(), int_ty.clone())),
                );
            }

            // ByteString -> ByteString (tail, init, reverse)
            for (id, name) in [
                (11258, "Data.ByteString.tail"),
                (11259, "Data.ByteString.init"),
                (11265, "Data.ByteString.reverse"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(bs_ty.clone(), bs_ty.clone())),
                );
            }

            // ByteString -> ByteString -> ByteString (append)
            env.register_value(
                DefId::new(11260),
                Symbol::intern("Data.ByteString.append"),
                Scheme::mono(Ty::fun(bs_ty.clone(), Ty::fun(bs_ty.clone(), bs_ty.clone()))),
            );

            // Int -> ByteString -> ByteString (cons)
            env.register_value(
                DefId::new(11261),
                Symbol::intern("Data.ByteString.cons"),
                Scheme::mono(Ty::fun(int_ty.clone(), Ty::fun(bs_ty.clone(), bs_ty.clone()))),
            );

            // ByteString -> Int -> ByteString (snoc)
            env.register_value(
                DefId::new(11262),
                Symbol::intern("Data.ByteString.snoc"),
                Scheme::mono(Ty::fun(bs_ty.clone(), Ty::fun(int_ty.clone(), bs_ty.clone()))),
            );

            // Int -> ByteString -> ByteString (take, drop)
            for (id, name) in [
                (11263, "Data.ByteString.take"),
                (11264, "Data.ByteString.drop"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(int_ty.clone(), Ty::fun(bs_ty.clone(), bs_ty.clone()))),
                );
            }

            // Int -> ByteString -> Bool (elem)
            env.register_value(
                DefId::new(11266),
                Symbol::intern("Data.ByteString.elem"),
                Scheme::mono(Ty::fun(int_ty.clone(), Ty::fun(bs_ty.clone(), bool_ty.clone()))),
            );

            // ByteString -> Int -> Int (index)
            env.register_value(
                DefId::new(11267),
                Symbol::intern("Data.ByteString.index"),
                Scheme::mono(Ty::fun(bs_ty.clone(), Ty::fun(int_ty.clone(), int_ty.clone()))),
            );

            // ByteString -> ByteString -> Bool (eq, isPrefixOf, isSuffixOf)
            for (id, name) in [
                (11268, "Data.ByteString.eq"),
                (11270, "Data.ByteString.isPrefixOf"),
                (11271, "Data.ByteString.isSuffixOf"),
            ] {
                env.register_value(
                    DefId::new(id),
                    Symbol::intern(name),
                    Scheme::mono(Ty::fun(bs_ty.clone(), Ty::fun(bs_ty.clone(), bool_ty.clone()))),
                );
            }

            // ByteString -> ByteString -> Int (compare)
            env.register_value(
                DefId::new(11269),
                Symbol::intern("Data.ByteString.compare"),
                Scheme::mono(Ty::fun(bs_ty.clone(), Ty::fun(bs_ty.clone(), int_ty.clone()))),
            );

            // String -> IO ByteString (readFile)
            env.register_value(
                DefId::new(11272),
                Symbol::intern("Data.ByteString.readFile"),
                Scheme::mono(Ty::fun(string_ty.clone(), io_bs)),
            );

            // String -> ByteString -> IO () (writeFile)
            env.register_value(
                DefId::new(11273),
                Symbol::intern("Data.ByteString.writeFile"),
                Scheme::mono(Ty::fun(string_ty.clone(), Ty::fun(bs_ty.clone(), io_unit))),
            );
        }

        // Register transformer types and operations at fixed DefIds (10000+)
        self.register_transformer_ops(env);

        // Register MTL typeclasses and instances for cross-transformer operations
        self.register_mtl_classes(env);
    }

    /// Register monad transformer types and operations.
    ///
    /// These use fixed DefIds in the 10000+ range to avoid conflicts
    /// with the sequential allocation used by `register_primitive_ops`.
    fn register_transformer_ops(&self, env: &mut TypeEnv) {
        let a = TyVar::new_star(BUILTIN_TYVAR_A);
        let b = TyVar::new_star(BUILTIN_TYVAR_B);
        let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
        let m = TyVar::new(BUILTIN_TYVAR_M, m_kind.clone());
        let r_var = TyVar::new_star(BUILTIN_TYVAR_R);
        let s_var = TyVar::new_star(BUILTIN_TYVAR_S);

        // IO type for IO-specific signatures
        let io_ty = Ty::Con(self.io_con.clone());

        // Helper: m a
        let ma = |m: &TyVar, a: &TyVar| {
            Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())))
        };

        // === Identity (DefIds 10000-10006) ===
        // Identity :: a -> Identity a (newtype constructor)
        let identity_con = TyCon::new(Symbol::intern("Identity"), m_kind.clone());
        let identity_a = Ty::App(
            Box::new(Ty::Con(identity_con.clone())),
            Box::new(Ty::Var(a.clone())),
        );

        env.register_value(
            DefId::new(10000),
            Symbol::intern("Identity"),
            Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), identity_a.clone())),
        );
        env.register_value(
            DefId::new(10001),
            Symbol::intern("runIdentity"),
            Scheme::poly(vec![a.clone()], Ty::fun(identity_a, Ty::Var(a.clone()))),
        );

        // === MonadTrans / MonadIO (DefIds 10010-10012) ===
        // lift :: (MonadTrans t, Monad m) => m a -> t m a
        let t_kind = Kind::Arrow(
            Box::new(m_kind.clone()),
            Box::new(m_kind.clone()),
        );
        let t_var = TyVar::new(BUILTIN_TYVAR_T, t_kind);
        let t_m_a = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::Var(t_var.clone())),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::Var(a.clone())),
        );
        env.register_value(
            DefId::new(10010),
            Symbol::intern("lift"),
            Scheme::poly(
                vec![t_var.clone(), m.clone(), a.clone()],
                Ty::fun(ma(&m, &a), t_m_a),
            ),
        );

        // liftIO :: MonadIO m => IO a -> m a
        let io_a = Ty::App(Box::new(io_ty.clone()), Box::new(Ty::Var(a.clone())));
        env.register_value(
            DefId::new(10011),
            Symbol::intern("liftIO"),
            Scheme::poly(
                vec![m.clone(), a.clone()],
                Ty::fun(io_a.clone(), ma(&m, &a)),
            ),
        );

        // IO.liftIO is identity: IO a -> IO a
        env.register_value(
            DefId::new(10012),
            Symbol::intern("liftIO"),
            Scheme::poly(
                vec![a.clone()],
                Ty::fun(io_a.clone(), io_a.clone()),
            ),
        );

        // === ReaderT (DefIds 10020-10031) ===
        let reader_t_con = TyCon::new(
            Symbol::intern("ReaderT"),
            Kind::Arrow(
                Box::new(Kind::Star),
                Box::new(Kind::Arrow(
                    Box::new(m_kind.clone()),
                    Box::new(m_kind.clone()),
                )),
            ),
        );

        // ReaderT r m a ~ r -> m a (newtype)
        let reader_t_r_m_a = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(reader_t_con.clone())),
                    Box::new(Ty::Var(r_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::Var(a.clone())),
        );

        // ReaderT :: (r -> m a) -> ReaderT r m a
        env.register_value(
            DefId::new(10020),
            Symbol::intern("ReaderT"),
            Scheme::poly(
                vec![r_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    Ty::fun(Ty::Var(r_var.clone()), ma(&m, &a)),
                    reader_t_r_m_a.clone(),
                ),
            ),
        );

        // runReaderT :: ReaderT r m a -> r -> m a
        env.register_value(
            DefId::new(10021),
            Symbol::intern("runReaderT"),
            Scheme::poly(
                vec![r_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    reader_t_r_m_a.clone(),
                    Ty::fun(Ty::Var(r_var.clone()), ma(&m, &a)),
                ),
            ),
        );

        // ask :: Monad m => ReaderT r m r
        let reader_t_r_m_r = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(reader_t_con.clone())),
                    Box::new(Ty::Var(r_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::Var(r_var.clone())),
        );
        env.register_value(
            DefId::new(10029),
            Symbol::intern("ask"),
            Scheme::poly(vec![r_var.clone(), m.clone()], reader_t_r_m_r),
        );

        // asks :: Monad m => (r -> a) -> ReaderT r m a
        env.register_value(
            DefId::new(10030),
            Symbol::intern("asks"),
            Scheme::poly(
                vec![r_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    Ty::fun(Ty::Var(r_var.clone()), Ty::Var(a.clone())),
                    reader_t_r_m_a.clone(),
                ),
            ),
        );

        // local :: (r -> r) -> ReaderT r m a -> ReaderT r m a
        env.register_value(
            DefId::new(10031),
            Symbol::intern("local"),
            Scheme::poly(
                vec![r_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    Ty::fun(Ty::Var(r_var.clone()), Ty::Var(r_var.clone())),
                    Ty::fun(reader_t_r_m_a.clone(), reader_t_r_m_a.clone()),
                ),
            ),
        );

        // === StateT (DefIds 10040-10055) ===
        let state_t_con = TyCon::new(
            Symbol::intern("StateT"),
            Kind::Arrow(
                Box::new(Kind::Star),
                Box::new(Kind::Arrow(
                    Box::new(m_kind.clone()),
                    Box::new(m_kind.clone()),
                )),
            ),
        );

        // StateT s m a ~ s -> m (a, s) (newtype)
        let state_t_s_m_a = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(state_t_con.clone())),
                    Box::new(Ty::Var(s_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::Var(a.clone())),
        );

        let pair_a_s = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(s_var.clone())]);
        let m_pair_a_s = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(pair_a_s.clone()));

        // StateT :: (s -> m (a, s)) -> StateT s m a
        env.register_value(
            DefId::new(10040),
            Symbol::intern("StateT"),
            Scheme::poly(
                vec![s_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    Ty::fun(Ty::Var(s_var.clone()), m_pair_a_s.clone()),
                    state_t_s_m_a.clone(),
                ),
            ),
        );

        // runStateT :: StateT s m a -> s -> m (a, s)
        env.register_value(
            DefId::new(10041),
            Symbol::intern("runStateT"),
            Scheme::poly(
                vec![s_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    state_t_s_m_a.clone(),
                    Ty::fun(Ty::Var(s_var.clone()), m_pair_a_s),
                ),
            ),
        );

        // get :: Monad m => StateT s m s
        let state_t_s_m_s = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(state_t_con.clone())),
                    Box::new(Ty::Var(s_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::Var(s_var.clone())),
        );
        env.register_value(
            DefId::new(10049),
            Symbol::intern("get"),
            Scheme::poly(vec![s_var.clone(), m.clone()], state_t_s_m_s),
        );

        // put :: Monad m => s -> StateT s m ()
        let state_t_s_m_unit = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(state_t_con.clone())),
                    Box::new(Ty::Var(s_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::unit()),
        );
        env.register_value(
            DefId::new(10050),
            Symbol::intern("put"),
            Scheme::poly(
                vec![s_var.clone(), m.clone()],
                Ty::fun(Ty::Var(s_var.clone()), state_t_s_m_unit.clone()),
            ),
        );

        // modify :: Monad m => (s -> s) -> StateT s m ()
        env.register_value(
            DefId::new(10051),
            Symbol::intern("modify"),
            Scheme::poly(
                vec![s_var.clone(), m.clone()],
                Ty::fun(
                    Ty::fun(Ty::Var(s_var.clone()), Ty::Var(s_var.clone())),
                    state_t_s_m_unit,
                ),
            ),
        );

        // gets :: Monad m => (s -> a) -> StateT s m a
        env.register_value(
            DefId::new(10053),
            Symbol::intern("gets"),
            Scheme::poly(
                vec![s_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    Ty::fun(Ty::Var(s_var.clone()), Ty::Var(a.clone())),
                    state_t_s_m_a.clone(),
                ),
            ),
        );

        // evalStateT :: Monad m => StateT s m a -> s -> m a
        env.register_value(
            DefId::new(10054),
            Symbol::intern("evalStateT"),
            Scheme::poly(
                vec![s_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    state_t_s_m_a.clone(),
                    Ty::fun(Ty::Var(s_var.clone()), ma(&m, &a)),
                ),
            ),
        );

        // execStateT :: Monad m => StateT s m a -> s -> m s
        let m_s = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(s_var.clone())));
        env.register_value(
            DefId::new(10055),
            Symbol::intern("execStateT"),
            Scheme::poly(
                vec![s_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    state_t_s_m_a,
                    Ty::fun(Ty::Var(s_var.clone()), m_s),
                ),
            ),
        );

        // === ExceptT (DefIds 10060-10075) ===
        // ExceptT e m a ≅ m (Either e a)
        let e_var = TyVar::new_star(BUILTIN_TYVAR_E);
        let except_t_con = TyCon::new(
            Symbol::intern("ExceptT"),
            Kind::Arrow(
                Box::new(Kind::Star),
                Box::new(Kind::Arrow(
                    Box::new(m_kind.clone()),
                    Box::new(m_kind.clone()),
                )),
            ),
        );

        // ExceptT e m a
        let except_t_e_m_a = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(except_t_con.clone())),
                    Box::new(Ty::Var(e_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::Var(a.clone())),
        );

        // Either e a
        let either_con = TyCon::new(Symbol::intern("Either"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
        let either_e_a = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::Con(either_con.clone())),
                Box::new(Ty::Var(e_var.clone())),
            )),
            Box::new(Ty::Var(a.clone())),
        );
        let m_either_e_a = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(either_e_a.clone()));

        // ExceptT :: m (Either e a) -> ExceptT e m a
        env.register_value(
            DefId::new(10060),
            Symbol::intern("ExceptT"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone()],
                Ty::fun(m_either_e_a.clone(), except_t_e_m_a.clone()),
            ),
        );

        // runExceptT :: ExceptT e m a -> m (Either e a)
        env.register_value(
            DefId::new(10061),
            Symbol::intern("runExceptT"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone()],
                Ty::fun(except_t_e_m_a.clone(), m_either_e_a.clone()),
            ),
        );

        // throwE :: e -> ExceptT e m a
        env.register_value(
            DefId::new(10062),
            Symbol::intern("throwE"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone()],
                Ty::fun(Ty::Var(e_var.clone()), except_t_e_m_a.clone()),
            ),
        );

        // catchE :: ExceptT e m a -> (e -> ExceptT e m a) -> ExceptT e m a
        env.register_value(
            DefId::new(10063),
            Symbol::intern("catchE"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    except_t_e_m_a.clone(),
                    Ty::fun(
                        Ty::fun(Ty::Var(e_var.clone()), except_t_e_m_a.clone()),
                        except_t_e_m_a.clone(),
                    ),
                ),
            ),
        );

        // ExceptT e m ()
        let except_t_e_m_unit = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(except_t_con.clone())),
                    Box::new(Ty::Var(e_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::unit()),
        );

        // ExceptT.pure :: a -> ExceptT e m a (uses return pattern)
        env.register_value(
            DefId::new(10064),
            Symbol::intern("ExceptT.pure"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone()],
                Ty::fun(Ty::Var(a.clone()), except_t_e_m_a.clone()),
            ),
        );

        // ExceptT.>>= :: ExceptT e m a -> (a -> ExceptT e m b) -> ExceptT e m b
        let b_var = TyVar::new_star(BUILTIN_TYVAR_B);
        let except_t_e_m_b = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(except_t_con.clone())),
                    Box::new(Ty::Var(e_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::Var(b_var.clone())),
        );
        env.register_value(
            DefId::new(10065),
            Symbol::intern("ExceptT.>>="),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone(), b_var.clone()],
                Ty::fun(
                    except_t_e_m_a.clone(),
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), except_t_e_m_b.clone()),
                        except_t_e_m_b.clone(),
                    ),
                ),
            ),
        );

        // ExceptT.>> :: ExceptT e m a -> ExceptT e m b -> ExceptT e m b
        env.register_value(
            DefId::new(10066),
            Symbol::intern("ExceptT.>>"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone(), b_var.clone()],
                Ty::fun(
                    except_t_e_m_a.clone(),
                    Ty::fun(except_t_e_m_b.clone(), except_t_e_m_b.clone()),
                ),
            ),
        );

        // ExceptT.fmap :: (a -> b) -> ExceptT e m a -> ExceptT e m b
        env.register_value(
            DefId::new(10067),
            Symbol::intern("ExceptT.fmap"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone(), b_var.clone()],
                Ty::fun(
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(b_var.clone())),
                    Ty::fun(except_t_e_m_a.clone(), except_t_e_m_b.clone()),
                ),
            ),
        );

        // ExceptT.<*> :: ExceptT e m (a -> b) -> ExceptT e m a -> ExceptT e m b
        let except_t_e_m_a_to_b = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(except_t_con.clone())),
                    Box::new(Ty::Var(e_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::fun(Ty::Var(a.clone()), Ty::Var(b_var.clone()))),
        );
        env.register_value(
            DefId::new(10068),
            Symbol::intern("ExceptT.<*>"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone(), b_var.clone()],
                Ty::fun(
                    except_t_e_m_a_to_b,
                    Ty::fun(except_t_e_m_a.clone(), except_t_e_m_b.clone()),
                ),
            ),
        );

        // ExceptT.lift :: m a -> ExceptT e m a
        env.register_value(
            DefId::new(10069),
            Symbol::intern("ExceptT.lift"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone()],
                Ty::fun(ma(&m, &a), except_t_e_m_a.clone()),
            ),
        );

        // ExceptT.liftIO :: IO a -> ExceptT e m a
        env.register_value(
            DefId::new(10070),
            Symbol::intern("ExceptT.liftIO"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone()],
                Ty::fun(self.io_of(Ty::Var(a.clone())), except_t_e_m_a.clone()),
            ),
        );

        // === MonadError standard names (mtl-style aliases for throwE/catchE) ===
        // throwError :: e -> ExceptT e m a (alias for throwE)
        env.register_value(
            DefId::new(10071),
            Symbol::intern("throwError"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone()],
                Ty::fun(Ty::Var(e_var.clone()), except_t_e_m_a.clone()),
            ),
        );

        // catchError :: ExceptT e m a -> (e -> ExceptT e m a) -> ExceptT e m a (alias for catchE)
        env.register_value(
            DefId::new(10072),
            Symbol::intern("catchError"),
            Scheme::poly(
                vec![e_var.clone(), m.clone(), a.clone()],
                Ty::fun(
                    except_t_e_m_a.clone(),
                    Ty::fun(
                        Ty::fun(Ty::Var(e_var.clone()), except_t_e_m_a.clone()),
                        except_t_e_m_a.clone(),
                    ),
                ),
            ),
        );

        // === WriterT (DefIds 10080-10095) ===
        // WriterT w m a ≅ m (a, w)
        let w_var = TyVar::new_star(BUILTIN_TYVAR_W);
        let writer_t_con = TyCon::new(
            Symbol::intern("WriterT"),
            Kind::Arrow(
                Box::new(Kind::Star),
                Box::new(Kind::Arrow(
                    Box::new(m_kind.clone()),
                    Box::new(m_kind.clone()),
                )),
            ),
        );

        // WriterT w m a
        let writer_t_w_m_a = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(writer_t_con.clone())),
                    Box::new(Ty::Var(w_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::Var(a.clone())),
        );

        // (a, w) pair
        let pair_a_w = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(w_var.clone())]);
        let m_pair_a_w = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(pair_a_w.clone()));

        // WriterT :: m (a, w) -> WriterT w m a
        env.register_value(
            DefId::new(10080),
            Symbol::intern("WriterT"),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone()],
                Ty::fun(m_pair_a_w.clone(), writer_t_w_m_a.clone()),
            ),
        );

        // runWriterT :: WriterT w m a -> m (a, w)
        env.register_value(
            DefId::new(10081),
            Symbol::intern("runWriterT"),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone()],
                Ty::fun(writer_t_w_m_a.clone(), m_pair_a_w.clone()),
            ),
        );

        // tell :: w -> WriterT w m ()
        let writer_t_w_m_unit = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(writer_t_con.clone())),
                    Box::new(Ty::Var(w_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::unit()),
        );
        env.register_value(
            DefId::new(10082),
            Symbol::intern("tell"),
            Scheme::poly(
                vec![w_var.clone(), m.clone()],
                Ty::fun(Ty::Var(w_var.clone()), writer_t_w_m_unit.clone()),
            ),
        );

        // execWriterT :: WriterT w m a -> m w
        let m_w = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(w_var.clone())));
        env.register_value(
            DefId::new(10083),
            Symbol::intern("execWriterT"),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone()],
                Ty::fun(writer_t_w_m_a.clone(), m_w),
            ),
        );

        // WriterT.pure :: a -> WriterT w m a
        env.register_value(
            DefId::new(10084),
            Symbol::intern("WriterT.pure"),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone()],
                Ty::fun(Ty::Var(a.clone()), writer_t_w_m_a.clone()),
            ),
        );

        // WriterT.>>= :: WriterT w m a -> (a -> WriterT w m b) -> WriterT w m b
        let writer_t_w_m_b = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(writer_t_con.clone())),
                    Box::new(Ty::Var(w_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::Var(b_var.clone())),
        );
        env.register_value(
            DefId::new(10085),
            Symbol::intern("WriterT.>>="),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone(), b_var.clone()],
                Ty::fun(
                    writer_t_w_m_a.clone(),
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), writer_t_w_m_b.clone()),
                        writer_t_w_m_b.clone(),
                    ),
                ),
            ),
        );

        // WriterT.>> :: WriterT w m a -> WriterT w m b -> WriterT w m b
        env.register_value(
            DefId::new(10086),
            Symbol::intern("WriterT.>>"),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone(), b_var.clone()],
                Ty::fun(
                    writer_t_w_m_a.clone(),
                    Ty::fun(writer_t_w_m_b.clone(), writer_t_w_m_b.clone()),
                ),
            ),
        );

        // WriterT.fmap :: (a -> b) -> WriterT w m a -> WriterT w m b
        env.register_value(
            DefId::new(10087),
            Symbol::intern("WriterT.fmap"),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone(), b_var.clone()],
                Ty::fun(
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(b_var.clone())),
                    Ty::fun(writer_t_w_m_a.clone(), writer_t_w_m_b.clone()),
                ),
            ),
        );

        // WriterT.<*> :: WriterT w m (a -> b) -> WriterT w m a -> WriterT w m b
        let writer_t_w_m_a_to_b = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(writer_t_con.clone())),
                    Box::new(Ty::Var(w_var.clone())),
                )),
                Box::new(Ty::Var(m.clone())),
            )),
            Box::new(Ty::fun(Ty::Var(a.clone()), Ty::Var(b_var.clone()))),
        );
        env.register_value(
            DefId::new(10088),
            Symbol::intern("WriterT.<*>"),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone(), b_var.clone()],
                Ty::fun(
                    writer_t_w_m_a_to_b,
                    Ty::fun(writer_t_w_m_a.clone(), writer_t_w_m_b.clone()),
                ),
            ),
        );

        // WriterT.lift :: m a -> WriterT w m a
        env.register_value(
            DefId::new(10089),
            Symbol::intern("WriterT.lift"),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone()],
                Ty::fun(ma(&m, &a), writer_t_w_m_a.clone()),
            ),
        );

        // WriterT.liftIO :: IO a -> WriterT w m a
        env.register_value(
            DefId::new(10090),
            Symbol::intern("WriterT.liftIO"),
            Scheme::poly(
                vec![w_var.clone(), m.clone(), a.clone()],
                Ty::fun(self.io_of(Ty::Var(a.clone())), writer_t_w_m_a.clone()),
            ),
        );
    }

    /// Register dynamic tensor operations in the environment.
    ///
    /// This registers:
    /// - `toDynamic :: forall shape a. Tensor shape a -> DynTensor a`
    /// - `fromDynamic :: forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)`
    /// - `withDynShape :: forall a r. DynTensor a -> (forall shape. Tensor shape a -> r) -> r`
    /// - `dynShape :: forall a. DynTensor a -> [Int]`
    /// - `dynRank :: forall a. DynTensor a -> Int`
    /// - `MkShapeWitness :: forall shape. ShapeWitness shape` (data constructor)
    /// - `MkDynTensor :: forall shape a. Tensor shape a -> DynTensor a` (data constructor)
    pub fn register_dyn_tensor_ops(&self, env: &mut TypeEnv) {
        // Type variables for schemes
        let shape_var = TyVar::new(BUILTIN_TYVAR_SHAPE, Kind::nat_list());
        let a_var = TyVar::new_star(BUILTIN_TYVAR_A);
        let r_var = TyVar::new_star(BUILTIN_TYVAR_R);

        // Tensor shape a
        let tensor_shape_a = self.tensor_of(
            Ty::TyList(TyList::Var(shape_var.clone())),
            Ty::Var(a_var.clone()),
        );

        // DynTensor a
        let dyn_tensor_a = self.dyn_tensor_of(Ty::Var(a_var.clone()));

        // ShapeWitness shape
        let witness_shape = self.shape_witness_of(TyList::Var(shape_var.clone()));

        // Maybe (Tensor shape a)
        let maybe_tensor = self.maybe_of(tensor_shape_a.clone());

        // 1. toDynamic :: forall shape a. Tensor shape a -> DynTensor a
        let to_dynamic_ty = Ty::fun(tensor_shape_a.clone(), dyn_tensor_a.clone());
        let to_dynamic_scheme = Scheme::poly(vec![shape_var.clone(), a_var.clone()], to_dynamic_ty);
        env.register_value(
            DefId::new(BUILTIN_TO_DYNAMIC_ID),
            Symbol::intern("toDynamic"),
            to_dynamic_scheme,
        );

        // 2. fromDynamic :: forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)
        let from_dynamic_ty = Ty::fun(
            witness_shape.clone(),
            Ty::fun(dyn_tensor_a.clone(), maybe_tensor),
        );
        let from_dynamic_scheme =
            Scheme::poly(vec![shape_var.clone(), a_var.clone()], from_dynamic_ty);
        env.register_value(
            DefId::new(BUILTIN_FROM_DYNAMIC_ID),
            Symbol::intern("fromDynamic"),
            from_dynamic_scheme,
        );

        // 3. dynShape :: forall a. DynTensor a -> [Int]
        let dyn_shape_ty = Ty::fun(
            dyn_tensor_a.clone(),
            Ty::List(Box::new(self.int_ty.clone())),
        );
        let dyn_shape_scheme = Scheme::poly(vec![a_var.clone()], dyn_shape_ty);
        env.register_value(
            DefId::new(BUILTIN_DYN_SHAPE_ID),
            Symbol::intern("dynShape"),
            dyn_shape_scheme,
        );

        // 4. dynRank :: forall a. DynTensor a -> Int
        let dyn_rank_ty = Ty::fun(dyn_tensor_a.clone(), self.int_ty.clone());
        let dyn_rank_scheme = Scheme::poly(vec![a_var.clone()], dyn_rank_ty);
        env.register_value(
            DefId::new(BUILTIN_DYN_RANK_ID),
            Symbol::intern("dynRank"),
            dyn_rank_scheme,
        );

        // 5. withDynShape :: forall a r. DynTensor a -> (forall shape. Tensor shape a -> r) -> r
        // The continuation: forall shape. Tensor shape a -> r
        let inner_shape_var = TyVar::new(BUILTIN_TYVAR_SHAPE2, Kind::nat_list());
        let tensor_inner = self.tensor_of(
            Ty::TyList(TyList::Var(inner_shape_var.clone())),
            Ty::Var(a_var.clone()),
        );
        let continuation = Ty::Forall(
            vec![inner_shape_var],
            Box::new(Ty::fun(tensor_inner, Ty::Var(r_var.clone()))),
        );
        let with_dyn_shape_ty =
            Ty::fun(dyn_tensor_a, Ty::fun(continuation, Ty::Var(r_var.clone())));
        let with_dyn_shape_scheme = Scheme::poly(vec![a_var.clone(), r_var], with_dyn_shape_ty);
        env.register_value(
            DefId::new(BUILTIN_WITH_DYN_SHAPE_ID),
            Symbol::intern("withDynShape"),
            with_dyn_shape_scheme,
        );

        // 6. MkShapeWitness :: forall shape. ShapeWitness shape (data constructor)
        let mk_witness_scheme = Scheme::poly(vec![shape_var.clone()], witness_shape);
        env.register_data_con(
            DefId::new(BUILTIN_MK_SHAPE_WITNESS_ID),
            Symbol::intern("MkShapeWitness"),
            mk_witness_scheme,
        );

        // 7. MkDynTensor :: forall shape a. Tensor shape a -> DynTensor a (data constructor)
        let mk_dyn_tensor_ty = Ty::fun(tensor_shape_a, self.dyn_tensor_of(Ty::Var(a_var.clone())));
        let mk_dyn_tensor_scheme = Scheme::poly(vec![shape_var, a_var], mk_dyn_tensor_ty);
        env.register_data_con(
            DefId::new(BUILTIN_MK_DYN_TENSOR_ID),
            Symbol::intern("MkDynTensor"),
            mk_dyn_tensor_scheme,
        );
    }

    /// Register MTL-style typeclasses and instances for cross-transformer operations.
    ///
    /// This enables code like:
    /// ```haskell
    /// comp :: StateT Int (ReaderT String IO) Int
    /// comp = do
    ///     s <- get      -- MonadState operation (direct)
    ///     r <- ask      -- MonadReader operation (needs lifted instance)
    ///     return (s + length r)
    /// ```
    ///
    /// Registered typeclasses:
    /// - `MonadReader r m | m -> r` with methods: ask, asks, local
    /// - `MonadState s m | m -> s` with methods: get, put, modify, gets
    /// - `MonadError e m | m -> e` with methods: throwError, catchError
    /// - `MonadWriter w m | m -> w` with method: tell
    ///
    /// Also registers direct instances (e.g., `MonadReader r (ReaderT r m)`) and
    /// lifted instances (e.g., `MonadReader r m => MonadReader r (StateT s m)`).
    pub fn register_mtl_classes(&self, env: &mut TypeEnv) {
        use crate::env::{ClassInfo, FunDep, InstanceInfo};

        // Type variables
        let a = TyVar::new_star(BUILTIN_TYVAR_A);
        let r = TyVar::new_star(BUILTIN_TYVAR_R);
        let s = TyVar::new_star(BUILTIN_TYVAR_S);
        let e = TyVar::new_star(BUILTIN_TYVAR_E);
        let w = TyVar::new_star(BUILTIN_TYVAR_W);
        let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
        let m = TyVar::new(BUILTIN_TYVAR_M, m_kind.clone());

        // Type constructor symbols
        let reader_t_sym = Symbol::intern("ReaderT");
        let state_t_sym = Symbol::intern("StateT");
        let except_t_sym = Symbol::intern("ExceptT");
        let writer_t_sym = Symbol::intern("WriterT");

        // Class symbols
        let monad_reader_sym = Symbol::intern("MonadReader");
        let monad_state_sym = Symbol::intern("MonadState");
        let monad_error_sym = Symbol::intern("MonadError");
        let monad_writer_sym = Symbol::intern("MonadWriter");
        let monad_sym = Symbol::intern("Monad");
        let monoid_sym = Symbol::intern("Monoid");

        // Helper to build m a
        let ma = |m_var: &TyVar, a_var: &TyVar| -> Ty {
            Ty::App(
                Box::new(Ty::Var(m_var.clone())),
                Box::new(Ty::Var(a_var.clone())),
            )
        };

        // Helper to build transformer types: T x m a
        let transformer_type =
            |con_name: Symbol, x: &TyVar, m_var: &TyVar, a_var: &TyVar| -> Ty {
                let con = TyCon::new(
                    con_name,
                    Kind::Arrow(
                        Box::new(Kind::Star),
                        Box::new(Kind::Arrow(Box::new(m_kind.clone()), Box::new(m_kind.clone()))),
                    ),
                );
                Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::App(
                            Box::new(Ty::Con(con)),
                            Box::new(Ty::Var(x.clone())),
                        )),
                        Box::new(Ty::Var(m_var.clone())),
                    )),
                    Box::new(Ty::Var(a_var.clone())),
                )
            };

        // Helper to build just the transformer monad: T x m (without the final type parameter)
        let transformer_monad = |con_name: Symbol, x: &TyVar, m_var: &TyVar| -> Ty {
            let con = TyCon::new(
                con_name,
                Kind::Arrow(
                    Box::new(Kind::Star),
                    Box::new(Kind::Arrow(Box::new(m_kind.clone()), Box::new(m_kind.clone()))),
                ),
            );
            Ty::App(
                Box::new(Ty::App(
                    Box::new(Ty::Con(con)),
                    Box::new(Ty::Var(x.clone())),
                )),
                Box::new(Ty::Var(m_var.clone())),
            )
        };

        // =================================================================
        // MonadReader r m | m -> r
        // Methods: ask :: m r, asks :: (r -> a) -> m a, local :: (r -> r) -> m a -> m a
        // =================================================================
        let monad_reader_class = ClassInfo {
            name: monad_reader_sym,
            params: vec![r.clone(), m.clone()],
            fundeps: vec![FunDep {
                from: vec![1], // m
                to: vec![0],   // r
            }],
            supers: vec![monad_sym],
            methods: {
                let mut methods = rustc_hash::FxHashMap::default();
                // ask :: MonadReader r m => m r
                methods.insert(
                    Symbol::intern("ask"),
                    Scheme::qualified(
                        vec![r.clone(), m.clone()],
                        vec![Constraint::new_multi(
                            monad_reader_sym,
                            vec![Ty::Var(r.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        ma(&m, &r),
                    ),
                );
                // asks :: MonadReader r m => (r -> a) -> m a
                methods.insert(
                    Symbol::intern("asks"),
                    Scheme::qualified(
                        vec![r.clone(), m.clone(), a.clone()],
                        vec![Constraint::new_multi(
                            monad_reader_sym,
                            vec![Ty::Var(r.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        Ty::fun(
                            Ty::fun(Ty::Var(r.clone()), Ty::Var(a.clone())),
                            ma(&m, &a),
                        ),
                    ),
                );
                // local :: MonadReader r m => (r -> r) -> m a -> m a
                methods.insert(
                    Symbol::intern("local"),
                    Scheme::qualified(
                        vec![r.clone(), m.clone(), a.clone()],
                        vec![Constraint::new_multi(
                            monad_reader_sym,
                            vec![Ty::Var(r.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        Ty::fun(
                            Ty::fun(Ty::Var(r.clone()), Ty::Var(r.clone())),
                            Ty::fun(ma(&m, &a), ma(&m, &a)),
                        ),
                    ),
                );
                methods
            },
            assoc_types: vec![],
        };
        // Register methods by name AND by DefId to override the transformer-specific types.
        // This ensures that when code like `ask` is resolved to DefId 10029, the MTL-constrained
        // type is used instead of the naked transformer type.
        // DefIds: ask=10029, asks=10030, local=10031
        for (name, scheme) in &monad_reader_class.methods {
            let def_id = match name.as_str() {
                "ask" => DefId::new(10029),
                "asks" => DefId::new(10030),
                "local" => DefId::new(10031),
                _ => continue,
            };
            env.register_value(def_id, *name, scheme.clone());
        }
        env.register_class(monad_reader_class);

        // =================================================================
        // MonadState s m | m -> s
        // Methods: get :: m s, put :: s -> m (), modify :: (s -> s) -> m (), gets :: (s -> a) -> m a
        // =================================================================
        let monad_state_class = ClassInfo {
            name: monad_state_sym,
            params: vec![s.clone(), m.clone()],
            fundeps: vec![FunDep {
                from: vec![1], // m
                to: vec![0],   // s
            }],
            supers: vec![monad_sym],
            methods: {
                let mut methods = rustc_hash::FxHashMap::default();
                // get :: MonadState s m => m s
                methods.insert(
                    Symbol::intern("get"),
                    Scheme::qualified(
                        vec![s.clone(), m.clone()],
                        vec![Constraint::new_multi(
                            monad_state_sym,
                            vec![Ty::Var(s.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        ma(&m, &s),
                    ),
                );
                // put :: MonadState s m => s -> m ()
                methods.insert(
                    Symbol::intern("put"),
                    Scheme::qualified(
                        vec![s.clone(), m.clone()],
                        vec![Constraint::new_multi(
                            monad_state_sym,
                            vec![Ty::Var(s.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        Ty::fun(Ty::Var(s.clone()), Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::unit()))),
                    ),
                );
                // modify :: MonadState s m => (s -> s) -> m ()
                methods.insert(
                    Symbol::intern("modify"),
                    Scheme::qualified(
                        vec![s.clone(), m.clone()],
                        vec![Constraint::new_multi(
                            monad_state_sym,
                            vec![Ty::Var(s.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        Ty::fun(
                            Ty::fun(Ty::Var(s.clone()), Ty::Var(s.clone())),
                            Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::unit())),
                        ),
                    ),
                );
                // gets :: MonadState s m => (s -> a) -> m a
                methods.insert(
                    Symbol::intern("gets"),
                    Scheme::qualified(
                        vec![s.clone(), m.clone(), a.clone()],
                        vec![Constraint::new_multi(
                            monad_state_sym,
                            vec![Ty::Var(s.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        Ty::fun(
                            Ty::fun(Ty::Var(s.clone()), Ty::Var(a.clone())),
                            ma(&m, &a),
                        ),
                    ),
                );
                methods
            },
            assoc_types: vec![],
        };
        // Register methods by name AND by DefId to override the transformer-specific types.
        // DefIds: get=10049, put=10050, modify=10051, gets=10053
        for (name, scheme) in &monad_state_class.methods {
            let def_id = match name.as_str() {
                "get" => DefId::new(10049),
                "put" => DefId::new(10050),
                "modify" => DefId::new(10051),
                "gets" => DefId::new(10053),
                _ => continue,
            };
            env.register_value(def_id, *name, scheme.clone());
        }
        env.register_class(monad_state_class);

        // =================================================================
        // MonadError e m | m -> e
        // Methods: throwError :: e -> m a, catchError :: m a -> (e -> m a) -> m a
        // =================================================================
        let monad_error_class = ClassInfo {
            name: monad_error_sym,
            params: vec![e.clone(), m.clone()],
            fundeps: vec![FunDep {
                from: vec![1], // m
                to: vec![0],   // e
            }],
            supers: vec![monad_sym],
            methods: {
                let mut methods = rustc_hash::FxHashMap::default();
                // throwError :: MonadError e m => e -> m a
                methods.insert(
                    Symbol::intern("throwError"),
                    Scheme::qualified(
                        vec![e.clone(), m.clone(), a.clone()],
                        vec![Constraint::new_multi(
                            monad_error_sym,
                            vec![Ty::Var(e.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        Ty::fun(Ty::Var(e.clone()), ma(&m, &a)),
                    ),
                );
                // catchError :: MonadError e m => m a -> (e -> m a) -> m a
                methods.insert(
                    Symbol::intern("catchError"),
                    Scheme::qualified(
                        vec![e.clone(), m.clone(), a.clone()],
                        vec![Constraint::new_multi(
                            monad_error_sym,
                            vec![Ty::Var(e.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        Ty::fun(
                            ma(&m, &a),
                            Ty::fun(Ty::fun(Ty::Var(e.clone()), ma(&m, &a)), ma(&m, &a)),
                        ),
                    ),
                );
                methods
            },
            assoc_types: vec![],
        };
        // Register methods by name AND by DefId to override the transformer-specific types.
        // DefIds: throwError=10071, catchError=10072
        for (name, scheme) in &monad_error_class.methods {
            let def_id = match name.as_str() {
                "throwError" => DefId::new(10071),
                "catchError" => DefId::new(10072),
                _ => continue,
            };
            env.register_value(def_id, *name, scheme.clone());
        }
        env.register_class(monad_error_class);

        // =================================================================
        // MonadWriter w m | m -> w
        // Methods: tell :: w -> m ()
        // =================================================================
        let monad_writer_class = ClassInfo {
            name: monad_writer_sym,
            params: vec![w.clone(), m.clone()],
            fundeps: vec![FunDep {
                from: vec![1], // m
                to: vec![0],   // w
            }],
            supers: vec![monad_sym, monoid_sym],
            methods: {
                let mut methods = rustc_hash::FxHashMap::default();
                // tell :: MonadWriter w m => w -> m ()
                methods.insert(
                    Symbol::intern("tell"),
                    Scheme::qualified(
                        vec![w.clone(), m.clone()],
                        vec![Constraint::new_multi(
                            monad_writer_sym,
                            vec![Ty::Var(w.clone()), Ty::Var(m.clone())],
                            Span::default(),
                        )],
                        Ty::fun(Ty::Var(w.clone()), Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::unit()))),
                    ),
                );
                methods
            },
            assoc_types: vec![],
        };
        // Register methods by name AND by DefId to override the transformer-specific types.
        // DefIds: tell=10082
        for (name, scheme) in &monad_writer_class.methods {
            let def_id = match name.as_str() {
                "tell" => DefId::new(10082),
                _ => continue,
            };
            env.register_value(def_id, *name, scheme.clone());
        }
        env.register_class(monad_writer_class);

        // =================================================================
        // Direct Instances
        // =================================================================

        // instance Monad m => MonadReader r (ReaderT r m)
        let reader_t_r_m = transformer_monad(reader_t_sym, &r, &m);
        env.register_instance(InstanceInfo {
            class: monad_reader_sym,
            types: vec![Ty::Var(r.clone()), reader_t_r_m.clone()],
            context: vec![Constraint::new(monad_sym, Ty::Var(m.clone()), Span::default())],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance Monad m => MonadState s (StateT s m)
        let state_t_s_m = transformer_monad(state_t_sym, &s, &m);
        env.register_instance(InstanceInfo {
            class: monad_state_sym,
            types: vec![Ty::Var(s.clone()), state_t_s_m.clone()],
            context: vec![Constraint::new(monad_sym, Ty::Var(m.clone()), Span::default())],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance Monad m => MonadError e (ExceptT e m)
        let except_t_e_m = transformer_monad(except_t_sym, &e, &m);
        env.register_instance(InstanceInfo {
            class: monad_error_sym,
            types: vec![Ty::Var(e.clone()), except_t_e_m.clone()],
            context: vec![Constraint::new(monad_sym, Ty::Var(m.clone()), Span::default())],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance (Monoid w, Monad m) => MonadWriter w (WriterT w m)
        let writer_t_w_m = transformer_monad(writer_t_sym, &w, &m);
        env.register_instance(InstanceInfo {
            class: monad_writer_sym,
            types: vec![Ty::Var(w.clone()), writer_t_w_m.clone()],
            context: vec![
                Constraint::new(monoid_sym, Ty::Var(w.clone()), Span::default()),
                Constraint::new(monad_sym, Ty::Var(m.clone()), Span::default()),
            ],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // =================================================================
        // Lifted Instances (cross-transformer operations)
        // =================================================================

        // instance MonadReader r m => MonadReader r (StateT s m)
        env.register_instance(InstanceInfo {
            class: monad_reader_sym,
            types: vec![Ty::Var(r.clone()), state_t_s_m.clone()],
            context: vec![Constraint::new_multi(
                monad_reader_sym,
                vec![Ty::Var(r.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadState s m => MonadState s (ReaderT r m)
        env.register_instance(InstanceInfo {
            class: monad_state_sym,
            types: vec![Ty::Var(s.clone()), reader_t_r_m.clone()],
            context: vec![Constraint::new_multi(
                monad_state_sym,
                vec![Ty::Var(s.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadReader r m => MonadReader r (ExceptT e m)
        env.register_instance(InstanceInfo {
            class: monad_reader_sym,
            types: vec![Ty::Var(r.clone()), except_t_e_m.clone()],
            context: vec![Constraint::new_multi(
                monad_reader_sym,
                vec![Ty::Var(r.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadState s m => MonadState s (ExceptT e m)
        env.register_instance(InstanceInfo {
            class: monad_state_sym,
            types: vec![Ty::Var(s.clone()), except_t_e_m.clone()],
            context: vec![Constraint::new_multi(
                monad_state_sym,
                vec![Ty::Var(s.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadError e m => MonadError e (StateT s m)
        env.register_instance(InstanceInfo {
            class: monad_error_sym,
            types: vec![Ty::Var(e.clone()), state_t_s_m.clone()],
            context: vec![Constraint::new_multi(
                monad_error_sym,
                vec![Ty::Var(e.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadError e m => MonadError e (ReaderT r m)
        env.register_instance(InstanceInfo {
            class: monad_error_sym,
            types: vec![Ty::Var(e.clone()), reader_t_r_m.clone()],
            context: vec![Constraint::new_multi(
                monad_error_sym,
                vec![Ty::Var(e.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadReader r m => MonadReader r (WriterT w m)
        env.register_instance(InstanceInfo {
            class: monad_reader_sym,
            types: vec![Ty::Var(r.clone()), writer_t_w_m.clone()],
            context: vec![Constraint::new_multi(
                monad_reader_sym,
                vec![Ty::Var(r.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadState s m => MonadState s (WriterT w m)
        env.register_instance(InstanceInfo {
            class: monad_state_sym,
            types: vec![Ty::Var(s.clone()), writer_t_w_m.clone()],
            context: vec![Constraint::new_multi(
                monad_state_sym,
                vec![Ty::Var(s.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadError e m => MonadError e (WriterT w m)
        env.register_instance(InstanceInfo {
            class: monad_error_sym,
            types: vec![Ty::Var(e.clone()), writer_t_w_m.clone()],
            context: vec![Constraint::new_multi(
                monad_error_sym,
                vec![Ty::Var(e.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadWriter w m => MonadWriter w (StateT s m)
        env.register_instance(InstanceInfo {
            class: monad_writer_sym,
            types: vec![Ty::Var(w.clone()), state_t_s_m],
            context: vec![Constraint::new_multi(
                monad_writer_sym,
                vec![Ty::Var(w.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadWriter w m => MonadWriter w (ReaderT r m)
        env.register_instance(InstanceInfo {
            class: monad_writer_sym,
            types: vec![Ty::Var(w.clone()), reader_t_r_m],
            context: vec![Constraint::new_multi(
                monad_writer_sym,
                vec![Ty::Var(w.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

        // instance MonadWriter w m => MonadWriter w (ExceptT e m)
        env.register_instance(InstanceInfo {
            class: monad_writer_sym,
            types: vec![Ty::Var(w.clone()), except_t_e_m],
            context: vec![Constraint::new_multi(
                monad_writer_sym,
                vec![Ty::Var(w.clone()), Ty::Var(m.clone())],
                Span::default(),
            )],
            methods: rustc_hash::FxHashMap::default(),
            assoc_type_impls: vec![],
        });

    }
}

// DefId values for built-in constructors.
// These MUST match the order in bhc_lower::context::define_builtins!
//
// Lowering phase allocates:
// DefId layout (MUST match bhc-lower/src/context.rs):
// - Types: Int, Float, Double, Char, Bool, String, IO, Maybe, Either, Ordering,
//          NonEmpty, ExitCode, IOMode, BufferMode, Endo, Backwards, All, Any,
//          XdgDirectory, StdStream, TypeRep, SomeException, Permissions, Text (DefIds 0-23)
// - Constructors: True, False, Nothing, Just, Left, Right, LT, EQ, GT, [], :, (),
//                 (,), (,,), :|, Backwards, Endo, ExitSuccess, ExitFailure,
//                 SomeException, ReadMode, WriteMode, AppendMode, ReadWriteMode,
//                 NoBuffering, LineBuffering, BlockBuffering, XdgData, XdgConfig,
//                 XdgCache, CreatePipe, Inherit, UseHandle, NoStream, TypeRep,
//                 All, Any (DefIds 24-60)
// - Functions: +, -, *, /, ... (DefIds 61+)
//
// We skip the type DefIds (0-23) and start constructors at 24.
const BUILTIN_TYPE_COUNT: usize = 24; // All types registered in context.rs

const BUILTIN_TRUE_ID: usize = BUILTIN_TYPE_COUNT; // 24
const BUILTIN_FALSE_ID: usize = BUILTIN_TYPE_COUNT + 1; // 25
const BUILTIN_NOTHING_ID: usize = BUILTIN_TYPE_COUNT + 2; // 26
const BUILTIN_JUST_ID: usize = BUILTIN_TYPE_COUNT + 3; // 27
const BUILTIN_LEFT_ID: usize = BUILTIN_TYPE_COUNT + 4; // 28
const BUILTIN_RIGHT_ID: usize = BUILTIN_TYPE_COUNT + 5; // 29

const BUILTIN_CON_COUNT: usize = 6; // True, False, Nothing, Just, Left, Right

// Ordering constructors
const BUILTIN_LT_ID: usize = BUILTIN_TYPE_COUNT + BUILTIN_CON_COUNT; // 30
const BUILTIN_EQ_ID: usize = BUILTIN_LT_ID + 1; // 31
const BUILTIN_GT_ID: usize = BUILTIN_LT_ID + 2; // 32
const BUILTIN_ORDERING_COUNT: usize = 3; // LT, EQ, GT

// List and unit constructors
const BUILTIN_NIL_ID: usize = BUILTIN_TYPE_COUNT + BUILTIN_CON_COUNT + BUILTIN_ORDERING_COUNT; // 33
const BUILTIN_CONS_ID: usize = BUILTIN_NIL_ID + 1; // 34
const BUILTIN_UNIT_ID: usize = BUILTIN_NIL_ID + 2; // 35

// Count of list/unit constructors
const BUILTIN_LIST_UNIT_COUNT: usize = 3; // [], :, ()

// Tuple constructors - after list/unit constructors
const BUILTIN_PAIR_ID: usize = BUILTIN_NIL_ID + BUILTIN_LIST_UNIT_COUNT; // 36
const BUILTIN_TRIPLE_ID: usize = BUILTIN_PAIR_ID + 1; // 37

const BUILTIN_TUPLE_COUNT: usize = 2; // (,), (,,)

// Extra constructors after tuples (:|, Backwards, Endo, ExitSuccess, ExitFailure,
// SomeException, IOMode×4, BufferMode×3, XdgDirectory×3, StdStream×4, TypeRep, All, Any)
const BUILTIN_EXTRA_CON_COUNT: usize = 23; // All remaining constructors

// M9 Phase 5: Dynamic tensor operations - use reserved range to avoid conflicts
const BUILTIN_RESERVED_BASE: usize = 0xFFFF_0000;
const BUILTIN_TO_DYNAMIC_ID: usize = BUILTIN_RESERVED_BASE;
const BUILTIN_FROM_DYNAMIC_ID: usize = BUILTIN_RESERVED_BASE + 1;
const BUILTIN_DYN_SHAPE_ID: usize = BUILTIN_RESERVED_BASE + 2;
const BUILTIN_DYN_RANK_ID: usize = BUILTIN_RESERVED_BASE + 3;
const BUILTIN_WITH_DYN_SHAPE_ID: usize = BUILTIN_RESERVED_BASE + 4;
const BUILTIN_MK_SHAPE_WITNESS_ID: usize = BUILTIN_RESERVED_BASE + 5;
const BUILTIN_MK_DYN_TENSOR_ID: usize = BUILTIN_RESERVED_BASE + 6;

// Reserved TyVar IDs for built-in schemes
const BUILTIN_TYVAR_A: u32 = 0xFFFF_0000;
const BUILTIN_TYVAR_B: u32 = 0xFFFF_0001;
const BUILTIN_TYVAR_SHAPE: u32 = 0xFFFF_0002;
const BUILTIN_TYVAR_R: u32 = 0xFFFF_0003;
const BUILTIN_TYVAR_SHAPE2: u32 = 0xFFFF_0004;
pub(crate) const BUILTIN_TYVAR_M: u32 = 0xFFFF_0005; // For monad type constructor variable
pub(crate) const BUILTIN_TYVAR_F: u32 = 0xFFFF_0006; // For functor type constructor variable
const BUILTIN_TYVAR_S: u32 = 0xFFFF_0007; // For state type variable
const BUILTIN_TYVAR_T: u32 = 0xFFFF_0008; // For transformer type constructor variable
const BUILTIN_TYVAR_E: u32 = 0xFFFF_0009; // For error type variable (ExceptT)
const BUILTIN_TYVAR_W: u32 = 0xFFFF_000A; // For writer output type variable (WriterT)

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

    #[test]
    fn test_tensor_kind() {
        let builtins = Builtins::new();

        // Tensor has kind [Nat] -> * -> *
        assert_eq!(builtins.tensor_con.name, Symbol::intern("Tensor"));

        // Verify the kind structure
        match &builtins.tensor_con.kind {
            Kind::Arrow(arg, result) => {
                // First argument should be [Nat]
                match arg.as_ref() {
                    Kind::List(elem_kind) => {
                        assert!(matches!(elem_kind.as_ref(), Kind::Nat));
                    }
                    _ => panic!("expected [Nat] as first argument"),
                }
                // Result should be * -> *
                match result.as_ref() {
                    Kind::Arrow(elem, final_result) => {
                        assert!(elem.is_star());
                        assert!(final_result.is_star());
                    }
                    _ => panic!("expected * -> * as result"),
                }
            }
            _ => panic!("expected arrow kind"),
        }
    }

    #[test]
    fn test_tensor_of() {
        use bhc_types::TyList;

        let builtins = Builtins::new();

        // Create Tensor '[1024, 768] Float
        let shape = TyList::shape_from_dims(&[1024, 768]);
        let tensor_type = builtins.tensor_of(Ty::TyList(shape), builtins.float_ty.clone());

        // Verify structure
        match &tensor_type {
            Ty::App(f, elem) => {
                assert_eq!(**elem, builtins.float_ty);
                match f.as_ref() {
                    Ty::App(tensor, shape) => {
                        assert!(
                            matches!(tensor.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("Tensor"))
                        );
                        assert!(matches!(shape.as_ref(), Ty::TyList(_)));
                    }
                    _ => panic!("expected Tensor applied to shape"),
                }
            }
            _ => panic!("expected application"),
        }
    }

    // === M9 Phase 5: DynTensor tests ===

    #[test]
    fn test_dyn_tensor_kind() {
        let builtins = Builtins::new();

        // DynTensor has kind * -> *
        assert_eq!(builtins.dyn_tensor_con.name, Symbol::intern("DynTensor"));
        assert_eq!(builtins.dyn_tensor_con.kind, Kind::star_to_star());
    }

    #[test]
    fn test_shape_witness_kind() {
        let builtins = Builtins::new();

        // ShapeWitness has kind [Nat] -> *
        assert_eq!(
            builtins.shape_witness_con.name,
            Symbol::intern("ShapeWitness")
        );
        match &builtins.shape_witness_con.kind {
            Kind::Arrow(from, to) => {
                assert_eq!(**from, Kind::nat_list());
                assert_eq!(**to, Kind::Star);
            }
            _ => panic!("expected arrow kind"),
        }
    }

    #[test]
    fn test_dyn_tensor_of() {
        let builtins = Builtins::new();

        // Create DynTensor Float
        let dyn_type = builtins.dyn_tensor_of(builtins.float_ty.clone());

        match &dyn_type {
            Ty::App(f, elem) => {
                assert!(
                    matches!(f.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("DynTensor"))
                );
                assert_eq!(**elem, builtins.float_ty);
            }
            _ => panic!("expected application"),
        }
    }

    #[test]
    fn test_shape_witness_of() {
        let builtins = Builtins::new();

        // Create ShapeWitness '[1024, 768]
        let shape = TyList::shape_from_dims(&[1024, 768]);
        let witness = builtins.shape_witness_of(shape);

        match &witness {
            Ty::App(f, shape_arg) => {
                assert!(
                    matches!(f.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("ShapeWitness"))
                );
                assert!(matches!(shape_arg.as_ref(), Ty::TyList(_)));
            }
            _ => panic!("expected application"),
        }
    }

    #[test]
    fn test_register_dyn_tensor_ops() {
        let builtins = Builtins::new();
        let mut env = TypeEnv::new();
        builtins.register_dyn_tensor_ops(&mut env);

        // Check toDynamic is registered
        let to_dyn_scheme = env.lookup_local(Symbol::intern("toDynamic"));
        assert!(to_dyn_scheme.is_some(), "toDynamic should be registered");
        let to_dyn = to_dyn_scheme.unwrap();
        assert!(!to_dyn.is_mono(), "toDynamic should be polymorphic");
        assert_eq!(to_dyn.vars.len(), 2); // shape, a

        // Check fromDynamic is registered
        let from_dyn_scheme = env.lookup_local(Symbol::intern("fromDynamic"));
        assert!(
            from_dyn_scheme.is_some(),
            "fromDynamic should be registered"
        );
        let from_dyn = from_dyn_scheme.unwrap();
        assert_eq!(from_dyn.vars.len(), 2); // shape, a

        // Check dynShape is registered
        let dyn_shape_scheme = env.lookup_local(Symbol::intern("dynShape"));
        assert!(dyn_shape_scheme.is_some(), "dynShape should be registered");
        let dyn_shape = dyn_shape_scheme.unwrap();
        assert_eq!(dyn_shape.vars.len(), 1); // a

        // Check dynRank is registered
        let dyn_rank_scheme = env.lookup_local(Symbol::intern("dynRank"));
        assert!(dyn_rank_scheme.is_some(), "dynRank should be registered");

        // Check withDynShape is registered
        let with_dyn_scheme = env.lookup_local(Symbol::intern("withDynShape"));
        assert!(
            with_dyn_scheme.is_some(),
            "withDynShape should be registered"
        );

        // Check MkDynTensor data constructor is registered
        let mk_dyn = env.lookup_data_con(Symbol::intern("MkDynTensor"));
        assert!(mk_dyn.is_some(), "MkDynTensor should be registered");

        // Check MkShapeWitness data constructor is registered
        let mk_witness = env.lookup_data_con(Symbol::intern("MkShapeWitness"));
        assert!(mk_witness.is_some(), "MkShapeWitness should be registered");
    }

    #[test]
    fn test_to_dynamic_type_structure() {
        let builtins = Builtins::new();
        let mut env = TypeEnv::new();
        builtins.register_dyn_tensor_ops(&mut env);

        // toDynamic :: forall shape a. Tensor shape a -> DynTensor a
        let scheme = env.lookup_local(Symbol::intern("toDynamic")).unwrap();

        // Should have 2 type variables: shape and a
        assert_eq!(scheme.vars.len(), 2);
        // First var should have kind [Nat] (shape)
        assert_eq!(scheme.vars[0].kind, Kind::nat_list());
        // Second var should have kind * (a)
        assert_eq!(scheme.vars[1].kind, Kind::Star);

        // Body should be a function type
        assert!(scheme.ty.is_fun());
    }

    #[test]
    fn test_from_dynamic_type_structure() {
        let builtins = Builtins::new();
        let mut env = TypeEnv::new();
        builtins.register_dyn_tensor_ops(&mut env);

        // fromDynamic :: forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)
        let scheme = env.lookup_local(Symbol::intern("fromDynamic")).unwrap();

        // Should have 2 type variables
        assert_eq!(scheme.vars.len(), 2);

        // Body should be a function type (ShapeWitness shape -> ...)
        assert!(scheme.ty.is_fun());
    }
}
