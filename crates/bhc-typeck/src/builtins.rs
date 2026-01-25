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
            tensor_con,
            dyn_tensor_con,
            shape_witness_con,
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
                    Ty::fun(
                        Ty::Var(triple_b),
                        Ty::fun(Ty::Var(triple_c), triple_ty),
                    ),
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
        // Start after types (9) and constructors (11: 6 basic + 3 list/unit + 2 tuples) = 20
        // Order MUST match bhc_lower::context::define_builtins
        let mut next_id = BUILTIN_TYPE_COUNT + BUILTIN_CON_COUNT + BUILTIN_LIST_UNIT_COUNT + BUILTIN_TUPLE_COUNT;

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
            // a -> a -> Bool (for Ord types, we simplify to Int for now)
            Scheme::mono(Ty::fun(
                self.int_ty.clone(),
                Ty::fun(self.int_ty.clone(), self.bool_ty.clone()),
            ))
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
            // Monadic operators (list-specialized)
            (">>=", {
                // (>>=) :: [a] -> (a -> [b]) -> [b] (list monad)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(list_a, Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), list_b.clone()),
                        list_b,
                    )),
                )
            }),
            (">>", {
                // (>>) :: [a] -> [b] -> [b] (list monad)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(list_a, Ty::fun(list_b.clone(), list_b)),
                )
            }),
            ("=<<", {
                // (=<<) :: (a -> [b]) -> [a] -> [b] (list monad, flipped >>=)
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
            // Applicative/Functor operators (list-specialized)
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
            // Monadic operations (list-specialized)
            ("return", {
                // return :: a -> [a] (singleton list for list monad)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), list_a))
            }),
            ("pure", {
                // pure :: a -> [a] (same as return for lists)
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), list_a))
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
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone()))),
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
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
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
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_b),
                        Ty::fun(list_a, io_unit),
                    ),
                )
            }),
            ("forM", {
                // forM :: [a] -> (a -> IO b) -> IO [b] (flipped mapM)
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
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
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        list_a,
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), io_b), io_unit),
                    ),
                )
            }),
            ("sequence", {
                // sequence :: [IO a] -> IO [a] (IO-specialized)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let list_io_a = Ty::List(Box::new(io_a));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_list_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_a));
                Scheme::poly(vec![a.clone()], Ty::fun(list_io_a, io_list_a))
            }),
            ("sequence_", {
                // sequence_ :: [IO a] -> IO () (IO-specialized)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let list_io_a = Ty::List(Box::new(io_a));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(vec![a.clone()], Ty::fun(list_io_a, io_unit))
            }),
            ("when", {
                // when :: Bool -> IO () -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(self.bool_ty.clone(), Ty::fun(io_unit.clone(), io_unit)))
            }),
            ("unless", {
                // unless :: Bool -> IO () -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(self.bool_ty.clone(), Ty::fun(io_unit.clone(), io_unit)))
            }),
            ("void", {
                // void :: IO a -> IO ()
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(vec![a.clone()], Ty::fun(io_a, io_unit))
            }),
            ("liftIO", {
                // liftIO :: IO a -> IO a (identity for base IO)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(io_a.clone(), io_a))
            }),
            // Reader/State monad operations (simplified/polymorphic)
            ("ask", {
                // ask :: r -> r (simplified: Reader r r identity)
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))
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
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
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
                let io_pair = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(pair.clone()));
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
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
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
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
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
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(io_a.clone(), io_a))
            }),
            // Exception handling (simplified with String as exception type)
            ("catch", {
                // catch :: IO a -> (String -> IO a) -> IO a (simplified)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
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
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let either_result = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(self.string_ty.clone()),
                    )),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_either = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(either_result));
                Scheme::poly(vec![a.clone()], Ty::fun(io_a, io_either))
            }),
            ("throw", {
                // throw :: String -> a (simplified with String exception)
                Scheme::poly(vec![a.clone()], Ty::fun(self.string_ty.clone(), Ty::Var(a.clone())))
            }),
            ("throwIO", {
                // throwIO :: String -> IO a (simplified)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(self.string_ty.clone(), io_a))
            }),
            ("bracket", {
                // bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                let io_c = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(c.clone())));
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
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                let io_c = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(c.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(io_a, Ty::fun(io_b, Ty::fun(io_c.clone(), io_c))),
                )
            }),
            ("bracketOnError", {
                // bracketOnError :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                let io_c = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(c.clone())));
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
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(io_a.clone(), Ty::fun(io_b, io_a)),
                )
            }),
            ("onException", {
                // onException :: IO a -> IO b -> IO a
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(io_a.clone(), Ty::fun(io_b, io_a)),
                )
            }),
            ("handle", {
                // handle :: (String -> IO a) -> IO a -> IO a (flipped catch)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
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
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let maybe_b = Ty::App(Box::new(Ty::Con(self.maybe_con.clone())), Box::new(Ty::Var(b.clone())));
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
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let maybe_b = Ty::App(Box::new(Ty::Con(self.maybe_con.clone())), Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(self.string_ty.clone(), maybe_b),
                        Ty::fun(io_a.clone(), Ty::fun(Ty::fun(Ty::Var(b.clone()), io_a.clone()), io_a)),
                    ),
                )
            }),
            ("tryJust", {
                // tryJust :: (String -> Maybe b) -> IO a -> IO (Either b a)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let maybe_b = Ty::App(Box::new(Ty::Con(self.maybe_con.clone())), Box::new(Ty::Var(b.clone())));
                let either_ba = Ty::App(
                    Box::new(Ty::App(
                        Box::new(Ty::Con(self.either_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    )),
                    Box::new(Ty::Var(a.clone())),
                );
                let io_either = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(either_ba));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::fun(self.string_ty.clone(), maybe_b), Ty::fun(io_a, io_either)),
                )
            }),
            ("evaluate", {
                // evaluate :: a -> IO a (force evaluation)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), io_a))
            }),
            ("mask", {
                // mask :: ((IO a -> IO a) -> IO b) -> IO b (simplified)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::fun(Ty::fun(io_a.clone(), io_a), io_b.clone()), io_b),
                )
            }),
            ("mask_", {
                // mask_ :: IO a -> IO a
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(io_a.clone(), io_a))
            }),
            ("uninterruptibleMask", {
                // uninterruptibleMask :: ((IO a -> IO a) -> IO b) -> IO b
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::fun(Ty::fun(io_a.clone(), io_a), io_b.clone()), io_b),
                )
            }),
            ("uninterruptibleMask_", {
                // uninterruptibleMask_ :: IO a -> IO a
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(io_a.clone(), io_a))
            }),
            // IO operations (Handle abstracted as Int for now)
            ("hPutStr", {
                // hPutStr :: Handle -> String -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.string_ty.clone(), io_unit)))
            }),
            ("hPutStrLn", {
                // hPutStrLn :: Handle -> String -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.string_ty.clone(), io_unit)))
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
                let io_string = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_string))
            }),
            ("hGetContents", {
                // hGetContents :: Handle -> IO String
                let io_string = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.string_ty.clone()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_string))
            }),
            ("hClose", {
                // hClose :: Handle -> IO ()
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_unit))
            }),
            ("openFile", {
                // openFile :: FilePath -> IOMode -> IO Handle (FilePath = String, IOMode = Int, Handle = Int)
                let io_handle = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(self.string_ty.clone(), Ty::fun(self.int_ty.clone(), io_handle)))
            }),
            ("withFile", {
                // withFile :: FilePath -> IOMode -> (Handle -> IO r) -> IO r
                let io_r = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
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
                let io_bool = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.bool_ty.clone()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), io_bool))
            }),
            ("isEOF", {
                // isEOF :: IO Bool
                Scheme::mono(Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.bool_ty.clone())))
            }),
            ("getContents", {
                // getContents :: IO String
                Scheme::mono(Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(self.string_ty.clone())))
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
                Scheme::mono(Ty::fun(self.string_ty.clone(), Ty::fun(self.string_ty.clone(), io_unit)))
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
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))),
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
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
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
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
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
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone()))),
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
                let triple_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone()), Ty::Var(c.clone())]);
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
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(c.clone()), Ty::Var(d.clone())))),
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
                let triple_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone()), Ty::Var(c.clone())]);
                let list_triple = Ty::List(Box::new(triple_ty));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                let result = Ty::Tuple(vec![list_a, list_b, list_c]);
                Scheme::poly(vec![a.clone(), b.clone(), c.clone()], Ty::fun(list_triple, result))
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
                Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), Ty::fun(list_list_a, list_a)))
            }),
            ("intersperse", {
                // intersperse :: a -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)))
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
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
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
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
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
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))),
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
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.int_ty.clone())),
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
                Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(list_a.clone(), list_a)))
            }),
            ("genericDrop", {
                // genericDrop :: Integral i => i -> [a] -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(list_a.clone(), list_a)))
            }),
            ("genericSplitAt", {
                // genericSplitAt :: Integral i => i -> [a] -> ([a], [a])
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let pair = Ty::Tuple(vec![list_a.clone(), list_a.clone()]);
                Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(list_a, pair)))
            }),
            ("genericIndex", {
                // genericIndex :: Integral i => [a] -> i -> a
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::fun(self.int_ty.clone(), Ty::Var(a.clone()))))
            }),
            ("genericReplicate", {
                // genericReplicate :: Integral i => i -> a -> [a]
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), list_a)))
            }),
            // Prelude functions
            ("id", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
            ("const", Scheme::poly(
                vec![a.clone(), b.clone()],
                Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone()))),
            )),
            ("flip", {
                // flip :: (a -> b -> c) -> b -> a -> c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone()))),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone()))),
                    ),
                )
            }),
            ("error", Scheme::poly(vec![a.clone()], Ty::fun(self.string_ty.clone(), Ty::Var(a.clone())))),
            ("undefined", Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))),
            ("seq", Scheme::poly(
                vec![a.clone(), b.clone()],
                Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))),
            )),
            // Numeric operations
            ("fromInteger", Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))),
            ("fromRational", {
                // fromRational :: Rational -> Float (simplified)
                // Rational approximated as (Int, Int) tuple
                let rational = Ty::Tuple(vec![self.int_ty.clone(), self.int_ty.clone()]);
                Scheme::mono(Ty::fun(rational, self.float_ty.clone()))
            }),
            ("negate", Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))),
            ("abs", Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))),
            ("signum", Scheme::mono(Ty::fun(self.int_ty.clone(), self.int_ty.clone()))),
            ("sqrt", Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone()))),
            ("exp", Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone()))),
            ("log", Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone()))),
            ("sin", Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone()))),
            ("cos", Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone()))),
            ("tan", Scheme::mono(Ty::fun(self.float_ty.clone(), self.float_ty.clone()))),
            // Comparison
            ("compare", {
                // compare :: Ord a => a -> a -> Ordering (using Int as Ordering: -1, 0, 1)
                Scheme::qualified(
                    vec![a.clone()],
                    vec![ord_constraint(Ty::Var(a.clone()))],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.int_ty.clone())),
                )
            }),
            ("min", num_binop()),
            ("max", num_binop()),
            // Show
            ("show", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.string_ty.clone()))),
            // Boolean
            ("not", Scheme::mono(Ty::fun(self.bool_ty.clone(), self.bool_ty.clone()))),
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
            ("print", Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(Ty::unit()),
            )))),
            ("putStrLn", Scheme::mono(Ty::fun(self.string_ty.clone(), Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(Ty::unit()),
            )))),
            ("putStr", Scheme::mono(Ty::fun(self.string_ty.clone(), Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(Ty::unit()),
            )))),
            ("getLine", Scheme::mono(Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(self.string_ty.clone()),
            ))),
            ("readFile", Scheme::mono(Ty::fun(self.string_ty.clone(), Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(self.string_ty.clone()),
            )))),
            ("writeFile", Scheme::mono(Ty::fun(self.string_ty.clone(), Ty::fun(self.string_ty.clone(), Ty::App(
                Box::new(Ty::Con(self.io_con.clone())),
                Box::new(Ty::unit()),
            ))))),
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
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(pair_ty, Ty::Var(a.clone())))
            }),
            ("snd", {
                // snd :: (a, b) -> b
                let pair_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(pair_ty, Ty::Var(b.clone())))
            }),
            ("curry", {
                // curry :: ((a, b) -> c) -> a -> b -> c
                let c = TyVar::new_star(BUILTIN_TYVAR_B + 1);
                let pair_ty = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(pair_ty, Ty::Var(c.clone())),
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone()))),
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
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone()))),
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
            // Character functions
            ("ord", Scheme::mono(Ty::fun(self.char_ty.clone(), self.int_ty.clone()))),
            ("chr", Scheme::mono(Ty::fun(self.int_ty.clone(), self.char_ty.clone()))),
            ("isAlpha", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isAlphaNum", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isAscii", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isControl", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isDigit", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isHexDigit", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isLetter", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isLower", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isNumber", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isPrint", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isPunctuation", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isSpace", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isSymbol", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("isUpper", Scheme::mono(Ty::fun(self.char_ty.clone(), self.bool_ty.clone()))),
            ("toLower", Scheme::mono(Ty::fun(self.char_ty.clone(), self.char_ty.clone()))),
            ("toUpper", Scheme::mono(Ty::fun(self.char_ty.clone(), self.char_ty.clone()))),
            ("digitToInt", Scheme::mono(Ty::fun(self.char_ty.clone(), self.int_ty.clone()))),
            ("intToDigit", Scheme::mono(Ty::fun(self.int_ty.clone(), self.char_ty.clone()))),
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
                Scheme::poly(vec![a.clone()], Ty::fun(self.int_ty.clone(), Ty::Var(a.clone())))
            }),
            ("fromEnum", {
                // fromEnum :: a -> Int (polymorphic input)
                Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.int_ty.clone()))
            }),
            ("enumFrom", {
                // enumFrom :: Int -> [Int] (Int-specialized, [n..])
                let list_int = Ty::List(Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), list_int))
            }),
            ("enumFromThen", {
                // enumFromThen :: Int -> Int -> [Int] (Int-specialized, [n,m..])
                let list_int = Ty::List(Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), list_int)))
            }),
            ("enumFromTo", {
                // enumFromTo :: Int -> Int -> [Int] (Int-specialized, [n..m])
                let list_int = Ty::List(Box::new(self.int_ty.clone()));
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), list_int)))
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
                Scheme::poly(vec![a.clone()], Ty::fun(self.string_ty.clone(), Ty::Var(a.clone())))
            }),
            ("reads", {
                // reads :: String -> [(a, String)] (with remaining string)
                let pair = Ty::Tuple(vec![Ty::Var(a.clone()), self.string_ty.clone()]);
                let list_pair = Ty::List(Box::new(pair));
                Scheme::poly(vec![a.clone()], Ty::fun(self.string_ty.clone(), list_pair))
            }),
            ("readMaybe", {
                // readMaybe :: String -> Maybe a
                let maybe_a = Ty::App(Box::new(Ty::Con(self.maybe_con.clone())), Box::new(Ty::Var(a.clone())));
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
                Scheme::poly(vec![a.clone()], Ty::fun(self.string_ty.clone(), either_result))
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
            ("even", Scheme::mono(Ty::fun(self.int_ty.clone(), self.bool_ty.clone()))),
            ("odd", Scheme::mono(Ty::fun(self.int_ty.clone(), self.bool_ty.clone()))),
            ("gcd", num_binop()),
            ("lcm", num_binop()),
            ("quot", num_binop()),
            ("rem", num_binop()),
            ("quotRem", {
                // quotRem :: Integral a => a -> a -> (a, a)
                // Using Int for now (no type class constraints)
                let pair = Ty::Tuple(vec![self.int_ty.clone(), self.int_ty.clone()]);
                Scheme::mono(Ty::fun(self.int_ty.clone(), Ty::fun(self.int_ty.clone(), pair)))
            }),
            ("divMod", {
                // divMod :: Integral a => a -> a -> (a, a)
                // Using Int for now (no type class constraints)
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
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone()))),
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone()))),
                        ),
                    ),
                )
            }),
            ("fix", {
                // fix :: (a -> a) -> a
                Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())), Ty::Var(a.clone())),
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
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(either_ab, self.bool_ty.clone()))
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
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(either_ab, self.bool_ty.clone()))
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
                let maybe_a = Ty::App(Box::new(Ty::Con(self.maybe_con.clone())), Box::new(Ty::Var(a.clone())));
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
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
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
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), io_b),
                        Ty::fun(list_a, io_unit),
                    ),
                )
            }),
            ("for", {
                // for :: [a] -> (a -> IO b) -> IO [b] (flipped traverse)
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
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
                let io_b = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(b.clone())));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_unit = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::unit()));
                Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        list_a,
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), io_b), io_unit),
                    ),
                )
            }),
            ("sequenceA", {
                // sequenceA :: [IO a] -> IO [a] (same as sequence)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
                let list_io_a = Ty::List(Box::new(io_a));
                let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                let io_list_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(list_a));
                Scheme::poly(vec![a.clone()], Ty::fun(list_io_a, io_list_a))
            }),
            ("sequenceA_", {
                // sequenceA_ :: [IO a] -> IO () (same as sequence_)
                let io_a = Ty::App(Box::new(Ty::Con(self.io_con.clone())), Box::new(Ty::Var(a.clone())));
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
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), either_ab))
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
                Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(b.clone()), either_ab))
            }),
        ];

        for (name, scheme) in ops {
            let def_id = DefId::new(next_id);
            env.register_value(def_id, Symbol::intern(name), scheme);
            next_id += 1;
        }
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
        let to_dynamic_scheme = Scheme::poly(
            vec![shape_var.clone(), a_var.clone()],
            to_dynamic_ty,
        );
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
        let from_dynamic_scheme = Scheme::poly(
            vec![shape_var.clone(), a_var.clone()],
            from_dynamic_ty,
        );
        env.register_value(
            DefId::new(BUILTIN_FROM_DYNAMIC_ID),
            Symbol::intern("fromDynamic"),
            from_dynamic_scheme,
        );

        // 3. dynShape :: forall a. DynTensor a -> [Int]
        let dyn_shape_ty = Ty::fun(dyn_tensor_a.clone(), Ty::List(Box::new(self.int_ty.clone())));
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
        let with_dyn_shape_ty = Ty::fun(
            dyn_tensor_a,
            Ty::fun(continuation, Ty::Var(r_var.clone())),
        );
        let with_dyn_shape_scheme = Scheme::poly(
            vec![a_var.clone(), r_var],
            with_dyn_shape_ty,
        );
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
        let mk_dyn_tensor_scheme = Scheme::poly(
            vec![shape_var, a_var],
            mk_dyn_tensor_ty,
        );
        env.register_data_con(
            DefId::new(BUILTIN_MK_DYN_TENSOR_ID),
            Symbol::intern("MkDynTensor"),
            mk_dyn_tensor_scheme,
        );
    }
}

// DefId values for built-in constructors.
// These MUST match the order in bhc_lower::context::define_builtins!
//
// Lowering phase allocates:
// - Types: Int, Float, Double, Char, Bool, String, IO, Maybe, Either (DefIds 0-8)
// - Constructors: True, False, Nothing, Just, Left, Right (DefIds 9-14)
// - Functions: +, -, *, /, ... (DefIds 15+)
//
// We skip the type DefIds (0-8) and start constructors at 9.
const BUILTIN_TYPE_COUNT: usize = 9; // Int, Float, Double, Char, Bool, String, IO, Maybe, Either

const BUILTIN_TRUE_ID: usize = BUILTIN_TYPE_COUNT;     // 9
const BUILTIN_FALSE_ID: usize = BUILTIN_TYPE_COUNT + 1; // 10
const BUILTIN_NOTHING_ID: usize = BUILTIN_TYPE_COUNT + 2; // 11
const BUILTIN_JUST_ID: usize = BUILTIN_TYPE_COUNT + 3; // 12
const BUILTIN_LEFT_ID: usize = BUILTIN_TYPE_COUNT + 4; // 13
const BUILTIN_RIGHT_ID: usize = BUILTIN_TYPE_COUNT + 5; // 14

const BUILTIN_CON_COUNT: usize = 6; // True, False, Nothing, Just, Left, Right

// List and unit constructors - registered by both lowering and type checker
const BUILTIN_NIL_ID: usize = BUILTIN_TYPE_COUNT + BUILTIN_CON_COUNT; // 15
const BUILTIN_CONS_ID: usize = BUILTIN_TYPE_COUNT + BUILTIN_CON_COUNT + 1; // 16
const BUILTIN_UNIT_ID: usize = BUILTIN_TYPE_COUNT + BUILTIN_CON_COUNT + 2; // 17

// Count of list/unit constructors for calculating operator DefId start
const BUILTIN_LIST_UNIT_COUNT: usize = 3; // [], :, ()

// Tuple constructors - after list/unit constructors
const BUILTIN_PAIR_ID: usize = BUILTIN_TYPE_COUNT + BUILTIN_CON_COUNT + BUILTIN_LIST_UNIT_COUNT; // 18
const BUILTIN_TRIPLE_ID: usize = BUILTIN_PAIR_ID + 1; // 19

const BUILTIN_TUPLE_COUNT: usize = 2; // (,), (,,)

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
                        assert!(matches!(tensor.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("Tensor")));
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
        assert_eq!(builtins.shape_witness_con.name, Symbol::intern("ShapeWitness"));
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
                assert!(matches!(f.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("DynTensor")));
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
                assert!(matches!(f.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("ShapeWitness")));
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
        assert!(from_dyn_scheme.is_some(), "fromDynamic should be registered");
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
        assert!(with_dyn_scheme.is_some(), "withDynShape should be registered");

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
