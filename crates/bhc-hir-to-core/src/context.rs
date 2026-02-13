//! Lowering context for HIR to Core transformation.
//!
//! The `LowerContext` tracks state during the lowering process, including:
//! - Fresh variable generation
//! - Error collection
//! - Type environment
//! - Constructor metadata for ADTs

use bhc_core::{self as core, Bind, CoreModule, Var, VarId};
use bhc_hir::{DefId, Item, Module as HirModule, ValueDef};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Constraint, Scheme, Ty};
use rustc_hash::FxHashMap;

use crate::deriving::DerivingContext;
use crate::dictionary::{ClassInfo, ClassRegistry, DictContext, InstanceInfo};

/// Metadata about a data constructor.
///
/// This stores information needed to generate correct pattern matching code,
/// particularly the constructor's tag (position within its data type).
#[derive(Clone, Debug)]
pub struct ConstructorInfo {
    /// The name of the constructor.
    pub name: Symbol,
    /// The name of the data type this constructor belongs to.
    pub type_name: Symbol,
    /// The constructor's tag (0-based index within the data type's constructor list).
    pub tag: u32,
    /// The number of fields this constructor has.
    pub arity: u32,
    /// Field names for record constructors (in canonical order).
    /// Empty for positional constructors.
    pub field_names: Vec<Symbol>,
}

/// Metadata about a record field selector function.
///
/// This stores information needed to generate field access code.
#[derive(Clone, Debug)]
pub struct FieldSelectorInfo {
    /// The field name.
    pub field_name: Symbol,
    /// The constructor's DefId.
    pub con_id: DefId,
    /// The constructor's name.
    pub con_name: Symbol,
    /// The data type name.
    pub type_name: Symbol,
    /// The field's index within the constructor (0-based).
    pub field_index: usize,
    /// The total number of fields in the constructor.
    pub total_fields: usize,
}

use crate::expr::lower_expr;
use crate::{LowerError, LowerResult, TypeSchemeMap};

/// Context for the HIR to Core lowering pass.
pub struct LowerContext {
    /// Counter for generating fresh variable names.
    fresh_counter: u32,

    /// Mapping from HIR DefIds to Core variables.
    var_map: FxHashMap<DefId, Var>,

    /// Type schemes from the type checker (DefId -> Scheme).
    type_schemes: TypeSchemeMap,

    /// Constructor metadata (DefId -> ConstructorInfo).
    /// This maps constructor DefIds to their metadata including tag and type.
    constructor_map: FxHashMap<DefId, ConstructorInfo>,

    /// Field selector metadata (field name -> FieldSelectorInfo).
    /// This maps field names to their selector information for generating field access code.
    field_selector_map: FxHashMap<Symbol, FieldSelectorInfo>,

    /// Stack of in-scope dictionary variables.
    ///
    /// When lowering a constrained function like `f :: Num a => a -> a`,
    /// we push the dictionary variable `$dNum` onto this stack before lowering
    /// the body. When we encounter a reference to another constrained function
    /// that requires the same constraint, we can look up the dictionary here.
    ///
    /// Each entry maps constraint class names to their dictionary variables.
    dict_scope: Vec<FxHashMap<Symbol, Var>>,

    /// Registry of type classes and instances for dictionary construction.
    class_registry: ClassRegistry,

    /// Accumulated errors.
    errors: Vec<LowerError>,
}

impl LowerContext {
    /// Create a new lowering context.
    #[must_use]
    pub fn new() -> Self {
        let mut ctx = Self {
            // Start after builtin VarIds (builtins use 9-95, so start at 100 to be safe)
            fresh_counter: 100,
            var_map: FxHashMap::default(),
            type_schemes: FxHashMap::default(),
            constructor_map: FxHashMap::default(),
            field_selector_map: FxHashMap::default(),
            dict_scope: vec![FxHashMap::default()], // Start with empty root scope
            class_registry: ClassRegistry::new(),
            errors: Vec::new(),
        };
        ctx.register_builtins();
        ctx.register_builtin_constructors();
        ctx.register_builtin_classes();
        ctx
    }

    /// Set the type schemes from the type checker.
    pub fn set_type_schemes(&mut self, schemes: TypeSchemeMap) {
        self.type_schemes = schemes;
    }

    /// Look up the type for a definition from the type checker.
    ///
    /// Returns the monomorphic type from the scheme, or `Ty::Error` if not found.
    pub fn lookup_type(&self, def_id: DefId) -> Ty {
        self.type_schemes
            .get(&def_id)
            .map(|scheme| scheme.ty.clone())
            .unwrap_or(Ty::Error)
    }

    /// Look up the full type scheme for a definition, including constraints.
    ///
    /// Returns the complete scheme if found, or None if not found.
    #[must_use]
    pub fn lookup_scheme(&self, def_id: DefId) -> Option<&Scheme> {
        self.type_schemes.get(&def_id)
    }

    /// Register builtin constructor metadata.
    ///
    /// This sets up the constructor tags for builtin types (Bool, Maybe, Either, etc.)
    /// so pattern matching generates correct code.
    fn register_builtin_constructors(&mut self) {
        // Bool: False = 0, True = 1
        let bool_sym = Symbol::intern("Bool");
        self.constructor_map.insert(
            DefId::new(9),
            ConstructorInfo {
                name: Symbol::intern("True"),
                type_name: bool_sym,
                tag: 1,
                arity: 0,
                field_names: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(10),
            ConstructorInfo {
                name: Symbol::intern("False"),
                type_name: bool_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
            },
        );

        // Maybe: Nothing = 0, Just = 1
        let maybe_sym = Symbol::intern("Maybe");
        self.constructor_map.insert(
            DefId::new(11),
            ConstructorInfo {
                name: Symbol::intern("Nothing"),
                type_name: maybe_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(12),
            ConstructorInfo {
                name: Symbol::intern("Just"),
                type_name: maybe_sym,
                tag: 1,
                arity: 1,
                field_names: vec![],
            },
        );

        // Either: Left = 0, Right = 1
        let either_sym = Symbol::intern("Either");
        self.constructor_map.insert(
            DefId::new(13),
            ConstructorInfo {
                name: Symbol::intern("Left"),
                type_name: either_sym,
                tag: 0,
                arity: 1,
                field_names: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(14),
            ConstructorInfo {
                name: Symbol::intern("Right"),
                type_name: either_sym,
                tag: 1,
                arity: 1,
                field_names: vec![],
            },
        );

        // List: [] = 0, : = 1
        let list_sym = Symbol::intern("List");
        self.constructor_map.insert(
            DefId::new(15),
            ConstructorInfo {
                name: Symbol::intern("[]"),
                type_name: list_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
            },
        );
        self.constructor_map.insert(
            DefId::new(16),
            ConstructorInfo {
                name: Symbol::intern(":"),
                type_name: list_sym,
                tag: 1,
                arity: 2,
                field_names: vec![],
            },
        );

        // Unit: () = 0
        let unit_sym = Symbol::intern("Unit");
        self.constructor_map.insert(
            DefId::new(17),
            ConstructorInfo {
                name: Symbol::intern("()"),
                type_name: unit_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
            },
        );
    }

    /// Register builtin operators and constructors.
    ///
    /// DefIds must match the allocation order in bhc-lower and bhc-typeck.
    fn register_builtins(&mut self) {
        // DefIds 0-8: Types (not values, skip)
        // DefIds 9-14: Data constructors (True, False, Nothing, Just, Left, Right)
        let constructors = [
            (9, "True"),
            (10, "False"),
            (11, "Nothing"),
            (12, "Just"),
            (13, "Left"),
            (14, "Right"),
            (15, "[]"),
            (16, ":"),
            (17, "()"),
        ];

        for (id, name) in constructors {
            let def_id = DefId::new(id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(id),
                ty: Ty::Error, // Types resolved during evaluation
            };
            self.var_map.insert(def_id, var);
        }

        // DefIds 18+: Operators and functions
        // Order must match bhc-lower/src/context.rs define_builtins
        let operators = [
            // Arithmetic operators (18-26)
            "+",
            "-",
            "*",
            "/",
            "div",
            "mod",
            "^",
            "^^",
            "**",
            // Comparison operators (27-32)
            "==",
            "/=",
            "<",
            "<=",
            ">",
            ">=",
            // Boolean operators (33-34)
            "&&",
            "||",
            // List operators (35-37)
            ":",
            "++",
            "!!",
            // Function composition (38-39)
            ".",
            "$",
            // Monadic operators (40-41)
            ">>=",
            ">>",
            // Applicative operators (42-45)
            "<*>",
            "<$>",
            "*>",
            "<*",
            // Alternative operator (46)
            "<|>",
            // Monadic operations (47-48)
            "return",
            "pure",
            // List operations (49-62)
            "map",
            "filter",
            "foldr",
            "foldl",
            "foldl'",
            "concatMap",
            "head",
            "tail",
            "length",
            "null",
            "reverse",
            "take",
            "drop",
            "elem",
            // More list operations (63-70)
            "sum",
            "product",
            "and",
            "or",
            "any",
            "all",
            "maximum",
            "minimum",
            // Zip operations (71-72)
            "zip",
            "zipWith",
            // Prelude functions (73-79)
            "id",
            "const",
            "flip",
            "error",
            "undefined",
            "seq",
            // Numeric operations (80-88)
            "fromInteger",
            "fromRational",
            "negate",
            "abs",
            "signum",
            "sqrt",
            "exp",
            "log",
            "sin",
            "cos",
            "tan",
            // Comparison (89-90)
            "compare",
            "min",
            "max",
            // Show (91)
            "show",
            // Boolean (92-93)
            "not",
            "otherwise",
        ];

        let mut id = 18; // Start after constructors
        for name in operators {
            let def_id = DefId::new(id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(id),
                ty: Ty::Error, // Types resolved during evaluation
            };
            self.var_map.insert(def_id, var);
            id += 1;
        }

        // IO monad method implementations (DefIds 150-154)
        // These are referenced by dictionary construction for Functor/Applicative/Monad IO
        let io_methods: [(usize, &str); 5] = [
            (150, "fmap"),
            (151, "pure"),
            (152, "<*>"),
            (153, ">>="),
            (154, ">>"),
        ];
        for (method_id, name) in io_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }

        // Identity type + methods (DefIds 10000-10006)
        let identity_methods: [(usize, &str); 7] = [
            (10000, "Identity"),
            (10001, "runIdentity"),
            (10002, "Identity.fmap"),
            (10003, "Identity.pure"),
            (10004, "Identity.<*>"),
            (10005, "Identity.>>="),
            (10006, "Identity.>>"),
        ];
        for (method_id, name) in identity_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }

        // MonadTrans/MonadIO class methods + IO MonadIO instance (DefIds 10010-10012)
        let class_methods: [(usize, &str); 3] = [
            (10010, "lift"),
            (10011, "liftIO"),
            (10012, "IO.liftIO"),
        ];
        for (method_id, name) in class_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }

        // ReaderT type + instances + operations (DefIds 10020-10031)
        let reader_t_methods: [(usize, &str); 12] = [
            (10020, "ReaderT"),
            (10021, "runReaderT"),
            (10022, "ReaderT.fmap"),
            (10023, "ReaderT.pure"),
            (10024, "ReaderT.<*>"),
            (10025, "ReaderT.>>="),
            (10026, "ReaderT.>>"),
            (10027, "ReaderT.lift"),
            (10028, "ReaderT.liftIO"),
            (10029, "ask"),
            (10030, "asks"),
            (10031, "local"),
        ];
        for (method_id, name) in reader_t_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }

        // StateT type + instances + operations (DefIds 10040-10055)
        let state_t_methods: [(usize, &str); 15] = [
            (10040, "StateT"),
            (10041, "runStateT"),
            (10042, "StateT.fmap"),
            (10043, "StateT.pure"),
            (10044, "StateT.<*>"),
            (10045, "StateT.>>="),
            (10046, "StateT.>>"),
            (10047, "StateT.lift"),
            (10048, "StateT.liftIO"),
            (10049, "get"),
            (10050, "put"),
            (10051, "modify"),
            (10053, "gets"),
            (10054, "evalStateT"),
            (10055, "execStateT"),
        ];
        for (method_id, name) in state_t_methods {
            let def_id = DefId::new(method_id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(method_id),
                ty: Ty::Error,
            };
            self.var_map.insert(def_id, var);
        }
    }

    /// Register built-in type classes and their instances.
    ///
    /// This sets up the type class hierarchy (Eq, Ord, Num, etc.) and
    /// registers instances for built-in types (Int, Float, Bool, Char).
    fn register_builtin_classes(&mut self) {
        use bhc_types::{Kind, TyCon};

        // Helper to create a type constructor
        let make_ty = |name: &str| -> Ty { Ty::Con(TyCon::new(Symbol::intern(name), Kind::Star)) };

        // === Register Eq class ===
        // Methods: == (/=)
        // DefIds: == is 27, /= is 28
        let eq_class = ClassInfo {
            name: Symbol::intern("Eq"),
            methods: vec![Symbol::intern("=="), Symbol::intern("/=")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(eq_class);

        // === Register Ord class ===
        // Methods: compare, <, <=, >, >=, min, max
        // DefIds: < is 29, <= is 30, > is 31, >= is 32, compare is 89, min is 90, max is 91
        let ord_class = ClassInfo {
            name: Symbol::intern("Ord"),
            methods: vec![
                Symbol::intern("compare"),
                Symbol::intern("<"),
                Symbol::intern("<="),
                Symbol::intern(">"),
                Symbol::intern(">="),
                Symbol::intern("min"),
                Symbol::intern("max"),
            ],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Eq")],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(ord_class);

        // === Register Num class ===
        // Methods: +, -, *, negate, abs, signum, fromInteger
        // DefIds: + is 18, - is 19, * is 20, negate is 82, abs is 83, signum is 84, fromInteger is 80
        let num_class = ClassInfo {
            name: Symbol::intern("Num"),
            methods: vec![
                Symbol::intern("+"),
                Symbol::intern("-"),
                Symbol::intern("*"),
                Symbol::intern("negate"),
                Symbol::intern("abs"),
                Symbol::intern("signum"),
                Symbol::intern("fromInteger"),
            ],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(num_class);

        // === Register Fractional class ===
        // Methods: /, recip, fromRational
        // DefIds: / is 21, fromRational is 81
        let fractional_class = ClassInfo {
            name: Symbol::intern("Fractional"),
            methods: vec![
                Symbol::intern("/"),
                Symbol::intern("recip"),
                Symbol::intern("fromRational"),
            ],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Num")],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(fractional_class);

        // === Register Show class ===
        // Methods: show
        // DefIds: show is 92
        let show_class = ClassInfo {
            name: Symbol::intern("Show"),
            methods: vec![Symbol::intern("show")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(show_class);

        // === Register instances for Int ===
        let int_ty = make_ty("Int");
        self.register_builtin_instance("Eq", &int_ty, &[(27, "=="), (28, "/=")]);
        self.register_builtin_instance(
            "Ord",
            &int_ty,
            &[
                (89, "compare"),
                (29, "<"),
                (30, "<="),
                (31, ">"),
                (32, ">="),
                (90, "min"),
                (91, "max"),
            ],
        );
        self.register_builtin_instance(
            "Num",
            &int_ty,
            &[
                (18, "+"),
                (19, "-"),
                (20, "*"),
                (82, "negate"),
                (83, "abs"),
                (84, "signum"),
                (80, "fromInteger"),
            ],
        );
        self.register_builtin_instance("Show", &int_ty, &[(92, "show")]);

        // === Register instances for Float ===
        let float_ty = make_ty("Float");
        self.register_builtin_instance("Eq", &float_ty, &[(27, "=="), (28, "/=")]);
        self.register_builtin_instance(
            "Ord",
            &float_ty,
            &[
                (89, "compare"),
                (29, "<"),
                (30, "<="),
                (31, ">"),
                (32, ">="),
                (90, "min"),
                (91, "max"),
            ],
        );
        self.register_builtin_instance(
            "Num",
            &float_ty,
            &[
                (18, "+"),
                (19, "-"),
                (20, "*"),
                (82, "negate"),
                (83, "abs"),
                (84, "signum"),
                (80, "fromInteger"),
            ],
        );
        self.register_builtin_instance("Fractional", &float_ty, &[(21, "/"), (81, "fromRational")]);
        self.register_builtin_instance("Show", &float_ty, &[(92, "show")]);

        // === Register instances for Bool ===
        let bool_ty = make_ty("Bool");
        self.register_builtin_instance("Eq", &bool_ty, &[(27, "=="), (28, "/=")]);
        self.register_builtin_instance(
            "Ord",
            &bool_ty,
            &[
                (89, "compare"),
                (29, "<"),
                (30, "<="),
                (31, ">"),
                (32, ">="),
                (90, "min"),
                (91, "max"),
            ],
        );
        self.register_builtin_instance("Show", &bool_ty, &[(92, "show")]);

        // === Register instances for Char ===
        let char_ty = make_ty("Char");
        self.register_builtin_instance("Eq", &char_ty, &[(27, "=="), (28, "/=")]);
        self.register_builtin_instance(
            "Ord",
            &char_ty,
            &[
                (89, "compare"),
                (29, "<"),
                (30, "<="),
                (31, ">"),
                (32, ">="),
                (90, "min"),
                (91, "max"),
            ],
        );
        self.register_builtin_instance("Show", &char_ty, &[(92, "show")]);

        // === Register Functor class ===
        // Methods: fmap
        // fmap is also known as <$> (DefId 43)
        let functor_class = ClassInfo {
            name: Symbol::intern("Functor"),
            methods: vec![Symbol::intern("fmap")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(functor_class);

        // === Register Applicative class ===
        // Methods: pure, <*>
        // Superclass: Functor
        let applicative_class = ClassInfo {
            name: Symbol::intern("Applicative"),
            methods: vec![Symbol::intern("pure"), Symbol::intern("<*>")],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Functor")],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(applicative_class);

        // === Register Monad class ===
        // Methods: >>=, >>
        // Superclass: Applicative
        let monad_class = ClassInfo {
            name: Symbol::intern("Monad"),
            methods: vec![Symbol::intern(">>="), Symbol::intern(">>")],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Applicative")],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(monad_class);

        // === Register IO instances for Functor/Applicative/Monad ===
        // IO has kind * -> *, so we construct it as a type application
        let io_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
        let io_ty = Ty::Con(TyCon::new(Symbol::intern("IO"), io_kind));

        // Functor IO: fmap = DefId(150)
        self.register_builtin_instance("Functor", &io_ty, &[(150, "fmap")]);

        // Applicative IO: pure = DefId(151), <*> = DefId(152)
        // Superclass: Functor IO
        self.register_builtin_instance("Applicative", &io_ty, &[(151, "pure"), (152, "<*>")]);

        // Monad IO: >>= = DefId(153), >> = DefId(154)
        // Superclass: Applicative IO
        self.register_builtin_instance("Monad", &io_ty, &[(153, ">>="), (154, ">>")]);

        // === Register MonadTrans class ===
        // Methods: lift
        let monad_trans_class = ClassInfo {
            name: Symbol::intern("MonadTrans"),
            methods: vec![Symbol::intern("lift")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(monad_trans_class);

        // === Register MonadIO class ===
        // Methods: liftIO
        // Superclass: Monad
        let monad_io_class = ClassInfo {
            name: Symbol::intern("MonadIO"),
            methods: vec![Symbol::intern("liftIO")],
            method_types: FxHashMap::default(),
            superclasses: vec![Symbol::intern("Monad")],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        self.class_registry.register_class(monad_io_class);

        // MonadIO IO: liftIO = id (DefId 10012)
        self.register_builtin_instance("MonadIO", &io_ty, &[(10012, "liftIO")]);

        // === Register Identity type and instances ===
        let identity_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
        let identity_ty = Ty::Con(TyCon::new(Symbol::intern("Identity"), identity_kind));

        self.register_builtin_instance("Functor", &identity_ty, &[(10002, "fmap")]);
        self.register_builtin_instance(
            "Applicative",
            &identity_ty,
            &[(10003, "pure"), (10004, "<*>")],
        );
        self.register_builtin_instance("Monad", &identity_ty, &[(10005, ">>="), (10006, ">>")]);

        // === Register ReaderT instances ===
        // ReaderT r m is represented as a partially applied type constructor
        // For codegen, we match on the name "ReaderT" rather than the full type
        let reader_t_kind = Kind::Arrow(
            Box::new(Kind::Star),
            Box::new(Kind::Arrow(
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
            )),
        );
        let reader_t_ty = Ty::Con(TyCon::new(Symbol::intern("ReaderT"), reader_t_kind));

        self.register_builtin_instance("Functor", &reader_t_ty, &[(10022, "fmap")]);
        self.register_builtin_instance(
            "Applicative",
            &reader_t_ty,
            &[(10023, "pure"), (10024, "<*>")],
        );
        self.register_builtin_instance("Monad", &reader_t_ty, &[(10025, ">>="), (10026, ">>")]);
        self.register_builtin_instance("MonadTrans", &reader_t_ty, &[(10027, "lift")]);
        self.register_builtin_instance("MonadIO", &reader_t_ty, &[(10028, "liftIO")]);

        // === Register StateT instances ===
        let state_t_kind = Kind::Arrow(
            Box::new(Kind::Star),
            Box::new(Kind::Arrow(
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
                Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
            )),
        );
        let state_t_ty = Ty::Con(TyCon::new(Symbol::intern("StateT"), state_t_kind));

        self.register_builtin_instance("Functor", &state_t_ty, &[(10042, "fmap")]);
        self.register_builtin_instance(
            "Applicative",
            &state_t_ty,
            &[(10043, "pure"), (10044, "<*>")],
        );
        self.register_builtin_instance("Monad", &state_t_ty, &[(10045, ">>="), (10046, ">>")]);
        self.register_builtin_instance("MonadTrans", &state_t_ty, &[(10047, "lift")]);
        self.register_builtin_instance("MonadIO", &state_t_ty, &[(10048, "liftIO")]);
    }

    /// Helper to register a builtin instance with method DefIds.
    fn register_builtin_instance(
        &mut self,
        class_name: &str,
        instance_type: &Ty,
        methods: &[(usize, &str)],
    ) {
        let mut method_map = FxHashMap::default();
        for (def_id, name) in methods {
            method_map.insert(Symbol::intern(name), DefId::new(*def_id));
        }

        // For superclass instances, use the same instance type
        let class_info = self.class_registry.lookup_class(Symbol::intern(class_name));
        let superclass_instances = class_info
            .map(|c| {
                c.superclasses
                    .iter()
                    .map(|_| instance_type.clone())
                    .collect()
            })
            .unwrap_or_default();

        let instance_info = InstanceInfo {
            class: Symbol::intern(class_name),
            instance_types: vec![instance_type.clone()],
            methods: method_map,
            superclass_instances,
            assoc_type_impls: FxHashMap::default(),
        };

        self.class_registry.register_instance(instance_info);
    }

    /// Register builtins using DefIds from the lowering pass.
    ///
    /// This replaces the hardcoded DefIds with the actual DefIds assigned
    /// during AST-to-HIR lowering, ensuring consistency across passes.
    pub fn register_lowered_builtins(&mut self, defs: &crate::DefMap) {
        // Clear the existing hardcoded builtins
        self.var_map.clear();

        // Register all definitions from the lowering pass
        for (_def_id, def_info) in defs.iter() {
            let var = Var {
                name: def_info.name,
                id: VarId::new(def_info.id.index()),
                ty: Ty::Error, // Types resolved during evaluation
            };
            self.var_map.insert(def_info.id, var);
        }
    }

    /// Generate a fresh variable with the given base name.
    ///
    /// The name will be mangled with a counter to ensure uniqueness.
    /// For top-level bindings that need to preserve their original name,
    /// use `named_var` instead.
    pub fn fresh_var(&mut self, base: &str, ty: Ty, _span: Span) -> Var {
        let name = Symbol::intern(&format!("{}_{}", base, self.fresh_counter));
        self.fresh_counter += 1;
        Var {
            name,
            id: VarId::new(self.fresh_counter as usize),
            ty,
        }
    }

    /// Create a variable with a specific name (preserving the original name).
    ///
    /// Use this for top-level bindings where the name must be preserved
    /// for external visibility (e.g., `main`).
    pub fn named_var(&mut self, name: Symbol, ty: Ty) -> Var {
        self.fresh_counter += 1;
        Var {
            name,
            id: VarId::new(self.fresh_counter as usize),
            ty,
        }
    }

    /// Generate a fresh variable ID.
    pub fn fresh_id(&mut self) -> VarId {
        self.fresh_counter += 1;
        VarId::new(self.fresh_counter as usize)
    }

    /// Record an error.
    pub fn error(&mut self, err: LowerError) {
        self.errors.push(err);
    }

    /// Check if any errors have been recorded.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Take all recorded errors.
    pub fn take_errors(&mut self) -> Vec<LowerError> {
        std::mem::take(&mut self.errors)
    }

    /// Register a HIR definition with a Core variable.
    pub fn register_var(&mut self, def_id: DefId, var: Var) {
        self.var_map.insert(def_id, var);
    }

    /// Look up the Core variable for a HIR definition.
    #[must_use]
    pub fn lookup_var(&self, def_id: DefId) -> Option<&Var> {
        self.var_map.get(&def_id)
    }

    /// Register a data constructor with its metadata.
    pub fn register_constructor(&mut self, def_id: DefId, info: ConstructorInfo) {
        self.constructor_map.insert(def_id, info);
    }

    /// Look up constructor metadata for a given DefId.
    #[must_use]
    pub fn lookup_constructor(&self, def_id: DefId) -> Option<&ConstructorInfo> {
        self.constructor_map.get(&def_id)
    }

    /// Register a field selector function.
    pub fn register_field_selector(&mut self, field_name: Symbol, info: FieldSelectorInfo) {
        self.field_selector_map.insert(field_name, info);
    }

    /// Look up field selector metadata for a given field name.
    #[must_use]
    pub fn lookup_field_selector(&self, field_name: Symbol) -> Option<&FieldSelectorInfo> {
        self.field_selector_map.get(&field_name)
    }

    /// Push a new dictionary scope.
    pub fn push_dict_scope(&mut self) {
        self.dict_scope.push(FxHashMap::default());
    }

    /// Pop the current dictionary scope.
    pub fn pop_dict_scope(&mut self) {
        if self.dict_scope.len() > 1 {
            self.dict_scope.pop();
        }
    }

    /// Register a dictionary variable for a constraint in the current scope.
    pub fn register_dict(&mut self, class_name: Symbol, dict_var: Var) {
        if let Some(scope) = self.dict_scope.last_mut() {
            scope.insert(class_name, dict_var);
        }
    }

    /// Look up a dictionary variable for a constraint class.
    ///
    /// Searches from innermost to outermost scope.
    #[must_use]
    pub fn lookup_dict(&self, class_name: Symbol) -> Option<&Var> {
        for scope in self.dict_scope.iter().rev() {
            if let Some(var) = scope.get(&class_name) {
                return Some(var);
            }
        }
        None
    }

    /// Find an in-scope dictionary whose class has the given class as a superclass.
    ///
    /// For example, if we need an `Eq` dictionary but only have `Ord` in scope,
    /// this will find the `Ord` dictionary since `Ord` has `Eq` as a superclass.
    ///
    /// Returns `(subclass_name, dict_var)` if found.
    #[must_use]
    pub fn lookup_superclass_dict(&self, needed_class: Symbol) -> Option<(Symbol, &Var)> {
        for scope in self.dict_scope.iter().rev() {
            for (class_name, dict_var) in scope {
                // Check if this class has the needed class as a superclass
                if let Some(class_info) = self.class_registry.lookup_class(*class_name) {
                    if class_info.superclasses.contains(&needed_class) {
                        return Some((*class_name, dict_var));
                    }
                }
            }
        }
        None
    }

    /// Get all dictionary variables that match the given constraints.
    ///
    /// Returns dictionary variables in the same order as the constraints.
    pub fn lookup_dicts_for_constraints(&self, constraints: &[Constraint]) -> Vec<Option<Var>> {
        constraints
            .iter()
            .map(|c| self.lookup_dict(c.class).cloned())
            .collect()
    }

    /// Set the class registry for dictionary construction.
    pub fn set_class_registry(&mut self, registry: ClassRegistry) {
        self.class_registry = registry;
    }

    /// Get a reference to the class registry.
    #[must_use]
    pub fn class_registry(&self) -> &ClassRegistry {
        &self.class_registry
    }

    /// Try to resolve a dictionary for a constraint.
    ///
    /// Resolution order:
    /// 1. Direct lookup: Check if we have an in-scope dictionary for the class
    /// 2. Superclass extraction: Check if we have a dictionary for a subclass
    ///    (e.g., have Ord, need Eq - extract Eq from Ord)
    /// 3. Instance construction: For concrete types, construct from an instance
    ///
    /// Returns the dictionary expression and any bindings that need to be added.
    pub fn resolve_dictionary(
        &mut self,
        constraint: &Constraint,
        span: Span,
    ) -> Option<core::Expr> {
        // 1. First, try to find an in-scope dictionary variable directly
        if let Some(dict_var) = self.lookup_dict(constraint.class) {
            return Some(core::Expr::Var(dict_var.clone(), span));
        }

        // 2. Try superclass extraction: if we have Ord but need Eq, extract Eq from Ord
        if let Some((subclass, dict_var)) = self.lookup_superclass_dict(constraint.class) {
            // We found a dictionary for a class that has our needed class as a superclass
            // Extract the superclass dictionary
            if let Some(superclass_expr) = crate::dictionary::select_superclass(
                dict_var,
                subclass,
                constraint.class,
                &self.class_registry,
                span,
            ) {
                return Some(superclass_expr);
            }
        }

        // 3. If not in scope, try to construct from an instance
        // (only works for concrete types)
        if let Some(ty) = constraint.args.first() {
            if !has_type_variables(ty) {
                // Create a DictContext to construct the dictionary
                let mut dict_ctx = DictContext::new(&self.class_registry);
                let dict_expr = dict_ctx.get_dictionary(constraint, span)?;

                // If the dictionary construction generated bindings, wrap the
                // expression in let bindings
                let bindings = dict_ctx.take_bindings();
                if bindings.is_empty() {
                    return Some(dict_expr);
                }

                // Wrap in let bindings (innermost first)
                let mut result = dict_expr;
                for bind in bindings.into_iter().rev() {
                    result = core::Expr::Let(Box::new(bind), Box::new(result), span);
                }
                return Some(result);
            }
        }

        None
    }

    /// Resolve a class method call at a concrete type.
    ///
    /// When a class method (like `(+)` from `Num`) is called at a concrete type
    /// (like `Int`), we need to:
    /// 1. Construct the dictionary for that instance (e.g., `Num Int`)
    /// 2. Select the method from the dictionary
    ///
    /// Returns the method selection expression with any necessary let bindings.
    pub fn resolve_method_at_concrete_type(
        &mut self,
        method_name: Symbol,
        class_name: Symbol,
        concrete_type: &Ty,
        span: Span,
    ) -> Option<core::Expr> {
        // Create a constraint for the concrete type
        let constraint = Constraint::new(class_name, concrete_type.clone(), span);

        // Construct the dictionary
        let mut dict_ctx = DictContext::new(&self.class_registry);
        let dict_expr = dict_ctx.get_dictionary(&constraint, span)?;
        let bindings = dict_ctx.take_bindings();

        // Create a fresh variable to hold the dictionary
        let dict_var = self.fresh_var(&format!("$d{}", class_name.as_str()), Ty::Error, span);

        // Select the method from the dictionary
        let method_expr = crate::dictionary::select_method(
            &dict_var,
            class_name,
            method_name,
            &self.class_registry,
            span,
        )?;

        // Build the let expression:
        // let $dict = <dict_expr> in <method_expr>
        let dict_bind = Bind::NonRec(dict_var, Box::new(dict_expr));

        // If there are additional bindings from nested dictionary construction,
        // wrap them around the whole thing
        let mut result = core::Expr::Let(Box::new(dict_bind), Box::new(method_expr), span);
        for bind in bindings.into_iter().rev() {
            result = core::Expr::Let(Box::new(bind), Box::new(result), span);
        }

        Some(result)
    }

    /// Select a method from a dictionary.
    ///
    /// Given a dictionary variable and a method name, returns an expression
    /// that extracts that method from the dictionary.
    pub fn select_method_from_dict(
        &self,
        dict_var: &Var,
        class: Symbol,
        method_name: Symbol,
        span: Span,
    ) -> Option<core::Expr> {
        crate::dictionary::select_method(dict_var, class, method_name, &self.class_registry, span)
    }

    /// Check if a symbol is a class method.
    ///
    /// Returns the class name if the symbol is a method of some class.
    #[must_use]
    pub fn is_class_method(&self, method_name: Symbol) -> Option<Symbol> {
        for (class_name, class_info) in &self.class_registry.classes {
            if class_info.methods.contains(&method_name) {
                return Some(*class_name);
            }
        }
        None
    }

    /// Register a type class definition in the class registry.
    fn register_class_def(&mut self, class_def: &bhc_hir::ClassDef) {
        use crate::dictionary::AssocTypeInfo;

        let mut method_types = FxHashMap::default();
        let mut method_names = Vec::new();

        // Collect method signatures
        for method_sig in &class_def.methods {
            method_names.push(method_sig.name);
            method_types.insert(method_sig.name, method_sig.ty.clone());
        }

        // Collect default method DefIds
        let mut defaults = FxHashMap::default();
        for default_def in &class_def.defaults {
            defaults.insert(default_def.name, default_def.id);
        }

        // Collect associated type declarations
        let assoc_types: Vec<AssocTypeInfo> = class_def
            .assoc_types
            .iter()
            .map(|assoc| AssocTypeInfo {
                name: assoc.name,
                params: assoc.params.clone(),
                kind: assoc.kind.clone(),
                default: assoc.default.clone(),
            })
            .collect();

        let class_info = ClassInfo {
            name: class_def.name,
            methods: method_names,
            method_types,
            superclasses: class_def.supers.clone(),
            defaults,
            assoc_types,
        };

        self.class_registry.register_class(class_info);
    }

    /// Register a type class instance definition in the class registry.
    fn register_instance_def(&mut self, instance_def: &bhc_hir::InstanceDef) {
        // Collect method implementations
        let mut methods = FxHashMap::default();
        for method_def in &instance_def.methods {
            methods.insert(method_def.name, method_def.id);

            // Register the method implementation as a variable, but only
            // if not already registered (the first pass may have registered
            // it with a $instance_ prefix name for codegen detection).
            if self.lookup_var(method_def.id).is_none() {
                let var = self.named_var(method_def.name, Ty::Error);
                self.register_var(method_def.id, var);
            }
        }

        // Collect associated type implementations
        let mut assoc_type_impls = FxHashMap::default();
        for assoc_impl in &instance_def.assoc_type_impls {
            assoc_type_impls.insert(assoc_impl.name, assoc_impl.rhs.clone());
        }

        // Get the instance type (first type in the types list)
        let instance_type = instance_def.types.first().cloned().unwrap_or(Ty::Error);

        // For superclass instances, we need to figure out what types satisfy
        // the superclass constraints. For now, we assume the same instance type.
        let superclass_instances = instance_def
            .constraints
            .iter()
            .map(|_| instance_type.clone())
            .collect();

        let instance_info = InstanceInfo {
            class: instance_def.class,
            instance_types: vec![instance_type],
            methods,
            superclass_instances,
            assoc_type_impls,
        };

        self.class_registry.register_instance(instance_info);
    }

    /// Lower a HIR module to Core.
    pub fn lower_module(&mut self, module: &HirModule) -> LowerResult<CoreModule> {
        // First pass: collect all top-level definitions and create Core variables
        // We use named_var here to preserve the original names for external visibility
        for item in &module.items {
            match item {
                Item::Value(value_def) => {
                    // Look up the type from the type checker
                    let ty = self.lookup_type(value_def.id);
                    let var = self.named_var(value_def.name, ty);
                    self.register_var(value_def.id, var);
                }
                Item::Class(class_def) => {
                    // Also register variables for default method implementations
                    for default_def in &class_def.defaults {
                        let ty = self.lookup_type(default_def.id);
                        let var = self.named_var(default_def.name, ty);
                        self.register_var(default_def.id, var);
                    }
                }
                Item::Instance(instance_def) => {
                    // Pre-register instance method variables so they can be
                    // referenced during the lowering pass.
                    // Use $instance_{method}_{TypeName} naming convention
                    // so codegen can detect and dispatch manual instance methods.
                    let inst_type_name = instance_def
                        .types
                        .first()
                        .and_then(|ty| match ty {
                            Ty::Con(con) => Some(con.name.as_str().to_string()),
                            _ => None,
                        })
                        .unwrap_or_else(|| "Unknown".to_string());
                    for method_def in &instance_def.methods {
                        let ty = self.lookup_type(method_def.id);
                        let instance_name = Symbol::intern(&format!(
                            "$instance_{}_{}",
                            method_def.name, inst_type_name
                        ));
                        let var = self.named_var(instance_name, ty);
                        self.register_var(method_def.id, var);
                    }
                }
                _ => {}
            }
        }

        // Second pass: lower all items
        let mut bindings = Vec::new();
        let mut deriv_ctx = DerivingContext::new();

        for item in &module.items {
            match item {
                Item::Value(value_def) => {
                    if let Some(bind) = self.lower_value_def(value_def)? {
                        bindings.push(bind);
                    }
                }
                Item::Data(data_def) => {
                    // Register data constructors with their metadata
                    // The tag is the 0-based position in the constructor list
                    for (tag, con) in data_def.cons.iter().enumerate() {
                        let var = self.named_var(con.name, Ty::Error);
                        self.register_var(con.id, var);

                        // Calculate arity and field names based on field type
                        let (arity, field_names) = match &con.fields {
                            bhc_hir::ConFields::Positional(fields) => (fields.len() as u32, vec![]),
                            bhc_hir::ConFields::Named(fields) => {
                                // Register field selector functions
                                for field in fields {
                                    let selector_var = self.named_var(field.name, Ty::Error);
                                    self.register_var(field.id, selector_var);
                                    // Also register field metadata for later lookup
                                    self.register_field_selector(
                                        field.name,
                                        FieldSelectorInfo {
                                            field_name: field.name,
                                            con_id: con.id,
                                            con_name: con.name,
                                            type_name: data_def.name,
                                            field_index: fields
                                                .iter()
                                                .position(|f| f.id == field.id)
                                                .unwrap_or(0),
                                            total_fields: fields.len(),
                                        },
                                    );
                                }
                                let names: Vec<Symbol> = fields.iter().map(|f| f.name).collect();
                                (fields.len() as u32, names)
                            }
                        };

                        // Register constructor metadata
                        self.register_constructor(
                            con.id,
                            ConstructorInfo {
                                name: con.name,
                                type_name: data_def.name,
                                tag: tag as u32,
                                arity,
                                field_names,
                            },
                        );
                    }

                    // Process deriving clauses
                    if !data_def.deriving.is_empty() {
                        let derived_instances: Vec<_> = data_def
                            .deriving
                            .iter()
                            .filter_map(|class_name| {
                                deriv_ctx.derive_for_data(data_def, *class_name)
                            })
                            .collect();
                        for derived in derived_instances {
                            self.class_registry.register_instance(derived.instance);
                            bindings.extend(derived.bindings);
                        }
                    }
                }
                Item::Newtype(newtype_def) => {
                    // Register the newtype constructor
                    let var = self.named_var(newtype_def.con.name, Ty::Error);
                    self.register_var(newtype_def.con.id, var);

                    // Process deriving clauses
                    if !newtype_def.deriving.is_empty() {
                        let derived_instances: Vec<_> = newtype_def
                            .deriving
                            .iter()
                            .filter_map(|class_name| {
                                deriv_ctx.derive_for_newtype(newtype_def, *class_name)
                            })
                            .collect();
                        for derived in derived_instances {
                            self.class_registry.register_instance(derived.instance);
                            bindings.extend(derived.bindings);
                        }
                    }
                }
                Item::TypeAlias(_) => {
                    // Type aliases don't produce bindings
                }
                Item::Class(class_def) => {
                    // Register the class in the class registry for dictionary construction
                    self.register_class_def(class_def);

                    // Lower default method implementations
                    // Default methods need the class constraint, so we lower them specially
                    for default_def in &class_def.defaults {
                        if let Some(bind) = self.lower_default_method(class_def, default_def)? {
                            bindings.push(bind);
                        }
                    }
                }
                Item::Instance(instance_def) => {
                    // Register the instance in the class registry for dictionary construction
                    self.register_instance_def(instance_def);

                    // Lower instance method bodies to Core bindings.
                    // Each method in the instance provides an implementation that
                    // the evaluator needs to find.
                    for method_def in &instance_def.methods {
                        if let Some(bind) = self.lower_value_def(method_def)? {
                            bindings.push(bind);
                        }
                    }
                }
                Item::Fixity(_) => {
                    // Fixity declarations are only used during parsing
                }
                Item::Foreign(foreign) => {
                    // Foreign imports become special Core bindings
                    // Use named_var to preserve the original name
                    let var = self.named_var(
                        foreign.name,
                        Ty::Error, // Type from signature
                    );
                    self.register_var(foreign.id, var.clone());
                    // For now, we create a placeholder binding
                    // TODO: Proper foreign binding representation
                }
            }
        }

        // Check for errors
        if self.has_errors() {
            return Err(LowerError::Multiple(self.take_errors()));
        }

        Ok(CoreModule {
            name: module.name,
            bindings,
            exports: vec![],
            overloaded_strings: module.overloaded_strings,
        })
    }

    /// Lower a value definition to a Core binding.
    fn lower_value_def(&mut self, value_def: &ValueDef) -> LowerResult<Option<Bind>> {
        let var = self
            .lookup_var(value_def.id)
            .cloned()
            .ok_or_else(|| LowerError::Internal("missing variable for value def".into()))?;

        // Check if the definition has type class constraints
        let constraints = self
            .lookup_scheme(value_def.id)
            .map(|s| s.constraints.clone())
            .unwrap_or_default();

        // If there are constraints, create dictionary variables and push them into scope
        // BEFORE compiling the body, so references in the body can use them.
        let dict_vars: Vec<(Symbol, Var)> = constraints
            .iter()
            .map(|c| {
                let dict_var = self.make_dict_var(c);
                (c.class, dict_var)
            })
            .collect();

        // Push a new dictionary scope and register all dictionaries
        if !dict_vars.is_empty() {
            self.push_dict_scope();
            for (class_name, dict_var) in &dict_vars {
                self.register_dict(*class_name, dict_var.clone());
            }
        }

        // Compile equations to a single expression (now with dictionaries in scope)
        let mut body = self.compile_equations(value_def)?;

        // Pop the dictionary scope
        if !dict_vars.is_empty() {
            self.pop_dict_scope();
        }

        // If there are constraints, wrap the body in dictionary lambdas.
        // For example, a function `f :: Num a => a -> a` becomes:
        //   f = \$dNum -> \x -> ... (using $dNum for Num operations)
        if !dict_vars.is_empty() {
            // Add dictionary parameters in reverse order so the first
            // constraint gets the outermost lambda
            for (_, dict_var) in dict_vars.into_iter().rev() {
                body = core::Expr::Lam(dict_var, Box::new(body), value_def.span);
            }
        }

        Ok(Some(Bind::NonRec(var, Box::new(body))))
    }

    /// Lower a default method implementation from a class definition.
    ///
    /// Default methods are special because they implicitly have the class constraint.
    /// For example, in:
    /// ```text
    /// class Eq a where
    ///   (==) :: a -> a -> Bool
    ///   (/=) :: a -> a -> Bool
    ///   x /= y = not (x == y)  -- default
    /// ```
    ///
    /// The default `/=` has an implicit `Eq a` constraint. When lowered, it becomes:
    /// ```text
    /// $default_neq = \$dEq -> \x -> \y -> not (($sel_0 $dEq) x y)
    /// ```
    /// where `$sel_0` selects `(==)` from the Eq dictionary.
    fn lower_default_method(
        &mut self,
        class_def: &bhc_hir::ClassDef,
        default_def: &ValueDef,
    ) -> LowerResult<Option<Bind>> {
        let var = self
            .lookup_var(default_def.id)
            .cloned()
            .ok_or_else(|| LowerError::Internal("missing variable for default method".into()))?;

        // Default methods have the class constraint.
        // Create a constraint for the class with its type parameter.
        // The type parameter is from the class definition.
        let class_constraint = if let Some(type_param) = class_def.params.first() {
            Constraint::new(
                class_def.name,
                Ty::Var(type_param.clone()),
                default_def.span,
            )
        } else {
            // Class with no type parameters - unusual but handle it
            Constraint::new(class_def.name, Ty::Error, default_def.span)
        };

        // Create dictionary variable for the class constraint
        let dict_var = self.make_dict_var(&class_constraint);

        // Push dictionary scope and register the class dictionary
        self.push_dict_scope();
        self.register_dict(class_def.name, dict_var.clone());

        // Compile the default method body (now with class dictionary in scope)
        let mut body = self.compile_equations(default_def)?;

        // Pop the dictionary scope
        self.pop_dict_scope();

        // Wrap body in dictionary lambda
        body = core::Expr::Lam(dict_var, Box::new(body), default_def.span);

        Ok(Some(Bind::NonRec(var, Box::new(body))))
    }

    /// Create a dictionary variable for a type class constraint.
    ///
    /// The naming convention is `$d<ClassName>` to avoid conflicts with
    /// user-defined variables.
    fn make_dict_var(&mut self, constraint: &bhc_types::Constraint) -> Var {
        let dict_name = format!("$d{}", constraint.class.as_str());
        self.fresh_var(&dict_name, Ty::Error, constraint.span)
    }

    /// Compile multiple equations into a single Core expression.
    ///
    /// For simple definitions like `f = e`, this just lowers the expression.
    /// For pattern-matching definitions like:
    /// ```haskell
    /// f 0 = 1
    /// f n = n * f (n-1)
    /// ```
    /// This compiles to a lambda with a case expression.
    fn compile_equations(&mut self, value_def: &ValueDef) -> LowerResult<core::Expr> {
        if value_def.equations.is_empty() {
            return Err(LowerError::Internal(
                "value definition with no equations".into(),
            ));
        }

        // Simple case: single equation with no patterns
        if value_def.equations.len() == 1 && value_def.equations[0].pats.is_empty() {
            let eq = &value_def.equations[0];
            return lower_expr(self, &eq.rhs);
        }

        // Complex case: multiple equations or patterns
        // Figure out how many arguments the function takes
        let arity = value_def.equations[0].pats.len();

        if arity == 0 {
            // Multiple equations with no arguments - this is an error
            // but we'll just use the first one
            return lower_expr(self, &value_def.equations[0].rhs);
        }

        // Generate fresh variables for each argument
        let args: Vec<Var> = (0..arity)
            .map(|i| self.fresh_var(&format!("arg{}", i), Ty::Error, value_def.span))
            .collect();

        // Find which argument position(s) need constructor matching
        // If only one position has constructors, we can avoid tuple patterns
        let constructor_positions = self.find_constructor_positions(value_def);

        let case_expr = if arity == 1 || constructor_positions.len() != 1 {
            // Single argument or complex case: use original approach
            let scrutinee = if arity == 1 {
                core::Expr::Var(args[0].clone(), value_def.span)
            } else {
                // Multiple arguments: create a tuple scrutinee
                self.make_tuple_expr(&args, value_def.span)
            };

            // Compile pattern matching
            let case_alts = self.compile_pattern_match(value_def, &args)?;

            core::Expr::Case(
                Box::new(scrutinee),
                case_alts,
                Ty::Error, // Result type placeholder
                value_def.span,
            )
        } else {
            // Optimization: only one argument has constructor patterns
            // Case on just that argument, bind others directly
            let match_pos = constructor_positions[0];
            let match_arg = &args[match_pos];

            self.compile_single_position_match(value_def, &args, match_pos)?
        };

        // Wrap in lambdas
        let mut result = case_expr;
        for arg in args.into_iter().rev() {
            result = core::Expr::Lam(arg, Box::new(result), value_def.span);
        }

        Ok(result)
    }

    /// Make a tuple expression from variables.
    fn make_tuple_expr(&mut self, vars: &[Var], span: Span) -> core::Expr {
        // Build tuple constructor application
        let tuple_con_name = Symbol::intern(&format!("({})", ",".repeat(vars.len() - 1)));

        let mut expr = core::Expr::Var(
            Var {
                name: tuple_con_name,
                id: VarId::new(0),
                ty: Ty::Error,
            },
            span,
        );

        for var in vars {
            expr = core::Expr::App(
                Box::new(expr),
                Box::new(core::Expr::Var(var.clone(), span)),
                span,
            );
        }

        expr
    }

    /// Compile pattern matching for multiple equations.
    fn compile_pattern_match(
        &mut self,
        value_def: &ValueDef,
        args: &[Var],
    ) -> LowerResult<Vec<core::Alt>> {
        use crate::pattern::compile_match;
        compile_match(self, value_def, args)
    }

    /// Find which argument positions have constructor patterns.
    fn find_constructor_positions(&self, value_def: &ValueDef) -> Vec<usize> {
        use bhc_hir::Pat;

        let mut positions = Vec::new();
        let arity = value_def
            .equations
            .get(0)
            .map(|eq| eq.pats.len())
            .unwrap_or(0);

        for pos in 0..arity {
            let has_constructor = value_def.equations.iter().any(|eq| {
                eq.pats
                    .get(pos)
                    .map(|pat| matches!(pat, Pat::Con(_, _, _) | Pat::Lit(_, _)))
                    .unwrap_or(false)
            });
            if has_constructor {
                positions.push(pos);
            }
        }

        positions
    }

    /// Compile pattern matching when only one argument position has constructors.
    fn compile_single_position_match(
        &mut self,
        value_def: &ValueDef,
        args: &[Var],
        match_pos: usize,
    ) -> LowerResult<core::Expr> {
        use crate::pattern::{bind_pattern_vars, lower_pat_to_alt};
        use bhc_hir::Pat;

        let span = value_def.span;
        let mut alts = Vec::new();

        for eq in &value_def.equations {
            // Register all pattern variables before lowering RHS
            for (i, pat) in eq.pats.iter().enumerate() {
                let arg_var = args.get(i).cloned();
                bind_pattern_vars(self, pat, arg_var.as_ref());
            }

            // Lower the RHS
            let rhs = lower_expr(self, &eq.rhs)?;

            // Get the pattern at the match position
            if let Some(pat) = eq.pats.get(match_pos) {
                let alt = lower_pat_to_alt(self, pat, rhs, span)?;
                alts.push(alt);
            }
        }

        // Add default error case
        alts.push(core::Alt {
            con: core::AltCon::Default,
            binders: vec![],
            rhs: crate::pattern::make_pattern_error(span),
        });

        // Case on the matching argument
        Ok(core::Expr::Case(
            Box::new(core::Expr::Var(args[match_pos].clone(), span)),
            alts,
            Ty::Error,
            span,
        ))
    }
}

impl Default for LowerContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a type contains type variables.
fn has_type_variables(ty: &Ty) -> bool {
    match ty {
        Ty::Var(_) => true,
        Ty::Con(_) | Ty::Prim(_) | Ty::Error => false,
        Ty::App(f, a) | Ty::Fun(f, a) => has_type_variables(f) || has_type_variables(a),
        Ty::Tuple(tys) => tys.iter().any(has_type_variables),
        Ty::List(elem) => has_type_variables(elem),
        Ty::Forall(_, body) => has_type_variables(body),
        Ty::Nat(_) | Ty::TyList(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_var() {
        let mut ctx = LowerContext::new();
        let v1 = ctx.fresh_var("x", Ty::Error, Span::default());
        let v2 = ctx.fresh_var("x", Ty::Error, Span::default());
        assert_ne!(v1.id, v2.id);
    }
}
