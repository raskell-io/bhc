//! Type checking context and state management.
//!
//! This module provides the main type checking context [`TyCtxt`] which
//! manages all state during type inference including:
//!
//! - Fresh type variable generation
//! - Substitution accumulation
//! - Type environment management
//! - Diagnostic collection

use bhc_diagnostics::{Diagnostic, DiagnosticHandler, FullSpan};
use bhc_hir::{
    Binding, ClassDef, ConFields, DataDef, DefId, Equation, HirId, InstanceDef, Item, Module,
    NewtypeDef, Pat, ValueDef,
};
use bhc_intern::Symbol;
use bhc_span::{FileId, Span};
use bhc_types::{Kind, Scheme, Subst, Ty, TyCon, TyVar};
use rustc_hash::FxHashMap;

use crate::binding_groups::BindingGroup;
use crate::builtins::Builtins;
use crate::env::{ClassInfo, InstanceInfo, TypeEnv};
use crate::TypedModule;

/// Generator for fresh type variables.
///
/// Produces unique type variable IDs during type inference.
#[derive(Debug, Default)]
pub struct TyVarGen {
    next_id: u32,
}

impl TyVarGen {
    /// Create a new type variable generator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a fresh type variable with kind `*`.
    #[must_use]
    pub fn fresh(&mut self) -> TyVar {
        let id = self.next_id;
        self.next_id += 1;
        TyVar::new_star(id)
    }

    /// Generate a fresh type variable with the given kind.
    #[must_use]
    pub fn fresh_with_kind(&mut self, kind: Kind) -> TyVar {
        let id = self.next_id;
        self.next_id += 1;
        TyVar::new(id, kind)
    }
}

/// The main type checking context.
///
/// This struct holds all state needed during type inference for a module.
/// It manages type variables, substitutions, the type environment, and
/// error diagnostics.
#[derive(Debug)]
pub struct TyCtxt {
    /// Generator for fresh type variables.
    ty_var_gen: TyVarGen,

    /// Current substitution (accumulated unification results).
    pub(crate) subst: Subst,

    /// Type environment (bindings in scope).
    pub(crate) env: TypeEnv,

    /// Diagnostic handler for error collection.
    diag: DiagnosticHandler,

    /// Built-in types (Int, Bool, etc.).
    pub(crate) builtins: Builtins,

    /// Current file ID for error reporting.
    pub(crate) file_id: FileId,

    /// Inferred types for expressions (`HirId` -> Ty).
    pub(crate) expr_types: FxHashMap<HirId, Ty>,

    /// Type schemes for definitions (`DefId` -> Scheme).
    pub(crate) def_schemes: FxHashMap<DefId, Scheme>,

    /// Maps constructor DefId to named field definitions (name, type) pairs.
    /// Used for record construction type checking with out-of-order fields.
    pub(crate) con_field_defs: FxHashMap<DefId, Vec<(Symbol, Ty)>>,
}

impl TyCtxt {
    /// Create a new type checking context for the given file.
    #[must_use]
    pub fn new(file_id: FileId) -> Self {
        Self {
            ty_var_gen: TyVarGen::new(),
            subst: Subst::new(),
            env: TypeEnv::new(),
            diag: DiagnosticHandler::new(),
            builtins: Builtins::default(),
            file_id,
            expr_types: FxHashMap::default(),
            def_schemes: FxHashMap::default(),
            con_field_defs: FxHashMap::default(),
        }
    }

    /// Generate a fresh type variable with kind `*`.
    pub fn fresh_ty_var(&mut self) -> TyVar {
        self.ty_var_gen.fresh()
    }

    /// Generate a fresh type (`Ty::Var`) with kind `*`.
    pub fn fresh_ty(&mut self) -> Ty {
        Ty::Var(self.fresh_ty_var())
    }

    /// Generate a fresh type variable with the given kind.
    pub fn fresh_ty_var_with_kind(&mut self, kind: Kind) -> TyVar {
        self.ty_var_gen.fresh_with_kind(kind)
    }

    /// Apply the current substitution to a type.
    #[must_use]
    pub fn apply_subst(&self, ty: &Ty) -> Ty {
        self.subst.apply(ty)
    }

    /// Register built-in types in the environment.
    pub fn register_builtins(&mut self) {
        self.builtins = Builtins::new();

        // Register type constructors
        self.env
            .register_type_con(self.builtins.int_con.clone());
        self.env
            .register_type_con(self.builtins.float_con.clone());
        self.env
            .register_type_con(self.builtins.char_con.clone());
        self.env
            .register_type_con(self.builtins.bool_con.clone());
        self.env
            .register_type_con(self.builtins.string_con.clone());
        self.env
            .register_type_con(self.builtins.list_con.clone());
        self.env
            .register_type_con(self.builtins.maybe_con.clone());
        self.env
            .register_type_con(self.builtins.either_con.clone());
        self.env.register_type_con(self.builtins.io_con.clone());

        // Register shape-indexed tensor type constructors
        self.env
            .register_type_con(self.builtins.tensor_con.clone());
        self.env
            .register_type_con(self.builtins.dyn_tensor_con.clone());
        self.env
            .register_type_con(self.builtins.shape_witness_con.clone());

        // Register built-in data constructors
        self.builtins.register_data_cons(&mut self.env);

        // Register primitive operators (+, -, *, etc.)
        self.builtins.register_primitive_ops(&mut self.env);

        // Register dynamic tensor operations (toDynamic, fromDynamic, etc.)
        self.builtins.register_dyn_tensor_ops(&mut self.env);
    }

    /// Register builtins using the DefIds from the lowering pass.
    ///
    /// The lowering pass assigns DefIds to builtin functions and constructors.
    /// This method registers those builtins in the type environment with the
    /// correct DefIds so that type checking can find them.
    pub fn register_lowered_builtins(&mut self, defs: &crate::DefMap) {
        use bhc_lower::DefKind;
        use bhc_types::{TyVar, Kind};

        // Type variables for polymorphic types
        let a = TyVar::new_star(0xFFFF_0000);
        let b = TyVar::new_star(0xFFFF_0001);

        // First pass: register data constructors
        for (_def_id, def_info) in defs.iter() {
            // Only process constructor kinds
            if !matches!(def_info.kind, DefKind::Constructor | DefKind::StubConstructor) {
                continue;
            }

            let name = def_info.name.as_str();
            let scheme = match name {
                // Bool constructors
                "True" | "False" => Scheme::mono(self.builtins.bool_ty.clone()),

                // Maybe constructors
                "Nothing" => {
                    // Nothing :: Maybe a
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone()], maybe_a)
                }
                "Just" => {
                    // Just :: a -> Maybe a
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), maybe_a))
                }

                // Either constructors
                "Left" => {
                    // Left :: a -> Either a b
                    let either_ab = Ty::App(
                        Box::new(Ty::App(
                            Box::new(Ty::Con(self.builtins.either_con.clone())),
                            Box::new(Ty::Var(a.clone())),
                        )),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), either_ab))
                }
                "Right" => {
                    // Right :: b -> Either a b
                    let either_ab = Ty::App(
                        Box::new(Ty::App(
                            Box::new(Ty::Con(self.builtins.either_con.clone())),
                            Box::new(Ty::Var(a.clone())),
                        )),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(b.clone()), either_ab))
                }

                // List constructors
                "[]" => {
                    // [] :: [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], list_a)
                }
                ":" => {
                    // (:) :: a -> [a] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                    )
                }

                // Unit constructor
                "()" => Scheme::mono(Ty::unit()),

                // Tuple constructors
                "(,)" => {
                    // (,) :: a -> b -> (a, b)
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())])),
                        ),
                    )
                }
                "(,,)" => {
                    // (,,) :: a -> b -> c -> (a, b, c)
                    let c = TyVar::new_star(0xFFFF_0002);
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(
                                Ty::Var(b.clone()),
                                Ty::fun(
                                    Ty::Var(c.clone()),
                                    Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone()), Ty::Var(c.clone())]),
                                ),
                            ),
                        ),
                    )
                }

                // NonEmpty constructor
                ":|" => {
                    // (:|) :: a -> [a] -> NonEmpty a
                    // For now, approximate as a -> [a] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                    )
                }

                // For imported constructors that aren't known builtins,
                // create a function type based on the constructor's arity.
                _ => {
                    if let (Some(arity), Some(type_con_name), Some(type_param_count)) =
                        (def_info.arity, def_info.type_con_name, def_info.type_param_count)
                    {
                        // Build proper polymorphic type: forall a1 .. an b1 .. bm. b1 -> b2 -> ... -> bm -> TypeCon a1 .. an
                        // where a1..an are result type params and b1..bm are field type params

                        // Create result type parameters (a1 .. an)
                        let result_type_params: Vec<TyVar> = (0..type_param_count)
                            .map(|i| TyVar::new_star(0xFFFE_0000 + i as u32))
                            .collect();

                        // Create field type parameters (b1 .. bm) - these must also be quantified
                        let field_type_params: Vec<TyVar> = (0..arity)
                            .map(|i| TyVar::new_star(0xFFFF_0000 + i as u32))
                            .collect();

                        // Build the result type: TypeCon a1 a2 ... an
                        let kind = Self::compute_type_con_kind(type_param_count);
                        let type_con = TyCon::new(type_con_name, kind);
                        let result_ty = result_type_params.iter().fold(Ty::Con(type_con), |acc, param| {
                            Ty::App(Box::new(acc), Box::new(Ty::Var(param.clone())))
                        });

                        // Build the constructor type: b1 -> b2 -> ... -> bm -> ResultType
                        // Build from inside out: result <- bm <- bm-1 <- ... <- b1
                        let mut field_types: Vec<Ty> = field_type_params.iter()
                            .map(|tv| Ty::Var(tv.clone()))
                            .collect();

                        let mut con_ty = result_ty;
                        for field_ty in field_types.iter().rev() {
                            con_ty = Ty::fun(field_ty.clone(), con_ty);
                        }

                        // If we have field names, register field definitions for record construction
                        // Note: we don't store the field types here since they're quantified
                        // and will be instantiated fresh each time. We only need the names.
                        if let Some(ref field_names) = def_info.field_names {
                            let field_defs: Vec<(Symbol, Ty)> = field_names
                                .iter()
                                .zip(field_types.iter())
                                .map(|(name, ty)| (*name, ty.clone()))
                                .collect();
                            self.con_field_defs.insert(def_info.id, field_defs);
                        }

                        // Combine all type parameters: result params + field params
                        let all_params: Vec<TyVar> = result_type_params.into_iter()
                            .chain(field_type_params.into_iter())
                            .collect();

                        Scheme::poly(all_params, con_ty)
                    } else if let Some(arity) = def_info.arity {
                        // No type info, fall back to fresh type variables
                        let result = self.fresh_ty();
                        let mut con_ty = result;
                        for _ in 0..arity {
                            let arg = self.fresh_ty();
                            con_ty = Ty::fun(arg, con_ty);
                        }
                        Scheme::mono(con_ty)
                    } else {
                        // No arity info, fall back to fresh type variable
                        let fresh = self.fresh_ty();
                        Scheme::mono(fresh)
                    }
                }
            };

            // Register the constructor with its DefId from the lowering pass
            self.env.register_data_con(def_info.id, def_info.name, scheme);
        }

        // Helper to create common type schemes
        let num_binop = || {
            // a -> a -> a (for Num types, we simplify to Int for now)
            Scheme::mono(Ty::fun(
                self.builtins.int_ty.clone(),
                Ty::fun(self.builtins.int_ty.clone(), self.builtins.int_ty.clone()),
            ))
        };

        let cmp_binop = || {
            // a -> a -> Bool (for Ord types, we simplify to Int for now)
            Scheme::mono(Ty::fun(
                self.builtins.int_ty.clone(),
                Ty::fun(self.builtins.int_ty.clone(), self.builtins.bool_ty.clone()),
            ))
        };

        let eq_binop = || {
            // a -> a -> Bool (for Eq types, polymorphic)
            Scheme::poly(
                vec![a.clone()],
                Ty::fun(
                    Ty::Var(a.clone()),
                    Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                ),
            )
        };

        // For each def in the lowering pass's def map, register it with a type
        for (_def_id, def_info) in defs.iter() {
            let name = def_info.name.as_str();
            let scheme = match name {
                // Arithmetic operators
                "+" | "-" | "*" | "/" | "div" | "mod" | "^" | "^^" | "**" => num_binop(),
                // Comparison operators
                "==" | "/=" => eq_binop(),
                "<" | "<=" | ">" | ">=" => cmp_binop(),
                // Boolean operators
                "&&" | "||" => Scheme::mono(Ty::fun(
                    self.builtins.bool_ty.clone(),
                    Ty::fun(self.builtins.bool_ty.clone(), self.builtins.bool_ty.clone()),
                )),
                // List cons
                ":" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                    )
                }
                // List append
                "++" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                    )
                }
                // List indexing
                "!!" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(list_a, Ty::fun(self.builtins.int_ty.clone(), Ty::Var(a.clone()))),
                    )
                }
                // Function composition
                "." => {
                    let c = TyVar::new_star(0xFFFF_0002);
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
                }
                // Function application
                "$" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        ),
                    )
                }
                // map :: (a -> b) -> [a] -> [b]
                "map" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(list_a, list_b),
                        ),
                    )
                }
                // filter :: (a -> Bool) -> [a] -> [a]
                "filter" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                            Ty::fun(list_a.clone(), list_a),
                        ),
                    )
                }
                // head :: [a] -> a
                "head" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
                }
                // tail :: [a] -> [a]
                "tail" | "reverse" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
                }
                // length :: [a] -> Int
                "length" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, self.builtins.int_ty.clone()))
                }
                // null :: [a] -> Bool
                "null" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, self.builtins.bool_ty.clone()))
                }
                // take, drop :: Int -> [a] -> [a]
                "take" | "drop" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(self.builtins.int_ty.clone(), Ty::fun(list_a.clone(), list_a)),
                    )
                }
                // sum, product :: [Int] -> Int
                "sum" | "product" => {
                    let list_int = Ty::List(Box::new(self.builtins.int_ty.clone()));
                    Scheme::mono(Ty::fun(list_int, self.builtins.int_ty.clone()))
                }
                // maximum, minimum :: [a] -> a
                "maximum" | "minimum" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
                }
                // id :: a -> a
                "id" => Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                // const :: a -> b -> a
                "const" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone()))),
                ),
                // error :: String -> a
                "error" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.builtins.string_ty.clone(), Ty::Var(a.clone())),
                ),
                // undefined :: a
                "undefined" => Scheme::poly(vec![a.clone()], Ty::Var(a.clone())),
                // seq :: a -> b -> b
                "seq" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))),
                ),
                // negate, abs, signum :: Int -> Int
                "negate" | "abs" | "signum" => {
                    Scheme::mono(Ty::fun(self.builtins.int_ty.clone(), self.builtins.int_ty.clone()))
                }
                // not :: Bool -> Bool
                "not" => Scheme::mono(Ty::fun(self.builtins.bool_ty.clone(), self.builtins.bool_ty.clone())),
                // otherwise :: Bool
                "otherwise" => Scheme::mono(self.builtins.bool_ty.clone()),
                // show :: a -> String
                "show" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), self.builtins.string_ty.clone()),
                ),
                // print :: a -> IO ()
                "print" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::App(
                            Box::new(Ty::Con(self.builtins.io_con.clone())),
                            Box::new(Ty::unit()),
                        ),
                    ),
                ),
                // putStrLn, putStr :: String -> IO ()
                "putStrLn" | "putStr" => Scheme::mono(Ty::fun(
                    self.builtins.string_ty.clone(),
                    Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::unit()),
                    ),
                )),
                // getLine :: IO String
                "getLine" => Scheme::mono(Ty::App(
                    Box::new(Ty::Con(self.builtins.io_con.clone())),
                    Box::new(self.builtins.string_ty.clone()),
                )),
                // fst :: (a, b) -> a
                "fst" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]), Ty::Var(a.clone())),
                ),
                // snd :: (a, b) -> b
                "snd" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]), Ty::Var(b.clone())),
                ),
                // Unknown builtins - skip here, will be handled in second pass
                _ => continue,
            };

            // Register the builtin with its DefId from the lowering pass
            self.env.insert_global(def_info.id, scheme);
        }

        // Second pass: register any remaining definitions (imported items not in builtins)
        // with fresh type variables. We do this in a separate pass to avoid borrow conflicts
        // with the closures above.
        for (_def_id, def_info) in defs.iter() {
            // Skip constructors (handled in first pass)
            if matches!(def_info.kind, DefKind::Constructor | DefKind::StubConstructor) {
                continue;
            }
            // Skip if already registered
            if self.env.lookup_global(def_info.id).is_some() {
                continue;
            }
            // Register with a fresh polymorphic type
            let fresh = self.fresh_ty();
            let scheme = Scheme::mono(fresh);
            self.env.insert_global(def_info.id, scheme);
        }
    }

    /// Register a data type definition.
    pub fn register_data_type(&mut self, data: &DataDef) {
        // Register the type constructor
        let kind = Self::compute_type_con_kind(data.params.len());
        let tycon = TyCon::new(data.name, kind);
        self.env.register_type_con(tycon);

        // Register data constructors and field accessors
        for con in &data.cons {
            let scheme = self.compute_data_con_scheme(data, con);
            self.env.register_data_con(con.id, con.name, scheme);

            // Register field accessor functions for record constructors
            if let ConFields::Named(fields) = &con.fields {
                // Build the data type: T a1 a2 ... an
                let data_ty = Self::build_applied_type(data.name, &data.params);

                // Store field definitions for record type checking
                let field_defs: Vec<(Symbol, Ty)> = fields
                    .iter()
                    .map(|f| (f.name, f.ty.clone()))
                    .collect();
                self.con_field_defs.insert(con.id, field_defs);

                for field in fields {
                    // Field accessor type: T a1 ... an -> FieldType
                    let accessor_ty = Ty::fun(data_ty.clone(), field.ty.clone());
                    let accessor_scheme = Scheme::poly(data.params.clone(), accessor_ty);
                    // Register the field accessor as a global value
                    self.env.insert_global(field.id, accessor_scheme);
                }
            }
        }
    }

    /// Register a newtype definition.
    pub fn register_newtype(&mut self, newtype: &NewtypeDef) {
        // Register the type constructor
        let kind = Self::compute_type_con_kind(newtype.params.len());
        let tycon = TyCon::new(newtype.name, kind);
        self.env.register_type_con(tycon);

        // Register the single constructor
        let scheme = self.compute_newtype_con_scheme(newtype);
        self.env
            .register_data_con(newtype.con.id, newtype.con.name, scheme);

        // Register field accessor if this is a record-style newtype
        if let ConFields::Named(fields) = &newtype.con.fields {
            // Store field definitions for record construction type checking
            let field_defs: Vec<(Symbol, Ty)> = fields
                .iter()
                .map(|f| (f.name, f.ty.clone()))
                .collect();
            self.con_field_defs.insert(newtype.con.id, field_defs);

            // Build the newtype: T a1 a2 ... an
            let newtype_ty = Self::build_applied_type(newtype.name, &newtype.params);

            for field in fields {
                // Field accessor type: T a1 ... an -> FieldType
                let accessor_ty = Ty::fun(newtype_ty.clone(), field.ty.clone());
                let accessor_scheme = Scheme::poly(newtype.params.clone(), accessor_ty);
                // Register the field accessor as a global value
                self.env.insert_global(field.id, accessor_scheme);
            }
        }
    }

    /// Look up the named fields for a record constructor.
    ///
    /// Returns the fields as (name, type) pairs if the constructor is a record type,
    /// or None if it's a positional constructor.
    #[must_use]
    pub fn get_con_fields(&self, def_id: DefId) -> Option<&[(Symbol, Ty)]> {
        self.con_field_defs.get(&def_id).map(|v: &Vec<(Symbol, Ty)>| v.as_slice())
    }

    /// Register a type class definition.
    pub fn register_class(&mut self, class: &ClassDef) {
        // Build method signatures map
        let methods = class
            .methods
            .iter()
            .map(|m| (m.name, m.ty.clone()))
            .collect();

        let info = ClassInfo {
            name: class.name,
            params: class.params.clone(),
            supers: class.supers.clone(),
            methods,
        };

        self.env.register_class(info);

        // TODO: Register default method implementations
        // for default in &class.defaults {
        //     self.check_value_def(default);
        // }
    }

    /// Register a type class instance.
    pub fn register_instance(&mut self, instance: &InstanceDef) {
        // Build method implementations map
        let methods = instance
            .methods
            .iter()
            .map(|m| (m.name, m.id))
            .collect();

        let info = InstanceInfo {
            class: instance.class,
            types: instance.types.clone(),
            methods,
        };

        self.env.register_instance(info);

        // Type check the instance method implementations
        for method in &instance.methods {
            self.check_value_def(method);
        }
    }

    /// Compute the kind of a type constructor given its arity.
    fn compute_type_con_kind(arity: usize) -> Kind {
        let mut kind = Kind::Star;
        for _ in 0..arity {
            kind = Kind::Arrow(Box::new(Kind::Star), Box::new(kind));
        }
        kind
    }

    /// Compute the type scheme for a data constructor.
    fn compute_data_con_scheme(
        &self,
        data: &DataDef,
        con: &bhc_hir::ConDef,
    ) -> Scheme {
        use bhc_hir::ConFields;

        // Build the result type: T a1 a2 ... an
        let result_ty = Self::build_applied_type(data.name, &data.params);

        // Build the function type from field types to result type
        let field_types = match &con.fields {
            ConFields::Positional(tys) => tys.clone(),
            ConFields::Named(fields) => fields.iter().map(|f| f.ty.clone()).collect(),
        };

        // Fix the kinds of type variables in field types to match params.
        // The field types were lowered without kind information, so type
        // variables have kind Star. We need to update them to have the
        // correct kinds from data.params.
        let fixed_field_types: Vec<Ty> = field_types
            .into_iter()
            .map(|ty| Self::fix_type_var_kinds(&ty, &data.params))
            .collect();

        let con_ty = fixed_field_types
            .into_iter()
            .rev()
            .fold(result_ty, |acc, field_ty| Ty::fun(field_ty, acc));

        Scheme::poly(data.params.clone(), con_ty)
    }

    /// Fix the kinds of type variables in a type to match the given params.
    fn fix_type_var_kinds(ty: &Ty, params: &[TyVar]) -> Ty {
        match ty {
            Ty::Var(v) => {
                // Look for a param with the same ID and use its kind
                if let Some(param) = params.iter().find(|p| p.id == v.id) {
                    Ty::Var(param.clone())
                } else {
                    ty.clone()
                }
            }
            Ty::Con(_) | Ty::Prim(_) | Ty::Error => ty.clone(),
            Ty::App(f, a) => Ty::App(
                Box::new(Self::fix_type_var_kinds(f, params)),
                Box::new(Self::fix_type_var_kinds(a, params)),
            ),
            Ty::Fun(from, to) => Ty::Fun(
                Box::new(Self::fix_type_var_kinds(from, params)),
                Box::new(Self::fix_type_var_kinds(to, params)),
            ),
            Ty::Tuple(tys) => {
                Ty::Tuple(tys.iter().map(|t| Self::fix_type_var_kinds(t, params)).collect())
            }
            Ty::List(elem) => Ty::List(Box::new(Self::fix_type_var_kinds(elem, params))),
            Ty::Forall(vars, body) => {
                Ty::Forall(vars.clone(), Box::new(Self::fix_type_var_kinds(body, params)))
            }
            Ty::Nat(_) | Ty::TyList(_) => ty.clone(),
        }
    }

    /// Compute the type scheme for a newtype constructor.
    fn compute_newtype_con_scheme(&self, newtype: &NewtypeDef) -> Scheme {
        use bhc_hir::ConFields;

        // Build the result type: T a1 a2 ... an
        let result_ty = Self::build_applied_type(newtype.name, &newtype.params);

        // Newtype has exactly one field
        let field_ty = match &newtype.con.fields {
            ConFields::Positional(tys) => tys.first().cloned().unwrap_or_else(Ty::unit),
            ConFields::Named(fields) => {
                fields.first().map_or_else(Ty::unit, |f| f.ty.clone())
            }
        };

        // Fix the kinds of type variables in field type to match params
        let fixed_field_ty = Self::fix_type_var_kinds(&field_ty, &newtype.params);

        let con_ty = Ty::fun(fixed_field_ty, result_ty);
        Scheme::poly(newtype.params.clone(), con_ty)
    }

    /// Build an applied type: T a1 a2 ... an
    fn build_applied_type(name: bhc_intern::Symbol, params: &[TyVar]) -> Ty {
        let base = Ty::Con(TyCon::new(name, Self::compute_type_con_kind(params.len())));
        params
            .iter()
            .fold(base, |acc, param| Ty::App(Box::new(acc), Box::new(Ty::Var(param.clone()))))
    }

    /// Check a binding group (potentially mutually recursive bindings).
    pub fn check_binding_group(&mut self, group: &BindingGroup) {
        match group {
            BindingGroup::NonRecursive(item) => {
                self.check_item(item);
            }
            BindingGroup::Recursive(items) => {
                // For recursive groups, first add all bindings with fresh type variables
                let mut temp_schemes: Vec<(DefId, Scheme)> = Vec::new();

                for item in items {
                    if let Item::Value(value_def) = item {
                        let ty = value_def
                            .sig
                            .as_ref()
                            .map_or_else(|| self.fresh_ty(), |scheme| scheme.ty.clone());
                        let scheme = Scheme::mono(ty);
                        temp_schemes.push((value_def.id, scheme.clone()));
                        self.env.insert_global(value_def.id, scheme);
                    }
                }

                // Now type check each item
                for item in items {
                    self.check_item(item);
                }

                // Generalize the types
                for (def_id, _) in temp_schemes {
                    if let Some(scheme) = self.def_schemes.get(&def_id) {
                        let generalized = self.generalize(&scheme.ty);
                        self.def_schemes.insert(def_id, generalized.clone());
                        self.env.insert_global(def_id, generalized);
                    }
                }
            }
        }
    }

    /// Check a single item.
    fn check_item(&mut self, item: &Item) {
        match item {
            Item::Value(value_def) => self.check_value_def(value_def),
            Item::Class(class_def) => self.register_class(class_def),
            Item::Instance(instance_def) => self.register_instance(instance_def),
            // These are handled separately:
            // - Data/Newtype: registered in register_data_type/register_newtype
            // - TypeAlias: handled during type resolution
            // - Fixity: no type checking needed
            // - Foreign: uses declared type
            Item::Data(_)
            | Item::Newtype(_)
            | Item::TypeAlias(_)
            | Item::Fixity(_)
            | Item::Foreign(_) => {}
        }
    }

    /// Check a value definition.
    fn check_value_def(&mut self, value_def: &ValueDef) {
        // If there's a type signature, use it; otherwise infer
        let declared_ty = value_def.sig.as_ref().map(|s| s.ty.clone());

        // Infer the type from equations
        let inferred_ty = self.infer_equations(&value_def.equations, value_def.span);

        // If there's a declared type, unify with inferred type
        if let Some(declared) = &declared_ty {
            self.unify(declared, &inferred_ty, value_def.span);
        }

        // Generalize and store the scheme
        let final_ty = self.apply_subst(&inferred_ty);
        let scheme = value_def
            .sig
            .as_ref()
            .map_or_else(|| self.generalize(&final_ty), Clone::clone);

        self.def_schemes.insert(value_def.id, scheme.clone());
        self.env.insert_global(value_def.id, scheme);
    }

    /// Infer the type of a list of equations (function clauses).
    fn infer_equations(&mut self, equations: &[Equation], span: Span) -> Ty {
        if equations.is_empty() {
            return self.fresh_ty();
        }

        // All equations must have the same type
        let first_ty = self.infer_equation(&equations[0]);

        for eq in equations.iter().skip(1) {
            let eq_ty = self.infer_equation(eq);
            self.unify(&first_ty, &eq_ty, span);
        }

        first_ty
    }

    /// Infer the type of a single equation.
    fn infer_equation(&mut self, equation: &Equation) -> Ty {
        // Enter a new scope for pattern bindings
        self.env.push_scope();

        // Build function type from patterns
        let mut arg_types = Vec::new();
        for pat in &equation.pats {
            let pat_ty = self.infer_pattern(pat);
            arg_types.push(pat_ty);
        }

        // Infer RHS type
        let rhs_ty = self.infer_expr(&equation.rhs);

        // Exit scope
        self.env.pop_scope();

        // Build function type: arg1 -> arg2 -> ... -> result
        arg_types
            .into_iter()
            .rev()
            .fold(rhs_ty, |acc, arg_ty| Ty::fun(arg_ty, acc))
    }

    /// Emit a diagnostic error.
    pub fn emit_error(&mut self, diagnostic: Diagnostic) {
        self.diag.emit(diagnostic);
    }

    /// Check if any errors have been emitted.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.diag.has_errors()
    }

    /// Take all diagnostics.
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.diag.take_diagnostics()
    }

    /// Create a `FullSpan` from a Span.
    #[must_use]
    pub const fn full_span(&self, span: Span) -> FullSpan {
        FullSpan::new(self.file_id, span)
    }

    /// Convert the context into a `TypedModule`.
    #[must_use]
    pub fn into_typed_module(self, hir: Module) -> TypedModule {
        // Apply final substitution to all expression types
        let expr_types = self
            .expr_types
            .into_iter()
            .map(|(id, ty)| (id, self.subst.apply(&ty)))
            .collect();

        // Apply final substitution to all definition schemes
        let def_schemes = self
            .def_schemes
            .into_iter()
            .map(|(id, scheme)| {
                let applied_ty = self.subst.apply(&scheme.ty);
                (id, Scheme {
                    vars: scheme.vars,
                    constraints: scheme.constraints,
                    ty: applied_ty,
                })
            })
            .collect();

        TypedModule {
            hir,
            expr_types,
            def_schemes,
        }
    }
}

// Forward declarations for methods implemented in other modules
impl TyCtxt {
    /// Unify two types (implemented in unify.rs).
    pub fn unify(&mut self, t1: &Ty, t2: &Ty, span: Span) {
        crate::unify::unify(self, t1, t2, span);
    }

    /// Instantiate a type scheme (implemented in instantiate.rs).
    pub fn instantiate(&mut self, scheme: &Scheme) -> Ty {
        crate::instantiate::instantiate(self, scheme)
    }

    /// Generalize a type (implemented in generalize.rs).
    #[must_use]
    pub fn generalize(&self, ty: &Ty) -> Scheme {
        crate::generalize::generalize(self, ty)
    }

    /// Infer the type of an expression (implemented in infer.rs).
    pub fn infer_expr(&mut self, expr: &bhc_hir::Expr) -> Ty {
        crate::infer::infer_expr(self, expr)
    }

    /// Infer the type of a pattern (implemented in pattern.rs).
    pub fn infer_pattern(&mut self, pat: &Pat) -> Ty {
        crate::pattern::infer_pattern(self, pat)
    }

    /// Check a pattern against an expected type (implemented in pattern.rs).
    pub fn check_pattern(&mut self, pat: &Pat, expected: &Ty) {
        crate::pattern::check_pattern(self, pat, expected);
    }

    /// Check a binding (implemented in infer.rs).
    pub fn check_binding(&mut self, binding: &Binding) {
        crate::infer::check_binding(self, binding);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ty_var_gen() {
        let mut gen = TyVarGen::new();

        let v1 = gen.fresh();
        let v2 = gen.fresh();
        let v3 = gen.fresh();

        assert_eq!(v1.id, 0);
        assert_eq!(v2.id, 1);
        assert_eq!(v3.id, 2);
    }

    #[test]
    fn test_fresh_ty() {
        let mut ctx = TyCtxt::new(FileId::new(0));

        let ty1 = ctx.fresh_ty();
        let ty2 = ctx.fresh_ty();

        match (&ty1, &ty2) {
            (Ty::Var(v1), Ty::Var(v2)) => {
                assert_ne!(v1.id, v2.id);
            }
            _ => panic!("expected type variables"),
        }
    }
}
