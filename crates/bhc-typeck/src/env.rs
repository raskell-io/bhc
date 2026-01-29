//! Type environment with scoping support.
//!
//! This module provides [`TypeEnv`], which tracks type bindings at various
//! scopes during type checking. It supports:
//!
//! - Global definitions (module-level)
//! - Local bindings (let, lambda, case patterns)
//! - Type constructors
//! - Data constructors
//! - Type classes and instances

use bhc_hir::DefId;
use bhc_intern::Symbol;
use bhc_types::{Scheme, Ty, TyCon, TyVar};
use rustc_hash::FxHashMap;

/// Information about a data constructor.
#[derive(Clone, Debug)]
pub struct DataConInfo {
    /// The definition ID of the constructor.
    pub def_id: DefId,
    /// The name of the constructor.
    pub name: Symbol,
    /// The type scheme of the constructor.
    pub scheme: Scheme,
}

/// A functional dependency within a type class.
///
/// Represents `a b -> c` meaning "given types for parameters at indices in `from`,
/// the types at indices in `to` are uniquely determined".
#[derive(Clone, Debug)]
pub struct FunDep {
    /// Indices of determining type parameters.
    pub from: Vec<usize>,
    /// Indices of determined type parameters.
    pub to: Vec<usize>,
}

/// Information about an associated type within a class.
#[derive(Clone, Debug)]
pub struct AssocTypeInfo {
    /// The name of the associated type.
    pub name: Symbol,
    /// Additional type parameters beyond the class parameters.
    pub params: Vec<TyVar>,
    /// The result kind (usually `*`).
    pub kind: bhc_types::Kind,
    /// Optional default type definition.
    pub default: Option<Ty>,
}

/// Information about a type class.
#[derive(Clone, Debug)]
pub struct ClassInfo {
    /// The class name.
    pub name: Symbol,
    /// Type parameters.
    pub params: Vec<TyVar>,
    /// Functional dependencies.
    pub fundeps: Vec<FunDep>,
    /// Superclass names.
    pub supers: Vec<Symbol>,
    /// Method signatures (name -> type scheme).
    pub methods: FxHashMap<Symbol, Scheme>,
    /// Associated type declarations.
    pub assoc_types: Vec<AssocTypeInfo>,
}

/// An associated type implementation within an instance.
#[derive(Clone, Debug)]
pub struct AssocTypeImpl {
    /// The name of the associated type.
    pub name: Symbol,
    /// Type arguments (patterns matching the instance head).
    pub args: Vec<Ty>,
    /// The implementation type (right-hand side).
    pub rhs: Ty,
}

/// Information about a type class instance.
#[derive(Clone, Debug)]
pub struct InstanceInfo {
    /// The class being instantiated.
    pub class: Symbol,
    /// The instance types (e.g., `Int` for `instance Show Int`).
    pub types: Vec<Ty>,
    /// Method implementations (name -> DefId of the implementation).
    pub methods: FxHashMap<Symbol, DefId>,
    /// Associated type implementations.
    pub assoc_type_impls: Vec<AssocTypeImpl>,
}

/// The type environment during type checking.
///
/// Maintains bindings at various scopes:
/// - Global: Module-level definitions
/// - Local: Lambda-bound, let-bound, and pattern-bound variables
/// - Type constructors: Type names to `TyCon`
/// - Data constructors: Constructor names to `DataConInfo`
/// - Type classes: Class definitions
/// - Instances: Type class instances for resolution
#[derive(Debug)]
pub struct TypeEnv {
    /// Global definitions (module-level, indexed by `DefId`).
    globals: FxHashMap<DefId, Scheme>,

    /// Local bindings (scoped, using persistent data structure).
    /// Stack of scope frames, each containing Symbol -> Scheme mappings.
    locals: Vec<FxHashMap<Symbol, Scheme>>,

    /// Type constructors (name -> `TyCon`).
    type_cons: FxHashMap<Symbol, TyCon>,

    /// Data constructors (name -> `DataConInfo`).
    data_cons: FxHashMap<Symbol, DataConInfo>,

    /// Data constructors by `DefId` (for lookup by `DefRef`).
    data_cons_by_id: FxHashMap<DefId, DataConInfo>,

    /// Type classes (name -> `ClassInfo`).
    classes: FxHashMap<Symbol, ClassInfo>,

    /// Type class instances (class name -> list of instances).
    /// Multiple instances can exist for the same class with different types.
    instances: FxHashMap<Symbol, Vec<InstanceInfo>>,
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeEnv {
    /// Create a new empty type environment.
    #[must_use]
    pub fn new() -> Self {
        Self {
            globals: FxHashMap::default(),
            locals: vec![FxHashMap::default()], // Start with one scope
            type_cons: FxHashMap::default(),
            data_cons: FxHashMap::default(),
            data_cons_by_id: FxHashMap::default(),
            classes: FxHashMap::default(),
            instances: FxHashMap::default(),
        }
    }

    /// Push a new local scope.
    pub fn push_scope(&mut self) {
        self.locals.push(FxHashMap::default());
    }

    /// Pop the current local scope.
    pub fn pop_scope(&mut self) {
        if self.locals.len() > 1 {
            self.locals.pop();
        }
    }

    /// Insert a global binding.
    pub fn insert_global(&mut self, def_id: DefId, scheme: Scheme) {
        self.globals.insert(def_id, scheme);
    }

    /// Insert a local binding in the current scope.
    pub fn insert_local(&mut self, name: Symbol, scheme: Scheme) {
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name, scheme);
        }
    }

    /// Look up a global binding by `DefId`.
    #[must_use]
    pub fn lookup_global(&self, def_id: DefId) -> Option<&Scheme> {
        self.globals.get(&def_id)
    }

    /// Look up a local binding by name (searches all scopes from innermost).
    #[must_use]
    pub fn lookup_local(&self, name: Symbol) -> Option<&Scheme> {
        for scope in self.locals.iter().rev() {
            if let Some(scheme) = scope.get(&name) {
                return Some(scheme);
            }
        }
        None
    }

    /// Look up a binding (checks locals first, then globals by name).
    /// Note: For `DefRef` lookups, use `lookup_def_id` directly.
    #[must_use]
    pub fn lookup_by_name(&self, name: Symbol) -> Option<&Scheme> {
        // Check locals first
        if let Some(scheme) = self.lookup_local(name) {
            return Some(scheme);
        }

        // For global lookup by name, we'd need a separate mapping
        // In practice, use lookup_def_id for DefRef-based lookups
        None
    }

    /// Look up a binding by `DefId` (for resolved references).
    #[must_use]
    pub fn lookup_def_id(&self, def_id: DefId) -> Option<&Scheme> {
        // Check if it's a data constructor
        if let Some(info) = self.data_cons_by_id.get(&def_id) {
            return Some(&info.scheme);
        }

        // Check globals
        self.globals.get(&def_id)
    }

    /// Register a type constructor.
    pub fn register_type_con(&mut self, tycon: TyCon) {
        self.type_cons.insert(tycon.name, tycon);
    }

    /// Look up a type constructor by name.
    #[must_use]
    pub fn lookup_type_con(&self, name: Symbol) -> Option<&TyCon> {
        self.type_cons.get(&name)
    }

    /// Register a data constructor.
    pub fn register_data_con(&mut self, def_id: DefId, name: Symbol, scheme: Scheme) {
        let info = DataConInfo {
            def_id,
            name,
            scheme,
        };
        self.data_cons.insert(name, info.clone());
        self.data_cons_by_id.insert(def_id, info);
    }

    /// Register a built-in value (function) with both a DefId and name.
    ///
    /// This is used for built-in functions like `toDynamic`, `fromDynamic`, etc.
    /// The value can be looked up by both `DefId` and name.
    pub fn register_value(&mut self, def_id: DefId, name: Symbol, scheme: Scheme) {
        // Register in globals by DefId
        self.globals.insert(def_id, scheme.clone());
        // Also make it available by name in the outermost local scope
        if let Some(scope) = self.locals.first_mut() {
            scope.insert(name, scheme);
        }
    }

    /// Insert a global binding by name only (for class methods).
    ///
    /// This is used for type class methods which don't have a DefId at definition
    /// time, but need to be available for lookup by name in expressions.
    pub fn insert_global_by_name(&mut self, name: Symbol, scheme: Scheme) {
        // Insert into the outermost local scope so it's found by lookup_by_name
        if let Some(scope) = self.locals.first_mut() {
            scope.insert(name, scheme);
        }
    }

    /// Look up a data constructor by name.
    #[must_use]
    pub fn lookup_data_con(&self, name: Symbol) -> Option<&DataConInfo> {
        self.data_cons.get(&name)
    }

    /// Look up a data constructor by `DefId`.
    #[must_use]
    pub fn lookup_data_con_by_id(&self, def_id: DefId) -> Option<&DataConInfo> {
        self.data_cons_by_id.get(&def_id)
    }

    /// Register a type class.
    pub fn register_class(&mut self, info: ClassInfo) {
        self.classes.insert(info.name, info);
    }

    /// Look up a type class by name.
    #[must_use]
    pub fn lookup_class(&self, name: Symbol) -> Option<&ClassInfo> {
        self.classes.get(&name)
    }

    /// Register a type class instance.
    pub fn register_instance(&mut self, info: InstanceInfo) {
        self.instances
            .entry(info.class)
            .or_insert_with(Vec::new)
            .push(info);
    }

    /// Look up all instances for a class.
    #[must_use]
    pub fn lookup_instances(&self, class: Symbol) -> Option<&[InstanceInfo]> {
        self.instances.get(&class).map(|v| v.as_slice())
    }

    /// Resolve an instance for a class and type.
    ///
    /// Returns the instance info and a substitution mapping type variables
    /// in the instance head to concrete types. For example, matching
    /// `instance Eq a => Eq [a]` against `Eq [Int]` returns `{a -> Int}`.
    #[must_use]
    pub fn resolve_instance(
        &self,
        class: Symbol,
        ty: &Ty,
    ) -> Option<(&InstanceInfo, bhc_types::Subst)> {
        let instances = self.instances.get(&class)?;
        for inst in instances {
            if !inst.types.is_empty() {
                if let Some(subst) = bhc_types::types_match(&inst.types[0], ty) {
                    return Some((inst, subst));
                }
            }
        }
        None
    }

    /// Resolve an instance for a class with multiple type arguments.
    #[must_use]
    pub fn resolve_instance_multi(
        &self,
        class: Symbol,
        types: &[Ty],
    ) -> Option<(&InstanceInfo, bhc_types::Subst)> {
        let instances = self.instances.get(&class)?;
        for inst in instances {
            if let Some(subst) = bhc_types::types_match_multi(&inst.types, types) {
                return Some((inst, subst));
            }
        }
        None
    }

    /// Get all free type variables in the environment.
    ///
    /// This is used during generalization to determine which type variables
    /// should not be generalized (because they're free in the environment).
    #[must_use]
    pub fn free_vars(&self) -> Vec<TyVar> {
        let mut vars = Vec::new();

        // Collect free vars from globals
        for scheme in self.globals.values() {
            collect_scheme_free_vars(scheme, &mut vars);
        }

        // Collect free vars from locals
        for scope in &self.locals {
            for scheme in scope.values() {
                collect_scheme_free_vars(scheme, &mut vars);
            }
        }

        vars
    }

    /// Look up which class defines an associated type.
    ///
    /// Returns the class info and the associated type info if found.
    #[must_use]
    pub fn lookup_assoc_type(&self, name: Symbol) -> Option<(&ClassInfo, &AssocTypeInfo)> {
        for class in self.classes.values() {
            for assoc in &class.assoc_types {
                if assoc.name == name {
                    return Some((class, assoc));
                }
            }
        }
        None
    }

    /// Try to reduce a type family application.
    ///
    /// Given an associated type family name and its arguments, tries to find
    /// a matching instance and reduce to the concrete type.
    ///
    /// For example, if we have:
    /// - `class Collection c where type Elem c`
    /// - `instance Collection [a] where type Elem [a] = a`
    ///
    /// Then `reduce_type_family("Elem", [List Int])` returns `Some(Int)`.
    ///
    /// If an instance doesn't provide an implementation but the class has
    /// a default, the default is used instead:
    /// - `class Wrapper w where type Unwrap w; type Unwrap w = w`
    /// - `instance Wrapper Int`  -- no type Unwrap Int provided
    ///
    /// Then `reduce_type_family("Unwrap", [Int])` returns `Some(Int)`.
    #[must_use]
    pub fn reduce_type_family(&self, family_name: Symbol, args: &[Ty]) -> Option<Ty> {
        // Find which class defines this associated type
        let (class_info, assoc_info) = self.lookup_assoc_type(family_name)?;

        // Get all instances for this class
        let instances = self.instances.get(&class_info.name)?;

        // Try to find a matching instance
        for instance in instances {
            // Try to match the instance types against our arguments
            if let Some(subst) =
                self.match_instance_types(&instance.types, args, &class_info.params)
            {
                // Find the associated type implementation in this instance
                for impl_ in &instance.assoc_type_impls {
                    if impl_.name == family_name {
                        // Apply the substitution to the RHS
                        return Some(subst.apply(&impl_.rhs));
                    }
                }

                // Instance matched but doesn't provide this associated type.
                // Check if the class has a default.
                if let Some(default_ty) = &assoc_info.default {
                    // The default is written in terms of class type parameters.
                    // We need to substitute those with the concrete instance types.
                    //
                    // For example, if:
                    //   class Wrapper w where type Unwrap w = w
                    //   instance Wrapper Int
                    //
                    // The default `w` needs to be substituted with `Int`.
                    let default_subst = self.build_class_param_subst(&class_info.params, args);
                    return Some(default_subst.apply(default_ty));
                }
            }
        }

        None
    }

    /// Build a substitution from class type parameters to concrete argument types.
    ///
    /// For example, if class params are [a, b] and args are [Int, Bool],
    /// returns a substitution { a -> Int, b -> Bool }.
    fn build_class_param_subst(&self, class_params: &[TyVar], args: &[Ty]) -> bhc_types::Subst {
        use bhc_types::Subst;

        let mut subst = Subst::new();
        for (param, arg) in class_params.iter().zip(args.iter()) {
            subst.insert(param, arg.clone());
        }
        subst
    }

    /// Try to match instance types against arguments.
    ///
    /// Returns a substitution from type variables to concrete types if successful.
    fn match_instance_types(
        &self,
        instance_types: &[Ty],
        args: &[Ty],
        _class_params: &[TyVar],
    ) -> Option<bhc_types::Subst> {
        use bhc_types::Subst;

        if instance_types.len() != args.len() {
            return None;
        }

        // Collect all free type variables from the instance types.
        // These are the bindable variables for pattern matching.
        let mut bindable_vars = Vec::new();
        for inst_ty in instance_types {
            for v in inst_ty.free_vars() {
                if !bindable_vars.contains(&v) {
                    bindable_vars.push(v);
                }
            }
        }

        let mut subst = Subst::new();

        for (inst_ty, arg_ty) in instance_types.iter().zip(args.iter()) {
            // Try to unify the instance type with the argument
            if let Some(s) = self.match_types(inst_ty, arg_ty, &bindable_vars) {
                subst = subst.compose(&s);
            } else {
                return None;
            }
        }

        Some(subst)
    }

    /// Match a single type pattern against a concrete type.
    ///
    /// This is a simple one-way pattern match (instance type is a pattern).
    fn match_types(
        &self,
        pattern: &Ty,
        concrete: &Ty,
        bound_vars: &[TyVar],
    ) -> Option<bhc_types::Subst> {
        use bhc_types::Subst;

        match (pattern, concrete) {
            // Type variable in pattern: bind it
            (Ty::Var(v), ty) => {
                // Check if this is a bound variable from the instance
                let is_bound = bound_vars.iter().any(|bv| bv.id == v.id);
                if is_bound || pattern == concrete {
                    let mut subst = Subst::new();
                    subst.insert(v, ty.clone());
                    Some(subst)
                } else {
                    None
                }
            }
            // Same constructors: match recursively
            (Ty::Con(c1), Ty::Con(c2)) if c1.name == c2.name => Some(Subst::new()),
            // Application: match both parts
            (Ty::App(f1, a1), Ty::App(f2, a2)) => {
                let s1 = self.match_types(f1, f2, bound_vars)?;
                let a1_applied = s1.apply(a1);
                let s2 = self.match_types(&a1_applied, a2, bound_vars)?;
                Some(s1.compose(&s2))
            }
            // Function types: match both sides
            (Ty::Fun(a1, r1), Ty::Fun(a2, r2)) => {
                let s1 = self.match_types(a1, a2, bound_vars)?;
                let r1_applied = s1.apply(r1);
                let s2 = self.match_types(&r1_applied, r2, bound_vars)?;
                Some(s1.compose(&s2))
            }
            // Tuples: match element-wise
            (Ty::Tuple(ts1), Ty::Tuple(ts2)) if ts1.len() == ts2.len() => {
                let mut subst = Subst::new();
                for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                    let t1_applied = subst.apply(t1);
                    let s = self.match_types(&t1_applied, t2, bound_vars)?;
                    subst = subst.compose(&s);
                }
                Some(subst)
            }
            // Exact match for other cases
            (t1, t2) if t1 == t2 => Some(Subst::new()),
            // No match
            _ => None,
        }
    }
}

/// Collect free type variables from a scheme.
fn collect_scheme_free_vars(scheme: &Scheme, vars: &mut Vec<TyVar>) {
    let ty_vars = scheme.ty.free_vars();
    for v in ty_vars {
        // Only add if not bound by the scheme
        if !scheme.vars.contains(&v) && !vars.contains(&v) {
            vars.push(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_types::{Kind, Ty};

    #[test]
    fn test_scope_push_pop() {
        let mut env = TypeEnv::new();
        let x = Symbol::intern("x");
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));

        // Insert in outer scope
        env.insert_local(x, Scheme::mono(int_ty.clone()));
        assert!(env.lookup_local(x).is_some());

        // Push new scope
        env.push_scope();

        // Shadow x
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));
        env.insert_local(x, Scheme::mono(bool_ty.clone()));

        // Should find the shadowed binding
        let found = env.lookup_local(x).unwrap();
        assert_eq!(found.ty, bool_ty);

        // Pop scope
        env.pop_scope();

        // Should find the original binding
        let found = env.lookup_local(x).unwrap();
        assert_eq!(found.ty, int_ty);
    }

    #[test]
    fn test_global_lookup() {
        let mut env = TypeEnv::new();
        let def_id = DefId::new(42);
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));

        env.insert_global(def_id, Scheme::mono(int_ty.clone()));

        let found = env.lookup_global(def_id).unwrap();
        assert_eq!(found.ty, int_ty);
    }

    #[test]
    fn test_type_con_registration() {
        let mut env = TypeEnv::new();
        let name = Symbol::intern("Maybe");
        let tycon = TyCon::new(name, Kind::star_to_star());

        env.register_type_con(tycon.clone());

        let found = env.lookup_type_con(name).unwrap();
        assert_eq!(found.name, name);
    }

    #[test]
    fn test_lookup_assoc_type() {
        let mut env = TypeEnv::new();

        // Register a class with an associated type
        let collection = Symbol::intern("Collection");
        let elem = Symbol::intern("Elem");
        let c_var = TyVar::new_star(0);

        let class_info = ClassInfo {
            name: collection,
            params: vec![c_var.clone()],
            supers: vec![],
            methods: FxHashMap::default(),
            fundeps: vec![],
            assoc_types: vec![AssocTypeInfo {
                name: elem,
                params: vec![c_var.clone()],
                kind: Kind::Star,
                default: None,
            }],
        };

        env.classes.insert(collection, class_info);

        // Should find the associated type
        let result = env.lookup_assoc_type(elem);
        assert!(result.is_some());
        let (class_info, assoc_info) = result.unwrap();
        assert_eq!(class_info.name, collection);
        assert_eq!(assoc_info.name, elem);
    }

    #[test]
    fn test_lookup_assoc_type_not_found() {
        let env = TypeEnv::new();
        let unknown = Symbol::intern("Unknown");

        // Should not find a non-existent associated type
        assert!(env.lookup_assoc_type(unknown).is_none());
    }

    #[test]
    fn test_reduce_type_family_simple() {
        let mut env = TypeEnv::new();

        // Set up: class Collection c where type Elem c
        let collection = Symbol::intern("Collection");
        let elem = Symbol::intern("Elem");
        let list_tycon = Symbol::intern("List");
        let int_tycon = Symbol::intern("Int");
        let c_var = TyVar::new_star(0);
        let a_var = TyVar::new_star(1);

        let class_info = ClassInfo {
            name: collection,
            params: vec![c_var.clone()],
            supers: vec![],
            methods: FxHashMap::default(),
            fundeps: vec![],
            assoc_types: vec![AssocTypeInfo {
                name: elem,
                params: vec![c_var.clone()],
                kind: Kind::Star,
                default: None,
            }],
        };

        env.classes.insert(collection, class_info);

        // Set up: instance Collection [a] where type Elem [a] = a
        // The instance type is List a (which is Ty::App(List, a))
        let list_con = Ty::Con(TyCon::new(list_tycon, Kind::star_to_star()));
        let a_ty = Ty::Var(a_var.clone());
        let list_a = Ty::App(Box::new(list_con.clone()), Box::new(a_ty.clone()));

        let instance_info = InstanceInfo {
            class: collection,
            types: vec![list_a.clone()],
            methods: FxHashMap::default(),
            assoc_type_impls: vec![AssocTypeImpl {
                name: elem,
                args: vec![list_a.clone()],
                rhs: a_ty.clone(),
            }],
        };

        env.instances
            .entry(collection)
            .or_default()
            .push(instance_info);

        // Now reduce: Elem [Int] -> Int
        let int_ty = Ty::Con(TyCon::new(int_tycon, Kind::Star));
        let list_int = Ty::App(Box::new(list_con), Box::new(int_ty.clone()));

        let result = env.reduce_type_family(elem, &[list_int]);
        assert!(result.is_some());
        // The result should be Int (the substituted type)
        let reduced = result.unwrap();
        assert_eq!(reduced, int_ty);
    }

    #[test]
    fn test_reduce_type_family_no_matching_instance() {
        let mut env = TypeEnv::new();

        // Set up: class Collection c where type Elem c
        let collection = Symbol::intern("Collection");
        let elem = Symbol::intern("Elem");
        let c_var = TyVar::new_star(0);

        let class_info = ClassInfo {
            name: collection,
            params: vec![c_var.clone()],
            supers: vec![],
            methods: FxHashMap::default(),
            fundeps: vec![],
            assoc_types: vec![AssocTypeInfo {
                name: elem,
                params: vec![c_var],
                kind: Kind::Star,
                default: None,
            }],
        };

        env.classes.insert(collection, class_info);

        // No instances registered - should return None
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let result = env.reduce_type_family(elem, &[int_ty]);
        assert!(result.is_none());
    }

    #[test]
    fn test_match_types_variable() {
        let env = TypeEnv::new();
        let a_var = TyVar::new_star(0);
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));

        // Pattern: a, Concrete: Int, should bind a -> Int
        let pattern = Ty::Var(a_var.clone());
        let result = env.match_types(&pattern, &int_ty, &[a_var.clone()]);

        assert!(result.is_some());
        let subst = result.unwrap();
        assert_eq!(subst.apply(&pattern), int_ty);
    }

    #[test]
    fn test_match_types_constructor() {
        let env = TypeEnv::new();
        let int_sym = Symbol::intern("Int");
        let int_ty = Ty::Con(TyCon::new(int_sym, Kind::Star));

        // Pattern: Int, Concrete: Int, should match
        let result = env.match_types(&int_ty, &int_ty, &[]);
        assert!(result.is_some());

        // Pattern: Int, Concrete: Bool, should not match
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));
        let result = env.match_types(&int_ty, &bool_ty, &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_match_types_application() {
        let env = TypeEnv::new();
        let a_var = TyVar::new_star(0);
        let list_sym = Symbol::intern("List");
        let int_sym = Symbol::intern("Int");

        // Pattern: List a
        let list_con = Ty::Con(TyCon::new(list_sym, Kind::star_to_star()));
        let pattern = Ty::App(Box::new(list_con.clone()), Box::new(Ty::Var(a_var.clone())));

        // Concrete: List Int
        let int_ty = Ty::Con(TyCon::new(int_sym, Kind::Star));
        let concrete = Ty::App(Box::new(list_con), Box::new(int_ty.clone()));

        let result = env.match_types(&pattern, &concrete, &[a_var.clone()]);
        assert!(result.is_some());
        let subst = result.unwrap();
        assert_eq!(subst.apply(&Ty::Var(a_var)), int_ty);
    }

    #[test]
    fn test_reduce_type_family_with_default() {
        let mut env = TypeEnv::new();

        // Set up: class Wrapper w where type Unwrap w; type Unwrap w = w
        // The default says: Unwrap w defaults to w itself
        let wrapper = Symbol::intern("Wrapper");
        let unwrap = Symbol::intern("Unwrap");
        let int_tycon = Symbol::intern("Int");
        let w_var = TyVar::new_star(0);

        let class_info = ClassInfo {
            name: wrapper,
            params: vec![w_var.clone()],
            supers: vec![],
            methods: FxHashMap::default(),
            fundeps: vec![],
            assoc_types: vec![AssocTypeInfo {
                name: unwrap,
                params: vec![w_var.clone()],
                kind: Kind::Star,
                // Default: type Unwrap w = w
                default: Some(Ty::Var(w_var.clone())),
            }],
        };

        env.classes.insert(wrapper, class_info);

        // Set up: instance Wrapper Int (no type Unwrap Int provided)
        let int_ty = Ty::Con(TyCon::new(int_tycon, Kind::Star));

        let instance_info = InstanceInfo {
            class: wrapper,
            types: vec![int_ty.clone()],
            methods: FxHashMap::default(),
            // No associated type implementation - should use default
            assoc_type_impls: vec![],
        };

        env.instances
            .entry(wrapper)
            .or_default()
            .push(instance_info);

        // Now reduce: Unwrap Int -> Int (using the default)
        let result = env.reduce_type_family(unwrap, &[int_ty.clone()]);
        assert!(result.is_some());
        // The default is `w`, which gets substituted with `Int`
        let reduced = result.unwrap();
        assert_eq!(reduced, int_ty);
    }

    #[test]
    fn test_reduce_type_family_explicit_overrides_default() {
        let mut env = TypeEnv::new();

        // Set up: class Wrapper w where type Unwrap w; type Unwrap w = w
        let wrapper = Symbol::intern("Wrapper");
        let unwrap = Symbol::intern("Unwrap");
        let maybe_tycon = Symbol::intern("Maybe");
        let a_var = TyVar::new_star(0);
        let w_var = TyVar::new_star(1);

        let class_info = ClassInfo {
            name: wrapper,
            params: vec![w_var.clone()],
            supers: vec![],
            methods: FxHashMap::default(),
            fundeps: vec![],
            assoc_types: vec![AssocTypeInfo {
                name: unwrap,
                params: vec![w_var.clone()],
                kind: Kind::Star,
                // Default: type Unwrap w = w
                default: Some(Ty::Var(w_var.clone())),
            }],
        };

        env.classes.insert(wrapper, class_info);

        // Set up: instance Wrapper (Maybe a) where type Unwrap (Maybe a) = a
        // This OVERRIDES the default
        let maybe_con = Ty::Con(TyCon::new(maybe_tycon, Kind::star_to_star()));
        let a_ty = Ty::Var(a_var.clone());
        let maybe_a = Ty::App(Box::new(maybe_con.clone()), Box::new(a_ty.clone()));

        let instance_info = InstanceInfo {
            class: wrapper,
            types: vec![maybe_a.clone()],
            methods: FxHashMap::default(),
            // Explicit implementation overrides default
            assoc_type_impls: vec![AssocTypeImpl {
                name: unwrap,
                args: vec![maybe_a.clone()],
                rhs: a_ty.clone(),
            }],
        };

        env.instances
            .entry(wrapper)
            .or_default()
            .push(instance_info);

        // Now reduce: Unwrap (Maybe Int) -> Int (using the explicit impl, not default)
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let maybe_int = Ty::App(Box::new(maybe_con), Box::new(int_ty.clone()));

        let result = env.reduce_type_family(unwrap, &[maybe_int]);
        assert!(result.is_some());
        // The explicit implementation gives `a`, which gets substituted with `Int`
        let reduced = result.unwrap();
        assert_eq!(reduced, int_ty);
    }

    #[test]
    fn test_reduce_type_family_default_with_complex_type() {
        let mut env = TypeEnv::new();

        // Set up: class Container c where type Element c; type Element c = c
        // Here the default is the container itself
        let container = Symbol::intern("Container");
        let element = Symbol::intern("Element");
        let list_tycon = Symbol::intern("List");
        let int_tycon = Symbol::intern("Int");
        let c_var = TyVar::new_star(0);
        let a_var = TyVar::new_star(1);

        let class_info = ClassInfo {
            name: container,
            params: vec![c_var.clone()],
            supers: vec![],
            methods: FxHashMap::default(),
            fundeps: vec![],
            assoc_types: vec![AssocTypeInfo {
                name: element,
                params: vec![c_var.clone()],
                kind: Kind::Star,
                // Default: type Element c = c (identity)
                default: Some(Ty::Var(c_var.clone())),
            }],
        };

        env.classes.insert(container, class_info);

        // Set up: instance Container [a] (no type Element [a] provided)
        let list_con = Ty::Con(TyCon::new(list_tycon, Kind::star_to_star()));
        let a_ty = Ty::Var(a_var.clone());
        let list_a = Ty::App(Box::new(list_con.clone()), Box::new(a_ty));

        let instance_info = InstanceInfo {
            class: container,
            types: vec![list_a.clone()],
            methods: FxHashMap::default(),
            // No implementation - uses default
            assoc_type_impls: vec![],
        };

        env.instances
            .entry(container)
            .or_default()
            .push(instance_info);

        // Now reduce: Element [Int] -> [Int] (the default is the container itself)
        let int_ty = Ty::Con(TyCon::new(int_tycon, Kind::Star));
        let list_int = Ty::App(Box::new(list_con), Box::new(int_ty));

        let result = env.reduce_type_family(element, &[list_int.clone()]);
        assert!(result.is_some());
        // The default is `c`, which gets substituted with `[Int]`
        let reduced = result.unwrap();
        assert_eq!(reduced, list_int);
    }
}
