//! Type environment with scoping support.
//!
//! This module provides [`TypeEnv`], which tracks type bindings at various
//! scopes during type checking. It supports:
//!
//! - Global definitions (module-level)
//! - Local bindings (let, lambda, case patterns)
//! - Type constructors
//! - Data constructors

use bhc_hir::DefId;
use bhc_intern::Symbol;
use bhc_types::{Scheme, TyCon, TyVar};
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

/// The type environment during type checking.
///
/// Maintains bindings at various scopes:
/// - Global: Module-level definitions
/// - Local: Lambda-bound, let-bound, and pattern-bound variables
/// - Type constructors: Type names to `TyCon`
/// - Data constructors: Constructor names to `DataConInfo`
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
}
