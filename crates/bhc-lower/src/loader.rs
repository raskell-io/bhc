//! Module loading and resolution.
//!
//! This module provides functionality to load Haskell modules from the file system,
//! parse them, and collect their exports for name resolution.
//!
//! # Architecture
//!
//! ```text
//! load_module()
//!       │
//!       ▼
//! ┌──────────────────┐    ┌─────────────────┐
//! │  find_module()   │───▶│  ModuleCache    │
//! │  (file lookup)   │    │  (memoization)  │
//! └──────────────────┘    └─────────────────┘
//!       │
//!       ▼
//! ┌──────────────────┐
//! │  parse_module()  │
//! └──────────────────┘
//!       │
//!       ▼
//! ┌──────────────────┐
//! │ collect_exports()│
//! └──────────────────┘
//! ```

use bhc_ast as ast;
use bhc_hir::DefId;
use bhc_intern::Symbol;
use bhc_span::FileId;
use camino::Utf8PathBuf;
use rustc_hash::{FxHashMap, FxHashSet};
use thiserror::Error;

use crate::context::{DefKind, LowerContext};

/// Errors that can occur during module loading.
#[derive(Debug, Error)]
pub enum LoadError {
    /// Module file not found in search paths.
    #[error("module not found: {0}")]
    ModuleNotFound(String),

    /// Circular import detected.
    #[error("circular import detected: {0}")]
    CircularImport(String),

    /// IO error while reading module file.
    #[error("IO error reading {path}: {message}")]
    IoError {
        /// The path that failed to read.
        path: Utf8PathBuf,
        /// The error message.
        message: String,
    },

    /// Parse error in module file.
    #[error("parse error in {path}: {message}")]
    ParseError {
        /// The path with parse errors.
        path: Utf8PathBuf,
        /// The error message.
        message: String,
    },
}

/// Information about an exported constructor.
#[derive(Clone, Debug)]
pub struct ConstructorInfo {
    /// The DefId of the constructor.
    pub def_id: DefId,
    /// The number of fields/arguments the constructor takes.
    pub arity: usize,
    /// The name of the type constructor this belongs to.
    pub type_con_name: Symbol,
    /// The number of type parameters the type has.
    pub type_param_count: usize,
    /// The 0-based tag (position among the type's constructors).
    pub tag: u32,
    /// For record constructors, the ordered list of field names.
    /// None for positional constructors.
    pub field_names: Option<Vec<Symbol>>,
    /// Whether this constructor is a newtype constructor (identity at runtime).
    pub is_newtype: bool,
}

/// Exports collected from a loaded module.
///
/// Contains mappings from names to their DefIds for values, types, and constructors
/// exported by a module.
#[derive(Clone, Debug)]
pub struct ModuleExports {
    /// The module name as a symbol.
    pub name: Symbol,
    /// Exported values (functions, variables).
    pub values: FxHashMap<Symbol, DefId>,
    /// Exported types.
    pub types: FxHashMap<Symbol, DefId>,
    /// Exported data constructors with their arities.
    pub constructors: FxHashMap<Symbol, ConstructorInfo>,
}

impl ModuleExports {
    /// Creates a new empty exports collection for the given module name.
    #[must_use]
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            values: FxHashMap::default(),
            types: FxHashMap::default(),
            constructors: FxHashMap::default(),
        }
    }

    /// Returns true if the module exports nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty() && self.types.is_empty() && self.constructors.is_empty()
    }
}

/// Cache for loaded modules.
///
/// Stores exports from already-loaded modules and tracks modules currently being
/// loaded (for cycle detection).
#[derive(Debug, Default)]
pub struct ModuleCache {
    /// Cached exports from loaded modules.
    exports: FxHashMap<Symbol, ModuleExports>,
    /// Modules currently being loaded (for cycle detection).
    loading: FxHashSet<Symbol>,
}

impl ModuleCache {
    /// Creates a new empty module cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns cached exports for a module, if available.
    #[must_use]
    pub fn get(&self, name: Symbol) -> Option<&ModuleExports> {
        self.exports.get(&name)
    }

    /// Inserts exports for a module into the cache.
    pub fn insert(&mut self, name: Symbol, exports: ModuleExports) {
        self.exports.insert(name, exports);
    }

    /// Marks a module as currently loading. Returns false if already loading (cycle).
    pub fn begin_loading(&mut self, name: Symbol) -> bool {
        self.loading.insert(name)
    }

    /// Marks a module as finished loading.
    pub fn end_loading(&mut self, name: Symbol) {
        self.loading.remove(&name);
    }

    /// Returns true if the module is currently being loaded.
    #[must_use]
    pub fn is_loading(&self, name: Symbol) -> bool {
        self.loading.contains(&name)
    }
}

/// Convert a module name to a relative file path.
///
/// Transforms dots in the module name to directory separators.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(module_name_to_path("XMonad.StackSet"), "XMonad/StackSet.hs");
/// assert_eq!(module_name_to_path("Data.Map.Strict"), "Data/Map/Strict.hs");
/// ```
#[must_use]
pub fn module_name_to_path(name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(format!("{}.hs", name.replace('.', "/")))
}

/// Find a module file in the search paths.
///
/// Returns the full path to the module file if found, or `None` if not found
/// in any of the search paths.
#[must_use]
pub fn find_module_file(name: &str, search_paths: &[Utf8PathBuf]) -> Option<Utf8PathBuf> {
    let relative = module_name_to_path(name);
    search_paths
        .iter()
        .map(|base| base.join(&relative))
        .find(|p| p.exists())
}

/// Collect exports from an AST module.
///
/// Examines the module's declarations and export list to determine which names
/// are exported and creates DefIds for them.
///
/// # Export Rules
///
/// - If there is no export list, all top-level declarations are exported.
/// - If there is an export list, only explicitly listed items are exported.
/// - For data types, constructors and record fields may be exported depending
///   on the export specification.
pub fn collect_exports(module: &ast::Module, ctx: &mut LowerContext) -> ModuleExports {
    let module_name = module.name.as_ref().map_or_else(
        || Symbol::intern("Main"),
        |n| {
            let full_name = n
                .parts
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(".");
            Symbol::intern(&full_name)
        },
    );

    let mut exports = ModuleExports::new(module_name);

    // Determine if we export everything or have an explicit export list
    let export_all = module.exports.is_none();

    // Build set of explicitly exported names if we have an export list
    let explicit_exports: FxHashSet<Symbol> = module
        .exports
        .as_ref()
        .map(|exps| {
            exps.iter()
                .filter_map(|exp| match exp {
                    ast::Export::Var(ident, _) => Some(ident.name),
                    ast::Export::Type(ident, _, _) => Some(ident.name),
                    ast::Export::Pattern(ident, _) => Some(ident.name),
                    ast::Export::Module(_, _) => None, // Module re-exports handled separately
                })
                .collect()
        })
        .unwrap_or_default();

    // Collect exports from declarations
    for decl in &module.decls {
        collect_decl_exports(ctx, decl, &mut exports, export_all, &explicit_exports);
    }

    // Handle re-exported names: if the module has an explicit export list,
    // any exported name not yet in our exports map is likely re-exported from
    // an import. Create stub entries for these so downstream modules can
    // resolve the names. Without this, re-exports like
    //   module Foo (bar) where import Baz (bar)
    // would lose `bar` when Foo is loaded as a dependency.
    if let Some(export_list) = &module.exports {
        for exp in export_list {
            match exp {
                ast::Export::Var(ident, span) => {
                    if !exports.values.contains_key(&ident.name) {
                        let def_id = ctx.fresh_def_id();
                        ctx.define(def_id, ident.name, DefKind::Value, *span);
                        exports.values.insert(ident.name, def_id);
                    }
                }
                ast::Export::Type(ident, cons, span) => {
                    if !exports.types.contains_key(&ident.name) {
                        let def_id = ctx.fresh_def_id();
                        ctx.define(def_id, ident.name, DefKind::Type, *span);
                        exports.types.insert(ident.name, def_id);
                    }
                    // Also export constructors listed with the type
                    if let Some(con_list) = cons {
                        if con_list.is_empty() {
                            // Type(..) — we don't know the constructors, skip
                        } else {
                            for con_ident in con_list {
                                if !exports.values.contains_key(&con_ident.name)
                                    && !exports.constructors.contains_key(&con_ident.name)
                                {
                                    let con_def_id = ctx.fresh_def_id();
                                    ctx.define(
                                        con_def_id,
                                        con_ident.name,
                                        DefKind::Value,
                                        *span,
                                    );
                                    exports.values.insert(con_ident.name, con_def_id);
                                }
                            }
                        }
                    }
                }
                ast::Export::Module(_, _) => {
                    // Module re-exports (e.g., `module Data.Text`) — would need
                    // recursive module loading to handle fully. Skip for now.
                }
                ast::Export::Pattern(ident, span) => {
                    if !exports.values.contains_key(&ident.name) {
                        let def_id = ctx.fresh_def_id();
                        ctx.define(def_id, ident.name, DefKind::Value, *span);
                        exports.values.insert(ident.name, def_id);
                    }
                }
            }
        }
    }

    exports
}

/// Collect exports from a single declaration.
fn collect_decl_exports(
    ctx: &mut LowerContext,
    decl: &ast::Decl,
    exports: &mut ModuleExports,
    export_all: bool,
    explicit_exports: &FxHashSet<Symbol>,
) {
    match decl {
        ast::Decl::FunBind(fun_bind) => {
            // Skip pattern bindings (special name $patbind)
            if fun_bind.name.name.as_str() == "$patbind" {
                // For pattern bindings, we'd need to extract the bound variables
                // For now, skip them
                return;
            }

            let name = fun_bind.name.name;
            if export_all || explicit_exports.contains(&name) {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Value, fun_bind.span);
                exports.values.insert(name, def_id);
            }
        }

        ast::Decl::TypeSig(_type_sig) => {
            // Type signatures don't create new exports; they just annotate
            // function bindings which are handled separately.
        }

        ast::Decl::DataDecl(data_decl) => {
            let type_name = data_decl.name.name;

            // Check if type should be exported
            if export_all || explicit_exports.contains(&type_name) {
                // Export the type
                let type_def_id = ctx.fresh_def_id();
                ctx.define(type_def_id, type_name, DefKind::Type, data_decl.span);
                exports.types.insert(type_name, type_def_id);

                // Determine if constructors should be exported
                // In export_all mode, export all constructors
                // With explicit exports, check for Type(..) syntax
                let export_constructors =
                    export_all || should_export_constructors(type_name, explicit_exports);

                if export_constructors {
                    let type_param_count = data_decl.params.len();

                    for (tag, con) in data_decl.constrs.iter().enumerate() {
                        let con_name = con.name.name;
                        let con_def_id = ctx.fresh_def_id();

                        // Calculate constructor arity and extract field names from fields
                        let (arity, field_names) = match &con.fields {
                            ast::ConFields::Positional(fields) => (fields.len(), None),
                            ast::ConFields::Record(fields) => {
                                let names: Vec<Symbol> =
                                    fields.iter().map(|f| f.name.name).collect();
                                (fields.len(), Some(names))
                            }
                        };
                        ctx.define_constructor_with_type(
                            con_def_id,
                            con_name,
                            con.span,
                            arity,
                            type_name,
                            type_param_count,
                            field_names.clone(),
                        );
                        exports.constructors.insert(
                            con_name,
                            ConstructorInfo {
                                def_id: con_def_id,
                                arity,
                                type_con_name: type_name,
                                type_param_count,
                                tag: tag as u32,
                                field_names,
                                is_newtype: false,
                            },
                        );

                        // Export record field accessors
                        if let ast::ConFields::Record(fields) = &con.fields {
                            for field in fields {
                                let field_name = field.name.name;
                                let field_def_id = ctx.fresh_def_id();
                                ctx.define(field_def_id, field_name, DefKind::Value, field.span);
                                exports.values.insert(field_name, field_def_id);
                            }
                        }
                    }
                }
            }
        }

        ast::Decl::Newtype(newtype_decl) => {
            let type_name = newtype_decl.name.name;

            if export_all || explicit_exports.contains(&type_name) {
                // Export the type
                let type_def_id = ctx.fresh_def_id();
                ctx.define(type_def_id, type_name, DefKind::Type, newtype_decl.span);
                exports.types.insert(type_name, type_def_id);

                // Export the constructor
                let export_constructors =
                    export_all || should_export_constructors(type_name, explicit_exports);
                if export_constructors {
                    let con_name = newtype_decl.constr.name.name;
                    let con_def_id = ctx.fresh_def_id();
                    let type_param_count = newtype_decl.params.len();

                    // Extract field names for record newtypes
                    let field_names = match &newtype_decl.constr.fields {
                        ast::ConFields::Record(fields) => {
                            Some(fields.iter().map(|f| f.name.name).collect())
                        }
                        ast::ConFields::Positional(_) => None,
                    };

                    // Newtypes always have arity 1
                    ctx.define_constructor_with_type(
                        con_def_id,
                        con_name,
                        newtype_decl.constr.span,
                        1,
                        type_name,
                        type_param_count,
                        field_names.clone(),
                    );
                    exports.constructors.insert(
                        con_name,
                        ConstructorInfo {
                            def_id: con_def_id,
                            arity: 1,
                            type_con_name: type_name,
                            type_param_count,
                            tag: 0,
                            field_names,
                            is_newtype: true,
                        },
                    );

                    // Export record field if present
                    if let ast::ConFields::Record(fields) = &newtype_decl.constr.fields {
                        for field in fields {
                            let field_name = field.name.name;
                            let field_def_id = ctx.fresh_def_id();
                            ctx.define(field_def_id, field_name, DefKind::Value, field.span);
                            exports.values.insert(field_name, field_def_id);
                        }
                    }
                }
            }
        }

        ast::Decl::TypeAlias(type_alias) => {
            let name = type_alias.name.name;
            if export_all || explicit_exports.contains(&name) {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Type, type_alias.span);
                exports.types.insert(name, def_id);
            }
        }

        ast::Decl::ClassDecl(class_decl) => {
            let class_name = class_decl.name.name;
            if export_all || explicit_exports.contains(&class_name) {
                // Export the class
                let class_def_id = ctx.fresh_def_id();
                ctx.define(class_def_id, class_name, DefKind::Class, class_decl.span);
                exports.types.insert(class_name, class_def_id);

                // Export class methods
                for method_decl in &class_decl.methods {
                    if let ast::Decl::TypeSig(type_sig) = method_decl {
                        for name_ident in &type_sig.names {
                            let method_name = name_ident.name;
                            let method_def_id = ctx.fresh_def_id();
                            ctx.define(method_def_id, method_name, DefKind::Value, type_sig.span);
                            exports.values.insert(method_name, method_def_id);
                        }
                    }
                }
            }
        }

        ast::Decl::InstanceDecl(_) => {
            // Instance declarations don't directly export names
        }

        ast::Decl::Foreign(foreign) => {
            let name = foreign.name.name;
            if export_all || explicit_exports.contains(&name) {
                let def_id = ctx.fresh_def_id();
                // Lower the foreign import's type and quantify free vars
                // so the type checker sees a proper polymorphic scheme.
                let raw_ty = crate::lower::lower_type(ctx, &foreign.ty);
                let free_vars = raw_ty.free_vars();
                let scheme = if free_vars.is_empty() {
                    bhc_types::Scheme::mono(raw_ty)
                } else {
                    bhc_types::Scheme::poly(free_vars, raw_ty)
                };
                ctx.define_with_type(def_id, name, DefKind::Value, foreign.span, scheme);
                exports.values.insert(name, def_id);
            }
        }

        ast::Decl::Fixity(_)
        | ast::Decl::PragmaDecl(_)
        | ast::Decl::StandaloneDeriving(_)
        | ast::Decl::PatternSynonym(_)
        | ast::Decl::TypeFamilyDecl(_)
        | ast::Decl::TypeInstanceDecl(_)
        | ast::Decl::DataFamilyDecl(_)
        | ast::Decl::DataInstanceDecl(_) => {
            // These don't create exports (data/type families are type-level only,
            // constructors from data instances are handled via their family name)
        }
    }
}

/// Check if constructors should be exported for a type.
///
/// Returns true if the export list contains `Type(..)` for the given type name.
/// Since we don't track the `..` syntax in the simple explicit_exports set,
/// we conservatively return true if the type is exported.
fn should_export_constructors(_type_name: Symbol, _explicit_exports: &FxHashSet<Symbol>) -> bool {
    // TODO: Properly parse export specs to distinguish Type vs Type(..)
    // For now, if the type is explicitly exported, assume constructors are too
    true
}

/// Load a module from the file system, parse it, and collect its exports.
///
/// This function:
/// 1. Checks the cache for already-loaded modules
/// 2. Detects circular imports
/// 3. Finds the module file in the search paths
/// 4. Parses the module
/// 5. Collects and caches the exports
///
/// # Arguments
///
/// * `name` - The module name (e.g., "XMonad.StackSet")
/// * `search_paths` - Directories to search for the module file
/// * `cache` - Module cache for memoization and cycle detection
/// * `ctx` - Lowering context for creating DefIds
///
/// # Returns
///
/// Returns the module exports on success, or a `LoadError` on failure.
pub fn load_module(
    name: &str,
    search_paths: &[Utf8PathBuf],
    cache: &mut ModuleCache,
    ctx: &mut LowerContext,
) -> Result<ModuleExports, LoadError> {
    let sym = Symbol::intern(name);

    // Check cache first
    if let Some(exports) = cache.get(sym) {
        return Ok(exports.clone());
    }

    // Cycle detection
    if !cache.begin_loading(sym) {
        return Err(LoadError::CircularImport(name.to_string()));
    }

    // Find the module file
    let path = find_module_file(name, search_paths)
        .ok_or_else(|| LoadError::ModuleNotFound(name.to_string()))?;

    // Read the file
    let source = std::fs::read_to_string(&path).map_err(|e| LoadError::IoError {
        path: path.clone(),
        message: e.to_string(),
    })?;

    // Apply CPP preprocessing if needed
    let source = if needs_cpp(&source) {
        preprocess_cpp(&source)
    } else {
        source
    };

    // Parse the module
    // Use a fresh FileId - in a real implementation we'd track these
    let file_id = FileId::new(0);
    let (module, diagnostics) = bhc_parser::parse_module(&source, file_id);

    // Only fail if module couldn't be parsed at all (match driver behavior —
    // diagnostics may include non-fatal warnings that don't prevent parsing)
    let module = match module {
        Some(m) => m,
        None => {
            cache.end_loading(sym);
            let messages: Vec<_> = diagnostics.iter().map(|d| d.message.clone()).collect();
            return Err(LoadError::ParseError {
                path,
                message: messages.join("; "),
            });
        }
    };

    // Collect exports
    let exports = collect_exports(&module, ctx);

    // Cache the result
    cache.end_loading(sym);
    cache.insert(sym, exports.clone());

    Ok(exports)
}

/// Apply an import specification to filter module exports.
///
/// Handles both `import M (a, b, c)` (Only) and `import M hiding (a, b)` (Hiding).
pub fn apply_import_spec(exports: &ModuleExports, spec: &Option<ast::ImportSpec>) -> ModuleExports {
    match spec {
        None => exports.clone(), // Import everything
        Some(ast::ImportSpec::Only(items)) => {
            let mut filtered = ModuleExports::new(exports.name);
            for item in items {
                match item {
                    ast::Import::Var(ident, _) => {
                        if let Some(&def_id) = exports.values.get(&ident.name) {
                            filtered.values.insert(ident.name, def_id);
                        }
                    }
                    ast::Import::Type(ident, cons, _) => {
                        if let Some(&def_id) = exports.types.get(&ident.name) {
                            filtered.types.insert(ident.name, def_id);
                        }
                        // Handle constructors/class methods
                        if let Some(constructors) = cons {
                            if constructors.is_empty() {
                                // Type(..) or Class(..) — import all constructors
                                // AND all values (class methods are lowercase values)
                                for (&name, info) in &exports.constructors {
                                    filtered.constructors.insert(name, info.clone());
                                }
                                for (&name, &def_id) in &exports.values {
                                    filtered.values.insert(name, def_id);
                                }
                            } else {
                                for con in constructors {
                                    if let Some(info) = exports.constructors.get(&con.name) {
                                        filtered.constructors.insert(con.name, info.clone());
                                    }
                                    // Also check values (class methods listed explicitly)
                                    if let Some(&def_id) = exports.values.get(&con.name) {
                                        filtered.values.insert(con.name, def_id);
                                    }
                                }
                            }
                        } else {
                            // Type with no constructor list: import M (Foo)
                            // Don't import constructors
                        }
                    }
                    ast::Import::Pattern(ident, _) => {
                        // Pattern synonyms are imported as values
                        if let Some(&def_id) = exports.values.get(&ident.name) {
                            filtered.values.insert(ident.name, def_id);
                        }
                        // Also check constructors (pattern synonyms may be registered there)
                        if let Some(info) = exports.constructors.get(&ident.name) {
                            filtered.constructors.insert(ident.name, info.clone());
                        }
                    }
                }
            }
            filtered
        }
        Some(ast::ImportSpec::Hiding(items)) => {
            let mut filtered = exports.clone();
            for item in items {
                match item {
                    ast::Import::Var(ident, _) => {
                        filtered.values.remove(&ident.name);
                    }
                    ast::Import::Type(ident, cons, _) => {
                        filtered.types.remove(&ident.name);
                        // Remove constructors
                        if let Some(constructors) = cons {
                            for con in constructors {
                                filtered.constructors.remove(&con.name);
                            }
                        } else {
                            // Type(..) - hide all constructors
                            // For now, we'd need to track which constructors belong to which type
                            // This is a simplification
                        }
                    }
                    ast::Import::Pattern(ident, _) => {
                        // Pattern synonyms are removed as values/constructors
                        filtered.values.remove(&ident.name);
                        filtered.constructors.remove(&ident.name);
                    }
                }
            }
            filtered
        }
    }
}

/// Register imported names in the lowering context.
///
/// Registers both the qualified names (Module.name) and, for non-qualified imports,
/// the unqualified names.
pub fn register_imported_names(
    ctx: &mut LowerContext,
    import: &ast::ImportDecl,
    exports: &ModuleExports,
) {
    let module_name = import
        .module
        .parts
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(".");

    // Determine the qualifier to use for qualified access
    let qualifier = if let Some(alias) = &import.alias {
        alias
            .parts
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(".")
    } else {
        module_name.clone()
    };

    // Register values
    for (&name, &def_id) in &exports.values {
        // Register qualified name: Module.name -> name
        let qualified = Symbol::intern(&format!("{}.{}", qualifier, name.as_str()));
        ctx.register_qualified_name(qualified, name);
        // Also bind the qualified name directly so that resolve_qualified_var
        // finds it via lookup_value(aliased_name) without going through the
        // qualified_names indirection (which maps to the unqualified name and
        // might find a different DefId — e.g., a Prelude builtin).
        ctx.bind_value(qualified, def_id);

        // Bind the value in the context (if not already bound by a builtin)
        if ctx.lookup_value(name).is_none() {
            ctx.bind_value(name, def_id);
        }

        // For non-qualified imports, make the unqualified name available
        if !import.qualified {
            // The value is already bound above
        }
    }

    // Register types
    for (&name, &def_id) in &exports.types {
        let qualified = Symbol::intern(&format!("{}.{}", qualifier, name.as_str()));
        ctx.register_qualified_name(qualified, name);
        ctx.bind_type(qualified, def_id);

        if ctx.lookup_type(name).is_none() {
            ctx.bind_type(name, def_id);
        }
    }

    // Register constructors
    for (&name, info) in &exports.constructors {
        let qualified = Symbol::intern(&format!("{}.{}", qualifier, name.as_str()));
        ctx.register_qualified_name(qualified, name);
        ctx.bind_constructor(qualified, info.def_id);

        if ctx.lookup_constructor(name).is_none() {
            ctx.bind_constructor(name, info.def_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;

    #[test]
    fn test_module_name_to_path() {
        assert_eq!(
            module_name_to_path("XMonad.StackSet"),
            Utf8PathBuf::from("XMonad/StackSet.hs")
        );
        assert_eq!(
            module_name_to_path("Data.Map.Strict"),
            Utf8PathBuf::from("Data/Map/Strict.hs")
        );
        assert_eq!(module_name_to_path("Main"), Utf8PathBuf::from("Main.hs"));
    }

    #[test]
    fn test_module_cache() {
        let mut cache = ModuleCache::new();
        let name = Symbol::intern("Test.Module");

        // Initially empty
        assert!(cache.get(name).is_none());
        assert!(!cache.is_loading(name));

        // Begin loading
        assert!(cache.begin_loading(name));
        assert!(cache.is_loading(name));

        // Can't begin loading again (cycle detection)
        assert!(!cache.begin_loading(name));

        // End loading
        cache.end_loading(name);
        assert!(!cache.is_loading(name));

        // Insert exports
        let exports = ModuleExports::new(name);
        cache.insert(name, exports);
        assert!(cache.get(name).is_some());
    }

    #[test]
    fn test_module_exports_is_empty() {
        let name = Symbol::intern("Test");
        let mut exports = ModuleExports::new(name);
        assert!(exports.is_empty());

        exports.values.insert(Symbol::intern("foo"), DefId::new(0));
        assert!(!exports.is_empty());
    }
}

/// Check if source needs CPP preprocessing by scanning for `{-# LANGUAGE CPP #-}`.
fn needs_cpp(source: &str) -> bool {
    for line in source.lines().take(50) {
        let trimmed = line.trim();
        if trimmed.starts_with("{-#") && trimmed.contains("LANGUAGE") && trimmed.contains("CPP") {
            return true;
        }
        // Stop scanning after module declaration
        if trimmed.starts_with("module ") {
            break;
        }
    }
    false
}

/// Minimal CPP preprocessor for imported modules.
///
/// Handles `#if`/`#ifdef`/`#ifndef`/`#else`/`#elif`/`#endif`/`#define`/`#undef`.
/// Unrecognized `#` directives (like `#include`) are replaced with blank lines.
/// All undefined macros evaluate to false/0.
///
/// This is intentionally minimal — the driver's full CPP preprocessor handles
/// the primary compilation unit. This just needs to be good enough to parse
/// imported modules that use CPP for platform guards.
fn preprocess_cpp(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut defines: FxHashSet<String> = FxHashSet::default();
    // Pre-define some common macros
    defines.insert("__GLASGOW_HASKELL__".to_string());

    // Stack of (active, seen_true_branch) for nested #if
    let mut condition_stack: Vec<(bool, bool)> = Vec::new();

    fn is_active(stack: &[(bool, bool)]) -> bool {
        stack.iter().all(|(active, _)| *active)
    }

    for line in source.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with('#') {
            let directive = trimmed.trim_start_matches('#').trim();

            if directive.starts_with("ifdef ") {
                let macro_name = directive["ifdef ".len()..].trim();
                let macro_name = strip_cpp_line_comment(macro_name);
                let defined = defines.contains(macro_name);
                let parent_active = is_active(&condition_stack);
                let active = parent_active && defined;
                condition_stack.push((active, active));
                output.push('\n');
            } else if directive.starts_with("ifndef ") {
                let macro_name = directive["ifndef ".len()..].trim();
                let macro_name = strip_cpp_line_comment(macro_name);
                let defined = defines.contains(macro_name);
                let parent_active = is_active(&condition_stack);
                let active = parent_active && !defined;
                condition_stack.push((active, active));
                output.push('\n');
            } else if directive.starts_with("if ") {
                // Simplified: treat #if as false unless it references a defined macro
                let expr = directive["if ".len()..].trim();
                let parent_active = is_active(&condition_stack);
                let value = eval_simple_cpp_expr(expr, &defines);
                let active = parent_active && value;
                condition_stack.push((active, active));
                output.push('\n');
            } else if directive.starts_with("elif ") {
                let parent_active = if condition_stack.len() > 1 {
                    condition_stack[..condition_stack.len() - 1]
                        .iter()
                        .all(|(a, _)| *a)
                } else {
                    true
                };
                if let Some(last) = condition_stack.last_mut() {
                    if last.1 {
                        // Already found a true branch, skip this
                        last.0 = false;
                    } else {
                        let expr = directive["elif ".len()..].trim();
                        let value = eval_simple_cpp_expr(expr, &defines);
                        let active = parent_active && value;
                        last.0 = active;
                        if active {
                            last.1 = true;
                        }
                    }
                }
                output.push('\n');
            } else if directive == "else" || directive.starts_with("else ") || directive.starts_with("else/") {
                let parent_active = if condition_stack.len() > 1 {
                    condition_stack[..condition_stack.len() - 1]
                        .iter()
                        .all(|(a, _)| *a)
                } else {
                    true
                };
                if let Some(last) = condition_stack.last_mut() {
                    if last.1 {
                        // Already found a true branch
                        last.0 = false;
                    } else {
                        last.0 = parent_active;
                        last.1 = true;
                    }
                }
                output.push('\n');
            } else if directive == "endif" || directive.starts_with("endif ") || directive.starts_with("endif/") {
                condition_stack.pop();
                output.push('\n');
            } else if directive.starts_with("define ") && is_active(&condition_stack) {
                let rest = directive["define ".len()..].trim();
                if let Some(name) = rest.split_whitespace().next() {
                    defines.insert(name.to_string());
                }
                output.push('\n');
            } else if directive.starts_with("undef ") && is_active(&condition_stack) {
                let name = directive["undef ".len()..].trim();
                let name = strip_cpp_line_comment(name);
                defines.remove(name);
                output.push('\n');
            } else {
                // #include, #warning, #error, etc. — skip
                output.push('\n');
            }
        } else if is_active(&condition_stack) {
            output.push_str(line);
            output.push('\n');
        } else {
            // Inactive branch — emit blank line to preserve line numbers
            output.push('\n');
        }
    }

    output
}

/// Strip trailing C-style line comment from a CPP macro name.
fn strip_cpp_line_comment(s: &str) -> &str {
    if let Some(idx) = s.find("//") {
        s[..idx].trim()
    } else if let Some(idx) = s.find("/*") {
        s[..idx].trim()
    } else {
        s.trim()
    }
}

/// Evaluate a simple CPP `#if` expression.
/// Handles: `defined(X)`, `defined X`, `!defined(X)`, integer literals,
/// `X >= Y` comparisons, `&&`, `||`. Very simplified.
fn eval_simple_cpp_expr(expr: &str, defines: &FxHashSet<String>) -> bool {
    let expr = expr.trim();

    // Handle negation
    if let Some(rest) = expr.strip_prefix('!') {
        return !eval_simple_cpp_expr(rest.trim(), defines);
    }

    // Handle parenthesized expression
    if expr.starts_with('(') {
        if let Some(end) = find_matching_paren(expr) {
            let inner = &expr[1..end];
            if end + 1 >= expr.len() {
                return eval_simple_cpp_expr(inner, defines);
            }
        }
    }

    // Handle defined(X) or defined X
    if let Some(rest) = expr.strip_prefix("defined") {
        let rest = rest.trim();
        let name = if rest.starts_with('(') {
            rest.trim_start_matches('(')
                .trim_end_matches(')')
                .trim()
        } else {
            rest.split_whitespace().next().unwrap_or("")
        };
        return defines.contains(name);
    }

    // Handle && (logical AND)
    if let Some(idx) = expr.find("&&") {
        let left = &expr[..idx];
        let right = &expr[idx + 2..];
        return eval_simple_cpp_expr(left, defines) && eval_simple_cpp_expr(right, defines);
    }

    // Handle || (logical OR)
    if let Some(idx) = expr.find("||") {
        let left = &expr[..idx];
        let right = &expr[idx + 2..];
        return eval_simple_cpp_expr(left, defines) || eval_simple_cpp_expr(right, defines);
    }

    // Handle integer comparisons (>=, <=, >, <, ==, !=)
    // For simplicity, just check if expr is a non-zero integer
    if let Ok(n) = expr.parse::<i64>() {
        return n != 0;
    }

    // Handle __GLASGOW_HASKELL__ >= NNN patterns
    if expr.contains(">=") || expr.contains("<=") || expr.contains("==") || expr.contains("!=") {
        // For GHC version checks, assume we're a modern GHC (9.x = 908)
        // This is a simplification but works for most real-world CPP guards
        if expr.contains("__GLASGOW_HASKELL__") {
            // Pretend GHC 9.8 (908)
            let expr = expr.replace("__GLASGOW_HASKELL__", "908");
            return eval_comparison(&expr);
        }
        return false;
    }

    // Treat unknown identifiers as false (0)
    false
}

/// Evaluate a simple numeric comparison expression.
fn eval_comparison(expr: &str) -> bool {
    if let Some(idx) = expr.find(">=") {
        let left: i64 = expr[..idx].trim().parse().unwrap_or(0);
        let right: i64 = expr[idx + 2..].trim().parse().unwrap_or(0);
        left >= right
    } else if let Some(idx) = expr.find("<=") {
        let left: i64 = expr[..idx].trim().parse().unwrap_or(0);
        let right: i64 = expr[idx + 2..].trim().parse().unwrap_or(0);
        left <= right
    } else if let Some(idx) = expr.find("==") {
        let left: i64 = expr[..idx].trim().parse().unwrap_or(0);
        let right: i64 = expr[idx + 2..].trim().parse().unwrap_or(0);
        left == right
    } else if let Some(idx) = expr.find("!=") {
        let left: i64 = expr[..idx].trim().parse().unwrap_or(0);
        let right: i64 = expr[idx + 2..].trim().parse().unwrap_or(0);
        left != right
    } else if let Some(idx) = expr.find('>') {
        let left: i64 = expr[..idx].trim().parse().unwrap_or(0);
        let right: i64 = expr[idx + 1..].trim().parse().unwrap_or(0);
        left > right
    } else if let Some(idx) = expr.find('<') {
        let left: i64 = expr[..idx].trim().parse().unwrap_or(0);
        let right: i64 = expr[idx + 1..].trim().parse().unwrap_or(0);
        left < right
    } else {
        false
    }
}

/// Find the matching closing parenthesis.
fn find_matching_paren(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}
