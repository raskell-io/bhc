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
    /// Exported data constructors.
    pub constructors: FxHashMap<Symbol, DefId>,
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
                    ast::Export::Module(_, _) => None, // Module re-exports handled separately
                })
                .collect()
        })
        .unwrap_or_default();

    // Collect exports from declarations
    for decl in &module.decls {
        collect_decl_exports(ctx, decl, &mut exports, export_all, &explicit_exports);
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
                let export_constructors = export_all || should_export_constructors(type_name, explicit_exports);

                if export_constructors {
                    for con in &data_decl.constrs {
                        let con_name = con.name.name;
                        let con_def_id = ctx.fresh_def_id();
                        ctx.define(con_def_id, con_name, DefKind::Constructor, con.span);
                        exports.constructors.insert(con_name, con_def_id);

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
                let export_constructors = export_all || should_export_constructors(type_name, explicit_exports);
                if export_constructors {
                    let con_name = newtype_decl.constr.name.name;
                    let con_def_id = ctx.fresh_def_id();
                    ctx.define(con_def_id, con_name, DefKind::Constructor, newtype_decl.constr.span);
                    exports.constructors.insert(con_name, con_def_id);

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
                ctx.define(def_id, name, DefKind::Value, foreign.span);
                exports.values.insert(name, def_id);
            }
        }

        ast::Decl::Fixity(_) | ast::Decl::PragmaDecl(_) => {
            // These don't create exports
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

    // Parse the module
    // Use a fresh FileId - in a real implementation we'd track these
    let file_id = FileId::new(0);
    let (module, diagnostics) = bhc_parser::parse_module(&source, file_id);

    // Check for parse errors
    if !diagnostics.is_empty() {
        cache.end_loading(sym);
        let messages: Vec<_> = diagnostics.iter().map(|d| d.message.clone()).collect();
        return Err(LoadError::ParseError {
            path,
            message: messages.join("; "),
        });
    }

    let module = module.ok_or_else(|| LoadError::ParseError {
        path: path.clone(),
        message: "failed to parse module".to_string(),
    })?;

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
pub fn apply_import_spec(
    exports: &ModuleExports,
    spec: &Option<ast::ImportSpec>,
) -> ModuleExports {
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
                        // Handle constructors
                        if let Some(constructors) = cons {
                            for con in constructors {
                                if let Some(&def_id) = exports.constructors.get(&con.name) {
                                    filtered.constructors.insert(con.name, def_id);
                                }
                            }
                        } else {
                            // Type(..) - import all constructors of this type
                            // We need to find constructors associated with this type
                            // For now, import all constructors (conservative)
                            for (&name, &def_id) in &exports.constructors {
                                filtered.constructors.insert(name, def_id);
                            }
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

        // Bind the value in the context (if not already bound)
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

        if ctx.lookup_type(name).is_none() {
            ctx.bind_type(name, def_id);
        }
    }

    // Register constructors
    for (&name, &def_id) in &exports.constructors {
        let qualified = Symbol::intern(&format!("{}.{}", qualifier, name.as_str()));
        ctx.register_qualified_name(qualified, name);

        if ctx.lookup_constructor(name).is_none() {
            ctx.bind_constructor(name, def_id);
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
