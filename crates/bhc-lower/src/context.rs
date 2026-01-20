//! Lowering context and scope management.
//!
//! This module provides the context needed during AST to HIR lowering,
//! including:
//!
//! - Unique ID generation for definitions
//! - Scope management for name resolution
//! - Symbol tables mapping names to definitions

use bhc_hir::{DefId, DefRef, HirId};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use indexmap::IndexMap;
use rustc_hash::FxHashMap;

/// A unique identifier for scopes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScopeId(u32);

impl Idx for ScopeId {
    #[allow(clippy::cast_possible_truncation)]
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// The kind of definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefKind {
    /// A value binding (function or variable).
    Value,
    /// A data type.
    Type,
    /// A data constructor.
    Constructor,
    /// A type class.
    Class,
    /// A type variable.
    TyVar,
    /// A pattern variable.
    PatVar,
}

/// Information about a definition.
#[derive(Clone, Debug)]
pub struct DefInfo {
    /// The unique ID.
    pub id: DefId,
    /// The name.
    pub name: Symbol,
    /// The kind of definition.
    pub kind: DefKind,
    /// Source location.
    pub span: Span,
}

/// A scope containing name bindings.
#[derive(Debug)]
pub struct Scope {
    /// The scope ID.
    pub id: ScopeId,
    /// Parent scope, if any.
    pub parent: Option<ScopeId>,
    /// Value bindings (variables, functions).
    values: FxHashMap<Symbol, DefId>,
    /// Type bindings (types, type constructors).
    types: FxHashMap<Symbol, DefId>,
    /// Constructor bindings.
    constructors: FxHashMap<Symbol, DefId>,
}

impl Scope {
    /// Creates a new scope with the given ID and optional parent.
    fn new(id: ScopeId, parent: Option<ScopeId>) -> Self {
        Self {
            id,
            parent,
            values: FxHashMap::default(),
            types: FxHashMap::default(),
            constructors: FxHashMap::default(),
        }
    }

    /// Binds a value name in this scope.
    pub fn bind_value(&mut self, name: Symbol, def_id: DefId) -> Option<DefId> {
        self.values.insert(name, def_id)
    }

    /// Binds a type name in this scope.
    pub fn bind_type(&mut self, name: Symbol, def_id: DefId) -> Option<DefId> {
        self.types.insert(name, def_id)
    }

    /// Binds a constructor name in this scope.
    pub fn bind_constructor(&mut self, name: Symbol, def_id: DefId) -> Option<DefId> {
        self.constructors.insert(name, def_id)
    }

    /// Looks up a value in this scope only (not parents).
    pub fn lookup_value_local(&self, name: Symbol) -> Option<DefId> {
        self.values.get(&name).copied()
    }

    /// Looks up a type in this scope only (not parents).
    pub fn lookup_type_local(&self, name: Symbol) -> Option<DefId> {
        self.types.get(&name).copied()
    }

    /// Looks up a constructor in this scope only (not parents).
    pub fn lookup_constructor_local(&self, name: Symbol) -> Option<DefId> {
        self.constructors.get(&name).copied()
    }
}

/// Map from `DefId` to definition information.
pub type DefMap = IndexMap<DefId, DefInfo>;

/// The lowering context, holding all state needed during lowering.
pub struct LowerContext {
    /// Next DefId to allocate.
    next_def_id: u32,
    /// Next HirId to allocate.
    next_hir_id: u32,
    /// Next scope ID to allocate.
    next_scope_id: u32,
    /// All scopes.
    scopes: Vec<Scope>,
    /// Current scope.
    current_scope: ScopeId,
    /// Definition information.
    pub defs: DefMap,
    /// Errors collected during lowering.
    pub errors: Vec<crate::LowerError>,
    /// Import aliases: maps alias (e.g., "M") to full module name (e.g., "Data.Map")
    import_aliases: FxHashMap<Symbol, Symbol>,
    /// Qualified imports: maps "Module.name" to the unqualified name for resolution
    qualified_names: FxHashMap<Symbol, Symbol>,
}

impl Default for LowerContext {
    fn default() -> Self {
        Self::new()
    }
}

impl LowerContext {
    /// Creates a new lowering context.
    pub fn new() -> Self {
        // Create root scope
        let root_scope = Scope::new(ScopeId::new(0), None);
        Self {
            next_def_id: 0,
            next_hir_id: 0,
            next_scope_id: 1, // 0 is the root scope
            scopes: vec![root_scope],
            current_scope: ScopeId::new(0),
            defs: IndexMap::default(),
            errors: Vec::new(),
            import_aliases: FxHashMap::default(),
            qualified_names: FxHashMap::default(),
        }
    }

    /// Creates a new lowering context with builtins pre-defined.
    pub fn with_builtins() -> Self {
        let mut ctx = Self::new();
        ctx.define_builtins();
        ctx
    }

    /// Define builtin types and functions.
    fn define_builtins(&mut self) {
        // Define builtin types
        let builtin_types = [
            "Int",
            "Float",
            "Double",
            "Char",
            "Bool",
            "String",
            "IO",
            "Maybe",
            "Either",
        ];

        for name in builtin_types {
            let sym = Symbol::intern(name);
            let def_id = self.fresh_def_id();
            self.define(def_id, sym, DefKind::Type, Span::default());
            self.bind_type(sym, def_id);
        }

        // Define builtin constructors
        // Order MUST match bhc-typeck/src/builtins.rs BUILTIN_*_ID constants
        let builtin_cons = [
            ("True", "Bool"),   // DefId 9
            ("False", "Bool"),  // DefId 10
            ("Nothing", "Maybe"),  // DefId 11
            ("Just", "Maybe"),     // DefId 12
            ("Left", "Either"),    // DefId 13
            ("Right", "Either"),   // DefId 14
            ("[]", "List"),        // DefId 15 - list nil
            (":", "List"),         // DefId 16 - list cons
            ("()", "Unit"),        // DefId 17 - unit
            ("(,)", "Tuple2"),     // DefId 18 - pair constructor
            ("(,,)", "Tuple3"),    // DefId 19 - triple constructor
        ];

        for (con_name, _type_name) in builtin_cons {
            let sym = Symbol::intern(con_name);
            let def_id = self.fresh_def_id();
            self.define(def_id, sym, DefKind::Constructor, Span::default());
            self.bind_constructor(sym, def_id);
        }

        // Define builtin functions (starting at DefId 18)
        let builtin_funcs = [
            // Arithmetic operators
            "+",
            "-",
            "*",
            "/",
            "div",
            "mod",
            "^",
            "^^",
            "**",
            // Comparison operators
            "==",
            "/=",
            "<",
            "<=",
            ">",
            ">=",
            // Boolean operators
            "&&",
            "||",
            // List operators
            ":",
            "++",
            "!!",
            // Function composition
            ".",
            "$",
            // Monadic operators
            ">>=",
            ">>",
            "=<<",
            // Applicative/Functor operators
            "<*>",
            "<$>",
            "<$",
            "*>",
            "<*",
            "fmap",
            // Alternative operator
            "<|>",
            "empty",
            // Semigroup/Monoid operators
            "<>",
            "mempty",
            "mappend",
            "mconcat",
            // Monadic operations
            "return",
            "pure",
            "join",
            "liftM",
            "liftM2",
            "ap",
            "mapM",
            "mapM_",
            "forM",
            "forM_",
            "sequence",
            "sequence_",
            "when",
            "unless",
            "void",
            "liftIO",
            // Reader/State monad operations
            "ask",
            "asks",
            "local",
            "reader",
            "get",
            "gets",
            "put",
            "modify",
            "modify'",
            "state",
            "runReader",
            "runReaderT",
            "runState",
            "runStateT",
            "evalState",
            "evalStateT",
            "execState",
            "execStateT",
            "lift",
            // Exception handling
            "catch",
            "try",
            "throw",
            "throwIO",
            "bracket",
            "bracket_",
            "bracketOnError",
            "finally",
            "onException",
            "handle",
            "handleJust",
            "catchJust",
            "tryJust",
            "evaluate",
            "mask",
            "mask_",
            "uninterruptibleMask",
            "uninterruptibleMask_",
            // IO operations
            "hPutStr",
            "hPutStrLn",
            "hPrint",
            "hGetLine",
            "hGetContents",
            "hClose",
            "openFile",
            "withFile",
            "stdin",
            "stdout",
            "stderr",
            "hFlush",
            "hIsEOF",
            "isEOF",
            "getContents",
            "interact",
            "appendFile",
            // List operations
            "map",
            "filter",
            "foldr",
            "foldl",
            "foldl'",
            "concatMap",
            "head",
            "tail",
            "last",
            "init",
            "length",
            "null",
            "reverse",
            "take",
            "drop",
            "sum",
            "product",
            "maximum",
            "minimum",
            "zip",
            "zipWith",
            "zip3",
            "zipWith3",
            "unzip",
            "unzip3",
            "lines",
            "unlines",
            "words",
            "unwords",
            "concat",
            "intercalate",
            "intersperse",
            "transpose",
            "subsequences",
            "permutations",
            "scanl",
            "scanl'",
            "scanr",
            "iterate",
            "repeat",
            "replicate",
            "cycle",
            "splitAt",
            "span",
            "break",
            "takeWhile",
            "dropWhile",
            "group",
            "inits",
            "tails",
            "isPrefixOf",
            "isSuffixOf",
            "isInfixOf",
            "elem",
            "notElem",
            "lookup",
            "find",
            "partition",
            "nub",
            "delete",
            "union",
            "intersect",
            "sort",
            "sortBy",
            "sortOn",
            "insert",
            "genericLength",
            "genericTake",
            "genericDrop",
            "genericSplitAt",
            "genericIndex",
            "genericReplicate",
            // Prelude functions
            "id",
            "const",
            "flip",
            "error",
            "undefined",
            "seq",
            // Numeric operations
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
            // Comparison
            "compare",
            "min",
            "max",
            // Show
            "show",
            // Boolean
            "not",
            "otherwise",
            // Maybe
            "maybe",
            "fromMaybe",
            // Either
            "either",
            // IO
            "print",
            "putStrLn",
            "putStr",
            "getLine",
            "readFile",
            "writeFile",
            // Guard helper
            "guard",
            // Tuple functions
            "fst",
            "snd",
            "curry",
            "uncurry",
            "swap",
            // Character functions
            "ord",
            "chr",
            "isAlpha",
            "isAlphaNum",
            "isAscii",
            "isControl",
            "isDigit",
            "isHexDigit",
            "isLetter",
            "isLower",
            "isNumber",
            "isPrint",
            "isPunctuation",
            "isSpace",
            "isSymbol",
            "isUpper",
            "toLower",
            "toUpper",
            "digitToInt",
            "intToDigit",
            // Enum functions
            "succ",
            "pred",
            "toEnum",
            "fromEnum",
            "enumFrom",
            "enumFromThen",
            "enumFromTo",
            "enumFromThenTo",
            // Bounded
            "minBound",
            "maxBound",
            // Read functions
            "read",
            "reads",
            "readMaybe",
            "readEither",
            // Numeric conversion
            "toInteger",
            "toRational",
            "realToFrac",
            "fromIntegral",
            "truncate",
            "round",
            "ceiling",
            "floor",
            "even",
            "odd",
            "gcd",
            "lcm",
            "quot",
            "rem",
            "quotRem",
            "divMod",
            "recip",
            // System operations
            "getArgs",
            "getProgName",
            "getEnv",
            "lookupEnv",
            "exitSuccess",
            "exitFailure",
            "exitWith",
            "doesFileExist",
            "doesDirectoryExist",
            "getDirectoryContents",
            "getCurrentDirectory",
            "setCurrentDirectory",
            "createDirectory",
            "removeFile",
            "removeDirectory",
            "renameFile",
            "copyFile",
            "takeFileName",
            "takeDirectory",
            "takeExtension",
            "dropExtension",
            "replaceExtension",
            "splitPath",
            "joinPath",
            "(</>)",
            // Concurrency
            "forkIO",
            "killThread",
            "threadDelay",
            "newMVar",
            "newEmptyMVar",
            "takeMVar",
            "putMVar",
            "readMVar",
            "modifyMVar",
            "modifyMVar_",
            "withMVar",
            "newIORef",
            "readIORef",
            "writeIORef",
            "modifyIORef",
            "modifyIORef'",
            "atomicModifyIORef",
            "atomicModifyIORef'",
            // Control flow
            "forever",
            "replicateM",
            "replicateM_",
            "filterM",
            "foldM",
            "foldM_",
            "zipWithM",
            "zipWithM_",
            "mapAndUnzipM",
            "whenJust",
            "whenM",
            "unlessM",
            // Data.Function
            "on",
            "fix",
            // Data.Maybe (additional)
            "isJust",
            "isNothing",
            "listToMaybe",
            "maybeToList",
            "catMaybes",
            "mapMaybe",
            // Data.Either (additional)
            "isLeft",
            "isRight",
            "lefts",
            "rights",
            "partitionEithers",
            // Control.Applicative
            "optional",
            "some",
            "many",
            // Data.Foldable
            "fold",
            "foldMap",
            "toList",
            "any",
            "all",
            "and",
            "or",
            "asum",
            "msum",
            // Data.Traversable
            "traverse",
            "traverse_",
            "for",
            "for_",
            "sequenceA",
            "sequenceA_",
            // Common type constructors used as functions
            "Just",
            "Nothing",
            "Left",
            "Right",
        ];

        for name in builtin_funcs {
            let sym = Symbol::intern(name);
            let def_id = self.fresh_def_id();
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }
    }

    /// Allocates a fresh `DefId`.
    pub fn fresh_def_id(&mut self) -> DefId {
        let id = DefId::new(self.next_def_id as usize);
        self.next_def_id += 1;
        id
    }

    /// Allocates a fresh `HirId`.
    pub fn fresh_hir_id(&mut self) -> HirId {
        let id = HirId::new(self.next_hir_id as usize);
        self.next_hir_id += 1;
        id
    }

    /// Records a definition.
    pub fn define(&mut self, id: DefId, name: Symbol, kind: DefKind, span: Span) {
        self.defs.insert(
            id,
            DefInfo {
                id,
                name,
                kind,
                span,
            },
        );
    }

    /// Creates a `DefRef` for a definition.
    pub fn def_ref(&self, def_id: DefId, span: Span) -> DefRef {
        DefRef { def_id, span }
    }

    /// Gets the current scope.
    pub fn current_scope(&self) -> &Scope {
        &self.scopes[self.current_scope.index()]
    }

    /// Gets the current scope mutably.
    pub fn current_scope_mut(&mut self) -> &mut Scope {
        let idx = self.current_scope.index();
        &mut self.scopes[idx]
    }

    /// Enters a new scope.
    pub fn enter_scope(&mut self) -> ScopeId {
        let parent = Some(self.current_scope);
        let id = ScopeId::new(self.next_scope_id as usize);
        self.next_scope_id += 1;
        let scope = Scope::new(id, parent);
        self.scopes.push(scope);
        self.current_scope = id;
        id
    }

    /// Exits the current scope, returning to the parent.
    pub fn exit_scope(&mut self) {
        if let Some(parent) = self.scopes[self.current_scope.index()].parent {
            self.current_scope = parent;
        }
    }

    /// Runs a function in a new scope.
    pub fn in_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.enter_scope();
        let result = f(self);
        self.exit_scope();
        result
    }

    /// Binds a value in the current scope.
    pub fn bind_value(&mut self, name: Symbol, def_id: DefId) -> Option<DefId> {
        self.current_scope_mut().bind_value(name, def_id)
    }

    /// Binds a type in the current scope.
    pub fn bind_type(&mut self, name: Symbol, def_id: DefId) -> Option<DefId> {
        self.current_scope_mut().bind_type(name, def_id)
    }

    /// Binds a constructor in the current scope.
    pub fn bind_constructor(&mut self, name: Symbol, def_id: DefId) -> Option<DefId> {
        self.current_scope_mut().bind_constructor(name, def_id)
    }

    /// Looks up a value, searching parent scopes.
    pub fn lookup_value(&self, name: Symbol) -> Option<DefId> {
        let mut scope_id = Some(self.current_scope);
        while let Some(id) = scope_id {
            let scope = &self.scopes[id.index()];
            if let Some(def_id) = scope.lookup_value_local(name) {
                return Some(def_id);
            }
            scope_id = scope.parent;
        }
        None
    }

    /// Looks up a type, searching parent scopes.
    pub fn lookup_type(&self, name: Symbol) -> Option<DefId> {
        let mut scope_id = Some(self.current_scope);
        while let Some(id) = scope_id {
            let scope = &self.scopes[id.index()];
            if let Some(def_id) = scope.lookup_type_local(name) {
                return Some(def_id);
            }
            scope_id = scope.parent;
        }
        None
    }

    /// Looks up a constructor, searching parent scopes.
    pub fn lookup_constructor(&self, name: Symbol) -> Option<DefId> {
        let mut scope_id = Some(self.current_scope);
        while let Some(id) = scope_id {
            let scope = &self.scopes[id.index()];
            if let Some(def_id) = scope.lookup_constructor_local(name) {
                return Some(def_id);
            }
            scope_id = scope.parent;
        }
        None
    }

    /// Records an error.
    pub fn error(&mut self, err: crate::LowerError) {
        self.errors.push(err);
    }

    /// Returns true if any errors were recorded.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Takes all errors from the context.
    pub fn take_errors(&mut self) -> Vec<crate::LowerError> {
        std::mem::take(&mut self.errors)
    }

    /// Registers an import alias.
    ///
    /// For `import qualified Data.Map as M`, call:
    /// `register_import_alias("M", "Data.Map")`
    pub fn register_import_alias(&mut self, alias: Symbol, module: Symbol) {
        self.import_aliases.insert(alias, module);
    }

    /// Registers a qualified name mapping.
    ///
    /// For `import Data.Map (lookup)`, register:
    /// `register_qualified_name("Data.Map.lookup", "lookup")`
    pub fn register_qualified_name(&mut self, qualified: Symbol, unqualified: Symbol) {
        self.qualified_names.insert(qualified, unqualified);
    }

    /// Resolves a qualified variable reference like `M.lookup`.
    ///
    /// Returns the DefId if found, or None if not resolvable.
    pub fn resolve_qualified_var(&self, qualifier: Symbol, name: Symbol) -> Option<DefId> {
        // First, check if the qualifier is an alias
        let module = self.import_aliases.get(&qualifier).copied().unwrap_or(qualifier);

        // Try to look up as "Module.name"
        let qualified_name = Symbol::intern(&format!("{}.{}", module.as_str(), name.as_str()));

        // Check if we have a qualified name mapping
        if let Some(unqualified) = self.qualified_names.get(&qualified_name) {
            if let Some(def_id) = self.lookup_value(*unqualified) {
                return Some(def_id);
            }
        }

        // Try direct lookup of the qualified name
        if let Some(def_id) = self.lookup_value(qualified_name) {
            return Some(def_id);
        }

        // Try looking up the unqualified name directly (for builtins)
        self.lookup_value(name)
    }

    /// Resolves a qualified constructor reference like `M.Just`.
    ///
    /// Returns the DefId if found, or None if not resolvable.
    pub fn resolve_qualified_constructor(&self, qualifier: Symbol, name: Symbol) -> Option<DefId> {
        // First, check if the qualifier is an alias
        let module = self.import_aliases.get(&qualifier).copied().unwrap_or(qualifier);

        // Try to look up as "Module.Name"
        let qualified_name = Symbol::intern(&format!("{}.{}", module.as_str(), name.as_str()));

        // Check if we have a qualified name mapping
        if let Some(unqualified) = self.qualified_names.get(&qualified_name) {
            if let Some(def_id) = self.lookup_constructor(*unqualified) {
                return Some(def_id);
            }
        }

        // Try direct lookup of the qualified name
        if let Some(def_id) = self.lookup_constructor(qualified_name) {
            return Some(def_id);
        }

        // Try looking up the unqualified name directly (for builtins)
        self.lookup_constructor(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_binding() {
        let mut ctx = LowerContext::new();

        let x = Symbol::intern("x");
        let def_id = ctx.fresh_def_id();
        ctx.define(def_id, x, DefKind::Value, Span::default());
        ctx.bind_value(x, def_id);

        assert_eq!(ctx.lookup_value(x), Some(def_id));
    }

    #[test]
    fn test_nested_scopes() {
        let mut ctx = LowerContext::new();

        // Bind x in outer scope
        let x = Symbol::intern("x");
        let outer_def = ctx.fresh_def_id();
        ctx.define(outer_def, x, DefKind::Value, Span::default());
        ctx.bind_value(x, outer_def);

        // Enter inner scope
        ctx.enter_scope();

        // y is only in inner scope
        let y = Symbol::intern("y");
        let inner_def = ctx.fresh_def_id();
        ctx.define(inner_def, y, DefKind::Value, Span::default());
        ctx.bind_value(y, inner_def);

        // Can see both x and y
        assert_eq!(ctx.lookup_value(x), Some(outer_def));
        assert_eq!(ctx.lookup_value(y), Some(inner_def));

        // Exit inner scope
        ctx.exit_scope();

        // Can see x but not y
        assert_eq!(ctx.lookup_value(x), Some(outer_def));
        assert_eq!(ctx.lookup_value(y), None);
    }

    #[test]
    fn test_shadowing() {
        let mut ctx = LowerContext::new();

        let x = Symbol::intern("x");

        // Bind x in outer scope
        let outer_def = ctx.fresh_def_id();
        ctx.define(outer_def, x, DefKind::Value, Span::default());
        ctx.bind_value(x, outer_def);

        // Enter inner scope and shadow x
        ctx.enter_scope();
        let inner_def = ctx.fresh_def_id();
        ctx.define(inner_def, x, DefKind::Value, Span::default());
        ctx.bind_value(x, inner_def);

        // Inner scope sees shadowed x
        assert_eq!(ctx.lookup_value(x), Some(inner_def));

        ctx.exit_scope();

        // Outer scope sees original x
        assert_eq!(ctx.lookup_value(x), Some(outer_def));
    }

    #[test]
    fn test_builtins() {
        let ctx = LowerContext::with_builtins();

        // Check builtin types are defined
        let int = Symbol::intern("Int");
        assert!(ctx.lookup_type(int).is_some());

        // Check builtin constructors are defined
        let true_con = Symbol::intern("True");
        assert!(ctx.lookup_constructor(true_con).is_some());

        // Check builtin functions are defined
        let map_fn = Symbol::intern("map");
        assert!(ctx.lookup_value(map_fn).is_some());
    }
}
