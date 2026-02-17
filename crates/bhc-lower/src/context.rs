//! Lowering context and scope management.
//!
//! This module provides the context needed during AST to HIR lowering,
//! including:
//!
//! - Unique ID generation for definitions
//! - Scope management for name resolution
//! - Symbol tables mapping names to definitions

use bhc_ast as ast;
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
    /// A stub value (external package function not yet implemented).
    StubValue,
    /// A stub type (external package type not yet implemented).
    StubType,
    /// A stub constructor (external package constructor not yet implemented).
    StubConstructor,
}

impl DefKind {
    /// Returns true if this definition kind is a stub (placeholder for external package).
    pub fn is_stub(self) -> bool {
        matches!(
            self,
            DefKind::StubValue | DefKind::StubType | DefKind::StubConstructor
        )
    }
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
    /// For constructors, the number of fields/arguments. None for non-constructors.
    pub arity: Option<usize>,
    /// For constructors, the name of the type constructor. None for non-constructors.
    pub type_con_name: Option<Symbol>,
    /// For constructors, the number of type parameters the type has. None for non-constructors.
    pub type_param_count: Option<usize>,
    /// For record constructors, the ordered list of field names. None for positional constructors.
    pub field_names: Option<Vec<Symbol>>,
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
    /// Warnings collected during lowering.
    pub warnings: Vec<crate::LowerWarning>,
    /// Import aliases: maps alias (e.g., "M") to full module name (e.g., "Data.Map")
    import_aliases: FxHashMap<Symbol, Symbol>,
    /// Qualified imports: maps "Module.name" to the unqualified name for resolution
    qualified_names: FxHashMap<Symbol, Symbol>,
    /// Type signatures: maps function name to its declared type
    type_signatures: FxHashMap<Symbol, ast::Type>,
    /// Class parameter counts: maps class name to number of type parameters.
    /// Used to decompose multi-param instance types during lowering.
    class_param_counts: FxHashMap<Symbol, usize>,
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
            warnings: Vec::new(),
            import_aliases: FxHashMap::default(),
            qualified_names: FxHashMap::default(),
            type_signatures: FxHashMap::default(),
            class_param_counts: FxHashMap::default(),
        }
    }

    /// Creates a new lowering context with builtins pre-defined.
    pub fn with_builtins() -> Self {
        let mut ctx = Self::new();
        ctx.define_builtins();
        ctx.define_stubs();
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
            "Ordering",
            // Additional standard library types
            "NonEmpty",
            "ExitCode",
            "IOMode",
            "BufferMode",
            "Endo",
            "Backwards",
            "All",
            "Any",
            "XdgDirectory",
            "StdStream",
            "TypeRep",
            "SomeException",
            "Permissions",
            "Text", // Data.Text (packed UTF-8)
        ];

        for name in builtin_types {
            let sym = Symbol::intern(name);
            let def_id = self.fresh_def_id();
            self.define(def_id, sym, DefKind::Type, Span::default());
            self.bind_type(sym, def_id);
        }

        // Define builtin constructors with their arities
        // Order MUST match bhc-typeck/src/builtins.rs BUILTIN_*_ID constants
        // Format: (name, type, arity)
        let builtin_cons: &[(&str, &str, usize)] = &[
            ("True", "Bool", 0),     // DefId 9
            ("False", "Bool", 0),    // DefId 10
            ("Nothing", "Maybe", 0), // DefId 11
            ("Just", "Maybe", 1),    // DefId 12
            ("Left", "Either", 1),   // DefId 13
            ("Right", "Either", 1),  // DefId 14
            ("LT", "Ordering", 0),   // Ordering constructors
            ("EQ", "Ordering", 0),
            ("GT", "Ordering", 0),
            ("[]", "List", 0),       // DefId 15 - list nil
            (":", "List", 2),        // DefId 16 - list cons
            ("()", "Unit", 0),       // DefId 17 - unit
            ("(,)", "Tuple2", 2),    // DefId 18 - pair constructor
            ("(,,)", "Tuple3", 3),   // DefId 19 - triple constructor
            // NonEmpty constructor
            (":|", "NonEmpty", 2), // DefId 20 - NonEmpty cons (head :| tail)
            // Control.Applicative.Backwards
            ("Backwards", "Backwards", 1),
            // Data.Monoid
            ("Endo", "Endo", 1),
            // System.Exit
            ("ExitSuccess", "ExitCode", 0),
            ("ExitFailure", "ExitCode", 1),
            // Control.Exception
            ("SomeException", "SomeException", 1),
            // System.IO
            ("ReadMode", "IOMode", 0),
            ("WriteMode", "IOMode", 0),
            ("AppendMode", "IOMode", 0),
            ("ReadWriteMode", "IOMode", 0),
            ("NoBuffering", "BufferMode", 0),
            ("LineBuffering", "BufferMode", 0),
            ("BlockBuffering", "BufferMode", 1), // BlockBuffering (Maybe Int)
            // System.Directory
            ("XdgData", "XdgDirectory", 0),
            ("XdgConfig", "XdgDirectory", 0),
            ("XdgCache", "XdgDirectory", 0),
            // System.Process
            ("CreatePipe", "StdStream", 0),
            ("Inherit", "StdStream", 0),
            ("UseHandle", "StdStream", 1),
            ("NoStream", "StdStream", 0),
            // Data.Typeable
            ("TypeRep", "TypeRep", 1),
            // GHC.Generics
            ("All", "All", 1),
            ("Any", "Any", 1),
            // Note: X11/System.Posix stub constructors are now in define_stubs()
        ];

        for (con_name, _type_name, arity) in builtin_cons {
            let sym = Symbol::intern(con_name);
            let def_id = self.fresh_def_id();
            self.define_constructor(def_id, sym, Span::default(), *arity);
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
            "\\\\", // List difference
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
            "until",
            "asTypeOf",
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
            // Character functions registered at fixed DefIds below (10200+)
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
            "</>",
            // NonEmpty constructor used as infix operator
            ":|",
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
            // Monad fail
            "fail",
            // Control.Applicative.Backwards
            "forwards",
            // Data.Monoid
            "appEndo",
            "getAny",
            "getAll",
            // Control.Arrow
            "first",
            "second",
            "***",
            "&&&",
            "arr",
            "returnA",
            "<<<",
            ">>>",
            // Data.Bits
            ".|.",
            ".&.",
            "xor",
            "complement",
            "shiftL",
            "shiftR",
            "rotateL",
            "rotateR",
            "bit",
            "setBit",
            "clearBit",
            "complementBit",
            "testBit",
            "popCount",
            "zeroBits",
            // Data.Typeable
            "cast",
            "typeOf",
            "typeRep",
            // System.Directory
            "getAppUserDataDirectory",
            "getXdgDirectory",
            "createDirectoryIfMissing",
            "listDirectory",
            "getModificationTime",
            "getPermissions",
            "executable",
            "canonicalizePath",
            // System.FilePath
            "splitExtension",
            // System.Process
            "createProcess_",
            "waitForProcess",
            "proc",
            // System.Exit
            "exitSuccess",
            "exitFailure",
            "exitWith",
            // Data.Version
            "showVersion",
            "version",
            // System.Info
            "compilerName",
            "compilerVersion",
            "arch",
            "os",
            // Data.Ratio
            "%",
            // Data.Function (additional)
            "$!",
            // Control.Exception
            "fromException",
            "toException",
            "displayException",
            // ---- Phase 1 new PrimOps ----
            // Scans
            "scanl1",
            "scanr1",
            // Integral
            "subtract",
            // Data.List "By" variants
            "nubBy",
            "groupBy",
            "deleteBy",
            "unionBy",
            "intersectBy",
            "stripPrefix",
            // Accumulating maps
            "mapAccumL",
            "mapAccumR",
            "unfoldr",
            // Data.Char
            "toTitle",
            "isLatin1",
            "isAsciiLower",
            "isAsciiUpper",
            // Show helpers
            "showString",
            "showChar",
            "showParen",
            // IO
            "getChar",
            // Data.Function
            "&",
            // Container PrimOps: Data.Map
            "Data.Map.empty",
            "Data.Map.singleton",
            "Data.Map.null",
            "Data.Map.size",
            "Data.Map.member",
            "Data.Map.notMember",
            "Data.Map.lookup",
            "Data.Map.findWithDefault",
            "Data.Map.!",
            "Data.Map.insert",
            "Data.Map.insertWith",
            "Data.Map.delete",
            "Data.Map.adjust",
            "Data.Map.update",
            "Data.Map.alter",
            "Data.Map.union",
            "Data.Map.unionWith",
            "Data.Map.unionWithKey",
            "Data.Map.unions",
            "Data.Map.intersection",
            "Data.Map.intersectionWith",
            "Data.Map.difference",
            "Data.Map.differenceWith",
            "Data.Map.map",
            "Data.Map.mapWithKey",
            "Data.Map.mapKeys",
            "Data.Map.filter",
            "Data.Map.filterWithKey",
            "Data.Map.foldr",
            "Data.Map.foldl",
            "Data.Map.foldrWithKey",
            "Data.Map.foldlWithKey",
            "Data.Map.keys",
            "Data.Map.elems",
            "Data.Map.assocs",
            "Data.Map.toList",
            "Data.Map.toAscList",
            "Data.Map.toDescList",
            "Data.Map.fromList",
            "Data.Map.fromListWith",
            "Data.Map.keysSet",
            "Data.Map.isSubmapOf",
            // Container PrimOps: Data.Set
            "Data.Set.empty",
            "Data.Set.singleton",
            "Data.Set.null",
            "Data.Set.size",
            "Data.Set.member",
            "Data.Set.notMember",
            "Data.Set.insert",
            "Data.Set.delete",
            "Data.Set.union",
            "Data.Set.unions",
            "Data.Set.intersection",
            "Data.Set.difference",
            "Data.Set.isSubsetOf",
            "Data.Set.isProperSubsetOf",
            "Data.Set.map",
            "Data.Set.filter",
            "Data.Set.partition",
            "Data.Set.foldr",
            "Data.Set.foldl",
            "Data.Set.toList",
            "Data.Set.toAscList",
            "Data.Set.toDescList",
            "Data.Set.fromList",
            "Data.Set.elems",
            "Data.Set.findMin",
            "Data.Set.findMax",
            "Data.Set.lookupMin",
            "Data.Set.lookupMax",
            "Data.Set.deleteMin",
            "Data.Set.deleteMax",
            // Container PrimOps: Data.IntMap
            "Data.IntMap.empty",
            "Data.IntMap.singleton",
            "Data.IntMap.null",
            "Data.IntMap.size",
            "Data.IntMap.member",
            "Data.IntMap.lookup",
            "Data.IntMap.findWithDefault",
            "Data.IntMap.insert",
            "Data.IntMap.insertWith",
            "Data.IntMap.delete",
            "Data.IntMap.adjust",
            "Data.IntMap.union",
            "Data.IntMap.unionWith",
            "Data.IntMap.intersection",
            "Data.IntMap.difference",
            "Data.IntMap.map",
            "Data.IntMap.mapWithKey",
            "Data.IntMap.filter",
            "Data.IntMap.foldr",
            "Data.IntMap.foldlWithKey",
            "Data.IntMap.keys",
            "Data.IntMap.elems",
            "Data.IntMap.toList",
            "Data.IntMap.toAscList",
            "Data.IntMap.fromList",
            // Container PrimOps: Data.IntSet
            "Data.IntSet.empty",
            "Data.IntSet.singleton",
            "Data.IntSet.null",
            "Data.IntSet.size",
            "Data.IntSet.member",
            "Data.IntSet.insert",
            "Data.IntSet.delete",
            "Data.IntSet.union",
            "Data.IntSet.intersection",
            "Data.IntSet.difference",
            "Data.IntSet.isSubsetOf",
            "Data.IntSet.filter",
            "Data.IntSet.foldr",
            "Data.IntSet.toList",
            "Data.IntSet.fromList",
            // Container PrimOps: Data.Text (packed UTF-8)
            "Data.Text.empty",
            "Data.Text.singleton",
            "Data.Text.pack",
            "Data.Text.unpack",
            "Data.Text.null",
            "Data.Text.length",
            "Data.Text.head",
            "Data.Text.last",
            "Data.Text.tail",
            "Data.Text.init",
            "Data.Text.append",
            "Data.Text.<>",
            "Data.Text.reverse",
            "Data.Text.take",
            "Data.Text.takeEnd",
            "Data.Text.drop",
            "Data.Text.dropEnd",
            "Data.Text.isPrefixOf",
            "Data.Text.isSuffixOf",
            "Data.Text.isInfixOf",
            "Data.Text.toLower",
            "Data.Text.toUpper",
            "Data.Text.toCaseFold",
            "Data.Text.toTitle",
            "Data.Text.map",
            "Data.Text.eq",
            "Data.Text.==",
            "Data.Text.compare",
            // Additional Data.Text operations
            "Data.Text.filter",
            "Data.Text.foldl'",
            "Data.Text.concat",
            "Data.Text.intercalate",
            "Data.Text.strip",
            "Data.Text.words",
            "Data.Text.lines",
            "Data.Text.splitOn",
            "Data.Text.replace",
            // Data.Text.Encoding
            "Data.Text.Encoding.encodeUtf8",
            "Data.Text.Encoding.decodeUtf8",
            // Container PrimOps: Data.ByteString (packed bytes)
            "Data.ByteString.empty",
            "Data.ByteString.singleton",
            "Data.ByteString.pack",
            "Data.ByteString.unpack",
            "Data.ByteString.null",
            "Data.ByteString.length",
            "Data.ByteString.head",
            "Data.ByteString.last",
            "Data.ByteString.tail",
            "Data.ByteString.init",
            "Data.ByteString.append",
            "Data.ByteString.cons",
            "Data.ByteString.snoc",
            "Data.ByteString.take",
            "Data.ByteString.drop",
            "Data.ByteString.reverse",
            "Data.ByteString.elem",
            "Data.ByteString.index",
            "Data.ByteString.eq",
            "Data.ByteString.compare",
            "Data.ByteString.isPrefixOf",
            "Data.ByteString.isSuffixOf",
            "Data.ByteString.readFile",
            "Data.ByteString.writeFile",
            // ---- Phase 3: IO PrimOps (genuinely new) ----
            "hGetChar",
            "hPutChar",
            "hSetBuffering",
            "hGetBuffering",
            "hSeek",
            "hTell",
            "hFileSize",
            "setEnv",
            // ---- Phase 4: Control.* PrimOps (genuinely new) ----
            "liftM3",
            "liftM4",
            "liftM5",
            "mzero",
            "mplus",
            "mfilter",
            ">=>",
            "<=<",
            "liftA",
            "liftA2",
            "liftA3",
            "myThreadId",
            "throwTo",
            // ---- Phase 5: Data.* PrimOps (genuinely new) ----
            "comparing",
            "clamp",
            "foldr'",
            "foldl1",
            "foldr1",
            "maximumBy",
            "minimumBy",
            "fromString",
            "shift",
            "rotate",
            "countLeadingZeros",
            "countTrailingZeros",
            "asProxyTypeOf",
            "absurd",
            "vacuous",
        ];

        for name in builtin_funcs {
            let sym = Symbol::intern(name);
            let def_id = self.fresh_def_id();
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // Register monad transformer builtins at fixed DefIds (10000+).
        // These use explicit DefIds to avoid interfering with the sequential
        // allocation above. Must match bhc-typeck register_transformer_ops
        // and bhc-hir-to-core context.rs.
        let transformer_builtins: &[(usize, &str)] = &[
            // Identity (10000-10006)
            (10000, "Identity"),
            (10001, "runIdentity"),
            // MonadTrans / MonadIO (10010-10012)
            (10010, "lift"),
            (10011, "liftIO"),
            // ReaderT (10020-10031)
            (10020, "ReaderT"),
            (10021, "runReaderT"),
            (10029, "ask"),
            (10030, "asks"),
            (10031, "local"),
            // StateT (10040-10055)
            (10040, "StateT"),
            (10041, "runStateT"),
            (10049, "get"),
            (10050, "put"),
            (10051, "modify"),
            (10053, "gets"),
            (10054, "evalStateT"),
            (10055, "execStateT"),
            // ExceptT (10060-10075)
            (10060, "ExceptT"),
            (10061, "runExceptT"),
            (10062, "throwE"),
            (10063, "catchE"),
            // MonadError standard names (mtl-style aliases)
            (10071, "throwError"),
            (10072, "catchError"),
            // WriterT (10080-10095)
            (10080, "WriterT"),
            (10081, "runWriterT"),
            (10082, "tell"),
            (10083, "execWriterT"),
        ];

        for &(id, name) in transformer_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // Register type-specialized show functions at fixed DefIds (10100+).
        // Must match bhc-typeck builtins.rs register_primitive_ops.
        let show_builtins: &[(usize, &str)] = &[
            (10100, "showInt"),
            (10101, "showDouble"),
            (10102, "showFloat"),
            (10103, "showBool"),
            (10104, "showChar"),
            (10105, "showString"),
            (10106, "showList"),
            (10107, "showMaybe"),
            (10108, "showEither"),
            (10109, "showTuple2"),
            (10110, "showUnit"),
        ];

        for &(id, name) in show_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // Register character functions at fixed DefIds (10200+).
        // Must match bhc-typeck builtins.rs register_primitive_ops.
        let char_builtins: &[(usize, &str)] = &[
            (10200, "ord"),
            (10201, "chr"),
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
            (10216, "toLower"),
            (10217, "toUpper"),
            (10218, "digitToInt"),
            (10219, "intToDigit"),
        ];

        for &(id, name) in char_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // Register Data.Text.IO functions at fixed DefIds (10300+).
        // Must match bhc-typeck builtins.rs register_primitive_ops.
        let text_io_builtins: &[(usize, &str)] = &[
            (10300, "Data.Text.IO.readFile"),
            (10301, "Data.Text.IO.writeFile"),
            (10302, "Data.Text.IO.appendFile"),
            (10303, "Data.Text.IO.hGetContents"),
            (10304, "Data.Text.IO.hGetLine"),
            (10305, "Data.Text.IO.hPutStr"),
            (10306, "Data.Text.IO.hPutStrLn"),
            (10307, "Data.Text.IO.putStr"),
            (10308, "Data.Text.IO.putStrLn"),
            (10309, "Data.Text.IO.getLine"),
            (10310, "Data.Text.IO.getContents"),
        ];

        for &(id, name) in text_io_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // Numeric operations at fixed DefIds 10500-10507
        let numeric_builtins: &[(usize, &str)] = &[
            (10500, "even"),
            (10501, "odd"),
            (10502, "gcd"),
            (10503, "lcm"),
            (10504, "quot"),
            (10505, "rem"),
            (10506, "quotRem"),
            (10507, "divMod"),
        ];

        for &(id, name) in numeric_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // IORef operations at fixed DefIds 10400-10404
        let ioref_builtins: &[(usize, &str)] = &[
            (10400, "newIORef"),
            (10401, "readIORef"),
            (10402, "writeIORef"),
            (10403, "modifyIORef"),
            (10404, "modifyIORef'"),
        ];

        for &(id, name) in ioref_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // Data.Maybe / Data.Either / guard at fixed DefIds 10600-10622
        // These override the sequential array registrations to fix DefId alignment.
        let maybe_either_builtins: &[(usize, &str)] = &[
            (10600, "fromLeft"),
            (10601, "fromRight"),
            (10610, "fromMaybe"),
            (10611, "maybe"),
            (10612, "listToMaybe"),
            (10613, "maybeToList"),
            (10614, "catMaybes"),
            (10615, "mapMaybe"),
            (10616, "either"),
            (10617, "isLeft"),
            (10618, "isRight"),
            (10619, "lefts"),
            (10620, "rights"),
            (10621, "partitionEithers"),
            (10622, "guard"),
        ];

        for &(id, name) in maybe_either_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // Data.List completions at fixed DefIds 10700-10706
        let list_completions_builtins: &[(usize, &str)] = &[
            (10700, "scanr"),
            (10701, "scanl1"),
            (10702, "scanr1"),
            (10703, "unfoldr"),
            (10704, "intersect"),
            (10705, "zip3"),
            (10706, "zipWith3"),
        ];

        for &(id, name) in list_completions_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.16: List operations and Foldable basics at fixed DefIds 10800-10809
        let e16_builtins: &[(usize, &str)] = &[
            (10800, "elemIndex"),
            (10801, "findIndex"),
            (10802, "isPrefixOf"),
            (10803, "isSuffixOf"),
            (10804, "isInfixOf"),
            (10805, "tails"),
            (10806, "inits"),
            (10807, "maximumBy"),
            (10808, "minimumBy"),
            (10809, "foldMap"),
        ];

        for &(id, name) in e16_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.17: Ordering ADT - compare at fixed DefId
        let e17_builtins: &[(usize, &str)] = &[
            (10900, "compare"),
        ];

        for &(id, name) in e17_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.18: Monadic combinators at fixed DefIds
        let e18_builtins: &[(usize, &str)] = &[
            (11000, "filterM"),
            (11001, "foldM"),
            (11002, "foldM_"),
            (11003, "replicateM"),
            (11004, "replicateM_"),
            (11005, "zipWithM"),
            (11006, "zipWithM_"),
        ];

        for &(id, name) in e18_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.19: System.FilePath + System.Directory at fixed DefIds
        let e19_builtins: &[(usize, &str)] = &[
            (11100, "takeFileName"),
            (11101, "takeDirectory"),
            (11102, "takeExtension"),
            (11103, "dropExtension"),
            (11104, "takeBaseName"),
            (11105, "replaceExtension"),
            (11106, "isAbsolute"),
            (11107, "isRelative"),
            (11108, "hasExtension"),
            (11109, "splitExtension"),
            (11110, "setCurrentDirectory"),
            (11111, "removeDirectory"),
            (11112, "renameFile"),
            (11113, "copyFile"),
            (11114, "listDirectory"),
            (11115, "</>"),
        ];

        for &(id, name) in e19_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.20: Data.Text at fixed DefIds (fixes sequential array misalignment)
        let e20_text_builtins: &[(usize, &str)] = &[
            (11200, "Data.Text.empty"),
            (11201, "Data.Text.singleton"),
            (11202, "Data.Text.pack"),
            (11203, "Data.Text.unpack"),
            (11204, "Data.Text.null"),
            (11205, "Data.Text.length"),
            (11206, "Data.Text.head"),
            (11207, "Data.Text.last"),
            (11208, "Data.Text.tail"),
            (11209, "Data.Text.init"),
            (11210, "Data.Text.append"),
            (11211, "Data.Text.<>"),
            (11212, "Data.Text.reverse"),
            (11213, "Data.Text.take"),
            (11214, "Data.Text.takeEnd"),
            (11215, "Data.Text.drop"),
            (11216, "Data.Text.dropEnd"),
            (11217, "Data.Text.isPrefixOf"),
            (11218, "Data.Text.isSuffixOf"),
            (11219, "Data.Text.isInfixOf"),
            (11220, "Data.Text.toLower"),
            (11221, "Data.Text.toUpper"),
            (11222, "Data.Text.toCaseFold"),
            (11223, "Data.Text.toTitle"),
            (11224, "Data.Text.map"),
            (11225, "Data.Text.eq"),
            (11226, "Data.Text.=="),
            (11227, "Data.Text.compare"),
            (11228, "Data.Text.filter"),
            (11229, "Data.Text.foldl'"),
            (11230, "Data.Text.concat"),
            (11231, "Data.Text.intercalate"),
            (11232, "Data.Text.strip"),
            (11233, "Data.Text.words"),
            (11234, "Data.Text.lines"),
            (11235, "Data.Text.splitOn"),
            (11236, "Data.Text.replace"),
            // Data.Text.Encoding
            (11238, "Data.Text.Encoding.encodeUtf8"),
            (11239, "Data.Text.Encoding.decodeUtf8"),
        ];

        for &(id, name) in e20_text_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.20: Data.ByteString at fixed DefIds (fixes sequential array misalignment)
        let e20_bs_builtins: &[(usize, &str)] = &[
            (11250, "Data.ByteString.empty"),
            (11251, "Data.ByteString.singleton"),
            (11252, "Data.ByteString.pack"),
            (11253, "Data.ByteString.unpack"),
            (11254, "Data.ByteString.null"),
            (11255, "Data.ByteString.length"),
            (11256, "Data.ByteString.head"),
            (11257, "Data.ByteString.last"),
            (11258, "Data.ByteString.tail"),
            (11259, "Data.ByteString.init"),
            (11260, "Data.ByteString.append"),
            (11261, "Data.ByteString.cons"),
            (11262, "Data.ByteString.snoc"),
            (11263, "Data.ByteString.take"),
            (11264, "Data.ByteString.drop"),
            (11265, "Data.ByteString.reverse"),
            (11266, "Data.ByteString.elem"),
            (11267, "Data.ByteString.index"),
            (11268, "Data.ByteString.eq"),
            (11269, "Data.ByteString.compare"),
            (11270, "Data.ByteString.isPrefixOf"),
            (11271, "Data.ByteString.isSuffixOf"),
            (11272, "Data.ByteString.readFile"),
            (11273, "Data.ByteString.writeFile"),
        ];

        for &(id, name) in e20_bs_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.25: String type class methods at fixed DefIds
        let e25_string_builtins: &[(usize, &str)] = &[
            (11300, "fromString"),
            (11301, "read"),
            (11302, "readMaybe"),
        ];

        for &(id, name) in e25_string_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.26: List *By variants, sortOn, stripPrefix, insert, mapAccumL/R
        let e26_list_builtins: &[(usize, &str)] = &[
            (11400, "sortOn"),
            (11401, "nubBy"),
            (11402, "groupBy"),
            (11403, "deleteBy"),
            (11404, "unionBy"),
            (11405, "intersectBy"),
            (11406, "stripPrefix"),
            (11407, "insert"),
            (11408, "mapAccumL"),
            (11409, "mapAccumR"),
        ];

        for &(id, name) in e26_list_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.27: Data.Function / Data.Tuple builtins
        let e27_builtins: &[(usize, &str)] = &[
            (11500, "succ"),
            (11501, "pred"),
            (11502, "&"),
            (11503, "swap"),
            (11504, "curry"),
            (11505, "uncurry"),
        ];

        for &(id, name) in e27_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.28: Arithmetic, enum, folds, higher-order, IO input
        let e28_builtins: &[(usize, &str)] = &[
            (11600, "min"),
            (11601, "max"),
            (11602, "subtract"),
            (11603, "enumFrom"),
            (11604, "enumFromThen"),
            (11605, "enumFromThenTo"),
            (11606, "foldl1"),
            (11607, "foldr1"),
            (11608, "comparing"),
            (11609, "until"),
            (11610, "getChar"),
            (11611, "isEOF"),
            (11612, "getContents"),
            (11613, "interact"),
        ];

        for &(id, name) in e28_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.29: Map.mapMaybe
        let e29_builtins: &[(usize, &str)] = &[
            (11700, "Data.Map.mapMaybe"),
            (11701, "Data.Map.mapMaybeWithKey"),
        ];
        for &(id, name) in e29_builtins {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.43: Word types at fixed DefIds 11800-11809
        // Types are registered at fixed DefIds to avoid sequential array misalignment.
        let word_types: &[(usize, &str)] = &[
            (11800, "Word"),
            (11801, "Word8"),
            (11802, "Word16"),
            (11803, "Word32"),
            (11804, "Word64"),
        ];
        for &(id, name) in word_types {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Type, Span::default());
            self.bind_type(sym, def_id);
        }
        // Qualified names for Data.Word module
        let word_qualified: &[(usize, &str)] = &[
            (11805, "Data.Word.Word"),
            (11806, "Data.Word.Word8"),
            (11807, "Data.Word.Word16"),
            (11808, "Data.Word.Word32"),
            (11809, "Data.Word.Word64"),
        ];
        for &(id, name) in word_qualified {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // E.45: Integer type (arbitrary precision)
        {
            let def_id = DefId::new(11900);
            let sym = Symbol::intern("Integer");
            self.define(def_id, sym, DefKind::Type, Span::default());
            self.bind_type(sym, def_id);
        }

        // E.45: Integer RTS functions (fixed DefIds 11901-11920)
        let integer_fns: &[(usize, &str)] = &[
            (11901, "bhc_integer_from_i64"),
            (11902, "bhc_integer_from_str"),
            (11903, "bhc_integer_to_i64"),
            (11904, "bhc_integer_add"),
            (11905, "bhc_integer_sub"),
            (11906, "bhc_integer_mul"),
            (11907, "bhc_integer_div"),
            (11908, "bhc_integer_mod"),
            (11909, "bhc_integer_negate"),
            (11910, "bhc_integer_abs"),
            (11911, "bhc_integer_signum"),
            (11912, "bhc_integer_gcd"),
            (11913, "bhc_integer_eq"),
            (11914, "bhc_integer_lt"),
            (11915, "bhc_integer_le"),
            (11916, "bhc_integer_gt"),
            (11917, "bhc_integer_ge"),
            (11918, "bhc_integer_compare"),
            (11919, "bhc_integer_show"),
        ];
        for &(id, name) in integer_fns {
            let sym = Symbol::intern(name);
            let def_id = DefId::new(id);
            self.define(def_id, sym, DefKind::Value, Span::default());
            self.bind_value(sym, def_id);
        }

        // Ensure next_def_id is past the fixed DefId ranges
        if self.next_def_id <= 11920 {
            self.next_def_id = 11920;
        }
    }

    /// Define stub types, constructors, and functions for external packages.
    ///
    /// These are placeholders for X11, System.Posix, and other external dependencies
    /// that allow the lowering phase to succeed. They are marked with `DefKind::Stub*`
    /// variants so they can be tracked and eventually warned about when resolved.
    fn define_stubs(&mut self) {
        // Stub types from external packages
        let stub_types = [
            // Graphics.X11
            "Event",
            "Rectangle",
            "WindowChanges",
            // System.Posix
            "OpenMode",
            "Handler",
            // X11.Xlib.Extras
            "Full",
            // System.Locale
            "LocaleCategory",
        ];

        for name in stub_types {
            let sym = Symbol::intern(name);
            let def_id = self.fresh_def_id();
            self.define(def_id, sym, DefKind::StubType, Span::default());
            self.bind_type(sym, def_id);
        }

        // Stub constructors from external packages with arities
        // Format: (name, type, arity)
        let stub_cons: &[(&str, &str, usize)] = &[
            // Graphics.X11 - Event types (records with many fields, but pattern matched with {})
            ("KeyEvent", "Event", 0), // Actually has fields, but we use 0 for record-style
            ("ButtonEvent", "Event", 0),
            ("MotionEvent", "Event", 0),
            ("CrossingEvent", "Event", 0),
            ("ConfigureEvent", "Event", 0),
            ("ConfigureRequestEvent", "Event", 0),
            ("MapRequestEvent", "Event", 0),
            ("UnmapEvent", "Event", 0),
            ("DestroyWindowEvent", "Event", 0),
            ("PropertyEvent", "Event", 0),
            ("ClientMessageEvent", "Event", 0),
            ("MappingNotifyEvent", "Event", 0),
            // Graphics.X11 - Types
            ("Rectangle", "Rectangle", 4), // Rectangle x y width height
            ("WindowChanges", "WindowChanges", 0), // Record type
            // System.Posix
            ("ReadOnly", "OpenMode", 0),
            ("WriteOnly", "OpenMode", 0),
            ("ReadWrite", "OpenMode", 0),
            ("Default", "Handler", 0),
            ("Ignore", "Handler", 0),
            ("Catch", "Handler", 1), // Catch (Signal -> IO ())
            // X11.Xlib.Extras (LayoutClass)
            ("Full", "Full", 0),
            // System.Locale
            ("LC_ALL", "LocaleCategory", 0),
            ("LC_CTYPE", "LocaleCategory", 0),
        ];

        for (con_name, _type_name, arity) in stub_cons {
            let sym = Symbol::intern(con_name);
            let def_id = self.fresh_def_id();
            self.define_constructor(def_id, sym, Span::default(), *arity);
            self.bind_constructor(sym, def_id);
        }

        // Stub functions from external packages
        let stub_funcs = [
            // Graphics.X11 - Basic
            "openDisplay",
            "rootWindow",
            "defaultScreen",
            "selectInput",
            "sync",
            "internAtom",
            "allocaXEvent",
            "nextEvent",
            "getEvent",
            "sendEvent",
            "queryTree",
            "queryPointer",
            "mapWindow",
            "unmapWindow",
            "moveWindow",
            "resizeWindow",
            "moveResizeWindow",
            "configureWindow",
            "setInputFocus",
            "grabButton",
            "ungrabButton",
            "grabPointer",
            "ungrabPointer",
            "ungrabKeyboard",
            "warpPointer",
            "createWindow",
            "killClient",
            "restackWindows",
            "refreshKeyboardMapping",
            "keycodeToKeysym",
            "keysymToKeycodes",
            "getModifierMapping",
            "getWindowAttributes",
            "getWindowProperty8",
            "getWindowProperty32",
            "getTextProperty",
            "getClassHint",
            "getWMNormalHints",
            "getWMHints",
            "getWMProtocols",
            "getTransientForHint",
            "setWindowBorder",
            "setWindowBorderWidth",
            "changeProperty32",
            "allocNamedColor",
            "createFontCursor",
            "allowEvents",
            "checkMaskEvent",
            "setEventType",
            "setClientMessageEvent",
            "setClientMessageEvent'",
            "setConfigureEvent",
            "allocaSetWindowAttributes",
            "set_override_redirect",
            "set_event_mask",
            "xSetSelectionOwner",
            "xGetSelectionOwner",
            "xSetErrorHandler",
            "windowEvent",
            "xFree",
            "getScreenInfo",
            "xrrQueryExtension",
            "xrrUpdateConfiguration",
            "defaultColormap",
            "defaultScreenOfDisplay",
            "defaultVisualOfScreen",
            "copyFromParent",
            // Graphics.X11 - Constants
            "shiftMask",
            "mod1Mask",
            "mod2Mask",
            "mod3Mask",
            "mod4Mask",
            "mod5Mask",
            "lockMask",
            "controlMask",
            "anyModifier",
            "anyKey",
            "anyButton",
            "button1",
            "button2",
            "button3",
            "button4",
            "button5",
            "currentTime",
            "none",
            "grabModeAsync",
            "grabModeSync",
            "replayPointer",
            "revertToPointerRoot",
            "noEventMask",
            "keyPress",
            "keyRelease",
            "buttonPress",
            "buttonRelease",
            "enterNotify",
            "leaveNotify",
            "motionNotify",
            "propertyNotify",
            "configureNotify",
            "clientMessage",
            "destroyNotify",
            "mappingKeyboard",
            "mappingModifier",
            "notifyNormal",
            "enterWindowMask",
            "leaveWindowMask",
            "buttonPressMask",
            "buttonReleaseMask",
            "pointerMotionMask",
            "structureNotifyMask",
            "propertyChangeMask",
            "substructureNotifyMask",
            "substructureRedirectMask",
            "cWOverrideRedirect",
            "cWEventMask",
            "propModeReplace",
            "normalState",
            "iconicState",
            "withdrawnState",
            "waIsViewable",
            "inputHintBit",
            "selectionRequest",
            "xC_fleur",
            "xC_bottom_right_corner",
            "noSymbol",
            // Graphics.X11 - Keysyms
            "xK_Return",
            "xK_Tab",
            "xK_space",
            "xK_Escape",
            "xK_BackSpace",
            "xK_Delete",
            "xK_Home",
            "xK_End",
            "xK_Page_Up",
            "xK_Page_Down",
            "xK_Left",
            "xK_Right",
            "xK_Up",
            "xK_Down",
            "xK_Num_Lock",
            "xK_0",
            "xK_1",
            "xK_2",
            "xK_3",
            "xK_4",
            "xK_5",
            "xK_6",
            "xK_7",
            "xK_8",
            "xK_9",
            "xK_a",
            "xK_b",
            "xK_c",
            "xK_d",
            "xK_e",
            "xK_f",
            "xK_g",
            "xK_h",
            "xK_i",
            "xK_j",
            "xK_k",
            "xK_l",
            "xK_m",
            "xK_n",
            "xK_o",
            "xK_p",
            "xK_q",
            "xK_r",
            "xK_s",
            "xK_t",
            "xK_u",
            "xK_v",
            "xK_w",
            "xK_x",
            "xK_y",
            "xK_z",
            "xK_comma",
            "xK_period",
            "xK_slash",
            "xK_question",
            // Graphics.X11 - Event field accessors
            "ev_event_type",
            "ev_window",
            "ev_x_root",
            "ev_y_root",
            "ev_state",
            "ev_time",
            "ev_subwindow",
            "ev_mode",
            "ev_same_screen",
            "ev_request",
            "ev_value_mask",
            "ev_border_width",
            "ev_x",
            "ev_y",
            "ev_width",
            "ev_height",
            "ev_above",
            "ev_detail",
            // Graphics.X11 - Window attributes accessors
            "wa_x",
            "wa_y",
            "wa_width",
            "wa_height",
            "wa_border_width",
            "wa_override_redirect",
            "wa_map_state",
            "wa_colormap",
            // Graphics.X11 - Size hints accessors
            "sh_min_size",
            "sh_max_size",
            "sh_base_size",
            "sh_resize_inc",
            "sh_aspect",
            // Graphics.X11 - WM hints accessors
            "wmh_flags",
            "wmh_input",
            // Graphics.X11 - Class hint accessors
            "resName",
            "resClass",
            // Graphics.X11 - Text property accessors
            "tp_value",
            "extract",
            // Graphics.X11 - Color accessors
            "color_pixel",
            // Graphics.X11 - Rectangle accessors
            "rect_x",
            "rect_y",
            "rect_width",
            "rect_height",
            // Graphics.X11 - Other
            "get_EventType",
            "wM_NAME",
            "grab",
            "ungrabKey",
            "minCode",
            "maxCode",
            // System.Posix.Process
            "executeFile",
            "forkProcess",
            "createSession",
            "getAnyProcessStatus",
            // System.Posix.IO
            "openFd",
            "closeFd",
            "dupTo",
            "stdInput",
            "stdOutput",
            "stdError",
            "defaultFileFlags",
            // System.Posix.Signals
            "installHandler",
            "sigCHLD",
            "openEndedPipe",
            // System.Locale
            "setLocale",
            // System.IO (stubs) - hSetBuffering and hFlush moved to builtin_funcs
            "flush",
            // Data.Map/Set qualified functions (stubs)
            "M.fromListWith",
            "M.member",
            "M.notMember",
            "member",
            "notMember",
            "fromListWith",
            // Control.Exception qualified
            "E.catch",
            // Misc XMonad internal stubs
            "buildScript",
            "stackYaml",
            "flakeNix",
            "defaultNix",
            "compiledWithXinerama",
            "launch'",
            "Default.def",
            "def", // Data.Default
            "f",   // generic variable
            "width",
            "height",
            "least",
            "subtract",
            "maybeShow",
        ];

        for name in stub_funcs {
            let sym = Symbol::intern(name);
            let def_id = self.fresh_def_id();
            self.define(def_id, sym, DefKind::StubValue, Span::default());
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
                arity: None,
                type_con_name: None,
                type_param_count: None,
                field_names: None,
            },
        );
    }

    /// Records a constructor definition with its arity.
    pub fn define_constructor(&mut self, id: DefId, name: Symbol, span: Span, arity: usize) {
        self.defs.insert(
            id,
            DefInfo {
                id,
                name,
                kind: DefKind::Constructor,
                span,
                arity: Some(arity),
                type_con_name: None,
                type_param_count: None,
                field_names: None,
            },
        );
    }

    /// Records a constructor definition with full type information.
    pub fn define_constructor_with_type(
        &mut self,
        id: DefId,
        name: Symbol,
        span: Span,
        arity: usize,
        type_con_name: Symbol,
        type_param_count: usize,
        field_names: Option<Vec<Symbol>>,
    ) {
        self.defs.insert(
            id,
            DefInfo {
                id,
                name,
                kind: DefKind::Constructor,
                span,
                arity: Some(arity),
                type_con_name: Some(type_con_name),
                type_param_count: Some(type_param_count),
                field_names,
            },
        );
    }

    /// Creates a `DefRef` for a definition.
    pub fn def_ref(&self, def_id: DefId, span: Span) -> DefRef {
        DefRef { def_id, span }
    }

    /// Checks if a definition is a stub (external package placeholder).
    pub fn is_stub(&self, def_id: DefId) -> bool {
        self.defs
            .get(&def_id)
            .map(|info| info.kind.is_stub())
            .unwrap_or(false)
    }

    /// Gets the DefKind for a definition.
    pub fn def_kind(&self, def_id: DefId) -> Option<DefKind> {
        self.defs.get(&def_id).map(|info| info.kind)
    }

    /// Gets the field names for a record constructor, if any.
    pub fn get_constructor_field_names(&self, def_id: DefId) -> Option<Vec<Symbol>> {
        self.defs
            .get(&def_id)
            .and_then(|info| info.field_names.clone())
    }

    /// Emits a warning if the given definition is a stub.
    /// Returns true if a warning was emitted.
    pub fn warn_if_stub(&mut self, def_id: DefId, name: &str, span: Span) -> bool {
        if let Some(info) = self.defs.get(&def_id) {
            if info.kind.is_stub() {
                let kind = match info.kind {
                    DefKind::StubValue => "function",
                    DefKind::StubType => "type",
                    DefKind::StubConstructor => "constructor",
                    _ => return false,
                };
                self.warnings.push(crate::LowerWarning::StubUsed {
                    name: name.to_string(),
                    span,
                    kind,
                });
                return true;
            }
        }
        false
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
        let module = self
            .import_aliases
            .get(&qualifier)
            .copied()
            .unwrap_or(qualifier);

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
        let module = self
            .import_aliases
            .get(&qualifier)
            .copied()
            .unwrap_or(qualifier);

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

    /// Registers a type signature for a function.
    ///
    /// Called during the first pass of module lowering to collect all
    /// type signatures before lowering function definitions.
    pub fn register_type_signature(&mut self, name: Symbol, ty: ast::Type) {
        self.type_signatures.insert(name, ty);
    }

    /// Looks up a type signature for a function.
    ///
    /// Returns the AST type if a signature was declared, or None otherwise.
    pub fn lookup_type_signature(&self, name: Symbol) -> Option<&ast::Type> {
        self.type_signatures.get(&name)
    }

    /// Registers the number of type parameters for a class.
    ///
    /// Used to decompose multi-param instance types during lowering.
    pub fn register_class_param_count(&mut self, class_name: Symbol, count: usize) {
        self.class_param_counts.insert(class_name, count);
    }

    /// Looks up the number of type parameters for a class.
    pub fn lookup_class_param_count(&self, class_name: Symbol) -> Option<usize> {
        self.class_param_counts.get(&class_name).copied()
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
