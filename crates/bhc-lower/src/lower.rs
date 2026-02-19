//! Main AST to HIR lowering pass.
//!
//! This module contains the core lowering logic that transforms surface AST
//! into HIR by:
//!
//! 1. Resolving names (identifiers -> DefIds)
//! 2. Desugaring syntactic constructs
//! 3. Building HIR nodes

use bhc_ast as ast;
use bhc_hir as hir;
use bhc_intern::Symbol;
use bhc_span::Span;
use camino::Utf8PathBuf;

use crate::context::{DefKind, LowerContext};
use crate::desugar;
use crate::loader::{self, LoadError, ModuleCache};
use crate::resolve::{bind_pattern, collect_module_definitions, resolve_constructor, resolve_var};
use crate::{LowerError, LowerResult};

/// Configuration for the lowering pass.
#[derive(Clone, Debug, Default)]
pub struct LowerConfig {
    /// Whether to include builtins in the context.
    pub include_builtins: bool,
    /// Whether to report warnings for unused bindings.
    pub warn_unused: bool,
    /// Search paths for module imports.
    pub search_paths: Vec<Utf8PathBuf>,
}

/// Lower an AST module to HIR.
///
/// # Arguments
///
/// * `ctx` - The lowering context containing scope and definition information
/// * `module` - The AST module to lower
/// * `config` - Configuration including search paths for module imports
pub fn lower_module(
    ctx: &mut LowerContext,
    module: &ast::Module,
    config: &LowerConfig,
) -> LowerResult<hir::Module> {
    lower_module_with_cache(ctx, module, config, ModuleCache::new())
}

/// Lower an AST module to HIR with a pre-seeded module cache.
///
/// This is used during multi-module compilation: modules compiled earlier
/// have their exports inserted into the cache so that later modules can
/// resolve imports without loading from disk.
pub fn lower_module_with_cache(
    ctx: &mut LowerContext,
    module: &ast::Module,
    config: &LowerConfig,
    cache: ModuleCache,
) -> LowerResult<hir::Module> {
    let mut cache = cache;

    // Determine if we should inject an implicit Prelude import.
    // Skip if:
    // 1. The module is the Prelude itself
    // 2. The module has {-# LANGUAGE NoImplicitPrelude #-}
    // 3. The module already explicitly imports Prelude
    let module_name_str = module.name.as_ref().map(|n| n.to_string());
    let is_prelude_module = module_name_str.as_deref() == Some("Prelude")
        || module_name_str.as_deref() == Some("BHC.Prelude");

    let has_no_implicit_prelude = module.pragmas.iter().any(|p| {
        if let ast::PragmaKind::Language(exts) = &p.kind {
            exts.iter().any(|e| e.as_str() == "NoImplicitPrelude")
        } else {
            false
        }
    });

    let has_overloaded_strings = module.pragmas.iter().any(|p| {
        if let ast::PragmaKind::Language(exts) = &p.kind {
            exts.iter().any(|e| e.as_str() == "OverloadedStrings")
        } else {
            false
        }
    });

    let has_scoped_type_variables = module.pragmas.iter().any(|p| {
        if let ast::PragmaKind::Language(exts) = &p.kind {
            exts.iter().any(|e| e.as_str() == "ScopedTypeVariables")
        } else {
            false
        }
    });

    let has_generalized_newtype_deriving = module.pragmas.iter().any(|p| {
        if let ast::PragmaKind::Language(exts) = &p.kind {
            exts.iter()
                .any(|e| e.as_str() == "GeneralizedNewtypeDeriving")
        } else {
            false
        }
    });

    let has_flexible_instances = module.pragmas.iter().any(|p| {
        if let ast::PragmaKind::Language(exts) = &p.kind {
            exts.iter().any(|e| e.as_str() == "FlexibleInstances")
        } else {
            false
        }
    });

    let has_flexible_contexts = module.pragmas.iter().any(|p| {
        if let ast::PragmaKind::Language(exts) = &p.kind {
            exts.iter().any(|e| e.as_str() == "FlexibleContexts")
        } else {
            false
        }
    });

    let has_gadts = module.pragmas.iter().any(|p| {
        if let ast::PragmaKind::Language(exts) = &p.kind {
            exts.iter().any(|e| e.as_str() == "GADTs")
        } else {
            false
        }
    });

    let already_imports_prelude = module.imports.iter().any(|imp| {
        let name = imp
            .module
            .parts
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(".");
        name == "Prelude"
    });

    let should_inject_prelude =
        !is_prelude_module && !has_no_implicit_prelude && !already_imports_prelude;

    // Build the effective imports list, possibly prepending an implicit Prelude import
    let mut effective_imports = Vec::new();
    if should_inject_prelude {
        effective_imports.push(ast::ImportDecl {
            module: ast::ModuleName {
                parts: vec![Symbol::intern("Prelude")],
                span: Span::default(),
            },
            qualified: false,
            alias: None,
            spec: None,
            span: Span::default(),
        });
    }
    effective_imports.extend(module.imports.iter().cloned());

    // Process imports first to register aliases
    process_imports(ctx, &effective_imports, &config.search_paths, &mut cache);

    // First pass: collect all top-level definitions
    collect_module_definitions(ctx, module);

    // Second pass: lower all declarations
    let mut items = Vec::new();
    for decl in &module.decls {
        let new_items = lower_decl(ctx, decl)?;
        items.extend(new_items);
    }

    // Lower imports
    let imports = module.imports.iter().map(|imp| lower_import(imp)).collect();

    // Lower exports
    let exports = module
        .exports
        .as_ref()
        .map(|exps| exps.iter().map(|exp| lower_export(exp)).collect());

    // Check for errors
    if ctx.has_errors() {
        let errors = ctx.take_errors();
        return Err(LowerError::Multiple(errors));
    }

    Ok(hir::Module {
        name: module.name.as_ref().map_or_else(
            || Symbol::intern("Main"),
            |n| {
                // Combine module name parts into a single symbol
                let full_name = n
                    .parts
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                Symbol::intern(&full_name)
            },
        ),
        exports,
        imports,
        items,
        span: module.span,
        overloaded_strings: has_overloaded_strings,
        scoped_type_variables: has_scoped_type_variables,
        generalized_newtype_deriving: has_generalized_newtype_deriving,
        flexible_instances: has_flexible_instances,
        flexible_contexts: has_flexible_contexts,
        gadts: has_gadts,
    })
}

/// Process imports and register aliases.
///
/// This function attempts to load modules from the file system first. If a module
/// is not found in the search paths, it falls back to the hardcoded standard
/// library exports.
fn process_imports(
    ctx: &mut LowerContext,
    imports: &[ast::ImportDecl],
    search_paths: &[Utf8PathBuf],
    cache: &mut ModuleCache,
) {
    for import in imports {
        // Get the full module name
        let module_name = import
            .module
            .parts
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(".");
        let module_sym = Symbol::intern(&module_name);

        // Determine the alias to use
        let alias = if let Some(alias_name) = &import.alias {
            // Explicit alias: import qualified Data.Map as M
            let alias_str = alias_name
                .parts
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(".");
            Symbol::intern(&alias_str)
        } else if import.qualified {
            // No explicit alias for qualified import: use full module name
            // import qualified Data.Map means M.lookup is Data.Map.lookup
            module_sym
        } else {
            // Non-qualified import: no alias needed for qualified access
            // but we might still want to track it
            module_sym
        };

        // Register the alias
        ctx.register_import_alias(alias, module_sym);

        // For non-qualified imports, also register the last component as an alias
        // This handles `import Data.Map` allowing both `Data.Map.lookup` and `Map.lookup`
        if !import.qualified && import.module.parts.len() > 1 {
            if let Some(last_part) = import.module.parts.last() {
                let short_alias = Symbol::intern(last_part.as_str());
                ctx.register_import_alias(short_alias, module_sym);
            }
        }

        // Try to load the module from the file system
        match loader::load_module(&module_name, search_paths, cache, ctx) {
            Ok(exports) => {
                // Apply import specification (Only/Hiding)
                let filtered = loader::apply_import_spec(&exports, &import.spec);
                // Register the imported names
                loader::register_imported_names(ctx, import, &filtered);
                tracing::debug!(
                    module = %module_name,
                    values = filtered.values.len(),
                    types = filtered.types.len(),
                    constructors = filtered.constructors.len(),
                    "loaded module exports"
                );
            }
            Err(LoadError::ModuleNotFound(_)) => {
                // Module not found in search paths - fall back to standard library
                tracing::debug!(
                    module = %module_name,
                    "module not found in search paths, using standard library fallback"
                );
                register_standard_module_exports(ctx, &module_name);
            }
            Err(LoadError::CircularImport(ref cycle_module)) => {
                // Log circular import but don't fail - allow compilation to continue
                tracing::warn!(
                    module = %module_name,
                    cycle = %cycle_module,
                    "circular import detected"
                );
                // Still register standard exports as fallback
                register_standard_module_exports(ctx, &module_name);
            }
            Err(e) => {
                // Other errors (IO, parse) - log and fall back
                tracing::warn!(
                    module = %module_name,
                    error = %e,
                    "failed to load module, using standard library fallback"
                );
                register_standard_module_exports(ctx, &module_name);
            }
        }
    }
}

/// Register standard module exports.
///
/// This is a temporary solution until we have proper module loading.
/// It registers common functions from well-known modules.
fn register_standard_module_exports(ctx: &mut LowerContext, module_name: &str) {
    let exports: &[&str] = match module_name {
        "Prelude" => &[
            // Numeric
            "+",
            "-",
            "*",
            "negate",
            "abs",
            "signum",
            "fromInteger",
            "fromIntegral",
            "div",
            "mod",
            "even",
            "odd",
            "sum",
            "product",
            // Comparison
            "==",
            "/=",
            "<",
            "<=",
            ">",
            ">=",
            "compare",
            "min",
            "max",
            // Boolean
            "&&",
            "||",
            "not",
            "otherwise",
            // List operations
            "map",
            "filter",
            "head",
            "tail",
            "last",
            "init",
            "null",
            "length",
            "reverse",
            "foldl",
            "foldl'",
            "foldr",
            "concat",
            "concatMap",
            "zip",
            "zipWith",
            "unzip",
            "take",
            "drop",
            "splitAt",
            "takeWhile",
            "dropWhile",
            "span",
            "break",
            "elem",
            "notElem",
            "lookup",
            "replicate",
            "iterate",
            "repeat",
            "cycle",
            "any",
            "all",
            "and",
            "or",
            "lines",
            "unlines",
            "words",
            "unwords",
            "++",
            // Tuple
            "fst",
            "snd",
            "curry",
            "uncurry",
            "swap",
            // Function
            "id",
            "const",
            "flip",
            "$",
            ".",
            "seq",
            "error",
            "undefined",
            // Maybe
            "maybe",
            "fromMaybe",
            "isJust",
            "isNothing",
            // Either
            "either",
            // IO
            "putStrLn",
            "putStr",
            "print",
            "getLine",
            ">>",
            ">>=",
            "return",
            // Show
            "show",
            "showString",
            "showChar",
            "showParen",
            // Enum
            "enumFromTo",
            "succ",
            "pred",
            "toEnum",
            "fromEnum",
            // Integral
            "gcd",
            "lcm",
            "quot",
            "rem",
            "quotRem",
            "divMod",
            "subtract",
            "realToFrac",
            // List extras
            "maximum",
            "minimum",
            "scanl",
            "scanr",
            "scanl1",
            "scanr1",
            "zip3",
            "zipWith3",
            "unzip3",
            // Function
            "until",
            "asTypeOf",
            // IO
            "getChar",
            "getContents",
            "readFile",
            "writeFile",
            "appendFile",
            "interact",
        ],
        "Data.Map" | "Data.Map.Strict" => &[
            "empty",
            "singleton",
            "insert",
            "insertWith",
            "delete",
            "lookup",
            "member",
            "notMember",
            "null",
            "size",
            "union",
            "unionWith",
            "unionWithKey",
            "unions",
            "intersection",
            "intersectionWith",
            "difference",
            "differenceWith",
            "map",
            "mapWithKey",
            "mapKeys",
            "filter",
            "filterWithKey",
            "foldr",
            "foldl",
            "foldrWithKey",
            "foldlWithKey",
            "keys",
            "elems",
            "assocs",
            "toList",
            "fromList",
            "fromListWith",
            "toAscList",
            "toDescList",
            "findWithDefault",
            "adjust",
            "update",
            "alter",
            "keysSet",
            "isSubmapOf",
            "!",
        ],
        "Data.Set" => &[
            "empty",
            "singleton",
            "insert",
            "delete",
            "member",
            "notMember",
            "null",
            "size",
            "union",
            "unions",
            "intersection",
            "difference",
            "isSubsetOf",
            "isProperSubsetOf",
            "map",
            "filter",
            "partition",
            "foldr",
            "foldl",
            "toList",
            "fromList",
            "toAscList",
            "toDescList",
            "elems",
            "findMin",
            "findMax",
            "lookupMin",
            "lookupMax",
            "deleteMin",
            "deleteMax",
        ],
        "Data.IntMap" | "Data.IntMap.Strict" => &[
            "empty",
            "singleton",
            "insert",
            "insertWith",
            "delete",
            "lookup",
            "member",
            "null",
            "size",
            "union",
            "unionWith",
            "intersection",
            "difference",
            "map",
            "mapWithKey",
            "filter",
            "foldr",
            "foldlWithKey",
            "keys",
            "elems",
            "toList",
            "toAscList",
            "fromList",
            "findWithDefault",
            "adjust",
        ],
        "Data.IntSet" => &[
            "empty",
            "singleton",
            "insert",
            "delete",
            "member",
            "null",
            "size",
            "union",
            "intersection",
            "difference",
            "isSubsetOf",
            "filter",
            "foldr",
            "toList",
            "fromList",
        ],
        "Data.List" => &[
            "sort",
            "sortBy",
            "sortOn",
            "nub",
            "nubBy",
            "delete",
            "deleteBy",
            "union",
            "unionBy",
            "intersect",
            "intersectBy",
            "group",
            "groupBy",
            "intersperse",
            "intercalate",
            "transpose",
            "subsequences",
            "permutations",
            "foldl'",
            "find",
            "partition",
            "span",
            "break",
            "stripPrefix",
            "isPrefixOf",
            "isSuffixOf",
            "isInfixOf",
            "tails",
            "inits",
            "mapAccumL",
            "mapAccumR",
            "unfoldr",
            "genericLength",
            "genericTake",
            "genericDrop",
            "\\\\",
        ],
        "Data.Maybe" => &[
            "maybe",
            "isJust",
            "isNothing",
            "fromJust",
            "fromMaybe",
            "listToMaybe",
            "maybeToList",
            "catMaybes",
            "mapMaybe",
        ],
        "Data.Either" => &[
            "either",
            "isLeft",
            "isRight",
            "fromLeft",
            "fromRight",
            "lefts",
            "rights",
            "partitionEithers",
        ],
        "Data.List.NonEmpty" | "Data.Semigroup" => &[
            "head", "tail", "last", "init", "toList", "nonEmpty", "fromList", "(<>)", "sconcat",
            "stimes",
        ],
        "Control.Monad" => &[
            "when",
            "unless",
            "guard",
            "void",
            "join",
            "filterM",
            "mapM",
            "forM",
            "sequence",
            "replicateM",
            "replicateM_",
            "forever",
            "liftM",
            "liftM2",
            "liftM3",
            "liftM4",
            "liftM5",
            "ap",
            "mzero",
            "mplus",
            "msum",
            "mfilter",
            "mapAndUnzipM",
            "zipWithM",
            "zipWithM_",
            "foldM",
            "foldM_",
            ">=>",
            "<=<",
        ],
        "Control.Applicative" => &[
            "pure", "(<*>)", "(<$>)", "(*>)", "(<*)", "empty", "(<|>)", "some", "many", "optional",
            "liftA", "liftA2", "liftA3",
        ],
        "Control.Exception" => &[
            "Exception",
            "SomeException",
            "IOException",
            "ErrorCall",
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
            "throwTo",
        ],
        "Control.Concurrent" => &[
            "forkIO",
            "killThread",
            "threadDelay",
            "myThreadId",
            "throwTo",
        ],
        "Control.Concurrent.MVar" => &[
            "MVar",
            "newMVar",
            "newEmptyMVar",
            "takeMVar",
            "putMVar",
            "readMVar",
            "modifyMVar",
            "modifyMVar_",
            "withMVar",
        ],
        "Data.Foldable" => &[
            "fold",
            "foldMap",
            "foldMap'",
            "foldr",
            "foldr'",
            "foldl",
            "foldl'",
            "foldr1",
            "foldl1",
            "toList",
            "null",
            "length",
            "elem",
            "notElem",
            "maximum",
            "minimum",
            "maximumBy",
            "minimumBy",
            "sum",
            "product",
            "any",
            "all",
            "and",
            "or",
            "find",
            "concat",
            "concatMap",
            "asum",
            "msum",
            "traverse_",
            "for_",
            "sequenceA_",
            "mapM_",
            "forM_",
        ],
        "Data.Traversable" => &[
            "traverse",
            "sequenceA",
            "mapM",
            "sequence",
            "for",
            "forM",
            "mapAccumL",
            "mapAccumR",
        ],
        "Data.Monoid" => &[
            "mempty", "mappend", "mconcat", "(<>)", "Sum", "Product", "Any", "All", "First",
            "Last", "Endo", "Dual",
        ],
        "Data.Bits" => &[
            "(.&.)",
            "(.|.)",
            "xor",
            "complement",
            "shift",
            "rotate",
            "bit",
            "setBit",
            "clearBit",
            "complementBit",
            "testBit",
            "shiftL",
            "shiftR",
            "rotateL",
            "rotateR",
            "popCount",
            "zeroBits",
            "countLeadingZeros",
            "countTrailingZeros",
        ],
        "Data.Char" => &[
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
            "toTitle",
            "digitToInt",
            "intToDigit",
            "isLatin1",
            "isAsciiLower",
            "isAsciiUpper",
        ],
        "Data.Function" => &["id", "const", "flip", "($)", "(&)", "on", "fix"],
        "Data.Tuple" => &["fst", "snd", "curry", "uncurry", "swap"],
        "Data.Ord" => &[
            "compare",
            "(<)",
            "(<=)",
            "(>)",
            "(>=)",
            "max",
            "min",
            "comparing",
            "clamp",
            "Down",
        ],
        "Data.Eq" => &["(==)", "(/=)"],
        "Text.Read" => &[
            "read",
            "reads",
            "readMaybe",
            "readEither",
            "readPrec",
            "lex",
        ],
        "Text.Show" => &["show", "shows", "showString", "showChar", "showParen"],
        "System.IO" => &[
            "IO",
            "FilePath",
            "Handle",
            "IOMode",
            "stdin",
            "stdout",
            "stderr",
            "openFile",
            "hClose",
            "hGetChar",
            "hGetLine",
            "hGetContents",
            "hPutChar",
            "hPutStr",
            "hPutStrLn",
            "hPrint",
            "hFlush",
            "hIsEOF",
            "hSetBuffering",
            "hGetBuffering",
            "hSeek",
            "hTell",
            "hFileSize",
            "withFile",
            "readFile",
            "writeFile",
            "appendFile",
            "getLine",
            "getContents",
            "putStr",
            "putStrLn",
            "print",
        ],
        "Data.IORef" => &[
            "IORef",
            "newIORef",
            "readIORef",
            "writeIORef",
            "modifyIORef",
            "modifyIORef'",
            "atomicModifyIORef",
            "atomicModifyIORef'",
        ],
        "System.Exit" => &[
            "ExitCode",
            "ExitSuccess",
            "ExitFailure",
            "exitSuccess",
            "exitFailure",
            "exitWith",
        ],
        "System.Environment" => &[
            "getArgs",
            "getProgName",
            "getEnv",
            "lookupEnv",
            "setEnv",
        ],
        "System.Directory" => &[
            "doesFileExist",
            "doesDirectoryExist",
            "createDirectory",
            "createDirectoryIfMissing",
            "removeFile",
            "removeDirectory",
            "getCurrentDirectory",
            "setCurrentDirectory",
        ],
        "Data.String" => &[
            "IsString",
            "fromString",
            "lines",
            "words",
            "unlines",
            "unwords",
        ],
        "Data.Proxy" => &[
            "Proxy",
            "asProxyTypeOf",
        ],
        "Data.Void" => &[
            "Void",
            "absurd",
            "vacuous",
        ],
        "Data.Word" => &[
            "Word",
            "Word8",
            "Word16",
            "Word32",
            "Word64",
        ],
        "Data.Text.IO" => &[
            "readFile",
            "writeFile",
            "appendFile",
            "hGetContents",
            "hGetLine",
            "hPutStr",
            "hPutStrLn",
            "putStr",
            "putStrLn",
            "getLine",
            "getContents",
        ],
        _ => &[],
    };

    let module_sym = Symbol::intern(module_name);
    for &export in exports {
        let qualified_name = Symbol::intern(&format!("{}.{}", module_name, export));
        let unqualified = Symbol::intern(export);

        // If the qualified name is already directly bound (e.g. as a builtin
        // with its own DefId), don't register a qualified-to-unqualified mapping
        // that would redirect to a different Prelude function.
        if ctx.lookup_value(qualified_name).is_none() {
            ctx.register_qualified_name(qualified_name, unqualified);
        }

        // Also ensure the unqualified name is defined (if not already)
        if ctx.lookup_value(unqualified).is_none() {
            let def_id = ctx.fresh_def_id();
            ctx.define(def_id, unqualified, DefKind::Value, Span::default());
            ctx.bind_value(unqualified, def_id);
        }
    }
}

/// Lower a top-level declaration.
fn lower_decl(ctx: &mut LowerContext, decl: &ast::Decl) -> LowerResult<Vec<hir::Item>> {
    match decl {
        ast::Decl::FunBind(fun_bind) => {
            // Check for pattern binding (special name $patbind)
            if fun_bind.name.name.as_str() == "$patbind"
                && fun_bind.clauses.len() == 1
                && fun_bind.clauses[0].pats.len() == 1
            {
                // Pattern binding: (x, y) = expr
                // Generate a value definition for each variable in the pattern
                let items = lower_pattern_binding(ctx, fun_bind)?;
                Ok(items)
            } else {
                let item = lower_fun_bind(ctx, fun_bind)?;
                Ok(vec![hir::Item::Value(item)])
            }
        }

        ast::Decl::DataDecl(data_decl) => {
            let (data_def, accessors) = lower_data_decl_with_accessors(ctx, data_decl)?;
            let mut items = vec![hir::Item::Data(data_def)];
            for accessor in accessors {
                items.push(hir::Item::Value(accessor));
            }
            Ok(items)
        }

        ast::Decl::Newtype(newtype_decl) => {
            let (newtype_def, accessors) = lower_newtype_decl_with_accessors(ctx, newtype_decl)?;
            let mut items = Vec::with_capacity(1 + accessors.len());
            items.push(hir::Item::Newtype(newtype_def));
            for accessor in accessors {
                items.push(hir::Item::Value(accessor));
            }
            Ok(items)
        }

        ast::Decl::TypeAlias(type_alias) => {
            let item = lower_type_alias(ctx, type_alias)?;
            Ok(vec![hir::Item::TypeAlias(item)])
        }

        ast::Decl::ClassDecl(class_decl) => {
            let item = lower_class_decl(ctx, class_decl)?;
            Ok(vec![hir::Item::Class(item)])
        }

        ast::Decl::InstanceDecl(instance_decl) => {
            let item = lower_instance_decl(ctx, instance_decl)?;
            Ok(vec![hir::Item::Instance(item)])
        }

        ast::Decl::Fixity(fixity_decl) => {
            let item = lower_fixity_decl(fixity_decl);
            Ok(vec![hir::Item::Fixity(item)])
        }

        ast::Decl::Foreign(foreign) => {
            let item = lower_foreign_decl(ctx, foreign)?;
            Ok(vec![hir::Item::Foreign(item)])
        }

        // Type signatures are associated with their definitions
        ast::Decl::TypeSig(_) => Ok(vec![]),

        // Pragmas in declarations are handled at parse time or ignored
        ast::Decl::PragmaDecl(_) => Ok(vec![]),
    }
}

/// Lower a top-level pattern binding to multiple value definitions.
///
/// For `(x, y) = (1, 2)`, generates:
/// - `x = let (x, y) = (1, 2) in x`
/// - `y = let (x, y) = (1, 2) in y`
fn lower_pattern_binding(
    ctx: &mut LowerContext,
    fun_bind: &ast::FunBind,
) -> LowerResult<Vec<hir::Item>> {
    let clause = &fun_bind.clauses[0];
    let pat = &clause.pats[0];

    // Collect all variables bound by the pattern
    let bound_vars = collect_pattern_vars(pat);
    if bound_vars.is_empty() {
        return Ok(vec![]);
    }

    // For each variable, generate a value definition
    let mut items = Vec::new();
    for (var_name, var_span) in bound_vars {
        // Look up the DefId for this variable (should have been bound in collect_module_definitions)
        let def_id = ctx
            .lookup_value(var_name)
            .expect("pattern variable should be pre-bound");

        // Create a let expression: let pat = rhs in var
        let rhs_expr = ctx.in_scope(|ctx| {
            // Bind pattern variables in this scope
            bind_pattern(ctx, pat);

            // Lower the pattern
            let hir_pat = lower_pat(ctx, pat);

            // Lower the RHS
            let hir_rhs = lower_rhs(ctx, &clause.rhs);

            // Create the binding
            let binding = hir::Binding {
                pat: hir_pat,
                sig: None,
                rhs: hir_rhs,
                span: fun_bind.span,
            };

            // Look up the variable's def_id in this scope
            let var_def_id = ctx
                .lookup_value(var_name)
                .expect("pattern variable should be bound");

            // Create the body: just reference the variable
            let body = hir::Expr::Var(ctx.def_ref(var_def_id, var_span));

            // Create the let expression
            hir::Expr::Let(vec![binding], Box::new(body), fun_bind.span)
        });

        // Create the value definition
        let value_def = hir::ValueDef {
            id: def_id,
            name: var_name,
            sig: None,
            equations: vec![hir::Equation {
                pats: vec![],
                guards: vec![],
                rhs: rhs_expr,
                span: fun_bind.span,
            }],
            span: fun_bind.span,
        };

        items.push(hir::Item::Value(value_def));
    }

    Ok(items)
}

/// Collect all variable names bound by a pattern.
fn collect_pattern_vars(pat: &ast::Pat) -> Vec<(Symbol, Span)> {
    let mut vars = Vec::new();
    collect_pattern_vars_impl(pat, &mut vars);
    vars
}

fn collect_pattern_vars_impl(pat: &ast::Pat, vars: &mut Vec<(Symbol, Span)>) {
    match pat {
        ast::Pat::Var(ident, span) => {
            vars.push((ident.name, *span));
        }
        ast::Pat::As(ident, inner, span) => {
            vars.push((ident.name, *span));
            collect_pattern_vars_impl(inner, vars);
        }
        ast::Pat::Con(_, pats, _)
        | ast::Pat::QualCon(_, _, pats, _)
        | ast::Pat::Tuple(pats, _)
        | ast::Pat::List(pats, _) => {
            for p in pats {
                collect_pattern_vars_impl(p, vars);
            }
        }
        ast::Pat::Infix(left, _, right, _) => {
            collect_pattern_vars_impl(left, vars);
            collect_pattern_vars_impl(right, vars);
        }
        ast::Pat::Record(_, fields, _, _) | ast::Pat::QualRecord(_, _, fields, _, _) => {
            for field in fields {
                if let Some(p) = &field.pat {
                    collect_pattern_vars_impl(p, vars);
                } else {
                    // Punning: Foo { x } binds x
                    vars.push((field.name.name, field.span));
                }
            }
        }
        ast::Pat::Paren(inner, _) | ast::Pat::Ann(inner, _, _) => {
            collect_pattern_vars_impl(inner, vars);
        }
        ast::Pat::Lazy(inner, _) | ast::Pat::Bang(inner, _) => {
            collect_pattern_vars_impl(inner, vars);
        }
        ast::Pat::View(_, result_pat, _) => {
            collect_pattern_vars_impl(result_pat, vars);
        }
        ast::Pat::Wildcard(_) | ast::Pat::Lit(_, _) => {}
    }
}

/// Lower a function binding.
fn lower_fun_bind(ctx: &mut LowerContext, fun_bind: &ast::FunBind) -> LowerResult<hir::ValueDef> {
    let name = fun_bind.name.name;
    let def_id = ctx
        .lookup_value(name)
        .expect("function should be pre-bound");

    let mut equations = Vec::new();
    for clause in &fun_bind.clauses {
        let eq = lower_clause(ctx, clause)?;
        equations.push(eq);
    }

    // Look up the type signature if one was declared
    let sig = ctx.lookup_type_signature(name).cloned().map(|ty| {
        lower_type_to_scheme(ctx, &ty)
    });

    Ok(hir::ValueDef {
        id: def_id,
        name,
        sig,
        equations,
        span: fun_bind.span,
    })
}

/// Lower an instance method binding with a specific DefId.
/// Unlike `lower_fun_bind`, this takes a pre-assigned DefId rather than looking
/// it up by name, since instance methods should NOT overwrite the builtin
/// method binding (e.g. `show` should still resolve to the polymorphic builtin
/// inside the method body).
fn lower_instance_method(
    ctx: &mut LowerContext,
    fun_bind: &ast::FunBind,
    def_id: hir::DefId,
) -> LowerResult<hir::ValueDef> {
    let name = fun_bind.name.name;

    let mut equations = Vec::new();
    for clause in &fun_bind.clauses {
        let eq = lower_clause(ctx, clause)?;
        equations.push(eq);
    }

    // Look up the type signature if one was declared
    let sig = ctx.lookup_type_signature(name).cloned().map(|ty| {
        lower_type_to_scheme(ctx, &ty)
    });

    Ok(hir::ValueDef {
        id: def_id,
        name,
        sig,
        equations,
        span: fun_bind.span,
    })
}

/// Lower a function clause.
fn lower_clause(ctx: &mut LowerContext, clause: &ast::Clause) -> LowerResult<hir::Equation> {
    ctx.in_scope(|ctx| {
        // Bind pattern variables
        let mut pats = Vec::new();
        for ast_pat in &clause.pats {
            bind_pattern(ctx, ast_pat);
            let pat = lower_pat(ctx, ast_pat);
            pats.push(pat);
        }

        // Lower where bindings first (they're in scope for RHS)
        if !clause.wheres.is_empty() {
            // Enter a scope for where bindings
            ctx.enter_scope();
            for where_decl in &clause.wheres {
                if let ast::Decl::FunBind(fb) = where_decl {
                    // Check for pattern binding (special name $patbind)
                    if fb.name.name.as_str() == "$patbind"
                        && fb.clauses.len() == 1
                        && fb.clauses[0].pats.len() == 1
                    {
                        // Pattern binding: bind all variables in the pattern
                        bind_pattern(ctx, &fb.clauses[0].pats[0]);
                    } else {
                        // Regular function binding
                        let def_id = ctx.fresh_def_id();
                        ctx.define(def_id, fb.name.name, DefKind::Value, fb.span);
                        ctx.bind_value(fb.name.name, def_id);
                    }
                }
            }
        }

        // Lower RHS
        let (rhs, guards) = match &clause.rhs {
            ast::Rhs::Simple(expr, _) => (lower_expr(ctx, expr), Vec::new()),
            ast::Rhs::Guarded(guarded_rhss, _) => {
                // For guarded RHS, we desugar to nested if expressions
                let rhs = desugar::desugar_guarded_rhs(
                    ctx,
                    guarded_rhss,
                    clause.span,
                    &|ctx, e| lower_expr(ctx, e),
                    &|ctx, p| lower_pat(ctx, p),
                );

                // We don't need guards in HIR for this; they're already desugared
                (rhs, Vec::new())
            }
        };

        // Wrap in let if there are where bindings
        let final_rhs = if !clause.wheres.is_empty() {
            let bindings: Vec<hir::Binding> = clause
                .wheres
                .iter()
                .filter_map(|d| {
                    if let ast::Decl::FunBind(fb) = d {
                        // Check for pattern binding (special name $patbind)
                        if fb.name.name.as_str() == "$patbind"
                            && fb.clauses.len() == 1
                            && fb.clauses[0].pats.len() == 1
                        {
                            // Pattern binding: (x, y) = expr
                            let pat = lower_pat(ctx, &fb.clauses[0].pats[0]);
                            let rhs_expr = lower_rhs(ctx, &fb.clauses[0].rhs);
                            return Some(hir::Binding {
                                pat,
                                sig: None,
                                rhs: rhs_expr,
                                span: fb.span,
                            });
                        }

                        // Look up the DefId that was bound for this where binding
                        let def_id = ctx
                            .lookup_value(fb.name.name)
                            .expect("where binding should be bound");

                        // For simple bindings (no parameters)
                        if fb.clauses.len() == 1 && fb.clauses[0].pats.is_empty() {
                            let inner_clause = &fb.clauses[0];
                            let rhs_expr = if !inner_clause.wheres.is_empty() {
                                // Handle nested where bindings
                                ctx.enter_scope();
                                for nested_decl in &inner_clause.wheres {
                                    if let ast::Decl::FunBind(nested_fb) = nested_decl {
                                        let nested_def_id = ctx.fresh_def_id();
                                        ctx.define(
                                            nested_def_id,
                                            nested_fb.name.name,
                                            DefKind::Value,
                                            nested_fb.span,
                                        );
                                        ctx.bind_value(nested_fb.name.name, nested_def_id);
                                    }
                                }
                                let body = lower_rhs(ctx, &inner_clause.rhs);
                                let nested_bindings: Vec<hir::Binding> = inner_clause
                                    .wheres
                                    .iter()
                                    .filter_map(|d| {
                                        if let ast::Decl::FunBind(nested_fb) = d {
                                            let nested_def_id = ctx
                                                .lookup_value(nested_fb.name.name)
                                                .expect("nested where binding should be bound");
                                            if nested_fb.clauses.len() == 1
                                                && nested_fb.clauses[0].pats.is_empty()
                                            {
                                                let nested_rhs =
                                                    lower_rhs(ctx, &nested_fb.clauses[0].rhs);
                                                return Some(hir::Binding {
                                                    pat: hir::Pat::Var(
                                                        nested_fb.name.name,
                                                        nested_def_id,
                                                        nested_fb.span,
                                                    ),
                                                    sig: None,
                                                    rhs: nested_rhs,
                                                    span: nested_fb.span,
                                                });
                                            }
                                        }
                                        None
                                    })
                                    .collect();
                                ctx.exit_scope();
                                hir::Expr::Let(nested_bindings, Box::new(body), fb.span)
                            } else {
                                lower_rhs(ctx, &inner_clause.rhs)
                            };
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs: rhs_expr,
                                span: fb.span,
                            });
                        }

                        // For function bindings with parameters, lower to a lambda
                        // f x y = expr  =>  f = \x -> \y -> expr
                        if fb.clauses.len() == 1 {
                            let clause = &fb.clauses[0];
                            // Enter scope and bind pattern variables
                            ctx.enter_scope();
                            for p in &clause.pats {
                                bind_pattern(ctx, p);
                            }
                            let mut pats: Vec<hir::Pat> = Vec::new();
                            for p in &clause.pats {
                                pats.push(lower_pat(ctx, p));
                            }

                            // Handle nested where bindings
                            let body = if !clause.wheres.is_empty() {
                                ctx.enter_scope();
                                // Pre-bind nested where names
                                for nested_decl in &clause.wheres {
                                    if let ast::Decl::FunBind(nested_fb) = nested_decl {
                                        let nested_def_id = ctx.fresh_def_id();
                                        ctx.define(
                                            nested_def_id,
                                            nested_fb.name.name,
                                            DefKind::Value,
                                            nested_fb.span,
                                        );
                                        ctx.bind_value(nested_fb.name.name, nested_def_id);
                                    }
                                }
                                let rhs_expr = lower_rhs(ctx, &clause.rhs);
                                // Lower nested where bindings into a Let
                                let nested_bindings: Vec<hir::Binding> = clause
                                    .wheres
                                    .iter()
                                    .filter_map(|d| {
                                        if let ast::Decl::FunBind(nested_fb) = d {
                                            let nested_def_id = ctx
                                                .lookup_value(nested_fb.name.name)
                                                .expect("nested where binding should be bound");
                                            if nested_fb.clauses.len() == 1
                                                && nested_fb.clauses[0].pats.is_empty()
                                            {
                                                let nested_rhs =
                                                    lower_rhs(ctx, &nested_fb.clauses[0].rhs);
                                                return Some(hir::Binding {
                                                    pat: hir::Pat::Var(
                                                        nested_fb.name.name,
                                                        nested_def_id,
                                                        nested_fb.span,
                                                    ),
                                                    sig: None,
                                                    rhs: nested_rhs,
                                                    span: nested_fb.span,
                                                });
                                            }
                                        }
                                        None
                                    })
                                    .collect();
                                ctx.exit_scope();
                                hir::Expr::Let(nested_bindings, Box::new(rhs_expr), fb.span)
                            } else {
                                lower_rhs(ctx, &clause.rhs)
                            };

                            ctx.exit_scope();
                            // Create a lambda expression
                            let lam = hir::Expr::Lam(pats, Box::new(body), fb.span);
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs: lam,
                                span: fb.span,
                            });
                        }

                        // For multi-clause functions, create a lambda with case
                        // f 0 = a; f n = b  =>  f = \x -> case x of { 0 -> a; n -> b }
                        if !fb.clauses.is_empty() && !fb.clauses[0].pats.is_empty() {
                            let arity = fb.clauses[0].pats.len();
                            // Create fresh variables for the lambda parameters
                            let mut param_names = Vec::new();
                            let mut param_pats = Vec::new();
                            for i in 0..arity {
                                let name = Symbol::intern(&format!("_arg{}", i));
                                let param_def_id = ctx.fresh_def_id();
                                ctx.define(param_def_id, name, DefKind::Value, fb.span);
                                ctx.bind_value(name, param_def_id);
                                param_names.push((name, param_def_id));
                                param_pats.push(hir::Pat::Var(name, param_def_id, fb.span));
                            }

                            // Build case alternatives from clauses
                            let mut alts = Vec::new();
                            for clause in &fb.clauses {
                                // Enter a scope for this alternative's pattern bindings
                                ctx.enter_scope();

                                // First bind all pattern variables
                                for p in &clause.pats {
                                    bind_pattern(ctx, p);
                                }

                                // Create a tuple pattern from clause patterns
                                let pat = if clause.pats.len() == 1 {
                                    lower_pat(ctx, &clause.pats[0])
                                } else {
                                    // Create a tuple pattern for multi-argument case
                                    // Look up the existing tuple constructor from builtins
                                    let tuple_sym = Symbol::intern(&format!(
                                        "({})",
                                        ",".repeat(clause.pats.len().saturating_sub(1))
                                    ));
                                    let tuple_def_id = ctx
                                        .lookup_constructor(tuple_sym)
                                        .expect("tuple constructor should be in builtins");
                                    let tuple_ref = ctx.def_ref(tuple_def_id, fb.span);
                                    hir::Pat::Con(
                                        tuple_ref,
                                        clause.pats.iter().map(|p| lower_pat(ctx, p)).collect(),
                                        fb.span,
                                    )
                                };
                                let body = lower_rhs(ctx, &clause.rhs);
                                ctx.exit_scope();
                                alts.push(hir::CaseAlt {
                                    pat,
                                    guards: Vec::new(),
                                    rhs: body,
                                    span: clause.span,
                                });
                            }

                            // Create the scrutinee (tuple of params or single param)
                            let scrutinee = if param_names.len() == 1 {
                                hir::Expr::Var(hir::DefRef {
                                    def_id: param_names[0].1,
                                    span: fb.span,
                                })
                            } else {
                                hir::Expr::Tuple(
                                    param_names
                                        .iter()
                                        .map(|(_, def_id)| {
                                            hir::Expr::Var(hir::DefRef {
                                                def_id: *def_id,
                                                span: fb.span,
                                            })
                                        })
                                        .collect(),
                                    fb.span,
                                )
                            };

                            let case_expr = hir::Expr::Case(Box::new(scrutinee), alts, fb.span);

                            let lam = hir::Expr::Lam(param_pats, Box::new(case_expr), fb.span);
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs: lam,
                                span: fb.span,
                            });
                        }
                    }
                    None
                })
                .collect();

            ctx.exit_scope();

            if bindings.is_empty() {
                rhs
            } else {
                hir::Expr::Let(bindings, Box::new(rhs), clause.span)
            }
        } else {
            rhs
        };

        Ok(hir::Equation {
            pats,
            guards,
            rhs: final_rhs,
            span: clause.span,
        })
    })
}

/// Lower a right-hand side.
fn lower_rhs(ctx: &mut LowerContext, rhs: &ast::Rhs) -> hir::Expr {
    match rhs {
        ast::Rhs::Simple(expr, _) => lower_expr(ctx, expr),
        ast::Rhs::Guarded(guards, span) => desugar::desugar_guarded_rhs(
            ctx,
            guards,
            *span,
            &|ctx, e| lower_expr(ctx, e),
            &|ctx, p| lower_pat(ctx, p),
        ),
    }
}

/// Lower an expression.
fn lower_expr(ctx: &mut LowerContext, expr: &ast::Expr) -> hir::Expr {
    match expr {
        ast::Expr::Var(ident, span) => {
            let name = ident.name;
            if let Some(def_id) = resolve_var(ctx, name, *span) {
                hir::Expr::Var(ctx.def_ref(def_id, *span))
            } else {
                // Create placeholder for error recovery
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Value, *span);
                hir::Expr::Var(ctx.def_ref(def_id, *span))
            }
        }

        ast::Expr::QualVar(module_name, ident, span) => {
            // Qualified variable like M.foo or Data.Map.lookup
            let qualifier = Symbol::intern(&module_name.to_string());
            let name = ident.name;

            // Use the qualified name resolution which handles aliases
            if let Some(def_id) = ctx.resolve_qualified_var(qualifier, name) {
                // Warn if this is a stub
                let qual_name = format!("{}.{}", module_name.to_string(), name.as_str());
                ctx.warn_if_stub(def_id, &qual_name, *span);
                hir::Expr::Var(ctx.def_ref(def_id, *span))
            } else {
                // Fall back to creating a placeholder with the full qualified name
                let qual_name = format!("{}.{}", module_name.to_string(), name.as_str());
                let qual_sym = Symbol::intern(&qual_name);
                ctx.error(crate::LowerError::UnboundVar {
                    name: qual_name,
                    span: *span,
                });
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, qual_sym, DefKind::Value, *span);
                hir::Expr::Var(ctx.def_ref(def_id, *span))
            }
        }

        ast::Expr::Con(ident, span) => {
            let name = ident.name;
            if let Some(def_id) = resolve_constructor(ctx, name, *span) {
                hir::Expr::Con(ctx.def_ref(def_id, *span))
            } else {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Constructor, *span);
                hir::Expr::Con(ctx.def_ref(def_id, *span))
            }
        }

        ast::Expr::QualCon(module_name, ident, span) => {
            // Qualified constructor like M.Just or Data.Maybe.Just
            let qualifier = Symbol::intern(&module_name.to_string());
            let name = ident.name;

            // Use the qualified constructor resolution which handles aliases
            if let Some(def_id) = ctx.resolve_qualified_constructor(qualifier, name) {
                // Warn if this is a stub
                let qual_name = format!("{}.{}", module_name.to_string(), name.as_str());
                ctx.warn_if_stub(def_id, &qual_name, *span);
                hir::Expr::Con(ctx.def_ref(def_id, *span))
            } else {
                // Fall back to creating a placeholder with the full qualified name
                let qual_name = format!("{}.{}", module_name.to_string(), name.as_str());
                let qual_sym = Symbol::intern(&qual_name);
                ctx.error(crate::LowerError::UnboundCon {
                    name: qual_name,
                    span: *span,
                });
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, qual_sym, DefKind::Constructor, *span);
                hir::Expr::Con(ctx.def_ref(def_id, *span))
            }
        }

        ast::Expr::Lit(lit, span) => {
            let hir_lit = lower_lit(lit);
            hir::Expr::Lit(hir_lit, *span)
        }

        ast::Expr::App(fun, arg, span) => {
            let f = lower_expr(ctx, fun);
            let a = lower_expr(ctx, arg);
            hir::Expr::App(Box::new(f), Box::new(a), *span)
        }

        ast::Expr::Infix(lhs, op, rhs, span) => {
            // Desugar infix to prefix: a `op` b -> op a b
            // For qualified names like "E.catch", check the last part (after the last dot)
            let op_name_str = op.name.as_str();
            let is_con = if let Some(dot_pos) = op_name_str.rfind('.') {
                // Qualified name - check the part after the last dot
                op_name_str[dot_pos + 1..]
                    .chars()
                    .next()
                    .map_or(false, |c| c.is_uppercase())
            } else {
                // Unqualified name - check the first character
                op_name_str
                    .chars()
                    .next()
                    .map_or(false, |c| c.is_uppercase())
            };
            let op_expr = if is_con {
                // Constructor
                if let Some(def_id) = resolve_constructor(ctx, op.name, *span) {
                    hir::Expr::Con(ctx.def_ref(def_id, *span))
                } else {
                    hir::Expr::Error(*span)
                }
            } else {
                // Variable/operator
                if let Some(def_id) = resolve_var(ctx, op.name, *span) {
                    hir::Expr::Var(ctx.def_ref(def_id, *span))
                } else {
                    // Create placeholder
                    let def_id = ctx.fresh_def_id();
                    ctx.define(def_id, op.name, DefKind::Value, *span);
                    hir::Expr::Var(ctx.def_ref(def_id, *span))
                }
            };
            let l = lower_expr(ctx, lhs);
            let r = lower_expr(ctx, rhs);
            let app1 = hir::Expr::App(Box::new(op_expr), Box::new(l), *span);
            hir::Expr::App(Box::new(app1), Box::new(r), *span)
        }

        ast::Expr::Neg(inner, span) => {
            // Desugar negation: -e -> negate e
            let negate_sym = Symbol::intern("negate");
            if let Some(def_id) = ctx.lookup_value(negate_sym) {
                let negate = hir::Expr::Var(ctx.def_ref(def_id, *span));
                let e = lower_expr(ctx, inner);
                hir::Expr::App(Box::new(negate), Box::new(e), *span)
            } else {
                hir::Expr::Error(*span)
            }
        }

        ast::Expr::Lam(pats, body, span) => ctx.in_scope(|ctx| {
            let mut hir_pats = Vec::new();
            for p in pats {
                bind_pattern(ctx, p);
                hir_pats.push(lower_pat(ctx, p));
            }
            let e = lower_expr(ctx, body);
            hir::Expr::Lam(hir_pats, Box::new(e), *span)
        }),

        ast::Expr::Let(decls, body, span) => ctx.in_scope(|ctx| {
            // Bind all declarations first
            for decl in decls {
                if let ast::Decl::FunBind(fb) = decl {
                    // Check for pattern binding (special name $patbind)
                    if fb.name.name.as_str() == "$patbind"
                        && fb.clauses.len() == 1
                        && fb.clauses[0].pats.len() == 1
                    {
                        // Pattern binding: (x, y) = expr
                        // Bind variables from the pattern
                        bind_pattern(ctx, &fb.clauses[0].pats[0]);
                    } else {
                        // Regular function binding
                        let def_id = ctx.fresh_def_id();
                        ctx.define(def_id, fb.name.name, DefKind::Value, fb.span);
                        ctx.bind_value(fb.name.name, def_id);
                    }
                }
            }

            // Lower bindings
            let bindings: Vec<hir::Binding> = decls
                .iter()
                .filter_map(|d| {
                    if let ast::Decl::FunBind(fb) = d {
                        // Check for pattern binding (special name $patbind)
                        if fb.name.name.as_str() == "$patbind"
                            && fb.clauses.len() == 1
                            && fb.clauses[0].pats.len() == 1
                        {
                            // Pattern binding: (x, y) = expr
                            let pat = lower_pat(ctx, &fb.clauses[0].pats[0]);
                            let rhs_expr = lower_rhs(ctx, &fb.clauses[0].rhs);
                            return Some(hir::Binding {
                                pat,
                                sig: None,
                                rhs: rhs_expr,
                                span: fb.span,
                            });
                        }

                        let def_id = ctx
                            .lookup_value(fb.name.name)
                            .expect("let binding should be bound");

                        // Simple binding (no parameters): let x = expr
                        if fb.clauses.len() == 1 && fb.clauses[0].pats.is_empty() {
                            let rhs_expr = lower_rhs(ctx, &fb.clauses[0].rhs);
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs: rhs_expr,
                                span: fb.span,
                            });
                        }

                        // Single-clause function: let f x y = expr => f = \x -> \y -> expr
                        if fb.clauses.len() == 1 {
                            let clause = &fb.clauses[0];
                            ctx.enter_scope();
                            for p in &clause.pats {
                                bind_pattern(ctx, p);
                            }
                            let mut pats: Vec<hir::Pat> = Vec::new();
                            for p in &clause.pats {
                                pats.push(lower_pat(ctx, p));
                            }
                            let body_expr = lower_rhs(ctx, &clause.rhs);
                            ctx.exit_scope();
                            let lam = hir::Expr::Lam(pats, Box::new(body_expr), fb.span);
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs: lam,
                                span: fb.span,
                            });
                        }

                        // Multi-clause function: let f 0 = a; f n = b => f = \x -> case x of ...
                        if !fb.clauses.is_empty() && !fb.clauses[0].pats.is_empty() {
                            let arity = fb.clauses[0].pats.len();
                            // Create fresh variables for the lambda parameters
                            let mut param_names = Vec::new();
                            let mut param_pats = Vec::new();
                            for i in 0..arity {
                                let name = Symbol::intern(&format!("_arg{}", i));
                                let param_def_id = ctx.fresh_def_id();
                                ctx.define(param_def_id, name, DefKind::Value, fb.span);
                                ctx.bind_value(name, param_def_id);
                                param_names.push((name, param_def_id));
                                param_pats.push(hir::Pat::Var(name, param_def_id, fb.span));
                            }

                            // Build case alternatives from clauses
                            let mut alts = Vec::new();
                            for clause in &fb.clauses {
                                ctx.enter_scope();
                                for p in &clause.pats {
                                    bind_pattern(ctx, p);
                                }
                                let pat = if clause.pats.len() == 1 {
                                    lower_pat(ctx, &clause.pats[0])
                                } else {
                                    // Look up the existing tuple constructor from builtins
                                    let tuple_sym = Symbol::intern(&format!(
                                        "({})",
                                        ",".repeat(clause.pats.len().saturating_sub(1))
                                    ));
                                    let tuple_def_id = ctx
                                        .lookup_constructor(tuple_sym)
                                        .expect("tuple constructor should be in builtins");
                                    let tuple_ref = ctx.def_ref(tuple_def_id, fb.span);
                                    hir::Pat::Con(
                                        tuple_ref,
                                        clause.pats.iter().map(|p| lower_pat(ctx, p)).collect(),
                                        fb.span,
                                    )
                                };
                                let clause_body = lower_rhs(ctx, &clause.rhs);
                                ctx.exit_scope();
                                alts.push(hir::CaseAlt {
                                    pat,
                                    guards: Vec::new(),
                                    rhs: clause_body,
                                    span: clause.span,
                                });
                            }

                            // Create the scrutinee (tuple of params or single param)
                            let scrutinee = if param_names.len() == 1 {
                                hir::Expr::Var(hir::DefRef {
                                    def_id: param_names[0].1,
                                    span: fb.span,
                                })
                            } else {
                                let tuple_elems: Vec<hir::Expr> = param_names
                                    .iter()
                                    .map(|(_n, id)| {
                                        hir::Expr::Var(hir::DefRef {
                                            def_id: *id,
                                            span: fb.span,
                                        })
                                    })
                                    .collect();
                                hir::Expr::Tuple(tuple_elems, fb.span)
                            };

                            let case_expr = hir::Expr::Case(Box::new(scrutinee), alts, fb.span);
                            let lam = hir::Expr::Lam(param_pats, Box::new(case_expr), fb.span);
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs: lam,
                                span: fb.span,
                            });
                        }
                    }
                    None
                })
                .collect();

            let e = lower_expr(ctx, body);
            hir::Expr::Let(bindings, Box::new(e), *span)
        }),

        ast::Expr::If(cond, then_branch, else_branch, span) => {
            let c = lower_expr(ctx, cond);
            let t = lower_expr(ctx, then_branch);
            let e = lower_expr(ctx, else_branch);
            hir::Expr::If(Box::new(c), Box::new(t), Box::new(e), *span)
        }

        ast::Expr::Case(scrutinee, alts, span) => {
            let s = lower_expr(ctx, scrutinee);
            let hir_alts: Vec<hir::CaseAlt> = alts.iter().map(|alt| lower_alt(ctx, alt)).collect();
            hir::Expr::Case(Box::new(s), hir_alts, *span)
        }

        ast::Expr::Do(stmts, span) => desugar::desugar_do(
            ctx,
            stmts,
            *span,
            |ctx, e| lower_expr(ctx, e),
            |ctx, p| lower_pat(ctx, p),
        ),

        ast::Expr::ListComp(expr, stmts, span) => desugar::desugar_list_comp(
            ctx,
            expr,
            stmts,
            *span,
            |ctx, e| lower_expr(ctx, e),
            |ctx, p| lower_pat(ctx, p),
        ),

        ast::Expr::Tuple(exprs, span) => {
            let es: Vec<hir::Expr> = exprs.iter().map(|e| lower_expr(ctx, e)).collect();
            hir::Expr::Tuple(es, *span)
        }

        ast::Expr::List(exprs, span) => {
            let es: Vec<hir::Expr> = exprs.iter().map(|e| lower_expr(ctx, e)).collect();
            hir::Expr::List(es, *span)
        }

        ast::Expr::ArithSeq(seq, span) => {
            // Desugar arithmetic sequences
            lower_arith_seq(ctx, seq, *span)
        }

        ast::Expr::RecordCon(con, fields, has_wildcard, span) => {
            let con_name = con.name;
            if let Some(def_id) = resolve_constructor(ctx, con_name, *span) {
                let con_ref = ctx.def_ref(def_id, *span);
                let mut hir_fields = Vec::with_capacity(fields.len());
                for f in fields {
                    let value = match &f.value {
                        Some(e) => lower_expr(ctx, e),
                        None => {
                            // Punning: Foo { x } means Foo { x = x }
                            let name = f.name.name;
                            if let Some(def_id) = ctx.lookup_value(name) {
                                hir::Expr::Var(ctx.def_ref(def_id, f.span))
                            } else {
                                hir::Expr::Error(f.span)
                            }
                        }
                    };
                    hir_fields.push(hir::FieldExpr {
                        name: f.name.name,
                        value,
                        span: f.span,
                    });
                }
                // RecordWildCards: expand `..` to include remaining fields from scope
                if *has_wildcard {
                    if let Some(field_names) = ctx.get_constructor_field_names(def_id) {
                        let existing: rustc_hash::FxHashSet<_> =
                            hir_fields.iter().map(|f| f.name).collect();
                        for field_name in field_names {
                            if !existing.contains(&field_name) {
                                if let Some(val_id) = ctx.lookup_value(field_name) {
                                    hir_fields.push(hir::FieldExpr {
                                        name: field_name,
                                        value: hir::Expr::Var(ctx.def_ref(val_id, *span)),
                                        span: *span,
                                    });
                                }
                            }
                        }
                    }
                }
                hir::Expr::Record(con_ref, hir_fields, *span)
            } else {
                hir::Expr::Error(*span)
            }
        }

        ast::Expr::RecordUpd(base, fields, span) => {
            let b = lower_expr(ctx, base);
            let mut hir_fields = Vec::with_capacity(fields.len());
            for f in fields {
                let value = match &f.value {
                    Some(e) => lower_expr(ctx, e),
                    None => hir::Expr::Error(f.span),
                };
                hir_fields.push(hir::FieldExpr {
                    name: f.name.name,
                    value,
                    span: f.span,
                });
            }
            hir::Expr::RecordUpdate(Box::new(b), hir_fields, *span)
        }

        ast::Expr::Ann(expr, ty, span) => {
            let e = lower_expr(ctx, expr);
            let t = lower_type(ctx, ty);
            hir::Expr::Ann(Box::new(e), t, *span)
        }

        ast::Expr::Paren(inner, _) => lower_expr(ctx, inner),

        ast::Expr::Lazy(inner, _span) => {
            // For now, just lower the inner expression
            // TODO: Handle lazy block semantics
            lower_expr(ctx, inner)
        }

        // Wildcard in expression context is a typed hole
        ast::Expr::Wildcard(span) => hir::Expr::Error(*span),
    }
}

/// Lower a case alternative.
fn lower_alt(ctx: &mut LowerContext, alt: &ast::Alt) -> hir::CaseAlt {
    ctx.in_scope(|ctx| {
        // Bind pattern variables first
        bind_pattern(ctx, &alt.pat);
        let pat = lower_pat(ctx, &alt.pat);

        // Handle where bindings (they scope over the RHS)
        if !alt.wheres.is_empty() {
            ctx.enter_scope();
            // Pre-bind all where clause names
            for where_decl in &alt.wheres {
                if let ast::Decl::FunBind(fb) = where_decl {
                    if fb.name.name.as_str() == "$patbind"
                        && fb.clauses.len() == 1
                        && fb.clauses[0].pats.len() == 1
                    {
                        // Pattern binding: bind all variables in the pattern
                        bind_pattern(ctx, &fb.clauses[0].pats[0]);
                    } else {
                        // Regular function binding
                        let def_id = ctx.fresh_def_id();
                        ctx.define(def_id, fb.name.name, DefKind::Value, fb.span);
                        ctx.bind_value(fb.name.name, def_id);
                    }
                }
            }
        }

        // Lower the RHS (where clause bindings are now in scope)
        let rhs_expr = lower_rhs(ctx, &alt.rhs);

        // Wrap in let if there are where bindings
        let final_rhs = if !alt.wheres.is_empty() {
            let bindings: Vec<hir::Binding> = alt
                .wheres
                .iter()
                .filter_map(|d| {
                    if let ast::Decl::FunBind(fb) = d {
                        // Check for pattern binding (special name $patbind)
                        if fb.name.name.as_str() == "$patbind"
                            && fb.clauses.len() == 1
                            && fb.clauses[0].pats.len() == 1
                        {
                            // Pattern binding: (x, y) = expr or (x :| xs) = expr
                            let pat = lower_pat(ctx, &fb.clauses[0].pats[0]);
                            let rhs = lower_rhs(ctx, &fb.clauses[0].rhs);
                            return Some(hir::Binding {
                                pat,
                                sig: None,
                                rhs,
                                span: fb.span,
                            });
                        }

                        // Look up the DefId that was bound for this where binding
                        let def_id = ctx
                            .lookup_value(fb.name.name)
                            .expect("where binding should be bound");

                        // For simple bindings (no parameters)
                        if fb.clauses.len() == 1 && fb.clauses[0].pats.is_empty() {
                            let rhs = lower_rhs(ctx, &fb.clauses[0].rhs);
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs,
                                span: fb.span,
                            });
                        }

                        // For function bindings with parameters, lower to a lambda
                        if fb.clauses.len() == 1 {
                            let clause = &fb.clauses[0];
                            ctx.enter_scope();
                            for p in &clause.pats {
                                bind_pattern(ctx, p);
                            }
                            let mut pats: Vec<hir::Pat> = Vec::new();
                            for p in &clause.pats {
                                pats.push(lower_pat(ctx, p));
                            }
                            let body = lower_rhs(ctx, &clause.rhs);
                            ctx.exit_scope();

                            let lam = hir::Expr::Lam(pats, Box::new(body), fb.span);
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs: lam,
                                span: fb.span,
                            });
                        }

                        // Multi-clause where bindings would need more work
                        None
                    } else {
                        None
                    }
                })
                .collect();

            ctx.exit_scope();
            hir::Expr::Let(bindings, Box::new(rhs_expr), alt.span)
        } else {
            rhs_expr
        };

        hir::CaseAlt {
            pat,
            guards: vec![],
            rhs: final_rhs,
            span: alt.span,
        }
    })
}

/// Lower a pattern.
fn lower_pat(ctx: &mut LowerContext, pat: &ast::Pat) -> hir::Pat {
    match pat {
        ast::Pat::Var(ident, span) => {
            // Look up the DefId that was bound by bind_pattern
            let def_id = ctx
                .lookup_value(ident.name)
                .expect("pattern variable should be bound");
            hir::Pat::Var(ident.name, def_id, *span)
        }

        ast::Pat::Wildcard(span) => hir::Pat::Wild(*span),

        ast::Pat::Lit(lit, span) => {
            let hir_lit = lower_lit(lit);
            hir::Pat::Lit(hir_lit, *span)
        }

        ast::Pat::Con(ident, pats, span) => {
            let con_name = ident.name;
            if let Some(def_id) = resolve_constructor(ctx, con_name, *span) {
                let con_ref = ctx.def_ref(def_id, *span);
                let hir_pats: Vec<hir::Pat> = pats.iter().map(|p| lower_pat(ctx, p)).collect();
                hir::Pat::Con(con_ref, hir_pats, *span)
            } else {
                hir::Pat::Error(*span)
            }
        }

        ast::Pat::Infix(lhs, op, rhs, span) => {
            // Desugar infix pattern: x : xs -> (:) x xs
            let con_name = op.name;
            if let Some(def_id) = resolve_constructor(ctx, con_name, *span) {
                let con_ref = ctx.def_ref(def_id, *span);
                let l = lower_pat(ctx, lhs);
                let r = lower_pat(ctx, rhs);
                hir::Pat::Con(con_ref, vec![l, r], *span)
            } else {
                hir::Pat::Error(*span)
            }
        }

        ast::Pat::Tuple(pats, span) => {
            // Tuple pattern is sugar for tuple constructor
            let tuple_sym =
                Symbol::intern(&format!("({})", ",".repeat(pats.len().saturating_sub(1))));
            // Use existing constructor if available, otherwise create one
            let def_id = ctx.lookup_constructor(tuple_sym).unwrap_or_else(|| {
                let id = ctx.fresh_def_id();
                ctx.define(id, tuple_sym, DefKind::Constructor, *span);
                id
            });
            let tuple_ref = ctx.def_ref(def_id, *span);

            let hir_pats: Vec<hir::Pat> = pats.iter().map(|p| lower_pat(ctx, p)).collect();
            hir::Pat::Con(tuple_ref, hir_pats, *span)
        }

        ast::Pat::List(pats, span) => {
            // Desugar list pattern to cons chain
            desugar_list_pat(ctx, pats, *span)
        }

        ast::Pat::As(ident, inner, span) => {
            // Look up the DefId that was bound by bind_pattern
            let def_id = ctx
                .lookup_value(ident.name)
                .expect("as-pattern should be bound");
            let p = lower_pat(ctx, inner);
            hir::Pat::As(ident.name, def_id, Box::new(p), *span)
        }

        ast::Pat::Lazy(inner, _span) => {
            // For now, just lower the inner pattern
            lower_pat(ctx, inner)
        }

        ast::Pat::Bang(inner, _span) => {
            // For now, just lower the inner pattern
            lower_pat(ctx, inner)
        }

        ast::Pat::Record(con, fields, has_wildcard, span) => {
            let con_name = con.name;
            if let Some(def_id) = resolve_constructor(ctx, con_name, *span) {
                let con_ref = ctx.def_ref(def_id, *span);
                let mut hir_field_pats: Vec<hir::FieldPat> = Vec::with_capacity(fields.len());
                for f in fields {
                    let pat = match &f.pat {
                        Some(p) => lower_pat(ctx, p),
                        None => {
                            // Punned field: Foo { x } binds x
                            let field_def_id = ctx
                                .lookup_value(f.name.name)
                                .expect("punned field should be bound");
                            hir::Pat::Var(f.name.name, field_def_id, f.span)
                        }
                    };
                    hir_field_pats.push(hir::FieldPat {
                        name: f.name.name,
                        pat,
                        span: f.span,
                    });
                }
                // RecordWildCards: expand `..` to bind remaining fields as variables
                if *has_wildcard {
                    if let Some(field_names) = ctx.get_constructor_field_names(def_id) {
                        let existing: rustc_hash::FxHashSet<_> =
                            hir_field_pats.iter().map(|f| f.name).collect();
                        for field_name in field_names {
                            if !existing.contains(&field_name) {
                                // Reuse the DefId already bound by collect_pattern_bindings
                                let field_def_id = ctx
                                    .lookup_value(field_name)
                                    .expect("wildcard field should be bound");
                                hir_field_pats.push(hir::FieldPat {
                                    name: field_name,
                                    pat: hir::Pat::Var(field_name, field_def_id, *span),
                                    span: *span,
                                });
                            }
                        }
                    }
                }
                hir::Pat::RecordCon(con_ref, hir_field_pats, *span)
            } else {
                hir::Pat::Error(*span)
            }
        }

        ast::Pat::QualCon(module_name, ident, pats, span) => {
            // Qualified constructor like W.StackSet x y
            let qualifier = Symbol::intern(&module_name.to_string());
            let con_name = ident.name;
            if let Some(def_id) = ctx.resolve_qualified_constructor(qualifier, con_name) {
                let con_ref = ctx.def_ref(def_id, *span);
                let hir_pats: Vec<hir::Pat> = pats.iter().map(|p| lower_pat(ctx, p)).collect();
                hir::Pat::Con(con_ref, hir_pats, *span)
            } else {
                // Fall back to creating a placeholder with the full qualified name
                let qual_name = format!("{}.{}", module_name.to_string(), con_name.as_str());
                let qual_sym = Symbol::intern(&qual_name);
                ctx.error(crate::LowerError::UnboundCon {
                    name: qual_name,
                    span: *span,
                });
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, qual_sym, DefKind::Constructor, *span);
                let con_ref = ctx.def_ref(def_id, *span);
                let hir_pats: Vec<hir::Pat> = pats.iter().map(|p| lower_pat(ctx, p)).collect();
                hir::Pat::Con(con_ref, hir_pats, *span)
            }
        }

        ast::Pat::QualRecord(module_name, con, fields, _has_wildcard, span) => {
            // Qualified record pattern like XMonad.XConfig { modMask = m }
            let qualifier = Symbol::intern(&module_name.to_string());
            let con_name = con.name;
            if let Some(def_id) = ctx.resolve_qualified_constructor(qualifier, con_name) {
                let con_ref = ctx.def_ref(def_id, *span);
                let mut hir_field_pats: Vec<hir::FieldPat> = Vec::with_capacity(fields.len());
                for f in fields {
                    let pat = match &f.pat {
                        Some(p) => lower_pat(ctx, p),
                        None => {
                            // Punned field: Foo { x } binds x
                            let field_def_id = ctx
                                .lookup_value(f.name.name)
                                .expect("punned field should be bound");
                            hir::Pat::Var(f.name.name, field_def_id, f.span)
                        }
                    };
                    hir_field_pats.push(hir::FieldPat {
                        name: f.name.name,
                        pat,
                        span: f.span,
                    });
                }
                hir::Pat::RecordCon(con_ref, hir_field_pats, *span)
            } else {
                // Fall back to creating a placeholder with the full qualified name
                let qual_name = format!("{}.{}", module_name.to_string(), con_name.as_str());
                let qual_sym = Symbol::intern(&qual_name);
                ctx.error(crate::LowerError::UnboundCon {
                    name: qual_name,
                    span: *span,
                });
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, qual_sym, DefKind::Constructor, *span);
                let con_ref = ctx.def_ref(def_id, *span);
                let mut hir_field_pats: Vec<hir::FieldPat> = Vec::with_capacity(fields.len());
                for f in fields {
                    let pat = match &f.pat {
                        Some(p) => lower_pat(ctx, p),
                        None => {
                            let field_def_id = ctx
                                .lookup_value(f.name.name)
                                .expect("punned field should be bound");
                            hir::Pat::Var(f.name.name, field_def_id, f.span)
                        }
                    };
                    hir_field_pats.push(hir::FieldPat {
                        name: f.name.name,
                        pat,
                        span: f.span,
                    });
                }
                hir::Pat::RecordCon(con_ref, hir_field_pats, *span)
            }
        }

        ast::Pat::Paren(inner, _) => lower_pat(ctx, inner),

        ast::Pat::Ann(inner, ty, span) => {
            let p = lower_pat(ctx, inner);
            let t = lower_type(ctx, ty);
            hir::Pat::Ann(Box::new(p), t, *span)
        }

        // View patterns: (expr -> pat)
        // Lower both the view expression and result pattern into HIR.
        // The actual desugaring happens in HIR→Core pattern compilation.
        ast::Pat::View(view_expr, result_pat, span) => {
            let hir_view_expr = lower_expr(ctx, view_expr);
            let hir_result_pat = lower_pat(ctx, result_pat);
            hir::Pat::View(Box::new(hir_view_expr), Box::new(hir_result_pat), *span)
        }
    }
}

/// Desugar a list pattern [p1, p2, ...] to p1 : p2 : ... : []
fn desugar_list_pat(ctx: &mut LowerContext, pats: &[ast::Pat], span: Span) -> hir::Pat {
    let nil_sym = Symbol::intern("[]");
    let cons_sym = Symbol::intern(":");

    // Start with nil pattern
    let nil_def = ctx.lookup_constructor(nil_sym).unwrap_or_else(|| {
        let id = ctx.fresh_def_id();
        ctx.define(id, nil_sym, DefKind::Constructor, span);
        id
    });
    let nil_ref = ctx.def_ref(nil_def, span);

    pats.iter()
        .rev()
        .fold(hir::Pat::Con(nil_ref.clone(), vec![], span), |acc, p| {
            let cons_def = ctx.lookup_constructor(cons_sym).unwrap_or_else(|| {
                let id = ctx.fresh_def_id();
                ctx.define(id, cons_sym, DefKind::Constructor, span);
                id
            });
            let cons_ref = ctx.def_ref(cons_def, span);
            let hir_p = lower_pat(ctx, p);
            hir::Pat::Con(cons_ref, vec![hir_p, acc], span)
        })
}

/// Lower a literal.
fn lower_lit(lit: &ast::Lit) -> hir::Lit {
    match lit {
        ast::Lit::Int(n) => hir::Lit::Int(*n as i128),
        ast::Lit::Float(f) => hir::Lit::Float(*f),
        ast::Lit::Char(c) => hir::Lit::Char(*c),
        ast::Lit::String(s) => hir::Lit::String(Symbol::intern(s)),
    }
}

/// Lower a type.
/// Lower an AST type into a type scheme, preserving constraints.
///
/// If the type has a `Constrained` wrapper (`Eq a => a -> Bool`), the
/// constraints are extracted and placed in the scheme. Free type variables
/// in the type are quantified over.
fn lower_type_to_scheme(ctx: &mut LowerContext, ty: &ast::Type) -> bhc_types::Scheme {
    match ty {
        ast::Type::Constrained(constraints, inner, _span) => {
            let inner_ty = lower_type(ctx, inner);

            // Lower AST constraints to bhc_types::Constraint
            let type_constraints: Vec<bhc_types::Constraint> = constraints
                .iter()
                .map(|c| {
                    let args: Vec<bhc_types::Ty> =
                        c.args.iter().map(|a| lower_type(ctx, a)).collect();
                    bhc_types::Constraint::new_multi(c.class.name, args, c.span)
                })
                .collect();

            // Collect free type variables for quantification
            let mut fvs = inner_ty.free_vars();
            for c in &type_constraints {
                for arg in &c.args {
                    for v in arg.free_vars() {
                        if !fvs.iter().any(|fv| fv.id == v.id) {
                            fvs.push(v);
                        }
                    }
                }
            }

            if fvs.is_empty() && type_constraints.is_empty() {
                bhc_types::Scheme::mono(inner_ty)
            } else {
                bhc_types::Scheme::qualified(fvs, type_constraints, inner_ty)
            }
        }
        ast::Type::Forall(vars, inner, _) => {
            // Handle explicit forall: `forall a. C a => a -> a`
            let ty_vars: Vec<bhc_types::TyVar> = vars
                .iter()
                .map(|v| bhc_types::TyVar::new_star(v.name.name.as_u32()))
                .collect();
            let inner_scheme = lower_type_to_scheme(ctx, inner);
            bhc_types::Scheme {
                vars: ty_vars,
                constraints: inner_scheme.constraints,
                ty: inner_scheme.ty,
            }
        }
        _ => {
            // No constraints — produce a monomorphic scheme
            bhc_types::Scheme::mono(lower_type(ctx, ty))
        }
    }
}

fn lower_type(ctx: &mut LowerContext, ty: &ast::Type) -> bhc_types::Ty {
    match ty {
        ast::Type::Var(tyvar, _) => {
            bhc_types::Ty::Var(bhc_types::TyVar::new_star(tyvar.name.name.as_u32()))
        }

        ast::Type::Con(ident, _) => {
            bhc_types::Ty::Con(bhc_types::TyCon::new(ident.name, bhc_types::Kind::Star))
        }

        ast::Type::App(f, a, _) => {
            let fun_ty = lower_type(ctx, f);
            let arg_ty = lower_type(ctx, a);
            bhc_types::Ty::App(Box::new(fun_ty), Box::new(arg_ty))
        }

        ast::Type::Fun(from, to, _) => {
            let from_ty = lower_type(ctx, from);
            let to_ty = lower_type(ctx, to);
            bhc_types::Ty::Fun(Box::new(from_ty), Box::new(to_ty))
        }

        ast::Type::Tuple(tys, _) => {
            let hir_tys: Vec<bhc_types::Ty> = tys.iter().map(|t| lower_type(ctx, t)).collect();
            bhc_types::Ty::Tuple(hir_tys)
        }

        ast::Type::List(elem, _) => {
            let elem_ty = lower_type(ctx, elem);
            bhc_types::Ty::List(Box::new(elem_ty))
        }

        ast::Type::Paren(inner, _) => lower_type(ctx, inner),

        ast::Type::Forall(vars, inner, _) => {
            let ty_vars: Vec<bhc_types::TyVar> = vars
                .iter()
                .map(|v| bhc_types::TyVar::new_star(v.name.name.as_u32()))
                .collect();
            let inner_ty = lower_type(ctx, inner);
            bhc_types::Ty::Forall(ty_vars, Box::new(inner_ty))
        }

        ast::Type::Constrained(_, inner, _) => {
            // TODO: handle constraints properly
            lower_type(ctx, inner)
        }

        ast::Type::QualCon(module_name, ident, _) => {
            // Qualified type constructor like M.Map
            // Create a qualified name symbol by combining module and name
            let qual_name = format!("{}.{}", module_name.to_string(), ident.name.as_str());
            let symbol = Symbol::intern(&qual_name);
            bhc_types::Ty::Con(bhc_types::TyCon::new(symbol, bhc_types::Kind::Star))
        }

        ast::Type::NatLit(n, _) => bhc_types::Ty::nat_lit(*n),

        ast::Type::Bang(inner, _) | ast::Type::Lazy(inner, _) => {
            // Strip strictness/laziness annotations and lower the inner type
            lower_type(ctx, inner)
        }

        ast::Type::PromotedList(_, _) => {
            // TODO: handle promoted lists
            bhc_types::Ty::Error
        }
    }
}

/// Lower an arithmetic sequence.
fn lower_arith_seq(ctx: &mut LowerContext, seq: &ast::ArithSeq, span: Span) -> hir::Expr {
    // Desugar arithmetic sequences to enumFrom* calls
    let (func_name, args) = match seq {
        ast::ArithSeq::From(start) => ("enumFrom", vec![lower_expr(ctx, start)]),
        ast::ArithSeq::FromThen(start, next) => (
            "enumFromThen",
            vec![lower_expr(ctx, start), lower_expr(ctx, next)],
        ),
        ast::ArithSeq::FromTo(start, end) => (
            "enumFromTo",
            vec![lower_expr(ctx, start), lower_expr(ctx, end)],
        ),
        ast::ArithSeq::FromThenTo(start, next, end) => (
            "enumFromThenTo",
            vec![
                lower_expr(ctx, start),
                lower_expr(ctx, next),
                lower_expr(ctx, end),
            ],
        ),
    };

    let func_sym = Symbol::intern(func_name);
    let func = if let Some(def_id) = ctx.lookup_value(func_sym) {
        hir::Expr::Var(ctx.def_ref(def_id, span))
    } else {
        let def_id = ctx.fresh_def_id();
        ctx.define(def_id, func_sym, DefKind::Value, span);
        hir::Expr::Var(ctx.def_ref(def_id, span))
    };

    args.into_iter()
        .fold(func, |f, a| hir::Expr::App(Box::new(f), Box::new(a), span))
}

/// Lower an import declaration.
fn lower_import(imp: &ast::ImportDecl) -> hir::Import {
    let module_name = imp
        .module
        .parts
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(".");

    hir::Import {
        module: Symbol::intern(&module_name),
        qualified: imp.qualified,
        alias: imp.alias.as_ref().map(|a| {
            let alias_name = a
                .parts
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(".");
            Symbol::intern(&alias_name)
        }),
        items: imp.spec.as_ref().map(|spec| match spec {
            ast::ImportSpec::Only(items) | ast::ImportSpec::Hiding(items) => items
                .iter()
                .map(|item| match item {
                    ast::Import::Var(ident, span) => hir::ImportItem {
                        name: ident.name,
                        children: hir::ExportChildren::None,
                        span: *span,
                    },
                    ast::Import::Type(ident, children, span) => hir::ImportItem {
                        name: ident.name,
                        children: children.as_ref().map_or(hir::ExportChildren::None, |cs| {
                            hir::ExportChildren::Some(cs.iter().map(|c| c.name).collect())
                        }),
                        span: *span,
                    },
                })
                .collect(),
        }),
        hiding: matches!(imp.spec, Some(ast::ImportSpec::Hiding(_))),
        span: imp.span,
    }
}

/// Lower an export specification.
fn lower_export(exp: &ast::Export) -> hir::Export {
    match exp {
        ast::Export::Var(ident, span) => hir::Export {
            name: ident.name,
            children: hir::ExportChildren::None,
            span: *span,
        },
        ast::Export::Type(ident, children, span) => hir::Export {
            name: ident.name,
            children: children.as_ref().map_or(hir::ExportChildren::None, |cs| {
                hir::ExportChildren::Some(cs.iter().map(|c| c.name).collect())
            }),
            span: *span,
        },
        ast::Export::Module(module_name, span) => {
            let name = module_name
                .parts
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(".");
            hir::Export {
                name: Symbol::intern(&name),
                children: hir::ExportChildren::All,
                span: *span,
            }
        }
    }
}

/// Lower a data declaration and generate field accessor functions for records.
fn lower_data_decl_with_accessors(
    ctx: &mut LowerContext,
    data: &ast::DataDecl,
) -> LowerResult<(hir::DataDef, Vec<hir::ValueDef>)> {
    let data_def = lower_data_decl(ctx, data)?;

    // Generate field accessor functions for record constructors
    let mut accessors = Vec::new();
    for con in &data_def.cons {
        if let hir::ConFields::Named(fields) = &con.fields {
            for (field_idx, field) in fields.iter().enumerate() {
                let accessor = generate_field_accessor(ctx, con, fields.len(), field_idx, field);
                accessors.push(accessor);
            }
        }
    }

    Ok((data_def, accessors))
}

/// Generate a field accessor function for a record field.
///
/// For `data Point = Point { px :: Int, py :: Int }`, generates:
/// - `px (Point x _) = x`
/// - `py (Point _ y) = y`
fn generate_field_accessor(
    ctx: &mut LowerContext,
    con: &hir::ConDef,
    num_fields: usize,
    field_idx: usize,
    field: &hir::FieldDef,
) -> hir::ValueDef {
    // Create a pattern for matching the constructor
    // e.g., Point x _ for accessing the first field
    let mut sub_pats = Vec::with_capacity(num_fields);
    let mut result_var_def_id = None;
    let result_var_name = Symbol::intern("_field_val");

    for i in 0..num_fields {
        if i == field_idx {
            // This is the field we're extracting
            let var_def_id = ctx.fresh_def_id();
            result_var_def_id = Some(var_def_id);
            ctx.define(var_def_id, result_var_name, DefKind::Value, field.span);
            sub_pats.push(hir::Pat::Var(result_var_name, var_def_id, field.span));
        } else {
            // Other fields are wildcards
            sub_pats.push(hir::Pat::Wild(field.span));
        }
    }

    let result_var_def_id = result_var_def_id.expect("field_idx should be valid");

    // Build the constructor pattern
    let con_pat = hir::Pat::Con(ctx.def_ref(con.id, con.span), sub_pats, field.span);

    // The RHS is just the variable we bound
    let rhs = hir::Expr::Var(ctx.def_ref(result_var_def_id, field.span));

    // Create the equation: accessor (Con ... x ...) = x
    let equation = hir::Equation {
        pats: vec![con_pat],
        guards: vec![],
        rhs,
        span: field.span,
    };

    hir::ValueDef {
        id: field.id,
        name: field.name,
        sig: None, // Type is inferred from the registered type
        equations: vec![equation],
        span: field.span,
    }
}

/// Infer the kind of a type parameter from how it's used in constructor field types.
/// Returns the number of arguments the parameter takes (0 = kind *, 1 = kind * -> *, etc.)
fn infer_param_arity(param_name: Symbol, tys: &[&ast::Type]) -> usize {
    let mut max_arity: usize = 0;
    for ty in tys {
        max_arity = max_arity.max(infer_param_arity_in_type(param_name, ty));
    }
    max_arity
}

/// Recursively find the maximum arity for a type parameter in a type expression.
fn infer_param_arity_in_type(param_name: Symbol, ty: &ast::Type) -> usize {
    match ty {
        ast::Type::Var(v, _) => {
            // A bare type variable has arity 0 (it appears as itself)
            if v.name.name == param_name {
                0
            } else {
                0
            }
        }
        ast::Type::App(f, a, _) => {
            // Check if this application chain has our parameter at the base
            // e.g., for `f a b`, if f is our param, it takes 2 args
            let chain_arity = count_app_chain_arity(param_name, ty);
            // Also check recursively in both parts for nested uses
            let f_recursive = infer_param_arity_in_type(param_name, f);
            let a_recursive = infer_param_arity_in_type(param_name, a);
            chain_arity.max(f_recursive).max(a_recursive)
        }
        ast::Type::Fun(from, to, _) => infer_param_arity_in_type(param_name, from)
            .max(infer_param_arity_in_type(param_name, to)),
        ast::Type::Tuple(tys, _) => tys
            .iter()
            .map(|t| infer_param_arity_in_type(param_name, t))
            .max()
            .unwrap_or(0),
        ast::Type::List(elem, _) => infer_param_arity_in_type(param_name, elem),
        ast::Type::Paren(inner, _) => infer_param_arity_in_type(param_name, inner),
        ast::Type::Forall(_, inner, _) => infer_param_arity_in_type(param_name, inner),
        // Type constructors, qualified constructors, constrained types, promoted lists,
        // type-level literals, etc. don't contain our type parameter in a way that matters.
        ast::Type::Con(_, _)
        | ast::Type::QualCon(_, _, _)
        | ast::Type::Constrained(_, _, _)
        | ast::Type::PromotedList(_, _)
        | ast::Type::NatLit(_, _)
        | ast::Type::Bang(_, _)
        | ast::Type::Lazy(_, _) => 0,
    }
}

/// Count how many arguments a type parameter takes in a chain of applications.
/// For example, in `f a b`, if f is the param, it takes 2 arguments.
fn count_app_chain_arity(param_name: Symbol, ty: &ast::Type) -> usize {
    // Walk up the application spine to count args
    fn count_args(param_name: Symbol, ty: &ast::Type, depth: usize) -> usize {
        match ty {
            ast::Type::Var(v, _) if v.name.name == param_name => depth,
            ast::Type::App(f, _, _) => count_args(param_name, f, depth + 1),
            ast::Type::Paren(inner, _) => count_args(param_name, inner, depth),
            _ => 0,
        }
    }
    count_args(param_name, ty, 0)
}

/// Build a kind from an arity (0 = *, 1 = * -> *, 2 = * -> * -> *, etc.)
fn kind_from_arity(arity: usize) -> bhc_types::Kind {
    let mut kind = bhc_types::Kind::Star;
    for _ in 0..arity {
        kind = bhc_types::Kind::Arrow(Box::new(bhc_types::Kind::Star), Box::new(kind));
    }
    kind
}

/// Collect all field types from constructor declarations.
fn collect_field_types(constrs: &[ast::ConDecl]) -> Vec<&ast::Type> {
    let mut types = Vec::new();
    for con in constrs {
        match &con.fields {
            ast::ConFields::Positional(tys) => {
                for ty in tys {
                    types.push(ty);
                }
            }
            ast::ConFields::Record(fields) => {
                for field in fields {
                    types.push(&field.ty);
                }
            }
        }
    }
    types
}

/// Lower a data declaration.
fn lower_data_decl(ctx: &mut LowerContext, data: &ast::DataDecl) -> LowerResult<hir::DataDef> {
    let type_def_id = ctx
        .lookup_type(data.name.name)
        .expect("type should be pre-bound");

    let is_gadt = !data.gadt_constrs.is_empty();

    // Collect all field types from constructors to infer parameter kinds
    let field_types = collect_field_types(&data.constrs);

    // Infer kinds for each type parameter based on usage
    let params: Vec<bhc_types::TyVar> = data
        .params
        .iter()
        .map(|p| {
            let arity = infer_param_arity(p.name.name, &field_types);
            let kind = kind_from_arity(arity);
            bhc_types::TyVar::new(p.name.name.as_u32(), kind)
        })
        .collect();

    let cons: Vec<hir::ConDef> = if is_gadt {
        data.gadt_constrs
            .iter()
            .map(|c| lower_gadt_con_def(ctx, c))
            .collect()
    } else {
        data.constrs.iter().map(|c| lower_con_def(ctx, c)).collect()
    };

    let deriving: Vec<Symbol> = data.deriving.iter().map(|c| c.name).collect();

    Ok(hir::DataDef {
        id: type_def_id,
        name: data.name.name,
        params,
        cons,
        is_gadt,
        deriving,
        span: data.span,
    })
}

/// Lower a constructor definition.
fn lower_con_def(ctx: &mut LowerContext, con: &ast::ConDecl) -> hir::ConDef {
    let con_def_id = ctx
        .lookup_constructor(con.name.name)
        .expect("constructor should be pre-bound");

    let fields = match &con.fields {
        ast::ConFields::Positional(tys) => {
            let hir_tys: Vec<bhc_types::Ty> = tys.iter().map(|t| lower_type(ctx, t)).collect();
            hir::ConFields::Positional(hir_tys)
        }
        ast::ConFields::Record(fields) => {
            let hir_fields: Vec<hir::FieldDef> = fields
                .iter()
                .map(|f| {
                    // Look up the DefId for the field accessor that was bound during
                    // collect_module_definitions
                    let field_def_id = ctx
                        .lookup_value(f.name.name)
                        .expect("field accessor should be pre-bound");
                    hir::FieldDef {
                        id: field_def_id,
                        name: f.name.name,
                        ty: lower_type(ctx, &f.ty),
                        span: f.span,
                    }
                })
                .collect();
            hir::ConFields::Named(hir_fields)
        }
    };

    hir::ConDef {
        id: con_def_id,
        name: con.name.name,
        fields,
        gadt_return_ty: None,
        span: con.span,
    }
}

/// Lower a GADT constructor definition.
///
/// Decomposes the full constructor type `A -> B -> ... -> RetType` into
/// field types `[A, B, ...]` and a return type `RetType`.
fn lower_gadt_con_def(ctx: &mut LowerContext, con: &ast::GadtConDecl) -> hir::ConDef {
    let con_def_id = ctx
        .lookup_constructor(con.name.name)
        .expect("GADT constructor should be pre-bound");

    // Decompose the constructor type: strip forall, then peel function arrows
    let full_ty = &con.ty;

    // Strip forall if present
    let inner_ty = match full_ty {
        ast::Type::Forall(_, inner, _) => inner.as_ref(),
        _ => full_ty,
    };

    // Walk the function arrow chain to separate argument types from return type
    let mut arg_types = Vec::new();
    let mut current = inner_ty;
    while let ast::Type::Fun(from, to, _) = current {
        arg_types.push(lower_type(ctx, from));
        current = to.as_ref();
    }
    let return_ty = lower_type(ctx, current);

    hir::ConDef {
        id: con_def_id,
        name: con.name.name,
        fields: hir::ConFields::Positional(arg_types),
        gadt_return_ty: Some(return_ty),
        span: con.span,
    }
}

/// Lower a newtype declaration.
/// Lower a newtype declaration and generate field accessor functions for records.
fn lower_newtype_decl_with_accessors(
    ctx: &mut LowerContext,
    newtype: &ast::NewtypeDecl,
) -> LowerResult<(hir::NewtypeDef, Vec<hir::ValueDef>)> {
    let type_def_id = ctx
        .lookup_type(newtype.name.name)
        .expect("type should be pre-bound");

    let params: Vec<bhc_types::TyVar> = newtype
        .params
        .iter()
        .map(|p| bhc_types::TyVar::new_star(p.name.name.as_u32()))
        .collect();

    let con = lower_con_def(ctx, &newtype.constr);

    let deriving: Vec<Symbol> = newtype.deriving.iter().map(|c| c.name).collect();

    // Generate field accessor functions for record constructors
    let mut accessors = Vec::new();
    if let hir::ConFields::Named(fields) = &con.fields {
        for (field_idx, field) in fields.iter().enumerate() {
            let accessor = generate_field_accessor(ctx, &con, fields.len(), field_idx, field);
            accessors.push(accessor);
        }
    }

    let newtype_def = hir::NewtypeDef {
        id: type_def_id,
        name: newtype.name.name,
        params,
        con,
        deriving,
        span: newtype.span,
    };

    Ok((newtype_def, accessors))
}

/// Lower a type alias declaration.
fn lower_type_alias(
    ctx: &mut LowerContext,
    type_alias: &ast::TypeAlias,
) -> LowerResult<hir::TypeAlias> {
    let def_id = ctx
        .lookup_type(type_alias.name.name)
        .expect("type should be pre-bound");

    let params: Vec<bhc_types::TyVar> = type_alias
        .params
        .iter()
        .map(|p| bhc_types::TyVar::new_star(p.name.name.as_u32()))
        .collect();

    let ty = lower_type(ctx, &type_alias.ty);

    Ok(hir::TypeAlias {
        id: def_id,
        name: type_alias.name.name,
        params,
        ty,
        span: type_alias.span,
    })
}

/// Lower an AST kind to a bhc_types kind.
fn lower_kind(kind: &ast::Kind) -> bhc_types::Kind {
    match kind {
        ast::Kind::Star => bhc_types::Kind::Star,
        ast::Kind::Arrow(left, right) => {
            bhc_types::Kind::Arrow(Box::new(lower_kind(left)), Box::new(lower_kind(right)))
        }
        ast::Kind::Var(_ident) => {
            // For now, treat kind variables as Star
            // A more sophisticated implementation would do kind inference
            bhc_types::Kind::Star
        }
    }
}

/// Lower an AST associated type declaration to HIR.
fn lower_assoc_type(ctx: &mut LowerContext, assoc: &ast::AssocType) -> hir::AssocTypeSig {
    let id = ctx.fresh_def_id();

    // Convert type parameters
    let params: Vec<bhc_types::TyVar> = assoc
        .params
        .iter()
        .map(|p| bhc_types::TyVar::new_star(p.name.name.as_u32()))
        .collect();

    // Convert kind, defaulting to Star
    let kind = assoc
        .kind
        .as_ref()
        .map(lower_kind)
        .unwrap_or(bhc_types::Kind::Star);

    // Convert optional default type
    let default = assoc.default.as_ref().map(|ty| lower_type(ctx, ty));

    hir::AssocTypeSig {
        id,
        name: assoc.name.name,
        params,
        kind,
        default,
        span: assoc.span,
    }
}

/// Lower an AST associated type definition to HIR.
fn lower_assoc_type_def(ctx: &mut LowerContext, def: &ast::AssocTypeDef) -> hir::AssocTypeImpl {
    let args: Vec<bhc_types::Ty> = def.args.iter().map(|ty| lower_type(ctx, ty)).collect();

    let rhs = lower_type(ctx, &def.rhs);

    hir::AssocTypeImpl {
        name: def.name.name,
        args,
        rhs,
        span: def.span,
    }
}

/// Lower a class declaration.
fn lower_class_decl(ctx: &mut LowerContext, class: &ast::ClassDecl) -> LowerResult<hir::ClassDef> {
    let def_id = ctx
        .lookup_type(class.name.name)
        .expect("class should be pre-bound");

    let params: Vec<bhc_types::TyVar> = class
        .params
        .iter()
        .map(|p| bhc_types::TyVar::new_star(p.name.name.as_u32()))
        .collect();

    // Build a map from param name to index for fundep conversion
    let param_indices: std::collections::HashMap<Symbol, usize> = class
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.name, i))
        .collect();

    // Convert AST fundeps to HIR fundeps (names -> indices)
    let fundeps: Vec<hir::FunDep> = class
        .fundeps
        .iter()
        .map(|fd| {
            let from: Vec<usize> = fd
                .from
                .iter()
                .filter_map(|ident| param_indices.get(&ident.name).copied())
                .collect();
            let to: Vec<usize> = fd
                .to
                .iter()
                .filter_map(|ident| param_indices.get(&ident.name).copied())
                .collect();
            hir::FunDep {
                from,
                to,
                span: fd.span,
            }
        })
        .collect();

    let supers: Vec<Symbol> = class.context.iter().map(|c| c.class.name).collect();

    // Extract method signatures
    let methods: Vec<hir::MethodSig> = class
        .methods
        .iter()
        .filter_map(|m| {
            if let ast::Decl::TypeSig(sig) = m {
                Some(
                    sig.names
                        .iter()
                        .map(|n| hir::MethodSig {
                            name: n.name,
                            id: ctx.lookup_value(n.name).unwrap_or_else(|| ctx.fresh_def_id()),
                            ty: lower_type_to_scheme(ctx, &sig.ty),
                            span: sig.span,
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            }
        })
        .flatten()
        .collect();

    // Pre-bind default method implementations before lowering them
    // (TypeSig names are already bound in collect_module_definitions, but
    // default implementations may have names not in TypeSig or may appear first)
    for method in &class.methods {
        if let ast::Decl::FunBind(fb) = method {
            let name = fb.name.name;
            if ctx.lookup_value(name).is_none() {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Value, fb.span);
                ctx.bind_value(name, def_id);
            }
        }
    }

    // Extract default implementations
    let defaults: Vec<hir::ValueDef> = class
        .methods
        .iter()
        .filter_map(|m| {
            if let ast::Decl::FunBind(fb) = m {
                lower_fun_bind(ctx, fb).ok()
            } else {
                None
            }
        })
        .collect();

    // Lower associated type declarations
    let assoc_types: Vec<hir::AssocTypeSig> = class
        .assoc_types
        .iter()
        .map(|at| lower_assoc_type(ctx, at))
        .collect();

    Ok(hir::ClassDef {
        id: def_id,
        name: class.name.name,
        params,
        fundeps,
        supers,
        methods,
        defaults,
        assoc_types,
        span: class.span,
    })
}

/// Lower an instance declaration.
/// Flatten an AST type application spine into individual types.
///
/// For multi-param type classes, the parser produces a single type like
/// `App(Con("Int"), Con("String"))` for `instance C Int String`.
/// This function decomposes the spine into `[Int, String]` based on the
/// expected parameter count.
///
/// For single-param classes (param_count=1), returns the type as-is in a vec.
fn flatten_instance_type(ty: &ast::Type, param_count: usize) -> Vec<&ast::Type> {
    if param_count <= 1 {
        return vec![ty];
    }

    // Walk the App spine leftward to collect all types
    let mut spine = Vec::new();
    let mut current = ty;
    loop {
        match current {
            ast::Type::App(f, x, _) => {
                spine.push(x.as_ref());
                current = f.as_ref();
            }
            _ => {
                spine.push(current);
                break;
            }
        }
    }
    // spine is in reverse order (rightmost arg first), so reverse it
    spine.reverse();

    // If we have more types than params, take only the last param_count
    // (the excess would be a partially applied type constructor)
    if spine.len() > param_count {
        spine.split_off(spine.len() - param_count)
    } else {
        spine
    }
}

fn lower_instance_decl(
    ctx: &mut LowerContext,
    instance: &ast::InstanceDecl,
) -> LowerResult<hir::InstanceDef> {
    // Decompose the instance type based on class parameter count.
    // For multi-param classes like `instance Convertible Int String`,
    // the parser produces a single type App(Con("Int"), Con("String"))
    // which we flatten into [Int, String].
    let param_count = ctx
        .lookup_class_param_count(instance.class.name)
        .unwrap_or(1);
    let flattened = flatten_instance_type(&instance.ty, param_count);
    let types: Vec<bhc_types::Ty> = flattened.iter().map(|t| lower_type(ctx, t)).collect();

    let constraints: Vec<bhc_types::Constraint> = instance
        .context
        .iter()
        .map(|c| {
            let args: Vec<bhc_types::Ty> = c.args.iter().map(|a| lower_type(ctx, a)).collect();
            bhc_types::Constraint {
                class: c.class.name,
                args,
                span: c.span,
            }
        })
        .collect();

    // Lower instance methods with FRESH DefIds. We do NOT bind the method name
    // to the fresh DefId — this ensures that references to `show` in the method
    // body resolve to the builtin polymorphic `show`, not the instance method
    // itself (which would cause infinite recursion).
    let methods: Vec<hir::ValueDef> = instance
        .methods
        .iter()
        .filter_map(|m| {
            if let ast::Decl::FunBind(fb) = m {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, fb.name.name, DefKind::Value, fb.span);
                lower_instance_method(ctx, fb, def_id).ok()
            } else {
                None
            }
        })
        .collect();

    // Lower associated type definitions
    let assoc_type_impls: Vec<hir::AssocTypeImpl> = instance
        .assoc_type_defs
        .iter()
        .map(|def| lower_assoc_type_def(ctx, def))
        .collect();

    Ok(hir::InstanceDef {
        class: instance.class.name,
        types,
        constraints,
        methods,
        assoc_type_impls,
        span: instance.span,
    })
}

/// Lower a fixity declaration.
fn lower_fixity_decl(fixity: &ast::FixityDecl) -> hir::FixityDecl {
    let hir_fixity = match fixity.fixity {
        ast::Fixity::Left => hir::Fixity::Left,
        ast::Fixity::Right => hir::Fixity::Right,
        ast::Fixity::None => hir::Fixity::None,
    };

    hir::FixityDecl {
        fixity: hir_fixity,
        precedence: fixity.prec,
        ops: fixity.ops.iter().map(|o| o.name).collect(),
        span: fixity.span,
    }
}

/// Lower a foreign declaration.
fn lower_foreign_decl(
    ctx: &mut LowerContext,
    foreign: &ast::ForeignDecl,
) -> LowerResult<hir::ForeignDecl> {
    let def_id = ctx
        .lookup_value(foreign.name.name)
        .expect("foreign import should be pre-bound");

    // Map convention string to ForeignConvention
    let convention = match foreign.convention.as_str() {
        "ccall" | "capi" => hir::ForeignConvention::CCall,
        "stdcall" => hir::ForeignConvention::StdCall,
        "javascript" => hir::ForeignConvention::JavaScript,
        _ => hir::ForeignConvention::CCall, // Default
    };

    let ty = bhc_types::Scheme::mono(lower_type(ctx, &foreign.ty));

    Ok(hir::ForeignDecl {
        id: def_id,
        name: foreign.name.name,
        foreign_name: foreign
            .external_name
            .as_ref()
            .map_or_else(|| foreign.name.name, |s| Symbol::intern(s)),
        convention,
        ty,
        span: foreign.span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_simple_var() {
        let mut ctx = LowerContext::with_builtins();

        // Create a variable reference to a builtin
        let ident = bhc_intern::Ident::from_str("map");
        let expr = ast::Expr::Var(ident, Span::default());

        let result = lower_expr(&mut ctx, &expr);

        assert!(matches!(result, hir::Expr::Var(_)));
        assert!(!ctx.has_errors());
    }

    #[test]
    fn test_lower_literal() {
        let mut ctx = LowerContext::with_builtins();

        let expr = ast::Expr::Lit(ast::Lit::Int(42), Span::default());

        let result = lower_expr(&mut ctx, &expr);

        match result {
            hir::Expr::Lit(hir::Lit::Int(n), _) => assert_eq!(n, 42),
            _ => panic!("expected integer literal"),
        }
    }

    #[test]
    fn test_lower_application() {
        let mut ctx = LowerContext::with_builtins();

        // map id
        let map_ident = bhc_intern::Ident::from_str("map");
        let id_ident = bhc_intern::Ident::from_str("id");

        let expr = ast::Expr::App(
            Box::new(ast::Expr::Var(map_ident, Span::default())),
            Box::new(ast::Expr::Var(id_ident, Span::default())),
            Span::default(),
        );

        let result = lower_expr(&mut ctx, &expr);

        assert!(matches!(result, hir::Expr::App(_, _, _)));
    }
}
