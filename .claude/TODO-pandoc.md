# Road to Pandoc: BHC Compilation TODO

**Document ID:** BHC-TODO-PANDOC
**Status:** In Progress
**Created:** 2026-01-30
**Updated:** 2026-02-21

---

## Goal

Compile and run [Pandoc](https://github.com/jgm/pandoc), a ~60k LOC Haskell
document converter with ~80 transitive package dependencies. This serves as the
north-star integration target for BHC's real-world Haskell compatibility.

---

## Current State

BHC compiles real Haskell programs to native executables via LLVM:
- **163 native E2E tests** registered (including monad transformers, file IO, markdown parser, JSON parser, GADTs, type extensions)
- All intermediate milestones A–E.64 done

### Standard Library & IO (E.5–E.31)
- Monad transformers: StateT, ReaderT, ExceptT, WriterT all working
- Nested transformer stacks: all cross-transformer combinations working (E.55–E.57)
- MTL typeclasses: MonadReader, MonadState, MonadError, MonadWriter
- Exception handling: catch, bracket, finally, onException (E.5)
- Multi-package support with import paths (E.6)
- Data.Text: packed UTF-8 with 25+ operations (E.7)
- Data.ByteString: 24 RTS functions, Data.Text.Encoding bridge (E.8)
- Data.Char predicates + Char Enum ranges, first-class predicates (E.9, E.36, E.37)
- Data.Text.IO: native Text file/handle I/O (E.10)
- Show for compound/nested types via recursive ShowTypeDesc (E.11, E.31)
- Numeric ops: even/odd, gcd/lcm, divMod/quotRem, fromIntegral + IORef (E.12)
- Data.Maybe, Data.Either, Control.Monad combinators (E.13, E.14, E.18)
- Extensive Data.List: 70+ operations (E.15, E.16, E.26)
- Ordering ADT (LT/EQ/GT), compare returning Ordering (E.17)
- System.FilePath + System.Directory (E.19)
- Data.Map/Set/IntMap/IntSet: full operation sets (E.21, E.22, E.29)
- Stock deriving: Eq, Show, Ord for user-defined ADTs (E.23, E.24)
- Arithmetic, Enum, Folds, Higher-order, IO Input builtins (E.25–E.30)

### Language Extensions (E.32–E.64)
- OverloadedStrings + IsString typeclass (E.32)
- Record syntax: named fields, accessors, construction, update, RecordWildCards, NamedFieldPuns (E.33)
- ViewPatterns codegen with fallthrough (E.34)
- TupleSections + MultiWayIf (E.35)
- Manual typeclass instances with Show dispatch (E.38)
- User-defined typeclasses: dictionary-passing, higher-kinded, default methods, superclasses (E.39–E.41)
- DeriveAnyClass for user-defined typeclasses (E.42)
- Word types (Word8/16/32/64), Integer arbitrary precision, lazy let-bindings (E.43–E.45)
- ScopedTypeVariables (E.46)
- GeneralizedNewtypeDeriving with newtype erasure (E.47)
- FlexibleInstances, FlexibleContexts, instance context propagation (E.48)
- MultiParamTypeClasses (E.49)
- FunctionalDependencies (E.50)
- DeriveFunctor, DeriveFoldable, DeriveTraversable (E.51–E.53)
- DeriveEnum + DeriveBounded (E.54)
- Cross-transformer codegen: ReaderT/StateT, ExceptT/StateT+ReaderT, WriterT/StateT+ReaderT (E.55–E.57)
- Full lazy let-bindings for Haskell semantics (E.58)
- EmptyDataDecls + strict field annotations (E.59)
- GADTs with type refinement (E.60)
- TypeOperators for infix type syntax (E.61)
- StandaloneDeriving + PatternSynonyms + nested pattern fallthrough (E.62)
- DeriveGeneric + NFData/DeepSeq stubs for Pandoc compatibility (E.63)
- EmptyCase, StrictData, DefaultSignatures, OverloadedLists (E.64)

### Gap to Pandoc

**Completed (previously missing, now done):**
1. ~~OverloadedStrings + IsString~~ — Done (E.32)
2. ~~Record syntax~~ — Done (E.33): named fields, accessors, construction, update, RecordWildCards
3. ~~ViewPatterns codegen~~ — Done (E.34)
4. ~~TupleSections + MultiWayIf~~ — Done (E.35)
5. ~~GeneralizedNewtypeDeriving~~ — Done (E.47): newtype erasure lifting instances
6. ~~GHC.Generics~~ — Partial (E.63): DeriveGeneric stubs for Pandoc compatibility

**Still missing for Pandoc (prioritized):**
1. **Full package system** — Basic import paths work (E.6), but no Hackage .cabal parsing yet
2. **Lazy Text/ByteString** — Only strict variants implemented
3. **Template Haskell** — Required for aeson JSON deriving (alternative: full GHC.Generics)
4. **CPP preprocessing** — Pandoc and many deps use `#ifdef` for platform/version conditionals
5. **Exception hierarchy** — `Exception` typeclass with `toException`/`fromException`
6. **Full GHC.Generics** — E.63 added stubs; full `Rep` type family + `from`/`to` still needed
7. **parsec/megaparsec** — Pandoc depends on parsec for some formats
8. **aeson** — JSON serialization with ToJSON/FromJSON (requires TH or full Generics)
9. **Data.Sequence** — Finger tree (not started)
10. **process/time/network-uri** — External dependency packages

---

## Tier 1 — Showstoppers

These must be resolved before any real-world Haskell program can compile.

### 1.1 Package System Integration

**Status:** Basic import paths working (E.6), full Hackage integration not yet connected
**Scope:** Large

Multi-package support with `-I` import paths is working (E.6). The `bhc-package`
crate exists with TOML manifests, semver resolution, and lockfile support.
Pandoc depends on ~80 packages from Hackage.

- [x] Wire package resolution into `bhc-driver` compilation pipeline (basic import paths)
- [ ] Parse `.cabal` files (at minimum: exposed-modules, build-depends, hs-source-dirs)
- [ ] Resolve transitive dependency graph from a cabal file
- [ ] Fetch packages from Hackage (tar.gz download + unpack)
- [ ] Support `PackageImports` extension for disambiguating modules
- [ ] Handle conditional dependencies (flags, OS checks, impl checks)
- [ ] Generate and consume interface files (`.bhi`) across package boundaries
- [ ] Cache compiled packages to avoid recompilation

**Key files:**
- `crates/bhc-package/` — existing package infrastructure
- `crates/bhc-driver/` — compilation orchestration
- `crates/bhc-interface/` — module interface files

### 1.2 Data.Text and Data.ByteString

**Status:** ✅ Core APIs complete (E.7 + E.8), Text.IO complete (E.10), Lazy variants remaining
**Scope:** Medium (remaining: Lazy variants)

Data.Text (E.7): packed UTF-8 with 25+ operations. Data.ByteString (E.8): 24
RTS functions with identical memory layout. Data.Text.Encoding (E.8): zero-copy
encodeUtf8/decodeUtf8 bridge.

- [x] Implement packed UTF-8 `Text` representation (not `[Char]`)
- [x] Core Text API: pack, unpack, append, cons, snoc, head, tail, length,
      null, map, take, drop, toLower, toUpper, toCaseFold, toTitle,
      isPrefixOf, isSuffixOf, isInfixOf, eq, compare, singleton, empty,
      filter, foldl', concat, intercalate, strip, words, lines, splitOn, replace
- [x] Text.IO: readFile, writeFile, appendFile, hGetContents, hGetLine, hPutStr, hPutStrLn, putStr, putStrLn, getLine, getContents
- [x] Text.Encoding: encodeUtf8, decodeUtf8
- [ ] Text.Encoding: decodeUtf8' (with Either error handling)
- [ ] Lazy Text variant (Data.Text.Lazy, Data.Text.Lazy.IO)
- [x] ByteString: packed byte array type (identical layout to Text)
- [x] ByteString API (24 functions): pack, unpack, empty, singleton, append,
      cons, snoc, head, last, tail, init, length, null, take, drop, reverse,
      elem, index, eq, compare, isPrefixOf, isSuffixOf, readFile, writeFile
- [ ] ByteString.Lazy and ByteString.Builder
- [ ] SIMD-optimized operations where applicable (memchr, memcmp, etc.)

**Key files:**
- `stdlib/bhc-text/src/text.rs` — Text RTS (25+ FFI functions)
- `stdlib/bhc-text/src/bytestring.rs` — ByteString RTS (24 FFI functions)
- `crates/bhc-typeck/src/builtins.rs` — type registrations
- `crates/bhc-codegen/src/llvm/lower.rs` — VarIds 1000200-1000431

### 1.3 Full IO and Exception Handling

**Status:** ✅ Core exception handling complete (E.5), file IO working, directory ops complete (E.19)
**Scope:** Medium

Exception handling (catch, bracket, finally, onException) is working (E.5).
File IO (readFile, writeFile, openFile, hClose) is working. System ops
(getArgs, getEnv, exitWith) are working. Directory operations (E.19) complete.

- [x] Handle abstraction: `Handle`, `IOMode`
- [x] File operations: `openFile`, `hClose`, `hFlush`
- [x] Reading: `hGetLine`, `hGetContents`, `hIsEOF`
- [x] Writing: `hPutStr`, `hPutStrLn`
- [x] Standard handles: `stdin`, `stdout`, `stderr`
- [x] File-level: `readFile`, `writeFile`, `appendFile`
- [x] Exception types: `SomeException`, `IOException`, `ErrorCall`
- [x] Exception primitives: `throw`, `throwIO`, `catch`, `try`
- [x] Resource management: `bracket`, `bracket_`, `finally`, `onException`
- [ ] Exception hierarchy: `Exception` typeclass with `toException`/`fromException`
- [ ] Asynchronous exceptions: `mask`, `uninterruptibleMask` (at least stubs)
- [x] System operations: `getArgs`, `getProgName`, `getEnv`, `lookupEnv`
- [x] Exit: `exitSuccess`, `exitFailure`, `exitWith`
- [x] Directory: `doesFileExist`, `doesDirectoryExist`, `createDirectory`,
      `removeFile`, `removeDirectory`, `getCurrentDirectory`, `setCurrentDirectory`,
      `renameFile`, `copyFile`, `listDirectory` (E.19)
- [ ] Temporary files: `withTempFile`, `withTempDirectory`

**Key files:**
- `stdlib/bhc-system/` — system/IO crate
- `rts/bhc-rts/` — runtime entry points
- `crates/bhc-codegen/src/llvm/lower.rs` — codegen handlers

### 1.4 Template Haskell

**Status:** Syntax parsed, no evaluation
**Scope:** Large

Pandoc's dependencies (aeson, lens, generic-deriving) use TH for deriving
instances, generating boilerplate, and compile-time code generation.

- [ ] TH expression AST (`Language.Haskell.TH.Syntax`)
- [ ] Quotation brackets: `[| expr |]`, `[d| decls |]`, `[t| type |]`, `[p| pat |]`
- [ ] Splice evaluation: `$(expr)` at compile time
- [ ] Name lookup: `'name` and `''TypeName` quotes
- [ ] Reification: `reify`, `reifyInstances`, `reifyType`
- [ ] TH monad: `Q` monad with fresh name generation, module info, etc.
- [ ] Cross-stage persistence for spliced values
- [ ] `DeriveLift` and `Lift` class

**Alternative approach:** Instead of full TH, implement `deriving via`
`Generic` with a generics-sop or GHC.Generics infrastructure that covers
the most common TH use cases (JSON instances, Show, Eq, Ord). This would
unblock aeson/yaml without full TH.

---

## Tier 2 — Major Gaps

Required for Pandoc but solvable without architectural changes.

### 2.1 GADT and Type Family Completion

**Status:** ✅ GADTs working (E.60), type families partially type-checked
**Scope:** Medium (remaining: type families)

- [x] GADT type checking: refine types in branches based on constructor (E.60)
- [ ] Type family reduction during type checking
- [ ] Closed type families with overlapping equations
- [ ] Data families
- [ ] Associated type defaults
- [ ] Kind inference improvements (currently requires manual signatures)

### 2.2 Multi-Module Compilation

**Status:** Core workflow complete, persistence deferred
**Scope:** Medium

- [x] Compile multiple modules in dependency order (BFS discovery + Kahn's toposort)
- [x] Cross-module type info via ModuleRegistry (types flow between modules)
- [x] Separate compilation: each module to `.o`, link at end
- [x] Module-qualified symbol mangling (no link-time collisions)
- [ ] Generate and read `.bhi` interface files (infrastructure exists in bhc-interface, not wired)
- [ ] Handle mutual module recursion (`.hs-boot` files)
- [ ] Incremental recompilation (check timestamps / hashes)

### 2.3 Missing Standard Libraries

Each of these is a Pandoc dependency that must exist in BHC's stdlib or be
compiled from Hackage source.

#### containers (Data.Map, Data.Set, Data.Sequence, Data.IntMap, Data.IntSet)
- [x] Data.Map — RTS-backed BTreeMap (basic ops + WithKey variants done)
- [x] Data.Set — RTS-backed BTreeSet (full type support + unions/partition, E.22)
- [x] Data.IntMap — shares Map RTS (full type support, E.22)
- [x] Data.IntSet — shares Set RTS (full type support + filter/foldr, E.22)
- [ ] Data.Sequence — finger tree (not started)
- [x] Data.Map.update, Data.Map.alter, Data.Map.unions, Data.Map.keysSet (E.21)
- [ ] Data.Graph, Data.Tree (used by some Pandoc deps)

#### mtl / transformers
- [x] `runReaderT`, `runStateT`, `runExceptT`, `runWriterT` — all working
- [x] `ask`, `local`, `get`, `put`, `modify`, `throwError`, `catchError` — all working
- [x] `lift`, `liftIO` — working for single-layer transformers
- [x] MonadReader, MonadState, MonadError, MonadWriter classes — registered in type system
- [x] Codegen for nested transformer stacks: `StateT s (ReaderT r IO)` working
- [x] Codegen for nested transformer stacks: `ReaderT r (StateT s IO)` working (E.55)
- [x] ExceptT cross-transformer: ExceptT over StateT, ExceptT over ReaderT (E.56)
- [x] WriterT cross-transformer: WriterT over StateT, WriterT over ReaderT (E.57)

#### parsec / megaparsec
- [ ] Pandoc has its own parsers but depends on parsec for some formats
- [ ] Either port parsec source or implement a compatible API
- [ ] `ParsecT` monad transformer, combinators (`many`, `try`, `<|>`, etc.)

#### aeson (JSON)
- [ ] JSON value type (`Value`: Object, Array, String, Number, Bool, Null)
- [ ] `encode`, `decode`, `eitherDecode`
- [ ] `ToJSON` / `FromJSON` type classes
- [ ] Generic deriving for JSON instances (requires TH or GHC.Generics)

#### yaml
- [ ] YAML parsing (wraps libyaml via FFI or pure Haskell)
- [ ] `decodeFileEither`, `encodeFile`
- [ ] Pandoc uses YAML for document metadata

#### Other dependencies
- [x] `filepath` — file path manipulation (`</>`, `takeExtension`, etc.) (E.19)
- [x] `directory` — filesystem operations (E.19)
- [ ] `process` — spawn subprocesses
- [ ] `time` — date/time types
- [ ] `network-uri` — URI parsing
- [ ] `http-client` — HTTP requests (optional, for URL fetching)
- [ ] `skylighting` — syntax highlighting (large dep)
- [ ] `doctemplates` — Pandoc's template system
- [ ] `texmath` — TeX math parsing
- [ ] `xml-conduit` or `xml` — XML parsing
- [ ] `zip-archive` — ZIP file handling (for EPUB, DOCX)

### 2.4 Deriving Infrastructure

**Status:** ✅ Extensive — 8 stock derivable classes + DeriveAnyClass + GND + DeriveGeneric stubs
**Scope:** Small (remaining: full GHC.Generics, DerivingVia, Read, Ix)

- [x] `GHC.Generics` — DeriveGeneric stubs for Pandoc compatibility (E.63)
- [ ] Generic representations: `V1`, `U1`, `K1`, `M1`, `:+:`, `:*:` (full Rep type family)
- [ ] `from` / `to` methods for converting to/from generic rep
- [x] Derive `Generic` for user-defined types (E.63, stub — needs full Rep)
- [x] Stock deriving: `Eq`, `Show` for simple enums and ADTs with fields (E.23)
- [x] Stock deriving: `Ord` for simple enums and ADTs with fields (E.24)
- [x] Stock deriving: `Enum`, `Bounded` for enums (E.54)
- [x] Stock deriving: `Functor` (E.51)
- [x] Stock deriving: `Foldable` (E.52)
- [x] Stock deriving: `Traversable` (E.53)
- [ ] Stock deriving: `Read`, `Ix`
- [ ] `DerivingStrategies`: stock, newtype, anyclass, via
- [x] `DeriveAnyClass` for type classes with default method implementations (E.42)
- [ ] `DerivingVia` for newtype-based instance delegation
- [x] `GeneralizedNewtypeDeriving` for lifting instances through newtypes (E.47)
- [x] `StandaloneDeriving` (E.62)

---

## Tier 3 — Solvable with Current Architecture

### 3.1 Remaining Codegen Builtins

**Status:** ~500+ of 587 builtins lowered (E.13–E.31 added ~90+ functions + derived dispatches)
**Scope:** Small-Medium (ongoing)

- [ ] Monadic codegen: general `>>=`, `>>`, `return` via dictionary dispatch
- [x] `mapM_` (E.14), `when`, `unless` (E.14), `guard` (E.13)
- [x] `mapM`, `forM`, `forM_`, `sequence`, `sequence_`, `void`
- [x] `filterM`, `foldM`, `foldM_`, `replicateM`, `replicateM_`, `zipWithM`, `zipWithM_` (E.18)
- [x] `foldMap` (delegates to concatMap, E.16)
- [ ] Foldable/Traversable: `traverse`, `sequenceA`, `toList`
- [x] Data.Maybe: fromMaybe, maybe, listToMaybe, maybeToList, catMaybes, mapMaybe (E.13)
- [x] Data.Either: either, fromLeft, fromRight, lefts, rights, partitionEithers (E.13)
- [x] Data.List: any, all (E.14), scanr, scanl1, scanr1, unfoldr, intersect, zip3, zipWith3 (E.15)
- [x] Data.List: iterate, repeat, cycle (take-fused, E.15)
- [x] Data.List: elemIndex, findIndex, isPrefixOf, isSuffixOf, isInfixOf, tails, inits (E.16)
- [x] maximumBy, minimumBy (E.16), compare returns Ordering ADT (E.17)
- [x] Fixed stubs: maximum, minimum, and, or, Data.Map.notMember (E.16)
- [x] Data.Map.update, Data.Map.alter, Data.Map.unions, Data.Map.keysSet (E.21)
- [x] Data.Set.unions, Data.Set.partition (E.22)
- [x] Data.IntSet.filter, Data.IntSet.foldr (reuse Set implementations, E.22)
- [x] Full typeck/context.rs type entries for Data.Set (30), Data.IntMap (25), Data.IntSet (15) (E.22)
- [x] Fixed VarId suffix bug in Set/IntSet binary/predicate/extremum dispatches (E.22)
- [x] `show` dispatch for Bool-returning container builtins (Data.Map.member/null, Data.Set.member/null, etc.) (E.21)
- [x] `compare` returns Ordering ADT (LT/EQ/GT) with proper show support (E.17)

### 3.2 Numeric and Conversion Operations

- [x] `show` for standard types: showInt, showBool, showChar, showFloat (type-specialized, E.9)
- [x] `show` for compound types: String, [a], Maybe, Either, (a,b), () (E.11)
- [x] `show` for Double/Float literals (E.29)
- [x] `show` for nested compound types via recursive ShowTypeDesc (E.31)
- [x] `read` for Int (RTS bhc_read_int, E.25)
- [x] `readMaybe` for Int (RTS bhc_try_read_int, E.25)
- [x] `fromString` (identity, E.25)
- [ ] `reads` for parsing (general)
- [x] `fromIntegral`, `toInteger`, `fromInteger` (identity pass-through, E.12)
- [x] `even`, `odd` (inline LLVM srem, E.12)
- [x] `gcd`, `lcm` (RTS functions, E.12)
- [x] `divMod`, `quotRem` (floor-division / truncation, returns tuple, E.12)
- [x] `IORef`: newIORef, readIORef, writeIORef, modifyIORef (E.12)
- [ ] `realToFrac`
- [ ] `Rational` type and operations
- [x] `Data.Char` predicates: isAlpha, isDigit, isUpper, isLower, isAlphaNum, isSpace, isPunctuation, toUpper, toLower, ord, chr, digitToInt, intToDigit (E.9)
- [ ] `Data.Char` full Unicode categories (currently ASCII-only)

### 3.3 Performance (Core IR Optimization Pipeline)

**Status:** Not started — BHC currently has NO general-purpose optimizer
**Scope:** Large (foundational infrastructure)
**Reference:** `rules/013-optimization.md`, HBC/HCT simplifier architecture

BHC compiles to correct code but performs no Core IR optimization. Every
binding, beta-redex, and known-constructor case dispatch is passed unoptimized
to LLVM, which cannot reason about ADTs, closures, or thunks. Real Haskell
programs (pandoc) will exhibit thunk buildup, redundant allocation, and
unexploited compile-time information without these passes.

#### Phase O.1: Core Simplifier (CRITICAL — prerequisite for everything else)
- [ ] Beta reduction: `(\x -> body) arg` → `body[x := arg]`
- [ ] Case-of-known-constructor: `case Just 42 of { Just x -> x }` → `42`
- [ ] Dead binding elimination: remove unused let-bindings
- [ ] Constant folding: `1 + 2` → `3` for literals
- [ ] Inlining: substitute small/single-use bindings (reference-counting heuristic)
- [ ] Iterate to fixpoint (cap at 10 iterations)
- [ ] `-ddump-core-after-simpl` dump flag

#### Phase O.2: Pattern Match Compilation (HIGH — correctness + quality)
- [ ] Replace equation-by-equation compilation with Augustsson decision trees
- [ ] Exhaustiveness checking with non-exhaustive pattern warnings
- [ ] Overlap/redundancy detection with shadowed pattern warnings
- [ ] Guard compilation via nested case fallthrough

#### Phase O.3: Demand Analysis + Worker/Wrapper (MEDIUM — Default profile perf)
- [ ] Boolean-tree demand analysis for strictness signatures
- [ ] Fixpoint iteration for recursive binding groups
- [ ] Annotate strict arguments
- [ ] Worker/wrapper split for strict-arg functions (unboxed workers)
- [ ] `-ddump-core-after-demand` dump flag

#### Phase O.4: Dictionary Specialization (MEDIUM — typeclass perf)
- [ ] Direct method selection when dictionary is known constructor
- [ ] Monomorphize polymorphic functions at concrete call sites
- [ ] SPECIALIZE pragma support
- [ ] Second simplifier round to clean up after specialization

#### Key Files (to create)
```
crates/bhc-core/src/
├── simplify.rs              # Core simplifier
├── simplify/
│   ├── beta.rs              # Beta reduction
│   ├── case.rs              # Case transformations
│   ├── dead.rs              # Dead binding elimination
│   ├── fold.rs              # Constant folding
│   └── inline.rs            # Inlining decisions
├── demand.rs                # Demand analysis
├── worker_wrapper.rs        # Worker/wrapper transformation
└── specialize.rs            # Dictionary specialization
```

---

## Intermediate Milestones

Rather than jumping straight to Pandoc, build toward it incrementally:

### Milestone A: Multi-Module Program ✅
- [x] Compile a 3-file Haskell program with imports between modules
- [x] Verify type checking works across module boundaries
- [x] Verify codegen produces correct linked executable

### Milestone B: File Processing Utility ✅
- [x] Compile a program that reads a file, transforms content, writes output
- [x] Requires: File IO, String operations, basic error handling
- [x] Example: word count, line reversal, simple grep

### Milestone C: Simple Markdown Parser ✅
- [x] Compile a ~500 LOC Markdown-to-HTML converter
- [x] Requires: Text processing, Data.Map for link references, File IO
- [x] No external dependencies — self-contained

### Milestone D: StateT-Based Parser ✅
- [x] Compile a program using StateT for structured input parsing
- [x] Requires: Monad transformers working with String state
- [x] Example: CSV parser using `StateT String IO`
- [x] E2E test: `tier3_io/milestone_d_csv_parser` passes

### Milestone E: JSON/YAML Processing ✅
- [x] Compile a program that parses JSON, extracts fields, writes output
- [x] Self-contained JSON parser without external dependencies
- [x] E2E test: `tier3_io/milestone_e_json` passes (outputs "Alice" and "30" from `{"name": "Alice", "age": 30}`)

### Milestone E.5: Exception Handling ✅
- [x] Implement `throw`, `catch`, `try` for IO exceptions
- [x] Implement `bracket`, `finally`, `onException` for resource management
- [x] Exception hierarchy: `SomeException`, `IOException`, `ErrorCall`
- [x] E2E tests: `bracket_io`, `catch_file_error`, `exception_test`, `handle_io`

### Milestone E.6: Multi-Package Program ✅
- [x] Wire import paths into `bhc-driver` via `-I` flag
- [x] Compile programs that import from external package directories
- [x] E2E test: `tier3_io/package_import` passes

### Milestone E.7: Data.Text Foundation ✅
- [x] Implement packed UTF-8 `Text` type (not `[Char]`)
- [x] Core API: pack, unpack, append, length, null, take, drop, toUpper, toLower
- [x] RTS-backed implementation in bhc-text with UTF-8 encoding
- [x] E2E test: `tier3_io/text_basic` passes (pack, unpack, append, toUpper, take, drop)

### Milestone E.8: Data.ByteString + Text Completion ✅
- [x] ByteString RTS: 24 FFI functions with same memory layout as Text
- [x] ByteString type system: `bytestring_con`/`bytestring_ty` + 23 PrimOps
- [x] ByteString codegen: VarIds 1000400-1000423
- [x] Text.Encoding: `encodeUtf8` (zero-copy), `decodeUtf8` (validates UTF-8)
- [x] Additional Text ops: filter, foldl', concat, intercalate, strip, words, lines, splitOn, replace
- [x] E2E tests: `tier3_io/bytestring_basic` and `tier3_io/text_encoding` pass
- [x] 43 total E2E tests pass, 66 bhc-text unit tests pass

### Milestone E.9: Data.Char + Type-Specialized Show ✅
- [x] Data.Char predicates: isAlpha, isDigit, isUpper, isLower, isAlphaNum, isSpace, isPunctuation
- [x] Data.Char conversions: toUpper, toLower, ord, chr, digitToInt, intToDigit
- [x] Type-specialized show: showInt, showBool, showChar, showFloat
- [x] Proper ADT boolean returns from char predicates (tag 0=False, 1=True)
- [x] bhc-base linked, VarId bug fixes
- [x] E2E tests: `tier3_io/char_predicates`, `tier3_io/show_types`
- [x] 45 total E2E tests pass

### Milestone E.10: Data.Text.IO ✅
- [x] 7 RTS functions in bhc-text: readFile, writeFile, appendFile, hGetContents, hGetLine, hPutStr, hPutStrLn
- [x] 4 codegen-composed convenience functions: putStr, putStrLn, getLine, getContents
- [x] Handle functions use sentinel-pointer pattern (1=stdin, 2=stdout, 3=stderr)
- [x] Fixed import shadowing bug in register_standard_module_exports
- [x] E2E test: `tier3_io/text_io`
- [x] 46 total E2E tests pass

### Milestone E.11: Show Compound Types ✅
- [x] 6 RTS show functions: bhc_show_string, bhc_show_list, bhc_show_maybe, bhc_show_either, bhc_show_tuple2, bhc_show_unit
- [x] Expression-based type inference (infer_show_from_expr) since expr.ty() returns Error in Core IR
- [x] ShowCoerce extended: StringList, List, MaybeOf, EitherOf, Tuple2Of, Unit
- [x] RTS type tags: 0=Int, 1=Double, 2=Float, 3=Bool, 4=Char, 5=String for element formatting
- [x] VarIds 1000092-1000097, DefIds 10105-10110
- [x] E2E tests: show_string, show_list, show_maybe, show_either, show_tuple, show_unit
- [x] 52 total E2E tests pass

### Milestone E.12: Numeric Conversions + IORef ✅
- [x] `fromIntegral`/`toInteger`/`fromInteger` as identity pass-through
- [x] `even`/`odd` via inline LLVM srem, returning proper Bool ADT
- [x] `gcd`/`lcm` via RTS functions (VarIds 1000500-1000501)
- [x] `divMod` with floor-division semantics (sign adjustment for negative dividends)
- [x] `quotRem` with truncation-toward-zero (LLVM sdiv/srem)
- [x] IORef: newIORef, readIORef, writeIORef, modifyIORef (VarIds 1000502-1000504, DefIds 10400-10404)
- [x] Fixed DefIds 10500-10507 for numeric ops (bypasses 30-entry sequential array misalignment)
- [x] Show inference for Bool/Int-returning functions (expr_returns_bool, expr_returns_int)
- [x] E2E tests: numeric_ops, divmod, ioref_basic
- [x] 55 total E2E tests pass

### Milestone E.13: Data.Maybe + Data.Either + guard ✅
- [x] Data.Maybe: fromMaybe, maybe, listToMaybe, maybeToList, catMaybes, mapMaybe
- [x] Data.Either: either, fromLeft, fromRight, lefts, rights, partitionEithers
- [x] Control.Monad: guard
- [x] 13 pure LLVM codegen functions, no RTS needed
- [x] Fixed DefIds 10600-10622
- [x] Shared helper: `build_inline_reverse()` for catMaybes, lefts, rights, mapMaybe, partitionEithers
- [x] E2E tests: data_maybe, data_either, guard_basic
- [x] 58 total E2E tests pass

### Milestone E.14: when/unless + any/all + Closure Wrapping ✅
- [x] Fix when/unless Bool bug: use `extract_adt_tag()` not `ptr_to_int()` for Bool ADT
- [x] Implement `any`/`all` with loop + predicate closure + short-circuit
- [x] Add `even`/`odd` to `lower_builtin_direct` for first-class closure wrapping
- [x] E2E tests: when_unless, mapm_basic, any_all
- [x] 61 total E2E tests pass

### Milestone E.15: Data.List Completions ✅
- [x] Finite ops: scanr, scanl1, scanr1, unfoldr, intersect, zip3, zipWith3
- [x] Infinite generators (take-fused): iterate, repeat, cycle
- [x] Fixed DefIds 10700-10706
- [x] E2E tests: scanr_basic, unfoldr_basic, zip3_basic, take_iterate, intersect_basic
- [x] 66 total E2E tests pass

### Milestone E.16: Fix Broken Stubs + List Operations ✅
- [x] Fixed 5 broken stubs: maximum (accumulator loop), minimum, and (Bool tag short-circuit), or, Data.Map.notMember (XOR inversion)
- [x] 10 new functions: elemIndex, findIndex, isPrefixOf, isSuffixOf, isInfixOf, tails, inits, maximumBy, minimumBy, foldMap
- [x] Fixed DefIds 10800-10809
- [x] E2E tests: max_min_and_or, elem_index_prefix, tails_inits
- [x] 69 total E2E tests pass
- [x] Known limitation: `show` doesn't dispatch correctly for Bool-returning builtins (and/or/isPrefixOf); `compare` returns Int not Ordering ADT

### Milestone E.17: Ordering ADT + compare ✅
- [x] Ordering ADT (LT/EQ/GT) as zero-field ADT (tag 0=LT, 1=EQ, 2=GT)
- [x] `compare` returns Ordering instead of Int (fixed DefId 10900)
- [x] ShowCoerce::Ordering + RTS `bhc_show_ordering`
- [x] Fixed flat calling convention in maximumBy/minimumBy
- [x] E2E test: ordering_basic
- [x] 70 total E2E tests pass

### Milestone E.18: Monadic Combinators ✅
- [x] 7 pure LLVM codegen functions: filterM, foldM, foldM_, replicateM, replicateM_, zipWithM, zipWithM_
- [x] Fixed DefIds 11000-11006
- [x] Key pitfall: replicateM/replicateM_ re-lower action_expr each iteration, creating new blocks
- [x] E2E tests: monadic_combinators, zipwithm_basic
- [x] 70 total E2E tests pass (68+2 new, 4 pre-existing text/exception failures)

### Milestone E.19: System.FilePath + System.Directory ✅
- [x] System.FilePath: takeFileName, takeDirectory, takeExtension, dropExtension, takeBaseName, replaceExtension, isAbsolute, isRelative, hasExtension, splitExtension, </>
- [x] System.Directory: setCurrentDirectory, removeDirectory, renameFile, copyFile
- [x] 14 RTS FFI functions + 1 codegen-composed (splitExtension)
- [x] Fixed DefIds 11100-11115, VarIds 1000520-1000534
- [x] Key pitfall: `typeck/context.rs` type match must include ALL builtin types — `builtins.rs` `register_value()` alone is insufficient
- [x] Also fixed missing types for createDirectory, removeFile in typeck match
- [x] E2E tests: filepath_basic, directory_ops
- [x] 72 total E2E tests pass (70+2 new, 4 pre-existing text/exception failures)

### Milestone E.20: Fix DefId Misalignment for Text/ByteString/Exceptions ✅
- [x] Fixed DefId misalignment for Data.Text (38 funcs), Data.ByteString (24 funcs), Data.Text.Encoding (2 funcs)
- [x] Fixed DefIds 11200-11273 for all Text/ByteString/Encoding functions
- [x] Added typeck/context.rs match entries for throwIO/throw/try/evaluate
- [x] Added typeck/context.rs match entries for all Data.Text, Data.ByteString, Data.Map functions
- [x] 74 total E2E tests pass (72 existing + 4 previously-broken text/exception tests fixed)

### Milestone E.21: Data.Map Completion ✅
- [x] Linked bhc-containers in driver (1-line change)
- [x] Implemented 4 stubbed codegen functions: `unions` (cons-list fold), `keysSet` (iterate + set insert), `update` (lookup + Maybe closure + delete/insert), `alter` (build input Maybe + closure + delete/insert)
- [x] Fixed Bool ADT returns: container predicates (member, null, isSubmapOf, set_member, set_null) now use `allocate_bool_adt()` instead of `int_to_ptr()`
- [x] Fixed show inference: `expr_returns_bool()` now recognizes qualified container names (Data.Map.member, Data.Set.null, etc.)
- [x] Fixed type signatures for `update` (`b -> Maybe b`) and `alter` (`Maybe b -> Maybe b`) in builtins.rs + typeck/context.rs
- [x] E2E tests: map_basic (un-ignored), map_complete (new: update/alter/unions)
- [x] 76 total E2E tests pass (74 existing + 2 new, 0 failures)

### Milestone E.22: Data.Set/IntMap/IntSet Type Completion ✅
- [x] Added ~70 type match entries to typeck/context.rs for Data.Set (30), Data.IntMap (25), Data.IntSet (15)
- [x] Replaced 4 stub dispatches: Set.unions (cons-list walk + union accumulator), Set.partition (dual-accumulator with predicate + tuple return), IntSet.filter (reuse Set.filter), IntSet.foldr (reuse Set.foldr)
- [x] Fixed pre-existing VarId suffix bug: Set/IntSet binary/predicate/extremum dispatches passed suffix (e.g., 1127) instead of full VarId (1000127) — 13 dispatch sites fixed
- [x] E2E tests: set_basic (new: fromList/size/member/insert/delete/union/intersection/difference/filter/foldr), intmap_intset (new: IntSet size/member/filter/foldr + IntMap size/member/insert/delete)
- [x] 78 total E2E tests pass (76 existing + 2 new, 0 failures)

### Milestone E.23: Stock Deriving — Eq, Show for User ADTs ✅
- [x] Fixed `fresh_var` off-by-one bug in `deriving.rs`: name used counter N but VarId used N+1
- [x] Shared `DerivingContext` across all data types in a module (was recreated per type, causing VarId collision)
- [x] `fresh_counter` starts at 50000 to avoid collision with fixed DefId ranges (10000-11273)
- [x] Added `type_name: Option<String>` to `ConstructorMeta` for ADT type tracking
- [x] Pre-pass `detect_derived_instance_methods`: scans `$derived_show_*`/`$derived_eq_*` bindings
- [x] `strip_deriving_counter_suffix`: extracts clean type name from `Color_50000` → `Color`
- [x] `tag_constructors_with_type`: walks derived binding bodies, tags constructors with their type name
- [x] `infer_adt_type_from_expr`: checks constructor names in metadata for type dispatch
- [x] `lower_builtin_show` dispatches to derived show via indirect call `fn(env_ptr, value) -> string_ptr`
- [x] `PrimOp::Eq` dispatches to derived eq via indirect call `fn(env_ptr, lhs, rhs) -> Bool ADT` → `extract_adt_tag()`
- [x] Fixed `register_constructor` to preserve existing `type_name` when re-registering
- [x] E2E tests: derive_show (enum + ADT with fields), derive_eq (enum equality + inequality)
- [x] 80 total E2E tests pass (78 existing + 2 new, 0 failures)

### Milestone E.24: Stock Deriving — Ord for User ADTs ✅
- [x] Added `derived_compare_fns` dispatch table mirroring `derived_show_fns`/`derived_eq_fns` pattern
- [x] Extended `detect_derived_instance_methods` to detect `$derived_compare_*` bindings
- [x] `lower_builtin_compare` dispatches to derived compare for user ADTs before falling through to Int comparison
- [x] `PrimOp::Lt/Le/Gt/Ge` dispatch through derived compare: call → extract Ordering tag → compare tag
- [x] Made `compare` polymorphic (`a -> a -> Ordering`) in builtins.rs + fixed DefId block
- [x] Made `<`/`<=`/`>`/`>=` polymorphic (`a -> a -> Bool`) in builtins.rs + typeck/context.rs `cmp_binop()`
- [x] E2E test: derive_ord (compare on enums + comparison operators + multiple types)
- [x] 81 total E2E tests pass (80 existing + 1 new, 0 failures)

### Milestone E.25: String Type Class Methods ✅
- [x] `fromString` as identity pass-through
- [x] `read` (String→Int) via RTS `bhc_read_int`
- [x] `readMaybe` (String→Maybe Int) via RTS `bhc_try_read_int`
- [x] Fixed DefIds 11300-11302, VarIds 1000540-1000541
- [x] Show inference: readMaybe recognized as Maybe-returning
- [x] E2E test: string_read
- [x] 82 total E2E tests pass

### Milestone E.26: More List Operations ✅
- [x] 10 RTS functions: sortOn, nubBy, groupBy, deleteBy, unionBy, intersectBy, stripPrefix, insert, mapAccumL, mapAccumR
- [x] Internal helpers: extract_bool (dual Bool representation), call_eq_closure, alloc_nothing/just/tuple
- [x] Fixed DefIds 11400-11409, VarIds 1000550-1000559
- [x] Show inference: `expr_looks_like_list` integrated into `infer_show_from_expr` App case
- [x] E2E test: list_by_ops (13 assertions covering all 10 functions)
- [x] 83 total E2E tests pass

### Milestone E.27: Data.Function + Data.Tuple Builtins ✅
- [x] `succ`/`pred` via inline LLVM add/sub (Int → Int)
- [x] `(&)` reverse application operator (a → (a → b) → b)
- [x] `swap` for tuples: extract fields, allocate reversed tuple via `allocate_ptr_pair_tuple()`
- [x] `curry`: allocate tuple from (x, y), call f(tuple) via 1-arg closure
- [x] `uncurry`: extract fst/snd from pair, flat 3-arg call fn(env, fst, snd)
- [x] `fst`/`snd` added to `lower_builtin_direct` for first-class closure use (e.g., `map fst pairs`)
- [x] `succ`/`pred`/`swap` added to `lower_builtin_direct` for first-class closure use
- [x] Fixed DefIds 11500-11505, arity + dispatch entries
- [x] Show inference: succ/pred added to `expr_returns_int()`
- [x] Key pitfall: BHC compiles multi-arg functions as FLAT `fn(env, x, y)`, not curried — uncurry must use 3-arg call
- [x] E2E tests: data_function (succ/pred/(&)/map succ/map pred), tuple_functions (fst/snd/swap/curry/uncurry/map fst/map snd/map swap)
- [x] 85 total E2E tests pass (83 existing + 2 new, 0 failures)

### Milestone E.28: Arithmetic, Enum, Folds, Higher-Order, IO Input ✅
- [x] Arithmetic: `min`, `max`, `subtract` (inline LLVM compare+select / sub)
- [x] Enumeration: `enumFrom`, `enumFromThen`, `enumFromThenTo` (list-building loops with step)
- [x] Folds: `foldl1` (head as init, foldl tail), `foldr1` (reverse, head as init, foldl with flipped args)
- [x] Higher-order: `comparing` (call f on both args, inline compare), `until` (loop with predicate + transform)
- [x] IO Input: `getChar`, `isEOF`, `getContents` (3 RTS functions), `interact` (codegen-composed)
- [x] Partial builtin application: `create_partial_builtin_closure()` enables `map (min 5) xs` for codegen-only builtins
- [x] Fixed DefIds 11600-11613, VarIds 1000560-1000562
- [x] E2E tests: enum_functions (min/max/subtract/enum/foldl1/foldr1/until), fold_misc (foldl1/foldr1 with user fn, map with partial min/max/subtract)
- [x] 87 total E2E tests pass (85 existing + 2 new, 0 failures)

### Milestone E.29: flip + show Double + Data.Map.mapMaybe ✅
- [x] Fix flip calling convention: was using curried 2-step calls (segfault), fixed to flat 3-arg `fn(env, arg2, arg1)`
- [x] Show Double/Float literals: `ShowCoerce::Double` handles `FloatValue` directly (fpext f32→f64)
- [x] `expr_returns_double()`: recognizes unary (sqrt/sin/cos/...) and binary (/, **) Double-returning functions
- [x] `expr_looks_like_list`: Added Data.Map.toList/keys/elems/assocs, Data.Set.toList/elems
- [x] Data.Map.mapMaybe/mapMaybeWithKey: Fixed DefIds 11700-11701
- [x] flip/const added to `lower_builtin_direct` for first-class use
- [x] E2E tests: flip_test, show_double, map_maybe
- [x] 90 total E2E tests pass

### Milestone E.30: Unified Bool Extraction (extract_bool_tag) ✅
- [x] Two Bool representations: tagged-int-as-pointer (0/1) vs Bool ADT (heap struct)
- [x] `extract_bool_tag()`: checks `ptr_to_int <= 1` — if so, raw value; else loads ADT tag
- [x] Applied to: filter, takeWhile, dropWhile, span, break, find, partition
- [x] Phi predecessor pitfall: creates 3 new blocks; phi nodes must reference `bool_merge` block
- [x] E2E tests: filter_bool, list_predicate_ops, partition_test
- [x] 93 total E2E tests pass

### Milestone E.31: Show Nested Compound Types ✅
- [x] `ShowTypeDesc` struct: `#[repr(C)]` with `tag: i64`, `child1/child2: *const ShowTypeDesc`
- [x] Tags: 0-7 primitives (Int/Double/Float/Bool/Char/String/Unit/Ordering), 10-13 compounds (List/Maybe/Tuple2/Either)
- [x] `show_any()`: recursive dispatch in RTS, handles nested types at any depth
- [x] `show_any_prec()`: precedence-aware parens for constructor apps (Just x, Left x, Right x)
- [x] LLVM global descriptor trees built at compile time from expression structure analysis
- [x] `bhc_show_with_desc(ptr, desc)` FFI entry point (VarId 1000099)
- [x] Backward compatible: primitive ShowCoerce variants unchanged, compound types use new path
- [x] E2E tests: show_nested (list of tuples), show_nested_maybe (Maybe of list), show_nested_list (list of lists)
- [x] 96 total E2E tests pass

### Milestone E.32+: Road to Pandoc

#### E.32: OverloadedStrings + IsString ✅
- [x] `IsString` typeclass with `fromString :: String -> a` method
- [x] `OverloadedStrings` extension: string literals desugar to `fromString "..."` calls
- [x] `IsString` instances for Text, ByteString (via pack)
- [x] Identity instance for String

#### E.33: Record Syntax ✅
- [x] Named field declarations in data types
- [x] Field accessor functions (auto-generated from field names)
- [x] Record construction syntax `Foo { bar = 1, baz = "x" }`
- [x] Record update syntax `r { field = newVal }`
- [x] `RecordWildCards` extension (`Foo{..}` brings fields into scope)
- [x] `NamedFieldPuns` extension

#### E.34: ViewPatterns Codegen ✅
- [x] Lower `f -> pat` patterns to `let tmp = f arg in case tmp of pat -> ...`
- [x] Handle in case expressions and function argument patterns
- [x] Fallthrough semantics for non-matching patterns

#### E.35: TupleSections + MultiWayIf ✅
- [x] `TupleSections`: `(,x)` as partial tuple constructors, `(x,,z)` etc. — parser desugars to lambda
- [x] `MultiWayIf`: `if | cond1 -> e1 | cond2 -> e2 | otherwise -> e3` — parser desugars to nested if-then-else
- [x] Added `otherwise` to `lower_builtin_direct` for first-class use

#### E.36: Char Enum Ranges ✅
- [x] Polymorphic enum functions for Char type
- [x] Char range syntax: `['a'..'z']`

#### E.37: Char First-Class Predicates ✅
- [x] Data.Char predicates usable as first-class functions
- [x] Fix print Bool ADT dispatch

#### E.38: Manual Typeclass Instances ✅
- [x] Three-layer approach: lower → HIR-to-Core rename → codegen detect
- [x] `$instance_show_`/`$instance_==_`/`$instance_compare_` prefix dispatch

#### E.39: Dictionary-Passing for User-Defined Typeclasses ✅
- [x] Full dictionary-passing pipeline for user-defined typeclasses
- [x] ClassRegistry, DictContext, dict construction, `$sel_N` selectors

#### E.40: Higher-Kinded Dictionary Passing ✅
- [x] Dictionary passing for higher-kinded type variables (e.g., `Functor f`)

#### E.41: Default Methods + Superclass Constraints ✅
- [x] Default method implementations in typeclass declarations
- [x] Superclass constraint propagation

#### E.42: DeriveAnyClass ✅
- [x] Derive instances for user-defined typeclasses with default methods

#### E.43–E.45: Word Types + Integer + Lazy Let ✅
- [x] Word8/Word16/Word32/Word64 types with conversion operations
- [x] Integer arbitrary precision via `num-bigint` RTS (19 FFI functions)
- [x] Lazy let-bindings (initial support)

#### E.46: ScopedTypeVariables ✅
- [x] `ScopedTypeVariables` extension enabling type variable scoping

#### E.47: GeneralizedNewtypeDeriving ✅
- [x] Lift typeclass instances through `newtype` wrappers via newtype erasure
- [x] Support in `deriving` clause

#### E.48: FlexibleInstances + FlexibleContexts ✅
- [x] `FlexibleInstances` — instances on concrete types, nested types
- [x] `FlexibleContexts` — non-variable constraints in contexts
- [x] Instance context propagation

#### E.49: MultiParamTypeClasses ✅
- [x] Multiple type parameters in class declarations

#### E.50: FunctionalDependencies ✅
- [x] `| a -> b` functional dependency syntax in class declarations

#### E.51: DeriveFunctor ✅
- [x] Automatic `fmap` derivation for pure types

#### E.52: DeriveFoldable ✅
- [x] Automatic `foldr` derivation for user ADTs

#### E.53: DeriveTraversable ✅
- [x] Automatic `traverse` derivation for user ADTs

#### E.54: DeriveEnum + DeriveBounded ✅
- [x] Enum instances (toEnum/fromEnum) for simple enums
- [x] Bounded instances (minBound/maxBound) for simple enums

#### E.55: ReaderT-over-StateT Cross-Transformer ✅
- [x] `ReaderT r (StateT s IO)` nested codegen with 3-arg closures

#### E.56: ExceptT Cross-Transformer ✅
- [x] ExceptT over StateT and ExceptT over ReaderT codegen

#### E.57: WriterT Cross-Transformer ✅
- [x] WriterT over StateT and WriterT over ReaderT codegen

#### E.58: Full Lazy Let-Bindings ✅
- [x] Full Haskell-semantics lazy let-bindings

#### E.59: EmptyDataDecls + Strict Fields ✅
- [x] `EmptyDataDecls` extension (data types with no constructors)
- [x] `Type::Bang` for strict field annotations

#### E.60: GADTs ✅
- [x] GADT syntax with type refinement in pattern matches
- [x] Bool field extraction fix

#### E.61: TypeOperators ✅
- [x] Infix type syntax (e.g., `a :+: b`)

#### E.62: StandaloneDeriving + PatternSynonyms ✅
- [x] `deriving instance Eq Foo` standalone syntax
- [x] `pattern P x = Constructor x` bidirectional pattern synonyms
- [x] Nested pattern fallthrough fix

#### E.63: DeriveGeneric + NFData/DeepSeq Stubs ✅
- [x] `DeriveGeneric` generates stub Generic instances
- [x] NFData/DeepSeq stubs for Pandoc compatibility

#### E.64: EmptyCase + StrictData + DefaultSignatures + OverloadedLists ✅
- [x] `EmptyCase` — case with no alternatives
- [x] `StrictData` — all fields strict by default in module
- [x] `DefaultSignatures` — default method type signatures in classes
- [x] `OverloadedLists` — list literal desugaring via `fromList`

#### E.65+: Remaining Road to Pandoc (Proposed)

### Milestone F: Pandoc (Minimal)
- [ ] Compile Pandoc with a subset of readers/writers (e.g., Markdown → HTML only)
- [ ] Skip optional dependencies (skylighting, texmath, etc.)
- [ ] Requires: All Tier 1 and Tier 2 items

### Milestone G: Pandoc (Full)
- [ ] Compile full Pandoc with all readers/writers
- [ ] Pass Pandoc's test suite
- [ ] Performance within 2x of GHC-compiled Pandoc

---

## Key Files Reference

| File | Role |
|------|------|
| `crates/bhc-codegen/src/llvm/lower.rs` | LLVM lowering — add builtin handlers here |
| `crates/bhc-typeck/src/builtins.rs` | Type signatures for all builtins |
| `crates/bhc-core/src/eval/mod.rs` | Evaluator implementations |
| `crates/bhc-driver/` | Compilation orchestration |
| `crates/bhc-package/` | Package management |
| `crates/bhc-interface/` | Module interface files |
| `stdlib/bhc-base/` | Base library RTS functions |
| `stdlib/bhc-text/src/text.rs` | Text RTS (25+ FFI functions, E.7+E.8) |
| `stdlib/bhc-text/src/text_io.rs` | Text.IO RTS (7 FFI functions, E.10) |
| `stdlib/bhc-text/src/bytestring.rs` | ByteString RTS (24 FFI functions, E.8) |
| `stdlib/bhc-base/` | Base library: Data.Char predicates, show functions (E.9) |
| `stdlib/bhc-system/` | System/IO operations |
| `rts/bhc-rts/src/ffi.rs` | FFI functions for FilePath/Directory (E.19) |
| `stdlib/bhc-containers/` | Container data structures |
| `stdlib/bhc-transformers/` | Monad transformers |
| `rts/bhc-rts/` | Core runtime system |
| `crates/bhc-e2e-tests/fixtures/tier3_io/` | Transformer and IO test fixtures |

---

## Recent Progress

### 2026-02-21: Roadmap Assessment (E.32–E.64)

33 milestones completed since last update, adding 66+ E2E tests. Major areas:

**Language Extensions (E.32–E.35):**
- OverloadedStrings + IsString (E.32), Record syntax with wildcards/puns (E.33)
- ViewPatterns codegen (E.34), TupleSections + MultiWayIf (E.35)

**Typeclass Revolution (E.38–E.42):**
- Manual instances (E.38), dictionary-passing (E.39), higher-kinded (E.40)
- Default methods + superclasses (E.41), DeriveAnyClass (E.42)

**Type System Extensions (E.43–E.50):**
- Word types + Integer + lazy let (E.43–E.45), ScopedTypeVariables (E.46)
- GeneralizedNewtypeDeriving (E.47), FlexibleInstances/Contexts (E.48)
- MultiParamTypeClasses (E.49), FunctionalDependencies (E.50)

**Deriving Infrastructure (E.51–E.54):**
- DeriveFunctor (E.51), DeriveFoldable (E.52), DeriveTraversable (E.53)
- DeriveEnum + DeriveBounded (E.54)

**Monad Transformers (E.55–E.57):**
- ReaderT-over-StateT (E.55), ExceptT cross-transformer (E.56), WriterT cross-transformer (E.57)

**Advanced Features (E.58–E.64):**
- Full lazy let-bindings (E.58), EmptyDataDecls + strict fields (E.59)
- GADTs with type refinement (E.60), TypeOperators (E.61)
- StandaloneDeriving + PatternSynonyms (E.62), DeriveGeneric + NFData stubs (E.63)
- EmptyCase + StrictData + DefaultSignatures + OverloadedLists (E.64)

**Impact on Pandoc gaps:** Items #1-5 from the original "Missing for Pandoc" list are now complete
(OverloadedStrings, Records, ViewPatterns, TupleSections/MultiWayIf, GeneralizedNewtypeDeriving).
Remaining blockers: package system, CPP preprocessing, full GHC.Generics, type families.

### 2026-02-12: Milestones E.25–E.31
- E.25: String read/readMaybe/fromString (82 tests)
- E.26: 10 RTS list functions: sortOn, nubBy, groupBy, etc. (83 tests)
- E.27: succ/pred/(&)/swap/curry/uncurry (85 tests)
- E.28: 14 builtins (min/max/subtract/enum/folds/comparing/until/IO input), partial builtin application (87 tests)
- E.29: flip fix (flat 3-arg), show Double/Float, Data.Map.mapMaybe (90 tests)
- E.30: Unified Bool extraction (extract_bool_tag) for 7 list functions (93 tests)
- E.31: Recursive ShowTypeDesc for nested compound show (96 tests)

### 2026-02-11: Milestones E.20–E.24
- E.20: Fixed DefId misalignment for Text/ByteString/exceptions (74 tests)
- E.21: Data.Map completion (update/alter/unions), Bool ADT fixes (76 tests)
- E.22: Data.Set/IntMap/IntSet type completion, VarId suffix bug fix (78 tests)
- E.23: Stock deriving Eq/Show for user ADTs (80 tests)
- E.24: Stock deriving Ord, polymorphic compare (81 tests)

### 2026-02-09: Milestones E.15–E.19
- E.15: Data.List completions (scanr, unfoldr, zip3, iterate, repeat, cycle) (66 tests)
- E.16: Fix broken stubs + 10 new list operations (69 tests)
- E.17: Ordering ADT with compare (70 tests)
- E.18: 7 monadic combinators (70 tests)
- E.19: System.FilePath + System.Directory (72 tests)

### 2026-02-07: Milestones E.11–E.14
- E.11: Show compound types (52 tests)
- E.12: Numeric conversions + IORef (55 tests)
- E.13: Data.Maybe + Data.Either + guard (58 tests)
- E.14: when/unless + any/all + closure wrapping (61 tests)

### 2026-02-05–07: Milestones E.7–E.10
- E.7: Data.Text packed UTF-8 (43 tests)
- E.8: Data.ByteString + Text.Encoding (43 tests)
- E.9: Data.Char predicates + type-specialized show (45 tests)
- E.10: Data.Text.IO (46 tests)

### 2026-02-05: Milestones A–E (Foundations)
- Milestone A: Multi-module compilation
- Milestone B: File processing (word count, transform)
- Milestone C: Markdown parser (~500 LOC)
- Milestone D: StateT-based CSV parser
- Milestone E: JSON parser
- Nested transformer codegen (StateT over ReaderT)
- Exception handling (catch, bracket, finally)
