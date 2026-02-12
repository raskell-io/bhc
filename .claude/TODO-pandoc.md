# Road to Pandoc: BHC Compilation TODO

**Document ID:** BHC-TODO-PANDOC
**Status:** In Progress
**Created:** 2026-01-30
**Updated:** 2026-02-12

---

## Goal

Compile and run [Pandoc](https://github.com/jgm/pandoc), a ~60k LOC Haskell
document converter with ~80 transitive package dependencies. This serves as the
north-star integration target for BHC's real-world Haskell compatibility.

---

## Current State

BHC compiles real Haskell programs to native executables via LLVM:
- 85 native E2E tests passing (including monad transformers, file IO, markdown parser, JSON parser)
- Monad transformers: StateT, ReaderT, ExceptT, WriterT all working
- Nested transformer stacks: `StateT s (ReaderT r IO)` with cross-transformer `ask` working
- MTL typeclasses registered: MonadReader, MonadState, MonadError, MonadWriter
- Exception handling: catch, bracket, finally, onException (E.5)
- Multi-package support with import paths (E.6)
- Data.Text: packed UTF-8 with 25+ operations (E.7)
- Data.ByteString: 24 RTS functions, Data.Text.Encoding bridge (E.8)
- Data.Char predicates, type-specialized show functions (E.9)
- Data.Text.IO: native Text file/handle I/O (E.10)
- Show for compound types: String, [a], Maybe, Either, (a,b), () (E.11)
- Numeric ops: even/odd, gcd/lcm, divMod/quotRem, fromIntegral + IORef (E.12)
- Data.Maybe: fromMaybe, maybe, listToMaybe, maybeToList, catMaybes, mapMaybe (E.13)
- Data.Either: either, fromLeft, fromRight, lefts, rights, partitionEithers (E.13)
- Control.Monad: when, unless, guard, mapM_, any, all, filterM, foldM, foldM_, replicateM, replicateM_, zipWithM, zipWithM_ (E.13, E.14, E.18)
- Data.List: scanr, scanl1, scanr1, unfoldr, intersect, zip3, zipWith3 (E.15)
- Data.List: take-fused iterate, repeat, cycle (E.15)
- Data.List: elemIndex, findIndex, isPrefixOf, isSuffixOf, isInfixOf, tails, inits (E.16)
- Fixed stubs: maximum, minimum, and, or, Data.Map.notMember (E.16)
- maximumBy, minimumBy, foldMap (E.16)
- Ordering ADT (LT/EQ/GT), compare returning Ordering (E.17)
- System.FilePath: takeFileName, takeDirectory, takeExtension, dropExtension, takeBaseName, replaceExtension, isAbsolute, isRelative, hasExtension, splitExtension, </> (E.19)
- System.Directory: setCurrentDirectory, removeDirectory, renameFile, copyFile (E.19)
- Data.Map: full operation set including update, alter, unions, keysSet (E.21)
- Data.Set: full type support + unions, partition codegen (E.22)
- Data.IntMap/Data.IntSet: full type support + filter, foldr codegen (E.22)
- Fixed Bool ADT returns for container predicates (member, null, isSubmapOf) (E.21)
- Fixed VarId suffix bug in Set/IntSet binary/predicate dispatches (E.22)
- Stock deriving: Eq, Show, Ord for user-defined ADTs (E.23, E.24)
- Polymorphic compare and comparison operators for derived Ord (E.24)
- String methods: fromString, read, readMaybe (E.25)
- Data.List: sortOn, nubBy, groupBy, deleteBy, unionBy, intersectBy, stripPrefix, insert, mapAccumL, mapAccumR (E.26)
- Data.Function: succ, pred, (&) reverse application (E.27)
- Data.Tuple: swap, curry, uncurry + fst/snd as first-class closures (E.27)
- All intermediate milestones A–E.27 done

### Gap to Pandoc

**Completed:** Self-contained programs with transformers, parsing, file IO, Text, ByteString, Text.IO, Data.Char, show for compound types, numeric conversions, IORef, exceptions, multi-package imports, Data.Maybe/Either utilities, extensive Data.List operations, when/unless/guard/any/all, monadic combinators (filterM/foldM/replicateM/zipWithM), Ordering ADT with compare, System.FilePath + System.Directory, Data.Map complete (update/alter/unions/keysSet), fixed DefId misalignment for Text/ByteString/exceptions (E.20), Bool ADT for container predicates (E.21), Data.Set/IntMap/IntSet full type support + codegen completions (E.22), stock deriving Eq/Show/Ord for user ADTs (E.23, E.24), String read/readMaybe/fromString (E.25), sortOn/nubBy/groupBy/deleteBy/unionBy/intersectBy/stripPrefix/insert/mapAccumL/mapAccumR (E.26), succ/pred/(&)/swap/curry/uncurry + fst/snd as first-class closures (E.27)
**Missing for Pandoc:**
1. **Full package system** — Basic import paths work (E.6), but no Hackage .cabal parsing yet
2. **Lazy Text/ByteString** — Only strict variants implemented
3. **GHC.Generics or TH** — Required for aeson JSON deriving
4. ~~**show for Bool from builtins**~~ — Fixed in E.21: container predicates (member/null/isSubmapOf) now return proper Bool ADT; show inference recognizes qualified container names

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

**Status:** Parsed, partially type-checked
**Scope:** Medium

- [ ] GADT type checking: refine types in branches based on constructor
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
- [ ] Codegen for nested transformer stacks: `ReaderT r (StateT s IO)` (lifting StateT into ReaderT)

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

**Status:** Basic deriving works
**Scope:** Medium

- [ ] `GHC.Generics` — `Generic` class with `Rep` type family
- [ ] Generic representations: `V1`, `U1`, `K1`, `M1`, `:+:`, `:*:`
- [ ] `from` / `to` methods for converting to/from generic rep
- [ ] Derive `Generic` for user-defined types
- [x] Stock deriving: `Eq`, `Show` for simple enums and ADTs with fields (E.23)
- [x] Stock deriving: `Ord` for simple enums and ADTs with fields (E.24)
- [ ] Stock deriving: `Read`, `Bounded`, `Enum`, `Ix`
- [ ] `DerivingStrategies`: stock, newtype, anyclass, via
- [ ] `DeriveAnyClass` for type classes with default method implementations
- [ ] `DerivingVia` for newtype-based instance delegation
- [ ] `GeneralizedNewtypeDeriving` for lifting instances through newtypes

---

## Tier 3 — Solvable with Current Architecture

### 3.1 Remaining Codegen Builtins

**Status:** ~490+ of 587 builtins lowered (E.13–E.27 added ~80+ functions + derived dispatches)
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
- [ ] `show` for remaining types: Double, nested compound types
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

### 3.3 Performance

- [ ] Strictness analysis to avoid thunk buildup
- [ ] Specialization of polymorphic functions
- [ ] Worker/wrapper transformation
- [ ] Inlining for small functions
- [ ] Stream fusion for list operations

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

### 2026-02-12: Milestone E.27 Data.Function + Data.Tuple Builtins
- 6 new builtins: succ, pred, (&), swap, curry, uncurry
- `succ`/`pred`: inline LLVM `build_int_add`/`build_int_sub` with `coerce_to_int` → `int_to_ptr`
- `(&)`: reverse application — mirror of `lower_builtin_apply` with args swapped
- `swap`: extract fst/snd via `extract_adt_field`, allocate reversed tuple via new `allocate_ptr_pair_tuple()` helper
- `curry`: allocate tuple from (x, y) via `allocate_ptr_pair_tuple`, call f(tuple) via 1-arg closure
- `uncurry`: extract fst/snd from pair, flat 3-arg call `fn_ptr(f_env, fst, snd)` — BHC uses flat calling convention, NOT curried
- Added 5 entries to `lower_builtin_direct` for first-class closure use: fst, snd, succ, pred, swap
- Fixed DefIds 11500-11505, arity entries in `builtin_info()`, dispatch in `lower_builtin()`
- Key pitfall: uncurry initially used curried 2-step calls causing segfault — BHC compiles multi-arg user functions as FLAT `fn(env, x, y) -> result`
- E2E tests: data_function (7 assertions), tuple_functions (10 assertions)
- 85 E2E tests pass (83 existing + 2 new, 0 failures)

### 2026-02-11: Milestone E.26 More List Operations
- 10 new RTS functions in `stdlib/bhc-base/src/list.rs`: sortOn, nubBy, groupBy, deleteBy, unionBy, intersectBy, stripPrefix, insert, mapAccumL, mapAccumR
- Internal helpers: `extract_bool()` (handles both tagged-int-as-pointer and Bool ADT), `call_eq_closure()`, `alloc_nothing/just/tuple()`
- Fixed DefIds 11400-11409, VarIds 1000550-1000559
- Integrated `expr_looks_like_list()` into `infer_show_from_expr` App case for automatic list show dispatch
- Key pitfall: polymorphic extractors (fromMaybe, snd, head) must NOT be in `expr_looks_like_list` — causes segfaults in tests where they return non-list types. Use recognized list wrappers instead: `concat (maybeToList (...))`, `take 100 (snd (...))`
- E2E test: list_by_ops (13 assertions covering all 10 functions)
- 83 E2E tests pass (82 existing + 1 new, 0 failures)

### 2026-02-11: Milestone E.25 String Type Class Methods
- `fromString` as identity pass-through, `read` (String→Int via RTS bhc_read_int), `readMaybe` (String→Maybe Int via RTS bhc_try_read_int)
- Fixed DefIds 11300-11302, VarIds 1000540-1000541
- Show inference: readMaybe recognized as Maybe-returning
- E2E test: string_read
- 82 E2E tests pass

### 2026-02-12: Milestone E.24 Stock Deriving — Ord for User ADTs
- Added `derived_compare_fns` dispatch table (mirrors derived_show_fns/derived_eq_fns from E.23)
- Extended `detect_derived_instance_methods` to detect `$derived_compare_*` bindings
- `lower_builtin_compare` checks for user ADTs with derived Ord before falling through to Int comparison
- `PrimOp::Lt/Le/Gt/Ge` dispatch through derived compare: extract Ordering tag (0=LT, 1=EQ, 2=GT), compare (Lt: tag==0, Le: tag!=2, Gt: tag==2, Ge: tag!=0)
- Made `compare` and comparison operators polymorphic in THREE places: `builtins.rs`, `typeck/context.rs` `cmp_binop()`, and fixed DefId block — previously monomorphic (`Int -> Int -> ...`)
- E2E test: derive_ord (compare on enums, </<=/>/>=, multiple types)
- 81 E2E tests pass (80 existing + 1 new, 0 failures)

### 2026-02-12: Milestone E.23 Stock Deriving — Eq, Show for User ADTs
- Fixed `fresh_var` off-by-one in `deriving.rs`: name used counter value N but VarId used N+1 (counter incremented between them). Fixed by capturing counter before increment.
- Fixed `DerivingContext` creation: was recreated per data type in `context.rs`, causing VarId collision when multiple types derive in same module. Fixed by sharing single `DerivingContext` across module items.
- `fresh_counter` starts at 50000 to avoid collision with fixed DefId ranges (10000-11273)
- Added `type_name: Option<String>` to `ConstructorMeta` — tracks which data type each constructor belongs to
- Pre-pass `detect_derived_instance_methods` scans bindings for `$derived_show_*`/`$derived_eq_*`, populates dispatch tables `derived_show_fns`/`derived_eq_fns`, and calls `tag_constructors_with_type` to label constructors
- `strip_deriving_counter_suffix("Color_50000")` → `"Color"` using `rsplit_once('_')` + digit check
- `infer_adt_type_from_expr` checks Var/App/Let/Case expressions against constructor_metadata for type_name
- Show dispatch: `lower_builtin_show` calls derived show function via `build_indirect_call(fn_ptr, [env_ptr, value])` → returns string pointer
- Eq dispatch: `PrimOp::Eq` calls derived eq function via `build_indirect_call(fn_ptr, [env_ptr, lhs, rhs])` → returns Bool ADT → `extract_adt_tag()` for i64 result
- Fixed `register_constructor` to use `get_mut` + update (preserves existing `type_name`) instead of `.insert()` (which overwrote to None)
- E2E tests: derive_show (Red/Green/Blue enum + Circle/Rectangle ADT with fields), derive_eq (enum equality and inequality)
- 80 E2E tests pass (78 existing + 2 new, 0 failures)

### 2026-02-11: Milestone E.22 Data.Set/IntMap/IntSet Type Completion
- Added ~70 type match entries to typeck/context.rs for Data.Set (30 entries), Data.IntMap (25 entries), Data.IntSet (15 entries) — fixes triple registration pitfall where functions were in builtins.rs and lower/context.rs but missing from typeck/context.rs
- Replaced 4 stub dispatches in lower.rs: Set.unions (cons-list walk + bhc_set_union accumulator), Set.partition (dual-accumulator with predicate → tuple return via alloc_adt), IntSet.filter (reuse Set.filter), IntSet.foldr (reuse Set.foldr)
- Discovered and fixed pre-existing VarId suffix bug: `lower_builtin_set_binary`, `lower_builtin_set_predicate`, etc. take `rts_id: usize` and do `VarId::new(rts_id)` — dispatch sites were passing suffixes (e.g., 1127) instead of full VarIds (1000127), creating non-existent VarIds. Fixed 13 dispatch sites for Set and IntSet operations.
- E2E tests: set_basic (fromList/size/member/insert/delete/union/intersection/difference/filter/foldr), intmap_intset (IntSet size/member/filter/foldr + IntMap size/member/insert/delete)
- 78 E2E tests pass (76 existing + 2 new, 0 failures)

### 2026-02-10: Milestone E.21 Data.Map Completion
- Linked bhc-containers in driver so Data.Map RTS functions are available at link time
- Implemented 4 stubbed codegen functions: `unions` (cons-list fold via bhc_map_union), `keysSet` (iterate map keys into set), `update` (lookup + Maybe-returning closure + conditional delete/insert), `alter` (build input Maybe + closure + conditional delete/insert)
- Fixed Bool ADT returns for 5 container predicates: `map_member`, `map_null`, `map_is_submap_of`, `set_null`, `set_member` — changed from `int_to_ptr()` (tagged-int-as-pointer) to `allocate_bool_adt()` (proper ADT struct)
- Fixed show inference: `expr_returns_bool()` now recognizes qualified container names (Data.Map.member, Data.Map.null, Data.Set.member, Data.Set.null, Data.Map.isSubmapOf, etc.)
- Fixed type signatures for `Data.Map.update` and `Data.Map.alter` in both builtins.rs and typeck/context.rs — closure types must include Maybe (`b -> Maybe b` and `Maybe b -> Maybe b`)
- E2E tests: map_basic (un-ignored, was blocked by missing bhc-containers link), map_complete (new: tests update/alter/unions)
- 76 E2E tests pass (74 existing + 2 new, 0 failures)

### 2026-02-10: Milestone E.20 Fix DefId Misalignment
- Fixed DefId misalignment for Data.Text (38 funcs), Data.ByteString (24 funcs), Data.Text.Encoding (2 funcs) + exception functions
- Fixed DefIds 11200-11273 for all affected functions
- Added typeck/context.rs match entries for throwIO/throw/try/evaluate and all Data.Text/ByteString/Map functions
- 74 E2E tests pass (72 + 4 previously-broken text/exception tests fixed)

### 2026-02-09: Milestone E.19 System.FilePath + System.Directory
- 14 RTS FFI functions in `rts/bhc-rts/src/ffi.rs`: takeFileName, takeDirectory, takeExtension, dropExtension, takeBaseName, replaceExtension, isAbsolute, isRelative, hasExtension, combine, setCurrentDirectory, removeDirectory, renameFile, copyFile
- 1 codegen-composed function: splitExtension (calls dropExtension + takeExtension, packs into tuple)
- Fixed DefIds 11100-11115, VarIds 1000520-1000534
- Key bug found: `typeck/context.rs` has a separate type lookup match (`register_builtins_from_lowering_defs`) that must handle ALL builtins. Functions not in this match get fresh type variables in the second pass, causing type mismatch errors. This also affected previously-untested functions: createDirectory, removeFile, removeDirectory
- Added type entries for 15 E.19 functions + 4 previously-missing functions in typeck/context.rs
- E2E tests: filepath_basic (13 assertions), directory_ops (create/write/copy/rename/remove)
- 72 E2E tests pass (70 existing + 2 new)

### 2026-02-09: Milestones E.17–E.18
- E.17: Ordering ADT (LT/EQ/GT) with proper `compare` returning Ordering. ShowCoerce::Ordering. Fixed flat calling convention in maximumBy/minimumBy.
- E.18: 7 monadic combinators (filterM, foldM, foldM_, replicateM, replicateM_, zipWithM, zipWithM_). Fixed DefIds 11000-11006. Key pitfall: replicateM/replicateM_ re-lower action_expr each iteration which creates new blocks.
- 70 E2E tests pass after E.17-E.18

### 2026-02-07: Milestone E.16 Fix Broken Stubs + List Operations
- Fixed 5 broken stubs: `maximum` (proper accumulator loop with `icmp sgt`), `minimum` (`icmp slt`), `and` (Bool ADT tag check, short-circuit on False), `or` (short-circuit on True), `Data.Map.notMember` (call member, XOR tag with 1)
- 10 new functions at Fixed DefIds 10800-10809: `elemIndex`, `findIndex`, `isPrefixOf`, `isSuffixOf`, `isInfixOf`, `tails`, `inits`, `maximumBy`, `minimumBy`, `foldMap`
- `isSuffixOf` reverses both lists then runs isPrefixOf logic; `isInfixOf` runs outer loop with inner isPrefixOf at each position
- `tails`/`inits` build list-of-lists: tails walks consing suffixes; inits accumulates reversed elements, reverses prefix at each step
- `maximumBy`/`minimumBy` call 2-arg comparison closure (partial application pattern), check Ordering ADT tag
- `foldMap` delegates to `concatMap` (list Foldable specialization)
- Known limitations: `show` for Bool-returning builtins prints raw pointers (type inference doesn't propagate Bool through `and`/`or` return); `compare` returns Int not Ordering ADT so `maximumBy compare` doesn't type-check
- E2E tests: max_min_and_or, elem_index_prefix, tails_inits
- 69 E2E tests pass (66 existing + 3 new)

### 2026-02-07: Milestone E.15 Data.List Completions
- 7 finite list ops: `scanr`, `scanl1`, `scanr1`, `unfoldr`, `intersect`, `zip3`, `zipWith3`
- 3 infinite generators (take-fused): `iterate`, `repeat`, `cycle` — fused with `take` to avoid infinite loops
- Fixed DefIds 10700-10706
- E2E tests: scanr_basic, unfoldr_basic, zip3_basic, take_iterate, intersect_basic
- 66 E2E tests pass

### 2026-02-07: Milestone E.14 when/unless + any/all
- Fixed `when`/`unless` Bool bug: was using `ptr_to_int` (gives non-zero for ADT pointers), now uses `extract_adt_tag()`
- Implemented `any`/`all` with loop + predicate closure call + short-circuit on Bool tag
- Added `even`/`odd` entries in `lower_builtin_direct` so they can be passed as first-class function values (e.g., `any even xs`)
- E2E tests: when_unless, mapm_basic, any_all
- 61 E2E tests pass

### 2026-02-07: Milestone E.13 Data.Maybe + Data.Either + guard
- 13 pure LLVM codegen functions, no RTS needed
- Data.Maybe: `fromMaybe`, `maybe`, `listToMaybe`, `maybeToList`, `catMaybes`, `mapMaybe`
- Data.Either: `either`, `fromLeft`, `fromRight`, `lefts`, `rights`, `partitionEithers`
- Control.Monad: `guard` (returns `[()]` for True, `[]` for False)
- Implementation patterns: Group A (tag check + phi), Group B (closure call), Group C (filter loop + reverse), Group D (dual accumulator)
- Shared helper `build_inline_reverse()` used by catMaybes, lefts, rights, mapMaybe, partitionEithers
- Fixed DefIds 10600-10622
- E2E tests: data_maybe, data_either, guard_basic
- 58 E2E tests pass

### 2026-02-07: Milestone E.12 Numeric Conversions + IORef
- Numeric conversions: `fromIntegral`/`toInteger`/`fromInteger` as identity pass-through (BHC only has Int/Double/Float)
- `even`/`odd` via inline LLVM `srem(n, 2)`, returns proper Bool ADT via `allocate_bool_adt()`
- `gcd`/`lcm` via RTS functions `bhc_gcd`/`bhc_lcm` (Euclidean algorithm)
- `divMod` with floor-division semantics: adjusts quotient/remainder when signs differ
- `quotRem` with truncation-toward-zero: direct LLVM `sdiv`/`srem`
- Both return `(Int, Int)` tuples via `allocate_int_pair_tuple()` (24 bytes: tag + two int-as-ptr fields)
- IORef: 3 RTS functions (`bhc_new_ioref`/`bhc_read_ioref`/`bhc_write_ioref`) + codegen-composed `modifyIORef`
- Fixed DefIds 10500-10507 for numeric ops to bypass 30-entry sequential array misalignment bug
- Added `expr_returns_bool()`/`expr_returns_int()` to `infer_show_from_expr` for proper show dispatch
- VarIds 1000500-1000504, DefIds 10400-10404 (IORef) and 10500-10507 (numeric ops)
- 55 E2E tests pass (52 existing + 3 new: numeric_ops, divmod, ioref_basic)

### 2026-02-07: Milestone E.11 Show Compound Types
- 6 RTS show functions: bhc_show_string, bhc_show_list, bhc_show_maybe, bhc_show_either, bhc_show_tuple2, bhc_show_unit
- Expression-based type inference (`infer_show_from_expr`) as fallback since `expr.ty()` returns `Ty::Error` in Core IR
- ShowCoerce extended with 6 compound variants; RTS type tags (0-5) passed as i64 args for element formatting
- `bhc_show_list` special-cases tag==4 (Char) to format as String `"abc"` instead of `['a','b','c']`
- VarIds 1000092-1000097, DefIds 10105-10110
- 52 E2E tests pass (46 existing + 6 new show_* tests)

### 2026-02-07: Milestone E.10 Data.Text.IO
- 7 RTS functions in `stdlib/bhc-text/src/text_io.rs`: readFile, writeFile, appendFile, hGetContents, hGetLine, hPutStr, hPutStrLn
- 4 codegen-composed convenience functions: putStr, putStrLn, getLine, getContents
- Handle functions use same sentinel-pointer pattern as bhc-rts (1=stdin, 2=stdout, 3=stderr)
- VarIds 1000240-1000246, DefIds 10300-10310
- Fixed import shadowing bug: `register_standard_module_exports` must not call `register_qualified_name` when the qualified name is already directly bound — otherwise it redirects to the Prelude version
- 46 E2E tests pass

### 2026-02-06: Milestone E.9 Data.Char + Type-Specialized Show
- Data.Char predicates: isAlpha, isDigit, isUpper, isLower, isAlphaNum, isSpace, isPunctuation
- Data.Char conversions: toUpper, toLower, ord, chr, digitToInt, intToDigit
- Type-specialized show: showInt, showBool, showChar, showFloat with ShowCoerce enum
- Char predicates return proper ADT booleans (tag 0=False, 1=True) for showBool compatibility
- Fixed VarId 1000091 for showBool (was incorrectly 1000075), added showFloat at VarId 1000090
- bhc-base linked for char/show RTS functions
- 45 E2E tests pass

### 2026-02-06: Milestone E.8 Data.ByteString + Text Completion
- ByteString RTS: 24 FFI functions with identical memory layout to Text (`[data_ptr, offset, byte_len, ...bytes...]`)
- Data.Text.Encoding: `encodeUtf8` (zero-copy, shares UTF-8 buffer), `decodeUtf8` (validates UTF-8)
- Additional Text ops: filter, foldl', concat, intercalate, strip, words, lines, splitOn, replace
- Functions returning lists (words/lines/splitOn) build BHC cons-lists via `build_text_list()`
- VarIds: ByteString 1000400-1000423, Text.Encoding 1000430-1000431, new Text ops 1000227-1000236
- Fixed linker library search order: debug path now searched before release (prevents stale release `.dylib` shadowing)
- Qualified names must be registered in three places: builtins.rs, context.rs `define_builtins()`, and lower.rs
- 43 E2E tests pass, 66 bhc-text unit tests pass

### 2026-02-05: Milestone E.7 Data.Text Foundation Complete
- Implemented packed UTF-8 `Text` type via RTS-backed functions
- Core API: pack, unpack, append, length, null, take, drop, toUpper, toLower
- Registered in all three systems: typeck/builtins.rs, lower/context.rs, codegen/lower.rs
- VarIds 1000200-1000226 allocated for Data.Text RTS functions
- E2E test `tier3_io/text_basic` passes (outputs "5", "HELLO WORLD", "Hello", "World")
- Discovered three-system registration requirement: type checker, lowering context, AND codegen

### 2026-02-05: Milestone E JSON Parser Complete
- Self-contained JSON key-value parser compiles and runs correctly
- Demonstrates string parsing, field extraction, and integer conversion
- Discovered and fixed several codegen bugs:
  - Boolean operators (`&&`, `||`) now correctly extract ADT tags instead of using pointer values
  - `lower_binary_bool` returns proper Bool ADT pointers instead of raw integers
  - Added `extract_bool_value` helper for handling both raw booleans and Bool ADT pointers
- Identified workarounds for current limitations:
  - List wildcard patterns (`_`) should use explicit `[]` and `(_:_)` patterns
  - Duplicate cons patterns in case need if-then-else rewrite
  - Inline arithmetic in recursive calls needs let bindings
- E2E test `milestone_e_json` passes (outputs "Alice" and "30")

### 2026-02-05: Nested Transformer Codegen
- Implemented `StateT s (ReaderT r IO)` nested transformer support
- `ask` now works inside StateT computations over ReaderT
- 3-argument closure convention: `(closure_env, state, reader_env)`
- `apply_state_t_lift_to_value()` properly runs inner ReaderT actions
- E2E test `cross_state_reader` now passes (outputs `15` from `10 + length "Hello"`)
- MTL typeclasses (MonadReader, MonadState, etc.) registered in type system

### 2026-01-30: Milestone D Complete
- CSV parser using `StateT String IO` compiles and runs correctly
- Demonstrates monad transformer codegen with String state manipulation
- E2E test `milestone_d_csv_parser` passes
