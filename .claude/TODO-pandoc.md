# Road to Pandoc: BHC Compilation TODO

**Document ID:** BHC-TODO-PANDOC
**Status:** In Progress
**Created:** 2026-01-30
**Updated:** 2026-02-06

---

## Goal

Compile and run [Pandoc](https://github.com/jgm/pandoc), a ~60k LOC Haskell
document converter with ~80 transitive package dependencies. This serves as the
north-star integration target for BHC's real-world Haskell compatibility.

---

## Current State

BHC compiles real Haskell programs to native executables via LLVM:
- 43 native E2E tests passing (including monad transformers, file IO, markdown parser, JSON parser)
- Monad transformers: StateT, ReaderT, ExceptT, WriterT all working
- Nested transformer stacks: `StateT s (ReaderT r IO)` with cross-transformer `ask` working
- MTL typeclasses registered: MonadReader, MonadState, MonadError, MonadWriter
- Exception handling: catch, bracket, finally, onException (E.5)
- Multi-package support with import paths (E.6)
- Data.Text: packed UTF-8 with 25+ operations (E.7)
- Data.ByteString: 24 RTS functions, Data.Text.Encoding bridge (E.8)
- All intermediate milestones A–E.8 done

### Gap to Pandoc

**Completed:** Self-contained programs with transformers, parsing, file IO, Text, ByteString, exceptions, multi-package imports
**Missing for Pandoc:**
1. **Full package system** — Basic import paths work (E.6), but no Hackage .cabal parsing yet
2. **Lazy Text/ByteString** — Only strict variants implemented
3. **GHC.Generics or TH** — Required for aeson JSON deriving

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

**Status:** ✅ Core APIs complete (E.7 + E.8), Lazy variants remaining
**Scope:** Large (remaining: Lazy variants)

Data.Text (E.7): packed UTF-8 with 25+ operations. Data.ByteString (E.8): 24
RTS functions with identical memory layout. Data.Text.Encoding (E.8): zero-copy
encodeUtf8/decodeUtf8 bridge.

- [x] Implement packed UTF-8 `Text` representation (not `[Char]`)
- [x] Core Text API: pack, unpack, append, cons, snoc, head, tail, length,
      null, map, take, drop, toLower, toUpper, toCaseFold, toTitle,
      isPrefixOf, isSuffixOf, isInfixOf, eq, compare, singleton, empty,
      filter, foldl', concat, intercalate, strip, words, lines, splitOn, replace
- [ ] Text.IO: readFile, writeFile, hGetContents, hPutStr
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

**Status:** Core exception handling complete (E.5), file IO working, remaining: directory ops
**Scope:** Medium

Exception handling (catch, bracket, finally, onException) is working (E.5).
File IO (readFile, writeFile, openFile, hClose) is working. System ops
(getArgs, getEnv, exitWith) are working. Remaining: directory operations.

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
- [ ] Directory: `doesFileExist`, `doesDirectoryExist`, `createDirectory`,
      `removeFile`, `getDirectoryContents`, `getCurrentDirectory`
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
- [x] Data.Set — RTS-backed BTreeSet (basic ops done)
- [x] Data.IntMap — shares Map RTS (basic ops done)
- [x] Data.IntSet — shares Set RTS (basic ops done)
- [ ] Data.Sequence — finger tree (not started)
- [ ] Data.Map.update, Data.Map.alter, Data.Map.unions (still stubbed)
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
- [ ] `filepath` — file path manipulation (`</>`, `takeExtension`, etc.)
- [ ] `directory` — filesystem operations
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
- [ ] Stock deriving: `Eq`, `Ord`, `Show`, `Read`, `Bounded`, `Enum`, `Ix`
- [ ] `DerivingStrategies`: stock, newtype, anyclass, via
- [ ] `DeriveAnyClass` for type classes with default method implementations
- [ ] `DerivingVia` for newtype-based instance delegation
- [ ] `GeneralizedNewtypeDeriving` for lifting instances through newtypes

---

## Tier 3 — Solvable with Current Architecture

### 3.1 Remaining Codegen Builtins

**Status:** ~300 of 587 builtins lowered
**Scope:** Small-Medium (ongoing)

- [ ] Monadic codegen: general `>>=`, `>>`, `return` via dictionary dispatch
- [ ] `mapM`, `mapM_`, `forM`, `forM_`, `sequence`, `sequence_`
- [ ] `when`, `unless`, `void`, `guard`
- [ ] Foldable/Traversable: `traverse`, `sequenceA`, `foldMap`, `toList`
- [ ] Data.Map.update, Data.Map.alter, Data.Map.unions
- [ ] Data.Set.unions, Data.Set.partition
- [ ] Remaining string/list stubs

### 3.2 Numeric and Conversion Operations

- [ ] `show` for all standard types (Int, Float, Double, Char, Bool, lists)
- [ ] `read` / `reads` for parsing
- [ ] `fromIntegral`, `realToFrac`, `toInteger`, `fromInteger`
- [ ] `Rational` type and operations
- [ ] `Data.Char` Unicode categories (full Unicode, not just ASCII)

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
| `stdlib/bhc-text/src/bytestring.rs` | ByteString RTS (24 FFI functions, E.8) |
| `stdlib/bhc-system/` | System/IO operations |
| `stdlib/bhc-containers/` | Container data structures |
| `stdlib/bhc-transformers/` | Monad transformers |
| `rts/bhc-rts/` | Core runtime system |
| `crates/bhc-e2e-tests/fixtures/tier3_io/` | Transformer and IO test fixtures |

---

## Recent Progress

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
