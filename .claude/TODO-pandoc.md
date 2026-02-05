# Road to Pandoc: BHC Compilation TODO

**Document ID:** BHC-TODO-PANDOC
**Status:** In Progress
**Created:** 2026-01-30
**Updated:** 2026-02-05

---

## Goal

Compile and run [Pandoc](https://github.com/jgm/pandoc), a ~60k LOC Haskell
document converter with ~80 transitive package dependencies. This serves as the
north-star integration target for BHC's real-world Haskell compatibility.

---

## Current State

BHC compiles real Haskell programs to native executables via LLVM:
- 38 native E2E tests passing (including monad transformers, file IO, markdown parser, JSON parser)
- 19 WASM E2E tests passing
- Monad transformers: StateT, ReaderT, ExceptT, WriterT all working
- Nested transformer stacks: `StateT s (ReaderT r IO)` with cross-transformer `ask` working
- MTL typeclasses registered: MonadReader, MonadState, MonadError, MonadWriter
- Milestone E (JSON parser) complete — all intermediate milestones A–E done

### Gap to Pandoc

**Completed:** Self-contained single-file programs with transformers, parsing, file IO
**Missing for Pandoc:**
1. **Package system** — Can't import from Hackage packages yet
2. **Data.Text** — Pandoc uses Text everywhere, we only have String
3. **Exception handling** — Need catch/throw/bracket for robust code
4. **GHC.Generics or TH** — Required for aeson JSON deriving

---

## Tier 1 — Showstoppers

These must be resolved before any real-world Haskell program can compile.

### 1.1 Package System Integration

**Status:** Not connected
**Scope:** Large

The `bhc-package` crate exists with TOML manifests, semver resolution, and
lockfile support, but it is not wired into `bhc-driver`. Pandoc depends on
~80 packages from Hackage.

- [ ] Parse `.cabal` files (at minimum: exposed-modules, build-depends, hs-source-dirs)
- [ ] Resolve transitive dependency graph from a cabal file
- [ ] Fetch packages from Hackage (tar.gz download + unpack)
- [ ] Wire package resolution into `bhc-driver` compilation pipeline
- [ ] Support `PackageImports` extension for disambiguating modules
- [ ] Handle conditional dependencies (flags, OS checks, impl checks)
- [ ] Generate and consume interface files (`.bhi`) across package boundaries
- [ ] Cache compiled packages to avoid recompilation

**Key files:**
- `crates/bhc-package/` — existing package infrastructure
- `crates/bhc-driver/` — compilation orchestration
- `crates/bhc-interface/` — module interface files

### 1.2 Data.Text and Data.ByteString

**Status:** Stub modules exist
**Scope:** Large

Pandoc uses `Data.Text` pervasively. BHC only has `String` as `[Char]` linked
lists — orders of magnitude slower for document processing.

- [ ] Implement packed UTF-8 `Text` representation (not `[Char]`)
- [ ] Core Text API (~50 functions): pack, unpack, append, cons, snoc, head,
      tail, length, null, map, filter, foldl', foldr, intercalate, split,
      splitOn, strip, toLower, toUpper, isPrefixOf, isSuffixOf, isInfixOf,
      replace, breakOn, words, lines, unwords, unlines, etc.
- [ ] Text.IO: readFile, writeFile, hGetContents, hPutStr
- [ ] Text.Encoding: encodeUtf8, decodeUtf8, decodeUtf8'
- [ ] Lazy Text variant (Data.Text.Lazy, Data.Text.Lazy.IO)
- [ ] ByteString: packed byte array type
- [ ] ByteString API (~40 functions): pack, unpack, append, head, tail,
      length, null, map, filter, foldl', take, drop, splitAt, elem, etc.
- [ ] ByteString.Lazy and ByteString.Builder
- [ ] SIMD-optimized operations where applicable (memchr, memcmp, etc.)

**Key files:**
- `stdlib/bhc-text/` — text/bytestring crate (currently empty/stub)

### 1.3 Full IO and Exception Handling

**Status:** Basic print/getLine works; rest stubbed
**Scope:** Medium-Large

Pandoc reads/writes files, uses handles, and relies on structured exception
handling throughout.

- [ ] Handle abstraction: `Handle`, `IOMode`, `BufferMode`
- [ ] File operations: `openFile`, `hClose`, `hFlush`, `hSetBuffering`
- [ ] Reading: `hGetChar`, `hGetLine`, `hGetContents`, `hIsEOF`
- [ ] Writing: `hPutChar`, `hPutStr`, `hPutStrLn`
- [ ] Standard handles: `stdin`, `stdout`, `stderr`
- [ ] File-level: `readFile`, `writeFile`, `appendFile`
- [ ] Exception types: `SomeException`, `IOException`, `ErrorCall`
- [ ] Exception primitives: `throw`, `throwIO`, `catch`, `try`
- [ ] Resource management: `bracket`, `bracket_`, `finally`, `onException`
- [ ] Exception hierarchy: `Exception` typeclass with `toException`/`fromException`
- [ ] Asynchronous exceptions: `mask`, `uninterruptibleMask` (at least stubs)
- [ ] System operations: `getArgs`, `getProgName`, `getEnv`, `lookupEnv`
- [ ] Exit: `exitSuccess`, `exitFailure`, `exitWith`
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

### Milestone E.5: Exception Handling
- [ ] Implement `throw`, `catch`, `try` for IO exceptions
- [ ] Implement `bracket` for resource management
- [ ] Exception hierarchy: `SomeException`, `IOException`, `ErrorCall`
- [ ] E2E test: program that opens file, handles "file not found", cleans up

### Milestone E.6: Multi-Package Program
- [ ] Wire `bhc-package` into `bhc-driver`
- [ ] Parse minimal `.cabal` files (exposed-modules, build-depends, hs-source-dirs)
- [ ] Compile a program that imports from 2-3 simple Hackage packages
- [ ] Example: use `filepath` and `directory` packages

### Milestone E.7: Data.Text Foundation
- [ ] Implement packed UTF-8 `Text` type (not `[Char]`)
- [ ] Core API: pack, unpack, append, length, null, map, filter
- [ ] Text.IO: readFile, writeFile
- [ ] E2E test: read file as Text, process, write output

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
| `stdlib/bhc-text/` | Text/ByteString (to be implemented) |
| `stdlib/bhc-system/` | System/IO operations |
| `stdlib/bhc-containers/` | Container data structures |
| `stdlib/bhc-transformers/` | Monad transformers |
| `rts/bhc-rts/` | Core runtime system |
| `crates/bhc-e2e-tests/fixtures/tier3_io/` | Transformer and IO test fixtures |

---

## Recent Progress

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
