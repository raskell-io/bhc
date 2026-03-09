# Road to Zentinel: BHC Compilation TODO

**Document ID:** BHC-TODO-ZENTINEL
**Status:** Not Started
**Created:** 2026-03-09
**Target:** [zentinel-agent-policy](https://github.com/zentinelproxy/zentinel-agent-policy)

**See also:** [TODO-pandoc.md](TODO-pandoc.md) — primary north-star target (Pandoc),
[ROADMAP.md](ROADMAP.md) — milestone plan.

---

## Goal

Compile and run [zentinel-agent-policy](https://github.com/zentinelproxy/zentinel-agent-policy),
a ~1.5k LOC Haskell policy evaluation agent for the Zentinel reverse proxy. It supports
Cedar and Rego/OPA policy languages, communicates over Unix sockets, and uses STM-based
caching — a realistic mid-size Haskell service.

This is a **secondary north-star** behind Pandoc. Unlike Pandoc (~60k LOC, ~80 transitive
deps), zentinel-agent-policy is small enough to be tractable sooner while still exercising
real-world patterns: records, typeclasses, deriving, existential types, STM, JSON, CLI
parsing, process spawning, and Unix socket networking.

---

## Project Profile

| Property | Value |
|----------|-------|
| **LOC** | ~1,500 Haskell |
| **Modules** | 10 exposed + 1 executable + 4 test modules |
| **Language** | GHC2021 |
| **Direct deps** | 15 packages |
| **Build system** | Cabal 3.0 |

### Source Modules

```
Zentinel.Agent.Policy              -- Re-export facade
Zentinel.Agent.Policy.Types        -- Core ADTs (Decision, PolicyInput, AuditEntry, etc.)
Zentinel.Agent.Policy.Config       -- YAML config loading, CLI option parsing
Zentinel.Agent.Policy.Engine       -- Typeclass + existential wrapper (SomeEngine)
Zentinel.Agent.Policy.Cedar        -- Cedar policy engine (shells out to cedar CLI)
Zentinel.Agent.Policy.Rego         -- Rego/OPA policy engine (shells out to opa CLI)
Zentinel.Agent.Policy.Handler      -- PolicyAgent: init, request handling, metrics
Zentinel.Agent.Policy.Cache        -- LRU decision cache with STM
Zentinel.Agent.Policy.Input        -- Extract principal/action/resource from HTTP request
Zentinel.Agent.Policy.Protocol     -- Unix socket server, length-prefixed JSON protocol
app/Main.hs                        -- CLI entry point
```

### Extensions Used

All via `common extensions` stanza (project-wide):

- OverloadedStrings ✅ (BHC supports)
- DeriveGeneric ✅
- DeriveAnyClass ✅
- DerivingStrategies ✅
- GeneralizedNewtypeDeriving ✅
- LambdaCase ✅
- RecordWildCards ✅
- TypeApplications ✅
- StrictData ✅
- ImportQualifiedPost ✅
- ScopedTypeVariables ✅

Per-module:
- DuplicateRecordFields ✅ (Types.hs)

### Dependencies

| Package | Version | BHC Equivalent | Status |
|---------|---------|----------------|--------|
| base | >=4.18 | bhc-prelude + bhc-base | ✅ Partial |
| text | >=2.0 | bhc-text (H26.Text) | ✅ Supported |
| bytestring | >=0.11 | bhc-text (H26.Bytes) | ✅ Supported |
| containers | >=0.6 | bhc-containers | ✅ Supported |
| mtl | >=2.3 | bhc-transformers | ✅ Supported |
| stm | >=2.5 | bhc-concurrent | ✅ Supported |
| time | >=1.12 | bhc-utils (H26.Time) | ✅ Supported |
| temporary | >=1.3 | bhc-system | ✅ Supported |
| vector | >=0.13 | bhc-numeric | ⚠️ Partial |
| aeson | >=2.1 | — | ❌ Missing |
| hashable | >=1.4 | — | ❌ Missing |
| network | >=3.1 | — | ❌ Missing |
| optparse-applicative | >=0.18 | — | ❌ Missing |
| process | >=1.6 | — | ❌ Missing |
| unordered-containers | >=0.2 | — | ❌ Missing |
| yaml | >=0.11 | — | ❌ Missing |

Test-only deps: hspec, QuickCheck, directory (not required for compilation target).

---

## Gap Analysis

### What Already Works

BHC can already handle a large portion of the language features this project uses:

- **All 12 extensions** listed above are supported
- **GHC2021 edition** parsing works
- **Record types** with strict fields, named field access, RecordWildCards
- **Pattern matching** on ADT constructors (Decision, PolicyEngine, PolicySource, etc.)
- **Typeclass instances** with dictionary passing
- **DeriveGeneric + DeriveAnyClass** for user-defined classes
- **STM** (TVar, atomically, retry, orElse) via bhc-concurrent
- **Monad transformers** (ReaderT, StateT, ExceptT) via bhc-transformers
- **Data.Map.Strict**, Data.Text, Data.ByteString via stdlib
- **Qualified imports** (ImportQualifiedPost)
- **do-notation** with dictionary dispatch

### What's Missing

Organized by blocker severity, from hardest to easiest.

---

## Tier 0 — Language Feature Gaps

These are compiler-level features that must exist before any library or package
work can help.

### 0.1 Existential Quantification

**Status:** ❌ Not implemented
**Scope:** Medium-Large
**Impact:** Blocker — zentinel uses `SomeEngine` existential wrapper

The `Engine` module defines a typeclass `Engine e` and wraps concrete engine
types in an existential `data SomeEngine = forall e. Engine e => SomeEngine e`
to allow uniform handling of Cedar and Rego engines.

Required:
- [ ] Parse `forall` in data constructor contexts (`data T = forall a. C a => MkT a`)
- [ ] Type check existential construction (pack type + dictionary)
- [ ] Type check existential elimination (case match unpacks type + dictionary)
- [ ] Codegen: existential values carry dictionary pointer alongside payload
- [ ] Handle `GADTs`-style syntax for existentials (`data T where MkT :: C a => a -> T`)

**Workaround:** Could refactor zentinel to use a plain ADT instead:
```haskell
data EngineImpl = CedarImpl CedarEngine | RegoImpl RegoEngine
```
But this defeats the purpose of testing real-world code as-is.

**Key files (to modify):**
- `crates/bhc-parser/src/decl.rs` — data declaration parsing
- `crates/bhc-typeck/src/infer.rs` — existential pack/unpack
- `crates/bhc-codegen/src/llvm/lower.rs` — existential representation

### 0.2 `RankNTypes` (Rank-2 polymorphism)

**Status:** ❌ Not implemented
**Scope:** Medium
**Impact:** Likely blocker — many Hackage packages use `forall` in function arguments

While zentinel's own code may not use rank-2 types directly, its dependencies
(aeson, optparse-applicative) use them internally. Any path through Hackage
compilation requires at minimum rank-2 support.

Required:
- [ ] Parse `forall` in function argument positions
- [ ] Type check higher-rank types (subsumption, instantiation)
- [ ] Codegen: pass polymorphic functions as closures with implicit type args

### 0.3 `RecordDotSyntax` / `HasField` (optional)

**Status:** ❌ Not implemented
**Scope:** Small
**Impact:** Non-blocker — zentinel uses traditional record access, not dot syntax

Listed for completeness. Not required for this target.

---

## Tier 1 — Package Ecosystem

These are packages that zentinel depends on which BHC cannot currently compile
or provide. Ordered by criticality.

### 1.1 `aeson` — JSON serialization

**Status:** ❌ No package
**Scope:** Large
**Impact:** Hard blocker — every module uses `FromJSON`/`ToJSON`

zentinel derives `FromJSON` and `ToJSON` for all its core types via
`DeriveAnyClass` + `DeriveGeneric`. The protocol module serializes/deserializes
JSON over the Unix socket.

Options:
1. **Compile aeson from Hackage** — requires full package system + many transitive
   deps (attoparsec/parsec, scientific, text-short, th-abstraction or Generics)
2. **Provide BHC-native JSON** — H26.JSON already has `FromJSON`/`ToJSON` classes.
   Need to verify API compatibility or provide an aeson-compatibility shim.
3. **Ship a minimal `aeson` reimplementation** as a BHC builtin package that uses
   GHC.Generics for deriving (no Template Haskell needed)

Recommended: Option 3. Build a minimal `aeson`-compatible package in BHC's stdlib
that exports `Data.Aeson` with `FromJSON`, `ToJSON`, `Value`, `encode`, `decode`,
`eitherDecode`, and Generic-based default methods. This avoids the massive
transitive dependency tree of real aeson.

- [ ] `Data.Aeson` module with `Value` type (Object, Array, String, Number, Bool, Null)
- [ ] `FromJSON` / `ToJSON` typeclasses with `parseJSON` / `toJSON`
- [ ] Generic-based default instances (via GHC.Generics `Rep`)
- [ ] `encode :: ToJSON a => a -> ByteString`
- [ ] `decode :: FromJSON a => ByteString -> Maybe a`
- [ ] `eitherDecode :: FromJSON a => ByteString -> Either String a`
- [ ] JSON parser (recursive descent, not attoparsec-dependent)
- [ ] JSON builder/serializer
- [ ] `(.=)`, `(.:)`, `(.:?)`, `object`, `withObject`, `withText` combinators

### 1.2 `network` — Unix socket communication

**Status:** ❌ No package
**Scope:** Medium-Large
**Impact:** Hard blocker — Protocol module creates Unix socket server

zentinel's `Protocol` module:
- Creates a Unix domain socket
- Listens for connections
- Forks a handler per connection (`forkIO`)
- Reads/writes length-prefixed JSON messages

Options:
1. **Compile network from Hackage** — C FFI heavy, platform-specific
2. **Provide BHC-native network** — FFI wrappers around POSIX socket API
3. **Minimal Unix socket support** in bhc-system

Recommended: Option 3. Provide minimal socket FFI in bhc-system:

- [ ] `socket`, `bind`, `listen`, `accept`, `connect` FFI wrappers
- [ ] `AF_UNIX` / `SOCK_STREAM` constants
- [ ] `Network.Socket` compatibility shim (Socket type, SockAddr, etc.)
- [ ] `recv` / `send` / `sendAll` for ByteString
- [ ] `close` with exception safety

### 1.3 `process` — External CLI execution

**Status:** ❌ No package
**Scope:** Medium
**Impact:** Hard blocker — Cedar and Rego engines shell out to CLI tools

Both `Cedar.hs` and `Rego.hs` use `System.Process` to:
- Create a process (`createProcess` or `readProcessWithExitCode`)
- Capture stdout/stderr
- Check exit codes

Options:
1. **Compile process from Hackage** — moderate FFI, depends on base internals
2. **Provide BHC-native process** — FFI wrappers around `fork`/`exec`/`posix_spawn`

Recommended: Option 2. Minimal process support in bhc-system:

- [ ] `System.Process` compatibility: `readProcessWithExitCode`
- [ ] `createProcess` with `CreateProcess` record (stdin/stdout/stderr handles)
- [ ] `waitForProcess` / `ExitCode` type
- [ ] `proc` / `shell` helpers
- [ ] Pipe stdin/stdout/stderr to handles

### 1.4 `optparse-applicative` — CLI argument parsing

**Status:** ❌ No package
**Scope:** Medium
**Impact:** Blocker for executable, not for library

`app/Main.hs` and `Config.hs` use optparse-applicative for CLI parsing
(`Parser`, `execParser`, `info`, `strOption`, `long`, `metavar`, `value`, etc.).

Options:
1. **Compile from Hackage** — moderate dep tree (transformers, process, ansi-wchar-compat)
2. **Provide BHC-native alternative** — simpler arg parser
3. **Defer** — compile the library only, not the executable

Recommended: Option 3 initially (compile library modules only), then option 2
(simple argument parser) for full executable support.

- [ ] Decide on approach (defer executable or implement arg parser)
- [ ] If implementing: `Options.Applicative` shim with `Parser`, `execParser`,
      `strOption`, `long`, `short`, `metavar`, `value`, `help`, `info`, `fullDesc`

### 1.5 `yaml` — YAML configuration parsing

**Status:** ❌ No package
**Scope:** Medium
**Impact:** Blocker — config loading uses YAML

`Config.hs` loads agent configuration from YAML files via `Data.Yaml`
(`decodeFileEither`, custom `FromJSON` instances).

Options:
1. **Compile yaml from Hackage** — depends on libyaml C library + aeson
2. **Provide minimal YAML parser** — subset parser for config files
3. **Defer** — use JSON config instead (requires code changes)

Recommended: Option 2. A minimal YAML parser that handles the subset used by
zentinel configs (scalar values, maps, lists — no anchors/aliases/tags needed).

- [ ] `Data.Yaml` module with `decodeFileEither`
- [ ] YAML parser for maps, lists, scalars, nested structures
- [ ] Integration with aeson `Value` type (YAML parses to `Value`, then `FromJSON`)

### 1.6 `hashable` — Hashable typeclass

**Status:** ❌ No package
**Scope:** Small
**Impact:** Blocker — `PolicyInput`, `Decision`, etc. derive `Hashable` for cache keys

`Types.hs` derives `Hashable` via `DeriveAnyClass` on multiple types. The cache
module uses `Hashable` to compute cache keys.

Options:
1. **Compile from Hackage** — small dep tree
2. **Provide BHC-native Hashable** — typeclass + Generic-based deriving

Recommended: Option 2. Small, self-contained typeclass:

- [ ] `Data.Hashable` module with `Hashable` class (`hashWithSalt :: Int -> a -> Int`)
- [ ] Instances for base types (Int, Text, ByteString, Maybe, Either, tuples, lists)
- [ ] Generic-based default implementation for `DeriveAnyClass`
- [ ] `hash :: Hashable a => a -> Int` convenience function

### 1.7 `unordered-containers` — HashMap / HashSet

**Status:** ❌ No package
**Scope:** Medium
**Impact:** Moderate — used for some internal mappings

Options:
1. **Compile from Hackage** — depends on hashable
2. **Provide BHC-native HashMap** — HAMT implementation
3. **Substitute with Data.Map** — if performance isn't critical for this target

Recommended: Option 3 initially (zentinel doesn't have hot-path HashMap usage),
then option 2 for full compatibility.

- [ ] Assess whether Data.Map substitution is viable
- [ ] If needed: `Data.HashMap.Strict` with basic operations (insert, lookup, delete, fromList)

---

## Tier 2 — Package System Integration

Even if BHC provides builtin alternatives for zentinel's deps, the build system
must understand `.cabal` files, `common` stanzas, and conditional dependencies.

### 2.1 Cabal File Parsing

**Status:** ⚠️ Infrastructure exists in `bhc-package` + `hx-bhc`, untested end-to-end
**Scope:** Medium
**Impact:** Required for building from source checkout

zentinel's `.cabal` file uses:
- `cabal-version: 3.0`
- `common` stanzas (`warnings`, `extensions`) with `import:`
- `library` / `executable` / `test-suite` sections
- `build-depends` with version ranges
- `default-extensions` lists
- `ghc-options`
- `tested-with`

Required:
- [ ] Parse `common` stanzas and resolve `import:` references
- [ ] Parse `default-extensions` and apply to all modules in the section
- [ ] Parse `ghc-options` (at minimum: `-Wall`, `-threaded`, `-rtsopts`)
- [ ] Map `build-depends` package names to BHC builtins where applicable
- [ ] Handle `hs-source-dirs` for module resolution

### 2.2 Multi-Target Build

**Status:** ❌ Not tested
**Scope:** Medium
**Impact:** Required for building both library and executable

zentinel defines a library, an executable, and a test suite. BHC needs to:
- [ ] Build the library (compile all `exposed-modules` with `-c`, generate `.bhi` files)
- [ ] Build the executable (compile `app/Main.hs`, link against the library)
- [ ] Resolve intra-package dependencies (executable depends on library)

---

## Tier 3 — Runtime Features

### 3.1 Concurrent IO (`forkIO`)

**Status:** ⚠️ bhc-concurrent has structured concurrency, but GHC-style `forkIO` may differ
**Scope:** Small
**Impact:** Required — Protocol module forks a handler per connection

- [ ] Verify `forkIO` compatibility (or map to BHC's `spawn` within an implicit scope)
- [ ] Exception propagation from forked threads

### 3.2 Exception Handling

**Status:** ✅ Largely complete (catch, bracket, finally, onException)
**Scope:** Small
**Impact:** Required — Cedar/Rego engines catch process failures

- [x] `catch`, `bracket`, `finally`, `onException`
- [ ] `SomeException` existential (requires Tier 0.1 — existential types)
- [ ] `IOException` and `IOError` hierarchy
- [ ] `try :: Exception e => IO a -> IO (Either e a)`

### 3.3 Temporary File Handling

**Status:** ✅ bhc-system supports temporary files
**Scope:** Small
**Impact:** Required — Cedar/Rego write policies to temp files

- [ ] Verify `System.IO.Temp.withSystemTempFile` compatibility
- [ ] Verify `System.IO.Temp.withSystemTempDirectory` compatibility

### 3.4 Time Measurement

**Status:** ✅ bhc-utils has H26.Time
**Scope:** Small
**Impact:** Required — evaluation timing in nanoseconds

- [ ] Verify `Data.Time.Clock.getCurrentTime` compatibility
- [ ] Verify `Data.Time.Clock.diffUTCTime` compatibility
- [ ] `UTCTime` type with `FromJSON`/`ToJSON` instances

---

## Compilation Strategy

### Phase A: Library Type-Check (`bhc check`)

Get `bhc check` to succeed on all 10 library modules. This validates parsing,
name resolution, and type checking without requiring codegen or runtime.

**Prerequisites:** Tier 0.1 (existential types), Tier 1.1 (aeson shim),
Tier 1.6 (hashable shim)

**Approach:**
1. Provide stub packages for aeson, hashable, network, process, yaml,
   optparse-applicative, unordered-containers — modules that export the right
   types and function signatures but may have `undefined` implementations
2. Run `bhc check` on each module in dependency order
3. Fix parser/typechecker issues as they surface

### Phase B: Library Compilation (`bhc -c`)

Compile all library modules to object files. This exercises codegen for all the
patterns used: records, typeclasses, STM, pattern matching, existentials.

**Prerequisites:** Phase A complete, codegen support for existentials

**Approach:**
1. Compile modules in dependency order with `-c`
2. Generate `.bhi` interface files for cross-module imports
3. Verify all modules produce valid object files

### Phase C: Executable Linking

Link the executable, which requires all runtime features to be functional.

**Prerequisites:** Phase B complete, Tier 1.2 (network), Tier 1.3 (process),
Tier 1.4 (optparse-applicative or alternative), Tier 1.5 (yaml)

### Phase D: Runtime Verification

Run the executable and verify it handles policy evaluation correctly.

**Prerequisites:** Phase C complete, Tier 3 runtime features, cedar/opa CLIs
installed

---

## Overlap with Pandoc Target

Many gaps identified here are shared with the Pandoc target. Work on zentinel
directly advances Pandoc readiness:

| Gap | Zentinel | Pandoc | Notes |
|-----|----------|--------|-------|
| Existential types | ✅ Required | ✅ Required | Same compiler work |
| RankNTypes | ⚠️ Via deps | ✅ Required | Same compiler work |
| aeson | ✅ Required | ✅ Required | Same package |
| network | ✅ Required | ❌ Not needed | zentinel-specific |
| process | ✅ Required | ✅ Required | Same package |
| optparse-applicative | ✅ Required | ✅ Required | Same package |
| hashable | ✅ Required | ✅ Required | Same package |
| unordered-containers | ✅ Required | ✅ Required | Same package |
| yaml | ✅ Required | ❌ Not needed | zentinel-specific |
| parsec/megaparsec | ❌ Not needed | ✅ Required | Pandoc-specific |
| Package system | ✅ Required | ✅ Required | Same infrastructure |
| Cabal common stanzas | ✅ Required | ✅ Required | Same parser work |

**Conclusion:** zentinel serves as a stepping stone — it requires a subset of
Pandoc's gaps (minus parsec and the massive dep tree) plus network/yaml. Reaching
zentinel compilation proves the package system, existential types, and core library
shims are ready, significantly de-risking the Pandoc target.
