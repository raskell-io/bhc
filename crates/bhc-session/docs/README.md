# bhc-session

Compilation session management for the Basel Haskell Compiler.

## Overview

`bhc-session` provides the central state container for a BHC compilation. It holds:

- **Compilation options**: Optimization level, debug info, output type
- **Profile selection**: Default, Server, Numeric, or Edge
- **Search paths**: Where to find modules and libraries
- **Global configuration**: Target, features, flags

## Core Types

| Type | Description |
|------|-------------|
| `Session` | Central compilation state |
| `Options` | Compilation options |
| `Profile` | Compilation profile |
| `OptLevel` | Optimization level |
| `OutputType` | Output artifact type |
| `SearchPaths` | Module search paths |

## Profiles

BHC supports four compilation profiles per H26-SPEC:

```rust
pub enum Profile {
    /// Lazy evaluation, GC (traditional Haskell)
    Default,
    /// Bounded latency, incremental GC
    Server,
    /// Strict, unboxed, fusion (numeric computing)
    Numeric,
    /// Minimal runtime, WASM-friendly
    Edge,
}
```

### Profile Characteristics

| Profile | Evaluation | GC | Fusion | Target |
|---------|------------|-----|--------|--------|
| Default | Lazy | Mark-sweep | No | General |
| Server | Lazy | Incremental | No | Services |
| Numeric | Strict | Arena | Yes | HPC |
| Edge | Strict | Refcount | No | WASM |

## Options

```rust
pub struct Options {
    /// Optimization level
    pub opt_level: OptLevel,
    /// Debug information
    pub debug_info: DebugInfo,
    /// Output type
    pub output_type: OutputType,
    /// Search paths
    pub search_paths: SearchPaths,
    /// Target triple
    pub target: Option<String>,
    /// Extra flags
    pub flags: Flags,
}
```

### Optimization Levels

```rust
pub enum OptLevel {
    /// No optimization (-O0)
    None,
    /// Basic optimization (-O1)
    Less,
    /// Standard optimization (-O2)
    Default,
    /// Aggressive optimization (-O3)
    Aggressive,
    /// Size optimization (-Os)
    Size,
    /// Minimal size (-Oz)
    MinSize,
}
```

### Debug Information

```rust
pub enum DebugInfo {
    /// No debug info
    None,
    /// Line tables only
    LineTablesOnly,
    /// Full debug info
    Full,
}
```

### Output Types

```rust
pub enum OutputType {
    /// Object file (.o)
    Object,
    /// Assembly (.s)
    Assembly,
    /// LLVM IR (.ll)
    LlvmIr,
    /// LLVM bitcode (.bc)
    LlvmBc,
    /// Executable
    Executable,
    /// Static library (.a)
    StaticLib,
    /// Dynamic library (.so/.dylib/.dll)
    DynamicLib,
}
```

## Search Paths

```rust
pub struct SearchPaths {
    /// Module search paths
    pub modules: Vec<PathBuf>,
    /// Library search paths
    pub libraries: Vec<PathBuf>,
    /// Include paths
    pub includes: Vec<PathBuf>,
}

impl SearchPaths {
    pub fn find_module(&self, name: &str) -> Option<PathBuf>;
    pub fn find_library(&self, name: &str) -> Option<PathBuf>;
}
```

## Session

The `Session` holds all compilation state:

```rust
pub struct Session {
    /// Compilation options
    pub opts: Options,
    /// Selected profile
    pub profile: Profile,
    /// Source map for error reporting
    pub source_map: SourceMap,
    /// Diagnostic handler
    pub diag_handler: DiagnosticHandler,
    /// Interned strings
    pub interner: Interner,
}

impl Session {
    pub fn new(opts: Options, profile: Profile) -> Self;

    /// Check if numeric profile features are enabled
    pub fn is_numeric(&self) -> bool {
        matches!(self.profile, Profile::Numeric)
    }

    /// Check if strict evaluation is enabled
    pub fn is_strict(&self) -> bool {
        matches!(self.profile, Profile::Numeric | Profile::Edge)
    }
}
```

## Flags

Additional compilation flags:

```rust
pub struct Flags {
    /// Enable bounds checking
    pub bounds_check: bool,
    /// Enable overflow checking
    pub overflow_check: bool,
    /// Enable assertions
    pub assertions: bool,
    /// Dump intermediate IR
    pub dump_ir: Option<DumpIr>,
    /// Verbose output
    pub verbose: bool,
}

pub enum DumpIr {
    Hir,
    Core,
    Tensor,
    Loop,
    Llvm,
}
```

## Quick Start

```rust
use bhc_session::{Session, Options, Profile, OptLevel};

// Create session with options
let opts = Options {
    opt_level: OptLevel::Default,
    ..Default::default()
};

let session = Session::new(opts, Profile::Numeric);

// Check profile
if session.is_numeric() {
    // Enable tensor IR pipeline
}

// Find a module
if let Some(path) = session.opts.search_paths.find_module("Data.Vector") {
    // Load module from path
}
```

## Profile-Specific Behavior

```rust
impl Session {
    /// Get evaluation strategy for this profile
    pub fn eval_strategy(&self) -> EvalStrategy {
        match self.profile {
            Profile::Default | Profile::Server => EvalStrategy::Lazy,
            Profile::Numeric | Profile::Edge => EvalStrategy::Strict,
        }
    }

    /// Check if fusion is enabled
    pub fn fusion_enabled(&self) -> bool {
        matches!(self.profile, Profile::Numeric)
    }

    /// Get GC strategy
    pub fn gc_strategy(&self) -> GcStrategy {
        match self.profile {
            Profile::Default => GcStrategy::MarkSweep,
            Profile::Server => GcStrategy::Incremental,
            Profile::Numeric => GcStrategy::Arena,
            Profile::Edge => GcStrategy::RefCount,
        }
    }
}
```

## See Also

- `bhc-driver`: Uses session for compilation
- `bhc-diagnostics`: Error reporting
- `bhc-query`: Incremental compilation
- H26-SPEC Section 2: Compilation Profiles
