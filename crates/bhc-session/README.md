# bhc-session

Compiler session state and configuration for the Basel Haskell Compiler.

## Overview

This crate provides the central session management for the BHC compiler, including compilation options, target specification, diagnostic configuration, and global state that persists throughout a compilation unit.

## Key Types

| Type | Description |
|------|-------------|
| `Session` | Central compilation session state |
| `SessionRef` | Reference-counted session handle |
| `Options` | Compiler options and flags |
| `Profile` | Compilation profile (Default, Server, Numeric, Edge) |
| `OptLevel` | Optimization level |
| `DebugInfo` | Debug information level |
| `OutputType` | Output file type |

## Usage

### Creating a Session

```rust
use bhc_session::{Session, Options, Profile};

let options = Options {
    profile: Profile::Numeric,
    opt_level: OptLevel::Aggressive,
    debug_info: DebugInfo::None,
    output_type: OutputType::Executable,
    ..Default::default()
};

let session = Session::new(options)?;
```

### Working with Profiles

```rust
use bhc_session::Profile;

let profile = Profile::Numeric;

// Check profile characteristics
if profile.is_strict_by_default() {
    println!("Using strict evaluation");
}

if profile.has_fusion_guarantees() {
    println!("Fusion is guaranteed");
}
```

## Profiles

BHC supports multiple compilation profiles as specified in H26-SPEC:

| Profile | Use Case | Key Characteristics |
|---------|----------|---------------------|
| `Default` | General Haskell | Lazy evaluation, GC-managed |
| `Server` | Web services | Structured concurrency, bounded latency |
| `Numeric` | ML, linear algebra | Strict-by-default, fusion guaranteed |
| `Edge` | WASM, serverless | Minimal runtime footprint |

### Profile Selection

```rust
// Command-line equivalent: bhc --profile=numeric
let profile = Profile::Numeric;

// Check strictness
assert!(profile.is_strict_by_default());

// Check fusion
assert!(profile.has_fusion_guarantees());
```

## Optimization Levels

| Level | Description |
|-------|-------------|
| `None` | No optimizations (fastest compilation) |
| `Less` | Basic optimizations |
| `Default` | Standard optimizations |
| `Aggressive` | Maximum optimization |
| `Size` | Optimize for size |
| `SizeMin` | Aggressive size optimization |

## Debug Information

| Level | Description |
|-------|-------------|
| `None` | No debug information |
| `LineTablesOnly` | Line tables only (for profiling) |
| `Full` | Full debug information |

## Output Types

| Type | Description |
|------|-------------|
| `Object` | Object file (.o) |
| `StaticLib` | Static library (.a) |
| `DynamicLib` | Dynamic library (.so/.dylib) |
| `Executable` | Executable binary |
| `Assembly` | Assembly output |
| `LLVMIR` | LLVM IR output |
| `HIR` | HIR dump |
| `Core` | Core IR dump |

## Session Configuration

```rust
use bhc_session::{Session, Options};
use camino::Utf8PathBuf;

let options = Options {
    // Compilation profile
    profile: Profile::Default,

    // Optimization
    opt_level: OptLevel::Default,
    debug_info: DebugInfo::None,

    // Output
    output_type: OutputType::Executable,
    output_path: Some(Utf8PathBuf::from("output")),

    // Search paths
    include_paths: vec![Utf8PathBuf::from("/usr/include")],
    library_paths: vec![Utf8PathBuf::from("/usr/lib")],

    // Warnings
    warn_unused: true,
    warn_incomplete_patterns: true,

    ..Default::default()
};

let session = Session::new(options)?;
```

## Thread Safety

The session is designed for concurrent access:

```rust
use bhc_session::SessionRef;
use std::thread;

let session: SessionRef = Session::new(options)?.into();

let handles: Vec<_> = (0..4).map(|_| {
    let session = session.clone();
    thread::spawn(move || {
        // Each thread can read session options
        let profile = session.options().profile;
        // ...
    })
}).collect();
```

## Design Notes

- Session is immutable after creation for thread safety
- Options are validated during session creation
- Profile affects default values for many options
- Search paths are platform-aware

## Related Crates

- `bhc-driver` - Uses session for compilation orchestration
- `bhc-diagnostics` - Diagnostic handler configuration
- `bhc-target` - Target specification
- `bhc-query` - Query database uses session

## Specification References

- H26-SPEC Section 2: Runtime Profiles
- H26-SPEC Section 6: Strictness Model
