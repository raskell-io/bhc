# bhc-rts

Core Runtime System for the Basel Haskell Compiler.

## Overview

This crate is the main entry point for the BHC Runtime System (RTS). It integrates and coordinates all RTS components: memory management, garbage collection, and concurrency.

## Architecture

```
+-------------------+
|   BHC Compiler    |
+-------------------+
         |
         | generates code that calls
         v
+-------------------+-------------------+
|              bhc-rts                  |
|         (Core RTS - this crate)       |
+-------------------+-------------------+
         |
         +----------+----------+----------+
         |          |          |          |
         v          v          v          v
    +---------+ +--------+ +------+ +-----------+
    |rts-alloc| |rts-arena| |rts-gc| |rts-scheduler|
    +---------+ +--------+ +------+ +-----------+
```

## Key Types

| Type | Description |
|------|-------------|
| `Runtime` | Main runtime instance |
| `RuntimeConfig` | Runtime configuration |
| `Profile` | Execution profile |
| `Value` | Runtime value representation |

## Usage

### Basic Initialization

```rust
use bhc_rts::{Runtime, RuntimeConfig};

fn main() {
    let config = RuntimeConfig::default();
    let runtime = Runtime::new(config);

    runtime.run(|| {
        // Your program here
    });
}
```

### With Custom Configuration

```rust
use bhc_rts::{Runtime, RuntimeConfig, Profile};

let config = RuntimeConfig {
    profile: Profile::Numeric,
    heap_size: 64 * 1024 * 1024,  // 64 MB
    arena_size: 16 * 1024 * 1024, // 16 MB
    worker_threads: 8,
    ..Default::default()
};

let runtime = Runtime::new(config)?;
```

## Profiles

The RTS supports different execution profiles as specified in H26-SPEC:

| Profile | Characteristics |
|---------|-----------------|
| Default | Lazy evaluation, GC managed |
| Server | Bounded latency, incremental GC |
| Numeric | Strict, minimal GC, arena allocation |
| Edge | Minimal footprint, no GC |

### Profile Selection

```rust
use bhc_rts::Profile;

// Runtime behavior varies by profile
let profile = Profile::Numeric;

if profile.is_strict() {
    // Strict evaluation by default
}

if profile.has_gc() {
    // GC is active
}
```

## Memory Regions

The RTS manages three memory regions:

| Region | Allocation | Deallocation | GC |
|--------|------------|--------------|-----|
| Hot Arena | Bump pointer | Bulk free | None |
| Pinned Heap | malloc-style | Explicit | Never moved |
| General Heap | GC-managed | Automatic | May move |

## RTS Entry Points

The RTS exposes entry points for compiled code:

```rust
// Allocation
extern "C" fn bhc_alloc(size: usize) -> *mut u8;
extern "C" fn bhc_alloc_pinned(size: usize) -> *mut u8;
extern "C" fn bhc_arena_alloc(size: usize) -> *mut u8;

// Thunks
extern "C" fn bhc_update_thunk(thunk: *mut u8, value: *mut u8);
extern "C" fn bhc_enter_thunk(thunk: *mut u8) -> *mut u8;

// GC
extern "C" fn bhc_gc_safe_point();
extern "C" fn bhc_gc_root_push(ptr: *mut u8);
extern "C" fn bhc_gc_root_pop();
```

## Initialization Sequence

1. Parse runtime options
2. Initialize memory allocator
3. Initialize GC (if applicable)
4. Initialize scheduler (if multi-threaded)
5. Initialize I/O subsystem
6. Call main function
7. Run finalizers
8. Report statistics

## Design Notes

- Profile affects all subsystem behavior
- Memory layout is cache-aware
- GC safe points are inserted by compiler
- FFI calls require pinned memory

## Related Crates

- `bhc-rts-alloc` - Memory allocation primitives
- `bhc-rts-arena` - Hot arena allocator
- `bhc-rts-gc` - Garbage collector
- `bhc-rts-scheduler` - Task scheduler

## Specification References

- H26-SPEC Section 9: Memory Model
- H26-SPEC Section 10: Concurrency Model
- H26-SPEC Section 2: Runtime Profiles
