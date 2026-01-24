# bhc-rts

Core Runtime System for the Basel Haskell Compiler.

## Overview

The BHC Runtime System (RTS) is the execution environment for compiled Haskell programs. It manages memory, garbage collection, thunk evaluation, and concurrency. This crate integrates all RTS components and provides the entry point for compiled programs.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Compiled BHC Program                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    bhc-rts (this crate)              │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ Memory  │ │   GC    │ │ Thunks  │ │Scheduler│   │   │
│  │  │  Mgmt   │ │         │ │         │ │         │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           ▼               ▼               ▼                 │
│    ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│    │ rts-alloc │   │  rts-gc   │   │rts-scheduler│          │
│    │           │   │           │   │             │          │
│    └───────────┘   └───────────┘   └───────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Memory Regions

Per H26-SPEC Section 9, BHC uses three memory regions:

| Region | Purpose | Allocator | GC |
|--------|---------|-----------|-----|
| Hot Arena | Kernel temporaries | Bump pointer | None |
| Pinned Heap | FFI/device IO | malloc-style | Never moved |
| General Heap | Boxed values | GC-managed | May move |

### Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                        Address Space                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                        │
│  │   Hot Arena     │  Fast bump allocation, scope-based     │
│  │   (per-thread)  │  lifetime, zero GC interaction         │
│  └─────────────────┘                                        │
│  ┌─────────────────┐                                        │
│  │   Pinned Heap   │  Non-moving, explicit management,      │
│  │                 │  FFI-safe pointers                     │
│  └─────────────────┘                                        │
│  ┌─────────────────┐                                        │
│  │  General Heap   │  GC-managed, generational,             │
│  │   (Nursery)     │  objects may be moved                  │
│  ├─────────────────┤                                        │
│  │   (Survivor)    │                                        │
│  ├─────────────────┤                                        │
│  │   (Old Gen)     │                                        │
│  └─────────────────┘                                        │
│  ┌─────────────────┐                                        │
│  │     Stack       │  Native stack for execution            │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

## Runtime Profiles

| Profile | Evaluation | GC | Arena | Threading |
|---------|------------|-----|-------|-----------|
| Default | Lazy | Standard gen | Available | Green threads |
| Server | Lazy | Incremental | Available | Work-stealing |
| Numeric | Strict | Minimal | Primary | Parallel loops |
| Edge | Lazy | Minimal | Limited | Single-threaded |
| Realtime | Lazy | Bounded | Available | Prioritized |
| Embedded | Strict | None | Only | Single-threaded |

## RTS Entry Points

The RTS exports C-ABI functions called by compiled code:

### Allocation

```rust
/// Allocate in general heap
#[no_mangle]
pub extern "C" fn bhc_alloc(size: usize) -> *mut u8;

/// Allocate pinned memory (will not move)
#[no_mangle]
pub extern "C" fn bhc_alloc_pinned(size: usize) -> *mut u8;

/// Allocate in hot arena (thread-local)
#[no_mangle]
pub extern "C" fn bhc_arena_alloc(size: usize) -> *mut u8;

/// Free pinned memory
#[no_mangle]
pub extern "C" fn bhc_free_pinned(ptr: *mut u8);
```

### Thunk Operations

```rust
/// Enter a thunk (force evaluation)
#[no_mangle]
pub extern "C" fn bhc_enter_thunk(thunk: *mut Thunk) -> *mut Value;

/// Update a thunk with its value
#[no_mangle]
pub extern "C" fn bhc_update_thunk(thunk: *mut Thunk, value: *mut Value);

/// Create a new thunk
#[no_mangle]
pub extern "C" fn bhc_new_thunk(code: *const u8, env: *mut u8) -> *mut Thunk;
```

### GC Integration

```rust
/// GC safe point (may trigger collection)
#[no_mangle]
pub extern "C" fn bhc_gc_safe_point();

/// Push a GC root
#[no_mangle]
pub extern "C" fn bhc_gc_root_push(ptr: *mut u8);

/// Pop a GC root
#[no_mangle]
pub extern "C" fn bhc_gc_root_pop();

/// Request garbage collection
#[no_mangle]
pub extern "C" fn bhc_gc_collect();
```

### I/O Primitives

```rust
/// Print string to stdout
#[no_mangle]
pub extern "C" fn bhc_print_string(ptr: *const u8, len: usize);

/// Read line from stdin
#[no_mangle]
pub extern "C" fn bhc_read_line() -> *mut String;

/// Get program arguments
#[no_mangle]
pub extern "C" fn bhc_get_args() -> *mut Vec<String>;
```

## Initialization Sequence

```rust
pub fn init_runtime(config: RuntimeConfig) -> Result<Runtime> {
    // 1. Parse runtime options (from env/args)
    let opts = RuntimeOptions::from_env_and_args()?;

    // 2. Initialize memory subsystem
    let memory = MemoryManager::init(&opts)?;

    // 3. Initialize GC (if enabled)
    let gc = if config.profile.has_gc() {
        Some(GC::init(&opts, &memory)?)
    } else {
        None
    };

    // 4. Initialize scheduler (if multi-threaded)
    let scheduler = if config.profile.is_concurrent() {
        Some(Scheduler::init(&opts)?)
    } else {
        None
    };

    // 5. Initialize I/O subsystem
    let io = IoSubsystem::init()?;

    Ok(Runtime {
        memory,
        gc,
        scheduler,
        io,
        config,
    })
}
```

## Shutdown Sequence

```rust
pub fn shutdown_runtime(runtime: Runtime) {
    // 1. Cancel pending tasks
    if let Some(scheduler) = runtime.scheduler {
        scheduler.shutdown();
    }

    // 2. Run finalizers
    runtime.run_finalizers();

    // 3. Report statistics (if requested)
    if runtime.config.print_stats {
        runtime.print_statistics();
    }

    // 4. Release memory
    runtime.memory.release();
}
```

## Runtime Configuration

```rust
pub struct RuntimeConfig {
    /// Execution profile
    pub profile: Profile,

    /// Initial heap size
    pub heap_size: usize,

    /// Maximum heap size
    pub max_heap_size: usize,

    /// Hot arena size (per thread)
    pub arena_size: usize,

    /// Number of worker threads
    pub worker_threads: usize,

    /// GC configuration
    pub gc_config: GcConfig,

    /// Print statistics on exit
    pub print_stats: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Default,
            heap_size: 64 * 1024 * 1024,      // 64 MB
            max_heap_size: 1024 * 1024 * 1024, // 1 GB
            arena_size: 16 * 1024 * 1024,      // 16 MB
            worker_threads: num_cpus::get(),
            gc_config: GcConfig::default(),
            print_stats: false,
        }
    }
}
```

## Statistics

```rust
pub struct RuntimeStats {
    /// Memory statistics
    pub memory: MemoryStats,

    /// GC statistics
    pub gc: GcStats,

    /// Scheduler statistics
    pub scheduler: SchedulerStats,

    /// Total execution time
    pub total_time: Duration,
}

pub struct MemoryStats {
    pub peak_heap_usage: usize,
    pub total_allocated: usize,
    pub total_freed: usize,
    pub arena_peak_usage: usize,
    pub pinned_current: usize,
}
```

## See Also

- `bhc-rts-alloc` - Memory allocation primitives
- `bhc-rts-arena` - Hot arena allocator
- `bhc-rts-gc` - Garbage collector
- `bhc-rts-scheduler` - Task scheduler
