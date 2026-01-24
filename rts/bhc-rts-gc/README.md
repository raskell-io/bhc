# bhc-rts-gc

Generational garbage collector for the BHC Runtime System.

## Overview

This crate implements the garbage collector for the General Heap region as specified in H26-SPEC Section 9: Memory Model. It provides generational collection with support for pinned objects and incremental collection.

## Architecture

```
+------------------+------------------+------------------+
|   Nursery (G0)   |   Survivor (G1)  |   Old Gen (G2)   |
+------------------+------------------+------------------+
|                  |                  |                  |
|  Young objects   |  Promoted from   |  Long-lived      |
|  Bump alloc      |  G0 after 1      |  objects         |
|  Frequent GC     |  survival        |  Rare major GC   |
|                  |                  |                  |
+------------------+------------------+------------------+

+------------------+
|   Pinned Region  |
+------------------+
|                  |
|  Non-moving      |
|  objects         |
|  (FFI, DMA)      |
|                  |
+------------------+
```

## Key Types

| Type | Description |
|------|-------------|
| `GC` | Main garbage collector |
| `Generation` | A generation in the heap |
| `GcConfig` | GC configuration |
| `GcStats` | Collection statistics |
| `WriteBarrier` | Cross-generation write tracking |

## Usage

### Basic GC Operations

```rust
use bhc_rts_gc::{GC, GcConfig};

let config = GcConfig::default();
let gc = GC::new(config)?;

// Allocate in nursery
let ptr = gc.alloc(size, layout)?;

// Manual collection (usually automatic)
gc.collect_minor()?;
gc.collect_major()?;
```

### Pinned Allocation

```rust
use bhc_rts_gc::GC;

// Allocate pinned (will never move)
let ptr = gc.alloc_pinned(size)?;

// Safe for FFI
unsafe { c_function(ptr) };

// Must explicitly free
gc.free_pinned(ptr)?;
```

### GC Safe Points

```rust
// Compiler inserts safe points in generated code
extern "C" fn bhc_gc_safe_point() {
    if gc_requested() {
        suspend_and_gc();
    }
}
```

## Generations

| Generation | Purpose | Collection |
|------------|---------|------------|
| G0 (Nursery) | New allocations | Frequent (minor GC) |
| G1 (Survivor) | Survived 1 collection | Moderate |
| G2 (Old) | Long-lived objects | Rare (major GC) |
| Pinned | Non-moving | Never collected, explicit free |

## Write Barriers

Cross-generation references are tracked:

```rust
// When old object points to young object
fn write_barrier(old: *mut u8, young: *mut u8) {
    if generation(old) > generation(young) {
        remember_set.add(old);
    }
}
```

## Incremental Collection

For Server Profile, the GC supports incremental collection:

```rust
let config = GcConfig {
    incremental: true,
    max_pause_ms: 10,  // Target pause time
    ..Default::default()
};

// Collection happens in small increments
gc.collect_incremental()?;
```

## Configuration

```rust
pub struct GcConfig {
    /// Nursery size
    pub nursery_size: usize,

    /// Old generation size
    pub old_gen_size: usize,

    /// Enable incremental collection
    pub incremental: bool,

    /// Target maximum pause time (ms)
    pub max_pause_ms: u32,

    /// Promotion threshold (collections to survive)
    pub promotion_threshold: u8,

    /// Enable parallel collection
    pub parallel: bool,
}
```

## Statistics

```rust
let stats = gc.stats();
println!("Collections: minor={}, major={}",
    stats.minor_collections, stats.major_collections);
println!("Total allocated: {} bytes", stats.total_allocated);
println!("Current live: {} bytes", stats.current_live);
println!("Max pause: {} ms", stats.max_pause_ms);
```

## Profile-Specific Behavior

| Profile | GC Behavior |
|---------|-------------|
| Default | Standard generational GC |
| Server | Incremental, bounded pauses |
| Numeric | Disabled in hot paths, arena allocation |
| Edge | No GC, static allocation only |

## Design Goals

- Low latency for Server Profile (bounded pause times)
- High throughput for batch processing
- Deterministic behavior for Numeric Profile
- Safe FFI interop through pinned allocations

## Related Crates

- `bhc-rts` - Core runtime
- `bhc-rts-alloc` - Allocation primitives
- `bhc-rts-arena` - Arena allocation (GC-free)

## Specification References

- H26-SPEC Section 9: Memory Model
- H26-SPEC Section 9.3: General Heap
