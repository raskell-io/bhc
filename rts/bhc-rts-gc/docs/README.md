# bhc-rts-gc

Generational Garbage Collector for the Basel Haskell Compiler.

## Overview

This crate implements the garbage collector for the General Heap region as specified in H26-SPEC Section 9. It provides generational collection with support for pinned objects, write barriers, and incremental collection for bounded pause times.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     General Heap                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐   Frequent minor GC                    │
│  │  Nursery (G0)   │   Bump allocation                      │
│  │   Young objs    │   Copy to survivor on survival         │
│  └─────────────────┘                                        │
│           │                                                  │
│           │ promote                                          │
│           ▼                                                  │
│  ┌─────────────────┐   Moderate GC frequency                │
│  │  Survivor (G1)  │   Objects survived 1+ collections      │
│  │   Aging objs    │   Promote to old after threshold       │
│  └─────────────────┘                                        │
│           │                                                  │
│           │ promote                                          │
│           ▼                                                  │
│  ┌─────────────────┐   Rare major GC                        │
│  │   Old Gen (G2)  │   Long-lived objects                   │
│  │   Tenured objs  │   Mark-sweep or mark-compact           │
│  └─────────────────┘                                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                        │
│  │  Pinned Region  │   Never moved, explicit free           │
│  │   FFI buffers   │   Reference counted                    │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

## GC Algorithms

### Minor GC (Nursery Collection)

Copying collector for young generation:

```rust
fn minor_gc(&mut self) {
    // 1. Scan roots (stack, globals, remembered set)
    let roots = self.collect_roots();

    // 2. Copy live objects to survivor space
    for root in roots {
        self.evacuate(root);
    }

    // 3. Scavenge (process copied objects)
    while let Some(obj) = self.scavenge_queue.pop() {
        self.scavenge(obj);
    }

    // 4. Flip spaces
    self.nursery.flip();

    // 5. Update statistics
    self.stats.minor_collections += 1;
}
```

### Major GC (Full Collection)

Mark-sweep for old generation:

```rust
fn major_gc(&mut self) {
    // 1. Mark phase
    self.mark_phase();

    // 2. Sweep phase
    self.sweep_phase();

    // 3. Optional compaction
    if self.should_compact() {
        self.compact_phase();
    }

    // 4. Update statistics
    self.stats.major_collections += 1;
}
```

### Incremental GC (Server Profile)

Bounded pause times through incremental marking:

```rust
fn incremental_gc(&mut self, budget_ms: u32) {
    let deadline = Instant::now() + Duration::from_millis(budget_ms as u64);

    // Mark in increments
    while Instant::now() < deadline {
        if let Some(obj) = self.gray_set.pop() {
            self.mark_object(obj);
        } else {
            // Marking complete, do atomic sweep
            self.atomic_sweep();
            break;
        }
    }
}
```

## Write Barriers

Cross-generation references require write barriers:

```rust
/// Called when old object points to young object
#[inline]
pub fn write_barrier(old: *mut Object, young: *mut Object) {
    if generation(old) > generation(young) {
        // Add to remembered set
        REMEMBERED_SET.with(|rs| {
            rs.borrow_mut().insert(old);
        });
    }
}

/// Card marking for remembered set
fn card_mark(ptr: *mut Object) {
    let card_index = ptr as usize >> CARD_SHIFT;
    CARD_TABLE[card_index].store(DIRTY, Ordering::Relaxed);
}
```

## Object Layout

```rust
/// Object header (8 bytes on 64-bit)
#[repr(C)]
pub struct ObjectHeader {
    /// Mark bits + generation + tag
    bits: u64,
}

impl ObjectHeader {
    const MARK_BIT: u64 = 1 << 0;
    const PINNED_BIT: u64 = 1 << 1;
    const FORWARDED_BIT: u64 = 1 << 2;
    const GENERATION_MASK: u64 = 0b11 << 3;
    const TAG_MASK: u64 = 0xFF << 8;
    const SIZE_MASK: u64 = 0xFFFF_FFFF << 16;

    pub fn is_marked(&self) -> bool;
    pub fn set_marked(&mut self);
    pub fn clear_marked(&mut self);
    pub fn generation(&self) -> Generation;
    pub fn is_pinned(&self) -> bool;
    pub fn is_forwarded(&self) -> bool;
    pub fn forwarding_ptr(&self) -> *mut Object;
}
```

## GC Safe Points

The compiler inserts safe points where GC can occur:

```rust
/// Check for pending GC at safe point
#[inline]
pub fn safe_point() {
    if GC_REQUESTED.load(Ordering::Relaxed) {
        // Suspend mutator and wait for GC
        suspend_for_gc();
    }
}
```

Safe points are inserted:
- At function entry (for non-leaf functions)
- At loop back edges
- Before allocations
- At explicit checkpoints

## Configuration

```rust
pub struct GcConfig {
    /// Nursery size
    pub nursery_size: usize,

    /// Survivor space size
    pub survivor_size: usize,

    /// Old generation initial size
    pub old_gen_size: usize,

    /// Promotion threshold (collections to survive)
    pub promotion_threshold: u8,

    /// Enable incremental collection
    pub incremental: bool,

    /// Target maximum pause time (ms)
    pub max_pause_ms: u32,

    /// Enable parallel collection
    pub parallel: bool,

    /// Number of GC threads
    pub gc_threads: usize,
}
```

## Statistics

```rust
pub struct GcStats {
    /// Number of minor collections
    pub minor_collections: u64,

    /// Number of major collections
    pub major_collections: u64,

    /// Total time in GC (microseconds)
    pub total_gc_time_us: u64,

    /// Maximum pause time (microseconds)
    pub max_pause_us: u64,

    /// Bytes promoted to old generation
    pub bytes_promoted: u64,

    /// Bytes reclaimed
    pub bytes_reclaimed: u64,

    /// Current heap size
    pub current_heap_size: usize,
}
```

## Profile-Specific Behavior

| Profile | GC Behavior |
|---------|-------------|
| Default | Standard generational GC |
| Server | Incremental, bounded pauses (<10ms) |
| Numeric | Disabled in kernels, arena allocation |
| Edge | Minimal GC, small heap |
| Realtime | Bounded pauses (<1ms), arena-heavy |
| Embedded | No GC, static allocation |

## Tuning Parameters

```rust
// Default tuning parameters
const DEFAULT_NURSERY_SIZE: usize = 4 * 1024 * 1024;     // 4 MB
const DEFAULT_SURVIVOR_RATIO: usize = 8;                  // 1/8 of nursery
const DEFAULT_OLD_GEN_SIZE: usize = 64 * 1024 * 1024;    // 64 MB
const DEFAULT_PROMOTION_THRESHOLD: u8 = 2;                // 2 survivals
const DEFAULT_MAX_PAUSE_MS: u32 = 10;                     // 10 ms
```

## See Also

- `bhc-rts` - Core runtime
- `bhc-rts-alloc` - Allocation primitives
- `bhc-rts-arena` - Arena allocator (GC-free)
