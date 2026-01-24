# bhc-rts-alloc

Memory allocation primitives for the BHC Runtime System.

## Overview

This crate provides the foundational memory allocation primitives used by the BHC runtime. It defines core abstractions for memory regions as specified in H26-SPEC Section 9: Memory Model.

## Memory Regions

BHC defines three allocation regions:

| Region | Purpose | Allocator |
|--------|---------|-----------|
| Hot Arena | Kernel temporaries | Bump pointer |
| Pinned Heap | FFI/device IO | malloc-style |
| General Heap | Boxed structures | GC-managed |

## Key Types

| Type | Description |
|------|-------------|
| `Alignment` | Alignment requirements |
| `MemoryRegion` | Memory region trait |
| `AllocStats` | Allocation statistics |
| `AllocError` | Allocation error types |

## Usage

### Alignment

```rust
use bhc_rts_alloc::Alignment;

let align = Alignment::Simd256;
assert_eq!(align.as_usize(), 32);

// Check alignment
if is_aligned(ptr, Alignment::CacheLine) {
    // Cache-line aligned access
}
```

### Alignment Constants

| Alignment | Size | Use Case |
|-----------|------|----------|
| `Default` | 8 bytes | General purpose (64-bit) |
| `Simd128` | 16 bytes | SSE, NEON |
| `Simd256` | 32 bytes | AVX |
| `Simd512` | 64 bytes | AVX-512 |
| `CacheLine` | 64 bytes | Cache efficiency |
| `Page` | 4096 bytes | Page-aligned |

### Memory Region Trait

```rust
pub trait MemoryRegion {
    /// Allocate memory with given layout
    fn alloc(&self, layout: Layout) -> AllocResult<NonNull<u8>>;

    /// Deallocate memory
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);

    /// Get statistics
    fn stats(&self) -> AllocStats;
}
```

### Utility Functions

```rust
use bhc_rts_alloc::{align_up, is_aligned};

// Align value up to alignment
let aligned = align_up(17, 16);
assert_eq!(aligned, 32);

// Check if aligned
assert!(is_aligned(32, 16));
assert!(!is_aligned(17, 16));
```

## Allocation Statistics

```rust
pub struct AllocStats {
    /// Total bytes allocated
    pub total_allocated: usize,

    /// Total bytes deallocated
    pub total_deallocated: usize,

    /// Current live bytes
    pub current_live: usize,

    /// Number of allocations
    pub allocation_count: usize,

    /// Number of deallocations
    pub deallocation_count: usize,

    /// Peak memory usage
    pub peak_usage: usize,
}
```

## Error Types

```rust
pub enum AllocError {
    /// Out of memory
    OutOfMemory { requested: usize, available: usize },

    /// Invalid layout (size or alignment)
    InvalidLayout(LayoutError),

    /// Region exhausted
    RegionExhausted { region: &'static str },

    /// Alignment not supported
    UnsupportedAlignment(usize),
}
```

## SIMD-Aligned Allocation

```rust
use bhc_rts_alloc::{alloc_aligned, Alignment};

// Allocate SIMD-aligned buffer
let ptr = alloc_aligned::<f32>(1024, Alignment::Simd256)?;

// Now safe for AVX operations
unsafe {
    let v = _mm256_load_ps(ptr);
}
```

## Design Goals

- Zero-cost abstractions for allocation patterns
- Explicit control over memory placement
- Safe FFI interop through pinned allocations
- Support for SIMD-aligned allocations

## Related Crates

- `bhc-rts` - Core runtime
- `bhc-rts-arena` - Hot arena (uses these primitives)
- `bhc-rts-gc` - General heap GC

## Specification References

- H26-SPEC Section 9: Memory Model
- BHC-RULE-008: Memory Management
