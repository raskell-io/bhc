# bhc-rts-arena

Hot arena allocator for the BHC Runtime System.

## Overview

This crate implements the Hot Arena memory region as specified in H26-SPEC Section 9: Memory Model. The hot arena provides O(1) allocation via pointer bumping with scope-based lifetime management.

## Features

- **Bump allocation** - O(1) allocation via pointer bumping
- **Scope-based lifetime** - All allocations freed when arena scope ends
- **Zero GC interaction** - Arena memory is invisible to the garbage collector
- **Cache-friendly** - Sequential allocation pattern

## Key Types

| Type | Description |
|------|-------------|
| `HotArena` | Main arena allocator |
| `ArenaScope` | Scoped arena lifetime |
| `ArenaSlice<T>` | Slice allocated in arena |

## Usage

### Basic Arena Allocation

```rust
use bhc_rts_arena::{HotArena, with_arena};

// All allocations are freed when the arena scope ends
with_arena(1024 * 1024, |arena| {
    let buffer = arena.alloc_slice::<f32>(1000);
    compute_kernel(buffer);
    // buffer automatically freed here
});
```

### Manual Arena Management

```rust
use bhc_rts_arena::HotArena;

let mut arena = HotArena::new(DEFAULT_ARENA_SIZE)?;

// Allocate
let ptr = arena.alloc(size, align)?;

// Use the memory...

// Reset (free all allocations, reuse memory)
arena.reset();
```

### Nested Scopes

```rust
with_arena(size, |outer| {
    let a = outer.alloc_slice::<f32>(100);

    with_arena(size, |inner| {
        let b = inner.alloc_slice::<f32>(100);
        compute(a, b);
        // b freed here
    });

    // a still valid
});
```

## Allocation API

```rust
impl HotArena {
    /// Allocate raw bytes with alignment
    pub fn alloc(&self, size: usize, align: usize) -> AllocResult<NonNull<u8>>;

    /// Allocate a slice of T
    pub fn alloc_slice<T>(&self, len: usize) -> AllocResult<&mut [T]>;

    /// Allocate a single value
    pub fn alloc_value<T>(&self, value: T) -> AllocResult<&mut T>;

    /// Reset arena (free all allocations)
    pub fn reset(&mut self);

    /// Get current usage
    pub fn used(&self) -> usize;

    /// Get remaining capacity
    pub fn remaining(&self) -> usize;
}
```

## Alignment Support

```rust
use bhc_rts_arena::HotArena;
use bhc_rts_alloc::Alignment;

// SIMD-aligned allocation
let buffer = arena.alloc_aligned::<f32>(1024, Alignment::Simd256)?;
```

| Alignment | Size | Use Case |
|-----------|------|----------|
| Default | 8 bytes | General purpose |
| Simd128 | 16 bytes | SSE vectors |
| Simd256 | 32 bytes | AVX vectors |
| Simd512 | 64 bytes | AVX-512 vectors |
| CacheLine | 64 bytes | Cache efficiency |

## Thread Safety

Arenas are thread-local by default:

```rust
// Each thread gets its own arena
thread_local! {
    static ARENA: RefCell<HotArena> = RefCell::new(
        HotArena::new(DEFAULT_ARENA_SIZE).unwrap()
    );
}
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Allocate | O(1) | Pointer bump |
| Reset | O(1) | Reset cursor |
| Free individual | N/A | Not supported |

## Integration with Numeric Kernels

```rust
// Kernel uses arena for temporaries
fn matrix_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    with_arena(temp_size(a, b), |arena| {
        // Intermediate results in arena
        let tmp = arena.alloc_slice::<f32>(a.rows * b.cols);

        // Compute with temporary buffer
        compute_matmul(a, b, tmp);

        // Copy result out of arena
        Matrix::from_slice(tmp)
    })
}
```

## Design Goals

- Minimal allocation overhead for hot paths
- No fragmentation within a scope
- Deterministic memory usage
- Cache-friendly sequential allocation

## Related Crates

- `bhc-rts` - Core runtime
- `bhc-rts-alloc` - Allocation primitives
- `bhc-rts-gc` - General heap GC

## Specification References

- H26-SPEC Section 9: Memory Model
- H26-SPEC Section 9.1: Hot Arena
