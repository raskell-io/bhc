# bhc-ffi

Foreign Function Interface support for the Basel Haskell Compiler.

## Overview

`bhc-ffi` enables safe interoperation with external C libraries such as BLAS implementations. Features:

- **PinnedBuffer**: Buffers guaranteed not to move during GC
- **BLAS Provider**: Trait-based abstraction for BLAS backends
- **Safe FFI boundary**: `with_pinned_tensor` pattern ensures data validity
- **Alignment checking**: Verify pointer alignment requirements

## FFI Safety Model

```
┌────────────────────────────────────────────────────────────────┐
│                    Haskell / BHC Runtime                       │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐                                              │
│  │   Tensor     │  ──(pin)──>  ┌──────────────────┐           │
│  │  (may move)  │              │  PinnedBuffer     │           │
│  └──────────────┘              │  (never moves)    │           │
│                                └────────┬─────────┘           │
│                                         │                      │
├─────────────────────────────────────────┼──────────────────────┤
│                      FFI Boundary       │                      │
├─────────────────────────────────────────┼──────────────────────┤
│                                         │                      │
│  ┌──────────────────────────────────────▼───────────────────┐ │
│  │                   C Library (BLAS)                        │ │
│  │  - Receives raw pointer                                   │ │
│  │  - Pointer remains valid for call duration                │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

## Core Types

| Type | Description |
|------|-------------|
| `PinnedBuffer` | Buffer that won't be moved by GC |
| `PinnedSlice` | View into pinned buffer |
| `BlasProvider` | BLAS operation abstraction |
| `FfiSafe` | Marker trait for FFI-safe types |

## FfiSafe Trait

Types safe for FFI must implement `FfiSafe`:

```rust
/// Marker trait for types safe to pass to FFI.
///
/// # Safety
/// - Must be `Copy` with no drop glue
/// - Must have a well-defined C ABI layout
/// - Must not contain pointers to GC memory
pub unsafe trait FfiSafe: Copy + 'static {
    const C_TYPE_NAME: &'static str;
}

// Implemented for primitives
unsafe impl FfiSafe for f32 { const C_TYPE_NAME: &'static str = "float"; }
unsafe impl FfiSafe for f64 { const C_TYPE_NAME: &'static str = "double"; }
unsafe impl FfiSafe for i32 { const C_TYPE_NAME: &'static str = "int32_t"; }
unsafe impl FfiSafe for u64 { const C_TYPE_NAME: &'static str = "uint64_t"; }
```

## Pinned Buffers

```rust
use bhc_ffi::{with_pinned, with_pinned_mut};

// Immutable access
with_pinned(&tensor, |ptr, len| {
    unsafe { c_read_only_function(ptr, len) }
});

// Mutable access
with_pinned_mut(&mut tensor, |ptr, len| {
    unsafe { c_mutating_function(ptr, len) }
});
```

## BLAS Provider

```rust
use bhc_ffi::{BlasProvider, default_provider, Transpose};

let blas = default_provider();

// DGEMM: C = alpha * A * B + beta * C
blas.dgemm(
    Transpose::None,  // trans_a
    Transpose::None,  // trans_b
    m, n, k,          // dimensions
    1.0,              // alpha
    a_ptr, lda,       // A matrix
    b_ptr, ldb,       // B matrix
    0.0,              // beta
    c_ptr, ldc,       // C matrix (output)
)?;
```

## Matrix Operations

High-level matrix operations:

```rust
use bhc_ffi::{matmul, smatmul, dot, sdot, Matrix};

// Double-precision matrix multiply
let c = matmul(&a, &b)?;

// Single-precision matrix multiply
let c = smatmul(&a, &b)?;

// Double-precision dot product
let result = dot(&x, &y)?;

// Single-precision dot product
let result = sdot(&x, &y)?;
```

## Alignment Checking

```rust
use bhc_ffi::{is_aligned, is_aligned_to};

// Check alignment for a type
let data: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
assert!(is_aligned(data.as_ptr()));

// Check specific alignment
let ptr: *const u8 = ...;
assert!(is_aligned_to(ptr, 16)); // 16-byte aligned
```

## Error Handling

```rust
pub enum FfiError {
    /// Memory allocation failed
    AllocationFailed(String),
    /// Buffer not pinned
    NotPinned,
    /// Null pointer
    NullPointer,
    /// Size mismatch
    SizeMismatch { expected: usize, actual: usize },
    /// Alignment error
    AlignmentError { ptr: *const u8, required: usize },
    /// BLAS error
    BlasError(BlasError),
    /// Foreign function error code
    ForeignError(i32),
}
```

## Submodules

| Module | Description |
|--------|-------------|
| `blas` | BLAS provider abstraction |
| `pinned` | Pinned buffer implementation |
| `tensor` | High-level tensor operations |

## M4 Exit Criteria

- `matmul` can call external BLAS for large sizes
- Tensors stay pinned across FFI calls (verified by address stability)
- No GC movement of pinned allocations (stress test)

## See Also

- `bhc-rts`: Runtime system (GC integration)
- `bhc-tensor-ir`: Tensor operations that use FFI
- OpenBLAS documentation
- Intel MKL documentation
