# bhc-ffi

Foreign Function Interface support for the Basel Haskell Compiler.

## Overview

This crate provides FFI support for BHC, enabling safe interoperation with external C libraries such as BLAS implementations. It ensures memory safety across the FFI boundary by managing pinned buffers that won't be moved by the garbage collector.

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

## Key Types

| Type | Description |
|------|-------------|
| `PinnedBuffer` | Buffer guaranteed not to move during GC |
| `PinnedSlice` | Slice view into a pinned buffer |
| `BlasProvider` | Trait for BLAS implementations |
| `Matrix` | FFI-safe matrix representation |

## Usage

### Pinned Buffers

```rust
use bhc_ffi::{PinnedBuffer, with_pinned};

// Create a pinned buffer
let buffer = PinnedBuffer::new(1024)?;

// Safe: buffer guaranteed pinned for duration
with_pinned(&tensor, |ptr, len| {
    // Call C function with raw pointer
    unsafe { c_some_function(ptr, len) }
});
```

### BLAS Operations

```rust
use bhc_ffi::{BlasProvider, default_provider, Transpose};

let blas = default_provider();

// Matrix multiplication: C = alpha * A * B + beta * C
blas.dgemm(
    Transpose::NoTrans,  // A
    Transpose::NoTrans,  // B
    m, n, k,             // Dimensions
    1.0,                 // alpha
    &a, lda,             // A matrix
    &b, ldb,             // B matrix
    0.0,                 // beta
    &mut c, ldc,         // C matrix
)?;
```

### High-Level Matrix Operations

```rust
use bhc_ffi::{Matrix, matmul, dot};

let a = Matrix::new(m, k, data_a);
let b = Matrix::new(k, n, data_b);

// Matrix multiplication
let c = matmul(&a, &b)?;

// Dot product
let d = dot(&vec1, &vec2)?;
```

## BLAS Provider

The `BlasProvider` trait abstracts over BLAS implementations:

```rust
pub trait BlasProvider: Send + Sync {
    // Level 1: Vector-Vector
    fn daxpy(&self, n: i32, alpha: f64, x: &[f64], incx: i32,
             y: &mut [f64], incy: i32);
    fn ddot(&self, n: i32, x: &[f64], incx: i32,
            y: &[f64], incy: i32) -> f64;

    // Level 2: Matrix-Vector
    fn dgemv(&self, trans: Transpose, m: i32, n: i32,
             alpha: f64, a: &[f64], lda: i32,
             x: &[f64], incx: i32,
             beta: f64, y: &mut [f64], incy: i32);

    // Level 3: Matrix-Matrix
    fn dgemm(&self, transa: Transpose, transb: Transpose,
             m: i32, n: i32, k: i32,
             alpha: f64, a: &[f64], lda: i32,
             b: &[f64], ldb: i32,
             beta: f64, c: &mut [f64], ldc: i32);
}
```

### Available Providers

| Provider | Description | Platforms |
|----------|-------------|-----------|
| OpenBLAS | Open-source optimized BLAS | Linux, macOS, Windows |
| MKL | Intel Math Kernel Library | Linux, macOS, Windows |
| Accelerate | Apple's vDSP/vecLib | macOS, iOS |
| Reference | Pure Rust fallback | All |

### Provider Selection

```rust
use bhc_ffi::blas;

// Automatic selection (best available)
let provider = blas::default_provider();

// Specific provider
let openblas = blas::openblas_provider()?;
let mkl = blas::mkl_provider()?;
let accelerate = blas::accelerate_provider()?;
```

## Error Types

```rust
pub enum FfiError {
    /// Memory allocation failed
    AllocationFailed(String),

    /// Buffer is not pinned
    NotPinned,

    /// Null pointer encountered
    NullPointer,

    /// Size mismatch
    SizeMismatch { expected: usize, actual: usize },

    /// Alignment error
    AlignmentError { required: usize, actual: usize },
}

pub enum BlasError {
    /// Provider not available
    ProviderNotAvailable(String),

    /// Invalid argument
    InvalidArgument(String),

    /// Dimension mismatch
    DimensionMismatch { expected: (usize, usize), actual: (usize, usize) },
}
```

## M4 Exit Criteria

This crate implements M4 FFI requirements:

- `matmul` can call external BLAS for large sizes
- Tensors stay pinned across FFI calls (verified by address stability)
- No GC movement of pinned allocations (stress test passes)

## Design Notes

- All FFI pointers must come from pinned memory
- BLAS calls use row-major layout by default
- Provider selection happens at runtime
- Pure Rust fallback ensures portability

## Related Crates

- `bhc-tensor-ir` - Tensor operations that may use BLAS
- `bhc-numeric` - Numeric stdlib using FFI
- `bhc-rts` - Runtime memory management

## Specification References

- H26-SPEC Section 9.4: FFI Memory Model
- H26-SPEC Section 7.5: BLAS Integration
- BHC-RULE-010: FFI Guidelines
