# bhc-numeric

High-performance numeric primitives for the Basel Haskell Compiler.

## Overview

This crate provides the Rust-side numeric primitives for BHC, including SIMD operations, BLAS bindings, and tensor support. These primitives are essential for the Numeric Profile's performance guarantees.

## Key Features

| Feature | Description |
|---------|-------------|
| SIMD Types | Platform-specific vector types |
| BLAS Bindings | Linear algebra operations |
| Tensor Primitives | N-dimensional array support |
| FFI Bridge | Safe interface to Haskell |

## SIMD Types

```rust
#[repr(C, align(16))]
pub struct F32x4([f32; 4]);

#[repr(C, align(32))]
pub struct F32x8([f32; 8]);

#[repr(C, align(64))]
pub struct F64x8([f64; 8]);
```

## SIMD Operations

```rust
impl F32x4 {
    pub fn add(self, other: Self) -> Self;
    pub fn mul(self, other: Self) -> Self;
    pub fn fma(self, a: Self, b: Self) -> Self;  // Fused multiply-add
    pub fn sum(self) -> f32;                      // Horizontal sum
    pub fn dot(self, other: Self) -> f32;         // Dot product
}
```

## BLAS Interface

```rust
/// Matrix multiplication: C = alpha * A * B + beta * C
pub fn gemm(
    alpha: f64,
    a: &[f64], a_rows: usize, a_cols: usize,
    b: &[f64], b_rows: usize, b_cols: usize,
    beta: f64,
    c: &mut [f64],
);

/// Matrix-vector multiplication: y = alpha * A * x + beta * y
pub fn gemv(
    alpha: f64,
    a: &[f64], rows: usize, cols: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
);
```

## Tensor Primitives

```rust
/// Tensor descriptor
pub struct TensorDesc {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: DType,
}

/// Supported data types
pub enum DType {
    F32,
    F64,
    I32,
    I64,
}
```

## FFI Interface

These primitives are exposed to Haskell:

```haskell
-- SIMD vectors
foreign import ccall "bhc_f32x4_add" addF32x4 :: F32x4 -> F32x4 -> F32x4
foreign import ccall "bhc_f32x4_dot" dotF32x4 :: F32x4 -> F32x4 -> Float

-- BLAS
foreign import ccall "bhc_gemm" gemm :: Double -> Matrix -> Matrix -> Double -> Matrix -> IO ()
```

## Platform Support

| Platform | SIMD Support |
|----------|--------------|
| x86_64 | SSE4.2, AVX, AVX2, AVX-512 |
| aarch64 | NEON, SVE |
| wasm32 | SIMD128 |

## Design Notes

- SIMD types are aligned for optimal performance
- BLAS uses system library when available (OpenBLAS, MKL)
- Falls back to pure Rust implementation
- Thread-safe for parallel numeric workloads

## Related Crates

- `bhc-tensor-ir` - Tensor IR optimization
- `bhc-codegen` - SIMD code generation
- `bhc-gpu` - GPU acceleration
- `bhc-rts-arena` - Arena allocation for temporaries

## Specification References

- H26-SPEC Section 7: Numeric Computing
- H26-SPEC Section 7.1: SIMD Operations
- H26-SPEC Section 7.2: BLAS Integration
