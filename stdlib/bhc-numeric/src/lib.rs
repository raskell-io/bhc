//! BHC Numeric Library - Rust support
//!
//! High-performance numeric primitives for BHC.
//!
//! # Architecture
//!
//! Haskell Tensor/Vector/Matrix types are **defined in Haskell** (see
//! `hs/BHC/Numeric/*.hs`). This Rust crate provides the low-level
//! primitives that BHC-compiled code calls via FFI.
//!
//! # What belongs here
//!
//! - SIMD vector types (Vec4F32, Vec8F32, etc.) with platform intrinsics
//! - BLAS provider abstraction (OpenBLAS, MKL, Accelerate)
//! - Low-level memory layout for tensor backing stores
//! - FFI-exportable numeric primitives (dot, saxpy, gemm, etc.)
//! - Hot Arena allocator for kernel temporaries
//!
//! # What does NOT belong here
//!
//! - High-level Tensor/Vector/Matrix API (that's Haskell)
//! - Fusion logic (that's BHC compiler)
//! - Type class instances (that's Haskell)
//!
//! # Memory Model (H26-SPEC Section 9)
//!
//! BHC defines three allocation regions:
//!
//! | Region | Allocation | Deallocation | GC | Use Case |
//! |--------|------------|--------------|-----|----------|
//! | **Hot Arena** | Bump pointer O(1) | Bulk free at scope end | None | Kernel temporaries |
//! | **Pinned Heap** | malloc-style | Explicit/refcounted | Never moved | FFI, DMA, GPU |
//! | **General Heap** | GC-managed | Automatic | May move | Normal boxed data |
//!
//! # FFI Exports
//!
//! This crate exports C-ABI functions for BHC to call:
//! - `bhc_simd_dot_f32`, `bhc_simd_dot_f64` - Dot product
//! - `bhc_simd_sum_f32`, `bhc_simd_sum_f64` - Sum reduction
//! - `bhc_simd_saxpy` - SAXPY operation
//! - `bhc_arena_new`, `bhc_arena_free` - Arena lifecycle
//! - `bhc_arena_zeros_f64`, `bhc_arena_zeros_f32` - Arena tensor allocation
//! - `bhc_sparse_csr_new_f64`, `bhc_sparse_csr_new_f32` - CSR matrix creation
//! - `bhc_sparse_csr_spmv_f64`, `bhc_sparse_csr_spmv_f32` - Sparse matrix-vector multiply
//! - `bhc_sparse_csr_spmm_f64`, `bhc_sparse_csr_spmm_f32` - Sparse matrix-matrix multiply
//!
//! # Features
//!
//! - **SIMD**: SSE, AVX, AVX2, FMA intrinsics
//! - **BLAS**: Optional BLAS backend integration
//! - **Arena**: Hot arena for kernel temporaries
//! - **Portable**: Scalar fallbacks for all operations

#![warn(missing_docs)]
#![allow(unsafe_code)] // SIMD and arena require unsafe

pub mod arena;
pub mod blas;
pub mod matrix;
pub mod simd;
pub mod sparse;
pub mod tensor;
pub mod vector;
