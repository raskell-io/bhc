//! BHC Numeric Library
//!
//! High-performance numeric computing for BHC.
//!
//! # Features
//!
//! - **SIMD**: Automatic vectorization using AVX/SSE/NEON
//! - **Fusion**: Guaranteed fusion for standard patterns
//! - **BLAS**: Optional BLAS backend integration
//!
//! # Modules
//!
//! - `simd` - SIMD vector types and operations
//! - `tensor` - N-dimensional arrays with shape tracking
//! - `vector` - 1-D numeric vectors
//! - `matrix` - 2-D matrices with linear algebra
//! - `blas` - BLAS provider abstraction

#![warn(missing_docs)]
#![allow(unsafe_code)] // SIMD requires unsafe

pub mod blas;
pub mod matrix;
pub mod simd;
pub mod tensor;
pub mod vector;
