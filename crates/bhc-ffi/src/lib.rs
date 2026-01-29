//! # BHC FFI Primitives
//!
//! This crate provides Foreign Function Interface (FFI) support for BHC,
//! enabling safe interoperation with external C libraries such as BLAS
//! implementations.
//!
//! ## Overview
//!
//! Key components:
//!
//! - **PinnedBuffer**: A buffer that is guaranteed not to be moved by the GC,
//!   safe for passing to foreign code.
//! - **BLAS Provider**: A trait-based abstraction for BLAS implementations,
//!   allowing pluggable backends (OpenBLAS, MKL, Accelerate).
//! - **Safe FFI Boundary**: The `with_pinned_tensor` pattern ensures tensor
//!   data remains valid and unmoved during FFI calls.
//!
//! ## M4 Exit Criteria
//!
//! - `matmul` can call external BLAS for large sizes
//! - Tensors stay pinned across FFI calls (verified by address stability)
//! - No GC movement of pinned allocations (stress test)
//!
//! ## FFI Safety Model
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    Haskell / BHC Runtime                       │
//! ├────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐                                              │
//! │  │   Tensor     │  ──(pin)──>  ┌──────────────────┐           │
//! │  │  (may move)  │              │  PinnedBuffer     │           │
//! │  └──────────────┘              │  (never moves)    │           │
//! │                                └────────┬─────────┘           │
//! │                                         │                      │
//! ├─────────────────────────────────────────┼──────────────────────┤
//! │                      FFI Boundary       │                      │
//! ├─────────────────────────────────────────┼──────────────────────┤
//! │                                         │                      │
//! │  ┌──────────────────────────────────────▼───────────────────┐ │
//! │  │                   C Library (BLAS)                        │ │
//! │  │  - Receives raw pointer                                   │ │
//! │  │  - Pointer remains valid for call duration                │ │
//! │  └──────────────────────────────────────────────────────────┘ │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use bhc_ffi::{PinnedBuffer, with_pinned, BlasProvider, blas};
//!
//! // Safe: buffer guaranteed pinned for duration
//! with_pinned(&tensor, |ptr, len| {
//!     // Call C function with raw pointer
//!     unsafe { c_blas_function(ptr, len) }
//! });
//!
//! // Use BLAS provider for high-level operations
//! let provider = blas::default_provider();
//! provider.dgemm(...);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod blas;
pub mod pinned;
pub mod tensor;

pub use blas::{default_provider, BlasError, BlasProvider, BlasResult, Transpose};
pub use pinned::{with_pinned, with_pinned_mut, PinnedBuffer, PinnedSlice};
pub use tensor::{dot, matmul, sdot, smatmul, Matrix};

use thiserror::Error;

/// Errors that can occur during FFI operations.
#[derive(Clone, Debug, Error)]
pub enum FfiError {
    /// Memory allocation failed.
    #[error("FFI allocation failed: {0}")]
    AllocationFailed(String),

    /// Buffer is not pinned.
    #[error("buffer is not pinned, cannot pass to FFI")]
    NotPinned,

    /// Null pointer encountered.
    #[error("null pointer in FFI call")]
    NullPointer,

    /// Size mismatch between expected and actual.
    #[error("size mismatch: expected {expected}, got {actual}")]
    SizeMismatch {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    /// Alignment error.
    #[error("alignment error: pointer {ptr:p} not aligned to {required} bytes")]
    AlignmentError {
        /// The misaligned pointer.
        ptr: *const u8,
        /// Required alignment.
        required: usize,
    },

    /// BLAS operation failed.
    #[error("BLAS error: {0}")]
    BlasError(#[from] BlasError),

    /// Foreign function returned an error code.
    #[error("foreign function returned error code: {0}")]
    ForeignError(i32),
}

/// Result type for FFI operations.
pub type FfiResult<T> = Result<T, FfiError>;

/// Check if a pointer is properly aligned for a type.
#[inline]
#[must_use]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    let align = std::mem::align_of::<T>();
    (ptr as usize) % align == 0
}

/// Check if a pointer is properly aligned to a specific boundary.
#[inline]
#[must_use]
pub fn is_aligned_to(ptr: *const u8, align: usize) -> bool {
    (ptr as usize) % align == 0
}

/// Marker trait for types that can be safely passed to FFI.
///
/// This trait indicates that a type:
/// - Has a stable memory layout (repr(C) or primitive)
/// - Contains no pointers to GC-managed memory
/// - Can be safely copied by foreign code
///
/// # Safety
///
/// Implementors must ensure the type is safe for FFI:
/// - Must be `Copy` and have no drop glue
/// - Must have a well-defined C ABI layout
/// - Must not contain pointers that could become invalid
pub unsafe trait FfiSafe: Copy + 'static {
    /// The C-equivalent type name (for documentation).
    const C_TYPE_NAME: &'static str;
}

// Implement FfiSafe for primitive types
unsafe impl FfiSafe for f32 {
    const C_TYPE_NAME: &'static str = "float";
}

unsafe impl FfiSafe for f64 {
    const C_TYPE_NAME: &'static str = "double";
}

unsafe impl FfiSafe for i8 {
    const C_TYPE_NAME: &'static str = "int8_t";
}

unsafe impl FfiSafe for i16 {
    const C_TYPE_NAME: &'static str = "int16_t";
}

unsafe impl FfiSafe for i32 {
    const C_TYPE_NAME: &'static str = "int32_t";
}

unsafe impl FfiSafe for i64 {
    const C_TYPE_NAME: &'static str = "int64_t";
}

unsafe impl FfiSafe for u8 {
    const C_TYPE_NAME: &'static str = "uint8_t";
}

unsafe impl FfiSafe for u16 {
    const C_TYPE_NAME: &'static str = "uint16_t";
}

unsafe impl FfiSafe for u32 {
    const C_TYPE_NAME: &'static str = "uint32_t";
}

unsafe impl FfiSafe for u64 {
    const C_TYPE_NAME: &'static str = "uint64_t";
}

unsafe impl FfiSafe for usize {
    const C_TYPE_NAME: &'static str = "size_t";
}

unsafe impl FfiSafe for isize {
    const C_TYPE_NAME: &'static str = "ssize_t";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_check() {
        let aligned: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        assert!(is_aligned(aligned.as_ptr()));
    }

    #[test]
    fn test_ffi_safe_type_names() {
        assert_eq!(f32::C_TYPE_NAME, "float");
        assert_eq!(f64::C_TYPE_NAME, "double");
        assert_eq!(i32::C_TYPE_NAME, "int32_t");
        assert_eq!(u64::C_TYPE_NAME, "uint64_t");
    }
}
