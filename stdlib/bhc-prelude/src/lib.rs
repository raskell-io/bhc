//! BHC Prelude - Core types and functions
//!
//! This crate provides the Rust implementation of core BHC standard library
//! types and functions. These are exposed to Haskell through the FFI.
//!
//! # Overview
//!
//! The prelude contains:
//! - Core types: `Bool`, `Maybe`, `Either`, `Ordering`, tuples
//! - Core type classes: `Eq`, `Ord`, `Show`, `Num`, etc.
//! - Core functions: list operations, function combinators, numeric operations
//!
//! # FFI Conventions
//!
//! All FFI-exported functions follow these conventions:
//! - Use `#[no_mangle]` and `extern "C"` for C ABI
//! - Prefix with `bhc_` for namespace
//! - Return values through out-parameters for complex types
//! - Use `BhcResult` for fallible operations

#![warn(missing_docs)]
#![warn(unsafe_code)]
#![allow(clippy::module_name_repetitions)]

pub mod bool;
pub mod either;
pub mod function;
pub mod list;
pub mod maybe;
pub mod numeric;
pub mod ordering;
pub mod tuple;

/// Result type for FFI operations
#[repr(C)]
pub enum BhcResult<T> {
    /// Operation succeeded
    Ok(T),
    /// Operation failed with error code
    Err(BhcError),
}

/// Error codes for FFI operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BhcError {
    /// Null pointer passed where non-null expected
    NullPointer = 1,
    /// Index out of bounds
    IndexOutOfBounds = 2,
    /// Invalid argument
    InvalidArgument = 3,
    /// Out of memory
    OutOfMemory = 4,
    /// Division by zero
    DivisionByZero = 5,
    /// Arithmetic overflow
    Overflow = 6,
}

/// Common re-exports for FFI
pub mod ffi {
    pub use super::bool::*;
    pub use super::either::*;
    pub use super::list::*;
    pub use super::maybe::*;
    pub use super::numeric::*;
    pub use super::ordering::*;
}
