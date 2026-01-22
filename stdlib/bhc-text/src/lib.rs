//! BHC Text Library
//!
//! UTF-8 Text and ByteString types with SIMD acceleration.
//!
//! # Types
//!
//! - `text` - UTF-8 encoded text with SIMD operations
//! - `bytestring` - Raw byte arrays with pinned memory

#![warn(missing_docs)]
#![allow(unsafe_code)] // SIMD requires unsafe

pub mod bytestring;
pub mod text;
pub mod simd;
