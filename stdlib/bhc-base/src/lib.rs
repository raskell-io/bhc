//! BHC Base Library - Rust support
//!
//! This crate provides minimal Rust support for the BHC base library.
//!
//! # Architecture
//!
//! The BHC base library is **implemented in Haskell** (see `hs/BHC/Data/*.hs`
//! and `hs/BHC/Control/*.hs`). BHC compiles the Haskell source directly.
//!
//! This Rust crate only provides:
//! - Character primitives (Unicode operations via Rust's char handling)
//! - Re-exports from bhc-rts
//!
//! # What does NOT belong here
//!
//! - List, Monad, Applicative, Arrow implementations
//! - Those are Haskell and compiled by BHC

#![warn(missing_docs)]
#![allow(unsafe_code)]

pub mod char; // Unicode primitives - actually needed
