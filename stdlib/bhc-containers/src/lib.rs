//! BHC Containers Library
//!
//! Efficient container types for BHC.
//!
//! # Containers
//!
//! - `map` - Immutable ordered maps (weight-balanced trees)
//! - `set` - Immutable ordered sets
//! - `intmap` - Maps with Int keys (optimized)
//! - `intset` - Sets of Int values (optimized)
//! - `sequence` - Sequences (finger trees)

#![warn(missing_docs)]
#![warn(unsafe_code)]

pub mod intmap;
pub mod intset;
pub mod map;
pub mod sequence;
pub mod set;
