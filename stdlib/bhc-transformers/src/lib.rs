//! BHC Transformers Library
//!
//! Monad transformers for composing effects.
//!
//! # Transformers
//!
//! - `reader` - ReaderT for environment access
//! - `writer` - WriterT for logging/accumulation
//! - `state` - StateT for mutable state
//! - `except` - ExceptT for error handling
//! - `maybe` - MaybeT for optional values
//! - `identity` - IdentityT base transformer

#![warn(missing_docs)]
#![warn(unsafe_code)]

pub mod except;
pub mod identity;
pub mod maybe;
pub mod reader;
pub mod state;
pub mod writer;

// Note: Transformers are primarily implemented in Haskell.
// This Rust crate provides any performance-critical primitives.
