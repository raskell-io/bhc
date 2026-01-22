//! BHC Base Library
//!
//! This crate provides the core Data.*, Control.*, and System.* modules
//! for the BHC standard library.
//!
//! # Modules
//!
//! ## Data modules
//! - `list` - Extended list operations
//! - `char` - Character operations
//! - `string` - String operations
//! - `function` - Extended function combinators
//! - `ord` - Extended ordering operations
//!
//! ## Control modules
//! - `monad` - Extended monad operations
//! - `applicative` - Extended applicative operations
//! - `category` - Category theory abstractions
//! - `arrow` - Arrow combinators
//!
//! ## System modules
//! - `io` - Input/output operations
//! - `environment` - Environment variable access

#![warn(missing_docs)]
#![warn(unsafe_code)]

pub mod applicative;
pub mod arrow;
pub mod category;
pub mod char;
pub mod function;
pub mod io;
pub mod list;
pub mod monad;
pub mod string;
