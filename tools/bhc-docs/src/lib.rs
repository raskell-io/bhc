//! bhc-docs: Documentation generator for BHC.
//!
//! This crate provides a modern documentation generator that produces beautiful
//! HTML documentation with:
//!
//! - **Type search**: Search by type signature with unification
//! - **Interactive examples**: Runnable code via bhc-playground WASM
//! - **BHC features**: Fusion annotations, SIMD badges, profile behavior
//! - **Dark mode**: CSS variables with localStorage persistence
//! - **Keyboard navigation**: `/` for search, `j`/`k` for navigation
//!
//! # Architecture
//!
//! The documentation pipeline consists of:
//!
//! 1. **Extraction** ([`extract`]): Parse source files and extract documentation
//! 2. **Model** ([`model`]): Internal representation of documentation
//! 3. **Haddock** ([`haddock`]): Parse Haddock markup into structured content
//! 4. **Render** ([`render`]): Generate HTML, Markdown, or JSON output
//! 5. **Search** ([`search`]): Build and query type search index
//! 6. **Serve** ([`serve`]): Development server with live reload
//!
//! # Usage
//!
//! ```ignore
//! use bhc_docs::{build, extract, model};
//!
//! // Extract docs from source
//! let module_docs = extract::extract_module(&source)?;
//!
//! // Render to HTML
//! let html = render::html::render_module(&module_docs)?;
//! ```

#![warn(missing_docs)]

pub mod build;
pub mod coverage;
pub mod extract;
pub mod haddock;
pub mod model;
pub mod render;
pub mod search;
pub mod serve;

/// Re-export commonly used types.
pub use model::{DocItem, FunctionDoc, ModuleDoc, TypeDoc};
