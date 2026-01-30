//! # BHC AST to HIR Lowering
//!
//! This crate implements the lowering pass from surface AST to HIR (High-Level
//! Intermediate Representation). This pass performs:
//!
//! - **Desugaring**: Expand syntactic sugar like do-notation, list comprehensions,
//!   operator sections, and if-then-else expressions
//! - **Name resolution**: Resolve all identifiers to their definitions
//! - **Pattern compilation**: Convert complex patterns and guards
//!
//! ## Pipeline Position
//!
//! ```text
//! Source Code
//!     |
//!     v
//! [Parse/AST]  <- Surface syntax
//!     |
//!     v
//! [Lower]      <- THIS CRATE
//!     |
//!     v
//! [HIR]        <- Desugared, resolved
//!     |
//!     v
//! [Type Check] <- Type inference
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use bhc_lower::{lower_module, LowerConfig, LowerContext};
//! use camino::Utf8PathBuf;
//!
//! let ast_module: bhc_ast::Module = parse(...)?;
//! let mut ctx = LowerContext::new();
//! let config = LowerConfig {
//!     include_builtins: true,
//!     warn_unused: false,
//!     search_paths: vec![Utf8PathBuf::from("/path/to/sources")],
//! };
//! let hir_module = lower_module(&mut ctx, &ast_module, &config)?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

mod context;
mod desugar;
pub mod loader;
mod lower;
mod resolve;

pub use context::{DefKind, DefMap, LowerContext, Scope, ScopeId};
pub use loader::{ConstructorInfo, LoadError, ModuleCache, ModuleExports};
pub use lower::{lower_module, lower_module_with_cache, LowerConfig};

use bhc_span::Span;
use thiserror::Error;

/// Errors that can occur during lowering.
#[derive(Debug, Error)]
pub enum LowerError {
    /// An unbound variable was referenced.
    #[error("unbound variable: {name}")]
    UnboundVar {
        /// The variable name.
        name: String,
        /// Source location.
        span: Span,
    },

    /// An unbound type was referenced.
    #[error("unbound type: {name}")]
    UnboundType {
        /// The type name.
        name: String,
        /// Source location.
        span: Span,
    },

    /// An unbound constructor was referenced.
    #[error("unbound constructor: {name}")]
    UnboundCon {
        /// The constructor name.
        name: String,
        /// Source location.
        span: Span,
    },

    /// Duplicate definition in the same scope.
    #[error("duplicate definition: {name}")]
    DuplicateDefinition {
        /// The duplicate name.
        name: String,
        /// Location of the new definition.
        new_span: Span,
        /// Location of the existing definition.
        existing_span: Span,
    },

    /// Invalid pattern in binding position.
    #[error("invalid pattern in binding: {reason}")]
    InvalidPattern {
        /// Why the pattern is invalid.
        reason: String,
        /// Source location.
        span: Span,
    },

    /// Unsupported syntax that hasn't been implemented yet.
    #[error("unsupported syntax: {feature}")]
    Unsupported {
        /// Description of the unsupported feature.
        feature: String,
        /// Source location.
        span: Span,
    },

    /// Multiple errors collected during lowering.
    #[error("{}", display_multiple(.0))]
    Multiple(Vec<LowerError>),
}

fn display_multiple(errors: &[LowerError]) -> String {
    use std::fmt::Write;
    let mut s = format!("{} lowering error(s):\n", errors.len());
    for (i, err) in errors.iter().enumerate() {
        writeln!(&mut s, "  {}: {}", i + 1, err).unwrap();
    }
    s
}

/// Result type for lowering operations.
pub type LowerResult<T> = Result<T, LowerError>;

/// Warnings that can occur during lowering.
#[derive(Debug)]
pub enum LowerWarning {
    /// A stub definition was used (external package placeholder).
    StubUsed {
        /// The stub name.
        name: String,
        /// Source location where stub was used.
        span: Span,
        /// What kind of stub (value, type, or constructor).
        kind: &'static str,
    },
}

impl std::fmt::Display for LowerWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerWarning::StubUsed { name, kind, .. } => {
                write!(
                    f,
                    "stub {kind} `{name}` used (external package not implemented)"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_error_display() {
        let err = LowerError::UnboundVar {
            name: "foo".to_string(),
            span: Span::default(),
        };
        assert!(err.to_string().contains("unbound variable"));
    }
}
