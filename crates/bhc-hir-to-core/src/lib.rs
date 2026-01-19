//! # HIR to Core Lowering
//!
//! This crate transforms typed HIR (High-Level IR) into Core IR, the main
//! intermediate representation used for optimization.
//!
//! ## Key Transformations
//!
//! - **Pattern compilation**: Multi-argument lambdas and pattern matching
//!   are compiled into explicit case expressions
//! - **Binding analysis**: Let bindings are analyzed for mutual recursion
//! - **Guard expansion**: Pattern guards become nested conditionals
//! - **Type erasure**: Type annotations are preserved but simplified

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

mod binding;
mod context;
mod expr;
mod pattern;

use bhc_core::CoreModule;
use bhc_hir::{DefId, Module as HirModule};
use bhc_intern::Symbol;
use bhc_span::Span;
use indexmap::IndexMap;
use thiserror::Error;

pub use context::LowerContext;

/// Information about a definition from the lowering pass.
/// This mirrors `bhc_lower::context::DefInfo`.
#[derive(Clone, Debug)]
pub struct DefInfo {
    /// The unique ID.
    pub id: DefId,
    /// The name.
    pub name: Symbol,
}

/// Map from `DefId` to definition information.
pub type DefMap = IndexMap<DefId, DefInfo>;

/// Errors that can occur during HIR to Core lowering.
#[derive(Debug, Error)]
pub enum LowerError {
    /// An internal invariant was violated.
    #[error("internal error: {0}")]
    Internal(String),

    /// Pattern compilation failed.
    #[error("pattern compilation failed at {span:?}: {message}")]
    PatternError {
        /// Error message.
        message: String,
        /// Source location.
        span: Span,
    },

    /// Multiple errors occurred.
    #[error("multiple errors")]
    Multiple(Vec<LowerError>),
}

/// Result type for lowering operations.
pub type LowerResult<T> = Result<T, LowerError>;

/// Lower a HIR module to Core IR.
///
/// This is the main entry point for the HIR to Core transformation.
///
/// # Arguments
///
/// * `module` - The typed HIR module to lower
///
/// # Returns
///
/// A `CoreModule` containing the lowered bindings.
///
/// # Errors
///
/// Returns `LowerError` if lowering fails due to internal errors or
/// unsupported constructs.
pub fn lower_module(module: &HirModule) -> LowerResult<CoreModule> {
    lower_module_with_defs(module, None)
}

/// Lower a HIR module to Core IR with definition mappings from the lowering pass.
///
/// This function accepts the DefMap from the lowering context, which allows
/// the Core lowering to register builtins with the correct DefIds assigned
/// during the AST-to-HIR lowering pass.
///
/// # Arguments
///
/// * `module` - The typed HIR module to lower
/// * `defs` - Optional definition map from the lowering context
///
/// # Errors
///
/// Returns `LowerError` if lowering fails due to internal errors or
/// unsupported constructs.
pub fn lower_module_with_defs(
    module: &HirModule,
    defs: Option<&DefMap>,
) -> LowerResult<CoreModule> {
    let mut ctx = LowerContext::new();

    // If we have definition mappings from the lowering pass, use them
    // to register builtins with the correct DefIds
    if let Some(def_map) = defs {
        ctx.register_lowered_builtins(def_map);
    }

    ctx.lower_module(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_intern::Symbol;

    #[test]
    fn test_lower_empty_module() {
        let module = HirModule {
            name: Symbol::intern("Test"),
            exports: None,
            imports: vec![],
            items: vec![],
            span: Span::default(),
        };

        let result = lower_module(&module);
        assert!(result.is_ok());
        let core_module = result.unwrap();
        assert!(core_module.bindings.is_empty());
    }
}
