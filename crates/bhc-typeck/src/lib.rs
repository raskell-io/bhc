//! # BHC Type Checker
//!
//! This crate implements Hindley-Milner type inference for the Basel Haskell Compiler.
//! It operates on HIR (High-level Intermediate Representation) and produces typed HIR
//! suitable for lowering to Core IR.
//!
//! ## Overview
//!
//! The type checker implements Algorithm W with the following features:
//!
//! - **Let-polymorphism**: Types are generalized at let-bindings
//! - **Mutual recursion**: Binding groups are analyzed via SCC decomposition
//! - **Type signatures**: User-provided signatures are checked against inferred types
//! - **Error recovery**: Inference continues after errors using error types
//!
//! ## Algorithm
//!
//! Type inference proceeds in several phases:
//!
//! 1. **Binding group analysis**: Identify mutually recursive groups via SCC
//! 2. **Constraint generation**: Walk HIR and generate type constraints
//! 3. **Unification**: Solve constraints via substitution
//! 4. **Generalization**: Generalize types at let-bindings
//!
//! ## Usage
//!
//! ```ignore
//! use bhc_typeck::type_check_module;
//! use bhc_hir::Module;
//! use bhc_span::FileId;
//!
//! let result = type_check_module(&hir_module, file_id);
//! match result {
//!     Ok(typed_module) => {
//!         // Use typed_module.expr_types to get inferred types
//!     }
//!     Err(diagnostics) => {
//!         // Report type errors
//!     }
//! }
//! ```
//!
//! ## See Also
//!
//! - `bhc-hir`: Input HIR types
//! - `bhc-types`: Type representation
//! - `bhc-core`: Output Core IR (after lowering)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

mod binding_groups;
mod builtins;
mod context;
mod diagnostics;
mod env;
mod generalize;
mod infer;
mod instantiate;
pub mod kind_check;
pub mod nat_solver;
mod pattern;
pub mod shape_bridge;
pub mod shape_diagrams;
pub mod suggest;
pub mod type_families;
mod unify;

pub use context::TyCtxt;
pub use env::{DataConInfo, TypeEnv};
pub use kind_check::KindEnv;

use bhc_diagnostics::Diagnostic;
use bhc_hir::{DefId, HirId, Module};
use bhc_intern::Symbol;
use bhc_span::FileId;
use bhc_types::{Scheme, Ty};
use indexmap::IndexMap;
use rustc_hash::FxHashMap;

/// The result of type checking a module.
///
/// Contains the original HIR along with type annotations for all
/// expressions and definitions.
#[derive(Debug)]
pub struct TypedModule {
    /// The original HIR module.
    pub hir: Module,
    /// Inferred types for each expression (indexed by `HirId`).
    pub expr_types: FxHashMap<HirId, Ty>,
    /// Type schemes for each definition (indexed by `DefId`).
    pub def_schemes: FxHashMap<DefId, Scheme>,
}

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

/// Type check a HIR module.
///
/// This is the main entry point for type checking. It takes a resolved
/// HIR module and produces either a typed module or a list of diagnostics.
///
/// # Arguments
///
/// * `hir` - The HIR module to type check
/// * `file_id` - The file ID for error reporting
///
/// # Returns
///
/// * `Ok(TypedModule)` - Successfully typed module
/// * `Err(Vec<Diagnostic>)` - Type errors encountered
///
/// # Errors
///
/// Returns a `Vec<Diagnostic>` containing all type errors found during
/// type checking, such as type mismatches, unbound variables, and
/// occurs check failures.
///
/// # Example
///
/// ```ignore
/// let result = type_check_module(&module, file_id);
/// ```
pub fn type_check_module(hir: &Module, file_id: FileId) -> Result<TypedModule, Vec<Diagnostic>> {
    type_check_module_with_defs(hir, file_id, None)
}

/// Type check a HIR module with definition mappings from the lowering pass.
///
/// This function accepts the DefMap from the lowering context, which allows
/// the type checker to register builtins with the correct DefIds assigned
/// during lowering.
///
/// # Arguments
///
/// * `hir` - The HIR module to type check
/// * `file_id` - The file ID for error reporting
/// * `defs` - Optional definition map from the lowering context
pub fn type_check_module_with_defs(
    hir: &Module,
    file_id: FileId,
    defs: Option<&DefMap>,
) -> Result<TypedModule, Vec<Diagnostic>> {
    let mut ctx = TyCtxt::new(file_id);

    // Register built-in types
    ctx.register_builtins();

    // If we have definition mappings from the lowering pass, use them
    // to register builtins with the correct DefIds
    if let Some(def_map) = defs {
        ctx.register_lowered_builtins(def_map);
    }

    // Register data types from the module
    for item in &hir.items {
        if let bhc_hir::Item::Data(data) = item {
            ctx.register_data_type(data);
        }
        if let bhc_hir::Item::Newtype(newtype) = item {
            ctx.register_newtype(newtype);
        }
    }

    // Compute binding groups (SCCs) for mutual recursion
    let groups = binding_groups::compute_binding_groups(&hir.items);

    // Type check each binding group
    for group in groups {
        ctx.check_binding_group(&group);
    }

    if ctx.has_errors() {
        Err(ctx.take_diagnostics())
    } else {
        Ok(ctx.into_typed_module(hir.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_module() {
        use bhc_intern::Symbol;

        let module = Module {
            name: Symbol::intern("Test"),
            exports: None,
            imports: Vec::new(),
            items: Vec::new(),
            span: bhc_span::Span::DUMMY,
        };

        let result = type_check_module(&module, FileId::new(0));
        assert!(result.is_ok());
    }
}
