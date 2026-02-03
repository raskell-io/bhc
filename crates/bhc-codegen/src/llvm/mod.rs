//! LLVM backend for BHC code generation.
//!
//! This module provides the LLVM-based code generation backend using inkwell
//! (safe Rust bindings for LLVM). It handles:
//!
//! - Creating and managing LLVM contexts and modules
//! - Type mapping from BHC types to LLVM types
//! - Code generation from Core IR to LLVM IR
//! - Code generation from Loop IR to LLVM IR (Numeric Profile)
//! - Optimization passes
//! - Object file emission

mod context;
mod loop_lower;
mod lower;
mod module;
mod types;

pub use context::LlvmContext;
pub use loop_lower::{lower_loop_ir, lower_loop_irs, LoopLowering};
pub use lower::{
    lower_core_module, lower_core_module_multimodule,
    lower_core_module_multimodule_with_constructors, CompiledSymbol, ConstructorMeta, Lowering,
};
pub use module::{LlvmModule, LlvmModuleExt};
pub use types::TypeMapper;

use crate::{CodegenConfig, CodegenResult};
use bhc_target::TargetSpec;

/// The LLVM code generation backend.
///
/// This backend uses LLVM (via inkwell) to generate native code for
/// all supported target platforms.
pub struct LlvmBackend {
    _private: (),
}

impl LlvmBackend {
    /// Create a new LLVM backend.
    ///
    /// This initializes LLVM and prepares for code generation.
    #[must_use]
    pub fn new() -> Self {
        // Initialize LLVM targets
        inkwell::targets::Target::initialize_all(&inkwell::targets::InitializationConfig::default());

        Self { _private: () }
    }

    /// Check if LLVM is available.
    ///
    /// Always returns true since LLVM is statically linked via inkwell.
    #[must_use]
    pub const fn is_available(&self) -> bool {
        true
    }

    /// Get the backend name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        "llvm"
    }

    /// Check if this backend supports the given target.
    #[must_use]
    pub fn supports_target(&self, target: &TargetSpec) -> bool {
        // Check if LLVM has a target for this triple
        let triple = inkwell::targets::TargetTriple::create(&target.triple());
        inkwell::targets::Target::from_triple(&triple).is_ok()
    }

    /// Create a codegen context with the given configuration.
    pub fn create_context(&self, config: CodegenConfig) -> CodegenResult<LlvmContext> {
        LlvmContext::new(config)
    }
}

impl Default for LlvmBackend {
    fn default() -> Self {
        Self::new()
    }
}
