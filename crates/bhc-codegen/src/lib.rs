//! Code generation backend for BHC.
//!
//! This crate provides the infrastructure for generating native code from
//! the compiler's intermediate representations. The primary backend is LLVM.
//!
//! # Overview
//!
//! Code generation in BHC follows this pipeline:
//!
//! ```text
//! Core IR ──▶ LLVM IR ──▶ Object Code ──▶ Executable
//! ```
//!
//! For numeric code with the numeric profile:
//!
//! ```text
//! Core IR ──▶ Tensor IR ──▶ Loop IR ──▶ LLVM IR ──▶ Object Code
//! ```
//!
//! # Backend Architecture
//!
//! The codegen system is designed around traits that abstract over different
//! backends:
//!
//! - [`CodegenBackend`]: The main trait for code generation backends
//! - [`CodegenContext`]: Context holding state during code generation
//! - [`CodegenModule`]: A compilation unit in the backend's representation
//!
//! # LLVM Backend
//!
//! The LLVM backend provides:
//!
//! - Full optimization pipeline integration
//! - Target-specific code generation
//! - Debug information generation
//! - Link-time optimization (LTO) support
//!
//! # Example
//!
//! ```ignore
//! use bhc_codegen::{CodegenBackend, CodegenConfig, llvm::LlvmBackend};
//!
//! // Create the LLVM backend
//! let backend = LlvmBackend::new();
//!
//! // Create a context for code generation
//! let config = CodegenConfig::default();
//! let ctx = backend.create_context(config)?;
//!
//! // Create a module
//! let module = ctx.create_module("my_module")?;
//!
//! // ... add functions and generate code ...
//!
//! // Write output
//! module.write_to_file(Path::new("output.ll"), CodegenOutputType::LlvmIr)?;
//! ```

#![warn(missing_docs)]

pub mod llvm;

use bhc_session::{DebugInfo, OptLevel, OutputType};
use bhc_target::TargetSpec;
use std::path::Path;
use thiserror::Error;

// Re-export the LLVM backend as the default
pub use llvm::{LlvmBackend, LlvmContext, LlvmModule};

/// Errors that can occur during code generation.
#[derive(Debug, Error)]
pub enum CodegenError {
    /// The requested backend is not available.
    #[error("backend not available: {0}")]
    BackendNotAvailable(String),

    /// Target is not supported by the backend.
    #[error("unsupported target: {0}")]
    UnsupportedTarget(String),

    /// LLVM-specific error.
    #[error("LLVM error: {0}")]
    LlvmError(String),

    /// Failed to write output file.
    #[error("failed to write output: {path}")]
    OutputError {
        /// The path that could not be written.
        path: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// Internal code generation error.
    #[error("internal codegen error: {0}")]
    Internal(String),

    /// Type error during code generation.
    #[error("type error: {0}")]
    TypeError(String),

    /// Unsupported feature.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Result type for code generation operations.
pub type CodegenResult<T> = Result<T, CodegenError>;

/// Code generation configuration.
#[derive(Clone, Debug)]
pub struct CodegenConfig {
    /// Target specification.
    pub target: TargetSpec,
    /// Optimization level.
    pub opt_level: OptLevel,
    /// Debug information level.
    pub debug_info: DebugInfo,
    /// Whether to use PIC (position-independent code).
    pub pic: bool,
    /// Whether to generate frame pointers.
    pub frame_pointers: bool,
    /// Whether to enable LTO.
    pub lto: bool,
    /// CPU model to target (e.g., "generic", "native").
    pub cpu: String,
}

impl Default for CodegenConfig {
    fn default() -> Self {
        Self {
            target: bhc_target::host_target(),
            opt_level: OptLevel::Default,
            debug_info: DebugInfo::None,
            pic: false,
            frame_pointers: true,
            lto: false,
            cpu: "generic".to_string(),
        }
    }
}

impl CodegenConfig {
    /// Create a new codegen config for the given target.
    #[must_use]
    pub fn for_target(target: TargetSpec) -> Self {
        Self {
            target,
            ..Self::default()
        }
    }

    /// Set the optimization level.
    #[must_use]
    pub fn with_opt_level(mut self, level: OptLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Set the debug info level.
    #[must_use]
    pub fn with_debug_info(mut self, level: DebugInfo) -> Self {
        self.debug_info = level;
        self
    }

    /// Enable position-independent code.
    #[must_use]
    pub fn with_pic(mut self, pic: bool) -> Self {
        self.pic = pic;
        self
    }

    /// Enable LTO.
    #[must_use]
    pub fn with_lto(mut self, lto: bool) -> Self {
        self.lto = lto;
        self
    }

    /// Set the CPU model.
    #[must_use]
    pub fn with_cpu(mut self, cpu: impl Into<String>) -> Self {
        self.cpu = cpu.into();
        self
    }
}

/// The type of code being generated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenOutputType {
    /// Object file.
    Object,
    /// Assembly file.
    Assembly,
    /// LLVM IR (text).
    LlvmIr,
    /// LLVM bitcode.
    LlvmBitcode,
}

impl From<OutputType> for CodegenOutputType {
    fn from(value: OutputType) -> Self {
        match value {
            OutputType::Assembly => Self::Assembly,
            OutputType::LlvmIr => Self::LlvmIr,
            OutputType::LlvmBitcode => Self::LlvmBitcode,
            _ => Self::Object,
        }
    }
}

/// A module being compiled in the backend.
///
/// Note: Modules are not required to be Send because LLVM modules
/// are not thread-safe. Compilation of a single module happens
/// on a single thread.
pub trait CodegenModule {
    /// Get the module name.
    fn name(&self) -> &str;

    /// Verify the module is well-formed.
    fn verify(&self) -> CodegenResult<()>;

    /// Optimize the module.
    fn optimize(&mut self, level: OptLevel) -> CodegenResult<()>;

    /// Write the module to a file.
    fn write_to_file(&self, path: &Path, output_type: CodegenOutputType) -> CodegenResult<()>;

    /// Get the module as LLVM IR text (if supported).
    fn as_llvm_ir(&self) -> CodegenResult<String>;
}

/// Context for code generation.
pub trait CodegenContext: Send + Sync {
    /// The module type for this context.
    type Module: CodegenModule;

    /// Create a new module.
    fn create_module(&self, name: &str) -> CodegenResult<Self::Module>;

    /// Get the target specification.
    fn target(&self) -> &TargetSpec;

    /// Get the codegen configuration.
    fn config(&self) -> &CodegenConfig;
}

/// A code generation backend.
pub trait CodegenBackend: Send + Sync {
    /// The context type for this backend.
    type Context: CodegenContext;

    /// Get the backend name.
    fn name(&self) -> &'static str;

    /// Check if this backend supports the given target.
    fn supports_target(&self, target: &TargetSpec) -> bool;

    /// Create a codegen context with the given configuration.
    fn create_context(&self, config: CodegenConfig) -> CodegenResult<Self::Context>;
}

/// Type layout information for code generation.
#[derive(Clone, Debug)]
pub struct TypeLayout {
    /// Size in bytes.
    pub size: u64,
    /// Alignment in bytes.
    pub alignment: u64,
}

impl TypeLayout {
    /// Layout for a pointer on the given target.
    #[must_use]
    pub fn pointer(target: &TargetSpec) -> Self {
        let width = target.pointer_width() as u64;
        Self {
            size: width,
            alignment: width,
        }
    }

    /// Layout for an i64.
    #[must_use]
    pub const fn i64() -> Self {
        Self {
            size: 8,
            alignment: 8,
        }
    }

    /// Layout for an f64.
    #[must_use]
    pub const fn f64() -> Self {
        Self {
            size: 8,
            alignment: 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_config() {
        let config = CodegenConfig::default()
            .with_opt_level(OptLevel::Aggressive)
            .with_pic(true);

        assert_eq!(config.opt_level, OptLevel::Aggressive);
        assert!(config.pic);
    }

    #[test]
    fn test_llvm_backend_creation() {
        let backend = LlvmBackend::new();
        assert_eq!(backend.name(), "llvm");
        assert!(backend.is_available());
    }

    #[test]
    fn test_llvm_context_creation() {
        let backend = LlvmBackend::new();
        let config = CodegenConfig::default();
        let result = backend.create_context(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_module_creation() {
        let ctx = LlvmContext::new(CodegenConfig::default()).unwrap();
        let module = ctx.create_module("test").unwrap();

        assert_eq!(module.name(), "test");

        // Verify the module produces valid LLVM IR
        let ir = module.as_llvm_ir();
        assert!(ir.contains("ModuleID"));
        assert!(ir.contains("target triple"));
    }

    #[test]
    fn test_module_verification() {
        let ctx = LlvmContext::new(CodegenConfig::default()).unwrap();
        let module = ctx.create_module("test").unwrap();

        // Empty module should verify
        assert!(module.verify().is_ok());
    }
}
