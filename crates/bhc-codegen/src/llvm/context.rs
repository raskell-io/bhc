//! LLVM context wrapper.
//!
//! The context owns all LLVM data and must outlive any modules created from it.

use crate::{CodegenConfig, CodegenError, CodegenResult};
use bhc_target::TargetSpec;
use inkwell::context::Context;
use inkwell::targets::{CodeModel, RelocMode, Target, TargetMachine, TargetTriple};

use super::module::LlvmModule;

/// LLVM code generation context.
///
/// This wraps an inkwell `Context` and provides the configuration
/// needed for code generation. The context must outlive any modules
/// created from it.
pub struct LlvmContext {
    /// The underlying LLVM context.
    context: Context,
    /// The target machine for code generation.
    target_machine: TargetMachine,
    /// Code generation configuration.
    config: CodegenConfig,
}

// Safety: Context is Send+Sync in inkwell when not actively being modified
unsafe impl Send for LlvmContext {}
unsafe impl Sync for LlvmContext {}

impl LlvmContext {
    /// Create a new LLVM context for the given configuration.
    pub fn new(config: CodegenConfig) -> CodegenResult<Self> {
        let context = Context::create();

        // Create target machine
        let target_machine = Self::create_target_machine(&config)?;

        Ok(Self {
            context,
            target_machine,
            config,
        })
    }

    /// Create a target machine for the given configuration.
    fn create_target_machine(config: &CodegenConfig) -> CodegenResult<TargetMachine> {
        let triple_str = config.target.triple();
        let triple = TargetTriple::create(&triple_str);

        let target = Target::from_triple(&triple).map_err(|e| {
            CodegenError::UnsupportedTarget(format!("{}: {}", triple_str, e.to_string()))
        })?;

        let opt_level = match config.opt_level {
            bhc_session::OptLevel::None => inkwell::OptimizationLevel::None,
            bhc_session::OptLevel::Less => inkwell::OptimizationLevel::Less,
            bhc_session::OptLevel::Default => inkwell::OptimizationLevel::Default,
            bhc_session::OptLevel::Aggressive => inkwell::OptimizationLevel::Aggressive,
            // Size optimization maps to Less in LLVM (no direct size-opt level)
            bhc_session::OptLevel::Size | bhc_session::OptLevel::SizeMin => {
                inkwell::OptimizationLevel::Less
            }
        };

        let reloc_mode = if config.pic {
            RelocMode::PIC
        } else {
            RelocMode::Default
        };

        let code_model = CodeModel::Default;

        let cpu = &config.cpu;
        let features = ""; // Could be derived from target spec

        target
            .create_target_machine(&triple, cpu, features, opt_level, reloc_mode, code_model)
            .ok_or_else(|| {
                CodegenError::Internal(format!(
                    "failed to create target machine for {}",
                    triple_str
                ))
            })
    }

    /// Get a reference to the underlying LLVM context.
    #[must_use]
    pub fn llvm_context(&self) -> &Context {
        &self.context
    }

    /// Get a reference to the target machine.
    #[must_use]
    pub fn target_machine(&self) -> &TargetMachine {
        &self.target_machine
    }

    /// Get the data layout string for this target.
    #[must_use]
    pub fn data_layout(&self) -> String {
        self.target_machine
            .get_target_data()
            .get_data_layout()
            .as_str()
            .to_str()
            .unwrap_or("")
            .to_string()
    }

    /// Create a new module in this context.
    pub fn create_module(&self, name: &str) -> CodegenResult<LlvmModule<'_>> {
        LlvmModule::new(self, name)
    }

    /// Get the target specification.
    #[must_use]
    pub fn target(&self) -> &TargetSpec {
        &self.config.target
    }

    /// Get the codegen configuration.
    #[must_use]
    pub fn config(&self) -> &CodegenConfig {
        &self.config
    }
}
