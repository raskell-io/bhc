//! LLVM module wrapper.
//!
//! An LLVM module represents a single compilation unit and contains
//! functions, global variables, and type definitions.

use crate::{CodegenError, CodegenOutputType, CodegenResult};
use bhc_session::OptLevel;
use inkwell::builder::Builder;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::FileType;
use inkwell::types::FunctionType;
use inkwell::values::{FunctionValue, PointerValue};
use inkwell::AddressSpace;
use std::path::Path;

use super::context::LlvmContext;
use super::types::TypeMapper;

/// An LLVM module being compiled.
///
/// This wraps an inkwell `Module` and provides methods for adding
/// functions, emitting IR, and writing output files.
pub struct LlvmModule<'ctx> {
    /// The underlying LLVM module.
    module: Module<'ctx>,
    /// The IR builder for constructing instructions.
    builder: Builder<'ctx>,
    /// Type mapper for converting BHC types to LLVM types.
    type_mapper: TypeMapper<'ctx>,
    /// Module name.
    name: String,
}

// Note: LlvmModule is NOT Send because it contains references to the context
// This is fine because we create modules, use them, and dispose of them
// within a single thread context.

impl<'ctx> LlvmModule<'ctx> {
    /// Create a new LLVM module.
    pub fn new(ctx: &'ctx LlvmContext, name: &str) -> CodegenResult<Self> {
        let llvm_ctx = ctx.llvm_context();
        let module = llvm_ctx.create_module(name);

        // Set target triple and data layout
        module.set_triple(&ctx.target_machine().get_triple());
        module.set_data_layout(&ctx.target_machine().get_target_data().get_data_layout());

        let builder = llvm_ctx.create_builder();
        let type_mapper = TypeMapper::new(llvm_ctx);

        Ok(Self {
            module,
            builder,
            type_mapper,
            name: name.to_string(),
        })
    }

    /// Get a reference to the underlying LLVM module.
    #[must_use]
    pub fn llvm_module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Get the IR builder.
    #[must_use]
    pub fn builder(&self) -> &Builder<'ctx> {
        &self.builder
    }

    /// Get the type mapper.
    #[must_use]
    pub fn type_mapper(&self) -> &TypeMapper<'ctx> {
        &self.type_mapper
    }

    /// Add a function to the module.
    pub fn add_function(
        &self,
        name: &str,
        fn_type: FunctionType<'ctx>,
    ) -> FunctionValue<'ctx> {
        self.module.add_function(name, fn_type, None)
    }

    /// Get a function by name.
    #[must_use]
    pub fn get_function(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        self.module.get_function(name)
    }

    /// Add a global string constant.
    pub fn add_global_string(&self, name: &str, value: &str) -> PointerValue<'ctx> {
        let string_val = self.builder.build_global_string_ptr(value, name)
            .expect("failed to build global string");
        string_val.as_pointer_value()
    }

    /// Create the main entry point that calls the Haskell main function.
    ///
    /// This generates a C `main` function that:
    /// 1. Initializes the RTS
    /// 2. Calls the Haskell `main` function
    /// 3. Shuts down the RTS
    /// 4. Returns 0
    pub fn create_entry_point(&self, haskell_main: FunctionValue<'ctx>) -> CodegenResult<FunctionValue<'ctx>> {
        let i32_type = self.type_mapper.i32_type();
        let void_type = self.type_mapper.context().void_type();
        // Use opaque pointer type (LLVM 15+)
        let ptr_type = self.type_mapper.context().ptr_type(AddressSpace::default());

        // int main(int argc, char** argv)
        let main_type = i32_type.fn_type(
            &[i32_type.into(), ptr_type.into()],
            false,
        );
        let main_fn = self.add_function("main", main_type);

        // Declare bhc_rts_init(int argc, char** argv)
        let rts_init_type = void_type.fn_type(
            &[i32_type.into(), ptr_type.into()],
            false,
        );
        let rts_init = self.module.add_function("bhc_rts_init", rts_init_type, None);

        // Declare bhc_shutdown()
        let shutdown_type = void_type.fn_type(&[], false);
        let shutdown = self.module.add_function("bhc_shutdown", shutdown_type, None);

        // Create entry block
        let entry = self.type_mapper.context().append_basic_block(main_fn, "entry");
        self.builder.position_at_end(entry);

        // Get argc and argv parameters
        let argc = main_fn.get_nth_param(0)
            .ok_or_else(|| CodegenError::Internal("missing argc param".to_string()))?;
        let argv = main_fn.get_nth_param(1)
            .ok_or_else(|| CodegenError::Internal("missing argv param".to_string()))?;

        // Call bhc_rts_init(argc, argv)
        self.builder.build_call(rts_init, &[argc.into(), argv.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to build rts_init call: {:?}", e)))?;

        // Call Haskell main (returns void or a value we ignore)
        self.builder.build_call(haskell_main, &[], "")
            .map_err(|e| CodegenError::Internal(format!("failed to build call: {:?}", e)))?;

        // Call bhc_shutdown()
        self.builder.build_call(shutdown, &[], "")
            .map_err(|e| CodegenError::Internal(format!("failed to build shutdown call: {:?}", e)))?;

        // Return 0
        let zero = i32_type.const_int(0, false);
        self.builder.build_return(Some(&zero))
            .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;

        Ok(main_fn)
    }

    /// Emit the module to a file.
    fn emit_to_file(
        &self,
        ctx: &LlvmContext,
        path: &Path,
        file_type: FileType,
    ) -> CodegenResult<()> {
        ctx.target_machine()
            .write_to_file(&self.module, file_type, path)
            .map_err(|e| CodegenError::OutputError {
                path: path.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
            })
    }
}

impl<'ctx> LlvmModule<'ctx> {
    /// Get the module name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Verify the module is well-formed.
    pub fn verify(&self) -> CodegenResult<()> {
        self.module
            .verify()
            .map_err(|e| CodegenError::Internal(format!("LLVM verification failed: {}", e.to_string())))
    }

    /// Optimize the module using LLVM's optimization passes.
    ///
    /// Requires a reference to the context to access the target machine.
    pub fn optimize(&mut self, ctx: &LlvmContext, level: OptLevel) -> CodegenResult<()> {
        let passes = match level {
            OptLevel::None => return Ok(()),
            OptLevel::Less | OptLevel::Size | OptLevel::SizeMin => "default<O1>",
            OptLevel::Default => "default<O2>",
            OptLevel::Aggressive => "default<O3>",
        };

        let options = PassBuilderOptions::create();
        self.module
            .run_passes(passes, ctx.target_machine(), options)
            .map_err(|e| CodegenError::Internal(format!("optimization failed: {}", e.to_string())))
    }

    /// Write the module to a file.
    pub fn write_to_file(&self, path: &Path, output_type: CodegenOutputType) -> CodegenResult<()> {
        match output_type {
            CodegenOutputType::LlvmIr => {
                self.module.print_to_file(path).map_err(|e| CodegenError::OutputError {
                    path: path.display().to_string(),
                    source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
                })
            }
            CodegenOutputType::LlvmBitcode => {
                if self.module.write_bitcode_to_path(path) {
                    Ok(())
                } else {
                    Err(CodegenError::OutputError {
                        path: path.display().to_string(),
                        source: std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "failed to write bitcode",
                        ),
                    })
                }
            }
            CodegenOutputType::Assembly | CodegenOutputType::Object => {
                // Object/assembly emission requires the target machine
                // Use emit_object or emit_assembly methods instead
                Err(CodegenError::Internal(
                    "object/assembly emission requires target machine - use emit_object or emit_assembly".to_string(),
                ))
            }
        }
    }

    /// Get the module as LLVM IR text.
    #[must_use]
    pub fn as_llvm_ir(&self) -> String {
        self.module.print_to_string().to_string()
    }
}

/// Extension trait for emitting object files (requires context).
pub trait LlvmModuleExt<'ctx> {
    /// Emit the module to an object file.
    fn emit_object(&self, ctx: &LlvmContext, path: &Path) -> CodegenResult<()>;

    /// Emit the module to an assembly file.
    fn emit_assembly(&self, ctx: &LlvmContext, path: &Path) -> CodegenResult<()>;
}

impl<'ctx> LlvmModuleExt<'ctx> for LlvmModule<'ctx> {
    fn emit_object(&self, ctx: &LlvmContext, path: &Path) -> CodegenResult<()> {
        self.emit_to_file(ctx, path, FileType::Object)
    }

    fn emit_assembly(&self, ctx: &LlvmContext, path: &Path) -> CodegenResult<()> {
        self.emit_to_file(ctx, path, FileType::Assembly)
    }
}
