//! WASM code generation module.
//!
//! This module handles the conversion of Loop IR to WebAssembly instructions,
//! including type mapping, instruction emission, and module generation.

mod emitter;
mod simd;
/// Type mapping between Loop IR and WASM types.
pub mod types;

pub use emitter::WasmEmitter;
pub use simd::{SimdLowering, SimdPattern};
pub use types::{type_to_wasm, LoopTypeMapping};

use crate::{WasmConfig, WasmInstr, WasmResult, WasmType};
use bhc_codegen::{
    CodegenConfig, CodegenContext, CodegenError, CodegenModule, CodegenOutputType, CodegenResult,
};
use bhc_session::OptLevel;
use bhc_target::TargetSpec;
use rustc_hash::FxHashMap;
use std::path::Path;

/// A WASM function type signature.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WasmFuncType {
    /// Parameter types.
    pub params: Vec<WasmType>,
    /// Result types.
    pub results: Vec<WasmType>,
}

impl WasmFuncType {
    /// Create a new function type.
    #[must_use]
    pub fn new(params: Vec<WasmType>, results: Vec<WasmType>) -> Self {
        Self { params, results }
    }

    /// Format as WAT.
    #[must_use]
    pub fn to_wat(&self) -> String {
        let params = if self.params.is_empty() {
            String::new()
        } else {
            format!(
                "(param {})",
                self.params
                    .iter()
                    .map(|t| t.wat_name())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };

        let results = if self.results.is_empty() {
            String::new()
        } else {
            format!(
                "(result {})",
                self.results
                    .iter()
                    .map(|t| t.wat_name())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };

        format!("(func {params} {results})").trim().to_string()
    }
}

/// A WASM import.
#[derive(Clone, Debug)]
pub struct WasmImport {
    /// Module name.
    pub module: String,
    /// Field name.
    pub name: String,
    /// Import kind and type.
    pub kind: WasmImportKind,
}

/// Kind of WASM import.
#[derive(Clone, Debug)]
pub enum WasmImportKind {
    /// Function import.
    Func(WasmFuncType),
    /// Memory import (min pages, max pages).
    Memory(u32, Option<u32>),
    /// Global import (type, mutable).
    Global(WasmType, bool),
    /// Table import (min, max).
    Table(u32, Option<u32>),
}

impl WasmImport {
    /// Format as WAT.
    #[must_use]
    pub fn to_wat(&self) -> String {
        let kind_str = match &self.kind {
            WasmImportKind::Func(ty) => format!("(func {})", ty.to_wat()),
            WasmImportKind::Memory(min, max) => match max {
                Some(m) => format!("(memory {min} {m})"),
                None => format!("(memory {min})"),
            },
            WasmImportKind::Global(ty, mutable) => {
                if *mutable {
                    format!("(global (mut {}))", ty.wat_name())
                } else {
                    format!("(global {})", ty.wat_name())
                }
            }
            WasmImportKind::Table(min, max) => match max {
                Some(m) => format!("(table {min} {m} funcref)"),
                None => format!("(table {min} funcref)"),
            },
        };

        format!(
            "(import \"{}\" \"{}\" {})",
            self.module, self.name, kind_str
        )
    }
}

/// A WASM export.
#[derive(Clone, Debug)]
pub struct WasmExport {
    /// Export name.
    pub name: String,
    /// Export kind.
    pub kind: WasmExportKind,
}

/// Kind of WASM export.
#[derive(Clone, Debug)]
pub enum WasmExportKind {
    /// Function export (index).
    Func(u32),
    /// Memory export.
    Memory(u32),
    /// Global export.
    Global(u32),
    /// Table export.
    Table(u32),
}

impl WasmExport {
    /// Format as WAT.
    #[must_use]
    pub fn to_wat(&self) -> String {
        let kind_str = match &self.kind {
            WasmExportKind::Func(idx) => format!("(func {idx})"),
            WasmExportKind::Memory(idx) => format!("(memory {idx})"),
            WasmExportKind::Global(idx) => format!("(global {idx})"),
            WasmExportKind::Table(idx) => format!("(table {idx})"),
        };

        format!("(export \"{}\" {})", self.name, kind_str)
    }
}

/// A WASM function definition.
#[derive(Clone, Debug)]
pub struct WasmFunc {
    /// Function name (optional, for debugging).
    pub name: Option<String>,
    /// Function type.
    pub ty: WasmFuncType,
    /// Local variable types.
    pub locals: Vec<WasmType>,
    /// Function body.
    pub body: Vec<WasmInstr>,
    /// Whether this function is exported.
    pub exported: bool,
    /// Export name (if exported).
    pub export_name: Option<String>,
}

impl WasmFunc {
    /// Create a new function.
    #[must_use]
    pub fn new(ty: WasmFuncType) -> Self {
        Self {
            name: None,
            ty,
            locals: Vec::new(),
            body: Vec::new(),
            exported: false,
            export_name: None,
        }
    }

    /// Set the function name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add a local variable.
    pub fn add_local(&mut self, ty: WasmType) -> u32 {
        let idx = self.ty.params.len() as u32 + self.locals.len() as u32;
        self.locals.push(ty);
        idx
    }

    /// Emit an instruction.
    pub fn emit(&mut self, instr: WasmInstr) {
        self.body.push(instr);
    }

    /// Emit multiple instructions.
    pub fn emit_all(&mut self, instrs: impl IntoIterator<Item = WasmInstr>) {
        self.body.extend(instrs);
    }

    /// Format as WAT.
    #[must_use]
    pub fn to_wat(&self, idx: u32) -> String {
        let mut result = String::new();

        // Function header
        result.push_str("  (func");
        if let Some(name) = &self.name {
            result.push_str(&format!(" ${name}"));
        } else {
            result.push_str(&format!(" $f{idx}"));
        }

        // Parameters
        for (i, ty) in self.ty.params.iter().enumerate() {
            result.push_str(&format!(" (param $p{i} {})", ty.wat_name()));
        }

        // Results
        for ty in &self.ty.results {
            result.push_str(&format!(" (result {})", ty.wat_name()));
        }
        result.push('\n');

        // Locals
        for (i, ty) in self.locals.iter().enumerate() {
            result.push_str(&format!("    (local $l{i} {})\n", ty.wat_name()));
        }

        // Body
        for instr in &self.body {
            result.push_str(&format!("    {}\n", instr.to_wat()));
        }

        result.push_str("  )");
        result
    }
}

/// Memory descriptor for WASM linear memory.
#[derive(Clone, Debug)]
pub struct MemoryDesc {
    /// Minimum pages (64KB each).
    pub min: u32,
    /// Maximum pages (optional).
    pub max: Option<u32>,
    /// Whether memory is shared.
    pub shared: bool,
}

impl Default for MemoryDesc {
    fn default() -> Self {
        Self {
            min: 16,
            max: Some(256),
            shared: false,
        }
    }
}

impl MemoryDesc {
    /// Format as WAT.
    #[must_use]
    pub fn to_wat(&self) -> String {
        match self.max {
            Some(max) => format!("(memory {} {})", self.min, max),
            None => format!("(memory {})", self.min),
        }
    }
}

/// A WASM global variable.
#[derive(Clone, Debug)]
pub struct WasmGlobal {
    /// Global name (optional).
    pub name: Option<String>,
    /// Value type.
    pub ty: WasmType,
    /// Whether mutable.
    pub mutable: bool,
    /// Initial value.
    pub init: WasmInstr,
}

/// Code generation context for WASM.
pub struct WasmCodegenContext {
    /// Codegen configuration.
    config: CodegenConfig,
    /// WASM-specific configuration.
    wasm_config: WasmConfig,
    /// Next function index.
    next_func_idx: u32,
    /// Registered function types.
    func_types: FxHashMap<WasmFuncType, u32>,
}

impl WasmCodegenContext {
    /// Create a new WASM codegen context.
    pub fn new(config: CodegenConfig, wasm_config: WasmConfig) -> Self {
        Self {
            config,
            wasm_config,
            next_func_idx: 0,
            func_types: FxHashMap::default(),
        }
    }

    /// Get the WASM configuration.
    #[must_use]
    pub fn wasm_config(&self) -> &WasmConfig {
        &self.wasm_config
    }

    /// Check if SIMD is enabled.
    #[must_use]
    pub fn simd_enabled(&self) -> bool {
        self.wasm_config.simd_enabled && self.config.target.features.simd128
    }

    /// Allocate a new function index.
    pub fn alloc_func_idx(&mut self) -> u32 {
        let idx = self.next_func_idx;
        self.next_func_idx += 1;
        idx
    }

    /// Register a function type and get its index.
    pub fn register_func_type(&mut self, ty: WasmFuncType) -> u32 {
        if let Some(&idx) = self.func_types.get(&ty) {
            return idx;
        }
        let idx = self.func_types.len() as u32;
        self.func_types.insert(ty, idx);
        idx
    }
}

impl CodegenContext for WasmCodegenContext {
    type Module = WasmModule;

    fn create_module(&self, name: &str) -> CodegenResult<Self::Module> {
        Ok(WasmModule::new(
            name.to_string(),
            self.wasm_config.clone(),
            self.config.target.clone(),
        ))
    }

    fn target(&self) -> &TargetSpec {
        &self.config.target
    }

    fn config(&self) -> &CodegenConfig {
        &self.config
    }
}

/// A WASM module being compiled.
pub struct WasmModule {
    /// Module name.
    name: String,
    /// WASM configuration.
    wasm_config: WasmConfig,
    /// Target specification.
    target: TargetSpec,
    /// Function type definitions.
    types: Vec<WasmFuncType>,
    /// Imports.
    imports: Vec<WasmImport>,
    /// Functions.
    functions: Vec<WasmFunc>,
    /// Memory descriptor.
    memory: MemoryDesc,
    /// Globals.
    globals: Vec<WasmGlobal>,
    /// Exports.
    exports: Vec<WasmExport>,
    /// Data segments.
    data_segments: Vec<(u32, Vec<u8>)>,
}

impl WasmModule {
    /// Create a new WASM module.
    #[must_use]
    pub fn new(name: String, wasm_config: WasmConfig, target: TargetSpec) -> Self {
        let memory = MemoryDesc {
            min: wasm_config.initial_memory_pages,
            max: wasm_config.max_memory_pages,
            shared: false,
        };

        let mut module = Self {
            name,
            wasm_config,
            target,
            types: Vec::new(),
            imports: Vec::new(),
            functions: Vec::new(),
            memory,
            globals: Vec::new(),
            exports: Vec::new(),
            data_segments: Vec::new(),
        };

        // Add runtime exports if configured
        if module.wasm_config.export_memory {
            module.exports.push(WasmExport {
                name: "memory".to_string(),
                kind: WasmExportKind::Memory(0),
            });
        }

        module
    }

    /// Get the module name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a function type definition.
    pub fn add_type(&mut self, ty: WasmFuncType) -> u32 {
        if let Some(idx) = self.types.iter().position(|t| t == &ty) {
            return idx as u32;
        }
        let idx = self.types.len() as u32;
        self.types.push(ty);
        idx
    }

    /// Add an import.
    pub fn add_import(&mut self, import: WasmImport) {
        self.imports.push(import);
    }

    /// Add a function.
    pub fn add_function(&mut self, func: WasmFunc) -> u32 {
        let idx = self
            .imports
            .iter()
            .filter(|i| matches!(i.kind, WasmImportKind::Func(_)))
            .count() as u32
            + self.functions.len() as u32;

        // Add export if function is exported
        if func.exported {
            let export_name = func
                .export_name
                .clone()
                .or_else(|| func.name.clone())
                .unwrap_or_else(|| format!("f{idx}"));
            self.exports.push(WasmExport {
                name: export_name,
                kind: WasmExportKind::Func(idx),
            });
        }

        self.functions.push(func);
        idx
    }

    /// Add a global variable.
    pub fn add_global(&mut self, global: WasmGlobal) -> u32 {
        let idx = self.globals.len() as u32;
        self.globals.push(global);
        idx
    }

    /// Add a data segment.
    pub fn add_data_segment(&mut self, offset: u32, data: Vec<u8>) {
        self.data_segments.push((offset, data));
    }

    /// Add WASI imports for system interface support.
    ///
    /// This adds the standard WASI imports needed for IO and process control:
    /// - `fd_write`: Write to a file descriptor (for stdout/stderr)
    /// - `proc_exit`: Exit the process
    pub fn add_wasi_imports(&mut self) {
        use crate::wasi;
        for import in wasi::generate_wasi_imports() {
            self.add_import(import);
        }
    }

    /// Add runtime functions for WASI programs.
    ///
    /// This adds:
    /// - `alloc`: Bump allocator for linear memory
    /// - `print_i32`: Print an i32 to stdout
    /// - `_start`: Entry point that calls main
    ///
    /// Note: This must be called AFTER `add_wasi_imports()` for correct function indices.
    pub fn add_runtime_functions(&mut self) {
        use crate::wasi;

        // Count imported functions to determine function indices
        let num_imports = self
            .imports
            .iter()
            .filter(|i| matches!(i.kind, WasmImportKind::Func(_)))
            .count() as u32;

        // WASI import indices (these are the first functions)
        let fd_write_idx = wasi::FD_WRITE_IDX;
        let proc_exit_idx = wasi::PROC_EXIT_IDX;

        // Add the heap pointer global
        let heap_ptr_idx = self.add_global(WasmGlobal {
            name: Some("heap_ptr".to_string()),
            ty: WasmType::I32,
            mutable: true,
            init: WasmInstr::I32Const(65536), // Start heap at 64KB
        });

        // Add allocator function (first defined function)
        let alloc_func = wasi::generate_alloc_function(heap_ptr_idx);
        let _alloc_idx = self.add_function(alloc_func);

        // Add print_i32 function
        let print_func = wasi::generate_print_i32(fd_write_idx);
        let _print_i32_idx = self.add_function(print_func);

        // Add a placeholder main function
        let main_func = wasi::generate_placeholder_main();
        let main_idx = self.add_function(main_func);

        // Add _start function (entry point) - must be last so we know main's index
        let start_func = wasi::generate_start_function(main_idx, proc_exit_idx);
        self.add_function(start_func);
    }

    /// Write the WASM binary to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails or the file cannot be written.
    pub fn write_wasm(&self, path: impl AsRef<Path>) -> WasmResult<()> {
        use std::fs;

        let binary = self.to_wasm()?;
        fs::write(path.as_ref(), binary)
            .map_err(|e| crate::WasmError::Internal(format!("failed to write WASM file: {}", e)))
    }

    /// Generate WAT (WebAssembly Text) format.
    #[must_use]
    pub fn to_wat(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("(module ${}\n", self.name));

        // Types
        for (i, ty) in self.types.iter().enumerate() {
            output.push_str(&format!("  (type $t{i} {})\n", ty.to_wat()));
        }

        // Imports
        for import in &self.imports {
            output.push_str(&format!("  {}\n", import.to_wat()));
        }

        // Memory
        output.push_str(&format!("  {}\n", self.memory.to_wat()));

        // Globals
        for (i, global) in self.globals.iter().enumerate() {
            let name = global
                .name
                .as_ref()
                .map_or_else(|| format!("$g{i}"), |n| format!("${n}"));
            let mutability = if global.mutable { "(mut " } else { "" };
            let close = if global.mutable { ")" } else { "" };
            output.push_str(&format!(
                "  (global {} {}{}{} ({}))\n",
                name,
                mutability,
                global.ty.wat_name(),
                close,
                global.init.to_wat()
            ));
        }

        // Functions
        for (i, func) in self.functions.iter().enumerate() {
            output.push_str(&func.to_wat(i as u32));
            output.push('\n');
        }

        // Exports
        for export in &self.exports {
            output.push_str(&format!("  {}\n", export.to_wat()));
        }

        // Data segments
        for (offset, data) in &self.data_segments {
            output.push_str(&format!("  (data (i32.const {offset}) \""));
            for byte in data {
                if *byte >= 32 && *byte < 127 && *byte != b'"' && *byte != b'\\' {
                    output.push(*byte as char);
                } else {
                    output.push_str(&format!("\\{byte:02x}"));
                }
            }
            output.push_str("\")\n");
        }

        output.push_str(")\n");
        output
    }

    /// Generate WASM binary format.
    ///
    /// This is a simplified binary encoder. For production use, consider
    /// using a library like `wasm-encoder`.
    pub fn to_wasm(&self) -> WasmResult<Vec<u8>> {
        let mut encoder = BinaryEncoder::new();
        encoder.encode_module(self)?;
        Ok(encoder.finish())
    }
}

impl CodegenModule for WasmModule {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify(&self) -> CodegenResult<()> {
        // Basic validation
        for func in &self.functions {
            // Ensure all functions end with End instruction
            if let Some(last) = func.body.last() {
                if !matches!(last, WasmInstr::End | WasmInstr::Return) {
                    return Err(CodegenError::Internal(format!(
                        "Function '{}' does not end with end or return",
                        func.name.as_deref().unwrap_or("anonymous")
                    )));
                }
            }
        }
        Ok(())
    }

    fn optimize(&mut self, level: OptLevel) -> CodegenResult<()> {
        match level {
            OptLevel::None => {}
            OptLevel::Less
            | OptLevel::Default
            | OptLevel::Aggressive
            | OptLevel::Size
            | OptLevel::SizeMin => {
                // Basic optimizations
                for func in &mut self.functions {
                    optimize_function(func);
                }
            }
        }

        // Edge profile optimizations
        if self.wasm_config.optimize_size {
            // Remove debug names
            for func in &mut self.functions {
                func.name = None;
            }
            for global in &mut self.globals {
                global.name = None;
            }

            // Remove comments
            for func in &mut self.functions {
                func.body.retain(|i| !matches!(i, WasmInstr::Comment(_)));
            }
        }

        Ok(())
    }

    fn write_to_file(&self, path: &Path, output_type: CodegenOutputType) -> CodegenResult<()> {
        use std::fs;

        let content = match output_type {
            CodegenOutputType::Object => {
                // Write WASM binary
                self.to_wasm()
                    .map_err(|e| CodegenError::Internal(e.to_string()))?
            }
            CodegenOutputType::Assembly => {
                // Write WAT text
                self.to_wat().into_bytes()
            }
            CodegenOutputType::LlvmIr | CodegenOutputType::LlvmBitcode => {
                return Err(CodegenError::Internal(
                    "WASM module does not produce LLVM output".to_string(),
                ));
            }
        };

        fs::write(path, content).map_err(|e| CodegenError::OutputError {
            path: path.display().to_string(),
            source: e,
        })
    }

    fn as_llvm_ir(&self) -> CodegenResult<String> {
        Err(CodegenError::Internal(
            "WASM modules use WAT/WASM format, not LLVM IR".to_string(),
        ))
    }
}

/// Basic function-level optimizations.
fn optimize_function(func: &mut WasmFunc) {
    // Remove consecutive nops
    func.body.retain(|i| !matches!(i, WasmInstr::Nop));

    // TODO: Add more optimizations:
    // - Constant folding
    // - Dead code elimination
    // - Local variable coalescing
}

/// Binary encoder for WASM module format.
struct BinaryEncoder {
    output: Vec<u8>,
}

impl BinaryEncoder {
    fn new() -> Self {
        Self { output: Vec::new() }
    }

    fn finish(self) -> Vec<u8> {
        self.output
    }

    fn encode_module(&mut self, module: &WasmModule) -> WasmResult<()> {
        // WASM magic number
        self.output.extend_from_slice(b"\x00asm");
        // Version 1
        self.output.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]);

        // Build unified type table for consistent type index references
        let type_table = Self::collect_type_table(module);

        // Type section
        self.encode_type_section(&type_table)?;

        // Import section
        self.encode_import_section(module, &type_table)?;

        // Function section
        self.encode_function_section(module, &type_table)?;

        // Memory section
        self.encode_memory_section(module)?;

        // Global section
        self.encode_global_section(module)?;

        // Export section
        self.encode_export_section(module)?;

        // Code section
        self.encode_code_section(module)?;

        // Data section
        self.encode_data_section(module)?;

        Ok(())
    }

    /// Encode a memory operation's alignment and offset.
    /// WASM binary format requires alignment as log2(byte_alignment).
    fn encode_memarg(&mut self, byte_align: u32, offset: u32) {
        let log2_align = if byte_align == 0 {
            0
        } else {
            byte_align.trailing_zeros()
        };
        self.encode_uleb128(log2_align);
        self.encode_uleb128(offset);
    }

    fn encode_uleb128(&mut self, mut value: u32) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                self.output.push(byte);
                break;
            }
            self.output.push(byte | 0x80);
        }
    }

    fn encode_sleb128(&mut self, mut value: i32) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
            if done {
                self.output.push(byte);
                break;
            }
            self.output.push(byte | 0x80);
        }
    }

    fn encode_sleb128_64(&mut self, mut value: i64) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
            if done {
                self.output.push(byte);
                break;
            }
            self.output.push(byte | 0x80);
        }
    }

    fn encode_string(&mut self, s: &str) {
        self.encode_uleb128(s.len() as u32);
        self.output.extend_from_slice(s.as_bytes());
    }

    fn encode_type(&mut self, ty: WasmType) {
        let byte = match ty {
            WasmType::I32 => 0x7F,
            WasmType::I64 => 0x7E,
            WasmType::F32 => 0x7D,
            WasmType::F64 => 0x7C,
            WasmType::V128 => 0x7B,
            WasmType::FuncRef => 0x70,
            WasmType::ExternRef => 0x6F,
        };
        self.output.push(byte);
    }

    fn encode_section(&mut self, section_id: u8, content: Vec<u8>) {
        self.output.push(section_id);
        self.encode_uleb128(content.len() as u32);
        self.output.extend(content);
    }

    /// Collect all unique function types from the module and return them with
    /// an index map for lookup. The type table includes types from:
    /// 1. Explicit module types
    /// 2. Imported function types
    /// 3. Module function types
    fn collect_type_table(module: &WasmModule) -> Vec<WasmFuncType> {
        let mut types: Vec<WasmFuncType> = module.types.clone();
        // Add import function types
        for import in &module.imports {
            if let WasmImportKind::Func(ty) = &import.kind {
                if !types.contains(ty) {
                    types.push(ty.clone());
                }
            }
        }
        // Add module function types
        for func in &module.functions {
            if !types.contains(&func.ty) {
                types.push(func.ty.clone());
            }
        }
        types
    }

    /// Find the index of a function type in the type table.
    fn find_type_index(type_table: &[WasmFuncType], ty: &WasmFuncType) -> u32 {
        type_table.iter().position(|t| t == ty).unwrap_or(0) as u32
    }

    fn encode_type_section(&mut self, type_table: &[WasmFuncType]) -> WasmResult<()> {
        if type_table.is_empty() {
            return Ok(());
        }

        let mut content = Vec::new();
        let mut encoder = BinaryEncoder { output: content };

        encoder.encode_uleb128(type_table.len() as u32);
        for ty in type_table {
            encoder.output.push(0x60); // func type
            encoder.encode_uleb128(ty.params.len() as u32);
            for param in &ty.params {
                encoder.encode_type(*param);
            }
            encoder.encode_uleb128(ty.results.len() as u32);
            for result in &ty.results {
                encoder.encode_type(*result);
            }
        }

        content = encoder.output;
        self.encode_section(0x01, content);
        Ok(())
    }

    fn encode_import_section(
        &mut self,
        module: &WasmModule,
        type_table: &[WasmFuncType],
    ) -> WasmResult<()> {
        if module.imports.is_empty() {
            return Ok(());
        }

        let mut content = Vec::new();
        let mut encoder = BinaryEncoder { output: content };

        encoder.encode_uleb128(module.imports.len() as u32);
        for import in &module.imports {
            encoder.encode_string(&import.module);
            encoder.encode_string(&import.name);

            match &import.kind {
                WasmImportKind::Func(ty) => {
                    encoder.output.push(0x00); // func import
                    encoder.encode_uleb128(Self::find_type_index(type_table, ty));
                }
                WasmImportKind::Memory(min, max) => {
                    encoder.output.push(0x02); // memory import
                    if let Some(m) = max {
                        encoder.output.push(0x01); // has max
                        encoder.encode_uleb128(*min);
                        encoder.encode_uleb128(*m);
                    } else {
                        encoder.output.push(0x00); // no max
                        encoder.encode_uleb128(*min);
                    }
                }
                WasmImportKind::Global(ty, mutable) => {
                    encoder.output.push(0x03); // global import
                    encoder.encode_type(*ty);
                    encoder.output.push(if *mutable { 0x01 } else { 0x00 });
                }
                WasmImportKind::Table(min, max) => {
                    encoder.output.push(0x01); // table import
                    encoder.output.push(0x70); // funcref
                    if let Some(m) = max {
                        encoder.output.push(0x01);
                        encoder.encode_uleb128(*min);
                        encoder.encode_uleb128(*m);
                    } else {
                        encoder.output.push(0x00);
                        encoder.encode_uleb128(*min);
                    }
                }
            }
        }

        content = encoder.output;
        self.encode_section(0x02, content);
        Ok(())
    }

    fn encode_function_section(
        &mut self,
        module: &WasmModule,
        type_table: &[WasmFuncType],
    ) -> WasmResult<()> {
        if module.functions.is_empty() {
            return Ok(());
        }

        let mut content = Vec::new();
        let mut encoder = BinaryEncoder { output: content };

        encoder.encode_uleb128(module.functions.len() as u32);
        for func in &module.functions {
            encoder.encode_uleb128(Self::find_type_index(type_table, &func.ty));
        }

        content = encoder.output;
        self.encode_section(0x03, content);
        Ok(())
    }

    fn encode_memory_section(&mut self, module: &WasmModule) -> WasmResult<()> {
        let mut content = Vec::new();
        let mut encoder = BinaryEncoder { output: content };

        encoder.encode_uleb128(1); // 1 memory
        if let Some(max) = module.memory.max {
            encoder.output.push(0x01); // has max
            encoder.encode_uleb128(module.memory.min);
            encoder.encode_uleb128(max);
        } else {
            encoder.output.push(0x00); // no max
            encoder.encode_uleb128(module.memory.min);
        }

        content = encoder.output;
        self.encode_section(0x05, content);
        Ok(())
    }

    fn encode_global_section(&mut self, module: &WasmModule) -> WasmResult<()> {
        if module.globals.is_empty() {
            return Ok(());
        }

        let mut content = Vec::new();
        let mut encoder = BinaryEncoder { output: content };

        encoder.encode_uleb128(module.globals.len() as u32);
        for global in &module.globals {
            encoder.encode_type(global.ty);
            encoder
                .output
                .push(if global.mutable { 0x01 } else { 0x00 });
            encoder.encode_instr(&global.init)?;
            encoder.output.push(0x0B); // end
        }

        content = encoder.output;
        self.encode_section(0x06, content);
        Ok(())
    }

    fn encode_export_section(&mut self, module: &WasmModule) -> WasmResult<()> {
        if module.exports.is_empty() {
            return Ok(());
        }

        let mut content = Vec::new();
        let mut encoder = BinaryEncoder { output: content };

        encoder.encode_uleb128(module.exports.len() as u32);
        for export in &module.exports {
            encoder.encode_string(&export.name);
            match &export.kind {
                WasmExportKind::Func(idx) => {
                    encoder.output.push(0x00);
                    encoder.encode_uleb128(*idx);
                }
                WasmExportKind::Memory(idx) => {
                    encoder.output.push(0x02);
                    encoder.encode_uleb128(*idx);
                }
                WasmExportKind::Global(idx) => {
                    encoder.output.push(0x03);
                    encoder.encode_uleb128(*idx);
                }
                WasmExportKind::Table(idx) => {
                    encoder.output.push(0x01);
                    encoder.encode_uleb128(*idx);
                }
            }
        }

        content = encoder.output;
        self.encode_section(0x07, content);
        Ok(())
    }

    fn encode_code_section(&mut self, module: &WasmModule) -> WasmResult<()> {
        if module.functions.is_empty() {
            return Ok(());
        }

        let mut content = Vec::new();
        let mut encoder = BinaryEncoder { output: content };

        encoder.encode_uleb128(module.functions.len() as u32);
        for func in &module.functions {
            let mut func_body = Vec::new();
            let mut func_encoder = BinaryEncoder { output: func_body };

            // Locals
            func_encoder.encode_uleb128(func.locals.len() as u32);
            for local in &func.locals {
                func_encoder.encode_uleb128(1); // count
                func_encoder.encode_type(*local);
            }

            // Body
            for instr in &func.body {
                func_encoder.encode_instr(instr)?;
            }

            func_body = func_encoder.output;
            encoder.encode_uleb128(func_body.len() as u32);
            encoder.output.extend(func_body);
        }

        content = encoder.output;
        self.encode_section(0x0A, content);
        Ok(())
    }

    fn encode_data_section(&mut self, module: &WasmModule) -> WasmResult<()> {
        if module.data_segments.is_empty() {
            return Ok(());
        }

        let mut content = Vec::new();
        let mut encoder = BinaryEncoder { output: content };

        encoder.encode_uleb128(module.data_segments.len() as u32);
        for (offset, data) in &module.data_segments {
            encoder.output.push(0x00); // active segment, memory 0
            encoder.output.push(0x41); // i32.const
            encoder.encode_sleb128(*offset as i32);
            encoder.output.push(0x0B); // end
            encoder.encode_uleb128(data.len() as u32);
            encoder.output.extend(data);
        }

        content = encoder.output;
        self.encode_section(0x0B, content);
        Ok(())
    }

    fn encode_instr(&mut self, instr: &WasmInstr) -> WasmResult<()> {
        match instr {
            // Control
            WasmInstr::Unreachable => self.output.push(0x00),
            WasmInstr::Nop => self.output.push(0x01),
            WasmInstr::Block(ty) => {
                self.output.push(0x02);
                self.encode_block_type(ty);
            }
            WasmInstr::Loop(ty) => {
                self.output.push(0x03);
                self.encode_block_type(ty);
            }
            WasmInstr::If(ty) => {
                self.output.push(0x04);
                self.encode_block_type(ty);
            }
            WasmInstr::Else => self.output.push(0x05),
            WasmInstr::End => self.output.push(0x0B),
            WasmInstr::Br(l) => {
                self.output.push(0x0C);
                self.encode_uleb128(*l);
            }
            WasmInstr::BrIf(l) => {
                self.output.push(0x0D);
                self.encode_uleb128(*l);
            }
            WasmInstr::BrTable(labels, default) => {
                self.output.push(0x0E);
                self.encode_uleb128(labels.len() as u32);
                for l in labels {
                    self.encode_uleb128(*l);
                }
                self.encode_uleb128(*default);
            }
            WasmInstr::Return => self.output.push(0x0F),
            WasmInstr::Call(idx) => {
                self.output.push(0x10);
                self.encode_uleb128(*idx);
            }
            WasmInstr::CallIndirect(ty, table) => {
                self.output.push(0x11);
                self.encode_uleb128(*ty);
                self.encode_uleb128(*table);
            }

            // Parametric
            WasmInstr::Drop => self.output.push(0x1A),
            WasmInstr::Select => self.output.push(0x1B),

            // Variables
            WasmInstr::LocalGet(idx) => {
                self.output.push(0x20);
                self.encode_uleb128(*idx);
            }
            WasmInstr::LocalSet(idx) => {
                self.output.push(0x21);
                self.encode_uleb128(*idx);
            }
            WasmInstr::LocalTee(idx) => {
                self.output.push(0x22);
                self.encode_uleb128(*idx);
            }
            WasmInstr::GlobalGet(idx) => {
                self.output.push(0x23);
                self.encode_uleb128(*idx);
            }
            WasmInstr::GlobalSet(idx) => {
                self.output.push(0x24);
                self.encode_uleb128(*idx);
            }

            // Memory (alignment is in bytes, encode as log2)
            WasmInstr::I32Load(align, offset) => {
                self.output.push(0x28);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::I64Load(align, offset) => {
                self.output.push(0x29);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::F32Load(align, offset) => {
                self.output.push(0x2A);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::F64Load(align, offset) => {
                self.output.push(0x2B);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::I32Load8S(align, offset) => {
                self.output.push(0x2C);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::I32Load8U(align, offset) => {
                self.output.push(0x2D);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::I32Load16S(align, offset) => {
                self.output.push(0x2E);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::I32Load16U(align, offset) => {
                self.output.push(0x2F);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::I32Store(align, offset) => {
                self.output.push(0x36);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::I64Store(align, offset) => {
                self.output.push(0x37);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::F32Store(align, offset) => {
                self.output.push(0x38);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::F64Store(align, offset) => {
                self.output.push(0x39);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::I32Store8(align, offset) => {
                self.output.push(0x3A);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::I32Store16(align, offset) => {
                self.output.push(0x3B);
                self.encode_memarg(*align, *offset);
            }
            WasmInstr::MemorySize => {
                self.output.push(0x3F);
                self.output.push(0x00); // memory index
            }
            WasmInstr::MemoryGrow => {
                self.output.push(0x40);
                self.output.push(0x00); // memory index
            }

            // i32 numeric
            WasmInstr::I32Const(v) => {
                self.output.push(0x41);
                self.encode_sleb128(*v);
            }
            WasmInstr::I32Eqz => self.output.push(0x45),
            WasmInstr::I32Eq => self.output.push(0x46),
            WasmInstr::I32Ne => self.output.push(0x47),
            WasmInstr::I32LtS => self.output.push(0x48),
            WasmInstr::I32LtU => self.output.push(0x49),
            WasmInstr::I32GtS => self.output.push(0x4A),
            WasmInstr::I32GtU => self.output.push(0x4B),
            WasmInstr::I32LeS => self.output.push(0x4C),
            WasmInstr::I32LeU => self.output.push(0x4D),
            WasmInstr::I32GeS => self.output.push(0x4E),
            WasmInstr::I32GeU => self.output.push(0x4F),
            WasmInstr::I32Add => self.output.push(0x6A),
            WasmInstr::I32Sub => self.output.push(0x6B),
            WasmInstr::I32Mul => self.output.push(0x6C),
            WasmInstr::I32DivS => self.output.push(0x6D),
            WasmInstr::I32DivU => self.output.push(0x6E),
            WasmInstr::I32RemS => self.output.push(0x6F),
            WasmInstr::I32RemU => self.output.push(0x70),
            WasmInstr::I32And => self.output.push(0x71),
            WasmInstr::I32Or => self.output.push(0x72),
            WasmInstr::I32Xor => self.output.push(0x73),
            WasmInstr::I32Shl => self.output.push(0x74),
            WasmInstr::I32ShrS => self.output.push(0x75),
            WasmInstr::I32ShrU => self.output.push(0x76),

            // i64 numeric
            WasmInstr::I64Const(v) => {
                self.output.push(0x42);
                self.encode_sleb128_64(*v);
            }
            WasmInstr::I64Eqz => self.output.push(0x50),
            WasmInstr::I64Eq => self.output.push(0x51),
            WasmInstr::I64Ne => self.output.push(0x52),
            WasmInstr::I64LtS => self.output.push(0x53),
            WasmInstr::I64LtU => self.output.push(0x54),
            WasmInstr::I64GtS => self.output.push(0x55),
            WasmInstr::I64GtU => self.output.push(0x56),
            WasmInstr::I64LeS => self.output.push(0x57),
            WasmInstr::I64LeU => self.output.push(0x58),
            WasmInstr::I64GeS => self.output.push(0x59),
            WasmInstr::I64GeU => self.output.push(0x5A),
            WasmInstr::I64Add => self.output.push(0x7C),
            WasmInstr::I64Sub => self.output.push(0x7D),
            WasmInstr::I64Mul => self.output.push(0x7E),
            WasmInstr::I64DivS => self.output.push(0x7F),
            WasmInstr::I64DivU => self.output.push(0x80),
            WasmInstr::I64RemS => self.output.push(0x81),
            WasmInstr::I64RemU => self.output.push(0x82),
            WasmInstr::I64And => self.output.push(0x83),
            WasmInstr::I64Or => self.output.push(0x84),
            WasmInstr::I64Xor => self.output.push(0x85),
            WasmInstr::I64Shl => self.output.push(0x86),
            WasmInstr::I64ShrS => self.output.push(0x87),
            WasmInstr::I64ShrU => self.output.push(0x88),

            // f32 numeric
            WasmInstr::F32Const(v) => {
                self.output.push(0x43);
                self.output.extend_from_slice(&v.to_le_bytes());
            }
            WasmInstr::F32Eq => self.output.push(0x5B),
            WasmInstr::F32Ne => self.output.push(0x5C),
            WasmInstr::F32Lt => self.output.push(0x5D),
            WasmInstr::F32Gt => self.output.push(0x5E),
            WasmInstr::F32Le => self.output.push(0x5F),
            WasmInstr::F32Ge => self.output.push(0x60),
            WasmInstr::F32Abs => self.output.push(0x8B),
            WasmInstr::F32Neg => self.output.push(0x8C),
            WasmInstr::F32Ceil => self.output.push(0x8D),
            WasmInstr::F32Floor => self.output.push(0x8E),
            WasmInstr::F32Trunc => self.output.push(0x8F),
            WasmInstr::F32Nearest => self.output.push(0x90),
            WasmInstr::F32Sqrt => self.output.push(0x91),
            WasmInstr::F32Add => self.output.push(0x92),
            WasmInstr::F32Sub => self.output.push(0x93),
            WasmInstr::F32Mul => self.output.push(0x94),
            WasmInstr::F32Div => self.output.push(0x95),
            WasmInstr::F32Min => self.output.push(0x96),
            WasmInstr::F32Max => self.output.push(0x97),
            WasmInstr::F32Copysign => self.output.push(0x98),

            // f64 numeric
            WasmInstr::F64Const(v) => {
                self.output.push(0x44);
                self.output.extend_from_slice(&v.to_le_bytes());
            }
            WasmInstr::F64Eq => self.output.push(0x61),
            WasmInstr::F64Ne => self.output.push(0x62),
            WasmInstr::F64Lt => self.output.push(0x63),
            WasmInstr::F64Gt => self.output.push(0x64),
            WasmInstr::F64Le => self.output.push(0x65),
            WasmInstr::F64Ge => self.output.push(0x66),
            WasmInstr::F64Abs => self.output.push(0x99),
            WasmInstr::F64Neg => self.output.push(0x9A),
            WasmInstr::F64Ceil => self.output.push(0x9B),
            WasmInstr::F64Floor => self.output.push(0x9C),
            WasmInstr::F64Trunc => self.output.push(0x9D),
            WasmInstr::F64Nearest => self.output.push(0x9E),
            WasmInstr::F64Sqrt => self.output.push(0x9F),
            WasmInstr::F64Add => self.output.push(0xA0),
            WasmInstr::F64Sub => self.output.push(0xA1),
            WasmInstr::F64Mul => self.output.push(0xA2),
            WasmInstr::F64Div => self.output.push(0xA3),
            WasmInstr::F64Min => self.output.push(0xA4),
            WasmInstr::F64Max => self.output.push(0xA5),
            WasmInstr::F64Copysign => self.output.push(0xA6),

            // Conversions
            WasmInstr::I32WrapI64 => self.output.push(0xA7),
            WasmInstr::I32TruncF32S => self.output.push(0xA8),
            WasmInstr::I32TruncF32U => self.output.push(0xA9),
            WasmInstr::I32TruncF64S => self.output.push(0xAA),
            WasmInstr::I32TruncF64U => self.output.push(0xAB),
            WasmInstr::I64ExtendI32S => self.output.push(0xAC),
            WasmInstr::I64ExtendI32U => self.output.push(0xAD),
            WasmInstr::I64TruncF32S => self.output.push(0xAE),
            WasmInstr::I64TruncF32U => self.output.push(0xAF),
            WasmInstr::I64TruncF64S => self.output.push(0xB0),
            WasmInstr::I64TruncF64U => self.output.push(0xB1),
            WasmInstr::F32ConvertI32S => self.output.push(0xB2),
            WasmInstr::F32ConvertI32U => self.output.push(0xB3),
            WasmInstr::F32ConvertI64S => self.output.push(0xB4),
            WasmInstr::F32ConvertI64U => self.output.push(0xB5),
            WasmInstr::F32DemoteF64 => self.output.push(0xB6),
            WasmInstr::F64ConvertI32S => self.output.push(0xB7),
            WasmInstr::F64ConvertI32U => self.output.push(0xB8),
            WasmInstr::F64ConvertI64S => self.output.push(0xB9),
            WasmInstr::F64ConvertI64U => self.output.push(0xBA),
            WasmInstr::F64PromoteF32 => self.output.push(0xBB),
            WasmInstr::I32ReinterpretF32 => self.output.push(0xBC),
            WasmInstr::I64ReinterpretF64 => self.output.push(0xBD),
            WasmInstr::F32ReinterpretI32 => self.output.push(0xBE),
            WasmInstr::F64ReinterpretI64 => self.output.push(0xBF),

            // SIMD prefix byte is 0xFD
            WasmInstr::V128Load(align, offset) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x00);
                self.encode_uleb128(*align);
                self.encode_uleb128(*offset);
            }
            WasmInstr::V128Store(align, offset) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x0B);
                self.encode_uleb128(*align);
                self.encode_uleb128(*offset);
            }
            WasmInstr::V128Const(bytes) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x0C);
                self.output.extend_from_slice(bytes);
            }
            WasmInstr::I8x16Shuffle(lanes) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x0D);
                self.output.extend_from_slice(lanes);
            }
            WasmInstr::I8x16Splat => {
                self.output.push(0xFD);
                self.encode_uleb128(0x0F);
            }
            WasmInstr::I16x8Splat => {
                self.output.push(0xFD);
                self.encode_uleb128(0x10);
            }
            WasmInstr::I32x4Splat => {
                self.output.push(0xFD);
                self.encode_uleb128(0x11);
            }
            WasmInstr::I64x2Splat => {
                self.output.push(0xFD);
                self.encode_uleb128(0x12);
            }
            WasmInstr::F32x4Splat => {
                self.output.push(0xFD);
                self.encode_uleb128(0x13);
            }
            WasmInstr::F64x2Splat => {
                self.output.push(0xFD);
                self.encode_uleb128(0x14);
            }
            WasmInstr::I8x16ExtractLaneS(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x15);
                self.output.push(*lane);
            }
            WasmInstr::I8x16ExtractLaneU(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x16);
                self.output.push(*lane);
            }
            WasmInstr::I16x8ExtractLaneS(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x18);
                self.output.push(*lane);
            }
            WasmInstr::I16x8ExtractLaneU(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x19);
                self.output.push(*lane);
            }
            WasmInstr::I32x4ExtractLane(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x1B);
                self.output.push(*lane);
            }
            WasmInstr::I64x2ExtractLane(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x1D);
                self.output.push(*lane);
            }
            WasmInstr::F32x4ExtractLane(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x1F);
                self.output.push(*lane);
            }
            WasmInstr::F64x2ExtractLane(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x21);
                self.output.push(*lane);
            }
            WasmInstr::I32x4ReplaceLane(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x1C);
                self.output.push(*lane);
            }
            WasmInstr::F32x4ReplaceLane(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x20);
                self.output.push(*lane);
            }
            WasmInstr::F64x2ReplaceLane(lane) => {
                self.output.push(0xFD);
                self.encode_uleb128(0x22);
                self.output.push(*lane);
            }

            // SIMD arithmetic
            WasmInstr::F32x4Add => {
                self.output.push(0xFD);
                self.encode_uleb128(0xE4);
            }
            WasmInstr::F32x4Sub => {
                self.output.push(0xFD);
                self.encode_uleb128(0xE5);
            }
            WasmInstr::F32x4Mul => {
                self.output.push(0xFD);
                self.encode_uleb128(0xE6);
            }
            WasmInstr::F32x4Div => {
                self.output.push(0xFD);
                self.encode_uleb128(0xE7);
            }
            WasmInstr::F32x4Min => {
                self.output.push(0xFD);
                self.encode_uleb128(0xE8);
            }
            WasmInstr::F32x4Max => {
                self.output.push(0xFD);
                self.encode_uleb128(0xE9);
            }
            WasmInstr::F32x4Abs => {
                self.output.push(0xFD);
                self.encode_uleb128(0xE0);
            }
            WasmInstr::F32x4Neg => {
                self.output.push(0xFD);
                self.encode_uleb128(0xE1);
            }
            WasmInstr::F32x4Sqrt => {
                self.output.push(0xFD);
                self.encode_uleb128(0xE3);
            }
            WasmInstr::F32x4Ceil => {
                self.output.push(0xFD);
                self.encode_uleb128(0x67);
            }
            WasmInstr::F32x4Floor => {
                self.output.push(0xFD);
                self.encode_uleb128(0x68);
            }

            WasmInstr::F64x2Add => {
                self.output.push(0xFD);
                self.encode_uleb128(0xF0);
            }
            WasmInstr::F64x2Sub => {
                self.output.push(0xFD);
                self.encode_uleb128(0xF1);
            }
            WasmInstr::F64x2Mul => {
                self.output.push(0xFD);
                self.encode_uleb128(0xF2);
            }
            WasmInstr::F64x2Div => {
                self.output.push(0xFD);
                self.encode_uleb128(0xF3);
            }
            WasmInstr::F64x2Min => {
                self.output.push(0xFD);
                self.encode_uleb128(0xF4);
            }
            WasmInstr::F64x2Max => {
                self.output.push(0xFD);
                self.encode_uleb128(0xF5);
            }
            WasmInstr::F64x2Abs => {
                self.output.push(0xFD);
                self.encode_uleb128(0xEC);
            }
            WasmInstr::F64x2Neg => {
                self.output.push(0xFD);
                self.encode_uleb128(0xED);
            }
            WasmInstr::F64x2Sqrt => {
                self.output.push(0xFD);
                self.encode_uleb128(0xEF);
            }

            WasmInstr::I32x4Add => {
                self.output.push(0xFD);
                self.encode_uleb128(0xAE);
            }
            WasmInstr::I32x4Sub => {
                self.output.push(0xFD);
                self.encode_uleb128(0xB1);
            }
            WasmInstr::I32x4Mul => {
                self.output.push(0xFD);
                self.encode_uleb128(0xB5);
            }
            WasmInstr::I32x4Neg => {
                self.output.push(0xFD);
                self.encode_uleb128(0xA1);
            }
            WasmInstr::I32x4Shl => {
                self.output.push(0xFD);
                self.encode_uleb128(0xAB);
            }
            WasmInstr::I32x4ShrS => {
                self.output.push(0xFD);
                self.encode_uleb128(0xAC);
            }
            WasmInstr::I32x4ShrU => {
                self.output.push(0xFD);
                self.encode_uleb128(0xAD);
            }

            // SIMD bitwise
            WasmInstr::V128And => {
                self.output.push(0xFD);
                self.encode_uleb128(0x4E);
            }
            WasmInstr::V128Or => {
                self.output.push(0xFD);
                self.encode_uleb128(0x50);
            }
            WasmInstr::V128Xor => {
                self.output.push(0xFD);
                self.encode_uleb128(0x51);
            }
            WasmInstr::V128Not => {
                self.output.push(0xFD);
                self.encode_uleb128(0x4D);
            }
            WasmInstr::V128AndNot => {
                self.output.push(0xFD);
                self.encode_uleb128(0x4F);
            }
            WasmInstr::V128AnyTrue => {
                self.output.push(0xFD);
                self.encode_uleb128(0x53);
            }

            // Comments are not encoded
            WasmInstr::Comment(_) => {}
        }
        Ok(())
    }

    fn encode_block_type(&mut self, ty: &Option<WasmType>) {
        match ty {
            None => self.output.push(0x40), // void
            Some(WasmType::I32) => self.output.push(0x7F),
            Some(WasmType::I64) => self.output.push(0x7E),
            Some(WasmType::F32) => self.output.push(0x7D),
            Some(WasmType::F64) => self.output.push(0x7C),
            Some(WasmType::V128) => self.output.push(0x7B),
            Some(WasmType::FuncRef) => self.output.push(0x70),
            Some(WasmType::ExternRef) => self.output.push(0x6F),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_func_type_to_wat() {
        let ty = WasmFuncType::new(vec![WasmType::I32, WasmType::I32], vec![WasmType::I32]);
        assert!(ty.to_wat().contains("param"));
        assert!(ty.to_wat().contains("result"));
    }

    #[test]
    fn test_wasm_module_creation() {
        let config = WasmConfig::default();
        let target = bhc_target::targets::wasm32_wasi();
        let module = WasmModule::new("test".to_string(), config, target);

        assert_eq!(module.name(), "test");
    }

    #[test]
    fn test_wasm_module_to_wat() {
        let config = WasmConfig::default();
        let target = bhc_target::targets::wasm32_wasi();
        let mut module = WasmModule::new("test".to_string(), config, target);

        // Add a simple function
        let mut func = WasmFunc::new(WasmFuncType::new(
            vec![WasmType::I32, WasmType::I32],
            vec![WasmType::I32],
        ));
        func.name = Some("add".to_string());
        func.exported = true;
        func.emit(WasmInstr::LocalGet(0));
        func.emit(WasmInstr::LocalGet(1));
        func.emit(WasmInstr::I32Add);
        func.emit(WasmInstr::End);

        module.add_function(func);

        let wat = module.to_wat();
        assert!(wat.contains("(module $test"));
        assert!(wat.contains("$add"));
        assert!(wat.contains("i32.add"));
    }

    #[test]
    fn test_wasm_module_to_binary() {
        let config = WasmConfig::default();
        let target = bhc_target::targets::wasm32_wasi();
        let mut module = WasmModule::new("test".to_string(), config, target);

        let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![WasmType::I32]));
        func.emit(WasmInstr::I32Const(42));
        func.emit(WasmInstr::End);
        module.add_function(func);

        let binary = module.to_wasm().unwrap();

        // Check WASM magic number
        assert_eq!(&binary[0..4], b"\x00asm");
        // Check version
        assert_eq!(&binary[4..8], &[0x01, 0x00, 0x00, 0x00]);
    }
}
