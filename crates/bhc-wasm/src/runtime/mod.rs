//! WASM runtime support.
//!
//! This module provides runtime primitives for BHC programs running in
//! WebAssembly, including memory management, arena allocation, garbage
//! collection, and host function imports.

mod arena;
pub mod gc;
mod memory;

pub use arena::{ArenaConfig, WasmArena};
pub use gc::GcConfig;
pub use memory::{LinearMemory, MemoryLayout};

use crate::{WasmError, WasmInstr, WasmResult};

/// WASM page size (64KB).
pub const PAGE_SIZE: u32 = 65536;

/// Default stack size (64KB).
pub const DEFAULT_STACK_SIZE: u32 = 65536;

/// Default heap start (after stack).
pub const DEFAULT_HEAP_START: u32 = DEFAULT_STACK_SIZE;

/// Runtime configuration for WASM modules.
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Initial memory pages.
    pub initial_pages: u32,
    /// Maximum memory pages.
    pub max_pages: Option<u32>,
    /// Stack size in bytes.
    pub stack_size: u32,
    /// Enable arena allocator.
    pub enable_arena: bool,
    /// Arena size in bytes.
    pub arena_size: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            initial_pages: 32, // 2MB (room for stack + arena)
            max_pages: Some(256), // 16MB
            stack_size: DEFAULT_STACK_SIZE,
            enable_arena: true,
            arena_size: 1024 * 1024, // 1MB arena
        }
    }
}

impl RuntimeConfig {
    /// Configuration for Edge profile (minimal footprint).
    #[must_use]
    pub fn edge() -> Self {
        Self {
            initial_pages: 8, // 512KB (room for stack + arena)
            max_pages: Some(64), // 4MB
            stack_size: 32768, // 32KB
            enable_arena: true,
            arena_size: 256 * 1024, // 256KB arena
        }
    }

    /// Calculate the total initial memory requirement.
    #[must_use]
    pub fn total_initial_memory(&self) -> u32 {
        let arena_size = if self.enable_arena { self.arena_size } else { 0 };
        self.stack_size + arena_size
    }

    /// Validate the configuration.
    pub fn validate(&self) -> WasmResult<()> {
        if self.initial_pages == 0 {
            return Err(WasmError::MemoryError(
                "Initial pages must be greater than 0".to_string()
            ));
        }

        if let Some(max) = self.max_pages {
            if self.initial_pages > max {
                return Err(WasmError::MemoryError(
                    "Initial pages exceeds maximum pages".to_string()
                ));
            }
        }

        let required_bytes = self.total_initial_memory();
        let available_bytes = self.initial_pages * PAGE_SIZE;
        if required_bytes > available_bytes {
            return Err(WasmError::MemoryError(format!(
                "Required memory ({} bytes) exceeds initial allocation ({} bytes)",
                required_bytes, available_bytes
            )));
        }

        Ok(())
    }
}

/// Runtime function imports that BHC modules may need.
#[derive(Clone, Debug)]
pub struct RuntimeImports {
    /// Import module name for runtime functions.
    pub module_name: String,
    /// Enable console logging.
    pub enable_console: bool,
    /// Enable timing functions.
    pub enable_timing: bool,
    /// Enable memory debugging.
    pub enable_memory_debug: bool,
}

impl Default for RuntimeImports {
    fn default() -> Self {
        Self {
            module_name: "bhc".to_string(),
            enable_console: true,
            enable_timing: false,
            enable_memory_debug: false,
        }
    }
}

impl RuntimeImports {
    /// Generate import declarations for the runtime.
    #[must_use]
    pub fn generate_imports(&self) -> Vec<RuntimeImport> {
        let mut imports = Vec::new();

        if self.enable_console {
            imports.push(RuntimeImport {
                module: self.module_name.clone(),
                name: "log_i32".to_string(),
                signature: ImportSignature::Func {
                    params: vec![ImportType::I32],
                    results: vec![],
                },
            });
            imports.push(RuntimeImport {
                module: self.module_name.clone(),
                name: "log_f64".to_string(),
                signature: ImportSignature::Func {
                    params: vec![ImportType::F64],
                    results: vec![],
                },
            });
        }

        if self.enable_timing {
            imports.push(RuntimeImport {
                module: self.module_name.clone(),
                name: "now".to_string(),
                signature: ImportSignature::Func {
                    params: vec![],
                    results: vec![ImportType::F64],
                },
            });
        }

        if self.enable_memory_debug {
            imports.push(RuntimeImport {
                module: self.module_name.clone(),
                name: "debug_memory".to_string(),
                signature: ImportSignature::Func {
                    params: vec![ImportType::I32, ImportType::I32],
                    results: vec![],
                },
            });
        }

        imports
    }
}

/// A runtime import declaration.
#[derive(Clone, Debug)]
pub struct RuntimeImport {
    /// Module name.
    pub module: String,
    /// Import name.
    pub name: String,
    /// Import signature.
    pub signature: ImportSignature,
}

/// Import signature types.
#[derive(Clone, Debug)]
pub enum ImportSignature {
    /// Function import.
    Func {
        /// Parameter types.
        params: Vec<ImportType>,
        /// Result types.
        results: Vec<ImportType>,
    },
    /// Memory import.
    Memory {
        /// Minimum pages.
        min: u32,
        /// Maximum pages.
        max: Option<u32>,
    },
    /// Global import.
    Global {
        /// Value type.
        ty: ImportType,
        /// Whether mutable.
        mutable: bool,
    },
}

/// Types for imports.
#[derive(Clone, Copy, Debug)]
pub enum ImportType {
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
}

/// Standard exports that BHC modules provide.
#[derive(Clone, Debug)]
pub struct RuntimeExports {
    /// Export memory.
    pub memory: bool,
    /// Export _start entry point.
    pub start: bool,
    /// Export alloc function.
    pub alloc: bool,
    /// Export free function.
    pub free: bool,
    /// Export arena_reset function.
    pub arena_reset: bool,
}

impl Default for RuntimeExports {
    fn default() -> Self {
        Self {
            memory: true,
            start: true,
            alloc: true,
            free: true,
            arena_reset: true,
        }
    }
}

/// Generate the _start function body for WASM modules.
pub fn generate_start_function() -> Vec<WasmInstr> {
    vec![
        WasmInstr::Comment("Initialize runtime".to_string()),
        // Initialize arena pointer
        WasmInstr::I32Const(DEFAULT_HEAP_START as i32),
        WasmInstr::GlobalSet(0), // arena_ptr global
        WasmInstr::End,
    ]
}

/// Generate the alloc function body.
///
/// Simple bump allocator:
/// ```wat
/// (func $alloc (param $size i32) (result i32)
///   (local $ptr i32)
///   global.get $heap_ptr
///   local.set $ptr
///   global.get $heap_ptr
///   local.get $size
///   i32.add
///   global.set $heap_ptr
///   local.get $ptr
/// )
/// ```
pub fn generate_alloc_function() -> Vec<WasmInstr> {
    vec![
        WasmInstr::Comment("Bump allocator".to_string()),
        // Get current heap pointer
        WasmInstr::GlobalGet(0),
        WasmInstr::LocalTee(1), // Save as result
        // Bump heap pointer
        WasmInstr::LocalGet(0), // size parameter
        WasmInstr::I32Add,
        WasmInstr::GlobalSet(0),
        // Return old pointer
        WasmInstr::LocalGet(1),
        WasmInstr::End,
    ]
}

/// Generate the free function body.
///
/// Simple allocator doesn't support individual frees.
pub fn generate_free_function() -> Vec<WasmInstr> {
    vec![
        WasmInstr::Comment("Free (no-op for bump allocator)".to_string()),
        WasmInstr::End,
    ]
}

/// Generate the arena_reset function body.
///
/// Resets the arena to its initial state.
pub fn generate_arena_reset_function(heap_start: u32) -> Vec<WasmInstr> {
    vec![
        WasmInstr::Comment("Reset arena to initial state".to_string()),
        WasmInstr::I32Const(heap_start as i32),
        WasmInstr::GlobalSet(0),
        WasmInstr::End,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{WasmFunc, WasmFuncType, WasmModule};
    use crate::wasi;
    use crate::WasmConfig;

    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.initial_pages, 32);
        assert!(config.enable_arena);
    }

    #[test]
    fn test_runtime_config_edge() {
        let config = RuntimeConfig::edge();
        assert_eq!(config.initial_pages, 8);  // 512KB for stack + arena
        assert_eq!(config.stack_size, 32768);
    }

    #[test]
    fn test_runtime_config_validation() {
        let valid = RuntimeConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = RuntimeConfig {
            initial_pages: 0,
            ..RuntimeConfig::default()
        };
        assert!(invalid.validate().is_err());

        let too_small = RuntimeConfig {
            initial_pages: 1, // Only 64KB
            stack_size: 128 * 1024, // Needs 128KB
            ..RuntimeConfig::default()
        };
        assert!(too_small.validate().is_err());
    }

    #[test]
    fn test_runtime_imports() {
        let imports = RuntimeImports::default();
        let declarations = imports.generate_imports();

        assert!(!declarations.is_empty());
        assert!(declarations.iter().any(|i| i.name == "log_i32"));
    }

    #[test]
    fn test_generate_alloc() {
        let instrs = generate_alloc_function();
        assert!(!instrs.is_empty());
        assert!(matches!(instrs.last(), Some(WasmInstr::End)));
    }

    /// Verify that the complete WASM runtime stays under 100KB.
    ///
    /// This is a key requirement for the Edge profile where code size matters.
    /// The runtime includes:
    /// - Memory management (alloc, free, arena)
    /// - WASI interface functions (fd_write, proc_exit, etc.)
    /// - GC functions (mark, sweep, collect)
    /// - Entry point (_start)
    #[test]
    fn test_runtime_code_size_under_100kb() {
        use crate::runtime::gc::{self, GcConfig};

        let config = WasmConfig::edge_profile();
        let target = bhc_target::targets::wasm32_wasi();
        let mut module = WasmModule::new("bhc_runtime".to_string(), config, target);

        // Add heap pointer global
        let gc_config = GcConfig::default();
        module.add_global(gc::generate_heap_ptr_global(gc_config.heap_start));

        // Function indices for cross-references
        let mut func_idx = 0u32;

        // 1. Add basic allocator function
        let mut alloc_func = WasmFunc::new(WasmFuncType::new(
            vec![crate::WasmType::I32],  // size
            vec![crate::WasmType::I32],  // ptr
        ));
        alloc_func.name = Some("alloc".to_string());
        alloc_func.exported = true;
        for instr in generate_alloc_function() {
            alloc_func.emit(instr);
        }
        module.add_function(alloc_func);
        func_idx += 1;

        // 2. Add free function
        let mut free_func = WasmFunc::new(WasmFuncType::new(
            vec![crate::WasmType::I32],  // ptr
            vec![],
        ));
        free_func.name = Some("free".to_string());
        free_func.exported = true;
        for instr in generate_free_function() {
            free_func.emit(instr);
        }
        module.add_function(free_func);
        func_idx += 1;

        // 3. Add arena reset function
        let mut arena_reset_func = WasmFunc::new(WasmFuncType::new(
            vec![],
            vec![],
        ));
        arena_reset_func.name = Some("arena_reset".to_string());
        arena_reset_func.exported = true;
        for instr in generate_arena_reset_function(DEFAULT_HEAP_START) {
            arena_reset_func.emit(instr);
        }
        module.add_function(arena_reset_func);
        func_idx += 1;

        // 4. Add start function
        let mut start_func = WasmFunc::new(WasmFuncType::new(vec![], vec![]));
        start_func.name = Some("_start".to_string());
        start_func.exported = true;
        for instr in generate_start_function() {
            start_func.emit(instr);
        }
        module.add_function(start_func);
        func_idx += 1;

        // 5. Add GC init function
        let gc_init = gc::generate_gc_init(&gc_config);
        module.add_function(gc_init);
        func_idx += 1;

        // 6. Add GC alloc function
        let gc_alloc = gc::generate_gc_alloc(
            gc_config.heap_start,
            gc_config.heap_end,
            gc_config.gc_threshold,
            func_idx + 3, // gc_collect index (added later)
        );
        module.add_function(gc_alloc);
        func_idx += 1;

        // 7. Add GC mark function
        let gc_mark = gc::generate_gc_mark(gc_config.heap_start, gc_config.heap_end);
        module.add_function(gc_mark);
        func_idx += 1;

        // 8. Add GC sweep function
        let gc_sweep = gc::generate_gc_sweep(gc_config.heap_start, gc_config.heap_end);
        module.add_function(gc_sweep);
        func_idx += 1;

        // 9. Add GC collect function
        let gc_collect = gc::generate_gc_collect(
            gc_config.heap_start,
            gc_config.heap_end,
            func_idx - 2, // gc_mark index
            func_idx - 1, // gc_sweep index
        );
        module.add_function(gc_collect);
        func_idx += 1;

        // 10. Add GC stats function
        let gc_stats = gc::generate_gc_stats();
        module.add_function(gc_stats);
        func_idx += 1;

        // 11. Add GC root stack management
        let gc_root_push = gc::generate_gc_root_push(gc::ROOT_STACK_PTR_OFFSET);
        module.add_function(gc_root_push);
        func_idx += 1;

        let gc_root_pop = gc::generate_gc_root_pop(gc::ROOT_STACK_PTR_OFFSET);
        module.add_function(gc_root_pop);
        let _ = func_idx; // Last one, not needed further

        // 12. Add WASI helpers
        let print_i32 = wasi::generate_print_i32(0); // fd_write import index
        module.add_function(print_i32);

        let print_str = wasi::generate_print_str(0);
        module.add_function(print_str);

        let get_argc = wasi::generate_get_argc();
        module.add_function(get_argc);

        let get_argv = wasi::generate_get_argv();
        module.add_function(get_argv);

        let getenv = wasi::generate_getenv();
        module.add_function(getenv);

        // Generate binary
        let binary = module.to_wasm().expect("Failed to generate WASM binary");
        let size_kb = binary.len() as f64 / 1024.0;

        // Count functions we added (17 total)
        let function_count = 17;
        println!("Runtime binary size: {:.2} KB ({} bytes)", size_kb, binary.len());
        println!("Functions: {}", function_count);

        // Assert under 100KB
        const MAX_SIZE_KB: f64 = 100.0;
        assert!(
            size_kb < MAX_SIZE_KB,
            "Runtime size {:.2} KB exceeds limit of {} KB",
            size_kb,
            MAX_SIZE_KB
        );

        // Also verify it's a valid WASM module (has magic number)
        assert_eq!(&binary[0..4], b"\x00asm", "Invalid WASM magic number");
        assert_eq!(&binary[4..8], &[0x01, 0x00, 0x00, 0x00], "Invalid WASM version");
    }
}
