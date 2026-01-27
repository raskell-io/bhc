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
}
