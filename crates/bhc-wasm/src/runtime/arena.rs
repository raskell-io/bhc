//! Arena allocator for WASM linear memory.
//!
//! Provides a bump allocator that operates on WASM linear memory,
//! suitable for short-lived allocations in numeric kernels.

use crate::{WasmInstr, WasmType};

/// Arena allocator configuration.
#[derive(Clone, Debug)]
pub struct ArenaConfig {
    /// Start address in linear memory.
    pub start_address: u32,
    /// Size in bytes.
    pub size: u32,
    /// Alignment for allocations.
    pub alignment: u32,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            start_address: super::DEFAULT_HEAP_START,
            size: 1024 * 1024, // 1MB
            alignment: 16,     // 16-byte alignment for SIMD
        }
    }
}

impl ArenaConfig {
    /// Create an arena config for the Edge profile.
    #[must_use]
    pub fn edge() -> Self {
        Self {
            start_address: super::DEFAULT_HEAP_START,
            size: 256 * 1024, // 256KB
            alignment: 16,
        }
    }

    /// Get the end address of the arena.
    #[must_use]
    pub fn end_address(&self) -> u32 {
        self.start_address + self.size
    }
}

/// WASM arena allocator code generator.
///
/// Generates WASM instructions for arena-based allocation.
pub struct WasmArena {
    /// Arena configuration.
    config: ArenaConfig,
    /// Global variable index for arena pointer.
    ptr_global: u32,
    /// Global variable index for arena end.
    end_global: u32,
}

impl WasmArena {
    /// Create a new arena generator.
    #[must_use]
    pub fn new(config: ArenaConfig, ptr_global: u32, end_global: u32) -> Self {
        Self {
            config,
            ptr_global,
            end_global,
        }
    }

    /// Get the global variable index for the arena pointer.
    #[must_use]
    pub fn ptr_global(&self) -> u32 {
        self.ptr_global
    }

    /// Get the global variable index for the arena end.
    #[must_use]
    pub fn end_global(&self) -> u32 {
        self.end_global
    }

    /// Generate initialization code for the arena globals.
    #[must_use]
    pub fn generate_init(&self) -> Vec<GlobalInit> {
        vec![
            GlobalInit {
                name: "arena_ptr".to_string(),
                ty: WasmType::I32,
                mutable: true,
                init: WasmInstr::I32Const(self.config.start_address as i32),
            },
            GlobalInit {
                name: "arena_end".to_string(),
                ty: WasmType::I32,
                mutable: false,
                init: WasmInstr::I32Const(self.config.end_address() as i32),
            },
        ]
    }

    /// Generate arena allocation code.
    ///
    /// Allocates `size` bytes from the arena with proper alignment.
    /// Returns the allocated address or 0 if out of memory.
    ///
    /// ```wat
    /// ;; Align the current pointer
    /// global.get $arena_ptr
    /// local.get $alignment_mask
    /// i32.add
    /// local.get $neg_alignment_mask
    /// i32.and
    /// local.tee $aligned_ptr
    ///
    /// ;; Calculate new pointer
    /// local.get $size
    /// i32.add
    /// local.tee $new_ptr
    ///
    /// ;; Check bounds
    /// global.get $arena_end
    /// i32.gt_u
    /// if (result i32)
    ///   i32.const 0  ;; Out of memory
    /// else
    ///   local.get $new_ptr
    ///   global.set $arena_ptr
    ///   local.get $aligned_ptr
    /// end
    /// ```
    #[must_use]
    pub fn generate_alloc(&self) -> Vec<WasmInstr> {
        let alignment = self.config.alignment;
        let alignment_mask = alignment - 1;

        vec![
            WasmInstr::Comment(format!("Arena alloc (align={})", alignment)),
            // Align the current pointer
            WasmInstr::GlobalGet(self.ptr_global),
            WasmInstr::I32Const(alignment_mask as i32),
            WasmInstr::I32Add,
            WasmInstr::I32Const(-(alignment as i32)),
            WasmInstr::I32And,
            WasmInstr::LocalTee(1), // aligned_ptr
            // Add size to get new pointer
            WasmInstr::LocalGet(0), // size parameter
            WasmInstr::I32Add,
            WasmInstr::LocalTee(2), // new_ptr
            // Check if we exceeded arena bounds
            WasmInstr::GlobalGet(self.end_global),
            WasmInstr::I32GtU,
            // If out of bounds, return 0
            WasmInstr::If(Some(WasmType::I32)),
            WasmInstr::I32Const(0),
            WasmInstr::Else,
            // Update arena pointer and return aligned address
            WasmInstr::LocalGet(2), // new_ptr
            WasmInstr::GlobalSet(self.ptr_global),
            WasmInstr::LocalGet(1), // aligned_ptr
            WasmInstr::End,
        ]
    }

    /// Generate arena reset code.
    ///
    /// Resets the arena pointer to the start, effectively freeing all
    /// allocations at once.
    #[must_use]
    pub fn generate_reset(&self) -> Vec<WasmInstr> {
        vec![
            WasmInstr::Comment("Arena reset".to_string()),
            WasmInstr::I32Const(self.config.start_address as i32),
            WasmInstr::GlobalSet(self.ptr_global),
            WasmInstr::End,
        ]
    }

    /// Generate code to get the current arena usage in bytes.
    #[must_use]
    pub fn generate_usage(&self) -> Vec<WasmInstr> {
        vec![
            WasmInstr::Comment("Arena usage".to_string()),
            WasmInstr::GlobalGet(self.ptr_global),
            WasmInstr::I32Const(self.config.start_address as i32),
            WasmInstr::I32Sub,
        ]
    }

    /// Generate code to get the remaining arena space in bytes.
    #[must_use]
    pub fn generate_remaining(&self) -> Vec<WasmInstr> {
        vec![
            WasmInstr::Comment("Arena remaining".to_string()),
            WasmInstr::GlobalGet(self.end_global),
            WasmInstr::GlobalGet(self.ptr_global),
            WasmInstr::I32Sub,
        ]
    }

    /// Generate a scoped arena allocation.
    ///
    /// Saves the current arena pointer, executes the body, then
    /// restores the pointer (freeing all allocations made in the scope).
    #[must_use]
    pub fn generate_scoped(&self, body: Vec<WasmInstr>) -> Vec<WasmInstr> {
        let mut instrs = vec![
            WasmInstr::Comment("Arena scope begin".to_string()),
            // Save current pointer
            WasmInstr::GlobalGet(self.ptr_global),
            WasmInstr::LocalTee(0), // saved_ptr
        ];

        instrs.extend(body);

        instrs.extend(vec![
            WasmInstr::Comment("Arena scope end".to_string()),
            // Restore pointer
            WasmInstr::LocalGet(0), // saved_ptr
            WasmInstr::GlobalSet(self.ptr_global),
        ]);

        instrs
    }
}

/// Global variable initialization data.
#[derive(Clone, Debug)]
pub struct GlobalInit {
    /// Variable name.
    pub name: String,
    /// Variable type.
    pub ty: WasmType,
    /// Whether the variable is mutable.
    pub mutable: bool,
    /// Initialization instruction.
    pub init: WasmInstr,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_config_default() {
        let config = ArenaConfig::default();
        assert_eq!(config.alignment, 16);
        assert_eq!(config.size, 1024 * 1024);
    }

    #[test]
    fn test_arena_config_edge() {
        let config = ArenaConfig::edge();
        assert_eq!(config.size, 256 * 1024);
    }

    #[test]
    fn test_arena_end_address() {
        let config = ArenaConfig {
            start_address: 1000,
            size: 500,
            alignment: 8,
        };
        assert_eq!(config.end_address(), 1500);
    }

    #[test]
    fn test_arena_generate_init() {
        let config = ArenaConfig::default();
        let arena = WasmArena::new(config, 0, 1);
        let globals = arena.generate_init();

        assert_eq!(globals.len(), 2);
        assert_eq!(globals[0].name, "arena_ptr");
        assert!(globals[0].mutable);
        assert_eq!(globals[1].name, "arena_end");
        assert!(!globals[1].mutable);
    }

    #[test]
    fn test_arena_generate_alloc() {
        let config = ArenaConfig::default();
        let arena = WasmArena::new(config, 0, 1);
        let instrs = arena.generate_alloc();

        assert!(!instrs.is_empty());
        // Should contain bounds check
        assert!(instrs.iter().any(|i| matches!(i, WasmInstr::If(_))));
    }

    #[test]
    fn test_arena_generate_reset() {
        let config = ArenaConfig::default();
        let arena = WasmArena::new(config, 0, 1);
        let instrs = arena.generate_reset();

        assert!(!instrs.is_empty());
        assert!(matches!(instrs.last(), Some(WasmInstr::End)));
    }
}
