//! Garbage collection for WASM linear memory.
//!
//! This module implements a simple mark-sweep garbage collector that operates
//! entirely within WASM linear memory. It's designed for the Edge profile where
//! we need memory management but can't use the native RTS GC.
//!
//! # Memory Layout
//!
//! Objects in the GC heap have a header followed by payload:
//!
//! ```text
//! ┌────────────────────┬────────────────────┬─────────────────────┐
//! │    Header (8B)     │   Type Info (4B)   │     Payload...      │
//! ├────────────────────┼────────────────────┼─────────────────────┤
//! │ mark | size | tag  │   type_id/ptrs    │   actual data       │
//! └────────────────────┴────────────────────┴─────────────────────┘
//! ```
//!
//! Header format (8 bytes):
//! - Byte 0: Mark bit and flags
//! - Bytes 1-3: Reserved
//! - Bytes 4-7: Object size (including header)
//!
//! Type info (4 bytes):
//! - For boxed types: pointer count for traversal
//! - For arrays: element count

use crate::{WasmInstr, WasmType};
use crate::codegen::{WasmFunc, WasmFuncType, WasmGlobal};

/// Object header size in bytes.
pub const HEADER_SIZE: u32 = 12;

/// Mark bit position in header flags.
pub const MARK_BIT: u8 = 0x01;

/// Forwarding pointer bit (for compacting GC, future use).
pub const FORWARD_BIT: u8 = 0x02;

/// Memory offset for GC globals.
pub const GC_GLOBALS_OFFSET: u32 = 128;

/// Offset of heap_ptr global in memory.
pub const HEAP_PTR_OFFSET: u32 = GC_GLOBALS_OFFSET;

/// Offset of heap_end global in memory.
pub const HEAP_END_OFFSET: u32 = GC_GLOBALS_OFFSET + 4;

/// Offset of root_stack_ptr in memory.
pub const ROOT_STACK_PTR_OFFSET: u32 = GC_GLOBALS_OFFSET + 8;

/// Offset of root_stack_base in memory.
pub const ROOT_STACK_BASE_OFFSET: u32 = GC_GLOBALS_OFFSET + 12;

/// Offset of free_list head in memory.
pub const FREE_LIST_OFFSET: u32 = GC_GLOBALS_OFFSET + 16;

/// Offset of bytes_allocated counter.
pub const BYTES_ALLOCATED_OFFSET: u32 = GC_GLOBALS_OFFSET + 20;

/// Offset of collection_count counter.
pub const COLLECTION_COUNT_OFFSET: u32 = GC_GLOBALS_OFFSET + 24;

/// Size of GC root stack (number of pointers).
pub const ROOT_STACK_SIZE: u32 = 1024;

/// GC configuration for WASM.
#[derive(Clone, Debug)]
pub struct GcConfig {
    /// Heap start address.
    pub heap_start: u32,
    /// Heap end address.
    pub heap_end: u32,
    /// Root stack start address.
    pub root_stack_start: u32,
    /// Allocation threshold before triggering GC (bytes).
    pub gc_threshold: u32,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            heap_start: 65536,        // After stack (64KB)
            heap_end: 65536 * 8,      // 512KB heap
            root_stack_start: 256,    // Root stack at low memory
            gc_threshold: 65536 * 4,  // GC when 256KB allocated
        }
    }
}

/// Generate the heap_ptr global variable.
pub fn generate_heap_ptr_global(heap_start: u32) -> WasmGlobal {
    WasmGlobal {
        name: Some("__gc_heap_ptr".to_string()),
        ty: WasmType::I32,
        mutable: true,
        init: WasmInstr::I32Const(heap_start as i32),
    }
}

/// Generate the allocation function with GC integration.
///
/// `gc_alloc(size: i32) -> i32`
///
/// Allocates `size` bytes from the GC heap. If allocation would exceed
/// the threshold, triggers garbage collection first.
pub fn generate_gc_alloc(
    gc_collect_idx: u32,
    heap_ptr_global: u32,
    heap_end_global: u32,
) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![WasmType::I32], vec![WasmType::I32]));
    func.name = Some("gc_alloc".to_string());
    func.exported = true;

    let size_local = 0; // Parameter
    let aligned_size = func.add_local(WasmType::I32);
    let result_ptr = func.add_local(WasmType::I32);
    let new_heap_ptr = func.add_local(WasmType::I32);

    // Align size to 8 bytes and add header
    // aligned_size = ((size + HEADER_SIZE + 7) & ~7)
    func.emit(WasmInstr::LocalGet(size_local));
    func.emit(WasmInstr::I32Const(HEADER_SIZE as i32 + 7));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Const(-8)); // ~7
    func.emit(WasmInstr::I32And);
    func.emit(WasmInstr::LocalSet(aligned_size));

    // Check if we need to collect
    // if (heap_ptr + aligned_size > heap_end) { gc_collect(); }
    func.emit(WasmInstr::GlobalGet(heap_ptr_global));
    func.emit(WasmInstr::LocalGet(aligned_size));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::GlobalGet(heap_end_global));
    func.emit(WasmInstr::I32GtU);
    func.emit(WasmInstr::If(None));
    func.emit(WasmInstr::Call(gc_collect_idx));
    func.emit(WasmInstr::End);

    // Get current heap pointer as result
    func.emit(WasmInstr::GlobalGet(heap_ptr_global));
    func.emit(WasmInstr::LocalSet(result_ptr));

    // Calculate new heap pointer
    func.emit(WasmInstr::LocalGet(result_ptr));
    func.emit(WasmInstr::LocalGet(aligned_size));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(new_heap_ptr));

    // Check if allocation would overflow
    func.emit(WasmInstr::LocalGet(new_heap_ptr));
    func.emit(WasmInstr::GlobalGet(heap_end_global));
    func.emit(WasmInstr::I32GtU);
    func.emit(WasmInstr::If(None));
    // Out of memory - return 0
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::Return);
    func.emit(WasmInstr::End);

    // Update heap pointer
    func.emit(WasmInstr::LocalGet(new_heap_ptr));
    func.emit(WasmInstr::GlobalSet(heap_ptr_global));

    // Initialize header: clear mark bit, set size
    // header[0] = 0 (flags)
    func.emit(WasmInstr::LocalGet(result_ptr));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32Store8(0, 0));

    // header[4..8] = aligned_size
    func.emit(WasmInstr::LocalGet(result_ptr));
    func.emit(WasmInstr::I32Const(4));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalGet(aligned_size));
    func.emit(WasmInstr::I32Store(4, 0));

    // Return pointer to payload (after header)
    func.emit(WasmInstr::LocalGet(result_ptr));
    func.emit(WasmInstr::I32Const(HEADER_SIZE as i32));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::End);

    func
}

/// Generate the GC root push function.
///
/// `gc_root_push(ptr: i32)`
///
/// Pushes a pointer onto the GC root stack.
pub fn generate_gc_root_push(root_stack_ptr_offset: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![WasmType::I32], vec![]));
    func.name = Some("gc_root_push".to_string());
    func.exported = true;

    let ptr_local = 0; // Parameter
    let stack_ptr = func.add_local(WasmType::I32);

    // Load current root stack pointer
    func.emit(WasmInstr::I32Const(root_stack_ptr_offset as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::LocalSet(stack_ptr));

    // Store the pointer at stack_ptr
    func.emit(WasmInstr::LocalGet(stack_ptr));
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Store(4, 0));

    // Increment stack pointer
    func.emit(WasmInstr::I32Const(root_stack_ptr_offset as i32));
    func.emit(WasmInstr::LocalGet(stack_ptr));
    func.emit(WasmInstr::I32Const(4));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Store(4, 0));

    func.emit(WasmInstr::End);
    func
}

/// Generate the GC root pop function.
///
/// `gc_root_pop() -> i32`
///
/// Pops a pointer from the GC root stack.
pub fn generate_gc_root_pop(root_stack_ptr_offset: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![WasmType::I32]));
    func.name = Some("gc_root_pop".to_string());
    func.exported = true;

    let stack_ptr = func.add_local(WasmType::I32);

    // Decrement stack pointer
    func.emit(WasmInstr::I32Const(root_stack_ptr_offset as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::I32Const(4));
    func.emit(WasmInstr::I32Sub);
    func.emit(WasmInstr::LocalTee(stack_ptr));

    // Store decremented pointer back
    func.emit(WasmInstr::I32Const(root_stack_ptr_offset as i32));
    func.emit(WasmInstr::LocalGet(stack_ptr));
    func.emit(WasmInstr::I32Store(4, 0));

    // Load and return the value
    func.emit(WasmInstr::LocalGet(stack_ptr));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::End);

    func
}

/// Generate the mark function for a single object.
///
/// `gc_mark(ptr: i32)`
///
/// Sets the mark bit on an object. Does nothing if ptr is 0 or already marked.
pub fn generate_gc_mark(heap_start: u32, heap_end: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![WasmType::I32], vec![]));
    func.name = Some("gc_mark".to_string());

    let ptr_local = 0; // Parameter
    let header_ptr = func.add_local(WasmType::I32);
    let flags = func.add_local(WasmType::I32);

    // Check for null pointer
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Eqz);
    func.emit(WasmInstr::If(None));
    func.emit(WasmInstr::Return);
    func.emit(WasmInstr::End);

    // Calculate header pointer (ptr - HEADER_SIZE)
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Const(HEADER_SIZE as i32));
    func.emit(WasmInstr::I32Sub);
    func.emit(WasmInstr::LocalSet(header_ptr));

    // Bounds check: is this a valid heap pointer?
    func.emit(WasmInstr::LocalGet(header_ptr));
    func.emit(WasmInstr::I32Const(heap_start as i32));
    func.emit(WasmInstr::I32LtU);
    func.emit(WasmInstr::If(None));
    func.emit(WasmInstr::Return); // Not a heap pointer
    func.emit(WasmInstr::End);

    func.emit(WasmInstr::LocalGet(header_ptr));
    func.emit(WasmInstr::I32Const(heap_end as i32));
    func.emit(WasmInstr::I32GeU);
    func.emit(WasmInstr::If(None));
    func.emit(WasmInstr::Return); // Not a heap pointer
    func.emit(WasmInstr::End);

    // Load flags byte
    func.emit(WasmInstr::LocalGet(header_ptr));
    func.emit(WasmInstr::I32Load8U(0, 0));
    func.emit(WasmInstr::LocalSet(flags));

    // Check if already marked
    func.emit(WasmInstr::LocalGet(flags));
    func.emit(WasmInstr::I32Const(MARK_BIT as i32));
    func.emit(WasmInstr::I32And);
    func.emit(WasmInstr::If(None));
    func.emit(WasmInstr::Return); // Already marked
    func.emit(WasmInstr::End);

    // Set mark bit
    func.emit(WasmInstr::LocalGet(header_ptr));
    func.emit(WasmInstr::LocalGet(flags));
    func.emit(WasmInstr::I32Const(MARK_BIT as i32));
    func.emit(WasmInstr::I32Or);
    func.emit(WasmInstr::I32Store8(0, 0));

    func.emit(WasmInstr::End);
    func
}

/// Generate the sweep function.
///
/// `gc_sweep()`
///
/// Walks the heap and reclaims unmarked objects, clearing marks on live objects.
/// Uses a simple free list approach.
pub fn generate_gc_sweep(
    heap_start: u32,
    heap_ptr_global: u32,
    free_list_offset: u32,
) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![]));
    func.name = Some("gc_sweep".to_string());

    let current = func.add_local(WasmType::I32);
    let heap_end = func.add_local(WasmType::I32);
    let obj_size = func.add_local(WasmType::I32);
    let flags = func.add_local(WasmType::I32);
    let free_head = func.add_local(WasmType::I32);

    // Get current heap end
    func.emit(WasmInstr::GlobalGet(heap_ptr_global));
    func.emit(WasmInstr::LocalSet(heap_end));

    // Initialize free list head to 0
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::LocalSet(free_head));

    // Start at heap_start
    func.emit(WasmInstr::I32Const(heap_start as i32));
    func.emit(WasmInstr::LocalSet(current));

    // Main sweep loop
    func.emit(WasmInstr::Block(None)); // break target
    func.emit(WasmInstr::Loop(None));  // continue target

    // while (current < heap_end)
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::LocalGet(heap_end));
    func.emit(WasmInstr::I32GeU);
    func.emit(WasmInstr::BrIf(1)); // break if current >= heap_end

    // Load object size from header
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::I32Const(4));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::LocalSet(obj_size));

    // Load flags
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::I32Load8U(0, 0));
    func.emit(WasmInstr::LocalSet(flags));

    // Check mark bit
    func.emit(WasmInstr::LocalGet(flags));
    func.emit(WasmInstr::I32Const(MARK_BIT as i32));
    func.emit(WasmInstr::I32And);
    func.emit(WasmInstr::If(None));

    // Object is marked - clear mark bit for next cycle
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::LocalGet(flags));
    func.emit(WasmInstr::I32Const(!(MARK_BIT as i32)));
    func.emit(WasmInstr::I32And);
    func.emit(WasmInstr::I32Store8(0, 0));

    func.emit(WasmInstr::Else);

    // Object is unmarked - add to free list
    // Store current free_head at this object's location (for free list linking)
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::I32Const(8)); // After flags and size
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalGet(free_head));
    func.emit(WasmInstr::I32Store(4, 0));

    // Update free_head to this object
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::LocalSet(free_head));

    func.emit(WasmInstr::End); // end if

    // Advance to next object
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::LocalGet(obj_size));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(current));

    func.emit(WasmInstr::Br(0)); // continue loop
    func.emit(WasmInstr::End); // end loop
    func.emit(WasmInstr::End); // end block

    // Store free list head
    func.emit(WasmInstr::I32Const(free_list_offset as i32));
    func.emit(WasmInstr::LocalGet(free_head));
    func.emit(WasmInstr::I32Store(4, 0));

    func.emit(WasmInstr::End);
    func
}

/// Generate the main garbage collection function.
///
/// `gc_collect()`
///
/// Performs a full mark-sweep collection:
/// 1. Mark phase: traverse from roots
/// 2. Sweep phase: reclaim unmarked objects
pub fn generate_gc_collect(
    gc_mark_idx: u32,
    gc_sweep_idx: u32,
    root_stack_base_offset: u32,
    root_stack_ptr_offset: u32,
) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![]));
    func.name = Some("gc_collect".to_string());
    func.exported = true;

    let current = func.add_local(WasmType::I32);
    let stack_end = func.add_local(WasmType::I32);
    let root_ptr = func.add_local(WasmType::I32);

    // === Mark Phase ===
    // Load root stack bounds
    func.emit(WasmInstr::I32Const(root_stack_base_offset as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::LocalSet(current));

    func.emit(WasmInstr::I32Const(root_stack_ptr_offset as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::LocalSet(stack_end));

    // Iterate through root stack
    func.emit(WasmInstr::Block(None));
    func.emit(WasmInstr::Loop(None));

    // while (current < stack_end)
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::LocalGet(stack_end));
    func.emit(WasmInstr::I32GeU);
    func.emit(WasmInstr::BrIf(1));

    // Load root pointer
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::LocalSet(root_ptr));

    // Mark the root
    func.emit(WasmInstr::LocalGet(root_ptr));
    func.emit(WasmInstr::Call(gc_mark_idx));

    // Advance to next root
    func.emit(WasmInstr::LocalGet(current));
    func.emit(WasmInstr::I32Const(4));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(current));

    func.emit(WasmInstr::Br(0));
    func.emit(WasmInstr::End);
    func.emit(WasmInstr::End);

    // === Sweep Phase ===
    func.emit(WasmInstr::Call(gc_sweep_idx));

    // Increment collection count
    func.emit(WasmInstr::I32Const(COLLECTION_COUNT_OFFSET as i32));
    func.emit(WasmInstr::I32Const(COLLECTION_COUNT_OFFSET as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Store(4, 0));

    func.emit(WasmInstr::End);
    func
}

/// Generate the GC initialization function.
///
/// `gc_init()`
///
/// Initializes the GC state in memory.
pub fn generate_gc_init(config: &GcConfig) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![]));
    func.name = Some("gc_init".to_string());
    func.exported = true;

    // Initialize heap_ptr
    func.emit(WasmInstr::I32Const(HEAP_PTR_OFFSET as i32));
    func.emit(WasmInstr::I32Const(config.heap_start as i32));
    func.emit(WasmInstr::I32Store(4, 0));

    // Initialize heap_end
    func.emit(WasmInstr::I32Const(HEAP_END_OFFSET as i32));
    func.emit(WasmInstr::I32Const(config.heap_end as i32));
    func.emit(WasmInstr::I32Store(4, 0));

    // Initialize root stack pointer
    func.emit(WasmInstr::I32Const(ROOT_STACK_PTR_OFFSET as i32));
    func.emit(WasmInstr::I32Const(config.root_stack_start as i32));
    func.emit(WasmInstr::I32Store(4, 0));

    // Initialize root stack base
    func.emit(WasmInstr::I32Const(ROOT_STACK_BASE_OFFSET as i32));
    func.emit(WasmInstr::I32Const(config.root_stack_start as i32));
    func.emit(WasmInstr::I32Store(4, 0));

    // Initialize free list to null
    func.emit(WasmInstr::I32Const(FREE_LIST_OFFSET as i32));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32Store(4, 0));

    // Initialize counters
    func.emit(WasmInstr::I32Const(BYTES_ALLOCATED_OFFSET as i32));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32Store(4, 0));

    func.emit(WasmInstr::I32Const(COLLECTION_COUNT_OFFSET as i32));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32Store(4, 0));

    func.emit(WasmInstr::End);
    func
}

/// Generate a function to get GC statistics.
///
/// `gc_stats() -> i64`
///
/// Returns packed stats: (collection_count << 32) | bytes_allocated
pub fn generate_gc_stats() -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![WasmType::I64]));
    func.name = Some("gc_stats".to_string());
    func.exported = true;

    // Load collection_count as i64
    func.emit(WasmInstr::I32Const(COLLECTION_COUNT_OFFSET as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::I64ExtendI32U);

    // Shift left 32
    func.emit(WasmInstr::I64Const(32));
    func.emit(WasmInstr::I64Shl);

    // Load bytes_allocated as i64 and OR
    func.emit(WasmInstr::I32Const(BYTES_ALLOCATED_OFFSET as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::I64ExtendI32U);
    func.emit(WasmInstr::I64Or);

    func.emit(WasmInstr::End);
    func
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_config_default() {
        let config = GcConfig::default();
        assert!(config.heap_end > config.heap_start);
        assert!(config.gc_threshold > 0);
    }

    #[test]
    fn test_generate_gc_init() {
        let config = GcConfig::default();
        let func = generate_gc_init(&config);
        assert_eq!(func.name.as_deref(), Some("gc_init"));
        assert!(func.exported);
    }

    #[test]
    fn test_generate_gc_mark() {
        let func = generate_gc_mark(65536, 65536 * 8);
        assert_eq!(func.name.as_deref(), Some("gc_mark"));
    }

    #[test]
    fn test_generate_gc_collect() {
        let func = generate_gc_collect(0, 1, ROOT_STACK_BASE_OFFSET, ROOT_STACK_PTR_OFFSET);
        assert_eq!(func.name.as_deref(), Some("gc_collect"));
        assert!(func.exported);
    }

    #[test]
    fn test_generate_gc_alloc() {
        let func = generate_gc_alloc(0, 0, 1);
        assert_eq!(func.name.as_deref(), Some("gc_alloc"));
        assert!(func.exported);
    }

    #[test]
    fn test_generate_gc_stats() {
        let func = generate_gc_stats();
        assert_eq!(func.name.as_deref(), Some("gc_stats"));
        assert!(func.exported);
        // Should return i64
        assert_eq!(func.func_type.results.len(), 1);
        assert_eq!(func.func_type.results[0], WasmType::I64);
    }
}
