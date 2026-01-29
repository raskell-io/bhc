//! Linear memory management for WASM.
//!
//! Provides utilities for managing WASM linear memory layout,
//! including stack, heap, and data segment placement.

use crate::{WasmError, WasmResult};

/// WASM linear memory layout.
///
/// The default layout is:
/// ```text
/// ┌────────────────────┐  0x00000000
/// │     Data Segments  │
/// │     (constants)    │
/// ├────────────────────┤  data_end
/// │                    │
/// │       Stack        │
/// │         ↓          │
/// │                    │
/// ├────────────────────┤  stack_end = heap_start
/// │                    │
/// │    Arena/Heap      │
/// │         ↑          │
/// │                    │
/// ├────────────────────┤  heap_end
/// │                    │
/// │   (unused space)   │
/// │                    │
/// └────────────────────┘  memory_end
/// ```
#[derive(Clone, Debug)]
pub struct MemoryLayout {
    /// Start of data segments.
    pub data_start: u32,
    /// End of data segments (start of stack).
    pub data_end: u32,
    /// Start of stack (grows down from here).
    pub stack_start: u32,
    /// End of stack (stack base).
    pub stack_end: u32,
    /// Start of heap/arena.
    pub heap_start: u32,
    /// End of heap/arena.
    pub heap_end: u32,
    /// Total memory size.
    pub total_size: u32,
}

impl MemoryLayout {
    /// Create a new memory layout builder.
    #[must_use]
    pub fn builder() -> MemoryLayoutBuilder {
        MemoryLayoutBuilder::new()
    }

    /// Create a default layout for the given memory size.
    #[must_use]
    pub fn default_for_size(total_pages: u32) -> Self {
        let total_size = total_pages * super::PAGE_SIZE;
        let stack_size = 64 * 1024; // 64KB stack
        let data_size = 4 * 1024; // 4KB for data segments

        Self {
            data_start: 0,
            data_end: data_size,
            stack_start: data_size,
            stack_end: data_size + stack_size,
            heap_start: data_size + stack_size,
            heap_end: total_size,
            total_size,
        }
    }

    /// Get the size of the stack.
    #[must_use]
    pub fn stack_size(&self) -> u32 {
        self.stack_end - self.stack_start
    }

    /// Get the size of the heap/arena.
    #[must_use]
    pub fn heap_size(&self) -> u32 {
        self.heap_end - self.heap_start
    }

    /// Get the size of the data segment region.
    #[must_use]
    pub fn data_size(&self) -> u32 {
        self.data_end - self.data_start
    }

    /// Validate the layout.
    pub fn validate(&self) -> WasmResult<()> {
        if self.stack_end < self.stack_start {
            return Err(WasmError::MemoryError("Invalid stack region".to_string()));
        }

        if self.heap_end < self.heap_start {
            return Err(WasmError::MemoryError("Invalid heap region".to_string()));
        }

        if self.stack_end > self.heap_start {
            return Err(WasmError::MemoryError(
                "Stack overlaps with heap".to_string(),
            ));
        }

        if self.heap_end > self.total_size {
            return Err(WasmError::MemoryError(
                "Heap extends beyond memory".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for memory layouts.
pub struct MemoryLayoutBuilder {
    data_size: u32,
    stack_size: u32,
    heap_size: Option<u32>,
    total_pages: u32,
}

impl MemoryLayoutBuilder {
    /// Create a new builder with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_size: 4096,
            stack_size: 65536,
            heap_size: None,
            total_pages: 16,
        }
    }

    /// Set the data segment size.
    #[must_use]
    pub fn data_size(mut self, size: u32) -> Self {
        self.data_size = size;
        self
    }

    /// Set the stack size.
    #[must_use]
    pub fn stack_size(mut self, size: u32) -> Self {
        self.stack_size = size;
        self
    }

    /// Set the heap size (if not set, uses remaining memory).
    #[must_use]
    pub fn heap_size(mut self, size: u32) -> Self {
        self.heap_size = Some(size);
        self
    }

    /// Set the total number of memory pages.
    #[must_use]
    pub fn total_pages(mut self, pages: u32) -> Self {
        self.total_pages = pages;
        self
    }

    /// Build the memory layout.
    pub fn build(self) -> WasmResult<MemoryLayout> {
        let total_size = self.total_pages * super::PAGE_SIZE;
        let data_start = 0;
        let data_end = self.data_size;
        let stack_start = data_end;
        let stack_end = stack_start + self.stack_size;
        let heap_start = stack_end;

        let heap_end = match self.heap_size {
            Some(size) => heap_start + size,
            None => total_size,
        };

        if heap_end > total_size {
            return Err(WasmError::MemoryError(format!(
                "Layout requires {} bytes but only {} available",
                heap_end, total_size
            )));
        }

        let layout = MemoryLayout {
            data_start,
            data_end,
            stack_start,
            stack_end,
            heap_start,
            heap_end,
            total_size,
        };

        layout.validate()?;
        Ok(layout)
    }
}

impl Default for MemoryLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear memory manager.
///
/// Manages allocation of addresses in WASM linear memory.
pub struct LinearMemory {
    /// Memory layout.
    layout: MemoryLayout,
    /// Current data segment offset.
    data_offset: u32,
    /// Data segments.
    data_segments: Vec<DataSegment>,
}

impl LinearMemory {
    /// Create a new linear memory manager.
    #[must_use]
    pub fn new(layout: MemoryLayout) -> Self {
        Self {
            data_offset: layout.data_start,
            layout,
            data_segments: Vec::new(),
        }
    }

    /// Get the memory layout.
    #[must_use]
    pub fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Allocate space in the data segment region.
    ///
    /// Returns the offset of the allocated data.
    pub fn alloc_data(&mut self, data: Vec<u8>, alignment: u32) -> WasmResult<u32> {
        // Align the offset
        let aligned = (self.data_offset + alignment - 1) & !(alignment - 1);

        // Check if we have space
        if aligned + data.len() as u32 > self.layout.data_end {
            return Err(WasmError::MemoryError("Data segment overflow".to_string()));
        }

        let offset = aligned;
        self.data_segments.push(DataSegment { offset, data });
        self.data_offset = offset + self.data_segments.last().unwrap().data.len() as u32;

        Ok(offset)
    }

    /// Allocate an i32 constant in the data segment.
    pub fn alloc_i32(&mut self, value: i32) -> WasmResult<u32> {
        self.alloc_data(value.to_le_bytes().to_vec(), 4)
    }

    /// Allocate an i64 constant in the data segment.
    pub fn alloc_i64(&mut self, value: i64) -> WasmResult<u32> {
        self.alloc_data(value.to_le_bytes().to_vec(), 8)
    }

    /// Allocate an f32 constant in the data segment.
    pub fn alloc_f32(&mut self, value: f32) -> WasmResult<u32> {
        self.alloc_data(value.to_le_bytes().to_vec(), 4)
    }

    /// Allocate an f64 constant in the data segment.
    pub fn alloc_f64(&mut self, value: f64) -> WasmResult<u32> {
        self.alloc_data(value.to_le_bytes().to_vec(), 8)
    }

    /// Allocate a v128 constant in the data segment.
    pub fn alloc_v128(&mut self, value: [u8; 16]) -> WasmResult<u32> {
        self.alloc_data(value.to_vec(), 16)
    }

    /// Allocate an array of f32 values.
    pub fn alloc_f32_array(&mut self, values: &[f32]) -> WasmResult<u32> {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.alloc_data(data, 16) // 16-byte alignment for SIMD
    }

    /// Allocate an array of f64 values.
    pub fn alloc_f64_array(&mut self, values: &[f64]) -> WasmResult<u32> {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.alloc_data(data, 16)
    }

    /// Get all data segments.
    #[must_use]
    pub fn data_segments(&self) -> &[DataSegment] {
        &self.data_segments
    }

    /// Get the remaining data segment space.
    #[must_use]
    pub fn remaining_data_space(&self) -> u32 {
        self.layout.data_end - self.data_offset
    }

    /// Get the stack pointer value (initial SP).
    #[must_use]
    pub fn initial_stack_pointer(&self) -> u32 {
        self.layout.stack_end
    }

    /// Get the heap start address.
    #[must_use]
    pub fn heap_start(&self) -> u32 {
        self.layout.heap_start
    }
}

/// A data segment to be placed in linear memory.
#[derive(Clone, Debug)]
pub struct DataSegment {
    /// Offset in linear memory.
    pub offset: u32,
    /// Data bytes.
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_layout_default() {
        let layout = MemoryLayout::default_for_size(16); // 16 pages = 1MB
        assert_eq!(layout.total_size, 16 * 65536);
        assert!(layout.validate().is_ok());
    }

    #[test]
    fn test_memory_layout_builder() {
        let layout = MemoryLayout::builder()
            .data_size(8192)
            .stack_size(32768)
            .total_pages(8)
            .build()
            .unwrap();

        assert_eq!(layout.data_size(), 8192);
        assert_eq!(layout.stack_size(), 32768);
    }

    #[test]
    fn test_memory_layout_overflow() {
        let result = MemoryLayout::builder()
            .data_size(100000)
            .stack_size(100000)
            .heap_size(100000)
            .total_pages(1) // Only 64KB!
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_linear_memory_alloc_data() {
        let layout = MemoryLayout::default_for_size(16);
        let mut mem = LinearMemory::new(layout);

        let offset = mem.alloc_f32(3.14).unwrap();
        assert_eq!(offset, 0);

        let offset2 = mem.alloc_f64(2.718).unwrap();
        assert_eq!(offset2, 8); // Aligned to 8 bytes
    }

    #[test]
    fn test_linear_memory_alloc_array() {
        let layout = MemoryLayout::default_for_size(16);
        let mut mem = LinearMemory::new(layout);

        let values = [1.0f32, 2.0, 3.0, 4.0];
        let offset = mem.alloc_f32_array(&values).unwrap();
        assert_eq!(offset, 0);

        let segments = mem.data_segments();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].data.len(), 16); // 4 * 4 bytes
    }

    #[test]
    fn test_linear_memory_data_segment_overflow() {
        let layout = MemoryLayout::builder()
            .data_size(16)
            .stack_size(1024) // Small stack to fit in 1 page
            .total_pages(1)
            .build()
            .unwrap();

        let mut mem = LinearMemory::new(layout);

        // First allocation should succeed
        assert!(mem.alloc_f64(1.0).is_ok());

        // Second should fail (only 16 bytes available, first took 8)
        let result = mem.alloc_f64(2.0);
        // Actually should succeed since we have 8 bytes left
        assert!(result.is_ok());

        // Third should fail
        let result = mem.alloc_f64(3.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_layout_regions() {
        let layout = MemoryLayout::builder()
            .data_size(4096)
            .stack_size(65536)
            .heap_size(262144)
            .total_pages(16)
            .build()
            .unwrap();

        // Check non-overlapping
        assert!(layout.data_end <= layout.stack_start);
        assert!(layout.stack_end <= layout.heap_start);
        assert!(layout.heap_end <= layout.total_size);
    }
}
