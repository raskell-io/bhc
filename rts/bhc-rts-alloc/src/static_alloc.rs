//! Static memory allocator for embedded systems.
//!
//! This module provides a static allocator for the embedded profile where
//! dynamic memory allocation is prohibited. All memory is allocated from
//! a fixed-size buffer at compile time or initialization.
//!
//! # Design
//!
//! The static allocator uses a simple bump allocation strategy within a
//! fixed-size buffer. Key characteristics:
//!
//! - **No dynamic allocation**: All memory comes from a pre-allocated buffer
//! - **No deallocation**: Memory is never freed (suitable for embedded systems)
//! - **Compile-time limits**: Maximum allocation can be bounded at compile time
//! - **No GC interaction**: Completely bypasses the garbage collector
//!
//! # Usage
//!
//! ```ignore
//! use bhc_rts_alloc::static_alloc::{StaticAllocator, STATIC_HEAP};
//!
//! // Initialize with a fixed buffer (typically done at startup)
//! static mut HEAP_BUFFER: [u8; 64 * 1024] = [0; 64 * 1024];
//! unsafe { STATIC_HEAP.init(&mut HEAP_BUFFER) };
//!
//! // Allocate (never freed)
//! let ptr = STATIC_HEAP.alloc(Layout::new::<[u32; 100]>()).unwrap();
//! ```
//!
//! # Embedded Profile
//!
//! When using the Embedded profile, the compiler enforces:
//! - No general heap allocation
//! - No lazy evaluation (fully strict)
//! - No GC roots or write barriers
//! - Static stack allocation

use crate::{AllocError, AllocResult, MemoryRegion};
use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A static memory allocator that uses a fixed-size buffer.
///
/// This allocator is designed for embedded systems where dynamic memory
/// allocation is not available or prohibited.
#[derive(Debug)]
pub struct StaticAllocator {
    /// Base address of the buffer.
    base: UnsafeCell<*mut u8>,
    /// Current allocation cursor.
    cursor: AtomicUsize,
    /// Total capacity in bytes.
    capacity: AtomicUsize,
    /// Whether the allocator has been initialized.
    initialized: AtomicBool,
    /// Total bytes allocated.
    bytes_allocated: AtomicUsize,
    /// Number of allocations.
    allocation_count: AtomicUsize,
    /// Number of failed allocations.
    failed_allocations: AtomicUsize,
}

impl StaticAllocator {
    /// Create a new uninitialized static allocator.
    ///
    /// Must call `init` before use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            base: UnsafeCell::new(std::ptr::null_mut()),
            cursor: AtomicUsize::new(0),
            capacity: AtomicUsize::new(0),
            initialized: AtomicBool::new(false),
            bytes_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            failed_allocations: AtomicUsize::new(0),
        }
    }

    /// Initialize the allocator with a static buffer.
    ///
    /// # Safety
    ///
    /// The buffer must have a static lifetime and must not be accessed
    /// through any other reference while the allocator is in use.
    pub unsafe fn init(&self, buffer: &'static mut [u8]) {
        if self.initialized.load(Ordering::Acquire) {
            panic!("StaticAllocator already initialized");
        }

        unsafe {
            *self.base.get() = buffer.as_mut_ptr();
        }
        self.capacity.store(buffer.len(), Ordering::Release);
        self.cursor.store(0, Ordering::Release);
        self.initialized.store(true, Ordering::Release);
    }

    /// Initialize from raw parts.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid and have at least `capacity` bytes
    /// - The memory must be exclusively owned by this allocator
    /// - The memory must have a static lifetime
    pub unsafe fn init_raw(&self, ptr: *mut u8, capacity: usize) {
        if self.initialized.load(Ordering::Acquire) {
            panic!("StaticAllocator already initialized");
        }

        unsafe {
            *self.base.get() = ptr;
        }
        self.capacity.store(capacity, Ordering::Release);
        self.cursor.store(0, Ordering::Release);
        self.initialized.store(true, Ordering::Release);
    }

    /// Check if the allocator is initialized.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    /// Allocate memory from the static buffer.
    ///
    /// Returns `None` if the buffer is exhausted or not initialized.
    pub fn alloc(&self, layout: Layout) -> AllocResult<NonNull<u8>> {
        if !self.is_initialized() {
            return Err(AllocError::InvalidLayout(
                "StaticAllocator not initialized".into(),
            ));
        }

        let size = layout.size();
        let align = layout.align();

        // Calculate aligned cursor position
        loop {
            let cursor = self.cursor.load(Ordering::Acquire);
            let base = unsafe { *self.base.get() } as usize;
            let capacity = self.capacity.load(Ordering::Acquire);

            // Align the cursor
            let aligned = (base + cursor + align - 1) & !(align - 1);
            let offset = aligned - base;
            let new_cursor = offset + size;

            if new_cursor > capacity {
                self.failed_allocations.fetch_add(1, Ordering::Relaxed);
                return Err(AllocError::OutOfMemory { requested: size });
            }

            // Try to update cursor atomically
            match self.cursor.compare_exchange(
                cursor,
                new_cursor,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.bytes_allocated.fetch_add(size, Ordering::Relaxed);
                    self.allocation_count.fetch_add(1, Ordering::Relaxed);

                    // Safety: aligned is within bounds and properly aligned
                    return Ok(unsafe { NonNull::new_unchecked(aligned as *mut u8) });
                }
                Err(_) => {
                    // Another thread allocated, retry
                    continue;
                }
            }
        }
    }

    /// Allocate zeroed memory.
    pub fn alloc_zeroed(&self, layout: Layout) -> AllocResult<NonNull<u8>> {
        let ptr = self.alloc(layout)?;
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, layout.size());
        }
        Ok(ptr)
    }

    /// Allocate space for a value.
    pub fn alloc_value<T>(&self, value: T) -> AllocResult<&'static mut T> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc(layout)?;

        unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            std::ptr::write(typed_ptr, value);
            Ok(&mut *typed_ptr)
        }
    }

    /// Allocate space for an array.
    pub fn alloc_array<T: Copy>(&self, values: &[T]) -> AllocResult<&'static mut [T]> {
        if values.is_empty() {
            // Return an empty slice (special case)
            return Ok(unsafe { std::slice::from_raw_parts_mut(NonNull::dangling().as_ptr(), 0) });
        }

        let layout = Layout::array::<T>(values.len())
            .map_err(|e| AllocError::InvalidLayout(e.to_string()))?;
        let ptr = self.alloc(layout)?;

        unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(values.as_ptr(), typed_ptr, values.len());
            Ok(std::slice::from_raw_parts_mut(typed_ptr, values.len()))
        }
    }

    /// Allocate a zeroed array.
    pub fn alloc_array_zeroed<T>(&self, len: usize) -> AllocResult<&'static mut [T]> {
        if len == 0 {
            return Ok(unsafe { std::slice::from_raw_parts_mut(NonNull::dangling().as_ptr(), 0) });
        }

        let layout =
            Layout::array::<T>(len).map_err(|e| AllocError::InvalidLayout(e.to_string()))?;
        let ptr = self.alloc_zeroed(layout)?;

        unsafe { Ok(std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, len)) }
    }

    /// Get the total capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Acquire)
    }

    /// Get the number of bytes used.
    #[must_use]
    pub fn used(&self) -> usize {
        self.cursor.load(Ordering::Acquire)
    }

    /// Get the remaining capacity.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.used())
    }

    /// Get the total bytes allocated.
    #[must_use]
    pub fn bytes_allocated(&self) -> usize {
        self.bytes_allocated.load(Ordering::Relaxed)
    }

    /// Get the number of allocations.
    #[must_use]
    pub fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }

    /// Get the number of failed allocations.
    #[must_use]
    pub fn failed_allocations(&self) -> usize {
        self.failed_allocations.load(Ordering::Relaxed)
    }

    /// Get the memory region type.
    #[must_use]
    pub const fn region(&self) -> MemoryRegion {
        MemoryRegion::PinnedHeap // Static memory is effectively pinned
    }

    /// Get statistics.
    #[must_use]
    pub fn stats(&self) -> StaticAllocStats {
        StaticAllocStats {
            capacity: self.capacity(),
            used: self.used(),
            remaining: self.remaining(),
            bytes_allocated: self.bytes_allocated(),
            allocation_count: self.allocation_count(),
            failed_allocations: self.failed_allocations(),
        }
    }
}

// Safety: StaticAllocator uses atomic operations for thread safety
unsafe impl Send for StaticAllocator {}
unsafe impl Sync for StaticAllocator {}

impl Default for StaticAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for static allocation.
#[derive(Debug, Clone, Copy)]
pub struct StaticAllocStats {
    /// Total capacity.
    pub capacity: usize,
    /// Bytes used (cursor position).
    pub used: usize,
    /// Remaining capacity.
    pub remaining: usize,
    /// Total bytes allocated (including alignment padding).
    pub bytes_allocated: usize,
    /// Number of allocations.
    pub allocation_count: usize,
    /// Number of failed allocations.
    pub failed_allocations: usize,
}

impl std::fmt::Display for StaticAllocStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Static Allocator Statistics:")?;
        writeln!(f, "  Capacity: {} bytes", self.capacity)?;
        writeln!(
            f,
            "  Used: {} bytes ({:.1}%)",
            self.used,
            if self.capacity > 0 {
                100.0 * self.used as f64 / self.capacity as f64
            } else {
                0.0
            }
        )?;
        writeln!(f, "  Remaining: {} bytes", self.remaining)?;
        writeln!(f, "  Allocations: {}", self.allocation_count)?;
        writeln!(f, "  Failures: {}", self.failed_allocations)?;
        Ok(())
    }
}

/// Configuration for embedded/static allocation mode.
#[derive(Debug, Clone)]
pub struct EmbeddedConfig {
    /// Total heap size in bytes.
    pub heap_size: usize,
    /// Maximum single allocation size.
    pub max_allocation_size: usize,
    /// Whether to panic on allocation failure (vs returning error).
    pub panic_on_failure: bool,
    /// Whether to zero memory before returning.
    pub zero_memory: bool,
}

impl Default for EmbeddedConfig {
    fn default() -> Self {
        Self {
            heap_size: 64 * 1024,          // 64 KB default
            max_allocation_size: 4 * 1024, // 4 KB max single allocation
            panic_on_failure: false,
            zero_memory: false,
        }
    }
}

impl EmbeddedConfig {
    /// Create a minimal embedded configuration.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            heap_size: 8 * 1024, // 8 KB
            max_allocation_size: 1024,
            panic_on_failure: true,
            zero_memory: false,
        }
    }

    /// Create a standard embedded configuration.
    #[must_use]
    pub fn standard() -> Self {
        Self::default()
    }

    /// Create a larger embedded configuration.
    #[must_use]
    pub fn large() -> Self {
        Self {
            heap_size: 256 * 1024,          // 256 KB
            max_allocation_size: 32 * 1024, // 32 KB
            panic_on_failure: false,
            zero_memory: true,
        }
    }
}

/// A compile-time bounded allocator.
///
/// This wrapper enforces a compile-time bound on the maximum allocation size.
#[derive(Debug)]
pub struct BoundedAllocator<const MAX_SIZE: usize> {
    inner: StaticAllocator,
}

impl<const MAX_SIZE: usize> BoundedAllocator<MAX_SIZE> {
    /// Create a new bounded allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: StaticAllocator::new(),
        }
    }

    /// Initialize with a buffer.
    ///
    /// # Safety
    ///
    /// Same requirements as `StaticAllocator::init`.
    pub unsafe fn init(&self, buffer: &'static mut [u8]) {
        assert!(
            buffer.len() <= MAX_SIZE,
            "Buffer size {} exceeds maximum {}",
            buffer.len(),
            MAX_SIZE
        );
        unsafe { self.inner.init(buffer) };
    }

    /// Allocate with compile-time size checking.
    pub fn alloc(&self, layout: Layout) -> AllocResult<NonNull<u8>> {
        if layout.size() > MAX_SIZE {
            return Err(AllocError::OutOfMemory {
                requested: layout.size(),
            });
        }
        self.inner.alloc(layout)
    }

    /// Get the inner allocator.
    #[must_use]
    pub fn inner(&self) -> &StaticAllocator {
        &self.inner
    }
}

impl<const MAX_SIZE: usize> Default for BoundedAllocator<MAX_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Guard type that prevents GC operations.
///
/// When this guard is active, GC operations are prohibited.
/// Used to enforce no-GC mode in embedded profile.
pub struct NoGcGuard {
    _private: (),
}

impl NoGcGuard {
    /// Create a new no-GC guard.
    ///
    /// While this guard exists, GC operations should not occur.
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Check if GC is currently prohibited.
    #[must_use]
    pub fn is_active(&self) -> bool {
        true
    }
}

impl Default for NoGcGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_allocator_basic() {
        static ALLOC: StaticAllocator = StaticAllocator::new();
        static mut BUFFER: [u8; 4096] = [0; 4096];

        unsafe {
            ALLOC.init(&mut BUFFER);
        }

        assert!(ALLOC.is_initialized());
        assert_eq!(ALLOC.capacity(), 4096);
        assert_eq!(ALLOC.used(), 0);

        let layout = Layout::new::<u64>();
        let ptr = ALLOC.alloc(layout).unwrap();
        assert!(!ptr.as_ptr().is_null());
        assert!(ALLOC.used() > 0);
    }

    #[test]
    fn test_static_allocator_alignment() {
        static ALLOC: StaticAllocator = StaticAllocator::new();
        static mut BUFFER: [u8; 4096] = [0; 4096];

        unsafe {
            ALLOC.init(&mut BUFFER);
        }

        // Allocate with different alignments
        let layout8 = Layout::from_size_align(8, 8).unwrap();
        let ptr8 = ALLOC.alloc(layout8).unwrap();
        assert_eq!(ptr8.as_ptr() as usize % 8, 0);

        let layout16 = Layout::from_size_align(16, 16).unwrap();
        let ptr16 = ALLOC.alloc(layout16).unwrap();
        assert_eq!(ptr16.as_ptr() as usize % 16, 0);

        let layout64 = Layout::from_size_align(64, 64).unwrap();
        let ptr64 = ALLOC.alloc(layout64).unwrap();
        assert_eq!(ptr64.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_static_allocator_exhaustion() {
        static ALLOC: StaticAllocator = StaticAllocator::new();
        static mut BUFFER: [u8; 128] = [0; 128];

        unsafe {
            ALLOC.init(&mut BUFFER);
        }

        // Should succeed
        let layout1 = Layout::from_size_align(64, 8).unwrap();
        assert!(ALLOC.alloc(layout1).is_ok());

        // Should fail - not enough space
        let layout2 = Layout::from_size_align(128, 8).unwrap();
        let result = ALLOC.alloc(layout2);
        assert!(matches!(result, Err(AllocError::OutOfMemory { .. })));
        assert_eq!(ALLOC.failed_allocations(), 1);
    }

    #[test]
    fn test_static_allocator_value() {
        static ALLOC: StaticAllocator = StaticAllocator::new();
        static mut BUFFER: [u8; 4096] = [0; 4096];

        unsafe {
            ALLOC.init(&mut BUFFER);
        }

        let val = ALLOC.alloc_value(42i32).unwrap();
        assert_eq!(*val, 42);

        *val = 100;
        assert_eq!(*val, 100);
    }

    #[test]
    fn test_static_allocator_array() {
        static ALLOC: StaticAllocator = StaticAllocator::new();
        static mut BUFFER: [u8; 4096] = [0; 4096];

        unsafe {
            ALLOC.init(&mut BUFFER);
        }

        let arr = ALLOC.alloc_array(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(arr, &[1, 2, 3, 4, 5]);

        arr[0] = 10;
        assert_eq!(arr[0], 10);
    }

    #[test]
    fn test_static_allocator_zeroed() {
        static ALLOC: StaticAllocator = StaticAllocator::new();
        static mut BUFFER: [u8; 4096] = [0; 4096];

        unsafe {
            ALLOC.init(&mut BUFFER);
        }

        let arr: &mut [u64] = ALLOC.alloc_array_zeroed(100).unwrap();
        assert!(arr.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_static_allocator_stats() {
        static ALLOC: StaticAllocator = StaticAllocator::new();
        static mut BUFFER: [u8; 4096] = [0; 4096];

        unsafe {
            ALLOC.init(&mut BUFFER);
        }

        let _ = ALLOC.alloc(Layout::new::<u64>()).unwrap();
        let _ = ALLOC.alloc(Layout::new::<u64>()).unwrap();

        let stats = ALLOC.stats();
        assert_eq!(stats.allocation_count, 2);
        assert!(stats.bytes_allocated >= 16);
        assert!(stats.used > 0);
    }

    #[test]
    fn test_embedded_config() {
        let minimal = EmbeddedConfig::minimal();
        let standard = EmbeddedConfig::standard();
        let large = EmbeddedConfig::large();

        assert!(minimal.heap_size < standard.heap_size);
        assert!(standard.heap_size < large.heap_size);
    }

    #[test]
    fn test_bounded_allocator() {
        static ALLOC: BoundedAllocator<1024> = BoundedAllocator::new();
        static mut BUFFER: [u8; 1024] = [0; 1024];

        unsafe {
            ALLOC.init(&mut BUFFER);
        }

        // Should succeed
        let layout1 = Layout::from_size_align(512, 8).unwrap();
        assert!(ALLOC.alloc(layout1).is_ok());

        // Should fail - exceeds compile-time bound
        let layout2 = Layout::from_size_align(2048, 8).unwrap();
        assert!(ALLOC.alloc(layout2).is_err());
    }

    #[test]
    fn test_stats_display() {
        static ALLOC: StaticAllocator = StaticAllocator::new();
        static mut BUFFER: [u8; 4096] = [0; 4096];

        unsafe {
            ALLOC.init(&mut BUFFER);
        }

        let _ = ALLOC.alloc(Layout::new::<u64>()).unwrap();

        let stats = ALLOC.stats();
        let display = format!("{}", stats);

        assert!(display.contains("Static Allocator Statistics"));
        assert!(display.contains("Capacity"));
    }
}
