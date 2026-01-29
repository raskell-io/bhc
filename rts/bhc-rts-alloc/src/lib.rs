//! Memory allocation primitives for the BHC Runtime System.
//!
//! This crate provides the foundational memory allocation primitives used by
//! the BHC runtime. It defines the core abstractions for memory regions
//! as specified in H26-SPEC Section 9: Memory Model.
//!
//! # Memory Regions
//!
//! BHC defines three allocation regions:
//!
//! 1. **Hot Arena** - Bump allocator, freed at scope end (see `bhc-rts-arena`)
//! 2. **Pinned Heap** - Non-moving memory for FFI/device IO
//! 3. **General Heap** - GC-managed boxed structures (see `bhc-rts-gc`)
//!
//! # Design Goals
//!
//! - Zero-cost abstractions for allocation patterns
//! - Explicit control over memory placement
//! - Safe FFI interop through pinned allocations
//! - Support for SIMD-aligned allocations

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod static_alloc;

use std::alloc::{Layout, LayoutError};
use std::ptr::NonNull;

/// Alignment requirements for different allocation purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    /// Default alignment (8 bytes on 64-bit systems).
    Default,
    /// Cache line alignment (64 bytes).
    CacheLine,
    /// SIMD 128-bit alignment (16 bytes).
    Simd128,
    /// SIMD 256-bit alignment (32 bytes, AVX).
    Simd256,
    /// SIMD 512-bit alignment (64 bytes, AVX-512).
    Simd512,
    /// Page alignment (4096 bytes).
    Page,
}

impl Alignment {
    /// Get the alignment value in bytes.
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        match self {
            Self::Default => 8,
            Self::CacheLine => 64,
            Self::Simd128 => 16,
            Self::Simd256 => 32,
            Self::Simd512 => 64,
            Self::Page => 4096,
        }
    }
}

/// Memory region classification per H26-SPEC Section 9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegion {
    /// Hot arena: bump allocator with scope-based lifetime.
    /// Used for ephemeral allocations in numeric kernels.
    HotArena,
    /// Pinned heap: non-moving memory for FFI and device IO.
    /// Must not be relocated by GC.
    PinnedHeap,
    /// General heap: GC-managed boxed structures.
    /// May be moved during garbage collection.
    GeneralHeap,
    /// GPU device memory: high-bandwidth memory on a GPU device.
    /// Managed separately from host memory.
    DeviceMemory(DeviceMemoryKind),
}

/// Type of GPU device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryKind {
    /// NVIDIA CUDA device memory.
    Cuda(u32),
    /// AMD ROCm/HIP device memory.
    Rocm(u32),
}

/// Metadata for a memory block.
#[derive(Debug, Clone, Copy)]
pub struct BlockMeta {
    /// Size of the allocated block in bytes.
    pub size: usize,
    /// Alignment of the block.
    pub alignment: usize,
    /// Memory region this block belongs to.
    pub region: MemoryRegion,
    /// Whether this block is pinned (cannot be moved by GC).
    pub pinned: bool,
}

/// Result type for allocation operations.
pub type AllocResult<T> = Result<T, AllocError>;

/// Errors that can occur during allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    /// Out of memory.
    OutOfMemory {
        /// Requested allocation size.
        requested: usize,
    },
    /// Invalid layout (e.g., zero size or invalid alignment).
    InvalidLayout(String),
    /// Arena capacity exhausted.
    ArenaExhausted {
        /// Current arena usage.
        current: usize,
        /// Maximum arena capacity.
        capacity: usize,
    },
    /// Alignment requirement not met.
    AlignmentError {
        /// Requested alignment.
        requested: usize,
        /// Maximum supported alignment.
        supported: usize,
    },
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory { requested } => {
                write!(f, "out of memory: failed to allocate {requested} bytes")
            }
            Self::InvalidLayout(msg) => write!(f, "invalid layout: {msg}"),
            Self::ArenaExhausted { current, capacity } => {
                write!(
                    f,
                    "arena exhausted: {current} bytes used of {capacity} bytes capacity"
                )
            }
            Self::AlignmentError {
                requested,
                supported,
            } => {
                write!(
                    f,
                    "alignment error: requested {requested}, max supported {supported}"
                )
            }
        }
    }
}

impl std::error::Error for AllocError {}

impl From<LayoutError> for AllocError {
    fn from(e: LayoutError) -> Self {
        Self::InvalidLayout(e.to_string())
    }
}

/// Trait for memory allocators in the RTS.
///
/// This trait provides a unified interface for different allocation strategies
/// used by the runtime system.
pub trait Allocator {
    /// Allocate a block of memory with the given layout.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The returned pointer is properly deallocated using `deallocate`
    /// - The memory is not accessed after deallocation
    unsafe fn allocate(&self, layout: Layout) -> AllocResult<NonNull<u8>>;

    /// Deallocate a previously allocated block.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was allocated by this allocator with the same `layout`
    /// - `ptr` has not been deallocated before
    /// - No references to the memory exist after this call
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Allocate zeroed memory.
    ///
    /// # Safety
    ///
    /// Same requirements as `allocate`.
    unsafe fn allocate_zeroed(&self, layout: Layout) -> AllocResult<NonNull<u8>> {
        let ptr = unsafe { self.allocate(layout)? };
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, layout.size());
        }
        Ok(ptr)
    }

    /// Reallocate a block of memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was allocated by this allocator with `old_layout`
    /// - `new_layout.size()` is greater than zero
    unsafe fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_size: usize,
    ) -> AllocResult<NonNull<u8>> {
        let new_layout = Layout::from_size_align(new_size, old_layout.align())?;
        let new_ptr = unsafe { self.allocate(new_layout)? };

        let copy_size = old_layout.size().min(new_size);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), copy_size);
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }
}

/// A raw memory block with associated metadata.
#[derive(Debug)]
pub struct RawBlock {
    ptr: NonNull<u8>,
    layout: Layout,
    region: MemoryRegion,
}

impl RawBlock {
    /// Create a new raw block (for use by allocator implementations).
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` points to validly allocated memory
    /// with the given layout.
    #[must_use]
    pub const unsafe fn new(ptr: NonNull<u8>, layout: Layout, region: MemoryRegion) -> Self {
        Self {
            ptr,
            layout,
            region,
        }
    }

    /// Get the pointer to the block's data.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the non-null pointer to the block's data.
    #[inline]
    #[must_use]
    pub const fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }

    /// Get the layout of this block.
    #[inline]
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Get the size of this block in bytes.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> usize {
        self.layout.size()
    }

    /// Get the alignment of this block.
    #[inline]
    #[must_use]
    pub const fn align(&self) -> usize {
        self.layout.align()
    }

    /// Get the memory region this block belongs to.
    #[inline]
    #[must_use]
    pub const fn region(&self) -> MemoryRegion {
        self.region
    }
}

/// Statistics for memory allocation tracking.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocStats {
    /// Total bytes currently allocated.
    pub bytes_allocated: usize,
    /// Total number of allocations performed.
    pub allocation_count: usize,
    /// Total number of deallocations performed.
    pub deallocation_count: usize,
    /// Peak memory usage in bytes.
    pub peak_bytes: usize,
    /// Number of failed allocations.
    pub failed_allocations: usize,
}

impl AllocStats {
    /// Create new empty statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
            peak_bytes: 0,
            failed_allocations: 0,
        }
    }

    /// Record an allocation.
    pub fn record_alloc(&mut self, size: usize) {
        self.bytes_allocated += size;
        self.allocation_count += 1;
        self.peak_bytes = self.peak_bytes.max(self.bytes_allocated);
    }

    /// Record a deallocation.
    pub fn record_dealloc(&mut self, size: usize) {
        self.bytes_allocated = self.bytes_allocated.saturating_sub(size);
        self.deallocation_count += 1;
    }

    /// Record a failed allocation.
    pub fn record_failure(&mut self) {
        self.failed_allocations += 1;
    }
}

/// Utility function to align a size up to the given alignment.
#[inline]
#[must_use]
pub const fn align_up(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (size + align - 1) & !(align - 1)
}

/// Utility function to check if a pointer is aligned.
#[inline]
#[must_use]
pub fn is_aligned(ptr: *const u8, align: usize) -> bool {
    debug_assert!(align.is_power_of_two());
    (ptr as usize) & (align - 1) == 0
}

// ============================================================================
// Pinned Allocator Implementation
// ============================================================================

/// A pinned memory allocator that uses the system allocator.
///
/// Memory allocated through this allocator is guaranteed not to be moved
/// by the garbage collector, making it safe for FFI and device I/O.
///
/// # Example
///
/// ```
/// use bhc_rts_alloc::{PinnedAllocator, Allocator};
/// use std::alloc::Layout;
///
/// let alloc = PinnedAllocator::new();
/// unsafe {
///     let layout = Layout::from_size_align(1024, 64).unwrap();
///     let ptr = alloc.allocate(layout).unwrap();
///     // Use the memory...
///     alloc.deallocate(ptr, layout);
/// }
/// ```
#[derive(Debug, Default)]
pub struct PinnedAllocator {
    stats: std::cell::UnsafeCell<AllocStats>,
}

// Safety: The UnsafeCell is only used for interior mutability of statistics,
// which is only modified during allocation/deallocation operations.
// In practice, allocations should be synchronized at a higher level.
unsafe impl Send for PinnedAllocator {}
unsafe impl Sync for PinnedAllocator {}

impl PinnedAllocator {
    /// Create a new pinned allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stats: std::cell::UnsafeCell::new(AllocStats::new()),
        }
    }

    /// Get allocation statistics.
    #[must_use]
    pub fn stats(&self) -> AllocStats {
        // Safety: We're only reading the stats
        unsafe { *self.stats.get() }
    }

    fn record_alloc(&self, size: usize) {
        // Safety: Single-threaded modification during allocation
        unsafe { (*self.stats.get()).record_alloc(size) }
    }

    fn record_dealloc(&self, size: usize) {
        // Safety: Single-threaded modification during deallocation
        unsafe { (*self.stats.get()).record_dealloc(size) }
    }

    fn record_failure(&self) {
        // Safety: Single-threaded modification
        unsafe { (*self.stats.get()).record_failure() }
    }
}

impl Allocator for PinnedAllocator {
    unsafe fn allocate(&self, layout: Layout) -> AllocResult<NonNull<u8>> {
        if layout.size() == 0 {
            // Return a non-null dangling pointer for zero-sized allocations
            return Ok(NonNull::dangling());
        }

        // Use the global allocator
        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            self.record_failure();
            return Err(AllocError::OutOfMemory {
                requested: layout.size(),
            });
        }

        self.record_alloc(layout.size());

        // Safety: We just checked that ptr is not null
        Ok(unsafe { NonNull::new_unchecked(ptr) })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return; // Nothing to deallocate for zero-sized
        }

        self.record_dealloc(layout.size());
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }
}

/// A pinned memory buffer with automatic deallocation.
///
/// This type owns a block of pinned memory and ensures it is properly
/// deallocated when dropped. The memory is guaranteed not to be moved
/// by the garbage collector.
///
/// # Example
///
/// ```
/// use bhc_rts_alloc::PinnedBuffer;
///
/// // Allocate a pinned buffer for 1000 f64 values
/// let buffer: PinnedBuffer<f64> = PinnedBuffer::new(1000).unwrap();
///
/// // Write to the buffer
/// unsafe {
///     buffer.as_mut_ptr().write(42.0);
/// }
///
/// // Buffer is automatically freed when dropped
/// ```
pub struct PinnedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    allocator: PinnedAllocator,
}

// Safety: PinnedBuffer only contains a pointer to T and metadata.
// The Send/Sync bounds depend on T.
unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

impl<T> PinnedBuffer<T> {
    /// Allocate a new pinned buffer with the given number of elements.
    ///
    /// The memory is uninitialized. Use `new_zeroed` for zero-initialized memory.
    pub fn new(len: usize) -> AllocResult<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                allocator: PinnedAllocator::new(),
            });
        }

        let layout =
            Layout::array::<T>(len).map_err(|e| AllocError::InvalidLayout(e.to_string()))?;
        let allocator = PinnedAllocator::new();

        // Safety: layout is valid and non-zero sized
        let ptr = unsafe { allocator.allocate(layout)? };

        Ok(Self {
            ptr: ptr.cast(),
            len,
            allocator,
        })
    }

    /// Allocate a new pinned buffer with zeroed memory.
    pub fn new_zeroed(len: usize) -> AllocResult<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                allocator: PinnedAllocator::new(),
            });
        }

        let layout =
            Layout::array::<T>(len).map_err(|e| AllocError::InvalidLayout(e.to_string()))?;
        let allocator = PinnedAllocator::new();

        // Safety: layout is valid and non-zero sized
        let ptr = unsafe { allocator.allocate_zeroed(layout)? };

        Ok(Self {
            ptr: ptr.cast(),
            len,
            allocator,
        })
    }

    /// Allocate a new pinned buffer with the given alignment.
    pub fn new_aligned(len: usize, alignment: Alignment) -> AllocResult<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                allocator: PinnedAllocator::new(),
            });
        }

        let size = std::mem::size_of::<T>()
            .checked_mul(len)
            .ok_or_else(|| AllocError::InvalidLayout("size overflow".to_string()))?;
        let align = alignment.as_usize().max(std::mem::align_of::<T>());

        let layout = Layout::from_size_align(size, align)?;
        let allocator = PinnedAllocator::new();

        // Safety: layout is valid
        let ptr = unsafe { allocator.allocate(layout)? };

        Ok(Self {
            ptr: ptr.cast(),
            len,
            allocator,
        })
    }

    /// Get the length (number of elements) of this buffer.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if this buffer is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a raw pointer to the buffer's data.
    ///
    /// The pointer is guaranteed to be non-null and properly aligned for `T`,
    /// but the memory may be uninitialized.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable raw pointer to the buffer's data.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get a slice view of the buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all elements have been properly initialized.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice view of the buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all elements have been properly initialized.
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Copy data from a slice into this buffer.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() > self.len()`.
    pub fn copy_from_slice(&mut self, src: &[T])
    where
        T: Copy,
    {
        assert!(src.len() <= self.len, "source slice too large");
        // Safety: T is Copy, so we can safely copy the bytes
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr(), src.len());
        }
    }

    /// Copy data from this buffer to a slice.
    ///
    /// # Panics
    ///
    /// Panics if `dst.len() > self.len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all elements have been properly initialized.
    pub unsafe fn copy_to_slice(&self, dst: &mut [T])
    where
        T: Copy,
    {
        assert!(dst.len() <= self.len, "destination slice too large");
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), dst.as_mut_ptr(), dst.len());
        }
    }

    /// Get allocation statistics for this buffer.
    #[must_use]
    pub fn stats(&self) -> AllocStats {
        self.allocator.stats()
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            let layout = Layout::array::<T>(self.len).expect("layout was valid at allocation");
            // Safety: This pointer was allocated by our allocator with this layout
            unsafe {
                self.allocator.deallocate(self.ptr.cast(), layout);
            }
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for PinnedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

// ============================================================================
// FFI Functions for Pinned Memory
// ============================================================================

/// Allocate pinned memory for FFI use.
///
/// Returns null on failure. The caller is responsible for calling
/// `bhc_pinned_free` with the same size and alignment.
///
/// # Safety
///
/// The returned pointer must be freed using `bhc_pinned_free`.
#[no_mangle]
pub unsafe extern "C" fn bhc_pinned_alloc(size: usize, alignment: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }

    let align = if alignment == 0 { 8 } else { alignment };

    let layout = match Layout::from_size_align(size, align) {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };

    unsafe { std::alloc::alloc(layout) }
}

/// Allocate zeroed pinned memory for FFI use.
///
/// Returns null on failure.
///
/// # Safety
///
/// The returned pointer must be freed using `bhc_pinned_free`.
#[no_mangle]
pub unsafe extern "C" fn bhc_pinned_alloc_zeroed(size: usize, alignment: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }

    let align = if alignment == 0 { 8 } else { alignment };

    let layout = match Layout::from_size_align(size, align) {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };

    unsafe { std::alloc::alloc_zeroed(layout) }
}

/// Free pinned memory allocated with `bhc_pinned_alloc`.
///
/// # Safety
///
/// - `ptr` must have been allocated by `bhc_pinned_alloc` or `bhc_pinned_alloc_zeroed`
/// - `size` and `alignment` must match the original allocation
/// - The memory must not be accessed after this call
#[no_mangle]
pub unsafe extern "C" fn bhc_pinned_free(ptr: *mut u8, size: usize, alignment: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }

    let align = if alignment == 0 { 8 } else { alignment };

    let layout = match Layout::from_size_align(size, align) {
        Ok(l) => l,
        Err(_) => return,
    };

    unsafe { std::alloc::dealloc(ptr, layout) };
}

/// Reallocate pinned memory.
///
/// Returns null on failure, leaving the original memory intact.
///
/// # Safety
///
/// - `ptr` must have been allocated by `bhc_pinned_alloc`
/// - `old_size` and `alignment` must match the original allocation
#[no_mangle]
pub unsafe extern "C" fn bhc_pinned_realloc(
    ptr: *mut u8,
    old_size: usize,
    new_size: usize,
    alignment: usize,
) -> *mut u8 {
    if ptr.is_null() {
        return unsafe { bhc_pinned_alloc(new_size, alignment) };
    }

    if new_size == 0 {
        unsafe { bhc_pinned_free(ptr, old_size, alignment) };
        return std::ptr::null_mut();
    }

    let align = if alignment == 0 { 8 } else { alignment };

    let old_layout = match Layout::from_size_align(old_size, align) {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };

    let new_layout = match Layout::from_size_align(new_size, align) {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };

    let new_ptr = unsafe { std::alloc::alloc(new_layout) };
    if new_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let copy_size = old_size.min(new_size);
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, new_ptr, copy_size);
        std::alloc::dealloc(ptr, old_layout);
    }

    new_ptr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_values() {
        assert_eq!(Alignment::Default.as_usize(), 8);
        assert_eq!(Alignment::CacheLine.as_usize(), 64);
        assert_eq!(Alignment::Simd128.as_usize(), 16);
        assert_eq!(Alignment::Simd256.as_usize(), 32);
        assert_eq!(Alignment::Simd512.as_usize(), 64);
        assert_eq!(Alignment::Page.as_usize(), 4096);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(100, 64), 128);
    }

    #[test]
    fn test_is_aligned() {
        let aligned_ptr = 0x1000 as *const u8;
        let unaligned_ptr = 0x1001 as *const u8;

        assert!(is_aligned(aligned_ptr, 8));
        assert!(is_aligned(aligned_ptr, 16));
        assert!(is_aligned(aligned_ptr, 4096));
        assert!(!is_aligned(unaligned_ptr, 8));
    }

    #[test]
    fn test_alloc_stats() {
        let mut stats = AllocStats::new();

        stats.record_alloc(100);
        assert_eq!(stats.bytes_allocated, 100);
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.peak_bytes, 100);

        stats.record_alloc(200);
        assert_eq!(stats.bytes_allocated, 300);
        assert_eq!(stats.peak_bytes, 300);

        stats.record_dealloc(100);
        assert_eq!(stats.bytes_allocated, 200);
        assert_eq!(stats.deallocation_count, 1);
        assert_eq!(stats.peak_bytes, 300); // Peak unchanged
    }

    #[test]
    fn test_alloc_error_display() {
        let err = AllocError::OutOfMemory { requested: 1024 };
        assert!(err.to_string().contains("1024"));

        let err = AllocError::ArenaExhausted {
            current: 500,
            capacity: 1000,
        };
        assert!(err.to_string().contains("500"));
        assert!(err.to_string().contains("1000"));
    }

    // ========================================================================
    // Pinned Allocator Tests
    // ========================================================================

    #[test]
    fn test_pinned_allocator_basic() {
        let alloc = PinnedAllocator::new();
        let layout = Layout::from_size_align(1024, 8).unwrap();

        unsafe {
            let ptr = alloc.allocate(layout).unwrap();
            assert!(!ptr.as_ptr().is_null());

            // Write to the memory to verify it's usable
            ptr.as_ptr().write_bytes(0xAB, 1024);

            let stats = alloc.stats();
            assert_eq!(stats.bytes_allocated, 1024);
            assert_eq!(stats.allocation_count, 1);

            alloc.deallocate(ptr, layout);

            let stats = alloc.stats();
            assert_eq!(stats.bytes_allocated, 0);
            assert_eq!(stats.deallocation_count, 1);
        }
    }

    #[test]
    fn test_pinned_allocator_alignment() {
        let alloc = PinnedAllocator::new();

        // Test various alignments
        for align in [8, 16, 32, 64, 128, 256] {
            let layout = Layout::from_size_align(1024, align).unwrap();
            unsafe {
                let ptr = alloc.allocate(layout).unwrap();
                assert!(
                    is_aligned(ptr.as_ptr(), align),
                    "Pointer not aligned to {} bytes",
                    align
                );
                alloc.deallocate(ptr, layout);
            }
        }
    }

    #[test]
    fn test_pinned_allocator_zero_size() {
        let alloc = PinnedAllocator::new();
        let layout = Layout::from_size_align(0, 8).unwrap();

        unsafe {
            let ptr = alloc.allocate(layout).unwrap();
            // Zero-size allocation returns dangling pointer
            assert!(!ptr.as_ptr().is_null());

            // Deallocating zero-size is a no-op
            alloc.deallocate(ptr, layout);
        }
    }

    #[test]
    fn test_pinned_allocator_zeroed() {
        let alloc = PinnedAllocator::new();
        let layout = Layout::from_size_align(1024, 8).unwrap();

        unsafe {
            let ptr = alloc.allocate_zeroed(layout).unwrap();

            // Verify memory is zeroed
            let slice = std::slice::from_raw_parts(ptr.as_ptr(), 1024);
            for byte in slice {
                assert_eq!(*byte, 0);
            }

            alloc.deallocate(ptr, layout);
        }
    }

    #[test]
    fn test_pinned_buffer_basic() {
        let buffer: PinnedBuffer<f64> = PinnedBuffer::new(100).unwrap();
        assert_eq!(buffer.len(), 100);
        assert!(!buffer.is_empty());
        assert!(!buffer.as_ptr().is_null());

        // Write some values
        unsafe {
            buffer.as_mut_ptr().write(42.0);
            buffer.as_mut_ptr().add(99).write(99.0);
        }

        // Buffer stats should show allocation
        let stats = buffer.stats();
        assert!(stats.bytes_allocated > 0);
    }

    #[test]
    fn test_pinned_buffer_zeroed() {
        let buffer: PinnedBuffer<u64> = PinnedBuffer::new_zeroed(50).unwrap();
        assert_eq!(buffer.len(), 50);

        // Verify memory is zeroed
        unsafe {
            let slice = buffer.as_slice();
            for val in slice {
                assert_eq!(*val, 0);
            }
        }
    }

    #[test]
    fn test_pinned_buffer_aligned() {
        // Test SIMD-aligned buffer
        let buffer: PinnedBuffer<f32> = PinnedBuffer::new_aligned(256, Alignment::Simd256).unwrap();

        assert_eq!(buffer.len(), 256);
        assert!(
            is_aligned(buffer.as_ptr() as *const u8, 32),
            "Buffer not 32-byte aligned"
        );
    }

    #[test]
    fn test_pinned_buffer_empty() {
        let buffer: PinnedBuffer<i32> = PinnedBuffer::new(0).unwrap();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_pinned_buffer_copy() {
        let mut buffer: PinnedBuffer<i32> = PinnedBuffer::new_zeroed(10).unwrap();
        let src = [1, 2, 3, 4, 5];

        buffer.copy_from_slice(&src);

        unsafe {
            let slice = buffer.as_slice();
            assert_eq!(&slice[0..5], &src);
            assert_eq!(&slice[5..10], &[0, 0, 0, 0, 0]);
        }
    }

    #[test]
    fn test_pinned_buffer_drop() {
        // Create a buffer in a scope and verify it's deallocated
        {
            let _buffer: PinnedBuffer<f64> = PinnedBuffer::new(1000).unwrap();
            // Buffer exists here
        }
        // Buffer is dropped and memory freed
        // (No way to verify directly, but no memory leak)
    }

    #[test]
    fn test_ffi_pinned_alloc() {
        unsafe {
            let ptr = bhc_pinned_alloc(1024, 64);
            assert!(!ptr.is_null());
            assert!(is_aligned(ptr, 64));

            // Write to verify it's usable
            ptr.write_bytes(0xCD, 1024);

            bhc_pinned_free(ptr, 1024, 64);
        }
    }

    #[test]
    fn test_ffi_pinned_alloc_zeroed() {
        unsafe {
            let ptr = bhc_pinned_alloc_zeroed(512, 16);
            assert!(!ptr.is_null());

            // Verify zeroed
            let slice = std::slice::from_raw_parts(ptr, 512);
            for byte in slice {
                assert_eq!(*byte, 0);
            }

            bhc_pinned_free(ptr, 512, 16);
        }
    }

    #[test]
    fn test_ffi_pinned_realloc() {
        unsafe {
            // Allocate initial buffer
            let ptr = bhc_pinned_alloc(256, 8);
            assert!(!ptr.is_null());

            // Write some data
            for i in 0..256 {
                *ptr.add(i) = i as u8;
            }

            // Reallocate larger
            let new_ptr = bhc_pinned_realloc(ptr, 256, 512, 8);
            assert!(!new_ptr.is_null());

            // Verify data was preserved
            for i in 0..256 {
                assert_eq!(*new_ptr.add(i), i as u8);
            }

            bhc_pinned_free(new_ptr, 512, 8);
        }
    }

    #[test]
    fn test_ffi_pinned_null_safety() {
        unsafe {
            // Zero size returns null
            let ptr = bhc_pinned_alloc(0, 8);
            assert!(ptr.is_null());

            // Free null is safe
            bhc_pinned_free(std::ptr::null_mut(), 0, 8);
            bhc_pinned_free(std::ptr::null_mut(), 100, 8);

            // Realloc null allocates
            let ptr = bhc_pinned_realloc(std::ptr::null_mut(), 0, 100, 8);
            assert!(!ptr.is_null());
            bhc_pinned_free(ptr, 100, 8);
        }
    }
}
