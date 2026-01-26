//! Device memory management.
//!
//! This module provides types for allocating and managing GPU device memory.
//! It follows the BHC memory model (H26-SPEC Section 9) with an additional
//! `DeviceMemory` region for GPU-resident data.
//!
//! # Memory Hierarchy
//!
//! ```text
//! ┌────────────────────────────────────────┐
//! │            Host Memory                 │
//! │  ┌──────────────┐  ┌───────────────┐  │
//! │  │  Hot Arena   │  │ Pinned Buffer │  │
//! │  │  (GC may     │  │ (stable addr) │  │
//! │  │   move)      │  │               │  │
//! │  └──────────────┘  └───────┬───────┘  │
//! │                            │ DMA      │
//! ├────────────────────────────┼──────────┤
//! │        PCIe / NVLink       │          │
//! ├────────────────────────────┼──────────┤
//! │            Device Memory   │          │
//! │  ┌─────────────────────────▼───────┐  │
//! │  │         DeviceBuffer            │  │
//! │  │  (high-bandwidth GPU memory)    │  │
//! │  └─────────────────────────────────┘  │
//! └────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use bhc_gpu::{GpuContext, DeviceBuffer};
//!
//! let ctx = bhc_gpu::select_device(DeviceId(0))?;
//!
//! // Allocate device memory
//! let buf: DeviceBuffer<f32> = ctx.alloc(1024)?;
//! assert_eq!(buf.len(), 1024);
//!
//! // Transfer data to device
//! let host_data = vec![1.0f32; 1024];
//! ctx.copy_to_device(&host_data, &mut buf)?;
//! ```

use crate::device::{DeviceId, DeviceKind};
use crate::{GpuError, GpuResult};
use bhc_ffi::FfiSafe;
use serde::{Deserialize, Serialize};
use std::alloc::Layout;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A device memory pointer.
///
/// This is an opaque handle to memory on a GPU device. The actual
/// representation depends on the GPU runtime (CUDA CUdeviceptr, HIP void*).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub(crate) u64);

impl DevicePtr {
    /// Create a null device pointer.
    #[must_use]
    pub const fn null() -> Self {
        Self(0)
    }

    /// Check if this is a null pointer.
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Get the raw pointer value.
    #[must_use]
    pub const fn as_raw(self) -> u64 {
        self.0
    }

    /// Create from a raw pointer value.
    ///
    /// # Safety
    ///
    /// The caller must ensure the pointer value is valid for the
    /// associated device context.
    #[must_use]
    pub const unsafe fn from_raw(ptr: u64) -> Self {
        Self(ptr)
    }

    /// Offset the pointer by a number of bytes.
    #[must_use]
    pub const fn offset(self, bytes: usize) -> Self {
        Self(self.0 + bytes as u64)
    }
}

/// Allocation statistics for device memory.
#[derive(Debug, Default)]
pub struct DeviceAllocStats {
    /// Total bytes allocated.
    pub bytes_allocated: AtomicUsize,
    /// Peak memory usage.
    pub peak_bytes: AtomicUsize,
    /// Number of allocations.
    pub allocation_count: AtomicUsize,
    /// Number of deallocations.
    pub deallocation_count: AtomicUsize,
}

impl DeviceAllocStats {
    /// Create new empty stats.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes_allocated: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
        }
    }

    /// Record an allocation.
    pub fn record_alloc(&self, size: usize) {
        let prev = self.bytes_allocated.fetch_add(size, Ordering::SeqCst);
        let new_total = prev + size;
        // Update peak if necessary
        let mut current_peak = self.peak_bytes.load(Ordering::SeqCst);
        while new_total > current_peak {
            match self.peak_bytes.compare_exchange_weak(
                current_peak,
                new_total,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(p) => current_peak = p,
            }
        }
        self.allocation_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Record a deallocation.
    pub fn record_dealloc(&self, size: usize) {
        self.bytes_allocated.fetch_sub(size, Ordering::SeqCst);
        self.deallocation_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Get current allocated bytes.
    #[must_use]
    pub fn current_bytes(&self) -> usize {
        self.bytes_allocated.load(Ordering::SeqCst)
    }

    /// Get peak allocated bytes.
    #[must_use]
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::SeqCst)
    }
}

/// Internal allocation record.
struct DeviceAlloc {
    ptr: DevicePtr,
    size: usize,
    device: DeviceId,
    device_kind: DeviceKind,
}

impl Drop for DeviceAlloc {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Free device memory through the appropriate runtime
            match self.device_kind {
                #[cfg(feature = "cuda")]
                DeviceKind::Cuda => {
                    let _ = crate::runtime::cuda::free(self.ptr);
                }
                #[cfg(feature = "rocm")]
                DeviceKind::Rocm => {
                    let _ = crate::runtime::rocm::free(self.ptr);
                }
                DeviceKind::Mock | _ => {
                    // Mock device: no actual deallocation
                }
            }
        }
    }
}

/// A buffer of typed elements in device memory.
///
/// `DeviceBuffer` provides safe management of GPU device memory with
/// automatic cleanup on drop. It tracks the element type, length, and
/// owning device.
///
/// # Type Safety
///
/// The element type `T` must implement `FfiSafe`, ensuring it can be
/// safely transferred between host and device memory.
///
/// # Example
///
/// ```rust,ignore
/// use bhc_gpu::{GpuContext, DeviceBuffer};
///
/// let ctx = bhc_gpu::select_device(DeviceId(0))?;
///
/// // Allocate 1024 f32 elements on device
/// let buf: DeviceBuffer<f32> = ctx.alloc(1024)?;
///
/// // Copy data from host
/// let host_data = vec![1.0f32; 1024];
/// ctx.copy_to_device(&host_data, &mut buf)?;
///
/// // DeviceBuffer is automatically freed when dropped
/// ```
pub struct DeviceBuffer<T: FfiSafe> {
    alloc: Arc<DeviceAlloc>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: FfiSafe> DeviceBuffer<T> {
    /// Create a new device buffer.
    ///
    /// This is called internally by `GpuContext::alloc`.
    pub(crate) fn new(ptr: DevicePtr, len: usize, device: DeviceId, kind: DeviceKind) -> Self {
        Self {
            alloc: Arc::new(DeviceAlloc {
                ptr,
                size: len * std::mem::size_of::<T>(),
                device,
                device_kind: kind,
            }),
            len,
            _marker: PhantomData,
        }
    }

    /// Create an uninitialized buffer (for internal use).
    ///
    /// # Safety
    ///
    /// The caller must ensure the buffer is properly initialized before use.
    pub(crate) unsafe fn uninit(len: usize, device: DeviceId, kind: DeviceKind) -> GpuResult<Self> {
        let size = len * std::mem::size_of::<T>();
        let ptr = allocate_device_memory(size, device, kind)?;
        Ok(Self::new(ptr, len, device, kind))
    }

    /// Get the device pointer to the buffer data.
    #[must_use]
    pub fn as_ptr(&self) -> DevicePtr {
        self.alloc.ptr
    }

    /// Get the number of elements in the buffer.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.alloc.size
    }

    /// Get the device ID this buffer belongs to.
    #[must_use]
    pub fn device(&self) -> DeviceId {
        self.alloc.device
    }

    /// Get the device kind.
    #[must_use]
    pub fn device_kind(&self) -> DeviceKind {
        self.alloc.device_kind
    }

    /// Get the memory layout of the buffer.
    #[must_use]
    pub fn layout(&self) -> Layout {
        Layout::from_size_align(self.size_bytes(), std::mem::align_of::<T>()).expect("valid layout")
    }

    /// Get a raw byte view of the buffer.
    #[must_use]
    pub fn as_bytes(&self) -> DeviceBufferView<u8> {
        DeviceBufferView {
            ptr: self.alloc.ptr,
            len: self.size_bytes(),
            _marker: PhantomData,
        }
    }

    /// Get a mutable raw byte view of the buffer.
    #[must_use]
    pub fn as_bytes_mut(&mut self) -> DeviceBufferViewMut<u8> {
        DeviceBufferViewMut {
            ptr: self.alloc.ptr,
            len: self.size_bytes(),
            _marker: PhantomData,
        }
    }

    /// Create a view of a slice of the buffer.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    #[must_use]
    pub fn slice(&self, start: usize, len: usize) -> DeviceBufferView<T> {
        assert!(start + len <= self.len, "slice out of bounds");
        DeviceBufferView {
            ptr: self.alloc.ptr.offset(start * std::mem::size_of::<T>()),
            len,
            _marker: PhantomData,
        }
    }

    /// Create a mutable view of a slice of the buffer.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    #[must_use]
    pub fn slice_mut(&mut self, start: usize, len: usize) -> DeviceBufferViewMut<T> {
        assert!(start + len <= self.len, "slice out of bounds");
        DeviceBufferViewMut {
            ptr: self.alloc.ptr.offset(start * std::mem::size_of::<T>()),
            len,
            _marker: PhantomData,
        }
    }
}

impl<T: FfiSafe> Clone for DeviceBuffer<T> {
    fn clone(&self) -> Self {
        // This creates a shallow clone that shares the underlying allocation.
        // For a deep copy, use GpuContext::copy_device_to_device.
        Self {
            alloc: Arc::clone(&self.alloc),
            len: self.len,
            _marker: PhantomData,
        }
    }
}

impl<T: FfiSafe> std::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &self.alloc.ptr)
            .field("len", &self.len)
            .field("size_bytes", &self.size_bytes())
            .field("device", &self.alloc.device)
            .field("element_type", &std::any::type_name::<T>())
            .finish()
    }
}

/// An immutable view into a device buffer.
pub struct DeviceBufferView<T: FfiSafe> {
    ptr: DevicePtr,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: FfiSafe> DeviceBufferView<T> {
    /// Get the device pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> DevicePtr {
        self.ptr
    }

    /// Get the number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T: FfiSafe> std::fmt::Debug for DeviceBufferView<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBufferView")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

/// A mutable view into a device buffer.
pub struct DeviceBufferViewMut<T: FfiSafe> {
    ptr: DevicePtr,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: FfiSafe> DeviceBufferViewMut<T> {
    /// Get the device pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> DevicePtr {
        self.ptr
    }

    /// Get the number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T: FfiSafe> std::fmt::Debug for DeviceBufferViewMut<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBufferViewMut")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

/// Memory allocation flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocFlags {
    /// Zero-initialize the memory.
    pub zeroed: bool,
    /// Use managed memory (unified memory).
    pub managed: bool,
    /// Use write-combined memory (optimized for host writes).
    pub write_combined: bool,
}

impl AllocFlags {
    /// Default allocation flags.
    #[must_use]
    pub const fn default_flags() -> Self {
        Self {
            zeroed: false,
            managed: false,
            write_combined: false,
        }
    }

    /// Flags for zeroed memory.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            zeroed: true,
            managed: false,
            write_combined: false,
        }
    }
}

/// Allocate device memory.
fn allocate_device_memory(
    size: usize,
    _device: DeviceId,
    kind: DeviceKind,
) -> GpuResult<DevicePtr> {
    if size == 0 {
        return Ok(DevicePtr::null());
    }

    match kind {
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => crate::runtime::cuda::malloc(size),

        #[cfg(feature = "rocm")]
        DeviceKind::Rocm => crate::runtime::rocm::malloc(size),

        DeviceKind::Mock | _ => {
            // Mock allocation: use host memory
            let layout = Layout::from_size_align(size, 256)
                .map_err(|_| GpuError::AllocationFailed { size })?;

            // Safety: layout is valid
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                Err(GpuError::AllocationFailed { size })
            } else {
                Ok(DevicePtr(ptr as u64))
            }
        }
    }
}

/// Memory pool for device allocations.
///
/// Provides sub-allocation from a larger pool to reduce allocation overhead.
pub struct DeviceMemoryPool {
    device: DeviceId,
    device_kind: DeviceKind,
    block_size: usize,
    blocks: parking_lot::Mutex<Vec<PoolBlock>>,
    stats: DeviceAllocStats,
}

struct PoolBlock {
    ptr: DevicePtr,
    size: usize,
    used: usize,
}

impl DeviceMemoryPool {
    /// Create a new memory pool.
    #[must_use]
    pub fn new(device: DeviceId, device_kind: DeviceKind, block_size: usize) -> Self {
        Self {
            device,
            device_kind,
            block_size,
            blocks: parking_lot::Mutex::new(Vec::new()),
            stats: DeviceAllocStats::new(),
        }
    }

    /// Allocate from the pool.
    pub fn alloc<T: FfiSafe>(&self, len: usize) -> GpuResult<DeviceBuffer<T>> {
        let size = len * std::mem::size_of::<T>();
        let _align = std::mem::align_of::<T>(); // TODO: use for aligned allocation

        // For now, just do direct allocation
        // TODO: Implement proper pool sub-allocation
        let ptr = allocate_device_memory(size, self.device, self.device_kind)?;
        self.stats.record_alloc(size);

        Ok(DeviceBuffer::new(ptr, len, self.device, self.device_kind))
    }

    /// Get allocation statistics.
    #[must_use]
    pub fn stats(&self) -> &DeviceAllocStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_ptr() {
        let ptr = DevicePtr::null();
        assert!(ptr.is_null());
        assert_eq!(ptr.as_raw(), 0);

        let ptr2 = unsafe { DevicePtr::from_raw(0x1000) };
        assert!(!ptr2.is_null());
        assert_eq!(ptr2.as_raw(), 0x1000);

        let ptr3 = ptr2.offset(256);
        assert_eq!(ptr3.as_raw(), 0x1100);
    }

    #[test]
    fn test_alloc_stats() {
        let stats = DeviceAllocStats::new();
        stats.record_alloc(1024);
        assert_eq!(stats.current_bytes(), 1024);
        assert_eq!(stats.peak_bytes(), 1024);

        stats.record_alloc(2048);
        assert_eq!(stats.current_bytes(), 3072);
        assert_eq!(stats.peak_bytes(), 3072);

        stats.record_dealloc(1024);
        assert_eq!(stats.current_bytes(), 2048);
        assert_eq!(stats.peak_bytes(), 3072);
    }

    #[test]
    fn test_alloc_flags() {
        let flags = AllocFlags::default_flags();
        assert!(!flags.zeroed);
        assert!(!flags.managed);

        let zeroed = AllocFlags::zeroed();
        assert!(zeroed.zeroed);
    }

    #[test]
    fn test_mock_device_buffer() {
        // Test with mock allocation
        let ptr = allocate_device_memory(1024, DeviceId(0), DeviceKind::Mock).unwrap();
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(ptr, 256, DeviceId(0), DeviceKind::Mock);

        assert_eq!(buf.len(), 256);
        assert_eq!(buf.size_bytes(), 1024);
        assert!(!buf.is_empty());
        assert_eq!(buf.device(), DeviceId(0));
    }

    #[test]
    fn test_buffer_slice() {
        let ptr = allocate_device_memory(4096, DeviceId(0), DeviceKind::Mock).unwrap();
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(ptr, 1024, DeviceId(0), DeviceKind::Mock);

        let view = buf.slice(0, 256);
        assert_eq!(view.len(), 256);
        assert_eq!(view.size_bytes(), 1024);
    }

    #[test]
    #[should_panic(expected = "slice out of bounds")]
    fn test_buffer_slice_bounds() {
        let ptr = allocate_device_memory(1024, DeviceId(0), DeviceKind::Mock).unwrap();
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(ptr, 256, DeviceId(0), DeviceKind::Mock);

        // This should panic
        let _ = buf.slice(200, 100);
    }
}
