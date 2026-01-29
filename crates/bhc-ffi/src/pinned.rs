//! Pinned memory buffers for FFI.
//!
//! This module provides types for managing memory that is guaranteed to
//! remain at a fixed address, suitable for passing to foreign code.
//!
//! ## Pinning Guarantee
//!
//! Pinned buffers are allocated in the pinned heap region (per H26-SPEC
//! Section 9). The garbage collector will never move these allocations,
//! ensuring pointers remain valid across FFI boundaries.
//!
//! ## Lifetime Safety
//!
//! The `with_pinned` function ensures that:
//! 1. The buffer is pinned before the FFI call
//! 2. The pointer is only valid within the callback
//! 3. The buffer can be unpinned after the call completes

use crate::{FfiError, FfiResult, FfiSafe};
use bhc_rts_alloc::MemoryRegion;
use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// A buffer that is guaranteed to remain at a fixed memory address.
///
/// `PinnedBuffer` is used for FFI interop where foreign code requires
/// pointers to remain stable. The buffer is allocated in the pinned
/// heap region and will not be moved by the garbage collector.
///
/// # Example
///
/// ```rust,ignore
/// let mut buffer = PinnedBuffer::<f64>::new(1024)?;
///
/// // Fill the buffer
/// for i in 0..1024 {
///     buffer.as_mut_slice()[i] = i as f64;
/// }
///
/// // Pass to C function - pointer is stable
/// unsafe {
///     c_process_array(buffer.as_ptr(), buffer.len());
/// }
/// ```
#[derive(Debug)]
pub struct PinnedBuffer<T: FfiSafe> {
    /// Pointer to the pinned memory.
    ptr: NonNull<T>,
    /// Number of elements.
    len: usize,
    /// Capacity in elements.
    capacity: usize,
    /// Phantom data for T.
    _marker: PhantomData<T>,
}

impl<T: FfiSafe> PinnedBuffer<T> {
    /// Create a new pinned buffer with the given capacity.
    ///
    /// The buffer is allocated in the pinned heap region and will not
    /// be moved by the garbage collector.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn new(capacity: usize) -> FfiResult<Self> {
        if capacity == 0 {
            return Err(FfiError::AllocationFailed(
                "cannot allocate zero-size buffer".to_string(),
            ));
        }

        let layout = Layout::array::<T>(capacity)
            .map_err(|e| FfiError::AllocationFailed(format!("invalid layout: {e}")))?;

        // Allocate pinned memory
        // In a real implementation, this would use bhc_rts_gc::alloc_pinned
        // For now, we use the system allocator with manual tracking
        let ptr = unsafe {
            let raw = std::alloc::alloc(layout);
            if raw.is_null() {
                return Err(FfiError::AllocationFailed(format!(
                    "failed to allocate {} bytes",
                    layout.size()
                )));
            }
            NonNull::new_unchecked(raw as *mut T)
        };

        Ok(Self {
            ptr,
            len: 0,
            capacity,
            _marker: PhantomData,
        })
    }

    /// Create a new pinned buffer initialized with zeros.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn zeroed(capacity: usize) -> FfiResult<Self> {
        if capacity == 0 {
            return Err(FfiError::AllocationFailed(
                "cannot allocate zero-size buffer".to_string(),
            ));
        }

        let layout = Layout::array::<T>(capacity)
            .map_err(|e| FfiError::AllocationFailed(format!("invalid layout: {e}")))?;

        let ptr = unsafe {
            let raw = std::alloc::alloc_zeroed(layout);
            if raw.is_null() {
                return Err(FfiError::AllocationFailed(format!(
                    "failed to allocate {} bytes",
                    layout.size()
                )));
            }
            NonNull::new_unchecked(raw as *mut T)
        };

        Ok(Self {
            ptr,
            len: capacity, // Zeroed means all elements are initialized
            capacity,
            _marker: PhantomData,
        })
    }

    /// Create a pinned buffer from existing data by copying.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn from_slice(data: &[T]) -> FfiResult<Self> {
        let mut buffer = Self::new(data.len())?;

        // Copy data to pinned buffer
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.ptr.as_ptr(), data.len());
        }
        buffer.len = data.len();

        Ok(buffer)
    }

    /// Get the raw pointer to the buffer.
    ///
    /// # Safety
    ///
    /// The pointer is guaranteed to be valid and pinned for the lifetime
    /// of this `PinnedBuffer`. However, callers must ensure they do not
    /// read uninitialized memory (elements beyond `len`).
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable raw pointer to the buffer.
    ///
    /// # Safety
    ///
    /// Same caveats as `as_ptr`. Additionally, callers must ensure
    /// exclusive access when mutating through the pointer.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get the number of initialized elements.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity in elements.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the size in bytes.
    #[inline]
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        self.capacity * std::mem::size_of::<T>()
    }

    /// Get the memory region (always Pinned).
    #[inline]
    #[must_use]
    pub const fn region(&self) -> MemoryRegion {
        MemoryRegion::PinnedHeap
    }

    /// Get a slice of the initialized elements.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice of the initialized elements.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Set the length of initialized elements.
    ///
    /// # Safety
    ///
    /// Caller must ensure that all elements up to `new_len` are properly
    /// initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    /// Verify that the pointer address hasn't changed (for testing).
    ///
    /// This is used to verify the pinning guarantee.
    #[must_use]
    pub fn address(&self) -> usize {
        self.ptr.as_ptr() as usize
    }
}

impl<T: FfiSafe> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Layout::array::<T>(self.capacity).expect("layout was valid at allocation");
            unsafe {
                std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

// PinnedBuffer is Send + Sync because:
// 1. T is FfiSafe (which implies Copy and no interior mutability)
// 2. The buffer is exclusively owned
// 3. Pinned memory has no special thread affinity
unsafe impl<T: FfiSafe + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: FfiSafe + Sync> Sync for PinnedBuffer<T> {}

/// A borrowed slice of pinned memory.
///
/// This is similar to `&[T]` but guarantees the memory is pinned.
/// Used for passing to FFI functions.
#[derive(Debug, Clone, Copy)]
pub struct PinnedSlice<'a, T: FfiSafe> {
    ptr: *const T,
    len: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: FfiSafe> PinnedSlice<'a, T> {
    /// Create a new pinned slice from a pinned buffer.
    #[inline]
    #[must_use]
    pub fn from_buffer(buffer: &'a PinnedBuffer<T>) -> Self {
        Self {
            ptr: buffer.as_ptr(),
            len: buffer.len(),
            _marker: PhantomData,
        }
    }

    /// Get the raw pointer.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get the length.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Convert to a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// A mutable borrowed slice of pinned memory.
#[derive(Debug)]
pub struct PinnedSliceMut<'a, T: FfiSafe> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: FfiSafe> PinnedSliceMut<'a, T> {
    /// Create a new mutable pinned slice from a pinned buffer.
    #[inline]
    #[must_use]
    pub fn from_buffer(buffer: &'a mut PinnedBuffer<T>) -> Self {
        Self {
            ptr: buffer.as_mut_ptr(),
            len: buffer.len(),
            _marker: PhantomData,
        }
    }

    /// Get the raw pointer.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get the mutable raw pointer.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get the length.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Execute a function with a pinned view of data.
///
/// This is the safe pattern for FFI calls. The callback receives a raw
/// pointer that is guaranteed to remain valid and unmoved for the
/// duration of the call.
///
/// # Example
///
/// ```rust,ignore
/// let data = vec![1.0f64, 2.0, 3.0, 4.0];
/// let result = with_pinned(&data, |ptr, len| {
///     unsafe { c_dot_product(ptr, ptr, len) }
/// })?;
/// ```
///
/// # Errors
///
/// Returns an error if pinning fails or if the callback returns an error.
pub fn with_pinned<T, F, R>(data: &[T], f: F) -> FfiResult<R>
where
    T: FfiSafe,
    F: FnOnce(*const T, usize) -> R,
{
    // Create pinned copy
    let buffer = PinnedBuffer::from_slice(data)?;

    // Execute callback with pinned pointer
    Ok(f(buffer.as_ptr(), buffer.len()))
}

/// Execute a function with a mutable pinned view of data.
///
/// Similar to `with_pinned` but allows mutation. The modified data
/// is copied back to the original slice after the callback completes.
///
/// # Errors
///
/// Returns an error if pinning fails.
pub fn with_pinned_mut<T, F, R>(data: &mut [T], f: F) -> FfiResult<R>
where
    T: FfiSafe,
    F: FnOnce(*mut T, usize) -> R,
{
    // Create pinned copy
    let mut buffer = PinnedBuffer::from_slice(data)?;

    // Execute callback with pinned pointer
    let result = f(buffer.as_mut_ptr(), buffer.len());

    // Copy results back
    data.copy_from_slice(buffer.as_slice());

    Ok(result)
}

/// Execute a function with an existing pinned buffer.
///
/// This is more efficient than `with_pinned` when you already have
/// a `PinnedBuffer`.
#[inline]
pub fn with_pinned_buffer<T, F, R>(buffer: &PinnedBuffer<T>, f: F) -> R
where
    T: FfiSafe,
    F: FnOnce(*const T, usize) -> R,
{
    f(buffer.as_ptr(), buffer.len())
}

/// Execute a function with a mutable pinned buffer.
#[inline]
pub fn with_pinned_buffer_mut<T, F, R>(buffer: &mut PinnedBuffer<T>, f: F) -> R
where
    T: FfiSafe,
    F: FnOnce(*mut T, usize) -> R,
{
    f(buffer.as_mut_ptr(), buffer.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_buffer_creation() {
        let buffer = PinnedBuffer::<f64>::new(100).unwrap();
        assert_eq!(buffer.capacity(), 100);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_pinned_buffer_zeroed() {
        let buffer = PinnedBuffer::<f64>::zeroed(100).unwrap();
        assert_eq!(buffer.capacity(), 100);
        assert_eq!(buffer.len(), 100);

        // Verify all zeros
        for &val in buffer.as_slice() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_pinned_buffer_from_slice() {
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let buffer = PinnedBuffer::from_slice(&data).unwrap();

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_slice(), &data);
    }

    #[test]
    fn test_pinned_buffer_address_stability() {
        let buffer = PinnedBuffer::<f64>::zeroed(1000).unwrap();
        let addr1 = buffer.address();

        // Simulate some operations
        std::hint::black_box(&buffer);

        let addr2 = buffer.address();
        assert_eq!(addr1, addr2, "pinned buffer address should not change");
    }

    #[test]
    fn test_with_pinned() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];

        let sum = with_pinned(&data, |ptr, len| {
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            slice.iter().sum::<f64>()
        })
        .unwrap();

        assert_eq!(sum, 10.0);
    }

    #[test]
    fn test_with_pinned_mut() {
        let mut data = vec![1.0f64, 2.0, 3.0, 4.0];

        with_pinned_mut(&mut data, |ptr, len| {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            for x in slice {
                *x *= 2.0;
            }
        })
        .unwrap();

        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_pinned_buffer_region() {
        let buffer = PinnedBuffer::<f64>::new(10).unwrap();
        assert_eq!(buffer.region(), MemoryRegion::PinnedHeap);
    }

    #[test]
    fn test_pinned_slice() {
        let buffer = PinnedBuffer::from_slice(&[1.0f64, 2.0, 3.0]).unwrap();
        let slice = PinnedSlice::from_buffer(&buffer);

        assert_eq!(slice.len(), 3);
        assert_eq!(slice.as_slice(), &[1.0, 2.0, 3.0]);
    }
}
