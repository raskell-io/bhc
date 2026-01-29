//! Hot Arena integration for tensor temporaries
//!
//! This module provides arena-backed tensor operations for kernel temporaries.
//! The Hot Arena uses bump allocation (O(1)) with bulk deallocation at scope end,
//! avoiding GC overhead for ephemeral allocations.
//!
//! # Memory Model
//!
//! BHC defines three allocation regions (H26-SPEC Section 9):
//!
//! | Region | Allocation | Deallocation | GC | Use Case |
//! |--------|------------|--------------|-----|----------|
//! | **Hot Arena** | Bump pointer O(1) | Bulk free at scope end | None | Kernel temporaries |
//! | **Pinned Heap** | malloc-style | Explicit/refcounted | Never moved | FFI, DMA, GPU |
//! | **General Heap** | GC-managed | Automatic | May move | Normal boxed data |
//!
//! # Usage
//!
//! ```ignore
//! use bhc_numeric::arena::with_tensor_arena;
//!
//! // Temporaries are freed when scope ends
//! with_tensor_arena(1024 * 1024, |arena| {
//!     let tmp1 = arena_zeros_f64(arena, &[1024, 768]);
//!     let tmp2 = arena_zeros_f64(arena, &[768, 512]);
//!     // ... compute with tmp1, tmp2 ...
//!     // Both freed automatically here
//! });
//! ```
//!
//! # Safety
//!
//! Arena allocations MUST NOT escape their scope. The type system helps enforce
//! this through lifetime parameters. FFI code must be careful not to store
//! pointers beyond the arena scope.

use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr::NonNull;

// ============================================================================
// Arena Types
// ============================================================================

/// A hot arena for tensor temporaries.
///
/// Uses bump allocation for O(1) allocation. Memory is freed in bulk
/// when the arena is dropped or reset.
pub struct TensorArena {
    /// Start of arena memory
    base: NonNull<u8>,
    /// Current allocation pointer (bump pointer)
    ptr: NonNull<u8>,
    /// End of arena memory
    end: NonNull<u8>,
    /// Total capacity in bytes
    capacity: usize,
    /// Number of active allocations (for debugging)
    alloc_count: usize,
}

impl TensorArena {
    /// Create a new arena with the specified capacity.
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0 or allocation fails.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Arena capacity must be positive");

        let layout = Layout::from_size_align(capacity, 64).expect("Invalid layout");
        let base = unsafe { std::alloc::alloc(layout) };

        if base.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        let base = unsafe { NonNull::new_unchecked(base) };
        let end = unsafe { NonNull::new_unchecked(base.as_ptr().add(capacity)) };

        TensorArena {
            base,
            ptr: base,
            end,
            capacity,
            alloc_count: 0,
        }
    }

    /// Allocate bytes from the arena with the specified alignment.
    ///
    /// Returns `None` if the arena is exhausted.
    #[inline]
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let ptr = self.ptr.as_ptr() as usize;
        let aligned = (ptr + align - 1) & !(align - 1);
        let new_ptr = aligned + size;

        if new_ptr > self.end.as_ptr() as usize {
            return None;
        }

        self.ptr = unsafe { NonNull::new_unchecked(new_ptr as *mut u8) };
        self.alloc_count += 1;

        Some(unsafe { NonNull::new_unchecked(aligned as *mut u8) })
    }

    /// Allocate a slice of `T` from the arena.
    ///
    /// The memory is uninitialized - caller must initialize before reading.
    #[inline]
    pub fn alloc_slice<T>(&mut self, len: usize) -> Option<NonNull<T>> {
        let size = std::mem::size_of::<T>() * len;
        let align = std::mem::align_of::<T>();
        self.alloc(size, align).map(|p| p.cast())
    }

    /// Allocate a slice of `T` from the arena, zeroed.
    #[inline]
    pub fn alloc_slice_zeroed<T: Copy>(&mut self, len: usize) -> Option<NonNull<T>> {
        let ptr = self.alloc_slice::<T>(len)?;
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, len);
        }
        Some(ptr)
    }

    /// Allocate a slice and initialize with a value.
    #[inline]
    pub fn alloc_slice_with<T: Clone>(&mut self, len: usize, value: T) -> Option<NonNull<T>> {
        let ptr = self.alloc_slice::<T>(len)?;
        unsafe {
            for i in 0..len {
                std::ptr::write(ptr.as_ptr().add(i), value.clone());
            }
        }
        Some(ptr)
    }

    /// Reset the arena, freeing all allocations.
    ///
    /// All pointers into this arena become invalid after this call.
    #[inline]
    pub fn reset(&mut self) {
        self.ptr = self.base;
        self.alloc_count = 0;
    }

    /// Get the number of bytes used.
    #[inline]
    pub fn used(&self) -> usize {
        self.ptr.as_ptr() as usize - self.base.as_ptr() as usize
    }

    /// Get the number of bytes available.
    #[inline]
    pub fn available(&self) -> usize {
        self.capacity - self.used()
    }

    /// Get the total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the number of allocations.
    #[inline]
    pub fn alloc_count(&self) -> usize {
        self.alloc_count
    }

    /// Check if the arena is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.alloc_count == 0
    }
}

impl Drop for TensorArena {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, 64).expect("Invalid layout");
        unsafe {
            std::alloc::dealloc(self.base.as_ptr(), layout);
        }
    }
}

// Safety: TensorArena can be sent between threads
unsafe impl Send for TensorArena {}

// ============================================================================
// Arena-Scoped Tensor Types
// ============================================================================

/// An arena-allocated tensor that cannot escape its arena scope.
///
/// The lifetime `'a` ties this tensor to its arena, preventing
/// use-after-free bugs at compile time.
pub struct ArenaTensor<'a, T> {
    /// Pointer to tensor data
    data: NonNull<T>,
    /// Shape dimensions
    shape: Vec<usize>,
    /// Strides for each dimension
    strides: Vec<usize>,
    /// Number of elements
    len: usize,
    /// Phantom to tie to arena lifetime
    _marker: PhantomData<&'a T>,
}

impl<'a, T> ArenaTensor<'a, T> {
    /// Get the shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw data pointer.
    ///
    /// # Safety
    ///
    /// The pointer is only valid for the lifetime `'a`.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable raw data pointer.
    ///
    /// # Safety
    ///
    /// The pointer is only valid for the lifetime `'a`.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_ptr()
    }

    /// Get element at flat index.
    #[inline]
    pub fn get_flat(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        unsafe { Some(&*self.data.as_ptr().add(index)) }
    }

    /// Get mutable element at flat index.
    #[inline]
    pub fn get_flat_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        unsafe { Some(&mut *self.data.as_ptr().add(index)) }
    }

    /// Get element at multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut flat_idx = 0;
        for (i, (&idx, &stride)) in indices.iter().zip(self.strides.iter()).enumerate() {
            if idx >= self.shape[i] {
                return None;
            }
            flat_idx += idx * stride;
        }
        unsafe { Some(&*self.data.as_ptr().add(flat_idx)) }
    }

    /// Get mutable element at multi-dimensional index.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut flat_idx = 0;
        for (i, (&idx, &stride)) in indices.iter().zip(self.strides.iter()).enumerate() {
            if idx >= self.shape[i] {
                return None;
            }
            flat_idx += idx * stride;
        }
        unsafe { Some(&mut *self.data.as_ptr().add(flat_idx)) }
    }
}

// ============================================================================
// Arena Operations for f32
// ============================================================================

/// Allocate a zeroed f32 tensor in the arena.
pub fn arena_zeros_f32<'a>(
    arena: &'a mut TensorArena,
    shape: &[usize],
) -> Option<ArenaTensor<'a, f32>> {
    let len: usize = shape.iter().product();
    if len == 0 {
        return None;
    }

    let data = arena.alloc_slice_zeroed::<f32>(len)?;
    let strides = compute_strides(shape);

    Some(ArenaTensor {
        data,
        shape: shape.to_vec(),
        strides,
        len,
        _marker: PhantomData,
    })
}

/// Allocate an f32 tensor filled with a value in the arena.
pub fn arena_full_f32<'a>(
    arena: &'a mut TensorArena,
    shape: &[usize],
    value: f32,
) -> Option<ArenaTensor<'a, f32>> {
    let len: usize = shape.iter().product();
    if len == 0 {
        return None;
    }

    let data = arena.alloc_slice_with::<f32>(len, value)?;
    let strides = compute_strides(shape);

    Some(ArenaTensor {
        data,
        shape: shape.to_vec(),
        strides,
        len,
        _marker: PhantomData,
    })
}

/// Allocate an f32 tensor of ones in the arena.
pub fn arena_ones_f32<'a>(
    arena: &'a mut TensorArena,
    shape: &[usize],
) -> Option<ArenaTensor<'a, f32>> {
    arena_full_f32(arena, shape, 1.0)
}

/// Clone a tensor slice into the arena.
pub fn arena_clone_f32<'a>(
    arena: &'a mut TensorArena,
    data: &[f32],
    shape: &[usize],
) -> Option<ArenaTensor<'a, f32>> {
    let len: usize = shape.iter().product();
    if len != data.len() || len == 0 {
        return None;
    }

    let ptr = arena.alloc_slice::<f32>(len)?;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), len);
    }

    let strides = compute_strides(shape);

    Some(ArenaTensor {
        data: ptr,
        shape: shape.to_vec(),
        strides,
        len,
        _marker: PhantomData,
    })
}

// ============================================================================
// Arena Operations for f64
// ============================================================================

/// Allocate a zeroed f64 tensor in the arena.
pub fn arena_zeros_f64<'a>(
    arena: &'a mut TensorArena,
    shape: &[usize],
) -> Option<ArenaTensor<'a, f64>> {
    let len: usize = shape.iter().product();
    if len == 0 {
        return None;
    }

    let data = arena.alloc_slice_zeroed::<f64>(len)?;
    let strides = compute_strides(shape);

    Some(ArenaTensor {
        data,
        shape: shape.to_vec(),
        strides,
        len,
        _marker: PhantomData,
    })
}

/// Allocate an f64 tensor filled with a value in the arena.
pub fn arena_full_f64<'a>(
    arena: &'a mut TensorArena,
    shape: &[usize],
    value: f64,
) -> Option<ArenaTensor<'a, f64>> {
    let len: usize = shape.iter().product();
    if len == 0 {
        return None;
    }

    let data = arena.alloc_slice_with::<f64>(len, value)?;
    let strides = compute_strides(shape);

    Some(ArenaTensor {
        data,
        shape: shape.to_vec(),
        strides,
        len,
        _marker: PhantomData,
    })
}

/// Allocate an f64 tensor of ones in the arena.
pub fn arena_ones_f64<'a>(
    arena: &'a mut TensorArena,
    shape: &[usize],
) -> Option<ArenaTensor<'a, f64>> {
    arena_full_f64(arena, shape, 1.0)
}

/// Clone a tensor slice into the arena.
pub fn arena_clone_f64<'a>(
    arena: &'a mut TensorArena,
    data: &[f64],
    shape: &[usize],
) -> Option<ArenaTensor<'a, f64>> {
    let len: usize = shape.iter().product();
    if len != data.len() || len == 0 {
        return None;
    }

    let ptr = arena.alloc_slice::<f64>(len)?;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), len);
    }

    let strides = compute_strides(shape);

    Some(ArenaTensor {
        data: ptr,
        shape: shape.to_vec(),
        strides,
        len,
        _marker: PhantomData,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute row-major strides for a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Execute a function with a tensor arena, automatically freeing
/// all arena allocations when the function returns.
///
/// This is the recommended way to use arena allocation.
///
/// # Arguments
///
/// * `capacity` - Arena size in bytes
/// * `f` - Function to execute with the arena
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::arena::with_tensor_arena;
///
/// let result = with_tensor_arena(1024 * 1024, |arena| {
///     let tmp = arena_zeros_f64(arena, &[1024]);
///     // ... use tmp ...
///     42
/// });
/// // All arena allocations are freed here
/// ```
pub fn with_tensor_arena<F, R>(capacity: usize, f: F) -> R
where
    F: FnOnce(&mut TensorArena) -> R,
{
    let mut arena = TensorArena::new(capacity);
    f(&mut arena)
}

// ============================================================================
// FFI Exports
// ============================================================================

/// Opaque arena handle for FFI.
pub type ArenaHandle = *mut TensorArena;

/// Opaque tensor handle for FFI.
#[repr(C)]
pub struct ArenaTensorHandle {
    data: *mut u8,
    shape: *const usize,
    shape_len: usize,
    elem_size: usize,
    len: usize,
}

/// Create a new tensor arena (FFI).
///
/// Returns a handle that must be freed with `bhc_arena_free`.
#[no_mangle]
pub extern "C" fn bhc_arena_new(capacity: usize) -> ArenaHandle {
    if capacity == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(TensorArena::new(capacity)))
}

/// Free a tensor arena (FFI).
///
/// All tensors allocated from this arena become invalid.
#[no_mangle]
pub extern "C" fn bhc_arena_free(arena: ArenaHandle) {
    if !arena.is_null() {
        unsafe {
            drop(Box::from_raw(arena));
        }
    }
}

/// Reset a tensor arena (FFI).
///
/// All tensors allocated from this arena become invalid.
#[no_mangle]
pub extern "C" fn bhc_arena_reset(arena: ArenaHandle) {
    if !arena.is_null() {
        unsafe {
            (*arena).reset();
        }
    }
}

/// Get arena memory usage info (FFI).
#[no_mangle]
pub extern "C" fn bhc_arena_used(arena: ArenaHandle) -> usize {
    if arena.is_null() {
        return 0;
    }
    unsafe { (*arena).used() }
}

/// Get arena available bytes (FFI).
#[no_mangle]
pub extern "C" fn bhc_arena_available(arena: ArenaHandle) -> usize {
    if arena.is_null() {
        return 0;
    }
    unsafe { (*arena).available() }
}

/// Allocate a zeroed f64 tensor in the arena (FFI).
///
/// Returns null if allocation fails.
#[no_mangle]
pub extern "C" fn bhc_arena_zeros_f64(
    arena: ArenaHandle,
    shape: *const usize,
    shape_len: usize,
) -> *mut f64 {
    if arena.is_null() || shape.is_null() || shape_len == 0 {
        return std::ptr::null_mut();
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };
    let len: usize = shape_slice.iter().product();

    if len == 0 {
        return std::ptr::null_mut();
    }

    let arena = unsafe { &mut *arena };
    match arena.alloc_slice_zeroed::<f64>(len) {
        Some(ptr) => ptr.as_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Allocate a zeroed f32 tensor in the arena (FFI).
///
/// Returns null if allocation fails.
#[no_mangle]
pub extern "C" fn bhc_arena_zeros_f32(
    arena: ArenaHandle,
    shape: *const usize,
    shape_len: usize,
) -> *mut f32 {
    if arena.is_null() || shape.is_null() || shape_len == 0 {
        return std::ptr::null_mut();
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };
    let len: usize = shape_slice.iter().product();

    if len == 0 {
        return std::ptr::null_mut();
    }

    let arena = unsafe { &mut *arena };
    match arena.alloc_slice_zeroed::<f32>(len) {
        Some(ptr) => ptr.as_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Allocate an f64 tensor filled with a value in the arena (FFI).
#[no_mangle]
pub extern "C" fn bhc_arena_full_f64(
    arena: ArenaHandle,
    shape: *const usize,
    shape_len: usize,
    value: f64,
) -> *mut f64 {
    if arena.is_null() || shape.is_null() || shape_len == 0 {
        return std::ptr::null_mut();
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };
    let len: usize = shape_slice.iter().product();

    if len == 0 {
        return std::ptr::null_mut();
    }

    let arena = unsafe { &mut *arena };
    match arena.alloc_slice_with::<f64>(len, value) {
        Some(ptr) => ptr.as_ptr(),
        None => std::ptr::null_mut(),
    }
}

/// Clone data into an arena-allocated f64 tensor (FFI).
#[no_mangle]
pub extern "C" fn bhc_arena_clone_f64(
    arena: ArenaHandle,
    data: *const f64,
    len: usize,
) -> *mut f64 {
    if arena.is_null() || data.is_null() || len == 0 {
        return std::ptr::null_mut();
    }

    let arena = unsafe { &mut *arena };
    match arena.alloc_slice::<f64>(len) {
        Some(ptr) => {
            unsafe {
                std::ptr::copy_nonoverlapping(data, ptr.as_ptr(), len);
            }
            ptr.as_ptr()
        }
        None => std::ptr::null_mut(),
    }
}

/// Allocate raw bytes in the arena (FFI).
///
/// Useful for custom data structures.
#[no_mangle]
pub extern "C" fn bhc_arena_alloc(arena: ArenaHandle, size: usize, align: usize) -> *mut u8 {
    if arena.is_null() || size == 0 {
        return std::ptr::null_mut();
    }

    let arena = unsafe { &mut *arena };
    match arena.alloc(size, align) {
        Some(ptr) => ptr.as_ptr(),
        None => std::ptr::null_mut(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = TensorArena::new(1024);
        assert_eq!(arena.capacity(), 1024);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.available(), 1024);
        assert!(arena.is_empty());
    }

    #[test]
    fn test_arena_alloc_basic() {
        let mut arena = TensorArena::new(1024);

        let ptr1 = arena.alloc(64, 8).expect("allocation should succeed");
        assert!(!ptr1.as_ptr().is_null());
        assert_eq!(arena.alloc_count(), 1);

        let ptr2 = arena.alloc(128, 16).expect("allocation should succeed");
        assert!(!ptr2.as_ptr().is_null());
        assert_eq!(arena.alloc_count(), 2);
    }

    #[test]
    fn test_arena_alloc_exhaustion() {
        let mut arena = TensorArena::new(128);

        // First allocation should succeed
        let _ptr1 = arena.alloc(64, 8).expect("should succeed");

        // Second large allocation should fail
        let ptr2 = arena.alloc(100, 8);
        assert!(ptr2.is_none());
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = TensorArena::new(1024);

        arena.alloc(256, 8).expect("should succeed");
        arena.alloc(256, 8).expect("should succeed");

        assert!(arena.used() > 0);
        assert_eq!(arena.alloc_count(), 2);

        arena.reset();

        assert_eq!(arena.used(), 0);
        assert_eq!(arena.alloc_count(), 0);
        assert!(arena.is_empty());
    }

    #[test]
    fn test_arena_zeros_f64() {
        let mut arena = TensorArena::new(1024 * 1024);

        let tensor = arena_zeros_f64(&mut arena, &[10, 20]).expect("should allocate");
        assert_eq!(tensor.shape(), &[10, 20]);
        assert_eq!(tensor.len(), 200);

        // Check all zeros
        for i in 0..tensor.len() {
            assert_eq!(*tensor.get_flat(i).unwrap(), 0.0);
        }
    }

    #[test]
    fn test_arena_ones_f64() {
        let mut arena = TensorArena::new(1024 * 1024);

        let tensor = arena_ones_f64(&mut arena, &[5, 5]).expect("should allocate");
        assert_eq!(tensor.shape(), &[5, 5]);
        assert_eq!(tensor.len(), 25);

        // Check all ones
        for i in 0..tensor.len() {
            assert_eq!(*tensor.get_flat(i).unwrap(), 1.0);
        }
    }

    #[test]
    fn test_arena_clone_f64() {
        let mut arena = TensorArena::new(1024 * 1024);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = arena_clone_f64(&mut arena, &data, &[2, 3]).expect("should allocate");

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(*tensor.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(*tensor.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_arena_tensor_get_mut() {
        let mut arena = TensorArena::new(1024 * 1024);

        let mut tensor = arena_zeros_f64(&mut arena, &[3, 3]).expect("should allocate");

        *tensor.get_mut(&[1, 1]).unwrap() = 42.0;
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 42.0);
    }

    #[test]
    fn test_with_tensor_arena() {
        let result = with_tensor_arena(1024 * 1024, |arena| {
            // First allocation
            let len1 = {
                let t1 = arena_zeros_f64(arena, &[100]).expect("should allocate");
                t1.len()
            };

            // Second allocation (after first borrow ends)
            let len2 = {
                let t2 = arena_ones_f64(arena, &[100]).expect("should allocate");
                t2.len()
            };

            len1 + len2
        });

        assert_eq!(result, 200);
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_ffi_arena() {
        let arena = bhc_arena_new(1024 * 1024);
        assert!(!arena.is_null());

        let shape = [10usize, 20];
        let ptr = bhc_arena_zeros_f64(arena, shape.as_ptr(), shape.len());
        assert!(!ptr.is_null());

        assert!(bhc_arena_used(arena) > 0);

        bhc_arena_reset(arena);
        assert_eq!(bhc_arena_used(arena), 0);

        bhc_arena_free(arena);
    }

    #[test]
    fn test_f32_operations() {
        let mut arena = TensorArena::new(1024 * 1024);

        let zeros = arena_zeros_f32(&mut arena, &[10, 10]).expect("should allocate");
        assert_eq!(zeros.len(), 100);

        let ones = arena_ones_f32(&mut arena, &[5, 5]).expect("should allocate");
        assert_eq!(*ones.get_flat(0).unwrap(), 1.0f32);
    }
}
