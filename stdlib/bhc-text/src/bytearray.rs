//! ByteArray FFI primitives for BHC ByteString.
//!
//! These functions implement the `extern "C"` FFI interface that
//! `hs/BHC/Data/ByteString.hs` imports. ByteArrays are heap-allocated
//! byte buffers returned as opaque `*mut u8` pointers to LLVM.

use std::alloc::{self, Layout};
use std::ptr;

/// Opaque ByteArray: a heap-allocated byte buffer.
///
/// Represented as a fat pointer struct so we can recover the length.
/// Layout: [u64 capacity][u64 length][...bytes...]
///
/// The pointer returned to the RTS points to the header (capacity field).
const HEADER_SIZE: usize = 16; // 2 × u64

// ============================================================
// Allocation
// ============================================================

/// Allocate a new ByteArray of the given capacity.
///
/// Returns a pointer to the header. The data region starts at
/// `ptr + HEADER_SIZE`.
///
/// # Safety
///
/// Caller must eventually free the returned pointer via the allocator.
#[no_mangle]
pub extern "C" fn bhc_bytearray_malloc(capacity: i64) -> *mut u8 {
    let cap = capacity.max(0) as usize;
    let total = HEADER_SIZE + cap;
    let layout = Layout::from_size_align(total, 8).expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            return ptr::null_mut();
        }
        // Store capacity
        (ptr as *mut u64).write(cap as u64);
        // Store length = 0
        (ptr as *mut u64).add(1).write(0);
        ptr
    }
}

// ============================================================
// Accessors
// ============================================================

/// Return a pointer to the raw data region of a ByteArray.
///
/// Given a ByteArray header pointer, returns `header + HEADER_SIZE`.
#[no_mangle]
pub extern "C" fn bhc_bytearray_contents(array: *mut u8) -> *mut u8 {
    if array.is_null() {
        return ptr::null_mut();
    }
    unsafe { array.add(HEADER_SIZE) }
}

/// Index into a ByteArray, returning the byte at the given offset.
///
/// The `array` parameter is the header pointer. The `index` is a
/// byte offset into the data region.
#[no_mangle]
pub extern "C" fn bhc_bytearray_index(array: *mut u8, index: i64) -> i64 {
    if array.is_null() || index < 0 {
        return 0;
    }
    unsafe {
        let data = array.add(HEADER_SIZE);
        *data.add(index as usize) as i64
    }
}

// ============================================================
// Copy
// ============================================================

/// Copy bytes from a ByteArray into a destination buffer.
///
/// `dst` is a raw pointer (typically from `bhc_bytearray_contents`).
/// `src` is a ByteArray header pointer.
/// `offset` is the byte offset into the source data region.
/// `len` is the number of bytes to copy.
#[no_mangle]
pub extern "C" fn bhc_bytearray_copy(dst: *mut u8, src: *mut u8, offset: i64, len: i64) {
    if dst.is_null() || src.is_null() || len <= 0 {
        return;
    }
    let off = offset.max(0) as usize;
    let n = len as usize;
    unsafe {
        let src_data = src.add(HEADER_SIZE).add(off);
        ptr::copy_nonoverlapping(src_data, dst, n);
    }
}

// ============================================================
// Pointer arithmetic
// ============================================================

/// Advance a pointer by `offset` bytes.
#[no_mangle]
pub extern "C" fn bhc_ptr_plus(ptr: *mut u8, offset: i64) -> *mut u8 {
    if ptr.is_null() {
        return ptr::null_mut();
    }
    unsafe { ptr.offset(offset as isize) }
}

// ============================================================
// Byte poke
// ============================================================

/// Write a single byte at the given offset from a base pointer.
#[no_mangle]
pub extern "C" fn bhc_poke_byte(ptr: *mut u8, offset: i64, value: i64) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        *ptr.offset(offset as isize) = value as u8;
    }
}

// ============================================================
// C string length
// ============================================================

/// Return the length of a null-terminated C string (like `strlen`).
#[no_mangle]
pub extern "C" fn bhc_cstring_length(ptr: *mut u8) -> i64 {
    if ptr.is_null() {
        return 0;
    }
    let mut len: i64 = 0;
    unsafe {
        while *ptr.offset(len as isize) != 0 {
            len += 1;
        }
    }
    len
}

// ============================================================
// Peek array
// ============================================================

/// Read `count` bytes starting at `ptr` and return a new ByteArray
/// containing them.
///
/// This is used by `packCStringLen` in ByteString.hs.
#[no_mangle]
pub extern "C" fn bhc_peek_array(count: i64, ptr: *mut u8) -> *mut u8 {
    if ptr.is_null() || count <= 0 {
        return bhc_bytearray_malloc(0);
    }
    let n = count as usize;
    let array = bhc_bytearray_malloc(count);
    if array.is_null() {
        return array;
    }
    unsafe {
        let dst = array.add(HEADER_SIZE);
        ptr::copy_nonoverlapping(ptr, dst, n);
        // Update stored length
        (array as *mut u64).add(1).write(n as u64);
    }
    array
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malloc_and_contents() {
        let arr = bhc_bytearray_malloc(16);
        assert!(!arr.is_null());
        let data = bhc_bytearray_contents(arr);
        assert!(!data.is_null());
        assert_eq!(data, unsafe { arr.add(HEADER_SIZE) });
    }

    #[test]
    fn test_poke_and_index() {
        let arr = bhc_bytearray_malloc(8);
        let data = bhc_bytearray_contents(arr);
        bhc_poke_byte(data, 0, 42);
        bhc_poke_byte(data, 1, 99);
        assert_eq!(bhc_bytearray_index(arr, 0), 42);
        assert_eq!(bhc_bytearray_index(arr, 1), 99);
    }

    #[test]
    fn test_copy() {
        let src = bhc_bytearray_malloc(4);
        let src_data = bhc_bytearray_contents(src);
        bhc_poke_byte(src_data, 0, 10);
        bhc_poke_byte(src_data, 1, 20);
        bhc_poke_byte(src_data, 2, 30);

        let dst = bhc_bytearray_malloc(4);
        let dst_data = bhc_bytearray_contents(dst);
        bhc_bytearray_copy(dst_data, src, 0, 3);

        assert_eq!(bhc_bytearray_index(dst, 0), 10);
        assert_eq!(bhc_bytearray_index(dst, 1), 20);
        assert_eq!(bhc_bytearray_index(dst, 2), 30);
    }

    #[test]
    fn test_ptr_plus() {
        let arr = bhc_bytearray_malloc(8);
        let data = bhc_bytearray_contents(arr);
        bhc_poke_byte(data, 0, 1);
        bhc_poke_byte(data, 4, 5);
        let offset_ptr = bhc_ptr_plus(data, 4);
        unsafe {
            assert_eq!(*offset_ptr, 5);
        }
    }

    #[test]
    fn test_cstring_length() {
        let arr = bhc_bytearray_malloc(8);
        let data = bhc_bytearray_contents(arr);
        bhc_poke_byte(data, 0, b'H' as i64);
        bhc_poke_byte(data, 1, b'i' as i64);
        bhc_poke_byte(data, 2, 0); // null terminator
        assert_eq!(bhc_cstring_length(data), 2);
    }

    #[test]
    fn test_peek_array() {
        let src = bhc_bytearray_malloc(4);
        let src_data = bhc_bytearray_contents(src);
        bhc_poke_byte(src_data, 0, 65);
        bhc_poke_byte(src_data, 1, 66);
        bhc_poke_byte(src_data, 2, 67);

        let result = bhc_peek_array(3, src_data);
        assert!(!result.is_null());
        assert_eq!(bhc_bytearray_index(result, 0), 65);
        assert_eq!(bhc_bytearray_index(result, 1), 66);
        assert_eq!(bhc_bytearray_index(result, 2), 67);
    }

    #[test]
    fn test_null_safety() {
        assert!(bhc_bytearray_contents(ptr::null_mut()).is_null());
        assert_eq!(bhc_bytearray_index(ptr::null_mut(), 0), 0);
        assert!(bhc_ptr_plus(ptr::null_mut(), 5).is_null());
        assert_eq!(bhc_cstring_length(ptr::null_mut()), 0);
        // These should not crash:
        bhc_bytearray_copy(ptr::null_mut(), ptr::null_mut(), 0, 0);
        bhc_poke_byte(ptr::null_mut(), 0, 0);
    }
}
