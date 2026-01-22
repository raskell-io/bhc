//! ByteString type
//!
//! Raw byte arrays with pinned memory for FFI safety.

/// ByteString - immutable byte array
#[repr(C)]
pub struct ByteString {
    ptr: *const u8,
    len: usize,
}

impl ByteString {
    /// Create empty bytestring
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    /// Length in bytes
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Get bytestring length
#[no_mangle]
pub extern "C" fn bhc_bytestring_length(ptr: *const u8, len: usize) -> usize {
    len
}

/// Check if bytestring is empty
#[no_mangle]
pub extern "C" fn bhc_bytestring_null(len: usize) -> bhc_prelude::bool::Bool {
    bhc_prelude::bool::Bool::from_bool(len == 0)
}

/// Copy bytestring
#[no_mangle]
pub extern "C" fn bhc_bytestring_copy(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
) {
    if src.is_null() || dst.is_null() || src_len == 0 {
        return;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, src_len);
    }
}

/// Concatenate two bytestrings
#[no_mangle]
pub extern "C" fn bhc_bytestring_append(
    a_ptr: *const u8,
    a_len: usize,
    b_ptr: *const u8,
    b_len: usize,
    out: *mut u8,
) -> usize {
    if out.is_null() {
        return 0;
    }

    let mut offset = 0;

    if !a_ptr.is_null() && a_len > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(a_ptr, out, a_len);
        }
        offset = a_len;
    }

    if !b_ptr.is_null() && b_len > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(b_ptr, out.add(offset), b_len);
        }
    }

    a_len + b_len
}

/// Find byte in bytestring
#[no_mangle]
pub extern "C" fn bhc_bytestring_elem(ptr: *const u8, len: usize, byte: u8) -> i64 {
    if ptr.is_null() || len == 0 {
        return -1;
    }

    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };

    for (i, &b) in bytes.iter().enumerate() {
        if b == byte {
            return i as i64;
        }
    }

    -1
}

/// Compare two bytestrings
#[no_mangle]
pub extern "C" fn bhc_bytestring_compare(
    a_ptr: *const u8,
    a_len: usize,
    b_ptr: *const u8,
    b_len: usize,
) -> bhc_prelude::ordering::Ordering {
    use std::cmp::Ordering as StdOrdering;

    let a = if a_ptr.is_null() || a_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(a_ptr, a_len) }
    };

    let b = if b_ptr.is_null() || b_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(b_ptr, b_len) }
    };

    match a.cmp(b) {
        StdOrdering::Less => bhc_prelude::ordering::Ordering::LT,
        StdOrdering::Equal => bhc_prelude::ordering::Ordering::EQ,
        StdOrdering::Greater => bhc_prelude::ordering::Ordering::GT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elem() {
        let bytes = [1u8, 2, 3, 4, 5];
        assert_eq!(bhc_bytestring_elem(bytes.as_ptr(), bytes.len(), 3), 2);
        assert_eq!(bhc_bytestring_elem(bytes.as_ptr(), bytes.len(), 6), -1);
    }
}
