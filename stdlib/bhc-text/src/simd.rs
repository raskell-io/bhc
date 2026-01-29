//! SIMD-accelerated text operations
//!
//! Platform-specific SIMD implementations for text processing.
//! These are low-level primitives called from BHC-compiled Haskell.

// ============================================================
// UTF-8 Validation
// ============================================================

/// Validate UTF-8 encoding.
///
/// Returns the index of the first invalid byte, or -1 if valid.
#[no_mangle]
pub extern "C" fn bhc_utf8_validate(ptr: *const u8, len: usize) -> i64 {
    if ptr.is_null() || len == 0 {
        return -1; // Empty is valid
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    match std::str::from_utf8(bytes) {
        Ok(_) => -1,
        Err(e) => e.valid_up_to() as i64,
    }
}

// ============================================================
// Byte Search (memchr-style)
// ============================================================

/// Find first occurrence of a byte.
///
/// Returns the index or -1 if not found.
#[no_mangle]
pub extern "C" fn bhc_memchr(ptr: *const u8, len: usize, byte: u8) -> i64 {
    if ptr.is_null() || len == 0 {
        return -1;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    // TODO: Use actual SIMD (memchr crate) for large inputs
    bytes
        .iter()
        .position(|&b| b == byte)
        .map_or(-1, |i| i as i64)
}

/// Find last occurrence of a byte.
///
/// Returns the index or -1 if not found.
#[no_mangle]
pub extern "C" fn bhc_memrchr(ptr: *const u8, len: usize, byte: u8) -> i64 {
    if ptr.is_null() || len == 0 {
        return -1;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    bytes
        .iter()
        .rposition(|&b| b == byte)
        .map_or(-1, |i| i as i64)
}

/// Count occurrences of a byte.
#[no_mangle]
pub extern "C" fn bhc_memcnt(ptr: *const u8, len: usize, byte: u8) -> usize {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    bytes.iter().filter(|&&b| b == byte).count()
}

// ============================================================
// Substring Search
// ============================================================

/// Find first occurrence of a pattern.
///
/// Returns the byte index or -1 if not found.
#[no_mangle]
pub extern "C" fn bhc_text_find_substring(
    haystack: *const u8,
    haystack_len: usize,
    needle: *const u8,
    needle_len: usize,
) -> i64 {
    if haystack.is_null() || needle.is_null() {
        return -1;
    }
    if needle_len == 0 {
        return 0;
    }
    if needle_len > haystack_len {
        return -1;
    }
    let h = unsafe { std::slice::from_raw_parts(haystack, haystack_len) };
    let n = unsafe { std::slice::from_raw_parts(needle, needle_len) };
    // TODO: Use SIMD-accelerated search (e.g., Rabin-Karp with SIMD)
    for i in 0..=(haystack_len - needle_len) {
        if &h[i..i + needle_len] == n {
            return i as i64;
        }
    }
    -1
}

// ============================================================
// ASCII Operations (SIMD-friendly)
// ============================================================

/// Check if all bytes are ASCII (< 128).
#[no_mangle]
pub extern "C" fn bhc_is_ascii(ptr: *const u8, len: usize) -> bool {
    if ptr.is_null() || len == 0 {
        return true;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    // This compiles to efficient SIMD on modern compilers
    bytes.iter().all(|&b| b < 128)
}

/// Convert ASCII bytes to uppercase in-place.
///
/// Only converts ASCII letters (a-z), leaves other bytes unchanged.
#[no_mangle]
pub extern "C" fn bhc_ascii_to_upper(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    for b in bytes.iter_mut() {
        if *b >= b'a' && *b <= b'z' {
            *b -= 32;
        }
    }
}

/// Convert ASCII bytes to lowercase in-place.
///
/// Only converts ASCII letters (A-Z), leaves other bytes unchanged.
#[no_mangle]
pub extern "C" fn bhc_ascii_to_lower(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    for b in bytes.iter_mut() {
        if *b >= b'A' && *b <= b'Z' {
            *b += 32;
        }
    }
}

// ============================================================
// Memory Operations
// ============================================================

/// Compare two byte sequences.
///
/// Returns -1 (less), 0 (equal), or 1 (greater).
#[no_mangle]
pub extern "C" fn bhc_memcmp(a: *const u8, a_len: usize, b: *const u8, b_len: usize) -> i32 {
    use std::cmp::Ordering;
    let a_slice = if a.is_null() || a_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(a, a_len) }
    };
    let b_slice = if b.is_null() || b_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(b, b_len) }
    };
    match a_slice.cmp(b_slice) {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

/// Copy bytes from source to destination.
#[no_mangle]
pub extern "C" fn bhc_memcpy(dst: *mut u8, src: *const u8, len: usize) {
    if dst.is_null() || src.is_null() || len == 0 {
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf8_validate() {
        let valid = b"hello";
        assert_eq!(bhc_utf8_validate(valid.as_ptr(), valid.len()), -1);

        let invalid = [0xff, 0xfe];
        assert_eq!(bhc_utf8_validate(invalid.as_ptr(), invalid.len()), 0);
    }

    #[test]
    fn test_memchr() {
        let data = b"hello";
        assert_eq!(bhc_memchr(data.as_ptr(), data.len(), b'l'), 2);
        assert_eq!(bhc_memchr(data.as_ptr(), data.len(), b'z'), -1);
    }

    #[test]
    fn test_memrchr() {
        let data = b"hello";
        assert_eq!(bhc_memrchr(data.as_ptr(), data.len(), b'l'), 3);
    }

    #[test]
    fn test_is_ascii() {
        assert!(bhc_is_ascii(b"hello".as_ptr(), 5));
        assert!(!bhc_is_ascii([0x80].as_ptr(), 1));
    }

    #[test]
    fn test_ascii_case() {
        let mut data = *b"Hello World";
        bhc_ascii_to_upper(data.as_mut_ptr(), data.len());
        assert_eq!(&data, b"HELLO WORLD");

        bhc_ascii_to_lower(data.as_mut_ptr(), data.len());
        assert_eq!(&data, b"hello world");
    }

    #[test]
    fn test_find_substring() {
        let haystack = b"hello world";
        let needle = b"world";
        assert_eq!(
            bhc_text_find_substring(
                haystack.as_ptr(),
                haystack.len(),
                needle.as_ptr(),
                needle.len()
            ),
            6
        );

        let missing = b"xyz";
        assert_eq!(
            bhc_text_find_substring(
                haystack.as_ptr(),
                haystack.len(),
                missing.as_ptr(),
                missing.len()
            ),
            -1
        );
    }
}
