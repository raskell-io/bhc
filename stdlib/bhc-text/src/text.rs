//! UTF-8 Text type
//!
//! Efficient text representation with SIMD-accelerated operations.

use std::str;

/// UTF-8 encoded text
///
/// Internally stores UTF-8 bytes with length tracking.
#[repr(C)]
pub struct Text {
    ptr: *const u8,
    len: usize,
}

impl Text {
    /// Create empty text
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

/// Get text length in bytes
#[no_mangle]
pub extern "C" fn bhc_text_length(ptr: *const u8, len: usize) -> usize {
    len
}

/// Get text length in characters (code points)
#[no_mangle]
pub extern "C" fn bhc_text_char_count(ptr: *const u8, len: usize) -> usize {
    if ptr.is_null() || len == 0 {
        return 0;
    }

    // SAFETY: Caller guarantees valid UTF-8
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let s = unsafe { str::from_utf8_unchecked(bytes) };
    s.chars().count()
}

/// Check if text is ASCII
#[no_mangle]
pub extern "C" fn bhc_text_is_ascii(ptr: *const u8, len: usize) -> bhc_prelude::bool::Bool {
    if ptr.is_null() || len == 0 {
        return bhc_prelude::bool::Bool::True;
    }

    // SAFETY: Caller guarantees valid pointer
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };

    // SIMD-friendly check: all bytes < 128
    bhc_prelude::bool::Bool::from_bool(bytes.iter().all(|&b| b < 128))
}

/// Convert text to uppercase
///
/// Returns new length (may be different due to Unicode expansion)
#[no_mangle]
pub extern "C" fn bhc_text_to_upper(
    ptr: *const u8,
    len: usize,
    out: *mut u8,
    out_cap: usize,
) -> usize {
    if ptr.is_null() || out.is_null() || len == 0 {
        return 0;
    }

    // SAFETY: Caller guarantees valid UTF-8 and sufficient output capacity
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let s = unsafe { str::from_utf8_unchecked(bytes) };

    let upper = s.to_uppercase();
    let upper_bytes = upper.as_bytes();

    if upper_bytes.len() > out_cap {
        return 0; // Not enough space
    }

    unsafe {
        std::ptr::copy_nonoverlapping(upper_bytes.as_ptr(), out, upper_bytes.len());
    }

    upper_bytes.len()
}

/// Convert text to lowercase
#[no_mangle]
pub extern "C" fn bhc_text_to_lower(
    ptr: *const u8,
    len: usize,
    out: *mut u8,
    out_cap: usize,
) -> usize {
    if ptr.is_null() || out.is_null() || len == 0 {
        return 0;
    }

    // SAFETY: Caller guarantees valid UTF-8 and sufficient output capacity
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let s = unsafe { str::from_utf8_unchecked(bytes) };

    let lower = s.to_lowercase();
    let lower_bytes = lower.as_bytes();

    if lower_bytes.len() > out_cap {
        return 0;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(lower_bytes.as_ptr(), out, lower_bytes.len());
    }

    lower_bytes.len()
}

/// Find substring (returns offset or -1)
#[no_mangle]
pub extern "C" fn bhc_text_find(
    haystack_ptr: *const u8,
    haystack_len: usize,
    needle_ptr: *const u8,
    needle_len: usize,
) -> i64 {
    if haystack_ptr.is_null() || needle_ptr.is_null() {
        return -1;
    }

    if needle_len == 0 {
        return 0;
    }

    if needle_len > haystack_len {
        return -1;
    }

    // SAFETY: Caller guarantees valid pointers
    let haystack = unsafe { std::slice::from_raw_parts(haystack_ptr, haystack_len) };
    let needle = unsafe { std::slice::from_raw_parts(needle_ptr, needle_len) };

    // Simple search (TODO: use SIMD for longer patterns)
    for i in 0..=(haystack_len - needle_len) {
        if &haystack[i..i + needle_len] == needle {
            return i as i64;
        }
    }

    -1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_count() {
        let s = "hello";
        assert_eq!(bhc_text_char_count(s.as_ptr(), s.len()), 5);

        let s = "hello 世界";
        assert_eq!(bhc_text_char_count(s.as_ptr(), s.len()), 8);
    }

    #[test]
    fn test_find() {
        let haystack = "hello world";
        let needle = "world";
        assert_eq!(
            bhc_text_find(haystack.as_ptr(), haystack.len(), needle.as_ptr(), needle.len()),
            6
        );

        let needle = "xyz";
        assert_eq!(
            bhc_text_find(haystack.as_ptr(), haystack.len(), needle.as_ptr(), needle.len()),
            -1
        );
    }
}
