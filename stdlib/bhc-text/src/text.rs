//! Text FFI primitives for BHC Data.Text.
//!
//! These functions implement the `extern "C"` FFI interface that
//! `hs/BHC/Data/Text.hs` imports. BhcText is a heap-allocated UTF-8
//! text value with offset/length slicing, returned as `*mut u8` to LLVM.
//!
//! # Representation
//!
//! A BhcText is laid out as:
//!
//! ```text
//! [u64 data_ptr][u64 offset][u64 byte_len][...UTF-8 bytes...]
//! ```
//!
//! The header is 24 bytes. `data_ptr` points to the start of the
//! byte buffer (which may be shared across slices). For self-contained
//! texts, `data_ptr` points to `header + HEADER_SIZE`.

use std::alloc::{self, Layout};
use std::ptr;
use std::str;

use crate::simd::{bhc_ascii_to_lower, bhc_ascii_to_upper, bhc_is_ascii, bhc_text_find_substring};

/// Header: data_ptr (8) + offset (8) + byte_len (8) = 24 bytes.
const HEADER_SIZE: usize = 24;

// ============================================================
// Internal helpers
// ============================================================

/// Read the data pointer from a BhcText header.
unsafe fn text_data_ptr(text: *const u8) -> *const u8 {
    *(text as *const *const u8)
}

/// Read the byte offset from a BhcText header.
unsafe fn text_offset(text: *const u8) -> usize {
    *((text as *const u64).add(1)) as usize
}

/// Read the byte length from a BhcText header.
unsafe fn text_byte_len(text: *const u8) -> usize {
    *((text as *const u64).add(2)) as usize
}

/// Get a slice view of the text's active bytes.
unsafe fn text_bytes(text: *const u8) -> &'static [u8] {
    let data = text_data_ptr(text);
    let off = text_offset(text);
    let len = text_byte_len(text);
    std::slice::from_raw_parts(data.add(off), len)
}

/// Get the text as a `&str` (assumes valid UTF-8).
unsafe fn text_as_str(text: *const u8) -> &'static str {
    str::from_utf8_unchecked(text_bytes(text))
}

/// Allocate a new self-contained BhcText from a byte slice.
///
/// The returned pointer owns both header and data.
fn alloc_text_from_bytes(bytes: &[u8]) -> *mut u8 {
    let total = HEADER_SIZE + bytes.len();
    let layout = Layout::from_size_align(total, 8).expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc(layout);
        if ptr.is_null() {
            return ptr::null_mut();
        }
        // data_ptr = ptr + HEADER_SIZE
        let data_start = ptr.add(HEADER_SIZE);
        (ptr as *mut *mut u8).write(data_start);
        // offset = 0
        (ptr as *mut u64).add(1).write(0);
        // byte_len = bytes.len()
        (ptr as *mut u64).add(2).write(bytes.len() as u64);
        // Copy data
        if !bytes.is_empty() {
            ptr::copy_nonoverlapping(bytes.as_ptr(), data_start, bytes.len());
        }
        ptr
    }
}

/// Allocate a BhcText that is a slice (view) of another text's data.
///
/// `source_data_ptr` is the data buffer pointer from the source text.
fn alloc_text_slice(source_data_ptr: *const u8, offset: usize, byte_len: usize) -> *mut u8 {
    // Header-only allocation (no data copy)
    let layout = Layout::from_size_align(HEADER_SIZE, 8).expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc(layout);
        if ptr.is_null() {
            return ptr::null_mut();
        }
        (ptr as *mut *const u8).write(source_data_ptr);
        (ptr as *mut u64).add(1).write(offset as u64);
        (ptr as *mut u64).add(2).write(byte_len as u64);
        ptr
    }
}

// ============================================================
// Creation
// ============================================================

/// Return an empty Text.
#[no_mangle]
pub extern "C" fn bhc_text_empty() -> *mut u8 {
    alloc_text_from_bytes(&[])
}

/// Create a Text containing a single Unicode codepoint.
///
/// `codepoint` is passed as an i64 representing a Haskell `Char` (Unicode scalar).
#[no_mangle]
pub extern "C" fn bhc_text_singleton(codepoint: i64) -> *mut u8 {
    let c = char::from_u32(codepoint as u32).unwrap_or('\u{FFFD}');
    let mut buf = [0u8; 4];
    let s = c.encode_utf8(&mut buf);
    alloc_text_from_bytes(s.as_bytes())
}

// ============================================================
// Basic interface
// ============================================================

/// Test whether a Text is empty.
///
/// Returns 1 for true, 0 for false.
#[no_mangle]
pub extern "C" fn bhc_text_null(text: *const u8) -> i64 {
    if text.is_null() {
        return 1;
    }
    unsafe {
        if text_byte_len(text) == 0 {
            1
        } else {
            0
        }
    }
}

/// Return the number of Unicode codepoints in a Text.
#[no_mangle]
pub extern "C" fn bhc_text_length(text: *const u8) -> i64 {
    if text.is_null() {
        return 0;
    }
    unsafe { text_as_str(text).chars().count() as i64 }
}

/// Compare two Texts for equality.
///
/// Returns 1 if equal, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_text_eq(a: *const u8, b: *const u8) -> i64 {
    if a.is_null() && b.is_null() {
        return 1;
    }
    if a.is_null() || b.is_null() {
        return 0;
    }
    unsafe {
        if text_bytes(a) == text_bytes(b) {
            1
        } else {
            0
        }
    }
}

/// Lexicographic comparison of two Texts.
///
/// Returns -1 (LT), 0 (EQ), or 1 (GT). Mapped to Haskell `Ordering`
/// as tag values.
#[no_mangle]
pub extern "C" fn bhc_text_compare(a: *const u8, b: *const u8) -> i64 {
    use std::cmp::Ordering;
    let a_bytes = if a.is_null() {
        &[]
    } else {
        unsafe { text_bytes(a) }
    };
    let b_bytes = if b.is_null() {
        &[]
    } else {
        unsafe { text_bytes(b) }
    };
    match a_bytes.cmp(b_bytes) {
        Ordering::Less => 0,    // LT tag = 0
        Ordering::Equal => 1,   // EQ tag = 1
        Ordering::Greater => 2, // GT tag = 2
    }
}

/// Extract the first codepoint of a Text.
///
/// Returns the codepoint as an i64 (Haskell Char). Panics on empty text.
#[no_mangle]
pub extern "C" fn bhc_text_head(text: *const u8) -> i64 {
    if text.is_null() {
        return 0;
    }
    unsafe {
        let s = text_as_str(text);
        s.chars().next().map_or(0, |c| c as i64)
    }
}

/// Extract the last codepoint of a Text.
#[no_mangle]
pub extern "C" fn bhc_text_last(text: *const u8) -> i64 {
    if text.is_null() {
        return 0;
    }
    unsafe {
        let s = text_as_str(text);
        s.chars().next_back().map_or(0, |c| c as i64)
    }
}

/// Return all characters after the first (advance past first codepoint).
///
/// Returns a new Text that is a slice of the original.
#[no_mangle]
pub extern "C" fn bhc_text_tail(text: *const u8) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    unsafe {
        let s = text_as_str(text);
        if s.is_empty() {
            return bhc_text_empty();
        }
        let first_len = s.chars().next().map_or(0, |c| c.len_utf8());
        let data = text_data_ptr(text);
        let off = text_offset(text) + first_len;
        let len = text_byte_len(text) - first_len;
        alloc_text_slice(data, off, len)
    }
}

/// Return all characters except the last.
#[no_mangle]
pub extern "C" fn bhc_text_init(text: *const u8) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    unsafe {
        let s = text_as_str(text);
        if s.is_empty() {
            return bhc_text_empty();
        }
        let last_len = s.chars().next_back().map_or(0, |c| c.len_utf8());
        let data = text_data_ptr(text);
        let off = text_offset(text);
        let len = text_byte_len(text) - last_len;
        alloc_text_slice(data, off, len)
    }
}

// ============================================================
// Concatenation
// ============================================================

/// Append two Texts.
#[no_mangle]
pub extern "C" fn bhc_text_append(a: *const u8, b: *const u8) -> *mut u8 {
    let a_bytes = if a.is_null() {
        &[]
    } else {
        unsafe { text_bytes(a) }
    };
    let b_bytes = if b.is_null() {
        &[]
    } else {
        unsafe { text_bytes(b) }
    };

    if a_bytes.is_empty() && !b.is_null() {
        // Return b as-is (slice view)
        return unsafe {
            alloc_text_slice(text_data_ptr(b), text_offset(b), text_byte_len(b))
        };
    }
    if b_bytes.is_empty() && !a.is_null() {
        return unsafe {
            alloc_text_slice(text_data_ptr(a), text_offset(a), text_byte_len(a))
        };
    }

    let mut combined = Vec::with_capacity(a_bytes.len() + b_bytes.len());
    combined.extend_from_slice(a_bytes);
    combined.extend_from_slice(b_bytes);
    alloc_text_from_bytes(&combined)
}

// ============================================================
// Transformations
// ============================================================

/// Reverse a Text (by codepoints).
#[no_mangle]
pub extern "C" fn bhc_text_reverse(text: *const u8) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    unsafe {
        let s = text_as_str(text);
        let reversed: String = s.chars().rev().collect();
        alloc_text_from_bytes(reversed.as_bytes())
    }
}

/// Map a function over each codepoint, building a new Text.
///
/// `func_ptr` is an opaque closure pointer. The RTS calls
/// `(func_ptr)(codepoint) -> codepoint` for each character.
///
/// The closure is represented as a pair: (fn_ptr, env_ptr).
/// `fn_ptr` has signature `extern "C" fn(env: *mut u8, char: i64) -> i64`.
#[no_mangle]
pub extern "C" fn bhc_text_map(
    fn_ptr: extern "C" fn(*mut u8, i64) -> i64,
    env_ptr: *mut u8,
    text: *const u8,
) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    unsafe {
        let s = text_as_str(text);
        let mut result = String::with_capacity(s.len());
        for c in s.chars() {
            let mapped = fn_ptr(env_ptr, c as i64);
            let mapped_char = char::from_u32(mapped as u32).unwrap_or('\u{FFFD}');
            result.push(mapped_char);
        }
        alloc_text_from_bytes(result.as_bytes())
    }
}

// ============================================================
// Substrings
// ============================================================

/// Take the first `n` codepoints.
#[no_mangle]
pub extern "C" fn bhc_text_take(n: i64, text: *const u8) -> *mut u8 {
    if text.is_null() || n <= 0 {
        return bhc_text_empty();
    }
    unsafe {
        let s = text_as_str(text);
        let count = n as usize;
        let byte_end: usize = s
            .char_indices()
            .take(count)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        alloc_text_slice(text_data_ptr(text), text_offset(text), byte_end)
    }
}

/// Take the last `n` codepoints.
#[no_mangle]
pub extern "C" fn bhc_text_take_end(n: i64, text: *const u8) -> *mut u8 {
    if text.is_null() || n <= 0 {
        return bhc_text_empty();
    }
    unsafe {
        let s = text_as_str(text);
        let total = s.chars().count();
        let count = n as usize;
        if count >= total {
            return alloc_text_slice(text_data_ptr(text), text_offset(text), text_byte_len(text));
        }
        let skip = total - count;
        let byte_start: usize = s
            .char_indices()
            .nth(skip)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        let off = text_offset(text) + byte_start;
        let len = text_byte_len(text) - byte_start;
        alloc_text_slice(text_data_ptr(text), off, len)
    }
}

/// Drop the first `n` codepoints.
#[no_mangle]
pub extern "C" fn bhc_text_drop(n: i64, text: *const u8) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    if n <= 0 {
        return unsafe {
            alloc_text_slice(text_data_ptr(text), text_offset(text), text_byte_len(text))
        };
    }
    unsafe {
        let s = text_as_str(text);
        let count = n as usize;
        let byte_start: usize = s
            .char_indices()
            .nth(count)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        let off = text_offset(text) + byte_start;
        let len = text_byte_len(text) - byte_start;
        alloc_text_slice(text_data_ptr(text), off, len)
    }
}

/// Drop the last `n` codepoints.
#[no_mangle]
pub extern "C" fn bhc_text_drop_end(n: i64, text: *const u8) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    if n <= 0 {
        return unsafe {
            alloc_text_slice(text_data_ptr(text), text_offset(text), text_byte_len(text))
        };
    }
    unsafe {
        let s = text_as_str(text);
        let total = s.chars().count();
        let count = n as usize;
        if count >= total {
            return bhc_text_empty();
        }
        let keep = total - count;
        let byte_end: usize = s
            .char_indices()
            .take(keep)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        alloc_text_slice(text_data_ptr(text), text_offset(text), byte_end)
    }
}

// ============================================================
// Predicates
// ============================================================

/// Test whether `prefix` is a prefix of `text`.
///
/// Returns 1 for true, 0 for false.
#[no_mangle]
pub extern "C" fn bhc_text_is_prefix_of(prefix: *const u8, text: *const u8) -> i64 {
    let p_bytes = if prefix.is_null() {
        &[]
    } else {
        unsafe { text_bytes(prefix) }
    };
    let t_bytes = if text.is_null() {
        &[]
    } else {
        unsafe { text_bytes(text) }
    };
    if t_bytes.starts_with(p_bytes) {
        1
    } else {
        0
    }
}

/// Test whether `suffix` is a suffix of `text`.
#[no_mangle]
pub extern "C" fn bhc_text_is_suffix_of(suffix: *const u8, text: *const u8) -> i64 {
    let s_bytes = if suffix.is_null() {
        &[]
    } else {
        unsafe { text_bytes(suffix) }
    };
    let t_bytes = if text.is_null() {
        &[]
    } else {
        unsafe { text_bytes(text) }
    };
    if t_bytes.ends_with(s_bytes) {
        1
    } else {
        0
    }
}

/// Test whether `needle` occurs anywhere in `haystack`.
#[no_mangle]
pub extern "C" fn bhc_text_is_infix_of(needle: *const u8, haystack: *const u8) -> i64 {
    let n_bytes = if needle.is_null() {
        return 1; // empty is infix of anything
    } else {
        unsafe { text_bytes(needle) }
    };
    let h_bytes = if haystack.is_null() {
        &[]
    } else {
        unsafe { text_bytes(haystack) }
    };
    if n_bytes.is_empty() {
        return 1;
    }
    let result = bhc_text_find_substring(
        h_bytes.as_ptr(),
        h_bytes.len(),
        n_bytes.as_ptr(),
        n_bytes.len(),
    );
    if result >= 0 { 1 } else { 0 }
}

// ============================================================
// Case conversion
// ============================================================

/// Convert text to lowercase.
///
/// Uses ASCII fast path from simd.rs when all bytes are ASCII,
/// falls back to full Unicode via Rust's `to_lowercase()`.
#[no_mangle]
pub extern "C" fn bhc_text_to_lower(text: *const u8) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    unsafe {
        let bytes = text_bytes(text);
        if bytes.is_empty() {
            return bhc_text_empty();
        }
        if bhc_is_ascii(bytes.as_ptr(), bytes.len()) {
            // ASCII fast path: copy and convert in-place
            let mut buf = bytes.to_vec();
            bhc_ascii_to_lower(buf.as_mut_ptr(), buf.len());
            alloc_text_from_bytes(&buf)
        } else {
            let s = text_as_str(text);
            let lower: String = s.to_lowercase();
            alloc_text_from_bytes(lower.as_bytes())
        }
    }
}

/// Convert text to uppercase.
#[no_mangle]
pub extern "C" fn bhc_text_to_upper(text: *const u8) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    unsafe {
        let bytes = text_bytes(text);
        if bytes.is_empty() {
            return bhc_text_empty();
        }
        if bhc_is_ascii(bytes.as_ptr(), bytes.len()) {
            let mut buf = bytes.to_vec();
            bhc_ascii_to_upper(buf.as_mut_ptr(), buf.len());
            alloc_text_from_bytes(&buf)
        } else {
            let s = text_as_str(text);
            let upper: String = s.to_uppercase();
            alloc_text_from_bytes(upper.as_bytes())
        }
    }
}

/// Unicode case folding (canonical lowercase for case-insensitive comparison).
#[no_mangle]
pub extern "C" fn bhc_text_to_case_fold(text: *const u8) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    unsafe {
        let s = text_as_str(text);
        // Rust's to_lowercase() is a reasonable approximation of case folding
        // for most use cases. Full Unicode case folding would require the
        // unicode-case-folding crate.
        let folded: String = s.to_lowercase();
        alloc_text_from_bytes(folded.as_bytes())
    }
}

/// Title case conversion (first letter of each word uppercase).
#[no_mangle]
pub extern "C" fn bhc_text_to_title(text: *const u8) -> *mut u8 {
    if text.is_null() {
        return bhc_text_empty();
    }
    unsafe {
        let s = text_as_str(text);
        let mut result = String::with_capacity(s.len());
        let mut at_word_start = true;
        for c in s.chars() {
            if c.is_whitespace() {
                result.push(c);
                at_word_start = true;
            } else if at_word_start {
                for u in c.to_uppercase() {
                    result.push(u);
                }
                at_word_start = false;
            } else {
                for l in c.to_lowercase() {
                    result.push(l);
                }
            }
        }
        alloc_text_from_bytes(result.as_bytes())
    }
}

// ============================================================
// Pack / Unpack helpers
// ============================================================

/// Pack a BHC cons-list of Char into a Text.
///
/// The list is represented as an opaque pointer. This function walks the
/// RTS list structure using the standard ADT layout:
///
/// - Nil: tag == 0
/// - Cons head tail: tag == 1, fields[0] = head (Char as i64), fields[1] = tail
///
/// `list_ptr` is the head of the BHC cons-list.
#[no_mangle]
pub extern "C" fn bhc_text_pack(list_ptr: *const u8) -> *mut u8 {
    if list_ptr.is_null() {
        return bhc_text_empty();
    }
    let mut result = String::new();
    let mut current = list_ptr;
    unsafe {
        loop {
            let tag = *(current as *const i64);
            if tag == 0 {
                // Nil
                break;
            }
            // Cons: tag == 1, fields at offset 8
            let fields_base = (current as *const *const u8).add(1);
            let head_raw = *fields_base;
            // Head is a Char encoded as a pointer (boxed int)
            let codepoint = head_raw as i64;
            let c = char::from_u32(codepoint as u32).unwrap_or('\u{FFFD}');
            result.push(c);
            // Tail
            current = *fields_base.add(1);
            if current.is_null() {
                break;
            }
        }
    }
    alloc_text_from_bytes(result.as_bytes())
}

/// Return the number of codepoints in a Text (for unpack iteration).
#[no_mangle]
pub extern "C" fn bhc_text_char_count(text: *const u8) -> i64 {
    bhc_text_length(text)
}

/// Return the codepoint at the given index (for unpack iteration).
///
/// `index` is a codepoint index (not byte index).
#[no_mangle]
pub extern "C" fn bhc_text_char_at(text: *const u8, index: i64) -> i64 {
    if text.is_null() || index < 0 {
        return 0;
    }
    unsafe {
        let s = text_as_str(text);
        s.chars()
            .nth(index as usize)
            .map_or(0, |c| c as i64)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_text(s: &str) -> *mut u8 {
        alloc_text_from_bytes(s.as_bytes())
    }

    #[test]
    fn test_empty() {
        let t = bhc_text_empty();
        assert_eq!(bhc_text_null(t), 1);
        assert_eq!(bhc_text_length(t), 0);
    }

    #[test]
    fn test_singleton() {
        let t = bhc_text_singleton('A' as i64);
        assert_eq!(bhc_text_null(t), 0);
        assert_eq!(bhc_text_length(t), 1);
        assert_eq!(bhc_text_head(t), 'A' as i64);
    }

    #[test]
    fn test_singleton_multibyte() {
        let t = bhc_text_singleton('ü' as i64);
        assert_eq!(bhc_text_length(t), 1);
        assert_eq!(bhc_text_head(t), 'ü' as i64);
    }

    #[test]
    fn test_length_ascii() {
        let t = make_text("Hello");
        assert_eq!(bhc_text_length(t), 5);
    }

    #[test]
    fn test_length_multibyte() {
        let t = make_text("café");
        assert_eq!(bhc_text_length(t), 4);
    }

    #[test]
    fn test_eq() {
        let a = make_text("hello");
        let b = make_text("hello");
        let c = make_text("world");
        assert_eq!(bhc_text_eq(a, b), 1);
        assert_eq!(bhc_text_eq(a, c), 0);
    }

    #[test]
    fn test_compare() {
        let a = make_text("abc");
        let b = make_text("abd");
        let c = make_text("abc");
        assert_eq!(bhc_text_compare(a, b), 0); // LT
        assert_eq!(bhc_text_compare(a, c), 1); // EQ
        assert_eq!(bhc_text_compare(b, a), 2); // GT
    }

    #[test]
    fn test_head_last() {
        let t = make_text("Hello");
        assert_eq!(bhc_text_head(t), 'H' as i64);
        assert_eq!(bhc_text_last(t), 'o' as i64);
    }

    #[test]
    fn test_tail() {
        let t = make_text("Hello");
        let tail = bhc_text_tail(t);
        assert_eq!(bhc_text_length(tail), 4);
        assert_eq!(bhc_text_head(tail), 'e' as i64);
    }

    #[test]
    fn test_init() {
        let t = make_text("Hello");
        let init = bhc_text_init(t);
        assert_eq!(bhc_text_length(init), 4);
        assert_eq!(bhc_text_last(init), 'l' as i64);
    }

    #[test]
    fn test_append() {
        let a = make_text("Hello");
        let b = make_text(" World");
        let c = bhc_text_append(a, b);
        assert_eq!(bhc_text_length(c), 11);
        assert_eq!(bhc_text_head(c), 'H' as i64);
        assert_eq!(bhc_text_last(c), 'd' as i64);
    }

    #[test]
    fn test_reverse() {
        let t = make_text("Hello");
        let r = bhc_text_reverse(t);
        assert_eq!(bhc_text_head(r), 'o' as i64);
        assert_eq!(bhc_text_last(r), 'H' as i64);
    }

    #[test]
    fn test_take() {
        let t = make_text("Hello World");
        let taken = bhc_text_take(5, t);
        assert_eq!(bhc_text_length(taken), 5);
        assert_eq!(bhc_text_head(taken), 'H' as i64);
        assert_eq!(bhc_text_last(taken), 'o' as i64);
    }

    #[test]
    fn test_drop() {
        let t = make_text("Hello World");
        let dropped = bhc_text_drop(6, t);
        assert_eq!(bhc_text_length(dropped), 5);
        assert_eq!(bhc_text_head(dropped), 'W' as i64);
    }

    #[test]
    fn test_take_end() {
        let t = make_text("Hello World");
        let taken = bhc_text_take_end(5, t);
        assert_eq!(bhc_text_length(taken), 5);
        assert_eq!(bhc_text_head(taken), 'W' as i64);
    }

    #[test]
    fn test_drop_end() {
        let t = make_text("Hello World");
        let dropped = bhc_text_drop_end(6, t);
        assert_eq!(bhc_text_length(dropped), 5);
        assert_eq!(bhc_text_last(dropped), 'o' as i64);
    }

    #[test]
    fn test_is_prefix_of() {
        let prefix = make_text("Hello");
        let text = make_text("Hello World");
        let other = make_text("World");
        assert_eq!(bhc_text_is_prefix_of(prefix, text), 1);
        assert_eq!(bhc_text_is_prefix_of(other, text), 0);
    }

    #[test]
    fn test_is_suffix_of() {
        let suffix = make_text("World");
        let text = make_text("Hello World");
        let other = make_text("Hello");
        assert_eq!(bhc_text_is_suffix_of(suffix, text), 1);
        assert_eq!(bhc_text_is_suffix_of(other, text), 0);
    }

    #[test]
    fn test_is_infix_of() {
        let needle = make_text("lo Wo");
        let haystack = make_text("Hello World");
        let missing = make_text("xyz");
        assert_eq!(bhc_text_is_infix_of(needle, haystack), 1);
        assert_eq!(bhc_text_is_infix_of(missing, haystack), 0);
    }

    #[test]
    fn test_to_lower() {
        let t = make_text("Hello WORLD");
        let lower = bhc_text_to_lower(t);
        unsafe {
            assert_eq!(text_as_str(lower), "hello world");
        }
    }

    #[test]
    fn test_to_upper() {
        let t = make_text("Hello World");
        let upper = bhc_text_to_upper(t);
        unsafe {
            assert_eq!(text_as_str(upper), "HELLO WORLD");
        }
    }

    #[test]
    fn test_to_title() {
        let t = make_text("hello world");
        let title = bhc_text_to_title(t);
        unsafe {
            assert_eq!(text_as_str(title), "Hello World");
        }
    }

    #[test]
    fn test_to_case_fold() {
        let t = make_text("Hello WORLD");
        let folded = bhc_text_to_case_fold(t);
        unsafe {
            assert_eq!(text_as_str(folded), "hello world");
        }
    }

    #[test]
    fn test_char_count_and_at() {
        let t = make_text("café");
        assert_eq!(bhc_text_char_count(t), 4);
        assert_eq!(bhc_text_char_at(t, 0), 'c' as i64);
        assert_eq!(bhc_text_char_at(t, 3), 'é' as i64);
    }

    #[test]
    fn test_null_safety() {
        assert_eq!(bhc_text_null(ptr::null()), 1);
        assert_eq!(bhc_text_length(ptr::null()), 0);
        assert_eq!(bhc_text_head(ptr::null()), 0);
        assert_eq!(bhc_text_last(ptr::null()), 0);
        assert_eq!(bhc_text_eq(ptr::null(), ptr::null()), 1);
        let _ = bhc_text_tail(ptr::null());
        let _ = bhc_text_init(ptr::null());
        let _ = bhc_text_append(ptr::null(), ptr::null());
        let _ = bhc_text_reverse(ptr::null());
    }

    #[test]
    fn test_multibyte_operations() {
        // Test with emoji (4-byte UTF-8)
        let t = make_text("Hi 🌍!");
        assert_eq!(bhc_text_length(t), 5);
        assert_eq!(bhc_text_head(t), 'H' as i64);
        assert_eq!(bhc_text_char_at(t, 3), '🌍' as i64);

        let taken = bhc_text_take(3, t);
        assert_eq!(bhc_text_length(taken), 3);

        let dropped = bhc_text_drop(3, t);
        assert_eq!(bhc_text_length(dropped), 2);
        assert_eq!(bhc_text_head(dropped), '🌍' as i64);
    }
}
