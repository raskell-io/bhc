//! String operations
//!
//! C-callable RTS functions for string manipulation used by
//! LLVM-generated code. Strings are lists of characters where each
//! character is a Unicode code point stored as `i64` cast to `*mut u8`.

use std::alloc::{alloc, Layout};

// ---------------------------------------------------------------------------
// ADT helpers (internal, duplicated from list.rs for module independence)
// ---------------------------------------------------------------------------

unsafe fn get_tag(ptr: *mut u8) -> i64 {
    *(ptr as *const i64)
}

unsafe fn get_field(ptr: *mut u8, index: usize) -> *mut u8 {
    *(ptr.add(8 + index * 8) as *const *mut u8)
}

unsafe fn alloc_nil() -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(8, 8);
    let ptr = alloc(layout);
    *(ptr as *mut i64) = 0;
    ptr
}

unsafe fn alloc_cons(head: *mut u8, tail: *mut u8) -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(24, 8);
    let ptr = alloc(layout);
    *(ptr as *mut i64) = 1;
    *(ptr.add(8) as *mut *mut u8) = head;
    *(ptr.add(16) as *mut *mut u8) = tail;
    ptr
}

unsafe fn list_to_vec(mut list: *mut u8) -> Vec<*mut u8> {
    let mut vec = Vec::new();
    loop {
        if get_tag(list) == 0 {
            break;
        }
        vec.push(get_field(list, 0));
        list = get_field(list, 1);
    }
    vec
}

unsafe fn vec_to_list(slice: &[*mut u8]) -> *mut u8 {
    let mut result = alloc_nil();
    for &elem in slice.iter().rev() {
        result = alloc_cons(elem, result);
    }
    result
}

// ---------------------------------------------------------------------------
// Character helpers
// ---------------------------------------------------------------------------

/// Extract the codepoint from a boxed character (pointer IS the value).
fn char_val(c: *mut u8) -> i64 {
    c as i64
}

/// Box a codepoint as a character pointer.
fn box_char(cp: i64) -> *mut u8 {
    cp as *mut u8
}

/// Test whether a codepoint is whitespace (space, tab, newline, CR).
fn is_whitespace(cp: i64) -> bool {
    cp == 32 || cp == 9 || cp == 10 || cp == 13
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Split a string (char list) by newline characters.
///
/// Follows Haskell `lines` semantics:
/// - `lines "" = [""]`
/// - `lines "\n" = [""]`
/// - `lines "abc" = ["abc"]`
/// - `lines "abc\ndef" = ["abc","def"]`
/// - `lines "abc\n" = ["abc"]`
#[no_mangle]
pub unsafe extern "C" fn bhc_string_lines(str_list: *mut u8) -> *mut u8 {
    let chars = list_to_vec(str_list);
    let newline: i64 = 10;

    if chars.is_empty() {
        // lines "" = [""]
        let empty_str = alloc_nil();
        return vec_to_list(&[empty_str]);
    }

    let mut result_lines: Vec<Vec<*mut u8>> = Vec::new();
    let mut current: Vec<*mut u8> = Vec::new();

    for &ch in &chars {
        if char_val(ch) == newline {
            result_lines.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }

    // Haskell: trailing newline does NOT produce a trailing empty string.
    // But if the string does not end with newline, the last segment is added.
    if !current.is_empty() {
        result_lines.push(current);
    }

    let sublists: Vec<*mut u8> = result_lines.iter().map(|l| vec_to_list(l)).collect();
    vec_to_list(&sublists)
}

/// Join strings with newlines appended to each.
///
/// `unlines ["abc","def"] = "abc\ndef\n"`
#[no_mangle]
pub unsafe extern "C" fn bhc_string_unlines(lines_list: *mut u8) -> *mut u8 {
    let lines = list_to_vec(lines_list);
    let newline = box_char(10);
    let mut result: Vec<*mut u8> = Vec::new();

    for &line_ptr in &lines {
        let line_chars = list_to_vec(line_ptr);
        result.extend_from_slice(&line_chars);
        result.push(newline);
    }

    vec_to_list(&result)
}

/// Split a string by whitespace.
///
/// Follows Haskell `words` semantics:
/// - `words "" = []`
/// - `words "  abc  def  " = ["abc","def"]`
/// - Leading, trailing, and multiple whitespace is collapsed.
#[no_mangle]
pub unsafe extern "C" fn bhc_string_words(str_list: *mut u8) -> *mut u8 {
    let chars = list_to_vec(str_list);

    let mut result_words: Vec<Vec<*mut u8>> = Vec::new();
    let mut current: Vec<*mut u8> = Vec::new();

    for &ch in &chars {
        if is_whitespace(char_val(ch)) {
            if !current.is_empty() {
                result_words.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        result_words.push(current);
    }

    let sublists: Vec<*mut u8> = result_words.iter().map(|w| vec_to_list(w)).collect();
    vec_to_list(&sublists)
}

/// Join strings with single spaces.
///
/// `unwords ["abc","def"] = "abc def"`
#[no_mangle]
pub unsafe extern "C" fn bhc_string_unwords(words_list: *mut u8) -> *mut u8 {
    let words = list_to_vec(words_list);
    let space = box_char(32);

    if words.is_empty() {
        return alloc_nil();
    }

    let mut result: Vec<*mut u8> = Vec::new();
    for (i, &word_ptr) in words.iter().enumerate() {
        if i > 0 {
            result.push(space);
        }
        let word_chars = list_to_vec(word_ptr);
        result.extend_from_slice(&word_chars);
    }

    vec_to_list(&result)
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn make_string(s: &str) -> *mut u8 {
        let ptrs: Vec<*mut u8> = s.chars().map(|c| box_char(c as i64)).collect();
        vec_to_list(&ptrs)
    }

    unsafe fn collect_string(list: *mut u8) -> String {
        list_to_vec(list)
            .into_iter()
            .map(|p| char::from_u32(char_val(p) as u32).unwrap_or('\u{FFFD}'))
            .collect()
    }

    unsafe fn collect_strings(list: *mut u8) -> Vec<String> {
        list_to_vec(list)
            .into_iter()
            .map(|sub| collect_string(sub))
            .collect()
    }

    // -- lines tests --

    #[test]
    fn test_lines_empty() {
        unsafe {
            let s = make_string("");
            let result = bhc_string_lines(s);
            assert_eq!(collect_strings(result), vec![""]);
        }
    }

    #[test]
    fn test_lines_no_newline() {
        unsafe {
            let s = make_string("abc");
            let result = bhc_string_lines(s);
            assert_eq!(collect_strings(result), vec!["abc"]);
        }
    }

    #[test]
    fn test_lines_with_newline() {
        unsafe {
            let s = make_string("abc\ndef");
            let result = bhc_string_lines(s);
            assert_eq!(collect_strings(result), vec!["abc", "def"]);
        }
    }

    #[test]
    fn test_lines_trailing_newline() {
        unsafe {
            let s = make_string("abc\n");
            let result = bhc_string_lines(s);
            assert_eq!(collect_strings(result), vec!["abc"]);
        }
    }

    #[test]
    fn test_lines_only_newline() {
        unsafe {
            let s = make_string("\n");
            let result = bhc_string_lines(s);
            assert_eq!(collect_strings(result), vec![""]);
        }
    }

    // -- unlines tests --

    #[test]
    fn test_unlines() {
        unsafe {
            let a = make_string("abc");
            let b = make_string("def");
            let lines = vec_to_list(&[a, b]);
            let result = bhc_string_unlines(lines);
            assert_eq!(collect_string(result), "abc\ndef\n");
        }
    }

    // -- words tests --

    #[test]
    fn test_words_empty() {
        unsafe {
            let s = make_string("");
            let result = bhc_string_words(s);
            assert_eq!(collect_strings(result), Vec::<String>::new());
        }
    }

    #[test]
    fn test_words_whitespace() {
        unsafe {
            let s = make_string("  abc  def  ");
            let result = bhc_string_words(s);
            assert_eq!(collect_strings(result), vec!["abc", "def"]);
        }
    }

    #[test]
    fn test_words_tabs_and_newlines() {
        unsafe {
            let s = make_string("hello\tworld\nfoo");
            let result = bhc_string_words(s);
            assert_eq!(collect_strings(result), vec!["hello", "world", "foo"]);
        }
    }

    // -- unwords tests --

    #[test]
    fn test_unwords() {
        unsafe {
            let a = make_string("abc");
            let b = make_string("def");
            let words = vec_to_list(&[a, b]);
            let result = bhc_string_unwords(words);
            assert_eq!(collect_string(result), "abc def");
        }
    }

    #[test]
    fn test_unwords_empty() {
        unsafe {
            let words = alloc_nil();
            let result = bhc_string_unwords(words);
            assert_eq!(collect_string(result), "");
        }
    }
}
