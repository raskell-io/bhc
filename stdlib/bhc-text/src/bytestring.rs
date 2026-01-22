//! ByteString type
//!
//! Raw byte arrays for binary data processing.
//!
//! # Overview
//!
//! The `ByteString` type represents an immutable sequence of bytes,
//! suitable for binary data, network protocols, and file I/O.
//! Operations are optimized for byte-level processing.
//!
//! # Examples
//!
//! ```ignore
//! use bhc_text::bytestring::ByteString;
//!
//! let bs = ByteString::pack(b"hello");
//! assert_eq!(bs.len(), 5);
//! assert_eq!(bs.index(0), Some(104));
//! ```

use std::rc::Rc;

// ============================================================
// Core ByteString Type
// ============================================================

/// Immutable byte array.
///
/// `ByteString` provides an efficient representation for binary data,
/// with reference counting for efficient copying and slicing.
#[derive(Clone)]
pub struct ByteString {
    data: Rc<Vec<u8>>,
    start: usize,
    len: usize,
}

impl ByteString {
    /// Create an empty bytestring.
    #[inline]
    pub fn empty() -> Self {
        ByteString {
            data: Rc::new(Vec::new()),
            start: 0,
            len: 0,
        }
    }

    /// Create bytestring from a byte slice.
    #[inline]
    pub fn pack(bytes: &[u8]) -> Self {
        ByteString {
            data: Rc::new(bytes.to_vec()),
            start: 0,
            len: bytes.len(),
        }
    }

    /// Create bytestring from a single byte.
    #[inline]
    pub fn singleton(b: u8) -> Self {
        Self::pack(&[b])
    }

    /// Convert bytestring to a Vec<u8>.
    #[inline]
    pub fn unpack(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }

    /// Get a slice view of the bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data[self.start..self.start + self.len]
    }

    /// Length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the bytestring is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    // --------------------------------------------------------
    // Construction
    // --------------------------------------------------------

    /// Prepend a byte.
    #[inline]
    pub fn cons(b: u8, bs: &ByteString) -> ByteString {
        let mut result = Vec::with_capacity(1 + bs.len());
        result.push(b);
        result.extend_from_slice(bs.as_bytes());
        Self::pack(&result)
    }

    /// Append a byte.
    #[inline]
    pub fn snoc(bs: &ByteString, b: u8) -> ByteString {
        let mut result = Vec::with_capacity(bs.len() + 1);
        result.extend_from_slice(bs.as_bytes());
        result.push(b);
        Self::pack(&result)
    }

    /// Append two bytestrings.
    #[inline]
    pub fn append(a: &ByteString, b: &ByteString) -> ByteString {
        if a.is_empty() {
            return b.clone();
        }
        if b.is_empty() {
            return a.clone();
        }
        let mut result = Vec::with_capacity(a.len() + b.len());
        result.extend_from_slice(a.as_bytes());
        result.extend_from_slice(b.as_bytes());
        Self::pack(&result)
    }

    /// Get the first byte and remaining bytes.
    #[inline]
    pub fn uncons(&self) -> Option<(u8, ByteString)> {
        if self.is_empty() {
            None
        } else {
            let b = self.as_bytes()[0];
            let rest = ByteString {
                data: self.data.clone(),
                start: self.start + 1,
                len: self.len - 1,
            };
            Some((b, rest))
        }
    }

    /// Get the last byte and preceding bytes.
    #[inline]
    pub fn unsnoc(&self) -> Option<(ByteString, u8)> {
        if self.is_empty() {
            None
        } else {
            let b = self.as_bytes()[self.len - 1];
            let init = ByteString {
                data: self.data.clone(),
                start: self.start,
                len: self.len - 1,
            };
            Some((init, b))
        }
    }

    // --------------------------------------------------------
    // Basic Operations
    // --------------------------------------------------------

    /// Get the first byte.
    #[inline]
    pub fn head(&self) -> Option<u8> {
        if self.is_empty() {
            None
        } else {
            Some(self.as_bytes()[0])
        }
    }

    /// Get the last byte.
    #[inline]
    pub fn last(&self) -> Option<u8> {
        if self.is_empty() {
            None
        } else {
            Some(self.as_bytes()[self.len - 1])
        }
    }

    /// Get all but the first byte.
    #[inline]
    pub fn tail(&self) -> Option<ByteString> {
        if self.is_empty() {
            None
        } else {
            Some(ByteString {
                data: self.data.clone(),
                start: self.start + 1,
                len: self.len - 1,
            })
        }
    }

    /// Get all but the last byte.
    #[inline]
    pub fn init(&self) -> Option<ByteString> {
        if self.is_empty() {
            None
        } else {
            Some(ByteString {
                data: self.data.clone(),
                start: self.start,
                len: self.len - 1,
            })
        }
    }

    // --------------------------------------------------------
    // Transformations
    // --------------------------------------------------------

    /// Apply a function to each byte.
    pub fn map<F>(&self, f: F) -> ByteString
    where
        F: Fn(u8) -> u8,
    {
        let result: Vec<u8> = self.as_bytes().iter().map(|&b| f(b)).collect();
        Self::pack(&result)
    }

    /// Reverse the bytestring.
    pub fn reverse(&self) -> ByteString {
        let result: Vec<u8> = self.as_bytes().iter().rev().copied().collect();
        Self::pack(&result)
    }

    /// Insert byte between elements.
    pub fn intersperse(sep: u8, bs: &ByteString) -> ByteString {
        if bs.len() <= 1 {
            return bs.clone();
        }
        let bytes = bs.as_bytes();
        let mut result = Vec::with_capacity(bytes.len() * 2 - 1);
        for (i, &b) in bytes.iter().enumerate() {
            if i > 0 {
                result.push(sep);
            }
            result.push(b);
        }
        Self::pack(&result)
    }

    /// Join bytestrings with separator.
    pub fn intercalate(sep: &ByteString, bss: &[ByteString]) -> ByteString {
        if bss.is_empty() {
            return ByteString::empty();
        }
        let total_len: usize = bss.iter().map(|bs| bs.len()).sum::<usize>()
            + sep.len() * (bss.len() - 1);
        let mut result = Vec::with_capacity(total_len);
        for (i, bs) in bss.iter().enumerate() {
            if i > 0 {
                result.extend_from_slice(sep.as_bytes());
            }
            result.extend_from_slice(bs.as_bytes());
        }
        Self::pack(&result)
    }

    // --------------------------------------------------------
    // Folds
    // --------------------------------------------------------

    /// Left fold over bytes.
    pub fn foldl<B, F>(&self, init: B, f: F) -> B
    where
        F: Fn(B, u8) -> B,
    {
        self.as_bytes().iter().fold(init, |acc, &b| f(acc, b))
    }

    /// Right fold over bytes.
    pub fn foldr<B, F>(&self, init: B, f: F) -> B
    where
        F: Fn(u8, B) -> B,
    {
        self.as_bytes().iter().rev().fold(init, |acc, &b| f(b, acc))
    }

    /// Concatenate a list of bytestrings.
    pub fn concat(bss: &[ByteString]) -> ByteString {
        let total_len: usize = bss.iter().map(|bs| bs.len()).sum();
        let mut result = Vec::with_capacity(total_len);
        for bs in bss {
            result.extend_from_slice(bs.as_bytes());
        }
        Self::pack(&result)
    }

    /// Map and concatenate.
    pub fn concat_map<F>(&self, f: F) -> ByteString
    where
        F: Fn(u8) -> ByteString,
    {
        let results: Vec<ByteString> = self.as_bytes().iter().map(|&b| f(b)).collect();
        Self::concat(&results)
    }

    /// Check if any byte satisfies a predicate.
    pub fn any<F>(&self, f: F) -> bool
    where
        F: Fn(u8) -> bool,
    {
        self.as_bytes().iter().any(|&b| f(b))
    }

    /// Check if all bytes satisfy a predicate.
    pub fn all<F>(&self, f: F) -> bool
    where
        F: Fn(u8) -> bool,
    {
        self.as_bytes().iter().all(|&b| f(b))
    }

    /// Find maximum byte.
    pub fn maximum(&self) -> Option<u8> {
        self.as_bytes().iter().copied().max()
    }

    /// Find minimum byte.
    pub fn minimum(&self) -> Option<u8> {
        self.as_bytes().iter().copied().min()
    }

    // --------------------------------------------------------
    // Substrings
    // --------------------------------------------------------

    /// Take first n bytes.
    #[inline]
    pub fn take(&self, n: usize) -> ByteString {
        let take_len = n.min(self.len);
        ByteString {
            data: self.data.clone(),
            start: self.start,
            len: take_len,
        }
    }

    /// Take last n bytes.
    #[inline]
    pub fn take_end(&self, n: usize) -> ByteString {
        if n >= self.len {
            return self.clone();
        }
        let skip = self.len - n;
        ByteString {
            data: self.data.clone(),
            start: self.start + skip,
            len: n,
        }
    }

    /// Drop first n bytes.
    #[inline]
    pub fn drop(&self, n: usize) -> ByteString {
        if n >= self.len {
            return ByteString::empty();
        }
        ByteString {
            data: self.data.clone(),
            start: self.start + n,
            len: self.len - n,
        }
    }

    /// Drop last n bytes.
    #[inline]
    pub fn drop_end(&self, n: usize) -> ByteString {
        if n >= self.len {
            return ByteString::empty();
        }
        ByteString {
            data: self.data.clone(),
            start: self.start,
            len: self.len - n,
        }
    }

    /// Split at position n.
    #[inline]
    pub fn split_at(&self, n: usize) -> (ByteString, ByteString) {
        (self.take(n), self.drop(n))
    }

    /// Take bytes while predicate holds.
    pub fn take_while<F>(&self, f: F) -> ByteString
    where
        F: Fn(u8) -> bool,
    {
        let bytes = self.as_bytes();
        let mut take_len = 0;
        for &b in bytes {
            if !f(b) {
                break;
            }
            take_len += 1;
        }
        self.take(take_len)
    }

    /// Drop bytes while predicate holds.
    pub fn drop_while<F>(&self, f: F) -> ByteString
    where
        F: Fn(u8) -> bool,
    {
        let bytes = self.as_bytes();
        let mut drop_count = 0;
        for &b in bytes {
            if !f(b) {
                break;
            }
            drop_count += 1;
        }
        self.drop(drop_count)
    }

    /// Split on predicate.
    pub fn span<F>(&self, f: F) -> (ByteString, ByteString)
    where
        F: Fn(u8) -> bool,
    {
        let bytes = self.as_bytes();
        let mut split_at = 0;
        for &b in bytes {
            if !f(b) {
                break;
            }
            split_at += 1;
        }
        (self.take(split_at), self.drop(split_at))
    }

    /// Split on first occurrence of byte.
    pub fn break_byte(&self, sep: u8) -> (ByteString, ByteString) {
        let bytes = self.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b == sep {
                return (self.take(i), self.drop(i));
            }
        }
        (self.clone(), ByteString::empty())
    }

    // --------------------------------------------------------
    // Splitting
    // --------------------------------------------------------

    /// Split on separator byte.
    pub fn split(&self, sep: u8) -> Vec<ByteString> {
        let bytes = self.as_bytes();
        let mut result = Vec::new();
        let mut start = 0;

        for (i, &b) in bytes.iter().enumerate() {
            if b == sep {
                result.push(ByteString {
                    data: self.data.clone(),
                    start: self.start + start,
                    len: i - start,
                });
                start = i + 1;
            }
        }

        result.push(ByteString {
            data: self.data.clone(),
            start: self.start + start,
            len: bytes.len() - start,
        });

        result
    }

    /// Split on newlines.
    pub fn lines(&self) -> Vec<ByteString> {
        self.split(b'\n')
    }

    /// Split on whitespace.
    pub fn words(&self) -> Vec<ByteString> {
        let bytes = self.as_bytes();
        let mut result = Vec::new();
        let mut start = None;

        for (i, &b) in bytes.iter().enumerate() {
            if b.is_ascii_whitespace() {
                if let Some(s) = start {
                    result.push(ByteString {
                        data: self.data.clone(),
                        start: self.start + s,
                        len: i - s,
                    });
                    start = None;
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }

        if let Some(s) = start {
            result.push(ByteString {
                data: self.data.clone(),
                start: self.start + s,
                len: bytes.len() - s,
            });
        }

        result
    }

    /// Join lines with newlines.
    pub fn unlines(bss: &[ByteString]) -> ByteString {
        Self::intercalate(&ByteString::singleton(b'\n'), bss)
    }

    /// Join words with spaces.
    pub fn unwords(bss: &[ByteString]) -> ByteString {
        Self::intercalate(&ByteString::singleton(b' '), bss)
    }

    // --------------------------------------------------------
    // Predicates
    // --------------------------------------------------------

    /// Check if starts with prefix.
    pub fn is_prefix_of(&self, bs: &ByteString) -> bool {
        if self.len > bs.len {
            return false;
        }
        self.as_bytes() == &bs.as_bytes()[..self.len]
    }

    /// Check if ends with suffix.
    pub fn is_suffix_of(&self, bs: &ByteString) -> bool {
        if self.len > bs.len {
            return false;
        }
        self.as_bytes() == &bs.as_bytes()[bs.len - self.len..]
    }

    /// Check if is a subsequence.
    pub fn is_infix_of(&self, bs: &ByteString) -> bool {
        if self.len > bs.len {
            return false;
        }
        let needle = self.as_bytes();
        let haystack = bs.as_bytes();
        for i in 0..=(haystack.len() - needle.len()) {
            if &haystack[i..i + needle.len()] == needle {
                return true;
            }
        }
        false
    }

    /// Strip prefix if present.
    pub fn strip_prefix(&self, prefix: &ByteString) -> Option<ByteString> {
        if prefix.is_prefix_of(self) {
            Some(self.drop(prefix.len))
        } else {
            None
        }
    }

    /// Strip suffix if present.
    pub fn strip_suffix(&self, suffix: &ByteString) -> Option<ByteString> {
        if suffix.is_suffix_of(self) {
            Some(self.drop_end(suffix.len))
        } else {
            None
        }
    }

    // --------------------------------------------------------
    // Search
    // --------------------------------------------------------

    /// Filter bytes by predicate.
    pub fn filter<F>(&self, f: F) -> ByteString
    where
        F: Fn(u8) -> bool,
    {
        let result: Vec<u8> = self.as_bytes().iter().copied().filter(|&b| f(b)).collect();
        Self::pack(&result)
    }

    /// Check if byte is in bytestring.
    pub fn elem(&self, b: u8) -> bool {
        self.as_bytes().contains(&b)
    }

    /// Check if byte is NOT in bytestring.
    pub fn not_elem(&self, b: u8) -> bool {
        !self.elem(b)
    }

    /// Find first byte satisfying predicate.
    pub fn find<F>(&self, f: F) -> Option<u8>
    where
        F: Fn(u8) -> bool,
    {
        self.as_bytes().iter().copied().find(|&b| f(b))
    }

    /// Partition by predicate.
    pub fn partition<F>(&self, f: F) -> (ByteString, ByteString)
    where
        F: Fn(u8) -> bool,
    {
        let mut yes = Vec::new();
        let mut no = Vec::new();
        for &b in self.as_bytes() {
            if f(b) {
                yes.push(b);
            } else {
                no.push(b);
            }
        }
        (Self::pack(&yes), Self::pack(&no))
    }

    // --------------------------------------------------------
    // Indexing
    // --------------------------------------------------------

    /// Get byte at index.
    #[inline]
    pub fn index(&self, i: usize) -> Option<u8> {
        if i < self.len {
            Some(self.as_bytes()[i])
        } else {
            None
        }
    }

    /// Find index of first occurrence of byte.
    pub fn elem_index(&self, b: u8) -> Option<usize> {
        self.as_bytes().iter().position(|&x| x == b)
    }

    /// Find all indices of byte.
    pub fn elem_indices(&self, b: u8) -> Vec<usize> {
        self.as_bytes()
            .iter()
            .enumerate()
            .filter(|(_, &x)| x == b)
            .map(|(i, _)| i)
            .collect()
    }

    /// Find index of first byte satisfying predicate.
    pub fn find_index<F>(&self, f: F) -> Option<usize>
    where
        F: Fn(u8) -> bool,
    {
        self.as_bytes().iter().position(|&b| f(b))
    }

    /// Find all indices of bytes satisfying predicate.
    pub fn find_indices<F>(&self, f: F) -> Vec<usize>
    where
        F: Fn(u8) -> bool,
    {
        self.as_bytes()
            .iter()
            .enumerate()
            .filter(|(_, &b)| f(b))
            .map(|(i, _)| i)
            .collect()
    }

    /// Count occurrences of byte.
    pub fn count(&self, b: u8) -> usize {
        self.as_bytes().iter().filter(|&&x| x == b).count()
    }

    // --------------------------------------------------------
    // Zipping
    // --------------------------------------------------------

    /// Zip two bytestrings into pairs.
    pub fn zip(a: &ByteString, b: &ByteString) -> Vec<(u8, u8)> {
        a.as_bytes()
            .iter()
            .zip(b.as_bytes().iter())
            .map(|(&x, &y)| (x, y))
            .collect()
    }

    /// Zip two bytestrings with a function.
    pub fn zip_with<C, F>(f: F, a: &ByteString, b: &ByteString) -> Vec<C>
    where
        F: Fn(u8, u8) -> C,
    {
        a.as_bytes()
            .iter()
            .zip(b.as_bytes().iter())
            .map(|(&x, &y)| f(x, y))
            .collect()
    }

    // --------------------------------------------------------
    // Conversion
    // --------------------------------------------------------

    /// Try to convert to UTF-8 string.
    pub fn decode_utf8(&self) -> Option<String> {
        std::str::from_utf8(self.as_bytes())
            .ok()
            .map(|s| s.to_string())
    }

    /// Convert to UTF-8, replacing invalid sequences.
    pub fn decode_utf8_lossy(&self) -> String {
        String::from_utf8_lossy(self.as_bytes()).into_owned()
    }

    /// Create from UTF-8 string.
    pub fn encode_utf8(s: &str) -> ByteString {
        Self::pack(s.as_bytes())
    }

    /// Create copy of data in a new allocation.
    pub fn copy(&self) -> ByteString {
        Self::pack(self.as_bytes())
    }

    // --------------------------------------------------------
    // Comparison
    // --------------------------------------------------------

    /// Compare two bytestrings lexicographically.
    pub fn compare(&self, other: &ByteString) -> std::cmp::Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

// ============================================================
// Trait Implementations
// ============================================================

impl Default for ByteString {
    fn default() -> Self {
        Self::empty()
    }
}

impl PartialEq for ByteString {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for ByteString {}

impl PartialOrd for ByteString {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ByteString {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl std::hash::Hash for ByteString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

impl std::fmt::Debug for ByteString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ByteString({:?})", self.as_bytes())
    }
}

impl From<&[u8]> for ByteString {
    fn from(bytes: &[u8]) -> Self {
        Self::pack(bytes)
    }
}

impl From<Vec<u8>> for ByteString {
    fn from(bytes: Vec<u8>) -> Self {
        Self::pack(&bytes)
    }
}

impl From<&str> for ByteString {
    fn from(s: &str) -> Self {
        Self::encode_utf8(s)
    }
}

impl From<String> for ByteString {
    fn from(s: String) -> Self {
        Self::encode_utf8(&s)
    }
}

// ============================================================
// FFI Functions
// ============================================================

/// Get bytestring length.
#[no_mangle]
pub extern "C" fn bhc_bytestring_length(ptr: *const u8, len: usize) -> usize {
    len
}

/// Check if bytestring is empty.
#[no_mangle]
pub extern "C" fn bhc_bytestring_null(len: usize) -> bhc_prelude::bool::Bool {
    bhc_prelude::bool::Bool::from_bool(len == 0)
}

/// Copy bytestring.
#[no_mangle]
pub extern "C" fn bhc_bytestring_copy(src: *const u8, src_len: usize, dst: *mut u8) {
    if src.is_null() || dst.is_null() || src_len == 0 {
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, src_len);
    }
}

/// Concatenate two bytestrings.
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

/// Find byte in bytestring.
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

/// Compare two bytestrings.
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

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let bs = ByteString::empty();
        assert!(bs.is_empty());
        assert_eq!(bs.len(), 0);
    }

    #[test]
    fn test_pack_unpack() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(bs.len(), 5);
        assert_eq!(bs.unpack(), b"hello".to_vec());
    }

    #[test]
    fn test_singleton() {
        let bs = ByteString::singleton(42);
        assert_eq!(bs.len(), 1);
        assert_eq!(bs.head(), Some(42));
    }

    #[test]
    fn test_cons_snoc() {
        let bs = ByteString::pack(b"ello");
        let bs2 = ByteString::cons(b'h', &bs);
        assert_eq!(bs2.unpack(), b"hello".to_vec());

        let bs3 = ByteString::snoc(&bs2, b'!');
        assert_eq!(bs3.unpack(), b"hello!".to_vec());
    }

    #[test]
    fn test_append() {
        let a = ByteString::pack(b"hello");
        let b = ByteString::pack(b" world");
        let c = ByteString::append(&a, &b);
        assert_eq!(c.unpack(), b"hello world".to_vec());
    }

    #[test]
    fn test_uncons_unsnoc() {
        let bs = ByteString::pack(b"hello");
        let (h, rest) = bs.uncons().unwrap();
        assert_eq!(h, b'h');
        assert_eq!(rest.unpack(), b"ello".to_vec());

        let (init, o) = bs.unsnoc().unwrap();
        assert_eq!(init.unpack(), b"hell".to_vec());
        assert_eq!(o, b'o');
    }

    #[test]
    fn test_head_last() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(bs.head(), Some(b'h'));
        assert_eq!(bs.last(), Some(b'o'));

        assert_eq!(ByteString::empty().head(), None);
        assert_eq!(ByteString::empty().last(), None);
    }

    #[test]
    fn test_tail_init() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(bs.tail().unwrap().unpack(), b"ello".to_vec());
        assert_eq!(bs.init().unwrap().unpack(), b"hell".to_vec());
    }

    #[test]
    fn test_map() {
        let bs = ByteString::pack(b"hello");
        let upper = bs.map(|b| b.to_ascii_uppercase());
        assert_eq!(upper.unpack(), b"HELLO".to_vec());
    }

    #[test]
    fn test_reverse() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(bs.reverse().unpack(), b"olleh".to_vec());
    }

    #[test]
    fn test_intersperse() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(
            ByteString::intersperse(b',', &bs).unpack(),
            b"h,e,l,l,o".to_vec()
        );
    }

    #[test]
    fn test_folds() {
        let bs = ByteString::pack(&[1, 2, 3]);
        let sum: u16 = bs.foldl(0u16, |acc, b| acc + b as u16);
        assert_eq!(sum, 6);

        assert!(bs.all(|b| b < 10));
        assert!(bs.any(|b| b == 2));
        assert!(!bs.any(|b| b == 5));

        assert_eq!(bs.maximum(), Some(3));
        assert_eq!(bs.minimum(), Some(1));
    }

    #[test]
    fn test_take_drop() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(bs.take(3).unpack(), b"hel".to_vec());
        assert_eq!(bs.drop(3).unpack(), b"lo".to_vec());
        assert_eq!(bs.take_end(3).unpack(), b"llo".to_vec());
        assert_eq!(bs.drop_end(3).unpack(), b"he".to_vec());
    }

    #[test]
    fn test_split_at() {
        let bs = ByteString::pack(b"hello");
        let (a, b) = bs.split_at(3);
        assert_eq!(a.unpack(), b"hel".to_vec());
        assert_eq!(b.unpack(), b"lo".to_vec());
    }

    #[test]
    fn test_split() {
        let bs = ByteString::pack(b"hello,world,foo");
        let parts = bs.split(b',');
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].unpack(), b"hello".to_vec());
        assert_eq!(parts[1].unpack(), b"world".to_vec());
        assert_eq!(parts[2].unpack(), b"foo".to_vec());
    }

    #[test]
    fn test_lines_words() {
        let bs = ByteString::pack(b"hello\nworld");
        let lines = bs.lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].unpack(), b"hello".to_vec());
        assert_eq!(lines[1].unpack(), b"world".to_vec());

        let bs2 = ByteString::pack(b"hello world foo");
        let words = bs2.words();
        assert_eq!(words.len(), 3);
    }

    #[test]
    fn test_predicates() {
        let hello = ByteString::pack(b"hello");
        let prefix = ByteString::pack(b"hel");
        let suffix = ByteString::pack(b"llo");
        let infix = ByteString::pack(b"ell");

        assert!(prefix.is_prefix_of(&hello));
        assert!(suffix.is_suffix_of(&hello));
        assert!(infix.is_infix_of(&hello));
        assert!(!suffix.is_prefix_of(&hello));
    }

    #[test]
    fn test_filter_find() {
        let bs = ByteString::pack(b"hello123");
        let letters = bs.filter(|b| b.is_ascii_alphabetic());
        assert_eq!(letters.unpack(), b"hello".to_vec());

        let digit = bs.find(|b| b.is_ascii_digit());
        assert_eq!(digit, Some(b'1'));

        assert!(bs.elem(b'e'));
        assert!(!bs.elem(b'z'));
    }

    #[test]
    fn test_index() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(bs.index(0), Some(b'h'));
        assert_eq!(bs.index(4), Some(b'o'));
        assert_eq!(bs.index(5), None);
    }

    #[test]
    fn test_elem_index() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(bs.elem_index(b'l'), Some(2));
        assert_eq!(bs.elem_indices(b'l'), vec![2, 3]);
        assert_eq!(bs.elem_index(b'z'), None);
    }

    #[test]
    fn test_count() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(bs.count(b'l'), 2);
        assert_eq!(bs.count(b'z'), 0);
    }

    #[test]
    fn test_zip() {
        let a = ByteString::pack(b"abc");
        let b = ByteString::pack(b"123");
        let pairs = ByteString::zip(&a, &b);
        assert_eq!(pairs, vec![(b'a', b'1'), (b'b', b'2'), (b'c', b'3')]);
    }

    #[test]
    fn test_decode_utf8() {
        let bs = ByteString::pack(b"hello");
        assert_eq!(bs.decode_utf8(), Some("hello".to_string()));

        let invalid = ByteString::pack(&[0xff, 0xfe]);
        assert_eq!(invalid.decode_utf8(), None);
    }

    // FFI tests
    #[test]
    fn test_ffi_elem() {
        let bytes = [1u8, 2, 3, 4, 5];
        assert_eq!(bhc_bytestring_elem(bytes.as_ptr(), bytes.len(), 3), 2);
        assert_eq!(bhc_bytestring_elem(bytes.as_ptr(), bytes.len(), 6), -1);
    }
}
