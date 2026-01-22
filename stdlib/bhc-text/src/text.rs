//! UTF-8 Text type
//!
//! Efficient text representation with UTF-8 encoding.
//!
//! # Overview
//!
//! The `Text` type represents a Unicode string stored as UTF-8 bytes.
//! Operations are optimized for common patterns and many can be accelerated
//! with SIMD on supported platforms.
//!
//! # Examples
//!
//! ```ignore
//! use bhc_text::text::Text;
//!
//! let t = Text::pack("hello");
//! assert_eq!(t.len(), 5);
//! assert_eq!(t.unpack(), "hello");
//! ```

use std::rc::Rc;
use std::str;

// ============================================================
// Core Text Type
// ============================================================

/// UTF-8 encoded immutable text.
///
/// `Text` provides an efficient representation for Unicode strings,
/// using UTF-8 encoding internally. It supports sharing through
/// reference counting for efficient copying.
#[derive(Clone)]
pub struct Text {
    data: Rc<Vec<u8>>,
    start: usize,
    len: usize,
}

impl Text {
    /// Create an empty text.
    #[inline]
    pub fn empty() -> Self {
        Text {
            data: Rc::new(Vec::new()),
            start: 0,
            len: 0,
        }
    }

    /// Create text from a string slice.
    #[inline]
    pub fn pack(s: &str) -> Self {
        Text {
            data: Rc::new(s.as_bytes().to_vec()),
            start: 0,
            len: s.len(),
        }
    }

    /// Create text from a single character.
    #[inline]
    pub fn singleton(c: char) -> Self {
        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        Self::pack(s)
    }

    /// Convert text to a String.
    #[inline]
    pub fn unpack(&self) -> String {
        self.as_str().to_string()
    }

    /// Get a string slice view of the text.
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: Text invariant guarantees valid UTF-8
        unsafe { str::from_utf8_unchecked(self.as_bytes()) }
    }

    /// Get the underlying bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data[self.start..self.start + self.len]
    }

    /// Length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the text is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Length in Unicode code points (characters).
    #[inline]
    pub fn char_count(&self) -> usize {
        self.as_str().chars().count()
    }

    /// Compare length with a given value efficiently.
    ///
    /// Returns `Ordering::Less`, `Ordering::Equal`, or `Ordering::Greater`.
    /// More efficient than computing full length when you just need comparison.
    #[inline]
    pub fn compare_length(&self, n: usize) -> std::cmp::Ordering {
        self.len.cmp(&n)
    }

    // --------------------------------------------------------
    // Construction
    // --------------------------------------------------------

    /// Prepend a character to text.
    #[inline]
    pub fn cons(c: char, text: &Text) -> Text {
        let mut result = Self::singleton(c).unpack();
        result.push_str(text.as_str());
        Self::pack(&result)
    }

    /// Append a character to text.
    #[inline]
    pub fn snoc(text: &Text, c: char) -> Text {
        let mut result = text.unpack();
        result.push(c);
        Self::pack(&result)
    }

    /// Append two texts.
    #[inline]
    pub fn append(a: &Text, b: &Text) -> Text {
        if a.is_empty() {
            return b.clone();
        }
        if b.is_empty() {
            return a.clone();
        }
        let mut result = a.unpack();
        result.push_str(b.as_str());
        Self::pack(&result)
    }

    /// Get the first character and remaining text.
    #[inline]
    pub fn uncons(&self) -> Option<(char, Text)> {
        let mut chars = self.as_str().chars();
        let c = chars.next()?;
        let rest = chars.as_str();
        Some((c, Self::pack(rest)))
    }

    /// Get the last character and preceding text.
    #[inline]
    pub fn unsnoc(&self) -> Option<(Text, char)> {
        let s = self.as_str();
        let c = s.chars().last()?;
        let init = &s[..s.len() - c.len_utf8()];
        Some((Self::pack(init), c))
    }

    // --------------------------------------------------------
    // Basic Operations
    // --------------------------------------------------------

    /// Get the first character.
    #[inline]
    pub fn head(&self) -> Option<char> {
        self.as_str().chars().next()
    }

    /// Get the last character.
    #[inline]
    pub fn last(&self) -> Option<char> {
        self.as_str().chars().last()
    }

    /// Get all but the first character.
    #[inline]
    pub fn tail(&self) -> Option<Text> {
        let s = self.as_str();
        let mut chars = s.chars();
        chars.next()?;
        Some(Self::pack(chars.as_str()))
    }

    /// Get all but the last character.
    #[inline]
    pub fn init(&self) -> Option<Text> {
        let s = self.as_str();
        let c = s.chars().last()?;
        Some(Self::pack(&s[..s.len() - c.len_utf8()]))
    }

    // --------------------------------------------------------
    // Transformations
    // --------------------------------------------------------

    /// Apply a function to each character.
    pub fn map<F>(&self, f: F) -> Text
    where
        F: Fn(char) -> char,
    {
        let result: String = self.as_str().chars().map(f).collect();
        Self::pack(&result)
    }

    /// Reverse the text.
    pub fn reverse(&self) -> Text {
        let result: String = self.as_str().chars().rev().collect();
        Self::pack(&result)
    }

    /// Insert text between elements.
    pub fn intersperse(sep: char, text: &Text) -> Text {
        let s = text.as_str();
        if s.is_empty() {
            return Text::empty();
        }
        let result: String = s
            .chars()
            .enumerate()
            .flat_map(|(i, c)| {
                if i == 0 {
                    vec![c]
                } else {
                    vec![sep, c]
                }
            })
            .collect();
        Self::pack(&result)
    }

    /// Join texts with a separator.
    pub fn intercalate(sep: &Text, texts: &[Text]) -> Text {
        if texts.is_empty() {
            return Text::empty();
        }
        let result = texts
            .iter()
            .map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join(sep.as_str());
        Self::pack(&result)
    }

    /// Replace all occurrences of a pattern.
    pub fn replace(needle: &Text, replacement: &Text, haystack: &Text) -> Text {
        let result = haystack.as_str().replace(needle.as_str(), replacement.as_str());
        Self::pack(&result)
    }

    /// Convert to uppercase.
    pub fn to_upper(&self) -> Text {
        Self::pack(&self.as_str().to_uppercase())
    }

    /// Convert to lowercase.
    pub fn to_lower(&self) -> Text {
        Self::pack(&self.as_str().to_lowercase())
    }

    /// Convert to title case (first char of each word uppercase).
    pub fn to_title(&self) -> Text {
        let mut result = String::with_capacity(self.len);
        let mut make_upper = true;
        for c in self.as_str().chars() {
            if c.is_whitespace() {
                result.push(c);
                make_upper = true;
            } else if make_upper {
                for uc in c.to_uppercase() {
                    result.push(uc);
                }
                make_upper = false;
            } else {
                for lc in c.to_lowercase() {
                    result.push(lc);
                }
            }
        }
        Self::pack(&result)
    }

    /// Case fold for case-insensitive comparison.
    pub fn case_fold(&self) -> Text {
        // Unicode case folding - for simplicity, use lowercase
        // In production, this should use proper Unicode case folding
        self.to_lower()
    }

    // --------------------------------------------------------
    // Folds
    // --------------------------------------------------------

    /// Left fold over characters.
    pub fn foldl<B, F>(&self, init: B, f: F) -> B
    where
        F: Fn(B, char) -> B,
    {
        self.as_str().chars().fold(init, f)
    }

    /// Right fold over characters.
    pub fn foldr<B, F>(&self, init: B, f: F) -> B
    where
        F: Fn(char, B) -> B,
    {
        self.as_str().chars().rev().fold(init, |acc, c| f(c, acc))
    }

    /// Concatenate a list of texts.
    pub fn concat(texts: &[Text]) -> Text {
        let total_len: usize = texts.iter().map(|t| t.len()).sum();
        let mut result = String::with_capacity(total_len);
        for t in texts {
            result.push_str(t.as_str());
        }
        Self::pack(&result)
    }

    /// Map and concatenate.
    pub fn concat_map<F>(&self, f: F) -> Text
    where
        F: Fn(char) -> Text,
    {
        let results: Vec<Text> = self.as_str().chars().map(f).collect();
        Self::concat(&results)
    }

    /// Check if any character satisfies a predicate.
    pub fn any<F>(&self, f: F) -> bool
    where
        F: Fn(char) -> bool,
    {
        self.as_str().chars().any(f)
    }

    /// Check if all characters satisfy a predicate.
    pub fn all<F>(&self, f: F) -> bool
    where
        F: Fn(char) -> bool,
    {
        self.as_str().chars().all(f)
    }

    /// Find maximum character.
    pub fn maximum(&self) -> Option<char> {
        self.as_str().chars().max()
    }

    /// Find minimum character.
    pub fn minimum(&self) -> Option<char> {
        self.as_str().chars().min()
    }

    // --------------------------------------------------------
    // Special Folds
    // --------------------------------------------------------

    /// Check if text is all ASCII.
    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.as_bytes().iter().all(|&b| b < 128)
    }

    /// Check if text is all Latin-1 (first 256 Unicode code points).
    #[inline]
    pub fn is_latin1(&self) -> bool {
        self.as_str().chars().all(|c| c as u32 <= 255)
    }

    // --------------------------------------------------------
    // Substrings
    // --------------------------------------------------------

    /// Take first n characters.
    pub fn take(&self, n: usize) -> Text {
        let s = self.as_str();
        let result: String = s.chars().take(n).collect();
        Self::pack(&result)
    }

    /// Take last n characters.
    pub fn take_end(&self, n: usize) -> Text {
        let s = self.as_str();
        let char_count = s.chars().count();
        if n >= char_count {
            return self.clone();
        }
        let skip = char_count - n;
        let result: String = s.chars().skip(skip).collect();
        Self::pack(&result)
    }

    /// Drop first n characters.
    pub fn drop(&self, n: usize) -> Text {
        let s = self.as_str();
        let result: String = s.chars().skip(n).collect();
        Self::pack(&result)
    }

    /// Drop last n characters.
    pub fn drop_end(&self, n: usize) -> Text {
        let s = self.as_str();
        let char_count = s.chars().count();
        if n >= char_count {
            return Text::empty();
        }
        let take_count = char_count - n;
        let result: String = s.chars().take(take_count).collect();
        Self::pack(&result)
    }

    /// Split at position n.
    pub fn split_at(&self, n: usize) -> (Text, Text) {
        (self.take(n), self.drop(n))
    }

    /// Take characters while predicate holds.
    pub fn take_while<F>(&self, f: F) -> Text
    where
        F: Fn(char) -> bool,
    {
        let result: String = self.as_str().chars().take_while(|&c| f(c)).collect();
        Self::pack(&result)
    }

    /// Take characters from end while predicate holds.
    pub fn take_while_end<F>(&self, f: F) -> Text
    where
        F: Fn(char) -> bool,
    {
        let s = self.as_str();
        let chars: Vec<char> = s.chars().collect();
        let mut i = chars.len();
        while i > 0 && f(chars[i - 1]) {
            i -= 1;
        }
        let result: String = chars[i..].iter().collect();
        Self::pack(&result)
    }

    /// Drop characters while predicate holds.
    pub fn drop_while<F>(&self, f: F) -> Text
    where
        F: Fn(char) -> bool,
    {
        let result: String = self.as_str().chars().skip_while(|&c| f(c)).collect();
        Self::pack(&result)
    }

    /// Drop characters from end while predicate holds.
    pub fn drop_while_end<F>(&self, f: F) -> Text
    where
        F: Fn(char) -> bool,
    {
        let s = self.as_str();
        let chars: Vec<char> = s.chars().collect();
        let mut i = chars.len();
        while i > 0 && f(chars[i - 1]) {
            i -= 1;
        }
        let result: String = chars[..i].iter().collect();
        Self::pack(&result)
    }

    /// Drop whitespace from both ends.
    pub fn strip(&self) -> Text {
        Self::pack(self.as_str().trim())
    }

    /// Drop whitespace from start.
    pub fn strip_start(&self) -> Text {
        Self::pack(self.as_str().trim_start())
    }

    /// Drop whitespace from end.
    pub fn strip_end(&self) -> Text {
        Self::pack(self.as_str().trim_end())
    }

    /// Split at first occurrence of pattern.
    pub fn break_on(&self, needle: &Text) -> (Text, Text) {
        if let Some(pos) = self.as_str().find(needle.as_str()) {
            let (a, b) = self.as_str().split_at(pos);
            (Self::pack(a), Self::pack(b))
        } else {
            (self.clone(), Text::empty())
        }
    }

    /// Split at last occurrence of pattern.
    pub fn break_on_end(&self, needle: &Text) -> (Text, Text) {
        if let Some(pos) = self.as_str().rfind(needle.as_str()) {
            let (a, b) = self.as_str().split_at(pos + needle.len());
            (Self::pack(a), Self::pack(b))
        } else {
            (Text::empty(), self.clone())
        }
    }

    /// Split on predicate.
    pub fn span<F>(&self, f: F) -> (Text, Text)
    where
        F: Fn(char) -> bool,
    {
        let s = self.as_str();
        let mut byte_idx = 0;
        for c in s.chars() {
            if !f(c) {
                break;
            }
            byte_idx += c.len_utf8();
        }
        let (a, b) = s.split_at(byte_idx);
        (Self::pack(a), Self::pack(b))
    }

    // --------------------------------------------------------
    // Breaking into Lines and Words
    // --------------------------------------------------------

    /// Split into lines.
    pub fn lines(&self) -> Vec<Text> {
        self.as_str().lines().map(Self::pack).collect()
    }

    /// Split into words.
    pub fn words(&self) -> Vec<Text> {
        self.as_str().split_whitespace().map(Self::pack).collect()
    }

    /// Join lines with newlines.
    pub fn unlines(texts: &[Text]) -> Text {
        let mut result = String::new();
        for (i, t) in texts.iter().enumerate() {
            if i > 0 {
                result.push('\n');
            }
            result.push_str(t.as_str());
        }
        Self::pack(&result)
    }

    /// Join words with spaces.
    pub fn unwords(texts: &[Text]) -> Text {
        Self::intercalate(&Text::pack(" "), texts)
    }

    // --------------------------------------------------------
    // Predicates
    // --------------------------------------------------------

    /// Check if starts with prefix.
    pub fn is_prefix_of(&self, text: &Text) -> bool {
        text.as_str().starts_with(self.as_str())
    }

    /// Check if ends with suffix.
    pub fn is_suffix_of(&self, text: &Text) -> bool {
        text.as_str().ends_with(self.as_str())
    }

    /// Check if is a substring.
    pub fn is_infix_of(&self, text: &Text) -> bool {
        text.as_str().contains(self.as_str())
    }

    /// Strip prefix if present.
    pub fn strip_prefix(&self, prefix: &Text) -> Option<Text> {
        self.as_str()
            .strip_prefix(prefix.as_str())
            .map(Self::pack)
    }

    /// Strip suffix if present.
    pub fn strip_suffix(&self, suffix: &Text) -> Option<Text> {
        self.as_str()
            .strip_suffix(suffix.as_str())
            .map(Self::pack)
    }

    // --------------------------------------------------------
    // Search
    // --------------------------------------------------------

    /// Filter characters by predicate.
    pub fn filter<F>(&self, f: F) -> Text
    where
        F: Fn(char) -> bool,
    {
        let result: String = self.as_str().chars().filter(|&c| f(c)).collect();
        Self::pack(&result)
    }

    /// Find first character satisfying predicate.
    pub fn find<F>(&self, f: F) -> Option<char>
    where
        F: Fn(char) -> bool,
    {
        self.as_str().chars().find(|&c| f(c))
    }

    /// Check if character is in text.
    pub fn elem(&self, c: char) -> bool {
        self.as_str().contains(c)
    }

    /// Partition by predicate.
    pub fn partition<F>(&self, f: F) -> (Text, Text)
    where
        F: Fn(char) -> bool,
    {
        let mut yes = String::new();
        let mut no = String::new();
        for c in self.as_str().chars() {
            if f(c) {
                yes.push(c);
            } else {
                no.push(c);
            }
        }
        (Self::pack(&yes), Self::pack(&no))
    }

    // --------------------------------------------------------
    // Indexing
    // --------------------------------------------------------

    /// Get character at index (0-based).
    pub fn index(&self, i: usize) -> Option<char> {
        self.as_str().chars().nth(i)
    }

    /// Find index of first character satisfying predicate.
    pub fn find_index<F>(&self, f: F) -> Option<usize>
    where
        F: Fn(char) -> bool,
    {
        self.as_str().chars().position(|c| f(c))
    }

    /// Count occurrences of pattern.
    pub fn count(&self, needle: &Text) -> usize {
        self.as_str().matches(needle.as_str()).count()
    }

    // --------------------------------------------------------
    // Zipping
    // --------------------------------------------------------

    /// Zip two texts into pairs.
    pub fn zip(a: &Text, b: &Text) -> Vec<(char, char)> {
        a.as_str().chars().zip(b.as_str().chars()).collect()
    }

    /// Zip two texts with a function.
    pub fn zip_with<C, F>(f: F, a: &Text, b: &Text) -> Vec<C>
    where
        F: Fn(char, char) -> C,
    {
        a.as_str()
            .chars()
            .zip(b.as_str().chars())
            .map(|(x, y)| f(x, y))
            .collect()
    }
}

// ============================================================
// Trait Implementations
// ============================================================

impl Default for Text {
    fn default() -> Self {
        Self::empty()
    }
}

impl PartialEq for Text {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for Text {}

impl PartialOrd for Text {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Text {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl std::hash::Hash for Text {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl std::fmt::Debug for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Text({:?})", self.as_str())
    }
}

impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for Text {
    fn from(s: &str) -> Self {
        Self::pack(s)
    }
}

impl From<String> for Text {
    fn from(s: String) -> Self {
        Self::pack(&s)
    }
}

impl From<char> for Text {
    fn from(c: char) -> Self {
        Self::singleton(c)
    }
}

// ============================================================
// FFI Functions (for BHC runtime)
// ============================================================

/// Get text length in bytes.
#[no_mangle]
pub extern "C" fn bhc_text_length(ptr: *const u8, len: usize) -> usize {
    len
}

/// Get text length in characters (code points).
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

/// Check if text is ASCII.
#[no_mangle]
pub extern "C" fn bhc_text_is_ascii(ptr: *const u8, len: usize) -> bhc_prelude::bool::Bool {
    if ptr.is_null() || len == 0 {
        return bhc_prelude::bool::Bool::True;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    bhc_prelude::bool::Bool::from_bool(bytes.iter().all(|&b| b < 128))
}

/// Convert text to uppercase.
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
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let s = unsafe { str::from_utf8_unchecked(bytes) };
    let upper = s.to_uppercase();
    let upper_bytes = upper.as_bytes();
    if upper_bytes.len() > out_cap {
        return 0;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(upper_bytes.as_ptr(), out, upper_bytes.len());
    }
    upper_bytes.len()
}

/// Convert text to lowercase.
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

/// Find substring (returns offset or -1).
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
    let haystack = unsafe { std::slice::from_raw_parts(haystack_ptr, haystack_len) };
    let needle = unsafe { std::slice::from_raw_parts(needle_ptr, needle_len) };
    for i in 0..=(haystack_len - needle_len) {
        if &haystack[i..i + needle_len] == needle {
            return i as i64;
        }
    }
    -1
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let t = Text::empty();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.char_count(), 0);
    }

    #[test]
    fn test_pack_unpack() {
        let t = Text::pack("hello");
        assert_eq!(t.len(), 5);
        assert_eq!(t.unpack(), "hello");
        assert_eq!(t.as_str(), "hello");
    }

    #[test]
    fn test_singleton() {
        let t = Text::singleton('a');
        assert_eq!(t.unpack(), "a");

        let t = Text::singleton('世');
        assert_eq!(t.char_count(), 1);
        assert_eq!(t.len(), 3); // UTF-8 encoding
    }

    #[test]
    fn test_unicode() {
        let t = Text::pack("hello 世界");
        assert_eq!(t.char_count(), 8);
        assert!(t.len() > 8); // UTF-8 bytes > char count
    }

    #[test]
    fn test_cons_snoc() {
        let t = Text::pack("ello");
        let t2 = Text::cons('h', &t);
        assert_eq!(t2.unpack(), "hello");

        let t3 = Text::snoc(&t2, '!');
        assert_eq!(t3.unpack(), "hello!");
    }

    #[test]
    fn test_append() {
        let a = Text::pack("hello");
        let b = Text::pack(" world");
        let c = Text::append(&a, &b);
        assert_eq!(c.unpack(), "hello world");
    }

    #[test]
    fn test_uncons_unsnoc() {
        let t = Text::pack("hello");
        let (c, rest) = t.uncons().unwrap();
        assert_eq!(c, 'h');
        assert_eq!(rest.unpack(), "ello");

        let (init, c) = t.unsnoc().unwrap();
        assert_eq!(init.unpack(), "hell");
        assert_eq!(c, 'o');
    }

    #[test]
    fn test_head_last() {
        let t = Text::pack("hello");
        assert_eq!(t.head(), Some('h'));
        assert_eq!(t.last(), Some('o'));

        assert_eq!(Text::empty().head(), None);
        assert_eq!(Text::empty().last(), None);
    }

    #[test]
    fn test_tail_init() {
        let t = Text::pack("hello");
        assert_eq!(t.tail().unwrap().unpack(), "ello");
        assert_eq!(t.init().unwrap().unpack(), "hell");
    }

    #[test]
    fn test_map() {
        let t = Text::pack("hello");
        let upper = t.map(|c| c.to_ascii_uppercase());
        assert_eq!(upper.unpack(), "HELLO");
    }

    #[test]
    fn test_reverse() {
        let t = Text::pack("hello");
        assert_eq!(t.reverse().unpack(), "olleh");

        let t = Text::pack("a世b");
        assert_eq!(t.reverse().unpack(), "b世a");
    }

    #[test]
    fn test_intersperse() {
        let t = Text::pack("hello");
        assert_eq!(Text::intersperse(',', &t).unpack(), "h,e,l,l,o");
    }

    #[test]
    fn test_intercalate() {
        let texts = vec![
            Text::pack("hello"),
            Text::pack("world"),
            Text::pack("foo"),
        ];
        let sep = Text::pack(", ");
        assert_eq!(Text::intercalate(&sep, &texts).unpack(), "hello, world, foo");
    }

    #[test]
    fn test_replace() {
        let t = Text::pack("hello world");
        let result = Text::replace(&Text::pack("world"), &Text::pack("rust"), &t);
        assert_eq!(result.unpack(), "hello rust");
    }

    #[test]
    fn test_case_conversion() {
        let t = Text::pack("Hello World");
        assert_eq!(t.to_upper().unpack(), "HELLO WORLD");
        assert_eq!(t.to_lower().unpack(), "hello world");
        assert_eq!(t.to_title().unpack(), "Hello World");

        let t2 = Text::pack("hello world");
        assert_eq!(t2.to_title().unpack(), "Hello World");
    }

    #[test]
    fn test_folds() {
        let t = Text::pack("abc");
        let sum = t.foldl(0u32, |acc, c| acc + c as u32);
        assert_eq!(sum, 97 + 98 + 99);

        assert!(t.all(|c| c.is_alphabetic()));
        assert!(t.any(|c| c == 'b'));
        assert!(!t.any(|c| c == 'z'));
    }

    #[test]
    fn test_take_drop() {
        let t = Text::pack("hello");
        assert_eq!(t.take(3).unpack(), "hel");
        assert_eq!(t.drop(3).unpack(), "lo");
        assert_eq!(t.take_end(3).unpack(), "llo");
        assert_eq!(t.drop_end(3).unpack(), "he");
    }

    #[test]
    fn test_split_at() {
        let t = Text::pack("hello");
        let (a, b) = t.split_at(3);
        assert_eq!(a.unpack(), "hel");
        assert_eq!(b.unpack(), "lo");
    }

    #[test]
    fn test_strip() {
        let t = Text::pack("  hello  ");
        assert_eq!(t.strip().unpack(), "hello");
        assert_eq!(t.strip_start().unpack(), "hello  ");
        assert_eq!(t.strip_end().unpack(), "  hello");
    }

    #[test]
    fn test_lines_words() {
        let t = Text::pack("hello\nworld\nfoo");
        let lines = t.lines();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].unpack(), "hello");

        let t2 = Text::pack("hello world foo");
        let words = t2.words();
        assert_eq!(words.len(), 3);
        assert_eq!(words[1].unpack(), "world");

        let rejoined = Text::unlines(&lines);
        assert_eq!(rejoined.unpack(), "hello\nworld\nfoo");

        let reworded = Text::unwords(&words);
        assert_eq!(reworded.unpack(), "hello world foo");
    }

    #[test]
    fn test_predicates() {
        let hello = Text::pack("hello");
        let prefix = Text::pack("hel");
        let suffix = Text::pack("llo");
        let infix = Text::pack("ell");

        assert!(prefix.is_prefix_of(&hello));
        assert!(suffix.is_suffix_of(&hello));
        assert!(infix.is_infix_of(&hello));
        assert!(!suffix.is_prefix_of(&hello));
    }

    #[test]
    fn test_filter_find() {
        let t = Text::pack("hello123");
        let letters = t.filter(|c| c.is_alphabetic());
        assert_eq!(letters.unpack(), "hello");

        let digit = t.find(|c| c.is_ascii_digit());
        assert_eq!(digit, Some('1'));

        assert!(t.elem('e'));
        assert!(!t.elem('z'));
    }

    #[test]
    fn test_index() {
        let t = Text::pack("hello");
        assert_eq!(t.index(0), Some('h'));
        assert_eq!(t.index(4), Some('o'));
        assert_eq!(t.index(5), None);
    }

    #[test]
    fn test_is_ascii_latin1() {
        assert!(Text::pack("hello").is_ascii());
        assert!(!Text::pack("héllo").is_ascii());
        assert!(Text::pack("héllo").is_latin1());
        assert!(!Text::pack("hello世界").is_latin1());
    }

    #[test]
    fn test_zip() {
        let a = Text::pack("abc");
        let b = Text::pack("123");
        let pairs = Text::zip(&a, &b);
        assert_eq!(pairs, vec![('a', '1'), ('b', '2'), ('c', '3')]);
    }

    #[test]
    fn test_count() {
        let t = Text::pack("abracadabra");
        assert_eq!(t.count(&Text::pack("a")), 5);
        assert_eq!(t.count(&Text::pack("abra")), 2);
    }

    // FFI tests
    #[test]
    fn test_ffi_char_count() {
        let s = "hello";
        assert_eq!(bhc_text_char_count(s.as_ptr(), s.len()), 5);

        let s = "hello 世界";
        assert_eq!(bhc_text_char_count(s.as_ptr(), s.len()), 8);
    }

    #[test]
    fn test_ffi_find() {
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
