//! Source location tracking and span management for BHC.
//!
//! This crate provides types for tracking source locations throughout
//! the compilation pipeline, enabling accurate error reporting and
//! source mapping.

#![warn(missing_docs)]

use serde::{Deserialize, Serialize};

/// A byte offset into a source file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct BytePos(pub u32);

impl BytePos {
    /// The zero position.
    pub const ZERO: Self = Self(0);

    /// Create a new byte position.
    #[must_use]
    pub const fn new(pos: u32) -> Self {
        Self(pos)
    }

    /// Get the raw byte offset.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Get the raw byte offset as usize.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl std::ops::Add<u32> for BytePos {
    type Output = Self;

    fn add(self, rhs: u32) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl std::ops::Sub for BytePos {
    type Output = u32;

    fn sub(self, rhs: Self) -> Self::Output {
        self.0 - rhs.0
    }
}

/// A span of source code, represented as a half-open byte range [lo, hi).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    /// The start of the span (inclusive).
    pub lo: BytePos,
    /// The end of the span (exclusive).
    pub hi: BytePos,
}

impl Span {
    /// A dummy span for generated code or when location is irrelevant.
    pub const DUMMY: Self = Self {
        lo: BytePos::ZERO,
        hi: BytePos::ZERO,
    };

    /// Create a new span from byte positions.
    #[must_use]
    pub const fn new(lo: BytePos, hi: BytePos) -> Self {
        Self { lo, hi }
    }

    /// Create a span from raw byte offsets.
    #[must_use]
    pub const fn from_raw(lo: u32, hi: u32) -> Self {
        Self {
            lo: BytePos(lo),
            hi: BytePos(hi),
        }
    }

    /// Check if this is a dummy span.
    #[must_use]
    pub const fn is_dummy(self) -> bool {
        self.lo.0 == 0 && self.hi.0 == 0
    }

    /// Get the length of the span in bytes.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.hi.0 - self.lo.0
    }

    /// Check if the span is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.lo.0 == self.hi.0
    }

    /// Merge two spans into one that covers both.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            lo: BytePos(self.lo.0.min(other.lo.0)),
            hi: BytePos(self.hi.0.max(other.hi.0)),
        }
    }

    /// Create a span that covers from the start of self to the end of other.
    #[must_use]
    pub const fn to(self, other: Self) -> Self {
        Self {
            lo: self.lo,
            hi: other.hi,
        }
    }

    /// Shrink the span to a single point at the start.
    #[must_use]
    pub const fn shrink_to_lo(self) -> Self {
        Self {
            lo: self.lo,
            hi: self.lo,
        }
    }

    /// Shrink the span to a single point at the end.
    #[must_use]
    pub const fn shrink_to_hi(self) -> Self {
        Self {
            lo: self.hi,
            hi: self.hi,
        }
    }

    /// Check if this span contains the given byte position.
    #[must_use]
    pub const fn contains(self, pos: BytePos) -> bool {
        self.lo.0 <= pos.0 && pos.0 < self.hi.0
    }
}

impl Default for Span {
    fn default() -> Self {
        Self::DUMMY
    }
}

/// A value with an associated span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    /// The value.
    pub node: T,
    /// The span of the value in source code.
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Create a new spanned value.
    #[must_use]
    pub const fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }

    /// Map the inner value while preserving the span.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Spanned<U> {
        Spanned {
            node: f(self.node),
            span: self.span,
        }
    }
}

/// A unique identifier for a source file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct FileId(pub u32);

impl FileId {
    /// Create a new file ID.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

/// A span with an associated file ID for cross-file spans.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FullSpan {
    /// The file this span belongs to.
    pub file: FileId,
    /// The span within the file.
    pub span: Span,
}

impl FullSpan {
    /// Create a new full span.
    #[must_use]
    pub const fn new(file: FileId, span: Span) -> Self {
        Self { file, span }
    }
}

/// Line and column information for a source location.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LineCol {
    /// 1-indexed line number.
    pub line: u32,
    /// 1-indexed column number (in UTF-8 code units).
    pub col: u32,
}

impl LineCol {
    /// Create a new line/column pair.
    #[must_use]
    pub const fn new(line: u32, col: u32) -> Self {
        Self { line, col }
    }
}

/// Information about a source file.
#[derive(Clone, Debug)]
pub struct SourceFile {
    /// The file ID.
    pub id: FileId,
    /// The file name or path.
    pub name: String,
    /// The source code content.
    pub src: String,
    /// Byte offsets of line starts.
    line_starts: Vec<BytePos>,
}

impl SourceFile {
    /// Create a new source file.
    #[must_use]
    pub fn new(id: FileId, name: String, src: String) -> Self {
        let line_starts = std::iter::once(BytePos::ZERO)
            .chain(
                src.match_indices('\n')
                    .map(|(i, _)| BytePos::new(i as u32 + 1)),
            )
            .collect();

        Self {
            id,
            name,
            src,
            line_starts,
        }
    }

    /// Get the line/column for a byte position.
    #[must_use]
    pub fn lookup_line_col(&self, pos: BytePos) -> LineCol {
        let line_idx = self
            .line_starts
            .partition_point(|&start| start.0 <= pos.0)
            .saturating_sub(1);

        let line_start = self.line_starts[line_idx];
        let col = pos.0 - line_start.0 + 1;

        LineCol {
            line: line_idx as u32 + 1,
            col,
        }
    }

    /// Get the 0-indexed line number for a byte position.
    #[must_use]
    pub fn lookup_line(&self, pos: BytePos) -> usize {
        self.line_starts
            .partition_point(|&start| start.0 <= pos.0)
            .saturating_sub(1)
    }

    /// Get the source text for a span.
    #[must_use]
    pub fn source_text(&self, span: Span) -> &str {
        &self.src[span.lo.as_usize()..span.hi.as_usize()]
    }

    /// Get the number of lines in the file.
    #[must_use]
    pub fn num_lines(&self) -> usize {
        self.line_starts.len()
    }

    /// Get the content of a specific line (0-indexed).
    #[must_use]
    pub fn line_content(&self, line_idx: usize) -> Option<&str> {
        if line_idx >= self.line_starts.len() {
            return None;
        }

        let start = self.line_starts[line_idx].as_usize();
        let end = if line_idx + 1 < self.line_starts.len() {
            // Next line start minus 1 to exclude the newline
            self.line_starts[line_idx + 1].as_usize().saturating_sub(1)
        } else {
            self.src.len()
        };

        Some(&self.src[start..end])
    }

    /// Get the byte offset of the start of a line (0-indexed).
    #[must_use]
    pub fn line_start(&self, line_idx: usize) -> Option<BytePos> {
        self.line_starts.get(line_idx).copied()
    }

    /// Get span information for rendering: start line, start col, end line, end col.
    ///
    /// Note: Spans are half-open `[lo, hi)`, so the end position is computed
    /// from the last included byte (`hi - 1`) for non-empty spans.
    #[must_use]
    pub fn span_lines(&self, span: Span) -> SpanLines {
        let start = self.lookup_line_col(span.lo);
        // For the end, use the last included byte (hi - 1) for non-empty spans
        let end = if span.hi.0 > span.lo.0 {
            self.lookup_line_col(BytePos(span.hi.0 - 1))
        } else {
            start
        };
        SpanLines {
            start_line: start.line as usize,
            start_col: start.col as usize,
            end_line: end.line as usize,
            // For end column, add 1 since we want the position after the last char
            end_col: end.col as usize + 1,
        }
    }
}

/// Information about which lines a span covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanLines {
    /// 1-indexed start line.
    pub start_line: usize,
    /// 1-indexed start column.
    pub start_col: usize,
    /// 1-indexed end line.
    pub end_line: usize,
    /// 1-indexed end column.
    pub end_col: usize,
}

impl SpanLines {
    /// Check if this span covers multiple lines.
    #[must_use]
    pub fn is_multiline(&self) -> bool {
        self.start_line != self.end_line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_operations() {
        let span1 = Span::from_raw(10, 20);
        let span2 = Span::from_raw(15, 30);

        assert_eq!(span1.len(), 10);
        assert_eq!(span1.merge(span2), Span::from_raw(10, 30));
        assert!(span1.contains(BytePos::new(15)));
        assert!(!span1.contains(BytePos::new(25)));
    }

    #[test]
    fn test_source_file_line_lookup() {
        let src = "line 1\nline 2\nline 3";
        let file = SourceFile::new(FileId::new(0), "test.hs".to_string(), src.to_string());

        assert_eq!(file.lookup_line_col(BytePos::new(0)), LineCol::new(1, 1));
        assert_eq!(file.lookup_line_col(BytePos::new(7)), LineCol::new(2, 1));
        assert_eq!(file.lookup_line_col(BytePos::new(10)), LineCol::new(2, 4));
    }

    #[test]
    fn test_line_content() {
        let src = "first line\nsecond line\nthird line";
        let file = SourceFile::new(FileId::new(0), "test.hs".to_string(), src.to_string());

        assert_eq!(file.line_content(0), Some("first line"));
        assert_eq!(file.line_content(1), Some("second line"));
        assert_eq!(file.line_content(2), Some("third line"));
        assert_eq!(file.line_content(3), None);
    }

    #[test]
    fn test_span_lines() {
        let src = "line 1\nline 2\nline 3";
        let file = SourceFile::new(FileId::new(0), "test.hs".to_string(), src.to_string());

        // Single line span
        let span = Span::from_raw(0, 6);
        let lines = file.span_lines(span);
        assert_eq!(lines.start_line, 1);
        assert_eq!(lines.end_line, 1);
        assert!(!lines.is_multiline());

        // Multi-line span
        let span = Span::from_raw(0, 14);
        let lines = file.span_lines(span);
        assert_eq!(lines.start_line, 1);
        assert_eq!(lines.end_line, 2);
        assert!(lines.is_multiline());
    }
}
