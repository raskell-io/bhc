//! Document management.
//!
//! This module handles open documents and their content.

use dashmap::DashMap;
use lsp_types::{Position, Range, Uri};
use ropey::Rope;

/// A managed document.
#[derive(Debug)]
pub struct Document {
    /// Document URI.
    pub uri: Uri,
    /// Document content as rope for efficient editing.
    pub content: Rope,
    /// Document version.
    pub version: i32,
}

impl Document {
    /// Create a new document.
    pub fn new(uri: Uri, content: String, version: i32) -> Self {
        Self {
            uri,
            content: Rope::from_str(&content),
            version,
        }
    }

    /// Get the full text content.
    pub fn text(&self) -> String {
        self.content.to_string()
    }

    /// Get text in a range.
    pub fn text_range(&self, range: Range) -> Option<String> {
        let start = self.position_to_offset(range.start)?;
        let end = self.position_to_offset(range.end)?;

        if start > end || end > self.content.len_chars() {
            return None;
        }

        Some(self.content.slice(start..end).to_string())
    }

    /// Convert LSP position to rope offset.
    pub fn position_to_offset(&self, pos: Position) -> Option<usize> {
        let line = pos.line as usize;
        if line >= self.content.len_lines() {
            return None;
        }

        let line_start = self.content.line_to_char(line);
        let line_len = self.content.line(line).len_chars();
        let col = pos.character as usize;

        if col > line_len {
            // Allow position at end of line
            Some(line_start + line_len)
        } else {
            Some(line_start + col)
        }
    }

    /// Convert rope offset to LSP position.
    pub fn offset_to_position(&self, offset: usize) -> Option<Position> {
        if offset > self.content.len_chars() {
            return None;
        }

        let line = self.content.char_to_line(offset);
        let line_start = self.content.line_to_char(line);
        let col = offset - line_start;

        Some(Position {
            line: line as u32,
            character: col as u32,
        })
    }

    /// Apply a text change.
    pub fn apply_change(&mut self, range: Range, new_text: &str) {
        if let (Some(start), Some(end)) = (
            self.position_to_offset(range.start),
            self.position_to_offset(range.end),
        ) {
            self.content.remove(start..end);
            self.content.insert(start, new_text);
        }
    }

    /// Get the number of lines.
    pub fn line_count(&self) -> usize {
        self.content.len_lines()
    }

    /// Get a specific line.
    pub fn line(&self, line_num: usize) -> Option<String> {
        if line_num >= self.content.len_lines() {
            return None;
        }
        Some(self.content.line(line_num).to_string())
    }

    /// Get the word at a position.
    pub fn word_at(&self, pos: Position) -> Option<String> {
        let offset = self.position_to_offset(pos)?;
        let text = self.content.to_string();
        let chars: Vec<char> = text.chars().collect();

        if offset >= chars.len() {
            return None;
        }

        // Find word boundaries
        let mut start = offset;
        while start > 0 && is_word_char(chars[start - 1]) {
            start -= 1;
        }

        let mut end = offset;
        while end < chars.len() && is_word_char(chars[end]) {
            end += 1;
        }

        if start == end {
            return None;
        }

        Some(chars[start..end].iter().collect())
    }
}

/// Check if a character is part of a word.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '\''
}

/// Manages open documents.
pub struct DocumentManager {
    /// Open documents.
    documents: DashMap<Uri, Document>,
}

impl DocumentManager {
    /// Create a new document manager.
    pub fn new() -> Self {
        Self {
            documents: DashMap::new(),
        }
    }

    /// Open a document.
    pub fn open(&self, uri: Uri, content: String, version: i32) {
        let doc = Document::new(uri.clone(), content, version);
        self.documents.insert(uri, doc);
    }

    /// Update a document.
    pub fn update(&self, uri: &Uri, content: String, version: i32) {
        if let Some(mut doc) = self.documents.get_mut(uri) {
            doc.content = Rope::from_str(&content);
            doc.version = version;
        }
    }

    /// Apply a change to a document.
    pub fn apply_change(&self, uri: &Uri, range: Range, new_text: &str) {
        if let Some(mut doc) = self.documents.get_mut(uri) {
            doc.apply_change(range, new_text);
        }
    }

    /// Close a document.
    pub fn close(&self, uri: &Uri) {
        self.documents.remove(uri);
    }

    /// Get document content.
    pub fn get_content(&self, uri: &Uri) -> Option<String> {
        self.documents.get(uri).map(|doc| doc.text())
    }

    /// Get a document.
    pub fn get(&self, uri: &Uri) -> Option<dashmap::mapref::one::Ref<'_, Uri, Document>> {
        self.documents.get(uri)
    }

    /// Check if a document is open.
    pub fn is_open(&self, uri: &Uri) -> bool {
        self.documents.contains_key(uri)
    }

    /// Get all open document URIs.
    pub fn open_documents(&self) -> Vec<Uri> {
        self.documents.iter().map(|r| r.key().clone()).collect()
    }

    /// Get the number of open documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if no documents are open.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

impl Default for DocumentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_uri() -> Uri {
        "file:///test.hs".parse().unwrap()
    }

    #[test]
    fn test_document_creation() {
        let doc = Document::new(
            test_uri(),
            "module Test where\n\nmain = putStrLn \"Hello\"\n".to_string(),
            1,
        );

        assert_eq!(doc.line_count(), 4);
        assert_eq!(doc.line(0), Some("module Test where\n".to_string()));
    }

    #[test]
    fn test_position_to_offset() {
        let doc = Document::new(test_uri(), "line 1\nline 2\nline 3\n".to_string(), 1);

        assert_eq!(
            doc.position_to_offset(Position {
                line: 0,
                character: 0
            }),
            Some(0)
        );
        assert_eq!(
            doc.position_to_offset(Position {
                line: 0,
                character: 4
            }),
            Some(4)
        );
        assert_eq!(
            doc.position_to_offset(Position {
                line: 1,
                character: 0
            }),
            Some(7)
        );
        assert_eq!(
            doc.position_to_offset(Position {
                line: 2,
                character: 0
            }),
            Some(14)
        );
    }

    #[test]
    fn test_offset_to_position() {
        let doc = Document::new(test_uri(), "line 1\nline 2\nline 3\n".to_string(), 1);

        assert_eq!(
            doc.offset_to_position(0),
            Some(Position {
                line: 0,
                character: 0
            })
        );
        assert_eq!(
            doc.offset_to_position(4),
            Some(Position {
                line: 0,
                character: 4
            })
        );
        assert_eq!(
            doc.offset_to_position(7),
            Some(Position {
                line: 1,
                character: 0
            })
        );
    }

    #[test]
    fn test_apply_change() {
        let mut doc = Document::new(test_uri(), "hello world".to_string(), 1);

        doc.apply_change(
            Range {
                start: Position {
                    line: 0,
                    character: 6,
                },
                end: Position {
                    line: 0,
                    character: 11,
                },
            },
            "BHC",
        );

        assert_eq!(doc.text(), "hello BHC");
    }

    #[test]
    fn test_word_at() {
        let doc = Document::new(test_uri(), "let foo = bar".to_string(), 1);

        assert_eq!(
            doc.word_at(Position {
                line: 0,
                character: 5
            }),
            Some("foo".to_string())
        );
        assert_eq!(
            doc.word_at(Position {
                line: 0,
                character: 10
            }),
            Some("bar".to_string())
        );
    }

    #[test]
    fn test_document_manager() {
        let manager = DocumentManager::new();
        let uri = test_uri();

        manager.open(uri.clone(), "test content".to_string(), 1);
        assert!(manager.is_open(&uri));
        assert_eq!(manager.len(), 1);

        assert_eq!(manager.get_content(&uri), Some("test content".to_string()));

        manager.close(&uri);
        assert!(!manager.is_open(&uri));
        assert!(manager.is_empty());
    }
}
