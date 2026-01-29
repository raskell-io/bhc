//! Analysis engine for parsing and type checking.
//!
//! This module handles parsing source code and producing diagnostics.
//!
//! Note: This is a stub implementation that provides the interface.
//! Full integration with bhc parsing requires aligning with the specific
//! crate APIs which may change.

use dashmap::DashMap;
use lsp_types::{Position, Range, Uri};
use std::sync::Arc;

/// Result of analyzing a document.
#[derive(Clone, Debug, Default)]
pub struct AnalysisResult {
    /// Diagnostics from parsing and type checking.
    pub diagnostics: Vec<lsp_types::Diagnostic>,
    /// Symbol table.
    pub symbols: Vec<Symbol>,
}

/// A symbol in the document.
#[derive(Clone, Debug)]
pub struct Symbol {
    /// Symbol name.
    pub name: String,
    /// Symbol kind.
    pub kind: SymbolKind,
    /// Location range.
    pub range: Range,
    /// Selection range (for the name itself).
    pub selection_range: Range,
    /// Children symbols.
    pub children: Vec<Symbol>,
    /// Type signature (if available).
    pub type_sig: Option<String>,
    /// Documentation.
    pub documentation: Option<String>,
}

/// Kind of symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolKind {
    /// Module.
    Module,
    /// Function.
    Function,
    /// Variable.
    Variable,
    /// Type.
    Type,
    /// Type class.
    Class,
    /// Constructor.
    Constructor,
    /// Field.
    Field,
    /// Type alias.
    TypeAlias,
    /// Pattern.
    Pattern,
}

impl From<SymbolKind> for lsp_types::SymbolKind {
    fn from(kind: SymbolKind) -> Self {
        match kind {
            SymbolKind::Module => lsp_types::SymbolKind::MODULE,
            SymbolKind::Function => lsp_types::SymbolKind::FUNCTION,
            SymbolKind::Variable => lsp_types::SymbolKind::VARIABLE,
            SymbolKind::Type => lsp_types::SymbolKind::STRUCT,
            SymbolKind::Class => lsp_types::SymbolKind::INTERFACE,
            SymbolKind::Constructor => lsp_types::SymbolKind::CONSTRUCTOR,
            SymbolKind::Field => lsp_types::SymbolKind::FIELD,
            SymbolKind::TypeAlias => lsp_types::SymbolKind::TYPE_PARAMETER,
            SymbolKind::Pattern => lsp_types::SymbolKind::CONSTANT,
        }
    }
}

/// Analysis engine for documents.
pub struct AnalysisEngine {
    /// Cached analysis results.
    cache: DashMap<String, Arc<AnalysisResult>>,
}

impl AnalysisEngine {
    /// Create a new analysis engine.
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
        }
    }

    /// Analyze a document and return diagnostics.
    pub fn analyze(&self, content: &str, uri: &Uri) -> Vec<lsp_types::Diagnostic> {
        let result = self.full_analysis(content, uri);
        let key = uri.to_string();
        let diagnostics = result.diagnostics.clone();
        self.cache.insert(key, Arc::new(result));
        diagnostics
    }

    /// Get cached analysis result.
    pub fn get_cached(&self, uri: &Uri) -> Option<Arc<AnalysisResult>> {
        let key = uri.to_string();
        self.cache.get(&key).map(|r| Arc::clone(&r))
    }

    /// Perform full analysis.
    fn full_analysis(&self, content: &str, _uri: &Uri) -> AnalysisResult {
        let diagnostics = Vec::new();
        let mut symbols = Vec::new();

        // Extract module name from content
        let module_name = extract_module_name(content).unwrap_or("Main");

        // Add module symbol
        symbols.push(Symbol {
            name: module_name.to_string(),
            kind: SymbolKind::Module,
            range: Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: module_name.len() as u32 + 7,
                },
            },
            selection_range: Range {
                start: Position {
                    line: 0,
                    character: 7,
                },
                end: Position {
                    line: 0,
                    character: module_name.len() as u32 + 7,
                },
            },
            children: Vec::new(),
            type_sig: None,
            documentation: None,
        });

        // Simple symbol extraction (top-level definitions)
        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("--") || trimmed.starts_with("{-") {
                continue;
            }

            // Function definitions: name arg1 arg2 = ...
            if let Some(name) = extract_function_name(trimmed) {
                if !name.starts_with("module") && !name.starts_with("import") && !is_keyword(&name)
                {
                    let start_col = line.find(&name).unwrap_or(0);
                    symbols.push(Symbol {
                        name: name.clone(),
                        kind: SymbolKind::Function,
                        range: Range {
                            start: Position {
                                line: line_num as u32,
                                character: 0,
                            },
                            end: Position {
                                line: line_num as u32,
                                character: line.len() as u32,
                            },
                        },
                        selection_range: Range {
                            start: Position {
                                line: line_num as u32,
                                character: start_col as u32,
                            },
                            end: Position {
                                line: line_num as u32,
                                character: (start_col + name.len()) as u32,
                            },
                        },
                        children: Vec::new(),
                        type_sig: None,
                        documentation: None,
                    });
                }
            }

            // Type signatures: name :: Type
            if trimmed.contains(" :: ") {
                if let Some((name, type_sig)) = trimmed.split_once(" :: ") {
                    let name = name.trim();
                    if is_valid_identifier(name) {
                        let start_col = line.find(name).unwrap_or(0);
                        symbols.push(Symbol {
                            name: name.to_string(),
                            kind: SymbolKind::Function,
                            range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: 0,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: line.len() as u32,
                                },
                            },
                            selection_range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: start_col as u32,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: (start_col + name.len()) as u32,
                                },
                            },
                            children: Vec::new(),
                            type_sig: Some(type_sig.trim().to_string()),
                            documentation: None,
                        });
                    }
                }
            }

            // Data types: data TypeName = ...
            if trimmed.starts_with("data ") {
                if let Some(rest) = trimmed.strip_prefix("data ") {
                    let name = rest.split_whitespace().next().unwrap_or("");
                    if is_valid_type_name(name) {
                        let start_col = line.find(name).unwrap_or(0);
                        symbols.push(Symbol {
                            name: name.to_string(),
                            kind: SymbolKind::Type,
                            range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: 0,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: line.len() as u32,
                                },
                            },
                            selection_range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: start_col as u32,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: (start_col + name.len()) as u32,
                                },
                            },
                            children: Vec::new(),
                            type_sig: None,
                            documentation: None,
                        });
                    }
                }
            }

            // Type aliases: type TypeName = ...
            if trimmed.starts_with("type ") {
                if let Some(rest) = trimmed.strip_prefix("type ") {
                    let name = rest.split_whitespace().next().unwrap_or("");
                    if is_valid_type_name(name) {
                        let start_col = line.find(name).unwrap_or(0);
                        symbols.push(Symbol {
                            name: name.to_string(),
                            kind: SymbolKind::TypeAlias,
                            range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: 0,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: line.len() as u32,
                                },
                            },
                            selection_range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: start_col as u32,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: (start_col + name.len()) as u32,
                                },
                            },
                            children: Vec::new(),
                            type_sig: None,
                            documentation: None,
                        });
                    }
                }
            }

            // Classes: class ClassName a where ...
            if trimmed.starts_with("class ") {
                if let Some(rest) = trimmed.strip_prefix("class ") {
                    let name = rest.split_whitespace().next().unwrap_or("");
                    if is_valid_type_name(name) {
                        let start_col = line.find(name).unwrap_or(0);
                        symbols.push(Symbol {
                            name: name.to_string(),
                            kind: SymbolKind::Class,
                            range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: 0,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: line.len() as u32,
                                },
                            },
                            selection_range: Range {
                                start: Position {
                                    line: line_num as u32,
                                    character: start_col as u32,
                                },
                                end: Position {
                                    line: line_num as u32,
                                    character: (start_col + name.len()) as u32,
                                },
                            },
                            children: Vec::new(),
                            type_sig: None,
                            documentation: None,
                        });
                    }
                }
            }
        }

        AnalysisResult {
            diagnostics,
            symbols,
        }
    }
}

/// Extract module name from content.
fn extract_module_name(content: &str) -> Option<&str> {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("module ") {
            let rest = trimmed.strip_prefix("module ")?;
            return rest.split_whitespace().next();
        }
    }
    None
}

/// Extract function name from a line.
fn extract_function_name(line: &str) -> Option<String> {
    let trimmed = line.trim();

    // Skip if it's a keyword line
    if trimmed.starts_with("module ")
        || trimmed.starts_with("import ")
        || trimmed.starts_with("data ")
        || trimmed.starts_with("type ")
        || trimmed.starts_with("newtype ")
        || trimmed.starts_with("class ")
        || trimmed.starts_with("instance ")
        || trimmed.starts_with("where")
        || trimmed.starts_with("let ")
        || trimmed.starts_with("in ")
        || trimmed.starts_with("|")
        || trimmed.starts_with("=")
    {
        return None;
    }

    // Look for pattern: identifier (args)* = ...
    // or: identifier (args)* | guard = ...
    let first_word = trimmed.split_whitespace().next()?;

    if is_valid_identifier(first_word) && (trimmed.contains('=') || trimmed.contains(" | ")) {
        return Some(first_word.to_string());
    }

    None
}

/// Check if a string is a valid Haskell identifier.
fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let first = s.chars().next().unwrap();
    if !first.is_ascii_lowercase() && first != '_' {
        return false;
    }

    s.chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '\'')
}

/// Check if a string is a valid type name.
fn is_valid_type_name(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let first = s.chars().next().unwrap();
    if !first.is_ascii_uppercase() {
        return false;
    }

    s.chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '\'')
}

/// Check if a string is a Haskell keyword.
fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "case"
            | "class"
            | "data"
            | "default"
            | "deriving"
            | "do"
            | "else"
            | "foreign"
            | "if"
            | "import"
            | "in"
            | "infix"
            | "infixl"
            | "infixr"
            | "instance"
            | "let"
            | "module"
            | "newtype"
            | "of"
            | "then"
            | "type"
            | "where"
            | "_"
    )
}

impl Default for AnalysisEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_engine_creation() {
        let engine = AnalysisEngine::new();
        assert!(engine.cache.is_empty());
    }

    #[test]
    fn test_is_valid_identifier() {
        assert!(is_valid_identifier("foo"));
        assert!(is_valid_identifier("fooBar"));
        assert!(is_valid_identifier("foo'"));
        assert!(is_valid_identifier("_foo"));
        assert!(!is_valid_identifier("Foo"));
        assert!(!is_valid_identifier("123"));
        assert!(!is_valid_identifier(""));
    }

    #[test]
    fn test_is_valid_type_name() {
        assert!(is_valid_type_name("Foo"));
        assert!(is_valid_type_name("FooBar"));
        assert!(!is_valid_type_name("foo"));
        assert!(!is_valid_type_name(""));
    }

    #[test]
    fn test_extract_module_name() {
        assert_eq!(extract_module_name("module Test where"), Some("Test"));
        assert_eq!(
            extract_module_name("module Data.List where"),
            Some("Data.List")
        );
        assert_eq!(
            extract_module_name("-- comment\nmodule Main where"),
            Some("Main")
        );
        assert_eq!(extract_module_name("no module here"), None);
    }
}
