//! Symbol management.
//!
//! This module provides document symbols and workspace symbols.

use crate::analysis::{AnalysisEngine, AnalysisResult, Symbol, SymbolKind};
use crate::document::DocumentManager;
use lsp_types::{DocumentSymbol, Location, Uri, WorkspaceSymbol};

/// Get document symbols from analysis results.
pub fn document_symbols(analysis: &AnalysisResult) -> Vec<DocumentSymbol> {
    analysis
        .symbols
        .iter()
        .map(|s| symbol_to_document_symbol(s))
        .collect()
}

/// Convert an analysis symbol to a document symbol.
fn symbol_to_document_symbol(symbol: &Symbol) -> DocumentSymbol {
    #[allow(deprecated)]
    DocumentSymbol {
        name: symbol.name.clone(),
        detail: symbol.type_sig.clone(),
        kind: symbol.kind.into(),
        tags: None,
        deprecated: None,
        range: symbol.range.clone(),
        selection_range: symbol.selection_range.clone(),
        children: if symbol.children.is_empty() {
            None
        } else {
            Some(
                symbol
                    .children
                    .iter()
                    .map(|c| symbol_to_document_symbol(c))
                    .collect(),
            )
        },
    }
}

/// Search for workspace symbols.
pub fn workspace_symbols(
    documents: &DocumentManager,
    analysis: &AnalysisEngine,
    query: &str,
) -> Vec<WorkspaceSymbol> {
    let mut results = Vec::new();
    let query_lower = query.to_lowercase();

    for uri in documents.open_documents() {
        if let Some(result) = analysis.get_cached(&uri) {
            for symbol in &result.symbols {
                if matches_query(&symbol.name, &query_lower) {
                    results.push(symbol_to_workspace_symbol(symbol, &uri));
                }

                // Check children
                for child in &symbol.children {
                    if matches_query(&child.name, &query_lower) {
                        results.push(symbol_to_workspace_symbol(child, &uri));
                    }
                }
            }
        }
    }

    // Sort by relevance
    results.sort_by(|a, b| {
        // Exact matches first
        let a_exact = a.name.to_lowercase() == query_lower;
        let b_exact = b.name.to_lowercase() == query_lower;

        if a_exact && !b_exact {
            std::cmp::Ordering::Less
        } else if !a_exact && b_exact {
            std::cmp::Ordering::Greater
        } else {
            // Then prefix matches
            let a_prefix = a.name.to_lowercase().starts_with(&query_lower);
            let b_prefix = b.name.to_lowercase().starts_with(&query_lower);

            if a_prefix && !b_prefix {
                std::cmp::Ordering::Less
            } else if !a_prefix && b_prefix {
                std::cmp::Ordering::Greater
            } else {
                a.name.cmp(&b.name)
            }
        }
    });

    results
}

/// Check if a symbol name matches a query.
fn matches_query(name: &str, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }

    let name_lower = name.to_lowercase();

    // Substring match
    if name_lower.contains(query) {
        return true;
    }

    // Fuzzy match (all query chars appear in order)
    let mut query_chars = query.chars().peekable();
    for c in name_lower.chars() {
        if let Some(&qc) = query_chars.peek() {
            if c == qc {
                query_chars.next();
            }
        }
    }

    query_chars.peek().is_none()
}

/// Convert a symbol to a workspace symbol.
fn symbol_to_workspace_symbol(symbol: &Symbol, uri: &Uri) -> WorkspaceSymbol {
    #[allow(deprecated)]
    WorkspaceSymbol {
        name: symbol.name.clone(),
        kind: symbol.kind.into(),
        tags: None,
        container_name: None,
        location: lsp_types::OneOf::Left(Location {
            uri: uri.clone(),
            range: symbol.selection_range.clone(),
        }),
        data: None,
    }
}

/// Symbol outline for a document.
pub struct SymbolOutline {
    /// Top-level symbols.
    pub symbols: Vec<OutlineSymbol>,
}

/// A symbol in the outline.
pub struct OutlineSymbol {
    /// Symbol name.
    pub name: String,
    /// Symbol kind.
    pub kind: SymbolKind,
    /// Line number.
    pub line: u32,
    /// Children.
    pub children: Vec<OutlineSymbol>,
}

impl SymbolOutline {
    /// Create an outline from analysis results.
    pub fn from_analysis(analysis: &AnalysisResult) -> Self {
        let symbols = analysis
            .symbols
            .iter()
            .map(|s| Self::symbol_to_outline(s))
            .collect();

        Self { symbols }
    }

    fn symbol_to_outline(symbol: &Symbol) -> OutlineSymbol {
        OutlineSymbol {
            name: symbol.name.clone(),
            kind: symbol.kind,
            line: symbol.range.start.line,
            children: symbol
                .children
                .iter()
                .map(|c| Self::symbol_to_outline(c))
                .collect(),
        }
    }

    /// Get symbols at a line.
    pub fn symbols_at_line(&self, line: u32) -> Vec<&OutlineSymbol> {
        let mut result = Vec::new();

        fn find_at_line<'a>(
            symbols: &'a [OutlineSymbol],
            line: u32,
            result: &mut Vec<&'a OutlineSymbol>,
        ) {
            for symbol in symbols {
                if symbol.line == line {
                    result.push(symbol);
                }
                find_at_line(&symbol.children, line, result);
            }
        }

        find_at_line(&self.symbols, line, &mut result);
        result
    }
}

/// Get the symbol path (breadcrumbs) at a position.
pub fn symbol_path(analysis: &AnalysisResult, line: u32) -> Vec<String> {
    let mut path = Vec::new();

    fn find_containing<'a>(symbols: &'a [Symbol], line: u32, path: &mut Vec<String>) -> bool {
        for symbol in symbols {
            if symbol.range.start.line <= line && symbol.range.end.line >= line {
                path.push(symbol.name.clone());

                // Check children for more specific match
                if !find_containing(&symbol.children, line, path) {
                    // No more specific child found
                }

                return true;
            }
        }
        false
    }

    find_containing(&analysis.symbols, line, &mut path);
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use lsp_types::{Position, Range};

    fn test_symbol(name: &str, kind: SymbolKind) -> Symbol {
        Symbol {
            name: name.to_string(),
            kind,
            range: Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: 10,
                },
            },
            selection_range: Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: name.len() as u32,
                },
            },
            children: Vec::new(),
            type_sig: None,
            documentation: None,
        }
    }

    #[test]
    fn test_matches_query() {
        assert!(matches_query("fooBar", "foo"));
        assert!(matches_query("fooBar", "bar"));
        assert!(matches_query("fooBar", "fb")); // Fuzzy
        assert!(matches_query("fooBar", ""));
        assert!(!matches_query("foo", "bar"));
    }

    #[test]
    fn test_symbol_to_document_symbol() {
        let symbol = test_symbol("test", SymbolKind::Function);
        let doc_symbol = symbol_to_document_symbol(&symbol);

        assert_eq!(doc_symbol.name, "test");
        assert_eq!(doc_symbol.kind, lsp_types::SymbolKind::FUNCTION);
    }
}
