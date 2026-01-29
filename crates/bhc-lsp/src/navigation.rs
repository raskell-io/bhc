//! Navigation (go to definition, find references).
//!
//! This module provides navigation features like go to definition
//! and find all references.

use crate::analysis::{AnalysisResult, Symbol};
use crate::document::Document;
use lsp_types::{Location, Position, Range, Uri};

/// Find the definition of a symbol at a position.
pub fn find_definition(
    doc: &Document,
    analysis: &AnalysisResult,
    position: Position,
) -> Option<Location> {
    // Find the word at the position
    let word = doc.word_at(position)?;

    // Search for the symbol definition
    for symbol in &analysis.symbols {
        if let Some(location) = find_definition_in_symbol(&word, symbol, &doc.uri) {
            return Some(location);
        }
    }

    None
}

/// Search for a definition within a symbol tree.
fn find_definition_in_symbol(name: &str, symbol: &Symbol, uri: &Uri) -> Option<Location> {
    // Check if this symbol matches
    if symbol.name == name {
        return Some(Location {
            uri: uri.clone(),
            range: symbol.selection_range.clone(),
        });
    }

    // Check children
    for child in &symbol.children {
        if let Some(loc) = find_definition_in_symbol(name, child, uri) {
            return Some(loc);
        }
    }

    None
}

/// Find all references to a symbol at a position.
pub fn find_references(
    doc: &Document,
    _analysis: &AnalysisResult,
    position: Position,
    uri: &Uri,
) -> Vec<Location> {
    let mut locations = Vec::new();

    // Find the word at the position
    let word = match doc.word_at(position) {
        Some(w) => w,
        None => return locations,
    };

    // Find all occurrences of the word in the document
    let content = doc.text();
    for (line_num, line) in content.lines().enumerate() {
        let mut col = 0;
        for word_match in line.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '\'') {
            if word_match == word {
                locations.push(Location {
                    uri: uri.clone(),
                    range: Range {
                        start: Position {
                            line: line_num as u32,
                            character: col as u32,
                        },
                        end: Position {
                            line: line_num as u32,
                            character: (col + word_match.len()) as u32,
                        },
                    },
                });
            }
            // Move past the word and the delimiter
            col += word_match.len();
            // Account for the delimiter
            if col < line.len() {
                col += 1;
            }
        }
    }

    locations
}

/// Find all symbols matching a pattern.
pub fn find_symbols_matching(analysis: &AnalysisResult, pattern: &str, uri: &Uri) -> Vec<Location> {
    let mut locations = Vec::new();
    let pattern_lower = pattern.to_lowercase();

    fn search_symbol(symbol: &Symbol, pattern: &str, uri: &Uri, locations: &mut Vec<Location>) {
        if symbol.name.to_lowercase().contains(pattern) {
            locations.push(Location {
                uri: uri.clone(),
                range: symbol.selection_range.clone(),
            });
        }

        for child in &symbol.children {
            search_symbol(child, pattern, uri, locations);
        }
    }

    for symbol in &analysis.symbols {
        search_symbol(symbol, &pattern_lower, uri, &mut locations);
    }

    locations
}

/// Check if a position is at a definition.
pub fn is_at_definition(analysis: &AnalysisResult, position: Position) -> bool {
    for symbol in &analysis.symbols {
        if in_range(position, &symbol.selection_range) {
            return true;
        }
        for child in &symbol.children {
            if in_range(position, &child.selection_range) {
                return true;
            }
        }
    }
    false
}

/// Check if a position is within a range.
fn in_range(pos: Position, range: &Range) -> bool {
    if pos.line < range.start.line || pos.line > range.end.line {
        return false;
    }

    if pos.line == range.start.line && pos.character < range.start.character {
        return false;
    }

    if pos.line == range.end.line && pos.character > range.end.character {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn test_uri() -> Uri {
        "file:///test.hs".parse().unwrap()
    }

    #[test]
    fn test_in_range() {
        let range = Range {
            start: Position {
                line: 1,
                character: 5,
            },
            end: Position {
                line: 1,
                character: 10,
            },
        };

        assert!(in_range(
            Position {
                line: 1,
                character: 5
            },
            &range
        ));
        assert!(in_range(
            Position {
                line: 1,
                character: 7
            },
            &range
        ));
        assert!(in_range(
            Position {
                line: 1,
                character: 10
            },
            &range
        ));
        assert!(!in_range(
            Position {
                line: 1,
                character: 4
            },
            &range
        ));
        assert!(!in_range(
            Position {
                line: 1,
                character: 11
            },
            &range
        ));
    }
}
