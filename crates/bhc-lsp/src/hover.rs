//! Hover information.
//!
//! This module provides hover information for symbols.

use crate::analysis::{AnalysisResult, Symbol, SymbolKind};
use crate::document::Document;
use lsp_types::{Hover, HoverContents, MarkupContent, MarkupKind, Position, Range};

/// Get hover information for a position.
pub fn hover_info(doc: &Document, analysis: &AnalysisResult, position: Position) -> Option<Hover> {
    // Find the word at the position
    let word = doc.word_at(position)?;

    // Look for the symbol in the analysis results
    for symbol in &analysis.symbols {
        if let Some(hover) = symbol_hover(&word, symbol, position) {
            return Some(hover);
        }

        // Check children
        for child in &symbol.children {
            if let Some(hover) = symbol_hover(&word, child, position) {
                return Some(hover);
            }
        }
    }

    // If we didn't find a symbol, provide basic word info
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: format!("`{}`", word),
        }),
        range: None,
    })
}

/// Get hover info for a specific symbol.
fn symbol_hover(word: &str, symbol: &Symbol, position: Position) -> Option<Hover> {
    // Check if the position is within the symbol's range
    if !in_range(position, &symbol.selection_range) {
        return None;
    }

    // Check if the word matches the symbol name
    if symbol.name != word {
        return None;
    }

    let mut content = String::new();

    // Add symbol kind
    content.push_str(&format!("**{}**\n\n", kind_name(symbol.kind)));

    // Add name with type signature
    if let Some(ref type_sig) = symbol.type_sig {
        content.push_str(&format!(
            "```haskell\n{} :: {}\n```\n",
            symbol.name, type_sig
        ));
    } else {
        content.push_str(&format!("```haskell\n{}\n```\n", symbol.name));
    }

    // Add documentation
    if let Some(ref doc) = symbol.documentation {
        content.push_str("\n---\n\n");
        content.push_str(doc);
    }

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: content,
        }),
        range: Some(symbol.selection_range.clone()),
    })
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

/// Get the display name for a symbol kind.
fn kind_name(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::Module => "Module",
        SymbolKind::Function => "Function",
        SymbolKind::Variable => "Variable",
        SymbolKind::Type => "Type",
        SymbolKind::Class => "Type Class",
        SymbolKind::Constructor => "Constructor",
        SymbolKind::Field => "Field",
        SymbolKind::TypeAlias => "Type Alias",
        SymbolKind::Pattern => "Pattern",
    }
}

/// Built-in type information.
pub fn builtin_hover(name: &str) -> Option<Hover> {
    let info = match name {
        "Int" => "```haskell\ndata Int\n```\n\nFixed-precision integer type.",
        "Integer" => "```haskell\ndata Integer\n```\n\nArbitrary-precision integer type.",
        "Float" => "```haskell\ndata Float\n```\n\nSingle-precision floating point.",
        "Double" => "```haskell\ndata Double\n```\n\nDouble-precision floating point.",
        "Bool" => "```haskell\ndata Bool = False | True\n```\n\nBoolean type.",
        "Char" => "```haskell\ndata Char\n```\n\nUnicode character type.",
        "String" => "```haskell\ntype String = [Char]\n```\n\nString type (list of characters).",
        "Maybe" => "```haskell\ndata Maybe a = Nothing | Just a\n```\n\nOptional value type.",
        "Either" => "```haskell\ndata Either a b = Left a | Right b\n```\n\nSum type for error handling.",
        "IO" => "```haskell\ndata IO a\n```\n\nI/O action type.",
        "map" => "```haskell\nmap :: (a -> b) -> [a] -> [b]\n```\n\nApply a function to each element of a list.",
        "filter" => "```haskell\nfilter :: (a -> Bool) -> [a] -> [a]\n```\n\nFilter elements that satisfy a predicate.",
        "foldl" => "```haskell\nfoldl :: (b -> a -> b) -> b -> [a] -> b\n```\n\nLeft-associative fold.",
        "foldr" => "```haskell\nfoldr :: (a -> b -> b) -> b -> [a] -> b\n```\n\nRight-associative fold.",
        "sum" => "```haskell\nsum :: Num a => [a] -> a\n```\n\nSum of a list.",
        "product" => "```haskell\nproduct :: Num a => [a] -> a\n```\n\nProduct of a list.",
        "length" => "```haskell\nlength :: [a] -> Int\n```\n\nLength of a list.",
        "head" => "```haskell\nhead :: [a] -> a\n```\n\nFirst element of a list (partial).",
        "tail" => "```haskell\ntail :: [a] -> [a]\n```\n\nAll but the first element (partial).",
        "putStrLn" => "```haskell\nputStrLn :: String -> IO ()\n```\n\nPrint a string with newline.",
        "print" => "```haskell\nprint :: Show a => a -> IO ()\n```\n\nPrint a value.",
        _ => return None,
    };

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: info.to_string(),
        }),
        range: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
                character: 7
            },
            &range
        ));
        assert!(!in_range(
            Position {
                line: 0,
                character: 7
            },
            &range
        ));
        assert!(!in_range(
            Position {
                line: 1,
                character: 3
            },
            &range
        ));
        assert!(!in_range(
            Position {
                line: 1,
                character: 12
            },
            &range
        ));
    }

    #[test]
    fn test_builtin_hover() {
        assert!(builtin_hover("Int").is_some());
        assert!(builtin_hover("map").is_some());
        assert!(builtin_hover("unknown_thing").is_none());
    }
}
