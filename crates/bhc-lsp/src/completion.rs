//! Code completion.
//!
//! This module provides code completion suggestions.

use crate::analysis::{AnalysisEngine, SymbolKind};
use crate::config::CompletionConfig;
use crate::document::Document;
use lsp_types::{
    CompletionItem, CompletionItemKind, Documentation, InsertTextFormat, MarkupContent, MarkupKind,
    Position,
};
use std::sync::Arc;

/// Provide completion items at a position.
pub fn complete(
    doc: &Document,
    analysis: &Arc<AnalysisEngine>,
    position: Position,
    config: &CompletionConfig,
) -> Vec<CompletionItem> {
    if !config.enabled {
        return Vec::new();
    }

    let mut items = Vec::new();

    // Get the prefix being typed
    let prefix = get_completion_prefix(doc, position);

    // Add keyword completions
    items.extend(keyword_completions(&prefix));

    // Add builtin completions
    items.extend(builtin_completions(&prefix));

    // Add symbols from analysis
    if let Some(result) = analysis.get_cached(&doc.uri) {
        for symbol in &result.symbols {
            if symbol.name.starts_with(&prefix) {
                items.push(symbol_to_completion(symbol));
            }

            // Check children
            for child in &symbol.children {
                if child.name.starts_with(&prefix) {
                    items.push(symbol_to_completion(child));
                }
            }
        }
    }

    // Add snippet completions
    if config.snippets {
        items.extend(snippet_completions(&prefix));
    }

    // Sort and limit results
    items.sort_by(|a, b| {
        // Prioritize exact prefix matches
        let a_exact = a.label.starts_with(&prefix);
        let b_exact = b.label.starts_with(&prefix);

        if a_exact && !b_exact {
            std::cmp::Ordering::Less
        } else if !a_exact && b_exact {
            std::cmp::Ordering::Greater
        } else {
            a.label.cmp(&b.label)
        }
    });

    items.truncate(config.max_results);
    items
}

/// Get the completion prefix at a position.
fn get_completion_prefix(doc: &Document, position: Position) -> String {
    // Get the word being typed
    doc.word_at(position).unwrap_or_default()
}

/// Convert a symbol to a completion item.
fn symbol_to_completion(symbol: &crate::analysis::Symbol) -> CompletionItem {
    CompletionItem {
        label: symbol.name.clone(),
        kind: Some(symbol_kind_to_completion_kind(symbol.kind)),
        detail: symbol.type_sig.clone(),
        documentation: symbol.documentation.as_ref().map(|doc| {
            Documentation::MarkupContent(MarkupContent {
                kind: MarkupKind::Markdown,
                value: doc.clone(),
            })
        }),
        ..Default::default()
    }
}

/// Convert symbol kind to completion item kind.
fn symbol_kind_to_completion_kind(kind: SymbolKind) -> CompletionItemKind {
    match kind {
        SymbolKind::Module => CompletionItemKind::MODULE,
        SymbolKind::Function => CompletionItemKind::FUNCTION,
        SymbolKind::Variable => CompletionItemKind::VARIABLE,
        SymbolKind::Type => CompletionItemKind::STRUCT,
        SymbolKind::Class => CompletionItemKind::INTERFACE,
        SymbolKind::Constructor => CompletionItemKind::CONSTRUCTOR,
        SymbolKind::Field => CompletionItemKind::FIELD,
        SymbolKind::TypeAlias => CompletionItemKind::TYPE_PARAMETER,
        SymbolKind::Pattern => CompletionItemKind::CONSTANT,
    }
}

/// Keyword completions.
fn keyword_completions(prefix: &str) -> Vec<CompletionItem> {
    let keywords = [
        ("module", "Module declaration"),
        ("import", "Import declaration"),
        ("qualified", "Qualified import"),
        ("as", "Import alias"),
        ("hiding", "Hide imports"),
        ("where", "Where clause"),
        ("let", "Let binding"),
        ("in", "In clause"),
        ("if", "If expression"),
        ("then", "Then branch"),
        ("else", "Else branch"),
        ("case", "Case expression"),
        ("of", "Case alternatives"),
        ("do", "Do notation"),
        ("data", "Data type declaration"),
        ("newtype", "Newtype declaration"),
        ("type", "Type alias"),
        ("class", "Type class declaration"),
        ("instance", "Type class instance"),
        ("deriving", "Derive instances"),
        ("infixl", "Left-associative infix"),
        ("infixr", "Right-associative infix"),
        ("infix", "Non-associative infix"),
        ("forall", "Universal quantification"),
    ];

    keywords
        .iter()
        .filter(|(kw, _)| kw.starts_with(prefix))
        .map(|(kw, doc)| CompletionItem {
            label: kw.to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            detail: Some(doc.to_string()),
            ..Default::default()
        })
        .collect()
}

/// Builtin function completions.
fn builtin_completions(prefix: &str) -> Vec<CompletionItem> {
    let builtins = [
        (
            "map",
            "(a -> b) -> [a] -> [b]",
            "Apply a function to each element",
        ),
        (
            "filter",
            "(a -> Bool) -> [a] -> [a]",
            "Filter elements by predicate",
        ),
        (
            "foldl",
            "(b -> a -> b) -> b -> [a] -> b",
            "Left-associative fold",
        ),
        (
            "foldr",
            "(a -> b -> b) -> b -> [a] -> b",
            "Right-associative fold",
        ),
        ("sum", "Num a => [a] -> a", "Sum of a list"),
        ("product", "Num a => [a] -> a", "Product of a list"),
        ("length", "[a] -> Int", "Length of a list"),
        ("head", "[a] -> a", "First element (partial)"),
        ("tail", "[a] -> [a]", "All but first (partial)"),
        ("init", "[a] -> [a]", "All but last (partial)"),
        ("last", "[a] -> a", "Last element (partial)"),
        ("take", "Int -> [a] -> [a]", "Take first n elements"),
        ("drop", "Int -> [a] -> [a]", "Drop first n elements"),
        ("reverse", "[a] -> [a]", "Reverse a list"),
        ("concat", "[[a]] -> [a]", "Concatenate lists"),
        (
            "concatMap",
            "(a -> [b]) -> [a] -> [b]",
            "Map and concatenate",
        ),
        ("zip", "[a] -> [b] -> [(a, b)]", "Zip two lists"),
        (
            "zipWith",
            "(a -> b -> c) -> [a] -> [b] -> [c]",
            "Zip with a function",
        ),
        ("unzip", "[(a, b)] -> ([a], [b])", "Unzip a list of pairs"),
        ("elem", "Eq a => a -> [a] -> Bool", "Check membership"),
        (
            "notElem",
            "Eq a => a -> [a] -> Bool",
            "Check non-membership",
        ),
        (
            "lookup",
            "Eq a => a -> [(a, b)] -> Maybe b",
            "Lookup in association list",
        ),
        ("null", "[a] -> Bool", "Check if empty"),
        ("and", "[Bool] -> Bool", "Conjunction of list"),
        ("or", "[Bool] -> Bool", "Disjunction of list"),
        (
            "any",
            "(a -> Bool) -> [a] -> Bool",
            "Any element satisfies predicate",
        ),
        (
            "all",
            "(a -> Bool) -> [a] -> Bool",
            "All elements satisfy predicate",
        ),
        ("maximum", "Ord a => [a] -> a", "Maximum element"),
        ("minimum", "Ord a => [a] -> a", "Minimum element"),
        ("putStrLn", "String -> IO ()", "Print with newline"),
        ("putStr", "String -> IO ()", "Print without newline"),
        ("print", "Show a => a -> IO ()", "Print a value"),
        ("getLine", "IO String", "Read a line"),
        ("readFile", "FilePath -> IO String", "Read file contents"),
        (
            "writeFile",
            "FilePath -> String -> IO ()",
            "Write file contents",
        ),
        ("pure", "Applicative f => a -> f a", "Lift a value"),
        ("return", "Monad m => a -> m a", "Return a value"),
        (
            "fmap",
            "Functor f => (a -> b) -> f a -> f b",
            "Map over functor",
        ),
        ("id", "a -> a", "Identity function"),
        ("const", "a -> b -> a", "Constant function"),
        ("flip", "(a -> b -> c) -> b -> a -> c", "Flip arguments"),
        ("undefined", "a", "Bottom value (undefined)"),
        ("error", "String -> a", "Throw an error"),
    ];

    builtins
        .iter()
        .filter(|(name, _, _)| name.starts_with(prefix))
        .map(|(name, ty, doc)| CompletionItem {
            label: name.to_string(),
            kind: Some(CompletionItemKind::FUNCTION),
            detail: Some(ty.to_string()),
            documentation: Some(Documentation::String(doc.to_string())),
            ..Default::default()
        })
        .collect()
}

/// Snippet completions.
fn snippet_completions(prefix: &str) -> Vec<CompletionItem> {
    let snippets = [
        (
            "main",
            "main :: IO ()\nmain = ${1:undefined}",
            "Main function",
        ),
        (
            "module",
            "module ${1:Module} where\n\n$0",
            "Module declaration",
        ),
        ("import", "import ${1:Module}", "Import declaration"),
        (
            "importq",
            "import qualified ${1:Module} as ${2:M}",
            "Qualified import",
        ),
        ("data", "data ${1:Type} = ${2:Constructor}", "Data type"),
        (
            "newtype",
            "newtype ${1:Type} = ${2:Constructor} ${3:WrappedType}",
            "Newtype",
        ),
        (
            "class",
            "class ${1:Class} ${2:a} where\n  ${0}",
            "Type class",
        ),
        (
            "instance",
            "instance ${1:Class} ${2:Type} where\n  ${0}",
            "Type class instance",
        ),
        (
            "case",
            "case ${1:expr} of\n  ${2:pattern} -> ${0}",
            "Case expression",
        ),
        (
            "if",
            "if ${1:condition}\n  then ${2:thenBranch}\n  else ${0:elseBranch}",
            "If expression",
        ),
        ("let", "let ${1:x} = ${2:expr}\nin ${0}", "Let binding"),
        ("where", "where\n  ${1:x} = ${0}", "Where clause"),
        ("do", "do\n  ${0}", "Do notation"),
        ("lambda", "\\${1:x} -> ${0}", "Lambda expression"),
    ];

    snippets
        .iter()
        .filter(|(name, _, _)| name.starts_with(prefix))
        .map(|(name, snippet, doc)| CompletionItem {
            label: name.to_string(),
            kind: Some(CompletionItemKind::SNIPPET),
            detail: Some(doc.to_string()),
            insert_text: Some(snippet.to_string()),
            insert_text_format: Some(InsertTextFormat::SNIPPET),
            ..Default::default()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_completions() {
        let items = keyword_completions("mo");
        assert!(items.iter().any(|i| i.label == "module"));
    }

    #[test]
    fn test_builtin_completions() {
        let items = builtin_completions("ma");
        assert!(items.iter().any(|i| i.label == "map"));
        assert!(items.iter().any(|i| i.label == "maximum"));
    }

    #[test]
    fn test_snippet_completions() {
        let items = snippet_completions("mai");
        assert!(items.iter().any(|i| i.label == "main"));
    }
}
