//! Type search engine.
//!
//! This module implements type-based search for documentation. It allows
//! searching by:
//!
//! - Name (fuzzy matching)
//! - Type signature (with unification)
//!
//! The type search uses fingerprinting and unification to match queries
//! like `a -> [a] -> [a]` against functions like `cons :: a -> [a] -> [a]`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Search configuration.
pub struct SearchConfig {
    /// Documentation directory.
    pub dir: std::path::PathBuf,
    /// Search query.
    pub query: String,
    /// Enable type search (unify type variables).
    pub type_search: bool,
    /// Maximum results.
    pub limit: usize,
}

/// A search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Module name.
    pub module: String,
    /// Item name.
    pub name: String,
    /// Item kind.
    pub kind: String,
    /// Type signature (if any).
    pub signature: Option<String>,
    /// Brief documentation.
    pub doc: Option<String>,
    /// Relevance score (higher is better).
    pub score: f64,
}

impl fmt::Display for SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.module, self.name)?;
        if let Some(sig) = &self.signature {
            write!(f, " :: {}", sig)?;
        }
        if let Some(doc) = &self.doc {
            write!(f, "\n    {}", doc)?;
        }
        Ok(())
    }
}

/// Run a search query.
pub fn run(config: SearchConfig) -> Result<Vec<SearchResult>> {
    let index_path = config.dir.join("search-index.json");
    let index_json = std::fs::read_to_string(&index_path)?;
    let index: Vec<serde_json::Value> = serde_json::from_str(&index_json)?;

    let mut results = Vec::new();

    for entry in index {
        let name = entry["name"].as_str().unwrap_or("");
        let module = entry["module"].as_str().unwrap_or("");
        let kind = entry["kind"].as_str().unwrap_or("");
        let signature = entry["signature"].as_str().map(String::from);
        let doc = entry["doc"].as_str().map(String::from);

        let score = if config.type_search && signature.is_some() {
            // Type-based search
            match_type(&config.query, signature.as_deref().unwrap())
        } else {
            // Name-based search
            match_name(&config.query, name)
        };

        if score > 0.0 {
            results.push(SearchResult {
                module: module.to_string(),
                name: name.to_string(),
                kind: kind.to_string(),
                signature,
                doc,
                score,
            });
        }
    }

    // Sort by score (descending)
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Limit results
    results.truncate(config.limit);

    Ok(results)
}

/// Match a name against a query (fuzzy).
fn match_name(query: &str, name: &str) -> f64 {
    let query = query.to_lowercase();
    let name_lower = name.to_lowercase();

    if name_lower == query {
        return 1.0;
    }

    if name_lower.starts_with(&query) {
        return 0.9;
    }

    if name_lower.contains(&query) {
        return 0.7;
    }

    // Simple fuzzy matching
    let mut score = 0.0;
    let mut query_chars = query.chars().peekable();
    let mut consecutive = 0;

    for c in name_lower.chars() {
        if let Some(&qc) = query_chars.peek() {
            if c == qc {
                query_chars.next();
                consecutive += 1;
                score += 0.1 + (consecutive as f64 * 0.05);
            } else {
                consecutive = 0;
            }
        }
    }

    if query_chars.peek().is_none() {
        score / query.len() as f64
    } else {
        0.0
    }
}

/// Match a type signature against a query type.
fn match_type(query: &str, signature: &str) -> f64 {
    let query_ty = parse_simple_type(query);
    let sig_ty = parse_simple_type(signature);

    if unify(&query_ty, &sig_ty) {
        1.0
    } else if partial_match(&query_ty, &sig_ty) {
        0.5
    } else {
        0.0
    }
}

/// Simple type representation for matching.
#[derive(Debug, Clone, PartialEq)]
enum SimpleType {
    Var(String),
    Con(String),
    Arrow(Box<SimpleType>, Box<SimpleType>),
    App(Box<SimpleType>, Box<SimpleType>),
    List(Box<SimpleType>),
    Tuple(Vec<SimpleType>),
}

/// Parse a simple type from a string.
fn parse_simple_type(s: &str) -> SimpleType {
    let s = s.trim();

    // Arrow
    if let Some((left, right)) = split_arrow(s) {
        return SimpleType::Arrow(
            Box::new(parse_simple_type(left)),
            Box::new(parse_simple_type(right)),
        );
    }

    // List
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        return SimpleType::List(Box::new(parse_simple_type(inner)));
    }

    // Tuple
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        if inner.contains(',') {
            let parts: Vec<_> = split_comma(inner)
                .iter()
                .map(|p| parse_simple_type(p))
                .collect();
            return SimpleType::Tuple(parts);
        }
        // Parenthesized type
        return parse_simple_type(inner);
    }

    // Application
    if let Some((func, arg)) = split_app(s) {
        return SimpleType::App(
            Box::new(parse_simple_type(func)),
            Box::new(parse_simple_type(arg)),
        );
    }

    // Variable or constructor
    if s.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
        SimpleType::Var(s.to_string())
    } else {
        SimpleType::Con(s.to_string())
    }
}

/// Split at the outermost `->`.
fn split_arrow(s: &str) -> Option<(&str, &str)> {
    let mut depth = 0;
    let bytes = s.as_bytes();

    for i in 0..bytes.len().saturating_sub(1) {
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'-' if depth == 0 && bytes.get(i + 1) == Some(&b'>') => {
                return Some((s[..i].trim(), s[i + 2..].trim()));
            }
            _ => {}
        }
    }

    None
}

/// Split by comma at depth 0.
fn split_comma(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }

    parts.push(s[start..].trim());
    parts
}

/// Split at the first space (for type application).
fn split_app(s: &str) -> Option<(&str, &str)> {
    let mut depth = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ' ' if depth == 0 => {
                return Some((s[..i].trim(), s[i + 1..].trim()));
            }
            _ => {}
        }
    }

    None
}

/// Unify two types (allowing variable binding).
fn unify(query: &SimpleType, target: &SimpleType) -> bool {
    use std::collections::HashMap;
    let mut bindings: HashMap<String, SimpleType> = HashMap::new();
    unify_with_bindings(query, target, &mut bindings)
}

fn unify_with_bindings(
    query: &SimpleType,
    target: &SimpleType,
    bindings: &mut std::collections::HashMap<String, SimpleType>,
) -> bool {
    match (query, target) {
        (SimpleType::Var(v), t) => {
            if let Some(bound) = bindings.get(v) {
                bound == t
            } else {
                bindings.insert(v.clone(), t.clone());
                true
            }
        }
        (SimpleType::Con(a), SimpleType::Con(b)) => a == b,
        (SimpleType::Arrow(f1, t1), SimpleType::Arrow(f2, t2)) => {
            unify_with_bindings(f1, f2, bindings) && unify_with_bindings(t1, t2, bindings)
        }
        (SimpleType::App(f1, a1), SimpleType::App(f2, a2)) => {
            unify_with_bindings(f1, f2, bindings) && unify_with_bindings(a1, a2, bindings)
        }
        (SimpleType::List(e1), SimpleType::List(e2)) => unify_with_bindings(e1, e2, bindings),
        (SimpleType::Tuple(es1), SimpleType::Tuple(es2)) if es1.len() == es2.len() => es1
            .iter()
            .zip(es2.iter())
            .all(|(a, b)| unify_with_bindings(a, b, bindings)),
        _ => false,
    }
}

/// Check for partial match (some structure matches).
fn partial_match(query: &SimpleType, target: &SimpleType) -> bool {
    // Check if target contains query as a subtype
    if unify(query, target) {
        return true;
    }

    match target {
        SimpleType::Arrow(f, t) => partial_match(query, f) || partial_match(query, t),
        SimpleType::App(f, a) => partial_match(query, f) || partial_match(query, a),
        SimpleType::List(e) => partial_match(query, e),
        SimpleType::Tuple(es) => es.iter().any(|e| partial_match(query, e)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_parsing() {
        let ty = parse_simple_type("a -> [a] -> [a]");
        assert!(matches!(ty, SimpleType::Arrow(_, _)));
    }

    #[test]
    fn test_unification() {
        let query = parse_simple_type("a -> [a] -> [a]");
        let target = parse_simple_type("Int -> [Int] -> [Int]");
        assert!(unify(&query, &target));
    }

    #[test]
    fn test_name_matching() {
        assert!(match_name("map", "map") > 0.9);
        assert!(match_name("map", "fmap") > 0.5);
        assert!(match_name("xyz", "abc") == 0.0);
    }
}
