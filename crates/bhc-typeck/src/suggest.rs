//! Suggestion engine for "Did you mean?" hints.
//!
//! This module provides functionality for suggesting similar names when
//! a user references an undefined variable or misspells an identifier.
//!
//! Uses Levenshtein edit distance for fuzzy matching.

use bhc_intern::Symbol;

/// Maximum edit distance to consider for suggestions.
const MAX_EDIT_DISTANCE: usize = 3;

/// Minimum name length to suggest (avoid suggesting single-char names).
const MIN_SUGGESTION_LENGTH: usize = 2;

/// Compute the Levenshtein edit distance between two strings.
///
/// Returns the minimum number of single-character edits (insertions,
/// deletions, or substitutions) required to transform `a` into `b`.
#[must_use]
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_len = a.chars().count();
    let b_len = b.chars().count();

    // Optimize for common cases
    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }
    if a == b {
        return 0;
    }

    // Use two-row optimization for space efficiency
    let mut prev_row: Vec<usize> = (0..=b_len).collect();
    let mut curr_row: Vec<usize> = vec![0; b_len + 1];

    for (i, a_char) in a.chars().enumerate() {
        curr_row[0] = i + 1;

        for (j, b_char) in b.chars().enumerate() {
            let cost = if a_char == b_char { 0 } else { 1 };
            curr_row[j + 1] = (prev_row[j + 1] + 1) // deletion
                .min(curr_row[j] + 1) // insertion
                .min(prev_row[j] + cost); // substitution
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[b_len]
}

/// A suggestion with its edit distance.
#[derive(Clone, Debug)]
pub struct Suggestion {
    /// The suggested name.
    pub name: Symbol,
    /// The edit distance from the original.
    pub distance: usize,
}

/// Find similar names from a list of candidates.
///
/// Returns suggestions sorted by edit distance (closest first).
#[must_use]
pub fn find_similar_names(target: &str, candidates: &[Symbol]) -> Vec<Suggestion> {
    let target_lower = target.to_lowercase();
    let mut suggestions: Vec<Suggestion> = candidates
        .iter()
        .filter_map(|candidate| {
            let name = candidate.as_str();
            // Skip very short names
            if name.len() < MIN_SUGGESTION_LENGTH {
                return None;
            }

            // Calculate distance (case-insensitive for comparison)
            let name_lower = name.to_lowercase();
            let distance = edit_distance(&target_lower, &name_lower);

            // Only include if within threshold and reasonable
            if distance <= MAX_EDIT_DISTANCE && distance < target.len() {
                Some(Suggestion {
                    name: *candidate,
                    distance,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort by distance (closest first), then alphabetically for ties
    suggestions.sort_by(|a, b| {
        a.distance
            .cmp(&b.distance)
            .then_with(|| a.name.cmp(&b.name))
    });

    // Return top suggestions (limit to avoid overwhelming user)
    suggestions.truncate(3);
    suggestions
}

/// Check if a name looks like a typo of another name.
///
/// This is more lenient than `find_similar_names` for single suggestions.
#[must_use]
pub fn is_likely_typo(typed: &str, candidate: &str) -> bool {
    let distance = edit_distance(&typed.to_lowercase(), &candidate.to_lowercase());

    // More lenient for longer names
    let threshold = match typed.len() {
        0..=2 => 1,
        3..=5 => 2,
        _ => 3,
    };

    distance <= threshold && distance > 0
}

/// Format suggestions as a human-readable string.
#[must_use]
pub fn format_suggestions(suggestions: &[Suggestion]) -> Option<String> {
    if suggestions.is_empty() {
        return None;
    }

    if suggestions.len() == 1 {
        return Some(format!("did you mean `{}`?", suggestions[0].name.as_str()));
    }

    let names: Vec<&str> = suggestions.iter().map(|s| s.name.as_str()).collect();
    if names.len() == 2 {
        Some(format!("did you mean `{}` or `{}`?", names[0], names[1]))
    } else {
        let last = names.last().unwrap();
        let rest = &names[..names.len() - 1];
        Some(format!(
            "did you mean `{}`, or `{}`?",
            rest.iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", "),
            last
        ))
    }
}

/// Common Haskell/BHC function names for fallback suggestions.
pub const COMMON_FUNCTIONS: &[&str] = &[
    // Prelude basics
    "map",
    "filter",
    "foldl",
    "foldr",
    "length",
    "reverse",
    "concat",
    "head",
    "tail",
    "init",
    "last",
    "null",
    "sum",
    "product",
    "maximum",
    "minimum",
    "take",
    "drop",
    "zip",
    "zipWith",
    "unzip",
    "replicate",
    "repeat",
    "iterate",
    "cycle",
    // Common type functions
    "show",
    "read",
    "print",
    "putStrLn",
    "getLine",
    // Maybe
    "Just",
    "Nothing",
    "maybe",
    "fromMaybe",
    "isJust",
    "isNothing",
    // Either
    "Left",
    "Right",
    "either",
    "fromLeft",
    "fromRight",
    // Bool
    "True",
    "False",
    "not",
    "otherwise",
    // Numeric
    "abs",
    "signum",
    "negate",
    "fromIntegral",
    "realToFrac",
    // BHC Tensor
    "matmul",
    "transpose",
    "reshape",
    "zeros",
    "ones",
    "tensor",
    "toDynamic",
    "fromDynamic",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_edit_distance_one_char() {
        assert_eq!(edit_distance("hello", "hallo"), 1); // substitution
        assert_eq!(edit_distance("hello", "hell"), 1); // deletion
        assert_eq!(edit_distance("hello", "helloo"), 1); // insertion
    }

    #[test]
    fn test_edit_distance_multiple() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("saturday", "sunday"), 3);
    }

    #[test]
    fn test_edit_distance_empty() {
        assert_eq!(edit_distance("", "hello"), 5);
        assert_eq!(edit_distance("hello", ""), 5);
        assert_eq!(edit_distance("", ""), 0);
    }

    #[test]
    fn test_find_similar_names() {
        let candidates: Vec<Symbol> = vec![
            Symbol::intern("map"),
            Symbol::intern("fmap"),
            Symbol::intern("filter"),
            Symbol::intern("fold"),
            Symbol::intern("mop"),
        ];

        let suggestions = find_similar_names("maap", &candidates);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].name.as_str(), "map");
    }

    #[test]
    fn test_is_likely_typo() {
        assert!(is_likely_typo("mpa", "map"));
        assert!(is_likely_typo("fitler", "filter"));
        assert!(!is_likely_typo("xyz", "abc"));
    }

    #[test]
    fn test_format_suggestions_single() {
        let suggestions = vec![Suggestion {
            name: Symbol::intern("map"),
            distance: 1,
        }];
        let formatted = format_suggestions(&suggestions);
        assert_eq!(formatted, Some("did you mean `map`?".to_string()));
    }

    #[test]
    fn test_format_suggestions_multiple() {
        let suggestions = vec![
            Suggestion {
                name: Symbol::intern("map"),
                distance: 1,
            },
            Suggestion {
                name: Symbol::intern("fmap"),
                distance: 2,
            },
        ];
        let formatted = format_suggestions(&suggestions);
        assert_eq!(formatted, Some("did you mean `map` or `fmap`?".to_string()));
    }

    #[test]
    fn test_case_insensitive() {
        let candidates: Vec<Symbol> = vec![Symbol::intern("MyFunction")];
        let suggestions = find_similar_names("myfunction", &candidates);
        assert!(!suggestions.is_empty());
    }
}
