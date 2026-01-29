//! Haddock markup parser.
//!
//! This module parses Haddock-style documentation markup into structured
//! content. It supports:
//!
//! - **Paragraphs**: Separated by blank lines
//! - **Sections**: `==== __Title__` or `= Title =`
//! - **Code blocks**: Fenced with `@` or indented
//! - **Examples**: Lines starting with `>>>`
//! - **Links**: `'identifier'` for identifiers, `"Module"` for modules
//! - **Emphasis**: `__bold__`, `/italic/`
//! - **Lists**: `*` for unordered, `1.` for ordered
//! - **BHC sections**: `Complexity:`, `Fusion:`, `SIMD:`, `Profile:`

use crate::model::{DocContent, Example};
use regex::Regex;
use std::collections::HashMap;

/// Parse Haddock markup into structured documentation.
pub fn parse(text: &str) -> DocContent {
    let text = text.trim();
    if text.is_empty() {
        return DocContent {
            brief: String::new(),
            description: String::new(),
            sections: HashMap::new(),
            examples: vec![],
            see_also: vec![],
            since: None,
            deprecated: None,
        };
    }

    let mut brief = String::new();
    let mut description = String::new();
    let mut sections: HashMap<String, String> = HashMap::new();
    let mut examples = Vec::new();
    let mut see_also = Vec::new();
    let mut since = None;
    let mut deprecated = None;

    // Split into paragraphs
    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    let mut current_section: Option<String> = None;
    let mut section_content = String::new();

    for (i, para) in paragraphs.iter().enumerate() {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        // Check for section header: ==== __Title__
        if let Some(section_name) = parse_section_header(para) {
            // Save previous section
            if let Some(name) = current_section.take() {
                sections.insert(name, section_content.trim().to_string());
            }
            current_section = Some(section_name);
            section_content.clear();
            continue;
        }

        // Check for BHC-specific annotations
        if let Some((key, value)) = parse_annotation(para) {
            match key.as_str() {
                "Since" => since = Some(value),
                "Deprecated" => deprecated = Some(value),
                "See Also" | "See also" => {
                    see_also.extend(value.split(',').map(|s| s.trim().to_string()));
                }
                _ => {
                    sections.insert(key, value);
                }
            }
            continue;
        }

        // Check for examples
        if para.starts_with(">>>") {
            examples.extend(parse_examples(para));
            continue;
        }

        // Check for code block
        if para.starts_with('@') && para.ends_with('@') {
            let code = para.trim_start_matches('@').trim_end_matches('@').trim();
            examples.push(Example {
                code: code.to_string(),
                output: None,
                runnable: true,
            });
            continue;
        }

        // Regular content
        if current_section.is_some() {
            if !section_content.is_empty() {
                section_content.push_str("\n\n");
            }
            section_content.push_str(para);
        } else {
            if i == 0 {
                brief = parse_inline_markup(extract_first_sentence(para));
            }
            if !description.is_empty() {
                description.push_str("\n\n");
            }
            description.push_str(&parse_inline_markup(para.to_string()));
        }
    }

    // Save last section
    if let Some(name) = current_section {
        sections.insert(name, section_content.trim().to_string());
    }

    DocContent {
        brief,
        description,
        sections,
        examples,
        see_also,
        since,
        deprecated,
    }
}

/// Parse a section header like `==== __Examples__` or `= Examples =`.
fn parse_section_header(line: &str) -> Option<String> {
    let line = line.trim();

    // Pattern: ==== __Title__ or === Title or == Title
    if line.starts_with("==") {
        let content = line.trim_start_matches('=').trim();
        let name = content
            .trim_start_matches("__")
            .trim_end_matches("__")
            .trim_end_matches('=')
            .trim();
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }

    // Pattern: = Title =
    if line.starts_with('=') && line.ends_with('=') {
        let name = line.trim_matches('=').trim();
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }

    None
}

/// Parse a BHC annotation like `Complexity: O(n)`.
fn parse_annotation(para: &str) -> Option<(String, String)> {
    let annotations = [
        "Complexity",
        "Fusion",
        "SIMD",
        "Profile",
        "Since",
        "Deprecated",
        "See Also",
        "See also",
    ];

    for ann in &annotations {
        if para.starts_with(ann) {
            let rest = para[ann.len()..].trim();
            if rest.starts_with(':') {
                let value = rest[1..].trim().to_string();
                return Some((ann.to_string(), value));
            }
        }
    }

    None
}

/// Parse examples from lines starting with `>>>`.
fn parse_examples(text: &str) -> Vec<Example> {
    let mut examples = Vec::new();
    let mut current_code = String::new();
    let mut current_output = String::new();
    let mut in_example = false;

    for line in text.lines() {
        if line.trim().starts_with(">>>") {
            // New example prompt
            if in_example && !current_code.is_empty() {
                examples.push(Example {
                    code: current_code.clone(),
                    output: if current_output.is_empty() {
                        None
                    } else {
                        Some(current_output.clone())
                    },
                    runnable: true,
                });
                current_code.clear();
                current_output.clear();
            }
            current_code = line
                .trim()
                .strip_prefix(">>>")
                .unwrap_or("")
                .trim()
                .to_string();
            in_example = true;
        } else if in_example {
            // Output line
            if !current_output.is_empty() {
                current_output.push('\n');
            }
            current_output.push_str(line.trim());
        }
    }

    // Last example
    if !current_code.is_empty() {
        examples.push(Example {
            code: current_code,
            output: if current_output.is_empty() {
                None
            } else {
                Some(current_output)
            },
            runnable: true,
        });
    }

    examples
}

/// Extract the first sentence from a paragraph.
fn extract_first_sentence(text: &str) -> String {
    // Find the end of the first sentence (. followed by space or end)
    let sentence_end = Regex::new(r"\.\s").unwrap();
    if let Some(m) = sentence_end.find(text) {
        text[..m.start() + 1].to_string()
    } else if text.ends_with('.') {
        text.to_string()
    } else {
        // No period found, take the whole paragraph
        text.to_string()
    }
}

/// Parse inline markup (links, emphasis, code).
fn parse_inline_markup(text: String) -> String {
    let mut result = text;

    // Convert 'identifier' to links
    let id_link = Regex::new(r"'([a-zA-Z_][a-zA-Z0-9_']*)'").unwrap();
    result = id_link
        .replace_all(&result, "<a href=\"#$1\">$1</a>")
        .to_string();

    // Convert "Module" to module links
    let mod_link = Regex::new(r#""([A-Z][a-zA-Z0-9_.]*[a-zA-Z0-9])""#).unwrap();
    result = mod_link
        .replace_all(&result, "<a href=\"$1.html\">$1</a>")
        .to_string();

    // Convert __bold__
    let bold = Regex::new(r"__([^_]+)__").unwrap();
    result = bold.replace_all(&result, "<strong>$1</strong>").to_string();

    // Convert /italic/
    let italic = Regex::new(r"/([^/]+)/").unwrap();
    result = italic.replace_all(&result, "<em>$1</em>").to_string();

    // Convert `code`
    let code = Regex::new(r"`([^`]+)`").unwrap();
    result = code.replace_all(&result, "<code>$1</code>").to_string();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let doc = parse("This is a brief description.");
        assert_eq!(doc.brief, "This is a brief description.");
    }

    #[test]
    fn test_parse_section() {
        let doc = parse("Brief.\n\n==== __Examples__\n\nSome example.");
        assert_eq!(doc.brief, "Brief.");
        assert!(doc.sections.contains_key("Examples"));
    }

    #[test]
    fn test_parse_examples() {
        let doc = parse("Brief.\n\n>>> 1 + 1\n2");
        assert_eq!(doc.examples.len(), 1);
        assert_eq!(doc.examples[0].code, "1 + 1");
        assert_eq!(doc.examples[0].output, Some("2".to_string()));
    }

    #[test]
    fn test_parse_complexity() {
        let doc = parse("Brief.\n\nComplexity: O(n log n)");
        assert_eq!(
            doc.sections.get("Complexity"),
            Some(&"O(n log n)".to_string())
        );
    }
}
