//! Golden file comparison utilities.

use thiserror::Error;

/// Errors that can occur during golden comparison.
#[derive(Debug, Error)]
pub enum GoldenError {
    /// Failed to read golden file.
    #[error("Failed to read golden file: {0}")]
    ReadError(#[from] std::io::Error),

    /// Golden file does not exist.
    #[error("Golden file does not exist: {0}")]
    NotFound(String),
}

/// Result of comparing actual output to expected golden output.
pub struct GoldenComparison {
    expected: String,
    actual: String,
    matches: bool,
}

impl GoldenComparison {
    /// Create a new comparison between expected and actual output.
    pub fn new(expected: &str, actual: &str) -> Self {
        // Normalize line endings and trailing whitespace
        let expected_norm = normalize_output(expected);
        let actual_norm = normalize_output(actual);

        let matches = expected_norm == actual_norm;

        Self {
            expected: expected_norm,
            actual: actual_norm,
            matches,
        }
    }

    /// Check if the outputs match.
    pub fn matches(&self) -> bool {
        self.matches
    }

    /// Get a human-readable diff between expected and actual.
    pub fn diff(&self) -> String {
        if self.matches {
            return String::from("(no differences)");
        }

        let mut result = String::new();

        let expected_lines: Vec<&str> = self.expected.lines().collect();
        let actual_lines: Vec<&str> = self.actual.lines().collect();

        // Simple line-by-line diff
        let max_lines = expected_lines.len().max(actual_lines.len());

        for i in 0..max_lines {
            let exp = expected_lines.get(i).copied().unwrap_or("");
            let act = actual_lines.get(i).copied().unwrap_or("");

            if exp != act {
                if i < expected_lines.len() {
                    result.push_str(&format!("- {}\n", exp));
                }
                if i < actual_lines.len() {
                    result.push_str(&format!("+ {}\n", act));
                }
            } else {
                result.push_str(&format!("  {}\n", exp));
            }
        }

        result
    }

    /// Get the expected output.
    pub fn expected(&self) -> &str {
        &self.expected
    }

    /// Get the actual output.
    pub fn actual(&self) -> &str {
        &self.actual
    }
}

/// Normalize output for comparison.
///
/// This handles:
/// - Line ending differences (CRLF vs LF)
/// - Trailing whitespace on lines
/// - Trailing newlines at end of output
fn normalize_output(s: &str) -> String {
    s.lines()
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
}

/// Load a golden file from disk.
pub fn load_golden(path: &std::path::Path) -> Result<String, GoldenError> {
    if !path.exists() {
        return Err(GoldenError::NotFound(path.display().to_string()));
    }
    Ok(std::fs::read_to_string(path)?)
}

/// Update a golden file with new content.
///
/// This is used when intentionally updating expected outputs.
pub fn update_golden(path: &std::path::Path, content: &str) -> Result<(), GoldenError> {
    let normalized = normalize_output(content);
    std::fs::write(path, format!("{}\n", normalized))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let cmp = GoldenComparison::new("Hello, World!", "Hello, World!");
        assert!(cmp.matches());
    }

    #[test]
    fn test_line_ending_normalization() {
        let cmp = GoldenComparison::new("Hello\r\nWorld", "Hello\nWorld");
        assert!(cmp.matches());
    }

    #[test]
    fn test_trailing_whitespace_normalization() {
        let cmp = GoldenComparison::new("Hello  \nWorld   ", "Hello\nWorld");
        assert!(cmp.matches());
    }

    #[test]
    fn test_trailing_newline_normalization() {
        let cmp = GoldenComparison::new("Hello\n\n\n", "Hello");
        assert!(cmp.matches());
    }

    #[test]
    fn test_mismatch() {
        let cmp = GoldenComparison::new("Hello", "Goodbye");
        assert!(!cmp.matches());
    }

    #[test]
    fn test_diff_output() {
        let cmp = GoldenComparison::new("line1\nline2", "line1\nchanged");
        let diff = cmp.diff();
        assert!(diff.contains("- line2"));
        assert!(diff.contains("+ changed"));
    }

    #[test]
    fn test_normalize_output() {
        assert_eq!(normalize_output("Hello  \r\n  World  \n\n"), "Hello\n  World");
    }

    #[test]
    fn test_multiline_match() {
        let expected = "Line 1\nLine 2\nLine 3";
        let actual = "Line 1\nLine 2\nLine 3\n";
        let cmp = GoldenComparison::new(expected, actual);
        assert!(cmp.matches());
    }
}
