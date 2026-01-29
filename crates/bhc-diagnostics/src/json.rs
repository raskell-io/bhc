//! JSON diagnostic output format.
//!
//! This module provides machine-readable JSON output for diagnostics,
//! compatible with IDE tooling and CI systems.
//!
//! ## Format
//!
//! The output format is designed to be compatible with common conventions:
//!
//! ```json
//! {
//!   "message": "error message",
//!   "code": "E0001",
//!   "severity": "error",
//!   "spans": [
//!     {
//!       "file": "src/main.hs",
//!       "line_start": 10,
//!       "line_end": 10,
//!       "column_start": 5,
//!       "column_end": 15,
//!       "is_primary": true,
//!       "label": "expected Int, found String"
//!     }
//!   ],
//!   "notes": ["consider using show"],
//!   "suggestions": [
//!     {
//!       "message": "try this",
//!       "replacement": "show x"
//!     }
//!   ]
//! }
//! ```

use serde::{Deserialize, Serialize};

use crate::{Applicability, Diagnostic, Severity, SourceMap};

/// A diagnostic in JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonDiagnostic {
    /// The main error message.
    pub message: String,

    /// The error code, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,

    /// The severity level.
    pub severity: JsonSeverity,

    /// The source spans associated with this diagnostic.
    pub spans: Vec<JsonSpan>,

    /// Additional notes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,

    /// Suggested fixes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<JsonSuggestion>,

    /// Child diagnostics (for related information).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<JsonDiagnostic>,
}

/// Severity level in JSON format.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JsonSeverity {
    /// Internal compiler error.
    Bug,
    /// A fatal error.
    Error,
    /// A warning.
    Warning,
    /// Informational note.
    Note,
    /// Help message.
    Help,
}

impl From<Severity> for JsonSeverity {
    fn from(severity: Severity) -> Self {
        match severity {
            Severity::Bug => Self::Bug,
            Severity::Error => Self::Error,
            Severity::Warning => Self::Warning,
            Severity::Note => Self::Note,
            Severity::Help => Self::Help,
        }
    }
}

/// A source span in JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonSpan {
    /// The file path.
    pub file: String,

    /// The starting line (1-indexed).
    pub line_start: usize,

    /// The ending line (1-indexed).
    pub line_end: usize,

    /// The starting column (1-indexed).
    pub column_start: usize,

    /// The ending column (1-indexed).
    pub column_end: usize,

    /// Whether this is the primary span.
    pub is_primary: bool,

    /// The label for this span, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,

    /// The source text at this span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// A suggested fix in JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonSuggestion {
    /// The suggestion message.
    pub message: String,

    /// The replacement text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement: Option<String>,

    /// The span to replace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<JsonSpan>,

    /// How applicable this suggestion is.
    pub applicability: JsonApplicability,
}

/// Applicability in JSON format.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JsonApplicability {
    /// The fix is definitely correct.
    MachineApplicable,
    /// The fix might be correct.
    MaybeIncorrect,
    /// The fix has placeholders.
    HasPlaceholders,
    /// The fix is just a hint.
    Unspecified,
}

impl From<Applicability> for JsonApplicability {
    fn from(applicability: Applicability) -> Self {
        match applicability {
            Applicability::MachineApplicable => Self::MachineApplicable,
            Applicability::MaybeIncorrect => Self::MaybeIncorrect,
            Applicability::HasPlaceholders => Self::HasPlaceholders,
            Applicability::Unspecified => Self::Unspecified,
        }
    }
}

/// Convert a diagnostic to JSON format.
pub fn diagnostic_to_json(diagnostic: &Diagnostic, source_map: &SourceMap) -> JsonDiagnostic {
    let spans = diagnostic
        .labels
        .iter()
        .filter_map(|label| {
            let file = source_map.get_file(label.span.file)?;
            let span_info = file.span_lines(label.span.span);

            Some(JsonSpan {
                file: file.name.clone(),
                line_start: span_info.start_line,
                line_end: span_info.end_line,
                column_start: span_info.start_col,
                column_end: span_info.end_col,
                is_primary: label.primary,
                label: if label.message.is_empty() {
                    None
                } else {
                    Some(label.message.clone())
                },
                text: if label.span.span.is_dummy() {
                    None
                } else {
                    Some(file.source_text(label.span.span).to_string())
                },
            })
        })
        .collect();

    let suggestions = diagnostic
        .suggestions
        .iter()
        .map(|s| {
            let span = source_map.get_file(s.span.file).map(|file| {
                let span_info = file.span_lines(s.span.span);
                JsonSpan {
                    file: file.name.clone(),
                    line_start: span_info.start_line,
                    line_end: span_info.end_line,
                    column_start: span_info.start_col,
                    column_end: span_info.end_col,
                    is_primary: true,
                    label: None,
                    text: None,
                }
            });

            JsonSuggestion {
                message: s.message.clone(),
                replacement: if s.replacement.is_empty() {
                    None
                } else {
                    Some(s.replacement.clone())
                },
                span,
                applicability: s.applicability.into(),
            }
        })
        .collect();

    JsonDiagnostic {
        message: diagnostic.message.clone(),
        code: diagnostic.code.clone(),
        severity: diagnostic.severity.into(),
        spans,
        notes: diagnostic.notes.clone(),
        suggestions,
        children: Vec::new(),
    }
}

/// Convert multiple diagnostics to JSON format.
#[must_use]
pub fn diagnostics_to_json(
    diagnostics: &[Diagnostic],
    source_map: &SourceMap,
) -> Vec<JsonDiagnostic> {
    diagnostics
        .iter()
        .map(|d| diagnostic_to_json(d, source_map))
        .collect()
}

/// Serialize diagnostics to a JSON string.
pub fn to_json_string(
    diagnostics: &[Diagnostic],
    source_map: &SourceMap,
) -> Result<String, serde_json::Error> {
    let json_diags = diagnostics_to_json(diagnostics, source_map);
    serde_json::to_string_pretty(&json_diags)
}

/// Serialize diagnostics to a JSON string (compact, one per line).
pub fn to_json_lines(
    diagnostics: &[Diagnostic],
    source_map: &SourceMap,
) -> Result<String, serde_json::Error> {
    let mut output = String::new();
    for diag in diagnostics {
        let json_diag = diagnostic_to_json(diag, source_map);
        output.push_str(&serde_json::to_string(&json_diag)?);
        output.push('\n');
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_span::{FileId, FullSpan, Span};

    fn create_test_source_map() -> SourceMap {
        let mut sm = SourceMap::new();
        sm.add_file(
            "test.hs".to_string(),
            "module Test where\n\nfoo :: Int -> Int\nfoo x = x + y\n".to_string(),
        );
        sm
    }

    #[test]
    fn test_json_diagnostic() {
        let sm = create_test_source_map();

        let diag = Diagnostic::error("undefined variable `y`")
            .with_code("E0001")
            .with_label(
                FullSpan::new(FileId::new(0), Span::from_raw(50, 51)),
                "not found in this scope",
            )
            .with_note("consider defining `y`");

        let json_diag = diagnostic_to_json(&diag, &sm);

        assert_eq!(json_diag.message, "undefined variable `y`");
        assert_eq!(json_diag.code, Some("E0001".to_string()));
        assert_eq!(json_diag.severity, JsonSeverity::Error);
        assert_eq!(json_diag.spans.len(), 1);
        assert_eq!(json_diag.spans[0].file, "test.hs");
        assert!(json_diag.spans[0].is_primary);
        assert_eq!(json_diag.notes.len(), 1);
    }

    #[test]
    fn test_json_serialization() {
        let sm = create_test_source_map();

        let diag = Diagnostic::error("test error")
            .with_code("E0001")
            .with_label(FullSpan::new(FileId::new(0), Span::from_raw(0, 6)), "here");

        let json_str = to_json_string(&[diag], &sm).unwrap();

        assert!(json_str.contains("\"message\": \"test error\""));
        assert!(json_str.contains("\"severity\": \"error\""));
        assert!(json_str.contains("\"code\": \"E0001\""));
    }

    #[test]
    fn test_json_lines() {
        let sm = create_test_source_map();

        let diag1 = Diagnostic::error("error 1");
        let diag2 = Diagnostic::warning("warning 1");

        let output = to_json_lines(&[diag1, diag2], &sm).unwrap();
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("error 1"));
        assert!(lines[1].contains("warning 1"));
    }
}
