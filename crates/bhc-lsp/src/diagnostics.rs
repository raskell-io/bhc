//! Diagnostics conversion and management.
//!
//! This module provides diagnostic creation helpers.

use lsp_types::{DiagnosticSeverity, Range};

/// Create an error diagnostic.
pub fn error(message: impl Into<String>, range: Range) -> lsp_types::Diagnostic {
    lsp_types::Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::ERROR),
        code: None,
        code_description: None,
        source: Some("bhc".to_string()),
        message: message.into(),
        related_information: None,
        tags: None,
        data: None,
    }
}

/// Create a warning diagnostic.
pub fn warning(message: impl Into<String>, range: Range) -> lsp_types::Diagnostic {
    lsp_types::Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::WARNING),
        code: None,
        code_description: None,
        source: Some("bhc".to_string()),
        message: message.into(),
        related_information: None,
        tags: None,
        data: None,
    }
}

/// Create an info diagnostic.
pub fn info(message: impl Into<String>, range: Range) -> lsp_types::Diagnostic {
    lsp_types::Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::INFORMATION),
        code: None,
        code_description: None,
        source: Some("bhc".to_string()),
        message: message.into(),
        related_information: None,
        tags: None,
        data: None,
    }
}

/// Create a hint diagnostic.
pub fn hint(message: impl Into<String>, range: Range) -> lsp_types::Diagnostic {
    lsp_types::Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::HINT),
        code: None,
        code_description: None,
        source: Some("bhc".to_string()),
        message: message.into(),
        related_information: None,
        tags: None,
        data: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lsp_types::Position;

    #[test]
    fn test_create_error() {
        let diag = error(
            "Test error",
            Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: 5,
                },
            },
        );

        assert_eq!(diag.severity, Some(DiagnosticSeverity::ERROR));
        assert_eq!(diag.message, "Test error");
    }

    #[test]
    fn test_create_warning() {
        let diag = warning(
            "Test warning",
            Range {
                start: Position {
                    line: 1,
                    character: 0,
                },
                end: Position {
                    line: 1,
                    character: 10,
                },
            },
        );

        assert_eq!(diag.severity, Some(DiagnosticSeverity::WARNING));
        assert_eq!(diag.message, "Test warning");
    }
}
