//! Error reporting and diagnostics for BHC.
//!
//! This crate provides rich error reporting with source code snippets,
//! suggestions, and structured diagnostic output in the style of Rust/Cargo.
//!
//! ## Features
//!
//! - Cargo-style terminal output with colors and underlines
//! - Machine-readable JSON diagnostic format
//! - Error code explanations via `--explain`
//! - Multi-line span support with context
//!
//! ## Example
//!
//! ```ignore
//! use bhc_diagnostics::{Diagnostic, SourceMap, CargoRenderer};
//!
//! let mut sm = SourceMap::new();
//! sm.add_file("test.hs".into(), "foo = x + 1".into());
//!
//! let diag = Diagnostic::error("undefined variable")
//!     .with_code("E0003")
//!     .with_label(span, "not found in scope");
//!
//! let renderer = CargoRenderer::new(&sm);
//! renderer.render_all(&[diag]);
//! ```

#![warn(missing_docs)]

pub mod explain;
pub mod json;
pub mod lsp;
pub mod render;

use bhc_span::{FileId, SourceFile};
pub use bhc_span::{FullSpan, Span};
use serde::{Deserialize, Serialize};
use std::io::Write;

// Re-exports for convenience
pub use explain::{all_error_codes, format_explanation, get_explanation, print_explanation};
pub use json::{diagnostic_to_json, diagnostics_to_json, to_json_lines, to_json_string};
pub use json::{JsonApplicability, JsonDiagnostic, JsonSeverity, JsonSpan, JsonSuggestion};
pub use lsp::{
    publish_diagnostics, to_code_actions, to_hover, to_lsp_diagnostic, to_lsp_diagnostics,
    LspCodeAction, LspDiagnostic, LspHover, LspRange, LspSeverity, LspTextEdit,
    PublishDiagnosticsParams,
};
pub use render::{colors, CargoRenderer, RenderConfig};

/// The severity level of a diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// A bug in the compiler itself.
    Bug,
    /// A fatal error that prevents compilation.
    Error,
    /// A warning that doesn't prevent compilation.
    Warning,
    /// A note providing additional context.
    Note,
    /// Help text with suggestions.
    Help,
}

impl Severity {
    /// Get the ANSI color code for this severity.
    #[must_use]
    pub fn color(self) -> &'static str {
        match self {
            Self::Bug => "\x1b[1;35m",     // Bold magenta
            Self::Error => "\x1b[1;31m",   // Bold red
            Self::Warning => "\x1b[1;33m", // Bold yellow
            Self::Note => "\x1b[1;36m",    // Bold cyan
            Self::Help => "\x1b[1;32m",    // Bold green
        }
    }

    /// Get the label for this severity.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Bug => "internal compiler error",
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Note => "note",
            Self::Help => "help",
        }
    }
}

/// A labeled span for diagnostics.
#[derive(Clone, Debug)]
pub struct Label {
    /// The span being labeled.
    pub span: FullSpan,
    /// The message for this label.
    pub message: String,
    /// Whether this is the primary label.
    pub primary: bool,
}

impl Label {
    /// Create a primary label.
    #[must_use]
    pub fn primary(span: FullSpan, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            primary: true,
        }
    }

    /// Create a secondary label.
    #[must_use]
    pub fn secondary(span: FullSpan, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            primary: false,
        }
    }
}

/// A diagnostic message with source locations and suggestions.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    /// The severity of this diagnostic.
    pub severity: Severity,
    /// The main message.
    pub message: String,
    /// An optional error code.
    pub code: Option<String>,
    /// Labeled spans with messages.
    pub labels: Vec<Label>,
    /// Additional notes.
    pub notes: Vec<String>,
    /// Suggested fixes.
    pub suggestions: Vec<Suggestion>,
}

impl Diagnostic {
    /// Create a new error diagnostic.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            code: None,
            labels: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Create a new warning diagnostic.
    #[must_use]
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            code: None,
            labels: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Create a new bug diagnostic (internal compiler error).
    #[must_use]
    pub fn bug(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Bug,
            message: message.into(),
            code: None,
            labels: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Add an error code.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Add a primary label.
    #[must_use]
    pub fn with_label(mut self, span: FullSpan, message: impl Into<String>) -> Self {
        self.labels.push(Label::primary(span, message));
        self
    }

    /// Add a secondary label.
    #[must_use]
    pub fn with_secondary_label(mut self, span: FullSpan, message: impl Into<String>) -> Self {
        self.labels.push(Label::secondary(span, message));
        self
    }

    /// Add a note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Add a suggestion.
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Check if this is an error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self.severity, Severity::Error | Severity::Bug)
    }
}

/// A suggested fix for a diagnostic.
#[derive(Clone, Debug)]
pub struct Suggestion {
    /// The message describing the suggestion.
    pub message: String,
    /// The span to replace.
    pub span: FullSpan,
    /// The replacement text.
    pub replacement: String,
    /// The applicability of this suggestion.
    pub applicability: Applicability,
}

impl Suggestion {
    /// Create a new suggestion.
    #[must_use]
    pub fn new(
        message: impl Into<String>,
        span: FullSpan,
        replacement: impl Into<String>,
        applicability: Applicability,
    ) -> Self {
        Self {
            message: message.into(),
            span,
            replacement: replacement.into(),
            applicability,
        }
    }
}

/// How applicable a suggestion is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Applicability {
    /// The fix is definitely what the user wants.
    MachineApplicable,
    /// The fix might be what the user wants.
    MaybeIncorrect,
    /// The fix has placeholders the user must fill in.
    HasPlaceholders,
    /// The fix is just a hint, not directly applicable.
    Unspecified,
}

/// A handler for collecting and emitting diagnostics.
#[derive(Debug, Default)]
pub struct DiagnosticHandler {
    diagnostics: Vec<Diagnostic>,
    error_count: usize,
    warning_count: usize,
}

impl DiagnosticHandler {
    /// Create a new diagnostic handler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit a diagnostic.
    pub fn emit(&mut self, diagnostic: Diagnostic) {
        match diagnostic.severity {
            Severity::Error | Severity::Bug => self.error_count += 1,
            Severity::Warning => self.warning_count += 1,
            _ => {}
        }
        self.diagnostics.push(diagnostic);
    }

    /// Check if any errors have been emitted.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Get the number of errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Get the number of warnings.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Get all diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Take all diagnostics, leaving the handler empty.
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.error_count = 0;
        self.warning_count = 0;
        std::mem::take(&mut self.diagnostics)
    }
}

/// A source map for looking up files and locations.
#[derive(Debug, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    /// Create a new empty source map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file to the source map.
    pub fn add_file(&mut self, name: String, src: String) -> FileId {
        let id = FileId::new(self.files.len() as u32);
        self.files.push(SourceFile::new(id, name, src));
        id
    }

    /// Get a file by ID.
    #[must_use]
    pub fn get_file(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    /// Get the number of files.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Check if the source map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

/// Render diagnostics to a writer.
pub struct DiagnosticRenderer<'a> {
    source_map: &'a SourceMap,
    use_colors: bool,
}

impl<'a> DiagnosticRenderer<'a> {
    /// Create a new renderer.
    #[must_use]
    pub fn new(source_map: &'a SourceMap) -> Self {
        Self {
            source_map,
            use_colors: true,
        }
    }

    /// Disable colors.
    #[must_use]
    pub fn without_colors(mut self) -> Self {
        self.use_colors = false;
        self
    }

    /// Render a diagnostic to the given writer.
    pub fn render(&self, diagnostic: &Diagnostic, w: &mut impl Write) -> std::io::Result<()> {
        let reset = if self.use_colors { "\x1b[0m" } else { "" };
        let color = if self.use_colors {
            diagnostic.severity.color()
        } else {
            ""
        };

        // Header
        write!(w, "{}{}", color, diagnostic.severity.label())?;
        if let Some(code) = &diagnostic.code {
            write!(w, "[{code}]")?;
        }
        writeln!(w, "{reset}: {}", diagnostic.message)?;

        // Labels
        for label in &diagnostic.labels {
            if let Some(file) = self.source_map.get_file(label.span.file) {
                let loc = file.lookup_line_col(label.span.span.lo);
                let arrow = if label.primary { "-->" } else { "   " };
                writeln!(w, " {arrow} {}:{}:{}", file.name, loc.line, loc.col)?;

                // Show source line
                if !label.span.span.is_dummy() {
                    let source = file.source_text(label.span.span);
                    writeln!(w, "   |")?;
                    writeln!(w, "   | {source}")?;
                    writeln!(w, "   | {}", "^".repeat(source.len().max(1)))?;
                    if !label.message.is_empty() {
                        writeln!(w, "   | {}", label.message)?;
                    }
                }
            }
        }

        // Notes
        for note in &diagnostic.notes {
            writeln!(w, " = note: {note}")?;
        }

        // Suggestions
        for suggestion in &diagnostic.suggestions {
            writeln!(w, " = help: {}", suggestion.message)?;
            if !suggestion.replacement.is_empty() {
                writeln!(w, "   |")?;
                writeln!(w, "   | {}", suggestion.replacement)?;
            }
        }

        writeln!(w)?;
        Ok(())
    }

    /// Render all diagnostics to stderr.
    pub fn render_all(&self, diagnostics: &[Diagnostic]) {
        let mut stderr = std::io::stderr().lock();
        for diag in diagnostics {
            let _ = self.render(diag, &mut stderr);
        }
    }
}

/// Trait for types that can produce diagnostics.
pub trait IntoDiagnostic {
    /// Convert into a diagnostic.
    fn into_diagnostic(self) -> Diagnostic;
}

impl IntoDiagnostic for Diagnostic {
    fn into_diagnostic(self) -> Diagnostic {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_builder() {
        let span = FullSpan::new(FileId::new(0), Span::from_raw(10, 20));

        let diag = Diagnostic::error("type mismatch")
            .with_code("E0001")
            .with_label(span, "expected `Int`, found `String`")
            .with_note("consider using `show` to convert to String");

        assert!(diag.is_error());
        assert_eq!(diag.code, Some("E0001".to_string()));
        assert_eq!(diag.labels.len(), 1);
        assert_eq!(diag.notes.len(), 1);
    }

    #[test]
    fn test_diagnostic_handler() {
        let mut handler = DiagnosticHandler::new();

        handler.emit(Diagnostic::error("error 1"));
        handler.emit(Diagnostic::warning("warning 1"));
        handler.emit(Diagnostic::error("error 2"));

        assert!(handler.has_errors());
        assert_eq!(handler.error_count(), 2);
        assert_eq!(handler.warning_count(), 1);
    }
}
