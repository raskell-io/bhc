//! Cargo-style diagnostic rendering.
//!
//! This module provides a renderer that produces error messages in the style
//! of Rust's compiler, with:
//! - Colored output
//! - Source code snippets with line numbers
//! - Underlines and carets for precise error locations
//! - Multiple labeled spans
//! - Notes and help messages

use std::io::Write;

use crate::{Diagnostic, Label, Severity, SourceMap, Suggestion};

/// ANSI color codes for terminal output.
pub mod colors {
    /// Reset all formatting.
    pub const RESET: &str = "\x1b[0m";
    /// Bold text.
    pub const BOLD: &str = "\x1b[1m";
    /// Red text (for errors).
    pub const RED: &str = "\x1b[31m";
    /// Bold red text.
    pub const BOLD_RED: &str = "\x1b[1;31m";
    /// Yellow text (for warnings).
    pub const YELLOW: &str = "\x1b[33m";
    /// Bold yellow text.
    pub const BOLD_YELLOW: &str = "\x1b[1;33m";
    /// Blue text (for notes, line numbers).
    pub const BLUE: &str = "\x1b[34m";
    /// Bold blue text.
    pub const BOLD_BLUE: &str = "\x1b[1;34m";
    /// Cyan text (for notes).
    pub const CYAN: &str = "\x1b[36m";
    /// Bold cyan text.
    pub const BOLD_CYAN: &str = "\x1b[1;36m";
    /// Green text (for help).
    pub const GREEN: &str = "\x1b[32m";
    /// Bold green text.
    pub const BOLD_GREEN: &str = "\x1b[1;32m";
    /// Magenta text (for internal errors).
    pub const MAGENTA: &str = "\x1b[35m";
    /// Bold magenta text.
    pub const BOLD_MAGENTA: &str = "\x1b[1;35m";
    /// White text.
    pub const WHITE: &str = "\x1b[37m";
    /// Bold white text.
    pub const BOLD_WHITE: &str = "\x1b[1;37m";
}

/// Configuration for the diagnostic renderer.
#[derive(Clone, Debug)]
pub struct RenderConfig {
    /// Whether to use colors in output.
    pub colors: bool,
    /// Whether to show error codes.
    pub show_codes: bool,
    /// Number of context lines to show before/after the error.
    pub context_lines: usize,
    /// Maximum width for the output.
    pub max_width: usize,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            colors: true,
            show_codes: true,
            context_lines: 0,
            max_width: 140,
        }
    }
}

impl RenderConfig {
    /// Create a config with colors enabled.
    #[must_use]
    pub fn colored() -> Self {
        Self::default()
    }

    /// Create a config without colors (for testing or piping).
    #[must_use]
    pub fn plain() -> Self {
        Self {
            colors: false,
            ..Self::default()
        }
    }
}

/// A Cargo-style diagnostic renderer.
pub struct CargoRenderer<'a> {
    source_map: &'a SourceMap,
    config: RenderConfig,
}

impl<'a> CargoRenderer<'a> {
    /// Create a new renderer with the given source map.
    #[must_use]
    pub fn new(source_map: &'a SourceMap) -> Self {
        Self {
            source_map,
            config: RenderConfig::default(),
        }
    }

    /// Create a new renderer with custom configuration.
    #[must_use]
    pub fn with_config(source_map: &'a SourceMap, config: RenderConfig) -> Self {
        Self { source_map, config }
    }

    /// Get the color for a severity level.
    fn severity_color(&self, severity: Severity) -> &'static str {
        if !self.config.colors {
            return "";
        }
        match severity {
            Severity::Bug => colors::BOLD_MAGENTA,
            Severity::Error => colors::BOLD_RED,
            Severity::Warning => colors::BOLD_YELLOW,
            Severity::Note => colors::BOLD_CYAN,
            Severity::Help => colors::BOLD_GREEN,
        }
    }

    /// Get the reset code if colors are enabled.
    fn reset(&self) -> &'static str {
        if self.config.colors {
            colors::RESET
        } else {
            ""
        }
    }

    /// Get blue color if enabled.
    fn blue(&self) -> &'static str {
        if self.config.colors {
            colors::BOLD_BLUE
        } else {
            ""
        }
    }

    /// Render a diagnostic to the given writer.
    pub fn render(&self, diagnostic: &Diagnostic, w: &mut impl Write) -> std::io::Result<()> {
        // Header: error[E0001]: message
        self.render_header(diagnostic, w)?;

        // Labels with source snippets
        self.render_labels(diagnostic, w)?;

        // Notes
        for note in &diagnostic.notes {
            self.render_note(note, w)?;
        }

        // Suggestions
        for suggestion in &diagnostic.suggestions {
            self.render_suggestion(suggestion, w)?;
        }

        writeln!(w)?;
        Ok(())
    }

    /// Render the header line.
    fn render_header(&self, diagnostic: &Diagnostic, w: &mut impl Write) -> std::io::Result<()> {
        let color = self.severity_color(diagnostic.severity);
        let reset = self.reset();

        write!(w, "{color}{}{reset}", diagnostic.severity.label())?;

        if self.config.show_codes {
            if let Some(code) = &diagnostic.code {
                write!(w, "{color}[{code}]{reset}")?;
            }
        }

        writeln!(w, ": {color}{}{reset}", diagnostic.message)?;

        Ok(())
    }

    /// Render all labels with source snippets.
    fn render_labels(&self, diagnostic: &Diagnostic, w: &mut impl Write) -> std::io::Result<()> {
        let blue = self.blue();
        let reset = self.reset();

        for (i, label) in diagnostic.labels.iter().enumerate() {
            let Some(file) = self.source_map.get_file(label.span.file) else {
                continue;
            };

            if label.span.span.is_dummy() {
                continue;
            }

            let span_info = file.span_lines(label.span.span);

            // Location line: --> file.hs:line:col
            let arrow = if label.primary && i == 0 {
                "-->"
            } else {
                "   "
            };
            writeln!(
                w,
                " {blue}{arrow}{reset} {}:{}:{}",
                file.name, span_info.start_line, span_info.start_col
            )?;

            // Calculate the width needed for line numbers
            let line_num_width = span_info.end_line.to_string().len().max(3);

            // Empty line before source
            writeln!(w, " {blue}{:>width$} |{reset}", "", width = line_num_width)?;

            // Render each line in the span
            if span_info.is_multiline() {
                self.render_multiline_span(file, &span_info, label, line_num_width, w)?;
            } else {
                self.render_single_line_span(file, &span_info, label, line_num_width, w)?;
            }
        }

        Ok(())
    }

    /// Render a single-line span with underline.
    fn render_single_line_span(
        &self,
        file: &bhc_span::SourceFile,
        span_info: &bhc_span::SpanLines,
        label: &Label,
        line_num_width: usize,
        w: &mut impl Write,
    ) -> std::io::Result<()> {
        let blue = self.blue();
        let reset = self.reset();
        let underline_color = if label.primary {
            self.severity_color(Severity::Error)
        } else {
            self.blue()
        };

        // Get the source line (0-indexed)
        let line_idx = span_info.start_line - 1;
        let Some(line_content) = file.line_content(line_idx) else {
            return Ok(());
        };

        // Source line with line number
        writeln!(
            w,
            " {blue}{:>width$} |{reset} {}",
            span_info.start_line,
            line_content,
            width = line_num_width
        )?;

        // Underline
        let start_col = span_info.start_col.saturating_sub(1);
        let end_col = span_info.end_col.saturating_sub(1);
        let underline_len = (end_col - start_col).max(1);

        let padding = " ".repeat(start_col);
        let underline = "^".repeat(underline_len);

        write!(
            w,
            " {blue}{:>width$} |{reset} {padding}{underline_color}{underline}{reset}",
            "",
            width = line_num_width
        )?;

        // Label message on the same line if short, otherwise on next line
        if !label.message.is_empty() {
            writeln!(w, " {underline_color}{}{reset}", label.message)?;
        } else {
            writeln!(w)?;
        }

        Ok(())
    }

    /// Render a multi-line span.
    fn render_multiline_span(
        &self,
        file: &bhc_span::SourceFile,
        span_info: &bhc_span::SpanLines,
        label: &Label,
        line_num_width: usize,
        w: &mut impl Write,
    ) -> std::io::Result<()> {
        let blue = self.blue();
        let reset = self.reset();
        let span_color = if label.primary {
            self.severity_color(Severity::Error)
        } else {
            self.blue()
        };

        for line_num in span_info.start_line..=span_info.end_line {
            let line_idx = line_num - 1;
            let Some(line_content) = file.line_content(line_idx) else {
                continue;
            };

            // Determine the span portion for this line
            let (_start_col, end_col, show_start, show_end) = if line_num == span_info.start_line {
                (span_info.start_col - 1, line_content.len(), true, false)
            } else if line_num == span_info.end_line {
                (0, span_info.end_col - 1, false, true)
            } else {
                (0, line_content.len(), false, false)
            };

            // Source line
            if show_start {
                // First line: show starting marker
                writeln!(
                    w,
                    " {blue}{:>width$} |{reset}   {span_color}/{reset} {}",
                    line_num,
                    line_content,
                    width = line_num_width
                )?;
            } else if show_end {
                // Last line: show ending marker
                writeln!(
                    w,
                    " {blue}{:>width$} |{reset} {span_color}|{reset} {}",
                    line_num,
                    line_content,
                    width = line_num_width
                )?;

                // Underline for last line
                let padding = " ".repeat(end_col);
                writeln!(
                    w,
                    " {blue}{:>width$} |{reset} {span_color}|_{padding}^{reset}",
                    "",
                    width = line_num_width
                )?;
            } else {
                // Middle line
                writeln!(
                    w,
                    " {blue}{:>width$} |{reset} {span_color}|{reset} {}",
                    line_num,
                    line_content,
                    width = line_num_width
                )?;
            }
        }

        // Label message
        if !label.message.is_empty() {
            writeln!(
                w,
                " {blue}{:>width$} |{reset} {span_color}{}{reset}",
                "",
                label.message,
                width = line_num_width
            )?;
        }

        Ok(())
    }

    /// Render a note.
    fn render_note(&self, note: &str, w: &mut impl Write) -> std::io::Result<()> {
        let blue = self.blue();
        let reset = self.reset();

        // Handle multi-line notes
        for (i, line) in note.lines().enumerate() {
            if i == 0 {
                writeln!(w, " {blue}={reset} {blue}note{reset}: {line}")?;
            } else {
                writeln!(w, "          {line}")?;
            }
        }

        Ok(())
    }

    /// Render a suggestion.
    fn render_suggestion(
        &self,
        suggestion: &Suggestion,
        w: &mut impl Write,
    ) -> std::io::Result<()> {
        let green = if self.config.colors {
            colors::BOLD_GREEN
        } else {
            ""
        };
        let reset = self.reset();
        let blue = self.blue();

        writeln!(
            w,
            " {blue}={reset} {green}help{reset}: {}",
            suggestion.message
        )?;

        if !suggestion.replacement.is_empty() {
            writeln!(w, "          {blue}|{reset}")?;
            for line in suggestion.replacement.lines() {
                writeln!(w, "          {blue}|{reset} {green}{line}{reset}")?;
            }
        }

        Ok(())
    }

    /// Render all diagnostics to stderr.
    pub fn render_all(&self, diagnostics: &[Diagnostic]) {
        let mut stderr = std::io::stderr().lock();
        for diag in diagnostics {
            let _ = self.render(diag, &mut stderr);
        }
    }

    /// Render a diagnostic to a string.
    #[must_use]
    pub fn render_to_string(&self, diagnostic: &Diagnostic) -> String {
        let mut buf = Vec::new();
        let _ = self.render(diagnostic, &mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_span::{FileId, FullSpan, Span};

    fn create_test_source_map() -> SourceMap {
        let mut sm = SourceMap::new();
        sm.add_file(
            "test.hs".to_string(),
            "module Test where\n\nfoo :: Int -> Int\nfoo x = x + y\n\nmain = foo 42\n".to_string(),
        );
        sm
    }

    #[test]
    fn test_simple_error() {
        let sm = create_test_source_map();
        let renderer = CargoRenderer::with_config(&sm, RenderConfig::plain());

        let diag = Diagnostic::error("undefined variable `y`")
            .with_code("E0001")
            .with_label(
                FullSpan::new(FileId::new(0), Span::from_raw(50, 51)),
                "not found in this scope",
            );

        let output = renderer.render_to_string(&diag);

        assert!(output.contains("error[E0001]"));
        assert!(output.contains("undefined variable `y`"));
        assert!(output.contains("test.hs:4:"));
        assert!(output.contains("not found in this scope"));
    }

    #[test]
    fn test_error_with_note() {
        let sm = create_test_source_map();
        let renderer = CargoRenderer::with_config(&sm, RenderConfig::plain());

        let diag = Diagnostic::error("type mismatch")
            .with_code("E0002")
            .with_label(
                FullSpan::new(FileId::new(0), Span::from_raw(50, 51)),
                "expected `Int`, found `String`",
            )
            .with_note("consider using `show` to convert to String");

        let output = renderer.render_to_string(&diag);

        assert!(output.contains("= note:"));
        assert!(output.contains("consider using `show`"));
    }

    #[test]
    fn test_warning() {
        let sm = create_test_source_map();
        let renderer = CargoRenderer::with_config(&sm, RenderConfig::plain());

        let diag = Diagnostic::warning("unused variable `x`")
            .with_code("W0001")
            .with_label(
                FullSpan::new(FileId::new(0), Span::from_raw(40, 41)),
                "this variable is never used",
            );

        let output = renderer.render_to_string(&diag);

        assert!(output.contains("warning[W0001]"));
        assert!(output.contains("unused variable"));
    }

    #[test]
    fn test_suggestion() {
        let sm = create_test_source_map();
        let renderer = CargoRenderer::with_config(&sm, RenderConfig::plain());

        let diag = Diagnostic::error("undefined variable `y`")
            .with_label(
                FullSpan::new(FileId::new(0), Span::from_raw(50, 51)),
                "not found",
            )
            .with_suggestion(crate::Suggestion::new(
                "did you mean `x`?",
                FullSpan::new(FileId::new(0), Span::from_raw(50, 51)),
                "x",
                crate::Applicability::MaybeIncorrect,
            ));

        let output = renderer.render_to_string(&diag);

        assert!(output.contains("= help:"));
        assert!(output.contains("did you mean `x`?"));
    }
}
