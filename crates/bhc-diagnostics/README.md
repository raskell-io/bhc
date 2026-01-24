# bhc-diagnostics

Error reporting and diagnostics for the Basel Haskell Compiler.

## Overview

This crate provides rich error reporting with source code snippets, suggestions, and structured diagnostic output in the style of Rust/Cargo. It supports multiple output formats including terminal rendering with colors, JSON for tooling, and LSP integration.

## Key Types

| Type | Description |
|------|-------------|
| `Diagnostic` | A diagnostic message with severity, labels, and suggestions |
| `Severity` | Error, Warning, Note, Help, or Bug |
| `Label` | A labeled span pointing to source code |
| `Suggestion` | A suggested fix with replacement text |
| `DiagnosticHandler` | Collects and tracks diagnostics |
| `SourceMap` | Maps file IDs to source files for rendering |
| `DiagnosticRenderer` | Renders diagnostics to terminal output |

## Usage

```rust
use bhc_diagnostics::{Diagnostic, SourceMap, DiagnosticRenderer, FullSpan};
use bhc_span::{FileId, Span};

// Create a source map and add files
let mut sm = SourceMap::new();
let file_id = sm.add_file("test.hs".into(), "foo = x + 1".into());

// Create a diagnostic
let span = FullSpan::new(file_id, Span::from_raw(6, 7));
let diag = Diagnostic::error("undefined variable")
    .with_code("E0003")
    .with_label(span, "not found in scope")
    .with_note("consider importing this identifier")
    .with_suggestion(Suggestion::new(
        "did you mean 'y'?",
        span,
        "y",
        Applicability::MaybeIncorrect,
    ));

// Render to terminal
let renderer = DiagnosticRenderer::new(&sm);
renderer.render_all(&[diag]);
```

## Output Formats

### Terminal (Cargo-style)

```text
error[E0003]: undefined variable
 --> test.hs:1:7
   |
   | foo = x + 1
   |       ^
   | not found in scope
 = note: consider importing this identifier
 = help: did you mean 'y'?
```

### JSON (for tooling)

```rust
use bhc_diagnostics::json::diagnostics_to_json;

let json = diagnostics_to_json(&diagnostics);
```

### LSP (for IDE integration)

```rust
use bhc_diagnostics::lsp::{to_lsp_diagnostics, publish_diagnostics};

let lsp_diags = to_lsp_diagnostics(&diagnostics, &source_map);
```

## Error Codes

Error codes follow the pattern `EXXXX` and can be explained via `--explain`:

```rust
use bhc_diagnostics::explain::{get_explanation, print_explanation};

if let Some(explanation) = get_explanation("E0001") {
    print_explanation("E0001", &explanation);
}
```

## Severity Levels

| Severity | Color | Usage |
|----------|-------|-------|
| Bug | Magenta | Internal compiler errors |
| Error | Red | Fatal errors preventing compilation |
| Warning | Yellow | Non-fatal issues |
| Note | Cyan | Additional context |
| Help | Green | Suggestions for fixes |

## Design Notes

- Diagnostics are immutable once created (builder pattern)
- Source locations use `FullSpan` (file ID + byte range)
- The handler tracks error/warning counts for exit codes
- JSON output is compatible with Cargo's diagnostic format

## Related Crates

- `bhc-span` - Source locations used by labels
- `bhc-lexer` - Produces diagnostics for lexical errors
- `bhc-parser` - Produces diagnostics for parse errors
- `bhc-typeck` - Produces diagnostics for type errors
