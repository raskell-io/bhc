# bhc-diagnostics

Error reporting and diagnostics for BHC.

## Overview

`bhc-diagnostics` provides rich error reporting with:

- **Cargo-style output**: Colors, underlines, source excerpts
- **JSON format**: Machine-readable diagnostics for tooling
- **LSP support**: IDE integration
- **Error codes**: Explanations via `--explain`
- **Suggestions**: Actionable fix recommendations

## Core Types

| Type | Description |
|------|-------------|
| `Diagnostic` | A complete diagnostic message |
| `Severity` | Error, warning, note, help, bug |
| `Label` | Span with message (primary/secondary) |
| `Suggestion` | Proposed fix with replacement text |
| `SourceMap` | File registry for span resolution |
| `DiagnosticHandler` | Collects and manages diagnostics |

## Quick Start

```rust
use bhc_diagnostics::{Diagnostic, SourceMap, DiagnosticRenderer};
use bhc_span::{FullSpan, FileId, Span};

// Set up source map
let mut sm = SourceMap::new();
let file_id = sm.add_file("Main.hs".into(), "foo = x + 1".into());

// Create diagnostic
let span = FullSpan::new(file_id, Span::from_raw(6, 7));
let diag = Diagnostic::error("undefined variable")
    .with_code("E0003")
    .with_label(span, "not found in scope")
    .with_note("did you mean `y`?");

// Render
let renderer = DiagnosticRenderer::new(&sm);
renderer.render_all(&[diag]);
```

Output:
```
error[E0003]: undefined variable
 --> Main.hs:1:7
   |
   | x
   | ^
   | not found in scope
 = note: did you mean `y`?
```

## Creating Diagnostics

### Error

```rust
let diag = Diagnostic::error("type mismatch")
    .with_code("E0001")
    .with_label(span, "expected `Int`, found `String`");
```

### Warning

```rust
let diag = Diagnostic::warning("unused variable")
    .with_code("W0001")
    .with_label(span, "this variable is never used")
    .with_note("prefix with underscore to silence: `_x`");
```

### Bug (Internal Error)

```rust
let diag = Diagnostic::bug("invariant violated")
    .with_label(span, "impossible state reached")
    .with_note("please report this bug");
```

## Labels

### Primary Label

The main location being reported:

```rust
diag.with_label(span, "error occurred here")
```

### Secondary Labels

Additional context from other locations:

```rust
let diag = Diagnostic::error("type mismatch")
    .with_label(expr_span, "this has type `String`")
    .with_secondary_label(expected_span, "but expected `Int` because of this");
```

Output:
```
error: type mismatch
 --> Main.hs:5:10
   |
   | foo = show 42
   |       ^^^^^^^
   |       this has type `String`
   |
 --> Main.hs:3:8
   |
   | foo :: Int
   |        ^^^
   |        but expected `Int` because of this
```

## Suggestions

### Machine-Applicable Fix

```rust
use bhc_diagnostics::{Suggestion, Applicability};

let suggestion = Suggestion::new(
    "add type annotation",
    span,
    ":: Int",
    Applicability::MachineApplicable,
);

let diag = Diagnostic::error("ambiguous type")
    .with_label(span, "type is ambiguous")
    .with_suggestion(suggestion);
```

### Maybe Incorrect

```rust
let suggestion = Suggestion::new(
    "did you mean this?",
    span,
    "length",
    Applicability::MaybeIncorrect,
);
```

### Has Placeholders

```rust
let suggestion = Suggestion::new(
    "add missing case",
    span,
    "Nothing -> <expr>",
    Applicability::HasPlaceholders,
);
```

## DiagnosticHandler

Collects diagnostics during compilation:

```rust
let mut handler = DiagnosticHandler::new();

// Emit diagnostics
handler.emit(Diagnostic::error("error 1"));
handler.emit(Diagnostic::warning("warning 1"));
handler.emit(Diagnostic::error("error 2"));

// Check status
assert!(handler.has_errors());
assert_eq!(handler.error_count(), 2);
assert_eq!(handler.warning_count(), 1);

// Get all diagnostics
for diag in handler.diagnostics() {
    println!("{:?}", diag.message);
}

// Take diagnostics (clears handler)
let all = handler.take_diagnostics();
```

## Rendering

### Terminal Output

```rust
use bhc_diagnostics::{DiagnosticRenderer, CargoRenderer, RenderConfig};

// Basic renderer
let renderer = DiagnosticRenderer::new(&source_map);
renderer.render_all(&diagnostics);

// Cargo-style renderer with config
let config = RenderConfig {
    colors: true,
    terminal_width: 120,
    context_lines: 2,
};
let renderer = CargoRenderer::new(&source_map, config);
renderer.render_all(&diagnostics);
```

### Without Colors

```rust
let renderer = DiagnosticRenderer::new(&source_map)
    .without_colors();
```

### To String

```rust
let mut output = Vec::new();
renderer.render(&diagnostic, &mut output)?;
let text = String::from_utf8(output)?;
```

## JSON Output

For tooling and IDE integration:

```rust
use bhc_diagnostics::json::{diagnostic_to_json, diagnostics_to_json};

// Single diagnostic
let json = diagnostic_to_json(&diagnostic, &source_map);
println!("{}", serde_json::to_string_pretty(&json)?);

// Multiple diagnostics (JSON lines)
let json_lines = to_json_lines(&diagnostics, &source_map);
```

JSON format:
```json
{
  "severity": "error",
  "code": "E0003",
  "message": "undefined variable",
  "spans": [
    {
      "file": "Main.hs",
      "line_start": 1,
      "line_end": 1,
      "column_start": 7,
      "column_end": 8,
      "is_primary": true,
      "label": "not found in scope"
    }
  ],
  "notes": ["did you mean `y`?"],
  "suggestions": []
}
```

## LSP Integration

Convert diagnostics for Language Server Protocol:

```rust
use bhc_diagnostics::lsp::{to_lsp_diagnostics, to_code_actions};

// Convert to LSP diagnostics
let lsp_diags = to_lsp_diagnostics(&diagnostics, &source_map);

// Publish to client
let params = publish_diagnostics(uri, lsp_diags, version);

// Get code actions for quick fixes
let actions = to_code_actions(&diagnostics, &source_map, range);
```

LSP types:
```rust
pub struct LspDiagnostic {
    pub range: LspRange,
    pub severity: LspSeverity,
    pub code: Option<String>,
    pub message: String,
    pub related_information: Vec<LspRelatedInfo>,
}

pub struct LspCodeAction {
    pub title: String,
    pub kind: String,
    pub diagnostics: Vec<LspDiagnostic>,
    pub edit: LspWorkspaceEdit,
}
```

## Error Explanations

Detailed explanations for error codes:

```rust
use bhc_diagnostics::explain::{get_explanation, print_explanation};

// Get explanation text
if let Some(explanation) = get_explanation("E0003") {
    println!("{}", explanation);
}

// Print formatted explanation
print_explanation("E0003")?;

// List all error codes
for code in all_error_codes() {
    println!("{}", code);
}
```

Explanation format:
```
E0003: Undefined variable

This error occurs when you reference a variable that hasn't been
defined in the current scope.

Example of erroneous code:

    foo = x + 1

The variable `x` is not in scope. You need to either:

1. Define `x` before using it:

    x = 42
    foo = x + 1

2. Add it as a parameter:

    foo x = x + 1

3. Import it from another module:

    import MyModule (x)
    foo = x + 1
```

## Severity Levels

```rust
pub enum Severity {
    /// Internal compiler error
    Bug,
    /// Fatal error preventing compilation
    Error,
    /// Warning that doesn't prevent compilation
    Warning,
    /// Additional context
    Note,
    /// Suggestions for fixing
    Help,
}

impl Severity {
    pub fn color(self) -> &'static str {
        match self {
            Bug => "\x1b[1;35m",      // Bold magenta
            Error => "\x1b[1;31m",    // Bold red
            Warning => "\x1b[1;33m",  // Bold yellow
            Note => "\x1b[1;36m",     // Bold cyan
            Help => "\x1b[1;32m",     // Bold green
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Bug => "internal compiler error",
            Error => "error",
            Warning => "warning",
            Note => "note",
            Help => "help",
        }
    }
}
```

## Multi-line Spans

```rust
let source = r#"foo = bar
    + baz
    + qux"#;

let span = FullSpan::new(file_id, Span::from_raw(6, 26));
let diag = Diagnostic::error("type error in expression")
    .with_label(span, "this expression");
```

Output:
```
error: type error in expression
 --> Main.hs:1:7
   |
 1 |   foo = bar
   |  _______^
 2 | |     + baz
 3 | |     + qux
   | |________^ this expression
```

## SourceMap

Manages source files for span resolution:

```rust
let mut sm = SourceMap::new();

// Add files
let id1 = sm.add_file("Main.hs".into(), src1);
let id2 = sm.add_file("Lib.hs".into(), src2);

// Look up files
let file = sm.get_file(id1).unwrap();
let line_col = file.lookup_line_col(BytePos::new(42));
let text = file.source_text(span);
```

## Colors

```rust
use bhc_diagnostics::colors;

// Check if colors should be used
let use_colors = colors::should_use_colors();

// Color constants
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

// Respects NO_COLOR and CLICOLOR environment variables
```

## Testing

```rust
#[test]
fn test_diagnostic_builder() {
    let span = FullSpan::new(FileId::new(0), Span::from_raw(10, 20));

    let diag = Diagnostic::error("type mismatch")
        .with_code("E0001")
        .with_label(span, "expected `Int`")
        .with_note("consider adding a type annotation");

    assert!(diag.is_error());
    assert_eq!(diag.code, Some("E0001".to_string()));
    assert_eq!(diag.labels.len(), 1);
    assert_eq!(diag.notes.len(), 1);
}
```

## Integration

Diagnostics flow through the compiler:

```
Parser → Diagnostics → DiagnosticHandler
                            ↓
TypeChecker → Diagnostics → DiagnosticHandler
                            ↓
                    DiagnosticRenderer → Terminal
                            ↓
                    JSON/LSP export → Tooling
```

## Error Code Registry

| Range | Category |
|-------|----------|
| E0001-E0099 | Parse errors |
| E0100-E0199 | Name resolution |
| E0200-E0299 | Type errors |
| E0300-E0399 | Pattern errors |
| E0400-E0499 | Import/export |
| E0500-E0599 | FFI errors |
| W0001-W0099 | Warnings |
