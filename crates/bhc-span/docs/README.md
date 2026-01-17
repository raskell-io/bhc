# bhc-span

Source location tracking and span management for accurate error reporting.

## Overview

`bhc-span` provides types for tracking source locations throughout the BHC compilation pipeline. Accurate spans are essential for:

- Precise error messages with source excerpts
- IDE features (go-to-definition, hover info)
- Source maps for debugging
- Multi-line diagnostic rendering

## Core Types

| Type | Description |
|------|-------------|
| `BytePos` | A byte offset into a source file |
| `Span` | A half-open byte range `[lo, hi)` |
| `Spanned<T>` | A value with an associated span |
| `FileId` | Unique identifier for a source file |
| `FullSpan` | Span with associated file ID |
| `LineCol` | 1-indexed line and column |
| `SourceFile` | File content with line index |

## Quick Start

```rust
use bhc_span::{Span, BytePos, SourceFile, FileId};

// Create a span
let span = Span::from_raw(10, 25);
assert_eq!(span.len(), 15);

// Work with source files
let src = "fn main =\n  print \"Hello\"";
let file = SourceFile::new(FileId::new(0), "Main.hs".to_string(), src.to_string());

// Look up line/column
let loc = file.lookup_line_col(BytePos::new(12));
assert_eq!(loc.line, 2);  // Line 2
assert_eq!(loc.col, 3);   // Column 3
```

## Span Operations

### Creating Spans

```rust
// From byte positions
let span = Span::new(BytePos::new(10), BytePos::new(20));

// From raw offsets
let span = Span::from_raw(10, 20);

// Dummy span for generated code
let span = Span::DUMMY;
```

### Combining Spans

```rust
let span1 = Span::from_raw(10, 20);
let span2 = Span::from_raw(15, 30);

// Merge: smallest enclosing span
let merged = span1.merge(span2);  // [10, 30)

// To: from start of first to end of second
let combined = span1.to(span2);   // [10, 30)
```

### Shrinking Spans

```rust
let span = Span::from_raw(10, 20);

// Point at start (for "expected X here" messages)
let point = span.shrink_to_lo();  // [10, 10)

// Point at end (for "missing X after Y" messages)
let point = span.shrink_to_hi();  // [20, 20)
```

## Spanned Values

Attach spans to AST nodes:

```rust
use bhc_span::Spanned;

#[derive(Debug)]
enum Expr {
    Var(String),
    Lit(i64),
}

let expr = Spanned::new(Expr::Lit(42), Span::from_raw(5, 7));

// Access the value
assert!(matches!(expr.node, Expr::Lit(42)));

// Access the span
assert_eq!(expr.span.len(), 2);

// Transform while preserving span
let mapped = expr.map(|e| format!("{:?}", e));
```

## Source File Management

```rust
use bhc_span::{SourceFile, FileId, BytePos, Span};

let source = r#"module Main where

main :: IO ()
main = putStrLn "Hello"
"#;

let file = SourceFile::new(
    FileId::new(0),
    "Main.hs".to_string(),
    source.to_string(),
);

// Line/column lookup
let loc = file.lookup_line_col(BytePos::new(20));
println!("{}:{}", loc.line, loc.col);

// Get line content
let line = file.line_content(2);  // 0-indexed
assert_eq!(line, Some("main :: IO ()"));

// Extract source text for a span
let span = Span::from_raw(0, 6);
assert_eq!(file.source_text(span), "module");

// Get span line info for rendering
let span_info = file.span_lines(span);
println!("Lines {}-{}", span_info.start_line, span_info.end_line);
```

## Multi-line Spans

```rust
let source = "if x > 0\n  then y\n  else z";
let file = SourceFile::new(FileId::new(0), "test.hs".to_string(), source.to_string());

let span = Span::from_raw(0, 26);  // Entire if-then-else
let info = file.span_lines(span);

assert!(info.is_multiline());
assert_eq!(info.start_line, 1);
assert_eq!(info.end_line, 3);
```

## Cross-file Spans

For spans that reference other files:

```rust
use bhc_span::{FullSpan, FileId, Span};

let span = FullSpan::new(
    FileId::new(1),
    Span::from_raw(100, 150),
);

// Used for "defined in X.hs" messages
println!("Defined in file {}", span.file.0);
```

## Integration with Diagnostics

See `bhc-diagnostics` for rendering spans in error messages:

```rust
// Spans enable precise error underlining:
//
//   Main.hs:3:5: error: Type mismatch
//       |
//     3 |     foo = bar + "string"
//       |           ^^^^^^^^^^^^^^
//       |
//   Expected: Int
//   Found: String
```

## Performance Notes

- `BytePos` and `Span` are `Copy` types (4-8 bytes)
- `FileId` is a simple `u32` wrapper
- Line index is computed once per file and cached
- O(log n) line lookup using binary search
