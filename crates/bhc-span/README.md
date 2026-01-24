# bhc-span

Source location tracking and span management for the Basel Haskell Compiler.

## Overview

This crate provides types for tracking source locations throughout the compilation pipeline, enabling accurate error reporting and source mapping.

## Key Types

| Type | Description |
|------|-------------|
| `BytePos` | A byte offset into a source file |
| `Span` | A half-open byte range `[lo, hi)` representing a region of source code |
| `Spanned<T>` | A value with an associated span |
| `FileId` | Unique identifier for a source file |
| `FullSpan` | A span with an associated file ID for cross-file spans |
| `LineCol` | Line and column information (1-indexed) |
| `SourceFile` | Information about a source file including content and line offsets |

## Usage

```rust
use bhc_span::{Span, BytePos, SourceFile, FileId};

// Create a span from byte positions
let span = Span::new(BytePos::new(10), BytePos::new(20));
assert_eq!(span.len(), 10);

// Create a source file and look up line/column
let file = SourceFile::new(
    FileId::new(0),
    "example.hs".to_string(),
    "main = putStrLn \"Hello\"".to_string()
);

let loc = file.lookup_line_col(BytePos::new(7));
assert_eq!(loc.line, 1);
assert_eq!(loc.col, 8);

// Merge spans
let span1 = Span::from_raw(10, 20);
let span2 = Span::from_raw(15, 30);
let merged = span1.merge(span2);
assert_eq!(merged, Span::from_raw(10, 30));
```

## Features

- Zero-cost span representation (just two `u32` values)
- Efficient line/column lookup with precomputed line starts
- Span merging and manipulation operations
- Serialization support via `serde`

## Design Notes

- Spans use byte offsets rather than character offsets for efficiency
- The `DUMMY` span (`[0, 0)`) is used for generated code or when location is irrelevant
- Line and column numbers are 1-indexed to match editor conventions

## Related Crates

- `bhc-diagnostics` - Uses spans for error reporting
- `bhc-lexer` - Attaches spans to tokens
- `bhc-parser` - Attaches spans to AST nodes
