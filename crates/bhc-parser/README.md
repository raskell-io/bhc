# bhc-parser

Parser for Haskell 2026 source code in the Basel Haskell Compiler.

## Overview

This crate provides a recursive descent parser that produces an AST from a token stream. It handles the full Haskell 2026 syntax including GHC extensions, with comprehensive error recovery and diagnostic reporting.

## Key Types

| Type | Description |
|------|-------------|
| `Parser` | The main parser struct |
| `ParseError` | Parsing error types |
| `ParseResult<T>` | Result type for parse operations |

## Usage

### Parsing a Module

```rust
use bhc_parser::parse_module;
use bhc_span::FileId;

let source = r#"
module Example where

import Data.List

factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)
"#;

let file_id = FileId::new(0);
let (module, diagnostics) = parse_module(source, file_id);

if let Some(module) = module {
    println!("Parsed module: {:?}", module.name);
    println!("Declarations: {}", module.decls.len());
}

for diag in diagnostics {
    eprintln!("{:?}", diag);
}
```

### Parsing an Expression

```rust
use bhc_parser::parse_expr;
use bhc_span::FileId;

let (expr, diagnostics) = parse_expr("\\x -> x + 1", FileId::new(0));

if let Some(expr) = expr {
    // Use the expression
}
```

## Supported Syntax

### Module Structure

- Module declarations with exports
- Import declarations (qualified, hiding, as)
- Pragmas (LANGUAGE, OPTIONS_GHC, INLINE, etc.)

### Declarations

- Type signatures (including multi-line)
- Function bindings with pattern matching
- Data types with deriving
- Newtypes
- Type aliases
- Type classes with associated types
- Instances with associated type definitions
- Foreign imports/exports
- Fixity declarations

### Expressions

- Literals (int, float, char, string)
- Variables and constructors (qualified)
- Function application
- Lambda expressions
- Let/in expressions
- If/then/else
- Case expressions
- Do notation
- Tuples and lists
- List comprehensions
- Arithmetic sequences
- Record construction and update
- Infix operators (with sections)
- Type annotations

### Patterns

- Wildcards, variables, literals
- Constructor patterns
- Infix patterns (e.g., `x:xs`)
- As-patterns (`xs@(x:_)`)
- Lazy patterns (`~pat`)
- Bang patterns (`!pat`)
- Record patterns
- View patterns

### Types

- Type variables and constructors
- Function types
- Tuple and list types
- Forall quantification
- Constrained types
- Strict/lazy annotations

## Error Recovery

The parser attempts to recover from errors to report multiple issues:

```rust
let source = "f x = let y = in y";  // Missing expression after '='
let (module, diagnostics) = parse_module(source, file_id);

// Parser reports error but may continue parsing
assert!(!diagnostics.is_empty());
```

## Layout Handling

The parser works with virtual tokens inserted by the lexer for layout:

```haskell
-- This input:
f x = case x of
  Just y -> y
  Nothing -> 0

-- Is seen by the parser as:
f x = case x of { Just y -> y ; Nothing -> 0 }
```

## Error Messages

Parse errors are converted to diagnostics with source locations:

```rust
impl ParseError {
    pub fn to_diagnostic(&self, file: FileId) -> Diagnostic {
        match self {
            Self::Unexpected { found, expected, span } =>
                Diagnostic::error(format!("unexpected {}, expected {}", found, expected))
                    .with_label(FullSpan::new(file, *span), "unexpected token here"),
            // ...
        }
    }
}
```

## GHC Extensions

The parser supports many GHC extensions:

- `LambdaCase`: `\case { Just x -> x; Nothing -> 0 }`
- `MultiWayIf`: `if | x > 0 -> 1 | otherwise -> 0`
- `PatternGuards`: `f x | Just y <- g x = y`
- `ViewPatterns`: `f (view -> pat) = ...`
- `RecordWildCards`: `f Foo{..} = ...`
- `BangPatterns`: `f !x = x`
- `TypeFamilies`: Associated type declarations/definitions

## Design Notes

- Recursive descent parser for predictable error handling
- Operator precedence handled via precedence climbing
- Multi-clause functions are merged during parsing
- Doc comments and pragmas are preserved in the AST

## Related Crates

- `bhc-lexer` - Produces token stream consumed by parser
- `bhc-ast` - AST types produced by parser
- `bhc-diagnostics` - Error reporting
- `bhc-span` - Source locations
