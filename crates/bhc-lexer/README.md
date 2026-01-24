# bhc-lexer

Lexical analysis for the Basel Haskell Compiler.

## Overview

This crate provides a lexer for Haskell 2026 source code, producing a stream of tokens with source locations. The lexer handles Haskell's unique layout rule (significant indentation) by inserting virtual braces and semicolons.

## Features

- All Haskell 2010 tokens plus GHC and H26 extensions
- Unicode identifiers and operators
- Layout rule (significant indentation)
- Qualified names (`Data.List.map`)
- Documentation comments (`-- |`, `{- | -}`)
- Pragmas (`{-# LANGUAGE ... #-}`)
- CPP directive handling (`#if`, `#else`, `#endif`)

## Key Types

| Type | Description |
|------|-------------|
| `Lexer` | The main lexer struct, implements `Iterator` |
| `Token` | A lexical token with its kind |
| `TokenKind` | The type of token (keyword, identifier, literal, etc.) |
| `LexerConfig` | Configuration options for lexing |

## Usage

### Basic Lexing

```rust
use bhc_lexer::{lex, Lexer, TokenKind};

// Simple interface
let tokens = lex("let x = 1 in x");

// Iterator interface
let lexer = Lexer::new("let x = 1 in x");
for spanned_token in lexer {
    println!("{:?} at {:?}", spanned_token.node.kind, spanned_token.span);
}
```

### Custom Configuration

```rust
use bhc_lexer::{Lexer, LexerConfig};

let config = LexerConfig {
    preserve_doc_comments: true,
    preserve_pragmas: true,
    warn_tabs: false,
    tab_width: 4,
};

let lexer = Lexer::with_config(source, config);
```

## Layout Rule

Haskell uses significant indentation (the "layout rule") to delimit blocks. After `where`, `let`, `do`, or `of`, the lexer inserts virtual braces and semicolons based on indentation:

```haskell
f x = case x of       -- layout starts after 'of'
  Just y -> y         -- virtual '{' inserted, column 2
  Nothing -> 0        -- virtual ';' inserted (same column)
                      -- virtual '}' inserted (dedent or EOF)
```

### Virtual Tokens

| Token | Meaning |
|-------|---------|
| `VirtualLBrace` | Implicit block start |
| `VirtualRBrace` | Implicit block end |
| `VirtualSemi` | Implicit semicolon |

## Token Categories

### Keywords

`case`, `class`, `data`, `deriving`, `do`, `else`, `forall`, `if`, `import`, `in`, `infix`, `infixl`, `infixr`, `instance`, `let`, `module`, `newtype`, `of`, `qualified`, `then`, `type`, `where`, `foreign`, `lazy`

### Literals

- Integer: `42`, `0xFF`, `0o77`, `0b1010`
- Float: `3.14`, `1e10`, `2.5e-3`
- Character: `'a'`, `'\n'`, `'\x1F'`
- String: `"hello"`, `"line1\nline2"`

### Operators and Punctuation

- Arrows: `->`, `<-`, `=>`, `::`, `..`
- Unicode: `→`, `←`, `⇒`, `∷`, `∀`
- Punctuation: `(`, `)`, `[`, `]`, `{`, `}`, `,`, `;`, `` ` ``

### Special Tokens

- Pragmas: `{-# LANGUAGE GADTs #-}`
- Doc comments: `-- | description`, `{- | block -}`
- Qualified names: `Data.List.map`, `M.Just`

## Error Handling

Lexer errors are represented as `TokenKind::Error(LexError)`:

```rust
pub enum LexError {
    InvalidChar(char),
    UnterminatedString,
    UnterminatedChar,
    EmptyCharLiteral,
    MultiCharLiteral,
    InvalidEscape(char),
    InvalidUnicodeEscape,
    InvalidNumber(String),
}
```

## Design Notes

- The lexer is implemented as an iterator over `Spanned<Token>`
- Layout state is tracked via a stack of indentation levels
- Qualified names are recognized during lexing (not parsing)
- Unicode identifiers follow UAX #31 (XID_Start, XID_Continue)

## Related Crates

- `bhc-span` - Source locations attached to tokens
- `bhc-intern` - String interning for identifiers
- `bhc-parser` - Consumes token stream to produce AST
