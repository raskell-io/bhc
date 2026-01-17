# bhc-lexer

Lexical analysis for Haskell 2026 with full layout rule support.

## Overview

`bhc-lexer` transforms source text into a stream of tokens, handling:

- **Layout rule**: Haskell's significant indentation via virtual braces/semicolons
- **Unicode**: Full Unicode identifier and operator support
- **Qualified names**: `Data.List.map` lexed as single token
- **Literals**: Integers, floats, characters, strings (including multi-line)
- **Pragmas**: `{-# LANGUAGE ... #-}`, `{-# OPTIONS ... #-}`
- **Comments**: Line (`--`) and block (`{- -}`) with nesting

## Core Types

| Type | Description |
|------|-------------|
| `Lexer` | Main lexer state machine |
| `Token` | A lexed token with kind and span |
| `TokenKind` | The type of token (keyword, ident, operator, etc.) |
| `LexError` | Lexical errors with recovery |
| `LayoutStack` | Tracks indentation context |

## Quick Start

```rust
use bhc_lexer::{Lexer, TokenKind};
use bhc_span::SourceFile;

let source = "module Main where\n\nmain = putStrLn \"Hello\"";
let file = SourceFile::new(FileId::new(0), "Main.hs".into(), source.into());

let mut lexer = Lexer::new(&file);

while let Some(token) = lexer.next_token() {
    println!("{:?}: {:?}", token.kind, token.span);
}
```

## Token Kinds

### Keywords

```rust
pub enum TokenKind {
    // Reserved words
    Case, Class, Data, Default, Deriving, Do, Else,
    Forall, Foreign, If, Import, In, Infix, Infixl,
    Infixr, Instance, Let, Module, Newtype, Of,
    Qualified, Then, Type, Where,

    // H26 extensions
    Lazy, Strict, Profile, Edition,

    // ...
}
```

### Identifiers and Operators

```rust
// Identifiers
Ident,           // foo, Bar, _x
QualIdent,       // Data.List.map
ConId,           // True, Just
QualConId,       // Data.Maybe.Just

// Operators
VarSym,          // +, *, >>=
ConSym,          // :, :+:
QualVarSym,      // Data.List.++
QualConSym,      // Data.List.:
```

### Literals

```rust
// Numeric
IntLit,          // 42, 0xFF, 0o17, 0b1010
FloatLit,        // 3.14, 1e-10, 2.5e3

// Text
CharLit,         // 'a', '\n', '\x1F'
StringLit,       // "hello", "line1\nline2"
MultilineString, // """...""" (H26)
```

### Layout Tokens

```rust
// Virtual tokens inserted by layout rule
VirtualLBrace,   // Implicit {
VirtualRBrace,   // Implicit }
VirtualSemi,     // Implicit ;

// Explicit braces
LBrace,          // {
RBrace,          // }
Semi,            // ;
```

## Layout Rule

The lexer implements Haskell's layout rule, converting:

```haskell
module Main where

main = do
  putStrLn "Hello"
  putStrLn "World"
```

Into a token stream equivalent to:

```haskell
module Main where { main = do { putStrLn "Hello" ; putStrLn "World" } }
```

### Layout Contexts

```rust
pub struct LayoutStack {
    contexts: Vec<LayoutContext>,
}

pub struct LayoutContext {
    /// Column number for this context
    indent: u32,
    /// Whether this is explicit ({) or implicit (layout)
    explicit: bool,
    /// The keyword that opened this context
    opener: Option<TokenKind>,
}
```

### Layout Keywords

These keywords trigger layout:

| Keyword | Context |
|---------|---------|
| `where` | Module/class/instance body |
| `let` | Let bindings |
| `do` | Do notation |
| `of` | Case alternatives |
| `\case` | Lambda-case alternatives |

### Layout Algorithm

```rust
impl Lexer {
    fn handle_layout(&mut self, token: &Token) {
        match token.kind {
            // Keywords that open layout
            TokenKind::Where | TokenKind::Let |
            TokenKind::Do | TokenKind::Of => {
                self.pending_layout = true;
            }

            // Start of line - check indentation
            _ if token.at_line_start => {
                let col = token.span.start_col;

                // Close contexts with greater indentation
                while let Some(ctx) = self.layout.top() {
                    if !ctx.explicit && col < ctx.indent {
                        self.emit_virtual_rbrace();
                        self.layout.pop();
                    } else {
                        break;
                    }
                }

                // Same indentation = new item
                if let Some(ctx) = self.layout.top() {
                    if !ctx.explicit && col == ctx.indent {
                        self.emit_virtual_semi();
                    }
                }
            }
            _ => {}
        }
    }
}
```

## Unicode Support

### Identifiers

Valid identifier characters follow Unicode categories:

```rust
fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '\''
}
```

### Operators

Operator characters include Unicode symbols:

```rust
fn is_operator_char(c: char) -> bool {
    matches!(c,
        '!' | '#' | '$' | '%' | '&' | '*' | '+' | '.' |
        '/' | '<' | '=' | '>' | '?' | '@' | '\\' | '^' |
        '|' | '-' | '~' | ':'
    ) || c.is_symbol() || c.is_punctuation()
}
```

## Error Recovery

The lexer continues after errors:

```rust
pub enum LexError {
    /// Unterminated string literal
    UnterminatedString { start: BytePos },

    /// Unterminated block comment
    UnterminatedComment { start: BytePos },

    /// Invalid character in source
    InvalidChar { pos: BytePos, char: char },

    /// Invalid escape sequence
    InvalidEscape { pos: BytePos, seq: String },

    /// Invalid numeric literal
    InvalidNumber { span: Span, reason: String },
}

impl Lexer {
    pub fn next_token(&mut self) -> Option<Token> {
        loop {
            match self.try_next_token() {
                Ok(token) => return Some(token),
                Err(err) => {
                    self.errors.push(err);
                    self.recover();
                    // Continue lexing
                }
            }
        }
    }
}
```

## Pragmas

### Language Pragmas

```rust
// {-# LANGUAGE GADTs, TypeFamilies #-}
Pragma {
    kind: PragmaKind::Language,
    content: "GADTs, TypeFamilies",
}
```

### Options Pragmas

```rust
// {-# OPTIONS_GHC -Wall #-}
Pragma {
    kind: PragmaKind::Options,
    tool: Some("GHC"),
    content: "-Wall",
}
```

### Inline Pragmas

```rust
// {-# INLINE foo #-}
// {-# NOINLINE bar #-}
// {-# INLINABLE baz #-}
Pragma {
    kind: PragmaKind::Inline(InlineSpec::Inline),
    target: "foo",
}
```

## Qualified Names

Qualified names are lexed as single tokens:

```rust
// "Data.List.map" becomes:
Token {
    kind: TokenKind::QualIdent,
    span: Span { lo: 0, hi: 13 },
    // Full text preserved for later resolution
}

// Parts can be extracted:
fn split_qualified(s: &str) -> (&str, &str) {
    match s.rsplit_once('.') {
        Some((qual, name)) => (qual, name),
        None => ("", s),
    }
}
```

## Lexer Configuration

```rust
pub struct LexerConfig {
    /// Enable H26 extensions
    pub h26_extensions: bool,

    /// Enable GHC extensions for compatibility
    pub ghc_compat: bool,

    /// Track comments for documentation
    pub preserve_comments: bool,

    /// Maximum nesting depth for block comments
    pub max_comment_depth: u32,
}

impl Default for LexerConfig {
    fn default() -> Self {
        Self {
            h26_extensions: true,
            ghc_compat: false,
            preserve_comments: false,
            max_comment_depth: 100,
        }
    }
}
```

## Performance

- Single-pass lexing with minimal backtracking
- Layout tokens inserted on-the-fly
- Spans reference source directly (no string copies)
- Tokens are small (kind + span = 12 bytes)

## Integration

The lexer produces tokens consumed by `bhc-parser`:

```rust
use bhc_lexer::Lexer;
use bhc_parser::Parser;

let lexer = Lexer::new(&source_file);
let tokens: Vec<Token> = lexer.collect();
let mut parser = Parser::new(&tokens);
let module = parser.parse_module()?;
```
