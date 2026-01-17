# bhc-parser

Recursive-descent parser for Haskell 2026.

## Overview

`bhc-parser` transforms a token stream into an Abstract Syntax Tree (AST). It implements:

- **Recursive descent**: Predictable, debuggable parsing
- **Operator precedence**: Pratt parsing for expressions
- **Error recovery**: Continue parsing after errors
- **Full H26 support**: All Haskell 2026 syntax

## Core Types

| Type | Description |
|------|-------------|
| `Parser` | Main parser state |
| `ParseResult<T>` | Result with diagnostic support |
| `ParseError` | Structured parse errors |
| `Precedence` | Operator precedence levels |

## Quick Start

```rust
use bhc_lexer::Lexer;
use bhc_parser::Parser;
use bhc_span::SourceFile;

let source = r#"
module Main where

main :: IO ()
main = putStrLn "Hello, World!"
"#;

let file = SourceFile::new(FileId::new(0), "Main.hs".into(), source.into());
let lexer = Lexer::new(&file);
let tokens: Vec<Token> = lexer.collect();

let mut parser = Parser::new(&tokens);
let module = parser.parse_module()?;
```

## Parsing Methods

### Module Level

```rust
impl Parser {
    /// Parse a complete module
    pub fn parse_module(&mut self) -> ParseResult<Module>;

    /// Parse module header: `module Foo.Bar (exports) where`
    pub fn parse_module_header(&mut self) -> ParseResult<Option<ModuleHeader>>;

    /// Parse import declaration
    pub fn parse_import(&mut self) -> ParseResult<ImportDecl>;

    /// Parse a single declaration
    pub fn parse_decl(&mut self) -> ParseResult<Decl>;
}
```

### Declarations

```rust
impl Parser {
    /// Parse type signature: `foo :: Int -> Int`
    pub fn parse_type_sig(&mut self) -> ParseResult<TypeSig>;

    /// Parse function binding: `foo x = x + 1`
    pub fn parse_fun_bind(&mut self) -> ParseResult<FunBind>;

    /// Parse data declaration: `data Maybe a = Nothing | Just a`
    pub fn parse_data_decl(&mut self) -> ParseResult<DataDecl>;

    /// Parse class declaration: `class Eq a where ...`
    pub fn parse_class_decl(&mut self) -> ParseResult<ClassDecl>;

    /// Parse instance declaration: `instance Eq Int where ...`
    pub fn parse_instance_decl(&mut self) -> ParseResult<InstanceDecl>;
}
```

### Expressions

```rust
impl Parser {
    /// Parse any expression
    pub fn parse_expr(&mut self) -> ParseResult<Expr>;

    /// Parse with operator precedence
    pub fn parse_expr_prec(&mut self, min_prec: Precedence) -> ParseResult<Expr>;

    /// Parse atomic expression (no operators)
    pub fn parse_atom(&mut self) -> ParseResult<Expr>;
}
```

### Patterns

```rust
impl Parser {
    /// Parse a pattern
    pub fn parse_pat(&mut self) -> ParseResult<Pat>;

    /// Parse atomic pattern (no infix constructors)
    pub fn parse_apat(&mut self) -> ParseResult<Pat>;
}
```

### Types

```rust
impl Parser {
    /// Parse a type expression
    pub fn parse_type(&mut self) -> ParseResult<Type>;

    /// Parse type with context: `Eq a => a -> a -> Bool`
    pub fn parse_qualified_type(&mut self) -> ParseResult<Type>;

    /// Parse atomic type
    pub fn parse_atype(&mut self) -> ParseResult<Type>;
}
```

## Operator Precedence

The parser uses Pratt parsing for expressions:

```rust
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precedence {
    Lowest,      // 0
    Or,          // 2  (||)
    And,         // 3  (&&)
    Compare,     // 4  (==, /=, <, >, <=, >=)
    Concat,      // 5  (++, <>)
    Add,         // 6  (+, -)
    Mul,         // 7  (*, /, `div`, `mod`)
    Power,       // 8  (^, ^^, **)
    Compose,     // 9  (.)
    App,         // 10 (function application)
}

#[derive(Clone, Copy)]
pub enum Assoc {
    Left,
    Right,
    None,
}
```

### Operator Table

Standard Haskell fixities:

```rust
const OPERATORS: &[(&str, Precedence, Assoc)] = &[
    ("$",  Precedence::Lowest,  Assoc::Right),
    ("||", Precedence::Or,      Assoc::Right),
    ("&&", Precedence::And,     Assoc::Right),
    ("==", Precedence::Compare, Assoc::None),
    ("/=", Precedence::Compare, Assoc::None),
    ("<",  Precedence::Compare, Assoc::None),
    (">",  Precedence::Compare, Assoc::None),
    ("<=", Precedence::Compare, Assoc::None),
    (">=", Precedence::Compare, Assoc::None),
    ("++", Precedence::Concat,  Assoc::Right),
    ("<>", Precedence::Concat,  Assoc::Right),
    ("+",  Precedence::Add,     Assoc::Left),
    ("-",  Precedence::Add,     Assoc::Left),
    ("*",  Precedence::Mul,     Assoc::Left),
    ("/",  Precedence::Mul,     Assoc::Left),
    ("^",  Precedence::Power,   Assoc::Right),
    (".",  Precedence::Compose, Assoc::Right),
];
```

### Pratt Parsing

```rust
impl Parser {
    fn parse_expr_prec(&mut self, min_prec: Precedence) -> ParseResult<Expr> {
        let mut lhs = self.parse_atom()?;

        while let Some((op, prec, assoc)) = self.peek_operator() {
            if prec < min_prec {
                break;
            }

            self.advance(); // consume operator

            let next_prec = match assoc {
                Assoc::Left => Precedence::from(prec.0 + 1),
                Assoc::Right => prec,
                Assoc::None => Precedence::from(prec.0 + 1),
            };

            let rhs = self.parse_expr_prec(next_prec)?;
            lhs = Expr::InfixApp(Box::new(lhs), op, Box::new(rhs));
        }

        Ok(lhs)
    }
}
```

## Error Recovery

The parser continues after errors to report multiple issues:

```rust
pub enum ParseError {
    /// Unexpected token
    Unexpected {
        found: TokenKind,
        expected: Vec<TokenKind>,
        span: Span,
    },

    /// Missing token
    Missing {
        expected: TokenKind,
        span: Span,
    },

    /// Invalid syntax construct
    InvalidSyntax {
        message: String,
        span: Span,
    },

    /// Unmatched bracket
    UnmatchedBracket {
        kind: BracketKind,
        open_span: Span,
    },
}
```

### Recovery Strategies

```rust
impl Parser {
    /// Skip to next declaration
    fn recover_to_decl(&mut self) {
        while !self.at_eof() {
            if self.at_decl_start() {
                break;
            }
            self.advance();
        }
    }

    /// Skip to closing bracket
    fn recover_to_close(&mut self, close: TokenKind) {
        let mut depth = 1;
        while !self.at_eof() && depth > 0 {
            match self.current_kind() {
                k if k.is_open_bracket() => depth += 1,
                k if k == close => depth -= 1,
                _ => {}
            }
            self.advance();
        }
    }
}
```

## Import Parsing

```rust
pub struct ImportDecl {
    /// Is this a qualified import?
    pub qualified: bool,
    /// Module being imported
    pub module: ModuleName,
    /// Alias: `import Foo as F`
    pub alias: Option<ModuleName>,
    /// Import spec: `(foo, bar)` or `hiding (baz)`
    pub spec: Option<ImportSpec>,
    /// Is this a package import? (H26)
    pub package: Option<String>,
    pub span: Span,
}

// import qualified Data.Map as M (lookup, insert)
// import "base" Data.List hiding (map)
```

## GADT Parsing

```rust
// data Expr a where
//   Lit  :: Int -> Expr Int
//   Add  :: Expr Int -> Expr Int -> Expr Int
//   IsZero :: Expr Int -> Expr Bool

impl Parser {
    fn parse_gadt_decl(&mut self) -> ParseResult<GadtDecl> {
        self.expect(TokenKind::Data)?;
        let name = self.parse_ident()?;
        let ty_vars = self.parse_ty_var_binders()?;
        self.expect(TokenKind::Where)?;

        let constrs = self.parse_gadt_constructors()?;

        Ok(GadtDecl { name, ty_vars, constrs, .. })
    }

    fn parse_gadt_constructor(&mut self) -> ParseResult<GadtConDecl> {
        let name = self.parse_con_ident()?;
        self.expect(TokenKind::DoubleColon)?;
        let ty = self.parse_type()?;
        Ok(GadtConDecl { name, ty, .. })
    }
}
```

## Layout Handling

The parser consumes layout tokens from the lexer:

```rust
impl Parser {
    /// Check for virtual semicolon (new declaration)
    fn at_semi(&self) -> bool {
        matches!(
            self.current_kind(),
            TokenKind::Semi | TokenKind::VirtualSemi
        )
    }

    /// Check for virtual close brace (end of block)
    fn at_close(&self) -> bool {
        matches!(
            self.current_kind(),
            TokenKind::RBrace | TokenKind::VirtualRBrace
        )
    }

    /// Parse a layout block
    fn parse_layout_block<T>(
        &mut self,
        parse_item: impl Fn(&mut Self) -> ParseResult<T>,
    ) -> ParseResult<Vec<T>> {
        let mut items = Vec::new();

        // Expect open brace (explicit or virtual)
        self.expect_open_brace()?;

        while !self.at_close() && !self.at_eof() {
            items.push(parse_item(self)?);

            // Expect semicolon between items
            if !self.at_close() {
                self.expect_semi()?;
            }
        }

        self.expect_close_brace()?;
        Ok(items)
    }
}
```

## Do Notation

```rust
// do
//   x <- getLine
//   let y = process x
//   putStrLn y
//   return ()

impl Parser {
    fn parse_do_expr(&mut self) -> ParseResult<Expr> {
        self.expect(TokenKind::Do)?;
        let stmts = self.parse_layout_block(Self::parse_stmt)?;

        // Last statement must be expression
        if let Some(Stmt::Bind(..)) = stmts.last() {
            return Err(ParseError::InvalidSyntax {
                message: "do block must end with expression".into(),
                span: stmts.last().unwrap().span(),
            });
        }

        Ok(Expr::Do(stmts))
    }

    fn parse_stmt(&mut self) -> ParseResult<Stmt> {
        if self.at(TokenKind::Let) {
            // let x = 1
            self.advance();
            let decls = self.parse_layout_block(Self::parse_decl)?;
            Ok(Stmt::Let(decls))
        } else {
            let expr = self.parse_expr()?;

            if self.at(TokenKind::LeftArrow) {
                // x <- action
                self.advance();
                // The expr we parsed is actually a pattern
                let pat = self.expr_to_pat(expr)?;
                let rhs = self.parse_expr()?;
                Ok(Stmt::Bind(pat, rhs))
            } else {
                Ok(Stmt::Expr(expr))
            }
        }
    }
}
```

## Configuration

```rust
pub struct ParserConfig {
    /// Enable H26 extensions
    pub h26_extensions: bool,

    /// Enable M9 dependent types
    pub m9_dependent_types: bool,

    /// Maximum expression nesting depth
    pub max_depth: u32,

    /// Error recovery mode
    pub recover: bool,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            h26_extensions: true,
            m9_dependent_types: false,
            max_depth: 256,
            recover: true,
        }
    }
}
```

## Performance

- Single-pass parsing
- No backtracking for common constructs
- Token lookahead limited to 2 tokens
- Direct AST construction (no intermediate representation)

## Diagnostics Integration

```rust
impl Parser {
    pub fn parse_with_diagnostics(
        &mut self,
        handler: &mut DiagnosticHandler,
    ) -> Option<Module> {
        match self.parse_module() {
            Ok(module) => Some(module),
            Err(errors) => {
                for err in errors {
                    handler.emit(err.into_diagnostic());
                }
                None
            }
        }
    }
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn parse_expr(s: &str) -> Expr {
        let file = SourceFile::new(FileId::new(0), "test".into(), s.into());
        let tokens: Vec<_> = Lexer::new(&file).collect();
        let mut parser = Parser::new(&tokens);
        parser.parse_expr().unwrap()
    }

    #[test]
    fn test_simple_app() {
        let expr = parse_expr("f x y");
        assert!(matches!(expr, Expr::App(..)));
    }

    #[test]
    fn test_operator_precedence() {
        let expr = parse_expr("1 + 2 * 3");
        // Should parse as 1 + (2 * 3)
        match expr {
            Expr::InfixApp(_, op, _) => {
                assert_eq!(op.as_str(), "+");
            }
            _ => panic!("expected infix app"),
        }
    }
}
```
