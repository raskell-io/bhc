# bhc-macros

Procedural macros for the Basel Haskell Compiler.

## Overview

`bhc-macros` provides procedural macros to reduce boilerplate and ensure consistency across the BHC codebase. Features:

- **Derive macros**: Internable, AstNode, IrNode
- **Attribute macros**: Query definitions
- **Function-like macros**: Error codes, token definitions

## Derive Macros

### Internable

Generate interning support for string-like types:

```rust
use bhc_macros::Internable;

#[derive(Internable)]
pub struct Symbol(String);

// Generated implementations:
// - Symbol::new(s) - Create from string
// - Symbol::as_str() - Get string reference
// - Display, AsRef<str>, From<String>, From<&str>, Borrow<str>
```

### AstNode

Generate common AST node traits:

```rust
use bhc_macros::AstNode;

#[derive(AstNode)]
pub struct Expression {
    pub kind: ExprKind,
    pub span: Span,
}

// Generated:
// - Debug implementation
// - dummy() test helper (when Default is implemented)
```

### IrNode

Generate common IR node traits:

```rust
use bhc_macros::IrNode;

#[derive(IrNode)]
pub struct CoreExpr {
    pub kind: CoreExprKind,
    pub ty: Type,
}

// Generated:
// - is_normal_form() method
// - size() method for complexity analysis
```

### EnumDispatch

Generate enum dispatch patterns:

```rust
use bhc_macros::EnumDispatch;

#[derive(EnumDispatch)]
pub enum Expr {
    Lit(LitExpr),
    Var(VarExpr),
    App(AppExpr),
}

// Generated:
// - map() method for applying functions
```

## Attribute Macros

### query

Define queries for the query system:

```rust
use bhc_macros::query;

#[query]
fn type_of(db: &dyn Database, expr: ExprId) -> Type {
    // Query implementation
}

// Optional: specify query name
#[query("TypeOfQuery")]
fn type_of(db: &dyn Database, expr: ExprId) -> Type {
    // Query implementation
}

// Generated:
// - Tracing instrumentation in debug builds
// - Query execution logging
```

## Function-Like Macros

### error_codes

Define diagnostic error codes:

```rust
use bhc_macros::error_codes;

error_codes! {
    E0001: "type mismatch",
    E0002: "undefined variable",
    E0003: "ambiguous type",
}

// Generated:
// pub mod error_codes {
//     pub const E0001: &str = "E0001";
//     pub const E0002: &str = "E0002";
//     pub const E0003: &str = "E0003";
// }
```

### define_tokens

Define token kinds for the lexer:

```rust
use bhc_macros::define_tokens;

define_tokens! {
    // Keywords
    Let = "let",
    In = "in",
    Where = "where",
    Case = "case",
    Of = "of",

    // Operators
    Plus = "+",
    Minus = "-",
    Arrow = "->",

    // Delimiters
    LParen = "(",
    RParen = ")",
}

// Generated:
// pub enum TokenKind {
//     Eof,
//     Error,
//     Let,
//     In,
//     Where,
//     ...
// }
```

### impl_visitor

Generate visitor pattern implementations:

```rust
use bhc_macros::impl_visitor;

impl_visitor! {
    Expr {
        Lit(LitExpr),
        Var(VarExpr),
        App(AppExpr),
    }
}

// Generated:
// pub trait ExprVisitor {
//     fn visit_lit(&mut self, node: &LitExpr);
//     fn visit_var(&mut self, node: &VarExpr);
//     fn visit_app(&mut self, node: &AppExpr);
// }
```

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
bhc-macros = { path = "../bhc-macros" }
```

Import and use:

```rust
use bhc_macros::{Internable, AstNode, query};

#[derive(Internable)]
pub struct Identifier(String);

#[derive(AstNode)]
pub struct Function {
    pub name: Identifier,
    pub params: Vec<Param>,
    pub body: Expr,
    pub span: Span,
}

#[query]
fn resolve_function(db: &dyn Database, id: FunctionId) -> ResolvedFunction {
    // ...
}
```

## Design Principles

1. **Reduce boilerplate**: Common patterns should be one-liners
2. **Compile-time guarantees**: Catch errors early
3. **Consistency**: Ensure uniform implementations across codebase
4. **Transparency**: Generated code should be understandable

## Error Messages

The macros strive to produce clear error messages:

```
error: Internable can only be derived for tuple structs with a single String field
  --> src/lib.rs:3:10
   |
3  | #[derive(Internable)]
   |          ^^^^^^^^^^
```

## See Also

- `bhc-intern`: Interning infrastructure
- `bhc-ast`: AST node definitions
- `bhc-core`: Core IR node definitions
- `bhc-query`: Query system
- syn/quote documentation for proc-macro development
