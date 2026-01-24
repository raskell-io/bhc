# bhc-macros

Procedural macros for the Basel Haskell Compiler.

## Overview

This crate provides procedural macros used throughout the BHC compiler infrastructure to reduce boilerplate and provide compile-time guarantees.

## Available Macros

### Derive Macros

| Macro | Description |
|-------|-------------|
| `Internable` | Derive interning support for string-like types |
| `AstNode` | Derive common AST node traits |
| `IrNode` | Derive common IR node traits |

### Attribute Macros

| Macro | Description |
|-------|-------------|
| `#[query]` | Define a query for the query system |
| `#[salsa_query]` | Alternative query definition syntax |

## Usage

### Internable

Generate interning support for string-like wrapper types:

```rust
use bhc_macros::Internable;

#[derive(Internable)]
pub struct Symbol(String);

// Generated impl includes:
// - Symbol::new(s) - create from string
// - symbol.as_str() - get string reference
// - Display, AsRef<str>, From<String>, From<&str>
```

### AstNode

Derive common traits for AST nodes:

```rust
use bhc_macros::AstNode;

#[derive(AstNode)]
pub struct FunctionDef {
    pub name: Symbol,
    pub params: Vec<Param>,
    pub body: Expr,
    #[span]
    pub span: Span,
}

// Generated impl includes:
// - Spanned trait implementation
// - Pretty printing support
// - Visitor pattern support
```

### IrNode

Derive common traits for IR nodes:

```rust
use bhc_macros::IrNode;

#[derive(IrNode)]
pub struct CoreExpr {
    pub kind: ExprKind,
    pub ty: Type,
    #[span]
    pub span: Span,
}

// Generated impl includes:
// - Type accessors
// - Span accessors
// - Substitution support
```

### Query Attribute

Define queries for the incremental compilation system:

```rust
use bhc_macros::query;

#[query]
fn parse_module(db: &dyn Database, file: FileId) -> Arc<Module> {
    let source = db.file_source(file);
    parser::parse(&source)
}

// Generated code:
// - Query trait definition
// - Memoization wrapper
// - Dependency tracking
```

## Generated Code Examples

### Internable Expansion

```rust
#[derive(Internable)]
pub struct Identifier(String);

// Expands to:
impl Identifier {
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for Identifier {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<String> for Identifier {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Identifier {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl std::borrow::Borrow<str> for Identifier {
    fn borrow(&self) -> &str {
        &self.0
    }
}
```

### AstNode Expansion

```rust
#[derive(AstNode)]
pub struct LetExpr {
    pub bindings: Vec<Binding>,
    pub body: Box<Expr>,
    #[span]
    pub span: Span,
}

// Expands to:
impl Spanned for LetExpr {
    fn span(&self) -> Span {
        self.span
    }
}

impl AstNode for LetExpr {
    fn visit<V: AstVisitor>(&self, visitor: &mut V) {
        for binding in &self.bindings {
            binding.visit(visitor);
        }
        self.body.visit(visitor);
    }

    fn visit_mut<V: AstVisitorMut>(&mut self, visitor: &mut V) {
        for binding in &mut self.bindings {
            binding.visit_mut(visitor);
        }
        self.body.visit_mut(visitor);
    }
}
```

## Integration with Other Crates

| Crate | Usage |
|-------|-------|
| `bhc-intern` | Uses `Internable` for `Symbol` |
| `bhc-ast` | Uses `AstNode` for AST types |
| `bhc-hir` | Uses `IrNode` for HIR types |
| `bhc-core` | Uses `IrNode` for Core IR types |
| `bhc-query` | Uses `#[query]` for query definitions |

## Design Notes

- Macros are hygienic and work with any crate structure
- Generated code is optimized for the common case
- Errors provide helpful diagnostics with span information
- Optional features controlled via macro attributes

## Related Crates

- `bhc-intern` - String interning
- `bhc-ast` - AST types
- `bhc-hir` - HIR types
- `bhc-core` - Core IR types
- `bhc-query` - Query system
