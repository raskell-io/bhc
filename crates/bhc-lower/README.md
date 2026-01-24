# bhc-lower

AST to HIR lowering for the Basel Haskell Compiler.

## Overview

This crate implements the lowering pass from surface AST to HIR (High-Level Intermediate Representation). It performs desugaring, name resolution, and pattern compilation, transforming the parsed syntax tree into a form suitable for type checking.

## Pipeline Position

```
Source Code
    │
    ▼
[Parse/AST]  ← Surface syntax
    │
    ▼
[Lower]      ← THIS CRATE
    │
    ▼
[HIR]        ← Desugared, resolved
    │
    ▼
[Type Check] ← Type inference
```

## Features

- **Desugaring**: Expand syntactic sugar (do-notation, list comprehensions, sections)
- **Name resolution**: Resolve all identifiers to their definitions
- **Pattern compilation**: Convert complex patterns and guards
- **Module loading**: Load and cache imported modules
- **Multi-file support**: Handle import dependencies

## Key Types

| Type | Description |
|------|-------------|
| `LowerContext` | Lowering context with scope and definitions |
| `DefMap` | Map from definition IDs to their info |
| `Scope` | Current binding scope |
| `ScopeId` | Unique scope identifier |
| `LowerConfig` | Configuration for the lowering pass |
| `ModuleCache` | Cache for loaded modules |
| `ModuleExports` | Exported definitions from a module |

## Usage

### Lowering a Module

```rust
use bhc_lower::{lower_module, LowerConfig, LowerContext};
use camino::Utf8PathBuf;

let ast_module: bhc_ast::Module = parse(...)?;
let mut ctx = LowerContext::new();
let config = LowerConfig {
    include_builtins: true,
    warn_unused: false,
    search_paths: vec![Utf8PathBuf::from("/path/to/sources")],
};
let hir_module = lower_module(&mut ctx, &ast_module, &config)?;
```

### Accessing Definition Information

```rust
use bhc_lower::{LowerContext, DefKind};

let ctx = LowerContext::new();

// After lowering, query definition info
for (def_id, info) in ctx.def_map() {
    match info.kind {
        DefKind::Value => println!("Value: {}", info.name),
        DefKind::Type => println!("Type: {}", info.name),
        DefKind::Constructor => println!("Constructor: {}", info.name),
    }
}
```

## Desugaring Transformations

| Source Syntax | Lowered Form |
|--------------|--------------|
| `do { x <- m; e }` | `m >>= \x -> e` |
| `[x \| x <- xs, p x]` | `concatMap (\x -> if p x then [x] else []) xs` |
| `if c then t else e` | `case c of { True -> t; False -> e }` |
| `(+ 1)` | `\x -> x + 1` |
| `(1 +)` | `\x -> 1 + x` |
| `f x y where z = e` | `let z = e in f x y` |
| `f a b = e₁; f c d = e₂` | `f = \x y -> case (x, y) of ...` |

## Name Resolution

The lowering pass resolves all names to unique `DefId` values:

```rust
pub enum DefKind {
    Value,       // Function or variable
    Type,        // Type constructor
    Constructor, // Data constructor
    Class,       // Type class
    Module,      // Module name
}

pub struct DefInfo {
    pub id: DefId,
    pub name: Symbol,
    pub kind: DefKind,
    pub span: Span,
}
```

### Scope Handling

```rust
// Push a new scope for a let binding
ctx.push_scope();

// Define a local variable
let def_id = ctx.define("x", DefKind::Value, span)?;

// Lookup resolves through scope chain
let resolved = ctx.lookup("x")?;

// Pop scope when done
ctx.pop_scope();
```

## Error Types

```rust
pub enum LowerError {
    /// Unbound variable reference
    UnboundVar { name: String, span: Span },

    /// Unbound type reference
    UnboundType { name: String, span: Span },

    /// Unbound constructor reference
    UnboundCon { name: String, span: Span },

    /// Duplicate definition in same scope
    DuplicateDefinition {
        name: String,
        new_span: Span,
        existing_span: Span,
    },

    /// Invalid pattern in binding position
    InvalidPattern { reason: String, span: Span },

    /// Unsupported syntax
    Unsupported { feature: String, span: Span },

    /// Multiple collected errors
    Multiple(Vec<LowerError>),
}
```

## Module Loading

The lowering pass can load imported modules:

```rust
use bhc_lower::{ModuleCache, ModuleExports, LoadError};

let mut cache = ModuleCache::new();

// Load a module
let exports = cache.load_module("Data.List", &search_paths)?;

// Access exported names
for (name, info) in &exports.values {
    println!("Exported value: {}", name);
}
```

## Warnings

```rust
pub enum LowerWarning {
    /// A stub definition was used (external package placeholder)
    StubUsed {
        name: String,
        span: Span,
        kind: &'static str,  // "value", "type", or "constructor"
    },
}
```

## Modules

| Module | Description |
|--------|-------------|
| `context` | Lowering context and scope management |
| `desugar` | Syntactic sugar expansion |
| `loader` | Module loading and caching |
| `lower` | Main lowering implementation |
| `resolve` | Name resolution |

## Design Notes

- Error recovery continues after failures when possible
- Scopes are nested to support lexical scoping
- Built-in definitions (Prelude) are optionally included
- Qualified names are resolved through module exports
- Source locations are preserved for error reporting

## Related Crates

- `bhc-ast` - Input AST from parsing
- `bhc-hir` - Output HIR types
- `bhc-typeck` - Type checking (consumes HIR)
- `bhc-span` - Source locations
- `bhc-intern` - Symbol interning

## Specification References

- H26-SPEC Section 3.1: Surface Syntax
- H26-SPEC Section 3.2: HIR Definition
- H26-SPEC Section 3.4: Desugaring Rules
