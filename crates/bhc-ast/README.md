# bhc-ast

Abstract syntax tree definitions for the Basel Haskell Compiler.

## Overview

This crate defines the AST produced by parsing Haskell 2026 source code. The AST preserves source locations and syntactic structure, supporting the full range of Haskell syntax including GHC extensions.

## Key Types

### Module Structure

| Type | Description |
|------|-------------|
| `Module` | A complete Haskell module |
| `ModuleName` | Qualified module name (`Data.List`) |
| `ImportDecl` | Import declaration with qualifiers and spec |
| `Export` | Export specification |
| `Pragma` | Language/compiler pragmas |

### Declarations

| Type | Description |
|------|-------------|
| `Decl` | Top-level declaration enum |
| `TypeSig` | Type signature (`foo :: Int -> Int`) |
| `FunBind` | Function binding with clauses |
| `DataDecl` | Data type declaration |
| `NewtypeDecl` | Newtype declaration |
| `TypeAlias` | Type synonym |
| `ClassDecl` | Type class definition |
| `InstanceDecl` | Type class instance |
| `ForeignDecl` | Foreign import/export |
| `FixityDecl` | Fixity declaration |

### Expressions

| Type | Description |
|------|-------------|
| `Expr` | Expression enum (30+ variants) |
| `Lit` | Literal values |
| `Alt` | Case alternative |
| `Stmt` | Do-notation statement |
| `FieldBind` | Record field binding |

### Patterns

| Type | Description |
|------|-------------|
| `Pat` | Pattern enum |
| `FieldPat` | Record field pattern |

### Types

| Type | Description |
|------|-------------|
| `Type` | Type expression |
| `TyVar` | Type variable |
| `Constraint` | Type class constraint |
| `Kind` | Kind expression |

## Usage

```rust
use bhc_ast::{Module, Expr, Pat, Type, Decl};

// The AST is typically produced by the parser
let module: Module = parse_module(source)?;

// Access module components
for import in &module.imports {
    println!("Importing: {}", import.module.to_string());
}

for decl in &module.decls {
    match decl {
        Decl::FunBind(fun) => {
            println!("Function: {}", fun.name.name.as_str());
        }
        Decl::DataDecl(data) => {
            println!("Data type: {}", data.name.name.as_str());
        }
        _ => {}
    }
}
```

## Expression Variants

```rust
pub enum Expr {
    Var(Ident, Span),                    // x
    QualVar(ModuleName, Ident, Span),    // M.x
    Con(Ident, Span),                    // Just
    QualCon(ModuleName, Ident, Span),    // M.Just
    Lit(Lit, Span),                      // 42, "hello"
    App(Box<Expr>, Box<Expr>, Span),     // f x
    Lam(Vec<Pat>, Box<Expr>, Span),      // \x -> e
    Let(Vec<Decl>, Box<Expr>, Span),     // let ... in e
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),  // if c then t else e
    Case(Box<Expr>, Vec<Alt>, Span),     // case e of { ... }
    Do(Vec<Stmt>, Span),                 // do { ... }
    Tuple(Vec<Expr>, Span),              // (a, b)
    List(Vec<Expr>, Span),               // [a, b, c]
    // ... and more
}
```

## Pattern Variants

```rust
pub enum Pat {
    Wildcard(Span),                      // _
    Var(Ident, Span),                    // x
    Lit(Lit, Span),                      // 42
    Con(Ident, Vec<Pat>, Span),          // Just x
    Infix(Box<Pat>, Ident, Box<Pat>, Span),  // x : xs
    Tuple(Vec<Pat>, Span),               // (a, b)
    List(Vec<Pat>, Span),                // [a, b]
    Record(Ident, Vec<FieldPat>, Span),  // Foo { bar = x }
    As(Ident, Box<Pat>, Span),           // xs@(x:_)
    Lazy(Box<Pat>, Span),                // ~pat
    Bang(Box<Pat>, Span),                // !pat
    // ... and more
}
```

## Type Variants

```rust
pub enum Type {
    Var(TyVar, Span),                    // a
    Con(Ident, Span),                    // Int
    QualCon(ModuleName, Ident, Span),    // M.Map
    App(Box<Type>, Box<Type>, Span),     // Maybe Int
    Fun(Box<Type>, Box<Type>, Span),     // a -> b
    Tuple(Vec<Type>, Span),              // (a, b)
    List(Box<Type>, Span),               // [a]
    Forall(Vec<TyVar>, Box<Type>, Span), // forall a. a -> a
    Constrained(Vec<Constraint>, Box<Type>, Span),  // Eq a => a -> Bool
    Bang(Box<Type>, Span),               // !Int (strict field)
    Lazy(Box<Type>, Span),               // ~Int (lazy field)
    // M9 extensions
    PromotedList(Vec<Type>, Span),       // '[1024, 768]
    NatLit(u64, Span),                   // 1024 (type-level)
}
```

## Language Extensions Supported

The AST supports many GHC extensions:

- **Type System**: GADTs, TypeFamilies, DataKinds, RankNTypes
- **Syntax**: LambdaCase, MultiWayIf, PatternGuards, ViewPatterns
- **Strictness**: BangPatterns, StrictData
- **Deriving**: DerivingVia, DerivingStrategies
- **Records**: RecordWildCards, NamedFieldPuns
- **FFI**: ForeignFunctionInterface

## Typed Indices

The AST uses typed indices from `bhc-index` for efficient arena allocation:

```rust
use bhc_index::define_index;

define_index! {
    pub struct ExprId;
    pub struct PatId;
    pub struct TypeId;
    pub struct DeclId;
}
```

## Design Notes

- All AST nodes carry `Span` for source locations
- Names use `Symbol` (interned) for efficient comparison
- Qualified names are represented explicitly
- The AST is syntax-oriented (desugaring happens in lowering)

## Related Crates

- `bhc-span` - Source locations
- `bhc-intern` - Symbol interning for identifiers
- `bhc-index` - Typed indices for arenas
- `bhc-parser` - Produces AST from tokens
- `bhc-lower` - Lowers AST to HIR
