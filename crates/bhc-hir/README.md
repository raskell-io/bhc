# bhc-hir

High-level Intermediate Representation for the Basel Haskell Compiler.

## Overview

This crate defines the HIR (High-level IR), a desugared representation that bridges the gap between the parsed AST and the typed Core IR. HIR preserves name resolution results while simplifying syntactic constructs.

## Features

- Desugared patterns (guards, view patterns flattened)
- Resolved names with `DefId` references
- Explicit binding groups for mutual recursion
- Type annotations preserved for type checking
- Pattern match compilation preparation

## Key Types

| Type | Description |
|------|-------------|
| `HirId` | Unique identifier for HIR nodes |
| `DefId` | Definition identifier for resolved names |
| `DefRef` | Reference to a definition (local or external) |
| `Expr` | HIR expression enum |
| `Pat` | HIR pattern enum |
| `Item` | Top-level item (function, data, class, instance) |
| `Module` | Complete HIR module |
| `BindingGroup` | Group of mutually recursive bindings |

## Usage

### Lowering AST to HIR

```rust
use bhc_hir::{Module, HirId, DefId};
use bhc_lower::lower_module;

let hir_module = lower_module(&ast_module, &session)?;

// Access definitions
for item in &hir_module.items {
    match item {
        Item::Function(func) => {
            println!("Function {:?}: {:?}", func.def_id, func.name);
        }
        Item::Data(data) => {
            println!("Data type {:?}", data.name);
        }
        _ => {}
    }
}
```

### Working with HIR IDs

```rust
use bhc_hir::{HirId, DefId, DefRef};

// HIR IDs are unique within a module
let expr_id: HirId = hir_arena.alloc_expr(expr);

// DefIds are unique across the compilation
let def_id: DefId = resolver.define(name, namespace);

// DefRef points to either local or external definitions
let def_ref = DefRef::Local(def_id);
let def_ref = DefRef::External(crate_num, def_id);
```

## Expression Variants

```rust
pub enum Expr {
    Var(DefRef, HirId),           // Variable reference
    Lit(Literal, HirId),          // Literal value
    App(Box<Expr>, Box<Expr>),    // Application
    Lam(Vec<Pat>, Box<Expr>),     // Lambda abstraction
    Let(BindingGroup, Box<Expr>), // Let binding
    Case(Box<Expr>, Vec<Alt>),    // Case expression
    If(Box<Expr>, Box<Expr>, Box<Expr>), // If-then-else
    Tuple(Vec<Expr>),             // Tuple construction
    List(Vec<Expr>),              // List construction
    RecordCon(DefRef, Vec<FieldBind>),   // Record construction
    RecordUpd(Box<Expr>, Vec<FieldBind>), // Record update
    TypeAnn(Box<Expr>, Type),     // Type annotation
    Error(HirId),                 // Error placeholder
}
```

## Pattern Variants

```rust
pub enum Pat {
    Wildcard(HirId),              // _
    Var(DefId, HirId),            // x (binding)
    Lit(Literal, HirId),          // 42
    Con(DefRef, Vec<Pat>, HirId), // Just x
    Tuple(Vec<Pat>, HirId),       // (a, b)
    List(Vec<Pat>, HirId),        // [a, b]
    As(DefId, Box<Pat>, HirId),   // xs@(x:_)
    Bang(Box<Pat>, HirId),        // !x (strict)
    Or(Vec<Pat>, HirId),          // p1 | p2
}
```

## Binding Groups

Mutually recursive definitions are grouped together:

```rust
pub struct BindingGroup {
    pub bindings: Vec<Binding>,
    pub is_recursive: bool,
}

pub struct Binding {
    pub def_id: DefId,
    pub name: Symbol,
    pub expr: Expr,
    pub sig: Option<TypeSig>,
}
```

## Desugaring Performed

| Source | HIR |
|--------|-----|
| `if c then t else e` | `case c of { True -> t; False -> e }` |
| Pattern guards | Nested case expressions |
| `do { stmts }` | Nested binds and sequences |
| List comprehensions | `concatMap` and `filter` |
| `where` clauses | `let` expressions |
| Multi-clause functions | Single function with case |
| Operator sections | Lambda expressions |

## Design Notes

- HIR is the input to type checking
- Name resolution is complete at this stage
- Pattern match exhaustiveness not yet checked
- Types are optional (inferred during type checking)
- Maintains source spans for error reporting

## Related Crates

- `bhc-ast` - Input AST from parsing
- `bhc-lower` - AST to HIR lowering
- `bhc-typeck` - Type checks HIR
- `bhc-hir-to-core` - HIR to Core lowering
- `bhc-span` - Source locations

## Specification References

- H26-SPEC Section 3.2: HIR Definition
- H26-SPEC Section 4: Type System (uses HIR)
