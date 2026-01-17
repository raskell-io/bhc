# bhc-index

Type-safe indices for compiler data structures.

## Overview

`bhc-index` provides a pattern for creating type-safe indices that prevent accidentally mixing up indices from different collections. This is crucial for compiler correctness where you might have:

- Expression indices
- Type indices
- Declaration indices
- Basic block indices

Using raw `usize` everywhere makes it easy to accidentally use an expression index to look up a type. Typed indices prevent this at compile time.

## Core Types

| Type | Description |
|------|-------------|
| `Idx` | Trait for typed indices |
| `IndexVec<I, T>` | Vector indexed by typed index `I` |
| `IndexMap<I, T>` | Alias for `IndexVec<I, Option<T>>` |
| `define_index!` | Macro to define new index types |

## Quick Start

```rust
use bhc_index::{define_index, IndexVec};

// Define typed indices
define_index! {
    /// Index into the expression arena.
    pub struct ExprId;

    /// Index into the type arena.
    pub struct TypeId;
}

// Create typed vectors
let mut exprs: IndexVec<ExprId, Expr> = IndexVec::new();
let mut types: IndexVec<TypeId, Type> = IndexVec::new();

// Push returns the typed index
let expr_id: ExprId = exprs.push(Expr::Lit(42));
let type_id: TypeId = types.push(Type::Int);

// Type-safe indexing
let expr = &exprs[expr_id];  // OK
// let bad = &exprs[type_id];  // Compile error! TypeId != ExprId
```

## Defining Index Types

Use the `define_index!` macro:

```rust
use bhc_index::define_index;

define_index! {
    /// Index into expressions.
    pub struct ExprId;

    /// Index into patterns.
    pub struct PatId;

    /// Index into types.
    pub struct TypeId;

    /// Index into declarations.
    pub struct DeclId;

    /// Index into basic blocks.
    pub struct BlockId;

    /// Index into instructions.
    pub struct InstrId;
}
```

Each generated type has:

```rust
impl ExprId {
    pub const fn new(idx: u32) -> Self;
    pub const fn from_usize(idx: usize) -> Self;
    pub const fn as_u32(self) -> u32;
    pub const fn as_usize(self) -> usize;
}

// Plus traits:
// - Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash
// - Debug, Display
// - From<u32>, From<usize>
// - Serialize, Deserialize
```

## IndexVec

A vector that uses typed indices:

```rust
use bhc_index::IndexVec;

let mut vec: IndexVec<ExprId, Expr> = IndexVec::new();

// Push and get index
let id: ExprId = vec.push(Expr::Lit(1));

// Index access
let expr: &Expr = &vec[id];

// Get with bounds check
let maybe: Option<&Expr> = vec.get(id);

// Mutable access
vec[id] = Expr::Lit(2);

// Iteration
for expr in &vec {
    println!("{:?}", expr);
}

// Enumerated iteration
for (id, expr) in vec.iter_enumerated() {
    println!("{}: {:?}", id, expr);
}

// Get indices
for id in vec.indices() {
    println!("Index: {}", id);
}
```

### Creating IndexVec

```rust
// Empty
let vec: IndexVec<ExprId, Expr> = IndexVec::new();

// With capacity
let vec: IndexVec<ExprId, Expr> = IndexVec::with_capacity(1000);

// From existing Vec
let data = vec![Expr::Lit(1), Expr::Lit(2)];
let vec: IndexVec<ExprId, Expr> = IndexVec::from_vec(data);

// From iterator
let vec: IndexVec<ExprId, i32> = (0..10).collect();
```

### Querying IndexVec

```rust
let vec: IndexVec<ExprId, Expr> = /* ... */;

// Length
let len = vec.len();
let empty = vec.is_empty();

// Next index (for pre-allocating)
let next: ExprId = vec.next_index();

// Raw access (escape hatch)
let raw: &Vec<Expr> = vec.raw();
let raw: Vec<Expr> = vec.into_raw();
```

## IndexMap

A sparse map using `Option<T>`:

```rust
use bhc_index::IndexMap;

let mut map: IndexMap<ExprId, TypeId> = IndexVec::new();

// Grow to accommodate index
while map.len() <= expr_id.as_usize() {
    map.push(None);
}

// Set value
map[expr_id] = Some(type_id);

// Get value
if let Some(ty) = &map[expr_id] {
    println!("Type: {}", ty);
}
```

## Use in Compiler IRs

Typical pattern for IR definitions:

```rust
use bhc_index::{define_index, IndexVec};

define_index! {
    pub struct ExprId;
    pub struct TypeId;
    pub struct LocalId;
}

/// A function body in the IR.
pub struct Body {
    /// All expressions in this body.
    exprs: IndexVec<ExprId, Expr>,
    /// Types for each expression.
    expr_types: IndexVec<ExprId, TypeId>,
    /// Local variable bindings.
    locals: IndexVec<LocalId, Local>,
    /// Entry expression.
    entry: ExprId,
}

impl Body {
    pub fn new_expr(&mut self, expr: Expr, ty: TypeId) -> ExprId {
        let id = self.exprs.push(expr);
        self.expr_types.push(ty);
        id
    }

    pub fn expr(&self, id: ExprId) -> &Expr {
        &self.exprs[id]
    }

    pub fn expr_type(&self, id: ExprId) -> TypeId {
        self.expr_types[id]
    }
}
```

## Parallel Vectors

Keep related data in sync:

```rust
pub struct TypedExprs {
    exprs: IndexVec<ExprId, Expr>,
    types: IndexVec<ExprId, Type>,
    spans: IndexVec<ExprId, Span>,
}

impl TypedExprs {
    pub fn push(&mut self, expr: Expr, ty: Type, span: Span) -> ExprId {
        let id = self.exprs.push(expr);
        let ty_id = self.types.push(ty);
        let span_id = self.spans.push(span);
        debug_assert_eq!(id.as_usize(), ty_id.as_usize());
        debug_assert_eq!(id.as_usize(), span_id.as_usize());
        id
    }
}
```

## Serialization

Index types serialize as integers:

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Module {
    exprs: IndexVec<ExprId, Expr>,
    entry: ExprId,
}

// Serializes as:
// { "exprs": [...], "entry": 5 }
```

## Performance Notes

- Index types are `#[repr(transparent)]` wrappers around `u32`
- Zero runtime overhead compared to raw indices
- `IndexVec` is a thin wrapper around `Vec`
- All operations are inlined

## The Idx Trait

For generic code over index types:

```rust
use bhc_index::Idx;

fn print_indices<I: Idx>(vec: &IndexVec<I, impl std::fmt::Debug>) {
    for (idx, val) in vec.iter().enumerate() {
        let i = I::new(idx);
        println!("{}: {:?}", i.index(), val);
    }
}
```
