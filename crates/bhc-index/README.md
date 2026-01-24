# bhc-index

Type-safe indices and indexed collections for the Basel Haskell Compiler.

## Overview

This crate provides a pattern for creating type-safe indices that prevent mixing up indices from different collections. This is crucial in a compiler where many different ID types exist (expression IDs, type IDs, variable IDs, etc.).

## Key Types

| Type | Description |
|------|-------------|
| `Idx` | Trait for typed indices |
| `IndexVec<I, T>` | A vector indexed by a typed index |
| `IndexMap<I, T>` | An index-based map (alias for `IndexVec<I, Option<T>>`) |

## Usage

### Defining Index Types

```rust
use bhc_index::define_index;

define_index! {
    /// Index into the expression arena.
    pub struct ExprId;

    /// Index into the type arena.
    pub struct TypeId;
}

// These are now distinct types - can't mix them up!
let expr_id: ExprId = ExprId::new(0);
let type_id: TypeId = TypeId::new(0);

// This won't compile:
// let bad: ExprId = type_id;
```

### Using IndexVec

```rust
use bhc_index::{IndexVec, define_index};

define_index! {
    pub struct NodeId;
}

let mut nodes: IndexVec<NodeId, String> = IndexVec::new();

// Push returns the typed index
let id1 = nodes.push("first".to_string());
let id2 = nodes.push("second".to_string());

// Index with the typed ID
assert_eq!(nodes[id1], "first");
assert_eq!(nodes[id2], "second");

// Iterate with indices
for (id, node) in nodes.iter_enumerated() {
    println!("{:?}: {}", id, node);
}
```

## Benefits

1. **Type Safety**: Can't accidentally use an `ExprId` where a `TypeId` is expected
2. **Self-Documenting**: The index type tells you what collection it indexes
3. **Zero Cost**: Compiles down to plain integer operations
4. **Debug Support**: Index types have meaningful debug output (`ExprId(42)`)

## Generated Index Type Features

The `define_index!` macro generates indices with:

- `Clone`, `Copy`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`
- `Debug` and `Display` implementations
- Conversion to/from `u32` and `usize`
- `Serialize` and `Deserialize` implementations
- The `Idx` trait implementation

## Design Notes

- Indices are represented as `u32` internally (supports up to 4 billion items)
- `IndexVec` is a thin wrapper around `Vec` with typed indexing
- The phantom type parameter ensures type safety at zero runtime cost

## Related Crates

- `bhc-ast` - Uses typed indices for AST node IDs
- `bhc-hir` - Uses typed indices for HIR definitions
- `bhc-core` - Uses typed indices for Core IR bindings
