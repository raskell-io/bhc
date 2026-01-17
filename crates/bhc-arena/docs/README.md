# bhc-arena

Arena allocators for efficient compiler data structure allocation.

## Overview

`bhc-arena` provides arena allocators that enable fast allocation of compiler data structures with automatic bulk deallocation. Arenas are essential for compiler performance:

- **Fast allocation**: Bump pointer allocation is O(1)
- **Cache-friendly**: Contiguous memory layout
- **Bulk deallocation**: All allocations freed at once when arena is dropped
- **No fragmentation**: No per-object free, no memory fragmentation

## Arena Types

| Type | Thread-safe | Drops values | Use case |
|------|-------------|--------------|----------|
| `Arena` | No | Yes | Per-pass AST nodes |
| `DroplessArena` | No | No | Copy types, interned data |
| `SyncArena` | Yes | No | Shared across threads |
| `TypedArena<T>` | No | Yes | Homogeneous collections |

## Quick Start

```rust
use bhc_arena::Arena;

let arena = Arena::new();

// Allocate values
let x: &mut i32 = arena.alloc(42);
let s: &str = arena.alloc_str("hello");
let slice: &mut [i32] = arena.alloc_slice(&[1, 2, 3]);

// All allocations freed when arena is dropped
```

## Arena

The primary arena for general use:

```rust
use bhc_arena::Arena;

// Create with default capacity
let arena = Arena::new();

// Create with pre-allocated capacity
let arena = Arena::with_capacity(1024 * 1024);  // 1 MB

// Allocate a single value
let node = arena.alloc(AstNode::Var("x".into()));

// Allocate a string
let name = arena.alloc_str("function_name");

// Allocate a slice (copy)
let items = arena.alloc_slice(&[1, 2, 3, 4, 5]);

// Allocate from an iterator
let doubled = arena.alloc_from_iter((0..10).map(|x| x * 2));

// Check memory usage
println!("Allocated: {} bytes", arena.bytes_allocated());
```

## DroplessArena

More efficient for types that don't need destructors:

```rust
use bhc_arena::DroplessArena;

let arena = DroplessArena::new();

// Good for Copy types
let x: &mut i32 = arena.alloc(42);

// Good for data that doesn't need Drop
let data: &mut [u8] = arena.alloc_from_iter(vec![1, 2, 3, 4]);

// Note: destructors are never called!
```

## SyncArena

Thread-safe arena for parallel compilation:

```rust
use bhc_arena::SyncArena;
use std::sync::Arc;
use std::thread;

let arena = Arc::new(SyncArena::new());

let handles: Vec<_> = (0..4).map(|i| {
    let arena = Arc::clone(&arena);
    thread::spawn(move || {
        // Safe to allocate from multiple threads
        let val = arena.alloc(i * 10);
        *val
    })
}).collect();

for h in handles {
    println!("{}", h.join().unwrap());
}
```

## TypedArena

Homogeneous arena for a single type (re-exported from `typed_arena`):

```rust
use bhc_arena::TypedArena;

struct AstNode {
    kind: NodeKind,
    children: Vec<usize>,
}

let arena: TypedArena<AstNode> = TypedArena::new();

// Allocate nodes
let node1 = arena.alloc(AstNode { kind: NodeKind::Var, children: vec![] });
let node2 = arena.alloc(AstNode { kind: NodeKind::App, children: vec![] });

// More cache-efficient than Arena for homogeneous data
```

## Compiler Pass Pattern

Typical usage in a compiler pass:

```rust
use bhc_arena::Arena;

pub struct TypeCheckPass<'a> {
    arena: &'a Arena,
    // ... other state
}

impl<'a> TypeCheckPass<'a> {
    pub fn new(arena: &'a Arena) -> Self {
        Self { arena }
    }

    pub fn check_expr(&self, expr: &Expr) -> &'a Type {
        match expr {
            Expr::Lit(n) => self.arena.alloc(Type::Int),
            Expr::Var(name) => self.lookup_var(name),
            Expr::App(f, x) => {
                let f_ty = self.check_expr(f);
                let x_ty = self.check_expr(x);
                self.arena.alloc(Type::App(f_ty, x_ty))
            }
        }
    }
}

// Usage:
fn type_check_module(module: &Module) -> TypedModule {
    let arena = Arena::new();
    let pass = TypeCheckPass::new(&arena);
    let result = pass.check_module(module);

    // Arena dropped here, all temporary types freed
    result.into_owned()
}
```

## Memory Tracking

Monitor arena memory usage:

```rust
let arena = Arena::new();

// Allocate some data
for i in 0..1000 {
    arena.alloc(vec![i; 100]);
}

// Check usage
println!("User bytes: {}", arena.bytes_allocated());
println!("Total bytes: {}", arena.allocated_bytes_including_metadata());
```

## Reset (Unsafe)

Reuse an arena without reallocating:

```rust
let mut arena = Arena::new();

for _ in 0..100 {
    // Allocate temporary data
    for i in 0..1000 {
        arena.alloc(i);
    }

    // Process...

    // Reset for next iteration
    // SAFETY: No references to arena data exist
    unsafe { arena.reset(); }
}
```

## Best Practices

1. **One arena per pass**: Create a new arena for each compiler pass
2. **Don't store arena refs in long-lived structures**: Arena data is invalidated on drop
3. **Use DroplessArena for interned data**: More efficient when Drop isn't needed
4. **Pre-size for large allocations**: Use `with_capacity` to avoid reallocations
5. **Track memory in debug builds**: Use `bytes_allocated()` to find memory issues

## Performance Comparison

| Operation | Arena | Standard Allocator |
|-----------|-------|-------------------|
| Allocate | O(1) bump | O(log n) or worse |
| Deallocate | O(1) bulk | O(1) per object |
| Memory overhead | ~0% | 8-16 bytes/object |
| Cache locality | Excellent | Poor |

## Dependencies

- `bumpalo`: Core bump allocator
- `typed_arena`: Type-specialized arenas
- `parking_lot`: Efficient mutex for SyncArena
