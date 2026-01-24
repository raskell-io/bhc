# bhc-arena

Arena allocators for efficient compiler data structure allocation.

## Overview

This crate provides arena allocators that enable fast allocation of compiler data structures with automatic bulk deallocation. Arenas are ideal for per-function or per-module compiler passes where many allocations have the same lifetime.

## Key Types

| Type | Description |
|------|-------------|
| `Arena` | Thread-local arena for fast, scoped allocations with byte tracking |
| `DroplessArena` | Arena for types that don't need destructors (more efficient) |
| `SyncArena` | Thread-safe arena that can be shared across threads |
| `Bump` | Re-exported from `bumpalo` for direct bump allocation |
| `TypedArena<T>` | Re-exported from `typed_arena` for type-specific arenas |

## Usage

```rust
use bhc_arena::Arena;

let arena = Arena::new();

// Allocate values - they live until the arena is dropped
let x = arena.alloc(42);
let y = arena.alloc("hello");

// Allocate slices
let slice = arena.alloc_slice(&[1, 2, 3, 4, 5]);

// Allocate from iterators
let from_iter = arena.alloc_from_iter(0..10);

// Track allocation statistics
println!("Bytes allocated: {}", arena.bytes_allocated());
```

## Performance

Arena allocation is O(1) bump-pointer allocation, making it significantly faster than individual heap allocations for compiler workloads:

- **Allocation**: Simple pointer bump (no free-list traversal)
- **Deallocation**: Bulk free when arena is dropped (no per-object overhead)
- **Cache locality**: Sequential allocation improves cache behavior

## When to Use

| Scenario | Recommended Arena |
|----------|-------------------|
| Per-function IR nodes | `Arena` |
| Interned strings | `DroplessArena` |
| Shared across threads | `SyncArena` |
| Single type, many instances | `TypedArena<T>` |

## Design Notes

- Allocations from an arena MUST NOT outlive the arena
- The `DroplessArena` is more efficient for `Copy` types as it skips destructor tracking
- `SyncArena` uses internal locking and is safe for concurrent use

## Related Crates

- `bhc-intern` - Uses arenas for string interning storage
- `bhc-ast` - Uses arenas for AST node allocation
- `bhc-core` - Uses arenas for Core IR allocation
