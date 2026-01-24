# bhc-data-structures

Common data structures for the Basel Haskell Compiler.

## Overview

This crate provides type aliases and wrappers around commonly used data structures, ensuring consistent hashing performance across the compiler. It standardizes on `FxHash` for fast, non-cryptographic hashing.

## Key Types

### Hash Collections

| Type | Description |
|------|-------------|
| `FxHashMap<K, V>` | Hash map using FxHasher (fast, deterministic) |
| `FxHashSet<T>` | Hash set using FxHasher |
| `FxIndexMap<K, V>` | Insertion-ordered hash map |
| `FxIndexSet<T>` | Insertion-ordered hash set |

### Small Vectors

| Type | Description |
|------|-------------|
| `TinyVec<T>` | SmallVec optimized for 0-4 elements inline |
| `SmallVec8<T>` | SmallVec optimized for 0-8 elements inline |

### Utilities

| Type | Description |
|------|-------------|
| `WorkQueue<T>` | Work queue for graph traversals (dedup built-in) |
| `FrozenMap<K, V>` | Immutable map for read-only access |
| `UnionFind` | Disjoint set data structure |

## Usage

```rust
use bhc_data_structures::{FxHashMap, FxHashSet, FxHashMapExt, WorkQueue};

// Create collections using the extension traits
let mut map: FxHashMap<String, i32> = FxHashMap::new();
map.insert("key".to_string(), 42);

let mut set: FxHashSet<i32> = FxHashSet::with_capacity(100);
set.insert(1);

// Work queue automatically deduplicates
let mut wq: WorkQueue<i32> = WorkQueue::new();
assert!(wq.push(1));   // true: first time
assert!(!wq.push(1));  // false: already seen
assert!(wq.push(2));   // true: new item

while let Some(item) = wq.pop() {
    // Process item, potentially pushing more
    if item < 10 {
        wq.push(item + 1);
    }
}
```

### Union-Find

```rust
use bhc_data_structures::UnionFind;

let mut uf = UnionFind::new(5);

uf.union(0, 1);
uf.union(2, 3);
uf.union(1, 3);

assert!(uf.same_set(0, 3));  // All connected
assert!(!uf.same_set(0, 4)); // 4 is separate
```

## Why FxHash?

FxHash is optimized for small keys (integers, short strings) which are common in compilers:

- **Fast**: Simple multiply-xor algorithm
- **Deterministic**: Same output across runs (unlike RandomState)
- **Good enough**: Not cryptographic, but suitable for compiler workloads

## Re-exports

This crate also re-exports commonly used types:

- `IndexMap`, `IndexSet` from `indexmap`
- `Mutex`, `RwLock` from `parking_lot`
- `SmallVec` from `smallvec`

## Design Notes

- Prefer `FxHashMap`/`FxHashSet` over `std::collections::HashMap`/`HashSet`
- Use `FxIndexMap`/`FxIndexSet` when iteration order matters
- Use `TinyVec` for fields that are usually 0-4 elements
- Use `WorkQueue` for BFS/worklist algorithms

## Related Crates

- `bhc-typeck` - Uses UnionFind for type unification
- `bhc-core` - Uses FxHashMap for variable environments
- `bhc-query` - Uses WorkQueue for incremental computation
