# bhc-data-structures

Common data structures for the Basel Haskell Compiler.

## Overview

`bhc-data-structures` re-exports and wraps commonly used data structures, ensuring consistent hashing and performance across the compiler. Features:

- **Fast hashing**: FxHasher-based maps and sets
- **Ordered collections**: Insertion-ordered maps and sets
- **Small vectors**: Stack-optimized small collections
- **Graph algorithms**: WorkQueue, UnionFind

## Core Types

| Type | Description |
|------|-------------|
| `FxHashMap` | Fast hash map using FxHasher |
| `FxHashSet` | Fast hash set using FxHasher |
| `FxIndexMap` | Insertion-ordered hash map |
| `FxIndexSet` | Insertion-ordered hash set |
| `TinyVec` | SmallVec with 4-element inline storage |
| `SmallVec8` | SmallVec with 8-element inline storage |
| `WorkQueue` | Deduplicating work queue |
| `FrozenMap` | Immutable hash map |
| `UnionFind` | Disjoint set data structure |

## Hash Maps and Sets

Using FxHasher for faster hashing (non-cryptographic):

```rust
use bhc_data_structures::{FxHashMap, FxHashSet, FxHashMapExt, FxHashSetExt};

// Create empty map/set
let mut map: FxHashMap<String, i32> = FxHashMap::new();
let mut set: FxHashSet<i32> = FxHashSet::new();

// With capacity
let map: FxHashMap<String, i32> = FxHashMap::with_capacity(100);

// Standard HashMap/HashSet operations
map.insert("key".into(), 42);
set.insert(123);
```

## Ordered Collections

Maintain insertion order:

```rust
use bhc_data_structures::{FxIndexMap, FxIndexSet};

let mut map: FxIndexMap<String, i32> = FxIndexMap::default();
map.insert("first".into(), 1);
map.insert("second".into(), 2);
map.insert("third".into(), 3);

// Iteration preserves insertion order
for (key, value) in &map {
    println!("{}: {}", key, value);
}
// Output: first: 1, second: 2, third: 3
```

## Small Vectors

Stack-optimized for common small sizes:

```rust
use bhc_data_structures::{TinyVec, SmallVec8, SmallVec};

// TinyVec: 4 elements inline (no heap allocation for small cases)
let mut params: TinyVec<String> = TinyVec::new();
params.push("a".into());
params.push("b".into());

// SmallVec8: 8 elements inline
let mut items: SmallVec8<i32> = SmallVec8::new();
for i in 0..8 {
    items.push(i); // Still on stack
}
items.push(8); // Now allocates on heap

// Arbitrary inline size
let custom: SmallVec<[i32; 16]> = SmallVec::new();
```

## WorkQueue

Deduplicating work queue for graph traversals:

```rust
use bhc_data_structures::WorkQueue;

let mut queue: WorkQueue<i32> = WorkQueue::new();

// Push items (returns true if new, false if already seen)
assert!(queue.push(1));   // true - first time
assert!(queue.push(2));   // true - first time
assert!(!queue.push(1));  // false - already seen

// Pop in FIFO order
assert_eq!(queue.pop(), Some(1));
assert_eq!(queue.pop(), Some(2));
assert_eq!(queue.pop(), None);

// Useful for BFS traversals
let mut visited: WorkQueue<NodeId> = WorkQueue::new();
visited.push(start_node);
while let Some(node) = visited.pop() {
    for neighbor in graph.neighbors(node) {
        visited.push(neighbor); // Automatically deduplicates
    }
}
```

## FrozenMap

Immutable map after construction:

```rust
use bhc_data_structures::{FrozenMap, FxHashMap};

// Build with FxHashMap
let mut builder = FxHashMap::new();
builder.insert("a", 1);
builder.insert("b", 2);

// Freeze
let frozen = FrozenMap::new(builder);

// Read-only access
assert_eq!(frozen.get(&"a"), Some(&1));
assert!(frozen.contains_key(&"b"));
assert_eq!(frozen.len(), 2);

// From iterator
let frozen: FrozenMap<&str, i32> = [("x", 10), ("y", 20)].into_iter().collect();
```

## UnionFind

Disjoint set data structure with path compression:

```rust
use bhc_data_structures::UnionFind;

// Create with n elements (0..n)
let mut uf = UnionFind::new(5);

// Union operations
uf.union(0, 1);  // Connect 0 and 1
uf.union(2, 3);  // Connect 2 and 3
uf.union(1, 3);  // Connect the two groups

// Query
assert!(uf.same_set(0, 3));   // true - transitively connected
assert!(!uf.same_set(0, 4)); // false - 4 is separate

// Find representative
let rep = uf.find(0); // Returns representative of set containing 0

// Useful for:
// - Type unification
// - Equivalence classes
// - Graph connectivity
```

## Re-exports

Commonly used external types:

```rust
use bhc_data_structures::{
    // indexmap types
    IndexMap, IndexSet,
    // parking_lot synchronization
    Mutex, RwLock,
    // smallvec
    SmallVec,
};
```

## Performance Notes

- **FxHasher**: Faster than SipHash for small keys (common in compilers)
- **SmallVec**: Avoid heap allocation for typical small collections
- **IndexMap**: O(1) lookup with O(n) iteration in insertion order
- **UnionFind**: Nearly O(1) amortized operations with path compression

## See Also

- `rustc_hash`: FxHasher implementation
- `indexmap`: Ordered collections
- `smallvec`: Stack-allocated vectors
- `parking_lot`: Fast synchronization primitives
