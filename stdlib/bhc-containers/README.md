# bhc-containers

Container data structures for the Basel Haskell Compiler.

## Overview

This crate provides the Rust-side support for BHC container types. The actual container implementations are in Haskell, with this crate providing build system integration.

## Containers

The following container types are provided (implemented in Haskell):

| Container | Description |
|-----------|-------------|
| `Data.Map` | Balanced binary search tree map |
| `Data.Set` | Balanced binary search tree set |
| `Data.IntMap` | Int-keyed map (Patricia trie) |
| `Data.IntSet` | Int set (Patricia trie) |
| `Data.Sequence` | Finger tree sequence |
| `Data.Graph` | Graph algorithms |

## Usage

```haskell
import qualified Data.Map as Map
import qualified Data.Set as Set

-- Maps
let m = Map.fromList [("a", 1), ("b", 2)]
Map.lookup "a" m  -- Just 1

-- Sets
let s = Set.fromList [1, 2, 3]
Set.member 2 s    -- True
```

## Performance Characteristics

| Operation | Map/Set | IntMap/IntSet | Sequence |
|-----------|---------|---------------|----------|
| Lookup | O(log n) | O(min(n, W)) | O(log n) |
| Insert | O(log n) | O(min(n, W)) | O(log n) |
| Delete | O(log n) | O(min(n, W)) | O(log n) |
| Union | O(m log(n/m + 1)) | O(n + m) | O(log(min(n, m))) |

Where W is the number of bits in an Int (typically 64).

## Design Notes

- Pure functional implementations
- Persistent (previous versions preserved)
- Spine-strict, value-lazy by default
- Strict variants available (`Data.Map.Strict`)

## Related Crates

- `bhc-prelude` - Basic types
- `bhc-base` - Standard library

## Specification References

- H26-SPEC Section 5: Standard Library
- H26-SPEC Section 5.4: Containers
