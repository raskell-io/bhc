# BHC Containers Library

Efficient container types for BHC.

## Overview

The BHC Containers library provides high-performance data structures:

- **Data.Map** - Immutable ordered maps (weight-balanced trees)
- **Data.Set** - Immutable ordered sets
- **Data.IntMap** - Maps with Int keys (Patricia tries)
- **Data.IntSet** - Sets of Int values (bit vectors)
- **Data.Sequence** - Sequences (finger trees)

## Quick Start

```haskell
import qualified BHC.Data.Map as Map
import qualified BHC.Data.Set as Set

main :: IO ()
main = do
  let m = Map.fromList [("a", 1), ("b", 2), ("c", 3)]
  print $ Map.lookup "b" m  -- Just 2

  let s = Set.fromList [1, 2, 3, 4, 5]
  print $ Set.member 3 s    -- True
```

## Performance

| Container | Insert | Lookup | Delete |
|-----------|--------|--------|--------|
| Map | O(log n) | O(log n) | O(log n) |
| Set | O(log n) | O(log n) | O(log n) |
| IntMap | O(min(n, W)) | O(min(n, W)) | O(min(n, W)) |
| IntSet | O(min(n, W)) | O(1) | O(min(n, W)) |
| Sequence | O(log n) | O(log n) | O(log n) |

W = word size (32 or 64 bits)

## See Also

- [BHC.Prelude](../bhc-prelude/docs/README.md) - Core types
- [Data.Map Design](DESIGN.md) - Implementation details
