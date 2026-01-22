# BHC Prelude

Core types and functions for the BHC standard library.

## Overview

The BHC Prelude provides the fundamental types and functions that are implicitly imported into every BHC module. It includes:

- **Core Types**: `Bool`, `Maybe`, `Either`, `Ordering`, tuples
- **Type Classes**: `Eq`, `Ord`, `Show`, `Num`, `Functor`, `Monad`, etc.
- **List Operations**: `map`, `filter`, `fold`, with guaranteed fusion
- **Numeric Operations**: Arithmetic, trigonometry, with SIMD acceleration
- **I/O Operations**: Basic console and file I/O

## Features

### Guaranteed Fusion

All composable list operations participate in fusion:

```haskell
-- These operations fuse into a single loop:
result = sum . map (*2) . filter even $ [1..1000000]
```

### Profile-Aware Behavior

Operations adapt to the active profile:

| Profile | Behavior |
|---------|----------|
| Default | Lazy evaluation, standard GC |
| Numeric | Strict evaluation, SIMD acceleration |
| Edge | Minimal footprint, no fusion overhead |

### SIMD Acceleration

Numeric operations automatically use SIMD instructions when available:

```haskell
-- Automatically vectorized on supported hardware
sumFloats :: [Float] -> Float
sumFloats = sum  -- Uses AVX/SSE when available
```

## Quick Start

The Prelude is imported automatically:

```haskell
module MyModule where

-- All Prelude functions available without import
main :: IO ()
main = do
  let numbers = [1..100]
  print $ sum numbers          -- 5050
  print $ product [1..10]      -- 3628800
  print $ filter even numbers  -- [2,4,6,...,100]
```

To disable implicit import:

```haskell
{-# LANGUAGE NoImplicitPrelude #-}
module MyModule where

import BHC.Prelude (IO, print, ($))  -- Import only what you need
```

## Performance Characteristics

### Complexity

| Function | Time | Space | Notes |
|----------|------|-------|-------|
| `length` | O(n) | O(1) | Traverses spine |
| `map` | O(n) | O(n) lazy | Fuses with consumers |
| `filter` | O(n) | O(n) lazy | Fuses with consumers |
| `foldl'` | O(n) | O(1) | Strict accumulator |
| `sum` | O(n) | O(1) | Strict, SIMD accelerated |
| `(++)` | O(n) | O(n) | n = length of first list |

### Fusion Guarantees

Per H26-SPEC Section 8, these patterns always fuse:

1. `map f (map g xs)` → single traversal
2. `sum (map f xs)` → no intermediate list
3. `filter p (map f xs)` → single traversal
4. `foldl' op z (map f xs)` → no intermediate list

## Differences from GHC Prelude

| Aspect | GHC | BHC |
|--------|-----|-----|
| Fusion | Stream fusion (sometimes fails) | Guaranteed for standard patterns |
| Strictness | Lazy by default | Profile-dependent |
| SIMD | Manual with vector package | Automatic in Numeric Profile |
| Error messages | Stack trace optional | Always includes location |

## Module Structure

```
BHC.Prelude
├── Core Types
│   ├── Bool, Maybe, Either, Ordering
│   └── Tuples (2-8)
├── Type Classes
│   ├── Eq, Ord, Enum, Bounded
│   ├── Num, Fractional, Floating
│   ├── Functor, Applicative, Monad
│   └── Foldable, Traversable
├── List Operations
│   ├── Basic: map, filter, fold
│   ├── Building: iterate, repeat, cycle
│   └── Searching: elem, lookup, find
└── I/O Operations
    ├── Console: putStr, getLine
    └── Files: readFile, writeFile
```

## See Also

- [DESIGN.md](DESIGN.md) - Design decisions and rationale
- [BENCHMARKS.md](BENCHMARKS.md) - Performance benchmarks
- [BHC.Data.List](../bhc-base/docs/List.md) - Extended list operations
- [BHC.Numeric](../bhc-numeric/docs/README.md) - Numeric computing
