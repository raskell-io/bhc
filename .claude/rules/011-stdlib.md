# Standard Library Guidelines

**Rule ID:** BHC-RULE-011
**Applies to:** All BHC standard library code

---

## Principles

1. **Performance First** — Every function designed for fusion, SIMD, and parallelism
2. **Profile-Aware** — Lazy in Default, strict in Numeric, minimal in Edge
3. **Zero-Copy** — Views and slices over copying where possible
4. **Transparent** — Clear performance characteristics, no hidden allocations
5. **Compatible** — Haskell 2010 semantics with BHC extensions

---

## Module Organization

### Namespace Hierarchy

```
BHC/
├── Prelude.hs              # Core types and functions (re-exports)
├── Data/
│   ├── List.hs             # List operations
│   ├── Maybe.hs            # Maybe type and operations
│   ├── Either.hs           # Either type and operations
│   ├── Tuple.hs            # Tuple operations
│   ├── Function.hs         # Function combinators
│   ├── Ord.hs              # Ordering operations
│   ├── Char.hs             # Character operations
│   ├── String.hs           # String operations
│   ├── Map.hs              # Immutable maps
│   ├── Set.hs              # Immutable sets
│   ├── IntMap.hs           # Int-keyed maps
│   ├── IntSet.hs           # Int sets
│   ├── Sequence.hs         # Finger trees
│   ├── Text.hs             # UTF-8 text
│   ├── ByteString.hs       # Byte arrays
│   └── ByteString/
│       ├── Lazy.hs         # Lazy byte strings
│       └── Builder.hs      # Efficient construction
├── Control/
│   ├── Monad.hs            # Monad operations
│   ├── Applicative.hs      # Applicative operations
│   ├── Functor.hs          # Functor operations
│   ├── Category.hs         # Category operations
│   └── Monad/
│       ├── Reader.hs       # Reader monad
│       ├── Writer.hs       # Writer monad
│       ├── State.hs        # State monad
│       ├── Except.hs       # Exception monad
│       └── Trans.hs        # Monad transformers
├── Numeric/
│   ├── Tensor.hs           # Tensor operations
│   ├── Vector.hs           # Numeric vectors
│   ├── Matrix.hs           # Matrix operations
│   ├── SIMD.hs             # SIMD primitives
│   └── BLAS.hs             # BLAS bindings
└── System/
    ├── IO.hs               # I/O operations
    └── Environment.hs      # Environment access
```

### Crate Organization

Each crate MUST follow this structure:

```
stdlib/bhc-<name>/
├── Cargo.toml              # Rust crate manifest
├── src/
│   ├── lib.rs              # Rust primitives
│   └── *.rs                # Rust implementation modules
├── hs/
│   └── BHC/
│       └── *.hs            # Haskell interface
└── docs/
    ├── README.md           # Overview and quick start
    ├── DESIGN.md           # Design decisions
    └── BENCHMARKS.md       # Performance data
```

---

## Export Patterns

### Explicit Exports

All modules MUST use explicit export lists:

```haskell
module BHC.Data.List
  ( -- * Basic operations
    (++), head, tail, last, init, null, length

    -- * Transformations
  , map, reverse, intersperse, intercalate

    -- * Reducing
  , foldl, foldl', foldr, sum, product

    -- * Building
  , iterate, repeat, replicate, cycle
  ) where
```

### Re-export Conventions

- `BHC.Prelude` re-exports commonly used functions from all modules
- Module-specific preludes (e.g., `BHC.Data.Map`) re-export qualified

```haskell
-- BHC.Data.Map exports Map type and operations
module BHC.Data.Map
  ( Map
  , empty, singleton, fromList
  , insert, delete, lookup
  , ...
  ) where

-- For qualified import pattern
import qualified BHC.Data.Map as Map
```

---

## Implementation Patterns

### Rust Primitives

Performance-critical operations SHOULD be implemented in Rust:

```rust
// src/list.rs
#[no_mangle]
pub extern "C" fn bhc_list_length(list: *const List) -> usize {
    // Fast native implementation
    unsafe { (*list).len() }
}
```

### Haskell Interface

The Haskell interface wraps Rust primitives:

```haskell
-- hs/BHC/Data/List.hs

-- | Get the length of a list.
--
-- ==== __Complexity__
--
-- * Time: O(n)
-- * Space: O(1)
--
-- ==== __Fusion__
--
-- This function participates in list fusion.
length :: [a] -> Int
length = primLength
{-# INLINE length #-}

foreign import ccall unsafe "bhc_list_length"
  primLength :: [a] -> Int
```

### Pure Haskell Fallbacks

For non-performance-critical code or when Rust isn't needed:

```haskell
-- Pure Haskell, but still optimized
intersperse :: a -> [a] -> [a]
intersperse _   []     = []
intersperse sep (x:xs) = x : go xs
  where
    go []     = []
    go (y:ys) = sep : y : go ys
{-# NOINLINE [1] intersperse #-}

-- Fusion rule
{-# RULES
"intersperse/build" forall sep (g :: forall b. (a -> b -> b) -> b -> b).
  intersperse sep (build g) = build (intersperseBuilder sep g)
  #-}
```

---

## Profile Annotations

### Default Profile

Standard lazy semantics:

```haskell
-- Lazy by default
map :: (a -> b) -> [a] -> [b]
map _ []     = []
map f (x:xs) = f x : map f xs
```

### Numeric Profile

Strict, unboxed operations:

```haskell
{-# PROFILE Numeric #-}

-- Strict accumulator
foldl' :: (b -> a -> b) -> b -> [a] -> b
foldl' f !acc []     = acc
foldl' f !acc (x:xs) = foldl' f (f acc x) xs
{-# INLINE foldl' #-}
```

### Edge Profile

Minimal footprint:

```haskell
{-# PROFILE Edge #-}

-- No fusion rules, minimal code size
map :: (a -> b) -> [a] -> [b]
map = mapSimple
```

---

## Fusion Contracts

### Guaranteed Fusion Patterns

Per H26-SPEC Section 8, these patterns MUST fuse:

```haskell
-- Pattern 1: map composition
{-# RULES "map/map" forall f g xs.
  map f (map g xs) = map (f . g) xs #-}

-- Pattern 2: zipWith with maps
{-# RULES "zipWith/map/map" forall f g h xs ys.
  zipWith f (map g xs) (map h ys) = zipWith (\x y -> f (g x) (h y)) xs ys #-}

-- Pattern 3: fold of map
{-# RULES "sum/map" forall f xs.
  sum (map f xs) = foldl' (\acc x -> acc + f x) 0 xs #-}

-- Pattern 4: strict fold of map
{-# RULES "foldl'/map" forall op z f xs.
  foldl' op z (map f xs) = foldl' (\acc x -> op acc (f x)) z xs #-}
```

### Verifying Fusion

```bash
# Check fusion occurred
bhc -fkernel-report MyModule.hs

# Expect output like:
# [Fusion] map/map fused at line 42
# [Fusion] sum/map fused at line 43
```

### Fusion Failures

When fusion cannot occur, document clearly:

```haskell
-- | Multiple uses prevent fusion.
--
-- __Warning__: If the result is used multiple times,
-- the list will be materialized. Consider:
--
-- @
-- let xs' = force (map f xs)  -- Explicit materialization
-- in (sum xs', product xs')
-- @
```

---

## Documentation Requirements

### Function Documentation

Every exported function MUST have:

```haskell
-- | Brief description.
--
-- More detailed explanation if needed.
--
-- ==== __Examples__
--
-- >>> map (+1) [1, 2, 3]
-- [2, 3, 4]
--
-- ==== __Complexity__
--
-- * Time: O(n)
-- * Space: O(1) for lazy consumption
--
-- ==== __Fusion__
--
-- This function participates in list fusion as both producer and consumer.
--
-- ==== __Profile Behavior__
--
-- * Default: Lazy evaluation
-- * Numeric: Strict evaluation, SIMD when applicable
-- * Edge: No fusion, minimal code
map :: (a -> b) -> [a] -> [b]
```

### Module Documentation

Every module MUST have a header:

```haskell
-- |
-- Module      : BHC.Data.List
-- Description : Efficient list operations with guaranteed fusion
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Maintainer  : bhc@example.com
-- Stability   : stable
--
-- This module provides list operations optimized for BHC's fusion
-- framework. All standard patterns from H26-SPEC Section 8 are
-- guaranteed to fuse.
--
-- = Quick Start
--
-- @
-- import BHC.Data.List
--
-- -- These operations fuse into a single loop:
-- result = sum . map (*2) . filter even $ [1..1000000]
-- @
--
-- = Performance Notes
--
-- * All operations O(n) unless noted
-- * Fusion eliminates intermediate allocations
-- * Numeric Profile enables SIMD vectorization
```

---

## Testing Requirements

### Test Structure

```
tests/
├── unit/           # Unit tests
├── property/       # QuickCheck properties
├── laws/           # Type class laws
├── fusion/         # Fusion verification
└── bench/          # Criterion benchmarks
```

### Property Tests

Every function MUST have property tests:

```haskell
-- tests/property/ListTests.hs
prop_map_identity :: [Int] -> Property
prop_map_identity xs = map id xs === xs

prop_map_composition :: Fun Int Int -> Fun Int Int -> [Int] -> Property
prop_map_composition (Fun _ f) (Fun _ g) xs =
  map f (map g xs) === map (f . g) xs

prop_foldl_foldr_reverse :: [Int] -> Property
prop_foldl_foldr_reverse xs =
  foldl (flip (:)) [] xs === reverse xs
```

### Fusion Tests

```haskell
-- tests/fusion/ListFusion.hs
test_map_map_fuses :: Assertion
test_map_map_fuses = do
  let ir = compileToIR "map (+1) (map (*2) xs)"
  countKernels ir @?= 1  -- Single fused kernel

test_sum_map_fuses :: Assertion
test_sum_map_fuses = do
  let ir = compileToIR "sum (map (*2) xs)"
  countAllocations ir @?= 0  -- No intermediate allocation
```

### Benchmarks

```haskell
-- tests/bench/ListBench.hs
benchmarks :: [Benchmark]
benchmarks =
  [ bgroup "map"
      [ bench "1K"  $ nf (map (+1)) list1k
      , bench "1M"  $ nf (map (+1)) list1m
      ]
  , bgroup "fusion"
      [ bench "map/map"      $ nf (map (+1) . map (*2)) list1m
      , bench "sum/map"      $ nf (sum . map (*2)) list1m
      , bench "filter/map"   $ nf (filter even . map (*2)) list1m
      ]
  ]
```

---

## Best Practices

### Do

- Use explicit strictness annotations in Numeric Profile
- Document complexity for all operations
- Include fusion rules for composable operations
- Test fusion actually occurs
- Benchmark against GHC baseline

### Don't

- Allocate in hot loops (Numeric Profile)
- Use lazy data structures in Numeric Profile
- Hide performance characteristics
- Skip fusion verification
- Assume correctness without property tests

### Checklist for New Functions

- [ ] Type signature with Haddock
- [ ] Complexity documented
- [ ] Fusion behavior documented
- [ ] Profile behavior documented
- [ ] Examples in documentation
- [ ] Property tests written
- [ ] Fusion tests (if applicable)
- [ ] Benchmarks (if performance-critical)
- [ ] Rust implementation (if performance-critical)
