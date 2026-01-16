# Performance Guidelines

**Rule ID:** BHC-RULE-006
**Applies to:** Performance-critical code, especially Numeric Profile

---

## Principles

1. **Measure first** — No optimization without profiling
2. **Predictability over peak** — Consistent performance beats occasional fast
3. **Transparency** — Users should understand performance characteristics
4. **Zero-cost abstractions** — Abstractions must compile away in Numeric Profile

---

## Numeric Profile Requirements

### No Hidden Allocations

In Numeric Profile, the following MUST NOT allocate on the general heap:

- Tight numeric loops
- Tensor kernel bodies
- SIMD operations

```haskell
-- Good: No allocation in hot path
dotProduct :: UArray Float -> UArray Float -> Float
dotProduct !xs !ys = go 0 0.0
  where
    !n = length xs
    go !i !acc
      | i >= n    = acc
      | otherwise = go (i + 1) (acc + xs ! i * ys ! i)

-- Bad: Allocates thunks
dotProductBad :: UArray Float -> UArray Float -> Float
dotProductBad xs ys = sum (zipWith (*) (toList xs) (toList ys))
```

### Strict by Default

All numeric code MUST use strict evaluation:

```haskell
{-# LANGUAGE BangPatterns #-}

-- Good: Explicit strictness
accumulate :: (a -> b -> a) -> a -> [b] -> a
accumulate _ !acc [] = acc
accumulate f !acc (x:xs) = accumulate f (f acc x) xs

-- Bad: Lazy accumulator builds thunks
accumulateBad :: (a -> b -> a) -> a -> [b] -> a
accumulateBad _ acc [] = acc
accumulateBad f acc (x:xs) = accumulateBad f (f acc x) xs
```

### Unboxed Types

Numeric primitives MUST be unboxed:

```haskell
-- Good: Unboxed
data Point = Point {-# UNPACK #-} !Float {-# UNPACK #-} !Float

-- Bad: Boxed fields
data PointBad = PointBad Float Float
```

---

## Fusion Guarantees

### Patterns That MUST Fuse

Per H26-SPEC Section 8, these patterns MUST fuse:

```haskell
-- Pattern 1: map composition
map f (map g xs)  -- -> map (f . g) xs

-- Pattern 2: zipWith with maps
zipWith f (map g a) (map h b)  -- -> single traversal

-- Pattern 3: fold of map
sum (map f xs)  -- -> single traversal

-- Pattern 4: strict fold of map
foldl' op z (map f xs)  -- -> single traversal
```

### Verifying Fusion

Use `-fkernel-report` to verify fusion:

```bash
bhc -fkernel-report -c MyModule.hs
# Output:
# [Kernel 1] dotProduct: FUSED
#   - zipWith (*) fused with sum
#   - Single traversal, no intermediate array
```

### When Fusion Fails

If fusion cannot occur, the compiler MUST either:
1. Emit a warning (with `-Wtensor-lowering`)
2. Fall back to correct scalar implementation

```haskell
-- May not fuse: non-linear use
let xs' = map f xs
in (sum xs', product xs')  -- xs' used twice, may materialize
```

---

## Memory Management

### Hot Arena Usage

Ephemeral allocations SHOULD use the Hot Arena:

```haskell
-- Good: Arena allocation for temporary
withArena $ \arena -> do
  tmp <- allocateArray arena n
  computeInPlace tmp
  copyToResult tmp result
  -- tmp automatically freed when arena scope ends
```

### Pinned vs Unpinned

- Use pinned memory for FFI interop
- Use unpinned for pure Haskell computation

```haskell
-- FFI boundary: use pinned
withPinnedTensor tensor $ \ptr -> do
  c_blas_dgemm ptr ...

-- Pure computation: unpinned is fine
let result = map (*2) tensor
```

### Avoiding GC Pressure

```haskell
-- Bad: Creates garbage
processItems :: [Item] -> [Result]
processItems = map process

-- Good: In-place when possible
processItemsInPlace :: MVector s Item -> MVector s Result -> ST s ()
processItemsInPlace src dst =
  forM_ [0..n-1] $ \i -> do
    item <- MV.read src i
    MV.write dst i (process item)
```

---

## SIMD Guidelines

### Use Vector Types

```haskell
-- Good: Explicit SIMD
sumVec :: Vec8F32 -> Vec8F32 -> Vec8F32
sumVec = vAdd

-- Will auto-vectorize in Numeric Profile
sumArray :: UArray Float -> Float
sumArray xs = sum xs  -- Compiler vectorizes this
```

### Alignment Requirements

SIMD operations require aligned memory:

```haskell
-- Ensure alignment
allocateAligned :: Int -> Int -> IO (Ptr a)
allocateAligned size alignment = ...

-- Or use aligned array types
newtype AlignedArray a = AlignedArray (UArray a)
  -- Guaranteed 64-byte aligned
```

### Horizontal Operations

Horizontal operations (within a vector) are expensive:

```haskell
-- Expensive: horizontal sum
horizontalSum :: Vec8F32 -> Float

-- Prefer: vertical operations across vectors
verticalSum :: Vec8F32 -> Vec8F32 -> Vec8F32
```

---

## Parallelism

### When to Parallelize

- Data size > 10K elements (rough threshold)
- Operations are independent
- No shared mutable state in hot path

### Parallel Constructs

```haskell
-- Parallel map
parMap :: (a -> b) -> Tensor a -> Tensor b

-- Parallel reduction
parReduce :: Monoid m => (a -> m) -> Tensor a -> m

-- Parallel for loop
parFor :: Range -> (Int -> ()) -> ()
```

### Chunk Size

```haskell
-- Good: Let scheduler choose
parMap f tensor

-- Good: Explicit chunking when you know better
parMapChunked 1024 f tensor

-- Bad: Too fine-grained
parFor (0, 1000000) $ \i -> ...  -- Overhead dominates
```

---

## Benchmarking

### Required Benchmarks

All performance-critical code MUST have benchmarks:

```haskell
-- benchmarks/NumericBench.hs
benchmarks :: [Benchmark]
benchmarks =
  [ bgroup "dot product"
      [ bench "1K"   $ nf (uncurry dot) (v1k, v1k)
      , bench "10K"  $ nf (uncurry dot) (v10k, v10k)
      , bench "100K" $ nf (uncurry dot) (v100k, v100k)
      , bench "1M"   $ nf (uncurry dot) (v1m, v1m)
      ]
  , bgroup "matmul"
      [ bench "64x64"   $ nf (uncurry matmul) (m64, m64)
      , bench "256x256" $ nf (uncurry matmul) (m256, m256)
      , bench "1024x1024" $ nf (uncurry matmul) (m1024, m1024)
      ]
  ]
```

### Benchmark Hygiene

- Warm up before measuring
- Run multiple iterations
- Report variance
- Compare against baseline

```bash
# Run benchmarks
cabal bench --benchmark-options '+RTS -T -RTS'

# Compare against baseline
cabal bench --benchmark-options '--baseline baseline.csv'
```

---

## Profiling

### CPU Profiling

```bash
# Build with profiling
cabal build --enable-profiling

# Run with profiling
./bhc +RTS -p -RTS input.hs

# Analyze
ghc-prof-flamegraph bhc.prof
```

### Heap Profiling

```bash
# Heap by type
./bhc +RTS -hT -RTS input.hs

# Heap by cost center
./bhc +RTS -hc -RTS input.hs

# View results
hp2ps -c bhc.hp
```

### Allocation Tracking

```haskell
-- In Numeric Profile, track allocations
{-# OPTIONS_GHC -ddump-simpl -dsuppress-all #-}

-- Check for unexpected allocations in Core output
```

---

## Performance Contracts

### Documenting Complexity

All public APIs MUST document complexity:

```haskell
-- | Matrix multiplication.
--
-- ==== __Complexity__
--
-- * Time: O(n * m * k) for (n x m) * (m x k) matrices
-- * Space: O(n * k) for result matrix
-- * Allocation: Single output allocation, no intermediates
matmul :: Tensor Float -> Tensor Float -> Tensor Float
```

### Performance Tests

Performance requirements SHOULD be tested:

```haskell
testPerformance :: TestTree
testPerformance = testGroup "Performance"
  [ testCase "dot 1M under 1ms" $ do
      let v = replicate 1000000 1.0
      t <- measureTime $ dot v v
      assertBool "Too slow" (t < 0.001)

  , testCase "matmul allocates once" $ do
      allocsBefore <- getAllocationCount
      _ <- evaluate $ matmul m256 m256
      allocsAfter <- getAllocationCount
      let allocs = allocsAfter - allocsBefore
      assertEqual "Extra allocations" 1 allocs
  ]
```

---

## Common Performance Pitfalls

### Space Leaks

```haskell
-- Bad: Space leak from lazy accumulator
mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)  -- Holds entire list

-- Good: Single pass
mean :: [Double] -> Double
mean xs = total / count
  where (total, count) = foldl' (\(!t, !c) x -> (t + x, c + 1)) (0, 0) xs
```

### Unnecessary Laziness

```haskell
-- Bad: Thunk buildup
processAll :: [Item] -> [Result]
processAll = map process  -- Lazy, builds thunks

-- Good: Force evaluation
processAll :: [Item] -> [Result]
processAll items = map process items `using` parList rdeepseq
```

### Boxing/Unboxing

```haskell
-- Bad: Boxing at polymorphic use site
genericSum :: Num a => [a] -> a  -- May box

-- Good: Specialized version
floatSum :: [Float] -> Float  -- Unboxed
{-# SPECIALIZE floatSum :: [Float] -> Float #-}
```
