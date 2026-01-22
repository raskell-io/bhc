# Numeric Standard Library Guidelines

**Rule ID:** BHC-RULE-012
**Applies to:** BHC numeric libraries (bhc-numeric, SIMD, tensors, BLAS)

---

## Principles

1. **Fusion Guaranteed** — All composable operations MUST fuse
2. **SIMD by Default** — Vectorized operations where possible
3. **Arena Allocation** — Temporaries in hot arena, not GC heap
4. **Strict Evaluation** — No lazy thunks in numeric code
5. **Predictable Performance** — Clear complexity, no hidden costs

---

## SIMD Type System

### Vector Types

BHC provides fixed-width SIMD types:

```haskell
-- Float vectors
data Vec2F32   -- 64-bit: 2 x f32
data Vec4F32   -- 128-bit: 4 x f32 (SSE)
data Vec8F32   -- 256-bit: 8 x f32 (AVX)
data Vec16F32  -- 512-bit: 16 x f32 (AVX-512)

-- Double vectors
data Vec2F64   -- 128-bit: 2 x f64 (SSE)
data Vec4F64   -- 256-bit: 4 x f64 (AVX)
data Vec8F64   -- 512-bit: 8 x f64 (AVX-512)

-- Integer vectors (32-bit)
data Vec4I32   -- 128-bit: 4 x i32
data Vec8I32   -- 256-bit: 8 x i32
data Vec16I32  -- 512-bit: 16 x i32

-- Integer vectors (64-bit)
data Vec2I64   -- 128-bit: 2 x i64
data Vec4I64   -- 256-bit: 4 x i64
data Vec8I64   -- 512-bit: 8 x i64
```

### Type Classes for SIMD

```haskell
-- | SIMD vector operations
class SIMDVector v where
  type Scalar v :: *
  type Width v :: Nat

  -- Construction
  broadcast :: Scalar v -> v
  fromList  :: [Scalar v] -> v

  -- Element access
  extract :: v -> Int -> Scalar v
  insert  :: v -> Int -> Scalar v -> v

  -- Arithmetic
  vAdd, vSub, vMul :: v -> v -> v
  vNeg, vAbs      :: v -> v

  -- Comparison
  vMin, vMax :: v -> v -> v
  vEq, vLt   :: v -> v -> Mask v

-- | Floating-point SIMD operations
class SIMDVector v => SIMDFloat v where
  vDiv   :: v -> v -> v
  vSqrt  :: v -> v
  vRsqrt :: v -> v  -- Approximate reciprocal sqrt
  vRcp   :: v -> v  -- Approximate reciprocal
  vFMA   :: v -> v -> v -> v  -- Fused multiply-add: a*b + c

-- | Horizontal reduction operations
class SIMDVector v => SIMDReduce v where
  vSum     :: v -> Scalar v
  vProduct :: v -> Scalar v
  vHMin    :: v -> Scalar v
  vHMax    :: v -> Scalar v
```

### Usage Guidelines

```haskell
-- Good: Explicit SIMD
dotProductSIMD :: UArray Float -> UArray Float -> Float
dotProductSIMD xs ys = go 0 0.0
  where
    !n = length xs
    !chunks = n `div` 8

    go !i !acc
      | i >= chunks = goScalar (i * 8) acc
      | otherwise   =
          let !vx = vLoadAligned (indexPtr xs (i * 8))
              !vy = vLoadAligned (indexPtr ys (i * 8))
              !vp = vMul vx vy
          in go (i + 1) (acc + vSum vp)

    goScalar !i !acc
      | i >= n    = acc
      | otherwise = goScalar (i + 1) (acc + xs ! i * ys ! i)
```

---

## Arena Allocation

### Hot Arena Usage

All numeric temporaries MUST use the hot arena:

```haskell
-- | Matrix multiply with arena allocation
matmul :: Arena s -> Matrix Float -> Matrix Float -> ST s (Matrix Float)
matmul arena a b = do
  -- Result in arena (caller must copy if needed)
  result <- arenaAllocMatrix arena (rows a) (cols b)

  -- Temporary tile buffer in arena
  tile <- arenaAlloc arena (tileSize * tileSize * sizeOf @Float)

  -- Compute
  computeTiled a b result tile
  pure result
```

### Arena Scoping

```haskell
-- Good: Arena lives for kernel scope
withKernelArena :: (forall s. Arena s -> ST s a) -> a
withKernelArena action = runST $ do
  arena <- newArena defaultKernelArenaSize
  result <- action arena
  -- Arena freed here, all temporaries gone
  pure result

-- Bad: Escaping arena allocation
leakArena :: Arena s -> ST s (Ptr Float)  -- WRONG
leakArena arena = arenaAlloc arena 1024    -- Ptr escapes!
```

### Alignment Requirements

```haskell
-- SIMD operations require aligned memory
arenaAllocAligned :: Arena s -> Int -> Int -> ST s (Ptr a)
arenaAllocAligned arena size alignment =
  -- Alignment must be power of 2 and >= 16 for SIMD
  assert (alignment >= 16 && isPowerOf2 alignment) $
  arenaAllocAlignedImpl arena size alignment

-- Convenience for common alignments
arenaAllocSSE  :: Arena s -> Int -> ST s (Ptr a)  -- 16-byte aligned
arenaAllocAVX  :: Arena s -> Int -> ST s (Ptr a)  -- 32-byte aligned
arenaAllocAVX512 :: Arena s -> Int -> ST s (Ptr a)  -- 64-byte aligned
```

---

## Strict Evaluation

### Bang Patterns Required

All numeric code MUST use strict evaluation:

```haskell
{-# LANGUAGE BangPatterns #-}

-- Good: All strict
sumStrict :: UArray Float -> Float
sumStrict arr = go 0 0.0
  where
    !n = length arr
    go !i !acc
      | i >= n    = acc
      | otherwise = go (i + 1) (acc + arr ! i)

-- Bad: Lazy accumulator (builds thunks)
sumBad :: UArray Float -> Float
sumBad arr = go 0 0.0
  where
    n = length arr
    go i acc  -- Missing bangs!
      | i >= n    = acc
      | otherwise = go (i + 1) (acc + arr ! i)
```

### Unboxed Types

Numeric primitives MUST be unboxed:

```haskell
-- Good: Unboxed
data Vec3 = Vec3 {-# UNPACK #-} !Float
                 {-# UNPACK #-} !Float
                 {-# UNPACK #-} !Float

-- Bad: Boxed (extra indirections)
data Vec3Bad = Vec3Bad Float Float Float

-- Good: Unboxed array
newtype UArray a = UArray (ByteArray#)

-- Bad: Boxed array of boxed elements
data BadArray a = BadArray [a]
```

---

## Fusion Verification

### Required Fusion Patterns

Per H26-SPEC Section 8, these MUST fuse in Numeric Profile:

```haskell
-- Pattern 1: Tensor map composition
{-# RULES "tmap/tmap" forall f g t.
  tMap f (tMap g t) = tMap (f . g) t #-}

-- Pattern 2: Tensor zipWith with maps
{-# RULES "tzipWith/tmap/tmap" forall f g h a b.
  tZipWith f (tMap g a) (tMap h b) =
    tZipWith (\x y -> f (g x) (h y)) a b #-}

-- Pattern 3: Reduction of map
{-# RULES "tsum/tmap" forall f t.
  tSum (tMap f t) = tFold (\acc x -> acc + f x) 0 t #-}

-- Pattern 4: Strict fold of map
{-# RULES "tfold/tmap" forall op z f t.
  tFold op z (tMap f t) = tFold (\acc x -> op acc (f x)) z t #-}
```

### Verifying Fusion

```bash
# Compile with kernel report
bhc -fkernel-report -fverify-fusion MyNumeric.hs

# Expected output:
# [Kernel k1] dotProduct: FUSED
#   Fused operations: tZipWith (*), tSum
#   Single traversal, no intermediate allocation
#   SIMD width: 8 x f32 (AVX)

# [Kernel k2] matmul: TILED
#   Tile size: 64x64
#   L1 cache resident
#   SIMD width: 8 x f32 (AVX)
```

### Fusion Failure is a Bug

In Numeric Profile, if a guaranteed pattern fails to fuse:

```haskell
-- This MUST fuse
result = tSum (tMap (*2) tensor)

-- If fusion fails, compiler emits ERROR, not warning
-- Error: Fusion failure for guaranteed pattern at MyModule.hs:42
--        Pattern: tSum/tMap
--        Reason: Non-linear use of intermediate
```

---

## Tensor Operations

### Shape-Indexed Tensors

```haskell
-- Type-safe tensor with compile-time shape
data Tensor (shape :: [Nat]) (dtype :: *) where
  Tensor :: Ptr dtype -> Shape -> Strides -> Tensor shape dtype

-- Shape inference
tMatMul :: Tensor '[m, k] Float -> Tensor '[k, n] Float -> Tensor '[m, n] Float

-- Type error if shapes don't match
-- tMatMul (tensor @'[3, 4]) (tensor @'[5, 6])
-- Error: Couldn't match '4' with '5' in matmul inner dimension
```

### Layout Tracking

```haskell
-- Layout affects performance, track at type level
data Layout = Contiguous | Strided | Tiled TileSize

data Tensor (layout :: Layout) shape dtype where
  ...

-- Contiguous operations are faster
tMatMulFast :: Tensor 'Contiguous '[m, k] Float
            -> Tensor 'Contiguous '[k, n] Float
            -> Tensor 'Contiguous '[m, n] Float

-- Strided tensors may need conversion
ensureContiguous :: Tensor l shape dtype -> Tensor 'Contiguous shape dtype
```

### Broadcasting

```haskell
-- Explicit broadcasting
tBroadcast :: KnownShape newShape
           => Tensor shape dtype
           -> Tensor newShape dtype

-- Example: [3] -> [4, 3]
vec :: Tensor '[3] Float
mat :: Tensor '[4, 3] Float
mat = tBroadcast @'[4, 3] vec  -- Broadcasts along first axis
```

---

## BLAS Integration

### Provider Abstraction

```haskell
-- Abstract over BLAS implementations
class BLASProvider (backend :: *) where
  -- Level 1: Vector-Vector
  axpy :: backend -> Float -> Vector Float -> Vector Float -> IO ()
  dot  :: backend -> Vector Float -> Vector Float -> IO Float
  nrm2 :: backend -> Vector Float -> IO Float

  -- Level 2: Matrix-Vector
  gemv :: backend -> Matrix Float -> Vector Float -> IO (Vector Float)

  -- Level 3: Matrix-Matrix
  gemm :: backend -> Matrix Float -> Matrix Float -> IO (Matrix Float)

-- Implementations
data OpenBLAS = OpenBLAS
data MKL = MKL
data Accelerate = Accelerate  -- Apple vDSP
data PureBHC = PureBHC        -- Fallback

instance BLASProvider OpenBLAS where
  gemm OpenBLAS a b = ...  -- FFI to openblas_dgemm

instance BLASProvider PureBHC where
  gemm PureBHC a b = ...   -- Pure Rust implementation
```

### Automatic Provider Selection

```haskell
-- Compiler selects best available provider
defaultBLAS :: IO SomeBLASProvider
defaultBLAS = do
  mkl <- tryLoadMKL
  case mkl of
    Just p  -> pure (SomeBLAS p)
    Nothing -> do
      openblas <- tryLoadOpenBLAS
      case openblas of
        Just p  -> pure (SomeBLAS p)
        Nothing -> pure (SomeBLAS PureBHC)
```

---

## Benchmark Requirements

### Required Benchmarks

Every numeric function MUST have benchmarks:

```haskell
benchmarks :: [Benchmark]
benchmarks =
  [ bgroup "dot product"
      [ bench "1K"   $ nf (uncurry dot) (v1k, v1k)
      , bench "10K"  $ nf (uncurry dot) (v10k, v10k)
      , bench "100K" $ nf (uncurry dot) (v100k, v100k)
      , bench "1M"   $ nf (uncurry dot) (v1m, v1m)
      ]
  , bgroup "matmul"
      [ bench "64x64"     $ nf (uncurry matmul) (m64, m64)
      , bench "256x256"   $ nf (uncurry matmul) (m256, m256)
      , bench "1024x1024" $ nf (uncurry matmul) (m1024, m1024)
      ]
  , bgroup "fusion"
      [ bench "map/map/sum 1M" $
          nf (tSum . tMap (+1) . tMap (*2)) tensor1m
      ]
  ]
```

### Comparison Targets

Benchmarks SHOULD compare against:

| Target | Description |
|--------|-------------|
| GHC + vector | GHC with vector package |
| NumPy | Python NumPy |
| Julia | Julia native arrays |
| Rust ndarray | Rust ndarray crate |
| Reference BLAS | OpenBLAS or MKL |

### Performance Targets

| Operation | Size | Target Time | Notes |
|-----------|------|-------------|-------|
| dot product | 1M | <500μs | SIMD vectorized |
| matmul | 256x256 | <2ms | Tiled, cache-aware |
| matmul | 1024x1024 | <200ms | BLAS backend |
| map fusion | 1M | <1ms | Zero allocation |

---

## Documentation Requirements

### Numeric Function Documentation

```haskell
-- | Compute the dot product of two vectors.
--
-- \[ \text{dot}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} x_i \cdot y_i \]
--
-- ==== __Examples__
--
-- >>> dot (fromList [1, 2, 3]) (fromList [4, 5, 6])
-- 32.0
--
-- ==== __Complexity__
--
-- * Time: O(n)
-- * Space: O(1) (no intermediate allocation)
--
-- ==== __SIMD__
--
-- Automatically vectorized using AVX-256 (8 floats at a time).
-- Falls back to SSE (4 floats) or scalar on older hardware.
--
-- ==== __Parallelization__
--
-- Parallelized for n > 10000 elements using work-stealing.
-- Use 'dotSeq' for guaranteed sequential execution.
--
-- ==== __Precision__
--
-- Uses compensated summation to minimize floating-point error.
-- For maximum precision, use 'dotKahan'.
dot :: Vector Float -> Vector Float -> Float
```

### SIMD Documentation

```haskell
-- | Fused multiply-add: computes @a * b + c@ in a single operation.
--
-- ==== __Hardware Support__
--
-- * AVX-512: Single `vfmadd` instruction
-- * AVX/FMA: Single `vfmadd` instruction
-- * SSE: Two instructions (mul + add)
--
-- ==== __Precision__
--
-- FMA maintains full precision for intermediate result (no rounding
-- between multiply and add). This can give different results than
-- @a * b + c@ computed separately.
--
-- ==== __Performance__
--
-- ~2x throughput compared to separate mul + add on FMA-capable hardware.
vFMA :: Vec8F32 -> Vec8F32 -> Vec8F32 -> Vec8F32
```

---

## Error Handling

### Runtime Checks (Debug Mode)

```haskell
-- Bounds checking in debug mode
(!) :: Tensor shape dtype -> Index shape -> dtype
tensor ! idx
#ifdef DEBUG
  | not (inBounds tensor idx) =
      error $ "Index out of bounds: " ++ show idx
            ++ " for shape " ++ show (shape tensor)
#endif
  | otherwise = unsafeIndex tensor idx
```

### Dimension Mismatch

```haskell
-- Runtime check for dynamic shapes
tMatMul :: Matrix Float -> Matrix Float -> Either TensorError (Matrix Float)
tMatMul a b
  | cols a /= rows b = Left $ DimensionMismatch
      { expected = cols a
      , actual = rows b
      , operation = "matmul"
      }
  | otherwise = Right $ unsafeMatMul a b
```

---

## Best Practices

### Do

- Use SIMD types for hot loops
- Allocate temporaries in hot arena
- Verify fusion with `-fkernel-report`
- Benchmark against reference implementations
- Document SIMD/parallelization behavior
- Use compensated summation for reductions

### Don't

- Allocate on GC heap in hot paths
- Use lazy evaluation in numeric code
- Ignore alignment requirements
- Skip fusion verification
- Assume fusion without testing
- Use boxed types for numeric data

### Checklist for Numeric Functions

- [ ] Strict evaluation (all bang patterns)
- [ ] Unboxed types
- [ ] Arena allocation for temporaries
- [ ] SIMD implementation
- [ ] Fusion rules
- [ ] Fusion verification test
- [ ] Complexity documented
- [ ] SIMD behavior documented
- [ ] Benchmarks against baseline
- [ ] Comparison with NumPy/Julia
