# BHC Numeric Library

High-performance numeric computing for BHC.

## Overview

The BHC Numeric library provides:

- **SIMD** - Vector types with hardware acceleration
- **Tensor** - N-dimensional arrays with shape tracking
- **Vector** - 1-D numeric vectors
- **Matrix** - 2-D matrices with linear algebra
- **BLAS** - BLAS provider abstraction

## Quick Start

```haskell
import BHC.Numeric.Vector
import BHC.Numeric.Matrix
import BHC.Numeric.SIMD

-- Dot product (SIMD accelerated)
main :: IO ()
main = do
  let x = fromList [1, 2, 3, 4, 5] :: Vector Float
  let y = fromList [5, 4, 3, 2, 1] :: Vector Float
  print $ dot x y  -- 35.0

-- Matrix multiplication
let a = fromLists [[1, 2], [3, 4]] :: Matrix Float
let b = fromLists [[5, 6], [7, 8]] :: Matrix Float
print $ matmul a b  -- [[19, 22], [43, 50]]
```

## Features

### SIMD Vector Types

```haskell
-- 128-bit vectors (SSE)
Vec4F32  -- 4 x Float
Vec2F64  -- 2 x Double

-- 256-bit vectors (AVX)
Vec8F32  -- 8 x Float
Vec4F64  -- 4 x Double

-- Operations
vAdd, vSub, vMul, vDiv :: Vec a -> Vec a -> Vec a
vFMA :: Vec a -> Vec a -> Vec a -> Vec a  -- Fused multiply-add
vSum :: Vec a -> Scalar a  -- Horizontal sum
```

### Guaranteed Fusion

```haskell
-- These operations fuse into a single kernel:
result = tSum (tMap (*2) (tFilter even tensor))

-- Verify with:
-- bhc -fkernel-report MyModule.hs
```

### BLAS Integration

```haskell
-- Automatic BLAS selection
import BHC.Numeric.BLAS

-- Uses OpenBLAS/MKL/Accelerate when available
gemm :: Matrix Float -> Matrix Float -> Matrix Float
```

## Performance

| Operation | Size | Time | Notes |
|-----------|------|------|-------|
| dot | 1M | 0.5ms | AVX |
| matmul | 256x256 | 2ms | BLAS |
| matmul | 1024x1024 | 180ms | BLAS |
| map fusion | 1M | 0.8ms | Zero alloc |

## See Also

- [SIMD Guide](DESIGN.md) - SIMD programming guide
- [Benchmarks](BENCHMARKS.md) - Performance comparisons
- [012-numeric-stdlib.md](../../.claude/rules/012-numeric-stdlib.md) - Guidelines
