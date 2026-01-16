# Documentation Standards

**Rule ID:** BHC-RULE-004
**Applies to:** All public APIs and complex internal code

---

## Principles

1. **Document the why, not the what** — Code shows what, docs explain why
2. **Keep docs close to code** — Haddock in source, not separate files
3. **Examples over prose** — Show, don't just tell
4. **Update when code changes** — Outdated docs are worse than none

---

## Required Documentation

### MUST Document

- All exported functions
- All exported types and their constructors
- All exported type classes
- Module headers
- Non-obvious algorithms
- Performance characteristics
- Unsafe operations

### SHOULD Document

- Complex internal functions
- Invariants and preconditions
- Design decisions

### MAY Omit Documentation For

- Trivial helper functions
- Self-explanatory one-liners
- Standard typeclass instances

---

## Haddock Format

### Module Header

Every module MUST have a header:

```haskell
-- |
-- Module      : BHC.Tensor.Fusion
-- Description : Guaranteed fusion passes for Tensor IR
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Maintainer  : bhc@example.com
-- Stability   : experimental
--
-- This module implements the guaranteed fusion transformations
-- specified in H26-SPEC Section 8. All patterns listed in Section 8.1
-- MUST fuse; fusion failure is a compiler bug.
--
-- = Overview
--
-- The fusion pass operates on 'TensorIR' and produces fused kernels
-- that execute without intermediate allocation.
--
-- = Usage
--
-- @
-- let fusedIR = runFusion defaultConfig tensorIR
-- @
--
-- = See Also
--
-- * 'BHC.Tensor.IR' for Tensor IR types
-- * 'BHC.Tensor.Lower' for lowering from Core
module BHC.Tensor.Fusion
  ( -- * Fusion Configuration
    FusionConfig (..)
  , defaultConfig

    -- * Running Fusion
  , runFusion
  , FusionResult (..)

    -- * Diagnostics
  , FusionReport
  , getFusionReport
  ) where
```

### Function Documentation

```haskell
-- | Fuse consecutive map operations into a single traversal.
--
-- Given @map f (map g xs)@, produces @map (f . g) xs@ which
-- traverses the tensor only once.
--
-- ==== __Examples__
--
-- >>> let t = fromList [1, 2, 3]
-- >>> runFusion $ TMap (+1) (TMap (*2) t)
-- TMap (\\x -> (x * 2) + 1) t
--
-- ==== __Complexity__
--
-- * Time: O(n) where n is the number of IR nodes
-- * Space: O(1) additional allocation
--
-- ==== __See Also__
--
-- * 'fuseZipMaps' for fusing zipWith with maps
-- * Section 8.1 of H26-SPEC for fusion guarantees
fuseMapChain :: TensorIR -> TensorIR
```

### Type Documentation

```haskell
-- | Configuration for the tensor fusion pass.
--
-- Controls fusion behavior including depth limits and
-- reporting options.
data FusionConfig = FusionConfig
  { fcMaxDepth :: !Int
    -- ^ Maximum nesting depth to fuse. Default: 10.
    -- Deeper chains are fused incrementally.

  , fcEnableReport :: !Bool
    -- ^ Whether to generate a detailed fusion report.
    -- Enabling this has ~5% overhead.

  , fcStrictCheck :: !Bool
    -- ^ If True, fail on any fusion failure for guaranteed
    -- patterns. If False, fall back to scalar loops with warning.
  }
```

### Typeclass Documentation

```haskell
-- | Types that can be stored in unboxed tensors.
--
-- Instances must satisfy:
--
-- * @'sizeOf' a@ returns the size in bytes of a single element
-- * @'alignment' a@ returns the required alignment
-- * Elements must be bitwise copyable (no embedded pointers)
--
-- ==== __Laws__
--
-- @
-- alignment a `mod` sizeOf a == 0
-- @
class TensorElem a where
  -- | Size of a single element in bytes.
  sizeOf :: proxy a -> Int

  -- | Required memory alignment in bytes.
  alignment :: proxy a -> Int

  -- | Read element from memory at given offset.
  peekElem :: Ptr a -> Int -> IO a

  -- | Write element to memory at given offset.
  pokeElem :: Ptr a -> Int -> a -> IO ()
```

---

## Documentation Sections

Use Haddock sections to organize module docs:

```haskell
module BHC.Core.IR
  ( -- * Types
    -- $types
    Expr (..)
  , Type (..)
  , Var (..)

    -- * Construction
    -- $construction
  , mkVar
  , mkLam
  , mkApp

    -- * Queries
  , freeVars
  , boundVars

    -- * Transformation
  , substitute
  , alphaRename
  ) where

-- $types
-- Core IR uses a typed representation where expressions
-- carry their types. This enables type-safe transformations
-- and optimization passes.

-- $construction
-- Smart constructors that maintain IR invariants.
-- Prefer these over raw constructors.
```

---

## Examples

### Code Examples

All non-trivial functions SHOULD have examples:

```haskell
-- | Reshape a tensor to a new shape.
--
-- The new shape must have the same total number of elements.
--
-- ==== __Examples__
--
-- >>> let t = zeros [2, 3]  -- 2x3 matrix
-- >>> reshape [6] t         -- flatten to vector
-- Tensor [6] ...
--
-- >>> reshape [3, 2] t      -- transpose shape
-- Tensor [3, 2] ...
--
-- >>> reshape [4] t         -- error: element count mismatch
-- Left (ShapeMismatch ...)
reshape :: Shape -> Tensor a -> Either ShapeError (Tensor a)
```

### Property Examples

For functions with important properties:

```haskell
-- | Matrix multiplication.
--
-- ==== __Properties__
--
-- prop> matmul (identity n) m == m
-- prop> matmul m (identity n) == m
-- prop> matmul a (matmul b c) == matmul (matmul a b) c
--
-- ==== __Complexity__
--
-- * Naive: O(n³)
-- * With BLAS backend: implementation-dependent, typically O(n^2.37)
matmul :: Tensor Float -> Tensor Float -> Tensor Float
```

---

## Warnings and Notes

### Partial Functions

```haskell
-- | Get the first element of a non-empty tensor.
--
-- __WARNING__: Partial function. Throws 'EmptyTensorError' if tensor
-- has zero elements. Prefer 'headMaybe' for safe access.
head :: Tensor a -> a
```

### Unsafe Functions

```haskell
-- | /Unsafe/: Create a tensor from a raw pointer without copying.
--
-- The caller MUST ensure:
--
-- * The pointer remains valid for the lifetime of the tensor
-- * The memory region has at least @product shape * sizeOf elem@ bytes
-- * The memory is properly aligned for the element type
--
-- Violating these requirements causes undefined behavior.
unsafeFromPtr :: TensorElem a => Shape -> Ptr a -> Tensor a
```

### Complexity Notes

```haskell
-- | Sort the elements of a tensor.
--
-- ==== __Complexity__
--
-- * Time: O(n log n) comparisons
-- * Space: O(n) temporary allocation (not in-place)
--
-- For large tensors, consider 'sortInPlace' to avoid allocation.
sort :: Ord a => Tensor a -> Tensor a
```

---

## Spec References

All implementations of spec requirements MUST reference the spec:

```haskell
-- | Fuse map composition as required by H26-SPEC Section 8.1.
--
-- This is a /guaranteed/ fusion pattern. Failure to fuse is
-- a conformance violation.
--
-- See: H26-SPEC-0001, Section 8.1, Pattern 1
fuseMapMap :: TensorIR -> TensorIR
```

---

## Architecture Documentation

### Design Docs

Major subsystems SHOULD have design docs in `spec/`:

```
spec/
├── H26-SPEC-0001-platform.md     # Main platform spec
├── BHC-ARCH-0001-compiler.md     # Compiler architecture
├── BHC-ARCH-0002-rts.md          # Runtime system design
├── BHC-ARCH-0003-tensor-ir.md    # Tensor IR design
└── BHC-DECISION-*.md             # Architecture decision records
```

### Architecture Decision Records

For significant decisions:

```markdown
# ADR-001: Use LLVM as Primary Backend

## Status
Accepted

## Context
We need a backend for code generation that supports multiple targets.

## Decision
We will use LLVM as the primary code generation backend.

## Consequences
- Good: Mature optimization pipeline
- Good: Multi-target support (x86, ARM, WASM)
- Bad: Large dependency
- Bad: LLVM version churn
```

---

## README Files

Each major directory SHOULD have a README:

```markdown
# BHC Tensor IR

This directory contains the Tensor IR implementation.

## Overview

Tensor IR is the intermediate representation used for numeric
computation optimization. It preserves shape and stride information
needed for fusion and vectorization.

## Directory Structure

- `IR.hs` - Core Tensor IR types
- `Lower.hs` - Lowering from Core IR
- `Fusion.hs` - Fusion passes
- `Shape.hs` - Shape analysis

## Key Concepts

- **TensorOp**: Operations on tensors
- **Shape**: Static or dynamic shape information
- **Stride**: Memory layout information

## See Also

- [H26-SPEC Section 7](../spec/H26-SPEC-0001.md#section-7) - Tensor Model
- [BHC-ARCH-0003](../spec/BHC-ARCH-0003.md) - Tensor IR Design
```
