# BHC Prelude Design

## Overview

This document describes the design decisions and rationale for the BHC Prelude.

## Design Goals

1. **Haskell 2010 Compatibility**: Semantically compatible with the Haskell 2010 Prelude
2. **Performance**: Zero-cost abstractions, guaranteed fusion, SIMD acceleration
3. **Profile Support**: Behavior adapts to Default, Numeric, Server, and Edge profiles
4. **Transparency**: Clear performance characteristics, no hidden allocations

## Architecture

### Layered Implementation

```
┌─────────────────────────────────────────────┐
│           Haskell Interface                 │
│         (BHC/Prelude.hs)                    │
├─────────────────────────────────────────────┤
│          Fusion Framework                   │
│        (Rewrite rules, inlining)            │
├─────────────────────────────────────────────┤
│          Rust Primitives                    │
│        (bhc-prelude/src/*.rs)               │
├─────────────────────────────────────────────┤
│            BHC Runtime                      │
│         (Memory, GC, FFI)                   │
└─────────────────────────────────────────────┘
```

### Type Representation

Core types are implemented in Rust for FFI efficiency:

```rust
// Bool: Single byte for C ABI compatibility
#[repr(C)]
pub enum Bool { False = 0, True = 1 }

// Maybe: Tagged union
#[repr(C)]
pub enum Maybe<T> { Nothing, Just(T) }

// Either: Tagged union
#[repr(C)]
pub enum Either<L, R> { Left(L), Right(R) }
```

## Key Decisions

### Decision 1: Guaranteed Fusion

**Context**: GHC's stream fusion sometimes fails silently, causing unexpected allocations.

**Decision**: BHC guarantees fusion for standard patterns. Failure is a compiler error.

**Rationale**:
- Predictable performance is more important than occasional best-case performance
- Developers can rely on documented fusion patterns
- Fusion failures are caught at compile time, not production

**Implementation**:
- Rewrite rules for standard patterns
- Compiler verification pass
- `-fverify-fusion` flag for checking

### Decision 2: Profile-Aware Strictness

**Context**: Lazy evaluation is core to Haskell, but causes space leaks in numeric code.

**Decision**: Strictness adapts to the active profile:
- Default Profile: Lazy evaluation (Haskell standard)
- Numeric Profile: Strict evaluation (no thunks in hot paths)
- Edge Profile: Selective strictness (minimal footprint)

**Rationale**:
- Preserves Haskell semantics for general code
- Enables predictable performance for numeric workloads
- Allows optimization for constrained environments

**Implementation**:
- `{-# PROFILE Numeric #-}` pragma
- Profile-specific function variants
- Compiler specialization

### Decision 3: Rust-Backed Primitives

**Context**: Some operations benefit from native code, especially SIMD and memory operations.

**Decision**: Performance-critical primitives are implemented in Rust, exposed via FFI.

**Rationale**:
- SIMD intrinsics are more accessible in Rust
- Memory safety guarantees complement Haskell's type safety
- Shared codebase with RTS components

**Considerations**:
- FFI overhead must be amortized over sufficient work
- Simple operations remain in pure Haskell
- Rust code must not trigger GC or callbacks

### Decision 4: Kahan Summation for Floats

**Context**: Naive floating-point summation accumulates errors.

**Decision**: Use compensated (Kahan) summation for `sum` on floating-point types.

**Rationale**:
- Significantly better precision for large arrays
- Minimal performance overhead (~10%)
- Matches scientific computing expectations

**Trade-offs**:
- Slightly slower than naive sum
- May give different results than GHC
- Can be bypassed with `foldl' (+) 0` for maximum speed

### Decision 5: Division Semantics

**Context**: Haskell's `div`/`mod` differ from C's `/`/`%` for negative numbers.

**Decision**: Provide both Haskell-style (`div`, `mod`) and C-style (`quot`, `rem`).

| Function | Behavior | Example: -7 `op` 3 |
|----------|----------|-------------------|
| `div` | Floor division | -3 |
| `mod` | Same sign as divisor | 2 |
| `quot` | Truncation | -2 |
| `rem` | Same sign as dividend | -1 |

**Implementation**: Both in Rust to ensure consistent behavior.

## Type Classes

### Numeric Hierarchy

```
       Num
      /   \
   Real   Fractional
     \     /    \
    RealFrac   Floating
        \       /
       RealFloat
```

**Design Notes**:
- `fromInteger` is the only way to create numeric literals
- `Fractional` adds division and `fromRational`
- `Floating` adds transcendental functions
- SIMD acceleration applies at each level

### Functor-Applicative-Monad

```
Functor
   |
Applicative
   |
Monad
```

**Design Notes**:
- `pure` is preferred over `return` (same implementation)
- `<*>` can be defined in terms of `>>=` but has dedicated implementation for efficiency
- `fail` is being deprecated (use `MonadFail`)

## Memory Model

### List Representation

Lists use spine-strict cons cells:

```
List a = Nil | Cons !a (List a)
         ^^^   ^^^^ ^
         |     |    +-- Strict head
         |     +------- Constructor
         +------------- Empty list
```

**Rationale**:
- Head strictness prevents thunk chains
- Still allows lazy tails for infinite lists
- Compatible with fusion framework

### Unboxed Numerics

Numeric types are unboxed in Numeric Profile:

```haskell
-- Default Profile: Boxed
data Int = I# Int#

-- Numeric Profile: Unboxed
newtype Int = Int Int#  -- No allocation
```

## Fusion Framework

### Rewrite Rules

Standard fusion patterns:

```haskell
{-# RULES
-- Map fusion
"map/map"    forall f g xs. map f (map g xs) = map (f . g) xs

-- Filter fusion
"filter/filter" forall p q xs. filter p (filter q xs) = filter (\x -> q x && p x) xs

-- Fold fusion
"sum/map"    forall f xs. sum (map f xs) = foldl' (\a x -> a + f x) 0 xs

-- Build/fold fusion
"fold/build" forall f z (g :: forall b. (a -> b -> b) -> b -> b).
             foldr f z (build g) = g f z
#-}
```

### Verification

The compiler verifies fusion with `-fverify-fusion`:

1. Parse and type-check code
2. Apply rewrite rules
3. Check that guaranteed patterns fused
4. Report errors for fusion failures

## Error Handling

### Error Messages

All errors include source location:

```haskell
error :: HasCallStack => String -> a
error msg = errorWithCallStack msg callStack
```

Output:
```
*** Exception: Index out of bounds: 10 for length 5
CallStack:
  (!!), src/Data/List.hs:42
  main, src/Main.hs:15
```

### Partial Functions

Partial functions are marked and have safe alternatives:

| Partial | Safe Alternative | Returns |
|---------|------------------|---------|
| `head` | `listToMaybe` | `Maybe a` |
| `tail` | `uncons` | `Maybe (a, [a])` |
| `(!)` | `lookup` | `Maybe a` |
| `fromJust` | `fromMaybe` | `a` (with default) |

## Testing Strategy

### Property Tests

Every function has QuickCheck properties:

```haskell
-- Functor laws
prop_functor_identity :: [Int] -> Bool
prop_functor_identity xs = fmap id xs == xs

prop_functor_composition :: Fun Int Int -> Fun Int Int -> [Int] -> Bool
prop_functor_composition (Fn f) (Fn g) xs =
  fmap (f . g) xs == (fmap f . fmap g) xs
```

### Fusion Tests

Fusion is verified programmatically:

```haskell
test_map_map_fuses :: Assertion
test_map_map_fuses = do
  let ir = compileToCore "map (+1) (map (*2) xs)"
  assertNoIntermediateList ir
```

## Future Directions

1. **Dependent Types**: Shape-safe operations with dependent types
2. **Linear Types**: Resource-safe I/O with linear types
3. **Effect System**: Algebraic effects for better abstraction
4. **GPU Offload**: Automatic GPU acceleration for large arrays
