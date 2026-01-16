# Type Safety Rules

**Rule ID:** BHC-RULE-002
**Applies to:** All BHC code, especially Core IR and type system

---

## Core Principles

1. **Make illegal states unrepresentable** — Use types to prevent invalid data
2. **Fail at compile time, not runtime** — Push errors left
3. **Preserve type information** — Don't erase types prematurely

---

## Newtypes for Semantic Distinction

### MUST Use Newtypes For

- IDs and indices that shouldn't be confused
- Units of measure
- Validated data

```haskell
-- Good: Can't mix up variable and type IDs
newtype VarId = VarId Int
  deriving (Eq, Ord, Show)

newtype TypeId = TypeId Int
  deriving (Eq, Ord, Show)

-- Good: Can't add bytes to elements
newtype ByteOffset = ByteOffset Int
newtype ElemOffset = ElemOffset Int

-- Good: Validated email can't be constructed invalidly
newtype Email = Email Text  -- Constructor not exported

mkEmail :: Text -> Either ValidationError Email
```

### SHOULD Use Newtypes For

- Domain concepts that wrap primitives
- Configuration values
- Measurement values

---

## Phantom Types

Use phantom types to track state at the type level:

```haskell
-- Track whether a tensor is on host or device
data Host
data Device

newtype Tensor (loc :: *) dtype = Tensor RawBuffer

-- Type-safe operations
toDevice :: Tensor Host a -> IO (Tensor Device a)
fromDevice :: Tensor Device a -> IO (Tensor Host a)

-- Can't accidentally run host ops on device tensors
hostOp :: Tensor Host Float -> Tensor Host Float
```

---

## Smart Constructors

### Pattern

- Don't export data constructors for validated types
- Export smart constructors that enforce invariants
- Export pattern synonyms for matching if needed

```haskell
module BHC.Core.Shape
  ( Shape  -- Abstract, no constructors
  , mkShape
  , pattern Shape
  , shapeRank
  , shapeDims
  ) where

-- Internal representation
data Shape = UnsafeShape [Int]

-- Smart constructor validates invariant
mkShape :: [Int] -> Either ShapeError Shape
mkShape dims
  | any (< 0) dims = Left NegativeDimension
  | otherwise = Right (UnsafeShape dims)

-- Pattern for matching (uni-directional)
pattern Shape :: [Int] -> Shape
pattern Shape dims <- UnsafeShape dims
```

---

## GADTs for IR Type Safety

The Core IR and Tensor IR SHOULD use GADTs to ensure type safety:

```haskell
-- Typed Core IR expressions
data Expr (t :: Type) where
  Lit     :: Int -> Expr IntT
  LitF    :: Float -> Expr FloatT
  Var     :: Var t -> Expr t
  App     :: Expr (a :-> b) -> Expr a -> Expr b
  Lam     :: Var a -> Expr b -> Expr (a :-> b)
  Add     :: Num t => Expr t -> Expr t -> Expr t
  If      :: Expr BoolT -> Expr t -> Expr t -> Expr t

-- Type-safe evaluation
eval :: Expr t -> Value t
eval (Lit n) = VInt n
eval (LitF f) = VFloat f
-- etc.
```

---

## Avoiding Partial Functions

### MUST NOT Use

- `head`, `tail`, `init`, `last` on possibly-empty lists
- `fromJust`
- `read` without validation
- Incomplete pattern matches

### MUST Use Instead

```haskell
-- Use NonEmpty when list must have elements
import Data.List.NonEmpty (NonEmpty(..))

processItems :: NonEmpty Item -> Result
processItems (x :| xs) = ...  -- Pattern match is total

-- Use safe alternatives
import Data.Maybe (fromMaybe, listToMaybe)

firstItem :: [a] -> Maybe a
firstItem = listToMaybe

-- Use explicit error handling
parseConfig :: Text -> Either ParseError Config
parseConfig txt = case decode txt of
  Nothing -> Left $ ParseError "Invalid JSON"
  Just c  -> Right c
```

### Pattern Match Warnings

- MUST compile with `-Wincomplete-patterns`
- MUST NOT use `-fno-warn-incomplete-patterns`
- All case expressions MUST be exhaustive

---

## Result Types Over Exceptions

### Prefer Either for Recoverable Errors

```haskell
-- Good: explicit error handling
data TypeError
  = Mismatch Type Type
  | UnboundVar Name
  | OccursCheck TyVar Type

typeCheck :: Expr -> Either TypeError Type
typeCheck expr = case expr of
  Var name -> lookupVar name  -- Returns Either
  App f x -> do
    fTy <- typeCheck f
    xTy <- typeCheck x
    unify fTy (xTy :-> freshVar)
```

### Use Exceptions for Unrecoverable Errors

- Programmer errors (invariant violations)
- Should never happen in correct code
- MUST include diagnostic information

```haskell
-- Internal invariant violation
invariant :: HasCallStack => Bool -> String -> a -> a
invariant True _ x = x
invariant False msg _ =
  error $ "Invariant violation: " ++ msg ++ "\n" ++ prettyCallStack callStack
```

---

## Avoiding Stringly-Typed Code

### MUST NOT

```haskell
-- Bad: string-typed operations
data Op = Op String

eval :: Op -> Int -> Int -> Int
eval (Op "+") = (+)
eval (Op "-") = (-)
eval (Op _)   = error "unknown op"  -- Runtime error!
```

### MUST

```haskell
-- Good: algebraic data type
data Op = Add | Sub | Mul | Div

eval :: Op -> Int -> Int -> Int
eval Add = (+)
eval Sub = (-)
eval Mul = (*)
eval Div = div  -- Total pattern match
```

---

## Type-Level Programming Guidelines

### When to Use

- Shape checking for tensors
- Protocol state machines
- Resource tracking (linear types)

### When to Avoid

- Simple enumerations (use regular ADTs)
- When it significantly complicates error messages
- When simpler runtime checks suffice

### Example: Shape-Safe Tensors

```haskell
{-# LANGUAGE DataKinds, TypeOperators, GADTs #-}

-- Type-level natural numbers
data Nat = Z | S Nat

-- Shape as type-level list
data Tensor (shape :: [Nat]) dtype where
  Scalar :: dtype -> Tensor '[] dtype
  Tensor :: Array dtype -> Tensor shape dtype

-- Type-safe matrix multiply
matmul :: Tensor '[m, k] Float
       -> Tensor '[k, n] Float
       -> Tensor '[m, n] Float
```

---

## Invariants Documentation

All types with non-trivial invariants MUST document them:

```haskell
-- | A sorted, non-empty list of unique elements.
--
-- Invariants:
-- - Length >= 1
-- - Elements are strictly ascending
-- - No duplicate elements
--
-- Construct using 'fromList' which validates invariants.
newtype SortedList a = UnsafeSortedList (NonEmpty a)

-- | Construct a SortedList, validating invariants.
--
-- Returns Nothing if:
-- - Input list is empty
-- - Input contains duplicates
fromList :: Ord a => [a] -> Maybe (SortedList a)
```
