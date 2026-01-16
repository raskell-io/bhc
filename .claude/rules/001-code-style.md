# Code Style Guidelines

**Rule ID:** BHC-RULE-001
**Applies to:** All Haskell code in the BHC codebase

---

## General Principles

1. **Readability over cleverness** — Code is read far more than written
2. **Consistency** — Follow existing patterns in the codebase
3. **Explicitness** — Prefer explicit over implicit when it aids understanding

---

## Formatting

### Line Length

- MUST NOT exceed 100 characters
- SHOULD aim for 80 characters for prose/comments
- Long type signatures MAY break at arrows

### Indentation

- MUST use 2 spaces (no tabs)
- Continuation lines MUST be indented relative to the construct they continue

```haskell
-- Good
longFunction :: SomeType
             -> AnotherType
             -> ResultType

-- Good (alternative)
longFunction
  :: SomeType
  -> AnotherType
  -> ResultType
```

### Alignment

- SHOULD align related items vertically when it improves readability
- MUST NOT over-align (causing excessive whitespace changes on edits)

```haskell
-- Good: aligned record fields
data Config = Config
  { configPort    :: Int
  , configHost    :: String
  , configTimeout :: Duration
  }

-- Good: aligned case alternatives (when short)
case x of
  Left err  -> handleError err
  Right val -> processValue val
```

### Blank Lines

- MUST have exactly one blank line between top-level definitions
- SHOULD use blank lines within functions to separate logical sections
- MUST NOT have trailing blank lines at end of file

---

## Imports

### Order

Imports MUST be grouped and ordered as follows:

1. Standard library (`base`, `containers`, etc.)
2. External dependencies (alphabetically)
3. Internal modules (alphabetically)

Each group separated by a blank line.

```haskell
import Control.Monad (forM_, when)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map

import qualified LLVM.AST as LLVM

import BHC.Core.IR (Expr, Type)
import BHC.Parser.AST (Module)
```

### Qualified Imports

- MUST use qualified imports for modules with common names (`Map`, `Set`, `Text`, `ByteString`)
- SHOULD use short, conventional qualifiers

```haskell
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import qualified Data.Text as T
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
```

### Explicit Import Lists

- SHOULD use explicit import lists for external modules
- MAY omit for internal modules in the same package
- MUST NOT use explicit lists that become unwieldy (>10 items)

---

## Declarations

### Type Signatures

- MUST have type signatures for all top-level bindings
- SHOULD have type signatures for non-trivial local bindings
- MAY omit for trivial local bindings (`let x = 1`)

### Function Definitions

- SHOULD use pattern matching over `case` when matching on a single argument
- MUST align guards vertically

```haskell
-- Good: pattern matching
eval (Lit n)       = pure n
eval (Add e1 e2)   = (+) <$> eval e1 <*> eval e2
eval (Var x)       = lookupVar x

-- Good: guards
classify n
  | n < 0     = Negative
  | n == 0    = Zero
  | otherwise = Positive
```

### Where vs Let

- SHOULD use `where` for auxiliary definitions used in guards/patterns
- SHOULD use `let` for inline computations
- MUST NOT nest `where` clauses deeply (max 1 level)

---

## Comments

### When to Comment

- MUST document non-obvious algorithms
- MUST document why, not what
- SHOULD NOT comment obvious code
- MUST document all public API functions (see 004-documentation.md)

### Comment Style

```haskell
-- Single line comment for brief notes

-- | Haddock documentation for exported items.
-- Continues on next line.
functionName :: Type -> Type

{-
   Block comments for longer explanations
   that span multiple lines.
-}

-- TODO: Brief description of what needs to be done
-- FIXME: Brief description of known issue
-- NOTE: Important information for maintainers
```

### Section Headers

Use section headers to organize large modules:

```haskell
-- ============================================================
-- Section Name
-- ============================================================

-- Or for subsections:

-- ------------------------------------------------------------
-- Subsection Name
-- ------------------------------------------------------------
```

---

## Language Extensions

### Approved Extensions (Always Safe)

These MAY be used freely:

```haskell
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
```

### Restricted Extensions (Require Justification)

These SHOULD be used sparingly with clear justification:

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE UndecidableInstances #-}
```

### Forbidden Extensions

These MUST NOT be used:

```haskell
{-# LANGUAGE Unsafe #-}           -- Use SafeHaskell instead
{-# LANGUAGE OverlappingInstances #-}  -- Use OVERLAPPING pragma
```

---

## Error Messages

- MUST be actionable (tell user what to do)
- SHOULD include source location
- SHOULD suggest fixes when possible

```haskell
-- Good
error $ "Type mismatch at " ++ showLoc loc ++ ": "
     ++ "expected " ++ showType expected
     ++ ", got " ++ showType actual
     ++ ". Did you mean to use `fromIntegral`?"

-- Bad
error "type error"
```

---

## Tooling

### Formatter

- MUST run `ormolu` or `fourmolu` before committing
- Configuration in `.ormolu` or `fourmolu.yaml`

### Linter

- MUST pass `hlint` with no warnings
- Custom rules in `.hlint.yaml`

### Pre-commit Hook

All code MUST pass:
```bash
ormolu --mode check
hlint .
cabal build
cabal test
```
