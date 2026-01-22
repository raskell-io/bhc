# BHC Base Library

Extended Data.*, Control.*, and System.* modules for BHC.

## Overview

The BHC Base library extends the Prelude with additional modules:

### Data Modules

- **Data.List** - Extended list operations (sort, group, partition)
- **Data.Char** - Unicode character operations
- **Data.String** - String manipulation
- **Data.Function** - Extended function combinators
- **Data.Ord** - Extended ordering operations

### Control Modules

- **Control.Monad** - Extended monadic operations
- **Control.Applicative** - Extended applicative operations
- **Control.Category** - Category theory abstractions

### System Modules

- **System.IO** - File and console I/O
- **System.Environment** - Environment variable access

## Quick Start

```haskell
import BHC.Data.List (sort, group, partition)
import BHC.Data.Char (isAlpha, toUpper)
import BHC.System.IO (readFile, writeFile)

main :: IO ()
main = do
  contents <- readFile "input.txt"
  let sorted = sort (lines contents)
  writeFile "output.txt" (unlines sorted)
```

## See Also

- [BHC.Prelude](../bhc-prelude/docs/README.md) - Core types and functions
- [BHC.Data.Text](../bhc-text/docs/README.md) - Efficient text processing
