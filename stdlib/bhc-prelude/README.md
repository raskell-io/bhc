# bhc-prelude

Minimal Rust support crate for the BHC Prelude.

## Overview

This crate provides the Rust-side support for the BHC Prelude. The actual Prelude is implemented in Haskell (`src/Prelude.hs`) and compiled by the BHC compiler itself. This Rust crate exists primarily for build system integration.

## Architecture

```
+------------------+
|   Prelude.hs     |  <- Real Prelude (Haskell)
+------------------+
         |
         | compiled by BHC
         v
+------------------+
|   bhc-prelude    |  <- This crate (build support)
+------------------+
```

## Contents

The Prelude provides fundamental types and functions that are automatically available in every Haskell module:

- Basic types (`Bool`, `Maybe`, `Either`, `Ordering`)
- Numeric classes (`Num`, `Integral`, `Fractional`)
- Common functions (`map`, `filter`, `fold`, etc.)
- I/O primitives (`IO`, `print`, `putStrLn`)

## Usage

The Prelude is implicitly imported into every module:

```haskell
-- Prelude is automatically available
main :: IO ()
main = print (map (+1) [1, 2, 3])
```

To avoid the implicit import:

```haskell
{-# LANGUAGE NoImplicitPrelude #-}
import qualified Prelude
```

## Design Notes

- The Prelude is intentionally minimal
- Advanced functionality lives in `bhc-base`
- Profile-specific behavior is handled at compile time
- Numeric operations respect the active profile

## Related Crates

- `bhc-base` - Extended standard library
- `bhc-numeric` - Numeric primitives
- `bhc-rts` - Runtime system

## Specification References

- H26-SPEC Section 5: Standard Library
- H26-SPEC Section 5.1: Prelude
