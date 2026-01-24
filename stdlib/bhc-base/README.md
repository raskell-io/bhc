# bhc-base

Extended standard library for the Basel Haskell Compiler.

## Overview

This crate provides Rust-side primitives for the BHC base library. It includes character handling primitives that are exposed to Haskell code through the FFI.

## Key Types

| Type | Description |
|------|-------------|
| `CharCategory` | Unicode general category |
| Character primitives | Case conversion, classification |

## Character Primitives

```rust
/// Check if character is alphabetic
pub fn is_alpha(c: char) -> bool;

/// Check if character is alphanumeric
pub fn is_alpha_num(c: char) -> bool;

/// Convert to uppercase
pub fn to_upper(c: char) -> char;

/// Convert to lowercase
pub fn to_lower(c: char) -> char;

/// Get Unicode general category
pub fn general_category(c: char) -> CharCategory;
```

## Unicode Categories

```rust
pub enum CharCategory {
    UppercaseLetter,
    LowercaseLetter,
    TitlecaseLetter,
    ModifierLetter,
    OtherLetter,
    NonSpacingMark,
    SpacingCombiningMark,
    EnclosingMark,
    DecimalNumber,
    LetterNumber,
    OtherNumber,
    ConnectorPunctuation,
    DashPunctuation,
    OpenPunctuation,
    ClosePunctuation,
    InitialQuote,
    FinalQuote,
    OtherPunctuation,
    MathSymbol,
    CurrencySymbol,
    ModifierSymbol,
    OtherSymbol,
    Space,
    LineSeparator,
    ParagraphSeparator,
    Control,
    Format,
    Surrogate,
    PrivateUse,
    NotAssigned,
}
```

## Haskell Interface

These primitives are exposed to Haskell through FFI:

```haskell
-- In Data.Char
foreign import ccall "bhc_is_alpha" isAlpha :: Char -> Bool
foreign import ccall "bhc_to_upper" toUpper :: Char -> Char
foreign import ccall "bhc_general_category" generalCategory :: Char -> GeneralCategory
```

## Design Notes

- Uses Rust's built-in Unicode support
- Full Unicode 15.0 compliance
- Zero-allocation for simple queries
- Thread-safe (pure functions)

## Related Crates

- `bhc-prelude` - Minimal prelude
- `bhc-text` - Text processing
- `bhc-rts` - Runtime system

## Specification References

- H26-SPEC Section 5: Standard Library
- H26-SPEC Section 5.2: Data.Char
