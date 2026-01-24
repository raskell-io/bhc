# bhc-text

High-performance text processing for the Basel Haskell Compiler.

## Overview

This crate provides SIMD-accelerated text primitives for BHC. It implements efficient UTF-8 processing, text search, and transformation operations.

## Key Features

| Feature | Description |
|---------|-------------|
| SIMD Search | Vectorized substring search |
| UTF-8 Validation | Fast parallel validation |
| Case Conversion | Bulk case transformations |
| Text Builder | Efficient text construction |

## SIMD-Accelerated Operations

### Substring Search

```rust
/// Find first occurrence of needle in haystack
/// Uses SIMD for needles <= 16 bytes
pub fn find(haystack: &str, needle: &str) -> Option<usize>;

/// Find all occurrences
pub fn find_all(haystack: &str, needle: &str) -> Vec<usize>;
```

### UTF-8 Validation

```rust
/// Validate UTF-8 using SIMD
pub fn is_valid_utf8(bytes: &[u8]) -> bool;

/// Find first invalid byte position
pub fn validate_utf8(bytes: &[u8]) -> Result<(), usize>;
```

### Case Conversion

```rust
/// Convert to uppercase (SIMD for ASCII)
pub fn to_uppercase(s: &str) -> String;

/// Convert to lowercase (SIMD for ASCII)
pub fn to_lowercase(s: &str) -> String;
```

## Text Builder

```rust
pub struct TextBuilder {
    buffer: Vec<u8>,
    len: usize,
}

impl TextBuilder {
    pub fn new() -> Self;
    pub fn with_capacity(cap: usize) -> Self;
    pub fn push_str(&mut self, s: &str);
    pub fn push_char(&mut self, c: char);
    pub fn build(self) -> String;
}
```

## FFI Interface

```haskell
-- Text search
foreign import ccall "bhc_text_find" find :: Text -> Text -> Maybe Int

-- Validation
foreign import ccall "bhc_is_valid_utf8" isValidUtf8 :: ByteString -> Bool

-- Case conversion
foreign import ccall "bhc_to_uppercase" toUpper :: Text -> Text
foreign import ccall "bhc_to_lowercase" toLower :: Text -> Text
```

## Performance

| Operation | Complexity | SIMD Speedup |
|-----------|------------|--------------|
| Search (short needle) | O(n) | 4-8x |
| UTF-8 validation | O(n) | 8-16x |
| ASCII case conversion | O(n) | 4-8x |
| Unicode case conversion | O(n) | 1x (no SIMD) |

## Design Notes

- UTF-8 internally, no transcoding overhead
- SIMD fast path for ASCII-heavy text
- Fallback to scalar for complex Unicode
- Zero-copy slicing where possible

## Related Crates

- `bhc-base` - Character primitives
- `bhc-prelude` - String basics
- `bhc-numeric` - SIMD infrastructure

## Specification References

- H26-SPEC Section 5: Standard Library
- H26-SPEC Section 5.3: Data.Text
