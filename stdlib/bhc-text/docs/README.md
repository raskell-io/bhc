# BHC Text Library

UTF-8 Text and ByteString types with SIMD acceleration.

## Overview

The BHC Text library provides efficient text processing:

- **Data.Text** - UTF-8 encoded text with SIMD operations
- **Data.Text.Lazy** - Lazy text for streaming
- **Data.ByteString** - Raw byte arrays (pinned for FFI)
- **Data.ByteString.Lazy** - Lazy byte strings
- **Data.ByteString.Builder** - Efficient construction

## Quick Start

```haskell
import qualified BHC.Data.Text as T
import qualified BHC.Data.ByteString as BS

main :: IO ()
main = do
  let text = T.pack "Hello, World!"
  print $ T.length text      -- 13
  print $ T.toUpper text     -- "HELLO, WORLD!"

  bytes <- BS.readFile "data.bin"
  print $ BS.length bytes
```

## Features

### SIMD Acceleration

Text operations automatically use SIMD when available:

- `isAscii` - Vectorized ASCII check
- `toLower/toUpper` - Batch case conversion
- `find` - Pattern searching

### UTF-8 Native

Text is stored as UTF-8 internally:

- Memory efficient for ASCII-heavy text
- Fast for most string operations
- Compatible with modern APIs

## Performance vs GHC

| Operation | BHC | GHC text | Notes |
|-----------|-----|----------|-------|
| `toLower` 1M | 0.8ms | 1.2ms | SIMD |
| `find` 1M | 0.3ms | 0.5ms | SIMD |
| `concat` | O(n) | O(n) | Same |

## See Also

- [Design](DESIGN.md) - Internal representation
- [Benchmarks](BENCHMARKS.md) - Performance data
