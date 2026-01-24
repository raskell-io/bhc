# bhc-utils

Utility primitives for the Basel Haskell Compiler.

## Overview

This crate provides Rust-side utility primitives for BHC, including time operations, random number generation, and JSON processing. High-level APIs are defined in Haskell, while this crate provides FFI primitives for operations requiring system access or high performance.

## Modules

| Module | Description |
|--------|-------------|
| `time` | Date, time, and duration |
| `random` | Random number generation |
| `json` | JSON parsing and serialization |

## Time

### Current Time

```haskell
import Data.Time

main :: IO ()
main = do
  now <- getCurrentTime
  today <- getCurrentDay
  zone <- getCurrentTimeZone
  print (now, today, zone)
```

### Measuring Elapsed Time

```haskell
import Data.Time

main :: IO ()
main = do
  start <- getMonotonicTime
  performComputation
  end <- getMonotonicTime
  print $ "Elapsed: " ++ show (end - start)
```

### Duration Operations

```haskell
import Data.Time

main :: IO ()
main = do
  let d = seconds 5 + milliseconds 500
  threadDelay d
```

### Formatting

```haskell
import Data.Time

main :: IO ()
main = do
  now <- getCurrentTime
  putStrLn $ formatTime defaultTimeLocale "%Y-%m-%d %H:%M:%S" now
```

## Random

### Basic Generation

```haskell
import System.Random

main :: IO ()
main = do
  -- Random Int
  n <- randomIO :: IO Int
  print n

  -- Random in range
  r <- randomRIO (1, 100) :: IO Int
  print r
```

### Pure Random

```haskell
import System.Random

main :: IO ()
main = do
  gen <- newStdGen
  let (value, gen') = random gen :: (Int, StdGen)
  let (values, _) = randomRs (1, 10) gen'
  print $ take 10 values
```

### Secure Random

```haskell
import System.Random

main :: IO ()
main = do
  -- Cryptographically secure random bytes
  bytes <- getEntropy 32
  print bytes
```

## JSON

### Parsing

```haskell
import Data.JSON

main :: IO ()
main = do
  let json = "{\"name\": \"Alice\", \"age\": 30}"
  case decode json of
    Just obj -> print (obj .: "name")
    Nothing  -> putStrLn "Parse error"
```

### Encoding

```haskell
import Data.JSON

data Person = Person { name :: String, age :: Int }
  deriving (Generic, ToJSON, FromJSON)

main :: IO ()
main = do
  let person = Person "Alice" 30
  putStrLn $ encode person
```

### Working with Values

```haskell
import Data.JSON

main :: IO ()
main = do
  let obj = object
        [ "name" .= "Alice"
        , "age"  .= (30 :: Int)
        , "tags" .= ["haskell", "rust"]
        ]
  print obj
```

## FFI Exports

### Time

| Function | Description |
|----------|-------------|
| `bhc_time_now` | Current system time |
| `bhc_time_monotonic` | Monotonic clock |
| `bhc_time_elapsed` | Time difference |

### Random

| Function | Description |
|----------|-------------|
| `bhc_rng_new` | Create new RNG |
| `bhc_rng_next_u64` | Generate u64 |
| `bhc_rng_seed` | Seed RNG |
| `bhc_entropy` | Get secure random bytes |

### JSON

| Function | Description |
|----------|-------------|
| `bhc_json_parse` | Parse JSON string |
| `bhc_json_serialize` | Serialize to JSON |

## Key Types

```rust
// Time types
pub struct Instant { ... }
pub struct Duration { ... }
pub struct DateTime { ... }
pub struct Date { ... }
pub struct Time { ... }

// Random
pub struct Rng { ... }

// JSON
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(HashMap<String, Json>),
}
```

## Design Notes

- Time uses monotonic clock for measurements
- Random uses cryptographically secure seeding
- JSON parsing is SIMD-accelerated
- All operations are thread-safe

## Related Crates

- `bhc-prelude` - Basic types
- `bhc-text` - Text processing
- `bhc-system` - System access

## Specification References

- H26-SPEC Section 5: Standard Library
- H26-SPEC Section 5.7: Data.Time
- H26-SPEC Section 5.8: System.Random
