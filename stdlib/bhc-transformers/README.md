# bhc-transformers

Monad transformers for the Basel Haskell Compiler.

## Overview

This crate provides Rust-side support for BHC monad transformers. The actual transformer implementations are in Haskell (`hs/BHC/Control/Monad/*.hs`), as monad transformers are quintessentially Haskell constructs that don't require Rust primitives.

## Transformers Provided

| Transformer | Description |
|-------------|-------------|
| `ReaderT` | Reader monad transformer |
| `WriterT` | Writer monad transformer |
| `StateT` | State monad transformer |
| `ExceptT` | Exception monad transformer |
| `MaybeT` | Maybe monad transformer |
| `IdentityT` | Identity monad transformer |
| `ContT` | Continuation monad transformer |
| `RWST` | Combined Reader-Writer-State transformer |

## Usage

### ReaderT

```haskell
import Control.Monad.Reader

type App = ReaderT Config IO

runApp :: Config -> App a -> IO a
runApp config app = runReaderT app config

getPort :: App Int
getPort = asks configPort
```

### StateT

```haskell
import Control.Monad.State

type Counter = StateT Int IO

increment :: Counter ()
increment = modify (+1)

runCounter :: Counter a -> IO (a, Int)
runCounter = flip runStateT 0
```

### ExceptT

```haskell
import Control.Monad.Except

type Result = ExceptT AppError IO

safeDiv :: Int -> Int -> Result Int
safeDiv _ 0 = throwError DivByZero
safeDiv x y = pure (x `div` y)
```

### RWST (Combined)

```haskell
import Control.Monad.RWS

type App = RWST Config [Log] AppState IO

app :: App Result
app = do
  cfg <- ask
  tell [LogInfo "Starting"]
  s <- get
  modify updateState
  pure result
```

## Monad Transformer Laws

All transformers satisfy the monad transformer laws:

```haskell
-- lift preserves return
lift . return = return

-- lift preserves bind
lift (m >>= f) = lift m >>= (lift . f)
```

## Design Notes

- This Rust crate is intentionally minimal
- All transformer logic is in Haskell
- BHC compiles transformers directly
- No FFI overhead for transformer operations

## Related Crates

- `bhc-prelude` - Monad class definitions
- `bhc-base` - Functor, Applicative, Monad

## Specification References

- H26-SPEC Section 5: Standard Library
- H26-SPEC Section 5.5: Monad Transformers
