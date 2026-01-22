# BHC Transformers Library

Monad transformers for composing effects.

## Overview

The BHC Transformers library provides monad transformers:

- **ReaderT** - Environment/configuration access
- **WriterT** - Logging/accumulation
- **StateT** - Mutable state
- **ExceptT** - Error handling
- **MaybeT** - Optional computation
- **IdentityT** - Base transformer

## Quick Start

```haskell
import BHC.Control.Monad.Reader
import BHC.Control.Monad.State
import BHC.Control.Monad.Except

-- Configuration reader
type App = ReaderT Config IO

runApp :: Config -> App a -> IO a
runApp config app = runReaderT app config

-- Stateful computation
increment :: State Int ()
increment = modify (+1)

-- Error handling
safeDiv :: Int -> Int -> Except String Int
safeDiv _ 0 = throwError "Division by zero"
safeDiv x y = return (x `div` y)
```

## Transformer Stack

Transformers can be stacked for combined effects:

```haskell
type MyApp = ReaderT Config (StateT AppState (ExceptT AppError IO))

runMyApp :: Config -> AppState -> MyApp a -> IO (Either AppError (a, AppState))
runMyApp config state app =
  runExceptT (runStateT (runReaderT app config) state)
```

## MTL-Style Classes

The library also provides MTL-style type classes:

```haskell
class MonadReader r m where
  ask :: m r
  local :: (r -> r) -> m a -> m a

class MonadState s m where
  get :: m s
  put :: s -> m ()

class MonadError e m where
  throwError :: e -> m a
  catchError :: m a -> (e -> m a) -> m a
```

## See Also

- [Design](DESIGN.md) - Transformer laws and design
- [BHC.Prelude](../bhc-prelude/docs/README.md) - Base Monad class
