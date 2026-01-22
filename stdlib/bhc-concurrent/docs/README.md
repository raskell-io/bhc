# BHC Concurrent Library

Structured concurrency for BHC.

## Overview

The BHC Concurrent library provides:

- **Scope** - Scoped task execution
- **Task** - Task creation and management
- **Channel** - Communication channels
- **STM** - Software transactional memory

## Quick Start

```haskell
import BHC.Control.Concurrent.Scope
import BHC.Control.Concurrent.Task
import BHC.Control.Concurrent.Channel

main :: IO ()
main = withScope $ \scope -> do
  -- Spawn concurrent tasks
  t1 <- spawn scope $ computeX
  t2 <- spawn scope $ computeY

  -- Wait for results
  x <- await t1
  y <- await t2

  print (x + y)
  -- Tasks are guaranteed to complete before withScope returns
```

## Features

### Structured Concurrency

All tasks complete within their scope:

```haskell
withScope :: (Scope -> IO a) -> IO a

-- Tasks cannot escape their scope
spawn :: Scope -> IO a -> IO (Task a)
await :: Task a -> IO a
```

### Cancellation

Cooperative cancellation with propagation:

```haskell
cancel :: Task a -> IO ()

-- Check for cancellation in long loops
checkpoint :: IO ()

-- Handle cancellation
onCancel :: IO a -> IO () -> IO a
```

### Deadlines

Time-bounded operations:

```haskell
withDeadline :: Duration -> IO a -> IO (Maybe a)

-- Example
result <- withDeadline (seconds 5) longOperation
```

### Channels

Bounded and unbounded communication:

```haskell
-- Bounded channel
chan <- newBoundedChan 100
writeChan chan value
value <- readChan chan

-- Unbounded channel
chan <- newChan
writeChan chan value
```

### STM

Software transactional memory:

```haskell
atomically :: STM a -> IO a

-- Transactional variables
newTVar :: a -> STM (TVar a)
readTVar :: TVar a -> STM a
writeTVar :: TVar a -> a -> STM ()

-- Composable transactions
retry :: STM a
orElse :: STM a -> STM a -> STM a
```

## Example: Producer-Consumer

```haskell
producerConsumer :: IO ()
producerConsumer = withScope $ \scope -> do
  chan <- newBoundedChan 10

  -- Producer
  producer <- spawn scope $ do
    forM_ [1..100] $ \i -> do
      writeChan chan i
      checkpoint

  -- Consumer
  consumer <- spawn scope $ do
    replicateM_ 100 $ do
      x <- readChan chan
      print x

  await producer
  await consumer
```

## See Also

- [009-concurrency.md](../../.claude/rules/009-concurrency.md) - Concurrency guidelines
- [Design](DESIGN.md) - Implementation details
