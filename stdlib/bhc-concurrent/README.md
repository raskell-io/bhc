# bhc-concurrent

Structured concurrency and STM primitives for the Basel Haskell Compiler.

## Overview

This crate provides Rust-side runtime primitives for BHC's structured concurrency model and Software Transactional Memory (STM). The high-level concurrency API is defined in Haskell, while this crate provides the low-level runtime support.

## Key Features

| Feature | Description |
|---------|-------------|
| Structured Concurrency | Scopes guarantee task completion |
| Cancellation | Cooperative cancellation with propagation |
| STM | Software transactional memory |
| Channels | MPSC and broadcast channels |

## Modules

| Module | Description |
|--------|-------------|
| `scope` | Structured concurrency scopes |
| `task` | Task handles and lifecycle |
| `stm` | STM runtime (TVar, atomically) |
| `channel` | Channel primitives |

## Structured Concurrency

### Scopes

```haskell
import Control.Concurrent.Scope

main :: IO ()
main = withScope $ \scope -> do
  task1 <- spawn scope $ computeX
  task2 <- spawn scope $ computeY
  x <- await task1
  y <- await task2
  print (x + y)
-- All tasks complete before scope exits
```

### Cancellation

```haskell
import Control.Concurrent.Scope

main :: IO ()
main = withScope $ \scope -> do
  task <- spawn scope longRunningTask
  -- Cancel after timeout
  threadDelay 1000000
  cancel task
```

### Deadlines

```haskell
import Control.Concurrent.Scope

main :: IO ()
main = do
  result <- withDeadline (seconds 5) $ \scope -> do
    spawn scope expensiveComputation >>= await
  case result of
    Just x  -> print x
    Nothing -> putStrLn "Timed out"
```

## STM

### TVar Operations

```haskell
import Control.Concurrent.STM

main :: IO ()
main = do
  counter <- newTVarIO 0
  atomically $ modifyTVar' counter (+1)
  value <- atomically $ readTVar counter
  print value
```

### Composable Transactions

```haskell
import Control.Concurrent.STM

transfer :: TVar Int -> TVar Int -> Int -> STM ()
transfer from to amount = do
  balance <- readTVar from
  when (balance < amount) retry
  modifyTVar' from (subtract amount)
  modifyTVar' to (+ amount)
```

### orElse Composition

```haskell
import Control.Concurrent.STM

tryBoth :: STM a -> STM a -> STM a
tryBoth action1 action2 = action1 `orElse` action2
```

## FFI Exports

This crate exports C-ABI functions for BHC:

| Function | Description |
|----------|-------------|
| `bhc_scope_new` | Create new scope |
| `bhc_spawn` | Spawn task in scope |
| `bhc_await` | Wait for task result |
| `bhc_cancel` | Cancel task |
| `bhc_tvar_new` | Create new TVar |
| `bhc_tvar_read` | Read TVar value |
| `bhc_tvar_write` | Write TVar value |
| `bhc_atomically` | Run STM transaction |
| `bhc_chan_new` | Create channel |
| `bhc_chan_send` | Send to channel |
| `bhc_chan_recv` | Receive from channel |

## Scope Guarantees

Per H26-SPEC Section 10.1:

- Tasks cannot escape their scope
- Parent cancellation propagates to children
- All tasks complete before scope exits
- Resources are cleaned up on cancellation

## Design Notes

- Work-stealing scheduler for efficiency
- Lock-free STM implementation
- Cooperative cancellation at safe points
- Minimal overhead for task spawning

## Related Crates

- `bhc-rts-scheduler` - Runtime scheduler
- `bhc-system` - System IO
- `bhc-prelude` - Basic concurrency types

## Specification References

- H26-SPEC Section 10: Concurrency Model
- H26-SPEC Section 10.1: Structured Concurrency
- H26-SPEC Section 10.2: STM
