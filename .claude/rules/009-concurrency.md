# Concurrency Guidelines

**Rule ID:** BHC-RULE-009
**Applies to:** Runtime system, parallel numeric operations, server workloads

---

## Structured Concurrency Model

BHC implements structured concurrency per H26-SPEC Section 10.

### Core Principle

**All concurrent operations happen within a scope that outlives them.**

```haskell
-- All tasks complete or cancel before scope exits
withScope :: (Scope -> IO a) -> IO a
withScope action = do
  scope <- newScope
  result <- action scope `onException` cancelAll scope
  waitAll scope
  pure result
```

### Task Lifecycle

```
spawn      await
  │          │
  ▼          ▼
┌────┐    ┌─────┐    ┌──────────┐    ┌───────┐
│ New│ → │Running│ → │Completing│ → │Completed│
└────┘    └─────┘    └──────────┘    └───────┘
             │                            ▲
             │ cancel                     │
             ▼                            │
          ┌──────────┐                   │
          │Cancelling│ ──────────────────┘
          └──────────┘
```

---

## Task API

### Spawning Tasks

```haskell
-- Spawn within a scope
spawn :: Scope -> IO a -> IO (Task a)
spawn scope action = do
  validateScope scope
  task <- newTask
  forkWithin scope $ do
    result <- action
    completeTask task result
  pure task

-- Usage
example :: IO (Int, Int)
example = withScope $ \scope -> do
  t1 <- spawn scope computeX
  t2 <- spawn scope computeY
  x <- await t1
  y <- await t2
  pure (x, y)
```

### Awaiting Results

```haskell
-- Wait for task completion
await :: Task a -> IO a
await task = do
  checkCancelled  -- Throw if current task cancelled
  blockUntilComplete task
  getResult task

-- Non-blocking check
poll :: Task a -> IO (Maybe a)
poll task = do
  status <- taskStatus task
  case status of
    Completed result -> pure (Just result)
    _                -> pure Nothing
```

### Cancellation

```haskell
-- Cancel a task and its subtasks
cancel :: Task a -> IO ()
cancel task = do
  markCancelled task
  propagateCancel (subtasks task)
  -- Task will terminate at next safe point

-- Check for cancellation
checkCancelled :: IO ()
checkCancelled = do
  task <- currentTask
  when (isCancelled task) $
    throwIO TaskCancelled
```

---

## Cancellation Semantics

### Cooperative Cancellation

Cancellation is **cooperative** — tasks check for cancellation at safe points:

```haskell
-- Safe points (where cancellation is checked)
-- - Between IO actions
-- - At explicit checkpoints
-- - Before blocking operations

-- Automatic safe points
readFile path  -- Safe point before blocking read

-- Explicit checkpoint
checkpoint :: IO ()
checkpoint = checkCancelled

-- Long computation should checkpoint
longComputation :: [Item] -> IO Result
longComputation items = do
  forM items $ \item -> do
    checkpoint  -- Allow cancellation between items
    processItem item
```

### Cancellation Propagation

```haskell
-- Parent cancellation propagates to children
withScope $ \parent -> do
  t1 <- spawn parent $ do
    withScope $ \child -> do
      t2 <- spawn child innerTask
      await t2

-- If parent scope is cancelled:
-- 1. t1 is marked cancelled
-- 2. child scope is cancelled
-- 3. t2 is marked cancelled
-- 4. All propagate to completion
```

### Cleanup on Cancellation

```haskell
-- Cleanup handlers run on cancellation
onCancel :: IO a -> IO () -> IO a
onCancel action cleanup =
  action `catch` \TaskCancelled -> do
    cleanup
    throwIO TaskCancelled

-- Using bracket
withResource :: IO r -> (r -> IO ()) -> (r -> IO a) -> IO a
withResource acquire release use =
  bracket acquire release use
  -- release runs even on cancellation
```

---

## Deadlines and Timeouts

### Deadlines

```haskell
-- Run action with deadline
withDeadline :: Duration -> IO a -> IO (Maybe a)
withDeadline duration action = do
  deadline <- addTime <$> getCurrentTime <*> pure duration
  withScope $ \scope -> do
    task <- spawn scope action
    timer <- spawn scope (waitUntil deadline >> cancel task)
    result <- try (await task)
    cancel timer
    case result of
      Right a -> pure (Just a)
      Left TaskCancelled -> pure Nothing

-- Usage
result <- withDeadline (seconds 5) longOperation
case result of
  Just x  -> useResult x
  Nothing -> handleTimeout
```

### Inherited Deadlines

```haskell
-- Deadline propagates to child tasks
withDeadline (seconds 10) $ withScope $ \scope -> do
  t1 <- spawn scope task1  -- Inherits 10s deadline
  t2 <- spawn scope task2  -- Inherits 10s deadline
  ...
```

---

## Parallel Numeric Operations

### Parallel Map

```haskell
-- Parallel map over tensor
parMap :: (a -> b) -> Tensor a -> Tensor b
parMap f tensor = unsafePerformIO $
  withScope $ \scope -> do
    let chunks = chunkTensor tensor
    tasks <- forM chunks $ \chunk ->
      spawn scope $ pure $ map f chunk
    results <- mapM await tasks
    pure $ concatChunks results
```

### Parallel Reduce

```haskell
-- Parallel reduction
parReduce :: Monoid m => (a -> m) -> Tensor a -> m
parReduce f tensor = unsafePerformIO $
  withScope $ \scope -> do
    let chunks = chunkTensor tensor
    tasks <- forM chunks $ \chunk ->
      spawn scope $ pure $ foldMap f chunk
    partials <- mapM await tasks
    pure $ mconcat partials
```

### Parallel For

```haskell
-- Parallel loop
parFor :: Range -> (Int -> IO ()) -> IO ()
parFor (lo, hi) body = withScope $ \scope -> do
  let chunks = chunkRange (lo, hi) workerCount
  tasks <- forM chunks $ \(clo, chi) ->
    spawn scope $ forM_ [clo..chi-1] body
  mapM_ await tasks
```

---

## Determinism

### Deterministic Mode

For reproducible results:

```haskell
-- Deterministic parallel execution
parMapDeterministic :: (a -> b) -> Tensor a -> Tensor b
parMapDeterministic f tensor =
  runDeterministic $ parMap f tensor

-- Implementation ensures:
-- 1. Fixed chunk sizes
-- 2. Fixed worker assignment
-- 3. Deterministic reduction order
```

### Non-Deterministic Float Warning

```haskell
-- Float operations may vary due to parallel ordering
parSum :: Tensor Float -> Float  -- MAY vary between runs

-- Document non-determinism
-- | Parallel sum of floating point values.
--
-- __Warning__: Results may vary slightly between runs due to
-- parallel reduction ordering. For bit-exact results, use
-- 'parSumDeterministic'.
parSum :: Tensor Float -> Float
```

---

## Atomics and Memory Ordering

### Atomic Types

```haskell
-- Atomic integer
data AtomicInt = AtomicInt (MutableByteArray RealWorld)

atomicRead :: AtomicInt -> IO Int
atomicWrite :: AtomicInt -> Int -> IO ()
atomicAdd :: AtomicInt -> Int -> IO Int  -- Returns old value
atomicCAS :: AtomicInt -> Int -> Int -> IO Bool
```

### Memory Ordering

```haskell
data MemoryOrder
  = Relaxed     -- No ordering guarantees
  | Acquire     -- Reads can't move before
  | Release     -- Writes can't move after
  | AcqRel      -- Both acquire and release
  | SeqCst      -- Sequentially consistent

atomicReadOrdered :: MemoryOrder -> AtomicInt -> IO Int
atomicWriteOrdered :: MemoryOrder -> AtomicInt -> Int -> IO ()
```

### Usage Guidelines

```haskell
-- Good: Clear ordering semantics
counter <- newAtomicInt 0
atomicWriteOrdered Release counter 42
-- All previous writes visible to acquirer

-- Bad: Unspecified ordering
atomicWrite counter 42  -- What ordering?
```

---

## Scheduler Configuration

### Work Stealing

```haskell
-- Scheduler uses work-stealing deques
data Scheduler = Scheduler
  { schedWorkers :: ![Worker]
  , schedGlobal  :: !(Deque Task)
  }

-- Worker tries local queue first, then steals
runWorker :: Worker -> IO ()
runWorker worker = do
  mtask <- popLocal (workerDeque worker)
  case mtask of
    Just task -> executeTask task
    Nothing -> do
      mtask' <- stealFrom (otherWorkers worker)
      case mtask' of
        Just task -> executeTask task
        Nothing -> yield  -- No work available
```

### Worker Count

```haskell
-- Configure worker count
setWorkerCount :: Int -> IO ()
setWorkerCount n = modifyScheduler $ \s ->
  s { schedWorkers = take n (schedWorkers s ++ newWorkers) }

-- Auto-detect from CPU count
autoWorkerCount :: IO Int
autoWorkerCount = getNumCapabilities
```

---

## Best Practices

### Do

- Use `withScope` for all concurrent operations
- Check cancellation in long-running loops
- Use deadlines for external operations
- Document non-determinism in parallel operations
- Use explicit memory ordering for atomics

### Don't

- Let tasks escape their scope
- Ignore cancellation requests
- Use busy-waiting (use proper synchronization)
- Share mutable state between parallel tasks without synchronization
- Assume parallel execution order

### Error Handling

```haskell
-- Errors in tasks propagate to awaiter
handleErrors :: IO ()
handleErrors = withScope $ \scope -> do
  t1 <- spawn scope (throwIO SomeError)
  t2 <- spawn scope (pure 42)
  -- await t1 will rethrow SomeError
  result <- try (await t1)
  case result of
    Left err -> handleError err
    Right x  -> useResult x
```

### Resource Cleanup

```haskell
-- Resources are cleaned up on scope exit
withResources :: IO ()
withResources = withScope $ \scope -> do
  file <- openFile "data.txt" ReadMode
  finally (process scope file) (hClose file)
  -- file is closed even if scope is cancelled
```
