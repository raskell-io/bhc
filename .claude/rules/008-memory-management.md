# Memory Management Rules

**Rule ID:** BHC-RULE-008
**Applies to:** Runtime system, array/tensor operations, FFI

---

## Memory Regions

BHC defines three allocation regions per H26-SPEC Section 9:

### 1. Hot Arena

**Purpose:** Ephemeral allocations in numeric kernels

| Property | Requirement |
|----------|-------------|
| Allocation | Bump pointer (O(1)) |
| Deallocation | Bulk free at scope end |
| GC interaction | None |
| Lifetime | Scoped |

```haskell
-- Hot Arena usage
withArena :: (Arena -> IO a) -> IO a
withArena action = do
  arena <- allocateArena defaultArenaSize
  result <- action arena
  freeArena arena
  pure result

arenaAlloc :: Arena -> Int -> IO (Ptr a)
arenaAlloc arena size
  | fits arena size = bumpAlloc arena size
  | otherwise = error "Arena exhausted"
```

### 2. Pinned Heap

**Purpose:** Memory that must not move (FFI, device IO, DMA)

| Property | Requirement |
|----------|-------------|
| Allocation | malloc-style |
| Deallocation | Explicit or ref-counted |
| GC interaction | Never moved |
| Lifetime | Explicit |

```haskell
-- Pinned allocation
allocPinned :: Int -> IO (Ptr a)
allocPinned size = mallocBytes size

-- MUST be explicitly freed or ref-counted
freePinned :: Ptr a -> IO ()
freePinned = free
```

### 3. General Heap

**Purpose:** Normal boxed allocations, managed by GC

| Property | Requirement |
|----------|-------------|
| Allocation | GC-managed |
| Deallocation | Automatic |
| GC interaction | May be moved |
| Lifetime | GC-determined |

---

## Allocation Guidelines

### Hot Arena Usage

```haskell
-- Good: Temporary buffer in arena
computeKernel :: Tensor -> Tensor -> IO Tensor
computeKernel x y = withArena $ \arena -> do
  -- Temporary lives only in this scope
  tmp <- arenaAlloc arena (tensorSize x)
  computeIntermediate x tmp
  computeFinal tmp y

-- Bad: Escaping arena allocation
leakArena :: IO (Ptr a)  -- WRONG: ptr escapes arena scope
leakArena = withArena $ \arena ->
  arenaAlloc arena 1024  -- Returns dangling pointer!
```

### Pinned Memory Usage

```haskell
-- Good: Pinned for FFI
withPinnedTensor :: Tensor -> (Ptr Float -> IO a) -> IO a
withPinnedTensor t action = do
  ptr <- getPinnedPtr t  -- Guaranteed not to move
  action ptr

-- Good: Pinned for async IO
asyncWrite :: Handle -> PinnedByteString -> IO ()
asyncWrite h bs = withPinnedPtr bs $ \ptr len ->
  c_async_write h ptr len
```

### When to Use Each Region

| Scenario | Region |
|----------|--------|
| Loop temporary buffer | Hot Arena |
| Kernel scratch space | Hot Arena |
| FFI buffer | Pinned |
| DMA buffer | Pinned |
| GPU transfer | Pinned |
| Normal data structure | General |
| Long-lived tensor data | Pinned or General |

---

## Tensor Memory Layout

### Contiguous Layout

```
Tensor shape [2, 3]:

Logical:     Memory:
┌───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ →  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
├───┼───┼───┤    └───┴───┴───┴───┴───┴───┘
│ 3 │ 4 │ 5 │     Strides: [3, 1]
└───┴───┴───┘
```

### Strided Layout (Views)

```
Transposed view (no copy):

Original [2, 3]:  View [3, 2]:
┌───┬───┬───┐     ┌───┬───┐
│ 0 │ 1 │ 2 │     │ 0 │ 3 │  ← Same memory
├───┼───┼───┤     ├───┼───┤     Strides: [1, 3]
│ 3 │ 4 │ 5 │     │ 1 │ 4 │
└───┴───┴───┘     ├───┼───┤
                  │ 2 │ 5 │
                  └───┴───┘
```

### Alignment Requirements

```haskell
-- Tensor alignment requirements
tensorAlignment :: DType -> Int
tensorAlignment Float32 = 4
tensorAlignment Float64 = 8
tensorAlignment Vec4F32 = 16  -- SIMD
tensorAlignment Vec8F32 = 32  -- AVX

-- Allocate with alignment
allocTensor :: Shape -> DType -> IO Tensor
allocTensor shape dtype = do
  let size = product shape * dtypeSize dtype
      align = tensorAlignment dtype
  ptr <- allocAligned size align
  pure $ Tensor ptr shape strides
```

---

## Ownership and Lifetimes

### Ownership Model

```haskell
-- Tensor owns its data
data Tensor a = Tensor
  { tensorData   :: !(ForeignPtr a)  -- Owned
  , tensorShape  :: !Shape
  , tensorStride :: !Stride
  }

-- View borrows data from parent
data TensorView a = TensorView
  { viewParent :: !Tensor a          -- Borrowed
  , viewOffset :: !Int
  , viewShape  :: !Shape
  , viewStride :: !Stride
  }
```

### Lifetime Rules

1. **Views MUST NOT outlive their parent tensor**
2. **Arena allocations MUST NOT escape their scope**
3. **Pinned memory MUST be explicitly managed**

```haskell
-- Safe: view used within parent's lifetime
withView :: Tensor a -> (TensorView a -> b) -> b
withView tensor f =
  let view = slice [0..10] tensor
  in f view

-- Unsafe: view might escape
unsafeView :: Tensor a -> TensorView a  -- DANGER
unsafeView = slice [0..10]
```

---

## Garbage Collection Interaction

### GC-Safe Points

```haskell
-- Safe points where GC can run
gcSafePoint :: IO ()

-- Mark critical section (no GC)
withNoGC :: IO a -> IO a
withNoGC action = do
  disableGC
  result <- action
  enableGC
  pure result
```

### Pinned Objects

```haskell
-- Objects that must not move
newPinnedByteArray :: Int -> IO (MutableByteArray RealWorld)
newPinnedByteArray size = do
  mba <- newByteArray size
  pin mba  -- Mark as pinned
  pure mba

-- Check if pinned
isPinned :: ByteArray -> Bool
```

### GC Pressure Monitoring

```haskell
-- Track allocations
data AllocStats = AllocStats
  { allocArena   :: !Int
  , allocPinned  :: !Int
  , allocGeneral :: !Int
  }

getStats :: IO AllocStats
```

---

## Memory Safety

### Bounds Checking

```haskell
-- Safe indexing (checked)
(!) :: Tensor a -> Index -> a
tensor ! idx
  | inBounds tensor idx = unsafeIndex tensor idx
  | otherwise = error $ "Index out of bounds: " ++ show idx

-- Unsafe indexing (unchecked, for hot paths)
unsafeIndex :: Tensor a -> Index -> a
unsafeIndex tensor idx = ...  -- No bounds check

-- Compile-time bounds (when possible)
indexSafe :: KnownShape shape => Tensor shape a -> ValidIndex shape -> a
```

### Use-After-Free Prevention

```haskell
-- Arena-scoped buffer (can't escape)
newtype ScopedBuffer s a = ScopedBuffer (Ptr a)

allocScoped :: Arena s -> Int -> ST s (ScopedBuffer s a)
allocScoped arena size = ScopedBuffer <$> arenaAlloc arena size

-- Type system prevents escape
leakBuffer :: ST s (ScopedBuffer s a)  -- Type error: s escapes
leakBuffer = runST $ allocScoped arena 100
```

### Double-Free Prevention

```haskell
-- ForeignPtr handles cleanup automatically
newTensor :: Shape -> IO Tensor
newTensor shape = do
  ptr <- mallocBytes size
  fp <- newForeignPtr finalizerFree ptr
  pure $ Tensor fp shape strides

-- Explicit free with invalidation
data ManagedBuffer = ManagedBuffer
  { mbPtr   :: !(IORef (Maybe (Ptr a)))
  , mbSize  :: !Int
  }

freeBuffer :: ManagedBuffer -> IO ()
freeBuffer mb = do
  mptr <- readIORef (mbPtr mb)
  case mptr of
    Nothing -> pure ()  -- Already freed
    Just ptr -> do
      free ptr
      writeIORef (mbPtr mb) Nothing
```

---

## Memory Diagnostics

### Allocation Tracing

```haskell
-- Trace allocations in debug mode
traceAlloc :: String -> Int -> IO (Ptr a) -> IO (Ptr a)
traceAlloc label size action = do
  ptr <- action
  recordAllocation label size ptr
  pure ptr

-- Allocation report
printAllocReport :: IO ()
printAllocReport = do
  stats <- getStats
  putStrLn $ "Arena: " ++ show (allocArena stats)
  putStrLn $ "Pinned: " ++ show (allocPinned stats)
  putStrLn $ "General: " ++ show (allocGeneral stats)
```

### Memory Leak Detection

```haskell
-- Track outstanding allocations
data AllocTracker = AllocTracker
  { atAllocations :: !(IORef (Map (Ptr ()) AllocInfo))
  }

trackAlloc :: AllocTracker -> Ptr a -> AllocInfo -> IO ()
trackFree :: AllocTracker -> Ptr a -> IO ()

checkLeaks :: AllocTracker -> IO [AllocInfo]
checkLeaks tracker = do
  allocs <- readIORef (atAllocations tracker)
  pure $ Map.elems allocs
```

### Heap Profiling Hooks

```haskell
-- RTS hooks for profiling
foreign export ccall "bhc_heap_alloc"
  heapAllocHook :: Word -> Ptr () -> IO ()

foreign export ccall "bhc_heap_free"
  heapFreeHook :: Ptr () -> IO ()
```

---

## Best Practices

### Do

- Use Hot Arena for loop temporaries
- Pin memory that crosses FFI boundary
- Free pinned memory explicitly
- Validate arena size before kernel
- Track allocation statistics in debug mode

### Don't

- Let arena allocations escape scope
- Forget to pin FFI buffers
- Allocate in tight loops without arena
- Mix pinned/unpinned in same data structure
- Ignore memory pressure warnings
