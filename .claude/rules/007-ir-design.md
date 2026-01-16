# IR Design Principles

**Rule ID:** BHC-RULE-007
**Applies to:** All intermediate representations (Core IR, Tensor IR, Loop IR)

---

## IR Pipeline Overview

```
Source Code
    │
    ▼
┌─────────────┐
│  Parse/AST  │  Concrete syntax
└─────────────┘
    │
    ▼
┌─────────────┐
│  Core IR    │  Typed, desugared, explicit
└─────────────┘
    │ (Numeric Profile)
    ▼
┌─────────────┐
│  Tensor IR  │  Shape/stride aware
└─────────────┘
    │
    ▼
┌─────────────┐
│  Loop IR    │  Explicit iteration
└─────────────┘
    │
    ▼
┌─────────────┐
│  Codegen    │  LLVM / Native
└─────────────┘
```

---

## Core IR Principles

### 1. Explicit Over Implicit

Core IR MUST make all implicit operations explicit:

```haskell
-- Source (implicit)
f x y = x + y

-- Core IR (explicit)
f = Lam x (Lam y (
      App (App (Var (+)) (Var x)) (Var y)
    ))
```

### 2. Typed Representation

Core IR MUST preserve types:

```haskell
data Expr t where
  Var  :: Name -> Type t -> Expr t
  Lit  :: Literal t -> Expr t
  App  :: Expr (a -> b) -> Expr a -> Expr b
  Lam  :: Name -> Type a -> Expr b -> Expr (a -> b)
  Let  :: Name -> Expr a -> Expr b -> Expr b
  Case :: Expr a -> [Alt a b] -> Expr b
```

### 3. A-Normal Form (ANF)

Complex expressions SHOULD be in A-normal form:

```haskell
-- Not ANF: nested applications
App (App f (App g x)) (App h y)

-- ANF: all arguments are atomic
Let tmp1 (App g x)
Let tmp2 (App h y)
Let tmp3 (App f tmp1)
App tmp3 tmp2
```

### 4. Preserving Source Information

Core IR MUST preserve source locations for error reporting:

```haskell
data Expr t = Expr
  { exprNode :: ExprNode t
  , exprLoc  :: SrcSpan
  , exprType :: Type t
  }
```

---

## Tensor IR Requirements

### H26-SPEC Section 7.3 Compliance

Tensor IR MUST track:

| Property | Type | Description |
|----------|------|-------------|
| `dtype` | `DType` | Element type |
| `shape` | `[Dim]` | Dimension sizes |
| `strides` | `[Stride]` | Byte strides |
| `layout` | `Layout` | Memory layout |
| `alias` | `Maybe BufferId` | Aliasing info |

```haskell
data TensorMeta = TensorMeta
  { tmDtype   :: !DType
  , tmShape   :: ![Dim]
  , tmStrides :: ![Stride]
  , tmLayout  :: !Layout
  , tmAlias   :: !(Maybe BufferId)
  }

data Layout
  = Contiguous
  | Strided
  | Tiled TileInfo
  deriving (Eq, Show)
```

### Tensor Operations

```haskell
data TensorOp
  -- Elementwise
  = TMap (Expr (a -> b)) TensorRef
  | TZipWith (Expr (a -> b -> c)) TensorRef TensorRef
  | TBroadcast Shape TensorRef

  -- Reductions
  | TReduce ReduceOp Axis TensorRef
  | TFold (Expr (a -> b -> a)) (Expr a) TensorRef

  -- Structure
  | TReshape Shape TensorRef
  | TSlice SliceSpec TensorRef
  | TTranspose Permutation TensorRef
  | TConcat Axis [TensorRef]

  -- Linear Algebra
  | TMatMul TensorRef TensorRef
  | TDot TensorRef TensorRef
  | TConv ConvSpec TensorRef TensorRef
```

### Shape Inference

Tensor IR MUST infer shapes statically when possible:

```haskell
-- Shape inference rules
inferShape :: TensorOp -> Either ShapeError Shape

inferShape (TMap _ t) = getShape t  -- Preserves shape

inferShape (TMatMul a b) = do
  [m, k1] <- getShape a
  [k2, n] <- getShape b
  when (k1 /= k2) $ Left $ DimMismatch k1 k2
  pure [m, n]

inferShape (TReshape newShape t) = do
  oldShape <- getShape t
  when (product oldShape /= product newShape) $
    Left $ ElementCountMismatch
  pure newShape
```

---

## Loop IR Design

### Explicit Iteration

Loop IR makes iteration structure explicit:

```haskell
data LoopIR
  = Loop LoopVar Bound Bound LoopBody
  | Parallel LoopVar Bound Bound ChunkSize LoopBody
  | Nested [LoopIR]
  | Statement Stmt

data LoopBody
  = Body [Stmt]
  | Tiled TileSize LoopBody
  | Vectorized VectorWidth LoopBody
```

### Loop Metadata

```haskell
data LoopMeta = LoopMeta
  { lmTrip      :: TripCount        -- Iteration count
  , lmParallel  :: Bool             -- Can parallelize?
  , lmVectorize :: Maybe VecWidth   -- Can vectorize?
  , lmTile      :: Maybe TileSpec   -- Tiling info
  , lmDeps      :: [LoopDep]        -- Dependencies
  }
```

### Polyhedral Representation (Optional)

For advanced loop optimizations:

```haskell
data PolyLoop = PolyLoop
  { plDomain     :: Set              -- Iteration domain
  , plSchedule   :: Schedule         -- Execution order
  , plAccessMaps :: [AccessRelation] -- Memory accesses
  }
```

---

## IR Transformation Guidelines

### 1. Sound Transformations

All transformations MUST preserve semantics:

```haskell
-- Transformation contract
transform :: IR -> IR
-- Invariant: forall program.
--   eval (transform program) == eval program
```

### 2. Documented Preconditions

Transformations MUST document preconditions:

```haskell
-- | Fuse consecutive maps into single traversal.
--
-- ==== __Preconditions__
--
-- * Input must be in ANF
-- * Both maps must have same iteration space
-- * No side effects in map functions
--
-- ==== __Postconditions__
--
-- * Output is single map with composed function
-- * No intermediate allocation
fuseMapMap :: TensorIR -> TensorIR
```

### 3. Verification Hooks

Include verification in debug mode:

```haskell
transform :: IR -> IR
transform ir =
  let result = transformImpl ir
  in assert (verify ir result) result

verify :: IR -> IR -> Bool
verify before after =
  typeCheckIR after &&
  shapeCheck after &&
  aliasCheck after
```

---

## IR Pretty Printing

### Requirements

- MUST be readable by humans
- MUST be parseable (for debugging)
- SHOULD show metadata inline

### Format

```
-- Core IR example
let %0 : Tensor [1024] Float = tensor.zeros [1024]
let %1 : Tensor [1024] Float = tensor.map (+1) %0
let %2 : Float = tensor.sum %1
return %2

-- Tensor IR example (with metadata)
kernel @k0 {
  input %x : Tensor [M, K] Float (contiguous)
  input %y : Tensor [K, N] Float (contiguous)
  output %z : Tensor [M, N] Float (contiguous)

  for i in 0..M parallel {
    for j in 0..N vectorize(8) {
      %acc = 0.0
      for k in 0..K {
        %acc += %x[i, k] * %y[k, j]
      }
      %z[i, j] = %acc
    }
  }
}
```

---

## Kernel Boundaries

### Defining Kernels

A kernel is a unit of computation that:
- Has defined inputs and outputs
- Executes without GC
- Has explicit memory allocation pattern

```haskell
data Kernel = Kernel
  { kName    :: KernelName
  , kInputs  :: [TensorDecl]
  , kOutputs :: [TensorDecl]
  , kBody    :: LoopNest
  , kAllocs  :: [AllocDecl]  -- Arena allocations
  }
```

### Kernel Fusion Decisions

Track fusion decisions:

```haskell
data FusionDecision
  = Fused [KernelId]         -- Kernels were fused
  | Materialized KernelId    -- Explicit materialization point
  | FusionBlocked Reason     -- Could not fuse, with reason

data Reason
  = MultipleUses             -- Result used multiple times
  | SideEffects              -- IO or mutable state
  | ShapeMismatch            -- Incompatible iteration spaces
  | UserRequested            -- materialize called
```

---

## IR Invariants

### Core IR Invariants

1. All variables are bound before use
2. Types are consistent
3. No duplicate names in scope
4. Source locations are valid

### Tensor IR Invariants

1. Shapes are consistent through operations
2. Strides are valid for the shape
3. Aliasing information is accurate
4. Fusion decisions are recorded

### Loop IR Invariants

1. Loop bounds are non-negative
2. Parallel loops have no dependencies
3. Vectorized loops have aligned access
4. Memory accesses are in bounds

---

## Debugging Support

### IR Dumps

```bash
# Dump all IRs
bhc -ddump-core -ddump-tensor -ddump-loop input.hs

# Dump specific passes
bhc -ddump-core-after-strictness input.hs
bhc -ddump-tensor-after-fusion input.hs
```

### IR Validation

```haskell
-- Run validation between passes
validateIR :: IR -> Either ValidationError ()

-- In pipeline
runPass :: Pass -> IR -> IO IR
runPass pass ir = do
  result <- applyPass pass ir
  when debugMode $ do
    case validateIR result of
      Left err -> fail $ "IR validation failed after " ++ passName pass
      Right () -> pure ()
  pure result
```

### Tracing

```haskell
-- Trace IR transformations
traceTransform :: String -> IR -> IR -> IO ()
traceTransform name before after = do
  logDebug $ "Transform: " ++ name
  logDebug $ "Before:\n" ++ prettyIR before
  logDebug $ "After:\n" ++ prettyIR after
  logDebug $ "Changed: " ++ show (before /= after)
```
