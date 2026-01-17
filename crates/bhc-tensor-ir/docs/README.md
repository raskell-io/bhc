# bhc-tensor-ir

Tensor Intermediate Representation for numeric optimization in BHC.

## Overview

Tensor IR is the key to BHC's numeric performance. It captures:

- **Shape and stride information**: Layout-aware optimization
- **Element types (dtypes)**: Unboxed numeric computation
- **Operation structure**: Fusion analysis
- **Aliasing information**: Safe in-place updates

## H26-SPEC Section 7 Compliance

Per specification, every tensor operation tracks:

| Property | Description |
|----------|-------------|
| `dtype` | Element type (Float32, Float64, Int32, etc.) |
| `shape` | Dimension sizes |
| `strides` | Byte strides per dimension |
| `layout` | Memory layout (contiguous, strided, tiled) |
| `alias` | Aliasing/ownership information |

## IR Pipeline Position

```
[Core IR]     ← General purpose optimizations
    ↓
    | (Numeric Profile only)
    ↓
[Tensor IR]   ← This crate: shape-aware, fusion-ready
    ↓
[Loop IR]     ← Explicit iteration
```

## Core Types

| Type | Description |
|------|-------------|
| `TensorOp` | Tensor operations |
| `TensorMeta` | Metadata (shape, stride, dtype) |
| `Kernel` | Fused computation unit |
| `Shape` | Tensor dimensions |
| `DType` | Element types |
| `Strides` | Memory strides |

## Data Types

```rust
pub enum DType {
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64, BFloat16,
    Complex64, Complex128,
}

impl DType {
    pub fn size_bytes(self) -> usize;
    pub fn alignment(self) -> usize;
    pub fn is_float(self) -> bool;
    pub fn is_integer(self) -> bool;
    pub fn is_signed(self) -> bool;
}
```

## Shapes and Strides

```rust
// Dimensions can be static or dynamic
pub enum Dim {
    Static(usize),
    Dynamic(Symbol),
}

// Shape is a list of dimensions
pub struct Shape(SmallVec<[Dim; 4]>);

impl Shape {
    pub fn from_static(dims: impl IntoIterator<Item = usize>) -> Self;
    pub fn scalar() -> Self;
    pub fn rank(&self) -> usize;
    pub fn num_elements(&self) -> Option<usize>;
    pub fn is_static(&self) -> bool;
}

// Memory strides
pub struct Strides(SmallVec<[i64; 4]>);

impl Strides {
    pub fn contiguous(shape: &Shape, elem_size: usize) -> Option<Self>;
    pub fn is_contiguous(&self, shape: &Shape, elem_size: usize) -> bool;
}
```

## Tensor Metadata

```rust
pub struct TensorMeta {
    pub dtype: DType,
    pub shape: Shape,
    pub strides: Strides,
    pub layout: Layout,
    pub alias: Option<BufferId>,
}

pub enum Layout {
    Contiguous,
    Strided,
    Tiled(TileInfo),
}
```

## Tensor Operations

### Elementwise

```rust
// Unary operations
TensorOp::Unary(UnaryOp::Neg, tensor)
TensorOp::Unary(UnaryOp::Sqrt, tensor)
TensorOp::Unary(UnaryOp::Exp, tensor)

// Binary operations
TensorOp::Binary(BinaryOp::Add, a, b)
TensorOp::Binary(BinaryOp::Mul, a, b)

// Map and ZipWith
TensorOp::Map(map_fn, tensor)
TensorOp::ZipWith(zip_fn, a, b)
```

### Reductions

```rust
// Reduce along axis
TensorOp::Reduce(ReduceOp::Sum, Axis::new(0), tensor)
TensorOp::Reduce(ReduceOp::Max, Axis::new(-1), tensor)

// Full reduction
TensorOp::ReduceAll(ReduceOp::Sum, tensor)

// Scan (prefix sum)
TensorOp::Scan(ReduceOp::Add, Axis::new(0), tensor)

// Fold with initial value
TensorOp::Fold(fold_fn, init, tensor)
```

### Structure Operations

```rust
// Reshape
TensorOp::Reshape(new_shape, tensor)

// Slice
TensorOp::Slice(slice_spec, tensor)

// Transpose
TensorOp::Transpose(Permutation::new([1, 0]), tensor)

// Broadcast
TensorOp::Broadcast(target_shape, tensor)

// Concatenate
TensorOp::Concat(Axis::new(0), vec![a, b, c])
```

### Linear Algebra

```rust
// Matrix multiplication
TensorOp::MatMul(a, b)

// Batched matmul
TensorOp::BatchMatMul(a, b)

// Dot product
TensorOp::Dot(a, b)

// Outer product
TensorOp::Outer(a, b)
```

### Convolution

```rust
TensorOp::Conv(ConvSpec {
    padding: smallvec![(1, 1), (1, 1)],
    strides: smallvec![1, 1],
    dilation: smallvec![1, 1],
    groups: 1,
}, input, kernel)
```

## Guaranteed Fusion

Per H26-SPEC Section 8, these patterns MUST fuse:

```rust
// 1. map f (map g x) → single traversal
// 2. zipWith f (map g a) (map h b) → single traversal
// 3. sum (map f x) → single traversal
// 4. foldl' op z (map f x) → single traversal
```

## Kernels

Kernels are the output of the fusion pass:

```rust
pub struct Kernel {
    pub id: KernelId,
    pub name: Symbol,
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<TensorRef>,
    pub body: KernelBody,
    pub allocs: Vec<AllocInfo>,
    pub fusion_info: FusionInfo,
}

pub enum KernelBody {
    Fused(Vec<TensorOp>),
    LoopNest(LoopNest),
}
```

### Fusion Information

```rust
pub struct FusionInfo {
    pub original_ops: Vec<Symbol>,
    pub decisions: Vec<FusionDecision>,
    pub complete: bool,
}

pub enum FusionDecision {
    Fused(Vec<Symbol>),
    Materialized(Symbol, MaterializeReason),
    Blocked(Symbol, FusionBlockReason),
}

pub enum MaterializeReason {
    MultipleUses,
    Explicit,
    ControlFlow,
}

pub enum FusionBlockReason {
    ShapeMismatch,
    DTypeMismatch,
    DataDependency,
    SideEffects,
}
```

## Memory Allocation

```rust
pub struct AllocInfo {
    pub buffer: BufferId,
    pub size: usize,
    pub alignment: usize,
    pub region: AllocRegion,
}

pub enum AllocRegion {
    /// Hot arena (bump allocated)
    HotArena,
    /// Pinned heap (for FFI)
    Pinned,
    /// GC-managed heap
    General,
    /// GPU device memory
    DeviceMemory(DeviceTarget),
}

pub enum DeviceTarget {
    Cuda(u32),
    Rocm(u32),
    Any,
}
```

## Submodules

### fusion - Fusion Analysis

```rust
use bhc_tensor_ir::fusion;

// Run fusion pass
let kernels = fusion::fuse_operations(&ops)?;

// Check fusion success
for kernel in &kernels {
    if !kernel.fusion_info.complete {
        // Report missed fusion
    }
}
```

### lower - Lowering to Loop IR

```rust
use bhc_tensor_ir::lower;

// Lower tensor ops to loop IR
let loop_ir = lower::lower_to_loops(&kernel)?;
```

## Operations Reference

### Unary Operations

| Op | Description |
|----|-------------|
| `Neg` | Negation |
| `Abs` | Absolute value |
| `Sqrt`, `Rsqrt` | Square root, reciprocal sqrt |
| `Exp`, `Log` | Exponential, logarithm |
| `Sin`, `Cos`, `Tan` | Trigonometric |
| `Tanh`, `Sigmoid`, `Relu` | Activations |
| `Ceil`, `Floor`, `Round` | Rounding |
| `Not` | Bitwise not |

### Binary Operations

| Op | Description |
|----|-------------|
| `Add`, `Sub`, `Mul`, `Div` | Arithmetic |
| `Mod`, `Pow` | Modulo, power |
| `Max`, `Min` | Element-wise min/max |
| `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge` | Comparisons |
| `And`, `Or`, `Xor` | Bitwise |
| `Shl`, `Shr` | Shifts |

### Reduce Operations

| Op | Description |
|----|-------------|
| `Sum` | Sum reduction |
| `Prod` | Product reduction |
| `Max`, `Min` | Max/min reduction |
| `All`, `Any` | Logical reductions |
| `Mean` | Mean (sum/count) |

## Error Handling

```rust
pub enum TensorIrError {
    ShapeMismatch { expected: Shape, got: Shape },
    InvalidAxis { axis: i32, rank: usize },
    DTypeMismatch { expected: DType, got: DType },
    FusionFailed { pattern: String },
}
```

## Example: Matrix Multiplication

```rust
// matmul: Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a

let a_meta = TensorMeta::new_contiguous(
    DType::Float32,
    Shape::from_static([1024, 768])
)?;

let b_meta = TensorMeta::new_contiguous(
    DType::Float32,
    Shape::from_static([768, 512])
)?;

let a_ref = TensorRef { id: TensorId::new(0), meta: a_meta };
let b_ref = TensorRef { id: TensorId::new(1), meta: b_meta };

let matmul_op = TensorOp::MatMul(a_ref, b_ref);
// Result shape: [1024, 512]
```

## See Also

- `bhc-core`: Core IR that lowers to Tensor IR
- `bhc-loop-ir`: Loop IR for explicit iteration
- H26-SPEC Section 7: Tensor Model
- H26-SPEC Section 8: Fusion Laws
