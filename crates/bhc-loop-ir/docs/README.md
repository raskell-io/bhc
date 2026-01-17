# bhc-loop-ir

Loop Intermediate Representation for low-level optimization in BHC.

## Overview

Loop IR is the lowest-level IR before code generation. It provides:

- **Explicit iteration**: Loops with bounds and strides
- **Vectorization information**: SIMD-izable loops
- **Parallelization hints**: Parallel execution markers
- **Memory access patterns**: Cache optimization

## IR Pipeline Position

```
[Tensor IR]  ← High-level tensor operations
    ↓
[Loop IR]    ← This crate: explicit iteration
    ↓
[Codegen]    ← LLVM IR / Native code
```

## Core Types

| Type | Description |
|------|-------------|
| `LoopIR` | Top-level IR structure |
| `Loop` | Single loop with bounds |
| `Stmt` | Statements within loops |
| `Value` | SSA values (registers) |
| `MemRef` | Memory references |
| `LoopType` | Types including SIMD vectors |

## M3 Deliverables

Loop IR implements these M3 features:

- **SIMD Types**: `VEC4F32`, `VEC8F32`, `VEC2F64`, `VEC4F64`
- **Auto-vectorization**: `VectorizePass`
- **Parallel primitives**: `ParFor`, `ParMap`, `ParReduce`
- **SIMD intrinsics**: `SimdIntrinsic`

## Loop IR Structure

```rust
pub struct LoopIR {
    /// Function name
    pub name: Symbol,
    /// Parameters
    pub params: Vec<Param>,
    /// Return type
    pub return_ty: LoopType,
    /// Body
    pub body: Body,
    /// Memory allocations
    pub allocs: Vec<Alloc>,
    /// Loop metadata
    pub loop_info: Vec<LoopMetadata>,
}
```

## Types

### Scalar Types

```rust
pub enum ScalarType {
    Bool,
    Int(u8),    // Int with bit width
    UInt(u8),   // Unsigned int
    Float(u8),  // Float with bit width
}

// Constants
ScalarType::F32  // Float(32)
ScalarType::F64  // Float(64)
ScalarType::I32  // Int(32)
ScalarType::I64  // Int(64)
```

### Loop Types

```rust
pub enum LoopType {
    Void,
    Scalar(ScalarType),
    Vector(ScalarType, u8),  // SIMD vector
    Ptr(Box<LoopType>),
}

// SIMD type constants
LoopType::VEC4F32  // 4x f32 (128-bit SSE/NEON)
LoopType::VEC8F32  // 8x f32 (256-bit AVX)
LoopType::VEC2F64  // 2x f64 (128-bit)
LoopType::VEC4F64  // 4x f64 (256-bit)
LoopType::VEC4I32  // 4x i32 (128-bit)
LoopType::VEC8I32  // 8x i32 (256-bit)
```

### Target Architecture

```rust
pub enum TargetArch {
    X86_64Sse,    // 128-bit
    X86_64Sse2,   // 128-bit
    X86_64Avx,    // 256-bit
    X86_64Avx2,   // 256-bit
    Aarch64Neon,  // 128-bit
    Generic,      // No vectorization
}

// Natural vector width per target
LoopType::natural_vector_width(ScalarType::F32, TargetArch::X86_64Avx2)
// Returns 8 (256-bit / 32-bit = 8 elements)
```

## Loops

```rust
pub struct Loop {
    pub id: LoopId,
    pub var: ValueId,       // Loop variable
    pub lower: Value,       // Lower bound (inclusive)
    pub upper: Value,       // Upper bound (exclusive)
    pub step: Value,        // Step size
    pub body: Body,
    pub attrs: LoopAttrs,
}

bitflags! {
    pub struct LoopAttrs: u32 {
        const PARALLEL = 0b0000_0001;
        const VECTORIZE = 0b0000_0010;
        const UNROLL = 0b0000_0100;
        const REDUCTION = 0b0000_1000;
        const INDEPENDENT = 0b0001_0000;
        const TILED = 0b0010_0000;
        const TILE_INNER = 0b0100_0000;
    }
}
```

### Loop Metadata

```rust
pub struct LoopMetadata {
    pub id: LoopId,
    pub trip_count: TripCount,
    pub vector_width: Option<u8>,
    pub parallel_chunk: Option<usize>,
    pub unroll_factor: Option<u8>,
    pub dependencies: Vec<LoopDependency>,
}

pub enum TripCount {
    Static(usize),
    Dynamic,
    Bounded(usize),
}
```

## Statements

```rust
pub enum Stmt {
    /// Assignment: `%v = op`
    Assign(ValueId, Op),
    /// Loop construct
    Loop(Loop),
    /// Conditional
    If(IfStmt),
    /// Memory store
    Store(MemRef, Value),
    /// Function call
    Call(Option<ValueId>, Symbol, Vec<Value>),
    /// Return
    Return(Option<Value>),
    /// Synchronization barrier
    Barrier(BarrierKind),
    /// Comment/annotation
    Comment(String),
}
```

## Operations

```rust
pub enum Op {
    // Memory
    Load(MemRef),

    // Arithmetic
    Binary(BinOp, Value, Value),
    Unary(UnOp, Value),
    Cmp(CmpOp, Value, Value),

    // Control
    Select(Value, Value, Value),
    Cast(Value, LoopType),

    // Vector operations
    Broadcast(Value, u8),      // Scalar to vector
    Extract(Value, u8),        // Vector to scalar
    Insert(Value, Value, u8),
    Shuffle(Value, Value, Vec<i32>),
    VecReduce(ReduceOp, Value),

    // FMA
    Fma(Value, Value, Value),  // a * b + c

    // Pointer
    PtrAdd(Value, Value),
    GetPtr(BufferId, Value),

    // SSA
    Phi(Vec<(BlockId, Value)>),
}
```

### Binary Operations

```rust
pub enum BinOp {
    // Arithmetic
    Add, Sub, Mul,
    SDiv, UDiv, FDiv,
    SRem, URem, FRem,

    // Bitwise
    And, Or, Xor,
    Shl, LShr, AShr,

    // Min/Max
    SMin, UMin, FMin,
    SMax, UMax, FMax,
}
```

### Unary Operations

```rust
pub enum UnOp {
    Neg, FNeg, Not,
    Abs, FAbs,
    Sqrt, Rsqrt,
    Floor, Ceil, Round, Trunc,
    Exp, Log, Sin, Cos,
}
```

## Memory Access

```rust
pub struct MemRef {
    pub buffer: BufferId,
    pub index: Value,
    pub elem_ty: LoopType,
    pub access: AccessPattern,
}

pub enum AccessPattern {
    Sequential,           // Stride 1
    Strided(i64),        // Fixed stride
    Random,              // Indirect
    Broadcast,           // Same element
    Affine(AffineAccess), // Linear combination
}

pub struct AffineAccess {
    pub coefficients: SmallVec<[(LoopId, i64); 4]>,
    pub offset: i64,
}
```

## Vectorization

```rust
use bhc_loop_ir::vectorize::{VectorizePass, VectorizeConfig};

let config = VectorizeConfig {
    target: TargetArch::X86_64Avx2,
    enable_fma: true,
    ..Default::default()
};

let mut pass = VectorizePass::new(config);
let analysis = pass.analyze(&loop_ir);

for (loop_id, info) in analysis {
    if info.vectorizable {
        println!("Loop {} can vectorize with width {}",
            loop_id, info.recommended_width);
    }
}
```

### SIMD Intrinsics

```rust
pub enum SimdIntrinsic {
    // Arithmetic
    VAdd, VSub, VMul, VDiv,
    VFma,  // Fused multiply-add

    // Comparison
    VCmpEq, VCmpLt, VCmpLe,

    // Shuffles
    VBlend, VShuffle,
    VBroadcast,

    // Reductions
    VHAdd,   // Horizontal add
    VHMax, VHMin,

    // Memory
    VLoad, VStore,
    VGather, VScatter,
}
```

## Parallelization

```rust
use bhc_loop_ir::parallel::{ParallelPass, ParallelConfig, ParFor, ParReduce};

let config = ParallelConfig {
    worker_count: 8,
    deterministic: true,
    min_chunk_size: 1024,
    ..Default::default()
};

let mut pass = ParallelPass::new(config);
let analysis = pass.analyze(&loop_ir);

for (loop_id, info) in analysis {
    if info.parallelizable {
        println!("Loop {} can run on {} chunks",
            loop_id, info.num_chunks);
    }
}
```

### Parallel Primitives

```rust
// Parallel for
pub struct ParFor {
    pub range: Range,
    pub body: Body,
    pub config: ParallelConfig,
}

// Parallel map
pub struct ParMap {
    pub size: usize,
    pub map_fn: Symbol,
    pub config: ParallelConfig,
}

// Parallel reduce
pub struct ParReduce {
    pub size: usize,
    pub op: ReduceOp,
    pub config: ParallelConfig,
}
```

### Parallel Strategies

```rust
pub enum ParallelStrategy {
    /// Static chunking (deterministic)
    Static,
    /// Dynamic work stealing
    Dynamic,
    /// Guided self-scheduling
    Guided,
}
```

## Lowering from Tensor IR

```rust
use bhc_loop_ir::lower::{lower_kernel, LowerConfig};

let config = LowerConfig {
    target: TargetArch::X86_64Avx2,
    tile_sizes: vec![64, 64, 64],
    enable_vectorization: true,
    enable_parallelization: true,
};

let loop_ir = lower_kernel(&kernel, config)?;
```

## Loop Transformations

| Transformation | Description |
|----------------|-------------|
| Tiling | Break into cache-friendly tiles |
| Vectorization | Convert to SIMD |
| Parallelization | Mark for parallel execution |
| Interchange | Reorder for better access |
| Unrolling | Reduce loop overhead |

## Dependencies

```rust
pub struct LoopDependency {
    pub source: LoopId,
    pub target: LoopId,
    pub kind: DependencyKind,
    pub distance: Option<Vec<i32>>,
}

pub enum DependencyKind {
    Flow,    // Read after write
    Anti,    // Write after read
    Output,  // Write after write
    Input,   // Read after read (locality)
}
```

## Error Handling

```rust
pub enum LoopIrError {
    TypeMismatch { expected: LoopType, got: LoopType },
    InvalidVectorWidth { width: u8, ty: ScalarType },
    OutOfBounds,
    InvalidTransform { reason: String },
}
```

## M3 Exit Criteria Tests

Loop IR includes integration tests for M3:

1. **Matmul auto-vectorizes** on x86_64 and aarch64
2. **Reductions scale linearly** up to 8 cores
3. **Deterministic mode** produces identical results

## See Also

- `bhc-tensor-ir`: Tensor IR that lowers to Loop IR
- `bhc-codegen`: Code generation from Loop IR
- H26-SPEC Section 7: Tensor Model (lowering)
