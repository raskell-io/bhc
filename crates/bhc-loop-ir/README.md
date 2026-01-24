# bhc-loop-ir

Loop Intermediate Representation for the Basel Haskell Compiler.

## Overview

This crate defines the Loop IR, a low-level representation with explicit loops, memory operations, and SIMD primitives. It bridges the gap between high-level Tensor IR and machine code, enabling vectorization, parallelization, and efficient code generation.

## Features

- Explicit loop nests with iteration bounds
- SIMD vector types and operations
- Parallel loop primitives (ParFor, ParMap, ParReduce)
- Memory references with aliasing information
- Loop transformations (tiling, unrolling, interchange)
- Target-specific SIMD: SSE, AVX, AVX-512, NEON

## Key Types

| Type | Description |
|------|-------------|
| `LoopIR` | Root of the Loop IR tree |
| `Loop` | Loop construct with bounds and body |
| `LoopType` | Loop classification (Sequential, Parallel, SIMD) |
| `Stmt` | Statement enum (assignments, stores, loops) |
| `Value` | Value representation (registers, constants) |
| `MemRef` | Memory reference with type and aliasing |
| `VectorType` | SIMD vector types |

## Usage

### Creating Loop Structures

```rust
use bhc_loop_ir::{Loop, LoopType, Stmt, Value, MemRef};

// Create a simple loop: for i in 0..n
let loop_ir = Loop {
    var: "i",
    lower: Value::Const(0),
    upper: Value::Var("n"),
    step: Value::Const(1),
    loop_type: LoopType::Sequential,
    body: vec![
        Stmt::Store {
            dst: MemRef::index("output", "i"),
            value: Value::Load(MemRef::index("input", "i")),
        },
    ],
};

// Parallel loop
let par_loop = Loop {
    loop_type: LoopType::Parallel { num_threads: 8 },
    ..loop_ir
};

// SIMD vectorized loop
let simd_loop = Loop {
    loop_type: LoopType::SIMD { width: 8 },
    step: Value::Const(8),
    ..loop_ir
};
```

### SIMD Operations

```rust
use bhc_loop_ir::{VectorType, VectorOp};

// Vector types
let v4f32 = VectorType::VEC4F32;  // 4x f32 (SSE)
let v8f32 = VectorType::VEC8F32;  // 8x f32 (AVX)
let v4f64 = VectorType::VEC4F64;  // 4x f64 (AVX)

// Vector operations
let vadd = Stmt::VectorOp {
    op: VectorOp::Add,
    dst: "v_result",
    lhs: Value::VecReg("v_a"),
    rhs: Value::VecReg("v_b"),
    ty: VectorType::VEC8F32,
};

// Vector load/store
let vload = Stmt::VectorLoad {
    dst: "v_data",
    src: MemRef::aligned("array", "i", 32),
    ty: VectorType::VEC8F32,
};
```

## Statement Variants

```rust
pub enum Stmt {
    // Scalar operations
    Assign { dst: String, value: Value },
    Store { dst: MemRef, value: Value },

    // Control flow
    Loop(Loop),
    If { cond: Value, then_body: Vec<Stmt>, else_body: Vec<Stmt> },

    // Vector operations
    VectorOp { op: VectorOp, dst: String, lhs: Value, rhs: Value, ty: VectorType },
    VectorLoad { dst: String, src: MemRef, ty: VectorType },
    VectorStore { dst: MemRef, value: Value, ty: VectorType },
    VectorBroadcast { dst: String, scalar: Value, ty: VectorType },
    VectorReduce { op: ReduceOp, dst: String, src: String, ty: VectorType },

    // Parallel primitives
    ParFor { var: String, range: Range, body: Vec<Stmt>, num_threads: usize },
    ParMap { input: MemRef, output: MemRef, kernel: Vec<Stmt> },
    ParReduce { input: MemRef, op: ReduceOp, init: Value },

    // Synchronization
    Barrier,
    Atomic { op: AtomicOp, dst: MemRef, value: Value },
}
```

## Loop Types

```rust
pub enum LoopType {
    /// Standard sequential loop
    Sequential,

    /// Parallel loop with thread count
    Parallel { num_threads: usize },

    /// SIMD vectorized loop
    SIMD { width: usize },

    /// Unrolled loop
    Unrolled { factor: usize },

    /// Tiled loop for cache optimization
    Tiled { tile_size: usize },
}
```

## Vector Types

| Type | Elements | Size | Target |
|------|----------|------|--------|
| `VEC4F32` | 4 × f32 | 128-bit | SSE, NEON |
| `VEC8F32` | 8 × f32 | 256-bit | AVX, AVX2 |
| `VEC16F32` | 16 × f32 | 512-bit | AVX-512 |
| `VEC2F64` | 2 × f64 | 128-bit | SSE2, NEON |
| `VEC4F64` | 4 × f64 | 256-bit | AVX |
| `VEC8F64` | 8 × f64 | 512-bit | AVX-512 |
| `VEC4I32` | 4 × i32 | 128-bit | SSE2, NEON |
| `VEC8I32` | 8 × i32 | 256-bit | AVX2 |

## Memory References

```rust
pub struct MemRef {
    /// Base pointer name
    pub base: String,

    /// Index expression
    pub index: IndexExpr,

    /// Element type
    pub elem_ty: ScalarType,

    /// Alignment requirement
    pub align: usize,

    /// Aliasing information
    pub alias_set: AliasSet,
}

impl MemRef {
    /// Simple indexed access: base[i]
    pub fn index(base: &str, idx: &str) -> Self;

    /// Aligned access for SIMD
    pub fn aligned(base: &str, idx: &str, align: usize) -> Self;

    /// Strided access: base[i * stride]
    pub fn strided(base: &str, idx: &str, stride: usize) -> Self;
}
```

## Target Architectures

```rust
pub enum Target {
    X86_64 {
        features: X86Features,  // SSE, AVX, AVX2, AVX512
    },
    AArch64 {
        features: ArmFeatures,  // NEON, SVE, SVE2
    },
    WASM {
        features: WasmFeatures, // SIMD128
    },
}

pub struct X86Features {
    pub sse: bool,
    pub sse2: bool,
    pub sse4_1: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
}
```

## Loop Transformations

The Loop IR supports these transformations:

| Transformation | Description |
|---------------|-------------|
| Tiling | Break loops into tiles for cache locality |
| Unrolling | Repeat loop body to reduce overhead |
| Vectorization | Convert scalar ops to SIMD ops |
| Interchange | Reorder nested loops |
| Fusion | Combine adjacent loops |
| Fission | Split loops for parallelism |
| Peeling | Handle edge cases separately |

## Design Notes

- Loop IR is the last stage before target-specific codegen
- SIMD width adapts to target architecture
- Aliasing information enables safe optimizations
- Parallel loops generate threading runtime calls
- Alignment requirements are explicit for SIMD

## Related Crates

- `bhc-tensor-ir` - Input tensor operations
- `bhc-codegen` - Machine code generation
- `bhc-target` - Target architecture definitions
- `bhc-gpu` - GPU-specific lowering (bypasses Loop IR)

## Specification References

- H26-SPEC Section 3.5: Loop IR
- H26-SPEC Section 9: SIMD Model (M3)
- H26-SPEC Section 10: Parallelism Model
