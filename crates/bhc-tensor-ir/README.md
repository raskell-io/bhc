# bhc-tensor-ir

Tensor Intermediate Representation for the Basel Haskell Compiler.

## Overview

This crate defines the Tensor IR, a specialized representation for numeric computations that enables aggressive optimization of array and matrix operations. It implements the M9 tensor model from the H26-SPEC, providing guaranteed fusion and efficient memory layout.

## Features

- Shape-indexed tensor types with compile-time dimension checking
- Guaranteed fusion for composable operations
- Memory layout optimization (row-major, column-major, strided)
- Automatic broadcasting semantics
- Element-wise, reduction, and contraction operations
- Fusion-friendly kernel representation

## Key Types

| Type | Description |
|------|-------------|
| `TensorOp` | Tensor operation enum |
| `TensorMeta` | Tensor metadata (dtype, shape, strides, layout) |
| `Kernel` | Fused operation kernel |
| `Shape` | Tensor dimensions |
| `DType` | Element data type |
| `Strides` | Memory strides for each dimension |
| `Layout` | Memory layout (RowMajor, ColMajor, Strided) |

## Usage

### Creating Tensor Operations

```rust
use bhc_tensor_ir::{TensorOp, TensorMeta, Shape, DType, Layout};

// Create tensor metadata
let meta = TensorMeta {
    dtype: DType::F32,
    shape: Shape::from([1024, 768]),
    strides: Strides::contiguous(&[1024, 768]),
    layout: Layout::RowMajor,
    alias: None,
};

// Element-wise operation: a + b
let add = TensorOp::Binary {
    op: BinOp::Add,
    lhs: tensor_a,
    rhs: tensor_b,
    out_meta: meta,
};

// Matrix multiplication
let matmul = TensorOp::Contraction {
    lhs: matrix_a,  // [m, k]
    rhs: matrix_b,  // [k, n]
    axes: (1, 0),   // Contract along k
    out_meta: matmul_meta,  // [m, n]
};
```

### Tensor Metadata (H26-SPEC Section 7.3)

```rust
pub struct TensorMeta {
    /// Element type: f32, f64, i32, i64, etc.
    pub dtype: DType,

    /// Shape: dimensions of the tensor
    pub shape: Shape,

    /// Strides: memory stride for each dimension
    pub strides: Strides,

    /// Layout: memory organization
    pub layout: Layout,

    /// Alias: memory aliasing information
    pub alias: Option<AliasInfo>,
}
```

## Operation Variants

```rust
pub enum TensorOp {
    // Creation
    Zeros(TensorMeta),
    Ones(TensorMeta),
    Full(Scalar, TensorMeta),
    FromData(Vec<Scalar>, TensorMeta),

    // Element-wise unary
    Unary { op: UnaryOp, input: TensorRef, out_meta: TensorMeta },

    // Element-wise binary
    Binary { op: BinOp, lhs: TensorRef, rhs: TensorRef, out_meta: TensorMeta },

    // Reductions
    Reduce { op: ReduceOp, input: TensorRef, axes: Vec<usize>, out_meta: TensorMeta },

    // Shape operations
    Reshape { input: TensorRef, new_shape: Shape },
    Transpose { input: TensorRef, perm: Vec<usize> },
    Broadcast { input: TensorRef, new_shape: Shape },
    Slice { input: TensorRef, ranges: Vec<Range> },

    // Contractions (matmul, einsum)
    Contraction { lhs: TensorRef, rhs: TensorRef, axes: (usize, usize), out_meta: TensorMeta },

    // Memory
    Copy { input: TensorRef, out_meta: TensorMeta },
    View { input: TensorRef, out_meta: TensorMeta },
}
```

## Guaranteed Fusion (H26-SPEC Section 8)

The Tensor IR guarantees fusion for these patterns:

```rust
// Pattern 1: Element-wise chains
// a.map(f).map(g).map(h) → a.map(f ∘ g ∘ h)

// Pattern 2: Map-reduce
// a.map(f).reduce(+) → fused map-reduce kernel

// Pattern 3: Broadcast-binary
// a + broadcast(b) → single pass with inline broadcast

// Pattern 4: Transpose-matmul
// matmul(transpose(a), b) → single matmul with transposed access
```

### Fusion in Practice

```rust
use bhc_tensor_ir::{Kernel, FusedOp};

// Multiple operations fuse into a single kernel
let kernel = Kernel {
    ops: vec![
        FusedOp::Load { src: tensor_a },
        FusedOp::Unary { op: UnaryOp::Exp },
        FusedOp::Binary { op: BinOp::Mul, rhs: tensor_b },
        FusedOp::Reduce { op: ReduceOp::Sum, axis: 1 },
        FusedOp::Store { dst: output },
    ],
    shape: output_shape,
    parallelism: Parallelism::DataParallel,
};
```

## Data Types

```rust
pub enum DType {
    F16,    // 16-bit float (half)
    F32,    // 32-bit float
    F64,    // 64-bit float (double)
    BF16,   // Brain float 16
    I8,     // 8-bit signed integer
    I16,    // 16-bit signed integer
    I32,    // 32-bit signed integer
    I64,    // 64-bit signed integer
    U8,     // 8-bit unsigned
    U32,    // 32-bit unsigned
    Bool,   // Boolean
}
```

## Memory Layouts

```rust
pub enum Layout {
    RowMajor,      // C-style: last dimension contiguous
    ColMajor,      // Fortran-style: first dimension contiguous
    Strided,       // Arbitrary strides (slices, transposes)
}

impl Strides {
    /// Contiguous row-major strides for shape [d0, d1, d2]
    /// → [d1*d2, d2, 1]
    pub fn contiguous(shape: &[usize]) -> Self;

    /// Contiguous column-major strides for shape [d0, d1, d2]
    /// → [1, d0, d0*d1]
    pub fn fortran(shape: &[usize]) -> Self;
}
```

## Design Notes

- Tensor IR operates on logical tensor operations, not loops
- Shape information enables static dimension checking
- Stride representation allows zero-copy views and slices
- Fusion decisions are made before lowering to Loop IR
- Alias analysis prevents incorrect optimizations

## Related Crates

- `bhc-core` - Input Core IR
- `bhc-loop-ir` - Output Loop IR for codegen
- `bhc-types` - Type-level shape representation (M9)
- `bhc-gpu` - GPU code generation from Tensor IR
- `bhc-codegen` - CPU code generation

## Specification References

- H26-SPEC Section 7: Tensor Model (M9)
- H26-SPEC Section 7.3: TensorMeta Requirements
- H26-SPEC Section 8: Fusion Guarantees
