//! # BHC Tensor IR
//!
//! This crate defines the Tensor Intermediate Representation for the Basel
//! Haskell Compiler. Tensor IR is specifically designed for numeric optimization
//! and provides the foundation for guaranteed fusion and vectorization.
//!
//! ## Overview
//!
//! Tensor IR is the key to BHC's numeric performance. It captures:
//!
//! - **Shape and stride information**: For layout-aware optimization
//! - **Element types (dtypes)**: For unboxed numeric computation
//! - **Operation structure**: For fusion analysis
//! - **Aliasing information**: For safe in-place updates
//!
//! ## H26-SPEC Section 7 Compliance
//!
//! Per the H26 specification, every tensor operation in Tensor IR must track:
//!
//! | Property | Description |
//! |----------|-------------|
//! | `dtype`  | Element type (Float32, Float64, Int32, etc.) |
//! | `shape`  | Dimension sizes |
//! | `strides`| Byte strides per dimension |
//! | `layout` | Memory layout (contiguous, strided, tiled) |
//! | `alias`  | Aliasing/ownership information |
//!
//! ## IR Pipeline Position
//!
//! ```text
//! Source Code
//!     |
//!     v
//! [Parse/AST]
//!     |
//!     v
//! [HIR]
//!     |
//!     v
//! [Core IR]   <- General purpose optimizations
//!     |
//!     | (Numeric Profile only)
//!     v
//! [Tensor IR] <- This crate: shape-aware, fusion-ready
//!     |
//!     v
//! [Loop IR]   <- Explicit iteration
//! ```
//!
//! ## Guaranteed Fusion Patterns
//!
//! Per H26-SPEC Section 8, these patterns MUST fuse:
//!
//! 1. `map f (map g x)` -> single traversal
//! 2. `zipWith f (map g a) (map h b)` -> single traversal
//! 3. `sum (map f x)` -> single traversal
//! 4. `foldl' op z (map f x)` -> single traversal
//!
//! ## Main Types
//!
//! - [`TensorOp`]: Tensor operations
//! - [`TensorMeta`]: Metadata (shape, stride, dtype)
//! - [`Kernel`]: A fused computation unit
//! - [`Shape`]: Tensor dimensions
//! - [`DType`]: Element types
//!
//! ## See Also
//!
//! - `bhc-core`: Core IR that lowers to Tensor IR
//! - `bhc-loop-ir`: Loop IR for explicit iteration
//! - H26-SPEC Section 7: Tensor Model
//! - H26-SPEC Section 8: Fusion Laws

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod fusion;
pub mod lower;

use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A unique identifier for tensor operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(u32);

impl Idx for TensorId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A unique identifier for kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelId(u32);

impl Idx for KernelId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A unique identifier for buffers (memory allocations).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(u32);

impl Idx for BufferId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Tensor element types (data types).
///
/// These represent the unboxed element types that can be stored
/// in tensors. Each dtype has known size and alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// Boolean (1 byte).
    Bool,
    /// 8-bit signed integer.
    Int8,
    /// 16-bit signed integer.
    Int16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 8-bit unsigned integer.
    UInt8,
    /// 16-bit unsigned integer.
    UInt16,
    /// 32-bit unsigned integer.
    UInt32,
    /// 64-bit unsigned integer.
    UInt64,
    /// 16-bit floating point (half precision).
    Float16,
    /// 32-bit floating point (single precision).
    Float32,
    /// 64-bit floating point (double precision).
    Float64,
    /// Brain floating point (bfloat16).
    BFloat16,
    /// Complex number (single precision).
    Complex64,
    /// Complex number (double precision).
    Complex128,
}

impl DType {
    /// Returns the size in bytes of this dtype.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 | Self::Float16 | Self::BFloat16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 | Self::Complex64 => 8,
            Self::Complex128 => 16,
        }
    }

    /// Returns the alignment in bytes for this dtype.
    #[must_use]
    pub const fn alignment(self) -> usize {
        self.size_bytes()
    }

    /// Returns true if this is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::Float16 | Self::Float32 | Self::Float64 | Self::BFloat16
        )
    }

    /// Returns true if this is an integer type.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::UInt8
                | Self::UInt16
                | Self::UInt32
                | Self::UInt64
        )
    }

    /// Returns true if this is a signed type.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::Float16
                | Self::Float32
                | Self::Float64
                | Self::BFloat16
                | Self::Complex64
                | Self::Complex128
        )
    }
}

/// A dimension size (may be static or dynamic).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    /// A statically known dimension.
    Static(usize),
    /// A dynamically determined dimension (symbolic).
    Dynamic(Symbol),
}

impl Dim {
    /// Returns the static value if known.
    #[must_use]
    pub const fn static_value(&self) -> Option<usize> {
        match self {
            Self::Static(n) => Some(*n),
            Self::Dynamic(_) => None,
        }
    }

    /// Returns true if this dimension is statically known.
    #[must_use]
    pub const fn is_static(&self) -> bool {
        matches!(self, Self::Static(_))
    }
}

/// Tensor shape (list of dimensions).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape(SmallVec<[Dim; 4]>);

impl Shape {
    /// Creates a new shape from dimensions.
    #[must_use]
    pub fn new(dims: impl IntoIterator<Item = Dim>) -> Self {
        Self(dims.into_iter().collect())
    }

    /// Creates a shape from static dimensions.
    #[must_use]
    pub fn from_static(dims: impl IntoIterator<Item = usize>) -> Self {
        Self(dims.into_iter().map(Dim::Static).collect())
    }

    /// Creates a scalar shape (rank 0).
    #[must_use]
    pub fn scalar() -> Self {
        Self(SmallVec::new())
    }

    /// Returns the rank (number of dimensions).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns the dimensions.
    #[must_use]
    pub fn dims(&self) -> &[Dim] {
        &self.0
    }

    /// Returns the total number of elements (if statically known).
    #[must_use]
    pub fn num_elements(&self) -> Option<usize> {
        self.0
            .iter()
            .try_fold(1usize, |acc, dim| dim.static_value().map(|n| acc * n))
    }

    /// Returns true if this is a scalar (rank 0).
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns true if all dimensions are statically known.
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.0.iter().all(Dim::is_static)
    }
}

/// Memory strides for each dimension.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Strides(SmallVec<[i64; 4]>);

impl Strides {
    /// Creates new strides.
    #[must_use]
    pub fn new(strides: impl IntoIterator<Item = i64>) -> Self {
        Self(strides.into_iter().collect())
    }

    /// Computes contiguous (row-major) strides for a shape.
    #[must_use]
    pub fn contiguous(shape: &Shape, elem_size: usize) -> Option<Self> {
        let mut strides = SmallVec::with_capacity(shape.rank());
        let mut stride = elem_size as i64;

        for dim in shape.dims().iter().rev() {
            strides.push(stride);
            stride *= dim.static_value()? as i64;
        }

        strides.reverse();
        Some(Self(strides))
    }

    /// Returns the stride values.
    #[must_use]
    pub fn values(&self) -> &[i64] {
        &self.0
    }

    /// Returns true if these strides represent contiguous memory.
    #[must_use]
    pub fn is_contiguous(&self, shape: &Shape, elem_size: usize) -> bool {
        if let Some(contiguous) = Self::contiguous(shape, elem_size) {
            self.0 == contiguous.0
        } else {
            false
        }
    }
}

/// Memory layout of a tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layout {
    /// Contiguous memory (row-major by default).
    Contiguous,
    /// Strided layout (possibly non-contiguous).
    Strided,
    /// Tiled layout for cache efficiency.
    Tiled(TileInfo),
}

/// Tiling information for cache-friendly layouts.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileInfo {
    /// Tile sizes for each dimension.
    pub tile_sizes: SmallVec<[usize; 4]>,
    /// The dimension order within tiles.
    pub inner_order: SmallVec<[usize; 4]>,
}

/// Tensor metadata per H26-SPEC Section 7.3.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Element type.
    pub dtype: DType,
    /// Tensor shape.
    pub shape: Shape,
    /// Memory strides.
    pub strides: Strides,
    /// Memory layout.
    pub layout: Layout,
    /// Aliasing information (buffer this tensor references).
    pub alias: Option<BufferId>,
}

impl TensorMeta {
    /// Creates metadata for a new contiguous tensor.
    #[must_use]
    pub fn new_contiguous(dtype: DType, shape: Shape) -> Option<Self> {
        let strides = Strides::contiguous(&shape, dtype.size_bytes())?;
        Some(Self {
            dtype,
            shape,
            strides,
            layout: Layout::Contiguous,
            alias: None,
        })
    }

    /// Returns the total size in bytes (if statically known).
    #[must_use]
    pub fn size_bytes(&self) -> Option<usize> {
        self.shape
            .num_elements()
            .map(|n| n * self.dtype.size_bytes())
    }
}

/// A reference to a tensor value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorRef {
    /// The tensor ID.
    pub id: TensorId,
    /// The metadata.
    pub meta: TensorMeta,
}

/// Tensor operations in the IR.
///
/// These operations form the building blocks of tensor computations.
/// The fusion pass analyzes these to produce fused kernels.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TensorOp {
    /// A constant tensor.
    Constant(ConstantOp),

    // === Elementwise Operations ===
    /// Unary elementwise operation.
    Unary(UnaryOp, TensorRef),
    /// Binary elementwise operation.
    Binary(BinaryOp, TensorRef, TensorRef),
    /// Map a function over elements.
    Map(MapFn, TensorRef),
    /// Zip two tensors with a function.
    ZipWith(ZipFn, TensorRef, TensorRef),

    // === Reductions ===
    /// Reduce along an axis.
    Reduce(ReduceOp, Axis, TensorRef),
    /// Full reduction to scalar.
    ReduceAll(ReduceOp, TensorRef),
    /// Scan (prefix sum) along an axis.
    Scan(ReduceOp, Axis, TensorRef),
    /// Fold with initial value.
    Fold(FoldFn, TensorRef, TensorRef),

    // === Structure Operations ===
    /// Reshape to a new shape.
    Reshape(Shape, TensorRef),
    /// Slice a region.
    Slice(SliceSpec, TensorRef),
    /// Transpose (permute dimensions).
    Transpose(Permutation, TensorRef),
    /// Broadcast to a larger shape.
    Broadcast(Shape, TensorRef),
    /// Concatenate along an axis.
    Concat(Axis, Vec<TensorRef>),
    /// Split along an axis.
    Split(Axis, Vec<usize>, TensorRef),

    // === Linear Algebra ===
    /// Matrix multiplication.
    MatMul(TensorRef, TensorRef),
    /// Batched matrix multiplication.
    BatchMatMul(TensorRef, TensorRef),
    /// Dot product.
    Dot(TensorRef, TensorRef),
    /// Outer product.
    Outer(TensorRef, TensorRef),

    // === Convolution ===
    /// Convolution operation.
    Conv(ConvSpec, TensorRef, TensorRef),

    // === Indexing ===
    /// Gather elements.
    Gather(Axis, TensorRef, TensorRef),
    /// Scatter elements.
    Scatter(Axis, TensorRef, TensorRef, TensorRef),
}

/// A constant tensor operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConstantOp {
    /// Zeros tensor.
    Zeros(TensorMeta),
    /// Ones tensor.
    Ones(TensorMeta),
    /// Tensor filled with a value.
    Full(TensorMeta, ScalarValue),
    /// Range/arange tensor.
    Range(DType, i64, i64, i64),
    /// Identity matrix.
    Eye(DType, usize),
}

/// Unary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Negation.
    Neg,
    /// Absolute value.
    Abs,
    /// Square root.
    Sqrt,
    /// Reciprocal square root.
    Rsqrt,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Hyperbolic tangent.
    Tanh,
    /// Sigmoid.
    Sigmoid,
    /// ReLU.
    Relu,
    /// Ceiling.
    Ceil,
    /// Floor.
    Floor,
    /// Round.
    Round,
    /// Bitwise not (integers).
    Not,
}

/// Binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Modulo.
    Mod,
    /// Power.
    Pow,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
    /// Equality.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise xor.
    Xor,
    /// Left shift.
    Shl,
    /// Right shift.
    Shr,
}

/// Reduction operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Sum reduction.
    Sum,
    /// Product reduction.
    Prod,
    /// Maximum reduction.
    Max,
    /// Minimum reduction.
    Min,
    /// Logical and.
    All,
    /// Logical or.
    Any,
    /// Mean (sum / count).
    Mean,
}

/// An axis specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Axis(pub i32);

impl Axis {
    /// Creates a new axis.
    #[must_use]
    pub const fn new(axis: i32) -> Self {
        Self(axis)
    }

    /// Normalizes a potentially negative axis to a positive index.
    #[must_use]
    pub const fn normalize(self, rank: usize) -> Option<usize> {
        let axis = if self.0 < 0 {
            (rank as i32) + self.0
        } else {
            self.0
        };
        if axis >= 0 && (axis as usize) < rank {
            Some(axis as usize)
        } else {
            None
        }
    }
}

/// A scalar value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScalarValue {
    /// Boolean.
    Bool(bool),
    /// Integer.
    Int(i64),
    /// Floating point.
    Float(f64),
}

/// A map function (element-wise transformation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapFn {
    /// The function name/identifier.
    pub name: Symbol,
    /// Source span.
    pub span: Span,
}

/// A zip function (combining two elements).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZipFn {
    /// The function name/identifier.
    pub name: Symbol,
    /// Source span.
    pub span: Span,
}

/// A fold function.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldFn {
    /// The combining function.
    pub name: Symbol,
    /// Source span.
    pub span: Span,
}

/// A slice specification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceSpec {
    /// Ranges for each dimension (start, stop, step).
    pub ranges: SmallVec<[SliceRange; 4]>,
}

/// A range within a slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceRange {
    /// Start index (inclusive).
    pub start: Option<i64>,
    /// Stop index (exclusive).
    pub stop: Option<i64>,
    /// Step size.
    pub step: i64,
}

/// A permutation for transpose.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permutation(SmallVec<[usize; 4]>);

impl Permutation {
    /// Creates a new permutation.
    #[must_use]
    pub fn new(perm: impl IntoIterator<Item = usize>) -> Self {
        Self(perm.into_iter().collect())
    }

    /// Returns the permutation as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// Returns true if this is the identity permutation.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.0.iter().enumerate().all(|(i, &p)| i == p)
    }
}

/// Convolution specification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvSpec {
    /// Padding per dimension.
    pub padding: SmallVec<[(usize, usize); 4]>,
    /// Stride per dimension.
    pub strides: SmallVec<[usize; 4]>,
    /// Dilation per dimension.
    pub dilation: SmallVec<[usize; 4]>,
    /// Number of groups.
    pub groups: usize,
}

/// A fused computation kernel.
///
/// Kernels are the output of the fusion pass. Each kernel
/// represents a unit of computation that executes without
/// intermediate allocation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Kernel {
    /// Unique kernel identifier.
    pub id: KernelId,
    /// Kernel name (for debugging/profiling).
    pub name: Symbol,
    /// Input tensors.
    pub inputs: Vec<TensorRef>,
    /// Output tensors.
    pub outputs: Vec<TensorRef>,
    /// The computation body.
    pub body: KernelBody,
    /// Allocation requirements.
    pub allocs: Vec<AllocInfo>,
    /// Fusion information.
    pub fusion_info: FusionInfo,
}

/// The body of a kernel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum KernelBody {
    /// A simple fused operation.
    Fused(Vec<TensorOp>),
    /// A loop nest (lowered from tensor ops).
    LoopNest(LoopNest),
}

/// A simple loop nest representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopNest {
    /// The loops from outermost to innermost.
    pub loops: Vec<LoopInfo>,
    /// The innermost computation.
    pub body: Vec<TensorOp>,
}

/// Information about a single loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopInfo {
    /// Loop variable name.
    pub var: Symbol,
    /// Lower bound.
    pub lower: i64,
    /// Upper bound.
    pub upper: Dim,
    /// Step size.
    pub step: i64,
    /// Whether this loop can be parallelized.
    pub parallel: bool,
    /// Whether this loop can be vectorized.
    pub vectorize: Option<usize>,
}

/// Allocation information for a kernel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllocInfo {
    /// The buffer being allocated.
    pub buffer: BufferId,
    /// Size in bytes.
    pub size: usize,
    /// Alignment requirement.
    pub alignment: usize,
    /// Allocation region.
    pub region: AllocRegion,
}

/// Memory regions for allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocRegion {
    /// Hot arena (bump allocated, scoped lifetime).
    HotArena,
    /// Pinned heap (for FFI).
    Pinned,
    /// General heap (GC managed).
    General,
    /// GPU device memory.
    DeviceMemory(DeviceTarget),
}

/// Target device for GPU memory allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceTarget {
    /// NVIDIA GPU (CUDA).
    Cuda(u32),
    /// AMD GPU (ROCm).
    Rocm(u32),
    /// Any available GPU.
    Any,
}

/// Fusion information for debugging and reporting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusionInfo {
    /// Original operations before fusion.
    pub original_ops: Vec<Symbol>,
    /// Fusion decisions made.
    pub decisions: Vec<FusionDecision>,
    /// Whether all expected fusions succeeded.
    pub complete: bool,
}

/// A fusion decision made by the compiler.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FusionDecision {
    /// Operations were successfully fused.
    Fused(Vec<Symbol>),
    /// A materialization point was inserted.
    Materialized(Symbol, MaterializeReason),
    /// Fusion was blocked for a reason.
    Blocked(Symbol, FusionBlockReason),
}

/// Why a tensor was materialized.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MaterializeReason {
    /// Used by multiple consumers.
    MultipleUses,
    /// Explicitly requested by programmer.
    Explicit,
    /// Required for control flow.
    ControlFlow,
}

/// Why fusion was blocked.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FusionBlockReason {
    /// Shape mismatch between operations.
    ShapeMismatch,
    /// Incompatible data types.
    DTypeMismatch,
    /// Data dependency prevents fusion.
    DataDependency,
    /// Side effects prevent reordering.
    SideEffects,
}

/// Errors in Tensor IR operations.
#[derive(Clone, Debug, thiserror::Error, Serialize, Deserialize)]
pub enum TensorIrError {
    /// Shape mismatch in operation.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Shape,
        /// Actual shape.
        got: Shape,
    },

    /// Invalid axis for operation.
    #[error("invalid axis {axis} for tensor of rank {rank}")]
    InvalidAxis {
        /// The axis specified.
        axis: i32,
        /// The tensor rank.
        rank: usize,
    },

    /// Type mismatch.
    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch {
        /// Expected dtype.
        expected: DType,
        /// Actual dtype.
        got: DType,
    },

    /// Fusion failed for guaranteed pattern.
    #[error("fusion failed for guaranteed pattern: {pattern}")]
    FusionFailed {
        /// The pattern that should have fused.
        pattern: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float64.size_bytes(), 8);
        assert_eq!(DType::Int32.size_bytes(), 4);
    }

    #[test]
    fn test_shape_num_elements() {
        let shape = Shape::from_static([2, 3, 4]);
        assert_eq!(shape.num_elements(), Some(24));
        assert_eq!(shape.rank(), 3);
    }

    #[test]
    fn test_strides_contiguous() {
        let shape = Shape::from_static([2, 3, 4]);
        let strides = Strides::contiguous(&shape, 4).unwrap();
        assert_eq!(strides.values(), &[48, 16, 4]);
    }

    #[test]
    fn test_axis_normalize() {
        let axis = Axis::new(-1);
        assert_eq!(axis.normalize(3), Some(2));

        let axis = Axis::new(1);
        assert_eq!(axis.normalize(3), Some(1));

        let axis = Axis::new(5);
        assert_eq!(axis.normalize(3), None);
    }

    #[test]
    fn test_permutation_identity() {
        let perm = Permutation::new([0, 1, 2]);
        assert!(perm.is_identity());

        let perm = Permutation::new([2, 0, 1]);
        assert!(!perm.is_identity());
    }
}
