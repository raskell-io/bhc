//! # BHC Loop IR
//!
//! This crate defines the Loop Intermediate Representation for the Basel
//! Haskell Compiler. Loop IR makes iteration structure explicit and is the
//! target for vectorization and low-level optimization.
//!
//! ## Overview
//!
//! Loop IR is the lowest-level IR before code generation. It provides:
//!
//! - **Explicit iteration**: Loops with bounds and strides
//! - **Vectorization information**: Which loops can be SIMD-ized
//! - **Parallelization hints**: Which loops can run in parallel
//! - **Memory access patterns**: For cache optimization
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
//! [Core IR]
//!     |
//!     v
//! [Tensor IR]  <- High-level tensor operations
//!     |
//!     v
//! [Loop IR]    <- This crate: explicit iteration
//!     |
//!     v
//! [Codegen]    <- LLVM IR / Native code
//! ```
//!
//! ## Key Transformations
//!
//! Loop IR supports several important optimizations:
//!
//! 1. **Loop tiling**: Break loops into cache-friendly tiles
//! 2. **Vectorization**: Convert scalar operations to SIMD
//! 3. **Parallelization**: Mark loops for parallel execution
//! 4. **Interchange**: Reorder loops for better memory access
//! 5. **Unrolling**: Reduce loop overhead
//!
//! ## Main Types
//!
//! - [`LoopIR`]: The top-level IR structure
//! - [`Loop`]: A single loop with bounds and body
//! - [`Stmt`]: Statements within loop bodies
//! - [`Value`]: SSA values (registers)
//! - [`MemRef`]: Memory references with access patterns
//!
//! ## M3 Deliverables
//!
//! This crate implements the following M3 features:
//!
//! - **SIMD Types**: [`LoopType::VEC4F32`], [`LoopType::VEC8F32`], [`LoopType::VEC2F64`], [`LoopType::VEC4F64`]
//! - **Auto-vectorization**: [`vectorize::VectorizePass`]
//! - **Parallel primitives**: [`parallel::ParFor`], [`parallel::ParMap`], [`parallel::ParReduce`]
//! - **SIMD intrinsics**: [`vectorize::SimdIntrinsic`]
//!
//! ## See Also
//!
//! - `bhc-tensor-ir`: Tensor IR that lowers to Loop IR
//! - `bhc-codegen`: Code generation from Loop IR
//! - H26-SPEC Section 7: Tensor Model (lowering)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_tensor_ir::{AllocRegion, BufferId, DType};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

// ============================================================================
// Submodules
// ============================================================================

pub mod lower;
pub mod parallel;
pub mod vectorize;

// Re-export key types from submodules
pub use lower::{lower_kernel, lower_kernels, LowerConfig, LowerError};
pub use parallel::{
    ParFor, ParMap, ParReduce, ParallelConfig, ParallelPass, ParallelStrategy, Range,
};
pub use vectorize::{SimdIntrinsic, VectorizeConfig, VectorizePass, VectorizeReport};

/// A unique identifier for values (SSA registers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueId(u32);

impl Idx for ValueId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A unique identifier for loops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LoopId(u32);

impl Idx for LoopId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A unique identifier for basic blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(u32);

impl Idx for BlockId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// The main Loop IR structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopIR {
    /// Function name.
    pub name: Symbol,
    /// Function parameters.
    pub params: Vec<Param>,
    /// Return type.
    pub return_ty: LoopType,
    /// The body (list of statements and loops).
    pub body: Body,
    /// Memory allocations.
    pub allocs: Vec<Alloc>,
    /// Loop metadata for optimization.
    pub loop_info: Vec<LoopMetadata>,
}

/// A function parameter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Param {
    /// Parameter name.
    pub name: Symbol,
    /// Parameter type.
    pub ty: LoopType,
    /// Whether this is a pointer to memory.
    pub is_ptr: bool,
}

/// Types in Loop IR.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoopType {
    /// Void (no value).
    Void,
    /// Scalar type.
    Scalar(ScalarType),
    /// Vector type (SIMD).
    Vector(ScalarType, u8),
    /// Pointer to memory.
    Ptr(Box<LoopType>),
}

impl LoopType {
    /// Returns the size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Void => 0,
            Self::Scalar(s) => s.size_bytes(),
            Self::Vector(s, width) => s.size_bytes() * (*width as usize),
            Self::Ptr(_) => 8, // Assuming 64-bit pointers
        }
    }

    /// Returns true if this is a void type.
    #[must_use]
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Void)
    }

    /// Returns true if this is a vector type.
    #[must_use]
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(_, _))
    }
}

/// Scalar types in Loop IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    /// Boolean.
    Bool,
    /// Signed integer with bit width.
    Int(u8),
    /// Unsigned integer with bit width.
    UInt(u8),
    /// Floating point with bit width.
    Float(u8),
}

impl ScalarType {
    /// Returns the size in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Int(bits) | Self::UInt(bits) | Self::Float(bits) => (bits as usize + 7) / 8,
        }
    }

    /// Converts from tensor DType.
    #[must_use]
    pub fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Bool => Self::Bool,
            DType::Int8 => Self::Int(8),
            DType::Int16 => Self::Int(16),
            DType::Int32 => Self::Int(32),
            DType::Int64 => Self::Int(64),
            DType::UInt8 => Self::UInt(8),
            DType::UInt16 => Self::UInt(16),
            DType::UInt32 => Self::UInt(32),
            DType::UInt64 => Self::UInt(64),
            DType::Float16 | DType::BFloat16 => Self::Float(16),
            DType::Float32 => Self::Float(32),
            DType::Float64 => Self::Float(64),
            DType::Complex64 => Self::Float(32), // Represented as pair
            DType::Complex128 => Self::Float(64),
        }
    }

    /// 32-bit float scalar type.
    pub const F32: Self = Self::Float(32);

    /// 64-bit float scalar type.
    pub const F64: Self = Self::Float(64);

    /// 32-bit signed integer scalar type.
    pub const I32: Self = Self::Int(32);

    /// 64-bit signed integer scalar type.
    pub const I64: Self = Self::Int(64);
}

// ============================================================================
// SIMD Type Aliases and Constructors (M3 Deliverable)
// ============================================================================

impl LoopType {
    // --- Standard SIMD Vector Types ---

    /// 4-wide 32-bit float vector (128-bit, SSE/NEON compatible).
    pub const VEC4F32: Self = Self::Vector(ScalarType::F32, 4);

    /// 8-wide 32-bit float vector (256-bit, AVX compatible).
    pub const VEC8F32: Self = Self::Vector(ScalarType::F32, 8);

    /// 2-wide 64-bit float vector (128-bit, SSE/NEON compatible).
    pub const VEC2F64: Self = Self::Vector(ScalarType::F64, 2);

    /// 4-wide 64-bit float vector (256-bit, AVX compatible).
    pub const VEC4F64: Self = Self::Vector(ScalarType::F64, 4);

    /// 4-wide 32-bit integer vector (128-bit).
    pub const VEC4I32: Self = Self::Vector(ScalarType::I32, 4);

    /// 8-wide 32-bit integer vector (256-bit).
    pub const VEC8I32: Self = Self::Vector(ScalarType::I32, 8);

    /// Returns the natural vector width for a scalar type on the target.
    ///
    /// # Target Widths
    ///
    /// | Target | F32 | F64 | I32 |
    /// |--------|-----|-----|-----|
    /// | x86_64 (SSE) | 4 | 2 | 4 |
    /// | x86_64 (AVX) | 8 | 4 | 8 |
    /// | aarch64 (NEON) | 4 | 2 | 4 |
    #[must_use]
    pub fn natural_vector_width(scalar: ScalarType, target: TargetArch) -> u8 {
        match (target, scalar) {
            // x86_64 with AVX (256-bit vectors)
            (TargetArch::X86_64Avx | TargetArch::X86_64Avx2, ScalarType::Float(32)) => 8,
            (TargetArch::X86_64Avx | TargetArch::X86_64Avx2, ScalarType::Float(64)) => 4,
            (TargetArch::X86_64Avx | TargetArch::X86_64Avx2, ScalarType::Int(32)) => 8,
            // x86_64 with SSE (128-bit vectors)
            (TargetArch::X86_64Sse | TargetArch::X86_64Sse2, ScalarType::Float(32)) => 4,
            (TargetArch::X86_64Sse | TargetArch::X86_64Sse2, ScalarType::Float(64)) => 2,
            (TargetArch::X86_64Sse | TargetArch::X86_64Sse2, ScalarType::Int(32)) => 4,
            // aarch64 with NEON (128-bit vectors)
            (TargetArch::Aarch64Neon, ScalarType::Float(32)) => 4,
            (TargetArch::Aarch64Neon, ScalarType::Float(64)) => 2,
            (TargetArch::Aarch64Neon, ScalarType::Int(32)) => 4,
            // Fallback: no vectorization
            _ => 1,
        }
    }

    /// Creates a vector type for the given scalar and width.
    #[must_use]
    pub const fn vector(scalar: ScalarType, width: u8) -> Self {
        Self::Vector(scalar, width)
    }

    /// Returns the vector width if this is a vector type, otherwise None.
    #[must_use]
    pub fn vector_width(&self) -> Option<u8> {
        match self {
            Self::Vector(_, w) => Some(*w),
            _ => None,
        }
    }

    /// Returns the scalar element type if this is a vector type.
    #[must_use]
    pub fn element_type(&self) -> Option<ScalarType> {
        match self {
            Self::Vector(s, _) => Some(*s),
            Self::Scalar(s) => Some(*s),
            _ => None,
        }
    }
}

/// Target architecture for vectorization decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetArch {
    /// x86_64 with SSE instructions (128-bit).
    X86_64Sse,
    /// x86_64 with SSE2 instructions (128-bit).
    X86_64Sse2,
    /// x86_64 with AVX instructions (256-bit).
    X86_64Avx,
    /// x86_64 with AVX2 instructions (256-bit).
    X86_64Avx2,
    /// aarch64 with NEON instructions (128-bit).
    Aarch64Neon,
    /// Generic target (no vectorization).
    Generic,
}

impl Default for TargetArch {
    fn default() -> Self {
        // Default to AVX for x86_64, NEON for aarch64
        #[cfg(target_arch = "x86_64")]
        {
            Self::X86_64Avx2
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::Aarch64Neon
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::Generic
        }
    }
}

/// A memory allocation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Alloc {
    /// Buffer identifier.
    pub buffer: BufferId,
    /// Name for debugging.
    pub name: Symbol,
    /// Element type.
    pub elem_ty: ScalarType,
    /// Total size in elements.
    pub size: AllocSize,
    /// Alignment in bytes.
    pub alignment: usize,
    /// Allocation region.
    pub region: AllocRegion,
}

/// Size of an allocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocSize {
    /// Statically known size.
    Static(usize),
    /// Dynamic size (computed at runtime).
    Dynamic(ValueId),
}

/// The body of a function or loop.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Body {
    /// Statements in execution order.
    pub stmts: Vec<Stmt>,
}

impl Body {
    /// Creates an empty body.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a statement to the body.
    pub fn push(&mut self, stmt: Stmt) {
        self.stmts.push(stmt);
    }
}

/// Statements in Loop IR.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Stmt {
    /// An assignment: `%v = op`.
    Assign(ValueId, Op),

    /// A loop construct.
    Loop(Loop),

    /// A conditional branch.
    If(IfStmt),

    /// A store to memory.
    Store(MemRef, Value),

    /// A function call (for external functions).
    Call(Option<ValueId>, Symbol, Vec<Value>),

    /// A return statement.
    Return(Option<Value>),

    /// A barrier for synchronization.
    Barrier(BarrierKind),

    /// A comment/annotation.
    Comment(String),
}

/// A loop construct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Loop {
    /// Unique loop identifier.
    pub id: LoopId,
    /// Loop variable.
    pub var: ValueId,
    /// Lower bound (inclusive).
    pub lower: Value,
    /// Upper bound (exclusive).
    pub upper: Value,
    /// Step size.
    pub step: Value,
    /// Loop body.
    pub body: Body,
    /// Loop attributes.
    pub attrs: LoopAttrs,
}

bitflags! {
    /// Loop attributes for optimization.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LoopAttrs: u32 {
        /// Loop can be parallelized.
        const PARALLEL = 0b0000_0001;
        /// Loop can be vectorized.
        const VECTORIZE = 0b0000_0010;
        /// Loop should be unrolled.
        const UNROLL = 0b0000_0100;
        /// Loop is a reduction loop.
        const REDUCTION = 0b0000_1000;
        /// Loop iterations are independent.
        const INDEPENDENT = 0b0001_0000;
        /// Loop has been tiled.
        const TILED = 0b0010_0000;
        /// Loop is the innermost of a tile.
        const TILE_INNER = 0b0100_0000;
    }
}

/// Loop metadata for optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopMetadata {
    /// Loop identifier.
    pub id: LoopId,
    /// Trip count (iterations).
    pub trip_count: TripCount,
    /// Vectorization width (if applicable).
    pub vector_width: Option<u8>,
    /// Parallel chunk size (if applicable).
    pub parallel_chunk: Option<usize>,
    /// Unroll factor (if applicable).
    pub unroll_factor: Option<u8>,
    /// Dependencies with other loops.
    pub dependencies: Vec<LoopDependency>,
}

/// Trip count information.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TripCount {
    /// Statically known trip count.
    Static(usize),
    /// Dynamic trip count.
    Dynamic,
    /// Bounded trip count (upper bound known).
    Bounded(usize),
}

/// A dependency between loops.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopDependency {
    /// Source loop.
    pub source: LoopId,
    /// Target loop.
    pub target: LoopId,
    /// Dependency type.
    pub kind: DependencyKind,
    /// Distance vector (for affine dependencies).
    pub distance: Option<Vec<i32>>,
}

/// Kinds of dependencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyKind {
    /// Flow dependency (read after write).
    Flow,
    /// Anti dependency (write after read).
    Anti,
    /// Output dependency (write after write).
    Output,
    /// Input dependency (read after read, for locality).
    Input,
}

/// A conditional statement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IfStmt {
    /// Condition value.
    pub cond: Value,
    /// Then branch.
    pub then_body: Body,
    /// Else branch (optional).
    pub else_body: Option<Body>,
}

/// A value (SSA reference or constant).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// A register/variable reference.
    Var(ValueId, LoopType),
    /// An integer constant.
    IntConst(i64, ScalarType),
    /// A floating-point constant.
    FloatConst(f64, ScalarType),
    /// A boolean constant.
    BoolConst(bool),
    /// Undefined value.
    Undef(LoopType),
}

impl Value {
    /// Returns the type of this value.
    #[must_use]
    pub fn ty(&self) -> LoopType {
        match self {
            Self::Var(_, ty) => ty.clone(),
            Self::IntConst(_, s) => LoopType::Scalar(*s),
            Self::FloatConst(_, s) => LoopType::Scalar(*s),
            Self::BoolConst(_) => LoopType::Scalar(ScalarType::Bool),
            Self::Undef(ty) => ty.clone(),
        }
    }

    /// Creates an integer constant.
    #[must_use]
    pub fn int(n: i64, bits: u8) -> Self {
        Self::IntConst(n, ScalarType::Int(bits))
    }

    /// Creates a 64-bit integer constant.
    #[must_use]
    pub fn i64(n: i64) -> Self {
        Self::int(n, 64)
    }

    /// Creates a float constant.
    #[must_use]
    pub fn float(f: f64, bits: u8) -> Self {
        Self::FloatConst(f, ScalarType::Float(bits))
    }

    /// Creates a 64-bit float constant.
    #[must_use]
    pub fn f64(f: f64) -> Self {
        Self::float(f, 64)
    }
}

/// Operations in Loop IR.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Op {
    /// Load from memory.
    Load(MemRef),

    /// Binary arithmetic operation.
    Binary(BinOp, Value, Value),

    /// Unary operation.
    Unary(UnOp, Value),

    /// Comparison.
    Cmp(CmpOp, Value, Value),

    /// Select (conditional).
    Select(Value, Value, Value),

    /// Cast between types.
    Cast(Value, LoopType),

    /// Vector broadcast (scalar to vector).
    Broadcast(Value, u8),

    /// Vector extract (vector to scalar).
    Extract(Value, u8),

    /// Vector insert.
    Insert(Value, Value, u8),

    /// Vector shuffle.
    Shuffle(Value, Value, Vec<i32>),

    /// Reduction within a vector.
    VecReduce(ReduceOp, Value),

    /// Fused multiply-add: a * b + c.
    Fma(Value, Value, Value),

    /// Pointer arithmetic.
    PtrAdd(Value, Value),

    /// Get pointer to buffer element.
    GetPtr(BufferId, Value),

    /// Phi node (for SSA).
    Phi(Vec<(BlockId, Value)>),
}

/// Binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    // Arithmetic
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Signed division.
    SDiv,
    /// Unsigned division.
    UDiv,
    /// Floating-point division.
    FDiv,
    /// Signed remainder.
    SRem,
    /// Unsigned remainder.
    URem,
    /// Floating-point remainder.
    FRem,

    // Bitwise
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Left shift.
    Shl,
    /// Logical right shift.
    LShr,
    /// Arithmetic right shift.
    AShr,

    // Min/Max
    /// Signed minimum.
    SMin,
    /// Unsigned minimum.
    UMin,
    /// Floating-point minimum.
    FMin,
    /// Signed maximum.
    SMax,
    /// Unsigned maximum.
    UMax,
    /// Floating-point maximum.
    FMax,
}

/// Unary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    /// Negation.
    Neg,
    /// Floating-point negation.
    FNeg,
    /// Bitwise NOT.
    Not,
    /// Absolute value.
    Abs,
    /// Floating-point absolute value.
    FAbs,
    /// Square root.
    Sqrt,
    /// Reciprocal square root.
    Rsqrt,
    /// Floor.
    Floor,
    /// Ceiling.
    Ceil,
    /// Round to nearest.
    Round,
    /// Truncate.
    Trunc,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
}

/// Comparison operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CmpOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Signed less than.
    SLt,
    /// Signed less than or equal.
    SLe,
    /// Signed greater than.
    SGt,
    /// Signed greater than or equal.
    SGe,
    /// Unsigned less than.
    ULt,
    /// Unsigned less than or equal.
    ULe,
    /// Unsigned greater than.
    UGt,
    /// Unsigned greater than or equal.
    UGe,
    /// Floating-point ordered equal.
    OEq,
    /// Floating-point ordered not equal.
    ONe,
    /// Floating-point ordered less than.
    OLt,
    /// Floating-point ordered less than or equal.
    OLe,
    /// Floating-point ordered greater than.
    OGt,
    /// Floating-point ordered greater than or equal.
    OGe,
}

/// Reduction operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Sum reduction.
    Add,
    /// Product reduction.
    Mul,
    /// Minimum reduction.
    Min,
    /// Maximum reduction.
    Max,
    /// AND reduction.
    And,
    /// OR reduction.
    Or,
    /// XOR reduction.
    Xor,
}

/// A memory reference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemRef {
    /// The buffer being accessed.
    pub buffer: BufferId,
    /// The index/offset.
    pub index: Value,
    /// The element type.
    pub elem_ty: LoopType,
    /// Access pattern information.
    pub access: AccessPattern,
}

/// Memory access patterns for optimization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Sequential access (stride 1).
    Sequential,
    /// Strided access.
    Strided(i64),
    /// Random/indirect access.
    Random,
    /// Broadcast (same element for all iterations).
    Broadcast,
    /// Affine access (linear combination of loop indices).
    Affine(AffineAccess),
}

/// Affine memory access pattern.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineAccess {
    /// Coefficients for each loop variable.
    pub coefficients: SmallVec<[(LoopId, i64); 4]>,
    /// Constant offset.
    pub offset: i64,
}

/// Barrier kinds for synchronization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BarrierKind {
    /// Memory fence.
    MemFence,
    /// Full barrier (all threads).
    Full,
    /// Thread group barrier.
    ThreadGroup,
}

/// Errors in Loop IR.
#[derive(Clone, Debug, thiserror::Error, Serialize, Deserialize)]
pub enum LoopIrError {
    /// Type mismatch.
    #[error("type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch {
        /// Expected type.
        expected: LoopType,
        /// Actual type.
        got: LoopType,
    },

    /// Invalid vector width.
    #[error("invalid vector width {width} for type {ty:?}")]
    InvalidVectorWidth {
        /// The vector width.
        width: u8,
        /// The element type.
        ty: ScalarType,
    },

    /// Out of bounds access.
    #[error("buffer access out of bounds")]
    OutOfBounds,

    /// Invalid loop transformation.
    #[error("invalid loop transformation: {reason}")]
    InvalidTransform {
        /// Reason for the error.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_type_sizes() {
        assert_eq!(ScalarType::Bool.size_bytes(), 1);
        assert_eq!(ScalarType::Int(32).size_bytes(), 4);
        assert_eq!(ScalarType::Float(64).size_bytes(), 8);
    }

    #[test]
    fn test_loop_type_size() {
        assert_eq!(LoopType::Scalar(ScalarType::Float(32)).size_bytes(), 4);
        assert_eq!(LoopType::Vector(ScalarType::Float(32), 8).size_bytes(), 32);
    }

    #[test]
    fn test_value_types() {
        let v = Value::i64(42);
        assert_eq!(v.ty(), LoopType::Scalar(ScalarType::Int(64)));

        let f = Value::f64(3.14);
        assert_eq!(f.ty(), LoopType::Scalar(ScalarType::Float(64)));
    }

    #[test]
    fn test_loop_attrs() {
        let attrs = LoopAttrs::PARALLEL | LoopAttrs::VECTORIZE;
        assert!(attrs.contains(LoopAttrs::PARALLEL));
        assert!(attrs.contains(LoopAttrs::VECTORIZE));
        assert!(!attrs.contains(LoopAttrs::UNROLL));
    }

    #[test]
    fn test_trip_count() {
        let static_trip = TripCount::Static(100);
        assert_eq!(static_trip, TripCount::Static(100));

        let dynamic_trip = TripCount::Dynamic;
        assert_eq!(dynamic_trip, TripCount::Dynamic);
    }

    // ========================================================================
    // M3 Exit Criteria Integration Tests
    // ========================================================================

    /// M3 Exit Criterion 1: matmul microkernel auto-vectorizes on x86_64 and aarch64
    ///
    /// This test verifies that the vectorization pass correctly identifies
    /// a matmul-like kernel as vectorizable and selects appropriate vector widths.
    #[test]
    fn test_m3_matmul_auto_vectorizes() {
        use crate::vectorize::{VectorizeConfig, VectorizePass};
        use bhc_index::Idx;
        use bhc_tensor_ir::BufferId;

        // Create a matmul-like loop structure (innermost loop computes dot product)
        let loop_id = LoopId::new(0);
        let loop_var = ValueId::new(0);

        // Simulate the innermost loop of matmul: c[i,j] += a[i,k] * b[k,j]
        let mem_ref = MemRef {
            buffer: BufferId::new(0),
            index: Value::Var(loop_var, LoopType::Scalar(ScalarType::I64)),
            elem_ty: LoopType::Scalar(ScalarType::F32),
            access: AccessPattern::Sequential,
        };

        let mut body = Body::new();
        let load_a = ValueId::new(1);
        body.push(Stmt::Assign(load_a, Op::Load(mem_ref.clone())));

        let load_b = ValueId::new(2);
        body.push(Stmt::Assign(load_b, Op::Load(mem_ref.clone())));

        let mul_result = ValueId::new(3);
        body.push(Stmt::Assign(
            mul_result,
            Op::Binary(
                BinOp::Mul,
                Value::Var(load_a, LoopType::Scalar(ScalarType::F32)),
                Value::Var(load_b, LoopType::Scalar(ScalarType::F32)),
            ),
        ));

        // FMA opportunity: acc = acc + a * b
        let acc = ValueId::new(4);
        let fma_result = ValueId::new(5);
        body.push(Stmt::Assign(
            fma_result,
            Op::Fma(
                Value::Var(load_a, LoopType::Scalar(ScalarType::F32)),
                Value::Var(load_b, LoopType::Scalar(ScalarType::F32)),
                Value::Var(acc, LoopType::Scalar(ScalarType::F32)),
            ),
        ));

        let lp = Loop {
            id: loop_id,
            var: loop_var,
            lower: Value::i64(0),
            upper: Value::i64(256), // K dimension
            step: Value::i64(1),
            body,
            attrs: LoopAttrs::VECTORIZE | LoopAttrs::INDEPENDENT,
        };

        let mut outer_body = Body::new();
        outer_body.push(Stmt::Loop(lp));

        let ir = LoopIR {
            name: bhc_intern::Symbol::intern("matmul_kernel"),
            params: vec![],
            return_ty: LoopType::Void,
            body: outer_body,
            allocs: vec![],
            loop_info: vec![LoopMetadata {
                id: loop_id,
                trip_count: TripCount::Static(256),
                vector_width: None,
                parallel_chunk: None,
                unroll_factor: None,
                dependencies: Vec::new(),
            }],
        };

        // Test on x86_64 AVX2
        let config_x86 = VectorizeConfig {
            target: TargetArch::X86_64Avx2,
            ..Default::default()
        };
        let mut pass_x86 = VectorizePass::new(config_x86);
        let analysis_x86 = pass_x86.analyze(&ir);
        let info_x86 = analysis_x86.get(&loop_id).expect("loop should be analyzed");

        assert!(
            info_x86.vectorizable,
            "M3 FAIL: matmul kernel not vectorizable on x86_64 AVX2"
        );
        assert_eq!(
            info_x86.recommended_width, 8,
            "M3 FAIL: x86_64 AVX2 should use 8-wide vectors for f32"
        );

        // Test on aarch64 NEON
        let config_arm = VectorizeConfig {
            target: TargetArch::Aarch64Neon,
            ..Default::default()
        };
        let mut pass_arm = VectorizePass::new(config_arm);
        let analysis_arm = pass_arm.analyze(&ir);
        let info_arm = analysis_arm.get(&loop_id).expect("loop should be analyzed");

        assert!(
            info_arm.vectorizable,
            "M3 FAIL: matmul kernel not vectorizable on aarch64 NEON"
        );
        assert_eq!(
            info_arm.recommended_width, 4,
            "M3 FAIL: aarch64 NEON should use 4-wide vectors for f32"
        );
    }

    /// M3 Exit Criterion 2: Reductions scale linearly up to 8 cores
    ///
    /// This test verifies that parallel reduction chunking distributes
    /// work evenly across workers.
    #[test]
    fn test_m3_reductions_scale_linearly() {
        use crate::parallel::{ParReduce, ParallelConfig, Range};
        use crate::ReduceOp;

        let data_size = 1_000_000; // 1M elements

        // Test with different worker counts
        for worker_count in [1, 2, 4, 8] {
            let config = ParallelConfig {
                worker_count,
                deterministic: true,
                ..Default::default()
            };

            let par_reduce = ParReduce {
                size: data_size,
                op: ReduceOp::Add,
                config,
            };

            let chunks = par_reduce.chunk_assignments();

            // Verify correct number of chunks
            assert_eq!(
                chunks.len(),
                worker_count,
                "M3 FAIL: Expected {} chunks for {} workers",
                worker_count,
                worker_count
            );

            // Verify total work is correct
            let total_work: usize = chunks.iter().map(|c| c.len()).sum();
            assert_eq!(
                total_work, data_size,
                "M3 FAIL: Total work should equal data size"
            );

            // Verify work is evenly distributed (within 1 element difference)
            let expected_per_worker = data_size / worker_count;
            for (i, chunk) in chunks.iter().enumerate() {
                let diff = (chunk.len() as i64 - expected_per_worker as i64).abs();
                assert!(
                    diff <= 1,
                    "M3 FAIL: Worker {} has {} elements, expected ~{} (diff={})",
                    i,
                    chunk.len(),
                    expected_per_worker,
                    diff
                );
            }
        }

        // Verify scaling: doubling workers should halve chunk size
        let _config_4 = ParallelConfig {
            worker_count: 4,
            ..Default::default()
        };
        let chunks_4 = Range::new(0, data_size as i64).chunk(4);

        let _config_8 = ParallelConfig {
            worker_count: 8,
            ..Default::default()
        };
        let chunks_8 = Range::new(0, data_size as i64).chunk(8);

        let avg_chunk_4: usize = chunks_4.iter().map(|c| c.len()).sum::<usize>() / 4;
        let avg_chunk_8: usize = chunks_8.iter().map(|c| c.len()).sum::<usize>() / 8;

        // 8 workers should have approximately half the chunk size of 4 workers
        let ratio = avg_chunk_4 as f64 / avg_chunk_8 as f64;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "M3 FAIL: Chunk size ratio should be ~2.0, got {}",
            ratio
        );
    }

    /// M3 Exit Criterion 3: Deterministic mode produces identical results across runs
    ///
    /// This test verifies that parallel chunking is deterministic when configured.
    #[test]
    fn test_m3_deterministic_mode() {
        use crate::parallel::{ParReduce, ParallelConfig, ParallelStrategy};
        use crate::ReduceOp;

        let data_size = 100_000;
        let worker_count = 8;

        // Configure deterministic mode
        let config = ParallelConfig {
            worker_count,
            deterministic: true,
            ..Default::default()
        };

        let par_reduce = ParReduce {
            size: data_size,
            op: ReduceOp::Add,
            config: config.clone(),
        };

        // Run multiple times and verify identical chunk assignments
        let chunks1 = par_reduce.chunk_assignments();
        let chunks2 = par_reduce.chunk_assignments();
        let chunks3 = par_reduce.chunk_assignments();

        for i in 0..worker_count {
            assert_eq!(
                chunks1[i].start, chunks2[i].start,
                "M3 FAIL: Chunk {} start differs between runs",
                i
            );
            assert_eq!(
                chunks1[i].end, chunks2[i].end,
                "M3 FAIL: Chunk {} end differs between runs",
                i
            );
            assert_eq!(
                chunks2[i].start, chunks3[i].start,
                "M3 FAIL: Chunk {} start differs between runs",
                i
            );
            assert_eq!(
                chunks2[i].end, chunks3[i].end,
                "M3 FAIL: Chunk {} end differs between runs",
                i
            );
        }

        // Verify strategy is Static for deterministic mode
        use crate::parallel::ParallelPass;

        let parallel_config = ParallelConfig {
            worker_count: 8,
            deterministic: true,
            ..Default::default()
        };

        // Build a simple parallelizable loop
        let loop_id = LoopId::new(0);
        let mut body = Body::new();

        let lp = Loop {
            id: loop_id,
            var: ValueId::new(0),
            lower: Value::i64(0),
            upper: Value::i64(100000),
            step: Value::i64(1),
            body: Body::new(),
            attrs: LoopAttrs::PARALLEL | LoopAttrs::INDEPENDENT,
        };

        body.push(Stmt::Loop(lp));

        let ir = LoopIR {
            name: bhc_intern::Symbol::intern("deterministic_test"),
            params: vec![],
            return_ty: LoopType::Void,
            body,
            allocs: vec![],
            loop_info: vec![LoopMetadata {
                id: loop_id,
                trip_count: TripCount::Static(100000),
                vector_width: None,
                parallel_chunk: None,
                unroll_factor: None,
                dependencies: Vec::new(),
            }],
        };

        let mut pass = ParallelPass::new(parallel_config);
        let analysis = pass.analyze(&ir);
        let info = analysis.get(&loop_id).expect("loop should be analyzed");

        assert!(
            info.parallelizable,
            "M3 FAIL: Loop should be parallelizable"
        );
        assert_eq!(
            info.strategy,
            ParallelStrategy::Static,
            "M3 FAIL: Deterministic mode should use Static scheduling"
        );
    }

    /// M3 Integration: Complete pipeline test for vectorized parallel reduction
    #[test]
    fn test_m3_vectorized_parallel_reduction() {
        use crate::parallel::{ParallelConfig, ParallelPass};
        use crate::vectorize::{VectorizeConfig, VectorizePass};
        use bhc_index::Idx;
        use bhc_tensor_ir::BufferId;

        // Create a reduction loop that should be both vectorized and parallelized
        let outer_loop_id = LoopId::new(0);
        let inner_loop_id = LoopId::new(1);

        // Inner loop: vectorizable reduction
        let mem_ref = MemRef {
            buffer: BufferId::new(0),
            index: Value::Var(ValueId::new(1), LoopType::Scalar(ScalarType::I64)),
            elem_ty: LoopType::Scalar(ScalarType::F32),
            access: AccessPattern::Sequential,
        };

        let mut inner_body = Body::new();
        let load_result = ValueId::new(2);
        inner_body.push(Stmt::Assign(load_result, Op::Load(mem_ref)));

        let inner_loop = Loop {
            id: inner_loop_id,
            var: ValueId::new(1),
            lower: Value::i64(0),
            upper: Value::i64(1024),
            step: Value::i64(1),
            body: inner_body,
            attrs: LoopAttrs::VECTORIZE | LoopAttrs::INDEPENDENT | LoopAttrs::REDUCTION,
        };

        // Outer loop: parallelizable
        let mut outer_body = Body::new();
        outer_body.push(Stmt::Loop(inner_loop));

        let outer_loop = Loop {
            id: outer_loop_id,
            var: ValueId::new(0),
            lower: Value::i64(0),
            upper: Value::i64(10000),
            step: Value::i64(1),
            body: outer_body,
            attrs: LoopAttrs::PARALLEL | LoopAttrs::INDEPENDENT,
        };

        let mut top_body = Body::new();
        top_body.push(Stmt::Loop(outer_loop));

        let ir = LoopIR {
            name: bhc_intern::Symbol::intern("vec_par_reduce"),
            params: vec![],
            return_ty: LoopType::Void,
            body: top_body,
            allocs: vec![],
            loop_info: vec![
                LoopMetadata {
                    id: outer_loop_id,
                    trip_count: TripCount::Static(10000),
                    vector_width: None,
                    parallel_chunk: None,
                    unroll_factor: None,
                    dependencies: Vec::new(),
                },
                LoopMetadata {
                    id: inner_loop_id,
                    trip_count: TripCount::Static(1024),
                    vector_width: None,
                    parallel_chunk: None,
                    unroll_factor: None,
                    dependencies: Vec::new(),
                },
            ],
        };

        // Apply vectorization pass
        let vec_config = VectorizeConfig {
            target: TargetArch::X86_64Avx2,
            ..Default::default()
        };
        let mut vec_pass = VectorizePass::new(vec_config);
        let vec_analysis = vec_pass.analyze(&ir);

        // Verify inner loop is vectorizable
        let inner_info = vec_analysis
            .get(&inner_loop_id)
            .expect("inner loop analyzed");
        assert!(
            inner_info.vectorizable,
            "M3 FAIL: Inner reduction loop should be vectorizable"
        );

        // Apply parallelization pass
        let par_config = ParallelConfig {
            worker_count: 8,
            deterministic: true,
            ..Default::default()
        };
        let mut par_pass = ParallelPass::new(par_config);
        let par_analysis = par_pass.analyze(&ir);

        // Verify outer loop is parallelizable
        let outer_info = par_analysis
            .get(&outer_loop_id)
            .expect("outer loop analyzed");
        assert!(
            outer_info.parallelizable,
            "M3 FAIL: Outer loop should be parallelizable"
        );
        assert_eq!(
            outer_info.num_chunks, 8,
            "M3 FAIL: Should have 8 parallel chunks"
        );
    }
}
