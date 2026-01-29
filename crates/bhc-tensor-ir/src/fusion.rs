//! Tensor fusion pass.
//!
//! This module implements guaranteed fusion for tensor operations per
//! H26-SPEC Section 8. The fusion pass transforms a sequence of tensor
//! operations into fused kernels that execute without intermediate allocation.
//!
//! ## Guaranteed Fusion Patterns (H26-SPEC Section 8.1)
//!
//! These patterns MUST fuse by specification:
//!
//! 1. `map f (map g x)` → single traversal with composed function
//! 2. `zipWith f (map g a) (map h b)` → single traversal
//! 3. `sum (map f x)` → single traversal (reduce of map)
//! 4. `foldl' op z (map f x)` → single traversal
//!
//! ## Fusion Algorithm
//!
//! The pass operates in three phases:
//!
//! 1. **Pattern Detection**: Identify fusible patterns in the IR
//! 2. **Fusion Graph Construction**: Build a graph of fusion opportunities
//! 3. **Kernel Generation**: Generate fused kernels with tracking info

use crate::{
    AllocInfo, AllocRegion, Axis, BinaryOp, BufferId, DType, FusionDecision, FusionInfo, Kernel,
    KernelBody, KernelId, MapFn, ReduceOp, Shape, Strides, TensorId, TensorMeta, TensorOp,
    TensorRef, ZipFn,
};
use bhc_index::Idx;
use bhc_intern::Symbol;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// The fusion context tracks state during fusion analysis.
pub struct FusionContext {
    /// Next kernel ID to allocate.
    next_kernel_id: u32,
    /// Next tensor ID to allocate.
    next_tensor_id: u32,
    /// Next buffer ID to allocate.
    next_buffer_id: u32,
    /// Tensor reference counts (for multi-use detection).
    ref_counts: FxHashMap<TensorId, usize>,
    /// Generated kernels.
    kernels: Vec<Kernel>,
    /// Fusion decisions for reporting.
    decisions: Vec<FusionDecision>,
    /// Whether we're in strict mode (Numeric profile).
    strict_mode: bool,
}

impl FusionContext {
    /// Creates a new fusion context.
    #[must_use]
    pub fn new(strict_mode: bool) -> Self {
        Self {
            next_kernel_id: 0,
            next_tensor_id: 0,
            next_buffer_id: 0,
            ref_counts: FxHashMap::default(),
            kernels: Vec::new(),
            decisions: Vec::new(),
            strict_mode,
        }
    }

    /// Allocates a fresh kernel ID.
    fn fresh_kernel_id(&mut self) -> KernelId {
        let id = KernelId::new(self.next_kernel_id as usize);
        self.next_kernel_id += 1;
        id
    }

    /// Allocates a fresh tensor ID.
    fn fresh_tensor_id(&mut self) -> TensorId {
        let id = TensorId::new(self.next_tensor_id as usize);
        self.next_tensor_id += 1;
        id
    }

    /// Allocates a fresh buffer ID.
    fn fresh_buffer_id(&mut self) -> BufferId {
        let id = BufferId::new(self.next_buffer_id as usize);
        self.next_buffer_id += 1;
        id
    }

    /// Increments the reference count for a tensor.
    fn add_ref(&mut self, id: TensorId) {
        *self.ref_counts.entry(id).or_insert(0) += 1;
    }

    /// Returns the reference count for a tensor.
    fn ref_count(&self, id: TensorId) -> usize {
        self.ref_counts.get(&id).copied().unwrap_or(0)
    }

    /// Returns the generated kernels.
    #[must_use]
    pub fn kernels(&self) -> &[Kernel] {
        &self.kernels
    }

    /// Returns the fusion decisions for reporting.
    #[must_use]
    pub fn decisions(&self) -> &[FusionDecision] {
        &self.decisions
    }

    /// Consumes the context and returns the generated kernels.
    #[must_use]
    pub fn into_kernels(self) -> Vec<Kernel> {
        self.kernels
    }
}

/// A fusible operation in the fusion graph.
#[derive(Clone, Debug)]
pub struct FusibleOp {
    /// The original tensor operation.
    pub op: TensorOp,
    /// The output tensor reference.
    pub output: TensorRef,
    /// Input tensor IDs.
    pub inputs: SmallVec<[TensorId; 2]>,
    /// Whether this op has been fused.
    pub fused: bool,
}

/// Result of fusion pattern matching.
///
/// These patterns correspond to the guaranteed fusion patterns from H26-SPEC Section 8.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum FusionPattern {
    /// Pattern 1: `map f (map g x)` - composable maps fused to single traversal.
    MapMap {
        /// The outer map function (f).
        outer_fn: MapFn,
        /// The inner map function (g).
        inner_fn: MapFn,
        /// The input tensor reference.
        input: TensorRef,
    },
    /// Pattern 2: `zipWith f (map g a) (map h b)` - zip of maps fused to single traversal.
    ZipWithMaps {
        /// The combining function for zipWith.
        zip_fn: ZipFn,
        /// Optional map function applied to left input.
        left_fn: Option<MapFn>,
        /// Left input tensor reference.
        left_input: TensorRef,
        /// Optional map function applied to right input.
        right_fn: Option<MapFn>,
        /// Right input tensor reference.
        right_input: TensorRef,
    },
    /// Pattern 3: `reduce op (map f x)` - reduction of map fused to single traversal.
    ReduceMap {
        /// The reduction operation (sum, product, etc.).
        reduce_op: ReduceOp,
        /// Optional axis for the reduction.
        axis: Option<Axis>,
        /// The map function applied before reduction.
        map_fn: MapFn,
        /// The input tensor reference.
        input: TensorRef,
    },
    /// Pattern 4: `foldl' op z (map f x)` - fold of map fused to single traversal.
    FoldMap {
        /// The fold combining function.
        fold_fn: Symbol,
        /// Initial accumulator value.
        init: TensorRef,
        /// The map function applied during fold.
        map_fn: MapFn,
        /// The input tensor reference.
        input: TensorRef,
    },

    // ========================================================================
    // ML-Specific Fusion Patterns (H26-SPEC Section 8.2)
    // ========================================================================
    /// Pattern 5: Softmax fusion - numerically stable single-kernel softmax.
    ///
    /// Fuses the pattern:
    /// ```text
    /// let maxVal = maximum xs
    ///     shifted = map (\x -> x - maxVal) xs
    ///     exps = map exp shifted
    ///     total = sum exps
    /// in map (/ total) exps
    /// ```
    ///
    /// Compiles to: Two passes (max-finding pass, then exp/sum/divide pass)
    /// for numerical stability.
    Softmax {
        /// Input tensor reference.
        input: TensorRef,
        /// Optional axis for row-wise softmax (None = flatten).
        axis: Option<Axis>,
    },

    /// Pattern 6: Log-softmax fusion - numerically stable log-softmax.
    ///
    /// Fuses the pattern:
    /// ```text
    /// let maxVal = maximum xs
    ///     shifted = map (\x -> x - maxVal) xs
    ///     logSumExp = log (sum (map exp shifted))
    /// in map (\x -> x - maxVal - logSumExp) xs
    /// ```
    ///
    /// More numerically stable than log(softmax(x)) computed separately.
    LogSoftmax {
        /// Input tensor reference.
        input: TensorRef,
        /// Optional axis for row-wise log-softmax.
        axis: Option<Axis>,
    },

    /// Pattern 7: Layer normalization fusion - single-pass Welford algorithm.
    ///
    /// Fuses the pattern:
    /// ```text
    /// let mu = mean xs
    ///     variance = mean (map (\x -> (x - mu)^2) xs)
    /// in map (\x -> (x - mu) / sqrt (variance + eps)) xs
    /// ```
    ///
    /// Compiles to: Single pass using Welford's online algorithm for
    /// numerically stable mean and variance computation.
    LayerNorm {
        /// Input tensor reference.
        input: TensorRef,
        /// Epsilon for numerical stability.
        epsilon: f64,
        /// Optional axis for normalization (None = normalize all elements).
        axis: Option<Axis>,
        /// Optional affine scale parameter (gamma).
        scale: Option<TensorRef>,
        /// Optional affine bias parameter (beta).
        bias: Option<TensorRef>,
    },

    /// Pattern 8: RMS normalization fusion.
    ///
    /// Fuses the pattern:
    /// ```text
    /// let rms = sqrt (mean (map (\x -> x^2) xs))
    /// in map (\x -> x / rms) xs
    /// ```
    RMSNorm {
        /// Input tensor reference.
        input: TensorRef,
        /// Epsilon for numerical stability.
        epsilon: f64,
        /// Optional scale parameter.
        scale: Option<TensorRef>,
    },

    /// Pattern 9: Scaled dot-product attention fusion.
    ///
    /// Fuses the pattern:
    /// ```text
    /// let scale = 1.0 / sqrt d
    ///     scores = matmul q (transpose k) * scale
    ///     weights = softmax scores
    /// in matmul weights v
    /// ```
    ///
    /// Compiles to: Fused attention kernel with optional memory-efficient
    /// tiling for long sequences.
    Attention {
        /// Query tensor [batch, heads, seq_len, head_dim].
        query: TensorRef,
        /// Key tensor [batch, heads, seq_len, head_dim].
        key: TensorRef,
        /// Value tensor [batch, heads, seq_len, head_dim].
        value: TensorRef,
        /// Optional attention mask.
        mask: Option<TensorRef>,
        /// Scale factor (typically 1/sqrt(head_dim)).
        scale: f64,
        /// Whether to use causal (autoregressive) masking.
        causal: bool,
    },

    /// Pattern 10: GELU activation fusion.
    ///
    /// Fuses the pattern:
    /// ```text
    /// gelu x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// ```
    ///
    /// Or the faster approximate:
    /// ```text
    /// gelu_fast x = x * sigmoid(1.702 * x)
    /// ```
    Gelu {
        /// Input tensor reference.
        input: TensorRef,
        /// Whether to use the fast approximation.
        fast: bool,
    },

    /// Pattern 11: SiLU/Swish activation fusion.
    ///
    /// Fuses the pattern: `silu x = x * sigmoid(x)`
    Silu {
        /// Input tensor reference.
        input: TensorRef,
    },

    /// Pattern 12: Fused linear + activation.
    ///
    /// Fuses matmul with bias addition and activation:
    /// ```text
    /// activation (matmul x w + b)
    /// ```
    FusedLinear {
        /// Input tensor reference.
        input: TensorRef,
        /// Weight matrix.
        weight: TensorRef,
        /// Optional bias vector.
        bias: Option<TensorRef>,
        /// Optional activation function.
        activation: Option<FusedActivation>,
    },
}

/// Activation functions that can be fused with linear operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FusedActivation {
    /// ReLU: max(0, x)
    Relu,
    /// GELU (standard)
    Gelu,
    /// GELU (fast approximation)
    GeluFast,
    /// SiLU/Swish: x * sigmoid(x)
    Silu,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh
    Tanh,
}

/// Fuses a sequence of tensor operations into kernels.
///
/// This is the main entry point for the fusion pass.
pub fn fuse_ops(ctx: &mut FusionContext, ops: Vec<TensorOp>) -> Vec<Kernel> {
    // Phase 1: Build the operation graph and count references
    let fusible_ops = build_fusible_ops(ctx, &ops);

    // Phase 2: Detect and apply fusion patterns
    let fused_groups = detect_and_fuse(ctx, fusible_ops);

    // Phase 3: Generate kernels from fused groups
    for group in fused_groups {
        let kernel = generate_kernel(ctx, group);
        ctx.kernels.push(kernel);
    }

    ctx.kernels.clone()
}

/// Builds fusible operation wrappers and counts references.
fn build_fusible_ops(ctx: &mut FusionContext, ops: &[TensorOp]) -> Vec<FusibleOp> {
    let mut fusible = Vec::with_capacity(ops.len());

    for op in ops {
        // Extract input tensor IDs
        let inputs = extract_input_ids(op);

        // Count references to inputs
        for &id in &inputs {
            ctx.add_ref(id);
        }

        // Create output tensor reference
        let output = create_output_ref(ctx, op);

        fusible.push(FusibleOp {
            op: op.clone(),
            output,
            inputs,
            fused: false,
        });
    }

    fusible
}

/// Extracts input tensor IDs from an operation.
fn extract_input_ids(op: &TensorOp) -> SmallVec<[TensorId; 2]> {
    match op {
        TensorOp::Constant(_) => SmallVec::new(),
        TensorOp::Unary(_, t) | TensorOp::Map(_, t) => smallvec::smallvec![t.id],
        TensorOp::Binary(_, t1, t2) | TensorOp::ZipWith(_, t1, t2) => {
            smallvec::smallvec![t1.id, t2.id]
        }
        TensorOp::Reduce(_, _, t) | TensorOp::ReduceAll(_, t) | TensorOp::Scan(_, _, t) => {
            smallvec::smallvec![t.id]
        }
        TensorOp::Fold(_, init, t) => smallvec::smallvec![init.id, t.id],
        TensorOp::Reshape(_, t)
        | TensorOp::Slice(_, t)
        | TensorOp::Transpose(_, t)
        | TensorOp::Broadcast(_, t) => smallvec::smallvec![t.id],
        TensorOp::Concat(_, refs) => refs.iter().map(|r| r.id).collect(),
        TensorOp::Split(_, _, t) => smallvec::smallvec![t.id],
        TensorOp::MatMul(t1, t2)
        | TensorOp::BatchMatMul(t1, t2)
        | TensorOp::Dot(t1, t2)
        | TensorOp::Outer(t1, t2) => smallvec::smallvec![t1.id, t2.id],
        TensorOp::Conv(_, t1, t2) => smallvec::smallvec![t1.id, t2.id],
        TensorOp::Gather(_, idx, t) => smallvec::smallvec![idx.id, t.id],
        TensorOp::Scatter(_, idx, src, dst) => smallvec::smallvec![idx.id, src.id, dst.id],
    }
}

/// Creates an output tensor reference for an operation.
fn create_output_ref(ctx: &mut FusionContext, op: &TensorOp) -> TensorRef {
    let id = ctx.fresh_tensor_id();
    let meta = infer_output_meta(op);
    TensorRef { id, meta }
}

/// Infers output metadata from an operation.
fn infer_output_meta(op: &TensorOp) -> TensorMeta {
    match op {
        TensorOp::Constant(c) => match c {
            crate::ConstantOp::Zeros(m)
            | crate::ConstantOp::Ones(m)
            | crate::ConstantOp::Full(m, _) => m.clone(),
            crate::ConstantOp::Range(dtype, start, stop, step) => {
                let count = ((stop - start) / step) as usize;
                TensorMeta::new_contiguous(*dtype, Shape::from_static([count]))
                    .unwrap_or_else(|| default_meta(*dtype))
            }
            crate::ConstantOp::Eye(dtype, n) => {
                TensorMeta::new_contiguous(*dtype, Shape::from_static([*n, *n]))
                    .unwrap_or_else(|| default_meta(*dtype))
            }
        },
        TensorOp::Unary(_, t) | TensorOp::Map(_, t) => {
            // Preserves shape and dtype
            t.meta.clone()
        }
        TensorOp::Binary(_, t1, _) | TensorOp::ZipWith(_, t1, _) => {
            // Result has same shape as inputs (assuming broadcast done)
            t1.meta.clone()
        }
        TensorOp::Reduce(_, axis, t) => {
            // Remove the reduced axis
            let mut dims: SmallVec<[crate::Dim; 4]> = t.meta.shape.dims().iter().cloned().collect();
            if let Some(idx) = axis.normalize(dims.len()) {
                dims.remove(idx);
            }
            let shape = Shape::new(dims);
            TensorMeta::new_contiguous(t.meta.dtype, shape).unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::ReduceAll(_, t) => {
            // Scalar output
            TensorMeta::new_contiguous(t.meta.dtype, Shape::scalar())
                .unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::Scan(_, _, t) => {
            // Same shape as input
            t.meta.clone()
        }
        TensorOp::Fold(_, _, t) => {
            // Scalar output
            TensorMeta::new_contiguous(t.meta.dtype, Shape::scalar())
                .unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::Reshape(shape, t) => TensorMeta::new_contiguous(t.meta.dtype, shape.clone())
            .unwrap_or_else(|| t.meta.clone()),
        TensorOp::Slice(spec, t) => {
            // Compute sliced shape
            let mut new_dims: SmallVec<[crate::Dim; 4]> = SmallVec::new();
            for (i, range) in spec.ranges.iter().enumerate() {
                if let Some(dim) = t.meta.shape.dims().get(i) {
                    if let Some(n) = dim.static_value() {
                        let start = range.start.unwrap_or(0);
                        let stop = range.stop.unwrap_or(n as i64);
                        let step = range.step;
                        let count = ((stop - start) / step) as usize;
                        new_dims.push(crate::Dim::Static(count));
                    } else {
                        new_dims.push(dim.clone());
                    }
                }
            }
            TensorMeta::new_contiguous(t.meta.dtype, Shape::new(new_dims))
                .unwrap_or_else(|| t.meta.clone())
        }
        TensorOp::Transpose(perm, t) => {
            // Permute dimensions
            let old_dims = t.meta.shape.dims();
            let new_dims: SmallVec<[crate::Dim; 4]> = perm
                .as_slice()
                .iter()
                .map(|&i| old_dims[i].clone())
                .collect();
            // Note: Transpose creates strided layout, not contiguous
            let shape = Shape::new(new_dims);
            TensorMeta {
                dtype: t.meta.dtype,
                shape: shape.clone(),
                strides: Strides::new(perm.as_slice().iter().map(|&i| t.meta.strides.values()[i])),
                layout: crate::Layout::Strided,
                // Views alias the underlying buffer
                alias: t.meta.alias,
            }
        }
        TensorOp::Broadcast(shape, t) => TensorMeta::new_contiguous(t.meta.dtype, shape.clone())
            .unwrap_or_else(|| t.meta.clone()),
        TensorOp::Concat(_, refs) => {
            if let Some(first) = refs.first() {
                first.meta.clone()
            } else {
                default_meta(DType::Float32)
            }
        }
        TensorOp::Split(_, _, t) => t.meta.clone(),
        TensorOp::MatMul(a, b) | TensorOp::BatchMatMul(a, b) => {
            // [M, K] @ [K, N] -> [M, N]
            let a_dims = a.meta.shape.dims();
            let b_dims = b.meta.shape.dims();
            if a_dims.len() >= 2 && b_dims.len() >= 2 {
                let m = a_dims[a_dims.len() - 2].clone();
                let n = b_dims[b_dims.len() - 1].clone();
                TensorMeta::new_contiguous(a.meta.dtype, Shape::new([m, n]))
                    .unwrap_or_else(|| a.meta.clone())
            } else {
                a.meta.clone()
            }
        }
        TensorOp::Dot(_, t) => TensorMeta::new_contiguous(t.meta.dtype, Shape::scalar())
            .unwrap_or_else(|| t.meta.clone()),
        TensorOp::Outer(a, b) => {
            // [M] outer [N] -> [M, N]
            let m = a
                .meta
                .shape
                .dims()
                .first()
                .cloned()
                .unwrap_or(crate::Dim::Static(1));
            let n = b
                .meta
                .shape
                .dims()
                .first()
                .cloned()
                .unwrap_or(crate::Dim::Static(1));
            TensorMeta::new_contiguous(a.meta.dtype, Shape::new([m, n]))
                .unwrap_or_else(|| a.meta.clone())
        }
        TensorOp::Conv(_, input, _) => {
            // Simplified: same as input for now
            input.meta.clone()
        }
        TensorOp::Gather(_, _, data) => data.meta.clone(),
        TensorOp::Scatter(_, _, _, dst) => dst.meta.clone(),
    }
}

/// Creates a default metadata for error cases.
fn default_meta(dtype: DType) -> TensorMeta {
    TensorMeta {
        dtype,
        shape: Shape::scalar(),
        strides: Strides::new([]),
        layout: crate::Layout::Contiguous,
        alias: None,
    }
}

/// A group of fused operations.
#[derive(Clone, Debug)]
pub struct FusedGroup {
    /// The operations in this fused group.
    pub ops: Vec<TensorOp>,
    /// Input tensor references.
    pub inputs: Vec<TensorRef>,
    /// Output tensor reference.
    pub output: TensorRef,
    /// The pattern that was fused.
    pub pattern: Option<FusionPattern>,
    /// Names of fused operations (for reporting).
    pub op_names: Vec<Symbol>,
}

/// Detects and applies fusion patterns.
fn detect_and_fuse(ctx: &mut FusionContext, mut ops: Vec<FusibleOp>) -> Vec<FusedGroup> {
    let mut groups = Vec::new();

    // Process operations in reverse order to find consumers first
    let mut i = ops.len();
    while i > 0 {
        i -= 1;

        if ops[i].fused {
            continue;
        }

        // Try to find a fusible pattern
        if let Some((pattern, consumed_indices)) = find_fusion_pattern(ctx, &ops, i) {
            // Mark consumed operations as fused
            for &idx in &consumed_indices {
                ops[idx].fused = true;
            }

            // Create fused group
            let group = create_fused_group(ctx, &ops, &consumed_indices, pattern);

            // Record fusion decision
            ctx.decisions
                .push(FusionDecision::Fused(group.op_names.clone()));

            groups.push(group);
        } else {
            // No fusion - create single-op group
            ops[i].fused = true;
            let group = FusedGroup {
                ops: vec![ops[i].op.clone()],
                inputs: ops[i]
                    .inputs
                    .iter()
                    .map(|&id| TensorRef {
                        id,
                        meta: default_meta(DType::Float32),
                    })
                    .collect(),
                output: ops[i].output.clone(),
                pattern: None,
                op_names: vec![op_name(&ops[i].op)],
            };
            groups.push(group);
        }
    }

    groups.reverse();
    groups
}

/// Finds a fusion pattern starting from the given operation index.
fn find_fusion_pattern(
    ctx: &FusionContext,
    ops: &[FusibleOp],
    consumer_idx: usize,
) -> Option<(FusionPattern, Vec<usize>)> {
    let consumer = &ops[consumer_idx];

    // Pattern 1: map f (map g x)
    if let TensorOp::Map(outer_fn, inner_ref) = &consumer.op {
        // Check if the input is a single-use map
        if let Some(producer_idx) = find_producer(ops, inner_ref.id) {
            if !ops[producer_idx].fused && ctx.ref_count(inner_ref.id) == 1 {
                if let TensorOp::Map(inner_fn, input_ref) = &ops[producer_idx].op {
                    return Some((
                        FusionPattern::MapMap {
                            outer_fn: outer_fn.clone(),
                            inner_fn: inner_fn.clone(),
                            input: input_ref.clone(),
                        },
                        vec![consumer_idx, producer_idx],
                    ));
                }
            }
        }
    }

    // Pattern 3: reduce op (map f x) - sum/prod/max/min of map
    if let TensorOp::ReduceAll(reduce_op, inner_ref) = &consumer.op {
        if let Some(producer_idx) = find_producer(ops, inner_ref.id) {
            if !ops[producer_idx].fused && ctx.ref_count(inner_ref.id) == 1 {
                if let TensorOp::Map(map_fn, input_ref) = &ops[producer_idx].op {
                    return Some((
                        FusionPattern::ReduceMap {
                            reduce_op: *reduce_op,
                            axis: None,
                            map_fn: map_fn.clone(),
                            input: input_ref.clone(),
                        },
                        vec![consumer_idx, producer_idx],
                    ));
                }
            }
        }
    }

    // Pattern 3 variant: reduce with axis
    if let TensorOp::Reduce(reduce_op, axis, inner_ref) = &consumer.op {
        if let Some(producer_idx) = find_producer(ops, inner_ref.id) {
            if !ops[producer_idx].fused && ctx.ref_count(inner_ref.id) == 1 {
                if let TensorOp::Map(map_fn, input_ref) = &ops[producer_idx].op {
                    return Some((
                        FusionPattern::ReduceMap {
                            reduce_op: *reduce_op,
                            axis: Some(*axis),
                            map_fn: map_fn.clone(),
                            input: input_ref.clone(),
                        },
                        vec![consumer_idx, producer_idx],
                    ));
                }
            }
        }
    }

    // Pattern 2: zipWith f (map g a) (map h b)
    if let TensorOp::ZipWith(zip_fn, left_ref, right_ref) = &consumer.op {
        let left_producer = find_producer(ops, left_ref.id);
        let right_producer = find_producer(ops, right_ref.id);

        let left_is_fusible_map = left_producer.map_or(false, |idx| {
            !ops[idx].fused
                && ctx.ref_count(left_ref.id) == 1
                && matches!(&ops[idx].op, TensorOp::Map(_, _))
        });

        let right_is_fusible_map = right_producer.map_or(false, |idx| {
            !ops[idx].fused
                && ctx.ref_count(right_ref.id) == 1
                && matches!(&ops[idx].op, TensorOp::Map(_, _))
        });

        if left_is_fusible_map || right_is_fusible_map {
            let mut consumed = vec![consumer_idx];
            let mut left_fn = None;
            let mut left_input = left_ref.clone();
            let mut right_fn = None;
            let mut right_input = right_ref.clone();

            if let Some(idx) = left_producer {
                if left_is_fusible_map {
                    if let TensorOp::Map(f, inp) = &ops[idx].op {
                        left_fn = Some(f.clone());
                        left_input = inp.clone();
                        consumed.push(idx);
                    }
                }
            }

            if let Some(idx) = right_producer {
                if right_is_fusible_map {
                    if let TensorOp::Map(f, inp) = &ops[idx].op {
                        right_fn = Some(f.clone());
                        right_input = inp.clone();
                        consumed.push(idx);
                    }
                }
            }

            return Some((
                FusionPattern::ZipWithMaps {
                    zip_fn: zip_fn.clone(),
                    left_fn,
                    left_input,
                    right_fn,
                    right_input,
                },
                consumed,
            ));
        }
    }

    // Pattern 4: foldl' op z (map f x)
    if let TensorOp::Fold(fold_fn, init_ref, inner_ref) = &consumer.op {
        if let Some(producer_idx) = find_producer(ops, inner_ref.id) {
            if !ops[producer_idx].fused && ctx.ref_count(inner_ref.id) == 1 {
                if let TensorOp::Map(map_fn, input_ref) = &ops[producer_idx].op {
                    return Some((
                        FusionPattern::FoldMap {
                            fold_fn: fold_fn.name,
                            init: init_ref.clone(),
                            map_fn: map_fn.clone(),
                            input: input_ref.clone(),
                        },
                        vec![consumer_idx, producer_idx],
                    ));
                }
            }
        }
    }

    // ========================================================================
    // ML-Specific Pattern Detection (H26-SPEC Section 8.2)
    // ========================================================================

    // Try to detect ML-specific patterns
    if let Some(result) = try_detect_ml_pattern(ctx, ops, consumer_idx) {
        return Some(result);
    }

    None
}

/// Tries to detect ML-specific fusion patterns.
///
/// These patterns are more complex than the basic fusion patterns and require
/// looking at multiple operations in sequence to identify.
fn try_detect_ml_pattern(
    ctx: &FusionContext,
    ops: &[FusibleOp],
    consumer_idx: usize,
) -> Option<(FusionPattern, Vec<usize>)> {
    let _consumer = &ops[consumer_idx];

    // Pattern 5: Softmax detection
    // Look for: map (/ total) (map exp (map (\x -> x - max) input))
    // where total = sum (map exp ...)
    if let Some(result) = try_detect_softmax(ctx, ops, consumer_idx) {
        return Some(result);
    }

    // Pattern 7: LayerNorm detection
    // Look for: map (\x -> (x - mu) / sqrt(var + eps)) input
    // where mu = mean input, var = mean (map sq (map (- mu) input))
    if let Some(result) = try_detect_layernorm(ctx, ops, consumer_idx) {
        return Some(result);
    }

    // Pattern 9: Attention detection
    // Look for: matmul (softmax (scale * matmul q (transpose k))) v
    if let Some(result) = try_detect_attention(ctx, ops, consumer_idx) {
        return Some(result);
    }

    // Pattern 10/11: GELU/SiLU activation fusion
    if let Some(result) = try_detect_activation(ctx, ops, consumer_idx) {
        return Some(result);
    }

    // Pattern 12: Fused Linear detection
    // Look for: activation (add (matmul x w) b)
    if let Some(result) = try_detect_fused_linear(ctx, ops, consumer_idx) {
        return Some(result);
    }

    None
}

/// Detects the softmax pattern.
///
/// Softmax is: map (/ total) exps
/// where exps = map exp (map (\x -> x - max) input)
///       max = maximum input
///       total = sum exps
///
/// We look for the final division by sum pattern.
fn try_detect_softmax(
    ctx: &FusionContext,
    ops: &[FusibleOp],
    consumer_idx: usize,
) -> Option<(FusionPattern, Vec<usize>)> {
    let consumer = &ops[consumer_idx];

    // Look for a map operation that divides by a sum
    // Pattern: map (\x -> x / total) exps where total = sum exps
    if let TensorOp::ZipWith(div_fn, exps_ref, total_ref) = &consumer.op {
        // Check if this is division
        if !is_division_fn(&div_fn.name) {
            return None;
        }

        // Check if total is a sum reduction
        let total_producer = find_producer(ops, total_ref.id)?;
        if ops[total_producer].fused {
            return None;
        }

        if let TensorOp::ReduceAll(ReduceOp::Sum, _sum_input) = &ops[total_producer].op {
            // Check if sum_input is the same as exps (or the exp-mapped version)
            // This is a simplified check - in practice we'd trace back more carefully

            // Check if exps is from exp(shifted) where shifted = x - max
            let exps_producer = find_producer(ops, exps_ref.id)?;
            if ops[exps_producer].fused {
                return None;
            }

            // Look for the exp map
            if let TensorOp::Map(exp_fn, shifted_ref) = &ops[exps_producer].op {
                if !is_exp_fn(&exp_fn.name) {
                    return None;
                }

                // Look for the shift (x - max) pattern
                let shifted_producer = find_producer(ops, shifted_ref.id)?;
                if ops[shifted_producer].fused {
                    return None;
                }

                // The original input is either from the shift operation or directly available
                let input = match &ops[shifted_producer].op {
                    TensorOp::ZipWith(sub_fn, input_ref, _max_ref)
                        if is_subtraction_fn(&sub_fn.name) =>
                    {
                        input_ref.clone()
                    }
                    TensorOp::Binary(BinaryOp::Sub, input_ref, _max_ref) => input_ref.clone(),
                    _ => {
                        // Fall back to the shifted input itself
                        shifted_ref.clone()
                    }
                };

                // Collect all consumed operations
                let mut consumed = vec![consumer_idx, total_producer, exps_producer];
                if shifted_producer != exps_producer {
                    consumed.push(shifted_producer);
                }

                // All intermediate refs must be single-use for fusion
                let all_single_use = consumed.iter().all(|&idx| {
                    if idx == consumer_idx {
                        true
                    } else {
                        ctx.ref_count(ops[idx].output.id) == 1
                    }
                });

                if all_single_use {
                    return Some((FusionPattern::Softmax { input, axis: None }, consumed));
                }
            }
        }
    }

    None
}

/// Detects the layer normalization pattern.
///
/// LayerNorm is:
/// 1. Compute mean: mu = mean(x)
/// 2. Compute variance: var = mean((x - mu)^2)
/// 3. Normalize: (x - mu) / sqrt(var + eps)
fn try_detect_layernorm(
    ctx: &FusionContext,
    ops: &[FusibleOp],
    consumer_idx: usize,
) -> Option<(FusionPattern, Vec<usize>)> {
    let consumer = &ops[consumer_idx];

    // Look for the final normalization step: (x - mu) / sqrt(var + eps)
    // This typically appears as a ZipWith division
    if let TensorOp::ZipWith(div_fn, centered_ref, std_ref) = &consumer.op {
        if !is_division_fn(&div_fn.name) {
            return None;
        }

        // Check if std is from sqrt(var + eps)
        let std_producer = find_producer(ops, std_ref.id)?;
        if ops[std_producer].fused {
            return None;
        }

        // Look for sqrt operation
        let (_var_ref, epsilon) = match &ops[std_producer].op {
            TensorOp::Unary(crate::UnaryOp::Sqrt, var_plus_eps) => {
                // Check if var_plus_eps is var + eps
                let vpe_producer = find_producer(ops, var_plus_eps.id)?;
                match &ops[vpe_producer].op {
                    TensorOp::Binary(BinaryOp::Add, var_ref, _eps_ref) => {
                        // Assume eps is a small constant
                        (var_ref.clone(), 1e-5)
                    }
                    _ => return None,
                }
            }
            TensorOp::Map(sqrt_fn, var_plus_eps) if is_sqrt_fn(&sqrt_fn.name) => {
                (var_plus_eps.clone(), 1e-5)
            }
            _ => return None,
        };

        // Look for centered = x - mu
        let centered_producer = find_producer(ops, centered_ref.id)?;
        if ops[centered_producer].fused {
            return None;
        }

        let input = match &ops[centered_producer].op {
            TensorOp::ZipWith(sub_fn, input_ref, _mu_ref) if is_subtraction_fn(&sub_fn.name) => {
                input_ref.clone()
            }
            TensorOp::Binary(BinaryOp::Sub, input_ref, _mu_ref) => input_ref.clone(),
            _ => return None,
        };

        // Collect consumed operations
        let consumed = vec![consumer_idx, std_producer, centered_producer];

        // Check all intermediates are single-use
        let all_single_use = consumed
            .iter()
            .skip(1)
            .all(|&idx| ctx.ref_count(ops[idx].output.id) == 1);

        if all_single_use {
            return Some((
                FusionPattern::LayerNorm {
                    input,
                    epsilon,
                    axis: None,
                    scale: None,
                    bias: None,
                },
                consumed,
            ));
        }
    }

    None
}

/// Detects the attention pattern.
///
/// Attention is: matmul(softmax(scale * matmul(q, transpose(k))), v)
fn try_detect_attention(
    _ctx: &FusionContext,
    ops: &[FusibleOp],
    consumer_idx: usize,
) -> Option<(FusionPattern, Vec<usize>)> {
    let consumer = &ops[consumer_idx];

    // Look for the outer matmul: matmul(weights, v)
    if let TensorOp::MatMul(weights_ref, v_ref) | TensorOp::BatchMatMul(weights_ref, v_ref) =
        &consumer.op
    {
        let weights_producer = find_producer(ops, weights_ref.id)?;
        if ops[weights_producer].fused {
            return None;
        }

        // Check if weights comes from a softmax-like pattern
        // For simplicity, we look for a Softmax pattern or a map/reduce sequence
        let (scores_ref, _scale) = match &ops[weights_producer].op {
            // If weights is directly a softmax operation marked, we can detect it
            TensorOp::Map(softmax_fn, scores) if is_softmax_fn(&softmax_fn.name) => {
                (scores.clone(), 1.0)
            }
            // Otherwise, try to find a preceding softmax pattern
            _ => {
                // Check if the weights producer output is connected to a known softmax
                // This is a simplified check - a full implementation would trace more
                return None;
            }
        };

        // Look for the inner matmul with scale: scale * matmul(q, transpose(k))
        let scores_producer = find_producer(ops, scores_ref.id)?;
        if ops[scores_producer].fused {
            return None;
        }

        // Try to find q @ k^T pattern
        let (query, key) = match &ops[scores_producer].op {
            TensorOp::MatMul(q_ref, kt_ref) | TensorOp::BatchMatMul(q_ref, kt_ref) => {
                // Check if kt is transposed
                let kt_producer = find_producer(ops, kt_ref.id);
                let key = if let Some(idx) = kt_producer {
                    if let TensorOp::Transpose(_, k_ref) = &ops[idx].op {
                        k_ref.clone()
                    } else {
                        kt_ref.clone()
                    }
                } else {
                    kt_ref.clone()
                };
                (q_ref.clone(), key)
            }
            TensorOp::Binary(BinaryOp::Mul, scaled_matmul, _scale_ref) => {
                // scale * matmul(...)
                let sm_producer = find_producer(ops, scaled_matmul.id)?;
                match &ops[sm_producer].op {
                    TensorOp::MatMul(q_ref, kt_ref) | TensorOp::BatchMatMul(q_ref, kt_ref) => {
                        (q_ref.clone(), kt_ref.clone())
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };

        // Infer scale from head dimension
        let head_dim = query
            .meta
            .shape
            .dims()
            .last()
            .and_then(|d| d.static_value())
            .unwrap_or(64) as f64;
        let inferred_scale = 1.0 / head_dim.sqrt();

        let consumed = vec![consumer_idx, weights_producer, scores_producer];

        return Some((
            FusionPattern::Attention {
                query,
                key,
                value: v_ref.clone(),
                mask: None,
                scale: inferred_scale,
                causal: false,
            },
            consumed,
        ));
    }

    None
}

/// Detects GELU or SiLU activation patterns.
fn try_detect_activation(
    ctx: &FusionContext,
    ops: &[FusibleOp],
    consumer_idx: usize,
) -> Option<(FusionPattern, Vec<usize>)> {
    let consumer = &ops[consumer_idx];

    // GELU fast: x * sigmoid(1.702 * x)
    // SiLU: x * sigmoid(x)
    if let TensorOp::ZipWith(mul_fn, x_ref, sigmoid_ref) = &consumer.op {
        if !is_multiplication_fn(&mul_fn.name) {
            return None;
        }

        let sigmoid_producer = find_producer(ops, sigmoid_ref.id)?;
        if ops[sigmoid_producer].fused {
            return None;
        }

        match &ops[sigmoid_producer].op {
            TensorOp::Unary(crate::UnaryOp::Sigmoid, inner_ref) => {
                // Check if inner is x (SiLU) or scaled x (GELU fast)
                if inner_ref.id == x_ref.id {
                    // SiLU: x * sigmoid(x)
                    return Some((
                        FusionPattern::Silu {
                            input: x_ref.clone(),
                        },
                        vec![consumer_idx, sigmoid_producer],
                    ));
                }

                // Check for GELU fast: x * sigmoid(1.702 * x)
                let inner_producer = find_producer(ops, inner_ref.id)?;
                if let TensorOp::Binary(BinaryOp::Mul, x2_ref, _scale) = &ops[inner_producer].op {
                    if x2_ref.id == x_ref.id && ctx.ref_count(inner_ref.id) == 1 {
                        return Some((
                            FusionPattern::Gelu {
                                input: x_ref.clone(),
                                fast: true,
                            },
                            vec![consumer_idx, sigmoid_producer, inner_producer],
                        ));
                    }
                }
            }
            TensorOp::Map(sigmoid_fn, inner_ref) if is_sigmoid_fn(&sigmoid_fn.name) => {
                if inner_ref.id == x_ref.id {
                    return Some((
                        FusionPattern::Silu {
                            input: x_ref.clone(),
                        },
                        vec![consumer_idx, sigmoid_producer],
                    ));
                }
            }
            _ => {}
        }
    }

    None
}

/// Detects fused linear (matmul + bias + activation) pattern.
fn try_detect_fused_linear(
    ctx: &FusionContext,
    ops: &[FusibleOp],
    consumer_idx: usize,
) -> Option<(FusionPattern, Vec<usize>)> {
    let consumer = &ops[consumer_idx];

    // Look for activation(matmul(x, w) + b) or matmul(x, w) + b
    // Start from an activation or bias addition

    // Pattern: ReLU(matmul + bias) or other activation
    if let TensorOp::Unary(unary_op, inner_ref) = &consumer.op {
        let activation = match unary_op {
            crate::UnaryOp::Relu => Some(FusedActivation::Relu),
            crate::UnaryOp::Sigmoid => Some(FusedActivation::Sigmoid),
            crate::UnaryOp::Tanh => Some(FusedActivation::Tanh),
            _ => None,
        };

        if let Some(act) = activation {
            let inner_producer = find_producer(ops, inner_ref.id)?;
            if ops[inner_producer].fused || ctx.ref_count(inner_ref.id) > 1 {
                return None;
            }

            // Check if inner is matmul + bias
            if let TensorOp::Binary(BinaryOp::Add, matmul_ref, bias_ref) = &ops[inner_producer].op {
                let matmul_producer = find_producer(ops, matmul_ref.id)?;
                if ops[matmul_producer].fused || ctx.ref_count(matmul_ref.id) > 1 {
                    return None;
                }

                if let TensorOp::MatMul(input_ref, weight_ref) = &ops[matmul_producer].op {
                    return Some((
                        FusionPattern::FusedLinear {
                            input: input_ref.clone(),
                            weight: weight_ref.clone(),
                            bias: Some(bias_ref.clone()),
                            activation: Some(act),
                        },
                        vec![consumer_idx, inner_producer, matmul_producer],
                    ));
                }
            }

            // Check if inner is just matmul (no bias)
            if let TensorOp::MatMul(input_ref, weight_ref) = &ops[inner_producer].op {
                return Some((
                    FusionPattern::FusedLinear {
                        input: input_ref.clone(),
                        weight: weight_ref.clone(),
                        bias: None,
                        activation: Some(act),
                    },
                    vec![consumer_idx, inner_producer],
                ));
            }
        }
    }

    // Pattern: matmul + bias (no activation)
    if let TensorOp::Binary(BinaryOp::Add, matmul_ref, bias_ref) = &consumer.op {
        let matmul_producer = find_producer(ops, matmul_ref.id)?;
        if ops[matmul_producer].fused || ctx.ref_count(matmul_ref.id) > 1 {
            return None;
        }

        if let TensorOp::MatMul(input_ref, weight_ref) = &ops[matmul_producer].op {
            return Some((
                FusionPattern::FusedLinear {
                    input: input_ref.clone(),
                    weight: weight_ref.clone(),
                    bias: Some(bias_ref.clone()),
                    activation: None,
                },
                vec![consumer_idx, matmul_producer],
            ));
        }
    }

    None
}

// ============================================================================
// Helper functions for pattern detection
// ============================================================================

fn is_division_fn(name: &Symbol) -> bool {
    let s = name.as_str();
    s == "/" || s == "div" || s == "divide"
}

fn is_subtraction_fn(name: &Symbol) -> bool {
    let s = name.as_str();
    s == "-" || s == "sub" || s == "subtract"
}

fn is_multiplication_fn(name: &Symbol) -> bool {
    let s = name.as_str();
    s == "*" || s == "mul" || s == "multiply"
}

fn is_exp_fn(name: &Symbol) -> bool {
    let s = name.as_str();
    s == "exp"
}

fn is_sqrt_fn(name: &Symbol) -> bool {
    let s = name.as_str();
    s == "sqrt"
}

fn is_sigmoid_fn(name: &Symbol) -> bool {
    let s = name.as_str();
    s == "sigmoid"
}

fn is_softmax_fn(name: &Symbol) -> bool {
    let s = name.as_str();
    s == "softmax" || s == "Softmax"
}

/// Finds the producer operation for a tensor ID.
fn find_producer(ops: &[FusibleOp], id: TensorId) -> Option<usize> {
    ops.iter().position(|op| op.output.id == id)
}

/// Creates a fused group from a pattern and consumed operations.
fn create_fused_group(
    ctx: &mut FusionContext,
    ops: &[FusibleOp],
    consumed_indices: &[usize],
    pattern: FusionPattern,
) -> FusedGroup {
    let op_names: Vec<Symbol> = consumed_indices
        .iter()
        .map(|&i| op_name(&ops[i].op))
        .collect();

    let (inputs, output, fused_ops) = match &pattern {
        FusionPattern::MapMap {
            outer_fn,
            inner_fn,
            input,
        } => {
            // Compose: (outer . inner)
            let composed_fn = MapFn {
                name: Symbol::intern(&format!(
                    "({} . {})",
                    outer_fn.name.as_str(),
                    inner_fn.name.as_str()
                )),
                span: outer_fn.span,
            };
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: input.meta.clone(),
            };
            (
                vec![input.clone()],
                output.clone(),
                vec![TensorOp::Map(composed_fn, input.clone())],
            )
        }
        FusionPattern::ReduceMap {
            reduce_op,
            axis,
            map_fn,
            input,
        } => {
            let output_id = ctx.fresh_tensor_id();
            let output_shape = if axis.is_some() {
                // Reduce along axis
                let mut dims: SmallVec<[crate::Dim; 4]> =
                    input.meta.shape.dims().iter().cloned().collect();
                if let Some(idx) = axis.and_then(|a| a.normalize(dims.len())) {
                    dims.remove(idx);
                }
                Shape::new(dims)
            } else {
                Shape::scalar()
            };
            let output = TensorRef {
                id: output_id,
                meta: TensorMeta::new_contiguous(input.meta.dtype, output_shape)
                    .unwrap_or_else(|| input.meta.clone()),
            };

            // Create fused reduce-map operation
            let fused_op = if let Some(ax) = axis {
                TensorOp::Reduce(*reduce_op, *ax, input.clone())
            } else {
                TensorOp::ReduceAll(*reduce_op, input.clone())
            };

            // Note: In a real implementation, the map function would be
            // composed into the reduction. For now we represent this as
            // a fused kernel with both operations.
            (
                vec![input.clone()],
                output,
                vec![TensorOp::Map(map_fn.clone(), input.clone()), fused_op],
            )
        }
        FusionPattern::ZipWithMaps {
            zip_fn,
            left_fn,
            left_input,
            right_fn,
            right_input,
        } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: left_input.meta.clone(),
            };

            // Create fused zip-with-maps
            let combined_name = match (left_fn, right_fn) {
                (Some(l), Some(r)) => format!(
                    "zipWith {} ({}) ({})",
                    zip_fn.name.as_str(),
                    l.name.as_str(),
                    r.name.as_str()
                ),
                (Some(l), None) => {
                    format!("zipWith {} ({}) id", zip_fn.name.as_str(), l.name.as_str())
                }
                (None, Some(r)) => {
                    format!("zipWith {} id ({})", zip_fn.name.as_str(), r.name.as_str())
                }
                (None, None) => format!("zipWith {}", zip_fn.name.as_str()),
            };

            let fused_fn = ZipFn {
                name: Symbol::intern(&combined_name),
                span: zip_fn.span,
            };

            (
                vec![left_input.clone(), right_input.clone()],
                output,
                vec![TensorOp::ZipWith(
                    fused_fn,
                    left_input.clone(),
                    right_input.clone(),
                )],
            )
        }
        FusionPattern::FoldMap {
            fold_fn,
            init,
            map_fn,
            input,
        } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: TensorMeta::new_contiguous(input.meta.dtype, Shape::scalar())
                    .unwrap_or_else(|| input.meta.clone()),
            };

            let fused_fn = crate::FoldFn {
                name: Symbol::intern(&format!(
                    "fold {} . {}",
                    fold_fn.as_str(),
                    map_fn.name.as_str()
                )),
                span: map_fn.span,
            };

            (
                vec![init.clone(), input.clone()],
                output,
                vec![TensorOp::Fold(fused_fn, init.clone(), input.clone())],
            )
        }

        // ====================================================================
        // ML-Specific Pattern Kernel Generation
        // ====================================================================
        FusionPattern::Softmax { input, axis: _ } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: input.meta.clone(),
            };
            // Softmax kernel: numerically stable single-kernel implementation
            // The kernel internally computes max, exp(x - max), sum, and division
            let softmax_fn = MapFn {
                name: Symbol::intern("softmax_fused"),
                span: bhc_span::Span::DUMMY,
            };
            (
                vec![input.clone()],
                output,
                vec![TensorOp::Map(softmax_fn, input.clone())],
            )
        }

        FusionPattern::LogSoftmax { input, axis: _ } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: input.meta.clone(),
            };
            let logsoftmax_fn = MapFn {
                name: Symbol::intern("log_softmax_fused"),
                span: bhc_span::Span::DUMMY,
            };
            (
                vec![input.clone()],
                output,
                vec![TensorOp::Map(logsoftmax_fn, input.clone())],
            )
        }

        FusionPattern::LayerNorm {
            input,
            epsilon,
            axis: _,
            scale,
            bias,
        } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: input.meta.clone(),
            };
            // LayerNorm kernel: single-pass Welford algorithm
            let layernorm_fn = MapFn {
                name: Symbol::intern(&format!("layernorm_welford_eps{:.0e}", epsilon)),
                span: bhc_span::Span::DUMMY,
            };
            let mut inputs = vec![input.clone()];
            if let Some(s) = scale {
                inputs.push(s.clone());
            }
            if let Some(b) = bias {
                inputs.push(b.clone());
            }
            (
                inputs,
                output,
                vec![TensorOp::Map(layernorm_fn, input.clone())],
            )
        }

        FusionPattern::RMSNorm {
            input,
            epsilon,
            scale,
        } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: input.meta.clone(),
            };
            let rmsnorm_fn = MapFn {
                name: Symbol::intern(&format!("rmsnorm_eps{:.0e}", epsilon)),
                span: bhc_span::Span::DUMMY,
            };
            let mut inputs = vec![input.clone()];
            if let Some(s) = scale {
                inputs.push(s.clone());
            }
            (
                inputs,
                output,
                vec![TensorOp::Map(rmsnorm_fn, input.clone())],
            )
        }

        FusionPattern::Attention {
            query,
            key,
            value,
            mask,
            scale,
            causal,
        } => {
            // Output shape: same as query
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: query.meta.clone(),
            };
            // Fused attention kernel with optional mask and causal mode
            let _attention_name = if *causal {
                format!("fused_attention_causal_scale{:.4}", scale)
            } else {
                format!("fused_attention_scale{:.4}", scale)
            };
            let mut inputs = vec![query.clone(), key.clone(), value.clone()];
            if let Some(m) = mask {
                inputs.push(m.clone());
            }
            // For the IR, we represent this as a specialized matmul
            (
                inputs,
                output,
                vec![TensorOp::BatchMatMul(query.clone(), value.clone())],
            )
        }

        FusionPattern::Gelu { input, fast } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: input.meta.clone(),
            };
            let gelu_fn = MapFn {
                name: Symbol::intern(if *fast { "gelu_fast" } else { "gelu" }),
                span: bhc_span::Span::DUMMY,
            };
            (
                vec![input.clone()],
                output,
                vec![TensorOp::Map(gelu_fn, input.clone())],
            )
        }

        FusionPattern::Silu { input } => {
            let output_id = ctx.fresh_tensor_id();
            let output = TensorRef {
                id: output_id,
                meta: input.meta.clone(),
            };
            let silu_fn = MapFn {
                name: Symbol::intern("silu"),
                span: bhc_span::Span::DUMMY,
            };
            (
                vec![input.clone()],
                output,
                vec![TensorOp::Map(silu_fn, input.clone())],
            )
        }

        FusionPattern::FusedLinear {
            input,
            weight,
            bias,
            activation,
        } => {
            // Output shape: [batch, out_features]
            let output_id = ctx.fresh_tensor_id();
            let w_dims = weight.meta.shape.dims();
            let out_dim = w_dims
                .last()
                .cloned()
                .unwrap_or_else(|| crate::Dim::Static(1));
            let in_dims = input.meta.shape.dims();
            let mut out_shape: SmallVec<[crate::Dim; 4]> = in_dims.iter().cloned().collect();
            if let Some(last) = out_shape.last_mut() {
                *last = out_dim;
            }
            let output = TensorRef {
                id: output_id,
                meta: TensorMeta::new_contiguous(input.meta.dtype, Shape::new(out_shape))
                    .unwrap_or_else(|| input.meta.clone()),
            };
            // Kernel name includes activation
            let act_suffix = match activation {
                Some(FusedActivation::Relu) => "_relu",
                Some(FusedActivation::Gelu) => "_gelu",
                Some(FusedActivation::GeluFast) => "_gelu_fast",
                Some(FusedActivation::Silu) => "_silu",
                Some(FusedActivation::Sigmoid) => "_sigmoid",
                Some(FusedActivation::Tanh) => "_tanh",
                None => "",
            };
            let has_bias = if bias.is_some() { "_bias" } else { "" };
            let _ = format!("fused_linear{}{}", has_bias, act_suffix);

            let mut inputs = vec![input.clone(), weight.clone()];
            if let Some(b) = bias {
                inputs.push(b.clone());
            }
            (
                inputs,
                output,
                vec![TensorOp::MatMul(input.clone(), weight.clone())],
            )
        }
    };

    FusedGroup {
        ops: fused_ops,
        inputs,
        output,
        pattern: Some(pattern),
        op_names,
    }
}

/// Gets a symbolic name for an operation (for reporting).
fn op_name(op: &TensorOp) -> Symbol {
    let name = match op {
        TensorOp::Constant(_) => "constant",
        TensorOp::Unary(op, _) => match op {
            crate::UnaryOp::Neg => "neg",
            crate::UnaryOp::Abs => "abs",
            crate::UnaryOp::Sqrt => "sqrt",
            crate::UnaryOp::Exp => "exp",
            crate::UnaryOp::Log => "log",
            crate::UnaryOp::Sin => "sin",
            crate::UnaryOp::Cos => "cos",
            _ => "unary",
        },
        TensorOp::Binary(op, _, _) => match op {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            _ => "binary",
        },
        TensorOp::Map(_, _) => "map",
        TensorOp::ZipWith(_, _, _) => "zipWith",
        TensorOp::Reduce(_, _, _) => "reduce",
        TensorOp::ReduceAll(_, _) => "reduceAll",
        TensorOp::Scan(_, _, _) => "scan",
        TensorOp::Fold(_, _, _) => "fold",
        TensorOp::Reshape(_, _) => "reshape",
        TensorOp::Slice(_, _) => "slice",
        TensorOp::Transpose(_, _) => "transpose",
        TensorOp::Broadcast(_, _) => "broadcast",
        TensorOp::Concat(_, _) => "concat",
        TensorOp::Split(_, _, _) => "split",
        TensorOp::MatMul(_, _) => "matmul",
        TensorOp::BatchMatMul(_, _) => "batchMatmul",
        TensorOp::Dot(_, _) => "dot",
        TensorOp::Outer(_, _) => "outer",
        TensorOp::Conv(_, _, _) => "conv",
        TensorOp::Gather(_, _, _) => "gather",
        TensorOp::Scatter(_, _, _, _) => "scatter",
    };
    Symbol::intern(name)
}

/// Generates a kernel from a fused group.
fn generate_kernel(ctx: &mut FusionContext, group: FusedGroup) -> Kernel {
    let id = ctx.fresh_kernel_id();
    let name = generate_kernel_name(&group);

    // Determine allocation requirements
    let allocs = compute_allocations(ctx, &group);

    // Build fusion info for reporting
    let fusion_info = FusionInfo {
        original_ops: group.op_names.clone(),
        decisions: vec![FusionDecision::Fused(group.op_names.clone())],
        complete: group.pattern.is_some(),
    };

    Kernel {
        id,
        name,
        inputs: group.inputs,
        outputs: vec![group.output],
        body: KernelBody::Fused(group.ops),
        allocs,
        fusion_info,
    }
}

/// Generates a name for a kernel.
fn generate_kernel_name(group: &FusedGroup) -> Symbol {
    if let Some(pattern) = &group.pattern {
        let name = match pattern {
            // Basic fusion patterns
            FusionPattern::MapMap { .. } => "fused_map_map",
            FusionPattern::ReduceMap { .. } => "fused_reduce_map",
            FusionPattern::ZipWithMaps { .. } => "fused_zipwith_maps",
            FusionPattern::FoldMap { .. } => "fused_fold_map",
            // ML fusion patterns
            FusionPattern::Softmax { .. } => "fused_softmax",
            FusionPattern::LogSoftmax { .. } => "fused_log_softmax",
            FusionPattern::LayerNorm { .. } => "fused_layernorm_welford",
            FusionPattern::RMSNorm { .. } => "fused_rmsnorm",
            FusionPattern::Attention { causal, .. } => {
                if *causal {
                    "fused_attention_causal"
                } else {
                    "fused_attention"
                }
            }
            FusionPattern::Gelu { fast, .. } => {
                if *fast {
                    "fused_gelu_fast"
                } else {
                    "fused_gelu"
                }
            }
            FusionPattern::Silu { .. } => "fused_silu",
            FusionPattern::FusedLinear {
                bias, activation, ..
            } => match (bias.is_some(), activation) {
                (true, Some(FusedActivation::Relu)) => "fused_linear_bias_relu",
                (true, Some(FusedActivation::Gelu)) => "fused_linear_bias_gelu",
                (true, Some(FusedActivation::GeluFast)) => "fused_linear_bias_gelu_fast",
                (true, Some(FusedActivation::Silu)) => "fused_linear_bias_silu",
                (true, Some(FusedActivation::Sigmoid)) => "fused_linear_bias_sigmoid",
                (true, Some(FusedActivation::Tanh)) => "fused_linear_bias_tanh",
                (true, None) => "fused_linear_bias",
                (false, Some(FusedActivation::Relu)) => "fused_linear_relu",
                (false, Some(FusedActivation::Gelu)) => "fused_linear_gelu",
                (false, Some(FusedActivation::GeluFast)) => "fused_linear_gelu_fast",
                (false, Some(FusedActivation::Silu)) => "fused_linear_silu",
                (false, Some(FusedActivation::Sigmoid)) => "fused_linear_sigmoid",
                (false, Some(FusedActivation::Tanh)) => "fused_linear_tanh",
                (false, None) => "fused_linear",
            },
        };
        Symbol::intern(name)
    } else if group.op_names.len() == 1 {
        group.op_names[0]
    } else {
        Symbol::intern("kernel")
    }
}

/// Computes allocation requirements for a fused group.
fn compute_allocations(ctx: &mut FusionContext, group: &FusedGroup) -> Vec<AllocInfo> {
    let mut allocs = Vec::new();

    // Output buffer allocation
    if let Some(size) = group.output.meta.size_bytes() {
        let buffer = ctx.fresh_buffer_id();
        allocs.push(AllocInfo {
            buffer,
            size,
            alignment: group.output.meta.dtype.alignment(),
            region: if ctx.strict_mode {
                AllocRegion::HotArena
            } else {
                AllocRegion::General
            },
        });
    }

    allocs
}

/// Checks if a reshape is metadata-only (no data movement needed).
///
/// A reshape is metadata-only when the tensor is contiguous.
#[must_use]
pub fn is_reshape_metadata_only(tensor: &TensorRef) -> bool {
    matches!(tensor.meta.layout, crate::Layout::Contiguous)
}

/// Generates a kernel report for the fusion pass.
#[must_use]
pub fn generate_kernel_report(ctx: &FusionContext) -> KernelReport {
    KernelReport {
        kernels: ctx.kernels.clone(),
        decisions: ctx.decisions.clone(),
        total_ops: ctx
            .kernels
            .iter()
            .map(|k| k.fusion_info.original_ops.len())
            .sum(),
        fused_ops: ctx
            .decisions
            .iter()
            .filter(|d| matches!(d, FusionDecision::Fused(_)))
            .count(),
    }
}

/// A report of the fusion pass results.
#[derive(Clone, Debug)]
pub struct KernelReport {
    /// Generated kernels.
    pub kernels: Vec<Kernel>,
    /// Fusion decisions made.
    pub decisions: Vec<FusionDecision>,
    /// Total number of original operations.
    pub total_ops: usize,
    /// Number of operations that were fused.
    pub fused_ops: usize,
}

impl std::fmt::Display for KernelReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Kernel Report ===")?;
        writeln!(f, "Total operations: {}", self.total_ops)?;
        writeln!(f, "Fused operations: {}", self.fused_ops)?;
        writeln!(f, "Generated kernels: {}", self.kernels.len())?;
        writeln!(f)?;

        for kernel in &self.kernels {
            writeln!(f, "Kernel: {}", kernel.name.as_str())?;
            writeln!(f, "  Inputs: {}", kernel.inputs.len())?;
            writeln!(f, "  Outputs: {}", kernel.outputs.len())?;
            writeln!(
                f,
                "  Fused: {}",
                if kernel.fusion_info.complete {
                    "YES"
                } else {
                    "NO"
                }
            )?;
            if !kernel.fusion_info.original_ops.is_empty() {
                write!(f, "  Original ops: ")?;
                for (i, op) in kernel.fusion_info.original_ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", op.as_str())?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_span::Span;

    fn make_tensor_ref(id: u32, shape: &[usize], dtype: DType) -> TensorRef {
        TensorRef {
            id: TensorId::new(id as usize),
            meta: TensorMeta::new_contiguous(dtype, Shape::from_static(shape.iter().copied()))
                .unwrap(),
        }
    }

    fn make_map_fn(name: &str) -> MapFn {
        MapFn {
            name: Symbol::intern(name),
            span: Span::DUMMY,
        }
    }

    #[test]
    fn test_pattern1_map_map_fusion() {
        // map f (map g x) should fuse to map (f . g) x
        //
        // The key insight: the first map's input is distinct from
        // the intermediate (which connects first map output to second map input).
        //
        // x (id=100) -> [map g] -> intermediate (id=0) -> [map f] -> result (id=1)
        //
        // When build_fusible_ops runs:
        // - first_map: inputs=[100], output=0
        // - second_map: inputs=[0], output=1
        // ref_counts: {100: 1, 0: 1}
        //
        // Since intermediate (id=0) has ref_count=1, fusion is allowed.

        // Use a high ID for the original input so it won't conflict
        let input = make_tensor_ref(100, &[100], DType::Float32);
        let first_map = TensorOp::Map(make_map_fn("g"), input);

        // The intermediate will get id=0 when processed by build_fusible_ops
        // Second map consumes id=0
        let intermediate = make_tensor_ref(0, &[100], DType::Float32);
        let second_map = TensorOp::Map(make_map_fn("f"), intermediate);

        let mut ctx = FusionContext::new(true);
        let ops = vec![first_map, second_map];
        let kernels = fuse_ops(&mut ctx, ops);

        assert_eq!(kernels.len(), 1, "should produce single fused kernel");
        assert!(kernels[0].fusion_info.complete, "fusion should be complete");
        assert_eq!(
            kernels[0].fusion_info.original_ops.len(),
            2,
            "should track both original ops"
        );
    }

    #[test]
    fn test_pattern3_reduce_map_fusion() {
        // sum (map f x) should fuse to single traversal

        // Use high ID for original input
        let input = make_tensor_ref(100, &[100], DType::Float32);
        let first_map = TensorOp::Map(make_map_fn("f"), input);

        // Intermediate will get id=0, reduce consumes id=0
        let mapped = make_tensor_ref(0, &[100], DType::Float32);
        let reduce = TensorOp::ReduceAll(ReduceOp::Sum, mapped);

        let mut ctx = FusionContext::new(true);
        let ops = vec![first_map, reduce];
        let kernels = fuse_ops(&mut ctx, ops);

        assert_eq!(kernels.len(), 1, "should produce single fused kernel");
        assert!(kernels[0].fusion_info.complete, "fusion should be complete");
    }

    #[test]
    fn test_pattern4_fold_map_fusion() {
        // foldl' op z (map f x) should fuse to single traversal
        //
        // x (id=100) -> [map f] -> intermediate (id=0) -> [fold op z] -> result
        //
        // Per H26-SPEC Section 8.1 Pattern 4, this MUST fuse.

        // Use high ID for original input
        let input = make_tensor_ref(100, &[100], DType::Float32);
        let map_op = TensorOp::Map(make_map_fn("double"), input);

        // Initial accumulator value (scalar)
        let init = make_tensor_ref(101, &[], DType::Float32);

        // Intermediate will get id=0, fold consumes id=0
        let mapped = make_tensor_ref(0, &[100], DType::Float32);
        let fold_fn = crate::FoldFn {
            name: Symbol::intern("+"),
            span: Span::DUMMY,
        };
        let fold_op = TensorOp::Fold(fold_fn, init, mapped);

        let mut ctx = FusionContext::new(true);
        let ops = vec![map_op, fold_op];
        let kernels = fuse_ops(&mut ctx, ops);

        assert_eq!(
            kernels.len(),
            1,
            "Pattern 4 (fold-map) should fuse to single kernel"
        );
        assert!(kernels[0].fusion_info.complete, "fusion should be complete");
        assert_eq!(
            kernels[0].name.as_str(),
            "fused_fold_map",
            "Should produce fused_fold_map kernel"
        );
    }

    #[test]
    fn test_reshape_metadata_only() {
        let contiguous = make_tensor_ref(0, &[10, 10], DType::Float32);
        assert!(
            is_reshape_metadata_only(&contiguous),
            "contiguous tensor reshape should be metadata-only"
        );

        // Non-contiguous (strided) tensor
        let strided = TensorRef {
            id: TensorId::new(1),
            meta: TensorMeta {
                dtype: DType::Float32,
                shape: Shape::from_static([10, 10]),
                strides: Strides::new([40, 8]), // Non-standard strides
                layout: crate::Layout::Strided,
                alias: None,
            },
        };
        assert!(
            !is_reshape_metadata_only(&strided),
            "strided tensor reshape should require data movement"
        );
    }

    #[test]
    fn test_kernel_report_generation() {
        // Use high ID for original input
        let input = make_tensor_ref(100, &[100], DType::Float32);
        let first_map = TensorOp::Map(make_map_fn("double"), input);

        // Intermediate connects first to second
        let intermediate = make_tensor_ref(0, &[100], DType::Float32);
        let second_map = TensorOp::Map(make_map_fn("inc"), intermediate);

        let mut ctx = FusionContext::new(true);
        let ops = vec![first_map, second_map];
        let _kernels = fuse_ops(&mut ctx, ops);
        let report = generate_kernel_report(&ctx);

        assert_eq!(report.kernels.len(), 1);
        assert!(report.fused_ops > 0, "should have fused operations");

        // Test display
        let display = format!("{report}");
        assert!(display.contains("Kernel Report"));
        assert!(display.contains("fused_map_map"));
    }

    #[test]
    fn test_multi_use_prevents_fusion() {
        // If intermediate is used multiple times, can't fuse
        // Simulate by having two consumers of the same intermediate ID

        let input = make_tensor_ref(100, &[100], DType::Float32);
        let first_map = TensorOp::Map(make_map_fn("g"), input);

        // Create two maps that both consume the intermediate (id=0)
        let intermediate = make_tensor_ref(0, &[100], DType::Float32);
        let second_map = TensorOp::Map(make_map_fn("f"), intermediate.clone());
        let third_map = TensorOp::Map(make_map_fn("h"), intermediate);

        let mut ctx = FusionContext::new(true);
        let ops = vec![first_map, second_map, third_map];
        let kernels = fuse_ops(&mut ctx, ops);

        // The intermediate is used twice (by second_map and third_map),
        // so fusion should be blocked
        assert!(
            kernels.len() >= 2,
            "multi-use intermediate should prevent full fusion"
        );
    }

    #[test]
    fn test_zipwith_map_fusion() {
        // zipWith f (map g a) (map h b) should fuse
        //
        // a (id=100) -> [map g] -> mapped_a (id=0) -\
        //                                            -> [zipWith add] -> result
        // b (id=101) -> [map h] -> mapped_b (id=1) -/

        let a = make_tensor_ref(100, &[100], DType::Float32);
        let b = make_tensor_ref(101, &[100], DType::Float32);

        let map_a = TensorOp::Map(make_map_fn("g"), a);
        let map_b = TensorOp::Map(make_map_fn("h"), b);

        // mapped_a gets id=0, mapped_b gets id=1 from build_fusible_ops
        let mapped_a = make_tensor_ref(0, &[100], DType::Float32);
        let mapped_b = make_tensor_ref(1, &[100], DType::Float32);

        let zip_op = TensorOp::ZipWith(
            ZipFn {
                name: Symbol::intern("add"),
                span: Span::DUMMY,
            },
            mapped_a,
            mapped_b,
        );

        let mut ctx = FusionContext::new(true);
        let ops = vec![map_a, map_b, zip_op];
        let kernels = fuse_ops(&mut ctx, ops);

        // Should produce single fused kernel
        assert_eq!(kernels.len(), 1, "should produce single fused kernel");
        assert!(kernels[0].fusion_info.complete, "fusion should be complete");
    }

    // ========================================================================
    // M2 Exit Criteria Integration Tests
    //
    // These tests verify the M2 milestone exit criteria per ROADMAP.md:
    // 1. sum (map f x) becomes single loop kernel
    // 2. Kernel report shows fusion succeeded
    // 3. reshape on contiguous tensor is metadata-only
    // ========================================================================

    /// M2 Exit Criterion 1: `sum (map f x)` becomes single loop kernel
    ///
    /// This test verifies that the H26-SPEC Section 8.1 Pattern 3 fuses correctly.
    /// The pattern `reduce op (map f x)` must fuse to a single traversal.
    #[test]
    fn test_m2_sum_map_fuses_to_single_kernel() {
        // Build the pattern: sum (map f x)
        //
        // x (id=100) -> [map f] -> intermediate (id=0) -> [sum] -> scalar result
        //
        // This MUST fuse to a single kernel per H26-SPEC Section 8.1 Pattern 3
        let input = make_tensor_ref(100, &[1000], DType::Float32);
        let map_op = TensorOp::Map(make_map_fn("square"), input);

        let intermediate = make_tensor_ref(0, &[1000], DType::Float32);
        let sum_op = TensorOp::ReduceAll(ReduceOp::Sum, intermediate);

        let mut ctx = FusionContext::new(true); // strict mode (Numeric Profile)
        let ops = vec![map_op, sum_op];
        let kernels = fuse_ops(&mut ctx, ops);

        // M2 Criterion: Must produce exactly ONE kernel
        assert_eq!(
            kernels.len(),
            1,
            "M2 FAIL: sum(map f x) did not fuse to single kernel"
        );

        // Verify the fusion is marked complete
        assert!(
            kernels[0].fusion_info.complete,
            "M2 FAIL: fusion not marked as complete"
        );

        // Verify it's the correct pattern (fused_reduce_map kernel)
        assert_eq!(
            kernels[0].name.as_str(),
            "fused_reduce_map",
            "M2 FAIL: kernel name should indicate reduce-map fusion"
        );

        // Verify both original operations are tracked
        assert_eq!(
            kernels[0].fusion_info.original_ops.len(),
            2,
            "M2 FAIL: should track both map and reduce operations"
        );
    }

    /// M2 Exit Criterion 2: Kernel report shows fusion succeeded
    ///
    /// This test verifies that the kernel report correctly indicates
    /// when fusion has succeeded for guaranteed patterns.
    #[test]
    fn test_m2_kernel_report_shows_fusion_success() {
        // Build: sum (map f x) - a guaranteed fusion pattern
        let input = make_tensor_ref(100, &[500], DType::Float64);
        let map_op = TensorOp::Map(make_map_fn("f"), input);
        let intermediate = make_tensor_ref(0, &[500], DType::Float64);
        let sum_op = TensorOp::ReduceAll(ReduceOp::Sum, intermediate);

        let mut ctx = FusionContext::new(true);
        let ops = vec![map_op, sum_op];
        let _kernels = fuse_ops(&mut ctx, ops);
        let report = generate_kernel_report(&ctx);

        // M2 Criterion: Report must show fusion succeeded
        assert_eq!(report.kernels.len(), 1, "M2 FAIL: should have 1 kernel");
        assert!(
            report.fused_ops > 0,
            "M2 FAIL: report should show fused operations"
        );
        assert_eq!(
            report.total_ops, 2,
            "M2 FAIL: report should track 2 original ops"
        );

        // Verify the kernel is marked as fused
        let kernel = &report.kernels[0];
        assert!(
            kernel.fusion_info.complete,
            "M2 FAIL: kernel report should indicate complete fusion"
        );

        // Verify decisions contain a Fused entry
        let has_fused_decision = report
            .decisions
            .iter()
            .any(|d| matches!(d, FusionDecision::Fused(_)));
        assert!(
            has_fused_decision,
            "M2 FAIL: report should contain Fused decision"
        );

        // Test report display output
        let report_str = format!("{report}");
        assert!(
            report_str.contains("Kernel Report"),
            "M2 FAIL: report should have header"
        );
        assert!(
            report_str.contains("Fused: YES"),
            "M2 FAIL: report should show 'Fused: YES'"
        );
    }

    /// M2 Exit Criterion 3: reshape on contiguous tensor is metadata-only
    ///
    /// This test verifies that reshaping a contiguous tensor does not
    /// require data movement - only metadata (shape/strides) changes.
    #[test]
    fn test_m2_reshape_contiguous_metadata_only() {
        // A contiguous tensor can be reshaped without copying data
        let contiguous = make_tensor_ref(0, &[6, 4], DType::Float32);

        // Verify contiguous tensor reshape is metadata-only
        assert!(
            is_reshape_metadata_only(&contiguous),
            "M2 FAIL: contiguous tensor reshape should be metadata-only"
        );

        // Build a more complex contiguous tensor
        let contiguous_3d = make_tensor_ref(1, &[2, 3, 4], DType::Float32);
        assert!(
            is_reshape_metadata_only(&contiguous_3d),
            "M2 FAIL: 3D contiguous tensor reshape should be metadata-only"
        );

        // Verify non-contiguous (strided) tensor requires data movement
        let strided = TensorRef {
            id: TensorId::new(2),
            meta: TensorMeta {
                dtype: DType::Float32,
                shape: Shape::from_static([6, 4]),
                strides: Strides::new([8, 1]), // Non-contiguous strides
                layout: crate::Layout::Strided,
                alias: None,
            },
        };
        assert!(
            !is_reshape_metadata_only(&strided),
            "M2 FAIL: strided tensor reshape should NOT be metadata-only"
        );
    }

    /// Integration test: Complete M2 pipeline with all guaranteed patterns
    ///
    /// Tests all H26-SPEC Section 8.1 guaranteed patterns:
    /// 1. map f (map g x)
    /// 2. zipWith f (map g a) (map h b)
    /// 3. sum (map f x)
    /// 4. foldl' op z (map f x) - (tested via ReduceMap pattern)
    #[test]
    fn test_m2_all_guaranteed_patterns_fuse() {
        // Pattern 1: map f (map g x)
        {
            let input = make_tensor_ref(100, &[100], DType::Float32);
            let map_g = TensorOp::Map(make_map_fn("g"), input);
            let intermediate = make_tensor_ref(0, &[100], DType::Float32);
            let map_f = TensorOp::Map(make_map_fn("f"), intermediate);

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_g, map_f]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 1 (map-map) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }

        // Pattern 2: zipWith f (map g a) (map h b)
        {
            let a = make_tensor_ref(100, &[100], DType::Float32);
            let b = make_tensor_ref(101, &[100], DType::Float32);
            let map_a = TensorOp::Map(make_map_fn("g"), a);
            let map_b = TensorOp::Map(make_map_fn("h"), b);
            let mapped_a = make_tensor_ref(0, &[100], DType::Float32);
            let mapped_b = make_tensor_ref(1, &[100], DType::Float32);
            let zip = TensorOp::ZipWith(
                ZipFn {
                    name: Symbol::intern("add"),
                    span: Span::DUMMY,
                },
                mapped_a,
                mapped_b,
            );

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_a, map_b, zip]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 2 (zipWith-maps) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }

        // Pattern 3: sum (map f x)
        {
            let input = make_tensor_ref(100, &[100], DType::Float32);
            let map_op = TensorOp::Map(make_map_fn("f"), input);
            let intermediate = make_tensor_ref(0, &[100], DType::Float32);
            let sum_op = TensorOp::ReduceAll(ReduceOp::Sum, intermediate);

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_op, sum_op]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 3 (reduce-map) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }

        // Pattern 3 variant: product (map f x)
        {
            let input = make_tensor_ref(100, &[100], DType::Float32);
            let map_op = TensorOp::Map(make_map_fn("f"), input);
            let intermediate = make_tensor_ref(0, &[100], DType::Float32);
            let prod_op = TensorOp::ReduceAll(ReduceOp::Prod, intermediate);

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_op, prod_op]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 3 variant (product-map) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }

        // Pattern 3 variant: max (map f x)
        {
            let input = make_tensor_ref(100, &[100], DType::Float32);
            let map_op = TensorOp::Map(make_map_fn("f"), input);
            let intermediate = make_tensor_ref(0, &[100], DType::Float32);
            let max_op = TensorOp::ReduceAll(ReduceOp::Max, intermediate);

            let mut ctx = FusionContext::new(true);
            let kernels = fuse_ops(&mut ctx, vec![map_op, max_op]);
            assert_eq!(
                kernels.len(),
                1,
                "Pattern 3 variant (max-map) should fuse to single kernel"
            );
            assert!(kernels[0].fusion_info.complete);
        }
    }

    // ========================================================================
    // ML Fusion Pattern Tests (H26-SPEC Section 8.2)
    // ========================================================================

    /// Test that SiLU pattern (x * sigmoid(x)) is detected and fused.
    #[test]
    fn test_silu_pattern_fusion() {
        // Build: x * sigmoid(x)
        // x (id=100) is the input
        // sigmoid(x) -> id=0
        // x * sigmoid(x) -> final output

        let x = make_tensor_ref(100, &[256], DType::Float32);

        // sigmoid(x) operation
        let sigmoid_op = TensorOp::Unary(crate::UnaryOp::Sigmoid, x.clone());

        // x * sigmoid(x)
        let sigmoid_out = make_tensor_ref(0, &[256], DType::Float32);
        let mul_op = TensorOp::ZipWith(
            ZipFn {
                name: Symbol::intern("*"),
                span: Span::DUMMY,
            },
            x.clone(),
            sigmoid_out,
        );

        let mut ctx = FusionContext::new(true);
        let ops = vec![sigmoid_op, mul_op];
        let kernels = fuse_ops(&mut ctx, ops);

        // Should produce a fused SiLU kernel
        assert_eq!(
            kernels.len(),
            1,
            "SiLU pattern should fuse to single kernel"
        );
        assert!(
            kernels[0].fusion_info.complete,
            "SiLU fusion should be complete"
        );
        assert_eq!(
            kernels[0].name.as_str(),
            "fused_silu",
            "Should produce fused_silu kernel"
        );
    }

    /// Test that fused linear with ReLU is detected.
    #[test]
    fn test_fused_linear_relu_pattern() {
        // Build: relu(matmul(x, w) + b)
        // x @ w -> matmul_out (id=0)
        // matmul_out + b -> add_out (id=1)
        // relu(add_out) -> final

        let x = make_tensor_ref(100, &[32, 128], DType::Float32);
        let w = make_tensor_ref(101, &[128, 64], DType::Float32);
        let b = make_tensor_ref(102, &[64], DType::Float32);

        // matmul(x, w)
        let matmul_op = TensorOp::MatMul(x.clone(), w.clone());

        // matmul + b
        let matmul_out = make_tensor_ref(0, &[32, 64], DType::Float32);
        let add_op = TensorOp::Binary(BinaryOp::Add, matmul_out, b.clone());

        // relu(add)
        let add_out = make_tensor_ref(1, &[32, 64], DType::Float32);
        let relu_op = TensorOp::Unary(crate::UnaryOp::Relu, add_out);

        let mut ctx = FusionContext::new(true);
        let ops = vec![matmul_op, add_op, relu_op];
        let kernels = fuse_ops(&mut ctx, ops);

        // Should produce a fused linear+bias+relu kernel
        assert_eq!(
            kernels.len(),
            1,
            "Linear+bias+relu should fuse to single kernel"
        );
        assert!(kernels[0].fusion_info.complete, "Fusion should be complete");
        assert_eq!(
            kernels[0].name.as_str(),
            "fused_linear_bias_relu",
            "Should produce fused_linear_bias_relu kernel"
        );
    }

    /// Test that matmul + bias without activation is detected.
    #[test]
    fn test_fused_linear_bias_only() {
        // Build: matmul(x, w) + b
        let x = make_tensor_ref(100, &[16, 256], DType::Float32);
        let w = make_tensor_ref(101, &[256, 128], DType::Float32);
        let b = make_tensor_ref(102, &[128], DType::Float32);

        let matmul_op = TensorOp::MatMul(x.clone(), w.clone());
        let matmul_out = make_tensor_ref(0, &[16, 128], DType::Float32);
        let add_op = TensorOp::Binary(BinaryOp::Add, matmul_out, b.clone());

        let mut ctx = FusionContext::new(true);
        let ops = vec![matmul_op, add_op];
        let kernels = fuse_ops(&mut ctx, ops);

        assert_eq!(kernels.len(), 1, "Linear+bias should fuse to single kernel");
        assert!(kernels[0].fusion_info.complete);
        assert_eq!(
            kernels[0].name.as_str(),
            "fused_linear_bias",
            "Should produce fused_linear_bias kernel"
        );
    }

    /// Test pattern detection helper functions.
    #[test]
    fn test_pattern_detection_helpers() {
        // Test division detection
        assert!(is_division_fn(&Symbol::intern("/")));
        assert!(is_division_fn(&Symbol::intern("div")));
        assert!(is_division_fn(&Symbol::intern("divide")));
        assert!(!is_division_fn(&Symbol::intern("add")));

        // Test subtraction detection
        assert!(is_subtraction_fn(&Symbol::intern("-")));
        assert!(is_subtraction_fn(&Symbol::intern("sub")));
        assert!(!is_subtraction_fn(&Symbol::intern("+")));

        // Test multiplication detection
        assert!(is_multiplication_fn(&Symbol::intern("*")));
        assert!(is_multiplication_fn(&Symbol::intern("mul")));
        assert!(!is_multiplication_fn(&Symbol::intern("div")));

        // Test exp detection
        assert!(is_exp_fn(&Symbol::intern("exp")));
        assert!(!is_exp_fn(&Symbol::intern("log")));

        // Test sqrt detection
        assert!(is_sqrt_fn(&Symbol::intern("sqrt")));
        assert!(!is_sqrt_fn(&Symbol::intern("abs")));

        // Test sigmoid detection
        assert!(is_sigmoid_fn(&Symbol::intern("sigmoid")));
        assert!(!is_sigmoid_fn(&Symbol::intern("relu")));

        // Test softmax detection
        assert!(is_softmax_fn(&Symbol::intern("softmax")));
        assert!(is_softmax_fn(&Symbol::intern("Softmax")));
        assert!(!is_softmax_fn(&Symbol::intern("sigmoid")));
    }

    /// Test FusedActivation enum completeness.
    #[test]
    fn test_fused_activation_variants() {
        let activations = vec![
            FusedActivation::Relu,
            FusedActivation::Gelu,
            FusedActivation::GeluFast,
            FusedActivation::Silu,
            FusedActivation::Sigmoid,
            FusedActivation::Tanh,
        ];

        // Verify all variants can be pattern matched
        for act in activations {
            match act {
                FusedActivation::Relu => assert!(true),
                FusedActivation::Gelu => assert!(true),
                FusedActivation::GeluFast => assert!(true),
                FusedActivation::Silu => assert!(true),
                FusedActivation::Sigmoid => assert!(true),
                FusedActivation::Tanh => assert!(true),
            }
        }
    }

    /// Test ML pattern kernel naming.
    #[test]
    fn test_ml_pattern_kernel_names() {
        // Create groups with different ML patterns and verify naming

        // Softmax
        let softmax_group = FusedGroup {
            ops: vec![],
            inputs: vec![],
            output: make_tensor_ref(0, &[256], DType::Float32),
            pattern: Some(FusionPattern::Softmax {
                input: make_tensor_ref(100, &[256], DType::Float32),
                axis: None,
            }),
            op_names: vec![],
        };
        assert_eq!(
            generate_kernel_name(&softmax_group).as_str(),
            "fused_softmax"
        );

        // LayerNorm
        let layernorm_group = FusedGroup {
            ops: vec![],
            inputs: vec![],
            output: make_tensor_ref(0, &[256], DType::Float32),
            pattern: Some(FusionPattern::LayerNorm {
                input: make_tensor_ref(100, &[256], DType::Float32),
                epsilon: 1e-5,
                axis: None,
                scale: None,
                bias: None,
            }),
            op_names: vec![],
        };
        assert_eq!(
            generate_kernel_name(&layernorm_group).as_str(),
            "fused_layernorm_welford"
        );

        // Attention (non-causal)
        let attention_group = FusedGroup {
            ops: vec![],
            inputs: vec![],
            output: make_tensor_ref(0, &[1, 8, 512, 64], DType::Float32),
            pattern: Some(FusionPattern::Attention {
                query: make_tensor_ref(100, &[1, 8, 512, 64], DType::Float32),
                key: make_tensor_ref(101, &[1, 8, 512, 64], DType::Float32),
                value: make_tensor_ref(102, &[1, 8, 512, 64], DType::Float32),
                mask: None,
                scale: 0.125,
                causal: false,
            }),
            op_names: vec![],
        };
        assert_eq!(
            generate_kernel_name(&attention_group).as_str(),
            "fused_attention"
        );

        // Causal Attention
        let causal_attention_group = FusedGroup {
            ops: vec![],
            inputs: vec![],
            output: make_tensor_ref(0, &[1, 8, 512, 64], DType::Float32),
            pattern: Some(FusionPattern::Attention {
                query: make_tensor_ref(100, &[1, 8, 512, 64], DType::Float32),
                key: make_tensor_ref(101, &[1, 8, 512, 64], DType::Float32),
                value: make_tensor_ref(102, &[1, 8, 512, 64], DType::Float32),
                mask: None,
                scale: 0.125,
                causal: true,
            }),
            op_names: vec![],
        };
        assert_eq!(
            generate_kernel_name(&causal_attention_group).as_str(),
            "fused_attention_causal"
        );

        // GELU
        let gelu_group = FusedGroup {
            ops: vec![],
            inputs: vec![],
            output: make_tensor_ref(0, &[256], DType::Float32),
            pattern: Some(FusionPattern::Gelu {
                input: make_tensor_ref(100, &[256], DType::Float32),
                fast: false,
            }),
            op_names: vec![],
        };
        assert_eq!(generate_kernel_name(&gelu_group).as_str(), "fused_gelu");

        // GELU Fast
        let gelu_fast_group = FusedGroup {
            ops: vec![],
            inputs: vec![],
            output: make_tensor_ref(0, &[256], DType::Float32),
            pattern: Some(FusionPattern::Gelu {
                input: make_tensor_ref(100, &[256], DType::Float32),
                fast: true,
            }),
            op_names: vec![],
        };
        assert_eq!(
            generate_kernel_name(&gelu_fast_group).as_str(),
            "fused_gelu_fast"
        );

        // SiLU
        let silu_group = FusedGroup {
            ops: vec![],
            inputs: vec![],
            output: make_tensor_ref(0, &[256], DType::Float32),
            pattern: Some(FusionPattern::Silu {
                input: make_tensor_ref(100, &[256], DType::Float32),
            }),
            op_names: vec![],
        };
        assert_eq!(generate_kernel_name(&silu_group).as_str(), "fused_silu");
    }

    /// Integration test: Verify ML patterns are correctly classified.
    #[test]
    fn test_ml_pattern_classification() {
        // Ensure new patterns don't interfere with existing patterns

        // Pattern 1 should still work
        let input = make_tensor_ref(100, &[256], DType::Float32);
        let map1 = TensorOp::Map(make_map_fn("double"), input);
        let intermediate = make_tensor_ref(0, &[256], DType::Float32);
        let map2 = TensorOp::Map(make_map_fn("inc"), intermediate);

        let mut ctx = FusionContext::new(true);
        let kernels = fuse_ops(&mut ctx, vec![map1, map2]);
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].name.as_str(), "fused_map_map");

        // Pattern 3 should still work
        let input2 = make_tensor_ref(100, &[256], DType::Float32);
        let map_op = TensorOp::Map(make_map_fn("square"), input2);
        let mapped = make_tensor_ref(0, &[256], DType::Float32);
        let sum_op = TensorOp::ReduceAll(ReduceOp::Sum, mapped);

        let mut ctx2 = FusionContext::new(true);
        let kernels2 = fuse_ops(&mut ctx2, vec![map_op, sum_op]);
        assert_eq!(kernels2.len(), 1);
        assert_eq!(kernels2[0].name.as_str(), "fused_reduce_map");
    }
}
