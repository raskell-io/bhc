//! # Lowering from Tensor IR to Loop IR
//!
//! This module converts fused Tensor IR kernels into explicit Loop IR
//! with iteration structure suitable for vectorization and code generation.
//!
//! ## Pipeline Position
//!
//! ```text
//! Tensor IR (fused kernels) → [lower.rs] → Loop IR → [vectorize.rs] → Vectorized Loop IR
//! ```
//!
//! ## Key Transformations
//!
//! 1. **Kernel to Function**: Each kernel becomes a Loop IR function
//! 2. **Shape to Loops**: Tensor shapes become loop nests
//! 3. **Operations to Statements**: Tensor ops become scalar statements
//! 4. **Access Patterns**: Memory access patterns are computed from strides

use crate::{
    AccessPattern, Alloc, BinOp, Body, Loop, LoopAttrs, LoopIR, LoopId, LoopMetadata, LoopType,
    MemRef, Op, Param, ScalarType, Stmt, TargetArch, TripCount, Value, ValueId,
};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_tensor_ir::{
    BufferId, Kernel, KernelBody, LoopNest as TensorLoopNest, ReduceOp as TensorReduceOp, TensorOp,
    TensorRef,
};
use rustc_hash::FxHashMap;
use thiserror::Error;

/// Errors that can occur during lowering.
#[derive(Clone, Debug, Error)]
pub enum LowerError {
    /// Unsupported tensor operation.
    #[error("unsupported tensor operation: {op}")]
    UnsupportedOp {
        /// Description of the unsupported operation.
        op: String,
    },

    /// Shape mismatch during lowering.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Invalid kernel structure.
    #[error("invalid kernel structure: {reason}")]
    InvalidKernel {
        /// Reason for the error.
        reason: String,
    },
}

/// Configuration for lowering.
#[derive(Clone, Debug)]
pub struct LowerConfig {
    /// Target architecture for vectorization hints.
    pub target: TargetArch,
    /// Whether to mark loops as potentially vectorizable.
    pub enable_vectorization: bool,
    /// Whether to mark loops as potentially parallelizable.
    pub enable_parallelization: bool,
    /// Minimum trip count for vectorization.
    pub vectorize_threshold: usize,
    /// Minimum trip count for parallelization.
    pub parallelize_threshold: usize,
}

impl Default for LowerConfig {
    fn default() -> Self {
        Self {
            target: TargetArch::default(),
            enable_vectorization: true,
            enable_parallelization: true,
            vectorize_threshold: 4,
            parallelize_threshold: 1024,
        }
    }
}

/// Context for lowering a single kernel.
struct LowerContext {
    /// Configuration.
    config: LowerConfig,
    /// Next value ID.
    next_value: u32,
    /// Next loop ID.
    next_loop: u32,
    /// Mapping from tensor refs to value IDs.
    tensor_values: FxHashMap<u64, ValueId>,
    /// Allocations for the lowered function.
    allocations: Vec<Alloc>,
    /// Loop metadata accumulated during lowering.
    loop_metadata: Vec<LoopMetadata>,
    /// Parameters for the lowered function.
    params: Vec<Param>,
}

impl LowerContext {
    fn new(config: LowerConfig) -> Self {
        Self {
            config,
            next_value: 0,
            next_loop: 0,
            tensor_values: FxHashMap::default(),
            allocations: Vec::new(),
            loop_metadata: Vec::new(),
            params: Vec::new(),
        }
    }

    fn fresh_value(&mut self) -> ValueId {
        let id = ValueId::new(self.next_value as usize);
        self.next_value += 1;
        id
    }

    fn fresh_loop(&mut self) -> LoopId {
        let id = LoopId::new(self.next_loop as usize);
        self.next_loop += 1;
        id
    }
}

/// Lower a collection of fused kernels to Loop IR.
///
/// # Arguments
///
/// * `kernels` - The fused kernels from Tensor IR
/// * `config` - Lowering configuration
///
/// # Returns
///
/// A vector of lowered Loop IR functions.
pub fn lower_kernels(kernels: &[Kernel], config: LowerConfig) -> Result<Vec<LoopIR>, LowerError> {
    kernels
        .iter()
        .map(|k| lower_kernel(k, config.clone()))
        .collect()
}

/// Lower a single kernel to Loop IR.
pub fn lower_kernel(kernel: &Kernel, config: LowerConfig) -> Result<LoopIR, LowerError> {
    let mut ctx = LowerContext::new(config);

    // Add input tensors as parameters
    for (i, input) in kernel.inputs.iter().enumerate() {
        let param = tensor_ref_to_param(input, i, &mut ctx);
        ctx.params.push(param);
    }

    // Add output tensors as parameters (mutable)
    for (i, output) in kernel.outputs.iter().enumerate() {
        let param = tensor_ref_to_param(output, kernel.inputs.len() + i, &mut ctx);
        ctx.params.push(param);
    }

    // Lower the kernel body
    let body = match &kernel.body {
        KernelBody::Fused(ops) => lower_fused_ops(ops, kernel, &mut ctx)?,
        KernelBody::LoopNest(nest) => lower_tensor_loop_nest(nest, &mut ctx)?,
    };

    Ok(LoopIR {
        name: kernel.name,
        params: ctx.params,
        return_ty: LoopType::Void,
        body,
        allocs: ctx.allocations,
        loop_info: ctx.loop_metadata,
    })
}

/// Convert a tensor reference to a function parameter.
fn tensor_ref_to_param(tensor: &TensorRef, index: usize, ctx: &mut LowerContext) -> Param {
    let elem_ty = ScalarType::from_dtype(tensor.meta.dtype);
    let value_id = ctx.fresh_value();

    // Register the tensor -> value mapping
    ctx.tensor_values.insert(tensor.id.index() as u64, value_id);

    Param {
        name: Symbol::intern(&format!("tensor_{}", index)),
        ty: LoopType::Ptr(Box::new(LoopType::Scalar(elem_ty))),
        is_ptr: true,
    }
}

/// Lower fused tensor operations to a loop body.
fn lower_fused_ops(
    ops: &[TensorOp],
    kernel: &Kernel,
    ctx: &mut LowerContext,
) -> Result<Body, LowerError> {
    // For fused ops, we generate a single loop nest over the output shape
    // The innermost loop body contains all the fused operations

    // Get the output shape from the first output tensor
    let output_shape: Vec<usize> = if let Some(output) = kernel.outputs.first() {
        output
            .meta
            .shape
            .dims()
            .iter()
            .map(|d| d.static_value().unwrap_or(0))
            .collect()
    } else {
        return Err(LowerError::InvalidKernel {
            reason: "kernel has no outputs".to_string(),
        });
    };

    // Generate loop nest for the output shape
    let (body, loop_vars) = generate_loop_nest(&output_shape, ctx)?;

    // Generate the inner loop body with fused operations
    let inner_stmts = lower_fused_ops_body(ops, &loop_vars, kernel, ctx)?;

    // Insert the inner statements into the innermost loop
    let mut result_body = body;
    insert_inner_stmts(&mut result_body, inner_stmts);

    Ok(result_body)
}

/// Generate a loop nest for the given shape.
/// Returns the body with nested loops and the loop variables.
fn generate_loop_nest(
    shape: &[usize],
    ctx: &mut LowerContext,
) -> Result<(Body, Vec<ValueId>), LowerError> {
    let mut loop_vars = Vec::with_capacity(shape.len());
    let mut loops = Vec::with_capacity(shape.len());

    for (dim_idx, &dim_size) in shape.iter().enumerate() {
        let loop_id = ctx.fresh_loop();
        let loop_var = ctx.fresh_value();
        loop_vars.push(loop_var);

        // Determine loop attributes based on config
        let mut attrs = LoopAttrs::INDEPENDENT;

        // Mark outer loops as potentially parallel
        if ctx.config.enable_parallelization
            && dim_idx == 0
            && dim_size >= ctx.config.parallelize_threshold
        {
            attrs |= LoopAttrs::PARALLEL;
        }

        // Mark innermost loop as potentially vectorizable
        if ctx.config.enable_vectorization
            && dim_idx == shape.len() - 1
            && dim_size >= ctx.config.vectorize_threshold
        {
            attrs |= LoopAttrs::VECTORIZE;
        }

        // Create loop metadata
        ctx.loop_metadata.push(LoopMetadata {
            id: loop_id,
            trip_count: TripCount::Static(dim_size),
            vector_width: None,   // Will be filled by vectorization pass
            parallel_chunk: None, // Will be filled by parallelization pass
            unroll_factor: None,
            dependencies: Vec::new(),
        });

        loops.push(Loop {
            id: loop_id,
            var: loop_var,
            lower: Value::i64(0),
            upper: Value::i64(dim_size as i64),
            step: Value::i64(1),
            body: Body::new(),
            attrs,
        });
    }

    // Build nested structure: outermost loop contains next loop, etc.
    let mut body = Body::new();
    if loops.is_empty() {
        return Ok((body, loop_vars));
    }

    // Nest loops from innermost to outermost
    let mut current_loop = loops.pop().unwrap();
    while let Some(mut outer) = loops.pop() {
        outer.body.push(Stmt::Loop(current_loop));
        current_loop = outer;
    }

    body.push(Stmt::Loop(current_loop));
    Ok((body, loop_vars))
}

/// Lower fused operations to statements.
fn lower_fused_ops_body(
    ops: &[TensorOp],
    loop_vars: &[ValueId],
    _kernel: &Kernel,
    ctx: &mut LowerContext,
) -> Result<Vec<Stmt>, LowerError> {
    let mut stmts = Vec::new();

    for op in ops {
        lower_tensor_op(op, loop_vars, &mut stmts, ctx)?;
    }

    Ok(stmts)
}

/// Lower a single tensor operation.
fn lower_tensor_op(
    op: &TensorOp,
    loop_vars: &[ValueId],
    stmts: &mut Vec<Stmt>,
    ctx: &mut LowerContext,
) -> Result<(), LowerError> {
    match op {
        TensorOp::Map(_func, input) => {
            // Load input element
            let input_val = load_tensor_element(input, loop_vars, stmts, ctx)?;

            // Apply function (simplified: assume unary arithmetic)
            let result = ctx.fresh_value();
            stmts.push(Stmt::Assign(result, Op::Unary(crate::UnOp::Neg, input_val)));

            Ok(())
        }

        TensorOp::ZipWith(_func, a, b) => {
            // Load both input elements
            let a_val = load_tensor_element(a, loop_vars, stmts, ctx)?;
            let b_val = load_tensor_element(b, loop_vars, stmts, ctx)?;

            // Apply binary function
            let result = ctx.fresh_value();
            stmts.push(Stmt::Assign(result, Op::Binary(BinOp::Add, a_val, b_val)));

            Ok(())
        }

        TensorOp::ReduceAll(reduce_op, input) => {
            lower_reduction(reduce_op, input, loop_vars, stmts, ctx)
        }

        TensorOp::Broadcast(_shape, input) => {
            // Broadcast is handled by adjusting memory access patterns
            let _ = load_tensor_element(input, loop_vars, stmts, ctx)?;
            Ok(())
        }

        TensorOp::Reshape(_shape, input) => {
            // Reshape is metadata-only for contiguous tensors
            let _ = load_tensor_element(input, loop_vars, stmts, ctx)?;
            Ok(())
        }

        TensorOp::Transpose(_perm, input) => {
            // Transpose adjusts strides
            let _ = load_tensor_element(input, loop_vars, stmts, ctx)?;
            Ok(())
        }

        _ => Err(LowerError::UnsupportedOp {
            op: format!("{:?}", std::mem::discriminant(op)),
        }),
    }
}

/// Load a tensor element at the current loop indices.
fn load_tensor_element(
    tensor: &TensorRef,
    loop_vars: &[ValueId],
    stmts: &mut Vec<Stmt>,
    ctx: &mut LowerContext,
) -> Result<Value, LowerError> {
    let elem_ty = ScalarType::from_dtype(tensor.meta.dtype);

    // Compute linear index from loop variables and strides
    let index = compute_linear_index(tensor, loop_vars)?;

    // Get buffer ID from tensor metadata
    let buffer_id = tensor
        .meta
        .alias
        .unwrap_or(BufferId::new(tensor.id.index()));

    // Create memory reference
    let mem_ref = MemRef {
        buffer: buffer_id,
        index,
        elem_ty: LoopType::Scalar(elem_ty),
        access: compute_access_pattern(tensor),
    };

    // Generate load
    let result = ctx.fresh_value();
    stmts.push(Stmt::Assign(result, Op::Load(mem_ref)));

    Ok(Value::Var(result, LoopType::Scalar(elem_ty)))
}

/// Compute linear index from loop variables and tensor strides.
fn compute_linear_index(_tensor: &TensorRef, loop_vars: &[ValueId]) -> Result<Value, LowerError> {
    // For a tensor with shape [N, M, K] and strides [s0, s1, s2],
    // the linear index is: i*s0 + j*s1 + k*s2

    if loop_vars.is_empty() {
        return Ok(Value::i64(0));
    }

    // For now, return a simple index using the first loop var
    // In a full implementation, we'd build the affine index expression
    let first_var = loop_vars[0];
    Ok(Value::Var(first_var, LoopType::Scalar(ScalarType::I64)))
}

/// Compute the memory access pattern for a tensor.
fn compute_access_pattern(tensor: &TensorRef) -> AccessPattern {
    let strides = tensor.meta.strides.values();

    // If innermost stride is 1, access is sequential
    if strides.last() == Some(&1) {
        AccessPattern::Sequential
    } else if let Some(&stride) = strides.last() {
        AccessPattern::Strided(stride)
    } else {
        AccessPattern::Random
    }
}

/// Lower a reduction operation.
fn lower_reduction(
    reduce_op: &TensorReduceOp,
    input: &TensorRef,
    loop_vars: &[ValueId],
    stmts: &mut Vec<Stmt>,
    ctx: &mut LowerContext,
) -> Result<(), LowerError> {
    let elem_ty = ScalarType::from_dtype(input.meta.dtype);
    let bits = elem_ty.size_bytes() as u8 * 8;

    // Initialize accumulator (comment as placeholder)
    let _init_val = match reduce_op {
        TensorReduceOp::Sum => Value::float(0.0, bits),
        TensorReduceOp::Prod => Value::float(1.0, bits),
        TensorReduceOp::Min => Value::float(f64::INFINITY, bits),
        TensorReduceOp::Max => Value::float(f64::NEG_INFINITY, bits),
        _ => Value::float(0.0, bits),
    };

    // Add accumulator initialization (will be at function start)
    stmts.push(Stmt::Comment(format!(
        "reduction accumulator for {:?}",
        reduce_op
    )));

    // Initialize acc value
    let acc = ctx.fresh_value();

    // Load input element
    let input_val = load_tensor_element(input, loop_vars, stmts, ctx)?;

    // Update accumulator
    let bin_op = match reduce_op {
        TensorReduceOp::Sum => BinOp::Add,
        TensorReduceOp::Prod => BinOp::Mul,
        TensorReduceOp::Min => BinOp::FMin,
        TensorReduceOp::Max => BinOp::FMax,
        _ => BinOp::Add,
    };

    let new_acc = ctx.fresh_value();
    stmts.push(Stmt::Assign(
        new_acc,
        Op::Binary(
            bin_op,
            Value::Var(acc, LoopType::Scalar(elem_ty)),
            input_val,
        ),
    ));

    Ok(())
}

/// Lower a Tensor IR loop nest to Loop IR.
fn lower_tensor_loop_nest(
    nest: &TensorLoopNest,
    ctx: &mut LowerContext,
) -> Result<Body, LowerError> {
    // TensorLoopNest already has explicit loop structure
    // Convert it to Loop IR format
    let mut loops = Vec::new();

    for loop_spec in &nest.loops {
        let loop_id = ctx.fresh_loop();
        let loop_var = ctx.fresh_value();

        let mut attrs = LoopAttrs::empty();
        if loop_spec.parallel {
            attrs |= LoopAttrs::PARALLEL;
        }
        if loop_spec.vectorize.is_some() {
            attrs |= LoopAttrs::VECTORIZE;
        }

        let trip_count = loop_spec
            .upper
            .static_value()
            .map(TripCount::Static)
            .unwrap_or(TripCount::Dynamic);

        let upper_bound = loop_spec.upper.static_value().unwrap_or(0) as i64;

        ctx.loop_metadata.push(LoopMetadata {
            id: loop_id,
            trip_count,
            vector_width: None,
            parallel_chunk: None,
            unroll_factor: None,
            dependencies: Vec::new(),
        });

        loops.push(Loop {
            id: loop_id,
            var: loop_var,
            lower: Value::i64(loop_spec.lower),
            upper: Value::i64(upper_bound),
            step: Value::i64(loop_spec.step),
            body: Body::new(),
            attrs,
        });
    }

    // Build nested structure
    let mut body = Body::new();
    if loops.is_empty() {
        return Ok(body);
    }

    let mut current_loop = loops.pop().unwrap();
    while let Some(mut outer) = loops.pop() {
        outer.body.push(Stmt::Loop(current_loop));
        current_loop = outer;
    }

    body.push(Stmt::Loop(current_loop));
    Ok(body)
}

/// Insert statements into the innermost loop body.
fn insert_inner_stmts(body: &mut Body, stmts: Vec<Stmt>) {
    fn find_innermost_and_insert(body: &mut Body, stmts: Vec<Stmt>) {
        if let Some(Stmt::Loop(ref mut lp)) = body.stmts.last_mut() {
            if lp.body.stmts.is_empty() || !matches!(lp.body.stmts.last(), Some(Stmt::Loop(_))) {
                // This is the innermost loop
                lp.body.stmts.extend(stmts);
            } else {
                // Recurse into nested loop
                find_innermost_and_insert(&mut lp.body, stmts);
            }
        } else {
            // No loops, just add statements directly
            body.stmts.extend(stmts);
        }
    }

    find_innermost_and_insert(body, stmts);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_span::Span;
    use bhc_tensor_ir::{
        DType, FusionInfo, KernelId, Layout, MapFn, Shape, Strides, TensorId, TensorMeta,
    };

    fn make_test_kernel() -> Kernel {
        let meta = TensorMeta {
            dtype: DType::Float32,
            shape: Shape::from_static([1024]),
            strides: Strides::new([1]),
            layout: Layout::Contiguous,
            alias: None,
        };

        let input = TensorRef {
            id: TensorId::new(0),
            meta: meta.clone(),
        };

        let output = TensorRef {
            id: TensorId::new(1),
            meta,
        };

        let map_fn = MapFn {
            name: Symbol::intern("f"),
            span: Span::DUMMY,
        };

        Kernel {
            id: KernelId::new(0),
            name: Symbol::intern("test_kernel"),
            inputs: vec![input.clone()],
            outputs: vec![output],
            body: KernelBody::Fused(vec![TensorOp::Map(map_fn, input)]),
            allocs: vec![],
            fusion_info: FusionInfo {
                original_ops: vec![],
                decisions: vec![],
                complete: true,
            },
        }
    }

    #[test]
    fn test_lower_simple_kernel() {
        let kernel = make_test_kernel();
        let config = LowerConfig::default();

        let result = lower_kernel(&kernel, config);
        assert!(result.is_ok());

        let loop_ir = result.unwrap();
        assert_eq!(loop_ir.name.as_str(), "test_kernel");
        assert_eq!(loop_ir.params.len(), 2); // input + output
        assert!(!loop_ir.body.stmts.is_empty());
    }

    #[test]
    fn test_lower_generates_loop_nest() {
        let kernel = make_test_kernel();
        let config = LowerConfig::default();

        let loop_ir = lower_kernel(&kernel, config).unwrap();

        // Should have a loop in the body
        assert!(matches!(loop_ir.body.stmts.first(), Some(Stmt::Loop(_))));
    }

    #[test]
    fn test_lower_marks_vectorizable() {
        let kernel = make_test_kernel();
        let mut config = LowerConfig::default();
        config.enable_vectorization = true;
        config.vectorize_threshold = 4;

        let loop_ir = lower_kernel(&kernel, config).unwrap();

        // Find the loop and check attributes
        if let Some(Stmt::Loop(lp)) = loop_ir.body.stmts.first() {
            assert!(lp.attrs.contains(LoopAttrs::VECTORIZE));
        }
    }

    #[test]
    fn test_sequential_access_pattern() {
        let meta = TensorMeta {
            dtype: DType::Float32,
            shape: Shape::from_static([1024]),
            strides: Strides::new([1]), // Contiguous
            layout: Layout::Contiguous,
            alias: None,
        };

        let tensor = TensorRef {
            id: TensorId::new(0),
            meta,
        };

        let pattern = compute_access_pattern(&tensor);
        assert_eq!(pattern, AccessPattern::Sequential);
    }

    #[test]
    fn test_strided_access_pattern() {
        let meta = TensorMeta {
            dtype: DType::Float32,
            shape: Shape::from_static([1024]),
            strides: Strides::new([4]), // Non-contiguous
            layout: Layout::Strided,
            alias: None,
        };

        let tensor = TensorRef {
            id: TensorId::new(0),
            meta,
        };

        let pattern = compute_access_pattern(&tensor);
        assert_eq!(pattern, AccessPattern::Strided(4));
    }
}
