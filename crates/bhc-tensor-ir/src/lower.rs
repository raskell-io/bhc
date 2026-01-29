//! Core IR to Tensor IR lowering pass.
//!
//! This module implements the lowering from Core IR to Tensor IR for the
//! Numeric Profile. It recognizes common patterns on tensor operations
//! and converts them to the Tensor IR representation.
//!
//! # Recognized Patterns
//!
//! The lowering pass recognizes these Core patterns:
//!
//! - `map f xs` → [`TensorOp::Map`]
//! - `zipWith f xs ys` → [`TensorOp::ZipWith`]
//! - `sum xs` → [`TensorOp::ReduceAll`] with [`ReduceOp::Sum`]
//! - `product xs` → [`TensorOp::ReduceAll`] with [`ReduceOp::Prod`]
//! - `foldl' f z xs` → [`TensorOp::Fold`]
//! - `reshape shape xs` → [`TensorOp::Reshape`]
//! - `transpose xs` → [`TensorOp::Transpose`]
//! - `slice spec xs` → [`TensorOp::Slice`]
//!
//! # See Also
//!
//! - H26-SPEC Section 7.3 for Tensor IR specification
//! - [`crate::fusion`] for the fusion pass that operates on Tensor IR

use bhc_core::{Expr, Literal, Var, VarId};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::Ty;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::{
    DType, Dim, FoldFn, Layout, MapFn, Permutation, ReduceOp, Shape, SliceRange, SliceSpec,
    Strides, TensorId, TensorMeta, TensorOp, TensorRef, ZipFn,
};

/// Context for the lowering pass.
///
/// Tracks state during lowering including fresh ID generation
/// and variable-to-tensor mappings.
#[derive(Debug)]
pub struct LowerContext {
    /// Next tensor ID to allocate.
    next_tensor_id: u32,
    /// Next buffer ID to allocate.
    #[allow(dead_code)]
    next_buffer_id: u32,
    /// Mapping from Core variables to tensor references.
    var_tensors: FxHashMap<VarId, TensorRef>,
    /// Generated tensor operations.
    ops: Vec<TensorOp>,
    /// Known built-in function names.
    builtins: BuiltinTable,
}

/// Table of recognized built-in function names.
#[derive(Debug)]
struct BuiltinTable {
    /// The `map` function symbol.
    map: Option<Symbol>,
    /// The `zipWith` function symbol.
    zip_with: Option<Symbol>,
    /// The `sum` function symbol.
    sum: Option<Symbol>,
    /// The `product` function symbol.
    product: Option<Symbol>,
    /// The `foldl'` function symbol.
    foldl: Option<Symbol>,
    /// The `foldr` function symbol.
    #[allow(dead_code)]
    foldr: Option<Symbol>,
    /// The `reshape` function symbol.
    reshape: Option<Symbol>,
    /// The `transpose` function symbol.
    transpose: Option<Symbol>,
    /// The `slice` function symbol.
    slice: Option<Symbol>,
    /// The `broadcast` function symbol.
    broadcast: Option<Symbol>,
    /// The `matmul` function symbol.
    matmul: Option<Symbol>,
    /// The `dot` function symbol.
    dot: Option<Symbol>,
}

impl BuiltinTable {
    /// Creates a new builtin table with symbols resolved from an interner.
    fn new() -> Self {
        Self {
            map: None,
            zip_with: None,
            sum: None,
            product: None,
            foldl: None,
            foldr: None,
            reshape: None,
            transpose: None,
            slice: None,
            broadcast: None,
            matmul: None,
            dot: None,
        }
    }

    /// Initializes builtins from known symbol values.
    ///
    /// This should be called with the actual interned symbols for
    /// the built-in tensor operations.
    #[allow(clippy::too_many_arguments)]
    fn with_symbols(
        map: Symbol,
        zip_with: Symbol,
        sum: Symbol,
        product: Symbol,
        foldl: Symbol,
        foldr: Symbol,
        reshape: Symbol,
        transpose: Symbol,
        slice: Symbol,
        broadcast: Symbol,
        matmul: Symbol,
        dot: Symbol,
    ) -> Self {
        Self {
            map: Some(map),
            zip_with: Some(zip_with),
            sum: Some(sum),
            product: Some(product),
            foldl: Some(foldl),
            foldr: Some(foldr),
            reshape: Some(reshape),
            transpose: Some(transpose),
            slice: Some(slice),
            broadcast: Some(broadcast),
            matmul: Some(matmul),
            dot: Some(dot),
        }
    }

    /// Checks if a symbol is the `map` function.
    fn is_map(&self, sym: Symbol) -> bool {
        self.map.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `zipWith` function.
    fn is_zip_with(&self, sym: Symbol) -> bool {
        self.zip_with.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `sum` function.
    fn is_sum(&self, sym: Symbol) -> bool {
        self.sum.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `product` function.
    fn is_product(&self, sym: Symbol) -> bool {
        self.product.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `foldl'` function.
    fn is_foldl(&self, sym: Symbol) -> bool {
        self.foldl.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `reshape` function.
    fn is_reshape(&self, sym: Symbol) -> bool {
        self.reshape.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `transpose` function.
    fn is_transpose(&self, sym: Symbol) -> bool {
        self.transpose.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `slice` function.
    fn is_slice(&self, sym: Symbol) -> bool {
        self.slice.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `broadcast` function.
    fn is_broadcast(&self, sym: Symbol) -> bool {
        self.broadcast.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `matmul` function.
    fn is_matmul(&self, sym: Symbol) -> bool {
        self.matmul.is_some_and(|s| s == sym)
    }

    /// Checks if a symbol is the `dot` function.
    fn is_dot(&self, sym: Symbol) -> bool {
        self.dot.is_some_and(|s| s == sym)
    }
}

/// Result of lowering - either a tensor reference or indicates
/// the expression doesn't lower to tensor ops.
#[derive(Debug, Clone)]
pub enum LowerResult {
    /// Expression lowered to a tensor.
    Tensor(TensorRef),
    /// Expression is a scalar value.
    Scalar(ScalarValue),
    /// Expression is a function (not lowered).
    Function,
    /// Expression cannot be lowered to tensor IR.
    NotTensor,
}

/// A scalar value from lowering.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    /// Integer literal.
    Int(i64),
    /// Float literal.
    Float(f64),
    /// Variable reference.
    Var(VarId),
}

impl LowerContext {
    /// Creates a new lowering context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_tensor_id: 0,
            next_buffer_id: 0,
            var_tensors: FxHashMap::default(),
            ops: Vec::new(),
            builtins: BuiltinTable::new(),
        }
    }

    /// Creates a lowering context with known builtin symbols.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn with_builtins(
        map: Symbol,
        zip_with: Symbol,
        sum: Symbol,
        product: Symbol,
        foldl: Symbol,
        foldr: Symbol,
        reshape: Symbol,
        transpose: Symbol,
        slice: Symbol,
        broadcast: Symbol,
        matmul: Symbol,
        dot: Symbol,
    ) -> Self {
        Self {
            next_tensor_id: 0,
            next_buffer_id: 0,
            var_tensors: FxHashMap::default(),
            ops: Vec::new(),
            builtins: BuiltinTable::with_symbols(
                map, zip_with, sum, product, foldl, foldr, reshape, transpose, slice, broadcast,
                matmul, dot,
            ),
        }
    }

    /// Allocates a fresh tensor ID.
    fn fresh_tensor_id(&mut self) -> TensorId {
        let id = TensorId::new(self.next_tensor_id as usize);
        self.next_tensor_id += 1;
        id
    }

    /// Registers a tensor for a variable.
    pub fn register_tensor(&mut self, var_id: VarId, tensor: TensorRef) {
        self.var_tensors.insert(var_id, tensor);
    }

    /// Looks up the tensor for a variable.
    pub fn lookup_tensor(&self, var_id: VarId) -> Option<&TensorRef> {
        self.var_tensors.get(&var_id)
    }

    /// Returns the generated operations.
    #[must_use]
    pub fn into_ops(self) -> Vec<TensorOp> {
        self.ops
    }

    /// Returns the operations as a reference.
    #[must_use]
    pub fn ops(&self) -> &[TensorOp] {
        &self.ops
    }

    /// Lowers a Core expression to Tensor IR.
    ///
    /// Returns the result of lowering, which may be a tensor reference,
    /// a scalar value, or an indication that the expression cannot be lowered.
    pub fn lower_expr(&mut self, expr: &Expr) -> LowerResult {
        match expr {
            Expr::Var(var, _) => self.lower_var(var),
            Expr::Lit(lit, ty, _) => self.lower_lit(lit, ty),
            Expr::App(f, arg, span) => self.lower_app(f, arg, *span),
            Expr::Let(bind, body, _) => self.lower_let(bind, body),
            Expr::Lam(_, _, _) => LowerResult::Function,
            Expr::TyLam(_, body, _) => self.lower_expr(body),
            Expr::TyApp(f, _, _) => self.lower_expr(f),
            Expr::Case(_, _, _, _) => LowerResult::NotTensor,
            Expr::Cast(e, _, _) => self.lower_expr(e),
            Expr::Tick(_, e, _) => self.lower_expr(e),
            Expr::Lazy(e, _) => self.lower_expr(e),
            Expr::Type(_, _) | Expr::Coercion(_, _) => LowerResult::NotTensor,
        }
    }

    /// Lowers a variable reference.
    fn lower_var(&self, var: &Var) -> LowerResult {
        if let Some(tensor) = self.var_tensors.get(&var.id) {
            LowerResult::Tensor(tensor.clone())
        } else {
            LowerResult::Scalar(ScalarValue::Var(var.id))
        }
    }

    /// Lowers a literal.
    fn lower_lit(&self, lit: &Literal, _ty: &Ty) -> LowerResult {
        match lit {
            Literal::Int(n) => LowerResult::Scalar(ScalarValue::Int(*n)),
            Literal::Integer(n) => LowerResult::Scalar(ScalarValue::Int(*n as i64)),
            Literal::Float(f) => LowerResult::Scalar(ScalarValue::Float(f64::from(*f))),
            Literal::Double(d) => LowerResult::Scalar(ScalarValue::Float(*d)),
            Literal::Char(_) | Literal::String(_) => LowerResult::NotTensor,
        }
    }

    /// Lowers a function application.
    ///
    /// This is where most tensor operations are recognized.
    fn lower_app(&mut self, f: &Expr, arg: &Expr, span: Span) -> LowerResult {
        // Collect all arguments (function applications are curried)
        let (func, args) = collect_app_args(f, arg);

        // Try to match known patterns
        if let Some(result) = self.try_lower_builtin(&func, &args, span) {
            return result;
        }

        // If the function is a tensor, this might be a user-defined operation
        // For now, we don't support that
        LowerResult::NotTensor
    }

    /// Tries to lower a built-in function application.
    fn try_lower_builtin(
        &mut self,
        func: &Expr,
        args: &[&Expr],
        span: Span,
    ) -> Option<LowerResult> {
        // Extract the function name
        let func_name = match func {
            Expr::Var(var, _) => var.name,
            _ => return None,
        };

        // Match against known builtins
        if self.builtins.is_map(func_name) && args.len() == 2 {
            return Some(self.lower_map(args[0], args[1], span));
        }

        if self.builtins.is_zip_with(func_name) && args.len() == 3 {
            return Some(self.lower_zip_with(args[0], args[1], args[2], span));
        }

        if self.builtins.is_sum(func_name) && args.len() == 1 {
            return Some(self.lower_reduce(ReduceOp::Sum, args[0], span));
        }

        if self.builtins.is_product(func_name) && args.len() == 1 {
            return Some(self.lower_reduce(ReduceOp::Prod, args[0], span));
        }

        if self.builtins.is_foldl(func_name) && args.len() == 3 {
            return Some(self.lower_fold(args[0], args[1], args[2], span));
        }

        if self.builtins.is_reshape(func_name) && args.len() == 2 {
            return Some(self.lower_reshape(args[0], args[1], span));
        }

        if self.builtins.is_transpose(func_name) && args.len() == 1 {
            return Some(self.lower_transpose(args[0], span));
        }

        if self.builtins.is_slice(func_name) && args.len() == 2 {
            return Some(self.lower_slice(args[0], args[1], span));
        }

        if self.builtins.is_broadcast(func_name) && args.len() == 2 {
            return Some(self.lower_broadcast(args[0], args[1], span));
        }

        if self.builtins.is_matmul(func_name) && args.len() == 2 {
            return Some(self.lower_matmul(args[0], args[1], span));
        }

        if self.builtins.is_dot(func_name) && args.len() == 2 {
            return Some(self.lower_dot(args[0], args[1], span));
        }

        None
    }

    /// Lowers `map f xs`.
    fn lower_map(&mut self, f: &Expr, xs: &Expr, span: Span) -> LowerResult {
        let xs_result = self.lower_expr(xs);
        let xs_tensor = match xs_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        // Extract the function symbol
        let map_fn = MapFn {
            name: extract_fn_name(f),
            span,
        };

        // Create output tensor
        let output = self.make_output_tensor(&xs_tensor.meta);

        // Create the Map operation
        let op = TensorOp::Map(map_fn, xs_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers `zipWith f xs ys`.
    fn lower_zip_with(&mut self, f: &Expr, xs: &Expr, ys: &Expr, span: Span) -> LowerResult {
        let xs_result = self.lower_expr(xs);
        let ys_result = self.lower_expr(ys);

        let xs_tensor = match xs_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        let ys_tensor = match ys_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        // Extract the function symbol
        let zip_fn = ZipFn {
            name: extract_fn_name(f),
            span,
        };

        // Create output tensor (broadcasting would happen here in a full impl)
        let output = self.make_output_tensor(&xs_tensor.meta);

        // Create the ZipWith operation
        let op = TensorOp::ZipWith(zip_fn, xs_tensor, ys_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers a reduction operation.
    fn lower_reduce(&mut self, reduce_op: ReduceOp, xs: &Expr, _span: Span) -> LowerResult {
        let xs_result = self.lower_expr(xs);
        let xs_tensor = match xs_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        // For a full reduction (sum/product), the result is a scalar tensor
        let output_meta = TensorMeta {
            dtype: xs_tensor.meta.dtype,
            shape: Shape::scalar(),
            strides: Strides::new([]),
            layout: Layout::Contiguous,
            alias: None,
        };

        let output = TensorRef {
            id: self.fresh_tensor_id(),
            meta: output_meta,
        };

        // Create the ReduceAll operation for full reduction
        let op = TensorOp::ReduceAll(reduce_op, xs_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers `foldl' f z xs`.
    fn lower_fold(&mut self, f: &Expr, z: &Expr, xs: &Expr, span: Span) -> LowerResult {
        let xs_result = self.lower_expr(xs);
        let z_result = self.lower_expr(z);

        let xs_tensor = match xs_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        let z_tensor = match z_result {
            LowerResult::Tensor(t) => t,
            LowerResult::Scalar(_) => {
                // For scalar initial values, create a scalar tensor
                self.make_scalar_tensor()
            }
            _ => return LowerResult::NotTensor,
        };

        // Extract fold function
        let fold_fn = FoldFn {
            name: extract_fn_name(f),
            span,
        };

        // Result is same shape as initial value
        let output = TensorRef {
            id: self.fresh_tensor_id(),
            meta: z_tensor.meta.clone(),
        };

        // Create the Fold operation
        let op = TensorOp::Fold(fold_fn, z_tensor, xs_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers `reshape shape xs`.
    fn lower_reshape(&mut self, shape_expr: &Expr, xs: &Expr, _span: Span) -> LowerResult {
        let xs_result = self.lower_expr(xs);
        let xs_tensor = match xs_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        // Extract shape from expression
        let new_shape = extract_shape(shape_expr).unwrap_or_else(|| xs_tensor.meta.shape.clone());

        // Create output with new shape
        let strides = Strides::contiguous(&new_shape, xs_tensor.meta.dtype.size_bytes())
            .unwrap_or_else(|| Strides::new([]));

        let output_meta = TensorMeta {
            dtype: xs_tensor.meta.dtype,
            shape: new_shape.clone(),
            strides,
            layout: Layout::Contiguous,
            alias: xs_tensor.meta.alias,
        };

        let output = TensorRef {
            id: self.fresh_tensor_id(),
            meta: output_meta,
        };

        // Create the Reshape operation
        let op = TensorOp::Reshape(new_shape, xs_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers `transpose xs`.
    fn lower_transpose(&mut self, xs: &Expr, _span: Span) -> LowerResult {
        let xs_result = self.lower_expr(xs);
        let xs_tensor = match xs_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        // Default transpose reverses dimensions
        let rank = xs_tensor.meta.shape.rank();
        let perm: SmallVec<[usize; 4]> = (0..rank).rev().collect();
        let permutation = Permutation::new(perm.clone());

        // Apply permutation to shape and strides
        let new_shape = apply_permutation_to_shape(&xs_tensor.meta.shape, &perm);
        let new_strides = apply_permutation_to_strides(&xs_tensor.meta.strides, &perm);

        let output_meta = TensorMeta {
            dtype: xs_tensor.meta.dtype,
            shape: new_shape,
            strides: new_strides,
            layout: Layout::Strided,
            alias: xs_tensor.meta.alias,
        };

        let output = TensorRef {
            id: self.fresh_tensor_id(),
            meta: output_meta,
        };

        // Create the Transpose operation
        let op = TensorOp::Transpose(permutation, xs_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers `slice spec xs`.
    fn lower_slice(&mut self, spec_expr: &Expr, xs: &Expr, _span: Span) -> LowerResult {
        let xs_result = self.lower_expr(xs);
        let xs_tensor = match xs_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        // Extract slice specification (identity if not recognized)
        let slice_spec = extract_slice_spec(spec_expr)
            .unwrap_or_else(|| make_identity_slice(xs_tensor.meta.shape.rank()));

        // Compute output shape from slice (for identity slice, same as input)
        let new_shape = compute_slice_output_shape(&slice_spec, &xs_tensor.meta.shape);

        let output_meta = TensorMeta {
            dtype: xs_tensor.meta.dtype,
            shape: new_shape,
            strides: xs_tensor.meta.strides.clone(), // Strides unchanged for basic slice
            layout: Layout::Strided,
            alias: xs_tensor.meta.alias,
        };

        let output = TensorRef {
            id: self.fresh_tensor_id(),
            meta: output_meta,
        };

        // Create the Slice operation
        let op = TensorOp::Slice(slice_spec, xs_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers `broadcast shape xs`.
    fn lower_broadcast(&mut self, shape_expr: &Expr, xs: &Expr, _span: Span) -> LowerResult {
        let xs_result = self.lower_expr(xs);
        let xs_tensor = match xs_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        // Extract target shape
        let target_shape =
            extract_shape(shape_expr).unwrap_or_else(|| xs_tensor.meta.shape.clone());

        // Compute broadcast strides (zeros where dimensions are expanded)
        let broadcast_strides = compute_broadcast_strides(&xs_tensor.meta, &target_shape);

        let output_meta = TensorMeta {
            dtype: xs_tensor.meta.dtype,
            shape: target_shape.clone(),
            strides: broadcast_strides,
            layout: Layout::Strided,
            alias: xs_tensor.meta.alias,
        };

        let output = TensorRef {
            id: self.fresh_tensor_id(),
            meta: output_meta,
        };

        // Create the Broadcast operation
        let op = TensorOp::Broadcast(target_shape, xs_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers `matmul a b`.
    fn lower_matmul(&mut self, a: &Expr, b: &Expr, _span: Span) -> LowerResult {
        let a_result = self.lower_expr(a);
        let b_result = self.lower_expr(b);

        let a_tensor = match a_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        let b_tensor = match b_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        // Compute output shape: [M, K] @ [K, N] = [M, N]
        let a_dims = a_tensor.meta.shape.dims();
        let b_dims = b_tensor.meta.shape.dims();

        let (m, n) = if a_dims.len() >= 2 && b_dims.len() >= 2 {
            let m = a_dims[a_dims.len() - 2];
            let n = b_dims[b_dims.len() - 1];
            (m, n)
        } else {
            // Handle vector-matrix multiplication cases
            return LowerResult::NotTensor;
        };

        let output_shape = Shape::new([m, n]);
        let output_strides = Strides::contiguous(&output_shape, a_tensor.meta.dtype.size_bytes())
            .unwrap_or_else(|| Strides::new([]));

        let output_meta = TensorMeta {
            dtype: a_tensor.meta.dtype,
            shape: output_shape,
            strides: output_strides,
            layout: Layout::Contiguous,
            alias: None,
        };

        let output = TensorRef {
            id: self.fresh_tensor_id(),
            meta: output_meta,
        };

        // Create the MatMul operation
        let op = TensorOp::MatMul(a_tensor, b_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers `dot a b`.
    fn lower_dot(&mut self, a: &Expr, b: &Expr, _span: Span) -> LowerResult {
        let a_result = self.lower_expr(a);
        let b_result = self.lower_expr(b);

        let a_tensor = match a_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        let b_tensor = match b_result {
            LowerResult::Tensor(t) => t,
            _ => return LowerResult::NotTensor,
        };

        // Dot product produces a scalar
        let output_meta = TensorMeta {
            dtype: a_tensor.meta.dtype,
            shape: Shape::scalar(),
            strides: Strides::new([]),
            layout: Layout::Contiguous,
            alias: None,
        };

        let output = TensorRef {
            id: self.fresh_tensor_id(),
            meta: output_meta,
        };

        // Create the Dot operation
        let op = TensorOp::Dot(a_tensor, b_tensor);
        self.ops.push(op);

        LowerResult::Tensor(output)
    }

    /// Lowers a let binding.
    fn lower_let(&mut self, bind: &bhc_core::Bind, body: &Expr) -> LowerResult {
        // Process the binding
        match bind {
            bhc_core::Bind::NonRec(var, rhs) => {
                let rhs_result = self.lower_expr(rhs);
                if let LowerResult::Tensor(tensor) = rhs_result {
                    self.register_tensor(var.id, tensor);
                }
            }
            bhc_core::Bind::Rec(bindings) => {
                // Recursive bindings are more complex; for now, skip
                for (var, rhs) in bindings {
                    let rhs_result = self.lower_expr(rhs);
                    if let LowerResult::Tensor(tensor) = rhs_result {
                        self.register_tensor(var.id, tensor);
                    }
                }
            }
        }

        // Lower the body
        self.lower_expr(body)
    }

    /// Creates an output tensor with same metadata.
    fn make_output_tensor(&mut self, input_meta: &TensorMeta) -> TensorRef {
        TensorRef {
            id: self.fresh_tensor_id(),
            meta: input_meta.clone(),
        }
    }

    /// Creates a scalar tensor.
    fn make_scalar_tensor(&mut self) -> TensorRef {
        TensorRef {
            id: self.fresh_tensor_id(),
            meta: TensorMeta {
                dtype: DType::Float64,
                shape: Shape::scalar(),
                strides: Strides::new([]),
                layout: Layout::Contiguous,
                alias: None,
            },
        }
    }
}

/// Applies a permutation to shape dimensions.
fn apply_permutation_to_shape(shape: &Shape, perm: &[usize]) -> Shape {
    let dims = shape.dims();
    let new_dims: SmallVec<[Dim; 4]> = perm.iter().map(|&i| dims[i]).collect();
    Shape::new(new_dims)
}

/// Applies a permutation to strides.
fn apply_permutation_to_strides(strides: &Strides, perm: &[usize]) -> Strides {
    let vals = strides.values();
    let new_vals: SmallVec<[i64; 4]> = perm.iter().map(|&i| vals[i]).collect();
    Strides::new(new_vals)
}

/// Creates an identity slice specification for a given rank.
fn make_identity_slice(rank: usize) -> SliceSpec {
    let ranges: SmallVec<[SliceRange; 4]> = (0..rank)
        .map(|_| SliceRange {
            start: None,
            stop: None,
            step: 1,
        })
        .collect();
    SliceSpec { ranges }
}

/// Computes output shape after applying a slice.
fn compute_slice_output_shape(slice: &SliceSpec, input_shape: &Shape) -> Shape {
    let dims = input_shape.dims();
    let mut new_dims: SmallVec<[Dim; 4]> = SmallVec::new();

    for (i, range) in slice.ranges.iter().enumerate() {
        if i >= dims.len() {
            break;
        }

        let dim = &dims[i];
        let new_dim = match (dim, range.start, range.stop, range.step) {
            // If no slice bounds specified (step=1), dimension is unchanged
            (d, None, None, 1) => *d,
            // For explicit slicing, compute new dimension
            (Dim::Static(n), start, stop, step) => {
                let s = start.unwrap_or(0) as usize;
                let e = stop.map(|x| x as usize).unwrap_or(*n);
                let st = step.unsigned_abs() as usize;
                let st = if st == 0 { 1 } else { st };
                let new_size = (e.saturating_sub(s) + st - 1) / st;
                Dim::Static(new_size)
            }
            // Dynamic dimensions with slicing become dynamic
            (Dim::Dynamic(sym), _, _, _) => Dim::Dynamic(*sym),
        };
        new_dims.push(new_dim);
    }

    // If slice has fewer ranges than input dimensions, keep remaining dims
    for dim in dims.iter().skip(slice.ranges.len()) {
        new_dims.push(*dim);
    }

    Shape::new(new_dims)
}

impl Default for LowerContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Collects all arguments from a curried application.
fn collect_app_args<'a>(f: &'a Expr, arg: &'a Expr) -> (&'a Expr, Vec<&'a Expr>) {
    let mut args = vec![arg];
    let mut current = f;

    while let Expr::App(inner_f, inner_arg, _) = current {
        args.push(inner_arg.as_ref());
        current = inner_f.as_ref();
    }

    args.reverse();
    (current, args)
}

/// Extracts a function name from an expression.
fn extract_fn_name(expr: &Expr) -> Symbol {
    match expr {
        Expr::Var(var, _) => var.name,
        Expr::Lam(_, _, _) => {
            // Anonymous lambda - would need to inline or generate name
            // For now, use a placeholder
            // SAFETY: 0 is a valid placeholder index
            unsafe { Symbol::from_raw(0) }
        }
        // SAFETY: 0 is a valid placeholder index
        _ => unsafe { Symbol::from_raw(0) },
    }
}

/// Extracts a shape from an expression.
fn extract_shape(_expr: &Expr) -> Option<Shape> {
    // In a full implementation, this would parse the shape from
    // a list literal or other representation
    None
}

/// Extracts a slice specification from an expression.
fn extract_slice_spec(_expr: &Expr) -> Option<SliceSpec> {
    // In a full implementation, this would parse the slice from
    // a data structure
    None
}

/// Computes broadcast strides.
fn compute_broadcast_strides(source: &TensorMeta, target_shape: &Shape) -> Strides {
    let source_rank = source.shape.rank();
    let target_rank = target_shape.rank();

    let mut strides: SmallVec<[i64; 4]> = SmallVec::new();

    for i in 0..target_rank {
        let source_idx = source_rank as isize - (target_rank as isize - i as isize);
        if source_idx < 0 {
            // Dimension was added, stride is 0
            strides.push(0);
        } else {
            let src_idx = source_idx as usize;
            let src_dim = source.shape.dims()[src_idx];
            let tgt_dim = target_shape.dims()[i];
            if src_dim == tgt_dim {
                strides.push(source.strides.values()[src_idx]);
            } else if src_dim == Dim::Static(1) {
                // Broadcasting, stride is 0
                strides.push(0);
            } else {
                // Incompatible, use original stride
                strides.push(source.strides.values()[src_idx]);
            }
        }
    }

    Strides::new(strides)
}

/// Lowers a Core module to Tensor IR operations.
///
/// This is the main entry point for lowering.
pub fn lower_module(module: &bhc_core::CoreModule) -> Vec<TensorOp> {
    let mut ctx = LowerContext::new();

    for bind in &module.bindings {
        match bind {
            bhc_core::Bind::NonRec(var, rhs) => {
                let result = ctx.lower_expr(rhs);
                if let LowerResult::Tensor(tensor) = result {
                    ctx.register_tensor(var.id, tensor);
                }
            }
            bhc_core::Bind::Rec(bindings) => {
                for (var, rhs) in bindings {
                    let result = ctx.lower_expr(rhs);
                    if let LowerResult::Tensor(tensor) = result {
                        ctx.register_tensor(var.id, tensor);
                    }
                }
            }
        }
    }

    ctx.into_ops()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_core::VarId;

    fn make_tensor_ref(id: u32, shape: &[usize], dtype: DType) -> TensorRef {
        let dims: SmallVec<[Dim; 4]> = shape.iter().map(|&d| Dim::Static(d)).collect();
        let shape = Shape::new(dims);
        let strides =
            Strides::contiguous(&shape, dtype.size_bytes()).unwrap_or_else(|| Strides::new([]));
        TensorRef {
            id: TensorId::new(id as usize),
            meta: TensorMeta {
                dtype,
                shape,
                strides,
                layout: Layout::Contiguous,
                alias: None,
            },
        }
    }

    #[test]
    fn test_lower_context_creation() {
        let ctx = LowerContext::new();
        assert!(ctx.ops().is_empty());
    }

    #[test]
    fn test_register_and_lookup_tensor() {
        let mut ctx = LowerContext::new();
        let tensor = make_tensor_ref(0, &[100], DType::Float32);
        let var_id = VarId::new(42);

        ctx.register_tensor(var_id, tensor.clone());

        let found = ctx.lookup_tensor(var_id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, tensor.id);
    }

    #[test]
    fn test_fresh_ids_increment() {
        let mut ctx = LowerContext::new();

        let id1 = ctx.fresh_tensor_id();
        let id2 = ctx.fresh_tensor_id();

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_collect_app_args() {
        // Build: f x y z
        // SAFETY: Valid symbol index for testing
        let f_var = bhc_core::Var::new(unsafe { Symbol::from_raw(1) }, VarId::new(0), Ty::Error);
        let f_expr = Expr::Var(f_var, Span::default());

        let x = Expr::Lit(Literal::Int(1), Ty::Error, Span::default());
        let y = Expr::Lit(Literal::Int(2), Ty::Error, Span::default());
        let z = Expr::Lit(Literal::Int(3), Ty::Error, Span::default());

        // (f x)
        let app1 = Expr::App(Box::new(f_expr), Box::new(x), Span::default());
        // ((f x) y)
        let app2 = Expr::App(Box::new(app1), Box::new(y), Span::default());
        // (((f x) y) z)
        let app3 = Expr::App(Box::new(app2), Box::new(z), Span::default());

        if let Expr::App(f, arg, _) = &app3 {
            let (func, args) = collect_app_args(f, arg);
            assert_eq!(args.len(), 3);
            // func should be the original Var
            assert!(matches!(func, Expr::Var(_, _)));
        }
    }

    #[test]
    fn test_lower_literal_int() {
        let ctx = LowerContext::new();
        let lit = Literal::Int(42);
        let result = ctx.lower_lit(&lit, &Ty::Error);

        match result {
            LowerResult::Scalar(ScalarValue::Int(n)) => assert_eq!(n, 42),
            _ => panic!("Expected scalar int"),
        }
    }

    #[test]
    fn test_lower_literal_float() {
        let ctx = LowerContext::new();
        let lit = Literal::Double(3.14);
        let result = ctx.lower_lit(&lit, &Ty::Error);

        match result {
            LowerResult::Scalar(ScalarValue::Float(f)) => {
                assert!((f - 3.14).abs() < f64::EPSILON);
            }
            _ => panic!("Expected scalar float"),
        }
    }

    #[test]
    fn test_broadcast_strides_expansion() {
        let source_meta = TensorMeta {
            dtype: DType::Float32,
            shape: Shape::new([Dim::Static(1), Dim::Static(10)]),
            strides: Strides::new([10, 1]),
            layout: Layout::Contiguous,
            alias: None,
        };

        let target_shape = Shape::new([Dim::Static(5), Dim::Static(1), Dim::Static(10)]);

        let broadcast = compute_broadcast_strides(&source_meta, &target_shape);

        // First dim (5) was added - stride should be 0
        assert_eq!(broadcast.values()[0], 0);
        // Second dim (1->1) - dims match, stride preserved (source stride 10)
        assert_eq!(broadcast.values()[1], 10);
        // Third dim (10) unchanged
        assert_eq!(broadcast.values()[2], 1);
    }

    #[test]
    fn test_broadcast_strides_broadcasting() {
        // Source [1, 5] broadcast to [3, 5]
        let source_meta = TensorMeta {
            dtype: DType::Float32,
            shape: Shape::new([Dim::Static(1), Dim::Static(5)]),
            strides: Strides::new([5, 1]),
            layout: Layout::Contiguous,
            alias: None,
        };

        let target_shape = Shape::new([Dim::Static(3), Dim::Static(5)]);

        let broadcast = compute_broadcast_strides(&source_meta, &target_shape);

        // First dim (1->3) - broadcasting, stride should be 0
        assert_eq!(broadcast.values()[0], 0);
        // Second dim (5->5) unchanged
        assert_eq!(broadcast.values()[1], 1);
    }

    // === View Operation Tests ===

    #[test]
    fn test_permutation_to_shape() {
        // 2D: [3, 4] with perm [1, 0] -> [4, 3]
        let shape = Shape::new([Dim::Static(3), Dim::Static(4)]);
        let perm = [1, 0];

        let new_shape = apply_permutation_to_shape(&shape, &perm);

        assert_eq!(new_shape.dims().len(), 2);
        assert_eq!(new_shape.dims()[0], Dim::Static(4));
        assert_eq!(new_shape.dims()[1], Dim::Static(3));
    }

    #[test]
    fn test_permutation_to_strides() {
        // 2D strides: [4, 1] with perm [1, 0] -> [1, 4]
        let strides = Strides::new([4, 1]);
        let perm = [1, 0];

        let new_strides = apply_permutation_to_strides(&strides, &perm);

        assert_eq!(new_strides.values().len(), 2);
        assert_eq!(new_strides.values()[0], 1);
        assert_eq!(new_strides.values()[1], 4);
    }

    #[test]
    fn test_transpose_3d() {
        // 3D: [2, 3, 4] with reverse perm [2, 1, 0] -> [4, 3, 2]
        let shape = Shape::new([Dim::Static(2), Dim::Static(3), Dim::Static(4)]);
        let strides = Strides::new([12, 4, 1]);
        let perm: SmallVec<[usize; 4]> = [2, 1, 0].into_iter().collect();

        let new_shape = apply_permutation_to_shape(&shape, &perm);
        let new_strides = apply_permutation_to_strides(&strides, &perm);

        assert_eq!(new_shape.dims()[0], Dim::Static(4));
        assert_eq!(new_shape.dims()[1], Dim::Static(3));
        assert_eq!(new_shape.dims()[2], Dim::Static(2));

        assert_eq!(new_strides.values()[0], 1);
        assert_eq!(new_strides.values()[1], 4);
        assert_eq!(new_strides.values()[2], 12);
    }

    #[test]
    fn test_identity_slice() {
        let slice = make_identity_slice(3);

        assert_eq!(slice.ranges.len(), 3);
        for r in slice.ranges.iter() {
            assert_eq!(r.start, None);
            assert_eq!(r.stop, None);
            assert_eq!(r.step, 1);
        }
    }

    #[test]
    fn test_slice_output_shape_identity() {
        let shape = Shape::new([Dim::Static(10), Dim::Static(20)]);
        let slice = make_identity_slice(2);

        let output = compute_slice_output_shape(&slice, &shape);

        // Identity slice preserves shape
        assert_eq!(output.dims()[0], Dim::Static(10));
        assert_eq!(output.dims()[1], Dim::Static(20));
    }

    #[test]
    fn test_slice_output_shape_with_bounds() {
        let shape = Shape::new([Dim::Static(10), Dim::Static(20)]);

        // Slice [2:8, 5:15] -> [6, 10]
        let slice = SliceSpec {
            ranges: smallvec::smallvec![
                SliceRange {
                    start: Some(2),
                    stop: Some(8),
                    step: 1
                },
                SliceRange {
                    start: Some(5),
                    stop: Some(15),
                    step: 1
                },
            ],
        };

        let output = compute_slice_output_shape(&slice, &shape);

        assert_eq!(output.dims()[0], Dim::Static(6));
        assert_eq!(output.dims()[1], Dim::Static(10));
    }

    #[test]
    fn test_slice_output_shape_with_step() {
        let shape = Shape::new([Dim::Static(10)]);

        // Slice [0:10:2] -> 5 elements (0, 2, 4, 6, 8)
        let slice = SliceSpec {
            ranges: smallvec::smallvec![SliceRange {
                start: Some(0),
                stop: Some(10),
                step: 2
            },],
        };

        let output = compute_slice_output_shape(&slice, &shape);

        assert_eq!(output.dims()[0], Dim::Static(5));
    }

    #[test]
    fn test_reshape_contiguous_is_metadata_only() {
        // This tests the M2 exit criteria: reshape on contiguous tensor is metadata-only
        // A contiguous tensor can be reshaped by just changing shape/strides metadata
        // without copying data

        use crate::fusion::is_reshape_metadata_only;

        // Contiguous tensor
        let tensor = make_tensor_ref(0, &[2, 3, 4], DType::Float32);
        assert!(is_reshape_metadata_only(&tensor));

        // Non-contiguous tensor (strided view)
        let strided_meta = TensorMeta {
            dtype: DType::Float32,
            shape: Shape::new([Dim::Static(3), Dim::Static(4)]),
            strides: Strides::new([8, 1]), // Non-standard strides
            layout: Layout::Strided,
            alias: None,
        };
        let strided = TensorRef {
            id: TensorId::new(1),
            meta: strided_meta,
        };
        assert!(!is_reshape_metadata_only(&strided));
    }
}
