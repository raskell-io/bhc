//! Lowering from Loop IR to LLVM IR.
//!
//! This module implements the translation from BHC's Loop IR to LLVM IR
//! for the Numeric profile. It handles:
//!
//! - Loop constructs: for loops, parallel loops
//! - SIMD operations: vector loads, stores, arithmetic
//! - Memory operations: aligned/unaligned access
//! - Scalar operations: arithmetic, comparisons, casts

use crate::{CodegenError, CodegenResult};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_loop_ir::{
    AccessPattern, Alloc, AllocSize, BinOp, Body, CmpOp, IfStmt, Loop, LoopAttrs, LoopIR, LoopType,
    MemRef, Op, Param, ReduceOp, ScalarType, Stmt, UnOp, Value, ValueId,
};
use bhc_tensor_ir::{AllocRegion, BufferId};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, VectorType};
use inkwell::values::{
    BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue, PointerValue, VectorValue,
};
use inkwell::AddressSpace;
use inkwell::FloatPredicate;
use inkwell::IntPredicate;
use rustc_hash::FxHashMap;

use super::context::LlvmContext;
use super::module::LlvmModule;

/// State for lowering Loop IR to LLVM IR.
pub struct LoopLowering<'ctx, 'm> {
    /// The underlying LLVM context.
    llvm_ctx: &'ctx Context,
    /// The LLVM module being generated.
    module: &'m LlvmModule<'ctx>,
    /// The IR builder.
    builder: Builder<'ctx>,
    /// Mapping from Loop IR values to LLVM values.
    values: FxHashMap<ValueId, BasicValueEnum<'ctx>>,
    /// Mapping from buffer IDs to LLVM pointers.
    buffers: FxHashMap<BufferId, PointerValue<'ctx>>,
    /// Counter for generating unique names.
    name_counter: u32,
    /// Current function being generated.
    current_fn: Option<FunctionValue<'ctx>>,
}

impl<'ctx, 'm> LoopLowering<'ctx, 'm> {
    /// Create a new Loop IR lowering context.
    pub fn new(ctx: &'ctx LlvmContext, module: &'m LlvmModule<'ctx>) -> Self {
        Self {
            llvm_ctx: ctx.llvm_context(),
            module,
            builder: ctx.llvm_context().create_builder(),
            values: FxHashMap::default(),
            buffers: FxHashMap::default(),
            name_counter: 0,
            current_fn: None,
        }
    }

    /// Generate a unique name.
    fn unique_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.name_counter);
        self.name_counter += 1;
        name
    }

    /// Lower a complete Loop IR function to LLVM.
    pub fn lower_function(&mut self, ir: &LoopIR) -> CodegenResult<FunctionValue<'ctx>> {
        // Clear state from previous lowering
        self.values.clear();
        self.buffers.clear();

        // Create function signature
        let fn_type = self.create_function_type(ir)?;
        let fn_name = ir.name.as_str();
        let function = self
            .module
            .llvm_module()
            .add_function(fn_name, fn_type, None);
        self.current_fn = Some(function);

        // Create entry block
        let entry = self.llvm_ctx.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // Bind parameters
        for (i, param) in ir.params.iter().enumerate() {
            let llvm_param = function
                .get_nth_param(i as u32)
                .ok_or_else(|| CodegenError::Internal(format!("Missing parameter {}", i)))?;
            // Store the parameter value
            let param_id = ValueId::new(i);
            self.values.insert(param_id, llvm_param);
        }

        // Allocate buffers
        for alloc in &ir.allocs {
            self.lower_alloc(alloc)?;
        }

        // Lower the body
        self.lower_body(&ir.body)?;

        // Add implicit void return if needed
        if ir.return_ty.is_void() {
            if self
                .builder
                .get_insert_block()
                .map_or(true, |b| b.get_terminator().is_none())
            {
                self.builder
                    .build_return(None)
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
            }
        }

        // Verify the function
        if function.verify(true) {
            Ok(function)
        } else {
            Err(CodegenError::Internal(format!(
                "Generated function {} failed verification",
                fn_name
            )))
        }
    }

    /// Create the LLVM function type for a Loop IR function.
    fn create_function_type(
        &self,
        ir: &LoopIR,
    ) -> CodegenResult<inkwell::types::FunctionType<'ctx>> {
        let param_types: Vec<BasicMetadataTypeEnum> = ir
            .params
            .iter()
            .map(|p| self.loop_type_to_llvm(&p.ty))
            .collect::<CodegenResult<Vec<_>>>()?
            .into_iter()
            .map(|t| t.into())
            .collect();

        let fn_type = if ir.return_ty.is_void() {
            self.llvm_ctx.void_type().fn_type(&param_types, false)
        } else {
            let ret_ty = self.loop_type_to_llvm(&ir.return_ty)?;
            ret_ty.fn_type(&param_types, false)
        };

        Ok(fn_type)
    }

    /// Convert a Loop IR type to an LLVM type.
    fn loop_type_to_llvm(&self, ty: &LoopType) -> CodegenResult<BasicTypeEnum<'ctx>> {
        match ty {
            LoopType::Void => Err(CodegenError::Internal(
                "Cannot convert void to basic type".to_string(),
            )),
            LoopType::Scalar(scalar) => self.scalar_type_to_llvm(*scalar),
            LoopType::Vector(scalar, width) => {
                let elem_ty = self.scalar_type_to_llvm(*scalar)?;
                let vec_ty = match elem_ty {
                    BasicTypeEnum::FloatType(f) => f.vec_type(*width as u32).into(),
                    BasicTypeEnum::IntType(i) => i.vec_type(*width as u32).into(),
                    _ => {
                        return Err(CodegenError::Internal(format!(
                            "Cannot create vector of {:?}",
                            elem_ty
                        )))
                    }
                };
                Ok(vec_ty)
            }
            LoopType::Ptr(inner) => {
                let ptr_ty = self.llvm_ctx.ptr_type(AddressSpace::default());
                Ok(ptr_ty.into())
            }
        }
    }

    /// Convert a scalar type to an LLVM type.
    fn scalar_type_to_llvm(&self, ty: ScalarType) -> CodegenResult<BasicTypeEnum<'ctx>> {
        let llvm_ty = match ty {
            ScalarType::Bool => self.llvm_ctx.bool_type().into(),
            ScalarType::Int(8) => self.llvm_ctx.i8_type().into(),
            ScalarType::Int(16) => self.llvm_ctx.i16_type().into(),
            ScalarType::Int(32) => self.llvm_ctx.i32_type().into(),
            ScalarType::Int(64) => self.llvm_ctx.i64_type().into(),
            ScalarType::UInt(8) => self.llvm_ctx.i8_type().into(),
            ScalarType::UInt(16) => self.llvm_ctx.i16_type().into(),
            ScalarType::UInt(32) => self.llvm_ctx.i32_type().into(),
            ScalarType::UInt(64) => self.llvm_ctx.i64_type().into(),
            ScalarType::Float(16) => self.llvm_ctx.f16_type().into(),
            ScalarType::Float(32) => self.llvm_ctx.f32_type().into(),
            ScalarType::Float(64) => self.llvm_ctx.f64_type().into(),
            ScalarType::Int(bits) | ScalarType::UInt(bits) => {
                self.llvm_ctx.custom_width_int_type(bits as u32).into()
            }
            ScalarType::Float(bits) => {
                return Err(CodegenError::Unsupported(format!(
                    "Float with {} bits",
                    bits
                )))
            }
        };
        Ok(llvm_ty)
    }

    /// Lower an allocation.
    fn lower_alloc(&mut self, alloc: &Alloc) -> CodegenResult<()> {
        let elem_ty = self.scalar_type_to_llvm(alloc.elem_ty)?;

        // Compute the size
        let size = match &alloc.size {
            AllocSize::Static(n) => self.llvm_ctx.i64_type().const_int(*n as u64, false),
            AllocSize::Dynamic(vid) => {
                let val = self.get_value(*vid)?;
                match val {
                    BasicValueEnum::IntValue(i) => i,
                    _ => {
                        return Err(CodegenError::TypeError(
                            "Dynamic size must be integer".to_string(),
                        ))
                    }
                }
            }
        };

        // For now, use alloca for stack allocation (Hot Arena)
        // In the future, this would dispatch to arena allocator
        let ptr = match alloc.region {
            AllocRegion::HotArena | AllocRegion::General => {
                // Stack allocate with alignment
                let array_ty = elem_ty.array_type(1024); // Placeholder size
                let alloca = self
                    .builder
                    .build_alloca(array_ty, alloc.name.as_str())
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                alloca
            }
            AllocRegion::Pinned | AllocRegion::DeviceMemory(_) => {
                // These would need malloc/device allocation
                // For now, fall back to stack
                let array_ty = elem_ty.array_type(1024);
                self.builder
                    .build_alloca(array_ty, alloc.name.as_str())
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
            }
        };

        self.buffers.insert(alloc.buffer, ptr);
        Ok(())
    }

    /// Lower a body (sequence of statements).
    fn lower_body(&mut self, body: &Body) -> CodegenResult<()> {
        for stmt in &body.stmts {
            self.lower_stmt(stmt)?;
        }
        Ok(())
    }

    /// Lower a statement.
    fn lower_stmt(&mut self, stmt: &Stmt) -> CodegenResult<()> {
        match stmt {
            Stmt::Assign(vid, op) => {
                let value = self.lower_op(op)?;
                self.values.insert(*vid, value);
            }
            Stmt::Loop(lp) => {
                self.lower_loop(lp)?;
            }
            Stmt::If(if_stmt) => {
                self.lower_if(if_stmt)?;
            }
            Stmt::Store(mem_ref, value) => {
                self.lower_store(mem_ref, value)?;
            }
            Stmt::Call(result, name, args) => {
                self.lower_call(*result, *name, args)?;
            }
            Stmt::Return(value) => {
                self.lower_return(value.as_ref())?;
            }
            Stmt::Barrier(_kind) => {
                // Insert memory barrier
                self.builder
                    .build_fence(
                        inkwell::AtomicOrdering::SequentiallyConsistent,
                        0, // syncscope ID (0 = system scope)
                        "fence",
                    )
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
            }
            Stmt::Comment(_) => {
                // Comments are no-ops in LLVM
            }
        }
        Ok(())
    }

    /// Lower a loop construct.
    fn lower_loop(&mut self, lp: &Loop) -> CodegenResult<()> {
        let function = self
            .current_fn
            .ok_or_else(|| CodegenError::Internal("No current function for loop".to_string()))?;

        // Create basic blocks
        let preheader = self.llvm_ctx.append_basic_block(function, "loop.preheader");
        let header = self.llvm_ctx.append_basic_block(function, "loop.header");
        let body_block = self.llvm_ctx.append_basic_block(function, "loop.body");
        let latch = self.llvm_ctx.append_basic_block(function, "loop.latch");
        let exit = self.llvm_ctx.append_basic_block(function, "loop.exit");

        // Branch to preheader
        self.builder
            .build_unconditional_branch(preheader)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Preheader: initialize loop variable
        self.builder.position_at_end(preheader);
        let lower_val = self.lower_value(&lp.lower)?;
        let lower_int = self.value_to_int(lower_val)?;
        self.builder
            .build_unconditional_branch(header)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Header: phi node and condition check
        self.builder.position_at_end(header);
        let i64_ty = self.llvm_ctx.i64_type();
        let phi = self
            .builder
            .build_phi(i64_ty, "loop.iv")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        phi.add_incoming(&[(&lower_int, preheader)]);

        let iv_val = phi.as_basic_value().into_int_value();
        self.values.insert(lp.var, BasicValueEnum::IntValue(iv_val));

        let upper_val = self.lower_value(&lp.upper)?;
        let upper_int = self.value_to_int(upper_val)?;

        let cond = self
            .builder
            .build_int_compare(IntPredicate::SLT, iv_val, upper_int, "loop.cond")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        self.builder
            .build_conditional_branch(cond, body_block, exit)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Body: lower loop body statements
        self.builder.position_at_end(body_block);
        self.lower_body(&lp.body)?;
        self.builder
            .build_unconditional_branch(latch)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Latch: increment and back-edge
        self.builder.position_at_end(latch);
        let step_val = self.lower_value(&lp.step)?;
        let step_int = self.value_to_int(step_val)?;

        // Get the current IV value (may have been updated in the body)
        let current_iv = self
            .values
            .get(&lp.var)
            .copied()
            .ok_or_else(|| CodegenError::Internal("Loop variable not found".to_string()))?;
        let current_iv_int = match current_iv {
            BasicValueEnum::IntValue(i) => i,
            _ => iv_val, // Fall back to phi value
        };

        let next_iv = self
            .builder
            .build_int_add(current_iv_int, step_int, "loop.next")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        phi.add_incoming(&[(&next_iv, latch)]);

        self.builder
            .build_unconditional_branch(header)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Continue after exit
        self.builder.position_at_end(exit);

        Ok(())
    }

    /// Lower an if statement.
    fn lower_if(&mut self, if_stmt: &IfStmt) -> CodegenResult<()> {
        let function = self
            .current_fn
            .ok_or_else(|| CodegenError::Internal("No current function for if".to_string()))?;

        let cond_val = self.lower_value(&if_stmt.cond)?;
        let cond_bool = match cond_val {
            BasicValueEnum::IntValue(i) => {
                // Convert to i1 if needed
                if i.get_type().get_bit_width() == 1 {
                    i
                } else {
                    self.builder
                        .build_int_compare(
                            IntPredicate::NE,
                            i,
                            i.get_type().const_zero(),
                            "cond.bool",
                        )
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                }
            }
            _ => {
                return Err(CodegenError::TypeError(
                    "If condition must be boolean/integer".to_string(),
                ))
            }
        };

        let then_block = self.llvm_ctx.append_basic_block(function, "if.then");
        let else_block = self.llvm_ctx.append_basic_block(function, "if.else");
        let merge_block = self.llvm_ctx.append_basic_block(function, "if.merge");

        self.builder
            .build_conditional_branch(cond_bool, then_block, else_block)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        // Then branch
        self.builder.position_at_end(then_block);
        self.lower_body(&if_stmt.then_body)?;
        if self
            .builder
            .get_insert_block()
            .map_or(true, |b| b.get_terminator().is_none())
        {
            self.builder
                .build_unconditional_branch(merge_block)
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        }

        // Else branch
        self.builder.position_at_end(else_block);
        if let Some(else_body) = &if_stmt.else_body {
            self.lower_body(else_body)?;
        }
        if self
            .builder
            .get_insert_block()
            .map_or(true, |b| b.get_terminator().is_none())
        {
            self.builder
                .build_unconditional_branch(merge_block)
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        }

        // Continue at merge
        self.builder.position_at_end(merge_block);

        Ok(())
    }

    /// Lower a store operation.
    fn lower_store(&mut self, mem_ref: &MemRef, value: &Value) -> CodegenResult<()> {
        let ptr = self.compute_address(mem_ref)?;
        let val = self.lower_value(value)?;

        self.builder
            .build_store(ptr, val)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(())
    }

    /// Lower a function call.
    fn lower_call(
        &mut self,
        result: Option<ValueId>,
        name: Symbol,
        args: &[Value],
    ) -> CodegenResult<()> {
        // Look up or declare the function
        let fn_name = name.as_str();
        let function = self
            .module
            .llvm_module()
            .get_function(fn_name)
            .ok_or_else(|| CodegenError::Internal(format!("Unknown function: {}", fn_name)))?;

        // Lower arguments
        let arg_vals: Vec<BasicValueEnum> = args
            .iter()
            .map(|a| self.lower_value(a))
            .collect::<CodegenResult<Vec<_>>>()?;

        let arg_meta: Vec<_> = arg_vals.iter().map(|v| (*v).into()).collect();

        let call_result = self
            .builder
            .build_call(function, &arg_meta, "call")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        if let Some(vid) = result {
            if let Some(ret_val) = call_result.try_as_basic_value().basic() {
                self.values.insert(vid, ret_val);
            }
        }

        Ok(())
    }

    /// Lower a return statement.
    fn lower_return(&mut self, value: Option<&Value>) -> CodegenResult<()> {
        match value {
            Some(v) => {
                let val = self.lower_value(v)?;
                self.builder
                    .build_return(Some(&val))
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
            }
            None => {
                self.builder
                    .build_return(None)
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Lower an operation.
    fn lower_op(&mut self, op: &Op) -> CodegenResult<BasicValueEnum<'ctx>> {
        match op {
            Op::Load(mem_ref) => self.lower_load(mem_ref),
            Op::Binary(bin_op, lhs, rhs) => self.lower_binary(*bin_op, lhs, rhs),
            Op::Unary(un_op, val) => self.lower_unary(*un_op, val),
            Op::Cmp(cmp_op, lhs, rhs) => self.lower_cmp(*cmp_op, lhs, rhs),
            Op::Select(cond, then_val, else_val) => self.lower_select(cond, then_val, else_val),
            Op::Cast(val, target_ty) => self.lower_cast(val, target_ty),
            Op::Broadcast(val, width) => self.lower_broadcast(val, *width),
            Op::Extract(vec, idx) => self.lower_extract(vec, *idx),
            Op::Insert(vec, val, idx) => self.lower_insert(vec, val, *idx),
            Op::Shuffle(v1, v2, mask) => self.lower_shuffle(v1, v2, mask),
            Op::VecReduce(reduce_op, val) => self.lower_vec_reduce(*reduce_op, val),
            Op::Fma(a, b, c) => self.lower_fma(a, b, c),
            Op::PtrAdd(ptr, offset) => self.lower_ptr_add(ptr, offset),
            Op::GetPtr(buffer, index) => self.lower_get_ptr(*buffer, index),
            Op::Phi(entries) => {
                // Phi nodes are handled specially during loop lowering
                Err(CodegenError::Internal(
                    "Phi nodes should be lowered in loop context".to_string(),
                ))
            }
        }
    }

    /// Lower a load operation.
    fn lower_load(&mut self, mem_ref: &MemRef) -> CodegenResult<BasicValueEnum<'ctx>> {
        let ptr = self.compute_address(mem_ref)?;
        let elem_ty = self.loop_type_to_llvm(&mem_ref.elem_ty)?;

        let load = self
            .builder
            .build_load(elem_ty, ptr, "load")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(load)
    }

    /// Compute the address for a memory reference.
    fn compute_address(&mut self, mem_ref: &MemRef) -> CodegenResult<PointerValue<'ctx>> {
        let base_ptr = self.buffers.get(&mem_ref.buffer).copied().ok_or_else(|| {
            CodegenError::Internal(format!("Unknown buffer: {:?}", mem_ref.buffer))
        })?;

        let index_val = self.lower_value(&mem_ref.index)?;
        let index_int = self.value_to_int(index_val)?;

        let elem_ty = self.loop_type_to_llvm(&mem_ref.elem_ty)?;

        // Use GEP to compute the address
        let ptr = unsafe {
            self.builder
                .build_gep(elem_ty, base_ptr, &[index_int], "gep")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?
        };

        Ok(ptr)
    }

    /// Lower a binary operation.
    fn lower_binary(
        &mut self,
        op: BinOp,
        lhs: &Value,
        rhs: &Value,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let lhs_val = self.lower_value(lhs)?;
        let rhs_val = self.lower_value(rhs)?;

        let result = match (lhs_val, rhs_val) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                self.lower_int_binary(op, l, r)?
            }
            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                self.lower_float_binary(op, l, r)?
            }
            (BasicValueEnum::VectorValue(l), BasicValueEnum::VectorValue(r)) => {
                self.lower_vector_binary(op, l, r)?
            }
            _ => {
                return Err(CodegenError::TypeError(format!(
                    "Mismatched types for binary op {:?}",
                    op
                )))
            }
        };

        Ok(result)
    }

    /// Lower an integer binary operation.
    fn lower_int_binary(
        &mut self,
        op: BinOp,
        lhs: IntValue<'ctx>,
        rhs: IntValue<'ctx>,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let result = match op {
            BinOp::Add => self
                .builder
                .build_int_add(lhs, rhs, "add")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::Sub => self
                .builder
                .build_int_sub(lhs, rhs, "sub")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::Mul => self
                .builder
                .build_int_mul(lhs, rhs, "mul")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::SDiv => self
                .builder
                .build_int_signed_div(lhs, rhs, "sdiv")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::UDiv => self
                .builder
                .build_int_unsigned_div(lhs, rhs, "udiv")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::SRem => self
                .builder
                .build_int_signed_rem(lhs, rhs, "srem")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::URem => self
                .builder
                .build_int_unsigned_rem(lhs, rhs, "urem")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::And => self
                .builder
                .build_and(lhs, rhs, "and")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::Or => self
                .builder
                .build_or(lhs, rhs, "or")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::Xor => self
                .builder
                .build_xor(lhs, rhs, "xor")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::Shl => self
                .builder
                .build_left_shift(lhs, rhs, "shl")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::LShr => self
                .builder
                .build_right_shift(lhs, rhs, false, "lshr")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::AShr => self
                .builder
                .build_right_shift(lhs, rhs, true, "ashr")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::SMin | BinOp::UMin | BinOp::SMax | BinOp::UMax => {
                // Use select for min/max
                let cmp = match op {
                    BinOp::SMin => IntPredicate::SLT,
                    BinOp::UMin => IntPredicate::ULT,
                    BinOp::SMax => IntPredicate::SGT,
                    BinOp::UMax => IntPredicate::UGT,
                    _ => unreachable!(),
                };
                let cond = self
                    .builder
                    .build_int_compare(cmp, lhs, rhs, "cmp")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                self.builder
                    .build_select(cond, lhs, rhs, "minmax")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into_int_value()
            }
            BinOp::FDiv | BinOp::FRem | BinOp::FMin | BinOp::FMax => {
                return Err(CodegenError::TypeError(
                    "Float operation on integer values".to_string(),
                ))
            }
        };
        Ok(result.into())
    }

    /// Lower a float binary operation.
    fn lower_float_binary(
        &mut self,
        op: BinOp,
        lhs: FloatValue<'ctx>,
        rhs: FloatValue<'ctx>,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let result = match op {
            BinOp::Add => self
                .builder
                .build_float_add(lhs, rhs, "fadd")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::Sub => self
                .builder
                .build_float_sub(lhs, rhs, "fsub")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::Mul => self
                .builder
                .build_float_mul(lhs, rhs, "fmul")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::FDiv => self
                .builder
                .build_float_div(lhs, rhs, "fdiv")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::FRem => self
                .builder
                .build_float_rem(lhs, rhs, "frem")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            BinOp::FMin => {
                let cond = self
                    .builder
                    .build_float_compare(FloatPredicate::OLT, lhs, rhs, "cmp")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                self.builder
                    .build_select(cond, lhs, rhs, "fmin")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into_float_value()
            }
            BinOp::FMax => {
                let cond = self
                    .builder
                    .build_float_compare(FloatPredicate::OGT, lhs, rhs, "cmp")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                self.builder
                    .build_select(cond, lhs, rhs, "fmax")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into_float_value()
            }
            _ => {
                return Err(CodegenError::TypeError(
                    "Integer operation on float values".to_string(),
                ))
            }
        };
        Ok(result.into())
    }

    /// Lower a vector binary operation.
    fn lower_vector_binary(
        &mut self,
        op: BinOp,
        lhs: VectorValue<'ctx>,
        rhs: VectorValue<'ctx>,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        // Vector operations use the same instructions as scalar
        // LLVM handles vectorization automatically
        let elem_ty = lhs.get_type().get_element_type();

        let result: BasicValueEnum<'ctx> = if elem_ty.is_int_type() {
            match op {
                BinOp::Add => self
                    .builder
                    .build_int_add(lhs, rhs, "vadd")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                BinOp::Sub => self
                    .builder
                    .build_int_sub(lhs, rhs, "vsub")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                BinOp::Mul => self
                    .builder
                    .build_int_mul(lhs, rhs, "vmul")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                _ => {
                    return Err(CodegenError::Unsupported(format!(
                        "Vector int operation {:?}",
                        op
                    )))
                }
            }
        } else {
            match op {
                BinOp::Add => self
                    .builder
                    .build_float_add(lhs, rhs, "vfadd")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                BinOp::Sub => self
                    .builder
                    .build_float_sub(lhs, rhs, "vfsub")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                BinOp::Mul => self
                    .builder
                    .build_float_mul(lhs, rhs, "vfmul")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                BinOp::FDiv => self
                    .builder
                    .build_float_div(lhs, rhs, "vfdiv")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                _ => {
                    return Err(CodegenError::Unsupported(format!(
                        "Vector float operation {:?}",
                        op
                    )))
                }
            }
        };

        Ok(result)
    }

    /// Lower a unary operation.
    fn lower_unary(&mut self, op: UnOp, val: &Value) -> CodegenResult<BasicValueEnum<'ctx>> {
        let v = self.lower_value(val)?;

        let result = match v {
            BasicValueEnum::IntValue(i) => self.lower_int_unary(op, i)?,
            BasicValueEnum::FloatValue(f) => self.lower_float_unary(op, f)?,
            BasicValueEnum::VectorValue(vec) => {
                // Handle vector unary ops
                self.lower_vector_unary(op, vec)?
            }
            _ => {
                return Err(CodegenError::TypeError(format!(
                    "Unsupported type for unary op {:?}",
                    op
                )))
            }
        };

        Ok(result)
    }

    /// Lower an integer unary operation.
    fn lower_int_unary(
        &mut self,
        op: UnOp,
        val: IntValue<'ctx>,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let result = match op {
            UnOp::Neg => self
                .builder
                .build_int_neg(val, "neg")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            UnOp::Not => self
                .builder
                .build_not(val, "not")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            UnOp::Abs => {
                // abs(x) = x < 0 ? -x : x
                let zero = val.get_type().const_zero();
                let is_neg = self
                    .builder
                    .build_int_compare(IntPredicate::SLT, val, zero, "is_neg")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                let neg_val = self
                    .builder
                    .build_int_neg(val, "neg")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
                self.builder
                    .build_select(is_neg, neg_val, val, "abs")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into_int_value()
            }
            _ => {
                return Err(CodegenError::TypeError(format!(
                    "Float unary op {:?} on integer",
                    op
                )))
            }
        };
        Ok(result.into())
    }

    /// Lower a float unary operation.
    fn lower_float_unary(
        &mut self,
        op: UnOp,
        val: FloatValue<'ctx>,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let result = match op {
            UnOp::FNeg | UnOp::Neg => self
                .builder
                .build_float_neg(val, "fneg")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?,
            UnOp::FAbs | UnOp::Abs => {
                // Use llvm.fabs intrinsic
                let intrinsic = Intrinsic::find("llvm.fabs")
                    .ok_or_else(|| CodegenError::Internal("llvm.fabs not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get fabs declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "fabs")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("fabs returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Sqrt => {
                let intrinsic = Intrinsic::find("llvm.sqrt")
                    .ok_or_else(|| CodegenError::Internal("llvm.sqrt not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get sqrt declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "sqrt")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("sqrt returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Floor => {
                let intrinsic = Intrinsic::find("llvm.floor")
                    .ok_or_else(|| CodegenError::Internal("llvm.floor not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get floor declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "floor")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("floor returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Ceil => {
                let intrinsic = Intrinsic::find("llvm.ceil")
                    .ok_or_else(|| CodegenError::Internal("llvm.ceil not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get ceil declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "ceil")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("ceil returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Round => {
                let intrinsic = Intrinsic::find("llvm.round")
                    .ok_or_else(|| CodegenError::Internal("llvm.round not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get round declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "round")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("round returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Trunc => {
                let intrinsic = Intrinsic::find("llvm.trunc")
                    .ok_or_else(|| CodegenError::Internal("llvm.trunc not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get trunc declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "trunc")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("trunc returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Exp => {
                let intrinsic = Intrinsic::find("llvm.exp")
                    .ok_or_else(|| CodegenError::Internal("llvm.exp not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get exp declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "exp")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("exp returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Log => {
                let intrinsic = Intrinsic::find("llvm.log")
                    .ok_or_else(|| CodegenError::Internal("llvm.log not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get log declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "log")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("log returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Sin => {
                let intrinsic = Intrinsic::find("llvm.sin")
                    .ok_or_else(|| CodegenError::Internal("llvm.sin not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get sin declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "sin")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("sin returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Cos => {
                let intrinsic = Intrinsic::find("llvm.cos")
                    .ok_or_else(|| CodegenError::Internal("llvm.cos not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get cos declaration".to_string())
                    })?;
                self.builder
                    .build_call(fn_val, &[val.into()], "cos")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("cos returned void".to_string()))?
                    .into_float_value()
            }
            UnOp::Rsqrt => {
                // rsqrt(x) = 1 / sqrt(x)
                let sqrt_intrinsic = Intrinsic::find("llvm.sqrt")
                    .ok_or_else(|| CodegenError::Internal("llvm.sqrt not found".to_string()))?;
                let fn_val = sqrt_intrinsic
                    .get_declaration(self.module.llvm_module(), &[val.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get sqrt declaration".to_string())
                    })?;
                let sqrt_val = self
                    .builder
                    .build_call(fn_val, &[val.into()], "sqrt")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("sqrt returned void".to_string()))?
                    .into_float_value();

                let one = val.get_type().const_float(1.0);
                self.builder
                    .build_float_div(one, sqrt_val, "rsqrt")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
            }
            UnOp::Not => return Err(CodegenError::TypeError("Bitwise NOT on float".to_string())),
        };
        Ok(result.into())
    }

    /// Lower a vector unary operation.
    fn lower_vector_unary(
        &mut self,
        op: UnOp,
        vec: VectorValue<'ctx>,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let elem_ty = vec.get_type().get_element_type();
        let result = if elem_ty.is_float_type() {
            match op {
                UnOp::FNeg | UnOp::Neg => self
                    .builder
                    .build_float_neg(vec, "vfneg")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                UnOp::FAbs | UnOp::Abs => {
                    let intrinsic = Intrinsic::find("llvm.fabs")
                        .ok_or_else(|| CodegenError::Internal("llvm.fabs not found".to_string()))?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get fabs declaration".to_string())
                        })?;
                    self.builder
                        .build_call(fn_val, &[vec.into()], "vfabs")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("vfabs returned void".to_string()))?
                }
                UnOp::Sqrt => {
                    let intrinsic = Intrinsic::find("llvm.sqrt")
                        .ok_or_else(|| CodegenError::Internal("llvm.sqrt not found".to_string()))?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get sqrt declaration".to_string())
                        })?;
                    self.builder
                        .build_call(fn_val, &[vec.into()], "vsqrt")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("vsqrt returned void".to_string()))?
                }
                _ => {
                    return Err(CodegenError::TypeError(format!(
                        "Unsupported float vector unary op {:?}",
                        op
                    )))
                }
            }
        } else {
            // Integer vector
            match op {
                UnOp::Neg => self
                    .builder
                    .build_int_neg(vec, "vneg")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                UnOp::Not => self
                    .builder
                    .build_not(vec, "vnot")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into(),
                _ => {
                    return Err(CodegenError::TypeError(format!(
                        "Unsupported int vector unary op {:?}",
                        op
                    )))
                }
            }
        };
        Ok(result)
    }

    /// Lower a comparison operation.
    fn lower_cmp(
        &mut self,
        op: CmpOp,
        lhs: &Value,
        rhs: &Value,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let l = self.lower_value(lhs)?;
        let r = self.lower_value(rhs)?;

        let result = match (l, r) {
            (BasicValueEnum::IntValue(li), BasicValueEnum::IntValue(ri)) => {
                let pred = match op {
                    CmpOp::Eq => IntPredicate::EQ,
                    CmpOp::Ne => IntPredicate::NE,
                    CmpOp::SLt => IntPredicate::SLT,
                    CmpOp::SLe => IntPredicate::SLE,
                    CmpOp::SGt => IntPredicate::SGT,
                    CmpOp::SGe => IntPredicate::SGE,
                    CmpOp::ULt => IntPredicate::ULT,
                    CmpOp::ULe => IntPredicate::ULE,
                    CmpOp::UGt => IntPredicate::UGT,
                    CmpOp::UGe => IntPredicate::UGE,
                    _ => {
                        return Err(CodegenError::TypeError(
                            "Float comparison on integers".to_string(),
                        ))
                    }
                };
                self.builder
                    .build_int_compare(pred, li, ri, "icmp")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into()
            }
            (BasicValueEnum::FloatValue(lf), BasicValueEnum::FloatValue(rf)) => {
                let pred = match op {
                    CmpOp::OEq | CmpOp::Eq => FloatPredicate::OEQ,
                    CmpOp::ONe | CmpOp::Ne => FloatPredicate::ONE,
                    CmpOp::OLt | CmpOp::SLt => FloatPredicate::OLT,
                    CmpOp::OLe | CmpOp::SLe => FloatPredicate::OLE,
                    CmpOp::OGt | CmpOp::SGt => FloatPredicate::OGT,
                    CmpOp::OGe | CmpOp::SGe => FloatPredicate::OGE,
                    _ => FloatPredicate::OEQ,
                };
                self.builder
                    .build_float_compare(pred, lf, rf, "fcmp")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .into()
            }
            _ => {
                return Err(CodegenError::TypeError(
                    "Mismatched types for comparison".to_string(),
                ))
            }
        };

        Ok(result)
    }

    /// Lower a select (ternary) operation.
    fn lower_select(
        &mut self,
        cond: &Value,
        then_val: &Value,
        else_val: &Value,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let c = self.lower_value(cond)?;
        let t = self.lower_value(then_val)?;
        let e = self.lower_value(else_val)?;

        let cond_bool = match c {
            BasicValueEnum::IntValue(i) => {
                if i.get_type().get_bit_width() == 1 {
                    i
                } else {
                    self.builder
                        .build_int_compare(
                            IntPredicate::NE,
                            i,
                            i.get_type().const_zero(),
                            "cond.bool",
                        )
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                }
            }
            _ => {
                return Err(CodegenError::TypeError(
                    "Select condition must be boolean".to_string(),
                ))
            }
        };

        let result = self
            .builder
            .build_select(cond_bool, t, e, "select")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(result)
    }

    /// Lower a cast operation.
    fn lower_cast(
        &mut self,
        val: &Value,
        target_ty: &LoopType,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let v = self.lower_value(val)?;
        let target = self.loop_type_to_llvm(target_ty)?;

        let result = match (v, target) {
            (BasicValueEnum::IntValue(i), BasicTypeEnum::IntType(ti)) => {
                let src_bits = i.get_type().get_bit_width();
                let dst_bits = ti.get_bit_width();
                if src_bits < dst_bits {
                    self.builder
                        .build_int_s_extend(i, ti, "sext")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .into()
                } else if src_bits > dst_bits {
                    self.builder
                        .build_int_truncate(i, ti, "trunc")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .into()
                } else {
                    v
                }
            }
            (BasicValueEnum::IntValue(i), BasicTypeEnum::FloatType(tf)) => self
                .builder
                .build_signed_int_to_float(i, tf, "sitofp")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                .into(),
            (BasicValueEnum::FloatValue(f), BasicTypeEnum::IntType(ti)) => self
                .builder
                .build_float_to_signed_int(f, ti, "fptosi")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                .into(),
            (BasicValueEnum::FloatValue(f), BasicTypeEnum::FloatType(tf)) => {
                let src_bits = match f.get_type().print_to_string().to_string().as_str() {
                    "half" => 16,
                    "float" => 32,
                    "double" => 64,
                    _ => 64,
                };
                let dst_bits = match tf.print_to_string().to_string().as_str() {
                    "half" => 16,
                    "float" => 32,
                    "double" => 64,
                    _ => 64,
                };
                if src_bits < dst_bits {
                    self.builder
                        .build_float_ext(f, tf, "fpext")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .into()
                } else if src_bits > dst_bits {
                    self.builder
                        .build_float_trunc(f, tf, "fptrunc")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .into()
                } else {
                    v
                }
            }
            _ => {
                return Err(CodegenError::Unsupported(format!(
                    "Cast from {:?} to {:?}",
                    v.get_type(),
                    target
                )))
            }
        };

        Ok(result)
    }

    /// Lower a broadcast operation (scalar to vector).
    fn lower_broadcast(&mut self, val: &Value, width: u8) -> CodegenResult<BasicValueEnum<'ctx>> {
        let scalar = self.lower_value(val)?;

        let vec_ty = match scalar {
            BasicValueEnum::FloatValue(f) => f.get_type().vec_type(width as u32),
            BasicValueEnum::IntValue(i) => i.get_type().vec_type(width as u32),
            _ => {
                return Err(CodegenError::TypeError(
                    "Cannot broadcast non-scalar".to_string(),
                ))
            }
        };

        // Create vector with all same value
        let mut vec = vec_ty.get_undef();
        for i in 0..width {
            let idx = self.llvm_ctx.i32_type().const_int(i as u64, false);
            vec = self
                .builder
                .build_insert_element(vec, scalar, idx, "broadcast")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        }

        Ok(vec.into())
    }

    /// Lower an extract operation (vector to scalar).
    fn lower_extract(&mut self, vec: &Value, idx: u8) -> CodegenResult<BasicValueEnum<'ctx>> {
        let v = self.lower_value(vec)?;
        let vector = match v {
            BasicValueEnum::VectorValue(vec) => vec,
            _ => {
                return Err(CodegenError::TypeError(
                    "Extract requires vector".to_string(),
                ))
            }
        };

        let index = self.llvm_ctx.i32_type().const_int(idx as u64, false);
        let result = self
            .builder
            .build_extract_element(vector, index, "extract")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(result)
    }

    /// Lower an insert operation.
    fn lower_insert(
        &mut self,
        vec: &Value,
        val: &Value,
        idx: u8,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let v = self.lower_value(vec)?;
        let scalar = self.lower_value(val)?;

        let vector = match v {
            BasicValueEnum::VectorValue(vec) => vec,
            _ => {
                return Err(CodegenError::TypeError(
                    "Insert requires vector".to_string(),
                ))
            }
        };

        let index = self.llvm_ctx.i32_type().const_int(idx as u64, false);
        let result = self
            .builder
            .build_insert_element(vector, scalar, index, "insert")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(result.into())
    }

    /// Lower a shuffle operation.
    fn lower_shuffle(
        &mut self,
        v1: &Value,
        v2: &Value,
        mask: &[i32],
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let vec1 = match self.lower_value(v1)? {
            BasicValueEnum::VectorValue(v) => v,
            _ => {
                return Err(CodegenError::TypeError(
                    "Shuffle requires vectors".to_string(),
                ))
            }
        };
        let vec2 = match self.lower_value(v2)? {
            BasicValueEnum::VectorValue(v) => v,
            _ => {
                return Err(CodegenError::TypeError(
                    "Shuffle requires vectors".to_string(),
                ))
            }
        };

        // Build mask vector
        let i32_ty = self.llvm_ctx.i32_type();
        let mask_vec: Vec<IntValue> = mask
            .iter()
            .map(|&i| {
                if i < 0 {
                    i32_ty.get_undef()
                } else {
                    i32_ty.const_int(i as u64, false)
                }
            })
            .collect();

        let result = self
            .builder
            .build_shuffle_vector(vec1, vec2, VectorType::const_vector(&mask_vec), "shuffle")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        Ok(result.into())
    }

    /// Lower a vector reduction operation.
    fn lower_vec_reduce(
        &mut self,
        op: ReduceOp,
        val: &Value,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let vec = match self.lower_value(val)? {
            BasicValueEnum::VectorValue(v) => v,
            _ => {
                return Err(CodegenError::TypeError(
                    "VecReduce requires vector".to_string(),
                ))
            }
        };

        let elem_ty = vec.get_type().get_element_type();

        // Use LLVM reduction intrinsics
        let result = if elem_ty.is_float_type() {
            match op {
                ReduceOp::Add => {
                    let intrinsic =
                        Intrinsic::find("llvm.vector.reduce.fadd").ok_or_else(|| {
                            CodegenError::Internal("reduce.fadd not found".to_string())
                        })?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get reduce.fadd".to_string())
                        })?;
                    let zero = elem_ty.into_float_type().const_float(0.0);
                    self.builder
                        .build_call(fn_val, &[zero.into(), vec.into()], "vreduce.fadd")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("reduce returned void".to_string()))?
                }
                ReduceOp::Mul => {
                    let intrinsic =
                        Intrinsic::find("llvm.vector.reduce.fmul").ok_or_else(|| {
                            CodegenError::Internal("reduce.fmul not found".to_string())
                        })?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get reduce.fmul".to_string())
                        })?;
                    let one = elem_ty.into_float_type().const_float(1.0);
                    self.builder
                        .build_call(fn_val, &[one.into(), vec.into()], "vreduce.fmul")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("reduce returned void".to_string()))?
                }
                ReduceOp::Min => {
                    let intrinsic =
                        Intrinsic::find("llvm.vector.reduce.fmin").ok_or_else(|| {
                            CodegenError::Internal("reduce.fmin not found".to_string())
                        })?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get reduce.fmin".to_string())
                        })?;
                    self.builder
                        .build_call(fn_val, &[vec.into()], "vreduce.fmin")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("reduce returned void".to_string()))?
                }
                ReduceOp::Max => {
                    let intrinsic =
                        Intrinsic::find("llvm.vector.reduce.fmax").ok_or_else(|| {
                            CodegenError::Internal("reduce.fmax not found".to_string())
                        })?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get reduce.fmax".to_string())
                        })?;
                    self.builder
                        .build_call(fn_val, &[vec.into()], "vreduce.fmax")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("reduce returned void".to_string()))?
                }
                _ => {
                    return Err(CodegenError::Unsupported(format!(
                        "Float reduction {:?}",
                        op
                    )))
                }
            }
        } else {
            // Integer reductions
            match op {
                ReduceOp::Add => {
                    let intrinsic = Intrinsic::find("llvm.vector.reduce.add").ok_or_else(|| {
                        CodegenError::Internal("reduce.add not found".to_string())
                    })?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get reduce.add".to_string())
                        })?;
                    self.builder
                        .build_call(fn_val, &[vec.into()], "vreduce.add")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("reduce returned void".to_string()))?
                }
                ReduceOp::Mul => {
                    let intrinsic = Intrinsic::find("llvm.vector.reduce.mul").ok_or_else(|| {
                        CodegenError::Internal("reduce.mul not found".to_string())
                    })?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get reduce.mul".to_string())
                        })?;
                    self.builder
                        .build_call(fn_val, &[vec.into()], "vreduce.mul")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("reduce returned void".to_string()))?
                }
                ReduceOp::And => {
                    let intrinsic = Intrinsic::find("llvm.vector.reduce.and").ok_or_else(|| {
                        CodegenError::Internal("reduce.and not found".to_string())
                    })?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get reduce.and".to_string())
                        })?;
                    self.builder
                        .build_call(fn_val, &[vec.into()], "vreduce.and")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("reduce returned void".to_string()))?
                }
                ReduceOp::Or => {
                    let intrinsic = Intrinsic::find("llvm.vector.reduce.or")
                        .ok_or_else(|| CodegenError::Internal("reduce.or not found".to_string()))?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get reduce.or".to_string())
                        })?;
                    self.builder
                        .build_call(fn_val, &[vec.into()], "vreduce.or")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("reduce returned void".to_string()))?
                }
                ReduceOp::Xor => {
                    let intrinsic = Intrinsic::find("llvm.vector.reduce.xor").ok_or_else(|| {
                        CodegenError::Internal("reduce.xor not found".to_string())
                    })?;
                    let fn_val = intrinsic
                        .get_declaration(self.module.llvm_module(), &[vec.get_type().into()])
                        .ok_or_else(|| {
                            CodegenError::Internal("Failed to get reduce.xor".to_string())
                        })?;
                    self.builder
                        .build_call(fn_val, &[vec.into()], "vreduce.xor")
                        .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CodegenError::Internal("reduce returned void".to_string()))?
                }
                _ => {
                    return Err(CodegenError::Unsupported(format!(
                        "Integer reduction {:?}",
                        op
                    )))
                }
            }
        };

        Ok(result)
    }

    /// Lower a fused multiply-add operation.
    fn lower_fma(
        &mut self,
        a: &Value,
        b: &Value,
        c: &Value,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let av = self.lower_value(a)?;
        let bv = self.lower_value(b)?;
        let cv = self.lower_value(c)?;

        match (av, bv, cv) {
            (
                BasicValueEnum::FloatValue(af),
                BasicValueEnum::FloatValue(bf),
                BasicValueEnum::FloatValue(cf),
            ) => {
                let intrinsic = Intrinsic::find("llvm.fma")
                    .ok_or_else(|| CodegenError::Internal("llvm.fma not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[af.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get fma declaration".to_string())
                    })?;
                let result = self
                    .builder
                    .build_call(fn_val, &[af.into(), bf.into(), cf.into()], "fma")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("fma returned void".to_string()))?;
                Ok(result)
            }
            (
                BasicValueEnum::VectorValue(av),
                BasicValueEnum::VectorValue(bv),
                BasicValueEnum::VectorValue(cv),
            ) => {
                // Vector FMA
                let intrinsic = Intrinsic::find("llvm.fma")
                    .ok_or_else(|| CodegenError::Internal("llvm.fma not found".to_string()))?;
                let fn_val = intrinsic
                    .get_declaration(self.module.llvm_module(), &[av.get_type().into()])
                    .ok_or_else(|| {
                        CodegenError::Internal("Failed to get vector fma declaration".to_string())
                    })?;
                let result = self
                    .builder
                    .build_call(fn_val, &[av.into(), bv.into(), cv.into()], "vfma")
                    .map_err(|e| CodegenError::LlvmError(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("vfma returned void".to_string()))?;
                Ok(result)
            }
            _ => {
                // Fall back to mul + add
                let mul = self.lower_binary(BinOp::Mul, a, b)?;
                let mul_val = match mul {
                    BasicValueEnum::FloatValue(f) => Value::FloatConst(0.0, ScalarType::F64), // placeholder
                    _ => return Err(CodegenError::TypeError("FMA requires floats".to_string())),
                };
                // This is a simplified fallback; in practice we'd need proper handling
                Err(CodegenError::Unsupported(
                    "FMA on non-float types".to_string(),
                ))
            }
        }
    }

    /// Lower a pointer addition.
    fn lower_ptr_add(
        &mut self,
        ptr: &Value,
        offset: &Value,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let ptr_val = self.lower_value(ptr)?;
        let offset_val = self.lower_value(offset)?;

        let ptr_ptr = match ptr_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => {
                return Err(CodegenError::TypeError(
                    "PtrAdd requires pointer".to_string(),
                ))
            }
        };
        let offset_int = self.value_to_int(offset_val)?;

        let i8_ty = self.llvm_ctx.i8_type();
        let result = unsafe {
            self.builder
                .build_gep(i8_ty, ptr_ptr, &[offset_int], "ptr_add")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?
        };

        Ok(result.into())
    }

    /// Lower a GetPtr operation (get pointer to buffer element).
    fn lower_get_ptr(
        &mut self,
        buffer: BufferId,
        index: &Value,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let base_ptr = self
            .buffers
            .get(&buffer)
            .copied()
            .ok_or_else(|| CodegenError::Internal(format!("Unknown buffer: {:?}", buffer)))?;

        let idx_val = self.lower_value(index)?;
        let idx_int = self.value_to_int(idx_val)?;

        let i8_ty = self.llvm_ctx.i8_type();
        let result = unsafe {
            self.builder
                .build_gep(i8_ty, base_ptr, &[idx_int], "getptr")
                .map_err(|e| CodegenError::LlvmError(e.to_string()))?
        };

        Ok(result.into())
    }

    /// Lower a Value to an LLVM value.
    fn lower_value(&mut self, val: &Value) -> CodegenResult<BasicValueEnum<'ctx>> {
        match val {
            Value::Var(vid, _ty) => self.get_value(*vid),
            Value::IntConst(n, ty) => {
                let int_ty = match ty {
                    ScalarType::Int(bits) | ScalarType::UInt(bits) => {
                        self.llvm_ctx.custom_width_int_type(*bits as u32)
                    }
                    ScalarType::Bool => self.llvm_ctx.bool_type(),
                    _ => {
                        return Err(CodegenError::TypeError(
                            "IntConst with non-int type".to_string(),
                        ))
                    }
                };
                Ok(int_ty.const_int(*n as u64, true).into())
            }
            Value::FloatConst(f, ty) => {
                let float_ty = match ty {
                    ScalarType::Float(16) => self.llvm_ctx.f16_type(),
                    ScalarType::Float(32) => self.llvm_ctx.f32_type(),
                    ScalarType::Float(64) => self.llvm_ctx.f64_type(),
                    _ => {
                        return Err(CodegenError::TypeError(
                            "FloatConst with non-float type".to_string(),
                        ))
                    }
                };
                Ok(float_ty.const_float(*f).into())
            }
            Value::BoolConst(b) => Ok(self.llvm_ctx.bool_type().const_int(*b as u64, false).into()),
            Value::Undef(ty) => {
                let llvm_ty = self.loop_type_to_llvm(ty)?;
                let undef = match llvm_ty {
                    BasicTypeEnum::IntType(t) => t.get_undef().into(),
                    BasicTypeEnum::FloatType(t) => t.get_undef().into(),
                    BasicTypeEnum::VectorType(t) => t.get_undef().into(),
                    BasicTypeEnum::PointerType(t) => t.get_undef().into(),
                    _ => {
                        return Err(CodegenError::Internal(
                            "Cannot create undef for type".to_string(),
                        ))
                    }
                };
                Ok(undef)
            }
        }
    }

    /// Get an already-lowered value by ID.
    fn get_value(&self, vid: ValueId) -> CodegenResult<BasicValueEnum<'ctx>> {
        self.values
            .get(&vid)
            .copied()
            .ok_or_else(|| CodegenError::Internal(format!("Undefined value: {:?}", vid)))
    }

    /// Convert a basic value to an integer.
    fn value_to_int(&self, val: BasicValueEnum<'ctx>) -> CodegenResult<IntValue<'ctx>> {
        match val {
            BasicValueEnum::IntValue(i) => Ok(i),
            _ => Err(CodegenError::TypeError(
                "Expected integer value".to_string(),
            )),
        }
    }
}

/// Lower a complete Loop IR module to LLVM.
pub fn lower_loop_ir<'a>(
    ctx: &'a LlvmContext,
    module: &'a LlvmModule<'a>,
    ir: &LoopIR,
) -> CodegenResult<FunctionValue<'a>> {
    let mut lowering = LoopLowering::new(ctx, module);
    lowering.lower_function(ir)
}

/// Lower multiple Loop IR functions.
pub fn lower_loop_irs<'a>(
    ctx: &'a LlvmContext,
    module: &'a LlvmModule<'a>,
    irs: &[LoopIR],
) -> CodegenResult<Vec<FunctionValue<'a>>> {
    let mut lowering = LoopLowering::new(ctx, module);
    irs.iter().map(|ir| lowering.lower_function(ir)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_lowering_creation() {
        let backend = crate::LlvmBackend::new();
        let config = crate::CodegenConfig::default();
        let ctx = backend.create_context(config).unwrap();
        let module = ctx.create_module("test").unwrap();

        let lowering = LoopLowering::new(&ctx, &module);
        assert!(lowering.values.is_empty());
        assert!(lowering.buffers.is_empty());
    }

    #[test]
    fn test_scalar_type_mapping() {
        let backend = crate::LlvmBackend::new();
        let config = crate::CodegenConfig::default();
        let ctx = backend.create_context(config).unwrap();
        let module = ctx.create_module("test").unwrap();

        let lowering = LoopLowering::new(&ctx, &module);

        // Test i32
        let i32_ty = lowering.scalar_type_to_llvm(ScalarType::Int(32)).unwrap();
        assert!(matches!(i32_ty, BasicTypeEnum::IntType(_)));

        // Test f64
        let f64_ty = lowering.scalar_type_to_llvm(ScalarType::Float(64)).unwrap();
        assert!(matches!(f64_ty, BasicTypeEnum::FloatType(_)));
    }
}
