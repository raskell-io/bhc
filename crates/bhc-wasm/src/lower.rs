//! Loop IR to WASM lowering.
//!
//! This module implements the lowering of Loop IR to WebAssembly instructions.
//! It handles statements, loops, operations, and memory access patterns.

use crate::codegen::{WasmFunc, WasmFuncType};
use crate::{WasmConfig, WasmError, WasmInstr, WasmResult, WasmType};
use bhc_index::Idx;
use bhc_loop_ir::{
    BinOp, Body, CmpOp, Loop, LoopIR, LoopType, MemRef, Op, ScalarType, Stmt, UnOp, Value, ValueId,
};
use bhc_tensor_ir::BufferId;
use rustc_hash::FxHashMap;

/// Lowers Loop IR to WASM functions.
pub struct LoopIRLowerer {
    /// Local variable mapping: ValueId -> local index.
    locals: FxHashMap<ValueId, u32>,
    /// Buffer base pointers: BufferId -> local index holding base address.
    buffers: FxHashMap<BufferId, u32>,
    /// Current loop depth (for branch targets).
    loop_depth: u32,
    /// WASM configuration.
    config: WasmConfig,
    /// Next local index to allocate.
    next_local: u32,
}

impl LoopIRLowerer {
    /// Create a new lowerer with the given configuration.
    #[must_use]
    pub fn new(config: WasmConfig) -> Self {
        Self {
            locals: FxHashMap::default(),
            buffers: FxHashMap::default(),
            loop_depth: 0,
            config,
            next_local: 0,
        }
    }

    /// Lower a Loop IR function to a WASM function.
    pub fn lower_function(&mut self, ir: &LoopIR) -> WasmResult<WasmFunc> {
        // Reset state for new function
        self.locals.clear();
        self.buffers.clear();
        self.loop_depth = 0;

        // Build parameter types
        let param_types: Vec<WasmType> = ir
            .params
            .iter()
            .map(|p| self.lower_loop_type(&p.ty))
            .collect();

        // Build result type (single return)
        let result_types: Vec<WasmType> = match &ir.return_ty {
            LoopType::Void => vec![],
            ty => vec![self.lower_loop_type(ty)],
        };

        let mut func = WasmFunc::new(WasmFuncType::new(param_types.clone(), result_types));
        func.name = Some(ir.name.as_str().to_string());

        // Assign parameter indices
        self.next_local = 0;
        for (i, param) in ir.params.iter().enumerate() {
            // Parameters have indices based on their position
            self.next_local = (i + 1) as u32;
            // Map param value to its index
            // Note: In LoopIR, params have their own ValueIds from the function signature
        }

        // Allocate locals for allocations
        for alloc in &ir.allocs {
            let local_idx = func.add_local(WasmType::I32);
            self.buffers.insert(alloc.buffer, local_idx);
        }

        // Lower the body
        self.lower_body(&mut func, &ir.body)?;

        // Ensure function ends properly
        func.emit(WasmInstr::End);

        Ok(func)
    }

    /// Lower a function body.
    fn lower_body(&mut self, func: &mut WasmFunc, body: &Body) -> WasmResult<()> {
        for stmt in &body.stmts {
            self.lower_stmt(func, stmt)?;
        }
        Ok(())
    }

    /// Lower a single statement.
    fn lower_stmt(&mut self, func: &mut WasmFunc, stmt: &Stmt) -> WasmResult<()> {
        match stmt {
            Stmt::Assign(value_id, op) => {
                // Allocate a local for this value if not already done
                let local_idx = self.get_or_create_local(func, *value_id, op);

                // Emit the operation
                self.lower_op(func, op)?;

                // Store result in local
                func.emit(WasmInstr::LocalSet(local_idx));
            }

            Stmt::Store(mem_ref, value) => {
                // Emit address calculation
                self.emit_address(func, mem_ref)?;

                // Emit value to store
                self.emit_value(func, value)?;

                // Emit store instruction based on type
                let store_instr = self.store_instr_for_memref(mem_ref);
                func.emit(store_instr);
            }

            Stmt::Loop(loop_) => {
                self.lower_loop(func, loop_)?;
            }

            Stmt::If(if_stmt) => {
                // Emit condition
                self.emit_value(func, &if_stmt.cond)?;

                // Start if block
                func.emit(WasmInstr::If(None));

                // Lower then body
                self.lower_body(func, &if_stmt.then_body)?;

                // Lower else body if present
                if let Some(ref else_stmts) = if_stmt.else_body {
                    func.emit(WasmInstr::Else);
                    self.lower_body(func, else_stmts)?;
                }

                func.emit(WasmInstr::End);
            }

            Stmt::Return(value) => {
                if let Some(v) = value {
                    self.emit_value(func, v)?;
                }
                func.emit(WasmInstr::Return);
            }

            Stmt::Comment(text) => {
                if self.config.debug_names {
                    func.emit(WasmInstr::Comment(text.clone()));
                }
            }

            Stmt::Call(result, _func_name, args) => {
                // Emit arguments
                for arg in args {
                    self.emit_value(func, arg)?;
                }
                // Call is not directly supported - would need function index resolution
                // For now, emit a placeholder
                if result.is_some() {
                    func.emit(WasmInstr::I32Const(0)); // Placeholder return value
                }
            }

            Stmt::Barrier(_kind) => {
                // Memory barriers don't have direct WASM equivalent in basic spec
                // In threaded WASM, this would use atomic.fence
                // For now, emit nothing (single-threaded assumption)
            }
        }

        Ok(())
    }

    /// Lower a loop construct.
    fn lower_loop(&mut self, func: &mut WasmFunc, loop_: &Loop) -> WasmResult<()> {
        // Get or create the loop variable local
        let loop_var_local = self.get_or_create_local_for_id(func, loop_.var, WasmType::I32);

        // Initialize loop variable to lower bound
        self.emit_value(func, &loop_.lower)?;
        func.emit(WasmInstr::LocalSet(loop_var_local));

        // Create block/loop structure for proper branching:
        // (block $break
        //   (loop $continue
        //     ... body ...
        //     (br_if $continue (i < end))
        //   )
        // )
        self.loop_depth += 1;

        func.emit(WasmInstr::Block(None)); // break target (depth 1)
        func.emit(WasmInstr::Loop(None)); // continue target (depth 0)

        // Loop body
        self.lower_body(func, &loop_.body)?;

        // Increment loop variable
        func.emit(WasmInstr::LocalGet(loop_var_local));
        func.emit(WasmInstr::I32Const(1));
        func.emit(WasmInstr::I32Add);
        func.emit(WasmInstr::LocalSet(loop_var_local));

        // Check condition: if var < upper bound, branch back to loop
        func.emit(WasmInstr::LocalGet(loop_var_local));
        self.emit_value(func, &loop_.upper)?;
        func.emit(WasmInstr::I32LtS);
        func.emit(WasmInstr::BrIf(0)); // branch to loop if condition true

        func.emit(WasmInstr::End); // end loop
        func.emit(WasmInstr::End); // end block

        self.loop_depth -= 1;

        Ok(())
    }

    /// Lower an operation and push result onto stack.
    fn lower_op(&mut self, func: &mut WasmFunc, op: &Op) -> WasmResult<()> {
        match op {
            Op::Load(mem_ref) => {
                self.emit_address(func, mem_ref)?;
                let load_instr = self.load_instr_for_memref(mem_ref);
                func.emit(load_instr);
            }

            Op::Binary(bin_op, lhs, rhs) => {
                self.emit_value(func, lhs)?;
                self.emit_value(func, rhs)?;
                let instr = self.binary_op_instr(bin_op, lhs)?;
                func.emit(instr);
            }

            Op::Unary(un_op, val) => {
                self.emit_value(func, val)?;
                let instr = self.unary_op_instr(un_op, val)?;
                func.emit(instr);
            }

            Op::Cmp(cmp_op, lhs, rhs) => {
                self.emit_value(func, lhs)?;
                self.emit_value(func, rhs)?;
                let instr = self.cmp_op_instr(cmp_op, lhs)?;
                func.emit(instr);
            }

            Op::Select(cond, then_val, else_val) => {
                self.emit_value(func, then_val)?;
                self.emit_value(func, else_val)?;
                self.emit_value(func, cond)?;
                func.emit(WasmInstr::Select);
            }

            Op::Cast(val, target_ty) => {
                self.emit_value(func, val)?;
                let instr = self.cast_instr(val, target_ty)?;
                if let Some(i) = instr {
                    func.emit(i);
                }
            }

            Op::Broadcast(val, lanes) => {
                // Splat scalar to vector
                self.emit_value(func, val)?;
                let splat_instr = match lanes {
                    4 => WasmInstr::F32x4Splat,
                    2 => WasmInstr::F64x2Splat,
                    _ => WasmInstr::I32x4Splat,
                };
                func.emit(splat_instr);
            }

            Op::Extract(val, lane) => {
                self.emit_value(func, val)?;
                // Default to f32x4 extract
                func.emit(WasmInstr::F32x4ExtractLane(*lane));
            }

            Op::Insert(vec_val, scalar_val, lane) => {
                self.emit_value(func, vec_val)?;
                self.emit_value(func, scalar_val)?;
                func.emit(WasmInstr::F32x4ReplaceLane(*lane));
            }

            Op::VecReduce(reduce_op, val) => {
                // Vector reduction - extract lanes and combine
                self.emit_value(func, val)?;
                // For now, emit a simple horizontal add for f32x4
                // This is a simplified implementation
                // Lane 0
                func.emit(WasmInstr::F32x4ExtractLane(0));
                // Would need to extract other lanes and add them
                // This is a placeholder for proper SIMD reduction
            }

            Op::Fma(a, b, c) => {
                // Fused multiply-add: a * b + c
                // WASM doesn't have native FMA, so we emit mul + add
                self.emit_value(func, a)?;
                self.emit_value(func, b)?;
                func.emit(WasmInstr::F32Mul);
                self.emit_value(func, c)?;
                func.emit(WasmInstr::F32Add);
            }

            Op::PtrAdd(base, offset) => {
                self.emit_value(func, base)?;
                self.emit_value(func, offset)?;
                func.emit(WasmInstr::I32Add);
            }

            Op::GetPtr(buffer_id, index) => {
                // Get base pointer from buffer
                if let Some(&local) = self.buffers.get(buffer_id) {
                    func.emit(WasmInstr::LocalGet(local));
                } else {
                    // Buffer not found, use constant 0 as fallback
                    func.emit(WasmInstr::I32Const(0));
                }
                // Add index * element_size
                self.emit_value(func, index)?;
                func.emit(WasmInstr::I32Const(4)); // Assume 4-byte elements
                func.emit(WasmInstr::I32Mul);
                func.emit(WasmInstr::I32Add);
            }

            Op::Phi(branches) => {
                // Phi nodes are handled by SSA construction
                // For now, just load from first branch as fallback
                if let Some((_, val)) = branches.first() {
                    self.emit_value(func, val)?;
                } else {
                    func.emit(WasmInstr::I32Const(0));
                }
            }

            Op::Shuffle(_, _, _) => {
                // Vector shuffle - complex to implement
                return Err(WasmError::NotSupported(
                    "vector shuffle not yet implemented".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Emit a value onto the stack.
    fn emit_value(&mut self, func: &mut WasmFunc, value: &Value) -> WasmResult<()> {
        match value {
            Value::IntConst(v, scalar_ty) => {
                let instr = match scalar_ty {
                    ScalarType::Int(32) | ScalarType::UInt(32) => WasmInstr::I32Const(*v as i32),
                    ScalarType::Int(64) | ScalarType::UInt(64) => WasmInstr::I64Const(*v),
                    ScalarType::Int(bits) | ScalarType::UInt(bits) if *bits <= 32 => {
                        WasmInstr::I32Const(*v as i32)
                    }
                    _ => WasmInstr::I64Const(*v),
                };
                func.emit(instr);
            }

            Value::FloatConst(v, scalar_ty) => {
                let instr = match scalar_ty {
                    ScalarType::Float(32) => WasmInstr::F32Const(*v as f32),
                    ScalarType::Float(64) => WasmInstr::F64Const(*v),
                    _ => WasmInstr::F64Const(*v),
                };
                func.emit(instr);
            }

            Value::BoolConst(v) => {
                func.emit(WasmInstr::I32Const(if *v { 1 } else { 0 }));
            }

            Value::Var(value_id, _ty) => {
                if let Some(&local) = self.locals.get(value_id) {
                    func.emit(WasmInstr::LocalGet(local));
                } else {
                    // Not found - might be a parameter
                    func.emit(WasmInstr::LocalGet(value_id.index() as u32));
                }
            }

            Value::Undef(ty) => {
                // Emit a zero value for undefined
                match ty {
                    LoopType::Scalar(ScalarType::Float(64)) => func.emit(WasmInstr::F64Const(0.0)),
                    LoopType::Scalar(ScalarType::Float(32)) => func.emit(WasmInstr::F32Const(0.0)),
                    LoopType::Scalar(ScalarType::Int(64) | ScalarType::UInt(64)) => {
                        func.emit(WasmInstr::I64Const(0))
                    }
                    _ => func.emit(WasmInstr::I32Const(0)),
                }
            }
        }

        Ok(())
    }

    /// Emit address calculation for a memory reference.
    fn emit_address(&mut self, func: &mut WasmFunc, mem_ref: &MemRef) -> WasmResult<()> {
        // Get base address from buffer
        if let Some(&local) = self.buffers.get(&mem_ref.buffer) {
            func.emit(WasmInstr::LocalGet(local));
        } else {
            // Unknown buffer - use global memory base (0)
            func.emit(WasmInstr::I32Const(0));
        }

        // Add index (which contains the byte offset calculation)
        self.emit_value(func, &mem_ref.index)?;

        // Calculate: base + index * element_size
        let elem_size = mem_ref.elem_ty.size_bytes() as i32;
        func.emit(WasmInstr::I32Const(elem_size));
        func.emit(WasmInstr::I32Mul);
        func.emit(WasmInstr::I32Add);

        Ok(())
    }

    /// Get the WASM load instruction for a memory reference.
    fn load_instr_for_memref(&self, mem_ref: &MemRef) -> WasmInstr {
        let align = mem_ref.elem_ty.size_bytes().min(4) as u32;
        match &mem_ref.elem_ty {
            LoopType::Scalar(ScalarType::Int(32)) | LoopType::Scalar(ScalarType::UInt(32)) => {
                WasmInstr::I32Load(align, 0)
            }
            LoopType::Scalar(ScalarType::Int(64)) | LoopType::Scalar(ScalarType::UInt(64)) => {
                WasmInstr::I64Load(align.min(8), 0)
            }
            LoopType::Scalar(ScalarType::Float(32)) => WasmInstr::F32Load(4, 0),
            LoopType::Scalar(ScalarType::Float(64)) => WasmInstr::F64Load(8, 0),
            _ => WasmInstr::I32Load(4, 0), // Default to i32
        }
    }

    /// Get the WASM store instruction for a memory reference.
    fn store_instr_for_memref(&self, mem_ref: &MemRef) -> WasmInstr {
        let align = mem_ref.elem_ty.size_bytes().min(4) as u32;
        match &mem_ref.elem_ty {
            LoopType::Scalar(ScalarType::Int(32)) | LoopType::Scalar(ScalarType::UInt(32)) => {
                WasmInstr::I32Store(align, 0)
            }
            LoopType::Scalar(ScalarType::Int(64)) | LoopType::Scalar(ScalarType::UInt(64)) => {
                WasmInstr::I64Store(align.min(8), 0)
            }
            LoopType::Scalar(ScalarType::Float(32)) => WasmInstr::F32Store(4, 0),
            LoopType::Scalar(ScalarType::Float(64)) => WasmInstr::F64Store(8, 0),
            _ => WasmInstr::I32Store(4, 0), // Default to i32
        }
    }

    /// Get the WASM instruction for a binary operation.
    fn binary_op_instr(&self, op: &BinOp, lhs: &Value) -> WasmResult<WasmInstr> {
        // Determine type from lhs value
        let is_float = self.is_float_value(lhs);
        let is_64bit = self.is_64bit_value(lhs);

        let instr = match (op, is_float, is_64bit) {
            (BinOp::Add, false, false) => WasmInstr::I32Add,
            (BinOp::Add, false, true) => WasmInstr::I64Add,
            (BinOp::Add, true, false) => WasmInstr::F32Add,
            (BinOp::Add, true, true) => WasmInstr::F64Add,

            (BinOp::Sub, false, false) => WasmInstr::I32Sub,
            (BinOp::Sub, false, true) => WasmInstr::I64Sub,
            (BinOp::Sub, true, false) => WasmInstr::F32Sub,
            (BinOp::Sub, true, true) => WasmInstr::F64Sub,

            (BinOp::Mul, false, false) => WasmInstr::I32Mul,
            (BinOp::Mul, false, true) => WasmInstr::I64Mul,
            (BinOp::Mul, true, false) => WasmInstr::F32Mul,
            (BinOp::Mul, true, true) => WasmInstr::F64Mul,

            (BinOp::SDiv, false, false) => WasmInstr::I32DivS,
            (BinOp::SDiv, false, true) => WasmInstr::I64DivS,
            (BinOp::UDiv, false, false) => WasmInstr::I32DivU,
            (BinOp::UDiv, false, true) => WasmInstr::I64DivU,
            (BinOp::FDiv, true, false) => WasmInstr::F32Div,
            (BinOp::FDiv, true, true) => WasmInstr::F64Div,

            (BinOp::SRem, false, false) => WasmInstr::I32RemS,
            (BinOp::SRem, false, true) => WasmInstr::I64RemS,
            (BinOp::URem, false, false) => WasmInstr::I32RemU,
            (BinOp::URem, false, true) => WasmInstr::I64RemU,

            (BinOp::And, _, false) => WasmInstr::I32And,
            (BinOp::And, _, true) => WasmInstr::I64And,
            (BinOp::Or, _, false) => WasmInstr::I32Or,
            (BinOp::Or, _, true) => WasmInstr::I64Or,
            (BinOp::Xor, _, false) => WasmInstr::I32Xor,
            (BinOp::Xor, _, true) => WasmInstr::I64Xor,

            (BinOp::Shl, _, false) => WasmInstr::I32Shl,
            (BinOp::Shl, _, true) => WasmInstr::I64Shl,
            (BinOp::LShr, _, false) => WasmInstr::I32ShrU,
            (BinOp::LShr, _, true) => WasmInstr::I64ShrU,
            (BinOp::AShr, _, false) => WasmInstr::I32ShrS,
            (BinOp::AShr, _, true) => WasmInstr::I64ShrS,

            _ => {
                return Err(WasmError::NotSupported(format!(
                    "binary op {:?} not supported",
                    op
                )));
            }
        };

        Ok(instr)
    }

    /// Get the WASM instruction for a unary operation.
    fn unary_op_instr(&self, op: &UnOp, val: &Value) -> WasmResult<WasmInstr> {
        let is_float = self.is_float_value(val);
        let is_64bit = self.is_64bit_value(val);

        let instr = match (op, is_float, is_64bit) {
            (UnOp::Neg, true, false) => WasmInstr::F32Neg,
            (UnOp::Neg, true, true) => WasmInstr::F64Neg,
            (UnOp::Abs, true, false) => WasmInstr::F32Abs,
            (UnOp::Abs, true, true) => WasmInstr::F64Abs,
            (UnOp::Sqrt, true, false) => WasmInstr::F32Sqrt,
            (UnOp::Sqrt, true, true) => WasmInstr::F64Sqrt,
            (UnOp::Ceil, true, false) => WasmInstr::F32Ceil,
            (UnOp::Ceil, true, true) => WasmInstr::F64Ceil,
            (UnOp::Floor, true, false) => WasmInstr::F32Floor,
            (UnOp::Floor, true, true) => WasmInstr::F64Floor,
            (UnOp::Trunc, true, false) => WasmInstr::F32Trunc,
            (UnOp::Trunc, true, true) => WasmInstr::F64Trunc,

            (UnOp::Not, _, false) => {
                // i32 not is done with xor -1
                return Err(WasmError::NotSupported(
                    "integer negation requires special handling".to_string(),
                ));
            }

            _ => {
                return Err(WasmError::NotSupported(format!(
                    "unary op {:?} not supported",
                    op
                )));
            }
        };

        Ok(instr)
    }

    /// Get the WASM instruction for a comparison operation.
    fn cmp_op_instr(&self, op: &CmpOp, lhs: &Value) -> WasmResult<WasmInstr> {
        let is_float = self.is_float_value(lhs);
        let is_64bit = self.is_64bit_value(lhs);

        let instr = match (op, is_float, is_64bit) {
            (CmpOp::Eq, false, false) => WasmInstr::I32Eq,
            (CmpOp::Eq, false, true) => WasmInstr::I64Eq,
            (CmpOp::Eq, true, false) => WasmInstr::F32Eq,
            (CmpOp::Eq, true, true) => WasmInstr::F64Eq,

            (CmpOp::Ne, false, false) => WasmInstr::I32Ne,
            (CmpOp::Ne, false, true) => WasmInstr::I64Ne,
            (CmpOp::Ne, true, false) => WasmInstr::F32Ne,
            (CmpOp::Ne, true, true) => WasmInstr::F64Ne,

            (CmpOp::SLt, false, false) => WasmInstr::I32LtS,
            (CmpOp::SLt, false, true) => WasmInstr::I64LtS,
            (CmpOp::ULt, false, false) => WasmInstr::I32LtU,
            (CmpOp::ULt, false, true) => WasmInstr::I64LtU,
            (CmpOp::OLt, true, false) => WasmInstr::F32Lt,
            (CmpOp::OLt, true, true) => WasmInstr::F64Lt,

            (CmpOp::SLe, false, false) => WasmInstr::I32LeS,
            (CmpOp::SLe, false, true) => WasmInstr::I64LeS,
            (CmpOp::ULe, false, false) => WasmInstr::I32LeU,
            (CmpOp::ULe, false, true) => WasmInstr::I64LeU,
            (CmpOp::OLe, true, false) => WasmInstr::F32Le,
            (CmpOp::OLe, true, true) => WasmInstr::F64Le,

            (CmpOp::SGt, false, false) => WasmInstr::I32GtS,
            (CmpOp::SGt, false, true) => WasmInstr::I64GtS,
            (CmpOp::UGt, false, false) => WasmInstr::I32GtU,
            (CmpOp::UGt, false, true) => WasmInstr::I64GtU,
            (CmpOp::OGt, true, false) => WasmInstr::F32Gt,
            (CmpOp::OGt, true, true) => WasmInstr::F64Gt,

            (CmpOp::SGe, false, false) => WasmInstr::I32GeS,
            (CmpOp::SGe, false, true) => WasmInstr::I64GeS,
            (CmpOp::UGe, false, false) => WasmInstr::I32GeU,
            (CmpOp::UGe, false, true) => WasmInstr::I64GeU,
            (CmpOp::OGe, true, false) => WasmInstr::F32Ge,
            (CmpOp::OGe, true, true) => WasmInstr::F64Ge,

            _ => {
                return Err(WasmError::NotSupported(format!(
                    "comparison op {:?} not supported",
                    op
                )));
            }
        };

        Ok(instr)
    }

    /// Get the WASM instruction for a type cast.
    fn cast_instr(&self, _val: &Value, target_ty: &LoopType) -> WasmResult<Option<WasmInstr>> {
        // Simplified cast handling - in practice would need source type info
        let instr = match target_ty {
            LoopType::Scalar(ScalarType::I32) => Some(WasmInstr::I32WrapI64),
            LoopType::Scalar(ScalarType::I64) => Some(WasmInstr::I64ExtendI32S),
            LoopType::Scalar(ScalarType::F32) => Some(WasmInstr::F32ConvertI32S),
            LoopType::Scalar(ScalarType::F64) => Some(WasmInstr::F64ConvertI32S),
            _ => None,
        };

        Ok(instr)
    }

    /// Convert Loop IR type to WASM type.
    fn lower_loop_type(&self, ty: &LoopType) -> WasmType {
        match ty {
            LoopType::Void => WasmType::I32, // Void represented as i32 in WASM
            LoopType::Scalar(ScalarType::Bool) => WasmType::I32,
            LoopType::Scalar(ScalarType::Int(bits) | ScalarType::UInt(bits)) => {
                if *bits <= 32 {
                    WasmType::I32
                } else {
                    WasmType::I64
                }
            }
            LoopType::Scalar(ScalarType::Float(bits)) => {
                if *bits <= 32 {
                    WasmType::F32
                } else {
                    WasmType::F64
                }
            }
            LoopType::Vector(_, _) => WasmType::V128,
            LoopType::Ptr(_) => WasmType::I32, // Pointers are i32 in WASM32
        }
    }

    /// Get or create a local for a value.
    fn get_or_create_local(&mut self, func: &mut WasmFunc, value_id: ValueId, op: &Op) -> u32 {
        if let Some(&local) = self.locals.get(&value_id) {
            return local;
        }

        // Determine type from the operation
        let wasm_ty = self.infer_op_type(op);
        let local_idx = func.add_local(wasm_ty);
        self.locals.insert(value_id, local_idx);
        local_idx
    }

    /// Get or create a local for a value ID with known type.
    fn get_or_create_local_for_id(
        &mut self,
        func: &mut WasmFunc,
        value_id: ValueId,
        ty: WasmType,
    ) -> u32 {
        if let Some(&local) = self.locals.get(&value_id) {
            return local;
        }

        let local_idx = func.add_local(ty);
        self.locals.insert(value_id, local_idx);
        local_idx
    }

    /// Infer the WASM type of an operation result.
    fn infer_op_type(&self, op: &Op) -> WasmType {
        match op {
            Op::Load(mem_ref) => self.lower_loop_type(&mem_ref.elem_ty),
            Op::Binary(_, _, _) => WasmType::I32, // Simplified
            Op::Cmp(_, _, _) => WasmType::I32,
            Op::Cast(_, target) => self.lower_loop_type(target),
            Op::Broadcast(_, _) => WasmType::V128,
            Op::Extract(_, _) => WasmType::F32,
            _ => WasmType::I32,
        }
    }

    /// Check if a value is a floating-point type.
    fn is_float_value(&self, value: &Value) -> bool {
        match value {
            Value::FloatConst(_, _) => true,
            Value::Var(_, ty) | Value::Undef(ty) => {
                matches!(ty, LoopType::Scalar(ScalarType::Float(_)))
            }
            _ => false,
        }
    }

    /// Check if a value is a 64-bit type.
    fn is_64bit_value(&self, value: &Value) -> bool {
        match value {
            Value::IntConst(_, ScalarType::Int(64) | ScalarType::UInt(64)) => true,
            Value::FloatConst(_, ScalarType::Float(64)) => true,
            Value::Var(_, ty) | Value::Undef(ty) => matches!(
                ty,
                LoopType::Scalar(
                    ScalarType::Int(64) | ScalarType::UInt(64) | ScalarType::Float(64)
                )
            ),
            _ => false,
        }
    }
}

/// Lower a Loop IR function to WASM.
///
/// This is the main entry point for Loop IR to WASM lowering.
pub fn lower_loop_ir(ir: &LoopIR, config: &WasmConfig) -> WasmResult<WasmFunc> {
    let mut lowerer = LoopIRLowerer::new(config.clone());
    lowerer.lower_function(ir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_loop_ir::Param;

    fn make_simple_loop_ir() -> LoopIR {
        LoopIR {
            name: Symbol::intern("test_func"),
            params: vec![Param {
                name: Symbol::intern("x"),
                ty: LoopType::Scalar(ScalarType::I32),
                is_ptr: false,
            }],
            return_ty: LoopType::Scalar(ScalarType::I32),
            allocs: vec![],
            body: Body {
                // Return the first parameter (parameter 0 is mapped to local 0)
                stmts: vec![Stmt::Return(Some(Value::Var(
                    ValueId::new(0),
                    LoopType::Scalar(ScalarType::I32),
                )))],
            },
            loop_info: vec![],
        }
    }

    #[test]
    fn test_lower_simple_function() {
        let ir = make_simple_loop_ir();
        let config = WasmConfig::default();
        let func = lower_loop_ir(&ir, &config).unwrap();

        assert_eq!(func.name.as_deref(), Some("test_func"));
        assert_eq!(func.ty.params.len(), 1);
        assert_eq!(func.ty.results.len(), 1);
    }
}
