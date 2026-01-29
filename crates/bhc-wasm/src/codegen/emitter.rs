//! WASM instruction emitter.
//!
//! Converts Loop IR operations to WASM instructions, handling both
//! scalar and SIMD operations.

use crate::codegen::types::{alignment_for_type, type_to_wasm, LoopTypeMapping};
use crate::{WasmError, WasmInstr, WasmResult, WasmType};
use bhc_index::Idx;
use bhc_loop_ir::{BinOp, CmpOp, LoopType, ScalarType, UnOp, Value};
use rustc_hash::FxHashMap;

/// WASM instruction emitter.
///
/// Translates Loop IR operations to sequences of WASM instructions.
pub struct WasmEmitter {
    /// Type mapping configuration.
    mapping: LoopTypeMapping,
    /// Generated instructions.
    instructions: Vec<WasmInstr>,
    /// Local variable mapping (ValueId -> local index).
    locals: FxHashMap<u32, u32>,
    /// Next local variable index.
    next_local: u32,
    /// Number of function parameters.
    param_count: u32,
}

impl WasmEmitter {
    /// Create a new emitter with the given type mapping.
    #[must_use]
    pub fn new(mapping: LoopTypeMapping, param_count: u32) -> Self {
        Self {
            mapping,
            instructions: Vec::new(),
            locals: FxHashMap::default(),
            next_local: param_count,
            param_count,
        }
    }

    /// Get the generated instructions.
    #[must_use]
    pub fn finish(self) -> Vec<WasmInstr> {
        self.instructions
    }

    /// Emit an instruction.
    pub fn emit(&mut self, instr: WasmInstr) {
        self.instructions.push(instr);
    }

    /// Emit multiple instructions.
    pub fn emit_all(&mut self, instrs: impl IntoIterator<Item = WasmInstr>) {
        self.instructions.extend(instrs);
    }

    /// Allocate a local variable for a value ID.
    pub fn alloc_local(&mut self, value_id: u32, _ty: WasmType) -> u32 {
        if let Some(&idx) = self.locals.get(&value_id) {
            return idx;
        }
        let idx = self.next_local;
        self.next_local += 1;
        self.locals.insert(value_id, idx);
        idx
    }

    /// Get the local index for a value ID.
    pub fn get_local(&self, value_id: u32) -> Option<u32> {
        self.locals.get(&value_id).copied()
    }

    /// Emit a value load (constant or variable reference).
    pub fn emit_value(&mut self, value: &Value) -> WasmResult<()> {
        match value {
            Value::Var(id, _ty) => {
                let idx = self.get_local(id.index() as u32).ok_or_else(|| {
                    WasmError::CodegenError(format!("Undefined variable: {:?}", id))
                })?;
                self.emit(WasmInstr::LocalGet(idx));
            }

            Value::IntConst(n, scalar) => match scalar {
                ScalarType::Int(bits) | ScalarType::UInt(bits) if *bits <= 32 => {
                    self.emit(WasmInstr::I32Const(*n as i32));
                }
                ScalarType::Int(64) | ScalarType::UInt(64) => {
                    self.emit(WasmInstr::I64Const(*n));
                }
                ScalarType::Bool => {
                    self.emit(WasmInstr::I32Const(if *n != 0 { 1 } else { 0 }));
                }
                _ => {
                    return Err(WasmError::NotSupported(format!(
                        "Integer constant of type {:?}",
                        scalar
                    )));
                }
            },

            Value::FloatConst(f, scalar) => match scalar {
                ScalarType::Float(32) => {
                    self.emit(WasmInstr::F32Const(*f as f32));
                }
                ScalarType::Float(64) => {
                    self.emit(WasmInstr::F64Const(*f));
                }
                _ => {
                    return Err(WasmError::NotSupported(format!(
                        "Float constant of type {:?}",
                        scalar
                    )));
                }
            },

            Value::BoolConst(b) => {
                self.emit(WasmInstr::I32Const(if *b { 1 } else { 0 }));
            }

            Value::Undef(ty) => {
                // Emit a zero value for undefined
                let wasm_ty = type_to_wasm(ty, &self.mapping)?;
                match wasm_ty {
                    WasmType::I32 => self.emit(WasmInstr::I32Const(0)),
                    WasmType::I64 => self.emit(WasmInstr::I64Const(0)),
                    WasmType::F32 => self.emit(WasmInstr::F32Const(0.0)),
                    WasmType::F64 => self.emit(WasmInstr::F64Const(0.0)),
                    WasmType::V128 => self.emit(WasmInstr::V128Const([0; 16])),
                    _ => {
                        return Err(WasmError::NotSupported(format!(
                            "Undefined value of type {:?}",
                            ty
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Emit a binary operation.
    pub fn emit_binary(&mut self, op: BinOp, ty: &LoopType) -> WasmResult<()> {
        let wasm_ty = type_to_wasm(ty, &self.mapping)?;

        match (op, wasm_ty) {
            // i32 operations
            (BinOp::Add, WasmType::I32) => self.emit(WasmInstr::I32Add),
            (BinOp::Sub, WasmType::I32) => self.emit(WasmInstr::I32Sub),
            (BinOp::Mul, WasmType::I32) => self.emit(WasmInstr::I32Mul),
            (BinOp::SDiv, WasmType::I32) => self.emit(WasmInstr::I32DivS),
            (BinOp::UDiv, WasmType::I32) => self.emit(WasmInstr::I32DivU),
            (BinOp::SRem, WasmType::I32) => self.emit(WasmInstr::I32RemS),
            (BinOp::URem, WasmType::I32) => self.emit(WasmInstr::I32RemU),
            (BinOp::And, WasmType::I32) => self.emit(WasmInstr::I32And),
            (BinOp::Or, WasmType::I32) => self.emit(WasmInstr::I32Or),
            (BinOp::Xor, WasmType::I32) => self.emit(WasmInstr::I32Xor),
            (BinOp::Shl, WasmType::I32) => self.emit(WasmInstr::I32Shl),
            (BinOp::LShr, WasmType::I32) => self.emit(WasmInstr::I32ShrU),
            (BinOp::AShr, WasmType::I32) => self.emit(WasmInstr::I32ShrS),

            // i64 operations
            (BinOp::Add, WasmType::I64) => self.emit(WasmInstr::I64Add),
            (BinOp::Sub, WasmType::I64) => self.emit(WasmInstr::I64Sub),
            (BinOp::Mul, WasmType::I64) => self.emit(WasmInstr::I64Mul),
            (BinOp::SDiv, WasmType::I64) => self.emit(WasmInstr::I64DivS),
            (BinOp::UDiv, WasmType::I64) => self.emit(WasmInstr::I64DivU),
            (BinOp::SRem, WasmType::I64) => self.emit(WasmInstr::I64RemS),
            (BinOp::URem, WasmType::I64) => self.emit(WasmInstr::I64RemU),
            (BinOp::And, WasmType::I64) => self.emit(WasmInstr::I64And),
            (BinOp::Or, WasmType::I64) => self.emit(WasmInstr::I64Or),
            (BinOp::Xor, WasmType::I64) => self.emit(WasmInstr::I64Xor),
            (BinOp::Shl, WasmType::I64) => self.emit(WasmInstr::I64Shl),
            (BinOp::LShr, WasmType::I64) => self.emit(WasmInstr::I64ShrU),
            (BinOp::AShr, WasmType::I64) => self.emit(WasmInstr::I64ShrS),

            // f32 operations
            (BinOp::Add, WasmType::F32) => self.emit(WasmInstr::F32Add),
            (BinOp::Sub, WasmType::F32) => self.emit(WasmInstr::F32Sub),
            (BinOp::Mul, WasmType::F32) => self.emit(WasmInstr::F32Mul),
            (BinOp::FDiv, WasmType::F32) => self.emit(WasmInstr::F32Div),
            (BinOp::FMin, WasmType::F32) => self.emit(WasmInstr::F32Min),
            (BinOp::FMax, WasmType::F32) => self.emit(WasmInstr::F32Max),

            // f64 operations
            (BinOp::Add, WasmType::F64) => self.emit(WasmInstr::F64Add),
            (BinOp::Sub, WasmType::F64) => self.emit(WasmInstr::F64Sub),
            (BinOp::Mul, WasmType::F64) => self.emit(WasmInstr::F64Mul),
            (BinOp::FDiv, WasmType::F64) => self.emit(WasmInstr::F64Div),
            (BinOp::FMin, WasmType::F64) => self.emit(WasmInstr::F64Min),
            (BinOp::FMax, WasmType::F64) => self.emit(WasmInstr::F64Max),

            // SIMD f32x4 operations
            (BinOp::Add, WasmType::V128) if is_f32x4(ty) => self.emit(WasmInstr::F32x4Add),
            (BinOp::Sub, WasmType::V128) if is_f32x4(ty) => self.emit(WasmInstr::F32x4Sub),
            (BinOp::Mul, WasmType::V128) if is_f32x4(ty) => self.emit(WasmInstr::F32x4Mul),
            (BinOp::FDiv, WasmType::V128) if is_f32x4(ty) => self.emit(WasmInstr::F32x4Div),
            (BinOp::FMin, WasmType::V128) if is_f32x4(ty) => self.emit(WasmInstr::F32x4Min),
            (BinOp::FMax, WasmType::V128) if is_f32x4(ty) => self.emit(WasmInstr::F32x4Max),

            // SIMD f64x2 operations
            (BinOp::Add, WasmType::V128) if is_f64x2(ty) => self.emit(WasmInstr::F64x2Add),
            (BinOp::Sub, WasmType::V128) if is_f64x2(ty) => self.emit(WasmInstr::F64x2Sub),
            (BinOp::Mul, WasmType::V128) if is_f64x2(ty) => self.emit(WasmInstr::F64x2Mul),
            (BinOp::FDiv, WasmType::V128) if is_f64x2(ty) => self.emit(WasmInstr::F64x2Div),
            (BinOp::FMin, WasmType::V128) if is_f64x2(ty) => self.emit(WasmInstr::F64x2Min),
            (BinOp::FMax, WasmType::V128) if is_f64x2(ty) => self.emit(WasmInstr::F64x2Max),

            // SIMD i32x4 operations
            (BinOp::Add, WasmType::V128) if is_i32x4(ty) => self.emit(WasmInstr::I32x4Add),
            (BinOp::Sub, WasmType::V128) if is_i32x4(ty) => self.emit(WasmInstr::I32x4Sub),
            (BinOp::Mul, WasmType::V128) if is_i32x4(ty) => self.emit(WasmInstr::I32x4Mul),
            (BinOp::Shl, WasmType::V128) if is_i32x4(ty) => self.emit(WasmInstr::I32x4Shl),
            (BinOp::AShr, WasmType::V128) if is_i32x4(ty) => self.emit(WasmInstr::I32x4ShrS),
            (BinOp::LShr, WasmType::V128) if is_i32x4(ty) => self.emit(WasmInstr::I32x4ShrU),

            // SIMD bitwise (works for all v128)
            (BinOp::And, WasmType::V128) => self.emit(WasmInstr::V128And),
            (BinOp::Or, WasmType::V128) => self.emit(WasmInstr::V128Or),
            (BinOp::Xor, WasmType::V128) => self.emit(WasmInstr::V128Xor),

            _ => {
                return Err(WasmError::NotSupported(format!(
                    "Binary operation {:?} on type {:?}",
                    op, ty
                )));
            }
        }
        Ok(())
    }

    /// Emit a unary operation.
    pub fn emit_unary(&mut self, op: UnOp, ty: &LoopType) -> WasmResult<()> {
        let wasm_ty = type_to_wasm(ty, &self.mapping)?;

        match (op, wasm_ty) {
            // f32 operations
            (UnOp::FNeg, WasmType::F32) | (UnOp::Neg, WasmType::F32) => {
                self.emit(WasmInstr::F32Neg);
            }
            (UnOp::FAbs, WasmType::F32) | (UnOp::Abs, WasmType::F32) => {
                self.emit(WasmInstr::F32Abs);
            }
            (UnOp::Sqrt, WasmType::F32) => self.emit(WasmInstr::F32Sqrt),
            (UnOp::Ceil, WasmType::F32) => self.emit(WasmInstr::F32Ceil),
            (UnOp::Floor, WasmType::F32) => self.emit(WasmInstr::F32Floor),
            (UnOp::Trunc, WasmType::F32) => self.emit(WasmInstr::F32Trunc),
            (UnOp::Round, WasmType::F32) => self.emit(WasmInstr::F32Nearest),

            // f64 operations
            (UnOp::FNeg, WasmType::F64) | (UnOp::Neg, WasmType::F64) => {
                self.emit(WasmInstr::F64Neg);
            }
            (UnOp::FAbs, WasmType::F64) | (UnOp::Abs, WasmType::F64) => {
                self.emit(WasmInstr::F64Abs);
            }
            (UnOp::Sqrt, WasmType::F64) => self.emit(WasmInstr::F64Sqrt),
            (UnOp::Ceil, WasmType::F64) => self.emit(WasmInstr::F64Ceil),
            (UnOp::Floor, WasmType::F64) => self.emit(WasmInstr::F64Floor),
            (UnOp::Trunc, WasmType::F64) => self.emit(WasmInstr::F64Trunc),
            (UnOp::Round, WasmType::F64) => self.emit(WasmInstr::F64Nearest),

            // i32 negation: 0 - x
            (UnOp::Neg, WasmType::I32) => {
                // Stack: [x] -> [0, x] -> [-x]
                // Save x, push 0, push x, subtract
                let temp = self.next_local;
                self.next_local += 1;
                self.emit(WasmInstr::LocalSet(temp));
                self.emit(WasmInstr::I32Const(0));
                self.emit(WasmInstr::LocalGet(temp));
                self.emit(WasmInstr::I32Sub);
            }

            // i64 negation: 0 - x
            (UnOp::Neg, WasmType::I64) => {
                let temp = self.next_local;
                self.next_local += 1;
                self.emit(WasmInstr::LocalSet(temp));
                self.emit(WasmInstr::I64Const(0));
                self.emit(WasmInstr::LocalGet(temp));
                self.emit(WasmInstr::I64Sub);
            }

            // NOT for integers: XOR with -1
            (UnOp::Not, WasmType::I32) => {
                self.emit(WasmInstr::I32Const(-1));
                self.emit(WasmInstr::I32Xor);
            }
            (UnOp::Not, WasmType::I64) => {
                self.emit(WasmInstr::I64Const(-1));
                self.emit(WasmInstr::I64Xor);
            }

            // SIMD f32x4 unary
            (UnOp::FNeg, WasmType::V128) | (UnOp::Neg, WasmType::V128) if is_f32x4(ty) => {
                self.emit(WasmInstr::F32x4Neg);
            }
            (UnOp::FAbs, WasmType::V128) | (UnOp::Abs, WasmType::V128) if is_f32x4(ty) => {
                self.emit(WasmInstr::F32x4Abs);
            }
            (UnOp::Sqrt, WasmType::V128) if is_f32x4(ty) => {
                self.emit(WasmInstr::F32x4Sqrt);
            }
            (UnOp::Ceil, WasmType::V128) if is_f32x4(ty) => {
                self.emit(WasmInstr::F32x4Ceil);
            }
            (UnOp::Floor, WasmType::V128) if is_f32x4(ty) => {
                self.emit(WasmInstr::F32x4Floor);
            }

            // SIMD f64x2 unary
            (UnOp::FNeg, WasmType::V128) | (UnOp::Neg, WasmType::V128) if is_f64x2(ty) => {
                self.emit(WasmInstr::F64x2Neg);
            }
            (UnOp::FAbs, WasmType::V128) | (UnOp::Abs, WasmType::V128) if is_f64x2(ty) => {
                self.emit(WasmInstr::F64x2Abs);
            }
            (UnOp::Sqrt, WasmType::V128) if is_f64x2(ty) => {
                self.emit(WasmInstr::F64x2Sqrt);
            }

            // SIMD i32x4 negation
            (UnOp::Neg, WasmType::V128) if is_i32x4(ty) => {
                self.emit(WasmInstr::I32x4Neg);
            }

            // SIMD bitwise NOT
            (UnOp::Not, WasmType::V128) => {
                self.emit(WasmInstr::V128Not);
            }

            _ => {
                return Err(WasmError::NotSupported(format!(
                    "Unary operation {:?} on type {:?}",
                    op, ty
                )));
            }
        }
        Ok(())
    }

    /// Emit a comparison operation.
    pub fn emit_comparison(&mut self, op: CmpOp, ty: &LoopType) -> WasmResult<()> {
        let wasm_ty = type_to_wasm(ty, &self.mapping)?;

        match (op, wasm_ty) {
            // i32 comparisons
            (CmpOp::Eq, WasmType::I32) => self.emit(WasmInstr::I32Eq),
            (CmpOp::Ne, WasmType::I32) => self.emit(WasmInstr::I32Ne),
            (CmpOp::SLt, WasmType::I32) => self.emit(WasmInstr::I32LtS),
            (CmpOp::SLe, WasmType::I32) => self.emit(WasmInstr::I32LeS),
            (CmpOp::SGt, WasmType::I32) => self.emit(WasmInstr::I32GtS),
            (CmpOp::SGe, WasmType::I32) => self.emit(WasmInstr::I32GeS),
            (CmpOp::ULt, WasmType::I32) => self.emit(WasmInstr::I32LtU),
            (CmpOp::ULe, WasmType::I32) => self.emit(WasmInstr::I32LeU),
            (CmpOp::UGt, WasmType::I32) => self.emit(WasmInstr::I32GtU),
            (CmpOp::UGe, WasmType::I32) => self.emit(WasmInstr::I32GeU),

            // i64 comparisons
            (CmpOp::Eq, WasmType::I64) => self.emit(WasmInstr::I64Eq),
            (CmpOp::Ne, WasmType::I64) => self.emit(WasmInstr::I64Ne),
            (CmpOp::SLt, WasmType::I64) => self.emit(WasmInstr::I64LtS),
            (CmpOp::SLe, WasmType::I64) => self.emit(WasmInstr::I64LeS),
            (CmpOp::SGt, WasmType::I64) => self.emit(WasmInstr::I64GtS),
            (CmpOp::SGe, WasmType::I64) => self.emit(WasmInstr::I64GeS),
            (CmpOp::ULt, WasmType::I64) => self.emit(WasmInstr::I64LtU),
            (CmpOp::ULe, WasmType::I64) => self.emit(WasmInstr::I64LeU),
            (CmpOp::UGt, WasmType::I64) => self.emit(WasmInstr::I64GtU),
            (CmpOp::UGe, WasmType::I64) => self.emit(WasmInstr::I64GeU),

            // f32 comparisons (ordered)
            (CmpOp::OEq, WasmType::F32) | (CmpOp::Eq, WasmType::F32) => {
                self.emit(WasmInstr::F32Eq);
            }
            (CmpOp::ONe, WasmType::F32) | (CmpOp::Ne, WasmType::F32) => {
                self.emit(WasmInstr::F32Ne);
            }
            (CmpOp::OLt, WasmType::F32) => self.emit(WasmInstr::F32Lt),
            (CmpOp::OLe, WasmType::F32) => self.emit(WasmInstr::F32Le),
            (CmpOp::OGt, WasmType::F32) => self.emit(WasmInstr::F32Gt),
            (CmpOp::OGe, WasmType::F32) => self.emit(WasmInstr::F32Ge),

            // f64 comparisons (ordered)
            (CmpOp::OEq, WasmType::F64) | (CmpOp::Eq, WasmType::F64) => {
                self.emit(WasmInstr::F64Eq);
            }
            (CmpOp::ONe, WasmType::F64) | (CmpOp::Ne, WasmType::F64) => {
                self.emit(WasmInstr::F64Ne);
            }
            (CmpOp::OLt, WasmType::F64) => self.emit(WasmInstr::F64Lt),
            (CmpOp::OLe, WasmType::F64) => self.emit(WasmInstr::F64Le),
            (CmpOp::OGt, WasmType::F64) => self.emit(WasmInstr::F64Gt),
            (CmpOp::OGe, WasmType::F64) => self.emit(WasmInstr::F64Ge),

            _ => {
                return Err(WasmError::NotSupported(format!(
                    "Comparison {:?} on type {:?}",
                    op, ty
                )));
            }
        }
        Ok(())
    }

    /// Emit a load from memory.
    pub fn emit_load(&mut self, ty: &LoopType, offset: u32) -> WasmResult<()> {
        let wasm_ty = type_to_wasm(ty, &self.mapping)?;
        let align = alignment_for_type(ty);

        match wasm_ty {
            WasmType::I32 => self.emit(WasmInstr::I32Load(align, offset)),
            WasmType::I64 => self.emit(WasmInstr::I64Load(align, offset)),
            WasmType::F32 => self.emit(WasmInstr::F32Load(align, offset)),
            WasmType::F64 => self.emit(WasmInstr::F64Load(align, offset)),
            WasmType::V128 => self.emit(WasmInstr::V128Load(align, offset)),
            _ => {
                return Err(WasmError::NotSupported(format!("Load of type {:?}", ty)));
            }
        }
        Ok(())
    }

    /// Emit a store to memory.
    pub fn emit_store(&mut self, ty: &LoopType, offset: u32) -> WasmResult<()> {
        let wasm_ty = type_to_wasm(ty, &self.mapping)?;
        let align = alignment_for_type(ty);

        match wasm_ty {
            WasmType::I32 => self.emit(WasmInstr::I32Store(align, offset)),
            WasmType::I64 => self.emit(WasmInstr::I64Store(align, offset)),
            WasmType::F32 => self.emit(WasmInstr::F32Store(align, offset)),
            WasmType::F64 => self.emit(WasmInstr::F64Store(align, offset)),
            WasmType::V128 => self.emit(WasmInstr::V128Store(align, offset)),
            _ => {
                return Err(WasmError::NotSupported(format!("Store of type {:?}", ty)));
            }
        }
        Ok(())
    }

    /// Emit a broadcast (splat) operation for SIMD.
    pub fn emit_broadcast(&mut self, scalar_ty: ScalarType, width: u8) -> WasmResult<()> {
        if !self.mapping.simd_enabled {
            return Err(WasmError::SimdNotAvailable(
                "SIMD broadcast not available".to_string(),
            ));
        }

        // The scalar value is already on the stack
        match (scalar_ty, width) {
            (ScalarType::Float(32), 4) => self.emit(WasmInstr::F32x4Splat),
            (ScalarType::Float(64), 2) => self.emit(WasmInstr::F64x2Splat),
            (ScalarType::Int(32), 4) | (ScalarType::UInt(32), 4) => {
                self.emit(WasmInstr::I32x4Splat);
            }
            (ScalarType::Int(64), 2) | (ScalarType::UInt(64), 2) => {
                self.emit(WasmInstr::I64x2Splat);
            }
            (ScalarType::Int(16), 8) | (ScalarType::UInt(16), 8) => {
                self.emit(WasmInstr::I16x8Splat);
            }
            (ScalarType::Int(8), 16) | (ScalarType::UInt(8), 16) | (ScalarType::Bool, 16) => {
                self.emit(WasmInstr::I8x16Splat);
            }
            _ => {
                return Err(WasmError::NotSupported(format!(
                    "Broadcast {:?}x{} not supported",
                    scalar_ty, width
                )));
            }
        }
        Ok(())
    }

    /// Emit an extract lane operation for SIMD.
    pub fn emit_extract(&mut self, scalar_ty: ScalarType, lane: u8) -> WasmResult<()> {
        if !self.mapping.simd_enabled {
            return Err(WasmError::SimdNotAvailable(
                "SIMD extract not available".to_string(),
            ));
        }

        match scalar_ty {
            ScalarType::Float(32) => self.emit(WasmInstr::F32x4ExtractLane(lane)),
            ScalarType::Float(64) => self.emit(WasmInstr::F64x2ExtractLane(lane)),
            ScalarType::Int(32) | ScalarType::UInt(32) => {
                self.emit(WasmInstr::I32x4ExtractLane(lane));
            }
            ScalarType::Int(64) | ScalarType::UInt(64) => {
                self.emit(WasmInstr::I64x2ExtractLane(lane));
            }
            ScalarType::Int(8) => self.emit(WasmInstr::I8x16ExtractLaneS(lane)),
            ScalarType::UInt(8) | ScalarType::Bool => {
                self.emit(WasmInstr::I8x16ExtractLaneU(lane));
            }
            ScalarType::Int(16) => self.emit(WasmInstr::I16x8ExtractLaneS(lane)),
            ScalarType::UInt(16) => self.emit(WasmInstr::I16x8ExtractLaneU(lane)),
            _ => {
                return Err(WasmError::NotSupported(format!(
                    "Extract lane for {:?} not supported",
                    scalar_ty
                )));
            }
        }
        Ok(())
    }

    /// Emit a FMA operation (a * b + c).
    ///
    /// WASM doesn't have native FMA, so we lower to mul + add.
    /// Note: This may have different precision than true FMA.
    pub fn emit_fma(&mut self, ty: &LoopType) -> WasmResult<()> {
        let wasm_ty = type_to_wasm(ty, &self.mapping)?;

        // Stack: [a, b, c]
        // We need: a * b + c

        match wasm_ty {
            WasmType::F32 => {
                // Save c, compute a * b, then add c
                let temp = self.next_local;
                self.next_local += 1;
                self.emit(WasmInstr::LocalSet(temp)); // save c
                self.emit(WasmInstr::F32Mul); // a * b
                self.emit(WasmInstr::LocalGet(temp)); // get c
                self.emit(WasmInstr::F32Add); // (a * b) + c
            }
            WasmType::F64 => {
                let temp = self.next_local;
                self.next_local += 1;
                self.emit(WasmInstr::LocalSet(temp));
                self.emit(WasmInstr::F64Mul);
                self.emit(WasmInstr::LocalGet(temp));
                self.emit(WasmInstr::F64Add);
            }
            WasmType::V128 if is_f32x4(ty) => {
                let temp = self.next_local;
                self.next_local += 1;
                self.emit(WasmInstr::LocalSet(temp));
                self.emit(WasmInstr::F32x4Mul);
                self.emit(WasmInstr::LocalGet(temp));
                self.emit(WasmInstr::F32x4Add);
            }
            WasmType::V128 if is_f64x2(ty) => {
                let temp = self.next_local;
                self.next_local += 1;
                self.emit(WasmInstr::LocalSet(temp));
                self.emit(WasmInstr::F64x2Mul);
                self.emit(WasmInstr::LocalGet(temp));
                self.emit(WasmInstr::F64x2Add);
            }
            _ => {
                return Err(WasmError::NotSupported(format!("FMA on type {:?}", ty)));
            }
        }
        Ok(())
    }

    /// Emit a type conversion.
    pub fn emit_cast(&mut self, from: &LoopType, to: &LoopType) -> WasmResult<()> {
        let from_wasm = type_to_wasm(from, &self.mapping)?;
        let to_wasm = type_to_wasm(to, &self.mapping)?;

        if from_wasm == to_wasm {
            return Ok(());
        }

        match (from_wasm, to_wasm) {
            // Integer widening
            (WasmType::I32, WasmType::I64) => {
                if is_signed(from) {
                    self.emit(WasmInstr::I64ExtendI32S);
                } else {
                    self.emit(WasmInstr::I64ExtendI32U);
                }
            }

            // Integer narrowing
            (WasmType::I64, WasmType::I32) => {
                self.emit(WasmInstr::I32WrapI64);
            }

            // Float widening
            (WasmType::F32, WasmType::F64) => {
                self.emit(WasmInstr::F64PromoteF32);
            }

            // Float narrowing
            (WasmType::F64, WasmType::F32) => {
                self.emit(WasmInstr::F32DemoteF64);
            }

            // Int to float
            (WasmType::I32, WasmType::F32) => {
                if is_signed(from) {
                    self.emit(WasmInstr::F32ConvertI32S);
                } else {
                    self.emit(WasmInstr::F32ConvertI32U);
                }
            }
            (WasmType::I32, WasmType::F64) => {
                if is_signed(from) {
                    self.emit(WasmInstr::F64ConvertI32S);
                } else {
                    self.emit(WasmInstr::F64ConvertI32U);
                }
            }
            (WasmType::I64, WasmType::F32) => {
                if is_signed(from) {
                    self.emit(WasmInstr::F32ConvertI64S);
                } else {
                    self.emit(WasmInstr::F32ConvertI64U);
                }
            }
            (WasmType::I64, WasmType::F64) => {
                if is_signed(from) {
                    self.emit(WasmInstr::F64ConvertI64S);
                } else {
                    self.emit(WasmInstr::F64ConvertI64U);
                }
            }

            // Float to int
            (WasmType::F32, WasmType::I32) => {
                if is_signed(to) {
                    self.emit(WasmInstr::I32TruncF32S);
                } else {
                    self.emit(WasmInstr::I32TruncF32U);
                }
            }
            (WasmType::F64, WasmType::I32) => {
                if is_signed(to) {
                    self.emit(WasmInstr::I32TruncF64S);
                } else {
                    self.emit(WasmInstr::I32TruncF64U);
                }
            }
            (WasmType::F32, WasmType::I64) => {
                if is_signed(to) {
                    self.emit(WasmInstr::I64TruncF32S);
                } else {
                    self.emit(WasmInstr::I64TruncF32U);
                }
            }
            (WasmType::F64, WasmType::I64) => {
                if is_signed(to) {
                    self.emit(WasmInstr::I64TruncF64S);
                } else {
                    self.emit(WasmInstr::I64TruncF64U);
                }
            }

            _ => {
                return Err(WasmError::NotSupported(format!(
                    "Cast from {:?} to {:?}",
                    from, to
                )));
            }
        }
        Ok(())
    }
}

/// Check if a type is f32x4 vector.
fn is_f32x4(ty: &LoopType) -> bool {
    matches!(ty, LoopType::Vector(ScalarType::Float(32), 4))
}

/// Check if a type is f64x2 vector.
fn is_f64x2(ty: &LoopType) -> bool {
    matches!(ty, LoopType::Vector(ScalarType::Float(64), 2))
}

/// Check if a type is i32x4 vector.
fn is_i32x4(ty: &LoopType) -> bool {
    matches!(
        ty,
        LoopType::Vector(ScalarType::Int(32), 4) | LoopType::Vector(ScalarType::UInt(32), 4)
    )
}

/// Check if a type is signed.
fn is_signed(ty: &LoopType) -> bool {
    match ty {
        LoopType::Scalar(ScalarType::Int(_)) => true,
        _ => false,
    }
}

/// Check if this is a reinterpret cast (not a conversion).
fn is_reinterpret(_from: &LoopType, _to: &LoopType) -> bool {
    // In the current implementation, we don't track this
    // A proper implementation would check if the cast is marked as reinterpret
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_loop_ir::ValueId;

    #[test]
    fn test_emit_int_constants() {
        let mapping = LoopTypeMapping::default();
        let mut emitter = WasmEmitter::new(mapping, 0);

        emitter
            .emit_value(&Value::IntConst(42, ScalarType::Int(32)))
            .unwrap();
        let instrs = emitter.finish();

        assert_eq!(instrs.len(), 1);
        assert!(matches!(instrs[0], WasmInstr::I32Const(42)));
    }

    #[test]
    fn test_emit_float_constants() {
        let mapping = LoopTypeMapping::default();
        let mut emitter = WasmEmitter::new(mapping, 0);

        emitter
            .emit_value(&Value::FloatConst(3.14, ScalarType::Float(64)))
            .unwrap();
        let instrs = emitter.finish();

        assert_eq!(instrs.len(), 1);
        assert!(matches!(instrs[0], WasmInstr::F64Const(f) if (f - 3.14).abs() < 0.001));
    }

    #[test]
    fn test_emit_binary_add() {
        let mapping = LoopTypeMapping::default();
        let mut emitter = WasmEmitter::new(mapping, 0);

        let ty = LoopType::Scalar(ScalarType::Float(32));
        emitter.emit_binary(BinOp::Add, &ty).unwrap();

        let instrs = emitter.finish();
        assert!(matches!(instrs[0], WasmInstr::F32Add));
    }

    #[test]
    fn test_emit_simd_add() {
        let mapping = LoopTypeMapping::default();
        let mut emitter = WasmEmitter::new(mapping, 0);

        let ty = LoopType::Vector(ScalarType::Float(32), 4);
        emitter.emit_binary(BinOp::Add, &ty).unwrap();

        let instrs = emitter.finish();
        assert!(matches!(instrs[0], WasmInstr::F32x4Add));
    }

    #[test]
    fn test_emit_broadcast() {
        let mapping = LoopTypeMapping::default();
        let mut emitter = WasmEmitter::new(mapping, 0);

        emitter.emit_broadcast(ScalarType::Float(32), 4).unwrap();
        let instrs = emitter.finish();
        assert!(matches!(instrs[0], WasmInstr::F32x4Splat));
    }

    #[test]
    fn test_emit_fma() {
        let mapping = LoopTypeMapping::default();
        let mut emitter = WasmEmitter::new(mapping, 0);

        let ty = LoopType::Scalar(ScalarType::Float(32));
        emitter.emit_fma(&ty).unwrap();

        let instrs = emitter.finish();
        // FMA lowers to: local.set, f32.mul, local.get, f32.add
        assert_eq!(instrs.len(), 4);
        assert!(matches!(instrs[1], WasmInstr::F32Mul));
        assert!(matches!(instrs[3], WasmInstr::F32Add));
    }

    #[test]
    fn test_emit_comparison() {
        let mapping = LoopTypeMapping::default();
        let mut emitter = WasmEmitter::new(mapping, 0);

        let ty = LoopType::Scalar(ScalarType::Int(32));
        emitter.emit_comparison(CmpOp::SLt, &ty).unwrap();

        let instrs = emitter.finish();
        assert!(matches!(instrs[0], WasmInstr::I32LtS));
    }
}
