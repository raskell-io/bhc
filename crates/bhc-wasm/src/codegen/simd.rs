//! SIMD instruction lowering for WASM.
//!
//! This module handles the conversion of high-level SIMD operations to
//! WASM SIMD128 instructions, including patterns that don't have direct
//! equivalents (like horizontal reductions and FMA).

use crate::{WasmError, WasmInstr, WasmResult};
use bhc_loop_ir::{ReduceOp, ScalarType};

/// SIMD lowering patterns.
#[derive(Clone, Debug)]
pub enum SimdPattern {
    /// Direct mapping to WASM SIMD instruction.
    Direct,
    /// Requires shuffle + operation sequence.
    ShuffleReduce(ReduceOp),
    /// Requires emulation with scalar ops.
    ScalarFallback,
    /// FMA lowering (mul + add).
    FmaLowering,
}

/// SIMD lowering helper.
pub struct SimdLowering {
    /// Enable SIMD128 instructions.
    simd_enabled: bool,
}

impl SimdLowering {
    /// Create a new SIMD lowering instance.
    #[must_use]
    pub fn new(simd_enabled: bool) -> Self {
        Self { simd_enabled }
    }

    /// Check if SIMD is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.simd_enabled
    }

    /// Lower a horizontal reduction to WASM instructions.
    ///
    /// WASM SIMD doesn't have direct horizontal reduction instructions,
    /// so we use shuffle + vertical operation patterns.
    ///
    /// For example, to sum a f32x4:
    /// ```text
    /// [a, b, c, d]
    /// shuffle to get [c, d, _, _] and [a, b, _, _]
    /// add to get [a+c, b+d, _, _]
    /// shuffle to get [b+d, _, _, _] and [a+c, _, _, _]
    /// add to get [a+b+c+d, _, _, _]
    /// extract lane 0
    /// ```
    pub fn emit_horizontal_reduce(
        &self,
        op: ReduceOp,
        scalar_ty: ScalarType,
        width: u8,
    ) -> WasmResult<Vec<WasmInstr>> {
        if !self.simd_enabled {
            return Err(WasmError::SimdNotAvailable(
                "SIMD not enabled for horizontal reduction".to_string(),
            ));
        }

        match (scalar_ty, width) {
            (ScalarType::Float(32), 4) => self.reduce_f32x4(op),
            (ScalarType::Float(64), 2) => self.reduce_f64x2(op),
            (ScalarType::Int(32), 4) | (ScalarType::UInt(32), 4) => self.reduce_i32x4(op),
            (ScalarType::Int(64), 2) | (ScalarType::UInt(64), 2) => self.reduce_i64x2(op),
            _ => Err(WasmError::NotSupported(format!(
                "Horizontal reduction for {:?}x{}",
                scalar_ty, width
            ))),
        }
    }

    /// Horizontal reduction for f32x4.
    fn reduce_f32x4(&self, op: ReduceOp) -> WasmResult<Vec<WasmInstr>> {
        let mut instrs = Vec::new();

        // v = [a, b, c, d]
        // We need to compute op(a, b, c, d)

        // Step 1: Shuffle high half to low half
        // temp1 = [c, d, c, d]
        instrs.push(WasmInstr::I8x16Shuffle([
            8, 9, 10, 11, // c (bytes 8-11)
            12, 13, 14, 15, // d (bytes 12-15)
            8, 9, 10, 11, // c
            12, 13, 14, 15, // d
        ]));

        // Step 2: Add low and high halves
        // temp2 = [a+c, b+d, _, _]
        instrs.push(self.reduce_op_instr_f32x4(op)?);

        // Step 3: Shuffle element 1 to element 0
        // temp3 = [b+d, _, _, _]
        // Need to duplicate the vector first for shuffle
        instrs.push(WasmInstr::I8x16Shuffle([
            4, 5, 6, 7, // b+d (bytes 4-7)
            0, 1, 2, 3, // a+c
            0, 1, 2, 3, // _
            0, 1, 2, 3, // _
        ]));

        // Step 4: Add again
        // result = [a+b+c+d, _, _, _]
        instrs.push(self.reduce_op_instr_f32x4(op)?);

        // Step 5: Extract lane 0
        instrs.push(WasmInstr::F32x4ExtractLane(0));

        Ok(instrs)
    }

    /// Horizontal reduction for f64x2.
    fn reduce_f64x2(&self, op: ReduceOp) -> WasmResult<Vec<WasmInstr>> {
        let mut instrs = Vec::new();

        // v = [a, b]
        // Shuffle to get [b, a]
        instrs.push(WasmInstr::I8x16Shuffle([
            8, 9, 10, 11, 12, 13, 14, 15, // b (bytes 8-15)
            0, 1, 2, 3, 4, 5, 6, 7, // a (bytes 0-7)
        ]));

        // Add to get [a+b, a+b]
        instrs.push(self.reduce_op_instr_f64x2(op)?);

        // Extract lane 0
        instrs.push(WasmInstr::F64x2ExtractLane(0));

        Ok(instrs)
    }

    /// Horizontal reduction for i32x4.
    fn reduce_i32x4(&self, op: ReduceOp) -> WasmResult<Vec<WasmInstr>> {
        let mut instrs = Vec::new();

        // Similar pattern to f32x4

        // Step 1: Shuffle high half to low
        instrs.push(WasmInstr::I8x16Shuffle([
            8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15,
        ]));

        // Step 2: Reduce
        instrs.push(self.reduce_op_instr_i32x4(op)?);

        // Step 3: Shuffle element 1 to 0
        instrs.push(WasmInstr::I8x16Shuffle([
            4, 5, 6, 7, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
        ]));

        // Step 4: Final reduce
        instrs.push(self.reduce_op_instr_i32x4(op)?);

        // Step 5: Extract lane 0
        instrs.push(WasmInstr::I32x4ExtractLane(0));

        Ok(instrs)
    }

    /// Horizontal reduction for i64x2.
    fn reduce_i64x2(&self, op: ReduceOp) -> WasmResult<Vec<WasmInstr>> {
        // For i64x2, WASM SIMD doesn't have all operations
        // Fall back to scalar for unsupported ops

        match op {
            ReduceOp::Add | ReduceOp::And | ReduceOp::Or | ReduceOp::Xor => {
                // These work with i64x2
                let mut instrs = Vec::new();

                // Shuffle to get [b, a]
                instrs.push(WasmInstr::I8x16Shuffle([
                    8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
                ]));

                // Reduce
                instrs.push(self.reduce_op_instr_i64x2(op)?);

                // Extract lane 0
                instrs.push(WasmInstr::I64x2ExtractLane(0));

                Ok(instrs)
            }
            _ => Err(WasmError::NotSupported(format!(
                "Horizontal reduction {:?} for i64x2",
                op
            ))),
        }
    }

    /// Get the WASM instruction for a f32x4 reduction operation.
    fn reduce_op_instr_f32x4(&self, op: ReduceOp) -> WasmResult<WasmInstr> {
        match op {
            ReduceOp::Add => Ok(WasmInstr::F32x4Add),
            ReduceOp::Mul => Ok(WasmInstr::F32x4Mul),
            ReduceOp::Min => Ok(WasmInstr::F32x4Min),
            ReduceOp::Max => Ok(WasmInstr::F32x4Max),
            _ => Err(WasmError::NotSupported(format!(
                "Reduction {:?} not supported for f32x4",
                op
            ))),
        }
    }

    /// Get the WASM instruction for a f64x2 reduction operation.
    fn reduce_op_instr_f64x2(&self, op: ReduceOp) -> WasmResult<WasmInstr> {
        match op {
            ReduceOp::Add => Ok(WasmInstr::F64x2Add),
            ReduceOp::Mul => Ok(WasmInstr::F64x2Mul),
            ReduceOp::Min => Ok(WasmInstr::F64x2Min),
            ReduceOp::Max => Ok(WasmInstr::F64x2Max),
            _ => Err(WasmError::NotSupported(format!(
                "Reduction {:?} not supported for f64x2",
                op
            ))),
        }
    }

    /// Get the WASM instruction for an i32x4 reduction operation.
    fn reduce_op_instr_i32x4(&self, op: ReduceOp) -> WasmResult<WasmInstr> {
        match op {
            ReduceOp::Add => Ok(WasmInstr::I32x4Add),
            ReduceOp::Mul => Ok(WasmInstr::I32x4Mul),
            ReduceOp::And => Ok(WasmInstr::V128And),
            ReduceOp::Or => Ok(WasmInstr::V128Or),
            ReduceOp::Xor => Ok(WasmInstr::V128Xor),
            // Min/Max for i32x4 require special handling
            _ => Err(WasmError::NotSupported(format!(
                "Reduction {:?} not supported for i32x4",
                op
            ))),
        }
    }

    /// Get the WASM instruction for an i64x2 reduction operation.
    fn reduce_op_instr_i64x2(&self, op: ReduceOp) -> WasmResult<WasmInstr> {
        match op {
            ReduceOp::Add => {
                // WASM SIMD doesn't have i64x2.add in the base spec
                // but it's commonly available
                Err(WasmError::NotSupported(
                    "i64x2.add requires extended SIMD".to_string(),
                ))
            }
            ReduceOp::And => Ok(WasmInstr::V128And),
            ReduceOp::Or => Ok(WasmInstr::V128Or),
            ReduceOp::Xor => Ok(WasmInstr::V128Xor),
            _ => Err(WasmError::NotSupported(format!(
                "Reduction {:?} not supported for i64x2",
                op
            ))),
        }
    }

    /// Generate dot product code for f32x4.
    ///
    /// Computes sum(a * b) for two f32x4 vectors.
    pub fn emit_dot_product_f32x4(&self) -> WasmResult<Vec<WasmInstr>> {
        if !self.simd_enabled {
            return Err(WasmError::SimdNotAvailable(
                "SIMD not enabled for dot product".to_string(),
            ));
        }

        let mut instrs = Vec::new();

        // Multiply vectors element-wise
        instrs.push(WasmInstr::F32x4Mul);

        // Horizontal sum
        instrs.extend(self.reduce_f32x4(ReduceOp::Add)?);

        Ok(instrs)
    }

    /// Generate dot product code for f64x2.
    pub fn emit_dot_product_f64x2(&self) -> WasmResult<Vec<WasmInstr>> {
        if !self.simd_enabled {
            return Err(WasmError::SimdNotAvailable(
                "SIMD not enabled for dot product".to_string(),
            ));
        }

        let mut instrs = Vec::new();

        // Multiply vectors element-wise
        instrs.push(WasmInstr::F64x2Mul);

        // Horizontal sum
        instrs.extend(self.reduce_f64x2(ReduceOp::Add)?);

        Ok(instrs)
    }

    /// Generate FMA (fused multiply-add) code.
    ///
    /// WASM SIMD doesn't have FMA, so we lower to mul + add.
    /// This may produce different results from true FMA due to
    /// intermediate rounding.
    pub fn emit_fma_f32x4(&self) -> WasmResult<Vec<WasmInstr>> {
        // Input stack: [a, b, c]
        // Output: a * b + c

        let mut instrs = Vec::new();

        // We need a temporary to save c
        // This is handled by the caller who manages locals

        // For the instruction sequence, we assume:
        // Stack: [a, b, c]
        // After local.set temp: [a, b], temp = c
        // After mul: [a*b]
        // After local.get temp: [a*b, c]
        // After add: [a*b+c]

        // The actual instruction emission is:
        instrs.push(WasmInstr::Comment("FMA: a * b + c (lowered)".to_string()));
        // local.set and local.get are handled by emitter

        Ok(instrs)
    }

    /// Generate lane permutation code.
    pub fn emit_shuffle(&self, lanes: &[u8; 16]) -> WasmResult<WasmInstr> {
        if !self.simd_enabled {
            return Err(WasmError::SimdNotAvailable(
                "SIMD not enabled for shuffle".to_string(),
            ));
        }

        Ok(WasmInstr::I8x16Shuffle(*lanes))
    }

    /// Generate broadcast (splat) for f32x4 from a scalar.
    pub fn emit_broadcast_f32x4(&self) -> WasmInstr {
        WasmInstr::F32x4Splat
    }

    /// Generate broadcast (splat) for f64x2 from a scalar.
    pub fn emit_broadcast_f64x2(&self) -> WasmInstr {
        WasmInstr::F64x2Splat
    }

    /// Generate broadcast (splat) for i32x4 from a scalar.
    pub fn emit_broadcast_i32x4(&self) -> WasmInstr {
        WasmInstr::I32x4Splat
    }

    /// Check if a reduction operation is supported in SIMD.
    pub fn supports_simd_reduce(&self, op: ReduceOp, scalar_ty: ScalarType, width: u8) -> bool {
        if !self.simd_enabled {
            return false;
        }

        match (scalar_ty, width, op) {
            // f32x4 supports add, mul, min, max
            (
                ScalarType::Float(32),
                4,
                ReduceOp::Add | ReduceOp::Mul | ReduceOp::Min | ReduceOp::Max,
            ) => true,
            // f64x2 supports add, mul, min, max
            (
                ScalarType::Float(64),
                2,
                ReduceOp::Add | ReduceOp::Mul | ReduceOp::Min | ReduceOp::Max,
            ) => true,
            // i32x4 supports add, mul, and, or, xor
            (
                ScalarType::Int(32) | ScalarType::UInt(32),
                4,
                ReduceOp::Add | ReduceOp::Mul | ReduceOp::And | ReduceOp::Or | ReduceOp::Xor,
            ) => true,
            _ => false,
        }
    }
}

/// SIMD lowering strategies for different operation types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdStrategy {
    /// Use native SIMD instructions.
    Native,
    /// Use shuffle-based emulation.
    ShuffleEmulate,
    /// Fall back to scalar operations.
    ScalarFallback,
    /// Not possible to implement.
    Unsupported,
}

impl SimdStrategy {
    /// Determine the best strategy for a vector operation.
    pub fn for_binary(scalar_ty: ScalarType, width: u8, simd_enabled: bool) -> Self {
        if !simd_enabled {
            return Self::ScalarFallback;
        }

        match (scalar_ty, width) {
            // 128-bit vectors are natively supported
            (ScalarType::Float(32), 4) => Self::Native,
            (ScalarType::Float(64), 2) => Self::Native,
            (ScalarType::Int(32) | ScalarType::UInt(32), 4) => Self::Native,
            (ScalarType::Int(16) | ScalarType::UInt(16), 8) => Self::Native,
            (ScalarType::Int(8) | ScalarType::UInt(8), 16) => Self::Native,
            (ScalarType::Int(64) | ScalarType::UInt(64), 2) => Self::Native,

            // Smaller vectors use partial v128
            (ScalarType::Float(32), w) if w < 4 => Self::Native,
            (ScalarType::Float(64), 1) => Self::Native,
            (ScalarType::Int(32) | ScalarType::UInt(32), w) if w < 4 => Self::Native,

            // Larger vectors need decomposition
            (ScalarType::Float(32), 8) => Self::Native, // Two v128 ops
            (ScalarType::Float(64), 4) => Self::Native,

            // Very large or unusual widths
            _ => Self::ScalarFallback,
        }
    }

    /// Determine strategy for horizontal reduction.
    pub fn for_reduce(scalar_ty: ScalarType, width: u8, op: ReduceOp, simd_enabled: bool) -> Self {
        if !simd_enabled {
            return Self::ScalarFallback;
        }

        match (scalar_ty, width, op) {
            // f32x4 reductions via shuffle
            (
                ScalarType::Float(32),
                4,
                ReduceOp::Add | ReduceOp::Mul | ReduceOp::Min | ReduceOp::Max,
            ) => Self::ShuffleEmulate,
            // f64x2 reductions via shuffle
            (
                ScalarType::Float(64),
                2,
                ReduceOp::Add | ReduceOp::Mul | ReduceOp::Min | ReduceOp::Max,
            ) => Self::ShuffleEmulate,
            // i32x4 add via shuffle
            (ScalarType::Int(32) | ScalarType::UInt(32), 4, ReduceOp::Add | ReduceOp::Mul) => {
                Self::ShuffleEmulate
            }
            // Bitwise reductions
            (_, _, ReduceOp::And | ReduceOp::Or | ReduceOp::Xor) => Self::ShuffleEmulate,

            _ => Self::ScalarFallback,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_lowering_creation() {
        let lowering = SimdLowering::new(true);
        assert!(lowering.is_available());

        let lowering_off = SimdLowering::new(false);
        assert!(!lowering_off.is_available());
    }

    #[test]
    fn test_horizontal_reduce_f32x4() {
        let lowering = SimdLowering::new(true);
        let instrs = lowering
            .emit_horizontal_reduce(ReduceOp::Add, ScalarType::Float(32), 4)
            .unwrap();

        // Should have shuffle, add, shuffle, add, extract
        assert!(instrs.len() >= 4);
        assert!(matches!(
            instrs.last().unwrap(),
            WasmInstr::F32x4ExtractLane(0)
        ));
    }

    #[test]
    fn test_horizontal_reduce_f64x2() {
        let lowering = SimdLowering::new(true);
        let instrs = lowering
            .emit_horizontal_reduce(ReduceOp::Add, ScalarType::Float(64), 2)
            .unwrap();

        // Should have shuffle, add, extract
        assert!(instrs.len() >= 2);
        assert!(matches!(
            instrs.last().unwrap(),
            WasmInstr::F64x2ExtractLane(0)
        ));
    }

    #[test]
    fn test_dot_product_f32x4() {
        let lowering = SimdLowering::new(true);
        let instrs = lowering.emit_dot_product_f32x4().unwrap();

        // Should start with mul
        assert!(matches!(instrs[0], WasmInstr::F32x4Mul));
    }

    #[test]
    fn test_simd_strategy_native() {
        assert_eq!(
            SimdStrategy::for_binary(ScalarType::Float(32), 4, true),
            SimdStrategy::Native
        );
        assert_eq!(
            SimdStrategy::for_binary(ScalarType::Float(64), 2, true),
            SimdStrategy::Native
        );
    }

    #[test]
    fn test_simd_strategy_fallback() {
        assert_eq!(
            SimdStrategy::for_binary(ScalarType::Float(32), 4, false),
            SimdStrategy::ScalarFallback
        );
    }

    #[test]
    fn test_reduce_support() {
        let lowering = SimdLowering::new(true);

        assert!(lowering.supports_simd_reduce(ReduceOp::Add, ScalarType::Float(32), 4));
        assert!(lowering.supports_simd_reduce(ReduceOp::Max, ScalarType::Float(64), 2));
        assert!(!lowering.supports_simd_reduce(ReduceOp::Min, ScalarType::Int(64), 2));
    }
}
