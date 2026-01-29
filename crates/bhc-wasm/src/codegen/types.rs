//! Type mapping from Loop IR to WASM types.
//!
//! This module handles the conversion of BHC's Loop IR types to their
//! WebAssembly equivalents, including scalar types, vectors (SIMD128),
//! and memory pointer types.

use crate::{WasmError, WasmResult, WasmType};
use bhc_loop_ir::{LoopType, ScalarType};
use bhc_target::Arch;

/// Mapping configuration for Loop IR types to WASM.
#[derive(Clone, Debug)]
pub struct LoopTypeMapping {
    /// Whether the target is 64-bit (wasm64).
    pub is_64bit: bool,
    /// Whether SIMD128 is available.
    pub simd_enabled: bool,
}

impl LoopTypeMapping {
    /// Create a mapping configuration for the given architecture.
    #[must_use]
    pub fn for_arch(arch: Arch, simd_enabled: bool) -> Self {
        Self {
            is_64bit: matches!(arch, Arch::Wasm64),
            simd_enabled,
        }
    }

    /// Get the WASM type for pointers.
    #[must_use]
    pub fn pointer_type(&self) -> WasmType {
        if self.is_64bit {
            WasmType::I64
        } else {
            WasmType::I32
        }
    }

    /// Get the WASM type for array indices.
    #[must_use]
    pub fn index_type(&self) -> WasmType {
        if self.is_64bit {
            WasmType::I64
        } else {
            WasmType::I32
        }
    }
}

impl Default for LoopTypeMapping {
    fn default() -> Self {
        Self {
            is_64bit: false,
            simd_enabled: true,
        }
    }
}

/// Convert a Loop IR type to a WASM type.
///
/// # Mapping Rules
///
/// | Loop IR Type | WASM Type |
/// |--------------|-----------|
/// | Void | (no type) |
/// | Bool | i32 |
/// | Int(8/16/32) | i32 |
/// | Int(64) | i64 |
/// | UInt(8/16/32) | i32 |
/// | UInt(64) | i64 |
/// | Float(32) | f32 |
/// | Float(64) | f64 |
/// | Vector(*, 4) f32 | v128 |
/// | Vector(*, 2) f64 | v128 |
/// | Ptr(*) | i32 or i64 |
///
/// # Errors
///
/// Returns an error if the type cannot be represented in WASM.
pub fn type_to_wasm(ty: &LoopType, mapping: &LoopTypeMapping) -> WasmResult<WasmType> {
    match ty {
        LoopType::Void => Err(WasmError::NotSupported(
            "Void type cannot be used as a value type".to_string(),
        )),

        LoopType::Scalar(scalar) => scalar_to_wasm(*scalar),

        LoopType::Vector(scalar, width) => {
            if !mapping.simd_enabled {
                return Err(WasmError::SimdNotAvailable(
                    "SIMD is not enabled for this target".to_string(),
                ));
            }
            vector_to_wasm(*scalar, *width)
        }

        LoopType::Ptr(_) => Ok(mapping.pointer_type()),
    }
}

/// Convert a scalar type to a WASM type.
fn scalar_to_wasm(scalar: ScalarType) -> WasmResult<WasmType> {
    match scalar {
        ScalarType::Bool => Ok(WasmType::I32),

        ScalarType::Int(bits) | ScalarType::UInt(bits) => match bits {
            8 | 16 | 32 => Ok(WasmType::I32),
            64 => Ok(WasmType::I64),
            _ => Err(WasmError::NotSupported(format!(
                "Integer with {bits} bits not supported in WASM"
            ))),
        },

        ScalarType::Float(bits) => {
            match bits {
                32 => Ok(WasmType::F32),
                64 => Ok(WasmType::F64),
                16 => {
                    // Float16 is not natively supported, use f32
                    Ok(WasmType::F32)
                }
                _ => Err(WasmError::NotSupported(format!(
                    "Float with {bits} bits not supported in WASM"
                ))),
            }
        }
    }
}

/// Convert a vector type to WASM v128.
fn vector_to_wasm(scalar: ScalarType, width: u8) -> WasmResult<WasmType> {
    // WASM SIMD128 supports 128-bit vectors only
    let scalar_bits = match scalar {
        ScalarType::Bool => 8,
        ScalarType::Int(bits) | ScalarType::UInt(bits) | ScalarType::Float(bits) => bits as usize,
    };

    let total_bits = scalar_bits * (width as usize);

    if total_bits == 128 {
        Ok(WasmType::V128)
    } else if total_bits < 128 {
        // Partial vectors can still use v128 with unused lanes
        Ok(WasmType::V128)
    } else {
        Err(WasmError::NotSupported(format!(
            "Vector type {scalar:?}x{width} ({total_bits} bits) exceeds WASM SIMD128"
        )))
    }
}

/// Get the WASM load/store alignment for a type.
#[must_use]
pub fn alignment_for_type(ty: &LoopType) -> u32 {
    match ty {
        LoopType::Void => 0,
        LoopType::Scalar(scalar) => scalar_alignment(*scalar),
        LoopType::Vector(_, _) => 4, // v128 uses 16-byte alignment, log2 = 4
        LoopType::Ptr(_) => 2,       // 4-byte alignment for wasm32, log2 = 2
    }
}

/// Get the alignment for a scalar type (as log2).
fn scalar_alignment(scalar: ScalarType) -> u32 {
    match scalar {
        ScalarType::Bool => 0,
        ScalarType::Int(8) | ScalarType::UInt(8) => 0,
        ScalarType::Int(16) | ScalarType::UInt(16) => 1,
        ScalarType::Int(32) | ScalarType::UInt(32) | ScalarType::Float(32) => 2,
        ScalarType::Int(64) | ScalarType::UInt(64) | ScalarType::Float(64) => 3,
        ScalarType::Float(16) => 1, // f16 has 2-byte alignment
        _ => 2,                     // Default to 4-byte alignment
    }
}

/// Get the size in bytes for a type.
#[must_use]
pub fn size_for_type(ty: &LoopType, is_64bit: bool) -> usize {
    match ty {
        LoopType::Void => 0,
        LoopType::Scalar(scalar) => scalar_size(*scalar),
        LoopType::Vector(scalar, width) => scalar_size(*scalar) * (*width as usize),
        LoopType::Ptr(_) => {
            if is_64bit {
                8
            } else {
                4
            }
        }
    }
}

/// Get the size in bytes for a scalar type.
fn scalar_size(scalar: ScalarType) -> usize {
    match scalar {
        ScalarType::Bool => 1,
        ScalarType::Int(bits) | ScalarType::UInt(bits) | ScalarType::Float(bits) => {
            (bits as usize + 7) / 8
        }
    }
}

/// Determine the appropriate SIMD lane type for a scalar.
#[must_use]
pub fn simd_lane_type(scalar: ScalarType) -> SimdLaneType {
    match scalar {
        ScalarType::Bool => SimdLaneType::I8x16,
        ScalarType::Int(8) | ScalarType::UInt(8) => SimdLaneType::I8x16,
        ScalarType::Int(16) | ScalarType::UInt(16) => SimdLaneType::I16x8,
        ScalarType::Int(32) | ScalarType::UInt(32) => SimdLaneType::I32x4,
        ScalarType::Int(64) | ScalarType::UInt(64) => SimdLaneType::I64x2,
        ScalarType::Float(32) => SimdLaneType::F32x4,
        ScalarType::Float(64) => SimdLaneType::F64x2,
        ScalarType::Float(16) => SimdLaneType::F32x4, // Promoted to f32
        _ => SimdLaneType::I32x4,                     // Default
    }
}

/// SIMD lane types in WASM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdLaneType {
    /// 16 x 8-bit integers.
    I8x16,
    /// 8 x 16-bit integers.
    I16x8,
    /// 4 x 32-bit integers.
    I32x4,
    /// 2 x 64-bit integers.
    I64x2,
    /// 4 x 32-bit floats.
    F32x4,
    /// 2 x 64-bit floats.
    F64x2,
}

impl SimdLaneType {
    /// Get the number of lanes.
    #[must_use]
    pub const fn lane_count(self) -> u8 {
        match self {
            Self::I8x16 => 16,
            Self::I16x8 => 8,
            Self::I32x4 | Self::F32x4 => 4,
            Self::I64x2 | Self::F64x2 => 2,
        }
    }

    /// Get the lane width in bits.
    #[must_use]
    pub const fn lane_bits(self) -> u8 {
        match self {
            Self::I8x16 => 8,
            Self::I16x8 => 16,
            Self::I32x4 | Self::F32x4 => 32,
            Self::I64x2 | Self::F64x2 => 64,
        }
    }

    /// Check if this is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32x4 | Self::F64x2)
    }

    /// Get the WAT type name.
    #[must_use]
    pub const fn wat_name(self) -> &'static str {
        match self {
            Self::I8x16 => "i8x16",
            Self::I16x8 => "i16x8",
            Self::I32x4 => "i32x4",
            Self::I64x2 => "i64x2",
            Self::F32x4 => "f32x4",
            Self::F64x2 => "f64x2",
        }
    }
}

/// Result type conversion information.
#[derive(Clone, Debug)]
pub struct TypeConversion {
    /// Source type.
    pub from: WasmType,
    /// Destination type.
    pub to: WasmType,
    /// Whether this is a widening conversion.
    pub widening: bool,
    /// Whether this is a floating-point to integer conversion.
    pub float_to_int: bool,
}

impl TypeConversion {
    /// Check if a conversion is needed between two Loop IR types.
    pub fn needed(from: &LoopType, to: &LoopType, mapping: &LoopTypeMapping) -> Option<Self> {
        let from_wasm = type_to_wasm(from, mapping).ok()?;
        let to_wasm = type_to_wasm(to, mapping).ok()?;

        if from_wasm == to_wasm {
            return None;
        }

        let widening = from_wasm.size_bytes() < to_wasm.size_bytes();
        let float_to_int = matches!(from_wasm, WasmType::F32 | WasmType::F64)
            && matches!(to_wasm, WasmType::I32 | WasmType::I64);

        Some(Self {
            from: from_wasm,
            to: to_wasm,
            widening,
            float_to_int,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_type_mapping() {
        let mapping = LoopTypeMapping::default();

        // Boolean maps to i32
        let bool_ty = LoopType::Scalar(ScalarType::Bool);
        assert_eq!(type_to_wasm(&bool_ty, &mapping).unwrap(), WasmType::I32);

        // i32 maps to i32
        let i32_ty = LoopType::Scalar(ScalarType::Int(32));
        assert_eq!(type_to_wasm(&i32_ty, &mapping).unwrap(), WasmType::I32);

        // i64 maps to i64
        let i64_ty = LoopType::Scalar(ScalarType::Int(64));
        assert_eq!(type_to_wasm(&i64_ty, &mapping).unwrap(), WasmType::I64);

        // f32 maps to f32
        let f32_ty = LoopType::Scalar(ScalarType::Float(32));
        assert_eq!(type_to_wasm(&f32_ty, &mapping).unwrap(), WasmType::F32);

        // f64 maps to f64
        let f64_ty = LoopType::Scalar(ScalarType::Float(64));
        assert_eq!(type_to_wasm(&f64_ty, &mapping).unwrap(), WasmType::F64);
    }

    #[test]
    fn test_vector_type_mapping() {
        let mapping = LoopTypeMapping::default();

        // f32x4 (128 bits) maps to v128
        let vec_f32x4 = LoopType::Vector(ScalarType::Float(32), 4);
        assert_eq!(type_to_wasm(&vec_f32x4, &mapping).unwrap(), WasmType::V128);

        // f64x2 (128 bits) maps to v128
        let vec_f64x2 = LoopType::Vector(ScalarType::Float(64), 2);
        assert_eq!(type_to_wasm(&vec_f64x2, &mapping).unwrap(), WasmType::V128);

        // i32x4 (128 bits) maps to v128
        let vec_i32x4 = LoopType::Vector(ScalarType::Int(32), 4);
        assert_eq!(type_to_wasm(&vec_i32x4, &mapping).unwrap(), WasmType::V128);
    }

    #[test]
    fn test_vector_type_without_simd() {
        let mapping = LoopTypeMapping {
            is_64bit: false,
            simd_enabled: false,
        };

        let vec_f32x4 = LoopType::Vector(ScalarType::Float(32), 4);
        assert!(type_to_wasm(&vec_f32x4, &mapping).is_err());
    }

    #[test]
    fn test_pointer_type_mapping() {
        // 32-bit
        let mapping_32 = LoopTypeMapping {
            is_64bit: false,
            simd_enabled: true,
        };
        let ptr_ty = LoopType::Ptr(Box::new(LoopType::Scalar(ScalarType::Float(32))));
        assert_eq!(type_to_wasm(&ptr_ty, &mapping_32).unwrap(), WasmType::I32);

        // 64-bit
        let mapping_64 = LoopTypeMapping {
            is_64bit: true,
            simd_enabled: true,
        };
        assert_eq!(type_to_wasm(&ptr_ty, &mapping_64).unwrap(), WasmType::I64);
    }

    #[test]
    fn test_simd_lane_type() {
        assert_eq!(simd_lane_type(ScalarType::Float(32)), SimdLaneType::F32x4);
        assert_eq!(simd_lane_type(ScalarType::Float(64)), SimdLaneType::F64x2);
        assert_eq!(simd_lane_type(ScalarType::Int(32)), SimdLaneType::I32x4);
        assert_eq!(simd_lane_type(ScalarType::Int(8)), SimdLaneType::I8x16);
    }

    #[test]
    fn test_alignment() {
        let i8_ty = LoopType::Scalar(ScalarType::Int(8));
        assert_eq!(alignment_for_type(&i8_ty), 0); // 1-byte alignment

        let i32_ty = LoopType::Scalar(ScalarType::Int(32));
        assert_eq!(alignment_for_type(&i32_ty), 2); // 4-byte alignment

        let f64_ty = LoopType::Scalar(ScalarType::Float(64));
        assert_eq!(alignment_for_type(&f64_ty), 3); // 8-byte alignment

        let v128_ty = LoopType::Vector(ScalarType::Float(32), 4);
        assert_eq!(alignment_for_type(&v128_ty), 4); // 16-byte alignment
    }

    #[test]
    fn test_size() {
        let is_64bit = false;

        let i32_ty = LoopType::Scalar(ScalarType::Int(32));
        assert_eq!(size_for_type(&i32_ty, is_64bit), 4);

        let f64_ty = LoopType::Scalar(ScalarType::Float(64));
        assert_eq!(size_for_type(&f64_ty, is_64bit), 8);

        let v128_ty = LoopType::Vector(ScalarType::Float(32), 4);
        assert_eq!(size_for_type(&v128_ty, is_64bit), 16);

        let ptr_ty = LoopType::Ptr(Box::new(LoopType::Void));
        assert_eq!(size_for_type(&ptr_ty, false), 4);
        assert_eq!(size_for_type(&ptr_ty, true), 8);
    }
}
