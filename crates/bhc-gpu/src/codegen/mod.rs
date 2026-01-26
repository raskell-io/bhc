//! GPU code generation infrastructure.
//!
//! This module provides code generation from Tensor IR to GPU assembly
//! (PTX for CUDA, AMDGCN for ROCm).
//!
//! # Code Generation Pipeline
//!
//! ```text
//! Tensor IR Kernel
//!        │
//!        ▼
//! ┌─────────────────┐
//! │ Kernel Analysis │ ──▶ Determine grid/block dimensions
//! └─────────────────┘      Memory access patterns
//!        │
//!        ▼
//! ┌─────────────────┐
//! │   IR Lowering   │ ──▶ Convert TensorOp to loop nests
//! └─────────────────┘
//!        │
//!        ├──────────────────┬────────────────────┐
//!        ▼                  ▼                    ▼
//! ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐
//! │ PTX Codegen │   │AMDGCN Codegen│   │ (Future: SPIR-V)│
//! └─────────────┘   └─────────────┘   └─────────────────┘
//! ```
//!
//! # PTX Generation
//!
//! For NVIDIA GPUs, we generate PTX (Parallel Thread Execution) assembly.
//! PTX is then JIT-compiled by the CUDA driver to device-specific code.
//!
//! # AMDGCN Generation
//!
//! For AMD GPUs, we generate AMDGCN assembly targeting specific GFX
//! architectures (gfx900, gfx90a, etc.).

pub mod amdgcn;
pub mod ptx;

use crate::device::DeviceInfo;
use crate::kernel::CompiledModule;
use crate::GpuResult;
use bhc_tensor_ir::{DType, Kernel, Shape};

/// Generate a mock compiled kernel for testing.
pub fn mock_compile_kernel(kernel: &Kernel, device: &DeviceInfo) -> GpuResult<CompiledModule> {
    let name = kernel.name.as_str().to_string();
    let code = format!(
        "; Mock kernel: {}\n; Device: {}\n; Inputs: {}\n; Outputs: {}\n",
        name,
        device.name,
        kernel.inputs.len(),
        kernel.outputs.len()
    );

    let mut module = CompiledModule::from_text(name.clone(), code, "mock".to_string());
    module.add_entry_point(name);

    Ok(module)
}

/// Data type to PTX/AMDGCN type name.
fn dtype_to_gpu_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "pred",
        DType::Int8 => "s8",
        DType::Int16 => "s16",
        DType::Int32 => "s32",
        DType::Int64 => "s64",
        DType::UInt8 => "u8",
        DType::UInt16 => "u16",
        DType::UInt32 => "u32",
        DType::UInt64 => "u64",
        DType::Float16 => "f16",
        DType::Float32 => "f32",
        DType::Float64 => "f64",
        DType::BFloat16 => "bf16",
        DType::Complex64 => "f32",  // Pair of f32
        DType::Complex128 => "f64", // Pair of f64
    }
}

/// Data type to register width in bits.
fn dtype_reg_width(dtype: DType) -> u32 {
    match dtype {
        DType::Bool => 1,
        DType::Int8 | DType::UInt8 => 8,
        DType::Int16 | DType::UInt16 | DType::Float16 | DType::BFloat16 => 16,
        DType::Int32 | DType::UInt32 | DType::Float32 => 32,
        DType::Int64 | DType::UInt64 | DType::Float64 | DType::Complex64 => 64,
        DType::Complex128 => 128,
    }
}

/// Kernel parameters for code generation.
#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Kernel name.
    pub name: String,
    /// Input parameter types.
    pub inputs: Vec<ParamType>,
    /// Output parameter types.
    pub outputs: Vec<ParamType>,
    /// Total shared memory required.
    pub shared_memory: usize,
    /// Recommended block size.
    pub block_size: u32,
}

/// Parameter type for kernel signature.
#[derive(Debug, Clone)]
pub struct ParamType {
    /// Parameter name.
    pub name: String,
    /// Data type.
    pub dtype: DType,
    /// Shape (for tensors).
    pub shape: Option<Shape>,
    /// Whether this is an output (mutable).
    pub is_output: bool,
}

impl KernelParams {
    /// Create from a Tensor IR kernel.
    pub fn from_kernel(kernel: &Kernel) -> Self {
        let inputs = kernel
            .inputs
            .iter()
            .enumerate()
            .map(|(i, t)| ParamType {
                name: format!("in{}", i),
                dtype: t.meta.dtype,
                shape: Some(t.meta.shape.clone()),
                is_output: false,
            })
            .collect();

        let outputs = kernel
            .outputs
            .iter()
            .enumerate()
            .map(|(i, t)| ParamType {
                name: format!("out{}", i),
                dtype: t.meta.dtype,
                shape: Some(t.meta.shape.clone()),
                is_output: true,
            })
            .collect();

        Self {
            name: kernel.name.as_str().to_string(),
            inputs,
            outputs,
            shared_memory: 0,
            block_size: 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(dtype_to_gpu_type(DType::Float32), "f32");
        assert_eq!(dtype_to_gpu_type(DType::Float64), "f64");
        assert_eq!(dtype_to_gpu_type(DType::Int32), "s32");
    }

    #[test]
    fn test_dtype_reg_width() {
        assert_eq!(dtype_reg_width(DType::Float32), 32);
        assert_eq!(dtype_reg_width(DType::Float64), 64);
        assert_eq!(dtype_reg_width(DType::Int8), 8);
    }
}
