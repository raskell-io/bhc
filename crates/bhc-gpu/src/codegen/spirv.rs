//! SPIR-V code generation for Vulkan/OpenCL compute shaders.
//!
//! This module generates SPIR-V assembly from Tensor IR kernels. SPIR-V is the
//! intermediate representation used by Vulkan compute shaders and OpenCL 2.0+.
//!
//! # SPIR-V Overview
//!
//! SPIR-V is a binary intermediate language with:
//! - Static single assignment (SSA) form
//! - Typed instructions
//! - Structured control flow
//! - Multiple execution models (Kernel for OpenCL, GLCompute for Vulkan)
//!
//! # Example SPIR-V Assembly
//!
//! ```spirv
//! ; SPIR-V
//! ; Version: 1.5
//! ; Generator: BHC
//! OpCapability Shader
//! OpMemoryModel Logical GLSL450
//! OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
//! OpExecutionMode %main LocalSize 256 1 1
//! ```

use super::{dtype_to_gpu_type, KernelParams};
use crate::device::DeviceInfo;
use crate::kernel::CompiledModule;
use crate::GpuResult;
use bhc_tensor_ir::{
    BinaryOp, DType, Kernel, KernelBody, MapFn, ReduceOp, TensorOp, UnaryOp, ZipFn,
};
use std::fmt::Write;

/// SPIR-V version to target.
const SPIRV_VERSION: &str = "1.5";

/// Generate SPIR-V module header.
pub fn generate_module_header(name: &str, device: &DeviceInfo) -> String {
    format!(
        "; SPIR-V\n\
         ; Version: {}\n\
         ; Generator: BHC (Basel Haskell Compiler)\n\
         ; Module: {}\n\
         ; Target: {}\n\
         \n\
         OpCapability Shader\n\
         OpCapability Int64\n\
         OpCapability Float64\n\
         OpMemoryModel Logical GLSL450\n\n",
        SPIRV_VERSION, name, device.arch_name()
    )
}

/// Compile a Tensor IR kernel to SPIR-V.
pub fn compile_kernel(kernel: &Kernel, device: &DeviceInfo) -> GpuResult<CompiledModule> {
    let params = KernelParams::from_kernel(kernel);
    let mut code = generate_module_header(&params.name, device);

    // Generate kernel entry point
    generate_kernel_entry(&mut code, &params, kernel)?;

    let mut module = CompiledModule::from_text(params.name.clone(), code, device.arch_name());
    module.add_entry_point(params.name);

    Ok(module)
}

/// Generate a kernel entry point.
fn generate_kernel_entry(
    code: &mut String,
    params: &KernelParams,
    kernel: &Kernel,
) -> GpuResult<()> {
    let block_size = params.block_size;

    // Type declarations
    writeln!(code, "; Type declarations").unwrap();
    writeln!(code, "%void = OpTypeVoid").unwrap();
    writeln!(code, "%func = OpTypeFunction %void").unwrap();
    writeln!(code, "%uint = OpTypeInt 32 0").unwrap();
    writeln!(code, "%int = OpTypeInt 32 1").unwrap();
    writeln!(code, "%uint64 = OpTypeInt 64 0").unwrap();
    writeln!(code, "%float = OpTypeFloat 32").unwrap();
    writeln!(code, "%double = OpTypeFloat 64").unwrap();
    writeln!(code, "%v3uint = OpTypeVector %uint 3").unwrap();
    writeln!(code, "%ptr_input_v3uint = OpTypePointer Input %v3uint").unwrap();
    writeln!(code, "%ptr_uniform_float = OpTypePointer Uniform %float").unwrap();
    writeln!(code, "%ptr_storage_float = OpTypePointer StorageBuffer %float").unwrap();
    writeln!(code).unwrap();

    // Constants
    writeln!(code, "; Constants").unwrap();
    writeln!(code, "%uint_0 = OpConstant %uint 0").unwrap();
    writeln!(code, "%uint_1 = OpConstant %uint 1").unwrap();
    writeln!(code, "%uint_2 = OpConstant %uint 2").unwrap();
    writeln!(code, "%uint_{} = OpConstant %uint {}", block_size, block_size).unwrap();
    writeln!(code).unwrap();

    // Built-in variables
    writeln!(code, "; Built-in variables").unwrap();
    writeln!(code, "%gl_GlobalInvocationID = OpVariable %ptr_input_v3uint Input").unwrap();
    writeln!(code, "OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId").unwrap();
    writeln!(code).unwrap();

    // Buffer declarations for inputs/outputs
    writeln!(code, "; Buffer declarations").unwrap();
    for (i, input) in params.inputs.iter().enumerate() {
        let ty = dtype_to_spirv_type(input.dtype);
        writeln!(
            code,
            "%buffer_in{} = OpVariable %ptr_storage_{} StorageBuffer",
            i, ty
        )
        .unwrap();
        writeln!(code, "OpDecorate %buffer_in{} DescriptorSet 0", i).unwrap();
        writeln!(code, "OpDecorate %buffer_in{} Binding {}", i, i).unwrap();
    }
    for (i, output) in params.outputs.iter().enumerate() {
        let ty = dtype_to_spirv_type(output.dtype);
        let binding = params.inputs.len() + i;
        writeln!(
            code,
            "%buffer_out{} = OpVariable %ptr_storage_{} StorageBuffer",
            i, ty
        )
        .unwrap();
        writeln!(code, "OpDecorate %buffer_out{} DescriptorSet 0", i).unwrap();
        writeln!(code, "OpDecorate %buffer_out{} Binding {}", i, binding).unwrap();
    }
    writeln!(code).unwrap();

    // Entry point declaration
    writeln!(code, "; Entry point").unwrap();
    writeln!(
        code,
        "OpEntryPoint GLCompute %main \"{}\" %gl_GlobalInvocationID",
        params.name
    )
    .unwrap();
    writeln!(
        code,
        "OpExecutionMode %main LocalSize {} 1 1",
        block_size
    )
    .unwrap();
    writeln!(code).unwrap();

    // Main function
    writeln!(code, "; Main function").unwrap();
    writeln!(code, "%main = OpFunction %void None %func").unwrap();
    writeln!(code, "%entry = OpLabel").unwrap();
    writeln!(code).unwrap();

    // Get global invocation ID
    writeln!(code, "; Get global thread index").unwrap();
    writeln!(
        code,
        "%gid_vec = OpLoad %v3uint %gl_GlobalInvocationID"
    )
    .unwrap();
    writeln!(
        code,
        "%gid = OpCompositeExtract %uint %gid_vec 0"
    )
    .unwrap();
    writeln!(code).unwrap();

    // Generate kernel body
    match &kernel.body {
        KernelBody::Fused(ops) => {
            generate_fused_ops(code, ops, params)?;
        }
        KernelBody::LoopNest(_nest) => {
            writeln!(code, "; Loop nest not yet implemented").unwrap();
        }
    }

    // Return
    writeln!(code, "OpReturn").unwrap();
    writeln!(code, "OpFunctionEnd").unwrap();

    Ok(())
}

/// Generate code for fused operations.
fn generate_fused_ops(code: &mut String, ops: &[TensorOp], params: &KernelParams) -> GpuResult<()> {
    writeln!(code, "; Fused operations").unwrap();

    for (i, op) in ops.iter().enumerate() {
        match op {
            TensorOp::Unary(unary_op, input) => {
                let dtype = input.meta.dtype;
                writeln!(code, "; Unary operation: {:?}", unary_op).unwrap();
                generate_unary_op(code, *unary_op, dtype, i)?;
            }
            TensorOp::Binary(binary_op, left, _right) => {
                let dtype = left.meta.dtype;
                writeln!(code, "; Binary operation: {:?}", binary_op).unwrap();
                generate_binary_op(code, *binary_op, dtype, i)?;
            }
            TensorOp::Map(map_fn, input) => {
                let dtype = input.meta.dtype;
                writeln!(code, "; Map operation: {}", map_fn.name.as_str()).unwrap();
                generate_map_op(code, map_fn, dtype, params, i)?;
            }
            TensorOp::ZipWith(zip_fn, left, _right) => {
                let dtype = left.meta.dtype;
                writeln!(code, "; ZipWith operation: {}", zip_fn.name.as_str()).unwrap();
                generate_zipwith_op(code, zip_fn, dtype, params, i)?;
            }
            TensorOp::Reduce(reduce_op, _axis, input) => {
                let dtype = input.meta.dtype;
                writeln!(code, "; Reduce operation: {:?}", reduce_op).unwrap();
                generate_reduce_op(code, *reduce_op, dtype, params, i)?;
            }
            TensorOp::ReduceAll(reduce_op, input) => {
                let dtype = input.meta.dtype;
                writeln!(code, "; ReduceAll operation: {:?}", reduce_op).unwrap();
                generate_reduce_op(code, *reduce_op, dtype, params, i)?;
            }
            _ => {
                writeln!(code, "; Unsupported operation").unwrap();
            }
        }
    }

    Ok(())
}

/// Generate SPIR-V for unary operation.
fn generate_unary_op(code: &mut String, op: UnaryOp, dtype: DType, idx: usize) -> GpuResult<()> {
    let ty = dtype_to_spirv_type(dtype);

    // Load input
    writeln!(code, "%unary_ptr{} = OpAccessChain %ptr_storage_{} %buffer_in0 %gid", idx, ty).unwrap();
    writeln!(code, "%unary_in{} = OpLoad %{} %unary_ptr{}", idx, ty, idx).unwrap();

    // Apply operation
    let result = match op {
        UnaryOp::Neg => {
            if dtype.is_float() {
                writeln!(code, "%unary_out{} = OpFNegate %{} %unary_in{}", idx, ty, idx).unwrap();
            } else {
                writeln!(code, "%unary_out{} = OpSNegate %{} %unary_in{}", idx, ty, idx).unwrap();
            }
            format!("%unary_out{}", idx)
        }
        UnaryOp::Abs => {
            writeln!(code, "%unary_out{} = OpExtInst %{} %GLSL_std_450 FAbs %unary_in{}", idx, ty, idx).unwrap();
            format!("%unary_out{}", idx)
        }
        UnaryOp::Sqrt => {
            writeln!(code, "%unary_out{} = OpExtInst %{} %GLSL_std_450 Sqrt %unary_in{}", idx, ty, idx).unwrap();
            format!("%unary_out{}", idx)
        }
        UnaryOp::Exp => {
            writeln!(code, "%unary_out{} = OpExtInst %{} %GLSL_std_450 Exp %unary_in{}", idx, ty, idx).unwrap();
            format!("%unary_out{}", idx)
        }
        UnaryOp::Log => {
            writeln!(code, "%unary_out{} = OpExtInst %{} %GLSL_std_450 Log %unary_in{}", idx, ty, idx).unwrap();
            format!("%unary_out{}", idx)
        }
        UnaryOp::Sin => {
            writeln!(code, "%unary_out{} = OpExtInst %{} %GLSL_std_450 Sin %unary_in{}", idx, ty, idx).unwrap();
            format!("%unary_out{}", idx)
        }
        UnaryOp::Cos => {
            writeln!(code, "%unary_out{} = OpExtInst %{} %GLSL_std_450 Cos %unary_in{}", idx, ty, idx).unwrap();
            format!("%unary_out{}", idx)
        }
        _ => {
            writeln!(code, "; Unsupported unary op: {:?}", op).unwrap();
            format!("%unary_in{}", idx)
        }
    };

    // Store output
    writeln!(code, "%out_ptr{} = OpAccessChain %ptr_storage_{} %buffer_out0 %gid", idx, ty).unwrap();
    writeln!(code, "OpStore %out_ptr{} {}", idx, result).unwrap();

    Ok(())
}

/// Generate SPIR-V for binary operation.
fn generate_binary_op(code: &mut String, op: BinaryOp, dtype: DType, idx: usize) -> GpuResult<()> {
    let ty = dtype_to_spirv_type(dtype);

    // Load inputs
    writeln!(code, "%bin_ptr_a{} = OpAccessChain %ptr_storage_{} %buffer_in0 %gid", idx, ty).unwrap();
    writeln!(code, "%bin_a{} = OpLoad %{} %bin_ptr_a{}", idx, ty, idx).unwrap();
    writeln!(code, "%bin_ptr_b{} = OpAccessChain %ptr_storage_{} %buffer_in1 %gid", idx, ty).unwrap();
    writeln!(code, "%bin_b{} = OpLoad %{} %bin_ptr_b{}", idx, ty, idx).unwrap();

    // Apply operation
    let result = match op {
        BinaryOp::Add => {
            if dtype.is_float() {
                writeln!(code, "%bin_out{} = OpFAdd %{} %bin_a{} %bin_b{}", idx, ty, idx, idx).unwrap();
            } else {
                writeln!(code, "%bin_out{} = OpIAdd %{} %bin_a{} %bin_b{}", idx, ty, idx, idx).unwrap();
            }
            format!("%bin_out{}", idx)
        }
        BinaryOp::Sub => {
            if dtype.is_float() {
                writeln!(code, "%bin_out{} = OpFSub %{} %bin_a{} %bin_b{}", idx, ty, idx, idx).unwrap();
            } else {
                writeln!(code, "%bin_out{} = OpISub %{} %bin_a{} %bin_b{}", idx, ty, idx, idx).unwrap();
            }
            format!("%bin_out{}", idx)
        }
        BinaryOp::Mul => {
            if dtype.is_float() {
                writeln!(code, "%bin_out{} = OpFMul %{} %bin_a{} %bin_b{}", idx, ty, idx, idx).unwrap();
            } else {
                writeln!(code, "%bin_out{} = OpIMul %{} %bin_a{} %bin_b{}", idx, ty, idx, idx).unwrap();
            }
            format!("%bin_out{}", idx)
        }
        BinaryOp::Div => {
            if dtype.is_float() {
                writeln!(code, "%bin_out{} = OpFDiv %{} %bin_a{} %bin_b{}", idx, ty, idx, idx).unwrap();
            } else if dtype.is_signed() {
                writeln!(code, "%bin_out{} = OpSDiv %{} %bin_a{} %bin_b{}", idx, ty, idx, idx).unwrap();
            } else {
                writeln!(code, "%bin_out{} = OpUDiv %{} %bin_a{} %bin_b{}", idx, ty, idx, idx).unwrap();
            }
            format!("%bin_out{}", idx)
        }
        _ => {
            writeln!(code, "; Unsupported binary op: {:?}", op).unwrap();
            format!("%bin_a{}", idx)
        }
    };

    // Store output
    writeln!(code, "%bin_out_ptr{} = OpAccessChain %ptr_storage_{} %buffer_out0 %gid", idx, ty).unwrap();
    writeln!(code, "OpStore %bin_out_ptr{} {}", idx, result).unwrap();

    Ok(())
}

/// Generate SPIR-V for map operation.
fn generate_map_op(
    code: &mut String,
    map_fn: &MapFn,
    dtype: DType,
    _params: &KernelParams,
    idx: usize,
) -> GpuResult<()> {
    let ty = dtype_to_spirv_type(dtype);
    let fn_name = map_fn.name.as_str();

    // Load input
    writeln!(code, "%map_ptr{} = OpAccessChain %ptr_storage_{} %buffer_in0 %gid", idx, ty).unwrap();
    writeln!(code, "%map_in{} = OpLoad %{} %map_ptr{}", idx, ty, idx).unwrap();

    // Apply map function based on name pattern
    let result = if fn_name.contains("mul") || fn_name.contains("*") {
        writeln!(code, "%map_const{} = OpConstant %{} 2.0", idx, ty).unwrap();
        writeln!(code, "%map_out{} = OpFMul %{} %map_in{} %map_const{}", idx, ty, idx, idx).unwrap();
        format!("%map_out{}", idx)
    } else if fn_name.contains("add") || fn_name.contains("+") {
        writeln!(code, "%map_const{} = OpConstant %{} 1.0", idx, ty).unwrap();
        writeln!(code, "%map_out{} = OpFAdd %{} %map_in{} %map_const{}", idx, ty, idx, idx).unwrap();
        format!("%map_out{}", idx)
    } else {
        // Identity
        format!("%map_in{}", idx)
    };

    // Store output
    writeln!(code, "%map_out_ptr{} = OpAccessChain %ptr_storage_{} %buffer_out0 %gid", idx, ty).unwrap();
    writeln!(code, "OpStore %map_out_ptr{} {}", idx, result).unwrap();

    Ok(())
}

/// Generate SPIR-V for zipwith operation.
fn generate_zipwith_op(
    code: &mut String,
    zip_fn: &ZipFn,
    dtype: DType,
    _params: &KernelParams,
    idx: usize,
) -> GpuResult<()> {
    let ty = dtype_to_spirv_type(dtype);
    let fn_name = zip_fn.name.as_str();

    // Load inputs
    writeln!(code, "%zip_ptr_a{} = OpAccessChain %ptr_storage_{} %buffer_in0 %gid", idx, ty).unwrap();
    writeln!(code, "%zip_a{} = OpLoad %{} %zip_ptr_a{}", idx, ty, idx).unwrap();
    writeln!(code, "%zip_ptr_b{} = OpAccessChain %ptr_storage_{} %buffer_in1 %gid", idx, ty).unwrap();
    writeln!(code, "%zip_b{} = OpLoad %{} %zip_ptr_b{}", idx, ty, idx).unwrap();

    // Apply zipwith function
    let result = if fn_name.contains("add") || fn_name.contains("+") {
        writeln!(code, "%zip_out{} = OpFAdd %{} %zip_a{} %zip_b{}", idx, ty, idx, idx).unwrap();
        format!("%zip_out{}", idx)
    } else if fn_name.contains("mul") || fn_name.contains("*") {
        writeln!(code, "%zip_out{} = OpFMul %{} %zip_a{} %zip_b{}", idx, ty, idx, idx).unwrap();
        format!("%zip_out{}", idx)
    } else if fn_name.contains("sub") || fn_name.contains("-") {
        writeln!(code, "%zip_out{} = OpFSub %{} %zip_a{} %zip_b{}", idx, ty, idx, idx).unwrap();
        format!("%zip_out{}", idx)
    } else {
        // Default to multiply
        writeln!(code, "%zip_out{} = OpFMul %{} %zip_a{} %zip_b{}", idx, ty, idx, idx).unwrap();
        format!("%zip_out{}", idx)
    };

    // Store output
    writeln!(code, "%zip_out_ptr{} = OpAccessChain %ptr_storage_{} %buffer_out0 %gid", idx, ty).unwrap();
    writeln!(code, "OpStore %zip_out_ptr{} {}", idx, result).unwrap();

    Ok(())
}

/// Generate SPIR-V for reduce operation.
fn generate_reduce_op(
    code: &mut String,
    op: ReduceOp,
    dtype: DType,
    _params: &KernelParams,
    idx: usize,
) -> GpuResult<()> {
    let ty = dtype_to_spirv_type(dtype);

    writeln!(code, "; Parallel reduction using subgroup operations").unwrap();

    // Load input
    writeln!(code, "%red_ptr{} = OpAccessChain %ptr_storage_{} %buffer_in0 %gid", idx, ty).unwrap();
    writeln!(code, "%red_in{} = OpLoad %{} %red_ptr{}", idx, ty, idx).unwrap();

    // Use subgroup reduction if available
    let result = match op {
        ReduceOp::Sum => {
            writeln!(code, "%red_out{} = OpGroupNonUniformFAdd %{} %uint_3 Reduce %red_in{}", idx, ty, idx).unwrap();
            format!("%red_out{}", idx)
        }
        ReduceOp::Prod => {
            writeln!(code, "%red_out{} = OpGroupNonUniformFMul %{} %uint_3 Reduce %red_in{}", idx, ty, idx).unwrap();
            format!("%red_out{}", idx)
        }
        ReduceOp::Min => {
            writeln!(code, "%red_out{} = OpGroupNonUniformFMin %{} %uint_3 Reduce %red_in{}", idx, ty, idx).unwrap();
            format!("%red_out{}", idx)
        }
        ReduceOp::Max => {
            writeln!(code, "%red_out{} = OpGroupNonUniformFMax %{} %uint_3 Reduce %red_in{}", idx, ty, idx).unwrap();
            format!("%red_out{}", idx)
        }
        _ => {
            writeln!(code, "; Unsupported reduce op: {:?}", op).unwrap();
            format!("%red_in{}", idx)
        }
    };

    // Store output (first thread only for full reduction)
    writeln!(code, "%is_first = OpIEqual %bool %gid %uint_0").unwrap();
    writeln!(code, "OpSelectionMerge %reduce_merge{} None", idx).unwrap();
    writeln!(code, "OpBranchConditional %is_first %reduce_store{} %reduce_merge{}", idx, idx).unwrap();
    writeln!(code, "%reduce_store{} = OpLabel", idx).unwrap();
    writeln!(code, "%red_out_ptr{} = OpAccessChain %ptr_storage_{} %buffer_out0 %uint_0", idx, ty).unwrap();
    writeln!(code, "OpStore %red_out_ptr{} {}", idx, result).unwrap();
    writeln!(code, "OpBranch %reduce_merge{}", idx).unwrap();
    writeln!(code, "%reduce_merge{} = OpLabel", idx).unwrap();

    Ok(())
}

/// Convert DType to SPIR-V type name.
fn dtype_to_spirv_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::Int8 | DType::UInt8 => "uint",
        DType::Int16 | DType::UInt16 => "uint",
        DType::Int32 => "int",
        DType::UInt32 => "uint",
        DType::Int64 | DType::UInt64 => "uint64",
        DType::Float16 | DType::BFloat16 => "float",
        DType::Float32 => "float",
        DType::Float64 => "double",
        DType::Complex64 | DType::Complex128 => "float",
    }
}

/// Generate an elementwise kernel.
pub fn generate_elementwise_kernel(
    params: &KernelParams,
    op_code: &str,
    dtype: DType,
) -> GpuResult<String> {
    let ty = dtype_to_spirv_type(dtype);
    let mut code = String::new();

    writeln!(code, "; Elementwise kernel: {}", op_code).unwrap();
    writeln!(code, "%elem_ptr = OpAccessChain %ptr_storage_{} %buffer_in0 %gid", ty).unwrap();
    writeln!(code, "%elem_in = OpLoad %{} %elem_ptr", ty).unwrap();
    writeln!(code, "%elem_out = {} %{} %elem_in", op_code, ty).unwrap();
    writeln!(code, "%elem_out_ptr = OpAccessChain %ptr_storage_{} %buffer_out0 %gid", ty).unwrap();
    writeln!(code, "OpStore %elem_out_ptr %elem_out").unwrap();

    Ok(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceInfo;

    fn make_spirv_device() -> DeviceInfo {
        let mut device = DeviceInfo::mock();
        device.kind = crate::device::DeviceKind::Spirv;
        device.compute_capability = (1, 2); // Vulkan 1.2
        device
    }

    #[test]
    fn test_spirv_module_header() {
        let device = make_spirv_device();
        let header = generate_module_header("test_kernel", &device);
        assert!(header.contains("SPIR-V"));
        assert!(header.contains("Version: 1.5"));
        assert!(header.contains("OpCapability Shader"));
    }

    #[test]
    fn test_spirv_dtype_conversion() {
        assert_eq!(dtype_to_spirv_type(DType::Float32), "float");
        assert_eq!(dtype_to_spirv_type(DType::Float64), "double");
        assert_eq!(dtype_to_spirv_type(DType::Int32), "int");
    }
}
