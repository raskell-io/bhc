//! AMDGCN code generation for AMD GPUs.
//!
//! This module generates AMDGCN assembly from Tensor IR kernels for
//! AMD GPUs (GCN, RDNA, CDNA architectures).
//!
//! # AMDGCN Overview
//!
//! AMDGCN is AMD's GPU ISA with:
//! - Scalar (SALU) and Vector (VALU) units
//! - 64-wide wavefronts (32-wide on RDNA)
//! - LDS (Local Data Share) = shared memory
//! - Various memory operations (flat, global, buffer)
//!
//! # Architecture Targets
//!
//! - **gfx900**: Vega 10 (MI25)
//! - **gfx906**: Vega 20 (MI50/MI60)
//! - **gfx908**: CDNA 1 (MI100)
//! - **gfx90a**: CDNA 2 (MI200)
//! - **gfx1030**: RDNA 2
//! - **gfx1100**: RDNA 3

use super::KernelParams;
use crate::device::DeviceInfo;
use crate::kernel::CompiledModule;
use crate::GpuResult;
use bhc_tensor_ir::{BinaryOp, DType, Kernel, KernelBody, UnaryOp};
use std::fmt::Write;

/// Generate AMDGCN module header.
pub fn generate_module_header(name: &str, device: &DeviceInfo) -> String {
    let arch = device.arch_name();
    format!(
        "; AMDGCN assembly for {}\n\
         ; Target: {}\n\
         ; Device: {}\n\
         \n\
         .amdgcn_target \"{}\"\n\
         .amdhsa_code_object_version 4\n\n",
        name, arch, device.name, arch
    )
}

/// Compile a Tensor IR kernel to AMDGCN.
pub fn compile_kernel(kernel: &Kernel, device: &DeviceInfo) -> GpuResult<CompiledModule> {
    let params = KernelParams::from_kernel(kernel);
    let mut code = generate_module_header(&params.name, device);

    // Generate kernel metadata
    generate_kernel_metadata(&mut code, &params, device)?;

    // Generate kernel code
    generate_kernel_code(&mut code, &params, kernel, device)?;

    let mut module = CompiledModule::from_text(params.name.clone(), code, device.arch_name());
    module.add_entry_point(params.name);

    Ok(module)
}

/// Generate kernel metadata for HSA.
fn generate_kernel_metadata(
    code: &mut String,
    params: &KernelParams,
    device: &DeviceInfo,
) -> GpuResult<()> {
    writeln!(code, ".amdhsa_kernel {}", params.name).unwrap();
    writeln!(
        code,
        "    .amdhsa_group_segment_fixed_size {}",
        params.shared_memory
    )
    .unwrap();
    writeln!(code, "    .amdhsa_private_segment_fixed_size 0").unwrap();
    writeln!(
        code,
        "    .amdhsa_kernarg_size {}",
        (params.inputs.len() + params.outputs.len()) * 8
    )
    .unwrap();
    writeln!(code, "    .amdhsa_user_sgpr_kernarg_segment_ptr 1").unwrap();

    // Wavefront size
    let wavefront_size =
        if device.arch_name().starts_with("gfx10") || device.arch_name().starts_with("gfx11") {
            32 // RDNA
        } else {
            64 // GCN/CDNA
        };
    writeln!(
        code,
        "    .amdhsa_wavefront_size32 {}",
        wavefront_size == 32
    )
    .unwrap();

    writeln!(code, ".end_amdhsa_kernel").unwrap();
    writeln!(code).unwrap();

    Ok(())
}

/// Generate the kernel code section.
fn generate_kernel_code(
    code: &mut String,
    params: &KernelParams,
    kernel: &Kernel,
    device: &DeviceInfo,
) -> GpuResult<()> {
    writeln!(code, ".text").unwrap();
    writeln!(code, ".globl {}", params.name).unwrap();
    writeln!(code, ".type {},@function", params.name).unwrap();
    writeln!(code, "{}:", params.name).unwrap();

    // Determine wavefront size
    let wave_size =
        if device.arch_name().starts_with("gfx10") || device.arch_name().starts_with("gfx11") {
            32
        } else {
            64
        };

    // Load kernel arguments
    writeln!(code, "    ; Load kernel arguments").unwrap();
    writeln!(
        code,
        "    s_load_dwordx2 s[0:1], s[4:5], 0x0    ; Load first arg pointer"
    )
    .unwrap();

    // Get thread ID
    writeln!(code, "    ; Calculate global thread ID").unwrap();
    writeln!(
        code,
        "    v_mov_b32 v0, s6                      ; workgroup_id_x"
    )
    .unwrap();
    writeln!(
        code,
        "    v_lshlrev_b32 v0, {}, v0            ; * workgroup_size",
        (params.block_size as f64).log2() as u32
    )
    .unwrap();
    writeln!(
        code,
        "    v_add_u32 v0, v0, v1                  ; + local_id_x"
    )
    .unwrap();

    // Generate kernel body
    match &kernel.body {
        KernelBody::Fused(ops) => {
            generate_fused_ops_amd(code, ops, params)?;
        }
        KernelBody::LoopNest(_nest) => {
            // TODO: Implement loop nest code generation
            writeln!(code, "    ; Loop nest (not implemented)").unwrap();
        }
    }

    // Return
    writeln!(code, "    s_endpgm").unwrap();
    writeln!(code).unwrap();

    Ok(())
}

/// Generate code for fused operations on AMD.
fn generate_fused_ops_amd(
    code: &mut String,
    ops: &[bhc_tensor_ir::TensorOp],
    _params: &KernelParams,
) -> GpuResult<()> {
    writeln!(code, "    ; Fused operations").unwrap();

    for (i, op) in ops.iter().enumerate() {
        match op {
            bhc_tensor_ir::TensorOp::Unary(unary_op, input) => {
                let dtype = input.meta.dtype;
                writeln!(code, "    ; Unary: {:?}", unary_op).unwrap();
                generate_unary_op_amd(code, *unary_op, dtype, i)?;
            }
            bhc_tensor_ir::TensorOp::Binary(binary_op, left, _right) => {
                let dtype = left.meta.dtype;
                writeln!(code, "    ; Binary: {:?}", binary_op).unwrap();
                generate_binary_op_amd(code, *binary_op, dtype, i)?;
            }
            _ => {
                writeln!(code, "    ; Unsupported op").unwrap();
            }
        }
    }

    Ok(())
}

/// Generate a unary operation for AMD.
fn generate_unary_op_amd(
    code: &mut String,
    op: UnaryOp,
    dtype: DType,
    idx: usize,
) -> GpuResult<()> {
    let v_in = format!("v{}", idx * 2);
    let v_out = format!("v{}", idx * 2 + 1);

    match (op, dtype) {
        (UnaryOp::Neg, DType::Float32) => {
            writeln!(code, "    v_mul_f32 {}, -1.0, {}", v_out, v_in).unwrap();
        }
        (UnaryOp::Abs, DType::Float32) => {
            writeln!(code, "    v_and_b32 {}, 0x7fffffff, {}", v_out, v_in).unwrap();
        }
        (UnaryOp::Sqrt, DType::Float32) => {
            writeln!(code, "    v_sqrt_f32 {}, {}", v_out, v_in).unwrap();
        }
        (UnaryOp::Rsqrt, DType::Float32) => {
            writeln!(code, "    v_rsq_f32 {}, {}", v_out, v_in).unwrap();
        }
        (UnaryOp::Exp, DType::Float32) => {
            writeln!(code, "    v_exp_f32 {}, {}", v_out, v_in).unwrap();
        }
        (UnaryOp::Log, DType::Float32) => {
            writeln!(code, "    v_log_f32 {}, {}", v_out, v_in).unwrap();
        }
        (UnaryOp::Sin, DType::Float32) => {
            writeln!(code, "    v_sin_f32 {}, {}", v_out, v_in).unwrap();
        }
        (UnaryOp::Cos, DType::Float32) => {
            writeln!(code, "    v_cos_f32 {}, {}", v_out, v_in).unwrap();
        }
        _ => {
            writeln!(code, "    ; Unsupported unary: {:?}", op).unwrap();
        }
    }

    Ok(())
}

/// Generate a binary operation for AMD.
fn generate_binary_op_amd(
    code: &mut String,
    op: BinaryOp,
    dtype: DType,
    idx: usize,
) -> GpuResult<()> {
    let v_a = format!("v{}", idx * 3);
    let v_b = format!("v{}", idx * 3 + 1);
    let v_out = format!("v{}", idx * 3 + 2);

    match (op, dtype) {
        (BinaryOp::Add, DType::Float32) => {
            writeln!(code, "    v_add_f32 {}, {}, {}", v_out, v_a, v_b).unwrap();
        }
        (BinaryOp::Sub, DType::Float32) => {
            writeln!(code, "    v_sub_f32 {}, {}, {}", v_out, v_a, v_b).unwrap();
        }
        (BinaryOp::Mul, DType::Float32) => {
            writeln!(code, "    v_mul_f32 {}, {}, {}", v_out, v_a, v_b).unwrap();
        }
        (BinaryOp::Div, DType::Float32) => {
            writeln!(code, "    v_rcp_f32 {}, {}", v_out, v_b).unwrap();
            writeln!(code, "    v_mul_f32 {}, {}, {}", v_out, v_a, v_out).unwrap();
        }
        (BinaryOp::Max, DType::Float32) => {
            writeln!(code, "    v_max_f32 {}, {}, {}", v_out, v_a, v_b).unwrap();
        }
        (BinaryOp::Min, DType::Float32) => {
            writeln!(code, "    v_min_f32 {}, {}, {}", v_out, v_a, v_b).unwrap();
        }
        (BinaryOp::Add, DType::Int32) => {
            writeln!(code, "    v_add_u32 {}, {}, {}", v_out, v_a, v_b).unwrap();
        }
        (BinaryOp::Mul, DType::Int32) => {
            writeln!(code, "    v_mul_lo_u32 {}, {}, {}", v_out, v_a, v_b).unwrap();
        }
        _ => {
            writeln!(code, "    ; Unsupported binary: {:?} {:?}", op, dtype).unwrap();
        }
    }

    Ok(())
}

/// Generate a simple elementwise kernel for AMD.
pub fn generate_elementwise_kernel(
    name: &str,
    op: BinaryOp,
    _dtype: DType, // TODO: use dtype for type-specific instructions
    device: &DeviceInfo,
) -> String {
    let arch = device.arch_name();

    format!(
        "; AMDGCN elementwise kernel: {name}\n\
         .amdgcn_target \"{arch}\"\n\
         \n\
         .amdhsa_kernel {name}\n\
             .amdhsa_group_segment_fixed_size 0\n\
             .amdhsa_private_segment_fixed_size 0\n\
             .amdhsa_kernarg_size 32\n\
             .amdhsa_user_sgpr_kernarg_segment_ptr 1\n\
         .end_amdhsa_kernel\n\
         \n\
         .text\n\
         .globl {name}\n\
         .type {name},@function\n\
         {name}:\n\
             ; Load kernel args\n\
             s_load_dwordx2 s[0:1], s[4:5], 0x0   ; a\n\
             s_load_dwordx2 s[2:3], s[4:5], 0x8   ; b\n\
             s_load_dwordx2 s[6:7], s[4:5], 0x10  ; c\n\
             s_load_dwordx2 s[8:9], s[4:5], 0x18  ; n\n\
             s_waitcnt lgkmcnt(0)\n\
             \n\
             ; Calculate global ID\n\
             v_mov_b32 v0, s12                    ; workgroup_id_x\n\
             v_lshlrev_b32 v0, 8, v0              ; * 256\n\
             v_add_u32 v0, v0, v1                 ; + local_id_x\n\
             \n\
             ; Bounds check\n\
             v_cmp_lt_u32 s[10:11], v0, s8\n\
             s_and_saveexec_b64 s[10:11], s[10:11]\n\
             \n\
             ; Calculate offset\n\
             v_lshlrev_b32 v2, 2, v0              ; * 4 bytes\n\
             \n\
             ; Load inputs\n\
             global_load_dword v3, v2, s[0:1]\n\
             global_load_dword v4, v2, s[2:3]\n\
             s_waitcnt vmcnt(0)\n\
             \n\
             ; {op_name}\n\
             {op_code}\n\
             \n\
             ; Store result\n\
             global_store_dword v2, v5, s[6:7]\n\
             \n\
             s_endpgm\n",
        op_name = format!("{:?}", op),
        op_code = match op {
            BinaryOp::Add => "v_add_f32 v5, v3, v4",
            BinaryOp::Sub => "v_sub_f32 v5, v3, v4",
            BinaryOp::Mul => "v_mul_f32 v5, v3, v4",
            BinaryOp::Div => "v_rcp_f32 v5, v4\n             v_mul_f32 v5, v3, v5",
            _ => "; unsupported op",
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_module_header() {
        let mut device = DeviceInfo::mock();
        device.kind = crate::device::DeviceKind::Rocm;
        device.compute_capability = (9, 8);

        let header = generate_module_header("test_kernel", &device);

        assert!(header.contains(".amdgcn_target"));
        assert!(header.contains("test_kernel"));
    }

    #[test]
    fn test_generate_elementwise_kernel() {
        let mut device = DeviceInfo::mock();
        device.kind = crate::device::DeviceKind::Rocm;
        device.compute_capability = (9, 8);

        let code =
            generate_elementwise_kernel("add_kernel", BinaryOp::Add, DType::Float32, &device);

        assert!(code.contains(".globl add_kernel"));
        assert!(code.contains("v_add_f32"));
        assert!(code.contains("s_endpgm"));
    }
}
