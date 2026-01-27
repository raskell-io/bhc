//! PTX code generation for NVIDIA GPUs.
//!
//! This module generates PTX (Parallel Thread Execution) assembly from
//! Tensor IR kernels. PTX is NVIDIA's virtual instruction set that gets
//! JIT-compiled by the CUDA driver to target-specific machine code.
//!
//! # PTX Overview
//!
//! PTX is a pseudo-assembly language with:
//! - Virtual registers (unlimited, driver allocates physical)
//! - Typed operations (f32.add, s32.mul, etc.)
//! - Special registers (%tid, %ntid, %ctaid, etc.)
//! - Memory spaces (global, shared, local, const)
//!
//! # Example PTX
//!
//! ```ptx
//! .version 7.0
//! .target sm_80
//! .address_size 64
//!
//! .visible .entry kernel_add(
//!     .param .u64 a,
//!     .param .u64 b,
//!     .param .u64 c,
//!     .param .u64 n
//! ) {
//!     .reg .u32 %tid, %idx;
//!     .reg .u64 %addr_a, %addr_b, %addr_c;
//!     .reg .f32 %val_a, %val_b, %result;
//!
//!     mov.u32 %tid, %tid.x;
//!     // ... compute and store
//!     ret;
//! }
//! ```

use super::{dtype_to_gpu_type, KernelParams};
use crate::device::DeviceInfo;
use crate::kernel::CompiledModule;
use crate::GpuResult;
use bhc_tensor_ir::{
    BinaryOp, DType, Kernel, KernelBody, MapFn, ReduceOp, TensorOp, UnaryOp, ZipFn,
};
use std::fmt::Write;

/// PTX version to target.
const PTX_VERSION: &str = "7.0";

/// Generate PTX module header.
pub fn generate_module_header(name: &str, device: &DeviceInfo) -> String {
    let arch = device.arch_name();
    format!(
        ".version {}\n\
         .target {}\n\
         .address_size 64\n\
         \n\
         // Module: {}\n\
         // Device: {}\n\n",
        PTX_VERSION, arch, name, device.name
    )
}

/// Compile a Tensor IR kernel to PTX.
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
    // Entry point signature
    writeln!(code, ".visible .entry {}(", params.name).unwrap();

    // Parameters
    let all_params: Vec<_> = params.inputs.iter().chain(params.outputs.iter()).collect();

    for (i, param) in all_params.iter().enumerate() {
        let sep = if i < all_params.len() - 1 { "," } else { "" };
        writeln!(
            code,
            "    .param .u64 ptr_{}{}  // {}",
            param.name, sep, param.name
        )
        .unwrap();
    }

    // Add size parameter
    writeln!(code, ") {{").unwrap();

    // Register declarations
    writeln!(code, "    // Register declarations").unwrap();
    writeln!(code, "    .reg .u32 %tid, %ntid, %ctaid;").unwrap();
    writeln!(code, "    .reg .u64 %idx, %n;").unwrap();
    writeln!(code, "    .reg .pred %p;").unwrap();
    writeln!(code).unwrap();

    // Thread index calculation
    writeln!(code, "    // Calculate global thread index").unwrap();
    writeln!(code, "    mov.u32 %tid, %tid.x;").unwrap();
    writeln!(code, "    mov.u32 %ntid, %ntid.x;").unwrap();
    writeln!(code, "    mov.u32 %ctaid, %ctaid.x;").unwrap();
    writeln!(code, "    mad.wide.u32 %idx, %ctaid, %ntid, %tid;").unwrap();
    writeln!(code).unwrap();

    // Generate kernel body based on operation
    match &kernel.body {
        KernelBody::Fused(ops) => {
            generate_fused_ops(code, ops, params)?;
        }
        KernelBody::LoopNest(nest) => {
            generate_loop_nest(code, nest, params)?;
        }
    }

    // Return
    writeln!(code, "    ret;").unwrap();
    writeln!(code, "}}").unwrap();

    Ok(())
}

/// Generate code for fused operations.
fn generate_fused_ops(code: &mut String, ops: &[TensorOp], params: &KernelParams) -> GpuResult<()> {
    writeln!(code, "    // Fused operations").unwrap();

    for (i, op) in ops.iter().enumerate() {
        match op {
            TensorOp::Unary(unary_op, input) => {
                let dtype = input.meta.dtype;
                let _ty = dtype_to_gpu_type(dtype);
                writeln!(code, "    // Unary: {:?}", unary_op).unwrap();
                generate_unary_op(code, *unary_op, dtype, i)?;
            }
            TensorOp::Binary(binary_op, left, _right) => {
                let dtype = left.meta.dtype;
                writeln!(code, "    // Binary: {:?}", binary_op).unwrap();
                generate_binary_op(code, *binary_op, dtype, i)?;
            }
            TensorOp::Map(map_fn, input) => {
                let dtype = input.meta.dtype;
                writeln!(code, "    // Map: {}", map_fn.name.as_str()).unwrap();
                generate_map_op(code, map_fn, dtype, params, i)?;
            }
            TensorOp::ZipWith(zip_fn, left, _right) => {
                let dtype = left.meta.dtype;
                writeln!(code, "    // ZipWith: {}", zip_fn.name.as_str()).unwrap();
                generate_zipwith_op(code, zip_fn, dtype, params, i)?;
            }
            TensorOp::Reduce(reduce_op, axis, input) => {
                let dtype = input.meta.dtype;
                writeln!(code, "    // Reduce: {:?} axis={}", reduce_op, axis.0).unwrap();
                generate_reduce_op(code, *reduce_op, dtype, params, i)?;
            }
            TensorOp::ReduceAll(reduce_op, input) => {
                let dtype = input.meta.dtype;
                writeln!(code, "    // ReduceAll: {:?}", reduce_op).unwrap();
                generate_reduce_all_op(code, *reduce_op, dtype, params, i)?;
            }
            _ => {
                writeln!(code, "    // Unsupported op").unwrap();
            }
        }
    }

    Ok(())
}

/// Generate code for a loop nest.
///
/// On GPU, the strategy is:
/// 1. Parallel loops map to the thread grid (threadIdx/blockIdx)
/// 2. Non-parallel loops become actual PTX loops
/// 3. The body is executed by each thread for its assigned iterations
fn generate_loop_nest(
    code: &mut String,
    nest: &bhc_tensor_ir::LoopNest,
    params: &KernelParams,
) -> GpuResult<()> {
    writeln!(code, "    // Loop nest code generation").unwrap();

    // Track how many parallel loops we've mapped to grid dimensions
    let mut grid_dim = 0;

    // Register declarations for loop variables
    for (i, loop_info) in nest.loops.iter().enumerate() {
        writeln!(
            code,
            "    .reg .s64 %loop_{}; // loop var: {}",
            i,
            loop_info.var.as_str()
        )
        .unwrap();
    }
    writeln!(code).unwrap();

    // Generate loop structure
    for (i, loop_info) in nest.loops.iter().enumerate() {
        if loop_info.parallel && grid_dim < 3 {
            // Map parallel loops to GPU grid dimensions
            generate_parallel_loop_header(code, i, loop_info, grid_dim)?;
            grid_dim += 1;
        } else {
            // Generate actual loop for non-parallel dimensions
            generate_sequential_loop_header(code, i, loop_info)?;
        }
    }

    // Generate the loop body
    writeln!(code, "    // Loop body").unwrap();
    generate_fused_ops(code, &nest.body, params)?;

    // Close loops in reverse order
    for (i, loop_info) in nest.loops.iter().enumerate().rev() {
        if loop_info.parallel && i < 3 {
            // Parallel loop - just a label for early exit
            writeln!(code, "loop_exit_{}:", i).unwrap();
        } else {
            // Sequential loop - close with branch back
            generate_sequential_loop_footer(code, i, loop_info)?;
        }
    }

    Ok(())
}

/// Generate header for a parallel loop mapped to GPU grid.
fn generate_parallel_loop_header(
    code: &mut String,
    loop_idx: usize,
    loop_info: &bhc_tensor_ir::LoopInfo,
    grid_dim: usize,
) -> GpuResult<()> {
    let dim_suffix = match grid_dim {
        0 => "x",
        1 => "y",
        2 => "z",
        _ => "x",
    };

    writeln!(
        code,
        "    // Parallel loop {} -> grid dimension {}",
        loop_info.var.as_str(),
        dim_suffix
    )
    .unwrap();

    // Calculate global index for this dimension
    writeln!(code, "    .reg .u32 %ptid_{}, %pntid_{}, %pctaid_{};", loop_idx, loop_idx, loop_idx).unwrap();
    writeln!(code, "    mov.u32 %ptid_{}, %tid.{};", loop_idx, dim_suffix).unwrap();
    writeln!(code, "    mov.u32 %pntid_{}, %ntid.{};", loop_idx, dim_suffix).unwrap();
    writeln!(code, "    mov.u32 %pctaid_{}, %ctaid.{};", loop_idx, dim_suffix).unwrap();
    writeln!(
        code,
        "    mad.wide.u32 %loop_{}, %pctaid_{}, %pntid_{}, %ptid_{};",
        loop_idx, loop_idx, loop_idx, loop_idx
    )
    .unwrap();

    // Add lower bound offset
    if loop_info.lower != 0 {
        writeln!(
            code,
            "    add.s64 %loop_{0}, %loop_{0}, {};",
            loop_idx, loop_info.lower
        )
        .unwrap();
    }

    // Apply step if not 1
    if loop_info.step != 1 {
        writeln!(
            code,
            "    mul.lo.s64 %loop_{0}, %loop_{0}, {};",
            loop_idx, loop_info.step
        )
        .unwrap();
    }

    // Bounds check
    writeln!(code, "    .reg .pred %pbound_{};", loop_idx).unwrap();
    match &loop_info.upper {
        bhc_tensor_ir::Dim::Fixed(n) => {
            writeln!(
                code,
                "    setp.ge.s64 %pbound_{}, %loop_{}, {};",
                loop_idx, loop_idx, n
            )
            .unwrap();
        }
        bhc_tensor_ir::Dim::Dynamic(sym) => {
            writeln!(
                code,
                "    setp.ge.s64 %pbound_{}, %loop_{}, %{}; // dynamic bound",
                loop_idx, loop_idx, sym.as_str()
            )
            .unwrap();
        }
    }
    writeln!(code, "    @%pbound_{} bra loop_exit_{};", loop_idx, loop_idx).unwrap();
    writeln!(code).unwrap();

    Ok(())
}

/// Generate header for a sequential (non-parallel) loop.
fn generate_sequential_loop_header(
    code: &mut String,
    loop_idx: usize,
    loop_info: &bhc_tensor_ir::LoopInfo,
) -> GpuResult<()> {
    writeln!(
        code,
        "    // Sequential loop {} [{}, {})",
        loop_info.var.as_str(),
        loop_info.lower,
        format_dim(&loop_info.upper)
    )
    .unwrap();

    // Initialize loop variable
    writeln!(code, "    mov.s64 %loop_{}, {};", loop_idx, loop_info.lower).unwrap();

    // Loop header label
    writeln!(code, "loop_header_{}:", loop_idx).unwrap();

    // Bounds check
    writeln!(code, "    .reg .pred %sbound_{};", loop_idx).unwrap();
    match &loop_info.upper {
        bhc_tensor_ir::Dim::Fixed(n) => {
            writeln!(
                code,
                "    setp.ge.s64 %sbound_{}, %loop_{}, {};",
                loop_idx, loop_idx, n
            )
            .unwrap();
        }
        bhc_tensor_ir::Dim::Dynamic(sym) => {
            writeln!(
                code,
                "    setp.ge.s64 %sbound_{}, %loop_{}, %{};",
                loop_idx, loop_idx, sym.as_str()
            )
            .unwrap();
        }
    }
    writeln!(code, "    @%sbound_{} bra loop_exit_{};", loop_idx, loop_idx).unwrap();
    writeln!(code).unwrap();

    Ok(())
}

/// Generate footer for a sequential loop.
fn generate_sequential_loop_footer(
    code: &mut String,
    loop_idx: usize,
    loop_info: &bhc_tensor_ir::LoopInfo,
) -> GpuResult<()> {
    // Increment loop variable
    writeln!(
        code,
        "    add.s64 %loop_{0}, %loop_{0}, {};",
        loop_idx, loop_info.step
    )
    .unwrap();

    // Branch back to header
    writeln!(code, "    bra loop_header_{};", loop_idx).unwrap();

    // Exit label
    writeln!(code, "loop_exit_{}:", loop_idx).unwrap();

    Ok(())
}

/// Format a dimension for display.
fn format_dim(dim: &bhc_tensor_ir::Dim) -> String {
    match dim {
        bhc_tensor_ir::Dim::Fixed(n) => n.to_string(),
        bhc_tensor_ir::Dim::Dynamic(sym) => sym.as_str().to_string(),
    }
}

/// Generate a unary operation.
fn generate_unary_op(code: &mut String, op: UnaryOp, dtype: DType, idx: usize) -> GpuResult<()> {
    let ty = dtype_to_gpu_type(dtype);
    let reg_in = format!("%in{}", idx);
    let reg_out = format!("%out{}", idx);

    match op {
        UnaryOp::Neg => {
            writeln!(code, "    neg.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Abs => {
            writeln!(code, "    abs.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Sqrt => {
            writeln!(code, "    sqrt.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Rsqrt => {
            writeln!(code, "    rsqrt.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Exp => {
            writeln!(code, "    ex2.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Log => {
            writeln!(code, "    lg2.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Sin => {
            writeln!(code, "    sin.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        UnaryOp::Cos => {
            writeln!(code, "    cos.approx.{} {}, {};", ty, reg_out, reg_in).unwrap();
        }
        _ => {
            writeln!(code, "    // Unsupported unary: {:?}", op).unwrap();
        }
    }

    Ok(())
}

/// Generate a binary operation.
fn generate_binary_op(code: &mut String, op: BinaryOp, dtype: DType, idx: usize) -> GpuResult<()> {
    let ty = dtype_to_gpu_type(dtype);
    let reg_a = format!("%a{}", idx);
    let reg_b = format!("%b{}", idx);
    let reg_out = format!("%out{}", idx);

    match op {
        BinaryOp::Add => {
            writeln!(code, "    add.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        BinaryOp::Sub => {
            writeln!(code, "    sub.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        BinaryOp::Mul => {
            writeln!(code, "    mul.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        BinaryOp::Div => {
            writeln!(
                code,
                "    div.approx.{} {}, {}, {};",
                ty, reg_out, reg_a, reg_b
            )
            .unwrap();
        }
        BinaryOp::Max => {
            writeln!(code, "    max.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        BinaryOp::Min => {
            writeln!(code, "    min.{} {}, {}, {};", ty, reg_out, reg_a, reg_b).unwrap();
        }
        _ => {
            writeln!(code, "    // Unsupported binary: {:?}", op).unwrap();
        }
    }

    Ok(())
}

/// Generate a map operation.
///
/// Maps a function over each element of the input tensor.
/// Common patterns: (*2), (+1), (negate), (abs), etc.
fn generate_map_op(
    code: &mut String,
    map_fn: &MapFn,
    dtype: DType,
    params: &KernelParams,
    idx: usize,
) -> GpuResult<()> {
    let ty = dtype_to_gpu_type(dtype);
    let fn_name = map_fn.name.as_str();

    // Declare registers for this operation
    writeln!(code, "    .reg .{} %map_in{}, %map_out{};", ty, idx, idx).unwrap();
    writeln!(code, "    .reg .u64 %map_addr{};", idx).unwrap();

    // Bounds check
    let size_param = params
        .inputs
        .first()
        .map(|p| &p.name)
        .unwrap_or(&"n".to_string());
    writeln!(code, "    ld.param.u64 %n, [{}];", size_param).unwrap();
    writeln!(code, "    setp.ge.u64 %p, %idx, %n;").unwrap();
    writeln!(code, "    @%p bra map_done{};", idx).unwrap();
    writeln!(code).unwrap();

    // Load input element
    let elem_size = dtype_element_size(dtype);
    if let Some(input) = params.inputs.first() {
        writeln!(code, "    // Load input element").unwrap();
        writeln!(
            code,
            "    ld.param.u64 %map_addr{}, [ptr_{}];",
            idx, input.name
        )
        .unwrap();
        writeln!(
            code,
            "    shl.b64 %map_addr{0}, %idx, {};  // idx * elem_size",
            idx,
            elem_size.trailing_zeros()
        )
        .unwrap();
        writeln!(
            code,
            "    add.u64 %map_addr{0}, %map_addr{0}, %map_addr{0};",
            idx
        )
        .unwrap();
        writeln!(
            code,
            "    ld.global.{} %map_in{}, [%map_addr{}];",
            ty, idx, idx
        )
        .unwrap();
    }
    writeln!(code).unwrap();

    // Apply the map function based on pattern matching the name
    writeln!(code, "    // Apply map function: {}", fn_name).unwrap();
    match parse_map_function(fn_name) {
        MapPattern::MulConst(c) => {
            writeln!(code, "    .reg .{} %map_const{};", ty, idx).unwrap();
            writeln!(
                code,
                "    mov.{} %map_const{}, {};",
                ty,
                idx,
                format_const(c, dtype)
            )
            .unwrap();
            writeln!(
                code,
                "    mul.{} %map_out{}, %map_in{}, %map_const{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        MapPattern::AddConst(c) => {
            writeln!(code, "    .reg .{} %map_const{};", ty, idx).unwrap();
            writeln!(
                code,
                "    mov.{} %map_const{}, {};",
                ty,
                idx,
                format_const(c, dtype)
            )
            .unwrap();
            writeln!(
                code,
                "    add.{} %map_out{}, %map_in{}, %map_const{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        MapPattern::Negate => {
            writeln!(code, "    neg.{} %map_out{}, %map_in{};", ty, idx, idx).unwrap();
        }
        MapPattern::Abs => {
            writeln!(code, "    abs.{} %map_out{}, %map_in{};", ty, idx, idx).unwrap();
        }
        MapPattern::Sqrt => {
            writeln!(
                code,
                "    sqrt.approx.{} %map_out{}, %map_in{};",
                ty, idx, idx
            )
            .unwrap();
        }
        MapPattern::Exp => {
            writeln!(
                code,
                "    ex2.approx.{} %map_out{}, %map_in{};",
                ty, idx, idx
            )
            .unwrap();
        }
        MapPattern::Log => {
            writeln!(
                code,
                "    lg2.approx.{} %map_out{}, %map_in{};",
                ty, idx, idx
            )
            .unwrap();
        }
        MapPattern::Unknown => {
            // Default: just copy input to output (identity)
            writeln!(
                code,
                "    mov.{} %map_out{}, %map_in{};  // Unknown fn, using identity",
                ty, idx, idx
            )
            .unwrap();
        }
    }
    writeln!(code).unwrap();

    // Store output element
    if let Some(output) = params.outputs.first() {
        writeln!(code, "    // Store output element").unwrap();
        writeln!(
            code,
            "    ld.param.u64 %map_addr{}, [ptr_{}];",
            idx, output.name
        )
        .unwrap();
        writeln!(
            code,
            "    shl.b64 %map_addr{0}, %idx, {};",
            idx,
            elem_size.trailing_zeros()
        )
        .unwrap();
        writeln!(
            code,
            "    add.u64 %map_addr{0}, %map_addr{0}, %map_addr{0};",
            idx
        )
        .unwrap();
        writeln!(
            code,
            "    st.global.{} [%map_addr{}], %map_out{};",
            ty, idx, idx
        )
        .unwrap();
    }

    writeln!(code, "map_done{}:", idx).unwrap();
    Ok(())
}

/// Generate a zipwith operation.
///
/// Combines two tensors element-wise using a binary function.
fn generate_zipwith_op(
    code: &mut String,
    zip_fn: &ZipFn,
    dtype: DType,
    params: &KernelParams,
    idx: usize,
) -> GpuResult<()> {
    let ty = dtype_to_gpu_type(dtype);
    let fn_name = zip_fn.name.as_str();

    // Declare registers
    writeln!(
        code,
        "    .reg .{} %zip_a{}, %zip_b{}, %zip_out{};",
        ty, idx, idx, idx
    )
    .unwrap();
    writeln!(
        code,
        "    .reg .u64 %zip_addr_a{}, %zip_addr_b{}, %zip_addr_out{};",
        idx, idx, idx
    )
    .unwrap();

    // Bounds check
    writeln!(code, "    setp.ge.u64 %p, %idx, %n;").unwrap();
    writeln!(code, "    @%p bra zip_done{};", idx).unwrap();
    writeln!(code).unwrap();

    let elem_size = dtype_element_size(dtype);

    // Load first input
    if params.inputs.len() >= 1 {
        let input_a = &params.inputs[0];
        writeln!(code, "    // Load first input").unwrap();
        writeln!(
            code,
            "    ld.param.u64 %zip_addr_a{}, [ptr_{}];",
            idx, input_a.name
        )
        .unwrap();
        writeln!(
            code,
            "    mul.wide.u32 %zip_addr_a{0}, %idx, {};",
            idx, elem_size
        )
        .unwrap();
        writeln!(
            code,
            "    add.u64 %zip_addr_a{0}, %zip_addr_a{0}, %zip_addr_a{0};",
            idx
        )
        .unwrap();
        writeln!(
            code,
            "    ld.global.{} %zip_a{}, [%zip_addr_a{}];",
            ty, idx, idx
        )
        .unwrap();
    }

    // Load second input
    if params.inputs.len() >= 2 {
        let input_b = &params.inputs[1];
        writeln!(code, "    // Load second input").unwrap();
        writeln!(
            code,
            "    ld.param.u64 %zip_addr_b{}, [ptr_{}];",
            idx, input_b.name
        )
        .unwrap();
        writeln!(
            code,
            "    mul.wide.u32 %zip_addr_b{0}, %idx, {};",
            idx, elem_size
        )
        .unwrap();
        writeln!(
            code,
            "    add.u64 %zip_addr_b{0}, %zip_addr_b{0}, %zip_addr_b{0};",
            idx
        )
        .unwrap();
        writeln!(
            code,
            "    ld.global.{} %zip_b{}, [%zip_addr_b{}];",
            ty, idx, idx
        )
        .unwrap();
    }
    writeln!(code).unwrap();

    // Apply the zip function
    writeln!(code, "    // Apply zip function: {}", fn_name).unwrap();
    match parse_zip_function(fn_name) {
        ZipPattern::Add => {
            writeln!(
                code,
                "    add.{} %zip_out{}, %zip_a{}, %zip_b{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ZipPattern::Sub => {
            writeln!(
                code,
                "    sub.{} %zip_out{}, %zip_a{}, %zip_b{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ZipPattern::Mul => {
            writeln!(
                code,
                "    mul.{} %zip_out{}, %zip_a{}, %zip_b{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ZipPattern::Div => {
            writeln!(
                code,
                "    div.approx.{} %zip_out{}, %zip_a{}, %zip_b{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ZipPattern::Max => {
            writeln!(
                code,
                "    max.{} %zip_out{}, %zip_a{}, %zip_b{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ZipPattern::Min => {
            writeln!(
                code,
                "    min.{} %zip_out{}, %zip_a{}, %zip_b{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ZipPattern::Unknown => {
            // Default: add
            writeln!(
                code,
                "    add.{} %zip_out{}, %zip_a{}, %zip_b{};  // Unknown fn, using add",
                ty, idx, idx, idx
            )
            .unwrap();
        }
    }
    writeln!(code).unwrap();

    // Store output
    if let Some(output) = params.outputs.first() {
        writeln!(code, "    // Store output").unwrap();
        writeln!(
            code,
            "    ld.param.u64 %zip_addr_out{}, [ptr_{}];",
            idx, output.name
        )
        .unwrap();
        writeln!(
            code,
            "    mul.wide.u32 %zip_addr_out{0}, %idx, {};",
            idx, elem_size
        )
        .unwrap();
        writeln!(
            code,
            "    add.u64 %zip_addr_out{0}, %zip_addr_out{0}, %zip_addr_out{0};",
            idx
        )
        .unwrap();
        writeln!(
            code,
            "    st.global.{} [%zip_addr_out{}], %zip_out{};",
            ty, idx, idx
        )
        .unwrap();
    }

    writeln!(code, "zip_done{}:", idx).unwrap();
    Ok(())
}

/// Generate a reduction operation along an axis.
///
/// Uses parallel reduction with shared memory.
fn generate_reduce_op(
    code: &mut String,
    reduce_op: ReduceOp,
    dtype: DType,
    params: &KernelParams,
    idx: usize,
) -> GpuResult<()> {
    // For axis reduction, we generate a block-based parallel reduction
    generate_parallel_reduction(code, reduce_op, dtype, params, idx, false)
}

/// Generate a full reduction to scalar.
fn generate_reduce_all_op(
    code: &mut String,
    reduce_op: ReduceOp,
    dtype: DType,
    params: &KernelParams,
    idx: usize,
) -> GpuResult<()> {
    generate_parallel_reduction(code, reduce_op, dtype, params, idx, true)
}

/// Generate parallel reduction kernel code.
///
/// This implements a tree-based parallel reduction using shared memory:
/// 1. Each thread loads one element
/// 2. Tree reduction within each block using shared memory
/// 3. Final atomic operation to combine block results
fn generate_parallel_reduction(
    code: &mut String,
    reduce_op: ReduceOp,
    dtype: DType,
    params: &KernelParams,
    idx: usize,
    _is_full: bool,
) -> GpuResult<()> {
    let ty = dtype_to_gpu_type(dtype);
    let elem_size = dtype_element_size(dtype);

    // Shared memory declaration (256 elements per block)
    writeln!(code, "    .shared .{} sdata{}[256];", ty, idx).unwrap();
    writeln!(code, "    .reg .{} %red_val{}, %red_tmp{};", ty, idx, idx).unwrap();
    writeln!(code, "    .reg .u64 %red_addr{};", idx).unwrap();
    writeln!(code, "    .reg .u32 %red_tid{}, %red_stride{};", idx, idx).unwrap();
    writeln!(code, "    .reg .pred %red_p{};", idx).unwrap();
    writeln!(code).unwrap();

    // Initialize with identity element
    let identity = reduce_identity(reduce_op, dtype);
    writeln!(code, "    // Initialize with identity element").unwrap();
    writeln!(code, "    mov.{} %red_val{}, {};", ty, idx, identity).unwrap();
    writeln!(code).unwrap();

    // Bounds check and load
    writeln!(code, "    // Load element if in bounds").unwrap();
    writeln!(code, "    setp.lt.u64 %red_p{}, %idx, %n;", idx).unwrap();
    writeln!(code, "    @!%red_p{} bra red_store{};", idx, idx).unwrap();

    if let Some(input) = params.inputs.first() {
        writeln!(
            code,
            "    ld.param.u64 %red_addr{}, [ptr_{}];",
            idx, input.name
        )
        .unwrap();
        writeln!(
            code,
            "    mul.wide.u32 %red_addr{0}, %idx, {};",
            idx, elem_size
        )
        .unwrap();
        writeln!(
            code,
            "    add.u64 %red_addr{0}, %red_addr{0}, %red_addr{0};",
            idx
        )
        .unwrap();
        writeln!(
            code,
            "    ld.global.{} %red_val{}, [%red_addr{}];",
            ty, idx, idx
        )
        .unwrap();
    }

    writeln!(code, "red_store{}:", idx).unwrap();
    writeln!(code).unwrap();

    // Store to shared memory
    writeln!(code, "    // Store to shared memory").unwrap();
    writeln!(code, "    mov.u32 %red_tid{}, %tid;", idx).unwrap();
    writeln!(
        code,
        "    mul.lo.u32 %red_stride{}, %red_tid{}, {};",
        idx, idx, elem_size
    )
    .unwrap();
    writeln!(
        code,
        "    st.shared.{} [sdata{} + %red_stride{}], %red_val{};",
        ty, idx, idx, idx
    )
    .unwrap();
    writeln!(code, "    bar.sync 0;").unwrap();
    writeln!(code).unwrap();

    // Tree reduction in shared memory
    writeln!(code, "    // Tree reduction").unwrap();
    for s in [128, 64, 32, 16, 8, 4, 2, 1] {
        writeln!(
            code,
            "    setp.lt.u32 %red_p{}, %red_tid{}, {};",
            idx, idx, s
        )
        .unwrap();
        writeln!(code, "    @!%red_p{} bra red_sync{}{};", idx, idx, s).unwrap();

        // Load neighbor value
        writeln!(
            code,
            "    add.u32 %red_stride{}, %red_tid{}, {};",
            idx, idx, s
        )
        .unwrap();
        writeln!(
            code,
            "    mul.lo.u32 %red_stride{0}, %red_stride{0}, {};",
            idx, elem_size
        )
        .unwrap();
        writeln!(
            code,
            "    ld.shared.{} %red_tmp{}, [sdata{} + %red_stride{}];",
            ty, idx, idx, idx
        )
        .unwrap();

        // Load own value
        writeln!(
            code,
            "    mul.lo.u32 %red_stride{}, %red_tid{}, {};",
            idx, idx, elem_size
        )
        .unwrap();
        writeln!(
            code,
            "    ld.shared.{} %red_val{}, [sdata{} + %red_stride{}];",
            ty, idx, idx, idx
        )
        .unwrap();

        // Combine
        generate_reduce_combine(code, reduce_op, dtype, idx)?;

        // Store back
        writeln!(
            code,
            "    st.shared.{} [sdata{} + %red_stride{}], %red_val{};",
            ty, idx, idx, idx
        )
        .unwrap();

        writeln!(code, "red_sync{}{}:", idx, s).unwrap();
        if s > 1 {
            writeln!(code, "    bar.sync 0;").unwrap();
        }
    }
    writeln!(code).unwrap();

    // Thread 0 writes final result with atomic operation
    writeln!(code, "    // Write final result").unwrap();
    writeln!(code, "    setp.eq.u32 %red_p{}, %red_tid{}, 0;", idx, idx).unwrap();
    writeln!(code, "    @!%red_p{} bra red_done{};", idx, idx).unwrap();

    if let Some(output) = params.outputs.first() {
        writeln!(
            code,
            "    ld.shared.{} %red_val{}, [sdata{}];",
            ty, idx, idx
        )
        .unwrap();
        writeln!(
            code,
            "    ld.param.u64 %red_addr{}, [ptr_{}];",
            idx, output.name
        )
        .unwrap();
        // Use atomic for combining across blocks
        generate_atomic_reduce(code, reduce_op, dtype, idx)?;
    }

    writeln!(code, "red_done{}:", idx).unwrap();

    Ok(())
}

/// Generate the combine operation for reduction.
fn generate_reduce_combine(
    code: &mut String,
    reduce_op: ReduceOp,
    dtype: DType,
    idx: usize,
) -> GpuResult<()> {
    let ty = dtype_to_gpu_type(dtype);

    match reduce_op {
        ReduceOp::Sum | ReduceOp::Mean => {
            writeln!(
                code,
                "    add.{} %red_val{}, %red_val{}, %red_tmp{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ReduceOp::Prod => {
            writeln!(
                code,
                "    mul.{} %red_val{}, %red_val{}, %red_tmp{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ReduceOp::Max => {
            writeln!(
                code,
                "    max.{} %red_val{}, %red_val{}, %red_tmp{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ReduceOp::Min => {
            writeln!(
                code,
                "    min.{} %red_val{}, %red_val{}, %red_tmp{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ReduceOp::All => {
            writeln!(
                code,
                "    and.b32 %red_val{}, %red_val{}, %red_tmp{};",
                idx, idx, idx
            )
            .unwrap();
        }
        ReduceOp::Any => {
            writeln!(
                code,
                "    or.b32 %red_val{}, %red_val{}, %red_tmp{};",
                idx, idx, idx
            )
            .unwrap();
        }
    }

    Ok(())
}

/// Generate atomic operation for final reduction across blocks.
fn generate_atomic_reduce(
    code: &mut String,
    reduce_op: ReduceOp,
    dtype: DType,
    idx: usize,
) -> GpuResult<()> {
    let ty = dtype_to_gpu_type(dtype);

    match reduce_op {
        ReduceOp::Sum | ReduceOp::Mean => {
            // PTX atomic add for floats
            if dtype == DType::Float32 || dtype == DType::Float64 {
                writeln!(
                    code,
                    "    atom.global.add.{} %red_tmp{}, [%red_addr{}], %red_val{};",
                    ty, idx, idx, idx
                )
                .unwrap();
            } else {
                writeln!(
                    code,
                    "    atom.global.add.{} %red_tmp{}, [%red_addr{}], %red_val{};",
                    ty, idx, idx, idx
                )
                .unwrap();
            }
        }
        ReduceOp::Max => {
            writeln!(
                code,
                "    atom.global.max.{} %red_tmp{}, [%red_addr{}], %red_val{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        ReduceOp::Min => {
            writeln!(
                code,
                "    atom.global.min.{} %red_tmp{}, [%red_addr{}], %red_val{};",
                ty, idx, idx, idx
            )
            .unwrap();
        }
        _ => {
            // For other ops, just store (single block case)
            writeln!(
                code,
                "    st.global.{} [%red_addr{}], %red_val{};",
                ty, idx, idx
            )
            .unwrap();
        }
    }

    Ok(())
}

/// Get the identity element for a reduction operation.
fn reduce_identity(reduce_op: ReduceOp, dtype: DType) -> String {
    match reduce_op {
        ReduceOp::Sum | ReduceOp::Mean => {
            if dtype == DType::Float32 || dtype == DType::Float64 {
                "0.0".to_string()
            } else {
                "0".to_string()
            }
        }
        ReduceOp::Prod => {
            if dtype == DType::Float32 || dtype == DType::Float64 {
                "1.0".to_string()
            } else {
                "1".to_string()
            }
        }
        ReduceOp::Max => {
            match dtype {
                DType::Float32 => "0ff7f7ffff".to_string(), // -FLT_MAX in hex
                DType::Float64 => "0dffefffffffffffff".to_string(), // -DBL_MAX
                DType::Int32 => "-2147483648".to_string(),
                DType::Int64 => "-9223372036854775808".to_string(),
                _ => "0".to_string(),
            }
        }
        ReduceOp::Min => {
            match dtype {
                DType::Float32 => "0x7f7fffff".to_string(), // FLT_MAX
                DType::Float64 => "0x7fefffffffffffff".to_string(), // DBL_MAX
                DType::Int32 => "2147483647".to_string(),
                DType::Int64 => "9223372036854775807".to_string(),
                _ => "0".to_string(),
            }
        }
        ReduceOp::All => "1".to_string(),
        ReduceOp::Any => "0".to_string(),
    }
}

/// Get element size in bytes for a dtype.
fn dtype_element_size(dtype: DType) -> u32 {
    match dtype {
        DType::Float16 | DType::BFloat16 => 2,
        DType::Float32 | DType::Int32 | DType::UInt32 => 4,
        DType::Float64 | DType::Int64 | DType::UInt64 => 8,
        _ => 4,
    }
}

/// Format a constant for PTX.
fn format_const(value: f64, dtype: DType) -> String {
    match dtype {
        DType::Float32 | DType::Float64 => format!("{:.6}", value),
        _ => format!("{}", value as i64),
    }
}

/// Patterns recognized in map functions.
#[derive(Debug)]
enum MapPattern {
    MulConst(f64),
    AddConst(f64),
    Negate,
    Abs,
    Sqrt,
    Exp,
    Log,
    Unknown,
}

/// Parse a map function name to determine the operation.
fn parse_map_function(name: &str) -> MapPattern {
    let name = name.trim();

    // Try to parse (*n) or (n*) patterns
    if name.starts_with("(*") && name.ends_with(')') {
        if let Ok(n) = name[2..name.len() - 1].trim().parse::<f64>() {
            return MapPattern::MulConst(n);
        }
    }
    if name.starts_with("(+") && name.ends_with(')') {
        if let Ok(n) = name[2..name.len() - 1].trim().parse::<f64>() {
            return MapPattern::AddConst(n);
        }
    }

    // Named functions
    match name {
        "negate" | "neg" | "(negate)" => MapPattern::Negate,
        "abs" | "(abs)" => MapPattern::Abs,
        "sqrt" | "(sqrt)" => MapPattern::Sqrt,
        "exp" | "(exp)" => MapPattern::Exp,
        "log" | "(log)" => MapPattern::Log,
        _ => MapPattern::Unknown,
    }
}

/// Patterns recognized in zip functions.
#[derive(Debug)]
enum ZipPattern {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Unknown,
}

/// Parse a zip function name to determine the operation.
fn parse_zip_function(name: &str) -> ZipPattern {
    let name = name.trim();

    match name {
        "(+)" | "add" | "+" => ZipPattern::Add,
        "(-)" | "sub" | "-" => ZipPattern::Sub,
        "(*)" | "mul" | "*" => ZipPattern::Mul,
        "(/)" | "div" | "/" => ZipPattern::Div,
        "max" | "(max)" => ZipPattern::Max,
        "min" | "(min)" => ZipPattern::Min,
        _ => ZipPattern::Unknown,
    }
}

/// Generate a simple elementwise kernel.
pub fn generate_elementwise_kernel(
    name: &str,
    op: BinaryOp,
    dtype: DType,
    device: &DeviceInfo,
) -> String {
    let ty = dtype_to_gpu_type(dtype);
    let arch = device.arch_name();

    format!(
        ".version {PTX_VERSION}\n\
         .target {arch}\n\
         .address_size 64\n\
         \n\
         .visible .entry {name}(\n\
             .param .u64 a,\n\
             .param .u64 b,\n\
             .param .u64 c,\n\
             .param .u64 n\n\
         ) {{\n\
             .reg .u32 %tid, %ntid, %ctaid;\n\
             .reg .u64 %idx, %len, %off;\n\
             .reg .{ty} %va, %vb, %vc;\n\
             .reg .pred %p;\n\
         \n\
             mov.u32 %tid, %tid.x;\n\
             mov.u32 %ntid, %ntid.x;\n\
             mov.u32 %ctaid, %ctaid.x;\n\
             mad.wide.u32 %idx, %ctaid, %ntid, %tid;\n\
         \n\
             ld.param.u64 %len, [n];\n\
             setp.ge.u64 %p, %idx, %len;\n\
             @%p bra done;\n\
         \n\
             shl.b64 %off, %idx, 2;\n\
         \n\
             ld.param.u64 %va, [a];\n\
             add.u64 %va, %va, %off;\n\
             ld.global.{ty} %va, [%va];\n\
         \n\
             ld.param.u64 %vb, [b];\n\
             add.u64 %vb, %vb, %off;\n\
             ld.global.{ty} %vb, [%vb];\n\
         \n\
             {op_code}\n\
         \n\
             ld.param.u64 %vc, [c];\n\
             add.u64 %vc, %vc, %off;\n\
             st.global.{ty} [%vc], %vc;\n\
         \n\
         done:\n\
             ret;\n\
         }}\n",
        op_code = match op {
            BinaryOp::Add => format!("add.{ty} %vc, %va, %vb;"),
            BinaryOp::Sub => format!("sub.{ty} %vc, %va, %vb;"),
            BinaryOp::Mul => format!("mul.{ty} %vc, %va, %vb;"),
            BinaryOp::Div => format!("div.approx.{ty} %vc, %va, %vb;"),
            _ => "// unsupported op".to_string(),
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_module_header() {
        let device = DeviceInfo::mock();
        let header = generate_module_header("test_kernel", &device);

        assert!(header.contains(".version 7.0"));
        assert!(header.contains(".target"));
        assert!(header.contains("test_kernel"));
    }

    #[test]
    fn test_generate_elementwise_kernel() {
        let device = DeviceInfo::mock();
        let ptx = generate_elementwise_kernel("add_kernel", BinaryOp::Add, DType::Float32, &device);

        assert!(ptx.contains(".visible .entry add_kernel"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("ret;"));
    }
}
