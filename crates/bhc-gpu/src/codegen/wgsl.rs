//! WebGPU Shading Language (WGSL) code generation.
//!
//! This module generates WGSL compute shaders from Tensor IR for WebGPU.
//!
//! # Overview
//!
//! WGSL is the shader language for WebGPU, designed for portability across
//! platforms. It has a Rust-like syntax with explicit memory semantics.
//!
//! # Key Features
//!
//! - Cross-platform (browsers, native via wgpu)
//! - Explicit workgroup (shared) memory
//! - Bounds-checked by default
//! - Modern syntax with explicit types

use crate::codegen::KernelParams;
use crate::device::DeviceInfo;
use crate::kernel::CompiledModule;
use crate::{GpuError, GpuResult};
use bhc_tensor_ir::{BinaryOp, DType, Kernel, KernelBody, ReduceOp, TensorOp, UnaryOp};

/// Generate WGSL module header.
fn generate_module_header(workgroup_size: u32) -> String {
    format!(
        r#"// BHC Generated WGSL Compute Shader
// WebGPU Shading Language

// Workgroup size
const WORKGROUP_SIZE: u32 = {0}u;

"#,
        workgroup_size
    )
}

/// Generate WGSL type name for a data type.
fn dtype_to_wgsl(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::Int8 | DType::Int16 | DType::Int32 => "i32",
        DType::Int64 => "i32", // WGSL doesn't have i64, use i32
        DType::UInt8 | DType::UInt16 | DType::UInt32 => "u32",
        DType::UInt64 => "u32", // WGSL doesn't have u64, use u32
        DType::Float16 => "f16",
        DType::Float32 => "f32",
        DType::Float64 => "f32", // WGSL f64 support is limited
        DType::BFloat16 => "f32",
        DType::Complex64 | DType::Complex128 => "f32", // Use f32 for component
    }
}

/// Generate WGSL unary operation.
fn generate_unary_op(op: UnaryOp, input: &str, dtype: DType) -> String {
    let ty = dtype_to_wgsl(dtype);
    match op {
        UnaryOp::Neg => format!("-{}", input),
        UnaryOp::Abs => format!("abs({})", input),
        UnaryOp::Sqrt => format!("sqrt({})", input),
        UnaryOp::Rsqrt => format!("inverseSqrt({})", input),
        UnaryOp::Exp => format!("exp({})", input),
        UnaryOp::Log => format!("log({})", input),
        UnaryOp::Sin => format!("sin({})", input),
        UnaryOp::Cos => format!("cos({})", input),
        UnaryOp::Tan => format!("tan({})", input),
        UnaryOp::Tanh => format!("tanh({})", input),
        UnaryOp::Sigmoid => format!("(1.0 / (1.0 + exp(-{})))", input),
        UnaryOp::Relu => format!("max({}, {}(0))", input, ty),
        UnaryOp::Floor => format!("floor({})", input),
        UnaryOp::Ceil => format!("ceil({})", input),
        UnaryOp::Round => format!("round({})", input),
        UnaryOp::Not => format!("!{}", input),
    }
}

/// Generate WGSL binary operation.
fn generate_binary_op(op: BinaryOp, left: &str, right: &str, dtype: DType) -> String {
    let ty = dtype_to_wgsl(dtype);
    match op {
        BinaryOp::Add => format!("({} + {})", left, right),
        BinaryOp::Sub => format!("({} - {})", left, right),
        BinaryOp::Mul => format!("({} * {})", left, right),
        BinaryOp::Div => format!("({} / {})", left, right),
        BinaryOp::Mod => format!("({} % {})", left, right),
        BinaryOp::Pow => format!("pow({}, {})", left, right),
        BinaryOp::Min => format!("min({}, {})", left, right),
        BinaryOp::Max => format!("max({}, {})", left, right),
        BinaryOp::And => format!("({} & {})", left, right),
        BinaryOp::Or => format!("({} | {})", left, right),
        BinaryOp::Xor => format!("({} ^ {})", left, right),
        BinaryOp::Shl => format!("({} << {})", left, right),
        BinaryOp::Shr => format!("({} >> {})", left, right),
        BinaryOp::Eq => format!("select({}(0), {}(1), {} == {})", ty, ty, left, right),
        BinaryOp::Ne => format!("select({}(0), {}(1), {} != {})", ty, ty, left, right),
        BinaryOp::Lt => format!("select({}(0), {}(1), {} < {})", ty, ty, left, right),
        BinaryOp::Le => format!("select({}(0), {}(1), {} <= {})", ty, ty, left, right),
        BinaryOp::Gt => format!("select({}(0), {}(1), {} > {})", ty, ty, left, right),
        BinaryOp::Ge => format!("select({}(0), {}(1), {} >= {})", ty, ty, left, right),
    }
}

/// Generate reduction identity value.
fn reduce_identity(op: ReduceOp, dtype: DType) -> String {
    let ty = dtype_to_wgsl(dtype);
    match op {
        ReduceOp::Sum | ReduceOp::Mean => format!("{}(0)", ty),
        ReduceOp::Prod => format!("{}(1)", ty),
        ReduceOp::Min => {
            if matches!(
                dtype,
                DType::Float16 | DType::Float32 | DType::Float64 | DType::BFloat16
            ) {
                "3.402823e+38".to_string() // FLT_MAX approximation
            } else {
                "2147483647i".to_string() // INT_MAX
            }
        }
        ReduceOp::Max => {
            if matches!(
                dtype,
                DType::Float16 | DType::Float32 | DType::Float64 | DType::BFloat16
            ) {
                "-3.402823e+38".to_string() // -FLT_MAX
            } else {
                "-2147483648i".to_string() // INT_MIN
            }
        }
        ReduceOp::All => "true".to_string(),
        ReduceOp::Any => "false".to_string(),
    }
}

/// Generate reduction operation.
fn reduce_op_wgsl(op: ReduceOp, acc: &str, val: &str) -> String {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => format!("{} + {}", acc, val),
        ReduceOp::Prod => format!("{} * {}", acc, val),
        ReduceOp::Min => format!("min({}, {})", acc, val),
        ReduceOp::Max => format!("max({}, {})", acc, val),
        ReduceOp::All => format!("{} && {}", acc, val),
        ReduceOp::Any => format!("{} || {}", acc, val),
    }
}

/// Generate a WGSL compute shader function for a kernel.
fn generate_compute_shader(params: &KernelParams, body: &str) -> String {
    let mut code = String::new();

    // Generate buffer bindings
    for (i, input) in params.inputs.iter().enumerate() {
        let ty = dtype_to_wgsl(input.dtype);
        code.push_str(&format!(
            "@group(0) @binding({0}) var<storage, read> {1}: array<{2}>;\n",
            i, input.name, ty
        ));
    }

    for (i, output) in params.outputs.iter().enumerate() {
        let ty = dtype_to_wgsl(output.dtype);
        let binding_idx = params.inputs.len() + i;
        code.push_str(&format!(
            "@group(0) @binding({0}) var<storage, read_write> {1}: array<{2}>;\n",
            binding_idx, output.name, ty
        ));
    }

    code.push('\n');

    // Shared memory for reductions if needed
    if params.shared_memory > 0 {
        let dtype = params
            .outputs
            .first()
            .map(|o| o.dtype)
            .unwrap_or(DType::Float32);
        let ty = dtype_to_wgsl(dtype);
        code.push_str(&format!(
            "var<workgroup> shared_data: array<{}, {}>;\n\n",
            ty, params.block_size
        ));
    }

    // Main compute function
    code.push_str(&format!(
        r#"@compute @workgroup_size({0})
fn {1}(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let gid = global_id.x;
    let lid = local_id.x;

{2}
}}
"#,
        params.block_size, params.name, body
    ));

    code
}

/// Compile a Tensor IR kernel to WGSL.
pub fn compile_kernel(kernel: &Kernel, device: &DeviceInfo) -> GpuResult<CompiledModule> {
    let params = KernelParams::from_kernel(kernel);

    // Determine workgroup size
    let workgroup_size = device.optimal_block_size_1d().min(256);

    // Generate module header
    let mut code = generate_module_header(workgroup_size);

    // Generate shader body based on kernel body
    let body = match &kernel.body {
        KernelBody::Fused(ops) => generate_fused_ops_body(ops, &params)?,
        KernelBody::LoopNest(_nest) => generate_loop_nest_body(&params)?,
    };

    // Generate the compute shader
    let shader = generate_compute_shader(&params, &body);
    code.push_str(&shader);

    let mut module = CompiledModule::from_text(params.name.clone(), code, "wgsl1.0".to_string());
    module.add_entry_point(params.name);

    Ok(module)
}

/// Generate WGSL body for fused tensor operations.
fn generate_fused_ops_body(ops: &[TensorOp], params: &KernelParams) -> GpuResult<String> {
    let mut body = String::new();

    // Bounds check
    body.push_str("    // Bounds check\n");
    body.push_str(&format!(
        "    if (gid >= arrayLength(&{})) {{\n        return;\n    }}\n\n",
        params.outputs.first().map(|o| o.name.as_str()).unwrap_or("out0")
    ));

    for (op_idx, op) in ops.iter().enumerate() {
        body.push_str(&format!("    // Operation {}\n", op_idx));
        body.push_str(&generate_op_wgsl(op, params)?);
        body.push('\n');
    }

    Ok(body)
}

/// Generate WGSL for a single tensor operation.
fn generate_op_wgsl(op: &TensorOp, params: &KernelParams) -> GpuResult<String> {
    match op {
        TensorOp::Unary(unary_op, _ref) => {
            let dtype = params.inputs.first().map(|i| i.dtype).unwrap_or(DType::Float32);
            let input_name = params.inputs.first().map(|i| i.name.as_str()).unwrap_or("in0");
            let output_name = params.outputs.first().map(|o| o.name.as_str()).unwrap_or("out0");
            let op_expr = generate_unary_op(*unary_op, &format!("{}[gid]", input_name), dtype);
            Ok(format!("    {}[gid] = {};\n", output_name, op_expr))
        }
        TensorOp::Binary(binary_op, _ref1, _ref2) => {
            let dtype = params.inputs.first().map(|i| i.dtype).unwrap_or(DType::Float32);
            let in0 = params.inputs.first().map(|i| i.name.as_str()).unwrap_or("in0");
            let in1 = params.inputs.get(1).map(|i| i.name.as_str()).unwrap_or("in1");
            let output_name = params.outputs.first().map(|o| o.name.as_str()).unwrap_or("out0");
            let op_expr = generate_binary_op(
                *binary_op,
                &format!("{}[gid]", in0),
                &format!("{}[gid]", in1),
                dtype,
            );
            Ok(format!("    {}[gid] = {};\n", output_name, op_expr))
        }
        TensorOp::Map(_map_fn, _ref) => {
            // Generic map - apply function to each element
            let input_name = params.inputs.first().map(|i| i.name.as_str()).unwrap_or("in0");
            let output_name = params.outputs.first().map(|o| o.name.as_str()).unwrap_or("out0");
            Ok(format!(
                "    {}[gid] = {}[gid] * 2.0; // TODO: Custom map function\n",
                output_name, input_name
            ))
        }
        TensorOp::ZipWith(_zip_fn, _ref1, _ref2) => {
            let in0 = params.inputs.first().map(|i| i.name.as_str()).unwrap_or("in0");
            let in1 = params.inputs.get(1).map(|i| i.name.as_str()).unwrap_or("in1");
            let output_name = params.outputs.first().map(|o| o.name.as_str()).unwrap_or("out0");
            Ok(format!(
                "    {}[gid] = {}[gid] + {}[gid]; // TODO: Custom zipWith function\n",
                output_name, in0, in1
            ))
        }
        TensorOp::Reduce(reduce_op, _axis, _ref) => {
            generate_reduce_wgsl(params, *reduce_op)
        }
        TensorOp::ReduceAll(reduce_op, _ref) => {
            generate_reduce_wgsl(params, *reduce_op)
        }
        TensorOp::MatMul(_ref1, _ref2) => {
            // Simple matrix multiply (not tiled)
            Ok(format!(
                r#"    // Matrix multiply (simple version)
    let row = gid / N;
    let col = gid % N;
    var sum = f32(0);
    for (var k = 0u; k < K; k = k + 1u) {{
        sum = sum + {}[row * K + k] * {}[k * N + col];
    }}
    {}[gid] = sum;
"#,
                params.inputs.first().map(|i| i.name.as_str()).unwrap_or("in0"),
                params.inputs.get(1).map(|i| i.name.as_str()).unwrap_or("in1"),
                params.outputs.first().map(|o| o.name.as_str()).unwrap_or("out0")
            ))
        }
        // Structure operations
        TensorOp::Reshape(_, _)
        | TensorOp::Slice(_, _)
        | TensorOp::Transpose(_, _)
        | TensorOp::Broadcast(_, _) => {
            let input_name = params.inputs.first().map(|i| i.name.as_str()).unwrap_or("in0");
            let output_name = params.outputs.first().map(|o| o.name.as_str()).unwrap_or("out0");
            Ok(format!("    {}[gid] = {}[gid];\n", output_name, input_name))
        }
        _ => Err(GpuError::NotSupported(format!(
            "Operation not supported in WGSL: {:?}",
            op
        ))),
    }
}

/// Generate WGSL code for reduction operations.
fn generate_reduce_wgsl(params: &KernelParams, reduce_op: ReduceOp) -> GpuResult<String> {
    let dtype = params.inputs.first().map(|i| i.dtype).unwrap_or(DType::Float32);
    let ty = dtype_to_wgsl(dtype);
    let input_name = params.inputs.first().map(|i| i.name.as_str()).unwrap_or("in0");
    let output_name = params.outputs.first().map(|o| o.name.as_str()).unwrap_or("out0");
    let reduce_expr = reduce_op_wgsl(reduce_op, "shared_data[lid]", "shared_data[lid + stride]");

    Ok(format!(
        r#"    // Load to shared memory
    shared_data[lid] = select({0}(0), {1}[gid], gid < arrayLength(&{1}));
    workgroupBarrier();

    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {{
        if (lid < stride) {{
            shared_data[lid] = {2};
        }}
        workgroupBarrier();
    }}

    // Write result
    if (lid == 0u) {{
        {3}[workgroup_id.x] = shared_data[0];
    }}
"#,
        ty, input_name, reduce_expr, output_name
    ))
}

/// Generate WGSL body for loop nest (fallback to simple copy).
fn generate_loop_nest_body(params: &KernelParams) -> GpuResult<String> {
    let input_name = params.inputs.first().map(|i| i.name.as_str()).unwrap_or("in0");
    let output_name = params.outputs.first().map(|o| o.name.as_str()).unwrap_or("out0");
    Ok(format!(
        r#"    // Loop nest (simplified)
    if (gid < arrayLength(&{0})) {{
        {1}[gid] = {0}[gid];
    }}
"#,
        input_name, output_name
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_tensor_ir::{KernelId, Layout, Shape, Strides, TensorId, TensorMeta, TensorRef, MapFn, FusionInfo};
    use bhc_intern::Symbol;

    fn make_test_kernel() -> Kernel {
        let dtype = DType::Float32;
        let shape = Shape::from_static([1024]);
        let strides = Strides::contiguous(&shape, dtype.size_bytes()).unwrap();
        let meta = TensorMeta {
            dtype,
            shape,
            strides,
            layout: Layout::Contiguous,
            alias: None,
        };

        let input_ref = TensorRef {
            id: TensorId::new(0),
            meta: meta.clone(),
        };
        let output_ref = TensorRef {
            id: TensorId::new(1),
            meta,
        };

        let map_fn = MapFn {
            name: Symbol::intern("mul_2"),
            span: bhc_span::Span::DUMMY,
        };

        Kernel {
            id: KernelId::new(0),
            name: Symbol::intern("test_kernel"),
            inputs: vec![input_ref],
            outputs: vec![output_ref],
            body: KernelBody::Fused(vec![TensorOp::Map(
                map_fn,
                TensorRef {
                    id: TensorId::new(0),
                    meta: TensorMeta::new_contiguous(DType::Float32, Shape::from_static([1024]))
                        .unwrap(),
                },
            )]),
            allocs: vec![],
            fusion_info: FusionInfo {
                original_ops: vec![],
                decisions: vec![],
                complete: true,
            },
        }
    }

    fn make_test_device() -> DeviceInfo {
        let mut device = DeviceInfo::mock();
        device.kind = crate::device::DeviceKind::WebGpu;
        device
    }

    #[test]
    fn test_compile_kernel() {
        let kernel = make_test_kernel();
        let device = make_test_device();
        let result = compile_kernel(&kernel, &device);
        assert!(result.is_ok());
        let module = result.unwrap();
        let code = String::from_utf8(module.code.clone()).unwrap();
        assert!(code.contains("@compute"));
        assert!(code.contains("@workgroup_size"));
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(dtype_to_wgsl(DType::Float32), "f32");
        assert_eq!(dtype_to_wgsl(DType::Int32), "i32");
        assert_eq!(dtype_to_wgsl(DType::UInt32), "u32");
    }

    #[test]
    fn test_unary_ops() {
        assert_eq!(generate_unary_op(UnaryOp::Neg, "x", DType::Float32), "-x");
        assert_eq!(
            generate_unary_op(UnaryOp::Sqrt, "x", DType::Float32),
            "sqrt(x)"
        );
    }

    #[test]
    fn test_binary_ops() {
        assert_eq!(
            generate_binary_op(BinaryOp::Add, "a", "b", DType::Float32),
            "(a + b)"
        );
        assert_eq!(
            generate_binary_op(BinaryOp::Mul, "a", "b", DType::Float32),
            "(a * b)"
        );
    }
}
