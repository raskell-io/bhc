//! Tensor IR to GPU lowering.
//!
//! This module implements the pipeline to lower Tensor IR kernels to GPU code.
//! It analyzes tensor operations for GPU suitability, generates optimal grid/block
//! configurations, and compiles Tensor IR to PTX or AMDGCN.
//!
//! # Overview
//!
//! The lowering pipeline:
//! 1. Analyze kernel operations for GPU suitability
//! 2. Determine optimal launch configuration (grid/block dimensions)
//! 3. Generate device code (PTX/AMDGCN)
//! 4. Optionally cache compiled modules
//!
//! # Usage
//!
//! ```rust,ignore
//! use bhc_gpu::lower::{GpuLowering, LoweringConfig};
//! use bhc_tensor_ir::Kernel;
//!
//! let config = LoweringConfig::default();
//! let lowering = GpuLowering::new(config);
//!
//! let compiled = lowering.lower_kernel(&kernel, &device_info)?;
//! ```

use crate::codegen::{amdgcn, metal, ptx, spirv, wgsl};
use crate::device::{DeviceInfo, DeviceKind};
use crate::kernel::{CompiledModule, LaunchConfig};
use crate::{GpuError, GpuResult};
use bhc_index::Idx;
use bhc_tensor_ir::{DType, Kernel, KernelBody, TensorOp};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// A compiled kernel ready for execution.
#[derive(Debug)]
pub struct CompiledKernel {
    /// The compiled module containing the kernel code.
    pub module: Arc<CompiledModule>,
    /// The kernel function name within the module.
    pub function_name: String,
    /// The recommended launch configuration.
    pub launch_config: LaunchConfig,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for GPU lowering.
#[derive(Clone, Debug)]
pub struct LoweringConfig {
    /// Maximum threads per block (default: 256).
    pub max_threads_per_block: u32,
    /// Preferred threads per block for simple kernels.
    pub default_threads_per_block: u32,
    /// Minimum elements per thread for efficiency.
    pub min_elements_per_thread: usize,
    /// Enable shared memory optimizations.
    pub use_shared_memory: bool,
    /// Maximum shared memory per block in bytes.
    pub max_shared_memory: usize,
    /// Enable kernel caching.
    pub enable_cache: bool,
    /// Verbose code generation (includes comments).
    pub verbose: bool,
}

impl Default for LoweringConfig {
    fn default() -> Self {
        Self {
            max_threads_per_block: 256,
            default_threads_per_block: 256,
            min_elements_per_thread: 4,
            use_shared_memory: true,
            max_shared_memory: 48 * 1024, // 48KB typical
            enable_cache: true,
            verbose: false,
        }
    }
}

// ============================================================================
// GPU Suitability Analysis
// ============================================================================

/// Analysis result for a kernel's GPU suitability.
#[derive(Clone, Debug)]
pub struct GpuSuitability {
    /// Whether this kernel is suitable for GPU execution.
    pub suitable: bool,
    /// Estimated parallelism (number of independent work items).
    pub parallelism: Option<usize>,
    /// Operations that can be executed on GPU.
    pub gpu_ops: Vec<GpuOpInfo>,
    /// Operations that would need CPU fallback.
    pub cpu_fallback_ops: Vec<String>,
    /// Estimated memory bandwidth requirement.
    pub memory_bandwidth: MemoryBandwidth,
    /// Whether the kernel is compute-bound or memory-bound.
    pub bottleneck: Bottleneck,
}

/// Information about a GPU-compatible operation.
#[derive(Clone, Debug)]
pub struct GpuOpInfo {
    /// Operation type.
    pub op_type: GpuOpType,
    /// Estimated FLOPS for this operation.
    pub estimated_flops: usize,
    /// Estimated memory accesses (bytes).
    pub memory_bytes: usize,
}

/// Types of GPU operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuOpType {
    /// Element-wise unary operation.
    UnaryElementwise,
    /// Element-wise binary operation.
    BinaryElementwise,
    /// Map operation.
    Map,
    /// ZipWith operation.
    ZipWith,
    /// Reduction operation.
    Reduce,
    /// Full reduction to scalar.
    ReduceAll,
    /// Matrix multiplication.
    MatMul,
    /// Batch matrix multiplication.
    BatchMatMul,
    /// Convolution.
    Convolution,
    /// Not GPU-suitable.
    NotSupported,
}

/// Memory bandwidth estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryBandwidth {
    /// Low bandwidth requirement.
    Low,
    /// Medium bandwidth requirement.
    Medium,
    /// High bandwidth requirement.
    High,
}

/// Performance bottleneck type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bottleneck {
    /// Compute-bound (arithmetic intensity high).
    Compute,
    /// Memory-bound (arithmetic intensity low).
    Memory,
    /// Latency-bound (small operations).
    Latency,
}

/// Analyze a kernel for GPU suitability.
pub fn analyze_kernel(kernel: &Kernel) -> GpuSuitability {
    let mut gpu_ops = Vec::new();
    let mut cpu_fallback_ops = Vec::new();
    let mut total_parallelism: Option<usize> = None;
    let mut total_flops = 0usize;
    let mut total_memory = 0usize;

    // Analyze kernel body
    match &kernel.body {
        KernelBody::Fused(ops) => {
            for op in ops {
                let info = analyze_op(op);
                if info.op_type == GpuOpType::NotSupported {
                    cpu_fallback_ops.push(format!("{:?}", op));
                } else {
                    total_flops += info.estimated_flops;
                    total_memory += info.memory_bytes;
                    gpu_ops.push(info);
                }
            }

            // Estimate parallelism from inputs/outputs
            for input in &kernel.inputs {
                if let Some(n) = input.meta.shape.num_elements() {
                    total_parallelism = Some(total_parallelism.map_or(n, |p| p.max(n)));
                }
            }
        }
        KernelBody::LoopNest(_nest) => {
            // Loop nests need more sophisticated analysis
            // For now, assume they're suitable if the kernel exists
            gpu_ops.push(GpuOpInfo {
                op_type: GpuOpType::Map,
                estimated_flops: 1000,
                memory_bytes: 1000,
            });
        }
    }

    // Determine memory bandwidth category
    let memory_bandwidth = if total_memory > 1024 * 1024 * 100 {
        MemoryBandwidth::High
    } else if total_memory > 1024 * 1024 {
        MemoryBandwidth::Medium
    } else {
        MemoryBandwidth::Low
    };

    // Determine bottleneck based on arithmetic intensity
    // Arithmetic intensity = FLOPS / Memory bytes
    let arithmetic_intensity = if total_memory > 0 {
        total_flops as f64 / total_memory as f64
    } else {
        0.0
    };

    let bottleneck = if arithmetic_intensity > 10.0 {
        Bottleneck::Compute
    } else if total_parallelism.map_or(false, |p| p < 1000) {
        Bottleneck::Latency
    } else {
        Bottleneck::Memory
    };

    let suitable = cpu_fallback_ops.is_empty() && total_parallelism.map_or(false, |p| p >= 1000);

    GpuSuitability {
        suitable,
        parallelism: total_parallelism,
        gpu_ops,
        cpu_fallback_ops,
        memory_bandwidth,
        bottleneck,
    }
}

/// Analyze a single tensor operation.
fn analyze_op(op: &TensorOp) -> GpuOpInfo {
    match op {
        TensorOp::Unary(_, ref_) => GpuOpInfo {
            op_type: GpuOpType::UnaryElementwise,
            estimated_flops: ref_.meta.shape.num_elements().unwrap_or(1),
            memory_bytes: ref_.meta.size_bytes().unwrap_or(0) * 2, // read + write
        },
        TensorOp::Binary(_, ref1, ref2) => {
            let n1 = ref1.meta.shape.num_elements().unwrap_or(1);
            let n2 = ref2.meta.shape.num_elements().unwrap_or(1);
            GpuOpInfo {
                op_type: GpuOpType::BinaryElementwise,
                estimated_flops: n1.max(n2),
                memory_bytes: ref1.meta.size_bytes().unwrap_or(0)
                    + ref2.meta.size_bytes().unwrap_or(0)
                    + n1.max(n2) * ref1.meta.dtype.size_bytes(),
            }
        }
        TensorOp::Map(_, ref_) => GpuOpInfo {
            op_type: GpuOpType::Map,
            estimated_flops: ref_.meta.shape.num_elements().unwrap_or(1),
            memory_bytes: ref_.meta.size_bytes().unwrap_or(0) * 2,
        },
        TensorOp::ZipWith(_, ref1, ref2) => {
            let n1 = ref1.meta.shape.num_elements().unwrap_or(1);
            let n2 = ref2.meta.shape.num_elements().unwrap_or(1);
            GpuOpInfo {
                op_type: GpuOpType::ZipWith,
                estimated_flops: n1.max(n2),
                memory_bytes: ref1.meta.size_bytes().unwrap_or(0)
                    + ref2.meta.size_bytes().unwrap_or(0)
                    + n1.max(n2) * ref1.meta.dtype.size_bytes(),
            }
        }
        TensorOp::Reduce(_, _, ref_) => GpuOpInfo {
            op_type: GpuOpType::Reduce,
            estimated_flops: ref_.meta.shape.num_elements().unwrap_or(1),
            memory_bytes: ref_.meta.size_bytes().unwrap_or(0),
        },
        TensorOp::ReduceAll(_, ref_) => GpuOpInfo {
            op_type: GpuOpType::ReduceAll,
            estimated_flops: ref_.meta.shape.num_elements().unwrap_or(1),
            memory_bytes: ref_.meta.size_bytes().unwrap_or(0),
        },
        TensorOp::MatMul(ref1, ref2) => {
            // MatMul: [M,K] x [K,N] = [M,N]
            // FLOPS = 2*M*N*K
            let shape1 = ref1.meta.shape.dims();
            let shape2 = ref2.meta.shape.dims();
            let (m, k, n) = if shape1.len() >= 2 && shape2.len() >= 2 {
                let m = shape1[0].static_value().unwrap_or(1);
                let k = shape1[1].static_value().unwrap_or(1);
                let n = shape2[1].static_value().unwrap_or(1);
                (m, k, n)
            } else {
                (1, 1, 1)
            };
            GpuOpInfo {
                op_type: GpuOpType::MatMul,
                estimated_flops: 2 * m * n * k,
                memory_bytes: ref1.meta.size_bytes().unwrap_or(0)
                    + ref2.meta.size_bytes().unwrap_or(0)
                    + m * n * ref1.meta.dtype.size_bytes(),
            }
        }
        TensorOp::BatchMatMul(ref1, ref2) => GpuOpInfo {
            op_type: GpuOpType::BatchMatMul,
            estimated_flops: ref1.meta.shape.num_elements().unwrap_or(1)
                * ref2.meta.shape.num_elements().unwrap_or(1),
            memory_bytes: ref1.meta.size_bytes().unwrap_or(0) + ref2.meta.size_bytes().unwrap_or(0),
        },
        TensorOp::Conv(_, ref1, ref2) => GpuOpInfo {
            op_type: GpuOpType::Convolution,
            estimated_flops: ref1.meta.shape.num_elements().unwrap_or(1)
                * ref2.meta.shape.num_elements().unwrap_or(1),
            memory_bytes: ref1.meta.size_bytes().unwrap_or(0) + ref2.meta.size_bytes().unwrap_or(0),
        },
        // Structure operations don't require computation
        TensorOp::Constant(_)
        | TensorOp::Reshape(_, _)
        | TensorOp::Slice(_, _)
        | TensorOp::Transpose(_, _)
        | TensorOp::Broadcast(_, _)
        | TensorOp::Concat(_, _)
        | TensorOp::Split(_, _, _) => GpuOpInfo {
            op_type: GpuOpType::Map, // Treat as simple memory operations
            estimated_flops: 0,
            memory_bytes: 0,
        },
        // These need special handling
        TensorOp::Fold(_, _, _)
        | TensorOp::Scan(_, _, _)
        | TensorOp::Dot(_, _)
        | TensorOp::Outer(_, _)
        | TensorOp::Gather(_, _, _)
        | TensorOp::Scatter(_, _, _, _) => GpuOpInfo {
            op_type: GpuOpType::NotSupported,
            estimated_flops: 0,
            memory_bytes: 0,
        },
    }
}

// ============================================================================
// Launch Configuration
// ============================================================================

/// Compute optimal launch configuration for a kernel.
pub fn compute_launch_config(
    kernel: &Kernel,
    device: &DeviceInfo,
    config: &LoweringConfig,
) -> LaunchConfig {
    // Get total number of elements to process
    let total_elements = kernel
        .inputs
        .iter()
        .chain(kernel.outputs.iter())
        .filter_map(|t| t.meta.shape.num_elements())
        .max()
        .unwrap_or(1);

    // Determine block size based on operation type
    let block_size = determine_block_size(kernel, device, config);

    // Calculate grid size
    let grid_size = (total_elements as u32 + block_size - 1) / block_size;

    // Cap grid size at device maximum
    let max_grid_size = device.max_grid_dim.0;
    let grid_size = grid_size.min(max_grid_size);

    // Calculate shared memory requirements
    let shared_memory = if config.use_shared_memory {
        estimate_shared_memory(kernel, block_size as usize)
    } else {
        0
    };

    LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem: shared_memory,
    }
}

/// Determine optimal block size for a kernel.
fn determine_block_size(kernel: &Kernel, device: &DeviceInfo, config: &LoweringConfig) -> u32 {
    // Check for reductions which benefit from power-of-2 block sizes
    let has_reduction = match &kernel.body {
        KernelBody::Fused(ops) => ops
            .iter()
            .any(|op| matches!(op, TensorOp::Reduce(_, _, _) | TensorOp::ReduceAll(_, _))),
        KernelBody::LoopNest(_) => false,
    };

    if has_reduction {
        // Use power-of-2 for efficient parallel reductions
        256
    } else {
        // Use default or device maximum
        config
            .default_threads_per_block
            .min(device.max_threads_per_block)
    }
}

/// Estimate shared memory requirements.
fn estimate_shared_memory(kernel: &Kernel, block_size: usize) -> usize {
    let dtype_size = kernel
        .inputs
        .first()
        .map(|t| t.meta.dtype.size_bytes())
        .unwrap_or(4);

    // Check for operations that benefit from shared memory
    let needs_shared = match &kernel.body {
        KernelBody::Fused(ops) => ops.iter().any(|op| {
            matches!(
                op,
                TensorOp::Reduce(_, _, _) | TensorOp::ReduceAll(_, _) | TensorOp::MatMul(_, _)
            )
        }),
        KernelBody::LoopNest(_) => false,
    };

    if needs_shared {
        // Allocate shared memory for one element per thread
        block_size * dtype_size
    } else {
        0
    }
}

// ============================================================================
// GPU Lowering
// ============================================================================

/// GPU lowering context.
pub struct GpuLowering {
    /// Configuration.
    config: LoweringConfig,
    /// Module cache (kernel hash -> compiled module).
    cache: parking_lot::RwLock<FxHashMap<u64, Arc<CompiledModule>>>,
}

impl GpuLowering {
    /// Create a new GPU lowering context.
    #[must_use]
    pub fn new(config: LoweringConfig) -> Self {
        Self {
            config,
            cache: parking_lot::RwLock::new(FxHashMap::default()),
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(LoweringConfig::default())
    }

    /// Lower a Tensor IR kernel to GPU code.
    pub fn lower_kernel(&self, kernel: &Kernel, device: &DeviceInfo) -> GpuResult<CompiledKernel> {
        // Check suitability
        let suitability = analyze_kernel(kernel);
        if !suitability.suitable {
            return Err(GpuError::NotSupported(format!(
                "Kernel not suitable for GPU: {:?}",
                suitability.cpu_fallback_ops
            )));
        }

        // Compute launch configuration
        let launch_config = compute_launch_config(kernel, device, &self.config);

        // Check cache
        let cache_key = self.compute_cache_key(kernel, device);
        if self.config.enable_cache {
            if let Some(module) = self.cache.read().get(&cache_key) {
                return Ok(CompiledKernel {
                    module: Arc::clone(module),
                    function_name: kernel_function_name(kernel),
                    launch_config,
                });
            }
        }

        // Generate code based on device type
        let code = match device.kind {
            DeviceKind::Cuda => self.generate_ptx(kernel, device, &launch_config)?,
            DeviceKind::Rocm => self.generate_amdgcn(kernel, device, &launch_config)?,
            DeviceKind::Spirv => self.generate_spirv(kernel, device, &launch_config)?,
            DeviceKind::Metal => self.generate_metal(kernel, device, &launch_config)?,
            DeviceKind::WebGpu => self.generate_wgsl(kernel, device, &launch_config)?,
            DeviceKind::Mock => self.generate_mock(kernel)?,
        };

        // Create compiled module
        let module = Arc::new(CompiledModule::from_text(
            kernel_function_name(kernel),
            code,
            device.arch_name(),
        ));

        // Cache the module
        if self.config.enable_cache {
            self.cache.write().insert(cache_key, Arc::clone(&module));
        }

        Ok(CompiledKernel {
            module,
            function_name: kernel_function_name(kernel),
            launch_config,
        })
    }

    /// Generate PTX code for NVIDIA GPUs.
    fn generate_ptx(
        &self,
        kernel: &Kernel,
        device: &DeviceInfo,
        _launch_config: &LaunchConfig,
    ) -> GpuResult<String> {
        // Use the compile_kernel function which handles all operation types
        let module = ptx::compile_kernel(kernel, device)?;
        // Convert Vec<u8> to String (PTX is UTF-8 text)
        String::from_utf8(module.code)
            .map_err(|e| GpuError::CompilationError(format!("Invalid UTF-8 in PTX: {}", e)))
    }

    /// Generate AMDGCN code for AMD GPUs.
    fn generate_amdgcn(
        &self,
        kernel: &Kernel,
        device: &DeviceInfo,
        _launch_config: &LaunchConfig,
    ) -> GpuResult<String> {
        // Use the compile_kernel function which handles all operation types
        let module = amdgcn::compile_kernel(kernel, device)?;
        // Convert Vec<u8> to String (AMDGCN is UTF-8 text)
        String::from_utf8(module.code)
            .map_err(|e| GpuError::CompilationError(format!("Invalid UTF-8 in AMDGCN: {}", e)))
    }

    /// Generate mock code for testing.
    fn generate_mock(&self, kernel: &Kernel) -> GpuResult<String> {
        Ok(format!(
            "; Mock GPU kernel: {}\n; Inputs: {}\n; Outputs: {}\n",
            kernel_function_name(kernel),
            kernel.inputs.len(),
            kernel.outputs.len()
        ))
    }

    /// Generate SPIR-V code for Vulkan/OpenCL.
    fn generate_spirv(
        &self,
        kernel: &Kernel,
        device: &DeviceInfo,
        _launch_config: &LaunchConfig,
    ) -> GpuResult<String> {
        let module = spirv::compile_kernel(kernel, device)?;
        String::from_utf8(module.code)
            .map_err(|e| GpuError::CompilationError(format!("Invalid UTF-8 in SPIR-V: {}", e)))
    }

    /// Generate Metal Shading Language code for Apple GPUs.
    fn generate_metal(
        &self,
        kernel: &Kernel,
        device: &DeviceInfo,
        _launch_config: &LaunchConfig,
    ) -> GpuResult<String> {
        let module = metal::compile_kernel(kernel, device)?;
        String::from_utf8(module.code)
            .map_err(|e| GpuError::CompilationError(format!("Invalid UTF-8 in Metal: {}", e)))
    }

    /// Generate WGSL code for WebGPU.
    fn generate_wgsl(
        &self,
        kernel: &Kernel,
        device: &DeviceInfo,
        _launch_config: &LaunchConfig,
    ) -> GpuResult<String> {
        let module = wgsl::compile_kernel(kernel, device)?;
        String::from_utf8(module.code)
            .map_err(|e| GpuError::CompilationError(format!("Invalid UTF-8 in WGSL: {}", e)))
    }

    /// Compute a cache key for a kernel.
    fn compute_cache_key(&self, kernel: &Kernel, device: &DeviceInfo) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        kernel.id.hash(&mut hasher);
        device.id.hash(&mut hasher);
        device.kind.hash(&mut hasher);
        hasher.finish()
    }

    /// Clear the module cache.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read();
        let entries = cache.len();
        let total_size: usize = cache.values().map(|m| m.code.len()).sum();
        (entries, total_size)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate a function name for a kernel.
fn kernel_function_name(kernel: &Kernel) -> String {
    format!("bhc_kernel_{}", kernel.id.index())
}

/// Get the data type from a tensor operation.
fn get_op_dtype(op: &TensorOp, kernel: &Kernel) -> DType {
    match op {
        TensorOp::Unary(_, ref_)
        | TensorOp::Map(_, ref_)
        | TensorOp::Reduce(_, _, ref_)
        | TensorOp::ReduceAll(_, ref_) => ref_.meta.dtype,
        TensorOp::Binary(_, ref1, _)
        | TensorOp::ZipWith(_, ref1, _)
        | TensorOp::MatMul(ref1, _)
        | TensorOp::BatchMatMul(ref1, _) => ref1.meta.dtype,
        _ => kernel
            .inputs
            .first()
            .map(|t| t.meta.dtype)
            .unwrap_or(DType::Float32),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_tensor_ir::{
        FusionInfo, KernelId, Layout, MapFn, Shape, Strides, TensorId, TensorMeta, TensorRef,
    };

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
        DeviceInfo::mock()
    }

    #[test]
    fn test_analyze_kernel() {
        let kernel = make_test_kernel();
        let suitability = analyze_kernel(&kernel);
        assert!(suitability.suitable);
        assert!(suitability.parallelism.is_some());
    }

    #[test]
    fn test_compute_launch_config() {
        let kernel = make_test_kernel();
        let device = make_test_device();
        let config = LoweringConfig::default();

        let launch = compute_launch_config(&kernel, &device, &config);
        assert!(launch.block_dim.0 > 0);
        assert!(launch.grid_dim.0 > 0);
    }

    #[test]
    fn test_gpu_lowering_mock() {
        let kernel = make_test_kernel();
        let device = make_test_device();
        let lowering = GpuLowering::with_defaults();

        let result = lowering.lower_kernel(&kernel, &device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache() {
        let kernel = make_test_kernel();
        let device = make_test_device();
        let lowering = GpuLowering::with_defaults();

        // First compilation
        let _ = lowering.lower_kernel(&kernel, &device);
        let (entries1, _) = lowering.cache_stats();
        assert_eq!(entries1, 1);

        // Second compilation should use cache
        let _ = lowering.lower_kernel(&kernel, &device);
        let (entries2, _) = lowering.cache_stats();
        assert_eq!(entries2, 1);

        // Clear cache
        lowering.clear_cache();
        let (entries3, _) = lowering.cache_stats();
        assert_eq!(entries3, 0);
    }
}
