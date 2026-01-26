//! # BHC GPU Backend
//!
//! This crate provides GPU code generation and runtime support for BHC,
//! enabling tensor computations to execute on GPU devices with automatic
//! kernel fusion across host/device boundaries.
//!
//! ## Overview
//!
//! The GPU backend implements CUDA and ROCm support through:
//!
//! - **Device Management**: GPU enumeration, selection, and context creation
//! - **Memory Management**: Device buffers, async transfers, pinned host memory
//! - **Kernel Compilation**: PTX/AMDGCN generation from Tensor IR
//! - **Kernel Launch**: Optimal grid/block configuration and execution
//!
//! ## Architecture
//!
//! ```text
//!                           ┌─────────────────────────────┐
//!                           │     Tensor IR Kernels       │
//!                           └─────────────┬───────────────┘
//!                                         │
//!                     ┌───────────────────┴───────────────────┐
//!                     ▼                                       ▼
//!           ┌─────────────────┐                    ┌─────────────────┐
//!           │  CPU Backend    │                    │  GPU Backend    │
//!           │  (LLVM IR)      │                    │  (PTX/AMDGCN)   │
//!           └─────────────────┘                    └────────┬────────┘
//!                                                           │
//!                                     ┌─────────────────────┴─────────────────────┐
//!                                     ▼                                           ▼
//!                           ┌─────────────────┐                        ┌─────────────────┐
//!                           │  CUDA Runtime   │                        │  ROCm Runtime   │
//!                           │  (cuBLAS, etc.) │                        │  (rocBLAS, etc.)│
//!                           └─────────────────┘                        └─────────────────┘
//! ```
//!
//! ## Features
//!
//! - `cuda`: Enable NVIDIA CUDA support
//! - `rocm`: Enable AMD ROCm/HIP support
//!
//! ## Usage
//!
//! ```rust,ignore
//! use bhc_gpu::{available_devices, select_device, DeviceId};
//!
//! // List available GPUs
//! let devices = available_devices();
//! for dev in &devices {
//!     println!("{}: {} ({} MB)", dev.id, dev.name, dev.memory_total / 1024 / 1024);
//! }
//!
//! // Select a device and create context
//! let ctx = select_device(DeviceId(0))?;
//!
//! // Allocate device memory
//! let d_buf: DeviceBuffer<f32> = ctx.alloc(1024)?;
//!
//! // Transfer data
//! ctx.copy_to_device(&host_data, &mut d_buf)?;
//!
//! // Compile and launch kernel
//! let kernel = ctx.compile_kernel(&tensor_ir_kernel)?;
//! ctx.launch(&kernel, &[&d_buf])?;
//! ```
//!
//! ## M7 Exit Criteria (from ROADMAP.md)
//!
//! - GPU device enumeration and selection works
//! - Device memory allocation/deallocation works
//! - Host<->device transfers work (sync and async)
//! - Basic kernels compile and execute (CUDA at minimum)
//! - Tensor IR kernels can target GPU backend
//! - Performance competitive with manual CUDA code for matmul

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod codegen;
pub mod context;
pub mod device;
pub mod kernel;
pub mod lower;
pub mod memory;
pub mod runtime;
pub mod transfer;

use bhc_codegen::{CodegenBackend, CodegenConfig, CodegenContext, CodegenError, CodegenResult};
use bhc_target::TargetSpec;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use thiserror::Error;

pub use context::GpuContext;
pub use device::{DeviceId, DeviceInfo, DeviceKind};
pub use kernel::{GpuKernel, KernelConfig, LaunchConfig};
pub use lower::{CompiledKernel, GpuLowering, GpuSuitability, LoweringConfig};
pub use memory::{DeviceBuffer, DevicePtr};
pub use transfer::{Transfer, TransferHandle, TransferQueue};

/// Errors that can occur during GPU operations.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No GPU devices available.
    #[error("no GPU devices available")]
    NoDevicesAvailable,

    /// Invalid device ID.
    #[error("invalid device ID: {0}")]
    InvalidDevice(DeviceId),

    /// Device memory allocation failed.
    #[error("device memory allocation failed: {size} bytes")]
    AllocationFailed {
        /// Requested size in bytes.
        size: usize,
    },

    /// Out of device memory.
    #[error("out of device memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Requested size.
        requested: usize,
        /// Available memory.
        available: usize,
    },

    /// Transfer error.
    #[error("transfer error: {0}")]
    TransferError(String),

    /// Kernel compilation error.
    #[error("kernel compilation failed: {0}")]
    CompilationError(String),

    /// Kernel launch error.
    #[error("kernel launch failed: {0}")]
    LaunchError(String),

    /// Invalid kernel configuration.
    #[error("invalid kernel configuration: {0}")]
    InvalidConfig(String),

    /// Runtime error from GPU driver.
    #[error("GPU runtime error: {0}")]
    RuntimeError(String),

    /// Feature not supported on this device.
    #[error("feature not supported: {0}")]
    NotSupported(String),

    /// CUDA-specific error.
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {code} - {message}")]
    CudaError {
        /// CUDA error code.
        code: i32,
        /// Error message.
        message: String,
    },

    /// ROCm/HIP-specific error.
    #[cfg(feature = "rocm")]
    #[error("ROCm error: {code} - {message}")]
    RocmError {
        /// HIP error code.
        code: i32,
        /// Error message.
        message: String,
    },

    /// Internal error (should not occur in correct usage).
    #[error("internal GPU error: {0}")]
    Internal(String),
}

/// Result type for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

/// Enumerate all available GPU devices.
///
/// Returns a list of device information for all GPUs detected on the system.
/// The list may be empty if no GPUs are available or if GPU support is not
/// compiled in.
///
/// # Example
///
/// ```rust,ignore
/// let devices = bhc_gpu::available_devices();
/// for dev in &devices {
///     println!("GPU {}: {} ({} compute units)",
///         dev.id.0, dev.name, dev.multiprocessor_count);
/// }
/// ```
#[must_use]
pub fn available_devices() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();

    #[cfg(feature = "cuda")]
    devices.extend(runtime::cuda::enumerate_devices().unwrap_or_default());

    #[cfg(feature = "rocm")]
    devices.extend(runtime::rocm::enumerate_devices().unwrap_or_default());

    // If no GPU runtime is compiled in, return mock device for testing
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        devices.push(DeviceInfo::mock());
    }

    devices
}

/// Select a GPU device and create a context.
///
/// # Errors
///
/// Returns `GpuError::InvalidDevice` if the device ID is not valid,
/// or `GpuError::RuntimeError` if context creation fails.
///
/// # Example
///
/// ```rust,ignore
/// let ctx = bhc_gpu::select_device(DeviceId(0))?;
/// ```
pub fn select_device(id: DeviceId) -> GpuResult<GpuContext> {
    let devices = available_devices();

    if id.0 as usize >= devices.len() {
        return Err(GpuError::InvalidDevice(id));
    }

    GpuContext::new(id)
}

/// Select the default GPU device.
///
/// Selects device 0 if available, or returns an error if no devices exist.
///
/// # Errors
///
/// Returns `GpuError::NoDevicesAvailable` if no GPUs are detected.
pub fn default_device() -> GpuResult<GpuContext> {
    let devices = available_devices();

    if devices.is_empty() {
        return Err(GpuError::NoDevicesAvailable);
    }

    select_device(DeviceId(0))
}

/// The GPU code generation backend.
///
/// Implements `CodegenBackend` for GPU targets, producing PTX or AMDGCN
/// code from the tensor IR.
pub struct GpuBackend {
    /// Available devices.
    devices: Vec<DeviceInfo>,
    /// Module cache for compiled kernels.
    module_cache: parking_lot::RwLock<FxHashMap<u64, Arc<kernel::CompiledModule>>>,
}

impl GpuBackend {
    /// Create a new GPU backend.
    ///
    /// Returns `None` if no GPU devices are available.
    #[must_use]
    pub fn new() -> Option<Self> {
        let devices = available_devices();
        if devices.is_empty() {
            return None;
        }
        Some(Self {
            devices,
            module_cache: parking_lot::RwLock::new(FxHashMap::default()),
        })
    }

    /// Get the available devices.
    #[must_use]
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Check if CUDA is available.
    #[must_use]
    pub fn has_cuda(&self) -> bool {
        self.devices
            .iter()
            .any(|d| matches!(d.kind, DeviceKind::Cuda))
    }

    /// Check if ROCm is available.
    #[must_use]
    pub fn has_rocm(&self) -> bool {
        self.devices
            .iter()
            .any(|d| matches!(d.kind, DeviceKind::Rocm))
    }
}

/// GPU codegen context.
pub struct GpuCodegenContext {
    config: CodegenConfig,
    device: DeviceInfo,
}

/// GPU codegen module.
pub struct GpuCodegenModule {
    name: String,
    code: String,
    device_kind: DeviceKind,
}

impl bhc_codegen::CodegenModule for GpuCodegenModule {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify(&self) -> CodegenResult<()> {
        // Placeholder: would verify GPU module
        Ok(())
    }

    fn optimize(&mut self, _level: bhc_session::OptLevel) -> CodegenResult<()> {
        // Placeholder: would run GPU-specific optimizations
        Ok(())
    }

    fn write_to_file(
        &self,
        path: &std::path::Path,
        _output_type: bhc_codegen::CodegenOutputType,
    ) -> CodegenResult<()> {
        std::fs::write(path, &self.code).map_err(|e| CodegenError::OutputError {
            path: path.display().to_string(),
            source: e,
        })
    }

    fn as_llvm_ir(&self) -> CodegenResult<String> {
        // GPU modules don't produce LLVM IR directly
        Err(CodegenError::Internal(
            "GPU modules use PTX/AMDGCN, not LLVM IR".to_string(),
        ))
    }
}

impl CodegenContext for GpuCodegenContext {
    type Module = GpuCodegenModule;

    fn create_module(&self, name: &str) -> CodegenResult<Self::Module> {
        let code = match self.device.kind {
            DeviceKind::Cuda => codegen::ptx::generate_module_header(name, &self.device),
            DeviceKind::Rocm => codegen::amdgcn::generate_module_header(name, &self.device),
            DeviceKind::Mock => format!("; Mock GPU module: {}\n", name),
        };

        Ok(GpuCodegenModule {
            name: name.to_string(),
            code,
            device_kind: self.device.kind,
        })
    }

    fn target(&self) -> &TargetSpec {
        &self.config.target
    }

    fn config(&self) -> &CodegenConfig {
        &self.config
    }
}

impl CodegenBackend for GpuBackend {
    type Context = GpuCodegenContext;

    fn name(&self) -> &'static str {
        "gpu"
    }

    fn supports_target(&self, target: &TargetSpec) -> bool {
        use bhc_target::Arch;
        matches!(target.arch, Arch::Nvptx64 | Arch::Amdgcn)
    }

    fn create_context(&self, config: CodegenConfig) -> CodegenResult<Self::Context> {
        let device = self.devices.first().ok_or_else(|| {
            CodegenError::BackendNotAvailable("no GPU devices available".to_string())
        })?;

        Ok(GpuCodegenContext {
            config,
            device: device.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_enumeration() {
        let devices = available_devices();
        // Should at least have mock device when no GPU runtime
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        assert!(!devices.is_empty());
    }

    #[test]
    fn test_gpu_backend_creation() {
        let backend = GpuBackend::new();
        // Should have backend available (at least mock)
        assert!(backend.is_some());
    }

    #[test]
    fn test_backend_name() {
        let backend = GpuBackend::new().unwrap();
        assert_eq!(backend.name(), "gpu");
    }

    #[test]
    fn test_error_display() {
        let err = GpuError::AllocationFailed { size: 1024 };
        assert!(err.to_string().contains("1024"));

        let err = GpuError::OutOfMemory {
            requested: 1000,
            available: 500,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("500"));
    }
}
