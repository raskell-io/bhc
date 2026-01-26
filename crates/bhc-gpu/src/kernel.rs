//! GPU kernel compilation and launch.
//!
//! This module provides types for compiling Tensor IR kernels to GPU code
//! and launching them on the device.
//!
//! # Kernel Compilation
//!
//! Kernels are compiled from Tensor IR through the codegen pipeline:
//!
//! ```text
//! Tensor IR Kernel
//!        │
//!        ▼
//! ┌─────────────────┐
//! │  GPU Codegen    │ ──▶ PTX (CUDA) or AMDGCN (ROCm)
//! └─────────────────┘
//!        │
//!        ▼
//! ┌─────────────────┐
//! │  JIT Compiler   │ ──▶ Device binary
//! └─────────────────┘
//!        │
//!        ▼
//! ┌─────────────────┐
//! │  GpuKernel      │ ──▶ Ready for launch
//! └─────────────────┘
//! ```
//!
//! # Launch Configuration
//!
//! Kernels require launch configuration specifying grid and block dimensions:
//!
//! ```rust,ignore
//! let config = LaunchConfig::for_elements(n, 256);
//! ctx.launch(&kernel, config, &args)?;
//! ```

use crate::device::DeviceInfo;
use crate::memory::DevicePtr;
use bhc_tensor_ir::{Kernel, KernelId, TensorRef};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A compiled GPU module containing one or more kernels.
#[derive(Debug)]
pub struct CompiledModule {
    /// Module name.
    pub name: String,
    /// The compiled code (PTX, AMDGCN, or binary).
    pub code: Vec<u8>,
    /// Whether this is binary (true) or text (false) code.
    pub is_binary: bool,
    /// Target architecture.
    pub arch: String,
    /// Entry points in this module.
    pub entry_points: Vec<String>,
}

impl CompiledModule {
    /// Create a new compiled module from text code.
    #[must_use]
    pub fn from_text(name: String, code: String, arch: String) -> Self {
        Self {
            name,
            code: code.into_bytes(),
            is_binary: false,
            arch,
            entry_points: Vec::new(),
        }
    }

    /// Create a new compiled module from binary code.
    #[must_use]
    pub fn from_binary(name: String, code: Vec<u8>, arch: String) -> Self {
        Self {
            name,
            code,
            is_binary: true,
            arch,
            entry_points: Vec::new(),
        }
    }

    /// Add an entry point.
    pub fn add_entry_point(&mut self, name: impl Into<String>) {
        self.entry_points.push(name.into());
    }

    /// Get the code as text (for text modules).
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        if self.is_binary {
            None
        } else {
            std::str::from_utf8(&self.code).ok()
        }
    }
}

/// A compiled GPU kernel ready for execution.
#[derive(Debug)]
pub struct GpuKernel {
    /// Kernel ID from Tensor IR.
    pub id: KernelId,
    /// Kernel name.
    pub name: String,
    /// Input tensor descriptors.
    pub inputs: Vec<TensorRef>,
    /// Output tensor descriptors.
    pub outputs: Vec<TensorRef>,
    /// Compiled module.
    module: Arc<CompiledModule>,
    /// Recommended launch configuration.
    pub recommended_config: KernelConfig,
}

impl GpuKernel {
    /// Create from a cached compiled module.
    pub(crate) fn from_cached(kernel: &Kernel, module: Arc<CompiledModule>) -> Self {
        Self {
            id: kernel.id,
            name: kernel.name.as_str().to_string(),
            inputs: kernel.inputs.clone(),
            outputs: kernel.outputs.clone(),
            module,
            recommended_config: KernelConfig::default(),
        }
    }

    /// Get the kernel name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the compiled module.
    #[must_use]
    pub fn module(&self) -> &CompiledModule {
        &self.module
    }

    /// Get the number of input tensors.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Get the number of output tensors.
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Compute optimal launch configuration for a given problem size.
    #[must_use]
    pub fn launch_config_for(&self, device: &DeviceInfo, problem_size: usize) -> LaunchConfig {
        let block_size = self.recommended_config.block_size.unwrap_or(256);
        let block_size = block_size.min(device.max_threads_per_block);

        let grid_size = (problem_size as u32 + block_size - 1) / block_size;
        let grid_size = grid_size.min(device.max_grid_dim.0);

        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem: self.recommended_config.shared_memory,
        }
    }
}

/// Kernel configuration hints.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Recommended block size.
    pub block_size: Option<u32>,
    /// Shared memory per block in bytes.
    pub shared_memory: usize,
    /// Maximum registers per thread.
    pub max_registers: Option<u32>,
    /// Cache preference.
    pub cache_preference: CachePreference,
}

impl KernelConfig {
    /// Create a config with default block size.
    #[must_use]
    pub fn with_block_size(block_size: u32) -> Self {
        Self {
            block_size: Some(block_size),
            ..Self::default()
        }
    }

    /// Set shared memory requirement.
    #[must_use]
    pub const fn with_shared_memory(mut self, bytes: usize) -> Self {
        self.shared_memory = bytes;
        self
    }
}

/// Cache preference for kernel execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CachePreference {
    /// Let the driver decide.
    #[default]
    None,
    /// Prefer shared memory over L1 cache.
    PreferShared,
    /// Prefer L1 cache over shared memory.
    PreferL1,
    /// Equal preference.
    Equal,
}

/// Launch configuration for kernel execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchConfig {
    /// Grid dimensions (blocks in x, y, z).
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads in x, y, z).
    pub block_dim: (u32, u32, u32),
    /// Dynamic shared memory size in bytes.
    pub shared_mem: usize,
}

impl LaunchConfig {
    /// Create a 1D launch configuration for a given number of elements.
    #[must_use]
    pub fn for_elements(n: usize, block_size: u32) -> Self {
        let grid_size = ((n as u32) + block_size - 1) / block_size;
        Self {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem: 0,
        }
    }

    /// Create a 2D launch configuration.
    #[must_use]
    pub fn for_2d(width: usize, height: usize, block_x: u32, block_y: u32) -> Self {
        let grid_x = ((width as u32) + block_x - 1) / block_x;
        let grid_y = ((height as u32) + block_y - 1) / block_y;
        Self {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_x, block_y, 1),
            shared_mem: 0,
        }
    }

    /// Set dynamic shared memory size.
    #[must_use]
    pub const fn with_shared_mem(mut self, bytes: usize) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Get total number of threads.
    #[must_use]
    pub const fn total_threads(&self) -> u64 {
        let grid = (self.grid_dim.0 as u64) * (self.grid_dim.1 as u64) * (self.grid_dim.2 as u64);
        let block =
            (self.block_dim.0 as u64) * (self.block_dim.1 as u64) * (self.block_dim.2 as u64);
        grid * block
    }

    /// Get total number of blocks.
    #[must_use]
    pub const fn total_blocks(&self) -> u64 {
        (self.grid_dim.0 as u64) * (self.grid_dim.1 as u64) * (self.grid_dim.2 as u64)
    }

    /// Get threads per block.
    #[must_use]
    pub const fn threads_per_block(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem: 0,
        }
    }
}

/// Kernel argument types.
#[derive(Clone, Debug)]
pub enum KernelArg {
    /// Device pointer.
    Ptr(DevicePtr),
    /// 32-bit integer.
    I32(i32),
    /// 64-bit integer.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Size type.
    Size(usize),
}

impl KernelArg {
    /// Get the size of this argument in bytes.
    #[must_use]
    pub const fn size(&self) -> usize {
        match self {
            Self::Ptr(_) => 8,
            Self::I32(_) | Self::F32(_) => 4,
            Self::I64(_) | Self::F64(_) | Self::Size(_) => 8,
        }
    }

    /// Write this argument to a byte buffer.
    pub fn write_to(&self, buf: &mut Vec<u8>) {
        match self {
            Self::Ptr(ptr) => buf.extend_from_slice(&ptr.as_raw().to_ne_bytes()),
            Self::I32(v) => buf.extend_from_slice(&v.to_ne_bytes()),
            Self::I64(v) => buf.extend_from_slice(&v.to_ne_bytes()),
            Self::F32(v) => buf.extend_from_slice(&v.to_ne_bytes()),
            Self::F64(v) => buf.extend_from_slice(&v.to_ne_bytes()),
            Self::Size(v) => buf.extend_from_slice(&(*v as u64).to_ne_bytes()),
        }
    }
}

impl From<DevicePtr> for KernelArg {
    fn from(ptr: DevicePtr) -> Self {
        Self::Ptr(ptr)
    }
}

impl From<i32> for KernelArg {
    fn from(v: i32) -> Self {
        Self::I32(v)
    }
}

impl From<i64> for KernelArg {
    fn from(v: i64) -> Self {
        Self::I64(v)
    }
}

impl From<f32> for KernelArg {
    fn from(v: f32) -> Self {
        Self::F32(v)
    }
}

impl From<f64> for KernelArg {
    fn from(v: f64) -> Self {
        Self::F64(v)
    }
}

impl From<usize> for KernelArg {
    fn from(v: usize) -> Self {
        Self::Size(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config_1d() {
        let config = LaunchConfig::for_elements(1000, 256);
        assert_eq!(config.grid_dim, (4, 1, 1));
        assert_eq!(config.block_dim, (256, 1, 1));
        assert_eq!(config.total_threads(), 1024);
    }

    #[test]
    fn test_launch_config_2d() {
        let config = LaunchConfig::for_2d(100, 100, 16, 16);
        assert_eq!(config.grid_dim, (7, 7, 1));
        assert_eq!(config.block_dim, (16, 16, 1));
    }

    #[test]
    fn test_launch_config_threads() {
        let config = LaunchConfig {
            grid_dim: (10, 20, 1),
            block_dim: (32, 8, 1),
            shared_mem: 0,
        };
        assert_eq!(config.total_blocks(), 200);
        assert_eq!(config.threads_per_block(), 256);
        assert_eq!(config.total_threads(), 51200);
    }

    #[test]
    fn test_kernel_arg_sizes() {
        assert_eq!(KernelArg::Ptr(DevicePtr::null()).size(), 8);
        assert_eq!(KernelArg::I32(0).size(), 4);
        assert_eq!(KernelArg::I64(0).size(), 8);
        assert_eq!(KernelArg::F32(0.0).size(), 4);
        assert_eq!(KernelArg::F64(0.0).size(), 8);
    }

    #[test]
    fn test_kernel_arg_write() {
        let mut buf = Vec::new();
        KernelArg::I32(42).write_to(&mut buf);
        assert_eq!(buf.len(), 4);
        assert_eq!(i32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]), 42);
    }

    #[test]
    fn test_compiled_module_text() {
        let module = CompiledModule::from_text(
            "test".to_string(),
            ".version 7.0\n.target sm_80\n".to_string(),
            "sm_80".to_string(),
        );
        assert!(!module.is_binary);
        assert!(module.as_text().is_some());
        assert!(module.as_text().unwrap().contains(".version"));
    }

    #[test]
    fn test_kernel_config() {
        let config = KernelConfig::with_block_size(512).with_shared_memory(4096);
        assert_eq!(config.block_size, Some(512));
        assert_eq!(config.shared_memory, 4096);
    }
}
