//! GPU device enumeration and information.
//!
//! This module provides types and functions for discovering and querying
//! GPU devices available on the system.
//!
//! # Device Discovery
//!
//! ```rust,ignore
//! use bhc_gpu::device::{DeviceId, DeviceInfo, DeviceKind};
//!
//! // Get all available devices
//! let devices = bhc_gpu::available_devices();
//!
//! for device in &devices {
//!     println!("Device {}: {}", device.id.0, device.name);
//!     println!("  Memory: {} MB", device.memory_total / 1024 / 1024);
//!     println!("  Compute units: {}", device.multiprocessor_count);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// A unique identifier for a GPU device.
///
/// Device IDs are assigned based on the order devices are enumerated
/// and are stable within a single program execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(pub u32);

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU:{}", self.0)
    }
}

/// The type of GPU device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceKind {
    /// NVIDIA GPU (CUDA).
    Cuda,
    /// AMD GPU (ROCm/HIP).
    Rocm,
    /// Mock device for testing without hardware.
    Mock,
}

impl DeviceKind {
    /// Get the display name for this device kind.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
            Self::Mock => "Mock",
        }
    }

    /// Check if this is a real hardware device.
    #[must_use]
    pub const fn is_hardware(self) -> bool {
        matches!(self, Self::Cuda | Self::Rocm)
    }
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Information about a GPU device.
///
/// Contains hardware capabilities and properties that affect kernel
/// compilation and execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device identifier.
    pub id: DeviceId,

    /// Device type (CUDA, ROCm, etc.).
    pub kind: DeviceKind,

    /// Human-readable device name.
    pub name: String,

    /// Compute capability (major, minor).
    ///
    /// For CUDA: e.g., (8, 6) for sm_86
    /// For ROCm: GFX version
    pub compute_capability: (u32, u32),

    /// Total device memory in bytes.
    pub memory_total: usize,

    /// Maximum threads per block.
    pub max_threads_per_block: u32,

    /// Maximum block dimensions (x, y, z).
    pub max_block_dim: (u32, u32, u32),

    /// Maximum grid dimensions (x, y, z).
    pub max_grid_dim: (u32, u32, u32),

    /// Warp/wavefront size.
    ///
    /// CUDA: 32 (warp size)
    /// AMD: 64 (wavefront size for most GCN/RDNA)
    pub warp_size: u32,

    /// Number of multiprocessors (CUDA SMs or AMD CUs).
    pub multiprocessor_count: u32,

    /// Shared memory per block in bytes.
    pub shared_memory_per_block: usize,

    /// Maximum registers per block.
    pub registers_per_block: u32,

    /// Memory clock rate in kHz.
    pub memory_clock_rate: u32,

    /// Memory bus width in bits.
    pub memory_bus_width: u32,

    /// L2 cache size in bytes.
    pub l2_cache_size: usize,

    /// Whether the device supports concurrent kernel execution.
    pub concurrent_kernels: bool,

    /// Whether the device supports unified memory addressing.
    pub unified_memory: bool,

    /// PCI bus ID (for multi-GPU identification).
    pub pci_bus_id: Option<u32>,
}

impl DeviceInfo {
    /// Create a mock device for testing.
    #[must_use]
    pub fn mock() -> Self {
        Self {
            id: DeviceId(0),
            kind: DeviceKind::Mock,
            name: "Mock GPU Device".to_string(),
            compute_capability: (8, 0),
            memory_total: 8 * 1024 * 1024 * 1024, // 8 GB
            max_threads_per_block: 1024,
            max_block_dim: (1024, 1024, 64),
            max_grid_dim: (2_147_483_647, 65535, 65535),
            warp_size: 32,
            multiprocessor_count: 80,
            shared_memory_per_block: 48 * 1024,
            registers_per_block: 65536,
            memory_clock_rate: 1_000_000,
            memory_bus_width: 256,
            l2_cache_size: 6 * 1024 * 1024,
            concurrent_kernels: true,
            unified_memory: true,
            pci_bus_id: Some(0),
        }
    }

    /// Get the architecture name for code generation.
    ///
    /// For CUDA: Returns PTX architecture (e.g., "sm_86")
    /// For ROCm: Returns GFX architecture (e.g., "gfx90a")
    #[must_use]
    pub fn arch_name(&self) -> String {
        match self.kind {
            DeviceKind::Cuda => {
                format!(
                    "sm_{}{}",
                    self.compute_capability.0, self.compute_capability.1
                )
            }
            DeviceKind::Rocm => {
                format!(
                    "gfx{}{}{}",
                    self.compute_capability.0,
                    self.compute_capability.1 / 10,
                    self.compute_capability.1 % 10
                )
            }
            DeviceKind::Mock => "mock_sm_80".to_string(),
        }
    }

    /// Get the memory bandwidth in GB/s (theoretical peak).
    #[must_use]
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        // bandwidth = clock_rate * bus_width / 8 * 2 (for DDR)
        let clock_ghz = f64::from(self.memory_clock_rate) / 1_000_000.0;
        let bus_bytes = f64::from(self.memory_bus_width) / 8.0;
        clock_ghz * bus_bytes * 2.0
    }

    /// Get the theoretical peak FLOPS (single precision).
    ///
    /// This is an estimate based on SM/CU count and typical configuration.
    #[must_use]
    pub fn peak_flops_sp(&self) -> f64 {
        // Rough estimate: 128 FP32 cores per SM, ~1.5 GHz typical
        let cores_per_sm = match self.kind {
            DeviceKind::Cuda => 128.0, // Ampere/Ada
            DeviceKind::Rocm => 64.0,  // RDNA 3
            DeviceKind::Mock => 128.0,
        };
        let clock_ghz = 1.5; // Assumed, should query actual value
        f64::from(self.multiprocessor_count) * cores_per_sm * clock_ghz * 2.0 * 1e9
    }

    /// Check if this device supports the given compute capability.
    #[must_use]
    pub fn supports_compute(&self, major: u32, minor: u32) -> bool {
        self.compute_capability.0 > major
            || (self.compute_capability.0 == major && self.compute_capability.1 >= minor)
    }

    /// Get optimal block size for a simple 1D kernel.
    ///
    /// Returns a block size that maximizes occupancy for simple kernels.
    #[must_use]
    pub fn optimal_block_size_1d(&self) -> u32 {
        // Generally 256 is a good default that works well across devices
        256.min(self.max_threads_per_block)
    }

    /// Calculate grid size for a given problem size and block size.
    #[must_use]
    pub fn grid_size_for(&self, problem_size: usize, block_size: u32) -> u32 {
        let grid = (problem_size as u32 + block_size - 1) / block_size;
        grid.min(self.max_grid_dim.0)
    }
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {} MB, {} SMs)",
            self.name,
            self.arch_name(),
            self.memory_total / 1024 / 1024,
            self.multiprocessor_count
        )
    }
}

/// Compute capability requirements for a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeCapability {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
}

impl ComputeCapability {
    /// Create a new compute capability.
    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// PTX target for this compute capability.
    #[must_use]
    pub fn ptx_target(&self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }

    /// Check if this capability is at least the given version.
    #[must_use]
    pub const fn at_least(&self, major: u32, minor: u32) -> bool {
        self.major > major || (self.major == major && self.minor >= minor)
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Common compute capabilities.
pub mod capabilities {
    use super::ComputeCapability;

    /// CUDA compute capability 7.0 (Volta).
    pub const SM_70: ComputeCapability = ComputeCapability::new(7, 0);

    /// CUDA compute capability 7.5 (Turing).
    pub const SM_75: ComputeCapability = ComputeCapability::new(7, 5);

    /// CUDA compute capability 8.0 (Ampere).
    pub const SM_80: ComputeCapability = ComputeCapability::new(8, 0);

    /// CUDA compute capability 8.6 (Ampere consumer).
    pub const SM_86: ComputeCapability = ComputeCapability::new(8, 6);

    /// CUDA compute capability 8.9 (Ada Lovelace).
    pub const SM_89: ComputeCapability = ComputeCapability::new(8, 9);

    /// CUDA compute capability 9.0 (Hopper).
    pub const SM_90: ComputeCapability = ComputeCapability::new(9, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id_display() {
        let id = DeviceId(3);
        assert_eq!(format!("{}", id), "GPU:3");
    }

    #[test]
    fn test_device_kind_name() {
        assert_eq!(DeviceKind::Cuda.name(), "CUDA");
        assert_eq!(DeviceKind::Rocm.name(), "ROCm");
    }

    #[test]
    fn test_mock_device() {
        let device = DeviceInfo::mock();
        assert_eq!(device.id, DeviceId(0));
        assert_eq!(device.kind, DeviceKind::Mock);
        assert!(device.memory_total > 0);
    }

    #[test]
    fn test_arch_name() {
        let mut device = DeviceInfo::mock();
        device.kind = DeviceKind::Cuda;
        device.compute_capability = (8, 6);
        assert_eq!(device.arch_name(), "sm_86");

        // ROCm: (9, 8) -> gfx908 (MI100)
        device.kind = DeviceKind::Rocm;
        device.compute_capability = (9, 8);
        assert_eq!(device.arch_name(), "gfx908");

        // ROCm: (9, 10) -> gfx910 (example)
        device.compute_capability = (9, 10);
        assert_eq!(device.arch_name(), "gfx910");
    }

    #[test]
    fn test_supports_compute() {
        let device = DeviceInfo::mock();
        assert!(device.supports_compute(7, 0));
        assert!(device.supports_compute(8, 0));
        assert!(!device.supports_compute(9, 0));
    }

    #[test]
    fn test_optimal_block_size() {
        let device = DeviceInfo::mock();
        let block_size = device.optimal_block_size_1d();
        assert!(block_size > 0);
        assert!(block_size <= device.max_threads_per_block);
    }

    #[test]
    fn test_grid_size_calculation() {
        let device = DeviceInfo::mock();
        assert_eq!(device.grid_size_for(256, 256), 1);
        assert_eq!(device.grid_size_for(257, 256), 2);
        assert_eq!(device.grid_size_for(1024, 256), 4);
    }

    #[test]
    fn test_compute_capability() {
        let cc = ComputeCapability::new(8, 6);
        assert_eq!(cc.ptx_target(), "sm_86");
        assert!(cc.at_least(8, 0));
        assert!(cc.at_least(8, 6));
        assert!(!cc.at_least(9, 0));
    }
}
