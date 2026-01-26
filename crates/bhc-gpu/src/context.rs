//! GPU context and stream management.
//!
//! This module provides the `GpuContext` type, which owns device state and
//! provides methods for memory allocation, kernel execution, and data transfer.
//!
//! # Context Lifecycle
//!
//! A `GpuContext` is created by selecting a device:
//!
//! ```rust,ignore
//! let ctx = bhc_gpu::select_device(DeviceId(0))?;
//! ```
//!
//! The context manages:
//! - Device selection and activation
//! - Stream creation and synchronization
//! - Kernel module caching
//! - Memory pool management
//!
//! # Streams
//!
//! GPU operations are submitted to streams for asynchronous execution.
//! Multiple streams can execute concurrently on modern GPUs.
//!
//! ```rust,ignore
//! // Default stream (synchronous within device)
//! ctx.copy_to_device(&host_data, &mut device_buf)?;
//!
//! // Named stream for concurrent operations
//! let stream = ctx.create_stream("compute")?;
//! ctx.launch_on_stream(&kernel, &args, &stream)?;
//! ```

use crate::device::{DeviceId, DeviceInfo, DeviceKind};
use crate::kernel::{CompiledModule, GpuKernel, LaunchConfig};
use crate::memory::{DeviceBuffer, DeviceMemoryPool, DevicePtr};
use crate::transfer::{TransferHandle, TransferQueue};
use crate::{available_devices, GpuError, GpuResult};
use bhc_ffi::FfiSafe;
use bhc_tensor_ir::{Kernel, KernelId};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A handle to a GPU execution stream.
///
/// Streams allow asynchronous and concurrent execution of GPU operations.
/// Operations submitted to the same stream execute in order, while
/// operations on different streams may execute concurrently.
#[derive(Clone, Debug)]
pub struct Stream {
    /// Stream handle (runtime-specific).
    pub(crate) handle: u64,
    /// Stream name for debugging.
    name: String,
    /// Device this stream belongs to.
    device: DeviceId,
}

impl Stream {
    /// Create a new stream handle.
    pub(crate) fn new(handle: u64, name: impl Into<String>, device: DeviceId) -> Self {
        Self {
            handle,
            name: name.into(),
            device,
        }
    }

    /// Get the stream name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the device ID.
    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.device
    }

    /// Check if this is the default (null) stream.
    #[must_use]
    pub const fn is_default(&self) -> bool {
        self.handle == 0
    }
}

/// GPU context that owns device state.
///
/// The context is the primary interface for GPU operations. It manages
/// device activation, memory allocation, kernel execution, and data transfer.
///
/// # Thread Safety
///
/// `GpuContext` is `Send + Sync` and can be shared across threads. Internal
/// synchronization ensures safe concurrent access to device resources.
///
/// # Example
///
/// ```rust,ignore
/// let ctx = bhc_gpu::select_device(DeviceId(0))?;
///
/// // Allocate device memory
/// let mut buf: DeviceBuffer<f32> = ctx.alloc(1024)?;
///
/// // Transfer data
/// let host_data = vec![1.0f32; 1024];
/// ctx.copy_to_device(&host_data, &mut buf)?;
///
/// // Launch kernel
/// let kernel = ctx.compile_kernel(&tensor_ir)?;
/// ctx.launch(&kernel, LaunchConfig::default(), &[&buf])?;
///
/// // Synchronize
/// ctx.synchronize()?;
/// ```
pub struct GpuContext {
    /// Device info.
    device: DeviceInfo,

    /// Default stream.
    default_stream: Stream,

    /// Created streams.
    streams: RwLock<FxHashMap<String, Stream>>,

    /// Compiled module cache.
    module_cache: RwLock<FxHashMap<KernelId, Arc<CompiledModule>>>,

    /// Memory pool.
    memory_pool: DeviceMemoryPool,

    /// Transfer queue for async operations.
    transfer_queue: TransferQueue,

    /// Next stream ID for creating unique handles.
    next_stream_id: AtomicU64,
}

impl GpuContext {
    /// Create a new GPU context for the given device.
    pub(crate) fn new(device_id: DeviceId) -> GpuResult<Self> {
        let devices = available_devices();
        let device = devices
            .into_iter()
            .find(|d| d.id == device_id)
            .ok_or(GpuError::InvalidDevice(device_id))?;

        // Initialize the device context
        Self::init_device(&device)?;

        let default_stream = Stream::new(0, "default", device_id);

        Ok(Self {
            device: device.clone(),
            default_stream,
            streams: RwLock::new(FxHashMap::default()),
            module_cache: RwLock::new(FxHashMap::default()),
            memory_pool: DeviceMemoryPool::new(
                device_id,
                device.kind,
                64 * 1024 * 1024, // 64 MB blocks
            ),
            transfer_queue: TransferQueue::new(device_id),
            next_stream_id: AtomicU64::new(1),
        })
    }

    /// Initialize the device (set current device, etc.).
    fn init_device(device: &DeviceInfo) -> GpuResult<()> {
        match device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => crate::runtime::cuda::set_device(device.id),

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => crate::runtime::rocm::set_device(device.id),

            DeviceKind::Mock | _ => Ok(()),
        }
    }

    /// Get the device info.
    #[must_use]
    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    /// Get the device ID.
    #[must_use]
    pub fn device_id(&self) -> DeviceId {
        self.device.id
    }

    /// Get the default stream.
    #[must_use]
    pub fn default_stream(&self) -> &Stream {
        &self.default_stream
    }

    /// Create a new stream.
    ///
    /// Named streams allow concurrent kernel execution and data transfer.
    pub fn create_stream(&self, name: impl Into<String>) -> GpuResult<Stream> {
        let name = name.into();

        // Check if stream already exists
        if let Some(stream) = self.streams.read().get(&name) {
            return Ok(stream.clone());
        }

        // Create new stream
        let handle = self.next_stream_id.fetch_add(1, Ordering::SeqCst);

        match self.device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => {
                let stream_handle = crate::runtime::cuda::create_stream()?;
                let stream = Stream::new(stream_handle, name.clone(), self.device.id);
                self.streams.write().insert(name, stream.clone());
                Ok(stream)
            }

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => {
                let stream_handle = crate::runtime::rocm::create_stream()?;
                let stream = Stream::new(stream_handle, name.clone(), self.device.id);
                self.streams.write().insert(name, stream.clone());
                Ok(stream)
            }

            DeviceKind::Mock | _ => {
                // Mock: just use incrementing handles
                let stream = Stream::new(handle, name.clone(), self.device.id);
                self.streams.write().insert(name, stream.clone());
                Ok(stream)
            }
        }
    }

    /// Allocate device memory.
    ///
    /// Returns a buffer of `len` elements of type `T` on the device.
    pub fn alloc<T: FfiSafe>(&self, len: usize) -> GpuResult<DeviceBuffer<T>> {
        self.memory_pool.alloc(len)
    }

    /// Allocate and zero-initialize device memory.
    pub fn alloc_zeroed<T: FfiSafe>(&self, len: usize) -> GpuResult<DeviceBuffer<T>> {
        let buf = self.alloc::<T>(len)?;

        // Zero the memory
        match self.device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => {
                crate::runtime::cuda::memset(buf.as_ptr(), 0, buf.size_bytes())?;
            }

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => {
                crate::runtime::rocm::memset(buf.as_ptr(), 0, buf.size_bytes())?;
            }

            DeviceKind::Mock | _ => {
                // Mock: zero host memory
                unsafe {
                    std::ptr::write_bytes(buf.as_ptr().as_raw() as *mut u8, 0, buf.size_bytes());
                }
            }
        }

        Ok(buf)
    }

    /// Copy data from host to device.
    ///
    /// # Panics
    ///
    /// Panics if the source and destination sizes don't match.
    pub fn copy_to_device<T: FfiSafe>(
        &self,
        src: &[T],
        dst: &mut DeviceBuffer<T>,
    ) -> GpuResult<()> {
        assert_eq!(
            src.len(),
            dst.len(),
            "copy_to_device: size mismatch ({} vs {})",
            src.len(),
            dst.len()
        );

        let size = src.len() * std::mem::size_of::<T>();

        match self.device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => crate::runtime::cuda::memcpy_host_to_device(
                dst.as_ptr(),
                src.as_ptr() as *const u8,
                size,
            ),

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => crate::runtime::rocm::memcpy_host_to_device(
                dst.as_ptr(),
                src.as_ptr() as *const u8,
                size,
            ),

            DeviceKind::Mock | _ => {
                // Mock: direct memcpy
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr() as *const u8,
                        dst.as_ptr().as_raw() as *mut u8,
                        size,
                    );
                }
                Ok(())
            }
        }
    }

    /// Copy data from device to host.
    ///
    /// # Panics
    ///
    /// Panics if the source and destination sizes don't match.
    pub fn copy_to_host<T: FfiSafe>(&self, src: &DeviceBuffer<T>, dst: &mut [T]) -> GpuResult<()> {
        assert_eq!(
            src.len(),
            dst.len(),
            "copy_to_host: size mismatch ({} vs {})",
            src.len(),
            dst.len()
        );

        let size = src.len() * std::mem::size_of::<T>();

        match self.device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => crate::runtime::cuda::memcpy_device_to_host(
                dst.as_mut_ptr() as *mut u8,
                src.as_ptr(),
                size,
            ),

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => crate::runtime::rocm::memcpy_device_to_host(
                dst.as_mut_ptr() as *mut u8,
                src.as_ptr(),
                size,
            ),

            DeviceKind::Mock | _ => {
                // Mock: direct memcpy
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().as_raw() as *const u8,
                        dst.as_mut_ptr() as *mut u8,
                        size,
                    );
                }
                Ok(())
            }
        }
    }

    /// Asynchronously copy data from host to device.
    pub fn copy_to_device_async<T: FfiSafe>(
        &self,
        src: &[T],
        dst: &mut DeviceBuffer<T>,
        stream: &Stream,
    ) -> GpuResult<TransferHandle> {
        self.transfer_queue.enqueue_host_to_device(src, dst, stream)
    }

    /// Asynchronously copy data from device to host.
    pub fn copy_to_host_async<T: FfiSafe>(
        &self,
        src: &DeviceBuffer<T>,
        dst: &mut [T],
        stream: &Stream,
    ) -> GpuResult<TransferHandle> {
        self.transfer_queue.enqueue_device_to_host(src, dst, stream)
    }

    /// Copy data between device buffers.
    pub fn copy_device_to_device<T: FfiSafe>(
        &self,
        src: &DeviceBuffer<T>,
        dst: &mut DeviceBuffer<T>,
    ) -> GpuResult<()> {
        assert_eq!(src.len(), dst.len(), "copy_device_to_device: size mismatch");

        let size = src.len() * std::mem::size_of::<T>();

        match self.device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => {
                crate::runtime::cuda::memcpy_device_to_device(dst.as_ptr(), src.as_ptr(), size)
            }

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => {
                crate::runtime::rocm::memcpy_device_to_device(dst.as_ptr(), src.as_ptr(), size)
            }

            DeviceKind::Mock | _ => {
                // Mock: direct memcpy
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().as_raw() as *const u8,
                        dst.as_ptr().as_raw() as *mut u8,
                        size,
                    );
                }
                Ok(())
            }
        }
    }

    /// Compile a Tensor IR kernel for this device.
    pub fn compile_kernel(&self, kernel: &Kernel) -> GpuResult<Arc<GpuKernel>> {
        // Check cache first
        if let Some(module) = self.module_cache.read().get(&kernel.id) {
            return Ok(Arc::new(GpuKernel::from_cached(kernel, module.clone())));
        }

        // Compile the kernel
        let compiled = match self.device.kind {
            DeviceKind::Cuda => crate::codegen::ptx::compile_kernel(kernel, &self.device)?,
            DeviceKind::Rocm => crate::codegen::amdgcn::compile_kernel(kernel, &self.device)?,
            DeviceKind::Mock => crate::codegen::mock_compile_kernel(kernel, &self.device)?,
        };

        let module = Arc::new(compiled);
        self.module_cache.write().insert(kernel.id, module.clone());

        Ok(Arc::new(GpuKernel::from_cached(kernel, module)))
    }

    /// Launch a compiled kernel.
    pub fn launch(
        &self,
        kernel: &GpuKernel,
        config: LaunchConfig,
        args: &[DevicePtr],
    ) -> GpuResult<()> {
        self.launch_on_stream(kernel, config, args, &self.default_stream)
    }

    /// Launch a kernel on a specific stream.
    pub fn launch_on_stream(
        &self,
        kernel: &GpuKernel,
        config: LaunchConfig,
        args: &[DevicePtr],
        stream: &Stream,
    ) -> GpuResult<()> {
        match self.device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => crate::runtime::cuda::launch_kernel(
                kernel.module(),
                kernel.name(),
                config.grid_dim,
                config.block_dim,
                config.shared_mem,
                stream.handle,
                args,
            ),

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => crate::runtime::rocm::launch_kernel(
                kernel.module(),
                kernel.name(),
                config.grid_dim,
                config.block_dim,
                config.shared_mem,
                stream.handle,
                args,
            ),

            DeviceKind::Mock | _ => {
                // Mock: just log the launch (args and stream unused in mock mode)
                let _ = (args, stream);
                tracing::debug!(
                    "Mock kernel launch: {} grid={:?} block={:?}",
                    kernel.name(),
                    config.grid_dim,
                    config.block_dim
                );
                Ok(())
            }
        }
    }

    /// Synchronize the default stream.
    pub fn synchronize(&self) -> GpuResult<()> {
        self.synchronize_stream(&self.default_stream)
    }

    /// Synchronize a specific stream.
    pub fn synchronize_stream(&self, stream: &Stream) -> GpuResult<()> {
        match self.device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => crate::runtime::cuda::synchronize_stream(stream.handle),

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => crate::runtime::rocm::synchronize_stream(stream.handle),

            DeviceKind::Mock | _ => {
                let _ = stream;
                Ok(())
            }
        }
    }

    /// Synchronize the entire device.
    pub fn device_synchronize(&self) -> GpuResult<()> {
        match self.device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => crate::runtime::cuda::device_synchronize(),

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => crate::runtime::rocm::device_synchronize(),

            DeviceKind::Mock | _ => Ok(()),
        }
    }

    /// Get memory info.
    #[must_use]
    pub fn memory_info(&self) -> (usize, usize) {
        match self.device.kind {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => crate::runtime::cuda::memory_info().unwrap_or((0, 0)),

            #[cfg(feature = "rocm")]
            DeviceKind::Rocm => crate::runtime::rocm::memory_info().unwrap_or((0, 0)),

            DeviceKind::Mock | _ => (self.device.memory_total, self.device.memory_total),
        }
    }

    /// Get allocation statistics.
    #[must_use]
    pub fn alloc_stats(&self) -> &crate::memory::DeviceAllocStats {
        self.memory_pool.stats()
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &self.device)
            .field("streams", &self.streams.read().len())
            .field("cached_modules", &self.module_cache.read().len())
            .finish()
    }
}

// Safety: GpuContext uses internal synchronization
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_creation() {
        let device = DeviceInfo::mock();
        let ctx = GpuContext::new(device.id).unwrap();

        let stream1 = ctx.create_stream("compute").unwrap();
        assert_eq!(stream1.name(), "compute");

        let stream2 = ctx.create_stream("transfer").unwrap();
        assert_eq!(stream2.name(), "transfer");

        // Same name returns same stream
        let stream1_again = ctx.create_stream("compute").unwrap();
        assert_eq!(stream1.handle, stream1_again.handle);
    }

    #[test]
    fn test_alloc_and_copy() {
        let ctx = GpuContext::new(DeviceId(0)).unwrap();

        let mut buf: DeviceBuffer<f32> = ctx.alloc(256).unwrap();
        assert_eq!(buf.len(), 256);

        let host_data = vec![1.0f32; 256];
        ctx.copy_to_device(&host_data, &mut buf).unwrap();

        let mut result = vec![0.0f32; 256];
        ctx.copy_to_host(&buf, &mut result).unwrap();

        assert_eq!(host_data, result);
    }

    #[test]
    fn test_zeroed_allocation() {
        let ctx = GpuContext::new(DeviceId(0)).unwrap();

        let buf: DeviceBuffer<f32> = ctx.alloc_zeroed(256).unwrap();
        let mut result = vec![1.0f32; 256];
        ctx.copy_to_host(&buf, &mut result).unwrap();

        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_device_to_device_copy() {
        let ctx = GpuContext::new(DeviceId(0)).unwrap();

        let mut src: DeviceBuffer<f32> = ctx.alloc(256).unwrap();
        let host_data = vec![42.0f32; 256];
        ctx.copy_to_device(&host_data, &mut src).unwrap();

        let mut dst: DeviceBuffer<f32> = ctx.alloc(256).unwrap();
        ctx.copy_device_to_device(&src, &mut dst).unwrap();

        let mut result = vec![0.0f32; 256];
        ctx.copy_to_host(&dst, &mut result).unwrap();

        assert_eq!(host_data, result);
    }

    #[test]
    fn test_memory_info() {
        let ctx = GpuContext::new(DeviceId(0)).unwrap();
        let (free, total) = ctx.memory_info();
        assert!(total > 0);
        assert!(free <= total);
    }
}
