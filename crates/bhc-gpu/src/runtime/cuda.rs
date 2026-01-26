//! CUDA runtime bindings.
//!
//! This module provides bindings to the CUDA Driver API for GPU operations.
//! It uses dynamic linking to load the CUDA library at runtime, allowing
//! the code to compile without CUDA installed.
//!
//! # CUDA Driver API
//!
//! We use the CUDA Driver API (cu* functions) rather than the Runtime API
//! (cuda* functions) for more control over context and module management.
//!
//! # Error Handling
//!
//! CUDA errors are translated to `GpuError::CudaError` with the error
//! code and descriptive message.
//!
//! # Dynamic Loading
//!
//! The CUDA library is loaded dynamically at runtime. This allows the code
//! to compile on systems without CUDA, and gracefully handle missing CUDA
//! at runtime.

use super::GpuRuntime;
use crate::device::{DeviceId, DeviceInfo, DeviceKind};
use crate::kernel::CompiledModule;
use crate::memory::DevicePtr;
use crate::{GpuError, GpuResult};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::sync::OnceLock;

// ============================================================================
// CUDA Types
// ============================================================================

/// CUDA result type (error code).
type CUresult = c_int;

/// CUDA device handle.
type CUdevice = c_int;

/// CUDA context handle.
type CUcontext = *mut c_void;

/// CUDA module handle.
type CUmodule = *mut c_void;

/// CUDA function handle.
type CUfunction = *mut c_void;

/// CUDA stream handle.
type CUstream = *mut c_void;

/// CUDA device pointer.
type CUdeviceptr = u64;

/// CUDA success code.
const CUDA_SUCCESS: CUresult = 0;

// CUDA device attributes
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: c_int = 2;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: c_int = 3;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: c_int = 4;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: c_int = 5;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: c_int = 6;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: c_int = 7;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: c_int = 9;
const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: c_int = 12;
const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: c_int = 13;
const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: c_int = 38;

// ============================================================================
// CUDA Function Signatures
// ============================================================================

type FnCuInit = unsafe extern "C" fn(flags: c_uint) -> CUresult;
type FnCuDeviceGetCount = unsafe extern "C" fn(count: *mut c_int) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult;
type FnCuDeviceGetName =
    unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
type FnCuDeviceGetAttribute =
    unsafe extern "C" fn(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
type FnCuDeviceTotalMem = unsafe extern "C" fn(bytes: *mut usize, dev: CUdevice) -> CUresult;
type FnCuCtxCreate =
    unsafe extern "C" fn(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
type FnCuCtxDestroy = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
type FnCuCtxGetCurrent = unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult;
type FnCuCtxSynchronize = unsafe extern "C" fn() -> CUresult;
type FnCuMemAlloc = unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
type FnCuMemFree = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;
type FnCuMemcpyHtoD =
    unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, bytecount: usize) -> CUresult;
type FnCuMemcpyDtoH =
    unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, bytecount: usize) -> CUresult;
type FnCuMemcpyDtoD =
    unsafe extern "C" fn(dst: CUdeviceptr, src: CUdeviceptr, bytecount: usize) -> CUresult;
type FnCuMemsetD8 = unsafe extern "C" fn(dstDevice: CUdeviceptr, uc: u8, n: usize) -> CUresult;
type FnCuMemGetInfo = unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> CUresult;
type FnCuModuleLoadData =
    unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult;
type FnCuModuleUnload = unsafe extern "C" fn(hmod: CUmodule) -> CUresult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult;
type FnCuStreamCreate = unsafe extern "C" fn(phStream: *mut CUstream, flags: c_uint) -> CUresult;
type FnCuStreamDestroy = unsafe extern "C" fn(hStream: CUstream) -> CUresult;
type FnCuStreamSynchronize = unsafe extern "C" fn(hStream: CUstream) -> CUresult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult;
type FnCuGetErrorString =
    unsafe extern "C" fn(error: CUresult, pStr: *mut *const c_char) -> CUresult;

// ============================================================================
// Dynamic Library Loading
// ============================================================================

/// CUDA API function pointers loaded dynamically.
struct CudaApi {
    cuInit: FnCuInit,
    cuDeviceGetCount: FnCuDeviceGetCount,
    cuDeviceGet: FnCuDeviceGet,
    cuDeviceGetName: FnCuDeviceGetName,
    cuDeviceGetAttribute: FnCuDeviceGetAttribute,
    cuDeviceTotalMem: FnCuDeviceTotalMem,
    cuCtxCreate: FnCuCtxCreate,
    cuCtxDestroy: FnCuCtxDestroy,
    cuCtxSetCurrent: FnCuCtxSetCurrent,
    cuCtxGetCurrent: FnCuCtxGetCurrent,
    cuCtxSynchronize: FnCuCtxSynchronize,
    cuMemAlloc: FnCuMemAlloc,
    cuMemFree: FnCuMemFree,
    cuMemcpyHtoD: FnCuMemcpyHtoD,
    cuMemcpyDtoH: FnCuMemcpyDtoH,
    cuMemcpyDtoD: FnCuMemcpyDtoD,
    cuMemsetD8: FnCuMemsetD8,
    cuMemGetInfo: FnCuMemGetInfo,
    cuModuleLoadData: FnCuModuleLoadData,
    cuModuleUnload: FnCuModuleUnload,
    cuModuleGetFunction: FnCuModuleGetFunction,
    cuStreamCreate: FnCuStreamCreate,
    cuStreamDestroy: FnCuStreamDestroy,
    cuStreamSynchronize: FnCuStreamSynchronize,
    cuLaunchKernel: FnCuLaunchKernel,
    cuGetErrorString: FnCuGetErrorString,
}

/// Global CUDA API instance, loaded once.
static CUDA_API: OnceLock<Option<CudaApi>> = OnceLock::new();

/// Try to load the CUDA driver library.
fn load_cuda_api() -> Option<CudaApi> {
    // Try different library names based on platform
    #[cfg(target_os = "linux")]
    let lib_names = ["libcuda.so.1", "libcuda.so"];

    #[cfg(target_os = "macos")]
    let lib_names = ["libcuda.dylib"];

    #[cfg(target_os = "windows")]
    let lib_names = ["nvcuda.dll"];

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    let lib_names: [&str; 0] = [];

    for lib_name in lib_names {
        if let Some(api) = try_load_cuda_from(lib_name) {
            return Some(api);
        }
    }

    None
}

/// Try to load CUDA from a specific library path.
fn try_load_cuda_from(lib_name: &str) -> Option<CudaApi> {
    tracing::debug!("Attempting to load CUDA from: {}", lib_name);

    // Safety: We're loading symbols from the CUDA driver library which
    // provides a stable ABI. The function signatures must match exactly.
    unsafe {
        let lib = libloading::Library::new(lib_name).ok()?;

        // Load all required symbols
        let cu_init: libloading::Symbol<FnCuInit> = lib.get(b"cuInit").ok()?;
        let cu_device_get_count: libloading::Symbol<FnCuDeviceGetCount> =
            lib.get(b"cuDeviceGetCount").ok()?;
        let cu_device_get: libloading::Symbol<FnCuDeviceGet> = lib.get(b"cuDeviceGet").ok()?;
        let cu_device_get_name: libloading::Symbol<FnCuDeviceGetName> =
            lib.get(b"cuDeviceGetName").ok()?;
        let cu_device_get_attribute: libloading::Symbol<FnCuDeviceGetAttribute> =
            lib.get(b"cuDeviceGetAttribute").ok()?;
        let cu_device_total_mem: libloading::Symbol<FnCuDeviceTotalMem> =
            lib.get(b"cuDeviceTotalMem_v2").ok()?;
        let cu_ctx_create: libloading::Symbol<FnCuCtxCreate> = lib.get(b"cuCtxCreate_v2").ok()?;
        let cu_ctx_destroy: libloading::Symbol<FnCuCtxDestroy> =
            lib.get(b"cuCtxDestroy_v2").ok()?;
        let cu_ctx_set_current: libloading::Symbol<FnCuCtxSetCurrent> =
            lib.get(b"cuCtxSetCurrent").ok()?;
        let cu_ctx_get_current: libloading::Symbol<FnCuCtxGetCurrent> =
            lib.get(b"cuCtxGetCurrent").ok()?;
        let cu_ctx_synchronize: libloading::Symbol<FnCuCtxSynchronize> =
            lib.get(b"cuCtxSynchronize").ok()?;
        let cu_mem_alloc: libloading::Symbol<FnCuMemAlloc> = lib.get(b"cuMemAlloc_v2").ok()?;
        let cu_mem_free: libloading::Symbol<FnCuMemFree> = lib.get(b"cuMemFree_v2").ok()?;
        let cu_memcpy_htod: libloading::Symbol<FnCuMemcpyHtoD> =
            lib.get(b"cuMemcpyHtoD_v2").ok()?;
        let cu_memcpy_dtoh: libloading::Symbol<FnCuMemcpyDtoH> =
            lib.get(b"cuMemcpyDtoH_v2").ok()?;
        let cu_memcpy_dtod: libloading::Symbol<FnCuMemcpyDtoD> =
            lib.get(b"cuMemcpyDtoD_v2").ok()?;
        let cu_memset_d8: libloading::Symbol<FnCuMemsetD8> = lib.get(b"cuMemsetD8_v2").ok()?;
        let cu_mem_get_info: libloading::Symbol<FnCuMemGetInfo> =
            lib.get(b"cuMemGetInfo_v2").ok()?;
        let cu_module_load_data: libloading::Symbol<FnCuModuleLoadData> =
            lib.get(b"cuModuleLoadData").ok()?;
        let cu_module_unload: libloading::Symbol<FnCuModuleUnload> =
            lib.get(b"cuModuleUnload").ok()?;
        let cu_module_get_function: libloading::Symbol<FnCuModuleGetFunction> =
            lib.get(b"cuModuleGetFunction").ok()?;
        let cu_stream_create: libloading::Symbol<FnCuStreamCreate> =
            lib.get(b"cuStreamCreate").ok()?;
        let cu_stream_destroy: libloading::Symbol<FnCuStreamDestroy> =
            lib.get(b"cuStreamDestroy_v2").ok()?;
        let cu_stream_synchronize: libloading::Symbol<FnCuStreamSynchronize> =
            lib.get(b"cuStreamSynchronize").ok()?;
        let cu_launch_kernel: libloading::Symbol<FnCuLaunchKernel> =
            lib.get(b"cuLaunchKernel").ok()?;
        let cu_get_error_string: libloading::Symbol<FnCuGetErrorString> =
            lib.get(b"cuGetErrorString").ok()?;

        // Convert to raw function pointers (leaks the library, keeping it loaded)
        let api = CudaApi {
            cuInit: *cu_init,
            cuDeviceGetCount: *cu_device_get_count,
            cuDeviceGet: *cu_device_get,
            cuDeviceGetName: *cu_device_get_name,
            cuDeviceGetAttribute: *cu_device_get_attribute,
            cuDeviceTotalMem: *cu_device_total_mem,
            cuCtxCreate: *cu_ctx_create,
            cuCtxDestroy: *cu_ctx_destroy,
            cuCtxSetCurrent: *cu_ctx_set_current,
            cuCtxGetCurrent: *cu_ctx_get_current,
            cuCtxSynchronize: *cu_ctx_synchronize,
            cuMemAlloc: *cu_mem_alloc,
            cuMemFree: *cu_mem_free,
            cuMemcpyHtoD: *cu_memcpy_htod,
            cuMemcpyDtoH: *cu_memcpy_dtoh,
            cuMemcpyDtoD: *cu_memcpy_dtod,
            cuMemsetD8: *cu_memset_d8,
            cuMemGetInfo: *cu_mem_get_info,
            cuModuleLoadData: *cu_module_load_data,
            cuModuleUnload: *cu_module_unload,
            cuModuleGetFunction: *cu_module_get_function,
            cuStreamCreate: *cu_stream_create,
            cuStreamDestroy: *cu_stream_destroy,
            cuStreamSynchronize: *cu_stream_synchronize,
            cuLaunchKernel: *cu_launch_kernel,
            cuGetErrorString: *cu_get_error_string,
        };

        // Leak the library to keep it loaded for the lifetime of the program
        std::mem::forget(lib);

        tracing::info!("Successfully loaded CUDA driver library: {}", lib_name);
        Some(api)
    }
}

/// Get the CUDA API, loading it if necessary.
fn get_cuda_api() -> Option<&'static CudaApi> {
    CUDA_API.get_or_init(load_cuda_api).as_ref()
}

/// Convert CUDA error to GpuError.
fn cuda_error(result: CUresult, context: &str) -> GpuError {
    if let Some(api) = get_cuda_api() {
        let mut error_str: *const c_char = std::ptr::null();
        unsafe {
            (api.cuGetErrorString)(result, &mut error_str);
            if !error_str.is_null() {
                let msg = CStr::from_ptr(error_str).to_string_lossy();
                return GpuError::CudaError(result, format!("{}: {}", context, msg));
            }
        }
    }
    GpuError::CudaError(result, format!("{}: error code {}", context, result))
}

/// Check CUDA result and convert to GpuResult.
fn check_cuda(result: CUresult, context: &str) -> GpuResult<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(cuda_error(result, context))
    }
}

/// CUDA runtime implementation.
///
/// This provides access to CUDA functionality through dynamically loaded
/// CUDA driver library. If CUDA is not available, all operations will
/// return appropriate errors.
pub struct CudaRuntime {
    initialized: std::sync::atomic::AtomicBool,
    context: parking_lot::Mutex<Option<CUcontext>>,
    current_device: std::sync::atomic::AtomicI32,
}

impl CudaRuntime {
    /// Create a new CUDA runtime.
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: std::sync::atomic::AtomicBool::new(false),
            context: parking_lot::Mutex::new(None),
            current_device: std::sync::atomic::AtomicI32::new(0),
        }
    }

    /// Check if CUDA is available on this system.
    #[must_use]
    pub fn is_available() -> bool {
        get_cuda_api().is_some()
    }

    /// Get the CUDA API, returning an error if not available.
    fn api(&self) -> GpuResult<&'static CudaApi> {
        get_cuda_api()
            .ok_or_else(|| GpuError::NotSupported("CUDA driver not available".to_string()))
    }

    /// Ensure CUDA is initialized.
    fn ensure_initialized(&self) -> GpuResult<()> {
        if !self.initialized.load(std::sync::atomic::Ordering::SeqCst) {
            self.init()?;
        }
        Ok(())
    }

    /// Ensure a context exists for the current device.
    fn ensure_context(&self) -> GpuResult<()> {
        self.ensure_initialized()?;
        let mut ctx_guard = self.context.lock();
        if ctx_guard.is_none() {
            let api = self.api()?;
            let device = self
                .current_device
                .load(std::sync::atomic::Ordering::SeqCst);
            let mut ctx: CUcontext = std::ptr::null_mut();
            unsafe {
                check_cuda((api.cuCtxCreate)(&mut ctx, 0, device), "cuCtxCreate")?;
            }
            *ctx_guard = Some(ctx);
        }
        Ok(())
    }
}

impl Default for CudaRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuRuntime for CudaRuntime {
    fn name(&self) -> &'static str {
        "CUDA"
    }

    fn init(&self) -> GpuResult<()> {
        let api = self.api()?;
        unsafe {
            check_cuda((api.cuInit)(0), "cuInit")?;
        }
        self.initialized
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn enumerate_devices(&self) -> GpuResult<Vec<DeviceInfo>> {
        self.ensure_initialized()?;
        let api = self.api()?;

        let mut count: c_int = 0;
        unsafe {
            check_cuda((api.cuDeviceGetCount)(&mut count), "cuDeviceGetCount")?;
        }

        let mut devices = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            let mut device: CUdevice = 0;
            unsafe {
                check_cuda((api.cuDeviceGet)(&mut device, ordinal), "cuDeviceGet")?;
            }

            // Get device name
            let mut name_buf = [0i8; 256];
            unsafe {
                check_cuda(
                    (api.cuDeviceGetName)(name_buf.as_mut_ptr(), 256, device),
                    "cuDeviceGetName",
                )?;
            }
            let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();

            // Get compute capability
            let mut major: c_int = 0;
            let mut minor: c_int = 0;
            unsafe {
                check_cuda(
                    (api.cuDeviceGetAttribute)(
                        &mut major,
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        device,
                    ),
                    "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)",
                )?;
                check_cuda(
                    (api.cuDeviceGetAttribute)(
                        &mut minor,
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        device,
                    ),
                    "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)",
                )?;
            }

            // Get total memory
            let mut total_mem: usize = 0;
            unsafe {
                check_cuda(
                    (api.cuDeviceTotalMem)(&mut total_mem, device),
                    "cuDeviceTotalMem",
                )?;
            }

            // Get multiprocessor count
            let mut sm_count: c_int = 0;
            unsafe {
                check_cuda(
                    (api.cuDeviceGetAttribute)(
                        &mut sm_count,
                        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                        device,
                    ),
                    "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)",
                )?;
            }

            devices.push(DeviceInfo {
                id: DeviceId(ordinal as u32),
                kind: DeviceKind::Cuda,
                name,
                compute_capability: (major as u32, minor as u32),
                total_memory: total_mem,
                multiprocessor_count: sm_count as u32,
            });
        }

        Ok(devices)
    }

    fn set_device(&self, device: DeviceId) -> GpuResult<()> {
        self.ensure_initialized()?;

        // Store the new device ordinal
        self.current_device
            .store(device.0 as i32, std::sync::atomic::Ordering::SeqCst);

        // If we have an existing context, destroy it and create a new one
        let mut ctx_guard = self.context.lock();
        if let Some(old_ctx) = ctx_guard.take() {
            let api = self.api()?;
            unsafe {
                // Best effort to destroy old context
                let _ = (api.cuCtxDestroy)(old_ctx);
            }
        }
        drop(ctx_guard);

        // Create new context for this device
        self.ensure_context()?;
        Ok(())
    }

    fn get_device(&self) -> GpuResult<DeviceId> {
        let device = self
            .current_device
            .load(std::sync::atomic::Ordering::SeqCst);
        Ok(DeviceId(device as u32))
    }

    fn malloc(&self, size: usize) -> GpuResult<DevicePtr> {
        self.ensure_context()?;
        let api = self.api()?;

        let mut dptr: CUdeviceptr = 0;
        unsafe {
            check_cuda((api.cuMemAlloc)(&mut dptr, size), "cuMemAlloc")?;
        }
        Ok(DevicePtr(dptr))
    }

    fn free(&self, ptr: DevicePtr) -> GpuResult<()> {
        let api = self.api()?;
        unsafe {
            check_cuda((api.cuMemFree)(ptr.0), "cuMemFree")?;
        }
        Ok(())
    }

    fn memset(&self, ptr: DevicePtr, value: u8, size: usize) -> GpuResult<()> {
        self.ensure_context()?;
        let api = self.api()?;
        unsafe {
            check_cuda((api.cuMemsetD8)(ptr.0, value, size), "cuMemsetD8")?;
        }
        Ok(())
    }

    fn memcpy_host_to_device(&self, dst: DevicePtr, src: *const u8, size: usize) -> GpuResult<()> {
        self.ensure_context()?;
        let api = self.api()?;
        unsafe {
            check_cuda(
                (api.cuMemcpyHtoD)(dst.0, src as *const c_void, size),
                "cuMemcpyHtoD",
            )?;
        }
        Ok(())
    }

    fn memcpy_device_to_host(&self, dst: *mut u8, src: DevicePtr, size: usize) -> GpuResult<()> {
        self.ensure_context()?;
        let api = self.api()?;
        unsafe {
            check_cuda(
                (api.cuMemcpyDtoH)(dst as *mut c_void, src.0, size),
                "cuMemcpyDtoH",
            )?;
        }
        Ok(())
    }

    fn memcpy_device_to_device(
        &self,
        dst: DevicePtr,
        src: DevicePtr,
        size: usize,
    ) -> GpuResult<()> {
        self.ensure_context()?;
        let api = self.api()?;
        unsafe {
            check_cuda((api.cuMemcpyDtoD)(dst.0, src.0, size), "cuMemcpyDtoD")?;
        }
        Ok(())
    }

    fn create_stream(&self) -> GpuResult<u64> {
        self.ensure_context()?;
        let api = self.api()?;

        let mut stream: CUstream = std::ptr::null_mut();
        unsafe {
            check_cuda((api.cuStreamCreate)(&mut stream, 0), "cuStreamCreate")?;
        }
        Ok(stream as u64)
    }

    fn destroy_stream(&self, stream: u64) -> GpuResult<()> {
        let api = self.api()?;
        unsafe {
            check_cuda((api.cuStreamDestroy)(stream as CUstream), "cuStreamDestroy")?;
        }
        Ok(())
    }

    fn synchronize_stream(&self, stream: u64) -> GpuResult<()> {
        let api = self.api()?;
        unsafe {
            check_cuda(
                (api.cuStreamSynchronize)(stream as CUstream),
                "cuStreamSynchronize",
            )?;
        }
        Ok(())
    }

    fn device_synchronize(&self) -> GpuResult<()> {
        self.ensure_context()?;
        let api = self.api()?;
        unsafe {
            check_cuda((api.cuCtxSynchronize)(), "cuCtxSynchronize")?;
        }
        Ok(())
    }

    fn load_module(&self, code: &[u8]) -> GpuResult<u64> {
        self.ensure_context()?;
        let api = self.api()?;

        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            check_cuda(
                (api.cuModuleLoadData)(&mut module, code.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
        }
        Ok(module as u64)
    }

    fn unload_module(&self, module: u64) -> GpuResult<()> {
        let api = self.api()?;
        unsafe {
            check_cuda((api.cuModuleUnload)(module as CUmodule), "cuModuleUnload")?;
        }
        Ok(())
    }

    fn get_function(&self, module: u64, name: &str) -> GpuResult<u64> {
        let api = self.api()?;

        let c_name = CString::new(name)
            .map_err(|_| GpuError::InvalidParameter(format!("Invalid function name: {}", name)))?;

        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            check_cuda(
                (api.cuModuleGetFunction)(&mut func, module as CUmodule, c_name.as_ptr()),
                "cuModuleGetFunction",
            )?;
        }
        Ok(func as u64)
    }

    fn launch_kernel(
        &self,
        function: u64,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem: usize,
        stream: u64,
        args: &[u64],
    ) -> GpuResult<()> {
        self.ensure_context()?;
        let api = self.api()?;

        // Prepare kernel arguments as void pointers
        let mut arg_ptrs: Vec<*mut c_void> = args
            .iter()
            .map(|arg| arg as *const u64 as *mut c_void)
            .collect();

        unsafe {
            check_cuda(
                (api.cuLaunchKernel)(
                    function as CUfunction,
                    grid_dim.0,
                    grid_dim.1,
                    grid_dim.2,
                    block_dim.0,
                    block_dim.1,
                    block_dim.2,
                    shared_mem as c_uint,
                    stream as CUstream,
                    arg_ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
        }
        Ok(())
    }

    fn memory_info(&self) -> GpuResult<(usize, usize)> {
        self.ensure_context()?;
        let api = self.api()?;

        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            check_cuda((api.cuMemGetInfo)(&mut free, &mut total), "cuMemGetInfo")?;
        }
        Ok((free, total))
    }
}

// ============================================================================
// Global Runtime Instance
// ============================================================================

/// Global CUDA runtime instance for standalone functions.
static CUDA_RUNTIME: OnceLock<CudaRuntime> = OnceLock::new();

/// Get the global CUDA runtime, creating it if necessary.
fn get_runtime() -> &'static CudaRuntime {
    CUDA_RUNTIME.get_or_init(CudaRuntime::new)
}

// ============================================================================
// Standalone Functions
// ============================================================================

/// Check if CUDA is available on this system.
#[must_use]
pub fn is_available() -> bool {
    CudaRuntime::is_available()
}

/// Enumerate CUDA devices.
pub fn enumerate_devices() -> GpuResult<Vec<DeviceInfo>> {
    get_runtime().enumerate_devices()
}

/// Set the current CUDA device.
pub fn set_device(device: DeviceId) -> GpuResult<()> {
    get_runtime().set_device(device)
}

/// Get the current CUDA device.
pub fn get_device() -> GpuResult<DeviceId> {
    get_runtime().get_device()
}

/// Allocate CUDA device memory.
pub fn malloc(size: usize) -> GpuResult<DevicePtr> {
    get_runtime().malloc(size)
}

/// Free CUDA device memory.
pub fn free(ptr: DevicePtr) -> GpuResult<()> {
    get_runtime().free(ptr)
}

/// Set device memory to a value.
pub fn memset(ptr: DevicePtr, value: u8, size: usize) -> GpuResult<()> {
    get_runtime().memset(ptr, value, size)
}

/// Copy data from host to device.
pub fn memcpy_host_to_device(dst: DevicePtr, src: *const u8, size: usize) -> GpuResult<()> {
    get_runtime().memcpy_host_to_device(dst, src, size)
}

/// Copy data from device to host.
pub fn memcpy_device_to_host(dst: *mut u8, src: DevicePtr, size: usize) -> GpuResult<()> {
    get_runtime().memcpy_device_to_host(dst, src, size)
}

/// Copy data between device buffers.
pub fn memcpy_device_to_device(dst: DevicePtr, src: DevicePtr, size: usize) -> GpuResult<()> {
    get_runtime().memcpy_device_to_device(dst, src, size)
}

/// Create a CUDA stream.
pub fn create_stream() -> GpuResult<u64> {
    get_runtime().create_stream()
}

/// Destroy a CUDA stream.
pub fn destroy_stream(stream: u64) -> GpuResult<()> {
    get_runtime().destroy_stream(stream)
}

/// Synchronize a CUDA stream.
pub fn synchronize_stream(stream: u64) -> GpuResult<()> {
    get_runtime().synchronize_stream(stream)
}

/// Synchronize the device.
pub fn device_synchronize() -> GpuResult<()> {
    get_runtime().device_synchronize()
}

/// Get memory info (free, total).
pub fn memory_info() -> GpuResult<(usize, usize)> {
    get_runtime().memory_info()
}

/// Load a PTX module.
pub fn load_module(ptx: &[u8]) -> GpuResult<u64> {
    get_runtime().load_module(ptx)
}

/// Unload a module.
pub fn unload_module(module: u64) -> GpuResult<()> {
    get_runtime().unload_module(module)
}

/// Get a function from a module.
pub fn get_function(module: u64, name: &str) -> GpuResult<u64> {
    get_runtime().get_function(module, name)
}

/// Launch a kernel.
pub fn launch_kernel(
    _module: &CompiledModule,
    name: &str,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem: usize,
    stream: u64,
    args: &[DevicePtr],
) -> GpuResult<()> {
    // Load the module and get the function
    let runtime = get_runtime();
    let module_handle = runtime.load_module(&_module.code)?;
    let func = runtime.get_function(module_handle, name)?;

    // Convert DevicePtr args to u64 for kernel launch
    let arg_values: Vec<u64> = args.iter().map(|p| p.0).collect();

    let result = runtime.launch_kernel(func, grid_dim, block_dim, shared_mem, stream, &arg_values);

    // Unload module (best effort)
    let _ = runtime.unload_module(module_handle);

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_runtime_creation() {
        let runtime = CudaRuntime::new();
        assert_eq!(runtime.name(), "CUDA");
    }

    #[test]
    fn test_cuda_api_types() {
        // Verify CUDA type sizes match expected
        assert_eq!(
            std::mem::size_of::<CUresult>(),
            std::mem::size_of::<c_int>()
        );
        assert_eq!(
            std::mem::size_of::<CUdevice>(),
            std::mem::size_of::<c_int>()
        );
        assert_eq!(
            std::mem::size_of::<CUdeviceptr>(),
            std::mem::size_of::<u64>()
        );
    }

    #[test]
    fn test_cuda_constants() {
        // Verify important constants
        assert_eq!(CUDA_SUCCESS, 0);
        assert_eq!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 75);
        assert_eq!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 76);
    }

    #[test]
    fn test_is_available() {
        // Just verify it doesn't crash - actual availability depends on system
        let _ = CudaRuntime::is_available();
        let _ = is_available();
    }

    #[test]
    fn test_runtime_without_cuda() {
        // On systems without CUDA, operations should fail gracefully
        if !CudaRuntime::is_available() {
            let runtime = CudaRuntime::new();
            assert!(runtime.init().is_err());
            assert!(runtime.enumerate_devices().is_err());
            assert!(runtime.malloc(1024).is_err());
        }
    }

    #[test]
    fn test_error_messages() {
        // Test error creation
        let err = GpuError::NotSupported("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
    }
}
