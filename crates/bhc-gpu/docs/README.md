# bhc-gpu

GPU code generation and runtime support for the Basel Haskell Compiler.

## Overview

`bhc-gpu` enables tensor computations on GPU devices with automatic kernel fusion across host/device boundaries. Features:

- **Device management**: GPU enumeration, selection, context creation
- **Memory management**: Device buffers, async transfers, pinned memory
- **Kernel compilation**: PTX (CUDA) and AMDGCN (ROCm) generation
- **Kernel launch**: Optimal grid/block configuration

## Architecture

```
                      ┌─────────────────────────┐
                      │   Tensor IR Kernels     │
                      └───────────┬─────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
      ┌─────────────────┐                ┌─────────────────┐
      │  CPU Backend    │                │  GPU Backend    │
      │  (LLVM IR)      │                │  (PTX/AMDGCN)   │
      └─────────────────┘                └────────┬────────┘
                                                  │
                            ┌─────────────────────┴─────────────────────┐
                            ▼                                           ▼
                  ┌─────────────────┐                        ┌─────────────────┐
                  │  CUDA Runtime   │                        │  ROCm Runtime   │
                  │  (cuBLAS, etc.) │                        │  (rocBLAS, etc.)│
                  └─────────────────┘                        └─────────────────┘
```

## Features

- `cuda`: Enable NVIDIA CUDA support
- `rocm`: Enable AMD ROCm/HIP support

## Core Types

| Type | Description |
|------|-------------|
| `GpuContext` | GPU execution context |
| `DeviceInfo` | GPU device information |
| `DeviceBuffer` | Device memory buffer |
| `GpuKernel` | Compiled kernel |
| `TransferQueue` | Async transfer queue |

## Device Enumeration

```rust
use bhc_gpu::{available_devices, select_device, DeviceId};

// List all GPUs
let devices = available_devices();
for dev in &devices {
    println!("{}: {} ({} MB)",
        dev.id.0,
        dev.name,
        dev.memory_total / 1024 / 1024
    );
}

// Select a specific device
let ctx = select_device(DeviceId(0))?;

// Or use the default device
let ctx = bhc_gpu::default_device()?;
```

## Device Information

```rust
pub struct DeviceInfo {
    /// Device identifier
    pub id: DeviceId,
    /// Device name
    pub name: String,
    /// Device kind (CUDA, ROCm, Mock)
    pub kind: DeviceKind,
    /// Total memory in bytes
    pub memory_total: usize,
    /// Number of multiprocessors/compute units
    pub multiprocessor_count: u32,
    /// Compute capability (CUDA) or GCN version (ROCm)
    pub compute_version: (u32, u32),
}

pub enum DeviceKind {
    Cuda,
    Rocm,
    Mock,  // For testing without GPU
}
```

## Memory Management

```rust
// Allocate device memory
let d_buf: DeviceBuffer<f32> = ctx.alloc(1024)?;

// Copy host to device
let host_data: Vec<f32> = vec![1.0; 1024];
ctx.copy_to_device(&host_data, &mut d_buf)?;

// Copy device to host
let mut result = vec![0.0f32; 1024];
ctx.copy_to_host(&d_buf, &mut result)?;

// Async transfers
let queue = ctx.transfer_queue();
let handle = queue.async_copy_to_device(&host_data, &mut d_buf)?;
handle.wait()?;
```

## Kernel Compilation

```rust
use bhc_gpu::{KernelConfig, LaunchConfig};

// Compile a kernel from Tensor IR
let kernel = ctx.compile_kernel(&tensor_ir_kernel)?;

// Configure launch parameters
let launch_config = LaunchConfig {
    grid_dim: (256, 1, 1),
    block_dim: (256, 1, 1),
    shared_memory: 0,
    stream: None,
};

// Launch kernel
ctx.launch(&kernel, launch_config, &[&d_input, &d_output])?;
```

## GPU Backend

Implements `CodegenBackend` for GPU targets:

```rust
use bhc_gpu::GpuBackend;
use bhc_codegen::{CodegenBackend, CodegenConfig};

let backend = GpuBackend::new().expect("No GPU available");

// Check available features
if backend.has_cuda() {
    println!("CUDA available");
}
if backend.has_rocm() {
    println!("ROCm available");
}

// Create codegen context
let config = CodegenConfig::for_target(gpu_target);
let ctx = backend.create_context(config)?;
```

## Error Handling

```rust
pub enum GpuError {
    /// No devices available
    NoDevicesAvailable,
    /// Invalid device ID
    InvalidDevice(DeviceId),
    /// Allocation failed
    AllocationFailed { size: usize },
    /// Out of memory
    OutOfMemory { requested: usize, available: usize },
    /// Transfer error
    TransferError(String),
    /// Kernel compilation error
    CompilationError(String),
    /// Kernel launch error
    LaunchError(String),
    /// Feature not supported
    NotSupported(String),
    /// Runtime error
    RuntimeError(String),
    // Platform-specific errors
    #[cfg(feature = "cuda")]
    CudaError { code: i32, message: String },
    #[cfg(feature = "rocm")]
    RocmError { code: i32, message: String },
}
```

## Example: Matrix Multiplication

```rust
use bhc_gpu::{default_device, DeviceBuffer};

let ctx = default_device()?;

// Allocate matrices (1024x1024)
let n = 1024;
let mut d_a: DeviceBuffer<f32> = ctx.alloc(n * n)?;
let mut d_b: DeviceBuffer<f32> = ctx.alloc(n * n)?;
let mut d_c: DeviceBuffer<f32> = ctx.alloc(n * n)?;

// Copy input data
ctx.copy_to_device(&host_a, &mut d_a)?;
ctx.copy_to_device(&host_b, &mut d_b)?;

// Compile and launch matmul kernel
let kernel = ctx.compile_kernel(&matmul_kernel)?;
ctx.launch(&kernel, launch_config, &[&d_a, &d_b, &d_c])?;

// Copy result back
ctx.copy_to_host(&d_c, &mut host_c)?;
```

## M7 Exit Criteria

- GPU device enumeration and selection works
- Device memory allocation/deallocation works
- Host<->device transfers work (sync and async)
- Basic kernels compile and execute (CUDA at minimum)
- Tensor IR kernels can target GPU backend
- Performance competitive with manual CUDA code for matmul

## Submodules

| Module | Description |
|--------|-------------|
| `codegen` | PTX/AMDGCN code generation |
| `context` | GPU context management |
| `device` | Device enumeration |
| `kernel` | Kernel compilation and launch |
| `memory` | Device memory management |
| `runtime` | CUDA/ROCm runtime bindings |
| `transfer` | Host-device transfers |

## See Also

- `bhc-tensor-ir`: Tensor IR that targets GPU
- `bhc-codegen`: General code generation
- `bhc-target`: Target specifications
- CUDA Programming Guide
- ROCm/HIP Documentation
