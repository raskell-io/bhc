# bhc-gpu

GPU code generation and runtime for the Basel Haskell Compiler.

## Overview

This crate provides GPU code generation and runtime support for BHC, enabling tensor computations to execute on GPU devices with automatic kernel fusion.

## Architecture

```
                          ┌─────────────────────────────┐
                          │     Tensor IR Kernels       │
                          └─────────────┬───────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
          ┌─────────────────┐                    ┌─────────────────┐
          │  CPU Backend    │                    │  GPU Backend    │
          │  (LLVM IR)      │                    │  (PTX/AMDGCN)   │
          └─────────────────┘                    └────────┬────────┘
                                                          │
                                    ┌─────────────────────┴─────────────────────┐
                                    ▼                                           ▼
                          ┌─────────────────┐                        ┌─────────────────┐
                          │  CUDA Runtime   │                        │  ROCm Runtime   │
                          │  (cuBLAS, etc.) │                        │  (rocBLAS, etc.)│
                          └─────────────────┘                        └─────────────────┘
```

## Features

- `cuda` - Enable NVIDIA CUDA support
- `rocm` - Enable AMD ROCm/HIP support

## Key Types

| Type | Description |
|------|-------------|
| `GpuContext` | GPU device context |
| `DeviceBuffer<T>` | Device memory buffer |
| `GpuKernel` | Compiled GPU kernel |
| `DeviceId` | GPU device identifier |
| `LaunchConfig` | Kernel launch configuration |

## Usage

### Device Enumeration

```rust
use bhc_gpu::{available_devices, select_device, DeviceId};

// List available GPUs
let devices = available_devices();
for dev in &devices {
    println!("{}: {} ({} MB)",
        dev.id,
        dev.name,
        dev.memory_total / 1024 / 1024
    );
}

// Select a device and create context
let ctx = select_device(DeviceId(0))?;
```

### Memory Management

```rust
use bhc_gpu::{DeviceBuffer, MemcpyKind};

// Allocate device memory
let d_buf: DeviceBuffer<f32> = ctx.alloc(1024)?;

// Transfer data to device
ctx.copy_to_device(&host_data, &mut d_buf)?;

// Transfer data from device
ctx.copy_to_host(&d_buf, &mut host_output)?;

// Async transfers
let stream = ctx.create_stream()?;
ctx.copy_to_device_async(&host_data, &mut d_buf, &stream)?;
stream.synchronize()?;
```

### Kernel Compilation and Launch

```rust
use bhc_gpu::{GpuKernel, LaunchConfig};

// Compile kernel from Tensor IR
let kernel = ctx.compile_kernel(&tensor_ir_kernel)?;

// Configure launch parameters
let config = LaunchConfig {
    grid_dim: (1024, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem: 0,
};

// Launch kernel
ctx.launch(&kernel, config, &[&d_a, &d_b, &d_c])?;
```

### Matrix Multiplication with cuBLAS

```rust
use bhc_gpu::blas::GpuBlas;

let blas = ctx.blas()?;

// C = alpha * A * B + beta * C
blas.gemm(
    Transpose::NoTrans,
    Transpose::NoTrans,
    m, n, k,
    1.0,            // alpha
    &d_a, lda,      // A
    &d_b, ldb,      // B
    0.0,            // beta
    &mut d_c, ldc,  // C
)?;
```

## Supported Operations

### Memory Operations

| Operation | Description |
|-----------|-------------|
| `alloc` | Allocate device memory |
| `alloc_pitched` | Allocate 2D memory with pitch |
| `free` | Free device memory |
| `copy_to_device` | Host to device transfer |
| `copy_to_host` | Device to host transfer |
| `copy_device_to_device` | Device to device copy |
| `memset` | Set device memory |

### Kernel Operations

| Operation | Description |
|-----------|-------------|
| `compile_kernel` | Compile Tensor IR to GPU kernel |
| `launch` | Launch kernel with configuration |
| `launch_async` | Async kernel launch |

### BLAS Operations

| Level | Operations |
|-------|------------|
| 1 | axpy, dot, nrm2, scal |
| 2 | gemv, trsv |
| 3 | gemm, trsm, syrk |

## Launch Configuration

```rust
pub struct LaunchConfig {
    /// Grid dimensions (blocks)
    pub grid_dim: (u32, u32, u32),

    /// Block dimensions (threads per block)
    pub block_dim: (u32, u32, u32),

    /// Shared memory per block (bytes)
    pub shared_mem: usize,
}

impl LaunchConfig {
    // Automatic configuration for 1D kernels
    pub fn for_1d(n: usize) -> Self {
        let block = 256;
        let grid = (n + block - 1) / block;
        Self {
            grid_dim: (grid as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem: 0,
        }
    }
}
```

## Error Types

```rust
pub enum GpuError {
    /// Device not found
    DeviceNotFound(DeviceId),

    /// Out of device memory
    OutOfMemory { requested: usize, available: usize },

    /// Kernel compilation failed
    CompilationFailed(String),

    /// Kernel launch failed
    LaunchFailed(String),

    /// CUDA/ROCm runtime error
    RuntimeError(String),
}
```

## M7 Exit Criteria

- GPU device enumeration and selection works
- Device memory allocation/deallocation works
- Host↔device transfers work (sync and async)
- Basic kernels compile and execute
- Tensor IR kernels can target GPU backend
- Performance competitive with manual CUDA for matmul

## Design Notes

- One context per device
- Streams enable async execution
- Pinned host memory for fast transfers
- Kernel caching avoids recompilation
- cuBLAS/rocBLAS for optimized linear algebra

## Related Crates

- `bhc-tensor-ir` - Tensor IR kernels
- `bhc-target` - GPU target specifications
- `bhc-ffi` - FFI for CUDA/ROCm libraries
- `bhc-codegen` - CPU codegen counterpart

## Specification References

- H26-SPEC Section 7.5: GPU Execution
- CUDA Programming Guide
- ROCm HIP Programming Guide
