# bhc-target

Target specifications and platform support for the Basel Haskell Compiler.

## Overview

This crate defines target machine specifications including architecture, operating system, ABI, and CPU features. It supports cross-compilation to multiple platforms.

## Supported Targets

| Target | Architecture | Notes |
|--------|-------------|-------|
| `x86_64-unknown-linux-gnu` | x86-64 | Linux with glibc |
| `x86_64-unknown-linux-musl` | x86-64 | Linux with musl |
| `x86_64-apple-darwin` | x86-64 | macOS (Intel) |
| `aarch64-apple-darwin` | AArch64 | macOS (Apple Silicon) |
| `aarch64-unknown-linux-gnu` | AArch64 | Linux ARM64 |
| `wasm32-unknown-wasi` | WebAssembly | WASI |
| `nvptx64-nvidia-cuda` | NVIDIA PTX | CUDA GPU |
| `amdgcn-amd-amdhsa` | AMD GCN | ROCm GPU |

## Key Types

| Type | Description |
|------|-------------|
| `TargetSpec` | Complete target specification |
| `Arch` | Target architecture |
| `Os` | Target operating system |
| `Abi` | Target ABI |
| `CpuFeatures` | CPU feature flags |

## Usage

### Getting Target Information

```rust
use bhc_target::{TargetSpec, Arch, Os};

// Get the host target
let host = TargetSpec::host();
println!("Host: {}", host.triple());

// Create a specific target
let target = TargetSpec::parse("aarch64-apple-darwin")?;
assert_eq!(target.arch, Arch::Aarch64);
assert_eq!(target.os, Os::Darwin);
```

### Architecture Properties

```rust
use bhc_target::Arch;

let arch = Arch::X86_64;

// Pointer width
assert_eq!(arch.pointer_width(), 64);

// Natural alignment
assert_eq!(arch.natural_alignment(), 8);

// Architecture name
assert_eq!(arch.name(), "x86_64");
```

## Architectures

```rust
pub enum Arch {
    X86_64,    // AMD64
    Aarch64,   // 64-bit ARM
    Wasm32,    // 32-bit WebAssembly
    Wasm64,    // 64-bit WebAssembly
    Riscv32,   // 32-bit RISC-V
    Riscv64,   // 64-bit RISC-V
    Nvptx64,   // NVIDIA PTX
    Amdgcn,    // AMD GCN/RDNA
}
```

## Operating Systems

```rust
pub enum Os {
    Linux,
    Darwin,    // macOS/iOS
    Windows,
    Wasi,      // WebAssembly System Interface
    None,      // Bare metal
    Cuda,      // NVIDIA CUDA
    Amdhsa,    // AMD HSA
}
```

## CPU Features

```rust
use bhc_target::{CpuFeatures, X86Feature};

let mut features = CpuFeatures::default();

// Enable SIMD features
features.enable(X86Feature::Sse2);
features.enable(X86Feature::Avx);
features.enable(X86Feature::Avx2);
features.enable(X86Feature::Fma);

// Check features
if features.has(X86Feature::Avx512f) {
    println!("AVX-512 available");
}
```

### x86-64 Features

| Feature | Description |
|---------|-------------|
| `Sse` | SSE (128-bit SIMD) |
| `Sse2` | SSE2 (integer SIMD) |
| `Sse3` | SSE3 (horizontal ops) |
| `Sse41` | SSE4.1 |
| `Sse42` | SSE4.2 |
| `Avx` | AVX (256-bit SIMD) |
| `Avx2` | AVX2 (256-bit integer) |
| `Avx512f` | AVX-512 Foundation |
| `Fma` | Fused multiply-add |

### AArch64 Features

| Feature | Description |
|---------|-------------|
| `Neon` | NEON SIMD |
| `Sve` | Scalable Vector Extension |
| `Sve2` | SVE2 |

## Target Triple

Targets are specified using the standard triple format: `<arch>-<vendor>-<os>-<abi>`

```rust
use bhc_target::TargetSpec;

// Parse from string
let target = TargetSpec::parse("x86_64-unknown-linux-gnu")?;

// Access components
println!("Architecture: {}", target.arch);
println!("Vendor: {}", target.vendor);
println!("OS: {}", target.os);
println!("ABI: {}", target.abi);

// Get the triple string
println!("Triple: {}", target.triple());
```

## Data Layout

```rust
use bhc_target::TargetSpec;

let target = TargetSpec::parse("x86_64-unknown-linux-gnu")?;

// Get data layout for LLVM
let layout = target.data_layout();
// e.g., "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
```

## Design Notes

- Target specifications are immutable after creation
- Host target is detected at compile time
- GPU targets require specific feature sets
- WASM targets have special handling for linear memory

## Related Crates

- `bhc-codegen` - Uses targets for code generation
- `bhc-session` - Session holds target configuration
- `bhc-driver` - Cross-compilation support
- `bhc-gpu` - GPU target specifications

## Specification References

- H26-SPEC Section 12: Target Platforms
- LLVM Target Triple Documentation
