# bhc-target

Target specification and platform support for the Basel Haskell Compiler.

## Overview

`bhc-target` defines compilation targets including:

- **Target triples**: Architecture, vendor, OS, ABI
- **CPU features**: SIMD capabilities, extensions
- **Data layout**: Sizes, alignments, endianness
- **Pre-defined targets**: Common platform configurations

## Core Types

| Type | Description |
|------|-------------|
| `TargetSpec` | Complete target specification |
| `Arch` | CPU architecture |
| `Os` | Operating system |
| `Vendor` | Platform vendor |
| `Abi` | Application binary interface |
| `CpuFeatures` | CPU feature flags |

## Target Triple

```
<arch>-<vendor>-<os>-<abi>

Examples:
  x86_64-unknown-linux-gnu
  aarch64-apple-darwin
  wasm32-unknown-unknown
  x86_64-pc-windows-msvc
```

## Architecture

```rust
pub enum Arch {
    /// x86 64-bit
    X86_64,
    /// x86 32-bit
    X86,
    /// ARM 64-bit
    Aarch64,
    /// ARM 32-bit
    Arm,
    /// WebAssembly 32-bit
    Wasm32,
    /// WebAssembly 64-bit
    Wasm64,
    /// RISC-V 64-bit
    Riscv64,
    /// RISC-V 32-bit
    Riscv32,
}

impl Arch {
    /// Pointer size in bits
    pub fn pointer_width(&self) -> u32 {
        match self {
            Arch::X86_64 | Arch::Aarch64 | Arch::Wasm64 | Arch::Riscv64 => 64,
            Arch::X86 | Arch::Arm | Arch::Wasm32 | Arch::Riscv32 => 32,
        }
    }

    /// Default alignment
    pub fn default_align(&self) -> u32;

    /// Endianness
    pub fn endian(&self) -> Endian;
}
```

## Operating System

```rust
pub enum Os {
    Linux,
    Darwin,
    Windows,
    FreeBsd,
    NetBsd,
    OpenBsd,
    Wasi,
    None,  // Bare metal / WASM
}

impl Os {
    /// Object file format
    pub fn object_format(&self) -> ObjectFormat {
        match self {
            Os::Linux | Os::FreeBsd | Os::NetBsd | Os::OpenBsd => ObjectFormat::Elf,
            Os::Darwin => ObjectFormat::MachO,
            Os::Windows => ObjectFormat::Coff,
            Os::Wasi | Os::None => ObjectFormat::Wasm,
        }
    }

    /// Dynamic library extension
    pub fn dylib_ext(&self) -> &'static str {
        match self {
            Os::Darwin => "dylib",
            Os::Windows => "dll",
            _ => "so",
        }
    }
}
```

## Vendor

```rust
pub enum Vendor {
    Unknown,
    Apple,
    Pc,
    Nvidia,
    Amd,
}
```

## ABI

```rust
pub enum Abi {
    /// GNU ABI (Linux default)
    Gnu,
    /// musl libc
    Musl,
    /// Android
    Android,
    /// MSVC (Windows)
    Msvc,
    /// MinGW (Windows with GNU)
    Mingw,
    /// macOS/iOS
    Darwin,
    /// EABI (embedded)
    Eabi,
    /// No specific ABI
    None,
}
```

## CPU Features

```rust
pub struct CpuFeatures {
    // x86 features
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,

    // ARM features
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,

    // WASM features
    pub simd128: bool,

    // General
    pub atomics: bool,
}

impl CpuFeatures {
    /// Detect host CPU features
    pub fn detect_host() -> Self;

    /// Features for a baseline target
    pub fn baseline(arch: Arch) -> Self;

    /// Natural vector width in bits
    pub fn vector_width(&self) -> u32 {
        if self.avx512f { 512 }
        else if self.avx || self.avx2 { 256 }
        else if self.sse2 || self.neon || self.simd128 { 128 }
        else { 0 }
    }
}
```

## Target Specification

```rust
pub struct TargetSpec {
    /// Target triple string
    pub triple: String,
    /// Architecture
    pub arch: Arch,
    /// Vendor
    pub vendor: Vendor,
    /// Operating system
    pub os: Os,
    /// ABI
    pub abi: Abi,
    /// CPU features
    pub features: CpuFeatures,
    /// Data layout (LLVM format)
    pub data_layout: String,
    /// Linker flavor
    pub linker_flavor: LinkerFlavor,
    /// Default CPU
    pub cpu: String,
}

impl TargetSpec {
    /// Parse from triple string
    pub fn parse(triple: &str) -> Result<Self, ParseError>;

    /// Get host target
    pub fn host() -> Self;

    /// Pointer size in bytes
    pub fn pointer_size(&self) -> usize {
        (self.arch.pointer_width() / 8) as usize
    }
}
```

## Data Layout

LLVM-compatible data layout string:

```rust
impl TargetSpec {
    /// Generate LLVM data layout string
    pub fn data_layout(&self) -> String {
        // Format: e-m:e-p:64:64-i64:64-i128:128-n8:16:32:64-S128
        // e = little endian
        // m:e = ELF mangling
        // p:64:64 = pointer 64-bit, 64-bit aligned
        // i64:64 = i64 is 64-bit aligned
        // n8:16:32:64 = native integer widths
        // S128 = stack alignment 128-bit
    }
}
```

## Pre-defined Targets

```rust
pub mod targets {
    use super::*;

    pub fn x86_64_unknown_linux_gnu() -> TargetSpec {
        TargetSpec {
            triple: "x86_64-unknown-linux-gnu".into(),
            arch: Arch::X86_64,
            vendor: Vendor::Unknown,
            os: Os::Linux,
            abi: Abi::Gnu,
            features: CpuFeatures::baseline(Arch::X86_64),
            cpu: "x86-64".into(),
            ..Default::default()
        }
    }

    pub fn aarch64_apple_darwin() -> TargetSpec {
        TargetSpec {
            triple: "aarch64-apple-darwin".into(),
            arch: Arch::Aarch64,
            vendor: Vendor::Apple,
            os: Os::Darwin,
            abi: Abi::Darwin,
            features: CpuFeatures {
                neon: true,
                ..CpuFeatures::baseline(Arch::Aarch64)
            },
            cpu: "apple-m1".into(),
            ..Default::default()
        }
    }

    pub fn wasm32_unknown_unknown() -> TargetSpec {
        TargetSpec {
            triple: "wasm32-unknown-unknown".into(),
            arch: Arch::Wasm32,
            vendor: Vendor::Unknown,
            os: Os::None,
            abi: Abi::None,
            features: CpuFeatures {
                simd128: true,
                ..Default::default()
            },
            cpu: "generic".into(),
            ..Default::default()
        }
    }
}
```

## Parsing Target Triples

```rust
pub fn parse_triple(triple: &str) -> Result<TargetSpec, ParseError> {
    let parts: Vec<&str> = triple.split('-').collect();

    let arch = parse_arch(parts.get(0).ok_or(ParseError::MissingArch)?)?;
    let vendor = parts.get(1).map(|s| parse_vendor(s)).unwrap_or(Ok(Vendor::Unknown))?;
    let os = parts.get(2).map(|s| parse_os(s)).unwrap_or(Ok(Os::None))?;
    let abi = parts.get(3).map(|s| parse_abi(s)).unwrap_or_else(|| default_abi(&os));

    Ok(TargetSpec {
        triple: triple.to_string(),
        arch,
        vendor,
        os,
        abi,
        features: CpuFeatures::baseline(arch),
        ..Default::default()
    })
}
```

## Linker Configuration

```rust
pub enum LinkerFlavor {
    /// GNU ld
    Ld,
    /// LLVM lld
    Lld,
    /// macOS linker
    Darwin,
    /// MSVC link.exe
    Msvc,
    /// wasm-ld
    WasmLld,
}

impl TargetSpec {
    pub fn linker_args(&self) -> Vec<String> {
        match self.linker_flavor {
            LinkerFlavor::Ld => vec!["-z".into(), "now".into()],
            LinkerFlavor::Darwin => vec!["-dead_strip".into()],
            LinkerFlavor::Msvc => vec!["/OPT:REF".into()],
            _ => vec![],
        }
    }
}
```

## Quick Start

```rust
use bhc_target::{TargetSpec, Arch, CpuFeatures};

// Parse a target triple
let target = TargetSpec::parse("x86_64-unknown-linux-gnu")?;

// Check architecture
if target.arch == Arch::X86_64 {
    // Use x86_64-specific code paths
}

// Check for SIMD support
if target.features.avx2 {
    // Use AVX2 vectorization
}

// Get host target
let host = TargetSpec::host();
println!("Compiling on: {}", host.triple);

// Cross-compilation
let cross = TargetSpec::parse("aarch64-apple-darwin")?;
```

## Feature Detection

```rust
use bhc_target::CpuFeatures;

// Detect host features at runtime
let features = CpuFeatures::detect_host();

if features.avx2 {
    println!("AVX2 available, vector width: 256 bits");
} else if features.sse2 {
    println!("SSE2 available, vector width: 128 bits");
}

// Check for FMA support
if features.fma {
    // Use fused multiply-add
}
```

## See Also

- `bhc-codegen`: Uses target for code generation
- `bhc-driver`: Target selection and configuration
- `bhc-loop-ir`: Uses target for vectorization decisions
- LLVM Target Triple documentation
