//! Target specifications and platform support for BHC.
//!
//! This crate defines target machine specifications including architecture,
//! operating system, ABI, and CPU features. It supports cross-compilation
//! to multiple platforms.
//!
//! # Supported Targets
//!
//! - **x86_64**: Linux, macOS, Windows
//! - **aarch64**: Linux, macOS (Apple Silicon)
//! - **wasm32**: WebAssembly (WASI)
//!
//! # Target Triple Format
//!
//! Targets are specified using the standard triple format:
//! `<arch>-<vendor>-<os>-<abi>`
//!
//! For example:
//! - `x86_64-unknown-linux-gnu`
//! - `aarch64-apple-darwin`
//! - `wasm32-unknown-wasi`

#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Target architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Arch {
    /// x86-64 (AMD64).
    X86_64,
    /// 64-bit ARM.
    Aarch64,
    /// 32-bit WebAssembly.
    Wasm32,
    /// 64-bit WebAssembly.
    Wasm64,
    /// 32-bit RISC-V.
    Riscv32,
    /// 64-bit RISC-V.
    Riscv64,
    /// NVIDIA PTX (64-bit).
    Nvptx64,
    /// AMD GCN/RDNA.
    Amdgcn,
}

impl Arch {
    /// Get the pointer width in bits for this architecture.
    #[must_use]
    pub const fn pointer_width(self) -> u32 {
        match self {
            Self::X86_64
            | Self::Aarch64
            | Self::Wasm64
            | Self::Riscv64
            | Self::Nvptx64
            | Self::Amdgcn => 64,
            Self::Wasm32 | Self::Riscv32 => 32,
        }
    }

    /// Get the natural alignment for this architecture.
    #[must_use]
    pub const fn natural_alignment(self) -> u32 {
        match self {
            Self::X86_64
            | Self::Aarch64
            | Self::Wasm64
            | Self::Riscv64
            | Self::Nvptx64
            | Self::Amdgcn => 8,
            Self::Wasm32 | Self::Riscv32 => 4,
        }
    }

    /// Get the name of this architecture.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
            Self::Wasm32 => "wasm32",
            Self::Wasm64 => "wasm64",
            Self::Riscv32 => "riscv32",
            Self::Riscv64 => "riscv64",
            Self::Nvptx64 => "nvptx64",
            Self::Amdgcn => "amdgcn",
        }
    }

    /// Check if this architecture supports SIMD.
    #[must_use]
    pub const fn has_simd(self) -> bool {
        matches!(
            self,
            Self::X86_64
                | Self::Aarch64
                | Self::Wasm32
                | Self::Wasm64
                | Self::Nvptx64
                | Self::Amdgcn
        )
    }

    /// Check if this is a GPU architecture.
    #[must_use]
    pub const fn is_gpu(self) -> bool {
        matches!(self, Self::Nvptx64 | Self::Amdgcn)
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Target operating system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Os {
    /// Linux.
    Linux,
    /// macOS (Darwin).
    MacOs,
    /// Windows.
    Windows,
    /// FreeBSD.
    FreeBsd,
    /// WASI (WebAssembly System Interface).
    Wasi,
    /// Bare metal (no OS).
    None,
}

impl Os {
    /// Get the name of this OS.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::MacOs => "darwin",
            Self::Windows => "windows",
            Self::FreeBsd => "freebsd",
            Self::Wasi => "wasi",
            Self::None => "none",
        }
    }

    /// Check if this OS is Unix-like.
    #[must_use]
    pub const fn is_unix(self) -> bool {
        matches!(self, Self::Linux | Self::MacOs | Self::FreeBsd)
    }

    /// Check if this OS supports dynamic linking.
    #[must_use]
    pub const fn supports_dynamic_linking(self) -> bool {
        !matches!(self, Self::Wasi | Self::None)
    }
}

impl fmt::Display for Os {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Target vendor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Vendor {
    /// Unknown vendor.
    Unknown,
    /// Apple.
    Apple,
    /// PC (generic).
    Pc,
}

impl Vendor {
    /// Get the name of this vendor.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Apple => "apple",
            Self::Pc => "pc",
        }
    }
}

impl fmt::Display for Vendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Target ABI (Application Binary Interface).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Abi {
    /// GNU ABI.
    Gnu,
    /// musl libc.
    Musl,
    /// MSVC.
    Msvc,
    /// No specific ABI.
    None,
}

impl Abi {
    /// Get the name of this ABI.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gnu => "gnu",
            Self::Musl => "musl",
            Self::Msvc => "msvc",
            Self::None => "",
        }
    }
}

impl fmt::Display for Abi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Endianness of the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Endianness {
    /// Little-endian byte order.
    Little,
    /// Big-endian byte order.
    Big,
}

/// CPU features that can be enabled.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuFeatures {
    /// SSE4.2 support (x86_64).
    pub sse42: bool,
    /// AVX support (x86_64).
    pub avx: bool,
    /// AVX2 support (x86_64).
    pub avx2: bool,
    /// AVX-512 support (x86_64).
    pub avx512: bool,
    /// NEON support (aarch64).
    pub neon: bool,
    /// SVE support (aarch64).
    pub sve: bool,
    /// SIMD128 support (wasm).
    pub simd128: bool,
}

impl CpuFeatures {
    /// Create a feature set with no features enabled.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            sse42: false,
            avx: false,
            avx2: false,
            avx512: false,
            neon: false,
            sve: false,
            simd128: false,
        }
    }

    /// Create a feature set with default features for the given architecture.
    #[must_use]
    pub fn default_for_arch(arch: Arch) -> Self {
        let mut features = Self::none();
        match arch {
            Arch::X86_64 => {
                features.sse42 = true;
            }
            Arch::Aarch64 => {
                features.neon = true;
            }
            Arch::Wasm32 | Arch::Wasm64 => {
                features.simd128 = true;
            }
            _ => {}
        }
        features
    }

    /// Get the maximum SIMD vector width in bits.
    #[must_use]
    pub const fn max_vector_width(&self) -> u32 {
        if self.avx512 {
            512
        } else if self.avx || self.avx2 {
            256
        } else if self.sse42 || self.neon || self.simd128 || self.sve {
            128
        } else {
            64
        }
    }
}

/// A complete target specification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetSpec {
    /// Target architecture.
    pub arch: Arch,
    /// Target vendor.
    pub vendor: Vendor,
    /// Target operating system.
    pub os: Os,
    /// Target ABI.
    pub abi: Abi,
    /// Endianness.
    pub endianness: Endianness,
    /// CPU features.
    pub features: CpuFeatures,
    /// Data layout string (LLVM format).
    pub data_layout: String,
}

impl TargetSpec {
    /// Get the target triple string.
    #[must_use]
    pub fn triple(&self) -> String {
        if self.abi == Abi::None {
            format!("{}-{}-{}", self.arch, self.vendor, self.os)
        } else {
            format!("{}-{}-{}-{}", self.arch, self.vendor, self.os, self.abi)
        }
    }

    /// Get the pointer width in bytes.
    #[must_use]
    pub const fn pointer_width(&self) -> u32 {
        self.arch.pointer_width() / 8
    }
}

impl fmt::Display for TargetSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.triple())
    }
}

/// Errors that can occur when parsing target specifications.
#[derive(Debug, Error)]
pub enum TargetError {
    /// Unknown architecture.
    #[error("unknown architecture: {0}")]
    UnknownArch(String),
    /// Unknown operating system.
    #[error("unknown operating system: {0}")]
    UnknownOs(String),
    /// Unknown vendor.
    #[error("unknown vendor: {0}")]
    UnknownVendor(String),
    /// Unknown ABI.
    #[error("unknown ABI: {0}")]
    UnknownAbi(String),
    /// Invalid target triple format.
    #[error("invalid target triple: {0}")]
    InvalidTriple(String),
}

/// Parse an architecture from a string.
fn parse_arch(s: &str) -> Result<Arch, TargetError> {
    match s {
        "x86_64" | "amd64" => Ok(Arch::X86_64),
        "aarch64" | "arm64" => Ok(Arch::Aarch64),
        "wasm32" => Ok(Arch::Wasm32),
        "wasm64" => Ok(Arch::Wasm64),
        "riscv32" => Ok(Arch::Riscv32),
        "riscv64" => Ok(Arch::Riscv64),
        "nvptx64" | "nvptx" => Ok(Arch::Nvptx64),
        "amdgcn" | "gcn" => Ok(Arch::Amdgcn),
        _ => Err(TargetError::UnknownArch(s.to_string())),
    }
}

/// Parse an OS from a string.
fn parse_os(s: &str) -> Result<Os, TargetError> {
    match s {
        "linux" => Ok(Os::Linux),
        "darwin" | "macos" => Ok(Os::MacOs),
        "windows" | "win32" => Ok(Os::Windows),
        "freebsd" => Ok(Os::FreeBsd),
        "wasi" => Ok(Os::Wasi),
        "none" => Ok(Os::None),
        _ => Err(TargetError::UnknownOs(s.to_string())),
    }
}

/// Parse a vendor from a string.
fn parse_vendor(s: &str) -> Result<Vendor, TargetError> {
    match s {
        "unknown" => Ok(Vendor::Unknown),
        "apple" => Ok(Vendor::Apple),
        "pc" => Ok(Vendor::Pc),
        _ => Err(TargetError::UnknownVendor(s.to_string())),
    }
}

/// Parse an ABI from a string.
fn parse_abi(s: &str) -> Result<Abi, TargetError> {
    match s {
        "gnu" => Ok(Abi::Gnu),
        "musl" => Ok(Abi::Musl),
        "msvc" => Ok(Abi::Msvc),
        "" => Ok(Abi::None),
        _ => Err(TargetError::UnknownAbi(s.to_string())),
    }
}

/// Parse a target triple string into a target specification.
///
/// # Errors
///
/// Returns an error if the triple format is invalid or contains unknown components.
pub fn parse_triple(triple: &str) -> Result<TargetSpec, TargetError> {
    let parts: Vec<&str> = triple.split('-').collect();

    if parts.len() < 3 {
        return Err(TargetError::InvalidTriple(triple.to_string()));
    }

    let arch = parse_arch(parts[0])?;
    let vendor = parse_vendor(parts[1])?;
    let os = parse_os(parts[2])?;
    let abi = if parts.len() > 3 {
        parse_abi(parts[3])?
    } else {
        Abi::None
    };

    let endianness = Endianness::Little; // All supported targets are little-endian
    let features = CpuFeatures::default_for_arch(arch);

    let data_layout = generate_data_layout(arch, &features);

    Ok(TargetSpec {
        arch,
        vendor,
        os,
        abi,
        endianness,
        features,
        data_layout,
    })
}

/// Generate the LLVM data layout string for a target.
fn generate_data_layout(arch: Arch, _features: &CpuFeatures) -> String {
    match arch {
        Arch::X86_64 => {
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128".to_string()
        }
        Arch::Aarch64 => "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        Arch::Wasm32 => "e-m:e-p:32:32-i64:64-n32:64-S128".to_string(),
        Arch::Wasm64 => "e-m:e-p:64:64-i64:64-n32:64-S128".to_string(),
        Arch::Riscv32 => "e-m:e-p:32:32-i64:64-n32-S128".to_string(),
        Arch::Riscv64 => "e-m:e-p:64:64-i64:64-i128:128-n64-S128".to_string(),
        Arch::Nvptx64 => {
            // NVIDIA PTX data layout (64-bit)
            "e-i64:64-i128:128-v16:16-v32:32-n16:32:64".to_string()
        }
        Arch::Amdgcn => {
            // AMD GCN data layout
            "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7".to_string()
        }
    }
}

/// Get the target specification for the host machine.
#[must_use]
pub fn host_target() -> TargetSpec {
    #[cfg(target_arch = "x86_64")]
    let arch = Arch::X86_64;
    #[cfg(target_arch = "aarch64")]
    let arch = Arch::Aarch64;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let arch = Arch::X86_64; // Fallback

    #[cfg(target_os = "linux")]
    let os = Os::Linux;
    #[cfg(target_os = "macos")]
    let os = Os::MacOs;
    #[cfg(target_os = "windows")]
    let os = Os::Windows;
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    let os = Os::Linux; // Fallback

    #[cfg(target_os = "macos")]
    let vendor = Vendor::Apple;
    #[cfg(not(target_os = "macos"))]
    let vendor = Vendor::Unknown;

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let abi = Abi::Gnu;
    #[cfg(all(target_os = "linux", target_env = "musl"))]
    let abi = Abi::Musl;
    #[cfg(target_os = "windows")]
    let abi = Abi::Msvc;
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    let abi = Abi::None;

    let features = CpuFeatures::default_for_arch(arch);
    let data_layout = generate_data_layout(arch, &features);

    TargetSpec {
        arch,
        vendor,
        os,
        abi,
        endianness: Endianness::Little,
        features,
        data_layout,
    }
}

/// Pre-defined target specifications.
pub mod targets {
    use super::*;

    /// x86_64 Linux GNU target.
    #[must_use]
    pub fn x86_64_linux_gnu() -> TargetSpec {
        parse_triple("x86_64-unknown-linux-gnu").unwrap()
    }

    /// x86_64 macOS target.
    #[must_use]
    pub fn x86_64_macos() -> TargetSpec {
        parse_triple("x86_64-apple-darwin").unwrap()
    }

    /// aarch64 Linux GNU target.
    #[must_use]
    pub fn aarch64_linux_gnu() -> TargetSpec {
        parse_triple("aarch64-unknown-linux-gnu").unwrap()
    }

    /// aarch64 macOS (Apple Silicon) target.
    #[must_use]
    pub fn aarch64_macos() -> TargetSpec {
        parse_triple("aarch64-apple-darwin").unwrap()
    }

    /// WebAssembly WASI target.
    #[must_use]
    pub fn wasm32_wasi() -> TargetSpec {
        parse_triple("wasm32-unknown-wasi").unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_x86_64_linux() {
        let target = parse_triple("x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(target.arch, Arch::X86_64);
        assert_eq!(target.os, Os::Linux);
        assert_eq!(target.abi, Abi::Gnu);
    }

    #[test]
    fn test_parse_aarch64_macos() {
        let target = parse_triple("aarch64-apple-darwin").unwrap();
        assert_eq!(target.arch, Arch::Aarch64);
        assert_eq!(target.os, Os::MacOs);
        assert_eq!(target.vendor, Vendor::Apple);
    }

    #[test]
    fn test_triple_roundtrip() {
        let original = "x86_64-unknown-linux-gnu";
        let target = parse_triple(original).unwrap();
        assert_eq!(target.triple(), original);
    }

    #[test]
    fn test_pointer_width() {
        assert_eq!(Arch::X86_64.pointer_width(), 64);
        assert_eq!(Arch::Aarch64.pointer_width(), 64);
        assert_eq!(Arch::Wasm32.pointer_width(), 32);
    }
}
