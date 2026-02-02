//! # BHC WebAssembly Backend
//!
//! This crate provides WebAssembly code generation for BHC, enabling programs
//! to run in browsers and edge environments with the Edge profile's minimal
//! runtime footprint.
//!
//! ## Overview
//!
//! The WASM backend implements:
//!
//! - **WASM Code Generation**: Generate `.wasm` binary and `.wat` text files
//! - **SIMD128 Support**: Map SIMD operations to WASM SIMD128 instructions
//! - **Linear Memory Management**: Arena allocation for numeric kernels
//! - **Edge Profile**: Minimal runtime for resource-constrained environments
//!
//! ## Architecture
//!
//! ```text
//!                           ┌─────────────────────────────┐
//!                           │       Loop IR Kernels       │
//!                           └─────────────┬───────────────┘
//!                                         │
//!                     ┌───────────────────┴───────────────────┐
//!                     ▼                                       ▼
//!           ┌─────────────────┐                    ┌─────────────────┐
//!           │   CPU Backend   │                    │   WASM Backend  │
//!           │   (LLVM IR)     │                    │   (WAT/WASM)    │
//!           └─────────────────┘                    └────────┬────────┘
//!                                                           │
//!                                     ┌─────────────────────┴─────────────────────┐
//!                                     ▼                                           ▼
//!                           ┌─────────────────┐                        ┌─────────────────┐
//!                           │   WASI Runtime  │                        │ Browser Runtime │
//!                           │   (standalone)  │                        │   (optional)    │
//!                           └─────────────────┘                        └─────────────────┘
//! ```
//!
//! ## Features
//!
//! - `simd128`: Enable WASM SIMD128 instructions (default)
//! - `browser`: Enable browser runtime support
//!
//! ## Usage
//!
//! ```rust,ignore
//! use bhc_wasm::WasmBackend;
//! use bhc_codegen::{CodegenBackend, CodegenConfig};
//! use bhc_target::targets::wasm32_wasi;
//!
//! // Create the WASM backend
//! let backend = WasmBackend::new();
//!
//! // Create a codegen context
//! let config = CodegenConfig::for_target(wasm32_wasi());
//! let ctx = backend.create_context(config)?;
//!
//! // Create and compile a module
//! let module = ctx.create_module("my_kernel")?;
//! module.write_to_file("output.wasm", CodegenOutputType::Object)?;
//! ```
//!
//! ## M8 Exit Criteria (from ROADMAP.md)
//!
//! - WebAssembly code generation produces valid modules
//! - SIMD128 used for vectorized numeric kernels
//! - Browser runtime loads and executes BHC code
//! - Edge profile produces minimal module size

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod codegen;
pub mod core_lower;
pub mod lower;
pub mod runtime;
pub mod wasi;

use bhc_codegen::{CodegenBackend, CodegenConfig, CodegenError, CodegenResult};
use bhc_target::{Arch, TargetSpec};
use thiserror::Error;

pub use codegen::{RuntimeIndices, WasmCodegenContext, WasmModule};

/// Errors that can occur during WASM operations.
#[derive(Debug, Error)]
pub enum WasmError {
    /// Feature not supported by the target.
    #[error("feature not supported: {0}")]
    NotSupported(String),

    /// Invalid WASM module.
    #[error("invalid WASM module: {0}")]
    InvalidModule(String),

    /// Code generation error.
    #[error("codegen error: {0}")]
    CodegenError(String),

    /// Memory allocation error.
    #[error("memory error: {0}")]
    MemoryError(String),

    /// SIMD operation not available.
    #[error("SIMD operation not available: {0}")]
    SimdNotAvailable(String),

    /// Internal error.
    #[error("internal WASM error: {0}")]
    Internal(String),
}

/// Result type for WASM operations.
pub type WasmResult<T> = Result<T, WasmError>;

/// Configuration for the WASM backend.
#[derive(Clone, Debug)]
pub struct WasmConfig {
    /// Enable SIMD128 instructions.
    pub simd_enabled: bool,
    /// Initial linear memory pages (64KB each).
    pub initial_memory_pages: u32,
    /// Maximum linear memory pages.
    pub max_memory_pages: Option<u32>,
    /// Export table for external access.
    pub export_memory: bool,
    /// Generate debug names in output.
    pub debug_names: bool,
    /// Optimize for size (Edge profile).
    pub optimize_size: bool,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            simd_enabled: cfg!(feature = "simd128"),
            initial_memory_pages: 16,    // 1MB initial
            max_memory_pages: Some(256), // 16MB max
            export_memory: true,
            debug_names: true,
            optimize_size: false,
        }
    }
}

impl WasmConfig {
    /// Create a config for the Edge profile (minimal footprint).
    #[must_use]
    pub fn edge_profile() -> Self {
        Self {
            simd_enabled: true,
            initial_memory_pages: 4,    // 256KB initial
            max_memory_pages: Some(64), // 4MB max
            export_memory: true,
            debug_names: false,
            optimize_size: true,
        }
    }

    /// Create a config for browser environments.
    #[must_use]
    pub fn browser_profile() -> Self {
        Self {
            simd_enabled: true,
            initial_memory_pages: 16,
            max_memory_pages: Some(1024), // 64MB max
            export_memory: true,
            debug_names: true,
            optimize_size: false,
        }
    }

    /// Create a config based on the compilation profile.
    #[must_use]
    pub fn for_profile(profile: bhc_session::Profile) -> Self {
        match profile {
            bhc_session::Profile::Edge => Self::edge_profile(),
            bhc_session::Profile::Numeric => Self {
                simd_enabled: true,
                initial_memory_pages: 32, // 2MB initial for numeric workloads
                max_memory_pages: Some(512), // 32MB max
                export_memory: true,
                debug_names: true,
                optimize_size: false,
            },
            _ => Self::default(),
        }
    }
}

/// The WebAssembly code generation backend.
///
/// Implements `CodegenBackend` for WASM targets, producing `.wasm` binary
/// or `.wat` text files from Loop IR.
pub struct WasmBackend {
    /// Backend configuration.
    config: WasmConfig,
}

impl WasmBackend {
    /// Create a new WASM backend with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: WasmConfig::default(),
        }
    }

    /// Create a WASM backend with custom configuration.
    #[must_use]
    pub fn with_config(config: WasmConfig) -> Self {
        Self { config }
    }

    /// Get the backend configuration.
    #[must_use]
    pub fn config(&self) -> &WasmConfig {
        &self.config
    }

    /// Check if SIMD128 is enabled.
    #[must_use]
    pub fn simd_enabled(&self) -> bool {
        self.config.simd_enabled
    }
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CodegenBackend for WasmBackend {
    type Context = WasmCodegenContext;

    fn name(&self) -> &'static str {
        "wasm"
    }

    fn supports_target(&self, target: &TargetSpec) -> bool {
        matches!(target.arch, Arch::Wasm32 | Arch::Wasm64)
    }

    fn create_context(&self, config: CodegenConfig) -> CodegenResult<Self::Context> {
        if !self.supports_target(&config.target) {
            return Err(CodegenError::UnsupportedTarget(format!(
                "WASM backend does not support {}",
                config.target.triple()
            )));
        }

        Ok(WasmCodegenContext::new(config, self.config.clone()))
    }
}

/// WASM value types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WasmType {
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 128-bit SIMD vector.
    V128,
    /// Function reference.
    FuncRef,
    /// External reference.
    ExternRef,
}

impl WasmType {
    /// Get the WAT text representation.
    #[must_use]
    pub const fn wat_name(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::V128 => "v128",
            Self::FuncRef => "funcref",
            Self::ExternRef => "externref",
        }
    }

    /// Get the size in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
            Self::V128 => 16,
            Self::FuncRef | Self::ExternRef => 4, // Reference size
        }
    }
}

/// WASM instruction opcodes.
#[derive(Clone, Debug, PartialEq)]
pub enum WasmInstr {
    // Control instructions
    /// Unreachable trap.
    Unreachable,
    /// No operation.
    Nop,
    /// Block start.
    Block(Option<WasmType>),
    /// Loop start.
    Loop(Option<WasmType>),
    /// If block.
    If(Option<WasmType>),
    /// Else branch.
    Else,
    /// End of block.
    End,
    /// Branch to label.
    Br(u32),
    /// Conditional branch.
    BrIf(u32),
    /// Table branch.
    BrTable(Vec<u32>, u32),
    /// Return from function.
    Return,
    /// Call function.
    Call(u32),
    /// Indirect call.
    CallIndirect(u32, u32),

    // Parametric instructions
    /// Drop value.
    Drop,
    /// Select based on condition.
    Select,

    // Variable instructions
    /// Get local variable.
    LocalGet(u32),
    /// Set local variable.
    LocalSet(u32),
    /// Tee local variable.
    LocalTee(u32),
    /// Get global variable.
    GlobalGet(u32),
    /// Set global variable.
    GlobalSet(u32),

    // Memory instructions
    /// Load i32 from memory.
    I32Load(u32, u32),
    /// Load i64 from memory.
    I64Load(u32, u32),
    /// Load f32 from memory.
    F32Load(u32, u32),
    /// Load f64 from memory.
    F64Load(u32, u32),
    /// Load 8-bit unsigned to i32.
    I32Load8U(u32, u32),
    /// Load 8-bit signed to i32.
    I32Load8S(u32, u32),
    /// Load 16-bit unsigned to i32.
    I32Load16U(u32, u32),
    /// Load 16-bit signed to i32.
    I32Load16S(u32, u32),
    /// Store i32 to memory.
    I32Store(u32, u32),
    /// Store i64 to memory.
    I64Store(u32, u32),
    /// Store f32 to memory.
    F32Store(u32, u32),
    /// Store f64 to memory.
    F64Store(u32, u32),
    /// Store low 8 bits of i32.
    I32Store8(u32, u32),
    /// Store low 16 bits of i32.
    I32Store16(u32, u32),
    /// Memory size.
    MemorySize,
    /// Memory grow.
    MemoryGrow,

    // Numeric instructions - i32
    /// i32 constant.
    I32Const(i32),
    /// i32 equals zero.
    I32Eqz,
    /// i32 equals.
    I32Eq,
    /// i32 not equals.
    I32Ne,
    /// i32 signed less than.
    I32LtS,
    /// i32 unsigned less than.
    I32LtU,
    /// i32 signed greater than.
    I32GtS,
    /// i32 unsigned greater than.
    I32GtU,
    /// i32 signed less or equal.
    I32LeS,
    /// i32 unsigned less or equal.
    I32LeU,
    /// i32 signed greater or equal.
    I32GeS,
    /// i32 unsigned greater or equal.
    I32GeU,
    /// i32 add.
    I32Add,
    /// i32 subtract.
    I32Sub,
    /// i32 multiply.
    I32Mul,
    /// i32 signed divide.
    I32DivS,
    /// i32 unsigned divide.
    I32DivU,
    /// i32 signed remainder.
    I32RemS,
    /// i32 unsigned remainder.
    I32RemU,
    /// i32 bitwise and.
    I32And,
    /// i32 bitwise or.
    I32Or,
    /// i32 bitwise xor.
    I32Xor,
    /// i32 shift left.
    I32Shl,
    /// i32 signed shift right.
    I32ShrS,
    /// i32 unsigned shift right.
    I32ShrU,

    // Numeric instructions - i64
    /// i64 constant.
    I64Const(i64),
    /// i64 equals zero.
    I64Eqz,
    /// i64 equals.
    I64Eq,
    /// i64 not equals.
    I64Ne,
    /// i64 signed less than.
    I64LtS,
    /// i64 unsigned less than.
    I64LtU,
    /// i64 signed greater than.
    I64GtS,
    /// i64 unsigned greater than.
    I64GtU,
    /// i64 signed less or equal.
    I64LeS,
    /// i64 unsigned less or equal.
    I64LeU,
    /// i64 signed greater or equal.
    I64GeS,
    /// i64 unsigned greater or equal.
    I64GeU,
    /// i64 add.
    I64Add,
    /// i64 subtract.
    I64Sub,
    /// i64 multiply.
    I64Mul,
    /// i64 signed divide.
    I64DivS,
    /// i64 unsigned divide.
    I64DivU,
    /// i64 signed remainder.
    I64RemS,
    /// i64 unsigned remainder.
    I64RemU,
    /// i64 bitwise and.
    I64And,
    /// i64 bitwise or.
    I64Or,
    /// i64 bitwise xor.
    I64Xor,
    /// i64 shift left.
    I64Shl,
    /// i64 signed shift right.
    I64ShrS,
    /// i64 unsigned shift right.
    I64ShrU,

    // Numeric instructions - f32
    /// f32 constant.
    F32Const(f32),
    /// f32 equals.
    F32Eq,
    /// f32 not equals.
    F32Ne,
    /// f32 less than.
    F32Lt,
    /// f32 greater than.
    F32Gt,
    /// f32 less or equal.
    F32Le,
    /// f32 greater or equal.
    F32Ge,
    /// f32 absolute value.
    F32Abs,
    /// f32 negate.
    F32Neg,
    /// f32 ceiling.
    F32Ceil,
    /// f32 floor.
    F32Floor,
    /// f32 truncate.
    F32Trunc,
    /// f32 nearest.
    F32Nearest,
    /// f32 square root.
    F32Sqrt,
    /// f32 add.
    F32Add,
    /// f32 subtract.
    F32Sub,
    /// f32 multiply.
    F32Mul,
    /// f32 divide.
    F32Div,
    /// f32 minimum.
    F32Min,
    /// f32 maximum.
    F32Max,
    /// f32 copysign.
    F32Copysign,

    // Numeric instructions - f64
    /// f64 constant.
    F64Const(f64),
    /// f64 equals.
    F64Eq,
    /// f64 not equals.
    F64Ne,
    /// f64 less than.
    F64Lt,
    /// f64 greater than.
    F64Gt,
    /// f64 less or equal.
    F64Le,
    /// f64 greater or equal.
    F64Ge,
    /// f64 absolute value.
    F64Abs,
    /// f64 negate.
    F64Neg,
    /// f64 ceiling.
    F64Ceil,
    /// f64 floor.
    F64Floor,
    /// f64 truncate.
    F64Trunc,
    /// f64 nearest.
    F64Nearest,
    /// f64 square root.
    F64Sqrt,
    /// f64 add.
    F64Add,
    /// f64 subtract.
    F64Sub,
    /// f64 multiply.
    F64Mul,
    /// f64 divide.
    F64Div,
    /// f64 minimum.
    F64Min,
    /// f64 maximum.
    F64Max,
    /// f64 copysign.
    F64Copysign,

    // Conversion instructions
    /// Wrap i64 to i32.
    I32WrapI64,
    /// Truncate f32 to signed i32.
    I32TruncF32S,
    /// Truncate f32 to unsigned i32.
    I32TruncF32U,
    /// Truncate f64 to signed i32.
    I32TruncF64S,
    /// Truncate f64 to unsigned i32.
    I32TruncF64U,
    /// Extend i32 to i64 signed.
    I64ExtendI32S,
    /// Extend i32 to i64 unsigned.
    I64ExtendI32U,
    /// Truncate f32 to signed i64.
    I64TruncF32S,
    /// Truncate f32 to unsigned i64.
    I64TruncF32U,
    /// Truncate f64 to signed i64.
    I64TruncF64S,
    /// Truncate f64 to unsigned i64.
    I64TruncF64U,
    /// Convert signed i32 to f32.
    F32ConvertI32S,
    /// Convert unsigned i32 to f32.
    F32ConvertI32U,
    /// Convert signed i64 to f32.
    F32ConvertI64S,
    /// Convert unsigned i64 to f32.
    F32ConvertI64U,
    /// Demote f64 to f32.
    F32DemoteF64,
    /// Convert signed i32 to f64.
    F64ConvertI32S,
    /// Convert unsigned i32 to f64.
    F64ConvertI32U,
    /// Convert signed i64 to f64.
    F64ConvertI64S,
    /// Convert unsigned i64 to f64.
    F64ConvertI64U,
    /// Promote f32 to f64.
    F64PromoteF32,
    /// Reinterpret f32 bits as i32.
    I32ReinterpretF32,
    /// Reinterpret f64 bits as i64.
    I64ReinterpretF64,
    /// Reinterpret i32 bits as f32.
    F32ReinterpretI32,
    /// Reinterpret i64 bits as f64.
    F64ReinterpretI64,

    // SIMD instructions (v128)
    /// Load v128 from memory.
    V128Load(u32, u32),
    /// Store v128 to memory.
    V128Store(u32, u32),
    /// v128 constant.
    V128Const([u8; 16]),
    /// i8x16 shuffle.
    I8x16Shuffle([u8; 16]),
    /// i8x16 splat (broadcast).
    I8x16Splat,
    /// i16x8 splat.
    I16x8Splat,
    /// i32x4 splat.
    I32x4Splat,
    /// i64x2 splat.
    I64x2Splat,
    /// f32x4 splat.
    F32x4Splat,
    /// f64x2 splat.
    F64x2Splat,
    /// i8x16 extract lane signed.
    I8x16ExtractLaneS(u8),
    /// i8x16 extract lane unsigned.
    I8x16ExtractLaneU(u8),
    /// i16x8 extract lane signed.
    I16x8ExtractLaneS(u8),
    /// i16x8 extract lane unsigned.
    I16x8ExtractLaneU(u8),
    /// i32x4 extract lane.
    I32x4ExtractLane(u8),
    /// i64x2 extract lane.
    I64x2ExtractLane(u8),
    /// f32x4 extract lane.
    F32x4ExtractLane(u8),
    /// f64x2 extract lane.
    F64x2ExtractLane(u8),
    /// i32x4 replace lane.
    I32x4ReplaceLane(u8),
    /// f32x4 replace lane.
    F32x4ReplaceLane(u8),
    /// f64x2 replace lane.
    F64x2ReplaceLane(u8),

    // SIMD arithmetic - f32x4
    /// f32x4 add.
    F32x4Add,
    /// f32x4 subtract.
    F32x4Sub,
    /// f32x4 multiply.
    F32x4Mul,
    /// f32x4 divide.
    F32x4Div,
    /// f32x4 minimum.
    F32x4Min,
    /// f32x4 maximum.
    F32x4Max,
    /// f32x4 absolute value.
    F32x4Abs,
    /// f32x4 negate.
    F32x4Neg,
    /// f32x4 square root.
    F32x4Sqrt,
    /// f32x4 ceiling.
    F32x4Ceil,
    /// f32x4 floor.
    F32x4Floor,

    // SIMD arithmetic - f64x2
    /// f64x2 add.
    F64x2Add,
    /// f64x2 subtract.
    F64x2Sub,
    /// f64x2 multiply.
    F64x2Mul,
    /// f64x2 divide.
    F64x2Div,
    /// f64x2 minimum.
    F64x2Min,
    /// f64x2 maximum.
    F64x2Max,
    /// f64x2 absolute value.
    F64x2Abs,
    /// f64x2 negate.
    F64x2Neg,
    /// f64x2 square root.
    F64x2Sqrt,

    // SIMD arithmetic - i32x4
    /// i32x4 add.
    I32x4Add,
    /// i32x4 subtract.
    I32x4Sub,
    /// i32x4 multiply.
    I32x4Mul,
    /// i32x4 negate.
    I32x4Neg,
    /// i32x4 shift left.
    I32x4Shl,
    /// i32x4 shift right signed.
    I32x4ShrS,
    /// i32x4 shift right unsigned.
    I32x4ShrU,

    // SIMD bitwise
    /// v128 bitwise and.
    V128And,
    /// v128 bitwise or.
    V128Or,
    /// v128 bitwise xor.
    V128Xor,
    /// v128 bitwise not.
    V128Not,
    /// v128 bitwise and-not.
    V128AndNot,
    /// v128 any lane true.
    V128AnyTrue,

    /// Comment (for debugging/output).
    Comment(String),
}

impl WasmInstr {
    /// Get the WAT text representation of this instruction.
    pub fn to_wat(&self) -> String {
        match self {
            // Control
            Self::Unreachable => "unreachable".to_string(),
            Self::Nop => "nop".to_string(),
            Self::Block(ty) => match ty {
                Some(t) => format!("block (result {})", t.wat_name()),
                None => "block".to_string(),
            },
            Self::Loop(ty) => match ty {
                Some(t) => format!("loop (result {})", t.wat_name()),
                None => "loop".to_string(),
            },
            Self::If(ty) => match ty {
                Some(t) => format!("if (result {})", t.wat_name()),
                None => "if".to_string(),
            },
            Self::Else => "else".to_string(),
            Self::End => "end".to_string(),
            Self::Br(l) => format!("br {l}"),
            Self::BrIf(l) => format!("br_if {l}"),
            Self::BrTable(labels, default) => {
                let labels_str: Vec<String> = labels.iter().map(|l| l.to_string()).collect();
                format!("br_table {} {}", labels_str.join(" "), default)
            }
            Self::Return => "return".to_string(),
            Self::Call(idx) => format!("call {idx}"),
            Self::CallIndirect(ty, table) => format!("call_indirect (type {ty}) {table}"),

            // Parametric
            Self::Drop => "drop".to_string(),
            Self::Select => "select".to_string(),

            // Variables
            Self::LocalGet(idx) => format!("local.get {idx}"),
            Self::LocalSet(idx) => format!("local.set {idx}"),
            Self::LocalTee(idx) => format!("local.tee {idx}"),
            Self::GlobalGet(idx) => format!("global.get {idx}"),
            Self::GlobalSet(idx) => format!("global.set {idx}"),

            // Memory
            Self::I32Load(align, offset) => format!("i32.load offset={offset} align={align}"),
            Self::I64Load(align, offset) => format!("i64.load offset={offset} align={align}"),
            Self::F32Load(align, offset) => format!("f32.load offset={offset} align={align}"),
            Self::F64Load(align, offset) => format!("f64.load offset={offset} align={align}"),
            Self::I32Load8U(align, offset) => format!("i32.load8_u offset={offset} align={align}"),
            Self::I32Load8S(align, offset) => format!("i32.load8_s offset={offset} align={align}"),
            Self::I32Load16U(align, offset) => {
                format!("i32.load16_u offset={offset} align={align}")
            }
            Self::I32Load16S(align, offset) => {
                format!("i32.load16_s offset={offset} align={align}")
            }
            Self::I32Store(align, offset) => format!("i32.store offset={offset} align={align}"),
            Self::I64Store(align, offset) => format!("i64.store offset={offset} align={align}"),
            Self::F32Store(align, offset) => format!("f32.store offset={offset} align={align}"),
            Self::F64Store(align, offset) => format!("f64.store offset={offset} align={align}"),
            Self::I32Store8(align, offset) => format!("i32.store8 offset={offset} align={align}"),
            Self::I32Store16(align, offset) => format!("i32.store16 offset={offset} align={align}"),
            Self::MemorySize => "memory.size".to_string(),
            Self::MemoryGrow => "memory.grow".to_string(),

            // i32 numeric
            Self::I32Const(v) => format!("i32.const {v}"),
            Self::I32Eqz => "i32.eqz".to_string(),
            Self::I32Eq => "i32.eq".to_string(),
            Self::I32Ne => "i32.ne".to_string(),
            Self::I32LtS => "i32.lt_s".to_string(),
            Self::I32LtU => "i32.lt_u".to_string(),
            Self::I32GtS => "i32.gt_s".to_string(),
            Self::I32GtU => "i32.gt_u".to_string(),
            Self::I32LeS => "i32.le_s".to_string(),
            Self::I32LeU => "i32.le_u".to_string(),
            Self::I32GeS => "i32.ge_s".to_string(),
            Self::I32GeU => "i32.ge_u".to_string(),
            Self::I32Add => "i32.add".to_string(),
            Self::I32Sub => "i32.sub".to_string(),
            Self::I32Mul => "i32.mul".to_string(),
            Self::I32DivS => "i32.div_s".to_string(),
            Self::I32DivU => "i32.div_u".to_string(),
            Self::I32RemS => "i32.rem_s".to_string(),
            Self::I32RemU => "i32.rem_u".to_string(),
            Self::I32And => "i32.and".to_string(),
            Self::I32Or => "i32.or".to_string(),
            Self::I32Xor => "i32.xor".to_string(),
            Self::I32Shl => "i32.shl".to_string(),
            Self::I32ShrS => "i32.shr_s".to_string(),
            Self::I32ShrU => "i32.shr_u".to_string(),

            // i64 numeric
            Self::I64Const(v) => format!("i64.const {v}"),
            Self::I64Eqz => "i64.eqz".to_string(),
            Self::I64Eq => "i64.eq".to_string(),
            Self::I64Ne => "i64.ne".to_string(),
            Self::I64LtS => "i64.lt_s".to_string(),
            Self::I64LtU => "i64.lt_u".to_string(),
            Self::I64GtS => "i64.gt_s".to_string(),
            Self::I64GtU => "i64.gt_u".to_string(),
            Self::I64LeS => "i64.le_s".to_string(),
            Self::I64LeU => "i64.le_u".to_string(),
            Self::I64GeS => "i64.ge_s".to_string(),
            Self::I64GeU => "i64.ge_u".to_string(),
            Self::I64Add => "i64.add".to_string(),
            Self::I64Sub => "i64.sub".to_string(),
            Self::I64Mul => "i64.mul".to_string(),
            Self::I64DivS => "i64.div_s".to_string(),
            Self::I64DivU => "i64.div_u".to_string(),
            Self::I64RemS => "i64.rem_s".to_string(),
            Self::I64RemU => "i64.rem_u".to_string(),
            Self::I64And => "i64.and".to_string(),
            Self::I64Or => "i64.or".to_string(),
            Self::I64Xor => "i64.xor".to_string(),
            Self::I64Shl => "i64.shl".to_string(),
            Self::I64ShrS => "i64.shr_s".to_string(),
            Self::I64ShrU => "i64.shr_u".to_string(),

            // f32 numeric
            Self::F32Const(v) => format!("f32.const {v}"),
            Self::F32Eq => "f32.eq".to_string(),
            Self::F32Ne => "f32.ne".to_string(),
            Self::F32Lt => "f32.lt".to_string(),
            Self::F32Gt => "f32.gt".to_string(),
            Self::F32Le => "f32.le".to_string(),
            Self::F32Ge => "f32.ge".to_string(),
            Self::F32Abs => "f32.abs".to_string(),
            Self::F32Neg => "f32.neg".to_string(),
            Self::F32Ceil => "f32.ceil".to_string(),
            Self::F32Floor => "f32.floor".to_string(),
            Self::F32Trunc => "f32.trunc".to_string(),
            Self::F32Nearest => "f32.nearest".to_string(),
            Self::F32Sqrt => "f32.sqrt".to_string(),
            Self::F32Add => "f32.add".to_string(),
            Self::F32Sub => "f32.sub".to_string(),
            Self::F32Mul => "f32.mul".to_string(),
            Self::F32Div => "f32.div".to_string(),
            Self::F32Min => "f32.min".to_string(),
            Self::F32Max => "f32.max".to_string(),
            Self::F32Copysign => "f32.copysign".to_string(),

            // f64 numeric
            Self::F64Const(v) => format!("f64.const {v}"),
            Self::F64Eq => "f64.eq".to_string(),
            Self::F64Ne => "f64.ne".to_string(),
            Self::F64Lt => "f64.lt".to_string(),
            Self::F64Gt => "f64.gt".to_string(),
            Self::F64Le => "f64.le".to_string(),
            Self::F64Ge => "f64.ge".to_string(),
            Self::F64Abs => "f64.abs".to_string(),
            Self::F64Neg => "f64.neg".to_string(),
            Self::F64Ceil => "f64.ceil".to_string(),
            Self::F64Floor => "f64.floor".to_string(),
            Self::F64Trunc => "f64.trunc".to_string(),
            Self::F64Nearest => "f64.nearest".to_string(),
            Self::F64Sqrt => "f64.sqrt".to_string(),
            Self::F64Add => "f64.add".to_string(),
            Self::F64Sub => "f64.sub".to_string(),
            Self::F64Mul => "f64.mul".to_string(),
            Self::F64Div => "f64.div".to_string(),
            Self::F64Min => "f64.min".to_string(),
            Self::F64Max => "f64.max".to_string(),
            Self::F64Copysign => "f64.copysign".to_string(),

            // Conversions
            Self::I32WrapI64 => "i32.wrap_i64".to_string(),
            Self::I32TruncF32S => "i32.trunc_f32_s".to_string(),
            Self::I32TruncF32U => "i32.trunc_f32_u".to_string(),
            Self::I32TruncF64S => "i32.trunc_f64_s".to_string(),
            Self::I32TruncF64U => "i32.trunc_f64_u".to_string(),
            Self::I64ExtendI32S => "i64.extend_i32_s".to_string(),
            Self::I64ExtendI32U => "i64.extend_i32_u".to_string(),
            Self::I64TruncF32S => "i64.trunc_f32_s".to_string(),
            Self::I64TruncF32U => "i64.trunc_f32_u".to_string(),
            Self::I64TruncF64S => "i64.trunc_f64_s".to_string(),
            Self::I64TruncF64U => "i64.trunc_f64_u".to_string(),
            Self::F32ConvertI32S => "f32.convert_i32_s".to_string(),
            Self::F32ConvertI32U => "f32.convert_i32_u".to_string(),
            Self::F32ConvertI64S => "f32.convert_i64_s".to_string(),
            Self::F32ConvertI64U => "f32.convert_i64_u".to_string(),
            Self::F32DemoteF64 => "f32.demote_f64".to_string(),
            Self::F64ConvertI32S => "f64.convert_i32_s".to_string(),
            Self::F64ConvertI32U => "f64.convert_i32_u".to_string(),
            Self::F64ConvertI64S => "f64.convert_i64_s".to_string(),
            Self::F64ConvertI64U => "f64.convert_i64_u".to_string(),
            Self::F64PromoteF32 => "f64.promote_f32".to_string(),
            Self::I32ReinterpretF32 => "i32.reinterpret_f32".to_string(),
            Self::I64ReinterpretF64 => "i64.reinterpret_f64".to_string(),
            Self::F32ReinterpretI32 => "f32.reinterpret_i32".to_string(),
            Self::F64ReinterpretI64 => "f64.reinterpret_i64".to_string(),

            // SIMD
            Self::V128Load(align, offset) => format!("v128.load offset={offset} align={align}"),
            Self::V128Store(align, offset) => format!("v128.store offset={offset} align={align}"),
            Self::V128Const(bytes) => {
                format!(
                    "v128.const i8x16 {}",
                    bytes
                        .iter()
                        .map(|b| b.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            }
            Self::I8x16Shuffle(lanes) => {
                format!(
                    "i8x16.shuffle {}",
                    lanes
                        .iter()
                        .map(|l| l.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            }
            Self::I8x16Splat => "i8x16.splat".to_string(),
            Self::I16x8Splat => "i16x8.splat".to_string(),
            Self::I32x4Splat => "i32x4.splat".to_string(),
            Self::I64x2Splat => "i64x2.splat".to_string(),
            Self::F32x4Splat => "f32x4.splat".to_string(),
            Self::F64x2Splat => "f64x2.splat".to_string(),
            Self::I8x16ExtractLaneS(lane) => format!("i8x16.extract_lane_s {lane}"),
            Self::I8x16ExtractLaneU(lane) => format!("i8x16.extract_lane_u {lane}"),
            Self::I16x8ExtractLaneS(lane) => format!("i16x8.extract_lane_s {lane}"),
            Self::I16x8ExtractLaneU(lane) => format!("i16x8.extract_lane_u {lane}"),
            Self::I32x4ExtractLane(lane) => format!("i32x4.extract_lane {lane}"),
            Self::I64x2ExtractLane(lane) => format!("i64x2.extract_lane {lane}"),
            Self::F32x4ExtractLane(lane) => format!("f32x4.extract_lane {lane}"),
            Self::F64x2ExtractLane(lane) => format!("f64x2.extract_lane {lane}"),
            Self::I32x4ReplaceLane(lane) => format!("i32x4.replace_lane {lane}"),
            Self::F32x4ReplaceLane(lane) => format!("f32x4.replace_lane {lane}"),
            Self::F64x2ReplaceLane(lane) => format!("f64x2.replace_lane {lane}"),

            // SIMD f32x4
            Self::F32x4Add => "f32x4.add".to_string(),
            Self::F32x4Sub => "f32x4.sub".to_string(),
            Self::F32x4Mul => "f32x4.mul".to_string(),
            Self::F32x4Div => "f32x4.div".to_string(),
            Self::F32x4Min => "f32x4.min".to_string(),
            Self::F32x4Max => "f32x4.max".to_string(),
            Self::F32x4Abs => "f32x4.abs".to_string(),
            Self::F32x4Neg => "f32x4.neg".to_string(),
            Self::F32x4Sqrt => "f32x4.sqrt".to_string(),
            Self::F32x4Ceil => "f32x4.ceil".to_string(),
            Self::F32x4Floor => "f32x4.floor".to_string(),

            // SIMD f64x2
            Self::F64x2Add => "f64x2.add".to_string(),
            Self::F64x2Sub => "f64x2.sub".to_string(),
            Self::F64x2Mul => "f64x2.mul".to_string(),
            Self::F64x2Div => "f64x2.div".to_string(),
            Self::F64x2Min => "f64x2.min".to_string(),
            Self::F64x2Max => "f64x2.max".to_string(),
            Self::F64x2Abs => "f64x2.abs".to_string(),
            Self::F64x2Neg => "f64x2.neg".to_string(),
            Self::F64x2Sqrt => "f64x2.sqrt".to_string(),

            // SIMD i32x4
            Self::I32x4Add => "i32x4.add".to_string(),
            Self::I32x4Sub => "i32x4.sub".to_string(),
            Self::I32x4Mul => "i32x4.mul".to_string(),
            Self::I32x4Neg => "i32x4.neg".to_string(),
            Self::I32x4Shl => "i32x4.shl".to_string(),
            Self::I32x4ShrS => "i32x4.shr_s".to_string(),
            Self::I32x4ShrU => "i32x4.shr_u".to_string(),

            // SIMD bitwise
            Self::V128And => "v128.and".to_string(),
            Self::V128Or => "v128.or".to_string(),
            Self::V128Xor => "v128.xor".to_string(),
            Self::V128Not => "v128.not".to_string(),
            Self::V128AndNot => "v128.andnot".to_string(),
            Self::V128AnyTrue => "v128.any_true".to_string(),

            Self::Comment(text) => format!(";; {text}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_backend_creation() {
        let backend = WasmBackend::new();
        assert_eq!(backend.name(), "wasm");
    }

    #[test]
    fn test_wasm_backend_supports_target() {
        let backend = WasmBackend::new();
        let wasm32 = bhc_target::targets::wasm32_wasi();
        assert!(backend.supports_target(&wasm32));

        let x86 = bhc_target::host_target();
        // Only true if we're on wasm (which we're not in tests)
        if !matches!(x86.arch, Arch::Wasm32 | Arch::Wasm64) {
            assert!(!backend.supports_target(&x86));
        }
    }

    #[test]
    fn test_wasm_config_default() {
        let config = WasmConfig::default();
        assert!(config.simd_enabled);
        assert_eq!(config.initial_memory_pages, 16);
        assert!(config.export_memory);
    }

    #[test]
    fn test_wasm_config_edge_profile() {
        let config = WasmConfig::edge_profile();
        assert!(config.optimize_size);
        assert!(!config.debug_names);
        assert_eq!(config.initial_memory_pages, 4);
    }

    #[test]
    fn test_wasm_type_wat_name() {
        assert_eq!(WasmType::I32.wat_name(), "i32");
        assert_eq!(WasmType::F64.wat_name(), "f64");
        assert_eq!(WasmType::V128.wat_name(), "v128");
    }

    #[test]
    fn test_wasm_instr_to_wat() {
        assert_eq!(WasmInstr::I32Const(42).to_wat(), "i32.const 42");
        assert_eq!(WasmInstr::F32Add.to_wat(), "f32.add");
        assert_eq!(WasmInstr::LocalGet(0).to_wat(), "local.get 0");
        assert_eq!(WasmInstr::F32x4Add.to_wat(), "f32x4.add");
    }
}
