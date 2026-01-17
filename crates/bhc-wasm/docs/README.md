# bhc-wasm

WebAssembly code generation for the Basel Haskell Compiler.

## Overview

`bhc-wasm` generates WebAssembly code for browser and edge environments. Features:

- **WASM code generation**: `.wasm` binary and `.wat` text files
- **SIMD128 support**: Map SIMD operations to WASM SIMD128 instructions
- **Linear memory management**: Arena allocation for numeric kernels
- **Edge profile**: Minimal runtime for resource-constrained environments

## Architecture

```
                      ┌─────────────────────────┐
                      │    Loop IR Kernels      │
                      └───────────┬─────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
      ┌─────────────────┐                ┌─────────────────┐
      │  CPU Backend    │                │  WASM Backend   │
      │  (LLVM IR)      │                │  (WAT/WASM)     │
      └─────────────────┘                └────────┬────────┘
                                                  │
                            ┌─────────────────────┴─────────────────────┐
                            ▼                                           ▼
                  ┌─────────────────┐                        ┌─────────────────┐
                  │  WASI Runtime   │                        │ Browser Runtime │
                  │  (standalone)   │                        │   (optional)    │
                  └─────────────────┘                        └─────────────────┘
```

## Features

- `simd128`: Enable WASM SIMD128 instructions (default)
- `browser`: Enable browser runtime support

## Core Types

| Type | Description |
|------|-------------|
| `WasmBackend` | WASM code generation backend |
| `WasmConfig` | Backend configuration |
| `WasmModule` | Compiled WASM module |
| `WasmType` | WASM value types |
| `WasmInstr` | WASM instructions |

## WasmConfig

```rust
pub struct WasmConfig {
    /// Enable SIMD128 instructions
    pub simd_enabled: bool,
    /// Initial linear memory pages (64KB each)
    pub initial_memory_pages: u32,
    /// Maximum linear memory pages
    pub max_memory_pages: Option<u32>,
    /// Export memory for external access
    pub export_memory: bool,
    /// Generate debug names in output
    pub debug_names: bool,
    /// Optimize for size (Edge profile)
    pub optimize_size: bool,
}
```

### Profile Configurations

```rust
// Default configuration
let config = WasmConfig::default();
// simd: true, memory: 1MB initial, 16MB max

// Edge profile (minimal footprint)
let config = WasmConfig::edge_profile();
// simd: true, memory: 256KB initial, 4MB max, no debug names

// Browser profile (generous limits)
let config = WasmConfig::browser_profile();
// simd: true, memory: 1MB initial, 64MB max
```

## WASM Backend

```rust
use bhc_wasm::WasmBackend;
use bhc_codegen::{CodegenBackend, CodegenConfig};
use bhc_target::targets::wasm32_wasi;

// Create the WASM backend
let backend = WasmBackend::new();

// Create a codegen context
let config = CodegenConfig::for_target(wasm32_wasi());
let ctx = backend.create_context(config)?;

// Create and compile a module
let module = ctx.create_module("my_kernel")?;
module.write_to_file("output.wasm", CodegenOutputType::Object)?;
```

## WASM Types

```rust
pub enum WasmType {
    I32,       // 32-bit integer
    I64,       // 64-bit integer
    F32,       // 32-bit float
    F64,       // 64-bit float
    V128,      // 128-bit SIMD vector
    FuncRef,   // Function reference
    ExternRef, // External reference
}

// Get WAT representation
assert_eq!(WasmType::F32.wat_name(), "f32");
assert_eq!(WasmType::V128.wat_name(), "v128");

// Get size in bytes
assert_eq!(WasmType::I32.size_bytes(), 4);
assert_eq!(WasmType::V128.size_bytes(), 16);
```

## WASM Instructions

The crate provides a comprehensive instruction set:

### Control Flow

```rust
WasmInstr::Block(Some(WasmType::I32))  // block (result i32)
WasmInstr::Loop(None)                   // loop
WasmInstr::If(Some(WasmType::I32))     // if (result i32)
WasmInstr::Br(0)                        // br 0
WasmInstr::BrIf(1)                      // br_if 1
WasmInstr::Call(5)                      // call 5
WasmInstr::Return                       // return
```

### Variables

```rust
WasmInstr::LocalGet(0)   // local.get 0
WasmInstr::LocalSet(1)   // local.set 1
WasmInstr::GlobalGet(0)  // global.get 0
```

### Memory

```rust
WasmInstr::I32Load(4, 0)   // i32.load offset=0 align=4
WasmInstr::F32Store(4, 8)  // f32.store offset=8 align=4
WasmInstr::MemoryGrow      // memory.grow
```

### Arithmetic

```rust
WasmInstr::I32Const(42)    // i32.const 42
WasmInstr::I32Add          // i32.add
WasmInstr::F32Mul          // f32.mul
WasmInstr::F64Sqrt         // f64.sqrt
```

### SIMD128 Operations

```rust
// Vector load/store
WasmInstr::V128Load(16, 0)   // v128.load offset=0 align=16
WasmInstr::V128Store(16, 0)  // v128.store

// Splat (broadcast)
WasmInstr::F32x4Splat        // f32x4.splat

// Lane operations
WasmInstr::F32x4ExtractLane(0)  // f32x4.extract_lane 0
WasmInstr::I32x4ReplaceLane(2)  // i32x4.replace_lane 2

// Arithmetic
WasmInstr::F32x4Add    // f32x4.add
WasmInstr::F32x4Mul    // f32x4.mul
WasmInstr::F64x2Sqrt   // f64x2.sqrt

// Bitwise
WasmInstr::V128And     // v128.and
WasmInstr::V128Or      // v128.or
WasmInstr::V128Xor     // v128.xor
```

## WAT Output

Instructions can be converted to WAT text format:

```rust
let instr = WasmInstr::I32Const(42);
assert_eq!(instr.to_wat(), "i32.const 42");

let instr = WasmInstr::F32x4Add;
assert_eq!(instr.to_wat(), "f32x4.add");

let instr = WasmInstr::LocalGet(0);
assert_eq!(instr.to_wat(), "local.get 0");
```

## Error Handling

```rust
pub enum WasmError {
    /// Feature not supported
    NotSupported(String),
    /// Invalid module
    InvalidModule(String),
    /// Code generation error
    CodegenError(String),
    /// Memory error
    MemoryError(String),
    /// SIMD not available
    SimdNotAvailable(String),
    /// Internal error
    Internal(String),
}
```

## M8 Exit Criteria

- WebAssembly code generation produces valid modules
- SIMD128 used for vectorized numeric kernels
- Browser runtime loads and executes BHC code
- Edge profile produces minimal module size

## Submodules

| Module | Description |
|--------|-------------|
| `codegen` | WASM code generation |
| `runtime` | WASI/browser runtime support |

## See Also

- `bhc-loop-ir`: Input IR for code generation
- `bhc-codegen`: General code generation backend
- `bhc-target`: WASM target specifications
- WebAssembly Specification
- WASI Documentation
