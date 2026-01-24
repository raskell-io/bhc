# bhc-wasm

WebAssembly code generation for the Basel Haskell Compiler.

## Overview

This crate provides WebAssembly code generation for BHC, enabling programs to run in browsers and edge environments with the Edge profile's minimal runtime footprint.

## Architecture

```
                          ┌─────────────────────────────┐
                          │       Loop IR Kernels       │
                          └─────────────┬───────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
          ┌─────────────────┐                    ┌─────────────────┐
          │   CPU Backend   │                    │   WASM Backend  │
          │   (LLVM IR)     │                    │   (WAT/WASM)    │
          └─────────────────┘                    └────────┬────────┘
                                                          │
                                    ┌─────────────────────┴─────────────────────┐
                                    ▼                                           ▼
                          ┌─────────────────┐                        ┌─────────────────┐
                          │   WASI Runtime  │                        │ Browser Runtime │
                          │   (standalone)  │                        │   (optional)    │
                          └─────────────────┘                        └─────────────────┘
```

## Features

- `simd128` - Enable WASM SIMD128 instructions (default)
- `browser` - Enable browser runtime support

## Key Types

| Type | Description |
|------|-------------|
| `WasmBackend` | WebAssembly code generation backend |
| `WasmModule` | Compiled WASM module |
| `WasmRuntime` | Runtime for executing WASM |
| `WasiContext` | WASI environment context |

## Usage

### Code Generation

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

### WASI Execution

```rust
use bhc_wasm::{WasiContext, WasmRuntime};

// Create WASI context
let wasi = WasiContext::new()
    .args(&["program", "arg1"])
    .env("KEY", "value")
    .preopens(&[(".", "/")])?;

// Load and execute module
let runtime = WasmRuntime::new()?;
let module = runtime.load_file("program.wasm")?;
let exit_code = runtime.run(&module, wasi)?;
```

### Browser Execution

```rust
use bhc_wasm::browser::{BrowserRuntime, JsInterface};

// Create browser runtime (requires `browser` feature)
let runtime = BrowserRuntime::new()?;

// Load module
let module = runtime.load_bytes(&wasm_bytes)?;

// Create JS interface
let js = JsInterface::new(&module)?;

// Call exported function
let result: i32 = js.call("add", &[1.into(), 2.into()])?;
```

## Output Types

| Type | Extension | Description |
|------|-----------|-------------|
| Binary | `.wasm` | Binary WASM module |
| Text | `.wat` | WebAssembly text format |

## SIMD128 Support

WASM SIMD128 provides 128-bit vector operations:

```rust
use bhc_wasm::simd::*;

// 4x f32 operations (128-bit)
let v1 = f32x4(1.0, 2.0, 3.0, 4.0);
let v2 = f32x4(5.0, 6.0, 7.0, 8.0);
let sum = f32x4_add(v1, v2);

// Mapping from Loop IR SIMD
// VEC4F32 -> v128 with f32x4 ops
// VEC2F64 -> v128 with f64x2 ops
```

## Linear Memory Management

```rust
use bhc_wasm::memory::{LinearMemory, ArenaAllocator};

// Create linear memory (pages = 64KB each)
let memory = LinearMemory::new(initial_pages, max_pages)?;

// Arena allocation within linear memory
let arena = ArenaAllocator::new(&memory, offset, size);
let ptr = arena.alloc(1024)?;
```

## WASI Imports

| Module | Functions |
|--------|-----------|
| `wasi_snapshot_preview1` | fd_read, fd_write, fd_close, ... |
| `wasi_snapshot_preview1` | path_open, path_unlink, ... |
| `wasi_snapshot_preview1` | clock_time_get, random_get, ... |
| `wasi_snapshot_preview1` | proc_exit, args_get, environ_get |

## Edge Profile Optimizations

For minimal module size:

```rust
use bhc_wasm::EdgeConfig;

let config = EdgeConfig {
    // Strip debug info
    strip_debug: true,

    // Minimal runtime
    minimal_runtime: true,

    // Optimize for size
    opt_size: true,

    // No GC (static allocation only)
    no_gc: true,
};
```

## Error Types

```rust
pub enum WasmError {
    /// Invalid WASM module
    InvalidModule(String),

    /// WASI error
    WasiError(String),

    /// Memory access out of bounds
    OutOfBounds { address: u32, size: u32 },

    /// Import not found
    ImportNotFound { module: String, name: String },
}
```

## M8 Exit Criteria

- WebAssembly code generation produces valid modules
- SIMD128 used for vectorized numeric kernels
- Browser runtime loads and executes BHC code
- Edge profile produces minimal module size

## Design Notes

- Linear memory starts at 0 for WASI compatibility
- Stack grows downward from end of memory
- Arena allocator avoids GC in Edge profile
- SIMD128 requires browser/runtime support check

## Related Crates

- `bhc-codegen` - Native codegen counterpart
- `bhc-loop-ir` - Input Loop IR
- `bhc-target` - WASM target specifications
- `bhc-session` - Edge profile configuration

## Specification References

- H26-SPEC Section 12.3: WebAssembly Target
- WebAssembly Core Specification
- WASI Specification
