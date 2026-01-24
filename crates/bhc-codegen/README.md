# bhc-codegen

Code generation backend for the Basel Haskell Compiler.

## Overview

This crate provides the infrastructure for generating native code from the compiler's intermediate representations. The primary backend is LLVM, supporting full optimization and multi-target code generation.

## Code Generation Pipeline

### Standard Pipeline (Default/Server Profiles)

```
Core IR ──▶ LLVM IR ──▶ Object Code ──▶ Executable
```

### Numeric Pipeline

```
Core IR ──▶ Tensor IR ──▶ Loop IR ──▶ LLVM IR ──▶ Object Code
```

## Key Types

| Type | Description |
|------|-------------|
| `CodegenBackend` | Trait for code generation backends |
| `CodegenContext` | Context holding state during code generation |
| `CodegenModule` | A compilation unit in backend representation |
| `CodegenConfig` | Configuration for code generation |
| `LlvmBackend` | LLVM backend implementation |

## Usage

### Basic Code Generation

```rust
use bhc_codegen::{CodegenBackend, CodegenConfig, llvm::LlvmBackend};

// Create the LLVM backend
let backend = LlvmBackend::new();

// Create a context for code generation
let config = CodegenConfig::default();
let ctx = backend.create_context(config)?;

// Create a module
let module = ctx.create_module("my_module")?;

// ... add functions and generate code ...

// Write output
module.write_to_file(Path::new("output.ll"), CodegenOutputType::LlvmIr)?;
```

### Lowering Core IR

```rust
use bhc_codegen::llvm::{lower_core_module, LlvmBackend};

let backend = LlvmBackend::new();
let ctx = backend.create_context(config)?;

// Lower Core IR to LLVM IR
let llvm_module = lower_core_module(&ctx, &core_module)?;

// Generate object file
llvm_module.write_to_file("output.o", CodegenOutputType::Object)?;
```

## LLVM Backend Features

- Full optimization pipeline integration
- Target-specific code generation
- Debug information generation (DWARF)
- Link-time optimization (LTO) support
- Profile-guided optimization (PGO)

## Output Types

| Type | Description |
|------|-------------|
| `Object` | Native object file (.o) |
| `Assembly` | Assembly source (.s) |
| `LlvmIr` | LLVM IR text (.ll) |
| `LlvmBitcode` | LLVM bitcode (.bc) |

## Optimization Levels

```rust
use bhc_codegen::CodegenConfig;
use bhc_session::OptLevel;

let config = CodegenConfig {
    opt_level: OptLevel::Aggressive,  // -O3
    ..Default::default()
};
```

| Level | Description |
|-------|-------------|
| `None` | No optimizations (-O0) |
| `Less` | Basic optimizations (-O1) |
| `Default` | Standard optimizations (-O2) |
| `Aggressive` | Aggressive optimizations (-O3) |
| `Size` | Optimize for size (-Os) |
| `SizeMin` | Minimal size (-Oz) |

## Function Generation

```rust
use bhc_codegen::llvm::FunctionBuilder;

// Create a function
let func = module.add_function("my_func", func_type)?;
let builder = FunctionBuilder::new(&ctx, func);

// Build basic blocks
let entry = builder.append_block("entry");
builder.position_at_end(entry);

// Add instructions
let sum = builder.build_add(a, b, "sum");
builder.build_return(sum);
```

## Calling Conventions

| Convention | Description |
|------------|-------------|
| `C` | Standard C calling convention |
| `Fast` | Fast calling convention (caller-saved) |
| `Cold` | Cold function (rarely called) |
| `Tail` | Tail call optimized |

## Debug Information

```rust
use bhc_codegen::CodegenConfig;
use bhc_session::DebugInfo;

let config = CodegenConfig {
    debug_info: DebugInfo::Full,  // Generate DWARF debug info
    ..Default::default()
};
```

## Vectorization

For numeric code, the codegen enables auto-vectorization:

```rust
use bhc_codegen::llvm::VectorizeConfig;

let config = VectorizeConfig {
    vector_width: 256,   // AVX
    enable_slp: true,    // Superword-level parallelism
    prefer_vector: true, // Prefer vector ops
};
```

## Error Types

```rust
pub enum CodegenError {
    /// Backend not available
    BackendNotAvailable(String),

    /// Unsupported target
    UnsupportedTarget(String),

    /// LLVM error
    LlvmError(String),

    /// Invalid IR
    InvalidIr(String),

    /// I/O error
    IoError(std::io::Error),
}
```

## Design Notes

- LLVM context is not thread-safe (one per thread)
- Modules can be compiled in parallel
- Target features affect generated code
- Debug info is optional and incremental

## Related Crates

- `bhc-core` - Input Core IR
- `bhc-loop-ir` - Input Loop IR (Numeric)
- `bhc-target` - Target specifications
- `bhc-linker` - Links generated objects
- `bhc-driver` - Compilation orchestration

## Specification References

- H26-SPEC Section 3.5: Code Generation
- LLVM Language Reference Manual
