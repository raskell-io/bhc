# bhc-codegen

Code generation backend for the Basel Haskell Compiler.

## Overview

`bhc-codegen` provides the infrastructure for generating native code from the compiler's intermediate representations. Features:

- **Backend abstraction**: Pluggable code generation backends
- **LLVM support**: Primary backend with full optimization pipeline
- **Target-specific codegen**: CPU features, calling conventions
- **LTO support**: Link-time optimization

## Pipeline Position

```
Loop IR ──▶ Backend IR ──▶ Object Code
                │
                ├──▶ LLVM IR ──▶ LLVM Backend
                │
                └──▶ (Future: Cranelift, etc.)
```

## Core Types

| Type | Description |
|------|-------------|
| `CodegenBackend` | Trait for backends |
| `CodegenContext` | Backend context |
| `CodegenModule` | Compilation unit |
| `CodegenConfig` | Configuration |
| `LlvmBackend` | LLVM backend |

## CodegenConfig

```rust
pub struct CodegenConfig {
    /// Target specification
    pub target: TargetSpec,
    /// Optimization level
    pub opt_level: OptLevel,
    /// Debug information level
    pub debug_info: DebugInfo,
    /// Position-independent code
    pub pic: bool,
    /// Frame pointers
    pub frame_pointers: bool,
    /// Link-time optimization
    pub lto: bool,
    /// Target CPU model
    pub cpu: String,
}
```

### Builder Pattern

```rust
let config = CodegenConfig::for_target(target)
    .with_opt_level(OptLevel::Aggressive)
    .with_debug_info(DebugInfo::Full)
    .with_pic(true)
    .with_lto(true);
```

## Output Types

```rust
pub enum CodegenOutputType {
    /// Object file (.o)
    Object,
    /// Assembly (.s)
    Assembly,
    /// LLVM IR text (.ll)
    LlvmIr,
    /// LLVM bitcode (.bc)
    LlvmBitcode,
}
```

## Backend Trait

```rust
pub trait CodegenBackend: Send + Sync {
    type Context: CodegenContext;

    /// Backend name
    fn name(&self) -> &'static str;

    /// Check target support
    fn supports_target(&self, target: &TargetSpec) -> bool;

    /// Create codegen context
    fn create_context(&self, config: CodegenConfig) -> CodegenResult<Self::Context>;
}
```

## Context Trait

```rust
pub trait CodegenContext: Send + Sync {
    type Module: CodegenModule;

    /// Create a new module
    fn create_module(&self, name: &str) -> CodegenResult<Self::Module>;

    /// Get target specification
    fn target(&self) -> &TargetSpec;

    /// Get configuration
    fn config(&self) -> &CodegenConfig;
}
```

## Module Trait

```rust
pub trait CodegenModule: Send {
    /// Module name
    fn name(&self) -> &str;

    /// Verify the module
    fn verify(&self) -> CodegenResult<()>;

    /// Run optimization passes
    fn optimize(&mut self, level: OptLevel) -> CodegenResult<()>;

    /// Write to file
    fn write_to_file(&self, path: &Path, output_type: CodegenOutputType) -> CodegenResult<()>;

    /// Get LLVM IR text
    fn as_llvm_ir(&self) -> CodegenResult<String>;
}
```

## LLVM Backend

```rust
use bhc_codegen::{LlvmBackend, CodegenConfig};

// Create backend
let backend = LlvmBackend::new().expect("LLVM not available");

// Create context
let config = CodegenConfig::default();
let ctx = backend.create_context(config)?;

// Create module
let mut module = ctx.create_module("main")?;

// Optimize
module.optimize(OptLevel::Default)?;

// Write output
module.write_to_file(
    Path::new("output.o"),
    CodegenOutputType::Object
)?;
```

## IR Builder

Simple IR construction helper:

```rust
use bhc_codegen::IrBuilder;

let mut builder = IrBuilder::new();

// Define a function
builder.define_function("add", "i32", "i32 %a, i32 %b");
builder.build_ret("i32 %result");

let ir = builder.build();
```

## Type Layout

```rust
use bhc_codegen::TypeLayout;

// Get layout for target
let ptr_layout = TypeLayout::pointer(&target);
println!("Pointer: {} bytes, {} align", ptr_layout.size, ptr_layout.alignment);

// Built-in layouts
let i64_layout = TypeLayout::i64();  // 8 bytes, 8 align
let f64_layout = TypeLayout::f64();  // 8 bytes, 8 align
```

## Error Handling

```rust
pub enum CodegenError {
    /// Backend not available
    BackendNotAvailable(String),
    /// Target not supported
    UnsupportedTarget(String),
    /// LLVM error
    LlvmError(String),
    /// Output write failed
    OutputError { path: String, source: io::Error },
    /// Internal error
    Internal(String),
}
```

## Quick Start

```rust
use bhc_codegen::{LlvmBackend, CodegenConfig, CodegenOutputType};
use bhc_target::TargetSpec;

// Parse target
let target = TargetSpec::parse("x86_64-unknown-linux-gnu")?;

// Configure codegen
let config = CodegenConfig::for_target(target)
    .with_opt_level(OptLevel::Default)
    .with_pic(true);

// Create backend and context
let backend = LlvmBackend::new().unwrap();
let ctx = backend.create_context(config)?;

// Create and compile module
let module = ctx.create_module("my_program")?;

// Write object file
module.write_to_file(
    Path::new("my_program.o"),
    CodegenOutputType::Object
)?;
```

## See Also

- `bhc-loop-ir`: Input IR for code generation
- `bhc-target`: Target specifications
- `bhc-linker`: Linking object files
- `bhc-gpu`: GPU code generation
- `bhc-wasm`: WebAssembly code generation
