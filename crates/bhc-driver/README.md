# bhc-driver

Compilation orchestration and pipeline for the Basel Haskell Compiler.

## Overview

This crate coordinates the entire compilation process, from source files to final output. It manages the compilation pipeline and handles orchestration of all compiler phases.

## Compilation Pipeline

```
Source Files
     │
     ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Parse  │ ──▶ │  Type   │ ──▶ │  Core   │
│         │     │  Check  │     │   IR    │
└─────────┘     └─────────┘     └─────────┘
                                     │
     ┌───────────────────────────────┘
     │
     ▼ (Numeric Profile)
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Tensor  │ ──▶ │  Loop   │ ──▶ │  Code   │
│   IR    │     │   IR    │     │   Gen   │
└─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                              ┌─────────┐
                              │  Link   │
                              └─────────┘
                                     │
                                     ▼
                                 Output
```

## Key Types

| Type | Description |
|------|-------------|
| `Driver` | Main compilation driver |
| `CompileError` | Errors during compilation |
| `CompileResult` | Result of compilation |
| `CompilationUnit` | A single file being compiled |

## Usage

### Basic Compilation

```rust
use bhc_driver::{Driver, compile_file};
use bhc_session::Session;

let session = Session::new(options)?;
let result = compile_file(&session, "Main.hs")?;

match result {
    CompileResult::Success { output_path, .. } => {
        println!("Compiled to: {}", output_path);
    }
    CompileResult::Errors { diagnostics } => {
        for diag in diagnostics {
            eprintln!("{}", diag);
        }
    }
}
```

### Full Pipeline

```rust
use bhc_driver::Driver;

let driver = Driver::new(session)?;

// Compile multiple files
for file in &source_files {
    driver.add_source(file)?;
}

// Run the full pipeline
let output = driver.compile()?;

// Access intermediate representations
if options.dump_core {
    for (name, core) in &output.core_modules {
        println!("=== {} ===\n{}", name, core.pretty_print());
    }
}
```

### Interpretation Mode

```rust
use bhc_driver::{interpret, Value};

// Run without generating an executable
let result: Value = interpret(&session, "let x = 1 + 2 in x * 3")?;
assert_eq!(result, Value::Int(9));
```

## Error Types

```rust
pub enum CompileError {
    /// Session creation failed
    SessionError(SessionError),

    /// Source file could not be read
    SourceReadError { path: Utf8PathBuf, source: io::Error },

    /// Parse errors occurred
    ParseError(usize),

    /// AST to HIR lowering failed
    LowerError(LowerError),

    /// Type checking failed
    TypeCheckError(usize),

    /// HIR to Core lowering failed
    CoreLowerError(CoreLowerError),

    /// Code generation failed
    CodegenError(CodegenError),

    /// Linking failed
    LinkError(LinkError),
}
```

## Compilation Phases

| Phase | Input | Output | Crate |
|-------|-------|--------|-------|
| Parse | Source | AST | `bhc-parser` |
| Lower | AST | HIR | `bhc-lower` |
| TypeCheck | HIR | Typed HIR | `bhc-typeck` |
| CoreLower | Typed HIR | Core | `bhc-hir-to-core` |
| Optimize | Core | Core | `bhc-core` |
| TensorLower | Core | Tensor IR | `bhc-tensor-ir` |
| LoopLower | Tensor IR | Loop IR | `bhc-loop-ir` |
| Codegen | Loop IR / Core | Object | `bhc-codegen` |
| Link | Objects | Executable | `bhc-linker` |

## Profile-Specific Pipeline

### Default/Server Profile

```
HIR → Core → Optimize → Codegen → Link
```

### Numeric Profile

```
HIR → Core → Tensor IR → Fusion → Loop IR → Vectorize → Codegen → Link
```

### Edge Profile

```
HIR → Core → Minimal Optimize → WASM Codegen
```

## Kernel Reports (Numeric Profile)

```rust
use bhc_driver::Driver;

let driver = Driver::new(session)?;
driver.add_source("Compute.hs")?;

// Get fusion report
let output = driver.compile()?;
for report in &output.kernel_reports {
    println!("Kernel: {}", report.name);
    println!("  Fused ops: {}", report.fused_count);
    println!("  SIMD width: {}", report.simd_width);
}
```

## Parallel Compilation

```rust
use bhc_driver::Driver;

let driver = Driver::new(session)?;

// Add all source files
for file in &source_files {
    driver.add_source(file)?;
}

// Compile in parallel (uses all available cores)
let output = driver.compile_parallel()?;
```

## Design Notes

- Each phase is independently testable
- Errors are collected and reported together
- Profile determines which phases are executed
- Parallel compilation respects module dependencies

## Related Crates

- `bhc-session` - Session configuration
- `bhc-parser` - Parsing phase
- `bhc-typeck` - Type checking phase
- `bhc-codegen` - Code generation
- `bhc-linker` - Linking phase

## Specification References

- H26-SPEC Section 3: IR Pipeline
- H26-SPEC Section 5: Optimization Passes
- H26-SPEC Section 8: Fusion (Numeric Profile)
