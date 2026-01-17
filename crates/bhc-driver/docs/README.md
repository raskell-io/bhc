# bhc-driver

Compilation driver and orchestration for the Basel Haskell Compiler.

## Overview

`bhc-driver` orchestrates the entire compilation pipeline, coordinating:

- **Phase execution**: Parse → TypeCheck → Core → Tensor → Loop → Codegen
- **Profile handling**: Different pipelines for different profiles
- **Multi-file compilation**: Module dependency ordering
- **Error aggregation**: Collect and report all diagnostics

## Core Types

| Type | Description |
|------|-------------|
| `Compiler` | Main compilation orchestrator |
| `CompilerBuilder` | Fluent builder for compiler |
| `CompilePhase` | Compilation phases |
| `CompileResult` | Result of compilation |
| `CompileUnit` | Single compilation unit |

## Compilation Pipeline

```
Source Files
    ↓
┌─────────────────────────────────────────────────┐
│ Parse Phase                                      │
│   Lexer → Parser → AST                          │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Lower to HIR                                     │
│   Name resolution, desugaring                   │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Type Check                                       │
│   Inference, checking, typed HIR                │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Lower to Core                                    │
│   Explicit types, A-normal form                 │
└─────────────────────────────────────────────────┘
    ↓ (Numeric Profile only)
┌─────────────────────────────────────────────────┐
│ Tensor IR                                        │
│   Shape inference, fusion analysis              │
└─────────────────────────────────────────────────┘
    ↓ (Numeric Profile only)
┌─────────────────────────────────────────────────┐
│ Loop IR                                          │
│   Vectorization, parallelization                │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Codegen                                          │
│   LLVM IR → Object code                         │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Link                                             │
│   Object files → Executable/Library             │
└─────────────────────────────────────────────────┘
```

## Compile Phases

```rust
pub enum CompilePhase {
    /// Lexing and parsing
    Parse,
    /// HIR lowering and name resolution
    HirLower,
    /// Type inference and checking
    TypeCheck,
    /// Core IR lowering
    CoreLower,
    /// Tensor IR (Numeric profile)
    TensorLower,
    /// Loop IR (Numeric profile)
    LoopLower,
    /// Code generation
    Codegen,
    /// Linking
    Link,
}

impl CompilePhase {
    /// Get all phases for a profile
    pub fn phases_for_profile(profile: Profile) -> Vec<CompilePhase> {
        match profile {
            Profile::Numeric => vec![
                Parse, HirLower, TypeCheck, CoreLower,
                TensorLower, LoopLower, Codegen, Link
            ],
            _ => vec![
                Parse, HirLower, TypeCheck, CoreLower,
                Codegen, Link
            ],
        }
    }
}
```

## Compiler

```rust
pub struct Compiler {
    /// Compilation session
    session: Session,
    /// Query database
    db: CompilerDatabase,
    /// Collected diagnostics
    diagnostics: Vec<Diagnostic>,
}

impl Compiler {
    /// Create with builder
    pub fn builder() -> CompilerBuilder;

    /// Compile a single file
    pub fn compile_file(&mut self, path: &Path) -> CompileResult;

    /// Compile multiple files
    pub fn compile_files(&mut self, paths: &[PathBuf]) -> CompileResult;

    /// Compile a crate
    pub fn compile_crate(&mut self, crate_root: &Path) -> CompileResult;

    /// Run up to a specific phase
    pub fn run_to_phase(&mut self, phase: CompilePhase) -> CompileResult;
}
```

## Compiler Builder

```rust
pub struct CompilerBuilder {
    profile: Profile,
    opt_level: OptLevel,
    target: Option<String>,
    output: Option<PathBuf>,
    search_paths: SearchPaths,
}

impl CompilerBuilder {
    pub fn new() -> Self;

    pub fn profile(mut self, profile: Profile) -> Self;
    pub fn opt_level(mut self, level: OptLevel) -> Self;
    pub fn target(mut self, triple: &str) -> Self;
    pub fn output(mut self, path: PathBuf) -> Self;
    pub fn search_path(mut self, path: PathBuf) -> Self;

    pub fn build(self) -> Result<Compiler, BuildError>;
}
```

## Quick Start

```rust
use bhc_driver::{Compiler, CompilePhase};
use bhc_session::Profile;

// Build compiler
let mut compiler = Compiler::builder()
    .profile(Profile::Numeric)
    .opt_level(OptLevel::Default)
    .target("x86_64-unknown-linux-gnu")
    .output("main".into())
    .build()?;

// Compile
let result = compiler.compile_file(Path::new("Main.hs"))?;

// Check for errors
if result.has_errors() {
    for diag in result.diagnostics() {
        eprintln!("{}", diag);
    }
    std::process::exit(1);
}
```

## Compile Result

```rust
pub struct CompileResult {
    /// Compilation succeeded
    pub success: bool,
    /// Output artifact (if any)
    pub output: Option<PathBuf>,
    /// Diagnostics (errors and warnings)
    pub diagnostics: Vec<Diagnostic>,
    /// Timing information
    pub timings: PhaseTimings,
}

impl CompileResult {
    pub fn has_errors(&self) -> bool;
    pub fn has_warnings(&self) -> bool;
    pub fn error_count(&self) -> usize;
    pub fn warning_count(&self) -> usize;
}

pub struct PhaseTimings {
    pub parse: Duration,
    pub type_check: Duration,
    pub core_lower: Duration,
    pub tensor_lower: Option<Duration>,
    pub loop_lower: Option<Duration>,
    pub codegen: Duration,
    pub link: Duration,
    pub total: Duration,
}
```

## Module Compilation

```rust
impl Compiler {
    /// Compile in dependency order
    pub fn compile_modules(&mut self, modules: &[ModuleId]) -> CompileResult {
        // 1. Build dependency graph
        let graph = self.build_dep_graph(modules);

        // 2. Topological sort
        let order = graph.topological_sort()?;

        // 3. Compile in order
        for module_id in order {
            self.compile_module(module_id)?;
        }

        Ok(CompileResult::success())
    }
}
```

## Target Detection

```rust
impl Compiler {
    /// Detect vectorization target
    fn detect_vector_target(&self) -> TargetArch {
        let target = self.session.opts.target.as_deref()
            .unwrap_or(env!("TARGET"));

        if target.starts_with("x86_64") {
            if self.has_feature("avx2") {
                TargetArch::X86_64Avx2
            } else if self.has_feature("avx") {
                TargetArch::X86_64Avx
            } else {
                TargetArch::X86_64Sse2
            }
        } else if target.starts_with("aarch64") {
            TargetArch::Aarch64Neon
        } else {
            TargetArch::Generic
        }
    }
}
```

## Incremental Compilation

```rust
impl Compiler {
    /// Incremental recompilation
    pub fn recompile_changed(&mut self, changed: &[FileId]) -> CompileResult {
        // Invalidate affected queries
        for file_id in changed {
            self.db.invalidate_file(*file_id);
        }

        // Recompile (queries automatically recompute as needed)
        self.compile_crate(&self.crate_root)
    }
}
```

## Parallel Compilation

```rust
impl Compiler {
    /// Compile modules in parallel where possible
    pub fn compile_parallel(&mut self, modules: &[ModuleId]) -> CompileResult {
        let graph = self.build_dep_graph(modules);

        // Find modules that can compile in parallel
        // (no dependencies between them)
        let levels = graph.parallel_levels();

        for level in levels {
            // Compile all modules in this level in parallel
            level.par_iter().try_for_each(|module_id| {
                self.compile_module(*module_id)
            })?;
        }

        Ok(CompileResult::success())
    }
}
```

## Callbacks

```rust
pub trait CompilerCallbacks {
    /// Called before each phase
    fn before_phase(&mut self, phase: CompilePhase) {}

    /// Called after each phase
    fn after_phase(&mut self, phase: CompilePhase, timing: Duration) {}

    /// Called on diagnostic
    fn on_diagnostic(&mut self, diag: &Diagnostic) {}
}

impl Compiler {
    pub fn with_callbacks<C: CompilerCallbacks>(
        mut self,
        callbacks: C
    ) -> Self;
}
```

## Dump IR

```rust
impl Compiler {
    /// Dump intermediate representation
    pub fn dump_ir(&self, phase: CompilePhase, module: ModuleId) -> String {
        match phase {
            CompilePhase::Parse => self.dump_ast(module),
            CompilePhase::HirLower => self.dump_hir(module),
            CompilePhase::CoreLower => self.dump_core(module),
            CompilePhase::TensorLower => self.dump_tensor(module),
            CompilePhase::LoopLower => self.dump_loop(module),
            CompilePhase::Codegen => self.dump_llvm(module),
            _ => String::new(),
        }
    }
}
```

## Error Handling

```rust
pub enum DriverError {
    /// Parse error
    Parse(Vec<Diagnostic>),
    /// Type error
    Type(Vec<Diagnostic>),
    /// Codegen error
    Codegen(String),
    /// Link error
    Link(String),
    /// IO error
    Io(std::io::Error),
    /// Module not found
    ModuleNotFound(String),
    /// Cyclic dependency
    CyclicDependency(Vec<ModuleId>),
}
```

## See Also

- `bhc-session`: Compilation options and profiles
- `bhc-query`: Incremental computation
- `bhc-codegen`: Code generation backend
- `bhc-linker`: Linking support
