# bhc

Main CLI entry point for the Basel Haskell Compiler.

## Overview

The `bhc` binary is the primary user-facing interface to the BHC compilation infrastructure. It orchestrates the entire compilation pipeline from source code to executable.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         bhc CLI                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  parse  │→ │  lower  │→ │ typeck  │→ │ codegen │        │
│  │  args   │  │  (HIR)  │  │         │  │         │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│       │            │            │            │              │
│       ▼            ▼            ▼            ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    bhc-driver                        │   │
│  │              (compilation orchestration)             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Command Structure

### Main Commands

| Command | Description |
|---------|-------------|
| `bhc <files>` | Compile files (default) |
| `bhc build` | Build project |
| `bhc check` | Type check only |
| `bhc run` | Compile and execute |
| `bhc eval <expr>` | Evaluate expression |
| `bhc repl` | Start interactive REPL |

### Subcommands

```rust
#[derive(Subcommand, Debug)]
enum Commands {
    /// Build the project
    Build {
        #[arg(short, long)]
        release: bool,
    },

    /// Type check without generating code
    Check {
        files: Vec<PathBuf>,
    },

    /// Compile and run
    Run {
        file: PathBuf,
        #[arg(last = true)]
        args: Vec<String>,
    },

    /// Evaluate an expression
    Eval {
        expression: String,
    },

    /// Start the REPL
    Repl,

    /// Clean build artifacts
    Clean,

    /// Initialize a new project
    Init {
        name: Option<String>,
    },
}
```

## Profile System

### Profile Selection

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Profile {
    /// Standard lazy Haskell
    Default,
    /// Bounded latency for servers
    Server,
    /// Strict numeric optimization
    Numeric,
    /// Minimal footprint (WASM/embedded)
    Edge,
    /// Bounded GC pauses
    Realtime,
    /// Static allocation only
    Embedded,
}
```

### Profile Effects

| Profile | Evaluation | GC | Fusion | SIMD |
|---------|------------|-----|--------|------|
| Default | Lazy | Standard | Opportunistic | No |
| Server | Lazy | Incremental | Opportunistic | No |
| Numeric | Strict | Minimal | Guaranteed | Yes |
| Edge | Lazy | Minimal | Opportunistic | No |
| Realtime | Lazy | Bounded | Opportunistic | No |
| Embedded | Strict | None | Opportunistic | Optional |

## IR Dump System

### Dump Stages

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum IrStage {
    /// Abstract syntax tree
    Ast,
    /// High-level IR (desugared)
    Hir,
    /// Core IR (typed, explicit)
    Core,
    /// Tensor IR (Numeric profile)
    Tensor,
    /// Loop IR (vectorization)
    Loop,
    /// LLVM IR
    Llvm,
}
```

### Dump Options

```bash
# Single stage
bhc --dump-ir=core Main.hs

# Multiple stages
bhc --dump-ir=hir --dump-ir=core Main.hs

# With optimization passes
bhc --dump-ir=core -ddump-core-passes Main.hs
```

## Target System

### Supported Targets

| Target | Triple | Status |
|--------|--------|--------|
| Native | `x86_64-unknown-linux-gnu` | ✅ |
| macOS | `aarch64-apple-darwin` | ✅ |
| Windows | `x86_64-pc-windows-msvc` | 🔄 |
| WASM | `wasm32-wasi` | ✅ |
| CUDA | `nvptx64-nvidia-cuda` | 🔄 |
| ROCm | `amdgcn-amd-amdhsa` | 🔄 |

### Target Selection

```bash
# Native (auto-detected)
bhc Main.hs

# Explicit target
bhc --target=wasm32-wasi Main.hs

# Cross-compilation
bhc --target=aarch64-apple-darwin Main.hs
```

## Optimization Levels

| Level | Description |
|-------|-------------|
| `-O0` | No optimization (default) |
| `-O1` | Basic optimizations |
| `-O2` | Standard optimizations |
| `-O3` | Aggressive optimizations |
| `-Os` | Optimize for size |

## Error Reporting

### Error Format

```
error[E0308]: type mismatch
  --> src/Main.hs:10:5
   |
10 |     x + "hello"
   |     ^^^^^^^^^^^ expected `Int`, found `String`
   |
   = note: cannot add Int and String
   = help: consider using `show` to convert Int to String
```

### Diagnostic Flags

| Flag | Description |
|------|-------------|
| `-Wall` | Enable all warnings |
| `-Werror` | Treat warnings as errors |
| `-Wno-<name>` | Disable specific warning |
| `--color=<auto\|always\|never>` | Color output |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Compilation error |
| 2 | Type error |
| 3 | Runtime error |
| 4 | Internal compiler error |
| 5 | Invalid arguments |

## Configuration

### Config File

`bhc.toml`:
```toml
[package]
name = "my-project"
version = "0.1.0"

[build]
profile = "default"
optimization = 2

[dependencies]
base = "1.0"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BHC_HOME` | BHC installation directory |
| `BHC_CACHE` | Compilation cache directory |
| `BHC_JOBS` | Default parallelism |
| `BHC_COLOR` | Color output preference |

## Implementation Notes

### Initialization Sequence

1. Parse command-line arguments
2. Load configuration file (if present)
3. Initialize session with profile/edition
4. Set up diagnostic emitter
5. Initialize query database
6. Dispatch to appropriate command handler

### Driver Integration

```rust
pub fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    init_tracing(cli.verbose);

    // Create session
    let session = Session::new(SessionConfig {
        profile: cli.profile.into(),
        edition: parse_edition(&cli.edition)?,
        opt_level: cli.opt_level,
        ..Default::default()
    })?;

    // Run driver
    let driver = Driver::new(session);

    match cli.command {
        Some(Commands::Build { .. }) => driver.build(&cli.files),
        Some(Commands::Check { .. }) => driver.check(&cli.files),
        Some(Commands::Run { .. }) => driver.run(&cli.files),
        None => driver.compile(&cli.files, cli.output),
    }
}
```

## See Also

- `bhci` - Interactive REPL
- `bhi` - IR inspector
- `bhc-driver` - Compilation orchestration
- `bhc-session` - Session management
