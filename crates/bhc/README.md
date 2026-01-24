# bhc

Main entry point for the Basel Haskell Compiler.

## Overview

This is the primary command-line interface for BHC, providing compilation, type checking, and code execution for Haskell 2026 source files.

## Installation

```bash
# From source
cargo install --path crates/bhc

# Or using the install script
curl -fsSL https://bhc.raskell.io/install.sh | sh
```

## Usage

### Basic Compilation

```bash
# Compile to executable
bhc Main.hs -o main

# Compile with optimization
bhc -O2 Main.hs -o main

# Type check only
bhc check Main.hs
```

### Subcommands

| Command | Description |
|---------|-------------|
| `build` | Compile source files |
| `check` | Type check without codegen |
| `run` | Compile and execute |
| `eval` | Evaluate expression |
| `repl` | Start interactive REPL |

### Profiles

```bash
# Default (lazy evaluation)
bhc Main.hs

# Server (bounded latency)
bhc --profile=server Main.hs

# Numeric (strict, SIMD, fusion)
bhc --profile=numeric Main.hs

# Edge (minimal runtime)
bhc --profile=edge Main.hs
```

### Haskell Editions

```bash
bhc --edition=Haskell2010 Main.hs
bhc --edition=GHC2021 Main.hs
bhc --edition=GHC2024 Main.hs
bhc --edition=H26 Main.hs  # Default
```

### IR Dumps

```bash
# Dump AST
bhc --dump-ir=ast Main.hs

# Dump HIR
bhc --dump-ir=hir Main.hs

# Dump Core IR
bhc --dump-ir=core Main.hs

# Dump Tensor IR (Numeric profile)
bhc --profile=numeric --dump-ir=tensor Main.hs

# Dump Loop IR
bhc --profile=numeric --dump-ir=loop Main.hs
```

### Kernel Reports

```bash
# Generate kernel fusion report (Numeric profile)
bhc --profile=numeric --kernel-report Main.hs
```

### Target Selection

```bash
# Native (default)
bhc Main.hs

# WebAssembly
bhc --target=wasi Main.hs -o app.wasm

# GPU (CUDA)
bhc --target=cuda --profile=numeric Main.hs
```

## CLI Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file name |
| `-O<n>` | Optimization level (0-3) |
| `--profile` | Compilation profile |
| `--edition` | Haskell edition |
| `--target` | Target platform |
| `--dump-ir` | Dump intermediate representation |
| `--kernel-report` | Show kernel fusion report |
| `-j, --jobs` | Parallel compilation jobs |
| `-v, --verbose` | Verbose output |

## Examples

### Hello World

```bash
echo 'main = putStrLn "Hello, World!"' > Main.hs
bhc Main.hs -o hello
./hello
```

### Numeric Computation

```bash
cat > compute.hs << 'EOF'
import BHC.Tensor

main = print $ sum $ map (*2) [1..1000000]
EOF

bhc --profile=numeric compute.hs -o compute
./compute
```

### WASM Target

```bash
bhc --target=wasi Main.hs -o app.wasm
wasmtime app.wasm
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Compilation error |
| 2 | Type error |
| 3 | Runtime error |
| 4 | Internal error |

## Related Tools

- `bhci` - Interactive REPL
- `bhi` - IR inspector

## Specification References

- H26-SPEC Section 1: Language Overview
- H26-SPEC Section 2: Runtime Profiles
