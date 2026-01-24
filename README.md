# Basel Haskell Compiler (BHC)

**A next-generation Haskell compiler and runtime for 2026: predictable performance, structured concurrency, and a tensor-native numeric pipeline.**

---

## Overview

BHC is a clean-slate Haskell compiler and runtime built to remain compatible with the spirit of Haskell while introducing modern profiles, a standardized runtime contract, and a tensor-native compilation pipeline.

BHC is named after Basel, Switzerland — as a deliberate successor culture to the Glasgow Haskell Compiler lineage, with a new focus on predictability, concurrency, and numerical computing.

### Philosophy

> BHC prioritizes **predictability over folklore**: if performance matters, the compiler tells you what happened. If concurrency matters, cancellation is structured. If numerics matter, fusion is guaranteed and kernels are traceable.

---

## Features

### Profiles

BHC supports multiple compilation profiles with distinct performance contracts:

| Profile | Use Case | Key Characteristics |
|---------|----------|---------------------|
| **Default** | General Haskell | Lazy evaluation, GC managed |
| **Server** | Web services, daemons | Concurrency, bounded latency, observability |
| **Numeric** | ML, linear algebra, tensors | Strict-by-default, unboxed, fusion guaranteed |
| **Edge** | Embedded, WASM | Minimal runtime footprint |

### Numeric Performance

- **Strict-by-default** in Numeric Profile — no hidden thunks
- **Guaranteed fusion** for standard patterns (`map`, `zipWith`, `fold`)
- **Tensor IR** with shape/stride tracking for optimal code generation
- **SIMD auto-vectorization** for modern CPUs
- **Parallel primitives** with deterministic scheduling

### Structured Concurrency

- Scoped task spawning with automatic cleanup
- Cooperative cancellation with propagation
- Deadline and timeout support
- Event tracing for observability

### Memory Model

- **Hot Arena** — Bump allocation for loop temporaries
- **Pinned Heap** — Non-moving memory for FFI/DMA
- **General Heap** — GC-managed allocations

---

## Conformance Levels

BHC targets the Haskell 2026 Platform specification:

- **H26-Core** — Language core + minimal runtime contract
- **H26-Platform** — Core + standard libraries + packaging
- **H26-Numeric** — Platform + Numeric Profile + Tensor IR guarantees

---

## CLI Tools

| Command | Description |
|---------|-------------|
| `bhc` | Compiler driver |
| `bhci` | Interactive REPL |
| `bhi` | IR inspector / kernel reports |

---

## Quick Start

```bash
# Build the compiler
cargo build --release

# Compile a program
./target/release/bhc hello.hs -o hello

# Run with Numeric Profile
./target/release/bhc --profile=numeric matmul.hs -o matmul

# View kernel fusion report
./target/release/bhc --profile=numeric --kernel-report tensor_ops.hs

# Try it in your browser
# Visit https://bhc.raskell.io/playground/
```

---

## Example

```haskell
{-# HASKELL_EDITION 2026 #-}
{-# PROFILE Numeric #-}

module Main where

import H26.Tensor

-- Dot product: guaranteed to fuse into single loop
dot :: Tensor Float -> Tensor Float -> Float
dot xs ys = sum (zipWith (*) xs ys)

-- Matrix multiply: auto-vectorized, parallel
matmul :: Tensor Float -> Tensor Float -> Tensor Float
matmul a b = parMap (\i ->
    parMap (\j -> dot (row i a) (col j b)) [0..n-1]
  ) [0..m-1]
  where
    (m, _) = shape a
    (_, n) = shape b

main :: IO ()
main = do
  let a = fromList [2, 3] [1, 2, 3, 4, 5, 6]
      b = fromList [3, 2] [1, 2, 3, 4, 5, 6]
      c = matmul a b
  print c
```

---

## Project Structure

```
bhc/
├── crates/                    # Rust compiler implementation
│   ├── bhc/                   # Main CLI binary
│   ├── bhc-driver/            # Compilation orchestration
│   ├── bhc-parser/            # Parsing (lexer, AST)
│   ├── bhc-typeck/            # Type inference & checking
│   ├── bhc-core/              # Core IR + interpreter
│   ├── bhc-tensor-ir/         # Tensor IR (Numeric profile)
│   ├── bhc-loop-ir/           # Loop IR (vectorization)
│   ├── bhc-codegen/           # Native code generation (LLVM)
│   ├── bhc-wasm/              # WebAssembly backend
│   ├── bhc-gpu/               # GPU backends (CUDA/ROCm)
│   └── bhc-playground/        # Browser WASM playground
├── rts/                       # Runtime system (Rust)
│   ├── bhc-rts/               # Core runtime
│   └── bhc-rts-gc/            # Garbage collector
├── stdlib/                    # Standard library
│   ├── bhc-prelude/           # Prelude primitives
│   ├── bhc-base/              # Base library
│   ├── bhc-containers/        # Data structures
│   ├── bhc-numeric/           # Numeric/SIMD/BLAS
│   └── H26/                   # H26 Platform modules
├── tools/                     # Additional tools
│   ├── bhci/                  # Interactive REPL
│   ├── bhi/                   # IR inspector
│   └── bhc-docs/              # Documentation generator
└── tests/                     # Test suites
```

---

## Roadmap

| Milestone | Name | Status |
|-----------|------|--------|
| Phase 1 | Native Hello World | ✅ Complete |
| Phase 2 | Language Completeness | 🟡 In Progress |
| Phase 3 | Numeric Profile | 🟡 Partial |
| Phase 4 | WASM Backend | 🟡 Partial |
| Phase 5 | Server Profile | 🔴 Not Started |
| Phase 6 | GPU Backend | 🔴 Skeleton |

See [ROADMAP.md](ROADMAP.md) for detailed milestone specifications.

---

## Documentation

- [Website](https://bhc.raskell.io) — Official website with guides and tutorials
- [API Docs](https://bhc.raskell.io/docs/api/) — Standard library reference (63 modules)
- [Playground](https://bhc.raskell.io/playground/) — Try BHC in your browser
- [ROADMAP.md](ROADMAP.md) — Implementation status and milestones
- [.claude/CLAUDE.md](.claude/CLAUDE.md) — Development guidelines

---

## Building

### Prerequisites

- Rust 1.75+ (stable toolchain)
- LLVM 17+ (for native codegen)
- wasm32-unknown-unknown target (for playground)

### Build Commands

```bash
# Build everything
cargo build

# Build release
cargo build --release

# Run tests
cargo test

# Run specific crate tests
cargo test -p bhc-parser

# Run benchmarks
cargo bench

# Build and run bhc
cargo run --bin bhc -- Main.hs
```

---

## Contributing

Contributions are welcome. Please read the guidelines in `.claude/rules/` before submitting changes.

### Commit Messages

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `perf:` Performance improvements
- `refactor:` Code restructuring
- `docs:` Documentation
- `test:` Test additions/changes

---

## License

BSD-3-Clause

---

## Acknowledgments

BHC builds on decades of research in functional programming, type systems, and compiler construction. We acknowledge the foundational work of the GHC team and the broader Haskell community.
