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
cd compiler && cabal build

# Compile a program
bhc hello.hs -o hello

# Run with Numeric Profile
bhc --profile=numeric matmul.hs -o matmul

# View kernel fusion report
bhc --profile=numeric -fkernel-report tensor_ops.hs
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
├── compiler/           # Compiler driver + front-end
│   └── src/
│       ├── Parser/     # Lexer, parser, AST
│       ├── TypeCheck/  # Type inference, typeclasses
│       ├── Core/       # Core IR
│       ├── Tensor/     # Tensor IR passes
│       ├── Loop/       # Loop IR, vectorization
│       └── Codegen/    # Backend code generation
├── rts/                # Runtime system
│   └── src/
│       ├── gc/         # Garbage collector
│       ├── arena/      # Hot arena allocator
│       └── scheduler/  # Task scheduler
├── stdlib/             # H26 Platform modules
│   └── H26/
│       ├── Tensor.hs
│       ├── Numeric.hs
│       ├── Concurrency.hs
│       └── ...
├── tests/              # Conformance test suite
├── spec/               # Specification documents
└── tools/              # bhci, bhi
```

---

## Roadmap

| Milestone | Name | Status |
|-----------|------|--------|
| M0 | Proof of Life | Not Started |
| M1 | Numeric Profile Skeleton | Not Started |
| M2 | Tensor IR v1 | Not Started |
| M3 | Vectorization + Parallel Loops | Not Started |
| M4 | Pinned Arrays + FFI | Not Started |
| M5 | Server Runtime Contract | Not Started |
| M6 | Platform Standardization | Not Started |

See [ROADMAP.md](.claude/ROADMAP.md) for detailed milestone specifications.

---

## Documentation

- [CLAUDE.md](.claude/CLAUDE.md) — Project overview and development guidelines
- [ROADMAP.md](.claude/ROADMAP.md) — Milestone schedule and exit criteria
- [rules/](.claude/rules/) — Code quality and design guidelines

---

## Building

### Prerequisites

- GHC 9.6+ (for bootstrapping)
- Cabal 3.10+
- LLVM 17+ (optional, for native codegen)

### Build Commands

```bash
# Build everything
cabal build all

# Run tests
cabal test

# Run benchmarks
cabal bench

# Build with optimizations
cabal build -O2
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
