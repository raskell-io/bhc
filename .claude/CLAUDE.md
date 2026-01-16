# Basel Haskell Compiler (BHC)

**Codename:** BHC
**Document ID:** BHC-SPEC-0001
**Scope:** Reference compiler + runtime for the Haskell 2026 Platform
**Primary Mission:** Modern runtime contract + world-class numeric performance

---

## Project Identity

BHC ("Basel Haskell Compiler") is a clean-slate Haskell compiler and runtime, built to remain compatible with the spirit of Haskell while introducing modern profiles (Default/Server/Numeric/Edge), a standardized runtime contract, and a tensor-native compilation pipeline.

BHC is named after Basel, Switzerland — as a deliberate successor culture to the Glasgow Haskell Compiler lineage, with a new focus on predictability, concurrency, and numerical computing.

### Philosophy

BHC prioritizes **predictability over folklore**: if performance matters, the compiler tells you what happened. If concurrency matters, cancellation is structured. If numerics matter, fusion is guaranteed and kernels are traceable.

### One-Liner

> BHC makes Haskell a serious 2026 systems and numeric platform — without sacrificing purity.

---

## Repository Structure

```
bhc/
├── compiler/           # Compiler driver + front-end (bhc)
│   ├── src/
│   │   ├── Parser/     # Lexer, parser, AST
│   │   ├── TypeCheck/  # Type inference, typeclasses
│   │   ├── Core/       # Core IR representation
│   │   ├── Tensor/     # Tensor IR passes
│   │   ├── Loop/       # Loop IR, vectorization
│   │   ├── Codegen/    # Backend code generation
│   │   └── Driver/     # CLI, orchestration
│   └── tests/
├── rts/                # Runtime system (bhc-rts)
│   ├── src/
│   │   ├── gc/         # Garbage collector
│   │   ├── arena/      # Hot arena allocator
│   │   ├── scheduler/  # Task scheduler
│   │   └── ffi/        # FFI support
│   └── tests/
├── stdlib/             # H26 Platform modules
│   ├── H26/
│   │   ├── Bytes.hs
│   │   ├── Text.hs
│   │   ├── Vector.hs
│   │   ├── Tensor.hs
│   │   ├── Numeric.hs
│   │   ├── Concurrency.hs
│   │   └── ...
│   └── tests/
├── tests/              # Conformance test suite
│   ├── semantic/
│   ├── runtime/
│   └── benchmarks/
├── spec/               # Specification documents
└── tools/              # bhci (REPL), bhi (inspector)
```

---

## CLI Tools

| Command | Description |
|---------|-------------|
| `bhc` | Compiler driver |
| `bhci` | Interactive REPL |
| `bhi` | IR inspector / kernel reports |

---

## Conformance Levels

BHC targets three conformance levels:

1. **H26-Core**: Language core + minimal runtime contract
2. **H26-Platform**: Core + required standard libraries + packaging metadata
3. **H26-Numeric**: Platform + Numeric Profile + Tensor IR guarantees

---

## Profiles

Profiles define behavioral + performance contracts. The compiler MUST allow selecting a profile per package.

| Profile | Use Case | Key Characteristics |
|---------|----------|---------------------|
| **Default** | General Haskell | Lazy evaluation, GC managed |
| **Server** | Web services, daemons | Concurrency, bounded latency, observability |
| **Numeric** | ML, linear algebra, tensors | Strict-by-default, unboxed, fusion guaranteed |
| **Edge** | Embedded, WASM | Minimal runtime footprint |

---

## Development Guidelines

### Language

BHC is implemented in **Haskell** (bootstrapping) with performance-critical runtime components in **Rust** or **C**.

### Core Principles

1. **Correctness first** — Semantic correctness is non-negotiable
2. **Predictable performance** — No hidden allocations or thunks in Numeric Profile
3. **Transparency** — Kernel reports, fusion diagnostics, allocation tracking
4. **Modularity** — Clean IR boundaries, pluggable backends

### Code Quality

- All code MUST pass the linter and formatter
- All public APIs MUST have documentation
- All new features MUST have tests
- Performance-critical code MUST have benchmarks

### Commit Messages

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `perf:` Performance improvements
- `refactor:` Code restructuring
- `docs:` Documentation
- `test:` Test additions/changes
- `chore:` Build, tooling, etc.

### Pull Requests

- Reference the relevant spec section (e.g., "Implements Section 7.3 Tensor IR")
- Include benchmark results for performance changes
- Update conformance tests when behavior changes

---

## Key Technical Concepts

### Tensor IR (Section 7)

The Tensor IR is the heart of BHC's numeric performance. Each kernel must have:
- Element type (dtype)
- Shape vector
- Stride vector
- Layout tags (contiguous, strided, tiled)
- Aliasing information

### Fusion Guarantees (Section 8)

These patterns MUST fuse without intermediate allocation:
1. `map f (map g x)`
2. `zipWith f (map g a) (map h b)`
3. `sum (map f x)`
4. `foldl' op z (map f x)`

### Memory Model (Section 9)

Three allocation spaces:
1. **Hot Arena** — Bump allocator, freed at scope end
2. **Pinned Heap** — Non-moving, for FFI/device IO
3. **General Heap** — GC-managed boxed structures

### Structured Concurrency (Section 10)

Required primitives:
- `withScope :: (Scope -> IO a) -> IO a`
- `spawn :: Scope -> IO a -> IO (Task a)`
- `cancel :: Task a -> IO ()`
- `await :: Task a -> IO a`

---

## Building

```bash
# Build compiler
cd compiler && cabal build

# Run tests
cabal test

# Build with optimizations
cabal build -O2

# Run benchmarks
cabal bench
```

---

## Testing

### Test Categories

1. **Semantic Tests** — Strictness, exceptions, determinism
2. **Runtime Tests** — Cancellation, concurrency, pinned allocation
3. **Numeric Benchmarks** — dot product, saxpy, matmul, reductions

### Running Conformance Suite

```bash
cd tests && ./run-conformance.sh
```

---

## References

- See `ROADMAP.md` for milestone schedule
- See `spec/` for full specification documents
- See `rules/` for detailed coding guidelines
