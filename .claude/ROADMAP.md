# BHC Roadmap

**Document ID:** BHC-ROAD-0001
**Status:** Active
**Last Updated:** 2026-01

---

## Overview

This roadmap tracks the implementation of the Basel Haskell Compiler (BHC) from proof-of-concept to H26-Numeric conformance. Each milestone has concrete exit criteria that must be met before proceeding.

---

## Milestone Summary

| Milestone | Name | Weeks | Status |
|-----------|------|-------|--------|
| M0 | Proof of Life | 1–4 | Not Started |
| M1 | Numeric Profile Skeleton | 5–10 | Not Started |
| M2 | Tensor IR v1 | 11–18 | Not Started |
| M3 | Vectorization + Parallel Loops | 19–28 | Not Started |
| M4 | Pinned Arrays + FFI | 29–36 | Not Started |
| M5 | Server Runtime Contract | 37–46 | Not Started |
| M6 | Platform Standardization | 47–60 | Not Started |

---

## M0 — Proof of Life (Weeks 1–4)

**Goal:** Demonstrate basic compiler pipeline works end-to-end.

### Deliverables

- [ ] Lexer and parser for core Haskell subset
- [ ] Minimal type checker (Hindley-Milner + basic typeclasses)
- [ ] Core IR representation
- [ ] Tree-walking interpreter for Core IR
- [ ] Default Profile only (lazy semantics)
- [ ] `UArray` prototype with `map`/`fold` semantics

### Exit Criteria

- `map`, `zipWith`, `sum` work correctly on `UArray`
- Can compile and interpret: `sum (map (+1) [1,2,3,4,5])`

### Key Files to Create

```
compiler/src/Parser/Lexer.hs
compiler/src/Parser/Parser.hs
compiler/src/Parser/AST.hs
compiler/src/TypeCheck/Infer.hs
compiler/src/Core/IR.hs
compiler/src/Core/Interpreter.hs
stdlib/H26/UArray.hs
```

---

## M1 — Numeric Profile Skeleton (Weeks 5–10)

**Goal:** Implement strict-by-default evaluation and unboxed primitives.

### Deliverables

- [ ] Numeric Profile strict-by-default semantics
- [ ] Unboxed primitive types: `I32`, `I64`, `F32`, `F64`
- [ ] Unboxed `UArray` representation (flat contiguous)
- [ ] Hot Arena allocator in RTS
- [ ] `lazy { }` escape hatch syntax
- [ ] Basic strictness analysis pass

### Exit Criteria

- Dot product compiles with zero thunks
- Allocation in hot loop is arena-only (instrumented)
- `let x = 1 + 2 in x * 3` evaluates strictly (no thunk created)

### Key Technical Decisions

1. **Strictness annotation syntax**: `lazy { expr }` vs `~expr`
2. **Arena lifetime**: Scope-based vs explicit deallocation
3. **Unboxed representation**: Stack vs dedicated registers

---

## M2 — Tensor IR v1 (Weeks 11–18)

**Goal:** Introduce Tensor type and guaranteed fusion.

### Deliverables

- [ ] `Tensor` type with shape/stride metadata
- [ ] View operations: `reshape`, `slice`, `transpose`
- [ ] Tensor IR representation
- [ ] Lowering pass: recognized patterns → Tensor IR
- [ ] Fusion pass for guaranteed patterns
- [ ] Kernel report mode (`-fkernel-report`)

### Exit Criteria

- `sum (map f x)` becomes single loop kernel
- Kernel report shows fusion succeeded
- `reshape` on contiguous tensor is metadata-only

### Guaranteed Fusion Patterns

These MUST fuse by end of M2:

```haskell
-- Pattern 1: map composition
map f (map g x)  →  map (f . g) x

-- Pattern 2: zip with maps
zipWith f (map g a) (map h b)  →  zipWith (\x y -> f (g x) (h y)) a b

-- Pattern 3: reduction of map
sum (map f x)  →  foldl' (\acc x -> acc + f x) 0 x

-- Pattern 4: fold of map
foldl' op z (map f x)  →  foldl' (\acc x -> op acc (f x)) z x
```

### Tensor IR Properties

Each Tensor IR node must track:
- `dtype`: Element type
- `shape`: `[Int]` dimension sizes
- `strides`: `[Int]` byte strides per dimension
- `layout`: `Contiguous | Strided | Tiled`
- `alias`: Pointer to underlying buffer (for views)

---

## M3 — Vectorization + Parallel Loops (Weeks 19–28)

**Goal:** Auto-vectorization and parallel execution.

### Deliverables

- [ ] Loop IR representation (post-Tensor IR)
- [ ] SIMD primitive types: `Vec4F32`, `Vec8F32`, `Vec2F64`, `Vec4F64`
- [ ] Auto-vectorization pass
- [ ] SIMD intrinsics: `add`, `mul`, `fmadd`, `hadd`
- [ ] `parFor`, `parMap`, `parReduce` primitives
- [ ] Work-stealing scheduler in RTS
- [ ] Deterministic chunking mode

### Exit Criteria

- `matmul` microkernel auto-vectorizes on x86_64 and aarch64
- Reductions scale linearly up to 8 cores
- Deterministic mode produces identical results across runs

### Parallel Constructs API

```haskell
parFor :: Range -> (Int -> ()) -> ()
parMap :: (a -> b) -> Tensor a -> Tensor b
parReduce :: Monoid m => (a -> m) -> Tensor a -> m
```

### Scheduling Contract

- Chunking MUST be deterministic given fixed worker count
- Non-deterministic mode allowed for floats (document variance)

---

## M4 — Pinned Arrays + FFI (Weeks 29–36)

**Goal:** Enable zero-copy FFI and external BLAS integration.

### Deliverables

- [ ] Pinned heap region in RTS
- [ ] `PinnedUArray` type
- [ ] Tensor buffers optionally pinned
- [ ] Safe FFI boundary with pinned buffer support
- [ ] BLAS provider interface
- [ ] Reference OpenBLAS integration

### Exit Criteria

- `matmul` can call external BLAS for large sizes
- Tensors stay pinned across FFI calls (verified by address stability)
- No GC movement of pinned allocations (stress test)

### FFI Safety Model

```haskell
-- Safe: buffer guaranteed pinned for duration
withPinnedTensor :: Tensor a -> (Ptr a -> IO b) -> IO b

-- Unsafe: caller responsible for lifetime
unsafeTensorPtr :: Tensor a -> Ptr a
```

---

## M5 — Server Runtime Contract (Weeks 37–46)

**Goal:** Production-ready concurrent runtime.

### Deliverables

- [ ] Structured concurrency primitives
- [ ] Cancellation propagation (cooperative)
- [ ] Deadline/timeout support
- [ ] Per-core scheduling
- [ ] Incremental/concurrent GC
- [ ] Event tracing hooks (GC, tasks, allocations)

### Exit Criteria

- Server workload runs concurrently without numeric kernel regressions
- Cancellation propagates within 1ms of request
- GC pause times < 10ms at p99

### Concurrency API

```haskell
withScope :: (Scope -> IO a) -> IO a
spawn :: Scope -> IO a -> IO (Task a)
cancel :: Task a -> IO ()
await :: Task a -> IO a
withDeadline :: Duration -> IO a -> IO (Maybe a)
```

### Event Tracing

Events that MUST be traceable:
- GC start/stop
- Task spawn/complete/cancel
- Arena allocation/reset
- Kernel compilation events

---

## M6 — Platform Standardization (Weeks 47–60)

**Goal:** Ship complete H26-Platform and conformance suite.

### Deliverables

- [ ] All H26 Platform modules implemented
- [ ] Conformance test suite (semantic + runtime + benchmarks)
- [ ] Package manifest schema
- [ ] Lockfile format
- [ ] Reproducible build verification
- [ ] Published benchmark results

### Exit Criteria

- "H26-Platform" claim is credible and testable
- All conformance tests pass
- Numeric benchmarks published and reproducible

### Platform Modules

| Module | Description | Priority |
|--------|-------------|----------|
| `H26.Bytes` | Byte arrays, slicing, pinned | Required |
| `H26.Text` | UTF-8 text | Required |
| `H26.Vector` | Boxed + unboxed vectors | Required |
| `H26.Time` | Time and duration | Required |
| `H26.Random` | Random number generation | Required |
| `H26.JSON` | Minimal JSON API | Required |
| `H26.FFI` | Foreign function interface | Required |
| `H26.Concurrency` | Scopes, tasks, cancellation | Required |
| `H26.Numeric` | Scalar ops, SIMD intrinsics | Required |
| `H26.Tensor` | Tensor ops, matmul, views | Required |
| `H26.BLAS` | Pluggable BLAS backend | Numeric |
| `H26.Device` | GPU execution contract | Optional |

### Conformance Tests

1. **Semantic Tests** (50+)
   - Strictness per profile
   - Exception propagation
   - Determinism requirements

2. **Runtime Tests** (30+)
   - Cancellation propagation
   - Concurrency correctness
   - Pinned allocation immovability
   - Atomic memory ordering

3. **Numeric Benchmarks** (10+)
   - Dot product
   - SAXPY
   - MatMul (small/medium/large)
   - Reductions (sum/max)
   - Fusion scenarios
   - Convolution (optional)

---

## Future Milestones (Post-1.0)

### M7 — GPU Backend (TBD)

- CUDA/ROCm code generation
- Device memory management
- Kernel fusion across host/device boundary

### M8 — WASM Target (TBD)

- WebAssembly code generation
- Browser runtime
- Edge profile optimization

### M9 — Dependent Types Preview (TBD)

- Shape-indexed tensors
- Compile-time dimension checking
- Gradual adoption path

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fusion complexity | High | Start with simple patterns, iterate |
| GC latency in Server | Medium | Incremental GC from M5 |
| SIMD portability | Medium | Abstract over target widths |
| FFI safety | High | Conservative defaults, unsafe escape hatch |
| Scope creep | High | Strict exit criteria per milestone |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| TBD | Implementation language | Haskell for compiler, Rust for RTS |
| TBD | LLVM vs custom backend | TBD based on M0 learnings |
| TBD | GC algorithm | TBD based on M5 requirements |

---

## References

- [H26-SPEC-0001] Haskell 2026 Platform & Runtime Specification
- [BHC-SPEC-0001] Basel Haskell Compiler Specification
- See `CLAUDE.md` for project overview
- See `rules/` for coding guidelines
