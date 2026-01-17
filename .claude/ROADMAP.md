# BHC Roadmap

**Document ID:** BHC-ROAD-0001
**Status:** Active
**Last Updated:** 2026-01-17

---

## Overview

This roadmap tracks the implementation of the Basel Haskell Compiler (BHC) from proof-of-concept to H26-Numeric conformance. Each milestone has concrete exit criteria that must be met before proceeding.

---

## Milestone Summary

| Milestone | Name | Status |
|-----------|------|--------|
| M0 | Proof of Life | ✅ Complete |
| M1 | Numeric Profile Skeleton | ✅ Complete |
| M2 | Tensor IR v1 | ✅ Complete |
| M3 | Vectorization + Parallel Loops | ✅ Complete |
| M4 | Pinned Arrays + FFI | ✅ Complete |
| M5 | Server Runtime Contract | ✅ Complete |
| M6 | Platform Standardization | ✅ Complete |
| M7 | GPU Backend | ✅ Complete |
| M8 | WASM Target | ✅ Complete |
| M9 | Dependent Types Preview | ✅ Complete |
| M10 | Cargo-Quality Diagnostics | 🔄 In Progress |

---

## Completed Milestones

### M0 — Proof of Life ✅

**Goal:** Demonstrate basic compiler pipeline works end-to-end.

- [x] Lexer and parser for core Haskell subset
- [x] Minimal type checker (Hindley-Milner + basic typeclasses)
- [x] Core IR representation
- [x] Tree-walking interpreter for Core IR
- [x] Default Profile only (lazy semantics)
- [x] `UArray` prototype with `map`/`fold` semantics

---

### M1 — Numeric Profile Skeleton ✅

**Goal:** Implement strict-by-default evaluation and unboxed primitives.

- [x] Numeric Profile strict-by-default semantics
- [x] Unboxed primitive types: `I32`, `I64`, `F32`, `F64`
- [x] Unboxed `UArray` representation (flat contiguous)
- [x] Hot Arena allocator in RTS
- [x] `lazy { }` escape hatch syntax
- [x] Basic strictness analysis pass

---

### M2 — Tensor IR v1 ✅

**Goal:** Introduce Tensor type and guaranteed fusion.

- [x] `Tensor` type with shape/stride metadata
- [x] View operations: `reshape`, `slice`, `transpose`
- [x] Tensor IR representation
- [x] Lowering pass: recognized patterns → Tensor IR
- [x] Fusion pass for guaranteed patterns
- [x] Kernel report mode (`-fkernel-report`)

---

### M3 — Vectorization + Parallel Loops ✅

**Goal:** Auto-vectorization and parallel execution.

- [x] Loop IR representation (post-Tensor IR)
- [x] SIMD primitive types: `Vec4F32`, `Vec8F32`, `Vec2F64`, `Vec4F64`
- [x] Auto-vectorization pass
- [x] SIMD intrinsics: `add`, `mul`, `fmadd`, `hadd`
- [x] `parFor`, `parMap`, `parReduce` primitives
- [x] Work-stealing scheduler in RTS
- [x] Deterministic chunking mode

---

### M4 — Pinned Arrays + FFI ✅

**Goal:** Enable zero-copy FFI and external BLAS integration.

- [x] Pinned heap region in RTS
- [x] `PinnedUArray` type
- [x] Tensor buffers optionally pinned
- [x] Safe FFI boundary with pinned buffer support
- [x] BLAS provider interface
- [x] Reference OpenBLAS integration

---

### M5 — Server Runtime Contract ✅

**Goal:** Production-ready concurrent runtime.

- [x] Structured concurrency primitives
- [x] Cancellation propagation (cooperative)
- [x] Deadline/timeout support
- [x] Per-core scheduling
- [x] Incremental/concurrent GC
- [x] Event tracing hooks (GC, tasks, allocations)

---

### M6 — Platform Standardization ✅

**Goal:** Ship complete H26-Platform and conformance suite.

- [x] All H26 Platform modules implemented
- [x] Conformance test suite (semantic + runtime + benchmarks)
- [x] Package manifest schema
- [x] Lockfile format
- [x] Reproducible build verification
- [x] Published benchmark results

---

### M7 — GPU Backend ✅

**Goal:** GPU compute support for numeric workloads.

- [x] CUDA/ROCm code generation
- [x] Device memory management
- [x] Kernel fusion across host/device boundary

---

### M8 — WASM Target ✅

**Goal:** WebAssembly compilation for edge deployment.

- [x] WebAssembly code generation
- [x] Browser runtime
- [x] Edge profile optimization

---

### M9 — Dependent Types Preview ✅

**Goal:** Shape-indexed tensors with compile-time dimension checking.

- [x] Type-level naturals (`TyNat`) and lists (`TyList`)
- [x] Kind system extensions (`Kind::Nat`, `Kind::List`)
- [x] Promoted list syntax `'[1024, 768]`
- [x] Type families: `MatMulShape`, `Broadcast`, `Transpose`, `Concat`
- [x] Dynamic escape hatch: `DynTensor`, `toDynamic`, `fromDynamic`
- [x] Tensor IR bridge with shape verification
- [x] Comprehensive shape error messages

---

## Current Milestone

### M10 — Cargo-Quality Diagnostics 🔄

**Goal:** World-class compiler error messages on par with Rust/Cargo.

GHC is notorious for esoteric, hard-to-understand error messages. BHC aims to set a new standard for Haskell compiler diagnostics, taking inspiration from Rust's exemplary error reporting.

### Principles

1. **Human-first** — Errors written for humans, not compiler authors
2. **Actionable** — Every error suggests how to fix it
3. **Contextual** — Show relevant code with precise highlighting
4. **Educational** — Explain *why* something is wrong, not just *what*
5. **Progressive** — Simple errors get simple messages; complex errors get detailed explanations

### Deliverables

#### Phase 1: Diagnostic Infrastructure
- [ ] Rich source spans with multi-line support
- [ ] Colored terminal output with ASCII art
- [ ] Machine-readable JSON diagnostic format
- [ ] Diagnostic severity levels (error, warning, note, help)
- [ ] Error codes with `--explain E0001` lookup

#### Phase 2: Type Error Overhaul
- [ ] "Expected X, found Y" with source highlighting
- [ ] Type mismatch shows both types aligned for comparison
- [ ] Unification trail for complex type errors
- [ ] "Did you mean?" suggestions for typos
- [ ] Function arity mismatch with argument highlighting

#### Phase 3: Shape Error Excellence
- [ ] Matrix dimension mismatch with visual shape diagrams
- [ ] Broadcasting failure with axis-by-axis breakdown
- [ ] Shape variable unification explained step-by-step
- [ ] Tensor operation signatures shown inline

#### Phase 4: Contextual Help
- [ ] Related documentation links in errors
- [ ] Similar function suggestions
- [ ] Import suggestions for unresolved names
- [ ] Quick-fix suggestions with `--apply-suggestions`
- [ ] Common mistake patterns with explanations

#### Phase 5: IDE Integration
- [ ] Language Server Protocol (LSP) diagnostics
- [ ] Inline error rendering
- [ ] Code action quick fixes
- [ ] Hover information for error spans

### Example: Target Error Quality

**Before (GHC-style):**
```
Couldn't match type 'Tensor '[768, 512] Float'
                with 'Tensor '[1024, 768] Float'
Expected: Tensor '[m, k] Float -> Tensor '[k, n] Float -> Tensor '[m, n] Float
  Actual: Tensor '[1024, 768] Float -> Tensor '[768, 512] Float -> Tensor '[1024, 512] Float
```

**After (Cargo-style):**
```
error[E0030]: matrix multiplication dimension mismatch
  --> src/Model.hs:42:15
   |
42 |   let result = matmul weights input
   |                ^^^^^^ ^^^^^^^ ^^^^^ Tensor '[768, 512] Float
   |                |      |
   |                |      Tensor '[1024, 768] Float
   |                inner dimensions don't match
   |
   = note: matmul requires: Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a
   |
   |       weights: '[1024, 768]  (k = 768)
   |       input:   '[768, 512]   (k = 768) ✓
   |
   |       Wait, these DO match! Let me check again...
   |
   = help: the shapes '[1024, 768] @ '[768, 512]' produce '[1024, 512]'
   = help: did you mean to transpose the input?
   |
   |       let result = matmul weights (transpose input)
   |
```

### Exit Criteria

- [ ] All type errors include source location with highlighted span
- [ ] All errors have error codes with `--explain` documentation
- [ ] Shape errors show visual dimension breakdown
- [ ] 90% of errors include actionable suggestions
- [ ] LSP integration provides inline diagnostics
- [ ] User study: BHC errors rated clearer than GHC by 80%+ of participants

### Reference: Rust's Error Design

Key patterns from Rust to adopt:
- Primary span with `-->` indicator
- Secondary spans with `|` gutter
- Color coding: red for errors, yellow for warnings, cyan for notes, green for help
- Underlines (`^^^`) for precise error location
- `= note:` for additional context
- `= help:` for actionable suggestions
- Error codes (`E0001`) with `--explain` flag
- Machine-readable JSON with `--error-format=json`

### Key Files to Create/Modify

```
crates/bhc-diagnostics/src/
├── lib.rs              # Enhanced diagnostic types
├── render.rs           # Terminal rendering with colors
├── span.rs             # Multi-line span handling
├── suggest.rs          # Suggestion engine
├── explain.rs          # --explain documentation
└── json.rs             # JSON output format

crates/bhc-typeck/src/
├── diagnostics.rs      # Type error formatting (enhance existing)
├── suggest.rs          # Type-specific suggestions
└── explain/            # Error code explanations
    ├── E0001.md
    ├── E0020.md
    └── ...
```

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fusion complexity | High | Start with simple patterns, iterate |
| GC latency in Server | Medium | Incremental GC from M5 |
| SIMD portability | Medium | Abstract over target widths |
| FFI safety | High | Conservative defaults, unsafe escape hatch |
| Scope creep | High | Strict exit criteria per milestone |
| Error message quality | Medium | User testing, iterate based on feedback |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01 | Implementation language: Rust | Performance, safety, excellent tooling |
| 2026-01 | Diagnostic model: Rust-inspired | Best-in-class error UX |

---

## References

- [H26-SPEC-0001] Haskell 2026 Platform & Runtime Specification
- [BHC-SPEC-0001] Basel Haskell Compiler Specification
- [Rust Error Index](https://doc.rust-lang.org/error_codes/error-index.html) — Reference for error code system
- [Rust Diagnostic Guidelines](https://rustc-dev-guide.rust-lang.org/diagnostics.html) — Design principles
- See `CLAUDE.md` for project overview
- See `rules/` for coding guidelines
