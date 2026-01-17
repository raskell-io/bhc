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
| M10 | Cargo-Quality Diagnostics | ✅ Complete |
| M11 | Real-World Haskell Compatibility | 🔄 In Progress |

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

### M10 — Cargo-Quality Diagnostics ✅

**Goal:** World-class compiler error messages on par with Rust/Cargo.

- [x] Phase 1: Diagnostic Infrastructure (Cargo-style rendering, JSON output, error codes)
- [x] Phase 2: Type Error Overhaul ("Did you mean?" suggestions, type alignment)
- [x] Phase 3: Shape Error Excellence (Visual ASCII diagrams for tensor shape errors)
- [x] Phase 4: Contextual Help (Doc links, related codes, common mistakes)
- [x] Phase 5: IDE Integration (LSP diagnostics, code actions, hover info)

---

## Current Milestone

### M11 — Real-World Haskell Compatibility 🔄

**Goal:** Enable BHC to compile real-world Haskell projects like xmonad, pandoc, and lens.

Currently, BHC's parser requires explicit braces `{}` and doesn't support many standard Haskell features. This milestone focuses on achieving compatibility with existing Haskell codebases.

### Principles

1. **Compatibility** — Parse and compile code written for GHC
2. **Gradual adoption** — Existing code should "just work"
3. **Standards-first** — Follow Haskell 2010/GHC extensions where sensible
4. **Clear errors** — When features aren't supported, say so clearly

### Deliverables

#### Phase 1: LANGUAGE Pragmas
- [ ] Parse `{-# LANGUAGE ExtensionName #-}` at module level
- [ ] Parse `{-# OPTIONS_GHC ... #-}` (ignore for now)
- [ ] Parse `{-# INLINE/NOINLINE/SPECIALIZE #-}` pragmas
- [ ] Track enabled extensions per module
- [ ] Common extensions: `OverloadedStrings`, `LambdaCase`, `BangPatterns`, etc.

#### Phase 2: Layout Rule
- [ ] Implement Haskell 2010 layout rule (Section 10.3)
- [ ] Implicit `{`, `}`, `;` insertion based on indentation
- [ ] Handle `where`, `let`, `do`, `of`, `case` layout contexts
- [ ] Mixed explicit/implicit layout support
- [ ] Error recovery for layout mistakes

#### Phase 3: Module System
- [ ] Full export list syntax: `module Foo (bar, Baz(..), module X) where`
- [ ] Import declarations with all forms:
  - `import Foo`
  - `import Foo (bar, baz)`
  - `import Foo hiding (bar)`
  - `import qualified Foo`
  - `import qualified Foo as F`
  - `import Foo (Type(..), pattern Pat)`
- [ ] Qualified names: `Data.Map.lookup`
- [ ] Hierarchical module names

#### Phase 4: Declarations
- [ ] Type class declarations with methods and default implementations
- [ ] Instance declarations with method implementations
- [ ] `deriving` clauses (stock: Eq, Ord, Show, Read, Enum, Bounded)
- [ ] Standalone deriving: `deriving instance Eq Foo`
- [ ] GADT syntax for data declarations
- [ ] Pattern synonyms: `pattern P x = ...`
- [ ] Foreign declarations: `foreign import/export`

#### Phase 5: Patterns & Expressions
- [ ] Pattern guards: `f x | Just y <- g x = ...`
- [ ] View patterns: `f (view -> pat) = ...`
- [ ] As-patterns: `f xs@(x:_) = ...`
- [ ] Record patterns: `f Foo{bar=x} = ...`
- [ ] Infix constructor patterns: `f (x:xs) = ...`
- [ ] Where clauses in function definitions
- [ ] Multi-way if: `if | cond1 -> e1 | cond2 -> e2`
- [ ] Lambda-case: `\case Pat1 -> e1; Pat2 -> e2`
- [ ] Typed holes: `_ :: Type`

#### Phase 6: Types
- [ ] `forall` quantification: `forall a. a -> a`
- [ ] Scoped type variables
- [ ] Type applications: `f @Int x`
- [ ] Kind signatures: `data Proxy (a :: k) = Proxy`
- [ ] Type families: `type family F a where ...`
- [ ] Associated type families in classes
- [ ] Constraint kinds: `(Show a, Eq a) => ...`

### Test Strategy

Each phase will be validated against real codebases:

| Phase | Test Target |
|-------|-------------|
| 1-2 | Parse xmonad's StackSet.hs without errors |
| 3 | Resolve imports in a multi-module project |
| 4 | Compile base library's Data.List |
| 5-6 | Parse lens library type signatures |

### Exit Criteria

- [ ] `bhc check` succeeds on xmonad source files
- [ ] `bhc check` succeeds on pandoc source files (excluding Template Haskell)
- [ ] All Haskell 2010 Report features supported
- [ ] Common GHC extensions parsed (even if semantics simplified)
- [ ] Clear error messages for unsupported extensions

### Key Files to Modify

```
crates/bhc-lexer/src/
├── lib.rs              # Layout token insertion
├── layout.rs           # NEW: Layout rule implementation

crates/bhc-parser/src/
├── lib.rs              # Module-level pragma parsing
├── decl.rs             # Class/instance/GADT declarations
├── expr.rs             # Pattern guards, lambda-case
├── pattern.rs          # As-patterns, view patterns
├── types.rs            # forall, type applications
├── pragma.rs           # NEW: Pragma parsing
├── module.rs           # NEW: Import/export parsing

crates/bhc-ast/src/
├── lib.rs              # New AST nodes for extensions
├── extension.rs        # NEW: Extension flag tracking
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
