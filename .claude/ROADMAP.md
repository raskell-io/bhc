# BHC Roadmap

**Document ID:** BHC-ROAD-0001
**Status:** Active
**Last Updated:** 2026-02-21

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

**Progress:** 163 E2E tests, 65 milestones (E.1–E.65), 30+ GHC extensions implemented.
BHC now compiles non-trivial Haskell programs with records, GADTs, typeclasses,
deriving, monad transformers, and most common GHC extensions. The focus has shifted
from basic syntax support to the remaining infrastructure for Pandoc compilation.

See `TODO-pandoc.md` for the detailed Pandoc compilation roadmap.

### Principles

1. **Compatibility** — Parse and compile code written for GHC
2. **Gradual adoption** — Existing code should "just work"
3. **Standards-first** — Follow Haskell 2010/GHC extensions where sensible
4. **Clear errors** — When features aren't supported, say so clearly

### Deliverables

#### Phase 1: LANGUAGE Pragmas ✅
- [x] Parse `{-# LANGUAGE ExtensionName #-}` at module level
- [x] Parse `{-# OPTIONS_GHC ... #-}`
- [x] Track enabled extensions per module
- [x] 30+ extensions supported: OverloadedStrings, LambdaCase, BangPatterns, GADTs, ScopedTypeVariables, FlexibleInstances, FlexibleContexts, MultiParamTypeClasses, FunctionalDependencies, GeneralizedNewtypeDeriving, DeriveGeneric, DeriveFunctor, DeriveFoldable, DeriveTraversable, DeriveAnyClass, StandaloneDeriving, TypeOperators, PatternSynonyms, ViewPatterns, TupleSections, MultiWayIf, RecordWildCards, NamedFieldPuns, EmptyDataDecls, EmptyCase, StrictData, DefaultSignatures, OverloadedLists, InstanceSigs, etc.
- [ ] Parse `{-# INLINE/NOINLINE/SPECIALIZE #-}` pragmas (parsed but not used by optimizer)

#### Phase 2: Layout Rule ✅
- [x] Implement Haskell 2010 layout rule (Section 10.3)
- [x] Implicit `{`, `}`, `;` insertion based on indentation
- [x] Handle `where`, `let`, `do`, `of`, `case` layout contexts
- [x] Mixed explicit/implicit layout support
- [x] Multi-line type signatures (continuation tokens prevent spurious VirtualSemi)
- [x] Guard syntax (`|`) handled as continuation
- [x] Multi-level dedent generates correct VirtualRBrace sequence
- [x] Nested layout blocks (where inside where, do inside do, let inside do)
- [x] `\case` (LambdaCase) layout detection
- [x] EOF cleanup closes remaining implicit blocks
- [ ] Error recovery for layout mistakes

**Note:** The layout rule is fully implemented in the lexer (~300 lines in `bhc-lexer/src/lib.rs`)
with VirtualLBrace/VirtualRBrace/VirtualSemi token insertion. All 163 E2E tests use
indentation-based layout (only 3 of 163 use explicit braces). Verified with 39 lexer unit
tests including edge cases and a comprehensive layout-focused E2E test (E.65).

#### Phase 3: Module System — Mostly Complete
- [x] Import declarations with common forms (import, qualified, as, hiding)
- [x] Qualified names: `Data.Map.lookup`
- [x] Hierarchical module names
- [x] Multi-module compilation with dependency ordering (E.6)
- [ ] Full export list syntax: `module Foo (bar, Baz(..), module X) where`
- [ ] `import Foo (Type(..), pattern Pat)` — pattern import syntax

#### Phase 4: Declarations ✅
- [x] Type class declarations with methods and default implementations (E.39–E.41)
- [x] Instance declarations with method implementations (E.38)
- [x] `deriving` clauses: Eq, Ord, Show, Enum, Bounded, Functor, Foldable, Traversable, Generic (E.23–E.24, E.51–E.54, E.63)
- [x] Standalone deriving: `deriving instance Eq Foo` (E.62)
- [x] GADT syntax for data declarations (E.60)
- [x] Pattern synonyms: `pattern P x = ...` (E.62)
- [ ] Foreign declarations: `foreign import/export`

#### Phase 5: Patterns & Expressions ✅
- [x] Pattern guards: `f x | Just y <- g x = ...`
- [x] View patterns: `f (view -> pat) = ...` (E.34)
- [x] As-patterns: `f xs@(x:_) = ...`
- [x] Record patterns: `f Foo{bar=x} = ...` (E.33)
- [x] Infix constructor patterns: `f (x:xs) = ...`
- [x] Where clauses in function definitions
- [x] Multi-way if: `if | cond1 -> e1 | cond2 -> e2` (E.35)
- [x] Lambda-case: `\case Pat1 -> e1; Pat2 -> e2`
- [ ] Typed holes: `_ :: Type`

#### Phase 6: Types — Mostly Complete
- [x] `forall` quantification: `forall a. a -> a`
- [x] Scoped type variables (E.46)
- [x] Kind signatures
- [x] Constraint kinds: `(Show a, Eq a) => ...`
- [x] MultiParamTypeClasses (E.49)
- [x] FunctionalDependencies (E.50)
- [x] FlexibleInstances, FlexibleContexts (E.48)
- [x] TypeOperators (E.61)
- [ ] Type applications: `f @Int x`
- [ ] Type families: `type family F a where ...`
- [ ] Associated type families in classes

#### Phase 7: Core IR Optimization Pipeline

BHC currently has no general-purpose optimizer — this is the most critical
infrastructure gap for compiling real Haskell programs. Design informed by
HBC's (Lennart Augustsson) proven simplifier and analysis passes.

- [ ] **Core Simplifier** — iterate-to-fixpoint pass with:
  - Beta reduction
  - Case-of-known-constructor
  - Dead binding elimination
  - Constant folding
  - Inlining (reference-counting + size threshold)
  - Case-of-case (with size budget)
- [ ] **Pattern Match Compilation** — replace naive equation-by-equation with:
  - Column-based decision tree generation (Augustsson algorithm)
  - Exhaustiveness checking and warnings
  - Overlap/redundancy detection and warnings
- [ ] **Demand Analysis** — per-function strictness (Default profile):
  - Boolean-tree abstract interpretation
  - Fixpoint iteration for recursive groups
  - Worker/wrapper transformation for strict arguments
- [ ] **Dictionary Specialization** — monomorphize typeclass-polymorphic code:
  - Direct method selection on known dictionaries
  - SPECIALIZE pragma support

See `rules/013-optimization.md` for detailed design.

### Test Strategy

Each phase will be validated against real codebases:

| Phase | Test Target | Status |
|-------|-------------|--------|
| 1 | Parse `{-# LANGUAGE ... #-}` pragmas | ✅ 30+ extensions |
| 2 | Parse xmonad's StackSet.hs without errors | Layout rule ✅, needs package system |
| 3 | Resolve imports in a multi-module project | ✅ Working (E.6) |
| 4 | Compile programs with classes, instances, GADTs | ✅ Working (E.38–E.64) |
| 5 | Parse lens library type signatures | Blocked on type families |
| 6 | Full type system coverage | Mostly done, type families remaining |
| 7 | Compiled programs run measurably faster | Not started |

### Exit Criteria

- [ ] `bhc check` succeeds on xmonad source files (needs package system)
- [ ] `bhc check` succeeds on pandoc source files (needs package system, CPP)
- [x] Common GHC extensions parsed and implemented (30+ extensions)
- [ ] All Haskell 2010 Report features supported (layout rule ✅, foreign decls remaining)
- [ ] Clear error messages for unsupported extensions
- [ ] Core simplifier reduces code size by ≥20% on test programs
- [ ] Non-exhaustive pattern matches produce compiler warnings

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

## Known Issues

### Phase 2: Closure Hardening — RESOLVED

**Found:** 2026-01-30 during blog post compiler testing
**Resolved:** 2026-01-30

The reported nested closure invocation segfault (exit code 139) has been resolved. All previously crashing programs now compile and run correctly:

```haskell
-- Previously crashed, now works correctly (output: 20)
twice f x = f (f x)
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)
main = print (twice (\x -> x * 2) (fib 5))
```

**Root cause:** The issue was fixed in prior codegen/parser work. Diagnostic testing confirmed all three features work both in isolation and combined:
- Closure argument used multiple times (`twice f x = f (f x)`) — works
- Multi-clause function definitions with pattern matching (`fib 0 = 0; fib 1 = 1; fib n = ...`) — works
- Combined program with both features — works

**Regression tests added:**
- `examples/twice.hs` — Higher-order function calling closure argument twice
- `examples/pattern-match-fib.hs` — Multi-clause function definition with literal patterns

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
