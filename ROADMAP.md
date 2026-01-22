# BHC Implementation Roadmap

This document provides a detailed implementation plan to deliver all features promised on [bhc.raskell.io](https://bhc.raskell.io).

## Current Status

**Alpha** — Native compilation works! The compiler can build and run simple Haskell programs.

| Component | Status | Notes |
|-----------|--------|-------|
| Parser/Lexer | ✅ Working | ~8,000 LOC, robust |
| Type Checker | ✅ Working | ~10,000 LOC, inference works |
| HIR Lowering | ✅ Working | ~6,000 LOC |
| Core IR | ✅ Working | Tree-walking interpreter + LLVM codegen |
| Tensor IR | 🟡 Partial | Types and basic passes |
| Loop IR | 🟡 Partial | Types and basic passes |
| Native Codegen | ✅ Working | LLVM backend via inkwell |
| WASM Codegen | 🟡 Partial | ~3,300 LOC, incomplete |
| GPU Codegen | 🔴 Skeleton | ~880 LOC stubs |
| Runtime | ✅ Working | Mark-sweep GC, basic IO primitives |

---

## Phase 1: Native Hello World ✅ COMPLETE

**Objective:** `bhc Main.hs -o main && ./main` prints "Hello, World!"

This is the critical path. Everything else depends on native code generation working.

### 1.1 LLVM Integration ✅

**Crate:** `bhc-codegen`
**Dependency:** [inkwell](https://crates.io/crates/inkwell) (safe LLVM bindings)

Tasks:
- [x] Add `inkwell` dependency to `bhc-codegen/Cargo.toml`
- [x] Create `LlvmContext` wrapping inkwell's `Context`
- [x] Create `LlvmModule` wrapping inkwell's `Module`
- [x] Implement `CodegenBackend` trait with real LLVM operations
- [x] Remove placeholder implementations
- [x] Add target triple detection from `bhc-target`
- [x] Test: Create and verify a simple LLVM module

### 1.2 Core IR to LLVM ✅

**Crate:** `bhc-codegen`

Tasks:
- [x] Define LLVM type mappings for Core IR types
  - `Int` → `i64`
  - `Bool` → `i1`
  - `Char` → `i32` (Unicode codepoint)
  - Boxed types → pointer to heap object
- [x] Implement `Lit` (literal) codegen
  - Integer literals
  - Character literals
  - String literals (pointer to static data)
- [x] Implement `Var` (variable) codegen
  - Local variables → LLVM alloca/load
  - Global variables → LLVM global references
- [x] Implement `App` (application) codegen
  - Direct function calls
  - Saturated applications
- [x] Implement `Lam` (lambda) codegen
  - Simple non-capturing lambdas as functions
  - Defer closures to Phase 2
- [x] Implement `Let` codegen
  - Non-recursive let bindings
  - Defer recursive bindings to Phase 2
- [x] Implement `Case` codegen (basic)
  - Pattern matching on Int
  - Pattern matching on Bool
  - Defer complex patterns to Phase 2
- [x] Test: Compile `main = 1 + 2` to working executable

### 1.3 Minimal Runtime System ✅

**Crate:** `bhc-rts`

Tasks:
- [x] Define object header layout
- [x] Define info table structure
- [x] Implement `bhc_alloc(size: usize) -> *mut u8`
- [x] Implement `bhc_init()` - runtime initialization
- [x] Implement `bhc_exit(code: i32)` - clean shutdown
- [x] Create RTS static library for linking
- [x] Test: Link a trivial program with RTS

### 1.4 Basic GC ✅

**Crate:** `bhc-rts-gc`

Tasks:
- [x] Implement root set tracking
- [x] Implement mark phase
- [x] Implement sweep phase
- [x] Add GC trigger (allocation threshold)
- [x] Test: Allocate objects, trigger GC, verify live objects survive

### 1.5 Linker Integration ✅

**Crate:** `bhc-linker`

Tasks:
- [x] Detect system linker (ld, lld, link.exe)
- [x] Generate object file from LLVM module
- [x] Link object file with RTS library
- [x] Handle platform-specific linking flags
  - macOS: `-lSystem`
  - Linux: `-lc -ldl -lpthread`
  - Windows: kernel32, etc.
- [x] Test: Full compile-link pipeline produces working executable

### 1.6 IO Primitives ✅

**Crate:** `bhc-rts`

Tasks:
- [x] Implement `bhc_print_int_ln(i: i64)` - print integer with newline
- [x] Implement `bhc_print_double_ln(d: f64)` - print double with newline
- [x] Implement `bhc_print_string_ln(ptr: *const u8, len: usize)` - print string with newline
- [x] Wire up Haskell `print` to RTS functions
- [x] Test: `main = print 42` program works

### Phase 1 Exit Criteria ✅

```bash
$ cat Main.hs
main = print 42

$ bhc run Main.hs
42
```

**Completed!**

---

## Phase 2: Language Completeness

**Objective:** Compile and run real Haskell programs.

### 2.1 Pattern Matching Codegen

**Crate:** `bhc-codegen`

Tasks:
- [ ] Implement constructor pattern matching
  - Match on data constructor tag
  - Extract fields
- [ ] Implement nested patterns
  - Flatten to decision tree
- [ ] Implement guards
- [ ] Implement as-patterns (`x@(Cons a b)`)
- [ ] Implement wildcard patterns
- [ ] Implement literal patterns
- [ ] Test: Pattern matching on Maybe, Either, lists

**Estimated effort:** 1 week

### 2.2 Closures

**Crate:** `bhc-codegen`, `bhc-rts`

Tasks:
- [ ] Define closure object layout
  ```
  | Header | Code Ptr | Free Var 1 | Free Var 2 | ... |
  ```
- [ ] Implement closure allocation
- [ ] Implement closure entry code (loads free vars, jumps to body)
- [ ] Implement free variable analysis in Core
- [ ] Generate closure-creating code for lambdas
- [ ] Test: Higher-order functions (`map`, `filter`)

**Estimated effort:** 1-2 weeks

### 2.3 Thunks & Laziness

**Crate:** `bhc-rts`, `bhc-codegen`

Tasks:
- [ ] Define thunk object layout
  ```
  | Header | Code Ptr | Payload... |
  ```
- [ ] Implement thunk evaluation (`force`)
  - Check if already evaluated (indirection)
  - Push update frame
  - Enter thunk code
  - Update with result
- [ ] Implement update frames
- [ ] Implement indirection handling
- [ ] Generate thunk-creating code for lazy bindings
- [ ] Test: Lazy infinite list `[1..]`

**Estimated effort:** 1-2 weeks

### 2.4 Type Classes

**Crate:** `bhc-typeck`, `bhc-codegen`

Tasks:
- [ ] Implement dictionary representation
  ```
  data EqDict a = EqDict { eq :: a -> a -> Bool, neq :: a -> a -> Bool }
  ```
- [ ] Implement dictionary passing for overloaded functions
- [ ] Implement dictionary construction for instances
- [ ] Handle superclass constraints
- [ ] Test: `Eq`, `Ord`, `Show` instances

**Estimated effort:** 1-2 weeks

### 2.5 Let/Where Bindings

**Crate:** `bhc-codegen`

Tasks:
- [ ] Implement non-recursive let (already partial)
- [ ] Implement recursive let (letrec)
  - Allocate thunks first
  - Backpatch references
- [ ] Implement where clauses (desugar to let)
- [ ] Test: Mutual recursion in let

**Estimated effort:** 3-5 days

### 2.6 Recursion & Tail Calls

**Crate:** `bhc-codegen`

Tasks:
- [ ] Detect tail call positions
- [ ] Implement tail call optimization (jump instead of call)
- [ ] Implement self-recursive tail calls as loops
- [ ] Test: `factorial 1000000` without stack overflow

**Estimated effort:** 3-5 days

### 2.7 Prelude Bootstrap

**Crate:** `stdlib`

Tasks:
- [ ] Compile basic list functions (`map`, `filter`, `foldr`, `foldl`)
- [ ] Compile Maybe functions
- [ ] Compile Either functions
- [ ] Compile basic numeric operations
- [ ] Create prelude interface file
- [ ] Test: Compile program using Prelude

**Estimated effort:** 1-2 weeks

### Phase 2 Exit Criteria

```haskell
-- Fibonacci
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n-1) + fib (n-2)

main = print (fib 30)
```

Compiles and runs correctly.

**Total estimated effort:** 6-10 weeks

---

## Phase 3: Numeric Profile

**Objective:** Deliver promised numeric performance: fusion, SIMD, parallelism.

### 3.1 Core to Tensor IR Lowering

**Crate:** `bhc-tensor-ir`

Tasks:
- [ ] Identify numeric operations in Core IR
- [ ] Lower array/vector operations to Tensor IR
- [ ] Track shape information
- [ ] Track element types
- [ ] Test: Lower `map (*2) xs` to TensorMap

### 3.2 Fusion Implementation

**Crate:** `bhc-tensor-ir`

Tasks:
- [ ] Implement vertical fusion (map-map)
- [ ] Implement horizontal fusion (zipWith)
- [ ] Implement reduction fusion (fold-map)
- [ ] Add fusion verification pass
- [ ] Generate fusion report (`--kernel-report`)
- [ ] Test: Verify 4 guaranteed patterns fuse

### 3.3 Tensor to Loop IR

**Crate:** `bhc-loop-ir`

Tasks:
- [ ] Generate explicit loop nests from Tensor ops
- [ ] Track loop bounds from shapes
- [ ] Generate index calculations from strides
- [ ] Handle broadcasting
- [ ] Test: TensorMap becomes `for` loop

### 3.4 SIMD Vectorization

**Crate:** `bhc-loop-ir`

Tasks:
- [ ] Identify vectorizable loops
- [ ] Compute vector width from target
- [ ] Generate vector types (f32x4, f64x4, etc.)
- [ ] Generate vector load/store
- [ ] Generate vector operations
- [ ] Handle loop remainders
- [ ] Test: `sum xs` uses SIMD

### 3.5 Parallel Loop Codegen

**Crate:** `bhc-loop-ir`

Tasks:
- [ ] Identify parallelizable loops (no dependencies)
- [ ] Generate parallel loop structure
- [ ] Integrate with RTS thread pool
- [ ] Handle reduction across threads
- [ ] Test: Parallel map scales with cores

### 3.6 Loop IR to LLVM

**Crate:** `bhc-codegen`

Tasks:
- [ ] Lower Loop IR to LLVM IR
- [ ] Emit LLVM vector intrinsics
- [ ] Use LLVM's loop optimizations
- [ ] Test: Generated assembly contains SIMD

### 3.7 Hot Arena Allocator

**Crate:** `bhc-rts`

Tasks:
- [ ] Implement arena allocation (`bhc_arena_alloc`)
- [ ] Implement bulk free (`bhc_arena_reset`)
- [ ] Generate arena scope markers in codegen
- [ ] Test: Kernel temporaries use arena

### 3.8 Pinned Buffers

**Crate:** `bhc-rts`

Tasks:
- [ ] Implement pinned allocation (`bhc_alloc_pinned`)
- [ ] Track pinned objects separately from GC heap
- [ ] Implement reference counting for pinned
- [ ] Test: FFI buffer survives GC

### Phase 3 Exit Criteria

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
main = print $ sum $ map (*2) [1..10000000]
```

- Runs 10x+ faster than interpreted
- `--kernel-report` shows fusion occurred
- Generated code uses SIMD

**Total estimated effort:** 8-12 weeks

---

## Phase 4: WASM Backend

**Objective:** `bhc --target=wasi Main.hs` produces working WebAssembly.

### 4.1 WASM Emitter

**Crate:** `bhc-wasm`

Tasks:
- [ ] Emit WASM binary format
- [ ] Map types to WASM types (i32, i64, f32, f64)
- [ ] Generate WASM functions from Core IR
- [ ] Handle indirect calls for closures
- [ ] Test: Simple function compiles to valid WASM

### 4.2 WASI Runtime Integration

**Crate:** `bhc-wasm`

Tasks:
- [ ] Import WASI functions (fd_write, fd_read, etc.)
- [ ] Map IO primitives to WASI calls
- [ ] Handle command-line arguments
- [ ] Handle environment variables
- [ ] Test: putStrLn works in WASM

### 4.3 Edge Profile RTS

**Crate:** `bhc-rts`

Tasks:
- [ ] Create minimal RTS variant for WASM
- [ ] Use linear memory for heap
- [ ] Implement GC within linear memory
- [ ] Minimize code size
- [ ] Test: Runtime < 100KB

### Phase 4 Exit Criteria

```bash
$ bhc --target=wasi Main.hs -o app.wasm
$ wasmtime app.wasm
Hello, World!
```

**Total estimated effort:** 4-6 weeks

---

## Phase 5: Server Profile

**Objective:** Structured concurrency with proper cancellation.

### 5.1 Task Scheduler

**Crate:** `bhc-rts`

Tasks:
- [ ] Implement work-stealing deque
- [ ] Implement worker threads
- [ ] Implement task spawning
- [ ] Implement task completion
- [ ] Test: Basic parallel task execution

### 5.2 Scope & Task Primitives

**Crate:** `bhc-concurrent`

Tasks:
- [ ] Implement `Scope` type
- [ ] Implement `withScope`
- [ ] Implement `spawn`
- [ ] Implement `await`
- [ ] Test: Concurrent tasks complete within scope

### 5.3 Cancellation

**Crate:** `bhc-concurrent`

Tasks:
- [ ] Implement cancellation tokens
- [ ] Implement `cancel`
- [ ] Implement cancellation propagation
- [ ] Implement cancellation checking
- [ ] Test: Cancelled task stops

### 5.4 STM

**Crate:** `bhc-concurrent`

Tasks:
- [ ] Implement `TVar` type
- [ ] Implement `atomically`
- [ ] Implement `retry`
- [ ] Implement `orElse`
- [ ] Implement conflict detection
- [ ] Test: Bank transfer example

### 5.5 Deadlines

**Crate:** `bhc-concurrent`

Tasks:
- [ ] Implement `withDeadline`
- [ ] Implement deadline propagation
- [ ] Test: Operation times out

### Phase 5 Exit Criteria

```haskell
main = withScope $ \scope -> do
  t1 <- spawn scope $ do
    threadDelay 1000000
    return 1
  t2 <- spawn scope $ do
    threadDelay 500000
    return 2
  r1 <- await t1
  r2 <- await t2
  print (r1 + r2)
```

Works with proper task management.

**Total estimated effort:** 6-8 weeks

---

## Phase 6: GPU Backend

**Objective:** Tensor operations run on GPU.

### 6.1 PTX Codegen

**Crate:** `bhc-gpu`

Tasks:
- [ ] Generate PTX from Tensor IR
- [ ] Map operations to CUDA intrinsics
- [ ] Handle thread indexing
- [ ] Test: Simple kernel compiles

### 6.2 AMDGCN Codegen

**Crate:** `bhc-gpu`

Tasks:
- [ ] Generate AMDGCN from Tensor IR
- [ ] Map operations to ROCm intrinsics
- [ ] Handle wavefront indexing
- [ ] Test: Simple kernel compiles

### 6.3 Device Memory Management

**Crate:** `bhc-gpu`

Tasks:
- [ ] Implement device allocation
- [ ] Implement host-device transfer
- [ ] Implement automatic transfer insertion
- [ ] Test: Data flows to/from GPU

### 6.4 Kernel Launch

**Crate:** `bhc-gpu`

Tasks:
- [ ] Implement kernel compilation (runtime PTX/AMDGCN)
- [ ] Implement kernel invocation
- [ ] Handle grid/block dimensions
- [ ] Test: Kernel executes on GPU

### Phase 6 Exit Criteria

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
main = do
  let a = fromList [1..1000000]
  let b = fromList [1..1000000]
  print $ sum $ zipWith (+) a b
```

Runs on GPU when `--target=cuda`.

**Total estimated effort:** 8-12 weeks

---

## Phase 7: Advanced Profiles

### 7.1 Realtime (Bounded GC)

Tasks:
- [ ] Implement incremental mark
- [ ] Implement concurrent sweep
- [ ] Implement pause time budgeting
- [ ] Test: GC pauses < 1ms

### 7.2 Embedded (No GC)

Tasks:
- [ ] Implement escape analysis
- [ ] Implement static allocation
- [ ] Reject programs requiring GC at compile time
- [ ] Test: Bare-metal program

**Total estimated effort:** 6-10 weeks

---

## Phase 8: Ecosystem

### 8.1 REPL

Tasks:
- [ ] Interactive evaluation loop
- [ ] Expression type inference
- [ ] Value pretty-printing
- [ ] Test: `:t map` shows type

### 8.2 Package Manager

Tasks:
- [ ] Package description format
- [ ] Dependency resolution
- [ ] Build orchestration
- [ ] Registry integration

### 8.3 LSP Server

Tasks:
- [ ] Diagnostics
- [ ] Go to definition
- [ ] Hover information
- [ ] Completions

**Total estimated effort:** 8-12 weeks

---

## Summary Timeline

| Phase | Description | Effort | Cumulative |
|-------|-------------|--------|------------|
| 1 | Native Hello World | 4-6 weeks | 4-6 weeks |
| 2 | Language Completeness | 6-10 weeks | 10-16 weeks |
| 3 | Numeric Profile | 8-12 weeks | 18-28 weeks |
| 4 | WASM Backend | 4-6 weeks | 22-34 weeks |
| 5 | Server Profile | 6-8 weeks | 28-42 weeks |
| 6 | GPU Backend | 8-12 weeks | 36-54 weeks |
| 7 | Advanced Profiles | 6-10 weeks | 42-64 weeks |
| 8 | Ecosystem | 8-12 weeks | 50-76 weeks |

**Total: 12-18 months for full feature parity with website claims.**

---

## Immediate Next Steps

Phase 1 is complete! The compiler can now build and run native executables.

1. **Pattern matching codegen** - Extend Case to handle data constructors
2. **Closures** - Implement closure allocation and capture
3. **Thunks & Laziness** - Support lazy evaluation properly
4. **Type class dictionaries** - Dictionary-passing for Eq, Ord, Show, etc.

The goal is to compile real Haskell programs like Fibonacci or list operations.
