# BHC Implementation Roadmap

This document provides a detailed implementation plan to deliver all features promised on [bhc.raskell.io](https://bhc.raskell.io).

## Current Status

**Beta** — The compiler is feature-complete for core language functionality. Native compilation, numeric optimization, and developer tooling all work.

| Component | Status | Notes |
|-----------|--------|-------|
| Parser/Lexer | ✅ Complete | ~8,000 LOC, robust |
| Type Checker | ✅ Complete | ~10,000 LOC, inference + type classes |
| HIR Lowering | ✅ Complete | ~6,000 LOC |
| Core IR | ✅ Complete | Interpreter + LLVM codegen |
| Tensor IR | ✅ Complete | Lowering, fusion, all 4 patterns |
| Loop IR | ✅ Complete | Vectorization, parallelization |
| Native Codegen | ✅ Complete | LLVM backend, 8,178 LOC in lower.rs |
| WASM Codegen | 🟡 95% | Emitter + WASI + GC + driver complete, testing blocked by LLVM |
| GPU Codegen | 🟡 80% | PTX/AMDGCN loop nest codegen complete |
| Runtime | ✅ Complete | Generational GC, incremental GC, arena, scheduler |
| REPL (bhci) | ✅ Complete | Interactive evaluation |
| Package Manager | ✅ Complete | Dependency resolution, registry |
| LSP Server | ✅ Complete | Diagnostics, go-to-def, hover, completions |
| Documentation | ✅ Complete | User guide, language reference, examples |

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
- [x] Implement `Lit` (literal) codegen
- [x] Implement `Var` (variable) codegen
- [x] Implement `App` (application) codegen
- [x] Implement `Lam` (lambda) codegen
- [x] Implement `Let` codegen
- [x] Implement `Case` codegen (basic)
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

## Phase 2: Language Completeness ✅ COMPLETE

**Objective:** Compile and run real Haskell programs.

### 2.1 Pattern Matching Codegen ✅

**Crate:** `bhc-codegen`
**Location:** `lower.rs` lines 6942-7850

Tasks:
- [x] Implement constructor pattern matching (`lower_case_datacon()`)
- [x] Implement nested patterns (field extraction, decision trees)
- [x] Implement guards
- [x] Implement as-patterns (`x@(Cons a b)`)
- [x] Implement wildcard patterns
- [x] Implement literal patterns (`lower_case_literal_int/float/string()`)
- [x] Test: Pattern matching on Maybe, Either, lists

### 2.2 Closures ✅

**Crate:** `bhc-codegen`, `bhc-rts`
**Location:** `lower.rs` lines 4147-5100

Tasks:
- [x] Define closure object layout (`{ fn_ptr, env_size, env[] }`)
- [x] Implement closure allocation (`alloc_closure()`)
- [x] Implement closure entry code
- [x] Implement free variable analysis (`free_vars()`, `collect_free_vars()`)
- [x] Generate closure-creating code for lambdas (`lower_lambda()`)
- [x] Test: Higher-order functions (`map`, `filter`)

### 2.3 Thunks & Laziness ✅

**Crate:** `bhc-rts`, `bhc-codegen`
**Location:** `lower.rs` lines 4426-4584

Tasks:
- [x] Define thunk object layout (`{ tag, eval_fn, env_size, env[] }`)
- [x] Implement thunk evaluation (`build_force()` → `bhc_force()`)
- [x] Implement thunk tag checking (`bhc_is_thunk()`)
- [x] Implement indirection handling
- [x] Generate thunk-creating code (`alloc_thunk()`, `lower_lazy()`)
- [x] Test: Lazy infinite list `[1..]`

### 2.4 Type Classes ✅

**Crate:** `bhc-typeck`, `bhc-codegen`
**Location:** `context.rs` lines 206-327, `env.rs` lines 287-306

Tasks:
- [x] Implement instance resolution algorithm
- [x] Implement dictionary passing via field selectors (`$sel_N`)
- [x] Implement dictionary construction for instances
- [x] Handle superclass constraints (e.g., `Ord a` implies `Eq a`)
- [x] Test: `Eq`, `Ord`, `Show` instances for primitives

### 2.5 Let/Where Bindings ✅

**Crate:** `bhc-codegen`
**Location:** `lower.rs` lines 6620-6700

Tasks:
- [x] Implement non-recursive let
- [x] Implement recursive let (letrec) - lifted to top-level functions
- [x] Implement where clauses (desugar to let)
- [x] Test: Mutual recursion in let

### 2.6 Recursion & Tail Calls ✅

**Crate:** `bhc-codegen`
**Location:** `lower.rs` lines 96-112

Tasks:
- [x] Detect tail call positions (`in_tail_position` tracking)
- [x] Implement tail call optimization (`call.set_tail_call(true)`)
- [x] Implement self-recursive tail calls
- [x] Test: `factorial 1000000` without stack overflow

### 2.7 Prelude Bootstrap ✅

**Crate:** `stdlib/bhc-prelude`
**Location:** `hs/BHC/Prelude.hs` (650+ lines)

Tasks:
- [x] Compile basic list functions (`map`, `filter`, `foldr`, `foldl`, 30+ functions)
- [x] Compile Maybe/Either functions
- [x] Compile numeric operations (100+ FFI primitives)
- [x] Implement 26 type classes (Eq, Ord, Num, Functor, Monad, etc.)
- [x] Test: Compile program using Prelude

### Phase 2 Exit Criteria ✅

```haskell
-- Fibonacci
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n-1) + fib (n-2)

main = print (fib 30)
```

**Completed!** Commit `745bbac` (Jan 26, 2026)

---

## Phase 3: Numeric Profile ✅ COMPLETE

**Objective:** Deliver promised numeric performance: fusion, SIMD, parallelism.

### 3.1 Core to Tensor IR Lowering ✅

**Crate:** `bhc-tensor-ir`
**Location:** `lower.rs` (1,259 lines)

Tasks:
- [x] Identify numeric operations via `BuiltinTable` (12+ operations)
- [x] Lower array/vector operations to Tensor IR
- [x] Track shape information via `TensorMeta`
- [x] Track element types
- [x] Test: Lower `map (*2) xs` to TensorMap

### 3.2 Fusion Implementation ✅

**Crate:** `bhc-tensor-ir`
**Location:** `fusion.rs` (2,715 lines)

Tasks:
- [x] Implement Pattern 1: `map f (map g x)` → single traversal (MapMap)
- [x] Implement Pattern 2: `zipWith f (map g a) (map h b)` → single traversal (ZipWithMaps)
- [x] Implement Pattern 3: `sum (map f x)` → single traversal (ReduceMap)
- [x] Implement Pattern 4: `foldl' op z (map f x)` → single traversal (FoldMap)
- [x] Add fusion verification via reference counting
- [x] Generate fusion report (`generate_kernel_report()` line 1823)
- [x] Test: Verify all 4 guaranteed patterns fuse

### 3.3 Tensor to Loop IR ✅

**Crate:** `bhc-loop-ir`
**Location:** `lower.rs` (500+ lines)

Tasks:
- [x] Generate explicit loop nests from Tensor ops (`lower_kernel()`)
- [x] Track loop bounds from shapes
- [x] Generate index calculations from strides
- [x] Handle broadcasting
- [x] Test: TensorMap becomes `for` loop

### 3.4 SIMD Vectorization ✅

**Crate:** `bhc-loop-ir`
**Location:** `vectorize.rs` (600+ lines)

Tasks:
- [x] Identify vectorizable loops (`VectorizePass`)
- [x] Compute vector width from target (`VectorizeConfig`)
- [x] Generate vector types (`LoopType::Vector(ScalarType, width)`)
- [x] Generate vector load/store
- [x] Generate vector operations with FMA detection
- [x] Handle loop remainders
- [x] Test: `sum xs` uses SIMD

### 3.5 Parallel Loop Codegen ✅

**Crate:** `bhc-loop-ir`
**Location:** `parallel.rs` (400+ lines)

Tasks:
- [x] Identify parallelizable loops (`ParallelPass`)
- [x] Generate parallel loop structure with 3 strategies (Static/Dynamic/Guided)
- [x] Integrate with RTS thread pool
- [x] Handle reduction across threads
- [x] Test: Parallel map scales with cores

### 3.6 Loop IR to LLVM ✅

**Crate:** `bhc-codegen`
**Location:** `llvm/loop_lower.rs` (74KB)

Tasks:
- [x] Lower Loop IR to LLVM IR (`LoopLowering` struct)
- [x] Emit LLVM vector intrinsics (fabs, sqrt, floor, ceil, FMA)
- [x] Use LLVM's loop optimizations
- [x] Test: Generated assembly contains SIMD

### 3.7 Hot Arena Allocator ✅

**Crate:** `bhc-rts-arena`
**Location:** `lib.rs` (400+ lines)

Tasks:
- [x] Implement arena allocation (`HotArena`, bump pointer O(1))
- [x] Implement bulk free (scope-based lifetime)
- [x] Support alignment (16/32/64 byte for SIMD)
- [x] Test: Kernel temporaries use arena

### 3.8 Pinned Buffers ✅

**Crate:** `bhc-rts-alloc`
**Location:** `lib.rs`

Tasks:
- [x] Implement pinned allocation (`PinnedAllocator`)
- [x] Track pinned objects separately from GC heap
- [x] Implement reference counting for pinned
- [x] Test: FFI buffer survives GC

### Phase 3 Exit Criteria ✅

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
main = print $ sum $ map (*2) [1..10000000]
```

- ✅ Fuses to single loop
- ✅ `--kernel-report` shows fusion occurred
- ✅ Generated code uses SIMD

**Completed!** Commit `312f08c`

---

## Phase 4: WASM Backend 🟡 90% COMPLETE

**Objective:** `bhc --target=wasi Main.hs` produces working WebAssembly.

### 4.1 WASM Emitter ✅

**Crate:** `bhc-wasm`
**Location:** `codegen/mod.rs` (1,656 lines)

Tasks:
- [x] Emit WASM binary format (all 11 sections, LEB128 encoding)
- [x] Map types to WASM types (i32, i64, f32, f64, v128)
- [x] Generate WASM functions (`WasmFunc`, `WasmFuncType`)
- [x] Handle indirect calls (`CallIndirect` instruction)
- [x] SIMD128 support (`codegen/simd.rs`)
- [x] WAT text generation
- [x] Test: Valid WASM binary output

### 4.2 WASI Runtime Integration ✅

**Crate:** `bhc-wasm`
**Location:** `wasi.rs` (800+ lines)

Tasks:
- [x] Import WASI functions (fd_write, fd_read, proc_exit, args_*, environ_*)
- [x] Map IO primitives to WASI calls (`generate_print_i32()`, `generate_print_str()`)
- [x] Bump allocator (`generate_alloc_function()`)
- [x] Handle command-line arguments (`generate_init_args()`, `generate_get_argc()`, `generate_get_argv()`)
- [x] Handle environment variables (`generate_init_environ()`, `generate_getenv()`)
- [x] Test: Basic print works

### 4.3 Edge Profile RTS 🟡

**Crate:** `bhc-wasm`
**Location:** `runtime/mod.rs` (369 lines), `runtime/gc.rs` (625 lines)

Tasks:
- [x] Configuration for minimal RTS (`RuntimeConfig::edge()`)
- [x] Memory layout definition (`MemoryLayout`)
- [x] Arena allocator for WASM (`WasmArena`)
- [x] Full GC within linear memory (`generate_gc_*` functions in gc.rs)
- [ ] Minimize code size verification
- [ ] Test: Runtime < 100KB

### 4.4 Driver Integration ✅

**Crate:** `bhc-driver`
**Location:** `lib.rs` lines 467-537, 1109-1115

Tasks:
- [x] Add `--target=wasi` flag handling (`is_wasm_target()` at line 1109)
- [x] Wire WASM backend into compilation pipeline (lines 467-537)
- [x] Register wasm32-wasi target (detected via "wasm" in target triple)
- [x] Generate `.wasm` output files (`write_wasm()` at line 525-527)
- [ ] Test: End-to-end compilation (blocked by LLVM version mismatch)

### Phase 4 Exit Criteria

```bash
$ bhc --target=wasi Main.hs -o app.wasm
$ wasmtime app.wasm
Hello, World!
```

**Blocker:** LLVM version mismatch (system has LLVM 21, llvm-sys expects LLVM 18) prevents full testing.

**Remaining effort:** ~1 week (resolve LLVM version, end-to-end testing)

---

## Phase 5: Server Profile 🟡 90% COMPLETE

**Objective:** Structured concurrency with proper cancellation.

### 5.1 Task Scheduler ✅

**Crate:** `bhc-rts-scheduler`
**Location:** `lib.rs` (1,459 lines)

Tasks:
- [x] Implement work-stealing deque (crossbeam)
- [x] Implement worker threads (configurable count)
- [x] Implement task spawning
- [x] Implement task completion with statistics
- [x] Test: 15 tests pass

### 5.2 Scope & Task Primitives ✅

**Crate:** `bhc-rts-scheduler`

Tasks:
- [x] Implement `Scope` type
- [x] Implement `with_scope()` (structured concurrency)
- [x] Implement `spawn()` within scope
- [x] Implement `await()` (blocking and non-blocking)
- [x] Test: Concurrent tasks complete within scope

### 5.3 Cancellation ✅

**Crate:** `bhc-rts-scheduler`
**Location:** lines 325-361

Tasks:
- [x] Implement cancellation tokens (thread-local flag)
- [x] Implement `cancel()` method
- [x] Implement cancellation propagation to children
- [x] Implement `check_cancelled()` cooperative checking
- [x] Test: Cancelled task stops

### 5.4 STM ✅

**Crate:** `bhc-concurrent`
**Location:** `stdlib/bhc-concurrent/src/stm.rs` (971 lines)

Tasks:
- [x] Implement `TVar` type with atomic versioning (lines 84-179)
- [x] Implement `atomically()` with retry/conflict handling (lines 393-460)
- [x] SATB write barriers
- [x] Implement `retry` primitive (lines 480-482)
- [x] Implement `orElse` combinator (lines 505-514)
- [x] Implement conflict detection (validation in `Transaction::commit()`)
- [x] Tests: 13 tests including bank transfer, producer-consumer (lines 782-970)

### 5.5 Deadlines ✅

**Crate:** `bhc-rts-scheduler`
**Location:** lines 1057-1109

Tasks:
- [x] Implement `with_deadline(duration, closure)`
- [x] Implement deadline propagation (timer thread)
- [x] Test: Operation times out

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

**Blockers:** None - core functionality complete.

**Remaining effort:** ~1 week (observability hooks, integration testing)

---

## Phase 6: GPU Backend 🟡 80% COMPLETE

**Objective:** Tensor operations run on GPU.

### 6.1 PTX Codegen ✅

**Crate:** `bhc-gpu`
**Location:** `codegen/ptx.rs` (1,168 lines)

Tasks:
- [x] PTX module header generation
- [x] Kernel entry point signatures
- [x] Parameter marshalling
- [x] Type mapping (`dtype_to_gpu_type`)
- [x] Loop nest code generation (`generate_loop_nest()`)
- [x] Map/ZipWith/Reduce operations
- [x] Parallel reduction with shared memory
- [ ] Test: Simple kernel compiles (blocked by LLVM)

### 6.2 AMDGCN Codegen ✅

**Crate:** `bhc-gpu`
**Location:** `codegen/amdgcn.rs` (580 lines)

Tasks:
- [x] AMDGCN module header
- [x] Kernel entry generation
- [x] Parameter handling
- [x] Loop nest code generation (`generate_loop_nest_amd()`)
- [x] Unary/binary operations
- [ ] Test: Simple kernel compiles (blocked by LLVM)

### 6.3 Device Memory Management ✅

**Crate:** `bhc-gpu`
**Location:** `memory.rs`

Tasks:
- [x] Implement device allocation (`DeviceBuffer<T>`)
- [x] Pool-based memory management
- [x] Alignment tracking
- [x] Safety checks for bounds

### 6.4 Host-Device Transfers ✅

**Crate:** `bhc-gpu`
**Location:** `transfer.rs`

Tasks:
- [x] Implement host→device transfer
- [x] Implement device→host transfer
- [x] Device-to-device copy
- [x] Async transfer support via streams
- [x] Test: Data flows to/from GPU

### 6.5 Kernel Launch 🟡

**Crate:** `bhc-gpu`
**Location:** `kernel.rs`

Tasks:
- [x] `GpuKernel` compiled kernel representation
- [x] `LaunchConfig` for grid/block dimensions
- [x] Launch parameter setup
- [ ] Full kernel execution pipeline
- [ ] Test: Kernel executes on GPU

### 6.6 Runtime Support ✅

**Crate:** `bhc-gpu`
**Location:** `runtime/cuda.rs`, `runtime/rocm.rs`

Tasks:
- [x] CUDA runtime integration (cuBLAS)
- [x] ROCm/HIP runtime support
- [x] Device enumeration and selection
- [x] Stream and context management

### Phase 6 Exit Criteria

```haskell
{-# OPTIONS_BHC -profile=numeric #-}
main = do
  let a = fromList [1..1000000]
  let b = fromList [1..1000000]
  print $ sum $ zipWith (+) a b
```

**Blockers:** LLVM version mismatch prevents full testing.

**Remaining effort:** ~1 week (kernel execution pipeline, testing)

---

## Phase 7: Advanced Profiles 🟡 90% COMPLETE

### 7.1 Realtime (Bounded GC) ✅

**Crate:** `bhc-rts-gc`
**Location:** `incremental.rs` (730 lines), `lib.rs` (1,500+ lines)

Tasks:
- [x] Tri-color marking infrastructure (White/Gray/Black)
- [x] `MarkState` enum (Idle/RootScanning/Marking/Remark/Complete)
- [x] SATB write barrier buffer
- [x] Pause budget configuration (`IncrementalConfig`, 500μs default)
- [x] Wire mark loop into main GC (`start_incremental_collect()`, `do_incremental_work()`, `finish_incremental_collect()`)
- [x] Pause time measurement (via `PauseMeasurement`, `PauseStats`)
- [x] Test: 32 tests pass including incremental GC cycle tests

### 7.2 Generational GC ✅

**Crate:** `bhc-rts-gc`
**Location:** `lib.rs` (1,526 lines)

Tasks:
- [x] Three-generation model (Nursery/Survivor/Old)
- [x] Write barriers for cross-generation references
- [x] Promotion logic
- [x] Collection statistics (`GcStats`)

### 7.3 Embedded (No GC) ✅

**Crate:** `bhc-rts-alloc`, `bhc-core`, `bhc-session`, `bhc-driver`
**Location:** `static_alloc.rs` (200+ lines), `escape.rs` (585 lines), `lib.rs`

Tasks:
- [x] Static allocator with fixed-size buffer
- [x] Bump pointer allocation (O(1))
- [x] No-GC design for embedded
- [x] Escape analysis (`analyze_escape()`, `check_embedded_safe()`)
- [x] EscapeStatus enum (NoEscape/EscapeReturn/EscapeCapture/EscapeStore/EscapeExternal)
- [x] Profile::Embedded with `is_gc_free()` and `requires_escape_analysis()`
- [x] CompileError::EscapeAnalysisFailed in driver
- [x] check_escape_analysis() in compilation pipeline
- [ ] Test: Bare-metal program (blocked by LLVM version)

### 7.4 Arena Allocation ✅

**Crate:** `bhc-rts-arena`

Tasks:
- [x] Hot arena for ephemeral allocations
- [x] Bulk deallocation at scope end
- [x] No GC interaction

### Phase 7 Exit Criteria

**Blockers:** Bare-metal testing (blocked by LLVM version mismatch).

**Remaining effort:** ~1-2 days (bare-metal testing once LLVM fixed)

---

## Phase 8: Ecosystem ✅ COMPLETE

### 8.1 REPL ✅

**Crate:** `tools/bhci`

Tasks:
- [x] Interactive evaluation loop (rustyline)
- [x] Expression type inference (`:type` command)
- [x] Value pretty-printing
- [x] Full command set (`:help`, `:load`, `:reload`, `:browse`, etc.)
- [x] Test: `:t map` shows type

### 8.2 Package Manager ✅

**Crate:** `bhc-package`

Tasks:
- [x] Package description format (TOML `bhc.toml`)
- [x] Dependency resolution with semver
- [x] Build orchestration
- [x] Registry integration
- [x] Lockfile management

### 8.3 LSP Server ✅

**Crate:** `bhc-lsp`

Tasks:
- [x] Diagnostics (errors, warnings)
- [x] Go to definition
- [x] Hover information
- [x] Completions
- [x] Document/workspace symbols
- [x] Code actions

### 8.4 Documentation ✅

Tasks:
- [x] User-facing documentation (`docs/getting-started.md`, `language.md`, `profiles.md`, `examples.md`)
- [x] Developer documentation (LSP server architecture)
- [x] API documentation for new crates

### Phase 8 Exit Criteria ✅

```bash
$ bhci
bhci> :t map
map :: (a -> b) -> [a] -> [b]

$ bhc-lsp  # Starts LSP server for IDE integration
```

**Completed!**

---

## Summary Timeline

| Phase | Description | Status | Completion |
|-------|-------------|--------|------------|
| 1 | Native Hello World | ✅ Complete | 100% |
| 2 | Language Completeness | ✅ Complete | 100% |
| 3 | Numeric Profile | ✅ Complete | 100% |
| 4 | WASM Backend | 🟡 In Progress | 95% |
| 5 | Server Profile | 🟡 In Progress | 90% |
| 6 | GPU Backend | 🟡 In Progress | 80% |
| 7 | Advanced Profiles | 🟡 In Progress | 90% |
| 8 | Ecosystem | ✅ Complete | 100% |

**Overall: ~94% complete**

---

## Remaining Work

### Phase 4 (WASM) - ~2-3 days
1. ~~Wire `--target=wasi` in bhc-driver~~ ✅
2. ~~Add args/environ WASI support~~ ✅
3. ~~Complete GC within linear memory~~ ✅ (mark-sweep GC in `runtime/gc.rs`)
4. Verify runtime code size < 100KB
5. End-to-end test with wasmtime (blocked by LLVM)

### Phase 5 (Server) - 1-2 weeks
1. Implement STM `retry` primitive
2. Implement STM `orElse` combinator
3. Complete transaction conflict detection

### Phase 6 (GPU) - 2-3 weeks
1. Implement PTX loop nest codegen
2. Implement AMDGCN loop nest codegen
3. Complete Tensor IR → GPU kernel lowering
4. End-to-end GPU test

### Phase 7 (Advanced) - ~1-2 days
1. ~~Wire incremental mark loop into GC~~ ✅
2. ~~Implement escape analysis for embedded profile~~ ✅
3. ~~Wire escape analysis into embedded profile compilation path~~ ✅
4. Bare-metal testing for embedded profile (blocked by LLVM)

---

## Immediate Next Steps

**Priority 1: WASM Backend (Phase 4)**
- Fastest path to completion
- All infrastructure exists, just needs driver wiring
- Enables serverless/browser deployment

**Priority 2: Server Profile (Phase 5)**
- STM primitives need completion
- Scheduler and concurrency already work

**Priority 3: GPU Backend (Phase 6)**
- Loop codegen is the main blocker
- Memory management and runtime already complete

**Priority 4: Advanced Profiles (Phase 7)**
- Incremental GC wired into main collector ✅
- Escape analysis implemented ✅
- Driver integration for embedded profile remaining
