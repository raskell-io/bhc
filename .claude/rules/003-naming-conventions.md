# Naming Conventions

**Rule ID:** BHC-RULE-003
**Applies to:** All BHC source code

---

## General Principles

1. **Descriptive** — Names should describe what, not how
2. **Consistent** — Follow established patterns
3. **Appropriate length** — Short for small scopes, longer for larger scopes
4. **Searchable** — Avoid single letters except in very short lambdas

---

## Case Conventions

| Entity | Convention | Example |
|--------|------------|---------|
| Types | PascalCase | `TensorIR`, `TypeEnv` |
| Type variables | lowercase, short | `a`, `m`, `f` |
| Data constructors | PascalCase | `Just`, `TensorOp` |
| Functions | camelCase | `parseExpr`, `runTypeCheck` |
| Variables | camelCase | `currentScope`, `elemType` |
| Constants | camelCase | `maxIterations`, `defaultTimeout` |
| Modules | PascalCase.Hierarchy | `BHC.Core.IR` |

---

## Module Naming

### Hierarchy

```
BHC                      -- Root namespace
├── Parser               -- Parsing subsystem
│   ├── Lexer           -- Tokenization
│   ├── Parser          -- Parsing
│   └── AST             -- Abstract syntax tree
├── TypeCheck            -- Type checking
│   ├── Infer           -- Type inference
│   ├── Unify           -- Unification
│   └── Env             -- Type environment
├── Core                 -- Core IR
│   ├── IR              -- IR data types
│   ├── Eval            -- Evaluation/interpretation
│   └── Transform       -- Transformations
├── Tensor               -- Tensor IR
│   ├── IR              -- Tensor IR types
│   ├── Lower           -- Lowering from Core
│   ├── Fusion          -- Fusion passes
│   └── Shape           -- Shape analysis
├── Codegen              -- Code generation
│   ├── LLVM            -- LLVM backend
│   └── Native          -- Native code
└── RTS                  -- Runtime system interfaces
    ├── Memory          -- Memory management
    ├── GC              -- Garbage collection
    └── Scheduler       -- Task scheduling
```

### Module Name Rules

- MUST match directory structure
- SHOULD be singular (`Parser`, not `Parsers`)
- Internal modules SHOULD use `.Internal` suffix
- Test modules MUST use `.Tests` suffix

---

## Function Naming

### Prefixes

| Prefix | Meaning | Example |
|--------|---------|---------|
| `mk` | Smart constructor | `mkShape`, `mkTensor` |
| `un` | Unwrap newtype | `unVarId`, `unShape` |
| `is` | Boolean predicate | `isValid`, `isEmpty` |
| `has` | Boolean predicate | `hasType`, `hasFusedOps` |
| `get` | Accessor (possibly partial) | `getField` |
| `set` | Setter | `setName`, `setShape` |
| `with` | Scoped resource | `withScope`, `withArena` |
| `to` | Conversion | `toList`, `toVector` |
| `from` | Conversion | `fromList`, `fromShape` |
| `parse` | Parse from text | `parseExpr`, `parseType` |
| `render` | Render to text | `renderExpr`, `renderType` |
| `run` | Execute effect | `runParser`, `runTypeCheck` |
| `eval` | Evaluate | `evalExpr`, `evalTensor` |
| `lower` | IR lowering | `lowerToTensor`, `lowerToLoop` |

### Suffixes

| Suffix | Meaning | Example |
|--------|---------|---------|
| `'` | Strict variant | `foldl'`, `insertWith'` |
| `_` | Variant ignoring result | `forM_`, `traverse_` |
| `M` | Monadic variant | `mapM`, `filterM` |
| `IO` | IO-specific variant | `readIO`, `printIO` |
| `Maybe` | Returns Maybe | `lookupMaybe` |
| `Either` | Returns Either | `parseEither` |
| `Unsafe` | Unsafe variant | `unsafeIndex`, `unsafePerformIO` |

---

## Variable Naming

### Common Short Names

These short names are acceptable in limited scopes:

| Name | Usage |
|------|-------|
| `x`, `y`, `z` | Generic values |
| `n`, `m`, `k` | Numbers, sizes |
| `i`, `j` | Loop indices |
| `f`, `g`, `h` | Functions |
| `a`, `b`, `c` | Type variables |
| `e` | Expression, error |
| `t` | Type |
| `s` | State |
| `r` | Result, reader env |
| `w` | Writer output |
| `acc` | Accumulator |
| `env` | Environment |
| `ctx` | Context |

### Longer Names for Larger Scopes

- Functions: MUST use descriptive names
- Module-level: MUST use descriptive names
- Loop bodies spanning >5 lines: SHOULD use descriptive names

```haskell
-- Good: descriptive names for larger scope
optimizeTensorGraph :: TensorGraph -> Config -> OptimizedGraph
optimizeTensorGraph graph config =
  let fusedGraph = applyFusion graph (fusionRules config)
      scheduledGraph = scheduleOps fusedGraph (targetDevice config)
  in lowerToLoops scheduledGraph

-- Good: short names in small lambda
map (\x -> x + 1) xs

-- Good: short names in list comprehension
[x * y | x <- xs, y <- ys, x > 0]
```

---

## Type Variable Naming

### Conventions

| Variable | Typical Usage |
|----------|---------------|
| `a`, `b`, `c` | Arbitrary types |
| `f`, `g` | Functor/Applicative |
| `m` | Monad |
| `t` | Traversable |
| `k` | Kind variable |
| `s` | State type |
| `r` | Reader environment |
| `w` | Writer output |
| `e` | Error type |

### Descriptive Type Variables

For complex signatures, use descriptive names:

```haskell
-- When roles are clear from context
fmap :: (a -> b) -> f a -> f b

-- When more clarity is needed
transformTensor
  :: (inputElem -> outputElem)
  -> Tensor inputShape inputElem
  -> Tensor outputShape outputElem
```

---

## IR Node Naming

### Core IR

| Constructor | Usage |
|-------------|-------|
| `Lit` | Literal values |
| `Var` | Variable reference |
| `App` | Function application |
| `Lam` | Lambda abstraction |
| `Let` | Let binding |
| `Case` | Case expression |
| `Type` | Type annotation |

### Tensor IR

| Constructor | Usage |
|-------------|-------|
| `TMap` | Elementwise map |
| `TZip` | Zip operation |
| `TReduce` | Reduction |
| `TSlice` | Slicing/view |
| `TReshape` | Reshape |
| `TTranspose` | Transpose |
| `TMatMul` | Matrix multiplication |
| `TConcat` | Concatenation |
| `TBroadcast` | Broadcasting |

---

## File Naming

| File Type | Convention | Example |
|-----------|------------|---------|
| Haskell module | PascalCase.hs | `Parser.hs`, `TensorIR.hs` |
| Test module | ...Tests.hs | `ParserTests.hs` |
| Benchmark | ...Bench.hs | `MatmulBench.hs` |
| Config | lowercase | `cabal.project`, `.hlint.yaml` |
| Documentation | UPPERCASE.md | `README.md`, `CONTRIBUTING.md` |

---

## Abbreviations

### Approved Abbreviations

| Full | Abbreviation | Usage |
|------|--------------|-------|
| configuration | config | `Config`, `configFile` |
| environment | env | `TypeEnv`, `env` |
| expression | expr | `Expr`, `parseExpr` |
| identifier | id | `VarId`, `getId` |
| information | info | `TypeInfo`, `getInfo` |
| initialization | init | `initState` |
| intermediate representation | IR | `CoreIR`, `TensorIR` |
| maximum | max | `maxSize` |
| minimum | min | `minSize` |
| number | num | `numElements` |
| reference | ref | `VarRef` |
| specification | spec | `TypeSpec` |
| temporary | tmp | `tmpBuffer` |

### Forbidden Abbreviations

- Don't abbreviate: `accumulator` → `acc` is fine, but `buffer` → `buf` is not
- Don't use: `cnt` (use `count`), `idx` (use `index` or `i`), `mgr` (use `manager`)
- Spell out when abbreviation is ambiguous

---

## Constants and Magic Numbers

- MUST name all magic numbers
- SHOULD use ALL_CAPS for truly constant values (Haskell convention: camelCase)
- MUST document meaning

```haskell
-- Good
defaultVectorWidth :: Int
defaultVectorWidth = 256  -- bits

maxFusionDepth :: Int
maxFusionDepth = 10

-- Bad
process xs = take 256 xs  -- What is 256?
```
