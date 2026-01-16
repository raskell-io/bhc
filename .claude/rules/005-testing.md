# Testing Requirements

**Rule ID:** BHC-RULE-005
**Applies to:** All BHC code

---

## Testing Philosophy

1. **Test behavior, not implementation** — Tests should survive refactoring
2. **Fast feedback** — Unit tests must be fast (<1s per test)
3. **Comprehensive coverage** — Critical paths need thorough testing
4. **Reproducibility** — Tests must be deterministic

---

## Test Categories

### Unit Tests

Test individual functions in isolation.

- MUST be fast (<100ms each)
- MUST be deterministic
- SHOULD test edge cases
- Location: `tests/Unit/`

```haskell
-- tests/Unit/Parser/LexerTests.hs
module Unit.Parser.LexerTests where

import Test.Tasty
import Test.Tasty.HUnit

import BHC.Parser.Lexer

tests :: TestTree
tests = testGroup "Lexer"
  [ testCase "lexes integer literals" $ do
      lex "42" @?= [TokInt 42]

  , testCase "lexes negative integers" $ do
      lex "-42" @?= [TokMinus, TokInt 42]

  , testCase "handles empty input" $ do
      lex "" @?= []
  ]
```

### Property Tests

Test invariants with random inputs.

- MUST use QuickCheck or Hedgehog
- SHOULD have at least 100 test cases per property
- SHOULD shrink failures to minimal examples
- Location: `tests/Property/`

```haskell
-- tests/Property/Tensor/ShapeTests.hs
module Property.Tensor.ShapeTests where

import Test.Tasty
import Test.Tasty.QuickCheck

import BHC.Tensor.Shape

tests :: TestTree
tests = testGroup "Shape Properties"
  [ testProperty "reshape preserves element count" $ \shape1 shape2 ->
      product shape1 == product shape2 ==>
        case reshape shape2 (zeros shape1) of
          Right t  -> tensorSize t === product shape1
          Left _   -> property False

  , testProperty "transpose is involutive" $ \t ->
      transpose (transpose t) === t

  , testProperty "slice produces smaller tensor" $ \t slice ->
      tensorSize (applySlice slice t) <= tensorSize t
  ]
```

### Integration Tests

Test component interactions.

- MAY be slower (up to 10s each)
- SHOULD test realistic scenarios
- Location: `tests/Integration/`

```haskell
-- tests/Integration/CompilerTests.hs
module Integration.CompilerTests where

import Test.Tasty
import Test.Tasty.HUnit

import BHC.Driver (compile)
import BHC.RTS (run)

tests :: TestTree
tests = testGroup "Compiler Integration"
  [ testCase "compiles and runs dot product" $ do
      let source = "let dot xs ys = sum (zipWith (*) xs ys) in dot [1,2,3] [4,5,6]"
      result <- compile source >>= run
      result @?= VInt 32

  , testCase "fusion occurs for map-map" $ do
      let source = "map (+1) (map (*2) [1,2,3])"
      ir <- compileToTensorIR source
      countKernels ir @?= 1  -- Fused to single kernel
  ]
```

### Conformance Tests

Test H26 specification compliance.

- MUST cover all normative requirements
- MUST reference spec section
- Location: `tests/Conformance/`

```haskell
-- tests/Conformance/Section8/FusionTests.hs
-- Tests for H26-SPEC Section 8: Fusion Laws

module Conformance.Section8.FusionTests where

tests :: TestTree
tests = testGroup "Section 8 - Fusion Laws"
  [ testGroup "8.1 - Guaranteed Fusion Patterns"
      [ testCase "8.1.1: map f (map g x) fuses" $ do
          -- MUST fuse per H26-SPEC-0001 Section 8.1 Pattern 1
          assertFused "map (+1) (map (*2) xs)"

      , testCase "8.1.2: zipWith f (map g a) (map h b) fuses" $ do
          -- MUST fuse per H26-SPEC-0001 Section 8.1 Pattern 2
          assertFused "zipWith (+) (map (*2) xs) (map (*3) ys)"

      , testCase "8.1.3: sum (map f x) fuses" $ do
          -- MUST fuse per H26-SPEC-0001 Section 8.1 Pattern 3
          assertFused "sum (map (*2) xs)"
      ]
  ]
```

### Benchmark Tests

Performance regression tests.

- MUST have baseline expectations
- MUST run in CI for regression detection
- Location: `tests/Benchmarks/`

```haskell
-- tests/Benchmarks/NumericBench.hs
module Benchmarks.NumericBench where

import Test.Tasty.Bench

benchmarks :: Benchmark
benchmarks = bgroup "Numeric"
  [ bench "dot product 1K" $ nf (dot v1k) v1k
  , bench "dot product 1M" $ nf (dot v1m) v1m
  , bench "matmul 64x64" $ nf (uncurry matmul) (m64, m64)
  , bench "matmul 256x256" $ nf (uncurry matmul) (m256, m256)
  ]

-- Regression assertions
regressionTests :: TestTree
regressionTests = testGroup "Performance Regression"
  [ testCase "dot product 1M under 1ms" $ do
      time <- benchmark $ dot v1m v1m
      assertBool "Too slow" (time < 0.001)
  ]
```

---

## Test Requirements by Component

### Parser

- [ ] All token types
- [ ] Valid syntax constructs
- [ ] Error recovery
- [ ] Source locations preserved

### Type Checker

- [ ] Type inference correctness
- [ ] Unification edge cases
- [ ] Error messages quality
- [ ] Typeclass resolution

### Core IR

- [ ] Evaluation correctness
- [ ] Transformation soundness
- [ ] Pretty printing round-trips

### Tensor IR

- [ ] Shape inference
- [ ] Stride calculation
- [ ] Fusion correctness
- [ ] Lowering soundness

### Codegen

- [ ] Generated code correctness
- [ ] Optimization preservation
- [ ] Target-specific features

### RTS

- [ ] Memory management
- [ ] Concurrency primitives
- [ ] Cancellation behavior
- [ ] GC correctness

---

## Test Data

### Golden Tests

For output stability:

```haskell
-- tests/Golden/ParserTests.hs
goldenTests :: TestTree
goldenTests = testGroup "Parser Golden"
  [ goldenVsFile "simple function"
      "golden/simple-func.expected"
      "golden/simple-func.actual"
      (parseAndDump "let f x = x + 1")
  ]
```

### Test Fixtures

Shared test data in `tests/fixtures/`:

```
tests/fixtures/
├── programs/          # Test programs
│   ├── valid/        # Should compile
│   └── invalid/      # Should fail with specific errors
├── tensors/          # Serialized tensor data
└── golden/           # Expected outputs
```

---

## Coverage Requirements

### Minimum Coverage

- Overall: 80% line coverage
- Critical paths (IR, codegen): 90% line coverage
- Conformance features: 100% requirement coverage

### Coverage Exclusions

MAY exclude from coverage:

- Debug/tracing code
- Error paths that "can't happen"
- Generated code

```haskell
-- Exclude from coverage with pragma
{-# COVERAGE_EXCLUDE #-}
debugTrace :: String -> IO ()
debugTrace = putStrLn
```

---

## Testing Utilities

### Test Helpers

Common helpers in `tests/TestUtils.hs`:

```haskell
module TestUtils where

-- | Assert that an expression compiles and evaluates to expected value
assertEval :: String -> Value -> Assertion
assertEval source expected = do
  result <- compile source >>= run
  result @?= expected

-- | Assert that code fuses to a single kernel
assertFused :: String -> Assertion
assertFused source = do
  ir <- compileToTensorIR source
  let kernels = countKernels ir
  assertEqual "Should fuse to single kernel" 1 kernels

-- | Assert compilation fails with specific error
assertCompileError :: String -> (CompileError -> Bool) -> Assertion
assertCompileError source predicate = do
  result <- tryCompile source
  case result of
    Left err | predicate err -> pure ()
    Left err -> assertFailure $ "Wrong error: " ++ show err
    Right _  -> assertFailure "Expected compilation to fail"
```

### Arbitrary Instances

For property testing:

```haskell
-- tests/Arbitrary.hs
module Arbitrary where

instance Arbitrary Shape where
  arbitrary = do
    rank <- choose (0, 4)
    dims <- vectorOf rank (choose (1, 100))
    pure $ Shape dims

  shrink (Shape dims) = Shape <$> shrink dims

instance Arbitrary (Tensor Float) where
  arbitrary = do
    shape <- arbitrary
    elems <- vectorOf (product shape) arbitrary
    pure $ fromList shape elems
```

---

## CI Integration

### Test Stages

1. **Fast** (<2 min): Unit tests, lint
2. **Full** (<10 min): All tests including integration
3. **Nightly**: Benchmarks, fuzzing, extended conformance

### Test Commands

```bash
# Run all unit tests
cabal test unit-tests

# Run property tests with more cases
cabal test property-tests --test-option=--quickcheck-tests=1000

# Run conformance suite
cabal test conformance-tests

# Run with coverage
cabal test --enable-coverage

# Run benchmarks
cabal bench
```

---

## Debugging Test Failures

### Reproducing Failures

Property test failures MUST include:

1. Seed for reproduction
2. Shrunk counterexample
3. All relevant inputs

```
Test failed with seed 12345
Counterexample (shrunk):
  input: Shape [0, 5]
  expected: valid reshape
  actual: dimension error
```

### Test Isolation

- Tests MUST NOT depend on global state
- Tests MUST NOT depend on execution order
- Tests SHOULD clean up after themselves
