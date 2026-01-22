-- Test: numeric-operations
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 3.2

{-# HASKELL_EDITION 2026 #-}

module NumericOpsTest where

-- ================================================================
-- Basic Arithmetic Operations
-- ================================================================

-- Addition
testAdd1 :: Int
testAdd1 = 1 + 2
-- Result: 3

testAdd2 :: Int
testAdd2 = 100 + 200 + 300
-- Result: 600

testAddNeg :: Int
testAddNeg = 10 + (-5)
-- Result: 5

-- Subtraction
testSub1 :: Int
testSub1 = 10 - 3
-- Result: 7

testSub2 :: Int
testSub2 = 5 - 10
-- Result: -5

-- Multiplication
testMul1 :: Int
testMul1 = 6 * 7
-- Result: 42

testMul2 :: Int
testMul2 = (-3) * 4
-- Result: -12

-- Division
testDiv1 :: Int
testDiv1 = 10 `div` 3
-- Result: 3

testDiv2 :: Int
testDiv2 = (-10) `div` 3
-- Result: -4 (rounds towards negative infinity)

-- Modulo
testMod1 :: Int
testMod1 = 10 `mod` 3
-- Result: 1

testMod2 :: Int
testMod2 = (-10) `mod` 3
-- Result: 2

-- Remainder
testRem1 :: Int
testRem1 = 10 `rem` 3
-- Result: 1

testRem2 :: Int
testRem2 = (-10) `rem` 3
-- Result: -1

-- ================================================================
-- Comparison Operations
-- ================================================================

-- Equality
testEq1 :: Bool
testEq1 = 5 == 5
-- Result: True

testEq2 :: Bool
testEq2 = 5 == 6
-- Result: False

-- Inequality
testNe1 :: Bool
testNe1 = 5 /= 6
-- Result: True

testNe2 :: Bool
testNe2 = 5 /= 5
-- Result: False

-- Less than
testLt1 :: Bool
testLt1 = 3 < 5
-- Result: True

testLt2 :: Bool
testLt2 = 5 < 3
-- Result: False

testLt3 :: Bool
testLt3 = 5 < 5
-- Result: False

-- Less than or equal
testLe1 :: Bool
testLe1 = 3 <= 5
-- Result: True

testLe2 :: Bool
testLe2 = 5 <= 5
-- Result: True

testLe3 :: Bool
testLe3 = 6 <= 5
-- Result: False

-- Greater than
testGt1 :: Bool
testGt1 = 5 > 3
-- Result: True

testGt2 :: Bool
testGt2 = 3 > 5
-- Result: False

-- Greater than or equal
testGe1 :: Bool
testGe1 = 5 >= 3
-- Result: True

testGe2 :: Bool
testGe2 = 5 >= 5
-- Result: True

-- ================================================================
-- Boolean Operations
-- ================================================================

-- And
testAnd1 :: Bool
testAnd1 = True && True
-- Result: True

testAnd2 :: Bool
testAnd2 = True && False
-- Result: False

testAnd3 :: Bool
testAnd3 = False && True
-- Result: False

-- Or
testOr1 :: Bool
testOr1 = False || True
-- Result: True

testOr2 :: Bool
testOr2 = False || False
-- Result: False

testOr3 :: Bool
testOr3 = True || False
-- Result: True

-- Not
testNot1 :: Bool
testNot1 = not True
-- Result: False

testNot2 :: Bool
testNot2 = not False
-- Result: True

-- ================================================================
-- Unary Numeric Operations
-- ================================================================

-- Negation
testNegate1 :: Int
testNegate1 = negate 5
-- Result: -5

testNegate2 :: Int
testNegate2 = negate (-5)
-- Result: 5

-- Absolute value
testAbs1 :: Int
testAbs1 = abs (-5)
-- Result: 5

testAbs2 :: Int
testAbs2 = abs 5
-- Result: 5

testAbs3 :: Int
testAbs3 = abs 0
-- Result: 0

-- Signum
testSignum1 :: Int
testSignum1 = signum 10
-- Result: 1

testSignum2 :: Int
testSignum2 = signum (-10)
-- Result: -1

testSignum3 :: Int
testSignum3 = signum 0
-- Result: 0

-- ================================================================
-- Combined Operations
-- ================================================================

-- Arithmetic with comparison
testCombined1 :: Bool
testCombined1 = (2 + 3) == 5
-- Result: True

testCombined2 :: Bool
testCombined2 = (10 - 3) > 5
-- Result: True (7 > 5)

-- Multiple operations
testCombined3 :: Int
testCombined3 = 2 + 3 * 4
-- Result: 14 (multiplication has higher precedence)

testCombined4 :: Int
testCombined4 = (2 + 3) * 4
-- Result: 20

-- Nested operations
testCombined5 :: Int
testCombined5 = abs (5 - 10)
-- Result: 5

-- Conditional with comparison
testCombined6 :: Int
testCombined6 = if 5 > 3 then 1 else 0
-- Result: 1

-- Using operations in guards
testGuard :: Int -> String
testGuard n
  | n < 0     = "negative"
  | n == 0    = "zero"
  | n < 100   = "small"
  | otherwise = "large"

testGuard1 :: String
testGuard1 = testGuard (-5)
-- Result: "negative"

testGuard2 :: String
testGuard2 = testGuard 0
-- Result: "zero"

testGuard3 :: String
testGuard3 = testGuard 50
-- Result: "small"

testGuard4 :: String
testGuard4 = testGuard 200
-- Result: "large"

-- ================================================================
-- Recursion with Numeric Operations
-- ================================================================

-- Factorial
factorial :: Int -> Int
factorial n = case n of
  0 -> 1
  _ -> n * factorial (n - 1)

testFactorial1 :: Int
testFactorial1 = factorial 0
-- Result: 1

testFactorial2 :: Int
testFactorial2 = factorial 5
-- Result: 120

testFactorial3 :: Int
testFactorial3 = factorial 10
-- Result: 3628800

-- Fibonacci
fib :: Int -> Int
fib n = case n of
  0 -> 0
  1 -> 1
  _ -> fib (n - 1) + fib (n - 2)

testFib1 :: Int
testFib1 = fib 0
-- Result: 0

testFib2 :: Int
testFib2 = fib 10
-- Result: 55

-- Sum of list (using recursion)
sumList :: [Int] -> Int
sumList xs = case xs of
  []     -> 0
  (y:ys) -> y + sumList ys

testSum1 :: Int
testSum1 = sumList []
-- Result: 0

testSum2 :: Int
testSum2 = sumList [1, 2, 3, 4, 5]
-- Result: 15

-- Length of list
lengthList :: [a] -> Int
lengthList xs = case xs of
  []     -> 0
  (_:ys) -> 1 + lengthList ys

testLength1 :: Int
testLength1 = lengthList []
-- Result: 0

testLength2 :: Int
testLength2 = lengthList [1, 2, 3, 4, 5]
-- Result: 5

-- ================================================================
-- Double (floating point) Operations
-- ================================================================

testDoubleAdd :: Double
testDoubleAdd = 1.5 + 2.5
-- Result: 4.0

testDoubleMul :: Double
testDoubleMul = 3.0 * 4.0
-- Result: 12.0

testDoubleDiv :: Double
testDoubleDiv = 10.0 / 4.0
-- Result: 2.5

testDoubleCmp :: Bool
testDoubleCmp = 3.14 > 3.0
-- Result: True

-- ================================================================
-- Main function to run all tests
-- ================================================================

main :: IO ()
main = do
  -- Basic arithmetic
  print testAdd1      -- Expected: 3
  print testAdd2      -- Expected: 600
  print testSub1      -- Expected: 7
  print testSub2      -- Expected: -5
  print testMul1      -- Expected: 42
  print testMul2      -- Expected: -12
  print testDiv1      -- Expected: 3
  print testMod1      -- Expected: 1
  print testRem1      -- Expected: 1

  -- Comparisons (print 1 for True, 0 for False)
  print testEq1       -- Expected: True (1)
  print testEq2       -- Expected: False (0)
  print testLt1       -- Expected: True (1)
  print testGt1       -- Expected: True (1)

  -- Boolean
  print testAnd1      -- Expected: True (1)
  print testOr1       -- Expected: True (1)
  print testNot1      -- Expected: False (0)

  -- Unary
  print testNegate1   -- Expected: -5
  print testAbs1      -- Expected: 5
  print testSignum1   -- Expected: 1
  print testSignum2   -- Expected: -1

  -- Combined
  print testCombined1 -- Expected: True (1)
  print testCombined3 -- Expected: 14
  print testCombined4 -- Expected: 20

  -- Recursion
  print testFactorial2 -- Expected: 120
  print testFib2       -- Expected: 55
  print testSum2       -- Expected: 15
  print testLength2    -- Expected: 5

  -- Double
  print testDoubleAdd  -- Expected: 4.0
  print testDoubleMul  -- Expected: 12.0
