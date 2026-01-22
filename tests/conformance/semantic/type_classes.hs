-- Test: type-classes
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 5.1

{-# HASKELL_EDITION 2026 #-}

module TypeClassesTest where

-- ================================================================
-- Numeric Literals and Defaulting
-- ================================================================

-- Integer literals should default to Int when unconstrained
testIntLiteral :: Int
testIntLiteral = 42
-- Result: 42

-- Negative integer literal
testNegIntLiteral :: Int
testNegIntLiteral = -17
-- Result: -17

-- Float literals should default to Float when unconstrained
testFloatLiteral :: Float
testFloatLiteral = 3.14
-- Result: 3.14

-- Negative float literal
testNegFloatLiteral :: Float
testNegFloatLiteral = -2.5
-- Result: -2.5

-- ================================================================
-- Arithmetic Operations
-- ================================================================

-- Addition with Int
testAddInt :: Int
testAddInt = 10 + 20
-- Result: 30

-- Subtraction with Int
testSubInt :: Int
testSubInt = 50 - 15
-- Result: 35

-- Multiplication with Int
testMulInt :: Int
testMulInt = 6 * 7
-- Result: 42

-- Division with Float
testDivFloat :: Float
testDivFloat = 10.0 / 4.0
-- Result: 2.5

-- ================================================================
-- Polymorphic Numeric Functions
-- ================================================================

-- Double a numeric value (polymorphic over Num)
double :: Int -> Int
double x = x + x

testDouble :: Int
testDouble = double 21
-- Result: 42

-- Square a numeric value
square :: Int -> Int
square x = x * x

testSquare :: Int
testSquare = square 5
-- Result: 25

-- ================================================================
-- Expressions with Mixed Literals
-- ================================================================

-- Multiple integer literals in expression
testMultipleLiterals :: Int
testMultipleLiterals = 1 + 2 + 3 + 4 + 5
-- Result: 15

-- Nested arithmetic
testNestedArithmetic :: Int
testNestedArithmetic = (10 + 5) * 2
-- Result: 30

-- ================================================================
-- Character and String Literals (non-overloaded)
-- ================================================================

-- Char literal has fixed type
testCharLiteral :: Char
testCharLiteral = 'a'

-- String literal has fixed type
testStringLiteral :: String
testStringLiteral = "hello"

-- ================================================================
-- Functions with Explicit Type Annotations
-- ================================================================

-- Explicit Int annotation
addInts :: Int -> Int -> Int
addInts x y = x + y

testAddInts :: Int
testAddInts = addInts 100 200
-- Result: 300

-- ================================================================
-- Let Bindings with Numeric Literals
-- ================================================================

-- Let with integer literal
testLetInt :: Int
testLetInt = let x = 10 in x + x
-- Result: 20

-- Nested let with literals
testNestedLet :: Int
testNestedLet = let a = 5
                    b = 7
                in a * b
-- Result: 35

-- ================================================================
-- Case Expressions with Numeric Results
-- ================================================================

-- Case returning numeric literals
boolToInt :: Bool -> Int
boolToInt b = case b of
  True  -> 1
  False -> 0

testBoolToIntTrue :: Int
testBoolToIntTrue = boolToInt True
-- Result: 1

testBoolToIntFalse :: Int
testBoolToIntFalse = boolToInt False
-- Result: 0

-- ================================================================
-- Conditionals with Numeric Results
-- ================================================================

-- If-then-else with integer branches
maxInt :: Int -> Int -> Int
maxInt x y = if x > y then x else y

testMaxInt :: Int
testMaxInt = maxInt 10 20
-- Result: 20

-- ================================================================
-- List with Numeric Elements
-- ================================================================

-- List of integers
testIntList :: [Int]
testIntList = [1, 2, 3, 4, 5]

-- Sum of list (using explicit recursion)
sumList :: [Int] -> Int
sumList xs = case xs of
  []     -> 0
  (y:ys) -> y + sumList ys

testSumList :: Int
testSumList = sumList [1, 2, 3, 4, 5]
-- Result: 15

-- ================================================================
-- Tuple with Numeric Elements
-- ================================================================

-- Tuple with integers
testIntTuple :: (Int, Int)
testIntTuple = (10, 20)

-- Extract and compute
sumPair :: (Int, Int) -> Int
sumPair p = case p of
  (a, b) -> a + b

testSumPair :: Int
testSumPair = sumPair (15, 27)
-- Result: 42

-- ================================================================
-- Main function to run all tests
-- ================================================================

main :: IO ()
main = do
  -- Literals
  putStrLn "=== Literal tests ==="
  print testIntLiteral      -- Expected: 42
  print testNegIntLiteral   -- Expected: -17
  print testFloatLiteral    -- Expected: 3.14
  print testNegFloatLiteral -- Expected: -2.5

  -- Arithmetic
  putStrLn "=== Arithmetic tests ==="
  print testAddInt          -- Expected: 30
  print testSubInt          -- Expected: 35
  print testMulInt          -- Expected: 42
  print testDivFloat        -- Expected: 2.5

  -- Polymorphic
  putStrLn "=== Polymorphic tests ==="
  print testDouble          -- Expected: 42
  print testSquare          -- Expected: 25

  -- Mixed
  putStrLn "=== Mixed literal tests ==="
  print testMultipleLiterals    -- Expected: 15
  print testNestedArithmetic    -- Expected: 30

  -- Explicit types
  putStrLn "=== Explicit type tests ==="
  print testAddInts         -- Expected: 300

  -- Let bindings
  putStrLn "=== Let binding tests ==="
  print testLetInt          -- Expected: 20
  print testNestedLet       -- Expected: 35

  -- Case expressions
  putStrLn "=== Case tests ==="
  print testBoolToIntTrue   -- Expected: 1
  print testBoolToIntFalse  -- Expected: 0

  -- Conditionals
  putStrLn "=== Conditional tests ==="
  print testMaxInt          -- Expected: 20

  -- Lists
  putStrLn "=== List tests ==="
  print testSumList         -- Expected: 15

  -- Tuples
  putStrLn "=== Tuple tests ==="
  print testSumPair         -- Expected: 42

  putStrLn "=== All type class tests completed ==="
