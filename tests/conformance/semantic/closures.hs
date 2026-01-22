-- Test: closures
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 3.6

{-# HASKELL_EDITION 2026 #-}

module ClosuresTest where

-- ================================================================
-- Simple Closures (capturing one variable)
-- ================================================================

-- A function that returns a closure capturing 'n'
addN :: Int -> Int -> Int
addN n = \x -> x + n

testAddN1 :: Int
testAddN1 = addN 10 5
-- Result: 15

testAddN2 :: Int
testAddN2 = addN 100 1
-- Result: 101

-- Function returning closure used directly
testAddNDirect :: Int
testAddNDirect = (\x -> x + 5) 10
-- Result: 15

-- ================================================================
-- Closures Capturing Multiple Variables
-- ================================================================

-- Closure capturing two variables
makeLinear :: Int -> Int -> Int -> Int
makeLinear a b = \x -> a * x + b

testLinear1 :: Int
testLinear1 = makeLinear 2 3 5
-- Result: 2 * 5 + 3 = 13

testLinear2 :: Int
testLinear2 = makeLinear 1 10 20
-- Result: 1 * 20 + 10 = 30

-- Closure capturing three variables
makeQuadratic :: Int -> Int -> Int -> Int -> Int
makeQuadratic a b c = \x -> a * x * x + b * x + c

testQuadratic :: Int
testQuadratic = makeQuadratic 1 0 0 5
-- Result: 1 * 25 + 0 + 0 = 25

-- ================================================================
-- Nested Closures
-- ================================================================

-- Closure that returns another closure
makeAdder :: Int -> (Int -> Int)
makeAdder n = \x -> x + n

applyAdder :: Int
applyAdder = (makeAdder 5) 3
-- Result: 8

-- Multi-level nesting
nestedClosures :: Int -> Int -> Int -> Int
nestedClosures a b c = a + b + c

testNested :: Int
testNested = nestedClosures 1 2 3
-- Result: 6

-- ================================================================
-- Higher-Order Functions with Closures
-- ================================================================

-- Apply a function twice
applyTwice :: (Int -> Int) -> Int -> Int
applyTwice f x = f (f x)

testApplyTwice :: Int
testApplyTwice = applyTwice (\x -> x + 1) 0
-- Result: 2

-- Apply with captured variable
testApplyTwiceClosure :: Int
testApplyTwiceClosure = applyTwice (addN 5) 0
-- Result: 10

-- ================================================================
-- Closures in Let Bindings
-- ================================================================

testLetClosure :: Int
testLetClosure =
  let multiplier = 3
      multiply = \x -> x * multiplier
  in multiply 7
-- Result: 21

testNestedLetClosure :: Int
testNestedLetClosure =
  let a = 2
      b = 3
      combine = \x -> a * x + b
  in combine 10
-- Result: 23

-- ================================================================
-- Map with Closures
-- ================================================================

-- Simple map implementation
map' :: (a -> b) -> [a] -> [b]
map' f xs = case xs of
  []     -> []
  (y:ys) -> f y : map' f ys

-- Sum a list
sumList :: [Int] -> Int
sumList xs = case xs of
  []     -> 0
  (y:ys) -> y + sumList ys

-- Map with a closure
testMapClosure :: Int
testMapClosure =
  let increment = 10
  in sumList (map' (\x -> x + increment) [1, 2, 3])
-- Result: (1+10) + (2+10) + (3+10) = 36

-- ================================================================
-- Filter with Closures
-- ================================================================

filter' :: (a -> Bool) -> [a] -> [a]
filter' p xs = case xs of
  []     -> []
  (y:ys) -> case p y of
    True  -> y : filter' p ys
    False -> filter' p ys

lengthList :: [a] -> Int
lengthList xs = case xs of
  []     -> 0
  (_:ys) -> 1 + lengthList ys

-- Filter with a closure capturing threshold
testFilterClosure :: Int
testFilterClosure =
  let threshold = 5
  in lengthList (filter' (\x -> x > threshold) [1, 2, 6, 7, 8, 3])
-- Result: 3 (elements > 5: 6, 7, 8)

-- ================================================================
-- Fold with Closures
-- ================================================================

foldl' :: (b -> a -> b) -> b -> [a] -> b
foldl' f acc xs = case xs of
  []     -> acc
  (y:ys) -> foldl' f (f acc y) ys

-- Fold with a closure
testFoldClosure :: Int
testFoldClosure =
  let multiplier = 2
  in foldl' (\acc x -> acc + x * multiplier) 0 [1, 2, 3]
-- Result: 0 + 1*2 + 2*2 + 3*2 = 12

-- ================================================================
-- Compose with Closures
-- ================================================================

compose :: (b -> c) -> (a -> b) -> a -> c
compose f g x = f (g x)

testCompose :: Int
testCompose =
  let double = \x -> x * 2
      addFive = \x -> x + 5
  in compose double addFive 10
-- Result: (10 + 5) * 2 = 30

-- ================================================================
-- Closures with Pattern Matching
-- ================================================================

maybeClosure :: Maybe Int -> Int
maybeClosure mx =
  let default_val = 100
      extract = \m -> case m of
        Nothing -> default_val
        Just x  -> x
  in extract mx

testMaybeClosureJust :: Int
testMaybeClosureJust = maybeClosure (Just 42)
-- Result: 42

testMaybeClosureNothing :: Int
testMaybeClosureNothing = maybeClosure Nothing
-- Result: 100

-- ================================================================
-- Partial Application
-- ================================================================

add3 :: Int -> Int -> Int -> Int
add3 x y z = x + y + z

testPartialApplication1 :: Int
testPartialApplication1 =
  let addTo10 = add3 10
  in addTo10 5 3
-- Result: 18

testPartialApplication2 :: Int
testPartialApplication2 =
  let addTo10And5 = add3 10 5
  in addTo10And5 7
-- Result: 22

-- ================================================================
-- Closures Escaping Scope
-- ================================================================

-- Returns a closure that captures local binding
makeCounter :: Int -> (Int -> Int)
makeCounter start =
  let current = start
  in \step -> current + step

testEscapingClosure :: Int
testEscapingClosure =
  let counter = makeCounter 100
  in counter 5
-- Result: 105

-- ================================================================
-- Main function to run all tests
-- ================================================================

main :: IO ()
main = do
  -- Simple closures
  print testAddN1           -- Expected: 15
  print testAddN2           -- Expected: 101
  print testAddNDirect      -- Expected: 15

  -- Multiple captures
  print testLinear1         -- Expected: 13
  print testLinear2         -- Expected: 30
  print testQuadratic       -- Expected: 25

  -- Nested closures
  print applyAdder          -- Expected: 8
  print testNested          -- Expected: 6

  -- Higher-order
  print testApplyTwice      -- Expected: 2
  print testApplyTwiceClosure -- Expected: 10

  -- Let bindings
  print testLetClosure      -- Expected: 21
  print testNestedLetClosure -- Expected: 23

  -- Map/filter/fold
  print testMapClosure      -- Expected: 36
  print testFilterClosure   -- Expected: 3
  print testFoldClosure     -- Expected: 12

  -- Compose
  print testCompose         -- Expected: 30

  -- Pattern matching
  print testMaybeClosureJust    -- Expected: 42
  print testMaybeClosureNothing -- Expected: 100

  -- Partial application
  print testPartialApplication1 -- Expected: 18
  print testPartialApplication2 -- Expected: 22

  -- Escaping scope
  print testEscapingClosure -- Expected: 105
