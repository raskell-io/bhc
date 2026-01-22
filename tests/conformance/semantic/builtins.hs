-- Test: builtins
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 3.7

{-# HASKELL_EDITION 2026 #-}

module BuiltinsTest where

-- ================================================================
-- List Operations
-- ================================================================

-- head: first element of a list
testHead1 :: Int
testHead1 = head [1, 2, 3]
-- Result: 1

testHead2 :: Int
testHead2 = head [42]
-- Result: 42

-- tail: rest of a list
testTailLength :: Int
testTailLength = length (tail [1, 2, 3, 4, 5])
-- Result: 4

testTailHead :: Int
testTailHead = head (tail [1, 2, 3])
-- Result: 2

-- null: check if list is empty
testNull1 :: Bool
testNull1 = null []
-- Result: True

testNull2 :: Bool
testNull2 = null [1, 2, 3]
-- Result: False

testNull3 :: Bool
testNull3 = null (tail [42])
-- Result: True

-- length: count elements
testLength1 :: Int
testLength1 = length []
-- Result: 0

testLength2 :: Int
testLength2 = length [1, 2, 3, 4, 5]
-- Result: 5

testLength3 :: Int
testLength3 = length [[1], [2, 3], [4, 5, 6]]
-- Result: 3

-- ================================================================
-- Tuple Operations
-- ================================================================

-- fst: first element of a pair
testFst1 :: Int
testFst1 = fst (1, 2)
-- Result: 1

testFst2 :: Int
testFst2 = fst (42, "hello")
-- Result: 42

-- snd: second element of a pair
testSnd1 :: Int
testSnd1 = snd (1, 2)
-- Result: 2

testSnd2 :: Int
testSnd2 = snd (100, 200)
-- Result: 200

-- Combined tuple operations
testTupleOps :: Int
testTupleOps = fst (10, 20) + snd (10, 20)
-- Result: 30

-- ================================================================
-- Maybe Operations
-- ================================================================

-- fromJust: extract value from Just
testFromJust1 :: Int
testFromJust1 = fromJust (Just 42)
-- Result: 42

testFromJust2 :: Int
testFromJust2 = fromJust (Just 0)
-- Result: 0

-- isJust: check if Maybe is Just
testIsJust1 :: Bool
testIsJust1 = isJust (Just 42)
-- Result: True

testIsJust2 :: Bool
testIsJust2 = isJust Nothing
-- Result: False

-- isNothing: check if Maybe is Nothing
testIsNothing1 :: Bool
testIsNothing1 = isNothing Nothing
-- Result: True

testIsNothing2 :: Bool
testIsNothing2 = isNothing (Just 42)
-- Result: False

-- Combined Maybe operations
testMaybeOps :: Int
testMaybeOps =
  if isJust (Just 10)
    then fromJust (Just 10)
    else 0
-- Result: 10

-- ================================================================
-- Either Operations
-- ================================================================

-- isLeft: check if Either is Left
testIsLeft1 :: Bool
testIsLeft1 = isLeft (Left 42)
-- Result: True

testIsLeft2 :: Bool
testIsLeft2 = isLeft (Right 42)
-- Result: False

-- isRight: check if Either is Right
testIsRight1 :: Bool
testIsRight1 = isRight (Right 42)
-- Result: True

testIsRight2 :: Bool
testIsRight2 = isRight (Left 42)
-- Result: False

-- Combined Either operations
testEitherOps :: Int
testEitherOps =
  let e = Right 100
  in if isRight e then 1 else 0
-- Result: 1

-- ================================================================
-- Identity and Const
-- ================================================================

-- id: identity function
testId1 :: Int
testId1 = id 42
-- Result: 42

testId2 :: Bool
testId2 = id True
-- Result: True

-- const: constant function
testConst1 :: Int
testConst1 = const 42 100
-- Result: 42

testConst2 :: Int
testConst2 = const 1 "ignored"
-- Result: 1

-- ================================================================
-- Seq
-- ================================================================

-- seq: force evaluation of first arg, return second
testSeq1 :: Int
testSeq1 = seq 1 42
-- Result: 42

testSeq2 :: Int
testSeq2 = seq (1 + 1) 100
-- Result: 100

-- ================================================================
-- Combining Builtins
-- ================================================================

-- Using multiple builtins together
testCombined1 :: Int
testCombined1 = length (tail [1, 2, 3, 4, 5])
-- Result: 4

testCombined2 :: Int
testCombined2 = head (tail (tail [1, 2, 3]))
-- Result: 3

testCombined3 :: Int
testCombined3 = fst (head [(1, 2), (3, 4)])
-- Result: 1

testCombined4 :: Bool
testCombined4 = null (tail (tail (tail [1, 2, 3])))
-- Result: True

-- ================================================================
-- Helper functions using builtins
-- ================================================================

-- Safe head using Maybe
safeHead :: [a] -> Maybe a
safeHead xs = case null xs of
  True  -> Nothing
  False -> Just (head xs)

testSafeHead1 :: Bool
testSafeHead1 = isJust (safeHead [1, 2, 3])
-- Result: True

testSafeHead2 :: Bool
testSafeHead2 = isNothing (safeHead [])
-- Result: True

-- Last element using recursion with builtins
last' :: [a] -> a
last' xs = case null (tail xs) of
  True  -> head xs
  False -> last' (tail xs)

testLast :: Int
testLast = last' [1, 2, 3, 4, 5]
-- Result: 5

-- Init (all but last) using recursion
init' :: [a] -> [a]
init' xs = case null (tail xs) of
  True  -> []
  False -> head xs : init' (tail xs)

testInitLength :: Int
testInitLength = length (init' [1, 2, 3, 4, 5])
-- Result: 4

-- ================================================================
-- List construction helpers (for testing)
-- ================================================================

-- Replicate using recursion
replicate' :: Int -> a -> [a]
replicate' n x = case n of
  0 -> []
  _ -> x : replicate' (n - 1) x

testReplicateLength :: Int
testReplicateLength = length (replicate' 10 42)
-- Result: 10

-- Take first n elements
take' :: Int -> [a] -> [a]
take' n xs = case n of
  0 -> []
  _ -> case null xs of
    True  -> []
    False -> head xs : take' (n - 1) (tail xs)

testTakeLength :: Int
testTakeLength = length (take' 3 [1, 2, 3, 4, 5])
-- Result: 3

-- Drop first n elements
drop' :: Int -> [a] -> [a]
drop' n xs = case n of
  0 -> xs
  _ -> case null xs of
    True  -> []
    False -> drop' (n - 1) (tail xs)

testDropLength :: Int
testDropLength = length (drop' 2 [1, 2, 3, 4, 5])
-- Result: 3

-- ================================================================
-- Main function to run all tests
-- ================================================================

main :: IO ()
main = do
  -- List operations
  print testHead1        -- Expected: 1
  print testHead2        -- Expected: 42
  print testTailLength   -- Expected: 4
  print testTailHead     -- Expected: 2
  print testNull1        -- Expected: True (1)
  print testNull2        -- Expected: False (0)
  print testNull3        -- Expected: True (1)
  print testLength1      -- Expected: 0
  print testLength2      -- Expected: 5
  print testLength3      -- Expected: 3

  -- Tuple operations
  print testFst1         -- Expected: 1
  print testFst2         -- Expected: 42
  print testSnd1         -- Expected: 2
  print testSnd2         -- Expected: 200
  print testTupleOps     -- Expected: 30

  -- Maybe operations
  print testFromJust1    -- Expected: 42
  print testFromJust2    -- Expected: 0
  print testIsJust1      -- Expected: True (1)
  print testIsJust2      -- Expected: False (0)
  print testIsNothing1   -- Expected: True (1)
  print testIsNothing2   -- Expected: False (0)
  print testMaybeOps     -- Expected: 10

  -- Either operations
  print testIsLeft1      -- Expected: True (1)
  print testIsLeft2      -- Expected: False (0)
  print testIsRight1     -- Expected: True (1)
  print testIsRight2     -- Expected: False (0)
  print testEitherOps    -- Expected: 1

  -- Identity and const
  print testId1          -- Expected: 42
  print testId2          -- Expected: True (1)
  print testConst1       -- Expected: 42
  print testConst2       -- Expected: 1

  -- Seq
  print testSeq1         -- Expected: 42
  print testSeq2         -- Expected: 100

  -- Combined
  print testCombined1    -- Expected: 4
  print testCombined2    -- Expected: 3
  print testCombined3    -- Expected: 1
  print testCombined4    -- Expected: True (1)

  -- Safe operations
  print testSafeHead1    -- Expected: True (1)
  print testSafeHead2    -- Expected: True (1)
  print testLast         -- Expected: 5
  print testInitLength   -- Expected: 4

  -- List helpers
  print testReplicateLength -- Expected: 10
  print testTakeLength   -- Expected: 3
  print testDropLength   -- Expected: 3
