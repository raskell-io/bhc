-- Test: pattern-matching
-- Category: semantic
-- Profile: default
-- Expected: success
-- Spec: H26-SPEC Section 3.4

{-# HASKELL_EDITION 2026 #-}

module PatternMatchingTest where

-- Simple patterns
test1 :: Int
test1 =
  let (x, y) = (1, 2)
  in x + y  -- Result: 3

-- List patterns
test2 :: Int
test2 = case [1, 2, 3] of
  []        -> 0
  [x]       -> x
  [x, y]    -> x + y
  (x:y:_)   -> x + y  -- Matches: 1 + 2 = 3

-- As-patterns
test3 :: ([Int], Int)
test3 = case [1, 2, 3] of
  xs@(x:_) -> (xs, x)  -- Returns ([1,2,3], 1)
  []       -> ([], 0)

-- View patterns (H26 feature)
test4 :: Int
test4 =
  let f (view reverse -> [3, 2, 1]) = 1
      f _                           = 0
  in f [1, 2, 3]  -- Result: 1

-- Guard patterns
test5 :: String
test5 = case 42 of
  n | n < 0     -> "negative"
    | n == 0    -> "zero"
    | n < 100   -> "small"
    | otherwise -> "large"
-- Result: "small"

-- Pattern guards with let
test6 :: Maybe Int
test6 = case Just 10 of
  Just x | let y = x * 2, y > 15 -> Just y
  _                              -> Nothing
-- Result: Just 20

-- Lazy patterns
test7 :: Int
test7 =
  let ~(x, y) = error "not evaluated"
  in 42  -- Result: 42 (pattern not forced)

-- Strict patterns (bang patterns)
test8 :: Int
test8 =
  let f !x = x + 1
  in f 41  -- Result: 42

-- Record patterns
data Point = Point { px :: Int, py :: Int }

test9 :: Int
test9 = case Point 3 4 of
  Point { px = x, py = y } -> x + y  -- Result: 7

-- Record wildcards
test10 :: Int
test10 = case Point 3 4 of
  Point {..} -> px + py  -- Result: 7 (fields bound by wildcard)

-- Nested patterns
data Tree a = Leaf a | Node (Tree a) (Tree a)

test11 :: Int
test11 = case Node (Leaf 1) (Node (Leaf 2) (Leaf 3)) of
  Leaf x                    -> x
  Node (Leaf x) (Leaf y)    -> x + y
  Node (Leaf x) (Node _ _)  -> x + 100  -- Matches: 1 + 100 = 101
  _                         -> 0

-- Literal patterns
test12 :: String
test12 = case 'a' of
  'a' -> "letter a"
  'b' -> "letter b"
  _   -> "other"
-- Result: "letter a"

-- String patterns
test13 :: Int
test13 = case "hello" of
  "hello" -> 1
  "world" -> 2
  _       -> 0
-- Result: 1

-- Numeric patterns
test14 :: String
test14 = case 3.14 of
  0.0 -> "zero"
  1.0 -> "one"
  _   -> "other"
-- Result: "other"

-- ================================================================
-- Tests for builtin ADT types (for LLVM codegen validation)
-- ================================================================

-- Maybe patterns
testMaybe1 :: Int
testMaybe1 = case Just 42 of
  Nothing -> 0
  Just x  -> x
-- Result: 42

testMaybe2 :: Int
testMaybe2 = case Nothing of
  Nothing -> 1
  Just _  -> 0
-- Result: 1

testMaybe3 :: Int
testMaybe3 =
  let fromMaybe def mx = case mx of
        Nothing -> def
        Just x  -> x
  in fromMaybe 0 (Just 100)
-- Result: 100

-- Either patterns
testEither1 :: Int
testEither1 = case Left 10 of
  Left x  -> x
  Right _ -> 0
-- Result: 10

testEither2 :: Int
testEither2 = case Right 20 of
  Left _  -> 0
  Right y -> y
-- Result: 20

testEither3 :: Int
testEither3 =
  let either f g e = case e of
        Left a  -> f a
        Right b -> g b
  in either (\x -> x + 1) (\y -> y * 2) (Right 15)
-- Result: 30

-- Bool patterns (explicit case)
testBool1 :: Int
testBool1 = case True of
  True  -> 1
  False -> 0
-- Result: 1

testBool2 :: Int
testBool2 = case False of
  True  -> 1
  False -> 0
-- Result: 0

-- List patterns with constructors
testList1 :: Int
testList1 = case [] of
  []    -> 0
  _:_   -> 1
-- Result: 0

testList2 :: Int
testList2 = case [1, 2, 3] of
  []     -> 0
  (x:xs) -> x + length xs
  where length [] = 0
        length (_:ys) = 1 + length ys
-- Result: 1 + 2 = 3

-- Nested Maybe patterns
testNestedMaybe :: Int
testNestedMaybe = case Just (Just 42) of
  Nothing        -> 0
  Just Nothing   -> 1
  Just (Just x)  -> x
-- Result: 42

-- Nested Either patterns
testNestedEither :: Int
testNestedEither = case Right (Left 5) of
  Left _           -> 0
  Right (Left x)   -> x
  Right (Right _)  -> 1
-- Result: 5

-- Combined Maybe and Either
testMaybeEither :: Int
testMaybeEither = case Just (Left 10) of
  Nothing          -> 0
  Just (Left x)    -> x
  Just (Right _)   -> 1
-- Result: 10

-- Unit pattern
testUnit :: Int
testUnit = case () of
  () -> 42
-- Result: 42

-- Tuple with Maybe
testTupleMaybe :: Int
testTupleMaybe = case (Just 1, Just 2) of
  (Nothing, _)       -> 0
  (_, Nothing)       -> 0
  (Just x, Just y)   -> x + y
-- Result: 3

-- Default/wildcard patterns
testDefault1 :: Int
testDefault1 = case Just 5 of
  _ -> 42
-- Result: 42

testDefault2 :: Int
testDefault2 = case Left "error" of
  Left _ -> 1
  _      -> 0
-- Result: 1

-- ================================================================
-- Main function to run all tests (for LLVM codegen testing)
-- ================================================================

main :: IO ()
main = do
  -- Basic ADT patterns
  print testMaybe1      -- Expected: 42
  print testMaybe2      -- Expected: 1
  print testMaybe3      -- Expected: 100
  print testEither1     -- Expected: 10
  print testEither2     -- Expected: 20
  print testEither3     -- Expected: 30
  print testBool1       -- Expected: 1
  print testBool2       -- Expected: 0
  print testList1       -- Expected: 0
  print testList2       -- Expected: 3

  -- Nested patterns
  print testNestedMaybe    -- Expected: 42
  print testNestedEither   -- Expected: 5
  print testMaybeEither    -- Expected: 10
  print testUnit           -- Expected: 42
  print testTupleMaybe     -- Expected: 3

  -- Default patterns
  print testDefault1   -- Expected: 42
  print testDefault2   -- Expected: 1
