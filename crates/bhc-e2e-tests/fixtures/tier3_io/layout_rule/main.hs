module Main where

-- Test 1: Module-level where with multiple top-level declarations
double :: Int -> Int
double x = x + x

triple :: Int -> Int
triple x = x + x + x

-- Test 2: Function where clause with multiple bindings
circleStats :: Int -> Int
circleStats r = area + circumference
  where
    area = r * r * 3
    circumference = 2 * 3 * r

-- Test 3: Nested where (where inside where)
nestedWhere :: Int -> Int
nestedWhere x = outer
  where
    outer = inner + 10
      where
        inner = x * 2

-- Test 4: case..of with multiple alternatives
classify :: Int -> Int
classify n = case n of
  0 -> 100
  1 -> 200
  _ -> 300

-- Test 5: Guards with multiple clauses
guardTest :: Int -> Int
guardTest x
  | x > 0     = 1
  | x < 0     = 2
  | otherwise  = 3

-- Test 6: let..in expression with multiple bindings
letTest :: Int -> Int
letTest x =
  let a = x + 1
      b = x + 2
  in a + b

-- Test 7: do block with let bindings (no in)
doLetTest :: IO ()
doLetTest = do
  let msg = "do-let works"
  putStrLn msg

-- Test 8: Nested do blocks (do inside do)
nestedDo :: IO ()
nestedDo = do
  putStrLn "outer-start"
  do
    putStrLn "inner"
  putStrLn "outer-end"

-- Test 9: Multi-line type signature (no VirtualSemi between lines)
longSig :: Int
        -> Int
        -> Int
longSig x y = x + y

-- Test 10: Class declaration with indented methods
class Describable a where
  describe :: a -> Int

data Color = Red | Green | Blue

instance Describable Color where
  describe Red   = 10
  describe Green = 20
  describe Blue  = 30

-- Test 11: Per-clause where blocks
perClauseWhere :: Int -> Int
perClauseWhere 0 = base
  where base = 42
perClauseWhere n = n + offset
  where offset = 100

-- Test 12: Continuation lines for operators
continuationOp :: Int -> Int
continuationOp x = x
  + 10
  + 20

main :: IO ()
main = do
  -- Test 1: Multiple top-level decls via layout
  putStrLn (show (double 5))
  putStrLn (show (triple 5))
  -- Test 2: where clause
  putStrLn (show (circleStats 10))
  -- Test 3: nested where
  putStrLn (show (nestedWhere 7))
  -- Test 4: case..of
  putStrLn (show (classify 0))
  putStrLn (show (classify 1))
  putStrLn (show (classify 99))
  -- Test 5: guards
  putStrLn (show (guardTest 5))
  putStrLn (show (guardTest (-3)))
  putStrLn (show (guardTest 0))
  -- Test 6: let..in
  putStrLn (show (letTest 10))
  -- Test 7: do + let
  doLetTest
  -- Test 8: nested do
  nestedDo
  -- Test 9: multi-line type sig
  putStrLn (show (longSig 3 4))
  -- Test 10: class/instance
  putStrLn (show (describe Red))
  putStrLn (show (describe Green))
  putStrLn (show (describe Blue))
  -- Test 11: per-clause where
  putStrLn (show (perClauseWhere 0))
  putStrLn (show (perClauseWhere 5))
  -- Test 12: continuation operators
  putStrLn (show (continuationOp 100))
