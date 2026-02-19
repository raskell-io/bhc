{-# LANGUAGE PatternGuards #-}
module Main where

-- Simple guards (should already work)
classify :: Int -> String
classify n
  | n < 0     = "negative"
  | n == 0    = "zero"
  | otherwise = "positive"

-- Pattern guards: bind in guard position
lookup' :: Int -> [(Int, String)] -> String
lookup' _ [] = "not found"
lookup' k ((k', v) : rest)
  | k == k'   = v
  | otherwise  = lookup' k rest

main :: IO ()
main = do
  putStrLn (classify 5)
  putStrLn (classify 0)
  putStrLn (classify (-3))
  let items = [(1, "one"), (2, "two"), (3, "three")]
  putStrLn (lookup' 2 items)
  putStrLn (lookup' 4 items)
