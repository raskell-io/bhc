{-# LANGUAGE ViewPatterns #-}
module Main where

-- Simple view function
double :: Int -> Int
double x = x * 2

-- View pattern in function argument: apply double, match literal result
showDoubled :: Int -> String
showDoubled (double -> 10) = "five"
showDoubled (double -> 20) = "ten"
showDoubled _ = "other"

-- View pattern with constructor result (Maybe)
safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x

showFirst :: [Int] -> String
showFirst (safeHead -> Just x) = show x
showFirst _ = "empty"

main :: IO ()
main = do
  putStrLn (showDoubled 5)
  putStrLn (showDoubled 10)
  putStrLn (showDoubled 3)
  putStrLn (showFirst [42, 1, 2])
  putStrLn (showFirst [])
