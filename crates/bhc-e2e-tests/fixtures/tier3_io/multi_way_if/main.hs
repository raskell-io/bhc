{-# LANGUAGE MultiWayIf #-}
module Main where

classify :: Int -> String
classify n = if | n < 0     -> "negative"
                | n == 0    -> "zero"
                | n < 10    -> "small"
                | n < 100   -> "medium"
                | otherwise -> "large"

abs' :: Int -> Int
abs' n = if | n < 0     -> negate n
            | otherwise -> n

main :: IO ()
main = do
  putStrLn (classify (-5))
  putStrLn (classify 0)
  putStrLn (classify 7)
  putStrLn (classify 42)
  putStrLn (classify 100)
  print (abs' (-3))
  print (abs' 5)
