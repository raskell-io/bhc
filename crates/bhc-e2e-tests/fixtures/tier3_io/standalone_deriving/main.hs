{-# LANGUAGE StandaloneDeriving #-}
module Main where

data Color = Red | Green | Blue
deriving instance Show Color
deriving instance Eq Color

main :: IO ()
main = do
  putStrLn (show Red)
  putStrLn (show Green)
  putStrLn (if Blue == Blue then "equal" else "not equal")
  putStrLn (if Red == Green then "equal" else "not equal")
