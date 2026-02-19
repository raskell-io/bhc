{-# LANGUAGE DeriveGeneric #-}
module Main where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData, deepseq, rnf, force)

data Color = Red | Green | Blue
  deriving (Show, Generic)

data Pair a b = Pair a b
  deriving (Show, Generic)

instance NFData Color
instance NFData a => NFData (Pair a b)

main :: IO ()
main = do
  -- Test force (identity in strict runtime)
  putStrLn (show (force Red))
  -- Test deepseq (evaluates first arg, returns second)
  deepseq Green (putStrLn "deepseq works")
  -- Test show on derived ADT
  putStrLn (show (Pair 1 2))
