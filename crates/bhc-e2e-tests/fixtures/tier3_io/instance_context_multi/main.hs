{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

class Describable a where
    describe :: a -> String

class Sizeable a where
    size :: a -> Int

data Color = Red | Green | Blue

instance Describable Color where
    describe Red   = "red"
    describe Green = "green"
    describe Blue  = "blue"

instance Sizeable Color where
    size _ = 1

data Pair a = MkPair a a

instance (Describable a, Sizeable a) => Describable (Pair a) where
    describe (MkPair x y) = "pair(" ++ describe x ++ "," ++ describe y ++ ",size=" ++ show (size x + size y) ++ ")"

main :: IO ()
main = do
    putStrLn (describe (MkPair Red Blue))
    putStrLn (describe (MkPair Green Green))
