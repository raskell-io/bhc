{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

class Describable a where
    describe :: a -> String

data Color = Red | Green | Blue

instance Describable Color where
    describe Red   = "color:red"
    describe Green = "color:green"
    describe Blue  = "color:blue"

data Wrapper a = Wrap a

instance Describable a => Describable (Wrapper a) where
    describe (Wrap x) = "Wrap(" ++ describe x ++ ")"

main :: IO ()
main = do
    putStrLn (describe Red)
    putStrLn (describe (Wrap Green))
