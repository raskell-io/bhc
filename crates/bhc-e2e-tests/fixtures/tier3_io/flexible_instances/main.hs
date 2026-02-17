{-# LANGUAGE FlexibleInstances #-}
module Main where

class Describable a where
    describe :: a -> String

data Color = Red | Green | Blue

instance Describable Color where
    describe Red   = "color:red"
    describe Green = "color:green"
    describe Blue  = "color:blue"

data Shape = Circle | Square

instance Describable Shape where
    describe Circle = "shape:circle"
    describe Square = "shape:square"

greet :: Describable a => a -> String
greet x = "Hello, " ++ describe x ++ "!"

main :: IO ()
main = do
    putStrLn (greet Red)
    putStrLn (greet Blue)
    putStrLn (describe Green)
    putStrLn (greet Circle)
    putStrLn (describe Square)
