{-# LANGUAGE MultiParamTypeClasses #-}
module Main where

class Combinable a b where
    combine :: a -> b -> String

data Color = Red | Green | Blue
data Shape = Circle | Square

instance Combinable Color Shape where
    combine c s = case c of
        Red   -> case s of
            Circle -> "red circle"
            Square -> "red square"
        Green -> case s of
            Circle -> "green circle"
            Square -> "green square"
        Blue  -> case s of
            Circle -> "blue circle"
            Square -> "blue square"

main :: IO ()
main = do
    putStrLn (combine Red Circle)
    putStrLn (combine Green Square)
    putStrLn (combine Blue Circle)
