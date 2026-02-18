{-# LANGUAGE DeriveFoldable #-}
module Main where

data Box a = Box a deriving Foldable
data Pair a = Pair a a deriving Foldable
data Maybe2 a = Nothing2 | Just2 a deriving Foldable

unbox :: Box a -> a
unbox (Box x) = x

main :: IO ()
main = do
    putStrLn (show (foldr (+) 0 (Box 42)))
    putStrLn (show (foldr (+) 0 (Pair 3 4)))
    putStrLn (show (foldr (+) 10 Nothing2))
    putStrLn (show (foldr (+) 10 (Just2 5)))
    putStrLn (show (foldr (:) [] (Pair 1 2)))
    putStrLn (show (length (foldr (:) [] (Box 99))))
