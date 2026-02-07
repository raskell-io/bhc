module Main where

main :: IO ()
main = do
    putStrLn (show (any even [1, 2, 3]))
    putStrLn (show (any even [1, 3, 5]))
    putStrLn (show (all even [2, 4, 6]))
    putStrLn (show (all even [2, 3, 6]))
    putStrLn "done"
