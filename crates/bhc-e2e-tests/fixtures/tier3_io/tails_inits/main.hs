module Main where

main :: IO ()
main = do
    let ts = tails [1, 2, 3]
    putStrLn (show (length ts))
    let is = inits [1, 2, 3]
    putStrLn (show (length is))
    putStrLn (show (maximum [3, 1, 4, 1, 5]))
    putStrLn (show (minimum [3, 1, 4, 1, 5]))
    putStrLn "done"
