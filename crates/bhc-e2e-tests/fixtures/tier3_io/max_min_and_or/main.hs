module Main where

main :: IO ()
main = do
    putStrLn (show (maximum [3, 1, 4, 1, 5, 9, 2, 6]))
    putStrLn (show (minimum [3, 1, 4, 1, 5, 9, 2, 6]))
    let b1 = and [True, True, True]
    putStrLn (if b1 then "True" else "False")
    let b2 = and [True, False, True]
    putStrLn (if b2 then "True" else "False")
    let b3 = or [False, False, True]
    putStrLn (if b3 then "True" else "False")
    let b4 = or [False, False, False]
    putStrLn (if b4 then "True" else "False")
    putStrLn "done"
