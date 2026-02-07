module Main where

printAndDouble :: Int -> IO Int
printAndDouble x = do
    putStrLn (show x)
    return (x * 2)

main :: IO ()
main = do
    results <- mapM printAndDouble [1, 2, 3]
    putStrLn (show (length results))
    mapM_ (\x -> putStrLn (show x)) [10, 20, 30]
    putStrLn "done"
