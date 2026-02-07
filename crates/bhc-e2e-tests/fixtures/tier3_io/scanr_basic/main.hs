main :: IO ()
main = do
    let xs = scanr (+) 0 [1, 2, 3, 4]
    mapM_ (\x -> putStrLn (show x)) xs
    let ys = scanl1 (+) [1, 2, 3, 4]
    mapM_ (\x -> putStrLn (show x)) ys
    putStrLn "done"
