main :: IO ()
main = do
    let xs = intersect [1, 2, 3, 4, 5] [2, 4, 6]
    mapM_ (\x -> putStrLn (show x)) xs
    putStrLn (show (length (intersect [1, 2, 3] [4, 5, 6])))
    putStrLn "done"
