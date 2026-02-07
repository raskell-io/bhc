main :: IO ()
main = do
    let triples = zip3 [1, 2, 3] [10, 20, 30] [100, 200, 300]
    putStrLn (show (length triples))
    let sums = zipWith3 (\a b c -> a + b + c) [1, 2] [10, 20] [100, 200]
    mapM_ (\x -> putStrLn (show x)) sums
    putStrLn "done"
