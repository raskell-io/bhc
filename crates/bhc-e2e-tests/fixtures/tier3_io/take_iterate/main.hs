main :: IO ()
main = do
    let xs = take 5 (iterate (* 2) 1)
    mapM_ (\x -> putStrLn (show x)) xs
    let ys = take 4 (repeat 42)
    putStrLn (show (length ys))
    let zs = take 7 (cycle [1, 2, 3])
    mapM_ (\x -> putStrLn (show x)) zs
    putStrLn "done"
