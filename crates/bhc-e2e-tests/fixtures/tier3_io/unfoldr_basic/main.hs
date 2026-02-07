countDown :: Int -> Maybe (Int, Int)
countDown 0 = Nothing
countDown n = Just (n, n - 1)

main :: IO ()
main = do
    let xs = unfoldr countDown 5
    mapM_ (\x -> putStrLn (show x)) xs
    putStrLn "done"
