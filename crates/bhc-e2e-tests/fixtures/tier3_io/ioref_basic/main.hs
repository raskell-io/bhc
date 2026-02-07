main :: IO ()
main = do
    ref <- newIORef 0
    writeIORef ref 42
    val <- readIORef ref
    putStrLn (show val)
    modifyIORef ref (\x -> x + 8)
    val2 <- readIORef ref
    putStrLn (show val2)
