main = putStr "hello " >>= \_ -> putStrLn "world" >>= \_ -> putStrLn "done"
