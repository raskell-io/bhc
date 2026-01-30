main = catch (putStrLn "safe action") (\e -> putStrLn ("caught: " ++ e))
