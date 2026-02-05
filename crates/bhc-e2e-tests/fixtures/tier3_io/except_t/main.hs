main = do
    result1 <- runExceptT successComp
    case result1 of
        Left e  -> putStrLn ("Left: " ++ e)
        Right a -> putStrLn ("Right: " ++ a)
    result2 <- runExceptT failComp
    case result2 of
        Left e  -> putStrLn ("Left: " ++ e)
        Right a -> putStrLn ("Right: " ++ a)

successComp = catchE (return "success") (\_ -> return "caught")

failComp = do
    throwE "failed"
    return "unreachable"
