main = do
    pair <- runWriterT computation
    putStrLn (fst pair)
    putStrLn (snd pair)

computation = do
    tell "hello "
    tell "world"
    return "done"
