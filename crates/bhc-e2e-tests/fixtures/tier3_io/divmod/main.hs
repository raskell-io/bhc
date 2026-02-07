main :: IO ()
main = do
    let dm1 = divMod 7 3
    putStrLn (show (fst dm1))
    putStrLn (show (snd dm1))
    let qr1 = quotRem 7 3
    putStrLn (show (fst qr1))
    putStrLn (show (snd qr1))
    let dm2 = divMod (-7) 3
    putStrLn (show (fst dm2))
    putStrLn (show (snd dm2))
    let qr2 = quotRem (-7) 3
    putStrLn (show (fst qr2))
    putStrLn (show (snd qr2))
