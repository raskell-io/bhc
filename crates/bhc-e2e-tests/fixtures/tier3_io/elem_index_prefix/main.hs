module Main where

showBoolResult :: Bool -> String
showBoolResult b = if b then "True" else "False"

main :: IO ()
main = do
    let mi = elemIndex 3 [1, 2, 3, 4, 5]
    putStrLn (show (fromMaybe (-1) mi))
    let mi2 = elemIndex 9 [1, 2, 3]
    putStrLn (show (fromMaybe (-1) mi2))
    let fi = findIndex even [1, 3, 4, 7]
    putStrLn (show (fromMaybe (-1) fi))
    putStrLn (showBoolResult (isPrefixOf [1, 2] [1, 2, 3]))
    putStrLn (showBoolResult (isPrefixOf [1, 3] [1, 2, 3]))
    putStrLn (showBoolResult (isSuffixOf [2, 3] [1, 2, 3]))
    putStrLn (showBoolResult (isInfixOf [2, 3] [1, 2, 3, 4]))
    putStrLn "done"
