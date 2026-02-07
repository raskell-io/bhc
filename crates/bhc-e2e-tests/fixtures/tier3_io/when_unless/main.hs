module Main where

main :: IO ()
main = do
    when True (putStrLn "when-true")
    when False (putStrLn "when-false")
    unless False (putStrLn "unless-false")
    unless True (putStrLn "unless-true")
    putStrLn "done"
