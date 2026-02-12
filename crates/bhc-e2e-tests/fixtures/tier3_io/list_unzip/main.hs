module Main where

main :: IO ()
main = do
  let pairs = [(1, 10), (2, 20), (3, 30)]
  let result = unzip pairs
  putStrLn (show (take 100 (fst result)))
  putStrLn (show (take 100 (snd result)))

  -- empty list
  let empty = unzip ([] :: [(Int, Int)])
  putStrLn (show (length (fst empty)))
  putStrLn (show (length (snd empty)))
