module Main where
main :: IO ()
main = do
  putStrLn (show (min 3 5))
  putStrLn (show (max 3 5))
  putStrLn (show (subtract 3 10))
  putStrLn (show (take 5 (enumFrom 1)))
  putStrLn (show (take 5 (enumFromThen 1 3)))
  putStrLn (show (enumFromThenTo 1 3 11))
  putStrLn (show (foldl1 (+) [1, 2, 3, 4, 5]))
  putStrLn (show (foldr1 (+) [1, 2, 3, 4, 5]))
  putStrLn (show (until (> 100) (* 2) 1))
