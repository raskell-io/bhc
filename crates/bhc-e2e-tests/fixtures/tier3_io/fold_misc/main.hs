module Main where
myMax :: Int -> Int -> Int
myMax x y = max x y
main :: IO ()
main = do
  putStrLn (show (foldl1 myMax [3, 1, 4, 1, 5, 9, 2, 6]))
  putStrLn (show (foldr1 myMax [3, 1, 4, 1, 5, 9, 2, 6]))
  putStrLn (show (map (min 5) [1, 3, 5, 7, 9]))
  putStrLn (show (map (max 5) [1, 3, 5, 7, 9]))
  putStrLn (show (map (subtract 1) [10, 20, 30]))
