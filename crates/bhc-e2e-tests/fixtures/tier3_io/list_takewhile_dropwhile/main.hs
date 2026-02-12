module Main where

main :: IO ()
main = do
  -- takeWhile
  putStrLn (show (takeWhile even [2, 4, 6, 1, 3]))
  putStrLn (show (takeWhile odd [1, 3, 5, 2, 4]))
  putStrLn (show (length (takeWhile even [1, 2, 3])))

  -- dropWhile
  putStrLn (show (dropWhile even [2, 4, 6, 1, 3]))
  putStrLn (show (dropWhile odd [1, 3, 5, 2, 4]))
  putStrLn (show (length (dropWhile even [])))
