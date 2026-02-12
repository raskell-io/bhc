module Main where

main :: IO ()
main = do
  putStrLn (show (succ 5))
  putStrLn (show (pred 5))
  putStrLn (show (succ 0))
  putStrLn (show (pred 0))

  -- (&) reverse application: (&) x f = f x
  putStrLn (show ((&) 10 succ))

  -- map succ / map pred on lists
  putStrLn (show (map succ [1, 2, 3]))
  putStrLn (show (map pred [10, 20, 30]))
