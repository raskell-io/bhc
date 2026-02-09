module Main where

main :: IO ()
main = do
  results <- zipWithM (\a b -> do { putStrLn (show (a + b)); return (a + b) }) [1,2,3] [10,20,30]
  putStrLn (show (length results))
  zipWithM_ (\a b -> putStrLn (show (a * b))) [2,3,4] [5,6,7]
  xs <- replicateM 3 (do { putStrLn "action"; return 42 })
  putStrLn (show (length xs))
  putStrLn "done"
