module Main where

main :: IO ()
main = do
  evens <- filterM (\x -> do { putStrLn (show x); return (even x) }) [1,2,3,4,5]
  putStrLn (show (length evens))
  result <- foldM (\acc x -> do { putStrLn (show acc); return (acc + x) }) 0 [1,2,3]
  putStrLn (show result)
  foldM_ (\acc x -> do { putStrLn (show (acc + x)); return (acc + x) }) 0 [10,20,30]
  replicateM_ 3 (putStrLn "hello")
  putStrLn "done"
