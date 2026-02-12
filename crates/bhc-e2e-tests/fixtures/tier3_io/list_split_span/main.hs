module Main where

showFstList :: ([Int], [Int]) -> String
showFstList p = show (take 100 (fst p))

showSndList :: ([Int], [Int]) -> String
showSndList p = show (take 100 (snd p))

main :: IO ()
main = do
  -- splitAt
  let p1 = splitAt 3 [1, 2, 3, 4, 5]
  putStrLn (showFstList p1)
  putStrLn (showSndList p1)

  -- span: takes while predicate is true
  let p2 = span even [2, 4, 5, 6, 7]
  putStrLn (showFstList p2)
  putStrLn (showSndList p2)

  -- break: takes while predicate is false (opposite of span)
  let p3 = break even [1, 3, 4, 6, 7]
  putStrLn (showFstList p3)
  putStrLn (showSndList p3)

  -- splitAt edge cases
  let p4 = splitAt 0 [1, 2, 3]
  putStrLn (show (length (fst p4)))
  putStrLn (show (length (snd p4)))

  let p5 = splitAt 10 [1, 2]
  putStrLn (showFstList p5)
  putStrLn (showSndList p5)
