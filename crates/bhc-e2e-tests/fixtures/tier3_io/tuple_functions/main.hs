module Main where

add :: Int -> Int -> Int
add x y = x + y

main :: IO ()
main = do
  putStrLn (show (fst (10, 20)))
  putStrLn (show (snd (10, 20)))

  -- swap
  let p = (1, 2)
  let sp = swap p
  putStrLn (show (fst sp))
  putStrLn (show (snd sp))

  -- curry: curry f x y = f (x, y)
  putStrLn (show (curry (\p -> fst p + snd p) 3 4))

  -- uncurry: uncurry f (x, y) = f x y
  putStrLn (show (uncurry add (5, 6)))

  -- map fst / map snd on list of pairs
  let pairs = [(1, 10), (2, 20), (3, 30)]
  putStrLn (show (map fst pairs))
  putStrLn (show (map snd pairs))

  -- map swap on list of pairs
  let swapped = map swap pairs
  putStrLn (show (map fst swapped))
  putStrLn (show (map snd swapped))
