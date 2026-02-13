{-# LANGUAGE TupleSections #-}
module Main where

main :: IO ()
main = do
  -- Basic tuple section: (,x) applies to an argument
  let f = (,10)
  let pair = f 42
  print (fst pair)
  print (snd pair)

  -- Tuple section with first element provided: (x,)
  let g = (10,)
  let pair2 = g 20
  print (fst pair2)
  print (snd pair2)

  -- Using tuple sections with map
  let tagged = map (,"hello") [1, 2, 3]
  print (fst (head tagged))
  putStrLn (snd (head tagged))
  print (length tagged)
