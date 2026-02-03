describe n = case n of
  0 -> "zero"
  1 -> "one"
  _ -> "other"

main = putStrLn (describe 0) >> putStrLn (describe 1) >> putStrLn (describe 42)
