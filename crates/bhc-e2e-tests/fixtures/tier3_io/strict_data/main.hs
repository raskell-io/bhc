{-# LANGUAGE StrictData #-}
data Pair a b = Pair a b

main :: IO ()
main = do
  let p = Pair (1 + 2) (3 + 4)
  case p of
    Pair x y -> putStrLn (show (x + y))
