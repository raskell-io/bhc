{-# LANGUAGE GADTs #-}

data Key a where
  IntKey  :: Int -> Key Int
  StrKey  :: String -> Key String

showKey :: Key Int -> Int
showKey (IntKey n) = n

main :: IO ()
main = do
  let k1 = IntKey 42
  let k2 = IntKey 99
  print (showKey k1)
  print (showKey k2)
  print (showKey (IntKey (showKey k1 + showKey k2)))
