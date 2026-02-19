{-# LANGUAGE StrictData #-}
module Main where

-- With StrictData, all fields are implicitly strict (no ! needed)
data Pair = Pair Int Int

data Config = Config
  { host :: String
  , port :: Int
  }

getFirst :: Pair -> Int
getFirst (Pair a _) = a

getSecond :: Pair -> Int
getSecond (Pair _ b) = b

main :: IO ()
main = do
  let p = Pair 10 20
  putStrLn (show (getFirst p))
  putStrLn (show (getSecond p))
  putStrLn (show (getFirst p + getSecond p))
  let cfg = Config { host = "localhost", port = 8080 }
  putStrLn (host cfg)
  putStrLn (show (port cfg))
