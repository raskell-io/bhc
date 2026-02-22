{-# LANGUAGE CPP #-}
module Main where

main :: IO ()
main = do
#if __GLASGOW_HASKELL__ >= 900
  putStrLn "GHC 9+"
#else
  putStrLn "GHC 8"
#endif
