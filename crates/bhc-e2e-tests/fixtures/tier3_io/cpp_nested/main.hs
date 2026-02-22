{-# LANGUAGE CPP #-}
module Main where

main :: IO ()
main = do
#ifdef darwin_HOST_OS
#if __GLASGOW_HASKELL__ >= 900
  putStrLn "macOS with modern GHC"
#else
  putStrLn "macOS with old GHC"
#endif
#else
  putStrLn "not macOS"
#endif
