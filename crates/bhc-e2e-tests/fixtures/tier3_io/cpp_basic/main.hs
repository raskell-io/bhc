{-# LANGUAGE CPP #-}
module Main where

main :: IO ()
main = do
#ifdef darwin_HOST_OS
  putStrLn "macOS"
#elif defined(linux_HOST_OS)
  putStrLn "Linux"
#elif defined(mingw32_HOST_OS)
  putStrLn "Windows"
#else
  putStrLn "Unknown"
#endif
