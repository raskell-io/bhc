-- let-bindings.hs
-- Demonstrates let expressions
--
-- Compile: bhc let-bindings.hs -o let-bindings
-- Run:     ./let-bindings

module Main where

main :: IO ()
main = print result
  where
    result = let x = 10
                 y = 20
             in x + y

-- Expected output: 30
