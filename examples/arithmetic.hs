-- arithmetic.hs
-- Demonstrates basic arithmetic operations
--
-- Compile: bhc arithmetic.hs -o arithmetic
-- Run:     ./arithmetic

module Main where

main :: IO ()
main = print (1 + 2 * 3)

-- Expected output: 7
