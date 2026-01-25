-- hello.hs
-- The classic Hello World program
--
-- Compile: bhc hello.hs -o hello
-- Run:     ./hello

module Main where

main :: IO ()
main = putStrLn "Hello from BHC!"
