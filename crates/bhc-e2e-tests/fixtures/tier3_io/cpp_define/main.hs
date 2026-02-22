{-# LANGUAGE CPP #-}
module Main where

#define GREETING "Hello from CPP"

main :: IO ()
main = putStrLn GREETING
