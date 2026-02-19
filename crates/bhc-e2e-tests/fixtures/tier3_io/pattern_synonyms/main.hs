{-# LANGUAGE PatternSynonyms #-}
module Main where

data Expr = Lit Int | Add Expr Expr | Neg Expr

pattern Zero = Lit 0
pattern One  = Lit 1
pattern Succ n = Add n One

isZero :: Expr -> String
isZero e = case e of
  Zero -> "yes"
  _    -> "no"

eval :: Expr -> Int
eval e = case e of
  Lit n   -> n
  Add a b -> eval a + eval b
  Neg a   -> negate (eval a)

main :: IO ()
main = do
  putStrLn (isZero Zero)
  putStrLn (isZero One)
  putStrLn (show (eval (Succ (Succ Zero))))
