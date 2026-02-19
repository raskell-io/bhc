{-# LANGUAGE GADTs #-}

data Expr a where
  LitInt  :: Int -> Expr Int
  LitBool :: Bool -> Expr Bool
  Add     :: Expr Int -> Expr Int -> Expr Int
  If      :: Expr Bool -> Expr a -> Expr a -> Expr a

eval :: Expr Int -> Int
eval (LitInt n)  = n
eval (Add x y)   = eval x + eval y
eval (If c t e)  = if evalBool c then eval t else eval e

evalBool :: Expr Bool -> Bool
evalBool (LitBool b) = b

main :: IO ()
main = do
  print (eval (LitInt 42))
  print (eval (Add (LitInt 1) (LitInt 2)))
  print (eval (Add (LitInt 10) (Add (LitInt 20) (LitInt 30))))
  print (eval (If (LitBool True) (LitInt 10) (LitInt 20)))
  print (eval (If (LitBool False) (LitInt 100) (LitInt 200)))
