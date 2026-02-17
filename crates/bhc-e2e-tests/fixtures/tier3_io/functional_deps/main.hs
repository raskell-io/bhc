{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}
module Main where

class Extract a b | a -> b where
    extract :: a -> b

data Wrapper = Wrapper Int

instance Extract Wrapper Int where
    extract w = case w of
        Wrapper n -> n

-- Without fundeps, `extract w` would have ambiguous return type `b`
-- With fundep `a -> b`, knowing `a = Wrapper` determines `b = Int`
double :: Wrapper -> Int
double w = extract w + extract w

main :: IO ()
main = do
    putStrLn (show (extract (Wrapper 42)))
    putStrLn (show (double (Wrapper 99)))
