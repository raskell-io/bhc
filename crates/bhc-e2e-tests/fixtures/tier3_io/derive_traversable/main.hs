{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
module Main where

data Box a = Box a deriving (Functor, Foldable, Traversable)
data Pair a = Pair a a deriving (Functor, Foldable, Traversable)
data Maybe2 a = Nothing2 | Just2 a deriving (Functor, Foldable, Traversable)

unbox :: Box a -> a
unbox (Box x) = x

pairFst :: Pair a -> a
pairFst (Pair x _) = x

pairSnd :: Pair a -> a
pairSnd (Pair _ y) = y

getJust2 :: Maybe2 a -> a
getJust2 (Just2 x) = x
getJust2 Nothing2 = error "empty"

double :: Int -> IO Int
double x = do
    putStrLn (show x)
    return (x * 2)

main :: IO ()
main = do
    r1 <- traverse double (Box 21)
    putStrLn (show (unbox r1))
    r2 <- traverse double (Pair 3 4)
    putStrLn (show (pairFst r2))
    putStrLn (show (pairSnd r2))
    r3 <- traverse double (Just2 5)
    putStrLn (show (getJust2 r3))
    r4 <- mapM double [10, 20]
    putStrLn (show (foldr (+) 0 r4))
