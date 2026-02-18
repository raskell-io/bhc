{-# LANGUAGE DeriveFunctor #-}
module Main where

data Box a = Box a deriving Functor
data Pair a = Pair a a deriving Functor

unbox :: Box a -> a
unbox (Box x) = x

getPairFst :: Pair a -> a
getPairFst (Pair x _) = x

getPairSnd :: Pair a -> a
getPairSnd (Pair _ y) = y

main :: IO ()
main = do
    putStrLn (show (unbox (fmap (+1) (Box 42))))
    let p = fmap (*2) (Pair 3 4)
    putStrLn (show (getPairFst p))
    putStrLn (show (getPairSnd p))
    putStrLn (show (fromMaybe 0 (fmap (+10) (Just 5))))
    putStrLn (show (fromMaybe 0 (fmap (+10) Nothing)))
    putStrLn (show (fmap (*3) [1,2,3]))
