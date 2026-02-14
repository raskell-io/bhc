module Main where

class MyFunctor f where
    myMap :: (a -> b) -> f a -> f b

data Box a = Box a
instance MyFunctor Box where
    myMap f (Box x) = Box (f x)

main :: IO ()
main = do
    let b = myMap (+1) (Box 42)
    case b of Box x -> print x
    let b2 = myMap (*10) (Box 5)
    case b2 of Box x -> print x
