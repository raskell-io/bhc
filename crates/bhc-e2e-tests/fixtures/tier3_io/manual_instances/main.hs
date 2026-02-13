module Main where

data Color = Red | Green | Blue

instance Show Color where
    show Red   = "Red"
    show Green = "Green"
    show Blue  = "Blue"

data Shape = Circle Int | Rectangle Int Int

instance Show Shape where
    show (Circle r)      = "Circle " ++ show r
    show (Rectangle w h) = "Rectangle " ++ show w ++ " " ++ show h

main :: IO ()
main = do
    -- Manual Show instances
    putStrLn (show Red)
    putStrLn (show Green)
    putStrLn (show Blue)
    putStrLn (show (Circle 5))
    putStrLn (show (Rectangle 3 4))
