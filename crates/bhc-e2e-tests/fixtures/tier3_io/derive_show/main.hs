data Color = Red | Green | Blue
  deriving (Show)

data Shape = Circle Int | Rectangle Int Int
  deriving (Show)

main :: IO ()
main = do
    putStrLn (show Red)
    putStrLn (show Green)
    putStrLn (show Blue)
    putStrLn (show (Circle 5))
    putStrLn (show (Rectangle 3 4))
