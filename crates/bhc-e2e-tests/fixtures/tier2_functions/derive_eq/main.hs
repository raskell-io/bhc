data Color = Red | Green | Blue
  deriving (Eq)

main :: IO ()
main = do
    putStrLn (if Red == Red then "yes" else "no")
    putStrLn (if Red == Blue then "yes" else "no")
    putStrLn (if Green /= Blue then "yes" else "no")
