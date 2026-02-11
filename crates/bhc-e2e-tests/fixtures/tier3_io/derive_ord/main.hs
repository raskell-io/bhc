data Color = Red | Green | Blue
  deriving (Eq, Ord, Show)

data Priority = Low | Medium | High
  deriving (Eq, Ord)

main :: IO ()
main = do
    -- compare on enums
    putStrLn (show (compare Red Blue))
    putStrLn (show (compare Green Green))
    putStrLn (show (compare Blue Red))
    -- comparison operators on enums
    putStrLn (if Red < Blue then "yes" else "no")
    putStrLn (if Blue <= Blue then "yes" else "no")
    putStrLn (if Blue > Red then "yes" else "no")
    putStrLn (if Red >= Green then "yes" else "no")
    -- compare different type
    putStrLn (if Low < High then "yes" else "no")
    putStrLn (if High > Medium then "yes" else "no")
