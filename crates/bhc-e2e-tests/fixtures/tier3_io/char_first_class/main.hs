module Main where

main :: IO ()
main = do
    -- Char predicates as first-class functions with filter
    putStrLn (filter isAlpha "Hello World 123")
    putStrLn (filter isDigit "abc123def456")
    putStrLn (filter isUpper "Hello World")
    putStrLn (filter isLower "Hello World")
    putStrLn (filter isAscii "hello")
    putStrLn (filter isLetter "abc-123-def")

    -- Char predicates with any/all
    print (any isDigit "hello123")
    print (any isDigit "hello")
    print (all isAlpha "hello")
    print (all isAlpha "hello123")

    -- Char conversions as first-class functions with map
    putStrLn (map toLower "HELLO WORLD")
    putStrLn (map toUpper "hello world")

    -- chr as first-class function (produces String)
    putStrLn (map chr [72, 101, 108, 108, 111])

    -- Combined: filter then map
    putStrLn (map toUpper (filter isAlpha "Hello World"))
    print (length (filter isDigit "a1b2c3d4e5"))

    -- ord and digitToInt used directly (not first-class) for verification
    print (ord 'A')
    print (digitToInt '9')
