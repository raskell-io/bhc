module Main where

main :: IO ()
main = do
  putStrLn (takeFileName "/usr/local/bin/ghc")
  putStrLn (takeDirectory "/usr/local/bin/ghc")
  putStrLn (takeExtension "report.tar.gz")
  putStrLn (dropExtension "report.tar.gz")
  putStrLn (takeBaseName "/path/to/file.txt")
  putStrLn (replaceExtension "file.txt" ".md")
  putStrLn (show (isAbsolute "/usr/bin"))
  putStrLn (show (isRelative "foo/bar"))
  putStrLn (show (hasExtension "file.txt"))
  putStrLn (show (hasExtension "README"))
  let pair = splitExtension "archive.tar.gz"
  putStrLn (fst pair)
  putStrLn (snd pair)
  putStrLn ("/" </> "usr" </> "local")
