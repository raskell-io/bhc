module Main where

main :: IO ()
main = do
  createDirectory "test_e19_dir"
  putStrLn "created dir"
  writeFile "test_e19_dir/hello.txt" "world"
  copyFile "test_e19_dir/hello.txt" "test_e19_dir/copy.txt"
  renameFile "test_e19_dir/copy.txt" "test_e19_dir/renamed.txt"
  removeFile "test_e19_dir/hello.txt"
  removeFile "test_e19_dir/renamed.txt"
  removeDirectory "test_e19_dir"
  putStrLn "done"
