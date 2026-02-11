import qualified Data.Set as Set

main :: IO ()
main = do
    let s1 = Set.fromList [3, 1, 4, 1, 5, 9]
    putStrLn (show (Set.size s1))
    putStrLn (show (Set.member 4 s1))
    putStrLn (show (Set.member 7 s1))
    let s2 = Set.insert 7 s1
    putStrLn (show (Set.size s2))
    let s3 = Set.delete 3 s1
    putStrLn (show (Set.size s3))
    let s4 = Set.fromList [1, 2, 3]
    let s5 = Set.fromList [3, 4, 5]
    putStrLn (show (Set.size (Set.union s4 s5)))
    putStrLn (show (Set.size (Set.intersection s4 s5)))
    putStrLn (show (Set.size (Set.difference s4 s5)))
    let evens = Set.filter even s1
    putStrLn (show (Set.size evens))
    let total = Set.foldr (+) 0 s1
    putStrLn (show total)
    putStrLn "done"
