import qualified Data.IntMap as IntMap
import qualified Data.IntSet as IntSet

main :: IO ()
main = do
    let is1 = IntSet.fromList [5, 3, 1, 4, 2]
    putStrLn (show (IntSet.size is1))
    putStrLn (show (IntSet.member 3 is1))
    putStrLn (show (IntSet.member 9 is1))
    let evens = IntSet.filter even is1
    putStrLn (show (IntSet.size evens))
    let total = IntSet.foldr (+) 0 is1
    putStrLn (show total)
    let im1 = IntMap.fromList [(1, 10), (2, 20), (3, 30)]
    putStrLn (show (IntMap.size im1))
    putStrLn (show (IntMap.member 2 im1))
    let im2 = IntMap.insert 4 40 im1
    putStrLn (show (IntMap.size im2))
    let im3 = IntMap.delete 1 im2
    putStrLn (show (IntMap.size im3))
    putStrLn "done"
