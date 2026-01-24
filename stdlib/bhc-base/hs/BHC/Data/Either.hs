-- |
-- Module      : BHC.Data.Either
-- Description : The Either type for error handling
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The 'Either' type represents values with two possibilities.

module BHC.Data.Either (
    Either(..),
    
    -- * Case analysis
    either,
    
    -- * Querying
    isLeft,
    isRight,
    
    -- * Extraction
    fromLeft,
    fromRight,
    
    -- * Partitioning
    lefts,
    rights,
    partitionEithers,
) where

import BHC.Prelude (Either(..), Bool(..), either)

-- | /O(1)/. Return 'True' iff the argument is 'Left'.
--
-- >>> isLeft (Left "error")
-- True
-- >>> isLeft (Right 42)
-- False
isLeft :: Either a b -> Bool
isLeft (Left _) = True
isLeft _        = False

-- | /O(1)/. Return 'True' iff the argument is 'Right'.
--
-- >>> isRight (Right 42)
-- True
-- >>> isRight (Left "error")
-- False
isRight :: Either a b -> Bool
isRight (Right _) = True
isRight _         = False

-- | /O(1)/. Extract from 'Left' with a default for 'Right'.
--
-- >>> fromLeft "default" (Left "actual")
-- "actual"
-- >>> fromLeft "default" (Right 42)
-- "default"
fromLeft :: a -> Either a b -> a
fromLeft _ (Left x) = x
fromLeft d _        = d

-- | /O(1)/. Extract from 'Right' with a default for 'Left'.
--
-- >>> fromRight 0 (Right 42)
-- 42
-- >>> fromRight 0 (Left "error")
-- 0
fromRight :: b -> Either a b -> b
fromRight _ (Right x) = x
fromRight d _         = d

-- | /O(n)/. Extract all 'Left' values from a list.
--
-- >>> lefts [Left 1, Right "a", Left 2, Right "b"]
-- [1,2]
lefts :: [Either a b] -> [a]
lefts = foldr go []
  where go (Left x)  acc = x : acc
        go (Right _) acc = acc

-- | /O(n)/. Extract all 'Right' values from a list.
--
-- >>> rights [Left 1, Right "a", Left 2, Right "b"]
-- ["a","b"]
rights :: [Either a b] -> [b]
rights = foldr go []
  where go (Left _)  acc = acc
        go (Right x) acc = x : acc

-- | /O(n)/. Partition a list of 'Either' into 'Left' and 'Right' values.
--
-- >>> partitionEithers [Left 1, Right "a", Left 2, Right "b"]
-- ([1,2],["a","b"])
partitionEithers :: [Either a b] -> ([a], [b])
partitionEithers = foldr go ([], [])
  where go (Left x)  (ls, rs) = (x:ls, rs)
        go (Right x) (ls, rs) = (ls, x:rs)
