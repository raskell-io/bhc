-- |
-- Module      : BHC.Data.Maybe
-- Description : The Maybe type and related operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- The 'Maybe' type encapsulates an optional value.

module BHC.Data.Maybe (
    Maybe(..),
    
    -- * Querying
    maybe,
    isJust,
    isNothing,
    
    -- * Extraction
    fromMaybe,
    fromJust,
    
    -- * Conversion
    listToMaybe,
    maybeToList,
    catMaybes,
    mapMaybe,
) where

import BHC.Prelude (Maybe(..), Bool(..), maybe)

-- | /O(1)/. Returns 'True' iff the argument is 'Just'.
--
-- >>> isJust (Just 3)
-- True
-- >>> isJust Nothing
-- False
isJust :: Maybe a -> Bool
isJust (Just _) = True
isJust Nothing  = False

-- | /O(1)/. Returns 'True' iff the argument is 'Nothing'.
--
-- >>> isNothing Nothing
-- True
-- >>> isNothing (Just 3)
-- False
isNothing :: Maybe a -> Bool
isNothing Nothing = True
isNothing _       = False

-- | /O(1)/. Extract the value with a default for 'Nothing'.
--
-- >>> fromMaybe 0 (Just 5)
-- 5
-- >>> fromMaybe 0 Nothing
-- 0
fromMaybe :: a -> Maybe a -> a
fromMaybe d Nothing  = d
fromMaybe _ (Just x) = x

-- | /O(1)/. Extract the value from 'Just'.
--
-- __Warning__: Partial function. Throws an error on 'Nothing'.
-- Prefer 'fromMaybe' or pattern matching.
fromJust :: Maybe a -> a
fromJust (Just x) = x
fromJust Nothing  = error "Maybe.fromJust: Nothing"

-- | /O(1)/. Return the first element of a list, or 'Nothing' if empty.
--
-- >>> listToMaybe [1, 2, 3]
-- Just 1
-- >>> listToMaybe []
-- Nothing
listToMaybe :: [a] -> Maybe a
listToMaybe []    = Nothing
listToMaybe (x:_) = Just x

-- | /O(1)/. Convert 'Maybe' to a singleton or empty list.
--
-- >>> maybeToList (Just 5)
-- [5]
-- >>> maybeToList Nothing
-- []
maybeToList :: Maybe a -> [a]
maybeToList Nothing  = []
maybeToList (Just x) = [x]

-- | /O(n)/. Extract all 'Just' values from a list.
--
-- >>> catMaybes [Just 1, Nothing, Just 3]
-- [1, 3]
catMaybes :: [Maybe a] -> [a]
catMaybes = mapMaybe id

-- | /O(n)/. Map a function that may fail and collect successes.
-- Combines 'map' and 'filter' in a single pass.
--
-- >>> mapMaybe (\x -> if even x then Just (x `div` 2) else Nothing) [1..6]
-- [1, 2, 3]
mapMaybe :: (a -> Maybe b) -> [a] -> [b]
mapMaybe _ []     = []
mapMaybe f (x:xs) = case f x of
    Nothing -> mapMaybe f xs
    Just y  -> y : mapMaybe f xs
