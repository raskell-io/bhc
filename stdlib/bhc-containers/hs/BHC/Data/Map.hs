-- |
-- Module      : BHC.Data.Map
-- Description : Ordered maps from keys to values
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- An efficient implementation of ordered maps from keys to values.

module BHC.Data.Map (
    -- * Map type
    Map,
    
    -- * Construction
    empty, singleton, fromList, fromListWith, fromListWithKey,
    
    -- * Insertion
    insert, insertWith, insertWithKey, insertLookupWithKey,
    
    -- * Deletion/Update
    delete, adjust, adjustWithKey, update, updateWithKey,
    alter, alterF,
    
    -- * Query
    lookup, (!?), (!), findWithDefault,
    member, notMember,
    null, size,
    
    -- * Combine
    union, unionWith, unionWithKey,
    unions, unionsWith,
    difference, differenceWith, differenceWithKey,
    intersection, intersectionWith, intersectionWithKey,
    
    -- * Traversal
    map, mapWithKey, traverseWithKey,
    mapAccum, mapAccumWithKey,
    mapKeys, mapKeysWith, mapKeysMonotonic,
    
    -- * Folds
    foldr, foldl, foldrWithKey, foldlWithKey,
    foldMapWithKey,
    
    -- * Conversion
    elems, keys, assocs, keysSet,
    toList, toAscList, toDescList,
    
    -- * Filter
    filter, filterWithKey,
    partition, partitionWithKey,
    mapMaybe, mapMaybeWithKey,
    mapEither, mapEitherWithKey,
    
    -- * Submap
    isSubmapOf, isSubmapOfBy,
    isProperSubmapOf, isProperSubmapOfBy,
    
    -- * Min/Max
    lookupMin, lookupMax,
    findMin, findMax,
    deleteMin, deleteMax,
    
    -- * Split
    split, splitLookup,
) where

import BHC.Prelude hiding (map, lookup, null, filter, foldr, foldl)
import qualified BHC.Prelude as P
import qualified BHC.Data.Set as Set

-- | A map from keys @k@ to values @a@.
data Map k a
    = Tip
    | Bin {-# UNPACK #-} !Int !k a !(Map k a) !(Map k a)
    deriving (Eq, Ord, Show, Read)

instance Functor (Map k) where
    fmap = map

instance Foldable (Map k) where
    foldr = foldr
    foldl = foldl
    null = null
    length = size

instance Traversable (Map k) where
    traverse f = traverseWithKey (const f)

instance (Ord k, Semigroup a) => Semigroup (Map k a) where
    (<>) = unionWith (<>)

instance (Ord k, Semigroup a) => Monoid (Map k a) where
    mempty = empty

-- ------------------------------------------------------------
-- Construction
-- ------------------------------------------------------------

-- | /O(1)/. The empty map.
--
-- >>> empty
-- fromList []
empty :: Map k a
empty = Tip

-- | /O(1)/. A map with a single element.
--
-- >>> singleton "hello" 1
-- fromList [("hello",1)]
singleton :: k -> a -> Map k a
singleton k x = Bin 1 k x Tip Tip

-- | /O(n * log n)/. Build a map from a list of key-value pairs.
-- If the list contains duplicate keys, the last value is kept.
--
-- >>> fromList [("a", 1), ("b", 2), ("a", 3)]
-- fromList [("a",3),("b",2)]
--
-- See also 'fromListWith' for custom combining.
fromList :: Ord k => [(k, a)] -> Map k a
fromList = P.foldl' (\m (k, x) -> insert k x m) empty

-- | /O(n * log n)/. Build a map from a list with a combining function.
--
-- >>> fromListWith (+) [("a", 1), ("b", 2), ("a", 3)]
-- fromList [("a",4),("b",2)]
fromListWith :: Ord k => (a -> a -> a) -> [(k, a)] -> Map k a
fromListWith f = fromListWithKey (\_ x y -> f x y)

-- | /O(n * log n)/. Build a map from a list with a key-aware combining function.
--
-- >>> fromListWithKey (\k x y -> show k ++ ":" ++ show (x + y)) [("a", 1), ("a", 2)]
-- fromList [("a","a:3")]
fromListWithKey :: Ord k => (k -> a -> a -> a) -> [(k, a)] -> Map k a
fromListWithKey f = P.foldl' ins empty
  where ins m (k, x) = insertWithKey f k x m

-- ------------------------------------------------------------
-- Query
-- ------------------------------------------------------------

-- | /O(log n)/. Lookup a value at a key in the map.
--
-- >>> lookup "a" (fromList [("a", 1), ("b", 2)])
-- Just 1
-- >>> lookup "c" (fromList [("a", 1), ("b", 2)])
-- Nothing
lookup :: Ord k => k -> Map k a -> Maybe a
lookup = go
  where
    go _ Tip = Nothing
    go k (Bin _ kx x l r) = case compare k kx of
        LT -> go k l
        GT -> go k r
        EQ -> Just x

-- | /O(log n)/. Flipped version of 'lookup'.
--
-- >>> fromList [("a", 1)] !? "a"
-- Just 1
(!?) :: Ord k => Map k a -> k -> Maybe a
(!?) = flip lookup
infixl 9 !?

-- | /O(log n)/. Find the value at a key.
--
-- __Warning__: Partial function. Throws an error if the key is not found.
-- Prefer 'lookup' or 'findWithDefault'.
--
-- >>> fromList [("a", 1)] ! "a"
-- 1
(!) :: Ord k => Map k a -> k -> a
m ! k = case lookup k m of
    Just x  -> x
    Nothing -> error "Map.!: key not found"
infixl 9 !

-- | /O(log n)/. Lookup a key with a default value.
--
-- >>> findWithDefault 0 "a" (fromList [("a", 1)])
-- 1
-- >>> findWithDefault 0 "b" (fromList [("a", 1)])
-- 0
findWithDefault :: Ord k => a -> k -> Map k a -> a
findWithDefault def k m = case lookup k m of
    Just x  -> x
    Nothing -> def

-- | /O(log n)/. Is the key a member of the map?
--
-- >>> member "a" (fromList [("a", 1)])
-- True
-- >>> member "b" (fromList [("a", 1)])
-- False
member :: Ord k => k -> Map k a -> Bool
member k m = case lookup k m of
    Just _  -> True
    Nothing -> False

-- | /O(log n)/. Is the key not a member of the map?
--
-- >>> notMember "b" (fromList [("a", 1)])
-- True
notMember :: Ord k => k -> Map k a -> Bool
notMember k = not . member k

-- | /O(1)/. Is the map empty?
--
-- >>> null empty
-- True
-- >>> null (singleton "a" 1)
-- False
null :: Map k a -> Bool
null Tip = True
null _   = False

-- | /O(1)/. The number of elements in the map.
--
-- >>> size empty
-- 0
-- >>> size (fromList [("a", 1), ("b", 2)])
-- 2
size :: Map k a -> Int
size Tip            = 0
size (Bin n _ _ _ _) = n

-- ------------------------------------------------------------
-- Insertion
-- ------------------------------------------------------------

-- | /O(log n)/. Insert a new key-value pair. If the key already exists,
-- the new value replaces the old one.
--
-- >>> insert "c" 3 (fromList [("a", 1), ("b", 2)])
-- fromList [("a",1),("b",2),("c",3)]
-- >>> insert "a" 3 (fromList [("a", 1), ("b", 2)])
-- fromList [("a",3),("b",2)]
insert :: Ord k => k -> a -> Map k a -> Map k a
insert = insertWith const

-- | /O(log n)/. Insert with a combining function.
-- @insertWith f key value mp@ will insert @value@ if @key@ does not exist.
-- If @key@ exists with @oldValue@, it will insert @f value oldValue@.
--
-- >>> insertWith (+) "a" 3 (fromList [("a", 1)])
-- fromList [("a",4)]
insertWith :: Ord k => (a -> a -> a) -> k -> a -> Map k a -> Map k a
insertWith f = insertWithKey (\_ x y -> f x y)

-- | /O(log n)/. Insert with a key-aware combining function.
--
-- >>> insertWithKey (\k new old -> show k ++ ":" ++ show (new + old)) "a" 3 (fromList [("a", 1)])
-- fromList [("a","a:4")]
insertWithKey :: Ord k => (k -> a -> a -> a) -> k -> a -> Map k a -> Map k a
insertWithKey f k x = go
  where
    go Tip = singleton k x
    go (Bin sz ky y l r) = case compare k ky of
        LT -> balance ky y (go l) r
        GT -> balance ky y l (go r)
        EQ -> Bin sz k (f k x y) l r

-- | /O(log n)/. Combine insert with lookup of the old value.
--
-- >>> insertLookupWithKey (\_ new old -> new + old) "a" 3 (fromList [("a", 1)])
-- (Just 1,fromList [("a",4)])
-- >>> insertLookupWithKey (\_ new old -> new + old) "b" 3 (fromList [("a", 1)])
-- (Nothing,fromList [("a",1),("b",3)])
insertLookupWithKey :: Ord k => (k -> a -> a -> a) -> k -> a -> Map k a -> (Maybe a, Map k a)
insertLookupWithKey f k x = go
  where
    go Tip = (Nothing, singleton k x)
    go (Bin sz ky y l r) = case compare k ky of
        LT -> let (found, l') = go l in (found, balance ky y l' r)
        GT -> let (found, r') = go r in (found, balance ky y l r')
        EQ -> (Just y, Bin sz k (f k x y) l r)

-- ------------------------------------------------------------
-- Deletion/Update
-- ------------------------------------------------------------

-- | /O(log n)/. Delete a key and its value from the map.
-- If the key is not present, the original map is returned.
--
-- >>> delete "a" (fromList [("a", 1), ("b", 2)])
-- fromList [("b",2)]
-- >>> delete "c" (fromList [("a", 1), ("b", 2)])
-- fromList [("a",1),("b",2)]
delete :: Ord k => k -> Map k a -> Map k a
delete = go
  where
    go _ Tip = Tip
    go k (Bin _ kx x l r) = case compare k kx of
        LT -> balance kx x (go k l) r
        GT -> balance kx x l (go k r)
        EQ -> glue l r

-- | /O(log n)/. Adjust a value at a specific key with a function.
-- If the key is not present, the original map is returned.
--
-- >>> adjust (+10) "a" (fromList [("a", 1), ("b", 2)])
-- fromList [("a",11),("b",2)]
adjust :: Ord k => (a -> a) -> k -> Map k a -> Map k a
adjust f = adjustWithKey (\_ x -> f x)

-- | /O(log n)/. Adjust a value with access to the key.
--
-- >>> adjustWithKey (\k x -> show k ++ ":" ++ show x) "a" (fromList [("a", 1)])
-- fromList [("a","a:1")]
adjustWithKey :: Ord k => (k -> a -> a) -> k -> Map k a -> Map k a
adjustWithKey f = go
  where
    go _ Tip = Tip
    go k (Bin sx kx x l r) = case compare k kx of
        LT -> Bin sx kx x (go k l) r
        GT -> Bin sx kx x l (go k r)
        EQ -> Bin sx kx (f kx x) l r

-- | /O(log n)/. Update a value at a key. If the function returns 'Nothing',
-- the element is deleted.
--
-- >>> update (\x -> if x > 1 then Just (x * 10) else Nothing) "a" (fromList [("a", 1)])
-- fromList []
-- >>> update (\x -> if x > 0 then Just (x * 10) else Nothing) "a" (fromList [("a", 1)])
-- fromList [("a",10)]
update :: Ord k => (a -> Maybe a) -> k -> Map k a -> Map k a
update f = updateWithKey (\_ x -> f x)

-- | /O(log n)/. Update a value with access to the key.
--
-- >>> updateWithKey (\k x -> Just (show k ++ show x)) "a" (fromList [("a", 1)])
-- fromList [("a","a1")]
updateWithKey :: Ord k => (k -> a -> Maybe a) -> k -> Map k a -> Map k a
updateWithKey f = go
  where
    go _ Tip = Tip
    go k (Bin _ kx x l r) = case compare k kx of
        LT -> balance kx x (go k l) r
        GT -> balance kx x l (go k r)
        EQ -> case f kx x of
            Just x' -> Bin (size l + size r + 1) kx x' l r
            Nothing -> glue l r

-- | /O(log n)/. The most general update function. Can insert, update, or delete.
--
-- >>> alter (const Nothing) "a" (fromList [("a", 1), ("b", 2)])
-- fromList [("b",2)]
-- >>> alter (const (Just 99)) "c" (fromList [("a", 1)])
-- fromList [("a",1),("c",99)]
-- >>> alter (fmap (+10)) "a" (fromList [("a", 1)])
-- fromList [("a",11)]
alter :: Ord k => (Maybe a -> Maybe a) -> k -> Map k a -> Map k a
alter f k = go
  where
    go Tip = case f Nothing of
        Nothing -> Tip
        Just x  -> singleton k x
    go (Bin sx kx x l r) = case compare k kx of
        LT -> balance kx x (go l) r
        GT -> balance kx x l (go r)
        EQ -> case f (Just x) of
            Just x' -> Bin sx kx x' l r
            Nothing -> glue l r

-- | /O(log n)/. Functor-based alter. Useful for working with effects.
--
-- >>> alterF (\_ -> [Nothing, Just 99]) "a" (fromList [("a", 1)])
-- [fromList [],fromList [("a",99)]]
alterF :: (Functor f, Ord k) => (Maybe a -> f (Maybe a)) -> k -> Map k a -> f (Map k a)
alterF f k m = fmap ins (f (lookup k m))
  where ins Nothing  = delete k m
        ins (Just x) = insert k x m

-- ------------------------------------------------------------
-- Combine
-- ------------------------------------------------------------

-- | /O(m * log(n\/m + 1)), m <= n/. Left-biased union of two maps.
-- If a key exists in both maps, the value from the left map is kept.
--
-- >>> union (fromList [("a", 1)]) (fromList [("a", 2), ("b", 3)])
-- fromList [("a",1),("b",3)]
union :: Ord k => Map k a -> Map k a -> Map k a
union = unionWith const

-- | /O(m * log(n\/m + 1)), m <= n/. Union with a combining function.
--
-- >>> unionWith (+) (fromList [("a", 1)]) (fromList [("a", 2), ("b", 3)])
-- fromList [("a",3),("b",3)]
unionWith :: Ord k => (a -> a -> a) -> Map k a -> Map k a -> Map k a
unionWith f = unionWithKey (\_ x y -> f x y)

-- | /O(m * log(n\/m + 1)), m <= n/. Union with a key-aware combining function.
unionWithKey :: Ord k => (k -> a -> a -> a) -> Map k a -> Map k a -> Map k a
unionWithKey f t1 t2 = P.foldl' ins t1 (toList t2)
  where ins m (k, x) = insertWithKey f k x m

-- | /O(m * log(n\/m + 1)), m <= n/. Union of a foldable of maps.
--
-- >>> unions [fromList [("a", 1)], fromList [("b", 2)], fromList [("a", 3)]]
-- fromList [("a",1),("b",2)]
unions :: (Foldable f, Ord k) => f (Map k a) -> Map k a
unions = P.foldl' union empty

-- | /O(m * log(n\/m + 1)), m <= n/. Union of maps with a combining function.
--
-- >>> unionsWith (+) [fromList [("a", 1)], fromList [("a", 2)]]
-- fromList [("a",3)]
unionsWith :: (Foldable f, Ord k) => (a -> a -> a) -> f (Map k a) -> Map k a
unionsWith f = P.foldl' (unionWith f) empty

-- | /O(m * log(n\/m + 1)), m <= n/. Difference of two maps.
-- Returns elements of the first map not in the second.
--
-- >>> difference (fromList [("a", 1), ("b", 2)]) (fromList [("a", 3)])
-- fromList [("b",2)]
difference :: Ord k => Map k a -> Map k b -> Map k a
difference = differenceWith (\_ _ -> Nothing)

-- | /O(m * log(n\/m + 1)), m <= n/. Difference with a combining function.
-- If the function returns 'Nothing', the element is removed.
differenceWith :: Ord k => (a -> b -> Maybe a) -> Map k a -> Map k b -> Map k a
differenceWith f = differenceWithKey (\_ x y -> f x y)

-- | /O(m * log(n\/m + 1)), m <= n/. Difference with a key-aware combining function.
differenceWithKey :: Ord k => (k -> a -> b -> Maybe a) -> Map k a -> Map k b -> Map k a
differenceWithKey f t1 t2 = filterWithKey check t1
  where check k x = case lookup k t2 of
            Nothing -> True
            Just y  -> case f k x y of
                Nothing -> False
                Just _  -> True

-- | /O(m * log(n\/m + 1)), m <= n/. Intersection of two maps.
-- Returns elements of the first map that are also in the second.
--
-- >>> intersection (fromList [("a", 1), ("b", 2)]) (fromList [("a", 3), ("c", 4)])
-- fromList [("a",1)]
intersection :: Ord k => Map k a -> Map k b -> Map k a
intersection = intersectionWith const

-- | /O(m * log(n\/m + 1)), m <= n/. Intersection with a combining function.
--
-- >>> intersectionWith (+) (fromList [("a", 1)]) (fromList [("a", 2)])
-- fromList [("a",3)]
intersectionWith :: Ord k => (a -> b -> c) -> Map k a -> Map k b -> Map k c
intersectionWith f = intersectionWithKey (\_ x y -> f x y)

-- | /O(m * log(n\/m + 1)), m <= n/. Intersection with a key-aware combining function.
intersectionWithKey :: Ord k => (k -> a -> b -> c) -> Map k a -> Map k b -> Map k c
intersectionWithKey f t1 t2 = mapMaybeWithKey go t1
  where go k x = case lookup k t2 of
            Nothing -> Nothing
            Just y  -> Just (f k x y)

-- ------------------------------------------------------------
-- Traversal
-- ------------------------------------------------------------

-- | /O(n)/. Map a function over all values in the map.
--
-- >>> map (+1) (fromList [("a", 1), ("b", 2)])
-- fromList [("a",2),("b",3)]
map :: (a -> b) -> Map k a -> Map k b
map f = mapWithKey (\_ x -> f x)

-- | /O(n)/. Map a function over all values with access to the key.
--
-- >>> mapWithKey (\k x -> show k ++ ":" ++ show x) (fromList [("a", 1)])
-- fromList [("a","a:1")]
mapWithKey :: (k -> a -> b) -> Map k a -> Map k b
mapWithKey _ Tip = Tip
mapWithKey f (Bin sx kx x l r) = Bin sx kx (f kx x) (mapWithKey f l) (mapWithKey f r)

-- | /O(n)/. Traverse the map with effects, providing access to keys.
--
-- >>> traverseWithKey (\k v -> if v > 0 then Just (k, v) else Nothing) (fromList [("a", 1)])
-- Just (fromList [("a",("a",1))])
traverseWithKey :: Applicative t => (k -> a -> t b) -> Map k a -> t (Map k b)
traverseWithKey _ Tip = pure Tip
traverseWithKey f (Bin s k x l r) =
    (\l' x' r' -> Bin s k x' l' r') <$> traverseWithKey f l <*> f k x <*> traverseWithKey f r

-- | /O(n)/. Thread an accumulating argument through the map in ascending order.
--
-- >>> mapAccum (\a x -> (a + x, show x)) 0 (fromList [("a", 1), ("b", 2)])
-- (3,fromList [("a","1"),("b","2")])
mapAccum :: (a -> b -> (a, c)) -> a -> Map k b -> (a, Map k c)
mapAccum f = mapAccumWithKey (\a _ x -> f a x)

-- | /O(n)/. Thread an accumulating argument with access to keys.
mapAccumWithKey :: (a -> k -> b -> (a, c)) -> a -> Map k b -> (a, Map k c)
mapAccumWithKey _ a Tip = (a, Tip)
mapAccumWithKey f a (Bin sx kx x l r) =
    let (a1, l') = mapAccumWithKey f a l
        (a2, x') = f a1 kx x
        (a3, r') = mapAccumWithKey f a2 r
    in (a3, Bin sx kx x' l' r')

-- | /O(n * log n)/. Map a function over the keys.
-- If the function maps distinct keys to the same key, the later value wins.
--
-- >>> mapKeys (++ "!") (fromList [("a", 1), ("b", 2)])
-- fromList [("a!",1),("b!",2)]
mapKeys :: Ord k2 => (k1 -> k2) -> Map k1 a -> Map k2 a
mapKeys f = fromList . P.map (\(k, x) -> (f k, x)) . toList

-- | /O(n * log n)/. Map over keys with a combining function for collisions.
--
-- >>> mapKeysWith (+) (const "x") (fromList [("a", 1), ("b", 2)])
-- fromList [("x",3)]
mapKeysWith :: Ord k2 => (a -> a -> a) -> (k1 -> k2) -> Map k1 a -> Map k2 a
mapKeysWith c f = fromListWith c . P.map (\(k, x) -> (f k, x)) . toList

-- | /O(n)/. Map over keys with a strictly monotonic function.
-- The precondition (monotonicity) is not checked.
--
-- __Warning__: If the function is not monotonic, the result is undefined.
--
-- >>> mapKeysMonotonic (++ "!") (fromList [("a", 1), ("b", 2)])
-- fromList [("a!",1),("b!",2)]
mapKeysMonotonic :: (k1 -> k2) -> Map k1 a -> Map k2 a
mapKeysMonotonic _ Tip = Tip
mapKeysMonotonic f (Bin sx kx x l r) =
    Bin sx (f kx) x (mapKeysMonotonic f l) (mapKeysMonotonic f r)

-- ------------------------------------------------------------
-- Folds
-- ------------------------------------------------------------

-- | /O(n)/. Fold the values in the map using a right-associative operator.
-- Values are folded in ascending key order.
--
-- >>> foldr (+) 0 (fromList [("a", 1), ("b", 2), ("c", 3)])
-- 6
foldr :: (a -> b -> b) -> b -> Map k a -> b
foldr f = foldrWithKey (\_ x z -> f x z)

-- | /O(n)/. Fold the values using a left-associative operator.
-- Values are folded in ascending key order.
--
-- >>> foldl (\acc x -> acc ++ show x) "" (fromList [("a", 1), ("b", 2)])
-- "12"
foldl :: (a -> b -> a) -> a -> Map k b -> a
foldl f = foldlWithKey (\z _ x -> f z x)

-- | /O(n)/. Fold with access to keys, right-associative.
--
-- >>> foldrWithKey (\k v acc -> (k, v) : acc) [] (fromList [("a", 1), ("b", 2)])
-- [("a",1),("b",2)]
foldrWithKey :: (k -> a -> b -> b) -> b -> Map k a -> b
foldrWithKey _ z Tip = z
foldrWithKey f z (Bin _ kx x l r) = foldrWithKey f (f kx x (foldrWithKey f z r)) l

-- | /O(n)/. Fold with access to keys, left-associative.
foldlWithKey :: (a -> k -> b -> a) -> a -> Map k b -> a
foldlWithKey _ z Tip = z
foldlWithKey f z (Bin _ kx x l r) = foldlWithKey f (f (foldlWithKey f z l) kx x) r

-- | /O(n)/. Fold the map into a monoid with access to keys.
--
-- >>> foldMapWithKey (\k v -> [k ++ "=" ++ show v]) (fromList [("a", 1), ("b", 2)])
-- ["a=1","b=2"]
foldMapWithKey :: Monoid m => (k -> a -> m) -> Map k a -> m
foldMapWithKey f = foldrWithKey (\k x m -> f k x <> m) mempty

-- ------------------------------------------------------------
-- Conversion
-- ------------------------------------------------------------

-- | /O(n)/. Return all values of the map in ascending key order.
--
-- >>> elems (fromList [("b", 2), ("a", 1)])
-- [1, 2]
elems :: Map k a -> [a]
elems = foldr (:) []

-- | /O(n)/. Return all keys of the map in ascending order.
--
-- >>> keys (fromList [("b", 2), ("a", 1)])
-- ["a", "b"]
keys :: Map k a -> [k]
keys = foldrWithKey (\k _ ks -> k:ks) []

-- | /O(n)/. Return all key-value pairs in ascending key order.
--
-- >>> assocs (fromList [("b", 2), ("a", 1)])
-- [("a",1),("b",2)]
assocs :: Map k a -> [(k, a)]
assocs = toAscList

-- | /O(n * log n)/. Convert the keys to a 'Set.Set'.
--
-- >>> keysSet (fromList [("a", 1), ("b", 2)])
-- fromList ["a","b"]
keysSet :: Map k a -> Set.Set k
keysSet = Set.fromList . keys

-- | /O(n)/. Convert to a list of key-value pairs in ascending key order.
--
-- >>> toList (fromList [("b", 2), ("a", 1)])
-- [("a",1),("b",2)]
toList :: Map k a -> [(k, a)]
toList = toAscList

-- | /O(n)/. Convert to an ascending list.
--
-- >>> toAscList (fromList [("b", 2), ("a", 1)])
-- [("a",1),("b",2)]
toAscList :: Map k a -> [(k, a)]
toAscList = foldrWithKey (\k x xs -> (k, x):xs) []

-- | /O(n)/. Convert to a descending list.
--
-- >>> toDescList (fromList [("a", 1), ("b", 2)])
-- [("b",2),("a",1)]
toDescList :: Map k a -> [(k, a)]
toDescList = foldlWithKey (\xs k x -> (k, x):xs) []

-- ------------------------------------------------------------
-- Filter
-- ------------------------------------------------------------

-- | /O(n)/. Filter all values that satisfy a predicate.
--
-- >>> filter (> 1) (fromList [("a", 1), ("b", 2), ("c", 3)])
-- fromList [("b",2),("c",3)]
filter :: (a -> Bool) -> Map k a -> Map k a
filter p = filterWithKey (\_ x -> p x)

-- | /O(n)/. Filter with access to keys.
--
-- >>> filterWithKey (\k v -> k == "a" || v > 1) (fromList [("a", 1), ("b", 2)])
-- fromList [("a",1),("b",2)]
filterWithKey :: (k -> a -> Bool) -> Map k a -> Map k a
filterWithKey _ Tip = Tip
filterWithKey p (Bin _ kx x l r)
    | p kx x    = link kx x (filterWithKey p l) (filterWithKey p r)
    | otherwise = merge (filterWithKey p l) (filterWithKey p r)

-- | /O(n)/. Partition the map according to a predicate.
-- The first map contains elements satisfying the predicate.
--
-- >>> partition (> 1) (fromList [("a", 1), ("b", 2), ("c", 3)])
-- (fromList [("b",2),("c",3)],fromList [("a",1)])
partition :: (a -> Bool) -> Map k a -> (Map k a, Map k a)
partition p = partitionWithKey (\_ x -> p x)

-- | /O(n)/. Partition with access to keys.
partitionWithKey :: (k -> a -> Bool) -> Map k a -> (Map k a, Map k a)
partitionWithKey _ Tip = (Tip, Tip)
partitionWithKey p (Bin _ kx x l r)
    | p kx x    = (link kx x l1 r1, merge l2 r2)
    | otherwise = (merge l1 r1, link kx x l2 r2)
  where
    (l1, l2) = partitionWithKey p l
    (r1, r2) = partitionWithKey p r

-- | /O(n)/. Map and filter in a single pass.
--
-- >>> mapMaybe (\x -> if x > 1 then Just (x * 10) else Nothing) (fromList [("a", 1), ("b", 2)])
-- fromList [("b",20)]
mapMaybe :: (a -> Maybe b) -> Map k a -> Map k b
mapMaybe f = mapMaybeWithKey (\_ x -> f x)

-- | /O(n)/. Map and filter with access to keys.
mapMaybeWithKey :: (k -> a -> Maybe b) -> Map k a -> Map k b
mapMaybeWithKey _ Tip = Tip
mapMaybeWithKey f (Bin _ kx x l r) = case f kx x of
    Just y  -> link kx y (mapMaybeWithKey f l) (mapMaybeWithKey f r)
    Nothing -> merge (mapMaybeWithKey f l) (mapMaybeWithKey f r)

-- | /O(n)/. Map and split into two maps based on 'Either'.
--
-- >>> mapEither (\x -> if x > 1 then Right (x * 10) else Left x) (fromList [("a", 1), ("b", 2)])
-- (fromList [("a",1)],fromList [("b",20)])
mapEither :: (a -> Either b c) -> Map k a -> (Map k b, Map k c)
mapEither f = mapEitherWithKey (\_ x -> f x)

-- | /O(n)/. Map and split with access to keys.
mapEitherWithKey :: (k -> a -> Either b c) -> Map k a -> (Map k b, Map k c)
mapEitherWithKey _ Tip = (Tip, Tip)
mapEitherWithKey f (Bin _ kx x l r) = case f kx x of
    Left y  -> (link kx y l1 r1, merge l2 r2)
    Right z -> (merge l1 r1, link kx z l2 r2)
  where
    (l1, l2) = mapEitherWithKey f l
    (r1, r2) = mapEitherWithKey f r

-- ------------------------------------------------------------
-- Submap
-- ------------------------------------------------------------

-- | /O(n * log n)/. Is the first map a submap of the second?
-- All keys in the first map must exist in the second with equal values.
--
-- >>> isSubmapOf (fromList [("a", 1)]) (fromList [("a", 1), ("b", 2)])
-- True
-- >>> isSubmapOf (fromList [("a", 2)]) (fromList [("a", 1), ("b", 2)])
-- False
isSubmapOf :: (Ord k, Eq a) => Map k a -> Map k a -> Bool
isSubmapOf = isSubmapOfBy (==)

-- | /O(n * log n)/. Submap check with a custom equality predicate.
isSubmapOfBy :: Ord k => (a -> b -> Bool) -> Map k a -> Map k b -> Bool
isSubmapOfBy f t1 t2 = P.all check (toList t1)
  where check (k, x) = case lookup k t2 of
            Nothing -> False
            Just y  -> f x y

-- | /O(n * log n)/. Is the first map a proper submap of the second?
-- (All keys present and at least one key missing)
--
-- >>> isProperSubmapOf (fromList [("a", 1)]) (fromList [("a", 1), ("b", 2)])
-- True
-- >>> isProperSubmapOf (fromList [("a", 1)]) (fromList [("a", 1)])
-- False
isProperSubmapOf :: (Ord k, Eq a) => Map k a -> Map k a -> Bool
isProperSubmapOf = isProperSubmapOfBy (==)

-- | /O(n * log n)/. Proper submap check with a custom equality predicate.
isProperSubmapOfBy :: Ord k => (a -> b -> Bool) -> Map k a -> Map k b -> Bool
isProperSubmapOfBy f t1 t2 = size t1 < size t2 && isSubmapOfBy f t1 t2

-- ------------------------------------------------------------
-- Min/Max
-- ------------------------------------------------------------

-- | /O(log n)/. Lookup the smallest key and its value.
--
-- >>> lookupMin (fromList [("b", 2), ("a", 1)])
-- Just ("a",1)
-- >>> lookupMin empty
-- Nothing
lookupMin :: Map k a -> Maybe (k, a)
lookupMin Tip = Nothing
lookupMin (Bin _ k x Tip _) = Just (k, x)
lookupMin (Bin _ _ _ l _) = lookupMin l

-- | /O(log n)/. Lookup the largest key and its value.
--
-- >>> lookupMax (fromList [("a", 1), ("b", 2)])
-- Just ("b",2)
lookupMax :: Map k a -> Maybe (k, a)
lookupMax Tip = Nothing
lookupMax (Bin _ k x _ Tip) = Just (k, x)
lookupMax (Bin _ _ _ _ r) = lookupMax r

-- | /O(log n)/. Find the smallest key and its value.
--
-- __Warning__: Partial function. Throws an error on empty map.
-- Prefer 'lookupMin'.
--
-- >>> findMin (fromList [("b", 2), ("a", 1)])
-- ("a",1)
findMin :: Map k a -> (k, a)
findMin m = case lookupMin m of
    Just kv -> kv
    Nothing -> error "Map.findMin: empty map"

-- | /O(log n)/. Find the largest key and its value.
--
-- __Warning__: Partial function. Throws an error on empty map.
-- Prefer 'lookupMax'.
--
-- >>> findMax (fromList [("a", 1), ("b", 2)])
-- ("b",2)
findMax :: Map k a -> (k, a)
findMax m = case lookupMax m of
    Just kv -> kv
    Nothing -> error "Map.findMax: empty map"

-- | /O(log n)/. Delete the smallest key. Returns the empty map if empty.
--
-- >>> deleteMin (fromList [("a", 1), ("b", 2)])
-- fromList [("b",2)]
deleteMin :: Map k a -> Map k a
deleteMin Tip = Tip
deleteMin (Bin _ _ _ Tip r) = r
deleteMin (Bin _ kx x l r) = balance kx x (deleteMin l) r

-- | /O(log n)/. Delete the largest key. Returns the empty map if empty.
--
-- >>> deleteMax (fromList [("a", 1), ("b", 2)])
-- fromList [("a",1)]
deleteMax :: Map k a -> Map k a
deleteMax Tip = Tip
deleteMax (Bin _ _ _ l Tip) = l
deleteMax (Bin _ kx x l r) = balance kx x l (deleteMax r)

-- ------------------------------------------------------------
-- Split
-- ------------------------------------------------------------

-- | /O(log n)/. Split the map at a key. Returns maps with keys less than
-- and greater than the given key. The key itself is discarded.
--
-- >>> split "b" (fromList [("a", 1), ("b", 2), ("c", 3)])
-- (fromList [("a",1)],fromList [("c",3)])
split :: Ord k => k -> Map k a -> (Map k a, Map k a)
split _ Tip = (Tip, Tip)
split k (Bin _ kx x l r) = case compare k kx of
    LT -> let (lt, gt) = split k l in (lt, link kx x gt r)
    GT -> let (lt, gt) = split k r in (link kx x l lt, gt)
    EQ -> (l, r)

-- | /O(log n)/. Split at a key and also return the value at that key.
--
-- >>> splitLookup "b" (fromList [("a", 1), ("b", 2), ("c", 3)])
-- (fromList [("a",1)],Just 2,fromList [("c",3)])
-- >>> splitLookup "d" (fromList [("a", 1), ("b", 2)])
-- (fromList [("a",1),("b",2)],Nothing,fromList [])
splitLookup :: Ord k => k -> Map k a -> (Map k a, Maybe a, Map k a)
splitLookup _ Tip = (Tip, Nothing, Tip)
splitLookup k (Bin _ kx x l r) = case compare k kx of
    LT -> let (lt, found, gt) = splitLookup k l in (lt, found, link kx x gt r)
    GT -> let (lt, found, gt) = splitLookup k r in (link kx x l lt, found, gt)
    EQ -> (l, Just x, r)

-- Internal helpers
balance :: k -> a -> Map k a -> Map k a -> Map k a
balance k x l r = Bin (size l + size r + 1) k x l r

link :: k -> a -> Map k a -> Map k a -> Map k a
link kx x Tip r = insertMin kx x r
link kx x l Tip = insertMax kx x l
link kx x l r = balance kx x l r

insertMin :: k -> a -> Map k a -> Map k a
insertMin kx x Tip = singleton kx x
insertMin kx x (Bin _ ky y l r) = balance ky y (insertMin kx x l) r

insertMax :: k -> a -> Map k a -> Map k a
insertMax kx x Tip = singleton kx x
insertMax kx x (Bin _ ky y l r) = balance ky y l (insertMax kx x r)

glue :: Map k a -> Map k a -> Map k a
glue Tip r = r
glue l Tip = l
glue l r
    | size l > size r = let (k, x) = findMax l in balance k x (deleteMax l) r
    | otherwise       = let (k, x) = findMin r in balance k x l (deleteMin r)

merge :: Map k a -> Map k a -> Map k a
merge Tip r = r
merge l Tip = l
merge l r = glue l r
