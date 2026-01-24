-- |
-- Module      : BHC.Data.Set
-- Description : Ordered sets
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.Data.Set (
    Set,
    
    -- * Construction
    empty, singleton, fromList, fromAscList, fromDescList,
    
    -- * Insertion
    insert,
    
    -- * Deletion
    delete,
    
    -- * Query
    member, notMember, lookupLT, lookupGT, lookupLE, lookupGE,
    null, size, isSubsetOf, isProperSubsetOf,
    
    -- * Combine
    union, unions, difference, (\\), intersection,
    disjoint,
    
    -- * Filter
    filter, partition, split, splitMember,
    
    -- * Map
    map, mapMonotonic,
    
    -- * Folds
    foldr, foldl, foldr', foldl',
    
    -- * Min/Max
    lookupMin, lookupMax, findMin, findMax,
    deleteMin, deleteMax, minView, maxView,
    
    -- * Conversion
    elems, toList, toAscList, toDescList,
) where

import BHC.Prelude hiding (map, null, filter, foldr, foldl)
import qualified BHC.Prelude as P

-- | A set of values @a@.
data Set a
    = Tip
    | Bin {-# UNPACK #-} !Int !a !(Set a) !(Set a)
    deriving (Eq, Ord, Show, Read)

instance Ord a => Semigroup (Set a) where
    (<>) = union

instance Ord a => Monoid (Set a) where
    mempty = empty

instance Foldable Set where
    foldr = foldr
    foldl = foldl
    null = null
    length = size
    elem = member
    toList = toList

-- ------------------------------------------------------------
-- Construction
-- ------------------------------------------------------------

-- | /O(1)/. The empty set.
--
-- >>> empty
-- fromList []
empty :: Set a
empty = Tip

-- | /O(1)/. A set with a single element.
--
-- >>> singleton 1
-- fromList [1]
singleton :: a -> Set a
singleton x = Bin 1 x Tip Tip

-- | /O(n * log n)/. Build a set from a list of elements.
--
-- >>> fromList [3, 1, 2, 1]
-- fromList [1,2,3]
fromList :: Ord a => [a] -> Set a
fromList = P.foldl' (flip insert) empty

-- | /O(n)/. Build a set from an ascending list of elements.
-- The precondition (ascending order) is not validated.
--
-- >>> fromAscList [1, 2, 3]
-- fromList [1,2,3]
fromAscList :: Eq a => [a] -> Set a
fromAscList = fromList

-- | /O(n)/. Build a set from a descending list of elements.
-- The precondition (descending order) is not validated.
--
-- >>> fromDescList [3, 2, 1]
-- fromList [1,2,3]
fromDescList :: Eq a => [a] -> Set a
fromDescList = fromList

-- ------------------------------------------------------------
-- Insertion
-- ------------------------------------------------------------

-- | /O(log n)/. Insert an element into the set.
-- If the element is already present, replaces it.
--
-- >>> insert 3 (fromList [1, 2])
-- fromList [1,2,3]
insert :: Ord a => a -> Set a -> Set a
insert x = go
  where
    go Tip = singleton x
    go (Bin sz y l r) = case compare x y of
        LT -> balance y (go l) r
        GT -> balance y l (go r)
        EQ -> Bin sz x l r

-- ------------------------------------------------------------
-- Deletion
-- ------------------------------------------------------------

-- | /O(log n)/. Delete an element from the set.
-- If the element is not present, the original set is returned.
--
-- >>> delete 2 (fromList [1, 2, 3])
-- fromList [1,3]
delete :: Ord a => a -> Set a -> Set a
delete x = go
  where
    go Tip = Tip
    go (Bin _ y l r) = case compare x y of
        LT -> balance y (go l) r
        GT -> balance y l (go r)
        EQ -> glue l r

-- ------------------------------------------------------------
-- Query
-- ------------------------------------------------------------

-- | /O(log n)/. Is the element a member of the set?
--
-- >>> member 2 (fromList [1, 2, 3])
-- True
-- >>> member 4 (fromList [1, 2, 3])
-- False
member :: Ord a => a -> Set a -> Bool
member x = go
  where
    go Tip = False
    go (Bin _ y l r) = case compare x y of
        LT -> go l
        GT -> go r
        EQ -> True

-- | /O(log n)/. Is the element not a member of the set?
--
-- >>> notMember 4 (fromList [1, 2, 3])
-- True
notMember :: Ord a => a -> Set a -> Bool
notMember x = not . member x

-- | /O(log n)/. Find the largest element smaller than the given one.
--
-- >>> lookupLT 3 (fromList [1, 2, 4, 5])
-- Just 2
-- >>> lookupLT 1 (fromList [1, 2, 4, 5])
-- Nothing
lookupLT :: Ord a => a -> Set a -> Maybe a
lookupLT = goNothing
  where
    goNothing _ Tip = Nothing
    goNothing x (Bin _ y l r)
        | x <= y = goNothing x l
        | otherwise = goJust x y r
    goJust _ best Tip = Just best
    goJust x best (Bin _ y l r)
        | x <= y = goJust x best l
        | otherwise = goJust x y r

-- | /O(log n)/. Find the smallest element larger than the given one.
--
-- >>> lookupGT 3 (fromList [1, 2, 4, 5])
-- Just 4
-- >>> lookupGT 5 (fromList [1, 2, 4, 5])
-- Nothing
lookupGT :: Ord a => a -> Set a -> Maybe a
lookupGT = goNothing
  where
    goNothing _ Tip = Nothing
    goNothing x (Bin _ y l r)
        | x < y = goJust x y l
        | otherwise = goNothing x r
    goJust _ best Tip = Just best
    goJust x best (Bin _ y l r)
        | x < y = goJust x y l
        | otherwise = goJust x best r

-- | /O(log n)/. Find the largest element less than or equal to the given one.
--
-- >>> lookupLE 3 (fromList [1, 2, 4, 5])
-- Just 2
-- >>> lookupLE 4 (fromList [1, 2, 4, 5])
-- Just 4
lookupLE :: Ord a => a -> Set a -> Maybe a
lookupLE = goNothing
  where
    goNothing _ Tip = Nothing
    goNothing x (Bin _ y l r) = case compare x y of
        LT -> goNothing x l
        EQ -> Just y
        GT -> goJust x y r
    goJust _ best Tip = Just best
    goJust x best (Bin _ y l r) = case compare x y of
        LT -> goJust x best l
        EQ -> Just y
        GT -> goJust x y r

-- | /O(log n)/. Find the smallest element greater than or equal to the given one.
--
-- >>> lookupGE 3 (fromList [1, 2, 4, 5])
-- Just 4
-- >>> lookupGE 4 (fromList [1, 2, 4, 5])
-- Just 4
lookupGE :: Ord a => a -> Set a -> Maybe a
lookupGE = goNothing
  where
    goNothing _ Tip = Nothing
    goNothing x (Bin _ y l r) = case compare x y of
        LT -> goJust x y l
        EQ -> Just y
        GT -> goNothing x r
    goJust _ best Tip = Just best
    goJust x best (Bin _ y l r) = case compare x y of
        LT -> goJust x y l
        EQ -> Just y
        GT -> goJust x best r

-- | /O(1)/. Is the set empty?
--
-- >>> null empty
-- True
-- >>> null (singleton 1)
-- False
null :: Set a -> Bool
null Tip = True
null _   = False

-- | /O(1)/. The number of elements in the set.
--
-- >>> size (fromList [1, 2, 3])
-- 3
size :: Set a -> Int
size Tip            = 0
size (Bin n _ _ _)  = n

-- | /O(n * log n)/. Is the first set a subset of the second?
--
-- >>> isSubsetOf (fromList [1, 2]) (fromList [1, 2, 3])
-- True
-- >>> isSubsetOf (fromList [1, 4]) (fromList [1, 2, 3])
-- False
isSubsetOf :: Ord a => Set a -> Set a -> Bool
isSubsetOf t1 t2 = size t1 <= size t2 && P.all (`member` t2) (toList t1)

-- | /O(n * log n)/. Is the first set a proper subset of the second?
-- (Subset but not equal)
--
-- >>> isProperSubsetOf (fromList [1, 2]) (fromList [1, 2, 3])
-- True
-- >>> isProperSubsetOf (fromList [1, 2]) (fromList [1, 2])
-- False
isProperSubsetOf :: Ord a => Set a -> Set a -> Bool
isProperSubsetOf t1 t2 = size t1 < size t2 && isSubsetOf t1 t2

-- ------------------------------------------------------------
-- Combine
-- ------------------------------------------------------------

-- | /O(m * log(n\/m + 1)), m <= n/. Union of two sets.
--
-- >>> union (fromList [1, 2]) (fromList [2, 3])
-- fromList [1,2,3]
union :: Ord a => Set a -> Set a -> Set a
union t1 Tip = t1
union Tip t2 = t2
union t1 t2 = P.foldl' (flip insert) t1 (toList t2)

-- | /O(m * log(n\/m + 1)), m <= n/. Union of a foldable of sets.
--
-- >>> unions [fromList [1, 2], fromList [2, 3], fromList [3, 4]]
-- fromList [1,2,3,4]
unions :: (Foldable f, Ord a) => f (Set a) -> Set a
unions = P.foldl' union empty

-- | /O(m * log(n\/m + 1)), m <= n/. Difference of two sets.
-- Elements in the first set but not in the second.
--
-- >>> difference (fromList [1, 2, 3]) (fromList [2, 3, 4])
-- fromList [1]
difference :: Ord a => Set a -> Set a -> Set a
difference t1 t2 = P.foldl' (flip delete) t1 (toList t2)

-- | /O(m * log(n\/m + 1)), m <= n/. Infix operator for 'difference'.
--
-- >>> fromList [1, 2, 3] \\ fromList [2, 3]
-- fromList [1]
(\\) :: Ord a => Set a -> Set a -> Set a
(\\) = difference
infixl 9 \\

-- | /O(m * log(n\/m + 1)), m <= n/. Intersection of two sets.
-- Elements present in both sets.
--
-- >>> intersection (fromList [1, 2, 3]) (fromList [2, 3, 4])
-- fromList [2,3]
intersection :: Ord a => Set a -> Set a -> Set a
intersection t1 t2 = filter (`member` t2) t1

-- | /O(m * log(n\/m + 1)), m <= n/. Check if two sets are disjoint (no common elements).
--
-- >>> disjoint (fromList [1, 2]) (fromList [3, 4])
-- True
-- >>> disjoint (fromList [1, 2]) (fromList [2, 3])
-- False
disjoint :: Ord a => Set a -> Set a -> Bool
disjoint t1 t2 = null (intersection t1 t2)

-- ------------------------------------------------------------
-- Filter
-- ------------------------------------------------------------

-- | /O(n)/. Filter elements satisfying a predicate.
--
-- >>> filter even (fromList [1, 2, 3, 4, 5])
-- fromList [2,4]
filter :: (a -> Bool) -> Set a -> Set a
filter _ Tip = Tip
filter p (Bin _ x l r)
    | p x       = link x (filter p l) (filter p r)
    | otherwise = merge (filter p l) (filter p r)

-- | /O(n)/. Partition the set according to a predicate.
-- First set contains elements satisfying the predicate.
--
-- >>> partition even (fromList [1, 2, 3, 4, 5])
-- (fromList [2,4],fromList [1,3,5])
partition :: (a -> Bool) -> Set a -> (Set a, Set a)
partition _ Tip = (Tip, Tip)
partition p (Bin _ x l r)
    | p x       = (link x l1 r1, merge l2 r2)
    | otherwise = (merge l1 r1, link x l2 r2)
  where
    (l1, l2) = partition p l
    (r1, r2) = partition p r

-- | /O(log n)/. Split the set at an element. Returns elements less than
-- and greater than the given element. The element itself is discarded.
--
-- >>> split 3 (fromList [1, 2, 3, 4, 5])
-- (fromList [1,2],fromList [4,5])
split :: Ord a => a -> Set a -> (Set a, Set a)
split _ Tip = (Tip, Tip)
split x (Bin _ y l r) = case compare x y of
    LT -> let (lt, gt) = split x l in (lt, link y gt r)
    GT -> let (lt, gt) = split x r in (link y l lt, gt)
    EQ -> (l, r)

-- | /O(log n)/. Split at an element and also return whether it was found.
--
-- >>> splitMember 3 (fromList [1, 2, 3, 4, 5])
-- (fromList [1,2],True,fromList [4,5])
-- >>> splitMember 6 (fromList [1, 2, 3, 4, 5])
-- (fromList [1,2,3,4,5],False,fromList [])
splitMember :: Ord a => a -> Set a -> (Set a, Bool, Set a)
splitMember _ Tip = (Tip, False, Tip)
splitMember x (Bin _ y l r) = case compare x y of
    LT -> let (lt, found, gt) = splitMember x l in (lt, found, link y gt r)
    GT -> let (lt, found, gt) = splitMember x r in (link y l lt, found, gt)
    EQ -> (l, True, r)

-- ------------------------------------------------------------
-- Map
-- ------------------------------------------------------------

-- | /O(n * log n)/. Map a function over the set.
-- If the function maps distinct elements to the same value,
-- the result will have fewer elements.
--
-- >>> map (+1) (fromList [1, 2, 3])
-- fromList [2,3,4]
map :: Ord b => (a -> b) -> Set a -> Set b
map f = fromList . P.map f . toList

-- | /O(n)/. Map a strictly monotonic function over the set.
-- The precondition (monotonicity) is not validated.
--
-- __Warning__: If the function is not monotonic, the result is undefined.
--
-- >>> mapMonotonic (+1) (fromList [1, 2, 3])
-- fromList [2,3,4]
mapMonotonic :: (a -> b) -> Set a -> Set b
mapMonotonic _ Tip = Tip
mapMonotonic f (Bin sz x l r) = Bin sz (f x) (mapMonotonic f l) (mapMonotonic f r)

-- ------------------------------------------------------------
-- Folds
-- ------------------------------------------------------------

-- | /O(n)/. Fold the elements using a right-associative operator.
-- Elements are folded in ascending order.
--
-- >>> foldr (\x acc -> show x ++ acc) "" (fromList [1, 2, 3])
-- "123"
foldr :: (a -> b -> b) -> b -> Set a -> b
foldr _ z Tip = z
foldr f z (Bin _ x l r) = foldr f (f x (foldr f z r)) l

-- | /O(n)/. Fold the elements using a left-associative operator.
-- Elements are folded in ascending order.
--
-- >>> foldl (\acc x -> acc ++ show x) "" (fromList [1, 2, 3])
-- "123"
foldl :: (a -> b -> a) -> a -> Set b -> a
foldl _ z Tip = z
foldl f z (Bin _ x l r) = foldl f (f (foldl f z l) x) r

-- | /O(n)/. Strict right fold.
foldr' :: (a -> b -> b) -> b -> Set a -> b
foldr' f z = go z
  where
    go !z' Tip = z'
    go !z' (Bin _ x l r) = go (f x (go z' r)) l

-- | /O(n)/. Strict left fold.
foldl' :: (a -> b -> a) -> a -> Set b -> a
foldl' f z = go z
  where
    go !z' Tip = z'
    go !z' (Bin _ x l r) = go (f (go z' l) x) r

-- ------------------------------------------------------------
-- Min/Max
-- ------------------------------------------------------------

-- | /O(log n)/. Lookup the minimum element.
--
-- >>> lookupMin (fromList [3, 1, 2])
-- Just 1
-- >>> lookupMin empty
-- Nothing
lookupMin :: Set a -> Maybe a
lookupMin Tip = Nothing
lookupMin (Bin _ x Tip _) = Just x
lookupMin (Bin _ _ l _) = lookupMin l

-- | /O(log n)/. Lookup the maximum element.
--
-- >>> lookupMax (fromList [1, 3, 2])
-- Just 3
lookupMax :: Set a -> Maybe a
lookupMax Tip = Nothing
lookupMax (Bin _ x _ Tip) = Just x
lookupMax (Bin _ _ _ r) = lookupMax r

-- | /O(log n)/. Find the minimum element.
--
-- __Warning__: Partial function. Throws an error on empty set.
-- Prefer 'lookupMin'.
--
-- >>> findMin (fromList [3, 1, 2])
-- 1
findMin :: Set a -> a
findMin s = case lookupMin s of
    Just x  -> x
    Nothing -> error "Set.findMin: empty set"

-- | /O(log n)/. Find the maximum element.
--
-- __Warning__: Partial function. Throws an error on empty set.
-- Prefer 'lookupMax'.
--
-- >>> findMax (fromList [1, 3, 2])
-- 3
findMax :: Set a -> a
findMax s = case lookupMax s of
    Just x  -> x
    Nothing -> error "Set.findMax: empty set"

-- | /O(log n)/. Delete the minimum element.
--
-- >>> deleteMin (fromList [1, 2, 3])
-- fromList [2,3]
deleteMin :: Set a -> Set a
deleteMin Tip = Tip
deleteMin (Bin _ _ Tip r) = r
deleteMin (Bin _ x l r) = balance x (deleteMin l) r

-- | /O(log n)/. Delete the maximum element.
--
-- >>> deleteMax (fromList [1, 2, 3])
-- fromList [1,2]
deleteMax :: Set a -> Set a
deleteMax Tip = Tip
deleteMax (Bin _ _ l Tip) = l
deleteMax (Bin _ x l r) = balance x l (deleteMax r)

-- | /O(log n)/. Retrieve and delete the minimum element.
--
-- >>> minView (fromList [1, 2, 3])
-- Just (1,fromList [2,3])
-- >>> minView empty
-- Nothing
minView :: Set a -> Maybe (a, Set a)
minView Tip = Nothing
minView s = Just (findMin s, deleteMin s)

-- | /O(log n)/. Retrieve and delete the maximum element.
--
-- >>> maxView (fromList [1, 2, 3])
-- Just (3,fromList [1,2])
maxView :: Set a -> Maybe (a, Set a)
maxView Tip = Nothing
maxView s = Just (findMax s, deleteMax s)

-- ------------------------------------------------------------
-- Conversion
-- ------------------------------------------------------------

-- | /O(n)/. Return all elements as a list in ascending order.
-- Same as 'toList'.
--
-- >>> elems (fromList [3, 1, 2])
-- [1,2,3]
elems :: Set a -> [a]
elems = toList

-- | /O(n)/. Convert to a list in ascending order.
--
-- >>> toList (fromList [3, 1, 2])
-- [1,2,3]
toList :: Set a -> [a]
toList = toAscList

-- | /O(n)/. Convert to an ascending list.
--
-- >>> toAscList (fromList [3, 1, 2])
-- [1,2,3]
toAscList :: Set a -> [a]
toAscList = foldr (:) []

-- | /O(n)/. Convert to a descending list.
--
-- >>> toDescList (fromList [1, 2, 3])
-- [3,2,1]
toDescList :: Set a -> [a]
toDescList = foldl (flip (:)) []

-- Internal
balance :: a -> Set a -> Set a -> Set a
balance x l r = Bin (size l + size r + 1) x l r

link :: a -> Set a -> Set a -> Set a
link x Tip r = insertMin x r
link x l Tip = insertMax x l
link x l r = balance x l r

insertMin :: a -> Set a -> Set a
insertMin x Tip = singleton x
insertMin x (Bin _ y l r) = balance y (insertMin x l) r

insertMax :: a -> Set a -> Set a
insertMax x Tip = singleton x
insertMax x (Bin _ y l r) = balance y l (insertMax x r)

glue :: Set a -> Set a -> Set a
glue Tip r = r
glue l Tip = l
glue l r
    | size l > size r = let m = findMax l in balance m (deleteMax l) r
    | otherwise       = let m = findMin r in balance m l (deleteMin r)

merge :: Set a -> Set a -> Set a
merge Tip r = r
merge l Tip = l
merge l r = glue l r
