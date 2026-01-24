-- |
-- Module      : BHC.Numeric.Vector
-- Description : Dense numeric vectors
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Unboxed numeric vectors with high-performance operations.
-- Vectors are 1-dimensional tensors with optimized operations.

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}

module BHC.Numeric.Vector (
    -- * Vector type
    Vector,

    -- * Construction
    empty, singleton,
    replicate, generate,
    fromList, fromListN,
    enumFromN, enumFromStepN,

    -- * Basic operations
    length, null,
    (!), (!?),
    head, last,
    tail, init,

    -- * Slicing
    slice, take, drop,
    splitAt, takeWhile, dropWhile,

    -- * Construction from vectors
    cons, snoc,
    (++), concat,

    -- * Element-wise operations
    map, imap,
    zipWith, zipWith3,
    izipWith, izipWith3,

    -- * Folds
    foldl, foldl', foldl1, foldl1',
    foldr, foldr1,
    ifoldl, ifoldl', ifoldr,

    -- * Specialized folds
    sum, product,
    maximum, minimum,
    maximumBy, minimumBy,
    all, any,

    -- * Scans
    prescanl, prescanl',
    postscanl, postscanl',
    scanl, scanl', scanl1, scanl1',

    -- * Search
    elem, notElem,
    find, findIndex,
    elemIndex, elemIndices,

    -- * Numeric operations
    dot, norm, normalize,
    add, sub, mul, div,
    scale, negate, abs,

    -- * Sorting
    sort, sortBy,
    indexed,

    -- * Conversion
    toList,
    convert,

    -- * Unboxed vectors
    UVector,

    -- * Type class
    VectorElem,
) where

import BHC.Prelude hiding (
    length, null, head, last, tail, init,
    take, drop, splitAt, takeWhile, dropWhile,
    (++), concat, map, zipWith, zipWith3,
    foldl, foldl1, foldr, foldr1,
    sum, product, maximum, minimum,
    all, any, elem, notElem,
    replicate, negate, abs
    )
import qualified BHC.Prelude as P
import Foreign.Ptr (Ptr, nullPtr)
import Foreign.ForeignPtr (ForeignPtr, newForeignPtr, withForeignPtr)
import Foreign.Marshal.Array (peekArray, withArrayLen)
import Foreign.Storable (Storable, sizeOf, peek, poke)
import System.IO.Unsafe (unsafePerformIO)

-- FFI imports for f64 vectors
foreign import ccall unsafe "bhc_vector_new_f64"
    c_vector_new_f64 :: Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_from_f64"
    c_vector_from_f64 :: Ptr Double -> Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_free_f64"
    c_vector_free_f64 :: Ptr VectorData -> IO ()

foreign import ccall unsafe "&bhc_vector_free_f64"
    c_vector_finalizer_f64 :: FunPtr (Ptr VectorData -> IO ())

foreign import ccall unsafe "bhc_vector_get_f64"
    c_vector_get_f64 :: Ptr VectorData -> Int -> IO Double

foreign import ccall unsafe "bhc_vector_len_f64"
    c_vector_len_f64 :: Ptr VectorData -> IO Int

foreign import ccall unsafe "bhc_vector_dot_f64"
    c_vector_dot_f64 :: Ptr VectorData -> Ptr VectorData -> IO Double

foreign import ccall unsafe "bhc_vector_sum_f64"
    c_vector_sum_f64 :: Ptr VectorData -> IO Double

foreign import ccall unsafe "bhc_vector_norm_f64"
    c_vector_norm_f64 :: Ptr VectorData -> IO Double

-- FFI imports for f32 vectors
foreign import ccall unsafe "bhc_vector_new_f32"
    c_vector_new_f32 :: Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_from_f32"
    c_vector_from_f32 :: Ptr Float -> Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_free_f32"
    c_vector_free_f32 :: Ptr VectorData -> IO ()

foreign import ccall unsafe "bhc_vector_get_f32"
    c_vector_get_f32 :: Ptr VectorData -> Int -> IO Float

foreign import ccall unsafe "bhc_vector_len_f32"
    c_vector_len_f32 :: Ptr VectorData -> IO Int

foreign import ccall unsafe "bhc_vector_dot_f32"
    c_vector_dot_f32 :: Ptr VectorData -> Ptr VectorData -> IO Float

foreign import ccall unsafe "bhc_vector_sum_f32"
    c_vector_sum_f32 :: Ptr VectorData -> IO Float

-- FFI imports for i64 vectors (Int)
foreign import ccall unsafe "bhc_vector_new_i64"
    c_vector_new_i64 :: Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_from_i64"
    c_vector_from_i64 :: Ptr Int -> Int -> IO (Ptr VectorData)

foreign import ccall unsafe "bhc_vector_get_i64"
    c_vector_get_i64 :: Ptr VectorData -> Int -> IO Int

foreign import ccall unsafe "bhc_vector_len_i64"
    c_vector_len_i64 :: Ptr VectorData -> IO Int

foreign import ccall unsafe "bhc_vector_sum_i64"
    c_vector_sum_i64 :: Ptr VectorData -> IO Int

-- ============================================================
-- Vector Type
-- ============================================================

-- | A contiguous array of elements.
-- Uses foreign pointer to Rust-managed memory.
data Vector a = Vector
    { vecPtr    :: !(ForeignPtr VectorData)
    , vecOffset :: !Int
    , vecLength :: !Int
    }

-- | Internal vector storage (opaque Rust type).
data VectorData

-- | Unboxed vector (no pointer indirection).
-- For now, same as Vector but may have specialized representation.
data UVector a = UVector
    { uvecPtr    :: !(ForeignPtr VectorData)
    , uvecOffset :: !Int
    , uvecLength :: !Int
    }

-- | Type class for vector element operations
class VectorElem a where
    vectorFromList :: [a] -> IO (Vector a)
    vectorGet :: Vector a -> Int -> IO a
    vectorLen :: Vector a -> IO Int
    vectorDot :: Vector a -> Vector a -> IO a
    vectorSum :: Vector a -> IO a

instance VectorElem Double where
    vectorFromList xs = do
        withArrayLen xs $ \len ptr -> do
            vptr <- c_vector_from_f64 ptr len
            fp <- newForeignPtr c_vector_finalizer_f64 vptr
            return $ Vector fp 0 len
    vectorGet (Vector fp off _) i = withForeignPtr fp $ \ptr ->
        c_vector_get_f64 ptr (off + i)
    vectorLen (Vector fp _ _) = withForeignPtr fp c_vector_len_f64
    vectorDot (Vector fp1 _ _) (Vector fp2 _ _) =
        withForeignPtr fp1 $ \p1 ->
        withForeignPtr fp2 $ \p2 ->
            c_vector_dot_f64 p1 p2
    vectorSum (Vector fp _ _) = withForeignPtr fp c_vector_sum_f64

instance VectorElem Float where
    vectorFromList xs = do
        withArrayLen xs $ \len ptr -> do
            vptr <- c_vector_from_f32 ptr len
            fp <- newForeignPtr c_vector_finalizer_f64 vptr  -- Uses same finalizer shape
            return $ Vector fp 0 len
    vectorGet (Vector fp off _) i = withForeignPtr fp $ \ptr ->
        c_vector_get_f32 ptr (off + i)
    vectorLen (Vector fp _ _) = withForeignPtr fp c_vector_len_f32
    vectorDot (Vector fp1 _ _) (Vector fp2 _ _) =
        withForeignPtr fp1 $ \p1 ->
        withForeignPtr fp2 $ \p2 ->
            c_vector_dot_f32 p1 p2
    vectorSum (Vector fp _ _) = withForeignPtr fp c_vector_sum_f32

instance VectorElem Int where
    vectorFromList xs = do
        withArrayLen xs $ \len ptr -> do
            vptr <- c_vector_from_i64 ptr len
            fp <- newForeignPtr c_vector_finalizer_f64 vptr  -- Uses same finalizer shape
            return $ Vector fp 0 len
    vectorGet (Vector fp off _) i = withForeignPtr fp $ \ptr ->
        c_vector_get_i64 ptr (off + i)
    vectorLen (Vector fp _ _) = withForeignPtr fp c_vector_len_i64
    vectorDot v1 v2 = foldl' (+) 0 (zipWith (*) v1 v2)  -- Pure implementation for Int
    vectorSum (Vector fp _ _) = withForeignPtr fp c_vector_sum_i64

-- ============================================================
-- Construction
-- ============================================================

-- | Empty vector.
empty :: Vector a
empty = Vector undefined 0 0

-- | Single element vector.
singleton :: VectorElem a => a -> Vector a
singleton x = fromList [x]

-- | Vector of @n@ copies of element.
replicate :: VectorElem a => Int -> a -> Vector a
replicate n x = fromList (P.replicate n x)

-- | Generate vector using function.
--
-- >>> generate 5 (*2)
-- [0, 2, 4, 6, 8]
generate :: VectorElem a => Int -> (Int -> a) -> Vector a
generate n f = fromList [f i | i <- [0..n-1]]

-- | Create vector from list.
fromList :: VectorElem a => [a] -> Vector a
fromList xs = unsafePerformIO $ vectorFromList xs
{-# NOINLINE fromList #-}

-- | Create vector from list with known length.
fromListN :: VectorElem a => Int -> [a] -> Vector a
fromListN n xs = fromList (P.take n xs)

-- | Enumerate from starting value.
--
-- >>> enumFromN 5 3
-- [5, 6, 7]
enumFromN :: (Num a, VectorElem a) => a -> Int -> Vector a
enumFromN start n = generate n (\i -> start + P.fromIntegral i)

-- | Enumerate with step.
--
-- >>> enumFromStepN 0 2 5
-- [0, 2, 4, 6, 8]
enumFromStepN :: (Num a, VectorElem a) => a -> a -> Int -> Vector a
enumFromStepN start step n = generate n (\i -> start + step * P.fromIntegral i)

-- ============================================================
-- Basic Operations
-- ============================================================

-- | Length of vector.
length :: Vector a -> Int
length = vecLength

-- | Is the vector empty?
null :: Vector a -> Bool
null v = length v == 0

-- | Index into vector (unsafe).
(!) :: VectorElem a => Vector a -> Int -> a
(!) v i = unsafePerformIO $ vectorGet v i
{-# NOINLINE (!) #-}

-- | Index into vector (safe).
(!?) :: VectorElem a => Vector a -> Int -> Maybe a
v !? i
    | i < 0 || i >= length v = Nothing
    | otherwise = Just (v ! i)

-- | First element (unsafe).
head :: VectorElem a => Vector a -> a
head v = v ! 0

-- | Last element (unsafe).
last :: VectorElem a => Vector a -> a
last v = v ! (length v - 1)

-- | All elements except first.
tail :: VectorElem a => Vector a -> Vector a
tail v = slice 1 (length v - 1) v

-- | All elements except last.
init :: VectorElem a => Vector a -> Vector a
init v = slice 0 (length v - 1) v

-- ============================================================
-- Slicing
-- ============================================================

-- | Extract slice starting at index with given length.
slice :: Int -> Int -> Vector a -> Vector a
slice start len v = v
    { vecOffset = vecOffset v + start
    , vecLength = len
    }

-- | Take first n elements.
take :: Int -> Vector a -> Vector a
take n v = slice 0 (P.min n (length v)) v

-- | Drop first n elements.
drop :: Int -> Vector a -> Vector a
drop n v = slice n (P.max 0 (length v - n)) v

-- | Split at index.
splitAt :: Int -> Vector a -> (Vector a, Vector a)
splitAt n v = (take n v, drop n v)

-- | Take while predicate holds.
takeWhile :: VectorElem a => (a -> Bool) -> Vector a -> Vector a
takeWhile p v = case findIndex (P.not . p) v of
    Nothing -> v
    Just i  -> take i v

-- | Drop while predicate holds.
dropWhile :: VectorElem a => (a -> Bool) -> Vector a -> Vector a
dropWhile p v = case findIndex (P.not . p) v of
    Nothing -> empty
    Just i  -> drop i v

-- ============================================================
-- Construction from Vectors
-- ============================================================

-- | Prepend element.
cons :: VectorElem a => a -> Vector a -> Vector a
cons x v = fromList (x : toList v)

-- | Append element.
snoc :: VectorElem a => Vector a -> a -> Vector a
snoc v x = fromList (toList v P.++ [x])

-- | Concatenate two vectors.
(++) :: VectorElem a => Vector a -> Vector a -> Vector a
v1 ++ v2 = fromList (toList v1 P.++ toList v2)

-- | Concatenate list of vectors.
concat :: VectorElem a => [Vector a] -> Vector a
concat vs = fromList (P.concatMap toList vs)

-- ============================================================
-- Element-wise Operations
-- ============================================================

-- | Map function over elements.
map :: (VectorElem a, VectorElem b) => (a -> b) -> Vector a -> Vector b
map f v = generate (length v) (\i -> f (v ! i))

-- | Map with index.
imap :: (VectorElem a, VectorElem b) => (Int -> a -> b) -> Vector a -> Vector b
imap f v = generate (length v) (\i -> f i (v ! i))

-- | Zip two vectors with function.
zipWith :: (VectorElem a, VectorElem b, VectorElem c) => (a -> b -> c) -> Vector a -> Vector b -> Vector c
zipWith f va vb =
    let n = P.min (length va) (length vb)
    in generate n (\i -> f (va ! i) (vb ! i))

-- | Zip three vectors with function.
zipWith3 :: (VectorElem a, VectorElem b, VectorElem c, VectorElem d) => (a -> b -> c -> d) -> Vector a -> Vector b -> Vector c -> Vector d
zipWith3 f va vb vc =
    let n = P.minimum [length va, length vb, length vc]
    in generate n (\i -> f (va ! i) (vb ! i) (vc ! i))

-- | Zip with index.
izipWith :: (VectorElem a, VectorElem b, VectorElem c) => (Int -> a -> b -> c) -> Vector a -> Vector b -> Vector c
izipWith f va vb =
    let n = P.min (length va) (length vb)
    in generate n (\i -> f i (va ! i) (vb ! i))

-- | Zip three with index.
izipWith3 :: (VectorElem a, VectorElem b, VectorElem c, VectorElem d) => (Int -> a -> b -> c -> d) -> Vector a -> Vector b -> Vector c -> Vector d
izipWith3 f va vb vc =
    let n = P.minimum [length va, length vb, length vc]
    in generate n (\i -> f i (va ! i) (vb ! i) (vc ! i))

-- ============================================================
-- Folds
-- ============================================================

-- | Left fold.
foldl :: VectorElem a => (b -> a -> b) -> b -> Vector a -> b
foldl f z v = go 0 z
  where
    n = length v
    go !i !acc
        | i >= n    = acc
        | otherwise = go (i + 1) (f acc (v ! i))

-- | Strict left fold.
foldl' :: VectorElem a => (b -> a -> b) -> b -> Vector a -> b
foldl' = foldl  -- Already strict due to bang patterns

-- | Left fold without starting value (unsafe).
foldl1 :: VectorElem a => (a -> a -> a) -> Vector a -> a
foldl1 f v = foldl f (head v) (tail v)

-- | Strict left fold without starting value.
foldl1' :: VectorElem a => (a -> a -> a) -> Vector a -> a
foldl1' = foldl1

-- | Right fold.
foldr :: VectorElem a => (a -> b -> b) -> b -> Vector a -> b
foldr f z v = go (length v - 1)
  where
    go i
        | i < 0     = z
        | otherwise = f (v ! i) (go (i - 1))

-- | Right fold without starting value (unsafe).
foldr1 :: VectorElem a => (a -> a -> a) -> Vector a -> a
foldr1 f v = foldr f (last v) (init v)

-- | Left fold with index.
ifoldl :: VectorElem a => (b -> Int -> a -> b) -> b -> Vector a -> b
ifoldl f z v = go 0 z
  where
    n = length v
    go !i !acc
        | i >= n    = acc
        | otherwise = go (i + 1) (f acc i (v ! i))

-- | Strict left fold with index.
ifoldl' :: VectorElem a => (b -> Int -> a -> b) -> b -> Vector a -> b
ifoldl' = ifoldl

-- | Right fold with index.
ifoldr :: VectorElem a => (Int -> a -> b -> b) -> b -> Vector a -> b
ifoldr f z v = go 0
  where
    n = length v
    go i
        | i >= n    = z
        | otherwise = f i (v ! i) (go (i + 1))

-- ============================================================
-- Specialized Folds
-- ============================================================

-- | Sum of elements.
sum :: (Num a, VectorElem a) => Vector a -> a
sum = foldl' (+) 0

-- | Product of elements.
product :: (Num a, VectorElem a) => Vector a -> a
product = foldl' (*) 1

-- | Maximum element (unsafe on empty).
maximum :: (Ord a, VectorElem a) => Vector a -> a
maximum = foldl1' P.max

-- | Minimum element (unsafe on empty).
minimum :: (Ord a, VectorElem a) => Vector a -> a
minimum = foldl1' P.min

-- | Maximum by comparison function.
maximumBy :: VectorElem a => (a -> a -> Ordering) -> Vector a -> a
maximumBy cmp = foldl1' (\a b -> if cmp a b == GT then a else b)

-- | Minimum by comparison function.
minimumBy :: VectorElem a => (a -> a -> Ordering) -> Vector a -> a
minimumBy cmp = foldl1' (\a b -> if cmp a b == LT then a else b)

-- | All elements satisfy predicate.
all :: VectorElem a => (a -> Bool) -> Vector a -> Bool
all p = foldl' (\acc x -> acc P.&& p x) True

-- | Any element satisfies predicate.
any :: VectorElem a => (a -> Bool) -> Vector a -> Bool
any p = foldl' (\acc x -> acc P.|| p x) False

-- ============================================================
-- Scans
-- ============================================================

-- | Prefix scan (exclusive).
prescanl :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
prescanl f z v = generate (length v) (\i ->
    foldl' f z (take i v))

-- | Strict prefix scan.
prescanl' :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
prescanl' = prescanl

-- | Postfix scan (inclusive).
postscanl :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
postscanl f z v = generate (length v) (\i ->
    foldl' f z (take (i + 1) v))

-- | Strict postfix scan.
postscanl' :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
postscanl' = postscanl

-- | Scan (like scanl but returns vector).
scanl :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
scanl = prescanl

-- | Strict scan.
scanl' :: (VectorElem a, VectorElem b) => (a -> b -> a) -> a -> Vector b -> Vector a
scanl' = prescanl'

-- | Scan without starting value.
scanl1 :: VectorElem a => (a -> a -> a) -> Vector a -> Vector a
scanl1 f v = postscanl f (head v) (tail v)

-- | Strict scan without starting value.
scanl1' :: VectorElem a => (a -> a -> a) -> Vector a -> Vector a
scanl1' = scanl1

-- ============================================================
-- Search
-- ============================================================

-- | Is element in vector?
elem :: (Eq a, VectorElem a) => a -> Vector a -> Bool
elem x = any (== x)

-- | Is element not in vector?
notElem :: (Eq a, VectorElem a) => a -> Vector a -> Bool
notElem x = P.not . elem x

-- | Find first element satisfying predicate.
find :: VectorElem a => (a -> Bool) -> Vector a -> Maybe a
find p v = case findIndex p v of
    Nothing -> Nothing
    Just i  -> Just (v ! i)

-- | Find index of first element satisfying predicate.
findIndex :: VectorElem a => (a -> Bool) -> Vector a -> Maybe Int
findIndex p v = go 0
  where
    n = length v
    go i
        | i >= n    = Nothing
        | p (v ! i) = Just i
        | otherwise = go (i + 1)

-- | Find index of element.
elemIndex :: (Eq a, VectorElem a) => a -> Vector a -> Maybe Int
elemIndex x = findIndex (== x)

-- | Find all indices of element.
elemIndices :: (Eq a, VectorElem a) => a -> Vector a -> Vector Int
elemIndices x v = fromList [i | i <- [0..length v - 1], v ! i == x]

-- ============================================================
-- Numeric Operations
-- ============================================================

-- | Dot product of two vectors.
--
-- >>> dot [1, 2, 3] [4, 5, 6]
-- 32
dot :: VectorElem a => Vector a -> Vector a -> a
dot v1 v2 = unsafePerformIO $ vectorDot v1 v2
{-# NOINLINE dot #-}

-- | Euclidean norm (L2).
norm :: (Floating a, VectorElem a) => Vector a -> a
norm v = P.sqrt (dot v v)

-- | Normalize to unit length.
normalize :: (Floating a, VectorElem a) => Vector a -> Vector a
normalize v = scale (1 / norm v) v

-- | Element-wise addition.
add :: (Num a, VectorElem a) => Vector a -> Vector a -> Vector a
add = zipWith (+)

-- | Element-wise subtraction.
sub :: (Num a, VectorElem a) => Vector a -> Vector a -> Vector a
sub = zipWith (-)

-- | Element-wise multiplication.
mul :: (Num a, VectorElem a) => Vector a -> Vector a -> Vector a
mul = zipWith (*)

-- | Element-wise division.
div :: (Fractional a, VectorElem a) => Vector a -> Vector a -> Vector a
div = zipWith (/)

-- | Scale vector by scalar.
scale :: (Num a, VectorElem a) => a -> Vector a -> Vector a
scale k = map (* k)

-- | Negate all elements.
negate :: (Num a, VectorElem a) => Vector a -> Vector a
negate = map P.negate

-- | Absolute value of all elements.
abs :: (Num a, VectorElem a) => Vector a -> Vector a
abs = map P.abs

-- ============================================================
-- Sorting
-- ============================================================

-- | Sort in ascending order.
sort :: (Ord a, VectorElem a) => Vector a -> Vector a
sort v = fromList (P.sort (toList v))

-- | Sort by comparison function.
sortBy :: VectorElem a => (a -> a -> Ordering) -> Vector a -> Vector a
sortBy cmp v = fromList (P.sortBy cmp (toList v))

-- | Pair each element with its index.
indexed :: (VectorElem a, VectorElem (Int, a)) => Vector a -> Vector (Int, a)
indexed v = generate (length v) (\i -> (i, v ! i))

-- ============================================================
-- Conversion
-- ============================================================

-- | Convert to list.
toList :: VectorElem a => Vector a -> [a]
toList v = [v ! i | i <- [0..length v - 1]]

-- | Convert between vector types.
convert :: Vector a -> UVector a
convert (Vector fp off len) = UVector fp off len
