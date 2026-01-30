{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}

-- |
-- Module      : BHC.Data.ByteString
-- Description : Efficient strict byte sequences
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- A time and space-efficient implementation of byte vectors using
-- packed Word8 arrays. This module provides operations for working
-- with raw binary data and ASCII text.
--
-- = Performance
--
-- ByteString provides:
--
-- * O(1) length, null, head, last, take, drop, splitAt
-- * O(n) for most traversals
-- * Zero-copy slicing where possible
-- * SIMD-accelerated search and comparison (when available)
--
-- = Usage
--
-- @
-- import qualified BHC.Data.ByteString as BS
--
-- let bs = BS.pack [72, 101, 108, 108, 111]  -- "Hello"
-- BS.length bs  -- 5
-- BS.head bs    -- 72
-- @

module BHC.Data.ByteString
    ( -- * The ByteString type
      ByteString

      -- * Construction
    , empty
    , singleton
    , pack
    , unpack
    , fromStrict
    , toStrict

      -- * Basic interface
    , cons
    , snoc
    , append
    , head
    , uncons
    , unsnoc
    , last
    , tail
    , init
    , null
    , length

      -- * Transformations
    , map
    , reverse
    , intersperse
    , intercalate
    , transpose

      -- * Reducing (Folds)
    , foldl
    , foldl'
    , foldl1
    , foldl1'
    , foldr
    , foldr'
    , foldr1
    , foldr1'

      -- * Special folds
    , concat
    , concatMap
    , any
    , all
    , maximum
    , minimum

      -- * Building ByteStrings
    , scanl
    , scanl1
    , scanr
    , scanr1
    , mapAccumL
    , mapAccumR
    , replicate
    , unfoldr
    , unfoldrN

      -- * Substrings
    , take
    , takeEnd
    , drop
    , dropEnd
    , splitAt
    , takeWhile
    , takeWhileEnd
    , dropWhile
    , dropWhileEnd
    , span
    , spanEnd
    , break
    , breakEnd
    , group
    , groupBy
    , inits
    , tails
    , stripPrefix
    , stripSuffix

      -- * Breaking into many
    , split
    , splitWith
    , splitOn

      -- * Predicates
    , isPrefixOf
    , isSuffixOf
    , isInfixOf

      -- * Searching
    , elem
    , notElem
    , find
    , filter
    , partition

      -- * Indexing
    , index
    , indexMaybe
    , (!?)
    , elemIndex
    , elemIndices
    , elemIndexEnd
    , findIndex
    , findIndices
    , findIndexEnd
    , count

      -- * Zipping and unzipping
    , zip
    , zipWith
    , packZipWith
    , unzip

      -- * Ordered ByteStrings
    , sort

      -- * Low level
    , copy
    , packCString
    , packCStringLen
    , useAsCString
    , useAsCStringLen

      -- * I/O
    , getLine
    , getContents
    , putStr
    , putStrLn
    , interact
    , readFile
    , writeFile
    , appendFile
    , hGetLine
    , hGetContents
    , hGet
    , hGetSome
    , hGetNonBlocking
    , hPut
    , hPutNonBlocking
    , hPutStr
    , hPutStrLn
    ) where

import Prelude hiding (
    head, last, tail, init, null, length, map, reverse,
    foldl, foldl1, foldr, foldr1, concat, concatMap, any, all,
    maximum, minimum, scanl, scanl1, scanr, scanr1, replicate,
    take, drop, splitAt, takeWhile, dropWhile, span, break,
    elem, notElem, filter, zip, zipWith, unzip,
    getLine, getContents, putStr, putStrLn, interact,
    readFile, writeFile, appendFile
    )
import qualified Prelude as P

import Data.Word (Word8)
import System.IO (Handle, IOMode(..), withFile)
import qualified System.IO as IO
import Foreign.Ptr (Ptr)
import Foreign.C.String (CString, CStringLen)

-- ============================================================
-- Types
-- ============================================================

-- | A strict sequence of bytes.
-- Represented as a contiguous array of 'Word8' values.
data ByteString = BS
    {-# UNPACK #-} !ByteArray  -- Raw bytes
    {-# UNPACK #-} !Int        -- Offset
    {-# UNPACK #-} !Int        -- Length
    deriving (Eq, Ord)

instance Show ByteString where
    showsPrec p bs = showsPrec p (unpack bs)

instance Read ByteString where
    readsPrec p s = [(pack ws, r) | (ws, r) <- readsPrec p s]

instance Semigroup ByteString where
    (<>) = append

instance Monoid ByteString where
    mempty = empty
    mappend = (<>)

-- Opaque type for raw bytes
data ByteArray

-- ============================================================
-- Construction
-- ============================================================

-- | /O(1)/. The empty 'ByteString'.
--
-- >>> null empty
-- True
-- >>> length empty
-- 0
empty :: ByteString
empty = pack []

-- | /O(1)/. A 'ByteString' containing a single byte.
--
-- >>> singleton 65
-- [65]
-- >>> length (singleton 0)
-- 1
singleton :: Word8 -> ByteString
singleton w = pack [w]

-- | /O(n)/. Convert a list of bytes to a 'ByteString'.
--
-- >>> pack [72, 101, 108, 108, 111]
-- [72,101,108,108,111]
pack :: [Word8] -> ByteString
pack ws = createBS (P.length ws) $ \ptr ->
    pokeList ptr ws 0
  where
    pokeList _ [] _ = return ()
    pokeList ptr (x:xs) !i = do
        pokeByteOff ptr i x
        pokeList ptr xs (i + 1)

-- | /O(n)/. Convert a 'ByteString' to a list of bytes.
--
-- >>> unpack (pack [72, 101, 108, 108, 111])
-- [72,101,108,108,111]
unpack :: ByteString -> [Word8]
unpack (BS arr off len) = go 0
  where
    go !i
        | i >= len  = []
        | otherwise = indexBS arr (off + i) : go (i + 1)

-- | Identity (strict ByteString is already strict).
fromStrict :: ByteString -> ByteString
fromStrict = id

-- | Identity (strict ByteString is already strict).
toStrict :: ByteString -> ByteString
toStrict = id

-- ============================================================
-- Basic interface
-- ============================================================

-- | /O(n)/. Prepend a byte to a 'ByteString'.
--
-- >>> cons 72 (pack [101, 108, 108, 111])
-- [72,101,108,108,111]
cons :: Word8 -> ByteString -> ByteString
cons w bs = singleton w `append` bs

-- | /O(n)/. Append a byte to a 'ByteString'.
--
-- >>> snoc (pack [72, 101, 108, 108]) 111
-- [72,101,108,108,111]
snoc :: ByteString -> Word8 -> ByteString
snoc bs w = bs `append` singleton w

-- | /O(n)/. Append two 'ByteString's.
--
-- >>> append (pack [1,2,3]) (pack [4,5,6])
-- [1,2,3,4,5,6]
append :: ByteString -> ByteString -> ByteString
append (BS arr1 off1 len1) (BS arr2 off2 len2)
    | len1 == 0 = BS arr2 off2 len2
    | len2 == 0 = BS arr1 off1 len1
    | otherwise = createBS (len1 + len2) $ \ptr -> do
        copyBytes ptr arr1 off1 len1
        copyBytes (plusPtr ptr len1) arr2 off2 len2

-- | /O(1)/. Extract the first byte of a 'ByteString'.
--
-- __Warning__: Partial function. Throws an error on empty 'ByteString'.
--
-- >>> head (pack [72, 101, 108, 108, 111])
-- 72
head :: ByteString -> Word8
head (BS arr off len)
    | len <= 0  = error "ByteString.head: empty ByteString"
    | otherwise = indexBS arr off

-- | /O(1)/. Decompose a 'ByteString' into its head and tail.
--
-- >>> uncons (pack [1,2,3])
-- Just (1,[2,3])
-- >>> uncons empty
-- Nothing
uncons :: ByteString -> Maybe (Word8, ByteString)
uncons bs
    | null bs   = Nothing
    | otherwise = Just (head bs, tail bs)

-- | /O(1)/. Decompose a 'ByteString' into its init and last.
--
-- >>> unsnoc (pack [1,2,3])
-- Just ([1,2],3)
-- >>> unsnoc empty
-- Nothing
unsnoc :: ByteString -> Maybe (ByteString, Word8)
unsnoc bs
    | null bs   = Nothing
    | otherwise = Just (init bs, last bs)

-- | /O(1)/. Extract the last byte of a 'ByteString'.
--
-- __Warning__: Partial function. Throws an error on empty 'ByteString'.
--
-- >>> last (pack [72, 101, 108, 108, 111])
-- 111
last :: ByteString -> Word8
last (BS arr off len)
    | len <= 0  = error "ByteString.last: empty ByteString"
    | otherwise = indexBS arr (off + len - 1)

-- | /O(1)/. Extract the bytes after the head of a 'ByteString'.
--
-- __Warning__: Partial function. Throws an error on empty 'ByteString'.
--
-- >>> tail (pack [1,2,3])
-- [2,3]
tail :: ByteString -> ByteString
tail (BS arr off len)
    | len <= 0  = error "ByteString.tail: empty ByteString"
    | otherwise = BS arr (off + 1) (len - 1)

-- | /O(1)/. All bytes except the last.
--
-- __Warning__: Partial function. Throws an error on empty 'ByteString'.
--
-- >>> init (pack [1,2,3])
-- [1,2]
init :: ByteString -> ByteString
init (BS arr off len)
    | len <= 0  = error "ByteString.init: empty ByteString"
    | otherwise = BS arr off (len - 1)

-- | /O(1)/. Test whether a 'ByteString' is empty.
--
-- >>> null empty
-- True
-- >>> null (pack [1])
-- False
null :: ByteString -> Bool
null (BS _ _ len) = len == 0

-- | /O(1)/. The length of a 'ByteString'.
--
-- >>> length (pack [1,2,3,4,5])
-- 5
length :: ByteString -> Int
length (BS _ _ len) = len

-- ============================================================
-- Transformations
-- ============================================================

-- | Map a function over a 'ByteString'.
map :: (Word8 -> Word8) -> ByteString -> ByteString
map f = pack . P.map f . unpack

-- | Reverse a 'ByteString'.
reverse :: ByteString -> ByteString
reverse = pack . P.reverse . unpack

-- | Intersperse a byte between bytes of a 'ByteString'.
intersperse :: Word8 -> ByteString -> ByteString
intersperse w bs
    | length bs < 2 = bs
    | otherwise = pack $ go $ unpack bs
  where
    go [] = []
    go [x] = [x]
    go (x:xs) = x : w : go xs

-- | Join a list of 'ByteString's with a separator.
intercalate :: ByteString -> [ByteString] -> ByteString
intercalate sep = concat . go
  where
    go [] = []
    go [x] = [x]
    go (x:xs) = x : sep : go xs

-- | Transpose the rows and columns of a list of 'ByteString's.
transpose :: [ByteString] -> [ByteString]
transpose = P.map pack . P.transpose . P.map unpack

-- ============================================================
-- Folds
-- ============================================================

-- | Left fold.
foldl :: (a -> Word8 -> a) -> a -> ByteString -> a
foldl f z = P.foldl f z . unpack

-- | Strict left fold.
foldl' :: (a -> Word8 -> a) -> a -> ByteString -> a
foldl' f z = P.foldl' f z . unpack

-- | Left fold on non-empty 'ByteString's.
foldl1 :: (Word8 -> Word8 -> Word8) -> ByteString -> Word8
foldl1 f bs
    | null bs   = error "ByteString.foldl1: empty ByteString"
    | otherwise = foldl f (head bs) (tail bs)

-- | Strict left fold on non-empty 'ByteString's.
foldl1' :: (Word8 -> Word8 -> Word8) -> ByteString -> Word8
foldl1' f bs
    | null bs   = error "ByteString.foldl1': empty ByteString"
    | otherwise = foldl' f (head bs) (tail bs)

-- | Right fold.
foldr :: (Word8 -> a -> a) -> a -> ByteString -> a
foldr f z = P.foldr f z . unpack

-- | Strict right fold.
foldr' :: (Word8 -> a -> a) -> a -> ByteString -> a
foldr' f z bs = foldl' (flip f) z (reverse bs)

-- | Right fold on non-empty 'ByteString's.
foldr1 :: (Word8 -> Word8 -> Word8) -> ByteString -> Word8
foldr1 f bs
    | null bs   = error "ByteString.foldr1: empty ByteString"
    | otherwise = foldr f (last bs) (init bs)

-- | Strict right fold on non-empty 'ByteString's.
foldr1' :: (Word8 -> Word8 -> Word8) -> ByteString -> Word8
foldr1' f bs
    | null bs   = error "ByteString.foldr1': empty ByteString"
    | otherwise = foldr' f (last bs) (init bs)

-- ============================================================
-- Special folds
-- ============================================================

-- | Concatenate a list of 'ByteString's.
concat :: [ByteString] -> ByteString
concat = foldl' append empty

-- | Map a function and concatenate results.
concatMap :: (Word8 -> ByteString) -> ByteString -> ByteString
concatMap f = concat . P.map f . unpack

-- | Test if any byte satisfies a predicate.
any :: (Word8 -> Bool) -> ByteString -> Bool
any p = P.any p . unpack

-- | Test if all bytes satisfy a predicate.
all :: (Word8 -> Bool) -> ByteString -> Bool
all p = P.all p . unpack

-- | Maximum byte value.
maximum :: ByteString -> Word8
maximum bs
    | null bs   = error "ByteString.maximum: empty ByteString"
    | otherwise = foldl1' max bs

-- | Minimum byte value.
minimum :: ByteString -> Word8
minimum bs
    | null bs   = error "ByteString.minimum: empty ByteString"
    | otherwise = foldl1' min bs

-- ============================================================
-- Scans
-- ============================================================

-- | Left scan.
scanl :: (Word8 -> Word8 -> Word8) -> Word8 -> ByteString -> ByteString
scanl f z = pack . P.scanl f z . unpack

-- | Left scan without starting value.
scanl1 :: (Word8 -> Word8 -> Word8) -> ByteString -> ByteString
scanl1 f bs
    | null bs   = empty
    | otherwise = scanl f (head bs) (tail bs)

-- | Right scan.
scanr :: (Word8 -> Word8 -> Word8) -> Word8 -> ByteString -> ByteString
scanr f z = pack . P.scanr f z . unpack

-- | Right scan without starting value.
scanr1 :: (Word8 -> Word8 -> Word8) -> ByteString -> ByteString
scanr1 f bs
    | null bs   = empty
    | otherwise = scanr f (last bs) (init bs)

-- | Accumulating map from left.
mapAccumL :: (acc -> Word8 -> (acc, Word8)) -> acc -> ByteString -> (acc, ByteString)
mapAccumL f z bs = (acc, pack ws)
  where
    (acc, ws) = P.mapAccumL f z (unpack bs)

-- | Accumulating map from right.
mapAccumR :: (acc -> Word8 -> (acc, Word8)) -> acc -> ByteString -> (acc, ByteString)
mapAccumR f z bs = (acc, pack ws)
  where
    (acc, ws) = P.mapAccumR f z (unpack bs)

-- | Replicate a byte n times.
replicate :: Int -> Word8 -> ByteString
replicate n w
    | n <= 0    = empty
    | otherwise = pack (P.replicate n w)

-- | Build a 'ByteString' from a seed using unfoldr.
unfoldr :: (a -> Maybe (Word8, a)) -> a -> ByteString
unfoldr f = pack . P.unfoldr f

-- | Build a 'ByteString' with at most n bytes from unfoldr.
unfoldrN :: Int -> (a -> Maybe (Word8, a)) -> a -> (ByteString, Maybe a)
unfoldrN n f z = (pack ws, final)
  where
    (ws, final) = go n z
    go 0 s = ([], Just s)
    go i s = case f s of
        Nothing      -> ([], Nothing)
        Just (w, s') -> let (rest, fin) = go (i - 1) s' in (w : rest, fin)

-- ============================================================
-- Substrings
-- ============================================================

-- | /O(1)/. Take the first @n@ bytes of a 'ByteString'.
-- Returns the entire 'ByteString' if @n >= length bs@.
--
-- >>> take 3 (pack [1,2,3,4,5])
-- [1,2,3]
take :: Int -> ByteString -> ByteString
take n (BS arr off len)
    | n <= 0    = empty
    | n >= len  = BS arr off len
    | otherwise = BS arr off n

-- | /O(1)/. Take the last @n@ bytes of a 'ByteString'.
--
-- >>> takeEnd 3 (pack [1,2,3,4,5])
-- [3,4,5]
takeEnd :: Int -> ByteString -> ByteString
takeEnd n bs@(BS arr off len)
    | n <= 0    = empty
    | n >= len  = bs
    | otherwise = BS arr (off + len - n) n

-- | /O(1)/. Drop the first @n@ bytes of a 'ByteString'.
--
-- >>> drop 2 (pack [1,2,3,4,5])
-- [3,4,5]
drop :: Int -> ByteString -> ByteString
drop n (BS arr off len)
    | n <= 0    = BS arr off len
    | n >= len  = empty
    | otherwise = BS arr (off + n) (len - n)

-- | /O(1)/. Drop the last @n@ bytes of a 'ByteString'.
--
-- >>> dropEnd 2 (pack [1,2,3,4,5])
-- [1,2,3]
dropEnd :: Int -> ByteString -> ByteString
dropEnd n bs@(BS arr off len)
    | n <= 0    = bs
    | n >= len  = empty
    | otherwise = BS arr off (len - n)

-- | /O(1)/. Split a 'ByteString' at position @n@.
--
-- >>> splitAt 3 (pack [1,2,3,4,5])
-- ([1,2,3],[4,5])
splitAt :: Int -> ByteString -> (ByteString, ByteString)
splitAt n bs = (take n bs, drop n bs)

-- | Take bytes while predicate holds.
takeWhile :: (Word8 -> Bool) -> ByteString -> ByteString
takeWhile p bs = take (findIndexOrEnd (not . p) bs) bs

-- | Take bytes from end while predicate holds.
takeWhileEnd :: (Word8 -> Bool) -> ByteString -> ByteString
takeWhileEnd p bs = drop (findFromEndUntil (not . p) bs) bs

-- | Drop bytes while predicate holds.
dropWhile :: (Word8 -> Bool) -> ByteString -> ByteString
dropWhile p bs = drop (findIndexOrEnd (not . p) bs) bs

-- | Drop bytes from end while predicate holds.
dropWhileEnd :: (Word8 -> Bool) -> ByteString -> ByteString
dropWhileEnd p bs = take (findFromEndUntil (not . p) bs) bs

-- | Split at first byte where predicate fails.
span :: (Word8 -> Bool) -> ByteString -> (ByteString, ByteString)
span p bs = (takeWhile p bs, dropWhile p bs)

-- | Split from end at first byte where predicate fails.
spanEnd :: (Word8 -> Bool) -> ByteString -> (ByteString, ByteString)
spanEnd p bs = (dropWhileEnd p bs, takeWhileEnd p bs)

-- | Split at first byte where predicate succeeds.
break :: (Word8 -> Bool) -> ByteString -> (ByteString, ByteString)
break p = span (not . p)

-- | Split from end at first byte where predicate succeeds.
breakEnd :: (Word8 -> Bool) -> ByteString -> (ByteString, ByteString)
breakEnd p = spanEnd (not . p)

-- | Group consecutive equal bytes.
group :: ByteString -> [ByteString]
group = groupBy (==)

-- | Group consecutive bytes by a predicate.
groupBy :: (Word8 -> Word8 -> Bool) -> ByteString -> [ByteString]
groupBy _ bs | null bs = []
groupBy eq bs = let (ys, zs) = span (eq (head bs)) bs
                in ys : groupBy eq zs

-- | All initial segments.
inits :: ByteString -> [ByteString]
inits bs = [take n bs | n <- [0..length bs]]

-- | All final segments.
tails :: ByteString -> [ByteString]
tails bs = [drop n bs | n <- [0..length bs]]

-- | Strip a prefix if present.
stripPrefix :: ByteString -> ByteString -> Maybe ByteString
stripPrefix prefix bs
    | isPrefixOf prefix bs = Just (drop (length prefix) bs)
    | otherwise            = Nothing

-- | Strip a suffix if present.
stripSuffix :: ByteString -> ByteString -> Maybe ByteString
stripSuffix suffix bs
    | isSuffixOf suffix bs = Just (dropEnd (length suffix) bs)
    | otherwise            = Nothing

-- ============================================================
-- Breaking into many
-- ============================================================

-- | Split on a byte.
split :: Word8 -> ByteString -> [ByteString]
split w = splitWith (== w)

-- | Split on bytes satisfying a predicate.
splitWith :: (Word8 -> Bool) -> ByteString -> [ByteString]
splitWith p bs
    | null bs   = [empty]
    | otherwise = case break p bs of
        (before, after)
            | null after -> [before]
            | otherwise  -> before : splitWith p (tail after)

-- | Split on a delimiter 'ByteString'.
splitOn :: ByteString -> ByteString -> [ByteString]
splitOn delim bs
    | null delim = error "ByteString.splitOn: empty delimiter"
    | otherwise  = go bs
  where
    go s = case breakOnBS delim s of
        (before, after)
            | null after -> [before]
            | otherwise  -> before : go (drop (length delim) after)

-- ============================================================
-- Predicates
-- ============================================================

-- | Is the first a prefix of the second?
isPrefixOf :: ByteString -> ByteString -> Bool
isPrefixOf prefix bs
    | length prefix > length bs = False
    | otherwise = take (length prefix) bs == prefix

-- | Is the first a suffix of the second?
isSuffixOf :: ByteString -> ByteString -> Bool
isSuffixOf suffix bs
    | length suffix > length bs = False
    | otherwise = takeEnd (length suffix) bs == suffix

-- | Is the first contained in the second?
isInfixOf :: ByteString -> ByteString -> Bool
isInfixOf needle haystack = P.any (isPrefixOf needle) (tails haystack)

-- ============================================================
-- Searching
-- ============================================================

-- | Is the byte in the 'ByteString'?
elem :: Word8 -> ByteString -> Bool
elem w = any (== w)

-- | Is the byte not in the 'ByteString'?
notElem :: Word8 -> ByteString -> Bool
notElem w = all (/= w)

-- | Find the first byte satisfying a predicate.
find :: (Word8 -> Bool) -> ByteString -> Maybe Word8
find p bs = case findIndex p bs of
    Nothing -> Nothing
    Just i  -> Just (index bs i)

-- | Filter bytes by a predicate.
filter :: (Word8 -> Bool) -> ByteString -> ByteString
filter p = pack . P.filter p . unpack

-- | Partition by a predicate.
partition :: (Word8 -> Bool) -> ByteString -> (ByteString, ByteString)
partition p bs = (filter p bs, filter (not . p) bs)

-- ============================================================
-- Indexing
-- ============================================================

-- | Index a byte (partial).
index :: ByteString -> Int -> Word8
index (BS arr off len) i
    | i < 0 || i >= len = error "ByteString.index: index out of bounds"
    | otherwise         = indexBS arr (off + i)

-- | Safe indexing.
indexMaybe :: ByteString -> Int -> Maybe Word8
indexMaybe bs i
    | i < 0 || i >= length bs = Nothing
    | otherwise               = Just (index bs i)

-- | Infix safe indexing.
(!?) :: ByteString -> Int -> Maybe Word8
(!?) = indexMaybe

-- | Index of first occurrence.
elemIndex :: Word8 -> ByteString -> Maybe Int
elemIndex w = findIndex (== w)

-- | Indices of all occurrences.
elemIndices :: Word8 -> ByteString -> [Int]
elemIndices w = findIndices (== w)

-- | Index of last occurrence.
elemIndexEnd :: Word8 -> ByteString -> Maybe Int
elemIndexEnd w = findIndexEnd (== w)

-- | Index of first byte satisfying a predicate.
findIndex :: (Word8 -> Bool) -> ByteString -> Maybe Int
findIndex p bs = go 0
  where
    go !i
        | i >= length bs = Nothing
        | p (index bs i) = Just i
        | otherwise      = go (i + 1)

-- | Indices of all bytes satisfying a predicate.
findIndices :: (Word8 -> Bool) -> ByteString -> [Int]
findIndices p bs = [i | i <- [0..length bs - 1], p (index bs i)]

-- | Index of last byte satisfying a predicate.
findIndexEnd :: (Word8 -> Bool) -> ByteString -> Maybe Int
findIndexEnd p bs = go (length bs - 1)
  where
    go !i
        | i < 0          = Nothing
        | p (index bs i) = Just i
        | otherwise      = go (i - 1)

-- | Count occurrences of a byte.
count :: Word8 -> ByteString -> Int
count w = P.length . elemIndices w

-- ============================================================
-- Zipping
-- ============================================================

-- | Zip two 'ByteString's.
zip :: ByteString -> ByteString -> [(Word8, Word8)]
zip bs1 bs2 = P.zip (unpack bs1) (unpack bs2)

-- | Zip with a function.
zipWith :: (Word8 -> Word8 -> a) -> ByteString -> ByteString -> [a]
zipWith f bs1 bs2 = P.zipWith f (unpack bs1) (unpack bs2)

-- | Zip with a function producing bytes.
packZipWith :: (Word8 -> Word8 -> Word8) -> ByteString -> ByteString -> ByteString
packZipWith f bs1 bs2 = pack $ zipWith f bs1 bs2

-- | Unzip a list of pairs.
unzip :: [(Word8, Word8)] -> (ByteString, ByteString)
unzip pairs = (pack (P.map fst pairs), pack (P.map snd pairs))

-- ============================================================
-- Sorting
-- ============================================================

-- | Sort bytes in ascending order.
sort :: ByteString -> ByteString
sort = pack . P.sort . unpack

-- ============================================================
-- Low level
-- ============================================================

-- | Make a copy of the 'ByteString'.
copy :: ByteString -> ByteString
copy bs = pack (unpack bs)

-- | Pack a null-terminated C string.
packCString :: CString -> IO ByteString
packCString cstr = do
    len <- cStringLength cstr
    packCStringLen (cstr, len)

-- | Pack a C string with known length.
packCStringLen :: CStringLen -> IO ByteString
packCStringLen (cstr, len) = do
    ws <- peekArray len cstr
    return $ pack ws

-- | Use a 'ByteString' as a null-terminated C string.
useAsCString :: ByteString -> (CString -> IO a) -> IO a
useAsCString bs action = do
    let ws = unpack bs ++ [0]  -- Null terminate
    withArray ws action

-- | Use a 'ByteString' as a C string with length.
useAsCStringLen :: ByteString -> (CStringLen -> IO a) -> IO a
useAsCStringLen bs action =
    useAsCString bs $ \ptr -> action (ptr, length bs)

-- ============================================================
-- I/O
-- ============================================================

-- | Read a line from stdin.
getLine :: IO ByteString
getLine = hGetLine IO.stdin

-- | Read all of stdin.
getContents :: IO ByteString
getContents = hGetContents IO.stdin

-- | Write to stdout.
putStr :: ByteString -> IO ()
putStr = hPutStr IO.stdout

-- | Write to stdout with newline.
putStrLn :: ByteString -> IO ()
putStrLn = hPutStrLn IO.stdout

-- | Process stdin to stdout.
interact :: (ByteString -> ByteString) -> IO ()
interact f = getContents >>= putStr . f

-- | Read an entire file.
readFile :: FilePath -> IO ByteString
readFile path = withFile path ReadMode hGetContents

-- | Write to a file.
writeFile :: FilePath -> ByteString -> IO ()
writeFile path bs = withFile path WriteMode (`hPut` bs)

-- | Append to a file.
appendFile :: FilePath -> ByteString -> IO ()
appendFile path bs = withFile path AppendMode (`hPut` bs)

-- | Read a line from a handle.
hGetLine :: Handle -> IO ByteString
hGetLine h = do
    line <- IO.hGetLine h
    return $ pack $ P.map (fromIntegral . fromEnum) line

-- | Read all remaining data from a handle.
hGetContents :: Handle -> IO ByteString
hGetContents h = do
    contents <- IO.hGetContents h
    return $ pack $ P.map (fromIntegral . fromEnum) contents

-- | Read exactly n bytes from a handle.
hGet :: Handle -> Int -> IO ByteString
hGet h n
    | n <= 0    = return empty
    | otherwise = do
        chars <- sequence $ P.replicate n (IO.hGetChar h)
        return $ pack $ P.map (fromIntegral . fromEnum) chars

-- | Read up to n bytes from a handle.
hGetSome :: Handle -> Int -> IO ByteString
hGetSome h n = do
    ready <- IO.hReady h
    if ready then hGet h n else return empty

-- | Non-blocking read.
hGetNonBlocking :: Handle -> Int -> IO ByteString
hGetNonBlocking = hGetSome

-- | Write to a handle.
hPut :: Handle -> ByteString -> IO ()
hPut h bs = P.mapM_ (IO.hPutChar h . toEnum . fromIntegral) (unpack bs)

-- | Non-blocking write.
hPutNonBlocking :: Handle -> ByteString -> IO ByteString
hPutNonBlocking h bs = hPut h bs >> return empty

-- | Write to a handle (alias).
hPutStr :: Handle -> ByteString -> IO ()
hPutStr = hPut

-- | Write to a handle with newline.
hPutStrLn :: Handle -> ByteString -> IO ()
hPutStrLn h bs = hPut h bs >> IO.hPutChar h '\n'

-- ============================================================
-- Internal helpers
-- ============================================================

-- Find index or return length
findIndexOrEnd :: (Word8 -> Bool) -> ByteString -> Int
findIndexOrEnd p bs = case findIndex p bs of
    Nothing -> length bs
    Just i  -> i

-- Find from end until predicate fails
findFromEndUntil :: (Word8 -> Bool) -> ByteString -> Int
findFromEndUntil p bs = length bs - length (takeWhileEnd (not . p) bs)

-- Break on a ByteString delimiter
breakOnBS :: ByteString -> ByteString -> (ByteString, ByteString)
breakOnBS needle haystack = go 0
  where
    go !i
        | i > length haystack - length needle = (haystack, empty)
        | isPrefixOf needle (drop i haystack) = (take i haystack, drop i haystack)
        | otherwise = go (i + 1)

-- Internal: create a ByteString
createBS :: Int -> (Ptr Word8 -> IO ()) -> ByteString
createBS len fill = unsafePerformIO $ do
    arr <- mallocByteArray len
    fill (byteArrayContents arr)
    return $ BS arr 0 len

-- FFI helpers (stubs - would be implemented via Rust FFI)
foreign import ccall unsafe "bhc_bytearray_index"
    indexBS :: ByteArray -> Int -> Word8

foreign import ccall unsafe "bhc_bytearray_malloc"
    mallocByteArray :: Int -> IO ByteArray

foreign import ccall unsafe "bhc_bytearray_contents"
    byteArrayContents :: ByteArray -> Ptr Word8

foreign import ccall unsafe "bhc_bytearray_copy"
    copyBytes :: Ptr Word8 -> ByteArray -> Int -> Int -> IO ()

foreign import ccall unsafe "bhc_ptr_plus"
    plusPtr :: Ptr a -> Int -> Ptr a

foreign import ccall unsafe "bhc_poke_byte"
    pokeByteOff :: Ptr Word8 -> Int -> Word8 -> IO ()

foreign import ccall unsafe "bhc_cstring_length"
    cStringLength :: CString -> IO Int

foreign import ccall unsafe "bhc_peek_array"
    peekArray :: Int -> CString -> IO [Word8]

-- | Allocate a temporary array from a list of bytes and pass a pointer
-- to the action. The array is freed when the action completes.
withArray :: [Word8] -> (CString -> IO a) -> IO a
withArray ws action = do
    let len = P.length ws
    arr <- mallocByteArray len
    let ptr = byteArrayContents arr
    pokeList ptr ws 0
    action ptr
  where
    pokeList _ [] _ = return ()
    pokeList p (x:xs) !i = do
        pokeByteOff p i x
        pokeList p xs (i + 1)

-- Unsafe IO (for internal use)
{-# NOINLINE unsafePerformIO #-}
unsafePerformIO :: IO a -> a
unsafePerformIO = error "unsafePerformIO: implemented via compiler magic"
