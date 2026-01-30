-- |
-- Module      : BHC.Data.ByteString.Lazy
-- Description : Lazy ByteString (thin wrapper over strict ByteString)
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Lazy 'ByteString' in BHC is currently a thin newtype wrapper
-- over strict 'Data.ByteString.ByteString'. All operations
-- delegate to strict variants.

module BHC.Data.ByteString.Lazy (
    -- * Type
    ByteString,

    -- * Conversion
    fromStrict,
    toStrict,
    fromChunks,
    toChunks,

    -- * Basic interface
    empty,
    singleton,
    pack,
    unpack,
    null,
    length,
    cons,
    snoc,
    append,
    head,
    last,
    tail,
    init,
    uncons,
    unsnoc,

    -- * Transformations
    map,
    reverse,
    intersperse,
    intercalate,

    -- * Folds
    foldl,
    foldl',
    foldr,
    concat,
    concatMap,
    any,
    all,
    maximum,
    minimum,

    -- * Substrings
    take,
    drop,
    splitAt,
    takeWhile,
    dropWhile,
    span,
    break,
    isPrefixOf,
    isSuffixOf,

    -- * Searching
    elem,
    notElem,
    find,
    filter,
    partition,

    -- * Indexing
    index,
    elemIndex,
    findIndex,
    count,

    -- * Zipping
    zip,
    zipWith,
    unzip,

    -- * I/O
    getContents,
    putStr,
    putStrLn,
    readFile,
    writeFile,
    appendFile,
    hGetContents,
    hGet,
    hPut,
    hPutStr,
    hPutStrLn,
) where

import Prelude hiding (
    null, length, map, reverse, concat, concatMap,
    any, all, maximum, minimum, take, drop, splitAt,
    takeWhile, dropWhile, span, break, elem, notElem,
    filter, zip, zipWith, unzip, head, last, tail, init,
    foldl, foldr, getContents, putStr, putStrLn,
    readFile, writeFile, appendFile
    )
import qualified Prelude as P

import Data.Word (Word8)
import System.IO (Handle)
import qualified BHC.Data.ByteString as BS

-- | Lazy 'ByteString'. Currently a newtype over strict 'BS.ByteString'.
newtype ByteString = ByteString BS.ByteString
  deriving (Eq, Ord)

instance Show ByteString where
  showsPrec p (ByteString bs) = showsPrec p bs

instance Semigroup ByteString where
  ByteString a <> ByteString b = ByteString (BS.append a b)

instance Monoid ByteString where
  mempty = ByteString BS.empty
  mappend = (<>)

-- ============================================================
-- Conversion
-- ============================================================

fromStrict :: BS.ByteString -> ByteString
fromStrict = ByteString

toStrict :: ByteString -> BS.ByteString
toStrict (ByteString bs) = bs

fromChunks :: [BS.ByteString] -> ByteString
fromChunks = ByteString . BS.concat

toChunks :: ByteString -> [BS.ByteString]
toChunks (ByteString bs)
  | BS.null bs = []
  | otherwise  = [bs]

-- ============================================================
-- Basic interface
-- ============================================================

empty :: ByteString
empty = ByteString BS.empty

singleton :: Word8 -> ByteString
singleton = ByteString . BS.singleton

pack :: [Word8] -> ByteString
pack = ByteString . BS.pack

unpack :: ByteString -> [Word8]
unpack (ByteString bs) = BS.unpack bs

null :: ByteString -> Bool
null (ByteString bs) = BS.null bs

length :: ByteString -> Int
length (ByteString bs) = BS.length bs

cons :: Word8 -> ByteString -> ByteString
cons w (ByteString bs) = ByteString (BS.cons w bs)

snoc :: ByteString -> Word8 -> ByteString
snoc (ByteString bs) w = ByteString (BS.snoc bs w)

append :: ByteString -> ByteString -> ByteString
append (ByteString a) (ByteString b) = ByteString (BS.append a b)

head :: ByteString -> Word8
head (ByteString bs) = BS.head bs

last :: ByteString -> Word8
last (ByteString bs) = BS.last bs

tail :: ByteString -> ByteString
tail (ByteString bs) = ByteString (BS.tail bs)

init :: ByteString -> ByteString
init (ByteString bs) = ByteString (BS.init bs)

uncons :: ByteString -> Maybe (Word8, ByteString)
uncons (ByteString bs) = case BS.uncons bs of
  Nothing     -> Nothing
  Just (w, r) -> Just (w, ByteString r)

unsnoc :: ByteString -> Maybe (ByteString, Word8)
unsnoc (ByteString bs) = case BS.unsnoc bs of
  Nothing     -> Nothing
  Just (r, w) -> Just (ByteString r, w)

-- ============================================================
-- Transformations
-- ============================================================

map :: (Word8 -> Word8) -> ByteString -> ByteString
map f (ByteString bs) = ByteString (BS.map f bs)

reverse :: ByteString -> ByteString
reverse (ByteString bs) = ByteString (BS.reverse bs)

intersperse :: Word8 -> ByteString -> ByteString
intersperse w (ByteString bs) = ByteString (BS.intersperse w bs)

intercalate :: ByteString -> [ByteString] -> ByteString
intercalate (ByteString sep) bss = ByteString (BS.intercalate sep (P.map toStrict bss))

-- ============================================================
-- Folds
-- ============================================================

foldl :: (a -> Word8 -> a) -> a -> ByteString -> a
foldl f z (ByteString bs) = BS.foldl f z bs

foldl' :: (a -> Word8 -> a) -> a -> ByteString -> a
foldl' f z (ByteString bs) = BS.foldl' f z bs

foldr :: (Word8 -> a -> a) -> a -> ByteString -> a
foldr f z (ByteString bs) = BS.foldr f z bs

concat :: [ByteString] -> ByteString
concat = ByteString . BS.concat . P.map toStrict

concatMap :: (Word8 -> ByteString) -> ByteString -> ByteString
concatMap f (ByteString bs) = ByteString (BS.concatMap (toStrict . f) bs)

any :: (Word8 -> Bool) -> ByteString -> Bool
any p (ByteString bs) = BS.any p bs

all :: (Word8 -> Bool) -> ByteString -> Bool
all p (ByteString bs) = BS.all p bs

maximum :: ByteString -> Word8
maximum (ByteString bs) = BS.maximum bs

minimum :: ByteString -> Word8
minimum (ByteString bs) = BS.minimum bs

-- ============================================================
-- Substrings
-- ============================================================

take :: Int -> ByteString -> ByteString
take n (ByteString bs) = ByteString (BS.take n bs)

drop :: Int -> ByteString -> ByteString
drop n (ByteString bs) = ByteString (BS.drop n bs)

splitAt :: Int -> ByteString -> (ByteString, ByteString)
splitAt n (ByteString bs) = let (a, b) = BS.splitAt n bs in (ByteString a, ByteString b)

takeWhile :: (Word8 -> Bool) -> ByteString -> ByteString
takeWhile p (ByteString bs) = ByteString (BS.takeWhile p bs)

dropWhile :: (Word8 -> Bool) -> ByteString -> ByteString
dropWhile p (ByteString bs) = ByteString (BS.dropWhile p bs)

span :: (Word8 -> Bool) -> ByteString -> (ByteString, ByteString)
span p (ByteString bs) = let (a, b) = BS.span p bs in (ByteString a, ByteString b)

break :: (Word8 -> Bool) -> ByteString -> (ByteString, ByteString)
break p (ByteString bs) = let (a, b) = BS.break p bs in (ByteString a, ByteString b)

isPrefixOf :: ByteString -> ByteString -> Bool
isPrefixOf (ByteString a) (ByteString b) = BS.isPrefixOf a b

isSuffixOf :: ByteString -> ByteString -> Bool
isSuffixOf (ByteString a) (ByteString b) = BS.isSuffixOf a b

-- ============================================================
-- Searching
-- ============================================================

elem :: Word8 -> ByteString -> Bool
elem w (ByteString bs) = BS.elem w bs

notElem :: Word8 -> ByteString -> Bool
notElem w (ByteString bs) = BS.notElem w bs

find :: (Word8 -> Bool) -> ByteString -> Maybe Word8
find p (ByteString bs) = BS.find p bs

filter :: (Word8 -> Bool) -> ByteString -> ByteString
filter p (ByteString bs) = ByteString (BS.filter p bs)

partition :: (Word8 -> Bool) -> ByteString -> (ByteString, ByteString)
partition p (ByteString bs) = let (a, b) = BS.partition p bs in (ByteString a, ByteString b)

-- ============================================================
-- Indexing
-- ============================================================

index :: ByteString -> Int -> Word8
index (ByteString bs) i = BS.index bs i

elemIndex :: Word8 -> ByteString -> Maybe Int
elemIndex w (ByteString bs) = BS.elemIndex w bs

findIndex :: (Word8 -> Bool) -> ByteString -> Maybe Int
findIndex p (ByteString bs) = BS.findIndex p bs

count :: Word8 -> ByteString -> Int
count w (ByteString bs) = BS.count w bs

-- ============================================================
-- Zipping
-- ============================================================

zip :: ByteString -> ByteString -> [(Word8, Word8)]
zip (ByteString a) (ByteString b) = BS.zip a b

zipWith :: (Word8 -> Word8 -> a) -> ByteString -> ByteString -> [a]
zipWith f (ByteString a) (ByteString b) = BS.zipWith f a b

unzip :: [(Word8, Word8)] -> (ByteString, ByteString)
unzip pairs = let (a, b) = BS.unzip pairs in (ByteString a, ByteString b)

-- ============================================================
-- I/O
-- ============================================================

getContents :: IO ByteString
getContents = fmap ByteString BS.getContents

putStr :: ByteString -> IO ()
putStr (ByteString bs) = BS.putStr bs

putStrLn :: ByteString -> IO ()
putStrLn (ByteString bs) = BS.putStrLn bs

readFile :: FilePath -> IO ByteString
readFile path = fmap ByteString (BS.readFile path)

writeFile :: FilePath -> ByteString -> IO ()
writeFile path (ByteString bs) = BS.writeFile path bs

appendFile :: FilePath -> ByteString -> IO ()
appendFile path (ByteString bs) = BS.appendFile path bs

hGetContents :: Handle -> IO ByteString
hGetContents h = fmap ByteString (BS.hGetContents h)

hGet :: Handle -> Int -> IO ByteString
hGet h n = fmap ByteString (BS.hGet h n)

hPut :: Handle -> ByteString -> IO ()
hPut h (ByteString bs) = BS.hPut h bs

hPutStr :: Handle -> ByteString -> IO ()
hPutStr = hPut

hPutStrLn :: Handle -> ByteString -> IO ()
hPutStrLn h (ByteString bs) = BS.hPutStrLn h bs
