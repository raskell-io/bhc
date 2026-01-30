-- |
-- Module      : BHC.Data.ByteString.Char8
-- Description : ByteString operations treating bytes as Chars
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Provides operations on 'ByteString' where bytes are treated
-- as Latin-1 encoded characters. This is useful for ASCII/Latin-1
-- text processing.

module BHC.Data.ByteString.Char8 (
    -- * The ByteString type
    ByteString,

    -- * Conversion
    pack,
    unpack,
    w2c,
    c2w,

    -- * Basic interface
    empty,
    singleton,
    cons,
    snoc,
    append,
    head,
    last,
    tail,
    init,
    null,
    length,
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

    -- * Substrings
    take,
    drop,
    splitAt,
    takeWhile,
    dropWhile,
    span,
    break,

    -- * Breaking into lines and words
    lines,
    words,
    unlines,
    unwords,

    -- * Predicates
    isPrefixOf,
    isSuffixOf,
    isInfixOf,

    -- * Searching
    elem,
    notElem,
    find,
    filter,

    -- * Indexing
    index,
    elemIndex,
    findIndex,
    count,

    -- * Numeric parsing
    readInt,
    readInteger,

    -- * I/O
    getLine,
    putStr,
    putStrLn,
    readFile,
    writeFile,
    appendFile,
    hGetLine,
    hGetContents,
    hGet,
    hPut,
    hPutStr,
    hPutStrLn,
) where

import Prelude hiding (
    head, last, tail, init, null, length, map, reverse,
    foldl, foldr, concat, concatMap, any, all,
    take, drop, splitAt, takeWhile, dropWhile, span, break,
    elem, notElem, filter, lines, words, unlines, unwords,
    getLine, putStr, putStrLn, readFile, writeFile, appendFile
    )
import qualified Prelude as P

import Data.Char (ord, chr, isSpace, isDigit)
import Data.Word (Word8)
import System.IO (Handle)

import qualified BHC.Data.ByteString as BS
import BHC.Data.ByteString (ByteString)

-- ============================================================
-- Char/Word8 conversion
-- ============================================================

-- | Convert a 'Word8' to a 'Char' (Latin-1).
w2c :: Word8 -> Char
w2c = chr . fromIntegral

-- | Convert a 'Char' to a 'Word8' (truncated to 8 bits).
c2w :: Char -> Word8
c2w = fromIntegral . ord

-- ============================================================
-- Conversion
-- ============================================================

-- | Pack a 'String' into a 'ByteString' using Latin-1 encoding.
pack :: String -> ByteString
pack = BS.pack . P.map c2w

-- | Unpack a 'ByteString' to a 'String' using Latin-1 encoding.
unpack :: ByteString -> String
unpack = P.map w2c . BS.unpack

-- ============================================================
-- Basic interface (delegating to BS)
-- ============================================================

empty :: ByteString
empty = BS.empty

singleton :: Char -> ByteString
singleton = BS.singleton . c2w

cons :: Char -> ByteString -> ByteString
cons c = BS.cons (c2w c)

snoc :: ByteString -> Char -> ByteString
snoc bs c = BS.snoc bs (c2w c)

append :: ByteString -> ByteString -> ByteString
append = BS.append

head :: ByteString -> Char
head = w2c . BS.head

last :: ByteString -> Char
last = w2c . BS.last

tail :: ByteString -> ByteString
tail = BS.tail

init :: ByteString -> ByteString
init = BS.init

null :: ByteString -> Bool
null = BS.null

length :: ByteString -> Int
length = BS.length

uncons :: ByteString -> Maybe (Char, ByteString)
uncons bs = case BS.uncons bs of
  Nothing     -> Nothing
  Just (w, r) -> Just (w2c w, r)

unsnoc :: ByteString -> Maybe (ByteString, Char)
unsnoc bs = case BS.unsnoc bs of
  Nothing     -> Nothing
  Just (r, w) -> Just (r, w2c w)

-- ============================================================
-- Transformations
-- ============================================================

map :: (Char -> Char) -> ByteString -> ByteString
map f = BS.map (c2w . f . w2c)

reverse :: ByteString -> ByteString
reverse = BS.reverse

intersperse :: Char -> ByteString -> ByteString
intersperse c = BS.intersperse (c2w c)

intercalate :: ByteString -> [ByteString] -> ByteString
intercalate = BS.intercalate

-- ============================================================
-- Folds
-- ============================================================

foldl :: (a -> Char -> a) -> a -> ByteString -> a
foldl f z = BS.foldl (\acc w -> f acc (w2c w)) z

foldl' :: (a -> Char -> a) -> a -> ByteString -> a
foldl' f z = BS.foldl' (\acc w -> f acc (w2c w)) z

foldr :: (Char -> a -> a) -> a -> ByteString -> a
foldr f z = BS.foldr (\w acc -> f (w2c w) acc) z

concat :: [ByteString] -> ByteString
concat = BS.concat

concatMap :: (Char -> ByteString) -> ByteString -> ByteString
concatMap f = BS.concatMap (f . w2c)

any :: (Char -> Bool) -> ByteString -> Bool
any p = BS.any (p . w2c)

all :: (Char -> Bool) -> ByteString -> Bool
all p = BS.all (p . w2c)

-- ============================================================
-- Substrings
-- ============================================================

take :: Int -> ByteString -> ByteString
take = BS.take

drop :: Int -> ByteString -> ByteString
drop = BS.drop

splitAt :: Int -> ByteString -> (ByteString, ByteString)
splitAt = BS.splitAt

takeWhile :: (Char -> Bool) -> ByteString -> ByteString
takeWhile p = BS.takeWhile (p . w2c)

dropWhile :: (Char -> Bool) -> ByteString -> ByteString
dropWhile p = BS.dropWhile (p . w2c)

span :: (Char -> Bool) -> ByteString -> (ByteString, ByteString)
span p = BS.span (p . w2c)

break :: (Char -> Bool) -> ByteString -> (ByteString, ByteString)
break p = BS.break (p . w2c)

-- ============================================================
-- Lines and words
-- ============================================================

-- | Split on newline characters (byte 10).
lines :: ByteString -> [ByteString]
lines bs
  | null bs   = []
  | otherwise = case BS.elemIndex 10 bs of
      Nothing -> [bs]
      Just i  -> BS.take i bs : lines (BS.drop (i + 1) bs)

-- | Split on whitespace (space=32, tab=9, newline=10, cr=13).
words :: ByteString -> [ByteString]
words = P.filter (not . null) . BS.splitWith isSpaceW8
  where
    isSpaceW8 w = w == 32 || w == 9 || w == 10 || w == 13

-- | Join lines with newline separators.
unlines :: [ByteString] -> ByteString
unlines = concat . P.map (\l -> append l (singleton '\n'))

-- | Join words with space separators.
unwords :: [ByteString] -> ByteString
unwords [] = empty
unwords ws = BS.intercalate (singleton ' ') ws

-- ============================================================
-- Predicates
-- ============================================================

isPrefixOf :: ByteString -> ByteString -> Bool
isPrefixOf = BS.isPrefixOf

isSuffixOf :: ByteString -> ByteString -> Bool
isSuffixOf = BS.isSuffixOf

isInfixOf :: ByteString -> ByteString -> Bool
isInfixOf = BS.isInfixOf

-- ============================================================
-- Searching
-- ============================================================

elem :: Char -> ByteString -> Bool
elem c = BS.elem (c2w c)

notElem :: Char -> ByteString -> Bool
notElem c = BS.notElem (c2w c)

find :: (Char -> Bool) -> ByteString -> Maybe Char
find p bs = fmap w2c (BS.find (p . w2c) bs)

filter :: (Char -> Bool) -> ByteString -> ByteString
filter p = BS.filter (p . w2c)

-- ============================================================
-- Indexing
-- ============================================================

index :: ByteString -> Int -> Char
index bs i = w2c (BS.index bs i)

elemIndex :: Char -> ByteString -> Maybe Int
elemIndex c = BS.elemIndex (c2w c)

findIndex :: (Char -> Bool) -> ByteString -> Maybe Int
findIndex p = BS.findIndex (p . w2c)

count :: Char -> ByteString -> Int
count c = BS.count (c2w c)

-- ============================================================
-- Numeric parsing
-- ============================================================

-- | Try to parse an 'Int' from the beginning of a 'ByteString'.
-- Returns the parsed integer and the remainder, or 'Nothing'.
readInt :: ByteString -> Maybe (Int, ByteString)
readInt bs = case uncons bs of
  Nothing -> Nothing
  Just ('-', rest) -> case readUnsigned rest of
    Nothing       -> Nothing
    Just (n, rem) -> Just (negate n, rem)
  Just (c, _)
    | isDigit c -> readUnsigned bs
    | otherwise -> Nothing
  where
    readUnsigned s = case BS.span (\w -> w >= 48 && w <= 57) s of
      (digits, rest)
        | null digits -> Nothing
        | otherwise   -> Just (parseDigits digits, rest)
    parseDigits = foldl' (\acc c -> acc * 10 + (ord c - 48)) 0

-- | Try to parse an 'Integer' from the beginning of a 'ByteString'.
readInteger :: ByteString -> Maybe (Integer, ByteString)
readInteger bs = case readInt bs of
  Nothing       -> Nothing
  Just (n, rem) -> Just (fromIntegral n, rem)

-- ============================================================
-- I/O
-- ============================================================

getLine :: IO ByteString
getLine = BS.getLine

putStr :: ByteString -> IO ()
putStr = BS.putStr

putStrLn :: ByteString -> IO ()
putStrLn = BS.putStrLn

readFile :: FilePath -> IO ByteString
readFile = BS.readFile

writeFile :: FilePath -> ByteString -> IO ()
writeFile = BS.writeFile

appendFile :: FilePath -> ByteString -> IO ()
appendFile = BS.appendFile

hGetLine :: Handle -> IO ByteString
hGetLine = BS.hGetLine

hGetContents :: Handle -> IO ByteString
hGetContents = BS.hGetContents

hGet :: Handle -> Int -> IO ByteString
hGet = BS.hGet

hPut :: Handle -> ByteString -> IO ()
hPut = BS.hPut

hPutStr :: Handle -> ByteString -> IO ()
hPutStr = BS.hPutStr

hPutStrLn :: Handle -> ByteString -> IO ()
hPutStrLn = BS.hPutStrLn
