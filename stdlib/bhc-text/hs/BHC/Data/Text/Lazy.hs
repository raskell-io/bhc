-- |
-- Module      : BHC.Data.Text.Lazy
-- Description : Lazy Text (thin wrapper over strict Text)
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Lazy 'Text' in BHC is currently a thin newtype wrapper over
-- strict 'Data.Text.Text'. All operations delegate to strict variants.

module BHC.Data.Text.Lazy (
    -- * Type
    Text,

    -- * Conversion
    fromStrict,
    toStrict,
    fromChunks,
    toChunks,

    -- * Basic interface
    pack,
    unpack,
    empty,
    singleton,
    null,
    length,
    append,
    cons,
    snoc,
    head,
    last,
    tail,
    init,
    uncons,

    -- * Transformations
    map,
    intercalate,
    intersperse,
    reverse,

    -- * Folds
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
    isPrefixOf,
    isSuffixOf,
    isInfixOf,
    stripPrefix,
    stripSuffix,

    -- * Searching
    filter,
    find,

    -- * Encoding
    encodeUtf8,
    decodeUtf8,
) where

import Prelude hiding (
    null, length, map, reverse, concat, concatMap,
    any, all, take, drop, splitAt, takeWhile, dropWhile,
    span, break, filter, head, last, tail, init
    )
import qualified Prelude as P

import qualified BHC.Data.Text as T
import qualified BHC.Data.ByteString as BS

import Data.String (IsString(..))

-- | Lazy 'Text'. Currently a newtype over strict 'T.Text'.
newtype Text = Text T.Text
  deriving (Eq, Ord)

instance Show Text where
  showsPrec p (Text t) = showsPrec p t

instance Semigroup Text where
  Text a <> Text b = Text (T.append a b)

instance Monoid Text where
  mempty = Text T.empty
  mappend = (<>)

instance IsString Text where
  fromString = pack

-- ============================================================
-- Conversion
-- ============================================================

fromStrict :: T.Text -> Text
fromStrict = Text

toStrict :: Text -> T.Text
toStrict (Text t) = t

fromChunks :: [T.Text] -> Text
fromChunks = Text . T.concat

toChunks :: Text -> [T.Text]
toChunks (Text t)
  | T.null t  = []
  | otherwise = [t]

-- ============================================================
-- Basic interface
-- ============================================================

pack :: String -> Text
pack = Text . T.pack

unpack :: Text -> String
unpack (Text t) = T.unpack t

empty :: Text
empty = Text T.empty

singleton :: Char -> Text
singleton = Text . T.singleton

null :: Text -> Bool
null (Text t) = T.null t

length :: Text -> Int
length (Text t) = T.length t

append :: Text -> Text -> Text
append (Text a) (Text b) = Text (T.append a b)

cons :: Char -> Text -> Text
cons c (Text t) = Text (T.cons c t)

snoc :: Text -> Char -> Text
snoc (Text t) c = Text (T.snoc t c)

head :: Text -> Char
head (Text t) = T.head t

last :: Text -> Char
last (Text t) = T.last t

tail :: Text -> Text
tail (Text t) = Text (T.tail t)

init :: Text -> Text
init (Text t) = Text (T.init t)

uncons :: Text -> Maybe (Char, Text)
uncons (Text t) = case T.uncons t of
  Nothing     -> Nothing
  Just (c, r) -> Just (c, Text r)

-- ============================================================
-- Transformations
-- ============================================================

map :: (Char -> Char) -> Text -> Text
map f (Text t) = Text (T.map f t)

intercalate :: Text -> [Text] -> Text
intercalate (Text sep) ts = Text (T.intercalate sep (P.map toStrict ts))

intersperse :: Char -> Text -> Text
intersperse c (Text t) = Text (T.intersperse c t)

reverse :: Text -> Text
reverse (Text t) = Text (T.reverse t)

-- ============================================================
-- Folds
-- ============================================================

foldl' :: (a -> Char -> a) -> a -> Text -> a
foldl' f z (Text t) = T.foldl' f z t

foldr :: (Char -> a -> a) -> a -> Text -> a
foldr f z (Text t) = T.foldr f z t

concat :: [Text] -> Text
concat = Text . T.concat . P.map toStrict

concatMap :: (Char -> Text) -> Text -> Text
concatMap f (Text t) = Text (T.concatMap (toStrict . f) t)

any :: (Char -> Bool) -> Text -> Bool
any p (Text t) = T.any p t

all :: (Char -> Bool) -> Text -> Bool
all p (Text t) = T.all p t

-- ============================================================
-- Substrings
-- ============================================================

take :: Int -> Text -> Text
take n (Text t) = Text (T.take n t)

drop :: Int -> Text -> Text
drop n (Text t) = Text (T.drop n t)

splitAt :: Int -> Text -> (Text, Text)
splitAt n (Text t) = let (a, b) = T.splitAt n t in (Text a, Text b)

takeWhile :: (Char -> Bool) -> Text -> Text
takeWhile p (Text t) = Text (T.takeWhile p t)

dropWhile :: (Char -> Bool) -> Text -> Text
dropWhile p (Text t) = Text (T.dropWhile p t)

span :: (Char -> Bool) -> Text -> (Text, Text)
span p (Text t) = let (a, b) = T.span p t in (Text a, Text b)

break :: (Char -> Bool) -> Text -> (Text, Text)
break p (Text t) = let (a, b) = T.break p t in (Text a, Text b)

isPrefixOf :: Text -> Text -> Bool
isPrefixOf (Text a) (Text b) = T.isPrefixOf a b

isSuffixOf :: Text -> Text -> Bool
isSuffixOf (Text a) (Text b) = T.isSuffixOf a b

isInfixOf :: Text -> Text -> Bool
isInfixOf (Text a) (Text b) = T.isInfixOf a b

stripPrefix :: Text -> Text -> Maybe Text
stripPrefix (Text p) (Text t) = fmap Text (T.stripPrefix p t)

stripSuffix :: Text -> Text -> Maybe Text
stripSuffix (Text s) (Text t) = fmap Text (T.stripSuffix s t)

-- ============================================================
-- Searching
-- ============================================================

filter :: (Char -> Bool) -> Text -> Text
filter p (Text t) = Text (T.filter p t)

find :: (Char -> Bool) -> Text -> Maybe Char
find p (Text t) = T.find p t

-- ============================================================
-- Encoding
-- ============================================================

encodeUtf8 :: Text -> BS.ByteString
encodeUtf8 (Text t) = T.encodeUtf8 t
  where
    -- Delegate to BHC.Data.Text.Encoding if available,
    -- otherwise pack as ASCII for now
    encodeUtf8Simple :: T.Text -> BS.ByteString
    encodeUtf8Simple txt = BS.pack (P.map (fromIntegral . fromEnum) (T.unpack txt))

decodeUtf8 :: BS.ByteString -> Text
decodeUtf8 bs = Text (T.pack (P.map (toEnum . fromIntegral) (BS.unpack bs)))
