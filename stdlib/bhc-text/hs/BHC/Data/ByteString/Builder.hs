{-# LANGUAGE BangPatterns #-}

-- |
-- Module      : BHC.Data.ByteString.Builder
-- Description : Efficient construction of ByteStrings
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- A Builder is an efficient way to construct strict 'ByteString's.
-- Builders support O(1) append, which makes them suitable for
-- building ByteStrings incrementally in a loop.
--
-- = Usage
--
-- @
-- import qualified BHC.Data.ByteString.Builder as B
-- import qualified BHC.Data.ByteString as BS
--
-- -- Build a ByteString efficiently
-- let builder = B.word8 72 <> B.word8 105 <> B.byteString (BS.pack [33])
-- BS.length (B.toLazyByteString builder)  -- 3
-- @
--
-- = Performance
--
-- Builders use an internal buffer that grows as needed. When the buffer
-- is full, it's flushed to a list of chunks. The final 'toLazyByteString'
-- concatenates all chunks.

module BHC.Data.ByteString.Builder
    ( -- * Builder type
      Builder

      -- * Running builders
    , toLazyByteString
    , toStrictByteString
    , hPutBuilder

      -- * Constructing builders
    , empty
    , singleton
    , word8
    , byteString
    , lazyByteString
    , shortByteString

      -- * Numeric encoding
      -- ** Big-endian
    , int16BE
    , int32BE
    , int64BE
    , word16BE
    , word32BE
    , word64BE
    , floatBE
    , doubleBE

      -- ** Little-endian
    , int16LE
    , int32LE
    , int64LE
    , word16LE
    , word32LE
    , word64LE
    , floatLE
    , doubleLE

      -- ** Host-endian
    , int16Host
    , int32Host
    , int64Host
    , word16Host
    , word32Host
    , word64Host
    , floatHost
    , doubleHost

      -- ** Decimal encoding
    , intDec
    , int8Dec
    , int16Dec
    , int32Dec
    , int64Dec
    , integerDec
    , wordDec
    , word8Dec
    , word16Dec
    , word32Dec
    , word64Dec

      -- ** Hexadecimal encoding
    , wordHex
    , word8Hex
    , word16Hex
    , word32Hex
    , word64Hex
    , word8HexFixed
    , word16HexFixed
    , word32HexFixed
    , word64HexFixed

      -- * Character encoding
    , char7
    , char8
    , string7
    , string8
    , charUtf8
    , stringUtf8
    ) where

import Prelude hiding (length)

import Data.Word (Word8, Word16, Word32, Word64)
import Data.Int (Int8, Int16, Int32, Int64)
import Data.Bits ((.&.), (.|.), shiftR, shiftL)
import Data.Char (ord)
import System.IO (Handle)

import qualified BHC.Data.ByteString as BS
import BHC.Data.ByteString (ByteString)

-- ============================================================
-- Builder type
-- ============================================================

-- | A builder for efficient 'ByteString' construction.
-- Internally uses a difference list of byte chunks.
newtype Builder = Builder { unBuilder :: [ByteString] -> [ByteString] }

instance Semigroup Builder where
    Builder f <> Builder g = Builder (f . g)

instance Monoid Builder where
    mempty = Builder id
    mappend = (<>)

instance Show Builder where
    show b = "Builder " ++ show (toStrictByteString b)

-- ============================================================
-- Running builders
-- ============================================================

-- | Convert a 'Builder' to a lazy 'ByteString'.
-- In BHC, this returns a strict ByteString (lazy not yet implemented).
toLazyByteString :: Builder -> ByteString
toLazyByteString = toStrictByteString

-- | Convert a 'Builder' to a strict 'ByteString'.
toStrictByteString :: Builder -> ByteString
toStrictByteString (Builder f) = BS.concat (f [])

-- | Write a 'Builder' to a 'Handle'.
hPutBuilder :: Handle -> Builder -> IO ()
hPutBuilder h = BS.hPut h . toStrictByteString

-- ============================================================
-- Constructing builders
-- ============================================================

-- | The empty 'Builder'.
empty :: Builder
empty = Builder id

-- | A 'Builder' producing a single byte.
singleton :: Word8 -> Builder
singleton w = Builder (BS.singleton w :)

-- | Alias for 'singleton'.
word8 :: Word8 -> Builder
word8 = singleton

-- | Create a 'Builder' from a strict 'ByteString'.
byteString :: ByteString -> Builder
byteString bs
    | BS.null bs = empty
    | otherwise  = Builder (bs :)

-- | Create a 'Builder' from a lazy 'ByteString'.
-- In BHC, this is the same as 'byteString'.
lazyByteString :: ByteString -> Builder
lazyByteString = byteString

-- | Create a 'Builder' from a short 'ByteString'.
-- In BHC, this is the same as 'byteString'.
shortByteString :: ByteString -> Builder
shortByteString = byteString

-- ============================================================
-- Big-endian encoding
-- ============================================================

-- | Encode a 16-bit signed integer in big-endian format.
int16BE :: Int16 -> Builder
int16BE = word16BE . fromIntegral

-- | Encode a 32-bit signed integer in big-endian format.
int32BE :: Int32 -> Builder
int32BE = word32BE . fromIntegral

-- | Encode a 64-bit signed integer in big-endian format.
int64BE :: Int64 -> Builder
int64BE = word64BE . fromIntegral

-- | Encode a 16-bit unsigned integer in big-endian format.
word16BE :: Word16 -> Builder
word16BE w = word8 (fromIntegral (w `shiftR` 8)) <>
             word8 (fromIntegral w)

-- | Encode a 32-bit unsigned integer in big-endian format.
word32BE :: Word32 -> Builder
word32BE w = word8 (fromIntegral (w `shiftR` 24)) <>
             word8 (fromIntegral (w `shiftR` 16)) <>
             word8 (fromIntegral (w `shiftR` 8)) <>
             word8 (fromIntegral w)

-- | Encode a 64-bit unsigned integer in big-endian format.
word64BE :: Word64 -> Builder
word64BE w = word32BE (fromIntegral (w `shiftR` 32)) <>
             word32BE (fromIntegral w)

-- | Encode a 'Float' in big-endian IEEE 754 format.
floatBE :: Float -> Builder
floatBE = word32BE . floatToWord32

-- | Encode a 'Double' in big-endian IEEE 754 format.
doubleBE :: Double -> Builder
doubleBE = word64BE . doubleToWord64

-- ============================================================
-- Little-endian encoding
-- ============================================================

-- | Encode a 16-bit signed integer in little-endian format.
int16LE :: Int16 -> Builder
int16LE = word16LE . fromIntegral

-- | Encode a 32-bit signed integer in little-endian format.
int32LE :: Int32 -> Builder
int32LE = word32LE . fromIntegral

-- | Encode a 64-bit signed integer in little-endian format.
int64LE :: Int64 -> Builder
int64LE = word64LE . fromIntegral

-- | Encode a 16-bit unsigned integer in little-endian format.
word16LE :: Word16 -> Builder
word16LE w = word8 (fromIntegral w) <>
             word8 (fromIntegral (w `shiftR` 8))

-- | Encode a 32-bit unsigned integer in little-endian format.
word32LE :: Word32 -> Builder
word32LE w = word8 (fromIntegral w) <>
             word8 (fromIntegral (w `shiftR` 8)) <>
             word8 (fromIntegral (w `shiftR` 16)) <>
             word8 (fromIntegral (w `shiftR` 24))

-- | Encode a 64-bit unsigned integer in little-endian format.
word64LE :: Word64 -> Builder
word64LE w = word32LE (fromIntegral w) <>
             word32LE (fromIntegral (w `shiftR` 32))

-- | Encode a 'Float' in little-endian IEEE 754 format.
floatLE :: Float -> Builder
floatLE = word32LE . floatToWord32

-- | Encode a 'Double' in little-endian IEEE 754 format.
doubleLE :: Double -> Builder
doubleLE = word64LE . doubleToWord64

-- ============================================================
-- Host-endian encoding
-- ============================================================

-- | Encode a 16-bit signed integer in host-endian format.
int16Host :: Int16 -> Builder
int16Host = int16LE  -- Assuming little-endian host

-- | Encode a 32-bit signed integer in host-endian format.
int32Host :: Int32 -> Builder
int32Host = int32LE

-- | Encode a 64-bit signed integer in host-endian format.
int64Host :: Int64 -> Builder
int64Host = int64LE

-- | Encode a 16-bit unsigned integer in host-endian format.
word16Host :: Word16 -> Builder
word16Host = word16LE

-- | Encode a 32-bit unsigned integer in host-endian format.
word32Host :: Word32 -> Builder
word32Host = word32LE

-- | Encode a 64-bit unsigned integer in host-endian format.
word64Host :: Word64 -> Builder
word64Host = word64LE

-- | Encode a 'Float' in host-endian IEEE 754 format.
floatHost :: Float -> Builder
floatHost = floatLE

-- | Encode a 'Double' in host-endian IEEE 754 format.
doubleHost :: Double -> Builder
doubleHost = doubleLE

-- ============================================================
-- Decimal encoding
-- ============================================================

-- | Encode an 'Int' as decimal ASCII.
intDec :: Int -> Builder
intDec n
    | n < 0     = word8 45 <> wordDec (fromIntegral (negate n))  -- '-'
    | otherwise = wordDec (fromIntegral n)

-- | Encode an 'Int8' as decimal ASCII.
int8Dec :: Int8 -> Builder
int8Dec = intDec . fromIntegral

-- | Encode an 'Int16' as decimal ASCII.
int16Dec :: Int16 -> Builder
int16Dec = intDec . fromIntegral

-- | Encode an 'Int32' as decimal ASCII.
int32Dec :: Int32 -> Builder
int32Dec = intDec . fromIntegral

-- | Encode an 'Int64' as decimal ASCII.
int64Dec :: Int64 -> Builder
int64Dec = intDec . fromIntegral

-- | Encode an 'Integer' as decimal ASCII.
integerDec :: Integer -> Builder
integerDec n
    | n < 0     = word8 45 <> go (negate n)
    | n == 0    = word8 48
    | otherwise = go n
  where
    go 0 = empty
    go i = go (i `div` 10) <> word8 (fromIntegral (48 + i `mod` 10))

-- | Encode a 'Word' as decimal ASCII.
wordDec :: Word -> Builder
wordDec 0 = word8 48  -- '0'
wordDec n = go n empty
  where
    go 0 !acc = acc
    go i !acc = go (i `div` 10) (word8 (fromIntegral (48 + i `mod` 10)) <> acc)

-- | Encode a 'Word8' as decimal ASCII.
word8Dec :: Word8 -> Builder
word8Dec = wordDec . fromIntegral

-- | Encode a 'Word16' as decimal ASCII.
word16Dec :: Word16 -> Builder
word16Dec = wordDec . fromIntegral

-- | Encode a 'Word32' as decimal ASCII.
word32Dec :: Word32 -> Builder
word32Dec = wordDec . fromIntegral

-- | Encode a 'Word64' as decimal ASCII.
word64Dec :: Word64 -> Builder
word64Dec = wordDec . fromIntegral

-- ============================================================
-- Hexadecimal encoding
-- ============================================================

-- | Encode a 'Word' as lowercase hexadecimal ASCII.
wordHex :: Word -> Builder
wordHex 0 = word8 48  -- '0'
wordHex n = go n empty
  where
    go 0 !acc = acc
    go i !acc = go (i `div` 16) (hexDigit (fromIntegral (i `mod` 16)) <> acc)

-- | Encode a 'Word8' as lowercase hexadecimal ASCII.
word8Hex :: Word8 -> Builder
word8Hex = wordHex . fromIntegral

-- | Encode a 'Word16' as lowercase hexadecimal ASCII.
word16Hex :: Word16 -> Builder
word16Hex = wordHex . fromIntegral

-- | Encode a 'Word32' as lowercase hexadecimal ASCII.
word32Hex :: Word32 -> Builder
word32Hex = wordHex . fromIntegral

-- | Encode a 'Word64' as lowercase hexadecimal ASCII.
word64Hex :: Word64 -> Builder
word64Hex = wordHex . fromIntegral

-- | Encode a 'Word8' as exactly 2 hex digits.
word8HexFixed :: Word8 -> Builder
word8HexFixed w = hexDigit (w `shiftR` 4) <> hexDigit (w .&. 0x0f)

-- | Encode a 'Word16' as exactly 4 hex digits.
word16HexFixed :: Word16 -> Builder
word16HexFixed w = word8HexFixed (fromIntegral (w `shiftR` 8)) <>
                   word8HexFixed (fromIntegral w)

-- | Encode a 'Word32' as exactly 8 hex digits.
word32HexFixed :: Word32 -> Builder
word32HexFixed w = word16HexFixed (fromIntegral (w `shiftR` 16)) <>
                   word16HexFixed (fromIntegral w)

-- | Encode a 'Word64' as exactly 16 hex digits.
word64HexFixed :: Word64 -> Builder
word64HexFixed w = word32HexFixed (fromIntegral (w `shiftR` 32)) <>
                   word32HexFixed (fromIntegral w)

-- | Encode a hex digit (0-15) as lowercase ASCII.
hexDigit :: Word8 -> Builder
hexDigit n
    | n < 10    = word8 (48 + n)      -- '0' + n
    | otherwise = word8 (87 + n)      -- 'a' - 10 + n

-- ============================================================
-- Character encoding
-- ============================================================

-- | Encode an ASCII character (truncated to 7 bits).
char7 :: Char -> Builder
char7 = word8 . fromIntegral . (0x7f .&.) . ord

-- | Encode a Latin-1 character (truncated to 8 bits).
char8 :: Char -> Builder
char8 = word8 . fromIntegral . (0xff .&.) . ord

-- | Encode an ASCII string.
string7 :: String -> Builder
string7 = foldMap char7

-- | Encode a Latin-1 string.
string8 :: String -> Builder
string8 = foldMap char8

-- | Encode a character as UTF-8.
charUtf8 :: Char -> Builder
charUtf8 c
    | cp < 0x80    = word8 (fromIntegral cp)
    | cp < 0x800   = word8 (fromIntegral (0xc0 .|. (cp `shiftR` 6))) <>
                     word8 (fromIntegral (0x80 .|. (cp .&. 0x3f)))
    | cp < 0x10000 = word8 (fromIntegral (0xe0 .|. (cp `shiftR` 12))) <>
                     word8 (fromIntegral (0x80 .|. ((cp `shiftR` 6) .&. 0x3f))) <>
                     word8 (fromIntegral (0x80 .|. (cp .&. 0x3f)))
    | otherwise    = word8 (fromIntegral (0xf0 .|. (cp `shiftR` 18))) <>
                     word8 (fromIntegral (0x80 .|. ((cp `shiftR` 12) .&. 0x3f))) <>
                     word8 (fromIntegral (0x80 .|. ((cp `shiftR` 6) .&. 0x3f))) <>
                     word8 (fromIntegral (0x80 .|. (cp .&. 0x3f)))
  where
    cp = ord c

-- | Encode a string as UTF-8.
stringUtf8 :: String -> Builder
stringUtf8 = foldMap charUtf8

-- ============================================================
-- Internal helpers
-- ============================================================

-- | Convert a 'Float' to its IEEE 754 bit representation.
foreign import ccall unsafe "bhc_float_to_word32"
    floatToWord32 :: Float -> Word32

-- | Convert a 'Double' to its IEEE 754 bit representation.
foreign import ccall unsafe "bhc_double_to_word64"
    doubleToWord64 :: Double -> Word64
