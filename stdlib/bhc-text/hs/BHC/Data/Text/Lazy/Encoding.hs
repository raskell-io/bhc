-- |
-- Module      : BHC.Data.Text.Lazy.Encoding
-- Description : Lazy Text encoding/decoding
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Encoding and decoding between lazy 'Text' and lazy 'ByteString'.
-- Delegates to strict encoding via 'fromStrict'/'toStrict'.

module BHC.Data.Text.Lazy.Encoding (
    -- * Decoding
    decodeUtf8,
    decodeUtf8',
    decodeUtf8With,
    decodeLatin1,
    decodeASCII,

    -- * Encoding
    encodeUtf8,
    encodeUtf16LE,
    encodeUtf16BE,
    encodeUtf32LE,
    encodeUtf32BE,
) where

import qualified BHC.Data.Text.Lazy as TL
import qualified BHC.Data.Text.Encoding as TE
import qualified BHC.Data.ByteString.Lazy as BL
import qualified BHC.Data.ByteString as BS
import qualified BHC.Data.Text as T

-- ============================================================
-- Decoding
-- ============================================================

-- | Decode a lazy 'BL.ByteString' to lazy 'TL.Text' as UTF-8.
decodeUtf8 :: BL.ByteString -> TL.Text
decodeUtf8 = TL.fromStrict . TE.decodeUtf8 . BL.toStrict

-- | Decode UTF-8, returning an error on invalid input.
decodeUtf8' :: BL.ByteString -> Either String TL.Text
decodeUtf8' bs = case TE.decodeUtf8' (BL.toStrict bs) of
    Left err -> Left err
    Right t  -> Right (TL.fromStrict t)

-- | Decode UTF-8 with a custom error handler.
decodeUtf8With :: (String -> Maybe Char) -> BL.ByteString -> TL.Text
decodeUtf8With onErr = TL.fromStrict . TE.decodeUtf8With onErr . BL.toStrict

-- | Decode a lazy 'BL.ByteString' as Latin-1 (ISO 8859-1).
decodeLatin1 :: BL.ByteString -> TL.Text
decodeLatin1 = TL.fromStrict . TE.decodeLatin1 . BL.toStrict

-- | Decode a lazy 'BL.ByteString' as ASCII.
decodeASCII :: BL.ByteString -> TL.Text
decodeASCII = TL.fromStrict . TE.decodeASCII . BL.toStrict

-- ============================================================
-- Encoding
-- ============================================================

-- | Encode lazy 'TL.Text' as UTF-8.
encodeUtf8 :: TL.Text -> BL.ByteString
encodeUtf8 = BL.fromStrict . TE.encodeUtf8 . TL.toStrict

-- | Encode lazy 'TL.Text' as UTF-16 little-endian.
encodeUtf16LE :: TL.Text -> BL.ByteString
encodeUtf16LE = BL.fromStrict . TE.encodeUtf16LE . TL.toStrict

-- | Encode lazy 'TL.Text' as UTF-16 big-endian.
encodeUtf16BE :: TL.Text -> BL.ByteString
encodeUtf16BE = BL.fromStrict . TE.encodeUtf16BE . TL.toStrict

-- | Encode lazy 'TL.Text' as UTF-32 little-endian.
encodeUtf32LE :: TL.Text -> BL.ByteString
encodeUtf32LE = BL.fromStrict . TE.encodeUtf32LE . TL.toStrict

-- | Encode lazy 'TL.Text' as UTF-32 big-endian.
encodeUtf32BE :: TL.Text -> BL.ByteString
encodeUtf32BE = BL.fromStrict . TE.encodeUtf32BE . TL.toStrict
