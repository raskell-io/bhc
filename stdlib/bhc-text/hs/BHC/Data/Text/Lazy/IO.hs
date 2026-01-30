-- |
-- Module      : BHC.Data.Text.Lazy.IO
-- Description : Lazy Text I/O operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- I/O operations for lazy 'Text'. Delegates to strict 'Text.IO'
-- via 'fromStrict'/'toStrict' conversion.

module BHC.Data.Text.Lazy.IO (
    -- * File I/O
    readFile,
    writeFile,
    appendFile,

    -- * Handle I/O
    hGetContents,
    hGetLine,
    hPutStr,
    hPutStrLn,

    -- * Console I/O
    putStr,
    putStrLn,
    getLine,
    getContents,
    interact,
) where

import Prelude hiding (
    readFile, writeFile, appendFile,
    putStr, putStrLn, getLine, getContents, interact
    )
import System.IO (Handle)

import BHC.Data.Text.Lazy (Text, fromStrict, toStrict)
import qualified BHC.Data.Text.IO as TIO

-- ============================================================
-- File I/O
-- ============================================================

readFile :: FilePath -> IO Text
readFile path = fmap fromStrict (TIO.readFile path)

writeFile :: FilePath -> Text -> IO ()
writeFile path t = TIO.writeFile path (toStrict t)

appendFile :: FilePath -> Text -> IO ()
appendFile path t = TIO.appendFile path (toStrict t)

-- ============================================================
-- Handle I/O
-- ============================================================

hGetContents :: Handle -> IO Text
hGetContents h = fmap fromStrict (TIO.hGetContents h)

hGetLine :: Handle -> IO Text
hGetLine h = fmap fromStrict (TIO.hGetLine h)

hPutStr :: Handle -> Text -> IO ()
hPutStr h t = TIO.hPutStr h (toStrict t)

hPutStrLn :: Handle -> Text -> IO ()
hPutStrLn h t = TIO.hPutStrLn h (toStrict t)

-- ============================================================
-- Console I/O
-- ============================================================

putStr :: Text -> IO ()
putStr t = TIO.putStr (toStrict t)

putStrLn :: Text -> IO ()
putStrLn t = TIO.putStrLn (toStrict t)

getLine :: IO Text
getLine = fmap fromStrict TIO.getLine

getContents :: IO Text
getContents = fmap fromStrict TIO.getContents

interact :: (Text -> Text) -> IO ()
interact f = getContents >>= putStr . f
