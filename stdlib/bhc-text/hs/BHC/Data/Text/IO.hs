-- |
-- Module      : BHC.Data.Text.IO
-- Description : Text I/O operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- I/O operations for strict 'Text' values. These functions
-- delegate to 'System.IO' string operations via 'pack'/'unpack'.

module BHC.Data.Text.IO (
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
import qualified System.IO as IO
import System.IO (Handle)

import BHC.Data.Text (Text)
import qualified BHC.Data.Text as T

-- ============================================================
-- File I/O
-- ============================================================

-- | Read the entire contents of a file as 'Text'.
readFile :: FilePath -> IO Text
readFile path = do
    s <- IO.readFile path
    return (T.pack s)

-- | Write 'Text' to a file, replacing its contents.
writeFile :: FilePath -> Text -> IO ()
writeFile path t = IO.writeFile path (T.unpack t)

-- | Append 'Text' to a file.
appendFile :: FilePath -> Text -> IO ()
appendFile path t = IO.appendFile path (T.unpack t)

-- ============================================================
-- Handle I/O
-- ============================================================

-- | Read the remaining contents of a 'Handle' as 'Text'.
hGetContents :: Handle -> IO Text
hGetContents h = do
    s <- IO.hGetContents h
    return (T.pack s)

-- | Read a single line from a 'Handle' as 'Text'.
hGetLine :: Handle -> IO Text
hGetLine h = do
    s <- IO.hGetLine h
    return (T.pack s)

-- | Write 'Text' to a 'Handle'.
hPutStr :: Handle -> Text -> IO ()
hPutStr h t = IO.hPutStr h (T.unpack t)

-- | Write 'Text' to a 'Handle', followed by a newline.
hPutStrLn :: Handle -> Text -> IO ()
hPutStrLn h t = IO.hPutStrLn h (T.unpack t)

-- ============================================================
-- Console I/O
-- ============================================================

-- | Write 'Text' to standard output.
putStr :: Text -> IO ()
putStr = hPutStr IO.stdout

-- | Write 'Text' to standard output, followed by a newline.
putStrLn :: Text -> IO ()
putStrLn = hPutStrLn IO.stdout

-- | Read a line from standard input as 'Text'.
getLine :: IO Text
getLine = hGetLine IO.stdin

-- | Read all input from standard input as 'Text'.
getContents :: IO Text
getContents = hGetContents IO.stdin

-- | Process standard input with a function and write to stdout.
interact :: (Text -> Text) -> IO ()
interact f = getContents >>= putStr . f
