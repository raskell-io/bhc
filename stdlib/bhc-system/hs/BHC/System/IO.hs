-- |
-- Module      : BHC.System.IO
-- Description : Input/Output operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.System.IO (
    -- * The IO monad
    IO,
    
    -- * Standard handles
    Handle,
    stdin, stdout, stderr,
    
    -- * Opening and closing
    openFile,
    hClose,
    withFile,
    
    -- * IO modes
    IOMode(..),
    
    -- * Reading
    hGetChar, hGetLine, hGetContents,
    hLookAhead, hReady,
    
    -- * Writing
    hPutChar, hPutStr, hPutStrLn, hPrint,
    hFlush,
    
    -- * Seeking
    hSeek, hTell,
    SeekMode(..),
    
    -- * Queries
    hIsEOF, hIsOpen, hIsClosed,
    hIsReadable, hIsWritable, hIsSeekable,
    
    -- * Buffering
    BufferMode(..),
    hSetBuffering, hGetBuffering,
    
    -- * File operations
    readFile, writeFile, appendFile,
    
    -- * Console I/O
    putChar, putStr, putStrLn, print,
    getChar, getLine, getContents,
    interact,
    
    -- * Errors
    IOError, ioError, userError,
    catch, try,
) where

import BHC.Prelude hiding (
    putChar, putStr, putStrLn, print,
    getChar, getLine, getContents, interact,
    readFile, writeFile, appendFile
    )

-- | An abstract handle to an I/O device (file, socket, terminal, etc.).
--
-- Handles are used for all input and output operations in BHC.
-- Standard handles 'stdin', 'stdout', and 'stderr' are always available.
data Handle

-- | The mode in which a file is opened.
--
-- * 'ReadMode' - Open for reading only
-- * 'WriteMode' - Open for writing only (truncates existing file)
-- * 'AppendMode' - Open for writing at the end of file
-- * 'ReadWriteMode' - Open for both reading and writing
data IOMode
    = ReadMode
    | WriteMode
    | AppendMode
    | ReadWriteMode
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | The mode for seeking within a file.
--
-- * 'AbsoluteSeek' - Seek to an absolute position from the start
-- * 'RelativeSeek' - Seek relative to the current position
-- * 'SeekFromEnd' - Seek relative to the end of the file
data SeekMode
    = AbsoluteSeek
    | RelativeSeek
    | SeekFromEnd
    deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Buffering mode for a handle.
--
-- * 'NoBuffering' - No buffering; each operation reads/writes immediately
-- * 'LineBuffering' - Buffer until a newline is encountered
-- * 'BlockBuffering' - Buffer a fixed number of bytes (or default if 'Nothing')
data BufferMode
    = NoBuffering
    | LineBuffering
    | BlockBuffering (Maybe Int)
    deriving (Eq, Ord, Show, Read)

-- ============================================================
-- Standard Handles
-- ============================================================

-- | Standard input handle. Connected to the terminal by default.
foreign import ccall "bhc_stdin" stdin :: Handle

-- | Standard output handle. Connected to the terminal by default.
foreign import ccall "bhc_stdout" stdout :: Handle

-- | Standard error handle. Used for error messages.
foreign import ccall "bhc_stderr" stderr :: Handle

-- ============================================================
-- Opening and Closing Files
-- ============================================================

-- | Open a file and return a 'Handle'.
--
-- >>> h <- openFile "data.txt" ReadMode
-- >>> contents <- hGetContents h
-- >>> hClose h
--
-- Consider using 'withFile' instead for automatic resource cleanup.
foreign import ccall "bhc_open_file"
    openFile :: FilePath -> IOMode -> IO Handle

-- | Close a 'Handle', releasing any associated resources.
--
-- Subsequent operations on a closed handle will fail.
foreign import ccall "bhc_close_handle"
    hClose :: Handle -> IO ()

-- | Open a file, perform an action, then close the file.
--
-- The handle is guaranteed to be closed even if an exception is raised.
--
-- >>> withFile "data.txt" ReadMode $ \h -> do
-- >>>     contents <- hGetContents h
-- >>>     putStrLn contents
withFile :: FilePath -> IOMode -> (Handle -> IO a) -> IO a
withFile path mode action = do
    h <- openFile path mode
    r <- action h `catch` \e -> hClose h >> throw e
    hClose h
    return r

-- ============================================================
-- Reading
-- ============================================================

-- | Read a single character from a handle.
foreign import ccall "bhc_hGetChar" hGetChar :: Handle -> IO Char

-- | Read a line from a handle (up to and including newline).
foreign import ccall "bhc_hGetLine" hGetLine :: Handle -> IO String

-- | Read the entire remaining contents of a handle as a string.
--
-- __Note__: For large files, consider streaming or lazy I/O approaches.
foreign import ccall "bhc_hGetContents" hGetContents :: Handle -> IO String

-- | Peek at the next character without consuming it.
foreign import ccall "bhc_hLookAhead" hLookAhead :: Handle -> IO Char

-- | Check if input is available on the handle (non-blocking).
foreign import ccall "bhc_hReady" hReady :: Handle -> IO Bool

-- ============================================================
-- Writing
-- ============================================================

-- | Write a single character to a handle.
foreign import ccall "bhc_hPutChar" hPutChar :: Handle -> Char -> IO ()

-- | Write a string to a handle.
foreign import ccall "bhc_hPutStr" hPutStr :: Handle -> String -> IO ()

-- | Write a string to a handle followed by a newline.
hPutStrLn :: Handle -> String -> IO ()
hPutStrLn h s = hPutStr h s >> hPutChar h '\n'

-- | Write a 'Show'-able value to a handle followed by a newline.
hPrint :: Show a => Handle -> a -> IO ()
hPrint h x = hPutStrLn h (show x)

foreign import ccall "bhc_hFlush" hFlush :: Handle -> IO ()

-- Seeking
foreign import ccall "bhc_hSeek" hSeek :: Handle -> SeekMode -> Integer -> IO ()
foreign import ccall "bhc_hTell" hTell :: Handle -> IO Integer

-- Queries
foreign import ccall "bhc_hIsEOF" hIsEOF :: Handle -> IO Bool
foreign import ccall "bhc_hIsOpen" hIsOpen :: Handle -> IO Bool
foreign import ccall "bhc_hIsClosed" hIsClosed :: Handle -> IO Bool
foreign import ccall "bhc_hIsReadable" hIsReadable :: Handle -> IO Bool
foreign import ccall "bhc_hIsWritable" hIsWritable :: Handle -> IO Bool
foreign import ccall "bhc_hIsSeekable" hIsSeekable :: Handle -> IO Bool

-- Buffering
foreign import ccall "bhc_hSetBuffering" hSetBuffering :: Handle -> BufferMode -> IO ()
foreign import ccall "bhc_hGetBuffering" hGetBuffering :: Handle -> IO BufferMode

-- ============================================================
-- File Operations
-- ============================================================

-- | Read the entire contents of a file as a string.
--
-- >>> contents <- readFile "data.txt"
-- >>> putStrLn contents
--
-- __Note__: For large files, consider streaming approaches.
foreign import ccall "bhc_readFile" readFile :: FilePath -> IO String

-- | Write a string to a file, replacing its contents.
--
-- >>> writeFile "output.txt" "Hello, World!"
--
-- Creates the file if it doesn't exist, truncates it if it does.
foreign import ccall "bhc_writeFile" writeFile :: FilePath -> String -> IO ()

-- | Append a string to a file.
--
-- >>> appendFile "log.txt" "New log entry\n"
--
-- Creates the file if it doesn't exist.
foreign import ccall "bhc_appendFile" appendFile :: FilePath -> String -> IO ()

-- ============================================================
-- Console I/O
-- ============================================================

-- | Write a character to standard output.
putChar :: Char -> IO ()
putChar = hPutChar stdout

-- | Write a string to standard output.
putStr :: String -> IO ()
putStr = hPutStr stdout

-- | Write a string to standard output, followed by a newline.
--
-- >>> putStrLn "Hello, World!"
-- Hello, World!
putStrLn :: String -> IO ()
putStrLn = hPutStrLn stdout

-- | Print a 'Show'-able value to standard output, followed by a newline.
--
-- >>> print [1, 2, 3]
-- [1,2,3]
print :: Show a => a -> IO ()
print = hPrint stdout

-- | Read a single character from standard input.
getChar :: IO Char
getChar = hGetChar stdin

-- | Read a line from standard input (without the newline character).
--
-- >>> name <- getLine
-- >>> putStrLn ("Hello, " ++ name ++ "!")
getLine :: IO String
getLine = hGetLine stdin

-- | Read all input from standard input until EOF.
getContents :: IO String
getContents = hGetContents stdin

-- | Process standard input with a function and write result to standard output.
--
-- >>> interact (map toUpper)  -- Converts all input to uppercase
interact :: (String -> String) -> IO ()
interact f = getContents >>= putStr . f

-- Errors
-- IOError and exception handling imported from Control.Exception
import BHC.Control.Exception (
    Exception(..), SomeException(..),
    throw, catch, try,
    )

data IOError = IOError String
    deriving (Show, Eq)

instance Exception IOError

ioError :: IOError -> IO a
ioError = throw

userError :: String -> IOError
userError = IOError
