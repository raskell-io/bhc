-- |
-- Module      : BHC.System.Exit
-- Description : Program exit codes and termination
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Functions for terminating a program with exit codes.

module BHC.System.Exit (
    -- * Exit codes
    ExitCode(..),
    exitWith,
    exitFailure,
    exitSuccess,

    -- * Exception-based exit
    ExitException(..),
    die,
) where

import BHC.Prelude

-- | An exit code from a process.
--
-- The 'Eq' instance treats all 'ExitFailure' codes as equal.
data ExitCode
    = ExitSuccess        -- ^ Exit code 0, indicating success
    | ExitFailure Int    -- ^ Non-zero exit code, indicating failure
    deriving (Eq, Ord, Read, Show)

-- | Exception type for program exit.
data ExitException = ExitException ExitCode
    deriving (Eq, Show)

instance Exception ExitException

-- | Terminate the program with the given exit code.
--
-- ==== __Example__
--
-- @
-- main = do
--     success <- runTask
--     if success
--         then exitSuccess
--         else exitWith (ExitFailure 1)
-- @
exitWith :: ExitCode -> IO a
exitWith code = do
    case code of
        ExitSuccess -> bhc_exit 0
        ExitFailure n -> bhc_exit n
    -- This shouldn't be reached, but satisfy the type checker
    error "exitWith: unreachable"

foreign import ccall "bhc_exit"
    bhc_exit :: Int -> IO ()

-- | Exit with a failure code (1).
exitFailure :: IO a
exitFailure = exitWith (ExitFailure 1)

-- | Exit with success code (0).
exitSuccess :: IO a
exitSuccess = exitWith ExitSuccess

-- | Print an error message and exit with failure.
--
-- ==== __Example__
--
-- @
-- main = do
--     args <- getArgs
--     when (null args) $ die "Usage: program <filename>"
--     processFile (head args)
-- @
die :: String -> IO a
die msg = do
    hPutStrLn stderr msg
    exitFailure
  where
    stderr = bhc_stderr
    hPutStrLn h s = do
        bhc_hPutStr h s
        bhc_hPutStr h "\n"

foreign import ccall "bhc_stderr" bhc_stderr :: Handle
foreign import ccall "bhc_hPutStr" bhc_hPutStr :: Handle -> String -> IO ()

data Handle  -- Opaque handle type

class Exception e
