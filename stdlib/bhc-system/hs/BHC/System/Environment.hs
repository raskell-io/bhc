-- |
-- Module      : BHC.System.Environment
-- Description : Environment variable and program argument access
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable

module BHC.System.Environment (
    -- * Program arguments
    getArgs,
    getProgName,
    withArgs,
    withProgName,

    -- * Environment variables
    getEnv,
    lookupEnv,
    setEnv,
    unsetEnv,
    getEnvironment,

    -- * System info
    getExecutablePath,
) where

import BHC.Prelude

-- | Get the command-line arguments.
--
-- The first element is NOT the program name (use 'getProgName' for that).
--
-- ==== __Example__
--
-- @
-- main = do
--     args <- getArgs
--     mapM_ putStrLn args
-- @
foreign import ccall "bhc_get_args"
    getArgs :: IO [String]

-- | Get the program name.
foreign import ccall "bhc_get_prog_name"
    getProgName :: IO String

-- | Run an IO action with modified command-line arguments.
withArgs :: [String] -> IO a -> IO a
withArgs newArgs action = do
    oldArgs <- getArgs
    setArgs newArgs
    result <- action
    setArgs oldArgs
    return result

-- | Run an IO action with a modified program name.
withProgName :: String -> IO a -> IO a
withProgName newName action = do
    oldName <- getProgName
    setProgName newName
    result <- action
    setProgName oldName
    return result

foreign import ccall "bhc_set_args" setArgs :: [String] -> IO ()
foreign import ccall "bhc_set_prog_name" setProgName :: String -> IO ()

-- | Get an environment variable.
--
-- Throws an exception if the variable is not set.
--
-- ==== __Example__
--
-- @
-- home <- getEnv "HOME"
-- putStrLn $ "Home directory: " ++ home
-- @
foreign import ccall "bhc_getenv"
    getEnv :: String -> IO String

-- | Look up an environment variable.
--
-- Returns 'Nothing' if the variable is not set.
--
-- ==== __Example__
--
-- @
-- mval <- lookupEnv "DEBUG"
-- case mval of
--     Just val -> putStrLn $ "DEBUG=" ++ val
--     Nothing  -> putStrLn "DEBUG not set"
-- @
foreign import ccall "bhc_lookupenv"
    lookupEnv :: String -> IO (Maybe String)

-- | Set an environment variable.
--
-- ==== __Example__
--
-- @
-- setEnv "MY_VAR" "my_value"
-- @
foreign import ccall "bhc_setenv"
    setEnv :: String -> String -> IO ()

-- | Unset (remove) an environment variable.
foreign import ccall "bhc_unsetenv"
    unsetEnv :: String -> IO ()

-- | Get all environment variables as a list of (name, value) pairs.
--
-- ==== __Example__
--
-- @
-- env <- getEnvironment
-- mapM_ (\(k, v) -> putStrLn $ k ++ "=" ++ v) env
-- @
foreign import ccall "bhc_get_environment"
    getEnvironment :: IO [(String, String)]

-- | Get the absolute path of the current executable.
foreign import ccall "bhc_get_executable_path"
    getExecutablePath :: IO FilePath
