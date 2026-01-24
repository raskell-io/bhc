-- |
-- Module      : BHC.System.FilePath
-- Description : File path manipulation
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Cross-platform file path manipulation utilities.

module BHC.System.FilePath (
    -- * Types
    FilePath,

    -- * Separators
    pathSeparator,
    pathSeparators,
    isPathSeparator,
    searchPathSeparator,
    isSearchPathSeparator,
    extSeparator,
    isExtSeparator,

    -- * Path operations
    -- ** Joining and splitting
    (</>),
    splitPath,
    joinPath,
    splitDirectories,

    -- ** File name operations
    takeFileName,
    replaceFileName,
    dropFileName,
    takeBaseName,
    replaceBaseName,
    takeDirectory,
    replaceDirectory,

    -- ** Extension operations
    takeExtension,
    replaceExtension,
    dropExtension,
    addExtension,
    hasExtension,
    (<.>),
    splitExtension,
    splitExtensions,
    takeExtensions,
    replaceExtensions,
    dropExtensions,

    -- ** Drive operations (Windows)
    takeDrive,
    hasDrive,
    dropDrive,
    isDrive,

    -- ** Trailing path separator
    hasTrailingPathSeparator,
    addTrailingPathSeparator,
    dropTrailingPathSeparator,

    -- ** Queries
    isAbsolute,
    isRelative,
    isValid,
    makeValid,

    -- ** Normalization
    normalise,
    equalFilePath,
    makeRelative,

    -- ** Search path
    splitSearchPath,
    getSearchPath,
) where

import BHC.Prelude

-- | The character that separates directories in a path.
pathSeparator :: Char
pathSeparator = '/'

-- | All characters that can separate directories.
pathSeparators :: [Char]
pathSeparators = "/\\"

-- | Check if a character is a path separator.
isPathSeparator :: Char -> Bool
isPathSeparator c = c `elem` pathSeparators

-- | The character that separates paths in the search path.
searchPathSeparator :: Char
searchPathSeparator = ':'

-- | Check if a character is the search path separator.
isSearchPathSeparator :: Char -> Bool
isSearchPathSeparator = (== searchPathSeparator)

-- | The character that separates file extensions.
extSeparator :: Char
extSeparator = '.'

-- | Check if a character is the extension separator.
isExtSeparator :: Char -> Bool
isExtSeparator = (== extSeparator)

-- | Join two paths.
--
-- ==== __Example__
--
-- >>> "/home" </> "user"
-- "/home/user"
(</>) :: FilePath -> FilePath -> FilePath
(</>) = joinPaths

foreign import ccall "bhc_filepath_join"
    joinPaths :: FilePath -> FilePath -> FilePath

-- | Split a path into components.
--
-- ==== __Example__
--
-- >>> splitPath "/home/user/file.txt"
-- ["/", "home/", "user/", "file.txt"]
foreign import ccall "bhc_filepath_split"
    splitPath :: FilePath -> [FilePath]

-- | Join path components.
joinPath :: [FilePath] -> FilePath
joinPath = foldr (</>) ""

-- | Split a path into directories.
foreign import ccall "bhc_filepath_split_dirs"
    splitDirectories :: FilePath -> [FilePath]

-- | Get the file name from a path.
--
-- ==== __Example__
--
-- >>> takeFileName "/home/user/file.txt"
-- "file.txt"
foreign import ccall "bhc_filepath_filename"
    takeFileName :: FilePath -> FilePath

-- | Replace the file name in a path.
replaceFileName :: FilePath -> String -> FilePath
replaceFileName path name = takeDirectory path </> name

-- | Remove the file name from a path.
dropFileName :: FilePath -> FilePath
dropFileName = takeDirectory

-- | Get the base name (file name without extension).
--
-- ==== __Example__
--
-- >>> takeBaseName "/home/user/file.txt"
-- "file"
foreign import ccall "bhc_filepath_basename"
    takeBaseName :: FilePath -> String

-- | Replace the base name in a path.
replaceBaseName :: FilePath -> String -> FilePath
replaceBaseName path name = takeDirectory path </> name <.> takeExtension path

-- | Get the directory from a path.
--
-- ==== __Example__
--
-- >>> takeDirectory "/home/user/file.txt"
-- "/home/user"
foreign import ccall "bhc_filepath_directory"
    takeDirectory :: FilePath -> FilePath

-- | Replace the directory in a path.
replaceDirectory :: FilePath -> FilePath -> FilePath
replaceDirectory path dir = dir </> takeFileName path

-- | Get the extension from a path.
--
-- ==== __Example__
--
-- >>> takeExtension "/home/user/file.txt"
-- ".txt"
foreign import ccall "bhc_filepath_extension"
    takeExtension :: FilePath -> String

-- | Replace the extension in a path.
foreign import ccall "bhc_filepath_replace_ext"
    replaceExtension :: FilePath -> String -> FilePath

-- | Remove the extension from a path.
foreign import ccall "bhc_filepath_drop_ext"
    dropExtension :: FilePath -> FilePath

-- | Add an extension to a path.
foreign import ccall "bhc_filepath_add_ext"
    addExtension :: FilePath -> String -> FilePath

-- | Check if a path has an extension.
foreign import ccall "bhc_filepath_has_ext"
    hasExtension :: FilePath -> Bool

-- | Infix version of 'addExtension'.
(<.>) :: FilePath -> String -> FilePath
(<.>) = addExtension

-- | Split path into (path without extension, extension).
splitExtension :: FilePath -> (String, String)
splitExtension path = (dropExtension path, takeExtension path)

-- | Split on all extensions.
foreign import ccall "bhc_filepath_split_exts"
    splitExtensions :: FilePath -> (FilePath, String)

-- | Get all extensions.
takeExtensions :: FilePath -> String
takeExtensions = snd . splitExtensions

-- | Replace all extensions.
replaceExtensions :: FilePath -> String -> FilePath
replaceExtensions path ext = fst (splitExtensions path) ++ ext

-- | Drop all extensions.
dropExtensions :: FilePath -> FilePath
dropExtensions = fst . splitExtensions

-- | Get the drive from a path (Windows).
foreign import ccall "bhc_filepath_drive"
    takeDrive :: FilePath -> FilePath

-- | Check if a path has a drive.
hasDrive :: FilePath -> Bool
hasDrive = not . null . takeDrive

-- | Remove the drive from a path.
foreign import ccall "bhc_filepath_drop_drive"
    dropDrive :: FilePath -> FilePath

-- | Check if a path is just a drive.
isDrive :: FilePath -> Bool
isDrive path = takeDrive path == path && not (null path)

-- | Check if a path has a trailing separator.
foreign import ccall "bhc_filepath_has_trailing_sep"
    hasTrailingPathSeparator :: FilePath -> Bool

-- | Add a trailing separator if not present.
foreign import ccall "bhc_filepath_add_trailing_sep"
    addTrailingPathSeparator :: FilePath -> FilePath

-- | Remove trailing separator if present.
foreign import ccall "bhc_filepath_drop_trailing_sep"
    dropTrailingPathSeparator :: FilePath -> FilePath

-- | Check if a path is absolute.
--
-- ==== __Example__
--
-- >>> isAbsolute "/home/user"
-- True
-- >>> isAbsolute "relative/path"
-- False
foreign import ccall "bhc_filepath_is_absolute"
    isAbsolute :: FilePath -> Bool

-- | Check if a path is relative.
isRelative :: FilePath -> Bool
isRelative = not . isAbsolute

-- | Check if a path is valid.
foreign import ccall "bhc_filepath_is_valid"
    isValid :: FilePath -> Bool

-- | Make a path valid by escaping invalid characters.
foreign import ccall "bhc_filepath_make_valid"
    makeValid :: FilePath -> FilePath

-- | Normalize a path.
--
-- ==== __Example__
--
-- >>> normalise "/home/../home/user/./file.txt"
-- "/home/user/file.txt"
foreign import ccall "bhc_filepath_normalise"
    normalise :: FilePath -> FilePath

-- | Check if two paths are equal (after normalization).
equalFilePath :: FilePath -> FilePath -> Bool
equalFilePath p1 p2 = normalise p1 == normalise p2

-- | Make a path relative to a base path.
foreign import ccall "bhc_filepath_make_relative"
    makeRelative :: FilePath -> FilePath -> FilePath

-- | Split a search path into individual paths.
foreign import ccall "bhc_filepath_split_search"
    splitSearchPath :: String -> [FilePath]

-- | Get the search path from the environment.
getSearchPath :: IO [FilePath]
getSearchPath = do
    path <- lookupEnv "PATH"
    return $ maybe [] splitSearchPath path
  where
    lookupEnv name = do
        val <- bhc_lookupenv name
        return val

foreign import ccall "bhc_lookupenv"
    bhc_lookupenv :: String -> IO (Maybe String)
