-- |
-- Module      : BHC.System.Directory
-- Description : Directory and file system operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Functions for working with the file system, including directory
-- operations, file queries, and permissions.

module BHC.System.Directory (
    -- * Actions on directories
    createDirectory,
    createDirectoryIfMissing,
    removeDirectory,
    removeDirectoryRecursive,
    renameDirectory,
    listDirectory,
    getDirectoryContents,

    -- * Current directory
    getCurrentDirectory,
    setCurrentDirectory,
    withCurrentDirectory,

    -- * Pre-defined directories
    getHomeDirectory,
    getAppUserDataDirectory,
    getUserDocumentsDirectory,
    getTemporaryDirectory,

    -- * Actions on files
    removeFile,
    renameFile,
    renamePath,
    copyFile,
    copyFileWithMetadata,

    -- * File existence and type
    doesPathExist,
    doesFileExist,
    doesDirectoryExist,

    -- * Symbolic links
    pathIsSymbolicLink,
    createFileLink,
    createDirectoryLink,
    removeDirectoryLink,
    getSymbolicLinkTarget,

    -- * Permissions
    Permissions(..),
    getPermissions,
    setPermissions,
    copyPermissions,

    -- * Timestamps
    getAccessTime,
    getModificationTime,
    setAccessTime,
    setModificationTime,

    -- * File size
    getFileSize,

    -- * Canonicalization
    canonicalizePath,
    makeAbsolute,
    makeRelativeToCurrentDirectory,

    -- * Searching
    findExecutable,
    findExecutables,
    findExecutablesInDirectories,
    findFile,
    findFiles,
    findFileWith,
    findFilesWith,

    -- * XDG directories
    XdgDirectory(..),
    getXdgDirectory,
    XdgDirectoryList(..),
    getXdgDirectoryList,
) where

import BHC.Prelude
import Data.Time.Clock (UTCTime)

-- | File permissions.
data Permissions = Permissions
    { readable   :: Bool
    , writable   :: Bool
    , executable :: Bool
    , searchable :: Bool
    } deriving (Eq, Ord, Read, Show)

-- | XDG Base Directory specification.
data XdgDirectory
    = XdgData     -- ^ User data directory
    | XdgConfig   -- ^ User configuration directory
    | XdgCache    -- ^ User cache directory
    | XdgState    -- ^ User state directory
    deriving (Eq, Ord, Read, Show, Enum, Bounded)

-- | XDG directory lists.
data XdgDirectoryList
    = XdgDataDirs   -- ^ System data directories
    | XdgConfigDirs -- ^ System configuration directories
    deriving (Eq, Ord, Read, Show, Enum, Bounded)

-- | Create a directory.
--
-- Fails if the parent directory doesn't exist or the directory already exists.
--
-- ==== __Example__
--
-- @
-- createDirectory "/tmp/mydir"
-- @
foreign import ccall "bhc_create_directory"
    createDirectory :: FilePath -> IO ()

-- | Create a directory, creating parent directories as needed.
--
-- If the first argument is 'True', also create parent directories.
--
-- ==== __Example__
--
-- @
-- createDirectoryIfMissing True "/tmp/parent/child/grandchild"
-- @
foreign import ccall "bhc_create_directory_if_missing"
    createDirectoryIfMissing :: Bool -> FilePath -> IO ()

-- | Remove an empty directory.
foreign import ccall "bhc_remove_directory"
    removeDirectory :: FilePath -> IO ()

-- | Remove a directory and all its contents recursively.
foreign import ccall "bhc_remove_directory_recursive"
    removeDirectoryRecursive :: FilePath -> IO ()

-- | Rename a directory.
foreign import ccall "bhc_rename_directory"
    renameDirectory :: FilePath -> FilePath -> IO ()

-- | List the contents of a directory.
--
-- Does not include \".\" and \"..\".
--
-- ==== __Example__
--
-- @
-- contents <- listDirectory "/tmp"
-- mapM_ putStrLn contents
-- @
foreign import ccall "bhc_list_directory"
    listDirectory :: FilePath -> IO [FilePath]

-- | Get directory contents including \".\" and \"..\".
foreign import ccall "bhc_get_directory_contents"
    getDirectoryContents :: FilePath -> IO [FilePath]

-- | Get the current working directory.
foreign import ccall "bhc_get_current_directory"
    getCurrentDirectory :: IO FilePath

-- | Set the current working directory.
foreign import ccall "bhc_set_current_directory"
    setCurrentDirectory :: FilePath -> IO ()

-- | Run an action in a different directory.
withCurrentDirectory :: FilePath -> IO a -> IO a
withCurrentDirectory dir action = do
    old <- getCurrentDirectory
    setCurrentDirectory dir
    result <- action `catch` \e -> setCurrentDirectory old >> throw e
    setCurrentDirectory old
    return result
  where
    catch :: IO a -> (SomeException -> IO a) -> IO a
    catch = catchException
    throw :: SomeException -> IO a
    throw = throwException

foreign import ccall "bhc_catch" catchException :: IO a -> (SomeException -> IO a) -> IO a
foreign import ccall "bhc_throw" throwException :: SomeException -> IO a

data SomeException = SomeException String

-- | Get the user's home directory.
foreign import ccall "bhc_get_home_directory"
    getHomeDirectory :: IO FilePath

-- | Get the application data directory for a given application name.
foreign import ccall "bhc_get_app_user_data_directory"
    getAppUserDataDirectory :: String -> IO FilePath

-- | Get the user's documents directory.
foreign import ccall "bhc_get_user_documents_directory"
    getUserDocumentsDirectory :: IO FilePath

-- | Get the system temporary directory.
foreign import ccall "bhc_get_temporary_directory"
    getTemporaryDirectory :: IO FilePath

-- | Remove a file.
foreign import ccall "bhc_remove_file"
    removeFile :: FilePath -> IO ()

-- | Rename a file.
foreign import ccall "bhc_rename_file"
    renameFile :: FilePath -> FilePath -> IO ()

-- | Rename a file or directory.
foreign import ccall "bhc_rename_path"
    renamePath :: FilePath -> FilePath -> IO ()

-- | Copy a file.
foreign import ccall "bhc_copy_file"
    copyFile :: FilePath -> FilePath -> IO ()

-- | Copy a file, preserving metadata.
foreign import ccall "bhc_copy_file_with_metadata"
    copyFileWithMetadata :: FilePath -> FilePath -> IO ()

-- | Check if a path exists.
foreign import ccall "bhc_does_path_exist"
    doesPathExist :: FilePath -> IO Bool

-- | Check if a file exists.
foreign import ccall "bhc_does_file_exist"
    doesFileExist :: FilePath -> IO Bool

-- | Check if a directory exists.
foreign import ccall "bhc_does_directory_exist"
    doesDirectoryExist :: FilePath -> IO Bool

-- | Check if a path is a symbolic link.
foreign import ccall "bhc_path_is_symbolic_link"
    pathIsSymbolicLink :: FilePath -> IO Bool

-- | Create a symbolic link to a file.
foreign import ccall "bhc_create_file_link"
    createFileLink :: FilePath -> FilePath -> IO ()

-- | Create a symbolic link to a directory.
foreign import ccall "bhc_create_directory_link"
    createDirectoryLink :: FilePath -> FilePath -> IO ()

-- | Remove a directory symbolic link.
foreign import ccall "bhc_remove_directory_link"
    removeDirectoryLink :: FilePath -> IO ()

-- | Get the target of a symbolic link.
foreign import ccall "bhc_get_symbolic_link_target"
    getSymbolicLinkTarget :: FilePath -> IO FilePath

-- | Get file permissions.
foreign import ccall "bhc_get_permissions"
    getPermissions :: FilePath -> IO Permissions

-- | Set file permissions.
foreign import ccall "bhc_set_permissions"
    setPermissions :: FilePath -> Permissions -> IO ()

-- | Copy permissions from one path to another.
copyPermissions :: FilePath -> FilePath -> IO ()
copyPermissions src dst = do
    perms <- getPermissions src
    setPermissions dst perms

-- | Get the last access time.
foreign import ccall "bhc_get_access_time"
    getAccessTime :: FilePath -> IO UTCTime

-- | Get the last modification time.
foreign import ccall "bhc_get_modification_time"
    getModificationTime :: FilePath -> IO UTCTime

-- | Set the last access time.
foreign import ccall "bhc_set_access_time"
    setAccessTime :: FilePath -> UTCTime -> IO ()

-- | Set the last modification time.
foreign import ccall "bhc_set_modification_time"
    setModificationTime :: FilePath -> UTCTime -> IO ()

-- | Get the size of a file in bytes.
foreign import ccall "bhc_get_file_size"
    getFileSize :: FilePath -> IO Integer

-- | Canonicalize a path (resolve symlinks and normalize).
foreign import ccall "bhc_canonicalize_path"
    canonicalizePath :: FilePath -> IO FilePath

-- | Make a path absolute.
foreign import ccall "bhc_make_absolute"
    makeAbsolute :: FilePath -> IO FilePath

-- | Make a path relative to the current directory.
makeRelativeToCurrentDirectory :: FilePath -> IO FilePath
makeRelativeToCurrentDirectory path = do
    cwd <- getCurrentDirectory
    return $ makeRelative cwd path
  where
    makeRelative base target = bhc_make_relative base target

foreign import ccall "bhc_filepath_make_relative"
    bhc_make_relative :: FilePath -> FilePath -> FilePath

-- | Find an executable on the PATH.
foreign import ccall "bhc_find_executable"
    findExecutable :: String -> IO (Maybe FilePath)

-- | Find all occurrences of an executable on the PATH.
foreign import ccall "bhc_find_executables"
    findExecutables :: String -> IO [FilePath]

-- | Find executables in specified directories.
foreign import ccall "bhc_find_executables_in_directories"
    findExecutablesInDirectories :: [FilePath] -> String -> IO [FilePath]

-- | Find a file in the given directories.
foreign import ccall "bhc_find_file"
    findFile :: [FilePath] -> String -> IO (Maybe FilePath)

-- | Find all matching files in the given directories.
foreign import ccall "bhc_find_files"
    findFiles :: [FilePath] -> String -> IO [FilePath]

-- | Find a file matching a predicate.
findFileWith :: (FilePath -> IO Bool) -> [FilePath] -> String -> IO (Maybe FilePath)
findFileWith predicate dirs name = do
    candidates <- findFiles dirs name
    filterM predicate candidates >>= return . listToMaybe
  where
    listToMaybe [] = Nothing
    listToMaybe (x:_) = Just x
    filterM _ [] = return []
    filterM p (x:xs) = do
        b <- p x
        rest <- filterM p xs
        return $ if b then x:rest else rest

-- | Find all files matching a predicate.
findFilesWith :: (FilePath -> IO Bool) -> [FilePath] -> String -> IO [FilePath]
findFilesWith predicate dirs name = do
    candidates <- findFiles dirs name
    filterM predicate candidates
  where
    filterM _ [] = return []
    filterM p (x:xs) = do
        b <- p x
        rest <- filterM p xs
        return $ if b then x:rest else rest

-- | Get an XDG directory.
foreign import ccall "bhc_get_xdg_directory"
    getXdgDirectory :: XdgDirectory -> FilePath -> IO FilePath

-- | Get an XDG directory list.
foreign import ccall "bhc_get_xdg_directory_list"
    getXdgDirectoryList :: XdgDirectoryList -> IO [FilePath]
