-- |
-- Module      : BHC.System.Process
-- Description : Process spawning and management
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Functions for spawning and managing external processes.

module BHC.System.Process (
    -- * Running simple commands
    callProcess,
    callCommand,
    spawnProcess,
    spawnCommand,
    readProcess,
    readProcessWithExitCode,
    readCreateProcess,
    readCreateProcessWithExitCode,

    -- * Process configuration
    CreateProcess(..),
    StdStream(..),
    CmdSpec(..),
    shell,
    proc,

    -- * Process handles
    ProcessHandle,
    Pid,
    getPid,

    -- * Running processes
    createProcess,
    createProcess_,
    cleanupProcess,

    -- * Waiting for processes
    waitForProcess,
    getProcessExitCode,
    terminateProcess,
    interruptProcessGroupOf,

    -- * Low-level process spawning
    runProcess,
    runCommand,
    runInteractiveProcess,
    runInteractiveCommand,

    -- * Environment and working directory
    withCreateProcess,
    cwd,
    env,

    -- * Utility
    showCommandForUser,
    system,
    rawSystem,
) where

import BHC.Prelude
import BHC.System.Exit (ExitCode(..))

-- | Process identifier.
type Pid = Int

-- | An opaque handle to a running process.
data ProcessHandle = ProcessHandle
    { phPid :: Pid
    , phProgram :: String
    }
    deriving (Eq, Show)

-- | Specification of how to start a process.
data CmdSpec
    = ShellCommand String           -- ^ A shell command
    | RawCommand FilePath [String]  -- ^ Program and arguments
    deriving (Eq, Show)

-- | How to handle standard streams.
data StdStream
    = Inherit       -- ^ Inherit from parent process
    | UseHandle Handle  -- ^ Use provided handle
    | CreatePipe    -- ^ Create a pipe
    | NoStream      -- ^ Close the stream
    deriving (Eq, Show)

data Handle  -- Opaque handle type

-- | Full process configuration.
data CreateProcess = CreateProcess
    { cmdspec       :: CmdSpec
    , cwd           :: Maybe FilePath
    , env           :: Maybe [(String, String)]
    , std_in        :: StdStream
    , std_out       :: StdStream
    , std_err       :: StdStream
    , close_fds     :: Bool
    , create_group  :: Bool
    , delegate_ctlc :: Bool
    , detach_console :: Bool
    , create_new_console :: Bool
    , new_session   :: Bool
    , child_group   :: Maybe Int
    , child_user    :: Maybe Int
    , use_process_jobs :: Bool
    }
    deriving (Eq, Show)

-- | Create a shell command specification.
shell :: String -> CreateProcess
shell cmd = CreateProcess
    { cmdspec = ShellCommand cmd
    , cwd = Nothing
    , env = Nothing
    , std_in = Inherit
    , std_out = Inherit
    , std_err = Inherit
    , close_fds = False
    , create_group = False
    , delegate_ctlc = False
    , detach_console = False
    , create_new_console = False
    , new_session = False
    , child_group = Nothing
    , child_user = Nothing
    , use_process_jobs = False
    }

-- | Create a raw command specification.
proc :: FilePath -> [String] -> CreateProcess
proc cmd args = CreateProcess
    { cmdspec = RawCommand cmd args
    , cwd = Nothing
    , env = Nothing
    , std_in = Inherit
    , std_out = Inherit
    , std_err = Inherit
    , close_fds = False
    , create_group = False
    , delegate_ctlc = False
    , detach_console = False
    , create_new_console = False
    , new_session = False
    , child_group = Nothing
    , child_user = Nothing
    , use_process_jobs = False
    }

-- | Call a process and wait for it to complete.
--
-- ==== __Example__
--
-- @
-- callProcess "ls" ["-la"]
-- @
callProcess :: FilePath -> [String] -> IO ()
callProcess cmd args = do
    ec <- rawSystem cmd args
    case ec of
        ExitSuccess -> return ()
        ExitFailure n -> error $ cmd ++ " exited with code " ++ show n

-- | Call a shell command and wait for it to complete.
--
-- ==== __Example__
--
-- @
-- callCommand "ls -la | grep txt"
-- @
callCommand :: String -> IO ()
callCommand cmd = do
    ec <- system cmd
    case ec of
        ExitSuccess -> return ()
        ExitFailure n -> error $ "command failed with code " ++ show n

-- | Spawn a process without waiting.
--
-- ==== __Example__
--
-- @
-- ph <- spawnProcess "editor" ["file.txt"]
-- -- do other work
-- waitForProcess ph
-- @
foreign import ccall "bhc_spawn_process"
    spawnProcess :: FilePath -> [String] -> IO ProcessHandle

-- | Spawn a shell command without waiting.
foreign import ccall "bhc_spawn_command"
    spawnCommand :: String -> IO ProcessHandle

-- | Run a process and capture its stdout.
--
-- ==== __Example__
--
-- @
-- output <- readProcess "echo" ["Hello, World!"] ""
-- putStr output
-- @
foreign import ccall "bhc_read_process"
    readProcess :: FilePath -> [String] -> String -> IO String

-- | Run a process and capture its stdout, stderr, and exit code.
foreign import ccall "bhc_read_process_with_exit_code"
    readProcessWithExitCode :: FilePath -> [String] -> String -> IO (ExitCode, String, String)

-- | Run a process with CreateProcess configuration.
readCreateProcess :: CreateProcess -> String -> IO String
readCreateProcess cp input = do
    (_, stdout, _) <- readCreateProcessWithExitCode cp input
    return stdout

-- | Run a process with CreateProcess configuration, capturing exit code.
foreign import ccall "bhc_read_create_process"
    readCreateProcessWithExitCode :: CreateProcess -> String -> IO (ExitCode, String, String)

-- | Create a process with full configuration.
foreign import ccall "bhc_create_process"
    createProcess :: CreateProcess -> IO (Maybe Handle, Maybe Handle, Maybe Handle, ProcessHandle)

-- | Create a process, ignoring the name parameter.
createProcess_ :: String -> CreateProcess -> IO (Maybe Handle, Maybe Handle, Maybe Handle, ProcessHandle)
createProcess_ _ = createProcess

-- | Clean up a process and its handles.
cleanupProcess :: (Maybe Handle, Maybe Handle, Maybe Handle, ProcessHandle) -> IO ()
cleanupProcess (mIn, mOut, mErr, ph) = do
    maybeClose mIn
    maybeClose mOut
    maybeClose mErr
    terminateProcess ph
  where
    maybeClose Nothing = return ()
    maybeClose (Just h) = bhc_close h

foreign import ccall "bhc_close_handle" bhc_close :: Handle -> IO ()

-- | Wait for a process to complete and return its exit code.
foreign import ccall "bhc_wait_for_process"
    waitForProcess :: ProcessHandle -> IO ExitCode

-- | Check if a process has exited without blocking.
foreign import ccall "bhc_get_process_exit_code"
    getProcessExitCode :: ProcessHandle -> IO (Maybe ExitCode)

-- | Terminate a process.
foreign import ccall "bhc_terminate_process"
    terminateProcess :: ProcessHandle -> IO ()

-- | Send an interrupt to the process group.
foreign import ccall "bhc_interrupt_process_group"
    interruptProcessGroupOf :: ProcessHandle -> IO ()

-- | Get the process ID.
getPid :: ProcessHandle -> IO (Maybe Pid)
getPid ph = return $ Just (phPid ph)

-- | Run a process with full control.
foreign import ccall "bhc_run_process"
    runProcess :: FilePath -> [String] -> Maybe FilePath -> Maybe [(String, String)]
               -> Maybe Handle -> Maybe Handle -> Maybe Handle -> IO ProcessHandle

-- | Run a shell command.
foreign import ccall "bhc_run_command"
    runCommand :: String -> IO ProcessHandle

-- | Run an interactive process with piped stdin/stdout/stderr.
foreign import ccall "bhc_run_interactive_process"
    runInteractiveProcess :: FilePath -> [String] -> Maybe FilePath -> Maybe [(String, String)]
                          -> IO (Handle, Handle, Handle, ProcessHandle)

-- | Run an interactive shell command.
foreign import ccall "bhc_run_interactive_command"
    runInteractiveCommand :: String -> IO (Handle, Handle, Handle, ProcessHandle)

-- | Run an action with a process, ensuring cleanup.
withCreateProcess :: CreateProcess -> ((Maybe Handle, Maybe Handle, Maybe Handle, ProcessHandle) -> IO a) -> IO a
withCreateProcess cp action = do
    handles <- createProcess cp
    result <- action handles `catch` \e -> cleanupProcess handles >> throw e
    cleanupProcess handles
    return result
  where
    catch :: IO a -> (SomeException -> IO a) -> IO a
    catch = catchException
    throw :: SomeException -> IO a
    throw = throwException

foreign import ccall "bhc_catch" catchException :: IO a -> (SomeException -> IO a) -> IO a
foreign import ccall "bhc_throw" throwException :: SomeException -> IO a

data SomeException = SomeException String

-- | Format a command for display.
showCommandForUser :: FilePath -> [String] -> String
showCommandForUser cmd args = unwords (cmd : map quote args)
  where
    quote s
        | any (`elem` " \t\"'\\") s = "\"" ++ escape s ++ "\""
        | otherwise = s
    escape [] = []
    escape (c:cs)
        | c `elem` "\"'\\" = '\\' : c : escape cs
        | otherwise = c : escape cs

-- | Run a shell command and return its exit code.
--
-- ==== __Example__
--
-- @
-- exitCode <- system "ls -la"
-- case exitCode of
--     ExitSuccess -> putStrLn "Success!"
--     ExitFailure n -> putStrLn $ "Failed with code " ++ show n
-- @
foreign import ccall "bhc_system"
    system :: String -> IO ExitCode

-- | Run a raw command with arguments and return its exit code.
foreign import ccall "bhc_raw_system"
    rawSystem :: FilePath -> [String] -> IO ExitCode
