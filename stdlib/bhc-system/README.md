# bhc-system

System and IO primitives for the Basel Haskell Compiler.

## Overview

This crate provides Rust-side OS interaction primitives for BHC. The high-level System.* API is defined in Haskell, while this crate provides FFI primitives for operations requiring system access.

## Modules

| Module | Description |
|--------|-------------|
| `io` | File handles and buffered I/O |
| `environment` | Environment variables and args |
| `filepath` | Path manipulation |
| `directory` | Directory operations |
| `exit` | Program exit codes |
| `process` | Process spawning |

## File I/O

### Reading Files

```haskell
import System.IO

main :: IO ()
main = do
  contents <- readFile "input.txt"
  putStrLn contents
```

### Writing Files

```haskell
import System.IO

main :: IO ()
main = do
  writeFile "output.txt" "Hello, World!"
  appendFile "output.txt" "\nMore content"
```

### Handles

```haskell
import System.IO

main :: IO ()
main = withFile "data.txt" ReadMode $ \handle -> do
  line <- hGetLine handle
  putStrLn line
```

## Environment

### Environment Variables

```haskell
import System.Environment

main :: IO ()
main = do
  path <- getEnv "PATH"
  home <- lookupEnv "HOME"
  args <- getArgs
  progName <- getProgName
  print (path, home, args, progName)
```

### Setting Variables

```haskell
import System.Environment

main :: IO ()
main = do
  setEnv "MY_VAR" "value"
  unsetEnv "OLD_VAR"
```

## Directory Operations

```haskell
import System.Directory

main :: IO ()
main = do
  cwd <- getCurrentDirectory
  files <- listDirectory "."
  createDirectory "new_dir"
  doesFileExist "file.txt" >>= print
  doesDirectoryExist "dir" >>= print
```

## FilePath Manipulation

```haskell
import System.FilePath

main :: IO ()
main = do
  let path = "dir" </> "file.txt"
  print $ takeExtension path      -- ".txt"
  print $ takeBaseName path       -- "file"
  print $ takeDirectory path      -- "dir"
  print $ replaceExtension path ".md"
```

## Process Spawning

```haskell
import System.Process

main :: IO ()
main = do
  -- Simple command
  callCommand "echo Hello"

  -- With capture
  output <- readProcess "ls" ["-la"] ""
  putStrLn output

  -- Full control
  (_, Just hout, _, ph) <- createProcess (proc "cat" ["file.txt"])
    { std_out = CreatePipe }
  contents <- hGetContents hout
  waitForProcess ph
```

## Exit Codes

```haskell
import System.Exit

main :: IO ()
main = do
  exitSuccess          -- Exit with code 0
  exitFailure          -- Exit with code 1
  exitWith (ExitFailure 42)  -- Exit with specific code
```

## FFI Exports

| Function | Description |
|----------|-------------|
| `bhc_open` | Open file handle |
| `bhc_read` | Read from handle |
| `bhc_write` | Write to handle |
| `bhc_close` | Close handle |
| `bhc_getenv` | Get environment variable |
| `bhc_setenv` | Set environment variable |
| `bhc_getargs` | Get program arguments |
| `bhc_mkdir` | Create directory |
| `bhc_rmdir` | Remove directory |
| `bhc_listdir` | List directory contents |
| `bhc_spawn` | Spawn process |
| `bhc_wait` | Wait for process |
| `bhc_exit` | Exit program |

## Design Notes

- Buffered I/O for performance
- UTF-8 encoding by default
- Cross-platform path handling
- Proper resource cleanup

## Related Crates

- `bhc-prelude` - IO monad
- `bhc-concurrent` - Async I/O
- `bhc-rts` - Runtime system

## Specification References

- H26-SPEC Section 5: Standard Library
- H26-SPEC Section 5.6: System.IO
