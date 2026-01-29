//! BHC System Library - Rust support
//!
//! OS interaction primitives for BHC.
//!
//! # Architecture
//!
//! The high-level System.* API is **defined in Haskell** (see
//! `hs/BHC/System/*.hs`). This Rust crate provides FFI primitives
//! for OS operations that cannot be expressed in pure Haskell.
//!
//! # What belongs here
//!
//! - File I/O primitives (open, read, write, close)
//! - Environment variable access
//! - Directory operations
//! - Process spawning
//! - Exit codes
//!
//! # What does NOT belong here
//!
//! - High-level Handle abstraction (that's Haskell)
//! - Monad instances for IO (that's Haskell)
//! - Exception handling (that's Haskell + RTS)
//!
//! # FFI Exports
//!
//! This crate exports C-ABI functions for BHC to call:
//! - IO: `bhc_open`, `bhc_read`, `bhc_write`, `bhc_close`
//! - Environment: `bhc_getenv`, `bhc_setenv`, `bhc_getargs`
//! - Directory: `bhc_mkdir`, `bhc_rmdir`, `bhc_listdir`
//! - Process: `bhc_spawn`, `bhc_wait`
//! - Exit: `bhc_exit`
//!
//! # Modules
//!
//! - [`io`] - File handles and buffered I/O
//! - [`environment`] - Environment variables and program arguments
//! - [`filepath`] - Path manipulation utilities
//! - [`directory`] - Directory operations
//! - [`exit`] - Program exit codes
//! - [`process`] - Process spawning and management

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod directory;
pub mod environment;
pub mod exit;
pub mod filepath;
pub mod io;
pub mod process;

// Re-export commonly used items
pub use directory::{
    create_directory, create_directory_all, current_directory, exists, is_directory, is_file,
    list_directory, remove_directory, remove_directory_all, remove_file, rename,
    set_current_directory,
};
pub use environment::{get_args, get_env, get_environment, lookup_env, set_env, unset_env};
pub use exit::{exit, exit_failure, exit_success, ExitCode};
pub use filepath::{
    extension, file_name, is_absolute, is_relative, join, normalize, parent, set_extension,
    split_extension, stem,
};
pub use io::{
    append_file, read_file, read_file_bytes, write_file, write_file_bytes, Handle, OpenMode,
};
pub use process::{spawn, Command, Process, ProcessOutput};
