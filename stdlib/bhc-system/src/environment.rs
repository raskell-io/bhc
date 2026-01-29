//! Environment variable and program argument access
//!
//! This module provides functions to access environment variables and
//! command-line arguments.
//!
//! # Example
//!
//! ```no_run
//! use bhc_system::environment::{get_args, get_env, lookup_env};
//!
//! // Get command line arguments
//! let args = get_args();
//! println!("Program: {}", args[0]);
//!
//! // Get environment variable (panics if not set)
//! let home = get_env("HOME");
//!
//! // Get environment variable (returns Option)
//! if let Some(path) = lookup_env("PATH") {
//!     println!("PATH: {}", path);
//! }
//! ```

use std::collections::HashMap;
use std::env;
use std::ffi::OsString;

/// Error type for environment operations
#[derive(Debug, Clone)]
pub struct EnvError {
    /// The variable name that caused the error
    pub var_name: String,
    /// Error message
    pub message: String,
}

impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Environment error for '{}': {}",
            self.var_name, self.message
        )
    }
}

impl std::error::Error for EnvError {}

/// Result type for environment operations
pub type EnvResult<T> = Result<T, EnvError>;

/// Get the command-line arguments
///
/// Returns a vector of strings, where the first element is typically
/// the program name.
///
/// # Example
///
/// ```no_run
/// use bhc_system::environment::get_args;
///
/// let args = get_args();
/// for (i, arg) in args.iter().enumerate() {
///     println!("arg[{}] = {}", i, arg);
/// }
/// ```
pub fn get_args() -> Vec<String> {
    env::args().collect()
}

/// Get the command-line arguments as OsStrings
///
/// This preserves the original encoding of arguments, which may not
/// be valid UTF-8 on some platforms.
pub fn get_args_os() -> Vec<OsString> {
    env::args_os().collect()
}

/// Get the program name (first argument)
pub fn get_program_name() -> String {
    env::args().next().unwrap_or_default()
}

/// Get an environment variable
///
/// # Panics
///
/// Panics if the variable is not set or contains invalid UTF-8.
///
/// # Example
///
/// ```no_run
/// use bhc_system::environment::get_env;
///
/// let home = get_env("HOME");
/// println!("Home directory: {}", home);
/// ```
pub fn get_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("Environment variable '{}' not set", name))
}

/// Look up an environment variable
///
/// Returns `Some(value)` if the variable is set, `None` otherwise.
///
/// # Example
///
/// ```no_run
/// use bhc_system::environment::lookup_env;
///
/// match lookup_env("DEBUG") {
///     Some(val) => println!("DEBUG={}", val),
///     None => println!("DEBUG not set"),
/// }
/// ```
pub fn lookup_env(name: &str) -> Option<String> {
    env::var(name).ok()
}

/// Look up an environment variable, returning a Result
///
/// Returns an error with details if the variable is not set.
pub fn lookup_env_result(name: &str) -> EnvResult<String> {
    env::var(name).map_err(|e| EnvError {
        var_name: name.to_string(),
        message: e.to_string(),
    })
}

/// Set an environment variable
///
/// # Example
///
/// ```no_run
/// use bhc_system::environment::{set_env, get_env};
///
/// set_env("MY_VAR", "my_value");
/// assert_eq!(get_env("MY_VAR"), "my_value");
/// ```
pub fn set_env(name: &str, value: &str) {
    env::set_var(name, value);
}

/// Remove an environment variable
///
/// # Example
///
/// ```no_run
/// use bhc_system::environment::{set_env, unset_env, lookup_env};
///
/// set_env("TEMP_VAR", "value");
/// unset_env("TEMP_VAR");
/// assert!(lookup_env("TEMP_VAR").is_none());
/// ```
pub fn unset_env(name: &str) {
    env::remove_var(name);
}

/// Get all environment variables
///
/// Returns a HashMap of all environment variable names and values.
///
/// # Example
///
/// ```no_run
/// use bhc_system::environment::get_environment;
///
/// let env = get_environment();
/// for (key, value) in &env {
///     println!("{}={}", key, value);
/// }
/// ```
pub fn get_environment() -> HashMap<String, String> {
    env::vars().collect()
}

/// Get the current working directory
pub fn current_dir() -> std::io::Result<String> {
    env::current_dir().map(|p| p.to_string_lossy().to_string())
}

/// Set the current working directory
pub fn set_current_dir(path: &str) -> std::io::Result<()> {
    env::set_current_dir(path)
}

/// Get the user's home directory
///
/// Returns `None` if the home directory cannot be determined.
pub fn home_dir() -> Option<String> {
    // Try HOME on Unix, USERPROFILE on Windows
    lookup_env("HOME").or_else(|| lookup_env("USERPROFILE"))
}

/// Get the temporary directory path
pub fn temp_dir() -> String {
    env::temp_dir().to_string_lossy().to_string()
}

/// Get the executable path
pub fn current_exe() -> std::io::Result<String> {
    env::current_exe().map(|p| p.to_string_lossy().to_string())
}

// FFI exports for BHC runtime

/// Get environment variable (FFI)
#[no_mangle]
pub extern "C" fn bhc_get_env(name: *const i8, out_len: *mut usize) -> *mut u8 {
    use std::ffi::CStr;

    if name.is_null() {
        return std::ptr::null_mut();
    }

    let name = unsafe { CStr::from_ptr(name) };
    let name = match name.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match lookup_env(name) {
        Some(value) => {
            let bytes = value.into_bytes();
            let len = bytes.len();
            let ptr = bytes.leak().as_mut_ptr();
            if !out_len.is_null() {
                unsafe { *out_len = len };
            }
            ptr
        }
        None => std::ptr::null_mut(),
    }
}

/// Set environment variable (FFI)
#[no_mangle]
pub extern "C" fn bhc_set_env(name: *const i8, value: *const i8) -> i32 {
    use std::ffi::CStr;

    if name.is_null() || value.is_null() {
        return -1;
    }

    let name = unsafe { CStr::from_ptr(name) };
    let value = unsafe { CStr::from_ptr(value) };

    let name = match name.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let value = match value.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    set_env(name, value);
    0
}

/// Get argument count (FFI)
#[no_mangle]
pub extern "C" fn bhc_argc() -> i32 {
    env::args().count() as i32
}

/// Get argument by index (FFI)
#[no_mangle]
pub extern "C" fn bhc_argv(index: i32, out_len: *mut usize) -> *mut u8 {
    let args: Vec<String> = env::args().collect();
    if index < 0 || index as usize >= args.len() {
        return std::ptr::null_mut();
    }

    let arg = args[index as usize].clone();
    let bytes = arg.into_bytes();
    let len = bytes.len();
    let ptr = bytes.leak().as_mut_ptr();
    if !out_len.is_null() {
        unsafe { *out_len = len };
    }
    ptr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_args() {
        let args = get_args();
        assert!(!args.is_empty());
    }

    #[test]
    fn test_set_and_get_env() {
        let name = "BHC_TEST_VAR_123";
        let value = "test_value_456";

        set_env(name, value);
        assert_eq!(get_env(name), value);

        unset_env(name);
        assert!(lookup_env(name).is_none());
    }

    #[test]
    fn test_lookup_env_missing() {
        let result = lookup_env("DEFINITELY_NOT_SET_VAR_XYZ_123");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_environment() {
        set_env("BHC_TEST_ENV_1", "value1");
        let env = get_environment();

        assert!(env.contains_key("BHC_TEST_ENV_1"));
        assert_eq!(env.get("BHC_TEST_ENV_1"), Some(&"value1".to_string()));

        unset_env("BHC_TEST_ENV_1");
    }

    #[test]
    fn test_temp_dir() {
        let temp = temp_dir();
        assert!(!temp.is_empty());
    }

    #[test]
    fn test_current_dir() {
        let dir = current_dir().unwrap();
        assert!(!dir.is_empty());
    }

    #[test]
    fn test_home_dir() {
        // This might be None in some test environments
        let _home = home_dir();
    }

    #[test]
    fn test_current_exe() {
        let exe = current_exe().unwrap();
        assert!(!exe.is_empty());
    }

    #[test]
    fn test_program_name() {
        let name = get_program_name();
        // Should at least return something (might be empty in some test runners)
        let _ = name;
    }

    #[test]
    fn test_lookup_env_result() {
        set_env("BHC_TEST_RESULT", "ok");
        assert!(lookup_env_result("BHC_TEST_RESULT").is_ok());

        unset_env("BHC_TEST_RESULT");
        assert!(lookup_env_result("BHC_TEST_RESULT").is_err());
    }
}
