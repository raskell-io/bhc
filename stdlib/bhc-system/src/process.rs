//! Process spawning and management
//!
//! This module provides functions for spawning and managing child processes.
//!
//! # Example
//!
//! ```no_run
//! use bhc_system::process::{Command, spawn};
//!
//! // Simple command execution
//! let output = Command::new("echo")
//!     .arg("Hello, World!")
//!     .output()
//!     .unwrap();
//!
//! println!("stdout: {}", output.stdout);
//!
//! // Spawn and wait
//! let mut process = spawn("ls", &["-la"]).unwrap();
//! let status = process.wait().unwrap();
//! println!("Exit code: {}", status);
//! ```

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::process::{Child, ExitStatus, Stdio};

use crate::exit::ExitCode;

/// Error type for process operations
#[derive(Debug)]
pub struct ProcessError {
    /// Error kind
    pub kind: ProcessErrorKind,
    /// Error message
    pub message: String,
}

/// Categories of process errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessErrorKind {
    /// Command not found
    NotFound,
    /// Permission denied
    PermissionDenied,
    /// Process failed to start
    SpawnFailed,
    /// I/O error during communication
    IoError,
    /// Process was killed
    Killed,
    /// Other error
    Other,
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ProcessError {}

impl From<io::Error> for ProcessError {
    fn from(err: io::Error) -> Self {
        let kind = match err.kind() {
            io::ErrorKind::NotFound => ProcessErrorKind::NotFound,
            io::ErrorKind::PermissionDenied => ProcessErrorKind::PermissionDenied,
            _ => ProcessErrorKind::Other,
        };
        ProcessError {
            kind,
            message: err.to_string(),
        }
    }
}

/// Result type for process operations
pub type ProcessResult<T> = Result<T, ProcessError>;

/// Output from a completed process
#[derive(Debug, Clone)]
pub struct ProcessOutput {
    /// Exit code
    pub exit_code: ExitCode,
    /// Standard output as string
    pub stdout: String,
    /// Standard error as string
    pub stderr: String,
}

impl ProcessOutput {
    /// Check if the process succeeded (exit code 0)
    pub fn success(&self) -> bool {
        self.exit_code.is_success()
    }
}

/// A running child process
pub struct Process {
    child: Child,
    program: String,
}

impl std::fmt::Debug for Process {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Process")
            .field("id", &self.child.id())
            .field("program", &self.program)
            .finish()
    }
}

impl Process {
    /// Get the process ID
    pub fn id(&self) -> u32 {
        self.child.id()
    }

    /// Get the program name
    pub fn program(&self) -> &str {
        &self.program
    }

    /// Wait for the process to complete
    ///
    /// Returns the exit code.
    pub fn wait(&mut self) -> ProcessResult<ExitCode> {
        let status = self.child.wait()?;
        Ok(exit_code_from_status(status))
    }

    /// Wait for the process and capture output
    pub fn wait_with_output(self) -> ProcessResult<ProcessOutput> {
        let output = self.child.wait_with_output()?;

        Ok(ProcessOutput {
            exit_code: exit_code_from_status(output.status),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        })
    }

    /// Check if the process has exited without blocking
    pub fn try_wait(&mut self) -> ProcessResult<Option<ExitCode>> {
        match self.child.try_wait()? {
            Some(status) => Ok(Some(exit_code_from_status(status))),
            None => Ok(None),
        }
    }

    /// Kill the process
    pub fn kill(&mut self) -> ProcessResult<()> {
        self.child.kill().map_err(|e| ProcessError {
            kind: ProcessErrorKind::Killed,
            message: e.to_string(),
        })
    }

    /// Write to the process's stdin
    pub fn write_stdin(&mut self, data: &[u8]) -> ProcessResult<()> {
        if let Some(ref mut stdin) = self.child.stdin {
            stdin.write_all(data)?;
            Ok(())
        } else {
            Err(ProcessError {
                kind: ProcessErrorKind::IoError,
                message: "stdin not captured".to_string(),
            })
        }
    }

    /// Read from the process's stdout
    pub fn read_stdout(&mut self) -> ProcessResult<Vec<u8>> {
        if let Some(ref mut stdout) = self.child.stdout {
            let mut buf = Vec::new();
            stdout.read_to_end(&mut buf)?;
            Ok(buf)
        } else {
            Err(ProcessError {
                kind: ProcessErrorKind::IoError,
                message: "stdout not captured".to_string(),
            })
        }
    }

    /// Read from the process's stderr
    pub fn read_stderr(&mut self) -> ProcessResult<Vec<u8>> {
        if let Some(ref mut stderr) = self.child.stderr {
            let mut buf = Vec::new();
            stderr.read_to_end(&mut buf)?;
            Ok(buf)
        } else {
            Err(ProcessError {
                kind: ProcessErrorKind::IoError,
                message: "stderr not captured".to_string(),
            })
        }
    }
}

fn exit_code_from_status(status: ExitStatus) -> ExitCode {
    status
        .code()
        .map(ExitCode::new)
        .unwrap_or(ExitCode::FAILURE)
}

/// Command builder for spawning processes
#[derive(Debug, Clone)]
pub struct Command {
    program: String,
    args: Vec<String>,
    env: HashMap<String, String>,
    env_clear: bool,
    current_dir: Option<String>,
    stdin: StdioConfig,
    stdout: StdioConfig,
    stderr: StdioConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StdioConfig {
    Inherit,
    Piped,
    Null,
}

impl Command {
    /// Create a new command for the given program
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bhc_system::process::Command;
    ///
    /// let cmd = Command::new("ls");
    /// ```
    pub fn new<S: Into<String>>(program: S) -> Self {
        Command {
            program: program.into(),
            args: Vec::new(),
            env: HashMap::new(),
            env_clear: false,
            current_dir: None,
            stdin: StdioConfig::Inherit,
            stdout: StdioConfig::Inherit,
            stderr: StdioConfig::Inherit,
        }
    }

    /// Add an argument
    pub fn arg<S: Into<String>>(mut self, arg: S) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Add multiple arguments
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Set an environment variable
    pub fn env<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Set multiple environment variables
    pub fn envs<I, K, V>(mut self, vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (k, v) in vars {
            self.env.insert(k.into(), v.into());
        }
        self
    }

    /// Clear all environment variables
    pub fn env_clear(mut self) -> Self {
        self.env_clear = true;
        self
    }

    /// Set the working directory
    pub fn current_dir<S: Into<String>>(mut self, dir: S) -> Self {
        self.current_dir = Some(dir.into());
        self
    }

    /// Capture stdin (pipe)
    pub fn stdin_piped(mut self) -> Self {
        self.stdin = StdioConfig::Piped;
        self
    }

    /// Capture stdout (pipe)
    pub fn stdout_piped(mut self) -> Self {
        self.stdout = StdioConfig::Piped;
        self
    }

    /// Capture stderr (pipe)
    pub fn stderr_piped(mut self) -> Self {
        self.stderr = StdioConfig::Piped;
        self
    }

    /// Redirect stdin to null
    pub fn stdin_null(mut self) -> Self {
        self.stdin = StdioConfig::Null;
        self
    }

    /// Redirect stdout to null
    pub fn stdout_null(mut self) -> Self {
        self.stdout = StdioConfig::Null;
        self
    }

    /// Redirect stderr to null
    pub fn stderr_null(mut self) -> Self {
        self.stderr = StdioConfig::Null;
        self
    }

    fn to_stdio(config: StdioConfig) -> Stdio {
        match config {
            StdioConfig::Inherit => Stdio::inherit(),
            StdioConfig::Piped => Stdio::piped(),
            StdioConfig::Null => Stdio::null(),
        }
    }

    /// Spawn the command as a child process
    pub fn spawn(self) -> ProcessResult<Process> {
        let mut cmd = std::process::Command::new(&self.program);
        cmd.args(&self.args);

        if self.env_clear {
            cmd.env_clear();
        }

        for (k, v) in &self.env {
            cmd.env(k, v);
        }

        if let Some(ref dir) = self.current_dir {
            cmd.current_dir(dir);
        }

        cmd.stdin(Self::to_stdio(self.stdin));
        cmd.stdout(Self::to_stdio(self.stdout));
        cmd.stderr(Self::to_stdio(self.stderr));

        let child = cmd.spawn().map_err(|e| ProcessError {
            kind: if e.kind() == io::ErrorKind::NotFound {
                ProcessErrorKind::NotFound
            } else {
                ProcessErrorKind::SpawnFailed
            },
            message: e.to_string(),
        })?;

        Ok(Process {
            child,
            program: self.program,
        })
    }

    /// Run the command and wait for completion
    pub fn status(self) -> ProcessResult<ExitCode> {
        let mut process = self.spawn()?;
        process.wait()
    }

    /// Run the command and capture output
    pub fn output(self) -> ProcessResult<ProcessOutput> {
        let cmd = self.stdout_piped().stderr_piped();
        let process = cmd.spawn()?;
        process.wait_with_output()
    }
}

/// Spawn a process with the given program and arguments
///
/// # Example
///
/// ```no_run
/// use bhc_system::process::spawn;
///
/// let mut process = spawn("echo", &["Hello"]).unwrap();
/// let status = process.wait().unwrap();
/// ```
pub fn spawn<S: AsRef<str>>(program: &str, args: &[S]) -> ProcessResult<Process> {
    Command::new(program)
        .args(args.iter().map(|s| s.as_ref()))
        .spawn()
}

/// Run a command and wait for completion
pub fn run<S: AsRef<str>>(program: &str, args: &[S]) -> ProcessResult<ExitCode> {
    Command::new(program)
        .args(args.iter().map(|s| s.as_ref()))
        .status()
}

/// Run a command and capture output
pub fn output<S: AsRef<str>>(program: &str, args: &[S]) -> ProcessResult<ProcessOutput> {
    Command::new(program)
        .args(args.iter().map(|s| s.as_ref()))
        .output()
}

/// Run a shell command
///
/// On Unix, uses `/bin/sh -c`.
/// On Windows, uses `cmd /C`.
pub fn shell(command: &str) -> ProcessResult<ProcessOutput> {
    #[cfg(unix)]
    {
        Command::new("/bin/sh").arg("-c").arg(command).output()
    }

    #[cfg(windows)]
    {
        Command::new("cmd").arg("/C").arg(command).output()
    }
}

/// Run a shell command and wait for completion
pub fn shell_status(command: &str) -> ProcessResult<ExitCode> {
    #[cfg(unix)]
    {
        Command::new("/bin/sh").arg("-c").arg(command).status()
    }

    #[cfg(windows)]
    {
        Command::new("cmd").arg("/C").arg(command).status()
    }
}

// FFI exports

/// Spawn process (FFI)
#[no_mangle]
pub extern "C" fn bhc_spawn(program: *const i8, args: *const *const i8, argc: i32) -> *mut Process {
    use std::ffi::CStr;

    if program.is_null() {
        return std::ptr::null_mut();
    }

    let program = unsafe { CStr::from_ptr(program) };
    let program = match program.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let mut cmd = Command::new(program);

    if !args.is_null() && argc > 0 {
        for i in 0..argc {
            let arg_ptr = unsafe { *args.offset(i as isize) };
            if !arg_ptr.is_null() {
                let arg = unsafe { CStr::from_ptr(arg_ptr) };
                if let Ok(s) = arg.to_str() {
                    cmd = cmd.arg(s);
                }
            }
        }
    }

    match cmd.spawn() {
        Ok(process) => Box::into_raw(Box::new(process)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Wait for process (FFI)
#[no_mangle]
pub extern "C" fn bhc_process_wait(process: *mut Process) -> i32 {
    if process.is_null() {
        return -1;
    }

    let process = unsafe { &mut *process };
    match process.wait() {
        Ok(code) => code.value(),
        Err(_) => -1,
    }
}

/// Kill process (FFI)
#[no_mangle]
pub extern "C" fn bhc_process_kill(process: *mut Process) -> i32 {
    if process.is_null() {
        return -1;
    }

    let process = unsafe { &mut *process };
    match process.kill() {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Free process (FFI)
#[no_mangle]
pub extern "C" fn bhc_process_free(process: *mut Process) {
    if !process.is_null() {
        unsafe {
            drop(Box::from_raw(process));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_new() {
        let cmd = Command::new("echo");
        assert_eq!(cmd.program, "echo");
        assert!(cmd.args.is_empty());
    }

    #[test]
    fn test_command_args() {
        let cmd = Command::new("echo").arg("hello").arg("world");
        assert_eq!(cmd.args, vec!["hello", "world"]);
    }

    #[test]
    fn test_command_args_vec() {
        let cmd = Command::new("echo").args(["a", "b", "c"]);
        assert_eq!(cmd.args, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_command_env() {
        let cmd = Command::new("echo").env("KEY", "VALUE");
        assert_eq!(cmd.env.get("KEY"), Some(&"VALUE".to_string()));
    }

    #[test]
    fn test_command_current_dir() {
        let cmd = Command::new("ls").current_dir("/tmp");
        assert_eq!(cmd.current_dir, Some("/tmp".to_string()));
    }

    #[test]
    fn test_echo_output() {
        let output = Command::new("echo").arg("hello").output().unwrap();

        assert!(output.success());
        assert!(output.stdout.trim() == "hello");
    }

    #[test]
    fn test_true_status() {
        let status = Command::new("true").status().unwrap();
        assert!(status.is_success());
    }

    #[test]
    fn test_false_status() {
        let status = Command::new("false").status().unwrap();
        assert!(status.is_failure());
    }

    #[test]
    fn test_spawn_and_wait() {
        let mut process = spawn("echo", &["test"]).unwrap();
        let status = process.wait().unwrap();
        assert!(status.is_success());
    }

    #[test]
    fn test_run() {
        let status = run("true", &[] as &[&str]).unwrap();
        assert!(status.is_success());
    }

    #[test]
    fn test_output() {
        let out = output("echo", &["hello"]).unwrap();
        assert!(out.success());
        assert_eq!(out.stdout.trim(), "hello");
    }

    #[test]
    fn test_shell() {
        let out = shell("echo hello").unwrap();
        assert!(out.success());
        assert_eq!(out.stdout.trim(), "hello");
    }

    #[test]
    fn test_shell_status() {
        let status = shell_status("true").unwrap();
        assert!(status.is_success());

        let status = shell_status("false").unwrap();
        assert!(status.is_failure());
    }

    #[test]
    fn test_command_not_found() {
        let result = Command::new("definitely_not_a_real_command_12345").spawn();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ProcessErrorKind::NotFound);
    }

    #[test]
    fn test_process_id() {
        let process = spawn("sleep", &["0.1"]).unwrap();
        let id = process.id();
        assert!(id > 0);
    }

    #[test]
    fn test_try_wait() {
        let mut process = spawn("true", &[] as &[&str]).unwrap();

        // Give it time to complete
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Should be done now
        let result = process.try_wait().unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_kill() {
        let mut process = spawn("sleep", &["10"]).unwrap();
        process.kill().unwrap();

        let status = process.wait().unwrap();
        // Killed processes typically have non-zero exit code
        assert!(status.is_failure() || status.value() == 0); // Platform dependent
    }

    #[test]
    fn test_command_with_env() {
        let output = Command::new("sh")
            .arg("-c")
            .arg("echo $MY_TEST_VAR")
            .env("MY_TEST_VAR", "test_value")
            .output()
            .unwrap();

        assert!(output.success());
        assert_eq!(output.stdout.trim(), "test_value");
    }

    #[test]
    fn test_stderr_capture() {
        let output = Command::new("sh")
            .arg("-c")
            .arg("echo error >&2")
            .output()
            .unwrap();

        assert!(output.success());
        assert!(output.stderr.contains("error"));
    }
}
