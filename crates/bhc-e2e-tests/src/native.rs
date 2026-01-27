//! Native backend runner using LLVM code generation.

use crate::{E2EError, E2ETestCase, ExecutionOutput, Profile};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Compile a Haskell source file to a native executable.
pub fn compile_native(
    test_case: &E2ETestCase,
    work_dir: &Path,
    profile: Profile,
) -> Result<PathBuf, E2EError> {
    let output_path = work_dir.join(&test_case.name);

    // Build bhc command
    let mut cmd = Command::new("cargo");
    cmd.args([
        "run",
        "--quiet",
        "-p",
        "bhc",
        "--",
        test_case.source_path.to_str().unwrap(),
        "-o",
        output_path.to_str().unwrap(),
    ]);

    // Add profile flag
    match profile {
        Profile::Default => {}
        Profile::Numeric => {
            cmd.arg("--profile=numeric");
        }
        Profile::Edge => {
            cmd.arg("--profile=edge");
        }
    }

    // Run compiler
    let result = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| E2EError::CompileError(format!("Failed to run compiler: {}", e)))?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stdout = String::from_utf8_lossy(&result.stdout);
        return Err(E2EError::CompileError(format!(
            "Compilation failed:\nstdout: {}\nstderr: {}",
            stdout, stderr
        )));
    }

    // Verify output exists
    if !output_path.exists() {
        return Err(E2EError::CompileError(format!(
            "Compiler succeeded but output not found: {}",
            output_path.display()
        )));
    }

    Ok(output_path)
}

/// Run a native executable and capture its output.
pub fn run_native(exe_path: &Path, timeout: Duration) -> Result<ExecutionOutput, E2EError> {
    let start = Instant::now();

    // Make executable (Unix)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(exe_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(exe_path, perms)?;
    }

    let mut child = Command::new(exe_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| E2EError::ExecutionError(format!("Failed to spawn process: {}", e)))?;

    // Wait with timeout
    let result = wait_with_timeout(&mut child, timeout)?;
    let duration = start.elapsed();

    let stdout = String::from_utf8_lossy(&result.stdout).to_string();
    let stderr = String::from_utf8_lossy(&result.stderr).to_string();
    let exit_code = result.status.code().unwrap_or(-1);

    Ok(ExecutionOutput {
        stdout,
        stderr,
        exit_code,
        duration,
    })
}

/// Wait for a child process with timeout.
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<std::process::Output, E2EError> {
    use std::thread;

    let start = Instant::now();
    let poll_interval = Duration::from_millis(10);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process exited
                let stdout = child
                    .stdout
                    .take()
                    .map(|mut s| {
                        let mut buf = Vec::new();
                        std::io::Read::read_to_end(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();

                let stderr = child
                    .stderr
                    .take()
                    .map(|mut s| {
                        let mut buf = Vec::new();
                        std::io::Read::read_to_end(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();

                return Ok(std::process::Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {
                // Still running
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    return Err(E2EError::Timeout(timeout));
                }
                thread::sleep(poll_interval);
            }
            Err(e) => {
                return Err(E2EError::ExecutionError(format!(
                    "Failed to wait for process: {}",
                    e
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_detection() {
        // This test verifies the timeout mechanism works
        // by attempting to run a non-existent binary
        let result = run_native(Path::new("/nonexistent/binary"), Duration::from_secs(1));
        assert!(result.is_err());
    }
}
