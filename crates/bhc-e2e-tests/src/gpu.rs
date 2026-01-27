//! GPU backend runner (mock and CUDA modes).

use crate::{E2EError, E2ETestCase, ExecutionOutput, Profile};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Compile a Haskell source file to GPU code.
///
/// In mock mode, this generates PTX but doesn't execute it.
/// In CUDA mode, this creates an executable that runs on GPU hardware.
pub fn compile_gpu(
    test_case: &E2ETestCase,
    work_dir: &Path,
    profile: Profile,
    mock_mode: bool,
) -> Result<PathBuf, E2EError> {
    // GPU compilation requires numeric profile
    if profile != Profile::Numeric {
        return Err(E2EError::CompileError(
            "GPU backend requires Numeric profile".into(),
        ));
    }

    let output_path = if mock_mode {
        work_dir.join(format!("{}.ptx", test_case.name))
    } else {
        work_dir.join(&test_case.name)
    };

    // Build bhc command with GPU target
    let mut cmd = Command::new("cargo");
    cmd.args([
        "run",
        "--quiet",
        "-p",
        "bhc",
        "--",
        test_case.source_path.to_str().unwrap(),
        "--target=cuda",
        "--profile=numeric",
        "-o",
        output_path.to_str().unwrap(),
    ]);

    if mock_mode {
        cmd.arg("--emit=ptx");
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
            "GPU compilation failed:\nstdout: {}\nstderr: {}",
            stdout, stderr
        )));
    }

    // Verify output exists
    if !output_path.exists() {
        return Err(E2EError::CompileError(format!(
            "Compiler succeeded but GPU output not found: {}",
            output_path.display()
        )));
    }

    Ok(output_path)
}

/// Run GPU code.
///
/// In mock mode, this verifies the PTX is valid without executing.
/// In CUDA mode, this runs the actual GPU code.
pub fn run_gpu(
    artifact_path: &Path,
    timeout: Duration,
    mock_mode: bool,
) -> Result<ExecutionOutput, E2EError> {
    let start = Instant::now();

    if mock_mode {
        // Mock mode: validate PTX structure without execution
        run_gpu_mock(artifact_path)
    } else {
        // Real CUDA execution
        #[cfg(feature = "cuda")]
        {
            run_gpu_cuda(artifact_path, timeout)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = timeout;
            Err(E2EError::BackendUnavailable(
                "CUDA feature not enabled. Build with --features cuda".into(),
            ))
        }
    }
    .map(|mut output| {
        output.duration = start.elapsed();
        output
    })
}

/// Mock GPU execution - validates PTX without running on hardware.
fn run_gpu_mock(ptx_path: &Path) -> Result<ExecutionOutput, E2EError> {
    // Read and validate PTX
    let ptx_content = std::fs::read_to_string(ptx_path)
        .map_err(|e| E2EError::ExecutionError(format!("Failed to read PTX: {}", e)))?;

    // Basic PTX validation
    if !ptx_content.contains(".version") {
        return Err(E2EError::ExecutionError(
            "Invalid PTX: missing .version directive".into(),
        ));
    }

    if !ptx_content.contains(".target") {
        return Err(E2EError::ExecutionError(
            "Invalid PTX: missing .target directive".into(),
        ));
    }

    // Check for entry point
    if !ptx_content.contains(".entry") && !ptx_content.contains(".func") {
        return Err(E2EError::ExecutionError(
            "Invalid PTX: no entry point or function defined".into(),
        ));
    }

    // Mock successful execution
    Ok(ExecutionOutput {
        stdout: "[GPU Mock] PTX validation successful\n".to_string(),
        stderr: String::new(),
        exit_code: 0,
        duration: Duration::from_millis(1),
    })
}

/// Real CUDA execution.
#[cfg(feature = "cuda")]
fn run_gpu_cuda(exe_path: &Path, timeout: Duration) -> Result<ExecutionOutput, E2EError> {
    use std::process::Command;

    let mut child = Command::new(exe_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| E2EError::ExecutionError(format!("Failed to spawn GPU process: {}", e)))?;

    // Wait with timeout (similar to native runner)
    let start = Instant::now();
    let poll_interval = Duration::from_millis(10);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
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

                return Ok(ExecutionOutput {
                    stdout: String::from_utf8_lossy(&stdout).to_string(),
                    stderr: String::from_utf8_lossy(&stderr).to_string(),
                    exit_code: status.code().unwrap_or(-1),
                    duration: start.elapsed(),
                });
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    return Err(E2EError::Timeout(timeout));
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => {
                return Err(E2EError::ExecutionError(format!(
                    "Failed to wait for GPU process: {}",
                    e
                )));
            }
        }
    }
}

/// Check if CUDA is available on the system.
pub fn cuda_available() -> bool {
    // Try to find nvcc or check for CUDA library
    Command::new("nvcc")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ptx_validation_valid() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            ".version 7.0\n.target sm_50\n.entry kernel() {{\n  ret;\n}}"
        )
        .unwrap();

        let result = run_gpu_mock(file.path());
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.exit_code, 0);
    }

    #[test]
    fn test_ptx_validation_missing_version() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, ".target sm_50\n.entry kernel() {{\n  ret;\n}}").unwrap();

        let result = run_gpu_mock(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_ptx_validation_missing_target() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, ".version 7.0\n.entry kernel() {{\n  ret;\n}}").unwrap();

        let result = run_gpu_mock(file.path());
        assert!(result.is_err());
    }
}
