//! Test execution engine for E2E tests.

use crate::{Backend, E2EError, E2ETestCase, GoldenComparison, Profile};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Output captured from test execution.
#[derive(Clone, Debug, Default)]
pub struct ExecutionOutput {
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Exit code.
    pub exit_code: i32,
    /// Execution duration.
    pub duration: Duration,
}

/// Result of running an E2E test.
#[derive(Debug)]
pub enum E2EResult {
    /// Test passed.
    Pass {
        /// How long compilation took.
        compile_duration: Duration,
        /// How long execution took.
        exec_duration: Duration,
    },
    /// Compilation failed.
    CompileError(String),
    /// Execution failed.
    ExecutionError(String),
    /// Output did not match expected.
    OutputMismatch {
        expected: String,
        actual: String,
        diff: String,
    },
    /// Exit code did not match expected.
    ExitCodeMismatch { expected: i32, actual: i32 },
    /// Test timed out.
    Timeout(Duration),
    /// Backend not available.
    Skipped(String),
}

impl E2EResult {
    /// Check if the test passed.
    pub fn is_pass(&self) -> bool {
        matches!(self, E2EResult::Pass { .. })
    }

    /// Check if the test was skipped.
    pub fn is_skipped(&self) -> bool {
        matches!(self, E2EResult::Skipped(_))
    }
}

/// Configuration for the E2E runner.
#[derive(Clone, Debug)]
pub struct RunnerConfig {
    /// Target backend.
    pub backend: Backend,
    /// Compilation profile.
    pub profile: Profile,
    /// Working directory for compilation artifacts.
    pub work_dir: PathBuf,
    /// Whether to keep artifacts after test.
    pub keep_artifacts: bool,
    /// Whether GPU tests run in mock mode.
    pub gpu_mock: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Native,
            profile: Profile::Default,
            work_dir: std::env::temp_dir().join("bhc-e2e"),
            keep_artifacts: false,
            gpu_mock: true,
        }
    }
}

/// E2E test runner that compiles and executes Haskell programs.
pub struct E2ERunner {
    config: RunnerConfig,
}

impl E2ERunner {
    /// Create a new runner with the given backend and profile.
    pub fn new(backend: Backend, profile: Profile) -> Self {
        Self {
            config: RunnerConfig {
                backend,
                profile,
                ..Default::default()
            },
        }
    }

    /// Create a runner for native backend with default profile.
    pub fn native(profile: Profile) -> Self {
        Self::new(Backend::Native, profile)
    }

    /// Create a runner for WASM backend.
    pub fn wasm(profile: Profile) -> Self {
        Self::new(Backend::Wasm, profile)
    }

    /// Create a runner for GPU backend in mock mode.
    pub fn gpu_mock(profile: Profile) -> Self {
        let mut runner = Self::new(Backend::Gpu, profile);
        runner.config.gpu_mock = true;
        runner
    }

    /// Create a runner for GPU backend with real CUDA.
    #[cfg(feature = "cuda")]
    pub fn gpu_cuda(profile: Profile) -> Self {
        let mut runner = Self::new(Backend::Gpu, profile);
        runner.config.gpu_mock = false;
        runner
    }

    /// Set custom working directory.
    pub fn with_work_dir(mut self, path: PathBuf) -> Self {
        self.config.work_dir = path;
        self
    }

    /// Keep artifacts after test for debugging.
    pub fn keep_artifacts(mut self) -> Self {
        self.config.keep_artifacts = true;
        self
    }

    /// Run a test from a fixture directory name.
    pub fn run_fixture(&self, fixture_name: &str) -> Result<E2EResult, E2EError> {
        let fixture_path = crate::fixtures_dir().join(fixture_name);
        let test_case = E2ETestCase::from_fixture(&fixture_path)?;
        self.run(&test_case)
    }

    /// Run a test case.
    pub fn run(&self, test_case: &E2ETestCase) -> Result<E2EResult, E2EError> {
        // Check if backend is supported for this test
        if !test_case.backends.contains(&self.config.backend) {
            return Ok(E2EResult::Skipped(format!(
                "Backend {} not enabled for this test",
                self.config.backend
            )));
        }

        // Create work directory
        let test_work_dir = self
            .config
            .work_dir
            .join(&test_case.name)
            .join(self.config.backend.to_string());
        std::fs::create_dir_all(&test_work_dir)?;

        // Compile
        let compile_start = Instant::now();
        let artifact_path = match self.compile(test_case, &test_work_dir) {
            Ok(path) => path,
            Err(e) => return Ok(E2EResult::CompileError(e.to_string())),
        };
        let compile_duration = compile_start.elapsed();

        // Execute
        let exec_start = Instant::now();
        let output = match self.execute(&artifact_path, test_case.timeout()) {
            Ok(output) => output,
            Err(E2EError::Timeout(d)) => return Ok(E2EResult::Timeout(d)),
            Err(e) => return Ok(E2EResult::ExecutionError(e.to_string())),
        };
        let exec_duration = exec_start.elapsed();

        // Verify exit code
        if output.exit_code != test_case.expected_exit_code {
            return Ok(E2EResult::ExitCodeMismatch {
                expected: test_case.expected_exit_code,
                actual: output.exit_code,
            });
        }

        // Verify output
        let comparison = GoldenComparison::new(&test_case.expected_stdout, &output.stdout);
        if !comparison.matches() {
            return Ok(E2EResult::OutputMismatch {
                expected: test_case.expected_stdout.clone(),
                actual: output.stdout,
                diff: comparison.diff(),
            });
        }

        // Clean up artifacts if not keeping them
        if !self.config.keep_artifacts {
            let _ = std::fs::remove_dir_all(&test_work_dir);
        }

        Ok(E2EResult::Pass {
            compile_duration,
            exec_duration,
        })
    }

    /// Compile the test case to an artifact.
    fn compile(&self, test_case: &E2ETestCase, work_dir: &Path) -> Result<PathBuf, E2EError> {
        match self.config.backend {
            Backend::Native => crate::native::compile_native(test_case, work_dir, self.config.profile),
            Backend::Wasm => crate::wasm::compile_wasm(test_case, work_dir, self.config.profile),
            Backend::Gpu => crate::gpu::compile_gpu(test_case, work_dir, self.config.profile, self.config.gpu_mock),
        }
    }

    /// Execute the compiled artifact.
    fn execute(&self, artifact_path: &Path, timeout: Duration) -> Result<ExecutionOutput, E2EError> {
        match self.config.backend {
            Backend::Native => crate::native::run_native(artifact_path, timeout),
            Backend::Wasm => crate::wasm::run_wasm(artifact_path, timeout),
            Backend::Gpu => crate::gpu::run_gpu(artifact_path, timeout, self.config.gpu_mock),
        }
    }
}

/// Format a detailed failure report.
pub fn format_failure_report(
    test_case: &E2ETestCase,
    backend: Backend,
    profile: Profile,
    result: &E2EResult,
    work_dir: Option<&Path>,
) -> String {
    let mut report = String::new();

    report.push_str(&format!(
        "\n=== E2E TEST FAILURE: {} ===\n",
        test_case.name
    ));
    report.push_str(&format!("Backend: {}\n", backend));
    report.push_str(&format!("Profile: {}\n", profile));
    report.push_str(&format!("Source: {}\n", test_case.source_path.display()));
    report.push('\n');

    match result {
        E2EResult::CompileError(msg) => {
            report.push_str("Stage: COMPILE\n");
            report.push_str(&format!("Error: {}\n", msg));
        }
        E2EResult::ExecutionError(msg) => {
            report.push_str("Stage: EXECUTE\n");
            report.push_str(&format!("Error: {}\n", msg));
        }
        E2EResult::OutputMismatch {
            expected,
            actual,
            diff,
        } => {
            report.push_str("Stage: VERIFY\n");
            report.push_str("\nExpected stdout:\n");
            for line in expected.lines() {
                report.push_str(&format!("  {}\n", line));
            }
            report.push_str("\nActual stdout:\n");
            for line in actual.lines() {
                report.push_str(&format!("  {}\n", line));
            }
            report.push_str("\nDiff:\n");
            report.push_str(diff);
        }
        E2EResult::ExitCodeMismatch { expected, actual } => {
            report.push_str("Stage: VERIFY\n");
            report.push_str(&format!(
                "Exit code: expected {}, got {}\n",
                expected, actual
            ));
        }
        E2EResult::Timeout(duration) => {
            report.push_str("Stage: EXECUTE\n");
            report.push_str(&format!("Timeout after {:?}\n", duration));
        }
        E2EResult::Pass { .. } | E2EResult::Skipped(_) => {
            // Not a failure
        }
    }

    if let Some(dir) = work_dir {
        report.push_str(&format!("\nDebug artifacts: {}\n", dir.display()));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_creation() {
        let runner = E2ERunner::native(Profile::Default);
        assert_eq!(runner.config.backend, Backend::Native);
        assert_eq!(runner.config.profile, Profile::Default);
    }

    #[test]
    fn test_result_is_pass() {
        let pass = E2EResult::Pass {
            compile_duration: Duration::from_secs(1),
            exec_duration: Duration::from_millis(100),
        };
        assert!(pass.is_pass());

        let fail = E2EResult::CompileError("error".into());
        assert!(!fail.is_pass());
    }

    #[test]
    fn test_result_is_skipped() {
        let skipped = E2EResult::Skipped("not supported".into());
        assert!(skipped.is_skipped());

        let pass = E2EResult::Pass {
            compile_duration: Duration::from_secs(1),
            exec_duration: Duration::from_millis(100),
        };
        assert!(!pass.is_skipped());
    }
}
