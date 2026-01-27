//! End-to-end testing framework for the BHC compiler.
//!
//! This crate provides infrastructure for compiling Haskell programs,
//! executing them on various backends (Native, WASM, GPU), and verifying
//! correct output through golden file comparison.
//!
//! # Test Tiers
//!
//! Tests are organized into tiers of increasing complexity:
//!
//! - **Tier 1 (Simple)**: Basic smoke tests - hello world, arithmetic, let bindings
//! - **Tier 2 (Functions)**: Lambdas, recursion, closures
//! - **Tier 3 (IO)**: IO operations and print statements
//! - **Tier 4 (Fusion)**: Numeric profile optimizations (map/map, sum/map)
//! - **Tier 5 (Benchmark)**: Performance tests, GPU kernels
//!
//! # Example
//!
//! ```no_run
//! use bhc_e2e_tests::{E2ERunner, Backend, Profile};
//!
//! let runner = E2ERunner::new(Backend::Native, Profile::Default);
//! let result = runner.run_fixture("tier1_simple/hello").unwrap();
//! assert!(result.is_pass());
//! ```

mod golden;
mod gpu;
mod native;
mod runner;
mod wasm;

pub use golden::{GoldenComparison, GoldenError};
pub use runner::{format_failure_report, E2EResult, E2ERunner, ExecutionOutput};

// Re-export backend-specific functions
pub use gpu::cuda_available;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

/// Backend targets for E2E tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Native executable via LLVM.
    Native,
    /// WebAssembly via wasmtime.
    Wasm,
    /// GPU via CUDA (mock mode available).
    Gpu,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Native => write!(f, "native"),
            Backend::Wasm => write!(f, "wasm"),
            Backend::Gpu => write!(f, "gpu"),
        }
    }
}

/// Compilation profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    /// Default lazy evaluation.
    Default,
    /// Strict numeric operations with fusion.
    Numeric,
    /// Minimal runtime for WASM/edge deployment.
    Edge,
}

impl std::fmt::Display for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Profile::Default => write!(f, "default"),
            Profile::Numeric => write!(f, "numeric"),
            Profile::Edge => write!(f, "edge"),
        }
    }
}

/// A test case specification loaded from `test.toml`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct E2ETestCase {
    /// Test name (derived from directory name).
    pub name: String,

    /// Path to source file.
    pub source_path: PathBuf,

    /// Backends this test should run on.
    #[serde(default = "default_backends")]
    pub backends: Vec<Backend>,

    /// Compilation profile.
    #[serde(default)]
    pub profile: Profile,

    /// Expected stdout content.
    pub expected_stdout: String,

    /// Expected exit code.
    #[serde(default)]
    pub expected_exit_code: i32,

    /// Timeout in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_backends() -> Vec<Backend> {
    vec![Backend::Native, Backend::Wasm]
}

fn default_timeout() -> u64 {
    30
}

impl Default for Profile {
    fn default() -> Self {
        Profile::Default
    }
}

impl E2ETestCase {
    /// Load a test case from a fixture directory.
    ///
    /// The directory must contain:
    /// - `main.hs`: The Haskell source file
    /// - `expected.txt`: Expected stdout output
    /// - `test.toml` (optional): Test configuration
    pub fn from_fixture(fixture_path: &std::path::Path) -> Result<Self, E2EError> {
        let name = fixture_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| E2EError::InvalidFixture("Invalid fixture path".into()))?
            .to_string();

        let source_path = fixture_path.join("main.hs");
        if !source_path.exists() {
            return Err(E2EError::InvalidFixture(format!(
                "Missing main.hs in fixture: {}",
                fixture_path.display()
            )));
        }

        let expected_path = fixture_path.join("expected.txt");
        let expected_stdout = if expected_path.exists() {
            std::fs::read_to_string(&expected_path).map_err(|e| {
                E2EError::InvalidFixture(format!("Failed to read expected.txt: {}", e))
            })?
        } else {
            String::new()
        };

        // Load optional test.toml
        let config_path = fixture_path.join("test.toml");
        let (backends, profile, expected_exit_code, timeout_secs) = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
                E2EError::InvalidFixture(format!("Failed to read test.toml: {}", e))
            })?;
            let config: TestConfig = toml::from_str(&config_str).map_err(|e| {
                E2EError::InvalidFixture(format!("Failed to parse test.toml: {}", e))
            })?;
            (
                config.test.backends.unwrap_or_else(default_backends),
                config.test.profile.unwrap_or_default(),
                config.test.expected_exit.unwrap_or(0),
                config.test.timeout_secs.unwrap_or(default_timeout()),
            )
        } else {
            (default_backends(), Profile::Default, 0, default_timeout())
        };

        Ok(Self {
            name,
            source_path,
            backends,
            profile,
            expected_stdout,
            expected_exit_code,
            timeout_secs,
        })
    }

    /// Get the timeout as a Duration.
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_secs)
    }
}

/// Configuration file format for test.toml.
#[derive(Debug, Deserialize)]
struct TestConfig {
    test: TestConfigInner,
}

#[derive(Debug, Deserialize)]
struct TestConfigInner {
    #[serde(default)]
    backends: Option<Vec<Backend>>,
    #[serde(default)]
    profile: Option<Profile>,
    #[serde(default)]
    expected_exit: Option<i32>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

/// Errors that can occur during E2E testing.
#[derive(Debug, Error)]
pub enum E2EError {
    /// Invalid fixture directory or missing files.
    #[error("Invalid fixture: {0}")]
    InvalidFixture(String),

    /// Compilation failed.
    #[error("Compilation error: {0}")]
    CompileError(String),

    /// Execution failed.
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Output did not match expected.
    #[error("Output mismatch:\n  expected: {expected:?}\n  actual: {actual:?}")]
    OutputMismatch { expected: String, actual: String },

    /// Exit code did not match expected.
    #[error("Exit code mismatch: expected {expected}, got {actual}")]
    ExitCodeMismatch { expected: i32, actual: i32 },

    /// Test timed out.
    #[error("Test timed out after {0:?}")]
    Timeout(Duration),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Backend not available.
    #[error("Backend not available: {0}")]
    BackendUnavailable(String),
}

/// Get the fixtures directory path.
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Discover all test fixtures in a tier directory.
pub fn discover_fixtures(tier: &str) -> Result<Vec<E2ETestCase>, E2EError> {
    let tier_path = fixtures_dir().join(tier);
    if !tier_path.exists() {
        return Ok(Vec::new());
    }

    let mut fixtures = Vec::new();
    for entry in std::fs::read_dir(&tier_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            match E2ETestCase::from_fixture(&path) {
                Ok(fixture) => fixtures.push(fixture),
                Err(e) => {
                    eprintln!("Warning: skipping fixture {:?}: {}", path, e);
                }
            }
        }
    }

    // Sort by name for deterministic order
    fixtures.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(fixtures)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_display() {
        assert_eq!(Backend::Native.to_string(), "native");
        assert_eq!(Backend::Wasm.to_string(), "wasm");
        assert_eq!(Backend::Gpu.to_string(), "gpu");
    }

    #[test]
    fn test_profile_display() {
        assert_eq!(Profile::Default.to_string(), "default");
        assert_eq!(Profile::Numeric.to_string(), "numeric");
        assert_eq!(Profile::Edge.to_string(), "edge");
    }

    #[test]
    fn test_default_backends() {
        let backends = default_backends();
        assert!(backends.contains(&Backend::Native));
        assert!(backends.contains(&Backend::Wasm));
    }
}
