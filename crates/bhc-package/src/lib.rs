//! Package management and manifest parsing for BHC.
//!
//! This crate handles BHC package definitions, manifest parsing,
//! dependency resolution, and lockfile management. Packages are the
//! unit of distribution and dependency management in BHC.
//!
//! # Manifest Format
//!
//! BHC uses TOML for package manifests (`bhc.toml`):
//!
//! ```toml
//! [package]
//! name = "my-package"
//! version = "0.1.0"
//! edition = "h26"
//! profile = "default"
//!
//! [dependencies]
//! base = "^1.0"
//! text = { version = "^2.0", features = ["unicode"] }
//!
//! [dev-dependencies]
//! test-framework = "^0.5"
//! ```
//!
//! # Profiles
//!
//! Packages can specify a default compilation profile:
//!
//! - `default` - Standard lazy Haskell
//! - `server` - Server workloads
//! - `numeric` - Numeric computing
//! - `edge` - Minimal runtime

#![warn(missing_docs)]

pub mod commands;
pub mod lockfile;
pub mod registry;
pub mod resolve;

use bhc_session::Profile;
use camino::{Utf8Path, Utf8PathBuf};
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during package operations.
#[derive(Debug, Error)]
pub enum PackageError {
    /// Manifest file not found.
    #[error("manifest not found: {0}")]
    ManifestNotFound(Utf8PathBuf),

    /// Invalid manifest format.
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),

    /// Invalid version specification.
    #[error("invalid version: {0}")]
    InvalidVersion(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// TOML parsing error.
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),
}

/// Result type for package operations.
pub type PackageResult<T> = Result<T, PackageError>;

/// The BHC language edition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Edition {
    /// H26 edition (Haskell 2026).
    #[default]
    #[serde(rename = "h26")]
    H26,
}

impl Edition {
    /// Get the edition name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::H26 => "h26",
        }
    }
}

/// Package metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageMetadata {
    /// Package name.
    pub name: String,
    /// Package version.
    #[serde(with = "version_serde")]
    pub version: Version,
    /// Language edition.
    #[serde(default)]
    pub edition: Edition,
    /// Default compilation profile.
    #[serde(default)]
    pub profile: Profile,
    /// Package description.
    #[serde(default)]
    pub description: Option<String>,
    /// Package license.
    #[serde(default)]
    pub license: Option<String>,
    /// Package authors.
    #[serde(default)]
    pub authors: Vec<String>,
    /// Repository URL.
    #[serde(default)]
    pub repository: Option<String>,
    /// Homepage URL.
    #[serde(default)]
    pub homepage: Option<String>,
    /// Keywords for search.
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Categories.
    #[serde(default)]
    pub categories: Vec<String>,
}

/// Custom serde for semver::Version.
pub(crate) mod version_serde {
    use semver::Version;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(version: &Version, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&version.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Version, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// A dependency specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dependency {
    /// Simple version requirement.
    Simple(String),
    /// Detailed dependency specification.
    Detailed(DetailedDependency),
}

impl Dependency {
    /// Parse the version requirement.
    ///
    /// # Errors
    ///
    /// Returns an error if the version string is invalid.
    pub fn version_req(&self) -> PackageResult<VersionReq> {
        let version_str = match self {
            Self::Simple(s) => s.as_str(),
            Self::Detailed(d) => d.version.as_str(),
        };
        VersionReq::parse(version_str).map_err(|e| PackageError::InvalidVersion(e.to_string()))
    }

    /// Get the features to enable.
    #[must_use]
    pub fn features(&self) -> &[String] {
        match self {
            Self::Simple(_) => &[],
            Self::Detailed(d) => &d.features,
        }
    }

    /// Check if this is an optional dependency.
    #[must_use]
    pub fn is_optional(&self) -> bool {
        match self {
            Self::Simple(_) => false,
            Self::Detailed(d) => d.optional,
        }
    }
}

/// Detailed dependency specification.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DetailedDependency {
    /// Version requirement.
    #[serde(default)]
    pub version: String,
    /// Features to enable.
    #[serde(default)]
    pub features: Vec<String>,
    /// Whether this dependency is optional.
    #[serde(default)]
    pub optional: bool,
    /// Git repository URL.
    #[serde(default)]
    pub git: Option<String>,
    /// Git branch.
    #[serde(default)]
    pub branch: Option<String>,
    /// Git tag.
    #[serde(default)]
    pub tag: Option<String>,
    /// Git revision.
    #[serde(default)]
    pub rev: Option<String>,
    /// Path to local dependency.
    #[serde(default)]
    pub path: Option<Utf8PathBuf>,
}

/// Library configuration.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LibraryConfig {
    /// Exposed modules.
    #[serde(default)]
    pub exposed_modules: Vec<String>,
    /// Source directory.
    #[serde(default = "default_src_dir")]
    pub src_dir: Utf8PathBuf,
}

fn default_src_dir() -> Utf8PathBuf {
    Utf8PathBuf::from("src")
}

/// Executable configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutableConfig {
    /// Executable name.
    pub name: String,
    /// Main module path.
    #[serde(default = "default_main")]
    pub main: String,
    /// Source directory.
    #[serde(default = "default_src_dir")]
    pub src_dir: Utf8PathBuf,
}

fn default_main() -> String {
    "Main".to_string()
}

/// Test configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestConfig {
    /// Test suite name.
    pub name: String,
    /// Main module.
    #[serde(default = "default_test_main")]
    pub main: String,
    /// Source directory.
    #[serde(default = "default_test_dir")]
    pub src_dir: Utf8PathBuf,
}

fn default_test_main() -> String {
    "Test.Main".to_string()
}

fn default_test_dir() -> Utf8PathBuf {
    Utf8PathBuf::from("test")
}

/// Benchmark configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Benchmark suite name.
    pub name: String,
    /// Main module.
    #[serde(default = "default_bench_main")]
    pub main: String,
    /// Source directory.
    #[serde(default = "default_bench_dir")]
    pub src_dir: Utf8PathBuf,
}

fn default_bench_main() -> String {
    "Bench.Main".to_string()
}

fn default_bench_dir() -> Utf8PathBuf {
    Utf8PathBuf::from("bench")
}

/// A complete package manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// Package metadata.
    pub package: PackageMetadata,
    /// Library configuration.
    #[serde(default)]
    pub library: Option<LibraryConfig>,
    /// Executable configurations.
    #[serde(default, rename = "bin")]
    pub executables: Vec<ExecutableConfig>,
    /// Test configurations.
    #[serde(default, rename = "test")]
    pub tests: Vec<TestConfig>,
    /// Benchmark configurations.
    #[serde(default, rename = "benchmark")]
    pub benchmarks: Vec<BenchmarkConfig>,
    /// Dependencies.
    #[serde(default)]
    pub dependencies: HashMap<String, Dependency>,
    /// Development dependencies.
    #[serde(default, rename = "dev-dependencies")]
    pub dev_dependencies: HashMap<String, Dependency>,
    /// Build dependencies.
    #[serde(default, rename = "build-dependencies")]
    pub build_dependencies: HashMap<String, Dependency>,
    /// Feature definitions.
    #[serde(default)]
    pub features: HashMap<String, Vec<String>>,
}

impl Manifest {
    /// Load a manifest from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(path: impl AsRef<Utf8Path>) -> PackageResult<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                PackageError::ManifestNotFound(path.to_path_buf())
            } else {
                PackageError::Io(e)
            }
        })?;
        Self::parse(&content)
    }

    /// Parse a manifest from TOML content.
    ///
    /// # Errors
    ///
    /// Returns an error if the content is not valid TOML or manifest format.
    pub fn parse(content: &str) -> PackageResult<Self> {
        toml::from_str(content).map_err(PackageError::Toml)
    }

    /// Get the package name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.package.name
    }

    /// Get the package version.
    #[must_use]
    pub fn version(&self) -> &Version {
        &self.package.version
    }

    /// Get the compilation profile.
    #[must_use]
    pub fn profile(&self) -> Profile {
        self.package.profile
    }

    /// Check if the package has a library component.
    #[must_use]
    pub fn has_library(&self) -> bool {
        self.library.is_some()
    }

    /// Check if the package has any executables.
    #[must_use]
    pub fn has_executables(&self) -> bool {
        !self.executables.is_empty()
    }
}

/// A resolved package in the dependency graph.
#[derive(Clone, Debug)]
pub struct ResolvedPackage {
    /// Package name.
    pub name: String,
    /// Resolved version.
    pub version: Version,
    /// Source location.
    pub source: PackageSource,
    /// Resolved dependencies.
    pub dependencies: Vec<String>,
    /// Enabled features.
    pub features: Vec<String>,
}

/// Source of a package.
#[derive(Clone, Debug)]
pub enum PackageSource {
    /// From the package registry.
    Registry,
    /// From a git repository.
    Git {
        /// Repository URL.
        url: String,
        /// Revision.
        rev: String,
    },
    /// From a local path.
    Path(Utf8PathBuf),
}

/// Package ID combining name and version.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackageId {
    /// Package name.
    pub name: String,
    /// Package version.
    pub version: Version,
}

impl PackageId {
    /// Create a new package ID.
    #[must_use]
    pub fn new(name: impl Into<String>, version: Version) -> Self {
        Self {
            name: name.into(),
            version,
        }
    }
}

impl std::fmt::Display for PackageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} v{}", self.name, self.version)
    }
}

/// Find the manifest file in a directory or its parents.
///
/// # Errors
///
/// Returns an error if no manifest is found.
pub fn find_manifest(start: impl AsRef<Utf8Path>) -> PackageResult<Utf8PathBuf> {
    let mut current = start.as_ref().to_path_buf();

    loop {
        let manifest_path = current.join("bhc.toml");
        if manifest_path.exists() {
            return Ok(manifest_path);
        }

        match current.parent() {
            Some(parent) => current = parent.to_path_buf(),
            None => {
                return Err(PackageError::ManifestNotFound(
                    start.as_ref().join("bhc.toml"),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MANIFEST: &str = r#"
[package]
name = "test-package"
version = "0.1.0"
edition = "h26"
description = "A test package"

[dependencies]
base = "^1.0"
text = { version = "^2.0", features = ["unicode"] }

[dev-dependencies]
test-framework = "^0.5"
"#;

    #[test]
    fn test_parse_manifest() {
        let manifest = Manifest::parse(SAMPLE_MANIFEST).unwrap();
        assert_eq!(manifest.name(), "test-package");
        assert_eq!(manifest.version().to_string(), "0.1.0");
        assert_eq!(manifest.package.edition, Edition::H26);
    }

    #[test]
    fn test_dependency_parsing() {
        let manifest = Manifest::parse(SAMPLE_MANIFEST).unwrap();

        let base = manifest.dependencies.get("base").unwrap();
        assert!(matches!(base, Dependency::Simple(_)));
        assert_eq!(base.version_req().unwrap().to_string(), "^1.0");

        let text = manifest.dependencies.get("text").unwrap();
        assert_eq!(text.features(), &["unicode"]);
    }

    #[test]
    fn test_package_id() {
        let id = PackageId::new("my-package", Version::new(1, 2, 3));
        assert_eq!(id.to_string(), "my-package v1.2.3");
    }
}
