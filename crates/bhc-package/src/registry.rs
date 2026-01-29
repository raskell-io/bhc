//! Package registry client for BHC.
//!
//! This module provides access to package registries for downloading
//! and publishing packages.
//!
//! # Registry Format
//!
//! The BHC registry is a git-based index similar to Cargo's crates.io.
//! Each package has an index file containing version information:
//!
//! ```json
//! {"name":"base","vers":"1.0.0","deps":[{"name":"ghc-prim","req":"^0.1"}],"cksum":"sha256:..."}
//! {"name":"base","vers":"1.0.1","deps":[{"name":"ghc-prim","req":"^0.1"}],"cksum":"sha256:..."}
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use bhc_package::registry::{Registry, RegistryConfig};
//!
//! let config = RegistryConfig::default();
//! let registry = Registry::new(config)?;
//!
//! // Get package index
//! let index = registry.get_package("base")?;
//!
//! // Download package
//! let tarball = registry.fetch_package("base", &Version::new(1, 0, 0))?;
//! ```

use camino::{Utf8Path, Utf8PathBuf};
use flate2::read::GzDecoder;
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use tar::Archive;
use thiserror::Error;
use tracing::{debug, info};

/// Default registry URL.
pub const DEFAULT_REGISTRY_URL: &str = "https://registry.bhc.raskell.io";

/// Default registry index URL.
pub const DEFAULT_INDEX_URL: &str = "https://index.bhc.raskell.io";

/// Registry errors.
#[derive(Debug, Error)]
pub enum RegistryError {
    /// Package not found.
    #[error("package not found: {0}")]
    PackageNotFound(String),

    /// Version not found.
    #[error("version {version} not found for package {package}")]
    VersionNotFound {
        /// Package name.
        package: String,
        /// Requested version.
        version: Version,
    },

    /// Package was yanked.
    #[error("package {package} v{version} was yanked: {reason}")]
    PackageYanked {
        /// Package name.
        package: String,
        /// Version.
        version: Version,
        /// Yank reason.
        reason: String,
    },

    /// Checksum mismatch.
    #[error("checksum mismatch for {package}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// Package name.
        package: String,
        /// Expected checksum.
        expected: String,
        /// Actual checksum.
        actual: String,
    },

    /// Network error.
    #[error("network error: {0}")]
    Network(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid index format.
    #[error("invalid index: {0}")]
    InvalidIndex(String),

    /// Authentication required.
    #[error("authentication required for {0}")]
    AuthRequired(String),

    /// Rate limited.
    #[error("rate limited, retry after {0} seconds")]
    RateLimited(u64),
}

/// Result type for registry operations.
pub type RegistryResult<T> = Result<T, RegistryError>;

/// Registry configuration.
#[derive(Clone, Debug)]
pub struct RegistryConfig {
    /// Registry API URL.
    pub api_url: String,
    /// Index URL (for sparse index or git).
    pub index_url: String,
    /// Local cache directory.
    pub cache_dir: Utf8PathBuf,
    /// Authentication token.
    pub token: Option<String>,
    /// Request timeout in seconds.
    pub timeout: u64,
    /// Whether to verify checksums.
    pub verify_checksums: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        let cache_dir = dirs::cache_dir()
            .map(|p| Utf8PathBuf::try_from(p).ok())
            .flatten()
            .unwrap_or_else(|| Utf8PathBuf::from(".cache"))
            .join("bhc")
            .join("registry");

        Self {
            api_url: DEFAULT_REGISTRY_URL.to_string(),
            index_url: DEFAULT_INDEX_URL.to_string(),
            cache_dir,
            token: None,
            timeout: 30,
            verify_checksums: true,
        }
    }
}

impl RegistryConfig {
    /// Create configuration with custom URLs.
    pub fn with_urls(api_url: impl Into<String>, index_url: impl Into<String>) -> Self {
        Self {
            api_url: api_url.into(),
            index_url: index_url.into(),
            ..Default::default()
        }
    }

    /// Set authentication token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Set cache directory.
    pub fn with_cache_dir(mut self, path: impl AsRef<Utf8Path>) -> Self {
        self.cache_dir = path.as_ref().to_path_buf();
        self
    }
}

/// Package registry trait.
pub trait PackageRegistry {
    /// Get package index information.
    fn get_package(&self, name: &str) -> RegistryResult<PackageIndex>;

    /// Fetch package tarball.
    fn fetch_package(&self, name: &str, version: &Version) -> RegistryResult<Vec<u8>>;

    /// Get registry configuration.
    fn config(&self) -> &RegistryConfig;
}

/// Package index information.
#[derive(Clone, Debug)]
pub struct PackageIndex {
    /// Package name.
    pub name: String,
    /// Available versions.
    pub versions: Vec<PackageVersionInfo>,
}

impl PackageIndex {
    /// Get the latest non-yanked version.
    pub fn latest(&self) -> Option<&PackageVersionInfo> {
        self.versions
            .iter()
            .filter(|v| !v.yanked)
            .max_by(|a, b| a.version.cmp(&b.version))
    }

    /// Get a specific version.
    pub fn get_version(&self, version: &Version) -> Option<&PackageVersionInfo> {
        self.versions.iter().find(|v| &v.version == version)
    }

    /// Get versions matching a requirement.
    pub fn versions_matching(&self, req: &VersionReq) -> Vec<&PackageVersionInfo> {
        self.versions
            .iter()
            .filter(|v| !v.yanked && req.matches(&v.version))
            .collect()
    }
}

/// Information about a package version.
#[derive(Clone, Debug)]
pub struct PackageVersionInfo {
    /// Version number.
    pub version: Version,
    /// Whether this version is yanked.
    pub yanked: bool,
    /// Dependencies.
    pub dependencies: HashMap<String, VersionReq>,
    /// Content checksum.
    pub checksum: Option<String>,
}

/// Index entry as stored in the registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct IndexEntry {
    name: String,
    #[serde(rename = "vers")]
    version: String,
    #[serde(default)]
    yanked: bool,
    #[serde(default)]
    deps: Vec<IndexDep>,
    #[serde(rename = "cksum")]
    checksum: Option<String>,
}

/// Dependency entry in index.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct IndexDep {
    name: String,
    req: String,
    #[serde(default)]
    features: Vec<String>,
    #[serde(default)]
    optional: bool,
}

/// HTTP-based package registry.
pub struct Registry {
    config: RegistryConfig,
}

impl Registry {
    /// Create a new registry client.
    pub fn new(config: RegistryConfig) -> RegistryResult<Self> {
        // Ensure cache directory exists
        fs::create_dir_all(&config.cache_dir)?;
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_registry() -> RegistryResult<Self> {
        Self::new(RegistryConfig::default())
    }

    /// Get the cache path for a package's index.
    fn index_cache_path(&self, name: &str) -> Utf8PathBuf {
        let prefix = get_index_prefix(name);
        self.config.cache_dir.join("index").join(prefix).join(name)
    }

    /// Get the cache path for a package tarball.
    fn tarball_cache_path(&self, name: &str, version: &Version) -> Utf8PathBuf {
        self.config
            .cache_dir
            .join("crates")
            .join(name)
            .join(format!("{name}-{version}.tar.gz"))
    }

    /// Fetch index from registry.
    fn fetch_index(&self, name: &str) -> RegistryResult<String> {
        let prefix = get_index_prefix(name);
        let url = format!("{}/{}/{}", self.config.index_url, prefix, name);

        debug!("Fetching index: {}", url);

        let mut request =
            ureq::get(&url).timeout(std::time::Duration::from_secs(self.config.timeout));

        if let Some(ref token) = self.config.token {
            request = request.set("Authorization", &format!("Bearer {}", token));
        }

        let response = request
            .call()
            .map_err(|e| RegistryError::Network(e.to_string()))?;

        if response.status() == 404 {
            return Err(RegistryError::PackageNotFound(name.to_string()));
        }

        if response.status() == 401 || response.status() == 403 {
            return Err(RegistryError::AuthRequired(name.to_string()));
        }

        if response.status() == 429 {
            let retry_after = response
                .header("Retry-After")
                .and_then(|h| h.parse().ok())
                .unwrap_or(60);
            return Err(RegistryError::RateLimited(retry_after));
        }

        response
            .into_string()
            .map_err(|e| RegistryError::Network(e.to_string()))
    }

    /// Parse index entries (NDJSON format).
    fn parse_index(&self, content: &str, name: &str) -> RegistryResult<PackageIndex> {
        let mut versions = Vec::new();

        for line in content.lines() {
            if line.is_empty() {
                continue;
            }

            let entry: IndexEntry = serde_json::from_str(line)
                .map_err(|e| RegistryError::InvalidIndex(e.to_string()))?;

            if entry.name != name {
                continue;
            }

            let version = Version::parse(&entry.version)
                .map_err(|e| RegistryError::InvalidIndex(format!("invalid version: {}", e)))?;

            let dependencies: HashMap<String, VersionReq> = entry
                .deps
                .into_iter()
                .filter_map(|d| VersionReq::parse(&d.req).ok().map(|req| (d.name, req)))
                .collect();

            versions.push(PackageVersionInfo {
                version,
                yanked: entry.yanked,
                dependencies,
                checksum: entry.checksum,
            });
        }

        if versions.is_empty() {
            return Err(RegistryError::PackageNotFound(name.to_string()));
        }

        Ok(PackageIndex {
            name: name.to_string(),
            versions,
        })
    }

    /// Fetch tarball from registry.
    fn fetch_tarball(&self, name: &str, version: &Version) -> RegistryResult<Vec<u8>> {
        let url = format!(
            "{}/api/v1/crates/{}/{}/download",
            self.config.api_url, name, version
        );

        debug!("Downloading: {}", url);

        let mut request =
            ureq::get(&url).timeout(std::time::Duration::from_secs(self.config.timeout));

        if let Some(ref token) = self.config.token {
            request = request.set("Authorization", &format!("Bearer {}", token));
        }

        let response = request
            .call()
            .map_err(|e| RegistryError::Network(e.to_string()))?;

        if response.status() == 404 {
            return Err(RegistryError::VersionNotFound {
                package: name.to_string(),
                version: version.clone(),
            });
        }

        let mut data = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut data)
            .map_err(|e| RegistryError::Network(e.to_string()))?;

        Ok(data)
    }

    /// Verify tarball checksum.
    fn verify_checksum(&self, data: &[u8], expected: &str) -> bool {
        let actual = compute_sha256(data);
        actual == expected
    }

    /// Extract tarball to directory.
    pub fn extract_package(
        &self,
        name: &str,
        version: &Version,
        target_dir: impl AsRef<Utf8Path>,
    ) -> RegistryResult<()> {
        let tarball = self.fetch_package(name, version)?;
        let decoder = GzDecoder::new(&tarball[..]);
        let mut archive = Archive::new(decoder);

        let target_dir = target_dir.as_ref();
        fs::create_dir_all(target_dir)?;

        archive.unpack(target_dir.as_std_path())?;

        info!("Extracted {} v{} to {}", name, version, target_dir);
        Ok(())
    }

    /// Search for packages.
    pub fn search(&self, query: &str) -> RegistryResult<Vec<SearchResult>> {
        let url = format!(
            "{}/api/v1/crates?q={}",
            self.config.api_url,
            urlencoding::encode(query)
        );

        let response = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(self.config.timeout))
            .call()
            .map_err(|e| RegistryError::Network(e.to_string()))?;

        let results: SearchResponse = response.into_json()?;
        Ok(results.crates)
    }

    /// Publish a package.
    pub fn publish(&self, tarball: &[u8], metadata: &PublishMetadata) -> RegistryResult<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| RegistryError::AuthRequired("publish".to_string()))?;

        let url = format!("{}/api/v1/crates/new", self.config.api_url);

        let metadata_json = serde_json::to_vec(metadata)?;

        // Format: u32 length of JSON + JSON + u32 length of tarball + tarball
        let mut body = Vec::new();
        body.extend(&(metadata_json.len() as u32).to_le_bytes());
        body.extend(&metadata_json);
        body.extend(&(tarball.len() as u32).to_le_bytes());
        body.extend(tarball);

        let response = ureq::put(&url)
            .set("Authorization", &format!("Bearer {}", token))
            .set("Content-Type", "application/octet-stream")
            .timeout(std::time::Duration::from_secs(self.config.timeout * 2))
            .send_bytes(&body)
            .map_err(|e| RegistryError::Network(e.to_string()))?;

        if response.status() != 200 {
            let error = response
                .into_string()
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(RegistryError::Network(error));
        }

        info!("Published {} v{}", metadata.name, metadata.vers);
        Ok(())
    }

    /// Yank a version.
    pub fn yank(&self, name: &str, version: &Version) -> RegistryResult<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| RegistryError::AuthRequired("yank".to_string()))?;

        let url = format!(
            "{}/api/v1/crates/{}/{}/yank",
            self.config.api_url, name, version
        );

        let response = ureq::delete(&url)
            .set("Authorization", &format!("Bearer {}", token))
            .timeout(std::time::Duration::from_secs(self.config.timeout))
            .call()
            .map_err(|e| RegistryError::Network(e.to_string()))?;

        if response.status() != 200 {
            let error = response
                .into_string()
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(RegistryError::Network(error));
        }

        info!("Yanked {} v{}", name, version);
        Ok(())
    }

    /// Unyank a version.
    pub fn unyank(&self, name: &str, version: &Version) -> RegistryResult<()> {
        let token = self
            .config
            .token
            .as_ref()
            .ok_or_else(|| RegistryError::AuthRequired("unyank".to_string()))?;

        let url = format!(
            "{}/api/v1/crates/{}/{}/unyank",
            self.config.api_url, name, version
        );

        let response = ureq::put(&url)
            .set("Authorization", &format!("Bearer {}", token))
            .timeout(std::time::Duration::from_secs(self.config.timeout))
            .call()
            .map_err(|e| RegistryError::Network(e.to_string()))?;

        if response.status() != 200 {
            let error = response
                .into_string()
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(RegistryError::Network(error));
        }

        info!("Unyanked {} v{}", name, version);
        Ok(())
    }
}

impl PackageRegistry for Registry {
    fn get_package(&self, name: &str) -> RegistryResult<PackageIndex> {
        // Try cache first
        let cache_path = self.index_cache_path(name);
        if cache_path.exists() {
            debug!("Using cached index for {}", name);
            let content = fs::read_to_string(&cache_path)?;
            return self.parse_index(&content, name);
        }

        // Fetch from registry
        let content = self.fetch_index(name)?;

        // Cache the result
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cache_path, &content)?;

        self.parse_index(&content, name)
    }

    fn fetch_package(&self, name: &str, version: &Version) -> RegistryResult<Vec<u8>> {
        // Try cache first
        let cache_path = self.tarball_cache_path(name, version);
        if cache_path.exists() {
            debug!("Using cached tarball for {} v{}", name, version);
            return Ok(fs::read(&cache_path)?);
        }

        // Get checksum from index
        let index = self.get_package(name)?;
        let version_info =
            index
                .get_version(version)
                .ok_or_else(|| RegistryError::VersionNotFound {
                    package: name.to_string(),
                    version: version.clone(),
                })?;

        if version_info.yanked {
            return Err(RegistryError::PackageYanked {
                package: name.to_string(),
                version: version.clone(),
                reason: "This version has been yanked".to_string(),
            });
        }

        // Fetch tarball
        let data = self.fetch_tarball(name, version)?;

        // Verify checksum
        if self.config.verify_checksums {
            if let Some(ref expected) = version_info.checksum {
                if !self.verify_checksum(&data, expected) {
                    return Err(RegistryError::ChecksumMismatch {
                        package: name.to_string(),
                        expected: expected.clone(),
                        actual: compute_sha256(&data),
                    });
                }
            }
        }

        // Cache the result
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cache_path, &data)?;

        Ok(data)
    }

    fn config(&self) -> &RegistryConfig {
        &self.config
    }
}

/// Search result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Package name.
    pub name: String,
    /// Latest version.
    pub max_version: String,
    /// Description.
    pub description: Option<String>,
    /// Download count.
    pub downloads: u64,
}

/// Search response from API.
#[derive(Debug, Deserialize)]
struct SearchResponse {
    crates: Vec<SearchResult>,
}

/// Metadata for publishing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublishMetadata {
    /// Package name.
    pub name: String,
    /// Version.
    pub vers: String,
    /// Dependencies.
    #[serde(default)]
    pub deps: Vec<PublishDep>,
    /// Features.
    #[serde(default)]
    pub features: HashMap<String, Vec<String>>,
    /// Authors.
    #[serde(default)]
    pub authors: Vec<String>,
    /// Description.
    pub description: Option<String>,
    /// Documentation URL.
    pub documentation: Option<String>,
    /// Homepage URL.
    pub homepage: Option<String>,
    /// README content.
    pub readme: Option<String>,
    /// Keywords.
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Categories.
    #[serde(default)]
    pub categories: Vec<String>,
    /// License.
    pub license: Option<String>,
    /// Repository URL.
    pub repository: Option<String>,
}

/// Dependency for publishing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublishDep {
    /// Dependency name.
    pub name: String,
    /// Version requirement.
    pub version_req: String,
    /// Features.
    #[serde(default)]
    pub features: Vec<String>,
    /// Optional.
    #[serde(default)]
    pub optional: bool,
    /// Target.
    pub target: Option<String>,
    /// Kind.
    #[serde(default = "default_dep_kind")]
    pub kind: String,
}

fn default_dep_kind() -> String {
    "normal".to_string()
}

/// Get index prefix for a package name.
/// This matches the crates.io index structure:
/// - 1 char: "1/"
/// - 2 chars: "2/"
/// - 3 chars: "3/{first_char}/"
/// - 4+ chars: "{first_two}/{next_two}/"
fn get_index_prefix(name: &str) -> String {
    let name = name.to_lowercase();
    match name.len() {
        1 => "1".to_string(),
        2 => "2".to_string(),
        3 => format!("3/{}", &name[0..1]),
        _ => format!("{}/{}", &name[0..2], &name[2..4]),
    }
}

/// Compute SHA256 checksum.
pub fn compute_sha256(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    format!("sha256:{}", hex::encode(hash))
}

/// Offline registry using cached data.
pub struct OfflineRegistry {
    config: RegistryConfig,
}

impl OfflineRegistry {
    /// Create an offline registry.
    pub fn new(config: RegistryConfig) -> Self {
        Self { config }
    }
}

impl PackageRegistry for OfflineRegistry {
    fn get_package(&self, name: &str) -> RegistryResult<PackageIndex> {
        let prefix = get_index_prefix(name);
        let cache_path = self.config.cache_dir.join("index").join(prefix).join(name);

        if !cache_path.exists() {
            return Err(RegistryError::PackageNotFound(name.to_string()));
        }

        let content = fs::read_to_string(&cache_path)?;

        // Parse NDJSON
        let mut versions = Vec::new();
        for line in content.lines() {
            if line.is_empty() {
                continue;
            }

            let entry: IndexEntry = serde_json::from_str(line)
                .map_err(|e| RegistryError::InvalidIndex(e.to_string()))?;

            if entry.name != name {
                continue;
            }

            let version = Version::parse(&entry.version)
                .map_err(|e| RegistryError::InvalidIndex(format!("invalid version: {}", e)))?;

            let dependencies: HashMap<String, VersionReq> = entry
                .deps
                .into_iter()
                .filter_map(|d| VersionReq::parse(&d.req).ok().map(|req| (d.name, req)))
                .collect();

            versions.push(PackageVersionInfo {
                version,
                yanked: entry.yanked,
                dependencies,
                checksum: entry.checksum,
            });
        }

        if versions.is_empty() {
            return Err(RegistryError::PackageNotFound(name.to_string()));
        }

        Ok(PackageIndex {
            name: name.to_string(),
            versions,
        })
    }

    fn fetch_package(&self, name: &str, version: &Version) -> RegistryResult<Vec<u8>> {
        let cache_path = self
            .config
            .cache_dir
            .join("crates")
            .join(name)
            .join(format!("{name}-{version}.tar.gz"));

        if !cache_path.exists() {
            return Err(RegistryError::VersionNotFound {
                package: name.to_string(),
                version: version.clone(),
            });
        }

        Ok(fs::read(&cache_path)?)
    }

    fn config(&self) -> &RegistryConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_prefix() {
        assert_eq!(get_index_prefix("a"), "1");
        assert_eq!(get_index_prefix("ab"), "2");
        assert_eq!(get_index_prefix("abc"), "3/a");
        assert_eq!(get_index_prefix("abcd"), "ab/cd");
        assert_eq!(get_index_prefix("base"), "ba/se");
        assert_eq!(get_index_prefix("text"), "te/xt");
    }

    #[test]
    fn test_checksum() {
        let data = b"hello world";
        let checksum = compute_sha256(data);
        assert!(checksum.starts_with("sha256:"));
        assert_eq!(checksum.len(), 7 + 64); // "sha256:" + 64 hex chars
    }

    #[test]
    fn test_parse_index_entry() {
        let entry = r#"{"name":"test","vers":"1.0.0","deps":[{"name":"base","req":"^1.0"}],"cksum":"sha256:abc"}"#;
        let parsed: IndexEntry = serde_json::from_str(entry).unwrap();
        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.version, "1.0.0");
        assert_eq!(parsed.deps.len(), 1);
    }
}
