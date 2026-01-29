//! Package management commands.
//!
//! This module provides high-level commands for package operations:
//!
//! - `init` - Create a new package
//! - `add` - Add a dependency
//! - `remove` - Remove a dependency
//! - `update` - Update dependencies
//! - `publish` - Publish to registry
//! - `search` - Search the registry
//! - `info` - Get package information
//!
//! # Example
//!
//! ```ignore
//! use bhc_package::commands::{InitCommand, AddCommand};
//!
//! // Create a new package
//! InitCommand::new("my-package")
//!     .library(true)
//!     .edition(Edition::H26)
//!     .execute()?;
//!
//! // Add a dependency
//! AddCommand::new("text", "^2.0")
//!     .features(vec!["unicode"])
//!     .execute()?;
//! ```

use crate::lockfile::{Lockfile, LOCKFILE_NAME};
use crate::registry::{
    PackageRegistry, PublishDep, PublishMetadata, Registry, RegistryConfig, SearchResult,
};
use crate::resolve::{Resolution, Resolver, ResolverConfig};
use crate::{
    Dependency, DetailedDependency, Edition, ExecutableConfig, LibraryConfig, Manifest,
    PackageError, PackageMetadata,
};
use bhc_session::Profile;
use camino::{Utf8Path, Utf8PathBuf};
use flate2::write::GzEncoder;
use flate2::Compression;
use semver::{Version, VersionReq};
use std::collections::HashMap;
use std::fs;
use tar::Builder;
use thiserror::Error;
use tracing::{debug, info};

/// Command errors.
#[derive(Debug, Error)]
pub enum CommandError {
    /// Package already exists.
    #[error("package already exists at {0}")]
    PackageExists(Utf8PathBuf),

    /// Manifest not found.
    #[error("no manifest found in {0}")]
    NoManifest(Utf8PathBuf),

    /// Dependency already exists.
    #[error("dependency {0} already exists")]
    DependencyExists(String),

    /// Dependency not found.
    #[error("dependency {0} not found")]
    DependencyNotFound(String),

    /// Invalid version requirement.
    #[error("invalid version: {0}")]
    InvalidVersion(String),

    /// Resolution failed.
    #[error("resolution failed: {0}")]
    ResolutionFailed(String),

    /// Publish failed.
    #[error("publish failed: {0}")]
    PublishFailed(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Package error.
    #[error("{0}")]
    Package(#[from] PackageError),

    /// Lockfile error.
    #[error("lockfile error: {0}")]
    Lockfile(#[from] crate::lockfile::LockfileError),

    /// TOML serialization error.
    #[error("TOML error: {0}")]
    Toml(#[from] toml::ser::Error),
}

/// Result type for commands.
pub type CommandResult<T> = Result<T, CommandError>;

/// Initialize a new package.
#[derive(Clone, Debug)]
pub struct InitCommand {
    /// Package name.
    name: String,
    /// Target directory.
    path: Utf8PathBuf,
    /// Create library structure.
    library: bool,
    /// Create executable structure.
    binary: bool,
    /// Language edition.
    edition: Edition,
    /// Default profile.
    profile: Profile,
}

impl InitCommand {
    /// Create a new init command.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: Utf8PathBuf::from("."),
            library: true,
            binary: false,
            edition: Edition::H26,
            profile: Profile::Default,
        }
    }

    /// Set the target directory.
    pub fn path(mut self, path: impl AsRef<Utf8Path>) -> Self {
        self.path = path.as_ref().to_path_buf();
        self
    }

    /// Enable/disable library structure.
    pub fn library(mut self, enabled: bool) -> Self {
        self.library = enabled;
        self
    }

    /// Enable/disable binary structure.
    pub fn binary(mut self, enabled: bool) -> Self {
        self.binary = enabled;
        self
    }

    /// Set the edition.
    pub fn edition(mut self, edition: Edition) -> Self {
        self.edition = edition;
        self
    }

    /// Set the default profile.
    pub fn profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }

    /// Execute the command.
    pub fn execute(self) -> CommandResult<()> {
        let manifest_path = self.path.join("bhc.toml");

        if manifest_path.exists() {
            return Err(CommandError::PackageExists(manifest_path));
        }

        // Create directory structure
        fs::create_dir_all(&self.path)?;

        // Create manifest
        let manifest = self.create_manifest();
        let manifest_toml = toml::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, manifest_toml)?;

        // Create source directories and files
        if self.library {
            let src_dir = self.path.join("src");
            fs::create_dir_all(&src_dir)?;

            let lib_file = src_dir.join("Lib.hs");
            if !lib_file.exists() {
                fs::write(
                    &lib_file,
                    format!(
                        "-- | {}\nmodule {} where\n\n-- Your library code here\n",
                        self.name,
                        to_module_name(&self.name)
                    ),
                )?;
            }
        }

        if self.binary {
            let app_dir = self.path.join("app");
            fs::create_dir_all(&app_dir)?;

            let main_file = app_dir.join("Main.hs");
            if !main_file.exists() {
                fs::write(
                    &main_file,
                    "module Main where\n\nmain :: IO ()\nmain = putStrLn \"Hello, World!\"\n",
                )?;
            }
        }

        // Create .gitignore
        let gitignore_path = self.path.join(".gitignore");
        if !gitignore_path.exists() {
            fs::write(
                &gitignore_path,
                "# BHC build artifacts\n/dist/\n/dist-newstyle/\n\n# Editor files\n*.swp\n*~\n.vscode/\n.idea/\n",
            )?;
        }

        info!("Created package '{}' at {}", self.name, self.path);
        Ok(())
    }

    fn create_manifest(&self) -> Manifest {
        Manifest {
            package: PackageMetadata {
                name: self.name.clone(),
                version: Version::new(0, 1, 0),
                edition: self.edition,
                profile: self.profile,
                description: None,
                license: Some("BSD-3-Clause".to_string()),
                authors: Vec::new(),
                repository: None,
                homepage: None,
                keywords: Vec::new(),
                categories: Vec::new(),
            },
            library: if self.library {
                Some(LibraryConfig {
                    exposed_modules: vec![to_module_name(&self.name)],
                    src_dir: Utf8PathBuf::from("src"),
                })
            } else {
                None
            },
            executables: if self.binary {
                vec![ExecutableConfig {
                    name: self.name.clone(),
                    main: "Main".to_string(),
                    src_dir: Utf8PathBuf::from("app"),
                }]
            } else {
                Vec::new()
            },
            tests: Vec::new(),
            benchmarks: Vec::new(),
            dependencies: HashMap::new(),
            dev_dependencies: HashMap::new(),
            build_dependencies: HashMap::new(),
            features: HashMap::new(),
        }
    }
}

/// Add a dependency to the package.
#[derive(Clone, Debug)]
pub struct AddCommand {
    /// Dependency name.
    name: String,
    /// Version requirement.
    version: String,
    /// Features to enable.
    features: Vec<String>,
    /// Optional dependency.
    optional: bool,
    /// Dev dependency.
    dev: bool,
    /// Build dependency.
    build: bool,
    /// Git source.
    git: Option<String>,
    /// Git branch.
    branch: Option<String>,
    /// Local path.
    path: Option<Utf8PathBuf>,
}

impl AddCommand {
    /// Create a new add command.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            features: Vec::new(),
            optional: false,
            dev: false,
            build: false,
            git: None,
            branch: None,
            path: None,
        }
    }

    /// Add features.
    pub fn features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }

    /// Make optional.
    pub fn optional(mut self, optional: bool) -> Self {
        self.optional = optional;
        self
    }

    /// Add as dev dependency.
    pub fn dev(mut self, dev: bool) -> Self {
        self.dev = dev;
        self
    }

    /// Add as build dependency.
    pub fn build(mut self, build: bool) -> Self {
        self.build = build;
        self
    }

    /// Use git source.
    pub fn git(mut self, url: impl Into<String>) -> Self {
        self.git = Some(url.into());
        self
    }

    /// Git branch.
    pub fn branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = Some(branch.into());
        self
    }

    /// Local path.
    pub fn path(mut self, path: impl AsRef<Utf8Path>) -> Self {
        self.path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Execute the command.
    pub fn execute(self) -> CommandResult<()> {
        let manifest_path = crate::find_manifest(".")?;
        let content = fs::read_to_string(&manifest_path)?;
        let mut manifest: Manifest = Manifest::parse(&content)?;

        // Check if dependency already exists
        let deps = if self.dev {
            &mut manifest.dev_dependencies
        } else if self.build {
            &mut manifest.build_dependencies
        } else {
            &mut manifest.dependencies
        };

        if deps.contains_key(&self.name) {
            return Err(CommandError::DependencyExists(self.name));
        }

        // Validate version requirement
        if self.git.is_none() && self.path.is_none() {
            VersionReq::parse(&self.version)
                .map_err(|e| CommandError::InvalidVersion(e.to_string()))?;
        }

        // Create dependency spec
        let dep = if self.features.is_empty()
            && !self.optional
            && self.git.is_none()
            && self.path.is_none()
        {
            Dependency::Simple(self.version.clone())
        } else {
            Dependency::Detailed(DetailedDependency {
                version: self.version.clone(),
                features: self.features.clone(),
                optional: self.optional,
                git: self.git.clone(),
                branch: self.branch.clone(),
                tag: None,
                rev: None,
                path: self.path.clone(),
            })
        };

        deps.insert(self.name.clone(), dep);

        // Write back manifest
        let manifest_toml = toml::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, manifest_toml)?;

        let dep_type = if self.dev {
            "dev dependency"
        } else if self.build {
            "build dependency"
        } else {
            "dependency"
        };

        info!("Added {} '{}' v{}", dep_type, self.name, self.version);
        Ok(())
    }
}

/// Remove a dependency.
#[derive(Clone, Debug)]
pub struct RemoveCommand {
    /// Dependency name.
    name: String,
    /// Dev dependency.
    dev: bool,
    /// Build dependency.
    build: bool,
}

impl RemoveCommand {
    /// Create a new remove command.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dev: false,
            build: false,
        }
    }

    /// Remove dev dependency.
    pub fn dev(mut self, dev: bool) -> Self {
        self.dev = dev;
        self
    }

    /// Remove build dependency.
    pub fn build(mut self, build: bool) -> Self {
        self.build = build;
        self
    }

    /// Execute the command.
    pub fn execute(self) -> CommandResult<()> {
        let manifest_path = crate::find_manifest(".")?;
        let content = fs::read_to_string(&manifest_path)?;
        let mut manifest: Manifest = Manifest::parse(&content)?;

        let deps = if self.dev {
            &mut manifest.dev_dependencies
        } else if self.build {
            &mut manifest.build_dependencies
        } else {
            &mut manifest.dependencies
        };

        if deps.remove(&self.name).is_none() {
            return Err(CommandError::DependencyNotFound(self.name));
        }

        let manifest_toml = toml::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, manifest_toml)?;

        info!("Removed dependency '{}'", self.name);
        Ok(())
    }
}

/// Update dependencies.
#[derive(Clone, Debug)]
pub struct UpdateCommand {
    /// Specific packages to update (empty = all).
    packages: Vec<String>,
    /// Whether to update aggressively.
    aggressive: bool,
    /// Dry run (don't write lockfile).
    dry_run: bool,
}

impl UpdateCommand {
    /// Create a new update command.
    pub fn new() -> Self {
        Self {
            packages: Vec::new(),
            aggressive: false,
            dry_run: false,
        }
    }

    /// Update specific packages.
    pub fn packages(mut self, packages: Vec<String>) -> Self {
        self.packages = packages;
        self
    }

    /// Aggressive update (allow semver-incompatible).
    pub fn aggressive(mut self, aggressive: bool) -> Self {
        self.aggressive = aggressive;
        self
    }

    /// Dry run.
    pub fn dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Execute the command.
    pub fn execute(self) -> CommandResult<Resolution> {
        let manifest_path = crate::find_manifest(".")?;
        let manifest = Manifest::load(&manifest_path)?;

        let lockfile_path = manifest_path.parent().unwrap().join(LOCKFILE_NAME);
        let lockfile = if lockfile_path.exists() {
            Some(Lockfile::load(&lockfile_path).ok())
        } else {
            None
        }
        .flatten();

        // Create registry
        let registry = Registry::new(RegistryConfig::default())
            .map_err(|e| CommandError::ResolutionFailed(e.to_string()))?;

        // Configure resolver
        let config = ResolverConfig {
            update: self.packages.is_empty(),
            update_packages: self.packages.clone(),
            ..Default::default()
        };

        let mut resolver = Resolver::with_config(&registry, config);
        if let Some(lf) = lockfile {
            resolver = resolver.with_lockfile(lf);
        }

        // Resolve
        let resolution = resolver
            .resolve(&manifest)
            .map_err(|e| CommandError::ResolutionFailed(e.to_string()))?;

        // Write lockfile
        if !self.dry_run {
            let lockfile = resolution.to_lockfile();
            lockfile.save(&lockfile_path)?;
            info!("Updated {} packages", resolution.len());
        } else {
            info!("Would update {} packages (dry run)", resolution.len());
        }

        Ok(resolution)
    }
}

impl Default for UpdateCommand {
    fn default() -> Self {
        Self::new()
    }
}

/// Publish package to registry.
#[derive(Clone, Debug)]
pub struct PublishCommand {
    /// Allow publishing with uncommitted changes.
    allow_dirty: bool,
    /// Dry run (validate but don't publish).
    dry_run: bool,
    /// Registry to publish to.
    registry: Option<String>,
    /// Authentication token.
    token: Option<String>,
}

impl PublishCommand {
    /// Create a new publish command.
    pub fn new() -> Self {
        Self {
            allow_dirty: false,
            dry_run: false,
            registry: None,
            token: None,
        }
    }

    /// Allow publishing with uncommitted changes.
    pub fn allow_dirty(mut self, allow: bool) -> Self {
        self.allow_dirty = allow;
        self
    }

    /// Dry run.
    pub fn dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Target registry.
    pub fn registry(mut self, url: impl Into<String>) -> Self {
        self.registry = Some(url.into());
        self
    }

    /// Authentication token.
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Execute the command.
    pub fn execute(self) -> CommandResult<()> {
        let manifest_path = crate::find_manifest(".")?;
        let manifest = Manifest::load(&manifest_path)?;
        let package_dir = manifest_path.parent().unwrap();

        // Check for uncommitted changes
        if !self.allow_dirty {
            if is_git_dirty(package_dir) {
                return Err(CommandError::PublishFailed(
                    "uncommitted changes. Use --allow-dirty to override".to_string(),
                ));
            }
        }

        // Build tarball
        let tarball = self.build_tarball(package_dir, &manifest)?;
        info!("Built tarball ({} bytes)", tarball.len());

        // Create publish metadata
        let metadata = self.create_metadata(&manifest)?;

        if self.dry_run {
            info!(
                "Would publish {} v{} (dry run)",
                manifest.name(),
                manifest.version()
            );
            return Ok(());
        }

        // Publish to registry
        let mut config = RegistryConfig::default();
        if let Some(ref url) = self.registry {
            config.api_url = url.clone();
        }
        if let Some(ref token) = self.token {
            config.token = Some(token.clone());
        }

        let registry =
            Registry::new(config).map_err(|e| CommandError::PublishFailed(e.to_string()))?;

        registry
            .publish(&tarball, &metadata)
            .map_err(|e| CommandError::PublishFailed(e.to_string()))?;

        info!("Published {} v{}", manifest.name(), manifest.version());
        Ok(())
    }

    fn build_tarball(
        &self,
        dir: impl AsRef<Utf8Path>,
        manifest: &Manifest,
    ) -> CommandResult<Vec<u8>> {
        let dir = dir.as_ref();
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut archive = Builder::new(encoder);

        // Get list of files to include
        let files = self.collect_files(dir)?;

        // Add files to archive
        let prefix = format!("{}-{}", manifest.name(), manifest.version());
        for file in &files {
            let rel_path = file.strip_prefix(dir).unwrap_or(file);
            let archive_path = format!("{}/{}", prefix, rel_path);

            debug!("Adding to tarball: {}", archive_path);

            let mut f = fs::File::open(file.as_std_path())?;
            archive.append_file(archive_path, &mut f)?;
        }

        let encoder = archive.into_inner()?;
        let tarball = encoder.finish()?;
        Ok(tarball)
    }

    fn collect_files(&self, dir: impl AsRef<Utf8Path>) -> CommandResult<Vec<Utf8PathBuf>> {
        let dir = dir.as_ref();
        let mut files = Vec::new();

        fn walk_dir(dir: &Utf8Path, files: &mut Vec<Utf8PathBuf>) -> std::io::Result<()> {
            for entry in fs::read_dir(dir.as_std_path())? {
                let entry = entry?;
                let path = Utf8PathBuf::try_from(entry.path()).unwrap();
                let name = path.file_name().unwrap_or("");

                // Skip hidden files and directories
                if name.starts_with('.') {
                    continue;
                }

                // Skip build artifacts
                if name == "dist" || name == "dist-newstyle" || name == "target" {
                    continue;
                }

                if entry.file_type()?.is_dir() {
                    walk_dir(&path, files)?;
                } else {
                    files.push(path);
                }
            }
            Ok(())
        }

        walk_dir(dir, &mut files)?;
        Ok(files)
    }

    fn create_metadata(&self, manifest: &Manifest) -> CommandResult<PublishMetadata> {
        let deps: Vec<PublishDep> = manifest
            .dependencies
            .iter()
            .map(|(name, dep)| {
                let version_req = match dep {
                    Dependency::Simple(v) => v.clone(),
                    Dependency::Detailed(d) => d.version.clone(),
                };
                let features = dep.features().to_vec();
                let optional = dep.is_optional();

                PublishDep {
                    name: name.clone(),
                    version_req,
                    features,
                    optional,
                    target: None,
                    kind: "normal".to_string(),
                }
            })
            .collect();

        Ok(PublishMetadata {
            name: manifest.name().to_string(),
            vers: manifest.version().to_string(),
            deps,
            features: manifest.features.clone(),
            authors: manifest.package.authors.clone(),
            description: manifest.package.description.clone(),
            documentation: None,
            homepage: manifest.package.homepage.clone(),
            readme: None, // Could read README.md
            keywords: manifest.package.keywords.clone(),
            categories: manifest.package.categories.clone(),
            license: manifest.package.license.clone(),
            repository: manifest.package.repository.clone(),
        })
    }
}

impl Default for PublishCommand {
    fn default() -> Self {
        Self::new()
    }
}

/// Search the package registry.
#[derive(Clone, Debug)]
pub struct SearchCommand {
    /// Search query.
    query: String,
    /// Maximum results.
    limit: usize,
}

impl SearchCommand {
    /// Create a new search command.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: 10,
        }
    }

    /// Set result limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Execute the command.
    pub fn execute(self) -> CommandResult<Vec<SearchResult>> {
        let registry = Registry::new(RegistryConfig::default())
            .map_err(|e| CommandError::ResolutionFailed(e.to_string()))?;

        let results = registry
            .search(&self.query)
            .map_err(|e| CommandError::ResolutionFailed(e.to_string()))?;

        Ok(results.into_iter().take(self.limit).collect())
    }
}

/// Get information about a package.
#[derive(Clone, Debug)]
pub struct InfoCommand {
    /// Package name.
    name: String,
    /// Specific version (None = latest).
    version: Option<Version>,
}

impl InfoCommand {
    /// Create a new info command.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
        }
    }

    /// Get info for specific version.
    pub fn version(mut self, version: Version) -> Self {
        self.version = Some(version);
        self
    }

    /// Execute the command.
    pub fn execute(self) -> CommandResult<PackageInfo> {
        let registry = Registry::new(RegistryConfig::default())
            .map_err(|e| CommandError::ResolutionFailed(e.to_string()))?;

        let index = registry
            .get_package(&self.name)
            .map_err(|e| CommandError::ResolutionFailed(e.to_string()))?;

        let version_info = if let Some(ref v) = self.version {
            index
                .get_version(v)
                .ok_or_else(|| CommandError::ResolutionFailed(format!("version {} not found", v)))?
        } else {
            index.latest().ok_or_else(|| {
                CommandError::ResolutionFailed("no versions available".to_string())
            })?
        };

        Ok(PackageInfo {
            name: index.name.clone(),
            version: version_info.version.clone(),
            yanked: version_info.yanked,
            dependencies: version_info
                .dependencies
                .iter()
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect(),
            available_versions: index.versions.iter().map(|v| v.version.clone()).collect(),
        })
    }
}

/// Package information.
#[derive(Clone, Debug)]
pub struct PackageInfo {
    /// Package name.
    pub name: String,
    /// Version.
    pub version: Version,
    /// Whether yanked.
    pub yanked: bool,
    /// Dependencies.
    pub dependencies: HashMap<String, String>,
    /// All available versions.
    pub available_versions: Vec<Version>,
}

/// Convert a package name to a module name.
fn to_module_name(name: &str) -> String {
    name.split(|c| c == '-' || c == '_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

/// Check if git working directory has uncommitted changes.
fn is_git_dirty(dir: impl AsRef<Utf8Path>) -> bool {
    use std::process::Command;

    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(dir.as_ref().as_std_path())
        .output();

    match status {
        Ok(output) => !output.stdout.is_empty(),
        Err(_) => false, // If git isn't available, assume not dirty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_module_name() {
        assert_eq!(to_module_name("my-package"), "MyPackage");
        assert_eq!(to_module_name("text"), "Text");
        assert_eq!(to_module_name("foo_bar"), "FooBar");
        assert_eq!(to_module_name("a-b-c"), "ABC");
    }
}
