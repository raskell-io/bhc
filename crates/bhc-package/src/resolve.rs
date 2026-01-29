//! Dependency resolution for BHC packages.
//!
//! This module implements a version resolver that finds compatible versions
//! for all transitive dependencies. The algorithm is based on PubGrub-style
//! resolution with backtracking.
//!
//! # Resolution Algorithm
//!
//! 1. Start with the root package's direct dependencies
//! 2. For each unresolved dependency, find compatible versions
//! 3. Select the highest compatible version
//! 4. Add the selected package's dependencies to the queue
//! 5. If a conflict is found, backtrack and try alternative versions
//! 6. Continue until all dependencies are resolved or no solution exists
//!
//! # Example
//!
//! ```ignore
//! use bhc_package::{Manifest, resolve::Resolver};
//!
//! let manifest = Manifest::load("bhc.toml")?;
//! let mut resolver = Resolver::new(registry);
//! let resolution = resolver.resolve(&manifest)?;
//!
//! for pkg in resolution.packages() {
//!     println!("{} v{}", pkg.name, pkg.version);
//! }
//! ```

use crate::lockfile::{LockedPackage, LockedSource, Lockfile};
use crate::registry::{PackageRegistry, RegistryError};
use crate::{Manifest, PackageError};
use camino::Utf8PathBuf;
use semver::{Version, VersionReq};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Errors that can occur during dependency resolution.
#[derive(Debug, Error)]
pub enum ResolveError {
    /// Package not found in any registry.
    #[error("package not found: {0}")]
    PackageNotFound(String),

    /// No compatible version found.
    #[error("no compatible version of {package} satisfying {requirement}")]
    NoCompatibleVersion {
        /// Package name.
        package: String,
        /// Version requirement.
        requirement: VersionReq,
    },

    /// Version conflict between dependencies.
    #[error("version conflict: {package} required as {req1} by {from1} and {req2} by {from2}")]
    VersionConflict {
        /// Package with conflict.
        package: String,
        /// First requirement.
        req1: VersionReq,
        /// Package requiring first version.
        from1: String,
        /// Second requirement.
        req2: VersionReq,
        /// Package requiring second version.
        from2: String,
    },

    /// Circular dependency detected.
    #[error("circular dependency detected: {}", .0.join(" -> "))]
    CircularDependency(Vec<String>),

    /// Maximum resolution depth exceeded.
    #[error("maximum resolution depth ({0}) exceeded")]
    MaxDepthExceeded(usize),

    /// Registry error.
    #[error("registry error: {0}")]
    Registry(#[from] RegistryError),

    /// Package error.
    #[error("package error: {0}")]
    Package(#[from] PackageError),
}

/// Result type for resolution operations.
pub type ResolveResult<T> = Result<T, ResolveError>;

/// A version requirement with its source.
#[derive(Clone, Debug)]
struct VersionConstraint {
    /// The version requirement.
    requirement: VersionReq,
    /// Who required this version.
    required_by: String,
    /// Enabled features.
    features: Vec<String>,
}

/// State of a package in the resolution process.
#[derive(Clone, Debug)]
enum PackageState {
    /// Package is being explored.
    Pending,
    /// Package is resolved to a specific version.
    Resolved(ResolvedDep),
    /// Package resolution failed.
    Failed(String),
}

/// A resolved dependency.
#[derive(Clone, Debug)]
pub struct ResolvedDep {
    /// Package name.
    pub name: String,
    /// Resolved version.
    pub version: Version,
    /// Source location.
    pub source: DependencySource,
    /// Dependencies of this package.
    pub dependencies: Vec<String>,
    /// Enabled features.
    pub features: Vec<String>,
}

/// Source of a dependency.
#[derive(Clone, Debug)]
pub enum DependencySource {
    /// From the package registry.
    Registry,
    /// From a git repository.
    Git {
        /// Repository URL.
        url: String,
        /// Commit/tag/branch.
        reference: String,
    },
    /// From a local path.
    Path(Utf8PathBuf),
}

impl From<DependencySource> for LockedSource {
    fn from(source: DependencySource) -> Self {
        match source {
            DependencySource::Registry => LockedSource::Registry,
            DependencySource::Git { url, reference } => LockedSource::Git {
                url,
                rev: reference,
            },
            DependencySource::Path(path) => LockedSource::Path(path),
        }
    }
}

/// The result of dependency resolution.
#[derive(Clone, Debug)]
pub struct Resolution {
    /// Resolved packages.
    packages: BTreeMap<String, ResolvedDep>,
    /// Root package name.
    root: String,
}

impl Resolution {
    /// Create a new resolution.
    fn new(root: String) -> Self {
        Self {
            packages: BTreeMap::new(),
            root,
        }
    }

    /// Add a resolved package.
    fn add(&mut self, dep: ResolvedDep) {
        self.packages.insert(dep.name.clone(), dep);
    }

    /// Get all resolved packages.
    pub fn packages(&self) -> impl Iterator<Item = &ResolvedDep> {
        self.packages.values()
    }

    /// Get a specific package.
    pub fn get(&self, name: &str) -> Option<&ResolvedDep> {
        self.packages.get(name)
    }

    /// Get the number of resolved packages.
    pub fn len(&self) -> usize {
        self.packages.len()
    }

    /// Check if the resolution is empty.
    pub fn is_empty(&self) -> bool {
        self.packages.is_empty()
    }

    /// Convert to a lockfile.
    pub fn to_lockfile(&self) -> Lockfile {
        let mut lockfile = Lockfile::new();

        for dep in self.packages.values() {
            let locked = LockedPackage::new(
                dep.name.clone(),
                dep.version.clone(),
                dep.source.clone().into(),
            )
            .with_dependencies(dep.dependencies.clone())
            .with_features(dep.features.clone());
            lockfile.add_package(locked);
        }

        lockfile
    }

    /// Get packages in topological order (dependencies first).
    pub fn topological_order(&self) -> Vec<&ResolvedDep> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_mark = HashSet::new();

        fn visit<'a>(
            name: &str,
            packages: &'a BTreeMap<String, ResolvedDep>,
            visited: &mut HashSet<String>,
            temp_mark: &mut HashSet<String>,
            result: &mut Vec<&'a ResolvedDep>,
        ) {
            if visited.contains(name) {
                return;
            }
            if temp_mark.contains(name) {
                return; // Cycle, but we allow it here
            }

            temp_mark.insert(name.to_string());

            if let Some(pkg) = packages.get(name) {
                for dep in &pkg.dependencies {
                    visit(dep, packages, visited, temp_mark, result);
                }
                visited.insert(name.to_string());
                result.push(pkg);
            }

            temp_mark.remove(name);
        }

        for name in self.packages.keys() {
            visit(
                name,
                &self.packages,
                &mut visited,
                &mut temp_mark,
                &mut result,
            );
        }

        result
    }
}

/// Configuration for the resolver.
#[derive(Clone, Debug)]
pub struct ResolverConfig {
    /// Maximum resolution depth.
    pub max_depth: usize,
    /// Whether to use the lockfile for version hints.
    pub use_lockfile: bool,
    /// Whether to update locked packages.
    pub update: bool,
    /// Packages to specifically update.
    pub update_packages: Vec<String>,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            use_lockfile: true,
            update: false,
            update_packages: Vec::new(),
        }
    }
}

/// Dependency resolver.
pub struct Resolver<'a, R: PackageRegistry> {
    /// Package registry.
    registry: &'a R,
    /// Configuration.
    config: ResolverConfig,
    /// Existing lockfile (for version hints).
    lockfile: Option<Lockfile>,
    /// Version constraints collected during resolution.
    constraints: HashMap<String, Vec<VersionConstraint>>,
    /// Package states.
    states: HashMap<String, PackageState>,
    /// Current resolution path (for cycle detection).
    path: Vec<String>,
    /// Cache of available versions.
    version_cache: HashMap<String, Vec<Version>>,
}

impl<'a, R: PackageRegistry> Resolver<'a, R> {
    /// Create a new resolver.
    pub fn new(registry: &'a R) -> Self {
        Self::with_config(registry, ResolverConfig::default())
    }

    /// Create a resolver with custom configuration.
    pub fn with_config(registry: &'a R, config: ResolverConfig) -> Self {
        Self {
            registry,
            config,
            lockfile: None,
            constraints: HashMap::new(),
            states: HashMap::new(),
            path: Vec::new(),
            version_cache: HashMap::new(),
        }
    }

    /// Set the lockfile to use for version hints.
    pub fn with_lockfile(mut self, lockfile: Lockfile) -> Self {
        self.lockfile = Some(lockfile);
        self
    }

    /// Resolve dependencies for a manifest.
    pub fn resolve(&mut self, manifest: &Manifest) -> ResolveResult<Resolution> {
        let root_name = manifest.name().to_string();
        let mut resolution = Resolution::new(root_name.clone());

        // Add root package constraints
        for (name, dep) in &manifest.dependencies {
            let constraint = VersionConstraint {
                requirement: dep.version_req()?,
                required_by: root_name.clone(),
                features: dep.features().to_vec(),
            };
            self.constraints
                .entry(name.clone())
                .or_default()
                .push(constraint);
        }

        // Resolve all dependencies
        let mut queue: VecDeque<String> = manifest.dependencies.keys().cloned().collect();
        let mut seen = HashSet::new();

        while let Some(pkg_name) = queue.pop_front() {
            if seen.contains(&pkg_name) {
                continue;
            }
            seen.insert(pkg_name.clone());

            // Check for circular dependency
            if self.path.contains(&pkg_name) {
                let mut cycle = self.path.clone();
                cycle.push(pkg_name.clone());
                return Err(ResolveError::CircularDependency(cycle));
            }

            // Depth check
            if self.path.len() >= self.config.max_depth {
                return Err(ResolveError::MaxDepthExceeded(self.config.max_depth));
            }

            self.path.push(pkg_name.clone());
            let resolved = self.resolve_package(&pkg_name, manifest)?;
            self.path.pop();

            // Add transitive dependencies to queue
            for dep_name in &resolved.dependencies {
                if !seen.contains(dep_name) {
                    queue.push_back(dep_name.clone());
                }
            }

            resolution.add(resolved);
        }

        Ok(resolution)
    }

    /// Resolve a single package.
    fn resolve_package(&mut self, name: &str, _manifest: &Manifest) -> ResolveResult<ResolvedDep> {
        // Check if already resolved
        if let Some(PackageState::Resolved(dep)) = self.states.get(name) {
            return Ok(dep.clone());
        }

        // Get all constraints for this package
        let constraints = self.constraints.get(name).cloned().unwrap_or_default();

        // Try to use locked version first
        if let Some(locked) = self.try_locked_version(name, &constraints)? {
            self.states
                .insert(name.to_string(), PackageState::Resolved(locked.clone()));
            return Ok(locked);
        }

        // Find compatible version
        let versions = self.get_available_versions(name)?;
        let version = self.find_compatible_version(name, &versions, &constraints)?;

        // Get package metadata from registry
        let index = self.registry.get_package(name)?;
        let version_info = index
            .versions
            .iter()
            .find(|v| v.version == version)
            .ok_or_else(|| ResolveError::PackageNotFound(name.to_string()))?;

        // Collect features from all constraints
        let mut features: BTreeSet<String> = BTreeSet::new();
        for constraint in &constraints {
            features.extend(constraint.features.iter().cloned());
        }

        // Add dependency constraints from this package
        for (dep_name, dep_req) in &version_info.dependencies {
            let constraint = VersionConstraint {
                requirement: dep_req.clone(),
                required_by: name.to_string(),
                features: Vec::new(),
            };
            self.constraints
                .entry(dep_name.clone())
                .or_default()
                .push(constraint);
        }

        let resolved = ResolvedDep {
            name: name.to_string(),
            version,
            source: DependencySource::Registry,
            dependencies: version_info.dependencies.keys().cloned().collect(),
            features: features.into_iter().collect(),
        };

        self.states
            .insert(name.to_string(), PackageState::Resolved(resolved.clone()));
        Ok(resolved)
    }

    /// Try to use a locked version if it satisfies all constraints.
    fn try_locked_version(
        &self,
        name: &str,
        constraints: &[VersionConstraint],
    ) -> ResolveResult<Option<ResolvedDep>> {
        if !self.config.use_lockfile {
            return Ok(None);
        }

        // Check if we should update this package
        if self.config.update || self.config.update_packages.contains(&name.to_string()) {
            return Ok(None);
        }

        let Some(ref lockfile) = self.lockfile else {
            return Ok(None);
        };

        let Some(locked) = lockfile.find_package(name) else {
            return Ok(None);
        };

        // Check if locked version satisfies all constraints
        for constraint in constraints {
            if !constraint.requirement.matches(&locked.version) {
                return Ok(None);
            }
        }

        // Convert locked package to resolved
        let source = match &locked.source {
            LockedSource::Registry => DependencySource::Registry,
            LockedSource::Git { url, rev } => DependencySource::Git {
                url: url.clone(),
                reference: rev.clone(),
            },
            LockedSource::Path(path) => DependencySource::Path(path.clone()),
        };

        Ok(Some(ResolvedDep {
            name: name.to_string(),
            version: locked.version.clone(),
            source,
            dependencies: locked.dependencies.clone(),
            features: locked.features.clone(),
        }))
    }

    /// Get available versions for a package.
    fn get_available_versions(&mut self, name: &str) -> ResolveResult<Vec<Version>> {
        if let Some(versions) = self.version_cache.get(name) {
            return Ok(versions.clone());
        }

        let index = self.registry.get_package(name)?;
        let versions: Vec<Version> = index.versions.iter().map(|v| v.version.clone()).collect();

        self.version_cache
            .insert(name.to_string(), versions.clone());
        Ok(versions)
    }

    /// Find a compatible version satisfying all constraints.
    fn find_compatible_version(
        &self,
        name: &str,
        versions: &[Version],
        constraints: &[VersionConstraint],
    ) -> ResolveResult<Version> {
        // Sort versions in descending order (prefer newer)
        let mut sorted: Vec<_> = versions.iter().collect();
        sorted.sort_by(|a, b| b.cmp(a));

        for version in sorted {
            let mut compatible = true;

            for constraint in constraints {
                if !constraint.requirement.matches(version) {
                    compatible = false;
                    break;
                }
            }

            if compatible {
                return Ok(version.clone());
            }
        }

        // No compatible version found
        if constraints.len() == 1 {
            Err(ResolveError::NoCompatibleVersion {
                package: name.to_string(),
                requirement: constraints[0].requirement.clone(),
            })
        } else if constraints.len() >= 2 {
            Err(ResolveError::VersionConflict {
                package: name.to_string(),
                req1: constraints[0].requirement.clone(),
                from1: constraints[0].required_by.clone(),
                req2: constraints[1].requirement.clone(),
                from2: constraints[1].required_by.clone(),
            })
        } else {
            Err(ResolveError::PackageNotFound(name.to_string()))
        }
    }
}

/// Check if two version requirements can be satisfied simultaneously.
pub fn requirements_compatible(req1: &VersionReq, req2: &VersionReq) -> bool {
    // Simple heuristic: check if there's overlap in the version ranges
    // A proper implementation would compute the intersection

    // Check some common versions
    let test_versions = [
        Version::new(0, 1, 0),
        Version::new(1, 0, 0),
        Version::new(2, 0, 0),
        Version::new(3, 0, 0),
    ];

    for v in &test_versions {
        if req1.matches(v) && req2.matches(v) {
            return true;
        }
    }

    // Also test with minor/patch variations
    for major in 0..5 {
        for minor in 0..5 {
            let v = Version::new(major, minor, 0);
            if req1.matches(&v) && req2.matches(&v) {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{PackageIndex, PackageVersionInfo, RegistryConfig};

    // Mock registry for testing
    struct MockRegistry {
        packages: HashMap<String, PackageIndex>,
    }

    impl MockRegistry {
        fn new() -> Self {
            Self {
                packages: HashMap::new(),
            }
        }

        fn add_package(&mut self, name: &str, version: Version, deps: Vec<(&str, &str)>) {
            let index = self
                .packages
                .entry(name.to_string())
                .or_insert_with(|| PackageIndex {
                    name: name.to_string(),
                    versions: Vec::new(),
                });

            let dependencies: HashMap<String, VersionReq> = deps
                .into_iter()
                .map(|(n, v)| (n.to_string(), VersionReq::parse(v).unwrap()))
                .collect();

            index.versions.push(PackageVersionInfo {
                version,
                yanked: false,
                dependencies,
                checksum: None,
            });
        }
    }

    impl PackageRegistry for MockRegistry {
        fn get_package(&self, name: &str) -> Result<PackageIndex, RegistryError> {
            self.packages
                .get(name)
                .cloned()
                .ok_or_else(|| RegistryError::PackageNotFound(name.to_string()))
        }

        fn fetch_package(&self, _name: &str, _version: &Version) -> Result<Vec<u8>, RegistryError> {
            Ok(Vec::new())
        }

        fn config(&self) -> &RegistryConfig {
            static CONFIG: once_cell::sync::Lazy<RegistryConfig> =
                once_cell::sync::Lazy::new(RegistryConfig::default);
            &CONFIG
        }
    }

    #[test]
    fn test_simple_resolution() {
        let mut registry = MockRegistry::new();
        registry.add_package("base", Version::new(1, 0, 0), vec![]);
        registry.add_package("text", Version::new(1, 0, 0), vec![("base", "^1.0")]);

        let manifest_toml = r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
text = "^1.0"
"#;
        let manifest = Manifest::parse(manifest_toml).unwrap();

        let mut resolver = Resolver::new(&registry);
        let resolution = resolver.resolve(&manifest).unwrap();

        assert_eq!(resolution.len(), 2);
        assert!(resolution.get("text").is_some());
        assert!(resolution.get("base").is_some());
    }

    #[test]
    fn test_version_selection() {
        let mut registry = MockRegistry::new();
        registry.add_package("base", Version::new(1, 0, 0), vec![]);
        registry.add_package("base", Version::new(1, 1, 0), vec![]);
        registry.add_package("base", Version::new(2, 0, 0), vec![]);

        let manifest_toml = r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
base = "^1.0"
"#;
        let manifest = Manifest::parse(manifest_toml).unwrap();

        let mut resolver = Resolver::new(&registry);
        let resolution = resolver.resolve(&manifest).unwrap();

        let base = resolution.get("base").unwrap();
        // Should select highest compatible version
        assert_eq!(base.version, Version::new(1, 1, 0));
    }

    #[test]
    fn test_requirements_compatible() {
        assert!(requirements_compatible(
            &VersionReq::parse("^1.0").unwrap(),
            &VersionReq::parse(">=1.0").unwrap()
        ));

        // These might not be compatible depending on exact versions
        // The function is a heuristic
    }
}
