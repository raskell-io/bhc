# bhc-package

Package management and manifest parsing for the Basel Haskell Compiler.

## Overview

This crate handles BHC package definitions, manifest parsing, dependency resolution, and lockfile management. Packages are the unit of distribution and dependency management in BHC.

## Manifest Format

BHC uses TOML for package manifests (`bhc.toml`):

```toml
[package]
name = "my-package"
version = "0.1.0"
edition = "h26"
profile = "default"

[dependencies]
base = "^1.0"
text = { version = "^2.0", features = ["unicode"] }

[dev-dependencies]
test-framework = "^0.5"
```

## Key Types

| Type | Description |
|------|-------------|
| `Package` | Complete package definition |
| `Manifest` | Parsed `bhc.toml` |
| `Dependency` | Dependency specification |
| `Lockfile` | Resolved dependency versions |
| `PackageId` | Unique package identifier |

## Usage

### Parsing a Manifest

```rust
use bhc_package::{Manifest, Package};
use camino::Utf8Path;

// Parse manifest from file
let manifest = Manifest::from_file("bhc.toml")?;

// Access package metadata
println!("Name: {}", manifest.package.name);
println!("Version: {}", manifest.package.version);

// List dependencies
for (name, dep) in &manifest.dependencies {
    println!("  {} = {}", name, dep.version_req);
}
```

### Creating a Package

```rust
use bhc_package::{Package, PackageMetadata};
use semver::Version;

let metadata = PackageMetadata {
    name: "my-package".to_string(),
    version: Version::new(0, 1, 0),
    edition: Edition::H26,
    profile: Profile::Default,
    authors: vec!["Me <me@example.com>".to_string()],
    description: Some("A cool package".to_string()),
    ..Default::default()
};

let package = Package::new(metadata)?;
```

## Package Metadata

```rust
pub struct PackageMetadata {
    /// Package name (kebab-case)
    pub name: String,

    /// Semantic version
    pub version: Version,

    /// Language edition
    pub edition: Edition,

    /// Compilation profile
    pub profile: Profile,

    /// Package authors
    pub authors: Vec<String>,

    /// Short description
    pub description: Option<String>,

    /// License identifier
    pub license: Option<String>,

    /// Homepage URL
    pub homepage: Option<String>,

    /// Repository URL
    pub repository: Option<String>,
}
```

## Dependencies

```rust
use bhc_package::Dependency;

// Simple version requirement
let dep = Dependency::version("^1.0");

// With features
let dep = Dependency::new("^1.0")
    .with_features(&["feature1", "feature2"]);

// Git dependency
let dep = Dependency::git("https://github.com/user/repo")
    .with_branch("main");

// Path dependency (for local development)
let dep = Dependency::path("../sibling-package");
```

### Dependency Types

| Type | Example |
|------|---------|
| Version | `"^1.0"` |
| Git | `{ git = "url", branch = "main" }` |
| Path | `{ path = "../pkg" }` |

## Profiles

Packages can specify a default compilation profile:

| Profile | Description |
|---------|-------------|
| `default` | Standard lazy Haskell |
| `server` | Server workloads |
| `numeric` | Numeric computing |
| `edge` | Minimal runtime |

## Lockfile

```rust
use bhc_package::lockfile::{Lockfile, LockedPackage};

// Read existing lockfile
let lockfile = Lockfile::read("bhc.lock")?;

// Check if dependencies are up to date
if lockfile.is_stale(&manifest) {
    // Resolve and update
    let resolved = resolve_dependencies(&manifest)?;
    let lockfile = Lockfile::from_resolved(&resolved);
    lockfile.write("bhc.lock")?;
}
```

### Lockfile Format

```toml
# bhc.lock - DO NOT EDIT
[[package]]
name = "base"
version = "1.2.3"
source = "registry"
checksum = "sha256:..."

[[package]]
name = "my-dep"
version = "0.5.0"
source = "git+https://github.com/..."
dependencies = ["base"]
```

## Dependency Resolution

```rust
use bhc_package::resolve::{resolve, ResolveConfig};

let config = ResolveConfig {
    registry_url: "https://packages.bhc.raskell.io",
    allow_prerelease: false,
    ..Default::default()
};

let resolution = resolve(&manifest, config)?;

for (id, version) in &resolution.packages {
    println!("{} = {}", id.name, version);
}
```

## Error Types

```rust
pub enum PackageError {
    /// Manifest file not found
    ManifestNotFound(Utf8PathBuf),

    /// Invalid manifest format
    InvalidManifest(String),

    /// Invalid version specification
    InvalidVersion(String),

    /// Dependency resolution failed
    ResolutionFailed(String),

    /// Conflicting dependencies
    VersionConflict { package: String, versions: Vec<Version> },

    /// IO error
    Io(std::io::Error),
}
```

## Design Notes

- Uses semantic versioning for all packages
- Lockfiles ensure reproducible builds
- Git dependencies allow pre-release development
- Profile propagates to dependencies by default

## Related Crates

- `bhc-session` - Profile configuration
- `bhc-interface` - Module interface files
- `bhc-driver` - Package building

## Specification References

- H26-SPEC Section 14: Package Management
- Semantic Versioning 2.0.0
