# bhc-package

Package management and manifest parsing for the Basel Haskell Compiler.

## Overview

`bhc-package` handles package definitions, manifest parsing, dependency resolution, and lockfile management. Features:

- **Manifest parsing**: `bhc.toml` configuration files
- **Dependency specification**: Version requirements, features, sources
- **Package resolution**: Version solving and lockfiles
- **Profile selection**: Default, Server, Numeric, Edge

## Manifest Format

BHC uses TOML for package manifests (`bhc.toml`):

```toml
[package]
name = "my-package"
version = "0.1.0"
edition = "h26"
profile = "default"
description = "My awesome package"
license = "MIT"
authors = ["Your Name <you@example.com>"]

[library]
exposed-modules = ["MyModule", "MyModule.Internal"]
src-dir = "src"

[[bin]]
name = "my-app"
main = "Main"

[dependencies]
base = "^1.0"
text = { version = "^2.0", features = ["unicode"] }

[dev-dependencies]
test-framework = "^0.5"

[features]
default = ["fast-math"]
fast-math = []
debug = ["verbose-errors"]
```

## Core Types

| Type | Description |
|------|-------------|
| `Manifest` | Complete package manifest |
| `PackageMetadata` | Package name, version, etc. |
| `Dependency` | Dependency specification |
| `PackageId` | Name + version identifier |
| `Edition` | Language edition (H26) |

## Editions

```rust
pub enum Edition {
    /// Haskell 2026 edition
    H26,
}
```

## Package Metadata

```rust
pub struct PackageMetadata {
    pub name: String,
    pub version: Version,        // semver
    pub edition: Edition,        // h26
    pub profile: Profile,        // default, server, numeric, edge
    pub description: Option<String>,
    pub license: Option<String>,
    pub authors: Vec<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
}
```

## Dependencies

```rust
pub enum Dependency {
    /// Simple: "^1.0"
    Simple(String),
    /// Detailed specification
    Detailed(DetailedDependency),
}

pub struct DetailedDependency {
    pub version: String,
    pub features: Vec<String>,
    pub optional: bool,
    pub git: Option<String>,
    pub branch: Option<String>,
    pub tag: Option<String>,
    pub rev: Option<String>,
    pub path: Option<Utf8PathBuf>,
}

// Get version requirement
let dep = Dependency::Simple("^1.0".into());
let req: VersionReq = dep.version_req()?;

// Check features
let features = dep.features();

// Check if optional
let optional = dep.is_optional();
```

## Loading Manifests

```rust
use bhc_package::{Manifest, find_manifest};

// Load from path
let manifest = Manifest::load("bhc.toml")?;

// Parse from string
let manifest = Manifest::parse(toml_content)?;

// Find manifest in directory tree
let path = find_manifest("/project/src/lib")?;
// Returns /project/bhc.toml

// Access package info
println!("Name: {}", manifest.name());
println!("Version: {}", manifest.version());
println!("Profile: {:?}", manifest.profile());
```

## Library Configuration

```rust
pub struct LibraryConfig {
    /// Exposed modules
    pub exposed_modules: Vec<String>,
    /// Source directory (default: "src")
    pub src_dir: Utf8PathBuf,
}

if manifest.has_library() {
    let lib = manifest.library.as_ref().unwrap();
    for module in &lib.exposed_modules {
        println!("Exposes: {}", module);
    }
}
```

## Executable Configuration

```rust
pub struct ExecutableConfig {
    /// Executable name
    pub name: String,
    /// Main module (default: "Main")
    pub main: String,
    /// Source directory
    pub src_dir: Utf8PathBuf,
}

for exe in &manifest.executables {
    println!("Binary: {} (main: {})", exe.name, exe.main);
}
```

## Test and Benchmark Configuration

```rust
pub struct TestConfig {
    pub name: String,
    pub main: String,      // default: "Test.Main"
    pub src_dir: Utf8PathBuf,  // default: "test"
}

pub struct BenchmarkConfig {
    pub name: String,
    pub main: String,      // default: "Bench.Main"
    pub src_dir: Utf8PathBuf,  // default: "bench"
}
```

## Package Resolution

```rust
pub struct ResolvedPackage {
    pub name: String,
    pub version: Version,
    pub source: PackageSource,
    pub dependencies: Vec<String>,
    pub features: Vec<String>,
}

pub enum PackageSource {
    Registry,
    Git { url: String, rev: String },
    Path(Utf8PathBuf),
}
```

## Package ID

```rust
pub struct PackageId {
    pub name: String,
    pub version: Version,
}

let id = PackageId::new("my-package", Version::new(1, 2, 3));
println!("{}", id);  // "my-package v1.2.3"
```

## Error Handling

```rust
pub enum PackageError {
    /// Manifest not found
    ManifestNotFound(Utf8PathBuf),
    /// Invalid manifest
    InvalidManifest(String),
    /// Invalid version
    InvalidVersion(String),
    /// IO error
    Io(std::io::Error),
    /// TOML parsing error
    Toml(toml::de::Error),
}
```

## Submodules

| Module | Description |
|--------|-------------|
| `lockfile` | Lockfile management |

## See Also

- `bhc-session`: Compilation profiles
- `bhc-driver`: Uses packages for compilation
- `bhc-interface`: Module interface files
- Cargo manifest documentation (inspiration)
