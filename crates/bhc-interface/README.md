# bhc-interface

Interface files for the Basel Haskell Compiler.

## Overview

This crate handles reading and writing interface files (`.bhi`), which contain the public API of compiled modules. Interface files enable separate compilation and type checking without requiring source code.

## File Format

Interface files contain:

- Module metadata (name, version, hash)
- Exported type signatures
- Exported function signatures
- Instance declarations
- Re-exports

## Key Types

| Type | Description |
|------|-------------|
| `Interface` | Complete module interface |
| `ExportedType` | Exported type definition |
| `ExportedValue` | Exported value signature |
| `ExportedInstance` | Exported class instance |

## Usage

### Reading an Interface

```rust
use bhc_interface::{Interface, read_interface};
use camino::Utf8Path;

// Read interface file
let iface = read_interface("Data/List.bhi")?;

// Access module information
println!("Module: {}", iface.module_name);
println!("Hash: {}", iface.content_hash);

// List exports
for (name, sig) in &iface.values {
    println!("  {} :: {}", name, sig);
}
```

### Writing an Interface

```rust
use bhc_interface::{Interface, InterfaceBuilder};
use bhc_types::Scheme;

let mut builder = InterfaceBuilder::new("MyModule");

// Add exported values
builder.add_value("myFunc", scheme)?;
builder.add_value("helper", helper_scheme)?;

// Add exported types
builder.add_type("MyType", type_def)?;

// Build and write
let iface = builder.build()?;
iface.write_to_file("MyModule.bhi")?;
```

## Interface Contents

### Module Metadata

```rust
pub struct ModuleMetadata {
    /// Module name (e.g., "Data.List")
    pub name: Symbol,

    /// Content hash for change detection
    pub content_hash: [u8; 32],

    /// Source file path (optional)
    pub source_path: Option<Utf8PathBuf>,

    /// Compilation timestamp
    pub compiled_at: SystemTime,

    /// Compiler version
    pub compiler_version: String,
}
```

### Exported Values

```rust
pub struct ExportedValue {
    /// Value name
    pub name: Symbol,

    /// Type scheme
    pub scheme: Scheme,

    /// Inlining information
    pub inline: InlineInfo,

    /// Strictness signature
    pub strictness: Option<StrictSig>,
}
```

### Exported Types

```rust
pub struct ExportedType {
    /// Type name
    pub name: Symbol,

    /// Type parameters
    pub params: Vec<TyVar>,

    /// Kind
    pub kind: Kind,

    /// Constructors (if data type)
    pub constructors: Vec<DataCon>,

    /// Is abstract (constructors hidden)?
    pub is_abstract: bool,
}
```

### Exported Instances

```rust
pub struct ExportedInstance {
    /// Class name
    pub class: Symbol,

    /// Instance types
    pub types: Vec<Ty>,

    /// Instance constraints
    pub constraints: Vec<Constraint>,
}
```

## Binary Format

```
+----------------+
| Magic (4 bytes)|  "BHCI"
+----------------+
| Version (4b)   |  Format version
+----------------+
| Metadata       |  Module metadata (bincode)
+----------------+
| Exports        |  Exported definitions (bincode)
+----------------+
| Instances      |  Instance declarations (bincode)
+----------------+
| Checksum       |  Content hash verification
+----------------+
```

## Format Versioning

```rust
pub const INTERFACE_VERSION: u32 = 1;
pub const INTERFACE_MAGIC: &[u8; 4] = b"BHCI";

// Version compatibility
pub fn is_compatible(version: u32) -> bool {
    version <= INTERFACE_VERSION
}
```

## Error Types

```rust
pub enum InterfaceError {
    /// Interface file not found
    NotFound(Utf8PathBuf),

    /// Invalid format (magic bytes mismatch)
    InvalidFormat(String),

    /// Version mismatch
    VersionMismatch { expected: u32, found: u32 },

    /// Serialization/deserialization error
    Serialization(String),

    /// Checksum verification failed
    ChecksumMismatch,

    /// IO error
    Io(std::io::Error),
}
```

## Interface Discovery

```rust
use bhc_interface::find_interface;

// Search for interface file
let iface_path = find_interface("Data.List", &search_paths)?;

// Load with caching
let cache = InterfaceCache::new();
let iface = cache.load("Data.List", &search_paths)?;
```

## Content Hashing

Interfaces include content hashes for change detection:

```rust
use bhc_interface::hash_module;

// Hash source module
let hash = hash_module(&hir_module);

// Check if interface is up to date
if existing_interface.content_hash != hash {
    // Recompile needed
}
```

## Design Notes

- Binary format for fast loading
- Content hashing enables incremental compilation
- Abstract types hide implementation details
- Inlining info supports cross-module optimization

## Related Crates

- `bhc-package` - Package management
- `bhc-typeck` - Uses interfaces for imports
- `bhc-types` - Type representation
- `bhc-query` - Interface caching

## Specification References

- H26-SPEC Section 15: Interface Files
- GHC Interface File Format (for comparison)
