# bhc-linker

Linking support for the Basel Haskell Compiler.

## Overview

This crate handles the final linking stage of compilation, combining object files and libraries into executables or shared libraries.

## Platform Support

| Platform | Default Linker | Notes |
|----------|---------------|-------|
| Linux | `cc` (gcc/clang) | Uses system linker driver |
| macOS | `cc` (clang) | Uses ld64 underneath |
| Windows | `link.exe` | MSVC linker |
| WASM | `wasm-ld` | LLVM WASM linker |

## Key Types

| Type | Description |
|------|-------------|
| `Linker` | Main linker interface |
| `LinkerConfig` | Linker configuration |
| `LinkOutputType` | Output type (executable, library) |
| `LinkLibrary` | Library to link against |

## Usage

### Basic Linking

```rust
use bhc_linker::{Linker, LinkerConfig, LinkOutputType};

let config = LinkerConfig {
    output_type: LinkOutputType::Executable,
    output_path: "my_program".into(),
    ..Default::default()
};

let linker = Linker::new(config)?;

// Add object files
linker.add_object("main.o")?;
linker.add_object("utils.o")?;

// Link
linker.link()?;
```

### Linking Libraries

```rust
use bhc_linker::{Linker, LinkLibrary};

let mut linker = Linker::new(config)?;

// Add search paths
linker.add_library_path("/usr/lib")?;
linker.add_library_path("/usr/local/lib")?;

// Link libraries
linker.add_library(LinkLibrary::Dynamic("m"))?;       // -lm
linker.add_library(LinkLibrary::Static("mylib"))?;   // libmylib.a
linker.add_library(LinkLibrary::Framework("CoreFoundation"))?;  // macOS

linker.link()?;
```

### Creating Libraries

```rust
use bhc_linker::{Linker, LinkerConfig, LinkOutputType};

// Static library
let static_config = LinkerConfig {
    output_type: LinkOutputType::StaticLib,
    output_path: "libfoo.a".into(),
    ..Default::default()
};

// Dynamic library
let dynamic_config = LinkerConfig {
    output_type: LinkOutputType::DynamicLib,
    output_path: "libfoo.so".into(),
    ..Default::default()
};
```

## Output Types

| Type | Description |
|------|-------------|
| `Executable` | Executable binary |
| `StaticLib` | Static library (.a, .lib) |
| `DynamicLib` | Shared library (.so, .dylib, .dll) |

## Linker Configuration

```rust
pub struct LinkerConfig {
    /// Output type
    pub output_type: LinkOutputType,

    /// Output path
    pub output_path: Utf8PathBuf,

    /// Library search paths
    pub library_paths: Vec<Utf8PathBuf>,

    /// Libraries to link
    pub libraries: Vec<LinkLibrary>,

    /// Additional linker flags
    pub extra_flags: Vec<String>,

    /// Enable link-time optimization
    pub lto: bool,

    /// Strip symbols from output
    pub strip: bool,

    /// Generate debug symbols
    pub debug: bool,
}
```

## Library Types

```rust
pub enum LinkLibrary {
    /// Dynamic library: -lfoo
    Dynamic(String),

    /// Static library: path/to/libfoo.a
    Static(String),

    /// Full path to library
    Path(Utf8PathBuf),

    /// macOS framework: -framework Foo
    Framework(String),

    /// Windows import library
    Import(String),
}
```

## Platform-Specific Behavior

### Linux

```rust
// Typical Linux linking
linker.add_library(LinkLibrary::Dynamic("pthread"))?;
linker.add_library(LinkLibrary::Dynamic("m"))?;
linker.add_library(LinkLibrary::Dynamic("dl"))?;
```

### macOS

```rust
// macOS frameworks
linker.add_library(LinkLibrary::Framework("Foundation"))?;
linker.add_library(LinkLibrary::Framework("CoreFoundation"))?;

// Rpath for dynamic libraries
linker.add_rpath("@executable_path/../lib")?;
```

### Windows

```rust
// Windows system libraries
linker.add_library(LinkLibrary::Import("kernel32"))?;
linker.add_library(LinkLibrary::Import("user32"))?;
```

### WASM

```rust
use bhc_linker::wasm::WasmLinker;

let linker = WasmLinker::new(config)?;
linker.add_object("module.o")?;

// Export functions
linker.export("main")?;
linker.export("my_function")?;

// Import from host
linker.import("env", "host_function")?;

linker.link()?;
```

## Error Types

```rust
pub enum LinkerError {
    /// Linker executable not found
    LinkerNotFound(String),

    /// Linker failed with error
    LinkerFailed { message: String, exit_code: Option<i32> },

    /// Failed to execute linker
    ExecutionFailed(std::io::Error),

    /// Invalid configuration
    InvalidConfig(String),

    /// Object file not found
    ObjectNotFound(Utf8PathBuf),

    /// Library not found
    LibraryNotFound(String),
}
```

## Link-Time Optimization

```rust
let config = LinkerConfig {
    lto: true,  // Enable LTO
    ..Default::default()
};

// LTO modes
pub enum LtoMode {
    /// No LTO
    None,
    /// Thin LTO (faster, parallel)
    Thin,
    /// Full LTO (slower, better optimization)
    Full,
}
```

## Design Notes

- Uses system linker by default for compatibility
- LLD available as alternative linker
- Rpath handling for relocatable binaries
- Symbol visibility control for libraries

## Related Crates

- `bhc-codegen` - Produces object files
- `bhc-driver` - Compilation orchestration
- `bhc-target` - Platform information
- `bhc-session` - Output configuration

## Specification References

- H26-SPEC Section 13: Linking
- System V ABI
- macOS Mach-O Format
- PE/COFF Format (Windows)
