# bhc-linker

Linking support for the Basel Haskell Compiler.

## Overview

`bhc-linker` handles the final linking stage, combining object files and libraries into executables or shared libraries. Features:

- **System linker invocation**: cc, lld, link.exe, wasm-ld
- **Library management**: Search paths, static/dynamic linking
- **Platform support**: Linux, macOS, Windows, WASM
- **LTO integration**: Link-time optimization support

## Platform Support

| Platform | Default Linker | Notes |
|----------|---------------|-------|
| Linux | `cc` (gcc/clang) | Uses system linker driver |
| macOS | `cc` (clang) | Uses ld64 underneath |
| Windows | `link.exe` | MSVC linker |
| WASM | `wasm-ld` | LLVM WASM linker |

## Core Types

| Type | Description |
|------|-------------|
| `LinkerConfig` | Linking configuration |
| `LinkOutputType` | Output type |
| `LinkLibrary` | Library reference |
| `Linker` | Linker trait |
| `SystemLinker` | System linker |
| `LldLinker` | LLD linker |

## LinkerConfig

```rust
pub struct LinkerConfig {
    /// Target specification
    pub target: TargetSpec,
    /// Output type
    pub output_type: LinkOutputType,
    /// Output path
    pub output_path: Utf8PathBuf,
    /// Object files to link
    pub objects: Vec<Utf8PathBuf>,
    /// Libraries to link
    pub libraries: Vec<LinkLibrary>,
    /// Library search paths
    pub library_paths: Vec<Utf8PathBuf>,
    /// Position-independent executable
    pub pie: bool,
    /// Link-time optimization
    pub lto: bool,
    /// Strip debug symbols
    pub strip: bool,
    /// Additional flags
    pub extra_flags: Vec<String>,
}
```

### Builder Pattern

```rust
use bhc_linker::{LinkerConfig, LinkLibrary, LinkOutputType};
use bhc_target::targets::x86_64_linux_gnu;

let config = LinkerConfig::new(x86_64_linux_gnu(), "my_program")
    .with_object("main.o")
    .with_object("lib.o")
    .with_library(LinkLibrary::named("m"))
    .with_library(LinkLibrary::named("pthread"))
    .with_library_path("/usr/local/lib")
    .output_type(LinkOutputType::Executable)
    .with_pie(true)
    .with_strip(true);
```

## Output Types

```rust
pub enum LinkOutputType {
    /// Executable binary
    Executable,
    /// Static library (.a)
    StaticLib,
    /// Dynamic/shared library (.so/.dylib/.dll)
    DynamicLib,
}
```

## Link Libraries

```rust
pub struct LinkLibrary {
    /// Library name (without lib prefix or extension)
    pub name: String,
    /// Link statically
    pub static_link: bool,
    /// Specific path to library
    pub path: Option<Utf8PathBuf>,
}

// Create library references
let libc = LinkLibrary::named("c");
let libm_static = LinkLibrary::static_lib("m");
let custom = LinkLibrary::with_path("mylib", "/path/to/libmylib.a");
```

## Linker Trait

```rust
pub trait Linker: Send + Sync {
    /// Linker name
    fn name(&self) -> &'static str;

    /// Check target support
    fn supports_target(&self, target: &TargetSpec) -> bool;

    /// Linker executable
    fn executable(&self) -> &str;

    /// Build command-line arguments
    fn build_args(&self, config: &LinkerConfig) -> Vec<String>;

    /// Run the linker
    fn link(&self, config: &LinkerConfig) -> LinkerResult<()>;
}
```

## System Linker

```rust
use bhc_linker::SystemLinker;
use bhc_target::targets::x86_64_linux_gnu;

// Create for target
let linker = SystemLinker::for_target(&x86_64_linux_gnu());

// Or with specific executable
let linker = SystemLinker::with_executable("clang");

// Link
linker.link(&config)?;
```

## LLD Linker

LLVM's linker with multiple flavors:

```rust
use bhc_linker::{LldLinker, LldFlavor};

pub enum LldFlavor {
    Gnu,    // ld.lld (Linux)
    Darwin, // ld64.lld (macOS)
    Msvc,   // lld-link (Windows)
    Wasm,   // wasm-ld
}

// Create for target
let linker = LldLinker::for_target(&target);

// Link
linker.link(&config)?;
```

## Quick Start

```rust
use bhc_linker::{link, LinkerConfig, LinkLibrary};
use bhc_target::targets::x86_64_linux_gnu;

let target = x86_64_linux_gnu();

let config = LinkerConfig::new(target, "a.out")
    .with_objects(["main.o", "util.o"])
    .with_library(LinkLibrary::named("c"));

// Link using appropriate linker for target
link(&config)?;
```

## Error Handling

```rust
pub enum LinkerError {
    /// Linker not found
    LinkerNotFound(String),
    /// Linker failed
    LinkerFailed {
        message: String,
        exit_code: Option<i32>,
    },
    /// Execution failed
    ExecutionFailed(std::io::Error),
    /// Invalid configuration
    InvalidConfig(String),
}
```

## Generated Arguments

### Unix (cc)

```
cc -o output main.o lib.o -L/usr/lib -lm -lpthread -pie -flto -s
```

### Windows (link.exe)

```
link.exe /OUT:output.exe main.obj lib.obj /LIBPATH:C:\libs m.lib
```

### WASM (wasm-ld)

```
wasm-ld -o output.wasm main.o --no-entry --export-all
```

## Example: Building an Executable

```rust
use bhc_linker::{LinkerConfig, LinkLibrary, LinkOutputType, link};
use bhc_target::TargetSpec;

// Parse target
let target = TargetSpec::parse("x86_64-unknown-linux-gnu")?;

// Configure linking
let config = LinkerConfig::new(target, "my_app")
    .with_object("build/main.o")
    .with_object("build/parser.o")
    .with_object("build/codegen.o")
    .with_library(LinkLibrary::named("c"))
    .with_library(LinkLibrary::named("m"))
    .with_library(LinkLibrary::static_lib("bhc_runtime"))
    .with_library_path("deps/lib")
    .output_type(LinkOutputType::Executable)
    .with_pie(true)
    .with_lto(true);

// Link
link(&config)?;

println!("Built: my_app");
```

## Example: Building a Shared Library

```rust
let config = LinkerConfig::new(target, "libmylib.so")
    .with_objects(["src.o", "util.o"])
    .output_type(LinkOutputType::DynamicLib);

link(&config)?;
```

## See Also

- `bhc-codegen`: Produces object files for linking
- `bhc-target`: Target specifications
- `bhc-driver`: Orchestrates compilation including linking
- GNU ld documentation
- LLVM lld documentation
