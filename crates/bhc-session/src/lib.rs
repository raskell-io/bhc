//! Compiler session state, options, and configuration for BHC.
//!
//! This crate provides the central session management for the BHC compiler,
//! including compilation options, diagnostic configuration, and global state
//! that persists throughout a compilation unit.
//!
//! # Overview
//!
//! The [`Session`] type is the primary entry point, holding all state needed
//! for a single compilation. It includes:
//!
//! - Compiler options and flags
//! - Target specification
//! - Diagnostic handler
//! - Search paths for modules and libraries
//!
//! # Profiles
//!
//! BHC supports multiple compilation profiles as specified in H26-SPEC:
//!
//! - **Default**: Standard lazy Haskell semantics
//! - **Server**: Optimized for concurrent server workloads
//! - **Numeric**: Strict-by-default with fusion guarantees
//! - **Edge**: Minimal runtime for embedded/WASM targets

#![warn(missing_docs)]

use camino::{Utf8Path, Utf8PathBuf};
use parking_lot::RwLock;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The compilation profile determines evaluation semantics and optimization strategy.
///
/// See H26-SPEC Section 4 for detailed profile specifications.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Profile {
    /// Standard lazy Haskell evaluation with GC-managed memory.
    #[default]
    Default,
    /// Optimized for server workloads: concurrency, bounded latency, observability.
    Server,
    /// Numeric computing: strict-by-default, unboxed, fusion guaranteed.
    Numeric,
    /// Minimal runtime footprint for embedded and WASM targets.
    Edge,
    /// Realtime profile: bounded GC pauses (<1ms), arena per-frame.
    Realtime,
    /// Bare-metal microcontrollers: no GC, static allocation only.
    /// Programs with escaping allocations are rejected at compile time.
    Embedded,
}

impl Profile {
    /// Returns true if this profile uses strict evaluation by default.
    #[must_use]
    pub const fn is_strict_by_default(self) -> bool {
        matches!(self, Self::Numeric | Self::Edge | Self::Embedded)
    }

    /// Returns true if fusion is guaranteed for this profile.
    #[must_use]
    pub const fn has_fusion_guarantees(self) -> bool {
        matches!(self, Self::Numeric)
    }

    /// Returns true if this profile requires escape analysis.
    ///
    /// For Embedded profile, programs with escaping allocations are rejected
    /// at compile time since there is no GC to manage memory.
    #[must_use]
    pub const fn requires_escape_analysis(self) -> bool {
        matches!(self, Self::Embedded)
    }

    /// Returns true if this profile has no garbage collector.
    #[must_use]
    pub const fn is_gc_free(self) -> bool {
        matches!(self, Self::Embedded)
    }
}

/// Optimization level for code generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptLevel {
    /// No optimizations (fastest compilation).
    None,
    /// Basic optimizations.
    #[default]
    Less,
    /// Standard optimizations (default).
    Default,
    /// Aggressive optimizations.
    Aggressive,
    /// Size-optimized output.
    Size,
    /// Aggressively size-optimized output.
    SizeMin,
}

/// Debug information level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DebugInfo {
    /// No debug information.
    #[default]
    None,
    /// Line tables only.
    LineTablesOnly,
    /// Full debug information.
    Full,
}

/// Output type for compilation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputType {
    /// Object file (.o).
    #[default]
    Object,
    /// Static library (.a).
    StaticLib,
    /// Dynamic library (.so/.dylib/.dll).
    DynamicLib,
    /// Executable binary.
    Executable,
    /// LLVM IR (.ll).
    LlvmIr,
    /// LLVM bitcode (.bc).
    LlvmBitcode,
    /// Assembly (.s).
    Assembly,
    /// WebAssembly module (.wasm).
    Wasm,
}

/// Compiler options that can be set via CLI or configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Options {
    /// The compilation profile to use.
    pub profile: Profile,
    /// Optimization level.
    pub opt_level: OptLevel,
    /// Debug information level.
    pub debug_info: DebugInfo,
    /// Output type.
    pub output_type: OutputType,
    /// Target triple (e.g., "x86_64-unknown-linux-gnu").
    pub target_triple: Option<String>,
    /// Output path for compiled artifacts.
    pub output_path: Option<Utf8PathBuf>,
    /// Search paths for module imports.
    pub import_paths: Vec<Utf8PathBuf>,
    /// Search paths for libraries.
    pub library_paths: Vec<Utf8PathBuf>,
    /// Libraries to link.
    pub libraries: Vec<String>,
    /// Enable all warnings.
    pub warn_all: bool,
    /// Treat warnings as errors.
    pub deny_warnings: bool,
    /// Generate kernel reports (Numeric profile).
    pub emit_kernel_report: bool,
    /// Dump intermediate representations.
    pub dump_ir: IrDumpOptions,
    /// Path to the BHC standard library.
    /// Used for implicit Prelude loading and module resolution.
    pub stdlib_path: Option<Utf8PathBuf>,
    /// Hackage package dependencies as "name:version" pairs.
    /// Example: `["filepath:1.4.100.0", "directory:1.3.8.0"]`
    pub hackage_packages: Vec<String>,
    /// Compile-only mode: produce .o files without linking.
    pub compile_only: bool,
    /// Output directory for object files (used with compile_only).
    pub output_object_dir: Option<Utf8PathBuf>,
    /// Output directory for interface files (used with compile_only).
    pub output_interface_dir: Option<Utf8PathBuf>,
    /// Package database paths for dependency lookup.
    pub package_dbs: Vec<Utf8PathBuf>,
    /// Exposed dependency package IDs.
    pub package_ids: Vec<String>,
    /// Enabled language extensions (e.g., "OverloadedStrings").
    pub extensions: Vec<String>,
    /// CPP preprocessor defines (e.g., "FOO", "BAR=1").
    pub cpp_defines: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            profile: Profile::Default,
            opt_level: OptLevel::Default,
            debug_info: DebugInfo::None,
            output_type: OutputType::Object,
            target_triple: None,
            output_path: None,
            import_paths: Vec::new(),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            warn_all: false,
            deny_warnings: false,
            emit_kernel_report: false,
            dump_ir: IrDumpOptions::default(),
            stdlib_path: None,
            hackage_packages: Vec::new(),
            compile_only: false,
            output_object_dir: None,
            output_interface_dir: None,
            package_dbs: Vec::new(),
            package_ids: Vec::new(),
            extensions: Vec::new(),
            cpp_defines: Vec::new(),
        }
    }
}

/// Options for dumping intermediate representations.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IrDumpOptions {
    /// Dump AST after parsing.
    pub dump_ast: bool,
    /// Dump Core IR.
    pub dump_core: bool,
    /// Dump Core IR after specific passes.
    pub dump_core_passes: Vec<String>,
    /// Dump Tensor IR.
    pub dump_tensor_ir: bool,
    /// Dump Loop IR.
    pub dump_loop_ir: bool,
    /// Dump LLVM IR.
    pub dump_llvm: bool,
}

/// Search path configuration for finding modules and libraries.
#[derive(Clone, Debug, Default)]
pub struct SearchPaths {
    /// Paths to search for source modules.
    pub module_paths: Vec<Utf8PathBuf>,
    /// Paths to search for interface files (.bhi).
    pub interface_paths: Vec<Utf8PathBuf>,
    /// Paths to search for libraries.
    pub library_paths: Vec<Utf8PathBuf>,
}

impl SearchPaths {
    /// Create a new empty search path configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a module search path.
    pub fn add_module_path(&mut self, path: impl Into<Utf8PathBuf>) {
        self.module_paths.push(path.into());
    }

    /// Add an interface search path.
    pub fn add_interface_path(&mut self, path: impl Into<Utf8PathBuf>) {
        self.interface_paths.push(path.into());
    }

    /// Add a library search path.
    pub fn add_library_path(&mut self, path: impl Into<Utf8PathBuf>) {
        self.library_paths.push(path.into());
    }
}

/// Errors that can occur during session operations.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// Configuration file not found.
    #[error("configuration file not found: {0}")]
    ConfigNotFound(Utf8PathBuf),
    /// Invalid configuration file.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Target not supported.
    #[error("unsupported target: {0}")]
    UnsupportedTarget(String),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// The compiler session holds all state for a compilation unit.
///
/// This is the primary type for managing compilation state and should be
/// created at the start of compilation and passed through all phases.
pub struct Session {
    /// Compiler options.
    pub options: Options,
    /// Search paths for modules and libraries.
    pub search_paths: SearchPaths,
    /// Set of already-loaded module names (for cycle detection).
    loaded_modules: RwLock<FxHashSet<String>>,
    /// Working directory for the session.
    working_dir: Utf8PathBuf,
}

impl Session {
    /// Create a new session with the given options.
    ///
    /// # Errors
    ///
    /// Returns an error if the current working directory cannot be determined.
    pub fn new(options: Options) -> Result<Self, SessionError> {
        let working_dir = std::env::current_dir()
            .map_err(SessionError::Io)?
            .try_into()
            .map_err(|e| SessionError::InvalidConfig(format!("invalid working dir: {e}")))?;

        Ok(Self {
            options,
            search_paths: SearchPaths::default(),
            loaded_modules: RwLock::new(FxHashSet::default()),
            working_dir,
        })
    }

    /// Create a new session with default options.
    ///
    /// # Errors
    ///
    /// Returns an error if the current working directory cannot be determined.
    pub fn with_defaults() -> Result<Self, SessionError> {
        Self::new(Options::default())
    }

    /// Get the working directory for this session.
    #[must_use]
    pub fn working_dir(&self) -> &Utf8Path {
        &self.working_dir
    }

    /// Get the compilation profile.
    #[must_use]
    pub fn profile(&self) -> Profile {
        self.options.profile
    }

    /// Check if a module has been loaded (for cycle detection).
    #[must_use]
    pub fn is_module_loaded(&self, name: &str) -> bool {
        self.loaded_modules.read().contains(name)
    }

    /// Mark a module as loaded.
    pub fn mark_module_loaded(&self, name: String) {
        self.loaded_modules.write().insert(name);
    }

    /// Get the output path, computing a default if not specified.
    #[must_use]
    pub fn output_path(&self, input_name: &str) -> Utf8PathBuf {
        if let Some(ref path) = self.options.output_path {
            path.clone()
        } else {
            let stem = Utf8Path::new(input_name).file_stem().unwrap_or(input_name);
            let ext = match self.options.output_type {
                OutputType::Object => "o",
                OutputType::StaticLib => "a",
                OutputType::DynamicLib => {
                    if cfg!(target_os = "macos") {
                        "dylib"
                    } else if cfg!(target_os = "windows") {
                        "dll"
                    } else {
                        "so"
                    }
                }
                OutputType::Executable => "",
                OutputType::LlvmIr => "ll",
                OutputType::LlvmBitcode => "bc",
                OutputType::Assembly => "s",
                OutputType::Wasm => "wasm",
            };
            if ext.is_empty() {
                Utf8PathBuf::from(stem)
            } else {
                Utf8PathBuf::from(format!("{stem}.{ext}"))
            }
        }
    }
}

/// A shared, thread-safe reference to a session.
pub type SessionRef = Arc<Session>;

/// Create a shared session reference.
#[must_use]
pub fn create_session(options: Options) -> Result<SessionRef, SessionError> {
    Ok(Arc::new(Session::new(options)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_properties() {
        assert!(!Profile::Default.is_strict_by_default());
        assert!(!Profile::Server.is_strict_by_default());
        assert!(Profile::Numeric.is_strict_by_default());
        assert!(Profile::Edge.is_strict_by_default());

        assert!(Profile::Numeric.has_fusion_guarantees());
        assert!(!Profile::Default.has_fusion_guarantees());
    }

    #[test]
    fn test_session_module_tracking() {
        let session = Session::with_defaults().unwrap();
        assert!(!session.is_module_loaded("Data.List"));
        session.mark_module_loaded("Data.List".to_string());
        assert!(session.is_module_loaded("Data.List"));
    }
}
