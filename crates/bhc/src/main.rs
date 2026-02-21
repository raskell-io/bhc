//! Basel Haskell Compiler (BHC) - Main Entry Point
//!
//! BHC is a next-generation Haskell compiler targeting the Haskell 2026 Platform.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use std::io;
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

/// Basel Haskell Compiler - A next-generation Haskell compiler for 2026
#[derive(Parser, Debug)]
#[command(name = "bhc")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The command to execute
    #[command(subcommand)]
    command: Option<Commands>,

    /// Input files to compile
    #[arg(value_name = "FILE")]
    files: Vec<PathBuf>,

    /// Output file name
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Compilation profile
    #[arg(long, value_enum, default_value = "default")]
    profile: Profile,

    /// Haskell edition
    #[arg(long, default_value = "2026")]
    edition: String,

    /// Optimization level (0-3)
    #[arg(short = 'O', long, default_value = "0")]
    opt_level: u8,

    /// Increase output verbosity (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Suppress non-error output
    #[arg(short, long)]
    quiet: bool,

    /// Emit kernel fusion report (Numeric profile)
    #[arg(long)]
    kernel_report: bool,

    /// Dump intermediate representations
    #[arg(long)]
    dump_ir: Option<IrStage>,

    /// Number of parallel jobs
    #[arg(short, long)]
    jobs: Option<usize>,

    /// Target triple (e.g., wasm32-wasi, x86_64-unknown-linux-gnu, cuda)
    #[arg(long)]
    target: Option<String>,

    /// Emit format for output (e.g., ptx for GPU, llvm-ir for debugging)
    #[arg(long)]
    emit: Option<String>,

    /// Additional import paths for module search
    #[arg(short = 'I', long = "import-path", value_name = "DIR")]
    import_paths: Vec<PathBuf>,

    /// Hackage packages to include (format: "name:version")
    #[arg(long = "package", value_name = "PKG:VER")]
    packages: Vec<String>,

    /// Print bare compiler version number (for toolchain detection)
    #[arg(long = "numeric-version", hide = true)]
    numeric_version: bool,

    /// Compile to object file only, do not link
    #[arg(short = 'c', long = "compile-only")]
    compile_only: bool,

    /// Output directory for object files (used with -c)
    #[arg(long = "odir", value_name = "DIR")]
    odir: Option<PathBuf>,

    /// Output directory for interface files (used with -c)
    #[arg(long = "hidir", value_name = "DIR")]
    hidir: Option<PathBuf>,

    /// Package database paths
    #[arg(long = "package-db", value_name = "PATH")]
    package_dbs: Vec<PathBuf>,

    /// Expose a dependency by package ID
    #[arg(long = "package-id", value_name = "ID")]
    package_ids: Vec<String>,

    /// Enable language extensions (e.g., -XOverloadedStrings)
    #[arg(short = 'X', value_name = "EXT", hide = true)]
    extensions: Vec<String>,

    /// Enable all warnings
    #[arg(long = "Wall", hide = true)]
    wall: bool,

    /// Treat warnings as errors
    #[arg(long = "Werror", hide = true)]
    werror: bool,
}

/// Compilation profile
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Profile {
    /// General Haskell with lazy evaluation
    Default,
    /// Optimized for server workloads
    Server,
    /// Optimized for numeric/tensor computation
    Numeric,
    /// Minimal runtime for embedded/WASM
    Edge,
}

/// IR stages for dumping
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum IrStage {
    /// Abstract syntax tree
    Ast,
    /// High-level IR
    Hir,
    /// Core IR
    Core,
    /// Tensor IR
    Tensor,
    /// Loop IR
    Loop,
    /// All stages
    All,
}

/// Subcommands
#[derive(Subcommand, Debug)]
enum Commands {
    /// Compile source files
    Build {
        /// Input files
        #[arg(value_name = "FILE")]
        files: Vec<PathBuf>,
    },

    /// Check source files without generating code
    Check {
        /// Input files
        #[arg(value_name = "FILE")]
        files: Vec<PathBuf>,
    },

    /// Run a Haskell program
    Run {
        /// The file to run
        file: PathBuf,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Start the interactive REPL
    Repl,

    /// Show version information
    Version,

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // --numeric-version: print bare version and exit (used by hx-bhc for toolchain detection)
    if cli.numeric_version {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    // Set up logging based on verbosity flags
    // Default: WARN (quiet operation), -v: INFO, -vv: DEBUG, -vvv: TRACE
    // --quiet: ERROR only
    let log_level = if cli.quiet {
        Level::ERROR
    } else {
        match cli.verbose {
            0 => Level::WARN,  // Default: only warnings and errors
            1 => Level::INFO,  // -v: informational messages
            2 => Level::DEBUG, // -vv: debug output
            _ => Level::TRACE, // -vvv: trace everything
        }
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Initialize the string interner with common keywords
    bhc_intern::kw::intern_all();

    match cli.command {
        Some(Commands::Build { ref files }) => {
            compile_files(files, &cli)?;
        }
        Some(Commands::Check { ref files }) => {
            check_files(files, &cli)?;
        }
        Some(Commands::Run { ref file, ref args }) => {
            run_file(file, args, &cli)?;
        }
        Some(Commands::Repl) => {
            start_repl(&cli)?;
        }
        Some(Commands::Version) => {
            print_version();
        }
        Some(Commands::Completions { shell }) => {
            generate_completions(shell);
        }
        None => {
            if cli.files.is_empty() {
                // No files specified, print help
                println!(
                    "Basel Haskell Compiler (BHC) v{}",
                    env!("CARGO_PKG_VERSION")
                );
                println!();
                println!("Usage: bhc [OPTIONS] [FILES]...");
                println!();
                println!("For more information, try '--help'");
            } else if cli.compile_only {
                compile_modules_only(&cli.files, &cli)?;
            } else {
                compile_files(&cli.files, &cli)?;
            }
        }
    }

    Ok(())
}

/// Generate shell completions
fn generate_completions(shell: Shell) {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "bhc", &mut io::stdout());
}

/// Compile source files
fn compile_files(files: &[PathBuf], cli: &Cli) -> Result<()> {
    use bhc_driver::CompilerBuilder;
    use camino::Utf8PathBuf;

    tracing::info!(
        "Compiling {} file(s) with {:?} profile",
        files.len(),
        cli.profile
    );

    // Convert profile
    let profile = match cli.profile {
        Profile::Default => bhc_session::Profile::Default,
        Profile::Server => bhc_session::Profile::Server,
        Profile::Numeric => bhc_session::Profile::Numeric,
        Profile::Edge => bhc_session::Profile::Edge,
    };

    // Determine output type based on target and emit flags
    let output_type = if cli.emit.as_deref() == Some("ptx") {
        bhc_session::OutputType::Object // PTX is emitted as object
    } else if cli.target.as_deref() == Some("wasm32-wasi") || cli.target.as_deref() == Some("wasm")
    {
        bhc_session::OutputType::Wasm
    } else {
        bhc_session::OutputType::Executable
    };

    // Build compiler with configuration
    let mut builder = CompilerBuilder::new()
        .profile(profile)
        .output_type(output_type)
        .emit_kernel_report(cli.kernel_report);

    // Set target if specified
    if let Some(ref target) = cli.target {
        builder = builder.target(target.clone());
    }

    // Set output path if specified
    if let Some(ref output) = cli.output {
        builder = builder.output_path(
            Utf8PathBuf::from_path_buf(output.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in output path"))?,
        );
    }

    // Add import paths
    for path in &cli.import_paths {
        builder = builder.import_path(
            Utf8PathBuf::from_path_buf(path.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in import path"))?,
        );
    }

    // Add Hackage packages
    for pkg in &cli.packages {
        builder = builder.hackage_package(pkg.clone());
    }

    let compiler = builder
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create compiler: {}", e))?;

    // Convert paths and compile
    let utf8_paths: Vec<Utf8PathBuf> = files
        .iter()
        .map(|p| {
            Utf8PathBuf::from_path_buf(p.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in path: {}", p.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    // Use ordered compilation for multiple files, or auto-discovery for single file
    let compile_result = if utf8_paths.len() > 1 {
        compiler.compile_files_ordered(utf8_paths.iter().map(|p| p.as_path()))
    } else if utf8_paths.len() == 1 {
        compiler.compile_with_discovery(utf8_paths[0].as_path())
    } else {
        compiler.compile_files(utf8_paths.iter().map(|p| p.as_path()))
    };

    match compile_result {
        Ok(outputs) => {
            for output in &outputs {
                tracing::info!("Generated: {}", output.path);
            }
            // Print final output path (always visible unless --quiet)
            if !cli.quiet {
                if let Some(output) = outputs.last() {
                    println!("{}", output.path);
                }
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Check source files without generating code
fn check_files(files: &[PathBuf], cli: &Cli) -> Result<()> {
    use bhc_driver::CompilerBuilder;
    use camino::Utf8PathBuf;

    tracing::info!("Checking {} file(s)", files.len());

    let profile = match cli.profile {
        Profile::Default => bhc_session::Profile::Default,
        Profile::Server => bhc_session::Profile::Server,
        Profile::Numeric => bhc_session::Profile::Numeric,
        Profile::Edge => bhc_session::Profile::Edge,
    };

    let compiler = CompilerBuilder::new().profile(profile).build()?;

    let mut has_errors = false;
    for file in files {
        let path = Utf8PathBuf::from_path_buf(file.clone())
            .map_err(|p| anyhow::anyhow!("Invalid UTF-8 path: {}", p.display()))?;

        match compiler.check_file(&path) {
            Ok(()) => println!("  {} OK", path),
            Err(e) => {
                eprintln!("  {} FAILED: {}", path, e);
                has_errors = true;
            }
        }
    }

    if has_errors {
        anyhow::bail!("Type checking failed");
    }

    Ok(())
}

/// Run a Haskell program
fn run_file(file: &PathBuf, _args: &[String], cli: &Cli) -> Result<()> {
    use bhc_driver::CompilerBuilder;
    use camino::Utf8PathBuf;

    tracing::info!("Running {}", file.display());

    // Convert profile
    let profile = match cli.profile {
        Profile::Default => bhc_session::Profile::Default,
        Profile::Server => bhc_session::Profile::Server,
        Profile::Numeric => bhc_session::Profile::Numeric,
        Profile::Edge => bhc_session::Profile::Edge,
    };

    // Build compiler
    let compiler = CompilerBuilder::new()
        .profile(profile)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create compiler: {}", e))?;

    // Convert path
    let path = Utf8PathBuf::from_path_buf(file.clone())
        .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in path"))?;

    // Run the file
    match compiler.run_file(&path) {
        Ok((_value, display)) => {
            println!("{}", display);
            Ok(())
        }
        Err(e) => {
            eprintln!("Execution error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Compile source files to object files only (no linking), with optional interface generation.
fn compile_modules_only(files: &[PathBuf], cli: &Cli) -> Result<()> {
    use bhc_driver::CompilerBuilder;
    use camino::Utf8PathBuf;

    tracing::info!(
        "Compiling {} file(s) to object files (compile-only mode)",
        files.len()
    );

    let profile = match cli.profile {
        Profile::Default => bhc_session::Profile::Default,
        Profile::Server => bhc_session::Profile::Server,
        Profile::Numeric => bhc_session::Profile::Numeric,
        Profile::Edge => bhc_session::Profile::Edge,
    };

    let mut builder = CompilerBuilder::new()
        .profile(profile)
        .compile_only(true)
        .output_type(bhc_session::OutputType::Object)
        .emit_kernel_report(cli.kernel_report);

    if let Some(ref target) = cli.target {
        builder = builder.target(target.clone());
    }

    if let Some(ref odir) = cli.odir {
        builder = builder.odir(
            Utf8PathBuf::from_path_buf(odir.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in odir path"))?,
        );
    }

    if let Some(ref hidir) = cli.hidir {
        builder = builder.hidir(
            Utf8PathBuf::from_path_buf(hidir.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in hidir path"))?,
        );
    }

    for path in &cli.import_paths {
        builder = builder.import_path(
            Utf8PathBuf::from_path_buf(path.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in import path"))?,
        );
    }

    for db in &cli.package_dbs {
        builder = builder.package_db(
            Utf8PathBuf::from_path_buf(db.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in package-db path"))?,
        );
    }

    for id in &cli.package_ids {
        builder = builder.package_id(id.clone());
    }

    let compiler = builder
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create compiler: {}", e))?;

    let utf8_paths: Vec<Utf8PathBuf> = files
        .iter()
        .map(|p| {
            Utf8PathBuf::from_path_buf(p.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in path: {}", p.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    for path in &utf8_paths {
        match compiler.compile_module_only(path) {
            Ok(output) => {
                tracing::info!("Generated: {}", output.path);
                if !cli.quiet {
                    println!("{}", output.path);
                }
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

/// Start the interactive REPL
fn start_repl(cli: &Cli) -> Result<()> {
    use std::process::Command;

    let bhc_exe = std::env::current_exe()?;
    let bhci_exe = bhc_exe.with_file_name("bhci");

    let mut args = vec![];
    match cli.profile {
        Profile::Numeric => args.extend(["--profile", "numeric"]),
        Profile::Server => args.extend(["--profile", "server"]),
        Profile::Edge => args.extend(["--profile", "edge"]),
        Profile::Default => {}
    }

    let status = Command::new(&bhci_exe)
        .args(&args)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to start bhci at {}: {}",
                bhci_exe.display(),
                e
            )
        })?;

    std::process::exit(status.code().unwrap_or(1));
}

/// Print version information
fn print_version() {
    println!("Basel Haskell Compiler (BHC)");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Targets Haskell 2026 Platform Specification");
    println!("Supported profiles: Default, Server, Numeric, Edge");
    println!();
    println!("Repository: https://github.com/raskell-io/bhc");
}
