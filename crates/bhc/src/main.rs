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

    match compiler.compile_files(utf8_paths.iter().map(|p| p.as_path())) {
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
    tracing::info!("Checking {} file(s)", files.len());
    // For now, just parse
    compile_files(files, cli)
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

/// Start the interactive REPL
fn start_repl(_cli: &Cli) -> Result<()> {
    println!("Basel Haskell Compiler Interactive (bhci)");
    println!("Type :help for help, :quit to exit");
    println!();
    // TODO: Implement REPL
    println!("(REPL not yet implemented)");
    Ok(())
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
