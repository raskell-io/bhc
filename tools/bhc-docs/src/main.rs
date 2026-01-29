//! bhc-docs: Documentation generator for BHC.
//!
//! A modern documentation tool for BHC that generates beautiful HTML
//! documentation with type search, interactive examples, and BHC-specific
//! features like fusion annotations and profile behavior.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "bhc-docs")]
#[command(author, version, about = "Documentation generator for BHC", long_about = None)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build documentation from source files
    Build {
        /// Input directory or file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = "docs")]
        output: PathBuf,

        /// Output format
        #[arg(short, long, default_value = "html")]
        format: OutputFormat,

        /// Enable playground integration for runnable examples
        #[arg(long)]
        playground: bool,

        /// Base URL for cross-referencing
        #[arg(long)]
        base_url: Option<String>,

        /// Documentation version (shown in header and version selector)
        #[arg(long)]
        version: Option<String>,

        /// Base URL for source code links (e.g., "https://github.com/raskell-io/bhc/blob/main")
        #[arg(long)]
        source_url: Option<String>,
    },

    /// Serve documentation with live reload
    Serve {
        /// Documentation directory
        #[arg(value_name = "DIR", default_value = "docs")]
        dir: PathBuf,

        /// Port to listen on
        #[arg(short, long, default_value = "3000")]
        port: u16,

        /// Watch for changes and auto-reload
        #[arg(short, long)]
        watch: bool,

        /// Source directory to watch (for rebuilding)
        #[arg(long)]
        source: Option<PathBuf>,
    },

    /// Search documentation by type signature
    Search {
        /// Documentation directory
        #[arg(value_name = "DIR")]
        dir: PathBuf,

        /// Search query (e.g., "a -> [a] -> [a]")
        #[arg(value_name = "QUERY")]
        query: String,

        /// Enable type search (unify type variables)
        #[arg(long)]
        type_search: bool,

        /// Maximum results to return
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Report documentation coverage
    Coverage {
        /// Input directory or file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Minimum coverage percentage (fail if below)
        #[arg(long)]
        threshold: Option<u8>,

        /// Output format for report
        #[arg(short, long, default_value = "text")]
        format: CoverageFormat,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    /// HTML documentation
    Html,
    /// Markdown files
    Markdown,
    /// JSON (for tooling integration)
    Json,
}

#[derive(Clone, Copy, ValueEnum)]
enum CoverageFormat {
    /// Human-readable text
    Text,
    /// JSON for CI integration
    Json,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    match cli.command {
        Commands::Build {
            input,
            output,
            format,
            playground,
            base_url,
            version,
            source_url,
        } => {
            tracing::info!("Building documentation from {:?}", input);
            bhc_docs::build::run(bhc_docs::build::BuildConfig {
                input,
                output,
                format: match format {
                    OutputFormat::Html => bhc_docs::build::Format::Html,
                    OutputFormat::Markdown => bhc_docs::build::Format::Markdown,
                    OutputFormat::Json => bhc_docs::build::Format::Json,
                },
                playground,
                base_url,
                version,
                source_url,
            })?;
        }

        Commands::Serve {
            dir,
            port,
            watch,
            source,
        } => {
            tracing::info!("Serving documentation from {:?} on port {}", dir, port);
            bhc_docs::serve::run(bhc_docs::serve::ServeConfig {
                dir,
                port,
                watch,
                source,
            })?;
        }

        Commands::Search {
            dir,
            query,
            type_search,
            limit,
        } => {
            let results = bhc_docs::search::run(bhc_docs::search::SearchConfig {
                dir,
                query,
                type_search,
                limit,
            })?;
            for result in results {
                println!("{}", result);
            }
        }

        Commands::Coverage {
            input,
            threshold,
            format,
        } => {
            let report =
                bhc_docs::coverage::run(bhc_docs::coverage::CoverageConfig { input, threshold })?;

            match format {
                CoverageFormat::Text => println!("{}", report),
                CoverageFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
            }
        }
    }

    Ok(())
}
