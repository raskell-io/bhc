//! Documentation build system.
//!
//! This module orchestrates the documentation generation process:
//! 1. Find source files
//! 2. Extract documentation from each file
//! 3. Render to the specified format

use anyhow::Result;
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::extract;
use crate::model::ModuleDoc;
use crate::render::{self, RenderConfig};

/// Build configuration.
pub struct BuildConfig {
    /// Input directory or file.
    pub input: PathBuf,
    /// Output directory.
    pub output: PathBuf,
    /// Output format.
    pub format: Format,
    /// Enable playground integration.
    pub playground: bool,
    /// Base URL for cross-referencing.
    pub base_url: Option<String>,
    /// Documentation version.
    pub version: Option<String>,
    /// Base URL for source code links.
    pub source_url: Option<String>,
}

/// Output format.
#[derive(Debug, Clone, Copy)]
pub enum Format {
    /// HTML format for web viewing.
    Html,
    /// Markdown format.
    Markdown,
    /// JSON format for programmatic access.
    Json,
}

/// Run the documentation build.
pub fn run(config: BuildConfig) -> Result<()> {
    tracing::info!("Building documentation...");

    // Find source files
    let source_files = find_source_files(&config.input)?;
    tracing::info!("Found {} source files", source_files.len());

    // Extract documentation from each file
    let mut docs: Vec<ModuleDoc> = Vec::new();
    for file in &source_files {
        tracing::debug!("Processing {:?}", file);
        match extract::extract_file(file) {
            Ok(module_doc) => docs.push(module_doc),
            Err(e) => tracing::warn!("Failed to process {:?}: {}", file, e),
        }
    }

    tracing::info!("Extracted documentation from {} modules", docs.len());

    // Render
    let render_config = RenderConfig {
        base_url: config.base_url,
        playground: config.playground,
        theme: "Solarized (dark)".to_string(),
        version: config.version,
        versions: Vec::new(), // TODO: discover versions from directory structure
        source_base_url: config.source_url,
    };

    let format = match config.format {
        Format::Html => render::Format::Html,
        Format::Markdown => render::Format::Markdown,
        Format::Json => render::Format::Json,
    };

    render::render(&docs, &config.output, format, &render_config)?;

    tracing::info!("Documentation written to {:?}", config.output);

    Ok(())
}

/// Find all Haskell source files in a directory.
fn find_source_files(path: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        if is_haskell_file(path) {
            files.push(path.clone());
        }
    } else if path.is_dir() {
        for entry in WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() && is_haskell_file(path) {
                files.push(path.to_path_buf());
            }
        }
    }

    Ok(files)
}

fn is_haskell_file(path: &std::path::Path) -> bool {
    path.extension()
        .map(|ext| ext == "hs" || ext == "lhs")
        .unwrap_or(false)
}
