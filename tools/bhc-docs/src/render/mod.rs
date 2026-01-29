//! Documentation rendering.
//!
//! This module renders the documentation model to various output formats:
//! - HTML (with syntax highlighting and interactivity)
//! - Markdown
//! - JSON (for tooling integration)

pub mod html;
pub mod json;
pub mod markdown;

use crate::model::ModuleDoc;
use anyhow::Result;
use std::path::Path;

/// Output format for documentation.
#[derive(Debug, Clone, Copy)]
pub enum Format {
    Html,
    Markdown,
    Json,
}

/// Render documentation to the specified format.
pub fn render(
    docs: &[ModuleDoc],
    output: &Path,
    format: Format,
    config: &RenderConfig,
) -> Result<()> {
    std::fs::create_dir_all(output)?;

    match format {
        Format::Html => html::render(docs, output, config),
        Format::Markdown => markdown::render(docs, output, config),
        Format::Json => json::render(docs, output, config),
    }
}

/// Configuration for rendering.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Base URL for links.
    pub base_url: Option<String>,
    /// Enable playground integration.
    pub playground: bool,
    /// Syntax highlighting theme.
    pub theme: String,
    /// Current documentation version.
    pub version: Option<String>,
    /// All available versions (for version selector).
    pub versions: Vec<String>,
    /// Base URL for source code links (e.g., "https://github.com/raskell-io/bhc/blob/main").
    pub source_base_url: Option<String>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            base_url: None,
            playground: false,
            theme: "Solarized (dark)".to_string(),
            version: None,
            versions: Vec::new(),
            source_base_url: None,
        }
    }
}
