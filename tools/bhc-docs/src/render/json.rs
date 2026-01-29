//! JSON documentation renderer.
//!
//! Outputs the documentation model as JSON for tooling integration.

use anyhow::Result;
use std::path::Path;

use super::RenderConfig;
use crate::model::ModuleDoc;

/// Render documentation to JSON.
pub fn render(docs: &[ModuleDoc], output: &Path, _config: &RenderConfig) -> Result<()> {
    // Write individual module files
    for module in docs {
        let filename = format!("{}.json", module.name.replace('.', "-"));
        let json = serde_json::to_string_pretty(module)?;
        std::fs::write(output.join(filename), json)?;
    }

    // Write combined index
    let index = serde_json::to_string_pretty(docs)?;
    std::fs::write(output.join("index.json"), index)?;

    Ok(())
}
