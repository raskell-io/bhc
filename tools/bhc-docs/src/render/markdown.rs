//! Markdown documentation renderer.

use anyhow::Result;
use std::path::Path;

use super::RenderConfig;
use crate::model::{DocItem, ModuleDoc};

/// Render documentation to Markdown.
pub fn render(docs: &[ModuleDoc], output: &Path, _config: &RenderConfig) -> Result<()> {
    for module in docs {
        let md = render_module(module);
        let filename = format!("{}.md", module.name.replace('.', "-"));
        std::fs::write(output.join(filename), md)?;
    }

    Ok(())
}

fn render_module(module: &ModuleDoc) -> String {
    let mut md = String::new();

    // Module header
    md.push_str(&format!("# {}\n\n", module.name));

    if let Some(doc) = &module.doc {
        md.push_str(&doc.description);
        md.push_str("\n\n");
    }

    // Functions
    let functions: Vec<_> = module
        .items
        .iter()
        .filter_map(|i| {
            if let DocItem::Function(f) = i {
                Some(f)
            } else {
                None
            }
        })
        .collect();

    if !functions.is_empty() {
        md.push_str("## Functions\n\n");
        for func in functions {
            md.push_str(&format!("### `{}`\n\n", func.name));
            md.push_str(&format!(
                "```haskell\n{} :: {}\n```\n\n",
                func.name, func.signature
            ));
            if let Some(doc) = &func.doc {
                md.push_str(&doc.description);
                md.push_str("\n\n");
            }
        }
    }

    // Types
    let types: Vec<_> = module
        .items
        .iter()
        .filter_map(|i| {
            if let DocItem::Type(t) = i {
                Some(t)
            } else {
                None
            }
        })
        .collect();

    if !types.is_empty() {
        md.push_str("## Types\n\n");
        for ty in types {
            md.push_str(&format!("### `{}`\n\n", ty.name));
            if let Some(doc) = &ty.doc {
                md.push_str(&doc.description);
                md.push_str("\n\n");
            }
        }
    }

    // Classes
    let classes: Vec<_> = module
        .items
        .iter()
        .filter_map(|i| {
            if let DocItem::Class(c) = i {
                Some(c)
            } else {
                None
            }
        })
        .collect();

    if !classes.is_empty() {
        md.push_str("## Type Classes\n\n");
        for class in classes {
            md.push_str(&format!("### `{}`\n\n", class.name));
            if let Some(doc) = &class.doc {
                md.push_str(&doc.description);
                md.push_str("\n\n");
            }
        }
    }

    md
}
