//! Documentation coverage reporting.
//!
//! This module analyzes documentation coverage and produces reports
//! showing what percentage of functions, types, and other items
//! are documented.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::extract;
use crate::model::{DocItem, ModuleDoc};

/// Coverage configuration.
pub struct CoverageConfig {
    /// Input directory or file.
    pub input: PathBuf,
    /// Minimum coverage threshold (fail if below).
    pub threshold: Option<u8>,
}

/// Coverage report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    /// Overall coverage percentage.
    pub total_coverage: f64,
    /// Total items.
    pub total_items: usize,
    /// Documented items.
    pub documented_items: usize,
    /// Per-module coverage.
    pub modules: Vec<ModuleCoverage>,
    /// Undocumented items.
    pub undocumented: Vec<UndocumentedItem>,
    /// Whether the threshold was met.
    pub passed: bool,
}

/// Coverage for a single module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleCoverage {
    /// Module name.
    pub name: String,
    /// Coverage percentage.
    pub coverage: f64,
    /// Total items.
    pub total: usize,
    /// Documented items.
    pub documented: usize,
}

/// An undocumented item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndocumentedItem {
    /// Module name.
    pub module: String,
    /// Item name.
    pub name: String,
    /// Item kind.
    pub kind: String,
    /// Source location.
    pub location: Option<String>,
}

impl fmt::Display for CoverageReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Documentation Coverage Report")?;
        writeln!(f, "==============================")?;
        writeln!(f)?;
        writeln!(
            f,
            "Overall: {:.1}% ({}/{})",
            self.total_coverage * 100.0,
            self.documented_items,
            self.total_items
        )?;
        writeln!(f)?;

        if !self.modules.is_empty() {
            writeln!(f, "By Module:")?;
            for module in &self.modules {
                let bar = coverage_bar(module.coverage);
                writeln!(
                    f,
                    "  {} {} {:.1}% ({}/{})",
                    module.name,
                    bar,
                    module.coverage * 100.0,
                    module.documented,
                    module.total
                )?;
            }
            writeln!(f)?;
        }

        if !self.undocumented.is_empty() {
            writeln!(f, "Undocumented Items:")?;
            for item in &self.undocumented {
                writeln!(f, "  - {}.{} ({})", item.module, item.name, item.kind)?;
            }
            writeln!(f)?;
        }

        if self.passed {
            writeln!(f, "PASSED")?;
        } else {
            writeln!(f, "FAILED: Below threshold")?;
        }

        Ok(())
    }
}

fn coverage_bar(coverage: f64) -> String {
    let filled = (coverage * 20.0) as usize;
    let empty = 20 - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

/// Run coverage analysis.
pub fn run(config: CoverageConfig) -> Result<CoverageReport> {
    // Find source files
    let source_files = find_source_files(&config.input)?;

    // Extract documentation from each file
    let mut docs: Vec<ModuleDoc> = Vec::new();
    for file in &source_files {
        match extract::extract_file(file) {
            Ok(module_doc) => docs.push(module_doc),
            Err(e) => tracing::warn!("Failed to process {:?}: {}", file, e),
        }
    }

    // Calculate coverage
    let mut total_items = 0;
    let mut documented_items = 0;
    let mut modules = Vec::new();
    let mut undocumented = Vec::new();

    for module in &docs {
        let mut mod_total = 0;
        let mut mod_documented = 0;

        for item in &module.items {
            mod_total += 1;
            if item.is_documented() {
                mod_documented += 1;
            } else {
                undocumented.push(UndocumentedItem {
                    module: module.name.clone(),
                    name: item.name().to_string(),
                    kind: match item {
                        DocItem::Function(_) => "function",
                        DocItem::Type(_) => "type",
                        DocItem::TypeAlias(_) => "type alias",
                        DocItem::Newtype(_) => "newtype",
                        DocItem::Class(_) => "class",
                        DocItem::Instance(_) => "instance",
                    }
                    .to_string(),
                    location: None,
                });
            }
        }

        total_items += mod_total;
        documented_items += mod_documented;

        if mod_total > 0 {
            modules.push(ModuleCoverage {
                name: module.name.clone(),
                coverage: mod_documented as f64 / mod_total as f64,
                total: mod_total,
                documented: mod_documented,
            });
        }
    }

    let total_coverage = if total_items > 0 {
        documented_items as f64 / total_items as f64
    } else {
        1.0
    };

    let passed = config
        .threshold
        .map(|t| (total_coverage * 100.0) >= t as f64)
        .unwrap_or(true);

    // Sort modules by coverage (ascending)
    modules.sort_by(|a, b| {
        a.coverage
            .partial_cmp(&b.coverage)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(CoverageReport {
        total_coverage,
        total_items,
        documented_items,
        modules,
        undocumented,
        passed,
    })
}

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
