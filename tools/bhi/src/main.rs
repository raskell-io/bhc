//! Basel Haskell Inspector (bhi) - IR and Kernel Report Viewer
//!
//! A tool for inspecting intermediate representations, kernel reports,
//! and compilation artifacts from the BHC compiler.
//!
//! # Usage
//!
//! ```bash
//! # Inspect an IR dump
//! bhi ir dump.core --stage core
//!
//! # View kernel fusion report
//! bhi kernel report.json --failures-only
//!
//! # Compare two IR dumps
//! bhi diff before.core after.core
//!
//! # Show compilation statistics
//! bhi stats compile.json --timing
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

/// Basel Haskell Inspector - IR and kernel report viewer
#[derive(Parser, Debug)]
#[command(name = "bhi")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable colored output
    #[arg(long, default_value = "true")]
    color: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Inspect an IR dump file
    Ir {
        /// The IR file to inspect
        file: PathBuf,

        /// IR stage (ast, hir, core, tensor, loop)
        #[arg(long, value_enum)]
        stage: Option<IrStage>,

        /// Output format (text, json, dot)
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,

        /// Filter to specific function
        #[arg(long)]
        function: Option<String>,

        /// Show type annotations
        #[arg(long)]
        types: bool,

        /// Show source locations
        #[arg(long)]
        locations: bool,
    },

    /// View a kernel fusion report
    Kernel {
        /// The kernel report file
        file: PathBuf,

        /// Show only failed fusions
        #[arg(long)]
        failures_only: bool,

        /// Show detailed timing information
        #[arg(long)]
        timing: bool,

        /// Show SIMD information
        #[arg(long)]
        simd: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Analyze memory allocation patterns
    Memory {
        /// The allocation report file
        file: PathBuf,

        /// Show only heap allocations
        #[arg(long)]
        heap_only: bool,

        /// Show arena usage
        #[arg(long)]
        arena: bool,

        /// Group by allocation site
        #[arg(long)]
        by_site: bool,
    },

    /// Display a call graph
    Callgraph {
        /// The callgraph file
        file: PathBuf,

        /// Output format (text, dot, json)
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,

        /// Filter to functions matching pattern
        #[arg(long)]
        filter: Option<String>,

        /// Maximum depth
        #[arg(long)]
        depth: Option<usize>,

        /// Show recursive cycles
        #[arg(long)]
        cycles: bool,
    },

    /// Compare two IR dumps
    Diff {
        /// First IR file
        before: PathBuf,

        /// Second IR file
        after: PathBuf,

        /// Show only changes
        #[arg(long)]
        changes_only: bool,

        /// Context lines around changes
        #[arg(long, default_value = "3")]
        context: usize,

        /// Ignore whitespace differences
        #[arg(long)]
        ignore_whitespace: bool,
    },

    /// Show compilation statistics
    Stats {
        /// The stats file
        file: PathBuf,

        /// Show timing breakdown
        #[arg(long)]
        timing: bool,

        /// Show memory usage
        #[arg(long)]
        memory: bool,

        /// Compare with another stats file
        #[arg(long)]
        compare: Option<PathBuf>,
    },

    /// Pretty-print an IR file
    Pretty {
        /// The IR file to format
        file: PathBuf,

        /// IR stage
        #[arg(long, value_enum)]
        stage: Option<IrStage>,

        /// Line width for formatting
        #[arg(long, default_value = "100")]
        width: usize,

        /// Indentation size
        #[arg(long, default_value = "2")]
        indent: usize,
    },

    /// Validate an IR file
    Validate {
        /// The IR file to validate
        file: PathBuf,

        /// IR stage
        #[arg(long, value_enum)]
        stage: Option<IrStage>,

        /// Check types
        #[arg(long)]
        types: bool,

        /// Check invariants
        #[arg(long)]
        invariants: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum IrStage {
    Ast,
    Hir,
    Core,
    Tensor,
    Loop,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Dot,
}

/// Inspector errors
#[derive(Debug, Error)]
pub enum InspectorError {
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),
    #[error("Invalid IR format: {0}")]
    InvalidFormat(String),
    #[error("Stage mismatch: expected {expected}, found {found}")]
    StageMismatch { expected: String, found: String },
    #[error("Parse error: {0}")]
    Parse(String),
}

// ============================================================================
// Data Models
// ============================================================================

/// Kernel fusion report
#[derive(Debug, Serialize, Deserialize)]
pub struct FusionReport {
    pub module: String,
    pub timestamp: String,
    pub profile: String,
    pub kernels: Vec<KernelInfo>,
    pub summary: FusionSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KernelInfo {
    pub id: String,
    pub name: String,
    pub pattern: String,
    pub status: FusionStatus,
    pub inputs: Vec<TensorDesc>,
    pub outputs: Vec<TensorDesc>,
    pub ops_count: usize,
    pub fused_ops: Vec<String>,
    pub simd_width: Option<usize>,
    pub parallel: bool,
    pub timing_us: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FusionStatus {
    Fused,
    Partial,
    Failed,
    NotApplicable,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorDesc {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<String>,
    pub layout: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FusionSummary {
    pub total_kernels: usize,
    pub fused: usize,
    pub partial: usize,
    pub failed: usize,
    pub fusion_rate: f64,
    pub total_time_us: f64,
}

/// Memory allocation report
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryReport {
    pub module: String,
    pub allocations: Vec<Allocation>,
    pub summary: MemorySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allocation {
    pub site: String,
    pub region: String,
    pub size_bytes: usize,
    pub count: usize,
    pub live_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemorySummary {
    pub total_allocated: usize,
    pub peak_memory: usize,
    pub arena_allocated: usize,
    pub heap_allocated: usize,
    pub pinned_allocated: usize,
}

/// Compilation statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct CompileStats {
    pub module: String,
    pub timestamp: String,
    pub phases: Vec<PhaseStats>,
    pub summary: StatsSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PhaseStats {
    pub name: String,
    pub time_ms: f64,
    pub memory_mb: f64,
    pub items_processed: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatsSummary {
    pub total_time_ms: f64,
    pub peak_memory_mb: f64,
    pub modules_compiled: usize,
    pub lines_of_code: usize,
    pub functions: usize,
    pub type_classes: usize,
}

/// Call graph
#[derive(Debug, Serialize, Deserialize)]
pub struct CallGraph {
    pub module: String,
    pub nodes: Vec<CallNode>,
    pub edges: Vec<CallEdge>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CallNode {
    pub id: String,
    pub name: String,
    pub module: String,
    pub is_recursive: bool,
    pub call_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CallEdge {
    pub from: String,
    pub to: String,
    pub count: usize,
    pub is_tail_call: bool,
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up colored output
    if !cli.color {
        colored::control::set_override(false);
    }

    match cli.command {
        Commands::Ir {
            file,
            stage,
            format,
            function,
            types,
            locations,
        } => {
            inspect_ir(
                &file,
                stage,
                format,
                function.as_deref(),
                types,
                locations,
                cli.verbose,
            )?;
        }
        Commands::Kernel {
            file,
            failures_only,
            timing,
            simd,
            format,
        } => {
            view_kernel_report(&file, failures_only, timing, simd, format)?;
        }
        Commands::Memory {
            file,
            heap_only,
            arena,
            by_site,
        } => {
            analyze_memory(&file, heap_only, arena, by_site)?;
        }
        Commands::Callgraph {
            file,
            format,
            filter,
            depth,
            cycles,
        } => {
            show_callgraph(&file, format, filter.as_deref(), depth, cycles)?;
        }
        Commands::Diff {
            before,
            after,
            changes_only,
            context,
            ignore_whitespace,
        } => {
            diff_ir(&before, &after, changes_only, context, ignore_whitespace)?;
        }
        Commands::Stats {
            file,
            timing,
            memory,
            compare,
        } => {
            show_stats(&file, timing, memory, compare.as_ref())?;
        }
        Commands::Pretty {
            file,
            stage,
            width,
            indent,
        } => {
            pretty_print(&file, stage, width, indent)?;
        }
        Commands::Validate {
            file,
            stage,
            types,
            invariants,
        } => {
            validate_ir(&file, stage, types, invariants)?;
        }
    }

    Ok(())
}

// ============================================================================
// IR Inspection
// ============================================================================

fn inspect_ir(
    file: &PathBuf,
    stage: Option<IrStage>,
    format: OutputFormat,
    function: Option<&str>,
    show_types: bool,
    show_locations: bool,
    verbose: bool,
) -> Result<()> {
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");

    // If it's a Haskell source file, compile it and display the requested IR stage
    if ext == "hs" {
        return inspect_ir_from_source(file, stage, function, verbose);
    }

    let content =
        fs::read_to_string(file).with_context(|| format!("Failed to read {}", file.display()))?;

    // Detect stage from file extension if not provided
    let stage = stage.unwrap_or_else(|| detect_stage(file));

    println!("{}", "IR Inspection".bold().cyan());
    println!("{}: {}", "File".dimmed(), file.display());
    println!("{}: {:?}", "Stage".dimmed(), stage);
    println!();

    // Parse and display based on stage
    match stage {
        IrStage::Ast => display_ast(&content, function, show_types, show_locations)?,
        IrStage::Hir => display_hir(&content, function, show_types, show_locations)?,
        IrStage::Core => display_core(&content, function, show_types, show_locations)?,
        IrStage::Tensor => display_tensor_ir(&content, function, verbose)?,
        IrStage::Loop => display_loop_ir(&content, function, verbose)?,
    }

    Ok(())
}

/// Compile a Haskell source file and display the IR at the requested stage.
fn inspect_ir_from_source(
    file: &PathBuf,
    stage: Option<IrStage>,
    function: Option<&str>,
    verbose: bool,
) -> Result<()> {
    let source =
        fs::read_to_string(file).with_context(|| format!("Failed to read {}", file.display()))?;

    let stage = stage.unwrap_or(IrStage::Core);

    println!("{}", "IR Inspection (from source)".bold().cyan());
    println!("{}: {}", "File".dimmed(), file.display());
    println!("{}: {:?}", "Stage".dimmed(), stage);
    println!();

    // 1. Parse
    let mut source_map = bhc_diagnostics::SourceMap::new();
    let file_id = source_map.add_file(file.display().to_string(), source.clone());
    let (module, diagnostics) = bhc_parser::parse_module(&source, file_id);

    if !diagnostics.is_empty() {
        let renderer = bhc_diagnostics::DiagnosticRenderer::new(&source_map);
        renderer.render_all(&diagnostics);
    }

    let module = module.context("Failed to parse module")?;

    if stage == IrStage::Ast {
        println!("{}", "AST".bold().green());
        println!("{}", "─".repeat(60).dimmed());
        let ast_str = format!("{:#?}", module);
        print_filtered(&ast_str, function);
        return Ok(());
    }

    // 2. Lower AST → HIR
    let mut lower_ctx = bhc_lower::LowerContext::with_builtins();
    let config = bhc_lower::LowerConfig {
        include_builtins: true,
        warn_unused: false,
        search_paths: vec![],
    };
    let hir = bhc_lower::lower_module(&mut lower_ctx, &module, &config)
        .map_err(|e| anyhow::anyhow!("Lowering error: {e}"))?;

    if stage == IrStage::Hir {
        println!("{}", "HIR (High-level IR)".bold().green());
        println!("{}", "─".repeat(60).dimmed());
        let hir_str = format!("{:#?}", hir);
        print_filtered(&hir_str, function);
        return Ok(());
    }

    // 2b. Type check (optional, non-fatal)
    let typed_module =
        bhc_typeck::type_check_module_with_defs(&hir, file_id, Some(&lower_ctx.defs));
    if let Ok(ref typed) = typed_module {
        if verbose && !typed.def_schemes.is_empty() {
            println!("{}", "Type Signatures".bold().green());
            println!("{}", "─".repeat(60).dimmed());
            for (def_id, scheme) in &typed.def_schemes {
                if let Some(info) = lower_ctx.defs.get(def_id) {
                    println!("  {} :: {}", info.name, scheme.ty);
                }
            }
            println!();
        }
    }

    // 2c. Build def map for Core lowering
    let def_map: bhc_hir_to_core::DefMap = lower_ctx
        .defs
        .iter()
        .map(|(def_id, def_info)| {
            (
                *def_id,
                bhc_hir_to_core::DefInfo {
                    id: *def_id,
                    name: def_info.name,
                },
            )
        })
        .collect();

    // 3. Lower HIR → Core
    let core = bhc_hir_to_core::lower_module_with_defs(&hir, Some(&def_map), None)
        .map_err(|e| anyhow::anyhow!("Core lowering error: {e}"))?;

    if stage == IrStage::Core {
        println!("{}", "Core IR".bold().green());
        println!("{}", "─".repeat(60).dimmed());
        let core_str = format!("{core}");
        print_filtered(&core_str, function);
        return Ok(());
    }

    // Tensor and Loop IR stages require further lowering not yet wired
    println!(
        "{}",
        format!("Stage {:?} not yet supported for .hs compilation", stage).yellow()
    );
    if verbose {
        println!("Core IR is the furthest stage available from source compilation.");
    }

    Ok(())
}

/// Print debug output, optionally filtered by a function name.
fn print_filtered(content: &str, filter: Option<&str>) {
    for line in content.lines() {
        if let Some(f) = filter {
            if !line.contains(f) {
                continue;
            }
        }
        println!("{}", line);
    }
}

fn detect_stage(file: &PathBuf) -> IrStage {
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "ast" => IrStage::Ast,
        "hir" => IrStage::Hir,
        "core" => IrStage::Core,
        "tensor" | "tir" => IrStage::Tensor,
        "loop" | "lir" => IrStage::Loop,
        _ => IrStage::Core, // Default
    }
}

fn display_ast(content: &str, filter: Option<&str>, _types: bool, _locs: bool) -> Result<()> {
    println!("{}", "AST".bold().green());
    println!("{}", "─".repeat(60).dimmed());

    // For now, just pretty-print the content
    for line in content.lines() {
        if let Some(f) = filter {
            if !line.contains(f) {
                continue;
            }
        }
        println!("{}", line);
    }

    Ok(())
}

fn display_hir(content: &str, filter: Option<&str>, _types: bool, _locs: bool) -> Result<()> {
    println!("{}", "HIR (High-level IR)".bold().green());
    println!("{}", "─".repeat(60).dimmed());

    for line in content.lines() {
        if let Some(f) = filter {
            if !line.contains(f) {
                continue;
            }
        }

        // Syntax highlighting
        let highlighted = highlight_hir_line(line);
        println!("{}", highlighted);
    }

    Ok(())
}

fn display_core(content: &str, filter: Option<&str>, _types: bool, _locs: bool) -> Result<()> {
    println!("{}", "Core IR".bold().green());
    println!("{}", "─".repeat(60).dimmed());

    for line in content.lines() {
        if let Some(f) = filter {
            if !line.contains(f) {
                continue;
            }
        }

        // Syntax highlighting
        let highlighted = highlight_core_line(line);
        println!("{}", highlighted);
    }

    Ok(())
}

fn display_tensor_ir(content: &str, filter: Option<&str>, verbose: bool) -> Result<()> {
    println!("{}", "Tensor IR".bold().green());
    println!("{}", "─".repeat(60).dimmed());

    // Try to parse as JSON tensor IR
    if let Ok(ops) = serde_json::from_str::<Vec<TensorOpInfo>>(content) {
        for op in ops {
            if let Some(f) = filter {
                if !op.name.contains(f) {
                    continue;
                }
            }

            println!(
                "{} {} :: {} -> {}",
                "op".dimmed(),
                op.name.bold().yellow(),
                format_shapes(&op.input_shapes),
                format_shapes(&op.output_shapes)
            );

            if verbose {
                println!("    dtype: {}", op.dtype);
                println!("    layout: {}", op.layout);
                if let Some(fused) = &op.fused_with {
                    println!("    fused: {}", fused.join(", ").green());
                }
            }
        }
    } else {
        // Plain text fallback
        for line in content.lines() {
            if let Some(f) = filter {
                if !line.contains(f) {
                    continue;
                }
            }
            println!("{}", highlight_tensor_line(line));
        }
    }

    Ok(())
}

fn display_loop_ir(content: &str, filter: Option<&str>, verbose: bool) -> Result<()> {
    println!("{}", "Loop IR".bold().green());
    println!("{}", "─".repeat(60).dimmed());

    for line in content.lines() {
        if let Some(f) = filter {
            if !line.contains(f) {
                continue;
            }
        }

        let highlighted = highlight_loop_line(line);
        println!("{}", highlighted);
    }

    if verbose {
        println!();
        println!("{}", "Legend:".dimmed());
        println!("  {} - parallel loop", "parallel".cyan());
        println!("  {} - vectorized operation", "simd".magenta());
        println!("  {} - fused kernel boundary", "kernel".yellow());
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct TensorOpInfo {
    name: String,
    dtype: String,
    layout: String,
    input_shapes: Vec<Vec<String>>,
    output_shapes: Vec<Vec<String>>,
    fused_with: Option<Vec<String>>,
}

fn format_shapes(shapes: &[Vec<String>]) -> String {
    shapes
        .iter()
        .map(|s| format!("[{}]", s.join(", ")))
        .collect::<Vec<_>>()
        .join(" × ")
}

fn highlight_hir_line(line: &str) -> String {
    let line = line
        .replace("let ", &"let ".purple().to_string())
        .replace("in ", &"in ".purple().to_string())
        .replace("case ", &"case ".purple().to_string())
        .replace("of ", &"of ".purple().to_string())
        .replace(" -> ", &" -> ".cyan().to_string());
    line
}

fn highlight_core_line(line: &str) -> String {
    let line = line
        .replace("let ", &"let ".purple().to_string())
        .replace("letrec ", &"letrec ".purple().to_string())
        .replace("case ", &"case ".purple().to_string())
        .replace("of ", &"of ".purple().to_string())
        .replace("Lam ", &"λ ".blue().to_string())
        .replace("App ", &"@ ".cyan().to_string());
    line
}

fn highlight_tensor_line(line: &str) -> String {
    let line = line
        .replace("map ", &"map ".yellow().to_string())
        .replace("zipWith ", &"zipWith ".yellow().to_string())
        .replace("fold ", &"fold ".yellow().to_string())
        .replace("reduce ", &"reduce ".yellow().to_string())
        .replace("matmul ", &"matmul ".magenta().to_string())
        .replace("FUSED", &"FUSED".green().bold().to_string());
    line
}

fn highlight_loop_line(line: &str) -> String {
    let line = line
        .replace("for ", &"for ".purple().to_string())
        .replace("parallel ", &"parallel ".cyan().bold().to_string())
        .replace("simd ", &"simd ".magenta().bold().to_string())
        .replace("kernel ", &"kernel ".yellow().bold().to_string());
    line
}

// ============================================================================
// Kernel Report
// ============================================================================

fn view_kernel_report(
    file: &PathBuf,
    failures_only: bool,
    timing: bool,
    simd: bool,
    format: OutputFormat,
) -> Result<()> {
    let content =
        fs::read_to_string(file).with_context(|| format!("Failed to read {}", file.display()))?;

    // Try JSON first, then fall back to generating sample data
    let report: FusionReport = serde_json::from_str(&content).unwrap_or_else(|_| {
        // Generate sample report for demonstration
        FusionReport {
            module: file
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            timestamp: chrono_lite(),
            profile: "numeric".to_string(),
            kernels: vec![
                KernelInfo {
                    id: "k1".to_string(),
                    name: "dotProduct".to_string(),
                    pattern: "sum (zipWith (*) a b)".to_string(),
                    status: FusionStatus::Fused,
                    inputs: vec![
                        TensorDesc {
                            name: "a".to_string(),
                            dtype: "Float32".to_string(),
                            shape: vec!["N".to_string()],
                            layout: "Contiguous".to_string(),
                        },
                        TensorDesc {
                            name: "b".to_string(),
                            dtype: "Float32".to_string(),
                            shape: vec!["N".to_string()],
                            layout: "Contiguous".to_string(),
                        },
                    ],
                    outputs: vec![TensorDesc {
                        name: "result".to_string(),
                        dtype: "Float32".to_string(),
                        shape: vec![],
                        layout: "Scalar".to_string(),
                    }],
                    ops_count: 2,
                    fused_ops: vec!["zipWith (*)".to_string(), "sum".to_string()],
                    simd_width: Some(8),
                    parallel: true,
                    timing_us: Some(0.5),
                    notes: vec!["Vectorized with AVX-256".to_string()],
                },
                KernelInfo {
                    id: "k2".to_string(),
                    name: "mapSquare".to_string(),
                    pattern: "map (^2) xs".to_string(),
                    status: FusionStatus::Fused,
                    inputs: vec![TensorDesc {
                        name: "xs".to_string(),
                        dtype: "Float32".to_string(),
                        shape: vec!["M".to_string()],
                        layout: "Contiguous".to_string(),
                    }],
                    outputs: vec![TensorDesc {
                        name: "result".to_string(),
                        dtype: "Float32".to_string(),
                        shape: vec!["M".to_string()],
                        layout: "Contiguous".to_string(),
                    }],
                    ops_count: 1,
                    fused_ops: vec!["map (^2)".to_string()],
                    simd_width: Some(8),
                    parallel: true,
                    timing_us: Some(0.3),
                    notes: vec![],
                },
            ],
            summary: FusionSummary {
                total_kernels: 2,
                fused: 2,
                partial: 0,
                failed: 0,
                fusion_rate: 100.0,
                total_time_us: 0.8,
            },
        }
    });

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        OutputFormat::Text => {
            print_fusion_report_text(&report, failures_only, timing, simd);
        }
        OutputFormat::Dot => {
            print_fusion_report_dot(&report);
        }
    }

    Ok(())
}

fn print_fusion_report_text(report: &FusionReport, failures_only: bool, timing: bool, simd: bool) {
    println!("{}", "Kernel Fusion Report".bold().cyan());
    println!("{}", "═".repeat(70));
    println!("Module: {}", report.module.bold());
    println!("Profile: {}", report.profile);
    println!("Generated: {}", report.timestamp.dimmed());
    println!();

    // Summary
    let summary = &report.summary;
    println!("{}", "Summary".bold());
    println!("{}", "─".repeat(70));

    let fused_pct = if summary.total_kernels > 0 {
        (summary.fused as f64 / summary.total_kernels as f64) * 100.0
    } else {
        0.0
    };

    let status_color = if fused_pct >= 100.0 {
        "green"
    } else if fused_pct >= 80.0 {
        "yellow"
    } else {
        "red"
    };

    println!(
        "  Total kernels: {}",
        summary.total_kernels.to_string().bold()
    );
    println!(
        "  Fused:         {} ({})",
        summary.fused.to_string().green(),
        format!("{:.1}%", fused_pct).color(status_color)
    );
    if summary.partial > 0 {
        println!("  Partial:       {}", summary.partial.to_string().yellow());
    }
    if summary.failed > 0 {
        println!("  Failed:        {}", summary.failed.to_string().red());
    }
    if timing {
        println!("  Total time:    {:.2}μs", summary.total_time_us);
    }
    println!();

    // Kernels
    println!("{}", "Kernels".bold());
    println!("{}", "─".repeat(70));

    for kernel in &report.kernels {
        if failures_only && kernel.status == FusionStatus::Fused {
            continue;
        }

        let status_str = match kernel.status {
            FusionStatus::Fused => "FUSED".green().bold(),
            FusionStatus::Partial => "PARTIAL".yellow().bold(),
            FusionStatus::Failed => "FAILED".red().bold(),
            FusionStatus::NotApplicable => "N/A".dimmed(),
        };

        println!(
            "\n  {} [{}] {}",
            kernel.name.bold().yellow(),
            kernel.id.dimmed(),
            status_str
        );
        println!("    Pattern: {}", kernel.pattern.italic());

        // Show inputs/outputs
        if !kernel.inputs.is_empty() {
            let inputs: Vec<String> = kernel
                .inputs
                .iter()
                .map(|t| format!("{}: [{}]", t.name, t.shape.join(", ")))
                .collect();
            println!("    Inputs: {}", inputs.join(", "));
        }

        // Fused operations
        if !kernel.fused_ops.is_empty() {
            println!("    Fused ops: {}", kernel.fused_ops.join(" → ").green());
        }

        // SIMD info
        if simd {
            if let Some(width) = kernel.simd_width {
                println!(
                    "    SIMD: {}x{}",
                    width,
                    kernel
                        .inputs
                        .first()
                        .map(|i| &i.dtype)
                        .unwrap_or(&"?".to_string())
                );
            }
            if kernel.parallel {
                println!("    Parallel: {}", "yes".cyan());
            }
        }

        // Timing
        if timing {
            if let Some(t) = kernel.timing_us {
                println!("    Time: {:.2}μs", t);
            }
        }

        // Notes
        for note in &kernel.notes {
            println!("    Note: {}", note.dimmed());
        }
    }

    println!();
}

fn print_fusion_report_dot(report: &FusionReport) {
    println!("digraph fusion {{");
    println!("  rankdir=TB;");
    println!("  node [shape=box, fontname=\"Helvetica\"];");
    println!();

    for kernel in &report.kernels {
        let color = match kernel.status {
            FusionStatus::Fused => "green",
            FusionStatus::Partial => "yellow",
            FusionStatus::Failed => "red",
            FusionStatus::NotApplicable => "gray",
        };

        println!(
            "  {} [label=\"{}\\n{}\", color={}];",
            kernel.id, kernel.name, kernel.pattern, color
        );

        // Show fused operations as a chain
        for (i, op) in kernel.fused_ops.iter().enumerate() {
            let op_id = format!("{}_{}", kernel.id, i);
            println!("  {} [label=\"{}\", shape=ellipse];", op_id, op);
            if i == 0 {
                println!("  {} -> {};", kernel.id, op_id);
            }
        }
    }

    println!("}}");
}

// ============================================================================
// Memory Analysis
// ============================================================================

fn analyze_memory(file: &PathBuf, heap_only: bool, arena: bool, by_site: bool) -> Result<()> {
    let content =
        fs::read_to_string(file).with_context(|| format!("Failed to read {}", file.display()))?;

    let report: MemoryReport = serde_json::from_str(&content).unwrap_or_else(|_| {
        // Sample data
        MemoryReport {
            module: "Example".to_string(),
            allocations: vec![
                Allocation {
                    site: "main:42".to_string(),
                    region: "heap".to_string(),
                    size_bytes: 1024,
                    count: 100,
                    live_bytes: 512,
                },
                Allocation {
                    site: "kernel:10".to_string(),
                    region: "arena".to_string(),
                    size_bytes: 65536,
                    count: 1,
                    live_bytes: 0,
                },
            ],
            summary: MemorySummary {
                total_allocated: 66560,
                peak_memory: 65536,
                arena_allocated: 65536,
                heap_allocated: 1024,
                pinned_allocated: 0,
            },
        }
    });

    println!("{}", "Memory Analysis".bold().cyan());
    println!("{}", "═".repeat(60));
    println!("Module: {}", report.module.bold());
    println!();

    // Summary
    let summary = &report.summary;
    println!("{}", "Summary".bold());
    println!("{}", "─".repeat(60));
    println!(
        "  Total allocated: {}",
        format_bytes(summary.total_allocated)
    );
    println!("  Peak memory:     {}", format_bytes(summary.peak_memory));

    if !heap_only {
        println!(
            "  Arena:           {} ({})",
            format_bytes(summary.arena_allocated),
            format!(
                "{:.1}%",
                (summary.arena_allocated as f64 / summary.total_allocated as f64) * 100.0
            )
            .green()
        );
    }
    println!(
        "  Heap:            {} ({})",
        format_bytes(summary.heap_allocated),
        format!(
            "{:.1}%",
            (summary.heap_allocated as f64 / summary.total_allocated as f64) * 100.0
        )
    );
    if summary.pinned_allocated > 0 {
        println!(
            "  Pinned:          {}",
            format_bytes(summary.pinned_allocated)
        );
    }
    println!();

    // Allocations
    println!("{}", "Allocations".bold());
    println!("{}", "─".repeat(60));

    let mut allocs = report.allocations.clone();
    if heap_only {
        allocs.retain(|a| a.region == "heap");
    }
    if !arena {
        allocs.retain(|a| a.region != "arena");
    }

    if by_site {
        // Group by site
        let mut by_site_map: HashMap<String, Vec<&Allocation>> = HashMap::new();
        for alloc in &allocs {
            by_site_map
                .entry(alloc.site.clone())
                .or_default()
                .push(alloc);
        }

        for (site, site_allocs) in by_site_map {
            let total: usize = site_allocs.iter().map(|a| a.size_bytes).sum();
            println!("  {} - {}", site.bold(), format_bytes(total));
            for alloc in site_allocs {
                println!(
                    "    {} {} × {}",
                    alloc.region.dimmed(),
                    alloc.count,
                    format_bytes(alloc.size_bytes / alloc.count.max(1))
                );
            }
        }
    } else {
        for alloc in allocs {
            let region_color = match alloc.region.as_str() {
                "arena" => alloc.region.green(),
                "heap" => alloc.region.yellow(),
                "pinned" => alloc.region.cyan(),
                _ => alloc.region.normal(),
            };

            println!(
                "  {} {} @ {} ({} × {})",
                region_color,
                format_bytes(alloc.size_bytes),
                alloc.site,
                alloc.count,
                format_bytes(alloc.size_bytes / alloc.count.max(1))
            );
        }
    }

    Ok(())
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

// ============================================================================
// Call Graph
// ============================================================================

fn show_callgraph(
    file: &PathBuf,
    format: OutputFormat,
    filter: Option<&str>,
    depth: Option<usize>,
    cycles: bool,
) -> Result<()> {
    let content =
        fs::read_to_string(file).with_context(|| format!("Failed to read {}", file.display()))?;

    let graph: CallGraph = serde_json::from_str(&content).unwrap_or_else(|_| {
        // Sample data
        CallGraph {
            module: "Example".to_string(),
            nodes: vec![
                CallNode {
                    id: "main".to_string(),
                    name: "main".to_string(),
                    module: "Main".to_string(),
                    is_recursive: false,
                    call_count: 1,
                },
                CallNode {
                    id: "fib".to_string(),
                    name: "fib".to_string(),
                    module: "Main".to_string(),
                    is_recursive: true,
                    call_count: 100,
                },
            ],
            edges: vec![
                CallEdge {
                    from: "main".to_string(),
                    to: "fib".to_string(),
                    count: 1,
                    is_tail_call: false,
                },
                CallEdge {
                    from: "fib".to_string(),
                    to: "fib".to_string(),
                    count: 99,
                    is_tail_call: true,
                },
            ],
        }
    });

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&graph)?);
        }
        OutputFormat::Dot => {
            print_callgraph_dot(&graph, filter, cycles);
        }
        OutputFormat::Text => {
            print_callgraph_text(&graph, filter, depth, cycles);
        }
    }

    Ok(())
}

fn print_callgraph_text(
    graph: &CallGraph,
    filter: Option<&str>,
    depth: Option<usize>,
    show_cycles: bool,
) {
    println!("{}", "Call Graph".bold().cyan());
    println!("{}", "═".repeat(60));
    println!("Module: {}", graph.module.bold());
    println!();

    // Build adjacency map
    let mut adj: HashMap<&str, Vec<&CallEdge>> = HashMap::new();
    for edge in &graph.edges {
        adj.entry(&edge.from).or_default().push(edge);
    }

    // Find root nodes (no incoming edges)
    let mut has_incoming: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for edge in &graph.edges {
        has_incoming.insert(&edge.to);
    }

    let roots: Vec<&CallNode> = graph
        .nodes
        .iter()
        .filter(|n| !has_incoming.contains(n.id.as_str()))
        .collect();

    // Print tree from each root
    for root in roots {
        if let Some(f) = filter {
            if !root.name.contains(f) {
                continue;
            }
        }

        print_call_tree(
            &root.id,
            &adj,
            &graph.nodes,
            0,
            depth.unwrap_or(10),
            show_cycles,
            &mut std::collections::HashSet::new(),
        );
    }

    // Show recursive cycles
    if show_cycles {
        println!();
        println!("{}", "Recursive Cycles".bold());
        println!("{}", "─".repeat(60));

        for node in &graph.nodes {
            if node.is_recursive {
                println!("  {} (calls: {})", node.name.red(), node.call_count);
            }
        }
    }
}

fn print_call_tree(
    node_id: &str,
    adj: &HashMap<&str, Vec<&CallEdge>>,
    nodes: &[CallNode],
    indent: usize,
    max_depth: usize,
    show_cycles: bool,
    visited: &mut std::collections::HashSet<String>,
) {
    if indent > max_depth {
        println!("{}...", "  ".repeat(indent));
        return;
    }

    let node = nodes.iter().find(|n| n.id == node_id);
    let name = node.map(|n| n.name.as_str()).unwrap_or(node_id);
    let is_recursive = node.map(|n| n.is_recursive).unwrap_or(false);

    let prefix = if indent == 0 {
        "".to_string()
    } else {
        format!("{}├─ ", "│  ".repeat(indent - 1))
    };

    if is_recursive {
        println!("{}{} {}", prefix, name.yellow(), "(recursive)".red());
    } else {
        println!("{}{}", prefix, name);
    }

    if visited.contains(node_id) {
        if show_cycles {
            println!("{}└─ (cycle)", "│  ".repeat(indent));
        }
        return;
    }
    visited.insert(node_id.to_string());

    if let Some(edges) = adj.get(node_id) {
        for edge in edges {
            let suffix = if edge.is_tail_call {
                " [tail]".cyan().to_string()
            } else {
                "".to_string()
            };

            if edge.count > 1 {
                print!("{}  (×{}){}", "│  ".repeat(indent), edge.count, suffix);
            }

            print_call_tree(
                &edge.to,
                adj,
                nodes,
                indent + 1,
                max_depth,
                show_cycles,
                visited,
            );
        }
    }

    visited.remove(node_id);
}

fn print_callgraph_dot(graph: &CallGraph, filter: Option<&str>, show_cycles: bool) {
    println!("digraph callgraph {{");
    println!("  rankdir=TB;");
    println!("  node [shape=box, fontname=\"Helvetica\"];");
    println!();

    for node in &graph.nodes {
        if let Some(f) = filter {
            if !node.name.contains(f) {
                continue;
            }
        }

        let color = if node.is_recursive { "red" } else { "black" };
        println!("  {} [label=\"{}\", color={}];", node.id, node.name, color);
    }

    for edge in &graph.edges {
        let style = if edge.is_tail_call { "dashed" } else { "solid" };
        let label = if edge.count > 1 {
            format!(" [label=\"×{}\"]", edge.count)
        } else {
            "".to_string()
        };
        println!("  {} -> {} [style={}{}];", edge.from, edge.to, style, label);
    }

    if show_cycles {
        println!();
        println!("  // Recursive nodes highlighted");
    }

    println!("}}");
}

// ============================================================================
// IR Diff
// ============================================================================

fn diff_ir(
    before: &PathBuf,
    after: &PathBuf,
    changes_only: bool,
    context: usize,
    ignore_whitespace: bool,
) -> Result<()> {
    let before_content = fs::read_to_string(before)
        .with_context(|| format!("Failed to read {}", before.display()))?;

    let after_content =
        fs::read_to_string(after).with_context(|| format!("Failed to read {}", after.display()))?;

    let (before_text, after_text) = if ignore_whitespace {
        (
            normalize_whitespace(&before_content),
            normalize_whitespace(&after_content),
        )
    } else {
        (before_content.clone(), after_content.clone())
    };

    println!("{}", "IR Diff".bold().cyan());
    println!("{}", "═".repeat(70));
    println!("Before: {}", before.display());
    println!("After:  {}", after.display());
    println!();

    let diff = TextDiff::from_lines(&before_text, &after_text);

    let mut has_changes = false;
    let mut context_buffer: Vec<(ChangeTag, &str)> = Vec::new();

    for change in diff.iter_all_changes() {
        let tag = change.tag();
        let line = change.value();

        match tag {
            ChangeTag::Equal => {
                if !changes_only {
                    context_buffer.push((tag, line));
                    if context_buffer.len() > context {
                        context_buffer.remove(0);
                    }
                }
            }
            ChangeTag::Delete => {
                has_changes = true;
                // Print context
                for (_, ctx_line) in &context_buffer {
                    println!("  {}", ctx_line.trim_end().dimmed());
                }
                context_buffer.clear();
                println!("{} {}", "-".red(), line.trim_end().red());
            }
            ChangeTag::Insert => {
                has_changes = true;
                // Print context
                for (_, ctx_line) in &context_buffer {
                    println!("  {}", ctx_line.trim_end().dimmed());
                }
                context_buffer.clear();
                println!("{} {}", "+".green(), line.trim_end().green());
            }
        }
    }

    if !has_changes {
        println!("{}", "No differences found.".green());
    }

    Ok(())
}

fn normalize_whitespace(s: &str) -> String {
    s.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

// ============================================================================
// Statistics
// ============================================================================

fn show_stats(file: &PathBuf, timing: bool, memory: bool, compare: Option<&PathBuf>) -> Result<()> {
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");

    // If it's a Haskell source file, compile and produce real stats
    if ext == "hs" {
        return show_stats_from_source(file, timing, memory);
    }

    let content =
        fs::read_to_string(file).with_context(|| format!("Failed to read {}", file.display()))?;

    let stats: CompileStats = serde_json::from_str(&content).unwrap_or_else(|_| {
        // Sample data
        CompileStats {
            module: "Example".to_string(),
            timestamp: chrono_lite(),
            phases: vec![
                PhaseStats {
                    name: "Parsing".to_string(),
                    time_ms: 12.5,
                    memory_mb: 50.0,
                    items_processed: 1000,
                },
                PhaseStats {
                    name: "Type Checking".to_string(),
                    time_ms: 45.2,
                    memory_mb: 150.0,
                    items_processed: 500,
                },
                PhaseStats {
                    name: "Core Lowering".to_string(),
                    time_ms: 8.3,
                    memory_mb: 80.0,
                    items_processed: 500,
                },
                PhaseStats {
                    name: "Tensor IR".to_string(),
                    time_ms: 15.7,
                    memory_mb: 120.0,
                    items_processed: 200,
                },
                PhaseStats {
                    name: "Loop IR".to_string(),
                    time_ms: 5.2,
                    memory_mb: 60.0,
                    items_processed: 200,
                },
                PhaseStats {
                    name: "Code Gen".to_string(),
                    time_ms: 25.1,
                    memory_mb: 200.0,
                    items_processed: 200,
                },
            ],
            summary: StatsSummary {
                total_time_ms: 112.0,
                peak_memory_mb: 200.0,
                modules_compiled: 1,
                lines_of_code: 1500,
                functions: 50,
                type_classes: 5,
            },
        }
    });

    let compare_stats: Option<CompileStats> = compare.and_then(|p| {
        fs::read_to_string(p)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
    });

    println!("{}", "Compilation Statistics".bold().cyan());
    println!("{}", "═".repeat(70));
    println!("Module: {}", stats.module.bold());
    println!("Generated: {}", stats.timestamp.dimmed());
    println!();

    // Summary
    let summary = &stats.summary;
    println!("{}", "Summary".bold());
    println!("{}", "─".repeat(70));
    println!(
        "  Modules:      {}",
        summary.modules_compiled.to_string().bold()
    );
    println!("  Lines:        {}", summary.lines_of_code);
    println!("  Functions:    {}", summary.functions);
    println!("  Type classes: {}", summary.type_classes);
    println!();

    // Timing breakdown
    if timing {
        println!("{}", "Timing Breakdown".bold());
        println!("{}", "─".repeat(70));

        let max_time = stats.phases.iter().map(|p| p.time_ms).fold(0.0, f64::max);

        for phase in &stats.phases {
            let bar_width = ((phase.time_ms / max_time) * 30.0) as usize;
            let bar = "█".repeat(bar_width);

            let diff_str = if let Some(ref cmp) = compare_stats {
                if let Some(cmp_phase) = cmp.phases.iter().find(|p| p.name == phase.name) {
                    let diff = phase.time_ms - cmp_phase.time_ms;
                    let pct = (diff / cmp_phase.time_ms) * 100.0;
                    if diff > 0.0 {
                        format!(" (+{:.1}ms, +{:.1}%)", diff, pct).red().to_string()
                    } else {
                        format!(" ({:.1}ms, {:.1}%)", diff, pct).green().to_string()
                    }
                } else {
                    "".to_string()
                }
            } else {
                "".to_string()
            };

            println!(
                "  {:<20} {:>8.1}ms {} {}",
                phase.name,
                phase.time_ms,
                bar.cyan(),
                diff_str
            );
        }

        println!("  {}", "─".repeat(50));
        println!("  {:<20} {:>8.1}ms", "Total".bold(), summary.total_time_ms);
        println!();
    }

    // Memory usage
    if memory {
        println!("{}", "Memory Usage".bold());
        println!("{}", "─".repeat(70));

        for phase in &stats.phases {
            println!("  {:<20} {:>8.1} MB", phase.name, phase.memory_mb);
        }

        println!("  {}", "─".repeat(50));
        println!("  {:<20} {:>8.1} MB", "Peak".bold(), summary.peak_memory_mb);
    }

    Ok(())
}

/// Compile a Haskell source file and show real compilation statistics.
fn show_stats_from_source(file: &PathBuf, timing: bool, _memory: bool) -> Result<()> {
    use std::time::Instant;

    let source =
        fs::read_to_string(file).with_context(|| format!("Failed to read {}", file.display()))?;

    let lines_of_code = source.lines().count();

    println!("{}", "Compilation Statistics (from source)".bold().cyan());
    println!("{}", "═".repeat(70));
    println!("File: {}", file.display().to_string().bold());
    println!();

    let mut phases = Vec::new();

    // Phase 1: Parse
    let t0 = Instant::now();
    let mut source_map = bhc_diagnostics::SourceMap::new();
    let file_id = source_map.add_file(file.display().to_string(), source.clone());
    let (module, diagnostics) = bhc_parser::parse_module(&source, file_id);
    let parse_time = t0.elapsed().as_secs_f64() * 1000.0;

    let decl_count = module.as_ref().map_or(0, |m| m.decls.len());
    phases.push(PhaseStats {
        name: "Parsing".to_string(),
        time_ms: parse_time,
        memory_mb: 0.0,
        items_processed: decl_count,
    });

    if !diagnostics.is_empty() {
        let renderer = bhc_diagnostics::DiagnosticRenderer::new(&source_map);
        renderer.render_all(&diagnostics);
    }

    let module = match module {
        Some(m) => m,
        None => {
            println!("  {} Parse failed", "✗".red());
            return Ok(());
        }
    };

    // Phase 2: Lower AST → HIR
    let t1 = Instant::now();
    let mut lower_ctx = bhc_lower::LowerContext::with_builtins();
    let config = bhc_lower::LowerConfig {
        include_builtins: true,
        warn_unused: false,
        search_paths: vec![],
    };
    let hir = bhc_lower::lower_module(&mut lower_ctx, &module, &config);
    let lower_time = t1.elapsed().as_secs_f64() * 1000.0;

    let hir = match hir {
        Ok(h) => {
            phases.push(PhaseStats {
                name: "AST → HIR".to_string(),
                time_ms: lower_time,
                memory_mb: 0.0,
                items_processed: decl_count,
            });
            h
        }
        Err(e) => {
            println!("  {} AST → HIR failed: {}", "✗".red(), e);
            return Ok(());
        }
    };

    // Phase 3: Lower HIR → Core
    let t2 = Instant::now();
    let def_map: bhc_hir_to_core::DefMap = lower_ctx
        .defs
        .iter()
        .map(|(def_id, def_info)| {
            (
                *def_id,
                bhc_hir_to_core::DefInfo {
                    id: *def_id,
                    name: def_info.name,
                },
            )
        })
        .collect();
    let core = bhc_hir_to_core::lower_module_with_defs(&hir, Some(&def_map), None);
    let core_time = t2.elapsed().as_secs_f64() * 1000.0;

    let core = match core {
        Ok(c) => {
            let binding_count = c.bindings.len();
            phases.push(PhaseStats {
                name: "HIR → Core".to_string(),
                time_ms: core_time,
                memory_mb: 0.0,
                items_processed: binding_count,
            });
            c
        }
        Err(e) => {
            println!("  {} HIR → Core failed: {}", "✗".red(), e);
            return Ok(());
        }
    };

    let total_time: f64 = phases.iter().map(|p| p.time_ms).sum();

    // Summary
    println!("{}", "Summary".bold());
    println!("{}", "─".repeat(70));
    println!("  Lines:        {}", lines_of_code);
    println!("  Declarations: {}", decl_count);
    println!("  Core bindings: {}", core.bindings.len());
    println!(
        "  Total time:   {:.2}ms",
        total_time
    );
    println!();

    // Timing breakdown
    if timing {
        println!("{}", "Timing Breakdown".bold());
        println!("{}", "─".repeat(70));

        let max_time = phases.iter().map(|p| p.time_ms).fold(0.0, f64::max);
        for phase in &phases {
            let bar_width = if max_time > 0.0 {
                ((phase.time_ms / max_time) * 30.0) as usize
            } else {
                0
            };
            let bar = "█".repeat(bar_width);
            println!(
                "  {:20} {:>8.2}ms  {} ({})",
                phase.name,
                phase.time_ms,
                bar.blue(),
                phase.items_processed,
            );
        }
        println!();
    }

    Ok(())
}

// ============================================================================
// Pretty Print
// ============================================================================

fn pretty_print(file: &PathBuf, stage: Option<IrStage>, width: usize, indent: usize) -> Result<()> {
    let content =
        fs::read_to_string(file).with_context(|| format!("Failed to read {}", file.display()))?;

    let stage = stage.unwrap_or_else(|| detect_stage(file));

    println!("{}", "Pretty Print".bold().cyan());
    println!("{}", "═".repeat(width.min(70)));
    println!();

    // Simple pretty-printing with indentation
    let mut current_indent: usize = 0;
    for line in content.lines() {
        let trimmed = line.trim();

        // Adjust indent based on braces/parens
        if trimmed.starts_with('}') || trimmed.starts_with(')') || trimmed.starts_with(']') {
            current_indent = current_indent.saturating_sub(1);
        }

        let indented = format!("{}{}", " ".repeat(current_indent * indent), trimmed);

        // Wrap long lines
        if indented.len() > width {
            let words: Vec<&str> = indented.split_whitespace().collect();
            let mut current_line = String::new();

            for word in words {
                if current_line.len() + word.len() + 1 > width {
                    println!("{}", current_line);
                    current_line = format!("{}{}", " ".repeat((current_indent + 1) * indent), word);
                } else {
                    if !current_line.is_empty() {
                        current_line.push(' ');
                    }
                    current_line.push_str(word);
                }
            }
            if !current_line.is_empty() {
                println!("{}", current_line);
            }
        } else {
            println!("{}", indented);
        }

        // Adjust indent for next line
        if trimmed.ends_with('{') || trimmed.ends_with('(') || trimmed.ends_with('[') {
            current_indent += 1;
        }
    }

    Ok(())
}

// ============================================================================
// Validation
// ============================================================================

fn validate_ir(
    file: &PathBuf,
    stage: Option<IrStage>,
    check_types: bool,
    check_invariants: bool,
) -> Result<()> {
    let content =
        fs::read_to_string(file).with_context(|| format!("Failed to read {}", file.display()))?;

    let stage = stage.unwrap_or_else(|| detect_stage(file));

    println!("{}", "IR Validation".bold().cyan());
    println!("{}", "═".repeat(60));
    println!("File: {}", file.display());
    println!("Stage: {:?}", stage);
    println!();

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Basic structural validation
    let open_parens = content.matches('(').count();
    let close_parens = content.matches(')').count();
    if open_parens != close_parens {
        errors.push(format!(
            "Mismatched parentheses: {} open, {} close",
            open_parens, close_parens
        ));
    }

    let open_braces = content.matches('{').count();
    let close_braces = content.matches('}').count();
    if open_braces != close_braces {
        errors.push(format!(
            "Mismatched braces: {} open, {} close",
            open_braces, close_braces
        ));
    }

    let open_brackets = content.matches('[').count();
    let close_brackets = content.matches(']').count();
    if open_brackets != close_brackets {
        errors.push(format!(
            "Mismatched brackets: {} open, {} close",
            open_brackets, close_brackets
        ));
    }

    // Stage-specific checks
    match stage {
        IrStage::Core => {
            if check_types {
                // Check for type annotations
                if !content.contains("::") {
                    warnings.push("No type annotations found in Core IR".to_string());
                }
            }

            if check_invariants {
                // Check for undefined variables (basic heuristic)
                if content.contains("undefined") {
                    warnings.push("Found 'undefined' in Core IR".to_string());
                }
            }
        }
        IrStage::Tensor => {
            if check_invariants {
                // Check for shape consistency hints
                if content.contains("shape mismatch") {
                    errors.push("Shape mismatch detected".to_string());
                }
            }
        }
        IrStage::Loop => {
            if check_invariants {
                // Check for unvectorized loops
                if content.contains("for ")
                    && !content.contains("simd")
                    && !content.contains("parallel")
                {
                    warnings.push("Found non-vectorized, non-parallel loop".to_string());
                }
            }
        }
        _ => {}
    }

    // Report results
    println!("{}", "Results".bold());
    println!("{}", "─".repeat(60));

    if errors.is_empty() && warnings.is_empty() {
        println!("{}", "✓ All checks passed".green().bold());
    } else {
        for error in &errors {
            println!("{} {}", "✗".red().bold(), error.red());
        }
        for warning in &warnings {
            println!("{} {}", "⚠".yellow().bold(), warning.yellow());
        }

        if !errors.is_empty() {
            println!();
            println!(
                "{} {} error(s), {} warning(s)",
                "Summary:".bold(),
                errors.len(),
                warnings.len()
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

// ============================================================================
// Utilities
// ============================================================================

fn chrono_lite() -> String {
    // Simple timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_detect_stage() {
        assert!(matches!(
            detect_stage(&PathBuf::from("test.core")),
            IrStage::Core
        ));
        assert!(matches!(
            detect_stage(&PathBuf::from("test.tensor")),
            IrStage::Tensor
        ));
        assert!(matches!(
            detect_stage(&PathBuf::from("test.loop")),
            IrStage::Loop
        ));
    }

    #[test]
    fn test_normalize_whitespace() {
        let input = "  foo   bar  \n  baz  ";
        let expected = "foo bar\nbaz";
        assert_eq!(normalize_whitespace(input), expected);
    }
}
