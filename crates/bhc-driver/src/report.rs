//! Kernel report generation for the Numeric profile.
//!
//! This module provides comprehensive reporting of compiler optimization decisions
//! for numeric code, including:
//!
//! - Fusion analysis and results
//! - SIMD vectorization opportunities
//! - Parallelization analysis
//! - Memory allocation tracking
//!
//! # Usage
//!
//! ```bash
//! bhc --kernel-report --profile=numeric Main.hs
//! ```
//!
//! This generates a detailed report showing what optimizations were applied
//! and why certain patterns did or didn't fuse.

use bhc_loop_ir::parallel::ParallelInfo;
use bhc_loop_ir::vectorize::VectorizeReport;
use bhc_loop_ir::LoopId;
use bhc_tensor_ir::fusion::KernelReport as FusionReport;
use bhc_tensor_ir::{AllocRegion, FusionDecision, Kernel};
use rustc_hash::FxHashMap;
use std::fmt;

// ============================================================================
// Comprehensive Kernel Report
// ============================================================================

/// Comprehensive kernel report aggregating all optimization analyses.
///
/// This report provides full transparency into compiler decisions per H26-SPEC,
/// showing exactly what optimizations were applied and why.
#[derive(Clone, Debug, Default)]
pub struct ComprehensiveKernelReport {
    /// Module name being reported on.
    pub module_name: String,

    /// Fusion analysis results from Tensor IR.
    pub fusion: FusionSummary,

    /// Vectorization analysis results from Loop IR.
    pub vectorization: VectorizationSummary,

    /// Parallelization analysis results from Loop IR.
    pub parallelization: ParallelizationSummary,

    /// Memory allocation summary.
    pub memory: MemorySummary,

    /// Generated kernel summaries.
    pub kernels: Vec<KernelSummary>,
}

/// Summary of fusion analysis results.
#[derive(Clone, Debug, Default)]
pub struct FusionSummary {
    /// Total number of tensor operations before fusion.
    pub total_ops: usize,
    /// Number of operations successfully fused.
    pub fused_ops: usize,
    /// Number of materialization points.
    pub materialization_points: usize,
    /// Fusion decisions grouped by pattern.
    pub patterns: Vec<PatternMatch>,
    /// Operations that couldn't be fused.
    pub blocked: Vec<BlockedFusion>,
    /// Whether all guaranteed patterns fused successfully.
    pub all_guaranteed_fused: bool,
}

/// A matched fusion pattern.
#[derive(Clone, Debug)]
pub struct PatternMatch {
    /// Pattern name (e.g., "map/map", "sum/map", "softmax").
    pub pattern: String,
    /// Number of occurrences.
    pub count: usize,
}

/// Information about a blocked fusion.
#[derive(Clone, Debug)]
pub struct BlockedFusion {
    /// Operation that couldn't be fused.
    pub operation: String,
    /// Reason for blocking.
    pub reason: String,
}

/// Summary of vectorization analysis.
#[derive(Clone, Debug, Default)]
pub struct VectorizationSummary {
    /// Total loops analyzed.
    pub total_loops: usize,
    /// Loops that were vectorized.
    pub vectorized_loops: usize,
    /// Detailed info per vectorized loop.
    pub details: Vec<VectorizedLoopDetail>,
    /// Loops that couldn't be vectorized.
    pub failed: Vec<(String, String)>, // (loop_id, reason)
}

/// Details about a vectorized loop.
#[derive(Clone, Debug)]
pub struct VectorizedLoopDetail {
    /// Loop identifier.
    pub loop_id: String,
    /// Vector width (e.g., 4, 8, 16).
    pub vector_width: u8,
    /// Element type (e.g., "f32", "f64", "i32").
    pub elem_type: String,
    /// Whether FMA instructions are used.
    pub has_fma: bool,
    /// Whether reduction is needed.
    pub has_reduction: bool,
}

/// Summary of parallelization analysis.
#[derive(Clone, Debug, Default)]
pub struct ParallelizationSummary {
    /// Total loops analyzed.
    pub total_loops: usize,
    /// Loops that can be parallelized.
    pub parallelizable_loops: usize,
    /// Detailed info per parallelizable loop.
    pub details: Vec<ParallelLoopDetail>,
    /// Whether deterministic mode is enabled.
    pub deterministic_mode: bool,
}

/// Details about a parallelizable loop.
#[derive(Clone, Debug)]
pub struct ParallelLoopDetail {
    /// Loop identifier.
    pub loop_id: String,
    /// Scheduling strategy.
    pub strategy: String,
    /// Recommended chunk size.
    pub chunk_size: usize,
    /// Number of chunks.
    pub num_chunks: usize,
    /// Whether this is a reduction.
    pub is_reduction: bool,
}

/// Summary of memory allocations.
#[derive(Clone, Debug, Default)]
pub struct MemorySummary {
    /// Hot arena allocations (kernel temporaries).
    pub arena_bytes: usize,
    /// Number of arena allocations.
    pub arena_count: usize,
    /// Pinned memory (FFI, DMA).
    pub pinned_bytes: usize,
    /// Number of pinned allocations.
    pub pinned_count: usize,
    /// General heap allocations.
    pub general_bytes: usize,
    /// Number of general allocations.
    pub general_count: usize,
    /// Device memory (GPU).
    pub device_bytes: usize,
    /// Number of device allocations.
    pub device_count: usize,
}

/// Summary of a generated kernel.
#[derive(Clone, Debug)]
pub struct KernelSummary {
    /// Kernel name/identifier.
    pub name: String,
    /// Number of inputs.
    pub input_count: usize,
    /// Number of outputs.
    pub output_count: usize,
    /// Number of fused operations.
    pub fused_ops: usize,
    /// Whether fusion is complete (all expected patterns fused).
    pub fusion_complete: bool,
    /// Original operations that were fused.
    pub original_ops: Vec<String>,
}

// ============================================================================
// Report Builder
// ============================================================================

impl ComprehensiveKernelReport {
    /// Create a new report builder for the given module.
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            ..Default::default()
        }
    }

    /// Add fusion analysis results.
    pub fn with_fusion(mut self, report: &FusionReport) -> Self {
        // Count patterns
        let mut pattern_counts: FxHashMap<String, usize> = FxHashMap::default();
        let mut blocked = Vec::new();

        for decision in &report.decisions {
            match decision {
                FusionDecision::Fused(ops) => {
                    // Determine pattern from fused ops
                    let pattern = classify_fusion_pattern(ops);
                    *pattern_counts.entry(pattern).or_insert(0) += 1;
                }
                FusionDecision::Materialized(op, _reason) => {
                    // Materialization point
                }
                FusionDecision::Blocked(op, reason) => {
                    blocked.push(BlockedFusion {
                        operation: op.as_str().to_string(),
                        reason: format!("{:?}", reason),
                    });
                }
            }
        }

        let patterns: Vec<PatternMatch> = pattern_counts
            .into_iter()
            .map(|(pattern, count)| PatternMatch { pattern, count })
            .collect();

        // Check if all guaranteed patterns fused
        let all_guaranteed_fused =
            blocked.is_empty() && report.kernels.iter().all(|k| k.fusion_info.complete);

        self.fusion = FusionSummary {
            total_ops: report.total_ops,
            fused_ops: report.fused_ops,
            materialization_points: report
                .decisions
                .iter()
                .filter(|d| matches!(d, FusionDecision::Materialized(_, _)))
                .count(),
            patterns,
            blocked,
            all_guaranteed_fused,
        };

        // Build kernel summaries
        self.kernels = report
            .kernels
            .iter()
            .map(|k| KernelSummary {
                name: k.name.as_str().to_string(),
                input_count: k.inputs.len(),
                output_count: k.outputs.len(),
                fused_ops: k.fusion_info.original_ops.len(),
                fusion_complete: k.fusion_info.complete,
                original_ops: k
                    .fusion_info
                    .original_ops
                    .iter()
                    .map(|s| s.as_str().to_string())
                    .collect(),
            })
            .collect();

        // Calculate memory from kernel allocs
        self.memory = calculate_memory_summary(&report.kernels);

        self
    }

    /// Add vectorization analysis results.
    pub fn with_vectorization(mut self, report: &VectorizeReport) -> Self {
        self.vectorization = VectorizationSummary {
            total_loops: report.vectorized_loops.len() + report.failed_loops.len(),
            vectorized_loops: report.vectorized_loops.len(),
            details: report
                .vectorized_loops
                .iter()
                .map(|info| VectorizedLoopDetail {
                    loop_id: format!("{:?}", info.loop_id),
                    vector_width: info.vector_width,
                    elem_type: vector_width_to_type(info.vector_width),
                    has_fma: info.has_fma,
                    has_reduction: info.has_reduction,
                })
                .collect(),
            failed: report
                .failed_loops
                .iter()
                .map(|(id, reason)| (format!("{:?}", id), reason.clone()))
                .collect(),
        };
        self
    }

    /// Add parallelization analysis results.
    pub fn with_parallelization(
        mut self,
        analysis: &FxHashMap<LoopId, ParallelInfo>,
        deterministic: bool,
    ) -> Self {
        let parallelizable: Vec<_> = analysis
            .iter()
            .filter(|(_, info)| info.parallelizable)
            .collect();

        self.parallelization = ParallelizationSummary {
            total_loops: analysis.len(),
            parallelizable_loops: parallelizable.len(),
            details: parallelizable
                .iter()
                .map(|(id, info)| ParallelLoopDetail {
                    loop_id: format!("{:?}", id),
                    strategy: format!("{:?}", info.strategy),
                    chunk_size: info.chunk_size,
                    num_chunks: info.num_chunks,
                    is_reduction: info.is_reduction,
                })
                .collect(),
            deterministic_mode: deterministic,
        };
        self
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Classify fused operations into a pattern name.
fn classify_fusion_pattern(ops: &[bhc_intern::Symbol]) -> String {
    let op_names: Vec<&str> = ops.iter().map(|s| s.as_str()).collect();

    // Check for known patterns
    if op_names.iter().all(|op| op.contains("map")) {
        return "map/map".to_string();
    }
    if op_names
        .iter()
        .any(|op| op.contains("sum") || op.contains("reduce"))
    {
        if op_names.iter().any(|op| op.contains("map")) {
            return "sum/map".to_string();
        }
    }
    if op_names.iter().any(|op| op.contains("zipWith")) {
        if op_names.iter().filter(|op| op.contains("map")).count() >= 2 {
            return "zipWith/map/map".to_string();
        }
    }
    if op_names.iter().any(|op| op.contains("softmax")) {
        return "softmax".to_string();
    }
    if op_names.iter().any(|op| op.contains("layernorm")) {
        return "layernorm".to_string();
    }
    if op_names.iter().any(|op| op.contains("attention")) {
        return "attention".to_string();
    }

    // Generic fusion
    format!("fused({})", ops.len())
}

/// Calculate memory summary from kernel allocations.
fn calculate_memory_summary(kernels: &[Kernel]) -> MemorySummary {
    let mut summary = MemorySummary::default();

    for kernel in kernels {
        for alloc in &kernel.allocs {
            match alloc.region {
                AllocRegion::HotArena => {
                    summary.arena_bytes += alloc.size;
                    summary.arena_count += 1;
                }
                AllocRegion::Pinned => {
                    summary.pinned_bytes += alloc.size;
                    summary.pinned_count += 1;
                }
                AllocRegion::General => {
                    summary.general_bytes += alloc.size;
                    summary.general_count += 1;
                }
                AllocRegion::DeviceMemory(_) => {
                    summary.device_bytes += alloc.size;
                    summary.device_count += 1;
                }
            }
        }
    }

    summary
}

/// Convert vector width to human-readable type.
fn vector_width_to_type(width: u8) -> String {
    match width {
        4 => "Vec4 (SSE f32 / NEON f32)".to_string(),
        8 => "Vec8 (AVX f32 / AVX f64)".to_string(),
        16 => "Vec16 (AVX-512 f32)".to_string(),
        2 => "Vec2 (SSE f64)".to_string(),
        _ => format!("Vec{}", width),
    }
}

/// Format bytes into human-readable size.
fn format_bytes(bytes: usize) -> String {
    if bytes == 0 {
        "0 B".to_string()
    } else if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

// ============================================================================
// Display Implementation
// ============================================================================

impl fmt::Display for ComprehensiveKernelReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header
        writeln!(f)?;
        writeln!(
            f,
            "╔══════════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            f,
            "║                    KERNEL REPORT: {}                     ",
            self.module_name
        )?;
        writeln!(
            f,
            "╚══════════════════════════════════════════════════════════════════╝"
        )?;
        writeln!(f)?;

        // Fusion Section
        writeln!(
            f,
            "┌─────────────────────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│ FUSION ANALYSIS                                                 │"
        )?;
        writeln!(
            f,
            "├─────────────────────────────────────────────────────────────────┤"
        )?;
        writeln!(
            f,
            "│ Total tensor operations: {:>6}                                 │",
            self.fusion.total_ops
        )?;
        writeln!(
            f,
            "│ Successfully fused:      {:>6}                                 │",
            self.fusion.fused_ops
        )?;
        writeln!(
            f,
            "│ Materialization points:  {:>6}                                 │",
            self.fusion.materialization_points
        )?;

        let status = if self.fusion.all_guaranteed_fused {
            "✓ YES"
        } else {
            "✗ NO"
        };
        writeln!(
            f,
            "│ All guaranteed patterns: {:>6}                                 │",
            status
        )?;

        if !self.fusion.patterns.is_empty() {
            writeln!(
                f,
                "│                                                                 │"
            )?;
            writeln!(
                f,
                "│ Fused patterns:                                                 │"
            )?;
            for pat in &self.fusion.patterns {
                writeln!(
                    f,
                    "│   • {:30} ({} occurrences)        │",
                    pat.pattern, pat.count
                )?;
            }
        }

        if !self.fusion.blocked.is_empty() {
            writeln!(
                f,
                "│                                                                 │"
            )?;
            writeln!(
                f,
                "│ Blocked fusions:                                                │"
            )?;
            for blocked in &self.fusion.blocked {
                writeln!(
                    f,
                    "│   ✗ {}: {}        │",
                    blocked.operation, blocked.reason
                )?;
            }
        }
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        // Vectorization Section
        writeln!(
            f,
            "┌─────────────────────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│ SIMD VECTORIZATION                                              │"
        )?;
        writeln!(
            f,
            "├─────────────────────────────────────────────────────────────────┤"
        )?;
        writeln!(
            f,
            "│ Total loops analyzed:    {:>6}                                 │",
            self.vectorization.total_loops
        )?;
        writeln!(
            f,
            "│ Vectorized loops:        {:>6}                                 │",
            self.vectorization.vectorized_loops
        )?;

        if !self.vectorization.details.is_empty() {
            writeln!(
                f,
                "│                                                                 │"
            )?;
            writeln!(
                f,
                "│ Vectorized:                                                     │"
            )?;
            for detail in &self.vectorization.details {
                let fma = if detail.has_fma { "FMA ✓" } else { "     " };
                let red = if detail.has_reduction {
                    "RED ✓"
                } else {
                    "     "
                };
                writeln!(
                    f,
                    "│   • {}: width={:>2}, {} {} {}│",
                    detail.loop_id, detail.vector_width, detail.elem_type, fma, red
                )?;
            }
        }

        if !self.vectorization.failed.is_empty() {
            writeln!(
                f,
                "│                                                                 │"
            )?;
            writeln!(
                f,
                "│ Not vectorized:                                                 │"
            )?;
            for (loop_id, reason) in &self.vectorization.failed {
                writeln!(
                    f,
                    "│   ✗ {}: {}                                │",
                    loop_id, reason
                )?;
            }
        }
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        // Parallelization Section
        writeln!(
            f,
            "┌─────────────────────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│ PARALLELIZATION                                                 │"
        )?;
        writeln!(
            f,
            "├─────────────────────────────────────────────────────────────────┤"
        )?;
        writeln!(
            f,
            "│ Total loops analyzed:    {:>6}                                 │",
            self.parallelization.total_loops
        )?;
        writeln!(
            f,
            "│ Parallelizable loops:    {:>6}                                 │",
            self.parallelization.parallelizable_loops
        )?;

        let det_status = if self.parallelization.deterministic_mode {
            "✓ YES"
        } else {
            "✗ NO"
        };
        writeln!(
            f,
            "│ Deterministic mode:      {:>6}                                 │",
            det_status
        )?;

        if !self.parallelization.details.is_empty() {
            writeln!(
                f,
                "│                                                                 │"
            )?;
            writeln!(
                f,
                "│ Parallel loops:                                                 │"
            )?;
            for detail in &self.parallelization.details {
                let red = if detail.is_reduction { "reduction" } else { "" };
                writeln!(
                    f,
                    "│   • {}: {} chunks={} size={} {}│",
                    detail.loop_id, detail.strategy, detail.num_chunks, detail.chunk_size, red
                )?;
            }
        }
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        // Memory Section
        writeln!(
            f,
            "┌─────────────────────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│ MEMORY ALLOCATION                                               │"
        )?;
        writeln!(
            f,
            "├─────────────────────────────────────────────────────────────────┤"
        )?;
        writeln!(
            f,
            "│ Hot Arena:    {:>10} ({} allocations)                      │",
            format_bytes(self.memory.arena_bytes),
            self.memory.arena_count
        )?;
        writeln!(
            f,
            "│ Pinned:       {:>10} ({} allocations)                      │",
            format_bytes(self.memory.pinned_bytes),
            self.memory.pinned_count
        )?;
        writeln!(
            f,
            "│ General Heap: {:>10} ({} allocations)                      │",
            format_bytes(self.memory.general_bytes),
            self.memory.general_count
        )?;
        writeln!(
            f,
            "│ Device (GPU): {:>10} ({} allocations)                      │",
            format_bytes(self.memory.device_bytes),
            self.memory.device_count
        )?;

        let total = self.memory.arena_bytes
            + self.memory.pinned_bytes
            + self.memory.general_bytes
            + self.memory.device_bytes;
        writeln!(
            f,
            "│                                                                 │"
        )?;
        writeln!(
            f,
            "│ Total:        {:>10}                                       │",
            format_bytes(total)
        )?;
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;

        // Kernel Summary Section
        writeln!(
            f,
            "┌─────────────────────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│ GENERATED KERNELS: {:>3}                                         │",
            self.kernels.len()
        )?;
        writeln!(
            f,
            "├─────────────────────────────────────────────────────────────────┤"
        )?;

        for kernel in &self.kernels {
            let status = if kernel.fusion_complete { "✓" } else { "✗" };
            writeln!(
                f,
                "│ {} {:20} in={} out={} fused={:>2} ops              │",
                status, kernel.name, kernel.input_count, kernel.output_count, kernel.fused_ops
            )?;
            if !kernel.original_ops.is_empty() && kernel.original_ops.len() <= 5 {
                let ops_str = kernel.original_ops.join(", ");
                if ops_str.len() < 55 {
                    writeln!(f, "│     ops: {:55}│", ops_str)?;
                }
            }
        }
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────────────┘"
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_empty_report() {
        let report = ComprehensiveKernelReport::new("TestModule");
        let output = format!("{}", report);
        assert!(output.contains("KERNEL REPORT"));
        assert!(output.contains("TestModule"));
        assert!(output.contains("FUSION ANALYSIS"));
        assert!(output.contains("SIMD VECTORIZATION"));
        assert!(output.contains("PARALLELIZATION"));
        assert!(output.contains("MEMORY ALLOCATION"));
    }

    #[test]
    fn test_vector_width_to_type() {
        assert!(vector_width_to_type(4).contains("Vec4"));
        assert!(vector_width_to_type(8).contains("Vec8"));
        assert!(vector_width_to_type(16).contains("Vec16"));
    }
}
