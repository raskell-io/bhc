//! # Parallel Loop Primitives
//!
//! This module implements parallel loop constructs for Loop IR:
//! - `parFor`: Parallel iteration over a range
//! - `parMap`: Parallel map over tensor elements
//! - `parReduce`: Parallel reduction with deterministic chunking
//!
//! ## M3 Exit Criteria
//!
//! - Reductions scale linearly up to 8 cores
//! - Deterministic mode produces identical results across runs
//!
//! ## Scheduling Contract (from H26-SPEC)
//!
//! - Chunking MUST be deterministic given fixed worker count
//! - Non-deterministic mode allowed for floats (document variance)

use crate::{
    BarrierKind, Loop, LoopAttrs, LoopIR, LoopId, LoopMetadata, ReduceOp, Stmt, TripCount,
};
use rustc_hash::FxHashMap;
use thiserror::Error;

/// Errors that can occur during parallelization.
#[derive(Clone, Debug, Error)]
pub enum ParallelError {
    /// Loop cannot be parallelized.
    #[error("loop {loop_id:?} cannot be parallelized: {reason}")]
    NotParallelizable {
        /// Loop identifier.
        loop_id: LoopId,
        /// Reason parallelization failed.
        reason: String,
    },

    /// Invalid chunk size.
    #[error("invalid chunk size {chunk_size} for trip count {trip_count}")]
    InvalidChunkSize {
        /// Requested chunk size.
        chunk_size: usize,
        /// Total trip count.
        trip_count: usize,
    },
}

/// Configuration for parallel execution.
#[derive(Clone, Debug)]
pub struct ParallelConfig {
    /// Number of worker threads.
    pub worker_count: usize,
    /// Minimum iterations per worker.
    pub min_iterations_per_worker: usize,
    /// Enable deterministic mode for reproducible results.
    pub deterministic: bool,
    /// Chunk size for work distribution (0 = auto).
    pub chunk_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus(),
            min_iterations_per_worker: 64,
            deterministic: true, // Default to deterministic for reproducibility
            chunk_size: 0,       // Auto
        }
    }
}

/// Get the number of CPUs (simplified).
fn num_cpus() -> usize {
    // In real implementation, would use std::thread::available_parallelism
    8
}

/// Parallel execution strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// Static scheduling with fixed chunk sizes.
    /// Deterministic: same work distribution across runs.
    Static,
    /// Dynamic scheduling with work-stealing.
    /// Non-deterministic: work distribution varies.
    Dynamic,
    /// Guided scheduling with decreasing chunk sizes.
    /// Semi-deterministic.
    Guided,
}

/// Result of parallelization analysis for a loop.
#[derive(Clone, Debug)]
pub struct ParallelInfo {
    /// Whether the loop can be parallelized.
    pub parallelizable: bool,
    /// Reason if not parallelizable.
    pub reason: Option<String>,
    /// Recommended chunk size.
    pub chunk_size: usize,
    /// Number of chunks.
    pub num_chunks: usize,
    /// Recommended strategy.
    pub strategy: ParallelStrategy,
    /// Whether reduction is needed.
    pub is_reduction: bool,
}

impl Default for ParallelInfo {
    fn default() -> Self {
        Self {
            parallelizable: false,
            reason: Some("not analyzed".to_string()),
            chunk_size: 0,
            num_chunks: 0,
            strategy: ParallelStrategy::Static,
            is_reduction: false,
        }
    }
}

/// Parallelization pass state.
pub struct ParallelPass {
    config: ParallelConfig,
    /// Analysis results per loop.
    analysis: FxHashMap<LoopId, ParallelInfo>,
}

impl ParallelPass {
    /// Create a new parallelization pass with the given configuration.
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            analysis: FxHashMap::default(),
        }
    }

    /// Analyze a Loop IR function for parallelization opportunities.
    pub fn analyze(&mut self, ir: &LoopIR) -> FxHashMap<LoopId, ParallelInfo> {
        self.analysis.clear();

        for stmt in &ir.body.stmts {
            self.analyze_stmt(stmt, &ir.loop_info);
        }

        self.analysis.clone()
    }

    /// Analyze a statement for parallelization.
    fn analyze_stmt(&mut self, stmt: &Stmt, loop_info: &[LoopMetadata]) {
        match stmt {
            Stmt::Loop(lp) => {
                let info = self.analyze_loop(lp, loop_info);
                self.analysis.insert(lp.id, info);

                // Recursively analyze nested loops (but typically only outermost is parallelized)
                for inner_stmt in &lp.body.stmts {
                    self.analyze_stmt(inner_stmt, loop_info);
                }
            }
            _ => {}
        }
    }

    /// Analyze a single loop for parallelization.
    fn analyze_loop(&self, lp: &Loop, loop_info: &[LoopMetadata]) -> ParallelInfo {
        let mut info = ParallelInfo::default();

        // Check if loop is marked as parallelizable
        if !lp.attrs.contains(LoopAttrs::PARALLEL) {
            info.reason = Some("loop not marked PARALLEL".to_string());
            return info;
        }

        // Check for independence (no loop-carried dependencies)
        if !lp.attrs.contains(LoopAttrs::INDEPENDENT) {
            info.reason = Some("loop has dependencies".to_string());
            return info;
        }

        // Get trip count
        let metadata = loop_info.iter().find(|m| m.id == lp.id);
        let trip_count = match metadata.map(|m| &m.trip_count) {
            Some(TripCount::Static(n)) => *n,
            Some(TripCount::Bounded(n)) => *n,
            _ => {
                info.reason = Some("dynamic trip count".to_string());
                return info;
            }
        };

        // Check if worth parallelizing
        let min_total = self.config.worker_count * self.config.min_iterations_per_worker;
        if trip_count < min_total {
            info.reason = Some(format!(
                "trip count {} below threshold {}",
                trip_count, min_total
            ));
            return info;
        }

        // Determine chunk size
        let chunk_size = if self.config.chunk_size > 0 {
            self.config.chunk_size
        } else {
            compute_chunk_size(trip_count, self.config.worker_count)
        };

        // Check if this is a reduction loop
        let is_reduction = lp.attrs.contains(LoopAttrs::REDUCTION);

        info.parallelizable = true;
        info.reason = None;
        info.chunk_size = chunk_size;
        info.num_chunks = (trip_count + chunk_size - 1) / chunk_size;
        info.is_reduction = is_reduction;
        info.strategy = if self.config.deterministic {
            ParallelStrategy::Static
        } else {
            ParallelStrategy::Dynamic
        };

        info
    }

    /// Apply parallelization to a Loop IR function.
    pub fn parallelize(&self, ir: &mut LoopIR) -> Result<ParallelReport, ParallelError> {
        let mut report = ParallelReport::default();

        for stmt in &mut ir.body.stmts {
            self.parallelize_stmt(stmt, &mut ir.loop_info, &mut report)?;
        }

        Ok(report)
    }

    /// Parallelize a statement.
    fn parallelize_stmt(
        &self,
        stmt: &mut Stmt,
        loop_info: &mut Vec<LoopMetadata>,
        report: &mut ParallelReport,
    ) -> Result<(), ParallelError> {
        match stmt {
            Stmt::Loop(lp) => {
                if let Some(info) = self.analysis.get(&lp.id) {
                    if info.parallelizable {
                        self.parallelize_loop(lp, info, loop_info, report)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Parallelize a single loop.
    fn parallelize_loop(
        &self,
        lp: &mut Loop,
        info: &ParallelInfo,
        loop_info: &mut Vec<LoopMetadata>,
        report: &mut ParallelReport,
    ) -> Result<(), ParallelError> {
        // Update loop metadata
        if let Some(meta) = loop_info.iter_mut().find(|m| m.id == lp.id) {
            meta.parallel_chunk = Some(info.chunk_size);
        }

        // For reduction loops, we need special handling
        if info.is_reduction {
            self.parallelize_reduction(lp, info)?;
        }

        // Record parallelization
        report.parallelized_loops.push(ParallelizedLoopInfo {
            loop_id: lp.id,
            chunk_size: info.chunk_size,
            num_chunks: info.num_chunks,
            strategy: info.strategy,
            is_reduction: info.is_reduction,
        });

        Ok(())
    }

    /// Parallelize a reduction loop.
    fn parallelize_reduction(
        &self,
        lp: &mut Loop,
        _info: &ParallelInfo,
    ) -> Result<(), ParallelError> {
        // For deterministic reductions:
        // 1. Each worker computes partial result
        // 2. Partial results are combined in fixed order

        // Add barrier before final reduction
        lp.body.push(Stmt::Barrier(BarrierKind::ThreadGroup));

        Ok(())
    }
}

/// Compute optimal chunk size for work distribution.
fn compute_chunk_size(trip_count: usize, worker_count: usize) -> usize {
    // Simple static chunking: divide evenly among workers
    // Round up to ensure all iterations are covered
    (trip_count + worker_count - 1) / worker_count
}

/// Report of parallelization results.
#[derive(Clone, Debug, Default)]
pub struct ParallelReport {
    /// Loops that were parallelized.
    pub parallelized_loops: Vec<ParallelizedLoopInfo>,
    /// Loops that could not be parallelized.
    pub failed_loops: Vec<(LoopId, String)>,
}

impl ParallelReport {
    /// Returns true if any loops were parallelized.
    pub fn any_parallelized(&self) -> bool {
        !self.parallelized_loops.is_empty()
    }

    /// Returns the total number of parallelized loops.
    pub fn count(&self) -> usize {
        self.parallelized_loops.len()
    }
}

impl std::fmt::Display for ParallelReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Parallelization Report")?;
        writeln!(f, "======================")?;
        writeln!(f, "Parallelized loops: {}", self.parallelized_loops.len())?;

        for info in &self.parallelized_loops {
            writeln!(
                f,
                "  Loop {:?}: chunks={}, chunk_size={}, strategy={:?}, reduction={}",
                info.loop_id, info.num_chunks, info.chunk_size, info.strategy, info.is_reduction
            )?;
        }

        if !self.failed_loops.is_empty() {
            writeln!(f, "\nFailed loops: {}", self.failed_loops.len())?;
            for (id, reason) in &self.failed_loops {
                writeln!(f, "  Loop {:?}: {}", id, reason)?;
            }
        }

        Ok(())
    }
}

/// Information about a parallelized loop.
#[derive(Clone, Debug)]
pub struct ParallelizedLoopInfo {
    /// Loop identifier.
    pub loop_id: LoopId,
    /// Chunk size for work distribution.
    pub chunk_size: usize,
    /// Number of chunks.
    pub num_chunks: usize,
    /// Scheduling strategy used.
    pub strategy: ParallelStrategy,
    /// Whether this is a reduction loop.
    pub is_reduction: bool,
}

// ============================================================================
// Parallel Primitives API (M3 Deliverable)
// ============================================================================

/// Range for parallel iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    /// Start of range (inclusive).
    pub start: i64,
    /// End of range (exclusive).
    pub end: i64,
    /// Step size.
    pub step: i64,
}

impl Range {
    /// Create a new range.
    pub fn new(start: i64, end: i64) -> Self {
        Self {
            start,
            end,
            step: 1,
        }
    }

    /// Create a range with custom step.
    pub fn with_step(start: i64, end: i64, step: i64) -> Self {
        Self { start, end, step }
    }

    /// Returns the number of iterations.
    pub fn len(&self) -> usize {
        if self.step > 0 {
            ((self.end - self.start + self.step - 1) / self.step) as usize
        } else if self.step < 0 {
            ((self.start - self.end - self.step - 1) / (-self.step)) as usize
        } else {
            0
        }
    }

    /// Returns true if the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Split range into chunks for parallel execution.
    pub fn chunk(&self, num_chunks: usize) -> Vec<Range> {
        if num_chunks == 0 || self.is_empty() {
            return vec![];
        }

        let total = self.len();
        let chunk_size = (total + num_chunks - 1) / num_chunks;

        let mut chunks = Vec::with_capacity(num_chunks);
        let mut current = self.start;

        for i in 0..num_chunks {
            let chunk_iters = if i == num_chunks - 1 {
                total - (i * chunk_size)
            } else {
                chunk_size.min(total - i * chunk_size)
            };

            if chunk_iters == 0 {
                break;
            }

            let chunk_end = current + (chunk_iters as i64) * self.step;
            chunks.push(Range {
                start: current,
                end: chunk_end,
                step: self.step,
            });

            current = chunk_end;
        }

        chunks
    }
}

/// Parallel for loop descriptor.
///
/// ```text
/// parFor(0..n, |i| {
///     // body executed in parallel
/// })
/// ```
#[derive(Clone, Debug)]
pub struct ParFor {
    /// Iteration range.
    pub range: Range,
    /// Parallel configuration.
    pub config: ParallelConfig,
}

impl ParFor {
    /// Create a new parallel for loop.
    pub fn new(range: Range) -> Self {
        Self {
            range,
            config: ParallelConfig::default(),
        }
    }

    /// Set the parallel configuration.
    pub fn with_config(mut self, config: ParallelConfig) -> Self {
        self.config = config;
        self
    }

    /// Generate chunk assignments for workers.
    pub fn chunk_assignments(&self) -> Vec<Range> {
        self.range.chunk(self.config.worker_count)
    }
}

/// Parallel map descriptor.
///
/// ```text
/// result = parMap(f, input)
/// // Equivalent to: result[i] = f(input[i]) for all i, in parallel
/// ```
#[derive(Clone, Debug)]
pub struct ParMap {
    /// Number of elements.
    pub size: usize,
    /// Parallel configuration.
    pub config: ParallelConfig,
}

impl ParMap {
    /// Create a new parallel map.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            config: ParallelConfig::default(),
        }
    }

    /// Generate chunk assignments for workers.
    pub fn chunk_assignments(&self) -> Vec<Range> {
        let range = Range::new(0, self.size as i64);
        range.chunk(self.config.worker_count)
    }
}

/// Parallel reduce descriptor.
///
/// ```text
/// result = parReduce(combine, map_fn, input)
/// // Equivalent to: result = combine(map_fn(input[0]), map_fn(input[1]), ...)
/// ```
#[derive(Clone, Debug)]
pub struct ParReduce {
    /// Number of elements.
    pub size: usize,
    /// Reduction operation.
    pub op: ReduceOp,
    /// Parallel configuration.
    pub config: ParallelConfig,
}

impl ParReduce {
    /// Create a new parallel reduce.
    pub fn new(size: usize, op: ReduceOp) -> Self {
        Self {
            size,
            op,
            config: ParallelConfig::default(),
        }
    }

    /// Set deterministic mode.
    pub fn deterministic(mut self, det: bool) -> Self {
        self.config.deterministic = det;
        self
    }

    /// Generate chunk assignments for workers.
    ///
    /// For deterministic mode, chunks are assigned in order:
    /// - Worker 0 gets elements [0, chunk_size)
    /// - Worker 1 gets elements [chunk_size, 2*chunk_size)
    /// - etc.
    ///
    /// Final reduction combines partial results in worker order.
    pub fn chunk_assignments(&self) -> Vec<Range> {
        let range = Range::new(0, self.size as i64);
        range.chunk(self.config.worker_count)
    }

    /// Returns the identity value for this reduction operation.
    pub fn identity(&self) -> f64 {
        match self.op {
            ReduceOp::Add => 0.0,
            ReduceOp::Mul => 1.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
            ReduceOp::And => 1.0, // All bits set
            ReduceOp::Or => 0.0,
            ReduceOp::Xor => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AccessPattern, BinOp, Body, LoopType, MemRef, Op, Param, Value, ValueId};
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_tensor_ir::BufferId;

    fn make_parallelizable_loop(trip_count: usize) -> (LoopIR, LoopId) {
        let loop_id = LoopId::new(0);
        let loop_var = ValueId::new(0);

        let mem_ref = MemRef {
            buffer: BufferId::new(0),
            index: Value::Var(loop_var, LoopType::Scalar(crate::ScalarType::I64)),
            elem_ty: LoopType::Scalar(crate::ScalarType::F32),
            access: AccessPattern::Sequential,
        };

        let mut body = Body::new();
        let load_result = ValueId::new(1);
        body.push(Stmt::Assign(load_result, Op::Load(mem_ref.clone())));

        let mul_result = ValueId::new(2);
        body.push(Stmt::Assign(
            mul_result,
            Op::Binary(
                BinOp::Mul,
                Value::Var(load_result, LoopType::Scalar(crate::ScalarType::F32)),
                Value::float(2.0, 32),
            ),
        ));

        body.push(Stmt::Store(
            mem_ref,
            Value::Var(mul_result, LoopType::Scalar(crate::ScalarType::F32)),
        ));

        let lp = Loop {
            id: loop_id,
            var: loop_var,
            lower: Value::i64(0),
            upper: Value::i64(trip_count as i64),
            step: Value::i64(1),
            body,
            attrs: LoopAttrs::PARALLEL | LoopAttrs::INDEPENDENT,
        };

        let mut outer_body = Body::new();
        outer_body.push(Stmt::Loop(lp));

        let ir = LoopIR {
            name: Symbol::intern("test_kernel"),
            params: vec![Param {
                name: Symbol::intern("data"),
                ty: LoopType::Ptr(Box::new(LoopType::Scalar(crate::ScalarType::F32))),
                is_ptr: true,
            }],
            return_ty: LoopType::Void,
            body: outer_body,
            allocs: vec![],
            loop_info: vec![LoopMetadata {
                id: loop_id,
                trip_count: TripCount::Static(trip_count),
                vector_width: None,
                parallel_chunk: None,
                unroll_factor: None,
                dependencies: Vec::new(),
            }],
        };

        (ir, loop_id)
    }

    #[test]
    fn test_parallel_analysis() {
        let (ir, loop_id) = make_parallelizable_loop(10000);

        let mut pass = ParallelPass::new(ParallelConfig::default());
        let analysis = pass.analyze(&ir);

        let info = analysis.get(&loop_id).expect("loop should be analyzed");
        assert!(info.parallelizable, "loop should be parallelizable");
        assert!(info.chunk_size > 0, "should have positive chunk size");
    }

    #[test]
    fn test_parallel_below_threshold() {
        let (ir, loop_id) = make_parallelizable_loop(100); // Below default threshold

        let mut pass = ParallelPass::new(ParallelConfig::default());
        let analysis = pass.analyze(&ir);

        let info = analysis.get(&loop_id).expect("loop should be analyzed");
        assert!(
            !info.parallelizable,
            "small loop should not be parallelizable"
        );
    }

    #[test]
    fn test_range_chunking() {
        let range = Range::new(0, 1000);
        let chunks = range.chunk(8);

        assert_eq!(chunks.len(), 8);

        // Verify all iterations are covered
        let total_iters: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total_iters, 1000);

        // Verify chunks are contiguous
        for i in 1..chunks.len() {
            assert_eq!(chunks[i].start, chunks[i - 1].end);
        }
    }

    #[test]
    fn test_range_chunking_uneven() {
        let range = Range::new(0, 103); // Not evenly divisible
        let chunks = range.chunk(8);

        let total_iters: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total_iters, 103);
    }

    #[test]
    fn test_par_for_chunks() {
        let par_for = ParFor::new(Range::new(0, 10000)).with_config(ParallelConfig {
            worker_count: 8,
            ..Default::default()
        });

        let chunks = par_for.chunk_assignments();
        assert_eq!(chunks.len(), 8);

        // Each chunk should have roughly equal work
        let sizes: Vec<_> = chunks.iter().map(|c| c.len()).collect();
        let avg = sizes.iter().sum::<usize>() / sizes.len();
        for size in sizes {
            assert!((size as i64 - avg as i64).abs() <= 1);
        }
    }

    #[test]
    fn test_par_reduce_deterministic() {
        let par_reduce = ParReduce::new(10000, ReduceOp::Add).deterministic(true);

        assert!(par_reduce.config.deterministic);

        // Chunks should be deterministic
        let chunks1 = par_reduce.chunk_assignments();
        let chunks2 = par_reduce.chunk_assignments();

        for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(c1.start, c2.start);
            assert_eq!(c1.end, c2.end);
        }
    }

    #[test]
    fn test_par_reduce_identity() {
        assert_eq!(ParReduce::new(100, ReduceOp::Add).identity(), 0.0);
        assert_eq!(ParReduce::new(100, ReduceOp::Mul).identity(), 1.0);
        assert_eq!(ParReduce::new(100, ReduceOp::Min).identity(), f64::INFINITY);
        assert_eq!(
            ParReduce::new(100, ReduceOp::Max).identity(),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_parallel_report_display() {
        let report = ParallelReport {
            parallelized_loops: vec![ParallelizedLoopInfo {
                loop_id: LoopId::new(0),
                chunk_size: 1250,
                num_chunks: 8,
                strategy: ParallelStrategy::Static,
                is_reduction: false,
            }],
            failed_loops: vec![],
        };

        let output = format!("{}", report);
        assert!(output.contains("Parallelized loops: 1"));
        assert!(output.contains("chunks=8"));
        assert!(output.contains("Static"));
    }

    #[test]
    fn test_deterministic_vs_dynamic_strategy() {
        let mut config = ParallelConfig::default();

        config.deterministic = true;
        let (ir, loop_id) = make_parallelizable_loop(10000);
        let mut pass_det = ParallelPass::new(config.clone());
        let analysis = pass_det.analyze(&ir);
        assert_eq!(analysis[&loop_id].strategy, ParallelStrategy::Static);

        config.deterministic = false;
        let mut pass_dyn = ParallelPass::new(config);
        let analysis = pass_dyn.analyze(&ir);
        assert_eq!(analysis[&loop_id].strategy, ParallelStrategy::Dynamic);
    }
}
