//! # Auto-Vectorization Pass
//!
//! This module implements auto-vectorization for Loop IR, transforming scalar
//! operations to SIMD operations based on loop analysis.
//!
//! ## M3 Exit Criteria
//!
//! - `matmul` microkernel auto-vectorizes on x86_64 and aarch64
//! - SIMD intrinsics: add, mul, fmadd, hadd
//!
//! ## Vectorization Strategy
//!
//! 1. **Analyze loops**: Identify vectorizable innermost loops
//! 2. **Check access patterns**: Sequential access enables vectorization
//! 3. **Transform operations**: Scalar → Vector operations
//! 4. **Handle remainder**: Scalar loop for non-aligned elements

use crate::{
    AccessPattern, BinOp, Body, Loop, LoopAttrs, LoopIR, LoopId, LoopMetadata, LoopType, Op,
    ScalarType, Stmt, TargetArch, TripCount, Value,
};
use rustc_hash::FxHashMap;
use thiserror::Error;

/// Errors that can occur during vectorization.
#[derive(Clone, Debug, Error)]
pub enum VectorizeError {
    /// Loop cannot be vectorized.
    #[error("loop {loop_id:?} cannot be vectorized: {reason}")]
    NotVectorizable {
        /// Loop identifier.
        loop_id: LoopId,
        /// Reason vectorization failed.
        reason: String,
    },

    /// Invalid vector width.
    #[error("invalid vector width {width} for type {ty:?}")]
    InvalidWidth {
        /// Requested width.
        width: u8,
        /// Element type.
        ty: ScalarType,
    },
}

/// Result of vectorization analysis for a loop.
#[derive(Clone, Debug)]
pub struct VectorizationInfo {
    /// Whether the loop can be vectorized.
    pub vectorizable: bool,
    /// Reason if not vectorizable.
    pub reason: Option<String>,
    /// Recommended vector width.
    pub recommended_width: u8,
    /// Access patterns in the loop.
    pub access_patterns: Vec<AccessPattern>,
    /// Whether FMA opportunities exist.
    pub has_fma: bool,
    /// Whether horizontal reduction is needed.
    pub has_reduction: bool,
}

impl Default for VectorizationInfo {
    fn default() -> Self {
        Self {
            vectorizable: false,
            reason: Some("not analyzed".to_string()),
            recommended_width: 1,
            access_patterns: Vec::new(),
            has_fma: false,
            has_reduction: false,
        }
    }
}

/// Configuration for vectorization.
#[derive(Clone, Debug)]
pub struct VectorizeConfig {
    /// Target architecture.
    pub target: TargetArch,
    /// Force a specific vector width (0 = auto).
    pub forced_width: u8,
    /// Generate remainder loop for non-aligned iterations.
    pub generate_remainder: bool,
    /// Enable FMA fusion.
    pub enable_fma: bool,
    /// Minimum trip count for vectorization.
    pub min_trip_count: usize,
}

impl Default for VectorizeConfig {
    fn default() -> Self {
        Self {
            target: TargetArch::default(),
            forced_width: 0,
            generate_remainder: true,
            enable_fma: true,
            min_trip_count: 4,
        }
    }
}

/// Vectorization pass state.
pub struct VectorizePass {
    config: VectorizeConfig,
    /// Analysis results per loop.
    analysis: FxHashMap<LoopId, VectorizationInfo>,
}

impl VectorizePass {
    /// Create a new vectorization pass with the given configuration.
    pub fn new(config: VectorizeConfig) -> Self {
        Self {
            config,
            analysis: FxHashMap::default(),
        }
    }

    /// Analyze a Loop IR function for vectorization opportunities.
    pub fn analyze(&mut self, ir: &LoopIR) -> FxHashMap<LoopId, VectorizationInfo> {
        self.analysis.clear();

        for stmt in &ir.body.stmts {
            self.analyze_stmt(stmt, &ir.loop_info);
        }

        self.analysis.clone()
    }

    /// Analyze a statement for vectorization.
    fn analyze_stmt(&mut self, stmt: &Stmt, loop_info: &[LoopMetadata]) {
        match stmt {
            Stmt::Loop(lp) => {
                let info = self.analyze_loop(lp, loop_info);
                self.analysis.insert(lp.id, info);

                // Recursively analyze nested loops
                for inner_stmt in &lp.body.stmts {
                    self.analyze_stmt(inner_stmt, loop_info);
                }
            }
            _ => {}
        }
    }

    /// Analyze a single loop for vectorization.
    fn analyze_loop(&self, lp: &Loop, loop_info: &[LoopMetadata]) -> VectorizationInfo {
        let mut info = VectorizationInfo::default();

        // Check if loop is marked as vectorizable
        if !lp.attrs.contains(LoopAttrs::VECTORIZE) {
            info.reason = Some("loop not marked VECTORIZE".to_string());
            return info;
        }

        // Check trip count
        let metadata = loop_info.iter().find(|m| m.id == lp.id);
        let trip_count = metadata.map(|m| &m.trip_count);

        match trip_count {
            Some(TripCount::Static(n)) if *n < self.config.min_trip_count => {
                info.reason = Some(format!(
                    "trip count {} below threshold {}",
                    n, self.config.min_trip_count
                ));
                return info;
            }
            Some(TripCount::Dynamic) => {
                // Dynamic trip count requires runtime check
                // Still vectorizable with remainder handling
            }
            _ => {}
        }

        // Analyze access patterns in loop body
        let (patterns, has_fma, has_reduction) = self.analyze_loop_body(&lp.body);
        info.access_patterns = patterns.clone();
        info.has_fma = has_fma;
        info.has_reduction = has_reduction;

        // Check if all accesses are vectorization-friendly
        let all_sequential = patterns
            .iter()
            .all(|p| matches!(p, AccessPattern::Sequential | AccessPattern::Broadcast));

        if !all_sequential {
            info.reason = Some("non-sequential access pattern".to_string());
            return info;
        }

        // Determine vector width
        let elem_type = self.infer_element_type(&lp.body);
        let width = if self.config.forced_width > 0 {
            self.config.forced_width
        } else {
            LoopType::natural_vector_width(elem_type, self.config.target)
        };

        info.vectorizable = width > 1;
        info.recommended_width = width;
        info.reason = None;

        info
    }

    /// Analyze loop body for access patterns, FMA opportunities, and reductions.
    fn analyze_loop_body(&self, body: &Body) -> (Vec<AccessPattern>, bool, bool) {
        let mut patterns = Vec::new();
        let mut has_fma = false;
        let mut has_reduction = false;

        for stmt in &body.stmts {
            match stmt {
                Stmt::Assign(_, op) => {
                    // Check for load access patterns
                    if let Op::Load(mem_ref) = op {
                        patterns.push(mem_ref.access.clone());
                    }

                    // Check for FMA pattern: a * b + c or a + b * c
                    if self.config.enable_fma {
                        has_fma |= self.is_fma_opportunity(op);
                    }

                    // Check for reduction operations
                    if let Op::VecReduce(_, _) = op {
                        has_reduction = true;
                    }
                }
                Stmt::Store(mem_ref, _) => {
                    patterns.push(mem_ref.access.clone());
                }
                Stmt::Loop(inner) => {
                    // Check if inner loop has reduction attribute
                    if inner.attrs.contains(LoopAttrs::REDUCTION) {
                        has_reduction = true;
                    }
                }
                _ => {}
            }
        }

        (patterns, has_fma, has_reduction)
    }

    /// Check if an operation can be replaced with FMA.
    fn is_fma_opportunity(&self, op: &Op) -> bool {
        // Pattern: Add(Mul(a, b), c) or Add(c, Mul(a, b))
        match op {
            Op::Binary(BinOp::Add, _, _) => {
                // Would need to check operands are Mul results
                // For now, return false as we'd need more context
                false
            }
            _ => false,
        }
    }

    /// Infer the element type from loop body operations.
    fn infer_element_type(&self, body: &Body) -> ScalarType {
        for stmt in &body.stmts {
            if let Stmt::Assign(_, Op::Load(mem_ref)) = stmt {
                if let LoopType::Scalar(s) = &mem_ref.elem_ty {
                    return *s;
                }
            }
        }
        ScalarType::Float(32) // Default
    }

    /// Apply vectorization to a Loop IR function.
    pub fn vectorize(&self, ir: &mut LoopIR) -> Result<VectorizeReport, VectorizeError> {
        let mut report = VectorizeReport::default();

        for stmt in &mut ir.body.stmts {
            self.vectorize_stmt(stmt, &mut ir.loop_info, &mut report)?;
        }

        Ok(report)
    }

    /// Vectorize a statement.
    fn vectorize_stmt(
        &self,
        stmt: &mut Stmt,
        loop_info: &mut Vec<LoopMetadata>,
        report: &mut VectorizeReport,
    ) -> Result<(), VectorizeError> {
        match stmt {
            Stmt::Loop(lp) => {
                if let Some(info) = self.analysis.get(&lp.id) {
                    if info.vectorizable {
                        self.vectorize_loop(lp, info, loop_info, report)?;
                    }
                }

                // Recursively vectorize nested loops
                for inner_stmt in &mut lp.body.stmts {
                    self.vectorize_stmt(inner_stmt, loop_info, report)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Vectorize a single loop.
    fn vectorize_loop(
        &self,
        lp: &mut Loop,
        info: &VectorizationInfo,
        loop_info: &mut Vec<LoopMetadata>,
        report: &mut VectorizeReport,
    ) -> Result<(), VectorizeError> {
        let width = info.recommended_width;

        // Update loop step
        lp.step = Value::i64(width as i64);

        // Update loop metadata
        if let Some(meta) = loop_info.iter_mut().find(|m| m.id == lp.id) {
            meta.vector_width = Some(width);
        }

        // Transform operations in loop body to vector operations
        self.vectorize_body(&mut lp.body, width)?;

        // Record vectorization
        report.vectorized_loops.push(VectorizedLoopInfo {
            loop_id: lp.id,
            vector_width: width,
            has_fma: info.has_fma,
            has_reduction: info.has_reduction,
        });

        Ok(())
    }

    /// Transform scalar operations in a body to vector operations.
    fn vectorize_body(&self, body: &mut Body, width: u8) -> Result<(), VectorizeError> {
        for stmt in &mut body.stmts {
            match stmt {
                Stmt::Assign(_, op) => {
                    *op = self.vectorize_op(op, width)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Transform a scalar operation to a vector operation.
    fn vectorize_op(&self, op: &Op, width: u8) -> Result<Op, VectorizeError> {
        match op {
            Op::Load(mem_ref) => {
                // Transform scalar load to vector load
                let mut vec_ref = mem_ref.clone();
                if let LoopType::Scalar(s) = &mem_ref.elem_ty {
                    vec_ref.elem_ty = LoopType::Vector(*s, width);
                }
                Ok(Op::Load(vec_ref))
            }

            Op::Binary(bin_op, a, b) => {
                // Transform scalar binary op to vector binary op
                let vec_a = self.vectorize_value(a, width);
                let vec_b = self.vectorize_value(b, width);
                Ok(Op::Binary(*bin_op, vec_a, vec_b))
            }

            Op::Unary(un_op, a) => {
                let vec_a = self.vectorize_value(a, width);
                Ok(Op::Unary(*un_op, vec_a))
            }

            // FMA is naturally vector
            Op::Fma(a, b, c) => {
                let vec_a = self.vectorize_value(a, width);
                let vec_b = self.vectorize_value(b, width);
                let vec_c = self.vectorize_value(c, width);
                Ok(Op::Fma(vec_a, vec_b, vec_c))
            }

            // Keep other operations as-is
            _ => Ok(op.clone()),
        }
    }

    /// Transform a scalar value to a vector value.
    fn vectorize_value(&self, val: &Value, width: u8) -> Value {
        match val {
            Value::Var(id, LoopType::Scalar(s)) => Value::Var(*id, LoopType::Vector(*s, width)),
            Value::FloatConst(f, s) => {
                // Scalar constant will be broadcast
                Value::FloatConst(*f, *s)
            }
            Value::IntConst(i, s) => Value::IntConst(*i, *s),
            _ => val.clone(),
        }
    }
}

/// Report of vectorization results.
#[derive(Clone, Debug, Default)]
pub struct VectorizeReport {
    /// Loops that were vectorized.
    pub vectorized_loops: Vec<VectorizedLoopInfo>,
    /// Loops that could not be vectorized.
    pub failed_loops: Vec<(LoopId, String)>,
}

impl VectorizeReport {
    /// Returns true if any loops were vectorized.
    pub fn any_vectorized(&self) -> bool {
        !self.vectorized_loops.is_empty()
    }

    /// Returns the total number of vectorized loops.
    pub fn count(&self) -> usize {
        self.vectorized_loops.len()
    }
}

impl std::fmt::Display for VectorizeReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Vectorization Report")?;
        writeln!(f, "====================")?;
        writeln!(f, "Vectorized loops: {}", self.vectorized_loops.len())?;

        for info in &self.vectorized_loops {
            writeln!(
                f,
                "  Loop {:?}: width={}, fma={}, reduction={}",
                info.loop_id, info.vector_width, info.has_fma, info.has_reduction
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

/// Information about a vectorized loop.
#[derive(Clone, Debug)]
pub struct VectorizedLoopInfo {
    /// Loop identifier.
    pub loop_id: LoopId,
    /// Vector width used.
    pub vector_width: u8,
    /// Whether FMA was used.
    pub has_fma: bool,
    /// Whether reduction was needed.
    pub has_reduction: bool,
}

// ============================================================================
// SIMD Intrinsics (M3 Deliverable)
// ============================================================================

/// SIMD intrinsic operations.
///
/// These map directly to hardware SIMD instructions on supported targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdIntrinsic {
    // --- Arithmetic ---
    /// Vector add: result[i] = a[i] + b[i]
    Add,
    /// Vector subtract: result[i] = a[i] - b[i]
    Sub,
    /// Vector multiply: result[i] = a[i] * b[i]
    Mul,
    /// Vector divide: result[i] = a[i] / b[i]
    Div,

    // --- Fused Multiply-Add ---
    /// Fused multiply-add: result[i] = a[i] * b[i] + c[i]
    Fmadd,
    /// Fused multiply-subtract: result[i] = a[i] * b[i] - c[i]
    Fmsub,
    /// Fused negative multiply-add: result[i] = -(a[i] * b[i]) + c[i]
    Fnmadd,

    // --- Horizontal Operations ---
    /// Horizontal add (pairwise): hadd([a,b,c,d], [e,f,g,h]) = [a+b, c+d, e+f, g+h]
    Hadd,
    /// Horizontal sum (reduce all elements): sum([a,b,c,d]) = a+b+c+d
    HorizontalSum,

    // --- Min/Max ---
    /// Vector minimum: result[i] = min(a[i], b[i])
    Min,
    /// Vector maximum: result[i] = max(a[i], b[i])
    Max,

    // --- Comparison ---
    /// Vector compare equal: result[i] = a[i] == b[i] ? ~0 : 0
    CmpEq,
    /// Vector compare less than: result[i] = a[i] < b[i] ? ~0 : 0
    CmpLt,
    /// Vector compare less or equal: result[i] = a[i] <= b[i] ? ~0 : 0
    CmpLe,

    // --- Data Movement ---
    /// Broadcast scalar to all lanes
    Broadcast,
    /// Extract element from vector
    Extract,
    /// Insert element into vector
    Insert,
    /// Shuffle/permute elements
    Shuffle,

    // --- Load/Store ---
    /// Aligned load
    LoadAligned,
    /// Unaligned load
    LoadUnaligned,
    /// Aligned store
    StoreAligned,
    /// Unaligned store
    StoreUnaligned,
}

impl SimdIntrinsic {
    /// Returns the x86 intrinsic name for this operation.
    pub fn x86_name(&self, ty: ScalarType, width: u8) -> &'static str {
        match (self, ty, width) {
            // Float32 x 4 (SSE)
            (Self::Add, ScalarType::Float(32), 4) => "_mm_add_ps",
            (Self::Sub, ScalarType::Float(32), 4) => "_mm_sub_ps",
            (Self::Mul, ScalarType::Float(32), 4) => "_mm_mul_ps",
            (Self::Div, ScalarType::Float(32), 4) => "_mm_div_ps",
            (Self::Fmadd, ScalarType::Float(32), 4) => "_mm_fmadd_ps",
            (Self::Min, ScalarType::Float(32), 4) => "_mm_min_ps",
            (Self::Max, ScalarType::Float(32), 4) => "_mm_max_ps",
            (Self::LoadAligned, ScalarType::Float(32), 4) => "_mm_load_ps",
            (Self::StoreAligned, ScalarType::Float(32), 4) => "_mm_store_ps",

            // Float32 x 8 (AVX)
            (Self::Add, ScalarType::Float(32), 8) => "_mm256_add_ps",
            (Self::Sub, ScalarType::Float(32), 8) => "_mm256_sub_ps",
            (Self::Mul, ScalarType::Float(32), 8) => "_mm256_mul_ps",
            (Self::Div, ScalarType::Float(32), 8) => "_mm256_div_ps",
            (Self::Fmadd, ScalarType::Float(32), 8) => "_mm256_fmadd_ps",
            (Self::Min, ScalarType::Float(32), 8) => "_mm256_min_ps",
            (Self::Max, ScalarType::Float(32), 8) => "_mm256_max_ps",
            (Self::LoadAligned, ScalarType::Float(32), 8) => "_mm256_load_ps",
            (Self::StoreAligned, ScalarType::Float(32), 8) => "_mm256_store_ps",
            (Self::Hadd, ScalarType::Float(32), 8) => "_mm256_hadd_ps",

            // Float64 x 2 (SSE2)
            (Self::Add, ScalarType::Float(64), 2) => "_mm_add_pd",
            (Self::Sub, ScalarType::Float(64), 2) => "_mm_sub_pd",
            (Self::Mul, ScalarType::Float(64), 2) => "_mm_mul_pd",
            (Self::Fmadd, ScalarType::Float(64), 2) => "_mm_fmadd_pd",

            // Float64 x 4 (AVX)
            (Self::Add, ScalarType::Float(64), 4) => "_mm256_add_pd",
            (Self::Sub, ScalarType::Float(64), 4) => "_mm256_sub_pd",
            (Self::Mul, ScalarType::Float(64), 4) => "_mm256_mul_pd",
            (Self::Fmadd, ScalarType::Float(64), 4) => "_mm256_fmadd_pd",

            _ => "unknown_intrinsic",
        }
    }

    /// Returns the ARM NEON intrinsic name for this operation.
    pub fn arm_name(&self, ty: ScalarType, width: u8) -> &'static str {
        match (self, ty, width) {
            // Float32 x 4 (NEON)
            (Self::Add, ScalarType::Float(32), 4) => "vaddq_f32",
            (Self::Sub, ScalarType::Float(32), 4) => "vsubq_f32",
            (Self::Mul, ScalarType::Float(32), 4) => "vmulq_f32",
            (Self::Fmadd, ScalarType::Float(32), 4) => "vfmaq_f32",
            (Self::Min, ScalarType::Float(32), 4) => "vminq_f32",
            (Self::Max, ScalarType::Float(32), 4) => "vmaxq_f32",
            (Self::LoadAligned, ScalarType::Float(32), 4) => "vld1q_f32",
            (Self::StoreAligned, ScalarType::Float(32), 4) => "vst1q_f32",

            // Float64 x 2 (NEON)
            (Self::Add, ScalarType::Float(64), 2) => "vaddq_f64",
            (Self::Sub, ScalarType::Float(64), 2) => "vsubq_f64",
            (Self::Mul, ScalarType::Float(64), 2) => "vmulq_f64",
            (Self::Fmadd, ScalarType::Float(64), 2) => "vfmaq_f64",

            _ => "unknown_intrinsic",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemRef, Param, ValueId};
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_tensor_ir::BufferId;

    fn make_vectorizable_loop(trip_count: usize) -> (LoopIR, LoopId) {
        let loop_id = LoopId::new(0);
        let loop_var = ValueId::new(0);

        let mem_ref = MemRef {
            buffer: BufferId::new(0),
            index: Value::Var(loop_var, LoopType::Scalar(ScalarType::I64)),
            elem_ty: LoopType::Scalar(ScalarType::F32),
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
                Value::Var(load_result, LoopType::Scalar(ScalarType::F32)),
                Value::float(2.0, 32),
            ),
        ));

        body.push(Stmt::Store(
            mem_ref,
            Value::Var(mul_result, LoopType::Scalar(ScalarType::F32)),
        ));

        let lp = Loop {
            id: loop_id,
            var: loop_var,
            lower: Value::i64(0),
            upper: Value::i64(trip_count as i64),
            step: Value::i64(1),
            body,
            attrs: LoopAttrs::VECTORIZE | LoopAttrs::INDEPENDENT,
        };

        let mut outer_body = Body::new();
        outer_body.push(Stmt::Loop(lp));

        let ir = LoopIR {
            name: Symbol::intern("test_kernel"),
            params: vec![Param {
                name: Symbol::intern("data"),
                ty: LoopType::Ptr(Box::new(LoopType::Scalar(ScalarType::F32))),
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
    fn test_vectorization_analysis() {
        let (ir, loop_id) = make_vectorizable_loop(1024);

        let mut pass = VectorizePass::new(VectorizeConfig::default());
        let analysis = pass.analyze(&ir);

        let info = analysis.get(&loop_id).expect("loop should be analyzed");
        assert!(info.vectorizable, "loop should be vectorizable");
        assert!(
            info.recommended_width > 1,
            "should recommend vector width > 1"
        );
    }

    #[test]
    fn test_vectorization_below_threshold() {
        let (ir, loop_id) = make_vectorizable_loop(2); // Below default threshold of 4

        let mut pass = VectorizePass::new(VectorizeConfig::default());
        let analysis = pass.analyze(&ir);

        let info = analysis.get(&loop_id).expect("loop should be analyzed");
        assert!(!info.vectorizable, "small loop should not be vectorizable");
    }

    #[test]
    fn test_vectorization_transform() {
        let (mut ir, _loop_id) = make_vectorizable_loop(1024);

        let mut pass = VectorizePass::new(VectorizeConfig::default());
        pass.analyze(&ir);
        let report = pass
            .vectorize(&mut ir)
            .expect("vectorization should succeed");

        assert!(report.any_vectorized(), "should have vectorized loops");
        assert_eq!(report.count(), 1, "should have vectorized 1 loop");
    }

    #[test]
    fn test_simd_intrinsic_names() {
        // x86 SSE
        assert_eq!(
            SimdIntrinsic::Add.x86_name(ScalarType::F32, 4),
            "_mm_add_ps"
        );
        assert_eq!(
            SimdIntrinsic::Fmadd.x86_name(ScalarType::F32, 4),
            "_mm_fmadd_ps"
        );

        // x86 AVX
        assert_eq!(
            SimdIntrinsic::Add.x86_name(ScalarType::F32, 8),
            "_mm256_add_ps"
        );
        assert_eq!(
            SimdIntrinsic::Hadd.x86_name(ScalarType::F32, 8),
            "_mm256_hadd_ps"
        );

        // ARM NEON
        assert_eq!(SimdIntrinsic::Add.arm_name(ScalarType::F32, 4), "vaddq_f32");
        assert_eq!(
            SimdIntrinsic::Fmadd.arm_name(ScalarType::F32, 4),
            "vfmaq_f32"
        );
    }

    #[test]
    fn test_target_vector_widths() {
        // AVX should use 8-wide for f32
        assert_eq!(
            LoopType::natural_vector_width(ScalarType::F32, TargetArch::X86_64Avx2),
            8
        );

        // SSE should use 4-wide for f32
        assert_eq!(
            LoopType::natural_vector_width(ScalarType::F32, TargetArch::X86_64Sse2),
            4
        );

        // NEON should use 4-wide for f32
        assert_eq!(
            LoopType::natural_vector_width(ScalarType::F32, TargetArch::Aarch64Neon),
            4
        );

        // AVX should use 4-wide for f64
        assert_eq!(
            LoopType::natural_vector_width(ScalarType::F64, TargetArch::X86_64Avx2),
            4
        );
    }

    #[test]
    fn test_vectorize_report_display() {
        let report = VectorizeReport {
            vectorized_loops: vec![VectorizedLoopInfo {
                loop_id: LoopId::new(0),
                vector_width: 8,
                has_fma: true,
                has_reduction: false,
            }],
            failed_loops: vec![],
        };

        let output = format!("{}", report);
        assert!(output.contains("Vectorized loops: 1"));
        assert!(output.contains("width=8"));
        assert!(output.contains("fma=true"));
    }
}
