//! BLAS Provider Interface
//!
//! This module defines a trait-based abstraction for Basic Linear Algebra
//! Subprograms (BLAS) implementations. This allows BHC to use different
//! BLAS backends (OpenBLAS, Intel MKL, Apple Accelerate) through a
//! unified interface.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    BlasProvider Trait                        │
//! └─────────────────────────────────────────────────────────────┘
//!            ▲              ▲              ▲
//!            │              │              │
//! ┌──────────┴───┐  ┌───────┴───────┐  ┌──┴──────────┐
//! │  OpenBLAS    │  │   Intel MKL   │  │  Accelerate │
//! │  (Default)   │  │   (Optional)  │  │  (macOS)    │
//! └──────────────┘  └───────────────┘  └─────────────┘
//! ```
//!
//! ## Fallback Strategy
//!
//! When no external BLAS is available, BHC falls back to a pure Rust
//! implementation. This ensures correctness at the cost of performance.
//!
//! ## M4 Exit Criteria
//!
//! - `matmul` can call external BLAS for large sizes
//! - BLAS operations use pinned buffers (no GC movement)

use crate::pinned::PinnedBuffer;
use thiserror::Error;

/// Errors that can occur during BLAS operations.
#[derive(Clone, Debug, Error)]
pub enum BlasError {
    /// Dimension mismatch for matrix operation.
    #[error("dimension mismatch: {operation} requires {expected}, got {actual}")]
    DimensionMismatch {
        /// The operation being performed.
        operation: &'static str,
        /// Expected dimensions.
        expected: String,
        /// Actual dimensions.
        actual: String,
    },

    /// Invalid leading dimension.
    #[error("invalid leading dimension: {ld} for matrix with {cols} columns")]
    InvalidLeadingDimension {
        /// Leading dimension provided.
        ld: usize,
        /// Number of columns.
        cols: usize,
    },

    /// BLAS library not available.
    #[error("BLAS library not available: {0}")]
    NotAvailable(String),

    /// Internal BLAS error.
    #[error("BLAS internal error: {0}")]
    Internal(String),
}

/// Result type for BLAS operations.
pub type BlasResult<T> = Result<T, BlasError>;

/// Matrix transpose option.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Transpose {
    /// No transpose (use matrix as-is).
    NoTrans = b'N',
    /// Transpose the matrix.
    Trans = b'T',
    /// Conjugate transpose (for complex).
    ConjTrans = b'C',
}

impl Transpose {
    /// Convert to CBLAS constant.
    #[must_use]
    pub const fn to_cblas(self) -> i32 {
        match self {
            Self::NoTrans => 111,   // CblasNoTrans
            Self::Trans => 112,     // CblasTrans
            Self::ConjTrans => 113, // CblasConjTrans
        }
    }

    /// Convert to BLAS character.
    #[must_use]
    pub const fn to_char(self) -> u8 {
        self as u8
    }
}

/// Matrix layout (row-major or column-major).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Layout {
    /// Row-major (C-style).
    RowMajor = 101,
    /// Column-major (Fortran-style).
    ColMajor = 102,
}

/// BLAS provider trait.
///
/// This trait defines the interface for BLAS implementations. BHC can
/// use any type implementing this trait for numeric operations.
///
/// All operations take pinned buffers to ensure memory stability during
/// the FFI call.
pub trait BlasProvider: Send + Sync {
    /// Get the name of this BLAS provider.
    fn name(&self) -> &'static str;

    /// Check if this provider is available.
    fn is_available(&self) -> bool;

    /// Get the number of threads used by this provider.
    fn num_threads(&self) -> usize;

    /// Set the number of threads.
    fn set_num_threads(&self, n: usize);

    // ========================================================================
    // Level 1 BLAS (vector operations)
    // ========================================================================

    /// Dot product: result = x · y
    fn ddot(&self, x: &PinnedBuffer<f64>, y: &PinnedBuffer<f64>) -> BlasResult<f64>;

    /// Single-precision dot product.
    fn sdot(&self, x: &PinnedBuffer<f32>, y: &PinnedBuffer<f32>) -> BlasResult<f32>;

    /// Vector scaling: x = alpha * x
    fn dscal(&self, alpha: f64, x: &mut PinnedBuffer<f64>) -> BlasResult<()>;

    /// Single-precision scaling.
    fn sscal(&self, alpha: f32, x: &mut PinnedBuffer<f32>) -> BlasResult<()>;

    /// AXPY: y = alpha * x + y
    fn daxpy(&self, alpha: f64, x: &PinnedBuffer<f64>, y: &mut PinnedBuffer<f64>)
        -> BlasResult<()>;

    /// Single-precision AXPY.
    fn saxpy(&self, alpha: f32, x: &PinnedBuffer<f32>, y: &mut PinnedBuffer<f32>)
        -> BlasResult<()>;

    /// Euclidean norm: ||x||_2
    fn dnrm2(&self, x: &PinnedBuffer<f64>) -> BlasResult<f64>;

    /// Single-precision norm.
    fn snrm2(&self, x: &PinnedBuffer<f32>) -> BlasResult<f32>;

    /// Sum of absolute values: ||x||_1
    fn dasum(&self, x: &PinnedBuffer<f64>) -> BlasResult<f64>;

    /// Single-precision absolute sum.
    fn sasum(&self, x: &PinnedBuffer<f32>) -> BlasResult<f32>;

    // ========================================================================
    // Level 3 BLAS (matrix operations)
    // ========================================================================

    /// General matrix multiplication: C = alpha * op(A) * op(B) + beta * C
    ///
    /// Where op(X) is X, X^T, or X^H depending on the transpose flag.
    ///
    /// # Arguments
    ///
    /// * `trans_a` - Transpose option for A
    /// * `trans_b` - Transpose option for B
    /// * `m` - Number of rows in op(A) and C
    /// * `n` - Number of columns in op(B) and C
    /// * `k` - Number of columns in op(A) and rows in op(B)
    /// * `alpha` - Scalar multiplier for A*B
    /// * `a` - Matrix A
    /// * `lda` - Leading dimension of A
    /// * `b` - Matrix B
    /// * `ldb` - Leading dimension of B
    /// * `beta` - Scalar multiplier for C
    /// * `c` - Matrix C (output)
    /// * `ldc` - Leading dimension of C
    #[allow(clippy::too_many_arguments)]
    fn dgemm(
        &self,
        trans_a: Transpose,
        trans_b: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &PinnedBuffer<f64>,
        lda: usize,
        b: &PinnedBuffer<f64>,
        ldb: usize,
        beta: f64,
        c: &mut PinnedBuffer<f64>,
        ldc: usize,
    ) -> BlasResult<()>;

    /// Single-precision GEMM.
    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        &self,
        trans_a: Transpose,
        trans_b: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &PinnedBuffer<f32>,
        lda: usize,
        b: &PinnedBuffer<f32>,
        ldb: usize,
        beta: f32,
        c: &mut PinnedBuffer<f32>,
        ldc: usize,
    ) -> BlasResult<()>;
}

/// Pure Rust fallback BLAS implementation.
///
/// This implementation is used when no external BLAS library is available.
/// It provides correct results but may be slower than optimized libraries.
#[derive(Debug, Default)]
pub struct FallbackBlas {
    num_threads: std::sync::atomic::AtomicUsize,
}

impl FallbackBlas {
    /// Create a new fallback BLAS provider.
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_threads: std::sync::atomic::AtomicUsize::new(1),
        }
    }
}

impl BlasProvider for FallbackBlas {
    fn name(&self) -> &'static str {
        "Fallback (Pure Rust)"
    }

    fn is_available(&self) -> bool {
        true // Always available
    }

    fn num_threads(&self) -> usize {
        self.num_threads.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn set_num_threads(&self, n: usize) {
        self.num_threads
            .store(n, std::sync::atomic::Ordering::Relaxed);
    }

    fn ddot(&self, x: &PinnedBuffer<f64>, y: &PinnedBuffer<f64>) -> BlasResult<f64> {
        if x.len() != y.len() {
            return Err(BlasError::DimensionMismatch {
                operation: "ddot",
                expected: format!("vectors of same length"),
                actual: format!("x.len={}, y.len={}", x.len(), y.len()),
            });
        }

        let result = x
            .as_slice()
            .iter()
            .zip(y.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum();

        Ok(result)
    }

    fn sdot(&self, x: &PinnedBuffer<f32>, y: &PinnedBuffer<f32>) -> BlasResult<f32> {
        if x.len() != y.len() {
            return Err(BlasError::DimensionMismatch {
                operation: "sdot",
                expected: format!("vectors of same length"),
                actual: format!("x.len={}, y.len={}", x.len(), y.len()),
            });
        }

        let result = x
            .as_slice()
            .iter()
            .zip(y.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum();

        Ok(result)
    }

    fn dscal(&self, alpha: f64, x: &mut PinnedBuffer<f64>) -> BlasResult<()> {
        for val in x.as_mut_slice() {
            *val *= alpha;
        }
        Ok(())
    }

    fn sscal(&self, alpha: f32, x: &mut PinnedBuffer<f32>) -> BlasResult<()> {
        for val in x.as_mut_slice() {
            *val *= alpha;
        }
        Ok(())
    }

    fn daxpy(
        &self,
        alpha: f64,
        x: &PinnedBuffer<f64>,
        y: &mut PinnedBuffer<f64>,
    ) -> BlasResult<()> {
        if x.len() != y.len() {
            return Err(BlasError::DimensionMismatch {
                operation: "daxpy",
                expected: format!("vectors of same length"),
                actual: format!("x.len={}, y.len={}", x.len(), y.len()),
            });
        }

        for (yi, xi) in y.as_mut_slice().iter_mut().zip(x.as_slice().iter()) {
            *yi += alpha * xi;
        }
        Ok(())
    }

    fn saxpy(
        &self,
        alpha: f32,
        x: &PinnedBuffer<f32>,
        y: &mut PinnedBuffer<f32>,
    ) -> BlasResult<()> {
        if x.len() != y.len() {
            return Err(BlasError::DimensionMismatch {
                operation: "saxpy",
                expected: format!("vectors of same length"),
                actual: format!("x.len={}, y.len={}", x.len(), y.len()),
            });
        }

        for (yi, xi) in y.as_mut_slice().iter_mut().zip(x.as_slice().iter()) {
            *yi += alpha * xi;
        }
        Ok(())
    }

    fn dnrm2(&self, x: &PinnedBuffer<f64>) -> BlasResult<f64> {
        let sum_sq: f64 = x.as_slice().iter().map(|v| v * v).sum();
        Ok(sum_sq.sqrt())
    }

    fn snrm2(&self, x: &PinnedBuffer<f32>) -> BlasResult<f32> {
        let sum_sq: f32 = x.as_slice().iter().map(|v| v * v).sum();
        Ok(sum_sq.sqrt())
    }

    fn dasum(&self, x: &PinnedBuffer<f64>) -> BlasResult<f64> {
        Ok(x.as_slice().iter().map(|v| v.abs()).sum())
    }

    fn sasum(&self, x: &PinnedBuffer<f32>) -> BlasResult<f32> {
        Ok(x.as_slice().iter().map(|v| v.abs()).sum())
    }

    #[allow(clippy::too_many_arguments)]
    fn dgemm(
        &self,
        trans_a: Transpose,
        trans_b: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &PinnedBuffer<f64>,
        lda: usize,
        b: &PinnedBuffer<f64>,
        ldb: usize,
        beta: f64,
        c: &mut PinnedBuffer<f64>,
        ldc: usize,
    ) -> BlasResult<()> {
        // Validate dimensions
        let (a_rows, _a_cols) = match trans_a {
            Transpose::NoTrans => (m, k),
            _ => (k, m),
        };
        let (b_rows, _b_cols) = match trans_b {
            Transpose::NoTrans => (k, n),
            _ => (n, k),
        };

        if a.len() < a_rows * lda {
            return Err(BlasError::DimensionMismatch {
                operation: "dgemm",
                expected: format!("A with at least {} elements", a_rows * lda),
                actual: format!("A.len={}", a.len()),
            });
        }

        if b.len() < b_rows * ldb {
            return Err(BlasError::DimensionMismatch {
                operation: "dgemm",
                expected: format!("B with at least {} elements", b_rows * ldb),
                actual: format!("B.len={}", b.len()),
            });
        }

        if c.len() < m * ldc {
            return Err(BlasError::DimensionMismatch {
                operation: "dgemm",
                expected: format!("C with at least {} elements", m * ldc),
                actual: format!("C.len={}", c.len()),
            });
        }

        let a_slice = a.as_slice();
        let b_slice = b.as_slice();
        let c_slice = c.as_mut_slice();

        // Naive GEMM: C = alpha * A * B + beta * C
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    let a_val = match trans_a {
                        Transpose::NoTrans => a_slice[i * lda + l],
                        _ => a_slice[l * lda + i],
                    };
                    let b_val = match trans_b {
                        Transpose::NoTrans => b_slice[l * ldb + j],
                        _ => b_slice[j * ldb + l],
                    };
                    sum += a_val * b_val;
                }
                c_slice[i * ldc + j] = alpha * sum + beta * c_slice[i * ldc + j];
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        &self,
        trans_a: Transpose,
        trans_b: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &PinnedBuffer<f32>,
        lda: usize,
        b: &PinnedBuffer<f32>,
        ldb: usize,
        beta: f32,
        c: &mut PinnedBuffer<f32>,
        ldc: usize,
    ) -> BlasResult<()> {
        // Validate dimensions (same as dgemm)
        let (a_rows, _a_cols) = match trans_a {
            Transpose::NoTrans => (m, k),
            _ => (k, m),
        };
        let (b_rows, _b_cols) = match trans_b {
            Transpose::NoTrans => (k, n),
            _ => (n, k),
        };

        if a.len() < a_rows * lda {
            return Err(BlasError::DimensionMismatch {
                operation: "sgemm",
                expected: format!("A with at least {} elements", a_rows * lda),
                actual: format!("A.len={}", a.len()),
            });
        }

        if b.len() < b_rows * ldb {
            return Err(BlasError::DimensionMismatch {
                operation: "sgemm",
                expected: format!("B with at least {} elements", b_rows * ldb),
                actual: format!("B.len={}", b.len()),
            });
        }

        if c.len() < m * ldc {
            return Err(BlasError::DimensionMismatch {
                operation: "sgemm",
                expected: format!("C with at least {} elements", m * ldc),
                actual: format!("C.len={}", c.len()),
            });
        }

        let a_slice = a.as_slice();
        let b_slice = b.as_slice();
        let c_slice = c.as_mut_slice();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_val = match trans_a {
                        Transpose::NoTrans => a_slice[i * lda + l],
                        _ => a_slice[l * lda + i],
                    };
                    let b_val = match trans_b {
                        Transpose::NoTrans => b_slice[l * ldb + j],
                        _ => b_slice[j * ldb + l],
                    };
                    sum += a_val * b_val;
                }
                c_slice[i * ldc + j] = alpha * sum + beta * c_slice[i * ldc + j];
            }
        }

        Ok(())
    }
}

/// OpenBLAS provider (when available).
#[cfg(feature = "openblas")]
pub mod openblas {
    use super::*;

    // CBLAS function declarations
    #[link(name = "openblas")]
    extern "C" {
        fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
        fn cblas_sdot(n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) -> f32;
        fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32);
        fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32);
        fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32);
        fn cblas_saxpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32);
        fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;
        fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32;
        fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64;
        fn cblas_sasum(n: i32, x: *const f32, incx: i32) -> f32;

        fn cblas_dgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );

        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );

        fn openblas_set_num_threads(n: i32);
        fn openblas_get_num_threads() -> i32;
    }

    /// OpenBLAS provider.
    #[derive(Debug, Default)]
    pub struct OpenBlas;

    impl OpenBlas {
        /// Create a new OpenBLAS provider.
        #[must_use]
        pub fn new() -> Self {
            Self
        }
    }

    impl BlasProvider for OpenBlas {
        fn name(&self) -> &'static str {
            "OpenBLAS"
        }

        fn is_available(&self) -> bool {
            true // Available if this module compiles
        }

        fn num_threads(&self) -> usize {
            unsafe { openblas_get_num_threads() as usize }
        }

        fn set_num_threads(&self, n: usize) {
            unsafe { openblas_set_num_threads(n as i32) }
        }

        fn ddot(&self, x: &PinnedBuffer<f64>, y: &PinnedBuffer<f64>) -> BlasResult<f64> {
            if x.len() != y.len() {
                return Err(BlasError::DimensionMismatch {
                    operation: "ddot",
                    expected: format!("vectors of same length"),
                    actual: format!("x.len={}, y.len={}", x.len(), y.len()),
                });
            }

            let result = unsafe { cblas_ddot(x.len() as i32, x.as_ptr(), 1, y.as_ptr(), 1) };
            Ok(result)
        }

        fn sdot(&self, x: &PinnedBuffer<f32>, y: &PinnedBuffer<f32>) -> BlasResult<f32> {
            if x.len() != y.len() {
                return Err(BlasError::DimensionMismatch {
                    operation: "sdot",
                    expected: format!("vectors of same length"),
                    actual: format!("x.len={}, y.len={}", x.len(), y.len()),
                });
            }

            let result = unsafe { cblas_sdot(x.len() as i32, x.as_ptr(), 1, y.as_ptr(), 1) };
            Ok(result)
        }

        fn dscal(&self, alpha: f64, x: &mut PinnedBuffer<f64>) -> BlasResult<()> {
            unsafe { cblas_dscal(x.len() as i32, alpha, x.as_mut_ptr(), 1) }
            Ok(())
        }

        fn sscal(&self, alpha: f32, x: &mut PinnedBuffer<f32>) -> BlasResult<()> {
            unsafe { cblas_sscal(x.len() as i32, alpha, x.as_mut_ptr(), 1) }
            Ok(())
        }

        fn daxpy(
            &self,
            alpha: f64,
            x: &PinnedBuffer<f64>,
            y: &mut PinnedBuffer<f64>,
        ) -> BlasResult<()> {
            if x.len() != y.len() {
                return Err(BlasError::DimensionMismatch {
                    operation: "daxpy",
                    expected: format!("vectors of same length"),
                    actual: format!("x.len={}, y.len={}", x.len(), y.len()),
                });
            }

            unsafe { cblas_daxpy(x.len() as i32, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1) }
            Ok(())
        }

        fn saxpy(
            &self,
            alpha: f32,
            x: &PinnedBuffer<f32>,
            y: &mut PinnedBuffer<f32>,
        ) -> BlasResult<()> {
            if x.len() != y.len() {
                return Err(BlasError::DimensionMismatch {
                    operation: "saxpy",
                    expected: format!("vectors of same length"),
                    actual: format!("x.len={}, y.len={}", x.len(), y.len()),
                });
            }

            unsafe { cblas_saxpy(x.len() as i32, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1) }
            Ok(())
        }

        fn dnrm2(&self, x: &PinnedBuffer<f64>) -> BlasResult<f64> {
            let result = unsafe { cblas_dnrm2(x.len() as i32, x.as_ptr(), 1) };
            Ok(result)
        }

        fn snrm2(&self, x: &PinnedBuffer<f32>) -> BlasResult<f32> {
            let result = unsafe { cblas_snrm2(x.len() as i32, x.as_ptr(), 1) };
            Ok(result)
        }

        fn dasum(&self, x: &PinnedBuffer<f64>) -> BlasResult<f64> {
            let result = unsafe { cblas_dasum(x.len() as i32, x.as_ptr(), 1) };
            Ok(result)
        }

        fn sasum(&self, x: &PinnedBuffer<f32>) -> BlasResult<f32> {
            let result = unsafe { cblas_sasum(x.len() as i32, x.as_ptr(), 1) };
            Ok(result)
        }

        #[allow(clippy::too_many_arguments)]
        fn dgemm(
            &self,
            trans_a: Transpose,
            trans_b: Transpose,
            m: usize,
            n: usize,
            k: usize,
            alpha: f64,
            a: &PinnedBuffer<f64>,
            lda: usize,
            b: &PinnedBuffer<f64>,
            ldb: usize,
            beta: f64,
            c: &mut PinnedBuffer<f64>,
            ldc: usize,
        ) -> BlasResult<()> {
            unsafe {
                cblas_dgemm(
                    Layout::RowMajor as i32,
                    trans_a.to_cblas(),
                    trans_b.to_cblas(),
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha,
                    a.as_ptr(),
                    lda as i32,
                    b.as_ptr(),
                    ldb as i32,
                    beta,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn sgemm(
            &self,
            trans_a: Transpose,
            trans_b: Transpose,
            m: usize,
            n: usize,
            k: usize,
            alpha: f32,
            a: &PinnedBuffer<f32>,
            lda: usize,
            b: &PinnedBuffer<f32>,
            ldb: usize,
            beta: f32,
            c: &mut PinnedBuffer<f32>,
            ldc: usize,
        ) -> BlasResult<()> {
            unsafe {
                cblas_sgemm(
                    Layout::RowMajor as i32,
                    trans_a.to_cblas(),
                    trans_b.to_cblas(),
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha,
                    a.as_ptr(),
                    lda as i32,
                    b.as_ptr(),
                    ldb as i32,
                    beta,
                    c.as_mut_ptr(),
                    ldc as i32,
                )
            }
            Ok(())
        }
    }
}

/// Get the default BLAS provider.
///
/// Returns OpenBLAS if available, otherwise falls back to pure Rust.
#[must_use]
pub fn default_provider() -> Box<dyn BlasProvider> {
    #[cfg(feature = "openblas")]
    {
        Box::new(openblas::OpenBlas::new())
    }

    #[cfg(not(feature = "openblas"))]
    {
        Box::new(FallbackBlas::new())
    }
}

/// Threshold for using external BLAS vs internal implementation.
///
/// For small matrices, the overhead of FFI may exceed the benefit.
pub const BLAS_THRESHOLD: usize = 64;

/// Check if BLAS should be used for a matrix of given dimensions.
#[must_use]
pub fn should_use_blas(m: usize, n: usize, k: usize) -> bool {
    // Use BLAS for matrices larger than threshold
    m >= BLAS_THRESHOLD || n >= BLAS_THRESHOLD || k >= BLAS_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_provider() -> FallbackBlas {
        FallbackBlas::new()
    }

    #[test]
    fn test_fallback_ddot() {
        let provider = get_provider();

        let x = PinnedBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = PinnedBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = provider.ddot(&x, &y).unwrap();
        assert!((result - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_fallback_dscal() {
        let provider = get_provider();

        let mut x = PinnedBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        provider.dscal(2.0, &mut x).unwrap();

        assert_eq!(x.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_fallback_daxpy() {
        let provider = get_provider();

        let x = PinnedBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut y = PinnedBuffer::from_slice(&[10.0, 20.0, 30.0, 40.0]).unwrap();

        provider.daxpy(2.0, &x, &mut y).unwrap();

        assert_eq!(y.as_slice(), &[12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn test_fallback_dnrm2() {
        let provider = get_provider();

        let x = PinnedBuffer::from_slice(&[3.0, 4.0]).unwrap();
        let result = provider.dnrm2(&x).unwrap();

        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fallback_dgemm() {
        let provider = get_provider();

        // 2x3 * 3x2 = 2x2
        let a = PinnedBuffer::from_slice(&[
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ])
        .unwrap();

        let b = PinnedBuffer::from_slice(&[
            7.0, 8.0, // row 0
            9.0, 10.0, // row 1
            11.0, 12.0, // row 2
        ])
        .unwrap();

        let mut c = PinnedBuffer::zeroed(4).unwrap();

        provider
            .dgemm(
                Transpose::NoTrans,
                Transpose::NoTrans,
                2,
                2,
                3,   // m, n, k
                1.0, // alpha
                &a,
                3, // A, lda
                &b,
                2,   // B, ldb
                0.0, // beta
                &mut c,
                2, // C, ldc
            )
            .unwrap();

        // Expected: [[58, 64], [139, 154]]
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose_to_cblas() {
        assert_eq!(Transpose::NoTrans.to_cblas(), 111);
        assert_eq!(Transpose::Trans.to_cblas(), 112);
        assert_eq!(Transpose::ConjTrans.to_cblas(), 113);
    }

    #[test]
    fn test_should_use_blas() {
        assert!(!should_use_blas(10, 10, 10));
        assert!(should_use_blas(100, 10, 10));
        assert!(should_use_blas(10, 100, 10));
        assert!(should_use_blas(10, 10, 100));
    }

    #[test]
    fn test_provider_name() {
        let provider = get_provider();
        assert_eq!(provider.name(), "Fallback (Pure Rust)");
    }
}
