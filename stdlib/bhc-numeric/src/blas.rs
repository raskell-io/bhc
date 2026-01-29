//! BLAS provider abstraction
//!
//! Abstraction over BLAS implementations (OpenBLAS, MKL, Accelerate).
//! Provides runtime detection and automatic selection of the best available
//! provider.

use std::sync::OnceLock;

// Import external BLAS providers when features are enabled
#[cfg(feature = "openblas")]
use crate::blas_openblas::OpenBlasProvider;

#[cfg(all(target_os = "macos", feature = "accelerate"))]
use crate::blas_accelerate::AccelerateProvider;

/// BLAS transpose option
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transpose {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
}

/// BLAS matrix layout
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RowMajor = 101,
    ColMajor = 102,
}

/// BLAS provider trait for f64 operations
pub trait BlasProviderF64: Send + Sync {
    /// Provider name
    fn name(&self) -> &'static str;

    /// Matrix-matrix multiply: C = alpha * op(A) * op(B) + beta * C
    fn dgemm(
        &self,
        layout: Layout,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: &[f64],
        lda: i32,
        b: &[f64],
        ldb: i32,
        beta: f64,
        c: &mut [f64],
        ldc: i32,
    );

    /// Matrix-vector multiply: y = alpha * op(A) * x + beta * y
    fn dgemv(
        &self,
        layout: Layout,
        trans: Transpose,
        m: i32,
        n: i32,
        alpha: f64,
        a: &[f64],
        lda: i32,
        x: &[f64],
        incx: i32,
        beta: f64,
        y: &mut [f64],
        incy: i32,
    );

    /// Dot product: x^T * y
    fn ddot(&self, n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64;

    /// Euclidean norm: ||x||_2
    fn dnrm2(&self, n: i32, x: &[f64], incx: i32) -> f64;

    /// Sum of absolute values: ||x||_1
    fn dasum(&self, n: i32, x: &[f64], incx: i32) -> f64;

    /// Scale vector: x = alpha * x
    fn dscal(&self, n: i32, alpha: f64, x: &mut [f64], incx: i32);

    /// Vector addition: y = alpha * x + y
    fn daxpy(&self, n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32);

    /// Copy vector: y = x
    fn dcopy(&self, n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32);

    /// Index of maximum absolute value
    fn idamax(&self, n: i32, x: &[f64], incx: i32) -> i32;
}

/// BLAS provider trait for f32 operations
pub trait BlasProviderF32: Send + Sync {
    /// Provider name
    fn name(&self) -> &'static str;

    /// Matrix-matrix multiply
    fn sgemm(
        &self,
        layout: Layout,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        b: &[f32],
        ldb: i32,
        beta: f32,
        c: &mut [f32],
        ldc: i32,
    );

    /// Matrix-vector multiply
    fn sgemv(
        &self,
        layout: Layout,
        trans: Transpose,
        m: i32,
        n: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        x: &[f32],
        incx: i32,
        beta: f32,
        y: &mut [f32],
        incy: i32,
    );

    /// Dot product
    fn sdot(&self, n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32;

    /// Euclidean norm
    fn snrm2(&self, n: i32, x: &[f32], incx: i32) -> f32;

    /// Scale vector
    fn sscal(&self, n: i32, alpha: f32, x: &mut [f32], incx: i32);

    /// Vector addition
    fn saxpy(&self, n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32);
}

// ============================================================
// Pure Rust BLAS Implementation (Fallback)
// ============================================================

/// Pure Rust BLAS implementation (fallback)
#[derive(Debug, Clone, Copy)]
pub struct PureRustBlas;

impl BlasProviderF64 for PureRustBlas {
    fn name(&self) -> &'static str {
        "PureRust"
    }

    fn dgemm(
        &self,
        layout: Layout,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: &[f64],
        lda: i32,
        b: &[f64],
        ldb: i32,
        beta: f64,
        c: &mut [f64],
        ldc: i32,
    ) {
        let m = m as usize;
        let n = n as usize;
        let k = k as usize;
        let lda = lda as usize;
        let ldb = ldb as usize;
        let ldc = ldc as usize;

        // Apply beta to C first
        if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let idx = match layout {
                        Layout::RowMajor => i * ldc + j,
                        Layout::ColMajor => j * ldc + i,
                    };
                    c[idx] *= beta;
                }
            }
        }

        // Compute alpha * A * B and add to C
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    let a_idx = match (layout, trans_a) {
                        (Layout::RowMajor, Transpose::NoTrans) => i * lda + l,
                        (Layout::RowMajor, _) => l * lda + i,
                        (Layout::ColMajor, Transpose::NoTrans) => l * lda + i,
                        (Layout::ColMajor, _) => i * lda + l,
                    };
                    let b_idx = match (layout, trans_b) {
                        (Layout::RowMajor, Transpose::NoTrans) => l * ldb + j,
                        (Layout::RowMajor, _) => j * ldb + l,
                        (Layout::ColMajor, Transpose::NoTrans) => j * ldb + l,
                        (Layout::ColMajor, _) => l * ldb + j,
                    };
                    sum += a[a_idx] * b[b_idx];
                }
                let c_idx = match layout {
                    Layout::RowMajor => i * ldc + j,
                    Layout::ColMajor => j * ldc + i,
                };
                c[c_idx] += alpha * sum;
            }
        }
    }

    fn dgemv(
        &self,
        layout: Layout,
        trans: Transpose,
        m: i32,
        n: i32,
        alpha: f64,
        a: &[f64],
        lda: i32,
        x: &[f64],
        incx: i32,
        beta: f64,
        y: &mut [f64],
        incy: i32,
    ) {
        let m = m as usize;
        let n = n as usize;
        let lda = lda as usize;
        let incx = incx as usize;
        let incy = incy as usize;

        let (rows, cols) = match trans {
            Transpose::NoTrans => (m, n),
            _ => (n, m),
        };

        // Apply beta to y
        for i in 0..rows {
            y[i * incy] *= beta;
        }

        // Compute alpha * A * x and add to y
        for i in 0..rows {
            let mut sum = 0.0;
            for j in 0..cols {
                let a_idx = match (layout, trans) {
                    (Layout::RowMajor, Transpose::NoTrans) => i * lda + j,
                    (Layout::RowMajor, _) => j * lda + i,
                    (Layout::ColMajor, Transpose::NoTrans) => j * lda + i,
                    (Layout::ColMajor, _) => i * lda + j,
                };
                sum += a[a_idx] * x[j * incx];
            }
            y[i * incy] += alpha * sum;
        }
    }

    fn ddot(&self, n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
        let n = n as usize;
        let incx = incx as usize;
        let incy = incy as usize;

        (0..n).map(|i| x[i * incx] * y[i * incy]).sum()
    }

    fn dnrm2(&self, n: i32, x: &[f64], incx: i32) -> f64 {
        let n = n as usize;
        let incx = incx as usize;

        (0..n)
            .map(|i| x[i * incx] * x[i * incx])
            .sum::<f64>()
            .sqrt()
    }

    fn dasum(&self, n: i32, x: &[f64], incx: i32) -> f64 {
        let n = n as usize;
        let incx = incx as usize;

        (0..n).map(|i| x[i * incx].abs()).sum()
    }

    fn dscal(&self, n: i32, alpha: f64, x: &mut [f64], incx: i32) {
        let n = n as usize;
        let incx = incx as usize;

        for i in 0..n {
            x[i * incx] *= alpha;
        }
    }

    fn daxpy(&self, n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
        let n = n as usize;
        let incx = incx as usize;
        let incy = incy as usize;

        for i in 0..n {
            y[i * incy] += alpha * x[i * incx];
        }
    }

    fn dcopy(&self, n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
        let n = n as usize;
        let incx = incx as usize;
        let incy = incy as usize;

        for i in 0..n {
            y[i * incy] = x[i * incx];
        }
    }

    fn idamax(&self, n: i32, x: &[f64], incx: i32) -> i32 {
        let n = n as usize;
        let incx = incx as usize;

        if n == 0 {
            return 0;
        }

        let mut max_idx = 0;
        let mut max_val = x[0].abs();

        for i in 1..n {
            let val = x[i * incx].abs();
            // Only update if strictly greater (keeps first occurrence in ties)
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        max_idx as i32
    }
}

impl BlasProviderF32 for PureRustBlas {
    fn name(&self) -> &'static str {
        "PureRust"
    }

    fn sgemm(
        &self,
        layout: Layout,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        b: &[f32],
        ldb: i32,
        beta: f32,
        c: &mut [f32],
        ldc: i32,
    ) {
        let m = m as usize;
        let n = n as usize;
        let k = k as usize;
        let lda = lda as usize;
        let ldb = ldb as usize;
        let ldc = ldc as usize;

        if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let idx = match layout {
                        Layout::RowMajor => i * ldc + j,
                        Layout::ColMajor => j * ldc + i,
                    };
                    c[idx] *= beta;
                }
            }
        }

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_idx = match (layout, trans_a) {
                        (Layout::RowMajor, Transpose::NoTrans) => i * lda + l,
                        (Layout::RowMajor, _) => l * lda + i,
                        (Layout::ColMajor, Transpose::NoTrans) => l * lda + i,
                        (Layout::ColMajor, _) => i * lda + l,
                    };
                    let b_idx = match (layout, trans_b) {
                        (Layout::RowMajor, Transpose::NoTrans) => l * ldb + j,
                        (Layout::RowMajor, _) => j * ldb + l,
                        (Layout::ColMajor, Transpose::NoTrans) => j * ldb + l,
                        (Layout::ColMajor, _) => l * ldb + j,
                    };
                    sum += a[a_idx] * b[b_idx];
                }
                let c_idx = match layout {
                    Layout::RowMajor => i * ldc + j,
                    Layout::ColMajor => j * ldc + i,
                };
                c[c_idx] += alpha * sum;
            }
        }
    }

    fn sgemv(
        &self,
        layout: Layout,
        trans: Transpose,
        m: i32,
        n: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        x: &[f32],
        incx: i32,
        beta: f32,
        y: &mut [f32],
        incy: i32,
    ) {
        let m = m as usize;
        let n = n as usize;
        let lda = lda as usize;
        let incx = incx as usize;
        let incy = incy as usize;

        let (rows, cols) = match trans {
            Transpose::NoTrans => (m, n),
            _ => (n, m),
        };

        for i in 0..rows {
            y[i * incy] *= beta;
        }

        for i in 0..rows {
            let mut sum = 0.0f32;
            for j in 0..cols {
                let a_idx = match (layout, trans) {
                    (Layout::RowMajor, Transpose::NoTrans) => i * lda + j,
                    (Layout::RowMajor, _) => j * lda + i,
                    (Layout::ColMajor, Transpose::NoTrans) => j * lda + i,
                    (Layout::ColMajor, _) => i * lda + j,
                };
                sum += a[a_idx] * x[j * incx];
            }
            y[i * incy] += alpha * sum;
        }
    }

    fn sdot(&self, n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
        let n = n as usize;
        let incx = incx as usize;
        let incy = incy as usize;

        (0..n).map(|i| x[i * incx] * y[i * incy]).sum()
    }

    fn snrm2(&self, n: i32, x: &[f32], incx: i32) -> f32 {
        let n = n as usize;
        let incx = incx as usize;

        (0..n)
            .map(|i| x[i * incx] * x[i * incx])
            .sum::<f32>()
            .sqrt()
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut [f32], incx: i32) {
        let n = n as usize;
        let incx = incx as usize;

        for i in 0..n {
            x[i * incx] *= alpha;
        }
    }

    fn saxpy(&self, n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        let n = n as usize;
        let incx = incx as usize;
        let incy = incy as usize;

        for i in 0..n {
            y[i * incy] += alpha * x[i * incx];
        }
    }
}

// ============================================================
// SIMD-Optimized Pure Rust Implementation
// ============================================================

/// SIMD-optimized Rust BLAS implementation
#[derive(Debug, Clone, Copy)]
pub struct SimdRustBlas;

impl BlasProviderF64 for SimdRustBlas {
    fn name(&self) -> &'static str {
        "SimdRust"
    }

    fn dgemm(
        &self,
        layout: Layout,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: &[f64],
        lda: i32,
        b: &[f64],
        ldb: i32,
        beta: f64,
        c: &mut [f64],
        ldc: i32,
    ) {
        // Use tiled matrix multiplication for better cache performance
        const TILE_SIZE: usize = 32;

        let m = m as usize;
        let n = n as usize;
        let k = k as usize;
        let lda = lda as usize;
        let ldb = ldb as usize;
        let ldc = ldc as usize;

        // Apply beta to C first
        if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let idx = match layout {
                        Layout::RowMajor => i * ldc + j,
                        Layout::ColMajor => j * ldc + i,
                    };
                    c[idx] *= beta;
                }
            }
        }

        // Tiled multiplication
        for i0 in (0..m).step_by(TILE_SIZE) {
            for j0 in (0..n).step_by(TILE_SIZE) {
                for l0 in (0..k).step_by(TILE_SIZE) {
                    let i_end = (i0 + TILE_SIZE).min(m);
                    let j_end = (j0 + TILE_SIZE).min(n);
                    let l_end = (l0 + TILE_SIZE).min(k);

                    for i in i0..i_end {
                        for j in j0..j_end {
                            let mut sum = 0.0;
                            for l in l0..l_end {
                                let a_idx = match (layout, trans_a) {
                                    (Layout::RowMajor, Transpose::NoTrans) => i * lda + l,
                                    (Layout::RowMajor, _) => l * lda + i,
                                    (Layout::ColMajor, Transpose::NoTrans) => l * lda + i,
                                    (Layout::ColMajor, _) => i * lda + l,
                                };
                                let b_idx = match (layout, trans_b) {
                                    (Layout::RowMajor, Transpose::NoTrans) => l * ldb + j,
                                    (Layout::RowMajor, _) => j * ldb + l,
                                    (Layout::ColMajor, Transpose::NoTrans) => j * ldb + l,
                                    (Layout::ColMajor, _) => l * ldb + j,
                                };
                                sum += a[a_idx] * b[b_idx];
                            }
                            let c_idx = match layout {
                                Layout::RowMajor => i * ldc + j,
                                Layout::ColMajor => j * ldc + i,
                            };
                            c[c_idx] += alpha * sum;
                        }
                    }
                }
            }
        }
    }

    fn dgemv(
        &self,
        layout: Layout,
        trans: Transpose,
        m: i32,
        n: i32,
        alpha: f64,
        a: &[f64],
        lda: i32,
        x: &[f64],
        incx: i32,
        beta: f64,
        y: &mut [f64],
        incy: i32,
    ) {
        // Delegate to pure rust for now
        PureRustBlas.dgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }

    fn ddot(&self, n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
        let n = n as usize;
        let incx = incx as usize;
        let incy = incy as usize;

        if incx == 1 && incy == 1 {
            // SIMD-friendly path
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx") {
                    return simd_ddot_avx(x, y, n);
                }
            }
        }

        // Fallback
        (0..n).map(|i| x[i * incx] * y[i * incy]).sum()
    }

    fn dnrm2(&self, n: i32, x: &[f64], incx: i32) -> f64 {
        let n = n as usize;
        let incx = incx as usize;

        if incx == 1 {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx") {
                    return simd_dnrm2_avx(x, n);
                }
            }
        }

        (0..n)
            .map(|i| x[i * incx] * x[i * incx])
            .sum::<f64>()
            .sqrt()
    }

    fn dasum(&self, n: i32, x: &[f64], incx: i32) -> f64 {
        PureRustBlas.dasum(n, x, incx)
    }

    fn dscal(&self, n: i32, alpha: f64, x: &mut [f64], incx: i32) {
        PureRustBlas.dscal(n, alpha, x, incx)
    }

    fn daxpy(&self, n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
        PureRustBlas.daxpy(n, alpha, x, incx, y, incy)
    }

    fn dcopy(&self, n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
        PureRustBlas.dcopy(n, x, incx, y, incy)
    }

    fn idamax(&self, n: i32, x: &[f64], incx: i32) -> i32 {
        PureRustBlas.idamax(n, x, incx)
    }
}

impl BlasProviderF32 for SimdRustBlas {
    fn name(&self) -> &'static str {
        "SimdRust"
    }

    fn sgemm(
        &self,
        layout: Layout,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        b: &[f32],
        ldb: i32,
        beta: f32,
        c: &mut [f32],
        ldc: i32,
    ) {
        // Use tiled matrix multiplication
        const TILE_SIZE: usize = 32;

        let m = m as usize;
        let n = n as usize;
        let k = k as usize;
        let lda = lda as usize;
        let ldb = ldb as usize;
        let ldc = ldc as usize;

        if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let idx = match layout {
                        Layout::RowMajor => i * ldc + j,
                        Layout::ColMajor => j * ldc + i,
                    };
                    c[idx] *= beta;
                }
            }
        }

        for i0 in (0..m).step_by(TILE_SIZE) {
            for j0 in (0..n).step_by(TILE_SIZE) {
                for l0 in (0..k).step_by(TILE_SIZE) {
                    let i_end = (i0 + TILE_SIZE).min(m);
                    let j_end = (j0 + TILE_SIZE).min(n);
                    let l_end = (l0 + TILE_SIZE).min(k);

                    for i in i0..i_end {
                        for j in j0..j_end {
                            let mut sum = 0.0f32;
                            for l in l0..l_end {
                                let a_idx = match (layout, trans_a) {
                                    (Layout::RowMajor, Transpose::NoTrans) => i * lda + l,
                                    (Layout::RowMajor, _) => l * lda + i,
                                    (Layout::ColMajor, Transpose::NoTrans) => l * lda + i,
                                    (Layout::ColMajor, _) => i * lda + l,
                                };
                                let b_idx = match (layout, trans_b) {
                                    (Layout::RowMajor, Transpose::NoTrans) => l * ldb + j,
                                    (Layout::RowMajor, _) => j * ldb + l,
                                    (Layout::ColMajor, Transpose::NoTrans) => j * ldb + l,
                                    (Layout::ColMajor, _) => l * ldb + j,
                                };
                                sum += a[a_idx] * b[b_idx];
                            }
                            let c_idx = match layout {
                                Layout::RowMajor => i * ldc + j,
                                Layout::ColMajor => j * ldc + i,
                            };
                            c[c_idx] += alpha * sum;
                        }
                    }
                }
            }
        }
    }

    fn sgemv(
        &self,
        layout: Layout,
        trans: Transpose,
        m: i32,
        n: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        x: &[f32],
        incx: i32,
        beta: f32,
        y: &mut [f32],
        incy: i32,
    ) {
        PureRustBlas.sgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }

    fn sdot(&self, n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
        PureRustBlas.sdot(n, x, incx, y, incy)
    }

    fn snrm2(&self, n: i32, x: &[f32], incx: i32) -> f32 {
        PureRustBlas.snrm2(n, x, incx)
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut [f32], incx: i32) {
        PureRustBlas.sscal(n, alpha, x, incx)
    }

    fn saxpy(&self, n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        PureRustBlas.saxpy(n, alpha, x, incx, y, incy)
    }
}

// SIMD helper functions
#[cfg(target_arch = "x86_64")]
fn simd_ddot_avx(x: &[f64], y: &[f64], n: usize) -> f64 {
    use std::arch::x86_64::*;

    let mut sum = 0.0;
    let chunks = n / 4;

    unsafe {
        let mut acc = _mm256_setzero_pd();

        for i in 0..chunks {
            let xv = _mm256_loadu_pd(x.as_ptr().add(i * 4));
            let yv = _mm256_loadu_pd(y.as_ptr().add(i * 4));
            acc = _mm256_fmadd_pd(xv, yv, acc);
        }

        // Horizontal sum
        let low = _mm256_castpd256_pd128(acc);
        let high = _mm256_extractf128_pd(acc, 1);
        let sum128 = _mm_add_pd(low, high);
        let sum128_high = _mm_unpackhi_pd(sum128, sum128);
        let result = _mm_add_sd(sum128, sum128_high);
        sum = _mm_cvtsd_f64(result);
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        sum += x[i] * y[i];
    }

    sum
}

#[cfg(target_arch = "x86_64")]
fn simd_dnrm2_avx(x: &[f64], n: usize) -> f64 {
    use std::arch::x86_64::*;

    let mut sum = 0.0;
    let chunks = n / 4;

    unsafe {
        let mut acc = _mm256_setzero_pd();

        for i in 0..chunks {
            let xv = _mm256_loadu_pd(x.as_ptr().add(i * 4));
            acc = _mm256_fmadd_pd(xv, xv, acc);
        }

        let low = _mm256_castpd256_pd128(acc);
        let high = _mm256_extractf128_pd(acc, 1);
        let sum128 = _mm_add_pd(low, high);
        let sum128_high = _mm_unpackhi_pd(sum128, sum128);
        let result = _mm_add_sd(sum128, sum128_high);
        sum = _mm_cvtsd_f64(result);
    }

    for i in (chunks * 4)..n {
        sum += x[i] * x[i];
    }

    sum.sqrt()
}

// ============================================================
// Provider Selection
// ============================================================

/// Available BLAS providers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlasBackend {
    /// Pure Rust (fallback)
    PureRust,
    /// SIMD-optimized Rust
    SimdRust,
    /// OpenBLAS
    OpenBlas,
    /// Intel MKL
    Mkl,
    /// Apple Accelerate
    Accelerate,
}

impl BlasBackend {
    /// Check if this backend is available on the current system
    pub fn is_available(&self) -> bool {
        match self {
            BlasBackend::PureRust => true,
            BlasBackend::SimdRust => true,
            #[cfg(feature = "openblas")]
            BlasBackend::OpenBlas => true, // Available when feature is enabled
            #[cfg(not(feature = "openblas"))]
            BlasBackend::OpenBlas => false,
            #[cfg(feature = "mkl")]
            BlasBackend::Mkl => true, // Available when feature is enabled
            #[cfg(not(feature = "mkl"))]
            BlasBackend::Mkl => false,
            #[cfg(all(target_os = "macos", feature = "accelerate"))]
            BlasBackend::Accelerate => true, // Available on macOS with feature
            #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
            BlasBackend::Accelerate => false,
        }
    }

    /// Get provider name
    pub fn name(&self) -> &'static str {
        match self {
            BlasBackend::PureRust => "PureRust",
            BlasBackend::SimdRust => "SimdRust",
            BlasBackend::OpenBlas => "OpenBLAS",
            BlasBackend::Mkl => "Intel MKL",
            BlasBackend::Accelerate => "Apple Accelerate",
        }
    }
}

/// Global BLAS provider selection
static DEFAULT_PROVIDER: OnceLock<BlasBackend> = OnceLock::new();

/// Get the default BLAS provider for f64
pub fn default_provider_f64() -> &'static dyn BlasProviderF64 {
    static PROVIDER: OnceLock<Box<dyn BlasProviderF64>> = OnceLock::new();
    PROVIDER
        .get_or_init(|| {
            let backend = detect_best_backend();
            match backend {
                #[cfg(all(target_os = "macos", feature = "accelerate"))]
                BlasBackend::Accelerate => Box::new(AccelerateProvider),
                #[cfg(feature = "openblas")]
                BlasBackend::OpenBlas => Box::new(OpenBlasProvider),
                BlasBackend::SimdRust => Box::new(SimdRustBlas),
                _ => Box::new(PureRustBlas),
            }
        })
        .as_ref()
}

/// Get the default BLAS provider for f32
pub fn default_provider_f32() -> &'static dyn BlasProviderF32 {
    static PROVIDER: OnceLock<Box<dyn BlasProviderF32>> = OnceLock::new();
    PROVIDER
        .get_or_init(|| {
            let backend = detect_best_backend();
            match backend {
                #[cfg(all(target_os = "macos", feature = "accelerate"))]
                BlasBackend::Accelerate => Box::new(AccelerateProvider),
                #[cfg(feature = "openblas")]
                BlasBackend::OpenBlas => Box::new(OpenBlasProvider),
                BlasBackend::SimdRust => Box::new(SimdRustBlas),
                _ => Box::new(PureRustBlas),
            }
        })
        .as_ref()
}

/// Detect the best available BLAS backend
pub fn detect_best_backend() -> BlasBackend {
    *DEFAULT_PROVIDER.get_or_init(|| {
        // Priority: Accelerate > MKL > OpenBLAS > SimdRust > PureRust
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        if BlasBackend::Accelerate.is_available() {
            return BlasBackend::Accelerate;
        }

        #[cfg(feature = "mkl")]
        if BlasBackend::Mkl.is_available() {
            return BlasBackend::Mkl;
        }

        #[cfg(feature = "openblas")]
        if BlasBackend::OpenBlas.is_available() {
            return BlasBackend::OpenBlas;
        }

        // Default to SIMD-optimized Rust
        BlasBackend::SimdRust
    })
}

/// Set the default BLAS backend
pub fn set_default_backend(backend: BlasBackend) -> Result<(), &'static str> {
    if !backend.is_available() {
        return Err("Requested BLAS backend is not available");
    }
    // Note: OnceLock can only be set once, so this is a no-op after first call
    let _ = DEFAULT_PROVIDER.set(backend);
    Ok(())
}

// ============================================================
// FFI Exports
// ============================================================

/// Get the name of the active BLAS provider
#[no_mangle]
pub extern "C" fn bhc_blas_provider_name() -> *const std::ffi::c_char {
    static NAME: OnceLock<std::ffi::CString> = OnceLock::new();
    NAME.get_or_init(|| {
        let name = detect_best_backend().name();
        std::ffi::CString::new(name).unwrap()
    })
    .as_ptr()
}

/// DGEMM: Double precision matrix-matrix multiply
#[no_mangle]
pub extern "C" fn bhc_blas_dgemm(
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
) {
    let a_slice = unsafe { std::slice::from_raw_parts(a, (m * lda) as usize) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, (k * ldb) as usize) };
    let c_slice = unsafe { std::slice::from_raw_parts_mut(c, (m * ldc) as usize) };

    default_provider_f64().dgemm(
        Layout::RowMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        alpha,
        a_slice,
        lda,
        b_slice,
        ldb,
        beta,
        c_slice,
        ldc,
    );
}

/// DDOT: Double precision dot product
#[no_mangle]
pub extern "C" fn bhc_blas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64 {
    let x_slice = unsafe { std::slice::from_raw_parts(x, (n * incx) as usize) };
    let y_slice = unsafe { std::slice::from_raw_parts(y, (n * incy) as usize) };

    default_provider_f64().ddot(n, x_slice, incx, y_slice, incy)
}

/// DNRM2: Double precision Euclidean norm
#[no_mangle]
pub extern "C" fn bhc_blas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64 {
    let x_slice = unsafe { std::slice::from_raw_parts(x, (n * incx) as usize) };

    default_provider_f64().dnrm2(n, x_slice, incx)
}

/// DSCAL: Double precision scale vector
#[no_mangle]
pub extern "C" fn bhc_blas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32) {
    let x_slice = unsafe { std::slice::from_raw_parts_mut(x, (n * incx) as usize) };

    default_provider_f64().dscal(n, alpha, x_slice, incx);
}

/// DAXPY: Double precision y = alpha * x + y
#[no_mangle]
pub extern "C" fn bhc_blas_daxpy(
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
) {
    let x_slice = unsafe { std::slice::from_raw_parts(x, (n * incx) as usize) };
    let y_slice = unsafe { std::slice::from_raw_parts_mut(y, (n * incy) as usize) };

    default_provider_f64().daxpy(n, alpha, x_slice, incx, y_slice, incy);
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddot() {
        let blas = PureRustBlas;
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        assert_eq!(blas.ddot(3, &x, 1, &y, 1), 32.0);
    }

    #[test]
    fn test_daxpy() {
        let blas = PureRustBlas;
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        blas.daxpy(3, 2.0, &x, 1, &mut y, 1);
        assert_eq!(y, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_dgemm_identity() {
        let blas = PureRustBlas;
        // 2x2 identity matrix multiply
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];

        blas.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );

        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dgemm_simple() {
        let blas = PureRustBlas;
        // [1 2] * [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        blas.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );

        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_dnrm2() {
        let blas = PureRustBlas;
        let x = vec![3.0, 4.0];
        assert!((blas.dnrm2(2, &x, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_provider_detection() {
        let backend = detect_best_backend();
        assert!(backend.is_available());
        println!("Detected BLAS backend: {:?}", backend);
    }

    #[test]
    fn test_simd_gemm() {
        let blas = SimdRustBlas;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        blas.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );

        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_ddot_large() {
        if !is_x86_feature_detected!("avx") {
            return;
        }

        let n = 1000;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();

        let result = simd_ddot_avx(&x, &y, n);
        let expected: f64 = (0..n).map(|i| (i * i * 2) as f64).sum();

        assert!((result - expected).abs() < 1e-6);
    }
}
