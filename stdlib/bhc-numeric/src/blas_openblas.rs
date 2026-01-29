//! OpenBLAS provider implementation
//!
//! This module provides BLAS operations using OpenBLAS as the backend.
//! OpenBLAS is a high-performance open-source BLAS library that supports
//! multi-threading and various CPU architectures.
//!
//! # Features
//!
//! Enable with the `openblas` feature:
//! ```toml
//! [dependencies]
//! bhc-numeric = { version = "0.1", features = ["openblas"] }
//! ```
//!
//! # Performance
//!
//! OpenBLAS provides highly optimized implementations for:
//! - Intel Haswell, Skylake, Ice Lake
//! - AMD Zen, Zen2, Zen3
//! - ARM Cortex-A53, A57, A72, A73
//!
//! Performance is typically within 90-100% of Intel MKL for most operations.

use crate::blas::{BlasProviderF32, BlasProviderF64, Layout, Transpose};

// ============================================================
// CBLAS FFI Declarations
// ============================================================

#[allow(non_camel_case_types)]
type CBLAS_ORDER = i32;
#[allow(non_camel_case_types)]
type CBLAS_TRANSPOSE = i32;

const CBLAS_ROW_MAJOR: CBLAS_ORDER = 101;
const CBLAS_COL_MAJOR: CBLAS_ORDER = 102;
const CBLAS_NO_TRANS: CBLAS_TRANSPOSE = 111;
const CBLAS_TRANS: CBLAS_TRANSPOSE = 112;
const CBLAS_CONJ_TRANS: CBLAS_TRANSPOSE = 113;

#[cfg(feature = "openblas")]
#[link(name = "openblas")]
extern "C" {
    // Level 1 BLAS (f64)
    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
    fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;
    fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64;
    fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32;
    fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32);
    fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32);
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32);

    // Level 2 BLAS (f64)
    fn cblas_dgemv(
        order: CBLAS_ORDER,
        trans: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );

    // Level 3 BLAS (f64)
    fn cblas_dgemm(
        order: CBLAS_ORDER,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
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

    // Level 1 BLAS (f32)
    fn cblas_sdot(n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) -> f32;
    fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32;
    fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32);
    fn cblas_saxpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32);

    // Level 2 BLAS (f32)
    fn cblas_sgemv(
        order: CBLAS_ORDER,
        trans: CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );

    // Level 3 BLAS (f32)
    fn cblas_sgemm(
        order: CBLAS_ORDER,
        transa: CBLAS_TRANSPOSE,
        transb: CBLAS_TRANSPOSE,
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
}

// ============================================================
// Helper Conversions
// ============================================================

fn to_cblas_order(layout: Layout) -> CBLAS_ORDER {
    match layout {
        Layout::RowMajor => CBLAS_ROW_MAJOR,
        Layout::ColMajor => CBLAS_COL_MAJOR,
    }
}

fn to_cblas_trans(trans: Transpose) -> CBLAS_TRANSPOSE {
    match trans {
        Transpose::NoTrans => CBLAS_NO_TRANS,
        Transpose::Trans => CBLAS_TRANS,
        Transpose::ConjTrans => CBLAS_CONJ_TRANS,
    }
}

// ============================================================
// OpenBLAS Provider
// ============================================================

/// OpenBLAS BLAS provider.
///
/// This provider uses OpenBLAS for high-performance BLAS operations.
/// It automatically detects CPU features and uses optimized kernels.
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::blas::{BlasProviderF64, Layout, Transpose};
/// use bhc_numeric::blas_openblas::OpenBlasProvider;
///
/// let provider = OpenBlasProvider;
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![5.0, 6.0, 7.0, 8.0];
/// let mut c = vec![0.0; 4];
///
/// provider.dgemm(
///     Layout::RowMajor,
///     Transpose::NoTrans, Transpose::NoTrans,
///     2, 2, 2,
///     1.0, &a, 2, &b, 2,
///     0.0, &mut c, 2
/// );
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenBlasProvider;

impl OpenBlasProvider {
    /// Create a new OpenBLAS provider.
    pub fn new() -> Self {
        OpenBlasProvider
    }

    /// Check if OpenBLAS is available.
    #[cfg(feature = "openblas")]
    pub fn is_available() -> bool {
        true
    }

    #[cfg(not(feature = "openblas"))]
    pub fn is_available() -> bool {
        false
    }
}

#[cfg(feature = "openblas")]
impl BlasProviderF64 for OpenBlasProvider {
    fn name(&self) -> &'static str {
        "OpenBLAS"
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
        unsafe {
            cblas_dgemm(
                to_cblas_order(layout),
                to_cblas_trans(trans_a),
                to_cblas_trans(trans_b),
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                beta,
                c.as_mut_ptr(),
                ldc,
            );
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
        unsafe {
            cblas_dgemv(
                to_cblas_order(layout),
                to_cblas_trans(trans),
                m,
                n,
                alpha,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                incx,
                beta,
                y.as_mut_ptr(),
                incy,
            );
        }
    }

    fn ddot(&self, n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
        unsafe { cblas_ddot(n, x.as_ptr(), incx, y.as_ptr(), incy) }
    }

    fn dnrm2(&self, n: i32, x: &[f64], incx: i32) -> f64 {
        unsafe { cblas_dnrm2(n, x.as_ptr(), incx) }
    }

    fn dasum(&self, n: i32, x: &[f64], incx: i32) -> f64 {
        unsafe { cblas_dasum(n, x.as_ptr(), incx) }
    }

    fn dscal(&self, n: i32, alpha: f64, x: &mut [f64], incx: i32) {
        unsafe {
            cblas_dscal(n, alpha, x.as_mut_ptr(), incx);
        }
    }

    fn daxpy(&self, n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
        unsafe {
            cblas_daxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy);
        }
    }

    fn dcopy(&self, n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
        unsafe {
            cblas_dcopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy);
        }
    }

    fn idamax(&self, n: i32, x: &[f64], incx: i32) -> i32 {
        unsafe { cblas_idamax(n, x.as_ptr(), incx) }
    }
}

#[cfg(feature = "openblas")]
impl BlasProviderF32 for OpenBlasProvider {
    fn name(&self) -> &'static str {
        "OpenBLAS"
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
        unsafe {
            cblas_sgemm(
                to_cblas_order(layout),
                to_cblas_trans(trans_a),
                to_cblas_trans(trans_b),
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                beta,
                c.as_mut_ptr(),
                ldc,
            );
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
        unsafe {
            cblas_sgemv(
                to_cblas_order(layout),
                to_cblas_trans(trans),
                m,
                n,
                alpha,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                incx,
                beta,
                y.as_mut_ptr(),
                incy,
            );
        }
    }

    fn sdot(&self, n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
        unsafe { cblas_sdot(n, x.as_ptr(), incx, y.as_ptr(), incy) }
    }

    fn snrm2(&self, n: i32, x: &[f32], incx: i32) -> f32 {
        unsafe { cblas_snrm2(n, x.as_ptr(), incx) }
    }

    fn sscal(&self, n: i32, alpha: f32, x: &mut [f32], incx: i32) {
        unsafe {
            cblas_sscal(n, alpha, x.as_mut_ptr(), incx);
        }
    }

    fn saxpy(&self, n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        unsafe {
            cblas_saxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy);
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
#[cfg(feature = "openblas")]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let p = OpenBlasProvider;
        assert_eq!(p.name(), "OpenBLAS");
    }

    #[test]
    fn test_ddot() {
        let p = OpenBlasProvider;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 6.0, 7.0, 8.0];
        let result = p.ddot(4, &x, 1, &y, 1);
        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_dgemm() {
        let p = OpenBlasProvider;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        p.dgemm(
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
}
