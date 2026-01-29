//! Apple Accelerate framework provider
//!
//! This module provides BLAS operations using Apple's Accelerate framework,
//! which includes highly optimized implementations for Apple Silicon (M1/M2/M3)
//! and Intel Macs.
//!
//! # Features
//!
//! Enable with the `accelerate` feature:
//! ```toml
//! [dependencies]
//! bhc-numeric = { version = "0.1", features = ["accelerate"] }
//! ```
//!
//! # Performance
//!
//! Accelerate is specifically optimized for Apple hardware:
//! - Apple Silicon: Uses AMX (Apple Matrix coprocessor) for matrix operations
//! - Intel Macs: Uses highly tuned AVX/AVX2 implementations
//!
//! On Apple Silicon, Accelerate typically outperforms OpenBLAS by 2-3x
//! for matrix operations due to AMX acceleration.
//!
//! # Platform Support
//!
//! This module is only available on macOS. It will not compile on other platforms.

use crate::blas::{BlasProviderF32, BlasProviderF64, Layout, Transpose};

// ============================================================
// Accelerate FFI Declarations
// ============================================================

#[cfg(all(target_os = "macos", feature = "accelerate"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    // CBLAS interface from Accelerate
    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
    fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;
    fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64;
    fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32;
    fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32);
    fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32);
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32);

    fn cblas_dgemv(
        order: i32,
        trans: i32,
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

    fn cblas_dgemm(
        order: i32,
        transa: i32,
        transb: i32,
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

    // Single precision
    fn cblas_sdot(n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) -> f32;
    fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32;
    fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32);
    fn cblas_saxpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32);

    fn cblas_sgemv(
        order: i32,
        trans: i32,
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

    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
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
// CBLAS Constants
// ============================================================

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_COL_MAJOR: i32 = 102;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;
const CBLAS_CONJ_TRANS: i32 = 113;

// ============================================================
// Helper Conversions
// ============================================================

fn to_cblas_order(layout: Layout) -> i32 {
    match layout {
        Layout::RowMajor => CBLAS_ROW_MAJOR,
        Layout::ColMajor => CBLAS_COL_MAJOR,
    }
}

fn to_cblas_trans(trans: Transpose) -> i32 {
    match trans {
        Transpose::NoTrans => CBLAS_NO_TRANS,
        Transpose::Trans => CBLAS_TRANS,
        Transpose::ConjTrans => CBLAS_CONJ_TRANS,
    }
}

// ============================================================
// Accelerate Provider
// ============================================================

/// Apple Accelerate BLAS provider.
///
/// This provider uses Apple's Accelerate framework for high-performance
/// BLAS operations on macOS.
///
/// # Performance Notes
///
/// On Apple Silicon (M1/M2/M3), Accelerate uses the AMX (Apple Matrix
/// coprocessor) for matrix operations, providing exceptional performance
/// that typically exceeds OpenBLAS by 2-3x for large matrices.
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::blas::{BlasProviderF64, Layout, Transpose};
/// use bhc_numeric::blas_accelerate::AccelerateProvider;
///
/// let provider = AccelerateProvider;
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
pub struct AccelerateProvider;

impl AccelerateProvider {
    /// Create a new Accelerate provider.
    pub fn new() -> Self {
        AccelerateProvider
    }

    /// Check if Accelerate is available (macOS only).
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    pub fn is_available() -> bool {
        true
    }

    #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
    pub fn is_available() -> bool {
        false
    }

    /// Check if running on Apple Silicon.
    #[cfg(target_os = "macos")]
    pub fn is_apple_silicon() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            true
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn is_apple_silicon() -> bool {
        false
    }
}

#[cfg(all(target_os = "macos", feature = "accelerate"))]
impl BlasProviderF64 for AccelerateProvider {
    fn name(&self) -> &'static str {
        if Self::is_apple_silicon() {
            "Accelerate (Apple Silicon)"
        } else {
            "Accelerate (Intel)"
        }
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

#[cfg(all(target_os = "macos", feature = "accelerate"))]
impl BlasProviderF32 for AccelerateProvider {
    fn name(&self) -> &'static str {
        if Self::is_apple_silicon() {
            "Accelerate (Apple Silicon)"
        } else {
            "Accelerate (Intel)"
        }
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
#[cfg(all(target_os = "macos", feature = "accelerate"))]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let p = AccelerateProvider;
        // Disambiguate by calling through trait
        let name = BlasProviderF64::name(&p);
        assert!(name.starts_with("Accelerate"));
    }

    #[test]
    fn test_ddot() {
        let p = AccelerateProvider;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 6.0, 7.0, 8.0];
        let result = p.ddot(4, &x, 1, &y, 1);
        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_dgemm() {
        let p = AccelerateProvider;
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

    #[test]
    fn test_is_apple_silicon() {
        // This will be true on M1/M2/M3 Macs, false on Intel Macs
        let is_arm = AccelerateProvider::is_apple_silicon();
        println!("Running on Apple Silicon: {}", is_arm);
    }
}
