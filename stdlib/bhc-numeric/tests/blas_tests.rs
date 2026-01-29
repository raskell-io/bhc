//! BLAS provider tests
//!
//! Tests for BLAS operations across different providers.
//! These tests verify correctness of:
//! - Level 1 BLAS (vector-vector operations)
//! - Level 2 BLAS (matrix-vector operations)
//! - Level 3 BLAS (matrix-matrix operations)
//!
//! Tests run against PureRustBlas by default and can be parameterized
//! for OpenBLAS, MKL, or Accelerate when those features are enabled.

use bhc_numeric::blas::{BlasProviderF32, BlasProviderF64, Layout, PureRustBlas, Transpose};

// ============================================================
// Level 1 BLAS: Vector-Vector Operations
// ============================================================

mod level1_tests {
    use super::*;

    fn provider() -> PureRustBlas {
        PureRustBlas
    }

    // ------------------------------------------------------------
    // DDOT: Dot product
    // ------------------------------------------------------------

    #[test]
    fn test_ddot_basic() {
        let p = provider();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 6.0, 7.0, 8.0];

        let result = p.ddot(4, &x, 1, &y, 1);

        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_ddot_strided() {
        let p = provider();
        let x = vec![1.0, 0.0, 2.0, 0.0, 3.0]; // use stride 2
        let y = vec![4.0, 5.0, 6.0];

        let result = p.ddot(3, &x, 2, &y, 1);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_ddot_single_element() {
        let p = provider();
        let x = vec![5.0];
        let y = vec![3.0];

        let result = p.ddot(1, &x, 1, &y, 1);

        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_ddot_orthogonal() {
        let p = provider();
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];

        let result = p.ddot(3, &x, 1, &y, 1);

        assert_eq!(result, 0.0);
    }

    // ------------------------------------------------------------
    // DNRM2: Euclidean norm
    // ------------------------------------------------------------

    #[test]
    fn test_dnrm2_basic() {
        let p = provider();
        let x = vec![3.0, 4.0];

        let result = p.dnrm2(2, &x, 1);

        assert_eq!(result, 5.0); // sqrt(9 + 16)
    }

    #[test]
    fn test_dnrm2_unit() {
        let p = provider();
        let x = vec![1.0, 0.0, 0.0];

        let result = p.dnrm2(3, &x, 1);

        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_dnrm2_zeros() {
        let p = provider();
        let x = vec![0.0, 0.0, 0.0];

        let result = p.dnrm2(3, &x, 1);

        assert_eq!(result, 0.0);
    }

    // ------------------------------------------------------------
    // DASUM: Sum of absolute values
    // ------------------------------------------------------------

    #[test]
    fn test_dasum_basic() {
        let p = provider();
        let x = vec![1.0, -2.0, 3.0, -4.0];

        let result = p.dasum(4, &x, 1);

        assert_eq!(result, 10.0); // 1 + 2 + 3 + 4
    }

    #[test]
    fn test_dasum_all_positive() {
        let p = provider();
        let x = vec![1.0, 2.0, 3.0];

        let result = p.dasum(3, &x, 1);

        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_dasum_all_negative() {
        let p = provider();
        let x = vec![-1.0, -2.0, -3.0];

        let result = p.dasum(3, &x, 1);

        assert_eq!(result, 6.0);
    }

    // ------------------------------------------------------------
    // DSCAL: Scale vector
    // ------------------------------------------------------------

    #[test]
    fn test_dscal_basic() {
        let p = provider();
        let mut x = vec![1.0, 2.0, 3.0, 4.0];

        p.dscal(4, 2.0, &mut x, 1);

        assert_eq!(x, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_dscal_zero() {
        let p = provider();
        let mut x = vec![1.0, 2.0, 3.0];

        p.dscal(3, 0.0, &mut x, 1);

        assert_eq!(x, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dscal_negative() {
        let p = provider();
        let mut x = vec![1.0, -2.0, 3.0];

        p.dscal(3, -1.0, &mut x, 1);

        assert_eq!(x, vec![-1.0, 2.0, -3.0]);
    }

    // ------------------------------------------------------------
    // DAXPY: y = alpha*x + y
    // ------------------------------------------------------------

    #[test]
    fn test_daxpy_basic() {
        let p = provider();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![5.0, 6.0, 7.0, 8.0];

        p.daxpy(4, 2.0, &x, 1, &mut y, 1);

        // y = 2*x + y
        assert_eq!(y, vec![7.0, 10.0, 13.0, 16.0]);
    }

    #[test]
    fn test_daxpy_alpha_one() {
        let p = provider();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];

        p.daxpy(3, 1.0, &x, 1, &mut y, 1);

        assert_eq!(y, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_daxpy_alpha_zero() {
        let p = provider();
        let x = vec![100.0, 200.0, 300.0];
        let mut y = vec![1.0, 2.0, 3.0];

        p.daxpy(3, 0.0, &x, 1, &mut y, 1);

        assert_eq!(y, vec![1.0, 2.0, 3.0]); // y unchanged
    }

    // ------------------------------------------------------------
    // DCOPY: Copy vector
    // ------------------------------------------------------------

    #[test]
    fn test_dcopy_basic() {
        let p = provider();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];

        p.dcopy(4, &x, 1, &mut y, 1);

        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dcopy_strided() {
        let p = provider();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 8];

        p.dcopy(4, &x, 1, &mut y, 2);

        assert_eq!(y, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
    }

    // ------------------------------------------------------------
    // IDAMAX: Index of max absolute value
    // ------------------------------------------------------------

    #[test]
    fn test_idamax_basic() {
        let p = provider();
        let x = vec![1.0, -5.0, 3.0, 2.0];

        let result = p.idamax(4, &x, 1);

        assert_eq!(result, 1); // |-5| is max (0-indexed)
    }

    #[test]
    fn test_idamax_first_element() {
        let p = provider();
        let x = vec![10.0, 1.0, 2.0];

        let result = p.idamax(3, &x, 1);

        assert_eq!(result, 0);
    }

    #[test]
    fn test_idamax_ties_first() {
        let p = provider();
        let x = vec![5.0, -5.0, 5.0];

        let result = p.idamax(3, &x, 1);

        assert_eq!(result, 0); // First occurrence
    }
}

// ============================================================
// Level 2 BLAS: Matrix-Vector Operations
// ============================================================

mod level2_tests {
    use super::*;

    fn provider() -> PureRustBlas {
        PureRustBlas
    }

    // ------------------------------------------------------------
    // DGEMV: y = alpha * op(A) * x + beta * y
    // ------------------------------------------------------------

    #[test]
    fn test_dgemv_no_trans() {
        let p = provider();
        // A = [[1, 2], [3, 4]] (2x2, row-major)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0, 0.0];

        p.dgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            2,
            2,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );

        // y = A * x = [[1*1 + 2*2], [3*1 + 4*2]] = [5, 11]
        assert_eq!(y, vec![5.0, 11.0]);
    }

    #[test]
    fn test_dgemv_trans() {
        let p = provider();
        // A = [[1, 2], [3, 4]] (2x2, row-major)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0, 0.0];

        p.dgemv(
            Layout::RowMajor,
            Transpose::Trans,
            2,
            2,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );

        // y = A^T * x = [[1*1 + 3*2], [2*1 + 4*2]] = [7, 10]
        assert_eq!(y, vec![7.0, 10.0]);
    }

    #[test]
    fn test_dgemv_with_beta() {
        let p = provider();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![10.0, 10.0];

        p.dgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            2,
            2,
            1.0,
            &a,
            2,
            &x,
            1,
            2.0,
            &mut y,
            1,
        );

        // y = 1.0 * A * x + 2.0 * y
        // A * x = [3, 7]
        // result = [3 + 20, 7 + 20] = [23, 27]
        assert_eq!(y, vec![23.0, 27.0]);
    }

    #[test]
    fn test_dgemv_rect_matrix() {
        let p = provider();
        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0];

        p.dgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            2,
            3,
            1.0,
            &a,
            3,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );

        // y = A * x = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
        assert_eq!(y, vec![14.0, 32.0]);
    }

    #[test]
    fn test_dgemv_alpha_scaling() {
        let p = provider();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0, 0.0];

        p.dgemv(
            Layout::RowMajor,
            Transpose::NoTrans,
            2,
            2,
            3.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );

        // y = 3 * [3, 7] = [9, 21]
        assert_eq!(y, vec![9.0, 21.0]);
    }
}

// ============================================================
// Level 3 BLAS: Matrix-Matrix Operations
// ============================================================

mod level3_tests {
    use super::*;

    fn provider() -> PureRustBlas {
        PureRustBlas
    }

    // ------------------------------------------------------------
    // DGEMM: C = alpha * op(A) * op(B) + beta * C
    // ------------------------------------------------------------

    #[test]
    fn test_dgemm_basic_2x2() {
        let p = provider();
        // A = [[1, 2], [3, 4]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        // B = [[5, 6], [7, 8]]
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

        // C = A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //           = [[19, 22], [43, 50]]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_dgemm_identity() {
        let p = provider();
        // A = [[1, 2], [3, 4]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        // I = [[1, 0], [0, 1]]
        let i = vec![1.0, 0.0, 0.0, 1.0];
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
            &i,
            2,
            0.0,
            &mut c,
            2,
        );

        // A * I = A
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dgemm_a_trans() {
        let p = provider();
        // A = [[1, 2], [3, 4]], A^T = [[1, 3], [2, 4]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        // B = [[1, 0], [0, 1]]
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];

        p.dgemm(
            Layout::RowMajor,
            Transpose::Trans,
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

        // C = A^T * I = A^T
        assert_eq!(c, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_dgemm_b_trans() {
        let p = provider();
        let a = vec![1.0, 0.0, 0.0, 1.0]; // I
                                          // B = [[1, 2], [3, 4]], B^T = [[1, 3], [2, 4]]
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];

        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::Trans,
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

        // C = I * B^T = B^T
        assert_eq!(c, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_dgemm_with_beta() {
        let p = provider();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];

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
            2.0,
            &mut c,
            2,
        );

        // C = A*B + 2*C = [[19, 22], [43, 50]] + [[2, 2], [2, 2]]
        assert_eq!(c, vec![21.0, 24.0, 45.0, 52.0]);
    }

    #[test]
    fn test_dgemm_alpha_scaling() {
        let p = provider();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0]; // I
        let mut c = vec![0.0; 4];

        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            3.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );

        // C = 3 * A * I = 3 * A
        assert_eq!(c, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_dgemm_rectangular() {
        let p = provider();
        // A = 2x3, B = 3x4, C = 2x4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut c = vec![0.0; 8];

        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            4,
            3,
            1.0,
            &a,
            3,
            &b,
            4,
            0.0,
            &mut c,
            4,
        );

        // C[0,0] = 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
        assert_eq!(c[0], 38.0);
        // C[0,3] = 1*4 + 2*8 + 3*12 = 4 + 16 + 36 = 56
        assert_eq!(c[3], 56.0);
    }

    #[test]
    fn test_dgemm_large() {
        let p = provider();
        let n = 64;
        let a = vec![1.0; n * n];
        let b = vec![1.0; n * n];
        let mut c = vec![0.0; n * n];

        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            n as i32,
            n as i32,
            n as i32,
            1.0,
            &a,
            n as i32,
            &b,
            n as i32,
            0.0,
            &mut c,
            n as i32,
        );

        // Each element should be n (sum of n 1*1 products)
        assert_eq!(c[0], n as f64);
        assert_eq!(c[n * n - 1], n as f64);
    }
}

// ============================================================
// Single Precision (f32) Tests
// ============================================================

mod single_precision_tests {
    use super::*;

    fn provider() -> PureRustBlas {
        PureRustBlas
    }

    #[test]
    fn test_sdot() {
        let p = provider();
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 6.0, 7.0, 8.0];

        let result = p.sdot(4, &x, 1, &y, 1);

        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_snrm2() {
        let p = provider();
        let x = vec![3.0f32, 4.0];

        let result = p.snrm2(2, &x, 1);

        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_saxpy() {
        let p = provider();
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![4.0f32, 5.0, 6.0];

        p.saxpy(3, 2.0, &x, 1, &mut y, 1);

        assert_eq!(y, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_sgemm() {
        let p = provider();
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];

        p.sgemm(
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

// ============================================================
// Numerical Accuracy Tests
// ============================================================

mod accuracy_tests {
    use super::*;

    fn provider() -> PureRustBlas {
        PureRustBlas
    }

    #[test]
    fn test_dot_product_accuracy() {
        let p = provider();
        // Test with values that might have precision issues
        let n = 1000;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 - (i as f64) * 0.001).collect();

        let result = p.ddot(n as i32, &x, 1, &y, 1);

        // Verify against simple loop computation
        let expected: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gemm_associativity() {
        let p = provider();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let c = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        // Compute (A*B)*C
        let mut ab = vec![0.0; 9];
        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            3,
            3,
            3,
            1.0,
            &a,
            3,
            &b,
            3,
            0.0,
            &mut ab,
            3,
        );
        let mut ab_c = vec![0.0; 9];
        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            3,
            3,
            3,
            1.0,
            &ab,
            3,
            &c,
            3,
            0.0,
            &mut ab_c,
            3,
        );

        // Compute A*(B*C)
        let mut bc = vec![0.0; 9];
        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            3,
            3,
            3,
            1.0,
            &b,
            3,
            &c,
            3,
            0.0,
            &mut bc,
            3,
        );
        let mut a_bc = vec![0.0; 9];
        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            3,
            3,
            3,
            1.0,
            &a,
            3,
            &bc,
            3,
            0.0,
            &mut a_bc,
            3,
        );

        // Results should be equal
        for i in 0..9 {
            assert!(
                (ab_c[i] - a_bc[i]).abs() < 1e-10,
                "Associativity failed at index {}: {} vs {}",
                i,
                ab_c[i],
                a_bc[i]
            );
        }
    }

    #[test]
    fn test_norm_versus_dot() {
        let p = provider();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let norm_sq = p.dnrm2(5, &x, 1).powi(2);
        let dot = p.ddot(5, &x, 1, &x, 1);

        // ||x||^2 = x . x
        assert!((norm_sq - dot).abs() < 1e-10);
    }
}

// ============================================================
// Edge Cases
// ============================================================

mod edge_cases {
    use super::*;

    fn provider() -> PureRustBlas {
        PureRustBlas
    }

    #[test]
    fn test_empty_dot() {
        let p = provider();
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];

        let result = p.ddot(0, &x, 1, &y, 1);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_single_element_gemm() {
        let p = provider();
        let a = vec![3.0];
        let b = vec![4.0];
        let mut c = vec![0.0];

        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            1,
            1,
            1,
            1.0,
            &a,
            1,
            &b,
            1,
            0.0,
            &mut c,
            1,
        );

        assert_eq!(c[0], 12.0);
    }

    #[test]
    fn test_zero_alpha_gemm() {
        let p = provider();
        let a = vec![100.0; 4];
        let b = vec![100.0; 4];
        let mut c = vec![1.0; 4];

        p.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            0.0,
            &a,
            2,
            &b,
            2,
            1.0,
            &mut c,
            2,
        );

        // With alpha=0 and beta=1, C should be unchanged
        assert_eq!(c, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_zero_beta_gemm() {
        let p = provider();
        let a = vec![1.0, 0.0, 0.0, 1.0]; // I
        let b = vec![1.0, 0.0, 0.0, 1.0]; // I
        let mut c = vec![999.0; 4]; // Garbage values

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

        // With beta=0, old C values are ignored
        assert_eq!(c, vec![1.0, 0.0, 0.0, 1.0]);
    }
}

// ============================================================
// Provider Comparison (when multiple providers available)
// ============================================================

#[cfg(feature = "openblas")]
mod provider_comparison {
    use super::*;
    use bhc_numeric::blas::OpenBlasProvider;

    #[test]
    fn test_gemm_consistency_with_openblas() {
        let pure = PureRustBlas;
        let openblas = OpenBlasProvider;

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c_pure = vec![0.0; 9];
        let mut c_openblas = vec![0.0; 9];

        pure.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            3,
            3,
            3,
            1.0,
            &a,
            3,
            &b,
            3,
            0.0,
            &mut c_pure,
            3,
        );

        openblas.dgemm(
            Layout::RowMajor,
            Transpose::NoTrans,
            Transpose::NoTrans,
            3,
            3,
            3,
            1.0,
            &a,
            3,
            &b,
            3,
            0.0,
            &mut c_openblas,
            3,
        );

        for i in 0..9 {
            assert!(
                (c_pure[i] - c_openblas[i]).abs() < 1e-10,
                "Results differ at index {}: pure={}, openblas={}",
                i,
                c_pure[i],
                c_openblas[i]
            );
        }
    }
}
