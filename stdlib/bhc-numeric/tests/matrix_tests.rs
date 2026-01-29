//! Comprehensive Matrix tests
//!
//! Tests for Matrix<T> operations including:
//! - Construction (zeros, identity, from data)
//! - Element access and mutation
//! - Matrix arithmetic (add, sub, scale)
//! - Matrix multiplication (matmul, BLAS integration)
//! - Transpose and reshaping
//! - Algebraic properties

use bhc_numeric::matrix::Matrix;

// ============================================================
// Construction Tests
// ============================================================

mod construction_tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m: Matrix<f64> = Matrix::zeros(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn test_ones() {
        let m: Matrix<f64> = Matrix::fill(2, 3, 1.0);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m[(i, j)], 1.0);
            }
        }
    }

    #[test]
    fn test_identity() {
        let m: Matrix<f64> = Matrix::identity(4);
        assert_eq!(m.rows(), 4);
        assert_eq!(m.cols(), 4);
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(m[(i, j)], 1.0);
                } else {
                    assert_eq!(m[(i, j)], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix::from_data(2, 3, data);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    fn test_from_rows() {
        let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let m = Matrix::from_rows(rows).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 1)], 5.0);
    }

    #[test]
    fn test_diagonal() {
        let diag = vec![1.0, 2.0, 3.0];
        let m: Matrix<f64> = Matrix::diagonal(&diag);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 3);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 2.0);
        assert_eq!(m[(2, 2)], 3.0);
        assert_eq!(m[(0, 1)], 0.0);
    }

    #[test]
    fn test_fill() {
        let m: Matrix<f64> = Matrix::fill(2, 2, 7.5);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(m[(i, j)], 7.5);
            }
        }
    }

    #[test]
    fn test_single_element() {
        let m: Matrix<f64> = Matrix::from_data(1, 1, vec![42.0]);
        assert_eq!(m.rows(), 1);
        assert_eq!(m.cols(), 1);
        assert_eq!(m[(0, 0)], 42.0);
    }
}

// ============================================================
// Shape and Properties Tests
// ============================================================

mod shape_tests {
    use super::*;

    #[test]
    fn test_shape() {
        let m: Matrix<f64> = Matrix::zeros(3, 5);
        assert_eq!(m.shape(), (3, 5));
    }

    #[test]
    fn test_is_square() {
        let square: Matrix<f64> = Matrix::zeros(4, 4);
        let rect: Matrix<f64> = Matrix::zeros(3, 5);
        assert!(square.is_square());
        assert!(!rect.is_square());
    }

    #[test]
    fn test_is_empty() {
        let m: Matrix<f64> = Matrix::zeros(0, 0);
        assert!(m.rows() == 0 || m.cols() == 0);
    }

    #[test]
    fn test_numel() {
        let m: Matrix<f64> = Matrix::zeros(3, 4);
        assert_eq!(m.rows() * m.cols(), 12);
    }
}

// ============================================================
// Element Access Tests
// ============================================================

mod access_tests {
    use super::*;

    #[test]
    fn test_index() {
        let m = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    fn test_get() {
        let m = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.get(0, 0), Some(&1.0));
        assert_eq!(m.get(1, 1), Some(&4.0));
        assert_eq!(m.get(2, 0), None);
        assert_eq!(m.get(0, 2), None);
    }

    #[test]
    fn test_get_mut() {
        let mut m = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        if let Some(x) = m.get_mut(1, 0) {
            *x = 100.0;
        }
        assert_eq!(m[(1, 0)], 100.0);
    }

    #[test]
    fn test_row() {
        let m = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row0 = m.row(0).unwrap();
        assert_eq!(row0, &[1.0, 2.0, 3.0]);
        let row1 = m.row(1).unwrap();
        assert_eq!(row1, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_col() {
        let m = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let col0 = m.col(0).unwrap();
        assert_eq!(col0, vec![1.0, 4.0]);
        let col2 = m.col(2).unwrap();
        assert_eq!(col2, vec![3.0, 6.0]);
    }

    #[test]
    fn test_as_slice() {
        let m = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let slice = m.as_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }
}

// ============================================================
// Transpose Tests
// ============================================================

mod transpose_tests {
    use super::*;

    #[test]
    fn test_transpose_square() {
        let m = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let t = m.transpose();
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(0, 1)], 3.0);
        assert_eq!(t[(1, 0)], 2.0);
        assert_eq!(t[(1, 1)], 4.0);
    }

    #[test]
    fn test_transpose_rect() {
        let m = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(0, 1)], 4.0);
        assert_eq!(t[(1, 0)], 2.0);
        assert_eq!(t[(2, 1)], 6.0);
    }

    #[test]
    fn test_transpose_involutive() {
        let m = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tt = m.transpose().transpose();
        assert_eq!(tt.rows(), m.rows());
        assert_eq!(tt.cols(), m.cols());
        for i in 0..m.rows() {
            for j in 0..m.cols() {
                assert_eq!(tt[(i, j)], m[(i, j)]);
            }
        }
    }

    #[test]
    fn test_transpose_identity() {
        let i: Matrix<f64> = Matrix::identity(3);
        let it = i.transpose();
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(it[(r, c)], i[(r, c)]);
            }
        }
    }
}

// ============================================================
// Arithmetic Tests
// ============================================================

mod arithmetic_tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.add(&b).unwrap();
        assert_eq!(c[(0, 0)], 6.0);
        assert_eq!(c[(0, 1)], 8.0);
        assert_eq!(c[(1, 0)], 10.0);
        assert_eq!(c[(1, 1)], 12.0);
    }

    #[test]
    fn test_sub() {
        let a = Matrix::from_data(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let b = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c[(0, 0)], 9.0);
        assert_eq!(c[(0, 1)], 18.0);
        assert_eq!(c[(1, 0)], 27.0);
        assert_eq!(c[(1, 1)], 36.0);
    }

    #[test]
    fn test_scale() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let scaled = a.scale(3.0);
        assert_eq!(scaled[(0, 0)], 3.0);
        assert_eq!(scaled[(0, 1)], 6.0);
        assert_eq!(scaled[(1, 0)], 9.0);
        assert_eq!(scaled[(1, 1)], 12.0);
    }

    #[test]
    fn test_negate() {
        let a = Matrix::from_data(2, 2, vec![1.0, -2.0, 3.0, -4.0]);
        let neg = a.scale(-1.0);
        assert_eq!(neg[(0, 0)], -1.0);
        assert_eq!(neg[(0, 1)], 2.0);
        assert_eq!(neg[(1, 0)], -3.0);
        assert_eq!(neg[(1, 1)], 4.0);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(a.add(&b).is_none());
    }

    #[test]
    fn test_hadamard_mul() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_data(2, 2, vec![2.0, 2.0, 2.0, 2.0]);
        let c = a.hadamard(&b).unwrap();
        assert_eq!(c[(0, 0)], 2.0);
        assert_eq!(c[(0, 1)], 4.0);
        assert_eq!(c[(1, 0)], 6.0);
        assert_eq!(c[(1, 1)], 8.0);
    }
}

// ============================================================
// Matrix Multiplication Tests
// ============================================================

mod matmul_tests {
    use super::*;

    #[test]
    fn test_matmul_square() {
        // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c[(0, 0)], 19.0);
        assert_eq!(c[(0, 1)], 22.0);
        assert_eq!(c[(1, 0)], 43.0);
        assert_eq!(c[(1, 1)], 50.0);
    }

    #[test]
    fn test_matmul_rect() {
        // [1 2 3] * [1 2] = [1*1+2*3+3*5  1*2+2*4+3*6] = [22 28]
        //           [3 4]
        //           [5 6]
        let a = Matrix::from_data(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::from_data(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.rows(), 1);
        assert_eq!(c.cols(), 2);
        assert_eq!(c[(0, 0)], 22.0);
        assert_eq!(c[(0, 1)], 28.0);
    }

    #[test]
    fn test_matmul_identity() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let i: Matrix<f64> = Matrix::identity(2);
        let ai = a.matmul(&i).unwrap();
        let ia = i.matmul(&a).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(ai[(r, c)], a[(r, c)]);
                assert_eq!(ia[(r, c)], a[(r, c)]);
            }
        }
    }

    #[test]
    fn test_matmul_zeros() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let z: Matrix<f64> = Matrix::zeros(2, 2);
        let az = a.matmul(&z).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(az[(r, c)], 0.0);
            }
        }
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = Matrix::from_data(2, 3, vec![1.0; 6]);
        let b = Matrix::from_data(2, 3, vec![1.0; 6]);
        assert!(a.matmul(&b).is_none()); // 2x3 * 2x3 is invalid
    }

    #[test]
    fn test_matmul_associative() {
        let a: Matrix<f64> = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: Matrix<f64> = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c: Matrix<f64> = Matrix::from_data(2, 2, vec![9.0, 10.0, 11.0, 12.0]);

        let ab_c = a.matmul(&b).unwrap().matmul(&c).unwrap();
        let a_bc = a.matmul(&b.matmul(&c).unwrap()).unwrap();

        for r in 0..2 {
            for col in 0..2 {
                assert!((ab_c[(r, col)] - a_bc[(r, col)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matmul_transpose_property() {
        // (AB)^T = B^T * A^T
        let a: Matrix<f64> = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b: Matrix<f64> = Matrix::from_data(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let ab_t = a.matmul(&b).unwrap().transpose();
        let bt_at = b.transpose().matmul(&a.transpose()).unwrap();

        for r in 0..ab_t.rows() {
            for c in 0..ab_t.cols() {
                assert!((ab_t[(r, c)] - bt_at[(r, c)]).abs() < 1e-10);
            }
        }
    }
}

// ============================================================
// Reduction Tests
// ============================================================

mod reduction_tests {
    use super::*;

    #[test]
    fn test_sum() {
        let m = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.sum(), 10.0);
    }

    #[test]
    fn test_trace() {
        let m = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(m.trace(), 15.0); // 1 + 5 + 9
    }

    #[test]
    fn test_trace_identity() {
        let i: Matrix<f64> = Matrix::identity(4);
        assert_eq!(i.trace(), 4.0);
    }
}

// ============================================================
// Large Matrix Tests
// ============================================================

mod large_matrix_tests {
    use super::*;

    #[test]
    fn test_large_zeros() {
        let m: Matrix<f64> = Matrix::zeros(100, 100);
        assert_eq!(m.sum(), 0.0);
    }

    #[test]
    fn test_large_identity() {
        let n = 100;
        let i: Matrix<f64> = Matrix::identity(n);
        assert_eq!(i.trace(), n as f64);
    }

    #[test]
    fn test_large_matmul() {
        let n = 64;
        let a: Matrix<f64> = Matrix::fill(n, n, 1.0);
        let b: Matrix<f64> = Matrix::fill(n, n, 1.0);
        let c = a.matmul(&b).unwrap();
        // Each element of C should be n (sum of n 1*1 products)
        assert_eq!(c[(0, 0)], n as f64);
        assert_eq!(c[(n - 1, n - 1)], n as f64);
    }

    #[test]
    fn test_256x256_matmul() {
        let n = 256;
        let i: Matrix<f64> = Matrix::identity(n);
        let a: Matrix<f64> = Matrix::fill(n, n, 2.0);
        let ai = a.matmul(&i).unwrap();
        // A * I = A
        for r in 0..n {
            for c in 0..n {
                assert_eq!(ai[(r, c)], a[(r, c)]);
            }
        }
    }
}

// ============================================================
// Integer Matrix Tests
// ============================================================

mod integer_tests {
    use super::*;

    #[test]
    fn test_i32_matrix() {
        let m: Matrix<i32> = Matrix::from_data(2, 2, vec![1, 2, 3, 4]);
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m.sum(), 10);
    }

    #[test]
    fn test_i64_matrix() {
        let m: Matrix<i64> =
            Matrix::from_data(2, 2, vec![1000000000, 2000000000, 3000000000, 4000000000]);
        assert_eq!(m.sum(), 10000000000);
    }
}

// ============================================================
// Edge Cases
// ============================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_nan_handling() {
        let m = Matrix::from_data(2, 2, vec![1.0, f64::NAN, 3.0, 4.0]);
        assert!(m.sum().is_nan());
    }

    #[test]
    fn test_infinity_handling() {
        let m = Matrix::from_data(2, 2, vec![1.0, f64::INFINITY, 3.0, 4.0]);
        assert!(m.sum().is_infinite());
    }

    #[test]
    fn test_very_small_elements() {
        let tiny = f64::MIN_POSITIVE;
        let m = Matrix::from_data(2, 2, vec![tiny, tiny, tiny, tiny]);
        assert!(m.sum() > 0.0);
    }

    #[test]
    fn test_row_vector() {
        let m = Matrix::from_data(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(m.rows(), 1);
        assert_eq!(m.cols(), 5);
        assert_eq!(m.sum(), 15.0);
    }

    #[test]
    fn test_column_vector() {
        let m = Matrix::from_data(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(m.rows(), 5);
        assert_eq!(m.cols(), 1);
        assert_eq!(m.sum(), 15.0);
    }
}

// ============================================================
// Algebraic Property Tests
// ============================================================

mod property_tests {
    use super::*;

    #[test]
    fn test_add_commutative() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let ab = a.add(&b).unwrap();
        let ba = b.add(&a).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(ab[(r, c)], ba[(r, c)]);
            }
        }
    }

    #[test]
    fn test_add_associative() {
        let a: Matrix<f64> = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: Matrix<f64> = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c: Matrix<f64> = Matrix::from_data(2, 2, vec![9.0, 10.0, 11.0, 12.0]);

        let ab_c = a.add(&b).unwrap().add(&c).unwrap();
        let a_bc = a.add(&b.add(&c).unwrap()).unwrap();

        for r in 0..2 {
            for col in 0..2 {
                assert!((ab_c[(r, col)] - a_bc[(r, col)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_scale_distributive() {
        let a: Matrix<f64> = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: Matrix<f64> = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let k: f64 = 3.0;

        // k * (A + B) = k*A + k*B
        let lhs = a.add(&b).unwrap().scale(k);
        let rhs = a.scale(k).add(&b.scale(k)).unwrap();

        for r in 0..2 {
            for c in 0..2 {
                assert!((lhs[(r, c)] - rhs[(r, c)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matmul_distributive() {
        let a: Matrix<f64> = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b: Matrix<f64> = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c: Matrix<f64> = Matrix::from_data(2, 2, vec![9.0, 10.0, 11.0, 12.0]);

        // A * (B + C) = A*B + A*C
        let lhs = a.matmul(&b.add(&c).unwrap()).unwrap();
        let rhs = a.matmul(&b).unwrap().add(&a.matmul(&c).unwrap()).unwrap();

        for r in 0..2 {
            for col in 0..2 {
                assert!((lhs[(r, col)] - rhs[(r, col)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_trace_add() {
        // tr(A + B) = tr(A) + tr(B)
        let a = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = Matrix::from_data(3, 3, vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.trace(), a.trace() + b.trace());
    }

    #[test]
    fn test_trace_scale() {
        // tr(k*A) = k * tr(A)
        let a = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let k = 5.0;
        assert_eq!(a.scale(k).trace(), k * a.trace());
    }
}
