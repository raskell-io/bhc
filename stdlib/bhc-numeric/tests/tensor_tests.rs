//! Comprehensive Tensor tests
//!
//! Tests for N-dimensional Tensor operations including:
//! - Shape construction and manipulation
//! - Tensor creation (zeros, ones, fill, from_data)
//! - Element-wise operations (map, zipWith)
//! - Reductions (sum, product, mean, min, max)
//! - Reshaping, transpose, and views
//! - Broadcasting semantics
//! - Index calculations and strides

use bhc_numeric::tensor::{Layout, Shape, Tensor};

// ============================================================
// Shape Tests
// ============================================================

mod shape_tests {
    use super::*;

    #[test]
    fn test_scalar_shape() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.num_elements(), 1);
    }

    #[test]
    fn test_vector_shape() {
        let s = Shape::vector(5);
        assert_eq!(s.rank(), 1);
        assert_eq!(s.dim(0), 5);
        assert_eq!(s.num_elements(), 5);
    }

    #[test]
    fn test_matrix_shape() {
        let s = Shape::matrix(3, 4);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.dim(0), 3);
        assert_eq!(s.dim(1), 4);
        assert_eq!(s.num_elements(), 12);
    }

    #[test]
    fn test_nd_shape() {
        let s = Shape::new(&[2, 3, 4, 5]);
        assert_eq!(s.rank(), 4);
        assert_eq!(s.num_elements(), 120);
    }

    #[test]
    fn test_empty_dimension() {
        let s = Shape::new(&[2, 0, 3]);
        assert_eq!(s.num_elements(), 0);
    }

    #[test]
    fn test_shape_equality() {
        let s1 = Shape::new(&[2, 3]);
        let s2 = Shape::matrix(2, 3);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_broadcast_compatible_same() {
        let s1 = Shape::new(&[3, 4]);
        let s2 = Shape::new(&[3, 4]);
        assert!(s1.is_broadcast_compatible(&s2));
    }

    #[test]
    fn test_broadcast_compatible_ones() {
        let s1 = Shape::new(&[3, 1]);
        let s2 = Shape::new(&[1, 4]);
        assert!(s1.is_broadcast_compatible(&s2));
    }

    #[test]
    fn test_broadcast_compatible_different_rank() {
        let s1 = Shape::new(&[3, 4]);
        let s2 = Shape::new(&[4]);
        assert!(s1.is_broadcast_compatible(&s2));
    }

    #[test]
    fn test_broadcast_incompatible() {
        let s1 = Shape::new(&[3, 4]);
        let s2 = Shape::new(&[3, 5]);
        assert!(!s1.is_broadcast_compatible(&s2));
    }

    #[test]
    fn test_broadcast_result_shape() {
        let s1 = Shape::new(&[3, 1]);
        let s2 = Shape::new(&[1, 4]);
        let result = s1.broadcast_shape(&s2).unwrap();
        assert_eq!(result.dims(), &[3, 4]);
    }

    #[test]
    fn test_broadcast_result_different_rank() {
        let s1 = Shape::new(&[2, 3, 4]);
        let s2 = Shape::new(&[4]);
        let result = s1.broadcast_shape(&s2).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);
    }
}

// ============================================================
// Tensor Construction Tests
// ============================================================

mod construction_tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::<f64>::zeros(&[2, 3]);
        assert_eq!(t.shape().dims(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(*t.get(&[i, j]).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn test_ones() {
        let t = Tensor::<f64>::ones(&[3, 2]);
        assert_eq!(t.shape().dims(), &[3, 2]);
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(*t.get(&[i, j]).unwrap(), 1.0);
            }
        }
    }

    #[test]
    fn test_full() {
        let t = Tensor::<f64>::full(&[2, 2], 7.5);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(*t.get(&[i, j]).unwrap(), 7.5);
            }
        }
    }

    #[test]
    fn test_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_data(data, &[2, 3]).unwrap();
        assert_eq!(*t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*t.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(*t.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(*t.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_scalar_tensor() {
        let t: Tensor<f64> = Tensor::scalar(42.0);
        assert_eq!(t.shape().rank(), 0);
        assert_eq!(t.len(), 1);
        assert_eq!(*t.get(&[]).unwrap(), 42.0);
    }

    #[test]
    fn test_arange() {
        let t: Tensor<f64> = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(t.shape().dims(), &[5]);
        assert_eq!(*t.get(&[0]).unwrap(), 0.0);
        assert_eq!(*t.get(&[4]).unwrap(), 4.0);
    }

    #[test]
    fn test_linspace() {
        let t: Tensor<f64> = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(t.shape().dims(), &[5]);
        assert_eq!(*t.get(&[0]).unwrap(), 0.0);
        assert_eq!(*t.get(&[4]).unwrap(), 1.0);
        assert!((*t.get(&[2]).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eye() {
        let t: Tensor<f64> = Tensor::eye(3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(*t.get(&[i, j]).unwrap(), 1.0);
                } else {
                    assert_eq!(*t.get(&[i, j]).unwrap(), 0.0);
                }
            }
        }
    }
}

// ============================================================
// Element Access Tests
// ============================================================

mod access_tests {
    use super::*;

    #[test]
    fn test_get_1d() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        assert_eq!(*t.get(&[0]).unwrap(), 1.0);
        assert_eq!(*t.get(&[4]).unwrap(), 5.0);
    }

    #[test]
    fn test_get_2d() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(*t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*t.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(*t.get(&[1, 1]).unwrap(), 5.0);
    }

    #[test]
    fn test_get_3d() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let t = Tensor::from_data(data, &[2, 3, 4]).unwrap();
        assert_eq!(*t.get(&[0, 0, 0]).unwrap(), 0.0);
        assert_eq!(*t.get(&[0, 0, 3]).unwrap(), 3.0);
        assert_eq!(*t.get(&[0, 1, 0]).unwrap(), 4.0);
        assert_eq!(*t.get(&[1, 0, 0]).unwrap(), 12.0);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.get(&[0, 0]), Some(&1.0));
        assert_eq!(t.get(&[1, 1]), Some(&4.0));
        assert_eq!(t.get(&[2, 0]), None);
    }

    #[test]
    fn test_get_flat() {
        let t = Tensor::from_data(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        assert_eq!(*t.get_flat(0).unwrap(), 10.0);
        assert_eq!(*t.get_flat(2).unwrap(), 30.0);
    }
}

// ============================================================
// Element-wise Operations Tests
// ============================================================

mod elementwise_tests {
    use super::*;

    #[test]
    fn test_map() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let t2 = t.map(|x| x * 2.0);
        assert_eq!(*t2.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*t2.get(&[1, 1]).unwrap(), 8.0);
    }

    #[test]
    fn test_map_preserves_shape() {
        let t = Tensor::<f64>::zeros(&[3, 4, 5]);
        let t2 = t.map(|x| x + 1.0);
        assert_eq!(t2.shape(), t.shape());
    }

    #[test]
    fn test_add() {
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(*c.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(*c.get(&[1, 1]).unwrap(), 44.0);
    }

    #[test]
    fn test_sub() {
        let a: Tensor<f64> = Tensor::from_data(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c = a.sub(&b).unwrap();
        assert_eq!(*c.get(&[0, 0]).unwrap(), 9.0);
        assert_eq!(*c.get(&[1, 1]).unwrap(), 36.0);
    }

    #[test]
    fn test_mul() {
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![2.0, 2.0, 2.0, 2.0], &[2, 2]).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(*c.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*c.get(&[1, 1]).unwrap(), 8.0);
    }

    #[test]
    fn test_div() {
        let a: Tensor<f64> = Tensor::from_data(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![2.0, 4.0, 5.0, 8.0], &[2, 2]).unwrap();
        let c = a.div(&b).unwrap();
        assert_eq!(*c.get(&[0, 0]).unwrap(), 5.0);
        assert_eq!(*c.get(&[1, 1]).unwrap(), 5.0);
    }

    #[test]
    fn test_mul_scalar() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let scaled = t.mul_scalar(3.0);
        assert_eq!(*scaled.get(&[0, 0]).unwrap(), 3.0);
        assert_eq!(*scaled.get(&[1, 1]).unwrap(), 12.0);
    }

    #[test]
    fn test_negate() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
        let neg = t.map(|x| -x);
        assert_eq!(*neg.get(&[0, 0]).unwrap(), -1.0);
        assert_eq!(*neg.get(&[0, 1]).unwrap(), 2.0);
    }

    #[test]
    fn test_zip_with() {
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let c = a.zip_with(&b, |x, y| x + y).unwrap();
        assert_eq!(*c.get(&[0]).unwrap(), 5.0);
        assert_eq!(*c.get(&[1]).unwrap(), 7.0);
        assert_eq!(*c.get(&[2]).unwrap(), 9.0);
    }

    #[test]
    fn test_shape_mismatch() {
        let a: Tensor<f64> = Tensor::from_data(vec![1.0; 4], &[2, 2]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![1.0; 6], &[2, 3]).unwrap();
        assert!(a.add(&b).is_err());
    }
}

// ============================================================
// Reduction Tests
// ============================================================

mod reduction_tests {
    use super::*;

    #[test]
    fn test_sum() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.sum(), 10.0);
    }

    #[test]
    fn test_product() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.product(), 24.0);
    }

    #[test]
    fn test_mean() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.mean(), 2.5);
    }

    #[test]
    fn test_min() {
        let t = Tensor::from_data(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]).unwrap();
        assert_eq!(t.min(), Some(1.0));
    }

    #[test]
    fn test_max() {
        let t = Tensor::from_data(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]).unwrap();
        assert_eq!(t.max(), Some(9.0));
    }

    #[test]
    fn test_sum_axis() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        // Sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
        let sum0 = t.sum_axis(0).unwrap();
        assert_eq!(sum0.shape().dims(), &[3]);
        assert_eq!(*sum0.get(&[0]).unwrap(), 5.0);
        assert_eq!(*sum0.get(&[1]).unwrap(), 7.0);
        assert_eq!(*sum0.get(&[2]).unwrap(), 9.0);

        // Sum along axis 1: [1+2+3, 4+5+6] = [6, 15]
        let sum1 = t.sum_axis(1).unwrap();
        assert_eq!(sum1.shape().dims(), &[2]);
        assert_eq!(*sum1.get(&[0]).unwrap(), 6.0);
        assert_eq!(*sum1.get(&[1]).unwrap(), 15.0);
    }

    #[test]
    fn test_mean_axis() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mean0 = t.mean_axis(0).unwrap();
        assert_eq!(*mean0.get(&[0]).unwrap(), 2.0); // (1+3)/2
        assert_eq!(*mean0.get(&[1]).unwrap(), 3.0); // (2+4)/2
    }

    #[test]
    fn test_norm() {
        let t: Tensor<f64> = Tensor::from_data(vec![3.0, 4.0], &[2]).unwrap();
        assert_eq!(t.norm(), 5.0); // sqrt(9+16)
    }

    #[test]
    fn test_scalar_sum() {
        let t: Tensor<f64> = Tensor::scalar(42.0);
        assert_eq!(t.sum(), 42.0);
    }
}

// ============================================================
// Reshape and View Tests
// ============================================================

mod reshape_tests {
    use super::*;

    #[test]
    fn test_reshape_same_elements() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape().dims(), &[3, 2]);
        assert_eq!(r.len(), 6);
    }

    #[test]
    fn test_reshape_to_1d() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let r = t.reshape(&[6]).unwrap();
        assert_eq!(r.shape().dims(), &[6]);
    }

    #[test]
    fn test_reshape_to_higher_dim() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
        let r = t.reshape(&[2, 3]).unwrap();
        assert_eq!(*r.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*r.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_reshape_invalid() {
        let t = Tensor::from_data(vec![1.0; 6], &[2, 3]).unwrap();
        assert!(t.reshape(&[2, 2]).is_err()); // 6 != 4
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let flat = t.reshape(&[6]).unwrap();
        assert_eq!(flat.shape().dims(), &[6]);
        assert_eq!(*flat.get(&[0]).unwrap(), 1.0);
        assert_eq!(*flat.get(&[5]).unwrap(), 6.0);
    }
}

// ============================================================
// Transpose and Permutation Tests
// ============================================================

mod transpose_tests {
    use super::*;

    #[test]
    fn test_transpose_2d() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let tt = t.transpose().unwrap();
        assert_eq!(tt.shape().dims(), &[3, 2]);
        assert_eq!(*tt.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tt.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(*tt.get(&[2, 0]).unwrap(), 3.0);
        assert_eq!(*tt.get(&[2, 1]).unwrap(), 6.0);
    }

    #[test]
    fn test_transpose_involutive() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let ttt = t.transpose().unwrap().transpose().unwrap();
        assert_eq!(ttt.shape(), t.shape());
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(*ttt.get(&[i, j]).unwrap(), *t.get(&[i, j]).unwrap());
            }
        }
    }

    #[test]
    fn test_permute() {
        let t = Tensor::<f64>::zeros(&[2, 3, 4]);
        let p = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape().dims(), &[4, 2, 3]);
    }
}

// ============================================================
// Slicing Tests
// ============================================================

mod slicing_tests {
    use super::*;

    #[test]
    fn test_slice_1d() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let s = t.slice(&[(1, 4)]).unwrap();
        assert_eq!(s.shape().dims(), &[3]);
        assert_eq!(*s.get(&[0]).unwrap(), 2.0);
        assert_eq!(*s.get(&[2]).unwrap(), 4.0);
    }

    #[test]
    fn test_slice_2d() {
        let t = Tensor::from_data((1..=12).map(|x| x as f64).collect(), &[3, 4]).unwrap();
        let s = t.slice(&[(0, 2), (1, 3)]).unwrap();
        assert_eq!(s.shape().dims(), &[2, 2]);
        assert_eq!(*s.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*s.get(&[1, 1]).unwrap(), 7.0);
    }
}

// ============================================================
// Broadcasting Tests
// ============================================================

mod broadcast_tests {
    use super::*;

    #[test]
    fn test_broadcast_scalar() {
        let scalar: Tensor<f64> = Tensor::scalar(5.0);
        let t: Tensor<f64> = Tensor::from_data(vec![1.0; 6], &[2, 3]).unwrap();
        let scalar_broadcasted = scalar.broadcast_to(&Shape::new(&[2, 3])).unwrap();
        let result = t.add(&scalar_broadcasted).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(*result.get(&[i, j]).unwrap(), 6.0);
            }
        }
    }

    #[test]
    fn test_broadcast_vector_to_matrix() {
        let v: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let broadcasted = v.broadcast_to(&Shape::new(&[2, 3])).unwrap();
        assert_eq!(broadcasted.shape().dims(), &[2, 3]);
        for i in 0..2 {
            assert_eq!(*broadcasted.get(&[i, 0]).unwrap(), 1.0);
            assert_eq!(*broadcasted.get(&[i, 1]).unwrap(), 2.0);
            assert_eq!(*broadcasted.get(&[i, 2]).unwrap(), 3.0);
        }
    }

    #[test]
    fn test_broadcast_add() {
        let a: Tensor<f64> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let b_broadcasted = b.broadcast_to(&Shape::new(&[2, 3])).unwrap();
        let c = a.add(&b_broadcasted).unwrap();
        assert_eq!(*c.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(*c.get(&[0, 1]).unwrap(), 22.0);
        assert_eq!(*c.get(&[1, 2]).unwrap(), 36.0);
    }
}

// ============================================================
// Linear Algebra Tests
// ============================================================

mod linalg_tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        assert_eq!(a.dot(&b).unwrap(), 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_matmul() {
        let a: Tensor<f64> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b: Tensor<f64> =
            Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);
        assert_eq!(*c.get(&[0, 0]).unwrap(), 58.0); // 1*7 + 2*9 + 3*11
        assert_eq!(*c.get(&[0, 1]).unwrap(), 64.0); // 1*8 + 2*10 + 3*12
    }

    #[test]
    fn test_outer_product() {
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![4.0, 5.0], &[2]).unwrap();
        let c = a.outer(&b).unwrap();
        assert_eq!(c.shape().dims(), &[3, 2]);
        assert_eq!(*c.get(&[0, 0]).unwrap(), 4.0);
        assert_eq!(*c.get(&[2, 1]).unwrap(), 15.0);
    }
}

// ============================================================
// Layout and Contiguity Tests
// ============================================================

mod layout_tests {
    use super::*;

    #[test]
    fn test_is_contiguous_new() {
        let t = Tensor::<f64>::zeros(&[2, 3]);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_transpose_not_contiguous() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0; 6], &[2, 3]).unwrap();
        let tt = t.transpose().unwrap();
        // Transpose creates a view with different strides
        // It may or may not be contiguous depending on implementation
        assert!(!tt.is_contiguous()); // Transpose produces strided layout
    }

    #[test]
    fn test_contiguous() {
        let t: Tensor<f64> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let tt = t.transpose().unwrap();
        let c = tt.contiguous();
        assert!(c.is_contiguous());
        // Values should be preserved
        assert_eq!(*c.get(&[0, 0]).unwrap(), *t.get(&[0, 0]).unwrap());
    }
}

// ============================================================
// Large Tensor Tests
// ============================================================

mod large_tensor_tests {
    use super::*;

    #[test]
    fn test_1m_elements() {
        let t: Tensor<f64> = Tensor::full(&[1000, 1000], 1.0);
        assert_eq!(t.len(), 1_000_000);
        assert_eq!(t.sum(), 1_000_000.0);
    }

    #[test]
    fn test_large_matmul() {
        let n = 64;
        let a: Tensor<f64> = Tensor::full(&[n, n], 1.0);
        let b: Tensor<f64> = Tensor::full(&[n, n], 1.0);
        let c = a.matmul(&b).unwrap();
        assert_eq!(*c.get(&[0, 0]).unwrap(), n as f64);
    }

    #[test]
    fn test_high_rank() {
        let t = Tensor::<f64>::zeros(&[2, 3, 4, 5, 6]);
        assert_eq!(t.shape().rank(), 5);
        assert_eq!(t.len(), 720);
    }
}

// ============================================================
// Edge Cases
// ============================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_nan_propagation() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, f64::NAN, 3.0], &[3]).unwrap();
        assert!(t.sum().is_nan());
    }

    #[test]
    fn test_infinity() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, f64::INFINITY, 3.0], &[3]).unwrap();
        assert!(t.sum().is_infinite());
    }

    #[test]
    fn test_empty_dim_tensor() {
        let t = Tensor::<f64>::zeros(&[2, 0, 3]);
        assert_eq!(t.len(), 0);
        assert_eq!(t.sum(), 0.0);
    }

    #[test]
    fn test_single_element_all_operations() {
        let t: Tensor<f64> = Tensor::from_data(vec![42.0], &[1, 1]).unwrap();
        assert_eq!(t.sum(), 42.0);
        assert_eq!(t.mean(), 42.0);
        assert_eq!(t.min(), Some(42.0));
        assert_eq!(t.max(), Some(42.0));
    }
}

// ============================================================
// Clone and Conversion Tests
// ============================================================

mod conversion_tests {
    use super::*;

    #[test]
    fn test_clone() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c = t.clone();
        assert_eq!(c.shape(), t.shape());
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(*c.get(&[i, j]).unwrap(), *t.get(&[i, j]).unwrap());
            }
        }
    }

    #[test]
    fn test_clone_produces_same_values() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c = t.clone();
        // Clones should have the same values
        assert_eq!(c.to_vec(), t.to_vec());
    }

    #[test]
    fn test_to_vec() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let v = t.to_vec();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
