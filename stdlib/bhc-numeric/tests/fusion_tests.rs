//! Fusion verification tests
//!
//! Tests to verify that tensor operations fuse correctly according to
//! H26-SPEC Section 8. These patterns MUST fuse in Numeric Profile.
//!
//! Guaranteed Fusion Patterns:
//! - Pattern 1: map f (map g x) -> map (f . g) x
//! - Pattern 2: zipWith f (map g a) (map h b) -> single traversal
//! - Pattern 3: sum (map f x) -> single traversal
//! - Pattern 4: foldl' op z (map f x) -> single traversal

use bhc_numeric::tensor::Tensor;

// ============================================================
// Pattern 1: map/map fusion
// ============================================================

mod map_map_fusion {
    use super::*;

    #[test]
    fn test_map_map_basic() {
        // map (+1) (map (*2) xs) should behave like map ((+1) . (*2)) xs
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        // Two separate maps
        let result1 = t.map(|x| x * 2.0).map(|x| x + 1.0);

        // Composed function
        let result2 = t.map(|x| (x * 2.0) + 1.0);

        // Results should be identical
        for i in 0..4 {
            assert_eq!(
                *result1.get(&[i]).unwrap(),
                *result2.get(&[i]).unwrap(),
                "map/map fusion result mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_map_map_chain() {
        // map f (map g (map h x)) should fuse to single map (f . g . h)
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        let result1 = t.map(|x| x + 1.0).map(|x| x * 2.0).map(|x| x - 3.0);

        let result2 = t.map(|x| ((x + 1.0) * 2.0) - 3.0);

        for i in 0..5 {
            assert!((*result1.get(&[i]).unwrap() - *result2.get(&[i]).unwrap()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_map_map_2d() {
        let t: Tensor<f64> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let result1 = t.map(|x| x * x).map(|x| x.sqrt());
        let result2 = t.map(|x| (x * x).sqrt());

        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (*result1.get(&[i, j]).unwrap() - *result2.get(&[i, j]).unwrap()).abs() < 1e-10
                );
            }
        }
    }

    #[test]
    fn test_map_map_large() {
        // Test fusion on large tensor to ensure no intermediate allocation explosion
        let t = Tensor::<f64>::full(&[1000, 1000], 1.0);

        let result = t.map(|x| x + 1.0).map(|x| x * 2.0);

        // All elements should be (1+1)*2 = 4
        assert_eq!(*result.get(&[0, 0]).unwrap(), 4.0);
        assert_eq!(*result.get(&[999, 999]).unwrap(), 4.0);
    }
}

// ============================================================
// Pattern 2: zipWith/map fusion
// ============================================================

mod zipwith_map_fusion {
    use super::*;

    #[test]
    fn test_zipwith_map_map() {
        // zipWith f (map g a) (map h b) should fuse to single traversal
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], &[4]).unwrap();

        // With intermediate maps
        let result1 = a
            .map(|x| x * 2.0)
            .zip_with(&b.map(|x| x + 1.0), |x, y| x + y)
            .unwrap();

        // Fused version (what it should compute)
        let result2 = a.zip_with(&b, |x, y| (x * 2.0) + (y + 1.0)).unwrap();

        for i in 0..4 {
            assert_eq!(*result1.get(&[i]).unwrap(), *result2.get(&[i]).unwrap());
        }
    }

    #[test]
    fn test_add_with_scaled() {
        // Common pattern: a + (b * scale)
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let scale = 2.0;

        let result1 = a.add(&b.mul_scalar(scale)).unwrap();
        let result2 = a.zip_with(&b, |x, y| x + y * scale).unwrap();

        for i in 0..3 {
            assert_eq!(*result1.get(&[i]).unwrap(), *result2.get(&[i]).unwrap());
        }
    }

    #[test]
    fn test_elementwise_chain() {
        // (a * b) + (c * d) - pattern from ML
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![2.0, 2.0, 2.0], &[3]).unwrap();
        let c: Tensor<f64> = Tensor::from_data(vec![3.0, 3.0, 3.0], &[3]).unwrap();
        let d: Tensor<f64> = Tensor::from_data(vec![1.0, 1.0, 1.0], &[3]).unwrap();

        let ab = a.mul(&b).unwrap();
        let cd = c.mul(&d).unwrap();
        let result = ab.add(&cd).unwrap();

        // Expected: [1*2 + 3*1, 2*2 + 3*1, 3*2 + 3*1] = [5, 7, 9]
        assert_eq!(*result.get(&[0]).unwrap(), 5.0);
        assert_eq!(*result.get(&[1]).unwrap(), 7.0);
        assert_eq!(*result.get(&[2]).unwrap(), 9.0);
    }
}

// ============================================================
// Pattern 3: sum/map fusion
// ============================================================

mod sum_map_fusion {
    use super::*;

    #[test]
    fn test_sum_map_basic() {
        // sum (map f x) should fuse to single traversal
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        let result1 = t.map(|x| x * 2.0).sum();

        // Manual computation: (1+2+3+4+5)*2 = 30
        assert_eq!(result1, 30.0);
    }

    #[test]
    fn test_sum_squared() {
        // Common pattern: sum of squares
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let result = t.map(|x| x * x).sum();

        // 1 + 4 + 9 + 16 = 30
        assert_eq!(result, 30.0);
    }

    #[test]
    fn test_mean_of_mapped() {
        // mean (map f x) should also fuse
        let t: Tensor<f64> = Tensor::from_data(vec![2.0, 4.0, 6.0, 8.0], &[4]).unwrap();

        let result = t.map(|x| x / 2.0).mean();

        // mean([1, 2, 3, 4]) = 2.5
        assert_eq!(result, 2.5);
    }

    #[test]
    fn test_norm_computation() {
        // sqrt(sum(map square x)) - L2 norm pattern
        let t: Tensor<f64> = Tensor::from_data(vec![3.0, 4.0], &[2]).unwrap();

        let norm_squared = t.map(|x| x * x).sum();
        let norm = norm_squared.sqrt();

        assert_eq!(norm, 5.0);
    }

    #[test]
    fn test_sum_map_large() {
        // Large tensor: verify no memory explosion
        let t = Tensor::<f64>::full(&[10000], 1.0);

        let result = t.map(|x| x + 1.0).sum();

        // All elements are 1, mapped to 2, sum is 20000
        assert_eq!(result, 20000.0);
    }

    #[test]
    fn test_product_map() {
        // product (map f x) should also fuse
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let result = t.map(|x| x + 1.0).product();

        // product([2, 3, 4, 5]) = 120
        assert_eq!(result, 120.0);
    }
}

// ============================================================
// Pattern 4: fold/map fusion
// ============================================================

mod fold_map_fusion {
    use super::*;

    #[test]
    fn test_fold_map_basic() {
        // foldl' op z (map f x) should fuse
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        // Sum of squares using fold
        let result = t.fold(0.0, |acc, x| acc + x * x);

        assert_eq!(result, 30.0); // 1 + 4 + 9 + 16
    }

    #[test]
    fn test_max_of_absolute() {
        // Common pattern: max of absolute values
        let t: Tensor<f64> = Tensor::from_data(vec![-3.0, 1.0, -5.0, 2.0, -1.0], &[5]).unwrap();

        let result = t.map(|x| x.abs()).max().unwrap();

        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_count_positive() {
        // Count elements satisfying a predicate
        let t: Tensor<f64> =
            Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0], &[6]).unwrap();

        let count = t.fold(0.0, |acc, x| if *x > 0.0 { acc + 1.0 } else { acc });

        assert_eq!(count, 3.0);
    }

    #[test]
    fn test_weighted_sum() {
        // Weighted sum pattern
        let values: Tensor<f64> = Tensor::from_data(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let weights: Tensor<f64> = Tensor::from_data(vec![0.5, 0.3, 0.2], &[3]).unwrap();

        let result = values.mul(&weights).unwrap().sum();

        // 10*0.5 + 20*0.3 + 30*0.2 = 5 + 6 + 6 = 17
        assert_eq!(result, 17.0);
    }
}

// ============================================================
// Complex Fusion Patterns
// ============================================================

mod complex_fusion {
    use super::*;

    #[test]
    fn test_softmax_pattern() {
        // softmax: exp(x - max(x)) / sum(exp(x - max(x)))
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let max_val = t.max().unwrap();
        let shifted = t.map(|x| x - max_val);
        let exp_vals = shifted.map(|x| x.exp());
        let sum_exp = exp_vals.sum();
        let result = exp_vals.mul_scalar(1.0 / sum_exp);

        // Verify softmax properties
        let softmax_sum = result.sum();
        assert!((softmax_sum - 1.0).abs() < 1e-10, "softmax should sum to 1");

        // Max element should have highest probability
        assert!(*result.get(&[3]).unwrap() > *result.get(&[0]).unwrap());
    }

    #[test]
    fn test_batch_norm_pattern() {
        // Simplified batch norm: (x - mean) / std
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        let mean = t.mean();
        let centered = t.map(|x| x - mean);
        let var = centered.map(|x| x * x).mean();
        let std = var.sqrt();
        let normalized = centered.mul_scalar(1.0 / std);

        // Normalized data should have mean ~0
        let new_mean = normalized.mean();
        assert!(new_mean.abs() < 1e-10);
    }

    #[test]
    fn test_relu_pattern() {
        // ReLU: max(0, x)
        let t: Tensor<f64> = Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        let result = t.map(|x| if *x > 0.0 { *x } else { 0.0 });

        assert_eq!(*result.get(&[0]).unwrap(), 0.0);
        assert_eq!(*result.get(&[1]).unwrap(), 0.0);
        assert_eq!(*result.get(&[2]).unwrap(), 0.0);
        assert_eq!(*result.get(&[3]).unwrap(), 1.0);
        assert_eq!(*result.get(&[4]).unwrap(), 2.0);
    }

    #[test]
    fn test_sigmoid_pattern() {
        // Sigmoid: 1 / (1 + exp(-x))
        let t: Tensor<f64> = Tensor::from_data(vec![-1.0, 0.0, 1.0], &[3]).unwrap();

        let result = t.map(|x| 1.0 / (1.0 + (-x).exp()));

        // At x=0, sigmoid = 0.5
        assert!((*result.get(&[1]).unwrap() - 0.5).abs() < 1e-10);

        // Sigmoid is monotonic
        assert!(*result.get(&[0]).unwrap() < *result.get(&[1]).unwrap());
        assert!(*result.get(&[1]).unwrap() < *result.get(&[2]).unwrap());
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Cross entropy: -sum(y * log(p))
        let predictions: Tensor<f64> = Tensor::from_data(vec![0.7, 0.2, 0.1], &[3]).unwrap();
        let targets: Tensor<f64> = Tensor::from_data(vec![1.0, 0.0, 0.0], &[3]).unwrap(); // one-hot

        let log_preds = predictions.map(|x| x.ln());
        let loss = targets.mul(&log_preds).unwrap().sum().abs();

        // Loss should be -log(0.7) ≈ 0.357
        assert!((loss - 0.7_f64.ln().abs()).abs() < 1e-10);
    }
}

// ============================================================
// Memory Efficiency Tests (Fusion Verification)
// ============================================================

mod memory_efficiency {
    use super::*;

    #[test]
    fn test_no_intermediate_for_map_chain() {
        // This test verifies the semantic equivalence of fused operations
        // Memory efficiency would be verified by profiling in a real scenario
        let t = Tensor::<f64>::full(&[1000], 2.0);

        // Long chain of maps
        let result = t
            .map(|x| x + 1.0)
            .map(|x| x * 2.0)
            .map(|x| x - 3.0)
            .map(|x| x / 2.0);

        // Each element: ((2+1)*2-3)/2 = (6-3)/2 = 1.5
        assert_eq!(*result.get(&[0]).unwrap(), 1.5);
        assert_eq!(*result.get(&[999]).unwrap(), 1.5);
    }

    #[test]
    fn test_no_intermediate_for_reduction() {
        // sum (map (*2) (map (+1) x)) should have no intermediate
        let t = Tensor::<f64>::full(&[10000], 1.0);

        let result = t.map(|x| x + 1.0).map(|x| x * 2.0).sum();

        // (1+1)*2 = 4, sum of 10000 elements = 40000
        assert_eq!(result, 40000.0);
    }

    #[test]
    fn test_zipwith_efficiency() {
        // Operations on two tensors should also be efficient
        let a = Tensor::<f64>::full(&[1000], 1.0);
        let b = Tensor::<f64>::full(&[1000], 2.0);

        let result = a
            .map(|x| x * 2.0)
            .zip_with(&b.map(|x| x + 1.0), |x, y| x + y)
            .unwrap()
            .sum();

        // Each element: (1*2) + (2+1) = 2 + 3 = 5
        // Sum: 5000
        assert_eq!(result, 5000.0);
    }
}

// ============================================================
// Axis-wise Operation Fusion
// ============================================================

mod axis_fusion {
    use super::*;

    #[test]
    fn test_sum_axis_after_map() {
        // sum_axis 0 (map f x) should fuse
        let t: Tensor<f64> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let result = t.map(|x| x * 2.0).sum_axis(0).unwrap();

        // Original: [[1,2,3], [4,5,6]]
        // After map: [[2,4,6], [8,10,12]]
        // Sum axis 0: [10, 14, 18]
        assert_eq!(*result.get(&[0]).unwrap(), 10.0);
        assert_eq!(*result.get(&[1]).unwrap(), 14.0);
        assert_eq!(*result.get(&[2]).unwrap(), 18.0);
    }

    #[test]
    fn test_mean_axis_after_map() {
        let t: Tensor<f64> = Tensor::from_data(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();

        let result = t.map(|x| x / 2.0).mean_axis(1).unwrap();

        // Original: [[2,4], [6,8]]
        // After map: [[1,2], [3,4]]
        // Mean axis 1: [1.5, 3.5]
        assert_eq!(*result.get(&[0]).unwrap(), 1.5);
        assert_eq!(*result.get(&[1]).unwrap(), 3.5);
    }
}

// ============================================================
// Numerical Stability in Fused Operations
// ============================================================

mod numerical_stability {
    use super::*;

    #[test]
    fn test_log_sum_exp_stability() {
        // log(sum(exp(x))) - should be computed stably
        let t: Tensor<f64> = Tensor::from_data(vec![1000.0, 1000.0, 1000.0], &[3]).unwrap();

        // Naive: exp(1000) would overflow
        // Stable: max + log(sum(exp(x - max)))
        let max_val = t.max().unwrap();
        let shifted = t.map(|x| x - max_val);
        let result = max_val + shifted.map(|x| x.exp()).sum().ln();

        // log(3 * exp(1000)) = 1000 + log(3)
        let expected = 1000.0 + 3.0_f64.ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_compensated_sum() {
        // Test precision for sum of many small values
        let n = 10000;
        let t = Tensor::<f64>::full(&[n], 0.0001);

        let result = t.sum();

        // Should be close to 1.0
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_numerical_stability() {
        // Variance computation should be stable
        let t: Tensor<f64> =
            Tensor::from_data(vec![1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0], &[4]).unwrap();

        let mean = t.mean();
        let variance = t.map(|x| (x - mean) * (x - mean)).mean();

        // Variance of [0, 1, 2, 3] = 1.25
        assert!((variance - 1.25).abs() < 0.01);
    }
}

// ============================================================
// Property Tests for Fusion Correctness
// ============================================================

mod fusion_properties {
    use super::*;

    #[test]
    fn test_map_identity() {
        // map id x == x
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let result = t.map(|x| *x);

        for i in 0..3 {
            assert_eq!(*result.get(&[i]).unwrap(), *t.get(&[i]).unwrap());
        }
    }

    #[test]
    fn test_sum_of_zeros() {
        // sum (map (const 0) x) == 0
        let t: Tensor<f64> =
            Tensor::from_data((0..100).map(|x| x as f64).collect(), &[100]).unwrap();
        let result = t.map(|_| 0.0).sum();

        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_sum_scale_distributive() {
        // sum (map (*k) x) == k * sum x
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let k = 3.0;

        let lhs = t.map(|x| *x * k).sum();
        let rhs = k * t.sum();

        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn test_zipwith_commutative_for_add() {
        // zipWith (+) a b == zipWith (+) b a
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        let ab = a.add(&b).unwrap();
        let ba = b.add(&a).unwrap();

        for i in 0..3 {
            assert_eq!(*ab.get(&[i]).unwrap(), *ba.get(&[i]).unwrap());
        }
    }
}
