//! M4 Exit Criteria Tests
//!
//! This file contains integration tests that verify the M4 milestone
//! exit criteria as specified in the ROADMAP:
//!
//! 1. `matmul` can call external BLAS for large sizes
//! 2. Tensors stay pinned across FFI calls (verified by address stability)
//! 3. No GC movement of pinned allocations (stress test)

use bhc_ffi::{
    blas::{should_use_blas, BlasProvider, FallbackBlas, BLAS_THRESHOLD},
    default_provider, matmul,
    pinned::PinnedBuffer,
    smatmul,
    tensor::Matrix,
};

// ============================================================================
// Exit Criterion 1: matmul can call external BLAS for large sizes
// ============================================================================

#[test]
fn test_matmul_dispatches_to_blas_for_large_matrices() {
    // Verify that should_use_blas returns true for large matrices
    let large = BLAS_THRESHOLD + 1;
    assert!(
        should_use_blas(large, large, large),
        "should_use_blas should return true for {}x{} matrices",
        large,
        large
    );

    // Verify threshold behavior
    let small = BLAS_THRESHOLD - 1;
    assert!(
        !should_use_blas(small, small, small),
        "should_use_blas should return false for {}x{} matrices",
        small,
        small
    );
}

#[test]
fn test_matmul_large_matrix_correctness() {
    let provider = default_provider();

    // Create large matrices (above BLAS threshold)
    let n = BLAS_THRESHOLD + 10;

    // A = identity matrix
    let mut a_data = vec![0.0f64; n * n];
    for i in 0..n {
        a_data[i * n + i] = 1.0;
    }
    let a = Matrix::from_slice(n, n, &a_data).unwrap();

    // B = matrix with known values
    let mut b_data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            b_data[i * n + j] = (i * n + j) as f64;
        }
    }
    let b = Matrix::from_slice(n, n, &b_data).unwrap();

    // C = A * B = I * B = B
    let c = matmul(provider.as_ref(), &a, &b).unwrap();

    // Verify result: C should equal B
    for i in 0..n {
        for j in 0..n {
            let expected = (i * n + j) as f64;
            let actual = c.get(i, j);
            assert!(
                (actual - expected).abs() < 1e-10,
                "Mismatch at ({}, {}): expected {}, got {}",
                i,
                j,
                expected,
                actual
            );
        }
    }
}

#[test]
fn test_matmul_various_sizes() {
    let provider = default_provider();

    // Test various matrix sizes around the BLAS threshold
    let sizes = [
        (10, 10, 10),    // Small
        (32, 32, 32),    // Medium-small
        (64, 64, 64),    // At threshold
        (100, 100, 100), // Above threshold
        (128, 64, 96),   // Non-square above threshold
        (50, 200, 75),   // Rectangular above threshold
    ];

    for (m, k, n) in sizes {
        // Create test matrices
        let a_data: Vec<f64> = (0..m * k).map(|i| (i % 10) as f64).collect();
        let b_data: Vec<f64> = (0..k * n).map(|i| (i % 10) as f64).collect();

        let a = Matrix::from_slice(m, k, &a_data).unwrap();
        let b = Matrix::from_slice(k, n, &b_data).unwrap();

        let c = matmul(provider.as_ref(), &a, &b).unwrap();

        assert_eq!(c.rows(), m);
        assert_eq!(c.cols(), n);

        // Spot check a few values by computing manually
        let mut expected_00 = 0.0;
        for l in 0..k {
            expected_00 += a.get(0, l) * b.get(l, 0);
        }
        assert!(
            (c.get(0, 0) - expected_00).abs() < 1e-6,
            "Mismatch for size {}x{}x{} at (0,0)",
            m,
            k,
            n
        );
    }
}

// ============================================================================
// Exit Criterion 2: Tensors stay pinned across FFI calls
// ============================================================================

#[test]
fn test_pinned_buffer_address_stability() {
    let buffer = PinnedBuffer::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let initial_address = buffer.address();

    // Perform various operations that would trigger GC in a managed language
    for _ in 0..1000 {
        std::hint::black_box(buffer.as_slice());
    }

    // Address should not have changed
    assert_eq!(
        buffer.address(),
        initial_address,
        "PinnedBuffer address changed unexpectedly"
    );
}

#[test]
fn test_matrix_address_stability_across_operations() {
    let provider = default_provider();

    // Create matrices
    let a = Matrix::from_slice(100, 100, &vec![1.0f64; 10000]).unwrap();
    let b = Matrix::from_slice(100, 100, &vec![2.0f64; 10000]).unwrap();

    let addr_a = a.address();
    let addr_b = b.address();

    // Perform matmul multiple times
    for _ in 0..10 {
        let _c = matmul(provider.as_ref(), &a, &b).unwrap();
    }

    // Addresses should not have changed
    assert_eq!(
        a.address(),
        addr_a,
        "Matrix A address changed during matmul operations"
    );
    assert_eq!(
        b.address(),
        addr_b,
        "Matrix B address changed during matmul operations"
    );
}

#[test]
fn test_pinned_buffer_address_stable_during_blas_call() {
    let provider = FallbackBlas::new();

    let x = PinnedBuffer::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = PinnedBuffer::from_slice(&[5.0f64, 4.0, 3.0, 2.0, 1.0]).unwrap();

    let addr_x = x.address();
    let addr_y = y.address();

    // Perform BLAS operation
    let _result = provider.ddot(&x, &y).unwrap();

    // Addresses should not have changed
    assert_eq!(x.address(), addr_x, "Buffer x address changed during ddot");
    assert_eq!(y.address(), addr_y, "Buffer y address changed during ddot");
}

// ============================================================================
// Exit Criterion 3: No GC movement of pinned allocations (stress test)
// ============================================================================

#[test]
fn test_pinned_memory_stress() {
    // Allocate many pinned buffers and verify addresses stay stable
    let num_buffers = 100;
    let buffer_size = 1000;

    // Create buffers and record addresses
    let buffers: Vec<PinnedBuffer<f64>> = (0..num_buffers)
        .map(|_| PinnedBuffer::zeroed(buffer_size).unwrap())
        .collect();

    let addresses: Vec<usize> = buffers.iter().map(|b| b.address()).collect();

    // Allocate and deallocate many temporary buffers to simulate GC pressure
    for _ in 0..1000 {
        let _temp = PinnedBuffer::<f64>::zeroed(100).unwrap();
        // temp is dropped here
    }

    // Verify all original addresses are unchanged
    for (i, buffer) in buffers.iter().enumerate() {
        assert_eq!(
            buffer.address(),
            addresses[i],
            "Buffer {} address changed under memory pressure",
            i
        );
    }
}

#[test]
fn test_concurrent_matmul_address_stability() {
    use std::sync::Arc;
    use std::thread;

    let provider = Arc::new(FallbackBlas::new());

    // Shared matrices (read-only during matmul)
    let a = Arc::new(Matrix::from_slice(50, 50, &vec![1.0f64; 2500]).unwrap());
    let b = Arc::new(Matrix::from_slice(50, 50, &vec![2.0f64; 2500]).unwrap());

    let addr_a = a.address();
    let addr_b = b.address();

    // Spawn threads that perform matmul
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let provider = Arc::clone(&provider);
            let a = Arc::clone(&a);
            let b = Arc::clone(&b);
            thread::spawn(move || {
                for _ in 0..100 {
                    let _c = matmul(provider.as_ref(), &a, &b).unwrap();
                }
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify addresses unchanged
    assert_eq!(
        a.address(),
        addr_a,
        "Matrix A address changed during concurrent access"
    );
    assert_eq!(
        b.address(),
        addr_b,
        "Matrix B address changed during concurrent access"
    );
}

#[test]
fn test_mixed_allocation_stress() {
    let provider = default_provider();

    // Create some pinned matrices
    let matrices: Vec<_> = (0..10)
        .map(|i| {
            let size = 20 + i * 5;
            Matrix::<f64>::zeros(size, size).unwrap()
        })
        .collect();

    let addresses: Vec<usize> = matrices.iter().map(|m| m.address()).collect();

    // Perform many operations that might trigger memory management
    for _ in 0..50 {
        // Create temporary matrices
        let temp_a = Matrix::from_slice(30, 40, &vec![1.0f64; 1200]).unwrap();
        let temp_b = Matrix::from_slice(40, 30, &vec![2.0f64; 1200]).unwrap();

        // Perform matmul (creates another temporary)
        let _c = matmul(provider.as_ref(), &temp_a, &temp_b).unwrap();

        // All temps dropped here
    }

    // Verify original matrices haven't moved
    for (i, matrix) in matrices.iter().enumerate() {
        assert_eq!(
            matrix.address(),
            addresses[i],
            "Matrix {} address changed during mixed allocation stress",
            i
        );
    }
}

// ============================================================================
// Additional verification tests
// ============================================================================

#[test]
fn test_blas_provider_available() {
    let provider = default_provider();
    assert!(
        provider.is_available(),
        "Default BLAS provider should be available"
    );
}

#[test]
fn test_matmul_single_precision() {
    let provider = default_provider();

    // Test single-precision matmul
    let a = Matrix::from_slice(100, 100, &vec![1.0f32; 10000]).unwrap();
    let b = Matrix::from_slice(100, 100, &vec![2.0f32; 10000]).unwrap();

    let c = smatmul(provider.as_ref(), &a, &b).unwrap();

    // Each element of C should be 2 * 100 = 200
    assert!((c.get(0, 0) - 200.0f32).abs() < 1e-4);
    assert!((c.get(50, 50) - 200.0f32).abs() < 1e-4);
    assert!((c.get(99, 99) - 200.0f32).abs() < 1e-4);
}

#[test]
fn test_matmul_non_square() {
    let provider = default_provider();

    // Test non-square matrices: (3x4) * (4x2) = (3x2)
    let a = Matrix::from_slice(
        3,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    let b = Matrix::from_slice(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

    let c = matmul(provider.as_ref(), &a, &b).unwrap();

    assert_eq!(c.shape(), (3, 2));

    // Verify computed values
    // c[0,0] = 1*1 + 2*3 + 3*5 + 4*7 = 1 + 6 + 15 + 28 = 50
    assert!((c.get(0, 0) - 50.0).abs() < 1e-10);

    // c[0,1] = 1*2 + 2*4 + 3*6 + 4*8 = 2 + 8 + 18 + 32 = 60
    assert!((c.get(0, 1) - 60.0).abs() < 1e-10);
}

#[test]
fn test_pinned_buffer_region() {
    use bhc_rts_alloc::MemoryRegion;

    let buffer = PinnedBuffer::<f64>::zeroed(100).unwrap();
    assert_eq!(buffer.region(), MemoryRegion::PinnedHeap);
}
