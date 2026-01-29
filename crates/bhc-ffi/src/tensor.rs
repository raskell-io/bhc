//! Concrete tensor types with pinned memory for FFI.
//!
//! This module provides a `Tensor` type that uses pinned memory,
//! enabling zero-copy FFI calls to BLAS and other native libraries.
//!
//! ## M4 Exit Criteria
//!
//! - Tensors stay pinned across FFI calls (verified by address stability)
//! - `matmul` can call external BLAS for large sizes

use crate::blas::{should_use_blas, BlasProvider, BlasResult, Transpose};
use crate::pinned::PinnedBuffer;
use crate::{FfiError, FfiResult, FfiSafe};

/// A 2D matrix backed by pinned memory.
///
/// This type is optimized for FFI interop with BLAS libraries.
/// The data is stored in row-major order and guaranteed to remain
/// at a fixed memory address.
#[derive(Debug)]
pub struct Matrix<T: FfiSafe> {
    /// Pinned data buffer.
    data: PinnedBuffer<T>,
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
}

impl<T: FfiSafe> Matrix<T> {
    /// Create a new zero-initialized matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn zeros(rows: usize, cols: usize) -> FfiResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(FfiError::AllocationFailed(
                "matrix dimensions must be non-zero".to_string(),
            ));
        }

        let data = PinnedBuffer::zeroed(rows * cols)?;
        Ok(Self { data, rows, cols })
    }

    /// Create a matrix from a slice (row-major order).
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails or size doesn't match.
    pub fn from_slice(rows: usize, cols: usize, data: &[T]) -> FfiResult<Self> {
        if rows * cols != data.len() {
            return Err(FfiError::SizeMismatch {
                expected: rows * cols,
                actual: data.len(),
            });
        }

        let buffer = PinnedBuffer::from_slice(data)?;
        Ok(Self {
            data: buffer,
            rows,
            cols,
        })
    }

    /// Get the number of rows.
    #[inline]
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    #[inline]
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Get the shape as (rows, cols).
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get the leading dimension (row stride in elements).
    ///
    /// For row-major layout, this equals the number of columns.
    #[inline]
    #[must_use]
    pub const fn ld(&self) -> usize {
        self.cols
    }

    /// Get the total number of elements.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Check if the matrix is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Get a reference to the underlying pinned buffer.
    #[inline]
    #[must_use]
    pub fn buffer(&self) -> &PinnedBuffer<T> {
        &self.data
    }

    /// Get a mutable reference to the underlying pinned buffer.
    #[inline]
    #[must_use]
    pub fn buffer_mut(&mut self) -> &mut PinnedBuffer<T> {
        &mut self.data
    }

    /// Get the data as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Get the data as a mutable slice.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Get the memory address (for pinning verification).
    #[must_use]
    pub fn address(&self) -> usize {
        self.data.address()
    }

    /// Get an element at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows && col < self.cols);
        self.data.as_slice()[row * self.cols + col]
    }

    /// Set an element at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.cols);
        self.data.as_mut_slice()[row * self.cols + col] = value;
    }
}

/// Matrix multiplication: C = A * B
///
/// Automatically dispatches to BLAS for large matrices.
///
/// # Arguments
///
/// * `provider` - BLAS provider to use for large matrices
/// * `a` - Left matrix (m x k)
/// * `b` - Right matrix (k x n)
///
/// # Returns
///
/// Result matrix C (m x n)
///
/// # Errors
///
/// Returns an error if dimensions don't match or allocation fails.
pub fn matmul(
    provider: &dyn BlasProvider,
    a: &Matrix<f64>,
    b: &Matrix<f64>,
) -> FfiResult<Matrix<f64>> {
    let (m, k1) = a.shape();
    let (k2, n) = b.shape();

    if k1 != k2 {
        return Err(FfiError::SizeMismatch {
            expected: k1,
            actual: k2,
        });
    }

    let k = k1;
    let mut c = Matrix::zeros(m, n)?;

    if should_use_blas(m, n, k) && provider.is_available() {
        // Use BLAS for large matrices
        matmul_blas(provider, a, b, &mut c)?;
    } else {
        // Fallback to naive implementation for small matrices
        matmul_naive(a, b, &mut c);
    }

    Ok(c)
}

/// Single-precision matrix multiplication: C = A * B
pub fn smatmul(
    provider: &dyn BlasProvider,
    a: &Matrix<f32>,
    b: &Matrix<f32>,
) -> FfiResult<Matrix<f32>> {
    let (m, k1) = a.shape();
    let (k2, n) = b.shape();

    if k1 != k2 {
        return Err(FfiError::SizeMismatch {
            expected: k1,
            actual: k2,
        });
    }

    let k = k1;
    let mut c = Matrix::zeros(m, n)?;

    if should_use_blas(m, n, k) && provider.is_available() {
        smatmul_blas(provider, a, b, &mut c)?;
    } else {
        smatmul_naive(a, b, &mut c);
    }

    Ok(c)
}

/// BLAS-accelerated double-precision matrix multiplication.
fn matmul_blas(
    provider: &dyn BlasProvider,
    a: &Matrix<f64>,
    b: &Matrix<f64>,
    c: &mut Matrix<f64>,
) -> FfiResult<()> {
    let (m, k) = a.shape();
    let (_, n) = b.shape();
    let lda = a.ld();
    let ldb = b.ld();
    let ldc = c.ld();

    provider
        .dgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0, // alpha
            a.buffer(),
            lda,
            b.buffer(),
            ldb,
            0.0, // beta
            c.buffer_mut(),
            ldc,
        )
        .map_err(FfiError::BlasError)
}

/// BLAS-accelerated single-precision matrix multiplication.
fn smatmul_blas(
    provider: &dyn BlasProvider,
    a: &Matrix<f32>,
    b: &Matrix<f32>,
    c: &mut Matrix<f32>,
) -> FfiResult<()> {
    let (m, k) = a.shape();
    let (_, n) = b.shape();
    let lda = a.ld();
    let ldb = b.ld();
    let ldc = c.ld();

    provider
        .sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a.buffer(),
            lda,
            b.buffer(),
            ldb,
            0.0,
            c.buffer_mut(),
            ldc,
        )
        .map_err(FfiError::BlasError)
}

/// Naive matrix multiplication (for small matrices or fallback).
fn matmul_naive(a: &Matrix<f64>, b: &Matrix<f64>, c: &mut Matrix<f64>) {
    let (m, k) = a.shape();
    let (_, n) = b.shape();

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a.get(i, l) * b.get(l, j);
            }
            c.set(i, j, sum);
        }
    }
}

/// Naive single-precision matrix multiplication.
fn smatmul_naive(a: &Matrix<f32>, b: &Matrix<f32>, c: &mut Matrix<f32>) {
    let (m, k) = a.shape();
    let (_, n) = b.shape();

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a.get(i, l) * b.get(l, j);
            }
            c.set(i, j, sum);
        }
    }
}

/// Dot product of two vectors.
pub fn dot(
    provider: &dyn BlasProvider,
    x: &PinnedBuffer<f64>,
    y: &PinnedBuffer<f64>,
) -> BlasResult<f64> {
    provider.ddot(x, y)
}

/// Single-precision dot product.
pub fn sdot(
    provider: &dyn BlasProvider,
    x: &PinnedBuffer<f32>,
    y: &PinnedBuffer<f32>,
) -> BlasResult<f32> {
    provider.sdot(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blas::FallbackBlas;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::<f64>::zeros(3, 4).unwrap();
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.len(), 12);
        assert_eq!(m.ld(), 4);
    }

    #[test]
    fn test_matrix_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix::from_slice(2, 3, &data).unwrap();

        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    fn test_matrix_set_get() {
        let mut m = Matrix::<f64>::zeros(2, 2).unwrap();
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(1, 0), 3.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    #[test]
    fn test_matrix_address_stability() {
        let m = Matrix::<f64>::zeros(100, 100).unwrap();
        let addr1 = m.address();

        // Simulate some work
        std::hint::black_box(&m);

        let addr2 = m.address();
        assert_eq!(addr1, addr2, "Matrix address should not change");
    }

    #[test]
    fn test_matmul_small() {
        // Small matrix (below BLAS threshold)
        let provider = FallbackBlas::new();

        let a = Matrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let b = Matrix::from_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let c = matmul(&provider, &a, &b).unwrap();

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        assert_eq!(c.get(0, 0), 58.0);
        assert_eq!(c.get(0, 1), 64.0);
        assert_eq!(c.get(1, 0), 139.0);
        assert_eq!(c.get(1, 1), 154.0);
    }

    #[test]
    fn test_matmul_large() {
        // Large matrix (above BLAS threshold)
        let provider = FallbackBlas::new();

        let n = 100;
        let mut a_data = vec![0.0f64; n * n];
        let mut b_data = vec![0.0f64; n * n];

        // Initialize as identity matrices
        for i in 0..n {
            a_data[i * n + i] = 1.0;
            b_data[i * n + i] = 1.0;
        }

        let a = Matrix::from_slice(n, n, &a_data).unwrap();
        let b = Matrix::from_slice(n, n, &b_data).unwrap();

        let c = matmul(&provider, &a, &b).unwrap();

        // Identity * Identity = Identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((c.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let provider = FallbackBlas::new();

        let a = Matrix::<f64>::zeros(2, 3).unwrap();
        let b = Matrix::<f64>::zeros(4, 2).unwrap(); // k=4 != 3

        let result = matmul(&provider, &a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_smatmul() {
        let provider = FallbackBlas::new();

        let a = Matrix::from_slice(2, 2, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();

        let b = Matrix::from_slice(2, 2, &[5.0f32, 6.0, 7.0, 8.0]).unwrap();

        let c = smatmul(&provider, &a, &b).unwrap();

        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_pinned_across_ffi_boundary() {
        // Verify address stability during operations
        let provider = FallbackBlas::new();

        let a = Matrix::from_slice(64, 64, &vec![1.0f64; 64 * 64]).unwrap();
        let b = Matrix::from_slice(64, 64, &vec![1.0f64; 64 * 64]).unwrap();

        let addr_a = a.address();
        let addr_b = b.address();

        // Perform matmul (which uses the buffers)
        let _c = matmul(&provider, &a, &b).unwrap();

        // Addresses should not have changed
        assert_eq!(
            a.address(),
            addr_a,
            "Matrix A address changed during matmul"
        );
        assert_eq!(
            b.address(),
            addr_b,
            "Matrix B address changed during matmul"
        );
    }
}
