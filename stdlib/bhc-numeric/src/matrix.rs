//! 2-D matrices
//!
//! Matrix operations with BLAS integration.
//!
//! # Overview
//!
//! `Matrix<T>` provides a 2-D row-major matrix with efficient linear
//! algebra operations. When available, operations delegate to BLAS.
//!
//! # FFI
//!
//! This module exports C-ABI functions for BHC-compiled Haskell to call:
//! - `bhc_matrix_from_f64`, `bhc_matrix_free_f64`
//! - `bhc_matrix_matmul_f64`, `bhc_matrix_transpose_f64`
//! - `bhc_matrix_add_f64`, `bhc_matrix_scale_f64`

use crate::blas::{BlasProvider, PureRustBlas};
use crate::vector::Vector;
use std::fmt;
use std::ops::{Index, IndexMut};

// ============================================================
// Core Matrix Type
// ============================================================

/// A 2-D row-major matrix
#[derive(Clone)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T> Matrix<T> {
    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Check if the matrix is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if the matrix is square
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Get a raw pointer to the data
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get a mutable raw pointer to the data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Get a slice view of the data (row-major)
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable slice view of the data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Convert linear index to (row, col)
    fn to_coords(&self, index: usize) -> (usize, usize) {
        (index / self.cols, index % self.cols)
    }

    /// Convert (row, col) to linear index
    fn to_index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }
}

impl<T: Clone> Matrix<T> {
    /// Create a new matrix from data and shape
    ///
    /// Data must be provided in row-major order.
    pub fn new(data: Vec<T>, rows: usize, cols: usize) -> Option<Self> {
        if data.len() != rows * cols {
            return None;
        }
        Some(Self { data, rows, cols })
    }

    /// Create from a flat slice with given shape
    pub fn from_slice(data: &[T], rows: usize, cols: usize) -> Option<Self> {
        if data.len() != rows * cols {
            return None;
        }
        Some(Self {
            data: data.to_vec(),
            rows,
            cols,
        })
    }

    /// Create a matrix of zeros
    pub fn zeros(rows: usize, cols: usize) -> Self
    where
        T: Default,
    {
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create a matrix filled with a value
    pub fn fill(rows: usize, cols: usize, value: T) -> Self {
        Self {
            data: vec![value; rows * cols],
            rows,
            cols,
        }
    }

    /// Get an element by (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[self.to_index(row, col)])
        } else {
            None
        }
    }

    /// Get a mutable element by (row, col)
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            let idx = self.to_index(row, col);
            Some(&mut self.data[idx])
        } else {
            None
        }
    }

    /// Set an element
    pub fn set(&mut self, row: usize, col: usize, value: T) -> bool {
        if row < self.rows && col < self.cols {
            let idx = self.to_index(row, col);
            self.data[idx] = value;
            true
        } else {
            false
        }
    }

    /// Get a row as a slice
    pub fn row(&self, row: usize) -> Option<&[T]> {
        if row < self.rows {
            let start = row * self.cols;
            Some(&self.data[start..start + self.cols])
        } else {
            None
        }
    }

    /// Get a column as a vector (requires copying)
    pub fn col(&self, col: usize) -> Option<Vec<T>> {
        if col < self.cols {
            Some((0..self.rows).map(|r| self.data[r * self.cols + col].clone()).collect())
        } else {
            None
        }
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        let mut result = Vec::with_capacity(self.len());
        for c in 0..self.cols {
            for r in 0..self.rows {
                result.push(self.data[r * self.cols + c].clone());
            }
        }
        Self {
            data: result,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Reshape the matrix (must have same total elements)
    pub fn reshape(&self, new_rows: usize, new_cols: usize) -> Option<Self> {
        if self.len() != new_rows * new_cols {
            return None;
        }
        Some(Self {
            data: self.data.clone(),
            rows: new_rows,
            cols: new_cols,
        })
    }

    /// Flatten to a 1-D vector
    pub fn flatten(&self) -> Vector<T> {
        Vector::from_vec(self.data.clone())
    }

    /// Create from nested vectors (rows)
    pub fn from_rows(rows: Vec<Vec<T>>) -> Option<Self> {
        if rows.is_empty() {
            return Some(Self {
                data: vec![],
                rows: 0,
                cols: 0,
            });
        }
        let num_rows = rows.len();
        let num_cols = rows[0].len();
        if rows.iter().any(|r| r.len() != num_cols) {
            return None;
        }
        let data: Vec<T> = rows.into_iter().flatten().collect();
        Some(Self {
            data,
            rows: num_rows,
            cols: num_cols,
        })
    }

    /// Convert to nested vectors
    pub fn to_rows(&self) -> Vec<Vec<T>> {
        (0..self.rows)
            .map(|r| self.row(r).unwrap().to_vec())
            .collect()
    }
}

// ============================================================
// Identity Matrix
// ============================================================

impl<T: Clone + Default> Matrix<T> {
    /// Create an identity matrix
    pub fn identity(n: usize) -> Self
    where
        T: From<u8>,
    {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = T::from(1u8);
        }
        m
    }

    /// Create a diagonal matrix from a vector
    pub fn diagonal(diag: &[T]) -> Self {
        let n = diag.len();
        let mut m = Self::zeros(n, n);
        for (i, val) in diag.iter().enumerate() {
            m.data[i * n + i] = val.clone();
        }
        m
    }

    /// Extract the diagonal as a vector
    pub fn get_diagonal(&self) -> Vector<T> {
        let n = self.rows.min(self.cols);
        let data: Vec<T> = (0..n).map(|i| self.data[i * self.cols + i].clone()).collect();
        Vector::from_vec(data)
    }
}

// ============================================================
// Numeric Operations
// ============================================================

impl<T> Matrix<T>
where
    T: Copy + Default + std::ops::Add<Output = T>,
{
    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.shape() != other.shape() {
            return None;
        }
        Some(Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Sum of all elements
    pub fn sum(&self) -> T {
        self.data.iter().copied().fold(T::default(), |acc, x| acc + x)
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>,
{
    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Option<Self> {
        if self.shape() != other.shape() {
            return None;
        }
        Some(Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
            rows: self.rows,
            cols: self.cols,
        })
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + std::ops::Mul<Output = T>,
{
    /// Element-wise multiplication (Hadamard product)
    pub fn hadamard(&self, other: &Self) -> Option<Self> {
        if self.shape() != other.shape() {
            return None;
        }
        Some(Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .collect(),
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: T) -> Self {
        Self {
            data: self.data.iter().map(|&x| x * scalar).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Matrix multiplication (naive O(n³) implementation)
    ///
    /// For better performance, use BLAS-backed operations.
    pub fn matmul(&self, other: &Self) -> Option<Self> {
        if self.cols != other.rows {
            return None;
        }
        let mut result = vec![T::default(); self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result[i * other.cols + j] = sum;
            }
        }
        Some(Self {
            data: result,
            rows: self.rows,
            cols: other.cols,
        })
    }

    /// Matrix-vector multiplication
    pub fn matvec(&self, vec: &Vector<T>) -> Option<Vector<T>> {
        if self.cols != vec.len() {
            return None;
        }
        let mut result = vec![T::default(); self.rows];
        for i in 0..self.rows {
            let mut sum = T::default();
            for j in 0..self.cols {
                sum = sum + self.data[i * self.cols + j] * vec[j];
            }
            result[i] = sum;
        }
        Some(Vector::from_vec(result))
    }

    /// Trace (sum of diagonal elements)
    pub fn trace(&self) -> T {
        let n = self.rows.min(self.cols);
        let mut sum = T::default();
        for i in 0..n {
            sum = sum + self.data[i * self.cols + i];
        }
        sum
    }
}

impl<T> Matrix<T>
where
    T: Copy,
{
    /// Map a function over all elements
    pub fn map<F, U>(&self, f: F) -> Matrix<U>
    where
        F: Fn(T) -> U,
    {
        Matrix {
            data: self.data.iter().map(|&x| f(x)).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// ============================================================
// Float-specific Operations
// ============================================================

impl Matrix<f64> {
    /// Frobenius norm
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.sum() / (self.len() as f64)
        }
    }

    /// Row means
    pub fn row_means(&self) -> Vector<f64> {
        let data: Vec<f64> = (0..self.rows)
            .map(|r| {
                let row = self.row(r).unwrap();
                row.iter().sum::<f64>() / (self.cols as f64)
            })
            .collect();
        Vector::from_vec(data)
    }

    /// Column means
    pub fn col_means(&self) -> Vector<f64> {
        let data: Vec<f64> = (0..self.cols)
            .map(|c| {
                let sum: f64 = (0..self.rows).map(|r| self.data[r * self.cols + c]).sum();
                sum / (self.rows as f64)
            })
            .collect();
        Vector::from_vec(data)
    }

    /// Minimum element
    pub fn min(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::min)
    }

    /// Maximum element
    pub fn max(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::max)
    }

    /// Apply exp element-wise
    pub fn exp(&self) -> Self {
        self.map(f64::exp)
    }

    /// Apply ln element-wise
    pub fn ln(&self) -> Self {
        self.map(f64::ln)
    }

    /// Apply abs element-wise
    pub fn abs(&self) -> Self {
        self.map(f64::abs)
    }

    /// Clamp all values
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        self.map(|x| x.clamp(min, max))
    }

    /// Matrix multiplication using BLAS
    pub fn matmul_blas(&self, other: &Self) -> Option<Self> {
        if self.cols != other.rows {
            return None;
        }
        let blas = PureRustBlas;
        let mut result = vec![0.0; self.rows * other.cols];
        blas.gemm(
            self.rows,
            other.cols,
            self.cols,
            1.0,
            &self.data,
            self.cols,
            &other.data,
            other.cols,
            0.0,
            &mut result,
            other.cols,
        );
        Some(Self {
            data: result,
            rows: self.rows,
            cols: other.cols,
        })
    }

    /// Solve linear system Ax = b (simple Gaussian elimination)
    ///
    /// For production use, prefer LAPACK-based solvers.
    pub fn solve(&self, b: &Vector<f64>) -> Option<Vector<f64>> {
        if !self.is_square() || self.rows != b.len() {
            return None;
        }

        let n = self.rows;
        let mut aug = vec![0.0; n * (n + 1)];

        // Build augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = self.data[i * n + j];
            }
            aug[i * (n + 1) + n] = b[i];
        }

        // Gaussian elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col * (n + 1) + col].abs();
            for row in (col + 1)..n {
                let val = aug[row * (n + 1) + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-10 {
                return None; // Singular matrix
            }

            // Swap rows
            if max_row != col {
                for j in 0..=n {
                    aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
                }
            }

            // Eliminate
            let pivot = aug[col * (n + 1) + col];
            for row in (col + 1)..n {
                let factor = aug[row * (n + 1) + col] / pivot;
                for j in col..=n {
                    aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i * (n + 1) + n];
            for j in (i + 1)..n {
                sum -= aug[i * (n + 1) + j] * x[j];
            }
            x[i] = sum / aug[i * (n + 1) + i];
        }

        Some(Vector::from_vec(x))
    }
}

impl Matrix<f32> {
    /// Frobenius norm
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        if self.is_empty() {
            0.0
        } else {
            self.sum() / (self.len() as f32)
        }
    }
}

// ============================================================
// Trait Implementations
// ============================================================

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}

impl<T: fmt::Debug> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix({}x{}, {:?})", self.rows, self.cols, self.data)
    }
}

impl<T: fmt::Display + Clone> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for r in 0..self.rows {
            write!(f, "  [")?;
            for c in 0..self.cols {
                if c > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.data[r * self.cols + c])?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

impl<T: PartialEq> PartialEq for Matrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl<T: Eq> Eq for Matrix<T> {}

// ============================================================
// FFI Exports - f64
// ============================================================

/// Create a matrix from raw data
///
/// # Safety
/// - `data` must point to `rows * cols` valid f64 values
/// - Data is expected in row-major order
#[no_mangle]
pub unsafe extern "C" fn bhc_matrix_from_f64(
    data: *const f64,
    rows: usize,
    cols: usize,
) -> *mut Matrix<f64> {
    if data.is_null() || rows == 0 || cols == 0 {
        return std::ptr::null_mut();
    }
    let slice = std::slice::from_raw_parts(data, rows * cols);
    match Matrix::from_slice(slice, rows, cols) {
        Some(m) => Box::into_raw(Box::new(m)),
        None => std::ptr::null_mut(),
    }
}

/// Free a matrix
#[no_mangle]
pub unsafe extern "C" fn bhc_matrix_free_f64(mat: *mut Matrix<f64>) {
    if !mat.is_null() {
        drop(Box::from_raw(mat));
    }
}

/// Get matrix rows
#[no_mangle]
pub extern "C" fn bhc_matrix_rows_f64(mat: *const Matrix<f64>) -> usize {
    if mat.is_null() {
        return 0;
    }
    unsafe { (*mat).rows() }
}

/// Get matrix cols
#[no_mangle]
pub extern "C" fn bhc_matrix_cols_f64(mat: *const Matrix<f64>) -> usize {
    if mat.is_null() {
        return 0;
    }
    unsafe { (*mat).cols() }
}

/// Get matrix data pointer
#[no_mangle]
pub extern "C" fn bhc_matrix_data_f64(mat: *const Matrix<f64>) -> *const f64 {
    if mat.is_null() {
        return std::ptr::null();
    }
    unsafe { (*mat).as_ptr() }
}

/// Get element at (row, col)
#[no_mangle]
pub unsafe extern "C" fn bhc_matrix_get_f64(
    mat: *const Matrix<f64>,
    row: usize,
    col: usize,
) -> f64 {
    if mat.is_null() {
        return 0.0;
    }
    (*mat).get(row, col).copied().unwrap_or(0.0)
}

/// Matrix multiplication
#[no_mangle]
pub unsafe extern "C" fn bhc_matrix_matmul_f64(
    a: *const Matrix<f64>,
    b: *const Matrix<f64>,
) -> *mut Matrix<f64> {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    match (*a).matmul(&*b) {
        Some(result) => Box::into_raw(Box::new(result)),
        None => std::ptr::null_mut(),
    }
}

/// Matrix transpose
#[no_mangle]
pub unsafe extern "C" fn bhc_matrix_transpose_f64(mat: *const Matrix<f64>) -> *mut Matrix<f64> {
    if mat.is_null() {
        return std::ptr::null_mut();
    }
    let result = (*mat).transpose();
    Box::into_raw(Box::new(result))
}

/// Matrix addition
#[no_mangle]
pub unsafe extern "C" fn bhc_matrix_add_f64(
    a: *const Matrix<f64>,
    b: *const Matrix<f64>,
) -> *mut Matrix<f64> {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    match (*a).add(&*b) {
        Some(result) => Box::into_raw(Box::new(result)),
        None => std::ptr::null_mut(),
    }
}

/// Matrix scalar multiplication
#[no_mangle]
pub unsafe extern "C" fn bhc_matrix_scale_f64(
    mat: *const Matrix<f64>,
    scalar: f64,
) -> *mut Matrix<f64> {
    if mat.is_null() {
        return std::ptr::null_mut();
    }
    let result = (*mat).scale(scalar);
    Box::into_raw(Box::new(result))
}

/// Create identity matrix
#[no_mangle]
pub extern "C" fn bhc_matrix_identity_f64(n: usize) -> *mut Matrix<f64> {
    if n == 0 {
        return std::ptr::null_mut();
    }
    let mat = Matrix::<f64>::identity(n);
    Box::into_raw(Box::new(mat))
}

/// Create zeros matrix
#[no_mangle]
pub extern "C" fn bhc_matrix_zeros_f64(rows: usize, cols: usize) -> *mut Matrix<f64> {
    if rows == 0 || cols == 0 {
        return std::ptr::null_mut();
    }
    let mat = Matrix::<f64>::zeros(rows, cols);
    Box::into_raw(Box::new(mat))
}

/// Compute Frobenius norm
#[no_mangle]
pub extern "C" fn bhc_matrix_norm_f64(mat: *const Matrix<f64>) -> f64 {
    if mat.is_null() {
        return 0.0;
    }
    unsafe { (*mat).norm() }
}

/// Compute trace
#[no_mangle]
pub extern "C" fn bhc_matrix_trace_f64(mat: *const Matrix<f64>) -> f64 {
    if mat.is_null() {
        return 0.0;
    }
    unsafe { (*mat).trace() }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.shape(), (2, 2));
    }

    #[test]
    fn test_matrix_creation_invalid() {
        let m: Option<Matrix<f64>> = Matrix::new(vec![1.0, 2.0, 3.0], 2, 2);
        assert!(m.is_none());
    }

    #[test]
    fn test_matrix_zeros() {
        let m: Matrix<f64> = Matrix::zeros(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m[(0, 0)], 0.0);
    }

    #[test]
    fn test_matrix_identity() {
        let m: Matrix<f64> = Matrix::identity(3);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 1.0);
        assert_eq!(m[(2, 2)], 1.0);
        assert_eq!(m[(0, 1)], 0.0);
    }

    #[test]
    fn test_matrix_get_set() {
        let mut m: Matrix<f64> = Matrix::zeros(2, 2);
        m.set(0, 1, 5.0);
        assert_eq!(m.get(0, 1), Some(&5.0));
        assert_eq!(m[(0, 1)], 5.0);
    }

    #[test]
    fn test_matrix_row_col() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        assert_eq!(m.row(0), Some([1.0, 2.0, 3.0].as_slice()));
        assert_eq!(m.col(1), Some(vec![2.0, 5.0]));
    }

    #[test]
    fn test_matrix_transpose() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(0, 1)], 4.0);
        assert_eq!(t[(1, 0)], 2.0);
    }

    #[test]
    fn test_matrix_add() {
        let a: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b: Matrix<f64> = Matrix::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_scale() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let s = m.scale(2.0);
        assert_eq!(s.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matrix_matmul() {
        // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
        // [3 4] x [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
        let a: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b: Matrix<f64> = Matrix::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matrix_matmul_non_square() {
        // [1 2 3]   [7  8 ]   [1*7+2*9+3*11  1*8+2*10+3*12]   [58  64]
        // [4 5 6] x [9  10] = [4*7+5*9+6*11  4*8+5*10+6*12] = [139 154]
        //           [11 12]
        let a: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let b: Matrix<f64> = Matrix::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), (2, 2));
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matrix_matvec() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0]);
        let result = m.matvec(&v).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 11.0]); // [1*1+2*2, 3*1+4*2]
    }

    #[test]
    fn test_matrix_trace() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(m.trace(), 5.0); // 1 + 4
    }

    #[test]
    fn test_matrix_norm() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 2.0, 0.0], 2, 2).unwrap();
        assert_eq!(m.norm(), 3.0); // sqrt(1 + 4 + 4 + 0)
    }

    #[test]
    fn test_matrix_diagonal() {
        let m: Matrix<f64> = Matrix::diagonal(&[1.0, 2.0, 3.0]);
        assert_eq!(m.shape(), (3, 3));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 2.0);
        assert_eq!(m[(2, 2)], 3.0);
        assert_eq!(m[(0, 1)], 0.0);
    }

    #[test]
    fn test_matrix_get_diagonal() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();
        let diag = m.get_diagonal();
        assert_eq!(diag.as_slice(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_matrix_reshape() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let r = m.reshape(3, 2).unwrap();
        assert_eq!(r.shape(), (3, 2));
        assert_eq!(r.as_slice(), m.as_slice());
    }

    #[test]
    fn test_matrix_solve() {
        // [2 1]   [x]   [5]     x = 2
        // [1 3] * [y] = [4]  => y = 1 (not quite, let's check)
        // 2x + y = 5, x + 3y = 4
        // From second: x = 4 - 3y
        // 2(4-3y) + y = 5 => 8 - 6y + y = 5 => -5y = -3 => y = 0.6
        // x = 4 - 1.8 = 2.2
        let m: Matrix<f64> = Matrix::new(vec![2.0, 1.0, 1.0, 3.0], 2, 2).unwrap();
        let b: Vector<f64> = Vector::from_vec(vec![5.0, 4.0]);
        let x = m.solve(&b).unwrap();

        // Verify Ax = b
        let result = m.matvec(&x).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_from_rows() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let m: Matrix<f64> = Matrix::from_rows(rows).unwrap();
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_matrix_display() {
        let m: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let s = format!("{}", m);
        assert!(s.contains("[1"));
        assert!(s.contains("2]"));
    }

    #[test]
    fn test_matrix_equality() {
        let a: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let c: Matrix<f64> = Matrix::new(vec![1.0, 2.0, 3.0, 5.0], 2, 2).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // FFI tests
    #[test]
    fn test_ffi_matrix_f64() {
        unsafe {
            let data = [1.0f64, 2.0, 3.0, 4.0];
            let mat = bhc_matrix_from_f64(data.as_ptr(), 2, 2);
            assert!(!mat.is_null());

            assert_eq!(bhc_matrix_rows_f64(mat), 2);
            assert_eq!(bhc_matrix_cols_f64(mat), 2);
            assert_eq!(bhc_matrix_get_f64(mat, 0, 0), 1.0);
            assert_eq!(bhc_matrix_get_f64(mat, 1, 1), 4.0);

            bhc_matrix_free_f64(mat);
        }
    }

    #[test]
    fn test_ffi_matrix_matmul() {
        unsafe {
            let data_a = [1.0f64, 2.0, 3.0, 4.0];
            let data_b = [5.0f64, 6.0, 7.0, 8.0];

            let a = bhc_matrix_from_f64(data_a.as_ptr(), 2, 2);
            let b = bhc_matrix_from_f64(data_b.as_ptr(), 2, 2);

            let c = bhc_matrix_matmul_f64(a, b);
            assert!(!c.is_null());
            assert_eq!(bhc_matrix_get_f64(c, 0, 0), 19.0);
            assert_eq!(bhc_matrix_get_f64(c, 0, 1), 22.0);
            assert_eq!(bhc_matrix_get_f64(c, 1, 0), 43.0);
            assert_eq!(bhc_matrix_get_f64(c, 1, 1), 50.0);

            bhc_matrix_free_f64(a);
            bhc_matrix_free_f64(b);
            bhc_matrix_free_f64(c);
        }
    }

    #[test]
    fn test_ffi_matrix_transpose() {
        unsafe {
            let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
            let mat = bhc_matrix_from_f64(data.as_ptr(), 2, 3);
            let t = bhc_matrix_transpose_f64(mat);

            assert_eq!(bhc_matrix_rows_f64(t), 3);
            assert_eq!(bhc_matrix_cols_f64(t), 2);

            bhc_matrix_free_f64(mat);
            bhc_matrix_free_f64(t);
        }
    }

    #[test]
    fn test_ffi_matrix_identity() {
        unsafe {
            let mat = bhc_matrix_identity_f64(3);
            assert!(!mat.is_null());
            assert_eq!(bhc_matrix_get_f64(mat, 0, 0), 1.0);
            assert_eq!(bhc_matrix_get_f64(mat, 1, 1), 1.0);
            assert_eq!(bhc_matrix_get_f64(mat, 0, 1), 0.0);
            bhc_matrix_free_f64(mat);
        }
    }
}
