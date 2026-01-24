//! Sparse tensor operations with SIMD-accelerated kernels.
//!
//! This module provides efficient sparse matrix operations for use by BHC's
//! numeric library. It implements CSR (Compressed Sparse Row) format with
//! SIMD-optimized SpMV and SpMM kernels.
//!
//! # Formats Supported
//!
//! - **CSR** (Compressed Sparse Row): Primary format for row-wise operations
//! - **CSC** (Compressed Sparse Column): For column-wise access
//! - **COO** (Coordinate): For construction and format conversion
//!
//! # FFI Exports
//!
//! This module exports C-ABI functions for BHC to call:
//! - `bhc_sparse_csr_spmv_f64` - Sparse matrix-vector multiply (f64)
//! - `bhc_sparse_csr_spmv_f32` - Sparse matrix-vector multiply (f32)
//! - `bhc_sparse_csr_new` - Create CSR matrix
//! - `bhc_sparse_csr_free` - Free CSR matrix

// ============================================================================
// CSR Sparse Matrix
// ============================================================================

/// CSR (Compressed Sparse Row) sparse matrix.
///
/// Memory layout:
/// - `row_ptrs`: `[rows + 1]` indices into `col_indices`/`values`
/// - `col_indices`: `[nnz]` column indices of non-zeros
/// - `values`: `[nnz]` non-zero values
#[repr(C)]
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    nnz: usize,
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
}

impl<T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>> CsrMatrix<T> {
    /// Create a new CSR matrix.
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `row_ptrs` - Row pointers (length = rows + 1)
    /// * `col_indices` - Column indices (length = nnz)
    /// * `values` - Non-zero values (length = nnz)
    pub fn new(
        rows: usize,
        cols: usize,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, &'static str> {
        if row_ptrs.len() != rows + 1 {
            return Err("row_ptrs must have length rows + 1");
        }
        if row_ptrs[0] != 0 {
            return Err("first row pointer must be 0");
        }
        if col_indices.len() != values.len() {
            return Err("col_indices and values must have same length");
        }
        let nnz = values.len();
        if row_ptrs[rows] != nnz {
            return Err("last row pointer must equal nnz");
        }

        Ok(Self {
            rows,
            cols,
            nnz,
            row_ptrs,
            col_indices,
            values,
        })
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of non-zero elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Density (nnz / total).
    pub fn density(&self) -> f64 {
        if self.rows == 0 || self.cols == 0 {
            0.0
        } else {
            self.nnz as f64 / (self.rows * self.cols) as f64
        }
    }

    /// Get row pointers slice.
    #[inline]
    pub fn row_ptrs(&self) -> &[usize] {
        &self.row_ptrs
    }

    /// Get column indices slice.
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get values slice.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }
}

// ============================================================================
// Sparse Matrix-Vector Multiply (SpMV)
// ============================================================================

/// Sparse matrix-vector multiply: y = A * x
///
/// Uses SIMD acceleration when available.
pub fn spmv_f64(matrix: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    assert_eq!(matrix.cols, x.len(), "x vector size mismatch");
    assert_eq!(matrix.rows, y.len(), "y vector size mismatch");

    // Clear output
    y.iter_mut().for_each(|v| *v = 0.0);

    for row in 0..matrix.rows {
        let row_start = matrix.row_ptrs[row];
        let row_end = matrix.row_ptrs[row + 1];

        let mut sum = 0.0f64;

        // Process elements in this row
        // Use SIMD for longer rows
        let row_nnz = row_end - row_start;

        if row_nnz >= 4 {
            // SIMD path for rows with 4+ elements
            sum = spmv_row_simd_f64(
                &matrix.col_indices[row_start..row_end],
                &matrix.values[row_start..row_end],
                x,
            );
        } else {
            // Scalar path for short rows
            for k in row_start..row_end {
                let col = matrix.col_indices[k];
                let val = matrix.values[k];
                sum += val * x[col];
            }
        }

        y[row] = sum;
    }
}

/// SIMD-accelerated row dot product for f64.
#[cfg(target_arch = "x86_64")]
fn spmv_row_simd_f64(col_indices: &[usize], values: &[f64], x: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let len = values.len();
    let mut sum = 0.0f64;

    // Process 4 elements at a time using AVX
    #[cfg(target_feature = "avx")]
    {
        let chunks = len / 4;
        let mut acc = unsafe { _mm256_setzero_pd() };

        for i in 0..chunks {
            let base = i * 4;

            // Gather x values (no gather instruction for indices, manual load)
            let x0 = x[col_indices[base]];
            let x1 = x[col_indices[base + 1]];
            let x2 = x[col_indices[base + 2]];
            let x3 = x[col_indices[base + 3]];

            let x_vec = unsafe { _mm256_set_pd(x3, x2, x1, x0) };
            let v_vec = unsafe { _mm256_loadu_pd(values.as_ptr().add(base)) };

            // Fused multiply-add if available
            #[cfg(target_feature = "fma")]
            {
                acc = unsafe { _mm256_fmadd_pd(v_vec, x_vec, acc) };
            }
            #[cfg(not(target_feature = "fma"))]
            {
                let prod = unsafe { _mm256_mul_pd(v_vec, x_vec) };
                acc = unsafe { _mm256_add_pd(acc, prod) };
            }
        }

        // Horizontal sum
        let acc_arr: [f64; 4] = unsafe { std::mem::transmute(acc) };
        sum = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3];

        // Process remaining elements
        for i in (chunks * 4)..len {
            let col = col_indices[i];
            let val = values[i];
            sum += val * x[col];
        }
    }

    #[cfg(not(target_feature = "avx"))]
    {
        // Fallback to scalar
        for i in 0..len {
            let col = col_indices[i];
            let val = values[i];
            sum += val * x[col];
        }
    }

    sum
}

#[cfg(not(target_arch = "x86_64"))]
fn spmv_row_simd_f64(col_indices: &[usize], values: &[f64], x: &[f64]) -> f64 {
    // Scalar fallback for non-x86 architectures
    let mut sum = 0.0;
    for (i, &val) in values.iter().enumerate() {
        let col = col_indices[i];
        sum += val * x[col];
    }
    sum
}

/// Sparse matrix-vector multiply for f32.
pub fn spmv_f32(matrix: &CsrMatrix<f32>, x: &[f32], y: &mut [f32]) {
    assert_eq!(matrix.cols, x.len(), "x vector size mismatch");
    assert_eq!(matrix.rows, y.len(), "y vector size mismatch");

    y.iter_mut().for_each(|v| *v = 0.0);

    for row in 0..matrix.rows {
        let row_start = matrix.row_ptrs[row];
        let row_end = matrix.row_ptrs[row + 1];

        let mut sum = 0.0f32;

        let row_nnz = row_end - row_start;

        if row_nnz >= 8 {
            sum = spmv_row_simd_f32(
                &matrix.col_indices[row_start..row_end],
                &matrix.values[row_start..row_end],
                x,
            );
        } else {
            for k in row_start..row_end {
                let col = matrix.col_indices[k];
                let val = matrix.values[k];
                sum += val * x[col];
            }
        }

        y[row] = sum;
    }
}

/// SIMD-accelerated row dot product for f32.
#[cfg(target_arch = "x86_64")]
fn spmv_row_simd_f32(col_indices: &[usize], values: &[f32], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = values.len();
    let mut sum = 0.0f32;

    #[cfg(target_feature = "avx")]
    {
        let chunks = len / 8;
        let mut acc = unsafe { _mm256_setzero_ps() };

        for i in 0..chunks {
            let base = i * 8;

            // Manual gather for x values
            let x_vals: [f32; 8] = [
                x[col_indices[base]],
                x[col_indices[base + 1]],
                x[col_indices[base + 2]],
                x[col_indices[base + 3]],
                x[col_indices[base + 4]],
                x[col_indices[base + 5]],
                x[col_indices[base + 6]],
                x[col_indices[base + 7]],
            ];

            let x_vec = unsafe { _mm256_loadu_ps(x_vals.as_ptr()) };
            let v_vec = unsafe { _mm256_loadu_ps(values.as_ptr().add(base)) };

            #[cfg(target_feature = "fma")]
            {
                acc = unsafe { _mm256_fmadd_ps(v_vec, x_vec, acc) };
            }
            #[cfg(not(target_feature = "fma"))]
            {
                let prod = unsafe { _mm256_mul_ps(v_vec, x_vec) };
                acc = unsafe { _mm256_add_ps(acc, prod) };
            }
        }

        // Horizontal sum
        let acc_arr: [f32; 8] = unsafe { std::mem::transmute(acc) };
        sum = acc_arr.iter().sum();

        // Process remaining elements
        for i in (chunks * 8)..len {
            let col = col_indices[i];
            let val = values[i];
            sum += val * x[col];
        }
    }

    #[cfg(not(target_feature = "avx"))]
    {
        for i in 0..len {
            let col = col_indices[i];
            let val = values[i];
            sum += val * x[col];
        }
    }

    sum
}

#[cfg(not(target_arch = "x86_64"))]
fn spmv_row_simd_f32(col_indices: &[usize], values: &[f32], x: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (i, &val) in values.iter().enumerate() {
        let col = col_indices[i];
        sum += val * x[col];
    }
    sum
}

// ============================================================================
// Sparse Matrix-Matrix Multiply (SpMM)
// ============================================================================

/// Sparse-dense matrix multiply: C = A * B
/// where A is sparse (m x k) and B is dense (k x n).
pub fn spmm_f64(
    matrix: &CsrMatrix<f64>,
    b: &[f64],
    b_cols: usize,
    c: &mut [f64],
) {
    let m = matrix.rows;
    let k = matrix.cols;
    let n = b_cols;

    assert_eq!(b.len(), k * n, "B matrix size mismatch");
    assert_eq!(c.len(), m * n, "C matrix size mismatch");

    // Clear output
    c.iter_mut().for_each(|v| *v = 0.0);

    // For each row of A
    for i in 0..m {
        let row_start = matrix.row_ptrs[i];
        let row_end = matrix.row_ptrs[i + 1];

        // For each non-zero in row i of A
        for k_idx in row_start..row_end {
            let k_col = matrix.col_indices[k_idx];
            let a_val = matrix.values[k_idx];

            // Add a_val * B[k_col, :] to C[i, :]
            for j in 0..n {
                c[i * n + j] += a_val * b[k_col * n + j];
            }
        }
    }
}

/// Sparse-dense matrix multiply for f32.
pub fn spmm_f32(
    matrix: &CsrMatrix<f32>,
    b: &[f32],
    b_cols: usize,
    c: &mut [f32],
) {
    let m = matrix.rows;
    let k = matrix.cols;
    let n = b_cols;

    assert_eq!(b.len(), k * n, "B matrix size mismatch");
    assert_eq!(c.len(), m * n, "C matrix size mismatch");

    c.iter_mut().for_each(|v| *v = 0.0);

    for i in 0..m {
        let row_start = matrix.row_ptrs[i];
        let row_end = matrix.row_ptrs[i + 1];

        for k_idx in row_start..row_end {
            let k_col = matrix.col_indices[k_idx];
            let a_val = matrix.values[k_idx];

            // SIMD-accelerate the row update
            #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
            {
                spmm_row_update_f32(
                    &mut c[i * n..(i + 1) * n],
                    &b[k_col * n..(k_col + 1) * n],
                    a_val,
                );
            }

            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx")))]
            {
                for j in 0..n {
                    c[i * n + j] += a_val * b[k_col * n + j];
                }
            }
        }
    }
}

/// SIMD-accelerated row update: c += a * b
#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
fn spmm_row_update_f32(c: &mut [f32], b: &[f32], a: f32) {
    use std::arch::x86_64::*;

    let n = c.len();
    let a_vec = unsafe { _mm256_set1_ps(a) };

    let chunks = n / 8;
    for i in 0..chunks {
        let base = i * 8;
        let b_vec = unsafe { _mm256_loadu_ps(b.as_ptr().add(base)) };
        let c_vec = unsafe { _mm256_loadu_ps(c.as_ptr().add(base)) };

        #[cfg(target_feature = "fma")]
        let result = unsafe { _mm256_fmadd_ps(a_vec, b_vec, c_vec) };
        #[cfg(not(target_feature = "fma"))]
        let result = unsafe { _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec)) };

        unsafe { _mm256_storeu_ps(c.as_mut_ptr().add(base), result) };
    }

    // Remaining elements
    for j in (chunks * 8)..n {
        c[j] += a * b[j];
    }
}

// ============================================================================
// FFI Exports
// ============================================================================

/// Opaque handle for CSR matrix (f64).
pub struct CsrMatrixF64Handle(CsrMatrix<f64>);

/// Opaque handle for CSR matrix (f32).
pub struct CsrMatrixF32Handle(CsrMatrix<f32>);

/// Create a new CSR matrix (f64).
///
/// # Safety
/// Caller must ensure all pointers are valid and arrays have correct lengths.
#[no_mangle]
pub unsafe extern "C" fn bhc_sparse_csr_new_f64(
    rows: usize,
    cols: usize,
    nnz: usize,
    row_ptrs: *const usize,
    col_indices: *const usize,
    values: *const f64,
) -> *mut CsrMatrixF64Handle {
    let row_ptrs_slice = std::slice::from_raw_parts(row_ptrs, rows + 1);
    let col_indices_slice = std::slice::from_raw_parts(col_indices, nnz);
    let values_slice = std::slice::from_raw_parts(values, nnz);

    match CsrMatrix::new(
        rows,
        cols,
        row_ptrs_slice.to_vec(),
        col_indices_slice.to_vec(),
        values_slice.to_vec(),
    ) {
        Ok(matrix) => Box::into_raw(Box::new(CsrMatrixF64Handle(matrix))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a CSR matrix (f64).
///
/// # Safety
/// Handle must be valid and not previously freed.
#[no_mangle]
pub unsafe extern "C" fn bhc_sparse_csr_free_f64(handle: *mut CsrMatrixF64Handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Sparse matrix-vector multiply (f64).
///
/// Computes y = A * x.
///
/// # Safety
/// All pointers must be valid, x must have length == cols, y must have length == rows.
#[no_mangle]
pub unsafe extern "C" fn bhc_sparse_csr_spmv_f64(
    handle: *const CsrMatrixF64Handle,
    x: *const f64,
    y: *mut f64,
) {
    let matrix = &(*handle).0;
    let x_slice = std::slice::from_raw_parts(x, matrix.cols);
    let y_slice = std::slice::from_raw_parts_mut(y, matrix.rows);

    spmv_f64(matrix, x_slice, y_slice);
}

/// Create a new CSR matrix (f32).
#[no_mangle]
pub unsafe extern "C" fn bhc_sparse_csr_new_f32(
    rows: usize,
    cols: usize,
    nnz: usize,
    row_ptrs: *const usize,
    col_indices: *const usize,
    values: *const f32,
) -> *mut CsrMatrixF32Handle {
    let row_ptrs_slice = std::slice::from_raw_parts(row_ptrs, rows + 1);
    let col_indices_slice = std::slice::from_raw_parts(col_indices, nnz);
    let values_slice = std::slice::from_raw_parts(values, nnz);

    match CsrMatrix::new(
        rows,
        cols,
        row_ptrs_slice.to_vec(),
        col_indices_slice.to_vec(),
        values_slice.to_vec(),
    ) {
        Ok(matrix) => Box::into_raw(Box::new(CsrMatrixF32Handle(matrix))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a CSR matrix (f32).
#[no_mangle]
pub unsafe extern "C" fn bhc_sparse_csr_free_f32(handle: *mut CsrMatrixF32Handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Sparse matrix-vector multiply (f32).
#[no_mangle]
pub unsafe extern "C" fn bhc_sparse_csr_spmv_f32(
    handle: *const CsrMatrixF32Handle,
    x: *const f32,
    y: *mut f32,
) {
    let matrix = &(*handle).0;
    let x_slice = std::slice::from_raw_parts(x, matrix.cols);
    let y_slice = std::slice::from_raw_parts_mut(y, matrix.rows);

    spmv_f32(matrix, x_slice, y_slice);
}

/// Sparse-dense matrix multiply (f64).
///
/// Computes C = A * B where A is sparse (m x k) and B is dense (k x n).
#[no_mangle]
pub unsafe extern "C" fn bhc_sparse_csr_spmm_f64(
    handle: *const CsrMatrixF64Handle,
    b: *const f64,
    b_cols: usize,
    c: *mut f64,
) {
    let matrix = &(*handle).0;
    let k = matrix.cols;
    let m = matrix.rows;
    let n = b_cols;

    let b_slice = std::slice::from_raw_parts(b, k * n);
    let c_slice = std::slice::from_raw_parts_mut(c, m * n);

    spmm_f64(matrix, b_slice, n, c_slice);
}

/// Sparse-dense matrix multiply (f32).
#[no_mangle]
pub unsafe extern "C" fn bhc_sparse_csr_spmm_f32(
    handle: *const CsrMatrixF32Handle,
    b: *const f32,
    b_cols: usize,
    c: *mut f32,
) {
    let matrix = &(*handle).0;
    let k = matrix.cols;
    let m = matrix.rows;
    let n = b_cols;

    let b_slice = std::slice::from_raw_parts(b, k * n);
    let c_slice = std::slice::from_raw_parts_mut(c, m * n);

    spmm_f32(matrix, b_slice, n, c_slice);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_creation() {
        // 3x3 identity matrix
        let row_ptrs = vec![0, 1, 2, 3];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0, 1.0, 1.0];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        assert_eq!(csr.rows(), 3);
        assert_eq!(csr.cols(), 3);
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn test_spmv_f64() {
        // 3x3 identity matrix
        let row_ptrs = vec![0, 1, 2, 3];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0, 1.0, 1.0];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        spmv_f64(&csr, &x, &mut y);

        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_spmv_f64_sparse() {
        // [2 0 1]     [1]   [2*1 + 1*3]   [5]
        // [0 3 0]  *  [2] = [3*2]       = [6]
        // [4 0 5]     [3]   [4*1 + 5*3]   [19]

        let row_ptrs = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 2];
        let values = vec![2.0, 1.0, 3.0, 4.0, 5.0];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        spmv_f64(&csr, &x, &mut y);

        assert_eq!(y, vec![5.0, 6.0, 19.0]);
    }

    #[test]
    fn test_spmm_f64() {
        // A: 2x2 identity
        let row_ptrs = vec![0, 1, 2];
        let col_indices = vec![0, 1];
        let values = vec![1.0, 1.0];

        let csr = CsrMatrix::new(2, 2, row_ptrs, col_indices, values).unwrap();

        // B: 2x2 matrix [[1,2], [3,4]]
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];

        spmm_f64(&csr, &b, 2, &mut c);

        // C should equal B (identity * B = B)
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_density() {
        let row_ptrs = vec![0, 1, 2, 3];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0, 1.0, 1.0];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let density = csr.density();
        assert!((density - 1.0 / 3.0).abs() < 1e-10);
    }
}
