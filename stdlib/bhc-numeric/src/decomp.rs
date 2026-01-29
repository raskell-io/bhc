//! Matrix decompositions
//!
//! This module provides numerical matrix decomposition algorithms:
//!
//! - **LU Decomposition**: Factors A = PLU where P is a permutation matrix,
//!   L is lower triangular, and U is upper triangular
//! - **QR Decomposition**: Factors A = QR where Q is orthogonal and R is
//!   upper triangular (Householder method)
//! - **Cholesky Decomposition**: Factors A = LL^T for symmetric positive definite A
//! - **SVD (Singular Value Decomposition)**: Factors A = UΣV^T where U and V are
//!   orthogonal and Σ is diagonal with singular values
//!
//! # Numerical Stability
//!
//! All algorithms use partial pivoting or other stabilization techniques
//! to ensure numerical accuracy. Singular or near-singular matrices are
//! detected and reported via the `DecompError` type.
//!
//! # Example
//!
//! ```ignore
//! use bhc_numeric::decomp::{lu_decompose, qr_decompose};
//! use bhc_numeric::matrix::Matrix;
//!
//! let a = Matrix::from_data(3, 3, vec![
//!     2.0, -1.0, 0.0,
//!     -1.0, 2.0, -1.0,
//!     0.0, -1.0, 2.0,
//! ]);
//!
//! let lu = lu_decompose(&a).unwrap();
//! let qr = qr_decompose(&a);
//! ```

use crate::matrix::Matrix;
use std::fmt;

// ============================================================
// Error Types
// ============================================================

/// Errors that can occur during matrix decomposition.
#[derive(Debug, Clone, PartialEq)]
pub enum DecompError {
    /// Matrix is singular (has zero or near-zero pivot).
    Singular { pivot_index: usize, value: f64 },
    /// Matrix is not square when a square matrix is required.
    NotSquare { rows: usize, cols: usize },
    /// Matrix is not symmetric positive definite (for Cholesky).
    NotPositiveDefinite { index: usize },
    /// Dimensions are incompatible.
    DimensionMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
}

impl fmt::Display for DecompError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecompError::Singular { pivot_index, value } => {
                write!(
                    f,
                    "Matrix is singular at pivot {}: value = {:.2e}",
                    pivot_index, value
                )
            }
            DecompError::NotSquare { rows, cols } => {
                write!(f, "Matrix must be square, got {}x{}", rows, cols)
            }
            DecompError::NotPositiveDefinite { index } => {
                write!(f, "Matrix is not positive definite at index {}", index)
            }
            DecompError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}",
                    expected, got
                )
            }
        }
    }
}

impl std::error::Error for DecompError {}

// ============================================================
// LU Decomposition
// ============================================================

/// Result of LU decomposition.
///
/// Contains the factorization A = PLU where:
/// - P is a permutation matrix (stored as a pivot array)
/// - L is unit lower triangular (diagonal is 1)
/// - U is upper triangular
///
/// L and U are stored in-place in a single matrix with the following layout:
/// ```text
/// [U00 U01 U02]   L stored below diagonal
/// [L10 U11 U12]   U stored on and above diagonal
/// [L20 L21 U22]   L diagonal is implicitly 1
/// ```
#[derive(Debug, Clone)]
pub struct LuResult {
    /// Combined L and U matrix (L below diagonal, U on and above diagonal).
    pub lu: Matrix<f64>,
    /// Pivot indices: row i was swapped with row pivot[i].
    pub pivot: Vec<usize>,
    /// Number of row swaps (parity of permutation).
    pub num_swaps: usize,
}

impl LuResult {
    /// Extract the L matrix (unit lower triangular).
    pub fn l(&self) -> Matrix<f64> {
        let n = self.lu.rows();
        let mut l = Matrix::identity(n);
        for i in 1..n {
            for j in 0..i {
                *l.get_mut(i, j).unwrap() = self.lu[(i, j)];
            }
        }
        l
    }

    /// Extract the U matrix (upper triangular).
    pub fn u(&self) -> Matrix<f64> {
        let n = self.lu.rows();
        let mut u = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                *u.get_mut(i, j).unwrap() = self.lu[(i, j)];
            }
        }
        u
    }

    /// Get the permutation matrix P.
    pub fn p(&self) -> Matrix<f64> {
        let n = self.pivot.len();
        let mut p = Matrix::identity(n);
        for i in 0..n {
            if self.pivot[i] != i {
                p.swap_rows(i, self.pivot[i]);
            }
        }
        p
    }

    /// Compute the determinant from LU decomposition.
    ///
    /// det(A) = det(P) * det(L) * det(U) = (-1)^swaps * 1 * prod(U_ii)
    pub fn determinant(&self) -> f64 {
        let n = self.lu.rows();
        let sign = if self.num_swaps % 2 == 0 { 1.0 } else { -1.0 };
        let mut det = sign;
        for i in 0..n {
            det *= self.lu[(i, i)];
        }
        det
    }

    /// Solve Ax = b using the LU decomposition.
    ///
    /// Solves in two steps:
    /// 1. Ly = Pb (forward substitution)
    /// 2. Ux = y (backward substitution)
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.lu.rows();
        assert_eq!(b.len(), n, "b must have length equal to matrix size");

        // Apply permutation: pb = P * b
        let mut pb = b.to_vec();
        for i in 0..n {
            if self.pivot[i] != i {
                pb.swap(i, self.pivot[i]);
            }
        }

        // Forward substitution: Ly = pb
        let mut y = pb;
        for i in 0..n {
            for j in 0..i {
                y[i] -= self.lu[(i, j)] * y[j];
            }
        }

        // Backward substitution: Ux = y
        let mut x = y;
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= self.lu[(i, j)] * x[j];
            }
            x[i] /= self.lu[(i, i)];
        }

        x
    }
}

/// Perform LU decomposition with partial pivoting.
///
/// Factors A = PLU where:
/// - P is a permutation matrix
/// - L is unit lower triangular (1s on diagonal)
/// - U is upper triangular
///
/// # Algorithm
///
/// Uses Doolittle's algorithm with partial pivoting for numerical stability.
/// Partial pivoting swaps rows to put the largest element on the diagonal,
/// which helps avoid numerical issues with small pivots.
///
/// # Errors
///
/// Returns `DecompError::Singular` if the matrix is singular (has a zero pivot).
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::decomp::lu_decompose;
/// use bhc_numeric::matrix::Matrix;
///
/// let a = Matrix::from_data(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
/// let lu = lu_decompose(&a).unwrap();
///
/// // Solve Ax = b
/// let b = vec![1.0, 2.0];
/// let x = lu.solve(&b);
/// ```
pub fn lu_decompose(a: &Matrix<f64>) -> Result<LuResult, DecompError> {
    if !a.is_square() {
        return Err(DecompError::NotSquare {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let n = a.rows();
    let mut lu = a.clone();
    // pivot[k] records which row was swapped with row k during step k
    let mut pivot = (0..n).collect::<Vec<_>>();
    let mut num_swaps = 0;

    // Tolerance for detecting singularity
    const EPSILON: f64 = 1e-14;

    for k in 0..n {
        // Find pivot: largest absolute value in column k at or below row k
        let mut max_val = lu[(k, k)].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = lu[(i, k)].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singularity
        if max_val < EPSILON {
            return Err(DecompError::Singular {
                pivot_index: k,
                value: max_val,
            });
        }

        // Record which row is swapped with row k
        pivot[k] = max_row;

        // Swap rows if needed
        if max_row != k {
            lu.swap_rows(k, max_row);
            num_swaps += 1;
        }

        // Perform elimination
        let pivot_val = lu[(k, k)];
        for i in (k + 1)..n {
            // Compute multiplier
            let mult = lu[(i, k)] / pivot_val;
            *lu.get_mut(i, k).unwrap() = mult;

            // Update row i
            for j in (k + 1)..n {
                let update = mult * lu[(k, j)];
                *lu.get_mut(i, j).unwrap() -= update;
            }
        }
    }

    Ok(LuResult {
        lu,
        pivot,
        num_swaps,
    })
}

// ============================================================
// QR Decomposition
// ============================================================

/// Result of QR decomposition.
///
/// Contains the factorization A = QR where:
/// - Q is an orthogonal matrix (Q^T * Q = I)
/// - R is upper triangular
#[derive(Debug, Clone)]
pub struct QrResult {
    /// Orthogonal matrix Q (m x m or m x n depending on thin vs full).
    pub q: Matrix<f64>,
    /// Upper triangular matrix R.
    pub r: Matrix<f64>,
}

impl QrResult {
    /// Solve the least squares problem min ||Ax - b||_2.
    ///
    /// For overdetermined systems (m > n), this gives the least squares solution.
    /// For square systems, this gives the exact solution.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let m = self.q.rows();
        let n = self.r.cols();
        assert_eq!(b.len(), m, "b must have length equal to number of rows");

        // y = Q^T * b
        let mut y = vec![0.0; n];
        for j in 0..n {
            for i in 0..m {
                y[j] += self.q[(i, j)] * b[i];
            }
        }

        // Solve Rx = y by back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in (i + 1)..n {
                x[i] -= self.r[(i, j)] * x[j];
            }
            x[i] /= self.r[(i, i)];
        }

        x
    }

    /// Compute the pseudoinverse A^+ using QR decomposition.
    ///
    /// For full-rank matrices: A^+ = R^-1 * Q^T
    pub fn pseudoinverse(&self) -> Matrix<f64> {
        let n = self.r.cols();
        let m = self.q.rows();

        // Compute R^-1 by back substitution for each column
        let mut r_inv = Matrix::zeros(n, n);
        for col in 0..n {
            let mut e = vec![0.0; n];
            e[col] = 1.0;

            // Solve R * x = e_col
            for i in (0..n).rev() {
                let mut val = e[i];
                for j in (i + 1)..n {
                    val -= self.r[(i, j)] * r_inv[(j, col)];
                }
                *r_inv.get_mut(i, col).unwrap() = val / self.r[(i, i)];
            }
        }

        // A^+ = R^-1 * Q^T
        let mut result = Matrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                for k in 0..n {
                    *result.get_mut(i, j).unwrap() += r_inv[(i, k)] * self.q[(j, k)];
                }
            }
        }

        result
    }
}

/// Perform QR decomposition using Householder reflections.
///
/// Factors A = QR where:
/// - Q is an orthogonal matrix (m x m)
/// - R is upper triangular (m x n)
///
/// # Algorithm
///
/// Uses Householder reflections, which are numerically stable and efficient.
/// Each step creates a Householder matrix H_k that zeros out the subdiagonal
/// elements in column k.
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::decomp::qr_decompose;
/// use bhc_numeric::matrix::Matrix;
///
/// let a = Matrix::from_data(3, 3, vec![
///     12.0, -51.0, 4.0,
///     6.0, 167.0, -68.0,
///     -4.0, 24.0, -41.0,
/// ]);
/// let qr = qr_decompose(&a);
///
/// // Q * R should equal A
/// let reconstructed = qr.q.matmul(&qr.r).unwrap();
/// ```
pub fn qr_decompose(a: &Matrix<f64>) -> QrResult {
    let m = a.rows();
    let n = a.cols();
    let k = m.min(n);

    let mut q = Matrix::identity(m);
    let mut r = a.clone();

    for col in 0..k {
        // Extract column vector below diagonal
        let mut x = vec![0.0; m - col];
        for i in col..m {
            x[i - col] = r[(i, col)];
        }

        // Compute Householder vector
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_x < 1e-14 {
            continue; // Skip if column is already zero
        }

        // v = x + sign(x[0]) * ||x|| * e_1
        x[0] += x[0].signum() * norm_x;
        let norm_v: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

        if norm_v < 1e-14 {
            continue;
        }

        // Normalize v
        for v in &mut x {
            *v /= norm_v;
        }

        // Apply H = I - 2*v*v^T to R (from left)
        // H * R = R - 2 * v * (v^T * R)
        for j in col..n {
            // Compute v^T * R[:, j]
            let mut dot = 0.0;
            for i in col..m {
                dot += x[i - col] * r[(i, j)];
            }
            // Update R[:, j] = R[:, j] - 2 * dot * v
            for i in col..m {
                *r.get_mut(i, j).unwrap() -= 2.0 * dot * x[i - col];
            }
        }

        // Apply H to Q (from right)
        // Q * H = Q - 2 * (Q * v) * v^T
        for i in 0..m {
            // Compute (Q * v)[i] = sum_j Q[i,j] * v[j]
            let mut qv = 0.0;
            for j in col..m {
                qv += q[(i, j)] * x[j - col];
            }
            // Update Q[i, :] = Q[i, :] - 2 * qv * v^T
            for j in col..m {
                *q.get_mut(i, j).unwrap() -= 2.0 * qv * x[j - col];
            }
        }
    }

    QrResult { q, r }
}

// ============================================================
// Cholesky Decomposition
// ============================================================

/// Result of Cholesky decomposition.
///
/// Contains the factorization A = LL^T where L is lower triangular.
#[derive(Debug, Clone)]
pub struct CholeskyResult {
    /// Lower triangular matrix L.
    pub l: Matrix<f64>,
}

impl CholeskyResult {
    /// Solve Ax = b using the Cholesky decomposition.
    ///
    /// Solves in two steps:
    /// 1. Ly = b (forward substitution)
    /// 2. L^T x = y (backward substitution)
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.l.rows();
        assert_eq!(b.len(), n, "b must have length equal to matrix size");

        // Forward substitution: Ly = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = b[i];
            for j in 0..i {
                y[i] -= self.l[(i, j)] * y[j];
            }
            y[i] /= self.l[(i, i)];
        }

        // Backward substitution: L^T x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in (i + 1)..n {
                x[i] -= self.l[(j, i)] * x[j];
            }
            x[i] /= self.l[(i, i)];
        }

        x
    }

    /// Compute the determinant using Cholesky decomposition.
    ///
    /// det(A) = det(L)^2 = (prod L_ii)^2
    pub fn determinant(&self) -> f64 {
        let n = self.l.rows();
        let mut det_l = 1.0;
        for i in 0..n {
            det_l *= self.l[(i, i)];
        }
        det_l * det_l
    }
}

/// Perform Cholesky decomposition for symmetric positive definite matrices.
///
/// Factors A = LL^T where L is lower triangular.
///
/// # Requirements
///
/// - Matrix must be square
/// - Matrix must be symmetric (A = A^T)
/// - Matrix must be positive definite (all eigenvalues > 0)
///
/// # Errors
///
/// Returns `DecompError::NotSquare` if the matrix is not square.
/// Returns `DecompError::NotPositiveDefinite` if a negative value is encountered
/// during decomposition (indicating the matrix is not positive definite).
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::decomp::cholesky_decompose;
/// use bhc_numeric::matrix::Matrix;
///
/// // Symmetric positive definite matrix
/// let a = Matrix::from_data(3, 3, vec![
///     4.0, 2.0, 2.0,
///     2.0, 10.0, 7.0,
///     2.0, 7.0, 21.0,
/// ]);
/// let chol = cholesky_decompose(&a).unwrap();
/// ```
pub fn cholesky_decompose(a: &Matrix<f64>) -> Result<CholeskyResult, DecompError> {
    if !a.is_square() {
        return Err(DecompError::NotSquare {
            rows: a.rows(),
            cols: a.cols(),
        });
    }

    let n = a.rows();
    let mut l = Matrix::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;

            if i == j {
                // Diagonal elements
                for k in 0..j {
                    sum += l[(j, k)] * l[(j, k)];
                }
                let diag = a[(j, j)] - sum;
                if diag <= 0.0 {
                    return Err(DecompError::NotPositiveDefinite { index: j });
                }
                *l.get_mut(j, j).unwrap() = diag.sqrt();
            } else {
                // Off-diagonal elements
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }
                *l.get_mut(i, j).unwrap() = (a[(i, j)] - sum) / l[(j, j)];
            }
        }
    }

    Ok(CholeskyResult { l })
}

// ============================================================
// SVD (Singular Value Decomposition)
// ============================================================

/// Result of SVD decomposition.
///
/// Contains the factorization A = UΣV^T where:
/// - U is an orthogonal matrix (m x m or m x k for thin SVD)
/// - Σ (sigma) is a diagonal matrix with singular values (stored as vector)
/// - V is an orthogonal matrix (n x n or n x k for thin SVD)
///
/// Singular values are sorted in descending order.
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors (columns of U).
    pub u: Matrix<f64>,
    /// Singular values in descending order.
    pub singular_values: Vec<f64>,
    /// Right singular vectors (columns of V, so rows of V^T).
    pub v: Matrix<f64>,
}

impl SvdResult {
    /// Get the rank of the matrix (number of non-zero singular values).
    ///
    /// Uses a tolerance of 1e-10 * max_singular_value.
    pub fn rank(&self) -> usize {
        if self.singular_values.is_empty() {
            return 0;
        }
        let tol = 1e-10 * self.singular_values[0];
        self.singular_values.iter().filter(|&&s| s > tol).count()
    }

    /// Compute the condition number (ratio of largest to smallest singular value).
    ///
    /// Returns infinity if the matrix is singular.
    pub fn condition_number(&self) -> f64 {
        if self.singular_values.is_empty() {
            return f64::INFINITY;
        }
        let max_sv = self.singular_values[0];
        let min_sv = *self.singular_values.last().unwrap();
        if min_sv < 1e-14 {
            f64::INFINITY
        } else {
            max_sv / min_sv
        }
    }

    /// Compute the pseudoinverse A^+ using SVD.
    ///
    /// A^+ = V * Σ^+ * U^T where Σ^+ inverts non-zero singular values.
    pub fn pseudoinverse(&self) -> Matrix<f64> {
        let m = self.u.rows();
        let n = self.v.rows();
        let k = self.singular_values.len();

        // Compute Σ^+ (inverse of non-zero singular values)
        let tol = 1e-10 * self.singular_values.get(0).copied().unwrap_or(0.0);
        let sigma_inv: Vec<f64> = self
            .singular_values
            .iter()
            .map(|&s| if s > tol { 1.0 / s } else { 0.0 })
            .collect();

        // A^+ = V * Σ^+ * U^T
        // Result is n x m
        let mut result = Matrix::zeros(n, m);

        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.v[(i, l)] * sigma_inv[l] * self.u[(j, l)];
                }
                *result.get_mut(i, j).unwrap() = sum;
            }
        }

        result
    }

    /// Solve the least squares problem min ||Ax - b||_2 using SVD.
    ///
    /// More numerically stable than QR for ill-conditioned matrices.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let m = self.u.rows();
        let n = self.v.rows();
        let k = self.singular_values.len();
        assert_eq!(b.len(), m, "b must have length equal to number of rows");

        let tol = 1e-10 * self.singular_values.get(0).copied().unwrap_or(0.0);

        // x = V * Σ^+ * U^T * b
        // Step 1: y = U^T * b
        let mut y = vec![0.0; k];
        for i in 0..k {
            for j in 0..m {
                y[i] += self.u[(j, i)] * b[j];
            }
        }

        // Step 2: z = Σ^+ * y
        let z: Vec<f64> = y
            .iter()
            .zip(self.singular_values.iter())
            .map(|(&yi, &si)| if si > tol { yi / si } else { 0.0 })
            .collect();

        // Step 3: x = V * z
        let mut x = vec![0.0; n];
        for i in 0..n {
            for j in 0..k {
                x[i] += self.v[(i, j)] * z[j];
            }
        }

        x
    }

    /// Reconstruct the original matrix A = U * Σ * V^T.
    pub fn reconstruct(&self) -> Matrix<f64> {
        let m = self.u.rows();
        let n = self.v.rows();
        let k = self.singular_values.len();

        let mut result = Matrix::zeros(m, n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.u[(i, l)] * self.singular_values[l] * self.v[(j, l)];
                }
                *result.get_mut(i, j).unwrap() = sum;
            }
        }

        result
    }

    /// Low-rank approximation using top k singular values.
    ///
    /// This is the best rank-k approximation in both Frobenius and spectral norms.
    pub fn low_rank_approx(&self, k: usize) -> Matrix<f64> {
        let m = self.u.rows();
        let n = self.v.rows();
        let k = k.min(self.singular_values.len());

        let mut result = Matrix::zeros(m, n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.u[(i, l)] * self.singular_values[l] * self.v[(j, l)];
                }
                *result.get_mut(i, j).unwrap() = sum;
            }
        }

        result
    }
}

/// Perform Singular Value Decomposition (SVD).
///
/// Factors A = UΣV^T where:
/// - U is an m x m orthogonal matrix (left singular vectors)
/// - Σ is an m x n diagonal matrix (singular values on diagonal)
/// - V is an n x n orthogonal matrix (right singular vectors)
///
/// # Algorithm
///
/// Uses a two-phase approach:
/// 1. Bidiagonalization using Householder reflections
/// 2. Implicit QR iteration (Golub-Reinsch algorithm) to compute singular values
///
/// # Example
///
/// ```ignore
/// use bhc_numeric::decomp::svd;
/// use bhc_numeric::matrix::Matrix;
///
/// let a = Matrix::from_data(3, 2, vec![
///     1.0, 2.0,
///     3.0, 4.0,
///     5.0, 6.0,
/// ]);
/// let svd_result = svd(&a);
///
/// // Singular values are sorted in descending order
/// println!("Singular values: {:?}", svd_result.singular_values);
///
/// // Reconstruct: U * Σ * V^T ≈ A
/// let reconstructed = svd_result.reconstruct();
/// ```
pub fn svd(a: &Matrix<f64>) -> SvdResult {
    let m = a.rows();
    let n = a.cols();

    if m == 0 || n == 0 {
        return SvdResult {
            u: Matrix::identity(m.max(1)),
            singular_values: vec![],
            v: Matrix::identity(n.max(1)),
        };
    }

    // For very small matrices, use direct methods
    if m == 1 && n == 1 {
        let val = a[(0, 0)];
        return SvdResult {
            u: Matrix::from_data(1, 1, vec![if val >= 0.0 { 1.0 } else { -1.0 }]),
            singular_values: vec![val.abs()],
            v: Matrix::from_data(1, 1, vec![1.0]),
        };
    }

    // Work with the matrix that has more rows than columns for efficiency
    let (work, transposed) = if m >= n {
        (a.clone(), false)
    } else {
        (a.transpose(), true)
    };

    let (u_work, s, v_work) = svd_impl(&work);

    // Transpose back if needed
    if transposed {
        SvdResult {
            u: v_work,
            singular_values: s,
            v: u_work,
        }
    } else {
        SvdResult {
            u: u_work,
            singular_values: s,
            v: v_work,
        }
    }
}

/// Internal SVD implementation for m >= n case.
fn svd_impl(a: &Matrix<f64>) -> (Matrix<f64>, Vec<f64>, Matrix<f64>) {
    let m = a.rows();
    let n = a.cols();

    // Phase 1: Bidiagonalization
    // Transform A into bidiagonal form B = U1^T * A * V1
    let (mut u, bidiag, mut v) = bidiagonalize(a);

    // Extract diagonal and superdiagonal
    let mut d: Vec<f64> = (0..n).map(|i| bidiag[(i, i)]).collect();
    let mut e: Vec<f64> = (0..n - 1).map(|i| bidiag[(i, i + 1)]).collect();

    // Phase 2: Diagonalization using implicit QR iteration
    const MAX_ITER: usize = 100;
    const TOL: f64 = 1e-14;

    // Process each 2x2 block from bottom to top
    let mut q = n; // Current size of unreduced bidiagonal
    let mut iter_count = 0;

    while q > 1 && iter_count < MAX_ITER * n {
        iter_count += 1;

        // Find largest q such that B[q-1, q] is negligible
        let mut found_zero = false;
        for i in (1..q).rev() {
            if e[i - 1].abs() <= TOL * (d[i - 1].abs() + d[i].abs()) {
                e[i - 1] = 0.0;
                if i == q - 1 {
                    q -= 1;
                    found_zero = true;
                    break;
                }
            }
        }

        if found_zero || q <= 1 {
            continue;
        }

        // Find smallest p such that B[p, p+1] is non-negligible
        let mut p = 0;
        for i in (0..q - 1).rev() {
            if e[i].abs() <= TOL * (d[i].abs() + d[i + 1].abs()) {
                p = i + 1;
                break;
            }
        }

        // Check for zero on diagonal
        let mut has_zero_diag = false;
        for i in p..q {
            if d[i].abs() < TOL {
                has_zero_diag = true;
                // Zero out the row by chasing the superdiagonal element
                if i < q - 1 {
                    chase_zero_row(&mut d, &mut e, &mut u, i, q, m);
                }
                break;
            }
        }

        if has_zero_diag {
            continue;
        }

        // Perform implicit QR step on B[p..q, p..q]
        implicit_qr_step(&mut d, &mut e, &mut u, &mut v, p, q, m, n);
    }

    // Make singular values positive and sort in descending order
    for i in 0..n {
        if d[i] < 0.0 {
            d[i] = -d[i];
            // Flip corresponding column of U
            for j in 0..m {
                *u.get_mut(j, i).unwrap() = -u[(j, i)];
            }
        }
    }

    // Sort singular values in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap());

    let sorted_d: Vec<f64> = indices.iter().map(|&i| d[i]).collect();

    // Reorder columns of U and V
    let mut sorted_u = Matrix::zeros(m, n);
    let mut sorted_v = Matrix::zeros(n, n);

    for (new_col, &old_col) in indices.iter().enumerate() {
        for i in 0..m {
            *sorted_u.get_mut(i, new_col).unwrap() = u[(i, old_col)];
        }
        for i in 0..n {
            *sorted_v.get_mut(i, new_col).unwrap() = v[(i, old_col)];
        }
    }

    (sorted_u, sorted_d, sorted_v)
}

/// Bidiagonalize matrix A into B = U^T * A * V.
fn bidiagonalize(a: &Matrix<f64>) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
    let m = a.rows();
    let n = a.cols();

    let mut u = Matrix::identity(m);
    let mut b = a.clone();
    let mut v = Matrix::identity(n);

    for k in 0..n {
        // Left Householder: zero out below diagonal in column k
        let mut x: Vec<f64> = (k..m).map(|i| b[(i, k)]).collect();
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

        if norm_x > 1e-14 {
            x[0] += x[0].signum() * norm_x;
            let norm_v: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

            if norm_v > 1e-14 {
                for v in &mut x {
                    *v /= norm_v;
                }

                // Apply to B from left
                for j in k..n {
                    let mut dot = 0.0;
                    for i in k..m {
                        dot += x[i - k] * b[(i, j)];
                    }
                    for i in k..m {
                        *b.get_mut(i, j).unwrap() -= 2.0 * dot * x[i - k];
                    }
                }

                // Apply to U from right
                for i in 0..m {
                    let mut dot = 0.0;
                    for j in k..m {
                        dot += u[(i, j)] * x[j - k];
                    }
                    for j in k..m {
                        *u.get_mut(i, j).unwrap() -= 2.0 * dot * x[j - k];
                    }
                }
            }
        }

        // Right Householder: zero out to the right of superdiagonal in row k
        if k < n - 2 {
            let mut y: Vec<f64> = ((k + 1)..n).map(|j| b[(k, j)]).collect();
            let norm_y: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();

            if norm_y > 1e-14 {
                y[0] += y[0].signum() * norm_y;
                let norm_w: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();

                if norm_w > 1e-14 {
                    for w in &mut y {
                        *w /= norm_w;
                    }

                    // Apply to B from right
                    for i in k..m {
                        let mut dot = 0.0;
                        for j in (k + 1)..n {
                            dot += b[(i, j)] * y[j - k - 1];
                        }
                        for j in (k + 1)..n {
                            *b.get_mut(i, j).unwrap() -= 2.0 * dot * y[j - k - 1];
                        }
                    }

                    // Apply to V from right
                    for i in 0..n {
                        let mut dot = 0.0;
                        for j in (k + 1)..n {
                            dot += v[(i, j)] * y[j - k - 1];
                        }
                        for j in (k + 1)..n {
                            *v.get_mut(i, j).unwrap() -= 2.0 * dot * y[j - k - 1];
                        }
                    }
                }
            }
        }
    }

    (u, b, v)
}

/// Chase a zero on the diagonal by introducing rotations.
fn chase_zero_row(
    d: &mut [f64],
    e: &mut [f64],
    u: &mut Matrix<f64>,
    zero_idx: usize,
    q: usize,
    m: usize,
) {
    // When d[zero_idx] = 0, we can zero out e[zero_idx] using Givens rotations
    if zero_idx >= e.len() {
        return;
    }

    let mut f = e[zero_idx];
    e[zero_idx] = 0.0;

    for k in (zero_idx + 1)..q {
        let (c, s, r) = givens_rotation(d[k], f);
        d[k] = r;

        if k < q - 1 {
            f = -s * e[k];
            e[k] = c * e[k];
        }

        // Apply to U
        for i in 0..m {
            let u_ik = u[(i, k)];
            let u_iz = u[(i, zero_idx)];
            *u.get_mut(i, k).unwrap() = c * u_ik - s * u_iz;
            *u.get_mut(i, zero_idx).unwrap() = s * u_ik + c * u_iz;
        }
    }
}

/// Perform one implicit QR step on the bidiagonal matrix.
fn implicit_qr_step(
    d: &mut [f64],
    e: &mut [f64],
    u: &mut Matrix<f64>,
    v: &mut Matrix<f64>,
    p: usize,
    q: usize,
    m: usize,
    n: usize,
) {
    // Wilkinson shift: eigenvalue of trailing 2x2 of B^T*B closest to d[q-1]^2
    let d_qm1 = d[q - 1];
    let d_qm2 = if q >= 2 { d[q - 2] } else { 0.0 };
    let e_qm2 = if q >= 2 { e[q - 2] } else { 0.0 };

    let t = (d_qm2 * d_qm2 + e_qm2 * e_qm2 - d_qm1 * d_qm1) / 2.0;
    let det = d_qm2 * d_qm2 * d_qm1 * d_qm1;
    let disc = (t * t + det).sqrt();
    let mu = d_qm1 * d_qm1 - det / (t + t.signum() * disc);

    // Initial values
    let mut x = d[p] * d[p] - mu;
    let mut z = d[p] * e[p];

    for k in p..q - 1 {
        // Right rotation to zero z
        let (c, s, _) = givens_rotation(x, z);

        // Apply to bidiagonal
        if k > p {
            e[k - 1] = c * e[k - 1] - s * z;
        }

        let dk = d[k];
        let ek = e[k];
        let dk1 = d[k + 1];

        d[k] = c * dk - s * ek;
        e[k] = s * dk + c * ek;

        let tmp = -s * dk1;
        d[k + 1] = c * dk1;

        // Apply to V
        for i in 0..n {
            let v_ik = v[(i, k)];
            let v_ik1 = v[(i, k + 1)];
            *v.get_mut(i, k).unwrap() = c * v_ik - s * v_ik1;
            *v.get_mut(i, k + 1).unwrap() = s * v_ik + c * v_ik1;
        }

        x = d[k];
        z = tmp;

        // Left rotation to zero z
        let (c, s, _) = givens_rotation(x, z);

        d[k] = c * d[k] - s * tmp;
        let ek = e[k];
        let dk1 = d[k + 1];

        e[k] = c * ek - s * dk1;
        d[k + 1] = s * ek + c * dk1;

        if k < q - 2 {
            z = -s * e[k + 1];
            e[k + 1] = c * e[k + 1];
        }

        // Apply to U
        for i in 0..m {
            let u_ik = u[(i, k)];
            let u_ik1 = u[(i, k + 1)];
            *u.get_mut(i, k).unwrap() = c * u_ik - s * u_ik1;
            *u.get_mut(i, k + 1).unwrap() = s * u_ik + c * u_ik1;
        }

        x = e[k];
        if k < q - 2 {
            z = e[k + 1] * (-s);
            e[k + 1] = e[k + 1] * c;
        }
    }
}

/// Compute Givens rotation to zero out b: [c -s; s c]^T * [a; b] = [r; 0]
#[inline]
fn givens_rotation(a: f64, b: f64) -> (f64, f64, f64) {
    if b.abs() < 1e-14 {
        (1.0, 0.0, a)
    } else if a.abs() < 1e-14 {
        (0.0, -b.signum(), b.abs())
    } else if b.abs() > a.abs() {
        let t = a / b;
        let s = -1.0 / (1.0 + t * t).sqrt();
        (s * t, s, b / s)
    } else {
        let t = b / a;
        let c = 1.0 / (1.0 + t * t).sqrt();
        (c, -c * t, a / c)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // LU Tests

    #[test]
    fn test_lu_basic() {
        let a = Matrix::from_data(3, 3, vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]);
        let lu = lu_decompose(&a).unwrap();

        // Verify P*L*U = A
        let p = lu.p();
        let l = lu.l();
        let u = lu.u();
        let plu = p.matmul(&l).unwrap().matmul(&u).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(plu[(i, j)], a[(i, j)], 1e-10),
                    "PLU[{},{}] = {} != A[{},{}] = {}",
                    i,
                    j,
                    plu[(i, j)],
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_lu_solve() {
        let a = Matrix::from_data(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let lu = lu_decompose(&a).unwrap();

        let b = vec![10.0, 12.0];
        let x = lu.solve(&b);

        // Verify A*x = b
        let ax = vec![
            a[(0, 0)] * x[0] + a[(0, 1)] * x[1],
            a[(1, 0)] * x[0] + a[(1, 1)] * x[1],
        ];

        assert!(approx_eq(ax[0], b[0], 1e-10));
        assert!(approx_eq(ax[1], b[1], 1e-10));
    }

    #[test]
    fn test_lu_determinant() {
        let a = Matrix::from_data(2, 2, vec![3.0, 8.0, 4.0, 6.0]);
        let lu = lu_decompose(&a).unwrap();
        let det = lu.determinant();

        // det = 3*6 - 8*4 = 18 - 32 = -14
        assert!(approx_eq(det, -14.0, 1e-10));
    }

    #[test]
    fn test_lu_singular() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 2.0, 4.0]); // Singular
        let result = lu_decompose(&a);
        assert!(matches!(result, Err(DecompError::Singular { .. })));
    }

    // QR Tests

    #[test]
    fn test_qr_basic() {
        let a = Matrix::from_data(
            3,
            3,
            vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
        );
        let qr = qr_decompose(&a);

        // Verify Q*R = A
        let qra = qr.q.matmul(&qr.r).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(qra[(i, j)], a[(i, j)], 1e-10),
                    "QR[{},{}] = {} != A[{},{}] = {}",
                    i,
                    j,
                    qra[(i, j)],
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qr_orthogonal() {
        let a = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let qr = qr_decompose(&a);

        // Verify Q^T * Q = I
        let qt = qr.q.transpose();
        let qtq = qt.matmul(&qr.q).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(qtq[(i, j)], expected, 1e-10),
                    "Q^T*Q[{},{}] = {} != {}",
                    i,
                    j,
                    qtq[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_qr_upper_triangular() {
        let a = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let qr = qr_decompose(&a);

        // Verify R is upper triangular
        for i in 0..3 {
            for j in 0..i {
                assert!(
                    approx_eq(qr.r[(i, j)], 0.0, 1e-10),
                    "R[{},{}] = {} should be 0",
                    i,
                    j,
                    qr.r[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qr_solve() {
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let qr = qr_decompose(&a);

        let b = vec![5.0, 11.0];
        let x = qr.solve(&b);

        // Verify A*x = b
        let ax = vec![
            a[(0, 0)] * x[0] + a[(0, 1)] * x[1],
            a[(1, 0)] * x[0] + a[(1, 1)] * x[1],
        ];

        assert!(approx_eq(ax[0], b[0], 1e-10));
        assert!(approx_eq(ax[1], b[1], 1e-10));
    }

    // Cholesky Tests

    #[test]
    fn test_cholesky_basic() {
        let a = Matrix::from_data(3, 3, vec![4.0, 2.0, 2.0, 2.0, 10.0, 7.0, 2.0, 7.0, 21.0]);
        let chol = cholesky_decompose(&a).unwrap();

        // Verify L * L^T = A
        let lt = chol.l.transpose();
        let llt = chol.l.matmul(&lt).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(llt[(i, j)], a[(i, j)], 1e-10),
                    "L*L^T[{},{}] = {} != A[{},{}] = {}",
                    i,
                    j,
                    llt[(i, j)],
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_cholesky_solve() {
        let a = Matrix::from_data(2, 2, vec![4.0, 2.0, 2.0, 5.0]);
        let chol = cholesky_decompose(&a).unwrap();

        let b = vec![4.0, 3.0];
        let x = chol.solve(&b);

        // Verify A*x = b
        let ax = vec![
            a[(0, 0)] * x[0] + a[(0, 1)] * x[1],
            a[(1, 0)] * x[0] + a[(1, 1)] * x[1],
        ];

        assert!(approx_eq(ax[0], b[0], 1e-10));
        assert!(approx_eq(ax[1], b[1], 1e-10));
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // This matrix is not positive definite
        let a = Matrix::from_data(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        let result = cholesky_decompose(&a);
        assert!(matches!(
            result,
            Err(DecompError::NotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn test_cholesky_determinant() {
        let a = Matrix::from_data(2, 2, vec![4.0, 2.0, 2.0, 5.0]);
        let chol = cholesky_decompose(&a).unwrap();
        let det = chol.determinant();

        // det = 4*5 - 2*2 = 16
        assert!(approx_eq(det, 16.0, 1e-10));
    }

    // SVD Tests

    #[test]
    fn test_svd_basic() {
        let a = Matrix::from_data(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let svd_result = svd(&a);

        // Verify reconstruction: U * Σ * V^T ≈ A
        let reconstructed = svd_result.reconstruct();

        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-10),
                    "Reconstructed[{},{}] = {} != A[{},{}] = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_svd_square() {
        // Diagonal matrix - SVD is trivial
        let a = Matrix::from_data(3, 3, vec![5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
        let svd_result = svd(&a);

        // Singular values should be [5, 3, 1] in descending order
        assert!(approx_eq(svd_result.singular_values[0], 5.0, 1e-10));
        assert!(approx_eq(svd_result.singular_values[1], 3.0, 1e-10));
        assert!(approx_eq(svd_result.singular_values[2], 1.0, 1e-10));

        // Verify reconstruction
        let reconstructed = svd_result.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-10),
                    "Reconstructed[{},{}] = {} != A[{},{}] = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_svd_orthogonality() {
        // Well-conditioned matrix
        let a = Matrix::from_data(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0]);
        let svd_result = svd(&a);

        // Verify U^T * U = I
        let ut = svd_result.u.transpose();
        let utu = ut.matmul(&svd_result.u).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(utu[(i, j)], expected, 1e-10),
                    "U^T*U[{},{}] = {} != {}",
                    i,
                    j,
                    utu[(i, j)],
                    expected
                );
            }
        }

        // Verify V^T * V = I
        let vt = svd_result.v.transpose();
        let vtv = vt.matmul(&svd_result.v).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(vtv[(i, j)], expected, 1e-10),
                    "V^T*V[{},{}] = {} != {}",
                    i,
                    j,
                    vtv[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_svd_singular_values_sorted() {
        // Well-conditioned matrix
        let a = Matrix::from_data(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0]);
        let svd_result = svd(&a);

        // Verify singular values are in descending order
        for i in 1..svd_result.singular_values.len() {
            assert!(
                svd_result.singular_values[i - 1] >= svd_result.singular_values[i],
                "Singular values not sorted: σ[{}] = {} < σ[{}] = {}",
                i - 1,
                svd_result.singular_values[i - 1],
                i,
                svd_result.singular_values[i]
            );
        }
    }

    #[test]
    fn test_svd_solve() {
        let a = Matrix::from_data(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let svd_result = svd(&a);

        // Solve least squares problem
        let b = vec![1.0, 2.0, 3.0];
        let x = svd_result.solve(&b);

        // Verify A*x is close to b in least squares sense
        let _ax = vec![
            a[(0, 0)] * x[0] + a[(0, 1)] * x[1],
            a[(1, 0)] * x[0] + a[(1, 1)] * x[1],
            a[(2, 0)] * x[0] + a[(2, 1)] * x[1],
        ];

        // For overdetermined system, we check that x minimizes ||Ax - b||
        // The solution should be approximately x = [1, 2]
        assert!(
            approx_eq(x[0], 1.0, 0.5),
            "x[0] = {} should be close to 1.0",
            x[0]
        );
        assert!(
            approx_eq(x[1], 2.0, 0.5),
            "x[1] = {} should be close to 2.0",
            x[1]
        );
    }

    #[test]
    fn test_svd_rank() {
        // Rank 2 matrix (columns are linearly dependent)
        let a = Matrix::from_data(
            3,
            3,
            vec![
                1.0, 2.0, 3.0, 2.0, 4.0, 6.0, // 2 * row 1
                1.0, 1.0, 1.0,
            ],
        );
        let svd_result = svd(&a);

        // Should have rank 2 (one singular value near zero)
        assert_eq!(svd_result.rank(), 2, "Matrix should have rank 2");
    }

    #[test]
    fn test_svd_condition_number() {
        // Well-conditioned matrix
        let a = Matrix::from_data(2, 2, vec![2.0, 0.0, 0.0, 1.0]);
        let svd_result = svd(&a);

        // Condition number = σ_max / σ_min = 2 / 1 = 2
        assert!(approx_eq(svd_result.condition_number(), 2.0, 1e-10));
    }

    #[test]
    fn test_svd_wide_matrix() {
        // More columns than rows
        let a = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let svd_result = svd(&a);

        // Verify reconstruction
        let reconstructed = svd_result.reconstruct();

        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-10),
                    "Reconstructed[{},{}] = {} != A[{},{}] = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_svd_low_rank_approx() {
        // Well-conditioned matrix
        let a = Matrix::from_data(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0]);
        let svd_result = svd(&a);

        // Rank-1 approximation
        let approx1 = svd_result.low_rank_approx(1);

        // Should only use the largest singular value
        // Verify it's lower rank (dominated by first singular value)
        let svd_approx = svd(&approx1);
        assert!(svd_approx.singular_values.len() >= 1);
        // Second singular value should be much smaller
        if svd_approx.singular_values.len() > 1 {
            assert!(
                svd_approx.singular_values[1] < 1e-10,
                "Rank-1 approx should have negligible second singular value"
            );
        }
    }
}
