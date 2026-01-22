//! BLAS provider abstraction
//!
//! Abstraction over BLAS implementations (OpenBLAS, MKL, Accelerate).

/// BLAS provider trait
pub trait BlasProvider {
    /// Matrix-matrix multiply: C = alpha * A * B + beta * C
    fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
        beta: f64,
        c: &mut [f64],
        ldc: usize,
    );

    /// Dot product
    fn dot(&self, x: &[f64], y: &[f64]) -> f64;

    /// Scale vector: x = alpha * x
    fn scal(&self, alpha: f64, x: &mut [f64]);

    /// Vector addition: y = alpha * x + y
    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]);
}

/// Pure Rust BLAS implementation (fallback)
pub struct PureRustBlas;

impl BlasProvider for PureRustBlas {
    fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
        beta: f64,
        c: &mut [f64],
        ldc: usize,
    ) {
        // Naive implementation (not optimized)
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * lda + l] * b[l * ldb + j];
                }
                c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
            }
        }
    }

    fn dot(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn scal(&self, alpha: f64, x: &mut [f64]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    fn axpy(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }
}

/// Get the default BLAS provider
pub fn default_provider() -> impl BlasProvider {
    PureRustBlas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let blas = PureRustBlas;
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        assert_eq!(blas.dot(&x, &y), 32.0);
    }

    #[test]
    fn test_axpy() {
        let blas = PureRustBlas;
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        blas.axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![6.0, 9.0, 12.0]);
    }
}
