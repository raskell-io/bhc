//! SIMD vector types and operations
//!
//! Provides fixed-width SIMD vector types with platform-specific implementations.
//!
//! # Vector Types
//!
//! - `Vec4F32` - 4 x f32 (128-bit, SSE)
//! - `Vec8F32` - 8 x f32 (256-bit, AVX)
//! - `Vec4F64` - 4 x f64 (256-bit, AVX)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 4 x f32 SIMD vector (128-bit)
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Vec4F32 {
    data: [f32; 4],
}

impl Vec4F32 {
    /// Create a new vector with all elements set to the same value
    #[inline]
    pub fn splat(x: f32) -> Self {
        Self { data: [x, x, x, x] }
    }

    /// Create a new vector from 4 values
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { data: [x, y, z, w] }
    }

    /// Create a zero vector
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Get element at index
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Set element at index
    #[inline]
    pub fn set(&mut self, idx: usize, val: f32) {
        self.data[idx] = val;
    }

    /// Add two vectors
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let a = _mm_loadu_ps(self.data.as_ptr());
                let b = _mm_loadu_ps(other.data.as_ptr());
                let r = _mm_add_ps(a, b);
                let mut result = Self::zero();
                _mm_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }

        Self {
            data: [
                self.data[0] + other.data[0],
                self.data[1] + other.data[1],
                self.data[2] + other.data[2],
                self.data[3] + other.data[3],
            ],
        }
    }

    /// Subtract two vectors
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let a = _mm_loadu_ps(self.data.as_ptr());
                let b = _mm_loadu_ps(other.data.as_ptr());
                let r = _mm_sub_ps(a, b);
                let mut result = Self::zero();
                _mm_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }

        Self {
            data: [
                self.data[0] - other.data[0],
                self.data[1] - other.data[1],
                self.data[2] - other.data[2],
                self.data[3] - other.data[3],
            ],
        }
    }

    /// Multiply two vectors element-wise
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let a = _mm_loadu_ps(self.data.as_ptr());
                let b = _mm_loadu_ps(other.data.as_ptr());
                let r = _mm_mul_ps(a, b);
                let mut result = Self::zero();
                _mm_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }

        Self {
            data: [
                self.data[0] * other.data[0],
                self.data[1] * other.data[1],
                self.data[2] * other.data[2],
                self.data[3] * other.data[3],
            ],
        }
    }

    /// Divide two vectors element-wise
    #[inline]
    pub fn div(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let a = _mm_loadu_ps(self.data.as_ptr());
                let b = _mm_loadu_ps(other.data.as_ptr());
                let r = _mm_div_ps(a, b);
                let mut result = Self::zero();
                _mm_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }

        Self {
            data: [
                self.data[0] / other.data[0],
                self.data[1] / other.data[1],
                self.data[2] / other.data[2],
                self.data[3] / other.data[3],
            ],
        }
    }

    /// Horizontal sum of all elements
    #[inline]
    pub fn sum(self) -> f32 {
        self.data[0] + self.data[1] + self.data[2] + self.data[3]
    }

    /// Horizontal product of all elements
    #[inline]
    pub fn product(self) -> f32 {
        self.data[0] * self.data[1] * self.data[2] * self.data[3]
    }

    /// Minimum element
    #[inline]
    pub fn min_elem(self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Maximum element
    #[inline]
    pub fn max_elem(self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
}

/// 8 x f32 SIMD vector (256-bit, AVX)
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct Vec8F32 {
    data: [f32; 8],
}

impl Vec8F32 {
    /// Create a new vector with all elements set to the same value
    #[inline]
    pub fn splat(x: f32) -> Self {
        Self { data: [x; 8] }
    }

    /// Create a zero vector
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Add two vectors
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") {
                let a = _mm256_loadu_ps(self.data.as_ptr());
                let b = _mm256_loadu_ps(other.data.as_ptr());
                let r = _mm256_add_ps(a, b);
                let mut result = Self::zero();
                _mm256_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }

        let mut result = Self::zero();
        for i in 0..8 {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    /// Multiply two vectors element-wise
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") {
                let a = _mm256_loadu_ps(self.data.as_ptr());
                let b = _mm256_loadu_ps(other.data.as_ptr());
                let r = _mm256_mul_ps(a, b);
                let mut result = Self::zero();
                _mm256_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }

        let mut result = Self::zero();
        for i in 0..8 {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }

    /// Horizontal sum
    #[inline]
    pub fn sum(self) -> f32 {
        self.data.iter().sum()
    }
}

// FFI exports

/// Dot product of two float arrays using SIMD
#[no_mangle]
pub extern "C" fn bhc_simd_dot_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }

    let a_slice = unsafe { std::slice::from_raw_parts(a, len) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, len) };

    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum = Vec8F32::zero();

    for i in 0..chunks {
        let av = Vec8F32 {
            data: a_slice[i * 8..][..8].try_into().unwrap(),
        };
        let bv = Vec8F32 {
            data: b_slice[i * 8..][..8].try_into().unwrap(),
        };
        sum = sum.add(av.mul(bv));
    }

    let mut result = sum.sum();

    // Handle remainder
    for i in (chunks * 8)..len {
        result += a_slice[i] * b_slice[i];
    }

    result
}

/// Sum of float array using SIMD
#[no_mangle]
pub extern "C" fn bhc_simd_sum_f32(ptr: *const f32, len: usize) -> f32 {
    if ptr.is_null() || len == 0 {
        return 0.0;
    }

    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    let chunks = len / 8;
    let mut sum = Vec8F32::zero();

    for i in 0..chunks {
        let v = Vec8F32 {
            data: slice[i * 8..][..8].try_into().unwrap(),
        };
        sum = sum.add(v);
    }

    let mut result = sum.sum();

    for i in (chunks * 8)..len {
        result += slice[i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec4_add() {
        let a = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F32::new(5.0, 6.0, 7.0, 8.0);
        let c = a.add(b);
        assert_eq!(c.data, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_vec4_sum() {
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.sum(), 10.0);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let dot = bhc_simd_dot_f32(a.as_ptr(), b.as_ptr(), a.len());
        assert_eq!(dot, 36.0);
    }
}
