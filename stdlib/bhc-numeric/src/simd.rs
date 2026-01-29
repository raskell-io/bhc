//! SIMD vector types and operations
//!
//! Provides fixed-width SIMD vector types with platform-specific implementations.
//!
//! # Platform Support
//!
//! This module automatically uses the best available SIMD instructions:
//!
//! | Platform | 128-bit | 256-bit | 512-bit |
//! |----------|---------|---------|---------|
//! | x86_64   | SSE/SSE2 | AVX/AVX2 | AVX-512 (future) |
//! | aarch64  | NEON     | -        | SVE (future) |
//!
//! On ARM (Apple Silicon, ARM servers), NEON intrinsics are always available
//! as they are part of the base aarch64 ISA.
//!
//! # Vector Types
//!
//! ## Float vectors
//! - `Vec2F32` - 2 x f32 (64-bit)
//! - `Vec4F32` - 4 x f32 (128-bit, SSE/NEON)
//! - `Vec8F32` - 8 x f32 (256-bit, AVX)
//! - `Vec2F64` - 2 x f64 (128-bit, SSE2/NEON)
//! - `Vec4F64` - 4 x f64 (256-bit, AVX)
//!
//! ## Integer vectors
//! - `Vec4I32` - 4 x i32 (128-bit, SSE2/NEON)
//! - `Vec8I32` - 8 x i32 (256-bit, AVX2)
//! - `Vec2I64` - 2 x i64 (128-bit, SSE2)
//! - `Vec4I64` - 4 x i64 (256-bit, AVX2)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================
// Vec2F32 - 2 x f32 (64-bit)
// ============================================================

/// 2 x f32 vector.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2F32 {
    /// The underlying data.
    pub data: [f32; 2],
}

impl Vec2F32 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: f32) -> Self {
        Self { data: [x, x] }
    }

    /// Create a new vector from values.
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self { data: [x, y] }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            data: [self.data[0] + other.data[0], self.data[1] + other.data[1]],
        }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            data: [self.data[0] - other.data[0], self.data[1] - other.data[1]],
        }
    }

    /// Element-wise multiplication.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self {
            data: [self.data[0] * other.data[0], self.data[1] * other.data[1]],
        }
    }

    /// Element-wise division.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        Self {
            data: [self.data[0] / other.data[0], self.data[1] / other.data[1]],
        }
    }

    /// Horizontal sum.
    #[inline]
    pub fn sum(self) -> f32 {
        self.data[0] + self.data[1]
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.mul(other).sum()
    }
}

// ============================================================
// Vec4F32 - 4 x f32 (128-bit, SSE)
// ============================================================

/// 4 x f32 SIMD vector (128-bit).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4F32 {
    /// The underlying data.
    pub data: [f32; 4],
}

impl Vec4F32 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: f32) -> Self {
        Self { data: [x, x, x, x] }
    }

    /// Create a new vector from 4 values.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { data: [x, y, z, w] }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Load from a slice (must have at least 4 elements).
    #[inline]
    pub fn load(slice: &[f32]) -> Self {
        Self {
            data: [slice[0], slice[1], slice[2], slice[3]],
        }
    }

    /// Store to a mutable slice.
    #[inline]
    pub fn store(&self, slice: &mut [f32]) {
        slice[0] = self.data[0];
        slice[1] = self.data[1];
        slice[2] = self.data[2];
        slice[3] = self.data[3];
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Set element at index.
    #[inline]
    pub fn set(&mut self, idx: usize, val: f32) {
        self.data[idx] = val;
    }

    /// Add two vectors.
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
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f32(self.data.as_ptr());
            let b = vld1q_f32(other.data.as_ptr());
            let r = vaddq_f32(a, b);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [
                self.data[0] + other.data[0],
                self.data[1] + other.data[1],
                self.data[2] + other.data[2],
                self.data[3] + other.data[3],
            ],
        }
    }

    /// Subtract two vectors.
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
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f32(self.data.as_ptr());
            let b = vld1q_f32(other.data.as_ptr());
            let r = vsubq_f32(a, b);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [
                self.data[0] - other.data[0],
                self.data[1] - other.data[1],
                self.data[2] - other.data[2],
                self.data[3] - other.data[3],
            ],
        }
    }

    /// Multiply two vectors element-wise.
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
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f32(self.data.as_ptr());
            let b = vld1q_f32(other.data.as_ptr());
            let r = vmulq_f32(a, b);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [
                self.data[0] * other.data[0],
                self.data[1] * other.data[1],
                self.data[2] * other.data[2],
                self.data[3] * other.data[3],
            ],
        }
    }

    /// Divide two vectors element-wise.
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
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f32(self.data.as_ptr());
            let b = vld1q_f32(other.data.as_ptr());
            let r = vdivq_f32(a, b);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [
                self.data[0] / other.data[0],
                self.data[1] / other.data[1],
                self.data[2] / other.data[2],
                self.data[3] / other.data[3],
            ],
        }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let a = _mm_loadu_ps(self.data.as_ptr());
                let b = _mm_loadu_ps(other.data.as_ptr());
                let r = _mm_min_ps(a, b);
                let mut result = Self::zero();
                _mm_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f32(self.data.as_ptr());
            let b = vld1q_f32(other.data.as_ptr());
            let r = vminq_f32(a, b);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [
                self.data[0].min(other.data[0]),
                self.data[1].min(other.data[1]),
                self.data[2].min(other.data[2]),
                self.data[3].min(other.data[3]),
            ],
        }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let a = _mm_loadu_ps(self.data.as_ptr());
                let b = _mm_loadu_ps(other.data.as_ptr());
                let r = _mm_max_ps(a, b);
                let mut result = Self::zero();
                _mm_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f32(self.data.as_ptr());
            let b = vld1q_f32(other.data.as_ptr());
            let r = vmaxq_f32(a, b);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [
                self.data[0].max(other.data[0]),
                self.data[1].max(other.data[1]),
                self.data[2].max(other.data[2]),
                self.data[3].max(other.data[3]),
            ],
        }
    }

    /// Element-wise square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let a = _mm_loadu_ps(self.data.as_ptr());
                let r = _mm_sqrt_ps(a);
                let mut result = Self::zero();
                _mm_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f32(self.data.as_ptr());
            let r = vsqrtq_f32(a);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [
                self.data[0].sqrt(),
                self.data[1].sqrt(),
                self.data[2].sqrt(),
                self.data[3].sqrt(),
            ],
        }
    }

    /// Element-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f32(self.data.as_ptr());
            let r = vabsq_f32(a);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(target_arch = "aarch64"))]
        Self {
            data: [
                self.data[0].abs(),
                self.data[1].abs(),
                self.data[2].abs(),
                self.data[3].abs(),
            ],
        }
    }

    /// Element-wise negation.
    #[inline]
    pub fn neg(self) -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f32(self.data.as_ptr());
            let r = vnegq_f32(a);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(target_arch = "aarch64"))]
        Self {
            data: [-self.data[0], -self.data[1], -self.data[2], -self.data[3]],
        }
    }

    /// Fused multiply-add: a * b + c.
    #[inline]
    pub fn fma(self, b: Self, c: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("fma") {
                let av = _mm_loadu_ps(self.data.as_ptr());
                let bv = _mm_loadu_ps(b.data.as_ptr());
                let cv = _mm_loadu_ps(c.data.as_ptr());
                let r = _mm_fmadd_ps(av, bv, cv);
                let mut result = Self::zero();
                _mm_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let av = vld1q_f32(self.data.as_ptr());
            let bv = vld1q_f32(b.data.as_ptr());
            let cv = vld1q_f32(c.data.as_ptr());
            let r = vfmaq_f32(cv, av, bv);
            let mut result = Self::zero();
            vst1q_f32(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.mul(b).add(c)
    }

    /// Horizontal sum of all elements.
    #[inline]
    pub fn sum(self) -> f32 {
        self.data[0] + self.data[1] + self.data[2] + self.data[3]
    }

    /// Horizontal product of all elements.
    #[inline]
    pub fn product(self) -> f32 {
        self.data[0] * self.data[1] * self.data[2] * self.data[3]
    }

    /// Minimum element.
    #[inline]
    pub fn min_elem(self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Maximum element.
    #[inline]
    pub fn max_elem(self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.mul(other).sum()
    }
}

// ============================================================
// Vec8F32 - 8 x f32 (256-bit, AVX)
// ============================================================

/// 8 x f32 SIMD vector (256-bit, AVX).
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec8F32 {
    /// The underlying data.
    pub data: [f32; 8],
}

impl Vec8F32 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: f32) -> Self {
        Self { data: [x; 8] }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Load from a slice.
    #[inline]
    pub fn load(slice: &[f32]) -> Self {
        Self {
            data: slice[..8].try_into().unwrap(),
        }
    }

    /// Store to a mutable slice.
    #[inline]
    pub fn store(&self, slice: &mut [f32]) {
        slice[..8].copy_from_slice(&self.data);
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Add two vectors.
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

    /// Subtract two vectors.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") {
                let a = _mm256_loadu_ps(self.data.as_ptr());
                let b = _mm256_loadu_ps(other.data.as_ptr());
                let r = _mm256_sub_ps(a, b);
                let mut result = Self::zero();
                _mm256_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        let mut result = Self::zero();
        for i in 0..8 {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }

    /// Multiply two vectors element-wise.
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

    /// Divide two vectors element-wise.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") {
                let a = _mm256_loadu_ps(self.data.as_ptr());
                let b = _mm256_loadu_ps(other.data.as_ptr());
                let r = _mm256_div_ps(a, b);
                let mut result = Self::zero();
                _mm256_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        let mut result = Self::zero();
        for i in 0..8 {
            result.data[i] = self.data[i] / other.data[i];
        }
        result
    }

    /// Fused multiply-add: a * b + c.
    #[inline]
    pub fn fma(self, b: Self, c: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("fma") {
                let av = _mm256_loadu_ps(self.data.as_ptr());
                let bv = _mm256_loadu_ps(b.data.as_ptr());
                let cv = _mm256_loadu_ps(c.data.as_ptr());
                let r = _mm256_fmadd_ps(av, bv, cv);
                let mut result = Self::zero();
                _mm256_storeu_ps(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        self.mul(b).add(c)
    }

    /// Horizontal sum.
    #[inline]
    pub fn sum(self) -> f32 {
        self.data.iter().sum()
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.mul(other).sum()
    }
}

// ============================================================
// Vec2F64 - 2 x f64 (128-bit, SSE2)
// ============================================================

/// 2 x f64 SIMD vector (128-bit).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2F64 {
    /// The underlying data.
    pub data: [f64; 2],
}

impl Vec2F64 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: f64) -> Self {
        Self { data: [x, x] }
    }

    /// Create a new vector from values.
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { data: [x, y] }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> f64 {
        self.data[idx]
    }

    /// Add two vectors.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse2") {
                let a = _mm_loadu_pd(self.data.as_ptr());
                let b = _mm_loadu_pd(other.data.as_ptr());
                let r = _mm_add_pd(a, b);
                let mut result = Self::zero();
                _mm_storeu_pd(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f64(self.data.as_ptr());
            let b = vld1q_f64(other.data.as_ptr());
            let r = vaddq_f64(a, b);
            let mut result = Self::zero();
            vst1q_f64(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [self.data[0] + other.data[0], self.data[1] + other.data[1]],
        }
    }

    /// Subtract two vectors.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse2") {
                let a = _mm_loadu_pd(self.data.as_ptr());
                let b = _mm_loadu_pd(other.data.as_ptr());
                let r = _mm_sub_pd(a, b);
                let mut result = Self::zero();
                _mm_storeu_pd(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f64(self.data.as_ptr());
            let b = vld1q_f64(other.data.as_ptr());
            let r = vsubq_f64(a, b);
            let mut result = Self::zero();
            vst1q_f64(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [self.data[0] - other.data[0], self.data[1] - other.data[1]],
        }
    }

    /// Multiply two vectors.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse2") {
                let a = _mm_loadu_pd(self.data.as_ptr());
                let b = _mm_loadu_pd(other.data.as_ptr());
                let r = _mm_mul_pd(a, b);
                let mut result = Self::zero();
                _mm_storeu_pd(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f64(self.data.as_ptr());
            let b = vld1q_f64(other.data.as_ptr());
            let r = vmulq_f64(a, b);
            let mut result = Self::zero();
            vst1q_f64(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [self.data[0] * other.data[0], self.data[1] * other.data[1]],
        }
    }

    /// Divide two vectors.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse2") {
                let a = _mm_loadu_pd(self.data.as_ptr());
                let b = _mm_loadu_pd(other.data.as_ptr());
                let r = _mm_div_pd(a, b);
                let mut result = Self::zero();
                _mm_storeu_pd(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a = vld1q_f64(self.data.as_ptr());
            let b = vld1q_f64(other.data.as_ptr());
            let r = vdivq_f64(a, b);
            let mut result = Self::zero();
            vst1q_f64(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self {
            data: [self.data[0] / other.data[0], self.data[1] / other.data[1]],
        }
    }

    /// Fused multiply-add: a * b + c.
    #[inline]
    pub fn fma(self, b: Self, c: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("fma") {
                let av = _mm_loadu_pd(self.data.as_ptr());
                let bv = _mm_loadu_pd(b.data.as_ptr());
                let cv = _mm_loadu_pd(c.data.as_ptr());
                let r = _mm_fmadd_pd(av, bv, cv);
                let mut result = Self::zero();
                _mm_storeu_pd(result.data.as_mut_ptr(), r);
                return result;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let av = vld1q_f64(self.data.as_ptr());
            let bv = vld1q_f64(b.data.as_ptr());
            let cv = vld1q_f64(c.data.as_ptr());
            let r = vfmaq_f64(cv, av, bv);
            let mut result = Self::zero();
            vst1q_f64(result.data.as_mut_ptr(), r);
            return result;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.mul(b).add(c)
    }

    /// Horizontal sum.
    #[inline]
    pub fn sum(self) -> f64 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            return vaddvq_f64(vld1q_f64(self.data.as_ptr()));
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.data[0] + self.data[1]
        }
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.mul(other).sum()
    }
}

// ============================================================
// Vec4F64 - 4 x f64 (256-bit, AVX)
// ============================================================

/// 4 x f64 SIMD vector (256-bit).
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4F64 {
    /// The underlying data.
    pub data: [f64; 4],
}

impl Vec4F64 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: f64) -> Self {
        Self { data: [x; 4] }
    }

    /// Create a new vector from values.
    #[inline]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { data: [x, y, z, w] }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Load from a slice.
    #[inline]
    pub fn load(slice: &[f64]) -> Self {
        Self {
            data: slice[..4].try_into().unwrap(),
        }
    }

    /// Store to a mutable slice.
    #[inline]
    pub fn store(&self, slice: &mut [f64]) {
        slice[..4].copy_from_slice(&self.data);
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> f64 {
        self.data[idx]
    }

    /// Add two vectors.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") {
                let a = _mm256_loadu_pd(self.data.as_ptr());
                let b = _mm256_loadu_pd(other.data.as_ptr());
                let r = _mm256_add_pd(a, b);
                let mut result = Self::zero();
                _mm256_storeu_pd(result.data.as_mut_ptr(), r);
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

    /// Subtract two vectors.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") {
                let a = _mm256_loadu_pd(self.data.as_ptr());
                let b = _mm256_loadu_pd(other.data.as_ptr());
                let r = _mm256_sub_pd(a, b);
                let mut result = Self::zero();
                _mm256_storeu_pd(result.data.as_mut_ptr(), r);
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

    /// Multiply two vectors.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") {
                let a = _mm256_loadu_pd(self.data.as_ptr());
                let b = _mm256_loadu_pd(other.data.as_ptr());
                let r = _mm256_mul_pd(a, b);
                let mut result = Self::zero();
                _mm256_storeu_pd(result.data.as_mut_ptr(), r);
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

    /// Horizontal sum.
    #[inline]
    pub fn sum(self) -> f64 {
        self.data.iter().sum()
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.mul(other).sum()
    }
}

// ============================================================
// Vec4I32 - 4 x i32 (128-bit, SSE2)
// ============================================================

/// 4 x i32 SIMD vector (128-bit).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vec4I32 {
    /// The underlying data.
    pub data: [i32; 4],
}

impl Vec4I32 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: i32) -> Self {
        Self { data: [x; 4] }
    }

    /// Create a new vector from values.
    #[inline]
    pub fn new(x: i32, y: i32, z: i32, w: i32) -> Self {
        Self { data: [x, y, z, w] }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0)
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> i32 {
        self.data[idx]
    }

    /// Add two vectors.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse2") {
                let a = _mm_loadu_si128(self.data.as_ptr() as *const __m128i);
                let b = _mm_loadu_si128(other.data.as_ptr() as *const __m128i);
                let r = _mm_add_epi32(a, b);
                let mut result = Self::zero();
                _mm_storeu_si128(result.data.as_mut_ptr() as *mut __m128i, r);
                return result;
            }
        }
        Self {
            data: [
                self.data[0].wrapping_add(other.data[0]),
                self.data[1].wrapping_add(other.data[1]),
                self.data[2].wrapping_add(other.data[2]),
                self.data[3].wrapping_add(other.data[3]),
            ],
        }
    }

    /// Subtract two vectors.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("sse2") {
                let a = _mm_loadu_si128(self.data.as_ptr() as *const __m128i);
                let b = _mm_loadu_si128(other.data.as_ptr() as *const __m128i);
                let r = _mm_sub_epi32(a, b);
                let mut result = Self::zero();
                _mm_storeu_si128(result.data.as_mut_ptr() as *mut __m128i, r);
                return result;
            }
        }
        Self {
            data: [
                self.data[0].wrapping_sub(other.data[0]),
                self.data[1].wrapping_sub(other.data[1]),
                self.data[2].wrapping_sub(other.data[2]),
                self.data[3].wrapping_sub(other.data[3]),
            ],
        }
    }

    /// Horizontal sum.
    #[inline]
    pub fn sum(self) -> i32 {
        self.data.iter().sum()
    }
}

// ============================================================
// Vec8I32 - 8 x i32 (256-bit, AVX2)
// ============================================================

/// 8 x i32 SIMD vector (256-bit).
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vec8I32 {
    /// The underlying data.
    pub data: [i32; 8],
}

impl Vec8I32 {
    /// Create a new vector with all elements set to the same value.
    #[inline]
    pub fn splat(x: i32) -> Self {
        Self { data: [x; 8] }
    }

    /// Create a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self::splat(0)
    }

    /// Get element at index.
    #[inline]
    pub fn get(&self, idx: usize) -> i32 {
        self.data[idx]
    }

    /// Add two vectors.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                let a = _mm256_loadu_si256(self.data.as_ptr() as *const __m256i);
                let b = _mm256_loadu_si256(other.data.as_ptr() as *const __m256i);
                let r = _mm256_add_epi32(a, b);
                let mut result = Self::zero();
                _mm256_storeu_si256(result.data.as_mut_ptr() as *mut __m256i, r);
                return result;
            }
        }
        let mut result = Self::zero();
        for i in 0..8 {
            result.data[i] = self.data[i].wrapping_add(other.data[i]);
        }
        result
    }

    /// Subtract two vectors.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                let a = _mm256_loadu_si256(self.data.as_ptr() as *const __m256i);
                let b = _mm256_loadu_si256(other.data.as_ptr() as *const __m256i);
                let r = _mm256_sub_epi32(a, b);
                let mut result = Self::zero();
                _mm256_storeu_si256(result.data.as_mut_ptr() as *mut __m256i, r);
                return result;
            }
        }
        let mut result = Self::zero();
        for i in 0..8 {
            result.data[i] = self.data[i].wrapping_sub(other.data[i]);
        }
        result
    }

    /// Horizontal sum.
    #[inline]
    pub fn sum(self) -> i32 {
        self.data.iter().sum()
    }
}

// ============================================================
// FFI Exports
// ============================================================

/// Dot product of two float arrays using SIMD.
#[no_mangle]
pub extern "C" fn bhc_simd_dot_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }
    let a_slice = unsafe { std::slice::from_raw_parts(a, len) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, len) };

    let chunks = len / 8;
    let _remainder = len % 8;
    let mut sum = Vec8F32::zero();

    for i in 0..chunks {
        let av = Vec8F32::load(&a_slice[i * 8..]);
        let bv = Vec8F32::load(&b_slice[i * 8..]);
        sum = sum.add(av.mul(bv));
    }

    let mut result = sum.sum();

    // Handle remainder
    for i in (chunks * 8)..len {
        result += a_slice[i] * b_slice[i];
    }

    result
}

/// Dot product of two double arrays using SIMD.
#[no_mangle]
pub extern "C" fn bhc_simd_dot_f64(a: *const f64, b: *const f64, len: usize) -> f64 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }
    let a_slice = unsafe { std::slice::from_raw_parts(a, len) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, len) };

    let chunks = len / 4;
    let mut sum = Vec4F64::zero();

    for i in 0..chunks {
        let av = Vec4F64::load(&a_slice[i * 4..]);
        let bv = Vec4F64::load(&b_slice[i * 4..]);
        sum = sum.add(av.mul(bv));
    }

    let mut result = sum.sum();

    for i in (chunks * 4)..len {
        result += a_slice[i] * b_slice[i];
    }

    result
}

/// Sum of float array using SIMD.
#[no_mangle]
pub extern "C" fn bhc_simd_sum_f32(ptr: *const f32, len: usize) -> f32 {
    if ptr.is_null() || len == 0 {
        return 0.0;
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    let chunks = len / 8;
    let mut sum = Vec8F32::zero();

    for i in 0..chunks {
        let v = Vec8F32::load(&slice[i * 8..]);
        sum = sum.add(v);
    }

    let mut result = sum.sum();

    for i in (chunks * 8)..len {
        result += slice[i];
    }

    result
}

/// Sum of double array using SIMD.
#[no_mangle]
pub extern "C" fn bhc_simd_sum_f64(ptr: *const f64, len: usize) -> f64 {
    if ptr.is_null() || len == 0 {
        return 0.0;
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    let chunks = len / 4;
    let mut sum = Vec4F64::zero();

    for i in 0..chunks {
        let v = Vec4F64::load(&slice[i * 4..]);
        sum = sum.add(v);
    }

    let mut result = sum.sum();

    for i in (chunks * 4)..len {
        result += slice[i];
    }

    result
}

/// SAXPY: y = a*x + y.
#[no_mangle]
pub extern "C" fn bhc_simd_saxpy(a: f32, x: *const f32, y: *mut f32, len: usize) {
    if x.is_null() || y.is_null() || len == 0 {
        return;
    }
    let x_slice = unsafe { std::slice::from_raw_parts(x, len) };
    let y_slice = unsafe { std::slice::from_raw_parts_mut(y, len) };

    let av = Vec8F32::splat(a);
    let chunks = len / 8;

    for i in 0..chunks {
        let xv = Vec8F32::load(&x_slice[i * 8..]);
        let yv = Vec8F32::load(&y_slice[i * 8..]);
        let rv = xv.fma(av, yv);
        rv.store(&mut y_slice[i * 8..]);
    }

    for i in (chunks * 8)..len {
        y_slice[i] = a * x_slice[i] + y_slice[i];
    }
}

// ============================================================
// SIMD Feature Detection
// ============================================================

/// SIMD features available on the current platform.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdFeatures {
    /// SSE support (x86_64)
    pub sse: bool,
    /// SSE2 support (x86_64)
    pub sse2: bool,
    /// SSE3 support (x86_64)
    pub sse3: bool,
    /// SSSE3 support (x86_64)
    pub ssse3: bool,
    /// SSE4.1 support (x86_64)
    pub sse41: bool,
    /// SSE4.2 support (x86_64)
    pub sse42: bool,
    /// AVX support (x86_64)
    pub avx: bool,
    /// AVX2 support (x86_64)
    pub avx2: bool,
    /// FMA support (x86_64)
    pub fma: bool,
    /// AVX-512F support (x86_64)
    pub avx512f: bool,
    /// NEON support (aarch64) - always true on aarch64
    pub neon: bool,
}

impl SimdFeatures {
    /// Detect SIMD features available on the current platform.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            SimdFeatures {
                sse: is_x86_feature_detected!("sse"),
                sse2: is_x86_feature_detected!("sse2"),
                sse3: is_x86_feature_detected!("sse3"),
                ssse3: is_x86_feature_detected!("ssse3"),
                sse41: is_x86_feature_detected!("sse4.1"),
                sse42: is_x86_feature_detected!("sse4.2"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                fma: is_x86_feature_detected!("fma"),
                avx512f: is_x86_feature_detected!("avx512f"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            SimdFeatures {
                sse: false,
                sse2: false,
                sse3: false,
                ssse3: false,
                sse41: false,
                sse42: false,
                avx: false,
                avx2: false,
                fma: false,
                avx512f: false,
                neon: true, // NEON is always available on aarch64
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdFeatures::default()
        }
    }

    /// Get the best available vector width for f32 operations (in bits).
    pub fn best_f32_width(&self) -> u32 {
        if self.avx512f {
            512
        } else if self.avx || self.avx2 {
            256
        } else if self.sse || self.neon {
            128
        } else {
            64 // Scalar fallback
        }
    }

    /// Get the best available vector width for f64 operations (in bits).
    pub fn best_f64_width(&self) -> u32 {
        if self.avx512f {
            512
        } else if self.avx || self.avx2 {
            256
        } else if self.sse2 || self.neon {
            128
        } else {
            64 // Scalar fallback
        }
    }
}

/// Detect SIMD features on the current platform.
///
/// This function returns a `SimdFeatures` struct describing the SIMD
/// capabilities of the current CPU.
///
/// # Example
///
/// ```ignore
/// let features = detect_simd_features();
/// if features.avx2 {
///     println!("AVX2 is available!");
/// }
/// if features.neon {
///     println!("NEON is available!");
/// }
/// ```
pub fn detect_simd_features() -> SimdFeatures {
    SimdFeatures::detect()
}

/// FFI export for SIMD feature detection.
///
/// Returns a bitfield indicating available SIMD features:
/// - Bit 0: SSE
/// - Bit 1: SSE2
/// - Bit 2: AVX
/// - Bit 3: AVX2
/// - Bit 4: FMA
/// - Bit 5: AVX-512F
/// - Bit 6: NEON
#[no_mangle]
pub extern "C" fn bhc_simd_features() -> u32 {
    let f = SimdFeatures::detect();
    let mut bits = 0u32;
    if f.sse {
        bits |= 1 << 0;
    }
    if f.sse2 {
        bits |= 1 << 1;
    }
    if f.avx {
        bits |= 1 << 2;
    }
    if f.avx2 {
        bits |= 1 << 3;
    }
    if f.fma {
        bits |= 1 << 4;
    }
    if f.avx512f {
        bits |= 1 << 5;
    }
    if f.neon {
        bits |= 1 << 6;
    }
    bits
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2f32() {
        let a = Vec2F32::new(1.0, 2.0);
        let b = Vec2F32::new(3.0, 4.0);
        assert_eq!(a.add(b).data, [4.0, 6.0]);
        assert_eq!(a.mul(b).data, [3.0, 8.0]);
        assert_eq!(a.dot(b), 11.0);
    }

    #[test]
    fn test_vec4f32_add() {
        let a = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F32::new(5.0, 6.0, 7.0, 8.0);
        let c = a.add(b);
        assert_eq!(c.data, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_vec4f32_mul() {
        let a = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F32::new(2.0, 2.0, 2.0, 2.0);
        let c = a.mul(b);
        assert_eq!(c.data, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_vec4f32_sum() {
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.sum(), 10.0);
    }

    #[test]
    fn test_vec4f32_dot() {
        let a = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F32::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(a.dot(b), 10.0);
    }

    #[test]
    fn test_vec4f32_min_max() {
        let a = Vec4F32::new(1.0, 4.0, 2.0, 3.0);
        let b = Vec4F32::new(2.0, 1.0, 5.0, 3.0);
        assert_eq!(a.min(b).data, [1.0, 1.0, 2.0, 3.0]);
        assert_eq!(a.max(b).data, [2.0, 4.0, 5.0, 3.0]);
    }

    #[test]
    fn test_vec4f32_sqrt() {
        let v = Vec4F32::new(4.0, 9.0, 16.0, 25.0);
        let s = v.sqrt();
        assert_eq!(s.data, [2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_vec8f32_add() {
        let a = Vec8F32::splat(1.0);
        let b = Vec8F32::splat(2.0);
        let c = a.add(b);
        assert_eq!(c.data, [3.0; 8]);
    }

    #[test]
    fn test_vec2f64() {
        let a = Vec2F64::new(1.0, 2.0);
        let b = Vec2F64::new(3.0, 4.0);
        assert_eq!(a.add(b).data, [4.0, 6.0]);
        assert_eq!(a.dot(b), 11.0);
    }

    #[test]
    fn test_vec4f64() {
        let a = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4F64::splat(1.0);
        assert_eq!(a.dot(b), 10.0);
    }

    #[test]
    fn test_vec4i32() {
        let a = Vec4I32::new(1, 2, 3, 4);
        let b = Vec4I32::new(5, 6, 7, 8);
        assert_eq!(a.add(b).data, [6, 8, 10, 12]);
        assert_eq!(a.sum(), 10);
    }

    #[test]
    fn test_vec8i32() {
        let a = Vec8I32::splat(1);
        let b = Vec8I32::splat(2);
        assert_eq!(a.add(b).data, [3; 8]);
        assert_eq!(a.sum(), 8);
    }

    #[test]
    fn test_ffi_dot_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32; 8];
        let dot = bhc_simd_dot_f32(a.as_ptr(), b.as_ptr(), a.len());
        assert_eq!(dot, 36.0);
    }

    #[test]
    fn test_ffi_dot_f64() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [1.0f64; 4];
        let dot = bhc_simd_dot_f64(a.as_ptr(), b.as_ptr(), a.len());
        assert_eq!(dot, 10.0);
    }

    #[test]
    fn test_ffi_sum_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let sum = bhc_simd_sum_f32(a.as_ptr(), a.len());
        assert_eq!(sum, 55.0);
    }

    #[test]
    fn test_ffi_saxpy() {
        let a = 2.0f32;
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut y = [1.0f32; 8];
        bhc_simd_saxpy(a, x.as_ptr(), y.as_mut_ptr(), x.len());
        assert_eq!(y, [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
    }
}
