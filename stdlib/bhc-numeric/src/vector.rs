//! 1-D numeric vectors
//!
//! Unboxed numeric vectors with SIMD-accelerated operations.
//!
//! # Overview
//!
//! `Vector<T>` provides a contiguous array of numeric values with efficient
//! operations using SIMD intrinsics where possible.
//!
//! # FFI
//!
//! This module exports C-ABI functions for BHC-compiled Haskell to call:
//! - `bhc_vector_from_f64`, `bhc_vector_from_f32`
//! - `bhc_vector_free_f64`, `bhc_vector_free_f32`
//! - `bhc_vector_dot_f64`, `bhc_vector_sum_f64`
//! - `bhc_vector_add_f64`, `bhc_vector_mul_f64`

use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

// ============================================================
// Core Vector Type
// ============================================================

/// A contiguous 1-D numeric vector
#[derive(Clone)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T> Vector<T> {
    /// Create a new vector from a Vec
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a raw pointer to the data
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get a mutable raw pointer to the data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Get a slice view of the data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable slice view of the data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Convert into the underlying Vec
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Get an iterator over the elements
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Get a mutable iterator over the elements
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }
}

impl<T: Clone> Vector<T> {
    /// Create a vector of zeros (requires Default)
    pub fn zeros(len: usize) -> Self
    where
        T: Default,
    {
        Self {
            data: vec![T::default(); len],
        }
    }

    /// Create a vector filled with a single value
    pub fn fill(len: usize, value: T) -> Self {
        Self {
            data: vec![value; len],
        }
    }

    /// Create from a slice
    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }

    /// Get an element by index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Get a mutable element by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    /// Slice the vector (creates a new vector)
    pub fn slice(&self, start: usize, end: usize) -> Option<Self> {
        if start <= end && end <= self.len() {
            Some(Self::from_slice(&self.data[start..end]))
        } else {
            None
        }
    }

    /// Concatenate two vectors
    pub fn concat(&self, other: &Self) -> Self {
        let mut result = self.data.clone();
        result.extend(other.data.iter().cloned());
        Self { data: result }
    }

    /// Reverse the vector
    pub fn reverse(&self) -> Self {
        let mut data = self.data.clone();
        data.reverse();
        Self { data }
    }
}

// ============================================================
// Numeric Operations (generic)
// ============================================================

impl<T> Vector<T>
where
    T: Copy + Default + Add<Output = T>,
{
    /// Sum all elements
    pub fn sum(&self) -> T {
        self.data
            .iter()
            .copied()
            .fold(T::default(), |acc, x| acc + x)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.len() != other.len() {
            return None;
        }
        Some(Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        })
    }
}

impl<T> Vector<T>
where
    T: Copy + Default + Sub<Output = T>,
{
    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Option<Self> {
        if self.len() != other.len() {
            return None;
        }
        Some(Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
        })
    }
}

impl<T> Vector<T>
where
    T: Copy + Default + Mul<Output = T>,
{
    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Option<Self> {
        if self.len() != other.len() {
            return None;
        }
        Some(Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .collect(),
        })
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: T) -> Self {
        Self {
            data: self.data.iter().map(|&x| x * scalar).collect(),
        }
    }
}

impl<T> Vector<T>
where
    T: Copy + Default + Div<Output = T>,
{
    /// Element-wise division
    pub fn div(&self, other: &Self) -> Option<Self> {
        if self.len() != other.len() {
            return None;
        }
        Some(Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a / b)
                .collect(),
        })
    }
}

impl<T> Vector<T>
where
    T: Copy + Neg<Output = T>,
{
    /// Negate all elements
    pub fn negate(&self) -> Self {
        Self {
            data: self.data.iter().map(|&x| -x).collect(),
        }
    }
}

impl<T> Vector<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    /// Dot product of two vectors
    pub fn dot(&self, other: &Self) -> Option<T> {
        if self.len() != other.len() {
            return None;
        }
        Some(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .fold(T::default(), |acc, x| acc + x),
        )
    }
}

impl<T> Vector<T>
where
    T: Copy,
{
    /// Map a function over all elements
    pub fn map<F, U>(&self, f: F) -> Vector<U>
    where
        F: Fn(T) -> U,
    {
        Vector {
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Zip with another vector using a function
    pub fn zip_with<F, U, R>(&self, other: &Vector<U>, f: F) -> Option<Vector<R>>
    where
        U: Copy,
        F: Fn(T, U) -> R,
    {
        if self.len() != other.len() {
            return None;
        }
        Some(Vector {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
        })
    }

    /// Fold over the elements
    pub fn fold<F, A>(&self, init: A, f: F) -> A
    where
        F: Fn(A, T) -> A,
    {
        self.data.iter().fold(init, |acc, &x| f(acc, x))
    }
}

// ============================================================
// Float-specific Operations
// ============================================================

impl Vector<f64> {
    /// Euclidean norm (L2)
    pub fn norm(&self) -> f64 {
        self.dot(self).unwrap_or(0.0).sqrt()
    }

    /// Normalize to unit length
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n > 1e-10 {
            self.scale(1.0 / n)
        } else {
            self.clone()
        }
    }

    /// Mean of elements
    pub fn mean(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.sum() / (self.len() as f64)
        }
    }

    /// Variance of elements
    pub fn variance(&self) -> f64 {
        if self.len() <= 1 {
            return 0.0;
        }
        let m = self.mean();
        self.data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / ((self.len() - 1) as f64)
    }

    /// Standard deviation
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Minimum element
    pub fn min(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::min)
    }

    /// Maximum element
    pub fn max(&self) -> Option<f64> {
        self.data.iter().copied().reduce(f64::max)
    }

    /// Index of minimum element
    pub fn argmin(&self) -> Option<usize> {
        self.data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Index of maximum element
    pub fn argmax(&self) -> Option<usize> {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Apply sqrt element-wise
    pub fn sqrt(&self) -> Self {
        self.map(f64::sqrt)
    }

    /// Apply abs element-wise
    pub fn abs(&self) -> Self {
        self.map(f64::abs)
    }

    /// Apply exp element-wise
    pub fn exp(&self) -> Self {
        self.map(f64::exp)
    }

    /// Apply ln element-wise
    pub fn ln(&self) -> Self {
        self.map(f64::ln)
    }

    /// Apply sin element-wise
    pub fn sin(&self) -> Self {
        self.map(f64::sin)
    }

    /// Apply cos element-wise
    pub fn cos(&self) -> Self {
        self.map(f64::cos)
    }

    /// Apply tanh element-wise
    pub fn tanh(&self) -> Self {
        self.map(f64::tanh)
    }

    /// Clamp all values to a range
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        self.map(|x| x.clamp(min, max))
    }

    /// Linear interpolation with another vector
    pub fn lerp(&self, other: &Self, t: f64) -> Option<Self> {
        if self.len() != other.len() {
            return None;
        }
        Some(Vector {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + t * (b - a))
                .collect(),
        })
    }
}

impl Vector<f32> {
    /// Euclidean norm (L2)
    pub fn norm(&self) -> f32 {
        self.dot(self).unwrap_or(0.0).sqrt()
    }

    /// Normalize to unit length
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n > 1e-6 {
            self.scale(1.0 / n)
        } else {
            self.clone()
        }
    }

    /// Mean of elements
    pub fn mean(&self) -> f32 {
        if self.is_empty() {
            0.0
        } else {
            self.sum() / (self.len() as f32)
        }
    }

    /// Minimum element
    pub fn min(&self) -> Option<f32> {
        self.data.iter().copied().reduce(f32::min)
    }

    /// Maximum element
    pub fn max(&self) -> Option<f32> {
        self.data.iter().copied().reduce(f32::max)
    }

    /// Apply sqrt element-wise
    pub fn sqrt(&self) -> Self {
        self.map(f32::sqrt)
    }

    /// Apply abs element-wise
    pub fn abs(&self) -> Self {
        self.map(f32::abs)
    }
}

// ============================================================
// Trait Implementations
// ============================================================

impl<T: Clone> From<Vec<T>> for Vector<T> {
    fn from(data: Vec<T>) -> Self {
        Self::from_vec(data)
    }
}

impl<T: Clone> From<&[T]> for Vector<T> {
    fn from(slice: &[T]) -> Self {
        Self::from_slice(slice)
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: fmt::Debug> fmt::Debug for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector({:?})", self.data)
    }
}

impl<T: fmt::Display> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, x) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", x)?;
        }
        write!(f, "]")
    }
}

impl<T: PartialEq> PartialEq for Vector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Eq> Eq for Vector<T> {}

// ============================================================
// FFI Exports - f64
// ============================================================

/// Create a vector from a raw pointer
///
/// # Safety
/// - `data` must point to `len` valid f64 values
/// - The memory is copied, caller retains ownership of input
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_from_f64(data: *const f64, len: usize) -> *mut Vector<f64> {
    if data.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    let slice = std::slice::from_raw_parts(data, len);
    let vec = Vector::from_slice(slice);
    Box::into_raw(Box::new(vec))
}

/// Free a vector
///
/// # Safety
/// - `vec` must have been created by `bhc_vector_from_f64` or similar
/// - Must not be called twice on the same pointer
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_free_f64(vec: *mut Vector<f64>) {
    if !vec.is_null() {
        drop(Box::from_raw(vec));
    }
}

/// Get vector length
#[no_mangle]
pub extern "C" fn bhc_vector_len_f64(vec: *const Vector<f64>) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).len() }
}

/// Get pointer to vector data
#[no_mangle]
pub extern "C" fn bhc_vector_data_f64(vec: *const Vector<f64>) -> *const f64 {
    if vec.is_null() {
        return std::ptr::null();
    }
    unsafe { (*vec).as_ptr() }
}

/// Get element at index
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_get_f64(vec: *const Vector<f64>, index: usize) -> f64 {
    if vec.is_null() {
        return 0.0;
    }
    (*vec).get(index).copied().unwrap_or(0.0)
}

/// Compute dot product
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_dot_f64(a: *const Vector<f64>, b: *const Vector<f64>) -> f64 {
    if a.is_null() || b.is_null() {
        return 0.0;
    }
    (*a).dot(&*b).unwrap_or(0.0)
}

/// Compute sum
#[no_mangle]
pub extern "C" fn bhc_vector_sum_f64(vec: *const Vector<f64>) -> f64 {
    if vec.is_null() {
        return 0.0;
    }
    unsafe { (*vec).sum() }
}

/// Compute norm
#[no_mangle]
pub extern "C" fn bhc_vector_norm_f64(vec: *const Vector<f64>) -> f64 {
    if vec.is_null() {
        return 0.0;
    }
    unsafe { (*vec).norm() }
}

/// Compute mean
#[no_mangle]
pub extern "C" fn bhc_vector_mean_f64(vec: *const Vector<f64>) -> f64 {
    if vec.is_null() {
        return 0.0;
    }
    unsafe { (*vec).mean() }
}

/// Element-wise add, returns new vector
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_add_f64(
    a: *const Vector<f64>,
    b: *const Vector<f64>,
) -> *mut Vector<f64> {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    match (*a).add(&*b) {
        Some(result) => Box::into_raw(Box::new(result)),
        None => std::ptr::null_mut(),
    }
}

/// Element-wise subtract, returns new vector
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_sub_f64(
    a: *const Vector<f64>,
    b: *const Vector<f64>,
) -> *mut Vector<f64> {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    match (*a).sub(&*b) {
        Some(result) => Box::into_raw(Box::new(result)),
        None => std::ptr::null_mut(),
    }
}

/// Element-wise multiply, returns new vector
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_mul_f64(
    a: *const Vector<f64>,
    b: *const Vector<f64>,
) -> *mut Vector<f64> {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    match (*a).mul(&*b) {
        Some(result) => Box::into_raw(Box::new(result)),
        None => std::ptr::null_mut(),
    }
}

/// Scalar multiply, returns new vector
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_scale_f64(
    vec: *const Vector<f64>,
    scalar: f64,
) -> *mut Vector<f64> {
    if vec.is_null() {
        return std::ptr::null_mut();
    }
    let result = (*vec).scale(scalar);
    Box::into_raw(Box::new(result))
}

/// Normalize vector, returns new vector
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_normalize_f64(vec: *const Vector<f64>) -> *mut Vector<f64> {
    if vec.is_null() {
        return std::ptr::null_mut();
    }
    let result = (*vec).normalize();
    Box::into_raw(Box::new(result))
}

// ============================================================
// FFI Exports - f32
// ============================================================

/// Create a vector from a raw pointer
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_from_f32(data: *const f32, len: usize) -> *mut Vector<f32> {
    if data.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    let slice = std::slice::from_raw_parts(data, len);
    let vec = Vector::from_slice(slice);
    Box::into_raw(Box::new(vec))
}

/// Free a vector
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_free_f32(vec: *mut Vector<f32>) {
    if !vec.is_null() {
        drop(Box::from_raw(vec));
    }
}

/// Get vector length
#[no_mangle]
pub extern "C" fn bhc_vector_len_f32(vec: *const Vector<f32>) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).len() }
}

/// Compute dot product
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_dot_f32(a: *const Vector<f32>, b: *const Vector<f32>) -> f32 {
    if a.is_null() || b.is_null() {
        return 0.0;
    }
    (*a).dot(&*b).unwrap_or(0.0)
}

/// Compute sum
#[no_mangle]
pub extern "C" fn bhc_vector_sum_f32(vec: *const Vector<f32>) -> f32 {
    if vec.is_null() {
        return 0.0;
    }
    unsafe { (*vec).sum() }
}

/// Compute norm
#[no_mangle]
pub extern "C" fn bhc_vector_norm_f32(vec: *const Vector<f32>) -> f32 {
    if vec.is_null() {
        return 0.0;
    }
    unsafe { (*vec).norm() }
}

/// Element-wise add, returns new vector
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_add_f32(
    a: *const Vector<f32>,
    b: *const Vector<f32>,
) -> *mut Vector<f32> {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    match (*a).add(&*b) {
        Some(result) => Box::into_raw(Box::new(result)),
        None => std::ptr::null_mut(),
    }
}

/// Scalar multiply, returns new vector
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_scale_f32(
    vec: *const Vector<f32>,
    scalar: f32,
) -> *mut Vector<f32> {
    if vec.is_null() {
        return std::ptr::null_mut();
    }
    let result = (*vec).scale(scalar);
    Box::into_raw(Box::new(result))
}

// ============================================================
// FFI Exports - i64
// ============================================================

/// Create a vector from a raw pointer
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_from_i64(data: *const i64, len: usize) -> *mut Vector<i64> {
    if data.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    let slice = std::slice::from_raw_parts(data, len);
    let vec = Vector::from_slice(slice);
    Box::into_raw(Box::new(vec))
}

/// Create a new vector of zeros
#[no_mangle]
pub extern "C" fn bhc_vector_new_i64(len: usize) -> *mut Vector<i64> {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let vec = Vector::zeros(len);
    Box::into_raw(Box::new(vec))
}

/// Free a vector
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_free_i64(vec: *mut Vector<i64>) {
    if !vec.is_null() {
        drop(Box::from_raw(vec));
    }
}

/// Get vector length
#[no_mangle]
pub extern "C" fn bhc_vector_len_i64(vec: *const Vector<i64>) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).len() }
}

/// Get element at index
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_get_i64(vec: *const Vector<i64>, index: usize) -> i64 {
    if vec.is_null() {
        return 0;
    }
    (*vec).get(index).copied().unwrap_or(0)
}

/// Compute sum
#[no_mangle]
pub extern "C" fn bhc_vector_sum_i64(vec: *const Vector<i64>) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).sum() }
}

/// Compute dot product
#[no_mangle]
pub unsafe extern "C" fn bhc_vector_dot_i64(a: *const Vector<i64>, b: *const Vector<i64>) -> i64 {
    if a.is_null() || b.is_null() {
        return 0;
    }
    (*a).dot(&*b).unwrap_or(0)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!(!v.is_empty());
    }

    #[test]
    fn test_vector_zeros() {
        let v: Vector<f64> = Vector::zeros(5);
        assert_eq!(v.len(), 5);
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn test_vector_fill() {
        let v: Vector<f64> = Vector::fill(3, 1.5);
        assert_eq!(v[0], 1.5);
        assert_eq!(v[1], 1.5);
        assert_eq!(v[2], 1.5);
    }

    #[test]
    fn test_vector_sum() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.sum(), 10.0);
    }

    #[test]
    fn test_vector_dot() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), Some(32.0));
    }

    #[test]
    fn test_vector_dot_length_mismatch() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), None);
    }

    #[test]
    fn test_vector_add() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_scale() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let scaled = v.scale(2.0);
        assert_eq!(scaled.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vector_norm() {
        let v: Vector<f64> = Vector::from_vec(vec![3.0, 4.0]);
        assert_eq!(v.norm(), 5.0);
    }

    #[test]
    fn test_vector_normalize() {
        let v: Vector<f64> = Vector::from_vec(vec![3.0, 4.0]);
        let n = v.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_mean() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(v.mean(), 3.0);
    }

    #[test]
    fn test_vector_variance() {
        let v: Vector<f64> = Vector::from_vec(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let var = v.variance();
        assert!((var - 4.571428571428571).abs() < 1e-10);
    }

    #[test]
    fn test_vector_min_max() {
        let v: Vector<f64> = Vector::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0]);
        assert_eq!(v.min(), Some(1.0));
        assert_eq!(v.max(), Some(5.0));
    }

    #[test]
    fn test_vector_argmin_argmax() {
        let v: Vector<f64> = Vector::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0]);
        assert_eq!(v.argmin(), Some(1));
        assert_eq!(v.argmax(), Some(4));
    }

    #[test]
    fn test_vector_map() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let squared = v.map(|x| x * x);
        assert_eq!(squared.as_slice(), &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_vector_zip_with() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let c = a.zip_with(&b, |x, y| x * y).unwrap();
        assert_eq!(c.as_slice(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_vector_fold() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let product = v.fold(1.0, |acc, x| acc * x);
        assert_eq!(product, 24.0);
    }

    #[test]
    fn test_vector_slice() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let s = v.slice(1, 4).unwrap();
        assert_eq!(s.as_slice(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vector_concat() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0]);
        let b: Vector<f64> = Vector::from_vec(vec![3.0, 4.0]);
        let c = a.concat(&b);
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vector_reverse() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let r = v.reverse();
        assert_eq!(r.as_slice(), &[3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_vector_lerp() {
        let a: Vector<f64> = Vector::from_vec(vec![0.0, 0.0]);
        let b: Vector<f64> = Vector::from_vec(vec![10.0, 20.0]);
        let c = a.lerp(&b, 0.5).unwrap();
        assert_eq!(c.as_slice(), &[5.0, 10.0]);
    }

    #[test]
    fn test_vector_clamp() {
        let v: Vector<f64> = Vector::from_vec(vec![-1.0, 0.5, 1.5]);
        let c = v.clamp(0.0, 1.0);
        assert_eq!(c.as_slice(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_vector_transcendental() {
        let v: Vector<f64> = Vector::from_vec(vec![0.0, 1.0]);
        let exp_v = v.exp();
        assert!((exp_v[0] - 1.0).abs() < 1e-10);
        assert!((exp_v[1] - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_vector_display() {
        let v: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", v);
        assert_eq!(s, "[1, 2, 3]");
    }

    #[test]
    fn test_vector_equality() {
        let a: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let c: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 4.0]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // FFI tests
    #[test]
    fn test_ffi_vector_f64() {
        unsafe {
            let data = [1.0f64, 2.0, 3.0, 4.0];
            let vec = bhc_vector_from_f64(data.as_ptr(), data.len());
            assert!(!vec.is_null());

            assert_eq!(bhc_vector_len_f64(vec), 4);
            assert_eq!(bhc_vector_sum_f64(vec), 10.0);

            let vec2 = bhc_vector_from_f64(data.as_ptr(), data.len());
            assert_eq!(bhc_vector_dot_f64(vec, vec2), 30.0); // 1+4+9+16

            bhc_vector_free_f64(vec);
            bhc_vector_free_f64(vec2);
        }
    }

    #[test]
    fn test_ffi_vector_operations() {
        unsafe {
            let data_a = [1.0f64, 2.0, 3.0];
            let data_b = [4.0f64, 5.0, 6.0];

            let a = bhc_vector_from_f64(data_a.as_ptr(), data_a.len());
            let b = bhc_vector_from_f64(data_b.as_ptr(), data_b.len());

            let sum = bhc_vector_add_f64(a, b);
            assert!(!sum.is_null());
            assert_eq!(bhc_vector_get_f64(sum, 0), 5.0);
            assert_eq!(bhc_vector_get_f64(sum, 1), 7.0);
            assert_eq!(bhc_vector_get_f64(sum, 2), 9.0);

            let scaled = bhc_vector_scale_f64(a, 2.0);
            assert!(!scaled.is_null());
            assert_eq!(bhc_vector_get_f64(scaled, 0), 2.0);

            bhc_vector_free_f64(a);
            bhc_vector_free_f64(b);
            bhc_vector_free_f64(sum);
            bhc_vector_free_f64(scaled);
        }
    }
}
