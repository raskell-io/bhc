//! Unboxed array prototype for M0.
//!
//! This module provides a simple unboxed array type that supports the core
//! operations needed for M0: `map`, `zipWith`, `fold`, and `sum`.
//!
//! # Example
//!
//! ```ignore
//! use bhc_core::uarray::UArray;
//!
//! let arr = UArray::from_vec(vec![1i64, 2, 3, 4, 5]);
//! let mapped = arr.map(|x| x + 1);
//! let result = mapped.sum(); // 20
//! ```

use std::fmt;
use std::ops::{Add, Mul};
use std::sync::Arc;

/// An unboxed array with contiguous storage.
///
/// The array stores elements in a flat, contiguous buffer for cache-friendly
/// access patterns. This is the foundation for numeric operations in BHC.
#[derive(Clone)]
pub struct UArray<T> {
    /// The underlying data storage.
    data: Arc<[T]>,
}

impl<T: fmt::Debug> fmt::Debug for UArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UArray{:?}", &*self.data)
    }
}

impl<T: PartialEq> PartialEq for UArray<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Clone> UArray<T> {
    /// Creates a new UArray from a vector.
    #[must_use]
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data: data.into() }
    }

    /// Creates a new UArray from a slice.
    #[must_use]
    pub fn from_slice(data: &[T]) -> Self {
        Self {
            data: data.to_vec().into(),
        }
    }

    /// Returns the length of the array.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a slice of the underlying data.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Gets an element by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Converts the array to a vector.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }

    /// Maps a function over each element, producing a new array.
    ///
    /// This is one of the core operations required for M0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let arr = UArray::from_vec(vec![1, 2, 3]);
    /// let doubled = arr.map(|x| x * 2);
    /// assert_eq!(doubled.to_vec(), vec![2, 4, 6]);
    /// ```
    #[must_use]
    pub fn map<U: Clone, F>(&self, f: F) -> UArray<U>
    where
        F: Fn(&T) -> U,
    {
        let data: Vec<U> = self.data.iter().map(f).collect();
        UArray::from_vec(data)
    }

    /// Zips two arrays with a function, producing a new array.
    ///
    /// The resulting array has the length of the shorter input.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = UArray::from_vec(vec![1, 2, 3]);
    /// let b = UArray::from_vec(vec![4, 5, 6]);
    /// let sum = a.zip_with(&b, |x, y| x + y);
    /// assert_eq!(sum.to_vec(), vec![5, 7, 9]);
    /// ```
    #[must_use]
    pub fn zip_with<U: Clone, V: Clone, F>(&self, other: &UArray<U>, f: F) -> UArray<V>
    where
        F: Fn(&T, &U) -> V,
    {
        let data: Vec<V> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| f(a, b))
            .collect();
        UArray::from_vec(data)
    }

    /// Left fold over the array elements.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let arr = UArray::from_vec(vec![1, 2, 3, 4, 5]);
    /// let sum = arr.fold(0, |acc, x| acc + x);
    /// assert_eq!(sum, 15);
    /// ```
    pub fn fold<A, F>(&self, init: A, f: F) -> A
    where
        F: Fn(A, &T) -> A,
    {
        self.data.iter().fold(init, f)
    }

    /// Right fold over the array elements.
    pub fn foldr<A, F>(&self, init: A, f: F) -> A
    where
        F: Fn(&T, A) -> A,
    {
        self.data.iter().rev().fold(init, |acc, x| f(x, acc))
    }
}

impl<T: Clone + Add<Output = T> + Default> UArray<T> {
    /// Computes the sum of all elements.
    ///
    /// This is one of the core operations required for M0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let arr = UArray::from_vec(vec![1, 2, 3, 4, 5]);
    /// assert_eq!(arr.sum(), 15);
    /// ```
    pub fn sum(&self) -> T {
        self.fold(T::default(), |acc, x| acc + x.clone())
    }
}

impl<T: Clone + Mul<Output = T> + From<i32>> UArray<T> {
    /// Computes the product of all elements.
    pub fn product(&self) -> T {
        self.fold(T::from(1), |acc, x| acc * x.clone())
    }
}

impl<T: Clone + PartialOrd> UArray<T> {
    /// Returns the maximum element, if any.
    pub fn maximum(&self) -> Option<T> {
        self.data
            .iter()
            .cloned()
            .reduce(|a, b| if a > b { a } else { b })
    }

    /// Returns the minimum element, if any.
    pub fn minimum(&self) -> Option<T> {
        self.data
            .iter()
            .cloned()
            .reduce(|a, b| if a < b { a } else { b })
    }
}

/// Specialized operations for numeric arrays.
impl UArray<i64> {
    /// Computes the dot product of two integer arrays.
    pub fn dot(&self, other: &UArray<i64>) -> i64 {
        self.zip_with(other, |a, b| a * b).sum()
    }
}

impl UArray<f64> {
    /// Computes the dot product of two float arrays.
    pub fn dot(&self, other: &UArray<f64>) -> f64 {
        self.zip_with(other, |a, b| a * b).sum()
    }

    /// Computes the mean of all elements.
    pub fn mean(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.sum() / self.len() as f64
        }
    }
}

/// Creates a UArray from a range.
impl UArray<i64> {
    /// Creates an array containing [start, start+1, ..., end-1].
    #[must_use]
    pub fn range(start: i64, end: i64) -> Self {
        Self::from_vec((start..end).collect())
    }

    /// Creates an array containing [0, 1, ..., n-1].
    #[must_use]
    pub fn iota(n: usize) -> Self {
        Self::from_vec((0..n as i64).collect())
    }
}

impl UArray<f64> {
    /// Creates an array of zeros.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self::from_vec(vec![0.0; n])
    }

    /// Creates an array of ones.
    #[must_use]
    pub fn ones(n: usize) -> Self {
        Self::from_vec(vec![1.0; n])
    }

    /// Creates a linearly spaced array.
    #[must_use]
    pub fn linspace(start: f64, end: f64, n: usize) -> Self {
        if n == 0 {
            return Self::from_vec(vec![]);
        }
        if n == 1 {
            return Self::from_vec(vec![start]);
        }
        let step = (end - start) / (n - 1) as f64;
        let data: Vec<f64> = (0..n).map(|i| start + step * i as f64).collect();
        Self::from_vec(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let arr = UArray::from_vec(vec![1, 2, 3]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(&1));
        assert_eq!(arr.get(1), Some(&2));
        assert_eq!(arr.get(2), Some(&3));
    }

    #[test]
    fn test_map() {
        let arr = UArray::from_vec(vec![1i64, 2, 3, 4, 5]);
        let mapped = arr.map(|x| x + 1);
        assert_eq!(mapped.to_vec(), vec![2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_zip_with() {
        let a = UArray::from_vec(vec![1i64, 2, 3]);
        let b = UArray::from_vec(vec![4i64, 5, 6]);
        let result = a.zip_with(&b, |x, y| x + y);
        assert_eq!(result.to_vec(), vec![5, 7, 9]);
    }

    #[test]
    fn test_zip_with_multiply() {
        let a = UArray::from_vec(vec![1i64, 2, 3]);
        let b = UArray::from_vec(vec![4i64, 5, 6]);
        let result = a.zip_with(&b, |x, y| x * y);
        assert_eq!(result.to_vec(), vec![4, 10, 18]);
    }

    #[test]
    fn test_fold() {
        let arr = UArray::from_vec(vec![1i64, 2, 3, 4, 5]);
        let sum = arr.fold(0, |acc, x| acc + x);
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_sum() {
        let arr = UArray::from_vec(vec![1i64, 2, 3, 4, 5]);
        assert_eq!(arr.sum(), 15);
    }

    #[test]
    fn test_product() {
        let arr = UArray::from_vec(vec![1i64, 2, 3, 4, 5]);
        assert_eq!(arr.product(), 120);
    }

    #[test]
    fn test_dot_product() {
        let a = UArray::from_vec(vec![1i64, 2, 3]);
        let b = UArray::from_vec(vec![4i64, 5, 6]);
        assert_eq!(a.dot(&b), 32); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_m0_exit_criteria() {
        // M0 Exit Criteria: sum (map (+1) [1,2,3,4,5]) == 20
        let arr = UArray::from_vec(vec![1i64, 2, 3, 4, 5]);
        let mapped = arr.map(|x| x + 1);
        let result = mapped.sum();
        assert_eq!(result, 20); // (2 + 3 + 4 + 5 + 6) = 20
    }

    #[test]
    fn test_float_operations() {
        let arr = UArray::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        assert!((arr.sum() - 15.0).abs() < f64::EPSILON);
        assert!((arr.mean() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_range() {
        let arr = UArray::<i64>::range(1, 6);
        assert_eq!(arr.to_vec(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_iota() {
        let arr = UArray::<i64>::iota(5);
        assert_eq!(arr.to_vec(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_linspace() {
        let arr = UArray::<f64>::linspace(0.0, 1.0, 5);
        let expected = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        for (a, b) in arr.to_vec().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_max_min() {
        let arr = UArray::from_vec(vec![3i64, 1, 4, 1, 5, 9, 2, 6]);
        assert_eq!(arr.maximum(), Some(9));
        assert_eq!(arr.minimum(), Some(1));
    }

    #[test]
    fn test_empty_array() {
        let arr: UArray<i64> = UArray::from_vec(vec![]);
        assert!(arr.is_empty());
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.sum(), 0);
    }

    #[test]
    fn test_chained_operations() {
        // Test a more complex chain: sum(zipWith(*) (map (+1) a) (map (*2) b))
        let a = UArray::from_vec(vec![1i64, 2, 3]);
        let b = UArray::from_vec(vec![1i64, 2, 3]);

        // map (+1) a = [2, 3, 4]
        // map (*2) b = [2, 4, 6]
        // zipWith (*) = [4, 12, 24]
        // sum = 40
        let a_mapped = a.map(|x| x + 1);
        let b_mapped = b.map(|x| x * 2);
        let zipped = a_mapped.zip_with(&b_mapped, |x, y| x * y);
        let result = zipped.sum();

        assert_eq!(result, 40);
    }
}
