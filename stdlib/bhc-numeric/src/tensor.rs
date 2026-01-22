//! N-dimensional tensors
//!
//! Shape-aware tensor operations with guaranteed fusion.
//!
//! # Overview
//!
//! This module provides a Tensor type that supports:
//! - Arbitrary dimensions (scalars, vectors, matrices, N-d arrays)
//! - Efficient memory layouts with strides
//! - Zero-copy views (transpose, slice, reshape)
//! - SIMD-accelerated operations
//! - Parallel operations for large tensors
//!
//! # Example
//!
//! ```
//! use bhc_numeric::tensor::{Tensor, Shape};
//!
//! let t = Tensor::<f64>::zeros(&[2, 3]);
//! let t2 = t.map(|x| x + 1.0);
//! let sum = t2.sum();
//! ```

use std::fmt;
use std::sync::Arc;

// ============================================================================
// Shape and Stride
// ============================================================================

/// A tensor shape (list of dimension sizes)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions
    pub fn new(dims: &[usize]) -> Self {
        Shape {
            dims: dims.to_vec(),
        }
    }

    /// Scalar shape (0 dimensions)
    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    /// Vector shape (1 dimension)
    pub fn vector(len: usize) -> Self {
        Shape { dims: vec![len] }
    }

    /// Matrix shape (2 dimensions)
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Shape {
            dims: vec![rows, cols],
        }
    }

    /// Number of dimensions (rank)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Get dimension at index
    pub fn dim(&self, i: usize) -> usize {
        self.dims[i]
    }

    /// Get all dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Total number of elements
    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// Check if shapes are broadcast-compatible
    pub fn is_broadcast_compatible(&self, other: &Shape) -> bool {
        let self_dims = self.dims.iter().rev();
        let other_dims = other.dims.iter().rev();

        for (a, b) in self_dims.zip(other_dims) {
            if *a != *b && *a != 1 && *b != 1 {
                return false;
            }
        }
        true
    }

    /// Compute broadcast result shape
    pub fn broadcast_shape(&self, other: &Shape) -> Option<Shape> {
        if !self.is_broadcast_compatible(other) {
            return None;
        }

        let max_rank = self.rank().max(other.rank());
        let mut result = vec![0; max_rank];

        for i in 0..max_rank {
            let self_dim = if i < self.rank() {
                self.dims[self.rank() - 1 - i]
            } else {
                1
            };
            let other_dim = if i < other.rank() {
                other.dims[other.rank() - 1 - i]
            } else {
                1
            };
            result[max_rank - 1 - i] = self_dim.max(other_dim);
        }

        Some(Shape { dims: result })
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

/// Compute row-major (C-style) strides for a shape
pub fn row_major_strides(shape: &Shape) -> Vec<usize> {
    if shape.rank() == 0 {
        return vec![];
    }

    let mut strides = vec![1; shape.rank()];
    for i in (0..shape.rank() - 1).rev() {
        strides[i] = strides[i + 1] * shape.dim(i + 1);
    }
    strides
}

/// Compute column-major (Fortran-style) strides for a shape
pub fn col_major_strides(shape: &Shape) -> Vec<usize> {
    if shape.rank() == 0 {
        return vec![];
    }

    let mut strides = vec![1; shape.rank()];
    for i in 1..shape.rank() {
        strides[i] = strides[i - 1] * shape.dim(i - 1);
    }
    strides
}

// ============================================================================
// Tensor Storage
// ============================================================================

/// Memory layout information
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Contiguous row-major (C-style)
    RowMajor,
    /// Contiguous column-major (Fortran-style)
    ColMajor,
    /// General strided layout
    Strided,
}

/// Tensor data storage (shared via Arc for cheap cloning)
#[derive(Clone)]
struct TensorStorage<T> {
    data: Arc<Vec<T>>,
    offset: usize,
}

impl<T: Clone> TensorStorage<T> {
    fn new(data: Vec<T>) -> Self {
        TensorStorage {
            data: Arc::new(data),
            offset: 0,
        }
    }

    fn with_offset(data: Arc<Vec<T>>, offset: usize) -> Self {
        TensorStorage { data, offset }
    }

    fn get(&self, index: usize) -> &T {
        &self.data[self.offset + index]
    }

    fn len(&self) -> usize {
        self.data.len() - self.offset
    }

    fn as_slice(&self) -> &[T] {
        &self.data[self.offset..]
    }

    fn make_contiguous(&self, shape: &Shape, strides: &[usize]) -> Vec<T> {
        let n = shape.num_elements();
        let mut result = Vec::with_capacity(n);

        for flat_idx in 0..n {
            let physical_idx = self.compute_index(flat_idx, shape, strides);
            result.push(self.data[self.offset + physical_idx].clone());
        }

        result
    }

    fn compute_index(&self, flat_idx: usize, shape: &Shape, strides: &[usize]) -> usize {
        let row_strides = row_major_strides(shape);
        let mut idx = 0;
        let mut remaining = flat_idx;

        for i in 0..shape.rank() {
            let coord = remaining / row_strides[i];
            remaining %= row_strides[i];
            idx += coord * strides[i];
        }

        idx
    }
}

// ============================================================================
// Tensor
// ============================================================================

/// An N-dimensional tensor with shape and stride information
#[derive(Clone)]
pub struct Tensor<T> {
    storage: TensorStorage<T>,
    shape: Shape,
    strides: Vec<usize>,
    layout: Layout,
}

impl<T: Clone + Default> Tensor<T> {
    /// Create a tensor filled with default values
    pub fn default_filled(shape: &[usize]) -> Self {
        let shape = Shape::new(shape);
        let n = shape.num_elements();
        let data = vec![T::default(); n];
        let strides = row_major_strides(&shape);

        Tensor {
            storage: TensorStorage::new(data),
            shape,
            strides,
            layout: Layout::RowMajor,
        }
    }
}

impl<T: Clone> Tensor<T> {
    /// Create a tensor from data and shape
    pub fn from_data(data: Vec<T>, shape: &[usize]) -> Result<Self, TensorError> {
        let shape = Shape::new(shape);
        let expected = shape.num_elements();

        if data.len() != expected {
            return Err(TensorError::ShapeMismatch {
                expected,
                actual: data.len(),
            });
        }

        let strides = row_major_strides(&shape);

        Ok(Tensor {
            storage: TensorStorage::new(data),
            shape,
            strides,
            layout: Layout::RowMajor,
        })
    }

    /// Create a scalar tensor
    pub fn scalar(value: T) -> Self {
        Tensor {
            storage: TensorStorage::new(vec![value]),
            shape: Shape::scalar(),
            strides: vec![],
            layout: Layout::RowMajor,
        }
    }

    /// Create a 1D tensor (vector)
    pub fn from_vec(data: Vec<T>) -> Self {
        let len = data.len();
        Tensor {
            storage: TensorStorage::new(data),
            shape: Shape::vector(len),
            strides: vec![1],
            layout: Layout::RowMajor,
        }
    }

    /// Get the shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the layout
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Number of dimensions
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Total number of elements
    pub fn len(&self) -> usize {
        self.shape.num_elements()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        matches!(self.layout, Layout::RowMajor | Layout::ColMajor)
    }

    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.rank() {
            return None;
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape.dim(i) {
                return None;
            }
        }

        let physical_idx = self.physical_index(indices);
        Some(self.storage.get(physical_idx))
    }

    /// Get element at flat index (row-major order)
    pub fn get_flat(&self, flat_idx: usize) -> Option<&T> {
        if flat_idx >= self.len() {
            return None;
        }

        let indices = self.unravel_index(flat_idx);
        let physical_idx = self.physical_index(&indices);
        Some(self.storage.get(physical_idx))
    }

    /// Compute physical storage index from multi-dimensional indices
    fn physical_index(&self, indices: &[usize]) -> usize {
        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum()
    }

    /// Convert flat index to multi-dimensional indices
    fn unravel_index(&self, mut flat_idx: usize) -> Vec<usize> {
        let row_strides = row_major_strides(&self.shape);
        let mut indices = vec![0; self.rank()];

        for i in 0..self.rank() {
            indices[i] = flat_idx / row_strides[i];
            flat_idx %= row_strides[i];
        }

        indices
    }

    /// Make tensor contiguous (copy if necessary)
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            let data = self.storage.make_contiguous(&self.shape, &self.strides);
            let strides = row_major_strides(&self.shape);

            Tensor {
                storage: TensorStorage::new(data),
                shape: self.shape.clone(),
                strides,
                layout: Layout::RowMajor,
            }
        }
    }

    /// Convert to a flat vector
    pub fn to_vec(&self) -> Vec<T> {
        (0..self.len())
            .map(|i| self.get_flat(i).unwrap().clone())
            .collect()
    }

    /// Map a function over all elements
    pub fn map<F, U>(&self, f: F) -> Tensor<U>
    where
        F: Fn(&T) -> U,
        U: Clone,
    {
        let data: Vec<U> = (0..self.len())
            .map(|i| f(self.get_flat(i).unwrap()))
            .collect();

        let strides = row_major_strides(&self.shape);

        Tensor {
            storage: TensorStorage::new(data),
            shape: self.shape.clone(),
            strides,
            layout: Layout::RowMajor,
        }
    }

    /// Map a function over two tensors element-wise
    pub fn zip_with<F, U, V>(&self, other: &Tensor<U>, f: F) -> Result<Tensor<V>, TensorError>
    where
        F: Fn(&T, &U) -> V,
        U: Clone,
        V: Clone,
    {
        if self.shape != other.shape {
            return Err(TensorError::ShapeIncompatible {
                shape1: self.shape.clone(),
                shape2: other.shape.clone(),
            });
        }

        let data: Vec<V> = (0..self.len())
            .map(|i| {
                let a = self.get_flat(i).unwrap();
                let b = other.get_flat(i).unwrap();
                f(a, b)
            })
            .collect();

        let strides = row_major_strides(&self.shape);

        Ok(Tensor {
            storage: TensorStorage::new(data),
            shape: self.shape.clone(),
            strides,
            layout: Layout::RowMajor,
        })
    }

    /// Fold over all elements
    pub fn fold<A, F>(&self, init: A, f: F) -> A
    where
        F: Fn(A, &T) -> A,
    {
        let mut acc = init;
        for i in 0..self.len() {
            acc = f(acc, self.get_flat(i).unwrap());
        }
        acc
    }

    /// Reshape the tensor (must have same number of elements)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        let new_shape = Shape::new(new_shape);

        if self.shape.num_elements() != new_shape.num_elements() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.num_elements(),
                actual: new_shape.num_elements(),
            });
        }

        // For contiguous tensors, we can just change the shape and strides
        if self.is_contiguous() {
            let new_strides = row_major_strides(&new_shape);
            Ok(Tensor {
                storage: self.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                layout: Layout::RowMajor,
            })
        } else {
            // Need to make contiguous first
            let contiguous = self.contiguous();
            contiguous.reshape(new_shape.dims())
        }
    }

    /// Transpose a 2D tensor
    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }

        let new_shape = Shape::matrix(self.shape.dim(1), self.shape.dim(0));
        let new_strides = vec![self.strides[1], self.strides[0]];

        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            layout: Layout::Strided,
        })
    }

    /// Permute dimensions
    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorError> {
        if axes.len() != self.rank() {
            return Err(TensorError::RankMismatch {
                expected: self.rank(),
                actual: axes.len(),
            });
        }

        // Check that axes is a valid permutation
        let mut sorted = axes.to_vec();
        sorted.sort();
        for (i, &v) in sorted.iter().enumerate() {
            if v != i {
                return Err(TensorError::InvalidPermutation);
            }
        }

        let new_dims: Vec<usize> = axes.iter().map(|&i| self.shape.dim(i)).collect();
        let new_strides: Vec<usize> = axes.iter().map(|&i| self.strides[i]).collect();

        Ok(Tensor {
            storage: self.storage.clone(),
            shape: Shape::new(&new_dims),
            strides: new_strides,
            layout: Layout::Strided,
        })
    }

    /// Slice the tensor along all dimensions
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<Self, TensorError> {
        if ranges.len() != self.rank() {
            return Err(TensorError::RankMismatch {
                expected: self.rank(),
                actual: ranges.len(),
            });
        }

        let mut new_dims = Vec::with_capacity(self.rank());
        let mut offset = 0;

        for (i, &(start, end)) in ranges.iter().enumerate() {
            if end <= start || end > self.shape.dim(i) {
                return Err(TensorError::IndexOutOfBounds {
                    index: end,
                    size: self.shape.dim(i),
                });
            }
            new_dims.push(end - start);
            offset += start * self.strides[i];
        }

        Ok(Tensor {
            storage: TensorStorage::with_offset(
                self.storage.data.clone(),
                self.storage.offset + offset,
            ),
            shape: Shape::new(&new_dims),
            strides: self.strides.clone(),
            layout: Layout::Strided,
        })
    }

    /// Concatenate tensors along an axis
    pub fn concat(tensors: &[&Tensor<T>], axis: usize) -> Result<Self, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyConcat);
        }

        let first = tensors[0];
        let rank = first.rank();

        if axis >= rank {
            return Err(TensorError::AxisOutOfBounds { axis, rank });
        }

        // Check all tensors have compatible shapes
        for t in tensors.iter().skip(1) {
            if t.rank() != rank {
                return Err(TensorError::RankMismatch {
                    expected: rank,
                    actual: t.rank(),
                });
            }
            for i in 0..rank {
                if i != axis && t.shape.dim(i) != first.shape.dim(i) {
                    return Err(TensorError::ShapeIncompatible {
                        shape1: first.shape.clone(),
                        shape2: t.shape.clone(),
                    });
                }
            }
        }

        // Compute new shape
        let concat_dim: usize = tensors.iter().map(|t| t.shape.dim(axis)).sum();
        let mut new_dims = first.shape.dims().to_vec();
        new_dims[axis] = concat_dim;

        // Collect data
        let new_shape = Shape::new(&new_dims);
        let n = new_shape.num_elements();
        let mut data = Vec::with_capacity(n);

        // Simple case: concatenating along first axis
        if axis == 0 {
            for t in tensors {
                data.extend(t.to_vec());
            }
        } else {
            // General case: need to interleave
            let outer_size: usize = (0..axis).map(|i| first.shape.dim(i)).product();
            let inner_size: usize = (axis + 1..rank).map(|i| first.shape.dim(i)).product();

            for outer_idx in 0..outer_size {
                for t in tensors {
                    let t_axis_size = t.shape.dim(axis);
                    for axis_idx in 0..t_axis_size {
                        for inner_idx in 0..inner_size {
                            let mut indices = vec![0; rank];
                            let mut remaining = outer_idx;
                            for i in (0..axis).rev() {
                                let dim_size: usize =
                                    (0..i).map(|j| first.shape.dim(j)).product::<usize>().max(1);
                                indices[i] = remaining / dim_size;
                                remaining %= dim_size;
                            }
                            indices[axis] = axis_idx;
                            remaining = inner_idx;
                            for i in ((axis + 1)..rank).rev() {
                                let dim_size: usize = ((axis + 1)..i)
                                    .map(|j| first.shape.dim(j))
                                    .product::<usize>()
                                    .max(1);
                                indices[i] = remaining / dim_size;
                                remaining %= dim_size;
                            }
                            data.push(t.get(&indices).unwrap().clone());
                        }
                    }
                }
            }
        }

        Tensor::from_data(data, &new_dims)
    }

    /// Broadcast tensor to a new shape
    pub fn broadcast_to(&self, target_shape: &Shape) -> Result<Self, TensorError> {
        if !self.shape.is_broadcast_compatible(target_shape) {
            return Err(TensorError::BroadcastIncompatible {
                shape1: self.shape.clone(),
                shape2: target_shape.clone(),
            });
        }

        let n = target_shape.num_elements();
        let mut data = Vec::with_capacity(n);

        let target_strides = row_major_strides(target_shape);

        for flat_idx in 0..n {
            let mut remaining = flat_idx;
            let mut indices = vec![0; target_shape.rank()];

            for i in 0..target_shape.rank() {
                indices[i] = remaining / target_strides[i];
                remaining %= target_strides[i];
            }

            // Map target indices to source indices (broadcast)
            let mut src_indices = vec![0; self.rank()];
            let rank_diff = target_shape.rank() - self.rank();

            for i in 0..self.rank() {
                let target_idx = indices[i + rank_diff];
                src_indices[i] = if self.shape.dim(i) == 1 {
                    0
                } else {
                    target_idx
                };
            }

            let src_physical_idx = self.physical_index(&src_indices);
            data.push(self.storage.get(src_physical_idx).clone());
        }

        Tensor::from_data(data, target_shape.dims())
    }
}

// Numeric operations
impl<T: Clone + Default + std::ops::Add<Output = T>> Tensor<T> {
    /// Sum all elements
    pub fn sum(&self) -> T {
        self.fold(T::default(), |acc, x| acc + x.clone())
    }
}

impl<T: Clone + Default + std::ops::Mul<Output = T>> Tensor<T> {
    /// Product of all elements
    pub fn product(&self) -> T
    where
        T: From<i32>,
    {
        self.fold(T::from(1), |acc, x| acc * x.clone())
    }
}

impl<T: Clone + PartialOrd> Tensor<T> {
    /// Minimum element
    pub fn min(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let first = self.get_flat(0).unwrap().clone();
        Some(self.fold(first, |acc, x| {
            if x < &acc {
                x.clone()
            } else {
                acc
            }
        }))
    }

    /// Maximum element
    pub fn max(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let first = self.get_flat(0).unwrap().clone();
        Some(self.fold(first, |acc, x| {
            if x > &acc {
                x.clone()
            } else {
                acc
            }
        }))
    }
}

// Float-specific operations
impl Tensor<f64> {
    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        let shape_obj = Shape::new(shape);
        let n = shape_obj.num_elements();
        let data = vec![0.0; n];
        let strides = row_major_strides(&shape_obj);

        Tensor {
            storage: TensorStorage::new(data),
            shape: shape_obj,
            strides,
            layout: Layout::RowMajor,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        let shape_obj = Shape::new(shape);
        let n = shape_obj.num_elements();
        let data = vec![1.0; n];
        let strides = row_major_strides(&shape_obj);

        Tensor {
            storage: TensorStorage::new(data),
            shape: shape_obj,
            strides,
            layout: Layout::RowMajor,
        }
    }

    /// Create a tensor filled with a specific value
    pub fn full(shape: &[usize], value: f64) -> Self {
        let shape_obj = Shape::new(shape);
        let n = shape_obj.num_elements();
        let data = vec![value; n];
        let strides = row_major_strides(&shape_obj);

        Tensor {
            storage: TensorStorage::new(data),
            shape: shape_obj,
            strides,
            layout: Layout::RowMajor,
        }
    }

    /// Create an identity matrix
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }

        Tensor {
            storage: TensorStorage::new(data),
            shape: Shape::matrix(n, n),
            strides: vec![n, 1],
            layout: Layout::RowMajor,
        }
    }

    /// Create a tensor with values from start to end (exclusive)
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        let n = ((end - start) / step).ceil() as usize;
        let data: Vec<f64> = (0..n).map(|i| start + (i as f64) * step).collect();

        Tensor {
            storage: TensorStorage::new(data),
            shape: Shape::vector(n),
            strides: vec![1],
            layout: Layout::RowMajor,
        }
    }

    /// Create a tensor with n evenly spaced values from start to end
    pub fn linspace(start: f64, end: f64, n: usize) -> Self {
        if n == 0 {
            return Tensor::from_vec(vec![]);
        }
        if n == 1 {
            return Tensor::from_vec(vec![start]);
        }

        let step = (end - start) / (n - 1) as f64;
        let data: Vec<f64> = (0..n).map(|i| start + (i as f64) * step).collect();

        Tensor {
            storage: TensorStorage::new(data),
            shape: Shape::vector(n),
            strides: vec![1],
            layout: Layout::RowMajor,
        }
    }

    /// Mean of all elements
    pub fn mean(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.sum() / (self.len() as f64)
    }

    /// Variance of all elements
    pub fn var(&self) -> f64 {
        if self.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let sum_sq = self.fold(0.0, |acc, x| acc + (x - mean).powi(2));
        sum_sq / ((self.len() - 1) as f64)
    }

    /// Standard deviation
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// L2 norm (Frobenius norm for matrices)
    pub fn norm(&self) -> f64 {
        self.fold(0.0, |acc, x| acc + x * x).sqrt()
    }

    /// L1 norm
    pub fn norm_l1(&self) -> f64 {
        self.fold(0.0, |acc, x| acc + x.abs())
    }

    /// Infinity norm
    pub fn norm_inf(&self) -> f64 {
        self.fold(0.0, |acc: f64, x| acc.max(x.abs()))
    }

    /// Dot product of two vectors
    pub fn dot(&self, other: &Tensor<f64>) -> Result<f64, TensorError> {
        if self.rank() != 1 || other.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: self.rank().max(other.rank()),
            });
        }

        if self.shape != other.shape {
            return Err(TensorError::ShapeIncompatible {
                shape1: self.shape.clone(),
                shape2: other.shape.clone(),
            });
        }

        let result = (0..self.len())
            .map(|i| self.get_flat(i).unwrap() * other.get_flat(i).unwrap())
            .sum();

        Ok(result)
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        if self.rank() != 2 || other.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank().max(other.rank()),
            });
        }

        let m = self.shape.dim(0);
        let k = self.shape.dim(1);
        let k2 = other.shape.dim(0);
        let n = other.shape.dim(1);

        if k != k2 {
            return Err(TensorError::DimensionMismatch {
                dim1: k,
                dim2: k2,
            });
        }

        let mut data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.get(&[i, l]).unwrap() * other.get(&[l, j]).unwrap();
                }
                data[i * n + j] = sum;
            }
        }

        Tensor::from_data(data, &[m, n])
    }

    /// Outer product of two vectors
    pub fn outer(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        if self.rank() != 1 || other.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: self.rank().max(other.rank()),
            });
        }

        let m = self.len();
        let n = other.len();
        let mut data = Vec::with_capacity(m * n);

        for i in 0..m {
            for j in 0..n {
                data.push(self.get_flat(i).unwrap() * other.get_flat(j).unwrap());
            }
        }

        Tensor::from_data(data, &[m, n])
    }

    /// Matrix-vector multiplication
    pub fn matvec(&self, vec: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        if self.rank() != 2 || vec.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }

        let m = self.shape.dim(0);
        let n = self.shape.dim(1);

        if n != vec.len() {
            return Err(TensorError::DimensionMismatch {
                dim1: n,
                dim2: vec.len(),
            });
        }

        let mut data = Vec::with_capacity(m);

        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += self.get(&[i, j]).unwrap() * vec.get_flat(j).unwrap();
            }
            data.push(sum);
        }

        Ok(Tensor::from_vec(data))
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        self.zip_with(other, |a, b| a + b)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        self.zip_with(other, |a, b| a - b)
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        self.zip_with(other, |a, b| a * b)
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        self.zip_with(other, |a, b| a / b)
    }

    /// Scalar addition
    pub fn add_scalar(&self, scalar: f64) -> Tensor<f64> {
        self.map(|x| x + scalar)
    }

    /// Scalar subtraction
    pub fn sub_scalar(&self, scalar: f64) -> Tensor<f64> {
        self.map(|x| x - scalar)
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f64) -> Tensor<f64> {
        self.map(|x| x * scalar)
    }

    /// Scalar division
    pub fn div_scalar(&self, scalar: f64) -> Tensor<f64> {
        self.map(|x| x / scalar)
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Tensor<f64> {
        self.map(|x| x.abs())
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Tensor<f64> {
        self.map(|x| x.sqrt())
    }

    /// Element-wise square
    pub fn square(&self) -> Tensor<f64> {
        self.map(|x| x * x)
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Tensor<f64> {
        self.map(|x| x.exp())
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> Tensor<f64> {
        self.map(|x| x.ln())
    }

    /// Element-wise sine
    pub fn sin(&self) -> Tensor<f64> {
        self.map(|x| x.sin())
    }

    /// Element-wise cosine
    pub fn cos(&self) -> Tensor<f64> {
        self.map(|x| x.cos())
    }

    /// Element-wise tangent
    pub fn tan(&self) -> Tensor<f64> {
        self.map(|x| x.tan())
    }

    /// Element-wise power
    pub fn pow(&self, n: f64) -> Tensor<f64> {
        self.map(|x| x.powf(n))
    }

    /// Sum along an axis
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor<f64>, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        let mut new_dims: Vec<usize> = self.shape.dims().to_vec();
        new_dims.remove(axis);

        if new_dims.is_empty() {
            return Ok(Tensor::scalar(self.sum()));
        }

        let new_shape = Shape::new(&new_dims);
        let n = new_shape.num_elements();
        let mut data = vec![0.0; n];

        let axis_size = self.shape.dim(axis);
        let outer_size: usize = (0..axis).map(|i| self.shape.dim(i)).product::<usize>().max(1);
        let inner_size: usize = (axis + 1..self.rank())
            .map(|i| self.shape.dim(i))
            .product::<usize>()
            .max(1);

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum = 0.0;
                for axis_idx in 0..axis_size {
                    let flat_idx = outer * axis_size * inner_size + axis_idx * inner_size + inner;
                    sum += *self.get_flat(flat_idx).unwrap();
                }
                data[outer * inner_size + inner] = sum;
            }
        }

        Tensor::from_data(data, &new_dims)
    }

    /// Mean along an axis
    pub fn mean_axis(&self, axis: usize) -> Result<Tensor<f64>, TensorError> {
        let sum = self.sum_axis(axis)?;
        let axis_size = self.shape.dim(axis) as f64;
        Ok(sum.div_scalar(axis_size))
    }
}

// f32 versions of numeric operations
impl Tensor<f32> {
    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        let shape_obj = Shape::new(shape);
        let n = shape_obj.num_elements();
        let data = vec![0.0f32; n];
        let strides = row_major_strides(&shape_obj);

        Tensor {
            storage: TensorStorage::new(data),
            shape: shape_obj,
            strides,
            layout: Layout::RowMajor,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        let shape_obj = Shape::new(shape);
        let n = shape_obj.num_elements();
        let data = vec![1.0f32; n];
        let strides = row_major_strides(&shape_obj);

        Tensor {
            storage: TensorStorage::new(data),
            shape: shape_obj,
            strides,
            layout: Layout::RowMajor,
        }
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        self.sum() / (self.len() as f32)
    }

    /// L2 norm
    pub fn norm(&self) -> f32 {
        self.fold(0.0f32, |acc, x| acc + x * x).sqrt()
    }

    /// Dot product
    pub fn dot(&self, other: &Tensor<f32>) -> Result<f32, TensorError> {
        if self.rank() != 1 || other.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: self.rank().max(other.rank()),
            });
        }

        if self.shape != other.shape {
            return Err(TensorError::ShapeIncompatible {
                shape1: self.shape.clone(),
                shape2: other.shape.clone(),
            });
        }

        let result = (0..self.len())
            .map(|i| self.get_flat(i).unwrap() * other.get_flat(i).unwrap())
            .sum();

        Ok(result)
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        if self.rank() != 2 || other.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank().max(other.rank()),
            });
        }

        let m = self.shape.dim(0);
        let k = self.shape.dim(1);
        let k2 = other.shape.dim(0);
        let n = other.shape.dim(1);

        if k != k2 {
            return Err(TensorError::DimensionMismatch {
                dim1: k,
                dim2: k2,
            });
        }

        let mut data = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += self.get(&[i, l]).unwrap() * other.get(&[l, j]).unwrap();
                }
                data[i * n + j] = sum;
            }
        }

        Tensor::from_data(data, &[m, n])
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Tensor<f32> {
        self.map(|x| x * scalar)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        self.zip_with(other, |a, b| a + b)
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Tensor operation errors
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum TensorError {
    /// Shape mismatch between expected and actual element counts
    ShapeMismatch {
        /// Expected number of elements
        expected: usize,
        /// Actual number of elements
        actual: usize,
    },

    /// Shapes are incompatible for the operation
    ShapeIncompatible {
        /// First shape
        shape1: Shape,
        /// Second shape
        shape2: Shape,
    },

    /// Rank mismatch
    RankMismatch {
        /// Expected rank
        expected: usize,
        /// Actual rank
        actual: usize,
    },

    /// Dimension mismatch
    DimensionMismatch {
        /// First dimension
        dim1: usize,
        /// Second dimension
        dim2: usize,
    },

    /// Index out of bounds
    IndexOutOfBounds {
        /// Index that was out of bounds
        index: usize,
        /// Size of the dimension
        size: usize,
    },

    /// Axis out of bounds
    AxisOutOfBounds {
        /// Axis that was out of bounds
        axis: usize,
        /// Rank of the tensor
        rank: usize,
    },

    /// Invalid permutation
    InvalidPermutation,

    /// Empty tensor list for concatenation
    EmptyConcat,

    /// Shapes incompatible for broadcasting
    BroadcastIncompatible {
        /// First shape
        shape1: Shape,
        /// Second shape
        shape2: Shape,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {} elements, got {}",
                    expected, actual
                )
            }
            TensorError::ShapeIncompatible { shape1, shape2 } => {
                write!(f, "Incompatible shapes: {} and {}", shape1, shape2)
            }
            TensorError::RankMismatch { expected, actual } => {
                write!(f, "Rank mismatch: expected {}, got {}", expected, actual)
            }
            TensorError::DimensionMismatch { dim1, dim2 } => {
                write!(f, "Dimension mismatch: {} vs {}", dim1, dim2)
            }
            TensorError::IndexOutOfBounds { index, size } => {
                write!(f, "Index {} out of bounds for dimension of size {}", index, size)
            }
            TensorError::AxisOutOfBounds { axis, rank } => {
                write!(f, "Axis {} out of bounds for tensor of rank {}", axis, rank)
            }
            TensorError::InvalidPermutation => write!(f, "Invalid permutation"),
            TensorError::EmptyConcat => write!(f, "Cannot concatenate empty tensor list"),
            TensorError::BroadcastIncompatible { shape1, shape2 } => {
                write!(f, "Shapes {} and {} are not broadcast-compatible", shape1, shape2)
            }
        }
    }
}

impl std::error::Error for TensorError {}

// ============================================================================
// Display
// ============================================================================

impl<T: Clone + fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.rank() {
            0 => write!(f, "{}", self.get_flat(0).unwrap()),
            1 => {
                write!(f, "[")?;
                for i in 0..self.len() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.get_flat(i).unwrap())?;
                }
                write!(f, "]")
            }
            2 => {
                writeln!(f, "[")?;
                for i in 0..self.shape.dim(0) {
                    write!(f, "  [")?;
                    for j in 0..self.shape.dim(1) {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", self.get(&[i, j]).unwrap())?;
                    }
                    writeln!(f, "]")?;
                }
                write!(f, "]")
            }
            _ => {
                write!(f, "Tensor(shape={}, data=[", self.shape)?;
                let max_show = 10;
                for i in 0..self.len().min(max_show) {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.get_flat(i).unwrap())?;
                }
                if self.len() > max_show {
                    write!(f, ", ...")?;
                }
                write!(f, "])")
            }
        }
    }
}

impl<T: Clone + fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("layout", &self.layout)
            .field("len", &self.len())
            .finish()
    }
}

// ============================================================================
// FFI Exports
// ============================================================================

/// Create a tensor from raw data (FFI)
#[no_mangle]
pub extern "C" fn bhc_tensor_from_f64(
    data: *const f64,
    data_len: usize,
    shape: *const usize,
    shape_len: usize,
) -> *mut Tensor<f64> {
    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len) };
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };

    match Tensor::from_data(data_slice.to_vec(), shape_slice) {
        Ok(tensor) => Box::into_raw(Box::new(tensor)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a tensor (FFI)
#[no_mangle]
pub extern "C" fn bhc_tensor_free_f64(tensor: *mut Tensor<f64>) {
    if !tensor.is_null() {
        unsafe {
            drop(Box::from_raw(tensor));
        }
    }
}

/// Get tensor sum (FFI)
#[no_mangle]
pub extern "C" fn bhc_tensor_sum_f64(tensor: *const Tensor<f64>) -> f64 {
    if tensor.is_null() {
        return 0.0;
    }
    unsafe { (*tensor).sum() }
}

/// Get tensor mean (FFI)
#[no_mangle]
pub extern "C" fn bhc_tensor_mean_f64(tensor: *const Tensor<f64>) -> f64 {
    if tensor.is_null() {
        return 0.0;
    }
    unsafe { (*tensor).mean() }
}

/// Get tensor norm (FFI)
#[no_mangle]
pub extern "C" fn bhc_tensor_norm_f64(tensor: *const Tensor<f64>) -> f64 {
    if tensor.is_null() {
        return 0.0;
    }
    unsafe { (*tensor).norm() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let s = Shape::new(&[2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.dim(0), 2);
        assert_eq!(s.dim(1), 3);
        assert_eq!(s.dim(2), 4);
        assert_eq!(s.num_elements(), 24);
    }

    #[test]
    fn test_shape_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.num_elements(), 1);
    }

    #[test]
    fn test_row_major_strides() {
        let s = Shape::new(&[2, 3, 4]);
        let strides = row_major_strides(&s);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_tensor_from_data() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.len(), 6);
    }

    #[test]
    fn test_tensor_get() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(*t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*t.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(*t.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(*t.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::<f64>::zeros(&[2, 3]);
        assert_eq!(t.sum(), 0.0);
        assert_eq!(t.len(), 6);
    }

    #[test]
    fn test_tensor_ones() {
        let t = Tensor::<f64>::ones(&[2, 3]);
        assert_eq!(t.sum(), 6.0);
    }

    #[test]
    fn test_tensor_eye() {
        let t = Tensor::eye(3);
        assert_eq!(t.sum(), 3.0);
        assert_eq!(*t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*t.get(&[1, 1]).unwrap(), 1.0);
        assert_eq!(*t.get(&[0, 1]).unwrap(), 0.0);
    }

    #[test]
    fn test_tensor_arange() {
        let t = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(t.len(), 5);
        assert_eq!(*t.get_flat(0).unwrap(), 0.0);
        assert_eq!(*t.get_flat(4).unwrap(), 4.0);
    }

    #[test]
    fn test_tensor_linspace() {
        let t = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(t.len(), 5);
        assert_eq!(*t.get_flat(0).unwrap(), 0.0);
        assert_eq!(*t.get_flat(4).unwrap(), 1.0);
        assert!((t.get_flat(2).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_map() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let t2 = t.map(|x| x * 2.0);
        assert_eq!(t2.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_tensor_zip_with() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_data(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let c = a.zip_with(&b, |x, y| x + y).unwrap();
        assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_fold() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let sum = t.fold(0.0, |acc, x| acc + x);
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn test_tensor_sum() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.sum(), 10.0);
    }

    #[test]
    fn test_tensor_mean() {
        let t: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        assert_eq!(t.mean(), 2.5);
    }

    #[test]
    fn test_tensor_min_max() {
        let t = Tensor::from_data(vec![3.0, 1.0, 4.0, 1.0, 5.0], &[5]).unwrap();
        assert_eq!(t.min(), Some(1.0));
        assert_eq!(t.max(), Some(5.0));
    }

    #[test]
    fn test_tensor_norm() {
        let t: Tensor<f64> = Tensor::from_data(vec![3.0, 4.0], &[2]).unwrap();
        assert_eq!(t.norm(), 5.0);
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t2 = t.reshape(&[3, 2]).unwrap();
        assert_eq!(t2.shape().dims(), &[3, 2]);
        assert_eq!(t2.len(), 6);
    }

    #[test]
    fn test_tensor_transpose() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t2 = t.transpose().unwrap();
        assert_eq!(t2.shape().dims(), &[3, 2]);
        assert_eq!(*t2.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*t2.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(*t2.get(&[1, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensor_slice() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t2 = t.slice(&[(0, 2), (1, 3)]).unwrap();
        assert_eq!(t2.shape().dims(), &[2, 2]);
        assert_eq!(*t2.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*t2.get(&[0, 1]).unwrap(), 3.0);
    }

    #[test]
    fn test_tensor_dot() {
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let dot = a.dot(&b).unwrap();
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_tensor_matmul() {
        // 2x3 * 3x2 = 2x2
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape().dims(), &[2, 2]);
        // [1,2,3] . [7,9,11] = 7+18+33 = 58
        // [1,2,3] . [8,10,12] = 8+20+36 = 64
        // [4,5,6] . [7,9,11] = 28+45+66 = 139
        // [4,5,6] . [8,10,12] = 32+50+72 = 154
        assert_eq!(*c.get(&[0, 0]).unwrap(), 58.0);
        assert_eq!(*c.get(&[0, 1]).unwrap(), 64.0);
        assert_eq!(*c.get(&[1, 0]).unwrap(), 139.0);
        assert_eq!(*c.get(&[1, 1]).unwrap(), 154.0);
    }

    #[test]
    fn test_tensor_outer() {
        let a = Tensor::from_data(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_data(vec![3.0, 4.0, 5.0], &[3]).unwrap();
        let c = a.outer(&b).unwrap();

        assert_eq!(c.shape().dims(), &[2, 3]);
        assert_eq!(*c.get(&[0, 0]).unwrap(), 3.0);
        assert_eq!(*c.get(&[0, 2]).unwrap(), 5.0);
        assert_eq!(*c.get(&[1, 0]).unwrap(), 6.0);
        assert_eq!(*c.get(&[1, 2]).unwrap(), 10.0);
    }

    #[test]
    fn test_tensor_matvec() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let v = Tensor::from_data(vec![1.0, 1.0, 1.0], &[3]).unwrap();
        let result = a.matvec(&v).unwrap();

        assert_eq!(result.shape().dims(), &[2]);
        assert_eq!(*result.get_flat(0).unwrap(), 6.0); // 1+2+3
        assert_eq!(*result.get_flat(1).unwrap(), 15.0); // 4+5+6
    }

    #[test]
    fn test_tensor_elementwise_ops() {
        let a: Tensor<f64> = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b: Tensor<f64> = Tensor::from_data(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        let add = a.add(&b).unwrap();
        assert_eq!(add.to_vec(), vec![5.0, 7.0, 9.0]);

        let sub = a.sub(&b).unwrap();
        assert_eq!(sub.to_vec(), vec![-3.0, -3.0, -3.0]);

        let mul = a.mul(&b).unwrap();
        assert_eq!(mul.to_vec(), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_tensor_scalar_ops() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let add = a.add_scalar(10.0);
        assert_eq!(add.to_vec(), vec![11.0, 12.0, 13.0]);

        let mul = a.mul_scalar(2.0);
        assert_eq!(mul.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_tensor_math_ops() {
        let a = Tensor::from_data(vec![1.0, 4.0, 9.0], &[3]).unwrap();
        let sqrt = a.sqrt();
        assert_eq!(sqrt.to_vec(), vec![1.0, 2.0, 3.0]);

        let b = Tensor::from_data(vec![-1.0, 2.0, -3.0], &[3]).unwrap();
        let abs = b.abs();
        assert_eq!(abs.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tensor_sum_axis() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let sum0 = t.sum_axis(0).unwrap();
        assert_eq!(sum0.shape().dims(), &[3]);
        assert_eq!(sum0.to_vec(), vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]

        let sum1 = t.sum_axis(1).unwrap();
        assert_eq!(sum1.shape().dims(), &[2]);
        assert_eq!(sum1.to_vec(), vec![6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_tensor_concat() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_data(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let c = Tensor::concat(&[&a, &b], 0).unwrap();

        assert_eq!(c.shape().dims(), &[6]);
        assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_broadcast_compatible() {
        let s1 = Shape::new(&[3, 1]);
        let s2 = Shape::new(&[1, 4]);
        assert!(s1.is_broadcast_compatible(&s2));

        let s3 = s1.broadcast_shape(&s2).unwrap();
        assert_eq!(s3.dims(), &[3, 4]);
    }

    #[test]
    fn test_tensor_var_std() {
        let t: Tensor<f64> = Tensor::from_data(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8]).unwrap();
        let mean = t.mean();
        assert_eq!(mean, 5.0);

        let var = t.var();
        assert!((var - 4.571428571428571).abs() < 1e-10);

        let std = t.std();
        assert!((std - 2.138089935299395).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_display() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let s = format!("{}", t);
        assert_eq!(s, "[1, 2, 3]");
    }

    #[test]
    fn test_f32_tensor() {
        let t = Tensor::<f32>::zeros(&[2, 3]);
        assert_eq!(t.sum(), 0.0f32);

        let ones = Tensor::<f32>::ones(&[2, 2]);
        assert_eq!(ones.sum(), 4.0f32);
    }
}
