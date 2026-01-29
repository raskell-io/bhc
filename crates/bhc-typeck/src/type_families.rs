//! Type families for shape computations.
//!
//! This module implements type-level functions (type families) for tensor
//! shape computations, enabling compile-time verification of tensor operations.
//!
//! ## Type Families
//!
//! ### MatMulShape
//!
//! Computes the result shape of matrix multiplication:
//!
//! ```text
//! MatMulShape '[m, k] '[k, n] = '[m, n]
//! ```
//!
//! ### Broadcast
//!
//! Computes the result shape of broadcasting (NumPy-style):
//!
//! ```text
//! Broadcast '[1, 3] '[2, 3] = '[2, 3]
//! Broadcast '[3] '[2, 3] = '[2, 3]
//! ```
//!
//! ### Transpose
//!
//! Reverses the shape for transposition:
//!
//! ```text
//! Transpose '[m, n] = '[n, m]
//! ```
//!
//! ## Usage
//!
//! These type families are evaluated during type checking to verify that
//! tensor operations have compatible shapes.

use bhc_types::{nat::TyNat, ty_list::TyList, Ty};

/// Result of reducing a type family application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReductionResult {
    /// The type family was successfully reduced to a concrete result.
    Reduced(TyList),
    /// The type family could not be reduced (e.g., polymorphic shapes).
    Stuck,
    /// The type family application is invalid (shape mismatch).
    Error(ShapeError),
}

/// Errors that can occur during shape computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// Inner dimensions don't match for matrix multiplication.
    MatMulDimensionMismatch {
        /// The inner dimension of the left matrix (number of columns).
        left_inner: TyNat,
        /// The inner dimension of the right matrix (number of rows).
        right_inner: TyNat,
    },
    /// Matrix multiplication requires exactly 2D tensors.
    MatMulRankMismatch {
        /// The rank (number of dimensions) of the left tensor.
        left_rank: usize,
        /// The rank (number of dimensions) of the right tensor.
        right_rank: usize,
    },
    /// Shapes are not compatible for broadcasting.
    BroadcastIncompatible {
        /// The first dimension that couldn't be broadcast.
        dim1: TyNat,
        /// The second dimension that couldn't be broadcast.
        dim2: TyNat,
    },
    /// Transpose requires exactly 2D tensors.
    TransposeRankMismatch {
        /// The actual rank of the tensor.
        rank: usize,
    },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::MatMulDimensionMismatch {
                left_inner,
                right_inner,
            } => {
                write!(
                    f,
                    "matrix multiplication dimension mismatch: inner dimensions {} and {} don't match",
                    left_inner, right_inner
                )
            }
            ShapeError::MatMulRankMismatch {
                left_rank,
                right_rank,
            } => {
                write!(
                    f,
                    "matrix multiplication requires 2D tensors, got {}D and {}D",
                    left_rank, right_rank
                )
            }
            ShapeError::BroadcastIncompatible { dim1, dim2 } => {
                write!(
                    f,
                    "dimensions {} and {} are not compatible for broadcasting",
                    dim1, dim2
                )
            }
            ShapeError::TransposeRankMismatch { rank } => {
                write!(f, "transpose requires 2D tensor, got {}D", rank)
            }
        }
    }
}

/// Computes the result shape of matrix multiplication.
///
/// For matrices A (shape [m, k]) and B (shape [k, n]), the result has shape [m, n].
///
/// # Arguments
///
/// * `left` - The shape of the left matrix
/// * `right` - The shape of the right matrix
///
/// # Returns
///
/// The result shape if shapes are compatible, or an error if not.
///
/// # Example
///
/// ```ignore
/// let a_shape = TyList::shape_from_dims(&[1024, 768]);
/// let b_shape = TyList::shape_from_dims(&[768, 512]);
/// let result = reduce_matmul_shape(&a_shape, &b_shape);
/// // result is Reduced('[1024, 512])
/// ```
pub fn reduce_matmul_shape(left: &TyList, right: &TyList) -> ReductionResult {
    // Extract dimensions from shapes
    let left_dims = match left.to_vec() {
        Some(dims) => dims,
        None => return ReductionResult::Stuck, // Polymorphic shape
    };

    let right_dims = match right.to_vec() {
        Some(dims) => dims,
        None => return ReductionResult::Stuck, // Polymorphic shape
    };

    // Matrix multiplication requires 2D tensors
    if left_dims.len() != 2 {
        return ReductionResult::Error(ShapeError::MatMulRankMismatch {
            left_rank: left_dims.len(),
            right_rank: right_dims.len(),
        });
    }

    if right_dims.len() != 2 {
        return ReductionResult::Error(ShapeError::MatMulRankMismatch {
            left_rank: left_dims.len(),
            right_rank: right_dims.len(),
        });
    }

    // Extract dimensions: left = [m, k], right = [k', n]
    let m = &left_dims[0];
    let k1 = &left_dims[1];
    let k2 = &right_dims[0];
    let n = &right_dims[1];

    // Check if inner dimensions match
    match (extract_nat(k1), extract_nat(k2)) {
        (Some(n1), Some(n2)) => {
            if !nats_equal(&n1, &n2) {
                return ReductionResult::Error(ShapeError::MatMulDimensionMismatch {
                    left_inner: n1,
                    right_inner: n2,
                });
            }
        }
        _ => return ReductionResult::Stuck, // Can't compare polymorphic dimensions
    }

    // Result shape is [m, n]
    ReductionResult::Reduced(TyList::from_vec(vec![m.clone(), n.clone()]))
}

/// Computes the result shape of broadcasting two shapes.
///
/// Broadcasting follows NumPy-style rules:
/// - Shapes are aligned from the right
/// - Dimensions must be equal, or one must be 1
/// - Missing dimensions are treated as 1
///
/// # Example
///
/// ```ignore
/// let a = TyList::shape_from_dims(&[1, 3]);
/// let b = TyList::shape_from_dims(&[2, 3]);
/// let result = reduce_broadcast(&a, &b);
/// // result is Reduced('[2, 3])
/// ```
pub fn reduce_broadcast(left: &TyList, right: &TyList) -> ReductionResult {
    let left_dims = match left.to_vec() {
        Some(dims) => dims,
        None => return ReductionResult::Stuck,
    };

    let right_dims = match right.to_vec() {
        Some(dims) => dims,
        None => return ReductionResult::Stuck,
    };

    let max_len = std::cmp::max(left_dims.len(), right_dims.len());
    let mut result = Vec::with_capacity(max_len);

    // Pad the shorter shape with implicit 1s
    let left_padded: Vec<_> = std::iter::repeat(Ty::Nat(TyNat::Lit(1)))
        .take(max_len - left_dims.len())
        .chain(left_dims.into_iter())
        .collect();

    let right_padded: Vec<_> = std::iter::repeat(Ty::Nat(TyNat::Lit(1)))
        .take(max_len - right_dims.len())
        .chain(right_dims.into_iter())
        .collect();

    // Compare dimensions pairwise
    for (l, r) in left_padded.into_iter().zip(right_padded.into_iter()) {
        match broadcast_dims(&l, &r) {
            Some(dim) => result.push(dim),
            None => {
                let ln = extract_nat(&l).unwrap_or(TyNat::Lit(0));
                let rn = extract_nat(&r).unwrap_or(TyNat::Lit(0));
                return ReductionResult::Error(ShapeError::BroadcastIncompatible {
                    dim1: ln,
                    dim2: rn,
                });
            }
        }
    }

    ReductionResult::Reduced(TyList::from_vec(result))
}

/// Computes the result shape of transposing a 2D tensor.
///
/// # Example
///
/// ```ignore
/// let shape = TyList::shape_from_dims(&[3, 4]);
/// let result = reduce_transpose(&shape);
/// // result is Reduced('[4, 3])
/// ```
pub fn reduce_transpose(shape: &TyList) -> ReductionResult {
    let dims = match shape.to_vec() {
        Some(dims) => dims,
        None => return ReductionResult::Stuck,
    };

    if dims.len() != 2 {
        return ReductionResult::Error(ShapeError::TransposeRankMismatch { rank: dims.len() });
    }

    ReductionResult::Reduced(TyList::from_vec(vec![dims[1].clone(), dims[0].clone()]))
}

/// Computes the result shape of concatenating tensors along an axis.
///
/// All dimensions except the concatenation axis must match.
///
/// # Arguments
///
/// * `left` - Shape of the first tensor
/// * `right` - Shape of the second tensor
/// * `axis` - The axis along which to concatenate
pub fn reduce_concat(left: &TyList, right: &TyList, axis: usize) -> ReductionResult {
    let left_dims = match left.to_vec() {
        Some(dims) => dims,
        None => return ReductionResult::Stuck,
    };

    let right_dims = match right.to_vec() {
        Some(dims) => dims,
        None => return ReductionResult::Stuck,
    };

    if left_dims.len() != right_dims.len() {
        return ReductionResult::Stuck; // Different ranks
    }

    if axis >= left_dims.len() {
        return ReductionResult::Stuck; // Invalid axis
    }

    let mut result = Vec::with_capacity(left_dims.len());

    for (i, (l, r)) in left_dims.iter().zip(right_dims.iter()).enumerate() {
        if i == axis {
            // Sum the dimensions along the concat axis
            match (extract_nat(l), extract_nat(r)) {
                (Some(TyNat::Lit(n1)), Some(TyNat::Lit(n2))) => {
                    result.push(Ty::Nat(TyNat::Lit(n1 + n2)));
                }
                (Some(n1), Some(n2)) => {
                    result.push(Ty::Nat(TyNat::Add(Box::new(n1), Box::new(n2))));
                }
                _ => return ReductionResult::Stuck,
            }
        } else {
            // Dimensions must match
            match (extract_nat(l), extract_nat(r)) {
                (Some(n1), Some(n2)) if nats_equal(&n1, &n2) => {
                    result.push(l.clone());
                }
                (Some(n1), Some(n2)) => {
                    return ReductionResult::Error(ShapeError::BroadcastIncompatible {
                        dim1: n1,
                        dim2: n2,
                    });
                }
                _ => return ReductionResult::Stuck,
            }
        }
    }

    ReductionResult::Reduced(TyList::from_vec(result))
}

/// Extracts a TyNat from a Ty, if it's a nat type.
fn extract_nat(ty: &Ty) -> Option<TyNat> {
    match ty {
        Ty::Nat(n) => Some(n.clone()),
        _ => None,
    }
}

/// Checks if two TyNat values are equal.
fn nats_equal(a: &TyNat, b: &TyNat) -> bool {
    match (a, b) {
        (TyNat::Lit(n1), TyNat::Lit(n2)) => n1 == n2,
        (TyNat::Var(v1), TyNat::Var(v2)) => v1 == v2,
        // TODO: Handle Add and Mul with algebraic simplification
        _ => false,
    }
}

/// Broadcasts two dimensions, returning the result dimension.
///
/// Returns None if the dimensions are incompatible for broadcasting.
fn broadcast_dims(left: &Ty, right: &Ty) -> Option<Ty> {
    match (extract_nat(left), extract_nat(right)) {
        (Some(TyNat::Lit(1)), Some(_)) => Some(right.clone()),
        (Some(_), Some(TyNat::Lit(1))) => Some(left.clone()),
        (Some(TyNat::Lit(n1)), Some(TyNat::Lit(n2))) if n1 == n2 => Some(left.clone()),
        (Some(TyNat::Var(v1)), Some(TyNat::Var(v2))) if v1 == v2 => Some(left.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_types::TyVar;

    #[test]
    fn test_matmul_valid() {
        // [1024, 768] x [768, 512] = [1024, 512]
        let a = TyList::shape_from_dims(&[1024, 768]);
        let b = TyList::shape_from_dims(&[768, 512]);

        match reduce_matmul_shape(&a, &b) {
            ReductionResult::Reduced(result) => {
                assert_eq!(result.to_static_dims(), Some(vec![1024, 512]));
            }
            _ => panic!("expected successful reduction"),
        }
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        // [1024, 768] x [512, 256] - inner dims don't match
        let a = TyList::shape_from_dims(&[1024, 768]);
        let b = TyList::shape_from_dims(&[512, 256]);

        match reduce_matmul_shape(&a, &b) {
            ReductionResult::Error(ShapeError::MatMulDimensionMismatch { .. }) => {}
            _ => panic!("expected dimension mismatch error"),
        }
    }

    #[test]
    fn test_matmul_rank_mismatch() {
        // [1024, 768, 3] x [768, 512] - left is 3D
        let a = TyList::shape_from_dims(&[1024, 768, 3]);
        let b = TyList::shape_from_dims(&[768, 512]);

        match reduce_matmul_shape(&a, &b) {
            ReductionResult::Error(ShapeError::MatMulRankMismatch { .. }) => {}
            _ => panic!("expected rank mismatch error"),
        }
    }

    #[test]
    fn test_broadcast_same_shape() {
        let a = TyList::shape_from_dims(&[2, 3]);
        let b = TyList::shape_from_dims(&[2, 3]);

        match reduce_broadcast(&a, &b) {
            ReductionResult::Reduced(result) => {
                assert_eq!(result.to_static_dims(), Some(vec![2, 3]));
            }
            _ => panic!("expected successful broadcast"),
        }
    }

    #[test]
    fn test_broadcast_with_ones() {
        // [1, 3] broadcast with [2, 3] = [2, 3]
        let a = TyList::shape_from_dims(&[1, 3]);
        let b = TyList::shape_from_dims(&[2, 3]);

        match reduce_broadcast(&a, &b) {
            ReductionResult::Reduced(result) => {
                assert_eq!(result.to_static_dims(), Some(vec![2, 3]));
            }
            _ => panic!("expected successful broadcast"),
        }
    }

    #[test]
    fn test_broadcast_different_ranks() {
        // [3] broadcast with [2, 3] = [2, 3]
        let a = TyList::shape_from_dims(&[3]);
        let b = TyList::shape_from_dims(&[2, 3]);

        match reduce_broadcast(&a, &b) {
            ReductionResult::Reduced(result) => {
                assert_eq!(result.to_static_dims(), Some(vec![2, 3]));
            }
            _ => panic!("expected successful broadcast"),
        }
    }

    #[test]
    fn test_broadcast_incompatible() {
        // [2, 3] broadcast with [2, 4] - incompatible
        let a = TyList::shape_from_dims(&[2, 3]);
        let b = TyList::shape_from_dims(&[2, 4]);

        match reduce_broadcast(&a, &b) {
            ReductionResult::Error(ShapeError::BroadcastIncompatible { .. }) => {}
            _ => panic!("expected broadcast incompatible error"),
        }
    }

    #[test]
    fn test_transpose() {
        let shape = TyList::shape_from_dims(&[3, 4]);

        match reduce_transpose(&shape) {
            ReductionResult::Reduced(result) => {
                assert_eq!(result.to_static_dims(), Some(vec![4, 3]));
            }
            _ => panic!("expected successful transpose"),
        }
    }

    #[test]
    fn test_transpose_wrong_rank() {
        let shape = TyList::shape_from_dims(&[2, 3, 4]);

        match reduce_transpose(&shape) {
            ReductionResult::Error(ShapeError::TransposeRankMismatch { rank: 3 }) => {}
            _ => panic!("expected rank mismatch error"),
        }
    }

    #[test]
    fn test_concat_valid() {
        // [2, 3] concat [2, 4] along axis 1 = [2, 7]
        let a = TyList::shape_from_dims(&[2, 3]);
        let b = TyList::shape_from_dims(&[2, 4]);

        match reduce_concat(&a, &b, 1) {
            ReductionResult::Reduced(result) => {
                assert_eq!(result.to_static_dims(), Some(vec![2, 7]));
            }
            _ => panic!("expected successful concat"),
        }
    }

    #[test]
    fn test_concat_axis_0() {
        // [2, 3] concat [4, 3] along axis 0 = [6, 3]
        let a = TyList::shape_from_dims(&[2, 3]);
        let b = TyList::shape_from_dims(&[4, 3]);

        match reduce_concat(&a, &b, 0) {
            ReductionResult::Reduced(result) => {
                assert_eq!(result.to_static_dims(), Some(vec![6, 3]));
            }
            _ => panic!("expected successful concat"),
        }
    }

    #[test]
    fn test_polymorphic_matmul_same_var() {
        use bhc_types::Kind;

        // [m, k] x [k, n] - when k is the same variable, reduction succeeds
        let m = TyVar::new(1, Kind::Nat);
        let k = TyVar::new(2, Kind::Nat);
        let n = TyVar::new(3, Kind::Nat);

        let a = TyList::from_vec(vec![
            Ty::Nat(TyNat::Var(m.clone())),
            Ty::Nat(TyNat::Var(k.clone())),
        ]);
        let b = TyList::from_vec(vec![Ty::Nat(TyNat::Var(k)), Ty::Nat(TyNat::Var(n.clone()))]);

        // Should reduce to [m, n] since k == k
        match reduce_matmul_shape(&a, &b) {
            ReductionResult::Reduced(result) => {
                // Result should be [m, n]
                let dims = result.to_vec().unwrap();
                assert_eq!(dims.len(), 2);
                assert!(matches!(&dims[0], Ty::Nat(TyNat::Var(v)) if v.id == m.id));
                assert!(matches!(&dims[1], Ty::Nat(TyNat::Var(v)) if v.id == n.id));
            }
            r => panic!("expected successful reduction, got {:?}", r),
        }
    }

    #[test]
    fn test_polymorphic_matmul_different_vars() {
        use bhc_types::Kind;

        // [m, k1] x [k2, n] - different inner dimensions, can't determine equality
        let m = TyVar::new(1, Kind::Nat);
        let k1 = TyVar::new(2, Kind::Nat);
        let k2 = TyVar::new(4, Kind::Nat); // Different variable
        let n = TyVar::new(3, Kind::Nat);

        let a = TyList::from_vec(vec![
            Ty::Nat(TyNat::Var(m)),
            Ty::Nat(TyNat::Var(k1.clone())),
        ]);
        let b = TyList::from_vec(vec![
            Ty::Nat(TyNat::Var(k2.clone())),
            Ty::Nat(TyNat::Var(n)),
        ]);

        // This should result in a dimension mismatch error since k1 != k2
        match reduce_matmul_shape(&a, &b) {
            ReductionResult::Error(ShapeError::MatMulDimensionMismatch {
                left_inner,
                right_inner,
            }) => {
                assert!(matches!(left_inner, TyNat::Var(v) if v.id == k1.id));
                assert!(matches!(right_inner, TyNat::Var(v) if v.id == k2.id));
            }
            r => panic!("expected dimension mismatch error, got {:?}", r),
        }
    }
}
