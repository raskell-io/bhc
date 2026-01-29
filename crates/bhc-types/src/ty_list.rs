//! Type-level lists for tensor shapes.
//!
//! This module implements type-level lists used to represent tensor shapes
//! in shape-indexed tensor types per H26-SPEC Section 7.
//!
//! ## Overview
//!
//! Type-level lists are used with promoted syntax:
//!
//! ```text
//! Tensor '[1024, 768] Float    -- 2D tensor with shape [1024, 768]
//! Tensor '[n, m] Float         -- Polymorphic 2D tensor
//! Tensor '[] Float             -- Scalar (0-rank tensor)
//! ```
//!
//! ## Representation
//!
//! - `Nil` - Empty list `'[]`
//! - `Cons(head, tail)` - List construction `head ': tail`
//! - `Var(v)` - Polymorphic shape variable
//! - `Append(xs, ys)` - List concatenation `xs ++ ys`

use serde::{Deserialize, Serialize};

use crate::{nat::TyNat, Ty, TyVar};

/// A type-level list, used primarily for tensor shapes.
///
/// Shapes are represented as lists of type-level naturals, enabling
/// compile-time dimension checking.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TyList {
    /// Empty list `'[]`.
    ///
    /// Represents a scalar (0-rank tensor).
    Nil,

    /// List construction `head ': tail`.
    ///
    /// The head is a type (usually `Ty::Nat` for shapes),
    /// and the tail is another `TyList`.
    Cons(Box<Ty>, Box<TyList>),

    /// A polymorphic shape variable.
    ///
    /// Example: `shape` in `Tensor shape Float`
    Var(TyVar),

    /// List concatenation `xs ++ ys`.
    ///
    /// Used for shape computations like broadcasting.
    Append(Box<TyList>, Box<TyList>),
}

impl TyList {
    /// Creates an empty type-level list.
    #[must_use]
    pub fn nil() -> Self {
        Self::Nil
    }

    /// Creates a cons cell with the given head and tail.
    #[must_use]
    pub fn cons(head: Ty, tail: TyList) -> Self {
        Self::Cons(Box::new(head), Box::new(tail))
    }

    /// Creates a type-level list from a slice of types.
    #[must_use]
    pub fn from_vec(elements: Vec<Ty>) -> Self {
        elements
            .into_iter()
            .rev()
            .fold(Self::Nil, |acc, elem| Self::cons(elem, acc))
    }

    /// Creates a shape (list of naturals) from dimension values.
    #[must_use]
    pub fn shape_from_dims(dims: &[u64]) -> Self {
        Self::from_vec(dims.iter().map(|&d| Ty::Nat(TyNat::lit(d))).collect())
    }

    /// Creates a shape with natural variables.
    #[must_use]
    pub fn shape_from_nat_vars(vars: Vec<TyNat>) -> Self {
        Self::from_vec(vars.into_iter().map(Ty::Nat).collect())
    }

    /// Appends two type-level lists.
    #[must_use]
    pub fn append(xs: TyList, ys: TyList) -> Self {
        // Simplify if xs is Nil
        match xs {
            TyList::Nil => ys,
            _ => Self::Append(Box::new(xs), Box::new(ys)),
        }
    }

    /// Returns true if this is an empty list.
    #[must_use]
    pub fn is_nil(&self) -> bool {
        matches!(self, Self::Nil)
    }

    /// Returns true if this list contains no variables.
    #[must_use]
    pub fn is_ground(&self) -> bool {
        match self {
            Self::Nil => true,
            Self::Cons(head, tail) => head.is_ground() && tail.is_ground(),
            Self::Var(_) => false,
            Self::Append(xs, ys) => xs.is_ground() && ys.is_ground(),
        }
    }

    /// Returns the length of this list if it can be statically determined.
    #[must_use]
    pub fn static_len(&self) -> Option<usize> {
        match self {
            Self::Nil => Some(0),
            Self::Cons(_, tail) => tail.static_len().map(|n| n + 1),
            Self::Var(_) => None,
            Self::Append(xs, ys) => Some(xs.static_len()? + ys.static_len()?),
        }
    }

    /// Converts this type-level list to a vector of types if it's a concrete list.
    ///
    /// Returns `None` if the list contains variables or append operations
    /// that can't be resolved.
    #[must_use]
    pub fn to_vec(&self) -> Option<Vec<Ty>> {
        match self {
            Self::Nil => Some(Vec::new()),
            Self::Cons(head, tail) => {
                let mut result = vec![(**head).clone()];
                result.extend(tail.to_vec()?);
                Some(result)
            }
            Self::Var(_) => None,
            Self::Append(xs, ys) => {
                let mut result = xs.to_vec()?;
                result.extend(ys.to_vec()?);
                Some(result)
            }
        }
    }

    /// Extracts concrete dimension values from a shape.
    ///
    /// Returns `None` if any dimension is not a concrete natural literal.
    #[must_use]
    pub fn to_static_dims(&self) -> Option<Vec<u64>> {
        let tys = self.to_vec()?;
        tys.into_iter()
            .map(|ty| match ty {
                Ty::Nat(TyNat::Lit(n)) => Some(n),
                _ => None,
            })
            .collect()
    }

    /// Collects all type variables occurring in this list.
    #[must_use]
    pub fn free_vars(&self) -> Vec<TyVar> {
        let mut vars = Vec::new();
        self.collect_free_vars(&mut vars);
        vars
    }

    fn collect_free_vars(&self, vars: &mut Vec<TyVar>) {
        match self {
            Self::Nil => {}
            Self::Cons(head, tail) => {
                head.collect_free_vars(vars);
                tail.collect_free_vars(vars);
            }
            Self::Var(v) => {
                if !vars.contains(v) {
                    vars.push(v.clone());
                }
            }
            Self::Append(xs, ys) => {
                xs.collect_free_vars(vars);
                ys.collect_free_vars(vars);
            }
        }
    }
}

impl std::fmt::Display for TyList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nil => write!(f, "'[]"),
            Self::Cons(_, _) => {
                write!(f, "'[")?;
                let mut current = self;
                let mut first = true;
                while let Self::Cons(head, tail) = current {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{head}")?;
                    first = false;
                    current = tail;
                }
                write!(f, "]")
            }
            Self::Var(v) => write!(f, "shape{}", v.id),
            Self::Append(xs, ys) => write!(f, "({xs} ++ {ys})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Kind;

    #[test]
    fn test_nil() {
        let nil = TyList::nil();
        assert!(nil.is_nil());
        assert!(nil.is_ground());
        assert_eq!(nil.static_len(), Some(0));
        assert_eq!(nil.to_vec(), Some(Vec::new()));
    }

    #[test]
    fn test_cons() {
        let list = TyList::cons(
            Ty::Nat(TyNat::lit(3)),
            TyList::cons(Ty::Nat(TyNat::lit(4)), TyList::nil()),
        );
        assert!(!list.is_nil());
        assert!(list.is_ground());
        assert_eq!(list.static_len(), Some(2));
    }

    #[test]
    fn test_from_vec() {
        let list = TyList::from_vec(vec![
            Ty::Nat(TyNat::lit(1)),
            Ty::Nat(TyNat::lit(2)),
            Ty::Nat(TyNat::lit(3)),
        ]);
        assert_eq!(list.static_len(), Some(3));
    }

    #[test]
    fn test_shape_from_dims() {
        let shape = TyList::shape_from_dims(&[1024, 768]);
        assert!(shape.is_ground());
        assert_eq!(shape.static_len(), Some(2));
        assert_eq!(shape.to_static_dims(), Some(vec![1024, 768]));
    }

    #[test]
    fn test_variable_shape() {
        let v = TyVar::new(0, Kind::List(Box::new(Kind::Nat)));
        let shape = TyList::Var(v.clone());
        assert!(!shape.is_ground());
        assert_eq!(shape.static_len(), None);
        assert_eq!(shape.to_vec(), None);
    }

    #[test]
    fn test_append() {
        let xs = TyList::shape_from_dims(&[1, 2]);
        let ys = TyList::shape_from_dims(&[3, 4]);
        let appended = TyList::append(xs, ys);
        // Append preserves structure (doesn't evaluate)
        match appended {
            TyList::Append(_, _) => {}
            _ => panic!("Expected Append variant"),
        }
    }

    #[test]
    fn test_append_nil_simplifies() {
        let xs = TyList::nil();
        let ys = TyList::shape_from_dims(&[1, 2]);
        let result = TyList::append(xs, ys.clone());
        // Appending nil should simplify
        assert_eq!(result, ys);
    }

    #[test]
    fn test_free_vars() {
        let v = TyVar::new(0, Kind::Nat);
        let shape = TyList::cons(Ty::Nat(TyNat::Var(v.clone())), TyList::nil());
        let vars = shape.free_vars();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains(&v));
    }

    #[test]
    fn test_display() {
        let nil = TyList::nil();
        assert_eq!(format!("{nil}"), "'[]");

        let shape = TyList::shape_from_dims(&[1024, 768]);
        assert_eq!(format!("{shape}"), "'[1024, 768]");
    }
}
