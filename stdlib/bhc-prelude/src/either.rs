//! Either type and operations
//!
//! The `Either` type represents values with two possibilities: a value of
//! type `Either a b` is either `Left a` or `Right b`.
//!
//! The `Either` type is sometimes used to represent a value which is either
//! correct or an error; by convention, the `Left` constructor is used to
//! hold an error value and the `Right` constructor is used to hold a
//! correct value (mnemonic: "right" also means "correct").

use crate::bool::Bool;

/// The Either type
///
/// Represents a value of one of two possible types (a disjoint union).
#[repr(C)]
pub enum Either<L, R> {
    /// Contains the left value
    Left(L),
    /// Contains the right value
    Right(R),
}

impl<L, R> Either<L, R> {
    /// Returns `true` if this is a `Left` value.
    #[inline]
    pub const fn is_left(&self) -> bool {
        matches!(self, Either::Left(_))
    }

    /// Returns `true` if this is a `Right` value.
    #[inline]
    pub const fn is_right(&self) -> bool {
        matches!(self, Either::Right(_))
    }

    /// Convert the left side of `Either<L, R>` to `Option<L>`.
    #[inline]
    pub fn left(self) -> Option<L> {
        match self {
            Either::Left(l) => Some(l),
            Either::Right(_) => None,
        }
    }

    /// Convert the right side of `Either<L, R>` to `Option<R>`.
    #[inline]
    pub fn right(self) -> Option<R> {
        match self {
            Either::Left(_) => None,
            Either::Right(r) => Some(r),
        }
    }

    /// Maps an `Either<L, R>` to `Either<L, R2>` by applying a function
    /// to a contained `Right` value, leaving a `Left` value untouched.
    #[inline]
    pub fn map<R2, F>(self, f: F) -> Either<L, R2>
    where
        F: FnOnce(R) -> R2,
    {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => Either::Right(f(r)),
        }
    }

    /// Maps an `Either<L, R>` to `Either<L2, R>` by applying a function
    /// to a contained `Left` value, leaving a `Right` value untouched.
    #[inline]
    pub fn map_left<L2, F>(self, f: F) -> Either<L2, R>
    where
        F: FnOnce(L) -> L2,
    {
        match self {
            Either::Left(l) => Either::Left(f(l)),
            Either::Right(r) => Either::Right(r),
        }
    }

    /// Apply one of two functions depending on contents.
    #[inline]
    pub fn either<T, F, G>(self, f: F, g: G) -> T
    where
        F: FnOnce(L) -> T,
        G: FnOnce(R) -> T,
    {
        match self {
            Either::Left(l) => f(l),
            Either::Right(r) => g(r),
        }
    }

    /// Returns the left value, or a default.
    #[inline]
    pub fn left_or(self, default: L) -> L {
        match self {
            Either::Left(l) => l,
            Either::Right(_) => default,
        }
    }

    /// Returns the right value, or a default.
    #[inline]
    pub fn right_or(self, default: R) -> R {
        match self {
            Either::Left(_) => default,
            Either::Right(r) => r,
        }
    }

    /// Converts from `&Either<L, R>` to `Either<&L, &R>`.
    #[inline]
    pub const fn as_ref(&self) -> Either<&L, &R> {
        match *self {
            Either::Left(ref l) => Either::Left(l),
            Either::Right(ref r) => Either::Right(r),
        }
    }

    /// Converts from `&mut Either<L, R>` to `Either<&mut L, &mut R>`.
    #[inline]
    pub fn as_mut(&mut self) -> Either<&mut L, &mut R> {
        match *self {
            Either::Left(ref mut l) => Either::Left(l),
            Either::Right(ref mut r) => Either::Right(r),
        }
    }
}

impl<L: Clone, R: Clone> Clone for Either<L, R> {
    fn clone(&self) -> Self {
        match self {
            Either::Left(l) => Either::Left(l.clone()),
            Either::Right(r) => Either::Right(r.clone()),
        }
    }
}

impl<L: Copy, R: Copy> Copy for Either<L, R> {}

impl<L: PartialEq, R: PartialEq> PartialEq for Either<L, R> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Either::Left(a), Either::Left(b)) => a == b,
            (Either::Right(a), Either::Right(b)) => a == b,
            _ => false,
        }
    }
}

impl<L: Eq, R: Eq> Eq for Either<L, R> {}

impl<L, R> From<Result<R, L>> for Either<L, R> {
    fn from(result: Result<R, L>) -> Self {
        match result {
            Ok(r) => Either::Right(r),
            Err(l) => Either::Left(l),
        }
    }
}

impl<L, R> From<Either<L, R>> for Result<R, L> {
    fn from(either: Either<L, R>) -> Self {
        match either {
            Either::Left(l) => Err(l),
            Either::Right(r) => Ok(r),
        }
    }
}

// FFI exports for Either<i64, i64> as a common case

/// Check if Either is Left
#[no_mangle]
pub extern "C" fn bhc_either_is_left_i64(e: &Either<i64, i64>) -> Bool {
    Bool::from_bool(e.is_left())
}

/// Check if Either is Right
#[no_mangle]
pub extern "C" fn bhc_either_is_right_i64(e: &Either<i64, i64>) -> Bool {
    Bool::from_bool(e.is_right())
}

/// Get left value, returns default for Right
#[no_mangle]
pub extern "C" fn bhc_either_from_left_i64(e: &Either<i64, i64>, default: i64) -> i64 {
    e.as_ref().left().copied().unwrap_or(default)
}

/// Get right value, returns default for Left
#[no_mangle]
pub extern "C" fn bhc_either_from_right_i64(e: &Either<i64, i64>, default: i64) -> i64 {
    e.as_ref().right().copied().unwrap_or(default)
}

/// Create a Left value
#[no_mangle]
pub extern "C" fn bhc_either_left_i64(x: i64) -> Either<i64, i64> {
    Either::Left(x)
}

/// Create a Right value
#[no_mangle]
pub extern "C" fn bhc_either_right_i64(x: i64) -> Either<i64, i64> {
    Either::Right(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_left_right() {
        let l: Either<i32, i32> = Either::Left(42);
        let r: Either<i32, i32> = Either::Right(42);

        assert!(l.is_left());
        assert!(!l.is_right());
        assert!(!r.is_left());
        assert!(r.is_right());
    }

    #[test]
    fn test_map() {
        let r: Either<i32, i32> = Either::Right(2);
        assert_eq!(r.map(|n| n * 2), Either::Right(4));

        let l: Either<i32, i32> = Either::Left(2);
        assert_eq!(l.map(|n| n * 2), Either::Left(2));
    }

    #[test]
    fn test_either() {
        let l: Either<i32, i32> = Either::Left(2);
        let r: Either<i32, i32> = Either::Right(3);

        assert_eq!(l.either(|x| x * 2, |x| x * 3), 4);
        assert_eq!(r.either(|x| x * 2, |x| x * 3), 9);
    }
}
