//! Either type and operations
//!
//! The `Either` type represents values with two possibilities: a value of
//! type `Either a b` is either `Left a` or `Right b`.
//!
//! The `Either` type is sometimes used to represent a value which is either
//! correct or an error; by convention, the `Left` constructor is used to
//! hold an error value and the `Right` constructor is used to hold a
//! correct value (mnemonic: "right" also means "correct").
//!
//! # Type Class Instances
//!
//! - `Eq`: Equality comparison
//! - `Ord`: Ordering (Left < Right, then by contained value)
//! - `Show`: Display representation
//! - `Functor`: `fmap` via `map` (maps over Right)
//! - `Applicative`: `pure` via `Right`, `<*>` via `ap`
//! - `Monad`: `>>=` via `and_then` (Right-biased)
//! - `Bifunctor`: `bimap` via `map_both`

use crate::bool::Bool;
use std::fmt;

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

    /// Monadic bind (>>=): Applies a function that returns an `Either` to the `Right` value.
    ///
    /// This is the Monad instance's `>>=` operation. It only applies `f` when
    /// this is a `Right` value; `Left` values are passed through unchanged.
    #[inline]
    pub fn and_then<R2, F>(self, f: F) -> Either<L, R2>
    where
        F: FnOnce(R) -> Either<L, R2>,
    {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => f(r),
        }
    }

    /// Maps both sides of the `Either` with the provided functions.
    ///
    /// This is the Bifunctor instance's `bimap` operation.
    #[inline]
    pub fn map_both<L2, R2, F, G>(self, f: F, g: G) -> Either<L2, R2>
    where
        F: FnOnce(L) -> L2,
        G: FnOnce(R) -> R2,
    {
        match self {
            Either::Left(l) => Either::Left(f(l)),
            Either::Right(r) => Either::Right(g(r)),
        }
    }

    /// Swap the left and right values.
    #[inline]
    pub fn flip(self) -> Either<R, L> {
        match self {
            Either::Left(l) => Either::Right(l),
            Either::Right(r) => Either::Left(r),
        }
    }

    /// Returns the contained `Left` value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is a `Right` with a custom panic message.
    #[inline]
    pub fn unwrap_left(self) -> L {
        match self {
            Either::Left(l) => l,
            Either::Right(_) => panic!("called `Either::unwrap_left()` on a `Right` value"),
        }
    }

    /// Returns the contained `Right` value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is a `Left` with a custom panic message.
    #[inline]
    pub fn unwrap_right(self) -> R {
        match self {
            Either::Left(_) => panic!("called `Either::unwrap_right()` on a `Left` value"),
            Either::Right(r) => r,
        }
    }

    /// Returns the left value or computes it from a closure.
    #[inline]
    pub fn left_or_else<F>(self, f: F) -> L
    where
        F: FnOnce(R) -> L,
    {
        match self {
            Either::Left(l) => l,
            Either::Right(r) => f(r),
        }
    }

    /// Returns the right value or computes it from a closure.
    #[inline]
    pub fn right_or_else<F>(self, f: F) -> R
    where
        F: FnOnce(L) -> R,
    {
        match self {
            Either::Left(l) => f(l),
            Either::Right(r) => r,
        }
    }

    /// Applies a function wrapped in `Either` to a value wrapped in `Either`.
    ///
    /// This is the Applicative instance's `<*>` operation.
    #[inline]
    pub fn ap<R2, F>(self, ef: Either<L, F>) -> Either<L, R2>
    where
        F: FnOnce(R) -> R2,
    {
        match (ef, self) {
            (Either::Left(l), _) => Either::Left(l),
            (_, Either::Left(l)) => Either::Left(l),
            (Either::Right(f), Either::Right(x)) => Either::Right(f(x)),
        }
    }

    /// Convert `Either<L, Either<L, R>>` to `Either<L, R>`.
    #[inline]
    pub fn flatten(self) -> Either<L, R>
    where
        R: Into<Either<L, R>>,
    {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => r.into(),
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

impl<L: PartialOrd, R: PartialOrd> PartialOrd for Either<L, R> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Either::Left(a), Either::Left(b)) => a.partial_cmp(b),
            (Either::Left(_), Either::Right(_)) => Some(std::cmp::Ordering::Less),
            (Either::Right(_), Either::Left(_)) => Some(std::cmp::Ordering::Greater),
            (Either::Right(a), Either::Right(b)) => a.partial_cmp(b),
        }
    }
}

impl<L: Ord, R: Ord> Ord for Either<L, R> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Either::Left(a), Either::Left(b)) => a.cmp(b),
            (Either::Left(_), Either::Right(_)) => std::cmp::Ordering::Less,
            (Either::Right(_), Either::Left(_)) => std::cmp::Ordering::Greater,
            (Either::Right(a), Either::Right(b)) => a.cmp(b),
        }
    }
}

impl<L: std::hash::Hash, R: std::hash::Hash> std::hash::Hash for Either<L, R> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Either::Left(l) => {
                0u8.hash(state);
                l.hash(state);
            }
            Either::Right(r) => {
                1u8.hash(state);
                r.hash(state);
            }
        }
    }
}

impl<L, R: Default> Default for Either<L, R> {
    /// Returns `Right(R::default())`.
    fn default() -> Self {
        Either::Right(R::default())
    }
}

impl<L: fmt::Debug, R: fmt::Debug> fmt::Debug for Either<L, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Either::Left(l) => write!(f, "Left({:?})", l),
            Either::Right(r) => write!(f, "Right({:?})", r),
        }
    }
}

impl<L: fmt::Display, R: fmt::Display> fmt::Display for Either<L, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Either::Left(l) => write!(f, "Left {}", l),
            Either::Right(r) => write!(f, "Right {}", r),
        }
    }
}

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

    #[test]
    fn test_and_then() {
        let safe_div = |x: i32| -> Either<&'static str, i32> {
            if x == 0 {
                Either::Left("division by zero")
            } else {
                Either::Right(100 / x)
            }
        };

        let r: Either<&str, i32> = Either::Right(2);
        assert_eq!(r.and_then(safe_div), Either::Right(50));

        let r: Either<&str, i32> = Either::Right(0);
        assert_eq!(r.and_then(safe_div), Either::Left("division by zero"));

        let l: Either<&str, i32> = Either::Left("error");
        assert_eq!(l.and_then(safe_div), Either::Left("error"));
    }

    #[test]
    fn test_map_both() {
        let l: Either<i32, i32> = Either::Left(2);
        assert_eq!(l.map_both(|x| x * 2, |x| x * 3), Either::Left(4));

        let r: Either<i32, i32> = Either::Right(3);
        assert_eq!(r.map_both(|x| x * 2, |x| x * 3), Either::Right(9));
    }

    #[test]
    fn test_flip() {
        let l: Either<i32, &str> = Either::Left(42);
        let r: Either<i32, &str> = Either::Right("hello");

        assert_eq!(l.flip(), Either::Right(42));
        assert_eq!(r.flip(), Either::Left("hello"));
    }

    #[test]
    fn test_ord() {
        let l1: Either<i32, i32> = Either::Left(1);
        let l2: Either<i32, i32> = Either::Left(2);
        let r1: Either<i32, i32> = Either::Right(1);
        let r2: Either<i32, i32> = Either::Right(2);

        assert!(l1 < l2);
        assert!(l2 < r1);
        assert!(r1 < r2);
        assert!(l1 < r1);
    }

    #[test]
    fn test_default() {
        let e: Either<(), i32> = Default::default();
        assert_eq!(e, Either::Right(0));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Either::<i32, i32>::Left(42)), "Left 42");
        assert_eq!(format!("{}", Either::<i32, i32>::Right(42)), "Right 42");
    }

    #[test]
    fn test_debug() {
        assert_eq!(format!("{:?}", Either::<i32, i32>::Left(42)), "Left(42)");
        assert_eq!(format!("{:?}", Either::<i32, i32>::Right(42)), "Right(42)");
    }

    #[test]
    fn test_left_right_or() {
        let l: Either<i32, i32> = Either::Left(42);
        let r: Either<i32, i32> = Either::Right(42);

        assert_eq!(l.left_or(0), 42);
        assert_eq!(r.left_or(0), 0);
        assert_eq!(l.right_or(0), 0);
        assert_eq!(r.right_or(0), 42);
    }

    #[test]
    fn test_left_right_or_else() {
        let l: Either<i32, i32> = Either::Left(42);
        let r: Either<i32, i32> = Either::Right(42);

        assert_eq!(l.left_or_else(|x| x * 2), 42);
        assert_eq!(r.left_or_else(|x| x * 2), 84);
        assert_eq!(l.right_or_else(|x| x * 2), 84);
        assert_eq!(r.right_or_else(|x| x * 2), 42);
    }

    #[test]
    fn test_result_conversion() {
        let ok: Result<i32, &str> = Ok(42);
        let err: Result<i32, &str> = Err("error");

        let e_ok: Either<&str, i32> = ok.into();
        let e_err: Either<&str, i32> = err.into();

        assert_eq!(e_ok, Either::Right(42));
        assert_eq!(e_err, Either::Left("error"));

        let back_ok: Result<i32, &str> = e_ok.into();
        let back_err: Result<i32, &str> = e_err.into();

        assert_eq!(back_ok, Ok(42));
        assert_eq!(back_err, Err("error"));
    }
}
