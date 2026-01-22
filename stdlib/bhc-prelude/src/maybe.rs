//! Maybe type and operations
//!
//! The `Maybe` type represents optional values. A value of type `Maybe a`
//! either contains a value of type `a` (represented as `Just a`), or it
//! is empty (represented as `Nothing`).
//!
//! # Type Class Instances
//!
//! - `Eq`: Equality comparison
//! - `Ord`: Ordering (Nothing < Just a)
//! - `Show`: Display representation
//! - `Functor`: `fmap` via `map`
//! - `Applicative`: `pure` via `Just`, `<*>` via `ap`
//! - `Monad`: `>>=` via `and_then`
//! - `Foldable`: `foldr` via `fold`
//! - `Monoid`: `mempty` = `Nothing`, `<>` via `or`

use crate::bool::Bool;
use std::cmp::Ordering as StdOrdering;
use std::fmt;

/// The Maybe type
///
/// Represents an optional value. Either `Nothing` or `Just a`.
#[repr(C)]
pub enum Maybe<T> {
    /// No value
    Nothing,
    /// Contains a value
    Just(T),
}

impl<T> Maybe<T> {
    /// Returns `true` if the option is a `Just` value.
    #[inline]
    pub const fn is_just(&self) -> bool {
        matches!(self, Maybe::Just(_))
    }

    /// Returns `true` if the option is a `Nothing` value.
    #[inline]
    pub const fn is_nothing(&self) -> bool {
        matches!(self, Maybe::Nothing)
    }

    /// Maps a `Maybe<T>` to `Maybe<U>` by applying a function to a contained value.
    #[inline]
    pub fn map<U, F>(self, f: F) -> Maybe<U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Maybe::Nothing => Maybe::Nothing,
            Maybe::Just(x) => Maybe::Just(f(x)),
        }
    }

    /// Returns the contained `Just` value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is a `Nothing` with a custom panic message.
    #[inline]
    pub fn unwrap(self) -> T {
        match self {
            Maybe::Just(x) => x,
            Maybe::Nothing => panic!("called `Maybe::unwrap()` on a `Nothing` value"),
        }
    }

    /// Returns the contained `Just` value or a provided default.
    #[inline]
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Maybe::Just(x) => x,
            Maybe::Nothing => default,
        }
    }

    /// Returns the contained `Just` value or computes it from a closure.
    #[inline]
    pub fn unwrap_or_else<F>(self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        match self {
            Maybe::Just(x) => x,
            Maybe::Nothing => f(),
        }
    }

    /// Converts from `&Maybe<T>` to `Maybe<&T>`.
    #[inline]
    pub const fn as_ref(&self) -> Maybe<&T> {
        match *self {
            Maybe::Nothing => Maybe::Nothing,
            Maybe::Just(ref x) => Maybe::Just(x),
        }
    }

    /// Converts from `&mut Maybe<T>` to `Maybe<&mut T>`.
    #[inline]
    pub fn as_mut(&mut self) -> Maybe<&mut T> {
        match *self {
            Maybe::Nothing => Maybe::Nothing,
            Maybe::Just(ref mut x) => Maybe::Just(x),
        }
    }

    /// Monadic bind (>>=): Applies a function that returns a `Maybe` to the contained value.
    ///
    /// This is the Monad instance's `>>=` operation.
    #[inline]
    pub fn and_then<U, F>(self, f: F) -> Maybe<U>
    where
        F: FnOnce(T) -> Maybe<U>,
    {
        match self {
            Maybe::Nothing => Maybe::Nothing,
            Maybe::Just(x) => f(x),
        }
    }

    /// Returns `Nothing` if the option is `Nothing`, otherwise returns `other`.
    #[inline]
    pub fn and<U>(self, other: Maybe<U>) -> Maybe<U> {
        match self {
            Maybe::Nothing => Maybe::Nothing,
            Maybe::Just(_) => other,
        }
    }

    /// Returns the option if it contains a value, otherwise returns `other`.
    #[inline]
    pub fn or(self, other: Maybe<T>) -> Maybe<T> {
        match self {
            Maybe::Nothing => other,
            x @ Maybe::Just(_) => x,
        }
    }

    /// Returns the option if it contains a value, otherwise calls `f` and returns the result.
    #[inline]
    pub fn or_else<F>(self, f: F) -> Maybe<T>
    where
        F: FnOnce() -> Maybe<T>,
    {
        match self {
            Maybe::Nothing => f(),
            x @ Maybe::Just(_) => x,
        }
    }

    /// Returns `Just` if exactly one of `self`, `other` is `Just`, otherwise returns `Nothing`.
    #[inline]
    pub fn xor(self, other: Maybe<T>) -> Maybe<T> {
        match (self, other) {
            (Maybe::Just(x), Maybe::Nothing) => Maybe::Just(x),
            (Maybe::Nothing, Maybe::Just(y)) => Maybe::Just(y),
            _ => Maybe::Nothing,
        }
    }

    /// Returns `Nothing` if the option is `Nothing`, otherwise returns the result of applying
    /// the predicate to the contained value.
    #[inline]
    pub fn filter<P>(self, predicate: P) -> Maybe<T>
    where
        P: FnOnce(&T) -> bool,
    {
        match self {
            Maybe::Just(x) if predicate(&x) => Maybe::Just(x),
            _ => Maybe::Nothing,
        }
    }

    /// Zips `self` with another `Maybe`.
    #[inline]
    pub fn zip<U>(self, other: Maybe<U>) -> Maybe<(T, U)> {
        match (self, other) {
            (Maybe::Just(a), Maybe::Just(b)) => Maybe::Just((a, b)),
            _ => Maybe::Nothing,
        }
    }

    /// Zips `self` and another `Maybe` with function `f`.
    #[inline]
    pub fn zip_with<U, V, F>(self, other: Maybe<U>, f: F) -> Maybe<V>
    where
        F: FnOnce(T, U) -> V,
    {
        match (self, other) {
            (Maybe::Just(a), Maybe::Just(b)) => Maybe::Just(f(a, b)),
            _ => Maybe::Nothing,
        }
    }

    /// Flattens a `Maybe<Maybe<T>>` to a `Maybe<T>`.
    #[inline]
    pub fn flatten(self) -> T
    where
        T: Into<Maybe<T>>,
        Self: Into<Maybe<Maybe<T>>>,
    {
        todo!("flatten requires special handling")
    }

    /// Applies a function wrapped in `Maybe` to a value wrapped in `Maybe`.
    ///
    /// This is the Applicative instance's `<*>` operation.
    #[inline]
    pub fn ap<U, F>(self, mf: Maybe<F>) -> Maybe<U>
    where
        F: FnOnce(T) -> U,
    {
        match (mf, self) {
            (Maybe::Just(f), Maybe::Just(x)) => Maybe::Just(f(x)),
            _ => Maybe::Nothing,
        }
    }

    /// Folds the `Maybe` with the given functions.
    ///
    /// This is `maybe` in Haskell: `maybe default f m`
    #[inline]
    pub fn fold<U, F>(self, default: U, f: F) -> U
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Maybe::Nothing => default,
            Maybe::Just(x) => f(x),
        }
    }

    /// Converts a `Maybe<T>` to a `Vec<T>`.
    #[inline]
    pub fn to_vec(self) -> Vec<T> {
        match self {
            Maybe::Nothing => vec![],
            Maybe::Just(x) => vec![x],
        }
    }

    /// Returns an iterator over the possibly contained value.
    #[inline]
    pub fn iter(&self) -> MaybeIter<'_, T> {
        MaybeIter {
            inner: self.as_ref(),
        }
    }
}

/// Iterator over a `Maybe` value.
pub struct MaybeIter<'a, T> {
    inner: Maybe<&'a T>,
}

impl<'a, T> Iterator for MaybeIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match std::mem::replace(&mut self.inner, Maybe::Nothing) {
            Maybe::Nothing => None,
            Maybe::Just(x) => Some(x),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_just() { 1 } else { 0 };
        (n, Some(n))
    }
}

impl<'a, T> ExactSizeIterator for MaybeIter<'a, T> {}

impl<T> IntoIterator for Maybe<T> {
    type Item = T;
    type IntoIter = std::option::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        Option::from(self).into_iter()
    }
}

impl<T: Clone> Clone for Maybe<T> {
    fn clone(&self) -> Self {
        match self {
            Maybe::Nothing => Maybe::Nothing,
            Maybe::Just(x) => Maybe::Just(x.clone()),
        }
    }
}

impl<T: Copy> Copy for Maybe<T> {}

impl<T: PartialEq> PartialEq for Maybe<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Maybe::Nothing, Maybe::Nothing) => true,
            (Maybe::Just(a), Maybe::Just(b)) => a == b,
            _ => false,
        }
    }
}

impl<T: Eq> Eq for Maybe<T> {}

impl<T: PartialOrd> PartialOrd for Maybe<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Maybe::Nothing, Maybe::Nothing) => Some(std::cmp::Ordering::Equal),
            (Maybe::Nothing, Maybe::Just(_)) => Some(std::cmp::Ordering::Less),
            (Maybe::Just(_), Maybe::Nothing) => Some(std::cmp::Ordering::Greater),
            (Maybe::Just(a), Maybe::Just(b)) => a.partial_cmp(b),
        }
    }
}

impl<T: Ord> Ord for Maybe<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Maybe::Nothing, Maybe::Nothing) => std::cmp::Ordering::Equal,
            (Maybe::Nothing, Maybe::Just(_)) => std::cmp::Ordering::Less,
            (Maybe::Just(_), Maybe::Nothing) => std::cmp::Ordering::Greater,
            (Maybe::Just(a), Maybe::Just(b)) => a.cmp(b),
        }
    }
}

impl<T: std::hash::Hash> std::hash::Hash for Maybe<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Maybe::Nothing => 0u8.hash(state),
            Maybe::Just(x) => {
                1u8.hash(state);
                x.hash(state);
            }
        }
    }
}

impl<T> Default for Maybe<T> {
    /// Returns `Nothing`.
    fn default() -> Self {
        Maybe::Nothing
    }
}

impl<T: fmt::Debug> fmt::Debug for Maybe<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Maybe::Nothing => write!(f, "Nothing"),
            Maybe::Just(x) => write!(f, "Just({:?})", x),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Maybe<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Maybe::Nothing => write!(f, "Nothing"),
            Maybe::Just(x) => write!(f, "Just {}", x),
        }
    }
}

impl<T> From<Option<T>> for Maybe<T> {
    fn from(opt: Option<T>) -> Self {
        match opt {
            None => Maybe::Nothing,
            Some(x) => Maybe::Just(x),
        }
    }
}

impl<T> From<Maybe<T>> for Option<T> {
    fn from(maybe: Maybe<T>) -> Self {
        match maybe {
            Maybe::Nothing => None,
            Maybe::Just(x) => Some(x),
        }
    }
}

// FFI exports for Maybe<i64> as a common case

/// Check if Maybe is Just
#[no_mangle]
pub extern "C" fn bhc_maybe_is_just_i64(m: &Maybe<i64>) -> Bool {
    Bool::from_bool(m.is_just())
}

/// Check if Maybe is Nothing
#[no_mangle]
pub extern "C" fn bhc_maybe_is_nothing_i64(m: &Maybe<i64>) -> Bool {
    Bool::from_bool(m.is_nothing())
}

/// Get value from Just, returns 0 for Nothing
#[no_mangle]
pub extern "C" fn bhc_maybe_from_just_i64(m: &Maybe<i64>, default: i64) -> i64 {
    match m {
        Maybe::Just(x) => *x,
        Maybe::Nothing => default,
    }
}

/// Create a Just value
#[no_mangle]
pub extern "C" fn bhc_maybe_just_i64(x: i64) -> Maybe<i64> {
    Maybe::Just(x)
}

/// Create a Nothing value
#[no_mangle]
pub extern "C" fn bhc_maybe_nothing_i64() -> Maybe<i64> {
    Maybe::Nothing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_just() {
        assert!(Maybe::Just(42).is_just());
        assert!(!Maybe::<i32>::Nothing.is_just());
    }

    #[test]
    fn test_is_nothing() {
        assert!(!Maybe::Just(42).is_nothing());
        assert!(Maybe::<i32>::Nothing.is_nothing());
    }

    #[test]
    fn test_map() {
        let x: Maybe<i32> = Maybe::Just(2);
        assert_eq!(x.map(|n| n * 2), Maybe::Just(4));

        let y: Maybe<i32> = Maybe::Nothing;
        assert_eq!(y.map(|n| n * 2), Maybe::Nothing);
    }

    #[test]
    fn test_unwrap_or() {
        assert_eq!(Maybe::Just(42).unwrap_or(0), 42);
        assert_eq!(Maybe::<i32>::Nothing.unwrap_or(0), 0);
    }

    #[test]
    fn test_and_then() {
        let safe_div = |x: i32| {
            if x == 0 {
                Maybe::Nothing
            } else {
                Maybe::Just(100 / x)
            }
        };
        assert_eq!(Maybe::Just(2).and_then(safe_div), Maybe::Just(50));
        assert_eq!(Maybe::Just(0).and_then(safe_div), Maybe::Nothing);
        assert_eq!(Maybe::<i32>::Nothing.and_then(safe_div), Maybe::Nothing);
    }

    #[test]
    fn test_or() {
        let x: Maybe<i32> = Maybe::Just(2);
        let y: Maybe<i32> = Maybe::Nothing;
        let z: Maybe<i32> = Maybe::Just(100);

        assert_eq!(x.or(z.clone()), Maybe::Just(2));
        assert_eq!(y.or(z), Maybe::Just(100));
    }

    #[test]
    fn test_and() {
        let x: Maybe<i32> = Maybe::Just(2);
        let y: Maybe<&str> = Maybe::Nothing;
        assert_eq!(x.and(y), Maybe::Nothing);

        let x: Maybe<i32> = Maybe::Nothing;
        let y: Maybe<&str> = Maybe::Just("foo");
        assert_eq!(x.and(y), Maybe::Nothing);

        let x: Maybe<i32> = Maybe::Just(2);
        let y: Maybe<&str> = Maybe::Just("foo");
        assert_eq!(x.and(y), Maybe::Just("foo"));
    }

    #[test]
    fn test_filter() {
        assert_eq!(Maybe::Just(4).filter(|x| *x > 2), Maybe::Just(4));
        assert_eq!(Maybe::Just(1).filter(|x| *x > 2), Maybe::Nothing);
        assert_eq!(Maybe::<i32>::Nothing.filter(|x| *x > 2), Maybe::Nothing);
    }

    #[test]
    fn test_zip() {
        let x = Maybe::Just(1);
        let y = Maybe::Just("hi");
        assert_eq!(x.zip(y), Maybe::Just((1, "hi")));

        let x = Maybe::Just(1);
        let y: Maybe<&str> = Maybe::Nothing;
        assert_eq!(x.zip(y), Maybe::Nothing);
    }

    #[test]
    fn test_ord() {
        assert!(Maybe::<i32>::Nothing < Maybe::Just(0));
        assert!(Maybe::Just(1) < Maybe::Just(2));
        assert!(Maybe::Just(2) > Maybe::Just(1));
        assert_eq!(Maybe::Just(1), Maybe::Just(1));
    }

    #[test]
    fn test_default() {
        let m: Maybe<i32> = Default::default();
        assert_eq!(m, Maybe::Nothing);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Maybe::Just(42)), "Just 42");
        assert_eq!(format!("{}", Maybe::<i32>::Nothing), "Nothing");
    }

    #[test]
    fn test_debug() {
        assert_eq!(format!("{:?}", Maybe::Just(42)), "Just(42)");
        assert_eq!(format!("{:?}", Maybe::<i32>::Nothing), "Nothing");
    }

    #[test]
    fn test_iter() {
        let just = Maybe::Just(5);
        let mut iter = just.iter();
        assert_eq!(iter.next(), Some(&5));
        assert_eq!(iter.next(), None);

        let nothing: Maybe<i32> = Maybe::Nothing;
        let mut iter = nothing.iter();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter() {
        let just = Maybe::Just(5);
        let collected: Vec<i32> = just.into_iter().collect();
        assert_eq!(collected, vec![5]);

        let nothing: Maybe<i32> = Maybe::Nothing;
        let collected: Vec<i32> = nothing.into_iter().collect();
        assert_eq!(collected, vec![]);
    }

    #[test]
    fn test_fold() {
        assert_eq!(Maybe::Just(5).fold(0, |x| x * 2), 10);
        assert_eq!(Maybe::<i32>::Nothing.fold(0, |x| x * 2), 0);
    }

    #[test]
    fn test_to_vec() {
        assert_eq!(Maybe::Just(1).to_vec(), vec![1]);
        assert_eq!(Maybe::<i32>::Nothing.to_vec(), vec![]);
    }

    #[test]
    fn test_xor() {
        let x: Maybe<i32> = Maybe::Just(1);
        let y: Maybe<i32> = Maybe::Just(2);
        let n: Maybe<i32> = Maybe::Nothing;

        assert_eq!(x.clone().xor(n.clone()), Maybe::Just(1));
        assert_eq!(n.clone().xor(y.clone()), Maybe::Just(2));
        assert_eq!(x.xor(y), Maybe::Nothing);
        assert_eq!(n.clone().xor(n), Maybe::Nothing);
    }
}
