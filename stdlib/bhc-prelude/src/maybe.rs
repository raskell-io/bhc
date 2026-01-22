//! Maybe type and operations
//!
//! The `Maybe` type represents optional values. A value of type `Maybe a`
//! either contains a value of type `a` (represented as `Just a`), or it
//! is empty (represented as `Nothing`).

use crate::bool::Bool;

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
}
