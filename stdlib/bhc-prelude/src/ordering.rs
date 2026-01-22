//! Ordering type and operations
//!
//! The `Ordering` type is used to represent the result of a comparison.

use std::cmp::Ordering as StdOrdering;

/// The Ordering type
///
/// Represents the result of a comparison between two values.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ordering {
    /// Less than
    LT = -1,
    /// Equal
    EQ = 0,
    /// Greater than
    GT = 1,
}

impl Ordering {
    /// Reverses the `Ordering`.
    ///
    /// * `LT` becomes `GT`
    /// * `GT` becomes `LT`
    /// * `EQ` stays `EQ`
    #[inline]
    pub const fn reverse(self) -> Self {
        match self {
            Ordering::LT => Ordering::GT,
            Ordering::EQ => Ordering::EQ,
            Ordering::GT => Ordering::LT,
        }
    }

    /// Chains two orderings.
    ///
    /// Returns `self` when it's not `EQ`, otherwise returns `other`.
    #[inline]
    pub const fn then(self, other: Self) -> Self {
        match self {
            Ordering::EQ => other,
            _ => self,
        }
    }

    /// Chains with a closure producing an ordering.
    #[inline]
    pub fn then_with<F>(self, f: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            Ordering::EQ => f(),
            _ => self,
        }
    }

    /// Returns `true` if the ordering is `LT`.
    #[inline]
    pub const fn is_lt(self) -> bool {
        matches!(self, Ordering::LT)
    }

    /// Returns `true` if the ordering is `EQ`.
    #[inline]
    pub const fn is_eq(self) -> bool {
        matches!(self, Ordering::EQ)
    }

    /// Returns `true` if the ordering is `GT`.
    #[inline]
    pub const fn is_gt(self) -> bool {
        matches!(self, Ordering::GT)
    }

    /// Returns `true` if the ordering is `LT` or `EQ`.
    #[inline]
    pub const fn is_le(self) -> bool {
        !self.is_gt()
    }

    /// Returns `true` if the ordering is `GT` or `EQ`.
    #[inline]
    pub const fn is_ge(self) -> bool {
        !self.is_lt()
    }

    /// Returns `true` if the ordering is `LT` or `GT`.
    #[inline]
    pub const fn is_ne(self) -> bool {
        !self.is_eq()
    }
}

impl From<StdOrdering> for Ordering {
    #[inline]
    fn from(ord: StdOrdering) -> Self {
        match ord {
            StdOrdering::Less => Ordering::LT,
            StdOrdering::Equal => Ordering::EQ,
            StdOrdering::Greater => Ordering::GT,
        }
    }
}

impl From<Ordering> for StdOrdering {
    #[inline]
    fn from(ord: Ordering) -> Self {
        match ord {
            Ordering::LT => StdOrdering::Less,
            Ordering::EQ => StdOrdering::Equal,
            Ordering::GT => StdOrdering::Greater,
        }
    }
}

// FFI exports

/// Compare two integers
#[no_mangle]
pub extern "C" fn bhc_compare_i64(a: i64, b: i64) -> Ordering {
    a.cmp(&b).into()
}

/// Compare two floats
#[no_mangle]
pub extern "C" fn bhc_compare_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).map(Into::into).unwrap_or(Ordering::EQ)
}

/// Reverse an ordering
#[no_mangle]
pub extern "C" fn bhc_ordering_reverse(ord: Ordering) -> Ordering {
    ord.reverse()
}

/// Chain two orderings
#[no_mangle]
pub extern "C" fn bhc_ordering_then(a: Ordering, b: Ordering) -> Ordering {
    a.then(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse() {
        assert_eq!(Ordering::LT.reverse(), Ordering::GT);
        assert_eq!(Ordering::EQ.reverse(), Ordering::EQ);
        assert_eq!(Ordering::GT.reverse(), Ordering::LT);
    }

    #[test]
    fn test_then() {
        assert_eq!(Ordering::LT.then(Ordering::GT), Ordering::LT);
        assert_eq!(Ordering::EQ.then(Ordering::GT), Ordering::GT);
        assert_eq!(Ordering::GT.then(Ordering::LT), Ordering::GT);
    }

    #[test]
    fn test_compare() {
        assert_eq!(bhc_compare_i64(1, 2), Ordering::LT);
        assert_eq!(bhc_compare_i64(2, 2), Ordering::EQ);
        assert_eq!(bhc_compare_i64(3, 2), Ordering::GT);
    }
}
