//! Ordering type and operations
//!
//! The `Ordering` type is used to represent the result of a comparison.
//!
//! # Type Class Instances
//!
//! - `Eq`: Equality comparison
//! - `Ord`: LT < EQ < GT
//! - `Show`: Display representation
//! - `Semigroup`: `(<>)` via `then` (first non-EQ wins)
//! - `Monoid`: `mempty` = `EQ`
//! - `Bounded`: `minBound` = `LT`, `maxBound` = `GT`

use std::cmp::Ordering as StdOrdering;
use std::fmt;

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

impl PartialOrd for Ordering {
    fn partial_cmp(&self, other: &Self) -> Option<StdOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordering {
    fn cmp(&self, other: &Self) -> StdOrdering {
        (*self as i8).cmp(&(*other as i8))
    }
}

impl Default for Ordering {
    /// Returns `EQ` (the identity element for `then`/`<>`).
    fn default() -> Self {
        Ordering::EQ
    }
}

impl fmt::Display for Ordering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ordering::LT => write!(f, "LT"),
            Ordering::EQ => write!(f, "EQ"),
            Ordering::GT => write!(f, "GT"),
        }
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

    #[test]
    fn test_ord() {
        assert!(Ordering::LT < Ordering::EQ);
        assert!(Ordering::EQ < Ordering::GT);
        assert!(Ordering::LT < Ordering::GT);
    }

    #[test]
    fn test_default() {
        assert_eq!(Ordering::default(), Ordering::EQ);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Ordering::LT), "LT");
        assert_eq!(format!("{}", Ordering::EQ), "EQ");
        assert_eq!(format!("{}", Ordering::GT), "GT");
    }

    #[test]
    fn test_predicates() {
        assert!(Ordering::LT.is_lt());
        assert!(!Ordering::LT.is_eq());
        assert!(!Ordering::LT.is_gt());
        assert!(Ordering::LT.is_le());
        assert!(!Ordering::LT.is_ge());
        assert!(Ordering::LT.is_ne());

        assert!(!Ordering::EQ.is_lt());
        assert!(Ordering::EQ.is_eq());
        assert!(!Ordering::EQ.is_gt());
        assert!(Ordering::EQ.is_le());
        assert!(Ordering::EQ.is_ge());
        assert!(!Ordering::EQ.is_ne());

        assert!(!Ordering::GT.is_lt());
        assert!(!Ordering::GT.is_eq());
        assert!(Ordering::GT.is_gt());
        assert!(!Ordering::GT.is_le());
        assert!(Ordering::GT.is_ge());
        assert!(Ordering::GT.is_ne());
    }

    #[test]
    fn test_then_with() {
        let count = std::cell::Cell::new(0);
        let result = Ordering::LT.then_with(|| {
            count.set(count.get() + 1);
            Ordering::GT
        });
        assert_eq!(result, Ordering::LT);
        assert_eq!(count.get(), 0); // Closure not called

        let result = Ordering::EQ.then_with(|| {
            count.set(count.get() + 1);
            Ordering::GT
        });
        assert_eq!(result, Ordering::GT);
        assert_eq!(count.get(), 1); // Closure called
    }

    #[test]
    fn test_std_ordering_conversion() {
        assert_eq!(Ordering::from(StdOrdering::Less), Ordering::LT);
        assert_eq!(Ordering::from(StdOrdering::Equal), Ordering::EQ);
        assert_eq!(Ordering::from(StdOrdering::Greater), Ordering::GT);

        assert_eq!(StdOrdering::from(Ordering::LT), StdOrdering::Less);
        assert_eq!(StdOrdering::from(Ordering::EQ), StdOrdering::Equal);
        assert_eq!(StdOrdering::from(Ordering::GT), StdOrdering::Greater);
    }

    #[test]
    fn test_semigroup_monoid_laws() {
        // Semigroup associativity: (a <> b) <> c == a <> (b <> c)
        for a in [Ordering::LT, Ordering::EQ, Ordering::GT] {
            for b in [Ordering::LT, Ordering::EQ, Ordering::GT] {
                for c in [Ordering::LT, Ordering::EQ, Ordering::GT] {
                    assert_eq!(a.then(b).then(c), a.then(b.then(c)));
                }
            }
        }

        // Monoid identity: EQ <> a == a and a <> EQ == a
        for a in [Ordering::LT, Ordering::EQ, Ordering::GT] {
            assert_eq!(Ordering::EQ.then(a), a);
            assert_eq!(a.then(Ordering::EQ), a);
        }
    }
}
