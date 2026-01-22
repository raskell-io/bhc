//! Boolean type and operations
//!
//! Provides the core `Bool` type and boolean operations for BHC.

/// BHC Boolean type
///
/// Represented as a single byte for FFI compatibility.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bool {
    /// False value
    False = 0,
    /// True value
    True = 1,
}

impl Bool {
    /// Convert from Rust bool
    #[inline]
    pub const fn from_bool(b: bool) -> Self {
        if b {
            Bool::True
        } else {
            Bool::False
        }
    }

    /// Convert to Rust bool
    #[inline]
    pub const fn to_bool(self) -> bool {
        matches!(self, Bool::True)
    }
}

impl From<bool> for Bool {
    #[inline]
    fn from(b: bool) -> Self {
        Self::from_bool(b)
    }
}

impl From<Bool> for bool {
    #[inline]
    fn from(b: Bool) -> Self {
        b.to_bool()
    }
}

// FFI exports

/// Boolean AND
#[no_mangle]
pub extern "C" fn bhc_bool_and(a: Bool, b: Bool) -> Bool {
    Bool::from_bool(a.to_bool() && b.to_bool())
}

/// Boolean OR
#[no_mangle]
pub extern "C" fn bhc_bool_or(a: Bool, b: Bool) -> Bool {
    Bool::from_bool(a.to_bool() || b.to_bool())
}

/// Boolean NOT
#[no_mangle]
pub extern "C" fn bhc_bool_not(a: Bool) -> Bool {
    Bool::from_bool(!a.to_bool())
}

/// Boolean XOR
#[no_mangle]
pub extern "C" fn bhc_bool_xor(a: Bool, b: Bool) -> Bool {
    Bool::from_bool(a.to_bool() ^ b.to_bool())
}

/// Boolean equality
#[no_mangle]
pub extern "C" fn bhc_bool_eq(a: Bool, b: Bool) -> Bool {
    Bool::from_bool(a == b)
}

/// Boolean inequality
#[no_mangle]
pub extern "C" fn bhc_bool_neq(a: Bool, b: Bool) -> Bool {
    Bool::from_bool(a != b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_and() {
        assert_eq!(bhc_bool_and(Bool::True, Bool::True), Bool::True);
        assert_eq!(bhc_bool_and(Bool::True, Bool::False), Bool::False);
        assert_eq!(bhc_bool_and(Bool::False, Bool::True), Bool::False);
        assert_eq!(bhc_bool_and(Bool::False, Bool::False), Bool::False);
    }

    #[test]
    fn test_bool_or() {
        assert_eq!(bhc_bool_or(Bool::True, Bool::True), Bool::True);
        assert_eq!(bhc_bool_or(Bool::True, Bool::False), Bool::True);
        assert_eq!(bhc_bool_or(Bool::False, Bool::True), Bool::True);
        assert_eq!(bhc_bool_or(Bool::False, Bool::False), Bool::False);
    }

    #[test]
    fn test_bool_not() {
        assert_eq!(bhc_bool_not(Bool::True), Bool::False);
        assert_eq!(bhc_bool_not(Bool::False), Bool::True);
    }
}
