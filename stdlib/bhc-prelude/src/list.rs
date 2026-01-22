//! List operations
//!
//! Fast list operations for BHC. The actual list type is managed by the
//! Haskell runtime; these are primitive operations exposed via FFI.
//!
//! # Implementation Notes
//!
//! These operations are designed to be called from Haskell via FFI.
//! The list representation uses the BHC runtime's spine-strict cons cells.

use crate::bool::Bool;
use crate::maybe::Maybe;

/// Opaque list type (actual representation in Haskell runtime)
#[repr(C)]
pub struct List<T> {
    _marker: std::marker::PhantomData<T>,
}

// Note: Most list operations are implemented in Haskell for fusion.
// These Rust implementations are for performance-critical operations
// that don't participate in fusion.

/// Check if a list is empty
///
/// O(1) time
#[no_mangle]
pub extern "C" fn bhc_list_null(len: usize) -> Bool {
    Bool::from_bool(len == 0)
}

/// Get the length of a list
///
/// Note: In actual implementation, this would traverse the list.
/// Here we assume length is cached or passed in.
#[no_mangle]
pub extern "C" fn bhc_list_length(list_ptr: *const u8) -> usize {
    if list_ptr.is_null() {
        return 0;
    }
    // In actual implementation, this would call into the RTS
    // to traverse the list spine and count elements.
    // For now, return 0 as placeholder.
    0
}

/// Sum of a list of i64 values
///
/// Uses strict accumulator to avoid space leaks.
/// Vectorized when possible.
#[no_mangle]
pub extern "C" fn bhc_list_sum_i64(ptr: *const i64, len: usize) -> i64 {
    if ptr.is_null() || len == 0 {
        return 0;
    }

    // SAFETY: Caller guarantees ptr points to valid array of len elements
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    // Vectorized sum
    slice.iter().copied().sum()
}

/// Sum of a list of f64 values
///
/// Uses Kahan summation for better precision.
#[no_mangle]
pub extern "C" fn bhc_list_sum_f64(ptr: *const f64, len: usize) -> f64 {
    if ptr.is_null() || len == 0 {
        return 0.0;
    }

    // SAFETY: Caller guarantees ptr points to valid array of len elements
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    // Kahan summation for better precision
    let mut sum = 0.0;
    let mut c = 0.0; // Compensation for lost low-order bits

    for &x in slice {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    sum
}

/// Product of a list of i64 values
#[no_mangle]
pub extern "C" fn bhc_list_product_i64(ptr: *const i64, len: usize) -> i64 {
    if ptr.is_null() || len == 0 {
        return 1;
    }

    // SAFETY: Caller guarantees ptr points to valid array of len elements
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    slice.iter().copied().product()
}

/// Product of a list of f64 values
#[no_mangle]
pub extern "C" fn bhc_list_product_f64(ptr: *const f64, len: usize) -> f64 {
    if ptr.is_null() || len == 0 {
        return 1.0;
    }

    // SAFETY: Caller guarantees ptr points to valid array of len elements
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    slice.iter().copied().product()
}

/// Maximum of a list of i64 values
///
/// Returns Maybe::Nothing for empty list.
#[no_mangle]
pub extern "C" fn bhc_list_maximum_i64(ptr: *const i64, len: usize) -> Maybe<i64> {
    if ptr.is_null() || len == 0 {
        return Maybe::Nothing;
    }

    // SAFETY: Caller guarantees ptr points to valid array of len elements
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    slice.iter().copied().max().into()
}

/// Minimum of a list of i64 values
///
/// Returns Maybe::Nothing for empty list.
#[no_mangle]
pub extern "C" fn bhc_list_minimum_i64(ptr: *const i64, len: usize) -> Maybe<i64> {
    if ptr.is_null() || len == 0 {
        return Maybe::Nothing;
    }

    // SAFETY: Caller guarantees ptr points to valid array of len elements
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    slice.iter().copied().min().into()
}

/// Check if all elements satisfy a predicate
///
/// Short-circuits on first false.
#[no_mangle]
pub extern "C" fn bhc_list_all_i64(
    ptr: *const i64,
    len: usize,
    pred: extern "C" fn(i64) -> Bool,
) -> Bool {
    if ptr.is_null() || len == 0 {
        return Bool::True;
    }

    // SAFETY: Caller guarantees ptr points to valid array of len elements
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    for &x in slice {
        if !pred(x).to_bool() {
            return Bool::False;
        }
    }

    Bool::True
}

/// Check if any element satisfies a predicate
///
/// Short-circuits on first true.
#[no_mangle]
pub extern "C" fn bhc_list_any_i64(
    ptr: *const i64,
    len: usize,
    pred: extern "C" fn(i64) -> Bool,
) -> Bool {
    if ptr.is_null() || len == 0 {
        return Bool::False;
    }

    // SAFETY: Caller guarantees ptr points to valid array of len elements
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    for &x in slice {
        if pred(x).to_bool() {
            return Bool::True;
        }
    }

    Bool::False
}

/// Check if an element is in the list
#[no_mangle]
pub extern "C" fn bhc_list_elem_i64(ptr: *const i64, len: usize, x: i64) -> Bool {
    if ptr.is_null() || len == 0 {
        return Bool::False;
    }

    // SAFETY: Caller guarantees ptr points to valid array of len elements
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    Bool::from_bool(slice.contains(&x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_i64() {
        let arr = [1i64, 2, 3, 4, 5];
        assert_eq!(bhc_list_sum_i64(arr.as_ptr(), arr.len()), 15);
        assert_eq!(bhc_list_sum_i64(std::ptr::null(), 0), 0);
    }

    #[test]
    fn test_sum_f64() {
        let arr = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let sum = bhc_list_sum_f64(arr.as_ptr(), arr.len());
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_product_i64() {
        let arr = [1i64, 2, 3, 4, 5];
        assert_eq!(bhc_list_product_i64(arr.as_ptr(), arr.len()), 120);
        assert_eq!(bhc_list_product_i64(std::ptr::null(), 0), 1);
    }

    #[test]
    fn test_maximum_minimum() {
        let arr = [3i64, 1, 4, 1, 5, 9, 2, 6];
        assert_eq!(bhc_list_maximum_i64(arr.as_ptr(), arr.len()), Maybe::Just(9));
        assert_eq!(bhc_list_minimum_i64(arr.as_ptr(), arr.len()), Maybe::Just(1));

        assert_eq!(bhc_list_maximum_i64(std::ptr::null(), 0), Maybe::Nothing);
        assert_eq!(bhc_list_minimum_i64(std::ptr::null(), 0), Maybe::Nothing);
    }

    #[test]
    fn test_elem() {
        let arr = [1i64, 2, 3, 4, 5];
        assert_eq!(bhc_list_elem_i64(arr.as_ptr(), arr.len(), 3), Bool::True);
        assert_eq!(bhc_list_elem_i64(arr.as_ptr(), arr.len(), 6), Bool::False);
    }
}
