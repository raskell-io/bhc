//! Extended list operations
//!
//! This module provides additional list operations beyond those in the Prelude.
//! All operations participate in fusion where applicable.

use bhc_prelude::bool::Bool;
use bhc_prelude::maybe::Maybe;
use bhc_prelude::ordering::Ordering;

/// Intersperse an element between list elements
///
/// O(n) time, O(n) space
#[no_mangle]
pub extern "C" fn bhc_list_intersperse_i64(
    ptr: *const i64,
    len: usize,
    sep: i64,
    out: *mut i64,
) -> usize {
    if ptr.is_null() || out.is_null() || len == 0 {
        return 0;
    }

    // SAFETY: Caller guarantees valid pointers and sufficient output space
    let input = unsafe { std::slice::from_raw_parts(ptr, len) };
    let output_len = if len > 0 { 2 * len - 1 } else { 0 };
    let output = unsafe { std::slice::from_raw_parts_mut(out, output_len) };

    let mut j = 0;
    for (i, &x) in input.iter().enumerate() {
        if i > 0 {
            output[j] = sep;
            j += 1;
        }
        output[j] = x;
        j += 1;
    }

    output_len
}

/// Reverse a list in-place
///
/// O(n) time, O(1) space
#[no_mangle]
pub extern "C" fn bhc_list_reverse_i64(ptr: *mut i64, len: usize) {
    if ptr.is_null() || len <= 1 {
        return;
    }

    // SAFETY: Caller guarantees valid pointer
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    slice.reverse();
}

/// Sort a list of i64 values
///
/// O(n log n) time, O(n) space (stable sort)
#[no_mangle]
pub extern "C" fn bhc_list_sort_i64(ptr: *mut i64, len: usize) {
    if ptr.is_null() || len <= 1 {
        return;
    }

    // SAFETY: Caller guarantees valid pointer
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    slice.sort();
}

/// Sort a list of f64 values
///
/// O(n log n) time, O(n) space (stable sort)
/// NaN values are sorted to the end
#[no_mangle]
pub extern "C" fn bhc_list_sort_f64(ptr: *mut f64, len: usize) {
    if ptr.is_null() || len <= 1 {
        return;
    }

    // SAFETY: Caller guarantees valid pointer
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
}

/// Remove duplicates from a sorted list
///
/// O(n) time, O(1) space
/// Returns new length
#[no_mangle]
pub extern "C" fn bhc_list_nub_sorted_i64(ptr: *mut i64, len: usize) -> usize {
    if ptr.is_null() || len <= 1 {
        return len;
    }

    // SAFETY: Caller guarantees valid pointer
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

    let mut write = 1;
    for read in 1..len {
        if slice[read] != slice[write - 1] {
            slice[write] = slice[read];
            write += 1;
        }
    }

    write
}

/// Group consecutive equal elements
///
/// Returns indices where groups start (length = number of groups + 1)
#[no_mangle]
pub extern "C" fn bhc_list_group_i64(
    ptr: *const i64,
    len: usize,
    out_indices: *mut usize,
) -> usize {
    if ptr.is_null() || out_indices.is_null() || len == 0 {
        if !out_indices.is_null() {
            unsafe { *out_indices = 0 };
        }
        return 1;
    }

    // SAFETY: Caller guarantees valid pointers
    let input = unsafe { std::slice::from_raw_parts(ptr, len) };

    let mut group_count = 1;
    unsafe {
        *out_indices = 0;
    }

    for i in 1..len {
        if input[i] != input[i - 1] {
            unsafe {
                *out_indices.add(group_count) = i;
            }
            group_count += 1;
        }
    }

    unsafe {
        *out_indices.add(group_count) = len;
    }

    group_count + 1
}

/// Check if a list is sorted
///
/// O(n) time, O(1) space
#[no_mangle]
pub extern "C" fn bhc_list_is_sorted_i64(ptr: *const i64, len: usize) -> Bool {
    if ptr.is_null() || len <= 1 {
        return Bool::True;
    }

    // SAFETY: Caller guarantees valid pointer
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    for i in 1..len {
        if slice[i] < slice[i - 1] {
            return Bool::False;
        }
    }

    Bool::True
}

/// Binary search in a sorted list
///
/// O(log n) time, O(1) space
/// Returns index if found, or insertion point negated and minus 1
#[no_mangle]
pub extern "C" fn bhc_list_binary_search_i64(ptr: *const i64, len: usize, target: i64) -> i64 {
    if ptr.is_null() || len == 0 {
        return -1; // Insert at position 0
    }

    // SAFETY: Caller guarantees valid pointer
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    match slice.binary_search(&target) {
        Ok(idx) => idx as i64,
        Err(idx) => -(idx as i64) - 1,
    }
}

/// Partition a list by a predicate
///
/// Elements satisfying the predicate come first.
/// Returns the index of the first element not satisfying the predicate.
#[no_mangle]
pub extern "C" fn bhc_list_partition_i64(
    ptr: *mut i64,
    len: usize,
    pred: extern "C" fn(i64) -> Bool,
) -> usize {
    if ptr.is_null() || len == 0 {
        return 0;
    }

    // SAFETY: Caller guarantees valid pointer
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

    let mut write = 0;
    for read in 0..len {
        if pred(slice[read]).to_bool() {
            slice.swap(read, write);
            write += 1;
        }
    }

    write
}

/// Find the index of the first element satisfying a predicate
///
/// O(n) time, O(1) space
#[no_mangle]
pub extern "C" fn bhc_list_find_index_i64(
    ptr: *const i64,
    len: usize,
    pred: extern "C" fn(i64) -> Bool,
) -> Maybe<usize> {
    if ptr.is_null() || len == 0 {
        return Maybe::Nothing;
    }

    // SAFETY: Caller guarantees valid pointer
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    for (i, &x) in slice.iter().enumerate() {
        if pred(x).to_bool() {
            return Maybe::Just(i);
        }
    }

    Maybe::Nothing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse() {
        let mut arr = [1i64, 2, 3, 4, 5];
        bhc_list_reverse_i64(arr.as_mut_ptr(), arr.len());
        assert_eq!(arr, [5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_sort() {
        let mut arr = [3i64, 1, 4, 1, 5, 9, 2, 6];
        bhc_list_sort_i64(arr.as_mut_ptr(), arr.len());
        assert_eq!(arr, [1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_binary_search() {
        let arr = [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(bhc_list_binary_search_i64(arr.as_ptr(), arr.len(), 5), 4);
        assert_eq!(bhc_list_binary_search_i64(arr.as_ptr(), arr.len(), 11), -11);
    }

    #[test]
    fn test_is_sorted() {
        let sorted = [1i64, 2, 3, 4, 5];
        let unsorted = [1i64, 3, 2, 4, 5];
        assert_eq!(bhc_list_is_sorted_i64(sorted.as_ptr(), sorted.len()), Bool::True);
        assert_eq!(bhc_list_is_sorted_i64(unsorted.as_ptr(), unsorted.len()), Bool::False);
    }
}
