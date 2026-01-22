//! Extended list operations
//!
//! This module provides additional list operations beyond those in the Prelude.
//! All operations participate in fusion where applicable.
//!
//! # Categories
//!
//! - **Basic operations**: take, drop, splitAt, reverse
//! - **Search**: elem, find, findIndex, filter, partition
//! - **Transformations**: map, concatMap, intersperse, intercalate, transpose
//! - **Reductions**: foldr, foldl, sum, product, maximum, minimum
//! - **Sublists**: takeWhile, dropWhile, span, break
//! - **Zipping**: zip, zipWith, unzip
//! - **Set operations**: nub, delete, union, intersect
//! - **Sorting**: sort, sortBy, sortOn, insert
//!
//! # Performance
//!
//! All operations are designed to fuse with other list operations.
//! Use `{-# RULES #-}` pragmas for fusion in Haskell code.

use bhc_prelude::bool::Bool;
use bhc_prelude::maybe::Maybe;
use bhc_prelude::ordering::Ordering;

// ============================================================
// Pure Rust implementations (for internal use and testing)
// ============================================================

/// Take the first n elements from a slice.
#[inline]
pub fn take<T>(n: usize, slice: &[T]) -> &[T] {
    if n >= slice.len() {
        slice
    } else {
        &slice[..n]
    }
}

/// Drop the first n elements from a slice.
#[inline]
pub fn drop<T>(n: usize, slice: &[T]) -> &[T] {
    if n >= slice.len() {
        &[]
    } else {
        &slice[n..]
    }
}

/// Split a slice at position n.
#[inline]
pub fn split_at<T>(n: usize, slice: &[T]) -> (&[T], &[T]) {
    let n = std::cmp::min(n, slice.len());
    slice.split_at(n)
}

/// Take elements while predicate holds.
pub fn take_while<T, P>(pred: P, slice: &[T]) -> &[T]
where
    P: Fn(&T) -> bool,
{
    let mut i = 0;
    while i < slice.len() && pred(&slice[i]) {
        i += 1;
    }
    &slice[..i]
}

/// Drop elements while predicate holds.
pub fn drop_while<T, P>(pred: P, slice: &[T]) -> &[T]
where
    P: Fn(&T) -> bool,
{
    let mut i = 0;
    while i < slice.len() && pred(&slice[i]) {
        i += 1;
    }
    &slice[i..]
}

/// Split at the first element not satisfying the predicate.
pub fn span<T, P>(pred: P, slice: &[T]) -> (&[T], &[T])
where
    P: Fn(&T) -> bool,
{
    let mut i = 0;
    while i < slice.len() && pred(&slice[i]) {
        i += 1;
    }
    slice.split_at(i)
}

/// Split at the first element satisfying the predicate.
pub fn break_<T, P>(pred: P, slice: &[T]) -> (&[T], &[T])
where
    P: Fn(&T) -> bool,
{
    span(|x| !pred(x), slice)
}

/// Zip two slices together.
pub fn zip<T: Clone, U: Clone>(xs: &[T], ys: &[U]) -> Vec<(T, U)> {
    xs.iter().zip(ys.iter()).map(|(a, b)| (a.clone(), b.clone())).collect()
}

/// Zip two slices with a function.
pub fn zip_with<T, U, V, F>(f: F, xs: &[T], ys: &[U]) -> Vec<V>
where
    F: Fn(&T, &U) -> V,
{
    xs.iter().zip(ys.iter()).map(|(a, b)| f(a, b)).collect()
}

/// Unzip a slice of pairs.
pub fn unzip<T: Clone, U: Clone>(pairs: &[(T, U)]) -> (Vec<T>, Vec<U>) {
    pairs.iter().map(|(a, b)| (a.clone(), b.clone())).unzip()
}

/// Replicate a value n times.
pub fn replicate<T: Clone>(n: usize, x: T) -> Vec<T> {
    vec![x; n]
}

/// Concatenate a slice of slices.
pub fn concat<T: Clone>(xss: &[Vec<T>]) -> Vec<T> {
    xss.iter().flat_map(|xs| xs.iter().cloned()).collect()
}

/// Map and concatenate.
pub fn concat_map<T, U, F>(f: F, xs: &[T]) -> Vec<U>
where
    F: Fn(&T) -> Vec<U>,
{
    xs.iter().flat_map(&f).collect()
}

/// Left scan with initial value.
pub fn scanl<T: Clone, U: Clone, F>(f: F, init: U, xs: &[T]) -> Vec<U>
where
    F: Fn(&U, &T) -> U,
{
    let mut result = Vec::with_capacity(xs.len() + 1);
    let mut acc = init;
    result.push(acc.clone());
    for x in xs {
        acc = f(&acc, x);
        result.push(acc.clone());
    }
    result
}

/// Right scan with initial value.
pub fn scanr<T: Clone, U: Clone, F>(f: F, init: U, xs: &[T]) -> Vec<U>
where
    F: Fn(&T, &U) -> U,
{
    let mut result = Vec::with_capacity(xs.len() + 1);
    let mut acc = init;
    result.push(acc.clone());
    for x in xs.iter().rev() {
        acc = f(x, &acc);
        result.push(acc.clone());
    }
    result.reverse();
    result
}

/// Insert separator between elements.
pub fn intersperse<T: Clone>(sep: T, xs: &[T]) -> Vec<T> {
    if xs.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(2 * xs.len() - 1);
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            result.push(sep.clone());
        }
        result.push(x.clone());
    }
    result
}

/// Insert a list between each element.
pub fn intercalate<T: Clone>(sep: &[T], xss: &[Vec<T>]) -> Vec<T> {
    if xss.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    for (i, xs) in xss.iter().enumerate() {
        if i > 0 {
            result.extend_from_slice(sep);
        }
        result.extend_from_slice(xs);
    }
    result
}

/// Transpose rows and columns.
pub fn transpose<T: Clone>(xss: &[Vec<T>]) -> Vec<Vec<T>> {
    if xss.is_empty() {
        return Vec::new();
    }
    let max_len = xss.iter().map(|xs| xs.len()).max().unwrap_or(0);
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let mut row = Vec::new();
        for xs in xss {
            if let Some(x) = xs.get(i) {
                row.push(x.clone());
            }
        }
        if !row.is_empty() {
            result.push(row);
        }
    }
    result
}

/// Remove duplicates (keeps first occurrence).
pub fn nub<T: Clone + PartialEq>(xs: &[T]) -> Vec<T> {
    let mut seen = Vec::new();
    for x in xs {
        if !seen.contains(x) {
            seen.push(x.clone());
        }
    }
    seen
}

/// Delete first occurrence of element.
pub fn delete<T: Clone + PartialEq>(x: &T, xs: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(xs.len());
    let mut found = false;
    for elem in xs {
        if !found && elem == x {
            found = true;
        } else {
            result.push(elem.clone());
        }
    }
    result
}

/// Set union (xs ++ (ys \\ xs)).
pub fn union<T: Clone + PartialEq>(xs: &[T], ys: &[T]) -> Vec<T> {
    let mut result = xs.to_vec();
    for y in ys {
        if !xs.contains(y) && !result[xs.len()..].contains(y) {
            result.push(y.clone());
        }
    }
    result
}

/// Set intersection.
pub fn intersect<T: Clone + PartialEq>(xs: &[T], ys: &[T]) -> Vec<T> {
    xs.iter().filter(|x| ys.contains(x)).cloned().collect()
}

/// Insert into a sorted list.
pub fn insert<T: Clone + Ord>(x: T, xs: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(xs.len() + 1);
    let mut inserted = false;
    for elem in xs {
        if !inserted && x <= *elem {
            result.push(x.clone());
            inserted = true;
        }
        result.push(elem.clone());
    }
    if !inserted {
        result.push(x);
    }
    result
}

/// Sort using custom comparison.
pub fn sort_by<T: Clone, F>(cmp: F, xs: &[T]) -> Vec<T>
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let mut result = xs.to_vec();
    result.sort_by(cmp);
    result
}

/// Sort by a key function.
pub fn sort_on<T: Clone, K: Ord, F>(f: F, xs: &[T]) -> Vec<T>
where
    F: Fn(&T) -> K,
{
    let mut result = xs.to_vec();
    result.sort_by_key(|x| f(x));
    result
}

// ============================================================
// FFI exports
// ============================================================

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

    // FFI tests
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

    // Pure Rust function tests
    #[test]
    fn test_take() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(take(3, &arr), &[1, 2, 3]);
        assert_eq!(take(0, &arr), &[]);
        assert_eq!(take(10, &arr), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_drop() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(drop(2, &arr), &[3, 4, 5]);
        assert_eq!(drop(0, &arr), &[1, 2, 3, 4, 5]);
        assert_eq!(drop(10, &arr), &[]);
    }

    #[test]
    fn test_split_at() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(split_at(2, &arr), (&[1, 2][..], &[3, 4, 5][..]));
        assert_eq!(split_at(0, &arr), (&[][..], &[1, 2, 3, 4, 5][..]));
        assert_eq!(split_at(5, &arr), (&[1, 2, 3, 4, 5][..], &[][..]));
    }

    #[test]
    fn test_take_while() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(take_while(|x| *x < 4, &arr), &[1, 2, 3]);
        assert_eq!(take_while(|_| true, &arr), &[1, 2, 3, 4, 5]);
        assert_eq!(take_while(|_| false, &arr), &[]);
    }

    #[test]
    fn test_drop_while() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(drop_while(|x| *x < 4, &arr), &[4, 5]);
        assert_eq!(drop_while(|_| true, &arr), &[]);
        assert_eq!(drop_while(|_| false, &arr), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_span() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(span(|x| *x < 4, &arr), (&[1, 2, 3][..], &[4, 5][..]));
    }

    #[test]
    fn test_break() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(break_(|x| *x >= 4, &arr), (&[1, 2, 3][..], &[4, 5][..]));
    }

    #[test]
    fn test_zip() {
        let xs = [1, 2, 3];
        let ys = ["a", "b", "c"];
        assert_eq!(zip(&xs, &ys), vec![(1, "a"), (2, "b"), (3, "c")]);

        let ys2 = ["a", "b"];
        assert_eq!(zip(&xs, &ys2), vec![(1, "a"), (2, "b")]);
    }

    #[test]
    fn test_zip_with() {
        let xs = [1, 2, 3];
        let ys = [10, 20, 30];
        assert_eq!(zip_with(|a, b| a + b, &xs, &ys), vec![11, 22, 33]);
    }

    #[test]
    fn test_unzip() {
        let pairs = [(1, "a"), (2, "b"), (3, "c")];
        let (xs, ys) = unzip(&pairs);
        assert_eq!(xs, vec![1, 2, 3]);
        assert_eq!(ys, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_replicate() {
        assert_eq!(replicate(3, 5), vec![5, 5, 5]);
        assert_eq!(replicate(0, 5), vec![]);
    }

    #[test]
    fn test_concat() {
        let xss = vec![vec![1, 2], vec![3], vec![4, 5, 6]];
        assert_eq!(concat(&xss), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_concat_map() {
        let xs = [1, 2, 3];
        assert_eq!(concat_map(|x| vec![*x, *x], &xs), vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn test_scanl() {
        let xs = [1, 2, 3];
        assert_eq!(scanl(|acc, x| acc + x, 0, &xs), vec![0, 1, 3, 6]);
    }

    #[test]
    fn test_scanr() {
        let xs = [1, 2, 3];
        assert_eq!(scanr(|x, acc| x + acc, 0, &xs), vec![6, 5, 3, 0]);
    }

    #[test]
    fn test_intersperse() {
        let xs = [1, 2, 3];
        assert_eq!(intersperse(0, &xs), vec![1, 0, 2, 0, 3]);
        assert_eq!(intersperse(0, &[1]), vec![1]);
        assert_eq!(intersperse(0, &[]), Vec::<i32>::new());
    }

    #[test]
    fn test_intercalate() {
        let sep = vec![0, 0];
        let xss = vec![vec![1, 2], vec![3, 4], vec![5]];
        assert_eq!(intercalate(&sep, &xss), vec![1, 2, 0, 0, 3, 4, 0, 0, 5]);
    }

    #[test]
    fn test_transpose() {
        let xss = vec![vec![1, 2, 3], vec![4, 5, 6]];
        assert_eq!(transpose(&xss), vec![vec![1, 4], vec![2, 5], vec![3, 6]]);

        let ragged = vec![vec![1, 2, 3], vec![4, 5]];
        assert_eq!(transpose(&ragged), vec![vec![1, 4], vec![2, 5], vec![3]]);
    }

    #[test]
    fn test_nub() {
        let xs = [1, 2, 1, 3, 2, 4, 1];
        assert_eq!(nub(&xs), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_delete() {
        let xs = [1, 2, 3, 2, 4];
        assert_eq!(delete(&2, &xs), vec![1, 3, 2, 4]);
        assert_eq!(delete(&5, &xs), vec![1, 2, 3, 2, 4]);
    }

    #[test]
    fn test_union() {
        let xs = [1, 2, 3];
        let ys = [2, 3, 4, 5];
        assert_eq!(union(&xs, &ys), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_intersect() {
        let xs = [1, 2, 3, 4];
        let ys = [2, 4, 6];
        assert_eq!(intersect(&xs, &ys), vec![2, 4]);
    }

    #[test]
    fn test_insert() {
        let xs = [1, 2, 4, 5];
        assert_eq!(insert(3, &xs), vec![1, 2, 3, 4, 5]);
        assert_eq!(insert(0, &xs), vec![0, 1, 2, 4, 5]);
        assert_eq!(insert(6, &xs), vec![1, 2, 4, 5, 6]);
    }

    #[test]
    fn test_sort_by() {
        let xs = [3, 1, 4, 1, 5, 9];
        assert_eq!(sort_by(|a, b| b.cmp(a), &xs), vec![9, 5, 4, 3, 1, 1]); // descending
    }

    #[test]
    fn test_sort_on() {
        let xs = ["hello", "hi", "hey"];
        assert_eq!(sort_on(|s| s.len(), &xs), vec!["hi", "hey", "hello"]);
    }
}
