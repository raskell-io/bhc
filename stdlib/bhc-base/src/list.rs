//! List operations
//!
//! C-callable RTS functions for list manipulation used by
//! LLVM-generated code. Lists are represented as heap-allocated
//! ADT nodes with Nil (tag=0) and Cons (tag=1, head, tail) layout.

use std::alloc::{alloc, Layout};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// ADT helpers (internal)
// ---------------------------------------------------------------------------

/// Read the tag (i64) at offset 0 of an ADT node.
unsafe fn get_tag(ptr: *mut u8) -> i64 {
    *(ptr as *const i64)
}

/// Read field at `index` (0-based) starting at offset 8.
unsafe fn get_field(ptr: *mut u8, index: usize) -> *mut u8 {
    *(ptr.add(8 + index * 8) as *const *mut u8)
}

/// Allocate a Nil node (tag=0, 8 bytes).
unsafe fn alloc_nil() -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(8, 8);
    let ptr = alloc(layout);
    *(ptr as *mut i64) = 0;
    ptr
}

/// Allocate a Cons node (tag=1, head at offset 8, tail at offset 16).
unsafe fn alloc_cons(head: *mut u8, tail: *mut u8) -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(24, 8);
    let ptr = alloc(layout);
    *(ptr as *mut i64) = 1;
    *(ptr.add(8) as *mut *mut u8) = head;
    *(ptr.add(16) as *mut *mut u8) = tail;
    ptr
}

// ---------------------------------------------------------------------------
// Conversion helpers (internal)
// ---------------------------------------------------------------------------

/// Collect a linked list into a `Vec<*mut u8>`.
unsafe fn list_to_vec(mut list: *mut u8) -> Vec<*mut u8> {
    let mut vec = Vec::new();
    loop {
        if get_tag(list) == 0 {
            break;
        }
        vec.push(get_field(list, 0));
        list = get_field(list, 1);
    }
    vec
}

/// Build a linked list from a slice, preserving order.
unsafe fn vec_to_list(slice: &[*mut u8]) -> *mut u8 {
    let mut result = alloc_nil();
    for &elem in slice.iter().rev() {
        result = alloc_cons(elem, result);
    }
    result
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Sort a linked list by comparing elements as `i64` (pointer cast).
///
/// Converts to `Vec`, sorts, and rebuilds a new list.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_sort(list: *mut u8) -> *mut u8 {
    let mut vec = list_to_vec(list);
    vec.sort_by_key(|&e| e as i64);
    vec_to_list(&vec)
}

/// Sort a linked list using a comparison closure.
///
/// `cmp_fn` is a pointer to a closure struct whose first field
/// (offset 0) is the code pointer with signature
/// `extern "C" fn(*mut u8, *mut u8, *mut u8) -> *mut u8`.
/// The return value encodes Ordering as LT=-1, EQ=0, GT=1 cast to
/// `*mut u8`.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_sort_by(cmp_fn: *mut u8, list: *mut u8) -> *mut u8 {
    let fn_ptr: extern "C" fn(*mut u8, *mut u8, *mut u8) -> *mut u8 =
        std::mem::transmute(*(cmp_fn as *const *mut u8));

    let mut vec = list_to_vec(list);
    vec.sort_by(|&a, &b| {
        let result = fn_ptr(cmp_fn, a, b) as i64;
        match result {
            r if r < 0 => std::cmp::Ordering::Less,
            0 => std::cmp::Ordering::Equal,
            _ => std::cmp::Ordering::Greater,
        }
    });
    vec_to_list(&vec)
}

/// Remove duplicate elements from a list (by `i64` value comparison).
///
/// Preserves the order of first occurrences.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_nub(list: *mut u8) -> *mut u8 {
    let vec = list_to_vec(list);
    let mut seen = HashSet::new();
    let deduped: Vec<*mut u8> = vec
        .into_iter()
        .filter(|&e| seen.insert(e as i64))
        .collect();
    vec_to_list(&deduped)
}

/// Group consecutive equal elements into sublists.
///
/// `group [1,1,2,2,2,3] = [[1,1],[2,2,2],[3]]`
#[no_mangle]
pub unsafe extern "C" fn bhc_list_group(list: *mut u8) -> *mut u8 {
    let vec = list_to_vec(list);
    if vec.is_empty() {
        return alloc_nil();
    }

    let mut groups: Vec<Vec<*mut u8>> = Vec::new();
    let mut current_group: Vec<*mut u8> = vec![vec[0]];

    for &elem in &vec[1..] {
        if elem as i64 == *current_group.last().unwrap() as i64 {
            current_group.push(elem);
        } else {
            groups.push(std::mem::take(&mut current_group));
            current_group.push(elem);
        }
    }
    groups.push(current_group);

    // Build list of lists: convert each group to a sublist, then wrap
    let sublists: Vec<*mut u8> = groups.iter().map(|g| vec_to_list(g)).collect();
    vec_to_list(&sublists)
}

/// Concatenate a list of lists with a separator list between each.
///
/// `intercalate ", " ["a","b","c"] = "a, b, c"` (for general lists).
#[no_mangle]
pub unsafe extern "C" fn bhc_list_intercalate(sep: *mut u8, lists: *mut u8) -> *mut u8 {
    let sep_vec = list_to_vec(sep);
    let outer = list_to_vec(lists);

    if outer.is_empty() {
        return alloc_nil();
    }

    let mut result: Vec<*mut u8> = Vec::new();
    for (i, &sublist_ptr) in outer.iter().enumerate() {
        if i > 0 {
            result.extend_from_slice(&sep_vec);
        }
        let sublist = list_to_vec(sublist_ptr);
        result.extend_from_slice(&sublist);
    }
    vec_to_list(&result)
}

/// Transpose a list of lists.
///
/// `transpose [[1,2],[3,4],[5,6]] = [[1,3,5],[2,4,6]]`
///
/// Follows Haskell semantics: rows of different lengths are handled
/// by skipping missing elements (shorter rows are ignored once
/// exhausted).
#[no_mangle]
pub unsafe extern "C" fn bhc_list_transpose(lists: *mut u8) -> *mut u8 {
    let rows: Vec<Vec<*mut u8>> = list_to_vec(lists)
        .into_iter()
        .map(|row_ptr| list_to_vec(row_ptr))
        .collect();

    if rows.is_empty() {
        return alloc_nil();
    }

    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut columns: Vec<Vec<*mut u8>> = Vec::with_capacity(max_cols);

    for col in 0..max_cols {
        let column: Vec<*mut u8> = rows
            .iter()
            .filter_map(|row| row.get(col).copied())
            .collect();
        columns.push(column);
    }

    let sublists: Vec<*mut u8> = columns.iter().map(|c| vec_to_list(c)).collect();
    vec_to_list(&sublists)
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn make_list(elems: &[i64]) -> *mut u8 {
        let ptrs: Vec<*mut u8> = elems.iter().map(|&v| v as *mut u8).collect();
        vec_to_list(&ptrs)
    }

    unsafe fn collect_i64(list: *mut u8) -> Vec<i64> {
        list_to_vec(list).into_iter().map(|p| p as i64).collect()
    }

    unsafe fn collect_nested(list: *mut u8) -> Vec<Vec<i64>> {
        list_to_vec(list)
            .into_iter()
            .map(|sub| collect_i64(sub))
            .collect()
    }

    #[test]
    fn test_sort_empty() {
        unsafe {
            let list = alloc_nil();
            let sorted = bhc_list_sort(list);
            assert_eq!(collect_i64(sorted), Vec::<i64>::new());
        }
    }

    #[test]
    fn test_sort() {
        unsafe {
            let list = make_list(&[3, 1, 4, 1, 5, 9, 2, 6]);
            let sorted = bhc_list_sort(list);
            assert_eq!(collect_i64(sorted), vec![1, 1, 2, 3, 4, 5, 6, 9]);
        }
    }

    #[test]
    fn test_nub() {
        unsafe {
            let list = make_list(&[1, 2, 3, 2, 1, 4]);
            let deduped = bhc_list_nub(list);
            assert_eq!(collect_i64(deduped), vec![1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_nub_empty() {
        unsafe {
            let list = alloc_nil();
            let deduped = bhc_list_nub(list);
            assert_eq!(collect_i64(deduped), Vec::<i64>::new());
        }
    }

    #[test]
    fn test_group() {
        unsafe {
            let list = make_list(&[1, 1, 2, 2, 2, 3]);
            let groups = bhc_list_group(list);
            assert_eq!(
                collect_nested(groups),
                vec![vec![1, 1], vec![2, 2, 2], vec![3]]
            );
        }
    }

    #[test]
    fn test_group_empty() {
        unsafe {
            let list = alloc_nil();
            let groups = bhc_list_group(list);
            assert_eq!(collect_nested(groups), Vec::<Vec<i64>>::new());
        }
    }

    #[test]
    fn test_intercalate() {
        unsafe {
            // intercalate [0] [[1,2],[3,4],[5,6]] = [1,2,0,3,4,0,5,6]
            let sep = make_list(&[0]);
            let a = make_list(&[1, 2]);
            let b = make_list(&[3, 4]);
            let c = make_list(&[5, 6]);
            let lists = vec_to_list(&[a, b, c]);
            let result = bhc_list_intercalate(sep, lists);
            assert_eq!(collect_i64(result), vec![1, 2, 0, 3, 4, 0, 5, 6]);
        }
    }

    #[test]
    fn test_transpose() {
        unsafe {
            // transpose [[1,2],[3,4],[5,6]] = [[1,3,5],[2,4,6]]
            let a = make_list(&[1, 2]);
            let b = make_list(&[3, 4]);
            let c = make_list(&[5, 6]);
            let lists = vec_to_list(&[a, b, c]);
            let result = bhc_list_transpose(lists);
            assert_eq!(
                collect_nested(result),
                vec![vec![1, 3, 5], vec![2, 4, 6]]
            );
        }
    }

    #[test]
    fn test_transpose_ragged() {
        unsafe {
            // transpose [[1,2,3],[4]] = [[1,4],[2],[3]]
            let a = make_list(&[1, 2, 3]);
            let b = make_list(&[4]);
            let lists = vec_to_list(&[a, b]);
            let result = bhc_list_transpose(lists);
            assert_eq!(
                collect_nested(result),
                vec![vec![1, 4], vec![2], vec![3]]
            );
        }
    }
}
