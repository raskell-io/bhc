//! BHC Containers Library - Rust FFI support for LLVM codegen
//!
//! Provides C-callable functions for container operations used by
//! generated LLVM code. Containers are opaque heap-allocated objects.
//!
//! At the LLVM level, all values are pointer-sized (`*mut u8`).
//! Map/Set use i64-cast pointer comparison for key ordering.

#![warn(missing_docs)]
#![allow(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::ptr;

// ========================================================================
// Opaque container types
// ========================================================================

/// Opaque Map type: BTreeMap<i64, *mut u8> behind a Box.
type RtsMap = BTreeMap<i64, *mut u8>;

/// Opaque Set type: BTreeSet<i64> behind a Box.
type RtsSet = BTreeSet<i64>;

// ========================================================================
// Data.Map operations
// ========================================================================

/// Create an empty map.
#[no_mangle]
pub extern "C" fn bhc_map_empty() -> *mut u8 {
    let m: Box<RtsMap> = Box::new(BTreeMap::new());
    Box::into_raw(m) as *mut u8
}

/// Create a singleton map.
#[no_mangle]
pub extern "C" fn bhc_map_singleton(key: i64, value: *mut u8) -> *mut u8 {
    let mut m = BTreeMap::new();
    m.insert(key, value);
    Box::into_raw(Box::new(m)) as *mut u8
}

/// Check if map is empty. Returns 1 if null, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_null(map_ptr: *mut u8) -> i64 {
    if map_ptr.is_null() { return 1; }
    let m = &*(map_ptr as *const RtsMap);
    if m.is_empty() { 1 } else { 0 }
}

/// Get the size of a map.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_size(map_ptr: *mut u8) -> i64 {
    if map_ptr.is_null() { return 0; }
    let m = &*(map_ptr as *const RtsMap);
    m.len() as i64
}

/// Check if a key is a member of the map. Returns 1 if member, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_member(key: i64, map_ptr: *mut u8) -> i64 {
    if map_ptr.is_null() { return 0; }
    let m = &*(map_ptr as *const RtsMap);
    if m.contains_key(&key) { 1 } else { 0 }
}

/// Lookup a key in the map. Returns the value pointer or null if not found.
/// The caller must wrap in Just/Nothing.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_lookup(key: i64, map_ptr: *mut u8) -> *mut u8 {
    if map_ptr.is_null() { return ptr::null_mut(); }
    let m = &*(map_ptr as *const RtsMap);
    match m.get(&key) {
        Some(&v) => v,
        None => ptr::null_mut(),
    }
}

/// Find with default: return the value for key, or default if not found.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_find_with_default(default: *mut u8, key: i64, map_ptr: *mut u8) -> *mut u8 {
    if map_ptr.is_null() { return default; }
    let m = &*(map_ptr as *const RtsMap);
    match m.get(&key) {
        Some(&v) => v,
        None => default,
    }
}

/// Insert a key-value pair into the map. Returns a new map (COW).
#[no_mangle]
pub unsafe extern "C" fn bhc_map_insert(key: i64, value: *mut u8, map_ptr: *mut u8) -> *mut u8 {
    let mut m = if map_ptr.is_null() {
        BTreeMap::new()
    } else {
        (*(map_ptr as *const RtsMap)).clone()
    };
    m.insert(key, value);
    Box::into_raw(Box::new(m)) as *mut u8
}

/// Delete a key from the map. Returns a new map (COW).
#[no_mangle]
pub unsafe extern "C" fn bhc_map_delete(key: i64, map_ptr: *mut u8) -> *mut u8 {
    if map_ptr.is_null() { return bhc_map_empty(); }
    let mut m = (*(map_ptr as *const RtsMap)).clone();
    m.remove(&key);
    Box::into_raw(Box::new(m)) as *mut u8
}

/// Union of two maps (left-biased). Returns a new map.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_union(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    let mut result = if map1.is_null() { BTreeMap::new() } else { (*(map1 as *const RtsMap)).clone() };
    if !map2.is_null() {
        let m2 = &*(map2 as *const RtsMap);
        for (&k, &v) in m2.iter() {
            result.entry(k).or_insert(v);
        }
    }
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Intersection of two maps (left-biased). Returns a new map.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_intersection(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    if map1.is_null() || map2.is_null() { return bhc_map_empty(); }
    let m1 = &*(map1 as *const RtsMap);
    let m2 = &*(map2 as *const RtsMap);
    let result: RtsMap = m1.iter()
        .filter(|(k, _)| m2.contains_key(k))
        .map(|(&k, &v)| (k, v))
        .collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Difference of two maps. Returns a new map with keys in m1 but not m2.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_difference(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    if map1.is_null() { return bhc_map_empty(); }
    if map2.is_null() {
        return Box::into_raw(Box::new((*(map1 as *const RtsMap)).clone())) as *mut u8;
    }
    let m1 = &*(map1 as *const RtsMap);
    let m2 = &*(map2 as *const RtsMap);
    let result: RtsMap = m1.iter()
        .filter(|(k, _)| !m2.contains_key(k))
        .map(|(&k, &v)| (k, v))
        .collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Get the keys of a map as a count + array.
/// Returns the number of keys. Writes key array to `out_keys` if non-null.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_keys_count(map_ptr: *mut u8) -> i64 {
    if map_ptr.is_null() { return 0; }
    let m = &*(map_ptr as *const RtsMap);
    m.len() as i64
}

/// Get a key at index from the map (for iteration).
#[no_mangle]
pub unsafe extern "C" fn bhc_map_key_at(map_ptr: *mut u8, index: i64) -> i64 {
    if map_ptr.is_null() { return 0; }
    let m = &*(map_ptr as *const RtsMap);
    m.keys().nth(index as usize).copied().unwrap_or(0)
}

/// Get a value at index from the map (for iteration).
#[no_mangle]
pub unsafe extern "C" fn bhc_map_value_at(map_ptr: *mut u8, index: i64) -> *mut u8 {
    if map_ptr.is_null() { return ptr::null_mut(); }
    let m = &*(map_ptr as *const RtsMap);
    m.values().nth(index as usize).copied().unwrap_or(ptr::null_mut())
}

/// Check if map1 is a submap of map2.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_is_submap_of(map1: *mut u8, map2: *mut u8) -> i64 {
    if map1.is_null() { return 1; }
    if map2.is_null() { return 0; }
    let m1 = &*(map1 as *const RtsMap);
    let m2 = &*(map2 as *const RtsMap);
    if m1.keys().all(|k| m2.contains_key(k)) { 1 } else { 0 }
}

// ========================================================================
// Data.Set operations
// ========================================================================

/// Create an empty set.
#[no_mangle]
pub extern "C" fn bhc_set_empty() -> *mut u8 {
    Box::into_raw(Box::new(BTreeSet::<i64>::new())) as *mut u8
}

/// Create a singleton set.
#[no_mangle]
pub extern "C" fn bhc_set_singleton(value: i64) -> *mut u8 {
    let mut s = BTreeSet::new();
    s.insert(value);
    Box::into_raw(Box::new(s)) as *mut u8
}

/// Check if set is empty.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_null(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() { return 1; }
    let s = &*(set_ptr as *const RtsSet);
    if s.is_empty() { 1 } else { 0 }
}

/// Get the size of a set.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_size(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() { return 0; }
    let s = &*(set_ptr as *const RtsSet);
    s.len() as i64
}

/// Check if a value is a member of the set.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_member(value: i64, set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() { return 0; }
    let s = &*(set_ptr as *const RtsSet);
    if s.contains(&value) { 1 } else { 0 }
}

/// Insert a value into the set. Returns a new set (COW).
#[no_mangle]
pub unsafe extern "C" fn bhc_set_insert(value: i64, set_ptr: *mut u8) -> *mut u8 {
    let mut s = if set_ptr.is_null() {
        BTreeSet::new()
    } else {
        (*(set_ptr as *const RtsSet)).clone()
    };
    s.insert(value);
    Box::into_raw(Box::new(s)) as *mut u8
}

/// Delete a value from the set. Returns a new set (COW).
#[no_mangle]
pub unsafe extern "C" fn bhc_set_delete(value: i64, set_ptr: *mut u8) -> *mut u8 {
    if set_ptr.is_null() { return bhc_set_empty(); }
    let mut s = (*(set_ptr as *const RtsSet)).clone();
    s.remove(&value);
    Box::into_raw(Box::new(s)) as *mut u8
}

/// Union of two sets. Returns a new set.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_union(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    let s1 = if set1.is_null() { BTreeSet::new() } else { (*(set1 as *const RtsSet)).clone() };
    let empty = BTreeSet::new();
    let s2 = if set2.is_null() { &empty } else { &*(set2 as *const RtsSet) };
    let result: BTreeSet<i64> = s1.union(s2).copied().collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Intersection of two sets. Returns a new set.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_intersection(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    if set1.is_null() || set2.is_null() { return bhc_set_empty(); }
    let s1 = &*(set1 as *const RtsSet);
    let s2 = &*(set2 as *const RtsSet);
    let result: BTreeSet<i64> = s1.intersection(s2).copied().collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Difference of two sets. Returns a new set.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_difference(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    if set1.is_null() { return bhc_set_empty(); }
    if set2.is_null() {
        return Box::into_raw(Box::new((*(set1 as *const RtsSet)).clone())) as *mut u8;
    }
    let s1 = &*(set1 as *const RtsSet);
    let s2 = &*(set2 as *const RtsSet);
    let result: BTreeSet<i64> = s1.difference(s2).copied().collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Check if set1 is a subset of set2.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_is_subset_of(set1: *mut u8, set2: *mut u8) -> i64 {
    if set1.is_null() { return 1; }
    if set2.is_null() { return 0; }
    let s1 = &*(set1 as *const RtsSet);
    let s2 = &*(set2 as *const RtsSet);
    if s1.is_subset(s2) { 1 } else { 0 }
}

/// Check if set1 is a proper subset of set2.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_is_proper_subset_of(set1: *mut u8, set2: *mut u8) -> i64 {
    if set1.is_null() { return if set2.is_null() { 0 } else { 1 }; }
    if set2.is_null() { return 0; }
    let s1 = &*(set1 as *const RtsSet);
    let s2 = &*(set2 as *const RtsSet);
    if s1.is_subset(s2) && s1.len() < s2.len() { 1 } else { 0 }
}

/// Get count of elements in set (for iteration).
#[no_mangle]
pub unsafe extern "C" fn bhc_set_elem_count(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() { return 0; }
    let s = &*(set_ptr as *const RtsSet);
    s.len() as i64
}

/// Get element at index from the set (for iteration).
#[no_mangle]
pub unsafe extern "C" fn bhc_set_elem_at(set_ptr: *mut u8, index: i64) -> i64 {
    if set_ptr.is_null() { return 0; }
    let s = &*(set_ptr as *const RtsSet);
    s.iter().nth(index as usize).copied().unwrap_or(0)
}

/// Find the minimum element of a set. Returns 0 if empty.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_find_min(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() { return 0; }
    let s = &*(set_ptr as *const RtsSet);
    s.iter().next().copied().unwrap_or(0)
}

/// Find the maximum element of a set. Returns 0 if empty.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_find_max(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() { return 0; }
    let s = &*(set_ptr as *const RtsSet);
    s.iter().next_back().copied().unwrap_or(0)
}

/// Delete the minimum element. Returns a new set.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_delete_min(set_ptr: *mut u8) -> *mut u8 {
    if set_ptr.is_null() { return bhc_set_empty(); }
    let mut s = (*(set_ptr as *const RtsSet)).clone();
    if let Some(&min) = s.iter().next() {
        s.remove(&min);
    }
    Box::into_raw(Box::new(s)) as *mut u8
}

/// Delete the maximum element. Returns a new set.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_delete_max(set_ptr: *mut u8) -> *mut u8 {
    if set_ptr.is_null() { return bhc_set_empty(); }
    let mut s = (*(set_ptr as *const RtsSet)).clone();
    if let Some(&max) = s.iter().next_back() {
        s.remove(&max);
    }
    Box::into_raw(Box::new(s)) as *mut u8
}

// ========================================================================
// Data.IntMap operations (identical to Map since Map also uses i64 keys)
// ========================================================================

/// Create an empty IntMap.
#[no_mangle]
pub extern "C" fn bhc_intmap_empty() -> *mut u8 {
    bhc_map_empty()
}

/// Create a singleton IntMap.
#[no_mangle]
pub extern "C" fn bhc_intmap_singleton(key: i64, value: *mut u8) -> *mut u8 {
    bhc_map_singleton(key, value)
}

/// Check if IntMap is empty.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_null(map_ptr: *mut u8) -> i64 {
    bhc_map_null(map_ptr)
}

/// Get IntMap size.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_size(map_ptr: *mut u8) -> i64 {
    bhc_map_size(map_ptr)
}

/// Check IntMap membership.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_member(key: i64, map_ptr: *mut u8) -> i64 {
    bhc_map_member(key, map_ptr)
}

/// IntMap lookup.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_lookup(key: i64, map_ptr: *mut u8) -> *mut u8 {
    bhc_map_lookup(key, map_ptr)
}

/// IntMap findWithDefault.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_find_with_default(default: *mut u8, key: i64, map_ptr: *mut u8) -> *mut u8 {
    bhc_map_find_with_default(default, key, map_ptr)
}

/// IntMap insert.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_insert(key: i64, value: *mut u8, map_ptr: *mut u8) -> *mut u8 {
    bhc_map_insert(key, value, map_ptr)
}

/// IntMap delete.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_delete(key: i64, map_ptr: *mut u8) -> *mut u8 {
    bhc_map_delete(key, map_ptr)
}

/// IntMap union.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_union(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    bhc_map_union(map1, map2)
}

/// IntMap intersection.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_intersection(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    bhc_map_intersection(map1, map2)
}

/// IntMap difference.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_difference(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    bhc_map_difference(map1, map2)
}

/// IntMap keys count.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_keys_count(map_ptr: *mut u8) -> i64 {
    bhc_map_keys_count(map_ptr)
}

/// IntMap key at index.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_key_at(map_ptr: *mut u8, index: i64) -> i64 {
    bhc_map_key_at(map_ptr, index)
}

/// IntMap value at index.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_value_at(map_ptr: *mut u8, index: i64) -> *mut u8 {
    bhc_map_value_at(map_ptr, index)
}

// ========================================================================
// Data.IntSet operations (identical to Set)
// ========================================================================

/// Create an empty IntSet.
#[no_mangle]
pub extern "C" fn bhc_intset_empty() -> *mut u8 {
    bhc_set_empty()
}

/// Create a singleton IntSet.
#[no_mangle]
pub extern "C" fn bhc_intset_singleton(value: i64) -> *mut u8 {
    bhc_set_singleton(value)
}

/// Check if IntSet is empty.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_null(set_ptr: *mut u8) -> i64 {
    bhc_set_null(set_ptr)
}

/// Get IntSet size.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_size(set_ptr: *mut u8) -> i64 {
    bhc_set_size(set_ptr)
}

/// Check IntSet membership.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_member(value: i64, set_ptr: *mut u8) -> i64 {
    bhc_set_member(value, set_ptr)
}

/// IntSet insert.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_insert(value: i64, set_ptr: *mut u8) -> *mut u8 {
    bhc_set_insert(value, set_ptr)
}

/// IntSet delete.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_delete(value: i64, set_ptr: *mut u8) -> *mut u8 {
    bhc_set_delete(value, set_ptr)
}

/// IntSet union.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_union(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    bhc_set_union(set1, set2)
}

/// IntSet intersection.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_intersection(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    bhc_set_intersection(set1, set2)
}

/// IntSet difference.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_difference(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    bhc_set_difference(set1, set2)
}

/// IntSet isSubsetOf.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_is_subset_of(set1: *mut u8, set2: *mut u8) -> i64 {
    bhc_set_is_subset_of(set1, set2)
}

/// IntSet element count.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_elem_count(set_ptr: *mut u8) -> i64 {
    bhc_set_elem_count(set_ptr)
}

/// IntSet element at index.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_elem_at(set_ptr: *mut u8, index: i64) -> i64 {
    bhc_set_elem_at(set_ptr, index)
}
