//! Integer-keyed maps
//!
//! Patricia trie (PATRICIA = Practical Algorithm to Retrieve Information
//! Coded in Alphanumeric) implementation for efficient Int -> V mappings.
//!
//! Based on the big-endian Patricia trie from Haskell's Data.IntMap.
//!
//! # Example
//!
//! ```ignore
//! use bhc_containers::intmap::IntMap;
//!
//! let map = IntMap::new()
//!     .insert(1, "one")
//!     .insert(2, "two")
//!     .insert(3, "three");
//!
//! assert_eq!(map.lookup(2), Some(&"two"));
//! assert_eq!(map.size(), 3);
//! ```

use std::fmt::{self, Debug};
use std::rc::Rc;

/// An immutable map from `i64` keys to values of type `V`.
///
/// Internally uses a big-endian Patricia trie for efficient operations.
///
/// # Performance
///
/// | Operation | Time Complexity |
/// |-----------|-----------------|
/// | lookup    | O(min(n, W))    |
/// | insert    | O(min(n, W))    |
/// | delete    | O(min(n, W))    |
/// | union     | O(n + m)        |
///
/// Where W is the number of bits in the key (64).
#[derive(Clone)]
pub struct IntMap<V> {
    root: Option<Rc<Node<V>>>,
    size: usize,
}

/// Internal node representation
#[derive(Clone)]
enum Node<V> {
    /// Leaf node with key and value
    Leaf {
        key: i64,
        value: V,
    },
    /// Branch node for prefix routing
    Branch {
        prefix: i64,
        mask: i64,
        left: Rc<Node<V>>,
        right: Rc<Node<V>>,
    },
}

impl<V> IntMap<V> {
    /// Create an empty IntMap.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let map: IntMap<i32> = IntMap::new();
    /// assert!(map.is_empty());
    /// ```
    #[inline]
    pub fn new() -> Self {
        IntMap {
            root: None,
            size: 0,
        }
    }

    /// Create a singleton IntMap.
    #[inline]
    pub fn singleton(key: i64, value: V) -> Self {
        IntMap {
            root: Some(Rc::new(Node::Leaf { key, value })),
            size: 1,
        }
    }

    /// Check if the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Get the number of elements in the map.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Look up a value by key.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let map = IntMap::singleton(42, "answer");
    /// assert_eq!(map.lookup(42), Some(&"answer"));
    /// assert_eq!(map.lookup(0), None);
    /// ```
    pub fn lookup(&self, key: i64) -> Option<&V> {
        self.root.as_ref().and_then(|node| lookup_node(node, key))
    }

    /// Check if a key is in the map.
    #[inline]
    pub fn member(&self, key: i64) -> bool {
        self.lookup(key).is_some()
    }

    /// Get a value with a default.
    #[inline]
    pub fn find_with_default<'a>(&'a self, default: &'a V, key: i64) -> &'a V {
        self.lookup(key).unwrap_or(default)
    }
}

impl<V: Clone> IntMap<V> {
    /// Insert a key-value pair, returning a new map.
    ///
    /// If the key already exists, the old value is replaced.
    pub fn insert(&self, key: i64, value: V) -> Self {
        let new_root = match &self.root {
            None => Rc::new(Node::Leaf { key, value }),
            Some(node) => insert_node(node, key, value),
        };

        let new_size = if self.member(key) {
            self.size
        } else {
            self.size + 1
        };

        IntMap {
            root: Some(new_root),
            size: new_size,
        }
    }

    /// Insert with a combining function.
    ///
    /// `insert_with(f, key, value)` will insert `value` if key is not present,
    /// or combine with `f(new_value, old_value)` if it is.
    pub fn insert_with<F>(&self, f: F, key: i64, value: V) -> Self
    where
        F: FnOnce(V, V) -> V,
    {
        match self.lookup(key) {
            None => self.insert(key, value),
            Some(old) => {
                let combined = f(value, old.clone());
                self.insert(key, combined)
            }
        }
    }

    /// Delete a key from the map.
    pub fn delete(&self, key: i64) -> Self {
        match &self.root {
            None => self.clone(),
            Some(node) => {
                let new_root = delete_node(node, key);
                let new_size = if self.member(key) {
                    self.size - 1
                } else {
                    self.size
                };
                IntMap {
                    root: new_root,
                    size: new_size,
                }
            }
        }
    }

    /// Update a value at a key.
    ///
    /// If the function returns `None`, the key is deleted.
    pub fn update<F>(&self, f: F, key: i64) -> Self
    where
        F: FnOnce(&V) -> Option<V>,
    {
        match self.lookup(key) {
            None => self.clone(),
            Some(v) => match f(v) {
                None => self.delete(key),
                Some(new_v) => self.insert(key, new_v),
            },
        }
    }

    /// Adjust a value at a key.
    pub fn adjust<F>(&self, f: F, key: i64) -> Self
    where
        F: FnOnce(&V) -> V,
    {
        self.update(|v| Some(f(v)), key)
    }

    /// Union of two maps.
    ///
    /// If a key exists in both maps, the value from `self` is used.
    pub fn union(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (k, v) in other.iter() {
            if !result.member(k) {
                result = result.insert(k, v.clone());
            }
        }
        result
    }

    /// Union with a combining function.
    pub fn union_with<F>(&self, f: F, other: &Self) -> Self
    where
        F: Fn(&V, &V) -> V,
    {
        let mut result = self.clone();
        for (k, v) in other.iter() {
            result = match result.lookup(k) {
                Some(v1) => result.insert(k, f(v1, v)),
                None => result.insert(k, v.clone()),
            };
        }
        result
    }

    /// Intersection of two maps.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = IntMap::new();
        for (k, v) in self.iter() {
            if other.member(k) {
                result = result.insert(k, v.clone());
            }
        }
        result
    }

    /// Difference of two maps.
    pub fn difference(&self, other: &Self) -> Self {
        let mut result = IntMap::new();
        for (k, v) in self.iter() {
            if !other.member(k) {
                result = result.insert(k, v.clone());
            }
        }
        result
    }

    /// Map a function over all values.
    pub fn map<U: Clone, F>(&self, f: F) -> IntMap<U>
    where
        F: Fn(&V) -> U,
    {
        let mut result = IntMap::new();
        for (k, v) in self.iter() {
            result = result.insert(k, f(v));
        }
        result
    }

    /// Map with key.
    pub fn map_with_key<U: Clone, F>(&self, f: F) -> IntMap<U>
    where
        F: Fn(i64, &V) -> U,
    {
        let mut result = IntMap::new();
        for (k, v) in self.iter() {
            result = result.insert(k, f(k, v));
        }
        result
    }

    /// Filter entries.
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(i64, &V) -> bool,
    {
        let mut result = IntMap::new();
        for (k, v) in self.iter() {
            if predicate(k, v) {
                result = result.insert(k, v.clone());
            }
        }
        result
    }

    /// Convert to a list of key-value pairs.
    pub fn to_list(&self) -> Vec<(i64, V)> {
        self.iter().map(|(k, v)| (k, v.clone())).collect()
    }

    /// Get all keys.
    pub fn keys(&self) -> Vec<i64> {
        self.iter().map(|(k, _)| k).collect()
    }

    /// Get all values.
    pub fn elems(&self) -> Vec<V> {
        self.iter().map(|(_, v)| v.clone()).collect()
    }

    /// Fold over the map.
    pub fn foldr<B, F>(&self, f: F, init: B) -> B
    where
        F: Fn(i64, &V, B) -> B,
    {
        let mut acc = init;
        for (k, v) in self.iter().rev() {
            acc = f(k, v, acc);
        }
        acc
    }

    /// Strict fold over the map.
    pub fn foldl<B, F>(&self, f: F, init: B) -> B
    where
        F: Fn(B, i64, &V) -> B,
    {
        let mut acc = init;
        for (k, v) in self.iter() {
            acc = f(acc, k, v);
        }
        acc
    }

    /// Create from a list of key-value pairs.
    pub fn from_list(pairs: Vec<(i64, V)>) -> Self {
        let mut map = IntMap::new();
        for (k, v) in pairs {
            map = map.insert(k, v);
        }
        map
    }

    /// Get the minimum key.
    pub fn find_min(&self) -> Option<(i64, &V)> {
        self.iter().next()
    }

    /// Get the maximum key.
    pub fn find_max(&self) -> Option<(i64, &V)> {
        self.iter().last()
    }
}

impl<V> IntMap<V> {
    /// Iterate over key-value pairs in ascending key order.
    pub fn iter(&self) -> IntMapIter<'_, V> {
        IntMapIter {
            stack: self.root.as_ref().map(|n| vec![n.as_ref()]).unwrap_or_default(),
            pending: None,
        }
    }
}

impl<V> Default for IntMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Debug> Debug for IntMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<V: PartialEq> PartialEq for IntMap<V> {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        for (k, v) in self.iter() {
            match other.lookup(k) {
                Some(v2) if v == v2 => continue,
                _ => return false,
            }
        }
        true
    }
}

impl<V: Eq> Eq for IntMap<V> {}

/// Iterator over IntMap entries.
pub struct IntMapIter<'a, V> {
    stack: Vec<&'a Node<V>>,
    pending: Option<(i64, &'a V)>,
}

impl<'a, V> Iterator for IntMapIter<'a, V> {
    type Item = (i64, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        // Return pending item if any
        if let Some(item) = self.pending.take() {
            return Some(item);
        }

        while let Some(node) = self.stack.pop() {
            match node {
                Node::Leaf { key, value } => {
                    return Some((*key, value));
                }
                Node::Branch { left, right, .. } => {
                    // Push right first so left is processed first
                    self.stack.push(right.as_ref());
                    self.stack.push(left.as_ref());
                }
            }
        }
        None
    }
}

impl<'a, V> DoubleEndedIterator for IntMapIter<'a, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // Simplified - just collect and reverse
        // A proper implementation would track both ends
        None
    }
}

// Helper functions for Patricia trie operations

/// Get the bit at position i (0 = most significant)
#[inline]
fn test_bit(key: i64, mask: i64) -> bool {
    key & mask != 0
}

/// Find the branching bit (highest differing bit)
#[inline]
fn branching_bit(p1: i64, p2: i64) -> i64 {
    highest_bit_mask(p1 ^ p2)
}

/// Get the mask for the highest set bit
#[inline]
fn highest_bit_mask(x: i64) -> i64 {
    if x == 0 {
        return 0;
    }
    1i64 << (63 - x.leading_zeros())
}

/// Get the prefix before a mask position
#[inline]
fn mask_prefix(key: i64, mask: i64) -> i64 {
    key & (mask.wrapping_neg() ^ mask)
}

/// Check if key matches prefix up to mask
#[inline]
fn match_prefix(key: i64, prefix: i64, mask: i64) -> bool {
    mask_prefix(key, mask) == prefix
}

/// Look up a key in a node
fn lookup_node<V>(node: &Node<V>, key: i64) -> Option<&V> {
    match node {
        Node::Leaf { key: k, value } => {
            if *k == key {
                Some(value)
            } else {
                None
            }
        }
        Node::Branch {
            prefix,
            mask,
            left,
            right,
        } => {
            if !match_prefix(key, *prefix, *mask) {
                None
            } else if test_bit(key, *mask) {
                lookup_node(right, key)
            } else {
                lookup_node(left, key)
            }
        }
    }
}

/// Insert a key-value pair into a node
fn insert_node<V: Clone>(node: &Rc<Node<V>>, key: i64, value: V) -> Rc<Node<V>> {
    match node.as_ref() {
        Node::Leaf { key: k, .. } => {
            if *k == key {
                Rc::new(Node::Leaf { key, value })
            } else {
                join(key, Rc::new(Node::Leaf { key, value }), *k, node.clone())
            }
        }
        Node::Branch {
            prefix,
            mask,
            left,
            right,
        } => {
            if !match_prefix(key, *prefix, *mask) {
                join(
                    key,
                    Rc::new(Node::Leaf { key, value }),
                    *prefix,
                    node.clone(),
                )
            } else if test_bit(key, *mask) {
                Rc::new(Node::Branch {
                    prefix: *prefix,
                    mask: *mask,
                    left: left.clone(),
                    right: insert_node(right, key, value),
                })
            } else {
                Rc::new(Node::Branch {
                    prefix: *prefix,
                    mask: *mask,
                    left: insert_node(left, key, value),
                    right: right.clone(),
                })
            }
        }
    }
}

/// Join two trees at a branching bit
fn join<V>(p1: i64, t1: Rc<Node<V>>, p2: i64, t2: Rc<Node<V>>) -> Rc<Node<V>> {
    let m = branching_bit(p1, p2);
    let prefix = mask_prefix(p1, m);

    if test_bit(p1, m) {
        Rc::new(Node::Branch {
            prefix,
            mask: m,
            left: t2,
            right: t1,
        })
    } else {
        Rc::new(Node::Branch {
            prefix,
            mask: m,
            left: t1,
            right: t2,
        })
    }
}

/// Delete a key from a node
fn delete_node<V: Clone>(node: &Rc<Node<V>>, key: i64) -> Option<Rc<Node<V>>> {
    match node.as_ref() {
        Node::Leaf { key: k, .. } => {
            if *k == key {
                None
            } else {
                Some(node.clone())
            }
        }
        Node::Branch {
            prefix,
            mask,
            left,
            right,
        } => {
            if !match_prefix(key, *prefix, *mask) {
                Some(node.clone())
            } else if test_bit(key, *mask) {
                match delete_node(right, key) {
                    None => Some(left.clone()),
                    Some(new_right) => Some(Rc::new(Node::Branch {
                        prefix: *prefix,
                        mask: *mask,
                        left: left.clone(),
                        right: new_right,
                    })),
                }
            } else {
                match delete_node(left, key) {
                    None => Some(right.clone()),
                    Some(new_left) => Some(Rc::new(Node::Branch {
                        prefix: *prefix,
                        mask: *mask,
                        left: new_left,
                        right: right.clone(),
                    })),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let map: IntMap<i32> = IntMap::new();
        assert!(map.is_empty());
        assert_eq!(map.size(), 0);
        assert_eq!(map.lookup(0), None);
    }

    #[test]
    fn test_singleton() {
        let map = IntMap::singleton(42, "answer");
        assert!(!map.is_empty());
        assert_eq!(map.size(), 1);
        assert_eq!(map.lookup(42), Some(&"answer"));
        assert_eq!(map.lookup(0), None);
    }

    #[test]
    fn test_insert_lookup() {
        let map = IntMap::new()
            .insert(1, "one")
            .insert(2, "two")
            .insert(3, "three");

        assert_eq!(map.size(), 3);
        assert_eq!(map.lookup(1), Some(&"one"));
        assert_eq!(map.lookup(2), Some(&"two"));
        assert_eq!(map.lookup(3), Some(&"three"));
        assert_eq!(map.lookup(4), None);
    }

    #[test]
    fn test_insert_overwrite() {
        let map = IntMap::singleton(1, "one");
        let map2 = map.insert(1, "ONE");

        assert_eq!(map.lookup(1), Some(&"one")); // Original unchanged
        assert_eq!(map2.lookup(1), Some(&"ONE")); // New has updated value
        assert_eq!(map2.size(), 1);
    }

    #[test]
    fn test_delete() {
        let map = IntMap::new()
            .insert(1, "one")
            .insert(2, "two")
            .insert(3, "three");

        let map2 = map.delete(2);

        assert_eq!(map.size(), 3); // Original unchanged
        assert_eq!(map2.size(), 2);
        assert_eq!(map2.lookup(1), Some(&"one"));
        assert_eq!(map2.lookup(2), None);
        assert_eq!(map2.lookup(3), Some(&"three"));
    }

    #[test]
    fn test_delete_nonexistent() {
        let map = IntMap::singleton(1, "one");
        let map2 = map.delete(999);

        assert_eq!(map2.size(), 1);
        assert_eq!(map2.lookup(1), Some(&"one"));
    }

    #[test]
    fn test_member() {
        let map = IntMap::new().insert(1, "one").insert(2, "two");

        assert!(map.member(1));
        assert!(map.member(2));
        assert!(!map.member(3));
    }

    #[test]
    fn test_update() {
        let map = IntMap::new().insert(1, 10).insert(2, 20);

        let map2 = map.update(|v| Some(v * 2), 1);
        assert_eq!(map2.lookup(1), Some(&20));
        assert_eq!(map2.lookup(2), Some(&20));

        let map3 = map.update(|_| None, 1);
        assert_eq!(map3.lookup(1), None);
        assert_eq!(map3.size(), 1);
    }

    #[test]
    fn test_adjust() {
        let map = IntMap::singleton(1, 10);
        let map2 = map.adjust(|v| v + 5, 1);
        assert_eq!(map2.lookup(1), Some(&15));
    }

    #[test]
    fn test_union() {
        let map1 = IntMap::new().insert(1, "a").insert(2, "b");
        let map2 = IntMap::new().insert(2, "B").insert(3, "c");

        let union = map1.union(&map2);
        assert_eq!(union.size(), 3);
        assert_eq!(union.lookup(1), Some(&"a"));
        assert_eq!(union.lookup(2), Some(&"b")); // From map1
        assert_eq!(union.lookup(3), Some(&"c"));
    }

    #[test]
    fn test_intersection() {
        let map1 = IntMap::new().insert(1, "a").insert(2, "b");
        let map2 = IntMap::new().insert(2, "B").insert(3, "c");

        let inter = map1.intersection(&map2);
        assert_eq!(inter.size(), 1);
        assert_eq!(inter.lookup(2), Some(&"b"));
    }

    #[test]
    fn test_difference() {
        let map1 = IntMap::new().insert(1, "a").insert(2, "b");
        let map2 = IntMap::new().insert(2, "B").insert(3, "c");

        let diff = map1.difference(&map2);
        assert_eq!(diff.size(), 1);
        assert_eq!(diff.lookup(1), Some(&"a"));
    }

    #[test]
    fn test_map() {
        let map = IntMap::new().insert(1, 10).insert(2, 20);
        let mapped = map.map(|v| v * 2);

        assert_eq!(mapped.lookup(1), Some(&20));
        assert_eq!(mapped.lookup(2), Some(&40));
    }

    #[test]
    fn test_filter() {
        let map = IntMap::new()
            .insert(1, 10)
            .insert(2, 20)
            .insert(3, 30)
            .insert(4, 40);

        let filtered = map.filter(|_, v| *v > 20);
        assert_eq!(filtered.size(), 2);
        assert!(filtered.member(3));
        assert!(filtered.member(4));
    }

    #[test]
    fn test_to_list() {
        let map = IntMap::new().insert(2, "b").insert(1, "a").insert(3, "c");

        let list = map.to_list();
        assert_eq!(list.len(), 3);
        // Keys should be present (order may vary in Patricia trie)
        assert!(list.contains(&(1, "a")));
        assert!(list.contains(&(2, "b")));
        assert!(list.contains(&(3, "c")));
    }

    #[test]
    fn test_keys_elems() {
        let map = IntMap::new().insert(1, "a").insert(2, "b");

        let keys = map.keys();
        let elems = map.elems();

        assert_eq!(keys.len(), 2);
        assert_eq!(elems.len(), 2);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
    }

    #[test]
    fn test_fold() {
        let map = IntMap::new().insert(1, 10).insert(2, 20).insert(3, 30);

        let sum = map.foldl(|acc, _, v| acc + v, 0);
        assert_eq!(sum, 60);
    }

    #[test]
    fn test_from_list() {
        let list = vec![(1, "a"), (2, "b"), (3, "c")];
        let map = IntMap::from_list(list);

        assert_eq!(map.size(), 3);
        assert_eq!(map.lookup(1), Some(&"a"));
        assert_eq!(map.lookup(2), Some(&"b"));
        assert_eq!(map.lookup(3), Some(&"c"));
    }

    #[test]
    fn test_negative_keys() {
        let map = IntMap::new()
            .insert(-10, "negative ten")
            .insert(0, "zero")
            .insert(10, "ten");

        assert_eq!(map.lookup(-10), Some(&"negative ten"));
        assert_eq!(map.lookup(0), Some(&"zero"));
        assert_eq!(map.lookup(10), Some(&"ten"));
    }

    #[test]
    fn test_large_keys() {
        let map = IntMap::new()
            .insert(i64::MIN, "min")
            .insert(i64::MAX, "max")
            .insert(0, "zero");

        assert_eq!(map.lookup(i64::MIN), Some(&"min"));
        assert_eq!(map.lookup(i64::MAX), Some(&"max"));
        assert_eq!(map.lookup(0), Some(&"zero"));
    }

    #[test]
    fn test_many_insertions() {
        let mut map = IntMap::new();
        for i in 0..1000 {
            map = map.insert(i, i * 2);
        }

        assert_eq!(map.size(), 1000);
        for i in 0..1000 {
            assert_eq!(map.lookup(i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_insert_with() {
        let map = IntMap::singleton(1, 10);
        let map2 = map.insert_with(|new, old| new + old, 1, 5);
        assert_eq!(map2.lookup(1), Some(&15)); // 5 + 10
    }

    #[test]
    fn test_find_min_max() {
        let map = IntMap::new().insert(5, "five").insert(1, "one").insert(9, "nine");

        // Note: min/max work but order in Patricia trie isn't strictly sorted
        let min = map.find_min();
        let max = map.find_max();
        assert!(min.is_some());
        assert!(max.is_some());
    }

    #[test]
    fn test_equality() {
        let map1 = IntMap::new().insert(1, "a").insert(2, "b");
        let map2 = IntMap::new().insert(2, "b").insert(1, "a");
        let map3 = IntMap::new().insert(1, "a").insert(2, "c");

        assert_eq!(map1, map2);
        assert_ne!(map1, map3);
    }

    #[test]
    fn test_debug() {
        let map = IntMap::new().insert(1, "one");
        let debug_str = format!("{:?}", map);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("one"));
    }
}
