//! Integer sets (Data.IntSet)
//!
//! Efficient sets of integers using Patricia tries (radix trees).
//! This provides O(min(n, W)) lookup, insert, and delete operations
//! where W is the number of bits in an Int (64).
//!
//! # Example
//!
//! ```ignore
//! use bhc_containers::intset::IntSet;
//!
//! let s1 = IntSet::from_list(&[1, 2, 3, 4, 5]);
//! let s2 = IntSet::from_list(&[3, 4, 5, 6, 7]);
//!
//! let union = s1.union(&s2);
//! let intersection = s1.intersection(&s2);
//! ```

use std::fmt::{self, Debug};
use std::rc::Rc;

// ============================================================
// Core Type
// ============================================================

/// An efficient set of integers.
///
/// This implementation uses a big-endian Patricia trie, which provides:
/// - O(min(n, W)) lookup, insert, delete (W = 64 bits)
/// - O(n + m) union, intersection, difference
/// - Naturally sorted iteration
#[derive(Clone)]
pub struct IntSet {
    root: Option<Rc<Node>>,
}

// Internal node representation
#[derive(Clone)]
enum Node {
    // A tip contains a single element
    Tip {
        key: i64,
    },
    // A branch node
    Bin {
        prefix: i64,
        mask: i64,
        left: Rc<Node>,
        right: Rc<Node>,
    },
}

// ============================================================
// Construction
// ============================================================

impl IntSet {
    /// Create an empty set.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s: IntSet = IntSet::new();
    /// assert!(s.is_empty());
    /// ```
    pub fn new() -> Self {
        IntSet { root: None }
    }

    /// Create a set with a single element.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s = IntSet::singleton(42);
    /// assert!(s.member(42));
    /// ```
    pub fn singleton(key: i64) -> Self {
        IntSet {
            root: Some(Rc::new(Node::Tip { key })),
        }
    }

    /// Create a set from a list of elements.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s = IntSet::from_list(&[1, 2, 3]);
    /// assert_eq!(s.size(), 3);
    /// ```
    pub fn from_list(elems: &[i64]) -> Self {
        let mut set = IntSet::new();
        for &elem in elems {
            set = set.insert(elem);
        }
        set
    }
}

impl Default for IntSet {
    fn default() -> Self {
        IntSet::new()
    }
}

// ============================================================
// Query
// ============================================================

impl IntSet {
    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Get the number of elements in the set.
    ///
    /// Time: O(n)
    pub fn size(&self) -> usize {
        match &self.root {
            None => 0,
            Some(node) => size_node(node),
        }
    }

    /// Check if an element is in the set.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s = IntSet::from_list(&[1, 2, 3]);
    /// assert!(s.member(2));
    /// assert!(!s.member(5));
    /// ```
    pub fn member(&self, key: i64) -> bool {
        match &self.root {
            None => false,
            Some(node) => member_node(node, key),
        }
    }

    /// Check if an element is not in the set.
    pub fn not_member(&self, key: i64) -> bool {
        !self.member(key)
    }

    /// Find the minimum element.
    ///
    /// Returns `None` for empty sets.
    pub fn find_min(&self) -> Option<i64> {
        match &self.root {
            None => None,
            Some(node) => Some(find_min_node(node)),
        }
    }

    /// Find the maximum element.
    ///
    /// Returns `None` for empty sets.
    pub fn find_max(&self) -> Option<i64> {
        match &self.root {
            None => None,
            Some(node) => Some(find_max_node(node)),
        }
    }

    /// Check if this set is a subset of another.
    pub fn is_subset_of(&self, other: &IntSet) -> bool {
        self.size() <= other.size() && self.to_list().iter().all(|k| other.member(*k))
    }

    /// Check if this set is a proper subset of another.
    pub fn is_proper_subset_of(&self, other: &IntSet) -> bool {
        self.size() < other.size() && self.is_subset_of(other)
    }

    /// Check if two sets are disjoint.
    pub fn disjoint(&self, other: &IntSet) -> bool {
        self.intersection(other).is_empty()
    }
}

fn size_node(node: &Node) -> usize {
    match node {
        Node::Tip { .. } => 1,
        Node::Bin { left, right, .. } => size_node(left) + size_node(right),
    }
}

fn member_node(node: &Node, key: i64) -> bool {
    match node {
        Node::Tip { key: k } => *k == key,
        Node::Bin {
            prefix,
            mask,
            left,
            right,
        } => {
            if !match_prefix(key, *prefix, *mask) {
                false
            } else if test_bit(key, *mask) {
                member_node(right, key)
            } else {
                member_node(left, key)
            }
        }
    }
}

fn find_min_node(node: &Node) -> i64 {
    match node {
        Node::Tip { key } => *key,
        Node::Bin { left, .. } => find_min_node(left),
    }
}

fn find_max_node(node: &Node) -> i64 {
    match node {
        Node::Tip { key } => *key,
        Node::Bin { right, .. } => find_max_node(right),
    }
}

// ============================================================
// Insertion / Deletion
// ============================================================

impl IntSet {
    /// Insert an element into the set.
    ///
    /// If the element is already present, the set is returned unchanged.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s = IntSet::new().insert(1).insert(2).insert(3);
    /// assert_eq!(s.size(), 3);
    /// ```
    pub fn insert(&self, key: i64) -> IntSet {
        match &self.root {
            None => IntSet::singleton(key),
            Some(node) => IntSet {
                root: Some(insert_node(node, key)),
            },
        }
    }

    /// Delete an element from the set.
    ///
    /// If the element is not present, the set is returned unchanged.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s = IntSet::from_list(&[1, 2, 3]);
    /// let s2 = s.delete(2);
    /// assert_eq!(s2.size(), 2);
    /// assert!(!s2.member(2));
    /// ```
    pub fn delete(&self, key: i64) -> IntSet {
        match &self.root {
            None => IntSet::new(),
            Some(node) => IntSet {
                root: delete_node(node, key),
            },
        }
    }
}

fn insert_node(node: &Rc<Node>, key: i64) -> Rc<Node> {
    match node.as_ref() {
        Node::Tip { key: k } => {
            if *k == key {
                Rc::clone(node)
            } else {
                join(key, Rc::new(Node::Tip { key }), *k, Rc::clone(node))
            }
        }
        Node::Bin {
            prefix,
            mask,
            left,
            right,
        } => {
            if !match_prefix(key, *prefix, *mask) {
                join(
                    key,
                    Rc::new(Node::Tip { key }),
                    *prefix,
                    Rc::clone(node),
                )
            } else if test_bit(key, *mask) {
                let new_right = insert_node(right, key);
                if Rc::ptr_eq(&new_right, right) {
                    Rc::clone(node)
                } else {
                    Rc::new(Node::Bin {
                        prefix: *prefix,
                        mask: *mask,
                        left: Rc::clone(left),
                        right: new_right,
                    })
                }
            } else {
                let new_left = insert_node(left, key);
                if Rc::ptr_eq(&new_left, left) {
                    Rc::clone(node)
                } else {
                    Rc::new(Node::Bin {
                        prefix: *prefix,
                        mask: *mask,
                        left: new_left,
                        right: Rc::clone(right),
                    })
                }
            }
        }
    }
}

fn delete_node(node: &Rc<Node>, key: i64) -> Option<Rc<Node>> {
    match node.as_ref() {
        Node::Tip { key: k } => {
            if *k == key {
                None
            } else {
                Some(Rc::clone(node))
            }
        }
        Node::Bin {
            prefix,
            mask,
            left,
            right,
        } => {
            if !match_prefix(key, *prefix, *mask) {
                Some(Rc::clone(node))
            } else if test_bit(key, *mask) {
                match delete_node(right, key) {
                    None => Some(Rc::clone(left)),
                    Some(new_right) => {
                        if Rc::ptr_eq(&new_right, right) {
                            Some(Rc::clone(node))
                        } else {
                            Some(Rc::new(Node::Bin {
                                prefix: *prefix,
                                mask: *mask,
                                left: Rc::clone(left),
                                right: new_right,
                            }))
                        }
                    }
                }
            } else {
                match delete_node(left, key) {
                    None => Some(Rc::clone(right)),
                    Some(new_left) => {
                        if Rc::ptr_eq(&new_left, left) {
                            Some(Rc::clone(node))
                        } else {
                            Some(Rc::new(Node::Bin {
                                prefix: *prefix,
                                mask: *mask,
                                left: new_left,
                                right: Rc::clone(right),
                            }))
                        }
                    }
                }
            }
        }
    }
}

// ============================================================
// Set Operations
// ============================================================

impl IntSet {
    /// Union of two sets.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s1 = IntSet::from_list(&[1, 2, 3]);
    /// let s2 = IntSet::from_list(&[3, 4, 5]);
    /// let union = s1.union(&s2);
    /// assert_eq!(union.size(), 5);
    /// ```
    pub fn union(&self, other: &IntSet) -> IntSet {
        match (&self.root, &other.root) {
            (None, _) => other.clone(),
            (_, None) => self.clone(),
            (Some(n1), Some(n2)) => IntSet {
                root: Some(union_node(n1, n2)),
            },
        }
    }

    /// Intersection of two sets.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s1 = IntSet::from_list(&[1, 2, 3]);
    /// let s2 = IntSet::from_list(&[2, 3, 4]);
    /// let inter = s1.intersection(&s2);
    /// assert_eq!(inter.to_list(), vec![2, 3]);
    /// ```
    pub fn intersection(&self, other: &IntSet) -> IntSet {
        match (&self.root, &other.root) {
            (None, _) | (_, None) => IntSet::new(),
            (Some(n1), Some(n2)) => IntSet {
                root: intersection_node(n1, n2),
            },
        }
    }

    /// Difference of two sets.
    ///
    /// Returns elements in self but not in other.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s1 = IntSet::from_list(&[1, 2, 3, 4]);
    /// let s2 = IntSet::from_list(&[2, 3]);
    /// let diff = s1.difference(&s2);
    /// assert_eq!(diff.to_list(), vec![1, 4]);
    /// ```
    pub fn difference(&self, other: &IntSet) -> IntSet {
        match (&self.root, &other.root) {
            (None, _) => IntSet::new(),
            (_, None) => self.clone(),
            (Some(n1), Some(n2)) => IntSet {
                root: difference_node(n1, n2),
            },
        }
    }

    /// Symmetric difference of two sets.
    ///
    /// Returns elements in either set but not both.
    pub fn symmetric_difference(&self, other: &IntSet) -> IntSet {
        self.difference(other).union(&other.difference(self))
    }
}

fn union_node(n1: &Rc<Node>, n2: &Rc<Node>) -> Rc<Node> {
    match (n1.as_ref(), n2.as_ref()) {
        (Node::Tip { key }, _) => insert_node(n2, *key),
        (_, Node::Tip { key }) => insert_node(n1, *key),
        (
            Node::Bin {
                prefix: p1,
                mask: m1,
                left: l1,
                right: r1,
            },
            Node::Bin {
                prefix: p2,
                mask: m2,
                left: l2,
                right: r2,
            },
        ) => {
            if *m1 == *m2 && *p1 == *p2 {
                // Same prefix and mask - union children
                Rc::new(Node::Bin {
                    prefix: *p1,
                    mask: *m1,
                    left: union_node(l1, l2),
                    right: union_node(r1, r2),
                })
            } else if *m1 > *m2 && match_prefix(*p2, *p1, *m1) {
                // p2 is more specific, merge n2 into appropriate branch of n1
                if test_bit(*p2, *m1) {
                    Rc::new(Node::Bin {
                        prefix: *p1,
                        mask: *m1,
                        left: Rc::clone(l1),
                        right: union_node(r1, n2),
                    })
                } else {
                    Rc::new(Node::Bin {
                        prefix: *p1,
                        mask: *m1,
                        left: union_node(l1, n2),
                        right: Rc::clone(r1),
                    })
                }
            } else if *m2 > *m1 && match_prefix(*p1, *p2, *m2) {
                // p1 is more specific, merge n1 into appropriate branch of n2
                if test_bit(*p1, *m2) {
                    Rc::new(Node::Bin {
                        prefix: *p2,
                        mask: *m2,
                        left: Rc::clone(l2),
                        right: union_node(n1, r2),
                    })
                } else {
                    Rc::new(Node::Bin {
                        prefix: *p2,
                        mask: *m2,
                        left: union_node(n1, l2),
                        right: Rc::clone(r2),
                    })
                }
            } else {
                // Disjoint prefixes - join them
                join(*p1, Rc::clone(n1), *p2, Rc::clone(n2))
            }
        }
    }
}

fn intersection_node(n1: &Rc<Node>, n2: &Rc<Node>) -> Option<Rc<Node>> {
    match (n1.as_ref(), n2.as_ref()) {
        (Node::Tip { key }, _) => {
            if member_node(n2, *key) {
                Some(Rc::clone(n1))
            } else {
                None
            }
        }
        (_, Node::Tip { key }) => {
            if member_node(n1, *key) {
                Some(Rc::clone(n2))
            } else {
                None
            }
        }
        (
            Node::Bin {
                prefix: p1,
                mask: m1,
                left: l1,
                right: r1,
            },
            Node::Bin {
                prefix: p2,
                mask: m2,
                left: l2,
                right: r2,
            },
        ) => {
            if *m1 == *m2 && *p1 == *p2 {
                bin_maybe(*p1, *m1, intersection_node(l1, l2), intersection_node(r1, r2))
            } else if *m1 > *m2 && match_prefix(*p2, *p1, *m1) {
                if test_bit(*p2, *m1) {
                    intersection_node(r1, n2)
                } else {
                    intersection_node(l1, n2)
                }
            } else if *m2 > *m1 && match_prefix(*p1, *p2, *m2) {
                if test_bit(*p1, *m2) {
                    intersection_node(n1, r2)
                } else {
                    intersection_node(n1, l2)
                }
            } else {
                None
            }
        }
    }
}

fn difference_node(n1: &Rc<Node>, n2: &Rc<Node>) -> Option<Rc<Node>> {
    match (n1.as_ref(), n2.as_ref()) {
        (Node::Tip { key }, _) => {
            if member_node(n2, *key) {
                None
            } else {
                Some(Rc::clone(n1))
            }
        }
        (_, Node::Tip { key }) => delete_node(n1, *key),
        (
            Node::Bin {
                prefix: p1,
                mask: m1,
                left: l1,
                right: r1,
            },
            Node::Bin {
                prefix: p2,
                mask: m2,
                left: l2,
                right: r2,
            },
        ) => {
            if *m1 == *m2 && *p1 == *p2 {
                bin_maybe(*p1, *m1, difference_node(l1, l2), difference_node(r1, r2))
            } else if *m1 > *m2 && match_prefix(*p2, *p1, *m1) {
                if test_bit(*p2, *m1) {
                    bin_maybe(*p1, *m1, Some(Rc::clone(l1)), difference_node(r1, n2))
                } else {
                    bin_maybe(*p1, *m1, difference_node(l1, n2), Some(Rc::clone(r1)))
                }
            } else if *m2 > *m1 && match_prefix(*p1, *p2, *m2) {
                if test_bit(*p1, *m2) {
                    difference_node(n1, r2)
                } else {
                    difference_node(n1, l2)
                }
            } else {
                Some(Rc::clone(n1))
            }
        }
    }
}

fn bin_maybe(
    prefix: i64,
    mask: i64,
    left: Option<Rc<Node>>,
    right: Option<Rc<Node>>,
) -> Option<Rc<Node>> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(l), Some(r)) => Some(Rc::new(Node::Bin {
            prefix,
            mask,
            left: l,
            right: r,
        })),
    }
}

// ============================================================
// Map / Filter / Fold
// ============================================================

impl IntSet {
    /// Map a function over all elements.
    ///
    /// Note: This may produce a smaller set if the function maps
    /// multiple elements to the same value.
    pub fn map<F>(&self, f: F) -> IntSet
    where
        F: Fn(i64) -> i64,
    {
        IntSet::from_list(&self.to_list().iter().map(|&x| f(x)).collect::<Vec<_>>())
    }

    /// Filter elements by a predicate.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s = IntSet::from_list(&[1, 2, 3, 4, 5]);
    /// let evens = s.filter(|x| x % 2 == 0);
    /// assert_eq!(evens.to_list(), vec![2, 4]);
    /// ```
    pub fn filter<F>(&self, predicate: F) -> IntSet
    where
        F: Fn(&i64) -> bool,
    {
        match &self.root {
            None => IntSet::new(),
            Some(node) => IntSet {
                root: filter_node(node, &predicate),
            },
        }
    }

    /// Partition elements by a predicate.
    ///
    /// Returns (elements satisfying predicate, elements not satisfying).
    pub fn partition<F>(&self, predicate: F) -> (IntSet, IntSet)
    where
        F: Fn(&i64) -> bool,
    {
        let mut yes = IntSet::new();
        let mut no = IntSet::new();
        for elem in self.to_list() {
            if predicate(&elem) {
                yes = yes.insert(elem);
            } else {
                no = no.insert(elem);
            }
        }
        (yes, no)
    }

    /// Fold over elements in ascending order.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s = IntSet::from_list(&[1, 2, 3, 4, 5]);
    /// let sum = s.fold(0, |acc, x| acc + x);
    /// assert_eq!(sum, 15);
    /// ```
    pub fn fold<B, F>(&self, init: B, f: F) -> B
    where
        F: Fn(B, i64) -> B,
    {
        self.to_list().into_iter().fold(init, f)
    }

    /// Fold over elements in descending order.
    pub fn fold_right<B, F>(&self, init: B, f: F) -> B
    where
        F: Fn(i64, B) -> B,
    {
        self.to_list().into_iter().rev().fold(init, |acc, x| f(x, acc))
    }
}

fn filter_node<F>(node: &Rc<Node>, predicate: &F) -> Option<Rc<Node>>
where
    F: Fn(&i64) -> bool,
{
    match node.as_ref() {
        Node::Tip { key } => {
            if predicate(key) {
                Some(Rc::clone(node))
            } else {
                None
            }
        }
        Node::Bin {
            prefix,
            mask,
            left,
            right,
        } => bin_maybe(
            *prefix,
            *mask,
            filter_node(left, predicate),
            filter_node(right, predicate),
        ),
    }
}

// ============================================================
// Conversion
// ============================================================

impl IntSet {
    /// Convert to a sorted list of elements.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let s = IntSet::from_list(&[3, 1, 4, 1, 5]);
    /// assert_eq!(s.to_list(), vec![1, 3, 4, 5]);
    /// ```
    pub fn to_list(&self) -> Vec<i64> {
        let mut result = Vec::new();
        if let Some(node) = &self.root {
            to_list_node(node, &mut result);
        }
        result.sort();
        result
    }

    /// Convert to an ascending list.
    pub fn to_asc_list(&self) -> Vec<i64> {
        self.to_list()
    }

    /// Convert to a descending list.
    pub fn to_desc_list(&self) -> Vec<i64> {
        let mut list = self.to_list();
        list.reverse();
        list
    }
}

fn to_list_node(node: &Node, result: &mut Vec<i64>) {
    match node {
        Node::Tip { key } => result.push(*key),
        Node::Bin { left, right, .. } => {
            to_list_node(left, result);
            to_list_node(right, result);
        }
    }
}

// ============================================================
// Iterator
// ============================================================

impl IntSet {
    /// Create an iterator over elements in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.to_list().into_iter()
    }
}

impl IntoIterator for IntSet {
    type Item = i64;
    type IntoIter = std::vec::IntoIter<i64>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_list().into_iter()
    }
}

impl<'a> IntoIterator for &'a IntSet {
    type Item = i64;
    type IntoIter = std::vec::IntoIter<i64>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_list().into_iter()
    }
}

// ============================================================
// Helper Functions
// ============================================================

/// Test if a bit is set
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

/// Join two nodes with different prefixes
fn join(p1: i64, t1: Rc<Node>, p2: i64, t2: Rc<Node>) -> Rc<Node> {
    let mask = branching_bit(p1, p2);
    let prefix = mask_prefix(p1, mask);
    if test_bit(p1, mask) {
        Rc::new(Node::Bin {
            prefix,
            mask,
            left: t2,
            right: t1,
        })
    } else {
        Rc::new(Node::Bin {
            prefix,
            mask,
            left: t1,
            right: t2,
        })
    }
}

// ============================================================
// Trait Implementations
// ============================================================

impl PartialEq for IntSet {
    fn eq(&self, other: &Self) -> bool {
        self.to_list() == other.to_list()
    }
}

impl Eq for IntSet {}

impl Debug for IntSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.to_list()).finish()
    }
}

impl FromIterator<i64> for IntSet {
    fn from_iter<T: IntoIterator<Item = i64>>(iter: T) -> Self {
        let elems: Vec<_> = iter.into_iter().collect();
        IntSet::from_list(&elems)
    }
}

impl Extend<i64> for IntSet {
    fn extend<T: IntoIterator<Item = i64>>(&mut self, iter: T) {
        for elem in iter {
            *self = self.insert(elem);
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let s: IntSet = IntSet::new();
        assert!(s.is_empty());
        assert_eq!(s.size(), 0);
    }

    #[test]
    fn test_singleton() {
        let s = IntSet::singleton(42);
        assert!(!s.is_empty());
        assert_eq!(s.size(), 1);
        assert!(s.member(42));
        assert!(!s.member(0));
    }

    #[test]
    fn test_insert() {
        let s = IntSet::new().insert(1).insert(2).insert(3);
        assert_eq!(s.size(), 3);
        assert!(s.member(1));
        assert!(s.member(2));
        assert!(s.member(3));
        assert!(!s.member(4));
    }

    #[test]
    fn test_insert_duplicate() {
        let s = IntSet::new().insert(1).insert(1).insert(1);
        assert_eq!(s.size(), 1);
    }

    #[test]
    fn test_delete() {
        let s = IntSet::from_list(&[1, 2, 3, 4, 5]);
        let s2 = s.delete(3);
        assert_eq!(s2.size(), 4);
        assert!(!s2.member(3));
        assert!(s2.member(1));
        assert!(s2.member(2));
        assert!(s2.member(4));
        assert!(s2.member(5));
    }

    #[test]
    fn test_delete_nonexistent() {
        let s = IntSet::from_list(&[1, 2, 3]);
        let s2 = s.delete(99);
        assert_eq!(s.to_list(), s2.to_list());
    }

    #[test]
    fn test_from_list() {
        let s = IntSet::from_list(&[3, 1, 4, 1, 5, 9, 2, 6]);
        assert_eq!(s.size(), 7); // Duplicate 1 removed
        assert_eq!(s.to_list(), vec![1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_union() {
        let s1 = IntSet::from_list(&[1, 2, 3]);
        let s2 = IntSet::from_list(&[3, 4, 5]);
        let union = s1.union(&s2);
        assert_eq!(union.to_list(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_intersection() {
        let s1 = IntSet::from_list(&[1, 2, 3, 4]);
        let s2 = IntSet::from_list(&[2, 3, 4, 5]);
        let inter = s1.intersection(&s2);
        assert_eq!(inter.to_list(), vec![2, 3, 4]);
    }

    #[test]
    fn test_difference() {
        let s1 = IntSet::from_list(&[1, 2, 3, 4]);
        let s2 = IntSet::from_list(&[2, 3]);
        let diff = s1.difference(&s2);
        assert_eq!(diff.to_list(), vec![1, 4]);
    }

    #[test]
    fn test_symmetric_difference() {
        let s1 = IntSet::from_list(&[1, 2, 3]);
        let s2 = IntSet::from_list(&[2, 3, 4]);
        let sym = s1.symmetric_difference(&s2);
        assert_eq!(sym.to_list(), vec![1, 4]);
    }

    #[test]
    fn test_filter() {
        let s = IntSet::from_list(&[1, 2, 3, 4, 5, 6]);
        let evens = s.filter(|x| x % 2 == 0);
        assert_eq!(evens.to_list(), vec![2, 4, 6]);
    }

    #[test]
    fn test_partition() {
        let s = IntSet::from_list(&[1, 2, 3, 4, 5, 6]);
        let (evens, odds) = s.partition(|x| x % 2 == 0);
        assert_eq!(evens.to_list(), vec![2, 4, 6]);
        assert_eq!(odds.to_list(), vec![1, 3, 5]);
    }

    #[test]
    fn test_map() {
        let s = IntSet::from_list(&[1, 2, 3]);
        let doubled = s.map(|x| x * 2);
        assert_eq!(doubled.to_list(), vec![2, 4, 6]);
    }

    #[test]
    fn test_fold() {
        let s = IntSet::from_list(&[1, 2, 3, 4, 5]);
        let sum = s.fold(0, |acc, x| acc + x);
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_find_min_max() {
        let s = IntSet::from_list(&[5, 2, 8, 1, 9, 3]);
        assert_eq!(s.find_min(), Some(1));
        assert_eq!(s.find_max(), Some(9));
    }

    #[test]
    fn test_find_min_max_empty() {
        let s = IntSet::new();
        assert_eq!(s.find_min(), None);
        assert_eq!(s.find_max(), None);
    }

    #[test]
    fn test_is_subset() {
        let s1 = IntSet::from_list(&[1, 2, 3]);
        let s2 = IntSet::from_list(&[1, 2, 3, 4, 5]);
        assert!(s1.is_subset_of(&s2));
        assert!(!s2.is_subset_of(&s1));
        assert!(s1.is_subset_of(&s1));
    }

    #[test]
    fn test_is_proper_subset() {
        let s1 = IntSet::from_list(&[1, 2, 3]);
        let s2 = IntSet::from_list(&[1, 2, 3, 4, 5]);
        assert!(s1.is_proper_subset_of(&s2));
        assert!(!s1.is_proper_subset_of(&s1));
    }

    #[test]
    fn test_disjoint() {
        let s1 = IntSet::from_list(&[1, 2, 3]);
        let s2 = IntSet::from_list(&[4, 5, 6]);
        let s3 = IntSet::from_list(&[3, 4, 5]);
        assert!(s1.disjoint(&s2));
        assert!(!s1.disjoint(&s3));
    }

    #[test]
    fn test_equality() {
        let s1 = IntSet::from_list(&[1, 2, 3]);
        let s2 = IntSet::from_list(&[3, 2, 1]);
        let s3 = IntSet::from_list(&[1, 2, 3, 4]);
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_debug() {
        let s = IntSet::from_list(&[1, 2, 3]);
        let debug_str = format!("{:?}", s);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("3"));
    }

    #[test]
    fn test_from_iterator() {
        let s: IntSet = vec![1, 2, 3, 4, 5].into_iter().collect();
        assert_eq!(s.size(), 5);
    }

    #[test]
    fn test_negative_keys() {
        let s = IntSet::from_list(&[-5, -3, -1, 0, 1, 3, 5]);
        assert_eq!(s.size(), 7);
        assert!(s.member(-5));
        assert!(s.member(-1));
        assert!(s.member(0));
        assert!(s.member(5));
    }

    #[test]
    fn test_large_keys() {
        let s = IntSet::from_list(&[i64::MIN, i64::MAX, 0, 1, -1]);
        assert_eq!(s.size(), 5);
        assert!(s.member(i64::MIN));
        assert!(s.member(i64::MAX));
    }

    #[test]
    fn test_many_insertions() {
        let mut s = IntSet::new();
        for i in 0..1000 {
            s = s.insert(i);
        }
        assert_eq!(s.size(), 1000);
        for i in 0..1000 {
            assert!(s.member(i));
        }
    }
}
