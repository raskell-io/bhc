//! Sequences (Data.Sequence)
//!
//! Efficient sequences providing:
//! - O(1) access to front and back elements
//! - O(log n) concatenation
//! - O(n) random access (simple implementation)
//!
//! # Example
//!
//! ```ignore
//! use bhc_containers::sequence::Seq;
//!
//! let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
//! let seq2 = seq.push_front(0).push_back(6);
//! assert_eq!(seq2.len(), 7);
//! ```

use std::fmt::{self, Debug};
use std::rc::Rc;

// ============================================================
// Core Type
// ============================================================

/// A functional sequence with efficient operations at both ends.
///
/// This implementation uses a balanced tree structure providing:
/// - O(1) access to head and last
/// - O(log n) push_front/push_back (amortized)
/// - O(log n) concatenation
#[derive(Clone)]
pub struct Seq<T> {
    tree: Tree<T>,
}

// Internal tree representation using a rope-like structure
#[derive(Clone)]
enum Tree<T> {
    Empty,
    Leaf(Rc<T>),
    Node {
        size: usize,
        left: Rc<Tree<T>>,
        right: Rc<Tree<T>>,
    },
}

// ============================================================
// Tree Implementation
// ============================================================

impl<T: Clone> Tree<T> {
    fn size(&self) -> usize {
        match self {
            Tree::Empty => 0,
            Tree::Leaf(_) => 1,
            Tree::Node { size, .. } => *size,
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Tree::Empty)
    }

    fn head(&self) -> Option<&T> {
        match self {
            Tree::Empty => None,
            Tree::Leaf(v) => Some(v.as_ref()),
            Tree::Node { left, right, .. } => {
                if left.is_empty() {
                    right.head()
                } else {
                    left.head()
                }
            }
        }
    }

    fn last(&self) -> Option<&T> {
        match self {
            Tree::Empty => None,
            Tree::Leaf(v) => Some(v.as_ref()),
            Tree::Node { left, right, .. } => {
                if right.is_empty() {
                    left.last()
                } else {
                    right.last()
                }
            }
        }
    }

    fn push_front(&self, elem: T) -> Tree<T> {
        let new_leaf = Tree::Leaf(Rc::new(elem));
        if self.is_empty() {
            new_leaf
        } else {
            Tree::Node {
                size: self.size() + 1,
                left: Rc::new(new_leaf),
                right: Rc::new(self.clone()),
            }
        }
    }

    fn push_back(&self, elem: T) -> Tree<T> {
        let new_leaf = Tree::Leaf(Rc::new(elem));
        if self.is_empty() {
            new_leaf
        } else {
            Tree::Node {
                size: self.size() + 1,
                left: Rc::new(self.clone()),
                right: Rc::new(new_leaf),
            }
        }
    }

    fn tail(&self) -> Option<Tree<T>> {
        match self {
            Tree::Empty => None,
            Tree::Leaf(_) => Some(Tree::Empty),
            Tree::Node { left, right, .. } => {
                if left.is_empty() {
                    right.tail()
                } else {
                    match left.tail() {
                        Some(new_left) if new_left.is_empty() => Some((**right).clone()),
                        Some(new_left) => Some(Tree::Node {
                            size: new_left.size() + right.size(),
                            left: Rc::new(new_left),
                            right: Rc::clone(right),
                        }),
                        None => Some((**right).clone()),
                    }
                }
            }
        }
    }

    fn init(&self) -> Option<Tree<T>> {
        match self {
            Tree::Empty => None,
            Tree::Leaf(_) => Some(Tree::Empty),
            Tree::Node { left, right, .. } => {
                if right.is_empty() {
                    left.init()
                } else {
                    match right.init() {
                        Some(new_right) if new_right.is_empty() => Some((**left).clone()),
                        Some(new_right) => Some(Tree::Node {
                            size: left.size() + new_right.size(),
                            left: Rc::clone(left),
                            right: Rc::new(new_right),
                        }),
                        None => Some((**left).clone()),
                    }
                }
            }
        }
    }

    fn concat(&self, other: &Tree<T>) -> Tree<T> {
        if self.is_empty() {
            other.clone()
        } else if other.is_empty() {
            self.clone()
        } else {
            Tree::Node {
                size: self.size() + other.size(),
                left: Rc::new(self.clone()),
                right: Rc::new(other.clone()),
            }
        }
    }

    fn get(&self, index: usize) -> Option<&T> {
        if index >= self.size() {
            return None;
        }
        match self {
            Tree::Empty => None,
            Tree::Leaf(v) => {
                if index == 0 {
                    Some(v.as_ref())
                } else {
                    None
                }
            }
            Tree::Node { left, right, .. } => {
                let left_size = left.size();
                if index < left_size {
                    left.get(index)
                } else {
                    right.get(index - left_size)
                }
            }
        }
    }

    fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.size());
        self.collect_into(&mut result);
        result
    }

    fn collect_into(&self, result: &mut Vec<T>) {
        match self {
            Tree::Empty => {}
            Tree::Leaf(v) => result.push((**v).clone()),
            Tree::Node { left, right, .. } => {
                left.collect_into(result);
                right.collect_into(result);
            }
        }
    }
}

// ============================================================
// Seq Implementation
// ============================================================

impl<T: Clone> Seq<T> {
    /// Create an empty sequence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let seq: Seq<i32> = Seq::new();
    /// assert!(seq.is_empty());
    /// ```
    pub fn new() -> Self {
        Seq { tree: Tree::Empty }
    }

    /// Create a sequence with a single element.
    pub fn singleton(elem: T) -> Self {
        Seq {
            tree: Tree::Leaf(Rc::new(elem)),
        }
    }

    /// Create a sequence from a slice.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
    /// assert_eq!(seq.len(), 5);
    /// ```
    pub fn from_list(elems: &[T]) -> Self {
        if elems.is_empty() {
            return Seq::new();
        }
        // Build a balanced tree
        Self::from_slice(elems)
    }

    fn from_slice(elems: &[T]) -> Self {
        if elems.is_empty() {
            Seq::new()
        } else if elems.len() == 1 {
            Seq::singleton(elems[0].clone())
        } else {
            let mid = elems.len() / 2;
            let left = Self::from_slice(&elems[..mid]);
            let right = Self::from_slice(&elems[mid..]);
            left.append(&right)
        }
    }

    /// Check if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.tree.size()
    }

    /// Add an element to the front.
    ///
    /// Time: O(log n)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let seq = Seq::from_list(&[2, 3, 4]);
    /// let seq2 = seq.push_front(1);
    /// assert_eq!(seq2.to_vec(), vec![1, 2, 3, 4]);
    /// ```
    pub fn push_front(&self, elem: T) -> Seq<T> {
        Seq {
            tree: self.tree.push_front(elem),
        }
    }

    /// Add an element to the back.
    ///
    /// Time: O(log n)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let seq = Seq::from_list(&[1, 2, 3]);
    /// let seq2 = seq.push_back(4);
    /// assert_eq!(seq2.to_vec(), vec![1, 2, 3, 4]);
    /// ```
    pub fn push_back(&self, elem: T) -> Seq<T> {
        Seq {
            tree: self.tree.push_back(elem),
        }
    }

    /// Alias for push_front (cons).
    pub fn cons(&self, elem: T) -> Seq<T> {
        self.push_front(elem)
    }

    /// Alias for push_back (snoc).
    pub fn snoc(&self, elem: T) -> Seq<T> {
        self.push_back(elem)
    }

    /// Get the first element.
    ///
    /// Time: O(log n)
    pub fn head(&self) -> Option<&T> {
        self.tree.head()
    }

    /// Get the last element.
    ///
    /// Time: O(log n)
    pub fn last(&self) -> Option<&T> {
        self.tree.last()
    }

    /// Remove the first element and return the rest.
    ///
    /// Time: O(log n)
    pub fn tail(&self) -> Option<Seq<T>> {
        self.tree.tail().map(|tree| Seq { tree })
    }

    /// Remove the last element and return the rest.
    ///
    /// Time: O(log n)
    pub fn init(&self) -> Option<Seq<T>> {
        self.tree.init().map(|tree| Seq { tree })
    }

    /// Split into head and tail.
    ///
    /// Returns `None` if the sequence is empty.
    pub fn uncons(&self) -> Option<(&T, Seq<T>)> {
        let head = self.head()?;
        let tail = self.tail()?;
        Some((head, tail))
    }

    /// Split into init and last.
    ///
    /// Returns `None` if the sequence is empty.
    pub fn unsnoc(&self) -> Option<(Seq<T>, &T)> {
        let last = self.last()?;
        let init = self.init()?;
        Some((init, last))
    }

    /// Concatenate two sequences.
    ///
    /// Time: O(log n)
    pub fn append(&self, other: &Seq<T>) -> Seq<T> {
        Seq {
            tree: self.tree.concat(&other.tree),
        }
    }

    /// Get element at index.
    ///
    /// Time: O(log n)
    pub fn get(&self, index: usize) -> Option<&T> {
        self.tree.get(index)
    }

    /// Update element at index.
    ///
    /// Time: O(n)
    pub fn update(&self, index: usize, elem: T) -> Option<Seq<T>> {
        if index >= self.len() {
            return None;
        }
        let mut vec = self.to_vec();
        vec[index] = elem;
        Some(Seq::from_list(&vec))
    }

    /// Split at index.
    ///
    /// Returns sequences of elements before and at/after index.
    ///
    /// Time: O(n)
    pub fn split_at(&self, index: usize) -> Option<(Seq<T>, Seq<T>)> {
        if index > self.len() {
            return None;
        }
        let vec = self.to_vec();
        let left = Seq::from_list(&vec[..index]);
        let right = Seq::from_list(&vec[index..]);
        Some((left, right))
    }

    /// Take first n elements.
    ///
    /// Time: O(n)
    pub fn take(&self, n: usize) -> Seq<T> {
        let n = n.min(self.len());
        Seq::from_list(&self.to_vec()[..n])
    }

    /// Drop first n elements.
    ///
    /// Time: O(n)
    pub fn drop(&self, n: usize) -> Seq<T> {
        let n = n.min(self.len());
        Seq::from_list(&self.to_vec()[n..])
    }

    /// Map a function over all elements.
    pub fn map<U: Clone, F>(&self, f: F) -> Seq<U>
    where
        F: Fn(&T) -> U,
    {
        let mapped: Vec<U> = self.to_vec().iter().map(f).collect();
        Seq::from_list(&mapped)
    }

    /// Filter elements by a predicate.
    pub fn filter<F>(&self, predicate: F) -> Seq<T>
    where
        F: Fn(&T) -> bool,
    {
        let filtered: Vec<T> = self.to_vec().into_iter().filter(|x| predicate(x)).collect();
        Seq::from_list(&filtered)
    }

    /// Fold from the left.
    pub fn fold_left<B, F>(&self, init: B, f: F) -> B
    where
        F: Fn(B, &T) -> B,
    {
        self.to_vec().iter().fold(init, f)
    }

    /// Fold from the right.
    pub fn fold_right<B, F>(&self, init: B, f: F) -> B
    where
        F: Fn(&T, B) -> B,
    {
        self.to_vec().iter().rev().fold(init, |acc, x| f(x, acc))
    }

    /// Reverse the sequence.
    pub fn reverse(&self) -> Seq<T> {
        let mut vec = self.to_vec();
        vec.reverse();
        Seq::from_list(&vec)
    }

    /// Convert to a Vec.
    pub fn to_vec(&self) -> Vec<T> {
        self.tree.to_vec()
    }

    /// Convert to a list (same as to_vec).
    pub fn to_list(&self) -> Vec<T> {
        self.to_vec()
    }

    /// Create an iterator.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.to_vec().into_iter()
    }

    /// Sort the sequence (requires Ord).
    pub fn sort(&self) -> Seq<T>
    where
        T: Ord,
    {
        let mut vec = self.to_vec();
        vec.sort();
        Seq::from_list(&vec)
    }

    /// Sort by a key function.
    pub fn sort_by<K: Ord, F>(&self, f: F) -> Seq<T>
    where
        F: Fn(&T) -> K,
    {
        let mut vec = self.to_vec();
        vec.sort_by_key(|x| f(x));
        Seq::from_list(&vec)
    }

    /// Zip two sequences together.
    pub fn zip<U: Clone>(&self, other: &Seq<U>) -> Seq<(T, U)> {
        let pairs: Vec<(T, U)> = self
            .to_vec()
            .into_iter()
            .zip(other.to_vec().into_iter())
            .collect();
        Seq::from_list(&pairs)
    }

    /// Zip with a function.
    pub fn zip_with<U: Clone, V: Clone, F>(&self, other: &Seq<U>, f: F) -> Seq<V>
    where
        F: Fn(&T, &U) -> V,
    {
        let results: Vec<V> = self
            .to_vec()
            .iter()
            .zip(other.to_vec().iter())
            .map(|(a, b)| f(a, b))
            .collect();
        Seq::from_list(&results)
    }
}

impl<T: Clone> Default for Seq<T> {
    fn default() -> Self {
        Seq::new()
    }
}

impl<T: Clone + PartialEq> PartialEq for Seq<T> {
    fn eq(&self, other: &Self) -> bool {
        self.to_vec() == other.to_vec()
    }
}

impl<T: Clone + Eq> Eq for Seq<T> {}

impl<T: Clone + Debug> Debug for Seq<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.to_vec()).finish()
    }
}

impl<T: Clone> FromIterator<T> for Seq<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elems: Vec<T> = iter.into_iter().collect();
        Seq::from_list(&elems)
    }
}

impl<T: Clone> IntoIterator for Seq<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_vec().into_iter()
    }
}

impl<'a, T: Clone> IntoIterator for &'a Seq<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_vec().into_iter()
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
        let seq: Seq<i32> = Seq::new();
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_singleton() {
        let seq = Seq::singleton(42);
        assert!(!seq.is_empty());
        assert_eq!(seq.len(), 1);
        assert_eq!(seq.head(), Some(&42));
    }

    #[test]
    fn test_from_list() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
        assert_eq!(seq.len(), 5);
        assert_eq!(seq.to_vec(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_push_front() {
        let seq = Seq::from_list(&[2, 3, 4]);
        let seq2 = seq.push_front(1);
        assert_eq!(seq2.to_vec(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_push_back() {
        let seq = Seq::from_list(&[1, 2, 3]);
        let seq2 = seq.push_back(4);
        assert_eq!(seq2.to_vec(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_head_last() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
        assert_eq!(seq.head(), Some(&1));
        assert_eq!(seq.last(), Some(&5));
    }

    #[test]
    fn test_head_last_empty() {
        let seq: Seq<i32> = Seq::new();
        assert_eq!(seq.head(), None);
        assert_eq!(seq.last(), None);
    }

    #[test]
    fn test_tail() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
        let tail = seq.tail().unwrap();
        assert_eq!(tail.to_vec(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_init() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
        let init = seq.init().unwrap();
        assert_eq!(init.to_vec(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_uncons() {
        let seq = Seq::from_list(&[1, 2, 3]);
        let (head, tail) = seq.uncons().unwrap();
        assert_eq!(*head, 1);
        assert_eq!(tail.to_vec(), vec![2, 3]);
    }

    #[test]
    fn test_unsnoc() {
        let seq = Seq::from_list(&[1, 2, 3]);
        let (init, last) = seq.unsnoc().unwrap();
        assert_eq!(init.to_vec(), vec![1, 2]);
        assert_eq!(*last, 3);
    }

    #[test]
    fn test_append() {
        let seq1 = Seq::from_list(&[1, 2, 3]);
        let seq2 = Seq::from_list(&[4, 5, 6]);
        let combined = seq1.append(&seq2);
        assert_eq!(combined.to_vec(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_get() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
        assert_eq!(seq.get(0), Some(&1));
        assert_eq!(seq.get(2), Some(&3));
        assert_eq!(seq.get(4), Some(&5));
        assert_eq!(seq.get(5), None);
    }

    #[test]
    fn test_split_at() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
        let (left, right) = seq.split_at(2).unwrap();
        assert_eq!(left.to_vec(), vec![1, 2]);
        assert_eq!(right.to_vec(), vec![3, 4, 5]);
    }

    #[test]
    fn test_take_drop() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
        assert_eq!(seq.take(3).to_vec(), vec![1, 2, 3]);
        assert_eq!(seq.drop(2).to_vec(), vec![3, 4, 5]);
    }

    #[test]
    fn test_map() {
        let seq = Seq::from_list(&[1, 2, 3]);
        let doubled = seq.map(|x| x * 2);
        assert_eq!(doubled.to_vec(), vec![2, 4, 6]);
    }

    #[test]
    fn test_filter() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5, 6]);
        let evens = seq.filter(|x| x % 2 == 0);
        assert_eq!(evens.to_vec(), vec![2, 4, 6]);
    }

    #[test]
    fn test_fold() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
        let sum = seq.fold_left(0, |acc, x| acc + x);
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_reverse() {
        let seq = Seq::from_list(&[1, 2, 3, 4, 5]);
        let reversed = seq.reverse();
        assert_eq!(reversed.to_vec(), vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_sort() {
        let seq = Seq::from_list(&[3, 1, 4, 1, 5, 9, 2, 6]);
        let sorted = seq.sort();
        assert_eq!(sorted.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_zip() {
        let seq1 = Seq::from_list(&[1, 2, 3]);
        let seq2 = Seq::from_list(&["a", "b", "c"]);
        let zipped = seq1.zip(&seq2);
        assert_eq!(zipped.to_vec(), vec![(1, "a"), (2, "b"), (3, "c")]);
    }

    #[test]
    fn test_equality() {
        let seq1 = Seq::from_list(&[1, 2, 3]);
        let seq2 = Seq::from_list(&[1, 2, 3]);
        let seq3 = Seq::from_list(&[1, 2, 4]);
        assert_eq!(seq1, seq2);
        assert_ne!(seq1, seq3);
    }

    #[test]
    fn test_debug() {
        let seq = Seq::from_list(&[1, 2, 3]);
        let debug_str = format!("{:?}", seq);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("3"));
    }

    #[test]
    fn test_from_iterator() {
        let seq: Seq<i32> = vec![1, 2, 3, 4, 5].into_iter().collect();
        assert_eq!(seq.len(), 5);
    }

    #[test]
    fn test_many_operations() {
        let mut seq = Seq::new();
        for i in 0..100 {
            seq = seq.push_back(i);
        }
        assert_eq!(seq.len(), 100);

        for i in 0..100 {
            seq = seq.push_front(100 + i);
        }
        assert_eq!(seq.len(), 200);

        // Test that we can still access elements
        assert_eq!(seq.head(), Some(&199));
        assert_eq!(seq.last(), Some(&99));
    }

    #[test]
    fn test_large_append() {
        let seq1 = Seq::from_list(&(0..50).collect::<Vec<_>>());
        let seq2 = Seq::from_list(&(50..100).collect::<Vec<_>>());
        let combined = seq1.append(&seq2);
        assert_eq!(combined.len(), 100);
        assert_eq!(combined.to_vec(), (0..100).collect::<Vec<_>>());
    }
}
