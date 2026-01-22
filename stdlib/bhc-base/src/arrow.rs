//! Arrow combinators
//!
//! Arrows are a generalization of functions that provide a way to express
//! computations with multiple inputs and outputs.
//!
//! # Overview
//!
//! The Arrow abstraction captures computations that:
//! - Take an input and produce an output
//! - Can be composed sequentially and in parallel
//! - Support first-class routing of data
//!
//! # Example
//!
//! ```
//! use bhc_base::arrow::{Arrow, arr, compose, first, second, split, fanout};
//!
//! // Create arrows from functions
//! let double = arr(|x: i32| x * 2);
//! let add_one = arr(|x: i32| x + 1);
//!
//! // Compose arrows: double then add_one
//! let combined = compose(double, add_one);
//! assert_eq!(combined.run(5), 11); // (5 * 2) + 1
//!
//! // Create fresh arrows for parallel composition
//! let double2 = arr(|x: i32| x * 2);
//! let add_one2 = arr(|x: i32| x + 1);
//! let both = fanout(double2, add_one2);
//! assert_eq!(both.run(5), (10, 6)); // (5 * 2, 5 + 1)
//! ```

use std::marker::PhantomData;

/// An arrow from input type `A` to output type `B`
pub trait Arrow<A, B> {
    /// Run the arrow on an input
    fn run(&self, input: A) -> B;
}

/// A function arrow wrapping a closure
pub struct FnArrow<A, B, F>
where
    F: Fn(A) -> B,
{
    f: F,
    _phantom: PhantomData<(A, B)>,
}

impl<A, B, F> FnArrow<A, B, F>
where
    F: Fn(A) -> B,
{
    /// Create a new function arrow
    pub fn new(f: F) -> Self {
        FnArrow {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<A, B, F> Arrow<A, B> for FnArrow<A, B, F>
where
    F: Fn(A) -> B,
{
    fn run(&self, input: A) -> B {
        (self.f)(input)
    }
}

impl<A, B, F: Clone> Clone for FnArrow<A, B, F>
where
    F: Fn(A) -> B,
{
    fn clone(&self) -> Self {
        FnArrow {
            f: self.f.clone(),
            _phantom: PhantomData,
        }
    }
}

/// Create an arrow from a function
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// assert_eq!(double.run(5), 10);
/// ```
pub fn arr<A, B, F>(f: F) -> FnArrow<A, B, F>
where
    F: Fn(A) -> B,
{
    FnArrow::new(f)
}

/// The identity arrow
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{identity, Arrow};
///
/// let id = identity::<i32>();
/// assert_eq!(id.run(42), 42);
/// ```
pub fn identity<A>() -> FnArrow<A, A, impl Fn(A) -> A>
where
    A: Clone,
{
    arr(|x| x)
}

/// Composed arrow
pub struct Composed<A, B, C, F, G>
where
    F: Arrow<A, B>,
    G: Arrow<B, C>,
{
    first: F,
    second: G,
    _phantom: PhantomData<(A, B, C)>,
}

impl<A, B, C, F, G> Arrow<A, C> for Composed<A, B, C, F, G>
where
    F: Arrow<A, B>,
    G: Arrow<B, C>,
{
    fn run(&self, input: A) -> C {
        let intermediate = self.first.run(input);
        self.second.run(intermediate)
    }
}

/// Compose two arrows: run first, then second
///
/// This is the `>>>` operator in Haskell.
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, compose, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// let add_one = arr(|x: i32| x + 1);
/// let combined = compose(double, add_one);
///
/// assert_eq!(combined.run(5), 11); // (5 * 2) + 1
/// ```
pub fn compose<A, B, C, F, G>(first: F, second: G) -> Composed<A, B, C, F, G>
where
    F: Arrow<A, B>,
    G: Arrow<B, C>,
{
    Composed {
        first,
        second,
        _phantom: PhantomData,
    }
}

/// Compose two arrows in reverse order: run second, then first
///
/// This is the `<<<` operator in Haskell.
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, compose_reverse, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// let add_one = arr(|x: i32| x + 1);
/// let combined = compose_reverse(add_one, double);
///
/// assert_eq!(combined.run(5), 11); // (5 * 2) + 1
/// ```
pub fn compose_reverse<A, B, C, F, G>(second: G, first: F) -> Composed<A, B, C, F, G>
where
    F: Arrow<A, B>,
    G: Arrow<B, C>,
{
    compose(first, second)
}

/// First arrow - apply arrow to first component of pair
pub struct First<A, B, C, F>
where
    F: Arrow<A, B>,
{
    arrow: F,
    _phantom: PhantomData<(A, B, C)>,
}

impl<A, B, C, F> Arrow<(A, C), (B, C)> for First<A, B, C, F>
where
    F: Arrow<A, B>,
{
    fn run(&self, input: (A, C)) -> (B, C) {
        let (a, c) = input;
        (self.arrow.run(a), c)
    }
}

/// Apply an arrow to the first component of a pair
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, first, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// let f = first::<_, _, &str, _>(double);
///
/// assert_eq!(f.run((5, "hello")), (10, "hello"));
/// ```
pub fn first<A, B, C, F>(arrow: F) -> First<A, B, C, F>
where
    F: Arrow<A, B>,
{
    First {
        arrow,
        _phantom: PhantomData,
    }
}

/// Second arrow - apply arrow to second component of pair
pub struct Second<A, B, C, F>
where
    F: Arrow<B, C>,
{
    arrow: F,
    _phantom: PhantomData<(A, B, C)>,
}

impl<A, B, C, F> Arrow<(A, B), (A, C)> for Second<A, B, C, F>
where
    F: Arrow<B, C>,
{
    fn run(&self, input: (A, B)) -> (A, C) {
        let (a, b) = input;
        (a, self.arrow.run(b))
    }
}

/// Apply an arrow to the second component of a pair
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, second, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// let s = second::<&str, _, _, _>(double);
///
/// assert_eq!(s.run(("hello", 5)), ("hello", 10));
/// ```
pub fn second<A, B, C, F>(arrow: F) -> Second<A, B, C, F>
where
    F: Arrow<B, C>,
{
    Second {
        arrow,
        _phantom: PhantomData,
    }
}

/// Split arrow - apply two arrows in parallel
pub struct Split<A, B, C, D, F, G>
where
    F: Arrow<A, B>,
    G: Arrow<C, D>,
{
    left: F,
    right: G,
    _phantom: PhantomData<(A, B, C, D)>,
}

impl<A, B, C, D, F, G> Arrow<(A, C), (B, D)> for Split<A, B, C, D, F, G>
where
    F: Arrow<A, B>,
    G: Arrow<C, D>,
{
    fn run(&self, input: (A, C)) -> (B, D) {
        let (a, c) = input;
        (self.left.run(a), self.right.run(c))
    }
}

/// Apply two arrows in parallel to a pair
///
/// This is the `***` operator in Haskell.
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, split, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// let negate = arr(|x: i32| -x);
/// let both = split(double, negate);
///
/// assert_eq!(both.run((5, 3)), (10, -3));
/// ```
pub fn split<A, B, C, D, F, G>(left: F, right: G) -> Split<A, B, C, D, F, G>
where
    F: Arrow<A, B>,
    G: Arrow<C, D>,
{
    Split {
        left,
        right,
        _phantom: PhantomData,
    }
}

/// Fanout arrow - apply two arrows to the same input
pub struct Fanout<A, B, C, F, G>
where
    F: Arrow<A, B>,
    G: Arrow<A, C>,
{
    left: F,
    right: G,
    _phantom: PhantomData<(A, B, C)>,
}

impl<A, B, C, F, G> Arrow<A, (B, C)> for Fanout<A, B, C, F, G>
where
    A: Clone,
    F: Arrow<A, B>,
    G: Arrow<A, C>,
{
    fn run(&self, input: A) -> (B, C) {
        (self.left.run(input.clone()), self.right.run(input))
    }
}

/// Apply two arrows to the same input, returning both results
///
/// This is the `&&&` operator in Haskell.
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, fanout, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// let square = arr(|x: i32| x * x);
/// let both = fanout(double, square);
///
/// assert_eq!(both.run(5), (10, 25));
/// ```
pub fn fanout<A, B, C, F, G>(left: F, right: G) -> Fanout<A, B, C, F, G>
where
    A: Clone,
    F: Arrow<A, B>,
    G: Arrow<A, C>,
{
    Fanout {
        left,
        right,
        _phantom: PhantomData,
    }
}

/// Constant arrow - always return the same value
pub struct Constant<A, B> {
    value: B,
    _phantom: PhantomData<A>,
}

impl<A, B: Clone> Arrow<A, B> for Constant<A, B> {
    fn run(&self, _input: A) -> B {
        self.value.clone()
    }
}

/// Create an arrow that always returns the same value
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{constant, Arrow};
///
/// let always_42 = constant::<&str, _>(42);
/// assert_eq!(always_42.run("anything"), 42);
/// assert_eq!(always_42.run("something else"), 42);
/// ```
pub fn constant<A, B: Clone>(value: B) -> Constant<A, B> {
    Constant {
        value,
        _phantom: PhantomData,
    }
}

/// Duplicate the input
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{dup, Arrow};
///
/// let d = dup::<i32>();
/// assert_eq!(d.run(5), (5, 5));
/// ```
pub fn dup<A: Clone>() -> FnArrow<A, (A, A), impl Fn(A) -> (A, A)> {
    arr(|x: A| (x.clone(), x))
}

/// Swap the components of a pair
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{swap, Arrow};
///
/// let s = swap::<i32, &str>();
/// assert_eq!(s.run((1, "hello")), ("hello", 1));
/// ```
pub fn swap<A, B>() -> FnArrow<(A, B), (B, A), impl Fn((A, B)) -> (B, A)> {
    arr(|(a, b)| (b, a))
}

/// Associate a nested pair to the left
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{assoc_left, Arrow};
///
/// let al = assoc_left::<i32, i32, i32>();
/// assert_eq!(al.run((1, (2, 3))), ((1, 2), 3));
/// ```
pub fn assoc_left<A, B, C>() -> FnArrow<(A, (B, C)), ((A, B), C), impl Fn((A, (B, C))) -> ((A, B), C)>
{
    arr(|(a, (b, c))| ((a, b), c))
}

/// Associate a nested pair to the right
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{assoc_right, Arrow};
///
/// let ar = assoc_right::<i32, i32, i32>();
/// assert_eq!(ar.run(((1, 2), 3)), (1, (2, 3)));
/// ```
pub fn assoc_right<A, B, C>(
) -> FnArrow<((A, B), C), (A, (B, C)), impl Fn(((A, B), C)) -> (A, (B, C))> {
    arr(|((a, b), c)| (a, (b, c)))
}

/// Extract the first component of a pair
pub fn fst<A, B>() -> FnArrow<(A, B), A, impl Fn((A, B)) -> A> {
    arr(|(a, _)| a)
}

/// Extract the second component of a pair
pub fn snd<A, B>() -> FnArrow<(A, B), B, impl Fn((A, B)) -> B> {
    arr(|(_, b)| b)
}

// ArrowChoice - arrows with choice

/// Left injection for Either
pub struct Left<A, B, C, F>
where
    F: Arrow<A, B>,
{
    arrow: F,
    _phantom: PhantomData<(A, B, C)>,
}

/// Apply arrow to Left, pass through Right
impl<A, B, C, F> Arrow<Result<A, C>, Result<B, C>> for Left<A, B, C, F>
where
    F: Arrow<A, B>,
{
    fn run(&self, input: Result<A, C>) -> Result<B, C> {
        match input {
            Ok(a) => Ok(self.arrow.run(a)),
            Err(c) => Err(c),
        }
    }
}

/// Apply an arrow only to Left values
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, left, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// let l = left::<_, _, &str, _>(double);
///
/// assert_eq!(l.run(Ok(5)), Ok(10));
/// assert_eq!(l.run(Err("error")), Err("error"));
/// ```
pub fn left<A, B, C, F>(arrow: F) -> Left<A, B, C, F>
where
    F: Arrow<A, B>,
{
    Left {
        arrow,
        _phantom: PhantomData,
    }
}

/// Right injection for Either
pub struct Right<A, B, C, F>
where
    F: Arrow<B, C>,
{
    arrow: F,
    _phantom: PhantomData<(A, B, C)>,
}

/// Apply arrow to Right, pass through Left
impl<A, B, C, F> Arrow<Result<A, B>, Result<A, C>> for Right<A, B, C, F>
where
    F: Arrow<B, C>,
{
    fn run(&self, input: Result<A, B>) -> Result<A, C> {
        match input {
            Ok(a) => Ok(a),
            Err(b) => Err(self.arrow.run(b)),
        }
    }
}

/// Apply an arrow only to Right/Err values
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, right, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// let r = right::<&str, _, _, _>(double);
///
/// assert_eq!(r.run(Ok("ok")), Ok("ok"));
/// assert_eq!(r.run(Err(5)), Err(10));
/// ```
pub fn right<A, B, C, F>(arrow: F) -> Right<A, B, C, F>
where
    F: Arrow<B, C>,
{
    Right {
        arrow,
        _phantom: PhantomData,
    }
}

/// Choice between two arrows based on Either
pub struct Choice<A, B, C, D, F, G>
where
    F: Arrow<A, C>,
    G: Arrow<B, D>,
{
    on_left: F,
    on_right: G,
    _phantom: PhantomData<(A, B, C, D)>,
}

impl<A, B, C, D, F, G> Arrow<Result<A, B>, Result<C, D>> for Choice<A, B, C, D, F, G>
where
    F: Arrow<A, C>,
    G: Arrow<B, D>,
{
    fn run(&self, input: Result<A, B>) -> Result<C, D> {
        match input {
            Ok(a) => Ok(self.on_left.run(a)),
            Err(b) => Err(self.on_right.run(b)),
        }
    }
}

/// Apply different arrows based on Either
///
/// This is the `+++` operator in Haskell.
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, choice, Arrow};
///
/// let double = arr(|x: i32| x * 2);
/// let negate = arr(|x: i32| -x);
/// let c = choice(double, negate);
///
/// assert_eq!(c.run(Ok(5)), Ok(10));
/// assert_eq!(c.run(Err(5)), Err(-5));
/// ```
pub fn choice<A, B, C, D, F, G>(on_left: F, on_right: G) -> Choice<A, B, C, D, F, G>
where
    F: Arrow<A, C>,
    G: Arrow<B, D>,
{
    Choice {
        on_left,
        on_right,
        _phantom: PhantomData,
    }
}

/// Fanin - merge two arrows with the same output type
pub struct Fanin<A, B, C, F, G>
where
    F: Arrow<A, C>,
    G: Arrow<B, C>,
{
    on_left: F,
    on_right: G,
    _phantom: PhantomData<(A, B, C)>,
}

impl<A, B, C, F, G> Arrow<Result<A, B>, C> for Fanin<A, B, C, F, G>
where
    F: Arrow<A, C>,
    G: Arrow<B, C>,
{
    fn run(&self, input: Result<A, B>) -> C {
        match input {
            Ok(a) => self.on_left.run(a),
            Err(b) => self.on_right.run(b),
        }
    }
}

/// Merge two arrows with the same output type
///
/// This is the `|||` operator in Haskell.
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, fanin, Arrow};
///
/// let show_int = arr(|x: i32| format!("int: {}", x));
/// let show_str = arr(|s: &str| format!("str: {}", s));
/// let f = fanin(show_int, show_str);
///
/// assert_eq!(f.run(Ok(42)), "int: 42");
/// assert_eq!(f.run(Err("hello")), "str: hello");
/// ```
pub fn fanin<A, B, C, F, G>(on_left: F, on_right: G) -> Fanin<A, B, C, F, G>
where
    F: Arrow<A, C>,
    G: Arrow<B, C>,
{
    Fanin {
        on_left,
        on_right,
        _phantom: PhantomData,
    }
}

// ArrowLoop - arrows with feedback

/// Loop arrow for feedback
pub struct Loop<A, B, C, F>
where
    F: Arrow<(A, C), (B, C)>,
{
    arrow: F,
    _phantom: PhantomData<(A, B, C)>,
}

// Note: Full ArrowLoop requires lazy evaluation which is complex in Rust.
// This is a simplified version that requires an initial value.

/// Create a loop with feedback (simplified version requiring initial state)
///
/// # Example
///
/// ```
/// use bhc_base::arrow::{arr, loop_with_init, Arrow};
///
/// // Accumulating sum: each input adds to running total
/// let summer = arr(|(x, acc): (i32, i32)| {
///     let new_acc = acc + x;
///     (new_acc, new_acc)
/// });
/// let looped = loop_with_init(summer, 0);
///
/// assert_eq!(looped.run(5), 5);   // 0 + 5 = 5
/// ```
pub fn loop_with_init<A, B, C, F>(arrow: F, init: C) -> impl Arrow<A, B>
where
    F: Arrow<(A, C), (B, C)>,
    C: Clone,
{
    arr(move |a| {
        let (b, _) = arrow.run((a, init.clone()));
        b
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arr() {
        let double = arr(|x: i32| x * 2);
        assert_eq!(double.run(5), 10);
    }

    #[test]
    fn test_identity() {
        let id = identity::<i32>();
        assert_eq!(id.run(42), 42);
    }

    #[test]
    fn test_compose() {
        let double = arr(|x: i32| x * 2);
        let add_one = arr(|x: i32| x + 1);
        let combined = compose(double, add_one);
        assert_eq!(combined.run(5), 11);
    }

    #[test]
    fn test_compose_reverse() {
        let double = arr(|x: i32| x * 2);
        let add_one = arr(|x: i32| x + 1);
        let combined = compose_reverse(add_one, double);
        assert_eq!(combined.run(5), 11);
    }

    #[test]
    fn test_first() {
        let double = arr(|x: i32| x * 2);
        let f = first::<_, _, &str, _>(double);
        assert_eq!(f.run((5, "hello")), (10, "hello"));
    }

    #[test]
    fn test_second() {
        let double = arr(|x: i32| x * 2);
        let s = second::<&str, _, _, _>(double);
        assert_eq!(s.run(("hello", 5)), ("hello", 10));
    }

    #[test]
    fn test_split() {
        let double = arr(|x: i32| x * 2);
        let negate = arr(|x: i32| -x);
        let both = split(double, negate);
        assert_eq!(both.run((5, 3)), (10, -3));
    }

    #[test]
    fn test_fanout() {
        let double = arr(|x: i32| x * 2);
        let square = arr(|x: i32| x * x);
        let both = fanout(double, square);
        assert_eq!(both.run(5), (10, 25));
    }

    #[test]
    fn test_constant() {
        let always_42 = constant::<&str, _>(42);
        assert_eq!(always_42.run("anything"), 42);
    }

    #[test]
    fn test_dup() {
        let d = dup::<i32>();
        assert_eq!(d.run(5), (5, 5));
    }

    #[test]
    fn test_swap() {
        let s = swap::<i32, &str>();
        assert_eq!(s.run((1, "hello")), ("hello", 1));
    }

    #[test]
    fn test_assoc() {
        let al = assoc_left::<i32, i32, i32>();
        assert_eq!(al.run((1, (2, 3))), ((1, 2), 3));

        let ar = assoc_right::<i32, i32, i32>();
        assert_eq!(ar.run(((1, 2), 3)), (1, (2, 3)));
    }

    #[test]
    fn test_fst_snd() {
        let f = fst::<i32, &str>();
        assert_eq!(f.run((1, "hello")), 1);

        let s = snd::<i32, &str>();
        assert_eq!(s.run((1, "hello")), "hello");
    }

    #[test]
    fn test_left() {
        let double = arr(|x: i32| x * 2);
        let l = left::<_, _, &str, _>(double);
        assert_eq!(l.run(Ok(5)), Ok(10));
        assert_eq!(l.run(Err("error")), Err("error"));
    }

    #[test]
    fn test_right() {
        let double = arr(|x: i32| x * 2);
        let r = right::<&str, _, _, _>(double);
        assert_eq!(r.run(Ok("ok")), Ok("ok"));
        assert_eq!(r.run(Err(5)), Err(10));
    }

    #[test]
    fn test_choice() {
        let double = arr(|x: i32| x * 2);
        let negate = arr(|x: i32| -x);
        let c = choice(double, negate);
        assert_eq!(c.run(Ok(5)), Ok(10));
        assert_eq!(c.run(Err(5)), Err(-5));
    }

    #[test]
    fn test_fanin() {
        let show_int = arr(|x: i32| format!("int: {}", x));
        let show_str = arr(|s: &str| format!("str: {}", s));
        let f = fanin(show_int, show_str);
        assert_eq!(f.run(Ok(42)), "int: 42");
        assert_eq!(f.run(Err("hello")), "str: hello");
    }

    #[test]
    fn test_complex_composition() {
        // Build a more complex arrow pipeline
        let parse_int = arr(|s: &str| s.parse::<i32>().unwrap_or(0));
        let double = arr(|x: i32| x * 2);
        let to_string = arr(|x: i32| x.to_string());

        let pipeline = compose(compose(parse_int, double), to_string);
        assert_eq!(pipeline.run("21"), "42");
    }
}
