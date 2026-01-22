//! Control.Category
//!
//! Category theory abstractions corresponding to Haskell's Control.Category
//! and Control.Arrow modules.
//!
//! # Example
//!
//! ```ignore
//! use bhc_base::category::*;
//!
//! // Compose functions using category operators
//! let f = |x: i32| x + 1;
//! let g = |x: i32| x * 2;
//! let h = compose(f, g); // f . g = (x * 2) + 1
//! assert_eq!(h(5), 11);
//! ```

use std::marker::PhantomData;

// ============================================================
// Category Abstraction
// ============================================================

/// Identity morphism.
///
/// For functions, this is the identity function.
#[inline]
pub fn id<A>(a: A) -> A {
    a
}

/// Composition of morphisms (right to left).
///
/// `compose f g = f . g` in Haskell notation.
/// `compose(f, g)(x) = f(g(x))`
#[inline]
pub fn compose<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(B) -> C,
    G: Fn(A) -> B,
{
    move |a| f(g(a))
}

/// Right-to-left composition (same as compose, Haskell <<<).
#[inline]
pub fn compose_rtl<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(B) -> C,
    G: Fn(A) -> B,
{
    compose(f, g)
}

/// Left-to-right composition (Haskell >>>).
///
/// `pipe(g, f)(x) = f(g(x))`
#[inline]
pub fn pipe<A, B, C, F, G>(g: G, f: F) -> impl Fn(A) -> C
where
    F: Fn(B) -> C,
    G: Fn(A) -> B,
{
    move |a| f(g(a))
}

// ============================================================
// Arrow-like Operations for Functions
// ============================================================

/// Apply a function to the first component of a pair.
///
/// `first f (a, b) = (f a, b)`
#[inline]
pub fn first<A, B, C, F>(f: F) -> impl Fn((A, C)) -> (B, C)
where
    F: Fn(A) -> B,
{
    move |(a, c)| (f(a), c)
}

/// Apply a function to the second component of a pair.
///
/// `second f (a, b) = (a, f b)`
#[inline]
pub fn second<A, B, C, F>(f: F) -> impl Fn((C, A)) -> (C, B)
where
    F: Fn(A) -> B,
{
    move |(c, a)| (c, f(a))
}

/// Combine two functions to run in parallel on a pair.
///
/// `split f g (a, b) = (f a, g b)`
/// This is `(***)` in Haskell.
#[inline]
pub fn split<A, B, C, D, F, G>(f: F, g: G) -> impl Fn((A, C)) -> (B, D)
where
    F: Fn(A) -> B,
    G: Fn(C) -> D,
{
    move |(a, c)| (f(a), g(c))
}

/// Fan-out: apply both functions to the same input and combine results.
///
/// `fanout f g x = (f x, g x)`
/// This is `(&&&)` in Haskell.
#[inline]
pub fn fanout<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> (B, C)
where
    A: Clone,
    F: Fn(A) -> B,
    G: Fn(A) -> C,
{
    move |a| (f(a.clone()), g(a))
}

// ============================================================
// ArrowChoice-like Operations
// ============================================================

/// Apply a function to the left component of an Either.
///
/// `left f (Left a) = Left (f a)`
/// `left f (Right c) = Right c`
pub fn left<A, B, C, F>(f: F) -> impl Fn(Result<A, C>) -> Result<B, C>
where
    F: Fn(A) -> B,
{
    move |e| match e {
        Ok(a) => Ok(f(a)),
        Err(c) => Err(c),
    }
}

/// Apply a function to the right component of an Either.
///
/// `right f (Left c) = Left c`
/// `right f (Right a) = Right (f a)`
pub fn right<A, B, C, F>(f: F) -> impl Fn(Result<C, A>) -> Result<C, B>
where
    F: Fn(A) -> B,
{
    move |e| match e {
        Ok(c) => Ok(c),
        Err(a) => Err(f(a)),
    }
}

/// Combine two functions to work on Either.
///
/// `choose f g (Left a) = Left (f a)`
/// `choose f g (Right b) = Right (g b)`
/// This is `(+++)` in Haskell.
pub fn choose<A, B, C, D, F, G>(f: F, g: G) -> impl Fn(Result<A, C>) -> Result<B, D>
where
    F: Fn(A) -> B,
    G: Fn(C) -> D,
{
    move |e| match e {
        Ok(a) => Ok(f(a)),
        Err(c) => Err(g(c)),
    }
}

/// Fan-in: choose between two functions based on Either.
///
/// `fanin f g (Left a) = f a`
/// `fanin f g (Right b) = g b`
/// This is `(|||)` in Haskell.
pub fn fanin<A, B, C, F, G>(f: F, g: G) -> impl Fn(Result<A, B>) -> C
where
    F: Fn(A) -> C,
    G: Fn(B) -> C,
{
    move |e| match e {
        Ok(a) => f(a),
        Err(b) => g(b),
    }
}

// ============================================================
// ArrowApply-like Operations
// ============================================================

/// Apply a function from a pair.
///
/// `app (f, x) = f x`
#[inline]
pub fn app<A, B, F>(pair: (F, A)) -> B
where
    F: FnOnce(A) -> B,
{
    let (f, a) = pair;
    f(a)
}

// ============================================================
// ArrowLoop placeholder
// ============================================================

/// Loop combinator for arrows (simplified).
///
/// This is a simplified version that works for functions.
/// The full ArrowLoop requires careful treatment of laziness.
pub fn loop_arrow<A, B, C, F>(f: F) -> impl Fn(A) -> B
where
    F: Fn((A, C)) -> (B, C),
    C: Default + Clone,
{
    move |a| {
        let c = C::default();
        let (b, _) = f((a, c));
        b
    }
}

// ============================================================
// Kleisli Category for Option
// ============================================================

/// Kleisli composition for Option (left to right).
///
/// `f >=> g = \a -> f a >>= g`
pub fn kleisli_compose<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> Option<C>
where
    F: Fn(A) -> Option<B>,
    G: Fn(B) -> Option<C>,
{
    move |a| f(a).and_then(|b| g(b))
}

/// Kleisli composition for Option (right to left).
///
/// `g <=< f = \a -> f a >>= g`
pub fn kleisli_compose_rtl<A, B, C, F, G>(g: G, f: F) -> impl Fn(A) -> Option<C>
where
    F: Fn(A) -> Option<B>,
    G: Fn(B) -> Option<C>,
{
    kleisli_compose(f, g)
}

// ============================================================
// Kleisli Category for Result
// ============================================================

/// Kleisli composition for Result (left to right).
pub fn result_kleisli_compose<A, B, C, E, F, G>(f: F, g: G) -> impl Fn(A) -> Result<C, E>
where
    F: Fn(A) -> Result<B, E>,
    G: Fn(B) -> Result<C, E>,
{
    move |a| f(a).and_then(|b| g(b))
}

// ============================================================
// Profunctor-like Operations
// ============================================================

/// Map over both arguments of a function (contravariant in first, covariant in second).
///
/// `dimap f g h = g . h . f`
#[inline]
pub fn dimap<A, B, C, D, F, G, H>(f: F, g: G, h: H) -> impl Fn(B) -> C
where
    F: Fn(B) -> A,
    G: Fn(D) -> C,
    H: Fn(A) -> D,
{
    move |b| g(h(f(b)))
}

/// Map over the input of a function (contravariant).
///
/// `lmap f g = g . f`
#[inline]
pub fn lmap<A, B, C, F, G>(f: F, g: G) -> impl Fn(B) -> C
where
    F: Fn(B) -> A,
    G: Fn(A) -> C,
{
    move |b| g(f(b))
}

/// Map over the output of a function (covariant).
///
/// `rmap f g = f . g`
#[inline]
pub fn rmap<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(B) -> C,
    G: Fn(A) -> B,
{
    move |a| f(g(a))
}

// ============================================================
// Newtype for Kleisli Arrows
// ============================================================

/// A Kleisli arrow wrapping a function `A -> M<B>`.
pub struct Kleisli<A, B, M> {
    /// The underlying function.
    pub run: Box<dyn Fn(A) -> M>,
    _phantom: PhantomData<B>,
}

impl<A: 'static, B> Kleisli<A, Option<B>, Option<B>> {
    /// Create a new Kleisli arrow for Option.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(A) -> Option<B> + 'static,
    {
        Kleisli {
            run: Box::new(f),
            _phantom: PhantomData,
        }
    }

    /// Run the Kleisli arrow.
    pub fn run_kleisli(&self, a: A) -> Option<B> {
        (self.run)(a)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id() {
        assert_eq!(id(42), 42);
        assert_eq!(id("hello"), "hello");
    }

    #[test]
    fn test_compose() {
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 2;
        let h = compose(f, g);
        assert_eq!(h(5), 11); // (5 * 2) + 1
    }

    #[test]
    fn test_pipe() {
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 2;
        let h = pipe(f, g);
        assert_eq!(h(5), 12); // (5 + 1) * 2
    }

    #[test]
    fn test_first() {
        let f = first(|x: i32| x * 2);
        assert_eq!(f((5, "hello")), (10, "hello"));
    }

    #[test]
    fn test_second() {
        let f = second(|x: i32| x * 2);
        assert_eq!(f(("hello", 5)), ("hello", 10));
    }

    #[test]
    fn test_split() {
        let f = split(|x: i32| x + 1, |s: &str| s.len());
        assert_eq!(f((5, "hello")), (6, 5));
    }

    #[test]
    fn test_fanout() {
        let f = fanout(|x: i32| x + 1, |x: i32| x * 2);
        assert_eq!(f(5), (6, 10));
    }

    #[test]
    fn test_left() {
        let f = left(|x: i32| x * 2);
        assert_eq!(f(Ok(5)), Ok(10));
        assert_eq!(f(Err("error")), Err("error"));
    }

    #[test]
    fn test_right() {
        let f = right(|x: i32| x * 2);
        assert_eq!(f(Ok("hello")), Ok("hello"));
        let result: Result<&str, i32> = f(Err(5));
        assert_eq!(result, Err(10));
    }

    #[test]
    fn test_choose() {
        let f = choose(|x: i32| x + 1, |s: &str| s.len());
        assert_eq!(f(Ok(5)), Ok(6));
        assert_eq!(f(Err("hello")), Err(5));
    }

    #[test]
    fn test_fanin() {
        let f = fanin(|x: i32| x.to_string(), |x: bool| x.to_string());
        assert_eq!(f(Ok(42)), "42");
        assert_eq!(f(Err(true)), "true");
    }

    #[test]
    fn test_app() {
        let result = app((|x: i32| x * 2, 5));
        assert_eq!(result, 10);
    }

    #[test]
    fn test_kleisli_compose() {
        let f = |x: i32| if x > 0 { Some(x * 2) } else { None };
        let g = |x: i32| if x < 100 { Some(x + 1) } else { None };
        let h = kleisli_compose(f, g);

        assert_eq!(h(5), Some(11)); // 5 * 2 = 10, 10 + 1 = 11
        assert_eq!(h(-1), None); // f fails
        assert_eq!(h(100), None); // g fails (200 >= 100)
    }

    #[test]
    fn test_result_kleisli() {
        let f = |x: i32| -> Result<i32, &str> {
            if x > 0 {
                Ok(x * 2)
            } else {
                Err("negative")
            }
        };
        let g = |x: i32| -> Result<i32, &str> {
            if x < 100 {
                Ok(x + 1)
            } else {
                Err("too big")
            }
        };
        let h = result_kleisli_compose(f, g);

        assert_eq!(h(5), Ok(11));
        assert_eq!(h(-1), Err("negative"));
    }

    #[test]
    fn test_dimap() {
        // dimap f g h = g . h . f
        // f: B -> A, h: A -> D, g: D -> C
        // Result: B -> C
        let f = |s: &str| s.len() as i32;  // &str -> i32
        let g = |x: i32| x.to_string();    // i32 -> String
        let h = |x: i32| x * 2;            // i32 -> i32
        let composed = dimap(f, g, h);
        // "hello" -> 5 (f) -> 10 (h) -> "10" (g)
        assert_eq!(composed("hello"), "10");
    }

    #[test]
    fn test_lmap() {
        let f = |x: &str| x.len();
        let g = |n: usize| n * 2;
        let composed = lmap(f, g);
        assert_eq!(composed("hello"), 10); // len("hello") = 5, 5 * 2 = 10
    }

    #[test]
    fn test_rmap() {
        let f = |n: i32| n.to_string();
        let g = |n: i32| n * 2;
        let composed = rmap(f, g);
        assert_eq!(composed(5), "10"); // 5 * 2 = 10, "10"
    }

    #[test]
    fn test_category_law_identity_left() {
        // id . f = f
        let f = |x: i32| x + 1;
        let composed = compose(id, f);
        assert_eq!(composed(5), 6);
    }

    #[test]
    fn test_category_law_identity_right() {
        // f . id = f
        let f = |x: i32| x + 1;
        let composed = compose(f, id);
        assert_eq!(composed(5), 6);
    }

    #[test]
    fn test_category_law_associativity() {
        // (f . g) . h = f . (g . h)
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 2;
        let h = |x: i32| x - 3;

        let left = compose(compose(f, g), h);
        let right = compose(f, compose(g, h));

        assert_eq!(left(10), right(10)); // Both should be ((10 - 3) * 2) + 1 = 15
    }

    #[test]
    fn test_arrow_law_identity() {
        // arr id = id
        // For functions, arr is just identity wrapper
        let f = id::<i32>;
        assert_eq!(f(42), 42);
    }

    #[test]
    fn test_arrow_law_first_id() {
        // first id = id
        let f = first(id::<i32>);
        assert_eq!(f((5, "hello")), (5, "hello"));
    }

    #[test]
    fn test_kleisli() {
        let k: Kleisli<i32, Option<i32>, Option<i32>> = Kleisli::new(|x: i32| Some(x * 2));
        assert_eq!(k.run_kleisli(5), Some(10));
    }
}
