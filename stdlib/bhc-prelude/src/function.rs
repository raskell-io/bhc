//! Function combinators
//!
//! Standard function combinators for BHC. Most of these are implemented
//! directly in Haskell, but some performance-critical ones have Rust
//! implementations.

/// Identity function
///
/// Returns its argument unchanged.
#[inline]
pub fn id<A>(x: A) -> A {
    x
}

/// Constant function
///
/// `const_(x)` returns a function that ignores its argument and returns `x`.
#[inline]
pub fn const_<A, B>(x: A) -> impl Fn(B) -> A
where
    A: Clone,
{
    move |_| x.clone()
}

/// Function composition
///
/// `compose(f, g)` returns a function that applies `g` first, then `f`.
#[inline]
pub fn compose<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(B) -> C,
    G: Fn(A) -> B,
{
    move |x| f(g(x))
}

/// Flip the arguments of a binary function
#[inline]
pub fn flip<A, B, C, F>(f: F) -> impl Fn(B, A) -> C
where
    F: Fn(A, B) -> C,
{
    move |b, a| f(a, b)
}

/// Apply a function to an argument
///
/// `apply(f, x)` is equivalent to `f(x)`.
/// This is the `$` operator in Haskell.
#[inline]
pub fn apply<A, B, F>(f: F, x: A) -> B
where
    F: FnOnce(A) -> B,
{
    f(x)
}

/// Reverse function application
///
/// `pipe(x, f)` is equivalent to `f(x)`.
/// This is the `&` operator in Haskell.
#[inline]
pub fn pipe<A, B, F>(x: A, f: F) -> B
where
    F: FnOnce(A) -> B,
{
    f(x)
}

/// Binary function lifted to operate on results of two functions
///
/// `on(op, f)` returns a function that applies `f` to both arguments
/// before combining with `op`.
#[inline]
pub fn on<A, B, C, F, G>(op: F, f: G) -> impl Fn(A, A) -> C
where
    F: Fn(B, B) -> C,
    G: Fn(A) -> B + Copy,
{
    move |x, y| op(f(x), f(y))
}

/// Fixed-point combinator (Y combinator)
///
/// Allows defining recursive functions without explicit self-reference.
pub fn fix<A, B, F>(f: F) -> impl Fn(A) -> B
where
    F: Fn(&dyn Fn(A) -> B, A) -> B + Copy,
{
    move |x| f(&|a| fix(f)(a), x)
}

// FFI exports

/// Identity for i64
#[no_mangle]
pub extern "C" fn bhc_id_i64(x: i64) -> i64 {
    x
}

/// Identity for f64
#[no_mangle]
pub extern "C" fn bhc_id_f64(x: f64) -> f64 {
    x
}

/// Constant function for i64
#[no_mangle]
pub extern "C" fn bhc_const_i64(x: i64, _y: i64) -> i64 {
    x
}

/// Constant function for f64
#[no_mangle]
pub extern "C" fn bhc_const_f64(x: f64, _y: f64) -> f64 {
    x
}

/// Flip arguments of a binary i64 function
///
/// Note: Function pointers are passed for composition
#[no_mangle]
pub extern "C" fn bhc_flip_i64(
    f: extern "C" fn(i64, i64) -> i64,
    a: i64,
    b: i64,
) -> i64 {
    f(b, a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id() {
        assert_eq!(id(42), 42);
        assert_eq!(id("hello"), "hello");
    }

    #[test]
    fn test_const() {
        let always_5 = const_::<i32, i32>(5);
        assert_eq!(always_5(10), 5);
        assert_eq!(always_5(999), 5);
    }

    #[test]
    fn test_compose() {
        let add1 = |x: i32| x + 1;
        let mul2 = |x: i32| x * 2;
        let add1_then_mul2 = compose(mul2, add1);
        assert_eq!(add1_then_mul2(3), 8); // (3 + 1) * 2
    }

    #[test]
    fn test_flip() {
        let sub = |a: i32, b: i32| a - b;
        let flipped_sub = flip(sub);
        assert_eq!(flipped_sub(3, 10), 7); // 10 - 3
    }

    #[test]
    fn test_on() {
        let compare_length = on(|a: usize, b: usize| a.cmp(&b), |s: &str| s.len());
        assert_eq!(compare_length("hello", "hi"), std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_fix() {
        // Factorial using fix
        let factorial = fix(|rec: &dyn Fn(u64) -> u64, n| {
            if n == 0 {
                1
            } else {
                n * rec(n - 1)
            }
        });
        assert_eq!(factorial(5), 120);
    }
}
