//! Tuple types and operations
//!
//! Provides tuple operations for BHC. Tuples are provided natively by
//! the Haskell runtime, but these helpers provide common operations.

/// Extract the first component of a pair.
#[inline]
pub fn fst<A, B>((a, _): (A, B)) -> A {
    a
}

/// Extract the second component of a pair.
#[inline]
pub fn snd<A, B>((_, b): (A, B)) -> B {
    b
}

/// Swap the components of a pair.
#[inline]
pub fn swap<A, B>((a, b): (A, B)) -> (B, A) {
    (b, a)
}

/// Curry a function that takes a pair.
///
/// Note: Returns a boxed function due to Rust's type system limitations.
#[inline]
pub fn curry<A, B, C, F>(f: F, a: A) -> impl Fn(B) -> C
where
    F: Fn((A, B)) -> C,
    A: Clone,
{
    move |b| f((a.clone(), b))
}

/// Uncurry a curried function.
#[inline]
pub fn uncurry<A, B, C, F>(f: F) -> impl Fn((A, B)) -> C
where
    F: Fn(A, B) -> C,
{
    move |(a, b)| f(a, b)
}

// FFI exports for pairs of i64

/// Get first element of pair
#[no_mangle]
pub extern "C" fn bhc_fst_i64(a: i64, _b: i64) -> i64 {
    a
}

/// Get second element of pair
#[no_mangle]
pub extern "C" fn bhc_snd_i64(_a: i64, b: i64) -> i64 {
    b
}

/// Swap pair components (returns through out parameters)
#[no_mangle]
pub extern "C" fn bhc_swap_i64(a: i64, b: i64, out_a: &mut i64, out_b: &mut i64) {
    *out_a = b;
    *out_b = a;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fst_snd() {
        let pair = (1, 2);
        assert_eq!(fst(pair), 1);
        assert_eq!(snd(pair), 2);
    }

    #[test]
    fn test_swap() {
        assert_eq!(swap((1, 2)), (2, 1));
    }

    #[test]
    fn test_curry_uncurry() {
        let add_pair = |(a, b): (i32, i32)| a + b;
        let add_with_1 = curry(add_pair, 1);
        assert_eq!(add_with_1(2), 3);

        let add = |a: i32, b: i32| a + b;
        let add_uncurried = uncurry(add);
        assert_eq!(add_uncurried((1, 2)), 3);
    }
}
