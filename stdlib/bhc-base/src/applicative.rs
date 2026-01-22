//! Control.Applicative
//!
//! Extended applicative functor operations corresponding to Haskell's
//! Control.Applicative module.
//!
//! # Example
//!
//! ```ignore
//! use bhc_base::applicative::*;
//!
//! // Lift functions into applicative context
//! let f = |x| x + 1;
//! let result = option_lift_a(f, Some(5));
//! assert_eq!(result, Some(6));
//! ```

// ============================================================
// Option Applicative Operations
// ============================================================

/// Lift a pure value into Option context.
///
/// This is `pure` for the Option applicative.
#[inline]
pub fn option_pure<A>(a: A) -> Option<A> {
    Some(a)
}

/// Apply a function in Option context to a value in Option context.
///
/// This is `<*>` for the Option applicative.
///
/// # Example
///
/// ```ignore
/// let f = Some(|x| x * 2);
/// let result = option_ap(f, Some(5));
/// assert_eq!(result, Some(10));
/// ```
pub fn option_ap<A, B, F>(of: Option<F>, oa: Option<A>) -> Option<B>
where
    F: FnOnce(A) -> B,
{
    match (of, oa) {
        (Some(f), Some(a)) => Some(f(a)),
        _ => None,
    }
}

/// Lift a unary function into Option context.
///
/// `lift_a f = fmap f`
#[inline]
pub fn option_lift_a<A, B, F>(f: F, oa: Option<A>) -> Option<B>
where
    F: FnOnce(A) -> B,
{
    oa.map(f)
}

/// Lift a binary function into Option context.
///
/// `lift_a2 f a b = f <$> a <*> b`
pub fn option_lift_a2<A, B, C, F>(f: F, oa: Option<A>, ob: Option<B>) -> Option<C>
where
    F: FnOnce(A, B) -> C,
{
    match (oa, ob) {
        (Some(a), Some(b)) => Some(f(a, b)),
        _ => None,
    }
}

/// Lift a ternary function into Option context.
pub fn option_lift_a3<A, B, C, D, F>(
    f: F,
    oa: Option<A>,
    ob: Option<B>,
    oc: Option<C>,
) -> Option<D>
where
    F: FnOnce(A, B, C) -> D,
{
    match (oa, ob, oc) {
        (Some(a), Some(b), Some(c)) => Some(f(a, b, c)),
        _ => None,
    }
}

/// Sequence actions, discarding the value of the first argument.
///
/// `a *> b = (id <$ a) <*> b`
pub fn option_right<A, B>(oa: Option<A>, ob: Option<B>) -> Option<B> {
    match (oa, ob) {
        (Some(_), Some(b)) => Some(b),
        _ => None,
    }
}

/// Sequence actions, discarding the value of the second argument.
///
/// `a <* b = liftA2 const a b`
pub fn option_left<A, B>(oa: Option<A>, ob: Option<B>) -> Option<A> {
    match (oa, ob) {
        (Some(a), Some(_)) => Some(a),
        _ => None,
    }
}

// ============================================================
// Result Applicative Operations
// ============================================================

/// Lift a pure value into Result context.
#[inline]
pub fn result_pure<A, E>(a: A) -> Result<A, E> {
    Ok(a)
}

/// Apply a function in Result context to a value in Result context.
pub fn result_ap<A, B, E, F>(rf: Result<F, E>, ra: Result<A, E>) -> Result<B, E>
where
    F: FnOnce(A) -> B,
{
    match (rf, ra) {
        (Ok(f), Ok(a)) => Ok(f(a)),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

/// Lift a binary function into Result context.
pub fn result_lift_a2<A, B, C, E, F>(f: F, ra: Result<A, E>, rb: Result<B, E>) -> Result<C, E>
where
    F: FnOnce(A, B) -> C,
{
    match (ra, rb) {
        (Ok(a), Ok(b)) => Ok(f(a, b)),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

/// Sequence actions, discarding the value of the first argument.
pub fn result_right<A, B, E>(ra: Result<A, E>, rb: Result<B, E>) -> Result<B, E> {
    match (ra, rb) {
        (Ok(_), Ok(b)) => Ok(b),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

/// Sequence actions, discarding the value of the second argument.
pub fn result_left<A, B, E>(ra: Result<A, E>, rb: Result<B, E>) -> Result<A, E> {
    match (ra, rb) {
        (Ok(a), Ok(_)) => Ok(a),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

// ============================================================
// Vec Applicative Operations
// ============================================================

/// Lift a pure value into Vec context.
#[inline]
pub fn vec_pure<A: Clone>(a: A) -> Vec<A> {
    vec![a]
}

/// Apply functions in Vec context to values in Vec context.
///
/// This is the cartesian product style applicative.
pub fn vec_ap<A: Clone, B, F>(vf: Vec<F>, va: Vec<A>) -> Vec<B>
where
    F: Fn(A) -> B,
{
    let mut result = Vec::with_capacity(vf.len() * va.len());
    for f in vf.iter() {
        for a in va.iter() {
            result.push(f(a.clone()));
        }
    }
    result
}

/// Lift a binary function into Vec context (cartesian product).
pub fn vec_lift_a2<A: Clone, B: Clone, C, F>(f: F, va: Vec<A>, vb: Vec<B>) -> Vec<C>
where
    F: Fn(A, B) -> C,
{
    let mut result = Vec::with_capacity(va.len() * vb.len());
    for a in va.iter() {
        for b in vb.iter() {
            result.push(f(a.clone(), b.clone()));
        }
    }
    result
}

// ============================================================
// ZipList Applicative Operations
// ============================================================

/// A wrapper for Vec with zippy applicative behavior.
///
/// Unlike the standard Vec applicative which does cartesian product,
/// ZipList zips the values together pairwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZipList<T>(pub Vec<T>);

impl<T> ZipList<T> {
    /// Create a new ZipList.
    pub fn new(v: Vec<T>) -> Self {
        ZipList(v)
    }

    /// Unwrap to the inner Vec.
    pub fn into_inner(self) -> Vec<T> {
        self.0
    }
}

/// Lift a pure value into ZipList context (infinite repetition, truncated).
pub fn ziplist_pure<A: Clone>(a: A) -> ZipList<A> {
    // In Haskell, pure returns an infinite list. We approximate with a single element.
    ZipList(vec![a])
}

/// Apply functions in ZipList to values in ZipList (zipping).
pub fn ziplist_ap<A: Clone, B, F>(zf: ZipList<F>, za: ZipList<A>) -> ZipList<B>
where
    F: Fn(A) -> B,
{
    let ZipList(fs) = zf;
    let ZipList(xs) = za;
    ZipList(
        fs.into_iter()
            .zip(xs.into_iter())
            .map(|(f, x)| f(x))
            .collect(),
    )
}

/// Lift a binary function into ZipList context (zipping).
pub fn ziplist_lift_a2<A: Clone, B: Clone, C, F>(f: F, za: ZipList<A>, zb: ZipList<B>) -> ZipList<C>
where
    F: Fn(A, B) -> C,
{
    let ZipList(xs) = za;
    let ZipList(ys) = zb;
    ZipList(
        xs.into_iter()
            .zip(ys.into_iter())
            .map(|(x, y)| f(x, y))
            .collect(),
    )
}

// ============================================================
// Alternative (MonadPlus-like for Applicative)
// ============================================================

/// Empty value for Option Alternative.
#[inline]
pub fn option_empty<A>() -> Option<A> {
    None
}

/// Choice operator for Option Alternative.
///
/// Returns the first `Some` value, or `None` if both are `None`.
pub fn option_alt<A>(oa: Option<A>, ob: Option<A>) -> Option<A> {
    oa.or(ob)
}

/// Empty value for Vec Alternative.
#[inline]
pub fn vec_empty<A>() -> Vec<A> {
    Vec::new()
}

/// Choice operator for Vec Alternative.
///
/// Concatenates the two vectors.
pub fn vec_alt<A>(mut va: Vec<A>, vb: Vec<A>) -> Vec<A> {
    va.extend(vb);
    va
}

/// Repeat an action zero or more times (for Vec).
pub fn vec_many<A: Clone, F>(f: F) -> Vec<Vec<A>>
where
    F: Fn() -> Vec<A>,
{
    // For finite structures, this returns [[]] or combinations
    let items = f();
    if items.is_empty() {
        vec![vec![]]
    } else {
        // Return all subsets would be exponential; just return the items and empty
        vec![vec![], items]
    }
}

/// Repeat an action one or more times (for Vec).
pub fn vec_some<A: Clone, F>(f: F) -> Vec<Vec<A>>
where
    F: Fn() -> Vec<A>,
{
    let items = f();
    if items.is_empty() {
        vec![]
    } else {
        vec![items]
    }
}

// ============================================================
// Const Applicative (for traversals)
// ============================================================

/// A constant functor that ignores its second type parameter.
///
/// Useful for implementing things like `foldMap`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Const<A, B> {
    /// The constant value.
    pub value: A,
    _phantom: std::marker::PhantomData<B>,
}

impl<A, B> Const<A, B> {
    /// Create a new Const value.
    pub fn new(value: A) -> Self {
        Const {
            value,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the constant value.
    pub fn get_const(self) -> A {
        self.value
    }
}

/// Map over Const (does nothing to the value).
pub fn const_map<A, B, C, F>(_f: F, c: Const<A, B>) -> Const<A, C>
where
    F: FnOnce(B) -> C,
{
    Const::new(c.value)
}

/// Apply for Const (combines using Monoid-like append).
pub fn const_ap<A, B, C, F>(cf: Const<A, F>, ca: Const<A, B>) -> Const<A, C>
where
    A: std::ops::Add<Output = A>,
{
    Const::new(cf.value + ca.value)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_pure() {
        assert_eq!(option_pure(42), Some(42));
    }

    #[test]
    fn test_option_ap() {
        let f: Option<fn(i32) -> i32> = Some(|x| x * 2);
        assert_eq!(option_ap(f, Some(5)), Some(10));
        let g: Option<fn(i32) -> i32> = None;
        assert_eq!(option_ap(g, Some(5)), None);
        let h: Option<fn(i32) -> i32> = Some(|x| x * 2);
        assert_eq!(option_ap(h, None), None);
    }

    #[test]
    fn test_option_lift_a2() {
        let result = option_lift_a2(|a, b| a + b, Some(3), Some(4));
        assert_eq!(result, Some(7));

        let result2: Option<i32> = option_lift_a2(|a: i32, b| a + b, None, Some(4));
        assert_eq!(result2, None);
    }

    #[test]
    fn test_option_lift_a3() {
        let result = option_lift_a3(|a, b, c| a + b + c, Some(1), Some(2), Some(3));
        assert_eq!(result, Some(6));
    }

    #[test]
    fn test_option_right() {
        assert_eq!(option_right(Some(1), Some(2)), Some(2));
        assert_eq!(option_right::<i32, i32>(None, Some(2)), None);
        assert_eq!(option_right::<i32, i32>(Some(1), None), None);
    }

    #[test]
    fn test_option_left() {
        assert_eq!(option_left(Some(1), Some(2)), Some(1));
        assert_eq!(option_left::<i32, i32>(None, Some(2)), None);
        assert_eq!(option_left::<i32, i32>(Some(1), None), None);
    }

    #[test]
    fn test_result_lift_a2() {
        let result: Result<i32, &str> = result_lift_a2(|a, b| a + b, Ok(3), Ok(4));
        assert_eq!(result, Ok(7));

        let result2: Result<i32, &str> = result_lift_a2(|a: i32, b| a + b, Err("error"), Ok(4));
        assert_eq!(result2, Err("error"));
    }

    #[test]
    fn test_vec_ap() {
        let fs: Vec<fn(i32) -> i32> = vec![|x| x + 1, |x| x * 2];
        let xs = vec![1, 2, 3];
        let result = vec_ap(fs, xs);
        // Cartesian product: [1+1, 2+1, 3+1, 1*2, 2*2, 3*2]
        assert_eq!(result, vec![2, 3, 4, 2, 4, 6]);
    }

    #[test]
    fn test_vec_lift_a2() {
        let result = vec_lift_a2(|a, b| a + b, vec![1, 2], vec![10, 20]);
        // Cartesian: [1+10, 1+20, 2+10, 2+20]
        assert_eq!(result, vec![11, 21, 12, 22]);
    }

    #[test]
    fn test_ziplist_ap() {
        let fs: ZipList<fn(i32) -> i32> = ZipList::new(vec![|x| x + 1, |x| x * 2, |x| x - 1]);
        let xs = ZipList::new(vec![1, 2, 3]);
        let result = ziplist_ap(fs, xs);
        // Zipped: [1+1, 2*2, 3-1]
        assert_eq!(result.into_inner(), vec![2, 4, 2]);
    }

    #[test]
    fn test_ziplist_lift_a2() {
        let result = ziplist_lift_a2(|a, b| a + b, ZipList::new(vec![1, 2, 3]), ZipList::new(vec![10, 20, 30]));
        assert_eq!(result.into_inner(), vec![11, 22, 33]);
    }

    #[test]
    fn test_option_alt() {
        assert_eq!(option_alt(Some(1), Some(2)), Some(1));
        assert_eq!(option_alt(None, Some(2)), Some(2));
        assert_eq!(option_alt(Some(1), None), Some(1));
        assert_eq!(option_alt::<i32>(None, None), None);
    }

    #[test]
    fn test_vec_alt() {
        assert_eq!(vec_alt(vec![1, 2], vec![3, 4]), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_const_functor() {
        let c: Const<i32, &str> = Const::new(42);
        let mapped: Const<i32, bool> = const_map(|_s: &str| true, c);
        assert_eq!(mapped.get_const(), 42);
    }

    #[test]
    fn test_applicative_laws_identity() {
        // pure id <*> v = v
        let v = Some(42);
        let result = option_ap(Some(|x: i32| x), v);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_applicative_laws_homomorphism() {
        // pure f <*> pure x = pure (f x)
        let f = |x: i32| x + 1;
        let x = 5;
        let left = option_ap(Some(f), Some(x));
        let right = Some(f(x));
        assert_eq!(left, right);
    }

    #[test]
    fn test_applicative_laws_interchange() {
        // u <*> pure y = pure ($ y) <*> u
        let u: Option<fn(i32) -> i32> = Some(|x| x * 2);
        let y = 3;
        let left = option_ap(u, Some(y));
        assert_eq!(left, Some(6));
    }
}
