//! Property tests for bhc-prelude
//!
//! These tests verify the algebraic laws and properties of the core types.

use bhc_prelude::either::Either;
use bhc_prelude::maybe::Maybe;
use bhc_prelude::ordering::Ordering;
use proptest::prelude::*;

// ============================================================
// Maybe property tests
// ============================================================

proptest! {
    // Functor laws for Maybe
    #[test]
    fn maybe_functor_identity(x in any::<i32>()) {
        let m = Maybe::Just(x);
        prop_assert_eq!(m.clone().map(|y| y), m);

        let n: Maybe<i32> = Maybe::Nothing;
        prop_assert_eq!(n.clone().map(|y: i32| y), n);
    }

    #[test]
    fn maybe_functor_composition(x in any::<i32>()) {
        let m = Maybe::Just(x);
        let f = |a: i32| a.wrapping_add(1);
        let g = |a: i32| a.wrapping_mul(2);

        // fmap (f . g) == fmap f . fmap g
        let left = m.clone().map(|a| f(g(a)));
        let right = m.map(g).map(f);
        prop_assert_eq!(left, right);
    }

    // Monad laws for Maybe
    #[test]
    fn maybe_monad_left_identity(x in any::<i32>()) {
        let f = |a: i32| if a % 2 == 0 { Maybe::Just(a.wrapping_mul(2)) } else { Maybe::Nothing };

        // return x >>= f == f x
        let left = Maybe::Just(x).and_then(f);
        let right = f(x);
        prop_assert_eq!(left, right);
    }

    #[test]
    fn maybe_monad_right_identity(x in any::<i32>()) {
        let m = Maybe::Just(x);

        // m >>= return == m
        let left = m.clone().and_then(Maybe::Just);
        prop_assert_eq!(left, m);
    }

    #[test]
    fn maybe_monad_associativity(x in any::<i32>()) {
        let m = Maybe::Just(x);
        let f = |a: i32| if a > 0 { Maybe::Just(a + 1) } else { Maybe::Nothing };
        let g = |a: i32| if a < 100 { Maybe::Just(a * 2) } else { Maybe::Nothing };

        // (m >>= f) >>= g == m >>= (\x -> f x >>= g)
        let left = m.clone().and_then(f).and_then(g);
        let right = m.and_then(|a| f(a).and_then(g));
        prop_assert_eq!(left, right);
    }

    // Maybe ordering properties
    #[test]
    fn maybe_ord_reflexive(x in any::<i32>()) {
        let m = Maybe::Just(x);
        prop_assert!(m <= m);
        prop_assert!(m >= m);
    }

    #[test]
    fn maybe_nothing_less_than_just(x in any::<i32>()) {
        let nothing: Maybe<i32> = Maybe::Nothing;
        let just = Maybe::Just(x);
        prop_assert!(nothing < just);
    }

    #[test]
    fn maybe_or_returns_first_just(x in any::<i32>(), y in any::<i32>()) {
        let m1 = Maybe::Just(x);
        let m2 = Maybe::Just(y);
        prop_assert_eq!(m1.clone().or(m2), m1);

        let nothing: Maybe<i32> = Maybe::Nothing;
        prop_assert_eq!(nothing.or(m1.clone()), m1);
    }

    #[test]
    fn maybe_and_requires_both_just(x in any::<i32>(), y in any::<i64>()) {
        let m1 = Maybe::Just(x);
        let m2 = Maybe::Just(y);
        prop_assert_eq!(m1.clone().and(m2.clone()), Maybe::Just(y));

        let nothing: Maybe<i32> = Maybe::Nothing;
        prop_assert_eq!(nothing.and(m2.clone()), Maybe::Nothing);
        prop_assert_eq!(m1.and::<i64>(Maybe::Nothing), Maybe::Nothing);
    }
}

// ============================================================
// Either property tests
// ============================================================

proptest! {
    // Functor laws for Either (Right-biased)
    #[test]
    fn either_functor_identity(x in any::<i32>()) {
        let r: Either<&str, i32> = Either::Right(x);
        prop_assert_eq!(r.clone().map(|y| y), r);

        let l: Either<&str, i32> = Either::Left("error");
        prop_assert_eq!(l.clone().map(|y: i32| y), l);
    }

    #[test]
    fn either_functor_composition(x in any::<i32>()) {
        let e: Either<&str, i32> = Either::Right(x);
        let f = |a: i32| a.wrapping_add(1);
        let g = |a: i32| a.wrapping_mul(2);

        let left = e.clone().map(|a| f(g(a)));
        let right = e.map(g).map(f);
        prop_assert_eq!(left, right);
    }

    // Monad laws for Either
    #[test]
    fn either_monad_left_identity(x in any::<i32>()) {
        let f = |a: i32| -> Either<&str, i32> {
            if a % 2 == 0 { Either::Right(a.wrapping_mul(2)) } else { Either::Left("odd") }
        };

        let left = Either::<&str, i32>::Right(x).and_then(f);
        let right = f(x);
        prop_assert_eq!(left, right);
    }

    #[test]
    fn either_monad_right_identity(x in any::<i32>()) {
        let m: Either<&str, i32> = Either::Right(x);
        let left = m.clone().and_then(Either::Right);
        prop_assert_eq!(left, m);
    }

    #[test]
    fn either_flip_involutive(x in any::<i32>()) {
        let r: Either<&str, i32> = Either::Right(x);
        prop_assert_eq!(r.clone().flip().flip(), r);

        let l: Either<i32, &str> = Either::Left(x);
        prop_assert_eq!(l.clone().flip().flip(), l);
    }

    // Either ordering
    #[test]
    fn either_left_less_than_right(l in any::<i32>(), r in any::<i32>()) {
        let left: Either<i32, i32> = Either::Left(l);
        let right: Either<i32, i32> = Either::Right(r);
        prop_assert!(left < right);
    }
}

// ============================================================
// Ordering property tests
// ============================================================

proptest! {
    // Semigroup associativity
    #[test]
    fn ordering_semigroup_associativity(
        a in prop_oneof![Just(Ordering::LT), Just(Ordering::EQ), Just(Ordering::GT)],
        b in prop_oneof![Just(Ordering::LT), Just(Ordering::EQ), Just(Ordering::GT)],
        c in prop_oneof![Just(Ordering::LT), Just(Ordering::EQ), Just(Ordering::GT)]
    ) {
        // (a <> b) <> c == a <> (b <> c)
        prop_assert_eq!(a.then(b).then(c), a.then(b.then(c)));
    }

    // Monoid identity
    #[test]
    fn ordering_monoid_identity(
        a in prop_oneof![Just(Ordering::LT), Just(Ordering::EQ), Just(Ordering::GT)]
    ) {
        // mempty <> a == a
        prop_assert_eq!(Ordering::EQ.then(a), a);
        // a <> mempty == a
        prop_assert_eq!(a.then(Ordering::EQ), a);
    }

    // reverse is involutive
    #[test]
    fn ordering_reverse_involutive(
        a in prop_oneof![Just(Ordering::LT), Just(Ordering::EQ), Just(Ordering::GT)]
    ) {
        prop_assert_eq!(a.reverse().reverse(), a);
    }
}

// ============================================================
// Cross-type property tests
// ============================================================

proptest! {
    // Maybe to Option and back
    #[test]
    fn maybe_option_roundtrip(x in any::<i32>()) {
        let m = Maybe::Just(x);
        let opt: Option<i32> = m.clone().into();
        let back: Maybe<i32> = opt.into();
        prop_assert_eq!(m, back);

        let n: Maybe<i32> = Maybe::Nothing;
        let opt2: Option<i32> = n.clone().into();
        let back2: Maybe<i32> = opt2.into();
        prop_assert_eq!(n, back2);
    }

    // Either to Result and back
    #[test]
    fn either_result_roundtrip(x in any::<i32>()) {
        let e: Either<&str, i32> = Either::Right(x);
        let res: Result<i32, &str> = e.clone().into();
        let back: Either<&str, i32> = res.into();
        prop_assert_eq!(e, back);

        let e2: Either<&str, i32> = Either::Left("error");
        let res2: Result<i32, &str> = e2.clone().into();
        let back2: Either<&str, i32> = res2.into();
        prop_assert_eq!(e2, back2);
    }
}
