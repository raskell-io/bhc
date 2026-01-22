//! RWS Monad - Combined Reader, Writer, State
//!
//! RWS combines three effects into one transformer:
//! - Reader: Access to a read-only environment
//! - Writer: Accumulating output (logging)
//! - State: Mutable state
//!
//! # Overview
//!
//! RWS is useful when you need all three effects together without the overhead
//! of stacking individual transformers. It's particularly common in interpreters,
//! parsers, and code generators.
//!
//! # Example
//!
//! ```
//! use bhc_transformers::rws::{RWS, RWST};
//!
//! // Environment type
//! #[derive(Clone)]
//! struct Config {
//!     debug: bool,
//! }
//!
//! // Log entries
//! type Log = Vec<String>;
//!
//! // State
//! type Counter = i32;
//!
//! // Example computation
//! let computation: RWS<Config, Log, Counter, i32> = RWS::new(|env: Config, state: Counter| {
//!     let mut log = Vec::new();
//!     if env.debug {
//!         log.push(format!("Counter is {}", state));
//!     }
//!     let result = state * 2;
//!     let new_state = state + 1;
//!     (result, log, new_state)
//! });
//!
//! let config = Config { debug: true };
//! let (result, log, final_state) = computation.run(config, 5);
//! assert_eq!(result, 10);
//! assert_eq!(final_state, 6);
//! assert!(!log.is_empty());
//! ```

use std::marker::PhantomData;

/// The RWS monad combining Reader, Writer, and State
///
/// - `R`: Reader environment type (read-only)
/// - `W`: Writer output type (must be a monoid for append)
/// - `S`: State type (mutable)
/// - `A`: Result type
pub struct RWS<R, W, S, A> {
    run_rws: Box<dyn Fn(R, S) -> (A, W, S)>,
}

impl<R, W, S, A> RWS<R, W, S, A>
where
    R: 'static,
    W: 'static,
    S: 'static,
    A: 'static,
{
    /// Create a new RWS computation
    ///
    /// # Example
    ///
    /// ```
    /// use bhc_transformers::rws::RWS;
    ///
    /// let comp: RWS<i32, Vec<String>, i32, i32> = RWS::new(|r, s| {
    ///     (r + s, vec!["computed".to_string()], s + 1)
    /// });
    /// ```
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(R, S) -> (A, W, S) + 'static,
    {
        RWS {
            run_rws: Box::new(f),
        }
    }

    /// Run the RWS computation with an environment and initial state
    pub fn run(self, r: R, s: S) -> (A, W, S) {
        (self.run_rws)(r, s)
    }

    /// Run and return only the result
    pub fn eval(self, r: R, s: S) -> A {
        let (a, _, _) = self.run(r, s);
        a
    }

    /// Run and return only the final state
    pub fn exec(self, r: R, s: S) -> S {
        let (_, _, s) = self.run(r, s);
        s
    }
}

impl<R, W, S, A> RWS<R, W, S, A>
where
    R: Clone + 'static,
    W: Default + 'static,
    S: Clone + 'static,
    A: 'static,
{
    /// Lift a pure value into RWS
    pub fn pure(a: A) -> Self
    where
        A: Clone,
    {
        RWS::new(move |_, s| (a.clone(), W::default(), s))
    }

    /// Map over the result
    pub fn map<B, F>(self, f: F) -> RWS<R, W, S, B>
    where
        B: 'static,
        F: Fn(A) -> B + 'static,
    {
        RWS::new(move |r, s| {
            let (a, w, s) = (self.run_rws)(r, s);
            (f(a), w, s)
        })
    }
}

impl<R, W, S, A> RWS<R, W, S, A>
where
    R: Clone + 'static,
    W: Default + Extend<<W as IntoIterator>::Item> + IntoIterator + 'static,
    S: Clone + 'static,
    A: 'static,
{
    /// Chain computations
    pub fn and_then<B, F>(self, f: F) -> RWS<R, W, S, B>
    where
        B: 'static,
        F: Fn(A) -> RWS<R, W, S, B> + 'static,
    {
        RWS::new(move |r: R, s| {
            let (a, mut w1, s1) = (self.run_rws)(r.clone(), s);
            let rws_b = f(a);
            let (b, w2, s2) = (rws_b.run_rws)(r, s1);
            w1.extend(w2);
            (b, w1, s2)
        })
    }
}

// Reader operations
impl<R, W, S, A> RWS<R, W, S, A>
where
    R: Clone + 'static,
    W: Default + 'static,
    S: Clone + 'static,
    A: 'static,
{
    /// Get the environment
    pub fn ask() -> RWS<R, W, S, R> {
        RWS::new(|r, s| (r, W::default(), s))
    }

    /// Get a value derived from the environment
    pub fn asks<F>(f: F) -> RWS<R, W, S, A>
    where
        F: Fn(R) -> A + 'static,
    {
        RWS::new(move |r, s| (f(r), W::default(), s))
    }

    /// Run with a modified environment
    pub fn local<F>(self, f: F) -> Self
    where
        F: Fn(R) -> R + 'static,
    {
        RWS::new(move |r, s| (self.run_rws)(f(r), s))
    }
}

// Writer operations
impl<R, W, S> RWS<R, W, S, ()>
where
    R: Clone + 'static,
    W: Clone + 'static,
    S: Clone + 'static,
{
    /// Write a value to the log
    pub fn tell(w: W) -> Self {
        RWS::new(move |_, s| ((), w.clone(), s))
    }
}

impl<R, W, S, A> RWS<R, W, S, A>
where
    R: Clone + 'static,
    W: Clone + Default + 'static,
    S: Clone + 'static,
    A: Clone + 'static,
{
    /// Get the accumulated output along with the result
    pub fn listen(self) -> RWS<R, W, S, (A, W)> {
        RWS::new(move |r, s| {
            let (a, w, s) = (self.run_rws)(r, s);
            ((a, w.clone()), w, s)
        })
    }

    /// Modify the output based on the result
    pub fn censor<F>(self, f: F) -> Self
    where
        F: Fn(W) -> W + 'static,
    {
        RWS::new(move |r, s| {
            let (a, w, s) = (self.run_rws)(r, s);
            (a, f(w), s)
        })
    }
}

// State operations
impl<R, W, S, A> RWS<R, W, S, A>
where
    R: Clone + 'static,
    W: Default + 'static,
    S: Clone + 'static,
    A: 'static,
{
    /// Get the current state
    pub fn get() -> RWS<R, W, S, S> {
        RWS::new(|_, s: S| (s.clone(), W::default(), s))
    }

    /// Set the state
    pub fn put(s: S) -> RWS<R, W, S, ()>
    where
        S: Clone,
    {
        RWS::new(move |_, _| ((), W::default(), s.clone()))
    }

    /// Modify the state with a function
    pub fn modify<F>(f: F) -> RWS<R, W, S, ()>
    where
        F: Fn(S) -> S + 'static,
    {
        RWS::new(move |_, s| ((), W::default(), f(s)))
    }

    /// Get a value derived from the state
    pub fn gets<F>(f: F) -> RWS<R, W, S, A>
    where
        F: Fn(S) -> A + 'static,
    {
        RWS::new(move |_, s: S| (f(s.clone()), W::default(), s))
    }
}

/// RWST monad transformer
///
/// Wraps another monad with Reader, Writer, State effects.
pub struct RWST<R, W, S, M, A> {
    run_rwst: Box<dyn Fn(R, S) -> M>,
    _phantom: PhantomData<(W, A)>,
}

impl<R, W, S, M, A> RWST<R, W, S, M, A>
where
    R: 'static,
    W: 'static,
    S: 'static,
    M: 'static,
    A: 'static,
{
    /// Create a new RWST computation
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(R, S) -> M + 'static,
    {
        RWST {
            run_rwst: Box::new(f),
            _phantom: PhantomData,
        }
    }

    /// Run the RWST computation
    pub fn run(self, r: R, s: S) -> M {
        (self.run_rwst)(r, s)
    }
}

/// Strict version of RWS that evaluates state strictly
pub struct RWS_<R, W, S, A> {
    run_rws: Box<dyn Fn(R, S) -> (A, W, S)>,
}

impl<R, W, S, A> RWS_<R, W, S, A>
where
    R: 'static,
    W: 'static,
    S: 'static,
    A: 'static,
{
    /// Create a new strict RWS computation
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(R, S) -> (A, W, S) + 'static,
    {
        RWS_ {
            run_rws: Box::new(f),
        }
    }

    /// Run the computation
    pub fn run(self, r: R, s: S) -> (A, W, S) {
        (self.run_rws)(r, s)
    }
}

/// Helper to create an RWS that only reads
pub fn reader<R, W, S, A, F>(f: F) -> RWS<R, W, S, A>
where
    R: Clone + 'static,
    W: Default + 'static,
    S: Clone + 'static,
    A: 'static,
    F: Fn(R) -> A + 'static,
{
    RWS::new(move |r, s| (f(r), W::default(), s))
}

/// Helper to create an RWS that only writes
pub fn writer<R, W, S, A>(a: A, w: W) -> RWS<R, W, S, A>
where
    R: Clone + 'static,
    W: Clone + 'static,
    S: Clone + 'static,
    A: Clone + 'static,
{
    RWS::new(move |_, s| (a.clone(), w.clone(), s))
}

/// Helper to create an RWS that only modifies state
pub fn state<R, W, S, A, F>(f: F) -> RWS<R, W, S, A>
where
    R: Clone + 'static,
    W: Default + 'static,
    S: 'static,
    A: 'static,
    F: Fn(S) -> (A, S) + 'static,
{
    RWS::new(move |_, s| {
        let (a, new_s) = f(s);
        (a, W::default(), new_s)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Config {
        multiplier: i32,
    }

    type Log = Vec<String>;
    type Counter = i32;

    #[test]
    fn test_rws_pure() {
        let rws: RWS<Config, Log, Counter, i32> = RWS::pure(42);
        let (result, log, state) = rws.run(Config { multiplier: 2 }, 0);
        assert_eq!(result, 42);
        assert!(log.is_empty());
        assert_eq!(state, 0);
    }

    #[test]
    fn test_rws_ask() {
        let rws: RWS<Config, Log, Counter, Config> = RWS::<Config, Log, Counter, Config>::ask();
        let (config, _, _) = rws.run(Config { multiplier: 5 }, 0);
        assert_eq!(config.multiplier, 5);
    }

    #[test]
    fn test_rws_asks() {
        let rws: RWS<Config, Log, Counter, i32> = RWS::asks(|c: Config| c.multiplier);
        let (mult, _, _) = rws.run(Config { multiplier: 7 }, 0);
        assert_eq!(mult, 7);
    }

    #[test]
    fn test_rws_tell() {
        let rws: RWS<Config, Log, Counter, ()> = RWS::tell(vec!["hello".to_string()]);
        let (_, log, _) = rws.run(Config { multiplier: 1 }, 0);
        assert_eq!(log, vec!["hello"]);
    }

    #[test]
    fn test_rws_get() {
        let rws: RWS<Config, Log, Counter, Counter> = RWS::<Config, Log, Counter, Counter>::get();
        let (state, _, _) = rws.run(Config { multiplier: 1 }, 42);
        assert_eq!(state, 42);
    }

    #[test]
    fn test_rws_put() {
        let rws: RWS<Config, Log, Counter, ()> = RWS::<Config, Log, Counter, ()>::put(100);
        let (_, _, state) = rws.run(Config { multiplier: 1 }, 0);
        assert_eq!(state, 100);
    }

    #[test]
    fn test_rws_modify() {
        let rws: RWS<Config, Log, Counter, ()> = RWS::<Config, Log, Counter, ()>::modify(|s| s + 10);
        let (_, _, state) = rws.run(Config { multiplier: 1 }, 5);
        assert_eq!(state, 15);
    }

    #[test]
    fn test_rws_map() {
        let rws: RWS<Config, Log, Counter, i32> = RWS::pure(10);
        let mapped = rws.map(|x| x * 2);
        let (result, _, _) = mapped.run(Config { multiplier: 1 }, 0);
        assert_eq!(result, 20);
    }

    #[test]
    fn test_rws_and_then() {
        let rws1: RWS<Config, Log, Counter, i32> = RWS::new(|_, s| (s, vec!["first".to_string()], s + 1));

        let rws2 = rws1.and_then(|x| RWS::new(move |_, s| (x * 2, vec!["second".to_string()], s + 1)));

        let (result, log, state) = rws2.run(Config { multiplier: 1 }, 5);
        assert_eq!(result, 10); // 5 * 2
        assert_eq!(log, vec!["first", "second"]);
        assert_eq!(state, 7); // 5 + 1 + 1
    }

    #[test]
    fn test_rws_local() {
        let rws: RWS<Config, Log, Counter, i32> = RWS::asks(|c: Config| c.multiplier);
        let local_rws = rws.local(|mut c: Config| {
            c.multiplier = 100;
            c
        });
        let (mult, _, _) = local_rws.run(Config { multiplier: 1 }, 0);
        assert_eq!(mult, 100);
    }

    #[test]
    fn test_rws_listen() {
        let rws: RWS<Config, Log, Counter, i32> = RWS::new(|_, s| (42, vec!["log".to_string()], s));

        let listened = rws.listen();
        let ((result, captured_log), log, _) = listened.run(Config { multiplier: 1 }, 0);

        assert_eq!(result, 42);
        assert_eq!(captured_log, vec!["log"]);
        assert_eq!(log, vec!["log"]);
    }

    #[test]
    fn test_rws_censor() {
        let rws: RWS<Config, Log, Counter, i32> =
            RWS::new(|_, s| (42, vec!["secret".to_string()], s));

        let censored = rws.censor(|_| vec!["[REDACTED]".to_string()]);
        let (_, log, _) = censored.run(Config { multiplier: 1 }, 0);

        assert_eq!(log, vec!["[REDACTED]"]);
    }

    #[test]
    fn test_rws_combined() {
        // A more realistic example combining all effects
        let computation: RWS<Config, Log, Counter, String> = RWS::new(|env: Config, state| {
            let mut log = Vec::new();
            log.push(format!("Starting with state {}", state));

            let result = state * env.multiplier;
            log.push(format!("Computed {} * {} = {}", state, env.multiplier, result));

            let new_state = state + 1;
            (format!("Result: {}", result), log, new_state)
        });

        let (result, log, state) = computation.run(Config { multiplier: 3 }, 10);

        assert_eq!(result, "Result: 30");
        assert_eq!(log.len(), 2);
        assert_eq!(state, 11);
    }

    #[test]
    fn test_reader_helper() {
        let rws = reader::<i32, Vec<String>, i32, i32, _>(|r| r * 2);
        let (result, _, _) = rws.run(21, 0);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_state_helper() {
        let rws = state::<i32, Vec<String>, i32, i32, _>(|s| (s * 2, s + 1));
        let (result, _, new_state) = rws.run(0, 5);
        assert_eq!(result, 10);
        assert_eq!(new_state, 6);
    }

    #[test]
    fn test_eval_exec() {
        let rws: RWS<i32, Vec<String>, i32, i32> = RWS::new(|r, s| (r + s, vec![], s + 1));

        let result = RWS::new(|r: i32, s: i32| (r + s, Vec::<String>::new(), s + 1)).eval(10, 5);
        assert_eq!(result, 15);

        let state = RWS::new(|_r: i32, s: i32| (0, Vec::<String>::new(), s + 1)).exec(10, 5);
        assert_eq!(state, 6);
    }
}
