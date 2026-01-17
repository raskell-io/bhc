//! Tree-walking interpreter for Core IR.
//!
//! This module implements a simple interpreter for Core IR expressions.
//! It supports both lazy (Default Profile) and strict (Numeric Profile)
//! evaluation modes.
//!
//! # Example
//!
//! ```ignore
//! use bhc_core::eval::{Evaluator, EvalMode};
//! use bhc_session::Profile;
//!
//! // Create evaluator from profile
//! let evaluator = Evaluator::with_profile(Profile::Numeric);
//! let result = evaluator.eval(&expr)?;
//!
//! // Or create with explicit mode
//! let evaluator = Evaluator::new(EvalMode::Lazy);
//! let result = evaluator.eval(&expr)?;
//! ```

mod env;
mod value;

pub use env::Env;
pub use value::{Closure, DataValue, PrimOp, Thunk, Value};

use std::cell::RefCell;
use std::collections::HashMap;

use bhc_intern::Symbol;
use bhc_session::Profile;
use thiserror::Error;

use crate::{AltCon, Bind, Expr, Literal, Var, VarId};

/// Errors that can occur during evaluation.
#[derive(Debug, Error)]
pub enum EvalError {
    /// An unbound variable was referenced.
    #[error("unbound variable: {0}")]
    UnboundVariable(String),

    /// Type mismatch during evaluation.
    #[error("type error: expected {expected}, got {got}")]
    TypeError {
        /// The expected type.
        expected: String,
        /// The actual type found.
        got: String,
    },

    /// A pattern match failed (non-exhaustive).
    #[error("pattern match failure")]
    PatternMatchFailure,

    /// Division by zero.
    #[error("division by zero")]
    DivisionByZero,

    /// An explicit error was raised.
    #[error("error: {0}")]
    UserError(String),

    /// Recursion depth exceeded.
    #[error("stack overflow: maximum recursion depth exceeded")]
    StackOverflow,

    /// A thunk was forced during its own evaluation (black hole).
    #[error("infinite loop detected (black hole)")]
    BlackHole,
}

/// Evaluation mode controlling strictness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Lazy evaluation (Default Profile).
    /// Arguments are wrapped in thunks and forced on demand.
    Lazy,

    /// Strict evaluation (Numeric Profile).
    /// Arguments are evaluated before function application.
    Strict,
}

impl From<Profile> for EvalMode {
    fn from(profile: Profile) -> Self {
        if profile.is_strict_by_default() {
            Self::Strict
        } else {
            Self::Lazy
        }
    }
}

/// The Core IR evaluator.
pub struct Evaluator {
    /// Evaluation mode (lazy or strict).
    mode: EvalMode,
    /// Primitive operations environment.
    primitives: HashMap<Symbol, Value>,
    /// Maximum recursion depth.
    max_depth: usize,
    /// Current recursion depth.
    depth: RefCell<usize>,
}

impl Evaluator {
    /// Creates a new evaluator with the given mode.
    #[must_use]
    pub fn new(mode: EvalMode) -> Self {
        let mut primitives = HashMap::new();
        Self::register_primitives(&mut primitives);

        Self {
            mode,
            primitives,
            max_depth: 10000,
            depth: RefCell::new(0),
        }
    }

    /// Creates a new evaluator from a compilation profile.
    ///
    /// This is the preferred way to create an evaluator when working with
    /// the BHC compilation pipeline. The profile determines the evaluation
    /// semantics:
    ///
    /// - `Profile::Default` / `Profile::Server` → Lazy evaluation
    /// - `Profile::Numeric` / `Profile::Edge` → Strict evaluation
    ///
    /// # Example
    ///
    /// ```ignore
    /// use bhc_core::eval::Evaluator;
    /// use bhc_session::Profile;
    ///
    /// let eval = Evaluator::with_profile(Profile::Numeric);
    /// // Expressions are now evaluated strictly
    /// ```
    #[must_use]
    pub fn with_profile(profile: Profile) -> Self {
        Self::new(EvalMode::from(profile))
    }

    /// Returns the evaluation mode of this evaluator.
    #[must_use]
    pub fn mode(&self) -> EvalMode {
        self.mode
    }

    /// Returns true if this evaluator uses strict evaluation.
    #[must_use]
    pub fn is_strict(&self) -> bool {
        self.mode == EvalMode::Strict
    }

    /// Sets the maximum recursion depth.
    pub fn set_max_depth(&mut self, depth: usize) {
        self.max_depth = depth;
    }

    /// Registers primitive operations in the environment.
    fn register_primitives(prims: &mut HashMap<Symbol, Value>) {
        let ops = [
            ("+", PrimOp::AddInt),
            ("-", PrimOp::SubInt),
            ("*", PrimOp::MulInt),
            ("div", PrimOp::DivInt),
            ("mod", PrimOp::ModInt),
            ("negate", PrimOp::NegInt),
            ("+.", PrimOp::AddDouble),
            ("-.", PrimOp::SubDouble),
            ("*.", PrimOp::MulDouble),
            ("/.", PrimOp::DivDouble),
            ("==", PrimOp::EqInt),
            ("/=", PrimOp::EqInt), // We'll negate the result
            ("<", PrimOp::LtInt),
            ("<=", PrimOp::LeInt),
            (">", PrimOp::GtInt),
            (">=", PrimOp::GeInt),
            ("&&", PrimOp::AndBool),
            ("||", PrimOp::OrBool),
            ("not", PrimOp::NotBool),
            ("seq", PrimOp::Seq),
            ("error", PrimOp::Error),
            // UArray operations
            ("fromList", PrimOp::UArrayFromList),
            ("toList", PrimOp::UArrayToList),
            ("uarrayMap", PrimOp::UArrayMap),
            ("uarrayZipWith", PrimOp::UArrayZipWith),
            ("uarrayFold", PrimOp::UArrayFold),
            ("sum", PrimOp::UArraySum),
            ("length", PrimOp::UArrayLength),
            ("range", PrimOp::UArrayRange),
        ];

        for (name, op) in ops {
            prims.insert(Symbol::intern(name), Value::PrimOp(op));
        }

        // Register list constructors
        prims.insert(Symbol::intern("[]"), Value::nil());
        prims.insert(Symbol::intern(":"), Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern(":"),
                ty_con: bhc_types::TyCon::new(Symbol::intern("[]"), bhc_types::Kind::star_to_star()),
                tag: 1,
                arity: 2,
            },
            args: vec![],
        }));
    }

    /// Evaluates an expression to a value.
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails (unbound variable, type error, etc.)
    pub fn eval(&self, expr: &Expr, env: &Env) -> Result<Value, EvalError> {
        // Check recursion depth
        {
            let mut depth = self.depth.borrow_mut();
            if *depth >= self.max_depth {
                return Err(EvalError::StackOverflow);
            }
            *depth += 1;
        }

        let result = self.eval_inner(expr, env);

        // Decrement depth
        *self.depth.borrow_mut() -= 1;

        result
    }

    fn eval_inner(&self, expr: &Expr, env: &Env) -> Result<Value, EvalError> {
        match expr {
            Expr::Var(var, _) => self.eval_var(var, env),

            Expr::Lit(lit, _, _) => Ok(self.eval_lit(lit)),

            Expr::App(fun, arg, _) => self.eval_app(fun, arg, env),

            Expr::TyApp(fun, _, _) => {
                // Type applications are erased at runtime
                self.eval(fun, env)
            }

            Expr::Lam(var, body, _) => Ok(Value::Closure(Closure {
                var: var.clone(),
                body: body.clone(),
                env: env.clone(),
            })),

            Expr::TyLam(_, body, _) => {
                // Type lambdas are erased at runtime
                self.eval(body, env)
            }

            Expr::Let(bind, body, _) => self.eval_let(bind, body, env),

            Expr::Case(scrut, alts, _, _) => self.eval_case(scrut, alts, env),

            Expr::Lazy(inner, _) => {
                // The lazy escape hatch: always create a thunk, even in strict mode.
                // This allows code that genuinely needs lazy evaluation to work
                // correctly in Numeric Profile.
                Ok(Value::Thunk(Thunk {
                    expr: inner.clone(),
                    env: env.clone(),
                }))
            }

            Expr::Cast(e, _, _) => {
                // Coercions are erased at runtime
                self.eval(e, env)
            }

            Expr::Tick(_, e, _) => {
                // Ticks are ignored during evaluation
                self.eval(e, env)
            }

            Expr::Type(_, _) | Expr::Coercion(_, _) => {
                // Types and coercions shouldn't be evaluated
                Ok(Value::unit())
            }
        }
    }

    fn eval_var(&self, var: &Var, env: &Env) -> Result<Value, EvalError> {
        // First check the local environment
        if let Some(value) = env.lookup(var.id) {
            return self.force(value.clone());
        }

        // Then check primitives
        if let Some(value) = self.primitives.get(&var.name) {
            return Ok(value.clone());
        }

        // Check if it's a primitive by its raw name
        if let Some(op) = PrimOp::from_name(var.name.as_str()) {
            return Ok(Value::PrimOp(op));
        }

        Err(EvalError::UnboundVariable(var.name.to_string()))
    }

    fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int(n) => Value::Int(*n),
            Literal::Integer(n) => Value::Integer(*n),
            Literal::Float(n) => Value::Float(*n),
            Literal::Double(n) => Value::Double(*n),
            Literal::Char(c) => Value::Char(*c),
            Literal::String(s) => Value::String(s.as_str().into()),
        }
    }

    fn eval_app(&self, fun: &Expr, arg: &Expr, env: &Env) -> Result<Value, EvalError> {
        let fun_val = self.eval(fun, env)?;

        // Evaluate or thunk the argument based on mode
        let arg_val = match self.mode {
            EvalMode::Strict => self.eval(arg, env)?,
            EvalMode::Lazy => Value::Thunk(Thunk {
                expr: Box::new(arg.clone()),
                env: env.clone(),
            }),
        };

        self.apply(fun_val, arg_val)
    }

    fn apply(&self, fun: Value, arg: Value) -> Result<Value, EvalError> {
        match fun {
            Value::Closure(closure) => {
                let new_env = closure.env.extend(closure.var.id, arg);
                self.eval(&closure.body, &new_env)
            }

            Value::PrimOp(op) => self.apply_primop(op, vec![arg]),

            Value::PartialPrimOp(op, mut args) => {
                // Add argument to partial application
                args.push(arg);
                self.apply_primop(op, args)
            }

            Value::Data(mut data) if !data.is_saturated() => {
                // Partial application of data constructor
                data.args.push(arg);
                Ok(Value::Data(data))
            }

            Value::Thunk(thunk) => {
                // Force the thunk and retry
                let forced = self.force_thunk(&thunk)?;
                self.apply(forced, arg)
            }

            other => Err(EvalError::TypeError {
                expected: "function".to_string(),
                got: format!("{other:?}"),
            }),
        }
    }

    fn apply_primop(&self, op: PrimOp, args: Vec<Value>) -> Result<Value, EvalError> {
        // Check if we have enough arguments
        if args.len() < op.arity() {
            // Return a partially applied primop as a closure
            // For simplicity, we'll create a wrapper
            return self.partial_primop(op, args);
        }

        // Force all arguments
        let forced: Result<Vec<_>, _> = args.into_iter().map(|a| self.force(a)).collect();
        let args = forced?;

        match op {
            PrimOp::AddInt => {
                let a = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let b = args[1].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[1]),
                })?;
                Ok(Value::Int(a.wrapping_add(b)))
            }

            PrimOp::SubInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::Int(a.wrapping_sub(b)))
            }

            PrimOp::MulInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::Int(a.wrapping_mul(b)))
            }

            PrimOp::DivInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                if b == 0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Int(a / b))
            }

            PrimOp::ModInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                if b == 0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Int(a % b))
            }

            PrimOp::NegInt => {
                let a = args[0].as_int().unwrap_or(0);
                Ok(Value::Int(-a))
            }

            PrimOp::AddDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::Double(a + b))
            }

            PrimOp::SubDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::Double(a - b))
            }

            PrimOp::MulDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::Double(a * b))
            }

            PrimOp::DivDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::Double(a / b))
            }

            PrimOp::NegDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                Ok(Value::Double(-a))
            }

            PrimOp::EqInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::bool(a == b))
            }

            PrimOp::LtInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::bool(a < b))
            }

            PrimOp::LeInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::bool(a <= b))
            }

            PrimOp::GtInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::bool(a > b))
            }

            PrimOp::GeInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::bool(a >= b))
            }

            PrimOp::EqDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                #[allow(clippy::float_cmp)]
                Ok(Value::bool(a == b))
            }

            PrimOp::LtDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::bool(a < b))
            }

            PrimOp::AndBool => {
                let a = args[0].as_bool().unwrap_or(false);
                let b = args[1].as_bool().unwrap_or(false);
                Ok(Value::bool(a && b))
            }

            PrimOp::OrBool => {
                let a = args[0].as_bool().unwrap_or(false);
                let b = args[1].as_bool().unwrap_or(false);
                Ok(Value::bool(a || b))
            }

            PrimOp::NotBool => {
                let a = args[0].as_bool().unwrap_or(false);
                Ok(Value::bool(!a))
            }

            PrimOp::IntToDouble => {
                let a = args[0].as_int().unwrap_or(0);
                Ok(Value::Double(a as f64))
            }

            PrimOp::DoubleToInt => {
                let a = args[0].as_double().unwrap_or(0.0);
                #[allow(clippy::cast_possible_truncation)]
                Ok(Value::Int(a as i64))
            }

            PrimOp::EqChar => {
                let a = match &args[0] {
                    Value::Char(c) => *c,
                    _ => '\0',
                };
                let b = match &args[1] {
                    Value::Char(c) => *c,
                    _ => '\0',
                };
                Ok(Value::bool(a == b))
            }

            PrimOp::CharToInt => {
                let c = match &args[0] {
                    Value::Char(c) => *c,
                    _ => '\0',
                };
                Ok(Value::Int(i64::from(u32::from(c))))
            }

            PrimOp::IntToChar => {
                let n = args[0].as_int().unwrap_or(0);
                #[allow(clippy::cast_sign_loss)]
                let c = char::from_u32(n as u32).unwrap_or('\0');
                Ok(Value::Char(c))
            }

            PrimOp::Seq => {
                // seq forces first argument, returns second
                let _ = self.force(args[0].clone())?;
                Ok(args[1].clone())
            }

            PrimOp::Error => {
                let msg = match &args[0] {
                    Value::String(s) => s.to_string(),
                    other => format!("{other:?}"),
                };
                Err(EvalError::UserError(msg))
            }

            // UArray operations
            PrimOp::UArrayFromList => {
                // Convert a list to a UArray
                let list = &args[0];
                if let Some(arr) = Value::uarray_int_from_list(list) {
                    Ok(arr)
                } else if let Some(arr) = Value::uarray_double_from_list(list) {
                    Ok(arr)
                } else {
                    Err(EvalError::TypeError {
                        expected: "list of Int or Double".into(),
                        got: format!("{list:?}"),
                    })
                }
            }

            PrimOp::UArrayToList => {
                // Convert a UArray back to a list
                args[0].uarray_to_list().ok_or_else(|| EvalError::TypeError {
                    expected: "UArray".into(),
                    got: format!("{:?}", args[0]),
                })
            }

            PrimOp::UArrayMap => {
                // map f arr
                let f = &args[0];
                let arr = &args[1];

                match arr {
                    Value::UArrayInt(uarr) => {
                        let mapped: Result<Vec<i64>, _> = uarr
                            .as_slice()
                            .iter()
                            .map(|x| {
                                let result = self.apply(f.clone(), Value::Int(*x))?;
                                let forced = self.force(result)?;
                                forced.as_int().ok_or_else(|| EvalError::TypeError {
                                    expected: "Int".into(),
                                    got: format!("{forced:?}"),
                                })
                            })
                            .collect();
                        Ok(Value::UArrayInt(crate::uarray::UArray::from_vec(mapped?)))
                    }
                    Value::UArrayDouble(uarr) => {
                        let mapped: Result<Vec<f64>, _> = uarr
                            .as_slice()
                            .iter()
                            .map(|x| {
                                let result = self.apply(f.clone(), Value::Double(*x))?;
                                let forced = self.force(result)?;
                                forced.as_double().ok_or_else(|| EvalError::TypeError {
                                    expected: "Double".into(),
                                    got: format!("{forced:?}"),
                                })
                            })
                            .collect();
                        Ok(Value::UArrayDouble(crate::uarray::UArray::from_vec(mapped?)))
                    }
                    // Also support mapping over lists for convenience
                    _ if args[1].as_list().is_some() => {
                        let list = args[1].as_list().unwrap();
                        let mapped: Result<Vec<Value>, _> = list
                            .iter()
                            .map(|x| {
                                let result = self.apply(f.clone(), x.clone())?;
                                self.force(result)
                            })
                            .collect();
                        Ok(Value::from_list(mapped?))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "UArray or List".into(),
                        got: format!("{arr:?}"),
                    }),
                }
            }

            PrimOp::UArrayZipWith => {
                // zipWith f arr1 arr2
                let f = &args[0];
                let arr1 = &args[1];
                let arr2 = &args[2];

                match (arr1, arr2) {
                    (Value::UArrayInt(a), Value::UArrayInt(b)) => {
                        let zipped: Result<Vec<i64>, _> = a
                            .as_slice()
                            .iter()
                            .zip(b.as_slice().iter())
                            .map(|(x, y)| {
                                let result = self.apply(
                                    self.apply(f.clone(), Value::Int(*x))?,
                                    Value::Int(*y),
                                )?;
                                let forced = self.force(result)?;
                                forced.as_int().ok_or_else(|| EvalError::TypeError {
                                    expected: "Int".into(),
                                    got: format!("{forced:?}"),
                                })
                            })
                            .collect();
                        Ok(Value::UArrayInt(crate::uarray::UArray::from_vec(zipped?)))
                    }
                    (Value::UArrayDouble(a), Value::UArrayDouble(b)) => {
                        let zipped: Result<Vec<f64>, _> = a
                            .as_slice()
                            .iter()
                            .zip(b.as_slice().iter())
                            .map(|(x, y)| {
                                let result = self.apply(
                                    self.apply(f.clone(), Value::Double(*x))?,
                                    Value::Double(*y),
                                )?;
                                let forced = self.force(result)?;
                                forced.as_double().ok_or_else(|| EvalError::TypeError {
                                    expected: "Double".into(),
                                    got: format!("{forced:?}"),
                                })
                            })
                            .collect();
                        Ok(Value::UArrayDouble(crate::uarray::UArray::from_vec(zipped?)))
                    }
                    // Support lists
                    _ if arr1.as_list().is_some() && arr2.as_list().is_some() => {
                        let list1 = arr1.as_list().unwrap();
                        let list2 = arr2.as_list().unwrap();
                        let zipped: Result<Vec<Value>, _> = list1
                            .iter()
                            .zip(list2.iter())
                            .map(|(x, y)| {
                                let result = self.apply(
                                    self.apply(f.clone(), x.clone())?,
                                    y.clone(),
                                )?;
                                self.force(result)
                            })
                            .collect();
                        Ok(Value::from_list(zipped?))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "two UArrays or Lists".into(),
                        got: format!("{arr1:?}, {arr2:?}"),
                    }),
                }
            }

            PrimOp::UArrayFold => {
                // fold f init arr
                let f = &args[0];
                let init = args[1].clone();
                let arr = &args[2];

                match arr {
                    Value::UArrayInt(uarr) => {
                        let mut acc = init;
                        for x in uarr.as_slice() {
                            let result = self.apply(
                                self.apply(f.clone(), acc)?,
                                Value::Int(*x),
                            )?;
                            acc = self.force(result)?;
                        }
                        Ok(acc)
                    }
                    Value::UArrayDouble(uarr) => {
                        let mut acc = init;
                        for x in uarr.as_slice() {
                            let result = self.apply(
                                self.apply(f.clone(), acc)?,
                                Value::Double(*x),
                            )?;
                            acc = self.force(result)?;
                        }
                        Ok(acc)
                    }
                    _ if arr.as_list().is_some() => {
                        let list = arr.as_list().unwrap();
                        let mut acc = init;
                        for x in list {
                            let result = self.apply(
                                self.apply(f.clone(), acc)?,
                                x,
                            )?;
                            acc = self.force(result)?;
                        }
                        Ok(acc)
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "UArray or List".into(),
                        got: format!("{arr:?}"),
                    }),
                }
            }

            PrimOp::UArraySum => {
                // sum arr - works on both UArrays and lists
                match &args[0] {
                    Value::UArrayInt(arr) => Ok(Value::Int(arr.sum())),
                    Value::UArrayDouble(arr) => Ok(Value::Double(arr.sum())),
                    _ if args[0].as_list().is_some() => {
                        let list = args[0].as_list().unwrap();
                        // Try to sum as integers first
                        let ints: Option<i64> = list.iter().map(Value::as_int).try_fold(0i64, |acc, x| {
                            x.map(|n| acc.wrapping_add(n))
                        });
                        if let Some(sum) = ints {
                            return Ok(Value::Int(sum));
                        }
                        // Try as doubles
                        let doubles: Option<f64> = list.iter().map(Value::as_double).try_fold(0.0f64, |acc, x| {
                            x.map(|n| acc + n)
                        });
                        if let Some(sum) = doubles {
                            return Ok(Value::Double(sum));
                        }
                        Err(EvalError::TypeError {
                            expected: "list of numbers".into(),
                            got: format!("{:?}", args[0]),
                        })
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "UArray or List".into(),
                        got: format!("{:?}", args[0]),
                    }),
                }
            }

            PrimOp::UArrayLength => {
                match &args[0] {
                    Value::UArrayInt(arr) => Ok(Value::Int(arr.len() as i64)),
                    Value::UArrayDouble(arr) => Ok(Value::Int(arr.len() as i64)),
                    _ if args[0].as_list().is_some() => {
                        let list = args[0].as_list().unwrap();
                        Ok(Value::Int(list.len() as i64))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "UArray or List".into(),
                        got: format!("{:?}", args[0]),
                    }),
                }
            }

            PrimOp::UArrayRange => {
                // range start end
                let start = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let end = args[1].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[1]),
                })?;
                Ok(Value::UArrayInt(crate::uarray::UArray::range(start, end)))
            }
        }
    }

    fn partial_primop(&self, op: PrimOp, args: Vec<Value>) -> Result<Value, EvalError> {
        // Return a partially applied primop that stores accumulated arguments
        Ok(Value::PartialPrimOp(op, args))
    }

    fn eval_let(&self, bind: &Bind, body: &Expr, env: &Env) -> Result<Value, EvalError> {
        match bind {
            Bind::NonRec(var, rhs) => {
                let rhs_val = match self.mode {
                    EvalMode::Strict => self.eval(rhs, env)?,
                    EvalMode::Lazy => Value::Thunk(Thunk {
                        expr: rhs.clone(),
                        env: env.clone(),
                    }),
                };
                let new_env = env.extend(var.id, rhs_val);
                self.eval(body, &new_env)
            }

            Bind::Rec(bindings) => {
                // For recursive bindings, we need to create thunks that
                // reference the environment containing all bindings
                let mut new_env = env.clone();

                // First pass: create placeholder thunks
                for (var, rhs) in bindings {
                    let thunk = Value::Thunk(Thunk {
                        expr: rhs.clone(),
                        env: env.clone(), // Will be updated
                    });
                    new_env = new_env.extend(var.id, thunk);
                }

                // Second pass: update thunks with the complete environment
                for (var, rhs) in bindings {
                    let thunk = Value::Thunk(Thunk {
                        expr: rhs.clone(),
                        env: new_env.clone(),
                    });
                    new_env = new_env.extend(var.id, thunk);
                }

                self.eval(body, &new_env)
            }
        }
    }

    fn eval_case(&self, scrut: &Expr, alts: &[crate::Alt], env: &Env) -> Result<Value, EvalError> {
        // Evaluate scrutinee to WHNF
        let scrut_val = self.eval(scrut, env)?;
        let scrut_val = self.force(scrut_val)?;

        // Find matching alternative
        for alt in alts {
            if let Some(bindings) = self.match_pattern(&alt.con, &alt.binders, &scrut_val)? {
                let new_env = env.extend_many(bindings);
                return self.eval(&alt.rhs, &new_env);
            }
        }

        Err(EvalError::PatternMatchFailure)
    }

    fn match_pattern(
        &self,
        pattern: &AltCon,
        binders: &[Var],
        value: &Value,
    ) -> Result<Option<Vec<(VarId, Value)>>, EvalError> {
        match pattern {
            AltCon::Default => {
                // Default matches anything
                Ok(Some(Vec::new()))
            }

            AltCon::Lit(lit) => {
                // Match literal
                let matches = match (lit, value) {
                    (Literal::Int(a), Value::Int(b)) => *a == *b,
                    (Literal::Integer(a), Value::Integer(b)) => *a == *b,
                    (Literal::Float(a), Value::Float(b)) => (*a - *b).abs() < f32::EPSILON,
                    (Literal::Double(a), Value::Double(b)) => (*a - *b).abs() < f64::EPSILON,
                    (Literal::Char(a), Value::Char(b)) => *a == *b,
                    (Literal::String(a), Value::String(b)) => a.as_str() == b.as_ref(),
                    _ => false,
                };
                Ok(if matches { Some(Vec::new()) } else { None })
            }

            AltCon::DataCon(con) => {
                // Match data constructor
                if let Value::Data(data) = value {
                    if data.con.name == con.name && data.con.tag == con.tag {
                        // Bind constructor arguments to pattern variables
                        let bindings: Vec<_> = binders
                            .iter()
                            .zip(data.args.iter())
                            .map(|(var, val)| (var.id, val.clone()))
                            .collect();
                        return Ok(Some(bindings));
                    }
                }
                Ok(None)
            }
        }
    }

    /// Forces a value to WHNF (evaluates thunks).
    pub fn force(&self, value: Value) -> Result<Value, EvalError> {
        match value {
            Value::Thunk(thunk) => self.force_thunk(&thunk),
            other => Ok(other),
        }
    }

    fn force_thunk(&self, thunk: &Thunk) -> Result<Value, EvalError> {
        self.eval(&thunk.expr, &thunk.env)
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new(EvalMode::Lazy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_span::Span;
    use bhc_types::Ty;

    fn make_var(name: &str, id: usize) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id), Ty::Error)
    }

    fn make_int(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), Ty::Error, Span::default())
    }

    #[test]
    fn test_eval_literal() {
        let eval = Evaluator::new(EvalMode::Strict);
        let expr = make_int(42);
        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_lambda_app() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (\x -> x) 42
        let x = make_var("x", 0);
        let lam = Expr::Lam(x.clone(), Box::new(Expr::Var(x, Span::default())), Span::default());
        let app = Expr::App(Box::new(lam), Box::new(make_int(42)), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_let() {
        let eval = Evaluator::new(EvalMode::Strict);

        // let x = 42 in x
        let x = make_var("x", 0);
        let bind = Bind::NonRec(x.clone(), Box::new(make_int(42)));
        let body = Expr::Var(x, Span::default());
        let expr = Expr::Let(Box::new(bind), Box::new(body), Span::default());

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_primop_add() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (+) 1 2
        let add = Expr::Var(make_var("+", 100), Span::default());
        let app1 = Expr::App(Box::new(add), Box::new(make_int(1)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(2)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(3)));
    }

    #[test]
    fn test_eval_primop_mul() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (*) 6 7
        let mul = Expr::Var(make_var("*", 100), Span::default());
        let app1 = Expr::App(Box::new(mul), Box::new(make_int(6)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(7)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_nested_let() {
        let eval = Evaluator::new(EvalMode::Strict);

        // let x = 1 in let y = 2 in x + y
        let x = make_var("x", 0);
        let y = make_var("y", 1);
        let add = Expr::Var(make_var("+", 100), Span::default());

        let add_xy = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(Expr::Var(y.clone(), Span::default())),
            Span::default(),
        );

        let inner_let = Expr::Let(
            Box::new(Bind::NonRec(y, Box::new(make_int(2)))),
            Box::new(add_xy),
            Span::default(),
        );

        let outer_let = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(make_int(1)))),
            Box::new(inner_let),
            Span::default(),
        );

        let result = eval.eval(&outer_let, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(3)));
    }

    #[test]
    fn test_eval_comparison() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (<) 1 2
        let lt = Expr::Var(make_var("<", 100), Span::default());
        let app1 = Expr::App(Box::new(lt), Box::new(make_int(1)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(2)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_lazy_evaluation() {
        let eval = Evaluator::new(EvalMode::Lazy);

        // let x = error "boom" in 42
        // In lazy mode, x is never forced so no error
        let x = make_var("x", 0);
        let error_call = Expr::App(
            Box::new(Expr::Var(make_var("error", 100), Span::default())),
            Box::new(Expr::Lit(
                Literal::String(Symbol::intern("boom")),
                Ty::Error,
                Span::default(),
            )),
            Span::default(),
        );

        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(error_call))),
            Box::new(make_int(42)),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_list_operations() {
        // Test creating and inspecting lists
        let list = Value::from_list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);

        let items = list.as_list().unwrap();
        assert_eq!(items.len(), 3);
        assert!(matches!(items[0], Value::Int(1)));
        assert!(matches!(items[1], Value::Int(2)));
        assert!(matches!(items[2], Value::Int(3)));
    }

    // =========================================================================
    // UArray Tests - M0 Exit Criteria
    // =========================================================================

    #[test]
    fn test_uarray_from_list() {
        // Create a list [1, 2, 3, 4, 5]
        let list = Value::from_list(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
            Value::Int(5),
        ]);

        // Convert to UArray
        let result = Value::uarray_int_from_list(&list).unwrap();
        match result {
            Value::UArrayInt(arr) => {
                assert_eq!(arr.len(), 5);
                assert_eq!(arr.to_vec(), vec![1, 2, 3, 4, 5]);
            }
            _ => panic!("Expected UArrayInt"),
        }
    }

    #[test]
    fn test_uarray_sum() {
        let eval = Evaluator::new(EvalMode::Strict);
        let sum_var = Expr::Var(make_var("sum", 100), Span::default());

        // Build: sum [1, 2, 3, 4, 5]
        // First, create a let binding for the list
        let list_var = make_var("xs", 0);
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);
        let sum_app = Expr::App(Box::new(sum_var), Box::new(Expr::Var(list_var.clone(), Span::default())), Span::default());
        let expr = Expr::Let(
            Box::new(Bind::NonRec(list_var, Box::new(list_expr))),
            Box::new(sum_app),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(15)));
    }

    #[test]
    fn test_uarray_map() {
        let eval = Evaluator::new(EvalMode::Strict);

        // map (+1) [1, 2, 3, 4, 5]
        let add_one = {
            let x = make_var("x", 0);
            let add = Expr::Var(make_var("+", 100), Span::default());
            Expr::Lam(
                x.clone(),
                Box::new(Expr::App(
                    Box::new(Expr::App(
                        Box::new(add),
                        Box::new(Expr::Var(x, Span::default())),
                        Span::default(),
                    )),
                    Box::new(make_int(1)),
                    Span::default(),
                )),
                Span::default(),
            )
        };

        let map_var = Expr::Var(make_var("uarrayMap", 101), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);

        // Build: uarrayMap (+1) [1, 2, 3, 4, 5]
        let app1 = Expr::App(Box::new(map_var), Box::new(add_one), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(list_expr), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();

        // Should be a list [2, 3, 4, 5, 6]
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 5);
        assert_eq!(list[0].as_int(), Some(2));
        assert_eq!(list[1].as_int(), Some(3));
        assert_eq!(list[2].as_int(), Some(4));
        assert_eq!(list[3].as_int(), Some(5));
        assert_eq!(list[4].as_int(), Some(6));
    }

    #[test]
    fn test_m0_exit_criteria_sum_map() {
        // M0 Exit Criteria: sum (map (+1) [1,2,3,4,5]) == 20
        let eval = Evaluator::new(EvalMode::Strict);

        // Build: sum (map (+1) [1, 2, 3, 4, 5])
        let add_one = {
            let x = make_var("x", 0);
            let add = Expr::Var(make_var("+", 100), Span::default());
            Expr::Lam(
                x.clone(),
                Box::new(Expr::App(
                    Box::new(Expr::App(
                        Box::new(add),
                        Box::new(Expr::Var(x, Span::default())),
                        Span::default(),
                    )),
                    Box::new(make_int(1)),
                    Span::default(),
                )),
                Span::default(),
            )
        };

        let map_var = Expr::Var(make_var("uarrayMap", 101), Span::default());
        let sum_var = Expr::Var(make_var("sum", 102), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);

        // map (+1) [1, 2, 3, 4, 5]
        let map_app = Expr::App(
            Box::new(Expr::App(Box::new(map_var), Box::new(add_one), Span::default())),
            Box::new(list_expr),
            Span::default(),
        );

        // sum (map (+1) [1, 2, 3, 4, 5])
        let sum_app = Expr::App(Box::new(sum_var), Box::new(map_app), Span::default());

        let result = eval.eval(&sum_app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(20)));
    }

    #[test]
    fn test_uarray_zipwith() {
        let eval = Evaluator::new(EvalMode::Strict);

        // zipWith (+) [1, 2, 3] [4, 5, 6]
        let add = Expr::Var(make_var("+", 100), Span::default());
        let zip_var = Expr::Var(make_var("uarrayZipWith", 101), Span::default());
        let list1 = build_list_expr(vec![1, 2, 3]);
        let list2 = build_list_expr(vec![4, 5, 6]);

        // zipWith (+) [1, 2, 3] [4, 5, 6]
        let app1 = Expr::App(Box::new(zip_var), Box::new(add), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(list1), Span::default());
        let app3 = Expr::App(Box::new(app2), Box::new(list2), Span::default());

        let result = eval.eval(&app3, &Env::new()).unwrap();

        // Should be [5, 7, 9]
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_int(), Some(5));
        assert_eq!(list[1].as_int(), Some(7));
        assert_eq!(list[2].as_int(), Some(9));
    }

    #[test]
    fn test_uarray_dot_product() {
        // Test dot product: sum (zipWith (*) [1, 2, 3] [4, 5, 6])
        // = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let eval = Evaluator::new(EvalMode::Strict);

        let mul = Expr::Var(make_var("*", 100), Span::default());
        let zip_var = Expr::Var(make_var("uarrayZipWith", 101), Span::default());
        let sum_var = Expr::Var(make_var("sum", 102), Span::default());
        let list1 = build_list_expr(vec![1, 2, 3]);
        let list2 = build_list_expr(vec![4, 5, 6]);

        // zipWith (*) [1, 2, 3] [4, 5, 6]
        let zip_app = Expr::App(
            Box::new(Expr::App(
                Box::new(Expr::App(Box::new(zip_var), Box::new(mul), Span::default())),
                Box::new(list1),
                Span::default(),
            )),
            Box::new(list2),
            Span::default(),
        );

        // sum (zipWith (*) [1, 2, 3] [4, 5, 6])
        let sum_app = Expr::App(Box::new(sum_var), Box::new(zip_app), Span::default());

        let result = eval.eval(&sum_app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(32)));
    }

    // Helper function to build a list expression
    fn build_list_expr(values: Vec<i64>) -> Expr {
        // Start with nil
        let mut expr = Expr::Var(
            Var::new(Symbol::intern("[]"), VarId::new(9999), Ty::Error),
            Span::default(),
        );

        // Build from the end: (:) x ((:) y nil)
        for val in values.into_iter().rev() {
            let cons_var = Expr::Var(
                Var::new(Symbol::intern(":"), VarId::new(9998), Ty::Error),
                Span::default(),
            );
            let val_expr = make_int(val);
            expr = Expr::App(
                Box::new(Expr::App(Box::new(cons_var), Box::new(val_expr), Span::default())),
                Box::new(expr),
                Span::default(),
            );
        }

        expr
    }

    // =========================================================================
    // M1 Tests - Numeric Profile Strict-by-Default
    // =========================================================================

    #[test]
    fn test_eval_mode_from_profile() {
        // Default and Server profiles use lazy evaluation
        assert_eq!(EvalMode::from(Profile::Default), EvalMode::Lazy);
        assert_eq!(EvalMode::from(Profile::Server), EvalMode::Lazy);

        // Numeric and Edge profiles use strict evaluation
        assert_eq!(EvalMode::from(Profile::Numeric), EvalMode::Strict);
        assert_eq!(EvalMode::from(Profile::Edge), EvalMode::Strict);
    }

    #[test]
    fn test_evaluator_with_profile() {
        let lazy_eval = Evaluator::with_profile(Profile::Default);
        assert!(!lazy_eval.is_strict());
        assert_eq!(lazy_eval.mode(), EvalMode::Lazy);

        let strict_eval = Evaluator::with_profile(Profile::Numeric);
        assert!(strict_eval.is_strict());
        assert_eq!(strict_eval.mode(), EvalMode::Strict);
    }

    #[test]
    fn test_strict_evaluation_forces_let_bindings() {
        // In strict mode, let bindings are eagerly evaluated
        // let x = 1 + 2 in x * 3
        // This is a key M1 exit criterion: strict evaluation of let-bindings

        let eval = Evaluator::with_profile(Profile::Numeric);

        let x = make_var("x", 0);
        let add = Expr::Var(make_var("+", 100), Span::default());
        let mul = Expr::Var(make_var("*", 101), Span::default());

        // 1 + 2
        let add_expr = Expr::App(
            Box::new(Expr::App(Box::new(add), Box::new(make_int(1)), Span::default())),
            Box::new(make_int(2)),
            Span::default(),
        );

        // x * 3
        let mul_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(mul),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(make_int(3)),
            Span::default(),
        );

        // let x = 1 + 2 in x * 3
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(add_expr))),
            Box::new(mul_expr),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(9))); // (1 + 2) * 3 = 9
    }

    #[test]
    fn test_strict_mode_evaluates_unused_bindings() {
        // In strict mode, error in binding is raised even if binding is unused
        // let x = error "boom" in 42 -> error in strict mode
        let eval = Evaluator::with_profile(Profile::Numeric);

        let x = make_var("x", 0);
        let error_call = Expr::App(
            Box::new(Expr::Var(make_var("error", 100), Span::default())),
            Box::new(Expr::Lit(
                Literal::String(Symbol::intern("boom")),
                Ty::Error,
                Span::default(),
            )),
            Span::default(),
        );

        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(error_call))),
            Box::new(make_int(42)),
            Span::default(),
        );

        // In strict mode, the error binding is evaluated even though unused
        let result = eval.eval(&expr, &Env::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_escape_hatch_in_strict_mode() {
        // The lazy { } escape hatch should make evaluation lazy even in strict mode
        // let x = lazy { error "boom" } in 42 -> 42 (thunk not forced)
        let eval = Evaluator::with_profile(Profile::Numeric);
        assert!(eval.is_strict());

        let x = make_var("x", 0);
        let error_call = Expr::App(
            Box::new(Expr::Var(make_var("error", 100), Span::default())),
            Box::new(Expr::Lit(
                Literal::String(Symbol::intern("boom")),
                Ty::Error,
                Span::default(),
            )),
            Span::default(),
        );

        // Wrap the error in lazy { }
        let lazy_error = Expr::Lazy(Box::new(error_call), Span::default());

        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(lazy_error))),
            Box::new(make_int(42)),
            Span::default(),
        );

        // With lazy escape hatch, the error is not evaluated
        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_lazy_escape_hatch_creates_thunk() {
        // lazy { expr } should create a thunk
        let eval = Evaluator::with_profile(Profile::Numeric);

        let lazy_expr = Expr::Lazy(Box::new(make_int(42)), Span::default());
        let result = eval.eval(&lazy_expr, &Env::new()).unwrap();

        // The result should be a thunk
        assert!(result.is_thunk());

        // When forced, it should give 42
        let forced = eval.force(result).unwrap();
        assert!(matches!(forced, Value::Int(42)));
    }

    #[test]
    fn test_m1_exit_criteria_strict_let_binding() {
        // M1 Exit Criterion: strict evaluation of `let x = 1 + 2 in x * 3`
        // evaluates to 9 with no thunks in Numeric Profile
        let eval = Evaluator::with_profile(Profile::Numeric);

        let x = make_var("x", 0);
        let add = Expr::Var(make_var("+", 100), Span::default());
        let mul = Expr::Var(make_var("*", 101), Span::default());

        // 1 + 2
        let add_expr = Expr::App(
            Box::new(Expr::App(Box::new(add), Box::new(make_int(1)), Span::default())),
            Box::new(make_int(2)),
            Span::default(),
        );

        // x * 3
        let mul_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(mul),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(make_int(3)),
            Span::default(),
        );

        // let x = 1 + 2 in x * 3
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(add_expr))),
            Box::new(mul_expr),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();

        // Result should be 9
        assert!(matches!(result, Value::Int(9)));

        // Result should not be a thunk (no thunks in strict mode result)
        assert!(!result.is_thunk());
    }
}
