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
//!
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
        ];

        for (name, op) in ops {
            prims.insert(Symbol::intern(name), Value::PrimOp(op));
        }
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
}
