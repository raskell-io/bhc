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

use bhc_index::Idx;
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
    /// Module-level environment for top-level bindings.
    /// This is always consulted for variable lookups, allowing recursive
    /// functions to find themselves without capturing a circular environment.
    module_env: RefCell<Option<Env>>,
    /// Stack of recursive environments for local let bindings.
    /// When evaluating the body of a recursive let, we push the environment
    /// containing all the recursive bindings here, so that closures can find
    /// their recursive references.
    rec_env_stack: RefCell<Vec<Env>>,
    /// Captured IO output from print/putStrLn/putStr operations.
    io_output: RefCell<String>,
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
            module_env: RefCell::new(None),
            rec_env_stack: RefCell::new(Vec::new()),
            io_output: RefCell::new(String::new()),
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

    /// Sets the module-level environment.
    ///
    /// This environment is always consulted when looking up variables,
    /// allowing recursive functions to find themselves without needing
    /// to capture a circular environment in their closures.
    pub fn set_module_env(&self, env: Env) {
        *self.module_env.borrow_mut() = Some(env);
    }

    /// Clears the module-level environment.
    pub fn clear_module_env(&self) {
        *self.module_env.borrow_mut() = None;
    }

    /// Returns the captured IO output from print/putStrLn/putStr operations.
    #[must_use]
    pub fn take_io_output(&self) -> String {
        self.io_output.borrow_mut().split_off(0)
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
            ("map", PrimOp::UArrayMap), // Standard list map
            ("uarrayZipWith", PrimOp::UArrayZipWith),
            ("uarrayFold", PrimOp::UArrayFold),
            ("sum", PrimOp::UArraySum),
            ("length", PrimOp::UArrayLength),
            ("range", PrimOp::UArrayRange),
            // List operations
            ("++", PrimOp::Concat),
            ("concat", PrimOp::Concat),
            ("concatMap", PrimOp::ConcatMap),
            ("append", PrimOp::Append),
            // Monad operations (list monad)
            (">>=", PrimOp::MonadBind),
            (">>", PrimOp::MonadThen),
            ("return", PrimOp::ListReturn),
            // Additional list operations
            ("foldr", PrimOp::Foldr),
            ("foldl", PrimOp::Foldl),
            ("foldl'", PrimOp::FoldlStrict),
            ("filter", PrimOp::Filter),
            ("zip", PrimOp::Zip),
            ("zipWith", PrimOp::ZipWith),
            ("take", PrimOp::Take),
            ("drop", PrimOp::Drop),
            ("head", PrimOp::Head),
            ("tail", PrimOp::Tail),
            ("last", PrimOp::Last),
            ("init", PrimOp::Init),
            ("reverse", PrimOp::Reverse),
            ("null", PrimOp::Null),
            ("!!", PrimOp::Index),
            ("replicate", PrimOp::Replicate),
            ("enumFromTo", PrimOp::EnumFromTo),
            // Char operations
            ("ord", PrimOp::CharToInt),
            ("chr", PrimOp::IntToChar),
            // Additional list/prelude operations
            ("even", PrimOp::Even),
            ("odd", PrimOp::Odd),
            ("elem", PrimOp::Elem),
            ("notElem", PrimOp::NotElem),
            ("takeWhile", PrimOp::TakeWhile),
            ("dropWhile", PrimOp::DropWhile),
            ("span", PrimOp::Span),
            ("break", PrimOp::Break),
            ("splitAt", PrimOp::SplitAt),
            ("iterate", PrimOp::Iterate),
            ("repeat", PrimOp::Repeat),
            ("cycle", PrimOp::Cycle),
            ("lookup", PrimOp::Lookup),
            ("unzip", PrimOp::Unzip),
            ("product", PrimOp::Product),
            ("flip", PrimOp::Flip),
            ("min", PrimOp::Min),
            ("max", PrimOp::Max),
            ("fromIntegral", PrimOp::FromIntegral),
            ("toInteger", PrimOp::FromIntegral),
            ("maybe", PrimOp::MaybeElim),
            ("fromMaybe", PrimOp::FromMaybe),
            ("either", PrimOp::EitherElim),
            ("isJust", PrimOp::IsJust),
            ("isNothing", PrimOp::IsNothing),
            ("abs", PrimOp::Abs),
            ("signum", PrimOp::Signum),
            ("curry", PrimOp::Curry),
            ("uncurry", PrimOp::Uncurry),
            ("swap", PrimOp::Swap),
            ("any", PrimOp::Any),
            ("all", PrimOp::All),
            ("and", PrimOp::And),
            ("or", PrimOp::Or),
            ("lines", PrimOp::Lines),
            ("unlines", PrimOp::Unlines),
            ("words", PrimOp::Words),
            ("unwords", PrimOp::Unwords),
            ("show", PrimOp::Show),
            ("id", PrimOp::Id),
            ("const", PrimOp::Const),
            // IO operations
            ("putStrLn", PrimOp::PutStrLn),
            ("putStr", PrimOp::PutStr),
            ("print", PrimOp::Print),
            ("getLine", PrimOp::GetLine),
        ];

        for (name, op) in ops {
            prims.insert(Symbol::intern(name), Value::PrimOp(op));
        }

        // Register list constructors
        prims.insert(Symbol::intern("[]"), Value::nil());
        prims.insert(
            Symbol::intern(":"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern(":"),
                    ty_con: bhc_types::TyCon::new(
                        Symbol::intern("[]"),
                        bhc_types::Kind::star_to_star(),
                    ),
                    tag: 1,
                    arity: 2,
                },
                args: vec![],
            }),
        );

        // Register tuple constructors
        // Unit tuple ()
        prims.insert(
            Symbol::intern("()"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("()"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 0,
                },
                args: vec![],
            }),
        );

        // Pair (,)
        prims.insert(
            Symbol::intern("(,)"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("(,)"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("(,)"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 2,
                },
                args: vec![],
            }),
        );

        // Triple (,,)
        prims.insert(
            Symbol::intern("(,,)"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("(,,)"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("(,,)"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 3,
                },
                args: vec![],
            }),
        );

        // Quadruple (,,,)
        prims.insert(
            Symbol::intern("(,,,)"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("(,,,)"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("(,,,)"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 4,
                },
                args: vec![],
            }),
        );

        // Quintuple (,,,,)
        prims.insert(
            Symbol::intern("(,,,,)"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("(,,,,)"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("(,,,,)"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 5,
                },
                args: vec![],
            }),
        );

        // Boolean constructors and otherwise
        prims.insert(Symbol::intern("True"), Value::bool(true));
        prims.insert(Symbol::intern("False"), Value::bool(false));
        prims.insert(Symbol::intern("otherwise"), Value::bool(true));
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

        // Then check the recursive environment stack (for local recursive lets)
        // This allows closures created in recursive bindings to find their
        // siblings and themselves
        for rec_env in self.rec_env_stack.borrow().iter().rev() {
            if let Some(value) = rec_env.lookup(var.id) {
                return self.force(value.clone());
            }
        }

        // Then check the module-level environment
        // This is crucial for recursive functions: when a closure is applied,
        // it can find other module-level bindings through this env
        if let Some(ref module_env) = *self.module_env.borrow() {
            if let Some(value) = module_env.lookup(var.id) {
                return self.force(value.clone());
            }
        }

        // Then check primitives
        if let Some(value) = self.primitives.get(&var.name) {
            // Arity-0 primops (like getLine) execute immediately
            if let Value::PrimOp(op) = value {
                if op.arity() == 0 {
                    return self.apply_primop(*op, vec![]);
                }
            }
            return Ok(value.clone());
        }

        // Check if it's a primitive by its raw name
        if let Some(op) = PrimOp::from_name(var.name.as_str()) {
            // Arity-0 primops (like getLine) execute immediately
            if op.arity() == 0 {
                return self.apply_primop(op, vec![]);
            }
            return Ok(Value::PrimOp(op));
        }

        // Check if it's a dictionary field selector ($sel_N)
        // These are generated by the HIR-to-Core lowering for type class
        // dictionary method extraction.
        if let Some(index) = Self::parse_selector_name(var.name.as_str()) {
            return Ok(Value::PrimOp(PrimOp::DictSelect(index)));
        }

        // Check if it looks like a data constructor (starts with uppercase)
        // User-defined constructors won't be in primitives or the env
        let name_str = var.name.as_str();
        if !name_str.is_empty() {
            let first_char = name_str.chars().next().unwrap();
            if first_char.is_uppercase() {
                // This is a constructor - create an unsaturated DataValue
                // We don't know the exact arity here, so use a large number
                // to allow arguments to be added. Pattern matching uses tags.
                let tycon = bhc_types::TyCon::new(var.name, bhc_types::Kind::Star);
                return Ok(Value::Data(DataValue {
                    con: crate::DataCon {
                        name: var.name,
                        ty_con: tycon,
                        tag: var.id.index() as u32,
                        arity: 100, // Large arity to accept arguments
                    },
                    args: vec![],
                }));
            }
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
                // Polymorphic equality: dispatch based on argument types.
                // This handles derived Eq for types with Int, Bool, Char,
                // Float, Double fields, and enum/ADT types.
                match (&args[0], &args[1]) {
                    (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a == b)),
                    (Value::Double(a), Value::Double(b)) =>
                    {
                        #[allow(clippy::float_cmp)]
                        Ok(Value::bool(a == b))
                    }
                    (Value::Float(a), Value::Float(b)) =>
                    {
                        #[allow(clippy::float_cmp)]
                        Ok(Value::bool(a == b))
                    }
                    (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a == b)),
                    (Value::String(a), Value::String(b)) => Ok(Value::bool(a == b)),
                    (Value::Data(a), Value::Data(b)) => {
                        // For nullary constructors (enums), compare tags
                        if a.args.is_empty() && b.args.is_empty() {
                            Ok(Value::bool(a.con.tag == b.con.tag))
                        } else if a.con.tag != b.con.tag {
                            // Different constructors are never equal
                            Ok(Value::bool(false))
                        } else {
                            // Same constructor: recursively compare all fields
                            if a.args.len() != b.args.len() {
                                Ok(Value::bool(false))
                            } else {
                                for (fa, fb) in a.args.iter().zip(b.args.iter()) {
                                    let fa = self.force(fa.clone())?;
                                    let fb = self.force(fb.clone())?;
                                    let eq = self.apply_primop(PrimOp::EqInt, vec![fa, fb])?;
                                    if eq.as_bool() == Some(false) {
                                        return Ok(Value::bool(false));
                                    }
                                }
                                Ok(Value::bool(true))
                            }
                        }
                    }
                    _ => {
                        // Fallback to int comparison for backward compatibility
                        let a = args[0].as_int().unwrap_or(0);
                        let b = args[1].as_int().unwrap_or(0);
                        Ok(Value::bool(a == b))
                    }
                }
            }

            PrimOp::LtInt => {
                // Polymorphic less-than for derived Ord support
                match (&args[0], &args[1]) {
                    (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a < b)),
                    (Value::Double(a), Value::Double(b)) => Ok(Value::bool(a < b)),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::bool(a < b)),
                    (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a < b)),
                    (Value::Data(a), Value::Data(b)) => Ok(Value::bool(a.con.tag < b.con.tag)),
                    _ => {
                        let a = args[0].as_int().unwrap_or(0);
                        let b = args[1].as_int().unwrap_or(0);
                        Ok(Value::bool(a < b))
                    }
                }
            }

            PrimOp::LeInt => match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a <= b)),
                (Value::Double(a), Value::Double(b)) => Ok(Value::bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::bool(a <= b)),
                (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a <= b)),
                (Value::Data(a), Value::Data(b)) => Ok(Value::bool(a.con.tag <= b.con.tag)),
                _ => {
                    let a = args[0].as_int().unwrap_or(0);
                    let b = args[1].as_int().unwrap_or(0);
                    Ok(Value::bool(a <= b))
                }
            },

            PrimOp::GtInt => match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a > b)),
                (Value::Double(a), Value::Double(b)) => Ok(Value::bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::bool(a > b)),
                (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a > b)),
                (Value::Data(a), Value::Data(b)) => Ok(Value::bool(a.con.tag > b.con.tag)),
                _ => {
                    let a = args[0].as_int().unwrap_or(0);
                    let b = args[1].as_int().unwrap_or(0);
                    Ok(Value::bool(a > b))
                }
            },

            PrimOp::GeInt => match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a >= b)),
                (Value::Double(a), Value::Double(b)) => Ok(Value::bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::bool(a >= b)),
                (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a >= b)),
                (Value::Data(a), Value::Data(b)) => Ok(Value::bool(a.con.tag >= b.con.tag)),
                _ => {
                    let a = args[0].as_int().unwrap_or(0);
                    let b = args[1].as_int().unwrap_or(0);
                    Ok(Value::bool(a >= b))
                }
            },

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
                args[0]
                    .uarray_to_list()
                    .ok_or_else(|| EvalError::TypeError {
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
                        Ok(Value::UArrayDouble(crate::uarray::UArray::from_vec(
                            mapped?,
                        )))
                    }
                    // Support mapping over lists - need to force thunks while traversing
                    _ => {
                        // Try to traverse list, forcing thunks along the way
                        let list_result = self.force_list(arr.clone())?;
                        let mapped: Result<Vec<Value>, _> = list_result
                            .iter()
                            .map(|x| {
                                let result = self.apply(f.clone(), x.clone())?;
                                self.force(result)
                            })
                            .collect();
                        Ok(Value::from_list(mapped?))
                    }
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
                        Ok(Value::UArrayDouble(crate::uarray::UArray::from_vec(
                            zipped?,
                        )))
                    }
                    // Support lists - force thunks while traversing
                    _ => {
                        let list1 = self.force_list(arr1.clone())?;
                        let list2 = self.force_list(arr2.clone())?;
                        let zipped: Result<Vec<Value>, _> = list1
                            .iter()
                            .zip(list2.iter())
                            .map(|(x, y)| {
                                let result =
                                    self.apply(self.apply(f.clone(), x.clone())?, y.clone())?;
                                self.force(result)
                            })
                            .collect();
                        Ok(Value::from_list(zipped?))
                    }
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
                            let result = self.apply(self.apply(f.clone(), acc)?, Value::Int(*x))?;
                            acc = self.force(result)?;
                        }
                        Ok(acc)
                    }
                    Value::UArrayDouble(uarr) => {
                        let mut acc = init;
                        for x in uarr.as_slice() {
                            let result =
                                self.apply(self.apply(f.clone(), acc)?, Value::Double(*x))?;
                            acc = self.force(result)?;
                        }
                        Ok(acc)
                    }
                    _ => {
                        // Force the list and fold over it
                        let list = self.force_list(arr.clone())?;
                        let mut acc = init;
                        for x in list {
                            let result = self.apply(self.apply(f.clone(), acc)?, x)?;
                            acc = self.force(result)?;
                        }
                        Ok(acc)
                    }
                }
            }

            PrimOp::UArraySum => {
                // sum arr - works on both UArrays and lists
                match &args[0] {
                    Value::UArrayInt(arr) => Ok(Value::Int(arr.sum())),
                    Value::UArrayDouble(arr) => Ok(Value::Double(arr.sum())),
                    _ => {
                        // Force the list and sum it
                        let list = self.force_list(args[0].clone())?;
                        // Try to sum as integers first
                        let ints: Option<i64> = list
                            .iter()
                            .map(Value::as_int)
                            .try_fold(0i64, |acc, x| x.map(|n| acc.wrapping_add(n)));
                        if let Some(sum) = ints {
                            return Ok(Value::Int(sum));
                        }
                        // Try as doubles
                        let doubles: Option<f64> = list
                            .iter()
                            .map(Value::as_double)
                            .try_fold(0.0f64, |acc, x| x.map(|n| acc + n));
                        if let Some(sum) = doubles {
                            return Ok(Value::Double(sum));
                        }
                        Err(EvalError::TypeError {
                            expected: "list of numbers".into(),
                            got: format!("{:?}", args[0]),
                        })
                    }
                }
            }

            PrimOp::UArrayLength => {
                match &args[0] {
                    Value::UArrayInt(arr) => Ok(Value::Int(arr.len() as i64)),
                    Value::UArrayDouble(arr) => Ok(Value::Int(arr.len() as i64)),
                    _ => {
                        // Force the list and get its length
                        let list = self.force_list(args[0].clone())?;
                        Ok(Value::Int(list.len() as i64))
                    }
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

            // List operations
            PrimOp::Concat => {
                // xs ++ ys - concatenate two lists
                let xs = self.force_list(args[0].clone())?;
                let ys = self.force_list(args[1].clone())?;
                let mut result = xs;
                result.extend(ys);
                Ok(Value::from_list(result))
            }

            PrimOp::ConcatMap => {
                // concatMap f xs - map f over xs and concatenate results
                let f = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut result = Vec::new();
                for x in xs {
                    let mapped = self.apply(f.clone(), x)?;
                    let forced = self.force(mapped)?;
                    let list = self.force_list(forced)?;
                    result.extend(list);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Append => {
                // append x xs - add x to the end of xs
                let x = args[0].clone();
                let xs = self.force_list(args[1].clone())?;
                let mut result = xs;
                result.push(x);
                Ok(Value::from_list(result))
            }

            // Monad operations for list
            PrimOp::ListBind => {
                // xs >>= f = concatMap f xs
                // Argument order: xs, f
                let xs = self.force_list(args[0].clone())?;
                let f = &args[1];
                let mut result = Vec::new();
                for x in xs {
                    let mapped = self.apply(f.clone(), x)?;
                    let forced = self.force(mapped)?;
                    let list = self.force_list(forced)?;
                    result.extend(list);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::ListThen => {
                // xs >> ys = xs >>= \_ -> ys
                // Argument order: xs, ys
                let xs = self.force_list(args[0].clone())?;
                let ys = args[1].clone();
                let ys_list = self.force_list(ys)?;
                let mut result = Vec::new();
                for _ in xs {
                    result.extend(ys_list.clone());
                }
                Ok(Value::from_list(result))
            }

            PrimOp::ListReturn => {
                // return x = [x]
                let x = args[0].clone();
                Ok(Value::from_list(vec![x]))
            }

            // Additional list operations
            PrimOp::Foldr => {
                // foldr f z xs
                // foldr f z [] = z
                // foldr f z (x:xs) = f x (foldr f z xs)
                let f = &args[0];
                let z = args[1].clone();
                let xs = self.force_list(args[2].clone())?;

                // Process from right to left
                let mut acc = z;
                for x in xs.into_iter().rev() {
                    acc = self.apply(self.apply(f.clone(), x)?, acc)?;
                }
                Ok(acc)
            }

            PrimOp::Foldl => {
                // foldl f z xs
                // foldl f z [] = z
                // foldl f z (x:xs) = foldl f (f z x) xs
                let f = &args[0];
                let z = args[1].clone();
                let xs = self.force_list(args[2].clone())?;

                let mut acc = z;
                for x in xs {
                    acc = self.apply(self.apply(f.clone(), acc)?, x)?;
                }
                Ok(acc)
            }

            PrimOp::FoldlStrict => {
                // foldl' f z xs - strict left fold
                let f = &args[0];
                let z = args[1].clone();
                let xs = self.force_list(args[2].clone())?;

                let mut acc = self.force(z)?;
                for x in xs {
                    acc = self.apply(self.apply(f.clone(), acc)?, x)?;
                    acc = self.force(acc)?;
                }
                Ok(acc)
            }

            PrimOp::Filter => {
                // filter p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;

                let mut result = Vec::new();
                for x in xs {
                    let pred_result = self.apply(p.clone(), x.clone())?;
                    let pred_forced = self.force(pred_result)?;
                    if pred_forced.as_bool().unwrap_or(false) {
                        result.push(x);
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Zip => {
                // zip xs ys = zipWith (,) xs ys
                let xs = self.force_list(args[0].clone())?;
                let ys = self.force_list(args[1].clone())?;

                let pairs: Vec<Value> = xs
                    .into_iter()
                    .zip(ys.into_iter())
                    .map(|(x, y)| {
                        use bhc_types::{Kind, TyCon};
                        Value::Data(DataValue {
                            con: crate::DataCon {
                                name: Symbol::intern("(,)"),
                                ty_con: TyCon::new(Symbol::intern("(,)"), Kind::Star),
                                tag: 0,
                                arity: 2,
                            },
                            args: vec![x, y],
                        })
                    })
                    .collect();
                Ok(Value::from_list(pairs))
            }

            PrimOp::ZipWith => {
                // zipWith f xs ys
                let f = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let ys = self.force_list(args[2].clone())?;

                let mut result = Vec::new();
                for (x, y) in xs.into_iter().zip(ys.into_iter()) {
                    let r = self.apply(self.apply(f.clone(), x)?, y)?;
                    result.push(self.force(r)?);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Take => {
                // take n xs
                let n = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let xs = self.force_list(args[1].clone())?;

                let n = n.max(0) as usize;
                let result: Vec<Value> = xs.into_iter().take(n).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Drop => {
                // drop n xs
                let n = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let xs = self.force_list(args[1].clone())?;

                let n = n.max(0) as usize;
                let result: Vec<Value> = xs.into_iter().skip(n).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Head => {
                // head xs - partial, errors on empty list
                let xs = self.force_list(args[0].clone())?;
                xs.into_iter()
                    .next()
                    .ok_or(EvalError::UserError("head: empty list".to_string()))
            }

            PrimOp::Tail => {
                // tail xs - partial, errors on empty list
                let xs = self.force_list(args[0].clone())?;
                if xs.is_empty() {
                    return Err(EvalError::UserError("tail: empty list".to_string()));
                }
                let result: Vec<Value> = xs.into_iter().skip(1).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Last => {
                // last xs - partial, errors on empty list
                let xs = self.force_list(args[0].clone())?;
                xs.into_iter()
                    .last()
                    .ok_or(EvalError::UserError("last: empty list".to_string()))
            }

            PrimOp::Init => {
                // init xs - partial, errors on empty list
                let xs = self.force_list(args[0].clone())?;
                if xs.is_empty() {
                    return Err(EvalError::UserError("init: empty list".to_string()));
                }
                let len = xs.len();
                let result: Vec<Value> = xs.into_iter().take(len - 1).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Reverse => {
                // reverse xs
                let xs = self.force_list(args[0].clone())?;
                let result: Vec<Value> = xs.into_iter().rev().collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Null => {
                // null xs - returns True if list is empty
                let xs = self.force_list(args[0].clone())?;
                Ok(Value::bool(xs.is_empty()))
            }

            PrimOp::Index => {
                // xs !! n - index into list
                let xs = self.force_list(args[0].clone())?;
                let n = args[1].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[1]),
                })?;

                if n < 0 {
                    return Err(EvalError::UserError("(!!): negative index".to_string()));
                }
                let n = n as usize;
                xs.into_iter()
                    .nth(n)
                    .ok_or_else(|| EvalError::UserError(format!("(!!): index {} too large", n)))
            }

            PrimOp::Replicate => {
                // replicate n x
                let n = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let x = args[1].clone();

                let n = n.max(0) as usize;
                let result: Vec<Value> = std::iter::repeat(x).take(n).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::EnumFromTo => {
                // enumFromTo start end = [start..end]
                let start = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let end = args[1].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[1]),
                })?;

                let result: Vec<Value> = (start..=end).map(Value::Int).collect();
                Ok(Value::from_list(result))
            }

            // Additional list/prelude operations
            PrimOp::Even => {
                let n = args[0].as_int().unwrap_or(0);
                Ok(Value::bool(n % 2 == 0))
            }

            PrimOp::Odd => {
                let n = args[0].as_int().unwrap_or(0);
                Ok(Value::bool(n % 2 != 0))
            }

            PrimOp::Elem => {
                // elem x xs
                let x = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let found = xs.iter().any(|y| self.values_equal(x, y));
                Ok(Value::bool(found))
            }

            PrimOp::NotElem => {
                // notElem x xs
                let x = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let found = xs.iter().any(|y| self.values_equal(x, y));
                Ok(Value::bool(!found))
            }

            PrimOp::TakeWhile => {
                // takeWhile p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut result = Vec::new();
                for x in xs {
                    let pred_result = self.apply(p.clone(), x.clone())?;
                    let pred_forced = self.force(pred_result)?;
                    if pred_forced.as_bool().unwrap_or(false) {
                        result.push(x);
                    } else {
                        break;
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::DropWhile => {
                // dropWhile p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut dropping = true;
                let mut result = Vec::new();
                for x in xs {
                    if dropping {
                        let pred_result = self.apply(p.clone(), x.clone())?;
                        let pred_forced = self.force(pred_result)?;
                        if pred_forced.as_bool().unwrap_or(false) {
                            continue;
                        }
                        dropping = false;
                    }
                    result.push(x);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Span => {
                // span p xs = (takeWhile p xs, dropWhile p xs)
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut prefix = Vec::new();
                let mut rest_start = 0;
                for (i, x) in xs.iter().enumerate() {
                    let pred_result = self.apply(p.clone(), x.clone())?;
                    let pred_forced = self.force(pred_result)?;
                    if pred_forced.as_bool().unwrap_or(false) {
                        prefix.push(x.clone());
                    } else {
                        rest_start = i;
                        break;
                    }
                    rest_start = i + 1;
                }
                let rest: Vec<Value> = xs.into_iter().skip(rest_start).collect();
                Ok(self.make_pair(Value::from_list(prefix), Value::from_list(rest)))
            }

            PrimOp::Break => {
                // break p xs = span (not . p) xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut prefix = Vec::new();
                let mut rest_start = 0;
                for (i, x) in xs.iter().enumerate() {
                    let pred_result = self.apply(p.clone(), x.clone())?;
                    let pred_forced = self.force(pred_result)?;
                    if !pred_forced.as_bool().unwrap_or(false) {
                        prefix.push(x.clone());
                    } else {
                        rest_start = i;
                        break;
                    }
                    rest_start = i + 1;
                }
                let rest: Vec<Value> = xs.into_iter().skip(rest_start).collect();
                Ok(self.make_pair(Value::from_list(prefix), Value::from_list(rest)))
            }

            PrimOp::SplitAt => {
                // splitAt n xs
                let n = args[0].as_int().unwrap_or(0).max(0) as usize;
                let xs = self.force_list(args[1].clone())?;
                let (prefix, rest) = if n >= xs.len() {
                    (xs, Vec::new())
                } else {
                    let mut v = xs;
                    let rest = v.split_off(n);
                    (v, rest)
                };
                Ok(self.make_pair(Value::from_list(prefix), Value::from_list(rest)))
            }

            PrimOp::Iterate => {
                // iterate f x - produces [x, f x, f (f x), ...]
                // Finite approximation: produce up to 1000 elements
                let f = &args[0];
                let mut current = args[1].clone();
                let limit = 1000;
                let mut result = Vec::with_capacity(limit);
                for _ in 0..limit {
                    result.push(current.clone());
                    current = self.apply(f.clone(), current)?;
                    current = self.force(current)?;
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Repeat => {
                // repeat x - infinite list of x (truncated to 1000)
                let x = args[0].clone();
                let result: Vec<Value> = std::iter::repeat(x).take(1000).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Cycle => {
                // cycle xs - infinite repetition (truncated to 1000)
                let xs = self.force_list(args[0].clone())?;
                if xs.is_empty() {
                    return Err(EvalError::UserError("cycle: empty list".to_string()));
                }
                let limit = 1000;
                let mut result = Vec::with_capacity(limit);
                let mut i = 0;
                while result.len() < limit {
                    result.push(xs[i % xs.len()].clone());
                    i += 1;
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Lookup => {
                // lookup k xs where xs is [(k, v)]
                let key = &args[0];
                let xs = self.force_list(args[1].clone())?;
                for pair in xs {
                    let pair = self.force(pair)?;
                    if let Value::Data(ref d) = pair {
                        if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                            let k = self.force(d.args[0].clone())?;
                            if self.values_equal(key, &k) {
                                let v = d.args[1].clone();
                                return Ok(self.make_just(v));
                            }
                        }
                    }
                }
                Ok(self.make_nothing())
            }

            PrimOp::Unzip => {
                // unzip :: [(a, b)] -> ([a], [b])
                let xs = self.force_list(args[0].clone())?;
                let mut as_list = Vec::new();
                let mut bs_list = Vec::new();
                for pair in xs {
                    let pair = self.force(pair)?;
                    if let Value::Data(ref d) = pair {
                        if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                            as_list.push(d.args[0].clone());
                            bs_list.push(d.args[1].clone());
                        }
                    }
                }
                Ok(self.make_pair(Value::from_list(as_list), Value::from_list(bs_list)))
            }

            PrimOp::Product => {
                // product xs
                match &args[0] {
                    Value::UArrayInt(arr) => {
                        let p: i64 = arr.as_slice().iter().product();
                        Ok(Value::Int(p))
                    }
                    _ => {
                        let list = self.force_list(args[0].clone())?;
                        let mut acc: i64 = 1;
                        for x in list {
                            if let Some(n) = x.as_int() {
                                acc = acc.wrapping_mul(n);
                            }
                        }
                        Ok(Value::Int(acc))
                    }
                }
            }

            PrimOp::Flip => {
                // flip f x y = f y x
                let f = &args[0];
                let x = args[1].clone();
                let y = args[2].clone();
                let partial = self.apply(f.clone(), y)?;
                self.apply(partial, x)
            }

            PrimOp::Min => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::Int(a.min(b)))
            }

            PrimOp::Max => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::Int(a.max(b)))
            }

            PrimOp::FromIntegral => {
                // Identity for now (Int -> Int)
                Ok(args[0].clone())
            }

            PrimOp::MaybeElim => {
                // maybe def f m
                let def = args[0].clone();
                let f = &args[1];
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Data(d) if d.con.name.as_str() == "Nothing" => Ok(def),
                    Value::Data(d) if d.con.name.as_str() == "Just" && d.args.len() == 1 => {
                        self.apply(f.clone(), d.args[0].clone())
                    }
                    _ => Ok(def),
                }
            }

            PrimOp::FromMaybe => {
                // fromMaybe def m
                let def = args[0].clone();
                let m = self.force(args[1].clone())?;
                match &m {
                    Value::Data(d) if d.con.name.as_str() == "Nothing" => Ok(def),
                    Value::Data(d) if d.con.name.as_str() == "Just" && d.args.len() == 1 => {
                        Ok(d.args[0].clone())
                    }
                    _ => Ok(def),
                }
            }

            PrimOp::EitherElim => {
                // either f g e
                let f = &args[0];
                let g = &args[1];
                let e = self.force(args[2].clone())?;
                match &e {
                    Value::Data(d) if d.con.name.as_str() == "Left" && d.args.len() == 1 => {
                        self.apply(f.clone(), d.args[0].clone())
                    }
                    Value::Data(d) if d.con.name.as_str() == "Right" && d.args.len() == 1 => {
                        self.apply(g.clone(), d.args[0].clone())
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Either".into(),
                        got: format!("{e:?}"),
                    }),
                }
            }

            PrimOp::IsJust => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Data(d) if d.con.name.as_str() == "Just" => Ok(Value::bool(true)),
                    Value::Data(d) if d.con.name.as_str() == "Nothing" => Ok(Value::bool(false)),
                    _ => Ok(Value::bool(false)),
                }
            }

            PrimOp::IsNothing => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Data(d) if d.con.name.as_str() == "Nothing" => Ok(Value::bool(true)),
                    Value::Data(d) if d.con.name.as_str() == "Just" => Ok(Value::bool(false)),
                    _ => Ok(Value::bool(true)),
                }
            }

            PrimOp::Abs => {
                let n = args[0].as_int().unwrap_or(0);
                Ok(Value::Int(n.abs()))
            }

            PrimOp::Signum => {
                let n = args[0].as_int().unwrap_or(0);
                Ok(Value::Int(n.signum()))
            }

            PrimOp::Curry => {
                // curry f x y = f (x, y)
                let f = &args[0];
                let x = args[1].clone();
                let y = args[2].clone();
                let pair = self.make_pair(x, y);
                self.apply(f.clone(), pair)
            }

            PrimOp::Uncurry => {
                // uncurry f (x, y) = f x y
                let f = &args[0];
                let pair = self.force(args[1].clone())?;
                if let Value::Data(ref d) = pair {
                    if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                        let partial = self.apply(f.clone(), d.args[0].clone())?;
                        return self.apply(partial, d.args[1].clone());
                    }
                }
                Err(EvalError::TypeError {
                    expected: "pair (a, b)".into(),
                    got: format!("{pair:?}"),
                })
            }

            PrimOp::Swap => {
                // swap (a, b) = (b, a)
                let pair = self.force(args[0].clone())?;
                if let Value::Data(ref d) = pair {
                    if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                        return Ok(self.make_pair(d.args[1].clone(), d.args[0].clone()));
                    }
                }
                Err(EvalError::TypeError {
                    expected: "pair (a, b)".into(),
                    got: format!("{pair:?}"),
                })
            }

            PrimOp::Any => {
                // any p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                for x in xs {
                    let pred_result = self.apply(p.clone(), x)?;
                    let pred_forced = self.force(pred_result)?;
                    if pred_forced.as_bool().unwrap_or(false) {
                        return Ok(Value::bool(true));
                    }
                }
                Ok(Value::bool(false))
            }

            PrimOp::All => {
                // all p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                for x in xs {
                    let pred_result = self.apply(p.clone(), x)?;
                    let pred_forced = self.force(pred_result)?;
                    if !pred_forced.as_bool().unwrap_or(true) {
                        return Ok(Value::bool(false));
                    }
                }
                Ok(Value::bool(true))
            }

            PrimOp::And => {
                // and xs - all True
                let xs = self.force_list(args[0].clone())?;
                for x in xs {
                    if !x.as_bool().unwrap_or(true) {
                        return Ok(Value::bool(false));
                    }
                }
                Ok(Value::bool(true))
            }

            PrimOp::Or => {
                // or xs - any True
                let xs = self.force_list(args[0].clone())?;
                for x in xs {
                    if x.as_bool().unwrap_or(false) {
                        return Ok(Value::bool(true));
                    }
                }
                Ok(Value::bool(false))
            }

            PrimOp::Lines => {
                // lines :: String -> [String]
                let s = self.value_to_string(&args[0])?;
                let result: Vec<Value> = s.lines().map(|l| Value::String(l.into())).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Unlines => {
                // unlines :: [String] -> String
                let xs = self.force_list(args[0].clone())?;
                let mut result = String::new();
                for x in xs {
                    let s = self.value_to_string(&x)?;
                    result.push_str(&s);
                    result.push('\n');
                }
                Ok(Value::String(result.into()))
            }

            PrimOp::Words => {
                // words :: String -> [String]
                let s = self.value_to_string(&args[0])?;
                let result: Vec<Value> = s
                    .split_whitespace()
                    .map(|w| Value::String(w.into()))
                    .collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Unwords => {
                // unwords :: [String] -> String
                let xs = self.force_list(args[0].clone())?;
                let mut parts = Vec::new();
                for x in xs {
                    let s = self.value_to_string(&x)?;
                    parts.push(s);
                }
                Ok(Value::String(parts.join(" ").into()))
            }

            PrimOp::Show => {
                // show :: a -> String
                let displayed = self.display_value(&args[0])?;
                Ok(Value::String(displayed.into()))
            }

            PrimOp::Id => {
                // id :: a -> a
                Ok(args[0].clone())
            }

            PrimOp::Const => {
                // const :: a -> b -> a
                Ok(args[0].clone())
            }

            // IO operations
            PrimOp::PutStrLn => {
                // putStrLn :: String -> IO ()
                let s = self.value_to_string(&args[0])?;
                println!("{s}");
                {
                    let mut buf = self.io_output.borrow_mut();
                    buf.push_str(&s);
                    buf.push('\n');
                }
                Ok(Value::unit())
            }

            PrimOp::PutStr => {
                // putStr :: String -> IO ()
                use std::io::Write;
                let s = self.value_to_string(&args[0])?;
                print!("{s}");
                std::io::stdout().flush().ok();
                self.io_output.borrow_mut().push_str(&s);
                Ok(Value::unit())
            }

            PrimOp::Print => {
                // print :: Show a => a -> IO ()
                let displayed = self.display_value(&args[0])?;
                println!("{displayed}");
                {
                    let mut buf = self.io_output.borrow_mut();
                    buf.push_str(&displayed);
                    buf.push('\n');
                }
                Ok(Value::unit())
            }

            PrimOp::GetLine => {
                // getLine :: IO String
                use std::io::BufRead;
                let stdin = std::io::stdin();
                let mut line = String::new();
                stdin
                    .lock()
                    .read_line(&mut line)
                    .map_err(|e| EvalError::UserError(format!("getLine failed: {e}")))?;
                // Remove trailing newline
                if line.ends_with('\n') {
                    line.pop();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                }
                Ok(Value::String(line.into()))
            }

            PrimOp::IoBind => {
                // (>>=) :: IO a -> (a -> IO b) -> IO b
                // In our interpreter, IO actions execute immediately,
                // so we just run the first action and pass result to continuation
                let io_a = args[0].clone();
                let f = &args[1];
                // "Execute" the IO action (it's already a value in our interpreter)
                // and apply the continuation
                self.apply(f.clone(), io_a)
            }

            PrimOp::IoThen => {
                // (>>) :: IO a -> IO b -> IO b
                // Execute first action (already done), return second
                Ok(args[1].clone())
            }

            PrimOp::IoReturn => {
                // return :: a -> IO a
                // In our interpreter, just return the value as-is
                Ok(args[0].clone())
            }

            PrimOp::DictSelect(index) => {
                // Extract field at `index` from a dictionary (tuple-like DataValue).
                // Dictionaries are represented as nested tuple applications:
                // e.g., (,,) method0 method1 method2
                let dict = self.force(args[0].clone())?;
                match dict {
                    Value::Data(ref d) => {
                        if index < d.args.len() {
                            self.force(d.args[index].clone())
                        } else {
                            Err(EvalError::UserError(format!(
                                "dictionary field selector $sel_{index} out of range \
                                 (dictionary has {} fields, constructor {})",
                                d.args.len(),
                                d.con.name
                            )))
                        }
                    }
                    // If the dictionary is a single-field value (not wrapped in a tuple),
                    // and we're selecting field 0, just return it directly.
                    _ if index == 0 => Ok(dict),
                    _ => Err(EvalError::TypeError {
                        expected: "dictionary (tuple)".into(),
                        got: format!("{dict:?}"),
                    }),
                }
            }

            PrimOp::MonadBind => {
                // Polymorphic (>>=): dispatch based on first argument type
                // If it's a list, use list bind (concatMap)
                // Otherwise, use IO bind (apply continuation to result)
                let first_arg = self.force(args[0].clone())?;

                if self.is_list_value(&first_arg) {
                    // List monad: xs >>= f = concatMap f xs
                    self.apply_primop(PrimOp::ListBind, args)
                } else {
                    // IO monad (or other): just apply f to the value
                    self.apply_primop(PrimOp::IoBind, args)
                }
            }

            PrimOp::MonadThen => {
                // Polymorphic (>>): dispatch based on first argument type
                let first_arg = self.force(args[0].clone())?;

                if self.is_list_value(&first_arg) {
                    // List monad: xs >> ys = xs >>= \_ -> ys
                    self.apply_primop(PrimOp::ListThen, args)
                } else {
                    // IO monad: just return second arg (first already executed)
                    self.apply_primop(PrimOp::IoThen, args)
                }
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
                // For recursive bindings, we need closures to be able to find
                // their recursive references. Since closures capture their env
                // at creation time and our envs are immutable, we use a
                // "recursive env stack" that is checked during variable lookup.
                //
                // Strategy:
                // 1. Evaluate all RHS expressions (typically lambdas) with current env
                //    This creates Closures that capture env (without recursive bindings)
                // 2. Build final_env with all the evaluated values
                // 3. Push final_env onto rec_env_stack before evaluating body
                // 4. Variable lookup checks rec_env_stack, so recursive calls work
                //
                // This approach works because:
                // - When a Closure is applied, its body is evaluated
                // - Variable lookup in the body checks rec_env_stack
                // - The rec_env_stack contains the binding for the recursive function

                let mut final_env = env.clone();

                // Evaluate all bindings and add to final_env
                // For lambdas, this just creates Closures (very cheap)
                for (var, rhs) in bindings {
                    let value = self.eval(rhs, env)?;
                    final_env = final_env.extend(var.id, value);
                }

                // Push the recursive environment onto the stack
                // This makes recursive bindings visible during body evaluation
                self.rec_env_stack.borrow_mut().push(final_env.clone());

                // Evaluate the body with final_env
                // The rec_env_stack ensures recursive calls can find their targets
                let result = self.eval(body, &final_env);

                // Pop the recursive environment
                self.rec_env_stack.borrow_mut().pop();

                result
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
                // If there's a binder, bind it to the scrutinee value
                if binders.is_empty() {
                    Ok(Some(Vec::new()))
                } else {
                    // For variable patterns like `case e of x -> ...`,
                    // bind x to the scrutinee value
                    let bindings: Vec<_> =
                        binders.iter().map(|var| (var.id, value.clone())).collect();
                    Ok(Some(bindings))
                }
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
                // Match data constructor by name (tags may differ due to ID allocation)
                if let Value::Data(data) = value {
                    if data.con.name == con.name {
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

    /// Checks if a value is a list (either empty list or cons cell).
    fn is_list_value(&self, value: &Value) -> bool {
        match value {
            Value::Data(d) => {
                let name = d.con.name.as_str();
                name == "[]" || name == ":"
            }
            Value::String(_) => true, // String is [Char]
            _ => false,
        }
    }

    /// Structural equality for values (used by elem, lookup, etc.)
    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Double(x), Value::Double(y)) => x == y,
            (Value::Char(x), Value::Char(y)) => x == y,
            (Value::String(x), Value::String(y)) => x == y,
            (Value::Data(x), Value::Data(y)) => {
                x.con.name == y.con.name
                    && x.args.len() == y.args.len()
                    && x.args
                        .iter()
                        .zip(y.args.iter())
                        .all(|(a, b)| self.values_equal(a, b))
            }
            _ => false,
        }
    }

    /// Create a pair value (a, b).
    fn make_pair(&self, a: Value, b: Value) -> Value {
        use bhc_types::{Kind, TyCon};
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern("(,)"),
                ty_con: TyCon::new(Symbol::intern("(,)"), Kind::Star),
                tag: 0,
                arity: 2,
            },
            args: vec![a, b],
        })
    }

    /// Create a Just value.
    fn make_just(&self, a: Value) -> Value {
        use bhc_types::{Kind, TyCon};
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern("Just"),
                ty_con: TyCon::new(Symbol::intern("Maybe"), Kind::star_to_star()),
                tag: 1,
                arity: 1,
            },
            args: vec![a],
        })
    }

    /// Create a Nothing value.
    fn make_nothing(&self) -> Value {
        use bhc_types::{Kind, TyCon};
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern("Nothing"),
                ty_con: TyCon::new(Symbol::intern("Maybe"), Kind::star_to_star()),
                tag: 0,
                arity: 0,
            },
            args: Vec::new(),
        })
    }

    /// Converts a Value to a String, for use in IO operations.
    /// Handles both String values and [Char] lists.
    fn value_to_string(&self, value: &Value) -> Result<String, EvalError> {
        let forced = self.force(value.clone())?;
        match &forced {
            Value::String(s) => Ok(s.to_string()),
            Value::Data(d) if d.con.name.as_str() == "[]" || d.con.name.as_str() == ":" => {
                // It's a list - try to interpret as [Char]
                let list = self.force_list(forced)?;
                let chars: Result<String, _> = list
                    .into_iter()
                    .map(|v| match v {
                        Value::Char(c) => Ok(c),
                        other => Err(EvalError::TypeError {
                            expected: "Char".into(),
                            got: format!("{other:?}"),
                        }),
                    })
                    .collect();
                chars
            }
            other => Err(EvalError::TypeError {
                expected: "String".into(),
                got: format!("{other:?}"),
            }),
        }
    }

    fn force_thunk(&self, thunk: &Thunk) -> Result<Value, EvalError> {
        self.eval(&thunk.expr, &thunk.env)
    }

    /// Forces a list structure, converting it to a Vec<Value>.
    /// This traverses the list spine, forcing thunks along the way.
    fn force_list(&self, value: Value) -> Result<Vec<Value>, EvalError> {
        let mut result = Vec::new();
        let mut current = self.force(value)?;

        loop {
            match &current {
                // String is [Char] - convert to list of characters
                Value::String(s) => {
                    for c in s.chars() {
                        result.push(Value::Char(c));
                    }
                    return Ok(result);
                }
                Value::Data(d) if d.con.name.as_str() == "[]" => {
                    return Ok(result);
                }
                Value::Data(d) if d.con.name.as_str() == ":" && d.args.len() == 2 => {
                    // Force the head element
                    let head = self.force(d.args[0].clone())?;
                    result.push(head);
                    // Force the tail and continue traversing
                    current = self.force(d.args[1].clone())?;
                }
                other => {
                    return Err(EvalError::TypeError {
                        expected: "List".into(),
                        got: format!("{other:?}"),
                    });
                }
            }
        }
    }

    /// Display a value, deeply forcing all thunks.
    ///
    /// This produces a readable string representation of a value,
    /// recursively forcing any thunks encountered.
    pub fn display_value(&self, value: &Value) -> Result<String, EvalError> {
        self.display_value_impl(value, 0)
    }

    fn display_value_impl(&self, value: &Value, depth: usize) -> Result<String, EvalError> {
        // Prevent infinite recursion
        if depth > 1000 {
            return Ok("<deep>".to_string());
        }

        // Force thunks first
        let forced = match value {
            Value::Thunk(thunk) => self.force_thunk(thunk)?,
            v => v.clone(),
        };

        match &forced {
            Value::Int(n) => Ok(n.to_string()),
            Value::Integer(n) => Ok(n.to_string()),
            Value::Float(n) => Ok(format!("{n}")),
            Value::Double(n) => Ok(format!("{n}")),
            Value::Char(c) => Ok(format!("{c:?}")),
            Value::String(s) => Ok(format!("{s:?}")),
            Value::Closure(c) => Ok(format!("<function {}>", c.var.name)),
            Value::PrimOp(op) => Ok(format!("<primop {op:?}>")),
            Value::PartialPrimOp(op, args) => {
                Ok(format!("<partial {op:?} applied to {} args>", args.len()))
            }
            Value::UArrayInt(arr) => {
                let elements: Vec<String> = arr.as_slice().iter().map(|n| n.to_string()).collect();
                Ok(format!("[{}]", elements.join(", ")))
            }
            Value::UArrayDouble(arr) => {
                let elements: Vec<String> = arr.as_slice().iter().map(|n| format!("{n}")).collect();
                Ok(format!("[{}]", elements.join(", ")))
            }
            Value::Data(d) => {
                let name = d.con.name.as_str();
                // Special case for lists
                if name == "[]" {
                    return Ok("[]".to_string());
                }
                if name == ":" {
                    // It's a list - collect all elements
                    let list = self.force_list(forced.clone())?;
                    let elements: Result<Vec<String>, _> = list
                        .iter()
                        .map(|v| self.display_value_impl(v, depth + 1))
                        .collect();
                    return Ok(format!("[{}]", elements?.join(", ")));
                }
                // Special case for tuples
                if name.starts_with('(') && name.ends_with(')') && name.contains(',') {
                    let elements: Result<Vec<String>, _> = d
                        .args
                        .iter()
                        .map(|v| self.display_value_impl(v, depth + 1))
                        .collect();
                    return Ok(format!("({})", elements?.join(", ")));
                }
                // General data constructor
                if d.args.is_empty() {
                    Ok(name.to_string())
                } else {
                    let args: Result<Vec<String>, _> = d
                        .args
                        .iter()
                        .map(|v| self.display_value_impl(v, depth + 1))
                        .collect();
                    Ok(format!("{} {}", name, args?.join(" ")))
                }
            }
            Value::Thunk(_) => {
                // Should have been forced above, but just in case
                Ok("<thunk>".to_string())
            }
        }
    }

    /// Parses a dictionary selector name like "$sel_0", "$sel_1", etc.
    /// Returns the field index if the name matches the pattern.
    fn parse_selector_name(name: &str) -> Option<usize> {
        name.strip_prefix("$sel_")
            .and_then(|n| n.parse::<usize>().ok())
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
        let lam = Expr::Lam(
            x.clone(),
            Box::new(Expr::Var(x, Span::default())),
            Span::default(),
        );
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
        let sum_app = Expr::App(
            Box::new(sum_var),
            Box::new(Expr::Var(list_var.clone(), Span::default())),
            Span::default(),
        );
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
            Box::new(Expr::App(
                Box::new(map_var),
                Box::new(add_one),
                Span::default(),
            )),
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
                Box::new(Expr::App(
                    Box::new(cons_var),
                    Box::new(val_expr),
                    Span::default(),
                )),
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
            Box::new(Expr::App(
                Box::new(add),
                Box::new(make_int(1)),
                Span::default(),
            )),
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
            Box::new(Expr::App(
                Box::new(add),
                Box::new(make_int(1)),
                Span::default(),
            )),
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

    // =========================================================================
    // Additional Primop Tests
    // =========================================================================

    #[test]
    fn test_eval_primop_sub() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (-) 10 3
        let sub = Expr::Var(make_var("-", 100), Span::default());
        let app1 = Expr::App(Box::new(sub), Box::new(make_int(10)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(3)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(7)));
    }

    #[test]
    fn test_eval_primop_div() {
        let eval = Evaluator::new(EvalMode::Strict);

        // div 10 3
        let div = Expr::Var(make_var("div", 100), Span::default());
        let app1 = Expr::App(Box::new(div), Box::new(make_int(10)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(3)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(3)));
    }

    #[test]
    fn test_eval_primop_mod() {
        let eval = Evaluator::new(EvalMode::Strict);

        // mod 10 3
        let modop = Expr::Var(make_var("mod", 100), Span::default());
        let app1 = Expr::App(Box::new(modop), Box::new(make_int(10)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(3)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(1)));
    }

    #[test]
    fn test_eval_primop_negate() {
        let eval = Evaluator::new(EvalMode::Strict);

        // negate 42
        let neg = Expr::Var(make_var("negate", 100), Span::default());
        let app = Expr::App(Box::new(neg), Box::new(make_int(42)), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(-42)));
    }

    #[test]
    fn test_eval_div_by_zero() {
        let eval = Evaluator::new(EvalMode::Strict);

        // div 10 0
        let div = Expr::Var(make_var("div", 100), Span::default());
        let app1 = Expr::App(Box::new(div), Box::new(make_int(10)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(0)), Span::default());

        let result = eval.eval(&app2, &Env::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_comparison_eq() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (==) 42 42
        let eq = Expr::Var(make_var("==", 100), Span::default());
        let app1 = Expr::App(Box::new(eq), Box::new(make_int(42)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(42)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_comparison_neq() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (==) 42 43
        let eq = Expr::Var(make_var("==", 100), Span::default());
        let app1 = Expr::App(Box::new(eq), Box::new(make_int(42)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(43)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(false));
    }

    #[test]
    fn test_eval_comparison_le() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (<=) 3 5
        let le = Expr::Var(make_var("<=", 100), Span::default());
        let app1 = Expr::App(Box::new(le), Box::new(make_int(3)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(5)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));

        // (<=) 5 5 (equal case)
        let le2 = Expr::Var(make_var("<=", 101), Span::default());
        let app3 = Expr::App(Box::new(le2), Box::new(make_int(5)), Span::default());
        let app4 = Expr::App(Box::new(app3), Box::new(make_int(5)), Span::default());

        let result2 = eval.eval(&app4, &Env::new()).unwrap();
        assert_eq!(result2.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_comparison_gt() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (>) 5 3
        let gt = Expr::Var(make_var(">", 100), Span::default());
        let app1 = Expr::App(Box::new(gt), Box::new(make_int(5)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(3)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_comparison_ge() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (>=) 5 5
        let ge = Expr::Var(make_var(">=", 100), Span::default());
        let app1 = Expr::App(Box::new(ge), Box::new(make_int(5)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(5)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_boolean_and() {
        let eval = Evaluator::new(EvalMode::Strict);

        // Create boolean values via comparison: (1 < 2) && (3 < 4) == true
        let lt1 = Expr::Var(make_var("<", 100), Span::default());
        let lt2 = Expr::Var(make_var("<", 101), Span::default());
        let and_op = Expr::Var(make_var("&&", 102), Span::default());

        // (1 < 2) = true
        let cmp1 = Expr::App(
            Box::new(Expr::App(
                Box::new(lt1),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // (3 < 4) = true
        let cmp2 = Expr::App(
            Box::new(Expr::App(
                Box::new(lt2),
                Box::new(make_int(3)),
                Span::default(),
            )),
            Box::new(make_int(4)),
            Span::default(),
        );

        // true && true
        let app1 = Expr::App(Box::new(and_op), Box::new(cmp1), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(cmp2), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_boolean_or() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (1 > 2) || (3 < 4) = false || true = true
        let gt = Expr::Var(make_var(">", 100), Span::default());
        let lt = Expr::Var(make_var("<", 101), Span::default());
        let or_op = Expr::Var(make_var("||", 102), Span::default());

        // (1 > 2) = false
        let cmp1 = Expr::App(
            Box::new(Expr::App(
                Box::new(gt),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // (3 < 4) = true
        let cmp2 = Expr::App(
            Box::new(Expr::App(
                Box::new(lt),
                Box::new(make_int(3)),
                Span::default(),
            )),
            Box::new(make_int(4)),
            Span::default(),
        );

        // false || true
        let app1 = Expr::App(Box::new(or_op), Box::new(cmp1), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(cmp2), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_boolean_not() {
        let eval = Evaluator::new(EvalMode::Strict);

        // not (1 > 2) = not false = true
        let gt = Expr::Var(make_var(">", 100), Span::default());
        let not_op = Expr::Var(make_var("not", 101), Span::default());

        // (1 > 2) = false
        let cmp = Expr::App(
            Box::new(Expr::App(
                Box::new(gt),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // not false
        let app = Expr::App(Box::new(not_op), Box::new(cmp), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_seq() {
        let eval = Evaluator::new(EvalMode::Lazy);

        // seq 1 42 = 42 (forces first arg, returns second)
        let seq_op = Expr::Var(make_var("seq", 100), Span::default());
        let app1 = Expr::App(Box::new(seq_op), Box::new(make_int(1)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(42)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_nested_arithmetic() {
        let eval = Evaluator::new(EvalMode::Strict);

        // ((1 + 2) * 3) - 4 = 9 - 4 = 5
        let add = Expr::Var(make_var("+", 100), Span::default());
        let mul = Expr::Var(make_var("*", 101), Span::default());
        let sub = Expr::Var(make_var("-", 102), Span::default());

        // 1 + 2
        let add_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // (1 + 2) * 3
        let mul_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(mul),
                Box::new(add_expr),
                Span::default(),
            )),
            Box::new(make_int(3)),
            Span::default(),
        );

        // ((1 + 2) * 3) - 4
        let sub_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(sub),
                Box::new(mul_expr),
                Span::default(),
            )),
            Box::new(make_int(4)),
            Span::default(),
        );

        let result = eval.eval(&sub_expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(5)));
    }

    #[test]
    fn test_eval_list_head() {
        let eval = Evaluator::new(EvalMode::Strict);

        // head [1, 2, 3] = 1
        let head_var = Expr::Var(make_var("head", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3]);
        let app = Expr::App(Box::new(head_var), Box::new(list_expr), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(1)));
    }

    #[test]
    fn test_eval_list_tail() {
        let eval = Evaluator::new(EvalMode::Strict);

        // tail [1, 2, 3] = [2, 3]
        let tail_var = Expr::Var(make_var("tail", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3]);
        let app = Expr::App(Box::new(tail_var), Box::new(list_expr), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].as_int(), Some(2));
        assert_eq!(list[1].as_int(), Some(3));
    }

    #[test]
    fn test_eval_list_length() {
        let eval = Evaluator::new(EvalMode::Strict);

        // length [1, 2, 3, 4, 5] = 5
        let length_var = Expr::Var(make_var("length", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);
        let app = Expr::App(Box::new(length_var), Box::new(list_expr), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(5)));
    }

    #[test]
    fn test_eval_list_null() {
        let eval = Evaluator::new(EvalMode::Strict);

        // null [] = true
        let null_var = Expr::Var(make_var("null", 100), Span::default());
        let nil = Expr::Var(
            Var::new(Symbol::intern("[]"), VarId::new(9999), Ty::Error),
            Span::default(),
        );
        let app = Expr::App(Box::new(null_var), Box::new(nil), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));

        // null [1] = false
        let null_var2 = Expr::Var(make_var("null", 101), Span::default());
        let list_expr = build_list_expr(vec![1]);
        let app2 = Expr::App(Box::new(null_var2), Box::new(list_expr), Span::default());

        let result2 = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result2.as_bool(), Some(false));
    }

    #[test]
    fn test_eval_list_reverse() {
        let eval = Evaluator::new(EvalMode::Strict);

        // reverse [1, 2, 3] = [3, 2, 1]
        let reverse_var = Expr::Var(make_var("reverse", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3]);
        let app = Expr::App(Box::new(reverse_var), Box::new(list_expr), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_int(), Some(3));
        assert_eq!(list[1].as_int(), Some(2));
        assert_eq!(list[2].as_int(), Some(1));
    }

    #[test]
    fn test_eval_list_take() {
        let eval = Evaluator::new(EvalMode::Strict);

        // take 2 [1, 2, 3, 4, 5] = [1, 2]
        let take_var = Expr::Var(make_var("take", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);
        let app1 = Expr::App(Box::new(take_var), Box::new(make_int(2)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(list_expr), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].as_int(), Some(1));
        assert_eq!(list[1].as_int(), Some(2));
    }

    #[test]
    fn test_eval_list_drop() {
        let eval = Evaluator::new(EvalMode::Strict);

        // drop 2 [1, 2, 3, 4, 5] = [3, 4, 5]
        let drop_var = Expr::Var(make_var("drop", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);
        let app1 = Expr::App(Box::new(drop_var), Box::new(make_int(2)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(list_expr), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_int(), Some(3));
        assert_eq!(list[1].as_int(), Some(4));
        assert_eq!(list[2].as_int(), Some(5));
    }

    #[test]
    fn test_eval_multiple_lambdas() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (\x -> \y -> x + y) 3 4 = 7
        let x = make_var("x", 0);
        let y = make_var("y", 1);
        let add = Expr::Var(make_var("+", 100), Span::default());

        // x + y
        let add_xy = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(Expr::Var(y.clone(), Span::default())),
            Span::default(),
        );

        // \y -> x + y
        let inner_lam = Expr::Lam(y, Box::new(add_xy), Span::default());

        // \x -> \y -> x + y
        let outer_lam = Expr::Lam(x, Box::new(inner_lam), Span::default());

        // (\x -> \y -> x + y) 3
        let app1 = Expr::App(Box::new(outer_lam), Box::new(make_int(3)), Span::default());

        // (\x -> \y -> x + y) 3 4
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(4)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(7)));
    }

    #[test]
    fn test_eval_closure_capture() {
        let eval = Evaluator::new(EvalMode::Strict);

        // let x = 10 in (\y -> x + y) 5 = 15
        let x = make_var("x", 0);
        let y = make_var("y", 1);
        let add = Expr::Var(make_var("+", 100), Span::default());

        // x + y
        let add_xy = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(Expr::Var(y.clone(), Span::default())),
            Span::default(),
        );

        // \y -> x + y
        let lam = Expr::Lam(y, Box::new(add_xy), Span::default());

        // (\y -> x + y) 5
        let app = Expr::App(Box::new(lam), Box::new(make_int(5)), Span::default());

        // let x = 10 in (\y -> x + y) 5
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(make_int(10)))),
            Box::new(app),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(15)));
    }
}
