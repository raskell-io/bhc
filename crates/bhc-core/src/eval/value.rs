//! Runtime values for the Core IR interpreter.
//!
//! This module defines the `Value` type that represents values during
//! interpretation of Core IR expressions.

use std::fmt;
use std::sync::Arc;

use bhc_intern::Symbol;

use crate::{DataCon, Expr, Var};

/// A runtime value produced by evaluating Core IR.
#[derive(Clone)]
pub enum Value {
    /// An integer value.
    Int(i64),

    /// An arbitrary precision integer.
    Integer(i128),

    /// A single-precision float.
    Float(f32),

    /// A double-precision float.
    Double(f64),

    /// A character.
    Char(char),

    /// A string.
    String(Arc<str>),

    /// A closure (lambda with captured environment).
    Closure(Closure),

    /// A data constructor value (fully or partially applied).
    Data(DataValue),

    /// A thunk (unevaluated expression with environment).
    /// Used for lazy evaluation in Default Profile.
    Thunk(Thunk),

    /// A special value representing a primitive operation.
    PrimOp(PrimOp),

    /// A partially applied primitive operation.
    PartialPrimOp(PrimOp, Vec<Value>),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(n) => write!(f, "{n}"),
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}f"),
            Self::Double(n) => write!(f, "{n}"),
            Self::Char(c) => write!(f, "{c:?}"),
            Self::String(s) => write!(f, "{s:?}"),
            Self::Closure(c) => write!(f, "<closure {}>", c.var.name),
            Self::Data(d) => {
                write!(f, "{}", d.con.name)?;
                for arg in &d.args {
                    write!(f, " {arg:?}")?;
                }
                Ok(())
            }
            Self::Thunk(_) => write!(f, "<thunk>"),
            Self::PrimOp(op) => write!(f, "<primop {op:?}>"),
            Self::PartialPrimOp(op, args) => {
                write!(f, "<partial {op:?} applied to {} args>", args.len())
            }
        }
    }
}

impl Value {
    /// Returns true if this value needs to be forced (is a thunk).
    #[must_use]
    pub fn is_thunk(&self) -> bool {
        matches!(self, Self::Thunk(_))
    }

    /// Converts an integer value, returning None if not an integer.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(n) => Some(*n),
            Self::Integer(n) => i64::try_from(*n).ok(),
            _ => None,
        }
    }

    /// Converts to a double value, returning None if not numeric.
    #[must_use]
    pub fn as_double(&self) -> Option<f64> {
        match self {
            Self::Double(n) => Some(*n),
            Self::Float(n) => Some(f64::from(*n)),
            Self::Int(n) => Some(*n as f64),
            Self::Integer(n) => Some(*n as f64),
            _ => None,
        }
    }

    /// Converts to a bool value (data constructor True/False).
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Data(d) if d.args.is_empty() => {
                let name = d.con.name.as_str();
                match name {
                    "True" => Some(true),
                    "False" => Some(false),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Creates a boolean value.
    #[must_use]
    pub fn bool(b: bool) -> Self {
        use bhc_types::{Kind, TyCon};
        let name = if b {
            Symbol::intern("True")
        } else {
            Symbol::intern("False")
        };
        Self::Data(DataValue {
            con: DataCon {
                name,
                ty_con: TyCon::new(Symbol::intern("Bool"), Kind::Star),
                tag: if b { 1 } else { 0 },
                arity: 0,
            },
            args: Vec::new(),
        })
    }

    /// Creates a unit value `()`.
    #[must_use]
    pub fn unit() -> Self {
        use bhc_types::{Kind, TyCon};
        Self::Data(DataValue {
            con: DataCon {
                name: Symbol::intern("()"),
                ty_con: TyCon::new(Symbol::intern("()"), Kind::Star),
                tag: 0,
                arity: 0,
            },
            args: Vec::new(),
        })
    }

    /// Creates an empty list `[]`.
    #[must_use]
    pub fn nil() -> Self {
        use bhc_types::{Kind, TyCon};
        Self::Data(DataValue {
            con: DataCon {
                name: Symbol::intern("[]"),
                ty_con: TyCon::new(Symbol::intern("[]"), Kind::star_to_star()),
                tag: 0,
                arity: 0,
            },
            args: Vec::new(),
        })
    }

    /// Creates a cons cell `x : xs`.
    #[must_use]
    pub fn cons(head: Value, tail: Value) -> Self {
        use bhc_types::{Kind, TyCon};
        Self::Data(DataValue {
            con: DataCon {
                name: Symbol::intern(":"),
                ty_con: TyCon::new(Symbol::intern("[]"), Kind::star_to_star()),
                tag: 1,
                arity: 2,
            },
            args: vec![head, tail],
        })
    }

    /// Converts a list value to a Vec, returning None if not a list.
    #[must_use]
    pub fn as_list(&self) -> Option<Vec<Value>> {
        let mut result = Vec::new();
        let mut current = self;

        loop {
            match current {
                Self::Data(d) if d.con.name.as_str() == "[]" => {
                    return Some(result);
                }
                Self::Data(d) if d.con.name.as_str() == ":" && d.args.len() == 2 => {
                    result.push(d.args[0].clone());
                    current = &d.args[1];
                }
                _ => return None,
            }
        }
    }

    /// Creates a list value from a Vec.
    #[must_use]
    pub fn from_list(values: Vec<Value>) -> Self {
        values
            .into_iter()
            .rev()
            .fold(Self::nil(), |acc, v| Self::cons(v, acc))
    }
}

/// A closure capturing a lambda and its environment.
#[derive(Clone)]
pub struct Closure {
    /// The bound variable.
    pub var: Var,
    /// The body expression.
    pub body: Box<Expr>,
    /// The captured environment.
    pub env: super::Env,
}

/// A data constructor value with its arguments.
#[derive(Clone, Debug)]
pub struct DataValue {
    /// The data constructor.
    pub con: DataCon,
    /// The constructor arguments (may be partial).
    pub args: Vec<Value>,
}

impl DataValue {
    /// Returns true if this data value is fully applied.
    #[must_use]
    pub fn is_saturated(&self) -> bool {
        self.args.len() == self.con.arity as usize
    }
}

/// A thunk representing an unevaluated expression.
#[derive(Clone)]
pub struct Thunk {
    /// The unevaluated expression.
    pub expr: Box<Expr>,
    /// The environment at thunk creation time.
    pub env: super::Env,
}

/// Primitive operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimOp {
    // Arithmetic
    /// Integer addition.
    AddInt,
    /// Integer subtraction.
    SubInt,
    /// Integer multiplication.
    MulInt,
    /// Integer division.
    DivInt,
    /// Integer modulo.
    ModInt,
    /// Integer negation.
    NegInt,

    // Floating point
    /// Double addition.
    AddDouble,
    /// Double subtraction.
    SubDouble,
    /// Double multiplication.
    MulDouble,
    /// Double division.
    DivDouble,
    /// Double negation.
    NegDouble,

    // Comparison
    /// Integer equality.
    EqInt,
    /// Integer less-than.
    LtInt,
    /// Integer less-than-or-equal.
    LeInt,
    /// Integer greater-than.
    GtInt,
    /// Integer greater-than-or-equal.
    GeInt,

    /// Double equality.
    EqDouble,
    /// Double less-than.
    LtDouble,

    // Boolean
    /// Boolean and.
    AndBool,
    /// Boolean or.
    OrBool,
    /// Boolean not.
    NotBool,

    // Conversion
    /// Int to Double.
    IntToDouble,
    /// Double to Int.
    DoubleToInt,

    // Char/String
    /// Character equality.
    EqChar,
    /// Char to Int (ord).
    CharToInt,
    /// Int to Char (chr).
    IntToChar,

    // Seq (for strict evaluation)
    /// Evaluate first arg to WHNF, return second.
    Seq,

    // Error
    /// Throw an error.
    Error,
}

impl PrimOp {
    /// Returns the arity of this primitive operation.
    #[must_use]
    pub fn arity(self) -> usize {
        match self {
            Self::NegInt | Self::NegDouble | Self::NotBool | Self::IntToDouble
            | Self::DoubleToInt | Self::CharToInt | Self::IntToChar | Self::Error => 1,
            _ => 2,
        }
    }

    /// Looks up a primitive operation by name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "+#" | "plusInt#" => Some(Self::AddInt),
            "-#" | "minusInt#" => Some(Self::SubInt),
            "*#" | "timesInt#" => Some(Self::MulInt),
            "quotInt#" => Some(Self::DivInt),
            "remInt#" => Some(Self::ModInt),
            "negateInt#" => Some(Self::NegInt),
            "+##" | "plusDouble#" => Some(Self::AddDouble),
            "-##" | "minusDouble#" => Some(Self::SubDouble),
            "*##" | "timesDouble#" => Some(Self::MulDouble),
            "/##" | "divideDouble#" => Some(Self::DivDouble),
            "negateDouble#" => Some(Self::NegDouble),
            "==#" | "eqInt#" => Some(Self::EqInt),
            "<#" | "ltInt#" => Some(Self::LtInt),
            "<=#" | "leInt#" => Some(Self::LeInt),
            ">#" | "gtInt#" => Some(Self::GtInt),
            ">=#" | "geInt#" => Some(Self::GeInt),
            "==##" | "eqDouble#" => Some(Self::EqDouble),
            "<##" | "ltDouble#" => Some(Self::LtDouble),
            "andBool" => Some(Self::AndBool),
            "orBool" => Some(Self::OrBool),
            "not" => Some(Self::NotBool),
            "int2Double#" => Some(Self::IntToDouble),
            "double2Int#" => Some(Self::DoubleToInt),
            "eqChar#" => Some(Self::EqChar),
            "ord#" => Some(Self::CharToInt),
            "chr#" => Some(Self::IntToChar),
            "seq" => Some(Self::Seq),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}
