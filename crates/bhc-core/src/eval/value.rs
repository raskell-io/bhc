//! Runtime values for the Core IR interpreter.
//!
//! This module defines the `Value` type that represents values during
//! interpretation of Core IR expressions.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use bhc_intern::Symbol;

use crate::uarray::UArray;
use crate::{DataCon, Expr, Var};

/// The kind of a file handle (System.IO).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleKind {
    Stdin,
    Stdout,
    Stderr,
    File,
}

/// A file handle wrapping a Rust file or stdio stream.
pub struct HandleValue {
    /// The kind of handle.
    pub kind: HandleKind,
    /// The underlying file, if any (None for closed handles).
    pub file: Mutex<Option<std::fs::File>>,
    /// Whether this is a read or write handle.
    pub readable: bool,
    /// Whether this is writable.
    pub writable: bool,
}

impl HandleValue {
    /// Create a handle for stdin.
    pub fn stdin() -> Self {
        Self { kind: HandleKind::Stdin, file: Mutex::new(None), readable: true, writable: false }
    }
    /// Create a handle for stdout.
    pub fn stdout() -> Self {
        Self { kind: HandleKind::Stdout, file: Mutex::new(None), readable: false, writable: true }
    }
    /// Create a handle for stderr.
    pub fn stderr() -> Self {
        Self { kind: HandleKind::Stderr, file: Mutex::new(None), readable: false, writable: true }
    }
    /// Create a file handle.
    pub fn from_file(file: std::fs::File, readable: bool, writable: bool) -> Self {
        Self { kind: HandleKind::File, file: Mutex::new(Some(file)), readable, writable }
    }
}

impl fmt::Debug for HandleValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<handle {:?}>", self.kind)
    }
}

impl Clone for HandleValue {
    fn clone(&self) -> Self {
        // Handles are shared via Arc, so cloning just copies metadata.
        // The file itself is not cloned — stdin/stdout/stderr are singletons.
        Self {
            kind: self.kind,
            file: Mutex::new(None), // Cloned handles lose file reference
            readable: self.readable,
            writable: self.writable,
        }
    }
}

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

    /// An unboxed integer array.
    UArrayInt(UArray<i64>),

    /// An unboxed double array.
    UArrayDouble(UArray<f64>),

    /// An ordered map (Data.Map) backed by BTreeMap.
    Map(Arc<BTreeMap<OrdValue, Value>>),

    /// An ordered set (Data.Set) backed by BTreeSet.
    Set(Arc<BTreeSet<OrdValue>>),

    /// An integer-keyed map (Data.IntMap) backed by BTreeMap<i64, Value>.
    IntMap(Arc<BTreeMap<i64, Value>>),

    /// An integer set (Data.IntSet) backed by BTreeSet<i64>.
    IntSet(Arc<BTreeSet<i64>>),

    /// A file handle (System.IO).
    Handle(Arc<HandleValue>),

    /// A mutable reference (Data.IORef).
    IORef(Arc<Mutex<Value>>),
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
            Self::UArrayInt(arr) => write!(f, "UArray[Int; {}]", arr.len()),
            Self::UArrayDouble(arr) => write!(f, "UArray[Double; {}]", arr.len()),
            Self::Map(m) => write!(f, "Map[size={}]", m.len()),
            Self::Set(s) => write!(f, "Set[size={}]", s.len()),
            Self::IntMap(m) => write!(f, "IntMap[size={}]", m.len()),
            Self::IntSet(s) => write!(f, "IntSet[size={}]", s.len()),
            Self::Handle(h) => write!(f, "{h:?}"),
            Self::IORef(_) => write!(f, "<IORef>"),
        }
    }
}

/// Returns true if a value needs parentheses when printed as a nested argument.
fn needs_parens(v: &Value) -> bool {
    matches!(v, Value::Data(d) if !d.args.is_empty()
        && d.con.name.as_str() != "[]"
        && d.con.name.as_str() != ":"
        && !d.con.name.as_str().starts_with('('))
}

/// Format a floating-point number ensuring a decimal point is present.
fn fmt_float(f: &mut fmt::Formatter<'_>, v: f64) -> fmt::Result {
    if v.is_nan() {
        write!(f, "NaN")
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            write!(f, "Infinity")
        } else {
            write!(f, "-Infinity")
        }
    } else if v.fract() == 0.0 {
        write!(f, "{v:.1}")
    } else {
        write!(f, "{v}")
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(n) => write!(f, "{n}"),
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => fmt_float(f, f64::from(*n)),
            Self::Double(n) => fmt_float(f, *n),
            Self::Char(c) => write!(f, "'{c}'"),
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Closure(_) => write!(f, "<<function>>"),
            Self::Thunk(_) => write!(f, "<<thunk>>"),
            Self::PrimOp(op) => write!(f, "<<primop: {op:?}>>"),
            Self::PartialPrimOp(op, _) => write!(f, "<<primop: {op:?} (partial)>>"),
            Self::Handle(_) => write!(f, "<<handle>>"),
            Self::IORef(_) => write!(f, "<<ioref>>"),
            Self::UArrayInt(arr) => {
                let items: Vec<_> = arr.to_vec();
                write!(f, "UArray {items:?}")
            }
            Self::UArrayDouble(arr) => {
                let items: Vec<_> = arr.to_vec();
                write!(f, "UArray {items:?}")
            }
            Self::Map(m) => {
                write!(f, "fromList [")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "({},{})", k.0, v)?;
                }
                write!(f, "]")
            }
            Self::Set(s) => {
                write!(f, "fromList [")?;
                for (i, v) in s.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", v.0)?;
                }
                write!(f, "]")
            }
            Self::IntMap(m) => {
                write!(f, "fromList [")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "({k},{v})")?;
                }
                write!(f, "]")
            }
            Self::IntSet(s) => {
                write!(f, "fromList [")?;
                for (i, v) in s.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Self::Data(d) => {
                let name = d.con.name.as_str();

                // Unit
                if name == "()" {
                    return write!(f, "()");
                }

                // Booleans
                if name == "True" || name == "False" {
                    return write!(f, "{name}");
                }

                // List: try to collect into a Vec
                if name == "[]" || name == ":" {
                    if let Some(elems) = self.as_list() {
                        // Check if it's a string (list of Char)
                        if !elems.is_empty()
                            && elems.iter().all(|e| matches!(e, Value::Char(_)))
                        {
                            let s: std::string::String = elems
                                .iter()
                                .map(|e| match e {
                                    Value::Char(c) => *c,
                                    _ => unreachable!(),
                                })
                                .collect();
                            return write!(f, "\"{s}\"");
                        }
                        write!(f, "[")?;
                        for (i, elem) in elems.iter().enumerate() {
                            if i > 0 {
                                write!(f, ",")?;
                            }
                            write!(f, "{elem}")?;
                        }
                        return write!(f, "]");
                    }
                    // Partial cons — fall through to generic data display
                }

                // Tuples: (,), (,,), etc.
                if name.starts_with('(') && name.ends_with(')') && name.contains(',') {
                    write!(f, "(")?;
                    for (i, arg) in d.args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{arg}")?;
                    }
                    return write!(f, ")");
                }

                // Nullary constructor
                if d.args.is_empty() {
                    return write!(f, "{name}");
                }

                // Constructor with args
                write!(f, "{name}")?;
                for arg in &d.args {
                    write!(f, " ")?;
                    if needs_parens(arg) {
                        write!(f, "({arg})")?;
                    } else {
                        write!(f, "{arg}")?;
                    }
                }
                Ok(())
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

    /// Creates an integer UArray from a list of int values.
    #[must_use]
    pub fn uarray_int_from_list(list: &Self) -> Option<Self> {
        let values = list.as_list()?;
        let ints: Option<Vec<i64>> = values.iter().map(Self::as_int).collect();
        Some(Self::UArrayInt(UArray::from_vec(ints?)))
    }

    /// Creates a double UArray from a list of double values.
    #[must_use]
    pub fn uarray_double_from_list(list: &Self) -> Option<Self> {
        let values = list.as_list()?;
        let doubles: Option<Vec<f64>> = values.iter().map(Self::as_double).collect();
        Some(Self::UArrayDouble(UArray::from_vec(doubles?)))
    }

    /// Converts a UArray to a list value.
    #[must_use]
    pub fn uarray_to_list(&self) -> Option<Self> {
        match self {
            Self::UArrayInt(arr) => {
                let values: Vec<Self> = arr.to_vec().into_iter().map(Self::Int).collect();
                Some(Self::from_list(values))
            }
            Self::UArrayDouble(arr) => {
                let values: Vec<Self> = arr.to_vec().into_iter().map(Self::Double).collect();
                Some(Self::from_list(values))
            }
            _ => None,
        }
    }

    /// Returns the UArray as an integer array, if applicable.
    #[must_use]
    pub fn as_uarray_int(&self) -> Option<&UArray<i64>> {
        match self {
            Self::UArrayInt(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns the UArray as a double array, if applicable.
    #[must_use]
    pub fn as_uarray_double(&self) -> Option<&UArray<f64>> {
        match self {
            Self::UArrayDouble(arr) => Some(arr),
            _ => None,
        }
    }
}

/// A newtype around `Value` that implements `Ord` for use as map/set keys.
///
/// Only Int, Integer, Char, String, Bool, and Data (by tag then args) support
/// ordering. Attempting to compare closures, thunks, or other non-orderable
/// values will treat them as equal (fallback).
#[derive(Clone, Debug)]
pub struct OrdValue(pub Value);

impl OrdValue {
    /// Extract the inner Value.
    pub fn into_inner(self) -> Value {
        self.0
    }

    /// Borrow the inner Value.
    pub fn inner(&self) -> &Value {
        &self.0
    }
}

impl PartialEq for OrdValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for OrdValue {}

impl PartialOrd for OrdValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        // Discriminant ordering: Int=0, Integer=1, Float=2, Double=3, Char=4, String=5, Data=6, other=7
        fn disc(v: &Value) -> u8 {
            match v {
                Value::Int(_) => 0,
                Value::Integer(_) => 1,
                Value::Float(_) => 2,
                Value::Double(_) => 3,
                Value::Char(_) => 4,
                Value::String(_) => 5,
                Value::Data(_) => 6,
                _ => 7,
            }
        }
        let d1 = disc(&self.0);
        let d2 = disc(&other.0);
        if d1 != d2 {
            return d1.cmp(&d2);
        }
        match (&self.0, &other.0) {
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Value::Double(a), Value::Double(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Value::Char(a), Value::Char(b)) => a.cmp(b),
            (Value::String(a), Value::String(b)) => a.cmp(b),
            (Value::Data(a), Value::Data(b)) => {
                match a.con.tag.cmp(&b.con.tag) {
                    Ordering::Equal => {
                        for (ax, bx) in a.args.iter().zip(b.args.iter()) {
                            let c = OrdValue(ax.clone()).cmp(&OrdValue(bx.clone()));
                            if c != Ordering::Equal {
                                return c;
                            }
                        }
                        a.args.len().cmp(&b.args.len())
                    }
                    ord => ord,
                }
            }
            _ => Ordering::Equal,
        }
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

    // UArray operations
    /// Create an integer UArray from a list.
    UArrayFromList,
    /// Convert a UArray back to a list.
    UArrayToList,
    /// Map a function over a UArray.
    UArrayMap,
    /// Zip two UArrays with a function.
    UArrayZipWith,
    /// Fold over a UArray.
    UArrayFold,
    /// Sum all elements in a UArray.
    UArraySum,
    /// Get the length of a UArray.
    UArrayLength,
    /// Create a range [start..end).
    UArrayRange,

    // List operations
    /// Concatenate two lists.
    Concat,
    /// Map a function over a list and concatenate results.
    ConcatMap,
    /// Append an element to a list.
    Append,

    // Monad operations (for list monad)
    /// Monadic bind (>>=) for lists: xs >>= f = concatMap f xs
    ListBind,
    /// Monadic then (>>) for lists: xs >> ys = xs >>= \_ -> ys
    ListThen,
    /// Monadic return for lists: return x = [x]
    ListReturn,

    // Additional list operations
    /// Right fold: foldr f z xs
    Foldr,
    /// Left fold: foldl f z xs
    Foldl,
    /// Strict left fold: foldl' f z xs
    FoldlStrict,
    /// Filter: filter p xs
    Filter,
    /// Zip two lists: zip xs ys
    Zip,
    /// Zip with function: zipWith f xs ys
    ZipWith,
    /// Take n elements: take n xs
    Take,
    /// Drop n elements: drop n xs
    Drop,
    /// Head of list: head xs
    Head,
    /// Tail of list: tail xs
    Tail,
    /// Last element: last xs
    Last,
    /// All but last: init xs
    Init,
    /// Reverse a list: reverse xs
    Reverse,
    /// Null check: null xs
    Null,
    /// Element at index: xs !! n
    Index,
    /// Replicate: replicate n x
    Replicate,
    /// Enumeration: enumFromTo start end
    EnumFromTo,

    // Additional list operations (second batch)
    /// Even predicate: even n
    Even,
    /// Odd predicate: odd n
    Odd,
    /// List membership: elem x xs
    Elem,
    /// List non-membership: notElem x xs
    NotElem,
    /// Take while predicate holds: takeWhile p xs
    TakeWhile,
    /// Drop while predicate holds: dropWhile p xs
    DropWhile,
    /// Split at predicate: span p xs
    Span,
    /// Split at predicate negation: break p xs
    Break,
    /// Split at index: splitAt n xs
    SplitAt,
    /// Iterate function: iterate f x (returns first 1000 elements)
    Iterate,
    /// Repeat value: repeat x (returns first 1000 elements)
    Repeat,
    /// Cycle list: cycle xs (returns first 1000 elements)
    Cycle,
    /// Lookup in assoc list: lookup k xs
    Lookup,
    /// Unzip pairs: unzip xs
    Unzip,
    /// Product of list: product xs
    Product,
    /// Flip function arguments: flip f x y = f y x
    Flip,
    /// Minimum of two: min a b
    Min,
    /// Maximum of two: max a b
    Max,
    /// Identity conversion: fromIntegral n
    FromIntegral,
    /// Maybe eliminator: maybe def f m
    MaybeElim,
    /// Default from Maybe: fromMaybe def m
    FromMaybe,
    /// Either eliminator: either f g e
    EitherElim,
    /// isJust :: Maybe a -> Bool
    IsJust,
    /// isNothing :: Maybe a -> Bool
    IsNothing,
    /// Absolute value: abs n
    Abs,
    /// Sign: signum n
    Signum,
    /// curry :: ((a, b) -> c) -> a -> b -> c
    Curry,
    /// uncurry :: (a -> b -> c) -> (a, b) -> c
    Uncurry,
    /// swap :: (a, b) -> (b, a)
    Swap,
    /// any :: (a -> Bool) -> [a] -> Bool
    Any,
    /// all :: (a -> Bool) -> [a] -> Bool
    All,
    /// and :: [Bool] -> Bool
    And,
    /// or :: [Bool] -> Bool
    Or,
    /// lines :: String -> [String]
    Lines,
    /// unlines :: [String] -> String
    Unlines,
    /// words :: String -> [String]
    Words,
    /// unwords :: [String] -> String
    Unwords,
    /// show :: a -> String
    Show,
    /// id :: a -> a
    Id,
    /// const :: a -> b -> a
    Const,

    // IO operations
    /// Print a string followed by newline.
    PutStrLn,
    /// Print a string without newline.
    PutStr,
    /// Print a value using Show (for now, uses Debug).
    Print,
    /// Read a line from stdin.
    GetLine,
    /// IO bind (>>=) for IO monad.
    IoBind,
    /// IO then (>>) for IO monad.
    IoThen,
    /// IO return/pure.
    IoReturn,

    // Polymorphic monad operations (dispatch based on first argument type)
    /// Polymorphic bind (>>=): dispatches to IoBind or ListBind based on first arg.
    MonadBind,
    /// Polymorphic then (>>): dispatches to IoThen or ListThen based on first arg.
    MonadThen,

    // Prelude: Enum operations
    /// succ :: Enum a => a -> a
    Succ,
    /// pred :: Enum a => a -> a
    Pred,
    /// toEnum :: Enum a => Int -> a
    ToEnum,
    /// fromEnum :: Enum a => a -> Int
    FromEnum,

    // Prelude: Integral operations
    /// gcd :: Integral a => a -> a -> a
    Gcd,
    /// lcm :: Integral a => a -> a -> a
    Lcm,
    /// quot :: Integral a => a -> a -> a
    Quot,
    /// rem :: Integral a => a -> a -> a
    Rem,
    /// quotRem :: Integral a => a -> a -> (a, a)
    QuotRem,
    /// divMod :: Integral a => a -> a -> (a, a)
    DivMod,
    /// subtract :: Num a => a -> a -> a
    Subtract,

    // Prelude: Scan operations
    /// scanl :: (b -> a -> b) -> b -> [a] -> [b]
    Scanl,
    /// scanr :: (a -> b -> b) -> b -> [a] -> [b]
    Scanr,
    /// scanl1 :: (a -> a -> a) -> [a] -> [a]
    Scanl1,
    /// scanr1 :: (a -> a -> a) -> [a] -> [a]
    Scanr1,

    // Prelude: More list operations
    /// maximum :: Ord a => [a] -> a
    Maximum,
    /// minimum :: Ord a => [a] -> a
    Minimum,
    /// zip3 :: [a] -> [b] -> [c] -> [(a,b,c)]
    Zip3,
    /// zipWith3 :: (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
    ZipWith3,
    /// unzip3 :: [(a,b,c)] -> ([a],[b],[c])
    Unzip3,

    // Prelude: Show helpers
    /// showString :: String -> ShowS
    ShowString,
    /// showChar :: Char -> ShowS
    ShowChar,
    /// showParen :: Bool -> ShowS -> ShowS
    ShowParen,

    // Prelude: IO operations
    /// getChar :: IO Char
    GetChar,
    /// getContents :: IO String
    GetContents,
    /// readFile :: FilePath -> IO String
    ReadFile,
    /// writeFile :: FilePath -> String -> IO ()
    WriteFile,
    /// appendFile :: FilePath -> String -> IO ()
    AppendFile,
    /// interact :: (String -> String) -> IO ()
    Interact,

    // System.IO handle operations
    /// stdin :: Handle
    Stdin,
    /// stdout :: Handle
    Stdout,
    /// stderr :: Handle
    Stderr,
    /// openFile :: FilePath -> IOMode -> IO Handle
    OpenFile,
    /// hClose :: Handle -> IO ()
    HClose,
    /// hGetChar :: Handle -> IO Char
    HGetChar,
    /// hGetLine :: Handle -> IO String
    HGetLine,
    /// hGetContents :: Handle -> IO String
    HGetContents,
    /// hPutChar :: Handle -> Char -> IO ()
    HPutChar,
    /// hPutStr :: Handle -> String -> IO ()
    HPutStr,
    /// hPutStrLn :: Handle -> String -> IO ()
    HPutStrLn,
    /// hPrint :: Show a => Handle -> a -> IO ()
    HPrint,
    /// hFlush :: Handle -> IO ()
    HFlush,
    /// hIsEOF :: Handle -> IO Bool
    HIsEOF,
    /// hSetBuffering :: Handle -> BufferMode -> IO ()
    HSetBuffering,
    /// hGetBuffering :: Handle -> IO BufferMode
    HGetBuffering,
    /// hSeek :: Handle -> SeekMode -> Integer -> IO ()
    HSeek,
    /// hTell :: Handle -> IO Integer
    HTell,
    /// hFileSize :: Handle -> IO Integer
    HFileSize,
    /// withFile :: FilePath -> IOMode -> (Handle -> IO r) -> IO r
    WithFile,

    // Data.IORef operations
    /// newIORef :: a -> IO (IORef a)
    NewIORef,
    /// readIORef :: IORef a -> IO a
    ReadIORef,
    /// writeIORef :: IORef a -> a -> IO ()
    WriteIORef,
    /// modifyIORef :: IORef a -> (a -> a) -> IO ()
    ModifyIORef,
    /// modifyIORef' :: IORef a -> (a -> a) -> IO ()
    ModifyIORefStrict,
    /// atomicModifyIORef :: IORef a -> (a -> (a, b)) -> IO b
    AtomicModifyIORef,
    /// atomicModifyIORef' :: IORef a -> (a -> (a, b)) -> IO b
    AtomicModifyIORefStrict,

    // System.Exit operations
    /// exitSuccess :: IO a
    ExitSuccess,
    /// exitFailure :: IO a
    ExitFailure,
    /// exitWith :: ExitCode -> IO a
    ExitWith,

    // System.Environment operations
    /// getArgs :: IO [String]
    GetArgs,
    /// getProgName :: IO String
    GetProgName,
    /// getEnv :: String -> IO String
    GetEnv,
    /// lookupEnv :: String -> IO (Maybe String)
    LookupEnv,
    /// setEnv :: String -> String -> IO ()
    SetEnv,

    // System.Directory operations
    /// doesFileExist :: FilePath -> IO Bool
    DoesFileExist,
    /// doesDirectoryExist :: FilePath -> IO Bool
    DoesDirectoryExist,
    /// createDirectory :: FilePath -> IO ()
    CreateDirectory,
    /// createDirectoryIfMissing :: Bool -> FilePath -> IO ()
    CreateDirectoryIfMissing,
    /// removeFile :: FilePath -> IO ()
    RemoveFile,
    /// removeDirectory :: FilePath -> IO ()
    RemoveDirectory,
    /// getCurrentDirectory :: IO String
    GetCurrentDirectory,
    /// setCurrentDirectory :: FilePath -> IO ()
    SetCurrentDirectory,

    // ---- Control.Monad ----
    /// when :: Bool -> IO () -> IO ()
    MonadWhen,
    /// unless :: Bool -> IO () -> IO ()
    MonadUnless,
    /// guard :: Bool -> [()]  (for list monad / MonadPlus)
    MonadGuard,
    /// void :: Functor f => f a -> f ()
    MonadVoid,
    /// join :: Monad m => m (m a) -> m a
    MonadJoin,
    /// ap :: Monad m => m (a -> b) -> m a -> m b
    MonadAp,
    /// liftM :: Monad m => (a -> b) -> m a -> m b
    LiftM,
    /// liftM2 :: Monad m => (a -> b -> c) -> m a -> m b -> m c
    LiftM2,
    /// liftM3 :: Monad m => (a -> b -> c -> d) -> m a -> m b -> m c -> m d
    LiftM3,
    /// liftM4 :: (a -> b -> c -> d -> e) -> m a -> m b -> m c -> m d -> m e
    LiftM4,
    /// liftM5 :: (a -> b -> c -> d -> e -> f) -> m a -> m b -> m c -> m d -> m e -> m f
    LiftM5,
    /// filterM :: Monad m => (a -> m Bool) -> [a] -> m [a]
    FilterM,
    /// mapAndUnzipM :: Monad m => (a -> m (b, c)) -> [a] -> m ([b], [c])
    MapAndUnzipM,
    /// zipWithM :: Monad m => (a -> b -> m c) -> [a] -> [b] -> m [c]
    ZipWithM,
    /// zipWithM_ :: Monad m => (a -> b -> m c) -> [a] -> [b] -> m ()
    ZipWithM_,
    /// foldM :: Monad m => (b -> a -> m b) -> b -> [a] -> m b
    FoldM,
    /// foldM_ :: Monad m => (b -> a -> m b) -> b -> [a] -> m ()
    FoldM_,
    /// replicateM :: Monad m => Int -> m a -> m [a]
    ReplicateM,
    /// replicateM_ :: Monad m => Int -> m a -> m ()
    ReplicateM_,
    /// forever :: Monad m => m a -> m b
    Forever,
    /// mzero :: MonadPlus m => m a
    Mzero,
    /// mplus :: MonadPlus m => m a -> m a -> m a
    Mplus,
    /// msum :: MonadPlus m => [m a] -> m a
    Msum,
    /// mfilter :: MonadPlus m => (a -> Bool) -> m a -> m a
    Mfilter,
    /// (>=>) :: Monad m => (a -> m b) -> (b -> m c) -> a -> m c
    KleisliCompose,
    /// (<=<) :: Monad m => (b -> m c) -> (a -> m b) -> a -> m c
    KleisliComposeFlip,

    // ---- Control.Applicative ----
    /// liftA :: Applicative f => (a -> b) -> f a -> f b
    LiftA,
    /// liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
    LiftA2,
    /// liftA3 :: Applicative f => (a -> b -> c -> d) -> f a -> f b -> f c -> f d
    LiftA3,
    /// optional :: Alternative f => f a -> f (Maybe a)
    Optional,

    // ---- Control.Exception ----
    /// catch :: IO a -> (SomeException -> IO a) -> IO a
    ExnCatch,
    /// try :: IO a -> IO (Either SomeException a)
    ExnTry,
    /// throw :: SomeException -> a
    ExnThrow,
    /// throwIO :: SomeException -> IO a
    ExnThrowIO,
    /// bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
    ExnBracket,
    /// bracket_ :: IO a -> IO b -> IO c -> IO c
    ExnBracket_,
    /// bracketOnError :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
    ExnBracketOnError,
    /// finally :: IO a -> IO b -> IO a
    ExnFinally,
    /// onException :: IO a -> IO b -> IO a
    ExnOnException,
    /// handle :: (SomeException -> IO a) -> IO a -> IO a
    ExnHandle,
    /// handleJust :: (SomeException -> Maybe b) -> (b -> IO a) -> IO a -> IO a
    ExnHandleJust,
    /// evaluate :: a -> IO a
    ExnEvaluate,
    /// mask :: ((IO a -> IO a) -> IO b) -> IO b
    ExnMask,
    /// mask_ :: IO a -> IO a
    ExnMask_,
    /// uninterruptibleMask :: ((IO a -> IO a) -> IO b) -> IO b
    ExnUninterruptibleMask,
    /// uninterruptibleMask_ :: IO a -> IO a
    ExnUninterruptibleMask_,

    // ---- Control.Concurrent ----
    /// forkIO :: IO () -> IO ThreadId
    ForkIO,
    /// threadDelay :: Int -> IO ()
    ThreadDelay,
    /// myThreadId :: IO ThreadId
    MyThreadId,
    /// newMVar :: a -> IO (MVar a)
    NewMVar,
    /// newEmptyMVar :: IO (MVar a)
    NewEmptyMVar,
    /// takeMVar :: MVar a -> IO a
    TakeMVar,
    /// putMVar :: MVar a -> a -> IO ()
    PutMVar,
    /// readMVar :: MVar a -> IO a
    ReadMVar,
    /// throwTo :: ThreadId -> SomeException -> IO ()
    ThrowTo,
    /// killThread :: ThreadId -> IO ()
    KillThread,

    // ---- Data.Ord ----
    /// comparing :: Ord b => (a -> b) -> a -> a -> Ordering
    Comparing,
    /// clamp :: Ord a => (a, a) -> a -> a
    Clamp,

    // ---- Data.Foldable ----
    /// fold :: Monoid m => [m] -> m (= mconcat for lists)
    Fold,
    /// foldMap :: Monoid m => (a -> m) -> [a] -> m
    FoldMap,
    /// foldr' :: (a -> b -> b) -> b -> [a] -> b (strict foldr)
    FoldrStrict,
    /// foldl1 :: (a -> a -> a) -> [a] -> a
    Foldl1,
    /// foldr1 :: (a -> a -> a) -> [a] -> a
    Foldr1,
    /// maximumBy :: (a -> a -> Ordering) -> [a] -> a
    MaximumBy,
    /// minimumBy :: (a -> a -> Ordering) -> [a] -> a
    MinimumBy,
    /// asum :: [Maybe a] -> Maybe a (or Alternative)
    Asum,
    /// traverse_ :: (a -> f b) -> [a] -> f ()
    Traverse_,
    /// for_ :: [a] -> (a -> f b) -> f ()
    For_,
    /// sequenceA_ :: [f a] -> f ()
    SequenceA_,

    // ---- Data.Traversable ----
    /// traverse :: (a -> f b) -> [a] -> f [b]
    Traverse,
    /// sequenceA :: [f a] -> f [a]
    SequenceA,

    // ---- Data.Monoid ----
    /// mempty :: Monoid a => a
    Mempty,
    /// mappend :: Monoid a => a -> a -> a
    Mappend,
    /// mconcat :: Monoid a => [a] -> a
    Mconcat,

    // ---- Data.String ----
    /// fromString :: String -> a (IsString class)
    FromString,

    // ---- Data.Bits ----
    /// (.&.) :: Bits a => a -> a -> a
    BitAnd,
    /// (.|.) :: Bits a => a -> a -> a
    BitOr,
    /// xor :: Bits a => a -> a -> a
    BitXor,
    /// complement :: Bits a => a -> a
    BitComplement,
    /// shift :: Bits a => a -> Int -> a
    BitShift,
    /// shiftL :: Bits a => a -> Int -> a
    BitShiftL,
    /// shiftR :: Bits a => a -> Int -> a
    BitShiftR,
    /// rotate :: Bits a => a -> Int -> a
    BitRotate,
    /// rotateL :: Bits a => a -> Int -> a
    BitRotateL,
    /// rotateR :: Bits a => a -> Int -> a
    BitRotateR,
    /// bit :: Bits a => Int -> a
    BitBit,
    /// setBit :: Bits a => a -> Int -> a
    BitSetBit,
    /// clearBit :: Bits a => a -> Int -> a
    BitClearBit,
    /// complementBit :: Bits a => a -> Int -> a
    BitComplementBit,
    /// testBit :: Bits a => a -> Int -> Bool
    BitTestBit,
    /// popCount :: Bits a => a -> Int
    BitPopCount,
    /// zeroBits :: Bits a => a
    BitZeroBits,
    /// countLeadingZeros :: FiniteBits a => a -> Int
    BitCountLeadingZeros,
    /// countTrailingZeros :: FiniteBits a => a -> Int
    BitCountTrailingZeros,

    // ---- Data.Proxy ----
    /// asProxyTypeOf :: a -> proxy a -> a
    AsProxyTypeOf,

    // ---- Data.Void ----
    /// absurd :: Void -> a
    Absurd,
    /// vacuous :: Functor f => f Void -> f a
    Vacuous,

    // Prelude: otherwise and misc
    /// otherwise :: Bool (always True)
    Otherwise,
    /// until :: (a -> Bool) -> (a -> a) -> a -> a
    Until,
    /// asTypeOf :: a -> a -> a
    AsTypeOf,
    /// realToFrac :: (Real a, Fractional b) => a -> b
    RealToFrac,

    // Data.List operations
    /// sort :: Ord a => [a] -> [a]
    Sort,
    /// sortBy :: (a -> a -> Ordering) -> [a] -> [a]
    SortBy,
    /// sortOn :: Ord b => (a -> b) -> [a] -> [a]
    SortOn,
    /// nub :: Eq a => [a] -> [a]
    Nub,
    /// nubBy :: (a -> a -> Bool) -> [a] -> [a]
    NubBy,
    /// group :: Eq a => [a] -> [[a]]
    Group,
    /// groupBy :: (a -> a -> Bool) -> [a] -> [[a]]
    GroupBy,
    /// intersperse :: a -> [a] -> [a]
    Intersperse,
    /// intercalate :: [a] -> [[a]] -> [a]
    Intercalate,
    /// transpose :: [[a]] -> [[a]]
    Transpose,
    /// subsequences :: [a] -> [[a]]
    Subsequences,
    /// permutations :: [a] -> [[a]]
    Permutations,
    /// partition :: (a -> Bool) -> [a] -> ([a], [a])
    Partition,
    /// find :: (a -> Bool) -> [a] -> Maybe a
    Find,
    /// stripPrefix :: Eq a => [a] -> [a] -> Maybe [a]
    StripPrefix,
    /// isPrefixOf :: Eq a => [a] -> [a] -> Bool
    IsPrefixOf,
    /// isSuffixOf :: Eq a => [a] -> [a] -> Bool
    IsSuffixOf,
    /// isInfixOf :: Eq a => [a] -> [a] -> Bool
    IsInfixOf,
    /// delete :: Eq a => a -> [a] -> [a]
    Delete,
    /// deleteBy :: (a -> a -> Bool) -> a -> [a] -> [a]
    DeleteBy,
    /// union :: Eq a => [a] -> [a] -> [a]
    Union,
    /// unionBy :: (a -> a -> Bool) -> [a] -> [a] -> [a]
    UnionBy,
    /// intersect :: Eq a => [a] -> [a] -> [a]
    Intersect,
    /// intersectBy :: (a -> a -> Bool) -> [a] -> [a] -> [a]
    IntersectBy,
    /// (\\) :: Eq a => [a] -> [a] -> [a]
    ListDiff,
    /// tails :: [a] -> [[a]]
    Tails,
    /// inits :: [a] -> [[a]]
    Inits,
    /// mapAccumL :: (acc -> x -> (acc, y)) -> acc -> [x] -> (acc, [y])
    MapAccumL,
    /// mapAccumR :: (acc -> x -> (acc, y)) -> acc -> [x] -> (acc, [y])
    MapAccumR,
    /// unfoldr :: (b -> Maybe (a, b)) -> b -> [a]
    Unfoldr,
    /// genericLength :: Num i => [a] -> i
    GenericLength,
    /// genericTake :: Integral i => i -> [a] -> [a]
    GenericTake,
    /// genericDrop :: Integral i => i -> [a] -> [a]
    GenericDrop,

    // Data.Char operations
    /// isAlpha :: Char -> Bool
    IsAlpha,
    /// isAlphaNum :: Char -> Bool
    IsAlphaNum,
    /// isAscii :: Char -> Bool
    IsAscii,
    /// isControl :: Char -> Bool
    IsControl,
    /// isDigit :: Char -> Bool
    IsDigit,
    /// isHexDigit :: Char -> Bool
    IsHexDigit,
    /// isLetter :: Char -> Bool
    IsLetter,
    /// isLower :: Char -> Bool
    IsLower,
    /// isNumber :: Char -> Bool
    IsNumber,
    /// isPrint :: Char -> Bool
    IsPrint,
    /// isPunctuation :: Char -> Bool
    IsPunctuation,
    /// isSpace :: Char -> Bool
    IsSpace,
    /// isSymbol :: Char -> Bool
    IsSymbol,
    /// isUpper :: Char -> Bool
    IsUpper,
    /// toLower :: Char -> Char
    ToLower,
    /// toUpper :: Char -> Char
    ToUpper,
    /// toTitle :: Char -> Char
    ToTitle,
    /// digitToInt :: Char -> Int
    DigitToInt,
    /// intToDigit :: Int -> Char
    IntToDigit,
    /// isLatin1 :: Char -> Bool
    IsLatin1,
    /// isAsciiLower :: Char -> Bool
    IsAsciiLower,
    /// isAsciiUpper :: Char -> Bool
    IsAsciiUpper,

    // Data.Function operations
    /// on :: (b -> b -> c) -> (a -> b) -> a -> a -> c
    On,
    /// fix :: (a -> a) -> a
    Fix,
    /// (&) :: a -> (a -> b) -> b
    Amp,

    // Data.Maybe additional operations
    /// listToMaybe :: [a] -> Maybe a
    ListToMaybe,
    /// maybeToList :: Maybe a -> [a]
    MaybeToList,
    /// catMaybes :: [Maybe a] -> [a]
    CatMaybes,
    /// mapMaybe :: (a -> Maybe b) -> [a] -> [b]
    MapMaybe,

    // Data.Either additional operations
    /// isLeft :: Either a b -> Bool
    IsLeft,
    /// isRight :: Either a b -> Bool
    IsRight,
    /// lefts :: [Either a b] -> [a]
    Lefts,
    /// rights :: [Either a b] -> [b]
    Rights,
    /// partitionEithers :: [Either a b] -> ([a], [b])
    PartitionEithers,

    // Numeric: math functions
    /// sqrt :: Floating a => a -> a
    Sqrt,
    /// exp :: Floating a => a -> a
    Exp,
    /// log :: Floating a => a -> a
    Log,
    /// sin :: Floating a => a -> a
    Sin,
    /// cos :: Floating a => a -> a
    Cos,
    /// tan :: Floating a => a -> a
    Tan,
    /// (^) :: (Num a, Integral b) => a -> b -> a
    Power,
    /// truncate :: (RealFrac a, Integral b) => a -> b
    Truncate,
    /// round :: (RealFrac a, Integral b) => a -> b
    Round,
    /// ceiling :: (RealFrac a, Integral b) => a -> b
    Ceiling,
    /// floor :: (RealFrac a, Integral b) => a -> b
    Floor,

    // Prelude: fst/snd
    /// fst :: (a, b) -> a
    Fst,
    /// snd :: (a, b) -> b
    Snd,

    // ========================================================
    // Data.Map PrimOps
    // ========================================================
    /// Data.Map.empty :: Map k v
    MapEmpty,
    /// Data.Map.singleton :: k -> v -> Map k v
    MapSingleton,
    /// Data.Map.null :: Map k v -> Bool
    MapNull,
    /// Data.Map.size :: Map k v -> Int
    MapSize,
    /// Data.Map.member :: Ord k => k -> Map k v -> Bool
    MapMember,
    /// Data.Map.notMember :: Ord k => k -> Map k v -> Bool
    MapNotMember,
    /// Data.Map.lookup :: Ord k => k -> Map k v -> Maybe v
    MapLookup,
    /// Data.Map.findWithDefault :: Ord k => v -> k -> Map k v -> v
    MapFindWithDefault,
    /// Data.Map.(!) :: Ord k => Map k v -> k -> v
    MapIndex,
    /// Data.Map.insert :: Ord k => k -> v -> Map k v -> Map k v
    MapInsert,
    /// Data.Map.insertWith :: Ord k => (v -> v -> v) -> k -> v -> Map k v -> Map k v
    MapInsertWith,
    /// Data.Map.delete :: Ord k => k -> Map k v -> Map k v
    MapDelete,
    /// Data.Map.adjust :: Ord k => (v -> v) -> k -> Map k v -> Map k v
    MapAdjust,
    /// Data.Map.update :: Ord k => (v -> Maybe v) -> k -> Map k v -> Map k v
    MapUpdate,
    /// Data.Map.alter :: Ord k => (Maybe v -> Maybe v) -> k -> Map k v -> Map k v
    MapAlter,
    /// Data.Map.union :: Ord k => Map k v -> Map k v -> Map k v
    MapUnion,
    /// Data.Map.unionWith :: Ord k => (v -> v -> v) -> Map k v -> Map k v -> Map k v
    MapUnionWith,
    /// Data.Map.unionWithKey :: Ord k => (k -> v -> v -> v) -> Map k v -> Map k v -> Map k v
    MapUnionWithKey,
    /// Data.Map.unions :: Ord k => [Map k v] -> Map k v
    MapUnions,
    /// Data.Map.intersection :: Ord k => Map k v -> Map k w -> Map k v
    MapIntersection,
    /// Data.Map.intersectionWith :: Ord k => (a -> b -> c) -> Map k a -> Map k b -> Map k c
    MapIntersectionWith,
    /// Data.Map.difference :: Ord k => Map k v -> Map k w -> Map k v
    MapDifference,
    /// Data.Map.differenceWith :: Ord k => (a -> b -> Maybe a) -> Map k a -> Map k b -> Map k a
    MapDifferenceWith,
    /// Data.Map.map :: (a -> b) -> Map k a -> Map k b
    MapMap,
    /// Data.Map.mapWithKey :: (k -> a -> b) -> Map k a -> Map k b
    MapMapWithKey,
    /// Data.Map.mapKeys :: Ord k2 => (k1 -> k2) -> Map k1 a -> Map k2 a
    MapMapKeys,
    /// Data.Map.filter :: (a -> Bool) -> Map k a -> Map k a
    MapFilter,
    /// Data.Map.filterWithKey :: (k -> a -> Bool) -> Map k a -> Map k a
    MapFilterWithKey,
    /// Data.Map.foldr :: (a -> b -> b) -> b -> Map k a -> b
    MapFoldr,
    /// Data.Map.foldl :: (a -> b -> a) -> a -> Map k b -> a
    MapFoldl,
    /// Data.Map.foldrWithKey :: (k -> a -> b -> b) -> b -> Map k a -> b
    MapFoldrWithKey,
    /// Data.Map.foldlWithKey :: (a -> k -> b -> a) -> a -> Map k b -> a
    MapFoldlWithKey,
    /// Data.Map.keys :: Map k v -> [k]
    MapKeys,
    /// Data.Map.elems :: Map k v -> [v]
    MapElems,
    /// Data.Map.assocs :: Map k v -> [(k, v)]
    MapAssocs,
    /// Data.Map.toList :: Map k v -> [(k, v)]
    MapToList,
    /// Data.Map.fromList :: Ord k => [(k, v)] -> Map k v
    MapFromList,
    /// Data.Map.fromListWith :: Ord k => (a -> a -> a) -> [(k, a)] -> Map k a
    MapFromListWith,
    /// Data.Map.toAscList :: Map k v -> [(k, v)]
    MapToAscList,
    /// Data.Map.toDescList :: Map k v -> [(k, v)]
    MapToDescList,
    /// Data.Map.isSubmapOf :: (Ord k, Eq v) => Map k v -> Map k v -> Bool
    MapIsSubmapOf,

    // ========================================================
    // Data.Set PrimOps
    // ========================================================
    /// Data.Set.empty :: Set a
    SetEmpty,
    /// Data.Set.singleton :: a -> Set a
    SetSingleton,
    /// Data.Set.null :: Set a -> Bool
    SetNull,
    /// Data.Set.size :: Set a -> Int
    SetSize,
    /// Data.Set.member :: Ord a => a -> Set a -> Bool
    SetMember,
    /// Data.Set.notMember :: Ord a => a -> Set a -> Bool
    SetNotMember,
    /// Data.Set.insert :: Ord a => a -> Set a -> Set a
    SetInsert,
    /// Data.Set.delete :: Ord a => a -> Set a -> Set a
    SetDelete,
    /// Data.Set.union :: Ord a => Set a -> Set a -> Set a
    SetUnion,
    /// Data.Set.unions :: Ord a => [Set a] -> Set a
    SetUnions,
    /// Data.Set.intersection :: Ord a => Set a -> Set a -> Set a
    SetIntersection,
    /// Data.Set.difference :: Ord a => Set a -> Set a -> Set a
    SetDifference,
    /// Data.Set.isSubsetOf :: Ord a => Set a -> Set a -> Bool
    SetIsSubsetOf,
    /// Data.Set.isProperSubsetOf :: Ord a => Set a -> Set a -> Bool
    SetIsProperSubsetOf,
    /// Data.Set.map :: Ord b => (a -> b) -> Set a -> Set b
    SetMap,
    /// Data.Set.filter :: (a -> Bool) -> Set a -> Set a
    SetFilter,
    /// Data.Set.partition :: (a -> Bool) -> Set a -> (Set a, Set a)
    SetPartition,
    /// Data.Set.foldr :: (a -> b -> b) -> b -> Set a -> b
    SetFoldr,
    /// Data.Set.foldl :: (a -> b -> a) -> a -> Set b -> a
    SetFoldl,
    /// Data.Set.toList :: Set a -> [a]
    SetToList,
    /// Data.Set.fromList :: Ord a => [a] -> Set a
    SetFromList,
    /// Data.Set.toAscList :: Set a -> [a]
    SetToAscList,
    /// Data.Set.toDescList :: Set a -> [a]
    SetToDescList,
    /// Data.Set.findMin :: Set a -> a
    SetFindMin,
    /// Data.Set.findMax :: Set a -> a
    SetFindMax,
    /// Data.Set.deleteMin :: Set a -> Set a
    SetDeleteMin,
    /// Data.Set.deleteMax :: Set a -> Set a
    SetDeleteMax,
    /// Data.Set.elems :: Set a -> [a]
    SetElems,
    /// Data.Set.lookupMin :: Set a -> Maybe a
    SetLookupMin,
    /// Data.Set.lookupMax :: Set a -> Maybe a
    SetLookupMax,

    // ========================================================
    // Data.IntMap PrimOps
    // ========================================================
    /// Data.IntMap.empty :: IntMap v
    IntMapEmpty,
    /// Data.IntMap.singleton :: Int -> v -> IntMap v
    IntMapSingleton,
    /// Data.IntMap.null :: IntMap v -> Bool
    IntMapNull,
    /// Data.IntMap.size :: IntMap v -> Int
    IntMapSize,
    /// Data.IntMap.member :: Int -> IntMap v -> Bool
    IntMapMember,
    /// Data.IntMap.lookup :: Int -> IntMap v -> Maybe v
    IntMapLookup,
    /// Data.IntMap.findWithDefault :: v -> Int -> IntMap v -> v
    IntMapFindWithDefault,
    /// Data.IntMap.insert :: Int -> v -> IntMap v -> IntMap v
    IntMapInsert,
    /// Data.IntMap.insertWith :: (v -> v -> v) -> Int -> v -> IntMap v -> IntMap v
    IntMapInsertWith,
    /// Data.IntMap.delete :: Int -> IntMap v -> IntMap v
    IntMapDelete,
    /// Data.IntMap.adjust :: (v -> v) -> Int -> IntMap v -> IntMap v
    IntMapAdjust,
    /// Data.IntMap.union :: IntMap v -> IntMap v -> IntMap v
    IntMapUnion,
    /// Data.IntMap.unionWith :: (v -> v -> v) -> IntMap v -> IntMap v -> IntMap v
    IntMapUnionWith,
    /// Data.IntMap.intersection :: IntMap v -> IntMap w -> IntMap v
    IntMapIntersection,
    /// Data.IntMap.difference :: IntMap v -> IntMap w -> IntMap v
    IntMapDifference,
    /// Data.IntMap.map :: (a -> b) -> IntMap a -> IntMap b
    IntMapMap,
    /// Data.IntMap.mapWithKey :: (Int -> a -> b) -> IntMap a -> IntMap b
    IntMapMapWithKey,
    /// Data.IntMap.filter :: (a -> Bool) -> IntMap a -> IntMap a
    IntMapFilter,
    /// Data.IntMap.foldr :: (a -> b -> b) -> b -> IntMap a -> b
    IntMapFoldr,
    /// Data.IntMap.foldlWithKey :: (a -> Int -> b -> a) -> a -> IntMap b -> a
    IntMapFoldlWithKey,
    /// Data.IntMap.keys :: IntMap v -> [Int]
    IntMapKeys,
    /// Data.IntMap.elems :: IntMap v -> [v]
    IntMapElems,
    /// Data.IntMap.toList :: IntMap v -> [(Int, v)]
    IntMapToList,
    /// Data.IntMap.fromList :: [(Int, v)] -> IntMap v
    IntMapFromList,
    /// Data.IntMap.toAscList :: IntMap v -> [(Int, v)]
    IntMapToAscList,

    // ========================================================
    // Data.IntSet PrimOps
    // ========================================================
    /// Data.IntSet.empty :: IntSet
    IntSetEmpty,
    /// Data.IntSet.singleton :: Int -> IntSet
    IntSetSingleton,
    /// Data.IntSet.null :: IntSet -> Bool
    IntSetNull,
    /// Data.IntSet.size :: IntSet -> Int
    IntSetSize,
    /// Data.IntSet.member :: Int -> IntSet -> Bool
    IntSetMember,
    /// Data.IntSet.insert :: Int -> IntSet -> IntSet
    IntSetInsert,
    /// Data.IntSet.delete :: Int -> IntSet -> IntSet
    IntSetDelete,
    /// Data.IntSet.union :: IntSet -> IntSet -> IntSet
    IntSetUnion,
    /// Data.IntSet.intersection :: IntSet -> IntSet -> IntSet
    IntSetIntersection,
    /// Data.IntSet.difference :: IntSet -> IntSet -> IntSet
    IntSetDifference,
    /// Data.IntSet.isSubsetOf :: IntSet -> IntSet -> Bool
    IntSetIsSubsetOf,
    /// Data.IntSet.filter :: (Int -> Bool) -> IntSet -> IntSet
    IntSetFilter,
    /// Data.IntSet.foldr :: (Int -> b -> b) -> b -> IntSet -> b
    IntSetFoldr,
    /// Data.IntSet.toList :: IntSet -> [Int]
    IntSetToList,
    /// Data.IntSet.fromList :: [Int] -> IntSet
    IntSetFromList,

    // Dictionary operations (generated by type class desugaring)
    /// Select field N from a dictionary (tuple). Generated as `$sel_N` by
    /// HIR-to-Core lowering for type class method extraction.
    DictSelect(usize),
}

impl PrimOp {
    /// Returns the arity of this primitive operation.
    #[must_use]
    pub fn arity(self) -> usize {
        match self {
            // Arity 0
            Self::GetLine
            | Self::GetChar
            | Self::GetContents
            | Self::Otherwise
            | Self::MapEmpty
            | Self::SetEmpty
            | Self::IntMapEmpty
            | Self::IntSetEmpty
            // IO arity 0
            | Self::Stdin
            | Self::Stdout
            | Self::Stderr
            | Self::ExitSuccess
            | Self::ExitFailure
            | Self::GetArgs
            | Self::GetProgName
            | Self::GetCurrentDirectory
            | Self::Mzero
            | Self::MyThreadId
            | Self::NewEmptyMVar
            | Self::Mempty
            | Self::BitZeroBits => 0,
            // Arity 1
            Self::NegInt
            | Self::NegDouble
            | Self::NotBool
            | Self::IntToDouble
            | Self::DoubleToInt
            | Self::CharToInt
            | Self::IntToChar
            | Self::Error
            | Self::UArrayFromList
            | Self::UArrayToList
            | Self::UArraySum
            | Self::UArrayLength
            | Self::ListReturn
            | Self::Head
            | Self::Tail
            | Self::Last
            | Self::Init
            | Self::Reverse
            | Self::Null
            | Self::Even
            | Self::Odd
            | Self::Cycle
            | Self::Unzip
            | Self::Product
            | Self::FromIntegral
            | Self::IsJust
            | Self::IsNothing
            | Self::Abs
            | Self::Signum
            | Self::Swap
            | Self::Repeat
            | Self::And
            | Self::Or
            | Self::Lines
            | Self::Unlines
            | Self::Words
            | Self::Unwords
            | Self::Show
            | Self::Id
            | Self::PutStrLn
            | Self::PutStr
            | Self::Print
            | Self::IoReturn
            | Self::DictSelect(_)
            | Self::Succ
            | Self::Pred
            | Self::ToEnum
            | Self::FromEnum
            | Self::Maximum
            | Self::Minimum
            | Self::Unzip3
            | Self::ReadFile
            | Self::RealToFrac
            | Self::Nub
            | Self::Group
            | Self::Transpose
            | Self::Subsequences
            | Self::Permutations
            | Self::Tails
            | Self::Inits
            | Self::GenericLength
            | Self::IsAlpha
            | Self::IsAlphaNum
            | Self::IsAscii
            | Self::IsControl
            | Self::IsDigit
            | Self::IsHexDigit
            | Self::IsLetter
            | Self::IsLower
            | Self::IsNumber
            | Self::IsPrint
            | Self::IsPunctuation
            | Self::IsSpace
            | Self::IsSymbol
            | Self::IsUpper
            | Self::ToLower
            | Self::ToUpper
            | Self::ToTitle
            | Self::DigitToInt
            | Self::IntToDigit
            | Self::IsLatin1
            | Self::IsAsciiLower
            | Self::IsAsciiUpper
            | Self::Fix
            | Self::ListToMaybe
            | Self::MaybeToList
            | Self::CatMaybes
            | Self::IsLeft
            | Self::IsRight
            | Self::Lefts
            | Self::Rights
            | Self::PartitionEithers
            | Self::Sqrt
            | Self::Exp
            | Self::Log
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Truncate
            | Self::Round
            | Self::Ceiling
            | Self::Floor
            | Self::Fst
            | Self::Snd
            | Self::Scanl1
            | Self::Scanr1
            | Self::ShowString
            | Self::ShowChar
            // Container arity 1
            | Self::MapNull
            | Self::MapSize
            | Self::MapKeys
            | Self::MapElems
            | Self::MapAssocs
            | Self::MapToList
            | Self::MapFromList
            | Self::MapToAscList
            | Self::MapToDescList
            | Self::MapUnions
            | Self::SetNull
            | Self::SetSize
            | Self::SetSingleton
            | Self::SetToList
            | Self::SetFromList
            | Self::SetToAscList
            | Self::SetToDescList
            | Self::SetFindMin
            | Self::SetFindMax
            | Self::SetDeleteMin
            | Self::SetDeleteMax
            | Self::SetElems
            | Self::SetLookupMin
            | Self::SetLookupMax
            | Self::IntMapNull
            | Self::IntMapSize
            | Self::IntMapKeys
            | Self::IntMapElems
            | Self::IntMapToList
            | Self::IntMapFromList
            | Self::IntMapToAscList
            | Self::IntSetNull
            | Self::IntSetSize
            | Self::IntSetSingleton
            | Self::IntSetToList
            | Self::IntSetFromList
            // IO arity 1
            | Self::HClose
            | Self::HGetChar
            | Self::HGetLine
            | Self::HGetContents
            | Self::HFlush
            | Self::HIsEOF
            | Self::HGetBuffering
            | Self::HTell
            | Self::HFileSize
            | Self::NewIORef
            | Self::ReadIORef
            | Self::ExitWith
            | Self::GetEnv
            | Self::LookupEnv
            | Self::DoesFileExist
            | Self::DoesDirectoryExist
            | Self::CreateDirectory
            | Self::RemoveFile
            | Self::RemoveDirectory
            | Self::SetCurrentDirectory
            | Self::MonadVoid
            | Self::MonadJoin
            | Self::Forever
            | Self::Optional
            | Self::ExnThrow
            | Self::ExnThrowIO
            | Self::ExnEvaluate
            | Self::ExnMask_
            | Self::ExnUninterruptibleMask_
            | Self::ForkIO
            | Self::ThreadDelay
            | Self::NewMVar
            | Self::TakeMVar
            | Self::ReadMVar
            | Self::KillThread
            | Self::Fold
            | Self::Mconcat
            | Self::FromString
            | Self::BitComplement
            | Self::BitBit
            | Self::BitPopCount
            | Self::BitCountLeadingZeros
            | Self::BitCountTrailingZeros
            | Self::SequenceA
            | Self::SequenceA_
            | Self::Asum
            | Self::Absurd
            | Self::Vacuous => 1,
            // Arity 2
            Self::UArrayMap
            | Self::UArrayRange
            | Self::Concat
            | Self::ConcatMap
            | Self::Append
            | Self::ListBind
            | Self::ListThen
            | Self::Filter
            | Self::Zip
            | Self::Take
            | Self::Drop
            | Self::Index
            | Self::Replicate
            | Self::EnumFromTo
            | Self::Elem
            | Self::NotElem
            | Self::TakeWhile
            | Self::DropWhile
            | Self::Span
            | Self::Break
            | Self::SplitAt
            | Self::Iterate
            | Self::Lookup
            | Self::Min
            | Self::Max
            | Self::FromMaybe
            | Self::Any
            | Self::All
            | Self::Const
            | Self::Uncurry
            | Self::IoBind
            | Self::IoThen
            | Self::MonadBind
            | Self::MonadThen
            | Self::Gcd
            | Self::Lcm
            | Self::Quot
            | Self::Rem
            | Self::QuotRem
            | Self::DivMod
            | Self::Subtract
            | Self::WriteFile
            | Self::AppendFile
            | Self::AsTypeOf
            | Self::SortBy
            | Self::SortOn
            | Self::NubBy
            | Self::GroupBy
            | Self::Intersperse
            | Self::Intercalate
            | Self::Partition
            | Self::Find
            | Self::StripPrefix
            | Self::IsPrefixOf
            | Self::IsSuffixOf
            | Self::IsInfixOf
            | Self::Delete
            | Self::Union
            | Self::Intersect
            | Self::ListDiff
            | Self::GenericTake
            | Self::GenericDrop
            | Self::MapMaybe
            | Self::Power
            | Self::On
            | Self::Amp
            | Self::Unfoldr
            | Self::Sort
            | Self::Interact
            | Self::ShowParen
            // Container arity 2
            | Self::MapSingleton
            | Self::MapMember
            | Self::MapNotMember
            | Self::MapLookup
            | Self::MapIndex
            | Self::MapDelete
            | Self::MapUnion
            | Self::MapIntersection
            | Self::MapDifference
            | Self::MapMap
            | Self::MapMapKeys
            | Self::MapFilter
            | Self::MapIsSubmapOf
            | Self::MapFromListWith
            | Self::SetMember
            | Self::SetNotMember
            | Self::SetInsert
            | Self::SetDelete
            | Self::SetUnion
            | Self::SetUnions
            | Self::SetIntersection
            | Self::SetDifference
            | Self::SetIsSubsetOf
            | Self::SetIsProperSubsetOf
            | Self::SetMap
            | Self::SetFilter
            | Self::SetPartition
            | Self::IntMapSingleton
            | Self::IntMapMember
            | Self::IntMapLookup
            | Self::IntMapDelete
            | Self::IntMapUnion
            | Self::IntMapIntersection
            | Self::IntMapDifference
            | Self::IntMapMap
            | Self::IntMapFilter
            | Self::IntMapMapWithKey
            | Self::IntSetMember
            | Self::IntSetInsert
            | Self::IntSetDelete
            | Self::IntSetUnion
            | Self::IntSetIntersection
            | Self::IntSetDifference
            | Self::IntSetIsSubsetOf
            | Self::IntSetFilter
            // IO arity 2
            | Self::OpenFile
            | Self::HPutChar
            | Self::HPutStr
            | Self::HPutStrLn
            | Self::HPrint
            | Self::HSetBuffering
            | Self::WriteIORef
            | Self::ModifyIORef
            | Self::ModifyIORefStrict
            | Self::AtomicModifyIORef
            | Self::AtomicModifyIORefStrict
            | Self::SetEnv
            | Self::CreateDirectoryIfMissing
            | Self::MonadWhen
            | Self::MonadUnless
            | Self::MonadGuard
            | Self::MonadAp
            | Self::LiftM
            | Self::LiftA
            | Self::FilterM
            | Self::ReplicateM
            | Self::ReplicateM_
            | Self::Mplus
            | Self::Msum
            | Self::Mfilter
            | Self::ExnCatch
            | Self::ExnTry
            | Self::ExnFinally
            | Self::ExnOnException
            | Self::ExnHandle
            | Self::ExnMask
            | Self::ExnUninterruptibleMask
            | Self::PutMVar
            | Self::ThrowTo
            | Self::FoldMap
            | Self::Foldl1
            | Self::Foldr1
            | Self::MaximumBy
            | Self::MinimumBy
            | Self::Traverse
            | Self::Traverse_
            | Self::For_
            | Self::Mappend
            | Self::BitAnd
            | Self::BitOr
            | Self::BitXor
            | Self::BitShift
            | Self::BitShiftL
            | Self::BitShiftR
            | Self::BitRotate
            | Self::BitRotateL
            | Self::BitRotateR
            | Self::BitSetBit
            | Self::BitClearBit
            | Self::BitComplementBit
            | Self::BitTestBit
            | Self::Clamp
            | Self::AsProxyTypeOf => 2,
            // Arity 3
            Self::UArrayZipWith
            | Self::UArrayFold
            | Self::Foldr
            | Self::Foldl
            | Self::FoldlStrict
            | Self::ZipWith
            | Self::Flip
            | Self::MaybeElim
            | Self::EitherElim
            | Self::Curry
            | Self::Scanl
            | Self::Scanr
            | Self::Zip3
            | Self::Until
            | Self::DeleteBy
            | Self::UnionBy
            | Self::IntersectBy
            | Self::MapAccumL
            | Self::MapAccumR
            | Self::FoldrStrict
            | Self::Comparing
            // Container arity 3
            | Self::MapInsert
            | Self::MapAdjust
            | Self::MapUpdate
            | Self::MapAlter
            | Self::MapUnionWith
            | Self::MapIntersectionWith
            | Self::MapDifferenceWith
            | Self::MapMapWithKey
            | Self::MapFilterWithKey
            | Self::MapFoldr
            | Self::MapFoldl
            | Self::MapFindWithDefault
            | Self::SetFoldr
            | Self::SetFoldl
            | Self::IntMapInsert
            | Self::IntMapAdjust
            | Self::IntMapUnionWith
            | Self::IntMapFoldr
            | Self::IntMapFindWithDefault
            | Self::IntSetFoldr
            | Self::MapUnionWithKey
            | Self::MapFoldrWithKey
            | Self::MapFoldlWithKey
            | Self::IntMapFoldlWithKey
            | Self::LiftM2
            | Self::LiftA2
            | Self::MapAndUnzipM
            | Self::ZipWithM
            | Self::ZipWithM_
            | Self::FoldM
            | Self::FoldM_
            | Self::KleisliCompose
            | Self::KleisliComposeFlip
            | Self::ExnBracket
            | Self::ExnBracket_
            | Self::ExnBracketOnError
            | Self::ExnHandleJust
            // IO arity 3
            | Self::WithFile
            | Self::HSeek => 3,
            // Arity 4
            Self::ZipWith3
            | Self::MapInsertWith
            | Self::IntMapInsertWith
            | Self::LiftM3
            | Self::LiftA3 => 4,
            // Arity 5
            Self::LiftM4 => 5,
            // Arity 6
            Self::LiftM5 => 6,
            // Default arity 2 for arithmetic/comparison ops
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
            "ord" | "ord#" => Some(Self::CharToInt),
            "chr" | "chr#" => Some(Self::IntToChar),
            "seq" => Some(Self::Seq),
            "error" => Some(Self::Error),
            // UArray operations
            "uarrayFromList" | "fromList" => Some(Self::UArrayFromList),
            "uarrayToList" | "toList" => Some(Self::UArrayToList),
            "uarrayMap" | "map" => Some(Self::UArrayMap),
            "uarrayZipWith" => Some(Self::UArrayZipWith),
            "uarrayFold" => Some(Self::UArrayFold),
            "uarraySum" | "sum" => Some(Self::UArraySum),
            "uarrayLength" | "length" => Some(Self::UArrayLength),
            "uarrayRange" | "range" => Some(Self::UArrayRange),
            // List operations
            "++" | "concat" => Some(Self::Concat),
            "concatMap" => Some(Self::ConcatMap),
            "append" => Some(Self::Append),
            // Monad operations (list monad for now)
            ">>=" => Some(Self::ListBind),
            ">>" => Some(Self::ListThen),
            "return" => Some(Self::ListReturn),
            // Additional list operations
            "foldr" => Some(Self::Foldr),
            "foldl" => Some(Self::Foldl),
            "foldl'" => Some(Self::FoldlStrict),
            "filter" => Some(Self::Filter),
            "zip" => Some(Self::Zip),
            "zipWith" => Some(Self::ZipWith),
            "take" => Some(Self::Take),
            "drop" => Some(Self::Drop),
            "head" => Some(Self::Head),
            "tail" => Some(Self::Tail),
            "last" => Some(Self::Last),
            "init" => Some(Self::Init),
            "reverse" => Some(Self::Reverse),
            "null" => Some(Self::Null),
            "!!" => Some(Self::Index),
            "replicate" => Some(Self::Replicate),
            "enumFromTo" => Some(Self::EnumFromTo),
            // Additional list/prelude operations
            "even" => Some(Self::Even),
            "odd" => Some(Self::Odd),
            "elem" => Some(Self::Elem),
            "notElem" => Some(Self::NotElem),
            "takeWhile" => Some(Self::TakeWhile),
            "dropWhile" => Some(Self::DropWhile),
            "span" => Some(Self::Span),
            "break" => Some(Self::Break),
            "splitAt" => Some(Self::SplitAt),
            "iterate" => Some(Self::Iterate),
            "repeat" => Some(Self::Repeat),
            "cycle" => Some(Self::Cycle),
            "lookup" => Some(Self::Lookup),
            "unzip" => Some(Self::Unzip),
            "product" => Some(Self::Product),
            "flip" => Some(Self::Flip),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "fromIntegral" | "toInteger" => Some(Self::FromIntegral),
            "maybe" => Some(Self::MaybeElim),
            "fromMaybe" => Some(Self::FromMaybe),
            "either" => Some(Self::EitherElim),
            "isJust" => Some(Self::IsJust),
            "isNothing" => Some(Self::IsNothing),
            "abs" => Some(Self::Abs),
            "signum" => Some(Self::Signum),
            "curry" => Some(Self::Curry),
            "uncurry" => Some(Self::Uncurry),
            "swap" => Some(Self::Swap),
            "any" => Some(Self::Any),
            "all" => Some(Self::All),
            "and" => Some(Self::And),
            "or" => Some(Self::Or),
            "lines" => Some(Self::Lines),
            "unlines" => Some(Self::Unlines),
            "words" => Some(Self::Words),
            "unwords" => Some(Self::Unwords),
            "show" => Some(Self::Show),
            "id" => Some(Self::Id),
            "const" => Some(Self::Const),
            // IO operations
            "putStrLn" => Some(Self::PutStrLn),
            "putStr" => Some(Self::PutStr),
            "print" => Some(Self::Print),
            "getLine" => Some(Self::GetLine),
            // Enum operations
            "succ" => Some(Self::Succ),
            "pred" => Some(Self::Pred),
            "toEnum" => Some(Self::ToEnum),
            "fromEnum" => Some(Self::FromEnum),
            // Integral operations
            "gcd" => Some(Self::Gcd),
            "lcm" => Some(Self::Lcm),
            "quot" => Some(Self::Quot),
            "rem" => Some(Self::Rem),
            "quotRem" => Some(Self::QuotRem),
            "divMod" => Some(Self::DivMod),
            "subtract" => Some(Self::Subtract),
            // Scan operations
            "scanl" => Some(Self::Scanl),
            "scanl'" => Some(Self::Scanl),
            "scanr" => Some(Self::Scanr),
            "scanl1" => Some(Self::Scanl1),
            "scanr1" => Some(Self::Scanr1),
            // More list operations
            "maximum" => Some(Self::Maximum),
            "minimum" => Some(Self::Minimum),
            "zip3" => Some(Self::Zip3),
            "zipWith3" => Some(Self::ZipWith3),
            "unzip3" => Some(Self::Unzip3),
            // Show helpers
            "showString" => Some(Self::ShowString),
            "showChar" => Some(Self::ShowChar),
            "showParen" => Some(Self::ShowParen),
            // IO operations (additional)
            "getChar" => Some(Self::GetChar),
            "getContents" => Some(Self::GetContents),
            "readFile" => Some(Self::ReadFile),
            "writeFile" => Some(Self::WriteFile),
            "appendFile" => Some(Self::AppendFile),
            "interact" => Some(Self::Interact),
            // System.IO handles
            "stdin" => Some(Self::Stdin),
            "stdout" => Some(Self::Stdout),
            "stderr" => Some(Self::Stderr),
            "openFile" | "System.IO.openFile" => Some(Self::OpenFile),
            "hClose" | "System.IO.hClose" => Some(Self::HClose),
            "hGetChar" | "System.IO.hGetChar" => Some(Self::HGetChar),
            "hGetLine" | "System.IO.hGetLine" => Some(Self::HGetLine),
            "hGetContents" | "System.IO.hGetContents" => Some(Self::HGetContents),
            "hPutChar" | "System.IO.hPutChar" => Some(Self::HPutChar),
            "hPutStr" | "System.IO.hPutStr" => Some(Self::HPutStr),
            "hPutStrLn" | "System.IO.hPutStrLn" => Some(Self::HPutStrLn),
            "hPrint" | "System.IO.hPrint" => Some(Self::HPrint),
            "hFlush" | "System.IO.hFlush" => Some(Self::HFlush),
            "hIsEOF" | "System.IO.hIsEOF" => Some(Self::HIsEOF),
            "hSetBuffering" | "System.IO.hSetBuffering" => Some(Self::HSetBuffering),
            "hGetBuffering" | "System.IO.hGetBuffering" => Some(Self::HGetBuffering),
            "hSeek" | "System.IO.hSeek" => Some(Self::HSeek),
            "hTell" | "System.IO.hTell" => Some(Self::HTell),
            "hFileSize" | "System.IO.hFileSize" => Some(Self::HFileSize),
            "withFile" | "System.IO.withFile" => Some(Self::WithFile),
            // Data.IORef
            "newIORef" | "Data.IORef.newIORef" => Some(Self::NewIORef),
            "readIORef" | "Data.IORef.readIORef" => Some(Self::ReadIORef),
            "writeIORef" | "Data.IORef.writeIORef" => Some(Self::WriteIORef),
            "modifyIORef" | "Data.IORef.modifyIORef" => Some(Self::ModifyIORef),
            "modifyIORef'" | "Data.IORef.modifyIORef'" => Some(Self::ModifyIORefStrict),
            "atomicModifyIORef" | "Data.IORef.atomicModifyIORef" => Some(Self::AtomicModifyIORef),
            "atomicModifyIORef'" | "Data.IORef.atomicModifyIORef'" => Some(Self::AtomicModifyIORefStrict),
            // System.Exit
            "exitSuccess" | "System.Exit.exitSuccess" => Some(Self::ExitSuccess),
            "exitFailure" | "System.Exit.exitFailure" => Some(Self::ExitFailure),
            "exitWith" | "System.Exit.exitWith" => Some(Self::ExitWith),
            // System.Environment
            "getArgs" | "System.Environment.getArgs" => Some(Self::GetArgs),
            "getProgName" | "System.Environment.getProgName" => Some(Self::GetProgName),
            "getEnv" | "System.Environment.getEnv" => Some(Self::GetEnv),
            "lookupEnv" | "System.Environment.lookupEnv" => Some(Self::LookupEnv),
            "setEnv" | "System.Environment.setEnv" => Some(Self::SetEnv),
            // System.Directory
            "doesFileExist" | "System.Directory.doesFileExist" => Some(Self::DoesFileExist),
            "doesDirectoryExist" | "System.Directory.doesDirectoryExist" => Some(Self::DoesDirectoryExist),
            "createDirectory" | "System.Directory.createDirectory" => Some(Self::CreateDirectory),
            "createDirectoryIfMissing" | "System.Directory.createDirectoryIfMissing" => Some(Self::CreateDirectoryIfMissing),
            "removeFile" | "System.Directory.removeFile" => Some(Self::RemoveFile),
            "removeDirectory" | "System.Directory.removeDirectory" => Some(Self::RemoveDirectory),
            "getCurrentDirectory" | "System.Directory.getCurrentDirectory" => Some(Self::GetCurrentDirectory),
            "setCurrentDirectory" | "System.Directory.setCurrentDirectory" => Some(Self::SetCurrentDirectory),
            // Misc Prelude
            "otherwise" => Some(Self::Otherwise),
            "until" => Some(Self::Until),
            "asTypeOf" => Some(Self::AsTypeOf),
            "realToFrac" => Some(Self::RealToFrac),
            // Data.List
            "sort" => Some(Self::Sort),
            "sortBy" => Some(Self::SortBy),
            "sortOn" => Some(Self::SortOn),
            "nub" => Some(Self::Nub),
            "nubBy" => Some(Self::NubBy),
            "group" => Some(Self::Group),
            "groupBy" => Some(Self::GroupBy),
            "intersperse" => Some(Self::Intersperse),
            "intercalate" => Some(Self::Intercalate),
            "transpose" => Some(Self::Transpose),
            "subsequences" => Some(Self::Subsequences),
            "permutations" => Some(Self::Permutations),
            "partition" => Some(Self::Partition),
            "find" => Some(Self::Find),
            "stripPrefix" => Some(Self::StripPrefix),
            "isPrefixOf" => Some(Self::IsPrefixOf),
            "isSuffixOf" => Some(Self::IsSuffixOf),
            "isInfixOf" => Some(Self::IsInfixOf),
            "delete" => Some(Self::Delete),
            "deleteBy" => Some(Self::DeleteBy),
            "union" => Some(Self::Union),
            "unionBy" => Some(Self::UnionBy),
            "intersect" => Some(Self::Intersect),
            "intersectBy" => Some(Self::IntersectBy),
            "\\\\" => Some(Self::ListDiff),
            "tails" => Some(Self::Tails),
            "inits" => Some(Self::Inits),
            "mapAccumL" => Some(Self::MapAccumL),
            "mapAccumR" => Some(Self::MapAccumR),
            "unfoldr" => Some(Self::Unfoldr),
            "genericLength" => Some(Self::GenericLength),
            "genericTake" => Some(Self::GenericTake),
            "genericDrop" => Some(Self::GenericDrop),
            // Data.Char
            "isAlpha" => Some(Self::IsAlpha),
            "isAlphaNum" => Some(Self::IsAlphaNum),
            "isAscii" => Some(Self::IsAscii),
            "isControl" => Some(Self::IsControl),
            "isDigit" => Some(Self::IsDigit),
            "isHexDigit" => Some(Self::IsHexDigit),
            "isLetter" => Some(Self::IsLetter),
            "isLower" => Some(Self::IsLower),
            "isNumber" => Some(Self::IsNumber),
            "isPrint" => Some(Self::IsPrint),
            "isPunctuation" => Some(Self::IsPunctuation),
            "isSpace" => Some(Self::IsSpace),
            "isSymbol" => Some(Self::IsSymbol),
            "isUpper" => Some(Self::IsUpper),
            "toLower" => Some(Self::ToLower),
            "toUpper" => Some(Self::ToUpper),
            "toTitle" => Some(Self::ToTitle),
            "digitToInt" => Some(Self::DigitToInt),
            "intToDigit" => Some(Self::IntToDigit),
            "isLatin1" => Some(Self::IsLatin1),
            "isAsciiLower" => Some(Self::IsAsciiLower),
            "isAsciiUpper" => Some(Self::IsAsciiUpper),
            // Data.Function
            "on" => Some(Self::On),
            "fix" => Some(Self::Fix),
            "&" => Some(Self::Amp),
            // Data.Maybe additional
            "listToMaybe" => Some(Self::ListToMaybe),
            "maybeToList" => Some(Self::MaybeToList),
            "catMaybes" => Some(Self::CatMaybes),
            "mapMaybe" => Some(Self::MapMaybe),
            // Data.Either additional
            "isLeft" => Some(Self::IsLeft),
            "isRight" => Some(Self::IsRight),
            "lefts" => Some(Self::Lefts),
            "rights" => Some(Self::Rights),
            "partitionEithers" => Some(Self::PartitionEithers),
            // Math functions
            "sqrt" => Some(Self::Sqrt),
            "exp" => Some(Self::Exp),
            "log" => Some(Self::Log),
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "tan" => Some(Self::Tan),
            "^" => Some(Self::Power),
            "truncate" => Some(Self::Truncate),
            "round" => Some(Self::Round),
            "ceiling" => Some(Self::Ceiling),
            "floor" => Some(Self::Floor),
            // Tuple
            "fst" => Some(Self::Fst),
            "snd" => Some(Self::Snd),
            // Data.Map (qualified)
            "Data.Map.empty" | "Data.Map.Strict.empty" | "Map.empty" => Some(Self::MapEmpty),
            "Data.Map.singleton" | "Data.Map.Strict.singleton" | "Map.singleton" => Some(Self::MapSingleton),
            "Data.Map.null" | "Data.Map.Strict.null" | "Map.null" => Some(Self::MapNull),
            "Data.Map.size" | "Data.Map.Strict.size" | "Map.size" => Some(Self::MapSize),
            "Data.Map.member" | "Data.Map.Strict.member" | "Map.member" => Some(Self::MapMember),
            "Data.Map.notMember" | "Data.Map.Strict.notMember" | "Map.notMember" => Some(Self::MapNotMember),
            "Data.Map.lookup" | "Data.Map.Strict.lookup" | "Map.lookup" => Some(Self::MapLookup),
            "Data.Map.findWithDefault" | "Map.findWithDefault" => Some(Self::MapFindWithDefault),
            "Data.Map.!" | "Map.!" => Some(Self::MapIndex),
            "Data.Map.insert" | "Data.Map.Strict.insert" | "Map.insert" => Some(Self::MapInsert),
            "Data.Map.insertWith" | "Map.insertWith" => Some(Self::MapInsertWith),
            "Data.Map.delete" | "Data.Map.Strict.delete" | "Map.delete" => Some(Self::MapDelete),
            "Data.Map.adjust" | "Map.adjust" => Some(Self::MapAdjust),
            "Data.Map.update" | "Map.update" => Some(Self::MapUpdate),
            "Data.Map.alter" | "Map.alter" => Some(Self::MapAlter),
            "Data.Map.union" | "Data.Map.Strict.union" | "Map.union" => Some(Self::MapUnion),
            "Data.Map.unionWith" | "Map.unionWith" => Some(Self::MapUnionWith),
            "Data.Map.unionWithKey" | "Map.unionWithKey" => Some(Self::MapUnionWithKey),
            "Data.Map.unions" | "Map.unions" => Some(Self::MapUnions),
            "Data.Map.intersection" | "Map.intersection" => Some(Self::MapIntersection),
            "Data.Map.intersectionWith" | "Map.intersectionWith" => Some(Self::MapIntersectionWith),
            "Data.Map.difference" | "Map.difference" => Some(Self::MapDifference),
            "Data.Map.differenceWith" | "Map.differenceWith" => Some(Self::MapDifferenceWith),
            "Data.Map.map" | "Data.Map.Strict.map" | "Map.map" => Some(Self::MapMap),
            "Data.Map.mapWithKey" | "Map.mapWithKey" => Some(Self::MapMapWithKey),
            "Data.Map.mapKeys" | "Map.mapKeys" => Some(Self::MapMapKeys),
            "Data.Map.filter" | "Map.filter" => Some(Self::MapFilter),
            "Data.Map.filterWithKey" | "Map.filterWithKey" => Some(Self::MapFilterWithKey),
            "Data.Map.foldr" | "Map.foldr" => Some(Self::MapFoldr),
            "Data.Map.foldl" | "Map.foldl" => Some(Self::MapFoldl),
            "Data.Map.foldrWithKey" | "Map.foldrWithKey" => Some(Self::MapFoldrWithKey),
            "Data.Map.foldlWithKey" | "Map.foldlWithKey" => Some(Self::MapFoldlWithKey),
            "Data.Map.keys" | "Map.keys" => Some(Self::MapKeys),
            "Data.Map.elems" | "Map.elems" => Some(Self::MapElems),
            "Data.Map.assocs" | "Map.assocs" => Some(Self::MapAssocs),
            "Data.Map.toList" | "Map.toList" => Some(Self::MapToList),
            "Data.Map.fromList" | "Data.Map.Strict.fromList" | "Map.fromList" => Some(Self::MapFromList),
            "Data.Map.fromListWith" | "Map.fromListWith" => Some(Self::MapFromListWith),
            "Data.Map.toAscList" | "Map.toAscList" => Some(Self::MapToAscList),
            "Data.Map.toDescList" | "Map.toDescList" => Some(Self::MapToDescList),
            "Data.Map.isSubmapOf" | "Map.isSubmapOf" => Some(Self::MapIsSubmapOf),
            // Data.Set (qualified)
            "Data.Set.empty" | "Set.empty" => Some(Self::SetEmpty),
            "Data.Set.singleton" | "Set.singleton" => Some(Self::SetSingleton),
            "Data.Set.null" | "Set.null" => Some(Self::SetNull),
            "Data.Set.size" | "Set.size" => Some(Self::SetSize),
            "Data.Set.member" | "Set.member" => Some(Self::SetMember),
            "Data.Set.notMember" | "Set.notMember" => Some(Self::SetNotMember),
            "Data.Set.insert" | "Set.insert" => Some(Self::SetInsert),
            "Data.Set.delete" | "Set.delete" => Some(Self::SetDelete),
            "Data.Set.union" | "Set.union" => Some(Self::SetUnion),
            "Data.Set.unions" | "Set.unions" => Some(Self::SetUnions),
            "Data.Set.intersection" | "Set.intersection" => Some(Self::SetIntersection),
            "Data.Set.difference" | "Set.difference" => Some(Self::SetDifference),
            "Data.Set.isSubsetOf" | "Set.isSubsetOf" => Some(Self::SetIsSubsetOf),
            "Data.Set.isProperSubsetOf" | "Set.isProperSubsetOf" => Some(Self::SetIsProperSubsetOf),
            "Data.Set.map" | "Set.map" => Some(Self::SetMap),
            "Data.Set.filter" | "Set.filter" => Some(Self::SetFilter),
            "Data.Set.partition" | "Set.partition" => Some(Self::SetPartition),
            "Data.Set.foldr" | "Set.foldr" => Some(Self::SetFoldr),
            "Data.Set.foldl" | "Set.foldl" => Some(Self::SetFoldl),
            "Data.Set.toList" | "Set.toList" => Some(Self::SetToList),
            "Data.Set.fromList" | "Set.fromList" => Some(Self::SetFromList),
            "Data.Set.toAscList" | "Set.toAscList" => Some(Self::SetToAscList),
            "Data.Set.toDescList" | "Set.toDescList" => Some(Self::SetToDescList),
            "Data.Set.findMin" | "Set.findMin" => Some(Self::SetFindMin),
            "Data.Set.findMax" | "Set.findMax" => Some(Self::SetFindMax),
            "Data.Set.deleteMin" | "Set.deleteMin" => Some(Self::SetDeleteMin),
            "Data.Set.deleteMax" | "Set.deleteMax" => Some(Self::SetDeleteMax),
            "Data.Set.elems" | "Set.elems" => Some(Self::SetElems),
            "Data.Set.lookupMin" | "Set.lookupMin" => Some(Self::SetLookupMin),
            "Data.Set.lookupMax" | "Set.lookupMax" => Some(Self::SetLookupMax),
            // Data.IntMap (qualified)
            "Data.IntMap.empty" | "Data.IntMap.Strict.empty" | "IntMap.empty" => Some(Self::IntMapEmpty),
            "Data.IntMap.singleton" | "IntMap.singleton" => Some(Self::IntMapSingleton),
            "Data.IntMap.null" | "IntMap.null" => Some(Self::IntMapNull),
            "Data.IntMap.size" | "IntMap.size" => Some(Self::IntMapSize),
            "Data.IntMap.member" | "IntMap.member" => Some(Self::IntMapMember),
            "Data.IntMap.lookup" | "IntMap.lookup" => Some(Self::IntMapLookup),
            "Data.IntMap.findWithDefault" | "IntMap.findWithDefault" => Some(Self::IntMapFindWithDefault),
            "Data.IntMap.insert" | "IntMap.insert" => Some(Self::IntMapInsert),
            "Data.IntMap.insertWith" | "IntMap.insertWith" => Some(Self::IntMapInsertWith),
            "Data.IntMap.delete" | "IntMap.delete" => Some(Self::IntMapDelete),
            "Data.IntMap.adjust" | "IntMap.adjust" => Some(Self::IntMapAdjust),
            "Data.IntMap.union" | "IntMap.union" => Some(Self::IntMapUnion),
            "Data.IntMap.unionWith" | "IntMap.unionWith" => Some(Self::IntMapUnionWith),
            "Data.IntMap.intersection" | "IntMap.intersection" => Some(Self::IntMapIntersection),
            "Data.IntMap.difference" | "IntMap.difference" => Some(Self::IntMapDifference),
            "Data.IntMap.map" | "IntMap.map" => Some(Self::IntMapMap),
            "Data.IntMap.mapWithKey" | "IntMap.mapWithKey" => Some(Self::IntMapMapWithKey),
            "Data.IntMap.filter" | "IntMap.filter" => Some(Self::IntMapFilter),
            "Data.IntMap.foldr" | "IntMap.foldr" => Some(Self::IntMapFoldr),
            "Data.IntMap.foldlWithKey" | "IntMap.foldlWithKey" => Some(Self::IntMapFoldlWithKey),
            "Data.IntMap.keys" | "IntMap.keys" => Some(Self::IntMapKeys),
            "Data.IntMap.elems" | "IntMap.elems" => Some(Self::IntMapElems),
            "Data.IntMap.toList" | "IntMap.toList" => Some(Self::IntMapToList),
            "Data.IntMap.fromList" | "IntMap.fromList" => Some(Self::IntMapFromList),
            "Data.IntMap.toAscList" | "IntMap.toAscList" => Some(Self::IntMapToAscList),
            // Data.IntSet (qualified)
            "Data.IntSet.empty" | "IntSet.empty" => Some(Self::IntSetEmpty),
            "Data.IntSet.singleton" | "IntSet.singleton" => Some(Self::IntSetSingleton),
            "Data.IntSet.null" | "IntSet.null" => Some(Self::IntSetNull),
            "Data.IntSet.size" | "IntSet.size" => Some(Self::IntSetSize),
            "Data.IntSet.member" | "IntSet.member" => Some(Self::IntSetMember),
            "Data.IntSet.insert" | "IntSet.insert" => Some(Self::IntSetInsert),
            "Data.IntSet.delete" | "IntSet.delete" => Some(Self::IntSetDelete),
            "Data.IntSet.union" | "IntSet.union" => Some(Self::IntSetUnion),
            "Data.IntSet.intersection" | "IntSet.intersection" => Some(Self::IntSetIntersection),
            "Data.IntSet.difference" | "IntSet.difference" => Some(Self::IntSetDifference),
            "Data.IntSet.isSubsetOf" | "IntSet.isSubsetOf" => Some(Self::IntSetIsSubsetOf),
            "Data.IntSet.filter" | "IntSet.filter" => Some(Self::IntSetFilter),
            "Data.IntSet.foldr" | "IntSet.foldr" => Some(Self::IntSetFoldr),
            "Data.IntSet.toList" | "IntSet.toList" => Some(Self::IntSetToList),
            "Data.IntSet.fromList" | "IntSet.fromList" => Some(Self::IntSetFromList),
            // Control.Monad
            "when" | "Control.Monad.when" => Some(Self::MonadWhen),
            "unless" | "Control.Monad.unless" => Some(Self::MonadUnless),
            "guard" | "Control.Monad.guard" => Some(Self::MonadGuard),
            "void" | "Control.Monad.void" | "Data.Functor.void" => Some(Self::MonadVoid),
            "join" | "Control.Monad.join" => Some(Self::MonadJoin),
            "ap" | "Control.Monad.ap" => Some(Self::MonadAp),
            "liftM" | "Control.Monad.liftM" => Some(Self::LiftM),
            "liftM2" | "Control.Monad.liftM2" => Some(Self::LiftM2),
            "liftM3" | "Control.Monad.liftM3" => Some(Self::LiftM3),
            "liftM4" | "Control.Monad.liftM4" => Some(Self::LiftM4),
            "liftM5" | "Control.Monad.liftM5" => Some(Self::LiftM5),
            "filterM" | "Control.Monad.filterM" => Some(Self::FilterM),
            "mapAndUnzipM" | "Control.Monad.mapAndUnzipM" => Some(Self::MapAndUnzipM),
            "zipWithM" | "Control.Monad.zipWithM" => Some(Self::ZipWithM),
            "zipWithM_" | "Control.Monad.zipWithM_" => Some(Self::ZipWithM_),
            "foldM" | "Control.Monad.foldM" => Some(Self::FoldM),
            "foldM_" | "Control.Monad.foldM_" => Some(Self::FoldM_),
            "replicateM" | "Control.Monad.replicateM" => Some(Self::ReplicateM),
            "replicateM_" | "Control.Monad.replicateM_" => Some(Self::ReplicateM_),
            "forever" | "Control.Monad.forever" => Some(Self::Forever),
            "mzero" | "Control.Monad.mzero" => Some(Self::Mzero),
            "mplus" | "Control.Monad.mplus" => Some(Self::Mplus),
            "msum" | "Control.Monad.msum" => Some(Self::Msum),
            "mfilter" | "Control.Monad.mfilter" => Some(Self::Mfilter),
            ">=>" | "Control.Monad.>=>" => Some(Self::KleisliCompose),
            "<=<" | "Control.Monad.<=<" => Some(Self::KleisliComposeFlip),
            // Control.Applicative
            "liftA" | "Control.Applicative.liftA" => Some(Self::LiftA),
            "liftA2" | "Control.Applicative.liftA2" => Some(Self::LiftA2),
            "liftA3" | "Control.Applicative.liftA3" => Some(Self::LiftA3),
            "optional" | "Control.Applicative.optional" => Some(Self::Optional),
            // Control.Exception
            "catch" | "Control.Exception.catch" => Some(Self::ExnCatch),
            "try" | "Control.Exception.try" => Some(Self::ExnTry),
            "throw" | "Control.Exception.throw" => Some(Self::ExnThrow),
            "throwIO" | "Control.Exception.throwIO" => Some(Self::ExnThrowIO),
            "bracket" | "Control.Exception.bracket" => Some(Self::ExnBracket),
            "bracket_" | "Control.Exception.bracket_" => Some(Self::ExnBracket_),
            "bracketOnError" | "Control.Exception.bracketOnError" => Some(Self::ExnBracketOnError),
            "finally" | "Control.Exception.finally" => Some(Self::ExnFinally),
            "onException" | "Control.Exception.onException" => Some(Self::ExnOnException),
            "handle" | "Control.Exception.handle" => Some(Self::ExnHandle),
            "handleJust" | "Control.Exception.handleJust" => Some(Self::ExnHandleJust),
            "evaluate" | "Control.Exception.evaluate" => Some(Self::ExnEvaluate),
            "mask" | "Control.Exception.mask" => Some(Self::ExnMask),
            "mask_" | "Control.Exception.mask_" => Some(Self::ExnMask_),
            "uninterruptibleMask" | "Control.Exception.uninterruptibleMask" => Some(Self::ExnUninterruptibleMask),
            "uninterruptibleMask_" | "Control.Exception.uninterruptibleMask_" => Some(Self::ExnUninterruptibleMask_),
            // Control.Concurrent
            "forkIO" | "Control.Concurrent.forkIO" => Some(Self::ForkIO),
            "threadDelay" | "Control.Concurrent.threadDelay" => Some(Self::ThreadDelay),
            "myThreadId" | "Control.Concurrent.myThreadId" => Some(Self::MyThreadId),
            "newMVar" | "Control.Concurrent.MVar.newMVar" => Some(Self::NewMVar),
            "newEmptyMVar" | "Control.Concurrent.MVar.newEmptyMVar" => Some(Self::NewEmptyMVar),
            "takeMVar" | "Control.Concurrent.MVar.takeMVar" => Some(Self::TakeMVar),
            "putMVar" | "Control.Concurrent.MVar.putMVar" => Some(Self::PutMVar),
            "readMVar" | "Control.Concurrent.MVar.readMVar" => Some(Self::ReadMVar),
            "throwTo" | "Control.Concurrent.throwTo" | "Control.Exception.throwTo" => Some(Self::ThrowTo),
            "killThread" | "Control.Concurrent.killThread" => Some(Self::KillThread),
            // Data.Ord
            "comparing" | "Data.Ord.comparing" => Some(Self::Comparing),
            "clamp" | "Data.Ord.clamp" => Some(Self::Clamp),
            // Data.Foldable
            "fold" | "Data.Foldable.fold" => Some(Self::Fold),
            "foldMap" | "Data.Foldable.foldMap" => Some(Self::FoldMap),
            "foldr'" | "Data.Foldable.foldr'" => Some(Self::FoldrStrict),
            "foldl1" | "Data.Foldable.foldl1" => Some(Self::Foldl1),
            "foldr1" | "Data.Foldable.foldr1" => Some(Self::Foldr1),
            "maximumBy" | "Data.Foldable.maximumBy" | "Data.List.maximumBy" => Some(Self::MaximumBy),
            "minimumBy" | "Data.Foldable.minimumBy" | "Data.List.minimumBy" => Some(Self::MinimumBy),
            "asum" | "Data.Foldable.asum" => Some(Self::Asum),
            "traverse_" | "Data.Foldable.traverse_" => Some(Self::Traverse_),
            "for_" | "Data.Foldable.for_" => Some(Self::For_),
            "sequenceA_" | "Data.Foldable.sequenceA_" => Some(Self::SequenceA_),
            // Data.Traversable
            "traverse" | "Data.Traversable.traverse" => Some(Self::Traverse),
            "sequenceA" | "Data.Traversable.sequenceA" => Some(Self::SequenceA),
            "Data.Traversable.mapAccumL" | "Data.List.mapAccumL" => Some(Self::MapAccumL),
            "Data.Traversable.mapAccumR" | "Data.List.mapAccumR" => Some(Self::MapAccumR),
            // Data.Monoid
            "mempty" | "Data.Monoid.mempty" => Some(Self::Mempty),
            "mappend" | "Data.Monoid.mappend" => Some(Self::Mappend),
            "mconcat" | "Data.Monoid.mconcat" => Some(Self::Mconcat),
            // Data.String
            "fromString" | "Data.String.fromString" => Some(Self::FromString),
            // Data.Bits
            ".&." | "Data.Bits..&." => Some(Self::BitAnd),
            ".|." | "Data.Bits..|." => Some(Self::BitOr),
            "xor" | "Data.Bits.xor" => Some(Self::BitXor),
            "complement" | "Data.Bits.complement" => Some(Self::BitComplement),
            "shift" | "Data.Bits.shift" => Some(Self::BitShift),
            "shiftL" | "Data.Bits.shiftL" => Some(Self::BitShiftL),
            "shiftR" | "Data.Bits.shiftR" => Some(Self::BitShiftR),
            "rotate" | "Data.Bits.rotate" => Some(Self::BitRotate),
            "rotateL" | "Data.Bits.rotateL" => Some(Self::BitRotateL),
            "rotateR" | "Data.Bits.rotateR" => Some(Self::BitRotateR),
            "bit" | "Data.Bits.bit" => Some(Self::BitBit),
            "setBit" | "Data.Bits.setBit" => Some(Self::BitSetBit),
            "clearBit" | "Data.Bits.clearBit" => Some(Self::BitClearBit),
            "complementBit" | "Data.Bits.complementBit" => Some(Self::BitComplementBit),
            "testBit" | "Data.Bits.testBit" => Some(Self::BitTestBit),
            "popCount" | "Data.Bits.popCount" => Some(Self::BitPopCount),
            "zeroBits" | "Data.Bits.zeroBits" => Some(Self::BitZeroBits),
            "countLeadingZeros" | "Data.Bits.countLeadingZeros" => Some(Self::BitCountLeadingZeros),
            "countTrailingZeros" | "Data.Bits.countTrailingZeros" => Some(Self::BitCountTrailingZeros),
            // Data.Proxy
            "asProxyTypeOf" | "Data.Proxy.asProxyTypeOf" => Some(Self::AsProxyTypeOf),
            // Data.Void
            "absurd" | "Data.Void.absurd" => Some(Self::Absurd),
            "vacuous" | "Data.Void.vacuous" => Some(Self::Vacuous),
            _ => None,
        }
    }
}
