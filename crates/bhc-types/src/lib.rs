//! # BHC Type System
//!
//! This crate implements the type system for the Basel Haskell Compiler (BHC),
//! including type representations, type inference, unification, and typeclass
//! resolution.
//!
//! ## Overview
//!
//! The BHC type system is based on Hindley-Milner type inference with extensions
//! for:
//! - Higher-kinded types
//! - Type classes with functional dependencies
//! - Type families
//! - GADTs (Generalized Algebraic Data Types)
//! - Rank-N polymorphism (limited)
//! - **M9 Dependent Types Preview**: Shape-indexed tensors with compile-time
//!   dimension checking
//!
//! ## Core Types
//!
//! The type system is built around several key types:
//!
//! - [`Ty`]: The main type representation
//! - [`TyVar`]: Type variables for polymorphism
//! - [`TyId`]: Unique type identifiers
//! - [`Kind`]: Kinds for higher-kinded types
//! - [`Scheme`]: Polymorphic type schemes
//! - [`TyNat`]: Type-level natural numbers (M9)
//! - [`TyList`]: Type-level lists for shapes (M9)
//!
//! ## Type Inference
//!
//! Type inference proceeds in several phases:
//!
//! 1. **Constraint generation**: Walk the AST and generate type constraints
//! 2. **Unification**: Solve constraints to find substitutions
//! 3. **Generalization**: Generalize types at let-bindings
//! 4. **Defaulting**: Apply defaulting rules for ambiguous types
//!
//! ## Shape-Indexed Tensors (M9)
//!
//! BHC supports shape-indexed tensor types for compile-time dimension checking:
//!
//! ```text
//! matmul :: Tensor '[m, k] Float -> Tensor '[k, n] Float -> Tensor '[m, n] Float
//! ```
//!
//! See [`nat`] and [`ty_list`] modules for type-level constructs.
//!
//! ## See Also
//!
//! - `bhc-hir`: High-level IR that uses these types
//! - `bhc-core`: Core IR with explicit type annotations
//! - H26-SPEC Section 4: Type System Specification
//! - H26-SPEC Section 7: Tensor Model

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod dyn_tensor;
pub mod nat;
pub mod ty_list;

pub use dyn_tensor::{dyn_tensor_of, dyn_tensor_tycon, shape_witness_of, shape_witness_tycon};
pub use nat::TyNat;
pub use ty_list::TyList;

use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use serde::{Deserialize, Serialize};

/// A unique identifier for types within the type system.
///
/// Type IDs are used to reference types in the type environment
/// and during type inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TyId(u32);

impl Idx for TyId {
    #[allow(clippy::cast_possible_truncation)]
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A type variable, used for polymorphic types.
///
/// Type variables can be either:
/// - **Unification variables**: Created during type inference, to be solved
/// - **Rigid variables**: Bound by forall quantifiers, cannot be unified
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TyVar {
    /// Unique identifier for this type variable.
    pub id: u32,
    /// The kind of this type variable.
    pub kind: Kind,
}

impl TyVar {
    /// Creates a new type variable with the given ID and kind.
    #[must_use]
    pub fn new(id: u32, kind: Kind) -> Self {
        Self { id, kind }
    }

    /// Creates a new type variable with kind `*` (Type).
    #[must_use]
    pub fn new_star(id: u32) -> Self {
        Self::new(id, Kind::Star)
    }
}

/// Kinds classify types, just as types classify values.
///
/// - `*` (Star): The kind of types that have values (e.g., `Int`, `Bool`)
/// - `* -> *`: The kind of type constructors (e.g., `Maybe`, `[]`)
/// - `Constraint`: The kind of type class constraints
/// - `Nat`: The kind of type-level natural numbers (M9)
/// - `List k`: The kind of type-level lists with element kind `k` (M9)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    /// `*` - The kind of proper types (types that have values).
    Star,
    /// `k1 -> k2` - The kind of type constructors.
    Arrow(Box<Kind>, Box<Kind>),
    /// `Constraint` - The kind of type class constraints.
    Constraint,
    /// A kind variable, for kind inference.
    Var(u32),

    // === M9 Dependent Types Preview ===
    /// `Nat` - The kind of type-level natural numbers.
    ///
    /// Used for tensor dimensions in shape-indexed types:
    /// ```text
    /// Tensor :: [Nat] -> * -> *
    /// ```
    Nat,

    /// `List k` - The kind of type-level lists with element kind `k`.
    ///
    /// Primarily used as `[Nat]` for tensor shapes:
    /// ```text
    /// '[1024, 768] :: [Nat]
    /// ```
    List(Box<Kind>),
}

impl Kind {
    /// Returns the kind `* -> *`.
    #[must_use]
    pub fn star_to_star() -> Self {
        Self::Arrow(Box::new(Self::Star), Box::new(Self::Star))
    }

    /// Returns true if this is a simple `*` kind.
    #[must_use]
    pub fn is_star(&self) -> bool {
        matches!(self, Self::Star)
    }

    /// Returns true if this is a `Nat` kind.
    #[must_use]
    pub fn is_nat(&self) -> bool {
        matches!(self, Self::Nat)
    }

    /// Returns the kind `[Nat]` for tensor shapes.
    #[must_use]
    pub fn nat_list() -> Self {
        Self::List(Box::new(Self::Nat))
    }

    /// Returns the kind `[Nat] -> * -> *` for the Tensor type constructor.
    #[must_use]
    pub fn tensor_kind() -> Self {
        // Tensor :: [Nat] -> * -> *
        Self::Arrow(Box::new(Self::nat_list()), Box::new(Self::star_to_star()))
    }

    /// Returns true if this is a list kind.
    #[must_use]
    pub fn is_list(&self) -> bool {
        matches!(self, Self::List(_))
    }

    /// Returns the element kind if this is a list kind.
    #[must_use]
    pub fn list_elem_kind(&self) -> Option<&Kind> {
        match self {
            Self::List(k) => Some(k),
            _ => None,
        }
    }
}

/// The main type representation in BHC.
///
/// Types are represented as an algebraic data type with various constructors
/// for different kinds of types in the Haskell type system.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ty {
    /// A type variable (e.g., `a` in `a -> a`).
    Var(TyVar),

    /// A type constructor (e.g., `Int`, `Maybe`, `[]`).
    Con(TyCon),

    /// An unboxed primitive type (e.g., `Int#`, `Double#`).
    /// Used in Numeric Profile for zero-overhead computation.
    Prim(PrimTy),

    /// Type application (e.g., `Maybe Int`, `Either String`).
    App(Box<Ty>, Box<Ty>),

    /// Function type (e.g., `Int -> Bool`).
    /// This is sugar for `App(App(TyCon("->"), a), b)`.
    Fun(Box<Ty>, Box<Ty>),

    /// Tuple type (e.g., `(Int, Bool, String)`).
    Tuple(Vec<Ty>),

    /// List type (e.g., `[Int]`).
    /// This is sugar for `App(TyCon("[]"), Int)`.
    List(Box<Ty>),

    /// A forall-quantified type (e.g., `forall a. a -> a`).
    Forall(Vec<TyVar>, Box<Ty>),

    /// An error type, used during type checking to allow recovery.
    Error,

    // === M9 Dependent Types Preview ===
    /// A type-level natural number (e.g., `1024` in `Tensor '[1024] Float`).
    ///
    /// Has kind `Nat`. Used for tensor dimensions.
    Nat(TyNat),

    /// A type-level list (e.g., `'[1024, 768]` in `Tensor '[1024, 768] Float`).
    ///
    /// Has kind `[Nat]` when used for tensor shapes.
    TyList(TyList),
}

impl Ty {
    /// Creates a unit type `()`.
    #[must_use]
    pub fn unit() -> Self {
        Self::Tuple(Vec::new())
    }

    /// Creates a function type `a -> b`.
    #[must_use]
    pub fn fun(from: Ty, to: Ty) -> Self {
        Self::Fun(Box::new(from), Box::new(to))
    }

    /// Creates a list type `[a]`.
    #[must_use]
    pub fn list(elem: Ty) -> Self {
        Self::List(Box::new(elem))
    }

    /// Creates an unboxed `Int#` type (64-bit signed integer).
    #[must_use]
    pub fn int_prim() -> Self {
        Self::Prim(PrimTy::I64)
    }

    /// Creates an unboxed `Double#` type (64-bit float).
    #[must_use]
    pub fn double_prim() -> Self {
        Self::Prim(PrimTy::F64)
    }

    /// Creates an unboxed `Float#` type (32-bit float).
    #[must_use]
    pub fn float_prim() -> Self {
        Self::Prim(PrimTy::F32)
    }

    /// Returns true if this is an unboxed primitive type.
    #[must_use]
    pub fn is_prim(&self) -> bool {
        matches!(self, Self::Prim(_))
    }

    /// Returns the primitive type if this is a Prim variant.
    #[must_use]
    pub fn as_prim(&self) -> Option<PrimTy> {
        match self {
            Self::Prim(p) => Some(*p),
            _ => None,
        }
    }

    /// Returns true if this is a function type.
    #[must_use]
    pub fn is_fun(&self) -> bool {
        matches!(self, Self::Fun(_, _))
    }

    /// Returns true if this is an error type.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error)
    }

    /// Returns the free type variables in this type.
    #[must_use]
    pub fn free_vars(&self) -> Vec<TyVar> {
        let mut vars = Vec::new();
        self.collect_free_vars(&mut vars);
        vars
    }

    fn collect_free_vars(&self, vars: &mut Vec<TyVar>) {
        match self {
            Self::Var(v) => {
                if !vars.contains(v) {
                    vars.push(v.clone());
                }
            }
            Self::Con(_) | Self::Prim(_) | Self::Error => {}
            Self::App(f, a) | Self::Fun(f, a) => {
                f.collect_free_vars(vars);
                a.collect_free_vars(vars);
            }
            Self::Tuple(tys) => {
                for ty in tys {
                    ty.collect_free_vars(vars);
                }
            }
            Self::List(elem) => elem.collect_free_vars(vars),
            Self::Forall(bound, body) => {
                let mut body_vars = Vec::new();
                body.collect_free_vars(&mut body_vars);
                for v in body_vars {
                    if !bound.contains(&v) && !vars.contains(&v) {
                        vars.push(v);
                    }
                }
            }
            // M9: Type-level naturals and lists
            Self::Nat(n) => {
                for v in n.free_vars() {
                    if !vars.contains(&v) {
                        vars.push(v);
                    }
                }
            }
            Self::TyList(l) => {
                for v in l.free_vars() {
                    if !vars.contains(&v) {
                        vars.push(v);
                    }
                }
            }
        }
    }

    /// Returns true if this type contains no unification variables.
    #[must_use]
    pub fn is_ground(&self) -> bool {
        match self {
            Self::Var(_) => false,
            Self::Con(_) | Self::Prim(_) | Self::Error => true,
            Self::App(f, a) | Self::Fun(f, a) => f.is_ground() && a.is_ground(),
            Self::Tuple(tys) => tys.iter().all(Ty::is_ground),
            Self::List(elem) => elem.is_ground(),
            Self::Forall(_, body) => body.is_ground(),
            Self::Nat(n) => n.is_ground(),
            Self::TyList(l) => l.is_ground(),
        }
    }

    /// Creates a type-level natural literal.
    #[must_use]
    pub fn nat_lit(n: u64) -> Self {
        Self::Nat(TyNat::lit(n))
    }

    /// Creates a shape type from dimension values.
    #[must_use]
    pub fn shape(dims: &[u64]) -> Self {
        Self::TyList(TyList::shape_from_dims(dims))
    }

    /// Returns true if this is a type-level natural.
    #[must_use]
    pub fn is_nat(&self) -> bool {
        matches!(self, Self::Nat(_))
    }

    /// Returns true if this is a type-level list.
    #[must_use]
    pub fn is_ty_list(&self) -> bool {
        matches!(self, Self::TyList(_))
    }

    /// Returns the type-level natural if this is a Nat variant.
    #[must_use]
    pub fn as_nat(&self) -> Option<&TyNat> {
        match self {
            Self::Nat(n) => Some(n),
            _ => None,
        }
    }

    /// Returns the type-level list if this is a TyList variant.
    #[must_use]
    pub fn as_ty_list(&self) -> Option<&TyList> {
        match self {
            Self::TyList(l) => Some(l),
            _ => None,
        }
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Var(v) => write!(f, "t{}", v.id),
            Self::Con(c) => write!(f, "{}", c.name.as_str()),
            Self::Prim(p) => write!(f, "{p}"),
            Self::App(fun, arg) => write!(f, "({fun} {arg})"),
            Self::Fun(from, to) => write!(f, "({from} -> {to})"),
            Self::Tuple(tys) => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ")")
            }
            Self::List(elem) => write!(f, "[{elem}]"),
            Self::Forall(vars, body) => {
                write!(f, "forall")?;
                for v in vars {
                    write!(f, " t{}", v.id)?;
                }
                write!(f, ". {body}")
            }
            Self::Error => write!(f, "<error>"),
            Self::Nat(n) => write!(f, "{n}"),
            Self::TyList(l) => write!(f, "{l}"),
        }
    }
}

/// A type constructor with its name and kind.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TyCon {
    /// The name of the type constructor.
    pub name: Symbol,
    /// The kind of the type constructor.
    pub kind: Kind,
}

impl TyCon {
    /// Creates a new type constructor with the given name and kind.
    #[must_use]
    pub fn new(name: Symbol, kind: Kind) -> Self {
        Self { name, kind }
    }
}

/// A polymorphic type scheme (forall-quantified type).
///
/// A scheme represents a type that may be instantiated with different
/// type arguments. For example, `forall a. a -> a` can be instantiated
/// as `Int -> Int` or `Bool -> Bool`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scheme {
    /// The bound type variables.
    pub vars: Vec<TyVar>,
    /// The constraints on the type variables.
    pub constraints: Vec<Constraint>,
    /// The underlying type.
    pub ty: Ty,
}

impl Scheme {
    /// Creates a monomorphic scheme (no quantified variables).
    #[must_use]
    pub fn mono(ty: Ty) -> Self {
        Self {
            vars: Vec::new(),
            constraints: Vec::new(),
            ty,
        }
    }

    /// Creates a polymorphic scheme with the given variables and type.
    #[must_use]
    pub fn poly(vars: Vec<TyVar>, ty: Ty) -> Self {
        Self {
            vars,
            constraints: Vec::new(),
            ty,
        }
    }

    /// Creates a qualified scheme with constraints.
    #[must_use]
    pub fn qualified(vars: Vec<TyVar>, constraints: Vec<Constraint>, ty: Ty) -> Self {
        Self {
            vars,
            constraints,
            ty,
        }
    }

    /// Returns true if this is a monomorphic scheme.
    #[must_use]
    pub fn is_mono(&self) -> bool {
        self.vars.is_empty()
    }
}

/// A type class constraint (e.g., `Eq a`, `Num a`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Constraint {
    /// The type class name.
    pub class: Symbol,
    /// The type arguments to the constraint.
    pub args: Vec<Ty>,
    /// Source location of the constraint.
    pub span: Span,
}

impl Constraint {
    /// Creates a new constraint with a single type argument.
    #[must_use]
    pub fn new(class: Symbol, ty: Ty, span: Span) -> Self {
        Self {
            class,
            args: vec![ty],
            span,
        }
    }

    /// Creates a new constraint with multiple type arguments.
    #[must_use]
    pub fn new_multi(class: Symbol, args: Vec<Ty>, span: Span) -> Self {
        Self { class, args, span }
    }
}

/// A substitution mapping type variables to types.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Subst {
    /// The mapping from type variable IDs to types.
    mapping: rustc_hash::FxHashMap<u32, Ty>,
}

impl Subst {
    /// Creates an empty substitution.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a mapping from a type variable to a type.
    pub fn insert(&mut self, var: &TyVar, ty: Ty) {
        self.mapping.insert(var.id, ty);
    }

    /// Looks up a type variable in the substitution.
    #[must_use]
    pub fn get(&self, var: &TyVar) -> Option<&Ty> {
        self.mapping.get(&var.id)
    }

    /// Checks if a type variable has a mapping in this substitution.
    #[must_use]
    pub fn contains(&self, var: &TyVar) -> bool {
        self.mapping.contains_key(&var.id)
    }

    /// Applies this substitution to a type.
    #[must_use]
    pub fn apply(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Var(v) => self.get(v).cloned().unwrap_or_else(|| ty.clone()),
            Ty::Con(_) | Ty::Prim(_) => ty.clone(),
            Ty::App(f, a) => Ty::App(Box::new(self.apply(f)), Box::new(self.apply(a))),
            Ty::Fun(from, to) => Ty::Fun(Box::new(self.apply(from)), Box::new(self.apply(to))),
            Ty::Tuple(tys) => Ty::Tuple(tys.iter().map(|t| self.apply(t)).collect()),
            Ty::List(elem) => Ty::List(Box::new(self.apply(elem))),
            Ty::Forall(vars, body) => {
                // Don't substitute bound variables
                let mut inner = self.clone();
                for v in vars {
                    inner.mapping.remove(&v.id);
                }
                Ty::Forall(vars.clone(), Box::new(inner.apply(body)))
            }
            Ty::Error => Ty::Error,
            // M9: Apply substitution to type-level naturals and lists
            Ty::Nat(n) => Ty::Nat(self.apply_nat(n)),
            Ty::TyList(l) => Ty::TyList(self.apply_ty_list(l)),
        }
    }

    /// Applies this substitution to a type-level natural.
    #[must_use]
    pub fn apply_nat(&self, n: &TyNat) -> TyNat {
        match n {
            TyNat::Lit(val) => TyNat::Lit(*val),
            TyNat::Var(v) => {
                // Check if this variable is mapped to a Nat type
                match self.get(v) {
                    Some(Ty::Nat(replacement)) => replacement.clone(),
                    Some(_) => n.clone(), // Type mismatch, keep original
                    None => n.clone(),
                }
            }
            TyNat::Add(a, b) => TyNat::add(self.apply_nat(a), self.apply_nat(b)),
            TyNat::Mul(a, b) => TyNat::mul(self.apply_nat(a), self.apply_nat(b)),
        }
    }

    /// Applies this substitution to a type-level list.
    #[must_use]
    pub fn apply_ty_list(&self, l: &TyList) -> TyList {
        match l {
            TyList::Nil => TyList::Nil,
            TyList::Cons(head, tail) => TyList::cons(self.apply(head), self.apply_ty_list(tail)),
            TyList::Var(v) => {
                // Check if this variable is mapped to a TyList type
                match self.get(v) {
                    Some(Ty::TyList(replacement)) => replacement.clone(),
                    Some(_) => l.clone(), // Type mismatch, keep original
                    None => l.clone(),
                }
            }
            TyList::Append(xs, ys) => {
                TyList::append(self.apply_ty_list(xs), self.apply_ty_list(ys))
            }
        }
    }

    /// Composes two substitutions: (self . other)
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for (k, v) in &other.mapping {
            result.mapping.insert(*k, self.apply(v));
        }
        for (k, v) in &self.mapping {
            result.mapping.entry(*k).or_insert_with(|| v.clone());
        }
        result
    }

    /// Returns true if this substitution is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }
}

/// Unboxed primitive types for the Numeric Profile.
///
/// These types represent machine-level primitives that are:
/// - Unboxed: No heap allocation, stored directly in registers/stack
/// - Strict: Always fully evaluated, no thunks
/// - Fixed-size: Known size at compile time
///
/// Used in Numeric Profile for zero-overhead numeric computation.
/// See H26-SPEC Section 6.2 for unboxed type requirements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimTy {
    /// 32-bit signed integer (Int32#).
    I32,
    /// 64-bit signed integer (Int64#, Int#).
    I64,
    /// 32-bit unsigned integer (Word32#).
    U32,
    /// 64-bit unsigned integer (Word64#, Word#).
    U64,
    /// 32-bit IEEE 754 float (Float#).
    F32,
    /// 64-bit IEEE 754 double (Double#).
    F64,
    /// 8-bit character/byte (Char#).
    Char,
    /// Machine-sized pointer (Addr#).
    Addr,
}

impl PrimTy {
    /// Returns the size in bytes of this primitive type.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::Addr => 8,
            Self::Char => 4, // Unicode code point
        }
    }

    /// Returns the alignment requirement in bytes.
    #[must_use]
    pub const fn alignment(self) -> usize {
        self.size_bytes()
    }

    /// Returns the GHC-style name for this primitive type.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::I32 => "Int32#",
            Self::I64 => "Int#",
            Self::U32 => "Word32#",
            Self::U64 => "Word#",
            Self::F32 => "Float#",
            Self::F64 => "Double#",
            Self::Char => "Char#",
            Self::Addr => "Addr#",
        }
    }

    /// Returns true if this is a signed integer type.
    #[must_use]
    pub const fn is_signed_int(self) -> bool {
        matches!(self, Self::I32 | Self::I64)
    }

    /// Returns true if this is an unsigned integer type.
    #[must_use]
    pub const fn is_unsigned_int(self) -> bool {
        matches!(self, Self::U32 | Self::U64)
    }

    /// Returns true if this is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Returns true if this is a numeric type.
    #[must_use]
    pub const fn is_numeric(self) -> bool {
        matches!(
            self,
            Self::I32 | Self::I64 | Self::U32 | Self::U64 | Self::F32 | Self::F64
        )
    }
}

impl std::fmt::Display for PrimTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Match a pattern type against a target type, returning a substitution.
///
/// Performs one-way pattern matching where type variables in the pattern
/// are bound to concrete types in the target. This enables polymorphic instance
/// matching like `instance Eq a => Eq [a]` to match `Eq [Int]` with `{a -> Int}`.
///
/// # Arguments
/// * `pattern` - The instance type pattern (may contain type variables)
/// * `target` - The concrete type to match against
///
/// # Returns
/// `Some(subst)` if the match succeeds, where `subst` maps type variables to types.
/// `None` if the types cannot be matched.
#[must_use]
pub fn types_match(pattern: &Ty, target: &Ty) -> Option<Subst> {
    let mut subst = Subst::new();
    if types_match_with_subst(pattern, target, &mut subst) {
        Some(subst)
    } else {
        None
    }
}

/// Helper function that accumulates substitutions during type matching.
///
/// Returns `true` if the pattern matches the target, accumulating bindings
/// from type variables to concrete types in `subst`.
pub fn types_match_with_subst(pattern: &Ty, target: &Ty, subst: &mut Subst) -> bool {
    match (pattern, target) {
        // Type variable in pattern: bind to target type
        (Ty::Var(v), _) => {
            if let Some(bound_ty) = subst.get(v) {
                // Must match the existing binding
                types_equal(bound_ty, target)
            } else {
                subst.insert(v, target.clone());
                true
            }
        }

        // Type constructors must match by name
        (Ty::Con(c1), Ty::Con(c2)) => c1.name == c2.name,

        // Primitive types must match exactly
        (Ty::Prim(p1), Ty::Prim(p2)) => p1 == p2,

        // Type applications: match both the function and argument
        (Ty::App(f1, a1), Ty::App(f2, a2)) => {
            types_match_with_subst(f1, f2, subst) && types_match_with_subst(a1, a2, subst)
        }

        // Function types: match argument and result types
        (Ty::Fun(a1, r1), Ty::Fun(a2, r2)) => {
            types_match_with_subst(a1, a2, subst) && types_match_with_subst(r1, r2, subst)
        }

        // Tuple types: must have same length and matching elements
        (Ty::Tuple(ts1), Ty::Tuple(ts2)) if ts1.len() == ts2.len() => ts1
            .iter()
            .zip(ts2.iter())
            .all(|(t1, t2)| types_match_with_subst(t1, t2, subst)),

        // List types: match element types
        (Ty::List(e1), Ty::List(e2)) => types_match_with_subst(e1, e2, subst),

        // Forall types: match bodies (simplified, no alpha-renaming)
        (Ty::Forall(_, body1), Ty::Forall(_, body2)) => types_match_with_subst(body1, body2, subst),

        // Type-level naturals
        (Ty::Nat(n1), Ty::Nat(n2)) => n1 == n2,

        // Type-level lists
        (Ty::TyList(l1), Ty::TyList(l2)) => l1 == l2,

        // Error types match anything (to avoid cascading errors)
        (Ty::Error, _) | (_, Ty::Error) => true,

        // All other combinations don't match
        _ => false,
    }
}

/// Match multiple pattern types against target types, returning combined substitution.
///
/// Both lists must have the same length. The resulting substitution combines
/// all bindings from matching each pair.
#[must_use]
pub fn types_match_multi(patterns: &[Ty], targets: &[Ty]) -> Option<Subst> {
    if patterns.len() != targets.len() {
        return None;
    }

    let mut subst = Subst::new();
    for (pattern, target) in patterns.iter().zip(targets.iter()) {
        if !types_match_with_subst(pattern, target, &mut subst) {
            return None;
        }
    }
    Some(subst)
}

/// Check if two types are structurally equal.
#[must_use]
pub fn types_equal(t1: &Ty, t2: &Ty) -> bool {
    match (t1, t2) {
        (Ty::Var(v1), Ty::Var(v2)) => v1.id == v2.id,
        (Ty::Con(c1), Ty::Con(c2)) => c1.name == c2.name,
        (Ty::Prim(p1), Ty::Prim(p2)) => p1 == p2,
        (Ty::App(f1, a1), Ty::App(f2, a2)) => types_equal(f1, f2) && types_equal(a1, a2),
        (Ty::Fun(a1, r1), Ty::Fun(a2, r2)) => types_equal(a1, a2) && types_equal(r1, r2),
        (Ty::Tuple(ts1), Ty::Tuple(ts2)) if ts1.len() == ts2.len() => ts1
            .iter()
            .zip(ts2.iter())
            .all(|(t1, t2)| types_equal(t1, t2)),
        (Ty::List(e1), Ty::List(e2)) => types_equal(e1, e2),
        (Ty::Error, Ty::Error) => true,
        _ => false,
    }
}

/// Errors that can occur during type operations.
#[derive(Clone, Debug, thiserror::Error, Serialize, Deserialize)]
pub enum TypeError {
    /// Unification failed between two types.
    #[error("type mismatch: expected {expected}, found {found}")]
    Mismatch {
        /// The expected type.
        expected: String,
        /// The found type.
        found: String,
        /// The location of the error.
        span: Span,
    },

    /// Occurs check failed (infinite type).
    #[error("infinite type: {var} occurs in {ty}")]
    OccursCheck {
        /// The type variable.
        var: String,
        /// The type containing the variable.
        ty: String,
        /// The location of the error.
        span: Span,
    },

    /// Unbound type variable.
    #[error("unbound type variable: {name}")]
    UnboundVar {
        /// The variable name.
        name: String,
        /// The location of the error.
        span: Span,
    },

    /// Kind mismatch.
    #[error("kind mismatch: expected {expected}, found {found}")]
    KindMismatch {
        /// The expected kind.
        expected: String,
        /// The found kind.
        found: String,
        /// The location of the error.
        span: Span,
    },

    /// Ambiguous type variable.
    #[error("ambiguous type variable: {var}")]
    Ambiguous {
        /// The ambiguous variable.
        var: String,
        /// The location of the error.
        span: Span,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ty_free_vars() {
        let a = TyVar::new_star(0);
        let b = TyVar::new_star(1);

        // `a -> b` has free vars {a, b}
        let ty = Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()));
        let vars = ty.free_vars();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&a));
        assert!(vars.contains(&b));
    }

    #[test]
    fn test_subst_apply() {
        let a = TyVar::new_star(0);
        // SAFETY: 0 is a valid symbol index for testing purposes
        let int_con = TyCon::new(unsafe { Symbol::from_raw(0) }, Kind::Star);

        let mut subst = Subst::new();
        subst.insert(&a, Ty::Con(int_con.clone()));

        let ty = Ty::fun(Ty::Var(a.clone()), Ty::Var(a));
        let result = subst.apply(&ty);

        match result {
            Ty::Fun(from, to) => {
                assert!(matches!(*from, Ty::Con(_)));
                assert!(matches!(*to, Ty::Con(_)));
            }
            _ => panic!("expected function type"),
        }
    }

    #[test]
    fn test_scheme_mono() {
        // SAFETY: 0 is a valid symbol index for testing purposes
        let int_con = TyCon::new(unsafe { Symbol::from_raw(0) }, Kind::Star);
        let scheme = Scheme::mono(Ty::Con(int_con));
        assert!(scheme.is_mono());
    }

    #[test]
    fn test_prim_ty_properties() {
        assert_eq!(PrimTy::I64.size_bytes(), 8);
        assert_eq!(PrimTy::I32.size_bytes(), 4);
        assert_eq!(PrimTy::F64.size_bytes(), 8);
        assert_eq!(PrimTy::F32.size_bytes(), 4);

        assert!(PrimTy::I64.is_signed_int());
        assert!(PrimTy::I32.is_signed_int());
        assert!(!PrimTy::U64.is_signed_int());

        assert!(PrimTy::F64.is_float());
        assert!(PrimTy::F32.is_float());
        assert!(!PrimTy::I64.is_float());

        assert!(PrimTy::I64.is_numeric());
        assert!(PrimTy::F64.is_numeric());
        assert!(!PrimTy::Char.is_numeric());
        assert!(!PrimTy::Addr.is_numeric());
    }

    #[test]
    fn test_prim_ty_names() {
        assert_eq!(PrimTy::I64.name(), "Int#");
        assert_eq!(PrimTy::F64.name(), "Double#");
        assert_eq!(PrimTy::F32.name(), "Float#");
        assert_eq!(PrimTy::Char.name(), "Char#");
    }

    #[test]
    fn test_ty_prim_constructors() {
        let int = Ty::int_prim();
        assert!(int.is_prim());
        assert_eq!(int.as_prim(), Some(PrimTy::I64));

        let double = Ty::double_prim();
        assert!(double.is_prim());
        assert_eq!(double.as_prim(), Some(PrimTy::F64));

        // Non-prim type
        let unit = Ty::unit();
        assert!(!unit.is_prim());
        assert_eq!(unit.as_prim(), None);
    }

    #[test]
    fn test_prim_ty_no_free_vars() {
        // Primitive types have no free type variables
        let prim = Ty::int_prim();
        assert!(prim.free_vars().is_empty());
    }

    #[test]
    fn test_subst_prim_unchanged() {
        // Substitution should leave primitive types unchanged
        let a = TyVar::new_star(0);
        let mut subst = Subst::new();
        subst.insert(&a, Ty::unit());

        let prim = Ty::int_prim();
        let result = subst.apply(&prim);
        assert_eq!(result, prim);
    }

    #[test]
    fn test_types_match_variable_binding() {
        let a = TyVar::new_star(0);
        let int_con = TyCon::new(Symbol::intern("Int"), Kind::Star);
        let int_ty = Ty::Con(int_con);

        let result = types_match(&Ty::Var(a.clone()), &int_ty);
        assert!(result.is_some());
        let subst = result.unwrap();
        assert_eq!(subst.apply(&Ty::Var(a)), int_ty);
    }

    #[test]
    fn test_types_match_constructor() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        assert!(types_match(&int_ty, &int_ty).is_some());
        assert!(types_match(&int_ty, &bool_ty).is_none());
    }

    #[test]
    fn test_types_match_application() {
        let a = TyVar::new_star(0);
        let list_con = Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::star_to_star()));
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));

        // Pattern: [] a, Target: [] Int
        let pattern = Ty::App(Box::new(list_con.clone()), Box::new(Ty::Var(a.clone())));
        let target = Ty::App(Box::new(list_con), Box::new(int_ty.clone()));

        let result = types_match(&pattern, &target);
        assert!(result.is_some());
        let subst = result.unwrap();
        assert_eq!(subst.apply(&Ty::Var(a)), int_ty);
    }

    #[test]
    fn test_types_match_multi_basic() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        assert!(types_match_multi(&[int_ty.clone()], &[int_ty.clone()]).is_some());
        assert!(types_match_multi(&[int_ty.clone()], &[bool_ty]).is_none());
        assert!(types_match_multi(&[], &[]).is_some());
        assert!(types_match_multi(&[int_ty.clone()], &[]).is_none());
    }
}
