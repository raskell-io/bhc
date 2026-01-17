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
//! ## See Also
//!
//! - `bhc-hir`: High-level IR that uses these types
//! - `bhc-core`: Core IR with explicit type annotations
//! - H26-SPEC Section 4: Type System Specification

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

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
}
