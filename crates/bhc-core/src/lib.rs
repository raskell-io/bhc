//! # BHC Core IR
//!
//! This crate defines the Core Intermediate Representation for the Basel
//! Haskell Compiler. Core IR is the typed, explicit, and optimizable
//! representation that serves as the main compilation target.
//!
//! ## Overview
//!
//! Core IR is inspired by GHC's Core language but adapted for BHC's needs.
//! It is:
//!
//! - **Typed**: Every expression carries its type explicitly
//! - **Explicit**: Type applications, coercions, and casts are visible
//! - **A-Normal Form**: Complex expressions are let-bound
//! - **Optimizable**: Designed for efficient transformation and analysis
//!
//! ## IR Pipeline Position
//!
//! ```text
//! Source Code
//!     |
//!     v
//! [Parse/AST]  <- Surface syntax
//!     |
//!     v
//! [HIR]        <- Desugared, resolved
//!     |
//!     v
//! [Core IR]    <- This crate: typed, explicit
//!     |
//!     | (for Numeric Profile)
//!     v
//! [Tensor IR]  <- Shape/stride aware
//!     |
//!     v
//! [Loop IR]    <- Iteration, vectorization
//! ```
//!
//! ## Design Principles
//!
//! 1. **Preserving semantics**: Transformations must preserve meaning
//! 2. **Type safety**: All expressions are well-typed by construction
//! 3. **Source tracking**: Maintain source locations for error reporting
//! 4. **Transformation friendly**: Easy to pattern-match and transform
//!
//! ## Main Types
//!
//! - [`Expr`]: Core IR expressions
//! - [`Bind`]: Let bindings (recursive and non-recursive)
//! - [`Alt`]: Case alternatives
//! - [`Var`]: Variables with their types
//!
//! ## See Also
//!
//! - `bhc-hir`: Source-level HIR that lowers to Core
//! - `bhc-tensor-ir`: Numeric IR for tensor operations
//! - `bhc-types`: Type system definitions
//! - H26-SPEC Section 5: Core IR Specification

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod escape;
pub mod eval;
pub mod pretty;
pub mod uarray;

use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Ty, TyCon, TyVar};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A unique identifier for Core IR expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExprId(u32);

impl Idx for ExprId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A variable in Core IR with its type.
///
/// In Core IR, every variable carries its type explicitly.
/// This enables type-safe transformations and optimizations.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Var {
    /// The variable name (may be compiler-generated).
    pub name: Symbol,
    /// The unique identifier for this variable.
    pub id: VarId,
    /// The type of this variable.
    pub ty: Ty,
}

impl Var {
    /// Creates a new variable with the given name, ID, and type.
    #[must_use]
    pub fn new(name: Symbol, id: VarId, ty: Ty) -> Self {
        Self { name, id, ty }
    }
}

/// A unique identifier for variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarId(u32);

impl Idx for VarId {
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Core IR expressions.
///
/// Core IR is a small, typed lambda calculus with:
/// - Explicit type abstractions and applications
/// - Let bindings (recursive and non-recursive)
/// - Case expressions with pattern matching
/// - Coercions for type-safe casts
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Expr {
    /// A variable reference.
    Var(Var, Span),

    /// A literal value.
    Lit(Literal, Ty, Span),

    /// Function application: `f x`.
    App(Box<Expr>, Box<Expr>, Span),

    /// Type application: `f @ty`.
    /// Applies a type argument to a polymorphic function.
    TyApp(Box<Expr>, Ty, Span),

    /// Lambda abstraction: `\x -> e`.
    Lam(Var, Box<Expr>, Span),

    /// Type abstraction: `/\a -> e`.
    /// Abstracts over a type variable.
    TyLam(TyVar, Box<Expr>, Span),

    /// Let binding: `let binds in e`.
    Let(Box<Bind>, Box<Expr>, Span),

    /// Case expression: `case e of { alts }`.
    Case(Box<Expr>, Vec<Alt>, Ty, Span),

    /// Lazy evaluation escape hatch: `lazy { e }`.
    ///
    /// In strict profiles (Numeric, Edge), this forces the inner expression
    /// to be evaluated lazily (wrapped in a thunk). This provides an
    /// escape hatch for code that genuinely needs lazy evaluation.
    ///
    /// See H26-SPEC Section 6.4 for the lazy escape hatch specification.
    Lazy(Box<Expr>, Span),

    /// Type cast with a coercion.
    Cast(Box<Expr>, Coercion, Span),

    /// A tick for profiling/debugging.
    Tick(Tick, Box<Expr>, Span),

    /// Type annotation (for documentation).
    Type(Ty, Span),

    /// Coercion value.
    Coercion(Coercion, Span),
}

impl Expr {
    /// Returns the source span of this expression.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Var(_, span)
            | Self::Lit(_, _, span)
            | Self::App(_, _, span)
            | Self::TyApp(_, _, span)
            | Self::Lam(_, _, span)
            | Self::TyLam(_, _, span)
            | Self::Let(_, _, span)
            | Self::Case(_, _, _, span)
            | Self::Lazy(_, span)
            | Self::Cast(_, _, span)
            | Self::Tick(_, _, span)
            | Self::Type(_, span)
            | Self::Coercion(_, span) => *span,
        }
    }

    /// Returns the type of this expression.
    ///
    /// This computes the type based on the expression structure.
    /// Since Core IR is explicitly typed, this is straightforward.
    #[must_use]
    pub fn ty(&self) -> Ty {
        match self {
            Self::Var(v, _) => v.ty.clone(),
            Self::Lit(_, ty, _) => ty.clone(),
            Self::App(f, _, _) => {
                // Result type of function application
                if let Ty::Fun(_, result) = f.ty() {
                    (*result).clone()
                } else {
                    Ty::Error
                }
            }
            Self::TyApp(f, ty_arg, _) => {
                // Instantiate the forall type
                if let Ty::Forall(vars, body) = f.ty() {
                    if let Some(var) = vars.first() {
                        // Substitute the first type variable
                        let mut subst = bhc_types::Subst::new();
                        subst.insert(var, ty_arg.clone());
                        subst.apply(&body)
                    } else {
                        (*body).clone()
                    }
                } else {
                    Ty::Error
                }
            }
            Self::Lam(x, body, _) => Ty::fun(x.ty.clone(), body.ty()),
            Self::TyLam(tv, body, _) => Ty::Forall(vec![tv.clone()], Box::new(body.ty())),
            Self::Let(_, body, _) => body.ty(),
            Self::Case(_, _, ty, _) => ty.clone(),
            Self::Lazy(e, _) => e.ty(),
            Self::Cast(_, coercion, _) => coercion.result_ty.clone(),
            Self::Tick(_, e, _) => e.ty(),
            Self::Type(_, _) => Ty::Error, // Types have kind, not type
            Self::Coercion(c, _) => c.result_ty.clone(),
        }
    }

    /// Returns true if this expression is a value (WHNF).
    #[must_use]
    pub fn is_value(&self) -> bool {
        match self {
            Self::Lit(_, _, _) | Self::Lam(_, _, _) | Self::TyLam(_, _, _) => true,
            // Lazy blocks create thunks, so they are values (a thunk is WHNF)
            Self::Lazy(_, _) => true,
            Self::Tick(_, e, _) => e.is_value(),
            Self::Cast(e, _, _) => e.is_value(),
            _ => false,
        }
    }

    /// Returns true if this expression is trivial (a variable or literal).
    #[must_use]
    pub fn is_trivial(&self) -> bool {
        matches!(
            self,
            Self::Var(_, _) | Self::Lit(_, _, _) | Self::Type(_, _)
        )
    }

    /// Returns the free variables in this expression.
    #[must_use]
    pub fn free_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        let mut bound = Vec::new();
        self.collect_free_vars(&mut vars, &mut bound);
        vars
    }

    fn collect_free_vars(&self, free: &mut Vec<Var>, bound: &mut Vec<VarId>) {
        match self {
            Self::Var(v, _) => {
                if !bound.contains(&v.id) && !free.iter().any(|fv| fv.id == v.id) {
                    free.push(v.clone());
                }
            }
            Self::Lit(_, _, _) | Self::Type(_, _) | Self::Coercion(_, _) => {}
            Self::App(f, a, _) => {
                f.collect_free_vars(free, bound);
                a.collect_free_vars(free, bound);
            }
            Self::TyApp(f, _, _) => f.collect_free_vars(free, bound),
            Self::Lam(x, body, _) => {
                bound.push(x.id);
                body.collect_free_vars(free, bound);
                bound.pop();
            }
            Self::TyLam(_, body, _) => body.collect_free_vars(free, bound),
            Self::Let(bind, body, _) => match bind.as_ref() {
                Bind::NonRec(x, rhs) => {
                    rhs.collect_free_vars(free, bound);
                    bound.push(x.id);
                    body.collect_free_vars(free, bound);
                    bound.pop();
                }
                Bind::Rec(bindings) => {
                    for (x, _) in bindings {
                        bound.push(x.id);
                    }
                    for (_, rhs) in bindings {
                        rhs.collect_free_vars(free, bound);
                    }
                    body.collect_free_vars(free, bound);
                    for _ in bindings {
                        bound.pop();
                    }
                }
            },
            Self::Case(scrut, alts, _, _) => {
                scrut.collect_free_vars(free, bound);
                for alt in alts {
                    for v in &alt.binders {
                        bound.push(v.id);
                    }
                    alt.rhs.collect_free_vars(free, bound);
                    for _ in &alt.binders {
                        bound.pop();
                    }
                }
            }
            Self::Cast(e, _, _) | Self::Tick(_, e, _) | Self::Lazy(e, _) => {
                e.collect_free_vars(free, bound);
            }
        }
    }
}

/// A literal value in Core IR.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    /// Machine integer (fixed width).
    Int(i64),
    /// Arbitrary precision integer.
    Integer(i128),
    /// Single-precision float.
    Float(f32),
    /// Double-precision float.
    Double(f64),
    /// A character.
    Char(char),
    /// A string (as a symbol for interning).
    String(Symbol),
}

impl Literal {
    /// Returns the Core type of this literal.
    #[must_use]
    pub fn core_type(&self) -> &'static str {
        match self {
            Self::Int(_) => "Int#",
            Self::Integer(_) => "Integer",
            Self::Float(_) => "Float#",
            Self::Double(_) => "Double#",
            Self::Char(_) => "Char#",
            Self::String(_) => "Addr#",
        }
    }
}

/// A binding group in a let expression.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Bind {
    /// A non-recursive binding: `let x = e`.
    NonRec(Var, Box<Expr>),
    /// Mutually recursive bindings: `let rec { x1 = e1; x2 = e2 }`.
    Rec(Vec<(Var, Box<Expr>)>),
}

impl Bind {
    /// Returns the variables bound by this binding.
    #[must_use]
    pub fn bound_vars(&self) -> SmallVec<[&Var; 4]> {
        match self {
            Self::NonRec(v, _) => smallvec::smallvec![v],
            Self::Rec(bindings) => bindings.iter().map(|(v, _)| v).collect(),
        }
    }

    /// Returns true if this is a recursive binding.
    #[must_use]
    pub fn is_recursive(&self) -> bool {
        matches!(self, Self::Rec(_))
    }
}

/// A case alternative.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Alt {
    /// The constructor or literal being matched.
    pub con: AltCon,
    /// Variables bound by the pattern.
    pub binders: Vec<Var>,
    /// The right-hand side expression.
    pub rhs: Expr,
}

/// The constructor in a case alternative.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AltCon {
    /// A data constructor.
    DataCon(DataCon),
    /// A literal pattern.
    Lit(Literal),
    /// The default case (matches anything).
    Default,
}

/// A data constructor reference.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DataCon {
    /// The constructor name.
    pub name: Symbol,
    /// The type constructor it belongs to.
    pub ty_con: TyCon,
    /// The tag (index) of this constructor.
    pub tag: u32,
    /// The arity (number of fields).
    pub arity: u32,
}

/// A type coercion for safe casting.
///
/// Coercions are proof terms that witness type equality.
/// They are used for:
/// - Newtype unwrapping/wrapping
/// - Type family reduction
/// - Axiom application
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Coercion {
    /// The source type.
    pub source_ty: Ty,
    /// The result type.
    pub result_ty: Ty,
    /// The kind of coercion.
    pub kind: CoercionKind,
}

/// The kind of coercion being applied.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CoercionKind {
    /// Reflexivity: `a ~ a`.
    Refl,
    /// Symmetry: if `a ~ b` then `b ~ a`.
    Sym(Box<Coercion>),
    /// Transitivity: if `a ~ b` and `b ~ c` then `a ~ c`.
    Trans(Box<Coercion>, Box<Coercion>),
    /// Newtype coercion.
    Newtype(Symbol),
    /// Axiom application.
    Axiom(Symbol, Vec<Ty>),
    /// Universal coercion (forall).
    Forall(TyVar, Box<Coercion>),
    /// Coercion for type application.
    App(Box<Coercion>, Box<Coercion>),
}

/// A tick for profiling or cost-center attribution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Tick {
    /// A profiling tick with a cost center name.
    Profiling(Symbol),
    /// A source note for debugging.
    SourceNote(Span),
}

/// A Core module containing definitions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoreModule {
    /// The module name.
    pub name: Symbol,
    /// The bindings in this module.
    pub bindings: Vec<Bind>,
    /// Foreign exports.
    pub exports: Vec<ForeignExport>,
}

/// A foreign export from Core.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForeignExport {
    /// The exported name.
    pub name: Symbol,
    /// The Core variable being exported.
    pub var: Var,
    /// The calling convention.
    pub convention: ForeignConv,
}

/// Foreign calling convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForeignConv {
    /// C calling convention.
    CCall,
    /// Standard call (Windows).
    StdCall,
}

/// Strictness information for bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Strictness {
    /// Lazy evaluation (default).
    Lazy,
    /// Strict evaluation (evaluated before use).
    Strict,
    /// Hyperstrict (evaluated to NF).
    Hyperstrict,
}

/// Occurrence information for optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OccInfo {
    /// How many times the variable is used.
    pub occ_count: OccCount,
    /// Whether it's used inside a lambda.
    pub inside_lam: bool,
    /// Whether it's used in a "one-shot" context.
    pub one_shot: bool,
}

/// Occurrence count for inlining decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OccCount {
    /// Never used (dead code).
    Dead,
    /// Used exactly once.
    Once,
    /// Used multiple times.
    Many,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_is_value() {
        let lit = Expr::Lit(Literal::Int(42), Ty::Error, Span::default());
        assert!(lit.is_value());

        // SAFETY: 0 is a valid symbol index for testing
        let var = Var::new(unsafe { Symbol::from_raw(0) }, VarId::new(0), Ty::Error);
        let var_expr = Expr::Var(var, Span::default());
        assert!(!var_expr.is_value());
    }

    #[test]
    fn test_expr_is_trivial() {
        let lit = Expr::Lit(Literal::Int(42), Ty::Error, Span::default());
        assert!(lit.is_trivial());

        // SAFETY: 0 is a valid symbol index for testing
        let var = Var::new(unsafe { Symbol::from_raw(0) }, VarId::new(0), Ty::Error);
        let var_expr = Expr::Var(var, Span::default());
        assert!(var_expr.is_trivial());
    }

    #[test]
    fn test_bind_bound_vars() {
        // SAFETY: 0 is a valid symbol index for testing
        let var = Var::new(unsafe { Symbol::from_raw(0) }, VarId::new(0), Ty::Error);
        let lit = Expr::Lit(Literal::Int(42), Ty::Error, Span::default());

        let non_rec = Bind::NonRec(var.clone(), Box::new(lit));
        let bound = non_rec.bound_vars();
        assert_eq!(bound.len(), 1);
        assert_eq!(bound[0].id, var.id);
    }
}
