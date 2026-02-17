//! # BHC High-Level Intermediate Representation (HIR)
//!
//! This crate defines the High-Level Intermediate Representation for the
//! Basel Haskell Compiler. HIR is produced after parsing and desugaring,
//! but before type checking and lowering to Core IR.
//!
//! ## Overview
//!
//! HIR serves as a bridge between the surface syntax (AST) and the typed
//! Core IR. It preserves enough structure for:
//!
//! - Name resolution results
//! - Pattern matching structure
//! - Module organization
//! - Source location tracking for error messages
//!
//! ## IR Pipeline Position
//!
//! ```text
//! Source Code
//!     |
//!     v
//! [Parse/AST]  <- Surface syntax, concrete
//!     |
//!     v
//! [HIR]        <- This crate: desugared, resolved
//!     |
//!     v
//! [Core IR]    <- Typed, explicit, optimizable
//!     |
//!     v
//! [Tensor IR]  <- Numeric optimizations
//!     |
//!     v
//! [Loop IR]    <- Iteration, vectorization
//! ```
//!
//! ## Key Features
//!
//! - **Desugared syntax**: Do-notation, list comprehensions, etc. are expanded
//! - **Resolved names**: All identifiers are resolved to their definitions
//! - **Pattern matching**: Complex patterns preserved for exhaustiveness checking
//! - **Type annotations**: Type signatures attached but not yet checked
//!
//! ## Main Types
//!
//! - [`Expr`]: HIR expressions
//! - [`Pat`]: HIR patterns
//! - [`Item`]: Top-level declarations
//! - [`HirId`]: Unique identifiers for HIR nodes
//!
//! ## See Also
//!
//! - `bhc-ast`: Surface syntax AST
//! - `bhc-core`: Typed Core IR
//! - `bhc-types`: Type representations used here

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Scheme, Ty, TyVar};
use serde::{Deserialize, Serialize};

/// A unique identifier for HIR nodes.
///
/// Every expression, pattern, and item in HIR has a unique `HirId`
/// that can be used to attach metadata (types, diagnostics, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirId(u32);

impl Idx for HirId {
    #[allow(clippy::cast_possible_truncation)]
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A reference to a definition (variable, function, constructor, etc.).
///
/// After name resolution, all identifiers are replaced with `DefRef`s
/// that point to their definitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DefRef {
    /// The unique ID of the definition.
    pub def_id: DefId,
    /// The source span of this reference.
    pub span: Span,
}

/// A unique identifier for definitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DefId(u32);

impl Idx for DefId {
    #[allow(clippy::cast_possible_truncation)]
    fn new(idx: usize) -> Self {
        Self(idx as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// HIR expressions.
///
/// Expressions in HIR are desugared from the surface syntax but not yet
/// typed. Complex syntax like do-notation has been expanded into simpler
/// constructs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Expr {
    /// A literal value (integer, float, char, string).
    Lit(Lit, Span),

    /// A variable reference.
    Var(DefRef),

    /// A data constructor.
    Con(DefRef),

    /// Function application: `f x`.
    App(Box<Expr>, Box<Expr>, Span),

    /// Lambda expression: `\x -> e`.
    Lam(Vec<Pat>, Box<Expr>, Span),

    /// Let binding: `let x = e1 in e2`.
    Let(Vec<Binding>, Box<Expr>, Span),

    /// Case expression: `case e of { alts }`.
    Case(Box<Expr>, Vec<CaseAlt>, Span),

    /// If expression: `if c then t else e`.
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),

    /// Tuple expression: `(a, b, c)`.
    Tuple(Vec<Expr>, Span),

    /// List expression: `[a, b, c]`.
    List(Vec<Expr>, Span),

    /// Record construction: `MkRecord { field = value }`.
    Record(DefRef, Vec<FieldExpr>, Span),

    /// Record field access: `r.field`.
    FieldAccess(Box<Expr>, Symbol, Span),

    /// Record update: `r { field = value }`.
    RecordUpdate(Box<Expr>, Vec<FieldExpr>, Span),

    /// Type annotation: `e :: ty`.
    Ann(Box<Expr>, Ty, Span),

    /// Explicit type application: `e @ty`.
    TypeApp(Box<Expr>, Ty, Span),

    /// An error placeholder for error recovery.
    Error(Span),
}

impl Expr {
    /// Returns the source span of this expression.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Lit(_, span)
            | Self::Var(DefRef { span, .. })
            | Self::Con(DefRef { span, .. })
            | Self::App(_, _, span)
            | Self::Lam(_, _, span)
            | Self::Let(_, _, span)
            | Self::Case(_, _, span)
            | Self::If(_, _, _, span)
            | Self::Tuple(_, span)
            | Self::List(_, span)
            | Self::Record(_, _, span)
            | Self::FieldAccess(_, _, span)
            | Self::RecordUpdate(_, _, span)
            | Self::Ann(_, _, span)
            | Self::TypeApp(_, _, span)
            | Self::Error(span) => *span,
        }
    }

    /// Returns true if this is an error expression.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

/// A literal value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Lit {
    /// Integer literal (arbitrary precision).
    Int(i128),
    /// Floating-point literal.
    Float(f64),
    /// Character literal.
    Char(char),
    /// String literal.
    String(Symbol),
}

/// A record field expression.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldExpr {
    /// The field name.
    pub name: Symbol,
    /// The field value.
    pub value: Expr,
    /// Source span.
    pub span: Span,
}

/// A record field pattern.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldPat {
    /// The field name.
    pub name: Symbol,
    /// The pattern for this field.
    pub pat: Pat,
    /// Source span.
    pub span: Span,
}

/// A let binding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Binding {
    /// The pattern being bound.
    pub pat: Pat,
    /// The type signature, if present.
    pub sig: Option<Scheme>,
    /// The right-hand side expression.
    pub rhs: Expr,
    /// Source span.
    pub span: Span,
}

/// A case alternative.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseAlt {
    /// The pattern to match.
    pub pat: Pat,
    /// Optional guard expressions.
    pub guards: Vec<Guard>,
    /// The result expression.
    pub rhs: Expr,
    /// Source span.
    pub span: Span,
}

/// A pattern guard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Guard {
    /// The guard condition.
    pub cond: Expr,
    /// Source span.
    pub span: Span,
}

/// HIR patterns.
///
/// Patterns are used in lambda arguments, let bindings, and case alternatives.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Pat {
    /// Wildcard pattern: `_`.
    Wild(Span),

    /// Variable pattern: `x`.
    /// The DefId is the definition ID assigned during lowering.
    Var(Symbol, DefId, Span),

    /// Literal pattern: `42`, `'a'`.
    Lit(Lit, Span),

    /// Constructor pattern: `Just x`, `(a, b)`.
    Con(DefRef, Vec<Pat>, Span),

    /// Record constructor pattern: `XConfig { modMask = m, borderWidth = b }`.
    /// Contains the constructor reference and named field patterns.
    /// Unlike positional `Con`, this matches fields by name, not position.
    RecordCon(DefRef, Vec<FieldPat>, Span),

    /// As-pattern: `x@pat`.
    /// The DefId is the definition ID assigned during lowering for the bound variable.
    As(Symbol, DefId, Box<Pat>, Span),

    /// Or-pattern: `pat1 | pat2` (for view patterns).
    Or(Box<Pat>, Box<Pat>, Span),

    /// Type-annotated pattern: `pat :: ty`.
    Ann(Box<Pat>, Ty, Span),

    /// View pattern: `(expr -> pat)`.
    /// The expression is applied to the scrutinee, and the result is matched against the pattern.
    View(Box<Expr>, Box<Pat>, Span),

    /// Error pattern for recovery.
    Error(Span),
}

impl Pat {
    /// Returns the source span of this pattern.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Wild(span)
            | Self::Var(_, _, span)
            | Self::Lit(_, span)
            | Self::Con(_, _, span)
            | Self::RecordCon(_, _, span)
            | Self::As(_, _, _, span)
            | Self::Or(_, _, span)
            | Self::Ann(_, _, span)
            | Self::View(_, _, span)
            | Self::Error(span) => *span,
        }
    }

    /// Returns the variables bound by this pattern.
    #[must_use]
    pub fn bound_vars(&self) -> Vec<Symbol> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut Vec<Symbol>) {
        match self {
            Self::Wild(_) | Self::Lit(_, _) | Self::Error(_) => {}
            Self::Var(name, _, _) => vars.push(*name),
            Self::Con(_, pats, _) => {
                for p in pats {
                    p.collect_vars(vars);
                }
            }
            Self::RecordCon(_, field_pats, _) => {
                for fp in field_pats {
                    fp.pat.collect_vars(vars);
                }
            }
            Self::As(name, _, inner, _) => {
                vars.push(*name);
                inner.collect_vars(vars);
            }
            Self::Or(left, right, _) => {
                left.collect_vars(vars);
                // Note: both branches must bind the same variables
                right.collect_vars(vars);
            }
            Self::Ann(inner, _, _) | Self::View(_, inner, _) => inner.collect_vars(vars),
        }
    }
}

/// A top-level item in a module.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Item {
    /// A function or value definition.
    Value(ValueDef),

    /// A data type definition.
    Data(DataDef),

    /// A newtype definition.
    Newtype(NewtypeDef),

    /// A type synonym.
    TypeAlias(TypeAlias),

    /// A type class definition.
    Class(ClassDef),

    /// A type class instance.
    Instance(InstanceDef),

    /// A fixity declaration.
    Fixity(FixityDecl),

    /// A foreign import.
    Foreign(ForeignDecl),
}

/// A value (function) definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValueDef {
    /// The unique ID of this definition.
    pub id: DefId,
    /// The name of the value.
    pub name: Symbol,
    /// The type signature, if present.
    pub sig: Option<Scheme>,
    /// The defining equations.
    pub equations: Vec<Equation>,
    /// Source span.
    pub span: Span,
}

/// A single equation in a function definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Equation {
    /// The patterns for function arguments.
    pub pats: Vec<Pat>,
    /// Optional guards.
    pub guards: Vec<Guard>,
    /// The right-hand side.
    pub rhs: Expr,
    /// Source span.
    pub span: Span,
}

/// A data type definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataDef {
    /// The unique ID of this definition.
    pub id: DefId,
    /// The name of the data type.
    pub name: Symbol,
    /// Type parameters.
    pub params: Vec<TyVar>,
    /// The data constructors.
    pub cons: Vec<ConDef>,
    /// Derived instances.
    pub deriving: Vec<Symbol>,
    /// Source span.
    pub span: Span,
}

/// A data constructor definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConDef {
    /// The unique ID of this constructor.
    pub id: DefId,
    /// The name of the constructor.
    pub name: Symbol,
    /// The field types.
    pub fields: ConFields,
    /// Source span.
    pub span: Span,
}

/// Constructor fields (positional or named).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConFields {
    /// Positional fields: `MkFoo Int String`.
    Positional(Vec<Ty>),
    /// Named fields (record syntax): `MkFoo { x :: Int, y :: String }`.
    Named(Vec<FieldDef>),
}

/// A record field definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldDef {
    /// The unique ID for the field accessor function.
    pub id: DefId,
    /// The field name.
    pub name: Symbol,
    /// The field type.
    pub ty: Ty,
    /// Source span.
    pub span: Span,
}

/// A newtype definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewtypeDef {
    /// The unique ID of this definition.
    pub id: DefId,
    /// The name of the newtype.
    pub name: Symbol,
    /// Type parameters.
    pub params: Vec<TyVar>,
    /// The single constructor.
    pub con: ConDef,
    /// Derived instances.
    pub deriving: Vec<Symbol>,
    /// Source span.
    pub span: Span,
}

/// A type synonym definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypeAlias {
    /// The unique ID of this definition.
    pub id: DefId,
    /// The name of the type alias.
    pub name: Symbol,
    /// Type parameters.
    pub params: Vec<TyVar>,
    /// The aliased type.
    pub ty: Ty,
    /// Source span.
    pub span: Span,
}

/// A functional dependency declaration.
///
/// Represents `a b -> c d` meaning "given a and b, c and d are uniquely determined".
/// The indices refer to positions in the class's type parameter list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunDep {
    /// Indices of determining type parameters (left side of ->).
    pub from: Vec<usize>,
    /// Indices of determined type parameters (right side of ->).
    pub to: Vec<usize>,
    /// Source span.
    pub span: Span,
}

/// A type class definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassDef {
    /// The unique ID of this definition.
    pub id: DefId,
    /// The class name.
    pub name: Symbol,
    /// Type parameters.
    pub params: Vec<TyVar>,
    /// Functional dependencies (e.g., `| a -> b` means `a` determines `b`).
    pub fundeps: Vec<FunDep>,
    /// Associated type declarations.
    pub assoc_types: Vec<AssocTypeSig>,
    /// Superclass constraints.
    pub supers: Vec<Symbol>,
    /// Method signatures.
    pub methods: Vec<MethodSig>,
    /// Default method implementations.
    pub defaults: Vec<ValueDef>,
    /// Source span.
    pub span: Span,
}

/// An associated type signature within a type class.
///
/// Example: In `class Collection c where type Elem c`, the `Elem` is an associated type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssocTypeSig {
    /// The unique ID of this associated type.
    pub id: DefId,
    /// The name of the associated type.
    pub name: Symbol,
    /// Additional type parameters beyond the class parameters.
    pub params: Vec<TyVar>,
    /// The result kind (usually `*`).
    pub kind: bhc_types::Kind,
    /// Optional default type definition.
    pub default: Option<Ty>,
    /// Source span.
    pub span: Span,
}

/// A method signature in a class definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MethodSig {
    /// The method name.
    pub name: Symbol,
    /// The definition ID (from name resolution).
    pub id: DefId,
    /// The method type (with class constraint implicit).
    pub ty: Scheme,
    /// Source span.
    pub span: Span,
}

/// A type class instance definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceDef {
    /// The class being instantiated.
    pub class: Symbol,
    /// The instance types.
    pub types: Vec<Ty>,
    /// Instance constraints (e.g., `Describable a` in `instance Describable a => Describable (Box a)`).
    pub constraints: Vec<bhc_types::Constraint>,
    /// Associated type implementations.
    pub assoc_type_impls: Vec<AssocTypeImpl>,
    /// Method implementations.
    pub methods: Vec<ValueDef>,
    /// Source span.
    pub span: Span,
}

/// An associated type implementation within an instance.
///
/// Example: In `instance Collection [a] where type Elem [a] = a`,
/// the `Elem [a] = a` is an associated type implementation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssocTypeImpl {
    /// The name of the associated type.
    pub name: Symbol,
    /// Type arguments (patterns matching the instance head).
    pub args: Vec<Ty>,
    /// The implementation type (right-hand side).
    pub rhs: Ty,
    /// Source span.
    pub span: Span,
}

/// A fixity declaration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixityDecl {
    /// The fixity (infixl, infixr, infix).
    pub fixity: Fixity,
    /// The precedence (0-9).
    pub precedence: u8,
    /// The operator names.
    pub ops: Vec<Symbol>,
    /// Source span.
    pub span: Span,
}

/// Operator fixity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fixity {
    /// Left-associative.
    Left,
    /// Right-associative.
    Right,
    /// Non-associative.
    None,
}

/// A foreign import declaration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForeignDecl {
    /// The unique ID of this definition.
    pub id: DefId,
    /// The Haskell name.
    pub name: Symbol,
    /// The foreign name/symbol.
    pub foreign_name: Symbol,
    /// The calling convention.
    pub convention: ForeignConvention,
    /// The type signature.
    pub ty: Scheme,
    /// Source span.
    pub span: Span,
}

/// Foreign calling convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForeignConvention {
    /// C calling convention.
    CCall,
    /// Standard call convention (Windows).
    StdCall,
    /// JavaScript FFI.
    JavaScript,
}

/// A HIR module.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Module {
    /// The module name.
    pub name: Symbol,
    /// Export list (None means export all).
    pub exports: Option<Vec<Export>>,
    /// Import declarations.
    pub imports: Vec<Import>,
    /// Top-level items.
    pub items: Vec<Item>,
    /// Source span.
    pub span: Span,
    /// Whether {-# LANGUAGE OverloadedStrings #-} is enabled.
    pub overloaded_strings: bool,
    /// Whether {-# LANGUAGE ScopedTypeVariables #-} is enabled.
    pub scoped_type_variables: bool,
    /// Whether {-# LANGUAGE GeneralizedNewtypeDeriving #-} is enabled.
    pub generalized_newtype_deriving: bool,
    /// Whether {-# LANGUAGE FlexibleInstances #-} is enabled.
    pub flexible_instances: bool,
    /// Whether {-# LANGUAGE FlexibleContexts #-} is enabled.
    pub flexible_contexts: bool,
}

/// An export specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Export {
    /// The exported name.
    pub name: Symbol,
    /// For types: export constructors/methods.
    pub children: ExportChildren,
    /// Source span.
    pub span: Span,
}

/// What children to export for a type/class.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExportChildren {
    /// Export nothing (just the type).
    None,
    /// Export all constructors/methods.
    All,
    /// Export specific constructors/methods.
    Some(Vec<Symbol>),
}

/// An import declaration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Import {
    /// The module being imported.
    pub module: Symbol,
    /// Whether this is a qualified import.
    pub qualified: bool,
    /// The alias (for qualified imports).
    pub alias: Option<Symbol>,
    /// The import list (None means import all).
    pub items: Option<Vec<ImportItem>>,
    /// Whether this is a hiding import.
    pub hiding: bool,
    /// Source span.
    pub span: Span,
}

/// An item in an import list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportItem {
    /// The imported name.
    pub name: Symbol,
    /// For types: imported constructors/methods.
    pub children: ExportChildren,
    /// Source span.
    pub span: Span,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hir_id_indexing() {
        let id = HirId::new(42);
        assert_eq!(id.index(), 42);
    }

    #[test]
    fn test_pat_bound_vars() {
        // SAFETY: These are valid symbol indices for testing purposes
        let x = unsafe { Symbol::from_raw(0) };
        let y = unsafe { Symbol::from_raw(1) };
        let def_x = DefId::new(1);
        let def_y = DefId::new(2);

        // Simple variable pattern
        let pat = Pat::Var(x, def_x, Span::default());
        assert_eq!(pat.bound_vars(), vec![x]);

        // Tuple pattern
        let tuple_pat = Pat::Con(
            DefRef {
                def_id: DefId::new(0),
                span: Span::default(),
            },
            vec![
                Pat::Var(x, def_x, Span::default()),
                Pat::Var(y, def_y, Span::default()),
            ],
            Span::default(),
        );
        let vars = tuple_pat.bound_vars();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&x));
        assert!(vars.contains(&y));
    }

    #[test]
    fn test_fundep_creation() {
        // Test creating a functional dependency: a -> b (param 0 determines param 1)
        let fundep = FunDep {
            from: vec![0],
            to: vec![1],
            span: Span::default(),
        };
        assert_eq!(fundep.from, vec![0]);
        assert_eq!(fundep.to, vec![1]);
    }

    #[test]
    fn test_class_with_fundeps() {
        use bhc_types::{Kind, TyVar};

        // Test: class Convert a b | a -> b where convert :: a -> b
        let a_var = TyVar::new(0, Kind::Star);
        let b_var = TyVar::new(1, Kind::Star);

        let class = ClassDef {
            id: DefId::new(100),
            name: Symbol::intern("Convert"),
            params: vec![a_var, b_var],
            fundeps: vec![FunDep {
                from: vec![0], // 'a' determines
                to: vec![1],   // 'b'
                span: Span::default(),
            }],
            supers: vec![],
            methods: vec![],
            defaults: vec![],
            assoc_types: vec![],
            span: Span::default(),
        };

        assert_eq!(class.fundeps.len(), 1);
        assert_eq!(class.fundeps[0].from, vec![0]);
        assert_eq!(class.fundeps[0].to, vec![1]);
    }

    #[test]
    fn test_class_with_multi_fundeps() {
        use bhc_types::{Kind, TyVar};

        // Test: class Collection c e i | c -> e, c -> i where ...
        // This represents a collection type 'c' that determines both
        // its element type 'e' and index type 'i'
        let c_var = TyVar::new(0, Kind::Star);
        let e_var = TyVar::new(1, Kind::Star);
        let i_var = TyVar::new(2, Kind::Star);

        let class = ClassDef {
            id: DefId::new(101),
            name: Symbol::intern("Collection"),
            params: vec![c_var, e_var, i_var],
            fundeps: vec![
                FunDep {
                    from: vec![0], // 'c' determines
                    to: vec![1],   // 'e'
                    span: Span::default(),
                },
                FunDep {
                    from: vec![0], // 'c' also determines
                    to: vec![2],   // 'i'
                    span: Span::default(),
                },
            ],
            supers: vec![],
            methods: vec![],
            defaults: vec![],
            assoc_types: vec![],
            span: Span::default(),
        };

        assert_eq!(class.fundeps.len(), 2);
        // First fundep: c -> e
        assert_eq!(class.fundeps[0].from, vec![0]);
        assert_eq!(class.fundeps[0].to, vec![1]);
        // Second fundep: c -> i
        assert_eq!(class.fundeps[1].from, vec![0]);
        assert_eq!(class.fundeps[1].to, vec![2]);
    }
}
