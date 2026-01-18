//! Abstract syntax tree definitions for BHC.
//!
//! This crate defines the AST produced by parsing Haskell 2026 source code.
//! The AST preserves source locations and syntactic structure.

#![warn(missing_docs)]

use bhc_index::define_index;
use bhc_intern::{Ident, Symbol};
use bhc_span::Span;

define_index! {
    /// Index into the expression arena.
    pub struct ExprId;

    /// Index into the pattern arena.
    pub struct PatId;

    /// Index into the type arena.
    pub struct TypeId;

    /// Index into the declaration arena.
    pub struct DeclId;
}

/// A Haskell module.
#[derive(Clone, Debug)]
pub struct Module {
    /// Module pragmas (LANGUAGE, OPTIONS_GHC, etc.).
    pub pragmas: Vec<Pragma>,
    /// Module name.
    pub name: Option<ModuleName>,
    /// Export list.
    pub exports: Option<Vec<Export>>,
    /// Import declarations.
    pub imports: Vec<ImportDecl>,
    /// Top-level declarations.
    pub decls: Vec<Decl>,
    /// Span of the entire module.
    pub span: Span,
}

// ============================================================
// Pragmas
// ============================================================

/// A pragma in the source code.
#[derive(Clone, Debug)]
pub struct Pragma {
    /// The kind of pragma.
    pub kind: PragmaKind,
    /// The span.
    pub span: Span,
}

/// The kind of pragma.
#[derive(Clone, Debug)]
pub enum PragmaKind {
    /// Language extension: `{-# LANGUAGE GADTs #-}`
    Language(Vec<Symbol>),
    /// GHC options: `{-# OPTIONS_GHC -Wall #-}`
    OptionsGhc(String),
    /// Inline pragma: `{-# INLINE foo #-}`
    Inline(Ident),
    /// No-inline pragma: `{-# NOINLINE foo #-}`
    NoInline(Ident),
    /// Inlinable pragma: `{-# INLINABLE foo #-}`
    Inlinable(Ident),
    /// Specialize pragma: `{-# SPECIALIZE foo :: Int -> Int #-}`
    Specialize(Ident, Type),
    /// Unpack pragma: `{-# UNPACK #-}`
    Unpack,
    /// No-unpack pragma: `{-# NOUNPACK #-}`
    NoUnpack,
    /// Source pragma (for generated code): `{-# SOURCE #-}`
    Source,
    /// Complete pragma: `{-# COMPLETE Pat1, Pat2 #-}`
    Complete(Vec<Ident>),
    /// Minimal pragma: `{-# MINIMAL foo | bar #-}`
    Minimal(String),
    /// Deprecated pragma: `{-# DEPRECATED foo "message" #-}`
    Deprecated(Option<Vec<Ident>>, String),
    /// Warning pragma: `{-# WARNING foo "message" #-}`
    Warning(Option<Vec<Ident>>, String),
    /// Unknown/unsupported pragma (preserved for compatibility)
    Other(String),
}

/// Known language extensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Extension {
    // Type system
    /// GADTs
    GADTs,
    /// Type families
    TypeFamilies,
    /// Data kinds
    DataKinds,
    /// Kind signatures
    KindSignatures,
    /// Rank-N types
    RankNTypes,
    /// Existential quantification
    ExistentialQuantification,
    /// Scoped type variables
    ScopedTypeVariables,
    /// Type applications
    TypeApplications,
    /// Flexible instances
    FlexibleInstances,
    /// Flexible contexts
    FlexibleContexts,
    /// Multi-param type classes
    MultiParamTypeClasses,
    /// Functional dependencies
    FunctionalDependencies,
    /// Undecidable instances
    UndecidableInstances,
    /// Overlapping instances
    OverlappingInstances,
    /// Constraint kinds
    ConstraintKinds,

    // Syntax
    /// Lambda case
    LambdaCase,
    /// Multi-way if
    MultiWayIf,
    /// Block arguments
    BlockArguments,
    /// Pattern guards
    PatternGuards,
    /// View patterns
    ViewPatterns,
    /// Pattern synonyms
    PatternSynonyms,
    /// Record wild cards
    RecordWildCards,
    /// Named field puns
    NamedFieldPuns,
    /// Overloaded strings
    OverloadedStrings,
    /// Overloaded lists
    OverloadedLists,
    /// Numeric underscores
    NumericUnderscores,
    /// Hex float literals
    HexFloatLiterals,
    /// Binary literals
    BinaryLiterals,
    /// Negative literals
    NegativeLiterals,

    // Strictness
    /// Bang patterns
    BangPatterns,
    /// Strict data
    StrictData,
    /// Strict
    Strict,

    // Deriving
    /// Derive functor
    DeriveFunctor,
    /// Derive foldable
    DeriveFoldable,
    /// Derive traversable
    DeriveTraversable,
    /// Derive generic
    DeriveGeneric,
    /// Derive data typeable
    DeriveDataTypeable,
    /// Derive lift
    DeriveLift,
    /// Deriving via
    DerivingVia,
    /// Deriving strategies
    DerivingStrategies,
    /// Generalized newtype deriving
    GeneralizedNewtypeDeriving,
    /// Standalone deriving
    StandaloneDeriving,

    // FFI
    /// Foreign function interface
    ForeignFunctionInterface,
    /// C API FFI
    CApiFFI,
    /// Unsafe FFI
    UnliftedFFITypes,

    // Other
    /// Template Haskell
    TemplateHaskell,
    /// Template Haskell quotes
    TemplateHaskellQuotes,
    /// Quasi quotes
    QuasiQuotes,
    /// Type operators
    TypeOperators,
    /// Explicit forall
    ExplicitForAll,
    /// Explicit namespaces
    ExplicitNamespaces,
    /// Empty data declarations
    EmptyDataDecls,
    /// Empty case
    EmptyCase,
    /// Instance sigs
    InstanceSigs,
    /// Default signatures
    DefaultSignatures,
    /// Named defaults
    NamedDefaults,

    /// Unknown extension (preserved)
    Unknown(Symbol),
}

impl Extension {
    /// Parse an extension name.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name {
            // Type system
            "GADTs" => Self::GADTs,
            "TypeFamilies" => Self::TypeFamilies,
            "DataKinds" => Self::DataKinds,
            "KindSignatures" => Self::KindSignatures,
            "RankNTypes" | "Rank2Types" | "PolymorphicComponents" => Self::RankNTypes,
            "ExistentialQuantification" => Self::ExistentialQuantification,
            "ScopedTypeVariables" => Self::ScopedTypeVariables,
            "TypeApplications" => Self::TypeApplications,
            "FlexibleInstances" => Self::FlexibleInstances,
            "FlexibleContexts" => Self::FlexibleContexts,
            "MultiParamTypeClasses" => Self::MultiParamTypeClasses,
            "FunctionalDependencies" => Self::FunctionalDependencies,
            "UndecidableInstances" => Self::UndecidableInstances,
            "OverlappingInstances" | "IncoherentInstances" => Self::OverlappingInstances,
            "ConstraintKinds" => Self::ConstraintKinds,

            // Syntax
            "LambdaCase" => Self::LambdaCase,
            "MultiWayIf" => Self::MultiWayIf,
            "BlockArguments" => Self::BlockArguments,
            "PatternGuards" => Self::PatternGuards,
            "ViewPatterns" => Self::ViewPatterns,
            "PatternSynonyms" => Self::PatternSynonyms,
            "RecordWildCards" => Self::RecordWildCards,
            "NamedFieldPuns" => Self::NamedFieldPuns,
            "OverloadedStrings" => Self::OverloadedStrings,
            "OverloadedLists" => Self::OverloadedLists,
            "NumericUnderscores" => Self::NumericUnderscores,
            "HexFloatLiterals" => Self::HexFloatLiterals,
            "BinaryLiterals" => Self::BinaryLiterals,
            "NegativeLiterals" => Self::NegativeLiterals,

            // Strictness
            "BangPatterns" => Self::BangPatterns,
            "StrictData" => Self::StrictData,
            "Strict" => Self::Strict,

            // Deriving
            "DeriveFunctor" => Self::DeriveFunctor,
            "DeriveFoldable" => Self::DeriveFoldable,
            "DeriveTraversable" => Self::DeriveTraversable,
            "DeriveGeneric" => Self::DeriveGeneric,
            "DeriveDataTypeable" => Self::DeriveDataTypeable,
            "DeriveLift" => Self::DeriveLift,
            "DerivingVia" => Self::DerivingVia,
            "DerivingStrategies" => Self::DerivingStrategies,
            "GeneralizedNewtypeDeriving" | "GeneralisedNewtypeDeriving" => Self::GeneralizedNewtypeDeriving,
            "StandaloneDeriving" => Self::StandaloneDeriving,

            // FFI
            "ForeignFunctionInterface" | "FFI" => Self::ForeignFunctionInterface,
            "CApiFFI" => Self::CApiFFI,
            "UnliftedFFITypes" => Self::UnliftedFFITypes,

            // Other
            "TemplateHaskell" => Self::TemplateHaskell,
            "TemplateHaskellQuotes" => Self::TemplateHaskellQuotes,
            "QuasiQuotes" => Self::QuasiQuotes,
            "TypeOperators" => Self::TypeOperators,
            "ExplicitForAll" => Self::ExplicitForAll,
            "ExplicitNamespaces" => Self::ExplicitNamespaces,
            "EmptyDataDecls" => Self::EmptyDataDecls,
            "EmptyCase" => Self::EmptyCase,
            "InstanceSigs" => Self::InstanceSigs,
            "DefaultSignatures" => Self::DefaultSignatures,
            "NamedDefaults" => Self::NamedDefaults,

            _ => Self::Unknown(Symbol::intern(name)),
        }
    }
}

/// A qualified module name like `Data.List`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModuleName {
    /// The components of the name.
    pub parts: Vec<Symbol>,
    /// The span.
    pub span: Span,
}

impl ModuleName {
    /// Get the fully qualified name as a string.
    #[must_use]
    pub fn to_string(&self) -> String {
        self.parts
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(".")
    }
}

/// An export specification.
#[derive(Clone, Debug)]
pub enum Export {
    /// Export a value: `foo`
    Var(Ident, Span),
    /// Export a type with optional constructors: `Foo(..)` or `Foo(A, B)`
    Type(Ident, Option<Vec<Ident>>, Span),
    /// Export a module: `module Data.List`
    Module(ModuleName, Span),
}

/// An import declaration.
#[derive(Clone, Debug)]
pub struct ImportDecl {
    /// The module being imported.
    pub module: ModuleName,
    /// Whether this is a qualified import.
    pub qualified: bool,
    /// The alias for qualified imports.
    pub alias: Option<ModuleName>,
    /// The import specification (hiding or explicit).
    pub spec: Option<ImportSpec>,
    /// The span.
    pub span: Span,
}

/// Import specification.
#[derive(Clone, Debug)]
pub enum ImportSpec {
    /// Import only the listed items.
    Only(Vec<Import>),
    /// Import everything except the listed items.
    Hiding(Vec<Import>),
}

/// A single import item.
#[derive(Clone, Debug)]
pub enum Import {
    /// Import a value.
    Var(Ident, Span),
    /// Import a type with optional constructors.
    Type(Ident, Option<Vec<Ident>>, Span),
}

/// A top-level declaration.
#[derive(Clone, Debug)]
pub enum Decl {
    /// Type signature: `foo :: Int -> Int`
    TypeSig(TypeSig),
    /// Function/value binding: `foo x = x + 1`
    FunBind(FunBind),
    /// Data type: `data Foo = A | B Int`
    DataDecl(DataDecl),
    /// Type alias: `type Foo = Bar`
    TypeAlias(TypeAlias),
    /// Newtype: `newtype Foo = Foo Bar`
    Newtype(NewtypeDecl),
    /// Class definition: `class Eq a where ...`
    ClassDecl(ClassDecl),
    /// Instance definition: `instance Eq Int where ...`
    InstanceDecl(InstanceDecl),
    /// Foreign import/export
    Foreign(ForeignDecl),
    /// Fixity declaration: `infixl 6 +`
    Fixity(FixityDecl),
}

/// A type signature.
#[derive(Clone, Debug)]
pub struct TypeSig {
    /// The names being typed.
    pub names: Vec<Ident>,
    /// The type.
    pub ty: Type,
    /// The span.
    pub span: Span,
}

/// A function binding.
#[derive(Clone, Debug)]
pub struct FunBind {
    /// The function name.
    pub name: Ident,
    /// The clauses (pattern matches).
    pub clauses: Vec<Clause>,
    /// The span.
    pub span: Span,
}

/// A clause in a function binding.
#[derive(Clone, Debug)]
pub struct Clause {
    /// The patterns for arguments.
    pub pats: Vec<Pat>,
    /// The right-hand side.
    pub rhs: Rhs,
    /// Local bindings.
    pub wheres: Vec<Decl>,
    /// The span.
    pub span: Span,
}

/// The right-hand side of a binding.
#[derive(Clone, Debug)]
pub enum Rhs {
    /// Simple: `= expr`
    Simple(Expr, Span),
    /// Guarded: `| guard = expr`
    Guarded(Vec<GuardedRhs>, Span),
}

/// A guarded right-hand side.
#[derive(Clone, Debug)]
pub struct GuardedRhs {
    /// The guards (can be multiple, e.g., `| pat <- expr, cond`).
    pub guards: Vec<Guard>,
    /// The body expression.
    pub body: Expr,
    /// The span.
    pub span: Span,
}

/// A guard in a guarded RHS.
#[derive(Clone, Debug)]
pub enum Guard {
    /// A pattern guard: `pat <- expr`
    Pattern(Pat, Expr, Span),
    /// A boolean guard: `expr`
    Expr(Expr, Span),
}

impl Guard {
    /// Get the span of this guard.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Pattern(_, _, s) | Self::Expr(_, s) => *s,
        }
    }
}

/// A data type declaration.
#[derive(Clone, Debug)]
pub struct DataDecl {
    /// The type name.
    pub name: Ident,
    /// Type parameters.
    pub params: Vec<TyVar>,
    /// Constructors.
    pub constrs: Vec<ConDecl>,
    /// Deriving clause.
    pub deriving: Vec<Ident>,
    /// The span.
    pub span: Span,
}

/// A data constructor declaration.
#[derive(Clone, Debug)]
pub struct ConDecl {
    /// Constructor name.
    pub name: Ident,
    /// Constructor fields.
    pub fields: ConFields,
    /// The span.
    pub span: Span,
}

/// Constructor fields.
#[derive(Clone, Debug)]
pub enum ConFields {
    /// Positional: `Foo Int String`
    Positional(Vec<Type>),
    /// Record: `Foo { bar :: Int, baz :: String }`
    Record(Vec<FieldDecl>),
}

/// A record field declaration.
#[derive(Clone, Debug)]
pub struct FieldDecl {
    /// Field name.
    pub name: Ident,
    /// Field type.
    pub ty: Type,
    /// The span.
    pub span: Span,
}

/// A type alias declaration.
#[derive(Clone, Debug)]
pub struct TypeAlias {
    /// The alias name.
    pub name: Ident,
    /// Type parameters.
    pub params: Vec<TyVar>,
    /// The aliased type.
    pub ty: Type,
    /// The span.
    pub span: Span,
}

/// A newtype declaration.
#[derive(Clone, Debug)]
pub struct NewtypeDecl {
    /// The type name.
    pub name: Ident,
    /// Type parameters.
    pub params: Vec<TyVar>,
    /// The constructor.
    pub constr: ConDecl,
    /// Deriving clause.
    pub deriving: Vec<Ident>,
    /// The span.
    pub span: Span,
}

/// A type class declaration.
#[derive(Clone, Debug)]
pub struct ClassDecl {
    /// Superclass constraints.
    pub context: Vec<Constraint>,
    /// Class name.
    pub name: Ident,
    /// Type parameters (multi-param type classes).
    pub params: Vec<TyVar>,
    /// Functional dependencies.
    pub fundeps: Vec<FunDep>,
    /// Method signatures and default implementations.
    pub methods: Vec<Decl>,
    /// The span.
    pub span: Span,
}

/// A functional dependency in a type class.
#[derive(Clone, Debug)]
pub struct FunDep {
    /// Variables that determine others.
    pub from: Vec<Ident>,
    /// Variables that are determined.
    pub to: Vec<Ident>,
    /// The span.
    pub span: Span,
}

/// An instance declaration.
#[derive(Clone, Debug)]
pub struct InstanceDecl {
    /// Instance constraints.
    pub context: Vec<Constraint>,
    /// Class name.
    pub class: Ident,
    /// Instance type.
    pub ty: Type,
    /// Method implementations.
    pub methods: Vec<Decl>,
    /// The span.
    pub span: Span,
}

/// A foreign declaration.
#[derive(Clone, Debug)]
pub struct ForeignDecl {
    /// Import or export.
    pub kind: ForeignKind,
    /// Calling convention.
    pub convention: Symbol,
    /// External name.
    pub external_name: Option<String>,
    /// Haskell name.
    pub name: Ident,
    /// Type signature.
    pub ty: Type,
    /// The span.
    pub span: Span,
}

/// Foreign declaration kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForeignKind {
    /// Foreign import.
    Import,
    /// Foreign export.
    Export,
}

/// A fixity declaration.
#[derive(Clone, Debug)]
pub struct FixityDecl {
    /// The fixity.
    pub fixity: Fixity,
    /// Precedence level (0-9).
    pub prec: u8,
    /// The operators.
    pub ops: Vec<Ident>,
    /// The span.
    pub span: Span,
}

/// Operator fixity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fixity {
    /// Left associative.
    Left,
    /// Right associative.
    Right,
    /// Non-associative.
    None,
}

/// An expression.
#[derive(Clone, Debug)]
pub enum Expr {
    /// Variable: `x`
    Var(Ident, Span),
    /// Qualified variable: `M.foo`, `Data.List.sort`
    QualVar(ModuleName, Ident, Span),
    /// Constructor: `Just`
    Con(Ident, Span),
    /// Qualified constructor: `M.Just`, `Data.Maybe.Nothing`
    QualCon(ModuleName, Ident, Span),
    /// Literal: `42`, `"hello"`
    Lit(Lit, Span),
    /// Application: `f x`
    App(Box<Expr>, Box<Expr>, Span),
    /// Lambda: `\x -> x`
    Lam(Vec<Pat>, Box<Expr>, Span),
    /// Let: `let x = 1 in x`
    Let(Vec<Decl>, Box<Expr>, Span),
    /// If: `if c then t else e`
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),
    /// Case: `case x of { ... }`
    Case(Box<Expr>, Vec<Alt>, Span),
    /// Do block: `do { ... }`
    Do(Vec<Stmt>, Span),
    /// Tuple: `(a, b, c)`
    Tuple(Vec<Expr>, Span),
    /// List: `[1, 2, 3]`
    List(Vec<Expr>, Span),
    /// Arithmetic sequence: `[1..10]`, `[1,3..10]`
    ArithSeq(ArithSeq, Span),
    /// List comprehension: `[x | x <- xs, x > 0]`
    ListComp(Box<Expr>, Vec<Stmt>, Span),
    /// Record construction: `Foo { bar = 1 }`
    RecordCon(Ident, Vec<FieldBind>, Span),
    /// Record update: `foo { bar = 1 }`
    RecordUpd(Box<Expr>, Vec<FieldBind>, Span),
    /// Infix operator: `a + b`
    Infix(Box<Expr>, Ident, Box<Expr>, Span),
    /// Negation: `-x`
    Neg(Box<Expr>, Span),
    /// Parenthesized expression
    Paren(Box<Expr>, Span),
    /// Type annotation: `x :: Int`
    Ann(Box<Expr>, Type, Span),
    /// Lazy block (H26): `lazy { ... }`
    Lazy(Box<Expr>, Span),
    /// Wildcard/hole: `_` (used in patterns parsed as expressions)
    Wildcard(Span),
}

impl Expr {
    /// Get the span of this expression.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Var(_, s)
            | Self::QualVar(_, _, s)
            | Self::Con(_, s)
            | Self::QualCon(_, _, s)
            | Self::Lit(_, s)
            | Self::App(_, _, s)
            | Self::Lam(_, _, s)
            | Self::Let(_, _, s)
            | Self::If(_, _, _, s)
            | Self::Case(_, _, s)
            | Self::Do(_, s)
            | Self::Tuple(_, s)
            | Self::List(_, s)
            | Self::ArithSeq(_, s)
            | Self::ListComp(_, _, s)
            | Self::RecordCon(_, _, s)
            | Self::RecordUpd(_, _, s)
            | Self::Infix(_, _, _, s)
            | Self::Neg(_, s)
            | Self::Paren(_, s)
            | Self::Ann(_, _, s)
            | Self::Lazy(_, s)
            | Self::Wildcard(s) => *s,
        }
    }
}

/// A literal value.
#[derive(Clone, Debug, PartialEq)]
pub enum Lit {
    /// Integer literal.
    Int(i64),
    /// Floating-point literal.
    Float(f64),
    /// Character literal.
    Char(char),
    /// String literal.
    String(String),
}

/// An arithmetic sequence.
#[derive(Clone, Debug)]
pub enum ArithSeq {
    /// `[from..]`
    From(Box<Expr>),
    /// `[from, then..]`
    FromThen(Box<Expr>, Box<Expr>),
    /// `[from..to]`
    FromTo(Box<Expr>, Box<Expr>),
    /// `[from, then..to]`
    FromThenTo(Box<Expr>, Box<Expr>, Box<Expr>),
}

/// A case alternative.
#[derive(Clone, Debug)]
pub struct Alt {
    /// The pattern.
    pub pat: Pat,
    /// The right-hand side.
    pub rhs: Rhs,
    /// Local bindings.
    pub wheres: Vec<Decl>,
    /// The span.
    pub span: Span,
}

/// A statement in a do block or list comprehension.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// Generator: `x <- xs`
    Generator(Pat, Expr, Span),
    /// Qualifier/guard: `x > 0`
    Qualifier(Expr, Span),
    /// Let binding: `let x = 1`
    LetStmt(Vec<Decl>, Span),
}

/// A field binding in a record.
#[derive(Clone, Debug)]
pub struct FieldBind {
    /// Optional module qualifier for disambiguated record fields (e.g., `XMonad.borderWidth`).
    pub qualifier: Option<ModuleName>,
    /// Field name.
    pub name: Ident,
    /// Field value (None for punning: `Foo { bar }` means `Foo { bar = bar }`)
    pub value: Option<Expr>,
    /// The span.
    pub span: Span,
}

/// A pattern.
#[derive(Clone, Debug)]
pub enum Pat {
    /// Wildcard: `_`
    Wildcard(Span),
    /// Variable: `x`
    Var(Ident, Span),
    /// Literal: `42`
    Lit(Lit, Span),
    /// Constructor: `Just x`
    Con(Ident, Vec<Pat>, Span),
    /// Infix constructor: `x : xs`
    Infix(Box<Pat>, Ident, Box<Pat>, Span),
    /// Tuple: `(a, b)`
    Tuple(Vec<Pat>, Span),
    /// List: `[a, b, c]`
    List(Vec<Pat>, Span),
    /// Record: `Foo { bar = x }`
    Record(Ident, Vec<FieldPat>, Span),
    /// As-pattern: `xs@(x:_)`
    As(Ident, Box<Pat>, Span),
    /// Lazy pattern: `~pat`
    Lazy(Box<Pat>, Span),
    /// Bang pattern: `!pat`
    Bang(Box<Pat>, Span),
    /// Parenthesized pattern
    Paren(Box<Pat>, Span),
    /// Type annotation: `x :: Int`
    Ann(Box<Pat>, Type, Span),
}

impl Pat {
    /// Get the span of this pattern.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Wildcard(s)
            | Self::Var(_, s)
            | Self::Lit(_, s)
            | Self::Con(_, _, s)
            | Self::Infix(_, _, _, s)
            | Self::Tuple(_, s)
            | Self::List(_, s)
            | Self::Record(_, _, s)
            | Self::As(_, _, s)
            | Self::Lazy(_, s)
            | Self::Bang(_, s)
            | Self::Paren(_, s)
            | Self::Ann(_, _, s) => *s,
        }
    }
}

/// A field pattern in a record.
#[derive(Clone, Debug)]
pub struct FieldPat {
    /// Optional module qualifier for disambiguated record fields (e.g., `XMonad.modMask`).
    pub qualifier: Option<ModuleName>,
    /// Field name.
    pub name: Ident,
    /// Pattern (None for punning).
    pub pat: Option<Pat>,
    /// The span.
    pub span: Span,
}

/// A type.
#[derive(Clone, Debug)]
pub enum Type {
    /// Type variable: `a`
    Var(TyVar, Span),
    /// Type constructor: `Int`, `Maybe`
    Con(Ident, Span),
    /// Qualified type constructor: `M.Map`, `Data.List.Sort`
    QualCon(ModuleName, Ident, Span),
    /// Application: `Maybe Int`
    App(Box<Type>, Box<Type>, Span),
    /// Function type: `a -> b`
    Fun(Box<Type>, Box<Type>, Span),
    /// Tuple type: `(a, b)`
    Tuple(Vec<Type>, Span),
    /// List type: `[a]`
    List(Box<Type>, Span),
    /// Parenthesized type
    Paren(Box<Type>, Span),
    /// Forall type: `forall a. a -> a`
    Forall(Vec<TyVar>, Box<Type>, Span),
    /// Constrained type: `Eq a => a -> a -> Bool`
    Constrained(Vec<Constraint>, Box<Type>, Span),

    // === M9 Dependent Types Preview ===

    /// Promoted list: `'[1024, 768]` for tensor shapes
    PromotedList(Vec<Type>, Span),
    /// Type-level natural literal: `1024` in type position
    NatLit(u64, Span),

    /// Strict type annotation: `!Int` in constructor fields
    Bang(Box<Type>, Span),

    /// Lazy type annotation: `~Int` in constructor fields
    Lazy(Box<Type>, Span),
}

impl Type {
    /// Get the span of this type.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Var(_, s)
            | Self::Con(_, s)
            | Self::QualCon(_, _, s)
            | Self::App(_, _, s)
            | Self::Fun(_, _, s)
            | Self::Tuple(_, s)
            | Self::List(_, s)
            | Self::Paren(_, s)
            | Self::Forall(_, _, s)
            | Self::Constrained(_, _, s)
            | Self::PromotedList(_, s)
            | Self::NatLit(_, s)
            | Self::Bang(_, s)
            | Self::Lazy(_, s) => *s,
        }
    }
}

/// A type variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TyVar {
    /// The name.
    pub name: Ident,
    /// The span.
    pub span: Span,
}

/// A type class constraint.
#[derive(Clone, Debug)]
pub struct Constraint {
    /// The class name.
    pub class: Ident,
    /// The type arguments.
    pub args: Vec<Type>,
    /// The span.
    pub span: Span,
}
