# bhc-core

Core Intermediate Representation for the Basel Haskell Compiler.

## Overview

Core IR is the typed, explicit, and optimizable representation at the heart of BHC. It is:

- **Typed**: Every expression carries its type explicitly
- **Explicit**: Type applications, coercions, and casts are visible
- **A-Normal Form**: Complex expressions are let-bound
- **Optimizable**: Designed for efficient transformation and analysis

## IR Pipeline Position

```
Source Code
    ↓
[Parse/AST]  ← Surface syntax
    ↓
[HIR]        ← Desugared, resolved
    ↓
[Core IR]    ← This crate: typed, explicit
    ↓
[Tensor IR]  ← Shape/stride aware (Numeric Profile)
    ↓
[Loop IR]    ← Iteration, vectorization
```

## Core Types

| Type | Description |
|------|-------------|
| `Expr` | Core IR expressions |
| `Var` | Variables with types |
| `Bind` | Let bindings |
| `Alt` | Case alternatives |
| `Literal` | Literal values |
| `Coercion` | Type coercions |
| `CoreModule` | A compiled module |

## Expressions

```rust
pub enum Expr {
    /// Variable reference
    Var(Var, Span),

    /// Literal value
    Lit(Literal, Ty, Span),

    /// Function application: `f x`
    App(Box<Expr>, Box<Expr>, Span),

    /// Type application: `f @ty`
    TyApp(Box<Expr>, Ty, Span),

    /// Lambda: `\x -> e`
    Lam(Var, Box<Expr>, Span),

    /// Type abstraction: `/\a -> e`
    TyLam(TyVar, Box<Expr>, Span),

    /// Let binding: `let binds in e`
    Let(Box<Bind>, Box<Expr>, Span),

    /// Case expression: `case e of { alts }`
    Case(Box<Expr>, Vec<Alt>, Ty, Span),

    /// Lazy escape hatch: `lazy { e }`
    Lazy(Box<Expr>, Span),

    /// Type cast with coercion
    Cast(Box<Expr>, Coercion, Span),

    /// Profiling tick
    Tick(Tick, Box<Expr>, Span),

    /// Type annotation
    Type(Ty, Span),

    /// Coercion value
    Coercion(Coercion, Span),
}
```

### Expression Properties

```rust
impl Expr {
    /// Get the span
    pub fn span(&self) -> Span;

    /// Get the type (computed from structure)
    pub fn ty(&self) -> Ty;

    /// Is this a value (WHNF)?
    pub fn is_value(&self) -> bool;

    /// Is this trivial (var or literal)?
    pub fn is_trivial(&self) -> bool;

    /// Get free variables
    pub fn free_vars(&self) -> Vec<Var>;
}
```

## Variables

Variables in Core carry their types:

```rust
pub struct Var {
    /// Variable name
    pub name: Symbol,
    /// Unique identifier
    pub id: VarId,
    /// The type
    pub ty: Ty,
}

// Create a variable
let x = Var::new(Symbol::intern("x"), VarId::new(0), Ty::int());
```

## Bindings

```rust
pub enum Bind {
    /// Non-recursive: `let x = e`
    NonRec(Var, Box<Expr>),

    /// Mutually recursive: `let rec { x1 = e1; x2 = e2 }`
    Rec(Vec<(Var, Box<Expr>)>),
}

impl Bind {
    /// Get bound variables
    pub fn bound_vars(&self) -> SmallVec<[&Var; 4]>;

    /// Is this recursive?
    pub fn is_recursive(&self) -> bool;
}
```

## Case Expressions

```rust
pub struct Alt {
    /// Constructor or literal being matched
    pub con: AltCon,
    /// Variables bound by the pattern
    pub binders: Vec<Var>,
    /// Right-hand side
    pub rhs: Expr,
}

pub enum AltCon {
    /// Data constructor: `Just x`
    DataCon(DataCon),
    /// Literal: `42`
    Lit(Literal),
    /// Default case
    Default,
}
```

### Data Constructors

```rust
pub struct DataCon {
    /// Constructor name
    pub name: Symbol,
    /// Type constructor it belongs to
    pub ty_con: TyCon,
    /// Tag (index) of this constructor
    pub tag: u32,
    /// Number of fields
    pub arity: u32,
}
```

## Literals

```rust
pub enum Literal {
    /// Machine integer
    Int(i64),
    /// Arbitrary precision integer
    Integer(i128),
    /// Single-precision float
    Float(f32),
    /// Double-precision float
    Double(f64),
    /// Character
    Char(char),
    /// String (interned)
    String(Symbol),
}

impl Literal {
    /// Get the Core type name
    pub fn core_type(&self) -> &'static str;
}
```

## Coercions

Type coercions are proof terms for type equality:

```rust
pub struct Coercion {
    /// Source type
    pub source_ty: Ty,
    /// Result type
    pub result_ty: Ty,
    /// Kind of coercion
    pub kind: CoercionKind,
}

pub enum CoercionKind {
    /// Reflexivity: `a ~ a`
    Refl,
    /// Symmetry: `a ~ b => b ~ a`
    Sym(Box<Coercion>),
    /// Transitivity: `a ~ b, b ~ c => a ~ c`
    Trans(Box<Coercion>, Box<Coercion>),
    /// Newtype coercion
    Newtype(Symbol),
    /// Axiom application
    Axiom(Symbol, Vec<Ty>),
    /// Universal coercion
    Forall(TyVar, Box<Coercion>),
    /// Coercion for type application
    App(Box<Coercion>, Box<Coercion>),
}
```

## Lazy Escape Hatch

In strict profiles (Numeric, Edge), `lazy { }` forces lazy evaluation:

```rust
// Lazy block wraps expression in thunk
Expr::Lazy(Box::new(expensive_computation), span)
```

From H26-SPEC Section 6.4:
- Default Profile: No effect (already lazy)
- Numeric/Edge Profile: Creates thunk, evaluated on demand

## Profiling Ticks

```rust
pub enum Tick {
    /// Cost center for profiling
    Profiling(Symbol),
    /// Source note for debugging
    SourceNote(Span),
}
```

## Strictness Analysis

```rust
pub enum Strictness {
    /// Lazy evaluation (default)
    Lazy,
    /// Strict (evaluated before use)
    Strict,
    /// Hyperstrict (evaluated to NF)
    Hyperstrict,
}
```

## Occurrence Analysis

For inlining decisions:

```rust
pub struct OccInfo {
    /// Usage count
    pub occ_count: OccCount,
    /// Used inside lambda?
    pub inside_lam: bool,
    /// One-shot context?
    pub one_shot: bool,
}

pub enum OccCount {
    /// Never used (dead code)
    Dead,
    /// Used exactly once
    Once,
    /// Used multiple times
    Many,
}
```

## Core Modules

```rust
pub struct CoreModule {
    /// Module name
    pub name: Symbol,
    /// Top-level bindings
    pub bindings: Vec<Bind>,
    /// Foreign exports
    pub exports: Vec<ForeignExport>,
}

pub struct ForeignExport {
    pub name: Symbol,
    pub var: Var,
    pub convention: ForeignConv,
}
```

## Submodules

### eval - Evaluation

```rust
use bhc_core::eval;

// Evaluate a Core expression
let result = eval::eval(&expr, &env)?;
```

### uarray - Unboxed Arrays

```rust
use bhc_core::uarray;

// Create unboxed array operations
let arr = uarray::new_array(elem_ty, size);
```

## Transformations

Core IR supports many optimizations:

| Transformation | Description |
|---------------|-------------|
| Beta reduction | `(\x -> e) v` → `e[x/v]` |
| Inlining | Replace variable with definition |
| Case-of-case | Nested case simplification |
| Case-of-known | Case on known constructor |
| Let floating | Move lets for better sharing |
| Strictness analysis | Determine evaluation order |
| Dead code elimination | Remove unused bindings |

## Type Safety

Core is explicitly typed:

```rust
// Type application visible
let id_int = Expr::TyApp(
    Box::new(id_expr),  // forall a. a -> a
    Ty::int(),          // @Int
    span,
);  // Int -> Int

// Lambda carries type of binder
let lam = Expr::Lam(
    Var::new(x, id, Ty::int()),  // x :: Int
    body,
    span,
);
```

## Integration

Core IR connects HIR to lower-level IRs:

```
HIR → [Core Lowering] → Core IR
                           ↓
           [Optimizations] (beta, inline, etc.)
                           ↓
                        Core IR
                           ↓
    [Tensor/Loop Lowering] (Numeric Profile)
                           ↓
                     Tensor IR / Loop IR
```

## See Also

- `bhc-hir`: Source HIR that lowers to Core
- `bhc-tensor-ir`: Numeric IR for tensor operations
- `bhc-loop-ir`: Loop IR for explicit iteration
- `bhc-types`: Type system definitions
- H26-SPEC Section 5: Core IR Specification
