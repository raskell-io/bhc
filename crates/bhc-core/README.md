# bhc-core

Core Intermediate Representation for the Basel Haskell Compiler.

## Overview

This crate defines the Core IR, a typed, explicit, and optimizable representation derived from GHC's Core. Core IR is the primary representation for optimization passes and serves as the bridge to lower-level code generation.

## Features

- Explicitly typed lambda calculus
- System FC with coercions
- A-Normal Form (ANF) ready
- Let-floating and inlining support
- Strictness annotations
- Lazy escape hatch for controlled laziness
- Unboxed types and primops

## Key Types

| Type | Description |
|------|-------------|
| `Expr` | Core expression enum |
| `Var` | Variable with type and metadata |
| `VarId` | Unique variable identifier |
| `Bind` | Binding (non-recursive or recursive) |
| `Alt` | Case alternative |
| `Literal` | Literal values |
| `Coercion` | Type coercions for GADTs/newtypes |
| `CoreModule` | Complete Core module |

## Usage

### Working with Core Expressions

```rust
use bhc_core::{Expr, Var, VarId, Bind, Literal};
use bhc_types::Ty;

// Create a variable
let x = Var::new(VarId::new(0), "x", Ty::int_prim());

// Create a literal
let lit = Expr::Lit(Literal::Int(42));

// Create an application: f x
let app = Expr::App(Box::new(f_expr), Box::new(x_expr));

// Create a lambda: \x -> x
let lam = Expr::Lam(x.clone(), Box::new(Expr::Var(x)));
```

### Pattern Matching on Core

```rust
fn count_apps(expr: &Expr) -> usize {
    match expr {
        Expr::App(f, _) => 1 + count_apps(f),
        Expr::Lam(_, body) => count_apps(body),
        Expr::Let(bind, body) => {
            let bind_count = match bind {
                Bind::NonRec(_, e) => count_apps(e),
                Bind::Rec(binds) => binds.iter().map(|(_, e)| count_apps(e)).sum(),
            };
            bind_count + count_apps(body)
        }
        Expr::Case(scrut, _, alts) => {
            count_apps(scrut) + alts.iter().map(|a| count_apps(&a.rhs)).sum::<usize>()
        }
        _ => 0,
    }
}
```

## Expression Variants

```rust
pub enum Expr {
    Var(Var),                     // Variable reference
    Lit(Literal),                 // Literal value
    App(Box<Expr>, Box<Expr>),    // Application
    Lam(Var, Box<Expr>),          // Lambda abstraction
    Let(Bind, Box<Expr>),         // Let binding
    Case(Box<Expr>, Var, Vec<Alt>), // Case expression
    Cast(Box<Expr>, Coercion),    // Type cast
    Type(Ty),                     // Type argument
    Coercion(Coercion),           // Coercion argument

    // Strictness control
    Strict(Box<Expr>),            // Force evaluation (!)
    Lazy(Box<Expr>),              // Defer evaluation (~)
}
```

## Binding Forms

```rust
pub enum Bind {
    NonRec(Var, Expr),            // let x = e
    Rec(Vec<(Var, Expr)>),        // let rec { x = e1; y = e2 }
}

pub struct Alt {
    pub con: AltCon,              // Constructor or literal
    pub binders: Vec<Var>,        // Bound variables
    pub rhs: Expr,                // Right-hand side
}

pub enum AltCon {
    DataCon(DataCon),             // Constructor: Just, Left
    Literal(Literal),             // Literal: 42, 'a'
    Default,                      // Default case: _
}
```

## Coercions

Type coercions enable safe type conversions:

```rust
pub enum Coercion {
    Refl(Ty),                     // Reflexivity: t ~ t
    Sym(Box<Coercion>),           // Symmetry: if c : t1 ~ t2, then sym c : t2 ~ t1
    Trans(Box<Coercion>, Box<Coercion>), // Transitivity
    App(Box<Coercion>, Box<Coercion>),   // Application
    Forall(Var, Box<Coercion>),   // Polymorphic coercion
    Inst(Box<Coercion>, Ty),      // Instantiation
    Newtype(TyCon, Ty),           // Newtype unwrapping
    Axiom(AxiomId, Vec<Ty>),      // Axiom instantiation
}
```

## Strict vs Lazy

Core provides explicit control over evaluation:

```rust
// Force immediate evaluation
let strict_expr = Expr::Strict(Box::new(expensive_computation));

// Defer evaluation (create thunk)
let lazy_expr = Expr::Lazy(Box::new(expensive_computation));

// Default: follows profile (Numeric = strict, Default = lazy)
let default_expr = some_expr;
```

## Optimization Passes

Core is designed for these optimizations:

| Pass | Description |
|------|-------------|
| Simplifier | Beta reduction, inlining, constant folding |
| Float-out | Let floating for sharing |
| Float-in | Let sinking for strictness |
| Strictness | Demand analysis for unboxing |
| CSE | Common subexpression elimination |
| Dead code | Remove unused bindings |
| Spec constr | Specialization at call sites |

## Design Notes

- All types are explicit (no inference needed)
- Variables carry occurrence info for optimization
- Coercions are erased in later stages
- ANF conversion happens before code generation
- Primops are represented as special variables

## Related Crates

- `bhc-hir` - Input representation
- `bhc-hir-to-core` - HIR to Core lowering
- `bhc-types` - Type representation
- `bhc-tensor-ir` - Numeric optimization (after Core)
- `bhc-codegen` - Code generation from Core

## Specification References

- H26-SPEC Section 3.3: Core IR Definition
- H26-SPEC Section 5: Optimization Passes
- H26-SPEC Section 6: Strictness Model
