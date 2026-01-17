# bhc-types

Type system representation for the Basel Haskell Compiler.

## Overview

`bhc-types` provides the type representation used throughout BHC, including:

- **Hindley-Milner types**: Variables, constructors, functions, foralls
- **Higher-kinded types**: Type constructors with arrow kinds
- **Type classes**: Constraints and qualified types
- **Unboxed primitives**: Machine-level types for Numeric Profile
- **M9 Dependent Types**: Shape-indexed tensors with compile-time checking

## Core Types

| Type | Description |
|------|-------------|
| `Ty` | The main type representation |
| `TyVar` | Type variables for polymorphism |
| `TyCon` | Type constructors with name and kind |
| `Kind` | Kinds classify types |
| `Scheme` | Polymorphic type schemes |
| `Constraint` | Type class constraints |
| `Subst` | Type variable substitutions |
| `PrimTy` | Unboxed primitive types |

## Quick Start

```rust
use bhc_types::{Ty, TyVar, TyCon, Kind, Scheme};
use bhc_intern::Symbol;

// Create a type variable
let a = TyVar::new_star(0);  // t0 :: *

// Create a function type: a -> a
let id_ty = Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()));

// Create a type scheme: forall a. a -> a
let scheme = Scheme::poly(vec![a], id_ty);
```

## Type Representation

### Ty Variants

```rust
pub enum Ty {
    /// Type variable: `a`, `t0`
    Var(TyVar),

    /// Type constructor: `Int`, `Maybe`
    Con(TyCon),

    /// Unboxed primitive: `Int#`, `Double#`
    Prim(PrimTy),

    /// Type application: `Maybe Int`
    App(Box<Ty>, Box<Ty>),

    /// Function type: `a -> b`
    Fun(Box<Ty>, Box<Ty>),

    /// Tuple type: `(Int, Bool)`
    Tuple(Vec<Ty>),

    /// List type: `[Int]`
    List(Box<Ty>),

    /// Forall type: `forall a. a -> a`
    Forall(Vec<TyVar>, Box<Ty>),

    /// Error type for recovery
    Error,

    // M9 Dependent Types
    /// Type-level natural: `1024`
    Nat(TyNat),

    /// Type-level list: `'[1024, 768]`
    TyList(TyList),
}
```

### Type Variables

```rust
pub struct TyVar {
    /// Unique identifier
    pub id: u32,
    /// Kind of this variable
    pub kind: Kind,
}

impl TyVar {
    // Create with kind *
    pub fn new_star(id: u32) -> Self;

    // Create with custom kind
    pub fn new(id: u32, kind: Kind) -> Self;
}
```

### Type Constructors

```rust
pub struct TyCon {
    /// Name (interned symbol)
    pub name: Symbol,
    /// Kind
    pub kind: Kind,
}

// Examples:
// Int :: *
// Maybe :: * -> *
// Either :: * -> * -> *
// Tensor :: [Nat] -> * -> *
```

## Kinds

Kinds classify types:

```rust
pub enum Kind {
    /// `*` - Proper types (have values)
    Star,

    /// `k1 -> k2` - Type constructors
    Arrow(Box<Kind>, Box<Kind>),

    /// `Constraint` - Type class constraints
    Constraint,

    /// Kind variable (for inference)
    Var(u32),

    // M9 Extensions
    /// `Nat` - Type-level naturals
    Nat,

    /// `List k` - Type-level lists
    List(Box<Kind>),
}
```

### Kind Examples

```rust
// Int :: *
Kind::Star

// Maybe :: * -> *
Kind::star_to_star()

// Either :: * -> * -> *
Kind::Arrow(
    Box::new(Kind::Star),
    Box::new(Kind::star_to_star())
)

// Tensor :: [Nat] -> * -> *
Kind::tensor_kind()
```

## Type Schemes

Polymorphic types with constraints:

```rust
pub struct Scheme {
    /// Bound type variables
    pub vars: Vec<TyVar>,
    /// Type class constraints
    pub constraints: Vec<Constraint>,
    /// The underlying type
    pub ty: Ty,
}

// forall a. a -> a
let id_scheme = Scheme::poly(vec![a], id_ty);

// forall a. Eq a => a -> a -> Bool
let eq_scheme = Scheme::qualified(
    vec![a.clone()],
    vec![Constraint::new(eq_class, Ty::Var(a.clone()), span)],
    Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a), bool_ty))
);

// Int (monomorphic)
let mono = Scheme::mono(int_ty);
```

## Constraints

Type class constraints:

```rust
pub struct Constraint {
    /// Class name (e.g., "Eq", "Num")
    pub class: Symbol,
    /// Type arguments
    pub args: Vec<Ty>,
    /// Source location
    pub span: Span,
}

// Eq a
Constraint::new(eq, Ty::Var(a), span)

// Functor f
Constraint::new(functor, Ty::Var(f), span)
```

## Substitutions

Type variable substitutions:

```rust
let mut subst = Subst::new();

// Map t0 to Int
subst.insert(&a, Ty::Con(int_con));

// Apply to a type
let ty = Ty::fun(Ty::Var(a.clone()), Ty::Var(a));
let result = subst.apply(&ty);
// Result: Int -> Int

// Compose substitutions
let combined = subst1.compose(&subst2);
```

## Unboxed Primitive Types

For the Numeric Profile:

```rust
pub enum PrimTy {
    I32,   // Int32#
    I64,   // Int#, Int64#
    U32,   // Word32#
    U64,   // Word#, Word64#
    F32,   // Float#
    F64,   // Double#
    Char,  // Char#
    Addr,  // Addr#
}

// Create primitive types
let int_prim = Ty::int_prim();     // Int#
let double = Ty::double_prim();    // Double#
let float = Ty::float_prim();      // Float#

// Properties
PrimTy::I64.size_bytes()     // 8
PrimTy::I64.alignment()      // 8
PrimTy::I64.name()           // "Int#"
PrimTy::F64.is_float()       // true
PrimTy::I64.is_numeric()     // true
```

## M9 Dependent Types Preview

### Type-Level Naturals

```rust
pub enum TyNat {
    /// Literal: `1024`
    Lit(u64),
    /// Variable: `n`
    Var(TyVar),
    /// Addition: `m + n`
    Add(Box<TyNat>, Box<TyNat>),
    /// Multiplication: `m * n`
    Mul(Box<TyNat>, Box<TyNat>),
}

// Create type-level naturals
let dim = TyNat::lit(1024);
let sum = TyNat::add(TyNat::lit(100), TyNat::lit(200));
```

### Type-Level Lists

```rust
pub enum TyList {
    /// Empty list: `'[]`
    Nil,
    /// Cons: `x ': xs`
    Cons(Box<Ty>, Box<TyList>),
    /// Variable
    Var(TyVar),
    /// Append: `xs ++ ys`
    Append(Box<TyList>, Box<TyList>),
}

// Shape for a 1024x768 matrix
let shape = TyList::shape_from_dims(&[1024, 768]);
// '[1024, 768]
```

### Tensor Types

```rust
// Tensor '[1024, 768] Float
let matrix_ty = Ty::App(
    Box::new(Ty::App(
        Box::new(Ty::Con(tensor_con)),
        Box::new(Ty::shape(&[1024, 768])),
    )),
    Box::new(Ty::Con(float_con)),
);

// matmul signature:
// Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a
```

## Type Utilities

### Free Variables

```rust
let ty = Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()));
let free = ty.free_vars();  // [a, b]
```

### Ground Types

```rust
// A type with no unification variables
let int_ty = Ty::Con(int_con);
assert!(int_ty.is_ground());

let poly = Ty::Var(TyVar::new_star(0));
assert!(!poly.is_ground());
```

### Display

```rust
let ty = Ty::fun(Ty::Var(a), Ty::Var(b));
println!("{}", ty);  // "(t0 -> t1)"

let forall = Ty::Forall(vec![a], Box::new(Ty::Var(a)));
println!("{}", forall);  // "forall t0. t0"
```

## Type Errors

```rust
pub enum TypeError {
    /// Type mismatch
    Mismatch { expected: String, found: String, span: Span },

    /// Infinite type (occurs check)
    OccursCheck { var: String, ty: String, span: Span },

    /// Unbound type variable
    UnboundVar { name: String, span: Span },

    /// Kind mismatch
    KindMismatch { expected: String, found: String, span: Span },

    /// Ambiguous type variable
    Ambiguous { var: String, span: Span },
}
```

## Integration

Types flow through the compiler:

```
Parser → AST (surface types)
         ↓
HIR Lowering → HIR (resolved types)
         ↓
Type Checker → Typed HIR (inferred Ty)
         ↓
Core Lowering → Core IR (explicit Ty)
         ↓
Codegen → LLVM/WASM
```

## Performance

- `Ty` uses `Box` for recursive cases to control size
- `TyVar` is 8 bytes (id + kind reference)
- `PrimTy` is 1 byte (enum discriminant)
- Substitution uses `FxHashMap` for fast lookup
- Types are cloned during inference (arena would help)

## See Also

- `bhc-typeck`: Type inference algorithm
- `bhc-hir`: Uses these types for typed HIR
- `bhc-core`: Uses these types for Core IR
- H26-SPEC Section 4: Type System Specification
- H26-SPEC Section 7: Tensor Model
