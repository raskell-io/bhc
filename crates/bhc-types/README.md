# bhc-types

Type representation for the Basel Haskell Compiler.

## Overview

This crate implements the type system for BHC, including type representations, kinds, schemes, and substitutions. It provides the foundation for type inference and checking based on Hindley-Milner with extensions for higher-kinded types, type classes, and M9 dependent types preview.

## Features

- Hindley-Milner type inference foundation
- Higher-kinded types
- Type classes with functional dependencies
- Type families
- GADTs support
- Rank-N polymorphism (limited)
- **M9 Preview**: Shape-indexed tensors with compile-time dimension checking

## Key Types

| Type | Description |
|------|-------------|
| `Ty` | The main type representation |
| `TyVar` | Type variables for polymorphism |
| `TyId` | Unique type identifiers |
| `Kind` | Kinds for higher-kinded types |
| `Scheme` | Polymorphic type schemes |
| `Constraint` | Type class constraints |
| `Subst` | Type substitution |
| `PrimTy` | Unboxed primitive types |
| `TyNat` | Type-level natural numbers (M9) |
| `TyList` | Type-level lists for shapes (M9) |

## Usage

### Basic Types

```rust
use bhc_types::{Ty, TyVar, TyCon, Kind, Scheme};
use bhc_intern::Symbol;

// Create type variables
let a = TyVar::new_star(0);  // a :: *

// Create a function type: a -> a
let identity_ty = Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()));

// Create a polymorphic scheme: forall a. a -> a
let identity_scheme = Scheme::poly(vec![a], identity_ty);
```

### Kind System

```rust
use bhc_types::Kind;

// Basic kinds
let star = Kind::Star;           // *
let arrow = Kind::star_to_star(); // * -> *

// Higher-kinded types
let functor_kind = Kind::Arrow(
    Box::new(Kind::star_to_star()),
    Box::new(Kind::Constraint),
);

// M9: Tensor kinds
let nat = Kind::Nat;              // Nat
let shape_kind = Kind::nat_list(); // [Nat]
let tensor_kind = Kind::tensor_kind(); // [Nat] -> * -> *
```

### Type Substitution

```rust
use bhc_types::{Ty, TyVar, Subst};

let a = TyVar::new_star(0);

let mut subst = Subst::new();
subst.insert(&a, Ty::int_prim());

// Apply substitution to `a -> a`
let ty = Ty::fun(Ty::Var(a.clone()), Ty::Var(a));
let result = subst.apply(&ty);
// Result: Int# -> Int#
```

### Primitive Types (Numeric Profile)

```rust
use bhc_types::{Ty, PrimTy};

// Unboxed primitives for zero-overhead computation
let int = Ty::int_prim();      // Int# (64-bit signed)
let double = Ty::double_prim(); // Double# (64-bit float)
let float = Ty::float_prim();  // Float# (32-bit float)

// Check properties
assert!(PrimTy::I64.is_signed_int());
assert!(PrimTy::F64.is_float());
assert_eq!(PrimTy::I64.size_bytes(), 8);
```

### Shape-Indexed Tensors (M9)

```rust
use bhc_types::{Ty, TyNat, TyList};

// Type-level natural: 1024
let dim = Ty::nat_lit(1024);

// Shape: '[1024, 768]
let shape = Ty::shape(&[1024, 768]);

// Tensor type: Tensor '[m, k] Float
// for matrix multiplication
// matmul :: Tensor '[m, k] Float -> Tensor '[k, n] Float -> Tensor '[m, n] Float
```

## Type Variants

```rust
pub enum Ty {
    Var(TyVar),           // Type variable: a
    Con(TyCon),           // Type constructor: Int, Maybe
    Prim(PrimTy),         // Unboxed primitive: Int#, Double#
    App(Box<Ty>, Box<Ty>), // Type application: Maybe Int
    Fun(Box<Ty>, Box<Ty>), // Function type: a -> b
    Tuple(Vec<Ty>),       // Tuple: (a, b, c)
    List(Box<Ty>),        // List: [a]
    Forall(Vec<TyVar>, Box<Ty>), // Forall: forall a. a -> a
    Error,                // Error type for recovery

    // M9 Dependent Types Preview
    Nat(TyNat),           // Type-level natural: 1024
    TyList(TyList),       // Type-level list: '[1024, 768]
}
```

## Kind Variants

```rust
pub enum Kind {
    Star,                 // * - Types with values
    Arrow(Box<Kind>, Box<Kind>), // k1 -> k2 - Type constructors
    Constraint,           // Constraint kind
    Var(u32),             // Kind variable

    // M9 Dependent Types Preview
    Nat,                  // Nat - Type-level naturals
    List(Box<Kind>),      // [k] - Type-level lists
}
```

## Primitive Types

| Type | Haskell Name | Size | Description |
|------|--------------|------|-------------|
| `I32` | `Int32#` | 4 bytes | 32-bit signed integer |
| `I64` | `Int#` | 8 bytes | 64-bit signed integer |
| `U32` | `Word32#` | 4 bytes | 32-bit unsigned integer |
| `U64` | `Word#` | 8 bytes | 64-bit unsigned integer |
| `F32` | `Float#` | 4 bytes | 32-bit IEEE 754 float |
| `F64` | `Double#` | 8 bytes | 64-bit IEEE 754 double |
| `Char` | `Char#` | 4 bytes | Unicode code point |
| `Addr` | `Addr#` | 8 bytes | Machine pointer |

## Design Notes

- Types are immutable and hashable for efficient comparison
- Substitutions are lazily applied for performance
- The `Error` type enables recovery from type errors
- M9 type-level features prepare for dependent type support

## Related Crates

- `bhc-typeck` - Type inference and checking
- `bhc-hir` - High-level IR that uses these types
- `bhc-core` - Core IR with explicit type annotations
- `bhc-intern` - Symbol interning for type names

## Specification References

- H26-SPEC Section 4: Type System Specification
- H26-SPEC Section 6.2: Unboxed Type Requirements
- H26-SPEC Section 7: Tensor Model
