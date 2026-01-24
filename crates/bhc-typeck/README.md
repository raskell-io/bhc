# bhc-typeck

Type inference and checking for the Basel Haskell Compiler.

## Overview

This crate implements Hindley-Milner type inference (Algorithm W) for BHC. It operates on HIR (High-level Intermediate Representation) and produces typed HIR suitable for lowering to Core IR.

## Features

- **Let-polymorphism**: Types are generalized at let-bindings
- **Mutual recursion**: Binding groups analyzed via SCC decomposition
- **Type signatures**: User signatures checked against inferred types
- **Error recovery**: Inference continues after errors using error types
- **Kind checking**: Validates kinds of type constructors
- **Type class resolution**: Dictionary passing compilation
- **Shape inference**: M9 tensor dimension checking

## Key Types

| Type | Description |
|------|-------------|
| `TyCtxt` | Type checking context with type environment |
| `TypeEnv` | Environment mapping names to type schemes |
| `TypedModule` | Result of type checking with inferred types |
| `KindEnv` | Kind environment for kind checking |

## Usage

### Type Checking a Module

```rust
use bhc_typeck::type_check_module;
use bhc_hir::Module;
use bhc_span::FileId;

let file_id = FileId::new(0);
let result = type_check_module(&hir_module, file_id);

match result {
    Ok(typed_module) => {
        // Access inferred types for expressions
        for (hir_id, ty) in &typed_module.expr_types {
            println!("{:?}: {}", hir_id, ty);
        }

        // Access type schemes for definitions
        for (def_id, scheme) in &typed_module.def_schemes {
            println!("{:?}: {:?}", def_id, scheme);
        }
    }
    Err(diagnostics) => {
        // Report type errors
        for diag in diagnostics {
            eprintln!("{:?}", diag);
        }
    }
}
```

### With Definition Mappings

```rust
use bhc_typeck::type_check_module_with_defs;

// When you have definition mappings from the lowering pass
let result = type_check_module_with_defs(&hir_module, file_id, Some(&def_map));
```

## Algorithm

Type inference proceeds in several phases:

### 1. Binding Group Analysis

Identify mutually recursive groups via SCC (Strongly Connected Components):

```haskell
-- These form a binding group:
even 0 = True
even n = odd (n - 1)

odd 0 = False
odd n = even (n - 1)
```

### 2. Constraint Generation

Walk HIR and generate type constraints:

```
Γ ⊢ e₁ : τ₁ → τ₂    Γ ⊢ e₂ : τ₁
─────────────────────────────────  (App)
        Γ ⊢ e₁ e₂ : τ₂
```

### 3. Unification

Solve constraints via substitution:

```rust
// Unify τ₁ = τ₂ produces a substitution
unify(Int -> a, Int -> Bool) = { a ↦ Bool }
```

### 4. Generalization

Generalize types at let-bindings:

```haskell
-- Inferred type: a -> a
-- Generalized: forall a. a -> a
id x = x
```

## Error Messages

Type errors produce rich diagnostics:

```text
error[E0001]: type mismatch
 --> example.hs:5:10
   |
 5 |   add x = x + True
   |               ^^^^ expected Int, found Bool
   |
 = note: in the expression `x + True`
 = help: the (+) operator requires both operands to have the same numeric type
```

## Modules

| Module | Description |
|--------|-------------|
| `binding_groups` | SCC computation for mutual recursion |
| `builtins` | Built-in type definitions |
| `context` | Type checking context (`TyCtxt`) |
| `diagnostics` | Error reporting |
| `env` | Type environment |
| `generalize` | Type generalization |
| `infer` | Core inference algorithm |
| `instantiate` | Type instantiation |
| `kind_check` | Kind checking |
| `nat_solver` | Type-level natural solver (M9) |
| `pattern` | Pattern type checking |
| `shape_bridge` | Shape type utilities (M9) |
| `suggest` | Error suggestions |
| `type_families` | Type family reduction |
| `unify` | Type unification |

## Built-in Types

The type checker provides built-in definitions for:

- Primitive types: `Int`, `Float`, `Double`, `Char`, `Bool`
- Collections: `[]`, `Maybe`, `Either`
- Functions: `(->)`
- Tuples: `(,)`, `(,,)`, etc.
- Type classes: `Eq`, `Ord`, `Num`, `Show`

## Type Classes

Type class resolution compiles to dictionary passing:

```haskell
-- Source
show :: Show a => a -> String

-- After elaboration (conceptually)
show :: ShowDict a -> a -> String
```

## M9 Shape Checking

For tensor types, the type checker validates dimension compatibility:

```haskell
-- Type-safe matrix multiply
matmul :: Tensor '[m, k] Float -> Tensor '[k, n] Float -> Tensor '[m, n] Float

-- This type checks:
matmul (zeros [3, 4]) (zeros [4, 5])  -- Tensor '[3, 5] Float

-- This fails:
matmul (zeros [3, 4]) (zeros [5, 6])  -- Error: dimension mismatch
```

## Design Notes

- Uses Algorithm W with extensions for type classes
- Error recovery enables reporting multiple errors
- Binding groups ensure correct inference order
- Kind checking prevents ill-kinded types

## Related Crates

- `bhc-types` - Type representation
- `bhc-hir` - Input HIR types
- `bhc-lower` - AST to HIR lowering (produces input)
- `bhc-hir-to-core` - HIR to Core lowering (consumes output)
- `bhc-diagnostics` - Error reporting

## Specification References

- H26-SPEC Section 4: Type System
- H26-SPEC Section 4.3: Type Inference
- H26-SPEC Section 7: Tensor Model (M9)
