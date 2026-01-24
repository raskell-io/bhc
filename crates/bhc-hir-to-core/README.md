# bhc-hir-to-core

HIR to Core IR lowering for the Basel Haskell Compiler.

## Overview

This crate transforms typed HIR (High-Level IR) into Core IR, the main intermediate representation used for optimization. It handles pattern compilation, dictionary passing for type classes, and explicit binding analysis.

## Pipeline Position

```
[HIR]           ← Desugared, resolved
    │
    ▼
[Type Check]    ← Type inference
    │
    ▼
[HIR-to-Core]   ← THIS CRATE
    │
    ▼
[Core IR]       ← Typed, explicit, optimizable
    │
    ▼
[Optimize]      ← Simplification, inlining
```

## Key Transformations

- **Pattern compilation**: Multi-argument lambdas and pattern matching compiled to explicit case expressions
- **Binding analysis**: Let bindings analyzed for mutual recursion
- **Guard expansion**: Pattern guards become nested conditionals
- **Dictionary passing**: Type class constraints compiled to dictionary parameters
- **Deriving**: Automatic instance generation for standard classes

## Key Types

| Type | Description |
|------|-------------|
| `LowerContext` | Lowering context with variable and class information |
| `DefMap` | Map from DefId to definition information |
| `TypeSchemeMap` | Map from DefId to type schemes |
| `ClassRegistry` | Registry of type classes and instances |

## Usage

### Basic Module Lowering

```rust
use bhc_hir_to_core::lower_module;
use bhc_hir::Module as HirModule;

let hir_module: HirModule = type_check(...)?;
let core_module = lower_module(&hir_module)?;

// Access lowered bindings
for bind in &core_module.bindings {
    match bind {
        Bind::NonRec(var, expr) => {
            println!("Binding: {} = ...", var.name);
        }
        Bind::Rec(pairs) => {
            println!("Recursive group with {} bindings", pairs.len());
        }
    }
}
```

### With Type Schemes

```rust
use bhc_hir_to_core::{lower_module_with_defs, DefMap, TypeSchemeMap};

// Provide type schemes from type checker for dictionary passing
let type_schemes: TypeSchemeMap = typeck_result.schemes;
let def_map: DefMap = lower_result.def_map;

let core_module = lower_module_with_defs(
    &hir_module,
    Some(&def_map),
    Some(&type_schemes),
)?;
```

## Dictionary Passing

Type class constraints are compiled to dictionary parameters:

```haskell
-- Source
f :: Num a => a -> a
f x = x + x

-- After HIR-to-Core (conceptually)
f = \$dNum -> \x -> (+) $dNum x x
```

### Dictionary Resolution

```rust
use bhc_hir_to_core::LowerContext;

let mut ctx = LowerContext::new();

// Register a dictionary in scope
ctx.push_dict_scope();
ctx.register_dict(Symbol::intern("Num"), num_dict_var);

// Resolve a constraint
let constraint = Constraint::new(
    Symbol::intern("Num"),
    int_ty,
    span,
);
let dict_expr = ctx.resolve_dictionary(&constraint, span);

ctx.pop_dict_scope();
```

### Superclass Extraction

When a superclass dictionary is needed, it's extracted from a subclass dictionary:

```haskell
-- Source: Ord has Eq as superclass
f :: Ord a => a -> a -> Bool
f x y = x == y   -- needs Eq, but we have Ord

-- Compilation extracts Eq from Ord dictionary
f = \$dOrd -> \x y -> (==) (sel_Eq $dOrd) x y
```

## Pattern Compilation

Multi-clause functions are compiled to case expressions:

```haskell
-- Source
map f []     = []
map f (x:xs) = f x : map f xs

-- After compilation
map = \f -> \arg0 -> case arg0 of
    []    -> []
    x:xs  -> f x : map f xs
```

### Guards

Pattern guards become nested conditionals:

```haskell
-- Source
abs x | x < 0     = negate x
      | otherwise = x

-- After compilation
abs = \x -> case x < 0 of
    True  -> negate x
    False -> x
```

## Class and Instance Registration

```rust
// Classes are registered during lowering
let registry = ctx.class_registry();

// Query class information
if let Some(class_info) = registry.lookup_class(Symbol::intern("Eq")) {
    println!("Methods: {:?}", class_info.methods);
    println!("Superclasses: {:?}", class_info.superclasses);
}

// Resolve instance
let result = registry.resolve_instance(Symbol::intern("Eq"), &int_ty);
if let Some((instance_info, subst)) = result {
    println!("Found instance with {} methods", instance_info.methods.len());
}
```

## Deriving

Automatic instance derivation is supported:

```rust
use bhc_hir_to_core::deriving;

// Generate Eq instance for a data type
let eq_instance = deriving::derive_eq(&data_def)?;

// Generate Show instance
let show_instance = deriving::derive_show(&data_def)?;
```

## Error Types

```rust
pub enum LowerError {
    /// Internal invariant violated
    Internal(String),

    /// Pattern compilation failed
    PatternError { message: String, span: Span },

    /// Multiple errors occurred
    Multiple(Vec<LowerError>),
}
```

## Modules

| Module | Description |
|--------|-------------|
| `binding` | Binding analysis and grouping |
| `context` | Lowering context and variable management |
| `deriving` | Automatic instance derivation |
| `dictionary` | Dictionary passing and class resolution |
| `expr` | Expression lowering |
| `pattern` | Pattern compilation |

## Built-in Classes

The following classes are pre-registered:

| Class | Methods | Superclasses |
|-------|---------|--------------|
| `Eq` | `==`, `/=` | - |
| `Ord` | `compare`, `<`, `>`, `<=`, `>=`, `min`, `max` | `Eq` |
| `Num` | `+`, `-`, `*`, `negate`, `abs`, `signum`, `fromInteger` | - |
| `Fractional` | `/`, `recip`, `fromRational` | `Num` |
| `Show` | `show`, `showsPrec`, `showList` | - |

## Built-in Instances

Pre-registered instances for primitive types:

- `Eq`: Int, Float, Double, Char, Bool
- `Ord`: Int, Float, Double, Char, Bool
- `Num`: Int, Float, Double
- `Fractional`: Float, Double
- `Show`: Int, Float, Double, Char, Bool

## Design Notes

- All types are explicit in Core (no inference needed)
- Dictionary parameters are prepended to function arguments
- Superclass dictionaries are extracted via selectors
- Pattern match compilation uses decision trees
- Default methods are lifted to top-level bindings

## Related Crates

- `bhc-hir` - Input HIR types
- `bhc-core` - Output Core IR
- `bhc-types` - Type representation
- `bhc-typeck` - Type checking (produces type schemes)

## Specification References

- H26-SPEC Section 3.3: Core IR Definition
- H26-SPEC Section 4.4: Dictionary Passing
- H26-SPEC Section 4.5: Pattern Compilation
