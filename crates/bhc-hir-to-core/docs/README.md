# bhc-hir-to-core

HIR to Core IR lowering for the Basel Haskell Compiler.

## Overview

This crate implements the second lowering pass in BHC's compilation pipeline, transforming typed HIR into Core IR. This is where Haskell's high-level constructs become explicit:

- **Type class desugaring**: Dictionary passing style
- **Pattern compilation**: Decision trees
- **Explicit types**: Type applications and abstractions
- **Coercion insertion**: Newtype and role coercions

## Pipeline Position

```
[HIR]         ← Typed, high-level
    ↓
[TypeCheck]   ← Infer types, resolve constraints
    ↓
[HIR→Core]    ← This crate: desugar classes, compile patterns
    ↓
[Core IR]     ← Typed, explicit, optimizable
```

## Core Transformations

### Dictionary Passing

Type class constraints become explicit dictionary parameters:

```haskell
-- HIR (implicit)
show :: Show a => a -> String
show x = ...

-- Core (explicit dictionary)
show :: forall a. ShowDict a -> a -> String
show @a dict x = ...
```

```rust
fn lower_constrained_type(
    &mut self,
    constraints: &[Constraint],
    ty: Ty,
) -> (Vec<Var>, Ty) {
    let dict_params: Vec<Var> = constraints
        .iter()
        .map(|c| self.fresh_dict_var(c))
        .collect();

    let dict_tys: Vec<Ty> = constraints
        .iter()
        .map(|c| self.dict_type(c))
        .collect();

    let result_ty = dict_tys.into_iter()
        .rfold(ty, |acc, dict_ty| Ty::Fun(dict_ty, acc));

    (dict_params, result_ty)
}
```

### Instance Resolution

```haskell
-- Source
print (Just 42)

-- After dictionary passing
print @(Maybe Int) (showMaybeDict @Int showIntDict) (Just @Int 42)
```

```rust
fn resolve_instance(
    &mut self,
    constraint: &Constraint,
) -> Result<Expr, InstanceError> {
    // Try to find matching instance
    let candidates = self.instances.lookup(constraint.class, &constraint.ty);

    match candidates.as_slice() {
        [] => Err(InstanceError::NoInstance(constraint.clone())),
        [instance] => {
            // Build dictionary from instance
            self.build_dictionary(instance, constraint)
        }
        _ => Err(InstanceError::Overlapping(constraint.clone(), candidates)),
    }
}

fn build_dictionary(
    &mut self,
    instance: &Instance,
    constraint: &Constraint,
) -> Result<Expr> {
    // Substitute type arguments
    let subst = self.match_instance(instance, constraint)?;

    // Recursively resolve superclass and method constraints
    let super_dicts: Vec<Expr> = instance.super_constraints
        .iter()
        .map(|c| self.resolve_instance(&c.apply(&subst)))
        .collect::<Result<_>>()?;

    // Build dictionary expression
    Ok(Expr::App(
        Expr::TyApp(Expr::Var(instance.dict_fun), constraint.ty.clone()),
        super_dicts,
    ))
}
```

### Pattern Compilation

Complex patterns become efficient decision trees:

```haskell
-- Source
f (Just (x:xs)) = ...
f (Just [])     = ...
f Nothing       = ...

-- Compiled
f arg = case arg of
  Just tmp1 -> case tmp1 of
    (:) x xs -> ...
    []       -> ...
  Nothing   -> ...
```

```rust
fn compile_patterns(
    &mut self,
    clauses: Vec<Clause>,
    scrutinee: Expr,
) -> Expr {
    // Build decision tree
    let tree = PatternCompiler::new()
        .with_clauses(clauses)
        .compile();

    // Convert to nested case expressions
    self.decision_tree_to_case(tree, scrutinee)
}

struct PatternCompiler {
    matrix: PatternMatrix,
}

impl PatternCompiler {
    fn compile(mut self) -> DecisionTree {
        if self.matrix.is_empty() {
            return DecisionTree::Fail;
        }

        if self.matrix.first_row_all_wildcards() {
            return DecisionTree::Leaf(self.matrix.first_action());
        }

        // Pick column to split on
        let col = self.select_column();

        // Get constructors in this column
        let ctors = self.matrix.constructors_at(col);

        // Build switch node
        let branches: Vec<_> = ctors.iter()
            .map(|ctor| {
                let specialized = self.matrix.specialize(col, ctor);
                (ctor.clone(), PatternCompiler { matrix: specialized }.compile())
            })
            .collect();

        let default = if self.needs_default(col) {
            Some(Box::new(self.compile_default(col)))
        } else {
            None
        };

        DecisionTree::Switch { col, branches, default }
    }
}
```

### Newtype Coercions

```haskell
-- Source
newtype Age = Age Int

toAge :: Int -> Age
toAge = Age

-- Core (with coercion)
toAge :: Int -> Age
toAge = \x -> x |> (Sym (CoAge))  -- Zero-cost coercion
```

```rust
fn lower_newtype_con(&mut self, con: &DataCon) -> Expr {
    let inner_ty = con.fields[0].ty.clone();
    let outer_ty = con.result_ty.clone();

    // Newtype constructor is just a coercion
    let coercion = Coercion::Newtype {
        con: con.id,
        inner: inner_ty.clone(),
        outer: outer_ty.clone(),
    };

    Expr::Lam(
        self.fresh_var("x", inner_ty),
        Box::new(Expr::Cast(
            Box::new(Expr::Var(self.current_var())),
            coercion,
        ))
    )
}
```

### Type Applications

```haskell
-- Source
id True

-- Core (explicit type application)
id @Bool True
```

```rust
fn lower_app(&mut self, fun: hir::Expr, arg: hir::Expr) -> Expr {
    let fun_ty = fun.ty.as_ref().expect("typed HIR");

    // Insert type applications for polymorphic functions
    let (ty_args, _) = self.instantiate_foralls(fun_ty);

    let core_fun = self.lower_expr(fun);
    let core_arg = self.lower_expr(arg);

    // Apply type arguments first
    let applied = ty_args.into_iter()
        .fold(core_fun, |f, ty| Expr::TyApp(Box::new(f), ty));

    Expr::App(Box::new(applied), Box::new(core_arg))
}
```

## Core Types

```rust
pub struct LowerContext<'a> {
    /// Type environment
    ty_env: &'a TypeEnv,

    /// Instance environment
    instances: &'a InstanceEnv,

    /// Current function being lowered
    current_fun: Option<FunId>,

    /// Fresh variable supply
    var_supply: VarSupply,

    /// Accumulated Core bindings
    bindings: Vec<CoreBind>,
}
```

## Lowering API

```rust
/// Lower a typed HIR module to Core
pub fn lower_module(
    ty_env: &TypeEnv,
    instances: &InstanceEnv,
    module: hir::TypedModule,
) -> Result<CoreModule, LowerError> {
    let mut ctx = LowerContext::new(ty_env, instances);

    for decl in module.decls {
        ctx.lower_decl(decl)?;
    }

    Ok(ctx.into_module())
}
```

## Optimizations During Lowering

### Inlining Trivial Dictionaries

```rust
fn maybe_inline_dict(&mut self, dict: Expr) -> Expr {
    match &dict {
        Expr::Var(v) if self.is_trivial_dict(v) => {
            self.inline_dict(v)
        }
        _ => dict,
    }
}
```

### Specializing Known Calls

```rust
fn lower_app_known(&mut self, fun: &KnownFun, args: Vec<Expr>) -> Expr {
    // If we know the function and all dict args, specialize
    if let Some(specialized) = self.try_specialize(fun, &args) {
        return specialized;
    }

    // Otherwise, normal application
    self.build_apps(Expr::Var(fun.var), args)
}
```

## Invariants

After lowering to Core:

1. **No type class constraints** - All resolved to dictionaries
2. **No pattern matching in lambdas** - All compiled to case
3. **Explicit type applications** - All polymorphic calls annotated
4. **A-Normal Form** - Complex expressions are let-bound

## See Also

- `bhc-hir` - HIR types
- `bhc-core` - Core IR types
- `bhc-typeck` - Type checking (produces typed HIR)
- `bhc-lower` - AST to HIR lowering
