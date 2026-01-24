# bhc-lower

AST to HIR lowering for the Basel Haskell Compiler.

## Overview

This crate implements the first lowering pass in BHC's compilation pipeline, transforming the parsed AST into High-level IR (HIR). This pass handles:

- **Desugaring**: Syntactic sugar to core constructs
- **Name resolution**: Binding identifiers to definitions
- **Scope analysis**: Tracking variable scopes and lifetimes
- **Import resolution**: Resolving module imports

## Pipeline Position

```
Source Code
    ↓
[Parse/AST]  ← Surface syntax (bhc-parser)
    ↓
[Lower]      ← This crate: desugar + resolve
    ↓
[HIR]        ← Desugared, resolved (bhc-hir)
    ↓
[TypeCheck]  ← Type inference (bhc-typeck)
```

## Core Types

| Type | Description |
|------|-------------|
| `LowerContext` | Lowering state and environment |
| `Resolver` | Name resolution engine |
| `Scope` | Variable scope tracking |
| `Desugar` | Desugaring transformations |

## Lowering Context

```rust
pub struct LowerContext<'a> {
    /// The session for configuration
    session: &'a Session,

    /// Diagnostic emitter
    diag: &'a DiagnosticEmitter,

    /// Interner for symbols
    interner: &'a Interner,

    /// Current module being lowered
    current_module: Option<ModuleId>,

    /// Scope stack
    scopes: Vec<Scope>,

    /// Generated HIR items
    items: Vec<HirItem>,
}
```

## Desugaring Transformations

### Do-Notation

```haskell
-- Source
do
  x <- getLine
  putStrLn x

-- Desugared
getLine >>= \x -> putStrLn x
```

```rust
fn desugar_do(stmts: Vec<DoStmt>) -> Expr {
    stmts.into_iter().rev().fold(
        last_expr,
        |acc, stmt| match stmt {
            DoStmt::Bind(pat, expr) =>
                Expr::App(
                    Expr::App(Expr::Var(sym!(">>="), bind),
                    expr),
                    Expr::Lam(pat, acc)
                ),
            DoStmt::Let(binds) =>
                Expr::Let(binds, acc),
            DoStmt::Expr(e) =>
                Expr::App(
                    Expr::App(Expr::Var(sym!(">>"), then),
                    e),
                    acc
                ),
        }
    )
}
```

### List Comprehensions

```haskell
-- Source
[x * 2 | x <- xs, x > 0]

-- Desugared
concatMap (\x -> if x > 0 then [x * 2] else []) xs
```

### Guards

```haskell
-- Source
f x
  | x > 0     = "positive"
  | x < 0     = "negative"
  | otherwise = "zero"

-- Desugared
f x = case () of
  _ | x > 0     -> "positive"
    | x < 0     -> "negative"
    | otherwise -> "zero"
```

### Pattern Guards

```haskell
-- Source
f x | Just y <- lookup x env = y
    | otherwise = defaultValue

-- Desugared
f x = case lookup x env of
  Just y -> y
  _ -> defaultValue
```

### Where Clauses

```haskell
-- Source
f x = y + z
  where
    y = x * 2
    z = x * 3

-- Desugared
f x = let y = x * 2
          z = x * 3
      in y + z
```

### Record Syntax

```haskell
-- Source
data Point = Point { x :: Int, y :: Int }

p { x = 10 }  -- Record update

-- Desugared
case p of
  Point _ y -> Point 10 y
```

### Operator Sections

```haskell
-- Source
(+ 1)    -- Right section
(1 +)    -- Left section

-- Desugared
(\x -> x + 1)
(\x -> 1 + x)
```

## Name Resolution

### Scope Structure

```rust
pub struct Scope {
    /// Variables in this scope
    bindings: HashMap<Symbol, VarId>,

    /// Type variables in this scope
    ty_vars: HashMap<Symbol, TyVarId>,

    /// Parent scope (if any)
    parent: Option<ScopeId>,

    /// Scope kind
    kind: ScopeKind,
}

pub enum ScopeKind {
    Module,
    Function,
    Let,
    Lambda,
    Case,
    Pattern,
}
```

### Resolution Algorithm

```rust
impl Resolver {
    pub fn resolve_name(&self, name: &Name) -> Result<Resolution, ResolutionError> {
        // Try local scopes first (innermost to outermost)
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.lookup(name.symbol) {
                return Ok(Resolution::Local(binding));
            }
        }

        // Try module-level bindings
        if let Some(def) = self.current_module.lookup(name.symbol) {
            return Ok(Resolution::Module(def));
        }

        // Try imported names
        if let Some(import) = self.imports.lookup(name) {
            return Ok(Resolution::Imported(import));
        }

        Err(ResolutionError::UnboundName(name.clone()))
    }
}
```

### Import Handling

```rust
pub fn resolve_import(&mut self, import: &Import) -> Result<(), ImportError> {
    let module = self.load_interface(import.module)?;

    match &import.spec {
        ImportSpec::All => {
            // import Foo
            for export in module.exports() {
                self.imports.insert(export.name, export);
            }
        }
        ImportSpec::Explicit(names) => {
            // import Foo (bar, baz)
            for name in names {
                let export = module.lookup_export(name)?;
                self.imports.insert(name, export);
            }
        }
        ImportSpec::Hiding(names) => {
            // import Foo hiding (bar)
            for export in module.exports() {
                if !names.contains(&export.name) {
                    self.imports.insert(export.name, export);
                }
            }
        }
    }

    if import.qualified {
        // import qualified Foo as F
        let qualifier = import.alias.unwrap_or(import.module);
        self.qualified_imports.insert(qualifier, module);
    }

    Ok(())
}
```

## Diagnostic Messages

### Name Resolution Errors

```
error[E0412]: cannot find value `foo` in this scope
  --> src/Main.hs:10:5
   |
10 |     foo + 1
   |     ^^^ not found in this scope
   |
   = help: did you mean `fob`?
```

### Shadowing Warnings

```
warning: name `x` shadows a binding from outer scope
  --> src/Main.hs:15:10
   |
12 | f x =
   |   - first binding of `x`
...
15 |     let x = 10
   |         ^ shadows outer `x`
```

## Lowering API

```rust
/// Lower a parsed module to HIR
pub fn lower_module(
    session: &Session,
    diag: &DiagnosticEmitter,
    module: ast::Module,
) -> Result<hir::Module, LowerError> {
    let mut ctx = LowerContext::new(session, diag);

    // Process imports first
    for import in &module.imports {
        ctx.resolve_import(import)?;
    }

    // Lower declarations
    for decl in module.decls {
        ctx.lower_decl(decl)?;
    }

    Ok(ctx.into_module())
}

/// Lower a single expression (for REPL)
pub fn lower_expr(
    session: &Session,
    diag: &DiagnosticEmitter,
    expr: ast::Expr,
) -> Result<hir::Expr, LowerError> {
    let mut ctx = LowerContext::new(session, diag);
    ctx.lower_expr(expr)
}
```

## Implementation Notes

### Preserving Source Locations

All lowered constructs preserve their source spans for error reporting:

```rust
fn lower_expr(&mut self, expr: ast::Expr) -> Result<hir::Expr> {
    let span = expr.span();
    let kind = match expr.kind {
        ast::ExprKind::Var(name) => {
            let res = self.resolve_name(&name)?;
            hir::ExprKind::Var(res)
        }
        // ...
    };
    Ok(hir::Expr { kind, span, ty: None })
}
```

### Handling Recursion

Recursive bindings require special handling:

```rust
fn lower_let_bindings(&mut self, binds: Vec<ast::Bind>) -> Result<Vec<hir::Bind>> {
    // First pass: add all names to scope
    for bind in &binds {
        let var_id = self.fresh_var(bind.name);
        self.current_scope().insert(bind.name, var_id);
    }

    // Second pass: lower RHS with all names in scope
    binds.into_iter()
        .map(|bind| self.lower_bind(bind))
        .collect()
}
```

## See Also

- `bhc-ast` - AST types
- `bhc-hir` - HIR types
- `bhc-hir-to-core` - HIR to Core lowering
- `bhc-typeck` - Type checking
