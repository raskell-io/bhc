# bhc-hir

High-Level Intermediate Representation for the Basel Haskell Compiler.

## Overview

`bhc-hir` is the bridge between surface syntax (AST) and typed Core IR. HIR preserves:

- **Name resolution results**: All identifiers resolved to definitions
- **Pattern matching structure**: For exhaustiveness checking
- **Module organization**: Imports, exports, items
- **Source locations**: For error reporting

## IR Pipeline Position

```
Source Code
    ↓
[Parse/AST]  ← Surface syntax, concrete
    ↓
[HIR]        ← This crate: desugared, resolved
    ↓
[Core IR]    ← Typed, explicit, optimizable
    ↓
[Tensor IR]  ← Numeric optimizations
    ↓
[Loop IR]    ← Iteration, vectorization
```

## Core Types

| Type | Description |
|------|-------------|
| `Module` | A complete HIR module |
| `Item` | Top-level declarations |
| `Expr` | HIR expressions |
| `Pat` | HIR patterns |
| `HirId` | Unique identifier for nodes |
| `DefId` | Unique identifier for definitions |
| `DefRef` | Reference to a definition |

## Module Structure

```rust
pub struct Module {
    /// Module name (interned)
    pub name: Symbol,
    /// Export list (None = export all)
    pub exports: Option<Vec<Export>>,
    /// Import declarations
    pub imports: Vec<Import>,
    /// Top-level items
    pub items: Vec<Item>,
    /// Source span
    pub span: Span,
}
```

## Items

```rust
pub enum Item {
    /// Function/value definition
    Value(ValueDef),
    /// Data type: `data Maybe a = Nothing | Just a`
    Data(DataDef),
    /// Newtype: `newtype Id a = Id a`
    Newtype(NewtypeDef),
    /// Type synonym: `type String = [Char]`
    TypeAlias(TypeAlias),
    /// Type class
    Class(ClassDef),
    /// Type class instance
    Instance(InstanceDef),
    /// Fixity declaration
    Fixity(FixityDecl),
    /// Foreign import
    Foreign(ForeignDecl),
}
```

## Expressions

HIR expressions are desugared from surface syntax:

```rust
pub enum Expr {
    /// Literal: `42`, `"hello"`
    Lit(Lit, Span),
    /// Variable reference
    Var(DefRef),
    /// Data constructor
    Con(DefRef),
    /// Application: `f x`
    App(Box<Expr>, Box<Expr>, Span),
    /// Lambda: `\x -> e`
    Lam(Vec<Pat>, Box<Expr>, Span),
    /// Let binding: `let x = e1 in e2`
    Let(Vec<Binding>, Box<Expr>, Span),
    /// Case expression
    Case(Box<Expr>, Vec<CaseAlt>, Span),
    /// If expression
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),
    /// Tuple: `(a, b)`
    Tuple(Vec<Expr>, Span),
    /// List: `[a, b]`
    List(Vec<Expr>, Span),
    /// Record construction
    Record(DefRef, Vec<FieldExpr>, Span),
    /// Field access: `r.field`
    FieldAccess(Box<Expr>, Symbol, Span),
    /// Record update
    RecordUpdate(Box<Expr>, Vec<FieldExpr>, Span),
    /// Type annotation: `e :: ty`
    Ann(Box<Expr>, Ty, Span),
    /// Type application: `e @ty`
    TypeApp(Box<Expr>, Ty, Span),
    /// Error placeholder
    Error(Span),
}
```

### Desugaring

Do-notation and list comprehensions are desugared to HIR:

```haskell
-- Source:
do
  x <- getLine
  print x

-- HIR (desugared):
getLine >>= \x -> print x
```

```haskell
-- Source:
[x * 2 | x <- xs, even x]

-- HIR (desugared):
concatMap (\x -> if even x then [x * 2] else []) xs
```

## Patterns

```rust
pub enum Pat {
    /// Wildcard: `_`
    Wild(Span),
    /// Variable: `x`
    Var(Symbol, Span),
    /// Literal: `42`
    Lit(Lit, Span),
    /// Constructor: `Just x`
    Con(DefRef, Vec<Pat>, Span),
    /// As-pattern: `x@pat`
    As(Symbol, Box<Pat>, Span),
    /// Or-pattern: `pat1 | pat2`
    Or(Box<Pat>, Box<Pat>, Span),
    /// Type annotation: `pat :: ty`
    Ann(Box<Pat>, Ty, Span),
    /// Error placeholder
    Error(Span),
}
```

### Bound Variables

```rust
let pat = Pat::Con(just_ref, vec![Pat::Var(x, span)], span);
let vars = pat.bound_vars();  // [x]
```

## Definitions

### Value Definitions

```rust
pub struct ValueDef {
    /// Unique ID
    pub id: DefId,
    /// Name
    pub name: Symbol,
    /// Optional type signature
    pub sig: Option<Scheme>,
    /// Defining equations
    pub equations: Vec<Equation>,
    pub span: Span,
}

pub struct Equation {
    /// Function argument patterns
    pub pats: Vec<Pat>,
    /// Guards
    pub guards: Vec<Guard>,
    /// Right-hand side
    pub rhs: Expr,
    pub span: Span,
}
```

### Data Types

```rust
pub struct DataDef {
    pub id: DefId,
    pub name: Symbol,
    pub params: Vec<TyVar>,
    pub cons: Vec<ConDef>,
    pub deriving: Vec<Symbol>,
    pub span: Span,
}

pub struct ConDef {
    pub id: DefId,
    pub name: Symbol,
    pub fields: ConFields,
    pub span: Span,
}

pub enum ConFields {
    /// Positional: `MkFoo Int String`
    Positional(Vec<Ty>),
    /// Named: `MkFoo { x :: Int, y :: String }`
    Named(Vec<FieldDef>),
}
```

### Type Classes

```rust
pub struct ClassDef {
    pub id: DefId,
    pub name: Symbol,
    pub params: Vec<TyVar>,
    pub supers: Vec<Symbol>,
    pub methods: Vec<MethodSig>,
    pub defaults: Vec<ValueDef>,
    pub span: Span,
}

pub struct InstanceDef {
    pub class: Symbol,
    pub types: Vec<Ty>,
    pub constraints: Vec<Symbol>,
    pub methods: Vec<ValueDef>,
    pub span: Span,
}
```

## Definition References

After name resolution, all names become `DefRef`:

```rust
pub struct DefRef {
    /// The definition being referenced
    pub def_id: DefId,
    /// Source span of this reference
    pub span: Span,
}

// Example: variable `x` resolved
Expr::Var(DefRef {
    def_id: DefId(42),
    span: span_of_x,
})
```

## Imports and Exports

```rust
pub struct Import {
    pub module: Symbol,
    pub qualified: bool,
    pub alias: Option<Symbol>,
    pub items: Option<Vec<ImportItem>>,
    pub hiding: bool,
    pub span: Span,
}

pub struct Export {
    pub name: Symbol,
    pub children: ExportChildren,
    pub span: Span,
}

pub enum ExportChildren {
    None,         // Just the type
    All,          // Type(..)
    Some(Vec<Symbol>), // Type(A, B)
}
```

## Foreign Declarations

```rust
pub struct ForeignDecl {
    pub id: DefId,
    pub name: Symbol,
    pub foreign_name: Symbol,
    pub convention: ForeignConvention,
    pub ty: Scheme,
    pub span: Span,
}

pub enum ForeignConvention {
    CCall,
    StdCall,
    JavaScript,
}
```

## Identifiers

```rust
// HirId: Unique ID for every HIR node
pub struct HirId(u32);

// DefId: Unique ID for definitions
pub struct DefId(u32);

// Both implement Idx for IndexVec usage
impl Idx for HirId { ... }
impl Idx for DefId { ... }
```

## Error Recovery

HIR supports error nodes for recovery:

```rust
// Expression error
Expr::Error(span)

// Pattern error
Pat::Error(span)

// Check for errors
if expr.is_error() {
    // Handle error case
}
```

## Serialization

HIR types derive `Serialize` and `Deserialize`:

```rust
let module: Module = ...;
let json = serde_json::to_string(&module)?;
```

## Integration

HIR is produced by lowering AST and consumed by type checking:

```
AST → [HIR Lowering] → HIR → [Type Check] → Typed HIR → [Core Lower] → Core IR
```

## See Also

- `bhc-ast`: Surface syntax AST
- `bhc-core`: Typed Core IR
- `bhc-types`: Type representations
- `bhc-typeck`: Type checking
