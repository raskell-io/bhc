# bhc-ast

Abstract Syntax Tree definitions for Haskell 2026.

## Overview

`bhc-ast` defines the concrete syntax tree produced by parsing. This is a direct representation of source syntax before any desugaring or name resolution.

Key characteristics:

- **Source-faithful**: Preserves syntactic details (parentheses, comments)
- **Spanned**: Every node has source location
- **Extensible**: Supports H26 extensions and M9 dependent types
- **Visitable**: Implements visitor pattern for traversals

## Core Types

| Type | Description |
|------|-------------|
| `Module` | A complete source module |
| `Decl` | Top-level declarations |
| `Expr` | Expressions |
| `Pat` | Patterns |
| `Type` | Type expressions |
| `Name` | Identifiers with span |

## Module Structure

```rust
pub struct Module {
    /// Module header: `module Foo.Bar where`
    pub header: Option<ModuleHeader>,
    /// Import declarations
    pub imports: Vec<ImportDecl>,
    /// Top-level declarations
    pub decls: Vec<Decl>,
    /// File-level span
    pub span: Span,
}

pub struct ModuleHeader {
    /// Module name
    pub name: ModuleName,
    /// Export list (None = export all)
    pub exports: Option<Vec<Export>>,
    /// Span of the header
    pub span: Span,
}

pub struct ModuleName {
    /// Qualified parts: ["Data", "List"]
    pub parts: Vec<Ident>,
    pub span: Span,
}
```

## Declarations

```rust
pub enum Decl {
    /// Type signature: `foo :: Int -> Int`
    TypeSig(TypeSig),

    /// Function binding: `foo x = x + 1`
    FunBind(FunBind),

    /// Pattern binding: `(x, y) = pair`
    PatBind(PatBind),

    /// Data type: `data Maybe a = Nothing | Just a`
    DataDecl(DataDecl),

    /// Newtype: `newtype Id a = Id a`
    NewtypeDecl(NewtypeDecl),

    /// Type synonym: `type String = [Char]`
    TypeDecl(TypeDecl),

    /// Type class: `class Eq a where ...`
    ClassDecl(ClassDecl),

    /// Instance: `instance Eq Int where ...`
    InstanceDecl(InstanceDecl),

    /// Foreign import/export
    ForeignDecl(ForeignDecl),

    /// Fixity declaration: `infixl 6 +`
    FixityDecl(FixityDecl),

    /// Default declaration: `default (Int, Double)`
    DefaultDecl(DefaultDecl),

    /// Deriving declaration (standalone)
    DerivingDecl(DerivingDecl),

    /// GADT: `data T a where ...` (H26)
    GadtDecl(GadtDecl),

    /// Type family: `type family F a` (H26)
    TypeFamilyDecl(TypeFamilyDecl),

    /// Data family: `data family D a` (H26)
    DataFamilyDecl(DataFamilyDecl),
}
```

### Function Bindings

```rust
pub struct FunBind {
    /// Function name
    pub name: Ident,
    /// Equations (one per line in source)
    pub matches: Vec<Match>,
    pub span: Span,
}

pub struct Match {
    /// Patterns for this equation
    pub pats: Vec<Pat>,
    /// Right-hand side
    pub rhs: Rhs,
    /// Local bindings
    pub where_clause: Option<WhereClause>,
    pub span: Span,
}

pub enum Rhs {
    /// Simple: `= expr`
    Unguarded(Expr),
    /// Guarded: `| cond = expr`
    Guarded(Vec<GuardedRhs>),
}
```

### Data Declarations

```rust
pub struct DataDecl {
    /// Type constructor name
    pub name: Ident,
    /// Type parameters
    pub ty_vars: Vec<TyVar>,
    /// Data constructors
    pub constrs: Vec<ConDecl>,
    /// Deriving clause
    pub deriving: Vec<DerivingClause>,
    pub span: Span,
}

pub enum ConDecl {
    /// Prefix: `Just a`
    Prefix {
        name: Ident,
        args: Vec<Type>,
    },
    /// Infix: `a :+: b`
    Infix {
        left: Type,
        op: Ident,
        right: Type,
    },
    /// Record: `Person { name :: String, age :: Int }`
    Record {
        name: Ident,
        fields: Vec<FieldDecl>,
    },
}
```

## Expressions

```rust
pub enum Expr {
    /// Variable: `x`
    Var(Name),

    /// Constructor: `Just`
    Con(Name),

    /// Literal: `42`, `"hello"`
    Lit(Literal),

    /// Application: `f x`
    App(Box<Expr>, Box<Expr>),

    /// Infix application: `x + y`
    InfixApp(Box<Expr>, Name, Box<Expr>),

    /// Lambda: `\x -> x + 1`
    Lam(Vec<Pat>, Box<Expr>),

    /// Let: `let x = 1 in x + 1`
    Let(Vec<Decl>, Box<Expr>),

    /// If: `if c then t else f`
    If(Box<Expr>, Box<Expr>, Box<Expr>),

    /// Case: `case x of { ... }`
    Case(Box<Expr>, Vec<Alt>),

    /// Do notation: `do { x <- m; return x }`
    Do(Vec<Stmt>),

    /// Tuple: `(a, b, c)`
    Tuple(Vec<Expr>),

    /// List: `[1, 2, 3]`
    List(Vec<Expr>),

    /// Arithmetic sequence: `[1..10]`, `[1,3..10]`
    ArithSeq(ArithSeqInfo),

    /// List comprehension: `[x | x <- xs, even x]`
    ListComp(Box<Expr>, Vec<Stmt>),

    /// Record construction: `Person { name = "Alice" }`
    RecordCon(Name, Vec<FieldBind>),

    /// Record update: `p { age = 30 }`
    RecordUpd(Box<Expr>, Vec<FieldBind>),

    /// Typed expression: `x :: Int`
    Typed(Box<Expr>, Type),

    /// Negation: `-x`
    Negate(Box<Expr>),

    /// Parenthesized: `(x + y)`
    Paren(Box<Expr>),

    /// Section: `(+ 1)`, `(1 +)`
    Section(Section),

    /// Lambda-case: `\case { ... }` (H26)
    LamCase(Vec<Alt>),

    /// Multi-way if: `if | c1 -> e1 | c2 -> e2` (H26)
    MultiIf(Vec<GuardedRhs>),

    /// Type application: `show @Int` (H26)
    TypeApp(Box<Expr>, Type),

    /// Dependent pair: `(x : T ** P x)` (M9)
    DepPair(Box<Pat>, Type, Box<Expr>),
}
```

### Statements (Do/Comprehension)

```rust
pub enum Stmt {
    /// Expression statement: `print x`
    Expr(Expr),

    /// Bind: `x <- getLine`
    Bind(Pat, Expr),

    /// Let: `let x = 1`
    Let(Vec<Decl>),

    /// Guard (in comprehensions): `even x`
    Guard(Expr),
}
```

### Case Alternatives

```rust
pub struct Alt {
    /// Pattern to match
    pub pat: Pat,
    /// Right-hand side
    pub rhs: Rhs,
    /// Local bindings
    pub where_clause: Option<WhereClause>,
    pub span: Span,
}
```

## Patterns

```rust
pub enum Pat {
    /// Variable: `x`
    Var(Ident),

    /// Wildcard: `_`
    Wildcard,

    /// Literal: `42`
    Lit(Literal),

    /// Constructor: `Just x`
    Con(Name, Vec<Pat>),

    /// Infix constructor: `x : xs`
    InfixCon(Box<Pat>, Name, Box<Pat>),

    /// Tuple: `(a, b)`
    Tuple(Vec<Pat>),

    /// List: `[a, b, c]`
    List(Vec<Pat>),

    /// As-pattern: `xs@(x:_)`
    As(Ident, Box<Pat>),

    /// Parenthesized: `(p)`
    Paren(Box<Pat>),

    /// Record pattern: `Person { name = n }`
    Record(Name, Vec<PatField>),

    /// Typed pattern: `x :: Int`
    Typed(Box<Pat>, Type),

    /// Lazy pattern: `~p`
    Lazy(Box<Pat>),

    /// Bang pattern: `!x` (H26)
    Bang(Box<Pat>),

    /// View pattern: `f -> p` (H26)
    View(Expr, Box<Pat>),

    /// Or-pattern: `p1 | p2` (H26)
    Or(Vec<Pat>),
}
```

## Types

```rust
pub enum Type {
    /// Type variable: `a`
    Var(Ident),

    /// Type constructor: `Int`, `Maybe`
    Con(Name),

    /// Application: `Maybe Int`
    App(Box<Type>, Box<Type>),

    /// Function: `a -> b`
    Fun(Box<Type>, Box<Type>),

    /// Tuple: `(Int, Bool)`
    Tuple(Vec<Type>),

    /// List: `[Int]`
    List(Box<Type>),

    /// Parenthesized: `(a)`
    Paren(Box<Type>),

    /// Forall: `forall a. a -> a`
    Forall(Vec<TyVar>, Box<Type>),

    /// Qualified: `Eq a => a -> a -> Bool`
    Qualified(Vec<Constraint>, Box<Type>),

    /// Kind annotation: `a :: *`
    Kinded(Box<Type>, Kind),

    /// Type operator: `a :+: b` (H26)
    InfixApp(Box<Type>, Name, Box<Type>),

    /// Promoted data constructor: `'Just` (H26)
    Promoted(Name),

    /// Type literal: `"hello"`, `42` (H26)
    Lit(TyLit),

    /// Dependent function: `(x : A) -> B x` (M9)
    DepFun(Ident, Box<Type>, Box<Type>),

    /// Dependent pair: `(x : A ** B x)` (M9)
    DepPair(Ident, Box<Type>, Box<Type>),
}
```

### Constraints

```rust
pub enum Constraint {
    /// Class constraint: `Eq a`
    Class(Name, Vec<Type>),

    /// Type equality: `a ~ b` (H26)
    Equality(Type, Type),

    /// Implicit parameter: `?x :: Int` (H26)
    ImplicitParam(Ident, Type),
}
```

## Literals

```rust
pub enum Literal {
    /// Integer: `42`, `0xFF`
    Int(i128, IntBase),

    /// Float: `3.14`, `1e-10`
    Float(f64),

    /// Character: `'a'`
    Char(char),

    /// String: `"hello"`
    String(String),
}

pub enum IntBase {
    Decimal,
    Hexadecimal,
    Octal,
    Binary,
}
```

## Names and Identifiers

```rust
/// An identifier from source
pub struct Ident {
    pub name: Symbol,
    pub span: Span,
}

/// A possibly qualified name
pub struct Name {
    /// Qualifier: `Data.List`
    pub qualifier: Option<ModuleName>,
    /// Local name: `map`
    pub ident: Ident,
    pub span: Span,
}

impl Name {
    pub fn is_qualified(&self) -> bool {
        self.qualifier.is_some()
    }

    pub fn as_str(&self) -> &str {
        self.ident.name.as_str()
    }
}
```

## Visitor Pattern

```rust
pub trait Visitor: Sized {
    fn visit_module(&mut self, module: &Module) {
        walk_module(self, module);
    }

    fn visit_decl(&mut self, decl: &Decl) {
        walk_decl(self, decl);
    }

    fn visit_expr(&mut self, expr: &Expr) {
        walk_expr(self, expr);
    }

    fn visit_pat(&mut self, pat: &Pat) {
        walk_pat(self, pat);
    }

    fn visit_type(&mut self, ty: &Type) {
        walk_type(self, ty);
    }

    fn visit_name(&mut self, name: &Name) {
        // Default: do nothing
    }
}

// Example: collect all free variables
struct FreeVarCollector {
    free: HashSet<Symbol>,
    bound: HashSet<Symbol>,
}

impl Visitor for FreeVarCollector {
    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Var(name) => {
                if !self.bound.contains(&name.ident.name) {
                    self.free.insert(name.ident.name);
                }
            }
            Expr::Lam(pats, body) => {
                // Bind pattern variables
                for pat in pats {
                    self.visit_pat(pat);
                }
                self.visit_expr(body);
            }
            _ => walk_expr(self, expr),
        }
    }
}
```

## Pretty Printing

```rust
use bhc_ast::pretty::PrettyPrint;

let expr = Expr::App(
    Box::new(Expr::Var(name("map"))),
    Box::new(Expr::Var(name("f"))),
);

println!("{}", expr.pretty());  // "map f"
```

## Serialization

AST types derive `Serialize` and `Deserialize` for tooling:

```rust
use serde_json;

let module: Module = parse("module Main where\nmain = return ()")?;
let json = serde_json::to_string_pretty(&module)?;
```

## H26 Extensions

Extensions from Haskell 2026:

| Feature | AST Node |
|---------|----------|
| Lambda-case | `Expr::LamCase` |
| Multi-way if | `Expr::MultiIf` |
| Type applications | `Expr::TypeApp` |
| View patterns | `Pat::View` |
| Or-patterns | `Pat::Or` |
| Bang patterns | `Pat::Bang` |
| GADTs | `Decl::GadtDecl` |
| Type families | `Decl::TypeFamilyDecl` |
| Promoted types | `Type::Promoted` |
| Type literals | `Type::Lit` |

## M9 Dependent Types

Milestone 9 features for dependent typing:

| Feature | AST Node |
|---------|----------|
| Dependent functions | `Type::DepFun` |
| Dependent pairs | `Type::DepPair`, `Expr::DepPair` |

## Integration

The AST is produced by `bhc-parser` and consumed by `bhc-hir` (lowering):

```
Source → Lexer → Tokens → Parser → AST → HIR
```
