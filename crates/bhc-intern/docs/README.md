# bhc-intern

String interning for efficient symbol handling.

## Overview

`bhc-intern` provides interned strings (symbols) that enable:

- **O(1) equality comparisons**: Compare integers instead of strings
- **Reduced memory**: Each unique string stored once
- **Fast hashing**: Pre-computed hash from interning
- **Global deduplication**: Same string always gets same symbol

## Core Types

| Type | Description |
|------|-------------|
| `Symbol` | An interned string (4 bytes) |
| `Ident` | An identifier with a name symbol |
| `kw::*` | Pre-interned keyword symbols |

## Quick Start

```rust
use bhc_intern::{Symbol, Ident};

// Intern a string
let sym = Symbol::intern("foo");

// Same string = same symbol
let sym2 = Symbol::intern("foo");
assert_eq!(sym, sym2);  // O(1) comparison

// Get the string back
assert_eq!(sym.as_str(), "foo");

// From string literals
let sym: Symbol = "bar".into();
```

## Symbol

### Creating Symbols

```rust
use bhc_intern::Symbol;

// Explicit interning
let sym = Symbol::intern("hello");

// From &str
let sym: Symbol = "world".into();

// From String
let s = String::from("test");
let sym: Symbol = s.into();
```

### Using Symbols

```rust
let sym = Symbol::intern("example");

// Get the string
let s: &'static str = sym.as_str();

// Get raw index (for serialization)
let idx: u32 = sym.as_u32();

// Check properties
assert!(!sym.is_empty());
assert_eq!(sym.len(), 7);

// Compare with strings directly
assert!(sym == "example");
```

### Symbol Comparison

```rust
let a = Symbol::intern("apple");
let b = Symbol::intern("banana");
let c = Symbol::intern("apple");

// Equality is O(1) - just compares indices
assert_eq!(a, c);
assert_ne!(a, b);

// Ordering uses string comparison
assert!(a < b);  // "apple" < "banana"
```

## Ident

Identifiers wrap symbols with additional semantics:

```rust
use bhc_intern::Ident;

// Create from string
let id = Ident::from_str("myFunction");

// Create from symbol
let sym = Symbol::intern("myFunction");
let id = Ident::new(sym);

// Access the name
assert_eq!(id.as_str(), "myFunction");
assert_eq!(id.name, sym);
```

## Pre-interned Keywords

Common keywords are pre-interned for fast lookup:

```rust
use bhc_intern::kw;

// Access pre-interned keywords
let case_kw = *kw::CASE;      // "case"
let class_kw = *kw::CLASS;    // "class"
let data_kw = *kw::DATA;      // "data"
let where_kw = *kw::WHERE;    // "where"

// Check if a symbol is a keyword
fn is_keyword(sym: Symbol) -> bool {
    sym == *kw::CASE || sym == *kw::CLASS || sym == *kw::DATA
    // ... etc
}

// Initialize all keywords at startup (optional, for performance)
kw::intern_all();
```

### Available Keywords

**Haskell Keywords:**
- `CASE`, `CLASS`, `DATA`, `DEFAULT`, `DERIVING`
- `DO`, `ELSE`, `FORALL`, `FOREIGN`, `IF`
- `IMPORT`, `IN`, `INFIX`, `INFIXL`, `INFIXR`
- `INSTANCE`, `LET`, `MODULE`, `NEWTYPE`, `OF`
- `QUALIFIED`, `THEN`, `TYPE`, `WHERE`

**BHC/H26 Extensions:**
- `LAZY`, `STRICT`, `PROFILE`, `EDITION`

**Common Types:**
- `INT`, `FLOAT`, `DOUBLE`, `BOOL`, `CHAR`, `STRING`, `UNIT`

**Common Constructors:**
- `TRUE`, `FALSE`, `JUST`, `NOTHING`, `LEFT`, `RIGHT`

**Special:**
- `UNDERSCORE` (`_`)

## Thread Safety

The interner is thread-safe using `RwLock`:

```rust
use bhc_intern::Symbol;
use std::thread;

let handles: Vec<_> = (0..10).map(|i| {
    thread::spawn(move || {
        // Safe to intern from multiple threads
        Symbol::intern(&format!("thread_{}", i))
    })
}).collect();

for h in handles {
    let sym = h.join().unwrap();
    println!("{}", sym);
}
```

## Serialization

Symbols serialize as their raw index:

```rust
use bhc_intern::Symbol;
use serde_json;

let sym = Symbol::intern("test");

// Serialize
let json = serde_json::to_string(&sym).unwrap();

// Deserialize
let sym2: Symbol = serde_json::from_str(&json).unwrap();
assert_eq!(sym, sym2);
```

## Use in AST

Typical usage in AST definitions:

```rust
use bhc_intern::{Symbol, Ident};
use bhc_span::Span;

struct Name {
    ident: Ident,
    span: Span,
}

enum Expr {
    Var(Name),
    Lit(Literal),
    App(Box<Expr>, Box<Expr>),
    Lam(Name, Box<Expr>),
}

impl Expr {
    fn free_vars(&self) -> HashSet<Symbol> {
        match self {
            Expr::Var(name) => {
                let mut set = HashSet::new();
                set.insert(name.ident.name);  // O(1) insert
                set
            }
            // ...
        }
    }
}
```

## Performance Notes

| Operation | Complexity |
|-----------|------------|
| `Symbol::intern` (new) | O(n) where n = string length |
| `Symbol::intern` (existing) | O(n) lookup + O(1) return |
| `Symbol::as_str` | O(1) |
| `Symbol == Symbol` | O(1) |
| `Symbol < Symbol` | O(n) string comparison |
| `Symbol` size | 4 bytes |

## Memory Model

```
┌─────────────────────────────────────────┐
│ Global Interner (LazyLock<Interner>)    │
├─────────────────────────────────────────┤
│ map: FxHashMap<&'static str, Symbol>    │
│ strings: Vec<&'static str>              │
│                                         │
│ strings[0] = "case"     → Symbol(0)     │
│ strings[1] = "class"    → Symbol(1)     │
│ strings[2] = "myFunc"   → Symbol(2)     │
│ ...                                     │
└─────────────────────────────────────────┘

Symbol(2) ─────────────────→ "myFunc"
```

Note: Interned strings are leaked (`Box::leak`) and live for the program's lifetime.
