# bhc-intern

String interning for efficient symbol handling in the Basel Haskell Compiler.

## Overview

This crate provides interned strings (symbols) that enable O(1) equality comparisons and reduced memory usage for repeated strings. A global interner ensures that each unique string is stored only once.

## Key Types

| Type | Description |
|------|-------------|
| `Symbol` | An interned string, cheap to copy and compare |
| `Ident` | An identifier with a name symbol |
| `kw::*` | Pre-interned keywords for common Haskell identifiers |

## Usage

```rust
use bhc_intern::{Symbol, Ident, kw};

// Intern a string
let s1 = Symbol::intern("hello");
let s2 = Symbol::intern("hello");

// O(1) comparison (just compares indices)
assert_eq!(s1, s2);

// Get the string value
assert_eq!(s1.as_str(), "hello");

// Use pre-interned keywords
kw::intern_all(); // Optional: pre-intern for better performance
assert_eq!(*kw::LET, "let");
assert_eq!(*kw::WHERE, "where");

// Create identifiers
let id = Ident::from_str("myFunction");
println!("{}", id); // prints: myFunction
```

## Pre-interned Keywords

The `kw` module provides pre-interned symbols for common Haskell keywords:

**Haskell Keywords**: `case`, `class`, `data`, `deriving`, `do`, `else`, `forall`, `foreign`, `if`, `import`, `in`, `infix`, `infixl`, `infixr`, `instance`, `let`, `module`, `newtype`, `of`, `qualified`, `then`, `type`, `where`

**BHC Extensions**: `lazy`, `strict`, `profile`, `edition`

**Common Types**: `Int`, `Float`, `Double`, `Bool`, `Char`, `String`, `()`

**Common Constructors**: `True`, `False`, `Just`, `Nothing`, `Left`, `Right`

## Performance

- **Interning**: O(1) average case (hash lookup), O(n) for new strings
- **Comparison**: O(1) - just compares integer indices
- **Memory**: Each unique string stored once, symbols are just 4 bytes

## Thread Safety

The global interner is thread-safe using `RwLock`. The fast path (already interned) only requires a read lock.

## Design Notes

- Symbols are `Copy` and very cheap to pass around
- The interner leaks memory intentionally (strings live forever)
- Use `kw::intern_all()` at startup for predictable performance

## Related Crates

- `bhc-ast` - Uses symbols for identifiers in the AST
- `bhc-hir` - Uses symbols for names in HIR
- `bhc-core` - Uses symbols for variable names in Core IR
