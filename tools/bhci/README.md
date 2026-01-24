# bhci

Interactive REPL for the Basel Haskell Compiler.

## Overview

bhci (Basel Haskell Compiler Interactive) provides an interactive environment for exploring Haskell 2026 code. It supports expression evaluation, type inspection, and incremental compilation.

## Usage

```bash
# Start the REPL
bhci

# With a specific profile
bhci --profile=numeric

# Load a file on startup
bhci Main.hs
```

## Session Example

```
Basel Haskell Compiler Interactive (bhci)
Version 0.1.0
Type :help for help, :quit to exit

bhci:001> 2 + 3
5
bhci:002> let double x = x * 2
bhci:003> double 21
42
bhci:004> :type double
double :: Num a => a -> a
bhci:005> :quit

Goodbye!
```

## Commands

| Command | Description |
|---------|-------------|
| `:quit` / `:q` | Exit the REPL |
| `:help` / `:h` | Show help |
| `:type <expr>` / `:t` | Show type of expression |
| `:kind <type>` / `:k` | Show kind of type |
| `:info <name>` / `:i` | Show information about name |
| `:load <file>` / `:l` | Load a Haskell file |
| `:reload` / `:r` | Reload current file |
| `:browse` / `:b` | Browse loaded modules |
| `:set <option>` | Set REPL option |
| `:unset <option>` | Unset REPL option |

## Options

```
bhci:001> :set +t        # Show types after evaluation
bhci:002> :set +s        # Show timing/stats
bhci:003> :set -Wall     # Enable all warnings
bhci:004> :unset +t      # Disable type display
```

## Multiline Input

```
bhci:001> :{
bhci:002| let
bhci:003|   fib 0 = 0
bhci:004|   fib 1 = 1
bhci:005|   fib n = fib (n-1) + fib (n-2)
bhci:006| :}
bhci:007> fib 10
55
```

## Type Inspection

```
bhci:001> :type map
map :: (a -> b) -> [a] -> [b]

bhci:002> :kind Maybe
Maybe :: * -> *

bhci:003> :info Functor
class Functor (f :: * -> *) where
  fmap :: (a -> b) -> f a -> f b
  (<$) :: a -> f b -> f a
```

## Debugging

```
bhci:001> :set -ddump-ir   # Show IR after evaluation
bhci:002> :step            # Step through evaluation
bhci:003> :trace           # Show evaluation trace
```

## Profile Switching

```
bhci:001> :set profile numeric
Switching to Numeric profile (strict evaluation)

bhci:002> :set profile default
Switching to Default profile (lazy evaluation)
```

## Design Notes

- Incremental compilation for fast feedback
- Preserves bindings across evaluations
- Supports all Haskell 2026 features
- Profile-aware evaluation

## Related Tools

- `bhc` - Compiler CLI
- `bhi` - IR inspector

## Specification References

- H26-SPEC Section 1: Language Overview
