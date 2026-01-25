# BHC Examples

Example programs demonstrating BHC's Phase 2 features.

## Running Examples

```bash
# Compile to native executable
bhc hello.hs -o hello
./hello

# Or use bhc run (interpreter mode)
bhc run hello.hs
```

## Examples

| File | Output | Description |
|------|--------|-------------|
| `hello.hs` | `Hello, World!` | Basic I/O with putStrLn |
| `arithmetic.hs` | `13` | Math operations with precedence |
| `functions.hs` | `18` | Function definitions, higher-order functions |
| `factorial.hs` | `3628800` | Recursive factorial (10!) |
| `fibonacci.hs` | `610` | Recursive Fibonacci (fib 15) |
| `let-bindings.hs` | `30` | Local variable bindings |
| `tuples.hs` | `42` | Tuple construction and access |

## Features Demonstrated

- **Native Code Generation**: All examples compile to standalone executables
- **Arithmetic**: Integer operations with correct precedence
- **Functions**: First-class functions, higher-order functions
- **Recursion**: Self-recursive functions
- **Let Bindings**: Local variable scoping
- **Tuples**: Pair construction with `fst` and `snd`
- **Conditionals**: `if-then-else` expressions
- **Top-level Bindings**: Module-level value definitions
