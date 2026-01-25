# BHC Examples

Simple examples demonstrating BHC v0.1.0-alpha capabilities.

## Prerequisites

Make sure BHC is installed and in your PATH:
```bash
export PATH="$HOME/.bhc/bin:$PATH"
export LIBRARY_PATH="$HOME/.bhc/lib:$LIBRARY_PATH"
```

## Examples

### hello.hs
The classic Hello World program.
```bash
bhc hello.hs -o hello
./hello
# Output: Hello from BHC!
```

### arithmetic.hs
Basic arithmetic operations.
```bash
bhc arithmetic.hs -o arithmetic
./arithmetic
# Output: 7
```

### let-bindings.hs
Let expressions and where clauses.
```bash
bhc let-bindings.hs -o let-bindings
./let-bindings
# Output: 30
```

## Compile All Examples

```bash
for f in *.hs; do
    name="${f%.hs}"
    echo "Compiling $f..."
    bhc "$f" -o "$name"
done
```

## More Information

- [Get Started Guide](https://bhc.raskell.io/get-started/)
- [FAQ](https://bhc.raskell.io/faq/)
- [GitHub Repository](https://github.com/raskell-io/bhc)
