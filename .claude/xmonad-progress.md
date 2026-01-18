# XMonad Compilation Progress

**Last Updated:** 2026-01-18
**Target:** Compile and run XMonad with BHC

## Current Status: BLOCKED - Parser Issues

XMonad source files are located at `/tmp/xmonad/src/XMonad/` (cloned from github.com/xmonad/xmonad).

## File-by-File Status

| File | Parsing Issues | Lowering | Type Check | Execution |
|------|----------------|----------|------------|-----------|
| ManageHook.hs | 152 | blocked | blocked | blocked |
| Layout.hs | 262 | blocked | blocked | blocked |
| Config.hs | 350 | blocked | blocked | blocked |
| StackSet.hs | 408 | blocked | blocked | blocked |
| Main.hs | 794 | blocked | blocked | blocked |
| Core.hs | 960 | blocked | blocked | blocked |
| Operations.hs | 1,808 | blocked | blocked | blocked |
| **TOTAL** | **4,734** | - | - | - |

## Parsing Issues by Category

| Priority | Issue | Count | Description | Status |
|----------|-------|-------|-------------|--------|
| P0 | Layout rules | 499 | Semicolons, braces in layout | TODO |
| P0 | Parentheses | 362 | Expression grouping in complex contexts | TODO |
| P0 | Operators | 304 | Operator parsing in various contexts | TODO |
| P1 | `<-` bindings | 175 | Do-notation, list comprehensions | TODO |
| P1 | Doc comments | 149 | Haddock `-- \|` style comments | TODO |
| P1 | Qualified names | 91 | `Module.identifier` references | TODO |
| P1 | `->` arrows | 85 | Function types, case branches | TODO |
| P2 | Backtick operators | 38 | `` `elem` `` infix application | TODO |
| P2 | `where` clauses | 15 | Where clause indentation/attachment | TODO |

## Specific Syntax Patterns Failing

### 1. Haddock Documentation Comments
```haskell
-- | This is a doc comment  -- FAILS: "unexpected documentation comment"
foo :: Int -> Int
foo x = x
```
**Fix needed in:** `crates/bhc-parser/src/lexer.rs` or `decl.rs`

### 2. Where Clauses with Indentation
```haskell
shiftWin n w s = case findTag w s of
                    Just from | guard -> go from s
                    _                 -> s
 where go from = ...  -- FAILS: "unexpected where"
```
**Fix needed in:** `crates/bhc-parser/src/decl.rs`

### 3. Qualified Names in Types
```haskell
foo :: W.StackSet i l a s sd -> ...  -- FAILS: "unexpected qualified constructor"
```
**Fix needed in:** `crates/bhc-parser/src/types.rs`

### 4. Do-Notation
```haskell
do x <- getLine  -- FAILS: "unexpected <-"
   print x
```
**Fix needed in:** `crates/bhc-parser/src/expr.rs`

### 5. Case with Guards
```haskell
case x of
  Just y | y > 0 -> ...  -- May have issues with guard parsing
```
**Fix needed in:** `crates/bhc-parser/src/expr.rs` or `pattern.rs`

## Dependencies Required

XMonad requires these external modules (import resolution not yet implemented):

### XMonad Internal
- XMonad.Core
- XMonad.StackSet
- XMonad.Operations

### Standard Library
- Control.Monad.Reader (ask, lift)
- Control.Exception (bracket, catch)
- Data.Maybe (fromMaybe, listToMaybe)
- Data.Monoid (mempty, mconcat, mappend)

### X11 Bindings
- Graphics.X11.Xlib
- Graphics.X11.Xlib.Extras

## Milestones

### M1: Parse All XMonad Files (0/7)
- [ ] ManageHook.hs parses without errors
- [ ] Layout.hs parses without errors
- [ ] Config.hs parses without errors
- [ ] StackSet.hs parses without errors
- [ ] Main.hs parses without errors
- [ ] Core.hs parses without errors
- [ ] Operations.hs parses without errors

### M2: Lower All XMonad Files (0/7)
- [ ] All files lower to HIR (requires import stubs)

### M3: Type Check XMonad (0/7)
- [ ] All files type check (requires Prelude + X11 type stubs)

### M4: Execute XMonad
- [ ] Main.main evaluates without runtime errors

## Test Commands

```bash
# Clone XMonad (if not present)
git clone --depth 1 https://github.com/xmonad/xmonad.git /tmp/xmonad

# Test single file parsing
cargo run -p bhc -- build /tmp/xmonad/src/XMonad/ManageHook.hs 2>&1 | grep "unexpected" | wc -l

# Test all files
for f in /tmp/xmonad/src/XMonad/*.hs; do
  errors=$(cargo run -p bhc -- build "$f" 2>&1 | grep -c "unexpected")
  echo "$(basename $f): $errors parsing issues"
done

# Categorize errors
cargo run -p bhc -- build /tmp/xmonad/src/XMonad/*.hs 2>&1 | \
  grep "unexpected" | sed 's/.*unexpected //' | sed 's/,.*//' | \
  sort | uniq -c | sort -rn | head -20
```

## Notes

- The VarId collision bug was fixed (fresh_counter now starts at 100)
- Basic pattern matching with multiple data types works
- Recursive functions work (fib, factorial tested)
- Import resolution is not implemented - all external refs fail at lowering
