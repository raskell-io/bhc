# XMonad Compilation Progress

**Last Updated:** 2026-01-18
**Target:** Compile and run XMonad with BHC

## Current Status: BLOCKED - Parser Issues

XMonad source files are located at `/tmp/xmonad/src/XMonad/` (cloned from github.com/xmonad/xmonad).

## File-by-File Status

| File | Parsing Issues | Lowering | Type Check | Execution |
|------|----------------|----------|------------|-----------|
| ManageHook.hs | ~~152~~ 134 | blocked | blocked | blocked |
| Layout.hs | ~~256~~ 246 | blocked | blocked | blocked |
| Config.hs | ~~350~~ 322 | blocked | blocked | blocked |
| StackSet.hs | ~~402~~ 360 | blocked | blocked | blocked |
| Main.hs | ~~794~~ 730 | blocked | blocked | blocked |
| Core.hs | ~~928~~ 840 | blocked | blocked | blocked |
| Operations.hs | ~~1,808~~ 1,572 | blocked | blocked | blocked |
| **TOTAL** | ~~4,690~~ **4,204** | - | - | - |

## Parsing Issues by Category

| Priority | Issue | Count | Description | Status |
|----------|-------|-------|-------------|--------|
| P0 | Layout rules | 466 | Semicolons, braces in layout | TODO |
| P0 | Parentheses | 345 | Expression grouping in complex contexts | TODO |
| P1 | `<-` bindings | 174 | Do-notation, list comprehensions | TODO |
| P1 | Doc comments | ~~149~~ 8 | Haddock `-- \|` style comments | DONE (141 fixed) |
| P1 | Qualified names | ~~136~~ 46 | `Module.identifier` references | PARTIAL (90 fixed) |
| P1 | `->` arrows | 85 | Function types, case branches | TODO |
| P2 | Backtick operators | ~38 | `` `elem` `` infix application | TODO |
| P2 | `where` clauses | 14 | Where clause indentation/attachment | TODO |
| P2 | Operators | 107 | Operator parsing in various contexts | TODO |

## Recent Fixes

### 1. Haddock Documentation Comments - FIXED
```haskell
-- | This is a doc comment  -- NOW WORKS
foo :: Int -> Int
foo x = x
```
**Fixed in:** `crates/bhc-parser/src/decl.rs` - Added `skip_doc_comments()` calls

### 2. Qualified Names in Types - FIXED
```haskell
foo :: M.Map Int String -> M.Map Int String  -- NOW WORKS
```
**Fixed in:** `crates/bhc-parser/src/types.rs` and `crates/bhc-ast/src/lib.rs`
- Added `Type::QualCon(ModuleName, Ident, Span)` variant
- Added `QualConId` token handling in type parser

### 3. Qualified Names in Expressions - FIXED
```haskell
bar = M.size (M.empty)  -- NOW WORKS
```
**Fixed in:** `crates/bhc-parser/src/expr.rs` and `crates/bhc-ast/src/lib.rs`
- Added `Expr::QualVar(ModuleName, Ident, Span)` and `Expr::QualCon(ModuleName, Ident, Span)` variants
- Added `QualIdent` and `QualConId` token handling in expression parser

### 4. Infix Operator Bindings - FIXED
```haskell
p --> f = p >>= \b -> if b then f else return mempty  -- NOW WORKS
```
**Fixed in:** `crates/bhc-parser/src/decl.rs`
- Added `is_infix_op_start()` and `parse_infix_op()` helpers
- Modified `parse_value_decl()` to detect infix bindings

## Specific Syntax Patterns Still Failing

### 1. Where Clauses with Indentation
```haskell
shiftWin n w s = case findTag w s of
                    Just from | guard -> go from s
                    _                 -> s
 where go from = ...  -- FAILS: "unexpected where"
```
**Fix needed in:** `crates/bhc-parser/src/decl.rs`

### 2. Do-Notation
```haskell
do x <- getLine  -- FAILS: "unexpected <-"
   print x
```
**Fix needed in:** `crates/bhc-parser/src/expr.rs`

### 3. Case with Guards
```haskell
case x of
  Just y | y > 0 -> ...  -- May have issues with guard parsing
```
**Fix needed in:** `crates/bhc-parser/src/expr.rs` or `pattern.rs`

### 4. Multi-level Qualified Names
```haskell
foo = Data.List.sort xs  -- May fail with deep qualification
```
**Note:** Single-level qualification (M.foo) now works

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
- Progress: 4,690 -> 4,204 errors (486 fixed, ~10% reduction)
