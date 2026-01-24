# bhc-playground

Browser-based Haskell interpreter for the BHC playground.

## Overview

This crate provides a WASM-compatible interface to the BHC frontend (lexer, parser, type checker) and the Core IR evaluator. It enables Haskell code to be validated and executed directly in the browser without native code generation.

## Architecture

```
Source Code
     │
     ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Parse  │ ──▶ │  Lower  │ ──▶ │  Type   │ ──▶ │  Core   │
│         │     │  (HIR)  │     │  Check  │     │  Lower  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                                                     │
                                                     ▼
                                                ┌─────────┐
                                                │  Eval   │
                                                │ (Interp)│
                                                └─────────┘
                                                     │
                                                     ▼
                                                  Result
```

## Rust Usage

```rust
use bhc_playground::{compile_and_run, PlaygroundResult};

let result = compile_and_run("main = 42");
match result {
    Ok(output) => println!("Result: {}", output.display),
    Err(e) => eprintln!("Error: {}", e.message),
}
```

## WASM Usage

```javascript
import init, { run_haskell } from 'bhc_playground';

await init();
const result = run_haskell("main = 42");
console.log(result);
// { "success": true, "display": "42", "type": "Int" }
```

## API

### `run_haskell(source: string): PlaygroundResult`

Compile and evaluate Haskell source code.

```typescript
interface PlaygroundResult {
  success: boolean;
  display?: string;     // Pretty-printed result
  type?: string;        // Inferred type
  ir?: string;          // Core IR (if requested)
  error?: PlaygroundError;
}

interface PlaygroundError {
  message: string;
  span?: SourceSpan;
  suggestions?: string[];
}
```

### `check_haskell(source: string): TypeCheckResult`

Type check without evaluation.

```typescript
interface TypeCheckResult {
  success: boolean;
  type?: string;
  errors?: PlaygroundError[];
}
```

### `format_haskell(source: string): string`

Format Haskell source code.

## Features

| Feature | Status |
|---------|--------|
| Expression evaluation | ✅ |
| Type inference | ✅ |
| Let bindings | ✅ |
| Pattern matching | ✅ |
| Type classes | ✅ |
| Custom types | ✅ |
| Module imports | 🔄 |
| IO actions | ❌ (sandboxed) |

## Building

```bash
# Build WASM package
wasm-pack build crates/bhc-playground --target web

# Build for Node.js
wasm-pack build crates/bhc-playground --target nodejs
```

## Deployment

The playground is deployed at https://play.bhc.raskell.io

```bash
# Deploy to playground
cd crates/bhc-playground
wasm-pack build --target web
cd www
npm run build
npm run deploy
```

## Limitations

- No native code generation (interpretation only)
- No FFI (WASM sandbox)
- No IO (except pure computations)
- Limited memory (WASM linear memory)
- Timeout for long-running computations

## Design Notes

- Uses BHC's real frontend (not a separate implementation)
- Core IR interpreter for evaluation
- WebAssembly for browser compatibility
- JSON-based API for easy integration
- Sandboxed execution for security

## Related Crates

- `bhc-parser` - Parsing
- `bhc-typeck` - Type checking
- `bhc-core` - Core IR and evaluator
- `bhc-lower` - AST to HIR lowering
- `bhc-hir-to-core` - HIR to Core lowering

## Specification References

- H26-SPEC Section 1: Language Overview
