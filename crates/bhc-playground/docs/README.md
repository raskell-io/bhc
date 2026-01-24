# bhc-playground

Browser-based Haskell interpreter for the BHC playground.

## Overview

This crate provides a complete Haskell execution environment that runs in the browser via WebAssembly. It uses BHC's real frontend and Core IR interpreter, enabling interactive Haskell exploration without native code generation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser Environment                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  JavaScript API                      │   │
│  │  run_haskell(source) → { success, display, type }   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              WASM Module (bhc-playground)            │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │  Parse  │→│  Lower  │→│ TypeCk  │→│  Eval   │   │   │
│  │  │(bhc-    │ │(bhc-    │ │(bhc-    │ │(bhc-    │   │   │
│  │  │ parser) │ │ lower)  │ │ typeck) │ │ core)   │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Compilation Pipeline

```rust
pub fn compile_and_run(source: &str) -> Result<PlaygroundResult, PlaygroundError> {
    // 1. Parse
    let ast = parse_module(source)?;

    // 2. Lower to HIR
    let hir = lower_to_hir(ast)?;

    // 3. Type check
    let typed_hir = type_check(hir)?;

    // 4. Lower to Core
    let core = lower_to_core(typed_hir)?;

    // 5. Evaluate
    let value = evaluate(core)?;

    // 6. Format result
    Ok(PlaygroundResult {
        success: true,
        display: format_value(&value),
        ty: format_type(&value.ty()),
        ..Default::default()
    })
}
```

## WASM Interface

### Exported Functions

```rust
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn run_haskell(source: &str) -> JsValue {
    let result = compile_and_run(source);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn check_type(source: &str) -> JsValue {
    let result = type_check_only(source);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn format_code(source: &str) -> String {
    format_haskell(source)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_completions(source: &str, position: usize) -> JsValue {
    let completions = compute_completions(source, position);
    serde_wasm_bindgen::to_value(&completions).unwrap()
}
```

### Result Types

```rust
#[derive(Serialize, Deserialize)]
pub struct PlaygroundResult {
    /// Whether execution succeeded
    pub success: bool,

    /// Pretty-printed result value
    pub display: Option<String>,

    /// Inferred type of the result
    pub ty: Option<String>,

    /// Core IR (if requested)
    pub ir: Option<String>,

    /// Execution time in milliseconds
    pub time_ms: Option<f64>,

    /// Error information (if failed)
    pub error: Option<PlaygroundError>,
}

#[derive(Serialize, Deserialize)]
pub struct PlaygroundError {
    /// Error message
    pub message: String,

    /// Source span of the error
    pub span: Option<SourceSpan>,

    /// Error code (e.g., "E0308")
    pub code: Option<String>,

    /// Suggested fixes
    pub suggestions: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}
```

## JavaScript Usage

```javascript
import init, { run_haskell, check_type, format_code } from 'bhc_playground';

// Initialize WASM module
await init();

// Run Haskell code
const result = run_haskell(`
  let factorial n = if n <= 1 then 1 else n * factorial (n - 1)
  in factorial 10
`);

if (result.success) {
  console.log(`Result: ${result.display}`);  // "3628800"
  console.log(`Type: ${result.ty}`);         // "Int"
} else {
  console.error(`Error: ${result.error.message}`);
}

// Type check only
const typeResult = check_type("map (+1) [1,2,3]");
console.log(`Type: ${typeResult.ty}`);  // "[Int]"

// Format code
const formatted = format_code("f x=x+1");
console.log(formatted);  // "f x = x + 1"
```

## Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| Basic expressions | ✅ | Literals, operators, application |
| Let bindings | ✅ | Local definitions |
| Lambda expressions | ✅ | Anonymous functions |
| Pattern matching | ✅ | Case expressions, guards |
| Type inference | ✅ | Hindley-Milner |
| Type annotations | ✅ | Explicit types |
| Type classes | ✅ | Standard classes |
| Custom types | ✅ | Data declarations |
| List comprehensions | ✅ | Desugared |
| Do notation | ✅ | For pure monads only |
| Module imports | 🔄 | Limited stdlib |
| IO actions | ❌ | Sandboxed environment |
| FFI | ❌ | No external calls |

## Prelude

The playground includes a minimal Prelude:

```haskell
-- Types
data Bool = False | True
data Maybe a = Nothing | Just a
data Either a b = Left a | Right b
data Ordering = LT | EQ | GT

-- Classes (selected)
class Eq a where (==), (/=) :: a -> a -> Bool
class Ord a where compare :: a -> a -> Ordering
class Show a where show :: a -> String
class Num a where (+), (-), (*) :: a -> a -> a
class Functor f where fmap :: (a -> b) -> f a -> f b
class Monad m where (>>=) :: m a -> (a -> m b) -> m b

-- Functions (selected)
id :: a -> a
const :: a -> b -> a
(.) :: (b -> c) -> (a -> b) -> a -> c
flip :: (a -> b -> c) -> b -> a -> c
map :: (a -> b) -> [a] -> [b]
filter :: (a -> Bool) -> [a] -> [a]
foldr :: (a -> b -> b) -> b -> [a] -> b
foldl :: (b -> a -> b) -> b -> [a] -> b
```

## Execution Limits

| Limit | Value | Reason |
|-------|-------|--------|
| Timeout | 5 seconds | Prevent infinite loops |
| Memory | 64 MB | WASM linear memory |
| Stack depth | 10,000 | Prevent stack overflow |
| Output size | 100 KB | Browser memory |

```rust
const EXECUTION_TIMEOUT_MS: u64 = 5000;
const MAX_MEMORY_BYTES: usize = 64 * 1024 * 1024;
const MAX_STACK_DEPTH: usize = 10_000;
const MAX_OUTPUT_BYTES: usize = 100 * 1024;
```

## Building

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for browser
wasm-pack build crates/bhc-playground --target web

# Build for Node.js
wasm-pack build crates/bhc-playground --target nodejs

# Build with optimizations
wasm-pack build crates/bhc-playground --target web --release
```

## Testing

```bash
# Rust tests
cargo test -p bhc-playground

# WASM tests (requires wasm-pack)
wasm-pack test --headless --firefox crates/bhc-playground
```

## Deployment

The playground is deployed at https://play.bhc.raskell.io

```bash
cd crates/bhc-playground/www
npm install
npm run build
npm run deploy
```

## Security Considerations

- **Sandboxed execution**: No file system or network access
- **Memory limits**: Bounded WASM linear memory
- **Timeout enforcement**: Prevents DoS via infinite loops
- **No eval/exec**: Cannot execute arbitrary code
- **CSP compatible**: No dynamic code generation

## See Also

- `bhc-parser` - Parsing
- `bhc-lower` - AST to HIR
- `bhc-typeck` - Type checking
- `bhc-core` - Core IR and evaluator
