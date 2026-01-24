# Playground Deployment

This document describes how the BHC playground WASM is built and deployed.

## Overview

The playground is automatically built and deployed via GitHub Actions whenever
relevant crates change. The WASM module is hosted on the BHC website.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Push to main  │ ──▶ │  Build WASM     │ ──▶ │  Deploy to      │
│   (crates/*)    │     │  (CI workflow)  │     │  bhc.raskell.io │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## CI Workflow

The playground is rebuilt when any of its dependencies change.

**Workflow file:** `.github/workflows/playground.yml`

### Trigger Paths

The workflow triggers on changes to these crates:

| Crate | Purpose |
|-------|---------|
| `bhc-playground` | The playground crate itself |
| `bhc-span` | Source locations |
| `bhc-intern` | String interning |
| `bhc-index` | Index types |
| `bhc-arena` | Memory arenas |
| `bhc-data-structures` | Shared data structures |
| `bhc-diagnostics` | Error reporting |
| `bhc-lexer` | Tokenization |
| `bhc-ast` | Abstract syntax tree |
| `bhc-parser` | Parsing |
| `bhc-types` | Type representation |
| `bhc-hir` | High-level IR |
| `bhc-lower` | AST → HIR lowering |
| `bhc-typeck` | Type inference & checking |
| `bhc-core` | Core IR + evaluator |
| `bhc-hir-to-core` | HIR → Core lowering |
| `bhc-session` | Compilation session |

### Keeping Paths in Sync

To verify the workflow paths match the actual dependencies:

```bash
# List dependencies from Cargo.toml
grep "^bhc-" crates/bhc-playground/Cargo.toml | sed 's/ .*//'

# Compare with workflow paths
grep "crates/bhc-" .github/workflows/playground.yml
```

If a new dependency is added, update the workflow paths.

## Build Process

### 1. Build WASM

```bash
cargo build -p bhc-playground --target wasm32-unknown-unknown --release
```

### 2. Generate JavaScript Bindings

```bash
wasm-bindgen --out-dir out --target web \
  target/wasm32-unknown-unknown/release/bhc_playground.wasm
```

This produces:
- `bhc_playground.js` - JavaScript glue code
- `bhc_playground_bg.wasm` - WebAssembly binary

### 3. Deploy to Website

The files are pushed to the website repository:

```
raskell-io/bhc.raskell.io
└── static/
    └── playground/
        ├── bhc_playground.js
        └── bhc_playground_bg.wasm
```

## Manual Deployment

To manually trigger a deployment:

```bash
# Via GitHub CLI
gh workflow run playground.yml

# Or via GitHub UI
# Go to Actions → Build and Deploy Playground → Run workflow
```

## Local Development

### Prerequisites

```bash
# Install WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-bindgen CLI
cargo install wasm-bindgen-cli
```

### Build Locally

```bash
# Build WASM
cargo build -p bhc-playground --target wasm32-unknown-unknown --release

# Generate JS bindings
mkdir -p playground-out
wasm-bindgen --out-dir playground-out --target web \
  target/wasm32-unknown-unknown/release/bhc_playground.wasm
```

### Test Locally

```bash
# Serve the playground locally (requires a simple HTTP server)
cd playground-out
python3 -m http.server 8000

# Open http://localhost:8000 in your browser
```

## Deployment URLs

| Environment | URL |
|-------------|-----|
| Production | https://bhc.raskell.io/playground/ |
| WASM file | https://bhc.raskell.io/static/playground/bhc_playground_bg.wasm |
| JS file | https://bhc.raskell.io/static/playground/bhc_playground.js |

## Troubleshooting

### WASM Build Fails

If the CI build fails with "can't find crate for `core`":

```bash
# Ensure WASM target is installed
rustup target add wasm32-unknown-unknown
```

### wasm-bindgen Version Mismatch

The wasm-bindgen CLI version must match the crate version:

```bash
# Check crate version
grep wasm-bindgen Cargo.lock | head -1

# Install matching CLI version
cargo install wasm-bindgen-cli --version 0.2.108
```

### Deployment Permission Denied

If the CI can't push to the website repo:

1. Check that `DOCS_DEPLOY_TOKEN2` secret exists
2. Verify the token has write access to `raskell-io/bhc.raskell.io`

## Related Documentation

- [README.md](../README.md) - Crate overview
- [docs/README.md](./README.md) - API documentation
- [.github/workflows/playground.yml](../../../.github/workflows/playground.yml) - CI workflow
