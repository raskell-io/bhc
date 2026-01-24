# bhi

IR and kernel report inspector for the Basel Haskell Compiler.

## Overview

bhi (Basel Haskell Inspector) is a diagnostic tool for inspecting intermediate representations, kernel reports, memory allocation patterns, and call graphs produced by the BHC compiler.

## Usage

```bash
# Inspect IR dump
bhi ir output.core

# View kernel report
bhi kernel report.json

# Analyze memory allocation
bhi memory alloc.log

# Display call graph
bhi callgraph cg.dot
```

## Subcommands

### ir - Inspect IR Dumps

```bash
# View Core IR
bhi ir program.core

# Specify stage
bhi ir --stage=hir program.hir

# Output as JSON
bhi ir --format=json program.core

# Output as DOT graph
bhi ir --format=dot program.core > graph.dot
```

### kernel - View Kernel Reports

```bash
# View full report
bhi kernel report.json

# Show only failed fusions
bhi kernel --failures-only report.json

# Include timing information
bhi kernel --timing report.json
```

Example output:
```
Kernel Fusion Report
====================

[Kernel k1] dotProduct
  Status: FUSED
  Operations: zipWith (*), sum
  Traversals: 1 (optimal)
  SIMD: 8 x f32 (AVX)

[Kernel k2] matmul
  Status: TILED
  Tile size: 64x64
  Cache: L1 resident
  SIMD: 8 x f32 (AVX)

[Kernel k3] softmax
  Status: PARTIAL
  Fused: map, sum
  Blocked: maximum (multiple uses)
  Recommendation: Consider caching max result
```

### memory - Analyze Allocations

```bash
# Full allocation report
bhi memory alloc.log

# Heap allocations only
bhi memory --heap-only alloc.log

# Arena usage
bhi memory --arena alloc.log
```

Example output:
```
Memory Allocation Report
========================

Hot Arena:
  Peak usage: 4.2 MB
  Allocations: 1,247
  Avg alloc: 3.4 KB

Pinned Heap:
  Current: 128 KB
  Peak: 256 KB
  FFI buffers: 3

General Heap:
  Live: 12.3 MB
  GC collections: 7
  Max pause: 2.1 ms
```

### callgraph - Display Call Graph

```bash
# Text format
bhi callgraph program.cg

# DOT format for visualization
bhi callgraph --format=dot program.cg | dot -Tpng > graph.png

# Filter by function
bhi callgraph --filter="compute*" program.cg
```

### diff - Compare IR Dumps

```bash
# Compare two IR files
bhi diff before.core after.core

# JSON output
bhi diff --format=json before.core after.core
```

## Output Formats

| Format | Description |
|--------|-------------|
| `text` | Human-readable text (default) |
| `json` | Machine-readable JSON |
| `dot` | GraphViz DOT format |

## Generating Reports

To generate reports from `bhc`:

```bash
# Generate IR dumps
bhc --dump-ir=core Main.hs > program.core
bhc --dump-ir=tensor --profile=numeric Main.hs > program.tensor

# Generate kernel report
bhc --profile=numeric --kernel-report Main.hs > report.json

# Generate allocation report
bhc --profile=numeric --alloc-report Main.hs > alloc.log
```

## Use Cases

### Debugging Fusion Failures

```bash
bhc --profile=numeric --kernel-report suspicious.hs > report.json
bhi kernel --failures-only report.json
```

### Analyzing Memory Usage

```bash
bhc --alloc-report compute.hs > alloc.log
bhi memory --arena alloc.log
```

### Visualizing IR

```bash
bhc --dump-ir=core Main.hs > program.core
bhi ir --format=dot program.core | dot -Tsvg > ir.svg
```

## Design Notes

- Parses BHC's internal IR formats
- Supports streaming for large reports
- Color output for terminals
- Integration with GraphViz

## Related Tools

- `bhc` - Compiler CLI
- `bhci` - Interactive REPL

## Specification References

- H26-SPEC Section 8: Fusion Reporting
- H26-SPEC Section 9: Memory Model
