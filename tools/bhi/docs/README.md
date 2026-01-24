# bhi

IR and Kernel Report Inspector for the Basel Haskell Compiler.

## Overview

bhi (Basel Haskell Inspector) is a diagnostic tool for examining compilation artifacts. It can parse and display intermediate representations, kernel fusion reports, memory allocation patterns, and call graphs produced by the BHC compiler.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          bhi                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Input Files                       │   │
│  │  .core  .hir  .tensor  .loop  .json  .log  .cg     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           ▼               ▼               ▼                 │
│    ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│    │ IR Parser │   │  Report   │   │  Graph    │           │
│    │           │   │  Parser   │   │  Parser   │           │
│    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘           │
│          │               │               │                  │
│          └───────────────┴───────────────┘                  │
│                          │                                  │
│                          ▼                                  │
│    ┌─────────────────────────────────────────────────────┐ │
│    │                   Formatters                         │ │
│    │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │ │
│    │  │  Text   │  │  JSON   │  │   DOT   │             │ │
│    │  └─────────┘  └─────────┘  └─────────┘             │ │
│    └─────────────────────────────────────────────────────┘ │
│                          │                                  │
│                          ▼                                  │
│    ┌─────────────────────────────────────────────────────┐ │
│    │                    Output                            │ │
│    └─────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Subcommands

### ir - IR Inspection

```rust
/// Parse and display IR files
pub fn cmd_ir(args: IrArgs) -> Result<()> {
    // Load IR file
    let content = std::fs::read_to_string(&args.file)?;

    // Detect or use specified stage
    let stage = args.stage.unwrap_or_else(|| detect_ir_stage(&args.file));

    // Parse IR
    let ir = match stage {
        IrStage::Ast => parse_ast(&content)?,
        IrStage::Hir => parse_hir(&content)?,
        IrStage::Core => parse_core(&content)?,
        IrStage::Tensor => parse_tensor(&content)?,
        IrStage::Loop => parse_loop(&content)?,
    };

    // Format output
    match args.format.as_str() {
        "text" => print_ir_text(&ir),
        "json" => print_ir_json(&ir),
        "dot" => print_ir_dot(&ir),
        _ => Err(Error::UnknownFormat(args.format)),
    }
}
```

### kernel - Kernel Report Inspection

```rust
/// Parse and display kernel fusion reports
pub fn cmd_kernel(args: KernelArgs) -> Result<()> {
    // Load report
    let content = std::fs::read_to_string(&args.file)?;
    let report: KernelReport = serde_json::from_str(&content)?;

    // Filter if requested
    let kernels: Vec<_> = if args.failures_only {
        report.kernels.iter()
            .filter(|k| !k.is_fully_fused())
            .collect()
    } else {
        report.kernels.iter().collect()
    };

    // Display
    println!("Kernel Fusion Report");
    println!("====================\n");

    for kernel in kernels {
        print_kernel(kernel, args.timing);
    }

    // Summary
    print_kernel_summary(&report);

    Ok(())
}

fn print_kernel(kernel: &Kernel, show_timing: bool) {
    println!("[Kernel {}] {}", kernel.id, kernel.name);
    println!("  Status: {}", format_status(&kernel.status));

    if let Some(ops) = &kernel.fused_ops {
        println!("  Fused operations: {}", ops.join(", "));
    }

    println!("  Traversals: {} ({})",
        kernel.traversals,
        if kernel.traversals == 1 { "optimal" } else { "suboptimal" }
    );

    if let Some(simd) = &kernel.simd {
        println!("  SIMD: {} x {} ({})", simd.width, simd.dtype, simd.instruction_set);
    }

    if let Some(blocked) = &kernel.blocked_fusions {
        println!("  Blocked fusions:");
        for block in blocked {
            println!("    - {}: {}", block.op, block.reason);
        }
    }

    if show_timing {
        if let Some(timing) = &kernel.timing {
            println!("  Compile time: {:?}", timing.compile_time);
            println!("  Estimated runtime: {:?}", timing.estimated_runtime);
        }
    }

    println!();
}
```

### memory - Memory Analysis

```rust
/// Analyze memory allocation patterns
pub fn cmd_memory(args: MemoryArgs) -> Result<()> {
    let content = std::fs::read_to_string(&args.file)?;
    let report: MemoryReport = parse_memory_report(&content)?;

    println!("Memory Allocation Report");
    println!("========================\n");

    // Hot Arena
    if !args.heap_only {
        println!("Hot Arena:");
        println!("  Peak usage: {}", format_bytes(report.arena.peak_usage));
        println!("  Allocations: {}", report.arena.allocation_count);
        println!("  Avg allocation: {}", format_bytes(
            report.arena.total_allocated / report.arena.allocation_count.max(1)
        ));
        println!();
    }

    // Pinned Heap
    println!("Pinned Heap:");
    println!("  Current: {}", format_bytes(report.pinned.current));
    println!("  Peak: {}", format_bytes(report.pinned.peak));
    println!("  FFI buffers: {}", report.pinned.ffi_buffers);
    println!();

    // General Heap
    if !args.arena {
        println!("General Heap:");
        println!("  Live: {}", format_bytes(report.heap.current_live));
        println!("  Peak: {}", format_bytes(report.heap.peak_usage));
        println!("  GC collections: {}", report.heap.gc_collections);
        println!("  Max GC pause: {:?}", report.heap.max_gc_pause);
        println!();
    }

    // Allocation breakdown by type
    if !report.by_type.is_empty() {
        println!("By Type:");
        for (ty, stats) in &report.by_type {
            println!("  {}: {} ({} allocations)",
                ty, format_bytes(stats.total), stats.count);
        }
    }

    Ok(())
}
```

### callgraph - Call Graph Display

```rust
/// Display and analyze call graphs
pub fn cmd_callgraph(args: CallgraphArgs) -> Result<()> {
    let content = std::fs::read_to_string(&args.file)?;
    let graph = parse_callgraph(&content)?;

    // Filter if requested
    let filtered = if let Some(pattern) = &args.filter {
        filter_graph(&graph, pattern)
    } else {
        graph
    };

    match args.format.as_str() {
        "text" => print_callgraph_text(&filtered),
        "dot" => print_callgraph_dot(&filtered),
        "json" => print_callgraph_json(&filtered),
        _ => Err(Error::UnknownFormat(args.format)),
    }
}

fn print_callgraph_text(graph: &CallGraph) {
    for node in &graph.nodes {
        println!("{}", node.name);
        for callee in &node.callees {
            println!("  └─▶ {}", callee);
        }
        println!();
    }
}

fn print_callgraph_dot(graph: &CallGraph) {
    println!("digraph callgraph {{");
    println!("  rankdir=LR;");
    println!("  node [shape=box];");

    for node in &graph.nodes {
        for callee in &node.callees {
            println!("  \"{}\" -> \"{}\";", node.name, callee);
        }
    }

    println!("}}");
}
```

### diff - IR Comparison

```rust
/// Compare two IR files
pub fn cmd_diff(args: DiffArgs) -> Result<()> {
    let before = load_ir(&args.before)?;
    let after = load_ir(&args.after)?;

    let diff = compute_ir_diff(&before, &after);

    match args.format.as_str() {
        "text" => print_diff_text(&diff),
        "json" => print_diff_json(&diff),
        _ => Err(Error::UnknownFormat(args.format)),
    }
}

fn print_diff_text(diff: &IrDiff) {
    println!("IR Diff");
    println!("=======\n");

    println!("Added:");
    for item in &diff.added {
        println!("  + {}", item);
    }

    println!("\nRemoved:");
    for item in &diff.removed {
        println!("  - {}", item);
    }

    println!("\nModified:");
    for (name, change) in &diff.modified {
        println!("  ~ {}", name);
        println!("    before: {}", change.before);
        println!("    after:  {}", change.after);
    }

    println!("\nSummary:");
    println!("  {} added, {} removed, {} modified",
        diff.added.len(), diff.removed.len(), diff.modified.len());
}
```

## IR Formats

### Core IR Format

```
-- Core IR text format
module Main where

-- Bindings
main :: IO ()
main = let %0 : String = "Hello, World!"
       in bhc_print_string %0

-- With types
id :: forall a. a -> a
id = /\a -> \(x : a) -> x

-- With applications
example :: Int
example = id @Int 42
```

### Tensor IR Format

```
-- Tensor IR text format
kernel @k0 {
  input %x : Tensor [1024] Float (contiguous, simd256)
  input %y : Tensor [1024] Float (contiguous, simd256)
  output %z : Tensor [1024] Float (contiguous, simd256)

  -- Fused: map (*2), map (+1), zipWith (+)
  for i in 0..1024 step 8 vectorize {
    %t0 = vload %x[i:i+8]
    %t1 = vmul %t0, broadcast(2.0)
    %t2 = vadd %t1, broadcast(1.0)
    %t3 = vload %y[i:i+8]
    %t4 = vadd %t2, %t3
    vstore %z[i:i+8], %t4
  }
}
```

## Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| text | .txt | Human reading |
| json | .json | Tool integration |
| dot | .dot | GraphViz visualization |

## See Also

- `bhc` - Compiler CLI
- `bhci` - Interactive REPL
- `bhc-core` - Core IR definition
- `bhc-tensor-ir` - Tensor IR definition
