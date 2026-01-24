# bhc-query

Query-based compilation system for the Basel Haskell Compiler.

## Overview

This crate provides the infrastructure for incremental, demand-driven compilation using a query-based architecture. It enables automatic memoization, dependency tracking, and cycle detection.

## Features

- **Incremental recomputation**: Only recompute what has changed
- **Demand-driven evaluation**: Queries computed only when needed
- **Automatic memoization**: Query results are cached
- **Cycle detection**: Detect and report recursive query cycles
- **Parallel evaluation**: Thread-safe query execution

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    QueryDatabase                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Query A    │  │  Query B    │  │  Query C    │ │
│  │  (cached)   │──│  (cached)   │──│  (pending)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                    Dependencies                      │
└─────────────────────────────────────────────────────┘
```

## Key Types

| Type | Description |
|------|-------------|
| `QueryDatabase` | Central database storing query results |
| `Revision` | Version number for tracking changes |
| `QueryId` | Unique identifier for a query invocation |
| `QueryState` | Current state of a query (NotComputed, InProgress, Memoized, Stale) |
| `QueryError` | Errors including cycles and panics |

## Usage

### Defining Queries

```rust
use bhc_query::{Query, QueryDatabase};

// Define a query trait
trait ParseQuery {
    fn parse(&self, file_id: FileId) -> Arc<Module>;
}

// Define a query that depends on another
trait TypeCheckQuery: ParseQuery {
    fn type_check(&self, file_id: FileId) -> Arc<TypedModule> {
        let module = self.parse(file_id);  // Dependency tracked
        // ... type check the module
    }
}
```

### Query Execution

```rust
use bhc_query::{QueryDatabase, Revision};

let mut db = QueryDatabase::new();

// Execute a query (will be computed and cached)
let result = db.query::<TypeCheckQuery>(file_id);

// Re-execute (will use cached result if inputs unchanged)
let cached = db.query::<TypeCheckQuery>(file_id);
assert_eq!(Arc::as_ptr(&result), Arc::as_ptr(&cached));
```

### Incremental Updates

```rust
// Mark an input as changed
db.set_input(file_id, new_source);

// Queries depending on this input are now stale
// Next query will recompute only what's necessary
let fresh = db.query::<TypeCheckQuery>(file_id);
```

## Revision Tracking

```rust
use bhc_query::Revision;

let rev = Revision::INITIAL;
assert_eq!(rev.as_u64(), 0);

let next = rev.next();
assert_eq!(next.as_u64(), 1);
```

## Query States

| State | Description |
|-------|-------------|
| `NotComputed` | Query has never been executed |
| `InProgress` | Query is currently being computed |
| `Memoized(rev)` | Query was computed in revision `rev` |
| `Stale` | Query needs recomputation |

## Error Handling

```rust
use bhc_query::QueryError;

match db.try_query::<SomeQuery>(key) {
    Ok(result) => { /* use result */ }
    Err(QueryError::Cycle(path)) => {
        // Handle cyclic dependency
        for query_id in path {
            eprintln!("  -> {:?}", query_id);
        }
    }
    Err(QueryError::Panic(msg)) => {
        // Query computation panicked
    }
    Err(QueryError::Cancelled) => {
        // Query was cancelled
    }
}
```

## Cycle Detection

The query system automatically detects cycles:

```rust
// If Query A depends on Query B, and Query B depends on Query A:
// QueryError::Cycle([QueryId(A), QueryId(B)]) is returned
```

## Thread Safety

Queries can be executed from multiple threads:

```rust
use std::thread;

let db = Arc::new(QueryDatabase::new());

let handles: Vec<_> = files.iter().map(|file| {
    let db = db.clone();
    let file = *file;
    thread::spawn(move || {
        db.query::<TypeCheckQuery>(file)
    })
}).collect();
```

## Design Notes

- Based on the salsa/rustc query model
- Uses interior mutability for concurrent access
- Revision numbers enable cheap staleness checks
- Query keys must be hashable and equality-comparable

## Related Crates

- `bhc-driver` - Uses queries for compilation orchestration
- `bhc-session` - Session context for queries
- `bhc-typeck` - Type checking queries
- `bhc-parser` - Parsing queries

## Specification References

- H26-SPEC Section 11: Incremental Compilation
