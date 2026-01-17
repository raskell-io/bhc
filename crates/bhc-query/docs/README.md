# bhc-query

Query-based incremental compilation infrastructure for BHC.

## Overview

`bhc-query` implements a demand-driven, memoized computation system inspired by rustc and salsa. Key features:

- **On-demand computation**: Only compute what's needed
- **Memoization**: Cache results, reuse across compilations
- **Incremental**: Track dependencies, invalidate minimally
- **Cycle detection**: Detect and report query cycles

## Core Types

| Type | Description |
|------|-------------|
| `Query` | Trait for defining queries |
| `QueryId` | Unique identifier for a query invocation |
| `QueryState` | Cached result state |
| `QueryStorage` | Memoization storage |
| `Revision` | Change tracking counter |
| `Database` | Query execution context |

## The Query Trait

```rust
pub trait Query: 'static {
    /// Input key type
    type Key: Clone + Hash + Eq;

    /// Output value type
    type Value: Clone;

    /// Compute the query result
    fn compute(db: &dyn Database, key: &Self::Key) -> Self::Value;

    /// Query name for debugging
    fn name() -> &'static str;
}
```

## Defining Queries

```rust
// Define a query for parsing a file
pub struct ParseFileQuery;

impl Query for ParseFileQuery {
    type Key = FileId;
    type Value = Arc<Ast>;

    fn compute(db: &dyn Database, file_id: &FileId) -> Arc<Ast> {
        let source = db.file_text(*file_id);
        Arc::new(parse(&source))
    }

    fn name() -> &'static str {
        "ParseFile"
    }
}

// Define a query for type checking
pub struct TypeCheckQuery;

impl Query for TypeCheckQuery {
    type Key = DefId;
    type Value = Arc<TypedDef>;

    fn compute(db: &dyn Database, def_id: &DefId) -> Arc<TypedDef> {
        let hir = db.query::<HirQuery>(def_id);
        Arc::new(type_check(&hir))
    }

    fn name() -> &'static str {
        "TypeCheck"
    }
}
```

## Query Storage

```rust
pub struct QueryStorage<Q: Query> {
    /// Cached results
    results: FxHashMap<Q::Key, QueryState<Q::Value>>,
    /// Revision when last computed
    revisions: FxHashMap<Q::Key, Revision>,
}

pub enum QueryState<V> {
    /// Not yet computed
    NotComputed,
    /// Currently being computed (cycle detection)
    InProgress,
    /// Computed and cached
    Cached(V),
}

impl<Q: Query> QueryStorage<Q> {
    /// Get or compute a query result
    pub fn get(&mut self, db: &dyn Database, key: &Q::Key) -> Q::Value;

    /// Invalidate a cached result
    pub fn invalidate(&mut self, key: &Q::Key);

    /// Check if result is fresh
    pub fn is_fresh(&self, key: &Q::Key, revision: Revision) -> bool;
}
```

## Revisions

Track changes for incremental compilation:

```rust
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Revision(u64);

impl Revision {
    pub const INITIAL: Revision = Revision(0);

    pub fn next(self) -> Revision {
        Revision(self.0 + 1)
    }
}

pub struct RevisionTracker {
    current: Revision,
    file_revisions: FxHashMap<FileId, Revision>,
}

impl RevisionTracker {
    /// Bump revision for a changed file
    pub fn file_changed(&mut self, file_id: FileId) {
        self.current = self.current.next();
        self.file_revisions.insert(file_id, self.current);
    }

    /// Get revision when file was last changed
    pub fn file_revision(&self, file_id: FileId) -> Revision;
}
```

## Database

The database provides query execution:

```rust
pub trait Database {
    /// Execute a query
    fn query<Q: Query>(&self, key: &Q::Key) -> Q::Value;

    /// Get file text (input query)
    fn file_text(&self, file_id: FileId) -> Arc<str>;

    /// Current revision
    fn revision(&self) -> Revision;
}

pub struct SimpleDatabase {
    /// File contents
    files: FxHashMap<FileId, Arc<str>>,
    /// Revision tracker
    revisions: RevisionTracker,
    /// Query storages (type-erased)
    storages: FxHashMap<TypeId, Box<dyn Any>>,
    /// Active query stack (cycle detection)
    active: RefCell<Vec<QueryId>>,
}
```

## Cycle Detection

```rust
impl SimpleDatabase {
    fn execute<Q: Query>(&self, key: &Q::Key) -> Q::Value {
        let query_id = QueryId::new::<Q>(key);

        // Check for cycle
        if self.active.borrow().contains(&query_id) {
            panic!("query cycle detected: {:?}", self.active.borrow());
        }

        // Push onto active stack
        self.active.borrow_mut().push(query_id);

        // Compute
        let result = Q::compute(self, key);

        // Pop from active stack
        self.active.borrow_mut().pop();

        result
    }
}
```

## Dependency Tracking

```rust
pub struct Dependencies {
    /// Queries this query depends on
    deps: Vec<QueryId>,
}

impl Dependencies {
    /// Record a dependency
    pub fn record(&mut self, query_id: QueryId);

    /// Check if any dependency changed
    pub fn any_changed(&self, db: &dyn Database) -> bool;
}
```

## Quick Start

```rust
use bhc_query::{Database, Query, SimpleDatabase};

// Create database
let mut db = SimpleDatabase::new();

// Add a file
let file_id = FileId::new(0);
db.set_file_text(file_id, "module Main where\nmain = print 42");

// Query parsing (computed on demand)
let ast = db.query::<ParseFileQuery>(&file_id);

// Query again (cached)
let ast2 = db.query::<ParseFileQuery>(&file_id);

// Modify file
db.set_file_text(file_id, "module Main where\nmain = print 43");

// Query again (recomputed due to change)
let ast3 = db.query::<ParseFileQuery>(&file_id);
```

## Incremental Workflow

```rust
// Initial compilation
let typed = db.query::<TypeCheckQuery>(&main_def);

// User edits file
db.set_file_text(file_id, new_source);

// Re-query (only recomputes affected queries)
let typed = db.query::<TypeCheckQuery>(&main_def);
// - ParseFileQuery: recomputed (file changed)
// - HirQuery: recomputed (depends on parse)
// - TypeCheckQuery: recomputed (depends on HIR)
// - Other files: cached (unchanged)
```

## Query Groups

Organize related queries:

```rust
pub trait ParserQueries: Database {
    fn parse_file(&self, file_id: FileId) -> Arc<Ast> {
        self.query::<ParseFileQuery>(&file_id)
    }

    fn file_items(&self, file_id: FileId) -> Arc<Vec<Item>> {
        self.query::<FileItemsQuery>(&file_id)
    }
}

pub trait TypeQueries: Database {
    fn type_of(&self, def_id: DefId) -> Ty {
        self.query::<TypeOfQuery>(&def_id)
    }

    fn infer_expr(&self, expr_id: ExprId) -> Ty {
        self.query::<InferExprQuery>(&expr_id)
    }
}
```

## Performance

- Queries are memoized by default
- Dependencies tracked automatically
- Minimal recomputation on changes
- Parallel query execution (with appropriate synchronization)

## See Also

- `bhc-driver`: Uses queries for compilation
- `bhc-session`: Compilation context
- salsa: Inspiration for the design
- rustc query system documentation
