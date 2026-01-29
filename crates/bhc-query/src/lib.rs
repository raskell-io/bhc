//! Query-based compilation system for BHC.
//!
//! This crate provides the infrastructure for incremental, demand-driven
//! compilation using a query-based architecture similar to rustc's query
//! system and salsa.
//!
//! # Overview
//!
//! The query system allows the compiler to:
//!
//! - **Incrementally recompute** only what has changed between compilations
//! - **Demand-driven evaluation** where queries are only computed when needed
//! - **Automatic memoization** of query results
//! - **Cycle detection** for recursive queries
//!
//! # Architecture
//!
//! Queries are defined as traits that can be implemented by a database.
//! The database stores memoized results and tracks dependencies between queries.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                    QueryDatabase                     │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
//! │  │  Query A    │  │  Query B    │  │  Query C    │ │
//! │  │  (cached)   │──│  (cached)   │──│  (pending)  │ │
//! │  └─────────────┘  └─────────────┘  └─────────────┘ │
//! │                    Dependencies                      │
//! └─────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHasher;
use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::hash::{BuildHasherDefault, Hash};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A revision number tracking database state changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Revision(u64);

impl Revision {
    /// The initial revision.
    pub const INITIAL: Self = Self(0);

    /// Create a new revision with the given value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Get the next revision.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Get the raw revision value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// A unique identifier for a query invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryId {
    /// The type of query.
    pub query_type: TypeId,
    /// A hash of the query key.
    pub key_hash: u64,
}

/// The state of a query computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryState {
    /// Query has not been computed yet.
    NotComputed,
    /// Query is currently being computed (cycle detection).
    InProgress,
    /// Query has been computed and memoized.
    Memoized(Revision),
    /// Query result is stale and needs recomputation.
    Stale,
}

/// Error types for the query system.
#[derive(Debug, Clone)]
pub enum QueryError {
    /// A cycle was detected in query dependencies.
    Cycle(Vec<QueryId>),
    /// The query panicked during computation.
    Panic(String),
    /// The query was cancelled.
    Cancelled,
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cycle(ids) => write!(f, "query cycle detected: {} queries involved", ids.len()),
            Self::Panic(msg) => write!(f, "query panicked: {msg}"),
            Self::Cancelled => write!(f, "query was cancelled"),
        }
    }
}

impl std::error::Error for QueryError {}

/// Result type for query computations.
pub type QueryResult<T> = Result<T, QueryError>;

/// A memoized query result with revision tracking.
#[derive(Debug)]
pub struct MemoizedResult<V> {
    /// The cached value.
    pub value: V,
    /// The revision when this was computed.
    pub computed_at: Revision,
    /// The revision when this was last verified.
    pub verified_at: Revision,
    /// Dependencies of this query (for invalidation).
    pub dependencies: Vec<QueryId>,
}

/// Trait for types that can be used as query keys.
pub trait QueryKey: Clone + Eq + Hash + Debug + Send + Sync + 'static {}

impl<T: Clone + Eq + Hash + Debug + Send + Sync + 'static> QueryKey for T {}

/// Trait for types that can be used as query values.
pub trait QueryValue: Clone + Debug + Send + Sync + 'static {}

impl<T: Clone + Debug + Send + Sync + 'static> QueryValue for T {}

/// A query definition with its computation function.
pub trait Query: Send + Sync + 'static {
    /// The key type for this query.
    type Key: QueryKey;
    /// The value type for this query.
    type Value: QueryValue;

    /// Compute the query result for the given key.
    fn compute(&self, db: &dyn QueryDatabase, key: &Self::Key) -> Self::Value;

    /// Get a human-readable name for this query.
    fn name(&self) -> &'static str;
}

/// The central database trait that all query databases must implement.
pub trait QueryDatabase: Send + Sync {
    /// Get the current revision of the database.
    fn current_revision(&self) -> Revision;

    /// Increment the revision (called when inputs change).
    fn increment_revision(&self) -> Revision;

    /// Mark a query as in-progress (for cycle detection).
    fn mark_in_progress(&self, id: QueryId);

    /// Unmark a query as in-progress.
    fn unmark_in_progress(&self, id: QueryId);

    /// Check if a query is in progress (cycle detection).
    fn is_in_progress(&self, id: QueryId) -> bool;

    /// Record a dependency from the current query to another.
    fn record_dependency(&self, from: QueryId, to: QueryId);
}

/// Storage for a specific query type.
pub struct QueryStorage<Q: Query> {
    /// The query implementation.
    query: Q,
    /// Memoized results keyed by query key.
    results: DashMap<Q::Key, MemoizedResult<Q::Value>, BuildHasherDefault<FxHasher>>,
}

impl<Q: Query> QueryStorage<Q> {
    /// Create new storage for a query.
    pub fn new(query: Q) -> Self {
        Self {
            query,
            results: DashMap::with_hasher(BuildHasherDefault::default()),
        }
    }

    /// Execute the query, returning a memoized result if available.
    pub fn get(&self, db: &dyn QueryDatabase, key: &Q::Key) -> QueryResult<Q::Value> {
        // Check for cached result
        if let Some(entry) = self.results.get(key) {
            if entry.computed_at >= db.current_revision() {
                tracing::trace!(query = Q::name(&self.query), "cache hit");
                return Ok(entry.value.clone());
            }
        }

        // Compute the result
        let query_id = self.make_query_id(key);

        // Check for cycles
        if db.is_in_progress(query_id) {
            return Err(QueryError::Cycle(vec![query_id]));
        }

        db.mark_in_progress(query_id);
        let value = self.query.compute(db, key);
        db.unmark_in_progress(query_id);

        // Memoize the result
        let current_rev = db.current_revision();
        self.results.insert(
            key.clone(),
            MemoizedResult {
                value: value.clone(),
                computed_at: current_rev,
                verified_at: current_rev,
                dependencies: Vec::new(),
            },
        );

        tracing::trace!(query = Q::name(&self.query), "computed and cached");

        Ok(value)
    }

    /// Invalidate all cached results for this query.
    pub fn invalidate_all(&self) {
        self.results.clear();
    }

    /// Invalidate a specific key.
    pub fn invalidate(&self, key: &Q::Key) {
        self.results.remove(key);
    }

    fn make_query_id(&self, key: &Q::Key) -> QueryId {
        use std::hash::Hasher;
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        QueryId {
            query_type: TypeId::of::<Q>(),
            key_hash: hasher.finish(),
        }
    }
}

/// A simple in-memory query database implementation.
pub struct SimpleDatabase {
    revision: AtomicU64,
    in_progress: DashMap<QueryId, (), BuildHasherDefault<FxHasher>>,
    dependencies: DashMap<QueryId, Vec<QueryId>, BuildHasherDefault<FxHasher>>,
}

impl SimpleDatabase {
    /// Create a new empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            revision: AtomicU64::new(0),
            in_progress: DashMap::with_hasher(BuildHasherDefault::default()),
            dependencies: DashMap::with_hasher(BuildHasherDefault::default()),
        }
    }
}

impl Default for SimpleDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryDatabase for SimpleDatabase {
    fn current_revision(&self) -> Revision {
        Revision(self.revision.load(Ordering::Acquire))
    }

    fn increment_revision(&self) -> Revision {
        Revision(self.revision.fetch_add(1, Ordering::AcqRel) + 1)
    }

    fn mark_in_progress(&self, id: QueryId) {
        self.in_progress.insert(id, ());
    }

    fn unmark_in_progress(&self, id: QueryId) {
        self.in_progress.remove(&id);
    }

    fn is_in_progress(&self, id: QueryId) -> bool {
        self.in_progress.contains_key(&id)
    }

    fn record_dependency(&self, from: QueryId, to: QueryId) {
        self.dependencies
            .entry(from)
            .or_insert_with(Vec::new)
            .push(to);
    }
}

/// A runtime for executing queries with a database.
pub struct QueryRuntime<DB> {
    /// The underlying database.
    pub db: DB,
}

impl<DB: QueryDatabase> QueryRuntime<DB> {
    /// Create a new runtime with the given database.
    pub fn new(db: DB) -> Self {
        Self { db }
    }

    /// Execute a query and return its result.
    pub fn query<Q: Query>(
        &self,
        storage: &QueryStorage<Q>,
        key: &Q::Key,
    ) -> QueryResult<Q::Value> {
        storage.get(&self.db, key)
    }
}

/// Marker trait for input queries that can be set directly.
pub trait InputQuery: Query + Sized {
    /// Set the value for an input query.
    fn set(storage: &QueryStorage<Self>, key: Self::Key, value: Self::Value);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestQuery;

    impl Query for TestQuery {
        type Key = u32;
        type Value = u64;

        fn compute(&self, _db: &dyn QueryDatabase, key: &u32) -> u64 {
            (*key as u64) * 2
        }

        fn name(&self) -> &'static str {
            "test_query"
        }
    }

    #[test]
    fn test_basic_query() {
        let db = SimpleDatabase::new();
        let storage = QueryStorage::new(TestQuery);
        let runtime = QueryRuntime::new(db);

        let result = runtime.query(&storage, &21).unwrap();
        assert_eq!(result, 42);

        // Second call should be cached
        let result2 = runtime.query(&storage, &21).unwrap();
        assert_eq!(result2, 42);
    }

    #[test]
    fn test_revision_tracking() {
        let db = SimpleDatabase::new();
        assert_eq!(db.current_revision(), Revision::INITIAL);

        let rev1 = db.increment_revision();
        assert_eq!(rev1, Revision::new(1));

        let rev2 = db.increment_revision();
        assert_eq!(rev2, Revision::new(2));
    }
}
