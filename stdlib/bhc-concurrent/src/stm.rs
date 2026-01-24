//! Software Transactional Memory
//!
//! STM provides composable atomic transactions for shared state.
//!
//! # Overview
//!
//! STM allows multiple threads to safely access and modify shared state
//! without explicit locking. Transactions are:
//!
//! - **Atomic**: Either all modifications commit or none do
//! - **Consistent**: Transactions always see a consistent view
//! - **Isolated**: Concurrent transactions don't interfere
//!
//! # Example
//!
//! ```ignore
//! let account1 = TVar::new(100);
//! let account2 = TVar::new(200);
//!
//! // Transfer 50 atomically
//! atomically(|| {
//!     let bal1 = account1.read_tx()?;
//!     let bal2 = account2.read_tx()?;
//!     account1.write_tx(bal1 - 50)?;
//!     account2.write_tx(bal2 + 50)?;
//!     Ok(())
//! });
//! ```
//!
//! # Retry and OrElse
//!
//! STM supports blocking operations via `retry` and alternatives via `orElse`:
//!
//! ```ignore
//! // Block until balance >= 50
//! atomically(|| {
//!     let bal = account.read_tx()?;
//!     if bal < 50 {
//!         return Err(StmAction::Retry);
//!     }
//!     account.write_tx(bal - 50)?;
//!     Ok(())
//! });
//! ```

use parking_lot::{Condvar, Mutex, RwLock};
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================================
// TVar - Transactional Variable
// ============================================================================

/// Global version clock for optimistic concurrency control.
static GLOBAL_VERSION: AtomicU64 = AtomicU64::new(0);

/// Unique identifier for a TVar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TVarId(u64);

impl TVarId {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Internal storage for a TVar.
struct TVarInner<T> {
    /// Current value.
    value: RwLock<T>,
    /// Version number (incremented on each write).
    version: AtomicU64,
    /// Waiters blocked on this TVar (for retry).
    waiters: Mutex<Vec<Arc<Condvar>>>,
}

/// A transactional variable.
///
/// TVars can only be safely read/written within an STM transaction.
pub struct TVar<T> {
    id: TVarId,
    inner: Arc<TVarInner<T>>,
}

impl<T: Clone + Send + Sync + 'static> TVar<T> {
    /// Create a new TVar with the given initial value.
    pub fn new(value: T) -> Self {
        Self {
            id: TVarId::new(),
            inner: Arc::new(TVarInner {
                value: RwLock::new(value),
                version: AtomicU64::new(0),
                waiters: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Get the TVar's unique identifier.
    #[must_use]
    pub fn id(&self) -> TVarId {
        self.id
    }

    /// Read the value within a transaction.
    ///
    /// The read is tracked for conflict detection.
    pub fn read_tx(&self) -> StmResult<T> {
        CURRENT_TX.with(|tx| {
            let mut tx = tx.borrow_mut();
            let tx = tx.as_mut().ok_or(StmError::NotInTransaction)?;
            tx.read(self)
        })
    }

    /// Write a value within a transaction.
    ///
    /// The write is buffered and only committed if the transaction succeeds.
    pub fn write_tx(&self, value: T) -> StmResult<()> {
        CURRENT_TX.with(|tx| {
            let mut tx = tx.borrow_mut();
            let tx = tx.as_mut().ok_or(StmError::NotInTransaction)?;
            tx.write(self, value);
            Ok(())
        })
    }

    /// Read the current value directly (outside a transaction).
    ///
    /// **Warning**: This provides no atomicity guarantees with other reads.
    pub fn read_direct(&self) -> T {
        self.inner.value.read().clone()
    }

    /// Write a value directly (outside a transaction).
    ///
    /// **Warning**: This provides no atomicity guarantees with other writes.
    pub fn write_direct(&self, value: T) {
        *self.inner.value.write() = value;
        self.inner.version.fetch_add(1, Ordering::Release);
        self.wake_waiters();
    }

    /// Read the current version.
    fn version(&self) -> u64 {
        self.inner.version.load(Ordering::Acquire)
    }

    /// Wake all threads waiting on this TVar.
    fn wake_waiters(&self) {
        let waiters = self.inner.waiters.lock();
        for waiter in waiters.iter() {
            waiter.notify_all();
        }
    }

    /// Register a waiter for retry.
    fn register_waiter(&self, condvar: Arc<Condvar>) {
        self.inner.waiters.lock().push(condvar);
    }

    /// Remove a waiter.
    fn unregister_waiter(&self, condvar: &Arc<Condvar>) {
        let mut waiters = self.inner.waiters.lock();
        waiters.retain(|w| !Arc::ptr_eq(w, condvar));
    }
}

impl<T: Clone + Send + Sync + 'static> Clone for TVar<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            inner: Arc::clone(&self.inner),
        }
    }
}

// Prevent Send/Sync issues
unsafe impl<T: Send + Sync> Send for TVar<T> {}
unsafe impl<T: Send + Sync> Sync for TVar<T> {}

// ============================================================================
// Transaction State
// ============================================================================

/// Result type for STM operations.
pub type StmResult<T> = Result<T, StmError>;

/// Errors that can occur in STM operations.
#[derive(Debug, Clone)]
pub enum StmError {
    /// Operation performed outside a transaction.
    NotInTransaction,
    /// Transaction should retry (block until TVars change).
    Retry,
    /// Transaction explicitly aborted.
    Aborted(String),
    /// Conflict detected during commit.
    Conflict,
}

/// A read entry in the transaction log.
struct ReadEntry {
    /// The TVar that was read (type-erased).
    tvar: Arc<dyn TVarOps>,
    /// Version at time of read.
    version: u64,
}

/// A write entry in the transaction log.
struct WriteEntry {
    /// The TVar to write (type-erased).
    tvar: Arc<dyn TVarOps>,
    /// The new value (boxed).
    value: Box<dyn Any + Send + Sync>,
}

/// Type-erased TVar operations for the transaction log.
trait TVarOps: Send + Sync {
    fn id(&self) -> TVarId;
    fn version(&self) -> u64;
    fn commit(&self, value: Box<dyn Any + Send + Sync>);
    fn register_waiter(&self, condvar: Arc<Condvar>);
    fn unregister_waiter(&self, condvar: &Arc<Condvar>);
}

impl<T: Clone + Send + Sync + 'static> TVarOps for TVar<T> {
    fn id(&self) -> TVarId {
        self.id
    }

    fn version(&self) -> u64 {
        TVar::version(self)
    }

    fn commit(&self, value: Box<dyn Any + Send + Sync>) {
        if let Ok(val) = value.downcast::<T>() {
            *self.inner.value.write() = *val;
            self.inner.version.fetch_add(1, Ordering::Release);
            self.wake_waiters();
        }
    }

    fn register_waiter(&self, condvar: Arc<Condvar>) {
        TVar::register_waiter(self, condvar);
    }

    fn unregister_waiter(&self, condvar: &Arc<Condvar>) {
        TVar::unregister_waiter(self, condvar);
    }
}

/// Transaction state.
struct Transaction {
    /// Read set: TVars read and their versions.
    reads: HashMap<TVarId, ReadEntry>,
    /// Write set: TVars written and their new values.
    writes: HashMap<TVarId, WriteEntry>,
    /// Cached read values for repeated reads.
    read_cache: HashMap<TVarId, Box<dyn Any + Send + Sync>>,
}

impl Transaction {
    fn new() -> Self {
        Self {
            reads: HashMap::new(),
            writes: HashMap::new(),
            read_cache: HashMap::new(),
        }
    }

    /// Read a TVar within this transaction.
    fn read<T: Clone + Send + Sync + 'static>(&mut self, tvar: &TVar<T>) -> StmResult<T> {
        let id = tvar.id();

        // Check write set first (read-your-writes).
        if let Some(entry) = self.writes.get(&id) {
            if let Some(val) = entry.value.downcast_ref::<T>() {
                return Ok(val.clone());
            }
        }

        // Check read cache.
        if let Some(cached) = self.read_cache.get(&id) {
            if let Some(val) = cached.downcast_ref::<T>() {
                return Ok(val.clone());
            }
        }

        // Read from TVar and record in read set.
        let value = tvar.read_direct();
        let version = tvar.version();

        self.reads.insert(
            id,
            ReadEntry {
                tvar: Arc::new(tvar.clone()) as Arc<dyn TVarOps>,
                version,
            },
        );
        self.read_cache
            .insert(id, Box::new(value.clone()) as Box<dyn Any + Send + Sync>);

        Ok(value)
    }

    /// Write a TVar within this transaction.
    fn write<T: Clone + Send + Sync + 'static>(&mut self, tvar: &TVar<T>, value: T) {
        let id = tvar.id();

        // Add to write set.
        self.writes.insert(
            id,
            WriteEntry {
                tvar: Arc::new(tvar.clone()) as Arc<dyn TVarOps>,
                value: Box::new(value.clone()) as Box<dyn Any + Send + Sync>,
            },
        );

        // Update read cache for read-your-writes.
        self.read_cache
            .insert(id, Box::new(value) as Box<dyn Any + Send + Sync>);
    }

    /// Validate the read set (no conflicts).
    fn validate(&self) -> bool {
        for entry in self.reads.values() {
            if entry.tvar.version() != entry.version {
                return false;
            }
        }
        true
    }

    /// Commit the transaction.
    fn commit(self) -> bool {
        // Global lock for commit (simplified - production would use fine-grained locking).
        static COMMIT_LOCK: Mutex<()> = Mutex::new(());
        let _guard = COMMIT_LOCK.lock();

        // Validate read set.
        if !self.validate() {
            return false;
        }

        // Apply writes.
        for entry in self.writes.into_values() {
            entry.tvar.commit(entry.value);
        }

        // Increment global version.
        GLOBAL_VERSION.fetch_add(1, Ordering::Release);

        true
    }

    /// Get all TVars in the read set (for retry).
    fn read_tvars(&self) -> Vec<Arc<dyn TVarOps>> {
        self.reads.values().map(|e| Arc::clone(&e.tvar)).collect()
    }
}

// Thread-local current transaction.
thread_local! {
    static CURRENT_TX: RefCell<Option<Transaction>> = const { RefCell::new(None) };
}

// ============================================================================
// Public API
// ============================================================================

/// Execute a transaction atomically.
///
/// The transaction will be retried if there are conflicts with concurrent
/// transactions. If the transaction calls `retry()`, it will block until
/// one of the read TVars changes.
///
/// # Example
///
/// ```ignore
/// let counter = TVar::new(0);
///
/// // Increment atomically
/// atomically(|| {
///     let n = counter.read_tx()?;
///     counter.write_tx(n + 1)?;
///     Ok(())
/// });
/// ```
pub fn atomically<T, F>(f: F) -> T
where
    F: Fn() -> StmResult<T>,
{
    loop {
        // Start a new transaction.
        CURRENT_TX.with(|tx| {
            *tx.borrow_mut() = Some(Transaction::new());
        });

        // Run the transaction.
        let result = f();

        // Take the transaction.
        let tx = CURRENT_TX.with(|tx| tx.borrow_mut().take()).unwrap();

        match result {
            Ok(value) => {
                // Try to commit.
                if tx.commit() {
                    return value;
                }
                // Conflict - retry.
                continue;
            }
            Err(StmError::Retry) => {
                // Block until a read TVar changes.
                let tvars = tx.read_tvars();
                if tvars.is_empty() {
                    // No TVars read - busy retry.
                    std::thread::yield_now();
                    continue;
                }

                // Register on all read TVars.
                let condvar = Arc::new(Condvar::new());
                let mutex = Mutex::new(());
                for tvar in &tvars {
                    tvar.register_waiter(Arc::clone(&condvar));
                }

                // Wait for a change.
                {
                    let guard = mutex.lock();
                    let _ = condvar.wait_for(&mut { guard }, std::time::Duration::from_millis(100));
                }

                // Unregister.
                for tvar in &tvars {
                    tvar.unregister_waiter(&condvar);
                }

                // Retry the transaction.
                continue;
            }
            Err(StmError::Conflict) => {
                // Conflict detected - retry.
                continue;
            }
            Err(StmError::Aborted(msg)) => {
                panic!("STM transaction aborted: {}", msg);
            }
            Err(StmError::NotInTransaction) => {
                panic!("STM operation outside transaction");
            }
        }
    }
}

/// Retry the current transaction.
///
/// The transaction will block until one of the TVars it has read changes,
/// then retry from the beginning.
///
/// # Example
///
/// ```ignore
/// // Block until queue is non-empty
/// atomically(|| {
///     let items = queue.read_tx()?;
///     if items.is_empty() {
///         return Err(StmError::Retry);
///     }
///     // Process items...
///     Ok(())
/// });
/// ```
pub fn retry<T>() -> StmResult<T> {
    Err(StmError::Retry)
}

/// Try the first action; if it retries, try the second.
///
/// # Example
///
/// ```ignore
/// // Take from either queue
/// atomically(|| {
///     or_else(
///         || {
///             let x = queue1.read_tx()?;
///             if x.is_empty() { retry()? }
///             Ok(x.pop())
///         },
///         || {
///             let x = queue2.read_tx()?;
///             if x.is_empty() { retry()? }
///             Ok(x.pop())
///         },
///     )
/// });
/// ```
pub fn or_else<T, F, G>(first: F, second: G) -> StmResult<T>
where
    F: FnOnce() -> StmResult<T>,
    G: FnOnce() -> StmResult<T>,
{
    match first() {
        Err(StmError::Retry) => second(),
        result => result,
    }
}

/// Abort the current transaction with a message.
pub fn abort<T>(msg: impl Into<String>) -> StmResult<T> {
    Err(StmError::Aborted(msg.into()))
}

/// Check a condition; retry if false.
pub fn check(condition: bool) -> StmResult<()> {
    if condition {
        Ok(())
    } else {
        retry()
    }
}

// ============================================================================
// TMVar - Transactional MVar
// ============================================================================

/// A transactional MVar (mutable variable that can be empty).
pub struct TMVar<T> {
    inner: TVar<Option<T>>,
}

impl<T: Clone + Send + Sync + 'static> TMVar<T> {
    /// Create a new TMVar with a value.
    pub fn new(value: T) -> Self {
        Self {
            inner: TVar::new(Some(value)),
        }
    }

    /// Create an empty TMVar.
    pub fn new_empty() -> Self {
        Self {
            inner: TVar::new(None),
        }
    }

    /// Take the value, blocking if empty.
    pub fn take(&self) -> StmResult<T> {
        let val = self.inner.read_tx()?;
        match val {
            Some(v) => {
                self.inner.write_tx(None)?;
                Ok(v)
            }
            None => retry(),
        }
    }

    /// Put a value, blocking if full.
    pub fn put(&self, value: T) -> StmResult<()> {
        let val = self.inner.read_tx()?;
        match val {
            Some(_) => retry(),
            None => {
                self.inner.write_tx(Some(value))?;
                Ok(())
            }
        }
    }

    /// Read the value without taking it, blocking if empty.
    pub fn read(&self) -> StmResult<T> {
        let val = self.inner.read_tx()?;
        match val {
            Some(v) => Ok(v),
            None => retry(),
        }
    }

    /// Try to take without blocking.
    pub fn try_take(&self) -> StmResult<Option<T>> {
        let val = self.inner.read_tx()?;
        match val {
            Some(v) => {
                self.inner.write_tx(None)?;
                Ok(Some(v))
            }
            None => Ok(None),
        }
    }

    /// Try to put without blocking.
    pub fn try_put(&self, value: T) -> StmResult<bool> {
        let val = self.inner.read_tx()?;
        match val {
            Some(_) => Ok(false),
            None => {
                self.inner.write_tx(Some(value))?;
                Ok(true)
            }
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> StmResult<bool> {
        Ok(self.inner.read_tx()?.is_none())
    }
}

impl<T: Clone + Send + Sync + 'static> Clone for TMVar<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// ============================================================================
// TChan - Transactional Channel
// ============================================================================

/// A transactional broadcast channel.
///
/// Multiple readers can subscribe via `dup_tchan`. Each reader gets its
/// own view of the channel.
pub struct TChan<T> {
    /// Write end (shared among writers).
    write_end: TVar<Vec<T>>,
    /// Read position for this reader.
    read_pos: TVar<usize>,
}

impl<T: Clone + Send + Sync + 'static> TChan<T> {
    /// Create a new empty channel.
    pub fn new() -> Self {
        Self {
            write_end: TVar::new(Vec::new()),
            read_pos: TVar::new(0),
        }
    }

    /// Write a value to the channel.
    pub fn write(&self, value: T) -> StmResult<()> {
        let mut items = self.write_end.read_tx()?;
        items.push(value);
        self.write_end.write_tx(items)?;
        Ok(())
    }

    /// Read a value from the channel, blocking if empty.
    pub fn read(&self) -> StmResult<T> {
        let items = self.write_end.read_tx()?;
        let pos = self.read_pos.read_tx()?;

        if pos >= items.len() {
            return retry();
        }

        let value = items[pos].clone();
        self.read_pos.write_tx(pos + 1)?;
        Ok(value)
    }

    /// Duplicate the channel (create a new reader at current position).
    pub fn dup(&self) -> StmResult<Self> {
        let pos = self.read_pos.read_tx()?;
        Ok(Self {
            write_end: self.write_end.clone(),
            read_pos: TVar::new(pos),
        })
    }

    /// Check if the channel is empty for this reader.
    pub fn is_empty(&self) -> StmResult<bool> {
        let items = self.write_end.read_tx()?;
        let pos = self.read_pos.read_tx()?;
        Ok(pos >= items.len())
    }
}

impl<T: Clone + Send + Sync + 'static> Default for TChan<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Send + Sync + 'static> Clone for TChan<T> {
    fn clone(&self) -> Self {
        Self {
            write_end: self.write_end.clone(),
            read_pos: self.read_pos.clone(),
        }
    }
}

// ============================================================================
// TQueue - Transactional Queue
// ============================================================================

/// A transactional FIFO queue.
pub struct TQueue<T> {
    read_end: TVar<Vec<T>>,
    write_end: TVar<Vec<T>>,
}

impl<T: Clone + Send + Sync + 'static> TQueue<T> {
    /// Create a new empty queue.
    pub fn new() -> Self {
        Self {
            read_end: TVar::new(Vec::new()),
            write_end: TVar::new(Vec::new()),
        }
    }

    /// Write a value to the queue.
    pub fn write(&self, value: T) -> StmResult<()> {
        let mut items = self.write_end.read_tx()?;
        items.push(value);
        self.write_end.write_tx(items)?;
        Ok(())
    }

    /// Read a value from the queue, blocking if empty.
    pub fn read(&self) -> StmResult<T> {
        let mut read_items = self.read_end.read_tx()?;

        if read_items.is_empty() {
            // Move from write end to read end.
            let write_items = self.write_end.read_tx()?;
            if write_items.is_empty() {
                return retry();
            }
            read_items = write_items.into_iter().rev().collect();
            self.write_end.write_tx(Vec::new())?;
        }

        let value = read_items.pop().unwrap();
        self.read_end.write_tx(read_items)?;
        Ok(value)
    }

    /// Try to read without blocking.
    pub fn try_read(&self) -> StmResult<Option<T>> {
        or_else(|| Ok(Some(self.read()?)), || Ok(None))
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> StmResult<bool> {
        let read_items = self.read_end.read_tx()?;
        let write_items = self.write_end.read_tx()?;
        Ok(read_items.is_empty() && write_items.is_empty())
    }
}

impl<T: Clone + Send + Sync + 'static> Default for TQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Send + Sync + 'static> Clone for TQueue<T> {
    fn clone(&self) -> Self {
        Self {
            read_end: self.read_end.clone(),
            write_end: self.write_end.clone(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI32;
    use std::thread;

    #[test]
    fn test_tvar_basic() {
        let var = TVar::new(42);

        let result: i32 = atomically(|| var.read_tx());
        assert_eq!(result, 42);

        atomically(|| var.write_tx(100));
        let result: i32 = atomically(|| var.read_tx());
        assert_eq!(result, 100);
    }

    #[test]
    fn test_tvar_read_your_writes() {
        let var = TVar::new(0);

        let result: i32 = atomically(|| {
            var.write_tx(42)?;
            var.read_tx() // Should see 42
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_transaction_atomicity() {
        let var1 = TVar::new(100);
        let var2 = TVar::new(200);

        // Transfer 50 from var1 to var2
        atomically(|| {
            let v1 = var1.read_tx()?;
            let v2 = var2.read_tx()?;
            var1.write_tx(v1 - 50)?;
            var2.write_tx(v2 + 50)?;
            Ok(())
        });

        assert_eq!(atomically(|| var1.read_tx()), 50);
        assert_eq!(atomically(|| var2.read_tx()), 250);
    }

    #[test]
    fn test_concurrent_increment() {
        let counter = TVar::new(0);
        let threads: Vec<_> = (0..10)
            .map(|_| {
                let counter = counter.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        atomically(|| {
                            let n = counter.read_tx()?;
                            counter.write_tx(n + 1)
                        });
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(atomically(|| counter.read_tx()), 1000);
    }

    #[test]
    fn test_or_else() {
        let var = TVar::new(0);

        let result: i32 = atomically(|| {
            or_else(
                || {
                    let n = var.read_tx()?;
                    if n == 0 {
                        retry()
                    } else {
                        Ok(n)
                    }
                },
                || Ok(42), // Fallback
            )
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_tmvar_basic() {
        let mvar = TMVar::new(42);

        let result: i32 = atomically(|| mvar.take());
        assert_eq!(result, 42);

        atomically(|| mvar.put(100));
        let result: i32 = atomically(|| mvar.read());
        assert_eq!(result, 100);
    }

    #[test]
    fn test_tmvar_empty() {
        let mvar: TMVar<i32> = TMVar::new_empty();
        assert!(atomically(|| mvar.is_empty()));

        atomically(|| mvar.put(42));
        assert!(!atomically(|| mvar.is_empty()));
    }

    #[test]
    fn test_tqueue_basic() {
        let queue = TQueue::new();

        atomically(|| queue.write(1));
        atomically(|| queue.write(2));
        atomically(|| queue.write(3));

        assert_eq!(atomically(|| queue.read()), 1);
        assert_eq!(atomically(|| queue.read()), 2);
        assert_eq!(atomically(|| queue.read()), 3);
    }

    #[test]
    fn test_tqueue_try_read_empty() {
        let queue: TQueue<i32> = TQueue::new();
        let result: Option<i32> = atomically(|| queue.try_read());
        assert_eq!(result, None);
    }

    #[test]
    fn test_check() {
        let var = TVar::new(10);

        // Should succeed immediately.
        atomically(|| {
            let n = var.read_tx()?;
            check(n > 5)?;
            Ok(n)
        });
    }

    #[test]
    fn test_producer_consumer() {
        let queue = TQueue::new();
        let done = TVar::new(false);
        let sum = Arc::new(AtomicI32::new(0));

        let queue_producer = queue.clone();
        let producer = thread::spawn(move || {
            for i in 1..=10 {
                atomically(|| queue_producer.write(i));
            }
            atomically(|| done.write_tx(true));
        });

        let queue_consumer = queue.clone();
        let sum_consumer = Arc::clone(&sum);
        let consumer = thread::spawn(move || {
            loop {
                let result: Option<i32> = atomically(|| queue_consumer.try_read());
                match result {
                    Some(n) => {
                        sum_consumer.fetch_add(n, Ordering::SeqCst);
                    }
                    None => {
                        // Check if done.
                        thread::sleep(std::time::Duration::from_millis(1));
                        let result: Option<i32> = atomically(|| queue_consumer.try_read());
                        if result.is_none() {
                            break;
                        }
                        if let Some(n) = result {
                            sum_consumer.fetch_add(n, Ordering::SeqCst);
                        }
                    }
                }
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();

        assert_eq!(sum.load(Ordering::SeqCst), 55); // 1+2+...+10
    }
}
