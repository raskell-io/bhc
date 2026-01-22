//! Structured concurrency scopes
//!
//! Scopes ensure all spawned tasks complete before the scope exits.
//!
//! # Overview
//!
//! Structured concurrency guarantees that concurrent operations are
//! properly bounded and cleaned up. Every task runs within a scope,
//! and when the scope exits, all tasks are either completed or cancelled.
//!
//! # Example
//!
//! ```
//! use bhc_concurrent::scope::{with_scope, spawn};
//!
//! with_scope(|scope| {
//!     let task1 = spawn(scope, || 1 + 1);
//!     let task2 = spawn(scope, || 2 + 2);
//!
//!     // Both tasks complete before scope exits
//!     assert_eq!(task1.join(), Some(2));
//!     assert_eq!(task2.join(), Some(4));
//! });
//! ```

use crate::task::{Task, TaskHandle, TaskState};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// A concurrency scope that manages task lifetimes
#[derive(Clone)]
pub struct Scope {
    inner: Arc<ScopeInner>,
}

struct ScopeInner {
    /// Number of active tasks
    active_tasks: AtomicUsize,
    /// Whether the scope has been cancelled
    cancelled: AtomicBool,
    /// Parent scope (for cancellation propagation)
    parent: Option<Scope>,
}

impl Scope {
    /// Create a new scope
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ScopeInner {
                active_tasks: AtomicUsize::new(0),
                cancelled: AtomicBool::new(false),
                parent: None,
            }),
        }
    }

    /// Create a child scope (inherits cancellation from parent)
    pub fn child(&self) -> Self {
        Self {
            inner: Arc::new(ScopeInner {
                active_tasks: AtomicUsize::new(0),
                cancelled: AtomicBool::new(false),
                parent: Some(self.clone()),
            }),
        }
    }

    /// Check if the scope has been cancelled
    pub fn is_cancelled(&self) -> bool {
        if self.inner.cancelled.load(Ordering::SeqCst) {
            return true;
        }
        // Check parent cancellation
        if let Some(ref parent) = self.inner.parent {
            if parent.is_cancelled() {
                return true;
            }
        }
        false
    }

    /// Cancel the scope and all its tasks
    pub fn cancel(&self) {
        self.inner.cancelled.store(true, Ordering::SeqCst);
    }

    /// Get the number of active tasks
    pub fn active_count(&self) -> usize {
        self.inner.active_tasks.load(Ordering::SeqCst)
    }

    /// Register a new task
    pub(crate) fn register_task(&self) {
        self.inner.active_tasks.fetch_add(1, Ordering::SeqCst);
    }

    /// Unregister a completed task
    pub(crate) fn unregister_task(&self) {
        self.inner.active_tasks.fetch_sub(1, Ordering::SeqCst);
    }

    /// Wait for all tasks to complete
    pub fn wait_all(&self) {
        while self.active_count() > 0 {
            std::thread::yield_now();
        }
    }

    /// Wait for all tasks with a timeout
    ///
    /// Returns true if all tasks completed, false if timed out.
    pub fn wait_all_timeout(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;

        while self.active_count() > 0 {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::yield_now();
        }
        true
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        // Only handle cleanup if we're the last reference to this scope
        if Arc::strong_count(&self.inner) == 1 {
            // Cancel any remaining tasks
            self.cancel();

            // Wait for all tasks to complete
            while self.active_count() > 0 {
                std::thread::yield_now();
            }
        }
    }
}

/// Execute a function within a new scope
///
/// All tasks spawned within the scope must complete before
/// this function returns.
///
/// # Example
///
/// ```
/// use bhc_concurrent::scope::with_scope;
///
/// let result = with_scope(|scope| {
///     // Spawn tasks here...
///     42
/// });
/// assert_eq!(result, 42);
/// ```
pub fn with_scope<F, R>(f: F) -> R
where
    F: FnOnce(&Scope) -> R,
{
    let scope = Scope::new();
    let result = f(&scope);
    scope.wait_all();
    result
}

/// Execute a function within a scope with a deadline
///
/// Returns None if the deadline is exceeded before completion.
///
/// # Example
///
/// ```
/// use bhc_concurrent::scope::with_deadline;
/// use std::time::Duration;
///
/// let result = with_deadline(Duration::from_secs(5), |_scope| {
///     // Must complete within 5 seconds
///     42
/// });
/// assert_eq!(result, Some(42));
/// ```
pub fn with_deadline<F, R>(timeout: Duration, f: F) -> Option<R>
where
    F: FnOnce(&Scope) -> R,
{
    let scope = Scope::new();
    let deadline = Instant::now() + timeout;

    // Run the function
    let result = f(&scope);

    // Cancel if past deadline
    if Instant::now() >= deadline {
        scope.cancel();
    }

    // Wait for completion with timeout
    if scope.wait_all_timeout(timeout) {
        Some(result)
    } else {
        scope.cancel();
        None
    }
}

/// Spawn a task within a scope
///
/// The task will run concurrently and must complete before the scope exits.
///
/// # Example
///
/// ```
/// use bhc_concurrent::scope::{with_scope, spawn};
///
/// with_scope(|scope| {
///     let handle = spawn(scope, || {
///         // Concurrent work
///         42
///     });
///
///     // Do other work...
///
///     // Get the result
///     let result = handle.join().unwrap();
/// });
/// ```
pub fn spawn<F, T>(scope: &Scope, f: F) -> SpawnHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    scope.register_task();

    let scope_clone = scope.clone();
    let task = Task::<T>::new();
    let task_clone = task.clone();

    let join_handle = thread::spawn(move || {
        // Set task to running
        task_clone.set_state(TaskState::Running);

        // Check for cancellation before starting
        if scope_clone.is_cancelled() {
            task_clone.set_state(TaskState::Cancelled);
            scope_clone.unregister_task();
            return None;
        }

        // Run the task
        let result = f();

        // Check for cancellation after completing
        if scope_clone.is_cancelled() || task_clone.is_cancel_requested() {
            task_clone.set_state(TaskState::Cancelled);
            scope_clone.unregister_task();
            return None;
        }

        // Mark complete
        task_clone.set_state(TaskState::Completed);
        scope_clone.unregister_task();
        Some(result)
    });

    SpawnHandle {
        task,
        join_handle: Some(join_handle),
    }
}

/// Handle to a spawned task
pub struct SpawnHandle<T> {
    task: Task<T>,
    join_handle: Option<JoinHandle<Option<T>>>,
}

impl<T> SpawnHandle<T> {
    /// Wait for the task to complete and get the result
    pub fn join(mut self) -> Option<T> {
        if let Some(handle) = self.join_handle.take() {
            match handle.join() {
                Ok(result) => result,
                Err(_) => None,
            }
        } else {
            None
        }
    }

    /// Check if the task is done
    pub fn is_done(&self) -> bool {
        self.task.is_done()
    }

    /// Get the current state
    pub fn state(&self) -> TaskState {
        self.task.state()
    }

    /// Request cancellation
    pub fn cancel(&self) {
        self.task.cancel();
    }

    /// Try to get the result without blocking
    ///
    /// Returns None if the task is still running.
    pub fn try_join(&self) -> Option<bool> {
        if self.is_done() {
            Some(self.task.state() == TaskState::Completed)
        } else {
            None
        }
    }
}

/// Spawn multiple tasks and wait for all to complete
pub fn spawn_all<F, T, I>(scope: &Scope, tasks: I) -> Vec<SpawnHandle<T>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
    I: IntoIterator<Item = F>,
{
    tasks.into_iter().map(|f| spawn(scope, f)).collect()
}

/// Check if the current execution context should be cancelled
///
/// This is a cooperative cancellation checkpoint.
pub fn check_cancelled(scope: &Scope) -> bool {
    scope.is_cancelled()
}

/// Explicit cancellation checkpoint that panics if cancelled
pub fn checkpoint(scope: &Scope) {
    if scope.is_cancelled() {
        panic!("Task cancelled");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI32;

    #[test]
    fn test_scope_creation() {
        let scope = Scope::new();
        assert!(!scope.is_cancelled());
        assert_eq!(scope.active_count(), 0);
    }

    #[test]
    fn test_scope_cancellation() {
        let scope = Scope::new();
        scope.cancel();
        assert!(scope.is_cancelled());
    }

    #[test]
    fn test_child_scope_inherits_cancellation() {
        let parent = Scope::new();
        let child = parent.child();

        assert!(!child.is_cancelled());
        parent.cancel();
        assert!(child.is_cancelled());
    }

    #[test]
    fn test_with_scope() {
        let result = with_scope(|scope| {
            assert!(!scope.is_cancelled());
            42
        });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_spawn_single_task() {
        let counter = Arc::new(AtomicI32::new(0));
        let counter_clone = counter.clone();

        with_scope(|scope| {
            let handle = spawn(scope, move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                42
            });

            let result = handle.join();
            assert_eq!(result, Some(42));
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_spawn_multiple_tasks() {
        let counter = Arc::new(AtomicI32::new(0));

        with_scope(|scope| {
            let handles: Vec<_> = (0..10)
                .map(|_| {
                    let c = counter.clone();
                    spawn(scope, move || {
                        c.fetch_add(1, Ordering::SeqCst);
                    })
                })
                .collect();

            // Wait for all tasks
            for handle in handles {
                handle.join();
            }
        });

        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_wait_all_timeout() {
        let scope = Scope::new();
        scope.register_task();

        // Should timeout since we never unregister the task
        let completed = scope.wait_all_timeout(Duration::from_millis(10));
        assert!(!completed);

        // Cleanup
        scope.unregister_task();
    }

    #[test]
    fn test_spawn_with_cancellation() {
        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = executed.clone();

        with_scope(|scope| {
            scope.cancel(); // Cancel before spawning

            let handle = spawn(scope, move || {
                executed_clone.store(true, Ordering::SeqCst);
            });

            // Task should be cancelled, not executed
            handle.join();
        });

        // Task should not have executed its body
        assert!(!executed.load(Ordering::SeqCst));
    }
}
