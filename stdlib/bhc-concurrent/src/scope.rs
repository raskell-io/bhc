//! Structured concurrency scopes
//!
//! Scopes ensure all spawned tasks complete before the scope exits.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// A concurrency scope that manages task lifetimes
pub struct Scope {
    /// Number of active tasks
    active_tasks: Arc<AtomicUsize>,
    /// Whether the scope has been cancelled
    cancelled: Arc<AtomicBool>,
}

impl Scope {
    /// Create a new scope
    pub fn new() -> Self {
        Self {
            active_tasks: Arc::new(AtomicUsize::new(0)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if the scope has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Cancel the scope and all its tasks
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Get the number of active tasks
    pub fn active_count(&self) -> usize {
        self.active_tasks.load(Ordering::SeqCst)
    }

    /// Register a new task
    pub(crate) fn register_task(&self) {
        self.active_tasks.fetch_add(1, Ordering::SeqCst);
    }

    /// Unregister a completed task
    pub(crate) fn unregister_task(&self) {
        self.active_tasks.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        // Cancel any remaining tasks
        self.cancel();

        // Wait for all tasks to complete
        while self.active_count() > 0 {
            std::thread::yield_now();
        }
    }
}

/// Execute a function within a new scope
///
/// All tasks spawned within the scope must complete before
/// this function returns.
pub fn with_scope<F, R>(f: F) -> R
where
    F: FnOnce(&Scope) -> R,
{
    let scope = Scope::new();
    f(&scope)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_with_scope() {
        let result = with_scope(|scope| {
            assert!(!scope.is_cancelled());
            42
        });
        assert_eq!(result, 42);
    }
}
