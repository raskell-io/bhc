//! Software Transactional Memory
//!
//! STM provides composable atomic transactions for shared state.

use parking_lot::RwLock;
use std::sync::Arc;

/// A transactional variable
pub struct TVar<T> {
    value: Arc<RwLock<T>>,
}

impl<T: Clone> TVar<T> {
    /// Create a new TVar with the given initial value
    pub fn new(value: T) -> Self {
        Self {
            value: Arc::new(RwLock::new(value)),
        }
    }

    /// Read the current value (non-transactional)
    pub fn read(&self) -> T {
        self.value.read().clone()
    }

    /// Write a new value (non-transactional)
    pub fn write(&self, value: T) {
        *self.value.write() = value;
    }

    /// Modify the value (non-transactional)
    pub fn modify<F>(&self, f: F)
    where
        F: FnOnce(T) -> T,
    {
        let mut guard = self.value.write();
        let new_value = f(guard.clone());
        *guard = new_value;
    }
}

impl<T: Clone> Clone for TVar<T> {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
        }
    }
}

/// Result of an STM transaction
pub enum StmResult<T> {
    /// Transaction completed successfully
    Success(T),
    /// Transaction should retry
    Retry,
    /// Transaction failed and should not retry
    Failure(String),
}

/// Execute a transaction atomically
///
/// Note: This is a simplified implementation. Full STM would track
/// read/write sets and detect conflicts.
pub fn atomically<T, F>(f: F) -> T
where
    F: Fn() -> StmResult<T>,
{
    loop {
        match f() {
            StmResult::Success(result) => return result,
            StmResult::Retry => {
                std::thread::yield_now();
                continue;
            }
            StmResult::Failure(msg) => {
                panic!("STM transaction failed: {}", msg);
            }
        }
    }
}

/// Retry the current transaction
pub fn retry<T>() -> StmResult<T> {
    StmResult::Retry
}

/// Choose between two alternatives
pub fn or_else<T, F, G>(first: F, second: G) -> StmResult<T>
where
    F: FnOnce() -> StmResult<T>,
    G: FnOnce() -> StmResult<T>,
{
    match first() {
        StmResult::Retry => second(),
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tvar_read_write() {
        let var = TVar::new(42);
        assert_eq!(var.read(), 42);

        var.write(100);
        assert_eq!(var.read(), 100);
    }

    #[test]
    fn test_tvar_modify() {
        let var = TVar::new(10);
        var.modify(|x| x * 2);
        assert_eq!(var.read(), 20);
    }

    #[test]
    fn test_atomically() {
        let counter = TVar::new(0);
        let counter_clone = counter.clone();

        let result = atomically(|| {
            let current = counter_clone.read();
            counter_clone.write(current + 1);
            StmResult::Success(current + 1)
        });

        assert_eq!(result, 1);
        assert_eq!(counter.read(), 1);
    }
}
