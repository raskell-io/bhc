//! Task management
//!
//! Tasks are units of concurrent execution within a scope.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Task state
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Task is created but not started
    New = 0,
    /// Task is running
    Running = 1,
    /// Task is being cancelled
    Cancelling = 2,
    /// Task has completed successfully
    Completed = 3,
    /// Task was cancelled
    Cancelled = 4,
    /// Task failed with an error
    Failed = 5,
}

/// A handle to a spawned task
pub struct Task<T> {
    state: Arc<AtomicU8>,
    result: Arc<std::sync::Mutex<Option<T>>>,
}

impl<T> Task<T> {
    /// Create a new task
    pub fn new() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(TaskState::New as u8)),
            result: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Get the current state of the task
    pub fn state(&self) -> TaskState {
        match self.state.load(Ordering::SeqCst) {
            0 => TaskState::New,
            1 => TaskState::Running,
            2 => TaskState::Cancelling,
            3 => TaskState::Completed,
            4 => TaskState::Cancelled,
            _ => TaskState::Failed,
        }
    }

    /// Check if the task is done (completed, cancelled, or failed)
    pub fn is_done(&self) -> bool {
        matches!(
            self.state(),
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        )
    }

    /// Request cancellation of the task
    pub fn cancel(&self) {
        let current = self.state.load(Ordering::SeqCst);
        if current == TaskState::Running as u8 {
            self.state
                .compare_exchange(
                    current,
                    TaskState::Cancelling as u8,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .ok();
        }
    }

    /// Check if cancellation has been requested
    pub fn is_cancel_requested(&self) -> bool {
        self.state.load(Ordering::SeqCst) == TaskState::Cancelling as u8
    }
}

impl<T> Default for Task<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if the current task should be cancelled
pub fn check_cancelled() -> bool {
    // In actual implementation, this checks the current task's state
    false
}

/// Explicit cancellation checkpoint
pub fn checkpoint() {
    if check_cancelled() {
        panic!("Task cancelled");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_state() {
        let task: Task<i32> = Task::new();
        assert_eq!(task.state(), TaskState::New);
        assert!(!task.is_done());
    }

    #[test]
    fn test_task_cancel() {
        let task: Task<i32> = Task::new();
        task.state.store(TaskState::Running as u8, Ordering::SeqCst);
        task.cancel();
        assert!(task.is_cancel_requested());
    }
}
