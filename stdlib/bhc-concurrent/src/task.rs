//! Task management
//!
//! Tasks are units of concurrent execution within a scope.
//!
//! # Overview
//!
//! A Task represents a unit of work that executes concurrently.
//! Tasks go through several states during their lifecycle:
//!
//! ```text
//! New -> Running -> Completed
//!           |
//!           +---> Cancelling -> Cancelled
//!           |
//!           +---> Failed
//! ```

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

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

impl TaskState {
    /// Check if the task is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        )
    }
}

/// A handle to a spawned task
pub struct Task<T> {
    inner: Arc<TaskInner<T>>,
}

// Manual Clone implementation that doesn't require T: Clone
impl<T> Clone for Task<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct TaskInner<T> {
    state: AtomicU8,
    result: Mutex<Option<T>>,
}

impl<T> Task<T> {
    /// Create a new task
    pub fn new() -> Self {
        Self {
            inner: Arc::new(TaskInner {
                state: AtomicU8::new(TaskState::New as u8),
                result: Mutex::new(None),
            }),
        }
    }

    /// Get the current state of the task
    pub fn state(&self) -> TaskState {
        match self.inner.state.load(Ordering::SeqCst) {
            0 => TaskState::New,
            1 => TaskState::Running,
            2 => TaskState::Cancelling,
            3 => TaskState::Completed,
            4 => TaskState::Cancelled,
            _ => TaskState::Failed,
        }
    }

    /// Set the task state (internal use)
    pub(crate) fn set_state(&self, state: TaskState) {
        self.inner.state.store(state as u8, Ordering::SeqCst);
    }

    /// Check if the task is done (completed, cancelled, or failed)
    pub fn is_done(&self) -> bool {
        self.state().is_terminal()
    }

    /// Check if the task is still running
    pub fn is_running(&self) -> bool {
        matches!(self.state(), TaskState::Running)
    }

    /// Request cancellation of the task
    pub fn cancel(&self) {
        let current = self.inner.state.load(Ordering::SeqCst);
        if current == TaskState::Running as u8 || current == TaskState::New as u8 {
            self.inner
                .state
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
        matches!(self.state(), TaskState::Cancelling | TaskState::Cancelled)
    }

    /// Store a result value
    pub(crate) fn set_result(&self, value: T) {
        let mut guard = self.inner.result.lock().unwrap();
        *guard = Some(value);
    }

    /// Take the result value (consumes it)
    pub fn take_result(&self) -> Option<T> {
        let mut guard = self.inner.result.lock().unwrap();
        guard.take()
    }
}

impl<T> Default for Task<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A handle trait for tasks
pub trait TaskHandle {
    /// Check if the task is done
    fn is_done(&self) -> bool;

    /// Request cancellation
    fn cancel(&self);

    /// Get the current state
    fn get_state(&self) -> TaskState;
}

impl<T> TaskHandle for Task<T> {
    fn is_done(&self) -> bool {
        self.is_done()
    }

    fn cancel(&self) {
        Task::cancel(self);
    }

    fn get_state(&self) -> TaskState {
        self.state()
    }
}

/// Check if the current task should be cancelled
///
/// Note: In a full implementation, this would check thread-local state.
pub fn check_cancelled() -> bool {
    // Simplified implementation
    false
}

/// Explicit cancellation checkpoint
///
/// Panics if the current task has been requested to cancel.
pub fn checkpoint() {
    if check_cancelled() {
        panic!("Task cancelled");
    }
}

/// A future-like abstraction for task completion
pub struct TaskFuture<T> {
    task: Task<T>,
}

impl<T: Clone> TaskFuture<T> {
    /// Create a new task future
    pub fn new(task: Task<T>) -> Self {
        Self { task }
    }

    /// Poll for completion
    ///
    /// Returns Some(result) if complete, None if still running.
    pub fn poll(&self) -> Option<Option<T>> {
        if self.task.is_done() {
            Some(self.task.take_result())
        } else {
            None
        }
    }

    /// Block until completion
    pub fn wait(&self) -> Option<T> {
        loop {
            if let Some(result) = self.poll() {
                return result;
            }
            std::thread::yield_now();
        }
    }

    /// Block with timeout
    pub fn wait_timeout(&self, timeout: std::time::Duration) -> Option<Option<T>> {
        let deadline = std::time::Instant::now() + timeout;

        loop {
            if let Some(result) = self.poll() {
                return Some(result);
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            std::thread::yield_now();
        }
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
    fn test_task_state_transitions() {
        let task: Task<i32> = Task::new();
        assert_eq!(task.state(), TaskState::New);

        task.set_state(TaskState::Running);
        assert_eq!(task.state(), TaskState::Running);
        assert!(task.is_running());

        task.set_state(TaskState::Completed);
        assert_eq!(task.state(), TaskState::Completed);
        assert!(task.is_done());
    }

    #[test]
    fn test_task_cancel() {
        let task: Task<i32> = Task::new();
        task.set_state(TaskState::Running);
        task.cancel();
        assert!(task.is_cancel_requested());
    }

    #[test]
    fn test_task_result() {
        let task: Task<i32> = Task::new();
        task.set_result(42);

        assert_eq!(task.take_result(), Some(42));
        assert_eq!(task.take_result(), None); // Consumed
    }

    #[test]
    fn test_task_clone() {
        let task: Task<i32> = Task::new();
        let task2 = task.clone();

        task.set_state(TaskState::Running);
        assert_eq!(task2.state(), TaskState::Running);
    }

    #[test]
    fn test_task_state_is_terminal() {
        assert!(!TaskState::New.is_terminal());
        assert!(!TaskState::Running.is_terminal());
        assert!(!TaskState::Cancelling.is_terminal());
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Cancelled.is_terminal());
        assert!(TaskState::Failed.is_terminal());
    }
}
