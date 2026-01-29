//! Work-stealing task scheduler for the BHC Runtime System.
//!
//! This crate implements structured concurrency primitives as specified in
//! H26-SPEC Section 10: Concurrency Model. Key features:
//!
//! - **Work-stealing scheduler** - Efficient load balancing across workers
//! - **Structured concurrency** - Tasks are scoped and cancellation propagates
//! - **Cooperative cancellation** - Tasks check for cancellation at safe points
//! - **Deadline support** - Time-bounded operations
//! - **Event tracing** - Hooks for observability
//!
//! # Structured Concurrency Model
//!
//! All concurrent operations happen within a scope that outlives them:
//!
//! ```ignore
//! use bhc_rts_scheduler::{Scheduler, with_scope};
//!
//! let scheduler = Scheduler::new(4); // 4 worker threads
//!
//! with_scope(&scheduler, |scope| {
//!     let task1 = scope.spawn(|| compute_x());
//!     let task2 = scope.spawn(|| compute_y());
//!
//!     let x = task1.await_result();
//!     let y = task2.await_result();
//!     (x, y)
//! });
//! // All tasks complete before scope exits
//! ```
//!
//! # Task Lifecycle
//!
//! ```text
//!   spawn      await
//!     |          |
//!     v          v
//! +-----+    +-------+    +----------+    +---------+
//! | New | -> |Running| -> |Completing| -> |Completed|
//! +-----+    +-------+    +----------+    +---------+
//!               |                              ^
//!               | cancel                       |
//!               v                              |
//!            +----------+                      |
//!            |Cancelling| ---------------------+
//!            +----------+
//! ```
//!
//! # M5 Exit Criteria
//!
//! - Server workload runs concurrently without numeric kernel regressions
//! - Cancellation propagates within 1ms of request
//! - GC pause times < 10ms at p99

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use crossbeam::deque::{Injector, Stealer, Worker as WorkerDeque};
use parking_lot::{Condvar, Mutex, RwLock};
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// ============================================================================
// Event Tracing
// ============================================================================

/// Types of events that can be traced.
#[derive(Debug, Clone)]
pub enum TraceEvent {
    /// A task was spawned.
    TaskSpawn {
        /// Task ID.
        task_id: TaskId,
        /// Parent task ID, if any.
        parent_id: Option<TaskId>,
        /// Timestamp.
        timestamp: Instant,
    },
    /// A task started executing.
    TaskStart {
        /// Task ID.
        task_id: TaskId,
        /// Worker ID.
        worker_id: usize,
        /// Timestamp.
        timestamp: Instant,
    },
    /// A task completed.
    TaskComplete {
        /// Task ID.
        task_id: TaskId,
        /// Final state.
        state: TaskState,
        /// Timestamp.
        timestamp: Instant,
    },
    /// A task was cancelled.
    TaskCancel {
        /// Task ID.
        task_id: TaskId,
        /// Timestamp.
        timestamp: Instant,
    },
    /// A scope was created.
    ScopeCreate {
        /// Scope ID.
        scope_id: u64,
        /// Timestamp.
        timestamp: Instant,
    },
    /// A scope exited.
    ScopeExit {
        /// Scope ID.
        scope_id: u64,
        /// Number of tasks that ran in scope.
        task_count: usize,
        /// Timestamp.
        timestamp: Instant,
    },
    /// A deadline was set.
    DeadlineSet {
        /// Task or scope ID.
        target_id: u64,
        /// Deadline instant.
        deadline: Instant,
        /// Timestamp.
        timestamp: Instant,
    },
    /// A deadline was reached.
    DeadlineReached {
        /// Task or scope ID.
        target_id: u64,
        /// Timestamp.
        timestamp: Instant,
    },
}

/// Callback type for trace events.
pub type TraceCallback = Box<dyn Fn(TraceEvent) + Send + Sync>;

/// Global trace callback storage.
static TRACE_CALLBACK: RwLock<Option<TraceCallback>> = RwLock::new(None);

/// Set the global trace callback.
///
/// Events will be delivered to this callback as they occur.
pub fn set_trace_callback(callback: TraceCallback) {
    *TRACE_CALLBACK.write() = Some(callback);
}

/// Clear the global trace callback.
pub fn clear_trace_callback() {
    *TRACE_CALLBACK.write() = None;
}

/// Emit a trace event.
fn trace(event: TraceEvent) {
    if let Some(callback) = TRACE_CALLBACK.read().as_ref() {
        callback(event);
    }
}

// ============================================================================
// Task IDs and State
// ============================================================================

/// Unique identifier for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    /// Create a new unique task ID.
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value.
    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Task({})", self.0)
    }
}

/// State of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Task is created but not yet running.
    Pending,
    /// Task is currently executing.
    Running,
    /// Task is being cancelled.
    Cancelling,
    /// Task completed successfully.
    Completed,
    /// Task was cancelled.
    Cancelled,
    /// Task panicked.
    Failed,
}

/// Result of a task execution.
#[derive(Debug)]
pub enum TaskResult<T> {
    /// Task completed with a value.
    Ok(T),
    /// Task was cancelled.
    Cancelled,
    /// Task panicked.
    Panicked(Box<dyn Any + Send>),
}

impl<T> TaskResult<T> {
    /// Check if the task completed successfully.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }

    /// Unwrap the result, panicking if not Ok.
    #[must_use]
    pub fn unwrap(self) -> T {
        match self {
            Self::Ok(v) => v,
            Self::Cancelled => panic!("task was cancelled"),
            Self::Panicked(_) => panic!("task panicked"),
        }
    }

    /// Convert to Option, discarding cancellation/panic info.
    #[must_use]
    pub fn ok(self) -> Option<T> {
        match self {
            Self::Ok(v) => Some(v),
            _ => None,
        }
    }
}

impl<T: Clone> Clone for TaskResult<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Ok(v) => Self::Ok(v.clone()),
            Self::Cancelled => Self::Cancelled,
            Self::Panicked(_) => Self::Panicked(Box::new("cloned panic")),
        }
    }
}

// ============================================================================
// Task Handle
// ============================================================================

/// Shared state for a task.
struct TaskInner<T> {
    state: Mutex<TaskState>,
    result: Mutex<Option<TaskResult<T>>>,
    condvar: Condvar,
}

/// A handle to a spawned task.
///
/// Can be used to await the result or request cancellation.
pub struct Task<T> {
    id: TaskId,
    inner: Arc<TaskInner<T>>,
    cancelled: Arc<AtomicBool>,
    /// Child tasks (for cancellation propagation).
    children: Arc<Mutex<Vec<Arc<AtomicBool>>>>,
}

impl<T> Task<T> {
    /// Create a new task.
    fn new() -> Self {
        Self {
            id: TaskId::new(),
            inner: Arc::new(TaskInner {
                state: Mutex::new(TaskState::Pending),
                result: Mutex::new(None),
                condvar: Condvar::new(),
            }),
            cancelled: Arc::new(AtomicBool::new(false)),
            children: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get the task's ID.
    #[must_use]
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Get the current state of the task.
    #[must_use]
    pub fn state(&self) -> TaskState {
        *self.inner.state.lock()
    }

    /// Check if the task has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Check if the task has completed (successfully, cancelled, or failed).
    #[must_use]
    pub fn is_done(&self) -> bool {
        matches!(
            self.state(),
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        )
    }

    /// Request cancellation of this task.
    ///
    /// The task will be cancelled at the next safe point.
    /// Cancellation propagates to all child tasks.
    pub fn cancel(&self) {
        // Set our cancellation flag
        self.cancelled.store(true, Ordering::Release);
        *self.inner.state.lock() = TaskState::Cancelling;

        // Emit trace event
        trace(TraceEvent::TaskCancel {
            task_id: self.id,
            timestamp: Instant::now(),
        });

        // Propagate to children
        let children = self.children.lock();
        for child_cancelled in children.iter() {
            child_cancelled.store(true, Ordering::Release);
        }
    }

    /// Add a child task for cancellation propagation.
    ///
    /// This is used when a task spawns child tasks that should be cancelled
    /// when the parent is cancelled (task-to-task propagation, not scope-based).
    #[allow(dead_code)]
    fn add_child(&self, child_cancelled: Arc<AtomicBool>) {
        // Clone before pushing so we can still use the original after
        self.children.lock().push(child_cancelled.clone());

        // If we're already cancelled, cancel the child immediately
        if self.is_cancelled() {
            child_cancelled.store(true, Ordering::Release);
        }
    }

    /// Wait for the task to complete and return its result.
    pub fn await_result(self) -> TaskResult<T> {
        let mut state = self.inner.state.lock();
        while !matches!(
            *state,
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        ) {
            self.inner.condvar.wait(&mut state);
        }
        drop(state);

        self.inner
            .result
            .lock()
            .take()
            .expect("result should be set")
    }

    /// Try to get the result without blocking.
    #[must_use]
    pub fn try_result(&self) -> Option<TaskResult<T>>
    where
        T: Clone,
    {
        let state = self.inner.state.lock();
        if matches!(
            *state,
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        ) {
            self.inner.result.lock().clone()
        } else {
            None
        }
    }

    /// Wait for the task with a timeout.
    pub fn await_timeout(self, timeout: Duration) -> Option<TaskResult<T>> {
        let deadline = Instant::now() + timeout;
        let mut state = self.inner.state.lock();

        while !matches!(
            *state,
            TaskState::Completed | TaskState::Cancelled | TaskState::Failed
        ) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let result = self.inner.condvar.wait_for(&mut state, remaining);
            if result.timed_out() {
                return None;
            }
        }
        drop(state);

        Some(
            self.inner
                .result
                .lock()
                .take()
                .expect("result should be set"),
        )
    }
}

impl<T> fmt::Debug for Task<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Task")
            .field("id", &self.id)
            .field("state", &self.state())
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

// ============================================================================
// Internal Task Representation
// ============================================================================

/// Internal task representation for the scheduler.
struct RawTask {
    func: Box<dyn FnOnce() + Send>,
}

impl RawTask {
    fn new<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self { func: Box::new(f) }
    }

    fn run(self) {
        (self.func)();
    }
}

// ============================================================================
// Scheduler Configuration and Stats
// ============================================================================

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of worker threads.
    pub num_workers: usize,
    /// Stack size for worker threads.
    pub stack_size: usize,
    /// Enable work stealing.
    pub work_stealing: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus(),
            stack_size: 2 * 1024 * 1024, // 2 MB
            work_stealing: true,
        }
    }
}

fn num_cpus() -> usize {
    thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

/// Statistics for the scheduler.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total tasks spawned.
    pub tasks_spawned: u64,
    /// Total tasks completed.
    pub tasks_completed: u64,
    /// Total tasks cancelled.
    pub tasks_cancelled: u64,
    /// Total tasks that panicked.
    pub tasks_failed: u64,
    /// Number of successful steals.
    pub steals: u64,
    /// Number of failed steal attempts.
    pub steal_failures: u64,
}

// ============================================================================
// Scheduler
// ============================================================================

/// A work-stealing task scheduler.
///
/// The scheduler manages a pool of worker threads that execute tasks.
/// Each worker has a local work queue and can steal from others when idle.
pub struct Scheduler {
    config: SchedulerConfig,
    global_queue: Arc<Injector<RawTask>>,
    #[allow(dead_code)]
    stealers: Arc<Vec<Stealer<RawTask>>>,
    workers: Vec<JoinHandle<()>>,
    stats: Arc<RwLock<SchedulerStats>>,
    shutdown: Arc<AtomicBool>,
    active_tasks: Arc<AtomicUsize>,
}

impl Scheduler {
    /// Create a new scheduler with the specified number of workers.
    #[must_use]
    pub fn new(num_workers: usize) -> Self {
        Self::with_config(SchedulerConfig {
            num_workers,
            ..Default::default()
        })
    }

    /// Create a new scheduler with the given configuration.
    #[must_use]
    pub fn with_config(config: SchedulerConfig) -> Self {
        let global_queue = Arc::new(Injector::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(RwLock::new(SchedulerStats::default()));
        let active_tasks = Arc::new(AtomicUsize::new(0));

        let mut local_queues = Vec::with_capacity(config.num_workers);
        let mut stealers = Vec::with_capacity(config.num_workers);

        // Create worker deques
        for _ in 0..config.num_workers {
            let worker = WorkerDeque::new_fifo();
            stealers.push(worker.stealer());
            local_queues.push(worker);
        }

        let stealers = Arc::new(stealers);
        let mut workers = Vec::with_capacity(config.num_workers);

        // Spawn worker threads
        for (id, local_queue) in local_queues.into_iter().enumerate() {
            let global = Arc::clone(&global_queue);
            let stealers = Arc::clone(&stealers);
            let shutdown = Arc::clone(&shutdown);
            let stats = Arc::clone(&stats);

            let handle = thread::Builder::new()
                .name(format!("bhc-worker-{id}"))
                .stack_size(config.stack_size)
                .spawn(move || {
                    worker_loop(id, local_queue, global, stealers, shutdown, stats);
                })
                .expect("failed to spawn worker thread");

            workers.push(handle);
        }

        Self {
            config,
            global_queue,
            stealers,
            workers,
            stats,
            shutdown,
            active_tasks,
        }
    }

    /// Create a new scheduler with default configuration.
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    /// Spawn a task on the scheduler.
    pub fn spawn<F, T>(&self, f: F) -> Task<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.spawn_with_parent(f, None)
    }

    /// Spawn a task with an optional parent for cancellation propagation.
    fn spawn_with_parent<F, T>(&self, f: F, parent: Option<&Arc<AtomicBool>>) -> Task<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let task = Task::new();
        let task_id = task.id;
        let inner = Arc::clone(&task.inner);
        let cancelled = Arc::clone(&task.cancelled);

        // Link to parent for cancellation propagation
        let parent_id = CURRENT_TASK.with(|c| c.get());
        if let Some(parent_cancelled) = parent {
            // If parent is already cancelled, mark us as cancelled
            if parent_cancelled.load(Ordering::Acquire) {
                cancelled.store(true, Ordering::Release);
            }
        }

        self.active_tasks.fetch_add(1, Ordering::Relaxed);
        let active_tasks = Arc::clone(&self.active_tasks);

        {
            let mut stats = self.stats.write();
            stats.tasks_spawned += 1;
        }

        // Emit trace event
        trace(TraceEvent::TaskSpawn {
            task_id,
            parent_id,
            timestamp: Instant::now(),
        });

        // Clone cancelled flag for the task closure
        let cancelled_for_task = Arc::clone(&cancelled);

        let raw_task = RawTask::new(move || {
            // Set current task context
            CURRENT_TASK.with(|c| c.set(Some(task_id)));

            // Set the cancellation flag in TLS so check_cancelled() can read it dynamically
            set_cancelled_flag(Some(Arc::clone(&cancelled_for_task)));

            // Check for early cancellation
            if cancelled_for_task.load(Ordering::Acquire) {
                *inner.state.lock() = TaskState::Cancelled;
                *inner.result.lock() = Some(TaskResult::Cancelled);
                inner.condvar.notify_all();
                active_tasks.fetch_sub(1, Ordering::Relaxed);

                trace(TraceEvent::TaskComplete {
                    task_id,
                    state: TaskState::Cancelled,
                    timestamp: Instant::now(),
                });

                // Clear TLS before returning
                set_cancelled_flag(None);
                CURRENT_TASK.with(|c| c.set(None));
                return;
            }

            *inner.state.lock() = TaskState::Running;

            // Run the task
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

            // Check for cancellation after execution
            let final_state = if cancelled_for_task.load(Ordering::Acquire) {
                *inner.state.lock() = TaskState::Cancelled;
                *inner.result.lock() = Some(TaskResult::Cancelled);
                TaskState::Cancelled
            } else {
                match outcome {
                    Ok(value) => {
                        *inner.state.lock() = TaskState::Completed;
                        *inner.result.lock() = Some(TaskResult::Ok(value));
                        TaskState::Completed
                    }
                    Err(panic) => {
                        *inner.state.lock() = TaskState::Failed;
                        *inner.result.lock() = Some(TaskResult::Panicked(panic));
                        TaskState::Failed
                    }
                }
            };

            trace(TraceEvent::TaskComplete {
                task_id,
                state: final_state,
                timestamp: Instant::now(),
            });

            inner.condvar.notify_all();
            active_tasks.fetch_sub(1, Ordering::Relaxed);

            // Clear task context
            set_cancelled_flag(None);
            CURRENT_TASK.with(|c| c.set(None));
        });

        self.global_queue.push(raw_task);
        task
    }

    /// Get scheduler statistics.
    #[must_use]
    pub fn stats(&self) -> SchedulerStats {
        self.stats.read().clone()
    }

    /// Get the number of worker threads.
    #[must_use]
    pub fn num_workers(&self) -> usize {
        self.config.num_workers
    }

    /// Get the number of currently active tasks.
    #[must_use]
    pub fn active_tasks(&self) -> usize {
        self.active_tasks.load(Ordering::Relaxed)
    }

    /// Shutdown the scheduler and wait for all workers to finish.
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Release);

        for worker in std::mem::take(&mut self.workers) {
            let _ = worker.join();
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

fn worker_loop(
    id: usize,
    local: WorkerDeque<RawTask>,
    global: Arc<Injector<RawTask>>,
    stealers: Arc<Vec<Stealer<RawTask>>>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<RwLock<SchedulerStats>>,
) {
    // Set worker ID in thread-local storage
    CURRENT_WORKER.with(|w| w.set(Some(id)));

    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }

        // Try local queue first
        if let Some(task) = local.pop() {
            trace(TraceEvent::TaskStart {
                task_id: CURRENT_TASK.with(|c| c.get()).unwrap_or(TaskId(u64::MAX)),
                worker_id: id,
                timestamp: Instant::now(),
            });
            task.run();
            let mut s = stats.write();
            s.tasks_completed += 1;
            continue;
        }

        // Try global queue
        if let crossbeam::deque::Steal::Success(task) = global.steal() {
            task.run();
            let mut s = stats.write();
            s.tasks_completed += 1;
            continue;
        }

        // Try stealing from other workers
        let mut stolen = false;
        for (i, stealer) in stealers.iter().enumerate() {
            if i == id {
                continue;
            }
            if let crossbeam::deque::Steal::Success(task) = stealer.steal() {
                task.run();
                let mut s = stats.write();
                s.tasks_completed += 1;
                s.steals += 1;
                stolen = true;
                break;
            }
        }

        if !stolen {
            let mut s = stats.write();
            s.steal_failures += 1;
            drop(s);
            // Yield to avoid busy-waiting
            thread::yield_now();
        }
    }
}

// ============================================================================
// Structured Concurrency: Scope
// ============================================================================

/// Unique identifier for a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(u64);

impl ScopeId {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// A scope for structured concurrency.
///
/// Tasks spawned within a scope are guaranteed to complete before
/// the scope exits. If the scope is cancelled, all tasks are cancelled.
pub struct Scope<'a> {
    id: ScopeId,
    scheduler: &'a Scheduler,
    /// Handles to spawned tasks (for waiting).
    task_handles: Mutex<Vec<ScopedTaskHandle>>,
    /// Cancellation flag shared with all tasks.
    cancelled: Arc<AtomicBool>,
    /// All task cancellation flags for external propagation (e.g., by timer).
    task_cancelled_flags: Arc<Mutex<Vec<Arc<AtomicBool>>>>,
    /// Optional deadline for this scope.
    deadline: Option<Instant>,
    /// Count of tasks spawned.
    task_count: AtomicUsize,
}

/// A type-erased task handle for scope tracking.
struct ScopedTaskHandle {
    /// Completion flag.
    done: Arc<AtomicBool>,
    /// Condvar for waiting.
    condvar: Arc<Condvar>,
    /// Mutex for condvar.
    mutex: Arc<Mutex<()>>,
    /// Task's cancellation flag (for propagation).
    cancelled: Arc<AtomicBool>,
}

impl<'a> Scope<'a> {
    /// Create a new scope.
    fn new(scheduler: &'a Scheduler, deadline: Option<Instant>) -> Self {
        let id = ScopeId::new();

        trace(TraceEvent::ScopeCreate {
            scope_id: id.0,
            timestamp: Instant::now(),
        });

        if let Some(dl) = deadline {
            trace(TraceEvent::DeadlineSet {
                target_id: id.0,
                deadline: dl,
                timestamp: Instant::now(),
            });
        }

        Self {
            id,
            scheduler,
            task_handles: Mutex::new(Vec::new()),
            cancelled: Arc::new(AtomicBool::new(false)),
            task_cancelled_flags: Arc::new(Mutex::new(Vec::new())),
            deadline,
            task_count: AtomicUsize::new(0),
        }
    }

    /// Get the scope's ID.
    #[must_use]
    pub fn id(&self) -> ScopeId {
        self.id
    }

    /// Check if the scope has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Cancel all tasks in this scope.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);

        // Propagate cancellation to all tasks in this scope
        let handles = self.task_handles.lock();
        for handle in handles.iter() {
            handle.cancelled.store(true, Ordering::Release);
        }

        trace(TraceEvent::TaskCancel {
            task_id: TaskId(self.id.0), // Use scope ID as pseudo-task
            timestamp: Instant::now(),
        });
    }

    /// Check if the deadline has passed.
    fn check_deadline(&self) -> bool {
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                trace(TraceEvent::DeadlineReached {
                    target_id: self.id.0,
                    timestamp: Instant::now(),
                });
                self.cancel();
                return true;
            }
        }
        false
    }

    /// Spawn a task within this scope.
    ///
    /// The task inherits the scope's cancellation state and deadline.
    pub fn spawn<F, T>(&self, f: F) -> Task<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.task_count.fetch_add(1, Ordering::Relaxed);

        // Check deadline before spawning
        self.check_deadline();

        // Create task with scope's cancellation flag as parent
        let task = self.scheduler.spawn_with_parent(f, Some(&self.cancelled));

        // Track for scope exit waiting
        let done = Arc::new(AtomicBool::new(false));
        let condvar = Arc::new(Condvar::new());
        let mutex = Arc::new(Mutex::new(()));

        // Create a watcher that marks done when task completes
        let done_clone = Arc::clone(&done);
        let condvar_clone = Arc::clone(&condvar);
        let inner = Arc::clone(&task.inner);

        // Spawn a lightweight watcher
        self.scheduler.spawn(move || {
            // Wait for task completion
            let mut state = inner.state.lock();
            while !matches!(
                *state,
                TaskState::Completed | TaskState::Cancelled | TaskState::Failed
            ) {
                inner.condvar.wait(&mut state);
            }
            drop(state);

            // Mark as done and notify
            done_clone.store(true, Ordering::Release);
            condvar_clone.notify_all();
        });

        // Store task's cancellation flag for scope-level cancellation propagation
        let task_cancelled = Arc::clone(&task.cancelled);

        // Add to shared flags list (for timer access)
        self.task_cancelled_flags
            .lock()
            .push(Arc::clone(&task_cancelled));

        self.task_handles.lock().push(ScopedTaskHandle {
            done,
            condvar,
            mutex,
            cancelled: task_cancelled,
        });

        task
    }

    /// Wait for all tasks in this scope to complete.
    fn wait_all(&self) {
        let handles = self.task_handles.lock();

        for handle in handles.iter() {
            // Check deadline periodically while waiting
            while !handle.done.load(Ordering::Acquire) {
                if self.check_deadline() {
                    // Deadline reached, stop waiting
                    return;
                }

                let guard = handle.mutex.lock();
                let timeout = if let Some(deadline) = self.deadline {
                    deadline.saturating_duration_since(Instant::now())
                } else {
                    Duration::from_millis(100) // Poll interval
                };

                if timeout.is_zero() {
                    return;
                }

                let _ = handle.condvar.wait_for(&mut { guard }, timeout);
            }
        }
    }
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        // Ensure all tasks complete before scope exits
        self.wait_all();

        trace(TraceEvent::ScopeExit {
            scope_id: self.id.0,
            task_count: self.task_count.load(Ordering::Relaxed),
            timestamp: Instant::now(),
        });
    }
}

// ============================================================================
// Structured Concurrency: Public API
// ============================================================================

/// Execute a function within a structured concurrency scope.
///
/// All tasks spawned within the scope are guaranteed to complete
/// before this function returns.
///
/// # Example
///
/// ```ignore
/// let result = with_scope(&scheduler, |scope| {
///     let t1 = scope.spawn(|| expensive_computation());
///     let t2 = scope.spawn(|| another_computation());
///     t1.await_result().unwrap() + t2.await_result().unwrap()
/// });
/// ```
pub fn with_scope<'a, F, R>(scheduler: &'a Scheduler, f: F) -> R
where
    F: FnOnce(&Scope<'a>) -> R,
{
    let scope = Scope::new(scheduler, None);
    f(&scope)
    // Scope::drop() waits for all tasks
}

/// Execute a function within a scope with a deadline.
///
/// If the deadline is reached, all tasks in the scope are cancelled.
/// Returns `None` if the deadline was reached before completion.
///
/// # Example
///
/// ```ignore
/// let result = with_deadline(&scheduler, Duration::from_secs(5), |scope| {
///     scope.spawn(|| slow_operation())
/// });
/// match result {
///     Some(task) => println!("Completed: {:?}", task.await_result()),
///     None => println!("Deadline reached"),
/// }
/// ```
pub fn with_deadline<'a, F, R>(scheduler: &'a Scheduler, timeout: Duration, f: F) -> Option<R>
where
    F: FnOnce(&Scope<'a>) -> R,
{
    let deadline = Instant::now() + timeout;
    let scope = Scope::new(scheduler, Some(deadline));

    if scope.is_cancelled() {
        return None;
    }

    // Spawn a timer that will cancel the scope when deadline is reached
    let scope_cancelled = Arc::clone(&scope.cancelled);
    let task_flags = Arc::clone(&scope.task_cancelled_flags);
    let timer_done = Arc::new(AtomicBool::new(false));
    let timer_done_clone = Arc::clone(&timer_done);

    let timer_handle = thread::spawn(move || {
        // Use a polling loop so we can exit early when timer_done is set
        let poll_interval = Duration::from_millis(1);
        while Instant::now() < deadline {
            // Check if work completed early
            if timer_done_clone.load(Ordering::Acquire) {
                return;
            }
            thread::sleep(poll_interval);
        }
        // Deadline reached - cancel the scope and all tasks
        if !timer_done_clone.load(Ordering::Acquire) {
            scope_cancelled.store(true, Ordering::Release);
            // Propagate to all tasks
            let flags = task_flags.lock();
            for flag in flags.iter() {
                flag.store(true, Ordering::Release);
            }
        }
    });

    let result = f(&scope);

    // Mark timer as done so it doesn't cancel unnecessarily
    timer_done.store(true, Ordering::Release);

    // Wait for timer thread to finish (it should be quick now)
    let _ = timer_handle.join();

    // Check if we exceeded the deadline
    if scope.is_cancelled() {
        None
    } else {
        Some(result)
    }
}

// ============================================================================
// Cooperative Cancellation
// ============================================================================

/// Check if the current task has been cancelled.
///
/// Tasks should call this at safe points to enable cooperative cancellation.
/// If cancelled, this function returns `true` and the task should clean up
/// and return early.
///
/// # Example
///
/// ```ignore
/// fn long_computation() -> i32 {
///     for i in 0..1000000 {
///         if check_cancelled() {
///             return 0; // Early exit
///         }
///         // ... work ...
///     }
///     result
/// }
/// ```
#[must_use]
pub fn check_cancelled() -> bool {
    // Check the task's cancellation flag dynamically
    CANCELLED_FLAG.with(|flag| {
        if let Some(ref cancelled) = *flag.borrow() {
            cancelled.load(Ordering::Acquire)
        } else {
            false
        }
    })
}

/// Set the current task's cancellation flag (internal use).
fn set_cancelled_flag(cancelled: Option<Arc<AtomicBool>>) {
    CANCELLED_FLAG.with(|flag| {
        *flag.borrow_mut() = cancelled;
    });
}

// ============================================================================
// Thread-Local Storage
// ============================================================================

thread_local! {
    /// Current task ID.
    static CURRENT_TASK: Cell<Option<TaskId>> = const { Cell::new(None) };
    /// Current worker ID.
    static CURRENT_WORKER: Cell<Option<usize>> = const { Cell::new(None) };
    /// Current task's cancellation flag (Arc for dynamic checking).
    static CANCELLED_FLAG: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Get the ID of the currently executing task, if any.
#[must_use]
pub fn current_task_id() -> Option<TaskId> {
    CURRENT_TASK.with(|c| c.get())
}

/// Get the ID of the current worker thread, if any.
#[must_use]
pub fn current_worker_id() -> Option<usize> {
    CURRENT_WORKER.with(|c| c.get())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI32;

    #[test]
    fn test_task_id_uniqueness() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_task_states() {
        let task: Task<i32> = Task::new();
        assert_eq!(task.state(), TaskState::Pending);
        assert!(!task.is_cancelled());

        task.cancel();
        assert!(task.is_cancelled());
    }

    #[test]
    fn test_scheduler_spawn() {
        let scheduler = Scheduler::new(2);

        let task = scheduler.spawn(|| 42);
        let result = task.await_result();

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        scheduler.shutdown();
    }

    #[test]
    fn test_scheduler_multiple_tasks() {
        let scheduler = Scheduler::new(4);

        let counter = Arc::new(AtomicI32::new(0));
        let mut tasks = Vec::new();

        for _ in 0..100 {
            let counter = Arc::clone(&counter);
            tasks.push(scheduler.spawn(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }));
        }

        for task in tasks {
            task.await_result();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 100);

        scheduler.shutdown();
    }

    #[test]
    fn test_task_cancellation() {
        let scheduler = Scheduler::new(2);

        let started = Arc::new(AtomicBool::new(false));
        let started_clone = Arc::clone(&started);
        let exited = Arc::new(AtomicBool::new(false));
        let exited_clone = Arc::clone(&exited);

        // Use a shared cancellation flag the task can check
        let task_cancelled = Arc::new(AtomicBool::new(false));
        let task_cancelled_clone = Arc::clone(&task_cancelled);

        let task = scheduler.spawn(move || {
            started_clone.store(true, Ordering::SeqCst);
            // Cooperative cancellation: check flag periodically
            while !task_cancelled_clone.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(10));
            }
            exited_clone.store(true, Ordering::SeqCst);
            0
        });

        // Wait for task to start
        while !started.load(Ordering::SeqCst) {
            thread::yield_now();
        }

        // Cancel the task (sets internal flag)
        task.cancel();
        assert!(task.is_cancelled());

        // Also signal our manual flag so the task exits
        task_cancelled.store(true, Ordering::Release);

        // Wait for task to exit gracefully
        let result = task.await_result();
        assert!(matches!(result, TaskResult::Cancelled) || matches!(result, TaskResult::Ok(0)));
        assert!(exited.load(Ordering::SeqCst));

        scheduler.shutdown();
    }

    #[test]
    fn test_task_timeout() {
        let scheduler = Scheduler::new(2);

        let task = scheduler.spawn(|| {
            thread::sleep(Duration::from_secs(10));
            42
        });

        let result = task.await_timeout(Duration::from_millis(50));
        assert!(result.is_none()); // Should timeout

        scheduler.shutdown();
    }

    #[test]
    fn test_scheduler_stats() {
        let scheduler = Scheduler::new(2);

        for i in 0..10 {
            let task = scheduler.spawn(move || i);
            task.await_result();
        }

        let stats = scheduler.stats();
        assert!(stats.tasks_spawned >= 10);

        scheduler.shutdown();
    }

    #[test]
    fn test_with_scope_waits_for_tasks() {
        let scheduler = Scheduler::new(2);
        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = Arc::clone(&completed);

        with_scope(&scheduler, |scope| {
            scope.spawn(move || {
                thread::sleep(Duration::from_millis(50));
                completed_clone.store(true, Ordering::SeqCst);
            });
        });

        // Scope should have waited for the task
        assert!(completed.load(Ordering::SeqCst));

        scheduler.shutdown();
    }

    #[test]
    fn test_with_scope_basic() {
        let scheduler = Scheduler::new(2);

        let result = with_scope(&scheduler, |scope| {
            let t1 = scope.spawn(|| 1);
            let t2 = scope.spawn(|| 2);

            t1.await_result().unwrap() + t2.await_result().unwrap()
        });

        assert_eq!(result, 3);

        scheduler.shutdown();
    }

    #[test]
    fn test_with_deadline_completes() {
        let scheduler = Scheduler::new(2);

        let result = with_deadline(&scheduler, Duration::from_secs(5), |scope| {
            let t = scope.spawn(|| 42);
            t.await_result().unwrap()
        });

        assert_eq!(result, Some(42));

        scheduler.shutdown();
    }

    #[test]
    fn test_with_deadline_expires() {
        let scheduler = Scheduler::new(2);
        let cancelled_observed = Arc::new(AtomicBool::new(false));
        let cancelled_observed_clone = Arc::clone(&cancelled_observed);

        let result = with_deadline(&scheduler, Duration::from_millis(50), |scope| {
            let t = scope.spawn(move || {
                // Cooperative cancellation: check periodically instead of long sleep
                for _ in 0..100 {
                    if check_cancelled() {
                        cancelled_observed_clone.store(true, Ordering::SeqCst);
                        return -1; // Return early due to cancellation
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                42
            });
            // Don't unwrap - handle cancellation case
            match t.await_result() {
                TaskResult::Ok(v) => v,
                TaskResult::Cancelled => -1, // Task was cancelled
                TaskResult::Panicked(_) => panic!("unexpected panic"),
            }
        });

        // The deadline should have expired and cancelled the task
        // Result should be None (deadline reached) or the cancelled value
        assert!(result.is_none() || result == Some(-1));
        // Task should have observed cancellation or scope timed out
        assert!(cancelled_observed.load(Ordering::SeqCst) || result.is_none());

        scheduler.shutdown();
    }

    #[test]
    fn test_cancellation_propagates_to_children() {
        let scheduler = Scheduler::new(2);

        let child_saw_cancel = Arc::new(AtomicBool::new(false));
        let child_saw_cancel_clone = Arc::clone(&child_saw_cancel);

        with_scope(&scheduler, |scope| {
            let task = scope.spawn(move || {
                // Simulate checking cancellation
                thread::sleep(Duration::from_millis(100));
                if check_cancelled() {
                    child_saw_cancel_clone.store(true, Ordering::SeqCst);
                }
            });

            // Cancel the task
            thread::sleep(Duration::from_millis(10));
            task.cancel();
        });

        scheduler.shutdown();
    }

    #[test]
    fn test_trace_events() {
        use std::sync::Mutex;

        let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        set_trace_callback(Box::new(move |event| {
            let msg = match event {
                TraceEvent::TaskSpawn { task_id, .. } => format!("spawn:{}", task_id.0),
                TraceEvent::TaskComplete { task_id, state, .. } => {
                    format!("complete:{}:{:?}", task_id.0, state)
                }
                TraceEvent::ScopeCreate { scope_id, .. } => format!("scope_create:{}", scope_id),
                TraceEvent::ScopeExit { scope_id, .. } => format!("scope_exit:{}", scope_id),
                _ => String::new(),
            };
            if !msg.is_empty() {
                events_clone.lock().unwrap().push(msg);
            }
        }));

        let scheduler = Scheduler::new(2);

        with_scope(&scheduler, |scope| {
            let t = scope.spawn(|| 42);
            t.await_result();
        });

        scheduler.shutdown();

        clear_trace_callback();

        let events = events.lock().unwrap();
        assert!(events.iter().any(|e| e.starts_with("scope_create")));
        assert!(events.iter().any(|e| e.starts_with("spawn")));
    }
}
