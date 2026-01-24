# bhc-rts-scheduler

Work-Stealing Task Scheduler for the Basel Haskell Compiler.

## Overview

This crate implements structured concurrency primitives as specified in H26-SPEC Section 10. It provides a work-stealing scheduler with cooperative cancellation, deadline support, and efficient task management for the Server Profile.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Scheduler                                │
├─────────────────────────────────────────────────────────────┤
│  Global Queue: [Task] [Task] [Task] ...                     │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │Worker 0 │  │Worker 1 │  │Worker 2 │  │Worker 3 │        │
│  │         │  │         │  │         │  │         │        │
│  │ Local:  │  │ Local:  │  │ Local:  │  │ Local:  │        │
│  │ [T][T]  │  │ [T]     │  │ [T][T]  │  │ []      │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│       │      steal │            │ steal      │              │
│       │◀───────────┴────────────┴───────────▶│              │
│       │                                      │              │
│       ▼                                      ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   OS Threads                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Task Lifecycle

```
     spawn
       │
       ▼
    ┌─────┐
    │ New │
    └──┬──┘
       │ schedule
       ▼
   ┌────────┐     yield/await
   │Running │◀──────────────────┐
   └───┬────┘                   │
       │                        │
       ├────────────────────────┘
       │
       │ complete              │ cancel
       ▼                       ▼
 ┌───────────┐          ┌───────────┐
 │Completing │          │Cancelling │
 └─────┬─────┘          └─────┬─────┘
       │                      │
       └──────────┬───────────┘
                  ▼
            ┌───────────┐
            │ Completed │
            └───────────┘
```

## Task API

```rust
/// Task handle
pub struct Task<T> {
    /// Task identifier
    id: TaskId,

    /// Result channel
    result: oneshot::Receiver<T>,

    /// Cancellation token
    cancel_token: CancelToken,
}

impl<T> Task<T> {
    /// Wait for task completion
    pub async fn await_result(self) -> T;

    /// Non-blocking poll
    pub fn poll(&self) -> Option<T>;

    /// Cancel the task
    pub fn cancel(&self);

    /// Check if completed
    pub fn is_completed(&self) -> bool;
}
```

## Scope API

```rust
/// Structured concurrency scope
pub struct Scope<'a> {
    /// Parent scope (if any)
    parent: Option<&'a Scope<'a>>,

    /// Active tasks
    tasks: Mutex<Vec<TaskHandle>>,

    /// Cancellation token
    cancel_token: CancelToken,

    /// Deadline (if set)
    deadline: Option<Instant>,
}

impl<'a> Scope<'a> {
    /// Spawn a task in this scope
    pub fn spawn<T, F>(&self, f: F) -> Task<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Spawn with explicit cancellation check
    pub fn spawn_cancellable<T, F>(&self, f: F) -> Task<T>
    where
        F: FnOnce(CancelToken) -> T + Send + 'static;
}

/// Run action within a scope
pub fn with_scope<T, F>(f: F) -> T
where
    F: FnOnce(&Scope) -> T,
{
    let scope = Scope::new();
    let result = f(&scope);
    scope.join_all(); // Wait for all tasks
    result
}
```

## Cancellation

### Cooperative Cancellation

```rust
/// Check for cancellation
pub fn check_cancelled() -> Result<(), Cancelled> {
    if current_task().is_cancelled() {
        Err(Cancelled)
    } else {
        Ok(())
    }
}

/// Cancellation token
pub struct CancelToken {
    cancelled: AtomicBool,
    waiters: Mutex<Vec<Waker>>,
}

impl CancelToken {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        // Wake all waiters
        for waker in self.waiters.lock().drain(..) {
            waker.wake();
        }
    }
}
```

### Cancellation Propagation

```rust
// Parent cancellation propagates to children
impl Scope<'_> {
    fn propagate_cancel(&self) {
        self.cancel_token.cancel();

        // Cancel all child tasks
        for task in self.tasks.lock().iter() {
            task.cancel();
        }

        // Propagate to child scopes
        for child in self.children.lock().iter() {
            child.propagate_cancel();
        }
    }
}
```

## Deadlines

```rust
/// Run with deadline
pub fn with_deadline<T, F>(duration: Duration, f: F) -> Option<T>
where
    F: FnOnce(&Scope) -> T,
{
    let deadline = Instant::now() + duration;
    let scope = Scope::with_deadline(deadline);

    let result = f(&scope);

    if Instant::now() < deadline {
        Some(result)
    } else {
        scope.cancel_all();
        None
    }
}
```

## Work Stealing

```rust
impl Worker {
    fn run(&self) {
        loop {
            // 1. Try local queue (LIFO for cache locality)
            if let Some(task) = self.local_queue.pop() {
                self.execute(task);
                continue;
            }

            // 2. Try global queue
            if let Some(task) = self.scheduler.global_queue.steal() {
                self.execute(task);
                continue;
            }

            // 3. Try stealing from other workers
            for other in self.scheduler.workers.iter() {
                if other.id != self.id {
                    if let Some(task) = other.local_queue.steal() {
                        self.execute(task);
                        continue 'outer;
                    }
                }
            }

            // 4. Park until work available
            self.park();
        }
    }
}
```

## Scheduler Configuration

```rust
pub struct SchedulerConfig {
    /// Number of worker threads
    pub worker_count: usize,

    /// Size of each worker's local queue
    pub local_queue_size: usize,

    /// Global queue size
    pub global_queue_size: usize,

    /// Enable work stealing
    pub enable_stealing: bool,

    /// Spin count before parking
    pub spin_count: u32,

    /// Stack size per task
    pub task_stack_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus::get(),
            local_queue_size: 256,
            global_queue_size: 4096,
            enable_stealing: true,
            spin_count: 100,
            task_stack_size: 2 * 1024 * 1024, // 2 MB
        }
    }
}
```

## Task States

```rust
pub enum TaskState {
    /// Created but not scheduled
    New,

    /// Waiting in queue
    Queued,

    /// Currently executing
    Running,

    /// Waiting for I/O or other task
    Blocked,

    /// Cancellation requested
    Cancelling,

    /// Finished, result pending pickup
    Completing,

    /// Done
    Completed,
}
```

## Statistics

```rust
pub struct SchedulerStats {
    /// Tasks spawned
    pub tasks_spawned: u64,

    /// Tasks completed
    pub tasks_completed: u64,

    /// Tasks cancelled
    pub tasks_cancelled: u64,

    /// Successful steals
    pub steals: u64,

    /// Failed steal attempts
    pub steal_failures: u64,

    /// Time workers spent parked
    pub park_time: Duration,

    /// Average task latency
    pub avg_task_latency: Duration,
}
```

## M5 Exit Criteria

Per specification:
- Server workload runs concurrently without numeric kernel regressions
- Cancellation propagates within 1ms of request
- GC pause times < 10ms at p99

## See Also

- `bhc-rts` - Core runtime
- `bhc-concurrent` - High-level concurrency API
- `bhc-rts-gc` - GC integration
