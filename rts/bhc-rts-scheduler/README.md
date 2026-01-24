# bhc-rts-scheduler

Work-stealing task scheduler for the BHC Runtime System.

## Overview

This crate implements structured concurrency primitives as specified in H26-SPEC Section 10: Concurrency Model. It provides a work-stealing scheduler with cooperative cancellation and deadline support.

## Task Lifecycle

```
  spawn      await
    |          |
    v          v
+-----+    +-------+    +----------+    +---------+
| New | -> |Running| -> |Completing| -> |Completed|
+-----+    +-------+    +----------+    +---------+
              |                              ^
              | cancel                       |
              v                              |
           +----------+                      |
           |Cancelling| ---------------------+
           +----------+
```

## Key Types

| Type | Description |
|------|-------------|
| `Scheduler` | Work-stealing task scheduler |
| `Scope` | Structured concurrency scope |
| `Task<T>` | Handle to a spawned task |
| `TaskState` | Current state of a task |

## Usage

### Structured Concurrency

```rust
use bhc_rts_scheduler::{Scheduler, with_scope};

let scheduler = Scheduler::new(4); // 4 worker threads

with_scope(&scheduler, |scope| {
    let task1 = scope.spawn(|| compute_x());
    let task2 = scope.spawn(|| compute_y());

    let x = task1.await_result();
    let y = task2.await_result();
    (x, y)
});
// All tasks complete before scope exits
```

### Cancellation

```rust
with_scope(&scheduler, |scope| {
    let task = scope.spawn(|| {
        loop {
            // Check for cancellation
            if cancelled() {
                return Err(Cancelled);
            }
            do_work();
        }
    });

    // Cancel after timeout
    thread::sleep(Duration::from_secs(1));
    task.cancel();
});
```

### Deadlines

```rust
use bhc_rts_scheduler::with_deadline;

let result = with_deadline(Duration::from_secs(5), |scope| {
    scope.spawn(|| long_running_task()).await_result()
});

match result {
    Some(value) => println!("Completed: {:?}", value),
    None => println!("Deadline exceeded"),
}
```

## Scheduler Configuration

```rust
pub struct SchedulerConfig {
    /// Number of worker threads
    pub worker_count: usize,

    /// Size of each worker's local queue
    pub local_queue_size: usize,

    /// Enable task stealing
    pub enable_stealing: bool,

    /// Spin count before parking
    pub spin_count: u32,
}
```

## Work Stealing

Workers maintain local queues and steal from others when idle:

```rust
// Worker execution loop
loop {
    // 1. Try local queue first
    if let Some(task) = local_queue.pop() {
        run_task(task);
        continue;
    }

    // 2. Try stealing from others
    for other in &workers {
        if let Some(task) = other.queue.steal() {
            run_task(task);
            continue 'outer;
        }
    }

    // 3. No work, park the thread
    park();
}
```

## Task States

| State | Description |
|-------|-------------|
| `New` | Task created, not yet scheduled |
| `Running` | Task is executing |
| `Completing` | Task finished, waiting for cleanup |
| `Completed` | Task done, result available |
| `Cancelling` | Cancellation requested |

## Scope Guarantees

- Tasks cannot escape their scope
- Parent cancellation propagates to children
- All tasks complete before scope exits
- Resources are cleaned up on cancellation

## M5 Exit Criteria

- Server workload runs concurrently without numeric kernel regressions
- Cancellation propagates within 1ms of request
- GC pause times < 10ms at p99

## Design Notes

- Uses crossbeam deques for work stealing
- Parking/unparking minimizes CPU usage
- Cancellation is cooperative (tasks check at safe points)
- Deadline checks happen at safe points

## Related Crates

- `bhc-rts` - Core runtime
- `bhc-concurrent` - High-level concurrency (stdlib)
- `bhc-rts-gc` - GC integration

## Specification References

- H26-SPEC Section 10: Concurrency Model
- H26-SPEC Section 10.1: Structured Concurrency
- BHC-RULE-009: Concurrency Guidelines
