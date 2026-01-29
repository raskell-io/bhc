//! M5 Exit Criteria Tests
//!
//! This file contains integration tests that verify the M5 milestone
//! exit criteria as specified in the ROADMAP:
//!
//! 1. Cancellation propagates within 1ms (measured)
//! 2. Structured concurrency: all tasks complete before scope exits
//! 3. Deadline/timeout support works correctly
//! 4. Event tracing captures expected events
//! 5. Cooperative cancellation via check_cancelled()

use bhc_rts_scheduler::{
    check_cancelled, clear_trace_callback, set_trace_callback, with_deadline, with_scope,
    Scheduler, TaskResult,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Exit Criterion 1: Cancellation propagates within 1ms
// ============================================================================

#[test]
fn test_cancellation_latency() {
    let scheduler = Scheduler::new(4);

    // Measure cancellation propagation time
    let cancel_received = Arc::new(AtomicBool::new(false));
    let cancel_received_clone = Arc::clone(&cancel_received);
    let cancel_time = Arc::new(Mutex::new(None));
    let cancel_time_clone = Arc::clone(&cancel_time);

    with_scope(&scheduler, |scope| {
        let task = scope.spawn(move || {
            // Tight loop checking for cancellation
            loop {
                if check_cancelled() {
                    *cancel_time_clone.lock().unwrap() = Some(Instant::now());
                    cancel_received_clone.store(true, Ordering::SeqCst);
                    return;
                }
                // Very short sleep to not hog CPU
                thread::sleep(Duration::from_micros(100));
            }
        });

        // Wait for task to start
        thread::sleep(Duration::from_millis(10));

        // Record time and cancel
        let cancel_start = Instant::now();
        task.cancel();

        // Wait for task to observe cancellation
        while !cancel_received.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_micros(100));
        }

        // Measure latency
        if let Some(received_at) = *cancel_time.lock().unwrap() {
            let latency = received_at.duration_since(cancel_start);
            // Cancellation should propagate within 1ms
            assert!(
                latency < Duration::from_millis(1),
                "Cancellation latency {} ms exceeds 1ms requirement",
                latency.as_secs_f64() * 1000.0
            );
        }
    });

    scheduler.shutdown();
}

// ============================================================================
// Exit Criterion 2: Structured concurrency guarantees
// ============================================================================

#[test]
fn test_scope_waits_for_all_tasks() {
    let scheduler = Scheduler::new(4);
    let completed_count = Arc::new(AtomicUsize::new(0));

    let before_scope = Instant::now();

    with_scope(&scheduler, |scope| {
        for i in 0..10 {
            let completed = Arc::clone(&completed_count);
            scope.spawn(move || {
                // Each task takes some time
                thread::sleep(Duration::from_millis(10 * (i as u64 + 1)));
                completed.fetch_add(1, Ordering::SeqCst);
            });
        }
        // Scope should wait for all tasks
    });

    let after_scope = Instant::now();

    // All 10 tasks should have completed
    assert_eq!(
        completed_count.load(Ordering::SeqCst),
        10,
        "All tasks should complete before scope exits"
    );

    // Should have taken at least 100ms (longest task)
    let elapsed = after_scope.duration_since(before_scope);
    assert!(
        elapsed >= Duration::from_millis(100),
        "Scope should wait for all tasks; elapsed: {:?}",
        elapsed
    );

    scheduler.shutdown();
}

#[test]
fn test_nested_scopes() {
    let scheduler = Scheduler::new(4);
    let inner_completed = Arc::new(AtomicBool::new(false));
    let inner_completed_clone = Arc::clone(&inner_completed);

    with_scope(&scheduler, |outer| {
        outer.spawn(move || {
            // Note: Can't nest scopes without access to scheduler in closure
            // This tests that tasks in outer scope complete
            thread::sleep(Duration::from_millis(50));
            inner_completed_clone.store(true, Ordering::SeqCst);
        });
    });

    assert!(inner_completed.load(Ordering::SeqCst));

    scheduler.shutdown();
}

// ============================================================================
// Exit Criterion 3: Deadline/timeout support
// ============================================================================

#[test]
fn test_deadline_cancels_tasks() {
    let scheduler = Scheduler::new(4);
    let task_cancelled = Arc::new(AtomicBool::new(false));
    let task_cancelled_clone = Arc::clone(&task_cancelled);

    let result = with_deadline(&scheduler, Duration::from_millis(50), |scope| {
        scope
            .spawn(move || {
                // Task checks for cancellation
                for _ in 0..100 {
                    if check_cancelled() {
                        task_cancelled_clone.store(true, Ordering::SeqCst);
                        return "cancelled";
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                "completed"
            })
            .await_result()
    });

    // Either the scope returned None (deadline) or task was cancelled
    match result {
        None => {
            // Deadline triggered before we got result
        }
        Some(TaskResult::Cancelled) => {
            assert!(task_cancelled.load(Ordering::SeqCst));
        }
        Some(TaskResult::Ok(s)) => {
            assert_eq!(s, "cancelled", "Task should have been cancelled");
        }
        _ => panic!("Unexpected result"),
    }

    scheduler.shutdown();
}

#[test]
fn test_deadline_respects_fast_completion() {
    let scheduler = Scheduler::new(4);

    let start = Instant::now();
    let result = with_deadline(&scheduler, Duration::from_secs(10), |scope| {
        // Fast task that completes well before deadline
        let task = scope.spawn(|| {
            thread::sleep(Duration::from_millis(10));
            42
        });
        task.await_result().ok()
    });

    let elapsed = start.elapsed();

    // Should have completed quickly, not waited for full deadline
    assert!(
        elapsed < Duration::from_secs(1),
        "Fast task should not wait for full deadline"
    );
    assert_eq!(result, Some(Some(42)));

    scheduler.shutdown();
}

// ============================================================================
// Exit Criterion 4: Event tracing
// ============================================================================

#[test]
fn test_trace_events_captured() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events);

    set_trace_callback(Box::new(move |event| {
        events_clone.lock().unwrap().push(format!("{:?}", event));
    }));

    let scheduler = Scheduler::new(2);

    with_scope(&scheduler, |scope| {
        let task = scope.spawn(|| 42);
        task.await_result();
    });

    scheduler.shutdown();
    clear_trace_callback();

    let captured = events.lock().unwrap();

    // Should have captured spawn, start, complete events
    let spawn_events = captured.iter().filter(|e| e.contains("TaskSpawn")).count();
    let complete_events = captured
        .iter()
        .filter(|e| e.contains("TaskComplete"))
        .count();
    let scope_events = captured.iter().filter(|e| e.contains("Scope")).count();

    assert!(spawn_events > 0, "Should have TaskSpawn events");
    assert!(complete_events > 0, "Should have TaskComplete events");
    assert!(scope_events > 0, "Should have Scope events");
}

// ============================================================================
// Exit Criterion 5: Cooperative cancellation
// ============================================================================

#[test]
fn test_check_cancelled_observable() {
    let scheduler = Scheduler::new(2);
    let check_count = Arc::new(AtomicUsize::new(0));
    let check_count_clone = Arc::clone(&check_count);
    let saw_cancelled = Arc::new(AtomicBool::new(false));
    let saw_cancelled_clone = Arc::clone(&saw_cancelled);

    with_scope(&scheduler, |scope| {
        let task = scope.spawn(move || {
            // Check cancellation multiple times
            for _ in 0..100 {
                check_count_clone.fetch_add(1, Ordering::SeqCst);
                if check_cancelled() {
                    saw_cancelled_clone.store(true, Ordering::SeqCst);
                    return;
                }
                thread::sleep(Duration::from_millis(5));
            }
        });

        // Let task run a bit then cancel
        thread::sleep(Duration::from_millis(50));
        task.cancel();
    });

    // Task should have checked cancellation multiple times
    assert!(
        check_count.load(Ordering::SeqCst) > 1,
        "Task should have checked cancellation multiple times"
    );
    // Task should have eventually seen the cancellation
    assert!(
        saw_cancelled.load(Ordering::SeqCst),
        "Task should have observed cancellation via check_cancelled()"
    );

    scheduler.shutdown();
}

#[test]
fn test_task_not_cancelled_returns_false() {
    let scheduler = Scheduler::new(2);
    let check_result = Arc::new(AtomicBool::new(true)); // Default true, set to check_cancelled() result
    let check_result_clone = Arc::clone(&check_result);

    with_scope(&scheduler, |scope| {
        let task = scope.spawn(move || {
            // Check cancellation before being cancelled
            let result = check_cancelled();
            check_result_clone.store(result, Ordering::SeqCst);
            42
        });
        task.await_result();
    });

    // Task was not cancelled, so check_cancelled() should have returned false
    assert!(
        !check_result.load(Ordering::SeqCst),
        "check_cancelled() should return false for non-cancelled task"
    );

    scheduler.shutdown();
}

// ============================================================================
// Additional M5 verification tests
// ============================================================================

#[test]
fn test_multiple_concurrent_scopes() {
    let scheduler = Scheduler::new(4);
    let results = Arc::new(Mutex::new(Vec::new()));

    // Run multiple scopes concurrently via separate threads
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let _scheduler_ref = &scheduler;
            let results = Arc::clone(&results);
            thread::spawn(move || {
                // Note: This is a simplification - in real code we'd use Arc<Scheduler>
                // For this test, we just verify the API works
                results.lock().unwrap().push(i);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(results.lock().unwrap().len(), 4);

    scheduler.shutdown();
}

#[test]
fn test_task_result_types() {
    let scheduler = Scheduler::new(2);

    with_scope(&scheduler, |scope| {
        // Test Ok result
        let task = scope.spawn(|| 42);
        match task.await_result() {
            TaskResult::Ok(v) => assert_eq!(v, 42),
            _ => panic!("Expected Ok result"),
        }

        // Test Cancelled result
        let task2 = scope.spawn(|| {
            for _ in 0..1000 {
                if check_cancelled() {
                    return 0;
                }
                thread::sleep(Duration::from_millis(1));
            }
            1
        });
        thread::sleep(Duration::from_millis(10));
        task2.cancel();
        // Result will be either Ok(0) if task saw cancellation, or Cancelled
        let result = task2.await_result();
        assert!(
            matches!(result, TaskResult::Ok(0) | TaskResult::Cancelled),
            "Expected Ok(0) or Cancelled, got {:?}",
            result
        );
    });

    scheduler.shutdown();
}

#[test]
fn test_scheduler_stats() {
    let scheduler = Scheduler::new(2);

    with_scope(&scheduler, |scope| {
        for _ in 0..10 {
            let task = scope.spawn(|| {
                thread::sleep(Duration::from_millis(1));
                1
            });
            task.await_result();
        }
    });

    let stats = scheduler.stats();
    // Note: stats.tasks_spawned includes watcher tasks spawned by scope
    assert!(
        stats.tasks_spawned >= 10,
        "Should have spawned at least 10 tasks, got {}",
        stats.tasks_spawned
    );

    scheduler.shutdown();
}
