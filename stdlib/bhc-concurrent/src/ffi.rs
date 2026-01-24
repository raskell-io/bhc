//! FFI exports for BHC Haskell bindings
//!
//! This module provides C-ABI functions that can be called from BHC-compiled
//! Haskell code via foreign imports.
//!
//! # Memory Management
//!
//! - Handles returned from `*_new` functions must be freed with corresponding `*_free`
//! - Pointers passed to functions must remain valid for the duration of the call
//! - Closures passed to scope functions are executed immediately

use crate::scope::{with_deadline, with_scope, Scope};
use crate::stm::{atomically, retry, TMVar, TQueue, TVar};
use std::ffi::c_void;
use std::ptr;
use std::time::Duration;

// ============================================================================
// Scope FFI
// ============================================================================

/// Opaque handle to a scope for FFI.
pub struct ScopeHandle {
    scope: *const Scope,
}

/// Opaque handle to a task for FFI.
pub struct TaskHandle {
    // We store a boxed closure result
    result: *mut c_void,
    completed: bool,
    cancelled: bool,
}

/// Run a function within a scope.
///
/// The callback receives a scope handle and should spawn tasks on it.
/// All tasks complete before this function returns.
///
/// # Safety
///
/// The callback must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_with_scope(
    callback: unsafe extern "C" fn(*const c_void, *const c_void) -> *mut c_void,
    user_data: *const c_void,
) -> *mut c_void {
    let result = with_scope(|scope| {
        let scope_ptr = scope as *const Scope as *const c_void;
        // SAFETY: callback is provided by BHC runtime
        unsafe { callback(scope_ptr, user_data) }
    });
    result
}

/// Run a function within a scope with a deadline.
///
/// Returns null if the deadline was exceeded.
///
/// # Safety
///
/// The callback must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_with_deadline(
    timeout_ms: u64,
    callback: unsafe extern "C" fn(*const c_void, *const c_void) -> *mut c_void,
    user_data: *const c_void,
) -> *mut c_void {
    let timeout = Duration::from_millis(timeout_ms);

    let result = with_deadline(timeout, |scope| {
        let scope_ptr = scope as *const Scope as *const c_void;
        // SAFETY: callback is provided by BHC runtime
        unsafe { callback(scope_ptr, user_data) }
    });

    match result {
        Some(ptr) => ptr,
        None => ptr::null_mut(),
    }
}

/// Spawn a task within a scope.
///
/// # Safety
///
/// - `scope_ptr` must be a valid pointer to a Scope
/// - `callback` must be a valid function pointer
#[no_mangle]
pub unsafe extern "C" fn bhc_spawn(
    scope_ptr: *const c_void,
    callback: unsafe extern "C" fn(*const c_void) -> *mut c_void,
    user_data: *const c_void,
) -> *mut TaskHandle {
    let scope = &*(scope_ptr as *const Scope);
    let user_data_copy = user_data as usize; // Copy the pointer value

    // Wrap the callback pointer as usize to make it Send-safe
    let callback_ptr = callback as usize;

    let handle = crate::scope::spawn(scope, move || {
        // SAFETY: callback is provided by BHC runtime
        // Convert back from usize to function pointer
        let cb: unsafe extern "C" fn(*const c_void) -> *mut c_void =
            unsafe { std::mem::transmute(callback_ptr) };
        let result = unsafe { cb(user_data_copy as *const c_void) };
        // Return as usize to be Send-safe
        result as usize
    });

    // Wait for result and package it
    let result = handle.join();
    let task_handle = Box::new(TaskHandle {
        result: result.map(|r| r as *mut c_void).unwrap_or(ptr::null_mut()),
        completed: result.is_some(),
        cancelled: result.is_none(),
    });

    Box::into_raw(task_handle)
}

/// Await a task's result.
///
/// # Safety
///
/// `task_ptr` must be a valid pointer to a TaskHandle.
#[no_mangle]
pub unsafe extern "C" fn bhc_await(task_ptr: *mut TaskHandle) -> *mut c_void {
    let task = &*task_ptr;
    task.result
}

/// Cancel a task.
///
/// # Safety
///
/// `task_ptr` must be a valid pointer to a TaskHandle.
#[no_mangle]
pub unsafe extern "C" fn bhc_cancel(_task_ptr: *mut TaskHandle) {
    // Note: In a full implementation, this would signal cancellation
    // to the running task. For now, cancellation is handled at spawn time.
}

/// Poll a task for completion.
///
/// Returns the result if complete, null otherwise.
///
/// # Safety
///
/// `task_ptr` must be a valid pointer to a TaskHandle.
#[no_mangle]
pub unsafe extern "C" fn bhc_poll(task_ptr: *mut TaskHandle) -> *mut c_void {
    let task = &*task_ptr;
    if task.completed {
        task.result
    } else {
        ptr::null_mut()
    }
}

/// Free a task handle.
///
/// # Safety
///
/// `task_ptr` must be a valid pointer to a TaskHandle created by `bhc_spawn`.
#[no_mangle]
pub unsafe extern "C" fn bhc_task_free(task_ptr: *mut TaskHandle) {
    if !task_ptr.is_null() {
        let _ = Box::from_raw(task_ptr);
    }
}

/// Check if the current context is cancelled.
///
/// Returns 1 if cancelled, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_check_cancelled() -> i32 {
    // In a full implementation, this would check thread-local cancellation state
    0
}

// ============================================================================
// STM FFI
// ============================================================================

/// Opaque handle to a TVar<i64>.
pub struct TVarI64Handle {
    tvar: TVar<i64>,
}

/// Create a new TVar with an i64 value.
#[no_mangle]
pub extern "C" fn bhc_tvar_new_i64(value: i64) -> *mut TVarI64Handle {
    let handle = Box::new(TVarI64Handle {
        tvar: TVar::new(value),
    });
    Box::into_raw(handle)
}

/// Read a TVar<i64> within a transaction.
///
/// # Safety
///
/// - `tvar_ptr` must be a valid pointer to a TVarI64Handle
/// - Must be called within an `atomically` block
#[no_mangle]
pub unsafe extern "C" fn bhc_tvar_read_i64(tvar_ptr: *mut TVarI64Handle) -> i64 {
    let handle = &*tvar_ptr;
    handle.tvar.read_tx().unwrap_or(0)
}

/// Write to a TVar<i64> within a transaction.
///
/// # Safety
///
/// - `tvar_ptr` must be a valid pointer to a TVarI64Handle
/// - Must be called within an `atomically` block
#[no_mangle]
pub unsafe extern "C" fn bhc_tvar_write_i64(tvar_ptr: *mut TVarI64Handle, value: i64) {
    let handle = &*tvar_ptr;
    let _ = handle.tvar.write_tx(value);
}

/// Free a TVar<i64> handle.
///
/// # Safety
///
/// `tvar_ptr` must be a valid pointer created by `bhc_tvar_new_i64`.
#[no_mangle]
pub unsafe extern "C" fn bhc_tvar_free_i64(tvar_ptr: *mut TVarI64Handle) {
    if !tvar_ptr.is_null() {
        let _ = Box::from_raw(tvar_ptr);
    }
}

/// Opaque handle to a TVar<f64>.
pub struct TVarF64Handle {
    tvar: TVar<f64>,
}

/// Create a new TVar with an f64 value.
#[no_mangle]
pub extern "C" fn bhc_tvar_new_f64(value: f64) -> *mut TVarF64Handle {
    let handle = Box::new(TVarF64Handle {
        tvar: TVar::new(value),
    });
    Box::into_raw(handle)
}

/// Read a TVar<f64> within a transaction.
///
/// # Safety
///
/// `tvar_ptr` must be a valid pointer to a TVarF64Handle.
#[no_mangle]
pub unsafe extern "C" fn bhc_tvar_read_f64(tvar_ptr: *mut TVarF64Handle) -> f64 {
    let handle = &*tvar_ptr;
    handle.tvar.read_tx().unwrap_or(0.0)
}

/// Write to a TVar<f64> within a transaction.
///
/// # Safety
///
/// `tvar_ptr` must be a valid pointer to a TVarF64Handle.
#[no_mangle]
pub unsafe extern "C" fn bhc_tvar_write_f64(tvar_ptr: *mut TVarF64Handle, value: f64) {
    let handle = &*tvar_ptr;
    let _ = handle.tvar.write_tx(value);
}

/// Free a TVar<f64> handle.
///
/// # Safety
///
/// `tvar_ptr` must be a valid pointer created by `bhc_tvar_new_f64`.
#[no_mangle]
pub unsafe extern "C" fn bhc_tvar_free_f64(tvar_ptr: *mut TVarF64Handle) {
    if !tvar_ptr.is_null() {
        let _ = Box::from_raw(tvar_ptr);
    }
}

/// Execute a transaction atomically.
///
/// The callback is executed repeatedly until it commits successfully.
///
/// # Safety
///
/// `callback` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_atomically(
    callback: unsafe extern "C" fn(*const c_void) -> i32,
    user_data: *const c_void,
) -> i32 {
    atomically(|| {
        let result = unsafe { callback(user_data) };
        if result < 0 {
            // Negative = retry
            retry()
        } else {
            Ok(result)
        }
    })
}

/// Signal that the current transaction should retry.
///
/// Returns a negative value to indicate retry.
#[no_mangle]
pub extern "C" fn bhc_retry() -> i32 {
    -1 // Signal retry
}

/// Check a condition in STM; retry if false.
#[no_mangle]
pub extern "C" fn bhc_check(condition: i32) -> i32 {
    if condition != 0 {
        0 // Success
    } else {
        -1 // Retry
    }
}

// ============================================================================
// TMVar FFI
// ============================================================================

/// Opaque handle to a TMVar<i64>.
pub struct TMVarI64Handle {
    tmvar: TMVar<i64>,
}

/// Create a new TMVar with a value.
#[no_mangle]
pub extern "C" fn bhc_tmvar_new_i64(value: i64) -> *mut TMVarI64Handle {
    let handle = Box::new(TMVarI64Handle {
        tmvar: TMVar::new(value),
    });
    Box::into_raw(handle)
}

/// Create a new empty TMVar.
#[no_mangle]
pub extern "C" fn bhc_tmvar_new_empty_i64() -> *mut TMVarI64Handle {
    let handle = Box::new(TMVarI64Handle {
        tmvar: TMVar::new_empty(),
    });
    Box::into_raw(handle)
}

/// Take from a TMVar (blocks if empty).
///
/// # Safety
///
/// `tmvar_ptr` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_tmvar_take_i64(
    tmvar_ptr: *mut TMVarI64Handle,
    out_value: *mut i64,
) -> i32 {
    let handle = &*tmvar_ptr;
    match handle.tmvar.take() {
        Ok(v) => {
            *out_value = v;
            0
        }
        Err(_) => -1, // Retry
    }
}

/// Put a value into a TMVar (blocks if full).
///
/// # Safety
///
/// `tmvar_ptr` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_tmvar_put_i64(tmvar_ptr: *mut TMVarI64Handle, value: i64) -> i32 {
    let handle = &*tmvar_ptr;
    match handle.tmvar.put(value) {
        Ok(()) => 0,
        Err(_) => -1, // Retry
    }
}

/// Free a TMVar handle.
///
/// # Safety
///
/// `tmvar_ptr` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_tmvar_free_i64(tmvar_ptr: *mut TMVarI64Handle) {
    if !tmvar_ptr.is_null() {
        let _ = Box::from_raw(tmvar_ptr);
    }
}

// ============================================================================
// TQueue FFI
// ============================================================================

/// Opaque handle to a TQueue<i64>.
pub struct TQueueI64Handle {
    queue: TQueue<i64>,
}

/// Create a new empty TQueue.
#[no_mangle]
pub extern "C" fn bhc_tqueue_new_i64() -> *mut TQueueI64Handle {
    let handle = Box::new(TQueueI64Handle {
        queue: TQueue::new(),
    });
    Box::into_raw(handle)
}

/// Write a value to the queue.
///
/// # Safety
///
/// `queue_ptr` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_tqueue_write_i64(queue_ptr: *mut TQueueI64Handle, value: i64) -> i32 {
    let handle = &*queue_ptr;
    match handle.queue.write(value) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Read a value from the queue (blocks if empty).
///
/// # Safety
///
/// `queue_ptr` and `out_value` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn bhc_tqueue_read_i64(
    queue_ptr: *mut TQueueI64Handle,
    out_value: *mut i64,
) -> i32 {
    let handle = &*queue_ptr;
    match handle.queue.read() {
        Ok(v) => {
            *out_value = v;
            0
        }
        Err(_) => -1, // Retry
    }
}

/// Try to read without blocking.
///
/// # Safety
///
/// `queue_ptr` and `out_value` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn bhc_tqueue_try_read_i64(
    queue_ptr: *mut TQueueI64Handle,
    out_value: *mut i64,
) -> i32 {
    let handle = &*queue_ptr;
    match handle.queue.try_read() {
        Ok(Some(v)) => {
            *out_value = v;
            1 // Got value
        }
        Ok(None) => 0,  // Empty
        Err(_) => -1,   // Retry
    }
}

/// Check if queue is empty.
///
/// # Safety
///
/// `queue_ptr` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_tqueue_is_empty_i64(queue_ptr: *mut TQueueI64Handle) -> i32 {
    let handle = &*queue_ptr;
    match handle.queue.is_empty() {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(_) => -1,
    }
}

/// Free a TQueue handle.
///
/// # Safety
///
/// `queue_ptr` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bhc_tqueue_free_i64(queue_ptr: *mut TQueueI64Handle) {
    if !queue_ptr.is_null() {
        let _ = Box::from_raw(queue_ptr);
    }
}

// ============================================================================
// Scheduler FFI (re-export from bhc-rts-scheduler)
// ============================================================================

// Note: The scheduler FFI is provided by bhc-rts-scheduler.
// These are convenience wrappers for common operations.

/// Get the number of available CPU cores.
#[no_mangle]
pub extern "C" fn bhc_num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tvar_ffi() {
        let handle = bhc_tvar_new_i64(42);
        assert!(!handle.is_null());

        atomically(|| {
            let val = unsafe { bhc_tvar_read_i64(handle) };
            assert_eq!(val, 42);
            unsafe { bhc_tvar_write_i64(handle, 100) };
            let val = unsafe { bhc_tvar_read_i64(handle) };
            assert_eq!(val, 100);
            Ok(())
        });

        unsafe { bhc_tvar_free_i64(handle) };
    }

    #[test]
    fn test_tqueue_ffi() {
        let queue = bhc_tqueue_new_i64();
        assert!(!queue.is_null());

        atomically(|| {
            unsafe { bhc_tqueue_write_i64(queue, 1) };
            unsafe { bhc_tqueue_write_i64(queue, 2) };
            Ok(())
        });

        atomically(|| {
            let mut val: i64 = 0;
            let result = unsafe { bhc_tqueue_read_i64(queue, &mut val) };
            assert_eq!(result, 0);
            assert_eq!(val, 1);
            Ok(())
        });

        unsafe { bhc_tqueue_free_i64(queue) };
    }

    #[test]
    fn test_num_cpus() {
        let cpus = bhc_num_cpus();
        assert!(cpus >= 1);
    }
}
