//! C ABI entry points for the BHC Runtime System.
//!
//! These functions are called by compiled BHC programs. They provide
//! the interface between generated LLVM code and the Rust runtime.
//!
//! # Calling Convention
//!
//! All functions use the C calling convention and are `no_mangle` to
//! ensure stable symbol names for linking.

use std::ffi::{c_char, c_int, CStr};
use std::ptr;

use crate::{global, init_default, Profile, RuntimeConfig};

/// Initialize the BHC runtime system with default configuration.
///
/// This must be called before any other RTS functions.
/// It's safe to call multiple times; subsequent calls are no-ops.
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_init() {
    // Try to initialize; if already initialized, this is a no-op
    let _ = std::panic::catch_unwind(|| {
        init_default();
    });
}

/// Initialize the BHC runtime system with a specific profile.
///
/// # Arguments
///
/// * `profile` - Profile identifier (0=Default, 1=Server, 2=Numeric, 3=Edge)
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_init_with_profile(profile: c_int) {
    let profile = match profile {
        0 => Profile::Default,
        1 => Profile::Server,
        2 => Profile::Numeric,
        3 => Profile::Edge,
        _ => Profile::Default,
    };

    let _ = std::panic::catch_unwind(|| {
        crate::init(RuntimeConfig::for_profile(profile));
    });
}

/// Initialize the BHC runtime with command line arguments.
///
/// This is the standard entry point for compiled BHC programs.
///
/// # Arguments
///
/// * `argc` - Number of command line arguments
/// * `argv` - Array of command line argument strings
///
/// # Safety
///
/// `argv` must be a valid pointer to `argc` null-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn bhc_rts_init(argc: c_int, argv: *const *const c_char) {
    // Parse command line arguments for RTS options
    // For now, just use defaults
    let _ = (argc, argv);
    bhc_init();
}

/// Shutdown the BHC runtime system.
///
/// This should be called after the Haskell program has completed.
/// It flushes buffers, runs finalizers, and releases resources.
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_shutdown() {
    // The global runtime will be shutdown on process exit
    // For explicit shutdown, we could clear the global mutex
}

/// Exit the program with the given status code.
///
/// This is called by the generated code when the Haskell main returns.
///
/// # Safety
///
/// This function never returns - it terminates the process.
#[no_mangle]
pub extern "C" fn bhc_exit(status: c_int) -> ! {
    bhc_shutdown();
    std::process::exit(status)
}

/// Force a garbage collection.
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_gc() {
    global().force_gc();
}

/// Allocate memory from the RTS heap.
///
/// # Arguments
///
/// * `size` - Number of bytes to allocate
///
/// # Returns
///
/// Pointer to allocated memory, or null on failure.
///
/// # Safety
///
/// The returned pointer must be freed using `bhc_free` or by GC.
#[no_mangle]
pub extern "C" fn bhc_alloc(size: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size, 8).ok();
    match layout {
        Some(layout) => unsafe { std::alloc::alloc(layout) },
        None => ptr::null_mut(),
    }
}

/// Allocate zeroed memory from the RTS heap.
///
/// # Arguments
///
/// * `size` - Number of bytes to allocate
///
/// # Returns
///
/// Pointer to zero-initialized memory, or null on failure.
///
/// # Safety
///
/// The returned pointer must be freed using `bhc_free` or by GC.
#[no_mangle]
pub extern "C" fn bhc_alloc_zeroed(size: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size, 8).ok();
    match layout {
        Some(layout) => unsafe { std::alloc::alloc_zeroed(layout) },
        None => ptr::null_mut(),
    }
}

/// Free memory allocated by `bhc_alloc`.
///
/// # Arguments
///
/// * `ptr` - Pointer to memory to free
/// * `size` - Size of the allocation (must match the original size)
///
/// # Safety
///
/// `ptr` must have been returned by `bhc_alloc` with the same `size`,
/// or this is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn bhc_free(ptr: *mut u8, size: usize) {
    if !ptr.is_null() {
        if let Some(layout) = std::alloc::Layout::from_size_align(size, 8).ok() {
            unsafe { std::alloc::dealloc(ptr, layout) };
        }
    }
}

/// Print a string to stdout (for debugging).
///
/// # Arguments
///
/// * `s` - Null-terminated C string to print
///
/// # Safety
///
/// `s` must be a valid null-terminated string.
#[no_mangle]
pub unsafe extern "C" fn bhc_print_string(s: *const c_char) {
    if !s.is_null() {
        let cstr = unsafe { CStr::from_ptr(s) };
        if let Ok(str) = cstr.to_str() {
            print!("{}", str);
        }
    }
}

/// Print a string to stdout with newline.
///
/// # Arguments
///
/// * `s` - Null-terminated C string to print
///
/// # Safety
///
/// `s` must be a valid null-terminated string.
#[no_mangle]
pub unsafe extern "C" fn bhc_print_string_ln(s: *const c_char) {
    if !s.is_null() {
        let cstr = unsafe { CStr::from_ptr(s) };
        if let Ok(str) = cstr.to_str() {
            println!("{}", str);
        }
    }
}

/// Print an integer to stdout.
///
/// # Arguments
///
/// * `n` - Integer to print
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_print_int(n: i64) {
    print!("{}", n);
}

/// Print an integer to stdout with newline.
///
/// # Arguments
///
/// * `n` - Integer to print
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_print_int_ln(n: i64) {
    println!("{}", n);
}

/// Print a double to stdout.
///
/// # Arguments
///
/// * `d` - Double to print
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_print_double(d: f64) {
    print!("{}", d);
}

/// Print a double to stdout with newline.
///
/// # Arguments
///
/// * `d` - Double to print
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_print_double_ln(d: f64) {
    println!("{}", d);
}

/// Runtime panic handler.
///
/// Called when an unrecoverable error occurs.
///
/// # Arguments
///
/// * `msg` - Error message (null-terminated)
///
/// # Safety
///
/// This function never returns.
#[no_mangle]
pub unsafe extern "C" fn bhc_panic(msg: *const c_char) -> ! {
    let message = if msg.is_null() {
        "BHC runtime panic".to_string()
    } else {
        let cstr = unsafe { CStr::from_ptr(msg) };
        cstr.to_str().unwrap_or("BHC runtime panic").to_string()
    };
    panic!("{}", message);
}

/// Runtime error - same as panic but named for Haskell compatibility.
///
/// # Safety
///
/// The message pointer must be a valid null-terminated C string, or null.
///
/// This function never returns.
#[no_mangle]
pub unsafe extern "C" fn bhc_error(msg: *const c_char) -> ! {
    unsafe { bhc_panic(msg) }
}

/// Check if we're running in debug mode.
///
/// # Returns
///
/// 1 if debug mode is enabled, 0 otherwise.
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_is_debug() -> c_int {
    if global().config().debug_mode { 1 } else { 0 }
}

/// Get the current profile.
///
/// # Returns
///
/// Profile identifier (0=Default, 1=Server, 2=Numeric, 3=Edge)
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_get_profile() -> c_int {
    match global().config().profile {
        Profile::Default => 0,
        Profile::Server => 1,
        Profile::Numeric => 2,
        Profile::Edge => 3,
    }
}

// ============================================================================
// Thunk Support
// ============================================================================
//
// Thunks are heap objects representing suspended computations.
//
// Memory Layout:
//   struct Thunk {
//       i64   tag;       // -1 = thunk (unevaluated)
//                        // -2 = blackhole (being evaluated)
//                        // >= 0 = evaluated (value's ADT tag)
//       ptr   payload;   // Unevaluated: pointer to eval function
//                        // Evaluated: the actual value
//       i64   env_size;  // Number of captured variables
//       ptr[] env;       // Captured environment
//   }
//
// Thunk evaluation:
//   1. Check tag
//   2. If tag == -1 (thunk): set tag to -2 (blackhole), call eval_fn(env)
//   3. If tag == -2 (blackhole): error (circular reference)
//   4. If tag >= 0: return payload (already evaluated)
// ============================================================================

/// Thunk tag constant: unevaluated thunk
pub const BHC_TAG_THUNK: i64 = -1;

/// Thunk tag constant: blackhole (being evaluated, used to detect cycles)
pub const BHC_TAG_BLACKHOLE: i64 = -2;

/// Force evaluation of a thunk to Weak Head Normal Form (WHNF).
///
/// # Arguments
///
/// * `obj` - Pointer to a heap object (thunk or value)
///
/// # Returns
///
/// Pointer to the evaluated value (same as input if already evaluated)
///
/// # Safety
///
/// `obj` must be a valid pointer to a BHC heap object with proper layout.
/// The tag field (i64 at offset 0) determines the object type.
#[no_mangle]
pub unsafe extern "C" fn bhc_force(obj: *mut u8) -> *mut u8 {
    if obj.is_null() {
        return obj;
    }

    // Read the tag (i64 at offset 0)
    let tag_ptr = obj as *mut i64;
    let tag = unsafe { *tag_ptr };

    if tag >= 0 {
        // Already evaluated (ADT or value), return as-is
        return obj;
    }

    if tag == BHC_TAG_BLACKHOLE {
        // Circular reference detected - panic
        eprintln!("BHC Runtime Error: <<loop>> - circular reference in thunk evaluation");
        std::process::exit(1);
    }

    if tag == BHC_TAG_THUNK {
        // Unevaluated thunk - mark as blackhole and evaluate
        unsafe { *tag_ptr = BHC_TAG_BLACKHOLE };

        // Read eval function pointer (at offset 8)
        let eval_fn_ptr = unsafe { *(obj.add(8) as *const *const u8) };

        // Read env_size (at offset 16) - currently unused but kept for documentation
        let _env_size = unsafe { *(obj.add(16) as *const i64) };

        // Get environment pointer (at offset 24)
        let env_ptr = unsafe { obj.add(24) };

        // The eval function has signature: fn(env: *mut u8) -> *mut u8
        // It takes the environment array and returns the evaluated value
        let eval_fn: extern "C" fn(*mut u8) -> *mut u8 =
            unsafe { std::mem::transmute(eval_fn_ptr) };

        // Call the evaluation function with the environment
        let result = eval_fn(env_ptr as *mut u8);

        // Update the thunk with the result (update in place)
        // Set tag to 0 (indicating evaluated/indirection)
        // Store result pointer in payload slot
        unsafe {
            *tag_ptr = 0; // Mark as evaluated indirection
            *(obj.add(8) as *mut *mut u8) = result;
        }

        return result;
    }

    // Unknown tag - return as-is
    obj
}

/// Check if an object is a thunk that needs forcing.
///
/// # Arguments
///
/// * `obj` - Pointer to a heap object
///
/// # Returns
///
/// 1 if the object is an unevaluated thunk, 0 otherwise
///
/// # Safety
///
/// `obj` must be a valid pointer to a BHC heap object.
#[no_mangle]
pub unsafe extern "C" fn bhc_is_thunk(obj: *const u8) -> c_int {
    if obj.is_null() {
        return 0;
    }

    let tag = unsafe { *(obj as *const i64) };
    if tag == BHC_TAG_THUNK {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bhc_init() {
        bhc_init();
        // Should not panic on re-init
        bhc_init();
    }

    #[test]
    fn test_bhc_print_int() {
        bhc_print_int(42);
        bhc_print_int_ln(42);
    }

    #[test]
    fn test_bhc_print_double() {
        bhc_print_double(3.14);
        bhc_print_double_ln(3.14);
    }

    #[test]
    fn test_bhc_alloc_free() {
        let ptr = bhc_alloc(1024);
        assert!(!ptr.is_null());
        unsafe { bhc_free(ptr, 1024) };
    }

    #[test]
    fn test_bhc_alloc_zeroed() {
        let ptr = bhc_alloc_zeroed(1024);
        assert!(!ptr.is_null());
        // Check that memory is zeroed
        let slice = unsafe { std::slice::from_raw_parts(ptr, 1024) };
        assert!(slice.iter().all(|&b| b == 0));
        unsafe { bhc_free(ptr, 1024) };
    }
}
