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

/// Print a single character (without newline).
///
/// # Arguments
///
/// * `c` - Character code to print (as i32)
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_print_char(c: i32) {
    if let Some(ch) = char::from_u32(c as u32) {
        print!("{}", ch);
    }
}

/// Print a boolean value (without newline).
///
/// # Arguments
///
/// * `b` - Boolean value (0 = False, non-zero = True)
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_print_bool(b: i64) {
    if b == 0 {
        print!("False");
    } else {
        print!("True");
    }
}

/// Print a boolean value (with newline).
///
/// # Arguments
///
/// * `b` - Boolean value (0 = False, non-zero = True)
///
/// # Safety
///
/// This function is safe to call from C code.
#[no_mangle]
pub extern "C" fn bhc_print_bool_ln(b: i64) {
    if b == 0 {
        println!("False");
    } else {
        println!("True");
    }
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
    if global().config().debug_mode {
        1
    } else {
        0
    }
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
        Profile::Realtime => 4,
        Profile::Embedded => 5,
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

// ============================================================================
// Primitive Operations for Haskell Type Classes
// ============================================================================
//
// These functions implement the primitive operations used by Haskell
// type class instances (Eq, Ord, Num, etc.) for basic types.
//
// Conventions:
// - Bool: i32 where 0 = False, 1 = True
// - Ordering: i32 where -1 = LT, 0 = EQ, 1 = GT
// - Int: i64 (64-bit signed integer)
// - Integer: i64 (simplified, should be arbitrary precision)
// - Float: f32
// - Double: f64
// - Char: u32 (Unicode code point)
// ============================================================================

// ----------------------------------------------------------------------------
// Int Primitives
// ----------------------------------------------------------------------------

/// Int equality
#[no_mangle]
pub extern "C" fn bhc_eq_int(a: i64, b: i64) -> i32 {
    if a == b {
        1
    } else {
        0
    }
}

/// Int comparison (returns Ordering: -1=LT, 0=EQ, 1=GT)
#[no_mangle]
pub extern "C" fn bhc_compare_int(a: i64, b: i64) -> i32 {
    match a.cmp(&b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Int less than
#[no_mangle]
pub extern "C" fn bhc_lt_int(a: i64, b: i64) -> i32 {
    if a < b {
        1
    } else {
        0
    }
}

/// Int less than or equal
#[no_mangle]
pub extern "C" fn bhc_le_int(a: i64, b: i64) -> i32 {
    if a <= b {
        1
    } else {
        0
    }
}

/// Int greater than
#[no_mangle]
pub extern "C" fn bhc_gt_int(a: i64, b: i64) -> i32 {
    if a > b {
        1
    } else {
        0
    }
}

/// Int greater than or equal
#[no_mangle]
pub extern "C" fn bhc_ge_int(a: i64, b: i64) -> i32 {
    if a >= b {
        1
    } else {
        0
    }
}

/// Int addition
#[no_mangle]
pub extern "C" fn bhc_add_int(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

/// Int subtraction
#[no_mangle]
pub extern "C" fn bhc_sub_int(a: i64, b: i64) -> i64 {
    a.wrapping_sub(b)
}

/// Int multiplication
#[no_mangle]
pub extern "C" fn bhc_mul_int(a: i64, b: i64) -> i64 {
    a.wrapping_mul(b)
}

/// Int negation
#[no_mangle]
pub extern "C" fn bhc_negate_int(a: i64) -> i64 {
    a.wrapping_neg()
}

/// Int quotient (truncated toward zero)
#[no_mangle]
pub extern "C" fn bhc_quot_int(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("divide by zero");
    }
    a / b
}

/// Int remainder (truncated toward zero)
#[no_mangle]
pub extern "C" fn bhc_rem_int(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("divide by zero");
    }
    a % b
}

/// Int division (truncated toward negative infinity)
#[no_mangle]
pub extern "C" fn bhc_div_int(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("divide by zero");
    }
    a.div_euclid(b)
}

/// Int modulus (Euclidean)
#[no_mangle]
pub extern "C" fn bhc_mod_int(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("divide by zero");
    }
    a.rem_euclid(b)
}

/// Minimum Int value
#[no_mangle]
pub extern "C" fn bhc_min_int() -> i64 {
    i64::MIN
}

/// Maximum Int value
#[no_mangle]
pub extern "C" fn bhc_max_int() -> i64 {
    i64::MAX
}

/// Int to Integer (for now, just returns the same value)
#[no_mangle]
pub extern "C" fn bhc_int_to_integer(a: i64) -> i64 {
    a
}

/// Integer to Int (for now, just returns the same value)
#[no_mangle]
pub extern "C" fn bhc_integer_to_int(a: i64) -> i64 {
    a
}

/// Show Int - returns a heap-allocated string
#[no_mangle]
pub extern "C" fn bhc_show_int(n: i64) -> *mut c_char {
    let s = format!("{}", n);
    let c_string = std::ffi::CString::new(s).unwrap();
    c_string.into_raw()
}

// ----------------------------------------------------------------------------
// Float Primitives
// ----------------------------------------------------------------------------

/// Float equality
#[no_mangle]
pub extern "C" fn bhc_eq_float(a: f32, b: f32) -> i32 {
    if a == b {
        1
    } else {
        0
    }
}

/// Float comparison
#[no_mangle]
pub extern "C" fn bhc_compare_float(a: f32, b: f32) -> i32 {
    match a.partial_cmp(&b) {
        Some(std::cmp::Ordering::Less) => -1,
        Some(std::cmp::Ordering::Equal) => 0,
        Some(std::cmp::Ordering::Greater) => 1,
        None => 0, // NaN comparison
    }
}

/// Float less than
#[no_mangle]
pub extern "C" fn bhc_lt_float(a: f32, b: f32) -> i32 {
    if a < b {
        1
    } else {
        0
    }
}

/// Float less than or equal
#[no_mangle]
pub extern "C" fn bhc_le_float(a: f32, b: f32) -> i32 {
    if a <= b {
        1
    } else {
        0
    }
}

/// Float greater than
#[no_mangle]
pub extern "C" fn bhc_gt_float(a: f32, b: f32) -> i32 {
    if a > b {
        1
    } else {
        0
    }
}

/// Float greater than or equal
#[no_mangle]
pub extern "C" fn bhc_ge_float(a: f32, b: f32) -> i32 {
    if a >= b {
        1
    } else {
        0
    }
}

/// Float addition
#[no_mangle]
pub extern "C" fn bhc_add_float(a: f32, b: f32) -> f32 {
    a + b
}

/// Float subtraction
#[no_mangle]
pub extern "C" fn bhc_sub_float(a: f32, b: f32) -> f32 {
    a - b
}

/// Float multiplication
#[no_mangle]
pub extern "C" fn bhc_mul_float(a: f32, b: f32) -> f32 {
    a * b
}

/// Float division
#[no_mangle]
pub extern "C" fn bhc_div_float(a: f32, b: f32) -> f32 {
    a / b
}

/// Float negation
#[no_mangle]
pub extern "C" fn bhc_negate_float(a: f32) -> f32 {
    -a
}

/// Float absolute value
#[no_mangle]
pub extern "C" fn bhc_abs_float(a: f32) -> f32 {
    a.abs()
}

/// Integer to Float
#[no_mangle]
pub extern "C" fn bhc_integer_to_float(a: i64) -> f32 {
    a as f32
}

// Transcendental functions for Float

#[no_mangle]
pub extern "C" fn bhc_exp_float(a: f32) -> f32 {
    a.exp()
}

#[no_mangle]
pub extern "C" fn bhc_log_float(a: f32) -> f32 {
    a.ln()
}

#[no_mangle]
pub extern "C" fn bhc_sqrt_float(a: f32) -> f32 {
    a.sqrt()
}

#[no_mangle]
pub extern "C" fn bhc_pow_float(a: f32, b: f32) -> f32 {
    a.powf(b)
}

#[no_mangle]
pub extern "C" fn bhc_sin_float(a: f32) -> f32 {
    a.sin()
}

#[no_mangle]
pub extern "C" fn bhc_cos_float(a: f32) -> f32 {
    a.cos()
}

#[no_mangle]
pub extern "C" fn bhc_tan_float(a: f32) -> f32 {
    a.tan()
}

#[no_mangle]
pub extern "C" fn bhc_asin_float(a: f32) -> f32 {
    a.asin()
}

#[no_mangle]
pub extern "C" fn bhc_acos_float(a: f32) -> f32 {
    a.acos()
}

#[no_mangle]
pub extern "C" fn bhc_atan_float(a: f32) -> f32 {
    a.atan()
}

#[no_mangle]
pub extern "C" fn bhc_sinh_float(a: f32) -> f32 {
    a.sinh()
}

#[no_mangle]
pub extern "C" fn bhc_cosh_float(a: f32) -> f32 {
    a.cosh()
}

#[no_mangle]
pub extern "C" fn bhc_tanh_float(a: f32) -> f32 {
    a.tanh()
}

#[no_mangle]
pub extern "C" fn bhc_asinh_float(a: f32) -> f32 {
    a.asinh()
}

#[no_mangle]
pub extern "C" fn bhc_acosh_float(a: f32) -> f32 {
    a.acosh()
}

#[no_mangle]
pub extern "C" fn bhc_atanh_float(a: f32) -> f32 {
    a.atanh()
}

/// Truncate Float to Int
#[no_mangle]
pub extern "C" fn bhc_truncate_float(a: f32) -> i64 {
    a.trunc() as i64
}

/// Round Float to Int
#[no_mangle]
pub extern "C" fn bhc_round_float(a: f32) -> i64 {
    a.round() as i64
}

/// Ceiling of Float
#[no_mangle]
pub extern "C" fn bhc_ceiling_float(a: f32) -> i64 {
    a.ceil() as i64
}

/// Floor of Float
#[no_mangle]
pub extern "C" fn bhc_floor_float(a: f32) -> i64 {
    a.floor() as i64
}

/// Show Float
#[no_mangle]
pub extern "C" fn bhc_show_float(n: f32) -> *mut c_char {
    let s = format!("{}", n);
    let c_string = std::ffi::CString::new(s).unwrap();
    c_string.into_raw()
}

// ----------------------------------------------------------------------------
// Double Primitives
// ----------------------------------------------------------------------------

/// Double equality
#[no_mangle]
pub extern "C" fn bhc_eq_double(a: f64, b: f64) -> i32 {
    if a == b {
        1
    } else {
        0
    }
}

/// Double comparison
#[no_mangle]
pub extern "C" fn bhc_compare_double(a: f64, b: f64) -> i32 {
    match a.partial_cmp(&b) {
        Some(std::cmp::Ordering::Less) => -1,
        Some(std::cmp::Ordering::Equal) => 0,
        Some(std::cmp::Ordering::Greater) => 1,
        None => 0, // NaN comparison
    }
}

/// Double less than
#[no_mangle]
pub extern "C" fn bhc_lt_double(a: f64, b: f64) -> i32 {
    if a < b {
        1
    } else {
        0
    }
}

/// Double less than or equal
#[no_mangle]
pub extern "C" fn bhc_le_double(a: f64, b: f64) -> i32 {
    if a <= b {
        1
    } else {
        0
    }
}

/// Double greater than
#[no_mangle]
pub extern "C" fn bhc_gt_double(a: f64, b: f64) -> i32 {
    if a > b {
        1
    } else {
        0
    }
}

/// Double greater than or equal
#[no_mangle]
pub extern "C" fn bhc_ge_double(a: f64, b: f64) -> i32 {
    if a >= b {
        1
    } else {
        0
    }
}

/// Double addition
#[no_mangle]
pub extern "C" fn bhc_add_double(a: f64, b: f64) -> f64 {
    a + b
}

/// Double subtraction
#[no_mangle]
pub extern "C" fn bhc_sub_double(a: f64, b: f64) -> f64 {
    a - b
}

/// Double multiplication
#[no_mangle]
pub extern "C" fn bhc_mul_double(a: f64, b: f64) -> f64 {
    a * b
}

/// Double division
#[no_mangle]
pub extern "C" fn bhc_div_double(a: f64, b: f64) -> f64 {
    a / b
}

/// Double negation
#[no_mangle]
pub extern "C" fn bhc_negate_double(a: f64) -> f64 {
    -a
}

/// Double absolute value
#[no_mangle]
pub extern "C" fn bhc_abs_double(a: f64) -> f64 {
    a.abs()
}

/// Integer to Double
#[no_mangle]
pub extern "C" fn bhc_integer_to_double(a: i64) -> f64 {
    a as f64
}

// Transcendental functions for Double

#[no_mangle]
pub extern "C" fn bhc_exp_double(a: f64) -> f64 {
    a.exp()
}

#[no_mangle]
pub extern "C" fn bhc_log_double(a: f64) -> f64 {
    a.ln()
}

#[no_mangle]
pub extern "C" fn bhc_sqrt_double(a: f64) -> f64 {
    a.sqrt()
}

#[no_mangle]
pub extern "C" fn bhc_pow_double(a: f64, b: f64) -> f64 {
    a.powf(b)
}

#[no_mangle]
pub extern "C" fn bhc_sin_double(a: f64) -> f64 {
    a.sin()
}

#[no_mangle]
pub extern "C" fn bhc_cos_double(a: f64) -> f64 {
    a.cos()
}

#[no_mangle]
pub extern "C" fn bhc_tan_double(a: f64) -> f64 {
    a.tan()
}

#[no_mangle]
pub extern "C" fn bhc_asin_double(a: f64) -> f64 {
    a.asin()
}

#[no_mangle]
pub extern "C" fn bhc_acos_double(a: f64) -> f64 {
    a.acos()
}

#[no_mangle]
pub extern "C" fn bhc_atan_double(a: f64) -> f64 {
    a.atan()
}

#[no_mangle]
pub extern "C" fn bhc_sinh_double(a: f64) -> f64 {
    a.sinh()
}

#[no_mangle]
pub extern "C" fn bhc_cosh_double(a: f64) -> f64 {
    a.cosh()
}

#[no_mangle]
pub extern "C" fn bhc_tanh_double(a: f64) -> f64 {
    a.tanh()
}

#[no_mangle]
pub extern "C" fn bhc_asinh_double(a: f64) -> f64 {
    a.asinh()
}

#[no_mangle]
pub extern "C" fn bhc_acosh_double(a: f64) -> f64 {
    a.acosh()
}

#[no_mangle]
pub extern "C" fn bhc_atanh_double(a: f64) -> f64 {
    a.atanh()
}

/// Truncate Double to Int
#[no_mangle]
pub extern "C" fn bhc_truncate_double(a: f64) -> i64 {
    a.trunc() as i64
}

/// Round Double to Int
#[no_mangle]
pub extern "C" fn bhc_round_double(a: f64) -> i64 {
    a.round() as i64
}

/// Ceiling of Double
#[no_mangle]
pub extern "C" fn bhc_ceiling_double(a: f64) -> i64 {
    a.ceil() as i64
}

/// Floor of Double
#[no_mangle]
pub extern "C" fn bhc_floor_double(a: f64) -> i64 {
    a.floor() as i64
}

/// Show Double
#[no_mangle]
pub extern "C" fn bhc_show_double(n: f64) -> *mut c_char {
    let s = format!("{}", n);
    let c_string = std::ffi::CString::new(s).unwrap();
    c_string.into_raw()
}

// ----------------------------------------------------------------------------
// Char Primitives
// ----------------------------------------------------------------------------

/// Char equality
#[no_mangle]
pub extern "C" fn bhc_eq_char(a: u32, b: u32) -> i32 {
    if a == b {
        1
    } else {
        0
    }
}

/// Char comparison
#[no_mangle]
pub extern "C" fn bhc_compare_char(a: u32, b: u32) -> i32 {
    match a.cmp(&b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Char less than
#[no_mangle]
pub extern "C" fn bhc_lt_char(a: u32, b: u32) -> i32 {
    if a < b {
        1
    } else {
        0
    }
}

/// Char less than or equal
#[no_mangle]
pub extern "C" fn bhc_le_char(a: u32, b: u32) -> i32 {
    if a <= b {
        1
    } else {
        0
    }
}

/// Char greater than
#[no_mangle]
pub extern "C" fn bhc_gt_char(a: u32, b: u32) -> i32 {
    if a > b {
        1
    } else {
        0
    }
}

/// Char greater than or equal
#[no_mangle]
pub extern "C" fn bhc_ge_char(a: u32, b: u32) -> i32 {
    if a >= b {
        1
    } else {
        0
    }
}

/// Char to Int (Unicode code point)
#[no_mangle]
pub extern "C" fn bhc_char_to_int(c: u32) -> i64 {
    c as i64
}

/// Int to Char (Unicode code point)
#[no_mangle]
pub extern "C" fn bhc_int_to_char(n: i64) -> u32 {
    n as u32
}

/// Show Char - returns the character representation with quotes
#[no_mangle]
pub extern "C" fn bhc_show_char(c: u32) -> *mut c_char {
    let s = if let Some(ch) = char::from_u32(c) {
        format!("'{}'", ch.escape_default())
    } else {
        format!("'\\x{:x}'", c)
    };
    let c_string = std::ffi::CString::new(s).unwrap();
    c_string.into_raw()
}

// ----------------------------------------------------------------------------
// Math: atan2 for Float and Double
// ----------------------------------------------------------------------------

/// atan2 for Float
#[no_mangle]
pub extern "C" fn bhc_atan2_float(y: f32, x: f32) -> f32 {
    y.atan2(x)
}

/// atan2 for Double
#[no_mangle]
pub extern "C" fn bhc_atan2_double(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Signum for Int
#[no_mangle]
pub extern "C" fn bhc_signum_int(a: i64) -> i64 {
    match a.cmp(&0) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Abs for Int
#[no_mangle]
pub extern "C" fn bhc_abs_int(a: i64) -> i64 {
    a.wrapping_abs()
}

/// Signum for Float
#[no_mangle]
pub extern "C" fn bhc_signum_float(a: f32) -> f32 {
    if a > 0.0 {
        1.0
    } else if a < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Signum for Double
#[no_mangle]
pub extern "C" fn bhc_signum_double(a: f64) -> f64 {
    if a > 0.0 {
        1.0
    } else if a < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ----------------------------------------------------------------------------
// IO: getLine, readFile, writeFile, appendFile
// ----------------------------------------------------------------------------

/// Read a line from stdin (FFI)
/// Returns a heap-allocated null-terminated C string, or null on error/EOF.
#[no_mangle]
pub extern "C" fn bhc_getLine() -> *mut c_char {
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) => ptr::null_mut(), // EOF
        Ok(_) => {
            // Remove trailing newline
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            match std::ffi::CString::new(line) {
                Ok(cstr) => cstr.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Print a newline to stdout
#[no_mangle]
pub extern "C" fn bhc_print_newline() {
    println!();
}

// ----------------------------------------------------------------------------
// String Memory Management
// ----------------------------------------------------------------------------

/// Free a string allocated by show functions
#[no_mangle]
pub unsafe extern "C" fn bhc_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = unsafe { std::ffi::CString::from_raw(s) };
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

    // Tests for Int primitives
    #[test]
    fn test_int_eq() {
        assert_eq!(bhc_eq_int(42, 42), 1);
        assert_eq!(bhc_eq_int(42, 43), 0);
    }

    #[test]
    fn test_int_compare() {
        assert_eq!(bhc_compare_int(1, 2), -1); // LT
        assert_eq!(bhc_compare_int(2, 2), 0); // EQ
        assert_eq!(bhc_compare_int(3, 2), 1); // GT
    }

    #[test]
    fn test_int_arithmetic() {
        assert_eq!(bhc_add_int(2, 3), 5);
        assert_eq!(bhc_sub_int(5, 3), 2);
        assert_eq!(bhc_mul_int(4, 5), 20);
        assert_eq!(bhc_negate_int(42), -42);
        assert_eq!(bhc_quot_int(7, 3), 2);
        assert_eq!(bhc_rem_int(7, 3), 1);
        assert_eq!(bhc_div_int(-7, 3), -3); // Euclidean division
        assert_eq!(bhc_mod_int(-7, 3), 2); // Euclidean modulus
    }

    #[test]
    fn test_int_bounds() {
        assert_eq!(bhc_min_int(), i64::MIN);
        assert_eq!(bhc_max_int(), i64::MAX);
    }

    // Tests for Float primitives
    #[test]
    fn test_float_arithmetic() {
        assert!((bhc_add_float(1.5, 2.5) - 4.0).abs() < 0.001);
        assert!((bhc_sub_float(5.0, 2.0) - 3.0).abs() < 0.001);
        assert!((bhc_mul_float(2.0, 3.0) - 6.0).abs() < 0.001);
        assert!((bhc_div_float(6.0, 2.0) - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_float_transcendentals() {
        assert!((bhc_sqrt_float(4.0) - 2.0).abs() < 0.001);
        assert!((bhc_sin_float(0.0)).abs() < 0.001);
        assert!((bhc_cos_float(0.0) - 1.0).abs() < 0.001);
        assert!((bhc_exp_float(0.0) - 1.0).abs() < 0.001);
        assert!((bhc_log_float(1.0)).abs() < 0.001);
    }

    // Tests for Double primitives
    #[test]
    fn test_double_arithmetic() {
        assert!((bhc_add_double(1.5, 2.5) - 4.0).abs() < 0.0001);
        assert!((bhc_sub_double(5.0, 2.0) - 3.0).abs() < 0.0001);
        assert!((bhc_mul_double(2.0, 3.0) - 6.0).abs() < 0.0001);
        assert!((bhc_div_double(6.0, 2.0) - 3.0).abs() < 0.0001);
    }

    #[test]
    fn test_double_rounding() {
        assert_eq!(bhc_truncate_double(3.7), 3);
        assert_eq!(bhc_truncate_double(-3.7), -3);
        assert_eq!(bhc_round_double(3.5), 4);
        assert_eq!(bhc_ceiling_double(3.1), 4);
        assert_eq!(bhc_floor_double(3.9), 3);
    }

    // Tests for Char primitives
    #[test]
    fn test_char_eq() {
        assert_eq!(bhc_eq_char('a' as u32, 'a' as u32), 1);
        assert_eq!(bhc_eq_char('a' as u32, 'b' as u32), 0);
    }

    #[test]
    fn test_char_conversion() {
        assert_eq!(bhc_char_to_int('A' as u32), 65);
        assert_eq!(bhc_int_to_char(65), 'A' as u32);
    }

    // Tests for Show functions
    #[test]
    fn test_show_int() {
        let s = bhc_show_int(42);
        let cstr = unsafe { std::ffi::CStr::from_ptr(s) };
        assert_eq!(cstr.to_str().unwrap(), "42");
        unsafe { bhc_free_string(s) };
    }

    #[test]
    fn test_show_char() {
        let s = bhc_show_char('a' as u32);
        let cstr = unsafe { std::ffi::CStr::from_ptr(s) };
        assert_eq!(cstr.to_str().unwrap(), "'a'");
        unsafe { bhc_free_string(s) };
    }
}
