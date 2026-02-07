//! C ABI entry points for the BHC Runtime System.
//!
//! These functions are called by compiled BHC programs. They provide
//! the interface between generated LLVM code and the Rust runtime.
//!
//! # Calling Convention
//!
//! All functions use the C calling convention and are `no_mangle` to
//! ensure stable symbol names for linking.

use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::AssertUnwindSafe;
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

/// GCD (Euclidean algorithm)
#[no_mangle]
pub extern "C" fn bhc_gcd(a: i64, b: i64) -> i64 {
    let mut x = a.abs();
    let mut y = b.abs();
    while y != 0 {
        let t = y;
        y = x % y;
        x = t;
    }
    x
}

/// LCM
#[no_mangle]
pub extern "C" fn bhc_lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / bhc_gcd(a, b) * b).abs()
    }
}

/// Create a new IORef (mutable reference cell)
#[no_mangle]
pub extern "C" fn bhc_new_ioref(val: *const u8) -> *mut u8 {
    let cell = Box::new(val);
    Box::into_raw(cell) as *mut u8
}

/// Read the value from an IORef
#[no_mangle]
pub extern "C" fn bhc_read_ioref(ref_ptr: *const u8) -> *const u8 {
    unsafe { *(ref_ptr as *const *const u8) }
}

/// Write a value to an IORef
#[no_mangle]
pub extern "C" fn bhc_write_ioref(ref_ptr: *mut u8, val: *const u8) {
    unsafe { *(ref_ptr as *mut *const u8) = val; }
}

/// Show Int - returns a heap-allocated string
#[no_mangle]
pub extern "C" fn bhc_show_int(n: i64) -> *mut c_char {
    let s = format!("{}", n);
    let c_string = std::ffi::CString::new(s).unwrap();
    c_string.into_raw()
}

/// Show Bool - returns "True" or "False" as a heap-allocated string.
/// Takes the ADT tag: 0 = False, 1 = True.
#[no_mangle]
pub extern "C" fn bhc_show_bool(tag: i64) -> *mut c_char {
    let s = if tag != 0 { "True" } else { "False" };
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
// Show: Compound Types (String, List, Maybe, Either, Tuple, Unit)
// ----------------------------------------------------------------------------

/// Format a single element for compound show functions.
/// type_tag: 0=Int, 1=Double, 2=Float, 3=Bool, 4=Char, 5=String
unsafe fn show_elem(ptr: *const u8, type_tag: i64) -> String {
    unsafe {
        match type_tag {
            0 => {
                // Int: pointer-as-integer
                let val = ptr as i64;
                format!("{}", val)
            }
            1 => {
                // Double: interpret as f64 bits
                let bits = ptr as i64;
                let val = f64::from_bits(bits as u64);
                format!("{}", val)
            }
            2 => {
                // Float: interpret as f32 bits (lower 32 bits)
                let bits = ptr as i32;
                let val = f32::from_bits(bits as u32);
                format!("{}", val)
            }
            3 => {
                // Bool: ADT with tag 0=False, 1=True
                let tag = *(ptr as *const i64);
                if tag != 0 { "True".to_string() } else { "False".to_string() }
            }
            4 => {
                // Char: stored as u32 via int-to-ptr
                let c = ptr as u32;
                if let Some(ch) = char::from_u32(c) {
                    format!("'{}'", ch.escape_default())
                } else {
                    format!("'\\x{:x}'", c)
                }
            }
            5 => {
                // String: [Char] linked list, show with quotes
                let s = read_char_list(ptr);
                format!("\"{}\"", s.escape_default())
            }
            _ => {
                // Default: treat as Int
                let val = ptr as i64;
                format!("{}", val)
            }
        }
    }
}

/// Read a [Char] linked list into a Rust String.
/// List layout: Nil=[tag=0], Cons=[tag=1][head@+8][tail@+16]
unsafe fn read_char_list(mut list_ptr: *const u8) -> String {
    unsafe {
        let mut result = String::new();
        while !list_ptr.is_null() {
            let tag = *(list_ptr as *const i64);
            if tag == 0 {
                break; // Nil
            }
            // Cons: head at +8, tail at +16
            let head = *(list_ptr.add(8) as *const *const u8);
            let char_val = head as u32;
            if let Some(ch) = char::from_u32(char_val) {
                result.push(ch);
            }
            list_ptr = *(list_ptr.add(16) as *const *const u8);
        }
        result
    }
}

/// Show String - wraps a [Char] list in quotes: "\"hello\""
#[no_mangle]
pub extern "C" fn bhc_show_string(list_ptr: *const u8) -> *mut c_char {
    let s = unsafe { read_char_list(list_ptr) };
    let shown = format!("\"{}\"", s.escape_default());
    let c_string = CString::new(shown).unwrap();
    c_string.into_raw()
}

/// Show List - formats a [a] list as "[el1,el2,el3]"
/// Special case: elem_type_tag==4 (Char) formats as String: "\"abc\""
#[no_mangle]
pub extern "C" fn bhc_show_list(list_ptr: *const u8, elem_type_tag: i64) -> *mut c_char {
    // Special case: [Char] is shown as a String
    if elem_type_tag == 4 {
        return bhc_show_string(list_ptr);
    }

    let mut elems = Vec::new();
    let mut cur = list_ptr;
    unsafe {
        while !cur.is_null() {
            let tag = *(cur as *const i64);
            if tag == 0 {
                break; // Nil
            }
            let head = *(cur.add(8) as *const *const u8);
            elems.push(show_elem(head, elem_type_tag));
            cur = *(cur.add(16) as *const *const u8);
        }
    }
    let shown = format!("[{}]", elems.join(","));
    let c_string = CString::new(shown).unwrap();
    c_string.into_raw()
}

/// Show Maybe - "Nothing" or "Just <val>"
#[no_mangle]
pub extern "C" fn bhc_show_maybe(maybe_ptr: *const u8, elem_type_tag: i64) -> *mut c_char {
    let tag = unsafe { *(maybe_ptr as *const i64) };
    let shown = if tag == 0 {
        "Nothing".to_string()
    } else {
        let val = unsafe { *(maybe_ptr.add(8) as *const *const u8) };
        format!("Just {}", unsafe { show_elem(val, elem_type_tag) })
    };
    let c_string = CString::new(shown).unwrap();
    c_string.into_raw()
}

/// Show Either - "Left <val>" or "Right <val>"
#[no_mangle]
pub extern "C" fn bhc_show_either(either_ptr: *const u8, left_type_tag: i64, right_type_tag: i64) -> *mut c_char {
    let tag = unsafe { *(either_ptr as *const i64) };
    let shown = if tag == 0 {
        let val = unsafe { *(either_ptr.add(8) as *const *const u8) };
        format!("Left {}", unsafe { show_elem(val, left_type_tag) })
    } else {
        let val = unsafe { *(either_ptr.add(8) as *const *const u8) };
        format!("Right {}", unsafe { show_elem(val, right_type_tag) })
    };
    let c_string = CString::new(shown).unwrap();
    c_string.into_raw()
}

/// Show Tuple2 - "(fst,snd)"
#[no_mangle]
pub extern "C" fn bhc_show_tuple2(tuple_ptr: *const u8, fst_type_tag: i64, snd_type_tag: i64) -> *mut c_char {
    let fst = unsafe { *(tuple_ptr.add(8) as *const *const u8) };
    let snd = unsafe { *(tuple_ptr.add(16) as *const *const u8) };
    let shown = format!("({},{})", unsafe { show_elem(fst, fst_type_tag) }, unsafe { show_elem(snd, snd_type_tag) });
    let c_string = CString::new(shown).unwrap();
    c_string.into_raw()
}

/// Show Unit - "()"
#[no_mangle]
pub extern "C" fn bhc_show_unit(_unit_ptr: *const u8) -> *mut c_char {
    let c_string = CString::new("()").unwrap();
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

// ----------------------------------------------------------------------------
// File IO: readFile, writeFile, appendFile
// ----------------------------------------------------------------------------

/// Compute the length of a C string (wrapper around strlen).
/// Used by `length` when applied to raw C strings (e.g. from readFile).
#[no_mangle]
pub extern "C" fn bhc_string_length(s: *const c_char) -> i64 {
    if s.is_null() {
        return 0;
    }
    unsafe { CStr::from_ptr(s) }.to_bytes().len() as i64
}

/// Read entire file contents into a heap-allocated C string.
/// Returns null on error.
#[no_mangle]
pub extern "C" fn bhc_readFile(path: *const c_char) -> *mut c_char {
    if path.is_null() {
        return ptr::null_mut();
    }
    let path_str = unsafe { CStr::from_ptr(path) }.to_str().unwrap_or("");
    match std::fs::read_to_string(path_str) {
        Ok(contents) => {
            CString::new(contents).map_or(ptr::null_mut(), |cs| cs.into_raw())
        }
        Err(e) => {
            // Throw an exception with the error message as a C string
            let msg = format!("{}: {}", path_str, e);
            let exc = CString::new(msg).map_or(ptr::null_mut(), |cs| cs.into_raw());
            bhc_throw(exc as *mut u8) as *mut c_char
        }
    }
}

/// Write a C string to a file, replacing its contents.
#[no_mangle]
pub extern "C" fn bhc_writeFile(path: *const c_char, content: *const c_char) {
    if path.is_null() || content.is_null() {
        return;
    }
    let p = unsafe { CStr::from_ptr(path) }.to_str().unwrap_or("");
    let c = unsafe { CStr::from_ptr(content) }.to_str().unwrap_or("");
    let _ = std::fs::write(p, c);
}

/// Append a C string to a file, creating it if needed.
#[no_mangle]
pub extern "C" fn bhc_appendFile(path: *const c_char, content: *const c_char) {
    if path.is_null() || content.is_null() {
        return;
    }
    let p = unsafe { CStr::from_ptr(path) }.to_str().unwrap_or("");
    let c = unsafe { CStr::from_ptr(content) }.to_str().unwrap_or("");
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(p)
    {
        let _ = f.write_all(c.as_bytes());
    }
}

// ----------------------------------------------------------------------------
// String split/join: lines, words, unlines, unwords
// ----------------------------------------------------------------------------

/// Allocate a cons cell: `[i64 tag=1, ptr head, ptr tail]` (24 bytes).
unsafe fn alloc_cons(head: *mut u8, tail: *mut u8) -> *mut u8 {
    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(24, 8);
        let cell = std::alloc::alloc(layout);
        *(cell as *mut i64) = 1; // tag = cons
        *(cell.add(8) as *mut *mut u8) = head;
        *(cell.add(16) as *mut *mut u8) = tail;
        cell
    }
}

/// Allocate a nil cell: `[i64 tag=0, ...]` (24 bytes, matching cons layout).
unsafe fn alloc_nil() -> *mut u8 {
    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(24, 8);
        let cell = std::alloc::alloc(layout);
        *(cell as *mut i64) = 0;
        cell
    }
}

/// Split a C string by `'\n'`, returning a cons-cell list of C strings.
#[no_mangle]
pub extern "C" fn bhc_string_lines(s: *const c_char) -> *mut u8 {
    if s.is_null() {
        return unsafe { alloc_nil() };
    }
    let rust_str = unsafe { CStr::from_ptr(s) }.to_str().unwrap_or("");
    // Match Haskell semantics: lines "a\nb\n" == ["a", "b"]
    // split('\n') would produce ["a", "b", ""], so filter trailing empty.
    let mut parts: Vec<&str> = rust_str.split('\n').collect();
    if parts.last() == Some(&"") {
        parts.pop();
    }
    let mut list = unsafe { alloc_nil() };
    for part in parts.iter().rev() {
        let cs = CString::new(*part).unwrap_or_default();
        list = unsafe { alloc_cons(cs.into_raw() as *mut u8, list) };
    }
    list
}

/// Split a C string by whitespace, returning a cons-cell list of C strings.
#[no_mangle]
pub extern "C" fn bhc_string_words(s: *const c_char) -> *mut u8 {
    if s.is_null() {
        return unsafe { alloc_nil() };
    }
    let rust_str = unsafe { CStr::from_ptr(s) }.to_str().unwrap_or("");
    let parts: Vec<&str> = rust_str.split_whitespace().collect();
    let mut list = unsafe { alloc_nil() };
    for part in parts.iter().rev() {
        let cs = CString::new(*part).unwrap_or_default();
        list = unsafe { alloc_cons(cs.into_raw() as *mut u8, list) };
    }
    list
}

/// Traverse a cons-cell list, collecting each head as a string.
fn collect_list_strings(list: *const u8) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = list;
    while !cur.is_null() {
        let tag = unsafe { *(cur as *const i64) };
        if tag == 0 {
            break;
        }
        let head = unsafe { *(cur.add(8) as *const *const c_char) };
        if !head.is_null() {
            let s = unsafe { CStr::from_ptr(head) }
                .to_str()
                .unwrap_or("");
            parts.push(s.to_string());
        }
        cur = unsafe { *(cur.add(16) as *const *const u8) };
    }
    parts
}

/// Join a cons-cell list of C strings, appending `'\n'` after each element.
/// Matches Haskell semantics: unlines ["a","b","c"] == "a\nb\nc\n"
#[no_mangle]
pub extern "C" fn bhc_string_unlines(list: *const u8) -> *mut c_char {
    let parts = collect_list_strings(list);
    let mut joined = String::new();
    for part in &parts {
        joined.push_str(part);
        joined.push('\n');
    }
    CString::new(joined).map_or(ptr::null_mut(), |cs| cs.into_raw())
}

/// Join a cons-cell list of C strings with `' '`.
#[no_mangle]
pub extern "C" fn bhc_string_unwords(list: *const u8) -> *mut c_char {
    let parts = collect_list_strings(list);
    let joined = parts.join(" ");
    CString::new(joined).map_or(ptr::null_mut(), |cs| cs.into_raw())
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

// ============================================================================
// Exception Handling
// ============================================================================
//
// BHC exceptions use thread-local storage to communicate exception state
// between bhc_throw and bhc_catch. This avoids unwinding across extern "C"
// boundaries, which is undefined behavior.
//
// Protocol:
//   1. bhc_catch saves the current exception state, calls the action.
//   2. If bhc_throw is called, it stores the exception pointer in TLS
//      and returns a sentinel null value.
//   3. bhc_catch checks TLS after the action returns; if an exception
//      was thrown, it invokes the handler.
//
// bhc_evaluate: Forces a value to WHNF (currently a no-op).
// bhc_mask/bhc_unmask: Async exception masking stubs.
// ============================================================================

use std::cell::Cell;

thread_local! {
    /// Thread-local exception state. `None` means no exception is pending.
    /// `Some(ptr)` means bhc_throw was called with the given exception pointer.
    static BHC_EXCEPTION: Cell<Option<*mut u8>> = const { Cell::new(None) };
}

/// Sentinel return value used by bhc_throw to indicate an exception was thrown.
/// bhc_catch checks for pending exceptions after the action returns.
const BHC_EXCEPTION_SENTINEL: *mut u8 = ptr::null_mut();

/// Throw a Haskell exception.
///
/// Stores the exception pointer in thread-local storage and aborts the
/// current computation by returning a sentinel value. When called from
/// within a bhc_catch scope, the catch handler will be invoked.
///
/// # Arguments
///
/// * `exception_ptr` - Pointer to the Haskell exception value on the heap.
///
/// # Safety
///
/// `exception_ptr` must be a valid pointer to a heap-allocated Haskell value.
/// This function diverges (never returns normally to the caller) by using
/// longjmp-style control flow via the RTS.
#[no_mangle]
pub extern "C" fn bhc_throw(exception_ptr: *mut u8) -> *mut u8 {
    BHC_EXCEPTION.with(|cell| {
        cell.set(Some(exception_ptr));
    });
    BHC_EXCEPTION_SENTINEL
}

/// Catch a Haskell exception.
///
/// Runs `action_fn(action_env)`. If the action (or any function it calls)
/// invokes `bhc_throw`, the exception is caught and `handler_fn` is called
/// with the exception pointer.
///
/// # Arguments
///
/// * `action_fn` - Function pointer for the IO action to run.
/// * `action_env` - Environment/closure pointer for the action.
/// * `handler_fn` - Function pointer for the exception handler.
/// * `handler_env` - Environment/closure pointer for the handler.
///
/// # Returns
///
/// The result of either the action or the handler.
///
/// # Safety
///
/// All function pointers and environment pointers must be valid.
#[no_mangle]
pub extern "C" fn bhc_catch(
    action_fn: extern "C" fn(*mut u8) -> *mut u8,
    action_env: *mut u8,
    handler_fn: extern "C" fn(*mut u8, *mut u8) -> *mut u8,
    handler_env: *mut u8,
) -> *mut u8 {
    // Save any pre-existing exception state
    let saved = BHC_EXCEPTION.with(|cell| cell.replace(None));

    // Run the action
    let result = action_fn(action_env);

    // Check if an exception was thrown during the action
    let thrown = BHC_EXCEPTION.with(|cell| cell.replace(None));

    match thrown {
        Some(exc_ptr) => {
            // Exception was thrown — invoke the handler
            handler_fn(handler_env, exc_ptr)
        }
        None => {
            // No exception — restore saved state and return result
            if saved.is_some() {
                BHC_EXCEPTION.with(|cell| cell.set(saved));
            }
            result
        }
    }
}

/// Force a value to Weak Head Normal Form.
///
/// Currently a no-op since BHC eagerly evaluates to WHNF.
/// This exists for `Control.Exception.evaluate` support.
///
/// # Arguments
///
/// * `val` - Pointer to the value to evaluate.
///
/// # Returns
///
/// The same pointer (value is already in WHNF).
#[no_mangle]
pub extern "C" fn bhc_evaluate(val: *mut u8) -> *mut u8 {
    val
}

/// Execute an IO action with guaranteed cleanup.
///
/// Runs `action_fn(action_env)`. Regardless of whether the action succeeds
/// or throws, `cleanup_fn(cleanup_env)` is always executed afterward.
/// If the action threw an exception, the exception is re-thrown after cleanup.
///
/// Implements Haskell's `finally :: IO a -> IO b -> IO a`.
///
/// # Arguments
///
/// * `action_fn` - Function pointer for the IO action.
/// * `action_env` - Environment/closure pointer for the action.
/// * `cleanup_fn` - Function pointer for the cleanup action.
/// * `cleanup_env` - Environment/closure pointer for the cleanup.
///
/// # Returns
///
/// The result of the action if no exception occurred.
#[no_mangle]
pub extern "C" fn bhc_finally(
    action_fn: extern "C" fn(*mut u8) -> *mut u8,
    action_env: *mut u8,
    cleanup_fn: extern "C" fn(*mut u8) -> *mut u8,
    cleanup_env: *mut u8,
) -> *mut u8 {
    // Save any pre-existing exception state
    let saved = BHC_EXCEPTION.with(|cell| cell.replace(None));

    // Run the action
    let result = action_fn(action_env);

    // Check if an exception was thrown
    let thrown = BHC_EXCEPTION.with(|cell| cell.replace(None));

    // Always run cleanup
    let _ = cleanup_fn(cleanup_env);

    match thrown {
        Some(exc_ptr) => {
            // Exception was thrown — re-throw after cleanup
            BHC_EXCEPTION.with(|cell| cell.set(Some(exc_ptr)));
            BHC_EXCEPTION_SENTINEL
        }
        None => {
            // No exception — restore saved state and return result
            if saved.is_some() {
                BHC_EXCEPTION.with(|cell| cell.set(saved));
            }
            result
        }
    }
}

/// Execute an IO action, running a handler only if an exception occurs.
///
/// Runs `action_fn(action_env)`. If the action throws an exception,
/// `handler_fn(handler_env)` is executed before re-throwing the exception.
/// Unlike `catch`, the exception is always re-thrown.
///
/// Implements Haskell's `onException :: IO a -> IO b -> IO a`.
///
/// # Arguments
///
/// * `action_fn` - Function pointer for the IO action.
/// * `action_env` - Environment/closure pointer for the action.
/// * `handler_fn` - Function pointer for the exception handler.
/// * `handler_env` - Environment/closure pointer for the handler.
///
/// # Returns
///
/// The result of the action if no exception occurred.
#[no_mangle]
pub extern "C" fn bhc_on_exception(
    action_fn: extern "C" fn(*mut u8) -> *mut u8,
    action_env: *mut u8,
    handler_fn: extern "C" fn(*mut u8) -> *mut u8,
    handler_env: *mut u8,
) -> *mut u8 {
    // Save any pre-existing exception state
    let saved = BHC_EXCEPTION.with(|cell| cell.replace(None));

    // Run the action
    let result = action_fn(action_env);

    // Check if an exception was thrown
    let thrown = BHC_EXCEPTION.with(|cell| cell.replace(None));

    match thrown {
        Some(exc_ptr) => {
            // Exception was thrown — run handler, then re-throw
            let _ = handler_fn(handler_env);
            BHC_EXCEPTION.with(|cell| cell.set(Some(exc_ptr)));
            BHC_EXCEPTION_SENTINEL
        }
        None => {
            // No exception — restore saved state and return result
            if saved.is_some() {
                BHC_EXCEPTION.with(|cell| cell.set(saved));
            }
            result
        }
    }
}

/// Acquire-use-release pattern with guaranteed cleanup.
///
/// Runs `acquire_fn(acquire_env)` to get a resource, then
/// `use_fn(use_env, resource)` to use it, and always runs
/// `release_fn(release_env, resource)` afterward (even on exception).
///
/// Implements Haskell's `bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c`.
///
/// # Arguments
///
/// * `acquire_fn` - Function pointer that acquires the resource.
/// * `acquire_env` - Environment/closure pointer for acquire.
/// * `release_fn` - Function pointer that releases the resource (takes env + resource).
/// * `release_env` - Environment/closure pointer for release.
/// * `use_fn` - Function pointer that uses the resource (takes env + resource).
/// * `use_env` - Environment/closure pointer for use.
///
/// # Returns
///
/// The result of use_fn if no exception occurred.
#[no_mangle]
pub extern "C" fn bhc_bracket(
    acquire_fn: extern "C" fn(*mut u8) -> *mut u8,
    acquire_env: *mut u8,
    release_fn: extern "C" fn(*mut u8, *mut u8) -> *mut u8,
    release_env: *mut u8,
    use_fn: extern "C" fn(*mut u8, *mut u8) -> *mut u8,
    use_env: *mut u8,
) -> *mut u8 {
    // Acquire the resource
    let resource = acquire_fn(acquire_env);

    // Check if acquire threw
    let acquire_thrown = BHC_EXCEPTION.with(|cell| cell.replace(None));
    if let Some(exc_ptr) = acquire_thrown {
        BHC_EXCEPTION.with(|cell| cell.set(Some(exc_ptr)));
        return BHC_EXCEPTION_SENTINEL;
    }

    // Use the resource
    let result = use_fn(use_env, resource);

    // Check if use threw
    let use_thrown = BHC_EXCEPTION.with(|cell| cell.replace(None));

    // Always release (even on exception)
    let _ = release_fn(release_env, resource);

    match use_thrown {
        Some(exc_ptr) => {
            // Use threw — re-throw after release
            BHC_EXCEPTION.with(|cell| cell.set(Some(exc_ptr)));
            BHC_EXCEPTION_SENTINEL
        }
        None => result,
    }
}

/// Mask asynchronous exceptions (stub).
///
/// Currently just runs the action without masking, since BHC
/// does not yet support asynchronous exceptions.
///
/// # Arguments
///
/// * `action_fn` - Function pointer for the IO action.
/// * `action_env` - Environment/closure pointer.
///
/// # Returns
///
/// The result of the action.
#[no_mangle]
pub extern "C" fn bhc_mask(
    action_fn: extern "C" fn(*mut u8) -> *mut u8,
    action_env: *mut u8,
) -> *mut u8 {
    action_fn(action_env)
}

/// Unmask asynchronous exceptions (stub).
///
/// Currently just runs the action, since BHC does not yet
/// support asynchronous exceptions.
///
/// # Arguments
///
/// * `action_fn` - Function pointer for the IO action.
/// * `action_env` - Environment/closure pointer.
///
/// # Returns
///
/// The result of the action.
#[no_mangle]
pub extern "C" fn bhc_unmask(
    action_fn: extern "C" fn(*mut u8) -> *mut u8,
    action_env: *mut u8,
) -> *mut u8 {
    action_fn(action_env)
}

// ============================================================================
// Handle-based IO
// ============================================================================
//
// Handles use sentinel pointers for standard streams:
//   1 = stdin, 2 = stdout, 3 = stderr
// Real file handles are heap-allocated BhcHandle structs leaked as *mut u8.
// ============================================================================

use std::io::{BufRead, Read, Seek, Write};

/// Internal handle representation tracking open state and capabilities.
struct BhcHandle {
    file: Option<std::fs::File>,
    readable: bool,
    writable: bool,
    closed: bool,
}

const HANDLE_STDIN: usize = 1;
const HANDLE_STDOUT: usize = 2;
const HANDLE_STDERR: usize = 3;

fn is_sentinel(handle: *mut u8) -> bool {
    let h = handle as usize;
    h == HANDLE_STDIN || h == HANDLE_STDOUT || h == HANDLE_STDERR
}

fn get_bhc_handle(handle: *mut u8) -> Option<&'static mut BhcHandle> {
    if handle.is_null() || is_sentinel(handle) {
        None
    } else {
        Some(unsafe { &mut *(handle as *mut BhcHandle) })
    }
}

// ----------------------------------------------------------------------------
// Standard handle accessors
// ----------------------------------------------------------------------------

/// Return sentinel pointer for stdin.
#[no_mangle]
pub extern "C" fn bhc_stdin() -> *mut u8 {
    HANDLE_STDIN as *mut u8
}

/// Return sentinel pointer for stdout.
#[no_mangle]
pub extern "C" fn bhc_stdout() -> *mut u8 {
    HANDLE_STDOUT as *mut u8
}

/// Return sentinel pointer for stderr.
#[no_mangle]
pub extern "C" fn bhc_stderr() -> *mut u8 {
    HANDLE_STDERR as *mut u8
}

// ----------------------------------------------------------------------------
// File open / close
// ----------------------------------------------------------------------------

/// Open a file and return a handle pointer.
///
/// Mode: 0=ReadMode, 1=WriteMode, 2=AppendMode, 3=ReadWriteMode.
/// Returns null on error.
#[no_mangle]
pub unsafe extern "C" fn bhc_open_file(path: *const c_char, mode: c_int) -> *mut u8 {
    if path.is_null() {
        return ptr::null_mut();
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let (file_result, readable, writable) = match mode {
        0 => (std::fs::File::open(path_str), true, false),
        1 => (std::fs::File::create(path_str), false, true),
        2 => (
            std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(path_str),
            false,
            true,
        ),
        3 => (
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(path_str),
            true,
            true,
        ),
        _ => return ptr::null_mut(),
    };

    match file_result {
        Ok(file) => {
            let handle = Box::new(BhcHandle {
                file: Some(file),
                readable,
                writable,
                closed: false,
            });
            Box::into_raw(handle) as *mut u8
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Close a handle. No-op for standard stream sentinels.
#[no_mangle]
pub unsafe extern "C" fn bhc_close_handle(handle: *mut u8) {
    if handle.is_null() || is_sentinel(handle) {
        return;
    }
    let bh = unsafe { &mut *(handle as *mut BhcHandle) };
    bh.file = None;
    bh.closed = true;
    // Note: We intentionally don't free the BhcHandle here to avoid
    // use-after-free if code still holds a reference. The memory will
    // be reclaimed on process exit.
}

// ----------------------------------------------------------------------------
// Reading
// ----------------------------------------------------------------------------

/// Read a single character from a handle.
/// Returns the character code, or -1 on EOF/error.
#[no_mangle]
pub unsafe extern "C" fn bhc_hGetChar(handle: *mut u8) -> c_int {
    let h = handle as usize;
    if h == HANDLE_STDIN {
        let mut buf = [0u8; 1];
        match std::io::stdin().read(&mut buf) {
            Ok(1) => buf[0] as c_int,
            _ => -1,
        }
    } else if let Some(bh) = get_bhc_handle(handle) {
        if let Some(ref mut f) = bh.file {
            let mut buf = [0u8; 1];
            match f.read(&mut buf) {
                Ok(1) => buf[0] as c_int,
                _ => -1,
            }
        } else {
            -1
        }
    } else {
        -1
    }
}

/// Read a line from a handle. Returns a heap-allocated C string, or null on EOF/error.
/// Reads byte-by-byte to avoid BufReader consuming ahead of the file position.
#[no_mangle]
pub unsafe extern "C" fn bhc_hGetLine(handle: *mut u8) -> *mut c_char {
    let h = handle as usize;

    if h == HANDLE_STDIN {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => return ptr::null_mut(),
            Ok(_) => {
                if line.ends_with('\n') {
                    line.pop();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                }
                return CString::new(line).map_or(ptr::null_mut(), |cs| cs.into_raw());
            }
            Err(_) => return ptr::null_mut(),
        }
    }

    if let Some(bh) = get_bhc_handle(handle) {
        if let Some(ref mut f) = bh.file {
            let mut line = Vec::new();
            let mut buf = [0u8; 1];
            let mut bytes_read = 0usize;
            loop {
                match f.read(&mut buf) {
                    Ok(0) => break,       // EOF
                    Ok(_) => {
                        bytes_read += 1;
                        if buf[0] == b'\n' {
                            break;
                        }
                        line.push(buf[0]);
                    }
                    Err(_) => break,
                }
            }
            // If nothing was read at all, we're at EOF
            if bytes_read == 0 {
                return ptr::null_mut();
            }
            // Remove trailing \r if present
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            match CString::new(line) {
                Ok(cs) => cs.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        } else {
            ptr::null_mut()
        }
    } else {
        ptr::null_mut()
    }
}

/// Read all remaining contents from a handle. Returns heap-allocated C string.
#[no_mangle]
pub unsafe extern "C" fn bhc_hGetContents(handle: *mut u8) -> *mut c_char {
    let h = handle as usize;
    let mut contents = String::new();

    let result = if h == HANDLE_STDIN {
        std::io::stdin().read_to_string(&mut contents)
    } else if let Some(bh) = get_bhc_handle(handle) {
        if let Some(ref mut f) = bh.file {
            f.read_to_string(&mut contents)
        } else {
            return ptr::null_mut();
        }
    } else {
        return ptr::null_mut();
    };

    match result {
        Ok(_) => CString::new(contents).map_or(ptr::null_mut(), |cs| cs.into_raw()),
        Err(_) => ptr::null_mut(),
    }
}

/// Check if a handle is at EOF. Returns 1 if EOF, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn bhc_hIsEOF(handle: *mut u8) -> c_int {
    if let Some(bh) = get_bhc_handle(handle) {
        if let Some(ref mut f) = bh.file {
            let mut buf = [0u8; 1];
            match f.read(&mut buf) {
                Ok(0) => 1,
                Ok(_) => {
                    // We read a byte, seek back
                    let _ = f.seek(std::io::SeekFrom::Current(-1));
                    0
                }
                Err(_) => 1,
            }
        } else {
            1
        }
    } else {
        0 // stdin: can't easily check EOF
    }
}

// ----------------------------------------------------------------------------
// Writing
// ----------------------------------------------------------------------------

/// Write a string to a handle.
#[no_mangle]
pub unsafe extern "C" fn bhc_hPutStr(handle: *mut u8, s: *const c_char) {
    if s.is_null() {
        return;
    }
    let str_val = match unsafe { CStr::from_ptr(s) }.to_str() {
        Ok(s) => s,
        Err(_) => return,
    };

    let h = handle as usize;
    if h == HANDLE_STDOUT {
        let _ = std::io::stdout().write_all(str_val.as_bytes());
    } else if h == HANDLE_STDERR {
        let _ = std::io::stderr().write_all(str_val.as_bytes());
    } else if let Some(bh) = get_bhc_handle(handle) {
        if let Some(ref mut f) = bh.file {
            let _ = f.write_all(str_val.as_bytes());
        }
    }
}

/// Write a single character to a handle.
#[no_mangle]
pub unsafe extern "C" fn bhc_hPutChar(handle: *mut u8, c: c_int) {
    let ch = match char::from_u32(c as u32) {
        Some(ch) => ch,
        None => return,
    };
    let mut buf = [0u8; 4];
    let s = ch.encode_utf8(&mut buf);

    let h = handle as usize;
    if h == HANDLE_STDOUT {
        let _ = std::io::stdout().write_all(s.as_bytes());
    } else if h == HANDLE_STDERR {
        let _ = std::io::stderr().write_all(s.as_bytes());
    } else if let Some(bh) = get_bhc_handle(handle) {
        if let Some(ref mut f) = bh.file {
            let _ = f.write_all(s.as_bytes());
        }
    }
}

/// Flush a handle's output buffer.
#[no_mangle]
pub unsafe extern "C" fn bhc_hFlush(handle: *mut u8) {
    let h = handle as usize;
    if h == HANDLE_STDOUT {
        let _ = std::io::stdout().flush();
    } else if h == HANDLE_STDERR {
        let _ = std::io::stderr().flush();
    } else if let Some(bh) = get_bhc_handle(handle) {
        if let Some(ref mut f) = bh.file {
            let _ = f.flush();
        }
    }
}

// ----------------------------------------------------------------------------
// Handle properties
// ----------------------------------------------------------------------------

/// Check if a handle is open. Returns 1 if open, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_hIsOpen(handle: *mut u8) -> c_int {
    if is_sentinel(handle) {
        return 1; // std handles are always open
    }
    if let Some(bh) = get_bhc_handle(handle) {
        if bh.closed { 0 } else { 1 }
    } else {
        0
    }
}

/// Check if a handle is closed. Returns 1 if closed, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_hIsClosed(handle: *mut u8) -> c_int {
    if is_sentinel(handle) {
        return 0;
    }
    if let Some(bh) = get_bhc_handle(handle) {
        if bh.closed { 1 } else { 0 }
    } else {
        1
    }
}

/// Check if a handle is readable. Returns 1 if readable, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_hIsReadable(handle: *mut u8) -> c_int {
    let h = handle as usize;
    if h == HANDLE_STDIN {
        return 1;
    }
    if h == HANDLE_STDOUT || h == HANDLE_STDERR {
        return 0;
    }
    if let Some(bh) = get_bhc_handle(handle) {
        if bh.readable { 1 } else { 0 }
    } else {
        0
    }
}

/// Check if a handle is writable. Returns 1 if writable, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_hIsWritable(handle: *mut u8) -> c_int {
    let h = handle as usize;
    if h == HANDLE_STDIN {
        return 0;
    }
    if h == HANDLE_STDOUT || h == HANDLE_STDERR {
        return 1;
    }
    if let Some(bh) = get_bhc_handle(handle) {
        if bh.writable { 1 } else { 0 }
    } else {
        0
    }
}

/// Check if a handle is seekable. Returns 1 if seekable, 0 otherwise.
#[no_mangle]
pub extern "C" fn bhc_hIsSeekable(handle: *mut u8) -> c_int {
    if is_sentinel(handle) {
        return 0; // std streams are not seekable
    }
    if let Some(bh) = get_bhc_handle(handle) {
        if bh.closed {
            return 0;
        }
        if let Some(ref mut f) = bh.file {
            // Try seeking to current position to test seekability
            match f.stream_position() {
                Ok(_) => 1,
                Err(_) => 0,
            }
        } else {
            0
        }
    } else {
        0
    }
}

/// Set buffering mode for a handle. mode: 0=NoBuffering, 1=LineBuffering, 2=BlockBuffering.
/// Currently a no-op stub since Rust handles buffering internally.
#[no_mangle]
pub extern "C" fn bhc_hSetBuffering(_handle: *mut u8, _mode: c_int) {
    // Stub: Rust's BufWriter/BufReader handle buffering.
}

/// Get buffering mode for a handle. Returns 0=NoBuffering, 1=LineBuffering, 2=BlockBuffering.
/// Currently returns 2 (BlockBuffering) as default.
#[no_mangle]
pub extern "C" fn bhc_hGetBuffering(_handle: *mut u8) -> c_int {
    2 // BlockBuffering
}

/// Seek within a handle.
/// mode: 0=AbsoluteSeek, 1=RelativeSeek, 2=SeekFromEnd.
#[no_mangle]
pub unsafe extern "C" fn bhc_hSeek(handle: *mut u8, mode: c_int, offset: i64) {
    if is_sentinel(handle) {
        return;
    }
    if let Some(bh) = get_bhc_handle(handle) {
        if let Some(ref mut f) = bh.file {
            let seek_from = match mode {
                0 => std::io::SeekFrom::Start(offset as u64),
                1 => std::io::SeekFrom::Current(offset),
                2 => std::io::SeekFrom::End(offset),
                _ => return,
            };
            let _ = f.seek(seek_from);
        }
    }
}

/// Get the current position in a handle. Returns -1 on error.
#[no_mangle]
pub unsafe extern "C" fn bhc_hTell(handle: *mut u8) -> i64 {
    if is_sentinel(handle) {
        return -1;
    }
    if let Some(bh) = get_bhc_handle(handle) {
        if let Some(ref mut f) = bh.file {
            match f.stream_position() {
                Ok(pos) => pos as i64,
                Err(_) => -1,
            }
        } else {
            -1
        }
    } else {
        -1
    }
}

// ----------------------------------------------------------------------------
// System / filesystem operations
// ----------------------------------------------------------------------------

/// Check if a file exists. Returns 1 if it exists, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn bhc_exists(path: *const c_char) -> c_int {
    if path.is_null() {
        return 0;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    if std::path::Path::new(path_str).exists() {
        1
    } else {
        0
    }
}

/// Check if a path is a directory. Returns 1 if directory, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn bhc_is_directory(path: *const c_char) -> c_int {
    if path.is_null() {
        return 0;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };
    if std::path::Path::new(path_str).is_dir() {
        1
    } else {
        0
    }
}

/// Get an environment variable. Returns heap-allocated C string, or null if not set.
#[no_mangle]
pub unsafe extern "C" fn bhc_get_env(name: *const c_char) -> *mut c_char {
    if name.is_null() {
        return ptr::null_mut();
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    match std::env::var(name_str) {
        Ok(val) => CString::new(val).map_or(ptr::null_mut(), |cs| cs.into_raw()),
        Err(_) => ptr::null_mut(),
    }
}

/// Lookup an environment variable. Returns heap-allocated C string, or null for Nothing.
#[no_mangle]
pub unsafe extern "C" fn bhc_lookupEnv(name: *const c_char) -> *mut c_char {
    // Same as bhc_get_env; null encodes Nothing
    unsafe { bhc_get_env(name) }
}

/// Get command line arguments as a cons-cell list of C strings.
#[no_mangle]
pub extern "C" fn bhc_get_args() -> *mut u8 {
    let args: Vec<String> = std::env::args().collect();
    let mut list = unsafe { alloc_nil() };
    for arg in args.iter().rev() {
        let cs = CString::new(arg.as_str()).unwrap_or_default();
        list = unsafe { alloc_cons(cs.into_raw() as *mut u8, list) };
    }
    list
}

/// Get the program name. Returns heap-allocated C string.
#[no_mangle]
pub extern "C" fn bhc_get_prog_name() -> *mut c_char {
    match std::env::args().next() {
        Some(name) => CString::new(name).map_or(ptr::null_mut(), |cs| cs.into_raw()),
        None => CString::new("bhc").map_or(ptr::null_mut(), |cs| cs.into_raw()),
    }
}

/// Get the current working directory. Returns heap-allocated C string.
#[no_mangle]
pub extern "C" fn bhc_get_current_directory() -> *mut c_char {
    match std::env::current_dir() {
        Ok(path) => {
            let s = path.to_string_lossy().into_owned();
            CString::new(s).map_or(ptr::null_mut(), |cs| cs.into_raw())
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Create a directory (and parents). No-op on error.
#[no_mangle]
pub unsafe extern "C" fn bhc_create_directory(path: *const c_char) {
    if path.is_null() {
        return;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return,
    };
    let _ = std::fs::create_dir_all(path_str);
}

/// Remove a file. No-op on error.
#[no_mangle]
pub unsafe extern "C" fn bhc_remove_file(path: *const c_char) {
    if path.is_null() {
        return;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return,
    };
    let _ = std::fs::remove_file(path_str);
}

/// List directory entries as a cons-cell list of C strings.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_directory(path: *const c_char) -> *mut u8 {
    if path.is_null() {
        return unsafe { alloc_nil() };
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return unsafe { alloc_nil() },
    };
    match std::fs::read_dir(path_str) {
        Ok(entries) => {
            let names: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            let mut list = unsafe { alloc_nil() };
            for name in names.iter().rev() {
                let cs = CString::new(name.as_str()).unwrap_or_default();
                list = unsafe { alloc_cons(cs.into_raw() as *mut u8, list) };
            }
            list
        }
        Err(_) => unsafe { alloc_nil() },
    }
}

/// Exit with success (code 0).
#[no_mangle]
pub extern "C" fn bhc_exit_success() -> ! {
    bhc_shutdown();
    std::process::exit(0)
}

/// Exit with failure (code 1).
#[no_mangle]
pub extern "C" fn bhc_exit_failure() -> ! {
    bhc_shutdown();
    std::process::exit(1)
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

    // Tests for exception handling
    #[test]
    fn test_bhc_evaluate() {
        let val = 42usize as *mut u8;
        let result = bhc_evaluate(val);
        assert_eq!(result, val);
    }

    #[test]
    fn test_bhc_catch_no_exception() {
        extern "C" fn action(_env: *mut u8) -> *mut u8 {
            42usize as *mut u8
        }
        extern "C" fn handler(_env: *mut u8, _exc: *mut u8) -> *mut u8 {
            99usize as *mut u8
        }
        let result = bhc_catch(action, ptr::null_mut(), handler, ptr::null_mut());
        assert_eq!(result as usize, 42);
    }

    #[test]
    fn test_bhc_catch_with_exception() {
        extern "C" fn action(_env: *mut u8) -> *mut u8 {
            bhc_throw(77usize as *mut u8)
        }
        extern "C" fn handler(_env: *mut u8, exc: *mut u8) -> *mut u8 {
            // Return the exception pointer as the result
            exc
        }
        let result = bhc_catch(action, ptr::null_mut(), handler, ptr::null_mut());
        assert_eq!(result as usize, 77);
    }

    #[test]
    fn test_bhc_mask_runs_action() {
        extern "C" fn action(_env: *mut u8) -> *mut u8 {
            123usize as *mut u8
        }
        let result = bhc_mask(action, ptr::null_mut());
        assert_eq!(result as usize, 123);
    }

    #[test]
    fn test_bhc_unmask_runs_action() {
        extern "C" fn action(_env: *mut u8) -> *mut u8 {
            456usize as *mut u8
        }
        let result = bhc_unmask(action, ptr::null_mut());
        assert_eq!(result as usize, 456);
    }

    #[test]
    fn test_bhc_finally_no_exception() {
        extern "C" fn action(_env: *mut u8) -> *mut u8 {
            42usize as *mut u8
        }
        extern "C" fn cleanup(_env: *mut u8) -> *mut u8 {
            // Cleanup runs but result is ignored
            99usize as *mut u8
        }
        let result = bhc_finally(action, ptr::null_mut(), cleanup, ptr::null_mut());
        assert_eq!(result as usize, 42);
    }

    #[test]
    fn test_bhc_finally_with_exception() {
        extern "C" fn action(_env: *mut u8) -> *mut u8 {
            bhc_throw(77usize as *mut u8)
        }
        extern "C" fn cleanup(_env: *mut u8) -> *mut u8 {
            ptr::null_mut()
        }
        // finally should re-throw; verify exception is pending
        let result = bhc_finally(action, ptr::null_mut(), cleanup, ptr::null_mut());
        assert_eq!(result, BHC_EXCEPTION_SENTINEL);
        // Exception should be pending in TLS
        let exc = BHC_EXCEPTION.with(|cell| cell.replace(None));
        assert_eq!(exc.unwrap() as usize, 77);
    }

    #[test]
    fn test_bhc_on_exception_no_exception() {
        extern "C" fn action(_env: *mut u8) -> *mut u8 {
            42usize as *mut u8
        }
        extern "C" fn handler(_env: *mut u8) -> *mut u8 {
            panic!("handler should not be called");
        }
        let result = bhc_on_exception(action, ptr::null_mut(), handler, ptr::null_mut());
        assert_eq!(result as usize, 42);
    }

    #[test]
    fn test_bhc_on_exception_with_exception() {
        extern "C" fn action(_env: *mut u8) -> *mut u8 {
            bhc_throw(77usize as *mut u8)
        }
        static mut HANDLER_CALLED: bool = false;
        extern "C" fn handler(_env: *mut u8) -> *mut u8 {
            unsafe { HANDLER_CALLED = true; }
            ptr::null_mut()
        }
        unsafe { HANDLER_CALLED = false; }
        let result = bhc_on_exception(action, ptr::null_mut(), handler, ptr::null_mut());
        assert_eq!(result, BHC_EXCEPTION_SENTINEL);
        assert!(unsafe { HANDLER_CALLED });
        // Exception should be pending in TLS (re-thrown)
        let exc = BHC_EXCEPTION.with(|cell| cell.replace(None));
        assert_eq!(exc.unwrap() as usize, 77);
    }

    #[test]
    fn test_bhc_bracket_no_exception() {
        static mut RELEASED: bool = false;
        extern "C" fn acquire(_env: *mut u8) -> *mut u8 {
            10usize as *mut u8
        }
        extern "C" fn release(_env: *mut u8, _resource: *mut u8) -> *mut u8 {
            unsafe { RELEASED = true; }
            ptr::null_mut()
        }
        extern "C" fn use_fn(_env: *mut u8, resource: *mut u8) -> *mut u8 {
            // Return resource + 32
            ((resource as usize) + 32) as *mut u8
        }
        unsafe { RELEASED = false; }
        let result = bhc_bracket(
            acquire, ptr::null_mut(),
            release, ptr::null_mut(),
            use_fn, ptr::null_mut(),
        );
        assert_eq!(result as usize, 42);
        assert!(unsafe { RELEASED });
    }

    #[test]
    fn test_bhc_bracket_use_throws() {
        static mut RELEASED: bool = false;
        extern "C" fn acquire(_env: *mut u8) -> *mut u8 {
            10usize as *mut u8
        }
        extern "C" fn release(_env: *mut u8, _resource: *mut u8) -> *mut u8 {
            unsafe { RELEASED = true; }
            ptr::null_mut()
        }
        extern "C" fn use_fn(_env: *mut u8, _resource: *mut u8) -> *mut u8 {
            bhc_throw(77usize as *mut u8)
        }
        unsafe { RELEASED = false; }
        let result = bhc_bracket(
            acquire, ptr::null_mut(),
            release, ptr::null_mut(),
            use_fn, ptr::null_mut(),
        );
        assert_eq!(result, BHC_EXCEPTION_SENTINEL);
        // Release must have been called even though use threw
        assert!(unsafe { RELEASED });
        // Exception should be pending in TLS
        let exc = BHC_EXCEPTION.with(|cell| cell.replace(None));
        assert_eq!(exc.unwrap() as usize, 77);
    }

    #[test]
    fn test_show_char() {
        let s = bhc_show_char('a' as u32);
        let cstr = unsafe { std::ffi::CStr::from_ptr(s) };
        assert_eq!(cstr.to_str().unwrap(), "'a'");
        unsafe { bhc_free_string(s) };
    }

    // Tests for handle-based IO
    #[test]
    fn test_std_handle_sentinels() {
        let stdin = bhc_stdin();
        let stdout = bhc_stdout();
        let stderr = bhc_stderr();
        assert_eq!(stdin as usize, HANDLE_STDIN);
        assert_eq!(stdout as usize, HANDLE_STDOUT);
        assert_eq!(stderr as usize, HANDLE_STDERR);
        assert!(is_sentinel(stdin));
        assert!(is_sentinel(stdout));
        assert!(is_sentinel(stderr));
    }

    #[test]
    fn test_handle_properties_std() {
        let stdin = bhc_stdin();
        let stdout = bhc_stdout();
        assert_eq!(bhc_hIsOpen(stdin), 1);
        assert_eq!(bhc_hIsClosed(stdin), 0);
        assert_eq!(bhc_hIsReadable(stdin), 1);
        assert_eq!(bhc_hIsWritable(stdin), 0);
        assert_eq!(bhc_hIsSeekable(stdin), 0);
        assert_eq!(bhc_hIsOpen(stdout), 1);
        assert_eq!(bhc_hIsReadable(stdout), 0);
        assert_eq!(bhc_hIsWritable(stdout), 1);
    }

    #[test]
    fn test_open_close_file() {
        use std::io::Write;
        // Create a temp file
        let tmp = std::env::temp_dir().join("bhc_test_open_close.txt");
        std::fs::write(&tmp, "hello").unwrap();
        let path = CString::new(tmp.to_str().unwrap()).unwrap();

        let handle = unsafe { bhc_open_file(path.as_ptr(), 0) }; // ReadMode
        assert!(!handle.is_null());
        assert!(!is_sentinel(handle));
        assert_eq!(bhc_hIsOpen(handle), 1);
        assert_eq!(bhc_hIsReadable(handle), 1);
        assert_eq!(bhc_hIsWritable(handle), 0);

        unsafe { bhc_close_handle(handle) };
        assert_eq!(bhc_hIsClosed(handle), 1);
        assert_eq!(bhc_hIsOpen(handle), 0);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_file_read_write() {
        let tmp = std::env::temp_dir().join("bhc_test_rw.txt");
        let path = CString::new(tmp.to_str().unwrap()).unwrap();

        // Write
        let wh = unsafe { bhc_open_file(path.as_ptr(), 1) }; // WriteMode
        assert!(!wh.is_null());
        let content = CString::new("line1\nline2\n").unwrap();
        unsafe { bhc_hPutStr(wh, content.as_ptr()) };
        unsafe { bhc_hFlush(wh) };
        unsafe { bhc_close_handle(wh) };

        // Read line by line
        let rh = unsafe { bhc_open_file(path.as_ptr(), 0) }; // ReadMode
        assert!(!rh.is_null());
        let line1 = unsafe { bhc_hGetLine(rh) };
        assert!(!line1.is_null());
        let l1 = unsafe { CStr::from_ptr(line1) }.to_str().unwrap();
        assert_eq!(l1, "line1");
        unsafe { bhc_free_string(line1) };

        let line2 = unsafe { bhc_hGetLine(rh) };
        assert!(!line2.is_null());
        let l2 = unsafe { CStr::from_ptr(line2) }.to_str().unwrap();
        assert_eq!(l2, "line2");
        unsafe { bhc_free_string(line2) };

        unsafe { bhc_close_handle(rh) };
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_exists_and_is_directory() {
        let tmp = std::env::temp_dir().join("bhc_test_exists.txt");
        std::fs::write(&tmp, "x").unwrap();
        let path = CString::new(tmp.to_str().unwrap()).unwrap();

        assert_eq!(unsafe { bhc_exists(path.as_ptr()) }, 1);
        assert_eq!(unsafe { bhc_is_directory(path.as_ptr()) }, 0);

        let dir_path = CString::new(std::env::temp_dir().to_str().unwrap()).unwrap();
        assert_eq!(unsafe { bhc_is_directory(dir_path.as_ptr()) }, 1);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_get_prog_name() {
        let name = bhc_get_prog_name();
        assert!(!name.is_null());
        unsafe { bhc_free_string(name) };
    }

    #[test]
    fn test_get_current_directory() {
        let dir = bhc_get_current_directory();
        assert!(!dir.is_null());
        unsafe { bhc_free_string(dir) };
    }

    #[test]
    fn test_get_args() {
        let args = bhc_get_args();
        assert!(!args.is_null());
        // At minimum there should be the program name
        let tag = unsafe { *(args as *const i64) };
        assert!(tag == 0 || tag == 1); // nil or cons
    }

    #[test]
    fn test_seek_tell() {
        let tmp = std::env::temp_dir().join("bhc_test_seek.txt");
        std::fs::write(&tmp, "abcdef").unwrap();
        let path = CString::new(tmp.to_str().unwrap()).unwrap();

        let h = unsafe { bhc_open_file(path.as_ptr(), 0) }; // ReadMode
        assert!(!h.is_null());

        // Tell should start at 0
        assert_eq!(unsafe { bhc_hTell(h) }, 0);

        // Seek to position 3
        unsafe { bhc_hSeek(h, 0, 3) }; // AbsoluteSeek
        assert_eq!(unsafe { bhc_hTell(h) }, 3);

        // Read char at position 3 should be 'd'
        let ch = unsafe { bhc_hGetChar(h) };
        assert_eq!(ch, b'd' as c_int);

        unsafe { bhc_close_handle(h) };
        let _ = std::fs::remove_file(&tmp);
    }
}
