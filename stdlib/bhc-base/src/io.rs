//! I/O operations
//!
//! File and console I/O primitives.

use std::ffi::{CStr, CString};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::raw::c_char;

use bhc_prelude::BhcError;

/// Read entire file contents
#[no_mangle]
pub extern "C" fn bhc_io_read_file(
    path: *const c_char,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if path.is_null() || out_ptr.is_null() || out_len.is_null() {
        return BhcError::NullPointer as i32;
    }

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return BhcError::InvalidArgument as i32,
        }
    };

    let mut file = match File::open(path_str) {
        Ok(f) => f,
        Err(_) => return -1,
    };

    let mut contents = Vec::new();
    if file.read_to_end(&mut contents).is_err() {
        return -2;
    }

    let len = contents.len();
    let ptr = contents.as_mut_ptr();
    std::mem::forget(contents);

    unsafe {
        *out_ptr = ptr;
        *out_len = len;
    }

    0
}

/// Write string to file
#[no_mangle]
pub extern "C" fn bhc_io_write_file(
    path: *const c_char,
    data: *const u8,
    len: usize,
) -> i32 {
    if path.is_null() || data.is_null() {
        return BhcError::NullPointer as i32;
    }

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return BhcError::InvalidArgument as i32,
        }
    };

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };

    let mut file = match File::create(path_str) {
        Ok(f) => f,
        Err(_) => return -1,
    };

    match file.write_all(data_slice) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

/// Append string to file
#[no_mangle]
pub extern "C" fn bhc_io_append_file(
    path: *const c_char,
    data: *const u8,
    len: usize,
) -> i32 {
    if path.is_null() || data.is_null() {
        return BhcError::NullPointer as i32;
    }

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return BhcError::InvalidArgument as i32,
        }
    };

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };

    let mut file = match OpenOptions::new().create(true).append(true).open(path_str) {
        Ok(f) => f,
        Err(_) => return -1,
    };

    match file.write_all(data_slice) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

/// Free buffer allocated by read_file
#[no_mangle]
pub extern "C" fn bhc_io_free_buffer(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            drop(Vec::from_raw_parts(ptr, len, len));
        }
    }
}

/// Print string to stdout
#[no_mangle]
pub extern "C" fn bhc_io_put_str(data: *const u8, len: usize) -> i32 {
    if data.is_null() && len > 0 {
        return BhcError::NullPointer as i32;
    }

    let data_slice = if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };

    match std::io::stdout().write_all(data_slice) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Print character to stdout
#[no_mangle]
pub extern "C" fn bhc_io_put_char(c: u32) -> i32 {
    match char::from_u32(c) {
        Some(ch) => {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            bhc_io_put_str(s.as_ptr(), s.len())
        }
        None => BhcError::InvalidArgument as i32,
    }
}

/// Get character from stdin
#[no_mangle]
pub extern "C" fn bhc_io_get_char() -> i32 {
    let mut buf = [0u8; 1];
    match std::io::stdin().read_exact(&mut buf) {
        Ok(_) => buf[0] as i32,
        Err(_) => -1,
    }
}

/// Flush stdout
#[no_mangle]
pub extern "C" fn bhc_io_flush_stdout() -> i32 {
    match std::io::stdout().flush() {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_put_char() {
        // Just test it doesn't crash
        assert_eq!(bhc_io_put_char('a' as u32), 0);
    }
}
