//! WASI (WebAssembly System Interface) support.
//!
//! This module provides WASI imports and runtime functions needed for
//! standalone WASM execution with wasmtime, wasmer, or other WASI runtimes.

use crate::codegen::{
    WasmExport, WasmExportKind, WasmFunc, WasmFuncType, WasmGlobal, WasmImport, WasmImportKind,
};
use crate::{WasmInstr, WasmType};

/// Generate the standard WASI imports needed for basic I/O.
///
/// Returns a list of imports for:
/// - `fd_write`: Write to a file descriptor (used for stdout/stderr)
/// - `proc_exit`: Exit the process with a status code
/// - `fd_read`: Read from a file descriptor
/// - `args_sizes_get`: Get command line argument sizes
/// - `args_get`: Get command line arguments
/// - `environ_sizes_get`: Get environment variable sizes
/// - `environ_get`: Get environment variables
pub fn generate_wasi_imports() -> Vec<WasmImport> {
    vec![
        // fd_write(fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) -> i32
        // Writes data to a file descriptor
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "fd_write".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(
                vec![WasmType::I32, WasmType::I32, WasmType::I32, WasmType::I32],
                vec![WasmType::I32],
            )),
        },
        // proc_exit(code: i32)
        // Terminates the process
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "proc_exit".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(vec![WasmType::I32], vec![])),
        },
        // fd_read(fd: i32, iovs: i32, iovs_len: i32, nread: i32) -> i32
        // Reads data from a file descriptor
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "fd_read".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(
                vec![WasmType::I32, WasmType::I32, WasmType::I32, WasmType::I32],
                vec![WasmType::I32],
            )),
        },
        // args_sizes_get(argc: i32, argv_buf_size: i32) -> i32
        // Returns the number of arguments and total buffer size needed
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "args_sizes_get".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(
                vec![WasmType::I32, WasmType::I32],
                vec![WasmType::I32],
            )),
        },
        // args_get(argv: i32, argv_buf: i32) -> i32
        // Fills argv with pointers to argument strings, argv_buf with the strings
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "args_get".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(
                vec![WasmType::I32, WasmType::I32],
                vec![WasmType::I32],
            )),
        },
        // environ_sizes_get(environc: i32, environ_buf_size: i32) -> i32
        // Returns the number of environment variables and total buffer size needed
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "environ_sizes_get".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(
                vec![WasmType::I32, WasmType::I32],
                vec![WasmType::I32],
            )),
        },
        // environ_get(environ: i32, environ_buf: i32) -> i32
        // Fills environ with pointers to env var strings, environ_buf with the strings
        WasmImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "environ_get".to_string(),
            kind: WasmImportKind::Func(WasmFuncType::new(
                vec![WasmType::I32, WasmType::I32],
                vec![WasmType::I32],
            )),
        },
    ]
}

/// Index of fd_write in imports (assuming it's the first function import).
pub const FD_WRITE_IDX: u32 = 0;
/// Index of proc_exit in imports.
pub const PROC_EXIT_IDX: u32 = 1;
/// Index of fd_read in imports.
pub const FD_READ_IDX: u32 = 2;
/// Index of args_sizes_get in imports.
pub const ARGS_SIZES_GET_IDX: u32 = 3;
/// Index of args_get in imports.
pub const ARGS_GET_IDX: u32 = 4;
/// Index of environ_sizes_get in imports.
pub const ENVIRON_SIZES_GET_IDX: u32 = 5;
/// Index of environ_get in imports.
pub const ENVIRON_GET_IDX: u32 = 6;

/// Stdout file descriptor.
pub const STDOUT_FD: i32 = 1;
/// Stderr file descriptor.
pub const STDERR_FD: i32 = 2;

/// Generate the heap pointer global variable.
///
/// This global tracks the current end of the heap for allocation.
/// Initial value points after static data (e.g., at 64KB = 65536).
pub fn generate_heap_pointer_global() -> WasmGlobal {
    WasmGlobal {
        name: Some("heap_ptr".to_string()),
        ty: WasmType::I32,
        mutable: true,
        init: WasmInstr::I32Const(65536), // Start heap at 64KB
    }
}

/// Generate a simple bump allocator function.
///
/// This implements: `alloc(size: i32) -> i32`
/// Returns a pointer to allocated memory by bumping the heap pointer.
pub fn generate_alloc_function(heap_ptr_global: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![WasmType::I32], vec![WasmType::I32]));
    func.name = Some("alloc".to_string());
    func.exported = true;
    func.export_name = Some("alloc".to_string());

    // Get current heap pointer
    func.emit(WasmInstr::GlobalGet(heap_ptr_global));

    // Duplicate for return value (local.tee pattern)
    let result_local = func.add_local(WasmType::I32);
    func.emit(WasmInstr::LocalTee(result_local));

    // Add size to heap pointer
    func.emit(WasmInstr::LocalGet(0)); // size parameter
    func.emit(WasmInstr::I32Add);

    // Align to 8 bytes: (ptr + 7) & ~7
    func.emit(WasmInstr::I32Const(7));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Const(-8)); // ~7 in two's complement
    func.emit(WasmInstr::I32And);

    // Store new heap pointer
    func.emit(WasmInstr::GlobalSet(heap_ptr_global));

    // Return original heap pointer
    func.emit(WasmInstr::LocalGet(result_local));
    func.emit(WasmInstr::End);

    func
}

/// Generate the print_i32 function for debugging.
///
/// Prints an i32 value to stdout using WASI fd_write.
/// Uses memory at a fixed offset for the iovec structure.
pub fn generate_print_i32(fd_write_idx: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![WasmType::I32], vec![]));
    func.name = Some("print_i32".to_string());
    func.exported = true;

    // Convert i32 to decimal string in memory
    // Use fixed memory locations:
    // - 0-16: scratch space for number string
    // - 16-24: iovec structure

    let num_local = func.add_local(WasmType::I32); // number to print
    let ptr_local = func.add_local(WasmType::I32); // string pointer
    let len_local = func.add_local(WasmType::I32); // string length
    let digit_local = func.add_local(WasmType::I32);
    let is_neg_local = func.add_local(WasmType::I32);

    // Store parameter in local
    func.emit(WasmInstr::LocalGet(0));
    func.emit(WasmInstr::LocalSet(num_local));

    // Start at end of buffer (position 15)
    func.emit(WasmInstr::I32Const(15));
    func.emit(WasmInstr::LocalSet(ptr_local));

    // Initialize length to 0
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::LocalSet(len_local));

    // Check if negative
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32LtS);
    func.emit(WasmInstr::LocalSet(is_neg_local));

    // If negative, negate
    func.emit(WasmInstr::LocalGet(is_neg_local));
    func.emit(WasmInstr::If(None));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Sub);
    func.emit(WasmInstr::LocalSet(num_local));
    func.emit(WasmInstr::End);

    // Handle zero case
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Eqz);
    func.emit(WasmInstr::If(None));
    // Store '0' at position 15
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Const(48)); // '0'
    func.emit(WasmInstr::I32Store8(0, 0));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::LocalSet(len_local));
    func.emit(WasmInstr::Else);

    // Convert digits loop
    func.emit(WasmInstr::Block(None)); // break target
    func.emit(WasmInstr::Loop(None)); // continue target

    // Get digit: num % 10
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Const(10));
    func.emit(WasmInstr::I32RemU);
    func.emit(WasmInstr::I32Const(48)); // '0'
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(digit_local));

    // Store digit (single byte)
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::LocalGet(digit_local));
    func.emit(WasmInstr::I32Store8(0, 0));

    // num = num / 10
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Const(10));
    func.emit(WasmInstr::I32DivU);
    func.emit(WasmInstr::LocalSet(num_local));

    // ptr--
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Sub);
    func.emit(WasmInstr::LocalSet(ptr_local));

    // len++
    func.emit(WasmInstr::LocalGet(len_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(len_local));

    // if num > 0, continue
    func.emit(WasmInstr::LocalGet(num_local));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32GtU);
    func.emit(WasmInstr::BrIf(0)); // branch to loop

    func.emit(WasmInstr::End); // end loop
    func.emit(WasmInstr::End); // end block
    func.emit(WasmInstr::End); // end if (not zero)

    // Adjust ptr to point to start of string
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(ptr_local));

    // If negative, add '-' prefix
    func.emit(WasmInstr::LocalGet(is_neg_local));
    func.emit(WasmInstr::If(None));
    // Decrement ptr and store '-'
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Sub);
    func.emit(WasmInstr::LocalTee(ptr_local));
    func.emit(WasmInstr::I32Const(45)); // '-'
    func.emit(WasmInstr::I32Store8(0, 0));
    func.emit(WasmInstr::LocalGet(len_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(len_local));
    func.emit(WasmInstr::End);

    // Set up iovec at memory offset 16
    // iovec.buf = ptr
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::LocalGet(ptr_local));
    func.emit(WasmInstr::I32Store(4, 0));

    // iovec.len = len
    func.emit(WasmInstr::I32Const(20));
    func.emit(WasmInstr::LocalGet(len_local));
    func.emit(WasmInstr::I32Store(4, 0));

    // Call fd_write(stdout, iovec_ptr, 1, nwritten_ptr)
    func.emit(WasmInstr::I32Const(STDOUT_FD)); // fd = stdout
    func.emit(WasmInstr::I32Const(16)); // iovs = 16
    func.emit(WasmInstr::I32Const(1)); // iovs_len = 1
    func.emit(WasmInstr::I32Const(24)); // nwritten = 24
    func.emit(WasmInstr::Call(fd_write_idx));
    func.emit(WasmInstr::Drop); // ignore return value

    // Print newline
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32Const(10)); // '\n'
    func.emit(WasmInstr::I32Store8(0, 0));
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::I32Store(4, 0));
    func.emit(WasmInstr::I32Const(20));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Store(4, 0));
    func.emit(WasmInstr::I32Const(STDOUT_FD));
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Const(24));
    func.emit(WasmInstr::Call(fd_write_idx));
    func.emit(WasmInstr::Drop);

    func.emit(WasmInstr::End);

    func
}

/// Generate a print_str function for printing string literals.
///
/// Takes a pointer and length, prints to stdout.
pub fn generate_print_str(fd_write_idx: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(
        vec![WasmType::I32, WasmType::I32], // ptr, len
        vec![],
    ));
    func.name = Some("print_str".to_string());
    func.exported = true;

    // Set up iovec at memory offset 16
    // iovec.buf = param 0 (ptr)
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::LocalGet(0)); // ptr
    func.emit(WasmInstr::I32Store(4, 0));

    // iovec.len = param 1 (len)
    func.emit(WasmInstr::I32Const(20));
    func.emit(WasmInstr::LocalGet(1)); // len
    func.emit(WasmInstr::I32Store(4, 0));

    // Call fd_write(stdout, iovec_ptr, 1, nwritten_ptr)
    func.emit(WasmInstr::I32Const(STDOUT_FD));
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Const(24));
    func.emit(WasmInstr::Call(fd_write_idx));
    func.emit(WasmInstr::Drop);

    func.emit(WasmInstr::End);

    func
}

/// Generate a print_str_ln function that prints a string followed by a newline.
///
/// Takes a pointer and length, prints the string then a `\n` to stdout.
/// The newline byte is stored in a data segment at `newline_offset`.
pub fn generate_print_str_ln(fd_write_idx: u32, newline_offset: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(
        vec![WasmType::I32, WasmType::I32], // ptr, len
        vec![],
    ));
    func.name = Some("print_str_ln".to_string());
    func.exported = true;

    // --- Print the string ---
    // Set up iovec at memory offset 16: iovec.buf = param 0 (ptr)
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::LocalGet(0)); // ptr
    func.emit(WasmInstr::I32Store(4, 0));

    // iovec.len = param 1 (len)
    func.emit(WasmInstr::I32Const(20));
    func.emit(WasmInstr::LocalGet(1)); // len
    func.emit(WasmInstr::I32Store(4, 0));

    // Call fd_write(stdout, iovec_ptr, 1, nwritten_ptr)
    func.emit(WasmInstr::I32Const(STDOUT_FD));
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Const(24));
    func.emit(WasmInstr::Call(fd_write_idx));
    func.emit(WasmInstr::Drop);

    // --- Print newline ---
    // Set up iovec: iovec.buf = newline_offset
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::I32Const(newline_offset as i32));
    func.emit(WasmInstr::I32Store(4, 0));

    // iovec.len = 1
    func.emit(WasmInstr::I32Const(20));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Store(4, 0));

    // Call fd_write(stdout, iovec_ptr, 1, nwritten_ptr)
    func.emit(WasmInstr::I32Const(STDOUT_FD));
    func.emit(WasmInstr::I32Const(16));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Const(24));
    func.emit(WasmInstr::Call(fd_write_idx));
    func.emit(WasmInstr::Drop);

    func.emit(WasmInstr::End);

    func
}

/// Offset where the newline byte is stored in the data segment.
pub const NEWLINE_DATA_OFFSET: u32 = 1020;

/// Generate the _start function that calls main.
///
/// This is the WASI entry point. It calls the Haskell main function
/// and then calls proc_exit with 0.
pub fn generate_start_function(main_func_idx: u32, proc_exit_idx: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![]));
    func.name = Some("_start".to_string());
    func.exported = true;
    func.export_name = Some("_start".to_string());

    // Call main
    func.emit(WasmInstr::Call(main_func_idx));

    // Drop result if main returns something
    func.emit(WasmInstr::Drop);

    // Call proc_exit(0)
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::Call(proc_exit_idx));

    func.emit(WasmInstr::End);

    func
}

/// Generate a simple main that just returns 0.
///
/// This is a placeholder main function.
pub fn generate_placeholder_main() -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![WasmType::I32]));
    func.name = Some("main".to_string());

    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::End);

    func
}

/// Generate a main that prints "Hello, World!".
///
/// This is useful for testing the WASM pipeline.
pub fn generate_hello_main(print_str_idx: u32, string_offset: u32, string_len: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![WasmType::I32]));
    func.name = Some("main".to_string());

    // Call print_str(string_offset, string_len)
    func.emit(WasmInstr::I32Const(string_offset as i32));
    func.emit(WasmInstr::I32Const(string_len as i32));
    func.emit(WasmInstr::Call(print_str_idx));

    // Return 0
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::End);

    func
}

/// The "Hello, World!\n" string bytes.
pub const HELLO_WORLD_STRING: &[u8] = b"Hello, World!\n";

/// Offset where the hello world string is stored in memory.
pub const HELLO_WORLD_OFFSET: u32 = 1024;

/// Memory offset for storing argc (argument count).
pub const ARGC_OFFSET: u32 = 256;
/// Memory offset for storing argv buffer size.
pub const ARGV_BUF_SIZE_OFFSET: u32 = 260;
/// Memory offset for storing argv pointers array.
pub const ARGV_OFFSET: u32 = 512;
/// Memory offset for storing argv string buffer.
pub const ARGV_BUF_OFFSET: u32 = 2048;

/// Memory offset for storing environc (environment variable count).
pub const ENVIRONC_OFFSET: u32 = 264;
/// Memory offset for storing environ buffer size.
pub const ENVIRON_BUF_SIZE_OFFSET: u32 = 268;
/// Memory offset for storing environ pointers array.
pub const ENVIRON_OFFSET: u32 = 768;
/// Memory offset for storing environ string buffer.
pub const ENVIRON_BUF_OFFSET: u32 = 4096;

/// Generate a function to initialize command-line arguments.
///
/// This calls args_sizes_get and args_get to populate memory with argv.
/// After calling, argc is at ARGC_OFFSET and argv pointers start at ARGV_OFFSET.
pub fn generate_init_args(args_sizes_get_idx: u32, args_get_idx: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![]));
    func.name = Some("__init_args".to_string());

    // Call args_sizes_get(argc_ptr, argv_buf_size_ptr)
    func.emit(WasmInstr::I32Const(ARGC_OFFSET as i32));
    func.emit(WasmInstr::I32Const(ARGV_BUF_SIZE_OFFSET as i32));
    func.emit(WasmInstr::Call(args_sizes_get_idx));
    func.emit(WasmInstr::Drop); // ignore errno

    // Call args_get(argv_ptr, argv_buf_ptr)
    func.emit(WasmInstr::I32Const(ARGV_OFFSET as i32));
    func.emit(WasmInstr::I32Const(ARGV_BUF_OFFSET as i32));
    func.emit(WasmInstr::Call(args_get_idx));
    func.emit(WasmInstr::Drop); // ignore errno

    func.emit(WasmInstr::End);
    func
}

/// Generate a function to get the argument count (argc).
///
/// Returns the number of command-line arguments.
pub fn generate_get_argc() -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![WasmType::I32]));
    func.name = Some("get_argc".to_string());
    func.exported = true;

    // Load argc from memory
    func.emit(WasmInstr::I32Const(ARGC_OFFSET as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::End);

    func
}

/// Generate a function to get an argument by index.
///
/// get_argv(index: i32) -> i32 (pointer to null-terminated string)
pub fn generate_get_argv() -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![WasmType::I32], vec![WasmType::I32]));
    func.name = Some("get_argv".to_string());
    func.exported = true;

    // Load argv[index]: argv_base + index * 4
    func.emit(WasmInstr::I32Const(ARGV_OFFSET as i32));
    func.emit(WasmInstr::LocalGet(0)); // index
    func.emit(WasmInstr::I32Const(4));
    func.emit(WasmInstr::I32Mul);
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Load(4, 0)); // load pointer
    func.emit(WasmInstr::End);

    func
}

/// Generate a function to initialize environment variables.
///
/// This calls environ_sizes_get and environ_get to populate memory.
/// After calling, environc is at ENVIRONC_OFFSET and environ pointers start at ENVIRON_OFFSET.
pub fn generate_init_environ(environ_sizes_get_idx: u32, environ_get_idx: u32) -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![]));
    func.name = Some("__init_environ".to_string());

    // Call environ_sizes_get(environc_ptr, environ_buf_size_ptr)
    func.emit(WasmInstr::I32Const(ENVIRONC_OFFSET as i32));
    func.emit(WasmInstr::I32Const(ENVIRON_BUF_SIZE_OFFSET as i32));
    func.emit(WasmInstr::Call(environ_sizes_get_idx));
    func.emit(WasmInstr::Drop); // ignore errno

    // Call environ_get(environ_ptr, environ_buf_ptr)
    func.emit(WasmInstr::I32Const(ENVIRON_OFFSET as i32));
    func.emit(WasmInstr::I32Const(ENVIRON_BUF_OFFSET as i32));
    func.emit(WasmInstr::Call(environ_get_idx));
    func.emit(WasmInstr::Drop); // ignore errno

    func.emit(WasmInstr::End);
    func
}

/// Generate a function to get the environment variable count.
///
/// Returns the number of environment variables.
pub fn generate_get_environc() -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![], vec![WasmType::I32]));
    func.name = Some("get_environc".to_string());
    func.exported = true;

    // Load environc from memory
    func.emit(WasmInstr::I32Const(ENVIRONC_OFFSET as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::End);

    func
}

/// Generate a function to get an environment variable by index.
///
/// get_environ(index: i32) -> i32 (pointer to null-terminated "KEY=VALUE" string)
pub fn generate_get_environ() -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(vec![WasmType::I32], vec![WasmType::I32]));
    func.name = Some("get_environ".to_string());
    func.exported = true;

    // Load environ[index]: environ_base + index * 4
    func.emit(WasmInstr::I32Const(ENVIRON_OFFSET as i32));
    func.emit(WasmInstr::LocalGet(0)); // index
    func.emit(WasmInstr::I32Const(4));
    func.emit(WasmInstr::I32Mul);
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Load(4, 0)); // load pointer
    func.emit(WasmInstr::End);

    func
}

/// Generate a function to look up an environment variable by name.
///
/// getenv(name_ptr: i32, name_len: i32) -> i32 (pointer to value or 0 if not found)
/// Note: This is a simple linear search. For better performance, use a hash map.
pub fn generate_getenv() -> WasmFunc {
    let mut func = WasmFunc::new(WasmFuncType::new(
        vec![WasmType::I32, WasmType::I32], // name_ptr, name_len
        vec![WasmType::I32],                // value_ptr or 0
    ));
    func.name = Some("getenv".to_string());
    func.exported = true;

    let i_local = func.add_local(WasmType::I32); // loop counter
    let count_local = func.add_local(WasmType::I32); // environc
    let env_ptr_local = func.add_local(WasmType::I32); // current environ string ptr
    let j_local = func.add_local(WasmType::I32); // char comparison index
    let match_local = func.add_local(WasmType::I32); // match flag

    // Load environc
    func.emit(WasmInstr::I32Const(ENVIRONC_OFFSET as i32));
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::LocalSet(count_local));

    // i = 0
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::LocalSet(i_local));

    // Outer loop: iterate through environ entries
    func.emit(WasmInstr::Block(None)); // break target for outer loop
    func.emit(WasmInstr::Loop(None)); // continue target for outer loop

    // if i >= count, break
    func.emit(WasmInstr::LocalGet(i_local));
    func.emit(WasmInstr::LocalGet(count_local));
    func.emit(WasmInstr::I32GeU);
    func.emit(WasmInstr::BrIf(1)); // break outer block

    // Get environ[i] pointer
    func.emit(WasmInstr::I32Const(ENVIRON_OFFSET as i32));
    func.emit(WasmInstr::LocalGet(i_local));
    func.emit(WasmInstr::I32Const(4));
    func.emit(WasmInstr::I32Mul);
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Load(4, 0));
    func.emit(WasmInstr::LocalSet(env_ptr_local));

    // Compare name with env entry (up to name_len chars, then check for '=')
    // j = 0, match = 1
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::LocalSet(j_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::LocalSet(match_local));

    // Inner loop: compare characters
    func.emit(WasmInstr::Block(None)); // break for inner loop
    func.emit(WasmInstr::Loop(None));

    // if j >= name_len, break (done comparing)
    func.emit(WasmInstr::LocalGet(j_local));
    func.emit(WasmInstr::LocalGet(1)); // name_len
    func.emit(WasmInstr::I32GeU);
    func.emit(WasmInstr::BrIf(1));

    // if name[j] != env[j], set match = 0 and break
    func.emit(WasmInstr::LocalGet(0)); // name_ptr
    func.emit(WasmInstr::LocalGet(j_local));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Load8U(0, 0)); // name[j]

    func.emit(WasmInstr::LocalGet(env_ptr_local));
    func.emit(WasmInstr::LocalGet(j_local));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Load8U(0, 0)); // env[j]

    func.emit(WasmInstr::I32Ne);
    func.emit(WasmInstr::If(None));
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::LocalSet(match_local));
    func.emit(WasmInstr::Br(2)); // break inner block
    func.emit(WasmInstr::End);

    // j++
    func.emit(WasmInstr::LocalGet(j_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(j_local));

    func.emit(WasmInstr::Br(0)); // continue inner loop
    func.emit(WasmInstr::End); // end inner loop
    func.emit(WasmInstr::End); // end inner block

    // Check if match and env[name_len] == '='
    func.emit(WasmInstr::LocalGet(match_local));
    func.emit(WasmInstr::If(None));

    func.emit(WasmInstr::LocalGet(env_ptr_local));
    func.emit(WasmInstr::LocalGet(1)); // name_len
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Load8U(0, 0));
    func.emit(WasmInstr::I32Const(61)); // '='
    func.emit(WasmInstr::I32Eq);
    func.emit(WasmInstr::If(None));

    // Found! Return pointer to value (after '=')
    func.emit(WasmInstr::LocalGet(env_ptr_local));
    func.emit(WasmInstr::LocalGet(1)); // name_len
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::I32Const(1)); // skip '='
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::Return);

    func.emit(WasmInstr::End); // end '=' check
    func.emit(WasmInstr::End); // end match check

    // i++
    func.emit(WasmInstr::LocalGet(i_local));
    func.emit(WasmInstr::I32Const(1));
    func.emit(WasmInstr::I32Add);
    func.emit(WasmInstr::LocalSet(i_local));

    func.emit(WasmInstr::Br(0)); // continue outer loop
    func.emit(WasmInstr::End); // end outer loop
    func.emit(WasmInstr::End); // end outer block

    // Not found, return 0
    func.emit(WasmInstr::I32Const(0));
    func.emit(WasmInstr::End);

    func
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_wasi_imports() {
        let imports = generate_wasi_imports();
        assert_eq!(imports.len(), 7);
        assert_eq!(imports[0].name, "fd_write");
        assert_eq!(imports[1].name, "proc_exit");
        assert_eq!(imports[2].name, "fd_read");
        assert_eq!(imports[3].name, "args_sizes_get");
        assert_eq!(imports[4].name, "args_get");
        assert_eq!(imports[5].name, "environ_sizes_get");
        assert_eq!(imports[6].name, "environ_get");
    }

    #[test]
    fn test_generate_alloc() {
        let func = generate_alloc_function(0);
        assert_eq!(func.name.as_deref(), Some("alloc"));
        assert!(func.exported);
        assert!(!func.body.is_empty());
    }

    #[test]
    fn test_generate_print_i32() {
        let func = generate_print_i32(0);
        assert_eq!(func.name.as_deref(), Some("print_i32"));
        assert!(func.exported);
    }

    #[test]
    fn test_generate_start() {
        let func = generate_start_function(5, 1);
        assert_eq!(func.name.as_deref(), Some("_start"));
        assert!(func.exported);
        // Should contain call to main and proc_exit
        let calls: Vec<_> = func
            .body
            .iter()
            .filter(|i| matches!(i, WasmInstr::Call(_)))
            .collect();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_generate_init_args() {
        let func = generate_init_args(ARGS_SIZES_GET_IDX, ARGS_GET_IDX);
        assert_eq!(func.name.as_deref(), Some("__init_args"));
        // Should contain 2 calls (args_sizes_get, args_get)
        let calls: Vec<_> = func
            .body
            .iter()
            .filter(|i| matches!(i, WasmInstr::Call(_)))
            .collect();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_generate_get_argc() {
        let func = generate_get_argc();
        assert_eq!(func.name.as_deref(), Some("get_argc"));
        assert!(func.exported);
    }

    #[test]
    fn test_generate_get_argv() {
        let func = generate_get_argv();
        assert_eq!(func.name.as_deref(), Some("get_argv"));
        assert!(func.exported);
    }

    #[test]
    fn test_generate_init_environ() {
        let func = generate_init_environ(ENVIRON_SIZES_GET_IDX, ENVIRON_GET_IDX);
        assert_eq!(func.name.as_deref(), Some("__init_environ"));
        // Should contain 2 calls (environ_sizes_get, environ_get)
        let calls: Vec<_> = func
            .body
            .iter()
            .filter(|i| matches!(i, WasmInstr::Call(_)))
            .collect();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_generate_get_environc() {
        let func = generate_get_environc();
        assert_eq!(func.name.as_deref(), Some("get_environc"));
        assert!(func.exported);
    }

    #[test]
    fn test_generate_get_environ() {
        let func = generate_get_environ();
        assert_eq!(func.name.as_deref(), Some("get_environ"));
        assert!(func.exported);
    }

    #[test]
    fn test_generate_getenv() {
        let func = generate_getenv();
        assert_eq!(func.name.as_deref(), Some("getenv"));
        assert!(func.exported);
        // Function should have proper signature (2 params, 1 result)
        assert_eq!(func.ty.params.len(), 2);
        assert_eq!(func.ty.results.len(), 1);
    }
}
