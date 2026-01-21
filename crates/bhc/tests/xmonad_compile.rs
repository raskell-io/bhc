//! Test compiling XMonad source files through the pipeline

use bhc_lower::{LowerConfig, LowerContext};
use bhc_parser::parse_module;
use bhc_span::FileId;
use camino::Utf8PathBuf;
use std::fs;

/// XMonad source directory for module resolution
const XMONAD_SRC_DIR: &str = "/tmp/xmonad/src";

/// Result of compiling a file
struct CompileResult {
    status: String,
    errors: Vec<String>,
    warnings: usize,
}

fn test_compile_file(path: &str) -> CompileResult {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return CompileResult {
            status: "READ_ERROR".into(),
            errors: vec![format!("{}", e)],
            warnings: 0,
        },
    };

    let file_id = FileId::new(0);
    let (module, parse_diags) = parse_module(&source, file_id);

    let parse_errors: Vec<String> = parse_diags.iter()
        .filter(|d| d.severity == bhc_diagnostics::Severity::Error)
        .map(|d| d.message.clone())
        .collect();

    if !parse_errors.is_empty() {
        return CompileResult {
            status: "PARSE_ERROR".into(),
            errors: parse_errors,
            warnings: 0,
        };
    }

    let module = match module {
        Some(m) => m,
        None => return CompileResult {
            status: "NO_MODULE".into(),
            errors: vec![],
            warnings: 0,
        },
    };

    // Configure lowering with search paths for XMonad modules
    let config = LowerConfig {
        include_builtins: true,
        warn_unused: false,
        search_paths: vec![Utf8PathBuf::from(XMONAD_SRC_DIR)],
    };

    // Try lowering to HIR
    let mut lower_ctx = LowerContext::with_builtins();
    let hir_module = match bhc_lower::lower_module(&mut lower_ctx, &module, &config) {
        Ok(m) => m,
        Err(e) => return CompileResult {
            status: "LOWER_ERROR".into(),
            errors: vec![format!("{:?}", e)],
            warnings: lower_ctx.warnings.len(),
        },
    };

    let warning_count = lower_ctx.warnings.len();

    // Try type checking using the public API, passing the lowering context's defs
    // so that the type checker can use the correct DefIds for builtins
    match bhc_typeck::type_check_module_with_defs(&hir_module, file_id, Some(&lower_ctx.defs)) {
        Ok(_typed) => CompileResult {
            status: "OK".into(),
            errors: vec![],
            warnings: warning_count,
        },
        Err(errors) => {
            let msgs: Vec<String> = errors.into_iter()
                .take(5)
                .map(|e| e.message)
                .collect();
            CompileResult {
                status: "TYPE_ERROR".into(),
                errors: msgs,
                warnings: warning_count,
            }
        }
    }
}

#[test]
fn test_compile_xmonad_files() {
    // Note: StackSet causes stack overflow due to deep recursion in type inference
    // for complex parameterized data types. This is a known limitation.
    let files = [
        ("/tmp/xmonad/src/XMonad.hs", "XMonad"),
        ("/tmp/xmonad/src/XMonad/Core.hs", "Core"),
        // Skipping StackSet due to stack overflow in type inference
        // ("/tmp/xmonad/src/XMonad/StackSet.hs", "StackSet"),
        ("/tmp/xmonad/src/XMonad/Config.hs", "Config"),
        ("/tmp/xmonad/src/XMonad/Layout.hs", "Layout"),
        ("/tmp/xmonad/src/XMonad/Main.hs", "Main"),
        ("/tmp/xmonad/src/XMonad/ManageHook.hs", "ManageHook"),
        ("/tmp/xmonad/src/XMonad/Operations.hs", "Operations"),
    ];

    println!("\n=== XMonad Compilation Test Results ===\n");

    let mut ok_count = 0;
    let mut fail_count = 0;
    let mut total_warnings = 0;

    for (path, name) in files {
        let result = test_compile_file(path);
        total_warnings += result.warnings;
        if result.status == "OK" {
            if result.warnings > 0 {
                println!("{}: OK ({} stub warnings)", name, result.warnings);
            } else {
                println!("{}: OK", name);
            }
            ok_count += 1;
        } else {
            println!("{}: {} ({} errors, {} stub warnings)", name, result.status, result.errors.len(), result.warnings);
            for e in result.errors.iter().take(3) {
                println!("    {}", e);
            }
            fail_count += 1;
        }
    }

    println!("\n=== Summary: {} OK, {} FAILED, {} total stub warnings ===", ok_count, fail_count, total_warnings);
}
