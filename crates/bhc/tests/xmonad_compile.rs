//! Test compiling XMonad source files through the pipeline

use bhc_lower::LowerContext;
use bhc_parser::parse_module;
use bhc_span::FileId;
use std::fs;

fn test_compile_file(path: &str) -> (String, Vec<String>) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return ("READ_ERROR".into(), vec![format!("{}", e)]),
    };

    let file_id = FileId::new(0);
    let (module, parse_diags) = parse_module(&source, file_id);

    let parse_errors: Vec<String> = parse_diags.iter()
        .filter(|d| d.severity == bhc_diagnostics::Severity::Error)
        .map(|d| d.message.clone())
        .collect();

    if !parse_errors.is_empty() {
        return ("PARSE_ERROR".into(), parse_errors);
    }

    let module = match module {
        Some(m) => m,
        None => return ("NO_MODULE".into(), vec![]),
    };

    // Try lowering to HIR
    let mut lower_ctx = LowerContext::with_builtins();
    let hir_module = match bhc_lower::lower_module(&mut lower_ctx, &module) {
        Ok(m) => m,
        Err(e) => return ("LOWER_ERROR".into(), vec![format!("{:?}", e)]),
    };

    // Try type checking using the public API
    match bhc_typeck::type_check_module(&hir_module, file_id) {
        Ok(_typed) => ("OK".into(), vec![]),
        Err(errors) => {
            let msgs: Vec<String> = errors.into_iter()
                .take(5)
                .map(|e| e.message)
                .collect();
            ("TYPE_ERROR".into(), msgs)
        }
    }
}

#[test]
fn test_compile_xmonad_files() {
    let files = [
        ("/tmp/xmonad/src/XMonad.hs", "XMonad"),
        ("/tmp/xmonad/src/XMonad/Core.hs", "Core"),
        ("/tmp/xmonad/src/XMonad/StackSet.hs", "StackSet"),
        ("/tmp/xmonad/src/XMonad/Config.hs", "Config"),
        ("/tmp/xmonad/src/XMonad/Layout.hs", "Layout"),
        ("/tmp/xmonad/src/XMonad/Main.hs", "Main"),
        ("/tmp/xmonad/src/XMonad/ManageHook.hs", "ManageHook"),
        ("/tmp/xmonad/src/XMonad/Operations.hs", "Operations"),
    ];
    
    println!("\n=== XMonad Compilation Test Results ===\n");
    
    let mut ok_count = 0;
    let mut fail_count = 0;
    
    for (path, name) in files {
        let (status, errors) = test_compile_file(path);
        if status == "OK" {
            println!("{}: OK", name);
            ok_count += 1;
        } else {
            println!("{}: {} ({} errors)", name, status, errors.len());
            for e in errors.iter().take(2) {
                println!("    {}", e);
            }
            fail_count += 1;
        }
    }
    
    println!("\n=== Summary: {} OK, {} FAILED ===", ok_count, fail_count);
}
