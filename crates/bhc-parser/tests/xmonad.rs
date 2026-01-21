//! Test parsing XMonad source files

use bhc_parser::parse_module;
use bhc_span::FileId;
use std::fs;

fn test_parse_file(path: &str) -> (usize, usize, usize) {
    let source = fs::read_to_string(path).expect("read file");
    let file_id = FileId::new(0);
    let (module, diagnostics) = parse_module(&source, file_id);
    
    let errors = diagnostics.iter()
        .filter(|d| d.severity == bhc_diagnostics::Severity::Error)
        .count();
    
    match module {
        Some(m) => (m.imports.len(), m.decls.len(), errors),
        None => (0, 0, errors),
    }
}

#[test]
fn test_xmonad_core() {
    let (imports, decls, errors) = test_parse_file("/tmp/xmonad/src/XMonad/Core.hs");
    println!("Core.hs: {} imports, {} decls, {} errors", imports, decls, errors);
    // Just check it parses at all
    assert!(decls > 0, "Should have some declarations");
}

#[test]
fn test_xmonad_stackset() {
    let (imports, decls, errors) = test_parse_file("/tmp/xmonad/src/XMonad/StackSet.hs");
    println!("StackSet.hs: {} imports, {} decls, {} errors", imports, decls, errors);
    assert!(decls > 0, "Should have some declarations");
}

#[test]
fn test_xmonad_main() {
    let (imports, decls, errors) = test_parse_file("/tmp/xmonad/src/XMonad.hs");
    println!("XMonad.hs: {} imports, {} decls, {} errors", imports, decls, errors);
    assert!(decls > 0 || imports > 0, "Should have content");
}

#[test]
fn test_all_xmonad_files() {
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

    let mut total_imports = 0;
    let mut total_decls = 0;
    let mut total_errors = 0;

    for (path, name) in files {
        let (imports, decls, errors) = test_parse_file(path);
        println!("{}: {} imports, {} decls, {} errors", name, imports, decls, errors);
        total_imports += imports;
        total_decls += decls;
        total_errors += errors;
    }

    println!("\nTOTAL: {} imports, {} decls, {} errors", total_imports, total_decls, total_errors);
}

#[test]
fn test_layout_parse_errors() {
    let path = "/tmp/xmonad/src/XMonad/Layout.hs";
    let source = fs::read_to_string(path).expect("read file");
    let file_id = FileId::new(0);
    let (_module, diagnostics) = parse_module(&source, file_id);

    println!("\n=== Layout.hs Parse Errors ===\n");
    for d in diagnostics.iter().filter(|d| d.severity == bhc_diagnostics::Severity::Error) {
        println!("Error: {}", d.message);
        for label in &d.labels {
            let offset = label.span.span.lo.0 as usize;
            let line = source[..offset].matches('\n').count() + 1;
            let line_start = source[..offset].rfind('\n').map(|x| x + 1).unwrap_or(0);
            let col = offset - line_start;
            println!("  At line {}, col {}: {}", line, col, label.message);
            // Show the source line
            let line_end = source[offset..].find('\n').map(|x| offset + x).unwrap_or(source.len());
            println!("    {}", &source[line_start..line_end]);
        }
        println!();
    }
}

#[test]
fn test_operations_parse_errors() {
    let path = "/tmp/xmonad/src/XMonad/Operations.hs";
    let source = fs::read_to_string(path).expect("read file");
    let file_id = FileId::new(0);
    let (_module, diagnostics) = parse_module(&source, file_id);

    println!("\n=== Operations.hs Parse Errors ===\n");
    for d in diagnostics.iter().filter(|d| d.severity == bhc_diagnostics::Severity::Error).take(10) {
        println!("Error: {}", d.message);
        for label in &d.labels {
            let offset = label.span.span.lo.0 as usize;
            let line = source[..offset].matches('\n').count() + 1;
            let line_start = source[..offset].rfind('\n').map(|x| x + 1).unwrap_or(0);
            let col = offset - line_start;
            println!("  At line {}, col {}: {}", line, col, label.message);
            // Show the source line
            let line_end = source[offset..].find('\n').map(|x| offset + x).unwrap_or(source.len());
            println!("    {}", &source[line_start..line_end]);
        }
        println!();
    }
}
