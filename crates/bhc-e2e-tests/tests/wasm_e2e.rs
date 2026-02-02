//! WASM backend E2E tests.
//!
//! These tests compile Haskell programs to WebAssembly and run them
//! using wasmtime with WASI support.

use bhc_e2e_tests::{format_failure_report, Backend, E2ERunner, Profile};

/// Run a single fixture test for WASM backend.
fn run_wasm_test(fixture_name: &str, profile: Profile) {
    let runner = E2ERunner::wasm(profile).keep_artifacts();
    let result = runner
        .run_fixture(fixture_name)
        .expect("Failed to run test");

    if !result.is_pass() && !result.is_skipped() {
        let fixture_path = bhc_e2e_tests::fixtures_dir().join(fixture_name);
        let test_case = bhc_e2e_tests::E2ETestCase::from_fixture(&fixture_path).unwrap();
        let report = format_failure_report(&test_case, Backend::Wasm, profile, &result, None);
        panic!("{}", report);
    }

    assert!(
        result.is_pass() || result.is_skipped(),
        "Test failed: {:?}",
        result
    );
}

// =============================================================================
// Tier 1: Simple Tests
// =============================================================================

#[test]
fn test_tier1_hello_wasm() {
    run_wasm_test("tier1_simple/hello", Profile::Default);
}

#[test]
fn test_tier1_arithmetic_wasm() {
    run_wasm_test("tier1_simple/arithmetic", Profile::Default);
}

// Tier 1: let bindings

#[test]
fn test_tier1_let_binding_wasm() {
    run_wasm_test("tier1_simple/let_binding", Profile::Default);
}

// Tier 1: list range (fused sum/enumFromTo)

#[test]
fn test_tier1_list_range_wasm() {
    run_wasm_test("tier1_simple/list_range", Profile::Default);
}

// =============================================================================
// Tier 2: Function Tests
// =============================================================================

#[test]
fn test_tier2_fibonacci_wasm() {
    run_wasm_test("tier2_functions/fibonacci", Profile::Default);
}

#[test]
fn test_tier2_factorial_wasm() {
    run_wasm_test("tier2_functions/factorial", Profile::Default);
}

#[test]
fn test_tier2_guards_wasm() {
    run_wasm_test("tier2_functions/guards", Profile::Default);
}

#[test]
fn test_tier2_pattern_match_wasm() {
    run_wasm_test("tier2_functions/pattern_match", Profile::Default);
}

#[test]
fn test_tier2_where_bindings_wasm() {
    run_wasm_test("tier2_functions/where_bindings", Profile::Default);
}

#[test]
fn test_tier2_mutual_recursion_wasm() {
    run_wasm_test("tier2_functions/mutual_recursion", Profile::Default);
}

// =============================================================================
// Tier 3: IO Tests
// =============================================================================

#[test]
fn test_tier3_print_sequence_wasm() {
    run_wasm_test("tier3_io/print_sequence", Profile::Default);
}

#[test]
fn test_tier3_multi_bind_wasm() {
    run_wasm_test("tier3_io/multi_bind", Profile::Default);
}

#[test]
fn test_tier3_catch_test_wasm() {
    run_wasm_test("tier3_io/catch_test", Profile::Default);
}

// =============================================================================
// Edge Profile Tests (minimal runtime)
// =============================================================================

#[test]
fn test_tier1_hello_wasm_edge() {
    run_wasm_test("tier1_simple/hello", Profile::Edge);
}

#[test]
fn test_tier1_arithmetic_wasm_edge() {
    run_wasm_test("tier1_simple/arithmetic", Profile::Edge);
}

#[test]
fn test_tier1_let_binding_wasm_edge() {
    run_wasm_test("tier1_simple/let_binding", Profile::Edge);
}

#[test]
fn test_tier2_factorial_wasm_edge() {
    run_wasm_test("tier2_functions/factorial", Profile::Edge);
}

// =============================================================================
// Numeric Profile Tests (SIMD, fusion)
// =============================================================================

#[test]
fn test_tier1_arithmetic_wasm_numeric() {
    run_wasm_test("tier1_simple/arithmetic", Profile::Numeric);
}
