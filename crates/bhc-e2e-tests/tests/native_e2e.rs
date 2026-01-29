//! Native backend E2E tests.
//!
//! These tests compile Haskell programs to native executables via LLVM
//! and verify correct output.

use bhc_e2e_tests::{discover_fixtures, format_failure_report, Backend, E2ERunner, Profile};

/// Run a single fixture test for native backend.
fn run_native_test(fixture_name: &str, profile: Profile) {
    let runner = E2ERunner::native(profile).keep_artifacts();
    let result = runner
        .run_fixture(fixture_name)
        .expect("Failed to run test");

    if !result.is_pass() && !result.is_skipped() {
        let fixture_path = bhc_e2e_tests::fixtures_dir().join(fixture_name);
        let test_case = bhc_e2e_tests::E2ETestCase::from_fixture(&fixture_path).unwrap();
        let report = format_failure_report(&test_case, Backend::Native, profile, &result, None);
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
fn test_tier1_hello_native() {
    run_native_test("tier1_simple/hello", Profile::Default);
}

#[test]
fn test_tier1_arithmetic_native() {
    run_native_test("tier1_simple/arithmetic", Profile::Default);
}

#[test]
fn test_tier1_list_range_native() {
    run_native_test("tier1_simple/list_range", Profile::Default);
}

// =============================================================================
// Tier 2: Function Tests
// =============================================================================

#[test]
fn test_tier2_fibonacci_native() {
    run_native_test("tier2_functions/fibonacci", Profile::Default);
}

#[test]
fn test_tier2_guards_native() {
    run_native_test("tier2_functions/guards", Profile::Default);
}

#[test]
fn test_tier2_pattern_match_native() {
    run_native_test("tier2_functions/pattern_match", Profile::Default);
}

#[test]
fn test_tier2_where_bindings_native() {
    run_native_test("tier2_functions/where_bindings", Profile::Default);
}

#[test]
fn test_tier2_mutual_recursion_native() {
    run_native_test("tier2_functions/mutual_recursion", Profile::Default);
}

// =============================================================================
// Tier 3: IO Tests
// =============================================================================

#[test]
fn test_tier3_print_sequence_native() {
    run_native_test("tier3_io/print_sequence", Profile::Default);
}

// =============================================================================
// Discovery Tests
// =============================================================================

#[test]
fn test_discover_tier1_fixtures() {
    let fixtures = discover_fixtures("tier1_simple").expect("Failed to discover fixtures");
    assert!(!fixtures.is_empty(), "Should find tier1 fixtures");

    // Verify we found the expected fixtures
    let names: Vec<_> = fixtures.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"hello"), "Should find hello fixture");
    assert!(
        names.contains(&"arithmetic"),
        "Should find arithmetic fixture"
    );
    assert!(
        names.contains(&"list_range"),
        "Should find list_range fixture"
    );
}

#[test]
fn test_discover_tier2_fixtures() {
    let fixtures = discover_fixtures("tier2_functions").expect("Failed to discover fixtures");
    assert!(!fixtures.is_empty(), "Should find tier2 fixtures");

    let names: Vec<_> = fixtures.iter().map(|f| f.name.as_str()).collect();
    assert!(
        names.contains(&"fibonacci"),
        "Should find fibonacci fixture"
    );
    assert!(names.contains(&"guards"), "Should find guards fixture");
    assert!(
        names.contains(&"pattern_match"),
        "Should find pattern_match fixture"
    );
    assert!(
        names.contains(&"where_bindings"),
        "Should find where_bindings fixture"
    );
    assert!(
        names.contains(&"mutual_recursion"),
        "Should find mutual_recursion fixture"
    );
}

// =============================================================================
// Numeric Profile Tests (when applicable)
// =============================================================================

#[test]
#[ignore = "Numeric profile native compilation not yet implemented"]
fn test_tier1_arithmetic_numeric_profile() {
    run_native_test("tier1_simple/arithmetic", Profile::Numeric);
}
