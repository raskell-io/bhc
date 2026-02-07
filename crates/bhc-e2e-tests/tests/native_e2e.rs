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
fn test_tier1_let_binding_native() {
    run_native_test("tier1_simple/let_binding", Profile::Default);
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
fn test_tier2_factorial_native() {
    run_native_test("tier2_functions/factorial", Profile::Default);
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

#[test]
fn test_tier2_lambda_native() {
    run_native_test("tier2_functions/lambda", Profile::Default);
}

#[test]
fn test_tier2_case_expr_native() {
    run_native_test("tier2_functions/case_expr", Profile::Default);
}

#[test]
fn test_tier2_custom_adt_native() {
    run_native_test("tier2_functions/custom_adt", Profile::Default);
}

#[test]
fn test_tier2_higher_order_native() {
    run_native_test("tier2_functions/higher_order", Profile::Default);
}

// =============================================================================
// Tier 2: Multi-Module Tests
// =============================================================================

#[test]
fn test_tier2_multimodule_basic_native() {
    run_native_test("tier2_functions/multimodule_basic", Profile::Default);
}

#[test]
fn test_tier2_multimodule_chain_native() {
    run_native_test("tier2_functions/multimodule_chain", Profile::Default);
}

// =============================================================================
// Tier 3: IO Tests
// =============================================================================

#[test]
fn test_tier3_print_sequence_native() {
    run_native_test("tier3_io/print_sequence", Profile::Default);
}

#[test]
fn test_tier3_file_stats_native() {
    run_native_test("tier3_io/file_stats", Profile::Default);
}

#[test]
fn test_tier3_file_reverse_native() {
    run_native_test("tier3_io/file_reverse", Profile::Default);
}

#[test]
fn test_tier3_bracket_io_native() {
    run_native_test("tier3_io/bracket_io", Profile::Default);
}

#[test]
fn test_tier3_catch_file_error_native() {
    run_native_test("tier3_io/catch_file_error", Profile::Default);
}

#[test]
fn test_tier3_exception_test_native() {
    run_native_test("tier3_io/exception_test", Profile::Default);
}

#[test]
fn test_tier3_handle_io_native() {
    run_native_test("tier3_io/handle_io", Profile::Default);
}

#[test]
fn test_tier3_system_ops_native() {
    run_native_test("tier3_io/system_ops", Profile::Default);
}

#[test]
fn test_tier3_milestone_b_wordcount_native() {
    run_native_test("tier3_io/milestone_b_wordcount", Profile::Default);
}

#[test]
fn test_tier3_milestone_b_transform_native() {
    run_native_test("tier3_io/milestone_b_transform", Profile::Default);
}

#[test]
fn test_tier3_milestone_c_markdown_native() {
    run_native_test("tier3_io/milestone_c_markdown", Profile::Default);
}

#[test]
fn test_tier3_reader_t_native() {
    run_native_test("tier3_io/reader_t", Profile::Default);
}

#[test]
fn test_tier3_state_t_native() {
    run_native_test("tier3_io/state_t", Profile::Default);
}

#[test]
fn test_tier3_state_t_string_native() {
    run_native_test("tier3_io/state_t_string", Profile::Default);
}

#[test]
fn test_tier3_state_t_case_native() {
    run_native_test("tier3_io/state_t_case", Profile::Default);
}

#[test]
fn test_tier3_milestone_d_csv_parser_native() {
    run_native_test("tier3_io/milestone_d_csv_parser", Profile::Default);
}

#[test]
fn test_tier3_milestone_e_json_native() {
    run_native_test("tier3_io/milestone_e_json", Profile::Default);
}

#[test]
fn test_tier3_text_basic_native() {
    run_native_test("tier3_io/text_basic", Profile::Default);
}

#[test]
fn test_tier3_bytestring_basic_native() {
    run_native_test("tier3_io/bytestring_basic", Profile::Default);
}

#[test]
fn test_tier3_text_encoding_native() {
    run_native_test("tier3_io/text_encoding", Profile::Default);
}

#[test]
fn test_tier3_except_t_native() {
    run_native_test("tier3_io/except_t", Profile::Default);
}

#[test]
fn test_tier3_writer_t_native() {
    run_native_test("tier3_io/writer_t", Profile::Default);
}

#[test]
fn test_tier3_monad_error_native() {
    run_native_test("tier3_io/monad_error", Profile::Default);
}

// Cross-transformer tests using MTL typeclasses (MonadReader, MonadState)
// NOTE: StateT-over-ReaderT is now supported. ReaderT-over-StateT requires additional work.
#[test]
fn test_tier3_cross_state_reader_native() {
    run_native_test("tier3_io/cross_state_reader", Profile::Default);
}

#[test]
#[ignore = "ReaderT r (StateT s IO) requires lifting StateT operations into ReaderT context - not yet implemented"]
fn test_tier3_cross_reader_state_native() {
    run_native_test("tier3_io/cross_reader_state", Profile::Default);
}

// Package import test - demonstrates importing from external package directories
#[test]
fn test_tier3_package_import_native() {
    run_native_test("tier3_io/package_import", Profile::Default);
}

// Data.Char predicates - isAlpha, isDigit, isSpace, toUpper, toLower, etc.
#[test]
fn test_tier3_char_predicates_native() {
    run_native_test("tier3_io/char_predicates", Profile::Default);
}

// Type-specialized show functions - showInt, showBool, showChar
#[test]
fn test_tier3_show_types_native() {
    run_native_test("tier3_io/show_types", Profile::Default);
}

// Data.Text.IO - native Text file I/O
#[test]
fn test_tier3_text_io_native() {
    run_native_test("tier3_io/text_io", Profile::Default);
}

// Show compound types
#[test]
fn test_tier3_show_string_native() {
    run_native_test("tier3_io/show_string", Profile::Default);
}

#[test]
fn test_tier3_show_list_native() {
    run_native_test("tier3_io/show_list", Profile::Default);
}

#[test]
fn test_tier3_show_maybe_native() {
    run_native_test("tier3_io/show_maybe", Profile::Default);
}

#[test]
fn test_tier3_show_either_native() {
    run_native_test("tier3_io/show_either", Profile::Default);
}

#[test]
fn test_tier3_show_tuple_native() {
    run_native_test("tier3_io/show_tuple", Profile::Default);
}

#[test]
fn test_tier3_show_unit_native() {
    run_native_test("tier3_io/show_unit", Profile::Default);
}

#[test]
fn test_tier3_numeric_ops_native() {
    run_native_test("tier3_io/numeric_ops", Profile::Default);
}

#[test]
fn test_tier3_divmod_native() {
    run_native_test("tier3_io/divmod", Profile::Default);
}

#[test]
fn test_tier3_ioref_basic_native() {
    run_native_test("tier3_io/ioref_basic", Profile::Default);
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
