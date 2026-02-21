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
fn test_tier3_cross_reader_state_native() {
    run_native_test("tier3_io/cross_reader_state", Profile::Default);
}

#[test]
fn test_tier3_cross_except_state_native() {
    run_native_test("tier3_io/cross_except_state", Profile::Default);
}

#[test]
fn test_tier3_cross_except_reader_native() {
    run_native_test("tier3_io/cross_except_reader", Profile::Default);
}

#[test]
fn test_tier3_cross_writer_state_native() {
    run_native_test("tier3_io/cross_writer_state", Profile::Default);
}

#[test]
fn test_tier3_cross_writer_reader_native() {
    run_native_test("tier3_io/cross_writer_reader", Profile::Default);
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

#[test]
fn test_tier3_data_maybe_native() {
    run_native_test("tier3_io/data_maybe", Profile::Default);
}

#[test]
fn test_tier3_data_either_native() {
    run_native_test("tier3_io/data_either", Profile::Default);
}

#[test]
fn test_tier3_guard_basic_native() {
    run_native_test("tier3_io/guard_basic", Profile::Default);
}

#[test]
fn test_tier3_when_unless_native() {
    run_native_test("tier3_io/when_unless", Profile::Default);
}

#[test]
fn test_tier3_mapm_basic_native() {
    run_native_test("tier3_io/mapm_basic", Profile::Default);
}

#[test]
fn test_tier3_any_all_native() {
    run_native_test("tier3_io/any_all", Profile::Default);
}

#[test]
fn test_tier3_scanr_basic_native() {
    run_native_test("tier3_io/scanr_basic", Profile::Default);
}

#[test]
fn test_tier3_unfoldr_basic_native() {
    run_native_test("tier3_io/unfoldr_basic", Profile::Default);
}

#[test]
fn test_tier3_zip3_basic_native() {
    run_native_test("tier3_io/zip3_basic", Profile::Default);
}

#[test]
fn test_tier3_take_iterate_native() {
    run_native_test("tier3_io/take_iterate", Profile::Default);
}

#[test]
fn test_tier3_intersect_basic_native() {
    run_native_test("tier3_io/intersect_basic", Profile::Default);
}

#[test]
fn test_tier3_max_min_and_or_native() {
    run_native_test("tier3_io/max_min_and_or", Profile::Default);
}

#[test]
fn test_tier3_elem_index_prefix_native() {
    run_native_test("tier3_io/elem_index_prefix", Profile::Default);
}

#[test]
fn test_tier3_tails_inits_native() {
    run_native_test("tier3_io/tails_inits", Profile::Default);
}

#[test]
fn test_tier3_ordering_basic_native() {
    run_native_test("tier3_io/ordering_basic", Profile::Default);
}

#[test]
fn test_tier3_monadic_combinators_native() {
    run_native_test("tier3_io/monadic_combinators", Profile::Default);
}

#[test]
fn test_tier3_zipwithm_basic_native() {
    run_native_test("tier3_io/zipwithm_basic", Profile::Default);
}

// System.FilePath operations
#[test]
fn test_tier3_filepath_basic_native() {
    run_native_test("tier3_io/filepath_basic", Profile::Default);
}

// System.Directory operations
#[test]
fn test_tier3_directory_ops_native() {
    run_native_test("tier3_io/directory_ops", Profile::Default);
}

// Data.Map operations
#[test]
fn test_tier3_map_basic_native() {
    run_native_test("tier3_io/map_basic", Profile::Default);
}

#[test]
fn test_tier3_map_complete_native() {
    run_native_test("tier3_io/map_complete", Profile::Default);
}

// Data.Set operations
#[test]
fn test_tier3_set_basic_native() {
    run_native_test("tier3_io/set_basic", Profile::Default);
}

// Data.IntMap + Data.IntSet operations
#[test]
fn test_tier3_intmap_intset_native() {
    run_native_test("tier3_io/intmap_intset", Profile::Default);
}

#[test]
fn test_tier3_string_read_native() {
    run_native_test("tier3_io/string_read", Profile::Default);
}

// E.30: List operations (splitAt, span, break, takeWhile, dropWhile, unzip)
#[test]
fn test_tier3_list_split_span_native() {
    run_native_test("tier3_io/list_split_span", Profile::Default);
}

#[test]
fn test_tier3_list_takewhile_dropwhile_native() {
    run_native_test("tier3_io/list_takewhile_dropwhile", Profile::Default);
}

#[test]
fn test_tier3_list_unzip_native() {
    run_native_test("tier3_io/list_unzip", Profile::Default);
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
// Deriving Tests (E.23)
// =============================================================================

#[test]
fn test_tier3_derive_show() {
    run_native_test("tier3_io/derive_show", Profile::Default);
}

#[test]
fn test_tier2_derive_eq() {
    run_native_test("tier2_functions/derive_eq", Profile::Default);
}

#[test]
fn test_tier3_derive_ord() {
    run_native_test("tier3_io/derive_ord", Profile::Default);
}

#[test]
fn test_tier3_list_by_ops_native() {
    run_native_test("tier3_io/list_by_ops", Profile::Default);
}

#[test]
fn test_tier3_data_function_native() {
    run_native_test("tier3_io/data_function", Profile::Default);
}

#[test]
fn test_tier3_tuple_functions_native() {
    run_native_test("tier3_io/tuple_functions", Profile::Default);
}

#[test]
fn test_tier3_enum_functions_native() {
    run_native_test("tier3_io/enum_functions", Profile::Default);
}

#[test]
fn test_tier3_fold_misc_native() {
    run_native_test("tier3_io/fold_misc", Profile::Default);
}

#[test]
fn test_tier3_flip_test_native() {
    run_native_test("tier3_io/flip_test", Profile::Default);
}

#[test]
fn test_tier3_show_double_native() {
    run_native_test("tier3_io/show_double", Profile::Default);
}

#[test]
fn test_tier3_map_maybe_native() {
    run_native_test("tier3_io/map_maybe", Profile::Default);
}

#[test]
fn test_tier3_show_nested_native() {
    run_native_test("tier3_io/show_nested", Profile::Default);
}

#[test]
fn test_tier3_show_nested_maybe_native() {
    run_native_test("tier3_io/show_nested_maybe", Profile::Default);
}

#[test]
fn test_tier3_show_nested_list_native() {
    run_native_test("tier3_io/show_nested_list", Profile::Default);
}

#[test]
fn test_tier3_overloaded_strings_native() {
    run_native_test("tier3_io/overloaded_strings", Profile::Default);
}

#[test]
fn test_tier3_record_basic_native() {
    run_native_test("tier3_io/record_basic", Profile::Default);
}

#[test]
fn test_tier3_record_update_native() {
    run_native_test("tier3_io/record_update", Profile::Default);
}

#[test]
fn test_tier3_record_pattern_native() {
    run_native_test("tier3_io/record_pattern", Profile::Default);
}

#[test]
fn test_tier3_record_multi_native() {
    run_native_test("tier3_io/record_multi", Profile::Default);
}

#[test]
fn test_tier3_record_wildcards_native() {
    run_native_test("tier3_io/record_wildcards", Profile::Default);
}

#[test]
fn test_tier3_view_patterns_native() {
    run_native_test("tier3_io/view_patterns", Profile::Default);
}

#[test]
fn test_tier3_multi_way_if_native() {
    run_native_test("tier3_io/multi_way_if", Profile::Default);
}

#[test]
fn test_tier3_tuple_sections_native() {
    run_native_test("tier3_io/tuple_sections", Profile::Default);
}

// E.36: Char Enum ranges
#[test]
fn test_tier3_char_ranges_native() {
    run_native_test("tier3_io/char_ranges", Profile::Default);
}

#[test]
fn test_tier3_char_enum_step_native() {
    run_native_test("tier3_io/char_enum_step", Profile::Default);
}

// E.37: Data.Char first-class predicates
#[test]
fn test_tier3_char_first_class_native() {
    run_native_test("tier3_io/char_first_class", Profile::Default);
}

// E.38: Manual typeclass instances
#[test]
fn test_tier3_manual_instances_native() {
    run_native_test("tier3_io/manual_instances", Profile::Default);
}

// Previously unregistered tests
#[test]
fn test_tier3_catch_test_native() {
    run_native_test("tier3_io/catch_test", Profile::Default);
}

#[test]
fn test_tier3_multi_bind_native() {
    run_native_test("tier3_io/multi_bind", Profile::Default);
}

#[test]
fn test_tier3_readfile_native() {
    run_native_test("tier3_io/readfile", Profile::Default);
}

// E.39: User-defined typeclasses with dictionary passing
#[test]
fn test_tier3_user_typeclass_native() {
    run_native_test("tier3_io/user_typeclass", Profile::Default);
}

#[test]
fn test_tier3_multi_method_class_native() {
    run_native_test("tier3_io/multi_method_class", Profile::Default);
}

// E.40: Higher-kinded dictionary passing for user-defined typeclasses
#[test]
fn test_tier3_hk_typeclass_native() {
    run_native_test("tier3_io/hk_typeclass", Profile::Default);
}

// E.41: Default methods and superclass constraints
#[test]
fn test_tier3_default_method_native() {
    run_native_test("tier3_io/default_method", Profile::Default);
}

#[test]
fn test_tier3_superclass_native() {
    run_native_test("tier3_io/superclass", Profile::Default);
}

// E.42: Deriving for user-defined typeclasses (DeriveAnyClass)
#[test]
fn test_tier3_derive_any_class_native() {
    run_native_test("tier3_io/derive_any_class", Profile::Default);
}

#[test]
fn test_tier3_derive_any_class_multi_native() {
    run_native_test("tier3_io/derive_any_class_multi", Profile::Default);
}

// =============================================================================
// E.63: DeriveGeneric + NFData/DeepSeq
// =============================================================================

#[test]
fn test_tier3_derive_generic_native() {
    run_native_test("tier3_io/derive_generic", Profile::Default);
}

// =============================================================================
// E.43: Word Types
// =============================================================================

#[test]
fn test_tier3_word_types_native() {
    run_native_test("tier3_io/word_types", Profile::Default);
}

#[test]
fn test_tier3_word_conversion_native() {
    run_native_test("tier3_io/word_conversion", Profile::Default);
}

// =============================================================================
// E.44: Lazy Let-Bindings
// =============================================================================

#[test]
fn test_tier3_lazy_let_native() {
    run_native_test("tier3_io/lazy_let", Profile::Default);
}

// =============================================================================
// E.58: Lazy Let-Bindings (full)
// =============================================================================

#[test]
fn test_tier3_lazy_let_basic_native() {
    run_native_test("tier3_io/lazy_let_basic", Profile::Default);
}

#[test]
fn test_tier3_lazy_let_conditional_native() {
    run_native_test("tier3_io/lazy_let_conditional", Profile::Default);
}

// =============================================================================
// E.45: Integer (Arbitrary Precision)
// =============================================================================

#[test]
fn test_tier3_integer_basic_native() {
    run_native_test("tier3_io/integer_basic", Profile::Default);
}

#[test]
fn test_tier3_integer_large_native() {
    run_native_test("tier3_io/integer_large", Profile::Default);
}

// =============================================================================
// E.46: ScopedTypeVariables
// =============================================================================

#[test]
fn test_tier3_scoped_tyvars_basic_native() {
    run_native_test("tier3_io/scoped_tyvars_basic", Profile::Default);
}

#[test]
fn test_tier3_scoped_tyvars_list_native() {
    run_native_test("tier3_io/scoped_tyvars_list", Profile::Default);
}

#[test]
fn test_tier3_gnd_basic_native() {
    run_native_test("tier3_io/gnd_basic", Profile::Default);
}

#[test]
fn test_tier3_gnd_newtype_erasure_native() {
    run_native_test("tier3_io/gnd_newtype_erasure", Profile::Default);
}

// =============================================================================
// E.48: FlexibleInstances, FlexibleContexts, Instance Context Propagation
// =============================================================================

#[test]
fn test_tier3_flexible_instances_native() {
    run_native_test("tier3_io/flexible_instances", Profile::Default);
}

#[test]
fn test_tier3_instance_context_native() {
    run_native_test("tier3_io/instance_context", Profile::Default);
}

#[test]
fn test_tier3_instance_context_multi_native() {
    run_native_test("tier3_io/instance_context_multi", Profile::Default);
}

#[test]
fn test_tier3_multi_param_typeclass_native() {
    run_native_test("tier3_io/multi_param_typeclass", Profile::Default);
}

// E.50: FunctionalDependencies
#[test]
fn test_tier3_functional_deps_native() {
    run_native_test("tier3_io/functional_deps", Profile::Default);
}

// E.51: DeriveFunctor + fmap for pure types
#[test]
fn test_tier3_derive_functor_native() {
    run_native_test("tier3_io/derive_functor", Profile::Default);
}

// E.52: DeriveFoldable + foldr for user ADTs
#[test]
fn test_tier3_derive_foldable_native() {
    run_native_test("tier3_io/derive_foldable", Profile::Default);
}

// E.53: DeriveTraversable + traverse for user ADTs
#[test]
fn test_tier3_derive_traversable_native() {
    run_native_test("tier3_io/derive_traversable", Profile::Default);
}

#[test]
fn test_tier3_derive_enum_native() {
    run_native_test("tier3_io/derive_enum", Profile::Default);
}

// =============================================================================
// E.59: Tier 1 Extensions (LambdaCase, NamedFieldPuns, InstanceSigs,
//       EmptyDataDecls, StrictData)
// =============================================================================

#[test]
fn test_tier3_lambda_case_native() {
    run_native_test("tier3_io/lambda_case", Profile::Default);
}

#[test]
fn test_tier3_named_field_puns_native() {
    run_native_test("tier3_io/named_field_puns", Profile::Default);
}

#[test]
fn test_tier3_instance_sigs_native() {
    run_native_test("tier3_io/instance_sigs", Profile::Default);
}

#[test]
fn test_tier3_empty_data_decl_native() {
    run_native_test("tier3_io/empty_data_decl", Profile::Default);
}

#[test]
fn test_tier3_strict_fields_native() {
    run_native_test("tier3_io/strict_fields", Profile::Default);
}

#[test]
fn test_tier3_gadt_basic_native() {
    run_native_test("tier3_io/gadt_basic", Profile::Default);
}

#[test]
fn test_tier3_gadt_phantom_native() {
    run_native_test("tier3_io/gadt_phantom", Profile::Default);
}

// =============================================================================
// E.62: Pandoc Tier 1+2 Extensions
// =============================================================================

#[test]
fn test_tier3_strict_data_native() {
    run_native_test("tier3_io/strict_data", Profile::Default);
}

#[test]
fn test_tier3_pattern_guards_native() {
    run_native_test("tier3_io/pattern_guards", Profile::Default);
}

#[test]
fn test_tier3_standalone_deriving_native() {
    run_native_test("tier3_io/standalone_deriving", Profile::Default);
}

#[test]
fn test_tier3_pattern_synonyms_native() {
    run_native_test("tier3_io/pattern_synonyms", Profile::Default);
}

// =============================================================================
// E.61: TypeOperators
// =============================================================================

#[test]
fn test_tier3_type_operators_basic_native() {
    run_native_test("tier3_io/type_operators_basic", Profile::Default);
}

#[test]
fn test_tier3_type_operators_prefix_native() {
    run_native_test("tier3_io/type_operators_prefix", Profile::Default);
}

// =============================================================================
// E.64: EmptyCase, DefaultSignatures, OverloadedLists
// =============================================================================

#[test]
fn test_tier3_empty_case_native() {
    run_native_test("tier3_io/empty_case", Profile::Default);
}

#[test]
fn test_tier3_default_signatures_native() {
    run_native_test("tier3_io/default_signatures", Profile::Default);
}

#[test]
fn test_tier3_overloaded_lists_native() {
    run_native_test("tier3_io/overloaded_lists", Profile::Default);
}

// =============================================================================
// E.65: Layout Rule Verification
// =============================================================================

#[test]
fn test_tier3_layout_rule_native() {
    run_native_test("tier3_io/layout_rule", Profile::Default);
}

// =============================================================================
// Numeric Profile Tests (when applicable)
// =============================================================================

#[test]
#[ignore = "Numeric profile native compilation not yet implemented"]
fn test_tier1_arithmetic_numeric_profile() {
    run_native_test("tier1_simple/arithmetic", Profile::Numeric);
}
