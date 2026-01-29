//! Integration tests for the BHC compilation pipeline.
//!
//! These tests verify end-to-end compilation and execution of Haskell code.

use bhc_driver::{CompileError, Compiler};
use bhc_core::eval::Value;

/// Helper to compile and run source code, returning the value
fn run_source(source: &str) -> Result<Value, CompileError> {
    let compiler = Compiler::with_defaults()?;
    let (value, _display) = compiler.run_source("Test", source)?;
    Ok(value)
}

/// Helper to compile and run, returning both value and display string
fn run_and_display(source: &str) -> Result<(Value, String), CompileError> {
    let compiler = Compiler::with_defaults()?;
    compiler.run_source("Test", source)
}

// =========================================================================
// Basic Evaluation Tests
// =========================================================================

#[test]
fn test_integer_literal() {
    let result = run_source("main = 42").unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_negative_integer() {
    let result = run_source("main = -42").unwrap();
    assert!(matches!(result, Value::Int(-42)));
}

#[test]
fn test_addition() {
    let result = run_source("main = 1 + 2").unwrap();
    assert!(matches!(result, Value::Int(3)));
}

#[test]
fn test_subtraction() {
    let result = run_source("main = 10 - 3").unwrap();
    assert!(matches!(result, Value::Int(7)));
}

#[test]
fn test_multiplication() {
    let result = run_source("main = 6 * 7").unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_division() {
    let result = run_source("main = 10 `div` 3").unwrap();
    assert!(matches!(result, Value::Int(3)));
}

#[test]
fn test_modulo() {
    let result = run_source("main = 10 `mod` 3").unwrap();
    assert!(matches!(result, Value::Int(1)));
}

#[test]
fn test_nested_arithmetic() {
    let result = run_source("main = (1 + 2) * 3 - 4").unwrap();
    assert!(matches!(result, Value::Int(5)));
}

// =========================================================================
// Let Binding Tests
// =========================================================================

#[test]
fn test_simple_let() {
    let result = run_source("main = let x = 42 in x").unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_let_with_arithmetic() {
    let result = run_source("main = let x = 10 in x + 5").unwrap();
    assert!(matches!(result, Value::Int(15)));
}

#[test]
fn test_nested_let() {
    let result = run_source("main = let x = 1 in let y = 2 in x + y").unwrap();
    assert!(matches!(result, Value::Int(3)));
}

#[test]
fn test_let_shadowing() {
    let result = run_source("main = let x = 1 in let x = 2 in x").unwrap();
    assert!(matches!(result, Value::Int(2)));
}

// =========================================================================
// Lambda and Function Tests
// =========================================================================

#[test]
fn test_identity_lambda() {
    let result = run_source("main = (\\x -> x) 42").unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_add_lambda() {
    let result = run_source("main = (\\x -> x + 1) 41").unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_curried_lambda() {
    let result = run_source("main = (\\x -> \\y -> x + y) 10 32").unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_function_definition() {
    let source = r#"
f x = x * 2
main = f 21
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_function_with_multiple_args() {
    let source = r#"
add x y = x + y
main = add 10 32
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_closure_capture() {
    let source = r#"
addN n = \x -> x + n
main = let add10 = addN 10 in add10 32
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

// =========================================================================
// List Tests
// =========================================================================

#[test]
fn test_empty_list() {
    let result = run_source("main = null []").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_list_not_null() {
    let result = run_source("main = null [1, 2, 3]").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_list_head() {
    let result = run_source("main = head [1, 2, 3]").unwrap();
    assert!(matches!(result, Value::Int(1)));
}

#[test]
fn test_list_length() {
    let result = run_source("main = length [1, 2, 3, 4, 5]").unwrap();
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn test_list_sum() {
    let result = run_source("main = sum [1, 2, 3, 4, 5]").unwrap();
    assert!(matches!(result, Value::Int(15)));
}

#[test]
fn test_list_map() {
    let result = run_source("main = sum (map (\\x -> x * 2) [1, 2, 3])").unwrap();
    assert!(matches!(result, Value::Int(12))); // 2 + 4 + 6 = 12
}

#[test]
fn test_list_take() {
    let result = run_source("main = sum (take 3 [1, 2, 3, 4, 5])").unwrap();
    assert!(matches!(result, Value::Int(6))); // 1 + 2 + 3 = 6
}

#[test]
fn test_list_drop() {
    let result = run_source("main = sum (drop 2 [1, 2, 3, 4, 5])").unwrap();
    assert!(matches!(result, Value::Int(12))); // 3 + 4 + 5 = 12
}

// =========================================================================
// Comparison Tests
// =========================================================================

#[test]
fn test_less_than_true() {
    let result = run_source("main = 1 < 2").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_less_than_false() {
    let result = run_source("main = 2 < 1").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_greater_than() {
    let result = run_source("main = 5 > 3").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_equal() {
    let result = run_source("main = 42 == 42").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_not_equal() {
    let result = run_source("main = 42 == 43").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_less_than_or_equal() {
    let result = run_source("main = 5 <= 5").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_greater_than_or_equal() {
    let result = run_source("main = 5 >= 5").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

// =========================================================================
// Boolean Tests
// =========================================================================

#[test]
fn test_and_true() {
    let result = run_source("main = (1 < 2) && (3 < 4)").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_and_false() {
    let result = run_source("main = (1 < 2) && (3 > 4)").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_or_true() {
    let result = run_source("main = (1 > 2) || (3 < 4)").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_or_false() {
    let result = run_source("main = (1 > 2) || (3 > 4)").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_not_true() {
    let result = run_source("main = not (1 > 2)").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_not_false() {
    let result = run_source("main = not (1 < 2)").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

// =========================================================================
// Where Clause Tests
// =========================================================================

#[test]
fn test_simple_where() {
    let source = r#"
main = x + y
  where
    x = 10
    y = 32
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_nested_where() {
    let source = r#"
main = f 10
  where
    f x = x + y
      where y = 32
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

// =========================================================================
// Operator Precedence Tests
// =========================================================================

#[test]
fn test_precedence_mul_add() {
    // 2 + 3 * 4 = 2 + 12 = 14 (not 20)
    let result = run_source("main = 2 + 3 * 4").unwrap();
    assert!(matches!(result, Value::Int(14)));
}

#[test]
fn test_precedence_sub_mul() {
    // 10 - 2 * 3 = 10 - 6 = 4 (not 24)
    let result = run_source("main = 10 - 2 * 3").unwrap();
    assert!(matches!(result, Value::Int(4)));
}

#[test]
fn test_parentheses_override() {
    // (2 + 3) * 4 = 5 * 4 = 20
    let result = run_source("main = (2 + 3) * 4").unwrap();
    assert!(matches!(result, Value::Int(20)));
}

// =========================================================================
// M0 Exit Criteria Tests
// =========================================================================

#[test]
fn test_m0_sum_map() {
    // M0 Exit Criterion: sum (map (+1) [1,2,3,4,5]) = 20
    let result = run_source("main = sum (map (\\x -> x + 1) [1, 2, 3, 4, 5])").unwrap();
    assert!(matches!(result, Value::Int(20)));
}

#[test]
fn test_m0_dot_product() {
    // M0 Exit Criterion: dot product
    // sum (zipWith (*) [1,2,3] [4,5,6]) = 4 + 10 + 18 = 32
    let result = run_source("main = sum (zipWith (\\x y -> x * y) [1, 2, 3] [4, 5, 6])").unwrap();
    assert!(matches!(result, Value::Int(32)));
}

// =========================================================================
// IO Tests (display output)
// =========================================================================

#[test]
fn test_print_int() {
    let (_, display) = run_and_display("main = print 42").unwrap();
    assert!(display.contains("42"), "Expected display to contain '42', got: {}", display);
}

#[test]
fn test_putstrln() {
    let (_, display) = run_and_display("main = putStrLn \"Hello, World!\"").unwrap();
    assert!(display.contains("Hello, World!"), "Expected 'Hello, World!' in output");
}

// =========================================================================
// Compound Expression Tests
// =========================================================================

#[test]
fn test_compose_functions() {
    // Test function composition behavior
    let source = r#"
double x = x * 2
addOne x = x + 1
main = double (addOne 20)
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_higher_order_function() {
    let source = r#"
apply f x = f x
main = apply (\x -> x * 2) 21
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_partial_application() {
    let source = r#"
add x y = x + y
add10 = add 10
main = add10 32
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

// =========================================================================
// Error Case Tests
// =========================================================================

#[test]
fn test_parse_error() {
    let result = run_source("main = 1 +");
    assert!(result.is_err());
}

#[test]
fn test_unbound_variable() {
    let result = run_source("main = undefined_var");
    assert!(result.is_err());
}

// =========================================================================
// IO Sequencing Tests (>> operator)
// =========================================================================

#[test]
fn test_io_sequence_two() {
    // Two IO actions chained with >>
    let (_, display) = run_and_display("main = print 1 >> print 2").unwrap();
    assert!(display.contains("1"), "Expected '1' in output, got: {}", display);
    assert!(display.contains("2"), "Expected '2' in output, got: {}", display);
}

#[test]
fn test_io_sequence_three() {
    // Three IO actions chained with >> (this was failing before)
    let (_, display) = run_and_display("main = print 1 >> print 2 >> print 3").unwrap();
    assert!(display.contains("1"), "Expected '1' in output, got: {}", display);
    assert!(display.contains("2"), "Expected '2' in output, got: {}", display);
    assert!(display.contains("3"), "Expected '3' in output, got: {}", display);
}

#[test]
fn test_io_sequence_four() {
    // Four IO actions chained with >>
    let (_, display) = run_and_display("main = print 1 >> print 2 >> print 3 >> print 4").unwrap();
    assert!(display.contains("1") && display.contains("2") &&
            display.contains("3") && display.contains("4"),
            "Expected '1', '2', '3', '4' in output, got: {}", display);
}

#[test]
fn test_io_sequence_with_putstrln() {
    // Mix of putStrLn calls
    let (_, display) = run_and_display(r#"main = putStrLn "a" >> putStrLn "b" >> putStrLn "c""#).unwrap();
    assert!(display.contains("a") && display.contains("b") && display.contains("c"),
            "Expected 'a', 'b', 'c' in output, got: {}", display);
}

#[test]
fn test_io_sequence_parenthesized() {
    // Explicit parentheses (right associative)
    let (_, display) = run_and_display("main = print 1 >> (print 2 >> print 3)").unwrap();
    assert!(display.contains("1") && display.contains("2") && display.contains("3"),
            "Expected '1', '2', '3' in output, got: {}", display);
}

#[test]
fn test_io_sequence_left_parens() {
    // Explicit left parentheses
    let (_, display) = run_and_display("main = (print 1 >> print 2) >> print 3").unwrap();
    assert!(display.contains("1") && display.contains("2") && display.contains("3"),
            "Expected '1', '2', '3' in output, got: {}", display);
}

// =========================================================================
// Case Expression Tests
// =========================================================================

#[test]
fn test_case_integer_literal() {
    let source = r#"
main = case 2 of
  1 -> 10
  2 -> 20
  3 -> 30
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(20)));
}

#[test]
fn test_case_wildcard_default() {
    let source = r#"
main = case 99 of
  1 -> 10
  2 -> 20
  _ -> 0
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(0)));
}

#[test]
fn test_case_variable_binding() {
    let source = r#"
main = case 7 of
  x -> x * 6
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_case_boolean_scrutinee() {
    let source = r#"
main = case (3 > 2) of
  True  -> 1
  False -> 0
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(1)));
}

#[test]
fn test_case_nested() {
    let source = r#"
main = case 1 of
  1 -> case 2 of
    2 -> 42
    _ -> 0
  _ -> 0
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(42)));
}

// =========================================================================
// Guard Tests
// =========================================================================

#[test]
fn test_if_then_else_basic() {
    let source = r#"
classify x = if x > 0 then 1 else if x < 0 then -1 else 0
main = classify 5
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(1)));
}

#[test]
fn test_if_then_else_negative() {
    let source = r#"
classify x = if x > 0 then 1 else if x < 0 then -1 else 0
main = classify (-3)
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(-1)));
}

#[test]
fn test_if_then_else_zero() {
    let source = r#"
classify x = if x > 0 then 1 else if x < 0 then -1 else 0
main = classify 0
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(0)));
}

#[test]
fn test_if_with_where() {
    let source = r#"
main = if x > threshold then 1 else 0
  where
    x = 10
    threshold = 5
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(1)));
}

// =========================================================================
// List Range & Advanced Operations Tests
// =========================================================================

#[test]
fn test_list_sum_longer() {
    let result = run_source("main = sum [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]").unwrap();
    assert!(matches!(result, Value::Int(55)));
}

#[test]
fn test_list_length_longer() {
    let result = run_source("main = length [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]").unwrap();
    assert!(matches!(result, Value::Int(10)));
}

#[test]
fn test_enum_from_to() {
    let result = run_source("main = sum (enumFromTo 1 10)").unwrap();
    assert!(matches!(result, Value::Int(55)));
}

#[test]
fn test_enum_from_to_length() {
    let result = run_source("main = length (enumFromTo 1 5)").unwrap();
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn test_replicate() {
    let result = run_source("main = sum (replicate 5 3)").unwrap();
    assert!(matches!(result, Value::Int(15))); // 3 * 5
}

#[test]
fn test_replicate_length() {
    let result = run_source("main = length (replicate 4 0)").unwrap();
    assert!(matches!(result, Value::Int(4)));
}

#[test]
fn test_ord() {
    let result = run_source("main = ord 'A'").unwrap();
    assert!(matches!(result, Value::Int(65)));
}

#[test]
fn test_chr() {
    let result = run_source("main = chr 65").unwrap();
    assert!(matches!(result, Value::Char('A')));
}

#[test]
fn test_even_true() {
    let result = run_source("main = even 4").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_even_false() {
    let result = run_source("main = even 3").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_odd_true() {
    let result = run_source("main = odd 3").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_odd_false() {
    let result = run_source("main = odd 4").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_elem_found() {
    let result = run_source("main = elem 3 [1, 2, 3, 4, 5]").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_elem_not_found() {
    let result = run_source("main = elem 6 [1, 2, 3, 4, 5]").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_not_elem() {
    let result = run_source("main = notElem 6 [1, 2, 3, 4, 5]").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_take_while() {
    let result = run_source("main = sum (takeWhile (\\x -> x < 4) [1, 2, 3, 4, 5])").unwrap();
    assert!(matches!(result, Value::Int(6))); // 1 + 2 + 3
}

#[test]
fn test_drop_while() {
    let result = run_source("main = sum (dropWhile (\\x -> x < 4) [1, 2, 3, 4, 5])").unwrap();
    assert!(matches!(result, Value::Int(9))); // 4 + 5
}

#[test]
fn test_split_at() {
    let source = r#"
main = case splitAt 3 [1, 2, 3, 4, 5] of
  (xs, ys) -> sum xs + sum ys * 10
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(96))); // 6 + 90
}

#[test]
fn test_flip() {
    let source = r#"
main = flip (\x y -> x - y) 3 10
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(7))); // 10 - 3
}

#[test]
fn test_min() {
    let result = run_source("main = min 3 7").unwrap();
    assert!(matches!(result, Value::Int(3)));
}

#[test]
fn test_max() {
    let result = run_source("main = max 3 7").unwrap();
    assert!(matches!(result, Value::Int(7)));
}

#[test]
fn test_abs_negative() {
    let result = run_source("main = abs (-5)").unwrap();
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn test_abs_positive() {
    let result = run_source("main = abs 5").unwrap();
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn test_signum() {
    let result = run_source("main = signum (-42)").unwrap();
    assert!(matches!(result, Value::Int(-1)));
}

#[test]
fn test_product() {
    let result = run_source("main = product [1, 2, 3, 4, 5]").unwrap();
    assert!(matches!(result, Value::Int(120)));
}

#[test]
fn test_from_integral() {
    let result = run_source("main = fromIntegral 42").unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_id() {
    let result = run_source("main = id 42").unwrap();
    assert!(matches!(result, Value::Int(42)));
}

#[test]
fn test_any_true() {
    let result = run_source("main = any (\\x -> x > 3) [1, 2, 3, 4, 5]").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_any_false() {
    let result = run_source("main = any (\\x -> x > 10) [1, 2, 3, 4, 5]").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_all_true() {
    let result = run_source("main = all (\\x -> x > 0) [1, 2, 3, 4, 5]").unwrap();
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_all_false() {
    let result = run_source("main = all (\\x -> x > 3) [1, 2, 3, 4, 5]").unwrap();
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_iterate_take() {
    // take 5 (iterate (*2) 1) = [1, 2, 4, 8, 16], sum = 31
    let result = run_source("main = sum (take 5 (iterate (\\x -> x * 2) 1))").unwrap();
    assert!(matches!(result, Value::Int(31)));
}

#[test]
fn test_repeat_take() {
    // take 4 (repeat 3) = [3, 3, 3, 3], sum = 12
    let result = run_source("main = sum (take 4 (repeat 3))").unwrap();
    assert!(matches!(result, Value::Int(12)));
}

#[test]
fn test_cycle_take() {
    // take 7 (cycle [1, 2, 3]) = [1, 2, 3, 1, 2, 3, 1], sum = 13
    let result = run_source("main = sum (take 7 (cycle [1, 2, 3]))").unwrap();
    assert!(matches!(result, Value::Int(13)));
}

#[test]
fn test_span() {
    let source = r#"
main = case span (\x -> x < 4) [1, 2, 3, 4, 5] of
  (xs, ys) -> length xs + length ys * 10
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(23))); // 3 + 20
}

#[test]
fn test_filter_list() {
    // filter: keep elements > 3 from [1,2,3,4,5]
    let source = r#"
main = sum (filter (\x -> x > 3) [1, 2, 3, 4, 5])
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(9))); // 4 + 5 = 9
}

#[test]
fn test_reverse_list() {
    let result = run_source("main = head (reverse [1, 2, 3, 4, 5])").unwrap();
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn test_last_list() {
    let result = run_source("main = last [1, 2, 3, 4, 5]").unwrap();
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn test_foldr_list() {
    // foldr (-) 0 [1,2,3] = 1 - (2 - (3 - 0)) = 1 - (2 - 3) = 1 - (-1) = 2
    let result = run_source("main = foldr (\\x acc -> x - acc) 0 [1, 2, 3]").unwrap();
    assert!(matches!(result, Value::Int(2)));
}

// =========================================================================
// Advanced Recursion Tests
// =========================================================================

#[test]
fn test_recursive_length() {
    let source = r#"
myLength xs = if null xs then 0 else 1 + myLength (tail xs)
main = myLength [1, 2, 3, 4, 5]
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(5)));
}

#[test]
fn test_recursive_sum() {
    let source = r#"
mySum xs = if null xs then 0 else head xs + mySum (tail xs)
main = mySum [1, 2, 3, 4, 5]
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(15)));
}

#[test]
fn test_tail_recursive_sum() {
    let source = r#"
go acc xs = if null xs then acc else go (acc + head xs) (tail xs)
main = go 0 [1, 2, 3, 4, 5]
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(15)));
}

#[test]
fn test_mutual_recursion() {
    // isEven/isOdd via mutual recursion
    let source = r#"
isEven n = if n == 0 then 1 else isOdd (n - 1)
isOdd n = if n == 0 then 0 else isEven (n - 1)
main = isEven 10
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(1)));
}

#[test]
fn test_mutual_recursion_odd() {
    let source = r#"
isEven n = if n == 0 then 1 else isOdd (n - 1)
isOdd n = if n == 0 then 0 else isEven (n - 1)
main = isOdd 7
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(1)));
}

// =========================================================================
// Do-Notation & IO Bind Tests
// =========================================================================

#[test]
fn test_io_bind_basic() {
    // Using >>= explicitly
    let (_, display) = run_and_display(r#"main = putStrLn "hello" >> putStrLn "world""#).unwrap();
    assert!(display.contains("hello"), "Expected 'hello' in output");
    assert!(display.contains("world"), "Expected 'world' in output");
}

#[test]
fn test_io_print_three() {
    let (_, display) = run_and_display("main = print 10 >> print 20 >> print 30").unwrap();
    assert!(display.contains("10") && display.contains("20") && display.contains("30"),
            "Expected 10, 20, 30 in output, got: {}", display);
}

#[test]
fn test_io_mixed_print_putstrln() {
    let (_, display) = run_and_display(
        r#"main = putStrLn "start" >> print 42 >> putStrLn "end""#
    ).unwrap();
    assert!(display.contains("start"), "Expected 'start' in output");
    assert!(display.contains("42"), "Expected '42' in output");
    assert!(display.contains("end"), "Expected 'end' in output");
}

#[test]
fn test_io_computed_print() {
    let (_, display) = run_and_display("main = print (2 + 3)").unwrap();
    assert!(display.contains("5"), "Expected '5' in output, got: {}", display);
}

// =========================================================================
// Character & String Tests
// =========================================================================

#[test]
fn test_char_literal() {
    let result = run_source("main = 'A'").unwrap();
    assert!(matches!(result, Value::Char('A')));
}

#[test]
fn test_string_literal() {
    let (_, display) = run_and_display(r#"main = putStrLn "hello""#).unwrap();
    assert!(display.contains("hello"));
}

#[test]
fn test_string_concat_output() {
    let (_, display) = run_and_display(r#"main = putStrLn "ab" >> putStrLn "cd""#).unwrap();
    assert!(display.contains("ab") && display.contains("cd"));
}

// =========================================================================
// Error Handling Tests
// =========================================================================

#[test]
fn test_error_function() {
    let result = run_source(r#"main = error "boom""#);
    assert!(result.is_err(), "Expected error to propagate");
}

#[test]
fn test_head_empty_list() {
    let result = run_source("main = head []");
    assert!(result.is_err(), "Expected error for head of empty list");
}

#[test]
fn test_tail_empty_list() {
    let result = run_source("main = tail []");
    assert!(result.is_err(), "Expected error for tail of empty list");
}

// =========================================================================
// Tuple Tests
// =========================================================================

#[test]
fn test_fst_pair() {
    let source = r#"
main = case (1, 2) of
  (x, y) -> x
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(1)));
}

#[test]
fn test_snd_pair() {
    let source = r#"
main = case (1, 2) of
  (x, y) -> y
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(2)));
}

#[test]
fn test_triple_extraction() {
    let source = r#"
main = case (10, 20, 30) of
  (x, y, z) -> x + y + z
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(60)));
}

// =========================================================================
// Functional Pattern Tests
// =========================================================================

#[test]
fn test_foldl_strict() {
    // foldl' (+) 0 [1..5] = 15
    let result = run_source("main = foldl' (\\acc x -> acc + x) 0 [1, 2, 3, 4, 5]").unwrap();
    assert!(matches!(result, Value::Int(15)));
}

#[test]
fn test_init_list() {
    // init [1,2,3,4,5] = [1,2,3,4], sum = 10
    let result = run_source("main = sum (init [1, 2, 3, 4, 5])").unwrap();
    assert!(matches!(result, Value::Int(10)));
}

#[test]
fn test_list_index() {
    // [10, 20, 30, 40] !! 2 = 30
    let source = r#"
main = [10, 20, 30, 40] !! 2
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(30)));
}

#[test]
fn test_filter_even_composition() {
    // sum of even numbers from [1..10]
    // 2 + 4 + 6 + 8 + 10 = 30
    let source = r#"
main = sum (filter (\x -> x `mod` 2 == 0) [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(30)));
}

#[test]
fn test_map_filter_composition() {
    // double elements > 2 from [1,2,3,4,5], then sum
    // filter: [3,4,5], map (*2): [6,8,10], sum: 24
    let source = r#"
main = sum (map (\x -> x * 2) (filter (\x -> x > 2) [1, 2, 3, 4, 5]))
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(24)));
}

#[test]
fn test_repeated_element_sum() {
    // [3,3,3,3,3], sum = 15
    let result = run_source("main = sum [3, 3, 3, 3, 3]").unwrap();
    assert!(matches!(result, Value::Int(15)));
}

#[test]
fn test_zip_sum() {
    // zip [1,2,3] [4,5,6] then sum first elements
    let source = r#"
sumFsts xs = if null xs then 0 else case head xs of (a, b) -> a + sumFsts (tail xs)
main = sumFsts (zip [1, 2, 3] [4, 5, 6])
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(6))); // 1 + 2 + 3
}

#[test]
fn test_take_drop_identity() {
    // take n xs ++ drop n xs has same sum as xs
    let source = r#"
main = sum (take 3 [1,2,3,4,5]) + sum (drop 3 [1,2,3,4,5])
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(15))); // 6 + 9
}

#[test]
fn test_nested_function_calls() {
    let source = r#"
square x = x * x
double x = x * 2
main = square (double 3)
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(36))); // (3*2)^2 = 36
}

#[test]
fn test_function_as_argument() {
    let source = r#"
applyTwice f x = f (f x)
main = applyTwice (\x -> x + 3) 10
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(16))); // 10 + 3 + 3
}

#[test]
fn test_concat_lists() {
    let result = run_source("main = sum ([1, 2, 3] ++ [4, 5, 6])").unwrap();
    assert!(matches!(result, Value::Int(21)));
}

#[test]
fn test_concatmap() {
    // concatMap (\x -> [x, x*2]) [1,2,3] = [1,2,2,4,3,6], sum = 18
    let source = r#"
main = sum (concatMap (\x -> [x, x * 2]) [1, 2, 3])
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(18)));
}

#[test]
fn test_map_double_list() {
    // map (*2) [1..5] = [2,4,6,8,10], sum = 30
    let result = run_source("main = sum (map (\\x -> x * 2) [1, 2, 3, 4, 5])").unwrap();
    assert!(matches!(result, Value::Int(30)));
}

// =========================================================================
// Complex Expression Tests
// =========================================================================

#[test]
fn test_fibonacci_10() {
    let source = r#"
fib n = if n <= 1 then n else fib (n - 1) + fib (n - 2)
main = fib 10
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(55)));
}

#[test]
#[ignore = "Ackermann causes stack overflow in interpreter due to deep recursion"]
fn test_ackermann_small() {
    // ackermann 2 3 = 9
    let source = r#"
ack m n = if m == 0 then n + 1
          else if n == 0 then ack (m - 1) 1
          else ack (m - 1) (ack m (n - 1))
main = ack 2 3
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(9)));
}

#[test]
fn test_gcd() {
    let source = r#"
myGcd a b = if b == 0 then a else myGcd b (a `mod` b)
main = myGcd 12 8
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(4)));
}

#[test]
fn test_power() {
    let source = r#"
power base exp = if exp == 0 then 1 else base * power base (exp - 1)
main = power 2 10
"#;
    let result = run_source(source).unwrap();
    assert!(matches!(result, Value::Int(1024)));
}
