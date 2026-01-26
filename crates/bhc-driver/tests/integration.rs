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
