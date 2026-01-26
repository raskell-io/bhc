//! # BHC Playground
//!
//! Browser-based Haskell interpreter for the BHC playground.
//!
//! This crate provides a WASM-compatible interface to the BHC frontend
//! (lexer, parser, type checker) and the Core IR evaluator. It allows
//! Haskell code to be validated and executed directly in the browser
//! without native code generation.
//!
//! ## Overview
//!
//! The playground implements the following pipeline:
//!
//! ```text
//! Source Code
//!      │
//!      ▼
//! ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
//! │  Parse  │ ──▶ │  Lower  │ ──▶ │  Type   │ ──▶ │  Core   │
//! │         │     │  (HIR)  │     │  Check  │     │  Lower  │
//! └─────────┘     └─────────┘     └─────────┘     └─────────┘
//!                                                      │
//!                                                      ▼
//!                                                 ┌─────────┐
//!                                                 │  Eval   │
//!                                                 │  (Interp)│
//!                                                 └─────────┘
//!                                                      │
//!                                                      ▼
//!                                                   Result
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use bhc_playground::{compile_and_run, PlaygroundResult};
//!
//! let result = compile_and_run("main = 42");
//! match result {
//!     Ok(output) => println!("Result: {}", output.display),
//!     Err(e) => eprintln!("Error: {}", e.message),
//! }
//! ```
//!
//! ## WASM Usage
//!
//! When compiled to WASM with wasm-bindgen, the API is exposed as:
//!
//! ```javascript
//! import init, { run_haskell } from 'bhc_playground';
//!
//! await init();
//! const result = run_haskell("main = 42");
//! console.log(result); // JSON: { "success": true, "display": "42", ... }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

use bhc_ast::Module as AstModule;
use bhc_core::eval::{Env, Evaluator, Value};
use bhc_core::{Bind, CoreModule, Expr, VarId};
use bhc_hir::Module as HirModule;
use bhc_intern::Symbol;
use bhc_lower::LowerContext;
use bhc_session::Profile;
use bhc_span::FileId;
use bhc_typeck::TypedModule;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// WASM setup for panics and allocation
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_init() {
    console_error_panic_hook::set_once();
}

/// Errors that can occur during playground execution.
#[derive(Debug, Error)]
pub enum PlaygroundError {
    /// Parse error occurred.
    #[error("parse error: {message}")]
    ParseError {
        /// Error message.
        message: String,
        /// Line number (1-indexed).
        line: Option<u32>,
        /// Column number (1-indexed).
        column: Option<u32>,
    },

    /// Type checking failed.
    #[error("type error: {message}")]
    TypeError {
        /// Error message.
        message: String,
        /// Line number (1-indexed).
        line: Option<u32>,
        /// Column number (1-indexed).
        column: Option<u32>,
    },

    /// Lowering failed.
    #[error("lowering error: {0}")]
    LowerError(String),

    /// Core lowering failed.
    #[error("core lowering error: {0}")]
    CoreLowerError(String),

    /// Evaluation failed.
    #[error("runtime error: {0}")]
    RuntimeError(String),

    /// Main function not found.
    #[error("main function not found")]
    NoMainFunction,
}

/// Result of successful playground execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundOutput {
    /// Human-readable display of the result.
    pub display: String,
    /// Type of the result (if known).
    pub result_type: Option<String>,
    /// Whether the result is a function (not fully evaluated).
    pub is_function: bool,
}

/// Serializable error for WASM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundErrorInfo {
    /// Error message.
    pub message: String,
    /// Error category.
    pub category: String,
    /// Line number (1-indexed, if available).
    pub line: Option<u32>,
    /// Column number (1-indexed, if available).
    pub column: Option<u32>,
}

/// Result type that can be serialized to JSON for WASM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundResult {
    /// Whether execution succeeded.
    pub success: bool,
    /// Output if successful.
    pub output: Option<PlaygroundOutput>,
    /// Error if failed.
    pub error: Option<PlaygroundErrorInfo>,
}

impl From<PlaygroundError> for PlaygroundErrorInfo {
    fn from(err: PlaygroundError) -> Self {
        let (message, category, line, column) = match err {
            PlaygroundError::ParseError { message, line, column } => {
                (message, "parse".to_string(), line, column)
            }
            PlaygroundError::TypeError { message, line, column } => {
                (message, "type".to_string(), line, column)
            }
            PlaygroundError::LowerError(msg) => (msg, "lower".to_string(), None, None),
            PlaygroundError::CoreLowerError(msg) => (msg, "core_lower".to_string(), None, None),
            PlaygroundError::RuntimeError(msg) => (msg, "runtime".to_string(), None, None),
            PlaygroundError::NoMainFunction => {
                ("main function not found".to_string(), "no_main".to_string(), None, None)
            }
        };
        Self { message, category, line, column }
    }
}

/// Compile and run Haskell source code.
///
/// This function takes Haskell source code, compiles it through all frontend
/// phases, and evaluates it using the interpreter. No native code generation
/// is performed, making this suitable for browser-based execution.
///
/// # Arguments
///
/// * `source` - Haskell source code to compile and run
///
/// # Returns
///
/// Returns `Ok(PlaygroundOutput)` if compilation and execution succeed,
/// or `Err(PlaygroundError)` with details about what went wrong.
///
/// # Example
///
/// ```ignore
/// let result = compile_and_run("main = 42");
/// assert!(result.is_ok());
/// assert_eq!(result.unwrap().display, "42");
/// ```
pub fn compile_and_run(source: &str) -> Result<PlaygroundOutput, PlaygroundError> {
    compile_and_run_with_profile(source, Profile::Default)
}

/// Compile and run Haskell source code with a specific profile.
///
/// # Arguments
///
/// * `source` - Haskell source code to compile and run
/// * `profile` - Runtime profile (affects evaluation semantics)
///
/// # Returns
///
/// Returns `Ok(PlaygroundOutput)` if compilation and execution succeed.
pub fn compile_and_run_with_profile(
    source: &str,
    profile: Profile,
) -> Result<PlaygroundOutput, PlaygroundError> {
    let interpreter = Interpreter::new(profile);
    interpreter.run(source)
}

/// The playground interpreter.
///
/// This encapsulates the compilation and execution pipeline.
pub struct Interpreter {
    profile: Profile,
}

impl Interpreter {
    /// Create a new interpreter with the given profile.
    #[must_use]
    pub fn new(profile: Profile) -> Self {
        Self { profile }
    }

    /// Create an interpreter with the default (lazy) profile.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(Profile::Default)
    }

    /// Create an interpreter with the numeric (strict) profile.
    #[must_use]
    pub fn numeric() -> Self {
        Self::new(Profile::Numeric)
    }

    /// Run source code and return the result.
    pub fn run(&self, source: &str) -> Result<PlaygroundOutput, PlaygroundError> {
        // Phase 1: Parse
        let file_id = FileId::new(0);
        let ast = self.parse(source, file_id)?;

        // Phase 2: Lower AST to HIR
        let (hir, lower_ctx) = self.lower(&ast)?;

        // Phase 3: Type check
        let typed = self.type_check(&hir, file_id, &lower_ctx)?;

        // Phase 4: Lower to Core IR
        let core = self.core_lower(&hir, &lower_ctx, &typed)?;

        // Phase 5: Evaluate
        self.evaluate(&core)
    }

    /// Parse source code into an AST.
    fn parse(&self, source: &str, file_id: FileId) -> Result<AstModule, PlaygroundError> {
        let (maybe_module, diagnostics) = bhc_parser::parse_module(source, file_id);

        match maybe_module {
            Some(module) => Ok(module),
            None => {
                // Return the first error with location info if available
                let (message, line, column) = if let Some(diag) = diagnostics.first() {
                    // Try to extract location from diagnostic
                    (diag.message.clone(), None, None)
                } else {
                    ("parse error".to_string(), None, None)
                };

                Err(PlaygroundError::ParseError { message, line, column })
            }
        }
    }

    /// Lower AST to HIR.
    fn lower(&self, ast: &AstModule) -> Result<(HirModule, LowerContext), PlaygroundError> {
        let mut ctx = LowerContext::with_builtins();

        let config = bhc_lower::LowerConfig {
            include_builtins: true,
            warn_unused: false,
            search_paths: vec![],
        };

        let hir = bhc_lower::lower_module(&mut ctx, ast, &config)
            .map_err(|e| PlaygroundError::LowerError(e.to_string()))?;

        if ctx.has_errors() {
            let errors = ctx.take_errors();
            let msg = errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(PlaygroundError::LowerError(msg));
        }

        Ok((hir, ctx))
    }

    /// Type check HIR.
    fn type_check(
        &self,
        hir: &HirModule,
        file_id: FileId,
        lower_ctx: &LowerContext,
    ) -> Result<TypedModule, PlaygroundError> {
        match bhc_typeck::type_check_module_with_defs(hir, file_id, Some(&lower_ctx.defs)) {
            Ok(typed) => Ok(typed),
            Err(diagnostics) => {
                let (message, line, column) = if let Some(diag) = diagnostics.first() {
                    (diag.message.clone(), None, None)
                } else {
                    ("type error".to_string(), None, None)
                };

                Err(PlaygroundError::TypeError { message, line, column })
            }
        }
    }

    /// Lower HIR to Core IR.
    fn core_lower(
        &self,
        hir: &HirModule,
        lower_ctx: &LowerContext,
        typed: &TypedModule,
    ) -> Result<CoreModule, PlaygroundError> {
        // Convert lower context's DefMap to hir-to-core's DefMap
        let def_map: bhc_hir_to_core::DefMap = lower_ctx
            .defs
            .iter()
            .map(|(def_id, def_info)| {
                (
                    *def_id,
                    bhc_hir_to_core::DefInfo {
                        id: *def_id,
                        name: def_info.name,
                    },
                )
            })
            .collect();

        bhc_hir_to_core::lower_module_with_defs(hir, Some(&def_map), Some(&typed.def_schemes))
            .map_err(|e| PlaygroundError::CoreLowerError(e.to_string()))
    }

    /// Evaluate Core IR and return the result.
    fn evaluate(&self, module: &CoreModule) -> Result<PlaygroundOutput, PlaygroundError> {
        let evaluator = Evaluator::with_profile(self.profile);

        // Build environment from module bindings
        let env = self.build_module_env(module, &evaluator)?;

        // Find and evaluate main
        let main_name = Symbol::intern("main");
        let main_expr = self
            .find_main_binding(module, main_name)
            .ok_or(PlaygroundError::NoMainFunction)?;

        let result = evaluator
            .eval(&main_expr, &env)
            .map_err(|e| PlaygroundError::RuntimeError(e.to_string()))?;

        // Force the result to WHNF
        let forced = evaluator
            .force(result)
            .map_err(|e| PlaygroundError::RuntimeError(e.to_string()))?;

        // Generate display string
        let display = evaluator
            .display_value(&forced)
            .map_err(|e| PlaygroundError::RuntimeError(e.to_string()))?;

        // Check if it's a function
        let is_function = matches!(forced, Value::Closure(_) | Value::PrimOp(_) | Value::PartialPrimOp(_, _));

        Ok(PlaygroundOutput {
            display,
            result_type: None, // TODO: track type info
            is_function,
        })
    }

    /// Build environment from module bindings.
    fn build_module_env(
        &self,
        module: &CoreModule,
        evaluator: &Evaluator,
    ) -> Result<Env, PlaygroundError> {
        use bhc_core::eval::Thunk;

        // Collect all bindings
        let mut all_bindings: Vec<(VarId, Box<Expr>)> = Vec::new();

        for bind in &module.bindings {
            match bind {
                Bind::NonRec(var, rhs) => {
                    all_bindings.push((var.id, rhs.clone()));
                }
                Bind::Rec(bindings) => {
                    for (var, rhs) in bindings {
                        all_bindings.push((var.id, rhs.clone()));
                    }
                }
            }
        }

        // Evaluate lambda bindings with empty env
        let empty_env = Env::new();
        let mut module_env = Env::new();

        for (var_id, rhs) in &all_bindings {
            if self.is_lambda_expr(rhs) {
                let value = evaluator
                    .eval(rhs, &empty_env)
                    .map_err(|e| PlaygroundError::RuntimeError(e.to_string()))?;
                module_env = module_env.extend(*var_id, value);
            }
        }

        // Set module env for recursive lookups
        evaluator.set_module_env(module_env.clone());

        // Add non-lambda bindings as thunks
        for (var_id, rhs) in &all_bindings {
            if !self.is_lambda_expr(rhs) {
                let thunk = Value::Thunk(Thunk {
                    expr: rhs.clone(),
                    env: empty_env.clone(),
                });
                module_env = module_env.extend(*var_id, thunk);
            }
        }

        evaluator.set_module_env(module_env.clone());

        Ok(module_env)
    }

    /// Check if an expression is a lambda.
    fn is_lambda_expr(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Lam(_, _, _))
    }

    /// Find the main binding in a module.
    fn find_main_binding(&self, module: &CoreModule, name: Symbol) -> Option<Expr> {
        for bind in &module.bindings {
            match bind {
                Bind::NonRec(var, rhs) if var.name == name => {
                    return Some((**rhs).clone());
                }
                Bind::Rec(bindings) => {
                    for (var, rhs) in bindings {
                        if var.name == name {
                            return Some((**rhs).clone());
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ============================================================================
// WASM API
// ============================================================================

/// Run Haskell code and return JSON result.
///
/// This is the main entry point for WASM usage.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn run_haskell(source: &str) -> String {
    let result = match compile_and_run(source) {
        Ok(output) => PlaygroundResult {
            success: true,
            output: Some(output),
            error: None,
        },
        Err(e) => PlaygroundResult {
            success: false,
            output: None,
            error: Some(e.into()),
        },
    };

    serde_json::to_string(&result).unwrap_or_else(|_| {
        r#"{"success":false,"error":{"message":"serialization failed","category":"internal"}}"#
            .to_string()
    })
}

/// Run Haskell code with numeric (strict) profile.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn run_haskell_numeric(source: &str) -> String {
    let result = match compile_and_run_with_profile(source, Profile::Numeric) {
        Ok(output) => PlaygroundResult {
            success: true,
            output: Some(output),
            error: None,
        },
        Err(e) => PlaygroundResult {
            success: false,
            output: None,
            error: Some(e.into()),
        },
    };

    serde_json::to_string(&result).unwrap_or_else(|_| {
        r#"{"success":false,"error":{"message":"serialization failed","category":"internal"}}"#
            .to_string()
    })
}

/// Check Haskell code for errors without running it.
///
/// Returns JSON with any parse or type errors found.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn check_haskell(source: &str) -> String {
    let interpreter = Interpreter::with_defaults();
    let file_id = FileId::new(0);

    // Parse
    let ast = match interpreter.parse(source, file_id) {
        Ok(ast) => ast,
        Err(e) => {
            let result = PlaygroundResult {
                success: false,
                output: None,
                error: Some(e.into()),
            };
            return serde_json::to_string(&result).unwrap_or_default();
        }
    };

    // Lower
    let (hir, lower_ctx) = match interpreter.lower(&ast) {
        Ok(r) => r,
        Err(e) => {
            let result = PlaygroundResult {
                success: false,
                output: None,
                error: Some(e.into()),
            };
            return serde_json::to_string(&result).unwrap_or_default();
        }
    };

    // Type check
    match interpreter.type_check(&hir, file_id, &lower_ctx) {
        Ok(_) => {
            let result = PlaygroundResult {
                success: true,
                output: Some(PlaygroundOutput {
                    display: "OK".to_string(),
                    result_type: None,
                    is_function: false,
                }),
                error: None,
            };
            serde_json::to_string(&result).unwrap_or_default()
        }
        Err(e) => {
            let result = PlaygroundResult {
                success: false,
                output: None,
                error: Some(e.into()),
            };
            serde_json::to_string(&result).unwrap_or_default()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_and_run_literal() {
        let result = compile_and_run("main = 42");
        assert!(result.is_ok(), "Should compile: {:?}", result.err());
        assert_eq!(result.unwrap().display, "42");
    }

    #[test]
    fn test_compile_and_run_arithmetic() {
        let result = compile_and_run("main = 2 + 3 * 4");
        assert!(result.is_ok(), "Should compile: {:?}", result.err());
        assert_eq!(result.unwrap().display, "14");
    }

    #[test]
    fn test_compile_and_run_let() {
        let result = compile_and_run("main = let x = 10 in x + 5");
        assert!(result.is_ok(), "Should compile: {:?}", result.err());
        assert_eq!(result.unwrap().display, "15");
    }

    #[test]
    fn test_compile_and_run_list() {
        let result = compile_and_run("main = [1, 2, 3]");
        assert!(result.is_ok(), "Should compile: {:?}", result.err());
        assert_eq!(result.unwrap().display, "[1, 2, 3]");
    }

    #[test]
    fn test_parse_error() {
        // Use syntax that definitely fails to parse
        let result = compile_and_run("main = (((");
        assert!(result.is_err(), "Should fail to parse");
        // Accept any error type - parse errors might surface as type or lower errors
        // depending on error recovery
    }

    #[test]
    fn test_no_main() {
        let result = compile_and_run("foo = 42");
        assert!(result.is_err(), "Should fail without main");
        match result.unwrap_err() {
            PlaygroundError::NoMainFunction => {}
            other => panic!("Expected NoMainFunction, got: {:?}", other),
        }
    }

    #[test]
    fn test_numeric_profile() {
        let result = compile_and_run_with_profile("main = 42", Profile::Numeric);
        assert!(result.is_ok(), "Should compile with numeric: {:?}", result.err());
        assert_eq!(result.unwrap().display, "42");
    }

    #[test]
    fn test_interpreter_api() {
        let interpreter = Interpreter::with_defaults();
        let result = interpreter.run("main = 42");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().display, "42");
    }

    #[test]
    fn test_recursive_function() {
        // Test recursion using let-rec within main
        let result = compile_and_run(
            "main = let factorial n = if n <= 1 then 1 else n * factorial (n - 1) in factorial 5",
        );
        assert!(result.is_ok(), "Should compile: {:?}", result.err());
        assert_eq!(result.unwrap().display, "120");
    }

    // =========================================================================
    // IO Sequencing Tests (>> operator)
    // =========================================================================
    //
    // NOTE: IO actions return (), so the display value is "()" not the printed output.
    // The printed output goes to stdout (visible in test output when running with --nocapture).
    // These tests verify that type checking passes for multi-statement IO chains.

    #[test]
    fn test_io_sequence_two_statements() {
        // Two IO actions chained with >>
        let result = compile_and_run("main = print 1 >> print 2");
        assert!(result.is_ok(), "Two IO actions should type check: {:?}", result.err());
        // IO () returns "()" - the printed values go to stdout
        assert_eq!(result.unwrap().display, "()", "IO action should return unit");
    }

    #[test]
    fn test_io_sequence_three_statements() {
        // Three IO actions chained with >> (this was failing before with type error)
        let result = compile_and_run("main = print 1 >> print 2 >> print 3");
        assert!(result.is_ok(), "Three IO actions should type check: {:?}", result.err());
        assert_eq!(result.unwrap().display, "()", "IO action should return unit");
    }

    #[test]
    fn test_io_sequence_four_statements() {
        // Four IO actions chained with >>
        let result = compile_and_run("main = print 1 >> print 2 >> print 3 >> print 4");
        assert!(result.is_ok(), "Four IO actions should type check: {:?}", result.err());
        assert_eq!(result.unwrap().display, "()", "IO action should return unit");
    }

    #[test]
    fn test_io_sequence_five_statements() {
        // Five IO actions chained with >>
        let result = compile_and_run("main = print 1 >> print 2 >> print 3 >> print 4 >> print 5");
        assert!(result.is_ok(), "Five IO actions should type check: {:?}", result.err());
        assert_eq!(result.unwrap().display, "()", "IO action should return unit");
    }

    #[test]
    fn test_io_sequence_with_putstrln() {
        // Mix of putStrLn calls chained
        let result = compile_and_run(r#"main = putStrLn "a" >> putStrLn "b" >> putStrLn "c""#);
        assert!(result.is_ok(), "putStrLn chain should type check: {:?}", result.err());
        assert_eq!(result.unwrap().display, "()", "IO action should return unit");
    }

    #[test]
    fn test_io_sequence_explicit_parens_right() {
        // Explicit right-associative parentheses
        let result = compile_and_run("main = print 1 >> (print 2 >> print 3)");
        assert!(result.is_ok(), "Right-paren chain should type check: {:?}", result.err());
        assert_eq!(result.unwrap().display, "()", "IO action should return unit");
    }

    #[test]
    fn test_io_sequence_explicit_parens_left() {
        // Explicit left-associative parentheses
        let result = compile_and_run("main = (print 1 >> print 2) >> print 3");
        assert!(result.is_ok(), "Left-paren chain should type check: {:?}", result.err());
        assert_eq!(result.unwrap().display, "()", "IO action should return unit");
    }
}
