//! # HIR to Core Lowering
//!
//! This crate transforms typed HIR (High-Level IR) into Core IR, the main
//! intermediate representation used for optimization.
//!
//! ## Key Transformations
//!
//! - **Pattern compilation**: Multi-argument lambdas and pattern matching
//!   are compiled into explicit case expressions
//! - **Binding analysis**: Let bindings are analyzed for mutual recursion
//! - **Guard expansion**: Pattern guards become nested conditionals
//! - **Type erasure**: Type annotations are preserved but simplified

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

mod binding;
mod context;
mod expr;
mod pattern;

use bhc_core::CoreModule;
use bhc_hir::{DefId, Module as HirModule};
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::Scheme;
use indexmap::IndexMap;
use rustc_hash::FxHashMap;
use thiserror::Error;

pub use context::LowerContext;

/// Information about a definition from the lowering pass.
/// This mirrors `bhc_lower::context::DefInfo`.
#[derive(Clone, Debug)]
pub struct DefInfo {
    /// The unique ID.
    pub id: DefId,
    /// The name.
    pub name: Symbol,
}

/// Map from `DefId` to definition information.
pub type DefMap = IndexMap<DefId, DefInfo>;

/// Map from `DefId` to type scheme (from type checker).
pub type TypeSchemeMap = FxHashMap<DefId, Scheme>;

/// Errors that can occur during HIR to Core lowering.
#[derive(Debug, Error)]
pub enum LowerError {
    /// An internal invariant was violated.
    #[error("internal error: {0}")]
    Internal(String),

    /// Pattern compilation failed.
    #[error("pattern compilation failed at {span:?}: {message}")]
    PatternError {
        /// Error message.
        message: String,
        /// Source location.
        span: Span,
    },

    /// Multiple errors occurred.
    #[error("multiple errors")]
    Multiple(Vec<LowerError>),
}

/// Result type for lowering operations.
pub type LowerResult<T> = Result<T, LowerError>;

/// Lower a HIR module to Core IR.
///
/// This is the main entry point for the HIR to Core transformation.
///
/// # Arguments
///
/// * `module` - The typed HIR module to lower
///
/// # Returns
///
/// A `CoreModule` containing the lowered bindings.
///
/// # Errors
///
/// Returns `LowerError` if lowering fails due to internal errors or
/// unsupported constructs.
pub fn lower_module(module: &HirModule) -> LowerResult<CoreModule> {
    lower_module_with_defs(module, None, None)
}

/// Lower a HIR module to Core IR with definition mappings from the lowering pass.
///
/// This function accepts the DefMap from the lowering context, which allows
/// the Core lowering to register builtins with the correct DefIds assigned
/// during the AST-to-HIR lowering pass.
///
/// # Arguments
///
/// * `module` - The typed HIR module to lower
/// * `defs` - Optional definition map from the lowering context
/// * `type_schemes` - Optional type schemes from the type checker
///
/// # Errors
///
/// Returns `LowerError` if lowering fails due to internal errors or
/// unsupported constructs.
pub fn lower_module_with_defs(
    module: &HirModule,
    defs: Option<&DefMap>,
    type_schemes: Option<&TypeSchemeMap>,
) -> LowerResult<CoreModule> {
    let mut ctx = LowerContext::new();

    // If we have definition mappings from the lowering pass, use them
    // to register builtins with the correct DefIds
    if let Some(def_map) = defs {
        ctx.register_lowered_builtins(def_map);
    }

    // If we have type schemes from the type checker, use them
    if let Some(schemes) = type_schemes {
        ctx.set_type_schemes(schemes.clone());
    }

    ctx.lower_module(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_hir::{Equation, Expr, Item, Lit, Pat, ValueDef};
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_types::{Constraint, Kind, TyCon, Ty, TyVar};

    #[test]
    fn test_lower_empty_module() {
        let module = HirModule {
            name: Symbol::intern("Test"),
            exports: None,
            imports: vec![],
            items: vec![],
            span: Span::default(),
        };

        let result = lower_module(&module);
        assert!(result.is_ok());
        let core_module = result.unwrap();
        assert!(core_module.bindings.is_empty());
    }

    #[test]
    fn test_constrained_function_gets_dict_lambdas() {
        // Create a constrained function: f :: Num a => a -> a
        // f x = x
        let f_def_id = DefId::new(100);
        let x_def_id = DefId::new(101);

        let num_class = Symbol::intern("Num");
        let a_var = TyVar::new(0, Kind::Star); // Use numeric ID

        // Create a type scheme with Num constraint
        let f_scheme = Scheme {
            vars: vec![a_var.clone()],
            constraints: vec![Constraint::new(
                num_class,
                Ty::Var(a_var.clone()),
                Span::default(),
            )],
            ty: Ty::Fun(
                Box::new(Ty::Var(a_var.clone())),
                Box::new(Ty::Var(a_var)),
            ),
        };

        // Create the HIR value definition
        let value_def = ValueDef {
            id: f_def_id,
            name: Symbol::intern("f"),
            sig: Some(f_scheme.clone()),
            equations: vec![Equation {
                pats: vec![Pat::Var(Symbol::intern("x"), x_def_id, Span::default())],
                guards: vec![],
                rhs: Expr::Var(bhc_hir::DefRef {
                    def_id: x_def_id,
                    span: Span::default(),
                }),
                span: Span::default(),
            }],
            span: Span::default(),
        };

        let module = HirModule {
            name: Symbol::intern("Test"),
            exports: None,
            imports: vec![],
            items: vec![Item::Value(value_def)],
            span: Span::default(),
        };

        // Set up type schemes
        let mut type_schemes = TypeSchemeMap::default();
        type_schemes.insert(f_def_id, f_scheme);

        let result = lower_module_with_defs(&module, None, Some(&type_schemes));
        assert!(result.is_ok());
        let core_module = result.unwrap();

        // We should have exactly one binding
        assert_eq!(core_module.bindings.len(), 1);

        // The binding should be for `f`
        let bind = &core_module.bindings[0];
        match bind {
            bhc_core::Bind::NonRec(var, expr) => {
                assert_eq!(var.name.as_str(), "f");

                // The expression should be a lambda (for the dictionary parameter)
                match expr.as_ref() {
                    bhc_core::Expr::Lam(dict_var, body, _) => {
                        // The dictionary parameter should have a name starting with $d
                        assert!(
                            dict_var.name.as_str().starts_with("$dNum"),
                            "Expected dictionary parameter starting with $dNum, got: {}",
                            dict_var.name.as_str()
                        );

                        // The body should be another lambda (for x)
                        match body.as_ref() {
                            bhc_core::Expr::Lam(x_var, _, _) => {
                                assert!(
                                    x_var.name.as_str().starts_with("arg")
                                        || x_var.name.as_str() == "x",
                                    "Expected argument named 'arg0' or 'x', got: {}",
                                    x_var.name.as_str()
                                );
                            }
                            _ => panic!("Expected inner lambda for x, got: {:?}", body),
                        }
                    }
                    _ => panic!("Expected outer lambda for dictionary, got: {:?}", expr),
                }
            }
            _ => panic!("Expected NonRec binding"),
        }
    }

    #[test]
    fn test_dict_passing_at_call_site() {
        // Create two constrained functions:
        // double :: Num a => a -> a
        // double x = x + x
        //
        // quadruple :: Num a => a -> a
        // quadruple x = double (double x)
        //
        // When we lower `quadruple`, the calls to `double` should pass
        // the dictionary variable from `quadruple`.

        let double_def_id = DefId::new(200);
        let quadruple_def_id = DefId::new(201);
        let x_def_id = DefId::new(202);
        let y_def_id = DefId::new(203);

        let num_class = Symbol::intern("Num");
        let a_var = TyVar::new(0, Kind::Star); // Use numeric ID

        // Create type scheme with Num constraint
        let num_a_to_a = Scheme {
            vars: vec![a_var.clone()],
            constraints: vec![Constraint::new(
                num_class,
                Ty::Var(a_var.clone()),
                Span::default(),
            )],
            ty: Ty::Fun(
                Box::new(Ty::Var(a_var.clone())),
                Box::new(Ty::Var(a_var)),
            ),
        };

        // double x = x (simplified - just returns x)
        let double_def = ValueDef {
            id: double_def_id,
            name: Symbol::intern("double"),
            sig: Some(num_a_to_a.clone()),
            equations: vec![Equation {
                pats: vec![Pat::Var(Symbol::intern("x"), x_def_id, Span::default())],
                guards: vec![],
                rhs: Expr::Var(bhc_hir::DefRef {
                    def_id: x_def_id,
                    span: Span::default(),
                }),
                span: Span::default(),
            }],
            span: Span::default(),
        };

        // quadruple y = double y (simplified - just one call to double)
        let quadruple_def = ValueDef {
            id: quadruple_def_id,
            name: Symbol::intern("quadruple"),
            sig: Some(num_a_to_a.clone()),
            equations: vec![Equation {
                pats: vec![Pat::Var(Symbol::intern("y"), y_def_id, Span::default())],
                guards: vec![],
                rhs: Expr::App(
                    Box::new(Expr::Var(bhc_hir::DefRef {
                        def_id: double_def_id,
                        span: Span::default(),
                    })),
                    Box::new(Expr::Var(bhc_hir::DefRef {
                        def_id: y_def_id,
                        span: Span::default(),
                    })),
                    Span::default(),
                ),
                span: Span::default(),
            }],
            span: Span::default(),
        };

        let module = HirModule {
            name: Symbol::intern("Test"),
            exports: None,
            imports: vec![],
            items: vec![Item::Value(double_def), Item::Value(quadruple_def)],
            span: Span::default(),
        };

        // Set up type schemes
        let mut type_schemes = TypeSchemeMap::default();
        type_schemes.insert(double_def_id, num_a_to_a.clone());
        type_schemes.insert(quadruple_def_id, num_a_to_a);

        let result = lower_module_with_defs(&module, None, Some(&type_schemes));
        assert!(result.is_ok());
        let core_module = result.unwrap();

        // We should have two bindings
        assert_eq!(core_module.bindings.len(), 2);

        // Find the quadruple binding
        let quadruple_bind = core_module.bindings.iter().find(|b| match b {
            bhc_core::Bind::NonRec(var, _) => var.name.as_str() == "quadruple",
            _ => false,
        });
        assert!(quadruple_bind.is_some(), "Should have quadruple binding");

        // The quadruple function should have the structure:
        // \$dNum -> \y -> (double $dNum) y
        // where `double $dNum` shows the dictionary being passed
        if let bhc_core::Bind::NonRec(_, expr) = quadruple_bind.unwrap() {
            // Check it's a lambda with dict param
            if let bhc_core::Expr::Lam(dict_var, body, _) = expr.as_ref() {
                assert!(
                    dict_var.name.as_str().starts_with("$dNum"),
                    "Outer lambda should bind dictionary"
                );

                // Inside should be another lambda for y
                if let bhc_core::Expr::Lam(_, inner_body, _) = body.as_ref() {
                    // The inner body should contain an application of double with a dict
                    // Let's just check that the core structure is built
                    fn count_apps(e: &bhc_core::Expr) -> usize {
                        match e {
                            bhc_core::Expr::App(f, x, _) => 1 + count_apps(f) + count_apps(x),
                            bhc_core::Expr::Case(scrut, alts, _, _) => {
                                count_apps(scrut)
                                    + alts.iter().map(|a| count_apps(&a.rhs)).sum::<usize>()
                            }
                            bhc_core::Expr::Lam(_, b, _) => count_apps(b),
                            bhc_core::Expr::Let(bind, body, _) => {
                                let bind_apps = match bind.as_ref() {
                                    bhc_core::Bind::NonRec(_, e) => count_apps(e),
                                    bhc_core::Bind::Rec(pairs) => {
                                        pairs.iter().map(|(_, e)| count_apps(e)).sum()
                                    }
                                };
                                bind_apps + count_apps(body)
                            }
                            _ => 0,
                        }
                    }

                    let apps = count_apps(inner_body);
                    // We should have at least 2 applications:
                    // 1. double applied to dictionary
                    // 2. (double dict) applied to y
                    assert!(
                        apps >= 2,
                        "Expected at least 2 applications for dict passing, got {}",
                        apps
                    );
                } else {
                    panic!("Expected inner lambda for y parameter");
                }
            } else {
                panic!("Expected outer lambda for dictionary parameter");
            }
        }
    }
}
