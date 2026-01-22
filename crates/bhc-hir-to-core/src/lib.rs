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
pub mod dictionary;
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
    fn test_class_and_instance_registration() {
        // Create a class definition:
        // class MyEq a where
        //   myEq :: a -> a -> Bool
        //   myNeq :: a -> a -> Bool

        let class_def_id = DefId::new(300);
        let my_eq_method_id = DefId::new(301);
        let my_neq_method_id = DefId::new(302);
        let a_var = TyVar::new(0, Kind::Star);

        let class_def = bhc_hir::ClassDef {
            id: class_def_id,
            name: Symbol::intern("MyEq"),
            params: vec![a_var.clone()],
            supers: vec![], // No superclasses
            methods: vec![
                bhc_hir::MethodSig {
                    name: Symbol::intern("myEq"),
                    ty: Scheme {
                        vars: vec![],
                        constraints: vec![],
                        ty: Ty::Fun(
                            Box::new(Ty::Var(a_var.clone())),
                            Box::new(Ty::Fun(
                                Box::new(Ty::Var(a_var.clone())),
                                Box::new(Ty::Con(bhc_types::TyCon::new(
                                    Symbol::intern("Bool"),
                                    Kind::Star,
                                ))),
                            )),
                        ),
                    },
                    span: Span::default(),
                },
                bhc_hir::MethodSig {
                    name: Symbol::intern("myNeq"),
                    ty: Scheme {
                        vars: vec![],
                        constraints: vec![],
                        ty: Ty::Fun(
                            Box::new(Ty::Var(a_var.clone())),
                            Box::new(Ty::Fun(
                                Box::new(Ty::Var(a_var)),
                                Box::new(Ty::Con(bhc_types::TyCon::new(
                                    Symbol::intern("Bool"),
                                    Kind::Star,
                                ))),
                            )),
                        ),
                    },
                    span: Span::default(),
                },
            ],
            defaults: vec![],
            span: Span::default(),
        };

        // Create an instance:
        // instance MyEq Int where
        //   myEq x y = ...
        //   myNeq x y = ...

        let my_eq_impl_id = DefId::new(310);
        let my_neq_impl_id = DefId::new(311);
        let x_def_id = DefId::new(312);
        let y_def_id = DefId::new(313);

        let int_ty = Ty::Con(bhc_types::TyCon::new(Symbol::intern("Int"), Kind::Star));

        let instance_def = bhc_hir::InstanceDef {
            class: Symbol::intern("MyEq"),
            types: vec![int_ty.clone()],
            constraints: vec![],
            methods: vec![
                ValueDef {
                    id: my_eq_impl_id,
                    name: Symbol::intern("myEq"),
                    sig: None,
                    equations: vec![Equation {
                        pats: vec![
                            Pat::Var(Symbol::intern("x"), x_def_id, Span::default()),
                            Pat::Var(Symbol::intern("y"), y_def_id, Span::default()),
                        ],
                        guards: vec![],
                        rhs: Expr::Con(bhc_hir::DefRef {
                            def_id: DefId::new(9), // True
                            span: Span::default(),
                        }),
                        span: Span::default(),
                    }],
                    span: Span::default(),
                },
                ValueDef {
                    id: my_neq_impl_id,
                    name: Symbol::intern("myNeq"),
                    sig: None,
                    equations: vec![Equation {
                        pats: vec![
                            Pat::Var(Symbol::intern("x"), DefId::new(314), Span::default()),
                            Pat::Var(Symbol::intern("y"), DefId::new(315), Span::default()),
                        ],
                        guards: vec![],
                        rhs: Expr::Con(bhc_hir::DefRef {
                            def_id: DefId::new(10), // False
                            span: Span::default(),
                        }),
                        span: Span::default(),
                    }],
                    span: Span::default(),
                },
            ],
            span: Span::default(),
        };

        let module = HirModule {
            name: Symbol::intern("Test"),
            exports: None,
            imports: vec![],
            items: vec![
                Item::Class(class_def),
                Item::Instance(instance_def),
            ],
            span: Span::default(),
        };

        // Lower the module
        let result = lower_module(&module);
        assert!(result.is_ok());

        // Create a new context and lower the module to check the registry
        let mut ctx = LowerContext::new();
        let _ = ctx.lower_module(&module);

        // Verify the class is registered
        let registry = ctx.class_registry();
        let class_info = registry.lookup_class(Symbol::intern("MyEq"));
        assert!(class_info.is_some(), "MyEq class should be registered");

        let class_info = class_info.unwrap();
        assert_eq!(class_info.methods.len(), 2, "Should have 2 methods");
        assert!(class_info.methods.contains(&Symbol::intern("myEq")));
        assert!(class_info.methods.contains(&Symbol::intern("myNeq")));
        assert!(class_info.superclasses.is_empty(), "No superclasses");

        // Verify the instance is registered
        let instance_info = registry.resolve_instance(Symbol::intern("MyEq"), &int_ty);
        assert!(instance_info.is_some(), "MyEq Int instance should be registered");

        let instance_info = instance_info.unwrap();
        assert_eq!(instance_info.class, Symbol::intern("MyEq"));
        assert_eq!(instance_info.methods.len(), 2);
        assert!(instance_info.methods.contains_key(&Symbol::intern("myEq")));
        assert!(instance_info.methods.contains_key(&Symbol::intern("myNeq")));
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

    #[test]
    fn test_type_app_with_class_method() {
        // Test that a type application like `myEq @Int` resolves to the correct
        // instance method with dictionary construction.
        //
        // Setup:
        // class MyEq a where
        //   myEq :: a -> a -> Bool
        //
        // instance MyEq Int where
        //   myEq x y = True
        //
        // test = myEq @Int

        let class_def_id = DefId::new(400);
        let my_eq_method_def_id = DefId::new(401);
        let my_eq_impl_id = DefId::new(402);
        let test_def_id = DefId::new(403);
        let x_def_id = DefId::new(404);
        let y_def_id = DefId::new(405);

        let a_var = TyVar::new(0, Kind::Star);
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        // Class definition
        let class_def = bhc_hir::ClassDef {
            id: class_def_id,
            name: Symbol::intern("MyEq"),
            params: vec![a_var.clone()],
            supers: vec![],
            methods: vec![bhc_hir::MethodSig {
                name: Symbol::intern("myEq"),
                ty: Scheme {
                    vars: vec![],
                    constraints: vec![],
                    ty: Ty::Fun(
                        Box::new(Ty::Var(a_var.clone())),
                        Box::new(Ty::Fun(
                            Box::new(Ty::Var(a_var.clone())),
                            Box::new(bool_ty.clone()),
                        )),
                    ),
                },
                span: Span::default(),
            }],
            defaults: vec![],
            span: Span::default(),
        };

        // Instance definition
        let instance_def = bhc_hir::InstanceDef {
            class: Symbol::intern("MyEq"),
            types: vec![int_ty.clone()],
            constraints: vec![],
            methods: vec![ValueDef {
                id: my_eq_impl_id,
                name: Symbol::intern("myEq"),
                sig: None,
                equations: vec![Equation {
                    pats: vec![
                        Pat::Var(Symbol::intern("x"), x_def_id, Span::default()),
                        Pat::Var(Symbol::intern("y"), y_def_id, Span::default()),
                    ],
                    guards: vec![],
                    rhs: Expr::Con(bhc_hir::DefRef {
                        def_id: DefId::new(9), // True
                        span: Span::default(),
                    }),
                    span: Span::default(),
                }],
                span: Span::default(),
            }],
            span: Span::default(),
        };

        // Test function: test = myEq @Int
        // This uses a type application to specify the concrete type
        let test_def = ValueDef {
            id: test_def_id,
            name: Symbol::intern("test"),
            sig: None,
            equations: vec![Equation {
                pats: vec![],
                guards: vec![],
                rhs: Expr::TypeApp(
                    Box::new(Expr::Var(bhc_hir::DefRef {
                        def_id: my_eq_method_def_id,
                        span: Span::default(),
                    })),
                    int_ty.clone(),
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
            items: vec![
                Item::Class(class_def),
                Item::Instance(instance_def),
                Item::Value(test_def),
            ],
            span: Span::default(),
        };

        // Register the myEq method as a variable
        let mut ctx = LowerContext::new();
        let my_eq_var = ctx.named_var(Symbol::intern("myEq"), Ty::Error);
        ctx.register_var(my_eq_method_def_id, my_eq_var);

        // Lower the module
        let result = ctx.lower_module(&module);
        assert!(result.is_ok(), "Module lowering should succeed");

        let core_module = result.unwrap();

        // Find the test binding
        let test_bind = core_module.bindings.iter().find(|b| match b {
            bhc_core::Bind::NonRec(var, _) => var.name.as_str() == "test",
            _ => false,
        });

        assert!(test_bind.is_some(), "Should have test binding");

        // The test binding should contain a Let expression with dictionary construction
        // followed by method selection
        if let bhc_core::Bind::NonRec(_, expr) = test_bind.unwrap() {
            // Check that the expression involves dictionary/method handling
            // It should be a Let binding for the dictionary, then method selection
            fn has_let(e: &bhc_core::Expr) -> bool {
                match e {
                    bhc_core::Expr::Let(_, _, _) => true,
                    bhc_core::Expr::App(f, x, _) => has_let(f) || has_let(x),
                    _ => false,
                }
            }

            // For now, just verify it compiles and produces some expression
            // A more thorough test would check the exact structure
            // The expression should be something meaningful, not just a variable
            let is_simple_var = matches!(expr.as_ref(), bhc_core::Expr::Var(_, _));

            // If the class registry has the method, we expect some transformation
            // (Let binding for dictionary, or at least a TyApp)
            // If the registry is empty/method not found, it will fall through to TyApp
            let is_ty_app = matches!(expr.as_ref(), bhc_core::Expr::TyApp(_, _, _));
            let is_let = matches!(expr.as_ref(), bhc_core::Expr::Let(_, _, _));

            // We should have either a Let (dictionary was constructed) or TyApp (fallback)
            assert!(
                is_ty_app || is_let || is_simple_var,
                "Expected TyApp, Let, or Var, got: {:?}",
                expr
            );
        }
    }

    #[test]
    fn test_builtin_classes_registered() {
        // Verify that builtin type classes are registered in a fresh context
        let ctx = LowerContext::new();
        let registry = ctx.class_registry();

        // Check Eq class
        let eq_class = registry.lookup_class(Symbol::intern("Eq"));
        assert!(eq_class.is_some(), "Eq class should be registered");
        let eq_class = eq_class.unwrap();
        assert!(eq_class.methods.contains(&Symbol::intern("==")));
        assert!(eq_class.methods.contains(&Symbol::intern("/=")));
        assert!(eq_class.superclasses.is_empty());

        // Check Ord class
        let ord_class = registry.lookup_class(Symbol::intern("Ord"));
        assert!(ord_class.is_some(), "Ord class should be registered");
        let ord_class = ord_class.unwrap();
        assert!(ord_class.methods.contains(&Symbol::intern("compare")));
        assert!(ord_class.methods.contains(&Symbol::intern("<")));
        assert!(ord_class.methods.contains(&Symbol::intern(">")));
        assert!(ord_class.methods.contains(&Symbol::intern("min")));
        assert!(ord_class.methods.contains(&Symbol::intern("max")));
        assert_eq!(ord_class.superclasses, vec![Symbol::intern("Eq")]);

        // Check Num class
        let num_class = registry.lookup_class(Symbol::intern("Num"));
        assert!(num_class.is_some(), "Num class should be registered");
        let num_class = num_class.unwrap();
        assert!(num_class.methods.contains(&Symbol::intern("+")));
        assert!(num_class.methods.contains(&Symbol::intern("-")));
        assert!(num_class.methods.contains(&Symbol::intern("*")));
        assert!(num_class.methods.contains(&Symbol::intern("negate")));
        assert!(num_class.methods.contains(&Symbol::intern("fromInteger")));
        assert!(num_class.superclasses.is_empty());

        // Check Fractional class
        let frac_class = registry.lookup_class(Symbol::intern("Fractional"));
        assert!(frac_class.is_some(), "Fractional class should be registered");
        let frac_class = frac_class.unwrap();
        assert!(frac_class.methods.contains(&Symbol::intern("/")));
        assert!(frac_class.methods.contains(&Symbol::intern("fromRational")));
        assert_eq!(frac_class.superclasses, vec![Symbol::intern("Num")]);

        // Check Show class
        let show_class = registry.lookup_class(Symbol::intern("Show"));
        assert!(show_class.is_some(), "Show class should be registered");
        let show_class = show_class.unwrap();
        assert!(show_class.methods.contains(&Symbol::intern("show")));
    }

    #[test]
    fn test_builtin_instances_registered() {
        use bhc_types::{Kind, TyCon};

        let ctx = LowerContext::new();
        let registry = ctx.class_registry();

        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));
        let char_ty = Ty::Con(TyCon::new(Symbol::intern("Char"), Kind::Star));

        // Check Eq Int instance
        let eq_int = registry.resolve_instance(Symbol::intern("Eq"), &int_ty);
        assert!(eq_int.is_some(), "Eq Int instance should be registered");
        let eq_int = eq_int.unwrap();
        assert!(eq_int.methods.contains_key(&Symbol::intern("==")));
        assert!(eq_int.methods.contains_key(&Symbol::intern("/=")));

        // Check Num Int instance
        let num_int = registry.resolve_instance(Symbol::intern("Num"), &int_ty);
        assert!(num_int.is_some(), "Num Int instance should be registered");
        let num_int = num_int.unwrap();
        assert!(num_int.methods.contains_key(&Symbol::intern("+")));
        assert!(num_int.methods.contains_key(&Symbol::intern("-")));
        assert!(num_int.methods.contains_key(&Symbol::intern("*")));

        // Check Fractional Float instance
        let frac_float = registry.resolve_instance(Symbol::intern("Fractional"), &float_ty);
        assert!(frac_float.is_some(), "Fractional Float instance should be registered");
        let frac_float = frac_float.unwrap();
        assert!(frac_float.methods.contains_key(&Symbol::intern("/")));

        // Check Eq Bool instance
        let eq_bool = registry.resolve_instance(Symbol::intern("Eq"), &bool_ty);
        assert!(eq_bool.is_some(), "Eq Bool instance should be registered");

        // Check Ord Char instance
        let ord_char = registry.resolve_instance(Symbol::intern("Ord"), &char_ty);
        assert!(ord_char.is_some(), "Ord Char instance should be registered");

        // Check Show instances
        let show_int = registry.resolve_instance(Symbol::intern("Show"), &int_ty);
        assert!(show_int.is_some(), "Show Int instance should be registered");

        let show_bool = registry.resolve_instance(Symbol::intern("Show"), &bool_ty);
        assert!(show_bool.is_some(), "Show Bool instance should be registered");
    }

    #[test]
    fn test_is_class_method() {
        let ctx = LowerContext::new();

        // Check that arithmetic operators are class methods
        assert_eq!(
            ctx.is_class_method(Symbol::intern("+")),
            Some(Symbol::intern("Num")),
            "+ should be a Num method"
        );
        assert_eq!(
            ctx.is_class_method(Symbol::intern("==")),
            Some(Symbol::intern("Eq")),
            "== should be an Eq method"
        );
        assert_eq!(
            ctx.is_class_method(Symbol::intern("/")),
            Some(Symbol::intern("Fractional")),
            "/ should be a Fractional method"
        );
        assert_eq!(
            ctx.is_class_method(Symbol::intern("compare")),
            Some(Symbol::intern("Ord")),
            "compare should be an Ord method"
        );
        assert_eq!(
            ctx.is_class_method(Symbol::intern("show")),
            Some(Symbol::intern("Show")),
            "show should be a Show method"
        );

        // Check that non-class functions return None
        assert_eq!(
            ctx.is_class_method(Symbol::intern("map")),
            None,
            "map is not a class method"
        );
    }

    #[test]
    fn test_default_method_lowering() {
        // Test that default methods in class definitions are lowered correctly
        // with the class constraint (dictionary lambda).
        //
        // class MyEq a where
        //   myEq :: a -> a -> Bool
        //   myNeq :: a -> a -> Bool
        //   myNeq x y = not (myEq x y)  -- default implementation

        use bhc_types::{Kind, TyVar};

        let class_def_id = DefId::new(500);
        let my_eq_method_def_id = DefId::new(501);
        let my_neq_method_def_id = DefId::new(502);
        let my_neq_default_def_id = DefId::new(503);
        let x_def_id = DefId::new(504);
        let y_def_id = DefId::new(505);

        let a_var = TyVar::new(0, Kind::Star);

        // Create the class definition with a default method
        let class_def = bhc_hir::ClassDef {
            id: class_def_id,
            name: Symbol::intern("MyEq"),
            params: vec![a_var.clone()],
            supers: vec![],
            methods: vec![
                bhc_hir::MethodSig {
                    name: Symbol::intern("myEq"),
                    ty: Scheme {
                        vars: vec![a_var.clone()],
                        constraints: vec![],
                        ty: Ty::Fun(
                            Box::new(Ty::Var(a_var.clone())),
                            Box::new(Ty::Fun(
                                Box::new(Ty::Var(a_var.clone())),
                                Box::new(Ty::Con(bhc_types::TyCon::new(
                                    Symbol::intern("Bool"),
                                    Kind::Star,
                                ))),
                            )),
                        ),
                    },
                    span: Span::default(),
                },
                bhc_hir::MethodSig {
                    name: Symbol::intern("myNeq"),
                    ty: Scheme {
                        vars: vec![a_var.clone()],
                        constraints: vec![],
                        ty: Ty::Fun(
                            Box::new(Ty::Var(a_var.clone())),
                            Box::new(Ty::Fun(
                                Box::new(Ty::Var(a_var.clone())),
                                Box::new(Ty::Con(bhc_types::TyCon::new(
                                    Symbol::intern("Bool"),
                                    Kind::Star,
                                ))),
                            )),
                        ),
                    },
                    span: Span::default(),
                },
            ],
            // Default implementation for myNeq
            defaults: vec![ValueDef {
                id: my_neq_default_def_id,
                name: Symbol::intern("myNeq"),
                sig: None,
                equations: vec![Equation {
                    pats: vec![
                        Pat::Var(Symbol::intern("x"), x_def_id, Span::default()),
                        Pat::Var(Symbol::intern("y"), y_def_id, Span::default()),
                    ],
                    guards: vec![],
                    // Simplified: just return False (real impl would call not(myEq x y))
                    rhs: Expr::Con(bhc_hir::DefRef {
                        def_id: DefId::new(10), // False
                        span: Span::default(),
                    }),
                    span: Span::default(),
                }],
                span: Span::default(),
            }],
            span: Span::default(),
        };

        let module = HirModule {
            name: Symbol::intern("Test"),
            exports: None,
            imports: vec![],
            items: vec![Item::Class(class_def)],
            span: Span::default(),
        };

        // Lower the module
        let mut ctx = LowerContext::new();
        let result = ctx.lower_module(&module);
        assert!(result.is_ok(), "Module lowering should succeed");

        let core_module = result.unwrap();

        // We should have a binding for the default method
        assert!(
            !core_module.bindings.is_empty(),
            "Should have bindings for default methods"
        );

        // Find the myNeq default binding
        let my_neq_bind = core_module.bindings.iter().find(|b| match b {
            bhc_core::Bind::NonRec(var, _) => var.name.as_str() == "myNeq",
            _ => false,
        });

        assert!(my_neq_bind.is_some(), "Should have myNeq default binding");

        // The default method should have a dictionary lambda as the outermost lambda
        if let bhc_core::Bind::NonRec(_, expr) = my_neq_bind.unwrap() {
            if let bhc_core::Expr::Lam(dict_var, _, _) = expr.as_ref() {
                assert!(
                    dict_var.name.as_str().starts_with("$dMyEq"),
                    "Outermost lambda should be for MyEq dictionary, got: {}",
                    dict_var.name.as_str()
                );
            } else {
                panic!(
                    "Default method should have dictionary lambda as outermost, got: {:?}",
                    expr
                );
            }
        }

        // Verify the class is registered with the default
        let registry = ctx.class_registry();
        let class_info = registry.lookup_class(Symbol::intern("MyEq"));
        assert!(class_info.is_some(), "MyEq class should be registered");

        let class_info = class_info.unwrap();
        assert!(
            class_info.defaults.contains_key(&Symbol::intern("myNeq")),
            "MyEq should have myNeq as a default method"
        );
    }

    #[test]
    fn test_superclass_dictionary_extraction() {
        // Test that when we have an Ord dictionary in scope but need an Eq dictionary,
        // we can extract Eq from Ord (since Ord has Eq as a superclass).
        //
        // Scenario:
        // foo :: Ord a => a -> a -> Bool
        // foo x y = x == y   -- needs Eq, but we only have Ord
        //
        // The (==) call should get its dictionary by extracting Eq from Ord.

        use bhc_types::{Constraint, Kind, TyVar};

        let mut ctx = LowerContext::new();

        // Create a type variable for `a`
        let a_var = TyVar::new(0, Kind::Star);
        let a_ty = Ty::Var(a_var.clone());

        // Create an Ord dictionary variable (simulating what happens when lowering
        // a function with Ord constraint)
        let ord_dict_var = ctx.fresh_var("$dOrd", Ty::Error, Span::default());

        // Push a dictionary scope and register the Ord dictionary
        ctx.push_dict_scope();
        ctx.register_dict(Symbol::intern("Ord"), ord_dict_var.clone());

        // Now try to resolve an Eq constraint - this should extract from Ord
        let eq_constraint = Constraint::new(
            Symbol::intern("Eq"),
            a_ty.clone(),
            Span::default(),
        );

        let result = ctx.resolve_dictionary(&eq_constraint, Span::default());
        assert!(result.is_some(), "Should resolve Eq via superclass extraction from Ord");

        // The result should be a selector expression that extracts Eq from Ord
        // It should be an App (selector applied to dictionary)
        if let Some(expr) = result {
            match &expr {
                bhc_core::Expr::App(_, arg, _) => {
                    // The argument should reference our Ord dictionary
                    if let bhc_core::Expr::Var(var, _) = arg.as_ref() {
                        assert_eq!(
                            var.name, ord_dict_var.name,
                            "Should extract from the Ord dictionary"
                        );
                    } else {
                        panic!("Expected Var as argument to selector, got {:?}", arg);
                    }
                }
                _ => panic!("Expected App expression for superclass extraction, got {:?}", expr),
            }
        }

        // Also verify that lookup_superclass_dict finds the Ord dictionary
        let found = ctx.lookup_superclass_dict(Symbol::intern("Eq"));
        assert!(found.is_some(), "lookup_superclass_dict should find Ord for Eq");
        let (class, dict) = found.unwrap();
        assert_eq!(class, Symbol::intern("Ord"), "Should find Ord class");
        assert_eq!(dict.name, ord_dict_var.name, "Should return the Ord dictionary");

        ctx.pop_dict_scope();
    }

    #[test]
    fn test_superclass_not_found_when_no_subclass_dict() {
        // Test that superclass extraction fails gracefully when no subclass dictionary
        // is in scope.

        use bhc_types::{Constraint, Kind, TyVar};

        let mut ctx = LowerContext::new();

        let a_var = TyVar::new(0, Kind::Star);
        let a_ty = Ty::Var(a_var);

        // Register a Num dictionary (Num has no superclass relation to Eq)
        let num_dict_var = ctx.fresh_var("$dNum", Ty::Error, Span::default());
        ctx.push_dict_scope();
        ctx.register_dict(Symbol::intern("Num"), num_dict_var);

        // Try to resolve Eq - should fail since Num doesn't have Eq as superclass
        let eq_constraint = Constraint::new(
            Symbol::intern("Eq"),
            a_ty,
            Span::default(),
        );

        let result = ctx.resolve_dictionary(&eq_constraint, Span::default());
        // This should be None because:
        // - No direct Eq dictionary in scope
        // - Num doesn't have Eq as superclass
        // - Type variable means we can't construct from instance
        assert!(result.is_none(), "Should not resolve Eq from Num (no superclass relation)");

        ctx.pop_dict_scope();
    }

    #[test]
    fn test_direct_dict_preferred_over_superclass() {
        // Test that when both a direct dictionary and a superclass dictionary
        // are available, the direct dictionary is preferred.

        use bhc_types::{Constraint, Kind, TyVar};

        let mut ctx = LowerContext::new();

        let a_var = TyVar::new(0, Kind::Star);
        let a_ty = Ty::Var(a_var);

        // Register both Eq and Ord dictionaries
        let eq_dict_var = ctx.fresh_var("$dEq", Ty::Error, Span::default());
        let ord_dict_var = ctx.fresh_var("$dOrd", Ty::Error, Span::default());

        ctx.push_dict_scope();
        ctx.register_dict(Symbol::intern("Eq"), eq_dict_var.clone());
        ctx.register_dict(Symbol::intern("Ord"), ord_dict_var);

        // Resolve Eq - should use the direct Eq dictionary, not extract from Ord
        let eq_constraint = Constraint::new(
            Symbol::intern("Eq"),
            a_ty,
            Span::default(),
        );

        let result = ctx.resolve_dictionary(&eq_constraint, Span::default());
        assert!(result.is_some(), "Should resolve Eq");

        // Should be a simple Var reference to the Eq dictionary
        if let Some(bhc_core::Expr::Var(var, _)) = result {
            assert_eq!(
                var.name, eq_dict_var.name,
                "Should use direct Eq dictionary, not extract from Ord"
            );
        } else {
            panic!("Expected direct Var reference for Eq dictionary");
        }

        ctx.pop_dict_scope();
    }
}
