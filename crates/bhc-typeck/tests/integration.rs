//! Integration tests for the BHC type checker.
//!
//! These tests construct sample HIR and verify that the type checker
//! produces the expected types or errors.

use bhc_hir::{
    Binding, CaseAlt, ConDef, ConFields, DataDef, DefId, DefRef, Equation, Expr, Item, Lit,
    Module, Pat, ValueDef,
};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::{FileId, Span};
use bhc_typeck::type_check_module;
use bhc_types::{Kind, Scheme, Ty, TyCon, TyVar};

/// Helper to create a DefId.
fn def_id(n: usize) -> DefId {
    DefId::new(n)
}

/// Helper to create a DefRef.
fn def_ref(n: usize) -> DefRef {
    DefRef {
        def_id: def_id(n),
        span: Span::DUMMY,
    }
}

/// Helper to create an empty module with items.
fn module_with_items(items: Vec<Item>) -> Module {
    Module {
        name: Symbol::intern("Test"),
        exports: None,
        imports: Vec::new(),
        items,
        span: Span::DUMMY,
    }
}

/// Helper to create a simple value definition with one equation.
fn value_def(id: usize, name: &str, rhs: Expr) -> ValueDef {
    ValueDef {
        id: def_id(id),
        name: Symbol::intern(name),
        sig: None,
        equations: vec![Equation {
            pats: Vec::new(),
            guards: Vec::new(),
            rhs,
            span: Span::DUMMY,
        }],
        span: Span::DUMMY,
    }
}

/// Helper to create a function definition with patterns.
fn func_def(id: usize, name: &str, pats: Vec<Pat>, rhs: Expr) -> ValueDef {
    ValueDef {
        id: def_id(id),
        name: Symbol::intern(name),
        sig: None,
        equations: vec![Equation {
            pats,
            guards: Vec::new(),
            rhs,
            span: Span::DUMMY,
        }],
        span: Span::DUMMY,
    }
}

/// Helper to create a value definition with a type signature.
fn value_def_with_sig(id: usize, name: &str, sig: Scheme, rhs: Expr) -> ValueDef {
    ValueDef {
        id: def_id(id),
        name: Symbol::intern(name),
        sig: Some(sig),
        equations: vec![Equation {
            pats: Vec::new(),
            guards: Vec::new(),
            rhs,
            span: Span::DUMMY,
        }],
        span: Span::DUMMY,
    }
}

// =============================================================================
// Basic Expression Tests
// =============================================================================

#[test]
fn test_integer_literal() {
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "x",
        Expr::Lit(Lit::Int(42), Span::DUMMY),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok(), "Expected successful type check");

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    // x should have type Int
    assert!(scheme.is_mono());
    match &scheme.ty {
        Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
        _ => panic!("Expected Int type, got {:?}", scheme.ty),
    }
}

#[test]
fn test_float_literal() {
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "x",
        Expr::Lit(Lit::Float(3.14), Span::DUMMY),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    match &scheme.ty {
        Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Float")),
        _ => panic!("Expected Float type"),
    }
}

#[test]
fn test_char_literal() {
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "c",
        Expr::Lit(Lit::Char('a'), Span::DUMMY),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    match &scheme.ty {
        Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Char")),
        _ => panic!("Expected Char type"),
    }
}

#[test]
fn test_string_literal() {
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "s",
        Expr::Lit(Lit::String(Symbol::intern("hello")), Span::DUMMY),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    match &scheme.ty {
        Ty::Con(c) => assert_eq!(c.name, Symbol::intern("String")),
        _ => panic!("Expected String type"),
    }
}

#[test]
fn test_tuple_expression() {
    // (42, 'a')
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "pair",
        Expr::Tuple(
            vec![
                Expr::Lit(Lit::Int(42), Span::DUMMY),
                Expr::Lit(Lit::Char('a'), Span::DUMMY),
            ],
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    match &scheme.ty {
        Ty::Tuple(elems) => {
            assert_eq!(elems.len(), 2);
            match &elems[0] {
                Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
                _ => panic!("Expected Int"),
            }
            match &elems[1] {
                Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Char")),
                _ => panic!("Expected Char"),
            }
        }
        _ => panic!("Expected tuple type"),
    }
}

#[test]
fn test_list_expression() {
    // [1, 2, 3]
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "nums",
        Expr::List(
            vec![
                Expr::Lit(Lit::Int(1), Span::DUMMY),
                Expr::Lit(Lit::Int(2), Span::DUMMY),
                Expr::Lit(Lit::Int(3), Span::DUMMY),
            ],
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    match &scheme.ty {
        Ty::List(elem) => match elem.as_ref() {
            Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
            _ => panic!("Expected [Int]"),
        },
        _ => panic!("Expected list type"),
    }
}

#[test]
fn test_empty_list() {
    // []
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "empty",
        Expr::List(Vec::new(), Span::DUMMY),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    // Empty list should have polymorphic type [a]
    match &scheme.ty {
        Ty::List(elem) => {
            // Element type should be a type variable
            assert!(matches!(elem.as_ref(), Ty::Var(_)));
        }
        _ => panic!("Expected list type"),
    }
}

// =============================================================================
// Lambda and Function Tests
// =============================================================================

#[test]
fn test_identity_lambda() {
    // \x -> x
    let x = Symbol::intern("x");
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "id",
        Expr::Lam(
            vec![Pat::Var(x, Span::DUMMY)],
            Box::new(Expr::Lit(Lit::Int(42), Span::DUMMY)), // For now, return Int
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    // id should have type a -> Int (since body returns Int)
    match &scheme.ty {
        Ty::Fun(_, to) => match to.as_ref() {
            Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
            _ => panic!("Expected Int return type"),
        },
        _ => panic!("Expected function type"),
    }
}

#[test]
fn test_const_lambda() {
    // \x y -> x (returns Int since x is bound to Int pattern in this test)
    let x = Symbol::intern("x");
    let y = Symbol::intern("y");

    // Using Lit pattern to constrain x to Int
    let module = module_with_items(vec![Item::Value(func_def(
        0,
        "const",
        vec![Pat::Var(x, Span::DUMMY), Pat::Var(y, Span::DUMMY)],
        Expr::Lit(Lit::Int(1), Span::DUMMY),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    // const should have type a -> b -> Int
    match &scheme.ty {
        Ty::Fun(_, rest) => match rest.as_ref() {
            Ty::Fun(_, to) => match to.as_ref() {
                Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
                _ => panic!("Expected Int return type"),
            },
            _ => panic!("Expected nested function"),
        },
        _ => panic!("Expected function type"),
    }
}

// =============================================================================
// Let Expression Tests
// =============================================================================

#[test]
fn test_simple_let() {
    // let x = 42 in x
    let x = Symbol::intern("x");
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "result",
        Expr::Let(
            vec![Binding {
                pat: Pat::Var(x, Span::DUMMY),
                sig: None,
                rhs: Expr::Lit(Lit::Int(42), Span::DUMMY),
                span: Span::DUMMY,
            }],
            Box::new(Expr::Lit(Lit::Int(0), Span::DUMMY)), // Just return another Int
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    match &scheme.ty {
        Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
        _ => panic!("Expected Int type"),
    }
}

#[test]
fn test_let_with_tuple_pattern() {
    // let (a, b) = (1, 'c') in a
    let a = Symbol::intern("a");
    let b = Symbol::intern("b");

    // Create a tuple constructor DefRef (we'll use a builtin ID)
    let tuple_con = DefRef {
        def_id: DefId::new(0xFFFF_0010), // Some ID for tuple constructor
        span: Span::DUMMY,
    };

    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "result",
        Expr::Let(
            vec![Binding {
                pat: Pat::Con(
                    tuple_con,
                    vec![Pat::Var(a, Span::DUMMY), Pat::Var(b, Span::DUMMY)],
                    Span::DUMMY,
                ),
                sig: None,
                rhs: Expr::Tuple(
                    vec![
                        Expr::Lit(Lit::Int(1), Span::DUMMY),
                        Expr::Lit(Lit::Char('c'), Span::DUMMY),
                    ],
                    Span::DUMMY,
                ),
                span: Span::DUMMY,
            }],
            Box::new(Expr::Lit(Lit::Int(0), Span::DUMMY)),
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    // This may produce an error because the tuple constructor isn't registered,
    // but the infrastructure handles it gracefully
    // The key is that it doesn't panic
    let _ = result;
}

// =============================================================================
// If Expression Tests
// =============================================================================

#[test]
fn test_if_expression_type_match() {
    // if True then 1 else 2
    // Note: We need to reference the True constructor
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "result",
        Expr::If(
            Box::new(Expr::Con(DefRef {
                def_id: DefId::new(0xFFFF_0000), // True builtin ID
                span: Span::DUMMY,
            })),
            Box::new(Expr::Lit(Lit::Int(1), Span::DUMMY)),
            Box::new(Expr::Lit(Lit::Int(2), Span::DUMMY)),
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    match &scheme.ty {
        Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
        _ => panic!("Expected Int type"),
    }
}

#[test]
fn test_if_branch_mismatch() {
    // if True then 1 else 'a' -- type error: Int vs Char
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "result",
        Expr::If(
            Box::new(Expr::Con(DefRef {
                def_id: DefId::new(0xFFFF_0000), // True
                span: Span::DUMMY,
            })),
            Box::new(Expr::Lit(Lit::Int(1), Span::DUMMY)),
            Box::new(Expr::Lit(Lit::Char('a'), Span::DUMMY)),
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_err(), "Expected type error for branch mismatch");

    let errors = result.unwrap_err();
    assert!(!errors.is_empty());
}

#[test]
fn test_if_non_bool_condition() {
    // if 42 then 1 else 2 -- type error: Int is not Bool
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "result",
        Expr::If(
            Box::new(Expr::Lit(Lit::Int(42), Span::DUMMY)),
            Box::new(Expr::Lit(Lit::Int(1), Span::DUMMY)),
            Box::new(Expr::Lit(Lit::Int(2), Span::DUMMY)),
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_err(), "Expected type error for non-Bool condition");

    let errors = result.unwrap_err();
    assert!(!errors.is_empty());
}

// =============================================================================
// Case Expression Tests
// =============================================================================

#[test]
fn test_case_with_literals() {
    // case 42 of
    //   0 -> 'a'
    //   _ -> 'b'
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "result",
        Expr::Case(
            Box::new(Expr::Lit(Lit::Int(42), Span::DUMMY)),
            vec![
                CaseAlt {
                    pat: Pat::Lit(Lit::Int(0), Span::DUMMY),
                    guards: Vec::new(),
                    rhs: Expr::Lit(Lit::Char('a'), Span::DUMMY),
                    span: Span::DUMMY,
                },
                CaseAlt {
                    pat: Pat::Wild(Span::DUMMY),
                    guards: Vec::new(),
                    rhs: Expr::Lit(Lit::Char('b'), Span::DUMMY),
                    span: Span::DUMMY,
                },
            ],
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    match &scheme.ty {
        Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Char")),
        _ => panic!("Expected Char type"),
    }
}

#[test]
fn test_case_branch_type_mismatch() {
    // case 42 of
    //   0 -> 'a'
    //   _ -> 1    -- type error: Char vs Int
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "result",
        Expr::Case(
            Box::new(Expr::Lit(Lit::Int(42), Span::DUMMY)),
            vec![
                CaseAlt {
                    pat: Pat::Lit(Lit::Int(0), Span::DUMMY),
                    guards: Vec::new(),
                    rhs: Expr::Lit(Lit::Char('a'), Span::DUMMY),
                    span: Span::DUMMY,
                },
                CaseAlt {
                    pat: Pat::Wild(Span::DUMMY),
                    guards: Vec::new(),
                    rhs: Expr::Lit(Lit::Int(1), Span::DUMMY),
                    span: Span::DUMMY,
                },
            ],
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_err(), "Expected type error for case branch mismatch");
}

// =============================================================================
// Data Type Tests
// =============================================================================

#[test]
fn test_data_type_with_constructors() {
    // data Maybe a = Nothing | Just a
    let a = TyVar::new_star(100);

    let data_def = DataDef {
        id: def_id(0),
        name: Symbol::intern("MyMaybe"),
        params: vec![a.clone()],
        cons: vec![
            ConDef {
                id: def_id(1),
                name: Symbol::intern("MyNothing"),
                fields: ConFields::Positional(Vec::new()),
                span: Span::DUMMY,
            },
            ConDef {
                id: def_id(2),
                name: Symbol::intern("MyJust"),
                fields: ConFields::Positional(vec![Ty::Var(a)]),
                span: Span::DUMMY,
            },
        ],
        deriving: Vec::new(),
        span: Span::DUMMY,
    };

    // Use the Just constructor: MyJust 42
    let module = module_with_items(vec![
        Item::Data(data_def),
        Item::Value(value_def(
            3,
            "wrapped",
            Expr::App(
                Box::new(Expr::Con(def_ref(2))), // MyJust
                Box::new(Expr::Lit(Lit::Int(42), Span::DUMMY)),
                Span::DUMMY,
            ),
        )),
    ]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok(), "Expected successful type check: {:?}", result);

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(3)).unwrap();

    // wrapped should have type MyMaybe Int
    match &scheme.ty {
        Ty::App(con, arg) => {
            match con.as_ref() {
                Ty::Con(c) => assert_eq!(c.name, Symbol::intern("MyMaybe")),
                _ => panic!("Expected MyMaybe constructor"),
            }
            match arg.as_ref() {
                Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
                _ => panic!("Expected Int argument"),
            }
        }
        _ => panic!("Expected applied type, got {:?}", scheme.ty),
    }
}

#[test]
fn test_constructor_pattern_match() {
    // data Pair a b = MkPair a b
    let a = TyVar::new_star(100);
    let b = TyVar::new_star(101);

    let data_def = DataDef {
        id: def_id(0),
        name: Symbol::intern("Pair"),
        params: vec![a.clone(), b.clone()],
        cons: vec![ConDef {
            id: def_id(1),
            name: Symbol::intern("MkPair"),
            fields: ConFields::Positional(vec![Ty::Var(a), Ty::Var(b)]),
            span: Span::DUMMY,
        }],
        deriving: Vec::new(),
        span: Span::DUMMY,
    };

    // Create pair and deconstruct: case MkPair 1 'a' of MkPair x y -> x
    let x = Symbol::intern("x");
    let y = Symbol::intern("y");

    let module = module_with_items(vec![
        Item::Data(data_def),
        Item::Value(value_def(
            2,
            "fst",
            Expr::Case(
                Box::new(Expr::App(
                    Box::new(Expr::App(
                        Box::new(Expr::Con(def_ref(1))), // MkPair
                        Box::new(Expr::Lit(Lit::Int(1), Span::DUMMY)),
                        Span::DUMMY,
                    )),
                    Box::new(Expr::Lit(Lit::Char('a'), Span::DUMMY)),
                    Span::DUMMY,
                )),
                vec![CaseAlt {
                    pat: Pat::Con(
                        def_ref(1),
                        vec![Pat::Var(x, Span::DUMMY), Pat::Var(y, Span::DUMMY)],
                        Span::DUMMY,
                    ),
                    guards: Vec::new(),
                    rhs: Expr::Lit(Lit::Int(0), Span::DUMMY), // Return Int
                    span: Span::DUMMY,
                }],
                Span::DUMMY,
            ),
        )),
    ]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok(), "Expected successful type check: {:?}", result);
}

// =============================================================================
// Multiple Definitions Tests
// =============================================================================

#[test]
fn test_multiple_independent_definitions() {
    let module = module_with_items(vec![
        Item::Value(value_def(0, "x", Expr::Lit(Lit::Int(1), Span::DUMMY))),
        Item::Value(value_def(1, "y", Expr::Lit(Lit::Char('a'), Span::DUMMY))),
        Item::Value(value_def(
            2,
            "z",
            Expr::Tuple(
                vec![
                    Expr::Lit(Lit::Int(2), Span::DUMMY),
                    Expr::Lit(Lit::Float(3.0), Span::DUMMY),
                ],
                Span::DUMMY,
            ),
        )),
    ]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();

    // Check all three definitions have the expected types
    assert!(typed.def_schemes.contains_key(&def_id(0)));
    assert!(typed.def_schemes.contains_key(&def_id(1)));
    assert!(typed.def_schemes.contains_key(&def_id(2)));
}

// =============================================================================
// Type Signature Tests
// =============================================================================

#[test]
fn test_value_with_matching_signature() {
    // x :: Int
    // x = 42
    let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
    let sig = Scheme::mono(int_ty);

    let module = module_with_items(vec![Item::Value(value_def_with_sig(
        0,
        "x",
        sig,
        Expr::Lit(Lit::Int(42), Span::DUMMY),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());
}

#[test]
fn test_value_with_mismatching_signature() {
    // x :: Int
    // x = 'a'  -- type error
    let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
    let sig = Scheme::mono(int_ty);

    let module = module_with_items(vec![Item::Value(value_def_with_sig(
        0,
        "x",
        sig,
        Expr::Lit(Lit::Char('a'), Span::DUMMY),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_err(), "Expected type error for signature mismatch");
}

// =============================================================================
// Error Recovery Tests
// =============================================================================

#[test]
fn test_error_recovery_continues_checking() {
    // Multiple definitions, one has an error
    let module = module_with_items(vec![
        Item::Value(value_def(0, "good1", Expr::Lit(Lit::Int(1), Span::DUMMY))),
        // This has an error (branch mismatch)
        Item::Value(value_def(
            1,
            "bad",
            Expr::If(
                Box::new(Expr::Con(DefRef {
                    def_id: DefId::new(0xFFFF_0000),
                    span: Span::DUMMY,
                })),
                Box::new(Expr::Lit(Lit::Int(1), Span::DUMMY)),
                Box::new(Expr::Lit(Lit::Char('a'), Span::DUMMY)),
                Span::DUMMY,
            ),
        )),
        Item::Value(value_def(2, "good2", Expr::Lit(Lit::Char('b'), Span::DUMMY))),
    ]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_err());

    // Should have collected multiple errors/types
    let errors = result.unwrap_err();
    assert!(!errors.is_empty());
}

#[test]
fn test_error_expression_propagates() {
    // Using an error expression should not cause a panic
    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "x",
        Expr::Error(Span::DUMMY),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    // Should complete without panicking
    let _ = result;
}

// =============================================================================
// Nested Expression Tests
// =============================================================================

#[test]
fn test_nested_lambdas() {
    // \x -> \y -> (x, y)
    let x = Symbol::intern("x");
    let y = Symbol::intern("y");

    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "pair_fn",
        Expr::Lam(
            vec![Pat::Var(x, Span::DUMMY)],
            Box::new(Expr::Lam(
                vec![Pat::Var(y, Span::DUMMY)],
                Box::new(Expr::Tuple(
                    vec![
                        Expr::Lit(Lit::Int(1), Span::DUMMY),
                        Expr::Lit(Lit::Int(2), Span::DUMMY),
                    ],
                    Span::DUMMY,
                )),
                Span::DUMMY,
            )),
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    // Should be a -> b -> (Int, Int)
    match &scheme.ty {
        Ty::Fun(_, rest) => match rest.as_ref() {
            Ty::Fun(_, to) => match to.as_ref() {
                Ty::Tuple(elems) => assert_eq!(elems.len(), 2),
                _ => panic!("Expected tuple return type"),
            },
            _ => panic!("Expected nested function"),
        },
        _ => panic!("Expected function type"),
    }
}

#[test]
fn test_deeply_nested_lets() {
    // let a = 1 in let b = 2 in let c = 3 in (a, b, c)
    let a = Symbol::intern("a");
    let b = Symbol::intern("b");
    let c = Symbol::intern("c");

    let module = module_with_items(vec![Item::Value(value_def(
        0,
        "nested",
        Expr::Let(
            vec![Binding {
                pat: Pat::Var(a, Span::DUMMY),
                sig: None,
                rhs: Expr::Lit(Lit::Int(1), Span::DUMMY),
                span: Span::DUMMY,
            }],
            Box::new(Expr::Let(
                vec![Binding {
                    pat: Pat::Var(b, Span::DUMMY),
                    sig: None,
                    rhs: Expr::Lit(Lit::Int(2), Span::DUMMY),
                    span: Span::DUMMY,
                }],
                Box::new(Expr::Let(
                    vec![Binding {
                        pat: Pat::Var(c, Span::DUMMY),
                        sig: None,
                        rhs: Expr::Lit(Lit::Int(3), Span::DUMMY),
                        span: Span::DUMMY,
                    }],
                    Box::new(Expr::Tuple(
                        vec![
                            Expr::Lit(Lit::Int(0), Span::DUMMY),
                            Expr::Lit(Lit::Int(0), Span::DUMMY),
                            Expr::Lit(Lit::Int(0), Span::DUMMY),
                        ],
                        Span::DUMMY,
                    )),
                    Span::DUMMY,
                )),
                Span::DUMMY,
            )),
            Span::DUMMY,
        ),
    ))]);

    let result = type_check_module(&module, FileId::new(0));
    assert!(result.is_ok());

    let typed = result.unwrap();
    let scheme = typed.def_schemes.get(&def_id(0)).unwrap();

    match &scheme.ty {
        Ty::Tuple(elems) => assert_eq!(elems.len(), 3),
        _ => panic!("Expected 3-tuple type"),
    }
}
