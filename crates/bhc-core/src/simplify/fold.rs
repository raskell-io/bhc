//! Constant folding for Core IR.
//!
//! Evaluates arithmetic operations on known literal values at compile time.

use bhc_intern::Symbol;

use crate::{Expr, Literal};

/// Attempt to fold a primitive operation applied to literal arguments.
///
/// Recognizes patterns like `PrimOp arg1 arg2` where the PrimOp is a known
/// arithmetic operation and both arguments are literals.
///
/// Returns `Some(folded_literal)` on success, `None` if not foldable.
pub fn try_constant_fold(expr: &Expr) -> Option<Expr> {
    // Pattern: App(App(Var(op), lit1), lit2) for binary ops
    if let Expr::App(inner_app, arg2, span) = expr {
        if let Expr::App(op_expr, arg1, _) = inner_app.as_ref() {
            if let Expr::Var(op_var, _) = op_expr.as_ref() {
                let op_name = op_var.name.as_str();
                return try_fold_binary(op_name, arg1, arg2, *span);
            }
        }
    }

    // Pattern: App(Var(op), lit) for unary ops
    if let Expr::App(op_expr, arg, span) = expr {
        if let Expr::Var(op_var, _) = op_expr.as_ref() {
            let op_name = op_var.name.as_str();
            return try_fold_unary(op_name, arg, *span);
        }
    }

    None
}

fn try_fold_binary(
    op: &str,
    arg1: &Expr,
    arg2: &Expr,
    span: bhc_span::Span,
) -> Option<Expr> {
    match (arg1, arg2) {
        (Expr::Lit(Literal::Int(a), ty, _), Expr::Lit(Literal::Int(b), _, _)) => {
            let result = match op {
                "+" | "GHC.Num.+" => a.checked_add(*b)?,
                "-" | "GHC.Num.-" => a.checked_sub(*b)?,
                "*" | "GHC.Num.*" => a.checked_mul(*b)?,
                _ => return None,
            };
            Some(Expr::Lit(Literal::Int(result), ty.clone(), span))
        }
        (Expr::Lit(Literal::Double(a), ty, _), Expr::Lit(Literal::Double(b), _, _)) => {
            let result = match op {
                "+" | "GHC.Num.+" => a + b,
                "-" | "GHC.Num.-" => a - b,
                "*" | "GHC.Num.*" => a * b,
                "/" | "GHC.Fractional./" => {
                    if *b == 0.0 {
                        return None;
                    }
                    a / b
                }
                _ => return None,
            };
            Some(Expr::Lit(Literal::Double(result), ty.clone(), span))
        }
        (Expr::Lit(Literal::Float(a), ty, _), Expr::Lit(Literal::Float(b), _, _)) => {
            let result = match op {
                "+" | "GHC.Num.+" => a + b,
                "-" | "GHC.Num.-" => a - b,
                "*" | "GHC.Num.*" => a * b,
                "/" | "GHC.Fractional./" => {
                    if *b == 0.0 {
                        return None;
                    }
                    a / b
                }
                _ => return None,
            };
            Some(Expr::Lit(Literal::Float(result), ty.clone(), span))
        }
        (Expr::Lit(Literal::String(a), ty, _), Expr::Lit(Literal::String(b), _, _)) => {
            match op {
                "++" => {
                    let result = format!("{}{}", a.as_str(), b.as_str());
                    Some(Expr::Lit(
                        Literal::String(Symbol::intern(&result)),
                        ty.clone(),
                        span,
                    ))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn try_fold_unary(op: &str, arg: &Expr, span: bhc_span::Span) -> Option<Expr> {
    match arg {
        Expr::Lit(Literal::Int(n), ty, _) => match op {
            "negate" | "GHC.Num.negate" => {
                Some(Expr::Lit(Literal::Int(-n), ty.clone(), span))
            }
            _ => None,
        },
        Expr::Lit(Literal::Double(n), ty, _) => match op {
            "negate" | "GHC.Num.negate" => {
                Some(Expr::Lit(Literal::Double(-n), ty.clone(), span))
            }
            _ => None,
        },
        Expr::Lit(Literal::Float(n), ty, _) => match op {
            "negate" | "GHC.Num.negate" => {
                Some(Expr::Lit(Literal::Float(-n), ty.clone(), span))
            }
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::Ty;
    use crate::{Var, VarId};

    fn mk_var(name: &str, id: u32) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id as usize), Ty::Error)
    }

    fn mk_int(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), Ty::Error, Span::default())
    }

    fn mk_double(n: f64) -> Expr {
        Expr::Lit(Literal::Double(n), Ty::Error, Span::default())
    }

    fn mk_binop(op: &str, a: Expr, b: Expr) -> Expr {
        Expr::App(
            Box::new(Expr::App(
                Box::new(Expr::Var(mk_var(op, 0), Span::default())),
                Box::new(a),
                Span::default(),
            )),
            Box::new(b),
            Span::default(),
        )
    }

    fn mk_unop(op: &str, a: Expr) -> Expr {
        Expr::App(
            Box::new(Expr::Var(mk_var(op, 0), Span::default())),
            Box::new(a),
            Span::default(),
        )
    }

    #[test]
    fn test_fold_int_add() {
        let e = mk_binop("+", mk_int(1), mk_int(2));
        let result = try_constant_fold(&e);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), Expr::Lit(Literal::Int(3), _, _)));
    }

    #[test]
    fn test_fold_int_mul() {
        let e = mk_binop("*", mk_int(5), mk_int(6));
        let result = try_constant_fold(&e);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), Expr::Lit(Literal::Int(30), _, _)));
    }

    #[test]
    fn test_fold_int_sub() {
        let e = mk_binop("-", mk_int(10), mk_int(3));
        let result = try_constant_fold(&e);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), Expr::Lit(Literal::Int(7), _, _)));
    }

    #[test]
    fn test_fold_double_add() {
        let e = mk_binop("+", mk_double(1.5), mk_double(2.5));
        let result = try_constant_fold(&e);
        assert!(result.is_some());
        if let Expr::Lit(Literal::Double(v), _, _) = result.unwrap() {
            assert!((v - 4.0).abs() < f64::EPSILON);
        } else {
            panic!("expected Double literal");
        }
    }

    #[test]
    fn test_fold_negate_int() {
        let e = mk_unop("negate", mk_int(42));
        let result = try_constant_fold(&e);
        assert!(result.is_some());
        assert!(matches!(
            result.unwrap(),
            Expr::Lit(Literal::Int(-42), _, _)
        ));
    }

    #[test]
    fn test_no_fold_for_variables() {
        let e = mk_binop(
            "+",
            Expr::Var(mk_var("x", 1), Span::default()),
            mk_int(2),
        );
        let result = try_constant_fold(&e);
        assert!(result.is_none());
    }

    #[test]
    fn test_no_fold_for_unknown_op() {
        let e = mk_binop("foo", mk_int(1), mk_int(2));
        let result = try_constant_fold(&e);
        assert!(result.is_none());
    }
}
