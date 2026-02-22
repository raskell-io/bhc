//! Case expression optimizations for Core IR.
//!
//! - Case-of-known-constructor: `case Con a b of { Con x y -> rhs }` => `rhs[a/x, b/y]`
//! - Case-of-literal: `case 42 of { 42 -> rhs }` => `rhs`
//! - Case-of-case: pushes outer case into inner case alternatives (with budget)

use rustc_hash::FxHashMap;

use crate::{Alt, AltCon, Expr, Literal};

use super::expr_util::expr_size;
use super::subst::substitute;

/// Attempt case-of-known-constructor or case-of-literal optimization.
///
/// Returns `Some(simplified)` if the scrutinee is a known constructor or literal,
/// `None` otherwise.
pub fn try_case_of_known(scrutinee: &Expr, alts: &[Alt]) -> Option<Expr> {
    match scrutinee {
        Expr::Lit(lit, _, _) => try_case_of_literal(lit, alts),
        _ => {
            // Try to detect constructor application: Con a1 a2 ... an
            if let Some((con_name, args)) = peel_constructor_app(scrutinee) {
                try_case_of_constructor(&con_name, &args, alts)
            } else {
                None
            }
        }
    }
}

/// Match a literal scrutinee against case alternatives.
fn try_case_of_literal(lit: &Literal, alts: &[Alt]) -> Option<Expr> {
    // First try exact literal match
    for alt in alts {
        if let AltCon::Lit(alt_lit) = &alt.con {
            if alt_lit == lit {
                return Some(alt.rhs.clone());
            }
        }
    }
    // Fall back to default
    for alt in alts {
        if alt.con == AltCon::Default {
            return Some(alt.rhs.clone());
        }
    }
    None
}

/// Peel a chain of `App` nodes to detect a saturated constructor application.
///
/// Returns `(constructor_name, [arg1, arg2, ...])` if the expression is
/// a constructor applied to arguments.
fn peel_constructor_app(expr: &Expr) -> Option<(String, Vec<Expr>)> {
    let mut current = expr;
    let mut args = Vec::new();

    loop {
        match current {
            Expr::App(f, a, _) => {
                args.push(*a.clone());
                current = f;
            }
            Expr::TyApp(f, _, _) => {
                // Skip type applications
                current = f;
            }
            Expr::Var(v, _) => {
                let name = v.name.as_str();
                // Constructor names start with uppercase or are special like (:), (,)
                if is_constructor_name(name) {
                    args.reverse();
                    return Some((name.to_string(), args));
                }
                return None;
            }
            _ => return None,
        }
    }
}

/// Check if a name looks like a data constructor.
fn is_constructor_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let first = name.chars().next().unwrap();
    first.is_uppercase()
        || name == ":"
        || name == "[]"
        || name.starts_with('(') && name.ends_with(')')
}

/// Match a known constructor application against case alternatives.
fn try_case_of_constructor(con_name: &str, args: &[Expr], alts: &[Alt]) -> Option<Expr> {
    // Find matching alternative
    for alt in alts {
        if let AltCon::DataCon(dc) = &alt.con {
            if dc.name.as_str() == con_name {
                // Build substitution from binder variables to constructor args
                let mut subst = FxHashMap::default();
                for (binder, arg) in alt.binders.iter().zip(args.iter()) {
                    subst.insert(binder.id, arg.clone());
                }
                return Some(substitute(alt.rhs.clone(), &subst));
            }
        }
    }
    // Fall back to default
    for alt in alts {
        if alt.con == AltCon::Default {
            return Some(alt.rhs.clone());
        }
    }
    None
}

/// Attempt case-of-case optimization with a size budget.
///
/// Transforms:
/// ```text
/// case (case x of { p1 -> e1; p2 -> e2 }) of alts
/// ```
/// into:
/// ```text
/// case x of { p1 -> case e1 of alts; p2 -> case e2 of alts }
/// ```
///
/// Only fires when the total duplicated code is within the budget.
pub fn try_case_of_case(
    scrutinee: &Expr,
    outer_alts: &[Alt],
    result_ty: &bhc_types::Ty,
    outer_span: bhc_span::Span,
    budget: usize,
) -> Option<Expr> {
    if let Expr::Case(inner_scrut, inner_alts, _, inner_span) = scrutinee {
        // Compute the cost of duplicating outer_alts into each inner alternative
        let outer_size: usize = outer_alts.iter().map(|a| expr_size(&a.rhs)).sum();
        let total_duplication = outer_size * inner_alts.len();

        if total_duplication > budget {
            return None;
        }

        // Push the outer case into each inner alternative
        let new_alts: Vec<Alt> = inner_alts
            .iter()
            .map(|inner_alt| {
                let new_rhs = Expr::Case(
                    Box::new(inner_alt.rhs.clone()),
                    outer_alts.to_vec(),
                    result_ty.clone(),
                    outer_span,
                );
                Alt {
                    con: inner_alt.con.clone(),
                    binders: inner_alt.binders.clone(),
                    rhs: new_rhs,
                }
            })
            .collect();

        Some(Expr::Case(
            inner_scrut.clone(),
            new_alts,
            result_ty.clone(),
            *inner_span,
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::{Ty, TyCon, Kind};
    use crate::{DataCon, Var, VarId};

    fn mk_var(name: &str, id: u32) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id as usize), Ty::Error)
    }

    fn mk_var_expr(name: &str, id: u32) -> Expr {
        Expr::Var(mk_var(name, id), Span::default())
    }

    fn mk_int(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), Ty::Error, Span::default())
    }

    fn mk_data_con(name: &str, tag: u32, arity: u32) -> DataCon {
        DataCon {
            name: Symbol::intern(name),
            ty_con: TyCon {
                name: Symbol::intern("T"),
                kind: Kind::Star,
            },
            tag,
            arity,
        }
    }

    #[test]
    fn test_case_of_literal() {
        // case 42 of { 42 -> "yes"; _ -> "no" }
        let alts = vec![
            Alt {
                con: AltCon::Lit(Literal::Int(42)),
                binders: vec![],
                rhs: mk_int(1),
            },
            Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: mk_int(0),
            },
        ];
        let result = try_case_of_known(&mk_int(42), &alts);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), Expr::Lit(Literal::Int(1), _, _)));
    }

    #[test]
    fn test_case_of_literal_default() {
        // case 99 of { 42 -> 1; _ -> 0 }
        let alts = vec![
            Alt {
                con: AltCon::Lit(Literal::Int(42)),
                binders: vec![],
                rhs: mk_int(1),
            },
            Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: mk_int(0),
            },
        ];
        let result = try_case_of_known(&mk_int(99), &alts);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), Expr::Lit(Literal::Int(0), _, _)));
    }

    #[test]
    fn test_case_of_known_constructor() {
        // case Just 42 of { Nothing -> 0; Just x -> x }
        let just_42 = Expr::App(
            Box::new(Expr::Var(mk_var("Just", 10), Span::default())),
            Box::new(mk_int(42)),
            Span::default(),
        );
        let alts = vec![
            Alt {
                con: AltCon::DataCon(mk_data_con("Nothing", 0, 0)),
                binders: vec![],
                rhs: mk_int(0),
            },
            Alt {
                con: AltCon::DataCon(mk_data_con("Just", 1, 1)),
                binders: vec![mk_var("x", 20)],
                rhs: mk_var_expr("x", 20),
            },
        ];
        let result = try_case_of_known(&just_42, &alts);
        assert!(result.is_some());
        // Should have substituted 42 for x
        assert!(matches!(result.unwrap(), Expr::Lit(Literal::Int(42), _, _)));
    }

    #[test]
    fn test_case_of_case_within_budget() {
        // case (case b of { True -> Just 1; False -> Nothing }) of { ... }
        let inner_alts = vec![
            Alt {
                con: AltCon::DataCon(mk_data_con("True", 1, 0)),
                binders: vec![],
                rhs: mk_int(1),
            },
            Alt {
                con: AltCon::DataCon(mk_data_con("False", 0, 0)),
                binders: vec![],
                rhs: mk_int(0),
            },
        ];
        let inner_case = Expr::Case(
            Box::new(mk_var_expr("b", 1)),
            inner_alts,
            Ty::Error,
            Span::default(),
        );
        let outer_alts = vec![Alt {
            con: AltCon::Default,
            binders: vec![],
            rhs: mk_int(99),
        }];

        let result =
            try_case_of_case(&inner_case, &outer_alts, &Ty::Error, Span::default(), 100);
        assert!(result.is_some());
        // Result should be: case b of { True -> case 1 of ...; False -> case 0 of ... }
        if let Expr::Case(scrut, alts, _, _) = result.unwrap() {
            assert!(matches!(*scrut, Expr::Var(_, _)));
            assert_eq!(alts.len(), 2);
            assert!(matches!(alts[0].rhs, Expr::Case(_, _, _, _)));
        } else {
            panic!("expected Case");
        }
    }

    #[test]
    fn test_case_of_case_over_budget() {
        // With budget 0, should not fire
        let inner_alts = vec![Alt {
            con: AltCon::Default,
            binders: vec![],
            rhs: mk_int(1),
        }];
        let inner_case = Expr::Case(
            Box::new(mk_var_expr("b", 1)),
            inner_alts,
            Ty::Error,
            Span::default(),
        );
        let outer_alts = vec![Alt {
            con: AltCon::Default,
            binders: vec![],
            rhs: mk_int(99),
        }];

        let result = try_case_of_case(&inner_case, &outer_alts, &Ty::Error, Span::default(), 0);
        assert!(result.is_none());
    }
}
