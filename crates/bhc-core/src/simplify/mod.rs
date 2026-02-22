//! Core IR simplifier.
//!
//! Iterates a set of local transformations to fixpoint:
//! - Beta reduction
//! - Case-of-known-constructor / case-of-literal
//! - Case-of-case (with size budget)
//! - Dead binding elimination
//! - Constant folding
//! - Inlining (single-use and small multi-use)
//!
//! This is the most impactful optimization pass for functional code.
//! LLVM cannot optimize above the level of algebraic data types and closures.

pub mod beta;
pub mod case;
pub mod dead;
pub mod expr_util;
pub mod fold;
pub mod inline;
pub mod occurrence;
pub mod subst;

use rustc_hash::FxHashMap;

use crate::{Alt, Bind, CoreModule, Expr, VarId};

/// Configuration for the simplifier.
#[derive(Clone, Debug)]
pub struct SimplifyConfig {
    /// Maximum number of simplification iterations (default 10).
    pub max_iterations: usize,
    /// Expression size threshold for inlining multi-use bindings (default 20).
    pub inline_threshold: usize,
    /// Size budget for case-of-case duplication (default 100).
    pub case_of_case_budget: usize,
    /// Whether to enable constant folding (default true).
    pub constant_fold: bool,
    /// Whether to enable case-of-case (default true).
    pub case_of_case: bool,
}

impl Default for SimplifyConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            inline_threshold: 20,
            case_of_case_budget: 100,
            constant_fold: true,
            case_of_case: true,
        }
    }
}

impl SimplifyConfig {
    /// Create a config from a `bhc_session::OptLevel`.
    #[must_use]
    pub fn from_opt_level(level: bhc_session::OptLevel) -> Self {
        match level {
            bhc_session::OptLevel::None => Self {
                max_iterations: 0,
                ..Self::default()
            },
            bhc_session::OptLevel::Less => Self {
                max_iterations: 4,
                inline_threshold: 10,
                case_of_case: false,
                ..Self::default()
            },
            bhc_session::OptLevel::Default | bhc_session::OptLevel::Size => Self::default(),
            bhc_session::OptLevel::Aggressive => Self {
                max_iterations: 15,
                inline_threshold: 40,
                case_of_case_budget: 200,
                ..Self::default()
            },
            bhc_session::OptLevel::SizeMin => Self {
                max_iterations: 5,
                inline_threshold: 5,
                case_of_case: false,
                ..Self::default()
            },
        }
    }
}

/// Statistics from a simplification run.
#[derive(Clone, Debug, Default)]
pub struct SimplifyStats {
    /// Number of iterations before reaching fixpoint.
    pub iterations: usize,
    /// Number of beta reductions performed.
    pub beta_reductions: usize,
    /// Number of case-of-known-constructor eliminations.
    pub case_of_known: usize,
    /// Number of case-of-case transformations.
    pub case_of_case: usize,
    /// Number of dead bindings removed.
    pub dead_bindings: usize,
    /// Number of constant folds performed.
    pub constant_folds: usize,
    /// Number of inlines performed.
    pub inlines: usize,
}

/// Simplify a Core module in place.
///
/// Returns statistics about the transformations performed.
pub fn simplify_module(module: &mut CoreModule, config: &SimplifyConfig) -> SimplifyStats {
    if config.max_iterations == 0 {
        return SimplifyStats::default();
    }

    let mut total_stats = SimplifyStats::default();

    for iteration in 0..config.max_iterations {
        total_stats.iterations = iteration + 1;
        let mut changed = false;

        // Phase 1: occurrence analysis over all bindings
        let occs = occurrence::analyze_module_occurrences(&module.bindings);

        // Top-level inlining is disabled: the codegen dispatches on
        // var.name.as_str() for builtins, constructors, and derived
        // methods. Inlining a top-level binding replaces Var references
        // with the binding's RHS, changing what the codegen sees.
        let inline_env: FxHashMap<VarId, Expr> = FxHashMap::default();

        // Phase 2: simplify each binding's RHS
        // Top-level dead binding elimination is disabled: any top-level
        // binding could be imported by another module, and occurrence
        // analysis is module-local so cannot see cross-module references.
        let mut new_bindings = Vec::with_capacity(module.bindings.len());
        for bind in std::mem::take(&mut module.bindings) {
            match bind {
                Bind::NonRec(var, rhs) => {
                    let mut stats = SimplifyStats::default();
                    let new_rhs =
                        simplify_expr(*rhs, &inline_env, config, &mut stats);
                    if stats.has_changes() {
                        changed = true;
                    }
                    total_stats.merge(&stats);
                    new_bindings.push(Bind::NonRec(var, Box::new(new_rhs)));
                }
                Bind::Rec(pairs) => {
                    let new_pairs: Vec<_> = pairs
                        .into_iter()
                        .map(|(v, rhs)| {
                            let mut stats = SimplifyStats::default();
                            let new_rhs =
                                simplify_expr(*rhs, &inline_env, config, &mut stats);
                            if stats.has_changes() {
                                changed = true;
                            }
                            total_stats.merge(&stats);
                            (v, Box::new(new_rhs))
                        })
                        .collect();
                    new_bindings.push(Bind::Rec(new_pairs));
                }
            }
        }

        module.bindings = new_bindings;

        if !changed {
            break;
        }
    }

    total_stats
}

/// Check if a top-level name must be preserved (never eliminated).
fn is_top_level_required(name: &str) -> bool {
    name == "main"
        || name.starts_with("$")
        || name.starts_with("bhc_")
        || name.contains("::")
}

impl SimplifyStats {
    fn has_changes(&self) -> bool {
        self.beta_reductions > 0
            || self.case_of_known > 0
            || self.case_of_case > 0
            || self.dead_bindings > 0
            || self.constant_folds > 0
            || self.inlines > 0
    }

    fn merge(&mut self, other: &SimplifyStats) {
        self.beta_reductions += other.beta_reductions;
        self.case_of_known += other.case_of_known;
        self.case_of_case += other.case_of_case;
        self.dead_bindings += other.dead_bindings;
        self.constant_folds += other.constant_folds;
        self.inlines += other.inlines;
    }
}

/// Recursively simplify an expression, applying all transformations.
fn simplify_expr(
    expr: Expr,
    inline_env: &FxHashMap<VarId, Expr>,
    config: &SimplifyConfig,
    stats: &mut SimplifyStats,
) -> Expr {
    match expr {
        // Variable: try inlining from top-level inline env
        Expr::Var(ref v, _) => {
            if let Some(replacement) = inline_env.get(&v.id) {
                stats.inlines += 1;
                simplify_expr(replacement.clone(), inline_env, config, stats)
            } else {
                expr
            }
        }

        // Literals, types, coercions: nothing to simplify
        Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => expr,

        // Application: simplify sub-expressions, then try beta/fold
        Expr::App(f, a, span) => {
            let f = simplify_expr(*f, inline_env, config, stats);
            let a = simplify_expr(*a, inline_env, config, stats);

            // Try beta reduction: (\x -> body) arg
            if let Some(reduced) = beta::try_beta_reduce(&f, &a, config.inline_threshold) {
                stats.beta_reductions += 1;
                return simplify_expr(reduced, inline_env, config, stats);
            }

            let result = Expr::App(Box::new(f), Box::new(a), span);

            // Try constant folding
            if config.constant_fold {
                if let Some(folded) = fold::try_constant_fold(&result) {
                    stats.constant_folds += 1;
                    return folded;
                }
            }

            result
        }

        // Type application: simplify the function
        Expr::TyApp(f, ty, span) => {
            let f = simplify_expr(*f, inline_env, config, stats);
            Expr::TyApp(Box::new(f), ty, span)
        }

        // Lambda: simplify body
        Expr::Lam(x, body, span) => {
            let body = simplify_expr(*body, inline_env, config, stats);
            Expr::Lam(x, Box::new(body), span)
        }

        // Type lambda: simplify body
        Expr::TyLam(tv, body, span) => {
            let body = simplify_expr(*body, inline_env, config, stats);
            Expr::TyLam(tv, Box::new(body), span)
        }

        // Let: simplify, then eliminate dead bindings or inline
        Expr::Let(bind, body, span) => {
            match *bind {
                Bind::NonRec(var, rhs) => {
                    let new_rhs = simplify_expr(*rhs, inline_env, config, stats);

                    // Run local occurrence analysis on the body
                    let body_occs = occurrence::analyze_occurrences(&body);

                    // Check if the binding is dead in the body
                    if dead::try_eliminate_dead_nonrec(var.id, &body, &body_occs) {
                        // Only eliminate if the RHS has no side effects.
                        // For now, conservatively treat all non-cheap RHS as
                        // potentially effectful to avoid dropping IO actions.
                        if expr_util::is_cheap(&new_rhs) {
                            stats.dead_bindings += 1;
                            return simplify_expr(*body, inline_env, config, stats);
                        }
                    }

                    // Check if we should inline this local binding.
                    // Only inline cheap expressions (variables, literals) to
                    // avoid creating IR that the codegen cannot handle.
                    let should_inline_local = expr_util::is_cheap(&new_rhs);

                    if should_inline_local {
                        stats.inlines += 1;
                        let substituted = subst::substitute_single(*body, var.id, &new_rhs);
                        return simplify_expr(substituted, inline_env, config, stats);
                    }

                    let new_body = simplify_expr(*body, inline_env, config, stats);
                    Expr::Let(
                        Box::new(Bind::NonRec(var, Box::new(new_rhs))),
                        Box::new(new_body),
                        span,
                    )
                }
                Bind::Rec(pairs) => {
                    let new_pairs: Vec<_> = pairs
                        .into_iter()
                        .map(|(v, rhs)| {
                            let new_rhs = simplify_expr(*rhs, inline_env, config, stats);
                            (v, Box::new(new_rhs))
                        })
                        .collect();
                    let new_body = simplify_expr(*body, inline_env, config, stats);
                    Expr::Let(
                        Box::new(Bind::Rec(new_pairs)),
                        Box::new(new_body),
                        span,
                    )
                }
            }
        }

        // Case: simplify scrutinee, then try case-of-known and case-of-case
        Expr::Case(scrut, alts, ty, span) => {
            let scrut = simplify_expr(*scrut, inline_env, config, stats);

            // Try case-of-known-constructor / case-of-literal
            if let Some(reduced) = case::try_case_of_known(&scrut, &alts) {
                stats.case_of_known += 1;
                return simplify_expr(reduced, inline_env, config, stats);
            }

            // Try case-of-case
            if config.case_of_case {
                if let Some(transformed) =
                    case::try_case_of_case(&scrut, &alts, &ty, span, config.case_of_case_budget)
                {
                    stats.case_of_case += 1;
                    return simplify_expr(transformed, inline_env, config, stats);
                }
            }

            // Simplify alternatives
            let new_alts: Vec<Alt> = alts
                .into_iter()
                .map(|alt| Alt {
                    con: alt.con,
                    binders: alt.binders,
                    rhs: simplify_expr(alt.rhs, inline_env, config, stats),
                })
                .collect();

            Expr::Case(Box::new(scrut), new_alts, ty, span)
        }

        // Lazy: simplify inner
        Expr::Lazy(e, span) => {
            let e = simplify_expr(*e, inline_env, config, stats);
            Expr::Lazy(Box::new(e), span)
        }

        // Cast: simplify inner
        Expr::Cast(e, coercion, span) => {
            let e = simplify_expr(*e, inline_env, config, stats);
            Expr::Cast(Box::new(e), coercion, span)
        }

        // Tick: simplify inner
        Expr::Tick(tick, e, span) => {
            let e = simplify_expr(*e, inline_env, config, stats);
            Expr::Tick(tick, Box::new(e), span)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::Ty;
    use crate::{AltCon, DataCon, Literal, Var};
    use bhc_types::{TyCon, Kind};

    fn mk_var(name: &str, id: u32) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id as usize), Ty::Error)
    }

    fn mk_var_expr(name: &str, id: u32) -> Expr {
        Expr::Var(mk_var(name, id), Span::default())
    }

    fn mk_int(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), Ty::Error, Span::default())
    }

    fn mk_module(bindings: Vec<Bind>) -> CoreModule {
        CoreModule {
            name: Symbol::intern("Test"),
            bindings,
            exports: vec![],
            overloaded_strings: false,
            constructors: vec![],
        }
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
    fn test_simplify_noop_at_o0() {
        let mut module = mk_module(vec![Bind::NonRec(mk_var("x", 1), Box::new(mk_int(42)))]);
        let config = SimplifyConfig {
            max_iterations: 0,
            ..Default::default()
        };
        let stats = simplify_module(&mut module, &config);
        assert_eq!(stats.iterations, 0);
    }

    #[test]
    fn test_simplify_dead_local_binding() {
        // main = let unused = 99 in 42
        // The local dead binding should be eliminated (RHS is cheap)
        let let_expr = Expr::Let(
            Box::new(Bind::NonRec(mk_var("unused", 1), Box::new(mk_int(99)))),
            Box::new(mk_int(42)),
            Span::default(),
        );
        let mut module = mk_module(vec![Bind::NonRec(mk_var("main", 2), Box::new(let_expr))]);
        let config = SimplifyConfig::default();
        let stats = simplify_module(&mut module, &config);
        // The local dead binding should be eliminated
        assert!(stats.dead_bindings > 0);

        // Result should be just 42 (no let wrapper)
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            assert!(
                matches!(rhs.as_ref(), Expr::Lit(Literal::Int(42), _, _)),
                "expected 42, got {:?}",
                rhs
            );
        } else {
            panic!("expected NonRec binding");
        }
    }

    #[test]
    fn test_simplify_constant_fold() {
        // main = 1 + 2
        let add = Expr::App(
            Box::new(Expr::App(
                Box::new(Expr::Var(mk_var("+", 0), Span::default())),
                Box::new(mk_int(1)),
                Span::default(),
            )),
            Box::new(mk_int(2)),
            Span::default(),
        );
        let mut module = mk_module(vec![Bind::NonRec(mk_var("main", 1), Box::new(add))]);
        let config = SimplifyConfig::default();
        let stats = simplify_module(&mut module, &config);
        assert!(stats.constant_folds > 0);

        // Verify the result is 3
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            assert!(matches!(rhs.as_ref(), Expr::Lit(Literal::Int(3), _, _)));
        } else {
            panic!("expected NonRec binding");
        }
    }

    #[test]
    fn test_simplify_beta_reduction() {
        // main = (\x -> x) 42
        let lam = Expr::Lam(mk_var("x", 10), Box::new(mk_var_expr("x", 10)), Span::default());
        let app = Expr::App(Box::new(lam), Box::new(mk_int(42)), Span::default());
        let mut module = mk_module(vec![Bind::NonRec(mk_var("main", 1), Box::new(app))]);
        let config = SimplifyConfig::default();
        let stats = simplify_module(&mut module, &config);
        assert!(stats.beta_reductions > 0);

        // Verify result is 42
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            assert!(matches!(rhs.as_ref(), Expr::Lit(Literal::Int(42), _, _)));
        } else {
            panic!("expected NonRec binding");
        }
    }

    #[test]
    fn test_simplify_case_of_known_constructor() {
        // main = case Just 42 of { Nothing -> 0; Just x -> x }
        let just_42 = Expr::App(
            Box::new(Expr::Var(mk_var("Just", 10), Span::default())),
            Box::new(mk_int(42)),
            Span::default(),
        );
        let case_expr = Expr::Case(
            Box::new(just_42),
            vec![
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
            ],
            Ty::Error,
            Span::default(),
        );
        let mut module = mk_module(vec![Bind::NonRec(mk_var("main", 1), Box::new(case_expr))]);
        let config = SimplifyConfig::default();
        let stats = simplify_module(&mut module, &config);
        assert!(stats.case_of_known > 0);

        // Verify result is 42
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            assert!(
                matches!(rhs.as_ref(), Expr::Lit(Literal::Int(42), _, _)),
                "expected 42, got {:?}",
                rhs
            );
        } else {
            panic!("expected NonRec binding");
        }
    }

    #[test]
    fn test_simplify_inline_let() {
        // main = let x = 42 in x + 1
        // After inlining x: main = 42 + 1
        // After constant folding: main = 43
        let body = Expr::App(
            Box::new(Expr::App(
                Box::new(Expr::Var(mk_var("+", 0), Span::default())),
                Box::new(mk_var_expr("x", 10)),
                Span::default(),
            )),
            Box::new(mk_int(1)),
            Span::default(),
        );
        let let_expr = Expr::Let(
            Box::new(Bind::NonRec(mk_var("x", 10), Box::new(mk_int(42)))),
            Box::new(body),
            Span::default(),
        );
        let mut module = mk_module(vec![Bind::NonRec(mk_var("main", 1), Box::new(let_expr))]);
        let config = SimplifyConfig::default();
        let stats = simplify_module(&mut module, &config);
        assert!(stats.inlines > 0);
        // After inline + fold: 42 + 1 = 43
        assert!(stats.constant_folds > 0);

        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            assert!(
                matches!(rhs.as_ref(), Expr::Lit(Literal::Int(43), _, _)),
                "expected 43, got {:?}",
                rhs
            );
        } else {
            panic!("expected NonRec binding");
        }
    }

    #[test]
    fn test_simplify_preserves_recursive_bindings() {
        // let rec f = \x -> f x in f 42
        let body_inner = Expr::App(
            Box::new(mk_var_expr("f", 1)),
            Box::new(mk_var_expr("x", 2)),
            Span::default(),
        );
        let lam = Expr::Lam(mk_var("x", 2), Box::new(body_inner), Span::default());
        let let_expr = Expr::Let(
            Box::new(Bind::Rec(vec![(mk_var("f", 1), Box::new(lam))])),
            Box::new(Expr::App(
                Box::new(mk_var_expr("f", 1)),
                Box::new(mk_int(42)),
                Span::default(),
            )),
            Span::default(),
        );
        let mut module = mk_module(vec![Bind::NonRec(mk_var("main", 3), Box::new(let_expr))]);
        let config = SimplifyConfig::default();
        let stats = simplify_module(&mut module, &config);
        // The recursive binding should still be present
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            assert!(matches!(rhs.as_ref(), Expr::Let(_, _, _)));
        }
    }

    #[test]
    fn test_simplify_multiple_iterations() {
        // Nested simplifications that require multiple passes:
        // let x = 1 + 2 in let y = x + 3 in y
        // Iteration 1: fold 1+2 => 3, inline x into y's rhs
        // Iteration 2: fold 3+3 => 6, inline y
        let inner_body = mk_var_expr("y", 20);
        let y_rhs = Expr::App(
            Box::new(Expr::App(
                Box::new(Expr::Var(mk_var("+", 0), Span::default())),
                Box::new(mk_var_expr("x", 10)),
                Span::default(),
            )),
            Box::new(mk_int(3)),
            Span::default(),
        );
        let inner_let = Expr::Let(
            Box::new(Bind::NonRec(mk_var("y", 20), Box::new(y_rhs))),
            Box::new(inner_body),
            Span::default(),
        );
        let x_rhs = Expr::App(
            Box::new(Expr::App(
                Box::new(Expr::Var(mk_var("+", 0), Span::default())),
                Box::new(mk_int(1)),
                Span::default(),
            )),
            Box::new(mk_int(2)),
            Span::default(),
        );
        let outer_let = Expr::Let(
            Box::new(Bind::NonRec(mk_var("x", 10), Box::new(x_rhs))),
            Box::new(inner_let),
            Span::default(),
        );
        let mut module = mk_module(vec![Bind::NonRec(mk_var("main", 1), Box::new(outer_let))]);
        let config = SimplifyConfig::default();
        let stats = simplify_module(&mut module, &config);

        // Should have performed inlines and constant folds
        assert!(stats.inlines > 0);
        assert!(stats.constant_folds > 0);

        // Final result should be 6
        if let Bind::NonRec(_, rhs) = &module.bindings[0] {
            assert!(
                matches!(rhs.as_ref(), Expr::Lit(Literal::Int(6), _, _)),
                "expected 6, got {:?}",
                rhs
            );
        } else {
            panic!("expected NonRec binding");
        }
    }
}
