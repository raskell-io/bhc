//! Pattern compilation for HIR to Core lowering.
//!
//! This module handles the compilation of HIR patterns into Core case
//! alternatives using the Augustsson/Sestoft column-based decision tree
//! algorithm.
//!
//! ## Pattern Compilation Strategy
//!
//! Multi-equation function definitions are compiled via a match matrix:
//! 1. Select the column with the most constructor information
//! 2. Group rows by head constructor in that column
//! 3. Generate a case dispatch with one branch per constructor group
//! 4. Recurse on remaining columns within each group
//!
//! This produces sharing of common tests across equations and enables
//! exhaustiveness and overlap checking.
//!
//! Single-alternative case expressions still use `lower_pat_to_alt`
//! for direct compilation without the matrix machinery.

use bhc_core::{self as core, Alt, AltCon, DataCon, Literal, Var, VarId};
use bhc_hir::{self as hir, Pat, ValueDef};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Kind, Ty, TyCon};
use std::collections::BTreeMap;

use crate::context::LowerContext;
use crate::expr::lower_expr;
use crate::{LowerError, LowerResult};

/// Lower a HIR pattern to a Core case alternative.
///
/// This creates a single alternative that matches the given pattern and
/// executes the given RHS expression.
pub fn lower_pat_to_alt(
    ctx: &mut LowerContext,
    pat: &hir::Pat,
    rhs: core::Expr,
    span: Span,
) -> LowerResult<Alt> {
    lower_pat_to_alt_with_fallthrough(ctx, pat, rhs, span, None)
}

/// Lower a HIR pattern to a Core case alternative with an optional fallthrough.
///
/// When `fallthrough` is `Some(expr)`, nested sub-pattern failures will fall
/// through to `expr` instead of generating a pattern match error. This is
/// needed for case expressions where a nested pattern failure (e.g., the
/// literal `0` in `Lit 0`) should try the remaining case alternatives.
pub fn lower_pat_to_alt_with_fallthrough(
    ctx: &mut LowerContext,
    pat: &hir::Pat,
    rhs: core::Expr,
    span: Span,
    fallthrough: Option<core::Expr>,
) -> LowerResult<Alt> {
    match pat {
        Pat::Wild(_) => {
            // Wildcard matches anything
            Ok(Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs,
            })
        }

        Pat::Var(name, def_id, _) => {
            // Variable pattern: bind the value to the variable
            // Use pre-registered variable if available (for guards),
            // otherwise create a fresh one
            let var = ctx.lookup_var(*def_id).cloned().unwrap_or_else(|| Var {
                name: *name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            });
            // Default pattern with a binder
            Ok(Alt {
                con: AltCon::Default,
                binders: vec![var],
                rhs,
            })
        }

        Pat::Lit(lit, _) => {
            // Literal pattern
            Ok(Alt {
                con: AltCon::Lit(hir_lit_to_core(lit)),
                binders: vec![],
                rhs,
            })
        }

        Pat::Con(def_ref, sub_pats, _) => {
            // Constructor pattern
            // We need to create binders for all sub-patterns
            let mut binders = Vec::with_capacity(sub_pats.len());
            let mut inner_rhs = rhs;

            // Process sub-patterns from right to left
            // For complex sub-patterns, we need to wrap the RHS in nested cases
            for (i, sub_pat) in sub_pats.iter().enumerate().rev() {
                let binder_name = format!("pat_{}", i);
                let binder = ctx.fresh_var(&binder_name, Ty::Error, span);

                match sub_pat {
                    Pat::Var(name, def_id, _) => {
                        // Look up the registered variable (from bind_pattern_vars)
                        // Fall back to creating a fresh one if not found
                        let var = if let Some(registered) = ctx.lookup_var(*def_id) {
                            registered.clone()
                        } else {
                            Var {
                                name: *name,
                                id: ctx.fresh_id(),
                                ty: Ty::Error,
                            }
                        };
                        binders.push(var);
                    }
                    Pat::Wild(_) => {
                        // Wildcard: use a fresh variable
                        binders.push(binder);
                    }
                    _ => {
                        // Complex sub-pattern: need nested case
                        let sub_alt = lower_pat_to_alt_with_fallthrough(ctx, sub_pat, inner_rhs.clone(), span, fallthrough.clone())?;
                        let default_rhs = fallthrough.clone().unwrap_or_else(|| make_pattern_error(span));
                        let default_alt = Alt {
                            con: AltCon::Default,
                            binders: vec![],
                            rhs: default_rhs,
                        };

                        inner_rhs = core::Expr::Case(
                            Box::new(core::Expr::Var(binder.clone(), span)),
                            vec![sub_alt, default_alt],
                            Ty::Error,
                            span,
                        );
                        binders.push(binder);
                    }
                }
            }

            // Reverse binders to match left-to-right order
            binders.reverse();

            // Create the data constructor
            // Look up the constructor metadata from the context
            let (con_name, type_name, tag, existential_dict_count) =
                if let Some(info) = ctx.lookup_constructor(def_ref.def_id) {
                    (info.name, info.type_name, info.tag, info.existential_dict_count)
                } else if let Some(var) = ctx.lookup_var(def_ref.def_id) {
                    // Fallback for constructors not in the map - use name-based lookup
                    let tag = get_constructor_tag(var.name.as_str(), def_ref.def_id.index() as u32);
                    (var.name, Symbol::intern("DataType"), tag, 0)
                } else {
                    // Last resort fallback
                    let name = Symbol::intern("Con");
                    (
                        name,
                        Symbol::intern("DataType"),
                        def_ref.def_id.index() as u32,
                        0,
                    )
                };

            // For existential constructors, prepend binders for dictionary fields.
            // These dictionaries are stored as the first N fields of the constructor
            // and are extracted by the pattern match.
            if existential_dict_count > 0 {
                let mut dict_binders = Vec::with_capacity(existential_dict_count as usize);
                for i in 0..existential_dict_count {
                    let dict_var = ctx.fresh_var(
                        &format!("$edict_{}", i),
                        Ty::Error,
                        span,
                    );
                    dict_binders.push(dict_var);
                }
                // Prepend dict binders before user-visible binders
                dict_binders.append(&mut binders);
                binders = dict_binders;
            }

            let total_arity = sub_pats.len() as u32 + existential_dict_count;
            let tycon = TyCon::new(type_name, Kind::Star);
            let con = DataCon {
                name: con_name,
                ty_con: tycon,
                tag,
                arity: total_arity,
            };

            Ok(Alt {
                con: AltCon::DataCon(con),
                binders,
                rhs: inner_rhs,
            })
        }

        Pat::RecordCon(def_ref, field_pats, _) => {
            // Record constructor pattern with proper field ordering
            // Look up the constructor's canonical field order and reorder binders accordingly

            // Get constructor metadata
            let (con_name, type_name, tag, canonical_fields) =
                if let Some(info) = ctx.lookup_constructor(def_ref.def_id) {
                    (
                        info.name,
                        info.type_name,
                        info.tag,
                        info.field_names.clone(),
                    )
                } else if let Some(var) = ctx.lookup_var(def_ref.def_id) {
                    let tag = get_constructor_tag(var.name.as_str(), def_ref.def_id.index() as u32);
                    (var.name, Symbol::intern("DataType"), tag, vec![])
                } else {
                    let name = Symbol::intern("Con");
                    (
                        name,
                        Symbol::intern("DataType"),
                        def_ref.def_id.index() as u32,
                        vec![],
                    )
                };

            // Build a map from field name to its pattern
            let mut field_map: std::collections::HashMap<Symbol, &hir::FieldPat> =
                field_pats.iter().map(|fp| (fp.name, fp)).collect();

            // Determine the arity - use canonical fields if available, otherwise pattern count
            let arity = if canonical_fields.is_empty() {
                field_pats.len()
            } else {
                canonical_fields.len()
            };

            // Create binders in canonical order
            let mut binders = Vec::with_capacity(arity);
            let mut inner_rhs = rhs;

            // Process fields in canonical order (or pattern order if no canonical info)
            let field_order: Vec<Symbol> = if canonical_fields.is_empty() {
                // No canonical order known, use pattern order
                field_pats.iter().map(|fp| fp.name).collect()
            } else {
                canonical_fields
            };

            // Process fields from right to left to build nested cases correctly
            for (i, field_name) in field_order.iter().enumerate().rev() {
                let binder_name = format!("field_{}", i);
                let binder = ctx.fresh_var(&binder_name, Ty::Error, span);

                if let Some(fp) = field_map.get(field_name) {
                    // Field is bound in the pattern
                    match &fp.pat {
                        Pat::Var(name, def_id, _) => {
                            let var = if let Some(registered) = ctx.lookup_var(*def_id) {
                                registered.clone()
                            } else {
                                Var {
                                    name: *name,
                                    id: ctx.fresh_id(),
                                    ty: Ty::Error,
                                }
                            };
                            binders.push(var);
                        }
                        Pat::Wild(_) => {
                            binders.push(binder);
                        }
                        _ => {
                            let sub_alt = lower_pat_to_alt_with_fallthrough(ctx, &fp.pat, inner_rhs.clone(), span, fallthrough.clone())?;
                            let default_rhs = fallthrough.clone().unwrap_or_else(|| make_pattern_error(span));
                            let default_alt = Alt {
                                con: AltCon::Default,
                                binders: vec![],
                                rhs: default_rhs,
                            };
                            inner_rhs = core::Expr::Case(
                                Box::new(core::Expr::Var(binder.clone(), span)),
                                vec![sub_alt, default_alt],
                                Ty::Error,
                                span,
                            );
                            binders.push(binder);
                        }
                    }
                } else {
                    // Field not mentioned in pattern - use wildcard
                    binders.push(binder);
                }
            }

            binders.reverse();

            let tycon = TyCon::new(type_name, Kind::Star);
            let con = DataCon {
                name: con_name,
                ty_con: tycon,
                tag,
                arity: arity as u32,
            };

            Ok(Alt {
                con: AltCon::DataCon(con),
                binders,
                rhs: inner_rhs,
            })
        }

        Pat::As(name, _def_id, inner_pat, inner_span) => {
            // As-pattern: bind the value and also match the inner pattern
            let var = Var {
                name: *name,
                id: ctx.fresh_id(),
                ty: Ty::Error,
            };

            // Create a nested case for the inner pattern
            let inner_alt = lower_pat_to_alt_with_fallthrough(ctx, inner_pat, rhs, span, fallthrough.clone())?;
            let default_rhs = fallthrough.clone().unwrap_or_else(|| make_pattern_error(span));
            let default_alt = Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: default_rhs,
            };

            // The outer pattern binds the variable, then checks the inner
            let inner_case = core::Expr::Case(
                Box::new(core::Expr::Var(var.clone(), span)),
                vec![inner_alt, default_alt],
                Ty::Error,
                *inner_span,
            );

            Ok(Alt {
                con: AltCon::Default,
                binders: vec![var],
                rhs: inner_case,
            })
        }

        Pat::Or(_left, _right, _) => {
            // Or-pattern: both branches should match with the same RHS
            // For now, we only support or-patterns at the top level of case alternatives,
            // where they can be expanded into multiple alternatives.
            // Nested or-patterns are more complex and require pattern flattening.
            //
            // The proper handling is done in `lower_or_pattern_to_alts` which
            // expands or-patterns into multiple alternatives.
            //
            // If we reach here, it means an or-pattern appeared in a nested position
            // (e.g., inside a constructor pattern). For now, we don't support this.
            ctx.error(LowerError::PatternError {
                message: "nested or-patterns not yet supported; use top-level or-patterns in case alternatives".into(),
                span,
            });
            Ok(Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: make_pattern_error(span),
            })
        }

        Pat::Ann(inner, _ty, _) => {
            // Type-annotated pattern: just use the inner pattern
            lower_pat_to_alt(ctx, inner, rhs, span)
        }

        Pat::View(view_expr, result_pat, view_span) => {
            // View pattern: (f -> pat)
            // Desugars to: bind scrutinee to tmp, apply f to tmp, match result against pat
            let tmp_var = ctx.fresh_var("_view_scrut", Ty::Error, *view_span);

            // Lower the view expression to Core
            let core_view_expr = lower_expr(ctx, view_expr)?;

            // Apply the view function to the scrutinee variable
            let applied = core::Expr::App(
                Box::new(core_view_expr),
                Box::new(core::Expr::Var(tmp_var.clone(), *view_span)),
                *view_span,
            );

            // Create a nested case: case (f tmp) of { result_pat -> rhs }
            let inner_alt = lower_pat_to_alt(ctx, result_pat, rhs, span)?;
            let default_alt = Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: make_pattern_error(span),
            };

            let inner_case = core::Expr::Case(
                Box::new(applied),
                vec![inner_alt, default_alt],
                Ty::Error,
                *view_span,
            );

            // Outer alt: match anything, bind to tmp_var, then do the inner case
            Ok(Alt {
                con: AltCon::Default,
                binders: vec![tmp_var],
                rhs: inner_case,
            })
        }

        Pat::Error(_) => {
            // Error pattern: generate error
            Ok(Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: make_pattern_error(span),
            })
        }
    }
}

// ============================================================
// Decision Tree Pattern Compilation (Augustsson/Sestoft)
// ============================================================

/// The head of a pattern for grouping purposes.
///
/// We use a string key for `Lit` because `Literal` contains floats
/// which don't implement `Eq`/`Ord`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum PatHead {
    /// Constructor with (type_name, con_name, tag, arity).
    Con(Symbol, Symbol, u32, u32),
    /// Literal, represented as a string key for grouping.
    Lit(String),
    /// Wildcard/variable — matches anything.
    Wild,
}

/// Convert a HIR literal to a Core literal.
fn hir_lit_to_core(lit: &hir::Lit) -> Literal {
    match lit {
        hir::Lit::Int(n) => Literal::Int(*n as i64),
        hir::Lit::Float(f) => Literal::Float(*f as f32),
        hir::Lit::Char(c) => Literal::Char(*c),
        hir::Lit::String(s) => Literal::String(*s),
    }
}

/// Create a string key for a literal (for grouping).
fn literal_key(lit: &hir::Lit) -> String {
    match lit {
        hir::Lit::Int(n) => format!("Int:{}", n),
        hir::Lit::Float(f) => format!("Float:{}", f),
        hir::Lit::Char(c) => format!("Char:{}", *c as u32),
        hir::Lit::String(s) => format!("Str:{}", s.as_str()),
    }
}

/// A row in the match matrix.
struct ClauseRow {
    /// One pattern per scrutinee column.
    pats: Vec<hir::Pat>,
    /// Optional guard condition.
    guard: Option<hir::Expr>,
    /// Right-hand side expression.
    rhs: hir::Expr,
    /// Source location.
    span: Span,
    /// Original row index (for overlap detection).
    row_index: usize,
}

/// The match matrix: rows of patterns compiled against a vector of scrutinees.
struct MatchMatrix {
    /// Variables being matched (one per column).
    scrutinees: Vec<Var>,
    /// Rows of equations.
    rows: Vec<ClauseRow>,
}

/// Compiled decision tree.
enum DecisionTree {
    /// Leaf: matched successfully.
    Leaf {
        /// Variable bindings from pattern matching.
        bindings: Vec<(VarId, core::Expr)>,
        /// Guard condition (if any).
        guard: Option<hir::Expr>,
        /// Right-hand side.
        rhs: hir::Expr,
        /// Source span.
        span: Span,
        /// Original row index.
        row_index: usize,
    },
    /// Switch on a scrutinee, branching by constructor/literal.
    Switch {
        /// Variable to scrutinize.
        scrutinee: Var,
        /// One branch per constructor: (AltCon, bound vars, sub-tree).
        branches: Vec<(AltCon, Vec<Var>, DecisionTree)>,
        /// Default branch (for wildcards / uncovered constructors).
        default: Option<Box<DecisionTree>>,
        /// Source span for diagnostics.
        span: Span,
    },
    /// Match failure — non-exhaustive patterns.
    Fail(Span),
}

/// Compile pattern matching for multiple equations using decision trees.
///
/// Returns a Core expression that performs the pattern dispatch.
/// The caller is responsible for wrapping this in lambdas for the arguments.
pub fn compile_match_to_expr(
    ctx: &mut LowerContext,
    value_def: &ValueDef,
    args: &[Var],
) -> LowerResult<core::Expr> {
    // Check for view patterns or guards — these need special handling and
    // bypass the decision tree algorithm.
    let has_view = value_def.equations.iter().any(|eq| {
        eq.pats.iter().any(|p| matches!(p, Pat::View(_, _, _)))
    });
    let has_guards = value_def.equations.iter().any(|eq| !eq.guards.is_empty());

    if has_view || has_guards {
        // Fall back to linear compilation for view patterns and guards
        let alts = compile_equations_linear(ctx, &value_def.equations, args, value_def.span)?;
        let scrutinee = if args.len() == 1 {
            core::Expr::Var(args[0].clone(), value_def.span)
        } else {
            // Multi-arg: create a tuple scrutinee
            let tuple_con_name =
                Symbol::intern(&format!("({})", ",".repeat(args.len() - 1)));
            let mut expr = core::Expr::Var(
                Var {
                    name: tuple_con_name,
                    id: VarId::new(0),
                    ty: Ty::Error,
                },
                value_def.span,
            );
            for arg in args {
                expr = core::Expr::App(
                    Box::new(expr),
                    Box::new(core::Expr::Var(arg.clone(), value_def.span)),
                    value_def.span,
                );
            }
            expr
        };
        return Ok(core::Expr::Case(
            Box::new(scrutinee),
            alts,
            Ty::Error,
            value_def.span,
        ));
    }

    // Build the match matrix from equations
    let mut rows = Vec::with_capacity(value_def.equations.len());
    for (i, eq) in value_def.equations.iter().enumerate() {
        // Flatten or-patterns into separate rows
        let expanded = expand_equation_or_patterns(eq);
        for expanded_eq in expanded {
            rows.push(ClauseRow {
                pats: expanded_eq.0,
                guard: expanded_eq.1,
                rhs: expanded_eq.2,
                span: eq.span,
                row_index: i,
            });
        }
    }

    let matrix = MatchMatrix {
        scrutinees: args.to_vec(),
        rows,
    };

    // Build the decision tree
    let tree = build_decision_tree(ctx, matrix, value_def.span);

    // Check exhaustiveness
    check_exhaustiveness_tree(ctx, &tree, value_def.name.as_str());

    // Check for redundant equations
    let total_equations = value_def.equations.len();
    check_overlap(ctx, &tree, total_equations, value_def.name.as_str());

    // Convert decision tree to Core expression
    tree_to_core(ctx, tree)
}

/// Expand or-patterns in an equation into multiple (pats, guard, rhs) tuples.
fn expand_equation_or_patterns(
    eq: &hir::Equation,
) -> Vec<(Vec<hir::Pat>, Option<hir::Expr>, hir::Expr)> {
    // For each pattern position, flatten or-patterns
    let flattened: Vec<Vec<hir::Pat>> = eq
        .pats
        .iter()
        .map(flatten_nested_or_patterns)
        .collect();

    if flattened.is_empty() {
        return vec![(vec![], eq.guards.first().map(|g| g.cond.clone()), eq.rhs.clone())];
    }

    // Cross product of all expanded patterns
    let combinations = cross_product(&flattened);

    let guard = if eq.guards.is_empty() {
        None
    } else {
        // Multiple guards are ANDed together
        Some(eq.guards[0].cond.clone())
    };

    combinations
        .into_iter()
        .map(|pats| (pats, guard.clone(), eq.rhs.clone()))
        .collect()
}

/// Extract the head of a pattern for grouping.
fn pat_head(ctx: &LowerContext, pat: &hir::Pat) -> PatHead {
    match pat {
        Pat::Wild(_) | Pat::Var(_, _, _) => PatHead::Wild,
        Pat::Lit(lit, _) => PatHead::Lit(literal_key(lit)),
        Pat::Con(def_ref, sub_pats, _) => {
            if let Some(info) = ctx.lookup_constructor(def_ref.def_id) {
                PatHead::Con(info.type_name, info.name, info.tag, info.arity)
            } else {
                // Fallback: use name-based lookup
                let name = ctx
                    .lookup_var(def_ref.def_id)
                    .map(|v| v.name)
                    .unwrap_or_else(|| Symbol::intern("Con"));
                let tag = get_constructor_tag(name.as_str(), def_ref.def_id.index() as u32);
                PatHead::Con(
                    Symbol::intern("DataType"),
                    name,
                    tag,
                    sub_pats.len() as u32,
                )
            }
        }
        Pat::RecordCon(def_ref, field_pats, _) => {
            if let Some(info) = ctx.lookup_constructor(def_ref.def_id) {
                PatHead::Con(info.type_name, info.name, info.tag, info.arity)
            } else {
                let name = ctx
                    .lookup_var(def_ref.def_id)
                    .map(|v| v.name)
                    .unwrap_or_else(|| Symbol::intern("Con"));
                let tag = get_constructor_tag(name.as_str(), def_ref.def_id.index() as u32);
                PatHead::Con(
                    Symbol::intern("DataType"),
                    name,
                    tag,
                    field_pats.len() as u32,
                )
            }
        }
        Pat::As(_, _, inner, _) => pat_head(ctx, inner),
        Pat::Ann(inner, _, _) => pat_head(ctx, inner),
        Pat::Or(left, _, _) => pat_head(ctx, left), // shouldn't happen after expansion
        Pat::View(_, _, _) => PatHead::Wild,        // handled separately
        Pat::Error(_) => PatHead::Wild,
    }
}

/// Extract the sub-patterns from a constructor pattern.
/// For a wildcard, returns `arity` wildcard sub-patterns.
fn pat_sub_patterns(ctx: &LowerContext, pat: &hir::Pat, arity: u32, span: Span) -> Vec<hir::Pat> {
    match pat {
        Pat::Con(_, sub_pats, _) => sub_pats.clone(),
        Pat::RecordCon(def_ref, field_pats, _) => {
            // Reorder fields into canonical order
            let canonical_fields = ctx
                .lookup_constructor(def_ref.def_id)
                .map(|info| info.field_names.clone())
                .unwrap_or_default();

            if canonical_fields.is_empty() {
                field_pats.iter().map(|fp| fp.pat.clone()).collect()
            } else {
                let field_map: std::collections::HashMap<Symbol, &hir::Pat> =
                    field_pats.iter().map(|fp| (fp.name, &fp.pat)).collect();
                canonical_fields
                    .iter()
                    .map(|name| {
                        field_map
                            .get(name)
                            .cloned()
                            .cloned()
                            .unwrap_or(Pat::Wild(span))
                    })
                    .collect()
            }
        }
        Pat::As(_, _, inner, _) => pat_sub_patterns(ctx, inner, arity, span),
        Pat::Ann(inner, _, _) => pat_sub_patterns(ctx, inner, arity, span),
        Pat::Wild(_) | Pat::Var(_, _, _) => {
            // Wildcard matches any constructor — generate wildcard sub-patterns
            (0..arity).map(|_| Pat::Wild(span)).collect()
        }
        _ => (0..arity).map(|_| Pat::Wild(span)).collect(),
    }
}

/// Select the best column to match on.
///
/// Heuristic: prefer columns with constructor patterns (more information
/// for branching). Among constructor columns, prefer the one with the
/// most distinct constructors.
fn select_column(ctx: &LowerContext, matrix: &MatchMatrix) -> usize {
    if matrix.scrutinees.is_empty() {
        return 0;
    }

    let ncols = matrix.scrutinees.len();
    let mut best_col = 0;
    let mut best_score: i32 = -1;

    for col in 0..ncols {
        let mut n_constructors = 0i32;
        let mut has_any_con = false;
        let mut seen = std::collections::HashSet::new();

        for row in &matrix.rows {
            if col < row.pats.len() {
                let head = pat_head(ctx, &row.pats[col]);
                match &head {
                    PatHead::Con(_, _, _, _) | PatHead::Lit(_) => {
                        has_any_con = true;
                        if seen.insert(head.clone()) {
                            n_constructors += 1;
                        }
                    }
                    PatHead::Wild => {}
                }
            }
        }

        if has_any_con && n_constructors > best_score {
            best_score = n_constructors;
            best_col = col;
        }
    }

    best_col
}

/// Build a decision tree from a match matrix.
fn build_decision_tree(ctx: &mut LowerContext, matrix: MatchMatrix, span: Span) -> DecisionTree {
    // Base case: no rows → match failure
    if matrix.rows.is_empty() {
        return DecisionTree::Fail(span);
    }

    // Check if first row is all wildcards/vars (match success)
    let all_wild = matrix.rows[0].pats.iter().all(|p| {
        matches!(pat_head(ctx, p), PatHead::Wild)
    });

    if all_wild || matrix.rows[0].pats.is_empty() {
        // Register variable bindings for variables in the first row
        for (col, pat) in matrix.rows[0].pats.iter().enumerate() {
            if col < matrix.scrutinees.len() {
                register_wildcard_bindings(ctx, pat, &matrix.scrutinees[col]);
            }
        }

        let rhs = matrix.rows[0].rhs.clone();
        let row_span = matrix.rows[0].span;
        let row_index = matrix.rows[0].row_index;

        return DecisionTree::Leaf {
            bindings: vec![],
            guard: None,
            rhs,
            span: row_span,
            row_index,
        };
    }

    // Select best column to match on
    let col = select_column(ctx, &matrix);
    let scrutinee = matrix.scrutinees[col].clone();
    let row_span = matrix.rows[0].span;

    // Group rows by head constructor/literal at the selected column
    let mut groups: BTreeMap<PatHead, Vec<&ClauseRow>> = BTreeMap::new();
    let mut default_rows: Vec<&ClauseRow> = Vec::new();

    for row in &matrix.rows {
        let head = if col < row.pats.len() {
            pat_head(ctx, &row.pats[col])
        } else {
            PatHead::Wild
        };

        match &head {
            PatHead::Wild => {
                default_rows.push(row);
                // Wildcards also participate in all constructor groups
                for (_, group) in groups.iter_mut() {
                    group.push(row);
                }
            }
            _ => {
                groups.entry(head.clone()).or_default().push(row);
                // Also add existing default rows to this new group
                for dr in &default_rows {
                    groups.get_mut(&head).unwrap().push(dr);
                }
            }
        }
    }

    // Build branches
    let mut branches: Vec<(AltCon, Vec<Var>, DecisionTree)> = Vec::new();

    for (head, group_rows) in &groups {
        match head {
            PatHead::Con(type_name, con_name, tag, arity) => {
                let tycon = TyCon::new(*type_name, Kind::Star);
                let con = DataCon {
                    name: *con_name,
                    ty_con: tycon,
                    tag: *tag,
                    arity: *arity,
                };

                // Create fresh variables for constructor fields
                let field_vars: Vec<Var> = (0..*arity)
                    .map(|i| {
                        ctx.fresh_var(
                            &format!("_pat{}_{}", col, i),
                            Ty::Error,
                            row_span,
                        )
                    })
                    .collect();

                // Register variable bindings for wild/var rows participating
                // in this constructor group. When a Var pattern at column `col`
                // is removed by specialization, we must bind it to the scrutinee.
                for row in group_rows {
                    if col < row.pats.len() {
                        if matches!(pat_head(ctx, &row.pats[col]), PatHead::Wild) {
                            register_wildcard_bindings(ctx, &row.pats[col], &scrutinee);
                        }
                    }
                }

                // Build sub-matrix: specialize for this constructor
                let sub_matrix = specialize_matrix(
                    ctx,
                    &matrix,
                    col,
                    head,
                    &field_vars,
                    group_rows,
                    row_span,
                );

                let sub_tree = build_decision_tree(ctx, sub_matrix, row_span);
                branches.push((AltCon::DataCon(con), field_vars, sub_tree));
            }
            PatHead::Lit(_) => {
                // Find the actual literal from the first row in this group
                let core_lit = group_rows
                    .iter()
                    .find_map(|row| {
                        if col < row.pats.len() {
                            if let Pat::Lit(lit, _) = &row.pats[col] {
                                Some(hir_lit_to_core(lit))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap_or(Literal::Int(0));

                // Register variable bindings for wild/var rows in this
                // literal group (same logic as constructor groups above).
                for row in group_rows {
                    if col < row.pats.len() {
                        if matches!(pat_head(ctx, &row.pats[col]), PatHead::Wild) {
                            register_wildcard_bindings(ctx, &row.pats[col], &scrutinee);
                        }
                    }
                }

                // Literal branches have no field variables
                let sub_matrix = specialize_matrix(
                    ctx,
                    &matrix,
                    col,
                    head,
                    &[],
                    group_rows,
                    row_span,
                );
                let sub_tree = build_decision_tree(ctx, sub_matrix, row_span);
                branches.push((AltCon::Lit(core_lit), vec![], sub_tree));
            }
            PatHead::Wild => unreachable!("wild should not appear as group key"),
        }
    }

    // Build default branch from rows that match anything at this column.
    // Register variable bindings for patterns being removed from column `col`.
    let default = if !default_rows.is_empty() {
        for row in &default_rows {
            if col < row.pats.len() {
                register_wildcard_bindings(ctx, &row.pats[col], &scrutinee);
            }
        }
        let sub_matrix = default_matrix(ctx, &matrix, col, &default_rows, row_span);
        Some(Box::new(build_decision_tree(ctx, sub_matrix, row_span)))
    } else {
        None
    };

    DecisionTree::Switch {
        scrutinee,
        branches,
        default,
        span: row_span,
    }
}

/// Specialize the match matrix for a given constructor in a column.
///
/// For rows matching the constructor, replace the column with sub-pattern columns.
/// For wildcard rows, expand to wildcard sub-patterns.
fn specialize_matrix(
    ctx: &LowerContext,
    matrix: &MatchMatrix,
    col: usize,
    head: &PatHead,
    field_vars: &[Var],
    group_rows: &[&ClauseRow],
    span: Span,
) -> MatchMatrix {
    let arity = field_vars.len();

    // New scrutinee vector: replace col with field variables
    let mut new_scrutinees = Vec::with_capacity(matrix.scrutinees.len() - 1 + arity);
    for (i, s) in matrix.scrutinees.iter().enumerate() {
        if i == col {
            new_scrutinees.extend(field_vars.iter().cloned());
        } else {
            new_scrutinees.push(s.clone());
        }
    }

    let mut new_rows = Vec::new();
    for row in group_rows {
        let pat = if col < row.pats.len() {
            &row.pats[col]
        } else {
            // Pad with wildcard if row is shorter
            &Pat::Wild(span)
        };

        // Get sub-patterns for this constructor
        let sub_pats = pat_sub_patterns(ctx, pat, arity as u32, span);

        // Build new pattern row: replace col with sub-patterns
        let mut new_pats = Vec::with_capacity(row.pats.len() - 1 + arity);
        for (i, p) in row.pats.iter().enumerate() {
            if i == col {
                new_pats.extend(sub_pats.iter().cloned());
            } else {
                new_pats.push(p.clone());
            }
        }

        new_rows.push(ClauseRow {
            pats: new_pats,
            guard: row.guard.clone(),
            rhs: row.rhs.clone(),
            span: row.span,
            row_index: row.row_index,
        });
    }

    MatchMatrix {
        scrutinees: new_scrutinees,
        rows: new_rows,
    }
}

/// Build the default sub-matrix (for rows matching any constructor).
fn default_matrix(
    _ctx: &LowerContext,
    matrix: &MatchMatrix,
    col: usize,
    default_rows: &[&ClauseRow],
    span: Span,
) -> MatchMatrix {
    // Remove the matched column
    let new_scrutinees: Vec<Var> = matrix
        .scrutinees
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != col)
        .map(|(_, s)| s.clone())
        .collect();

    let new_rows: Vec<ClauseRow> = default_rows
        .iter()
        .map(|row| {
            let new_pats: Vec<hir::Pat> = row
                .pats
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != col)
                .map(|(_, p)| p.clone())
                .collect();
            ClauseRow {
                pats: new_pats,
                guard: row.guard.clone(),
                rhs: row.rhs.clone(),
                span: row.span,
                row_index: row.row_index,
            }
        })
        .collect();

    MatchMatrix {
        scrutinees: new_scrutinees,
        rows: new_rows,
    }
}

/// Register variable bindings from a wildcard/var pattern into the context.
fn register_wildcard_bindings(ctx: &mut LowerContext, pat: &hir::Pat, scrutinee: &Var) {
    match pat {
        Pat::Var(_name, def_id, _) => {
            ctx.register_var(*def_id, scrutinee.clone());
        }
        Pat::As(_name, def_id, inner, _) => {
            ctx.register_var(*def_id, scrutinee.clone());
            register_wildcard_bindings(ctx, inner, scrutinee);
        }
        Pat::Ann(inner, _, _) => {
            register_wildcard_bindings(ctx, inner, scrutinee);
        }
        _ => {}
    }
}

/// Convert a decision tree to Core IR.
fn tree_to_core(ctx: &mut LowerContext, tree: DecisionTree) -> LowerResult<core::Expr> {
    match tree {
        DecisionTree::Leaf {
            bindings: _,
            guard,
            rhs,
            span,
            row_index: _,
        } => {
            let core_rhs = lower_expr(ctx, &rhs)?;

            if let Some(guard_expr) = guard {
                let core_guard = lower_expr(ctx, &guard_expr)?;
                Ok(make_if(core_guard, core_rhs, make_pattern_error(span), span))
            } else {
                Ok(core_rhs)
            }
        }

        DecisionTree::Switch {
            scrutinee,
            branches,
            default,
            span,
        } => {
            let scrut_expr = core::Expr::Var(scrutinee.clone(), span);

            let mut alts: Vec<Alt> = Vec::new();

            for (con, vars, sub_tree) in branches {
                // For existential constructors, push dict scope so the
                // sub-tree's RHS can resolve class method calls via dictionary.
                let existential_classes = if let AltCon::DataCon(ref dc) = con {
                    if let Some(info) = ctx.lookup_constructor_by_name(dc.name) {
                        if info.existential_dict_count > 0 {
                            info.existential_classes.clone()
                        } else {
                            vec![]
                        }
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                };

                // For existential constructors, the first N vars (where N =
                // existential_dict_count) are dictionary fields. Register them
                // as dicts so method calls in the sub-tree can resolve them.
                if !existential_classes.is_empty() {
                    ctx.push_dict_scope();
                    for (i, class_name) in existential_classes.iter().enumerate() {
                        if let Some(dict_var) = vars.get(i) {
                            ctx.register_dict(*class_name, dict_var.clone());
                        }
                    }
                }

                let sub_rhs = tree_to_core(ctx, sub_tree)?;

                if !existential_classes.is_empty() {
                    ctx.pop_dict_scope();
                }

                alts.push(Alt {
                    con,
                    binders: vars,
                    rhs: sub_rhs,
                });
            }

            if let Some(def_tree) = default {
                let def_rhs = tree_to_core(ctx, *def_tree)?;
                alts.push(Alt {
                    con: AltCon::Default,
                    binders: vec![],
                    rhs: def_rhs,
                });
            }

            // If no default and no branches, generate error
            if alts.is_empty() {
                return Ok(make_pattern_error(span));
            }

            Ok(core::Expr::Case(
                Box::new(scrut_expr),
                alts,
                Ty::Error,
                span,
            ))
        }

        DecisionTree::Fail(span) => Ok(make_pattern_error(span)),
    }
}

/// Check the decision tree for non-exhaustive patterns and emit warnings.
fn check_exhaustiveness_tree(ctx: &mut LowerContext, tree: &DecisionTree, func_name: &str) {
    match tree {
        DecisionTree::Fail(span) => {
            ctx.warn(format!(
                "warning: Pattern match(es) are non-exhaustive in '{}'\n\
                 Patterns not matched: (could not determine missing patterns)",
                func_name
            ));
        }
        DecisionTree::Switch {
            scrutinee,
            branches,
            default,
            span,
        } => {
            // Check if all constructors of the type are covered
            let covered_cons: Vec<Symbol> = branches
                .iter()
                .filter_map(|(con, _, _)| match con {
                    AltCon::DataCon(dc) => Some(dc.name),
                    _ => None,
                })
                .collect();

            // Get type name from first branch
            if let Some((AltCon::DataCon(dc), _, _)) = branches.first() {
                let type_name = dc.ty_con.name;
                let all_cons = ctx.constructors_for_type_name(type_name);

                if !all_cons.is_empty() && default.is_none() {
                    let missing: Vec<Symbol> = all_cons
                        .iter()
                        .filter(|(_, name, _)| !covered_cons.contains(name))
                        .map(|(_, name, _)| *name)
                        .collect();

                    if !missing.is_empty() {
                        let missing_names: Vec<&str> =
                            missing.iter().map(|s| s.as_str()).collect();
                        ctx.warn(format!(
                            "warning: Pattern match(es) are non-exhaustive in '{}'\n\
                             Patterns not matched: {}",
                            func_name,
                            missing_names.join(", ")
                        ));
                    }
                }
            }

            // Recurse into sub-trees
            for (_, _, sub_tree) in branches {
                check_exhaustiveness_tree(ctx, sub_tree, func_name);
            }
            if let Some(def) = default {
                check_exhaustiveness_tree(ctx, def, func_name);
            }
        }
        DecisionTree::Leaf { .. } => {
            // Leaf is fine — match succeeded
        }
    }
}

/// Check for redundant/unreachable equations.
fn check_overlap(
    ctx: &mut LowerContext,
    tree: &DecisionTree,
    total_equations: usize,
    func_name: &str,
) {
    let mut reached = std::collections::HashSet::new();
    collect_reached_rows(tree, &mut reached);

    for i in 0..total_equations {
        if !reached.contains(&i) {
            ctx.warn(format!(
                "warning: Pattern match is redundant in '{}'\n\
                 Equation {} is never reached",
                func_name,
                i + 1
            ));
        }
    }
}

/// Collect all row indices that are reachable in the decision tree.
fn collect_reached_rows(tree: &DecisionTree, reached: &mut std::collections::HashSet<usize>) {
    match tree {
        DecisionTree::Leaf { row_index, .. } => {
            reached.insert(*row_index);
        }
        DecisionTree::Switch {
            branches, default, ..
        } => {
            for (_, _, sub_tree) in branches {
                collect_reached_rows(sub_tree, reached);
            }
            if let Some(def) = default {
                collect_reached_rows(def, reached);
            }
        }
        DecisionTree::Fail(_) => {}
    }
}

/// Linear (equation-by-equation) compilation for view patterns and guards.
/// This preserves the original approach for cases the decision tree doesn't handle.
fn compile_equations_linear(
    ctx: &mut LowerContext,
    equations: &[hir::Equation],
    args: &[Var],
    span: Span,
) -> LowerResult<Vec<Alt>> {
    if equations.is_empty() {
        return Ok(vec![Alt {
            con: AltCon::Default,
            binders: vec![],
            rhs: make_pattern_error(span),
        }]);
    }

    let eq = &equations[0];
    let has_view = eq.pats.len() == 1 && matches!(&eq.pats[0], Pat::View(_, _, _));

    if has_view {
        // View pattern equation: compile with fallthrough to remaining equations
        let Pat::View(view_expr, result_pat, view_span) = &eq.pats[0] else {
            unreachable!()
        };

        // Bind pattern variables
        for (i, pat) in eq.pats.iter().enumerate() {
            let arg_var = args.get(i).cloned();
            bind_pattern_vars(ctx, pat, arg_var.as_ref());
        }

        let rhs = if eq.guards.is_empty() {
            lower_expr(ctx, &eq.rhs)?
        } else {
            compile_guarded_rhs(ctx, &eq.guards, &eq.rhs, eq.span)?
        };

        let remaining_alts =
            compile_equations_linear(ctx, &equations[1..], args, span)?;

        let fallthrough_expr = if let Some(arg) = args.first() {
            core::Expr::Case(
                Box::new(core::Expr::Var(arg.clone(), span)),
                remaining_alts,
                Ty::Error,
                span,
            )
        } else {
            make_pattern_error(span)
        };

        let core_view_expr = lower_expr(ctx, view_expr)?;
        let arg_var = args
            .first()
            .cloned()
            .unwrap_or_else(|| ctx.fresh_var("_arg", Ty::Error, *view_span));
        let applied = core::Expr::App(
            Box::new(core_view_expr),
            Box::new(core::Expr::Var(arg_var.clone(), *view_span)),
            *view_span,
        );

        let inner_alt = lower_pat_to_alt(ctx, result_pat, rhs, eq.span)?;
        let fallthrough_alt = Alt {
            con: AltCon::Default,
            binders: vec![],
            rhs: fallthrough_expr,
        };

        let view_case = core::Expr::Case(
            Box::new(applied),
            vec![inner_alt, fallthrough_alt],
            Ty::Error,
            *view_span,
        );

        Ok(vec![Alt {
            con: AltCon::Default,
            binders: vec![arg_var],
            rhs: view_case,
        }])
    } else {
        let mut alts = Vec::new();

        // Bind pattern variables
        for (i, pat) in eq.pats.iter().enumerate() {
            let arg_var = args.get(i).cloned();
            bind_pattern_vars(ctx, pat, arg_var.as_ref());
        }

        // For existential constructors in patterns, push dict scope
        // so the RHS can resolve class method calls via the dictionary.
        let existential_classes: Vec<Symbol> = eq
            .pats
            .iter()
            .flat_map(|pat| get_existential_classes_from_pat(ctx, pat))
            .collect();
        if !existential_classes.is_empty() {
            ctx.push_dict_scope();
            for class_name in &existential_classes {
                let dict_var = ctx.fresh_var(
                    &format!("$dict_{}", class_name.as_str()),
                    Ty::Error,
                    span,
                );
                ctx.register_dict(*class_name, dict_var);
            }
        }

        let rhs = if eq.guards.is_empty() {
            lower_expr(ctx, &eq.rhs)?
        } else {
            compile_guarded_rhs(ctx, &eq.guards, &eq.rhs, eq.span)?
        };

        if !existential_classes.is_empty() {
            ctx.pop_dict_scope();
        }

        if eq.pats.is_empty() {
            alts.push(Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs,
            });
        } else if eq.pats.len() == 1 {
            let expanded_alts = lower_pat_with_or_to_alts(ctx, &eq.pats[0], rhs, eq.span)?;
            alts.extend(expanded_alts);
        } else {
            let tuple_alt = compile_tuple_pattern(ctx, &eq.pats, rhs, eq.span)?;
            alts.push(tuple_alt);
        }

        if equations.len() > 1 {
            let remaining =
                compile_equations_linear(ctx, &equations[1..], args, span)?;
            alts.extend(remaining);
        } else {
            alts.push(Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: make_pattern_error(span),
            });
        }

        Ok(alts)
    }
}

/// Bind pattern variables in the context so they can be referenced in the RHS.
///
/// For a variable pattern like `x`, this registers the HIR DefId to map to
/// the corresponding Core variable (either the arg variable or a fresh one).
pub fn bind_pattern_vars(ctx: &mut LowerContext, pat: &hir::Pat, arg_var: Option<&Var>) {
    match pat {
        Pat::Var(_name, def_id, _span) => {
            // For a simple variable pattern, bind it to the arg variable
            // The arg variable comes from the scrutinee of the case expression
            if let Some(arg) = arg_var {
                ctx.register_var(*def_id, arg.clone());
            } else {
                // No arg var provided - create a fresh one
                let var = Var {
                    name: *_name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                };
                ctx.register_var(*def_id, var);
            }
        }
        Pat::Wild(_) => {
            // Wildcards don't bind any variables
        }
        Pat::Lit(_, _) => {
            // Literals don't bind any variables
        }
        Pat::Con(_, sub_pats, _) => {
            // Recursively bind sub-pattern variables
            // Each sub-pattern will need its own fresh variable
            for sub_pat in sub_pats {
                bind_pattern_vars(ctx, sub_pat, None);
            }
        }
        Pat::RecordCon(_, field_pats, _) => {
            // Recursively bind field pattern variables
            for fp in field_pats {
                bind_pattern_vars(ctx, &fp.pat, None);
            }
        }
        Pat::As(name, def_id, inner_pat, _) => {
            // As-pattern binds the name and also any variables in the inner pattern
            if let Some(arg) = arg_var {
                ctx.register_var(*def_id, arg.clone());
            } else {
                let var = Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                };
                ctx.register_var(*def_id, var);
            }
            bind_pattern_vars(ctx, inner_pat, None);
        }
        Pat::Or(left, right, _) => {
            // Or-patterns: both branches should bind the same variables
            bind_pattern_vars(ctx, left, arg_var);
            // Note: right should bind the same vars, but we skip for now
        }
        Pat::Ann(inner, _, _) => {
            bind_pattern_vars(ctx, inner, arg_var);
        }
        Pat::View(_, result_pat, _) => {
            // View pattern: variables are bound in the result pattern
            bind_pattern_vars(ctx, result_pat, None);
        }
        Pat::Error(_) => {
            // Error patterns don't bind variables
        }
    }
}

/// Compile a guarded RHS into nested conditionals.
///
/// In Haskell, guards on the same equation are ANDed together:
/// ```haskell
/// f x | g1, g2 = e  -- means: if g1 && g2 then e
/// ```
fn compile_guarded_rhs(
    ctx: &mut LowerContext,
    guards: &[hir::Guard],
    rhs: &hir::Expr,
    span: Span,
) -> LowerResult<core::Expr> {
    if guards.is_empty() {
        return lower_expr(ctx, rhs);
    }

    let rhs_core = lower_expr(ctx, rhs)?;

    // Multiple guards on the same RHS are ANDed together
    // f x | g1, g2 = e  compiles to: if g1 then (if g2 then e else error) else error
    // This is equivalent to: if (g1 && g2) then e else error
    let mut result = rhs_core;

    // Process guards in reverse order to build nested ifs correctly
    // The innermost if checks the last guard
    for guard in guards.iter().rev() {
        let cond = lower_expr(ctx, &guard.cond)?;
        result = make_if(cond, result, make_pattern_error(span), span);
    }

    Ok(result)
}

/// Compile a tuple of patterns into a single alternative.
fn compile_tuple_pattern(
    ctx: &mut LowerContext,
    pats: &[hir::Pat],
    rhs: core::Expr,
    span: Span,
) -> LowerResult<Alt> {
    // Create a tuple pattern that matches all the individual patterns
    let tuple_name = Symbol::intern(&format!("({})", ",".repeat(pats.len() - 1)));

    // Create binders and potentially nested cases for each sub-pattern
    let mut binders = Vec::with_capacity(pats.len());
    let mut inner_rhs = rhs;

    for (i, pat) in pats.iter().enumerate().rev() {
        let binder_name = format!("arg{}", i);
        let binder = ctx.fresh_var(&binder_name, Ty::Error, span);

        match pat {
            Pat::Var(name, _, _) => {
                let var = Var {
                    name: *name,
                    id: ctx.fresh_id(),
                    ty: Ty::Error,
                };
                binders.push(var);
            }
            Pat::Wild(_) => {
                binders.push(binder);
            }
            _ => {
                // Complex sub-pattern: need nested case
                let sub_alt = lower_pat_to_alt(ctx, pat, inner_rhs.clone(), span)?;
                let default_rhs = make_pattern_error(span);
                let default_alt = Alt {
                    con: AltCon::Default,
                    binders: vec![],
                    rhs: default_rhs,
                };

                inner_rhs = core::Expr::Case(
                    Box::new(core::Expr::Var(binder.clone(), span)),
                    vec![sub_alt, default_alt],
                    Ty::Error,
                    span,
                );
                binders.push(binder);
            }
        }
    }

    binders.reverse();

    let tuple_tycon = TyCon::new(tuple_name, Kind::Star);
    let tuple_con = DataCon {
        name: tuple_name,
        ty_con: tuple_tycon,
        tag: 0,
        arity: pats.len() as u32,
    };

    Ok(Alt {
        con: AltCon::DataCon(tuple_con),
        binders,
        rhs: inner_rhs,
    })
}

/// Create an if expression (case on Bool).
fn make_if(cond: core::Expr, then_br: core::Expr, else_br: core::Expr, span: Span) -> core::Expr {
    let bool_tycon = TyCon::new(Symbol::intern("Bool"), Kind::Star);
    let true_con = DataCon {
        name: Symbol::intern("True"),
        ty_con: bool_tycon.clone(),
        tag: 1,
        arity: 0,
    };
    let false_con = DataCon {
        name: Symbol::intern("False"),
        ty_con: bool_tycon,
        tag: 0,
        arity: 0,
    };

    core::Expr::Case(
        Box::new(cond),
        vec![
            Alt {
                con: AltCon::DataCon(true_con),
                binders: vec![],
                rhs: then_br,
            },
            Alt {
                con: AltCon::DataCon(false_con),
                binders: vec![],
                rhs: else_br,
            },
        ],
        Ty::Error,
        span,
    )
}

/// Get the canonical constructor tag for a given constructor name.
///
/// For builtin types (Bool, Maybe, Either, List, Unit), we use fixed tags
/// that match what the LLVM codegen expects. For user-defined types,
/// we fall back to the DefId index.
fn get_constructor_tag(name: &str, fallback: u32) -> u32 {
    match name {
        // Bool constructors
        "False" => 0,
        "True" => 1,

        // Maybe constructors
        "Nothing" => 0,
        "Just" => 1,

        // Either constructors
        "Left" => 0,
        "Right" => 1,

        // List constructors
        "[]" => 0,
        ":" => 1,

        // Unit constructor
        "()" => 0,

        // Tuple constructors (single constructor per type = tag 0)
        "(,)" => 0,
        "(,,)" => 0,
        "(,,,)" => 0,
        "(,,,,)" => 0,
        "(,,,,,)" => 0,
        "(,,,,,,)" => 0,
        "(,,,,,,,)" => 0,

        // Ordering constructors
        "LT" => 0,
        "EQ" => 1,
        "GT" => 2,

        // GHC.Generics representation constructors
        "U1" => 0,
        "K1" => 0,
        "M1" => 0,
        "L1" => 0,
        "R1" => 1,
        ":*:" => 0,

        // User-defined constructors: use fallback
        _ => fallback,
    }
}

/// Expand or-patterns into multiple alternatives.
///
/// Given a pattern like `Left x | Right x`, this produces alternatives
/// for both `Left x` and `Right x` with the same RHS.
/// Deeply flatten all or-patterns in a pattern, including nested ones.
///
/// For example:
/// - `Just (Left x | Right x)` → `[Just (Left x), Just (Right x)]`
/// - `(Left a | Right a, True | False)` → `[(Left a, True), (Left a, False), (Right a, True), (Right a, False)]`
pub fn flatten_nested_or_patterns(pat: &hir::Pat) -> Vec<hir::Pat> {
    match pat {
        Pat::Or(left, right, _) => {
            // First flatten both branches
            let mut result = flatten_nested_or_patterns(left);
            result.extend(flatten_nested_or_patterns(right));
            result
        }
        Pat::Con(def_ref, sub_pats, span) => {
            // Flatten each sub-pattern and compute cross-product
            flatten_con_pattern(*def_ref, sub_pats, *span)
        }
        Pat::RecordCon(def_ref, field_pats, span) => {
            // For record patterns, flatten field patterns
            flatten_record_pattern(*def_ref, field_pats, *span)
        }
        Pat::As(name, def_id, inner, span) => {
            // Flatten inner pattern and wrap each with As
            let inner_flattened = flatten_nested_or_patterns(inner);
            inner_flattened
                .into_iter()
                .map(|p| Pat::As(*name, *def_id, Box::new(p), *span))
                .collect()
        }
        Pat::Ann(inner, ty, span) => {
            // Flatten inner and wrap with annotation
            let inner_flattened = flatten_nested_or_patterns(inner);
            inner_flattened
                .into_iter()
                .map(|p| Pat::Ann(Box::new(p), ty.clone(), *span))
                .collect()
        }
        Pat::View(view_expr, result_pat, span) => {
            // Flatten result pattern and wrap with view
            let inner_flattened = flatten_nested_or_patterns(result_pat);
            inner_flattened
                .into_iter()
                .map(|p| Pat::View(view_expr.clone(), Box::new(p), *span))
                .collect()
        }
        // Patterns without sub-patterns - return as-is
        Pat::Wild(span) => vec![Pat::Wild(*span)],
        Pat::Var(name, def_id, span) => vec![Pat::Var(*name, *def_id, *span)],
        Pat::Lit(lit, span) => vec![Pat::Lit(lit.clone(), *span)],
        Pat::Error(span) => vec![Pat::Error(*span)],
    }
}

/// Flatten a constructor pattern with potentially nested or-patterns in sub-patterns.
fn flatten_con_pattern(
    def_ref: bhc_hir::DefRef,
    sub_pats: &[hir::Pat],
    span: Span,
) -> Vec<hir::Pat> {
    if sub_pats.is_empty() {
        return vec![Pat::Con(def_ref, vec![], span)];
    }

    // Flatten each sub-pattern
    let flattened_subs: Vec<Vec<hir::Pat>> =
        sub_pats.iter().map(flatten_nested_or_patterns).collect();

    // Compute cross-product of all combinations
    let combinations = cross_product(&flattened_subs);

    // Create a Con pattern for each combination
    combinations
        .into_iter()
        .map(|combo| Pat::Con(def_ref, combo, span))
        .collect()
}

/// Flatten a record constructor pattern with potentially nested or-patterns in fields.
fn flatten_record_pattern(
    def_ref: bhc_hir::DefRef,
    field_pats: &[hir::FieldPat],
    span: Span,
) -> Vec<hir::Pat> {
    if field_pats.is_empty() {
        return vec![Pat::RecordCon(def_ref, vec![], span)];
    }

    // Flatten each field pattern
    let flattened_fields: Vec<Vec<hir::FieldPat>> = field_pats
        .iter()
        .map(|fp| {
            let flattened = flatten_nested_or_patterns(&fp.pat);
            flattened
                .into_iter()
                .map(|p| hir::FieldPat {
                    name: fp.name,
                    pat: p,
                    span: fp.span,
                })
                .collect()
        })
        .collect();

    // Compute cross-product
    let combinations = cross_product(&flattened_fields);

    // Create a RecordCon pattern for each combination
    combinations
        .into_iter()
        .map(|combo| Pat::RecordCon(def_ref, combo, span))
        .collect()
}

/// Compute the cross-product of multiple vectors.
/// For example: [[a, b], [1, 2]] → [[a, 1], [a, 2], [b, 1], [b, 2]]
fn cross_product<T: Clone>(vecs: &[Vec<T>]) -> Vec<Vec<T>> {
    if vecs.is_empty() {
        return vec![vec![]];
    }

    let first = &vecs[0];
    let rest = cross_product(&vecs[1..]);

    let mut result = Vec::new();
    for item in first {
        for r in &rest {
            let mut combo = vec![item.clone()];
            combo.extend(r.clone());
            result.push(combo);
        }
    }
    result
}

/// Lower a pattern that may contain or-patterns (including nested ones) to multiple alternatives.
///
/// For `Left x | Right x -> e`, this produces:
/// - `Left x -> e`
/// - `Right x -> e`
///
/// For nested or-patterns like `Just (Left x | Right x) -> e`, this produces:
/// - `Just (Left x) -> e`
/// - `Just (Right x) -> e`
pub fn lower_pat_with_or_to_alts(
    ctx: &mut LowerContext,
    pat: &hir::Pat,
    rhs: core::Expr,
    span: Span,
) -> LowerResult<Vec<Alt>> {
    // Use deep flattening to handle nested or-patterns
    let expanded = flatten_nested_or_patterns(pat);
    let mut alts = Vec::with_capacity(expanded.len());

    for sub_pat in expanded {
        let alt = lower_pat_to_alt(ctx, &sub_pat, rhs.clone(), span)?;
        alts.push(alt);
    }

    Ok(alts)
}

/// Get existential class names from a pattern (for dictionary scope setup).
fn get_existential_classes_from_pat(ctx: &LowerContext, pat: &hir::Pat) -> Vec<Symbol> {
    match pat {
        hir::Pat::Con(def_ref, _, _) | hir::Pat::RecordCon(def_ref, _, _) => {
            if let Some(info) = ctx.lookup_constructor(def_ref.def_id) {
                if info.existential_dict_count > 0 {
                    return info.existential_classes.clone();
                }
            }
            vec![]
        }
        _ => vec![],
    }
}

/// Create a pattern match error expression.
pub fn make_pattern_error(span: Span) -> core::Expr {
    let error_var = Var {
        name: Symbol::intern("error"),
        id: VarId::new(0),
        ty: Ty::Error,
    };
    let msg = core::Expr::Lit(
        Literal::String(Symbol::intern("Non-exhaustive patterns")),
        Ty::Error,
        span,
    );
    core::Expr::App(
        Box::new(core::Expr::Var(error_var, span)),
        Box::new(msg),
        span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_hir::{DefId, DefRef};
    use bhc_index::Idx;

    #[test]
    fn test_lower_wildcard_pattern() {
        let mut ctx = LowerContext::new();
        let pat = Pat::Wild(Span::default());
        let rhs = core::Expr::Lit(Literal::Int(42), Ty::Error, Span::default());

        let result = lower_pat_to_alt(&mut ctx, &pat, rhs, Span::default());
        assert!(result.is_ok());

        let alt = result.unwrap();
        assert!(matches!(alt.con, AltCon::Default));
    }

    #[test]
    fn test_lower_literal_pattern() {
        let mut ctx = LowerContext::new();
        let pat = Pat::Lit(hir::Lit::Int(42), Span::default());
        let rhs = core::Expr::Lit(Literal::Int(1), Ty::Error, Span::default());

        let result = lower_pat_to_alt(&mut ctx, &pat, rhs, Span::default());
        assert!(result.is_ok());

        let alt = result.unwrap();
        assert!(matches!(alt.con, AltCon::Lit(Literal::Int(42))));
    }

    #[test]
    fn test_lower_var_pattern() {
        let mut ctx = LowerContext::new();
        let x = Symbol::intern("x");
        let def_id = DefId::new(0);
        let pat = Pat::Var(x, def_id, Span::default());
        let rhs = core::Expr::Lit(Literal::Int(1), Ty::Error, Span::default());

        let result = lower_pat_to_alt(&mut ctx, &pat, rhs, Span::default());
        assert!(result.is_ok());

        let alt = result.unwrap();
        assert!(matches!(alt.con, AltCon::Default));
        assert_eq!(alt.binders.len(), 1);
        assert_eq!(alt.binders[0].name, x);
    }

    #[test]
    fn test_flatten_or_patterns() {
        let span = Span::default();
        let def_id = DefId::new(100);
        let x = Symbol::intern("x");

        // Simple pattern - no expansion
        let simple_pat = Pat::Var(x, def_id, span);
        let expanded = flatten_nested_or_patterns(&simple_pat);
        assert_eq!(expanded.len(), 1);

        // Or-pattern - should expand to 2
        let left = Box::new(Pat::Lit(hir::Lit::Int(1), span));
        let right = Box::new(Pat::Lit(hir::Lit::Int(2), span));
        let or_pat = Pat::Or(left, right, span);
        let expanded = flatten_nested_or_patterns(&or_pat);
        assert_eq!(expanded.len(), 2);
    }

    #[test]
    fn test_lower_or_pattern_to_alts() {
        let mut ctx = LowerContext::new();
        let span = Span::default();
        let rhs = core::Expr::Lit(Literal::Int(42), Ty::Error, span);

        // Or-pattern: 1 | 2 -> 42
        let left = Box::new(Pat::Lit(hir::Lit::Int(1), span));
        let right = Box::new(Pat::Lit(hir::Lit::Int(2), span));
        let or_pat = Pat::Or(left, right, span);

        let result = lower_pat_with_or_to_alts(&mut ctx, &or_pat, rhs, span);
        assert!(result.is_ok());

        let alts = result.unwrap();
        assert_eq!(alts.len(), 2);
        assert!(matches!(alts[0].con, AltCon::Lit(Literal::Int(1))));
        assert!(matches!(alts[1].con, AltCon::Lit(Literal::Int(2))));
    }

    // ================================================================
    // Decision tree tests
    // ================================================================

    /// Helper: build a ClauseRow from patterns and a literal RHS.
    fn clause(pats: Vec<Pat>, rhs_val: i128, idx: usize) -> ClauseRow {
        ClauseRow {
            pats,
            guard: None,
            rhs: hir::Expr::Lit(hir::Lit::Int(rhs_val), Span::default()),
            span: Span::default(),
            row_index: idx,
        }
    }

    #[test]
    fn test_decision_tree_single_wildcard() {
        // f _ = 1  →  Leaf
        let mut ctx = LowerContext::new();
        let arg0 = ctx.fresh_var("arg0", Ty::Error, Span::default());
        let matrix = MatchMatrix {
            scrutinees: vec![arg0],
            rows: vec![clause(vec![Pat::Wild(Span::default())], 1, 0)],
        };
        let tree = build_decision_tree(&mut ctx, matrix, Span::default());
        assert!(matches!(tree, DecisionTree::Leaf { row_index: 0, .. }));
    }

    #[test]
    fn test_decision_tree_single_var() {
        // f x = 1  →  Leaf (x bound to arg0)
        let mut ctx = LowerContext::new();
        let arg0 = ctx.fresh_var("arg0", Ty::Error, Span::default());
        let x_id = DefId::new(50);
        let matrix = MatchMatrix {
            scrutinees: vec![arg0.clone()],
            rows: vec![clause(
                vec![Pat::Var(Symbol::intern("x"), x_id, Span::default())],
                1,
                0,
            )],
        };
        let tree = build_decision_tree(&mut ctx, matrix, Span::default());
        assert!(matches!(tree, DecisionTree::Leaf { row_index: 0, .. }));
        // x should be registered as arg0
        assert!(ctx.lookup_var(x_id).is_some());
    }

    #[test]
    fn test_decision_tree_literal_and_default() {
        // f 0 = 1; f n = 2  →  Switch(arg0, [Lit(0) → Leaf(1)], default → Leaf(2))
        let mut ctx = LowerContext::new();
        let arg0 = ctx.fresh_var("arg0", Ty::Error, Span::default());
        let n_id = DefId::new(51);
        let matrix = MatchMatrix {
            scrutinees: vec![arg0],
            rows: vec![
                clause(vec![Pat::Lit(hir::Lit::Int(0), Span::default())], 1, 0),
                clause(
                    vec![Pat::Var(Symbol::intern("n"), n_id, Span::default())],
                    2,
                    1,
                ),
            ],
        };
        let tree = build_decision_tree(&mut ctx, matrix, Span::default());
        match &tree {
            DecisionTree::Switch {
                branches, default, ..
            } => {
                assert_eq!(branches.len(), 1);
                assert!(matches!(&branches[0].0, AltCon::Lit(Literal::Int(0))));
                assert!(default.is_some());
                assert!(matches!(
                    default.as_ref().unwrap().as_ref(),
                    DecisionTree::Leaf { row_index: 1, .. }
                ));
            }
            _ => panic!("Expected Switch"),
        }
    }

    #[test]
    fn test_decision_tree_two_constructors() {
        use crate::context::ConstructorInfo;
        // f True = 1; f False = 0  →  Switch with 2 branches, no default
        let mut ctx = LowerContext::new();
        let arg0 = ctx.fresh_var("arg0", Ty::Error, Span::default());

        let true_id = DefId::new(60);
        let false_id = DefId::new(61);
        let bool_sym = Symbol::intern("Bool");
        let true_sym = Symbol::intern("True");
        let false_sym = Symbol::intern("False");

        ctx.register_constructor(
            true_id,
            ConstructorInfo {
                name: true_sym,
                type_name: bool_sym,
                tag: 1,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        ctx.register_constructor(
            false_id,
            ConstructorInfo {
                name: false_sym,
                type_name: bool_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );

        let true_ref = DefRef {
            def_id: true_id,
            span: Span::default(),
        };
        let false_ref = DefRef {
            def_id: false_id,
            span: Span::default(),
        };

        let matrix = MatchMatrix {
            scrutinees: vec![arg0],
            rows: vec![
                clause(vec![Pat::Con(true_ref, vec![], Span::default())], 1, 0),
                clause(vec![Pat::Con(false_ref, vec![], Span::default())], 0, 1),
            ],
        };
        let tree = build_decision_tree(&mut ctx, matrix, Span::default());
        match &tree {
            DecisionTree::Switch {
                branches, default, ..
            } => {
                assert_eq!(branches.len(), 2);
                assert!(default.is_none());
            }
            _ => panic!("Expected Switch"),
        }
    }

    #[test]
    fn test_exhaustiveness_complete_bool() {
        use crate::context::ConstructorInfo;
        // f True = 1; f False = 0  →  no warnings
        let mut ctx = LowerContext::new();
        let arg0 = ctx.fresh_var("arg0", Ty::Error, Span::default());
        let bool_sym = Symbol::intern("Bool");
        let true_sym = Symbol::intern("True");
        let false_sym = Symbol::intern("False");
        let true_id = DefId::new(70);
        let false_id = DefId::new(71);

        ctx.register_constructor(
            true_id,
            ConstructorInfo {
                name: true_sym,
                type_name: bool_sym,
                tag: 1,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        ctx.register_constructor(
            false_id,
            ConstructorInfo {
                name: false_sym,
                type_name: bool_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );

        let true_ref = DefRef {
            def_id: true_id,
            span: Span::default(),
        };
        let false_ref = DefRef {
            def_id: false_id,
            span: Span::default(),
        };

        let matrix = MatchMatrix {
            scrutinees: vec![arg0],
            rows: vec![
                clause(vec![Pat::Con(true_ref, vec![], Span::default())], 1, 0),
                clause(vec![Pat::Con(false_ref, vec![], Span::default())], 0, 1),
            ],
        };
        let tree = build_decision_tree(&mut ctx, matrix, Span::default());
        check_exhaustiveness_tree(&mut ctx, &tree, "f");
        let warnings = ctx.take_warnings();
        assert!(
            warnings.is_empty(),
            "Should have no exhaustiveness warnings for complete Bool match"
        );
    }

    #[test]
    fn test_exhaustiveness_incomplete_bool() {
        use crate::context::ConstructorInfo;
        // f True = 1  →  warning about missing False
        let mut ctx = LowerContext::new();
        let arg0 = ctx.fresh_var("arg0", Ty::Error, Span::default());
        let bool_sym = Symbol::intern("Bool");
        let true_sym = Symbol::intern("True");
        let false_sym = Symbol::intern("False");
        let true_id = DefId::new(80);
        let false_id = DefId::new(81);

        ctx.register_constructor(
            true_id,
            ConstructorInfo {
                name: true_sym,
                type_name: bool_sym,
                tag: 1,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );
        ctx.register_constructor(
            false_id,
            ConstructorInfo {
                name: false_sym,
                type_name: bool_sym,
                tag: 0,
                arity: 0,
                field_names: vec![],
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );

        let true_ref = DefRef {
            def_id: true_id,
            span: Span::default(),
        };

        let matrix = MatchMatrix {
            scrutinees: vec![arg0],
            rows: vec![clause(
                vec![Pat::Con(true_ref, vec![], Span::default())],
                1,
                0,
            )],
        };
        let tree = build_decision_tree(&mut ctx, matrix, Span::default());
        check_exhaustiveness_tree(&mut ctx, &tree, "f");
        let warnings = ctx.take_warnings();
        assert!(
            !warnings.is_empty(),
            "Should warn about missing False"
        );
        assert!(warnings[0].contains("False"), "Warning should mention False");
    }

    #[test]
    fn test_overlap_detection() {
        // f _ = 1; f 0 = 2  →  row 1 is redundant
        let mut ctx = LowerContext::new();
        let arg0 = ctx.fresh_var("arg0", Ty::Error, Span::default());
        let matrix = MatchMatrix {
            scrutinees: vec![arg0],
            rows: vec![
                clause(vec![Pat::Wild(Span::default())], 1, 0),
                clause(vec![Pat::Lit(hir::Lit::Int(0), Span::default())], 2, 1),
            ],
        };
        let tree = build_decision_tree(&mut ctx, matrix, Span::default());
        check_overlap(&mut ctx, &tree, 2, "f");
        let warnings = ctx.take_warnings();
        assert!(
            !warnings.is_empty(),
            "Should warn about redundant equation"
        );
        assert!(
            warnings[0].contains("redundant"),
            "Warning should say redundant"
        );
    }

    #[test]
    fn test_decision_tree_literal_patterns() {
        // f 0 = 1; f 1 = 2; f _ = 3  →  Switch on literals + default
        let mut ctx = LowerContext::new();
        let arg0 = ctx.fresh_var("arg0", Ty::Error, Span::default());
        let matrix = MatchMatrix {
            scrutinees: vec![arg0],
            rows: vec![
                clause(vec![Pat::Lit(hir::Lit::Int(0), Span::default())], 1, 0),
                clause(vec![Pat::Lit(hir::Lit::Int(1), Span::default())], 2, 1),
                clause(vec![Pat::Wild(Span::default())], 3, 2),
            ],
        };
        let tree = build_decision_tree(&mut ctx, matrix, Span::default());
        match &tree {
            DecisionTree::Switch {
                branches, default, ..
            } => {
                assert_eq!(branches.len(), 2, "Should have 2 literal branches");
                assert!(default.is_some(), "Should have default branch");
            }
            _ => panic!("Expected Switch"),
        }
    }

    #[test]
    fn test_or_pattern_expansion() {
        // Equation: f (1 | 2) = 42  →  expands to 2 rows
        let span = Span::default();
        let left = Box::new(Pat::Lit(hir::Lit::Int(1), span));
        let right = Box::new(Pat::Lit(hir::Lit::Int(2), span));
        let or_pat = Pat::Or(left, right, span);

        let eq = hir::Equation {
            pats: vec![or_pat],
            guards: vec![],
            rhs: hir::Expr::Lit(hir::Lit::Int(42), span),
            span,
        };

        let expanded = expand_equation_or_patterns(&eq);
        assert_eq!(expanded.len(), 2);
    }

    #[test]
    fn test_no_overlap_for_proper_dispatch() {
        // f 0 = 1; f n = 2  →  no redundancy (both equations reachable)
        let mut ctx = LowerContext::new();
        let arg0 = ctx.fresh_var("arg0", Ty::Error, Span::default());
        let n_id = DefId::new(90);
        let matrix = MatchMatrix {
            scrutinees: vec![arg0],
            rows: vec![
                clause(vec![Pat::Lit(hir::Lit::Int(0), Span::default())], 1, 0),
                clause(
                    vec![Pat::Var(Symbol::intern("n"), n_id, Span::default())],
                    2,
                    1,
                ),
            ],
        };
        let tree = build_decision_tree(&mut ctx, matrix, Span::default());
        check_overlap(&mut ctx, &tree, 2, "f");
        let warnings = ctx.take_warnings();
        assert!(
            warnings.is_empty(),
            "No equations should be redundant"
        );
    }

    #[test]
    fn test_record_pattern_with_field_ordering() {
        use crate::context::ConstructorInfo;

        let mut ctx = LowerContext::new();
        let span = Span::default();

        // Register a constructor with named fields: MkPerson { name, age }
        let con_id = DefId::new(200);
        let name_sym = Symbol::intern("name");
        let age_sym = Symbol::intern("age");
        let person_sym = Symbol::intern("Person");
        let mk_person_sym = Symbol::intern("MkPerson");

        ctx.register_constructor(
            con_id,
            ConstructorInfo {
                name: mk_person_sym,
                type_name: person_sym,
                tag: 0,
                arity: 2,
                field_names: vec![name_sym, age_sym], // canonical order: name, age
                is_newtype: false,
                existential_dict_count: 0,
                existential_classes: vec![],
            },
        );

        // Create a record pattern with fields in different order: { age = a, name = n }
        let n_def_id = DefId::new(201);
        let a_def_id = DefId::new(202);
        let n = Symbol::intern("n");
        let a = Symbol::intern("a");

        let field_pats = vec![
            hir::FieldPat {
                name: age_sym, // Out of order
                pat: Pat::Var(a, a_def_id, span),
                span,
            },
            hir::FieldPat {
                name: name_sym,
                pat: Pat::Var(n, n_def_id, span),
                span,
            },
        ];

        let def_ref = DefRef {
            def_id: con_id,
            span,
        };
        let pat = Pat::RecordCon(def_ref, field_pats, span);
        let rhs = core::Expr::Lit(Literal::Int(1), Ty::Error, span);

        let result = lower_pat_to_alt(&mut ctx, &pat, rhs, span);
        assert!(result.is_ok());

        let alt = result.unwrap();
        // Should have 2 binders in canonical order (name, age)
        assert_eq!(alt.binders.len(), 2);
    }
}
