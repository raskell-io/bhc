//! Pattern compilation for HIR to Core lowering.
//!
//! This module handles the compilation of HIR patterns into Core case
//! alternatives. The main challenge is compiling multi-equation function
//! definitions with overlapping patterns into efficient decision trees.
//!
//! ## Pattern Compilation Strategy
//!
//! We use a simplified version of the algorithm from:
//! "The Implementation of Functional Programming Languages" by SPJ
//!
//! For now, we use a straightforward approach that generates one alternative
//! per equation. This is correct but may generate redundant checks.

use bhc_core::{self as core, Alt, AltCon, DataCon, Literal, Var, VarId};
use bhc_hir::{self as hir, Pat, ValueDef};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Kind, Ty, TyCon};

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
            let core_lit = match lit {
                hir::Lit::Int(n) => Literal::Int(*n as i64),
                hir::Lit::Float(f) => Literal::Float(*f as f32),
                hir::Lit::Char(c) => Literal::Char(*c),
                hir::Lit::String(s) => Literal::String(*s),
            };
            Ok(Alt {
                con: AltCon::Lit(core_lit),
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
            let (con_name, type_name, tag) =
                if let Some(info) = ctx.lookup_constructor(def_ref.def_id) {
                    (info.name, info.type_name, info.tag)
                } else if let Some(var) = ctx.lookup_var(def_ref.def_id) {
                    // Fallback for constructors not in the map - use name-based lookup
                    let tag = get_constructor_tag(var.name.as_str(), def_ref.def_id.index() as u32);
                    (var.name, Symbol::intern("DataType"), tag)
                } else {
                    // Last resort fallback
                    let name = Symbol::intern("Con");
                    (
                        name,
                        Symbol::intern("DataType"),
                        def_ref.def_id.index() as u32,
                    )
                };

            let tycon = TyCon::new(type_name, Kind::Star);
            let con = DataCon {
                name: con_name,
                ty_con: tycon,
                tag,
                arity: sub_pats.len() as u32,
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

/// Compile pattern matching for multiple equations.
///
/// This takes a function definition with multiple equations and compiles
/// them into a list of Core case alternatives.
pub fn compile_match(
    ctx: &mut LowerContext,
    value_def: &ValueDef,
    args: &[Var],
) -> LowerResult<Vec<Alt>> {
    compile_equations(ctx, &value_def.equations, args, value_def.span)
}

/// Check if a pattern is (or contains at the top level) a view pattern.
fn is_view_pattern(pat: &hir::Pat) -> bool {
    matches!(pat, Pat::View(_, _, _))
}

/// Compile a slice of equations into Core case alternatives.
/// View pattern equations are compiled with fallthrough to remaining equations.
fn compile_equations(
    ctx: &mut LowerContext,
    equations: &[hir::Equation],
    args: &[Var],
    span: Span,
) -> LowerResult<Vec<Alt>> {
    if equations.is_empty() {
        // No equations left - add a default error case
        return Ok(vec![Alt {
            con: AltCon::Default,
            binders: vec![],
            rhs: make_pattern_error(span),
        }]);
    }

    // Check if the first equation has a view pattern (single-pattern case)
    let eq = &equations[0];
    let has_view = eq.pats.len() == 1 && is_view_pattern(&eq.pats[0]);

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

        // Lower the RHS
        let rhs = if eq.guards.is_empty() {
            lower_expr(ctx, &eq.rhs)?
        } else {
            compile_guarded_rhs(ctx, &eq.guards, &eq.rhs, eq.span)?
        };

        // Compile remaining equations for fallthrough
        let remaining_alts = compile_equations(ctx, &equations[1..], args, span)?;

        // Build fallthrough: case arg of { remaining_alts }
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

        // Lower the view expression
        let core_view_expr = lower_expr(ctx, view_expr)?;

        // Apply view function to the argument
        let arg_var = args
            .first()
            .cloned()
            .unwrap_or_else(|| ctx.fresh_var("_arg", Ty::Error, *view_span));
        let applied = core::Expr::App(
            Box::new(core_view_expr),
            Box::new(core::Expr::Var(arg_var.clone(), *view_span)),
            *view_span,
        );

        // Inner case: case (view arg) of { result_pat -> rhs; _ -> fallthrough }
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

        // Outer: Default alt that binds scrutinee and does the view case
        Ok(vec![Alt {
            con: AltCon::Default,
            binders: vec![arg_var],
            rhs: view_case,
        }])
    } else {
        // Normal equation (no view pattern) - use the existing flat approach
        let mut alts = Vec::new();

        // Bind pattern variables
        for (i, pat) in eq.pats.iter().enumerate() {
            let arg_var = args.get(i).cloned();
            bind_pattern_vars(ctx, pat, arg_var.as_ref());
        }

        // Lower the RHS
        let rhs = if eq.guards.is_empty() {
            lower_expr(ctx, &eq.rhs)?
        } else {
            compile_guarded_rhs(ctx, &eq.guards, &eq.rhs, eq.span)?
        };

        // Handle pattern matching
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

        // Add remaining equations
        if equations.len() > 1 {
            let remaining = compile_equations(ctx, &equations[1..], args, span)?;
            alts.extend(remaining);
        } else {
            // Last equation - add default error case
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

        // User-defined constructors: use fallback
        _ => fallback,
    }
}

/// Expand or-patterns into multiple alternatives.
///
/// Given a pattern like `Left x | Right x`, this produces alternatives
/// for both `Left x` and `Right x` with the same RHS.
/// NOTE: This only expands top-level or-patterns. Use `flatten_nested_or_patterns`
/// for deep expansion of nested or-patterns.
pub fn expand_or_patterns(pat: &hir::Pat) -> Vec<&hir::Pat> {
    match pat {
        Pat::Or(left, right, _) => {
            let mut result = expand_or_patterns(left);
            result.extend(expand_or_patterns(right));
            result
        }
        _ => vec![pat],
    }
}

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
    fn test_expand_or_patterns() {
        let span = Span::default();
        let def_id = DefId::new(100);
        let x = Symbol::intern("x");

        // Simple pattern - no expansion
        let simple_pat = Pat::Var(x, def_id, span);
        let expanded = expand_or_patterns(&simple_pat);
        assert_eq!(expanded.len(), 1);

        // Or-pattern - should expand to 2
        let left = Box::new(Pat::Lit(hir::Lit::Int(1), span));
        let right = Box::new(Pat::Lit(hir::Lit::Int(2), span));
        let or_pat = Pat::Or(left, right, span);
        let expanded = expand_or_patterns(&or_pat);
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
