//! Main AST to HIR lowering pass.
//!
//! This module contains the core lowering logic that transforms surface AST
//! into HIR by:
//!
//! 1. Resolving names (identifiers -> DefIds)
//! 2. Desugaring syntactic constructs
//! 3. Building HIR nodes

use bhc_ast as ast;
use bhc_hir as hir;
use bhc_intern::Symbol;
use bhc_span::Span;

use crate::context::{DefKind, LowerContext};
use crate::desugar;
use crate::resolve::{bind_pattern, collect_module_definitions, resolve_constructor, resolve_var};
use crate::{LowerError, LowerResult};

/// Configuration for the lowering pass.
#[derive(Clone, Debug, Default)]
pub struct LowerConfig {
    /// Whether to include builtins in the context.
    pub include_builtins: bool,
    /// Whether to report warnings for unused bindings.
    pub warn_unused: bool,
}

/// Lower an AST module to HIR.
pub fn lower_module(ctx: &mut LowerContext, module: &ast::Module) -> LowerResult<hir::Module> {
    // First pass: collect all top-level definitions
    collect_module_definitions(ctx, module);

    // Second pass: lower all declarations
    let mut items = Vec::new();
    for decl in &module.decls {
        if let Some(item) = lower_decl(ctx, decl)? {
            items.push(item);
        }
    }

    // Lower imports
    let imports = module
        .imports
        .iter()
        .map(|imp| lower_import(imp))
        .collect();

    // Lower exports
    let exports = module.exports.as_ref().map(|exps| {
        exps.iter().map(|exp| lower_export(exp)).collect()
    });

    // Check for errors
    if ctx.has_errors() {
        let errors = ctx.take_errors();
        return Err(LowerError::Multiple(errors));
    }

    Ok(hir::Module {
        name: module.name.as_ref().map_or_else(
            || Symbol::intern("Main"),
            |n| {
                // Combine module name parts into a single symbol
                let full_name = n.parts.iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                Symbol::intern(&full_name)
            },
        ),
        exports,
        imports,
        items,
        span: module.span,
    })
}

/// Lower a top-level declaration.
fn lower_decl(ctx: &mut LowerContext, decl: &ast::Decl) -> LowerResult<Option<hir::Item>> {
    match decl {
        ast::Decl::FunBind(fun_bind) => {
            let item = lower_fun_bind(ctx, fun_bind)?;
            Ok(Some(hir::Item::Value(item)))
        }

        ast::Decl::DataDecl(data_decl) => {
            let item = lower_data_decl(ctx, data_decl)?;
            Ok(Some(hir::Item::Data(item)))
        }

        ast::Decl::Newtype(newtype_decl) => {
            let item = lower_newtype_decl(ctx, newtype_decl)?;
            Ok(Some(hir::Item::Newtype(item)))
        }

        ast::Decl::TypeAlias(type_alias) => {
            let item = lower_type_alias(ctx, type_alias)?;
            Ok(Some(hir::Item::TypeAlias(item)))
        }

        ast::Decl::ClassDecl(class_decl) => {
            let item = lower_class_decl(ctx, class_decl)?;
            Ok(Some(hir::Item::Class(item)))
        }

        ast::Decl::InstanceDecl(instance_decl) => {
            let item = lower_instance_decl(ctx, instance_decl)?;
            Ok(Some(hir::Item::Instance(item)))
        }

        ast::Decl::Fixity(fixity_decl) => {
            let item = lower_fixity_decl(fixity_decl);
            Ok(Some(hir::Item::Fixity(item)))
        }

        ast::Decl::Foreign(foreign) => {
            let item = lower_foreign_decl(ctx, foreign)?;
            Ok(Some(hir::Item::Foreign(item)))
        }

        // Type signatures are associated with their definitions
        ast::Decl::TypeSig(_) => Ok(None),
    }
}

/// Lower a function binding.
fn lower_fun_bind(ctx: &mut LowerContext, fun_bind: &ast::FunBind) -> LowerResult<hir::ValueDef> {
    let name = fun_bind.name.name;
    let def_id = ctx
        .lookup_value(name)
        .expect("function should be pre-bound");

    let mut equations = Vec::new();
    for clause in &fun_bind.clauses {
        let eq = lower_clause(ctx, clause)?;
        equations.push(eq);
    }

    Ok(hir::ValueDef {
        id: def_id,
        name,
        sig: None, // TODO: look up type signature
        equations,
        span: fun_bind.span,
    })
}

/// Lower a function clause.
fn lower_clause(ctx: &mut LowerContext, clause: &ast::Clause) -> LowerResult<hir::Equation> {
    ctx.in_scope(|ctx| {
        // Bind pattern variables
        let mut pats = Vec::new();
        for ast_pat in &clause.pats {
            bind_pattern(ctx, ast_pat);
            let pat = lower_pat(ctx, ast_pat);
            pats.push(pat);
        }

        // Lower where bindings first (they're in scope for RHS)
        if !clause.wheres.is_empty() {
            // Enter a scope for where bindings
            ctx.enter_scope();
            for where_decl in &clause.wheres {
                if let ast::Decl::FunBind(fb) = where_decl {
                    let def_id = ctx.fresh_def_id();
                    ctx.define(def_id, fb.name.name, DefKind::Value, fb.span);
                    ctx.bind_value(fb.name.name, def_id);
                }
            }
        }

        // Lower RHS
        let (rhs, guards) = match &clause.rhs {
            ast::Rhs::Simple(expr, _) => (lower_expr(ctx, expr), Vec::new()),
            ast::Rhs::Guarded(guarded_rhss, _) => {
                // For guarded RHS, we desugar to nested if expressions
                let rhs = desugar::desugar_guarded_rhs(ctx, guarded_rhss, clause.span, &|ctx, e| {
                    lower_expr(ctx, e)
                });

                // We don't need guards in HIR for this; they're already desugared
                (rhs, Vec::new())
            }
        };

        // Wrap in let if there are where bindings
        let final_rhs = if !clause.wheres.is_empty() {
            let bindings: Vec<hir::Binding> = clause
                .wheres
                .iter()
                .filter_map(|d| {
                    if let ast::Decl::FunBind(fb) = d {
                        // For simple bindings
                        if fb.clauses.len() == 1 && fb.clauses[0].pats.is_empty() {
                            let rhs_expr = lower_rhs(ctx, &fb.clauses[0].rhs);
                            // Look up the DefId that was bound for this where binding
                            let def_id = ctx.lookup_value(fb.name.name)
                                .expect("where binding should be bound");
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs: rhs_expr,
                                span: fb.span,
                            });
                        }
                    }
                    None
                })
                .collect();

            ctx.exit_scope();

            if bindings.is_empty() {
                rhs
            } else {
                hir::Expr::Let(bindings, Box::new(rhs), clause.span)
            }
        } else {
            rhs
        };

        Ok(hir::Equation {
            pats,
            guards,
            rhs: final_rhs,
            span: clause.span,
        })
    })
}

/// Lower a right-hand side.
fn lower_rhs(ctx: &mut LowerContext, rhs: &ast::Rhs) -> hir::Expr {
    match rhs {
        ast::Rhs::Simple(expr, _) => lower_expr(ctx, expr),
        ast::Rhs::Guarded(guards, span) => {
            desugar::desugar_guarded_rhs(ctx, guards, *span, &|ctx, e| lower_expr(ctx, e))
        }
    }
}

/// Lower an expression.
fn lower_expr(ctx: &mut LowerContext, expr: &ast::Expr) -> hir::Expr {
    match expr {
        ast::Expr::Var(ident, span) => {
            let name = ident.name;
            if let Some(def_id) = resolve_var(ctx, name, *span) {
                hir::Expr::Var(ctx.def_ref(def_id, *span))
            } else {
                // Create placeholder for error recovery
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Value, *span);
                hir::Expr::Var(ctx.def_ref(def_id, *span))
            }
        }

        ast::Expr::QualVar(module_name, ident, span) => {
            // Qualified variable like M.foo
            let qual_name = format!("{}.{}", module_name.to_string(), ident.name.as_str());
            let name = Symbol::intern(&qual_name);
            if let Some(def_id) = resolve_var(ctx, name, *span) {
                hir::Expr::Var(ctx.def_ref(def_id, *span))
            } else {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Value, *span);
                hir::Expr::Var(ctx.def_ref(def_id, *span))
            }
        }

        ast::Expr::Con(ident, span) => {
            let name = ident.name;
            if let Some(def_id) = resolve_constructor(ctx, name, *span) {
                hir::Expr::Con(ctx.def_ref(def_id, *span))
            } else {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Constructor, *span);
                hir::Expr::Con(ctx.def_ref(def_id, *span))
            }
        }

        ast::Expr::QualCon(module_name, ident, span) => {
            // Qualified constructor like M.Just
            let qual_name = format!("{}.{}", module_name.to_string(), ident.name.as_str());
            let name = Symbol::intern(&qual_name);
            if let Some(def_id) = resolve_constructor(ctx, name, *span) {
                hir::Expr::Con(ctx.def_ref(def_id, *span))
            } else {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Constructor, *span);
                hir::Expr::Con(ctx.def_ref(def_id, *span))
            }
        }

        ast::Expr::Lit(lit, span) => {
            let hir_lit = lower_lit(lit);
            hir::Expr::Lit(hir_lit, *span)
        }

        ast::Expr::App(fun, arg, span) => {
            let f = lower_expr(ctx, fun);
            let a = lower_expr(ctx, arg);
            hir::Expr::App(Box::new(f), Box::new(a), *span)
        }

        ast::Expr::Infix(lhs, op, rhs, span) => {
            // Desugar infix to prefix: a `op` b -> op a b
            let op_expr = if op.name.as_str().chars().next().map_or(false, |c| c.is_uppercase()) {
                // Constructor
                if let Some(def_id) = resolve_constructor(ctx, op.name, *span) {
                    hir::Expr::Con(ctx.def_ref(def_id, *span))
                } else {
                    hir::Expr::Error(*span)
                }
            } else {
                // Variable/operator
                if let Some(def_id) = resolve_var(ctx, op.name, *span) {
                    hir::Expr::Var(ctx.def_ref(def_id, *span))
                } else {
                    // Create placeholder
                    let def_id = ctx.fresh_def_id();
                    ctx.define(def_id, op.name, DefKind::Value, *span);
                    hir::Expr::Var(ctx.def_ref(def_id, *span))
                }
            };
            let l = lower_expr(ctx, lhs);
            let r = lower_expr(ctx, rhs);
            let app1 = hir::Expr::App(Box::new(op_expr), Box::new(l), *span);
            hir::Expr::App(Box::new(app1), Box::new(r), *span)
        }

        ast::Expr::Neg(inner, span) => {
            // Desugar negation: -e -> negate e
            let negate_sym = Symbol::intern("negate");
            if let Some(def_id) = ctx.lookup_value(negate_sym) {
                let negate = hir::Expr::Var(ctx.def_ref(def_id, *span));
                let e = lower_expr(ctx, inner);
                hir::Expr::App(Box::new(negate), Box::new(e), *span)
            } else {
                hir::Expr::Error(*span)
            }
        }

        ast::Expr::Lam(pats, body, span) => ctx.in_scope(|ctx| {
            let mut hir_pats = Vec::new();
            for p in pats {
                bind_pattern(ctx, p);
                hir_pats.push(lower_pat(ctx, p));
            }
            let e = lower_expr(ctx, body);
            hir::Expr::Lam(hir_pats, Box::new(e), *span)
        }),

        ast::Expr::Let(decls, body, span) => ctx.in_scope(|ctx| {
            // Bind all declarations first
            for decl in decls {
                if let ast::Decl::FunBind(fb) = decl {
                    let def_id = ctx.fresh_def_id();
                    ctx.define(def_id, fb.name.name, DefKind::Value, fb.span);
                    ctx.bind_value(fb.name.name, def_id);
                }
            }

            // Lower bindings
            let bindings: Vec<hir::Binding> = decls
                .iter()
                .filter_map(|d| {
                    if let ast::Decl::FunBind(fb) = d {
                        if fb.clauses.len() == 1 && fb.clauses[0].pats.is_empty() {
                            let rhs_expr = lower_rhs(ctx, &fb.clauses[0].rhs);
                            // Look up the DefId that was bound above
                            let def_id = ctx.lookup_value(fb.name.name)
                                .expect("let binding should be bound");
                            return Some(hir::Binding {
                                pat: hir::Pat::Var(fb.name.name, def_id, fb.span),
                                sig: None,
                                rhs: rhs_expr,
                                span: fb.span,
                            });
                        }
                    }
                    None
                })
                .collect();

            let e = lower_expr(ctx, body);
            hir::Expr::Let(bindings, Box::new(e), *span)
        }),

        ast::Expr::If(cond, then_branch, else_branch, span) => {
            let c = lower_expr(ctx, cond);
            let t = lower_expr(ctx, then_branch);
            let e = lower_expr(ctx, else_branch);
            hir::Expr::If(Box::new(c), Box::new(t), Box::new(e), *span)
        }

        ast::Expr::Case(scrutinee, alts, span) => {
            let s = lower_expr(ctx, scrutinee);
            let hir_alts: Vec<hir::CaseAlt> = alts
                .iter()
                .map(|alt| lower_alt(ctx, alt))
                .collect();
            hir::Expr::Case(Box::new(s), hir_alts, *span)
        }

        ast::Expr::Do(stmts, span) => {
            desugar::desugar_do(
                ctx,
                stmts,
                *span,
                |ctx, e| lower_expr(ctx, e),
                |ctx, p| lower_pat(ctx, p),
            )
        }

        ast::Expr::ListComp(expr, stmts, span) => {
            desugar::desugar_list_comp(
                ctx,
                expr,
                stmts,
                *span,
                |ctx, e| lower_expr(ctx, e),
                |ctx, p| lower_pat(ctx, p),
            )
        }

        ast::Expr::Tuple(exprs, span) => {
            let es: Vec<hir::Expr> = exprs.iter().map(|e| lower_expr(ctx, e)).collect();
            hir::Expr::Tuple(es, *span)
        }

        ast::Expr::List(exprs, span) => {
            let es: Vec<hir::Expr> = exprs.iter().map(|e| lower_expr(ctx, e)).collect();
            hir::Expr::List(es, *span)
        }

        ast::Expr::ArithSeq(seq, span) => {
            // Desugar arithmetic sequences
            lower_arith_seq(ctx, seq, *span)
        }

        ast::Expr::RecordCon(con, fields, span) => {
            let con_name = con.name;
            if let Some(def_id) = resolve_constructor(ctx, con_name, *span) {
                let con_ref = ctx.def_ref(def_id, *span);
                let mut hir_fields = Vec::with_capacity(fields.len());
                for f in fields {
                    let value = match &f.value {
                        Some(e) => lower_expr(ctx, e),
                        None => {
                            // Punning: Foo { x } means Foo { x = x }
                            let name = f.name.name;
                            if let Some(def_id) = ctx.lookup_value(name) {
                                hir::Expr::Var(ctx.def_ref(def_id, f.span))
                            } else {
                                hir::Expr::Error(f.span)
                            }
                        }
                    };
                    hir_fields.push(hir::FieldExpr {
                        name: f.name.name,
                        value,
                        span: f.span,
                    });
                }
                hir::Expr::Record(con_ref, hir_fields, *span)
            } else {
                hir::Expr::Error(*span)
            }
        }

        ast::Expr::RecordUpd(base, fields, span) => {
            let b = lower_expr(ctx, base);
            let mut hir_fields = Vec::with_capacity(fields.len());
            for f in fields {
                let value = match &f.value {
                    Some(e) => lower_expr(ctx, e),
                    None => hir::Expr::Error(f.span),
                };
                hir_fields.push(hir::FieldExpr {
                    name: f.name.name,
                    value,
                    span: f.span,
                });
            }
            hir::Expr::RecordUpdate(Box::new(b), hir_fields, *span)
        }

        ast::Expr::Ann(expr, ty, span) => {
            let e = lower_expr(ctx, expr);
            let t = lower_type(ctx, ty);
            hir::Expr::Ann(Box::new(e), t, *span)
        }

        ast::Expr::Paren(inner, _) => lower_expr(ctx, inner),

        ast::Expr::Lazy(inner, _span) => {
            // For now, just lower the inner expression
            // TODO: Handle lazy block semantics
            lower_expr(ctx, inner)
        }
    }
}

/// Lower a case alternative.
fn lower_alt(ctx: &mut LowerContext, alt: &ast::Alt) -> hir::CaseAlt {
    ctx.in_scope(|ctx| {
        bind_pattern(ctx, &alt.pat);
        let pat = lower_pat(ctx, &alt.pat);

        let rhs = lower_rhs(ctx, &alt.rhs);

        hir::CaseAlt {
            pat,
            guards: vec![],
            rhs,
            span: alt.span,
        }
    })
}

/// Lower a pattern.
fn lower_pat(ctx: &mut LowerContext, pat: &ast::Pat) -> hir::Pat {
    match pat {
        ast::Pat::Var(ident, span) => {
            // Look up the DefId that was bound by bind_pattern
            let def_id = ctx.lookup_value(ident.name).expect("pattern variable should be bound");
            hir::Pat::Var(ident.name, def_id, *span)
        }

        ast::Pat::Wildcard(span) => hir::Pat::Wild(*span),

        ast::Pat::Lit(lit, span) => {
            let hir_lit = lower_lit(lit);
            hir::Pat::Lit(hir_lit, *span)
        }

        ast::Pat::Con(ident, pats, span) => {
            let con_name = ident.name;
            if let Some(def_id) = resolve_constructor(ctx, con_name, *span) {
                let con_ref = ctx.def_ref(def_id, *span);
                let hir_pats: Vec<hir::Pat> = pats.iter().map(|p| lower_pat(ctx, p)).collect();
                hir::Pat::Con(con_ref, hir_pats, *span)
            } else {
                hir::Pat::Error(*span)
            }
        }

        ast::Pat::Infix(lhs, op, rhs, span) => {
            // Desugar infix pattern: x : xs -> (:) x xs
            let con_name = op.name;
            if let Some(def_id) = resolve_constructor(ctx, con_name, *span) {
                let con_ref = ctx.def_ref(def_id, *span);
                let l = lower_pat(ctx, lhs);
                let r = lower_pat(ctx, rhs);
                hir::Pat::Con(con_ref, vec![l, r], *span)
            } else {
                hir::Pat::Error(*span)
            }
        }

        ast::Pat::Tuple(pats, span) => {
            // Tuple pattern is sugar for tuple constructor
            let tuple_sym = Symbol::intern(&format!("({})", ",".repeat(pats.len().saturating_sub(1))));
            let def_id = ctx.fresh_def_id();
            ctx.define(def_id, tuple_sym, DefKind::Constructor, *span);
            let tuple_ref = ctx.def_ref(def_id, *span);

            let hir_pats: Vec<hir::Pat> = pats.iter().map(|p| lower_pat(ctx, p)).collect();
            hir::Pat::Con(tuple_ref, hir_pats, *span)
        }

        ast::Pat::List(pats, span) => {
            // Desugar list pattern to cons chain
            desugar_list_pat(ctx, pats, *span)
        }

        ast::Pat::As(ident, inner, span) => {
            // Look up the DefId that was bound by bind_pattern
            let def_id = ctx.lookup_value(ident.name).expect("as-pattern should be bound");
            let p = lower_pat(ctx, inner);
            hir::Pat::As(ident.name, def_id, Box::new(p), *span)
        }

        ast::Pat::Lazy(inner, _span) => {
            // For now, just lower the inner pattern
            lower_pat(ctx, inner)
        }

        ast::Pat::Bang(inner, _span) => {
            // For now, just lower the inner pattern
            lower_pat(ctx, inner)
        }

        ast::Pat::Record(con, fields, span) => {
            let con_name = con.name;
            if let Some(def_id) = resolve_constructor(ctx, con_name, *span) {
                let con_ref = ctx.def_ref(def_id, *span);
                let mut hir_pats: Vec<hir::Pat> = Vec::with_capacity(fields.len());
                for f in fields {
                    let pat = match &f.pat {
                        Some(p) => lower_pat(ctx, p),
                        None => {
                            // Punned field: Foo { x } binds x
                            let field_def_id = ctx.lookup_value(f.name.name)
                                .expect("punned field should be bound");
                            hir::Pat::Var(f.name.name, field_def_id, f.span)
                        }
                    };
                    hir_pats.push(pat);
                }
                hir::Pat::Con(con_ref, hir_pats, *span)
            } else {
                hir::Pat::Error(*span)
            }
        }

        ast::Pat::Paren(inner, _) => lower_pat(ctx, inner),

        ast::Pat::Ann(inner, ty, span) => {
            let p = lower_pat(ctx, inner);
            let t = lower_type(ctx, ty);
            hir::Pat::Ann(Box::new(p), t, *span)
        }
    }
}

/// Desugar a list pattern [p1, p2, ...] to p1 : p2 : ... : []
fn desugar_list_pat(ctx: &mut LowerContext, pats: &[ast::Pat], span: Span) -> hir::Pat {
    let nil_sym = Symbol::intern("[]");
    let cons_sym = Symbol::intern(":");

    // Start with nil pattern
    let nil_def = ctx.lookup_constructor(nil_sym).unwrap_or_else(|| {
        let id = ctx.fresh_def_id();
        ctx.define(id, nil_sym, DefKind::Constructor, span);
        id
    });
    let nil_ref = ctx.def_ref(nil_def, span);

    pats.iter().rev().fold(
        hir::Pat::Con(nil_ref.clone(), vec![], span),
        |acc, p| {
            let cons_def = ctx.lookup_constructor(cons_sym).unwrap_or_else(|| {
                let id = ctx.fresh_def_id();
                ctx.define(id, cons_sym, DefKind::Constructor, span);
                id
            });
            let cons_ref = ctx.def_ref(cons_def, span);
            let hir_p = lower_pat(ctx, p);
            hir::Pat::Con(cons_ref, vec![hir_p, acc], span)
        },
    )
}

/// Lower a literal.
fn lower_lit(lit: &ast::Lit) -> hir::Lit {
    match lit {
        ast::Lit::Int(n) => hir::Lit::Int(*n as i128),
        ast::Lit::Float(f) => hir::Lit::Float(*f),
        ast::Lit::Char(c) => hir::Lit::Char(*c),
        ast::Lit::String(s) => hir::Lit::String(Symbol::intern(s)),
    }
}

/// Lower a type.
fn lower_type(ctx: &mut LowerContext, ty: &ast::Type) -> bhc_types::Ty {
    match ty {
        ast::Type::Var(tyvar, _) => {
            bhc_types::Ty::Var(bhc_types::TyVar::new_star(tyvar.name.name.as_u32()))
        }

        ast::Type::Con(ident, _) => {
            bhc_types::Ty::Con(bhc_types::TyCon::new(ident.name, bhc_types::Kind::Star))
        }

        ast::Type::App(f, a, _) => {
            let fun_ty = lower_type(ctx, f);
            let arg_ty = lower_type(ctx, a);
            bhc_types::Ty::App(Box::new(fun_ty), Box::new(arg_ty))
        }

        ast::Type::Fun(from, to, _) => {
            let from_ty = lower_type(ctx, from);
            let to_ty = lower_type(ctx, to);
            bhc_types::Ty::Fun(Box::new(from_ty), Box::new(to_ty))
        }

        ast::Type::Tuple(tys, _) => {
            let hir_tys: Vec<bhc_types::Ty> = tys.iter().map(|t| lower_type(ctx, t)).collect();
            bhc_types::Ty::Tuple(hir_tys)
        }

        ast::Type::List(elem, _) => {
            let elem_ty = lower_type(ctx, elem);
            bhc_types::Ty::List(Box::new(elem_ty))
        }

        ast::Type::Paren(inner, _) => lower_type(ctx, inner),

        ast::Type::Forall(vars, inner, _) => {
            let ty_vars: Vec<bhc_types::TyVar> = vars
                .iter()
                .map(|v| bhc_types::TyVar::new_star(v.name.name.as_u32()))
                .collect();
            let inner_ty = lower_type(ctx, inner);
            bhc_types::Ty::Forall(ty_vars, Box::new(inner_ty))
        }

        ast::Type::Constrained(_, inner, _) => {
            // TODO: handle constraints properly
            lower_type(ctx, inner)
        }

        ast::Type::QualCon(module_name, ident, _) => {
            // Qualified type constructor like M.Map
            // Create a qualified name symbol by combining module and name
            let qual_name = format!("{}.{}", module_name.to_string(), ident.name.as_str());
            let symbol = Symbol::intern(&qual_name);
            bhc_types::Ty::Con(bhc_types::TyCon::new(symbol, bhc_types::Kind::Star))
        }

        ast::Type::NatLit(n, _) => bhc_types::Ty::nat_lit(*n),

        ast::Type::PromotedList(_, _) | ast::Type::Bang(_, _) | ast::Type::Lazy(_, _) => {
            // TODO: handle these types
            bhc_types::Ty::Error
        }
    }
}

/// Lower an arithmetic sequence.
fn lower_arith_seq(ctx: &mut LowerContext, seq: &ast::ArithSeq, span: Span) -> hir::Expr {
    // Desugar arithmetic sequences to enumFrom* calls
    let (func_name, args) = match seq {
        ast::ArithSeq::From(start) => ("enumFrom", vec![lower_expr(ctx, start)]),
        ast::ArithSeq::FromThen(start, next) => {
            ("enumFromThen", vec![lower_expr(ctx, start), lower_expr(ctx, next)])
        }
        ast::ArithSeq::FromTo(start, end) => {
            ("enumFromTo", vec![lower_expr(ctx, start), lower_expr(ctx, end)])
        }
        ast::ArithSeq::FromThenTo(start, next, end) => (
            "enumFromThenTo",
            vec![
                lower_expr(ctx, start),
                lower_expr(ctx, next),
                lower_expr(ctx, end),
            ],
        ),
    };

    let func_sym = Symbol::intern(func_name);
    let func = if let Some(def_id) = ctx.lookup_value(func_sym) {
        hir::Expr::Var(ctx.def_ref(def_id, span))
    } else {
        let def_id = ctx.fresh_def_id();
        ctx.define(def_id, func_sym, DefKind::Value, span);
        hir::Expr::Var(ctx.def_ref(def_id, span))
    };

    args.into_iter()
        .fold(func, |f, a| hir::Expr::App(Box::new(f), Box::new(a), span))
}

/// Lower an import declaration.
fn lower_import(imp: &ast::ImportDecl) -> hir::Import {
    let module_name = imp.module.parts.iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(".");

    hir::Import {
        module: Symbol::intern(&module_name),
        qualified: imp.qualified,
        alias: imp.alias.as_ref().map(|a| {
            let alias_name = a.parts.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(".");
            Symbol::intern(&alias_name)
        }),
        items: imp.spec.as_ref().map(|spec| {
            match spec {
                ast::ImportSpec::Only(items) | ast::ImportSpec::Hiding(items) => {
                    items.iter().map(|item| match item {
                        ast::Import::Var(ident, span) => hir::ImportItem {
                            name: ident.name,
                            children: hir::ExportChildren::None,
                            span: *span,
                        },
                        ast::Import::Type(ident, children, span) => hir::ImportItem {
                            name: ident.name,
                            children: children.as_ref().map_or(
                                hir::ExportChildren::None,
                                |cs| hir::ExportChildren::Some(cs.iter().map(|c| c.name).collect()),
                            ),
                            span: *span,
                        },
                    }).collect()
                }
            }
        }),
        hiding: matches!(imp.spec, Some(ast::ImportSpec::Hiding(_))),
        span: imp.span,
    }
}

/// Lower an export specification.
fn lower_export(exp: &ast::Export) -> hir::Export {
    match exp {
        ast::Export::Var(ident, span) => hir::Export {
            name: ident.name,
            children: hir::ExportChildren::None,
            span: *span,
        },
        ast::Export::Type(ident, children, span) => hir::Export {
            name: ident.name,
            children: children.as_ref().map_or(
                hir::ExportChildren::None,
                |cs| hir::ExportChildren::Some(cs.iter().map(|c| c.name).collect()),
            ),
            span: *span,
        },
        ast::Export::Module(module_name, span) => {
            let name = module_name.parts.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(".");
            hir::Export {
                name: Symbol::intern(&name),
                children: hir::ExportChildren::All,
                span: *span,
            }
        }
    }
}

/// Lower a data declaration.
fn lower_data_decl(ctx: &mut LowerContext, data: &ast::DataDecl) -> LowerResult<hir::DataDef> {
    let type_def_id = ctx.lookup_type(data.name.name).expect("type should be pre-bound");

    let params: Vec<bhc_types::TyVar> = data
        .params
        .iter()
        .map(|p| bhc_types::TyVar::new_star(p.name.name.as_u32()))
        .collect();

    let cons: Vec<hir::ConDef> = data
        .constrs
        .iter()
        .map(|c| lower_con_def(ctx, c))
        .collect();

    let deriving: Vec<Symbol> = data
        .deriving
        .iter()
        .map(|c| c.name)
        .collect();

    Ok(hir::DataDef {
        id: type_def_id,
        name: data.name.name,
        params,
        cons,
        deriving,
        span: data.span,
    })
}

/// Lower a constructor definition.
fn lower_con_def(ctx: &mut LowerContext, con: &ast::ConDecl) -> hir::ConDef {
    let con_def_id = ctx
        .lookup_constructor(con.name.name)
        .expect("constructor should be pre-bound");

    let fields = match &con.fields {
        ast::ConFields::Positional(tys) => {
            let hir_tys: Vec<bhc_types::Ty> = tys.iter().map(|t| lower_type(ctx, t)).collect();
            hir::ConFields::Positional(hir_tys)
        }
        ast::ConFields::Record(fields) => {
            let hir_fields: Vec<hir::FieldDef> = fields
                .iter()
                .map(|f| hir::FieldDef {
                    name: f.name.name,
                    ty: lower_type(ctx, &f.ty),
                    span: f.span,
                })
                .collect();
            hir::ConFields::Named(hir_fields)
        }
    };

    hir::ConDef {
        id: con_def_id,
        name: con.name.name,
        fields,
        span: con.span,
    }
}

/// Lower a newtype declaration.
fn lower_newtype_decl(
    ctx: &mut LowerContext,
    newtype: &ast::NewtypeDecl,
) -> LowerResult<hir::NewtypeDef> {
    let type_def_id = ctx
        .lookup_type(newtype.name.name)
        .expect("type should be pre-bound");

    let params: Vec<bhc_types::TyVar> = newtype
        .params
        .iter()
        .map(|p| bhc_types::TyVar::new_star(p.name.name.as_u32()))
        .collect();

    let con = lower_con_def(ctx, &newtype.constr);

    let deriving: Vec<Symbol> = newtype
        .deriving
        .iter()
        .map(|c| c.name)
        .collect();

    Ok(hir::NewtypeDef {
        id: type_def_id,
        name: newtype.name.name,
        params,
        con,
        deriving,
        span: newtype.span,
    })
}

/// Lower a type alias declaration.
fn lower_type_alias(ctx: &mut LowerContext, type_alias: &ast::TypeAlias) -> LowerResult<hir::TypeAlias> {
    let def_id = ctx
        .lookup_type(type_alias.name.name)
        .expect("type should be pre-bound");

    let params: Vec<bhc_types::TyVar> = type_alias
        .params
        .iter()
        .map(|p| bhc_types::TyVar::new_star(p.name.name.as_u32()))
        .collect();

    let ty = lower_type(ctx, &type_alias.ty);

    Ok(hir::TypeAlias {
        id: def_id,
        name: type_alias.name.name,
        params,
        ty,
        span: type_alias.span,
    })
}

/// Lower a class declaration.
fn lower_class_decl(ctx: &mut LowerContext, class: &ast::ClassDecl) -> LowerResult<hir::ClassDef> {
    let def_id = ctx
        .lookup_type(class.name.name)
        .expect("class should be pre-bound");

    let params: Vec<bhc_types::TyVar> = vec![
        bhc_types::TyVar::new_star(class.param.name.name.as_u32())
    ];

    let supers: Vec<Symbol> = class
        .context
        .iter()
        .map(|c| c.class.name)
        .collect();

    // Extract method signatures
    let methods: Vec<hir::MethodSig> = class
        .methods
        .iter()
        .filter_map(|m| {
            if let ast::Decl::TypeSig(sig) = m {
                Some(
                    sig.names
                        .iter()
                        .map(|n| hir::MethodSig {
                            name: n.name,
                            ty: bhc_types::Scheme::mono(lower_type(ctx, &sig.ty)),
                            span: sig.span,
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            }
        })
        .flatten()
        .collect();

    // Extract default implementations
    let defaults: Vec<hir::ValueDef> = class
        .methods
        .iter()
        .filter_map(|m| {
            if let ast::Decl::FunBind(fb) = m {
                lower_fun_bind(ctx, fb).ok()
            } else {
                None
            }
        })
        .collect();

    Ok(hir::ClassDef {
        id: def_id,
        name: class.name.name,
        params,
        supers,
        methods,
        defaults,
        span: class.span,
    })
}

/// Lower an instance declaration.
fn lower_instance_decl(
    ctx: &mut LowerContext,
    instance: &ast::InstanceDecl,
) -> LowerResult<hir::InstanceDef> {
    let types: Vec<bhc_types::Ty> = vec![lower_type(ctx, &instance.ty)];

    let constraints: Vec<Symbol> = instance
        .context
        .iter()
        .map(|c| c.class.name)
        .collect();

    let methods: Vec<hir::ValueDef> = instance
        .methods
        .iter()
        .filter_map(|m| {
            if let ast::Decl::FunBind(fb) = m {
                lower_fun_bind(ctx, fb).ok()
            } else {
                None
            }
        })
        .collect();

    Ok(hir::InstanceDef {
        class: instance.class.name,
        types,
        constraints,
        methods,
        span: instance.span,
    })
}

/// Lower a fixity declaration.
fn lower_fixity_decl(fixity: &ast::FixityDecl) -> hir::FixityDecl {
    let hir_fixity = match fixity.fixity {
        ast::Fixity::Left => hir::Fixity::Left,
        ast::Fixity::Right => hir::Fixity::Right,
        ast::Fixity::None => hir::Fixity::None,
    };

    hir::FixityDecl {
        fixity: hir_fixity,
        precedence: fixity.prec,
        ops: fixity.ops.iter().map(|o| o.name).collect(),
        span: fixity.span,
    }
}

/// Lower a foreign declaration.
fn lower_foreign_decl(
    ctx: &mut LowerContext,
    foreign: &ast::ForeignDecl,
) -> LowerResult<hir::ForeignDecl> {
    let def_id = ctx
        .lookup_value(foreign.name.name)
        .expect("foreign import should be pre-bound");

    // Map convention string to ForeignConvention
    let convention = match foreign.convention.as_str() {
        "ccall" | "capi" => hir::ForeignConvention::CCall,
        "stdcall" => hir::ForeignConvention::StdCall,
        "javascript" => hir::ForeignConvention::JavaScript,
        _ => hir::ForeignConvention::CCall, // Default
    };

    let ty = bhc_types::Scheme::mono(lower_type(ctx, &foreign.ty));

    Ok(hir::ForeignDecl {
        id: def_id,
        name: foreign.name.name,
        foreign_name: foreign.external_name.as_ref().map_or_else(
            || foreign.name.name,
            |s| Symbol::intern(s),
        ),
        convention,
        ty,
        span: foreign.span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_simple_var() {
        let mut ctx = LowerContext::with_builtins();

        // Create a variable reference to a builtin
        let ident = bhc_intern::Ident::from_str("map");
        let expr = ast::Expr::Var(ident, Span::default());

        let result = lower_expr(&mut ctx, &expr);

        assert!(matches!(result, hir::Expr::Var(_)));
        assert!(!ctx.has_errors());
    }

    #[test]
    fn test_lower_literal() {
        let mut ctx = LowerContext::with_builtins();

        let expr = ast::Expr::Lit(
            ast::Lit::Int(42),
            Span::default(),
        );

        let result = lower_expr(&mut ctx, &expr);

        match result {
            hir::Expr::Lit(hir::Lit::Int(n), _) => assert_eq!(n, 42),
            _ => panic!("expected integer literal"),
        }
    }

    #[test]
    fn test_lower_application() {
        let mut ctx = LowerContext::with_builtins();

        // map id
        let map_ident = bhc_intern::Ident::from_str("map");
        let id_ident = bhc_intern::Ident::from_str("id");

        let expr = ast::Expr::App(
            Box::new(ast::Expr::Var(map_ident, Span::default())),
            Box::new(ast::Expr::Var(id_ident, Span::default())),
            Span::default(),
        );

        let result = lower_expr(&mut ctx, &expr);

        assert!(matches!(result, hir::Expr::App(_, _, _)));
    }
}
