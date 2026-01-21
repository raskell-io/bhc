//! Desugaring pass for syntactic sugar.
//!
//! This module handles the expansion of surface syntax constructs into
//! simpler HIR forms:
//!
//! - **Do-notation**: `do { x <- e1; e2 }` -> `e1 >>= \x -> e2`
//! - **List comprehensions**: `[e | x <- xs, p]` -> `concatMap (\x -> if p then [e] else []) xs`
//! - **If expressions**: Already have HIR representation
//! - **Guards**: Converted to case expressions with boolean matching

use bhc_ast as ast;
use bhc_hir as hir;
use bhc_intern::Symbol;
use bhc_span::Span;

use crate::context::LowerContext;
use crate::resolve::bind_pattern;

/// Desugar do-notation into monadic bind and sequence operations.
///
/// ```haskell
/// do { x <- e1; e2 }
/// -- becomes --
/// e1 >>= \x -> e2
///
/// do { e1; e2 }
/// -- becomes --
/// e1 >> e2
///
/// do { let x = e1; e2 }
/// -- becomes --
/// let x = e1 in e2
/// ```
pub fn desugar_do(
    ctx: &mut LowerContext,
    stmts: &[ast::Stmt],
    span: Span,
    lower_expr: impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    if stmts.is_empty() {
        // Empty do block - this is an error, but we handle it gracefully
        return hir::Expr::Error(span);
    }

    desugar_do_stmts(ctx, stmts, span, &lower_expr, &lower_pat)
}

fn desugar_do_stmts(
    ctx: &mut LowerContext,
    stmts: &[ast::Stmt],
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: &impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    match stmts {
        [] => hir::Expr::Error(span),

        // Final statement must be an expression (Qualifier)
        [ast::Stmt::Qualifier(e, _)] => lower_expr(ctx, e),

        // Generator: x <- e
        [ast::Stmt::Generator(pat, expr, stmt_span), rest @ ..] => {
            let e = lower_expr(ctx, expr);
            // Enter a new scope for the pattern variable
            ctx.enter_scope();
            // Bind pattern variables before lowering
            bind_pattern(ctx, pat);
            let p = lower_pat(ctx, pat);
            let body = desugar_do_stmts(ctx, rest, span, lower_expr, lower_pat);
            ctx.exit_scope();

            // e >>= \p -> body
            let bind_sym = Symbol::intern(">>=");
            let bind_ref = make_var_ref(ctx, bind_sym, *stmt_span);

            let lambda = hir::Expr::Lam(vec![p], Box::new(body), span);
            let bind_app = hir::Expr::App(Box::new(bind_ref), Box::new(e), *stmt_span);
            hir::Expr::App(Box::new(bind_app), Box::new(lambda), span)
        }

        // Qualifier (not the last one): e; ...
        [ast::Stmt::Qualifier(expr, stmt_span), rest @ ..] => {
            let e = lower_expr(ctx, expr);
            let body = desugar_do_stmts(ctx, rest, span, lower_expr, lower_pat);

            // e >> body
            let seq_sym = Symbol::intern(">>");
            let seq_ref = make_var_ref(ctx, seq_sym, *stmt_span);

            let seq_app = hir::Expr::App(Box::new(seq_ref), Box::new(e), *stmt_span);
            hir::Expr::App(Box::new(seq_app), Box::new(body), span)
        }

        // Let statement: let x = e
        [ast::Stmt::LetStmt(decls, stmt_span), rest @ ..] => {
            // Enter scope and bind let variables BEFORE processing the rest
            ctx.enter_scope();
            pre_bind_let_decls(ctx, decls);
            let body = desugar_do_stmts(ctx, rest, span, lower_expr, lower_pat);
            let result = desugar_let_decls(ctx, decls, body, *stmt_span, lower_expr, lower_pat);
            ctx.exit_scope();
            result
        }
    }
}

/// Pre-bind let declaration names into the current scope.
/// This must be called before desugaring the body that uses these bindings.
fn pre_bind_let_decls(ctx: &mut LowerContext, decls: &[ast::Decl]) {
    use crate::context::DefKind;

    for decl in decls {
        if let ast::Decl::FunBind(fun_bind) = decl {
            // Only handle simple bindings (single clause, no patterns)
            if fun_bind.clauses.len() == 1 && fun_bind.clauses[0].pats.is_empty() {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, fun_bind.name.name, DefKind::Value, fun_bind.span);
                ctx.bind_value(fun_bind.name.name, def_id);
            }
        }
    }
}

/// Desugar let declarations into HIR let bindings.
fn desugar_let_decls(
    ctx: &mut LowerContext,
    decls: &[ast::Decl],
    body: hir::Expr,
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: &impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    use crate::context::DefKind;

    // First pass: bind all names (if not already bound by pre_bind_let_decls)
    for decl in decls {
        if let ast::Decl::FunBind(fun_bind) = decl {
            if fun_bind.clauses.len() == 1 && fun_bind.clauses[0].pats.is_empty() {
                // Only bind if not already bound (e.g., from do-notation pre-binding)
                if ctx.lookup_value(fun_bind.name.name).is_none() {
                    let def_id = ctx.fresh_def_id();
                    ctx.define(def_id, fun_bind.name.name, DefKind::Value, fun_bind.span);
                    ctx.bind_value(fun_bind.name.name, def_id);
                }
            }
        }
    }

    // Second pass: create bindings
    let mut bindings = Vec::new();

    for decl in decls {
        if let ast::Decl::FunBind(fun_bind) = decl {
            // Simple function binding becomes a pattern binding
            if fun_bind.clauses.len() == 1 && fun_bind.clauses[0].pats.is_empty() {
                let clause = &fun_bind.clauses[0];
                let def_id = ctx.lookup_value(fun_bind.name.name)
                    .expect("do-let binding should be bound");
                let pat = hir::Pat::Var(fun_bind.name.name, def_id, fun_bind.span);
                let rhs = match &clause.rhs {
                    ast::Rhs::Simple(e, _) => lower_expr(ctx, e),
                    ast::Rhs::Guarded(guards, _) => {
                        desugar_guarded_rhs(ctx, guards, span, lower_expr, lower_pat)
                    }
                };

                bindings.push(hir::Binding {
                    pat,
                    sig: None,
                    rhs,
                    span: fun_bind.span,
                });
            }
        }
    }

    if bindings.is_empty() {
        body
    } else {
        hir::Expr::Let(bindings, Box::new(body), span)
    }
}

/// Desugar list comprehensions.
///
/// ```haskell
/// [e | x <- xs, p, y <- ys]
/// -- becomes --
/// concatMap (\x -> if p then concatMap (\y -> [e]) ys else []) xs
/// ```
pub fn desugar_list_comp(
    ctx: &mut LowerContext,
    expr: &ast::Expr,
    stmts: &[ast::Stmt],
    span: Span,
    lower_expr: impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    desugar_stmts_for_comp(ctx, expr, stmts, span, &lower_expr, &lower_pat)
}

fn desugar_stmts_for_comp(
    ctx: &mut LowerContext,
    expr: &ast::Expr,
    stmts: &[ast::Stmt],
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: &impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    match stmts {
        [] => {
            // [e] - singleton list
            let e = lower_expr(ctx, expr);
            hir::Expr::List(vec![e], span)
        }

        [ast::Stmt::Generator(pat, gen_expr, qual_span), rest @ ..] => {
            // x <- xs becomes concatMap (\x -> ...) xs
            let xs = lower_expr(ctx, gen_expr);
            // Enter a new scope for the pattern variable
            ctx.enter_scope();
            // Bind pattern variables before lowering
            bind_pattern(ctx, pat);
            let p = lower_pat(ctx, pat);
            let body = desugar_stmts_for_comp(ctx, expr, rest, span, lower_expr, lower_pat);
            ctx.exit_scope();

            let lambda = hir::Expr::Lam(vec![p], Box::new(body), span);

            let concat_map_sym = Symbol::intern("concatMap");
            let concat_map = make_var_ref(ctx, concat_map_sym, *qual_span);

            let app1 = hir::Expr::App(Box::new(concat_map), Box::new(lambda), *qual_span);
            hir::Expr::App(Box::new(app1), Box::new(xs), span)
        }

        [ast::Stmt::Qualifier(guard_expr, qual_span), rest @ ..] => {
            // p becomes if p then ... else []
            let cond = lower_expr(ctx, guard_expr);
            let then_branch = desugar_stmts_for_comp(ctx, expr, rest, span, lower_expr, lower_pat);
            let else_branch = hir::Expr::List(vec![], *qual_span);

            hir::Expr::If(
                Box::new(cond),
                Box::new(then_branch),
                Box::new(else_branch),
                span,
            )
        }

        [ast::Stmt::LetStmt(decls, qual_span), rest @ ..] => {
            // let x = e becomes let x = e in ...
            let body = desugar_stmts_for_comp(ctx, expr, rest, span, lower_expr, lower_pat);
            desugar_let_decls(ctx, decls, body, *qual_span, lower_expr, lower_pat)
        }
    }
}

/// Desugar guarded right-hand sides to nested if/case expressions.
///
/// For boolean guards:
/// ```haskell
/// | g1 = e1
/// | g2 = e2
/// | otherwise = e3
/// -- becomes --
/// if g1 then e1 else if g2 then e2 else e3
/// ```
///
/// For pattern guards:
/// ```haskell
/// | Just x <- mx = e
/// -- becomes --
/// case mx of { Just x -> e; _ -> error "..." }
/// ```
pub fn desugar_guarded_rhs(
    ctx: &mut LowerContext,
    guarded_rhss: &[ast::GuardedRhs],
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: &impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    guarded_rhss.iter().rev().fold(
        // Default: error "Non-exhaustive guards"
        make_pattern_match_error(ctx, span),
        |else_branch, grhs| {
            // Pass the AST body to desugar_guards - it will be lowered inside
            // the pattern guard scopes so that pattern variables are visible.
            desugar_guards_with_body(
                ctx,
                &grhs.guards,
                &grhs.body,
                else_branch,
                span,
                lower_expr,
                lower_pat,
            )
        },
    )
}

/// Desugar a sequence of guards with an AST body, properly scoping pattern variables.
///
/// This function defers lowering of the body until all pattern guard scopes have been
/// entered, ensuring that pattern-bound variables are visible in the body expression.
fn desugar_guards_with_body(
    ctx: &mut LowerContext,
    guards: &[ast::Guard],
    body: &ast::Expr,
    else_branch: hir::Expr,
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: &impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    match guards {
        [] => {
            // No more guards - now lower the body with all pattern variables in scope
            lower_expr(ctx, body)
        }
        [guard, rest @ ..] => {
            match guard {
                ast::Guard::Expr(cond_expr, _guard_span) => {
                    // Boolean guard: if cond then <recurse> else else_branch
                    let cond = lower_expr(ctx, cond_expr);
                    let then_branch = desugar_guards_with_body(
                        ctx,
                        rest,
                        body,
                        else_branch.clone(),
                        span,
                        lower_expr,
                        lower_pat,
                    );
                    hir::Expr::If(
                        Box::new(cond),
                        Box::new(then_branch),
                        Box::new(else_branch),
                        span,
                    )
                }
                ast::Guard::Pattern(pat, scrut_expr, guard_span) => {
                    // Pattern guard: case scrut of { pat -> <recurse>; _ -> else_branch }
                    // Lower the scrutinee before entering the scope (it shouldn't see pattern vars)
                    let scrut = lower_expr(ctx, scrut_expr);

                    // Enter a new scope for the pattern variables
                    ctx.enter_scope();
                    // Bind pattern variables BEFORE recursing
                    bind_pattern(ctx, pat);
                    let pat_hir = lower_pat(ctx, pat);

                    // Now recurse - the body will be lowered with pattern variables in scope
                    let inner = desugar_guards_with_body(
                        ctx,
                        rest,
                        body,
                        else_branch.clone(),
                        span,
                        lower_expr,
                        lower_pat,
                    );

                    let match_alt = hir::CaseAlt {
                        pat: pat_hir,
                        guards: vec![],
                        rhs: inner,
                        span: *guard_span,
                    };
                    ctx.exit_scope();

                    let default_alt = hir::CaseAlt {
                        pat: hir::Pat::Wild(*guard_span),
                        guards: vec![],
                        rhs: else_branch,
                        span: *guard_span,
                    };

                    hir::Expr::Case(
                        Box::new(scrut),
                        vec![match_alt, default_alt],
                        span,
                    )
                }
            }
        }
    }
}

/// Desugar a sequence of guards into nested if/case expressions.
/// (Legacy version - kept for compatibility, use desugar_guards_with_body for new code)
#[allow(dead_code)]
fn desugar_guards(
    ctx: &mut LowerContext,
    guards: &[ast::Guard],
    then_branch: hir::Expr,
    else_branch: hir::Expr,
    span: Span,
    lower_expr: &impl Fn(&mut LowerContext, &ast::Expr) -> hir::Expr,
    lower_pat: &impl Fn(&mut LowerContext, &ast::Pat) -> hir::Pat,
) -> hir::Expr {
    match guards {
        [] => then_branch,
        [guard, rest @ ..] => {
            // First, desugar the rest of the guards with the then_branch
            let inner = if rest.is_empty() {
                then_branch
            } else {
                desugar_guards(ctx, rest, then_branch, else_branch.clone(), span, lower_expr, lower_pat)
            };

            match guard {
                ast::Guard::Expr(cond_expr, _guard_span) => {
                    // Boolean guard: if cond then inner else else_branch
                    let cond = lower_expr(ctx, cond_expr);
                    hir::Expr::If(
                        Box::new(cond),
                        Box::new(inner),
                        Box::new(else_branch),
                        span,
                    )
                }
                ast::Guard::Pattern(pat, scrut_expr, guard_span) => {
                    // Pattern guard: case scrut of { pat -> inner; _ -> else_branch }
                    let scrut = lower_expr(ctx, scrut_expr);
                    // Enter a new scope for the pattern variables
                    ctx.enter_scope();
                    // Bind pattern variables before lowering
                    bind_pattern(ctx, pat);
                    let pat_hir = lower_pat(ctx, pat);

                    // Note: inner is already computed, so pattern variables are in scope for the RHS
                    let match_alt = hir::CaseAlt {
                        pat: pat_hir,
                        guards: vec![],
                        rhs: inner,
                        span: *guard_span,
                    };
                    ctx.exit_scope();

                    let default_alt = hir::CaseAlt {
                        pat: hir::Pat::Wild(*guard_span),
                        guards: vec![],
                        rhs: else_branch,
                        span: *guard_span,
                    };

                    hir::Expr::Case(
                        Box::new(scrut),
                        vec![match_alt, default_alt],
                        span,
                    )
                }
            }
        }
    }
}

/// Create a reference to a variable (looking it up in scope).
fn make_var_ref(ctx: &mut LowerContext, name: Symbol, span: Span) -> hir::Expr {
    if let Some(def_id) = ctx.lookup_value(name) {
        hir::Expr::Var(ctx.def_ref(def_id, span))
    } else {
        // If not found, create a placeholder (will be caught during type checking)
        let def_id = ctx.fresh_def_id();
        ctx.define(def_id, name, crate::context::DefKind::Value, span);
        hir::Expr::Var(ctx.def_ref(def_id, span))
    }
}

/// Create a pattern match failure error expression.
fn make_pattern_match_error(ctx: &mut LowerContext, span: Span) -> hir::Expr {
    let error_sym = Symbol::intern("error");
    let error_ref = make_var_ref(ctx, error_sym, span);
    let msg = hir::Expr::Lit(
        hir::Lit::String(Symbol::intern("Non-exhaustive patterns")),
        span,
    );
    hir::Expr::App(Box::new(error_ref), Box::new(msg), span)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_desugar_simple_do() {
        let mut ctx = LowerContext::with_builtins();

        // do { map } -- using a builtin that's bound
        let ident = bhc_intern::Ident::from_str("map");
        let stmts = vec![ast::Stmt::Qualifier(
            ast::Expr::Var(ident, Span::default()),
            Span::default(),
        )];

        let result = desugar_do(
            &mut ctx,
            &stmts,
            Span::default(),
            |ctx, e| {
                if let ast::Expr::Var(ident, span) = e {
                    let name = ident.name;
                    if let Some(def_id) = ctx.lookup_value(name) {
                        return hir::Expr::Var(ctx.def_ref(def_id, *span));
                    }
                }
                hir::Expr::Error(Span::default())
            },
            |_ctx, _p| hir::Pat::Wild(Span::default()),
        );

        // Result should be a Var (since `map` is a builtin)
        assert!(matches!(result, hir::Expr::Var(_)));
    }

    #[test]
    fn test_desugar_do_with_let() {
        let mut ctx = LowerContext::with_builtins();

        // do { let x = 5; return x }
        // The variable `x` bound in the let should be visible in the subsequent statement
        let x_ident = bhc_intern::Ident::from_str("x");
        let return_ident = bhc_intern::Ident::from_str("return");

        // Build: let x = 5
        let lit_5 = ast::Expr::Lit(ast::Lit::Int(5), Span::default());
        let let_decl = ast::Decl::FunBind(ast::FunBind {
            name: x_ident,
            clauses: vec![ast::Clause {
                pats: vec![],
                rhs: ast::Rhs::Simple(lit_5, Span::default()),
                wheres: vec![],
                span: Span::default(),
            }],
            span: Span::default(),
        });

        // Build: return x
        let return_expr = ast::Expr::Var(return_ident, Span::default());
        let x_expr = ast::Expr::Var(x_ident, Span::default());
        let return_x = ast::Expr::App(
            Box::new(return_expr),
            Box::new(x_expr),
            Span::default(),
        );

        let stmts = vec![
            ast::Stmt::LetStmt(vec![let_decl], Span::default()),
            ast::Stmt::Qualifier(return_x, Span::default()),
        ];

        let result = desugar_do(
            &mut ctx,
            &stmts,
            Span::default(),
            |ctx, e| {
                match e {
                    ast::Expr::Var(ident, span) => {
                        let name = ident.name;
                        if let Some(def_id) = ctx.lookup_value(name) {
                            hir::Expr::Var(ctx.def_ref(def_id, *span))
                        } else {
                            panic!("unbound variable: {}", name.as_str())
                        }
                    }
                    ast::Expr::Lit(ast::Lit::Int(n), span) => {
                        hir::Expr::Lit(hir::Lit::Int(*n as i128), *span)
                    }
                    ast::Expr::App(f, arg, span) => {
                        // Simplified: just lower f and arg directly
                        let f_expr = match f.as_ref() {
                            ast::Expr::Var(ident, span) => {
                                let name = ident.name;
                                if let Some(def_id) = ctx.lookup_value(name) {
                                    hir::Expr::Var(ctx.def_ref(def_id, *span))
                                } else {
                                    panic!("unbound variable in app: {}", name.as_str())
                                }
                            }
                            _ => hir::Expr::Error(Span::default()),
                        };
                        let arg_expr = match arg.as_ref() {
                            ast::Expr::Var(ident, span) => {
                                let name = ident.name;
                                if let Some(def_id) = ctx.lookup_value(name) {
                                    hir::Expr::Var(ctx.def_ref(def_id, *span))
                                } else {
                                    panic!("unbound variable in arg: {}", name.as_str())
                                }
                            }
                            _ => hir::Expr::Error(Span::default()),
                        };
                        hir::Expr::App(Box::new(f_expr), Box::new(arg_expr), *span)
                    }
                    _ => hir::Expr::Error(Span::default()),
                }
            },
            |_ctx, _p| hir::Pat::Wild(Span::default()),
        );

        // Result should be a Let expression wrapping the body
        assert!(matches!(result, hir::Expr::Let(_, _, _)),
            "expected Let expression, got {:?}", result);
    }
}
