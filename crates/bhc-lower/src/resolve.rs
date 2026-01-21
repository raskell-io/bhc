//! Name resolution for AST to HIR lowering.
//!
//! This module handles resolving identifiers in the AST to their definitions.
//! It tracks:
//!
//! - Variable bindings
//! - Type definitions
//! - Data constructors
//! - Type class methods
//!
//! The resolution is performed during the lowering pass, building up scopes
//! as we descend into the AST.

use bhc_ast as ast;
use bhc_hir::DefId;
use bhc_intern::Symbol;
use bhc_span::Span;

use crate::context::{DefKind, LowerContext};
use crate::LowerError;

/// Resolve a variable reference.
///
/// Returns the `DefId` if found, or records an error and returns `None`.
pub fn resolve_var(ctx: &mut LowerContext, name: Symbol, span: Span) -> Option<DefId> {
    if let Some(def_id) = ctx.lookup_value(name) {
        Some(def_id)
    } else {
        ctx.error(LowerError::UnboundVar {
            name: name.as_str().to_string(),
            span,
        });
        None
    }
}

/// Resolve a type reference.
pub fn resolve_type(ctx: &mut LowerContext, name: Symbol, span: Span) -> Option<DefId> {
    if let Some(def_id) = ctx.lookup_type(name) {
        Some(def_id)
    } else {
        ctx.error(LowerError::UnboundType {
            name: name.as_str().to_string(),
            span,
        });
        None
    }
}

/// Resolve a constructor reference.
pub fn resolve_constructor(ctx: &mut LowerContext, name: Symbol, span: Span) -> Option<DefId> {
    if let Some(def_id) = ctx.lookup_constructor(name) {
        Some(def_id)
    } else {
        ctx.error(LowerError::UnboundCon {
            name: name.as_str().to_string(),
            span,
        });
        None
    }
}

/// Bind a pattern, adding all bound variables to the current scope.
///
/// Returns the list of newly bound (name, DefId) pairs.
pub fn bind_pattern(ctx: &mut LowerContext, pat: &ast::Pat) -> Vec<(Symbol, DefId)> {
    let mut bindings = Vec::new();
    collect_pattern_bindings(ctx, pat, &mut bindings);
    bindings
}

fn collect_pattern_bindings(
    ctx: &mut LowerContext,
    pat: &ast::Pat,
    bindings: &mut Vec<(Symbol, DefId)>,
) {
    match pat {
        ast::Pat::Var(ident, span) => {
            let name = ident.name;
            let def_id = ctx.fresh_def_id();
            ctx.define(def_id, name, DefKind::PatVar, *span);

            // Check for duplicate binding in this pattern
            if ctx.current_scope().lookup_value_local(name).is_some() {
                ctx.error(LowerError::DuplicateDefinition {
                    name: name.as_str().to_string(),
                    new_span: *span,
                    existing_span: *span, // TODO: track original span
                });
            } else {
                ctx.bind_value(name, def_id);
                bindings.push((name, def_id));
            }
        }

        ast::Pat::As(ident, inner, span) => {
            let name = ident.name;
            let def_id = ctx.fresh_def_id();
            ctx.define(def_id, name, DefKind::PatVar, *span);

            if ctx.current_scope().lookup_value_local(name).is_some() {
                ctx.error(LowerError::DuplicateDefinition {
                    name: name.as_str().to_string(),
                    new_span: *span,
                    existing_span: *span,
                });
            } else {
                ctx.bind_value(name, def_id);
                bindings.push((name, def_id));
            }

            collect_pattern_bindings(ctx, inner, bindings);
        }

        ast::Pat::Con(_, pats, _) | ast::Pat::QualCon(_, _, pats, _) => {
            for p in pats {
                collect_pattern_bindings(ctx, p, bindings);
            }
        }

        ast::Pat::Tuple(pats, _) => {
            for p in pats {
                collect_pattern_bindings(ctx, p, bindings);
            }
        }

        ast::Pat::List(pats, _) => {
            for p in pats {
                collect_pattern_bindings(ctx, p, bindings);
            }
        }

        ast::Pat::Infix(left, _op, right, _) => {
            // Infix constructor pattern like `x : xs`
            collect_pattern_bindings(ctx, left, bindings);
            collect_pattern_bindings(ctx, right, bindings);
        }

        ast::Pat::Record(_, fields, _) | ast::Pat::QualRecord(_, _, fields, _) => {
            for field in fields {
                if let Some(p) = &field.pat {
                    collect_pattern_bindings(ctx, p, bindings);
                } else {
                    // Punning: `Foo { x }` binds `x`
                    let name = field.name.name;
                    let def_id = ctx.fresh_def_id();
                    ctx.define(def_id, name, DefKind::PatVar, field.span);
                    ctx.bind_value(name, def_id);
                    bindings.push((name, def_id));
                }
            }
        }

        ast::Pat::Paren(inner, _) | ast::Pat::Ann(inner, _, _) => {
            collect_pattern_bindings(ctx, inner, bindings);
        }

        ast::Pat::Lazy(inner, _) | ast::Pat::Bang(inner, _) => {
            collect_pattern_bindings(ctx, inner, bindings);
        }

        ast::Pat::Wildcard(_) | ast::Pat::Lit(_, _) => {
            // No bindings
        }

        ast::Pat::View(_view_expr, result_pat, _) => {
            // View pattern binds the result pattern
            collect_pattern_bindings(ctx, result_pat, bindings);
        }
    }
}

/// Pre-collect all top-level definitions in a module.
///
/// This allows forward references within the module.
pub fn collect_module_definitions(ctx: &mut LowerContext, module: &ast::Module) {
    for decl in &module.decls {
        match decl {
            ast::Decl::FunBind(fun_bind) => {
                // Check for pattern binding (special name $patbind)
                if fun_bind.name.name.as_str() == "$patbind"
                    && fun_bind.clauses.len() == 1
                    && fun_bind.clauses[0].pats.len() == 1
                {
                    // Pattern binding: bind all variables in the pattern
                    bind_pattern(ctx, &fun_bind.clauses[0].pats[0]);
                } else {
                    // Regular function binding
                    let name = fun_bind.name.name;
                    let def_id = ctx.fresh_def_id();
                    ctx.define(def_id, name, DefKind::Value, fun_bind.span);
                    ctx.bind_value(name, def_id);
                }
            }

            ast::Decl::DataDecl(data_decl) => {
                // Bind the type name
                let type_name = data_decl.name.name;
                let type_def_id = ctx.fresh_def_id();
                ctx.define(type_def_id, type_name, DefKind::Type, data_decl.span);
                ctx.bind_type(type_name, type_def_id);

                // Bind constructors
                for con in &data_decl.constrs {
                    let con_name = con.name.name;
                    let con_def_id = ctx.fresh_def_id();
                    ctx.define(con_def_id, con_name, DefKind::Constructor, con.span);
                    ctx.bind_constructor(con_name, con_def_id);

                    // Record fields also become functions
                    if let ast::ConFields::Record(fields) = &con.fields {
                        for field in fields {
                            let field_name = field.name.name;
                            let field_def_id = ctx.fresh_def_id();
                            ctx.define(field_def_id, field_name, DefKind::Value, field.span);
                            ctx.bind_value(field_name, field_def_id);
                        }
                    }
                }
            }

            ast::Decl::Newtype(newtype_decl) => {
                let type_name = newtype_decl.name.name;
                let type_def_id = ctx.fresh_def_id();
                ctx.define(type_def_id, type_name, DefKind::Type, newtype_decl.span);
                ctx.bind_type(type_name, type_def_id);

                let con_name = newtype_decl.constr.name.name;
                let con_def_id = ctx.fresh_def_id();
                ctx.define(con_def_id, con_name, DefKind::Constructor, newtype_decl.constr.span);
                ctx.bind_constructor(con_name, con_def_id);

                // Record fields also become functions (same as data declarations)
                if let ast::ConFields::Record(fields) = &newtype_decl.constr.fields {
                    for field in fields {
                        let field_name = field.name.name;
                        let field_def_id = ctx.fresh_def_id();
                        ctx.define(field_def_id, field_name, DefKind::Value, field.span);
                        ctx.bind_value(field_name, field_def_id);
                    }
                }
            }

            ast::Decl::TypeAlias(type_alias) => {
                let name = type_alias.name.name;
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Type, type_alias.span);
                ctx.bind_type(name, def_id);
            }

            ast::Decl::ClassDecl(class_decl) => {
                let name = class_decl.name.name;
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Class, class_decl.span);
                ctx.bind_type(name, def_id);

                // Bind method names
                for method in &class_decl.methods {
                    if let ast::Decl::TypeSig(sig) = method {
                        for method_name in &sig.names {
                            let name = method_name.name;
                            let method_def_id = ctx.fresh_def_id();
                            ctx.define(method_def_id, name, DefKind::Value, sig.span);
                            ctx.bind_value(name, method_def_id);
                        }
                    }
                }
            }

            ast::Decl::Foreign(foreign) => {
                let name = foreign.name.name;
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::Value, foreign.span);
                ctx.bind_value(name, def_id);
            }

            ast::Decl::TypeSig(sig) => {
                // Register the type signature for each name
                for name in &sig.names {
                    ctx.register_type_signature(name.name, sig.ty.clone());
                }
            }

            ast::Decl::Fixity(_) | ast::Decl::InstanceDecl(_) | ast::Decl::PragmaDecl(_) => {
                // These don't introduce new names
            }
        }
    }
}

/// Collect type variables from a type, binding them in the current scope.
pub fn bind_type_vars(ctx: &mut LowerContext, ty: &ast::Type) -> Vec<(Symbol, DefId)> {
    let mut bindings = Vec::new();
    collect_type_vars(ctx, ty, &mut bindings);
    bindings
}

fn collect_type_vars(
    ctx: &mut LowerContext,
    ty: &ast::Type,
    bindings: &mut Vec<(Symbol, DefId)>,
) {
    match ty {
        ast::Type::Var(tyvar, span) => {
            let name = tyvar.name.name;
            // Only bind if not already bound
            if ctx.lookup_type(name).is_none() && !bindings.iter().any(|(n, _)| *n == name) {
                let def_id = ctx.fresh_def_id();
                ctx.define(def_id, name, DefKind::TyVar, *span);
                ctx.bind_type(name, def_id);
                bindings.push((name, def_id));
            }
        }

        ast::Type::App(f, a, _) => {
            collect_type_vars(ctx, f, bindings);
            collect_type_vars(ctx, a, bindings);
        }

        ast::Type::Fun(from, to, _) => {
            collect_type_vars(ctx, from, bindings);
            collect_type_vars(ctx, to, bindings);
        }

        ast::Type::Tuple(tys, _) => {
            for t in tys {
                collect_type_vars(ctx, t, bindings);
            }
        }

        ast::Type::List(elem, _) => {
            collect_type_vars(ctx, elem, bindings);
        }

        ast::Type::Paren(inner, _) => {
            collect_type_vars(ctx, inner, bindings);
        }

        ast::Type::Constrained(constraints, inner, _) => {
            for c in constraints {
                for arg in &c.args {
                    collect_type_vars(ctx, arg, bindings);
                }
            }
            collect_type_vars(ctx, inner, bindings);
        }

        ast::Type::Forall(vars, inner, _) => {
            // Forall binds its own variables
            for var in vars {
                let name = var.name.name;
                if !bindings.iter().any(|(n, _)| *n == name) {
                    let def_id = ctx.fresh_def_id();
                    ctx.define(def_id, name, DefKind::TyVar, var.span);
                    bindings.push((name, def_id));
                }
            }
            collect_type_vars(ctx, inner, bindings);
        }

        ast::Type::Con(_, _)
        | ast::Type::QualCon(_, _, _)
        | ast::Type::NatLit(_, _)
        | ast::Type::PromotedList(_, _)
        | ast::Type::Bang(_, _)
        | ast::Type::Lazy(_, _) => {
            // No type variables to collect in these
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_builtin() {
        let mut ctx = LowerContext::with_builtins();

        let map = Symbol::intern("map");
        let result = resolve_var(&mut ctx, map, Span::default());
        assert!(result.is_some());
        assert!(!ctx.has_errors());
    }

    #[test]
    fn test_resolve_unbound() {
        let mut ctx = LowerContext::with_builtins();

        let unknown = Symbol::intern("unknownVariable");
        let result = resolve_var(&mut ctx, unknown, Span::default());
        assert!(result.is_none());
        assert!(ctx.has_errors());
    }

    #[test]
    fn test_resolve_type() {
        let mut ctx = LowerContext::with_builtins();

        let int = Symbol::intern("Int");
        let result = resolve_type(&mut ctx, int, Span::default());
        assert!(result.is_some());
    }

    #[test]
    fn test_resolve_constructor() {
        let mut ctx = LowerContext::with_builtins();

        let just = Symbol::intern("Just");
        let result = resolve_constructor(&mut ctx, just, Span::default());
        assert!(result.is_some());
    }
}
