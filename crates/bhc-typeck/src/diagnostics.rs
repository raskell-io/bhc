//! Diagnostic emission for type errors.
//!
//! This module provides functions to emit type error diagnostics with
//! helpful messages and suggestions.

use bhc_diagnostics::Diagnostic;
use bhc_hir::DefId;
use bhc_span::Span;
use bhc_types::{Ty, TyVar};

use crate::context::TyCtxt;

/// Emit a type mismatch error.
pub fn emit_type_mismatch(ctx: &mut TyCtxt, expected: &Ty, found: &Ty, span: Span) {
    let diag = Diagnostic::error(format!(
        "type mismatch: expected `{}`, found `{}`",
        pretty_ty(expected),
        pretty_ty(found)
    ))
    .with_code("E0001")
    .with_label(ctx.full_span(span), "type mismatch here");

    ctx.emit_error(diag);
}

/// Emit an occurs check error (infinite type).
pub fn emit_occurs_check_error(ctx: &mut TyCtxt, var: &TyVar, ty: &Ty, span: Span) {
    let diag = Diagnostic::error(format!(
        "infinite type: type variable `{}` occurs in `{}`",
        pretty_tyvar(var),
        pretty_ty(ty)
    ))
    .with_code("E0002")
    .with_label(ctx.full_span(span), "infinite type detected")
    .with_note("This would create an infinitely recursive type");

    ctx.emit_error(diag);
}

/// Emit an unbound variable error.
pub fn emit_unbound_var(ctx: &mut TyCtxt, def_id: DefId, span: Span) {
    let diag = Diagnostic::error(format!("unbound variable (DefId: {def_id:?})"))
        .with_code("E0003")
        .with_label(ctx.full_span(span), "not in scope");

    ctx.emit_error(diag);
}

/// Emit an unbound constructor error.
pub fn emit_unbound_constructor(ctx: &mut TyCtxt, def_id: DefId, span: Span) {
    let diag = Diagnostic::error(format!("unbound constructor (DefId: {def_id:?})"))
        .with_code("E0004")
        .with_label(ctx.full_span(span), "constructor not in scope");

    ctx.emit_error(diag);
}

/// Emit a "too many pattern arguments" error.
pub fn emit_too_many_pattern_args(ctx: &mut TyCtxt, span: Span) {
    let diag = Diagnostic::error("too many arguments in pattern")
        .with_code("E0005")
        .with_label(ctx.full_span(span), "extra arguments here");

    ctx.emit_error(diag);
}

/// Emit an ambiguous type variable error.
#[allow(dead_code)]
pub fn emit_ambiguous_type(ctx: &mut TyCtxt, var: &TyVar, span: Span) {
    let diag = Diagnostic::error(format!(
        "ambiguous type variable: `{}` could not be resolved",
        pretty_tyvar(var)
    ))
    .with_code("E0006")
    .with_label(ctx.full_span(span), "ambiguous type")
    .with_note("Consider adding a type annotation");

    ctx.emit_error(diag);
}

/// Emit a kind mismatch error.
#[allow(dead_code)]
pub fn emit_kind_mismatch(
    ctx: &mut TyCtxt,
    expected: &str,
    found: &str,
    span: Span,
) {
    let diag = Diagnostic::error(format!(
        "kind mismatch: expected `{expected}`, found `{found}`"
    ))
    .with_code("E0007")
    .with_label(ctx.full_span(span), "kind mismatch");

    ctx.emit_error(diag);
}

/// Emit a "missing type signature" warning.
#[allow(dead_code)]
pub fn emit_missing_signature(ctx: &mut TyCtxt, name: &str, ty: &Ty, span: Span) {
    let diag = Diagnostic::warning(format!(
        "missing type signature for `{name}`: inferred type is `{}`",
        pretty_ty(ty)
    ))
    .with_label(ctx.full_span(span), "add type signature");

    ctx.emit_error(diag);
}

/// Pretty-print a type.
fn pretty_ty(ty: &Ty) -> String {
    match ty {
        Ty::Var(v) => pretty_tyvar(v),
        Ty::Con(c) => c.name.as_str().to_string(),
        Ty::Prim(p) => p.name().to_string(),
        Ty::App(f, a) => format!("({} {})", pretty_ty(f), pretty_ty(a)),
        Ty::Fun(from, to) => format!("({} -> {})", pretty_ty(from), pretty_ty(to)),
        Ty::Tuple(tys) if tys.is_empty() => "()".to_string(),
        Ty::Tuple(tys) => {
            let inner: Vec<_> = tys.iter().map(pretty_ty).collect();
            format!("({})", inner.join(", "))
        }
        Ty::List(elem) => format!("[{}]", pretty_ty(elem)),
        Ty::Forall(vars, body) => {
            let var_names: Vec<_> = vars.iter().map(pretty_tyvar).collect();
            format!("forall {}. {}", var_names.join(" "), pretty_ty(body))
        }
        Ty::Error => "<error>".to_string(),
    }
}

/// Pretty-print a type variable.
fn pretty_tyvar(var: &TyVar) -> String {
    // Use a, b, c, ... for the first 26, then t1, t2, ...
    if var.id < 26 {
        // SAFETY: We've verified var.id < 26, so it fits in u8
        #[allow(clippy::cast_possible_truncation)]
        let c = (b'a' + var.id as u8) as char;
        c.to_string()
    } else {
        format!("t{}", var.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_intern::Symbol;
    use bhc_types::{Kind, TyCon};

    #[test]
    fn test_pretty_ty() {
        let int = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        assert_eq!(pretty_ty(&int), "Int");

        let a = TyVar::new_star(0);
        assert_eq!(pretty_ty(&Ty::Var(a)), "a");

        let func = Ty::fun(int.clone(), int.clone());
        assert_eq!(pretty_ty(&func), "(Int -> Int)");

        let list = Ty::List(Box::new(int));
        assert_eq!(pretty_ty(&list), "[Int]");
    }

    #[test]
    fn test_pretty_tyvar() {
        assert_eq!(pretty_tyvar(&TyVar::new_star(0)), "a");
        assert_eq!(pretty_tyvar(&TyVar::new_star(1)), "b");
        assert_eq!(pretty_tyvar(&TyVar::new_star(25)), "z");
        assert_eq!(pretty_tyvar(&TyVar::new_star(26)), "t26");
    }
}
