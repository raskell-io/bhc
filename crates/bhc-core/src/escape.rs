//! Escape analysis for Core IR.
//!
//! This module implements escape analysis to determine which allocations
//! can be stack-allocated (don't escape) vs heap-allocated (escape).
//!
//! # Escape Rules
//!
//! An allocation escapes its defining scope if:
//! - It is returned from a function
//! - It is stored in a data structure that escapes
//! - It is passed to a function that may store it
//! - It is captured in a closure that escapes
//!
//! # Use Cases
//!
//! - **Embedded profile**: Reject programs where allocations escape (no GC)
//! - **Optimization**: Stack-allocate non-escaping values for better performance
//! - **Arena allocation**: Determine arena lifetimes for realtime profile
//!
//! # Algorithm
//!
//! We use a backwards dataflow analysis:
//! 1. Mark return positions as escaping
//! 2. Propagate escape status backwards through the control flow
//! 3. Variables that flow to escaping positions are marked as escaping
//!
//! See H26-SPEC Section 9.2 for escape analysis specification.

use crate::{Bind, Expr, VarId};
use rustc_hash::{FxHashMap, FxHashSet};

/// Result of escape analysis for a Core IR expression.
#[derive(Debug, Clone)]
pub struct EscapeAnalysis {
    /// Variables that escape their defining scope.
    pub escaping: FxHashSet<VarId>,
    /// Variables that are captured in closures.
    pub captured: FxHashSet<VarId>,
    /// Variables that are returned from functions.
    pub returned: FxHashSet<VarId>,
    /// Allocation sites that escape.
    pub escaping_allocs: FxHashSet<VarId>,
}

impl EscapeAnalysis {
    /// Create a new empty escape analysis result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            escaping: FxHashSet::default(),
            captured: FxHashSet::default(),
            returned: FxHashSet::default(),
            escaping_allocs: FxHashSet::default(),
        }
    }

    /// Check if a variable escapes.
    #[must_use]
    pub fn escapes(&self, var: VarId) -> bool {
        self.escaping.contains(&var)
    }

    /// Check if a variable is captured in a closure.
    #[must_use]
    pub fn is_captured(&self, var: VarId) -> bool {
        self.captured.contains(&var)
    }

    /// Check if a variable is returned from a function.
    #[must_use]
    pub fn is_returned(&self, var: VarId) -> bool {
        self.returned.contains(&var)
    }

    /// Get all escaping variables.
    #[must_use]
    pub fn escaping_vars(&self) -> &FxHashSet<VarId> {
        &self.escaping
    }

    /// Check if the analysis found any escaping allocations.
    #[must_use]
    pub fn has_escaping_allocs(&self) -> bool {
        !self.escaping_allocs.is_empty()
    }
}

impl Default for EscapeAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape status of a variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeStatus {
    /// Does not escape its defining scope.
    NoEscape,
    /// Escapes via return.
    EscapeReturn,
    /// Escapes via closure capture.
    EscapeCapture,
    /// Escapes via being stored in an escaping structure.
    EscapeStore,
    /// Escapes via being passed to an external function.
    EscapeExternal,
}

/// Context for escape analysis.
#[derive(Debug)]
struct EscapeContext {
    /// Current escape status for each variable.
    status: FxHashMap<VarId, EscapeStatus>,
    /// Variables defined in the current scope.
    local_scope: FxHashSet<VarId>,
    /// Variables captured from outer scopes.
    captured: FxHashSet<VarId>,
    /// Whether we're analyzing inside a lambda (for capture detection).
    in_lambda: bool,
    /// Variables bound in enclosing lambdas.
    lambda_bound: FxHashSet<VarId>,
}

impl EscapeContext {
    fn new() -> Self {
        Self {
            status: FxHashMap::default(),
            local_scope: FxHashSet::default(),
            captured: FxHashSet::default(),
            in_lambda: false,
            lambda_bound: FxHashSet::default(),
        }
    }

    fn mark_escaping(&mut self, var: VarId, status: EscapeStatus) {
        // Only upgrade escape status, never downgrade
        self.status
            .entry(var)
            .and_modify(|s| {
                if *s == EscapeStatus::NoEscape {
                    *s = status;
                }
            })
            .or_insert(status);
    }
}

/// Analyze escape for a Core IR expression.
///
/// Returns an `EscapeAnalysis` containing information about which
/// variables and allocations escape their defining scope.
///
/// # Example
///
/// ```ignore
/// let expr = parse_core("let x = alloc() in return x");
/// let analysis = analyze_escape(&expr);
/// assert!(analysis.escapes(x_var_id));
/// ```
pub fn analyze_escape(expr: &Expr) -> EscapeAnalysis {
    let mut ctx = EscapeContext::new();
    let mut result = EscapeAnalysis::new();

    // Analyze the expression, treating the top level as returning
    analyze_expr(expr, &mut ctx, true);

    // Collect results
    for (var_id, status) in &ctx.status {
        if *status != EscapeStatus::NoEscape {
            result.escaping.insert(*var_id);
            result.escaping_allocs.insert(*var_id);
        }
        if *status == EscapeStatus::EscapeReturn {
            result.returned.insert(*var_id);
        }
    }

    for var_id in &ctx.captured {
        result.captured.insert(*var_id);
    }

    result
}

/// Analyze escape for an expression.
///
/// `in_return_position` indicates whether this expression's value
/// will be returned from the enclosing function.
fn analyze_expr(expr: &Expr, ctx: &mut EscapeContext, in_return_position: bool) {
    match expr {
        Expr::Var(var, _) => {
            // If a variable is in return position, it escapes
            if in_return_position {
                ctx.mark_escaping(var.id, EscapeStatus::EscapeReturn);
            }
            // Check for closure capture
            if ctx.in_lambda && !ctx.lambda_bound.contains(&var.id) {
                ctx.captured.insert(var.id);
                ctx.mark_escaping(var.id, EscapeStatus::EscapeCapture);
            }
        }

        Expr::Lit(_, _, _) => {
            // Literals don't escape (they're values)
        }

        Expr::App(func, arg, _) => {
            // Function itself doesn't escape through application
            analyze_expr(func, ctx, false);

            // Arguments passed to functions may escape
            // Conservative: assume external functions may store arguments
            if is_external_call(func) {
                mark_expr_escaping(arg, ctx, EscapeStatus::EscapeExternal);
            } else {
                // For known functions, we could do interprocedural analysis
                // For now, be conservative
                analyze_expr(arg, ctx, false);
            }

            // If the result is in return position, we need to track
            // what flows through the function (but that requires more
            // sophisticated analysis)
        }

        Expr::TyApp(func, _, _) => {
            // Type application doesn't affect escape
            analyze_expr(func, ctx, in_return_position);
        }

        Expr::Lam(param, body, _) => {
            // Enter lambda context
            let was_in_lambda = ctx.in_lambda;
            ctx.in_lambda = true;
            ctx.lambda_bound.insert(param.id);

            // The body is in return position within the lambda
            analyze_expr(body, ctx, true);

            // Exit lambda context
            ctx.lambda_bound.remove(&param.id);
            ctx.in_lambda = was_in_lambda;

            // A lambda itself escapes if it's in return position
            // (we don't track the lambda as a VarId, but captured vars do)
        }

        Expr::TyLam(_, body, _) => {
            // Type lambda doesn't change escape
            analyze_expr(body, ctx, in_return_position);
        }

        Expr::Let(bind, body, _) => {
            analyze_bind(bind, ctx);
            analyze_expr(body, ctx, in_return_position);
        }

        Expr::Case(scrutinee, alts, _, _) => {
            // Scrutinee doesn't escape just by being cased on
            analyze_expr(scrutinee, ctx, false);

            // Each alternative's RHS is in return position if the case is
            for alt in alts {
                for binder in &alt.binders {
                    ctx.local_scope.insert(binder.id);
                }
                analyze_expr(&alt.rhs, ctx, in_return_position);
                for binder in &alt.binders {
                    ctx.local_scope.remove(&binder.id);
                }
            }
        }

        Expr::Lazy(inner, _) => {
            // Lazy wraps the expression in a thunk
            // Variables captured in the thunk escape if the thunk escapes
            analyze_expr(inner, ctx, in_return_position);
        }

        Expr::Cast(inner, _, _) => {
            analyze_expr(inner, ctx, in_return_position);
        }

        Expr::Tick(_, inner, _) => {
            analyze_expr(inner, ctx, in_return_position);
        }

        Expr::Type(_, _) | Expr::Coercion(_, _) => {
            // Types and coercions don't have escape behavior
        }
    }
}

/// Analyze a binding.
fn analyze_bind(bind: &Bind, ctx: &mut EscapeContext) {
    match bind {
        Bind::NonRec(var, rhs) => {
            ctx.local_scope.insert(var.id);
            // RHS is not in return position (it's bound to a variable)
            analyze_expr(rhs, ctx, false);
        }
        Bind::Rec(bindings) => {
            // Add all bindings to scope first (for mutual recursion)
            for (var, _) in bindings {
                ctx.local_scope.insert(var.id);
            }
            // Analyze each RHS
            for (_, rhs) in bindings {
                analyze_expr(rhs, ctx, false);
            }
        }
    }
}

/// Mark all variables in an expression as escaping with the given status.
fn mark_expr_escaping(expr: &Expr, ctx: &mut EscapeContext, status: EscapeStatus) {
    match expr {
        Expr::Var(var, _) => {
            ctx.mark_escaping(var.id, status);
        }
        Expr::App(func, arg, _) => {
            mark_expr_escaping(func, ctx, status);
            mark_expr_escaping(arg, ctx, status);
        }
        Expr::TyApp(func, _, _) => {
            mark_expr_escaping(func, ctx, status);
        }
        Expr::Let(bind, body, _) => {
            match bind.as_ref() {
                Bind::NonRec(_, rhs) => mark_expr_escaping(rhs, ctx, status),
                Bind::Rec(bindings) => {
                    for (_, rhs) in bindings {
                        mark_expr_escaping(rhs, ctx, status);
                    }
                }
            }
            mark_expr_escaping(body, ctx, status);
        }
        Expr::Case(scrutinee, alts, _, _) => {
            mark_expr_escaping(scrutinee, ctx, status);
            for alt in alts {
                mark_expr_escaping(&alt.rhs, ctx, status);
            }
        }
        Expr::Lam(_, body, _) | Expr::TyLam(_, body, _) => {
            mark_expr_escaping(body, ctx, status);
        }
        Expr::Lazy(inner, _) | Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) => {
            mark_expr_escaping(inner, ctx, status);
        }
        Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => {}
    }
}

/// Check if an expression is a call to an external function.
fn is_external_call(expr: &Expr) -> bool {
    match expr {
        Expr::Var(var, _) => {
            // Check if the variable is an FFI import or primitive
            // This would normally check against a set of known externals
            // For now, check if name suggests it's external
            let name = var.name.as_str();
            name.starts_with("ffi_") || name.starts_with("prim_") || name.starts_with("foreign_")
        }
        Expr::TyApp(func, _, _) => is_external_call(func),
        _ => false,
    }
}

/// Check if a program is valid for the embedded profile (no escaping allocations).
///
/// Returns `Ok(())` if the program can run without GC, or `Err` with
/// a list of escaping allocations that would require GC.
pub fn check_embedded_safe(expr: &Expr) -> Result<(), Vec<EscapingAllocation>> {
    let analysis = analyze_escape(expr);

    if analysis.has_escaping_allocs() {
        let errors: Vec<_> = analysis
            .escaping_allocs
            .iter()
            .map(|var_id| EscapingAllocation {
                var_id: *var_id,
                reason: if analysis.is_returned(*var_id) {
                    EscapeReason::Returned
                } else if analysis.is_captured(*var_id) {
                    EscapeReason::Captured
                } else {
                    EscapeReason::PassedToExternal
                },
            })
            .collect();
        Err(errors)
    } else {
        Ok(())
    }
}

/// An allocation that escapes and would require GC.
#[derive(Debug, Clone)]
pub struct EscapingAllocation {
    /// The variable ID of the escaping allocation.
    pub var_id: VarId,
    /// Why the allocation escapes.
    pub reason: EscapeReason,
}

/// Reason why an allocation escapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeReason {
    /// Allocation is returned from a function.
    Returned,
    /// Allocation is captured in a closure.
    Captured,
    /// Allocation is passed to an external function.
    PassedToExternal,
    /// Allocation is stored in an escaping data structure.
    StoredInEscaping,
}

impl std::fmt::Display for EscapeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Returned => write!(f, "returned from function"),
            Self::Captured => write!(f, "captured in closure"),
            Self::PassedToExternal => write!(f, "passed to external function"),
            Self::StoredInEscaping => write!(f, "stored in escaping data structure"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Alt, AltCon, DataCon, Literal, Var};
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::{Kind, Ty, TyCon};

    fn make_var(name: &str, id: u32) -> Var {
        Var {
            name: Symbol::intern(name),
            id: VarId(id),
            ty: Ty::int_prim(),
        }
    }

    fn make_data_con(name: &str, tag: u32) -> DataCon {
        DataCon {
            name: Symbol::intern(name),
            ty_con: TyCon::new(Symbol::intern("Bool"), Kind::Star),
            tag,
            arity: 0,
        }
    }

    fn dummy_span() -> Span {
        Span::DUMMY
    }

    #[test]
    fn test_literal_no_escape() {
        let expr = Expr::Lit(Literal::Int(42), Ty::int_prim(), dummy_span());
        let analysis = analyze_escape(&expr);
        assert!(!analysis.has_escaping_allocs());
    }

    #[test]
    fn test_var_in_return_escapes() {
        let var = make_var("x", 1);
        let expr = Expr::Var(var.clone(), dummy_span());
        let analysis = analyze_escape(&expr);
        assert!(analysis.escapes(var.id));
        assert!(analysis.is_returned(var.id));
    }

    #[test]
    fn test_let_bound_no_escape() {
        // let x = 42 in x + 1
        let x = make_var("x", 1);
        let lit = Expr::Lit(Literal::Int(42), Ty::int_prim(), dummy_span());
        let body = Expr::Lit(Literal::Int(1), Ty::int_prim(), dummy_span()); // Simplified
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x.clone(), Box::new(lit))),
            Box::new(body),
            dummy_span(),
        );
        let analysis = analyze_escape(&expr);
        // x is bound to a literal, doesn't escape via return
        // (the body is just a literal, not x)
        assert!(!analysis.escapes(x.id));
    }

    #[test]
    fn test_let_returned_escapes() {
        // let x = 42 in x
        let x = make_var("x", 1);
        let lit = Expr::Lit(Literal::Int(42), Ty::int_prim(), dummy_span());
        let body = Expr::Var(x.clone(), dummy_span());
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x.clone(), Box::new(lit))),
            Box::new(body),
            dummy_span(),
        );
        let analysis = analyze_escape(&expr);
        // x is returned from the let, so it escapes
        assert!(analysis.escapes(x.id));
    }

    #[test]
    fn test_lambda_capture_escapes() {
        // let x = 42 in \y -> x
        let x = make_var("x", 1);
        let y = make_var("y", 2);
        let lit = Expr::Lit(Literal::Int(42), Ty::int_prim(), dummy_span());
        let lambda_body = Expr::Var(x.clone(), dummy_span());
        let lambda = Expr::Lam(y.clone(), Box::new(lambda_body), dummy_span());
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x.clone(), Box::new(lit))),
            Box::new(lambda),
            dummy_span(),
        );
        let analysis = analyze_escape(&expr);
        // x is captured in the lambda
        assert!(analysis.is_captured(x.id));
        assert!(analysis.escapes(x.id));
    }

    #[test]
    fn test_check_embedded_safe_ok() {
        // A simple literal is safe for embedded
        let expr = Expr::Lit(Literal::Int(42), Ty::int_prim(), dummy_span());
        assert!(check_embedded_safe(&expr).is_ok());
    }

    #[test]
    fn test_check_embedded_safe_escaping() {
        // let x = 42 in x (x escapes via return)
        let x = make_var("x", 1);
        let lit = Expr::Lit(Literal::Int(42), Ty::int_prim(), dummy_span());
        let body = Expr::Var(x.clone(), dummy_span());
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x.clone(), Box::new(lit))),
            Box::new(body),
            dummy_span(),
        );
        let result = check_embedded_safe(&expr);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].reason, EscapeReason::Returned);
    }

    #[test]
    fn test_case_alternatives() {
        // case x of { True -> y; False -> z }
        let x = make_var("x", 1);
        let y = make_var("y", 2);
        let z = make_var("z", 3);

        let scrutinee = Expr::Var(x.clone(), dummy_span());
        let alts = vec![
            Alt {
                con: AltCon::DataCon(make_data_con("True", 0)),
                binders: vec![],
                rhs: Expr::Var(y.clone(), dummy_span()),
            },
            Alt {
                con: AltCon::DataCon(make_data_con("False", 1)),
                binders: vec![],
                rhs: Expr::Var(z.clone(), dummy_span()),
            },
        ];
        let expr = Expr::Case(Box::new(scrutinee), alts, Ty::int_prim(), dummy_span());
        let analysis = analyze_escape(&expr);

        // Both y and z escape via return (they're in return position)
        assert!(analysis.escapes(y.id));
        assert!(analysis.escapes(z.id));
        // x is scrutinized, but doesn't escape
        assert!(!analysis.escapes(x.id));
    }
}
