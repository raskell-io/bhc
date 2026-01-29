//! Diagnostic emission for type errors.
//!
//! This module provides functions to emit type error diagnostics with
//! helpful messages and suggestions in Cargo/Rust style.
//!
//! ## Error Codes
//!
//! - `E0001-E0007`: Basic type errors (mismatch, unbound, occurs check)
//! - `E0008-E0019`: Reserved for basic type errors
//! - `E0020-E0029`: Shape/dimension errors (M9 dependent types)
//! - `E0030-E0039`: Tensor operation errors (matmul, broadcast, etc.)
//!
//! ## M10 Cargo-Quality Diagnostics
//!
//! This module implements Phase 2 of M10: Type Error Overhaul with:
//! - Aligned type comparisons for mismatches
//! - "Did you mean?" suggestions for typos
//! - Function arity mismatch with argument highlighting
//! - Detailed unification trails for complex errors

use bhc_diagnostics::{Applicability, Diagnostic, Suggestion};
use bhc_hir::DefId;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Ty, TyList, TyNat, TyVar};

use crate::context::TyCtxt;
use crate::shape_diagrams;
use crate::suggest::{find_similar_names, format_suggestions};

/// Emit a type mismatch error with aligned comparison.
///
/// This provides a Cargo-style error message with:
/// - Clear "expected X, found Y" formatting
/// - Aligned type comparison for easy visual diff
/// - Suggestions for common fixes
pub fn emit_type_mismatch(ctx: &mut TyCtxt, expected: &Ty, found: &Ty, span: Span) {
    let expected_str = pretty_ty(expected);
    let found_str = pretty_ty(found);

    let diag = Diagnostic::error(format!(
        "type mismatch: expected `{expected_str}`, found `{found_str}`"
    ))
    .with_code("E0001")
    .with_label(ctx.full_span(span), format!("expected `{expected_str}`"))
    .with_note(format_type_comparison(expected, found));

    ctx.emit_error(diag);
}

/// Emit a type mismatch error with additional context about where the expectation came from.
pub fn emit_type_mismatch_with_context(
    ctx: &mut TyCtxt,
    expected: &Ty,
    found: &Ty,
    span: Span,
    context: &str,
) {
    let expected_str = pretty_ty(expected);
    let found_str = pretty_ty(found);

    let diag = Diagnostic::error(format!(
        "type mismatch: expected `{expected_str}`, found `{found_str}`"
    ))
    .with_code("E0001")
    .with_label(
        ctx.full_span(span),
        format!("expected `{expected_str}` because {context}"),
    )
    .with_note(format_type_comparison(expected, found));

    ctx.emit_error(diag);
}

/// Format a visual comparison of two types for error messages.
fn format_type_comparison(expected: &Ty, found: &Ty) -> String {
    let expected_str = pretty_ty(expected);
    let found_str = pretty_ty(found);

    // Calculate padding for alignment
    let max_len = expected_str.len().max(found_str.len());
    let expected_padded = format!("{expected_str:>width$}", width = max_len);
    let found_padded = format!("{found_str:>width$}", width = max_len);

    format!("expected: {expected_padded}\n   found: {found_padded}")
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
    let diag = Diagnostic::error(format!("cannot find value `{def_id:?}` in this scope"))
        .with_code("E0003")
        .with_label(ctx.full_span(span), "not found in this scope")
        .with_note("variables must be defined before use");

    ctx.emit_error(diag);
}

/// Emit an unbound variable error with the actual name and suggestions.
///
/// This enhanced version provides "Did you mean?" suggestions based on
/// similar names in scope.
pub fn emit_unbound_var_named(
    ctx: &mut TyCtxt,
    name: Symbol,
    span: Span,
    names_in_scope: &[Symbol],
) {
    let name_str = name.as_str();

    let mut diag = Diagnostic::error(format!("cannot find value `{name_str}` in this scope"))
        .with_code("E0003")
        .with_label(ctx.full_span(span), "not found in this scope");

    // Find similar names for "Did you mean?" suggestions
    let suggestions = find_similar_names(name_str, names_in_scope);
    if let Some(suggestion_text) = format_suggestions(&suggestions) {
        diag = diag.with_note(suggestion_text);

        // If there's a single good suggestion, add a code suggestion
        if suggestions.len() == 1 && suggestions[0].distance <= 2 {
            let suggested_name = suggestions[0].name.as_str().to_string();
            diag = diag.with_suggestion(Suggestion::new(
                format!("a value with a similar name exists: `{suggested_name}`"),
                ctx.full_span(span),
                suggested_name,
                Applicability::MaybeIncorrect,
            ));
        }
    }

    ctx.emit_error(diag);
}

/// Emit an unbound constructor error.
pub fn emit_unbound_constructor(ctx: &mut TyCtxt, def_id: DefId, span: Span) {
    let diag = Diagnostic::error(format!(
        "cannot find data constructor `{def_id:?}` in this scope"
    ))
    .with_code("E0004")
    .with_label(ctx.full_span(span), "constructor not found")
    .with_note("data constructors must be imported or defined before use");

    ctx.emit_error(diag);
}

/// Emit an unbound constructor error with the actual name and suggestions.
pub fn emit_unbound_constructor_named(
    ctx: &mut TyCtxt,
    name: Symbol,
    span: Span,
    constructors_in_scope: &[Symbol],
) {
    let name_str = name.as_str();

    let mut diag = Diagnostic::error(format!(
        "cannot find data constructor `{name_str}` in this scope"
    ))
    .with_code("E0004")
    .with_label(ctx.full_span(span), "constructor not found");

    // Find similar constructor names
    let suggestions = find_similar_names(name_str, constructors_in_scope);
    if let Some(suggestion_text) = format_suggestions(&suggestions) {
        diag = diag.with_note(suggestion_text);

        if suggestions.len() == 1 && suggestions[0].distance <= 2 {
            let suggested_name = suggestions[0].name.as_str().to_string();
            diag = diag.with_suggestion(Suggestion::new(
                format!("a constructor with a similar name exists: `{suggested_name}`"),
                ctx.full_span(span),
                suggested_name,
                Applicability::MaybeIncorrect,
            ));
        }
    } else {
        diag = diag.with_note("data constructors must start with an uppercase letter");
    }

    ctx.emit_error(diag);
}

/// Emit a "too many pattern arguments" error.
pub fn emit_too_many_pattern_args(ctx: &mut TyCtxt, span: Span) {
    let diag = Diagnostic::error("too many arguments in pattern")
        .with_code("E0005")
        .with_label(ctx.full_span(span), "extra arguments here")
        .with_note("check the data constructor's definition for the expected number of fields");

    ctx.emit_error(diag);
}

/// Emit a pattern arity mismatch error with detailed information.
pub fn emit_pattern_arity_mismatch(
    ctx: &mut TyCtxt,
    constructor_name: &str,
    expected: usize,
    found: usize,
    span: Span,
) {
    let plural = |n: usize| if n == 1 { "argument" } else { "arguments" };

    let diag = Diagnostic::error(format!(
        "constructor `{constructor_name}` expects {expected} {}, but {found} {} supplied",
        plural(expected),
        if found == 1 { "was" } else { "were" }
    ))
    .with_code("E0005")
    .with_label(
        ctx.full_span(span),
        if found > expected {
            format!("{} extra {}", found - expected, plural(found - expected))
        } else {
            format!("{} missing {}", expected - found, plural(expected - found))
        },
    );

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
pub fn emit_kind_mismatch(ctx: &mut TyCtxt, expected: &str, found: &str, span: Span) {
    let diag = Diagnostic::error(format!(
        "kind mismatch: expected kind `{expected}`, found kind `{found}`"
    ))
    .with_code("E0007")
    .with_label(ctx.full_span(span), format!("expected kind `{expected}`"))
    .with_note("kinds classify types: * is for concrete types, * -> * is for type constructors");

    ctx.emit_error(diag);
}

/// Emit a "no instance" error when type class constraint cannot be satisfied.
pub fn emit_no_instance(ctx: &mut TyCtxt, class: Symbol, ty: &Ty, span: Span) {
    let class_name = class.as_str();
    let ty_str = pretty_ty(ty);

    let diag = Diagnostic::error(format!("no instance for `{class_name} {ty_str}`"))
        .with_code("E0040")
        .with_label(
            ctx.full_span(span),
            format!("no `{class_name}` instance for `{ty_str}`"),
        )
        .with_note(format!(
            "To use this operation, `{ty_str}` must be an instance of `{class_name}`.\n\
         Consider adding a type annotation or instance declaration."
        ));

    ctx.emit_error(diag);
}

// === Type Family Errors (E0041-E0049) ===

/// Emit a type family reduction failure error.
///
/// Called when an associated type family cannot be reduced because no
/// matching instance was found.
///
/// For example: `Elem Bool` when there's no `instance Collection Bool`.
pub fn emit_type_family_reduction_failed(
    ctx: &mut TyCtxt,
    family_name: Symbol,
    args: &[Ty],
    class_name: Option<Symbol>,
    span: Span,
) {
    let family_str = family_name.as_str();
    let args_str: Vec<_> = args.iter().map(pretty_ty).collect();
    let applied_str = if args_str.is_empty() {
        family_str.to_string()
    } else {
        format!("{} {}", family_str, args_str.join(" "))
    };

    let mut diag = Diagnostic::error(format!("cannot reduce type family `{applied_str}`"))
        .with_code("E0041")
        .with_label(ctx.full_span(span), "type family cannot be reduced");

    if let Some(class) = class_name {
        let class_str = class.as_str();
        diag = diag.with_note(format!(
            "`{family_str}` is an associated type of class `{class_str}`.\n\
             No matching instance of `{class_str}` was found for the given types.\n\
             Consider adding an instance declaration."
        ));
    } else {
        diag = diag.with_note(format!(
            "No matching type family instance was found for `{applied_str}`.\n\
             Consider adding a type family instance declaration."
        ));
    }

    ctx.emit_error(diag);
}

/// Emit an error when a type family is not defined.
pub fn emit_type_family_not_found(ctx: &mut TyCtxt, name: Symbol, span: Span) {
    let name_str = name.as_str();

    let diag = Diagnostic::error(format!(
        "type family `{name_str}` not found"
    ))
    .with_code("E0042")
    .with_label(ctx.full_span(span), "unknown type family")
    .with_note(
        "Type families must be declared in a class definition or as standalone type family declarations."
    );

    ctx.emit_error(diag);
}

/// Emit an error when an associated type is missing from an instance.
///
/// Called when an instance doesn't provide an implementation for an
/// associated type and the class has no default.
pub fn emit_missing_assoc_type_impl(
    ctx: &mut TyCtxt,
    class_name: Symbol,
    assoc_type: Symbol,
    instance_types: &[Ty],
    span: Span,
) {
    let class_str = class_name.as_str();
    let assoc_str = assoc_type.as_str();
    let types_str: Vec<_> = instance_types.iter().map(pretty_ty).collect();
    let instance_str = format!("{} {}", class_str, types_str.join(" "));

    let diag = Diagnostic::error(format!(
        "missing associated type `{assoc_str}` in instance `{instance_str}`"
    ))
    .with_code("E0043")
    .with_label(
        ctx.full_span(span),
        format!("missing `type {assoc_str} = ...`"),
    )
    .with_note(format!(
        "The class `{class_str}` requires an implementation for associated type `{assoc_str}`,\n\
         but this instance doesn't provide one and the class has no default.\n\n\
         Add a type definition:\n    type {assoc_str} ... = <your type>"
    ));

    ctx.emit_error(diag);
}

/// Emit an error when a type family instance overlaps with another.
#[allow(dead_code)]
pub fn emit_type_family_overlap(ctx: &mut TyCtxt, family_name: Symbol, args: &[Ty], span: Span) {
    let family_str = family_name.as_str();
    let args_str: Vec<_> = args.iter().map(pretty_ty).collect();
    let applied_str = if args_str.is_empty() {
        family_str.to_string()
    } else {
        format!("{} {}", family_str, args_str.join(" "))
    };

    let diag = Diagnostic::error(format!(
        "overlapping type family instances for `{applied_str}`"
    ))
    .with_code("E0044")
    .with_label(ctx.full_span(span), "overlapping instance")
    .with_note(
        "Multiple type family instances match these arguments.\n\
         Type family instances must not overlap.",
    );

    ctx.emit_error(diag);
}

// === M10 Phase 2: Function arity errors (E0008-E0010) ===

/// Emit a function arity mismatch error.
///
/// Called when a function is applied to the wrong number of arguments.
pub fn emit_function_arity_mismatch(
    ctx: &mut TyCtxt,
    function_name: &str,
    expected: usize,
    found: usize,
    span: Span,
    arg_spans: &[Span],
) {
    let plural = |n: usize| if n == 1 { "argument" } else { "arguments" };

    let mut diag = Diagnostic::error(format!(
        "function `{function_name}` takes {expected} {}, but {found} {} supplied",
        plural(expected),
        if found == 1 { "was" } else { "were" }
    ))
    .with_code("E0008")
    .with_label(
        ctx.full_span(span),
        format!("expected {expected} {}", plural(expected)),
    );

    // Highlight extra arguments
    if found > expected {
        for (i, &arg_span) in arg_spans.iter().enumerate().skip(expected) {
            if !arg_span.is_dummy() {
                diag = diag.with_secondary_label(
                    ctx.full_span(arg_span),
                    format!("unexpected {} argument", ordinal(i + 1)),
                );
            }
        }
    }

    ctx.emit_error(diag);
}

/// Emit an error when a non-function is applied to arguments.
pub fn emit_not_a_function(ctx: &mut TyCtxt, actual_type: &Ty, num_args: usize, span: Span) {
    let ty_str = pretty_ty(actual_type);
    let plural = if num_args == 1 {
        "argument"
    } else {
        "arguments"
    };

    let diag = Diagnostic::error(format!("expected function, found `{ty_str}`"))
        .with_code("E0009")
        .with_label(
            ctx.full_span(span),
            format!("cannot apply {num_args} {plural} to non-function"),
        )
        .with_note(format!(
            "the expression has type `{ty_str}`, which is not a function type"
        ));

    ctx.emit_error(diag);
}

/// Emit an error when too few arguments are provided (partial application context).
#[allow(dead_code)]
pub fn emit_partial_application_hint(
    ctx: &mut TyCtxt,
    function_name: &str,
    expected: usize,
    found: usize,
    result_type: &Ty,
    span: Span,
) {
    let plural = |n: usize| if n == 1 { "argument" } else { "arguments" };
    let remaining = expected - found;

    let diag = Diagnostic::warning(format!(
        "partial application: `{function_name}` expects {remaining} more {}",
        plural(remaining)
    ))
    .with_label(
        ctx.full_span(span),
        format!(
            "this returns `{}`, not a fully applied result",
            pretty_ty(result_type)
        ),
    )
    .with_note("partial application is valid, but this may not be what you intended");

    ctx.emit_error(diag);
}

/// Helper to format ordinal numbers.
fn ordinal(n: usize) -> String {
    match n {
        1 => "1st".to_string(),
        2 => "2nd".to_string(),
        3 => "3rd".to_string(),
        n => format!("{n}th"),
    }
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

// === M9 Dependent Types: Shape-indexed tensor diagnostics ===

/// Emit a dimension mismatch error for type-level naturals.
pub fn emit_dimension_mismatch(ctx: &mut TyCtxt, expected: u64, found: u64, span: Span) {
    let diag = Diagnostic::error(format!(
        "dimension mismatch: expected `{expected}`, found `{found}`"
    ))
    .with_code("E0020")
    .with_label(ctx.full_span(span), "incompatible tensor dimensions")
    .with_note("Tensor dimensions must match exactly for this operation");

    ctx.emit_error(diag);
}

/// Emit a type-level natural unification failure.
pub fn emit_nat_mismatch(ctx: &mut TyCtxt, n1: &TyNat, n2: &TyNat, span: Span) {
    let diag = Diagnostic::error(format!(
        "cannot unify type-level naturals: `{}` vs `{}`",
        pretty_nat(n1),
        pretty_nat(n2)
    ))
    .with_code("E0021")
    .with_label(ctx.full_span(span), "dimension constraint violation")
    .with_note("These dimensions cannot be proven equal");

    ctx.emit_error(diag);
}

/// Emit an occurs check error for type-level naturals.
pub fn emit_nat_occurs_check_error(ctx: &mut TyCtxt, var: &TyVar, n: &TyNat, span: Span) {
    let diag = Diagnostic::error(format!(
        "infinite dimension: variable `{}` occurs in `{}`",
        pretty_tyvar(var),
        pretty_nat(n)
    ))
    .with_code("E0022")
    .with_label(ctx.full_span(span), "circular dimension constraint");

    ctx.emit_error(diag);
}

/// Emit a shape length mismatch error.
///
/// ## M10 Phase 3: Visual Rank Comparison
///
/// Includes a visual diagram showing the rank difference.
pub fn emit_shape_length_mismatch(ctx: &mut TyCtxt, s1: &TyList, s2: &TyList, span: Span) {
    let len1 = s1.static_len().map_or(0, |n| n);
    let len2 = s2.static_len().map_or(0, |n| n);

    let len1_str = s1.static_len().map_or("?".to_string(), |n| n.to_string());
    let len2_str = s2.static_len().map_or("?".to_string(), |n| n.to_string());

    // Generate visual diagram
    let diagram = shape_diagrams::format_rank_mismatch(len1, len2, s1, s2);

    let diag = Diagnostic::error(format!(
        "shape rank mismatch: expected rank {len1_str}, found rank {len2_str}"
    ))
    .with_code("E0023")
    .with_label(ctx.full_span(span), "incompatible tensor ranks")
    .with_note(diagram);

    ctx.emit_error(diag);
}

/// Emit a shape mismatch error for type-level lists.
///
/// ## M10 Phase 3: Unification Trace
///
/// Includes a step-by-step trace of shape unification to help
/// users understand where the mismatch occurred.
pub fn emit_shape_mismatch(ctx: &mut TyCtxt, s1: &TyList, s2: &TyList, span: Span) {
    // Generate unification trace
    let trace = shape_diagrams::format_unification_trace(s1, s2, &[]);

    let diag = Diagnostic::error(format!(
        "shape mismatch: `{}` vs `{}`",
        pretty_ty_list(s1),
        pretty_ty_list(s2)
    ))
    .with_code("E0024")
    .with_label(ctx.full_span(span), "incompatible tensor shapes")
    .with_note(trace)
    .with_note("Ensure tensor dimensions are compatible for this operation");

    ctx.emit_error(diag);
}

/// Emit an occurs check error for type-level lists.
pub fn emit_ty_list_occurs_check_error(ctx: &mut TyCtxt, var: &TyVar, l: &TyList, span: Span) {
    let diag = Diagnostic::error(format!(
        "infinite shape: variable `{}` occurs in `{}`",
        pretty_tyvar(var),
        pretty_ty_list(l)
    ))
    .with_code("E0025")
    .with_label(ctx.full_span(span), "circular shape constraint");

    ctx.emit_error(diag);
}

/// Emit an error from the nat constraint solver.
pub fn emit_nat_solver_error(ctx: &mut TyCtxt, err: &crate::nat_solver::SolverError, span: Span) {
    use crate::nat_solver::SolverError;

    let (message, note) = match err {
        SolverError::LiteralMismatch { expected, found } => (
            format!("dimension mismatch: expected `{expected}`, found `{found}`"),
            Some("Tensor dimensions must match exactly".to_string()),
        ),
        SolverError::OccursCheck { var_id, term } => (
            format!(
                "infinite dimension: variable `n{}` occurs in `{}`",
                var_id,
                pretty_nat(term)
            ),
            Some("This would create an infinitely recursive dimension".to_string()),
        ),
        SolverError::Inconsistent { message } => (
            format!("inconsistent dimension constraint: {message}"),
            Some("These dimension constraints cannot all be satisfied".to_string()),
        ),
        SolverError::CannotSolve { constraint } => {
            let crate::nat_solver::NatConstraint::Equal(n1, n2) = constraint;
            (
                format!(
                    "cannot prove dimension equality: `{}` = `{}`",
                    pretty_nat(n1),
                    pretty_nat(n2)
                ),
                Some("Consider adding a type annotation or ensuring dimensions match".to_string()),
            )
        }
    };

    let mut diag = Diagnostic::error(message)
        .with_code("E0026")
        .with_label(ctx.full_span(span), "dimension constraint failure");

    if let Some(note_text) = note {
        diag = diag.with_note(note_text);
    }

    ctx.emit_error(diag);
}

// === M9 Phase 7: Enhanced shape error messages ===

/// Emit a matrix multiplication dimension mismatch error.
///
/// This provides detailed information about which dimensions don't match
/// with a visual ASCII diagram showing the shape mismatch.
///
/// ## M10 Phase 3: Visual Shape Diagrams
///
/// Includes an ASCII art diagram showing the matrices and highlighting
/// the mismatched inner dimensions.
pub fn emit_matmul_dimension_mismatch(
    ctx: &mut TyCtxt,
    left_shape: &TyList,
    right_shape: &TyList,
    inner_left: &TyNat,
    inner_right: &TyNat,
    span: Span,
) {
    // Generate visual diagram
    let diagram =
        shape_diagrams::format_matmul_diagram(left_shape, right_shape, inner_left, inner_right);

    let diag = Diagnostic::error(format!(
        "matrix multiplication dimension mismatch: inner dimensions {} and {} are not equal",
        pretty_nat(inner_left),
        pretty_nat(inner_right)
    ))
    .with_code("E0030")
    .with_label(ctx.full_span(span), "matmul dimension error")
    .with_note(diagram)
    .with_note(
        "Matrix multiplication requires: Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a",
    );

    ctx.emit_error(diag);
}

/// Emit a broadcast shape incompatibility error.
///
/// ## M10 Phase 3: Axis-by-Axis Breakdown
///
/// Includes a visual diagram showing compatibility at each axis,
/// with the failing axis clearly marked.
pub fn emit_broadcast_incompatible(
    ctx: &mut TyCtxt,
    shape1: &TyList,
    shape2: &TyList,
    axis: usize,
    dim1: &TyNat,
    dim2: &TyNat,
    span: Span,
) {
    // Generate visual diagram with axis-by-axis breakdown
    let diagram = shape_diagrams::format_broadcast_diagram(shape1, shape2, axis, dim1, dim2);

    let diag = Diagnostic::error(format!(
        "shapes cannot be broadcast together: {} and {}",
        pretty_ty_list(shape1),
        pretty_ty_list(shape2)
    ))
    .with_code("E0031")
    .with_label(
        ctx.full_span(span),
        format!("broadcast error at axis {axis}"),
    )
    .with_note(diagram);

    ctx.emit_error(diag);
}

/// Emit a transpose axis out of bounds error.
pub fn emit_transpose_axis_error(ctx: &mut TyCtxt, rank: usize, invalid_axis: usize, span: Span) {
    let diag = Diagnostic::error(format!(
        "transpose axis {invalid_axis} is out of bounds for tensor of rank {rank}"
    ))
    .with_code("E0032")
    .with_label(ctx.full_span(span), "invalid transpose axis")
    .with_note(format!(
        "Valid axis values are 0 to {} for a {rank}-dimensional tensor",
        rank.saturating_sub(1)
    ));

    ctx.emit_error(diag);
}

/// Emit a concatenation shape mismatch error.
pub fn emit_concat_shape_mismatch(
    ctx: &mut TyCtxt,
    axis: usize,
    expected_shape: &TyList,
    found_shape: &TyList,
    tensor_index: usize,
    span: Span,
) {
    let diag = Diagnostic::error(format!(
        "cannot concatenate tensors with incompatible shapes along axis {axis}"
    ))
    .with_code("E0033")
    .with_label(ctx.full_span(span), "concat shape mismatch")
    .with_note(format!(
        "Tensor {} has shape {} but expected shape compatible with {}\n\
         All tensors must have matching dimensions except along the concatenation axis",
        tensor_index,
        pretty_ty_list(found_shape),
        pretty_ty_list(expected_shape)
    ));

    ctx.emit_error(diag);
}

/// Emit an error when a specific dimension value is required.
pub fn emit_dimension_must_be(
    ctx: &mut TyCtxt,
    operation: &str,
    axis: usize,
    expected: u64,
    found: &TyNat,
    span: Span,
) {
    let diag = Diagnostic::error(format!(
        "{operation} requires dimension {axis} to be {expected}, but found {}",
        pretty_nat(found)
    ))
    .with_code("E0034")
    .with_label(
        ctx.full_span(span),
        format!("expected dimension {expected}"),
    );

    ctx.emit_error(diag);
}

/// Emit an error when a tensor doesn't have the expected rank.
///
/// ## M10 Phase 3: Visual Rank Comparison
///
/// Includes a visual representation of expected vs found dimensions.
pub fn emit_wrong_tensor_rank(
    ctx: &mut TyCtxt,
    operation: &str,
    expected_rank: usize,
    found_rank: usize,
    expected_shape: Option<&TyList>,
    found_shape: Option<&TyList>,
    span: Span,
) {
    let rank_desc = match expected_rank {
        0 => "scalar".to_string(),
        1 => "vector".to_string(),
        2 => "matrix".to_string(),
        n => format!("{n}-dimensional tensor"),
    };

    let found_desc = match found_rank {
        0 => "scalar".to_string(),
        1 => "vector".to_string(),
        2 => "matrix".to_string(),
        n => format!("{n}-dimensional tensor"),
    };

    let mut diag = Diagnostic::error(format!(
        "{operation} expects a {rank_desc} (rank {expected_rank}), but got a {found_desc} (rank {found_rank})"
    ))
    .with_code("E0035")
    .with_label(ctx.full_span(span), format!("expected rank {expected_rank}"));

    // Add visual diagram if shapes are available
    if let (Some(exp), Some(fnd)) = (expected_shape, found_shape) {
        let diagram = shape_diagrams::format_rank_mismatch(expected_rank, found_rank, exp, fnd);
        diag = diag.with_note(diagram);
    }

    ctx.emit_error(diag);
}

/// Emit a shape mismatch error with detailed axis information.
pub fn emit_shape_axis_mismatch(
    ctx: &mut TyCtxt,
    shape1: &TyList,
    shape2: &TyList,
    axis: usize,
    dim1: &TyNat,
    dim2: &TyNat,
    span: Span,
) {
    let diag = Diagnostic::error(format!(
        "shape mismatch at axis {axis}: expected {}, found {}",
        pretty_nat(dim1),
        pretty_nat(dim2)
    ))
    .with_code("E0036")
    .with_label(
        ctx.full_span(span),
        format!("dimension mismatch at axis {axis}"),
    )
    .with_note(format!(
        "Full shapes:\n\
         • Expected: {}\n\
         • Found:    {}",
        pretty_ty_list(shape1),
        pretty_ty_list(shape2)
    ));

    ctx.emit_error(diag);
}

/// Emit an error when DynTensor conversion fails at runtime.
pub fn emit_dyn_tensor_conversion_failed(
    ctx: &mut TyCtxt,
    expected_shape: &TyList,
    actual_dims: &[u64],
    span: Span,
) {
    let actual_str = format!(
        "[{}]",
        actual_dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let diag = Diagnostic::error(format!(
        "dynamic tensor shape {} does not match expected static shape {}",
        actual_str,
        pretty_ty_list(expected_shape)
    ))
    .with_code("E0037")
    .with_label(ctx.full_span(span), "fromDynamic failed")
    .with_note(
        "Use pattern matching with `fromDynamic` to handle shape mismatches:\n\
                case fromDynamic witness dynTensor of\n\
                  Just tensor -> ...\n\
                  Nothing -> handle mismatch",
    );

    ctx.emit_error(diag);
}

/// Emit a suggestion when shapes might be transposed.
///
/// ## M10 Phase 3: Visual Transpose Suggestion
///
/// Shows a visual diagram of the transpose operation that would fix the error.
pub fn emit_possible_transpose_suggestion(
    ctx: &mut TyCtxt,
    found: &TyList,
    expected: &TyList,
    span: Span,
) {
    // Check if shapes are compatible when transposed
    let found_dims = found.to_static_dims();
    let expected_dims = expected.to_static_dims();

    if let (Some(f), Some(e)) = (found_dims, expected_dims) {
        if f.len() == 2 && e.len() == 2 && f[0] == e[1] && f[1] == e[0] {
            // Generate visual transpose suggestion
            let diagram = shape_diagrams::format_transpose_suggestion(found, expected);

            let diag = Diagnostic::error(format!(
                "shape mismatch: expected {}, found {}",
                pretty_ty_list(expected),
                pretty_ty_list(found)
            ))
            .with_code("E0038")
            .with_label(ctx.full_span(span), "shapes appear transposed")
            .with_note(diagram);

            ctx.emit_error(diag);
            return;
        }
    }

    // Fall back to regular shape mismatch
    emit_shape_mismatch(ctx, expected, found, span);
}

/// Emit a helpful error when matmul arguments might be swapped.
pub fn emit_matmul_swap_suggestion(
    ctx: &mut TyCtxt,
    left_shape: &TyList,
    right_shape: &TyList,
    span: Span,
) {
    let left_dims = left_shape.to_static_dims();
    let right_dims = right_shape.to_static_dims();

    if let (Some(l), Some(r)) = (left_dims, right_dims) {
        if l.len() == 2 && r.len() == 2 {
            // Check if swapping would work
            if l[0] == r[1] {
                let diag = Diagnostic::error(format!(
                    "matrix multiplication shape mismatch: {} and {}",
                    pretty_ty_list(left_shape),
                    pretty_ty_list(right_shape)
                ))
                .with_code("E0039")
                .with_label(ctx.full_span(span), "matmul arguments may be swapped")
                .with_note(format!(
                    "Did you mean to swap the arguments?\n\
                     • Current:  matmul {} {}\n\
                     • Suggested: matmul {} {}",
                    pretty_ty_list(left_shape),
                    pretty_ty_list(right_shape),
                    pretty_ty_list(right_shape),
                    pretty_ty_list(left_shape)
                ));

                ctx.emit_error(diag);
                return;
            }

            // Check if transpose would help
            if l[1] != r[0] && l[1] == r[1] {
                let diag = Diagnostic::error(format!(
                    "matrix multiplication shape mismatch: {} and {}",
                    pretty_ty_list(left_shape),
                    pretty_ty_list(right_shape)
                ))
                .with_code("E0039")
                .with_label(ctx.full_span(span), "inner dimensions don't match")
                .with_note(format!(
                    "Did you mean to transpose the right matrix?\n\
                     • Current:  matmul {} {} -- inner dimensions: {} vs {}\n\
                     • Try: matmul {} (transpose {}) -- inner dimensions would match",
                    pretty_ty_list(left_shape),
                    pretty_ty_list(right_shape),
                    l[1],
                    r[0],
                    pretty_ty_list(left_shape),
                    pretty_ty_list(right_shape)
                ));

                ctx.emit_error(diag);
                return;
            }
        }
    }

    // Fall back to basic matmul error
    emit_shape_mismatch(ctx, left_shape, right_shape, span);
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
        // M9: Type-level naturals and lists
        Ty::Nat(n) => pretty_nat(n),
        Ty::TyList(l) => pretty_ty_list(l),
    }
}

/// Pretty-print a type-level natural.
fn pretty_nat(n: &TyNat) -> String {
    match n {
        TyNat::Lit(v) => v.to_string(),
        TyNat::Var(v) => pretty_tyvar(v),
        TyNat::Add(a, b) => format!("({} + {})", pretty_nat(a), pretty_nat(b)),
        TyNat::Mul(a, b) => format!("({} * {})", pretty_nat(a), pretty_nat(b)),
    }
}

/// Pretty-print a type-level list.
fn pretty_ty_list(l: &TyList) -> String {
    match l {
        TyList::Nil => "'[]".to_string(),
        TyList::Var(v) => pretty_tyvar(v),
        TyList::Cons(_, _) => {
            let mut parts = Vec::new();
            let mut current = l;
            while let TyList::Cons(head, tail) = current {
                parts.push(pretty_ty(head));
                current = tail;
            }
            if !matches!(current, TyList::Nil) {
                // Variable or append at the tail
                parts.push(format!("...{}", pretty_ty_list(current)));
            }
            format!("'[{}]", parts.join(", "))
        }
        TyList::Append(xs, ys) => {
            format!("({} ++ {})", pretty_ty_list(xs), pretty_ty_list(ys))
        }
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

    // === M9 Phase 7: Error message formatting tests ===

    #[test]
    fn test_pretty_nat_literal() {
        let n = TyNat::Lit(1024);
        assert_eq!(pretty_nat(&n), "1024");
    }

    #[test]
    fn test_pretty_nat_variable() {
        let var = TyVar::new(0, Kind::Nat);
        let n = TyNat::Var(var);
        assert_eq!(pretty_nat(&n), "a");
    }

    #[test]
    fn test_pretty_nat_arithmetic() {
        let m = TyNat::Var(TyVar::new(0, Kind::Nat));
        let k = TyNat::Var(TyVar::new(1, Kind::Nat));

        let add = TyNat::Add(Box::new(m.clone()), Box::new(k.clone()));
        assert_eq!(pretty_nat(&add), "(a + b)");

        let mul = TyNat::Mul(Box::new(m), Box::new(k));
        assert_eq!(pretty_nat(&mul), "(a * b)");
    }

    #[test]
    fn test_pretty_ty_list_nil() {
        let nil = TyList::Nil;
        assert_eq!(pretty_ty_list(&nil), "'[]");
    }

    #[test]
    fn test_pretty_ty_list_static() {
        let shape = TyList::shape_from_dims(&[1024, 768]);
        assert_eq!(pretty_ty_list(&shape), "'[1024, 768]");
    }

    #[test]
    fn test_pretty_ty_list_polymorphic() {
        let m = TyVar::new(0, Kind::Nat);
        let n = TyVar::new(1, Kind::Nat);
        let shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(m)), Ty::Nat(TyNat::Var(n))]);
        assert_eq!(pretty_ty_list(&shape), "'[a, b]");
    }

    #[test]
    fn test_pretty_ty_list_mixed() {
        let m = TyVar::new(0, Kind::Nat);
        let shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(m)), Ty::Nat(TyNat::Lit(784))]);
        assert_eq!(pretty_ty_list(&shape), "'[a, 784]");
    }

    #[test]
    fn test_pretty_ty_list_append() {
        let xs = TyList::shape_from_dims(&[1, 2]);
        let ys = TyList::shape_from_dims(&[3, 4]);
        let appended = TyList::Append(Box::new(xs), Box::new(ys));
        assert_eq!(pretty_ty_list(&appended), "('[1, 2] ++ '[3, 4])");
    }

    #[test]
    fn test_pretty_ty_nat_in_type() {
        let nat = Ty::Nat(TyNat::Lit(512));
        assert_eq!(pretty_ty(&nat), "512");
    }

    #[test]
    fn test_pretty_ty_ty_list_in_type() {
        let shape = TyList::shape_from_dims(&[32, 64]);
        let ty = Ty::TyList(shape);
        assert_eq!(pretty_ty(&ty), "'[32, 64]");
    }

    #[test]
    fn test_matmul_error_message_format() {
        // Verify the error message captures the essential information
        let left_shape = TyList::shape_from_dims(&[3, 5]);
        let right_shape = TyList::shape_from_dims(&[7, 4]);
        let inner_left = TyNat::Lit(5);
        let inner_right = TyNat::Lit(7);

        // Check that the pretty printing captures the dimension mismatch
        let left_str = pretty_ty_list(&left_shape);
        let right_str = pretty_ty_list(&right_shape);
        let inner_left_str = pretty_nat(&inner_left);
        let inner_right_str = pretty_nat(&inner_right);

        assert!(left_str.contains("3"));
        assert!(left_str.contains("5"));
        assert!(right_str.contains("7"));
        assert!(right_str.contains("4"));
        assert_eq!(inner_left_str, "5");
        assert_eq!(inner_right_str, "7");
    }

    #[test]
    fn test_broadcast_error_context() {
        // Test that we can capture axis-specific dimension info
        let shape1 = TyList::shape_from_dims(&[32, 64, 128]);
        let shape2 = TyList::shape_from_dims(&[32, 100, 128]);

        let s1_str = pretty_ty_list(&shape1);
        let s2_str = pretty_ty_list(&shape2);

        assert!(s1_str.contains("64"));
        assert!(s2_str.contains("100"));
    }

    #[test]
    fn test_tensor_type_display() {
        // Test displaying a full tensor type for error messages
        let shape = TyList::shape_from_dims(&[256, 128]);
        let float = Ty::Con(TyCon::new(Symbol::intern("Float32"), Kind::Star));

        // Build Tensor '[256, 128] Float32
        let tensor_con = TyCon::new(Symbol::intern("Tensor"), Kind::tensor_kind());
        let tensor_app = Ty::App(
            Box::new(Ty::App(
                Box::new(Ty::Con(tensor_con)),
                Box::new(Ty::TyList(shape)),
            )),
            Box::new(float),
        );

        let displayed = pretty_ty(&tensor_app);
        assert!(displayed.contains("Tensor"));
        assert!(displayed.contains("256"));
        assert!(displayed.contains("128"));
        assert!(displayed.contains("Float32"));
    }
}
