//! Visual shape diagrams for enhanced error messages.
//!
//! This module provides ASCII art diagrams for tensor shape errors,
//! making dimension mismatches visually clear at a glance.
//!
//! ## M10 Phase 3: Shape Error Excellence
//!
//! These diagrams follow Cargo-style error formatting with:
//! - Clear visual representation of tensor dimensions
//! - Highlighted mismatch points
//! - Step-by-step unification traces
//!
//! ## Example Output
//!
//! ```text
//! matmul dimension mismatch:
//!
//!   Left:   [3, 5]     Right: [7, 4]
//!           ───┬───           ───┬───
//!              │                 │
//!              └── k = 5        k = 7 ──┘
//!                     ↑           ↑
//!                     └───────────┘
//!                     dimensions must match
//! ```

use bhc_types::{TyList, TyNat, TyVar};

/// Format a matmul dimension mismatch with visual diagram.
///
/// Shows both matrices side-by-side with the mismatched inner dimensions
/// highlighted.
#[must_use]
pub fn format_matmul_diagram(
    left_shape: &TyList,
    right_shape: &TyList,
    inner_left: &TyNat,
    inner_right: &TyNat,
) -> String {
    let left_dims = shape_to_dims(left_shape);
    let right_dims = shape_to_dims(right_shape);

    let inner_left_str = format_nat(inner_left);
    let inner_right_str = format_nat(inner_right);

    // Build the visual diagram
    let mut lines = Vec::new();

    lines.push(String::new());
    lines.push("  Matrix multiplication shape mismatch:".to_string());
    lines.push(String::new());

    // Show matrix shapes
    let left_str = format!("[{}]", left_dims.join(", "));
    let right_str = format!("[{}]", right_dims.join(", "));

    lines.push(format!(
        "    Left matrix:  {}    Right matrix: {}",
        left_str, right_str
    ));
    lines.push(String::new());

    // Visual representation
    if left_dims.len() == 2 && right_dims.len() == 2 {
        // Standard 2D matrix multiplication
        let m = &left_dims[0];
        let k1 = &left_dims[1];
        let k2 = &right_dims[0];
        let n = &right_dims[1];

        lines.push(format!("    ┌─────────┐   ┌─────────┐   ┌─────────┐"));
        lines.push(format!("    │         │   │         │   │         │"));
        lines.push(format!(
            "    │  {m:^5}  │ × │  {k2:^5}  │ → │  {m:^5}  │",
            m = m,
            k2 = k2
        ));
        lines.push(format!("    │    ×    │   │    ×    │   │    ×    │"));
        lines.push(format!(
            "    │  {k1:^5}  │   │  {n:^5}  │   │  {n:^5}  │",
            k1 = k1,
            n = n
        ));
        lines.push(format!("    │         │   │         │   │         │"));
        lines.push(format!("    └─────────┘   └─────────┘   └─────────┘"));
        lines.push(String::new());
        lines.push(format!("         ↑              ↑"));
        lines.push(format!("         │              │"));
        lines.push(format!(
            "         └── k = {}    k = {} ──┘",
            inner_left_str, inner_right_str
        ));
        lines.push(format!("                  ↑"));
        lines.push(format!("        These dimensions must match!"));
    } else {
        // Non-standard shapes
        lines.push(format!("    Inner dimensions:"));
        lines.push(format!("      Left columns:  {}", inner_left_str));
        lines.push(format!("      Right rows:    {}", inner_right_str));
        lines.push(format!("                     ↑"));
        lines.push(format!("           Must be equal for matmul"));
    }

    lines.push(String::new());
    lines.push(format!(
        "  Hint: For A × B, columns of A must equal rows of B"
    ));

    lines.join("\n")
}

/// Format a broadcast error with axis-by-axis breakdown.
///
/// Shows which axes are compatible and which fail broadcasting rules.
#[must_use]
pub fn format_broadcast_diagram(
    shape1: &TyList,
    shape2: &TyList,
    failed_axis: usize,
    dim1: &TyNat,
    dim2: &TyNat,
) -> String {
    let dims1 = shape_to_dims(shape1);
    let dims2 = shape_to_dims(shape2);

    let mut lines = Vec::new();

    lines.push(String::new());
    lines.push("  Broadcasting shape mismatch:".to_string());
    lines.push(String::new());

    // Determine the longer shape for alignment
    let max_rank = dims1.len().max(dims2.len());

    // Pad shorter shape with "1" on the left (broadcasting semantics)
    let padded1: Vec<String> = std::iter::repeat("1".to_string())
        .take(max_rank.saturating_sub(dims1.len()))
        .chain(dims1.iter().cloned())
        .collect();
    let padded2: Vec<String> = std::iter::repeat("1".to_string())
        .take(max_rank.saturating_sub(dims2.len()))
        .chain(dims2.iter().cloned())
        .collect();

    // Calculate column widths
    let col_widths: Vec<usize> = (0..max_rank)
        .map(|i| padded1[i].len().max(padded2[i].len()).max(3))
        .collect();

    // Header row with axis numbers
    let header: String = col_widths
        .iter()
        .enumerate()
        .map(|(i, w)| format!("{:^width$}", format!("axis {}", i), width = w + 2))
        .collect::<Vec<_>>()
        .join(" ");
    lines.push(format!("    {}", header));

    // Separator
    let sep: String = col_widths
        .iter()
        .map(|w| "─".repeat(w + 2))
        .collect::<Vec<_>>()
        .join("─");
    lines.push(format!("    {}", sep));

    // Shape 1 row
    let row1: String = padded1
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let marker = if i == failed_axis { "►" } else { " " };
            format!("{}{:^width$}{}", marker, d, marker, width = col_widths[i])
        })
        .collect::<Vec<_>>()
        .join(" ");
    lines.push(format!("    {} (shape 1)", row1));

    // Shape 2 row
    let row2: String = padded2
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let marker = if i == failed_axis { "►" } else { " " };
            format!("{}{:^width$}{}", marker, d, marker, width = col_widths[i])
        })
        .collect::<Vec<_>>()
        .join(" ");
    lines.push(format!("    {} (shape 2)", row2));

    // Result row showing compatibility
    let result: String = (0..max_rank)
        .map(|i| {
            if i == failed_axis {
                format!("{:^width$}", "✗", width = col_widths[i] + 2)
            } else if padded1[i] == padded2[i] || padded1[i] == "1" || padded2[i] == "1" {
                format!("{:^width$}", "✓", width = col_widths[i] + 2)
            } else {
                format!("{:^width$}", "?", width = col_widths[i] + 2)
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    lines.push(format!("    {}", sep));
    lines.push(format!("    {} (compatible?)", result));

    lines.push(String::new());
    lines.push(format!(
        "  Error at axis {}: {} vs {}",
        failed_axis,
        format_nat(dim1),
        format_nat(dim2)
    ));
    lines.push(String::new());
    lines.push("  Broadcasting rules:".to_string());
    lines.push("    • Dimensions are compatible if they are equal".to_string());
    lines.push("    • OR if one of them is 1 (will be broadcast)".to_string());

    lines.join("\n")
}

/// Format a shape unification trace showing step-by-step matching.
///
/// Shows how the type checker tried to unify two shapes and where it failed.
#[must_use]
pub fn format_unification_trace(
    expected: &TyList,
    found: &TyList,
    unified_so_far: &[(String, String, bool)], // (expected_dim, found_dim, success)
) -> String {
    let mut lines = Vec::new();

    lines.push(String::new());
    lines.push("  Shape unification trace:".to_string());
    lines.push(String::new());

    let expected_dims = shape_to_dims(expected);
    let found_dims = shape_to_dims(found);

    lines.push(format!("    Expected: [{}]", expected_dims.join(", ")));
    lines.push(format!("    Found:    [{}]", found_dims.join(", ")));
    lines.push(String::new());

    if unified_so_far.is_empty() {
        // Show a simple comparison
        let max_rank = expected_dims.len().max(found_dims.len());
        for i in 0..max_rank {
            let e = expected_dims.get(i).map_or("_", String::as_str);
            let f = found_dims.get(i).map_or("_", String::as_str);
            let status = if e == f || e == "_" || f == "_" {
                "✓"
            } else {
                "✗"
            };
            lines.push(format!("    axis {}: {} vs {} → {}", i, e, f, status));
        }
    } else {
        // Show the trace
        lines.push("    Unification steps:".to_string());
        for (i, (exp, fnd, success)) in unified_so_far.iter().enumerate() {
            let status = if *success { "✓" } else { "✗ FAILED" };
            lines.push(format!("      {}. {} ~ {} → {}", i + 1, exp, fnd, status));
        }
    }

    lines.join("\n")
}

/// Format a rank mismatch error with visual comparison.
#[must_use]
pub fn format_rank_mismatch(
    expected_rank: usize,
    found_rank: usize,
    expected_shape: &TyList,
    found_shape: &TyList,
) -> String {
    let mut lines = Vec::new();

    let expected_dims = shape_to_dims(expected_shape);
    let found_dims = shape_to_dims(found_shape);

    lines.push(String::new());
    lines.push("  Tensor rank mismatch:".to_string());
    lines.push(String::new());

    // Visual representation of dimensions
    let exp_boxes = (0..expected_rank)
        .map(|_| "□")
        .collect::<Vec<_>>()
        .join(" ");
    let fnd_boxes = (0..found_rank).map(|_| "■").collect::<Vec<_>>().join(" ");

    lines.push(format!(
        "    Expected {} dimensions: {} = [{}]",
        expected_rank,
        exp_boxes,
        expected_dims.join(", ")
    ));
    lines.push(format!(
        "    Found {} dimensions:    {} = [{}]",
        found_rank,
        fnd_boxes,
        found_dims.join(", ")
    ));
    lines.push(String::new());

    // Describe the tensor type
    let expected_desc = rank_description(expected_rank);
    let found_desc = rank_description(found_rank);

    lines.push(format!(
        "    Expected a {} but got a {}",
        expected_desc, found_desc
    ));

    lines.join("\n")
}

/// Format a visual diagram for transpose suggestion.
#[must_use]
pub fn format_transpose_suggestion(found_shape: &TyList, expected_shape: &TyList) -> String {
    let found_dims = shape_to_dims(found_shape);
    let expected_dims = shape_to_dims(expected_shape);

    let mut lines = Vec::new();

    lines.push(String::new());
    lines.push("  Shape appears to be transposed:".to_string());
    lines.push(String::new());

    if found_dims.len() == 2 && expected_dims.len() == 2 {
        // Show 2D transpose visually
        let (m, n) = (&found_dims[0], &found_dims[1]);
        let (p, q) = (&expected_dims[0], &expected_dims[1]);

        lines.push(format!(
            "    Found:     [{} × {}]     Expected: [{} × {}]",
            m, n, p, q
        ));
        lines.push(String::new());
        lines.push(format!("    ┌─────────┐              ┌─────────┐"));
        lines.push(format!(
            "    │  {} × {} │   transpose  │  {} × {} │",
            m, n, n, m
        ));
        lines.push(format!("    └─────────┘      →       └─────────┘"));
        lines.push(String::new());
        lines.push("  Hint: Try using `transpose` on the tensor".to_string());
    } else {
        lines.push(format!("    Found:    [{}]", found_dims.join(", ")));
        lines.push(format!("    Expected: [{}]", expected_dims.join(", ")));
    }

    lines.join("\n")
}

/// Format element count mismatch for reshape errors.
#[must_use]
pub fn format_reshape_element_mismatch(from_shape: &TyList, to_shape: &TyList) -> String {
    let from_dims = shape_to_dims(from_shape);
    let to_dims = shape_to_dims(to_shape);

    let from_count = try_compute_element_count(&from_dims);
    let to_count = try_compute_element_count(&to_dims);

    let mut lines = Vec::new();

    lines.push(String::new());
    lines.push("  Reshape element count mismatch:".to_string());
    lines.push(String::new());

    lines.push(format!("    From shape: [{}]", from_dims.join(", ")));
    if let Some(count) = from_count {
        lines.push(format!(
            "                = {} × {} ... = {} elements",
            from_dims.join(" × "),
            "",
            count
        ));
    }

    lines.push(String::new());

    lines.push(format!("    To shape:   [{}]", to_dims.join(", ")));
    if let Some(count) = to_count {
        lines.push(format!(
            "                = {} × {} ... = {} elements",
            to_dims.join(" × "),
            "",
            count
        ));
    }

    lines.push(String::new());
    lines.push("  Reshape requires the same total number of elements".to_string());

    lines.join("\n")
}

/// Format a concat shape mismatch error.
#[must_use]
pub fn format_concat_mismatch(axis: usize, shapes: &[TyList], mismatch_index: usize) -> String {
    let mut lines = Vec::new();

    lines.push(String::new());
    lines.push(format!("  Cannot concatenate tensors along axis {}:", axis));
    lines.push(String::new());

    for (i, shape) in shapes.iter().enumerate() {
        let dims = shape_to_dims(shape);
        let marker = if i == mismatch_index { "►" } else { " " };
        lines.push(format!(
            "    {} tensor {}: [{}]",
            marker,
            i,
            dims.join(", ")
        ));
    }

    lines.push(String::new());
    lines.push("  All tensors must have matching dimensions except".to_string());
    lines.push(format!("  at the concatenation axis (axis {})", axis));

    lines.join("\n")
}

// === Helper functions ===

/// Convert a TyList to a vector of dimension strings.
fn shape_to_dims(shape: &TyList) -> Vec<String> {
    match shape.to_vec() {
        Some(tys) => tys
            .iter()
            .map(|ty| match ty {
                bhc_types::Ty::Nat(n) => format_nat(n),
                _ => "?".to_string(),
            })
            .collect(),
        None => vec!["...".to_string()],
    }
}

/// Format a TyNat for display.
fn format_nat(n: &TyNat) -> String {
    match n {
        TyNat::Lit(v) => v.to_string(),
        TyNat::Var(v) => format_tyvar(v),
        TyNat::Add(a, b) => format!("({} + {})", format_nat(a), format_nat(b)),
        TyNat::Mul(a, b) => format!("({} * {})", format_nat(a), format_nat(b)),
    }
}

/// Format a type variable for display.
fn format_tyvar(v: &TyVar) -> String {
    if v.id < 26 {
        #[allow(clippy::cast_possible_truncation)]
        let c = (b'a' + v.id as u8) as char;
        c.to_string()
    } else {
        format!("n{}", v.id)
    }
}

/// Get human-readable description for tensor rank.
fn rank_description(rank: usize) -> &'static str {
    match rank {
        0 => "scalar (0D tensor)",
        1 => "vector (1D tensor)",
        2 => "matrix (2D tensor)",
        3 => "3D tensor",
        4 => "4D tensor",
        _ => "higher-dimensional tensor",
    }
}

/// Try to compute element count if all dimensions are static.
fn try_compute_element_count(dims: &[String]) -> Option<u64> {
    let mut count: u64 = 1;
    for dim in dims {
        let n: u64 = dim.parse().ok()?;
        count = count.checked_mul(n)?;
    }
    Some(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_diagram() {
        let left = TyList::shape_from_dims(&[3, 5]);
        let right = TyList::shape_from_dims(&[7, 4]);
        let k1 = TyNat::Lit(5);
        let k2 = TyNat::Lit(7);

        let diagram = format_matmul_diagram(&left, &right, &k1, &k2);

        assert!(diagram.contains("Matrix multiplication shape mismatch"));
        assert!(diagram.contains("k = 5"));
        assert!(diagram.contains("k = 7"));
        assert!(diagram.contains("must match")); // "These dimensions must match!"
    }

    #[test]
    fn test_broadcast_diagram() {
        let shape1 = TyList::shape_from_dims(&[32, 64, 128]);
        let shape2 = TyList::shape_from_dims(&[32, 100, 128]);
        let dim1 = TyNat::Lit(64);
        let dim2 = TyNat::Lit(100);

        let diagram = format_broadcast_diagram(&shape1, &shape2, 1, &dim1, &dim2);

        assert!(diagram.contains("Broadcasting shape mismatch"));
        assert!(diagram.contains("axis 1"));
        assert!(diagram.contains("64"));
        assert!(diagram.contains("100"));
        assert!(diagram.contains("✗")); // Failed axis marker
    }

    #[test]
    fn test_broadcast_diagram_different_ranks() {
        let shape1 = TyList::shape_from_dims(&[3, 4]);
        let shape2 = TyList::shape_from_dims(&[2, 3, 4]);
        let dim1 = TyNat::Lit(1); // Implicit dimension for shape1
        let dim2 = TyNat::Lit(2);

        let diagram = format_broadcast_diagram(&shape1, &shape2, 0, &dim1, &dim2);

        assert!(diagram.contains("Broadcasting shape mismatch"));
    }

    #[test]
    fn test_rank_mismatch_diagram() {
        let expected = TyList::shape_from_dims(&[10, 20]);
        let found = TyList::shape_from_dims(&[10, 20, 30]);

        let diagram = format_rank_mismatch(2, 3, &expected, &found);

        assert!(diagram.contains("Tensor rank mismatch"));
        assert!(diagram.contains("2 dimensions"));
        assert!(diagram.contains("3 dimensions"));
        assert!(diagram.contains("matrix"));
        assert!(diagram.contains("3D tensor"));
    }

    #[test]
    fn test_transpose_suggestion() {
        let found = TyList::shape_from_dims(&[768, 1024]);
        let expected = TyList::shape_from_dims(&[1024, 768]);

        let diagram = format_transpose_suggestion(&found, &expected);

        assert!(diagram.contains("transposed"));
        assert!(diagram.contains("768"));
        assert!(diagram.contains("1024"));
    }

    #[test]
    fn test_reshape_element_mismatch() {
        let from = TyList::shape_from_dims(&[10, 20]);
        let to = TyList::shape_from_dims(&[15, 15]);

        let diagram = format_reshape_element_mismatch(&from, &to);

        assert!(diagram.contains("Reshape element count mismatch"));
        assert!(diagram.contains("[10, 20]"));
        assert!(diagram.contains("[15, 15]"));
        assert!(diagram.contains("200 elements"));
        assert!(diagram.contains("225 elements"));
    }

    #[test]
    fn test_concat_mismatch() {
        let shapes = vec![
            TyList::shape_from_dims(&[10, 20]),
            TyList::shape_from_dims(&[10, 30]),
            TyList::shape_from_dims(&[15, 20]), // Mismatched at non-concat axis
        ];

        let diagram = format_concat_mismatch(1, &shapes, 2);

        assert!(diagram.contains("Cannot concatenate"));
        assert!(diagram.contains("axis 1"));
        assert!(diagram.contains("[15, 20]")); // The problematic shape
    }

    #[test]
    fn test_unification_trace() {
        let expected = TyList::shape_from_dims(&[32, 64]);
        let found = TyList::shape_from_dims(&[32, 128]);
        let trace = vec![
            ("32".to_string(), "32".to_string(), true),
            ("64".to_string(), "128".to_string(), false),
        ];

        let diagram = format_unification_trace(&expected, &found, &trace);

        assert!(diagram.contains("unification trace"));
        assert!(diagram.contains("32 ~ 32"));
        assert!(diagram.contains("64 ~ 128"));
        assert!(diagram.contains("FAILED"));
    }

    #[test]
    fn test_polymorphic_matmul_diagram() {
        use bhc_types::Kind;

        let m = TyVar::new(0, Kind::Nat);
        let k1 = TyVar::new(1, Kind::Nat);
        let k2_val = 512u64;
        let n = TyVar::new(2, Kind::Nat);

        let left = TyList::from_vec(vec![
            bhc_types::Ty::Nat(TyNat::Var(m)),
            bhc_types::Ty::Nat(TyNat::Var(k1.clone())),
        ]);
        let right = TyList::from_vec(vec![
            bhc_types::Ty::Nat(TyNat::Lit(k2_val)),
            bhc_types::Ty::Nat(TyNat::Var(n)),
        ]);

        let inner_left = TyNat::Var(k1);
        let inner_right = TyNat::Lit(k2_val);

        let diagram = format_matmul_diagram(&left, &right, &inner_left, &inner_right);

        assert!(diagram.contains("Matrix multiplication"));
        assert!(diagram.contains("b")); // k1 as variable 'b'
        assert!(diagram.contains("512"));
    }

    #[test]
    fn test_format_nat_arithmetic() {
        let n = TyNat::Var(TyVar::new(0, bhc_types::Kind::Nat));
        let k = TyNat::Lit(2);
        let mul = TyNat::Mul(Box::new(n), Box::new(k));

        let formatted = format_nat(&mul);
        assert_eq!(formatted, "(a * 2)");
    }

    #[test]
    fn test_rank_descriptions() {
        assert_eq!(rank_description(0), "scalar (0D tensor)");
        assert_eq!(rank_description(1), "vector (1D tensor)");
        assert_eq!(rank_description(2), "matrix (2D tensor)");
        assert_eq!(rank_description(5), "higher-dimensional tensor");
    }
}
