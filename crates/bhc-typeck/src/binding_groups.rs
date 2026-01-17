//! Binding group analysis via SCC decomposition.
//!
//! This module computes strongly connected components (SCCs) of the
//! dependency graph between top-level bindings. This is necessary
//! for correctly handling mutually recursive definitions.
//!
//! ## Why SCCs?
//!
//! In Hindley-Milner type inference, the order of type checking matters:
//!
//! - Non-recursive bindings can be generalized immediately
//! - Mutually recursive bindings must be typed together
//!
//! By computing SCCs, we identify groups of bindings that reference
//! each other and must be typed as a unit.
//!
//! ## Example
//!
//! ```text
//! -- These form an SCC
//! even n = if n == 0 then True else odd (n-1)
//! odd n = if n == 0 then False else even (n-1)
//!
//! -- This is non-recursive
//! double x = x + x
//! ```

use bhc_hir::{DefId, Expr, Item};
use petgraph::algo::kosaraju_scc;
use petgraph::graph::{DiGraph, NodeIndex};
use rustc_hash::FxHashMap;

/// A group of bindings to type check together.
#[derive(Debug, Clone)]
pub enum BindingGroup<'a> {
    /// A non-recursive binding.
    NonRecursive(&'a Item),
    /// A group of mutually recursive bindings.
    Recursive(Vec<&'a Item>),
}

/// Compute binding groups from a list of items.
///
/// Returns groups in dependency order (earlier groups don't depend on later).
pub fn compute_binding_groups(items: &[Item]) -> Vec<BindingGroup<'_>> {
    // Build mapping from DefId to item index
    let mut def_to_idx: FxHashMap<DefId, usize> = FxHashMap::default();
    for (idx, item) in items.iter().enumerate() {
        if let Some(def_id) = get_item_def_id(item) {
            def_to_idx.insert(def_id, idx);
        }
    }

    // Build dependency graph
    let mut graph: DiGraph<usize, ()> = DiGraph::new();
    let mut idx_to_node: FxHashMap<usize, NodeIndex> = FxHashMap::default();

    // Add nodes
    for idx in 0..items.len() {
        let node = graph.add_node(idx);
        idx_to_node.insert(idx, node);
    }

    // Add edges (item depends on item it references)
    for (idx, item) in items.iter().enumerate() {
        let refs = collect_references(item);
        let from_node = idx_to_node[&idx];

        for ref_def_id in refs {
            if let Some(&ref_idx) = def_to_idx.get(&ref_def_id) {
                if ref_idx != idx {
                    // Edge from referencer to referenced
                    let to_node = idx_to_node[&ref_idx];
                    graph.add_edge(from_node, to_node, ());
                }
            }
        }
    }

    // Compute SCCs (Kosaraju's algorithm returns in reverse topological order)
    let sccs = kosaraju_scc(&graph);

    // Convert SCCs to binding groups
    let mut groups = Vec::new();
    for scc in sccs.into_iter().rev() {
        if scc.len() == 1 {
            let idx = graph[scc[0]];
            let item = &items[idx];

            // Check if it's self-recursive
            let def_id = get_item_def_id(item);
            let refs = collect_references(item);
            let is_self_recursive = def_id.is_some_and(|d| refs.contains(&d));

            if is_self_recursive {
                groups.push(BindingGroup::Recursive(vec![item]));
            } else {
                groups.push(BindingGroup::NonRecursive(item));
            }
        } else {
            // Multiple items in SCC = mutually recursive
            let scc_items: Vec<&Item> = scc.iter().map(|&node| &items[graph[node]]).collect();
            groups.push(BindingGroup::Recursive(scc_items));
        }
    }

    groups
}

/// Get the `DefId` of an item if it has one.
const fn get_item_def_id(item: &Item) -> Option<DefId> {
    match item {
        Item::Value(v) => Some(v.id),
        Item::Data(d) => Some(d.id),
        Item::Newtype(n) => Some(n.id),
        Item::TypeAlias(t) => Some(t.id),
        Item::Class(c) => Some(c.id),
        Item::Instance(_) | Item::Fixity(_) | Item::Foreign(_) => None,
    }
}

/// Collect all `DefId`s referenced by an item.
fn collect_references(item: &Item) -> Vec<DefId> {
    let mut refs = Vec::new();

    match item {
        Item::Value(value_def) => {
            for eq in &value_def.equations {
                collect_expr_references(&eq.rhs, &mut refs);
                for guard in &eq.guards {
                    collect_expr_references(&guard.cond, &mut refs);
                }
            }
        }
        Item::Instance(inst) => {
            for method in &inst.methods {
                for eq in &method.equations {
                    collect_expr_references(&eq.rhs, &mut refs);
                }
            }
        }
        Item::Class(class) => {
            for default in &class.defaults {
                for eq in &default.equations {
                    collect_expr_references(&eq.rhs, &mut refs);
                }
            }
        }
        _ => {}
    }

    refs
}

/// Collect `DefId`s from an expression.
fn collect_expr_references(expr: &Expr, refs: &mut Vec<DefId>) {
    match expr {
        Expr::Var(def_ref) | Expr::Con(def_ref) => {
            refs.push(def_ref.def_id);
        }
        Expr::App(f, a, _) => {
            collect_expr_references(f, refs);
            collect_expr_references(a, refs);
        }
        Expr::Lam(_, body, _) => {
            collect_expr_references(body, refs);
        }
        Expr::Let(bindings, body, _) => {
            for binding in bindings {
                collect_expr_references(&binding.rhs, refs);
            }
            collect_expr_references(body, refs);
        }
        Expr::Case(scrut, alts, _) => {
            collect_expr_references(scrut, refs);
            for alt in alts {
                collect_expr_references(&alt.rhs, refs);
                for guard in &alt.guards {
                    collect_expr_references(&guard.cond, refs);
                }
            }
        }
        Expr::If(c, t, e, _) => {
            collect_expr_references(c, refs);
            collect_expr_references(t, refs);
            collect_expr_references(e, refs);
        }
        Expr::Tuple(elems, _) | Expr::List(elems, _) => {
            for elem in elems {
                collect_expr_references(elem, refs);
            }
        }
        Expr::Record(con_ref, fields, _) => {
            refs.push(con_ref.def_id);
            for field in fields {
                collect_expr_references(&field.value, refs);
            }
        }
        Expr::FieldAccess(record, _, _) => {
            collect_expr_references(record, refs);
        }
        Expr::RecordUpdate(record, fields, _) => {
            collect_expr_references(record, refs);
            for field in fields {
                collect_expr_references(&field.value, refs);
            }
        }
        Expr::Ann(inner, _, _) | Expr::TypeApp(inner, _, _) => {
            collect_expr_references(inner, refs);
        }
        Expr::Lit(_, _) | Expr::Error(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_hir::{DefRef, Equation, ValueDef};
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;

    fn make_value_def(id: usize, name: &str, refs: Vec<usize>) -> Item {
        let def_id = DefId::new(id);
        let rhs = if refs.is_empty() {
            Expr::Lit(bhc_hir::Lit::Int(0), Span::DUMMY)
        } else {
            // Create application chain referencing other defs
            let mut expr = Expr::Var(DefRef {
                def_id: DefId::new(refs[0]),
                span: Span::DUMMY,
            });
            for &r in &refs[1..] {
                expr = Expr::App(
                    Box::new(expr),
                    Box::new(Expr::Var(DefRef {
                        def_id: DefId::new(r),
                        span: Span::DUMMY,
                    })),
                    Span::DUMMY,
                );
            }
            expr
        };

        Item::Value(ValueDef {
            id: def_id,
            name: Symbol::intern(name),
            sig: None,
            equations: vec![Equation {
                pats: Vec::new(),
                guards: Vec::new(),
                rhs,
                span: Span::DUMMY,
            }],
            span: Span::DUMMY,
        })
    }

    #[test]
    fn test_non_recursive() {
        let items = vec![
            make_value_def(0, "a", vec![]),
            make_value_def(1, "b", vec![0]),
            make_value_def(2, "c", vec![1]),
        ];

        let groups = compute_binding_groups(&items);

        // Should be 3 non-recursive groups in dependency order
        assert_eq!(groups.len(), 3);
        assert!(matches!(groups[0], BindingGroup::NonRecursive(_)));
        assert!(matches!(groups[1], BindingGroup::NonRecursive(_)));
        assert!(matches!(groups[2], BindingGroup::NonRecursive(_)));
    }

    #[test]
    fn test_self_recursive() {
        let items = vec![make_value_def(0, "fac", vec![0])]; // References itself

        let groups = compute_binding_groups(&items);

        assert_eq!(groups.len(), 1);
        assert!(matches!(groups[0], BindingGroup::Recursive(_)));
    }

    #[test]
    fn test_mutually_recursive() {
        let items = vec![
            make_value_def(0, "even", vec![1]), // even calls odd
            make_value_def(1, "odd", vec![0]),  // odd calls even
        ];

        let groups = compute_binding_groups(&items);

        assert_eq!(groups.len(), 1);
        match &groups[0] {
            BindingGroup::Recursive(items) => {
                assert_eq!(items.len(), 2);
            }
            _ => panic!("expected recursive group"),
        }
    }
}
