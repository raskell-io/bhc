//! Automatic instance derivation for standard type classes.
//!
//! This module implements automatic derivation of type class instances
//! for data types. When a data type declaration includes a `deriving` clause,
//! we generate the appropriate instance implementations.
//!
//! ## Supported Classes
//!
//! - `Eq`: Structural equality comparison
//! - `Ord`: Structural ordering comparison
//! - `Show`: String representation
//!
//! ## How Derivation Works
//!
//! For a data type like:
//! ```text
//! data Color = Red | Green | Blue deriving (Eq, Ord)
//! ```
//!
//! We generate instance methods that compare constructors by their tags
//! and recursively compare fields.

use bhc_core::{self as core, Alt, AltCon, Bind, DataCon, Var, VarId};
use bhc_hir::{ConFields, DataDef, NewtypeDef};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Kind, Ty, TyCon, TyVar};
use rustc_hash::FxHashMap;

use crate::dictionary::InstanceInfo;

/// Result of deriving an instance.
pub struct DerivedInstance {
    /// The instance info to register.
    pub instance: InstanceInfo,
    /// The generated method bindings.
    pub bindings: Vec<Bind>,
}

/// Context for generating derived instances.
pub struct DerivingContext {
    /// Fresh variable counter.
    fresh_counter: u32,
}

impl DerivingContext {
    /// Create a new deriving context.
    pub fn new() -> Self {
        Self {
            fresh_counter: 50000, // Start above all fixed DefId ranges (highest is ~11273)
        }
    }

    /// Generate a fresh variable.
    fn fresh_var(&mut self, prefix: &str, ty: Ty) -> Var {
        let n = self.fresh_counter;
        self.fresh_counter += 1;
        let name = Symbol::intern(&format!("{}_{}", prefix, n));
        Var {
            name,
            id: VarId::new(n as usize),
            ty,
        }
    }

    /// Derive an instance for a data type.
    pub fn derive_for_data(
        &mut self,
        data_def: &DataDef,
        class_name: Symbol,
    ) -> Option<DerivedInstance> {
        let class_str = class_name.as_str();
        match class_str {
            "Eq" => self.derive_eq_data(data_def),
            "Ord" => self.derive_ord_data(data_def),
            "Show" => self.derive_show_data(data_def),
            "Enum" => self.derive_enum_data(data_def),
            "Bounded" => self.derive_bounded_data(data_def),
            "Functor" => self.derive_functor_data(data_def),
            "Foldable" => self.derive_foldable_data(data_def),
            "Traversable" => self.derive_traversable_data(data_def),
            _ => {
                // Unsupported class for derivation
                None
            }
        }
    }

    /// Derive an instance for a newtype.
    pub fn derive_for_newtype(
        &mut self,
        newtype_def: &NewtypeDef,
        class_name: Symbol,
    ) -> Option<DerivedInstance> {
        let class_str = class_name.as_str();
        match class_str {
            "Eq" => self.derive_eq_newtype(newtype_def),
            "Ord" => self.derive_ord_newtype(newtype_def),
            "Show" => self.derive_show_newtype(newtype_def),
            "Functor" => self.derive_functor_newtype(newtype_def),
            "Foldable" => self.derive_foldable_newtype(newtype_def),
            "Traversable" => self.derive_traversable_newtype(newtype_def),
            _ => None,
        }
    }

    // =========================================================================
    // Eq derivation
    // =========================================================================

    /// Derive Eq for a data type.
    fn derive_eq_data(&mut self, data_def: &DataDef) -> Option<DerivedInstance> {
        let span = data_def.span;

        // Build the instance type: the data type applied to its parameters
        let instance_type = self.build_instance_type(data_def.name, &data_def.params);

        // Generate the (==) method
        let eq_method_var = self.fresh_var(
            &format!("$derived_eq_{}", data_def.name.as_str()),
            Ty::Error,
        );
        let eq_body = self.generate_eq_body(data_def, span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("=="),
            bhc_hir::DefId::new(eq_method_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Eq"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        let bindings = vec![Bind::NonRec(eq_method_var, Box::new(eq_body))];

        Some(DerivedInstance { instance, bindings })
    }

    /// Generate the body of the (==) method for a data type.
    fn generate_eq_body(&mut self, data_def: &DataDef, span: Span) -> core::Expr {
        let x_var = self.fresh_var("x", Ty::Error);
        let y_var = self.fresh_var("y", Ty::Error);

        // Check if all constructors have no fields (simple enum)
        let is_simple_enum = data_def.cons.iter().all(|con| match &con.fields {
            ConFields::Positional(fields) => fields.is_empty(),
            ConFields::Named(fields) => fields.is_empty(),
        });

        let body = if is_simple_enum {
            self.generate_enum_eq(data_def, &x_var, &y_var, span)
        } else {
            self.generate_complex_eq(data_def, &x_var, &y_var, span)
        };

        // Wrap in lambdas: \x -> \y -> body
        let inner = core::Expr::Lam(y_var, Box::new(body), span);
        core::Expr::Lam(x_var, Box::new(inner), span)
    }

    /// Generate equality check for simple enums (no fields).
    fn generate_enum_eq(
        &mut self,
        data_def: &DataDef,
        x_var: &Var,
        y_var: &Var,
        span: Span,
    ) -> core::Expr {
        let true_expr = self.make_bool(true, span);
        let false_expr = self.make_bool(false, span);
        let type_con = TyCon::new(data_def.name, Kind::Star);

        let mut outer_alts = Vec::new();

        for (tag, con) in data_def.cons.iter().enumerate() {
            // Inner case for y: match same constructor -> True, default -> False
            let data_con = DataCon {
                name: con.name,
                ty_con: type_con.clone(),
                tag: tag as u32,
                arity: 0,
            };

            let inner_match = Alt {
                con: AltCon::DataCon(data_con.clone()),
                binders: vec![],
                rhs: true_expr.clone(),
            };
            let inner_default = Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: false_expr.clone(),
            };

            let inner_case = self.make_case(
                core::Expr::Var(y_var.clone(), span),
                vec![inner_match, inner_default],
                span,
            );

            outer_alts.push(Alt {
                con: AltCon::DataCon(data_con),
                binders: vec![],
                rhs: inner_case,
            });
        }

        self.make_case(core::Expr::Var(x_var.clone(), span), outer_alts, span)
    }

    /// Generate equality check for data types with fields.
    fn generate_complex_eq(
        &mut self,
        data_def: &DataDef,
        x_var: &Var,
        y_var: &Var,
        span: Span,
    ) -> core::Expr {
        let true_expr = self.make_bool(true, span);
        let false_expr = self.make_bool(false, span);
        let type_con = TyCon::new(data_def.name, Kind::Star);

        let mut outer_alts = Vec::new();

        for (tag, con) in data_def.cons.iter().enumerate() {
            let field_count = match &con.fields {
                ConFields::Positional(fields) => fields.len(),
                ConFields::Named(fields) => fields.len(),
            };

            let data_con = DataCon {
                name: con.name,
                ty_con: type_con.clone(),
                tag: tag as u32,
                arity: field_count as u32,
            };

            // Generate fresh variables for x's fields
            let x_fields: Vec<Var> = (0..field_count)
                .map(|i| self.fresh_var(&format!("x{}", i), Ty::Error))
                .collect();

            // Generate fresh variables for y's fields
            let y_fields: Vec<Var> = (0..field_count)
                .map(|i| self.fresh_var(&format!("y{}", i), Ty::Error))
                .collect();

            // Generate the comparison: x0 == y0 && x1 == y1 && ...
            let comparison = if field_count == 0 {
                true_expr.clone()
            } else {
                self.generate_field_comparisons(&x_fields, &y_fields, span)
            };

            // Inner case: match y with the same constructor
            let inner_match = Alt {
                con: AltCon::DataCon(data_con.clone()),
                binders: y_fields,
                rhs: comparison,
            };
            let inner_default = Alt {
                con: AltCon::Default,
                binders: vec![],
                rhs: false_expr.clone(),
            };

            let inner_case = self.make_case(
                core::Expr::Var(y_var.clone(), span),
                vec![inner_match, inner_default],
                span,
            );

            outer_alts.push(Alt {
                con: AltCon::DataCon(data_con),
                binders: x_fields,
                rhs: inner_case,
            });
        }

        self.make_case(core::Expr::Var(x_var.clone(), span), outer_alts, span)
    }

    /// Generate field-by-field comparisons: x0 == y0 && x1 == y1 && ...
    fn generate_field_comparisons(
        &mut self,
        x_fields: &[Var],
        y_fields: &[Var],
        span: Span,
    ) -> core::Expr {
        assert_eq!(x_fields.len(), y_fields.len());

        if x_fields.is_empty() {
            return self.make_bool(true, span);
        }

        // Start with the last comparison and work backwards with &&
        let mut result = self.make_eq_call(
            &x_fields[x_fields.len() - 1],
            &y_fields[y_fields.len() - 1],
            span,
        );

        for i in (0..x_fields.len() - 1).rev() {
            let eq_call = self.make_eq_call(&x_fields[i], &y_fields[i], span);
            result = self.make_and(eq_call, result, span);
        }

        result
    }

    /// Make a call to (==) for two variables.
    fn make_eq_call(&self, x: &Var, y: &Var, span: Span) -> core::Expr {
        let eq_var = Var {
            name: Symbol::intern("=="),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        let eq_expr = core::Expr::Var(eq_var, span);
        let app1 = core::Expr::App(
            Box::new(eq_expr),
            Box::new(core::Expr::Var(x.clone(), span)),
            span,
        );
        core::Expr::App(
            Box::new(app1),
            Box::new(core::Expr::Var(y.clone(), span)),
            span,
        )
    }

    /// Make a call to (&&) for two boolean expressions.
    fn make_and(&self, left: core::Expr, right: core::Expr, span: Span) -> core::Expr {
        let and_var = Var {
            name: Symbol::intern("&&"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        let and_expr = core::Expr::Var(and_var, span);
        let app1 = core::Expr::App(Box::new(and_expr), Box::new(left), span);
        core::Expr::App(Box::new(app1), Box::new(right), span)
    }

    /// Helper to create a Case expression with Error type.
    fn make_case(&self, scrutinee: core::Expr, alts: Vec<Alt>, span: Span) -> core::Expr {
        core::Expr::Case(Box::new(scrutinee), alts, Ty::Error, span)
    }

    /// Derive Eq for a newtype.
    fn derive_eq_newtype(&mut self, newtype_def: &NewtypeDef) -> Option<DerivedInstance> {
        let span = newtype_def.span;
        let instance_type = self.build_instance_type(newtype_def.name, &newtype_def.params);
        let type_con = TyCon::new(newtype_def.name, Kind::Star);

        let eq_method_var = self.fresh_var(
            &format!("$derived_eq_{}", newtype_def.name.as_str()),
            Ty::Error,
        );

        let x_var = self.fresh_var("x", Ty::Error);
        let y_var = self.fresh_var("y", Ty::Error);
        let inner_x = self.fresh_var("inner_x", Ty::Error);
        let inner_y = self.fresh_var("inner_y", Ty::Error);

        let data_con = DataCon {
            name: newtype_def.con.name,
            ty_con: type_con,
            tag: 0,
            arity: 1,
        };

        // case x of { Con inner_x -> case y of { Con inner_y -> inner_x == inner_y } }
        let eq_call = self.make_eq_call(&inner_x, &inner_y, span);

        let inner_case = self.make_case(
            core::Expr::Var(y_var.clone(), span),
            vec![Alt {
                con: AltCon::DataCon(data_con.clone()),
                binders: vec![inner_y],
                rhs: eq_call,
            }],
            span,
        );

        let outer_case = self.make_case(
            core::Expr::Var(x_var.clone(), span),
            vec![Alt {
                con: AltCon::DataCon(data_con),
                binders: vec![inner_x],
                rhs: inner_case,
            }],
            span,
        );

        let body = core::Expr::Lam(
            x_var,
            Box::new(core::Expr::Lam(y_var, Box::new(outer_case), span)),
            span,
        );

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("=="),
            bhc_hir::DefId::new(eq_method_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Eq"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(eq_method_var, Box::new(body))],
        })
    }

    // =========================================================================
    // Ord derivation
    // =========================================================================

    /// Derive Ord for a data type.
    fn derive_ord_data(&mut self, data_def: &DataDef) -> Option<DerivedInstance> {
        let span = data_def.span;
        let instance_type = self.build_instance_type(data_def.name, &data_def.params);

        let compare_method_var = self.fresh_var(
            &format!("$derived_compare_{}", data_def.name.as_str()),
            Ty::Error,
        );

        let compare_body = self.generate_compare_body(data_def, span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("compare"),
            bhc_hir::DefId::new(compare_method_var.id.index()),
        );

        // Ord requires Eq as superclass
        let instance = InstanceInfo {
            class: Symbol::intern("Ord"),
            instance_types: vec![instance_type.clone()],
            methods,
            superclass_instances: vec![instance_type],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(compare_method_var, Box::new(compare_body))],
        })
    }

    /// Generate the body of the compare method.
    fn generate_compare_body(&mut self, data_def: &DataDef, span: Span) -> core::Expr {
        let x_var = self.fresh_var("x", Ty::Error);
        let y_var = self.fresh_var("y", Ty::Error);

        let is_simple_enum = data_def.cons.iter().all(|con| match &con.fields {
            ConFields::Positional(fields) => fields.is_empty(),
            ConFields::Named(fields) => fields.is_empty(),
        });

        let body = if is_simple_enum {
            self.generate_enum_compare(data_def, &x_var, &y_var, span)
        } else {
            self.generate_complex_compare(data_def, &x_var, &y_var, span)
        };

        let inner = core::Expr::Lam(y_var, Box::new(body), span);
        core::Expr::Lam(x_var, Box::new(inner), span)
    }

    /// Generate compare for simple enums.
    fn generate_enum_compare(
        &mut self,
        data_def: &DataDef,
        x_var: &Var,
        y_var: &Var,
        span: Span,
    ) -> core::Expr {
        let eq_ordering = self.make_ordering("EQ", span);
        let lt_ordering = self.make_ordering("LT", span);
        let gt_ordering = self.make_ordering("GT", span);
        let type_con = TyCon::new(data_def.name, Kind::Star);

        let mut outer_alts = Vec::new();

        for (x_idx, x_con) in data_def.cons.iter().enumerate() {
            let x_data_con = DataCon {
                name: x_con.name,
                ty_con: type_con.clone(),
                tag: x_idx as u32,
                arity: 0,
            };

            let mut inner_alts = Vec::new();

            for (y_idx, y_con) in data_def.cons.iter().enumerate() {
                let y_data_con = DataCon {
                    name: y_con.name,
                    ty_con: type_con.clone(),
                    tag: y_idx as u32,
                    arity: 0,
                };

                let result = if x_idx < y_idx {
                    lt_ordering.clone()
                } else if x_idx > y_idx {
                    gt_ordering.clone()
                } else {
                    eq_ordering.clone()
                };

                inner_alts.push(Alt {
                    con: AltCon::DataCon(y_data_con),
                    binders: vec![],
                    rhs: result,
                });
            }

            let inner_case = self.make_case(core::Expr::Var(y_var.clone(), span), inner_alts, span);

            outer_alts.push(Alt {
                con: AltCon::DataCon(x_data_con),
                binders: vec![],
                rhs: inner_case,
            });
        }

        self.make_case(core::Expr::Var(x_var.clone(), span), outer_alts, span)
    }

    /// Generate compare for data types with fields.
    fn generate_complex_compare(
        &mut self,
        data_def: &DataDef,
        x_var: &Var,
        y_var: &Var,
        span: Span,
    ) -> core::Expr {
        let lt_ordering = self.make_ordering("LT", span);
        let gt_ordering = self.make_ordering("GT", span);
        let type_con = TyCon::new(data_def.name, Kind::Star);

        let mut outer_alts = Vec::new();

        for (x_idx, x_con) in data_def.cons.iter().enumerate() {
            let x_field_count = match &x_con.fields {
                ConFields::Positional(fields) => fields.len(),
                ConFields::Named(fields) => fields.len(),
            };

            let x_data_con = DataCon {
                name: x_con.name,
                ty_con: type_con.clone(),
                tag: x_idx as u32,
                arity: x_field_count as u32,
            };

            let x_fields: Vec<Var> = (0..x_field_count)
                .map(|i| self.fresh_var(&format!("x{}", i), Ty::Error))
                .collect();

            let mut inner_alts = Vec::new();

            for (y_idx, y_con) in data_def.cons.iter().enumerate() {
                let y_field_count = match &y_con.fields {
                    ConFields::Positional(fields) => fields.len(),
                    ConFields::Named(fields) => fields.len(),
                };

                let y_data_con = DataCon {
                    name: y_con.name,
                    ty_con: type_con.clone(),
                    tag: y_idx as u32,
                    arity: y_field_count as u32,
                };

                let y_fields: Vec<Var> = (0..y_field_count)
                    .map(|i| self.fresh_var(&format!("y{}", i), Ty::Error))
                    .collect();

                let result = if x_idx < y_idx {
                    lt_ordering.clone()
                } else if x_idx > y_idx {
                    gt_ordering.clone()
                } else {
                    // Same constructor, compare fields lexicographically
                    self.generate_field_comparisons_ord(&x_fields, &y_fields, span)
                };

                inner_alts.push(Alt {
                    con: AltCon::DataCon(y_data_con),
                    binders: y_fields,
                    rhs: result,
                });
            }

            let inner_case = self.make_case(core::Expr::Var(y_var.clone(), span), inner_alts, span);

            outer_alts.push(Alt {
                con: AltCon::DataCon(x_data_con),
                binders: x_fields,
                rhs: inner_case,
            });
        }

        self.make_case(core::Expr::Var(x_var.clone(), span), outer_alts, span)
    }

    /// Generate lexicographic field comparison for Ord.
    fn generate_field_comparisons_ord(
        &mut self,
        x_fields: &[Var],
        y_fields: &[Var],
        span: Span,
    ) -> core::Expr {
        if x_fields.is_empty() {
            return self.make_ordering("EQ", span);
        }

        let ordering_con = TyCon::new(Symbol::intern("Ordering"), Kind::Star);
        let lt_con = DataCon {
            name: Symbol::intern("LT"),
            ty_con: ordering_con.clone(),
            tag: 0,
            arity: 0,
        };
        let eq_con = DataCon {
            name: Symbol::intern("EQ"),
            ty_con: ordering_con.clone(),
            tag: 1,
            arity: 0,
        };
        let gt_con = DataCon {
            name: Symbol::intern("GT"),
            ty_con: ordering_con,
            tag: 2,
            arity: 0,
        };

        let mut result = self.make_ordering("EQ", span);

        for i in (0..x_fields.len()).rev() {
            let cmp_call = self.make_compare_call(&x_fields[i], &y_fields[i], span);

            result = self.make_case(
                cmp_call,
                vec![
                    Alt {
                        con: AltCon::DataCon(lt_con.clone()),
                        binders: vec![],
                        rhs: self.make_ordering("LT", span),
                    },
                    Alt {
                        con: AltCon::DataCon(gt_con.clone()),
                        binders: vec![],
                        rhs: self.make_ordering("GT", span),
                    },
                    Alt {
                        con: AltCon::DataCon(eq_con.clone()),
                        binders: vec![],
                        rhs: result,
                    },
                ],
                span,
            );
        }

        result
    }

    /// Make a call to compare for two variables.
    fn make_compare_call(&self, x: &Var, y: &Var, span: Span) -> core::Expr {
        let compare_var = Var {
            name: Symbol::intern("compare"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        let compare_expr = core::Expr::Var(compare_var, span);
        let app1 = core::Expr::App(
            Box::new(compare_expr),
            Box::new(core::Expr::Var(x.clone(), span)),
            span,
        );
        core::Expr::App(
            Box::new(app1),
            Box::new(core::Expr::Var(y.clone(), span)),
            span,
        )
    }

    /// Derive Ord for a newtype.
    fn derive_ord_newtype(&mut self, newtype_def: &NewtypeDef) -> Option<DerivedInstance> {
        let span = newtype_def.span;
        let instance_type = self.build_instance_type(newtype_def.name, &newtype_def.params);
        let type_con = TyCon::new(newtype_def.name, Kind::Star);

        let compare_method_var = self.fresh_var(
            &format!("$derived_compare_{}", newtype_def.name.as_str()),
            Ty::Error,
        );

        let x_var = self.fresh_var("x", Ty::Error);
        let y_var = self.fresh_var("y", Ty::Error);
        let inner_x = self.fresh_var("inner_x", Ty::Error);
        let inner_y = self.fresh_var("inner_y", Ty::Error);

        let data_con = DataCon {
            name: newtype_def.con.name,
            ty_con: type_con,
            tag: 0,
            arity: 1,
        };

        let compare_call = self.make_compare_call(&inner_x, &inner_y, span);

        let inner_case = self.make_case(
            core::Expr::Var(y_var.clone(), span),
            vec![Alt {
                con: AltCon::DataCon(data_con.clone()),
                binders: vec![inner_y],
                rhs: compare_call,
            }],
            span,
        );

        let outer_case = self.make_case(
            core::Expr::Var(x_var.clone(), span),
            vec![Alt {
                con: AltCon::DataCon(data_con),
                binders: vec![inner_x],
                rhs: inner_case,
            }],
            span,
        );

        let body = core::Expr::Lam(
            x_var,
            Box::new(core::Expr::Lam(y_var, Box::new(outer_case), span)),
            span,
        );

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("compare"),
            bhc_hir::DefId::new(compare_method_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Ord"),
            instance_types: vec![instance_type.clone()],
            methods,
            superclass_instances: vec![instance_type],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(compare_method_var, Box::new(body))],
        })
    }

    // =========================================================================
    // Show derivation
    // =========================================================================

    /// Derive Show for a data type.
    fn derive_show_data(&mut self, data_def: &DataDef) -> Option<DerivedInstance> {
        let span = data_def.span;
        let instance_type = self.build_instance_type(data_def.name, &data_def.params);

        let show_method_var = self.fresh_var(
            &format!("$derived_show_{}", data_def.name.as_str()),
            Ty::Error,
        );

        let show_body = self.generate_show_body(data_def, span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("show"),
            bhc_hir::DefId::new(show_method_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Show"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(show_method_var, Box::new(show_body))],
        })
    }

    /// Generate the body of the show method.
    fn generate_show_body(&mut self, data_def: &DataDef, span: Span) -> core::Expr {
        let x_var = self.fresh_var("x", Ty::Error);
        let type_con = TyCon::new(data_def.name, Kind::Star);

        let mut alts = Vec::new();

        for (tag, con) in data_def.cons.iter().enumerate() {
            let field_count = match &con.fields {
                ConFields::Positional(fields) => fields.len(),
                ConFields::Named(fields) => fields.len(),
            };

            let data_con = DataCon {
                name: con.name,
                ty_con: type_con.clone(),
                tag: tag as u32,
                arity: field_count as u32,
            };

            let fields: Vec<Var> = (0..field_count)
                .map(|i| self.fresh_var(&format!("f{}", i), Ty::Error))
                .collect();

            let show_expr = if field_count == 0 {
                // Just the constructor name
                self.make_string(con.name.as_str(), span)
            } else {
                // Constructor name followed by shown fields
                self.generate_show_with_fields(con.name, &fields, span)
            };

            alts.push(Alt {
                con: AltCon::DataCon(data_con),
                binders: fields,
                rhs: show_expr,
            });
        }

        let case_expr = self.make_case(core::Expr::Var(x_var.clone(), span), alts, span);

        core::Expr::Lam(x_var, Box::new(case_expr), span)
    }

    /// Generate show expression for constructor with fields.
    fn generate_show_with_fields(
        &mut self,
        con_name: Symbol,
        fields: &[Var],
        span: Span,
    ) -> core::Expr {
        // Generate: con_name ++ " " ++ show f0 ++ " " ++ show f1 ++ ...
        let mut result = self.make_string(con_name.as_str(), span);

        for field in fields {
            // Append " "
            result = self.make_append(result, self.make_string(" ", span), span);
            // Append show field
            let show_call = self.make_show_call(field, span);
            result = self.make_append(result, show_call, span);
        }

        result
    }

    /// Make a call to show for a variable.
    fn make_show_call(&self, x: &Var, span: Span) -> core::Expr {
        let show_var = Var {
            name: Symbol::intern("show"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        core::Expr::App(
            Box::new(core::Expr::Var(show_var, span)),
            Box::new(core::Expr::Var(x.clone(), span)),
            span,
        )
    }

    /// Make a string append operation (++).
    fn make_append(&self, left: core::Expr, right: core::Expr, span: Span) -> core::Expr {
        let append_var = Var {
            name: Symbol::intern("++"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        let append_expr = core::Expr::Var(append_var, span);
        let app1 = core::Expr::App(Box::new(append_expr), Box::new(left), span);
        core::Expr::App(Box::new(app1), Box::new(right), span)
    }

    /// Derive Show for a newtype.
    fn derive_show_newtype(&mut self, newtype_def: &NewtypeDef) -> Option<DerivedInstance> {
        let span = newtype_def.span;
        let instance_type = self.build_instance_type(newtype_def.name, &newtype_def.params);
        let type_con = TyCon::new(newtype_def.name, Kind::Star);

        let show_method_var = self.fresh_var(
            &format!("$derived_show_{}", newtype_def.name.as_str()),
            Ty::Error,
        );

        let x_var = self.fresh_var("x", Ty::Error);
        let inner_var = self.fresh_var("inner", Ty::Error);

        let data_con = DataCon {
            name: newtype_def.con.name,
            ty_con: type_con,
            tag: 0,
            arity: 1,
        };

        // show (Con x) = "Con " ++ show x
        let con_str = self.make_string(newtype_def.con.name.as_str(), span);
        let space_str = self.make_string(" ", span);
        let show_inner = self.make_show_call(&inner_var, span);

        let result = self.make_append(self.make_append(con_str, space_str, span), show_inner, span);

        let case_expr = self.make_case(
            core::Expr::Var(x_var.clone(), span),
            vec![Alt {
                con: AltCon::DataCon(data_con),
                binders: vec![inner_var],
                rhs: result,
            }],
            span,
        );

        let body = core::Expr::Lam(x_var, Box::new(case_expr), span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("show"),
            bhc_hir::DefId::new(show_method_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Show"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(show_method_var, Box::new(body))],
        })
    }

    // =========================================================================
    // Enum derivation
    // =========================================================================

    /// Derive Enum for a data type.
    ///
    /// Enum can only be derived for enumeration types (all constructors have no fields).
    fn derive_enum_data(&mut self, data_def: &DataDef) -> Option<DerivedInstance> {
        let span = data_def.span;

        // Check all constructors have no fields
        let is_enum = data_def.cons.iter().all(|con| match &con.fields {
            ConFields::Positional(fields) => fields.is_empty(),
            ConFields::Named(fields) => fields.is_empty(),
        });

        if !is_enum {
            return None;
        }

        let instance_type = self.build_instance_type(data_def.name, &data_def.params);

        // Generate fromEnum method: fromEnum Con_i = i
        let from_enum_var = self.fresh_var(
            &format!("$derived_fromEnum_{}", data_def.name.as_str()),
            Ty::Error,
        );
        let from_enum_body = self.generate_from_enum_body(data_def, span);

        // Generate toEnum method: toEnum i = case i of { 0 -> Con_0; 1 -> Con_1; ... }
        let to_enum_var = self.fresh_var(
            &format!("$derived_toEnum_{}", data_def.name.as_str()),
            Ty::Error,
        );
        let to_enum_body = self.generate_to_enum_body(data_def, span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("fromEnum"),
            bhc_hir::DefId::new(from_enum_var.id.index()),
        );
        methods.insert(
            Symbol::intern("toEnum"),
            bhc_hir::DefId::new(to_enum_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Enum"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        let bindings = vec![
            Bind::NonRec(from_enum_var, Box::new(from_enum_body)),
            Bind::NonRec(to_enum_var, Box::new(to_enum_body)),
        ];

        Some(DerivedInstance { instance, bindings })
    }

    /// Generate the body of fromEnum: converts constructor to its tag.
    fn generate_from_enum_body(&mut self, data_def: &DataDef, span: Span) -> core::Expr {
        let x_var = self.fresh_var("x", Ty::Error);
        let ty_con = TyCon::new(data_def.name, Kind::Star);

        let alts: Vec<Alt> = data_def
            .cons
            .iter()
            .enumerate()
            .map(|(tag, con)| {
                let data_con = DataCon {
                    name: con.name,
                    ty_con: ty_con.clone(),
                    tag: tag as u32,
                    arity: 0,
                };
                Alt {
                    con: AltCon::DataCon(data_con),
                    binders: vec![],
                    rhs: self.make_int(tag as i64, span),
                }
            })
            .collect();

        let case_expr = self.make_case(core::Expr::Var(x_var.clone(), span), alts, span);
        core::Expr::Lam(x_var, Box::new(case_expr), span)
    }

    /// Generate the body of toEnum: converts Int to constructor.
    fn generate_to_enum_body(&mut self, data_def: &DataDef, span: Span) -> core::Expr {
        let n_var = self.fresh_var("n", Ty::Error);
        let ty_con = TyCon::new(data_def.name, Kind::Star);

        let alts: Vec<Alt> = data_def
            .cons
            .iter()
            .enumerate()
            .map(|(tag, con)| {
                let data_con = DataCon {
                    name: con.name,
                    ty_con: ty_con.clone(),
                    tag: tag as u32,
                    arity: 0,
                };
                // Pattern match on the Int literal
                Alt {
                    con: AltCon::Lit(core::Literal::Int(tag as i64)),
                    binders: vec![],
                    rhs: self.make_constructor(data_con, span),
                }
            })
            .collect();

        // Add a default case that calls error
        let mut all_alts = alts;
        all_alts.push(Alt {
            con: AltCon::Default,
            binders: vec![],
            rhs: self.make_error("toEnum: bad argument", span),
        });

        let case_expr = self.make_case(core::Expr::Var(n_var.clone(), span), all_alts, span);
        core::Expr::Lam(n_var, Box::new(case_expr), span)
    }

    // =========================================================================
    // Bounded derivation
    // =========================================================================

    /// Derive Bounded for a data type.
    ///
    /// Bounded can only be derived for enumeration types (all constructors have no fields).
    fn derive_bounded_data(&mut self, data_def: &DataDef) -> Option<DerivedInstance> {
        let span = data_def.span;

        // Check all constructors have no fields
        let is_enum = data_def.cons.iter().all(|con| match &con.fields {
            ConFields::Positional(fields) => fields.is_empty(),
            ConFields::Named(fields) => fields.is_empty(),
        });

        if !is_enum || data_def.cons.is_empty() {
            return None;
        }

        let instance_type = self.build_instance_type(data_def.name, &data_def.params);
        let ty_con = TyCon::new(data_def.name, Kind::Star);

        // minBound = first constructor
        let min_bound_var = self.fresh_var(
            &format!("$derived_minBound_{}", data_def.name.as_str()),
            Ty::Error,
        );
        let first_con = &data_def.cons[0];
        let min_data_con = DataCon {
            name: first_con.name,
            ty_con: ty_con.clone(),
            tag: 0,
            arity: 0,
        };
        let min_bound_body = self.make_constructor(min_data_con, span);

        // maxBound = last constructor
        let max_bound_var = self.fresh_var(
            &format!("$derived_maxBound_{}", data_def.name.as_str()),
            Ty::Error,
        );
        let last_con = &data_def.cons[data_def.cons.len() - 1];
        let max_data_con = DataCon {
            name: last_con.name,
            ty_con: ty_con.clone(),
            tag: (data_def.cons.len() - 1) as u32,
            arity: 0,
        };
        let max_bound_body = self.make_constructor(max_data_con, span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("minBound"),
            bhc_hir::DefId::new(min_bound_var.id.index()),
        );
        methods.insert(
            Symbol::intern("maxBound"),
            bhc_hir::DefId::new(max_bound_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Bounded"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        let bindings = vec![
            Bind::NonRec(min_bound_var, Box::new(min_bound_body)),
            Bind::NonRec(max_bound_var, Box::new(max_bound_body)),
        ];

        Some(DerivedInstance { instance, bindings })
    }

    // =========================================================================
    // Functor derivation
    // =========================================================================

    /// Derive Functor for a data type.
    ///
    /// Functor can only be derived for types with exactly one type parameter
    /// that appears in covariant position.
    fn derive_functor_data(&mut self, data_def: &DataDef) -> Option<DerivedInstance> {
        // Functor requires exactly one type parameter
        if data_def.params.len() != 1 {
            return None;
        }

        let span = data_def.span;
        let type_param = &data_def.params[0];

        // Build instance type for Functor (just the type constructor, no params)
        let instance_type = Ty::Con(TyCon::new(data_def.name, Kind::star_to_star()));

        // Generate fmap method
        let fmap_var = self.fresh_var(
            &format!("$derived_fmap_{}", data_def.name.as_str()),
            Ty::Error,
        );
        let fmap_body = self.generate_fmap_body(data_def, type_param, span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("fmap"),
            bhc_hir::DefId::new(fmap_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Functor"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(fmap_var, Box::new(fmap_body))],
        })
    }

    /// Generate the body of fmap for a data type.
    ///
    /// fmap f x = case x of
    ///   Con1 a1 a2 ... -> Con1 (f a1) a2 ...  -- apply f where type param appears
    ///   Con2 b1 b2 ... -> Con2 b1 (f b2) ...
    fn generate_fmap_body(
        &mut self,
        data_def: &DataDef,
        type_param: &TyVar,
        span: Span,
    ) -> core::Expr {
        let f_var = self.fresh_var("f", Ty::Error);
        let x_var = self.fresh_var("x", Ty::Error);
        let ty_con = TyCon::new(data_def.name, Kind::Star);

        let alts: Vec<Alt> = data_def
            .cons
            .iter()
            .enumerate()
            .map(|(tag, con)| {
                let field_types = match &con.fields {
                    ConFields::Positional(fields) => fields.clone(),
                    ConFields::Named(fields) => fields.iter().map(|f| f.ty.clone()).collect(),
                };

                // Create binders for each field
                let binders: Vec<Var> = field_types
                    .iter()
                    .enumerate()
                    .map(|(i, _)| self.fresh_var(&format!("a{}", i), Ty::Error))
                    .collect();

                // Build the result: apply f to fields that contain the type param
                let mut result_args: Vec<core::Expr> = vec![];
                for (i, field_ty) in field_types.iter().enumerate() {
                    let field_var = &binders[i];
                    if self.type_contains_param(field_ty, type_param) {
                        // Apply f to this field
                        let mapped = core::Expr::App(
                            Box::new(core::Expr::Var(f_var.clone(), span)),
                            Box::new(core::Expr::Var(field_var.clone(), span)),
                            span,
                        );
                        result_args.push(mapped);
                    } else {
                        // Keep field unchanged
                        result_args.push(core::Expr::Var(field_var.clone(), span));
                    }
                }

                let data_con = DataCon {
                    name: con.name,
                    ty_con: ty_con.clone(),
                    tag: tag as u32,
                    arity: field_types.len() as u32,
                };

                // Build constructor application
                let rhs = self.apply_constructor(data_con.clone(), result_args, span);

                Alt {
                    con: AltCon::DataCon(data_con),
                    binders,
                    rhs,
                }
            })
            .collect();

        let case_expr = self.make_case(core::Expr::Var(x_var.clone(), span), alts, span);
        let inner = core::Expr::Lam(x_var, Box::new(case_expr), span);
        core::Expr::Lam(f_var, Box::new(inner), span)
    }

    /// Derive Functor for a newtype.
    fn derive_functor_newtype(&mut self, newtype_def: &NewtypeDef) -> Option<DerivedInstance> {
        // Functor requires exactly one type parameter
        if newtype_def.params.len() != 1 {
            return None;
        }

        let span = newtype_def.span;
        let type_param = &newtype_def.params[0];

        // Check if the wrapped type contains the type parameter
        let field_ty = self.get_newtype_field_ty(newtype_def)?;
        if !self.type_contains_param(&field_ty, type_param) {
            return None;
        }

        let instance_type = Ty::Con(TyCon::new(newtype_def.name, Kind::star_to_star()));

        // fmap f (Con x) = Con (f x)
        let fmap_var = self.fresh_var(
            &format!("$derived_fmap_{}", newtype_def.name.as_str()),
            Ty::Error,
        );

        let f_var = self.fresh_var("f", Ty::Error);
        let x_var = self.fresh_var("x", Ty::Error);
        let inner_var = self.fresh_var("inner", Ty::Error);

        let ty_con = TyCon::new(newtype_def.name, Kind::Star);
        let data_con = DataCon {
            name: newtype_def.con.name,
            ty_con: ty_con.clone(),
            tag: 0,
            arity: 1,
        };

        // f inner
        let mapped = core::Expr::App(
            Box::new(core::Expr::Var(f_var.clone(), span)),
            Box::new(core::Expr::Var(inner_var.clone(), span)),
            span,
        );

        // Con (f inner)
        let result = self.apply_constructor(data_con.clone(), vec![mapped], span);

        let alt = Alt {
            con: AltCon::DataCon(data_con),
            binders: vec![inner_var],
            rhs: result,
        };

        let case_expr = self.make_case(core::Expr::Var(x_var.clone(), span), vec![alt], span);
        let inner_lam = core::Expr::Lam(x_var, Box::new(case_expr), span);
        let fmap_body = core::Expr::Lam(f_var, Box::new(inner_lam), span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("fmap"),
            bhc_hir::DefId::new(fmap_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Functor"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(fmap_var, Box::new(fmap_body))],
        })
    }

    // =========================================================================
    // Foldable derivation
    // =========================================================================

    /// Derive Foldable for a data type.
    fn derive_foldable_data(&mut self, data_def: &DataDef) -> Option<DerivedInstance> {
        if data_def.params.len() != 1 {
            return None;
        }

        let span = data_def.span;
        let type_param = &data_def.params[0];

        let instance_type = Ty::Con(TyCon::new(data_def.name, Kind::star_to_star()));

        // Generate foldr method
        let foldr_var = self.fresh_var(
            &format!("$derived_foldr_{}", data_def.name.as_str()),
            Ty::Error,
        );
        let foldr_body = self.generate_foldr_body(data_def, type_param, span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("foldr"),
            bhc_hir::DefId::new(foldr_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Foldable"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(foldr_var, Box::new(foldr_body))],
        })
    }

    /// Generate the body of foldr for a data type.
    ///
    /// foldr f z x = case x of
    ///   Con1 a1 a2 ... -> f a1 (f a2 (... z))  -- fold where type param appears
    ///   Con2 -> z  -- no type param fields
    fn generate_foldr_body(
        &mut self,
        data_def: &DataDef,
        type_param: &TyVar,
        span: Span,
    ) -> core::Expr {
        let f_var = self.fresh_var("f", Ty::Error);
        let z_var = self.fresh_var("z", Ty::Error);
        let x_var = self.fresh_var("x", Ty::Error);
        let ty_con = TyCon::new(data_def.name, Kind::Star);

        let alts: Vec<Alt> = data_def
            .cons
            .iter()
            .enumerate()
            .map(|(tag, con)| {
                let field_types = match &con.fields {
                    ConFields::Positional(fields) => fields.clone(),
                    ConFields::Named(fields) => fields.iter().map(|f| f.ty.clone()).collect(),
                };

                let binders: Vec<Var> = field_types
                    .iter()
                    .enumerate()
                    .map(|(i, _)| self.fresh_var(&format!("a{}", i), Ty::Error))
                    .collect();

                // Collect fields that contain the type parameter (in reverse for foldr)
                let param_fields: Vec<&Var> = field_types
                    .iter()
                    .enumerate()
                    .filter(|(_, ty)| self.type_contains_param(ty, type_param))
                    .map(|(i, _)| &binders[i])
                    .collect();

                // Build: f a1 (f a2 (... z))
                let mut result = core::Expr::Var(z_var.clone(), span);
                for field_var in param_fields.into_iter().rev() {
                    result = core::Expr::App(
                        Box::new(core::Expr::App(
                            Box::new(core::Expr::Var(f_var.clone(), span)),
                            Box::new(core::Expr::Var(field_var.clone(), span)),
                            span,
                        )),
                        Box::new(result),
                        span,
                    );
                }

                let data_con = DataCon {
                    name: con.name,
                    ty_con: ty_con.clone(),
                    tag: tag as u32,
                    arity: field_types.len() as u32,
                };

                Alt {
                    con: AltCon::DataCon(data_con),
                    binders,
                    rhs: result,
                }
            })
            .collect();

        let case_expr = self.make_case(core::Expr::Var(x_var.clone(), span), alts, span);
        let lam1 = core::Expr::Lam(x_var, Box::new(case_expr), span);
        let lam2 = core::Expr::Lam(z_var, Box::new(lam1), span);
        core::Expr::Lam(f_var, Box::new(lam2), span)
    }

    /// Derive Foldable for a newtype.
    fn derive_foldable_newtype(&mut self, newtype_def: &NewtypeDef) -> Option<DerivedInstance> {
        if newtype_def.params.len() != 1 {
            return None;
        }

        let span = newtype_def.span;
        let type_param = &newtype_def.params[0];

        let field_ty = self.get_newtype_field_ty(newtype_def)?;
        if !self.type_contains_param(&field_ty, type_param) {
            return None;
        }

        let instance_type = Ty::Con(TyCon::new(newtype_def.name, Kind::star_to_star()));

        // foldr f z (Con x) = f x z
        let foldr_var = self.fresh_var(
            &format!("$derived_foldr_{}", newtype_def.name.as_str()),
            Ty::Error,
        );

        let f_var = self.fresh_var("f", Ty::Error);
        let z_var = self.fresh_var("z", Ty::Error);
        let x_var = self.fresh_var("x", Ty::Error);
        let inner_var = self.fresh_var("inner", Ty::Error);

        let ty_con = TyCon::new(newtype_def.name, Kind::Star);
        let data_con = DataCon {
            name: newtype_def.con.name,
            ty_con,
            tag: 0,
            arity: 1,
        };

        // f inner z
        let result = core::Expr::App(
            Box::new(core::Expr::App(
                Box::new(core::Expr::Var(f_var.clone(), span)),
                Box::new(core::Expr::Var(inner_var.clone(), span)),
                span,
            )),
            Box::new(core::Expr::Var(z_var.clone(), span)),
            span,
        );

        let alt = Alt {
            con: AltCon::DataCon(data_con),
            binders: vec![inner_var],
            rhs: result,
        };

        let case_expr = self.make_case(core::Expr::Var(x_var.clone(), span), vec![alt], span);
        let lam1 = core::Expr::Lam(x_var, Box::new(case_expr), span);
        let lam2 = core::Expr::Lam(z_var, Box::new(lam1), span);
        let foldr_body = core::Expr::Lam(f_var, Box::new(lam2), span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("foldr"),
            bhc_hir::DefId::new(foldr_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Foldable"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(foldr_var, Box::new(foldr_body))],
        })
    }

    // =========================================================================
    // Traversable derivation
    // =========================================================================

    /// Derive Traversable for a data type.
    fn derive_traversable_data(&mut self, data_def: &DataDef) -> Option<DerivedInstance> {
        if data_def.params.len() != 1 {
            return None;
        }

        let span = data_def.span;
        let type_param = &data_def.params[0];

        let instance_type = Ty::Con(TyCon::new(data_def.name, Kind::star_to_star()));

        // Generate traverse method
        let traverse_var = self.fresh_var(
            &format!("$derived_traverse_{}", data_def.name.as_str()),
            Ty::Error,
        );
        let traverse_body = self.generate_traverse_body(data_def, type_param, span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("traverse"),
            bhc_hir::DefId::new(traverse_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Traversable"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(traverse_var, Box::new(traverse_body))],
        })
    }

    /// Generate the body of traverse for a data type.
    ///
    /// traverse f x = case x of
    ///   Con1 a1 a2 ... -> Con1 <$> f a1 <*> f a2 <*> ...
    ///   Con2 b1 -> pure (Con2 b1)  -- no type param fields
    fn generate_traverse_body(
        &mut self,
        data_def: &DataDef,
        type_param: &TyVar,
        span: Span,
    ) -> core::Expr {
        let f_var = self.fresh_var("f", Ty::Error);
        let x_var = self.fresh_var("x", Ty::Error);
        let ty_con = TyCon::new(data_def.name, Kind::Star);

        let alts: Vec<Alt> = data_def
            .cons
            .iter()
            .enumerate()
            .map(|(tag, con)| {
                let field_types = match &con.fields {
                    ConFields::Positional(fields) => fields.clone(),
                    ConFields::Named(fields) => fields.iter().map(|f| f.ty.clone()).collect(),
                };

                let binders: Vec<Var> = field_types
                    .iter()
                    .enumerate()
                    .map(|(i, _)| self.fresh_var(&format!("a{}", i), Ty::Error))
                    .collect();

                let data_con = DataCon {
                    name: con.name,
                    ty_con: ty_con.clone(),
                    tag: tag as u32,
                    arity: field_types.len() as u32,
                };

                // Check if any field contains the type parameter
                let has_param_fields = field_types
                    .iter()
                    .any(|ty| self.type_contains_param(ty, type_param));

                let rhs = if has_param_fields {
                    // Build: Con <$> f a1 <*> f a2 <*> pure a3 ...
                    self.build_traversal(
                        data_con.clone(),
                        &binders,
                        &field_types,
                        type_param,
                        &f_var,
                        span,
                    )
                } else {
                    // pure (Con a1 a2 ...)
                    let con_app = self.apply_constructor(
                        data_con.clone(),
                        binders
                            .iter()
                            .map(|v| core::Expr::Var(v.clone(), span))
                            .collect(),
                        span,
                    );
                    self.make_pure(con_app, span)
                };

                Alt {
                    con: AltCon::DataCon(data_con),
                    binders,
                    rhs,
                }
            })
            .collect();

        let case_expr = self.make_case(core::Expr::Var(x_var.clone(), span), alts, span);
        let inner = core::Expr::Lam(x_var, Box::new(case_expr), span);
        core::Expr::Lam(f_var, Box::new(inner), span)
    }

    /// Build the traversal expression for a constructor.
    fn build_traversal(
        &mut self,
        data_con: DataCon,
        binders: &[Var],
        field_types: &[Ty],
        type_param: &TyVar,
        f_var: &Var,
        span: Span,
    ) -> core::Expr {
        // Start with pure Con
        let con_expr = self.make_constructor(data_con, span);

        if binders.is_empty() {
            return self.make_pure(con_expr, span);
        }

        // First, apply pure to the constructor
        let mut result = self.make_pure(con_expr, span);

        // Then <*> each field (either f field or pure field)
        for (i, (field_var, field_ty)) in binders.iter().zip(field_types.iter()).enumerate() {
            let field_expr = if self.type_contains_param(field_ty, type_param) {
                // f field
                core::Expr::App(
                    Box::new(core::Expr::Var(f_var.clone(), span)),
                    Box::new(core::Expr::Var(field_var.clone(), span)),
                    span,
                )
            } else {
                // pure field
                self.make_pure(core::Expr::Var(field_var.clone(), span), span)
            };

            // result <*> field_expr
            result = self.make_ap(result, field_expr, span);
        }

        result
    }

    /// Derive Traversable for a newtype.
    fn derive_traversable_newtype(&mut self, newtype_def: &NewtypeDef) -> Option<DerivedInstance> {
        if newtype_def.params.len() != 1 {
            return None;
        }

        let span = newtype_def.span;
        let type_param = &newtype_def.params[0];

        let field_ty = self.get_newtype_field_ty(newtype_def)?;
        if !self.type_contains_param(&field_ty, type_param) {
            return None;
        }

        let instance_type = Ty::Con(TyCon::new(newtype_def.name, Kind::star_to_star()));

        // traverse f (Con x) = Con <$> f x
        let traverse_var = self.fresh_var(
            &format!("$derived_traverse_{}", newtype_def.name.as_str()),
            Ty::Error,
        );

        let f_var = self.fresh_var("f", Ty::Error);
        let x_var = self.fresh_var("x", Ty::Error);
        let inner_var = self.fresh_var("inner", Ty::Error);

        let ty_con = TyCon::new(newtype_def.name, Kind::Star);
        let data_con = DataCon {
            name: newtype_def.con.name,
            ty_con,
            tag: 0,
            arity: 1,
        };

        // f inner
        let f_applied = core::Expr::App(
            Box::new(core::Expr::Var(f_var.clone(), span)),
            Box::new(core::Expr::Var(inner_var.clone(), span)),
            span,
        );

        // Con <$> f inner
        let con_expr = self.make_constructor(data_con.clone(), span);
        let result = self.make_fmap(con_expr, f_applied, span);

        let alt = Alt {
            con: AltCon::DataCon(data_con),
            binders: vec![inner_var],
            rhs: result,
        };

        let case_expr = self.make_case(core::Expr::Var(x_var.clone(), span), vec![alt], span);
        let inner_lam = core::Expr::Lam(x_var, Box::new(case_expr), span);
        let traverse_body = core::Expr::Lam(f_var, Box::new(inner_lam), span);

        let mut methods = FxHashMap::default();
        methods.insert(
            Symbol::intern("traverse"),
            bhc_hir::DefId::new(traverse_var.id.index()),
        );

        let instance = InstanceInfo {
            class: Symbol::intern("Traversable"),
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
            instance_constraints: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(traverse_var, Box::new(traverse_body))],
        })
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Build the instance type from a type name and parameters.
    fn build_instance_type(&self, name: Symbol, params: &[TyVar]) -> Ty {
        let base = Ty::Con(TyCon::new(name, Kind::Star));
        if params.is_empty() {
            base
        } else {
            params.iter().fold(base, |acc, param| {
                Ty::App(Box::new(acc), Box::new(Ty::Var(param.clone())))
            })
        }
    }

    /// Make a Bool literal.
    fn make_bool(&self, value: bool, span: Span) -> core::Expr {
        let bool_con = TyCon::new(Symbol::intern("Bool"), Kind::Star);
        let data_con = DataCon {
            name: Symbol::intern(if value { "True" } else { "False" }),
            ty_con: bool_con.clone(),
            tag: if value { 1 } else { 0 },
            arity: 0,
        };
        let var = Var {
            name: data_con.name,
            id: VarId::new(0),
            ty: Ty::Con(bool_con),
        };
        core::Expr::Var(var, span)
    }

    /// Make an Ordering value.
    fn make_ordering(&self, ordering: &str, span: Span) -> core::Expr {
        let ordering_con = TyCon::new(Symbol::intern("Ordering"), Kind::Star);
        let (tag, name) = match ordering {
            "LT" => (0, "LT"),
            "EQ" => (1, "EQ"),
            "GT" => (2, "GT"),
            _ => (1, "EQ"),
        };
        let data_con = DataCon {
            name: Symbol::intern(name),
            ty_con: ordering_con.clone(),
            tag,
            arity: 0,
        };
        let var = Var {
            name: data_con.name,
            id: VarId::new(0),
            ty: Ty::Con(ordering_con),
        };
        core::Expr::Var(var, span)
    }

    /// Make a String literal.
    fn make_string(&self, s: &str, span: Span) -> core::Expr {
        let string_ty = Ty::Con(TyCon::new(Symbol::intern("String"), Kind::Star));
        core::Expr::Lit(core::Literal::String(Symbol::intern(s)), string_ty, span)
    }

    /// Make an Int literal.
    fn make_int(&self, n: i64, span: Span) -> core::Expr {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        core::Expr::Lit(core::Literal::Int(n), int_ty, span)
    }

    /// Make a constructor expression (no arguments).
    fn make_constructor(&self, data_con: DataCon, span: Span) -> core::Expr {
        let var = Var {
            name: data_con.name,
            id: VarId::new(data_con.tag as usize),
            ty: Ty::Con(data_con.ty_con.clone()),
        };
        core::Expr::Var(var, span)
    }

    /// Apply a constructor to arguments.
    fn apply_constructor(
        &self,
        data_con: DataCon,
        args: Vec<core::Expr>,
        span: Span,
    ) -> core::Expr {
        let mut result = self.make_constructor(data_con, span);
        for arg in args {
            result = core::Expr::App(Box::new(result), Box::new(arg), span);
        }
        result
    }

    /// Make an error call.
    fn make_error(&self, msg: &str, span: Span) -> core::Expr {
        let error_var = Var {
            name: Symbol::intern("error"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        let msg_expr = self.make_string(msg, span);
        core::Expr::App(
            Box::new(core::Expr::Var(error_var, span)),
            Box::new(msg_expr),
            span,
        )
    }

    /// Check if a type contains a type variable.
    fn type_contains_param(&self, ty: &Ty, param: &TyVar) -> bool {
        match ty {
            Ty::Var(v) => v.id == param.id,
            Ty::Con(_) => false,
            Ty::App(f, a) => {
                self.type_contains_param(f, param) || self.type_contains_param(a, param)
            }
            Ty::Fun(a, r) => {
                self.type_contains_param(a, param) || self.type_contains_param(r, param)
            }
            Ty::Tuple(ts) => ts.iter().any(|t| self.type_contains_param(t, param)),
            Ty::List(e) => self.type_contains_param(e, param),
            Ty::Forall(_, inner) => self.type_contains_param(inner, param),
            Ty::Prim(_) => false,
            Ty::Nat(_) => false,
            Ty::TyList(list) => self.tylist_contains_param(list, param),
            Ty::Error => false,
        }
    }

    /// Check if a TyList contains a type variable.
    fn tylist_contains_param(&self, list: &bhc_types::TyList, param: &TyVar) -> bool {
        use bhc_types::TyList;
        match list {
            TyList::Nil => false,
            TyList::Cons(head, tail) => {
                self.type_contains_param(head, param) || self.tylist_contains_param(tail, param)
            }
            TyList::Var(v) => v.id == param.id,
            TyList::Append(xs, ys) => {
                self.tylist_contains_param(xs, param) || self.tylist_contains_param(ys, param)
            }
        }
    }

    /// Get the wrapped type from a newtype's constructor.
    fn get_newtype_field_ty(&self, newtype_def: &NewtypeDef) -> Option<Ty> {
        match &newtype_def.con.fields {
            ConFields::Positional(fields) if fields.len() == 1 => Some(fields[0].clone()),
            ConFields::Named(fields) if fields.len() == 1 => Some(fields[0].ty.clone()),
            _ => None,
        }
    }

    /// Make a `pure x` expression.
    fn make_pure(&self, x: core::Expr, span: Span) -> core::Expr {
        let pure_var = Var {
            name: Symbol::intern("pure"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        core::Expr::App(Box::new(core::Expr::Var(pure_var, span)), Box::new(x), span)
    }

    /// Make a `f <*> x` expression.
    fn make_ap(&self, f: core::Expr, x: core::Expr, span: Span) -> core::Expr {
        let ap_var = Var {
            name: Symbol::intern("<*>"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        core::Expr::App(
            Box::new(core::Expr::App(
                Box::new(core::Expr::Var(ap_var, span)),
                Box::new(f),
                span,
            )),
            Box::new(x),
            span,
        )
    }

    /// Make a `f <$> x` expression (fmap f x).
    fn make_fmap(&self, f: core::Expr, x: core::Expr, span: Span) -> core::Expr {
        let fmap_var = Var {
            name: Symbol::intern("<$>"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        core::Expr::App(
            Box::new(core::Expr::App(
                Box::new(core::Expr::Var(fmap_var, span)),
                Box::new(f),
                span,
            )),
            Box::new(x),
            span,
        )
    }
}

impl Default for DerivingContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_eq_simple_enum() {
        let mut ctx = DerivingContext::new();

        // data Color = Red | Green | Blue deriving Eq
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Color"),
            params: vec![],
            cons: vec![
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(2),
                    name: Symbol::intern("Red"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(3),
                    name: Symbol::intern("Green"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(4),
                    name: Symbol::intern("Blue"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
            ],
            deriving: vec![Symbol::intern("Eq")],
            span: Span::default(),
        };

        let result = ctx.derive_eq_data(&data_def);
        assert!(result.is_some());

        let derived = result.unwrap();
        assert_eq!(derived.instance.class.as_str(), "Eq");
        assert!(!derived.bindings.is_empty());
    }

    #[test]
    fn test_derive_eq_with_fields() {
        let mut ctx = DerivingContext::new();

        // data Point = Point Int Int deriving Eq
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Point"),
            params: vec![],
            cons: vec![bhc_hir::ConDef {
                id: bhc_hir::DefId::new(2),
                name: Symbol::intern("Point"),
                fields: ConFields::Positional(vec![
                    Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star)),
                    Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star)),
                ]),
                span: Span::default(),
            }],
            deriving: vec![Symbol::intern("Eq")],
            span: Span::default(),
        };

        let result = ctx.derive_eq_data(&data_def);
        assert!(result.is_some());

        let derived = result.unwrap();
        assert_eq!(derived.instance.class.as_str(), "Eq");
        assert!(!derived.bindings.is_empty());
    }

    #[test]
    fn test_derive_ord_enum() {
        let mut ctx = DerivingContext::new();

        // data Order = First | Second | Third deriving Ord
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Order"),
            params: vec![],
            cons: vec![
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(2),
                    name: Symbol::intern("First"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(3),
                    name: Symbol::intern("Second"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(4),
                    name: Symbol::intern("Third"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
            ],
            deriving: vec![Symbol::intern("Ord")],
            span: Span::default(),
        };

        let result = ctx.derive_ord_data(&data_def);
        assert!(result.is_some());

        let derived = result.unwrap();
        assert_eq!(derived.instance.class.as_str(), "Ord");
        assert!(!derived.bindings.is_empty());
    }

    #[test]
    fn test_derive_show_enum() {
        let mut ctx = DerivingContext::new();

        // data Color = Red | Green deriving Show
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Color"),
            params: vec![],
            cons: vec![
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(2),
                    name: Symbol::intern("Red"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(3),
                    name: Symbol::intern("Green"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
            ],
            deriving: vec![Symbol::intern("Show")],
            span: Span::default(),
        };

        let result = ctx.derive_show_data(&data_def);
        assert!(result.is_some());

        let derived = result.unwrap();
        assert_eq!(derived.instance.class.as_str(), "Show");
        assert!(!derived.bindings.is_empty());
    }

    #[test]
    fn test_derive_enum_simple() {
        let mut ctx = DerivingContext::new();

        // data Color = Red | Green | Blue deriving Enum
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Color"),
            params: vec![],
            cons: vec![
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(2),
                    name: Symbol::intern("Red"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(3),
                    name: Symbol::intern("Green"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(4),
                    name: Symbol::intern("Blue"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
            ],
            deriving: vec![Symbol::intern("Enum")],
            span: Span::default(),
        };

        let result = ctx.derive_enum_data(&data_def);
        assert!(result.is_some());

        let derived = result.unwrap();
        assert_eq!(derived.instance.class.as_str(), "Enum");
        // Should have fromEnum and toEnum
        assert!(derived
            .instance
            .methods
            .contains_key(&Symbol::intern("fromEnum")));
        assert!(derived
            .instance
            .methods
            .contains_key(&Symbol::intern("toEnum")));
        assert_eq!(derived.bindings.len(), 2);
    }

    #[test]
    fn test_derive_enum_fails_with_fields() {
        let mut ctx = DerivingContext::new();

        // data Maybe a = Nothing | Just a  -- should NOT derive Enum
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Maybe"),
            params: vec![TyVar::new(0, Kind::Star)],
            cons: vec![
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(2),
                    name: Symbol::intern("Nothing"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(3),
                    name: Symbol::intern("Just"),
                    fields: ConFields::Positional(vec![Ty::Var(TyVar::new(0, Kind::Star))]),
                    span: Span::default(),
                },
            ],
            deriving: vec![Symbol::intern("Enum")],
            span: Span::default(),
        };

        let result = ctx.derive_enum_data(&data_def);
        assert!(result.is_none()); // Should fail - not a simple enum
    }

    #[test]
    fn test_derive_bounded_simple() {
        let mut ctx = DerivingContext::new();

        // data Color = Red | Green | Blue deriving Bounded
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Color"),
            params: vec![],
            cons: vec![
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(2),
                    name: Symbol::intern("Red"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(3),
                    name: Symbol::intern("Green"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
                bhc_hir::ConDef {
                    id: bhc_hir::DefId::new(4),
                    name: Symbol::intern("Blue"),
                    fields: ConFields::Positional(vec![]),
                    span: Span::default(),
                },
            ],
            deriving: vec![Symbol::intern("Bounded")],
            span: Span::default(),
        };

        let result = ctx.derive_bounded_data(&data_def);
        assert!(result.is_some());

        let derived = result.unwrap();
        assert_eq!(derived.instance.class.as_str(), "Bounded");
        assert!(derived
            .instance
            .methods
            .contains_key(&Symbol::intern("minBound")));
        assert!(derived
            .instance
            .methods
            .contains_key(&Symbol::intern("maxBound")));
        assert_eq!(derived.bindings.len(), 2);
    }

    #[test]
    fn test_derive_functor_simple() {
        let mut ctx = DerivingContext::new();

        // data Box a = Box a deriving Functor
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Box"),
            params: vec![TyVar::new(0, Kind::Star)],
            cons: vec![bhc_hir::ConDef {
                id: bhc_hir::DefId::new(2),
                name: Symbol::intern("Box"),
                fields: ConFields::Positional(vec![Ty::Var(TyVar::new(0, Kind::Star))]),
                span: Span::default(),
            }],
            deriving: vec![Symbol::intern("Functor")],
            span: Span::default(),
        };

        let result = ctx.derive_functor_data(&data_def);
        assert!(result.is_some());

        let derived = result.unwrap();
        assert_eq!(derived.instance.class.as_str(), "Functor");
        assert!(derived
            .instance
            .methods
            .contains_key(&Symbol::intern("fmap")));
    }

    #[test]
    fn test_derive_functor_fails_no_param() {
        let mut ctx = DerivingContext::new();

        // data Unit = Unit  -- should NOT derive Functor (no type param)
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Unit"),
            params: vec![],
            cons: vec![bhc_hir::ConDef {
                id: bhc_hir::DefId::new(2),
                name: Symbol::intern("Unit"),
                fields: ConFields::Positional(vec![]),
                span: Span::default(),
            }],
            deriving: vec![Symbol::intern("Functor")],
            span: Span::default(),
        };

        let result = ctx.derive_functor_data(&data_def);
        assert!(result.is_none()); // Should fail - no type parameter
    }

    #[test]
    fn test_derive_foldable_simple() {
        let mut ctx = DerivingContext::new();

        // data Box a = Box a deriving Foldable
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Box"),
            params: vec![TyVar::new(0, Kind::Star)],
            cons: vec![bhc_hir::ConDef {
                id: bhc_hir::DefId::new(2),
                name: Symbol::intern("Box"),
                fields: ConFields::Positional(vec![Ty::Var(TyVar::new(0, Kind::Star))]),
                span: Span::default(),
            }],
            deriving: vec![Symbol::intern("Foldable")],
            span: Span::default(),
        };

        let result = ctx.derive_foldable_data(&data_def);
        assert!(result.is_some());

        let derived = result.unwrap();
        assert_eq!(derived.instance.class.as_str(), "Foldable");
        assert!(derived
            .instance
            .methods
            .contains_key(&Symbol::intern("foldr")));
    }

    #[test]
    fn test_derive_traversable_simple() {
        let mut ctx = DerivingContext::new();

        // data Box a = Box a deriving Traversable
        let data_def = DataDef {
            id: bhc_hir::DefId::new(1),
            name: Symbol::intern("Box"),
            params: vec![TyVar::new(0, Kind::Star)],
            cons: vec![bhc_hir::ConDef {
                id: bhc_hir::DefId::new(2),
                name: Symbol::intern("Box"),
                fields: ConFields::Positional(vec![Ty::Var(TyVar::new(0, Kind::Star))]),
                span: Span::default(),
            }],
            deriving: vec![Symbol::intern("Traversable")],
            span: Span::default(),
        };

        let result = ctx.derive_traversable_data(&data_def);
        assert!(result.is_some());

        let derived = result.unwrap();
        assert_eq!(derived.instance.class.as_str(), "Traversable");
        assert!(derived
            .instance
            .methods
            .contains_key(&Symbol::intern("traverse")));
    }
}
