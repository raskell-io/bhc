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
            fresh_counter: 10000, // Start high to avoid collisions
        }
    }

    /// Generate a fresh variable.
    fn fresh_var(&mut self, prefix: &str, ty: Ty) -> Var {
        let name = Symbol::intern(&format!("{}_{}", prefix, self.fresh_counter));
        self.fresh_counter += 1;
        Var {
            name,
            id: VarId::new(self.fresh_counter as usize),
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
            instance_type,
            methods,
            superclass_instances: vec![],
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

        self.make_case(
            core::Expr::Var(x_var.clone(), span),
            outer_alts,
            span,
        )
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

        self.make_case(
            core::Expr::Var(x_var.clone(), span),
            outer_alts,
            span,
        )
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
        let mut result =
            self.make_eq_call(&x_fields[x_fields.len() - 1], &y_fields[y_fields.len() - 1], span);

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
            instance_type,
            methods,
            superclass_instances: vec![],
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
            instance_type: instance_type.clone(),
            methods,
            superclass_instances: vec![instance_type],
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

            let inner_case = self.make_case(
                core::Expr::Var(y_var.clone(), span),
                inner_alts,
                span,
            );

            outer_alts.push(Alt {
                con: AltCon::DataCon(x_data_con),
                binders: vec![],
                rhs: inner_case,
            });
        }

        self.make_case(
            core::Expr::Var(x_var.clone(), span),
            outer_alts,
            span,
        )
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

            let inner_case = self.make_case(
                core::Expr::Var(y_var.clone(), span),
                inner_alts,
                span,
            );

            outer_alts.push(Alt {
                con: AltCon::DataCon(x_data_con),
                binders: x_fields,
                rhs: inner_case,
            });
        }

        self.make_case(
            core::Expr::Var(x_var.clone(), span),
            outer_alts,
            span,
        )
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
            instance_type: instance_type.clone(),
            methods,
            superclass_instances: vec![instance_type],
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
            instance_type,
            methods,
            superclass_instances: vec![],
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

        let case_expr = self.make_case(
            core::Expr::Var(x_var.clone(), span),
            alts,
            span,
        );

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

        let result = self.make_append(
            self.make_append(con_str, space_str, span),
            show_inner,
            span,
        );

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
            instance_type,
            methods,
            superclass_instances: vec![],
        };

        Some(DerivedInstance {
            instance,
            bindings: vec![Bind::NonRec(show_method_var, Box::new(body))],
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
}
