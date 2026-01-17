//! Type checking context and state management.
//!
//! This module provides the main type checking context [`TyCtxt`] which
//! manages all state during type inference including:
//!
//! - Fresh type variable generation
//! - Substitution accumulation
//! - Type environment management
//! - Diagnostic collection

use bhc_diagnostics::{Diagnostic, DiagnosticHandler, FullSpan};
use bhc_hir::{
    Binding, DataDef, DefId, Equation, HirId, Item, Module, NewtypeDef, Pat, ValueDef,
};
use bhc_span::{FileId, Span};
use bhc_types::{Kind, Scheme, Subst, Ty, TyCon, TyVar};
use rustc_hash::FxHashMap;

use crate::binding_groups::BindingGroup;
use crate::builtins::Builtins;
use crate::env::TypeEnv;
use crate::TypedModule;

/// Generator for fresh type variables.
///
/// Produces unique type variable IDs during type inference.
#[derive(Debug, Default)]
pub struct TyVarGen {
    next_id: u32,
}

impl TyVarGen {
    /// Create a new type variable generator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a fresh type variable with kind `*`.
    #[must_use]
    pub fn fresh(&mut self) -> TyVar {
        let id = self.next_id;
        self.next_id += 1;
        TyVar::new_star(id)
    }

    /// Generate a fresh type variable with the given kind.
    #[must_use]
    pub fn fresh_with_kind(&mut self, kind: Kind) -> TyVar {
        let id = self.next_id;
        self.next_id += 1;
        TyVar::new(id, kind)
    }
}

/// The main type checking context.
///
/// This struct holds all state needed during type inference for a module.
/// It manages type variables, substitutions, the type environment, and
/// error diagnostics.
#[derive(Debug)]
pub struct TyCtxt {
    /// Generator for fresh type variables.
    ty_var_gen: TyVarGen,

    /// Current substitution (accumulated unification results).
    pub(crate) subst: Subst,

    /// Type environment (bindings in scope).
    pub(crate) env: TypeEnv,

    /// Diagnostic handler for error collection.
    diag: DiagnosticHandler,

    /// Built-in types (Int, Bool, etc.).
    pub(crate) builtins: Builtins,

    /// Current file ID for error reporting.
    pub(crate) file_id: FileId,

    /// Inferred types for expressions (`HirId` -> Ty).
    pub(crate) expr_types: FxHashMap<HirId, Ty>,

    /// Type schemes for definitions (`DefId` -> Scheme).
    pub(crate) def_schemes: FxHashMap<DefId, Scheme>,
}

impl TyCtxt {
    /// Create a new type checking context for the given file.
    #[must_use]
    pub fn new(file_id: FileId) -> Self {
        Self {
            ty_var_gen: TyVarGen::new(),
            subst: Subst::new(),
            env: TypeEnv::new(),
            diag: DiagnosticHandler::new(),
            builtins: Builtins::default(),
            file_id,
            expr_types: FxHashMap::default(),
            def_schemes: FxHashMap::default(),
        }
    }

    /// Generate a fresh type variable with kind `*`.
    pub fn fresh_ty_var(&mut self) -> TyVar {
        self.ty_var_gen.fresh()
    }

    /// Generate a fresh type (`Ty::Var`) with kind `*`.
    pub fn fresh_ty(&mut self) -> Ty {
        Ty::Var(self.fresh_ty_var())
    }

    /// Generate a fresh type variable with the given kind.
    pub fn fresh_ty_var_with_kind(&mut self, kind: Kind) -> TyVar {
        self.ty_var_gen.fresh_with_kind(kind)
    }

    /// Apply the current substitution to a type.
    #[must_use]
    pub fn apply_subst(&self, ty: &Ty) -> Ty {
        self.subst.apply(ty)
    }

    /// Register built-in types in the environment.
    pub fn register_builtins(&mut self) {
        self.builtins = Builtins::new();

        // Register type constructors
        self.env
            .register_type_con(self.builtins.int_con.clone());
        self.env
            .register_type_con(self.builtins.float_con.clone());
        self.env
            .register_type_con(self.builtins.char_con.clone());
        self.env
            .register_type_con(self.builtins.bool_con.clone());
        self.env
            .register_type_con(self.builtins.string_con.clone());
        self.env
            .register_type_con(self.builtins.list_con.clone());
        self.env
            .register_type_con(self.builtins.maybe_con.clone());
        self.env
            .register_type_con(self.builtins.either_con.clone());
        self.env.register_type_con(self.builtins.io_con.clone());

        // Register built-in data constructors
        self.builtins.register_data_cons(&mut self.env);
    }

    /// Register a data type definition.
    pub fn register_data_type(&mut self, data: &DataDef) {
        // Register the type constructor
        let kind = Self::compute_type_con_kind(data.params.len());
        let tycon = TyCon::new(data.name, kind);
        self.env.register_type_con(tycon);

        // Register data constructors
        for con in &data.cons {
            let scheme = self.compute_data_con_scheme(data, con);
            self.env.register_data_con(con.id, con.name, scheme);
        }
    }

    /// Register a newtype definition.
    pub fn register_newtype(&mut self, newtype: &NewtypeDef) {
        // Register the type constructor
        let kind = Self::compute_type_con_kind(newtype.params.len());
        let tycon = TyCon::new(newtype.name, kind);
        self.env.register_type_con(tycon);

        // Register the single constructor
        let scheme = self.compute_newtype_con_scheme(newtype);
        self.env
            .register_data_con(newtype.con.id, newtype.con.name, scheme);
    }

    /// Compute the kind of a type constructor given its arity.
    fn compute_type_con_kind(arity: usize) -> Kind {
        let mut kind = Kind::Star;
        for _ in 0..arity {
            kind = Kind::Arrow(Box::new(Kind::Star), Box::new(kind));
        }
        kind
    }

    /// Compute the type scheme for a data constructor.
    fn compute_data_con_scheme(
        &self,
        data: &DataDef,
        con: &bhc_hir::ConDef,
    ) -> Scheme {
        use bhc_hir::ConFields;

        // Build the result type: T a1 a2 ... an
        let result_ty = Self::build_applied_type(data.name, &data.params);

        // Build the function type from field types to result type
        let field_types = match &con.fields {
            ConFields::Positional(tys) => tys.clone(),
            ConFields::Named(fields) => fields.iter().map(|f| f.ty.clone()).collect(),
        };

        let con_ty = field_types
            .into_iter()
            .rev()
            .fold(result_ty, |acc, field_ty| Ty::fun(field_ty, acc));

        Scheme::poly(data.params.clone(), con_ty)
    }

    /// Compute the type scheme for a newtype constructor.
    fn compute_newtype_con_scheme(&self, newtype: &NewtypeDef) -> Scheme {
        use bhc_hir::ConFields;

        // Build the result type: T a1 a2 ... an
        let result_ty = Self::build_applied_type(newtype.name, &newtype.params);

        // Newtype has exactly one field
        let field_ty = match &newtype.con.fields {
            ConFields::Positional(tys) => tys.first().cloned().unwrap_or_else(Ty::unit),
            ConFields::Named(fields) => {
                fields.first().map_or_else(Ty::unit, |f| f.ty.clone())
            }
        };

        let con_ty = Ty::fun(field_ty, result_ty);
        Scheme::poly(newtype.params.clone(), con_ty)
    }

    /// Build an applied type: T a1 a2 ... an
    fn build_applied_type(name: bhc_intern::Symbol, params: &[TyVar]) -> Ty {
        let base = Ty::Con(TyCon::new(name, Self::compute_type_con_kind(params.len())));
        params
            .iter()
            .fold(base, |acc, param| Ty::App(Box::new(acc), Box::new(Ty::Var(param.clone()))))
    }

    /// Check a binding group (potentially mutually recursive bindings).
    pub fn check_binding_group(&mut self, group: &BindingGroup) {
        match group {
            BindingGroup::NonRecursive(item) => {
                self.check_item(item);
            }
            BindingGroup::Recursive(items) => {
                // For recursive groups, first add all bindings with fresh type variables
                let mut temp_schemes: Vec<(DefId, Scheme)> = Vec::new();

                for item in items {
                    if let Item::Value(value_def) = item {
                        let ty = value_def
                            .sig
                            .as_ref()
                            .map_or_else(|| self.fresh_ty(), |scheme| scheme.ty.clone());
                        let scheme = Scheme::mono(ty);
                        temp_schemes.push((value_def.id, scheme.clone()));
                        self.env.insert_global(value_def.id, scheme);
                    }
                }

                // Now type check each item
                for item in items {
                    self.check_item(item);
                }

                // Generalize the types
                for (def_id, _) in temp_schemes {
                    if let Some(scheme) = self.def_schemes.get(&def_id) {
                        let generalized = self.generalize(&scheme.ty);
                        self.def_schemes.insert(def_id, generalized.clone());
                        self.env.insert_global(def_id, generalized);
                    }
                }
            }
        }
    }

    /// Check a single item.
    fn check_item(&mut self, item: &Item) {
        match item {
            Item::Value(value_def) => self.check_value_def(value_def),
            // These are handled separately:
            // - Data/Newtype: registered in register_data_type/register_newtype
            // - TypeAlias: handled during type resolution
            // - Class/Instance: handled separately
            // - Fixity: no type checking needed
            // - Foreign: uses declared type
            Item::Data(_)
            | Item::Newtype(_)
            | Item::TypeAlias(_)
            | Item::Class(_)
            | Item::Instance(_)
            | Item::Fixity(_)
            | Item::Foreign(_) => {}
        }
    }

    /// Check a value definition.
    fn check_value_def(&mut self, value_def: &ValueDef) {
        // If there's a type signature, use it; otherwise infer
        let declared_ty = value_def.sig.as_ref().map(|s| s.ty.clone());

        // Infer the type from equations
        let inferred_ty = self.infer_equations(&value_def.equations, value_def.span);

        // If there's a declared type, unify with inferred type
        if let Some(declared) = &declared_ty {
            self.unify(declared, &inferred_ty, value_def.span);
        }

        // Generalize and store the scheme
        let final_ty = self.apply_subst(&inferred_ty);
        let scheme = value_def
            .sig
            .as_ref()
            .map_or_else(|| self.generalize(&final_ty), Clone::clone);

        self.def_schemes.insert(value_def.id, scheme.clone());
        self.env.insert_global(value_def.id, scheme);
    }

    /// Infer the type of a list of equations (function clauses).
    fn infer_equations(&mut self, equations: &[Equation], span: Span) -> Ty {
        if equations.is_empty() {
            return self.fresh_ty();
        }

        // All equations must have the same type
        let first_ty = self.infer_equation(&equations[0]);

        for eq in equations.iter().skip(1) {
            let eq_ty = self.infer_equation(eq);
            self.unify(&first_ty, &eq_ty, span);
        }

        first_ty
    }

    /// Infer the type of a single equation.
    fn infer_equation(&mut self, equation: &Equation) -> Ty {
        // Enter a new scope for pattern bindings
        self.env.push_scope();

        // Build function type from patterns
        let mut arg_types = Vec::new();
        for pat in &equation.pats {
            let pat_ty = self.infer_pattern(pat);
            arg_types.push(pat_ty);
        }

        // Infer RHS type
        let rhs_ty = self.infer_expr(&equation.rhs);

        // Exit scope
        self.env.pop_scope();

        // Build function type: arg1 -> arg2 -> ... -> result
        arg_types
            .into_iter()
            .rev()
            .fold(rhs_ty, |acc, arg_ty| Ty::fun(arg_ty, acc))
    }

    /// Emit a diagnostic error.
    pub fn emit_error(&mut self, diagnostic: Diagnostic) {
        self.diag.emit(diagnostic);
    }

    /// Check if any errors have been emitted.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.diag.has_errors()
    }

    /// Take all diagnostics.
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.diag.take_diagnostics()
    }

    /// Create a `FullSpan` from a Span.
    #[must_use]
    pub const fn full_span(&self, span: Span) -> FullSpan {
        FullSpan::new(self.file_id, span)
    }

    /// Convert the context into a `TypedModule`.
    #[must_use]
    pub fn into_typed_module(self, hir: Module) -> TypedModule {
        // Apply final substitution to all expression types
        let expr_types = self
            .expr_types
            .into_iter()
            .map(|(id, ty)| (id, self.subst.apply(&ty)))
            .collect();

        // Apply final substitution to all definition schemes
        let def_schemes = self
            .def_schemes
            .into_iter()
            .map(|(id, scheme)| {
                let applied_ty = self.subst.apply(&scheme.ty);
                (id, Scheme {
                    vars: scheme.vars,
                    constraints: scheme.constraints,
                    ty: applied_ty,
                })
            })
            .collect();

        TypedModule {
            hir,
            expr_types,
            def_schemes,
        }
    }
}

// Forward declarations for methods implemented in other modules
impl TyCtxt {
    /// Unify two types (implemented in unify.rs).
    pub fn unify(&mut self, t1: &Ty, t2: &Ty, span: Span) {
        crate::unify::unify(self, t1, t2, span);
    }

    /// Instantiate a type scheme (implemented in instantiate.rs).
    pub fn instantiate(&mut self, scheme: &Scheme) -> Ty {
        crate::instantiate::instantiate(self, scheme)
    }

    /// Generalize a type (implemented in generalize.rs).
    #[must_use]
    pub fn generalize(&self, ty: &Ty) -> Scheme {
        crate::generalize::generalize(self, ty)
    }

    /// Infer the type of an expression (implemented in infer.rs).
    pub fn infer_expr(&mut self, expr: &bhc_hir::Expr) -> Ty {
        crate::infer::infer_expr(self, expr)
    }

    /// Infer the type of a pattern (implemented in pattern.rs).
    pub fn infer_pattern(&mut self, pat: &Pat) -> Ty {
        crate::pattern::infer_pattern(self, pat)
    }

    /// Check a pattern against an expected type (implemented in pattern.rs).
    pub fn check_pattern(&mut self, pat: &Pat, expected: &Ty) {
        crate::pattern::check_pattern(self, pat, expected);
    }

    /// Check a binding (implemented in infer.rs).
    pub fn check_binding(&mut self, binding: &Binding) {
        crate::infer::check_binding(self, binding);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ty_var_gen() {
        let mut gen = TyVarGen::new();

        let v1 = gen.fresh();
        let v2 = gen.fresh();
        let v3 = gen.fresh();

        assert_eq!(v1.id, 0);
        assert_eq!(v2.id, 1);
        assert_eq!(v3.id, 2);
    }

    #[test]
    fn test_fresh_ty() {
        let mut ctx = TyCtxt::new(FileId::new(0));

        let ty1 = ctx.fresh_ty();
        let ty2 = ctx.fresh_ty();

        match (&ty1, &ty2) {
            (Ty::Var(v1), Ty::Var(v2)) => {
                assert_ne!(v1.id, v2.id);
            }
            _ => panic!("expected type variables"),
        }
    }
}
