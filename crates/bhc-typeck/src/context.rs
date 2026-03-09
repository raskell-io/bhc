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
    Binding, ClassDef, ConFields, DataDef, DefId, Equation, HirId, InstanceDef, Item, Module,
    NewtypeDef, Pat, ValueDef,
};
use bhc_intern::Symbol;
use bhc_span::{FileId, Span};
use bhc_types::{Constraint, Kind, Scheme, Subst, Ty, TyCon, TyVar};
use rustc_hash::FxHashMap;

use crate::binding_groups::BindingGroup;
use crate::builtins::{Builtins, BUILTIN_TYVAR_F, BUILTIN_TYVAR_M};
use crate::diagnostics;
use crate::env::{ClassInfo, InstanceInfo, TypeEnv};
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

    /// DefIds that have explicit type signatures — their schemes should not
    /// be modified by the finalization substitution step (the type variables
    /// are universally quantified, not unification variables).
    pub(crate) explicit_sig_defs: std::collections::HashSet<DefId>,

    /// Maps constructor DefId to named field definitions (name, type) pairs.
    /// Used for record construction type checking with out-of-order fields.
    pub(crate) con_field_defs: FxHashMap<DefId, Vec<(Symbol, Ty)>>,

    /// Collected type class constraints during inference.
    /// These are solved after inference completes or defaulted if ambiguous.
    pub(crate) constraints: Vec<Constraint>,

    /// Given constraints from existential pattern matches.
    /// When pattern matching on an existential constructor, the existential
    /// constraints become available as evidence in the alternative body.
    /// These suppress matching "wanted" constraints from being emitted.
    pub(crate) given_constraints: Vec<Constraint>,

    /// Whether {-# LANGUAGE OverloadedStrings #-} is enabled.
    pub(crate) overloaded_strings: bool,

    /// Whether {-# LANGUAGE OverloadedLists #-} is enabled.
    pub(crate) overloaded_lists: bool,

    /// Whether {-# LANGUAGE ScopedTypeVariables #-} is enabled.
    pub(crate) scoped_type_variables: bool,

    /// Scoped type variables: maps forall-bound variable ID → fresh unification type.
    /// When ScopedTypeVariables is enabled, forall-bound vars from explicit signatures
    /// are available in expression/pattern type annotations within the function body.
    pub(crate) scoped_type_vars: FxHashMap<u32, Ty>,

    /// Classes defined by user code (from HIR ClassDef items).
    /// Constraints for these classes flow through dict-passing.
    /// Builtin classes (Show, Eq, Monad, MonadState, etc.) are handled by codegen.
    pub(crate) user_defined_classes: rustc_hash::FxHashSet<Symbol>,

    /// Type names that are GADTs (have constructors with explicit return types).
    /// Used to enable save/restore of substitution in case expressions.
    pub(crate) gadt_types: rustc_hash::FxHashSet<Symbol>,

    /// Maps field name → Vec<(constructor DefId, accessor DefId)>.
    /// Used for FieldAccess to find which constructor a field belongs to
    /// and look up the accessor function's type scheme.
    pub(crate) field_name_to_con: FxHashMap<Symbol, Vec<(DefId, DefId)>>,

    /// Maps type constructor name → Vec<DefId> (constructor DefIds).
    /// Used for RecordUpdate to find constructors of the record type.
    pub(crate) type_to_data_cons: FxHashMap<Symbol, Vec<DefId>>,

    /// User-defined type aliases: maps alias name → (params, rhs type).
    /// Used during unification to expand type aliases transparently.
    /// E.g., `type Fallible a = Either Failure a` stores
    /// `"Fallible" → ([a], Either Failure a)`.
    pub(crate) type_aliases: FxHashMap<Symbol, (Vec<bhc_types::TyVar>, Ty)>,
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
            explicit_sig_defs: std::collections::HashSet::new(),
            con_field_defs: FxHashMap::default(),
            constraints: Vec::new(),
            given_constraints: Vec::new(),
            overloaded_strings: false,
            overloaded_lists: false,
            scoped_type_variables: false,
            scoped_type_vars: FxHashMap::default(),
            user_defined_classes: rustc_hash::FxHashSet::default(),
            gadt_types: rustc_hash::FxHashSet::default(),
            field_name_to_con: FxHashMap::default(),
            type_to_data_cons: FxHashMap::default(),
            type_aliases: FxHashMap::default(),
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

    /// Push scoped type variables from a signature's forall-bound vars.
    ///
    /// When ScopedTypeVariables is enabled, forall-bound type variables from
    /// an explicit signature are available in expression/pattern annotations
    /// within the function body.
    ///
    /// `subst` maps the scheme's bound variable IDs → fresh unification types.
    pub(crate) fn push_scoped_type_vars(&mut self, subst: &FxHashMap<u32, Ty>) {
        for (var_id, ty) in subst {
            self.scoped_type_vars.insert(*var_id, ty.clone());
        }
    }

    /// Remove scoped type variables (when exiting function scope).
    pub(crate) fn pop_scoped_type_vars(&mut self, var_ids: &[u32]) {
        for var_id in var_ids {
            self.scoped_type_vars.remove(var_id);
        }
    }

    /// Resolve scoped type variables in a type.
    ///
    /// Walks the type and replaces any `Ty::Var(v)` where `v.id` matches
    /// a scoped type variable with the corresponding fresh unification type.
    pub(crate) fn resolve_scoped_type_vars(&self, ty: &Ty) -> Ty {
        if self.scoped_type_vars.is_empty() {
            return ty.clone();
        }
        crate::instantiate::substitute(ty, &self.scoped_type_vars)
    }

    /// Emit a type class constraint.
    ///
    /// Constraints are collected during type inference and solved later.
    /// For example, `emit_constraint("Num", ty, span)` records that `ty`
    /// must have a `Num` instance.
    pub fn emit_constraint(&mut self, class: Symbol, ty: Ty, span: bhc_span::Span) {
        self.constraints.push(Constraint::new(class, ty, span));
    }

    /// Emit a type class constraint with multiple type arguments.
    ///
    /// Used for multi-parameter type classes and for emitting constraints
    /// from type schemes during instantiation.
    pub fn emit_constraint_multi(
        &mut self,
        class: Symbol,
        args: Vec<Ty>,
        span: bhc_span::Span,
    ) {
        // Check if this constraint is satisfied by a given constraint
        // from an existential pattern match. If so, don't emit it as wanted.
        if self.is_satisfied_by_given(class, &args) {
            return;
        }
        self.constraints
            .push(Constraint::new_multi(class, args, span));
    }

    /// Check if a constraint is satisfied by a given constraint from an
    /// existential pattern match.
    fn is_satisfied_by_given(&self, class: Symbol, args: &[Ty]) -> bool {
        let resolved_args: Vec<Ty> = args.iter().map(|t| self.subst.apply(t)).collect();
        self.given_constraints.iter().any(|gc| {
            gc.class == class
                && gc.args.len() == resolved_args.len()
                && gc.args.iter().zip(resolved_args.iter()).all(|(a, b)| {
                    let ga = self.subst.apply(a);
                    ga == *b
                })
        })
    }

    /// Add a given constraint from an existential pattern match.
    pub fn push_given_constraint(&mut self, class: Symbol, args: Vec<Ty>, span: bhc_span::Span) {
        self.given_constraints
            .push(Constraint::new_multi(class, args, span));
    }

    /// Get the current number of given constraints (for scoped save/restore).
    pub fn given_constraints_len(&self) -> usize {
        self.given_constraints.len()
    }

    /// Restore given constraints to a previous length (scoped pop).
    pub fn restore_given_constraints(&mut self, len: usize) {
        self.given_constraints.truncate(len);
    }

    /// Check if a class was defined by user code (not a builtin).
    ///
    /// Only user-defined class constraints are emitted during instantiation
    /// for dict-passing. Builtin classes (Show, Eq, Monad, MonadState, etc.)
    /// are handled by codegen and their constraints are not emitted.
    pub fn is_user_defined_class(&self, class: Symbol) -> bool {
        self.user_defined_classes.contains(&class)
    }

    /// Apply the current substitution to all collected constraints.
    pub fn apply_subst_to_constraints(&mut self) {
        self.constraints = self
            .constraints
            .iter()
            .map(|c| Constraint {
                class: c.class,
                args: c.args.iter().map(|t| self.subst.apply(t)).collect(),
                span: c.span,
            })
            .collect();
    }

    /// Propagate superclass constraints.
    ///
    /// For each constraint `C ty` where class C has superclasses S1, S2, ...,
    /// this generates additional constraints `S1 ty`, `S2 ty`, etc.
    ///
    /// For example, if we have `Ord a` and `Ord` has superclass `Eq`,
    /// this will also generate `Eq a`.
    fn propagate_superclass_constraints(&mut self) {
        let mut new_constraints = Vec::new();
        let mut seen = rustc_hash::FxHashSet::default();

        // Track which (class, type) pairs we've already processed
        for c in &self.constraints {
            seen.insert((c.class, c.args.clone()));
        }

        // Process constraints and generate superclass constraints
        let mut worklist: Vec<Constraint> = self.constraints.clone();

        while let Some(constraint) = worklist.pop() {
            // Look up the class to get its superclasses
            let supers = match self.env.lookup_class(constraint.class) {
                Some(info) => info.supers.clone(),
                None => continue, // Unknown class, skip
            };

            // Generate a constraint for each superclass
            for super_class in supers {
                let key = (super_class, constraint.args.clone());
                if !seen.contains(&key) {
                    seen.insert(key);
                    let new_constraint = Constraint {
                        class: super_class,
                        args: constraint.args.clone(),
                        span: constraint.span,
                    };
                    new_constraints.push(new_constraint.clone());
                    worklist.push(new_constraint); // Process transitively
                }
            }
        }

        // Add all new constraints
        self.constraints.extend(new_constraints);
    }

    /// Solve collected constraints using instance resolution.
    ///
    /// For each constraint:
    /// 1. Apply current substitution to get the concrete type
    /// 2. Propagate superclass constraints
    /// 3. Apply functional dependency improvement
    /// 4. Look up an instance for the class and type
    /// 5. If found, the constraint is satisfied
    /// 6. If not found and type is a variable, try defaulting
    /// 7. If not found and type is concrete, emit an error
    pub fn solve_constraints(&mut self) {
        // First apply substitution to all constraints
        self.apply_subst_to_constraints();

        // Propagate superclass constraints: if we have `Ord a`, also generate `Eq a`
        self.propagate_superclass_constraints();

        // Apply functional dependency improvement
        // This may add new substitutions based on fundeps
        self.apply_fundep_improvement();

        // Re-apply substitution after fundep improvement
        self.apply_subst_to_constraints();

        // Take constraints to avoid borrow conflicts
        let constraints = std::mem::take(&mut self.constraints);

        // Collect results: (constraint, needs_error)
        let mut results: Vec<(Constraint, bool)> = Vec::new();

        for constraint in constraints {
            if constraint.args.is_empty() {
                continue;
            }

            // Apply substitution to all constraint arguments
            let args: Vec<Ty> = constraint.args.iter().map(|t| self.subst.apply(t)).collect();

            // Try to solve the constraint
            if self.try_solve_constraint_recursive(&constraint.class, &args, 0) {
                continue;
            }

            // Check for built-in numeric types that satisfy common classes (single-arg only)
            if args.len() == 1 && self.is_builtin_instance(constraint.class, &args[0]) {
                continue;
            }

            // If all args are type variables, try defaulting (single-arg only)
            if args.len() == 1 {
                if let Ty::Var(ref v) = args[0] {
                    if self.try_default_constraint(constraint.class, v) {
                        continue;
                    }
                }
            }

            // Could not solve - record for later error reporting
            results.push((constraint, true));
        }

        // Emit errors for unsolved constraints
        for (constraint, _needs_error) in results {
            let ty = self.subst.apply(&constraint.args[0]);
            diagnostics::emit_no_instance(self, constraint.class, &ty, constraint.span);
        }
    }

    /// Solve constraints accumulated since `start_idx`, returning unsolved ones.
    ///
    /// Unlike `solve_constraints()`, this does NOT emit errors for unsolved
    /// constraints. Instead, it returns them so they can be incorporated into
    /// a type scheme via `generalize_with_constraints()`.
    ///
    /// Constraints before `start_idx` are left untouched.
    pub fn solve_constraints_partition(&mut self, start_idx: usize) -> Vec<Constraint> {
        // Apply functional dependency improvement before solving.
        // This may add new substitutions based on fundeps (e.g., `a -> b` in
        // `class C a b | a -> b` determines `b` when `a` is known).
        self.apply_fundep_improvement();

        // Apply substitution to the new constraints (including any from fundep improvement)
        for c in self.constraints[start_idx..].iter_mut() {
            c.args = c.args.iter().map(|t| self.subst.apply(t)).collect();
        }

        // Extract only the new constraints (leave earlier ones in place)
        let new_constraints: Vec<Constraint> = self.constraints.drain(start_idx..).collect();

        let mut unsolved = Vec::new();

        for constraint in new_constraints {
            if constraint.args.is_empty() {
                continue;
            }

            // Apply substitution to all constraint arguments
            let args: Vec<Ty> = constraint
                .args
                .iter()
                .map(|t| self.subst.apply(t))
                .collect();

            // Try to solve the constraint
            if self.try_solve_constraint_recursive(&constraint.class, &args, 0) {
                continue;
            }

            // Check for built-in instances (single-arg only)
            if args.len() == 1 && self.is_builtin_instance(constraint.class, &args[0]) {
                continue;
            }

            // If all args are type variables, try defaulting (single-arg only)
            if args.len() == 1 {
                if let Ty::Var(ref v) = args[0] {
                    if self.try_default_constraint(constraint.class, v) {
                        continue;
                    }
                }
            }

            // Check if the constraint is on concrete types (no type variables).
            // If so, it's an unsolvable error, not a deferred constraint.
            let has_type_vars = args.iter().any(|t| !t.free_vars().is_empty());
            if !has_type_vars {
                // Concrete constraint that can't be solved — this is a type error
                // (e.g., Num Bool from `if 42 then ...`)
                let args_str = args
                    .iter()
                    .map(|t| format!("{:?}", t))
                    .collect::<Vec<_>>()
                    .join(" ");
                self.diag.emit(Diagnostic::error(format!(
                    "No instance for `{} {}`",
                    constraint.class.as_str(),
                    args_str
                )));
                continue;
            }

            // Constraint involves type variables and can't be resolved yet —
            // return it for generalization into the type scheme
            unsolved.push(Constraint::new_multi(
                constraint.class,
                args,
                constraint.span,
            ));
        }

        unsolved
    }

    /// Try to solve a constraint recursively, checking instance contexts.
    ///
    /// For multi-parameter typeclasses like `MonadReader r m`, this resolves
    /// instances and recursively checks their context constraints.
    ///
    /// For example, to solve `MonadReader String (StateT Int (ReaderT String IO))`:
    /// 1. Find instance `MonadReader r m => MonadReader r (StateT s m)`
    /// 2. Match: r = String, s = Int, m = ReaderT String IO
    /// 3. Recursively solve context: `MonadReader String (ReaderT String IO)`
    /// 4. Find direct instance `Monad m => MonadReader r (ReaderT r m)`
    /// 5. Match: r = String, m = IO
    /// 6. Recursively solve context: `Monad IO` (builtin - satisfied)
    fn try_solve_constraint_recursive(
        &self,
        class: &Symbol,
        args: &[Ty],
        depth: usize,
    ) -> bool {
        // Prevent infinite recursion
        if depth > 50 {
            return false;
        }

        // Handle single-argument constraints with existing resolve_instance
        if args.len() == 1 {
            if self.env.resolve_instance(*class, &args[0]).is_some() {
                return true;
            }
            // Also check built-in instances
            if self.is_builtin_instance(*class, &args[0]) {
                return true;
            }
        }

        // Check for built-in multi-argument instances (IO is always a Monad)
        if self.is_builtin_instance_multi(*class, args) {
            return true;
        }

        // Try to resolve with multi-arg instance matching
        if let Some((inst, subst)) = self.env.resolve_instance_multi(*class, args) {
            // Found a matching instance. Now check the instance context.
            // Each context constraint must also be satisfiable.
            for ctx in &inst.context {
                let specialized_args: Vec<Ty> = ctx.args.iter().map(|t| subst.apply(t)).collect();
                if !self.try_solve_constraint_recursive(&ctx.class, &specialized_args, depth + 1) {
                    return false;
                }
            }
            return true;
        }

        false
    }

    /// Check if a multi-argument constraint has a built-in instance.
    fn is_builtin_instance_multi(&self, class: Symbol, args: &[Ty]) -> bool {
        let class_name = class.as_str();

        match class_name {
            // E.63: Generic/NFData satisfied by any concrete type
            "Generic" | "NFData" if args.len() == 1 => {
                return !matches!(&args[0], Ty::Var(_));
            }
            // Monad IO is always satisfied; also State, Reader, Writer, RWS
            "Monad" if args.len() == 1 => {
                if let Ty::Con(tycon) = &args[0] {
                    if tycon.name.as_str() == "IO" {
                        return true;
                    }
                }
                // Check for partially applied type constructors: State s, Reader r, Writer w, RWS r w s, etc.
                if let Ty::App(f, _) = &args[0] {
                    if let Ty::Con(tycon) = f.as_ref() {
                        let name = tycon.name.as_str();
                        if matches!(name, "State" | "StateT" | "Reader" | "ReaderT"
                            | "Writer" | "WriterT" | "RWS" | "RWST" | "ST"
                            | "Identity" | "Maybe" | "Either") {
                            return true;
                        }
                    }
                    // Check for nested applications like StateT s IO, ReaderT r IO, etc.
                    if let Ty::App(ff, _) = f.as_ref() {
                        if let Ty::Con(tycon) = ff.as_ref() {
                            let name = tycon.name.as_str();
                            if matches!(name, "StateT" | "ReaderT" | "WriterT" | "RWST"
                                | "ExceptT" | "Either" | "RWS") {
                                return true;
                            }
                        }
                    }
                }
                // Also check nested transformer applications that bottom out in IO
                self.monad_bottoms_out_in_io(&args[0])
            }
            // Monoid instances
            "Monoid" if args.len() == 1 => {
                // [a] is a Monoid
                if let Ty::List(_) = &args[0] {
                    return true;
                }
                // String is a Monoid
                if let Ty::Con(tycon) = &args[0] {
                    if tycon.name.as_str() == "String" {
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Check if a monad type bottoms out in IO (e.g., StateT s IO, ReaderT r (StateT s IO), etc.)
    fn monad_bottoms_out_in_io(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Con(tycon) if tycon.name.as_str() == "IO" => true,
            Ty::App(f, _) => {
                // Check if this is a transformer application T x m
                // We need to recursively check m
                match f.as_ref() {
                    Ty::App(_, inner_m) => self.monad_bottoms_out_in_io(inner_m),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Check if a type has a built-in instance for a class.
    fn is_builtin_instance(&self, class: Symbol, ty: &Ty) -> bool {
        let class_name = class.as_str();

        // E.63: Generic and NFData are satisfied by any concrete type
        // (Generic is a stub class; NFData is a no-op in strict runtime)
        if matches!(class_name, "Generic" | "NFData") {
            return !matches!(ty, Ty::Var(_));
        }

        // In check mode, be permissive for IsString, Semigroup, Monoid, and Hashable
        // on any concrete type constructor. This handles type aliases like
        // MimeType = Text, FilePath = String, Extensions, etc. that may have
        // derived or newtype-derived instances that BHC cannot resolve.
        if matches!(class_name, "IsString" | "Semigroup" | "Monoid" | "Hashable") {
            match ty {
                Ty::Con(_) | Ty::App(_, _) | Ty::List(_) | Ty::Tuple(_) => return true,
                _ => {}
            }
        }

        match ty {
            Ty::Con(tycon) => {
                let type_name = tycon.name.as_str();
                // Normalize qualified type names (e.g., "T.Text" -> "Text")
                let normalized = if type_name.contains('.') {
                    type_name.rsplit('.').next().unwrap_or(type_name)
                } else {
                    type_name
                };
                matches!(
                    (class_name, normalized),
                    // Num instances
                    ("Num", "Int") | ("Num", "Float") | ("Num", "Double") | ("Num", "Integer") |
                    ("Num", "Word") | ("Num", "Word8") | ("Num", "Word16") | ("Num", "Word32") | ("Num", "Word64") |
                    ("Num", "RowSpan") | ("Num", "ColSpan") | ("Num", "RowHeadColumns") |
                    ("Num", "RowNumber") | ("Num", "ColNumber") |
                    // Eq instances
                    ("Eq", "Int") | ("Eq", "Float") | ("Eq", "Double") | ("Eq", "Bool") |
                    ("Eq", "Char") | ("Eq", "String") | ("Eq", "Integer") |
                    ("Eq", "Text") | ("Eq", "ByteString") |
                    ("Eq", "Word") | ("Eq", "Word8") | ("Eq", "Word16") | ("Eq", "Word32") | ("Eq", "Word64") |
                    ("Eq", "RowSpan") | ("Eq", "ColSpan") | ("Eq", "RowHeadColumns") |
                    ("Eq", "RowNumber") | ("Eq", "ColNumber") |
                    ("Eq", "Alignment") | ("Eq", "ColWidth") |
                    ("Eq", "Inline") | ("Eq", "Block") | ("Eq", "Pandoc") | ("Eq", "Meta") |
                    ("Eq", "MetaValue") | ("Eq", "Format") | ("Eq", "QuoteType") | ("Eq", "MathType") |
                    ("Eq", "ListNumberStyle") | ("Eq", "ListNumberDelim") | ("Eq", "CitationMode") |
                    ("Eq", "QName") | ("Eq", "Attr") | ("Eq", "Element") | ("Eq", "Content") |
                    ("Eq", "CData") | ("Eq", "CDataKind") |
                    // Ord instances
                    ("Ord", "Int") | ("Ord", "Float") | ("Ord", "Double") | ("Ord", "Char") | ("Ord", "Integer") |
                    ("Ord", "Text") | ("Ord", "ByteString") |
                    ("Ord", "Word") | ("Ord", "Word8") | ("Ord", "Word16") | ("Ord", "Word32") | ("Ord", "Word64") |
                    ("Ord", "RowSpan") | ("Ord", "ColSpan") | ("Ord", "RowHeadColumns") |
                    ("Ord", "RowNumber") | ("Ord", "ColNumber") |
                    ("Ord", "Alignment") | ("Ord", "Inline") | ("Ord", "Block") |
                    ("Ord", "QName") | ("Ord", "Attr") | ("Ord", "Element") | ("Ord", "Content") |
                    ("Ord", "CData") | ("Ord", "CDataKind") |
                    // Show instances
                    ("Show", "Int") | ("Show", "Float") | ("Show", "Double") | ("Show", "Bool") |
                    ("Show", "Char") | ("Show", "String") | ("Show", "Integer") |
                    ("Show", "Text") | ("Show", "ByteString") |
                    ("Show", "Word") | ("Show", "Word8") | ("Show", "Word16") | ("Show", "Word32") | ("Show", "Word64") |
                    ("Show", "RowSpan") | ("Show", "ColSpan") | ("Show", "RowHeadColumns") |
                    ("Show", "Inline") | ("Show", "Block") | ("Show", "Pandoc") | ("Show", "Meta") |
                    ("Show", "Alignment") | ("Show", "ColWidth") |
                    ("Show", "QName") | ("Show", "Attr") | ("Show", "Element") | ("Show", "Content") |
                    ("Show", "CData") | ("Show", "CDataKind") |
                    // Read instances
                    ("Read", "Int") | ("Read", "Float") | ("Read", "Double") |
                    ("Read", "Char") | ("Read", "String") | ("Read", "Integer") |
                    // Fractional instances
                    ("Fractional", "Float") | ("Fractional", "Double") |
                    // IsString instances
                    ("IsString", "String") | ("IsString", "[Char]") |
                    ("IsString", "Text") | ("IsString", "ByteString") |
                    ("IsString", "Doc") |
                    // IsList instances (OverloadedLists)
                    ("IsList", "[]") |
                    // Enum/Bounded instances for Word types and Integer
                    ("Enum", "Integer") |
                    ("Enum", "RowSpan") | ("Enum", "ColSpan") | ("Enum", "RowHeadColumns") |
                    ("Enum", "Word") | ("Enum", "Word8") | ("Enum", "Word16") | ("Enum", "Word32") | ("Enum", "Word64") |
                    ("Bounded", "Word") | ("Bounded", "Word8") | ("Bounded", "Word16") | ("Bounded", "Word32") | ("Bounded", "Word64") |
                    // Integral instances
                    ("Integral", "Integer") |
                    ("Integral", "Word") | ("Integral", "Word8") | ("Integral", "Word16") | ("Integral", "Word32") | ("Integral", "Word64")
                )
            }
            // List instances: Eq [a], Ord [a], Show [a] if element type has the instance
            // IsString [Char] is always valid (String = [Char])
            Ty::List(elem) => {
                if class_name == "IsString" {
                    matches!(elem.as_ref(), Ty::Con(tc) if tc.name.as_str() == "Char")
                } else if class_name == "IsList" {
                    true // [a] is always an IsList instance
                } else {
                    matches!(class_name, "Eq" | "Ord" | "Show" | "Read") && self.is_builtin_instance(class, elem)
                }
            }
            // Tuple instances: Eq (a, b), Ord (a, b), Show (a, b), Read (a, b) if all elements have the instance
            Ty::Tuple(elems) => {
                matches!(class_name, "Eq" | "Ord" | "Show" | "Read")
                    && elems
                        .iter()
                        .all(|elem| self.is_builtin_instance(class, elem))
            }
            // Maybe instances: handled via App pattern
            // Either instances: handled via App pattern
            Ty::App(con, arg) => {
                // Check for Maybe a, Either a b, etc.
                if let Ty::Con(tycon) = con.as_ref() {
                    let type_name = tycon.name.as_str();
                    match (class_name, type_name) {
                        // Maybe a has Eq, Ord, Show, Read if a does
                        ("Eq", "Maybe") | ("Ord", "Maybe") | ("Show", "Maybe") | ("Read", "Maybe") => {
                            self.is_builtin_instance(class, arg)
                        }
                        // IO a has Show (for debugging)
                        ("Show", "IO") => true,
                        // Doc a is an IsString instance
                        ("IsString", "Doc") => true,
                        _ => false,
                    }
                } else if let Ty::App(inner_con, inner_arg) = con.as_ref() {
                    // Handle Either a b (nested application)
                    if let Ty::Con(tycon) = inner_con.as_ref() {
                        let type_name = tycon.name.as_str();
                        match (class_name, type_name) {
                            // Either a b has Eq, Ord, Show if both a and b do
                            ("Eq", "Either") | ("Ord", "Either") | ("Show", "Either") => {
                                self.is_builtin_instance(class, inner_arg)
                                    && self.is_builtin_instance(class, arg)
                            }
                            _ => false,
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            // Type variables can satisfy constraints in some contexts
            // (deferred to constraint solving)
            Ty::Var(_) => false,
            _ => false,
        }
    }

    /// Try to default a type variable to satisfy a constraint.
    ///
    /// Following Haskell's defaulting rules:
    /// - Num defaults to Int (or Integer in standard Haskell)
    /// - Fractional defaults to Double
    fn try_default_constraint(&mut self, class: Symbol, var: &TyVar) -> bool {
        let class_name = class.as_str();

        let default_ty = match class_name {
            "Num" | "Integral" | "Enum" | "Bounded" => {
                // Default to Int
                Some(self.builtins.int_ty.clone())
            }
            "Fractional" | "Floating" | "RealFrac" | "RealFloat" => {
                // Default to Double/Float
                Some(self.builtins.float_ty.clone())
            }
            "IsString" => {
                // Default to String ([Char]), matching GHC behavior
                Some(self.builtins.string_ty.clone())
            }
            "IsList" => {
                // Default to list type, matching GHC behavior
                // Create [Int] as a reasonable default, but the element
                // type will be constrained by the literal elements anyway
                Some(Ty::List(Box::new(self.builtins.int_ty.clone())))
            }
            "Eq" | "Ord" | "Show" | "Read" => {
                // These don't have defaults - leave ambiguous
                None
            }
            _ => None,
        };

        if let Some(ty) = default_ty {
            self.subst.insert(var, ty);
            true
        } else {
            false
        }
    }

    /// Apply functional dependency improvement to constraints.
    ///
    /// For each constraint `C t1 t2 ... tn` where class C has fundeps:
    /// - For each fundep `from_indices -> to_indices`:
    ///   - If all types at `from_indices` are ground (no unresolved type vars)
    ///   - Find instances where those types match
    ///   - Unify the constraint's types at `to_indices` with the instance's types
    ///
    /// This allows type inference to propagate information through fundeps.
    /// For example, if we have:
    ///   class Convert a b | a -> b
    ///   instance Convert Int String
    /// And constraint `Convert Int ?x`, fundep improvement will unify `?x` with `String`.
    fn apply_fundep_improvement(&mut self) {
        // Iterate until no more improvements can be made
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100; // Prevent infinite loops

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            // Apply current substitution to constraints
            self.apply_subst_to_constraints();

            // Clone constraints to avoid borrow issues
            let constraints = self.constraints.clone();

            for constraint in &constraints {
                // Look up the class to get its fundeps
                let class_info = match self.env.lookup_class(constraint.class) {
                    Some(info) => info.clone(),
                    None => continue, // Unknown class, skip
                };

                if class_info.fundeps.is_empty() {
                    continue; // No fundeps for this class
                }

                // Get the constraint's type arguments (after substitution)
                let args: Vec<Ty> = constraint
                    .args
                    .iter()
                    .map(|t| self.subst.apply(t))
                    .collect();

                // For each fundep, check if we can improve
                for fundep in &class_info.fundeps {
                    // Check if all "from" types are ground
                    let from_types: Vec<&Ty> =
                        fundep.from.iter().filter_map(|&i| args.get(i)).collect();

                    if from_types.iter().any(|t| self.contains_unresolved_var(t)) {
                        continue; // Not all "from" types are ground yet
                    }

                    // Find matching instances
                    if let Some(instances) = self.env.lookup_instances(constraint.class) {
                        for inst in instances.clone() {
                            // Check if this instance matches on the "from" positions
                            let inst_matches = fundep.from.iter().all(|&i| {
                                match (args.get(i), inst.types.get(i)) {
                                    (Some(arg_ty), Some(inst_ty)) => {
                                        self.types_match_for_fundep(arg_ty, inst_ty)
                                    }
                                    _ => false,
                                }
                            });

                            if inst_matches {
                                // Unify the "to" positions
                                for &to_idx in &fundep.to {
                                    if let (Some(arg_ty), Some(inst_ty)) =
                                        (args.get(to_idx), inst.types.get(to_idx))
                                    {
                                        // If arg_ty is a type variable, unify it with inst_ty
                                        if let Ty::Var(v) = arg_ty {
                                            if !self.subst.contains(v) {
                                                self.subst.insert(v, inst_ty.clone());
                                                changed = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check if a type contains any unresolved type variables.
    fn contains_unresolved_var(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Var(v) => !self.subst.contains(v),
            Ty::Con(_) | Ty::Prim(_) | Ty::Error | Ty::Nat(_) | Ty::TyList(_) => false,
            Ty::App(f, a) => self.contains_unresolved_var(f) || self.contains_unresolved_var(a),
            Ty::Fun(a, b) => self.contains_unresolved_var(a) || self.contains_unresolved_var(b),
            Ty::Tuple(elems) => elems.iter().any(|t| self.contains_unresolved_var(t)),
            Ty::List(elem) => self.contains_unresolved_var(elem),
            Ty::Forall(_, body) => self.contains_unresolved_var(body),
        }
    }

    /// Check if two types match for fundep purposes.
    /// This is a structural match that treats type constructors as equal if names match.
    fn types_match_for_fundep(&self, ty1: &Ty, ty2: &Ty) -> bool {
        match (ty1, ty2) {
            (Ty::Con(c1), Ty::Con(c2)) => c1.name == c2.name,
            (Ty::App(f1, a1), Ty::App(f2, a2)) => {
                self.types_match_for_fundep(f1, f2) && self.types_match_for_fundep(a1, a2)
            }
            (Ty::Fun(a1, b1), Ty::Fun(a2, b2)) => {
                self.types_match_for_fundep(a1, a2) && self.types_match_for_fundep(b1, b2)
            }
            (Ty::Tuple(e1), Ty::Tuple(e2)) => {
                e1.len() == e2.len()
                    && e1
                        .iter()
                        .zip(e2.iter())
                        .all(|(t1, t2)| self.types_match_for_fundep(t1, t2))
            }
            (Ty::List(e1), Ty::List(e2)) => self.types_match_for_fundep(e1, e2),
            (Ty::Prim(p1), Ty::Prim(p2)) => p1 == p2,
            (Ty::Nat(n1), Ty::Nat(n2)) => n1 == n2,
            _ => false,
        }
    }

    /// Build the Pandoc `Attr` type: `(Text, [Text], [(Text, Text)])`.
    fn pandoc_attr_ty(&self) -> Ty {
        let text = self.builtins.text_ty.clone();
        Ty::Tuple(vec![
            text.clone(),
            Ty::List(Box::new(text.clone())),
            Ty::List(Box::new(Ty::Tuple(vec![text.clone(), text]))),
        ])
    }

    /// Build the Pandoc `ColSpec` type: `(Alignment, ColWidth)`.
    fn pandoc_colspec_ty(&self) -> Ty {
        Ty::Tuple(vec![
            Ty::Con(TyCon::new(Symbol::intern("Alignment"), Kind::Star)),
            Ty::Con(TyCon::new(Symbol::intern("ColWidth"), Kind::Star)),
        ])
    }

    /// Register built-in types in the environment.
    pub fn register_builtins(&mut self) {
        self.builtins = Builtins::new();

        // Register type constructors
        self.env.register_type_con(self.builtins.int_con.clone());
        self.env.register_type_con(self.builtins.float_con.clone());
        self.env.register_type_con(self.builtins.char_con.clone());
        self.env.register_type_con(self.builtins.bool_con.clone());
        self.env.register_type_con(self.builtins.string_con.clone());
        self.env.register_type_con(self.builtins.list_con.clone());
        self.env.register_type_con(self.builtins.maybe_con.clone());
        self.env.register_type_con(self.builtins.either_con.clone());
        self.env.register_type_con(self.builtins.io_con.clone());

        // Register Integer type constructor
        self.env
            .register_type_con(self.builtins.integer_con.clone());

        // Register Word type constructors
        self.env.register_type_con(self.builtins.word_con.clone());
        self.env.register_type_con(self.builtins.word8_con.clone());
        self.env.register_type_con(self.builtins.word16_con.clone());
        self.env.register_type_con(self.builtins.word32_con.clone());
        self.env.register_type_con(self.builtins.word64_con.clone());

        // Register Rational type constructor
        self.env
            .register_type_con(self.builtins.rational_con.clone());

        // Register shape-indexed tensor type constructors
        self.env.register_type_con(self.builtins.tensor_con.clone());
        self.env
            .register_type_con(self.builtins.dyn_tensor_con.clone());
        self.env
            .register_type_con(self.builtins.shape_witness_con.clone());

        // Register built-in data constructors
        self.builtins.register_data_cons(&mut self.env);

        // Register primitive operators (+, -, *, etc.)
        self.builtins.register_primitive_ops(&mut self.env);

        // Register dynamic tensor operations (toDynamic, fromDynamic, etc.)
        self.builtins.register_dyn_tensor_ops(&mut self.env);
    }

    /// Register builtins using the DefIds from the lowering pass.
    ///
    /// The lowering pass assigns DefIds to builtin functions and constructors.
    /// This method registers those builtins in the type environment with the
    /// correct DefIds so that type checking can find them.
    pub fn register_lowered_builtins(&mut self, defs: &crate::DefMap) {
        use bhc_lower::DefKind;
        use bhc_types::TyVar;

        // Type variables for polymorphic types
        let a = TyVar::new_star(0xFFFF_0000);
        let b = TyVar::new_star(0xFFFF_0001);
        let c = TyVar::new_star(0xFFFF_0002);

        // First pass: register data constructors
        for (_def_id, def_info) in defs.iter() {
            // Only process constructor kinds
            if !matches!(
                def_info.kind,
                DefKind::Constructor | DefKind::StubConstructor
            ) {
                continue;
            }

            let name = def_info.name.as_str();
            let scheme = match name {
                // Bool constructors
                "True" | "False" => Scheme::mono(self.builtins.bool_ty.clone()),

                // Maybe constructors
                "Nothing" => {
                    // Nothing :: Maybe a
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone()], maybe_a)
                }
                "Just" => {
                    // Just :: a -> Maybe a
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), maybe_a))
                }

                // Either constructors
                "Left" => {
                    // Left :: a -> Either a b
                    let either_ab = Ty::App(
                        Box::new(Ty::App(
                            Box::new(Ty::Con(self.builtins.either_con.clone())),
                            Box::new(Ty::Var(a.clone())),
                        )),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), either_ab),
                    )
                }
                "Right" => {
                    // Right :: b -> Either a b
                    let either_ab = Ty::App(
                        Box::new(Ty::App(
                            Box::new(Ty::Con(self.builtins.either_con.clone())),
                            Box::new(Ty::Var(a.clone())),
                        )),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(b.clone()), either_ab),
                    )
                }

                // List constructors
                "[]" => {
                    // [] :: [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], list_a)
                }
                ":" => {
                    // (:) :: a -> [a] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                    )
                }

                // Unit constructor
                "()" => Scheme::mono(Ty::unit()),

                // Tuple constructors
                "(,)" => {
                    // (,) :: a -> b -> (a, b)
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(
                                Ty::Var(b.clone()),
                                Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]),
                            ),
                        ),
                    )
                }
                "(,,)" => {
                    // (,,) :: a -> b -> c -> (a, b, c)
                    let c = TyVar::new_star(0xFFFF_0002);
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(
                                Ty::Var(b.clone()),
                                Ty::fun(
                                    Ty::Var(c.clone()),
                                    Ty::Tuple(vec![
                                        Ty::Var(a.clone()),
                                        Ty::Var(b.clone()),
                                        Ty::Var(c.clone()),
                                    ]),
                                ),
                            ),
                        ),
                    )
                }

                // NonEmpty constructor
                ":|" => {
                    // (:|) :: a -> [a] -> NonEmpty a
                    // For now, approximate as a -> [a] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                    )
                }

                // Text.Pandoc.Definition (pandoc-types) constructors
                // Only match these for stub constructors — user-defined constructors
                // with the same name (e.g. CodeBlock in a local module) should use
                // the generic handler based on arity.
                // Inline type and constructors
                "Str" if def_info.kind == DefKind::StubConstructor => {
                    // Str :: Text -> Inline
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con);
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), inline_ty))
                }
                "Space" | "SoftBreak" | "LineBreak" if def_info.kind == DefKind::StubConstructor => {
                    // Space :: Inline, SoftBreak :: Inline, LineBreak :: Inline
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    Scheme::mono(Ty::Con(inline_con))
                }
                "Emph" | "Underline" | "Strong" | "Strikeout" | "Superscript"
                | "Subscript" | "SmallCaps" if def_info.kind == DefKind::StubConstructor => {
                    // Emph :: [Inline] -> Inline, etc.
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con.clone());
                    let list_inline = Ty::List(Box::new(Ty::Con(inline_con)));
                    Scheme::mono(Ty::fun(list_inline, inline_ty))
                }
                "Span" if def_info.kind == DefKind::StubConstructor => {
                    // Span :: Attr -> [Inline] -> Inline
                    // Attr = (Text, [Text], [(Text, Text)])
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con.clone());
                    let list_inline = Ty::List(Box::new(Ty::Con(inline_con)));
                    let attr_ty = self.pandoc_attr_ty();
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(list_inline, inline_ty)))
                }
                "Code" if def_info.kind == DefKind::StubConstructor => {
                    // Code :: Attr -> Text -> Inline
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con);
                    let attr_ty = self.pandoc_attr_ty();
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(self.builtins.text_ty.clone(), inline_ty)))
                }
                "RawInline" if def_info.kind == DefKind::StubConstructor => {
                    // RawInline :: Format -> Text -> Inline
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con);
                    let format_ty = Ty::Con(TyCon::new(Symbol::intern("Format"), Kind::Star));
                    Scheme::mono(Ty::fun(format_ty, Ty::fun(self.builtins.text_ty.clone(), inline_ty)))
                }
                "Link" | "Image" if def_info.kind == DefKind::StubConstructor => {
                    // Link :: Attr -> [Inline] -> Target -> Inline
                    // Target = (Text, Text)
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con.clone());
                    let list_inline = Ty::List(Box::new(Ty::Con(inline_con)));
                    let attr_ty = self.pandoc_attr_ty();
                    let target_ty = Ty::Tuple(vec![self.builtins.text_ty.clone(), self.builtins.text_ty.clone()]);
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(list_inline, Ty::fun(target_ty, inline_ty))))
                }
                "Note" if def_info.kind == DefKind::StubConstructor => {
                    // Note :: [Block] -> Inline
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con);
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let list_block = Ty::List(Box::new(Ty::Con(block_con)));
                    Scheme::mono(Ty::fun(list_block, inline_ty))
                }
                "Math" if def_info.kind == DefKind::StubConstructor => {
                    // Math :: MathType -> Text -> Inline
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con);
                    let mathtype_ty = Ty::Con(TyCon::new(Symbol::intern("MathType"), Kind::Star));
                    Scheme::mono(Ty::fun(mathtype_ty, Ty::fun(self.builtins.text_ty.clone(), inline_ty)))
                }
                "Quoted" if def_info.kind == DefKind::StubConstructor => {
                    // Quoted :: QuoteType -> [Inline] -> Inline
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con.clone());
                    let list_inline = Ty::List(Box::new(Ty::Con(inline_con)));
                    let quotetype_ty = Ty::Con(TyCon::new(Symbol::intern("QuoteType"), Kind::Star));
                    Scheme::mono(Ty::fun(quotetype_ty, Ty::fun(list_inline, inline_ty)))
                }
                // Block type and constructors
                "Plain" | "Para" if def_info.kind == DefKind::StubConstructor => {
                    // Plain :: [Inline] -> Block, Para :: [Inline] -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con);
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let list_inline = Ty::List(Box::new(Ty::Con(inline_con)));
                    Scheme::mono(Ty::fun(list_inline, block_ty))
                }
                "LineBlock" if def_info.kind == DefKind::StubConstructor => {
                    // LineBlock :: [[Inline]] -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con);
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let list_list_inline = Ty::List(Box::new(Ty::List(Box::new(Ty::Con(inline_con)))));
                    Scheme::mono(Ty::fun(list_list_inline, block_ty))
                }
                "CodeBlock" if def_info.kind == DefKind::StubConstructor => {
                    // CodeBlock :: Attr -> Text -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con);
                    let attr_ty = self.pandoc_attr_ty();
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(self.builtins.text_ty.clone(), block_ty)))
                }
                "RawBlock" if def_info.kind == DefKind::StubConstructor => {
                    // RawBlock :: Format -> Text -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con);
                    let format_ty = Ty::Con(TyCon::new(Symbol::intern("Format"), Kind::Star));
                    Scheme::mono(Ty::fun(format_ty, Ty::fun(self.builtins.text_ty.clone(), block_ty)))
                }
                "BlockQuote" if def_info.kind == DefKind::StubConstructor => {
                    // BlockQuote :: [Block] -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con.clone());
                    let list_block = Ty::List(Box::new(Ty::Con(block_con)));
                    Scheme::mono(Ty::fun(list_block, block_ty))
                }
                "BulletList" if def_info.kind == DefKind::StubConstructor => {
                    // BulletList :: [[Block]] -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con.clone());
                    let list_list_block = Ty::List(Box::new(Ty::List(Box::new(Ty::Con(block_con)))));
                    Scheme::mono(Ty::fun(list_list_block, block_ty))
                }
                "Header" if def_info.kind == DefKind::StubConstructor => {
                    // Header :: Int -> Attr -> [Inline] -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con);
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let list_inline = Ty::List(Box::new(Ty::Con(inline_con)));
                    let attr_ty = self.pandoc_attr_ty();
                    Scheme::mono(Ty::fun(self.builtins.int_ty.clone(), Ty::fun(attr_ty, Ty::fun(list_inline, block_ty))))
                }
                "HorizontalRule" if def_info.kind == DefKind::StubConstructor => {
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    Scheme::mono(Ty::Con(block_con))
                }
                "Div" if def_info.kind == DefKind::StubConstructor => {
                    // Div :: Attr -> [Block] -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con.clone());
                    let list_block = Ty::List(Box::new(Ty::Con(block_con)));
                    let attr_ty = self.pandoc_attr_ty();
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(list_block, block_ty)))
                }
                "Figure" if def_info.kind == DefKind::StubConstructor => {
                    // Figure :: Attr -> Caption -> [Block] -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con.clone());
                    let list_block = Ty::List(Box::new(Ty::Con(block_con)));
                    let attr_ty = self.pandoc_attr_ty();
                    let caption_ty = Ty::Con(TyCon::new(Symbol::intern("Caption"), Kind::Star));
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(caption_ty, Ty::fun(list_block, block_ty))))
                }
                // Pandoc type constructors
                "Pandoc" if def_info.kind == DefKind::StubConstructor => {
                    // Pandoc :: Meta -> [Block] -> Pandoc
                    let pandoc_con = TyCon::new(Symbol::intern("Pandoc"), Kind::Star);
                    let pandoc_ty = Ty::Con(pandoc_con);
                    let meta_ty = Ty::Con(TyCon::new(Symbol::intern("Meta"), Kind::Star));
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let list_block = Ty::List(Box::new(Ty::Con(block_con)));
                    Scheme::mono(Ty::fun(meta_ty, Ty::fun(list_block, pandoc_ty)))
                }
                "Meta" if def_info.kind == DefKind::StubConstructor => {
                    // Meta :: Map Text MetaValue -> Meta
                    let meta_con = TyCon::new(Symbol::intern("Meta"), Kind::Star);
                    let meta_ty = Ty::Con(meta_con);
                    let metavalue_ty = Ty::Con(TyCon::new(Symbol::intern("MetaValue"), Kind::Star));
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_ty = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(self.builtins.text_ty.clone()))), Box::new(metavalue_ty));
                    Scheme::mono(Ty::fun(map_ty, meta_ty))
                }
                // Format constructor
                "Format" if def_info.kind == DefKind::StubConstructor => {
                    // Format :: Text -> Format
                    let format_con = TyCon::new(Symbol::intern("Format"), Kind::Star);
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), Ty::Con(format_con)))
                }
                // MetaValue constructors
                "MetaMap" if def_info.kind == DefKind::StubConstructor => {
                    let mv_con = TyCon::new(Symbol::intern("MetaValue"), Kind::Star);
                    let mv_ty = Ty::Con(mv_con);
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_ty = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(self.builtins.text_ty.clone()))), Box::new(mv_ty.clone()));
                    Scheme::mono(Ty::fun(map_ty, mv_ty))
                }
                "MetaList" if def_info.kind == DefKind::StubConstructor => {
                    let mv_con = TyCon::new(Symbol::intern("MetaValue"), Kind::Star);
                    let mv_ty = Ty::Con(mv_con.clone());
                    Scheme::mono(Ty::fun(Ty::List(Box::new(Ty::Con(mv_con))), mv_ty))
                }
                "MetaBool" if def_info.kind == DefKind::StubConstructor => {
                    let mv_con = TyCon::new(Symbol::intern("MetaValue"), Kind::Star);
                    Scheme::mono(Ty::fun(self.builtins.bool_ty.clone(), Ty::Con(mv_con)))
                }
                "MetaString" if def_info.kind == DefKind::StubConstructor => {
                    let mv_con = TyCon::new(Symbol::intern("MetaValue"), Kind::Star);
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), Ty::Con(mv_con)))
                }
                "MetaInlines" if def_info.kind == DefKind::StubConstructor => {
                    let mv_con = TyCon::new(Symbol::intern("MetaValue"), Kind::Star);
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::List(Box::new(Ty::Con(inline_con))), Ty::Con(mv_con)))
                }
                "MetaBlocks" if def_info.kind == DefKind::StubConstructor => {
                    let mv_con = TyCon::new(Symbol::intern("MetaValue"), Kind::Star);
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::List(Box::new(Ty::Con(block_con))), Ty::Con(mv_con)))
                }
                // RowSpan, ColSpan, RowHeadColumns newtypes (used in AnnotatedTable, GridTable)
                "RowSpan" | "ColSpan" | "RowHeadColumns" if def_info.kind == DefKind::StubConstructor => {
                    // RowSpan :: Int -> RowSpan, etc.
                    let con = TyCon::new(Symbol::intern(name), Kind::Star);
                    Scheme::mono(Ty::fun(self.builtins.int_ty.clone(), Ty::Con(con)))
                }
                // Row, Cell, etc. from pandoc-types
                "Row" if def_info.kind == DefKind::StubConstructor => {
                    // Row :: Attr -> [Cell] -> Row
                    let row_con = TyCon::new(Symbol::intern("Row"), Kind::Star);
                    let cell_con = TyCon::new(Symbol::intern("Cell"), Kind::Star);
                    let attr_ty = self.pandoc_attr_ty();
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(Ty::List(Box::new(Ty::Con(cell_con))), Ty::Con(row_con))))
                }
                "Cell" if def_info.kind == DefKind::StubConstructor => {
                    // Cell :: Attr -> Alignment -> RowSpan -> ColSpan -> [Block] -> Cell
                    let cell_con = TyCon::new(Symbol::intern("Cell"), Kind::Star);
                    let cell_ty = Ty::Con(cell_con);
                    let attr_ty = self.pandoc_attr_ty();
                    let alignment_ty = Ty::Con(TyCon::new(Symbol::intern("Alignment"), Kind::Star));
                    let rowspan_ty = Ty::Con(TyCon::new(Symbol::intern("RowSpan"), Kind::Star));
                    let colspan_ty = Ty::Con(TyCon::new(Symbol::intern("ColSpan"), Kind::Star));
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let list_block = Ty::List(Box::new(Ty::Con(block_con)));
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(alignment_ty, Ty::fun(rowspan_ty, Ty::fun(colspan_ty, Ty::fun(list_block, cell_ty))))))
                }
                // Alignment constructors
                "AlignLeft" | "AlignRight" | "AlignCenter" | "AlignDefault" if def_info.kind == DefKind::StubConstructor => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("Alignment"), Kind::Star)))
                }
                // MathType constructors
                "DisplayMath" | "InlineMath" if def_info.kind == DefKind::StubConstructor => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("MathType"), Kind::Star)))
                }
                // QuoteType constructors
                "SingleQuote" | "DoubleQuote" if def_info.kind == DefKind::StubConstructor => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("QuoteType"), Kind::Star)))
                }
                // ListNumberStyle constructors
                "DefaultStyle" | "Example" | "Decimal" | "LowerRoman" | "UpperRoman"
                | "LowerAlpha" | "UpperAlpha" if def_info.kind == DefKind::StubConstructor => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("ListNumberStyle"), Kind::Star)))
                }
                // ListNumberDelim constructors
                "DefaultDelim" | "Period" | "OneParen" | "TwoParens" if def_info.kind == DefKind::StubConstructor => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("ListNumberDelim"), Kind::Star)))
                }
                // Table constructor (the big one)
                "Table" if def_info.kind == DefKind::StubConstructor => {
                    // Table :: Attr -> Caption -> [ColSpec] -> TableHead -> [TableBody] -> TableFoot -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con);
                    let attr_ty = self.pandoc_attr_ty();
                    let caption_ty = Ty::Con(TyCon::new(Symbol::intern("Caption"), Kind::Star));
                    let colspec_ty = self.pandoc_colspec_ty();
                    let list_colspec = Ty::List(Box::new(colspec_ty));
                    let tablehead_ty = Ty::Con(TyCon::new(Symbol::intern("TableHead"), Kind::Star));
                    let tablebody_ty = Ty::Con(TyCon::new(Symbol::intern("TableBody"), Kind::Star));
                    let list_tablebody = Ty::List(Box::new(tablebody_ty));
                    let tablefoot_ty = Ty::Con(TyCon::new(Symbol::intern("TableFoot"), Kind::Star));
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(caption_ty, Ty::fun(list_colspec,
                        Ty::fun(tablehead_ty, Ty::fun(list_tablebody, Ty::fun(tablefoot_ty, block_ty)))))))
                }
                "TableHead" if def_info.kind == DefKind::StubConstructor => {
                    // TableHead :: Attr -> [Row] -> TableHead
                    let th_con = TyCon::new(Symbol::intern("TableHead"), Kind::Star);
                    let row_con = TyCon::new(Symbol::intern("Row"), Kind::Star);
                    let attr_ty = self.pandoc_attr_ty();
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(Ty::List(Box::new(Ty::Con(row_con))), Ty::Con(th_con))))
                }
                "TableBody" if def_info.kind == DefKind::StubConstructor => {
                    // TableBody :: Attr -> RowHeadColumns -> [Row] -> [Row] -> TableBody
                    let tb_con = TyCon::new(Symbol::intern("TableBody"), Kind::Star);
                    let row_con = TyCon::new(Symbol::intern("Row"), Kind::Star);
                    let attr_ty = self.pandoc_attr_ty();
                    let rhc_ty = Ty::Con(TyCon::new(Symbol::intern("RowHeadColumns"), Kind::Star));
                    let list_row = Ty::List(Box::new(Ty::Con(row_con)));
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(rhc_ty, Ty::fun(list_row.clone(), Ty::fun(list_row, Ty::Con(tb_con))))))
                }
                "TableFoot" if def_info.kind == DefKind::StubConstructor => {
                    // TableFoot :: Attr -> [Row] -> TableFoot
                    let tf_con = TyCon::new(Symbol::intern("TableFoot"), Kind::Star);
                    let row_con = TyCon::new(Symbol::intern("Row"), Kind::Star);
                    let attr_ty = self.pandoc_attr_ty();
                    Scheme::mono(Ty::fun(attr_ty, Ty::fun(Ty::List(Box::new(Ty::Con(row_con))), Ty::Con(tf_con))))
                }
                "Caption" if def_info.kind == DefKind::StubConstructor => {
                    // Caption :: Maybe [Inline] -> [Block] -> Caption
                    let caption_con = TyCon::new(Symbol::intern("Caption"), Kind::Star);
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let maybe_short = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::List(Box::new(Ty::Con(inline_con)))),
                    );
                    let list_block = Ty::List(Box::new(Ty::Con(block_con)));
                    Scheme::mono(Ty::fun(maybe_short, Ty::fun(list_block, Ty::Con(caption_con))))
                }
                // ColWidth constructors
                "ColWidth" if def_info.kind == DefKind::StubConstructor => {
                    // ColWidth :: Double -> ColWidth
                    let cw_con = TyCon::new(Symbol::intern("ColWidth"), Kind::Star);
                    let double_ty = Ty::Con(TyCon::new(Symbol::intern("Double"), Kind::Star));
                    Scheme::mono(Ty::fun(double_ty, Ty::Con(cw_con)))
                }
                "ColWidthDefault" if def_info.kind == DefKind::StubConstructor => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("ColWidth"), Kind::Star)))
                }
                // OrderedList, DefinitionList, Cite
                "OrderedList" if def_info.kind == DefKind::StubConstructor => {
                    // OrderedList :: ListAttributes -> [[Block]] -> Block
                    // ListAttributes = (Int, ListNumberStyle, ListNumberDelim)
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con.clone());
                    let list_list_block = Ty::List(Box::new(Ty::List(Box::new(Ty::Con(block_con)))));
                    let list_attrs = Ty::Tuple(vec![
                        self.builtins.int_ty.clone(),
                        Ty::Con(TyCon::new(Symbol::intern("ListNumberStyle"), Kind::Star)),
                        Ty::Con(TyCon::new(Symbol::intern("ListNumberDelim"), Kind::Star)),
                    ]);
                    Scheme::mono(Ty::fun(list_attrs, Ty::fun(list_list_block, block_ty)))
                }
                "DefinitionList" if def_info.kind == DefKind::StubConstructor => {
                    // DefinitionList :: [([Inline], [[Block]])] -> Block
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let block_ty = Ty::Con(block_con.clone());
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let item = Ty::Tuple(vec![
                        Ty::List(Box::new(Ty::Con(inline_con))),
                        Ty::List(Box::new(Ty::List(Box::new(Ty::Con(block_con))))),
                    ]);
                    Scheme::mono(Ty::fun(Ty::List(Box::new(item)), block_ty))
                }
                "Cite" if def_info.kind == DefKind::StubConstructor => {
                    // Cite :: [Citation] -> [Inline] -> Inline
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let inline_ty = Ty::Con(inline_con.clone());
                    let list_inline = Ty::List(Box::new(Ty::Con(inline_con)));
                    let citation_ty = Ty::Con(TyCon::new(Symbol::intern("Citation"), Kind::Star));
                    Scheme::mono(Ty::fun(Ty::List(Box::new(citation_ty)), Ty::fun(list_inline, inline_ty)))
                }
                // CitationMode constructors
                "AuthorInText" | "SuppressAuthor" | "NormalCitation" if def_info.kind == DefKind::StubConstructor => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("CitationMode"), Kind::Star)))
                }
                // Many newtype (used by Blocks/Inlines)
                "Many" if def_info.kind == DefKind::StubConstructor => {
                    // Many :: Seq a -> Many a
                    let a = TyVar::new_star(0xFFFF_0000);
                    let many_con = TyCon::new(Symbol::intern("Many"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    let seq_con = TyCon::new(Symbol::intern("Seq"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::App(Box::new(Ty::Con(seq_con)), Box::new(Ty::Var(a.clone()))),
                        Ty::App(Box::new(Ty::Con(many_con)), Box::new(Ty::Var(a))),
                    ))
                }
                // RowNumber / ColNumber newtypes
                "RowNumber" | "ColNumber" if def_info.kind == DefKind::StubConstructor => {
                    let con = TyCon::new(Symbol::intern(name), Kind::Star);
                    Scheme::mono(Ty::fun(self.builtins.int_ty.clone(), Ty::Con(con)))
                }
                // Data.Monoid newtypes: Any, All
                "Any" if def_info.kind == DefKind::StubConstructor => {
                    let any_con = TyCon::new(Symbol::intern("Any"), Kind::Star);
                    Scheme::mono(Ty::fun(self.builtins.bool_ty.clone(), Ty::Con(any_con)))
                }
                "All" if def_info.kind == DefKind::StubConstructor => {
                    let all_con = TyCon::new(Symbol::intern("All"), Kind::Star);
                    Scheme::mono(Ty::fun(self.builtins.bool_ty.clone(), Ty::Con(all_con)))
                }
                "Null" if def_info.kind == DefKind::StubConstructor => {
                    // Null :: Block (a no-content block)
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("Block"), Kind::Star)))
                }

                // For imported constructors that aren't known builtins,
                // create a function type based on the constructor's arity.
                _ => {
                    if let (Some(arity), Some(type_con_name), Some(type_param_count)) = (
                        def_info.arity,
                        def_info.type_con_name,
                        def_info.type_param_count,
                    ) {
                        // Build proper polymorphic type: forall a1 .. an b1 .. bm. b1 -> b2 -> ... -> bm -> TypeCon a1 .. an
                        // where a1..an are result type params and b1..bm are field type params

                        // Create result type parameters (a1 .. an)
                        let result_type_params: Vec<TyVar> = (0..type_param_count)
                            .map(|i| TyVar::new_star(0xFFFE_0000 + i as u32))
                            .collect();

                        // Create field type parameters (b1 .. bm) - these must also be quantified
                        let field_type_params: Vec<TyVar> = (0..arity)
                            .map(|i| TyVar::new_star(0xFFFF_0000 + i as u32))
                            .collect();

                        // Build the result type: TypeCon a1 a2 ... an
                        let kind = Self::compute_type_con_kind(type_param_count);
                        let type_con = TyCon::new(type_con_name, kind);
                        let result_ty = result_type_params
                            .iter()
                            .fold(Ty::Con(type_con), |acc, param| {
                                Ty::App(Box::new(acc), Box::new(Ty::Var(param.clone())))
                            });

                        // Build the constructor type: b1 -> b2 -> ... -> bm -> ResultType
                        // Build from inside out: result <- bm <- bm-1 <- ... <- b1
                        let mut field_types: Vec<Ty> = field_type_params
                            .iter()
                            .map(|tv| Ty::Var(tv.clone()))
                            .collect();

                        let mut con_ty = result_ty;
                        for field_ty in field_types.iter().rev() {
                            con_ty = Ty::fun(field_ty.clone(), con_ty);
                        }

                        // Track constructor → type mapping for RecordUpdate
                        self.type_to_data_cons
                            .entry(type_con_name)
                            .or_default()
                            .push(def_info.id);

                        // If we have field names, register field definitions for record construction
                        // Note: we don't store the field types here since they're quantified
                        // and will be instantiated fresh each time. We only need the names.
                        if let Some(ref field_names) = def_info.field_names {
                            let field_defs: Vec<(Symbol, Ty)> = field_names
                                .iter()
                                .zip(field_types.iter())
                                .map(|(name, ty)| (*name, ty.clone()))
                                .collect();
                            self.con_field_defs.insert(def_info.id, field_defs);
                        }

                        // Combine all type parameters: result params + field params
                        let all_params: Vec<TyVar> = result_type_params
                            .into_iter()
                            .chain(field_type_params.into_iter())
                            .collect();

                        Scheme::poly(all_params, con_ty)
                    } else if let Some(arity) = def_info.arity {
                        // No type info, fall back to fresh type variables
                        let result = self.fresh_ty();
                        let mut con_ty = result;
                        for _ in 0..arity {
                            let arg = self.fresh_ty();
                            con_ty = Ty::fun(arg, con_ty);
                        }
                        Scheme::mono(con_ty)
                    } else {
                        // No arity info, fall back to fresh type variable
                        let fresh = self.fresh_ty();
                        Scheme::mono(fresh)
                    }
                }
            };

            // Register the constructor with its DefId from the lowering pass
            self.env
                .register_data_con(def_info.id, def_info.name, scheme);
        }

        // Helper to create common type schemes
        let num_binop = || {
            // a -> a -> a (polymorphic, allows Int, Integer, Float, Double)
            Scheme::poly(
                vec![a.clone()],
                Ty::fun(
                    Ty::Var(a.clone()),
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                ),
            )
        };

        let cmp_binop = || {
            // a -> a -> Bool (for Ord types, polymorphic like eq_binop)
            Scheme::poly(
                vec![a.clone()],
                Ty::fun(
                    Ty::Var(a.clone()),
                    Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                ),
            )
        };

        let eq_binop = || {
            // a -> a -> Bool (for Eq types, polymorphic)
            Scheme::poly(
                vec![a.clone()],
                Ty::fun(
                    Ty::Var(a.clone()),
                    Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                ),
            )
        };

        // For each def in the lowering pass's def map, register it with a type
        for (_def_id, def_info) in defs.iter() {
            let raw_name = def_info.name.as_str();
            // Normalize ByteString variant module names so they share type signatures,
            // EXCEPT for Char8-specific functions that need Char instead of Word8/Int.
            let normalized_name;
            let is_char8 = raw_name.starts_with("Data.ByteString.Char8.") || raw_name.starts_with("Data.ByteString.Lazy.Char8.");
            let name = if raw_name.starts_with("Data.ByteString.Char8.") {
                // Char8 functions that use Char where strict ByteString uses Word8/Int
                // are handled separately below — don't normalize them.
                let fn_name = &raw_name["Data.ByteString.Char8.".len()..];
                match fn_name {
                    "pack" | "unpack" | "head" | "last" | "singleton" | "cons" | "snoc"
                    | "uncons" | "map" | "filter" | "elem" | "notElem"
                    | "find" | "intercalate" | "concatMap" | "any" | "all"
                    | "foldr" | "foldl" | "foldr'" | "foldl'" => raw_name,
                    _ => {
                        normalized_name = raw_name.replacen("Data.ByteString.Char8.", "Data.ByteString.", 1);
                        normalized_name.as_str()
                    }
                }
            } else if raw_name.starts_with("Data.ByteString.Lazy.Char8.") {
                let fn_name = &raw_name["Data.ByteString.Lazy.Char8.".len()..];
                match fn_name {
                    "pack" | "unpack" | "head" | "last" | "singleton" | "cons" | "snoc"
                    | "uncons" | "map" | "filter" | "elem" | "notElem"
                    | "find" | "intercalate" | "concatMap" | "any" | "all"
                    | "foldr" | "foldl" | "foldr'" | "foldl'" => raw_name,
                    _ => {
                        normalized_name = raw_name.replacen("Data.ByteString.Lazy.Char8.", "Data.ByteString.Lazy.", 1);
                        normalized_name.as_str()
                    }
                }
            } else {
                raw_name
            };
            let scheme = match name {
                // Arithmetic operators
                "+" | "-" | "*" | "/" | "div" | "mod" | "^" | "^^" | "**" => num_binop(),
                // Comparison operators
                "==" | "/=" => eq_binop(),
                "<" | "<=" | ">" | ">=" => cmp_binop(),
                // Boolean operators
                "&&" | "||" => Scheme::mono(Ty::fun(
                    self.builtins.bool_ty.clone(),
                    Ty::fun(self.builtins.bool_ty.clone(), self.builtins.bool_ty.clone()),
                )),
                // List cons
                ":" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                    )
                }
                // List append
                "++" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                    )
                }
                // List indexing
                "!!" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            list_a,
                            Ty::fun(self.builtins.int_ty.clone(), Ty::Var(a.clone())),
                        ),
                    )
                }
                // Function composition
                "." => {
                    let c = TyVar::new_star(0xFFFF_0002);
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                            Ty::fun(
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                            ),
                        ),
                    )
                }
                // Function application
                "$" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    ),
                ),
                // Monadic operators (polymorphic over monad m)
                // (>>) :: m a -> m b -> m b
                ">>" => {
                    let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                    let m = TyVar::new(BUILTIN_TYVAR_M, m_kind);
                    let ma = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())));
                    let mb = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![m.clone(), a.clone(), b.clone()],
                        Ty::fun(ma, Ty::fun(mb.clone(), mb)),
                    )
                }
                // (>>=) :: m a -> (a -> m b) -> m b
                ">>=" => {
                    let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                    let m = TyVar::new(BUILTIN_TYVAR_M, m_kind);
                    let ma = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())));
                    let mb = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![m.clone(), a.clone(), b.clone()],
                        Ty::fun(
                            ma,
                            Ty::fun(Ty::fun(Ty::Var(a.clone()), mb.clone()), mb),
                        ),
                    )
                }
                // (=<<) :: (a -> m b) -> m a -> m b (flipped >>=)
                "=<<" => {
                    let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                    let m = TyVar::new(BUILTIN_TYVAR_M, m_kind);
                    let ma = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())));
                    let mb = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![m.clone(), a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), mb.clone()),
                            Ty::fun(ma, mb),
                        ),
                    )
                }
                // return :: a -> m a
                "return" => {
                    let m_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                    let m = TyVar::new(BUILTIN_TYVAR_M, m_kind);
                    let ma = Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![m, a.clone()], Ty::fun(Ty::Var(a.clone()), ma))
                }
                // pure :: a -> f a
                "pure" => {
                    let f_kind = Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star));
                    let f = TyVar::new(BUILTIN_TYVAR_F, f_kind);
                    let fa = Ty::App(Box::new(Ty::Var(f.clone())), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![f, a.clone()], Ty::fun(Ty::Var(a.clone()), fa))
                }
                // map :: (a -> b) -> [a] -> [b]
                "map" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(list_a, list_b),
                        ),
                    )
                }
                // filter :: (a -> Bool) -> [a] -> [a]
                "filter" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                            Ty::fun(list_a.clone(), list_a),
                        ),
                    )
                }
                // head :: [a] -> a
                "head" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
                }
                // tail :: [a] -> [a]
                "tail" | "reverse" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
                }
                // length :: [a] -> Int
                "length" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(list_a, self.builtins.int_ty.clone()),
                    )
                }
                // null :: [a] -> Bool
                "null" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(list_a, self.builtins.bool_ty.clone()),
                    )
                }
                // take, drop :: Int -> [a] -> [a]
                "take" | "drop" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            self.builtins.int_ty.clone(),
                            Ty::fun(list_a.clone(), list_a),
                        ),
                    )
                }
                // sum, product :: [Int] -> Int
                "sum" | "product" => {
                    let list_int = Ty::List(Box::new(self.builtins.int_ty.clone()));
                    Scheme::mono(Ty::fun(list_int, self.builtins.int_ty.clone()))
                }
                // foldl :: (b -> a -> b) -> b -> c -> b
                // Container arg is polymorphic — codegen dispatches by expression structure.
                "foldl" | "foldl'" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(
                                Ty::Var(b.clone()),
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            ),
                            Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(c.clone()), Ty::Var(b.clone()))),
                        ),
                    )
                }
                // foldr :: Foldable t => (a -> b -> b) -> b -> t a -> b
                // Container arg is polymorphic — codegen dispatches by expression structure.
                "foldr" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(
                                Ty::Var(a.clone()),
                                Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())),
                            ),
                            Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(c.clone()), Ty::Var(b.clone()))),
                        ),
                    )
                }
                // zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
                "zipWith" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                    let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(
                                Ty::Var(a.clone()),
                                Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                            ),
                            Ty::fun(list_a, Ty::fun(list_b, list_c)),
                        ),
                    )
                }
                // zip :: [a] -> [b] -> [(a, b)]
                "zip" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                    let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                    let list_pair = Ty::List(Box::new(pair_ab));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(list_a, Ty::fun(list_b, list_pair)),
                    )
                }
                // maximum, minimum :: [a] -> a
                "maximum" | "minimum" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
                }
                // last :: [a] -> a
                "last" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
                }
                // init :: [a] -> [a]
                "init" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
                }
                // concat :: [[a]] -> [a]
                "concat" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_list_a = Ty::List(Box::new(list_a.clone()));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_list_a, list_a))
                }
                // concatMap :: (a -> [b]) -> [a] -> [b]
                "concatMap" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), list_b.clone()),
                            Ty::fun(list_a, list_b),
                        ),
                    )
                }
                // id :: a -> a
                "id" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                ),
                // const :: a -> b -> a
                "const" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone())),
                    ),
                ),
                // error :: String -> a
                "error" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(self.builtins.string_ty.clone(), Ty::Var(a.clone())),
                ),
                // undefined :: a
                "undefined" => Scheme::poly(vec![a.clone()], Ty::Var(a.clone())),
                // seq :: a -> b -> b
                "seq" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())),
                    ),
                ),
                // negate, abs, signum :: a -> a (polymorphic)
                "negate" | "abs" | "signum" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                ),
                // not :: Bool -> Bool
                "not" => Scheme::mono(Ty::fun(
                    self.builtins.bool_ty.clone(),
                    self.builtins.bool_ty.clone(),
                )),
                // otherwise :: Bool
                "otherwise" => Scheme::mono(self.builtins.bool_ty.clone()),
                // show :: a -> String
                "show" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), self.builtins.string_ty.clone()),
                ),
                // print :: a -> IO ()
                "print" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::App(
                            Box::new(Ty::Con(self.builtins.io_con.clone())),
                            Box::new(Ty::unit()),
                        ),
                    ),
                ),
                // putStrLn, putStr :: String -> IO ()
                "putStrLn" | "putStr" => Scheme::mono(Ty::fun(
                    self.builtins.string_ty.clone(),
                    Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::unit()),
                    ),
                )),
                // getLine :: IO String
                "getLine" => Scheme::mono(Ty::App(
                    Box::new(Ty::Con(self.builtins.io_con.clone())),
                    Box::new(self.builtins.string_ty.clone()),
                )),
                // fst :: (a, b) -> a
                "fst" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]),
                        Ty::Var(a.clone()),
                    ),
                ),
                // snd :: (a, b) -> b
                "snd" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]),
                        Ty::Var(b.clone()),
                    ),
                ),
                // replicate :: Int -> a -> [a]
                "replicate" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            self.builtins.int_ty.clone(),
                            Ty::fun(Ty::Var(a.clone()), list_a),
                        ),
                    )
                }
                // enumFromTo :: a -> a -> [a] (Enum a =>)
                "enumFromTo" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), list_a)),
                    )
                }
                // ord :: Char -> Int
                "ord" => Scheme::mono(Ty::fun(
                    self.builtins.char_ty.clone(),
                    self.builtins.int_ty.clone(),
                )),
                // chr :: Int -> Char
                "chr" => Scheme::mono(Ty::fun(
                    self.builtins.int_ty.clone(),
                    self.builtins.char_ty.clone(),
                )),
                // flip :: (a -> b -> c) -> b -> a -> c
                "flip" => Scheme::poly(
                    vec![a.clone(), b.clone(), c.clone()],
                    Ty::fun(
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                        ),
                        Ty::fun(
                            Ty::Var(b.clone()),
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                        ),
                    ),
                ),
                // even, odd :: Int -> Bool
                "even" | "odd" => Scheme::mono(Ty::fun(
                    self.builtins.int_ty.clone(),
                    self.builtins.bool_ty.clone(),
                )),
                // succ, pred :: a -> a (Enum a =>)
                "succ" | "pred" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                ),
                // (&) :: a -> (a -> b) -> b
                "&" => Scheme::poly(
                    vec![a.clone(), b.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::Var(b.clone()),
                        ),
                    ),
                ),
                // elem, notElem :: a -> [a] -> Bool
                "elem" | "notElem" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(list_a, self.builtins.bool_ty.clone()),
                        ),
                    )
                }
                // takeWhile, dropWhile :: (a -> Bool) -> [a] -> [a]
                "takeWhile" | "dropWhile" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                            Ty::fun(list_a.clone(), list_a),
                        ),
                    )
                }
                // span :: (a -> Bool) -> [a] -> ([a], [a])
                "span" | "break" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let pair = Ty::Tuple(vec![list_a.clone(), list_a.clone()]);
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                            Ty::fun(list_a, pair),
                        ),
                    )
                }
                // splitAt :: Int -> [a] -> ([a], [a])
                "splitAt" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let pair = Ty::Tuple(vec![list_a.clone(), list_a.clone()]);
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(self.builtins.int_ty.clone(), Ty::fun(list_a, pair)),
                    )
                }
                // iterate :: (a -> a) -> a -> [a]
                "iterate" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                            Ty::fun(Ty::Var(a.clone()), list_a),
                        ),
                    )
                }
                // repeat :: a -> [a]
                "repeat" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), list_a))
                }
                // cycle :: [a] -> [a]
                "cycle" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
                }
                // lookup :: a -> [(a, b)] -> Maybe b
                "lookup" => {
                    let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                    let list_pairs = Ty::List(Box::new(pair_ab));
                    let maybe_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_pairs, maybe_b)),
                    )
                }
                // unzip :: [(a, b)] -> ([a], [b])
                "unzip" => {
                    let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                    let list_pairs = Ty::List(Box::new(pair_ab));
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_b = Ty::List(Box::new(Ty::Var(b.clone())));
                    let result = Ty::Tuple(vec![list_a, list_b]);
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(list_pairs, result))
                }
                // maybe :: b -> (a -> b) -> Maybe a -> b
                "maybe" => {
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::Var(b.clone()),
                            Ty::fun(
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                                Ty::fun(maybe_a, Ty::Var(b.clone())),
                            ),
                        ),
                    )
                }
                // fromMaybe :: a -> Maybe a -> a
                "fromMaybe" => {
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(maybe_a, Ty::Var(a.clone()))),
                    )
                }
                // either :: (a -> c) -> (b -> c) -> Either a b -> c
                "either" => {
                    let either_ab = Ty::App(
                        Box::new(Ty::App(
                            Box::new(Ty::Con(self.builtins.either_con.clone())),
                            Box::new(Ty::Var(a.clone())),
                        )),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                            Ty::fun(
                                Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                                Ty::fun(either_ab, Ty::Var(c.clone())),
                            ),
                        ),
                    )
                }
                // min, max :: a -> a -> a (polymorphic)
                "min" | "max" => Scheme::poly(
                    vec![a.clone()],
                    Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                    ),
                ),
                // E.28: Arithmetic, enum, folds, higher-order, IO input
                "subtract" => Scheme::mono(Ty::fun(
                    self.builtins.int_ty.clone(),
                    Ty::fun(self.builtins.int_ty.clone(), self.builtins.int_ty.clone()),
                )),
                "enumFrom" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), list_a))
                }
                "enumFromThen" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), list_a)),
                    )
                }
                "enumFromThenTo" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), list_a)),
                        ),
                    )
                }
                "foldl1" | "foldr1" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                            Ty::fun(list_a, Ty::Var(a.clone())),
                        ),
                    )
                }
                "comparing" => {
                    let ordering_ty = Ty::Con(self.builtins.ordering_con.clone());
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.int_ty.clone()),
                            Ty::fun(
                                Ty::Var(a.clone()),
                                Ty::fun(Ty::Var(a.clone()), ordering_ty),
                            ),
                        ),
                    )
                }
                "until" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                            Ty::fun(
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                            ),
                        ),
                    )
                }
                "getChar" => Scheme::mono(Ty::App(
                    Box::new(Ty::Con(self.builtins.io_con.clone())),
                    Box::new(self.builtins.char_ty.clone()),
                )),
                "isEOF" => Scheme::mono(Ty::App(
                    Box::new(Ty::Con(self.builtins.io_con.clone())),
                    Box::new(self.builtins.bool_ty.clone()),
                )),
                "getContents" => {
                    let string_ty = Ty::List(Box::new(self.builtins.char_ty.clone()));
                    let io_string = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(string_ty),
                    );
                    Scheme::mono(io_string)
                }
                "interact" => {
                    let string_ty = Ty::List(Box::new(self.builtins.char_ty.clone()));
                    let io_unit = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::unit()),
                    );
                    Scheme::mono(Ty::fun(
                        Ty::fun(string_ty.clone(), string_ty),
                        io_unit,
                    ))
                }
                // fromIntegral :: (Integral a, Num b) => a -> b (polymorphic)
                "fromIntegral" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    )
                }
                // toInteger :: Integral a => a -> Integer
                "toInteger" => Scheme::mono(Ty::fun(
                    self.builtins.int_ty.clone(),
                    self.builtins.int_ty.clone(),
                )),
                // any :: (a -> Bool) -> [a] -> Bool
                "any" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                            Ty::fun(list_a, self.builtins.bool_ty.clone()),
                        ),
                    )
                }
                // all :: (a -> Bool) -> [a] -> Bool
                "all" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                            Ty::fun(list_a, self.builtins.bool_ty.clone()),
                        ),
                    )
                }
                // and :: [Bool] -> Bool
                "and" => {
                    let list_bool = Ty::List(Box::new(self.builtins.bool_ty.clone()));
                    Scheme::mono(Ty::fun(list_bool, self.builtins.bool_ty.clone()))
                }
                // or :: [Bool] -> Bool
                "or" => {
                    let list_bool = Ty::List(Box::new(self.builtins.bool_ty.clone()));
                    Scheme::mono(Ty::fun(list_bool, self.builtins.bool_ty.clone()))
                }
                // lines :: String -> [String]
                "lines" => {
                    let list_string = Ty::List(Box::new(self.builtins.string_ty.clone()));
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), list_string))
                }
                // unlines :: [String] -> String
                "unlines" => {
                    let list_string = Ty::List(Box::new(self.builtins.string_ty.clone()));
                    Scheme::mono(Ty::fun(list_string, self.builtins.string_ty.clone()))
                }
                // words :: String -> [String]
                "words" => {
                    let list_string = Ty::List(Box::new(self.builtins.string_ty.clone()));
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), list_string))
                }
                // unwords :: [String] -> String
                "unwords" => {
                    let list_string = Ty::List(Box::new(self.builtins.string_ty.clone()));
                    Scheme::mono(Ty::fun(list_string, self.builtins.string_ty.clone()))
                }
                // isJust :: Maybe a -> Bool
                "isJust" | "isNothing" => {
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(maybe_a, self.builtins.bool_ty.clone()),
                    )
                }
                // curry :: ((a, b) -> c) -> a -> b -> c
                "curry" => {
                    let pair = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(pair, Ty::Var(c.clone())),
                            Ty::fun(
                                Ty::Var(a.clone()),
                                Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                            ),
                        ),
                    )
                }
                // uncurry :: (a -> b -> c) -> (a, b) -> c
                "uncurry" => {
                    let pair = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(
                                Ty::Var(a.clone()),
                                Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                            ),
                            Ty::fun(pair, Ty::Var(c.clone())),
                        ),
                    )
                }
                // swap :: (a, b) -> (b, a)
                "swap" => {
                    let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                    let pair_ba = Ty::Tuple(vec![Ty::Var(b.clone()), Ty::Var(a.clone())]);
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(pair_ab, pair_ba))
                }
                // readFile :: String -> IO String
                "readFile" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::App(
                            Box::new(Ty::Con(self.builtins.io_con.clone())),
                            Box::new(self.builtins.string_ty.clone()),
                        ),
                    ))
                }
                // writeFile :: String -> String -> IO ()
                "writeFile" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::fun(
                            self.builtins.string_ty.clone(),
                            Ty::App(
                                Box::new(Ty::Con(self.builtins.io_con.clone())),
                                Box::new(Ty::unit()),
                            ),
                        ),
                    ))
                }
                // appendFile :: String -> String -> IO ()
                "appendFile" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::fun(
                            self.builtins.string_ty.clone(),
                            Ty::App(
                                Box::new(Ty::Con(self.builtins.io_con.clone())),
                                Box::new(Ty::unit()),
                            ),
                        ),
                    ))
                }
                // catch :: IO a -> (SomeException -> IO a) -> IO a
                // (simplified: exception type is String)
                "catch" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let handler = Ty::fun(
                        self.builtins.string_ty.clone(),
                        io_a.clone(),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(io_a.clone(), Ty::fun(handler, io_a)),
                    )
                }
                // bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
                "bracket" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let io_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    let io_c = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(c.clone())),
                    );
                    let cleanup = Ty::fun(Ty::Var(a.clone()), io_b);
                    let body = Ty::fun(Ty::Var(a.clone()), io_c.clone());
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(io_a, Ty::fun(cleanup, Ty::fun(body, io_c))),
                    )
                }
                // bracket_ :: IO a -> IO b -> IO c -> IO c
                "bracket_" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let io_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    let io_c = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(c.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(io_a, Ty::fun(io_b, Ty::fun(io_c.clone(), io_c))),
                    )
                }
                // finally :: IO a -> IO b -> IO a
                "finally" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let io_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(io_a.clone(), Ty::fun(io_b, io_a)),
                    )
                }
                // handle :: (SomeException -> IO a) -> IO a -> IO a
                "handle" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let handler = Ty::fun(
                        self.builtins.string_ty.clone(),
                        io_a.clone(),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(handler, Ty::fun(io_a.clone(), io_a)),
                    )
                }
                // onException :: IO a -> IO b -> IO a
                "onException" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let io_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(io_a.clone(), Ty::fun(io_b, io_a)),
                    )
                }
                // throwIO :: e -> IO a  (simplified: String -> IO a)
                "throwIO" | "throw" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(self.builtins.string_ty.clone(), io_a),
                    )
                }
                // try :: IO a -> IO (Either SomeException a)
                "try" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let either_exc_a = Ty::App(
                        Box::new(Ty::App(
                            Box::new(Ty::Con(self.builtins.either_con.clone())),
                            Box::new(self.builtins.string_ty.clone()),
                        )),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let io_either = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(either_exc_a),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(io_a, io_either),
                    )
                }
                // evaluate :: a -> IO a
                "evaluate" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), io_a),
                    )
                }
                // mask :: ((IO a -> IO a) -> IO b) -> IO b
                "mask" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let io_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    let restore = Ty::fun(io_a.clone(), io_a);
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(restore, io_b.clone()), io_b),
                    )
                }
                // mask_ :: IO a -> IO a
                "mask_" | "uninterruptibleMask_" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone()], Ty::fun(io_a.clone(), io_a))
                }
                // uninterruptibleMask :: ((IO a -> IO a) -> IO b) -> IO b
                "uninterruptibleMask" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let io_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    let restore = Ty::fun(io_a.clone(), io_a);
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(restore, io_b.clone()), io_b),
                    )
                }
                // getMaskingState :: IO a (polymorphic, returns MaskingState at runtime)
                "getMaskingState" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone()], io_a)
                }
                // openFile :: FilePath -> IOMode -> IO Handle
                // Simplified: String -> a -> IO b (IOMode and Handle are opaque)
                "openFile" => {
                    let io_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(self.builtins.string_ty.clone(), Ty::fun(Ty::Var(a.clone()), io_b)),
                    )
                }
                // hGetLine :: Handle -> IO String
                // Simplified: a -> IO String (Handle is opaque)
                "hGetLine" | "hGetContents" => {
                    let io_string = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(self.builtins.string_ty.clone()),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), io_string),
                    )
                }
                // hClose :: Handle -> IO ()
                // Simplified: a -> IO ()
                "hClose" => {
                    let io_unit = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::unit()),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), io_unit),
                    )
                }
                // hPutStr :: Handle -> String -> IO ()
                "hPutStr" | "hPutStrLn" => {
                    let io_unit = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::unit()),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.string_ty.clone(), io_unit)),
                    )
                }
                // withFile :: FilePath -> IOMode -> (Handle -> IO a) -> IO a
                "withFile" => {
                    let io_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    let callback = Ty::fun(Ty::Var(b.clone()), io_a.clone());
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            self.builtins.string_ty.clone(),
                            Ty::fun(Ty::Var(c.clone()), Ty::fun(callback, io_a)),
                        ),
                    )
                }
                // doesFileExist :: FilePath -> IO Bool
                "doesFileExist" | "doesDirectoryExist" => {
                    let io_bool = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(self.builtins.bool_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), io_bool))
                }
                // E.19: System.Directory — String -> IO ()
                "createDirectory" | "removeFile" | "removeDirectory" | "setCurrentDirectory" => {
                    let io_unit = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::unit()),
                    );
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), io_unit))
                }
                // E.19: System.Directory — String -> String -> IO ()
                "renameFile" | "copyFile" => {
                    let io_unit = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::unit()),
                    );
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::fun(self.builtins.string_ty.clone(), io_unit),
                    ))
                }
                // E.19: System.Directory — listDirectory :: String -> IO [String]
                "listDirectory" => {
                    let io_list_string = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::List(Box::new(self.builtins.string_ty.clone()))),
                    );
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), io_list_string))
                }
                // E.19: System.FilePath — String -> String
                "takeFileName" | "takeDirectory" | "takeExtension"
                | "dropExtension" | "takeBaseName" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        self.builtins.string_ty.clone(),
                    ))
                }
                // E.19: System.FilePath — String -> String -> String
                "replaceExtension" | "</>" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::fun(
                            self.builtins.string_ty.clone(),
                            self.builtins.string_ty.clone(),
                        ),
                    ))
                }
                // E.19: System.FilePath — String -> Bool
                "isAbsolute" | "isRelative" | "hasExtension" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        self.builtins.bool_ty.clone(),
                    ))
                }
                // E.19: System.FilePath — splitExtension :: String -> (String, String)
                "splitExtension" => {
                    let tuple_ss = Ty::Tuple(vec![
                        self.builtins.string_ty.clone(),
                        self.builtins.string_ty.clone(),
                    ]);
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), tuple_ss))
                }
                // E.20: Data.Text — Text (constant)
                "Data.Text.empty" => {
                    Scheme::mono(self.builtins.text_ty.clone())
                }
                // Data.Text: Char -> Text
                "Data.Text.singleton" => {
                    Scheme::mono(Ty::fun(self.builtins.char_ty.clone(), self.builtins.text_ty.clone()))
                }
                // Data.Text: String -> Text
                "Data.Text.pack" => {
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), self.builtins.text_ty.clone()))
                }
                // Data.Text: Text -> String
                "Data.Text.unpack" => {
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), self.builtins.string_ty.clone()))
                }
                // Data.Text: Text -> Bool
                "Data.Text.null" => {
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), self.builtins.bool_ty.clone()))
                }
                // Data.Text: Text -> Int
                "Data.Text.length" | "Data.Text.compareLength" => {
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), self.builtins.int_ty.clone()))
                }
                // Data.Text: Text -> Char
                "Data.Text.head" | "Data.Text.last" => {
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), self.builtins.char_ty.clone()))
                }
                // Data.Text: Text -> Text
                "Data.Text.tail" | "Data.Text.init" | "Data.Text.reverse"
                | "Data.Text.toLower" | "Data.Text.toUpper" | "Data.Text.toCaseFold"
                | "Data.Text.toTitle" | "Data.Text.strip" => {
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()))
                }
                // Data.Text: Text -> Text -> Text
                "Data.Text.append" | "Data.Text.<>" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                    ))
                }
                // Data.Text: Int -> Text -> Text
                "Data.Text.take" | "Data.Text.takeEnd" | "Data.Text.drop" | "Data.Text.dropEnd" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                    ))
                }
                // Data.Text: Text -> Text -> Bool
                "Data.Text.isPrefixOf" | "Data.Text.isSuffixOf" | "Data.Text.isInfixOf"
                | "Data.Text.eq" | "Data.Text.==" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.bool_ty.clone()),
                    ))
                }
                // Data.Text: (Char -> Char) -> Text -> Text
                "Data.Text.map" => {
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.char_ty.clone()),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                    ))
                }
                // Data.Text: Text -> Text -> Int
                "Data.Text.compare" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.int_ty.clone()),
                    ))
                }
                // Data.Text: (Char -> Bool) -> Text -> Text
                "Data.Text.filter" | "Data.Text.takeWhile" | "Data.Text.dropWhile" => {
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                    ))
                }
                // Data.Text: foldl' :: (a -> Char -> a) -> a -> Text -> a
                "Data.Text.foldl'" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.char_ty.clone(), Ty::Var(a.clone()))),
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.text_ty.clone(), Ty::Var(a.clone()))),
                        ),
                    )
                }
                // Data.Text: [Text] -> Text
                "Data.Text.concat" => {
                    Scheme::mono(Ty::fun(
                        Ty::List(Box::new(self.builtins.text_ty.clone())),
                        self.builtins.text_ty.clone(),
                    ))
                }
                // Data.Text: Text -> [Text] -> Text
                "Data.Text.intercalate" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(
                            Ty::List(Box::new(self.builtins.text_ty.clone())),
                            self.builtins.text_ty.clone(),
                        ),
                    ))
                }
                // Data.Text: Text -> [Text]
                "Data.Text.words" | "Data.Text.lines" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::List(Box::new(self.builtins.text_ty.clone())),
                    ))
                }
                // Data.Text: Text -> Text -> [Text]
                "Data.Text.splitOn" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(
                            self.builtins.text_ty.clone(),
                            Ty::List(Box::new(self.builtins.text_ty.clone())),
                        ),
                    ))
                }
                // Data.Text: Text -> Text -> Text -> Text
                "Data.Text.replace" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(
                            self.builtins.text_ty.clone(),
                            Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                        ),
                    ))
                }
                // Data.Text: Char -> Text -> Text (cons), Text -> Char -> Text (snoc)
                "Data.Text.cons" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.char_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                    ))
                }
                "Data.Text.snoc" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.text_ty.clone()),
                    ))
                }
                // Data.Text: Int -> Text -> Text (replicate)
                "Data.Text.replicate" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                    ))
                }
                // Data.Text: Int -> Char -> Text -> Text (justifyLeft, justifyRight)
                "Data.Text.justifyLeft" | "Data.Text.justifyRight" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(
                            self.builtins.char_ty.clone(),
                            Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                        ),
                    ))
                }
                // Data.Text: Text -> Text -> (Text, Text) — breakOn, breakOnEnd
                "Data.Text.breakOn" | "Data.Text.breakOnEnd" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(
                            self.builtins.text_ty.clone(),
                            Ty::Tuple(vec![self.builtins.text_ty.clone(), self.builtins.text_ty.clone()]),
                        ),
                    ))
                }
                // Data.Text: (Char -> Bool) -> Text -> (Text, Text) — span, break
                "Data.Text.span" | "Data.Text.break" => {
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(
                            self.builtins.text_ty.clone(),
                            Ty::Tuple(vec![self.builtins.text_ty.clone(), self.builtins.text_ty.clone()]),
                        ),
                    ))
                }
                // Data.Text: (Char -> a) -> Text -> [a] — concatMap returns Text
                "Data.Text.concatMap" => {
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.text_ty.clone()),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                    ))
                }
                // Data.Text: Text -> Text -> [Text] — breakOnAll
                "Data.Text.breakOnAll" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(
                            self.builtins.text_ty.clone(),
                            Ty::List(Box::new(Ty::Tuple(vec![
                                self.builtins.text_ty.clone(),
                                self.builtins.text_ty.clone(),
                            ]))),
                        ),
                    ))
                }
                // Data.Text: Text -> Bool — any/all
                "Data.Text.any" | "Data.Text.all" => {
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.bool_ty.clone()),
                    ))
                }
                // Data.Text: Char -> Text -> [Text] — splitOn with Char version
                "Data.Text.unfoldr" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.maybe_of(Ty::Tuple(vec![self.builtins.char_ty.clone(), Ty::Var(a.clone())]))),
                            Ty::fun(Ty::Var(a.clone()), self.builtins.text_ty.clone()),
                        ),
                    )
                }
                // Data.Text: (a -> Char -> a) -> a -> Text -> a — foldl
                "Data.Text.foldl" | "Data.Text.foldr" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.char_ty.clone(), Ty::Var(a.clone()))),
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.text_ty.clone(), Ty::Var(a.clone()))),
                        ),
                    )
                }
                // Data.Text: Text -> Int -> Char — index
                "Data.Text.index" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(self.builtins.int_ty.clone(), self.builtins.char_ty.clone()),
                    ))
                }
                // Data.Text: Text -> Maybe (Char, Text) — uncons
                "Data.Text.uncons" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        self.builtins.maybe_of(Ty::Tuple(vec![
                            self.builtins.char_ty.clone(),
                            self.builtins.text_ty.clone(),
                        ])),
                    ))
                }
                // Data.Text: [Text] -> Text — unlines, unwords
                "Data.Text.unlines" | "Data.Text.unwords" => {
                    Scheme::mono(Ty::fun(
                        Ty::List(Box::new(self.builtins.text_ty.clone())),
                        self.builtins.text_ty.clone(),
                    ))
                }
                // Data.Text: Text -> Text -> Maybe Text — stripPrefix, stripSuffix
                "Data.Text.stripPrefix" | "Data.Text.stripSuffix" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(
                            self.builtins.text_ty.clone(),
                            self.builtins.maybe_of(self.builtins.text_ty.clone()),
                        ),
                    ))
                }
                // Data.Text: Text -> Text -> Int — count
                "Data.Text.count" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), self.builtins.int_ty.clone()),
                    ))
                }
                // Data.Text: (Char -> Bool) -> Text -> Maybe Char — find
                "Data.Text.find" => {
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(
                            self.builtins.text_ty.clone(),
                            self.builtins.maybe_of(self.builtins.char_ty.clone()),
                        ),
                    ))
                }
                // Data.Text: (Char -> Bool) -> Text -> Maybe Int — findIndex
                "Data.Text.findIndex" => {
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(
                            self.builtins.text_ty.clone(),
                            self.builtins.maybe_of(self.builtins.int_ty.clone()),
                        ),
                    ))
                }
                // Data.Text: (Char -> Char -> Bool) -> Text -> [Text] — groupBy
                "Data.Text.groupBy" => {
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone())),
                        Ty::fun(self.builtins.text_ty.clone(), Ty::List(Box::new(self.builtins.text_ty.clone()))),
                    ))
                }
                // Data.Text: Char -> Text -> [Text] — splitAt returning (Text, Text)
                "Data.Text.splitAt" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), Ty::Tuple(vec![self.builtins.text_ty.clone(), self.builtins.text_ty.clone()])),
                    ))
                }
                // Data.Text.Encoding: Text -> ByteString
                "Data.Text.Encoding.encodeUtf8" | "Data.Text.Encoding.encodeUtf16LE" => {
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), self.builtins.bytestring_ty.clone()))
                }
                // Data.Text.Encoding: ByteString -> Text
                "Data.Text.Encoding.decodeUtf8" | "Data.Text.Encoding.decodeUtf8With"
                | "Data.Text.Encoding.decodeLatin1" | "Data.Text.Encoding.decodeUtf16LE" => {
                    Scheme::mono(Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.text_ty.clone()))
                }
                // Data.Text.Lazy: pack/unpack
                "Data.Text.Lazy.pack" => {
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), self.builtins.text_ty.clone()))
                }
                "Data.Text.Lazy.unpack" => {
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), self.builtins.string_ty.clone()))
                }
                // Data.Text.Lazy.Encoding: TL.Text -> BL.ByteString
                "Data.Text.Lazy.Encoding.encodeUtf8" => {
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), self.builtins.bytestring_ty.clone()))
                }
                // Data.Text.Lazy.Encoding: BL.ByteString -> TL.Text
                "Data.Text.Lazy.Encoding.decodeUtf8" | "Data.Text.Lazy.Encoding.decodeUtf8With"
                | "Data.Text.Lazy.Encoding.decodeUtf8'" => {
                    Scheme::mono(Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.text_ty.clone()))
                }
                // E.20: Data.ByteString — ByteString (constant)
                "Data.ByteString.empty" => {
                    Scheme::mono(self.builtins.bytestring_ty.clone())
                }
                // Data.ByteString: Int -> ByteString
                "Data.ByteString.singleton" => {
                    Scheme::mono(Ty::fun(self.builtins.int_ty.clone(), self.builtins.bytestring_ty.clone()))
                }
                // Data.ByteString.Char8: Char-oriented interfaces
                "Data.ByteString.Char8.pack" | "Data.ByteString.Lazy.Char8.pack" => {
                    // pack :: String -> ByteString (Char8 uses [Char] not [Word8])
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        self.builtins.bytestring_ty.clone(),
                    ))
                }
                "Data.ByteString.Char8.unpack" | "Data.ByteString.Lazy.Char8.unpack" => {
                    // unpack :: ByteString -> String
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        self.builtins.string_ty.clone(),
                    ))
                }
                "Data.ByteString.Char8.head" | "Data.ByteString.Char8.last"
                | "Data.ByteString.Lazy.Char8.head" | "Data.ByteString.Lazy.Char8.last" => {
                    // head/last :: ByteString -> Char
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        self.builtins.char_ty.clone(),
                    ))
                }
                "Data.ByteString.Char8.singleton" | "Data.ByteString.Lazy.Char8.singleton" => {
                    // singleton :: Char -> ByteString
                    Scheme::mono(Ty::fun(
                        self.builtins.char_ty.clone(),
                        self.builtins.bytestring_ty.clone(),
                    ))
                }
                "Data.ByteString.Char8.cons" | "Data.ByteString.Lazy.Char8.cons"
                | "Data.ByteString.Char8.snoc" | "Data.ByteString.Lazy.Char8.snoc" => {
                    // cons :: Char -> ByteString -> ByteString
                    // snoc :: ByteString -> Char -> ByteString (same signature shape for permissiveness)
                    Scheme::mono(Ty::fun(
                        self.builtins.char_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                "Data.ByteString.Char8.map" | "Data.ByteString.Lazy.Char8.map" => {
                    // map :: (Char -> Char) -> ByteString -> ByteString
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.char_ty.clone()),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                "Data.ByteString.Char8.filter" | "Data.ByteString.Lazy.Char8.filter" => {
                    // filter :: (Char -> Bool) -> ByteString -> ByteString
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                "Data.ByteString.Char8.elem" | "Data.ByteString.Char8.notElem"
                | "Data.ByteString.Lazy.Char8.elem" | "Data.ByteString.Lazy.Char8.notElem" => {
                    // elem :: Char -> ByteString -> Bool
                    Scheme::mono(Ty::fun(
                        self.builtins.char_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bool_ty.clone()),
                    ))
                }
                "Data.ByteString.Char8.find" | "Data.ByteString.Lazy.Char8.find" => {
                    // find :: (Char -> Bool) -> ByteString -> Maybe Char
                    let maybe_char = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(self.builtins.char_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(self.builtins.bytestring_ty.clone(), maybe_char),
                    ))
                }
                "Data.ByteString.Char8.foldr" | "Data.ByteString.Char8.foldr'"
                | "Data.ByteString.Lazy.Char8.foldr" | "Data.ByteString.Lazy.Char8.foldr'" => {
                    // foldr :: (Char -> a -> a) -> a -> ByteString -> a
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.bytestring_ty.clone(), Ty::Var(a.clone()))),
                    ))
                }
                "Data.ByteString.Char8.foldl" | "Data.ByteString.Char8.foldl'"
                | "Data.ByteString.Lazy.Char8.foldl" | "Data.ByteString.Lazy.Char8.foldl'" => {
                    // foldl :: (a -> Char -> a) -> a -> ByteString -> a
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.char_ty.clone(), Ty::Var(a.clone()))),
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.bytestring_ty.clone(), Ty::Var(a.clone()))),
                    ))
                }
                "Data.ByteString.Char8.any" | "Data.ByteString.Char8.all"
                | "Data.ByteString.Lazy.Char8.any" | "Data.ByteString.Lazy.Char8.all" => {
                    // any/all :: (Char -> Bool) -> ByteString -> Bool
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bool_ty.clone()),
                    ))
                }
                "Data.ByteString.Char8.concatMap" | "Data.ByteString.Lazy.Char8.concatMap" => {
                    // concatMap :: (Char -> ByteString) -> ByteString -> ByteString
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bytestring_ty.clone()),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                "Data.ByteString.Char8.intercalate" | "Data.ByteString.Lazy.Char8.intercalate" => {
                    // intercalate :: ByteString -> [ByteString] -> ByteString
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        Ty::fun(Ty::List(Box::new(self.builtins.bytestring_ty.clone())), self.builtins.bytestring_ty.clone()),
                    ))
                }
                "Data.ByteString.Char8.uncons" | "Data.ByteString.Lazy.Char8.uncons" => {
                    // uncons :: ByteString -> Maybe (Char, ByteString)
                    let pair = Ty::Tuple(vec![self.builtins.char_ty.clone(), self.builtins.bytestring_ty.clone()]);
                    let maybe_pair = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(pair),
                    );
                    Scheme::mono(Ty::fun(self.builtins.bytestring_ty.clone(), maybe_pair))
                }
                // Data.ByteString: [Int] -> ByteString
                "Data.ByteString.pack" => {
                    Scheme::mono(Ty::fun(
                        Ty::List(Box::new(self.builtins.int_ty.clone())),
                        self.builtins.bytestring_ty.clone(),
                    ))
                }
                // Data.ByteString: ByteString -> [Int]
                "Data.ByteString.unpack" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        Ty::List(Box::new(self.builtins.int_ty.clone())),
                    ))
                }
                // Data.ByteString: ByteString -> Bool
                "Data.ByteString.null" => {
                    Scheme::mono(Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bool_ty.clone()))
                }
                // Data.ByteString: ByteString -> Int
                "Data.ByteString.length" | "Data.ByteString.head" | "Data.ByteString.last" => {
                    Scheme::mono(Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.int_ty.clone()))
                }
                // Data.ByteString: ByteString -> ByteString
                "Data.ByteString.tail" | "Data.ByteString.init" | "Data.ByteString.reverse" => {
                    Scheme::mono(Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()))
                }
                // Data.ByteString: ByteString -> ByteString -> ByteString
                "Data.ByteString.append" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                // Data.ByteString: Int -> ByteString -> ByteString
                "Data.ByteString.cons" | "Data.ByteString.take" | "Data.ByteString.drop" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                // Data.ByteString: ByteString -> Int -> ByteString
                "Data.ByteString.snoc" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        Ty::fun(self.builtins.int_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                // Data.ByteString: Int -> ByteString -> Bool
                "Data.ByteString.elem" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bool_ty.clone()),
                    ))
                }
                // Data.ByteString: ByteString -> Int -> Int
                "Data.ByteString.index" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        Ty::fun(self.builtins.int_ty.clone(), self.builtins.int_ty.clone()),
                    ))
                }
                // Data.ByteString: ByteString -> ByteString -> Bool
                "Data.ByteString.eq" | "Data.ByteString.isPrefixOf" | "Data.ByteString.isSuffixOf" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bool_ty.clone()),
                    ))
                }
                // Data.ByteString: ByteString -> ByteString -> Int
                "Data.ByteString.compare" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.int_ty.clone()),
                    ))
                }
                // Data.ByteString: String -> IO ByteString
                "Data.ByteString.readFile" => {
                    let io_bs = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(self.builtins.bytestring_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), io_bs))
                }
                // Data.ByteString: String -> ByteString -> IO ()
                "Data.ByteString.writeFile" => {
                    let io_unit = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::unit()),
                    );
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), io_unit),
                    ))
                }
                // Data.ByteString: (Int -> Bool) -> ByteString -> ByteString
                "Data.ByteString.filter" => {
                    // Polymorphic element for Char8 compat: (a -> Bool) -> ByteString -> ByteString
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                            Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                        ),
                    )
                }
                "Data.ByteString.hGetContents" | "Data.ByteString.hGet" => {
                    // hGetContents :: Handle -> IO ByteString (Handle polymorphic)
                    let io_bs = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(self.builtins.bytestring_ty.clone()),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), io_bs),
                    )
                }
                // Data.ByteString: (a -> b -> a) -> a -> ByteString -> a (b polymorphic for Char8)
                "Data.ByteString.foldl'" | "Data.ByteString.foldl" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone()))),
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.bytestring_ty.clone(), Ty::Var(a.clone()))),
                        ),
                    )
                }
                // Data.ByteString: [ByteString] -> ByteString
                "Data.ByteString.concat" => {
                    Scheme::mono(Ty::fun(
                        Ty::List(Box::new(self.builtins.bytestring_ty.clone())),
                        self.builtins.bytestring_ty.clone(),
                    ))
                }
                // Data.ByteString: ByteString -> [ByteString] -> ByteString
                "Data.ByteString.intercalate" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.bytestring_ty.clone(),
                        Ty::fun(
                            Ty::List(Box::new(self.builtins.bytestring_ty.clone())),
                            self.builtins.bytestring_ty.clone(),
                        ),
                    ))
                }
                // Data.ByteString: Int -> ByteString
                "Data.ByteString.replicate" => {
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(self.builtins.int_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                // Data.ByteString: (Int -> Int) -> ByteString -> ByteString
                "Data.ByteString.map" => {
                    // Polymorphic for Char8 compat: (a -> a) -> ByteString -> ByteString
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                            Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                        ),
                    )
                }
                // Data.ByteString: Handle -> IO ByteString
                "Data.ByteString.hGetContents" | "Data.ByteString.hGet" => {
                    let io_bs = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(self.builtins.bytestring_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(self.builtins.int_ty.clone(), io_bs))
                }
                // Data.ByteString: Handle -> ByteString -> IO ()
                "Data.ByteString.hPutStr" | "Data.ByteString.hPut" => {
                    let io_unit = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::unit()),
                    );
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), io_unit),
                    ))
                }
                // E.20: Data.Map — opaque polymorphic types
                // Map is treated as opaque (Ty::Var) — all ops use poly(a, b)
                "Data.Map.empty" => {
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(vec![a.clone(), b.clone()], map_kv)
                }
                "Data.Map.singleton" => {
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), map_kv)),
                    )
                }
                "Data.Map.null" => {
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(map_kv, self.builtins.bool_ty.clone()),
                    )
                }
                "Data.Map.isSubmapOf" => {
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(map_kv.clone(), Ty::fun(map_kv, self.builtins.bool_ty.clone())),
                    )
                }
                "Data.Map.size" => {
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(map_kv, self.builtins.int_ty.clone()),
                    )
                }
                "Data.Map.member" | "Data.Map.notMember" => {
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(map_kv, self.builtins.bool_ty.clone())),
                    )
                }
                "Data.Map.lookup" => {
                    // lookup :: Ord k => k -> Map k v -> Maybe v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(map_kv, self.builtins.maybe_of(Ty::Var(b.clone())))),
                    )
                }
                "Data.Map.!" => {
                    // (!) :: Ord k => Map k v -> k -> v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(map_kv, Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
                    )
                }
                "Data.Map.delete" => {
                    // delete :: Ord k => k -> Map k v -> Map k v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(map_kv.clone(), map_kv)),
                    )
                }
                "Data.Map.findWithDefault" => {
                    // findWithDefault :: Ord k => v -> k -> Map k v -> v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::fun(map_kv, Ty::Var(b.clone())))),
                    )
                }
                "Data.Map.insert" => {
                    // insert :: Ord k => k -> v -> Map k v -> Map k v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::fun(map_kv.clone(), map_kv))),
                    )
                }
                "Data.Map.insertWith" => {
                    // insertWith :: Ord k => (v -> v -> v) -> k -> v -> Map k v -> Map k v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    let merge_fn = Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(merge_fn, Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::fun(map_kv.clone(), map_kv)))),
                    )
                }
                "Data.Map.adjust" => {
                    // adjust :: Ord k => (v -> v) -> k -> Map k v -> Map k v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::fun(map_kv.clone(), map_kv))),
                    )
                }
                "Data.Map.update" => {
                    // update :: Ord k => (v -> Maybe v) -> k -> Map k v -> Map k v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(b.clone()), self.builtins.maybe_of(Ty::Var(b.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::fun(map_kv.clone(), map_kv))),
                    )
                }
                "Data.Map.union" | "Data.Map.intersection" | "Data.Map.difference" => {
                    // union :: Ord k => Map k v -> Map k v -> Map k v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(map_kv.clone(), Ty::fun(map_kv.clone(), map_kv)),
                    )
                }
                "Data.Map.map" => {
                    // map :: (v -> w) -> Map k v -> Map k w
                    let k = a.clone();
                    let v = b.clone();
                    let w = TyVar::new_star(0xFFFF_0002);
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con.clone())), Box::new(Ty::Var(k.clone())))), Box::new(Ty::Var(v.clone())));
                    let map_kw = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(k.clone())))), Box::new(Ty::Var(w.clone())));
                    let fun_vw = Ty::fun(Ty::Var(v.clone()), Ty::Var(w.clone()));
                    Scheme::poly(
                        vec![k, v, w],
                        Ty::fun(fun_vw, Ty::fun(map_kv, map_kw)),
                    )
                }
                "Data.Map.filter" => {
                    // filter :: (v -> Bool) -> Map k v -> Map k v
                    let k = a.clone();
                    let v = b.clone();
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(k.clone())))), Box::new(Ty::Var(v.clone())));
                    Scheme::poly(
                        vec![k, v],
                        Ty::fun(Ty::fun(Ty::Var(b.clone()), self.builtins.bool_ty.clone()), Ty::fun(map_kv.clone(), map_kv)),
                    )
                }
                "Data.Map.toList" | "Data.Map.toAscList" | "Data.Map.toDescList" | "Data.Map.assocs" => {
                    // toList :: Map k v -> [(k, v)]
                    let k = a.clone();
                    let v = b.clone();
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(k.clone())))), Box::new(Ty::Var(v.clone())));
                    let pair = Ty::Tuple(vec![Ty::Var(k.clone()), Ty::Var(v.clone())]);
                    Scheme::poly(vec![k, v], Ty::fun(map_kv, Ty::List(Box::new(pair))))
                }
                "Data.Map.keys" => {
                    // keys :: Map k v -> [k]
                    let k = a.clone();
                    let v = b.clone();
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(k.clone())))), Box::new(Ty::Var(v.clone())));
                    Scheme::poly(vec![k, v], Ty::fun(map_kv, Ty::List(Box::new(Ty::Var(a.clone())))))
                }
                "Data.Map.elems" => {
                    // elems :: Map k v -> [v]
                    let k = a.clone();
                    let v = b.clone();
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(k.clone())))), Box::new(Ty::Var(v.clone())));
                    Scheme::poly(vec![k, v], Ty::fun(map_kv, Ty::List(Box::new(Ty::Var(b.clone())))))
                }
                "Data.Map.fromList" => {
                    // fromList :: Ord k => [(k, v)] -> Map k v
                    let k = a.clone();
                    let v = b.clone();
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(k.clone())))), Box::new(Ty::Var(v.clone())));
                    let pair = Ty::Tuple(vec![Ty::Var(k.clone()), Ty::Var(v.clone())]);
                    Scheme::poly(vec![k, v], Ty::fun(Ty::List(Box::new(pair)), map_kv))
                }
                "Data.Map.fromListWith" => {
                    // fromListWith :: Ord k => (v -> v -> v) -> [(k, v)] -> Map k v
                    let k = a.clone();
                    let v = b.clone();
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(k.clone())))), Box::new(Ty::Var(v.clone())));
                    let pair = Ty::Tuple(vec![Ty::Var(k.clone()), Ty::Var(v.clone())]);
                    let merge_fn = Ty::fun(Ty::Var(v.clone()), Ty::fun(Ty::Var(v.clone()), Ty::Var(v.clone())));
                    Scheme::poly(vec![k, v], Ty::fun(merge_fn, Ty::fun(Ty::List(Box::new(pair)), map_kv)))
                }
                "Data.Map.foldr" => {
                    // foldr :: (v -> b -> b) -> b -> Map k v -> b
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let c = TyVar::new_star(0xFFFF_0003);
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(c.clone())))), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(map_kv, Ty::Var(b.clone())))),
                    )
                }
                "Data.Map.foldl" | "Data.Map.foldl'" | "Data.Map.foldr'" => {
                    // foldl :: (b -> v -> b) -> b -> Map k v -> b
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let c = TyVar::new_star(0xFFFF_0003);
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(c.clone())))), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c],
                        Ty::fun(Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(map_kv, Ty::Var(b.clone())))),
                    )
                }
                "Data.Map.alter" => {
                    // alter :: Ord k => (Maybe v -> Maybe v) -> k -> Map k v -> Map k v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    let maybe_b = self.builtins.maybe_of(Ty::Var(b.clone()));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(maybe_b.clone(), maybe_b), Ty::fun(Ty::Var(a.clone()), Ty::fun(map_kv.clone(), map_kv))),
                    )
                }
                "Data.Map.unionWith" | "Data.Map.intersectionWith" => {
                    // unionWith :: Ord k => (v -> v -> v) -> Map k v -> Map k v -> Map k v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    let merge_fn = Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(merge_fn, Ty::fun(map_kv.clone(), Ty::fun(map_kv.clone(), map_kv))),
                    )
                }
                "Data.Map.unions" => {
                    // unions :: Ord k => [Map k v] -> Map k v
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_kv = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(a.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::List(Box::new(map_kv.clone())), map_kv),
                    )
                }
                "Data.Map.mapMaybe" => {
                    // mapMaybe :: (v -> Maybe w) -> Map k v -> Map k w
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let c = TyVar::new_star(0xFFFF_0003);
                    let map_ka = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con.clone())), Box::new(Ty::Var(c.clone())))), Box::new(Ty::Var(a.clone())));
                    let map_kb = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(Ty::Var(c.clone())))), Box::new(Ty::Var(b.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), self.builtins.maybe_of(Ty::Var(b.clone()))), Ty::fun(map_ka, map_kb)),
                    )
                }
                "Data.Map.differenceWith" | "Data.Map.unionWithKey" | "Data.Map.mapWithKey"
                | "Data.Map.mapKeys" | "Data.Map.filterWithKey" | "Data.Map.foldrWithKey"
                | "Data.Map.foldlWithKey" | "Data.Map.keysSet"
                | "Data.Map.mapMaybeWithKey" => {
                    // Permissive fallback: a -> b
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    )
                }
                // Data.Set operations
                "Data.Set.empty" => {
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Data.Set.fromList" => {
                    // fromList :: Ord a => [a] -> Set a
                    let set_con = TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    let set_a = Ty::App(Box::new(Ty::Con(set_con)), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::List(Box::new(Ty::Var(a.clone()))), set_a),
                    )
                }
                "Data.Set.toList" | "Data.Set.toAscList" | "Data.Set.toDescList" | "Data.Set.elems" => {
                    // toList :: Set a -> [a]
                    let set_con = TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    let set_a = Ty::App(Box::new(Ty::Con(set_con)), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(set_a, Ty::List(Box::new(Ty::Var(a.clone())))),
                    )
                }
                "Data.Set.singleton" => {
                    // singleton :: a -> Set a
                    let set_con = TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    let set_a = Ty::App(Box::new(Ty::Con(set_con)), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), set_a),
                    )
                }
                "Data.Set.findMin" | "Data.Set.findMax" => {
                    // findMin/findMax :: Set a -> a
                    let set_con = TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    let set_a = Ty::App(Box::new(Ty::Con(set_con)), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(set_a, Ty::Var(a.clone())),
                    )
                }
                "Data.Set.lookupMin" | "Data.Set.lookupMax" => {
                    // lookupMin/lookupMax :: Set a -> Maybe a
                    let set_con = TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    let set_a = Ty::App(Box::new(Ty::Con(set_con)), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(set_a, self.builtins.maybe_of(Ty::Var(a.clone()))),
                    )
                }
                "Data.Set.unions" => {
                    // unions :: Ord a => [Set a] -> Set a
                    let set_con = TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    let set_a = Ty::App(Box::new(Ty::Con(set_con)), Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::List(Box::new(set_a.clone())), set_a),
                    )
                }
                "Data.Set.null" => {
                    let set_a = Ty::App(
                        Box::new(Ty::Con(TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))))),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(set_a, self.builtins.bool_ty.clone()),
                    )
                }
                "Data.Set.size" | "Data.Set.length" => {
                    let set_a = Ty::App(
                        Box::new(Ty::Con(TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))))),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(set_a, self.builtins.int_ty.clone()),
                    )
                }
                "Data.Set.member" | "Data.Set.notMember" => {
                    let set_a = Ty::App(
                        Box::new(Ty::Con(TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))))),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(set_a, self.builtins.bool_ty.clone())),
                    )
                }
                "Data.Set.insert" | "Data.Set.delete" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))),
                    )
                }
                "Data.Set.union" | "Data.Set.intersection" | "Data.Set.difference" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.Set.isSubsetOf" | "Data.Set.isProperSubsetOf" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone())),
                    )
                }
                "Data.Set.deleteMin" | "Data.Set.deleteMax" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                    )
                }
                "Data.Set.map" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
                    )
                }
                "Data.Set.filter" => {
                    let set_a = Ty::App(
                        Box::new(Ty::Con(TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))))),
                        Box::new(Ty::Var(a.clone())),
                    );
                    // filter :: (a -> Bool) -> Set a -> Set a
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()), Ty::fun(set_a.clone(), set_a)),
                    )
                }
                "Data.Set.partition" => {
                    let set_a = Ty::App(
                        Box::new(Ty::Con(TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))))),
                        Box::new(Ty::Var(a.clone())),
                    );
                    // partition :: (a -> Bool) -> Set a -> (Set a, Set a)
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                            Ty::fun(set_a.clone(), Ty::Tuple(vec![set_a.clone(), set_a])),
                        ),
                    )
                }
                "Data.Set.foldr" | "Data.Set.foldl"
                | "Data.Set.foldr'" | "Data.Set.foldl'" => {
                    let set_a = Ty::App(
                        Box::new(Ty::Con(TyCon::new(Symbol::intern("Set"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))))),
                        Box::new(Ty::Var(a.clone())),
                    );
                    // foldr :: (a -> b -> b) -> b -> Set a -> b
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))),
                            Ty::fun(Ty::Var(b.clone()), Ty::fun(set_a, Ty::Var(b.clone()))),
                        ),
                    )
                }

                // Data.IntMap operations
                "Data.IntMap.empty" => {
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Data.IntMap.singleton" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.IntMap.null" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                    )
                }
                "Data.IntMap.size" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), self.builtins.int_ty.clone()),
                    )
                }
                "Data.IntMap.member" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone())),
                    )
                }
                "Data.IntMap.lookup" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
                    )
                }
                "Data.IntMap.findWithDefault" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
                    )
                }
                "Data.IntMap.insert" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
                    )
                }
                "Data.IntMap.insertWith" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())), Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))),
                    )
                }
                "Data.IntMap.delete" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.IntMap.adjust" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())), Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
                    )
                }
                "Data.IntMap.union" | "Data.IntMap.intersection" | "Data.IntMap.difference" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.IntMap.unionWith" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
                    )
                }
                "Data.IntMap.map" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
                    )
                }
                "Data.IntMap.mapWithKey" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
                    )
                }
                "Data.IntMap.filter" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.IntMap.foldr" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
                    )
                }
                "Data.IntMap.foldlWithKey" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone())))), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone())))),
                    )
                }
                "Data.IntMap.keys" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                    )
                }
                "Data.IntMap.elems" | "Data.IntMap.toList"
                | "Data.IntMap.toAscList" | "Data.IntMap.fromList" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    )
                }

                // Data.IntSet operations
                "Data.IntSet.empty" => {
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Data.IntSet.singleton" => {
                    Scheme::mono(Ty::fun(self.builtins.int_ty.clone(), Ty::Var(a.clone())))
                }
                "Data.IntSet.null" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                    )
                }
                "Data.IntSet.size" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), self.builtins.int_ty.clone()),
                    )
                }
                "Data.IntSet.member" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone())),
                    )
                }
                "Data.IntSet.insert" | "Data.IntSet.delete" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.IntSet.union" | "Data.IntSet.intersection" | "Data.IntSet.difference" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.IntSet.isSubsetOf" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone())),
                    )
                }
                "Data.IntSet.filter" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::fun(self.builtins.int_ty.clone(), self.builtins.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.IntSet.foldr" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
                    )
                }
                "Data.IntSet.toList" | "Data.IntSet.fromList" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                    )
                }

                // E.25: String type class methods
                // fromString :: IsString a => String -> a
                "fromString" => {
                    let a = TyVar::new_star(0xFFFF_0100);
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::Var(a),
                    ))
                }
                // E.64: OverloadedLists — fromList :: [a] -> [a] (identity for lists)
                // Also serves as Text.Pandoc.Builder.fromList :: [a] -> Many a (≈ [a])
                "fromList" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
                }
                // toList :: [a] -> [a] (Many a ≈ [a] in our model)
                "toList" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
                }
                // unMany :: Many a -> Seq a (≈ [a] -> [a])
                "unMany" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
                }
                // singleton :: a -> [a] (works for lists, Many, etc.)
                "singleton" => {
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::List(Box::new(Ty::Var(a.clone()))),
                    ))
                }
                // isNull :: [a] -> Bool (for Many/Foldable)
                "isNull" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, self.builtins.bool_ty.clone()))
                }
                // trimInlines :: [Inline] -> [Inline]
                "trimInlines" => {
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    let list_inline = Ty::List(Box::new(Ty::Con(inline_con)));
                    Scheme::mono(Ty::fun(list_inline.clone(), list_inline))
                }
                // nullAttr :: (Text, [Text], [(Text, Text)])
                "nullAttr" => {
                    Scheme::mono(self.pandoc_attr_ty())
                }
                // nullMeta :: Meta
                "nullMeta" => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("Meta"), Kind::Star)))
                }
                // walk :: Walkable a b => (a -> a) -> b -> b
                "walk" => {
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                        Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())),
                    ))
                }
                // walkM :: (Walkable a b, Monad m) => (a -> m a) -> b -> m b
                "walkM" => {
                    let m = TyVar::new(0xFFFF_0003, Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    Scheme::poly(vec![a.clone(), b.clone(), m.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(a.clone())))),
                        Ty::fun(Ty::Var(b.clone()), Ty::App(Box::new(Ty::Var(m.clone())), Box::new(Ty::Var(b.clone())))),
                    ))
                }
                // query :: (Walkable a b, Monoid c) => (a -> c) -> b -> c
                "query" => {
                    Scheme::poly(vec![a.clone(), b.clone(), c.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                        Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                    ))
                }
                // doc :: [Block] -> Pandoc
                "doc" => {
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    let pandoc_con = TyCon::new(Symbol::intern("Pandoc"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::List(Box::new(Ty::Con(block_con))), Ty::Con(pandoc_con)))
                }
                // docTitle, docAuthors, docDate :: Pandoc -> [Inline]
                "docTitle" | "docDate" => {
                    let pandoc_con = TyCon::new(Symbol::intern("Pandoc"), Kind::Star);
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(pandoc_con), Ty::List(Box::new(Ty::Con(inline_con)))))
                }
                "docAuthors" => {
                    let pandoc_con = TyCon::new(Symbol::intern("Pandoc"), Kind::Star);
                    let inline_con = TyCon::new(Symbol::intern("Inline"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(pandoc_con), Ty::List(Box::new(Ty::List(Box::new(Ty::Con(inline_con)))))))
                }
                // lookupMeta :: Text -> Meta -> Maybe MetaValue
                "lookupMeta" => {
                    let meta_con = TyCon::new(Symbol::intern("Meta"), Kind::Star);
                    let mv_con = TyCon::new(Symbol::intern("MetaValue"), Kind::Star);
                    let maybe_mv = Ty::App(Box::new(Ty::Con(self.builtins.maybe_con.clone())), Box::new(Ty::Con(mv_con)));
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), Ty::fun(Ty::Con(meta_con), maybe_mv)))
                }
                // emptyCell :: Cell
                "emptyCell" => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("Cell"), Kind::Star)))
                }
                // emptyCaption :: Caption
                "emptyCaption" => {
                    Scheme::mono(Ty::Con(TyCon::new(Symbol::intern("Caption"), Kind::Star)))
                }
                // simpleCell :: [Block] -> Cell
                "simpleCell" => {
                    let cell_con = TyCon::new(Symbol::intern("Cell"), Kind::Star);
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::List(Box::new(Ty::Con(block_con))), Ty::Con(cell_con)))
                }
                // setMeta :: ToMetaValue a => Text -> a -> Meta -> Meta
                "setMeta" => {
                    let meta_con = TyCon::new(Symbol::intern("Meta"), Kind::Star);
                    let meta_ty = Ty::Con(meta_con);
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(meta_ty.clone(), meta_ty)),
                    ))
                }
                // toMetaValue :: ToMetaValue a => a -> MetaValue
                "toMetaValue" => {
                    let mv_con = TyCon::new(Symbol::intern("MetaValue"), Kind::Star);
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Con(mv_con)))
                }
                // Data.Monoid: getAny :: Any -> Bool, getAll :: All -> Bool
                "getAny" => {
                    let any_con = TyCon::new(Symbol::intern("Any"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(any_con), self.builtins.bool_ty.clone()))
                }
                "getAll" => {
                    let all_con = TyCon::new(Symbol::intern("All"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(all_con), self.builtins.bool_ty.clone()))
                }
                // unTableBodies :: [TableBody] -> [Row]
                "unTableBodies" => {
                    let tablebody_con = TyCon::new(Symbol::intern("TableBody"), Kind::Star);
                    let row_con = TyCon::new(Symbol::intern("Row"), Kind::Star);
                    Scheme::mono(Ty::fun(
                        Ty::List(Box::new(Ty::Con(tablebody_con))),
                        Ty::List(Box::new(Ty::Con(row_con))),
                    ))
                }
                // cellBody :: Cell -> [Block]
                "cellBody" => {
                    let cell_con = TyCon::new(Symbol::intern("Cell"), Kind::Star);
                    let block_con = TyCon::new(Symbol::intern("Block"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(cell_con), Ty::List(Box::new(Ty::Con(block_con)))))
                }
                // State monad operations
                // evalState :: State s a -> s -> a
                "evalState" => {
                    let s = TyVar::new_star(0xFFFF_0001);
                    let state_con = TyCon::new(Symbol::intern("State"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let state_sa = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(state_con)), Box::new(Ty::Var(s.clone())))), Box::new(Ty::Var(a.clone())));
                    let result = Ty::fun(state_sa, Ty::fun(Ty::Var(s.clone()), Ty::Var(a.clone())));
                    Scheme::poly(vec![s, a.clone()], result)
                }
                // runState :: State s a -> s -> (a, s)
                "runState" => {
                    let s = TyVar::new_star(0xFFFF_0001);
                    let state_con = TyCon::new(Symbol::intern("State"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let state_sa = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(state_con)), Box::new(Ty::Var(s.clone())))), Box::new(Ty::Var(a.clone())));
                    let result = Ty::fun(state_sa, Ty::fun(Ty::Var(s.clone()), Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(s.clone())])));
                    Scheme::poly(vec![s, a.clone()], result)
                }
                // execState :: State s a -> s -> s
                "execState" => {
                    let s = TyVar::new_star(0xFFFF_0001);
                    let state_con = TyCon::new(Symbol::intern("State"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let state_sa = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(state_con)), Box::new(Ty::Var(s.clone())))), Box::new(Ty::Var(a.clone())));
                    let result = Ty::fun(state_sa, Ty::fun(Ty::Var(s.clone()), Ty::Var(s.clone())));
                    Scheme::poly(vec![s, a.clone()], result)
                }
                // get :: MonadState s m => m s
                "get" => {
                    let s = TyVar::new_star(0xFFFF_0001);
                    let m = TyVar::new(0xFFFF_0003, Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    Scheme::poly(vec![s.clone(), m.clone()],
                        Ty::App(Box::new(Ty::Var(m)), Box::new(Ty::Var(s))))
                }
                // put :: MonadState s m => s -> m ()
                "put" => {
                    let s = TyVar::new_star(0xFFFF_0001);
                    let m = TyVar::new(0xFFFF_0003, Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    let unit_ty = Ty::Tuple(vec![]);
                    Scheme::poly(vec![s.clone(), m.clone()],
                        Ty::fun(Ty::Var(s), Ty::App(Box::new(Ty::Var(m)), Box::new(unit_ty))))
                }
                // modify :: MonadState s m => (s -> s) -> m ()
                "modify" | "modify'" => {
                    let s = TyVar::new_star(0xFFFF_0001);
                    let m = TyVar::new(0xFFFF_0003, Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    let unit_ty = Ty::Tuple(vec![]);
                    Scheme::poly(vec![s.clone(), m.clone()],
                        Ty::fun(Ty::fun(Ty::Var(s.clone()), Ty::Var(s)),
                            Ty::App(Box::new(Ty::Var(m)), Box::new(unit_ty))))
                }
                // gets :: MonadState s m => (s -> a) -> m a
                "gets" => {
                    let s = TyVar::new_star(0xFFFF_0001);
                    let m = TyVar::new(0xFFFF_0003, Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)));
                    Scheme::poly(vec![a.clone(), s.clone(), m.clone()],
                        Ty::fun(Ty::fun(Ty::Var(s), Ty::Var(a.clone())),
                            Ty::App(Box::new(Ty::Var(m)), Box::new(Ty::Var(a.clone())))))
                }
                "read" => Scheme::poly(vec![a.clone()], Ty::fun(
                    self.builtins.string_ty.clone(),
                    Ty::Var(a.clone()),
                )),
                // reads :: Read a => String -> [(a, String)]
                "reads" => {
                    let pair = Ty::Tuple(vec![Ty::Var(a.clone()), self.builtins.string_ty.clone()]);
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::List(Box::new(pair)),
                    ))
                }
                // lex :: String -> [(String, String)]
                "lex" => {
                    let pair = Ty::Tuple(vec![self.builtins.string_ty.clone(), self.builtins.string_ty.clone()]);
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::List(Box::new(pair)),
                    ))
                }
                "readMaybe" => {
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        self.builtins.string_ty.clone(),
                        maybe_a,
                    ))
                }

                // E.26: List *By variants, sortOn, stripPrefix, insert, mapAccumL/R
                "sortOn" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(list_a.clone(), list_a),
                        ),
                    )
                }
                "nubBy" | "groupBy" | "deleteBy" | "unionBy" | "intersectBy" => {
                    // These all take (a -> a -> Bool) as first arg
                    // Individual return types differ but the polymorphic scheme a -> a works
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let eq_closure = Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                    );
                    match name {
                        "nubBy" => Scheme::poly(
                            vec![a.clone()],
                            Ty::fun(eq_closure, Ty::fun(list_a.clone(), list_a)),
                        ),
                        "groupBy" => Scheme::poly(
                            vec![a.clone()],
                            Ty::fun(
                                eq_closure,
                                Ty::fun(list_a.clone(), Ty::List(Box::new(list_a))),
                            ),
                        ),
                        "deleteBy" => Scheme::poly(
                            vec![a.clone()],
                            Ty::fun(
                                eq_closure,
                                Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                            ),
                        ),
                        "unionBy" | "intersectBy" => Scheme::poly(
                            vec![a.clone()],
                            Ty::fun(
                                eq_closure,
                                Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)),
                            ),
                        ),
                        _ => unreachable!(),
                    }
                }
                "stripPrefix" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let maybe_list_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(list_a.clone()),
                    );
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(list_a.clone(), Ty::fun(list_a, maybe_list_a)),
                    )
                }
                "insert" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)),
                    )
                }
                "mapAccumL" | "mapAccumR" => {
                    let pair_ab = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                    let pair_a_listb = Ty::Tuple(vec![
                        Ty::Var(a.clone()),
                        Ty::List(Box::new(Ty::Var(b.clone()))),
                    ]);
                    let list_c = Ty::List(Box::new(Ty::Var(c.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(c.clone()), pair_ab)),
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(list_c, pair_a_listb)),
                        ),
                    )
                }

                // Data.Text.IO operations
                "Data.Text.IO.hPutStr" | "Data.Text.IO.hPutStrLn" => {
                    // Handle -> Text -> IO ()
                    let handle_ty = Ty::Con(TyCon::new(Symbol::intern("Handle"), Kind::Star));
                    let io_unit = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Tuple(vec![])),
                    );
                    Scheme::mono(Ty::fun(handle_ty, Ty::fun(self.builtins.text_ty.clone(), io_unit)))
                }
                "Data.Text.IO.hGetContents" | "Data.Text.IO.hGetLine" => {
                    // Handle -> IO Text
                    let handle_ty = Ty::Con(TyCon::new(Symbol::intern("Handle"), Kind::Star));
                    let io_text = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(self.builtins.text_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(handle_ty, io_text))
                }
                "Data.Text.IO.putStr" | "Data.Text.IO.putStrLn" => {
                    // Text -> IO ()
                    let io_unit = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(Ty::Tuple(vec![])),
                    );
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), io_unit))
                }
                "Data.Text.IO.getContents" | "Data.Text.IO.getLine" => {
                    // IO Text
                    let io_text = Ty::App(
                        Box::new(Ty::Con(self.builtins.io_con.clone())),
                        Box::new(self.builtins.text_ty.clone()),
                    );
                    Scheme::mono(io_text)
                }
                "Data.Text.IO.readFile" | "Data.Text.IO.writeFile" => {
                    // FilePath -> IO Text / FilePath -> Text -> IO ()
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))
                }

                // Data.Text: groupBy, splitOn, breakOn, etc. (missing from earlier stubs)
                "Data.Text.groupBy" => {
                    // (Char -> Char -> Bool) -> Text -> [Text]
                    let f_ty = Ty::fun(
                        self.builtins.char_ty.clone(),
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                    );
                    let list_text = Ty::List(Box::new(self.builtins.text_ty.clone()));
                    Scheme::mono(Ty::fun(f_ty, Ty::fun(self.builtins.text_ty.clone(), list_text)))
                }
                "Data.Text.splitOn" => {
                    // Text -> Text -> [Text]
                    let list_text = Ty::List(Box::new(self.builtins.text_ty.clone()));
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), list_text),
                    ))
                }
                "Data.Text.breakOn" | "Data.Text.breakOnEnd" => {
                    // Text -> Text -> (Text, Text)
                    let pair_text = Ty::Tuple(vec![
                        self.builtins.text_ty.clone(),
                        self.builtins.text_ty.clone(),
                    ]);
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), pair_text),
                    ))
                }
                "Data.Text.splitAt" => {
                    // Int -> Text -> (Text, Text)
                    let pair_text = Ty::Tuple(vec![
                        self.builtins.text_ty.clone(),
                        self.builtins.text_ty.clone(),
                    ]);
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(self.builtins.text_ty.clone(), pair_text),
                    ))
                }
                "Data.Text.span" | "Data.Text.break" => {
                    // (Char -> Bool) -> Text -> (Text, Text)
                    let pred_ty = Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone());
                    let pair_text = Ty::Tuple(vec![
                        self.builtins.text_ty.clone(),
                        self.builtins.text_ty.clone(),
                    ]);
                    Scheme::mono(Ty::fun(
                        pred_ty,
                        Ty::fun(self.builtins.text_ty.clone(), pair_text),
                    ))
                }
                "Data.Text.find" => {
                    // (Char -> Bool) -> Text -> Maybe Char
                    let pred_ty = Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone());
                    let maybe_char = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(self.builtins.char_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(pred_ty, Ty::fun(self.builtins.text_ty.clone(), maybe_char)))
                }
                "Data.Text.index" => {
                    // Text -> Int -> Char
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(self.builtins.int_ty.clone(), self.builtins.char_ty.clone()),
                    ))
                }
                "Data.Text.any" | "Data.Text.all" => {
                    // (Char -> Bool) -> Text -> Bool
                    let pred_ty = Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone());
                    Scheme::mono(Ty::fun(pred_ty, Ty::fun(self.builtins.text_ty.clone(), self.builtins.bool_ty.clone())))
                }
                "Data.Text.foldr" => {
                    // (Char -> a -> a) -> a -> Text -> a
                    let f_ty = Ty::fun(self.builtins.char_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        f_ty,
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.text_ty.clone(), Ty::Var(a.clone()))),
                    ))
                }
                "Data.Text.unfoldr" => {
                    // (a -> Maybe (Char, a)) -> a -> Text
                    let pair = Ty::Tuple(vec![self.builtins.char_ty.clone(), Ty::Var(a.clone())]);
                    let maybe_pair = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(pair),
                    );
                    let f_ty = Ty::fun(Ty::Var(a.clone()), maybe_pair);
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        f_ty,
                        Ty::fun(Ty::Var(a.clone()), self.builtins.text_ty.clone()),
                    ))
                }
                "Data.Text.replace" => {
                    // Text -> Text -> Text -> Text
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(
                            self.builtins.text_ty.clone(),
                            Ty::fun(self.builtins.text_ty.clone(), self.builtins.text_ty.clone()),
                        ),
                    ))
                }

                // Text.DocLayout operations — permissive stubs (Doc a ≈ a)
                // Using permissive types because Doc has IsString/Semigroup/Monoid
                // instances, and BHC doesn't yet fully integrate these with
                // OverloadedStrings to produce Doc values from string literals.
                "Text.DocLayout.empty" | "Text.DocLayout.cr" | "Text.DocLayout.blankline"
                | "Text.DocLayout.space" | "Text.DocLayout.blanklines" => {
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Text.DocLayout.text" | "Text.DocLayout.literal" | "Text.DocLayout.char" => {
                    // a -> b (permissive)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))
                }
                "Text.DocLayout.braces" | "Text.DocLayout.brackets" | "Text.DocLayout.parens"
                | "Text.DocLayout.quotes" | "Text.DocLayout.doubleQuotes"
                | "Text.DocLayout.chomp" | "Text.DocLayout.nowrap"
                | "Text.DocLayout.afterBreak" | "Text.DocLayout.inside"
                | "Text.DocLayout.flush" => {
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))
                }
                "Text.DocLayout.nest" | "Text.DocLayout.hang"
                | "Text.DocLayout.cblock"
                | "Text.DocLayout.lblock" | "Text.DocLayout.rblock" => {
                    // Int -> a -> a
                    Scheme::poly(vec![a.clone()], Ty::fun(self.builtins.int_ty.clone(), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))
                }
                "Text.DocLayout.$$" | "Text.DocLayout.<>" | "Text.DocLayout.$$"
                | "Text.DocLayout.<+>" | "Text.DocLayout.</>"
                | "Text.DocLayout.$+$" => {
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                    ))
                }
                "Text.DocLayout.vcat" | "Text.DocLayout.hcat" | "Text.DocLayout.hsep"
                | "Text.DocLayout.vsep" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, Ty::Var(a.clone())))
                }
                "Text.DocLayout.render" => {
                    // Maybe Int -> a -> Text
                    let maybe_int = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(self.builtins.int_ty.clone()),
                    );
                    Scheme::poly(vec![a.clone()], Ty::fun(maybe_int, Ty::fun(Ty::Var(a.clone()), self.builtins.text_ty.clone())))
                }
                "Text.DocLayout.isEmpty" => {
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()))
                }
                "Text.DocLayout.offset" | "Text.DocLayout.minOffset"
                | "Text.DocLayout.updateColumn" | "Text.DocLayout.height"
                | "Text.DocLayout.realLength" | "Text.DocLayout.charWidth" => {
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.builtins.int_ty.clone()))
                }
                "Text.DocLayout.prefixed" | "Text.DocLayout.beforeNonBlank" => {
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))))
                }

                // Text.Emoji operations
                "Text.Emoji.emojis" => {
                    // [(Text, Text)]
                    let pair_text = Ty::Tuple(vec![self.builtins.text_ty.clone(), self.builtins.text_ty.clone()]);
                    Scheme::mono(Ty::List(Box::new(pair_text)))
                }
                "Text.Emoji.emojiFromAlias" => {
                    // Text -> Maybe Text
                    let maybe_text = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(self.builtins.text_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), maybe_text))
                }
                "Text.Emoji.aliasesFromEmoji" => {
                    // Text -> [Text]
                    let list_text = Ty::List(Box::new(self.builtins.text_ty.clone()));
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), list_text))
                }

                // Network.Mime operations
                "Network.Mime.defaultMimeMap" => {
                    // Map Extension MimeType  (Map Text ByteString)
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_ty = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(self.builtins.text_ty.clone()))), Box::new(self.builtins.bytestring_ty.clone()));
                    Scheme::mono(map_ty)
                }
                "Network.Mime.defaultMimeType" => {
                    // MimeType (ByteString)
                    Scheme::mono(self.builtins.bytestring_ty.clone())
                }
                "Network.Mime.mimeByExt" => {
                    // Map Extension MimeType -> MimeType -> Text -> MimeType
                    let map_con = TyCon::new(Symbol::intern("Map"), Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star)))));
                    let map_ty = Ty::App(Box::new(Ty::App(Box::new(Ty::Con(map_con)), Box::new(self.builtins.text_ty.clone()))), Box::new(self.builtins.bytestring_ty.clone()));
                    Scheme::mono(Ty::fun(map_ty, Ty::fun(self.builtins.bytestring_ty.clone(), Ty::fun(self.builtins.text_ty.clone(), self.builtins.bytestring_ty.clone()))))
                }
                "Network.Mime.fileNameExtensions" => {
                    // a -> a -> a (permissive)
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))
                }

                // Control.Category operations
                "Control.Category.id" => {
                    // cat a a (use a -> a as stub)
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))
                }
                "Control.Category.<<<" | "Control.Category.." => {
                    // (b -> c) -> (a -> b) -> (a -> c) (function composition)
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                            Ty::fun(
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                            ),
                        ),
                    )
                }
                "Control.Category.>>>" => {
                    // (a -> b) -> (b -> c) -> (a -> c) (forward composition)
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            Ty::fun(
                                Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(c.clone())),
                            ),
                        ),
                    )
                }

                // Text.HTML.TagSoup.Entity operations
                "Text.HTML.TagSoup.Entity.htmlEntities"
                | "Text.HTML.TagSoup.Entity.xmlEntities" => {
                    // [(String, String)]
                    let pair = Ty::Tuple(vec![self.builtins.string_ty.clone(), self.builtins.string_ty.clone()]);
                    Scheme::mono(Ty::List(Box::new(pair)))
                }
                "Text.HTML.TagSoup.Entity.lookupEntity"
                | "Text.HTML.TagSoup.Entity.lookupNamedEntity" => {
                    // String -> [Tag String]  (permissive: String -> [a])
                    Scheme::poly(vec![a.clone()], Ty::fun(self.builtins.string_ty.clone(), Ty::List(Box::new(Ty::Var(a.clone())))))
                }
                "Text.HTML.TagSoup.Entity.lookupNumericEntity" => {
                    // String -> Maybe Char (permissive)
                    let maybe_char = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Con(TyCon::new(Symbol::intern("Char"), Kind::Star))),
                    );
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), maybe_char))
                }
                "Text.HTML.TagSoup.Entity.escapeXML" => {
                    // String -> String
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), self.builtins.string_ty.clone()))
                }

                // Commonmark.Entity operations
                "Commonmark.Entity.lookupEntity"
                | "Commonmark.lookupEntity" => {
                    // Text -> Maybe Text
                    let maybe_text = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(self.builtins.text_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), maybe_text))
                }

                // Text.Pandoc.XML.Light types — constructors
                // Attr :: QName -> Text -> Attr
                "Attr" | "Text.Pandoc.XML.Light.Attr" | "Text.Pandoc.XML.Light.Types.Attr" => {
                    let qname_con = TyCon::new(Symbol::intern("QName"), Kind::Star);
                    let attr_con = TyCon::new(Symbol::intern("Attr"), Kind::Star);
                    Scheme::mono(Ty::fun(
                        Ty::Con(qname_con),
                        Ty::fun(self.builtins.text_ty.clone(), Ty::Con(attr_con)),
                    ))
                }
                // QName :: Text -> Maybe Text -> Maybe Text -> QName
                "QName" | "Text.Pandoc.XML.Light.QName" | "Text.Pandoc.XML.Light.Types.QName" => {
                    let qname_con = TyCon::new(Symbol::intern("QName"), Kind::Star);
                    let maybe_text = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(self.builtins.text_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(
                        self.builtins.text_ty.clone(),
                        Ty::fun(maybe_text.clone(), Ty::fun(maybe_text, Ty::Con(qname_con))),
                    ))
                }
                // Element :: QName -> [Attr] -> [Content] -> Maybe Integer -> Element
                "Element" | "Text.Pandoc.XML.Light.Element" | "Text.Pandoc.XML.Light.Types.Element" => {
                    let qname_con = TyCon::new(Symbol::intern("QName"), Kind::Star);
                    let attr_con = TyCon::new(Symbol::intern("Attr"), Kind::Star);
                    let content_con = TyCon::new(Symbol::intern("Content"), Kind::Star);
                    let element_con = TyCon::new(Symbol::intern("Element"), Kind::Star);
                    let maybe_int = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(self.builtins.int_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(
                        Ty::Con(qname_con),
                        Ty::fun(
                            Ty::List(Box::new(Ty::Con(attr_con))),
                            Ty::fun(
                                Ty::List(Box::new(Ty::Con(content_con))),
                                Ty::fun(maybe_int, Ty::Con(element_con)),
                            ),
                        ),
                    ))
                }
                // Elem :: Element -> Content
                "Elem" | "Text.Pandoc.XML.Light.Elem" | "Text.Pandoc.XML.Light.Types.Elem" => {
                    let element_con = TyCon::new(Symbol::intern("Element"), Kind::Star);
                    let content_con = TyCon::new(Symbol::intern("Content"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(element_con), Ty::Con(content_con)))
                }
                // CRef :: Text -> Content
                "CRef" | "Text.Pandoc.XML.Light.CRef" | "Text.Pandoc.XML.Light.Types.CRef" => {
                    let content_con = TyCon::new(Symbol::intern("Content"), Kind::Star);
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), Ty::Con(content_con)))
                }

                // Text.Pandoc.XML.Light types and accessors
                "Text.Pandoc.XML.Light.elName" | "Text.Pandoc.XML.Light.Types.elName"
                | "Text.XML.Light.elName" | "elName" => {
                    // Element -> QName
                    let element_con = TyCon::new(Symbol::intern("Element"), Kind::Star);
                    let qname_con = TyCon::new(Symbol::intern("QName"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(element_con), Ty::Con(qname_con)))
                }
                "Text.Pandoc.XML.Light.elAttribs" | "Text.Pandoc.XML.Light.Types.elAttribs"
                | "Text.XML.Light.elAttribs" | "elAttribs" => {
                    // Element -> [Attr]
                    let element_con = TyCon::new(Symbol::intern("Element"), Kind::Star);
                    let attr_con = TyCon::new(Symbol::intern("Attr"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(element_con), Ty::List(Box::new(Ty::Con(attr_con)))))
                }
                "Text.Pandoc.XML.Light.elContent" | "Text.Pandoc.XML.Light.Types.elContent"
                | "Text.XML.Light.elContent" | "Text.Pandoc.XML.Light.elChildren"
                | "Text.Pandoc.XML.Light.Types.elChildren" | "elContent" | "elChildren" => {
                    // Element -> [Content]
                    let element_con = TyCon::new(Symbol::intern("Element"), Kind::Star);
                    let content_con = TyCon::new(Symbol::intern("Content"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(element_con), Ty::List(Box::new(Ty::Con(content_con)))))
                }
                "Text.Pandoc.XML.Light.qName" | "Text.Pandoc.XML.Light.Types.qName"
                | "Text.XML.Light.qName" | "qName" => {
                    // QName -> Text
                    let qname_con = TyCon::new(Symbol::intern("QName"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(qname_con), self.builtins.text_ty.clone()))
                }
                "Text.Pandoc.XML.Light.qURI" | "Text.Pandoc.XML.Light.Types.qURI"
                | "Text.XML.Light.qURI" | "Text.Pandoc.XML.Light.qPrefix"
                | "Text.Pandoc.XML.Light.Types.qPrefix" | "qURI" | "qPrefix" => {
                    // QName -> Maybe Text
                    let qname_con = TyCon::new(Symbol::intern("QName"), Kind::Star);
                    let maybe_text = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(self.builtins.text_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(Ty::Con(qname_con), maybe_text))
                }
                "Text.Pandoc.XML.Light.attrKey" | "Text.Pandoc.XML.Light.Types.attrKey"
                | "attrKey" => {
                    // Attr -> QName
                    let attr_con = TyCon::new(Symbol::intern("Attr"), Kind::Star);
                    let qname_con = TyCon::new(Symbol::intern("QName"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(attr_con), Ty::Con(qname_con)))
                }
                "Text.Pandoc.XML.Light.attrVal" | "Text.Pandoc.XML.Light.Types.attrVal"
                | "attrVal" => {
                    // Attr -> Text
                    let attr_con = TyCon::new(Symbol::intern("Attr"), Kind::Star);
                    Scheme::mono(Ty::fun(Ty::Con(attr_con), self.builtins.text_ty.clone()))
                }
                "Text.Pandoc.XML.Light.Proc.findAttr" | "Text.Pandoc.XML.Light.findAttr"
                | "Text.XML.Light.findAttr" => {
                    // QName -> Element -> Maybe Text (permissive: a -> b -> Maybe Text)
                    let maybe_text = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(self.builtins.text_ty.clone()),
                    );
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), maybe_text)))
                }
                "Text.Pandoc.XML.Light.Proc.findChild" | "Text.Pandoc.XML.Light.findChild"
                | "Text.Pandoc.XML.Light.Proc.findElement" | "Text.Pandoc.XML.Light.findElement"
                | "Text.XML.Light.findChild" | "Text.XML.Light.findElement" => {
                    // QName -> Element -> Maybe Element (permissive: a -> b -> Maybe b)
                    let maybe_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), maybe_b)))
                }
                "Text.Pandoc.XML.Light.Proc.findChildren" | "Text.Pandoc.XML.Light.findChildren"
                | "Text.Pandoc.XML.Light.Proc.filterChildren" | "Text.Pandoc.XML.Light.filterChildren"
                | "Text.XML.Light.findChildren" => {
                    // QName -> Element -> [Element] (permissive: a -> b -> [b])
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::List(Box::new(Ty::Var(b.clone()))))))
                }
                "Text.Pandoc.XML.Light.Proc.strContent" | "Text.Pandoc.XML.Light.strContent"
                | "Text.XML.Light.strContent" => {
                    // Element -> Text (permissive: a -> Text)
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.builtins.text_ty.clone()))
                }
                "Text.Pandoc.XML.Light.unqual" | "Text.Pandoc.XML.Light.Types.unqual"
                | "Text.XML.Light.unqual" => {
                    // Text -> QName (permissive: Text -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(self.builtins.text_ty.clone(), Ty::Var(a.clone())))
                }
                "Text.Pandoc.XML.Light.Output.showElement"
                | "Text.Pandoc.XML.Light.Output.ppElement"
                | "Text.Pandoc.XML.Light.Output.ppTopElement"
                | "Text.Pandoc.XML.Light.Output.showTopElement"
                | "Text.Pandoc.XML.Light.showElement"
                | "Text.Pandoc.XML.Light.ppElement"
                | "Text.Pandoc.XML.Light.ppTopElement"
                | "Text.XML.Light.showElement"
                | "Text.XML.Light.ppElement"
                | "Text.XML.Light.ppTopElement"
                | "Text.XML.Light.Output.ppElement"
                | "Text.XML.Light.Output.ppTopElement"
                | "Text.XML.Light.Output.showElement" => {
                    // Element -> Text (permissive: a -> Text)
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.builtins.text_ty.clone()))
                }
                "Text.Pandoc.XML.Light.parseXML" | "Text.XML.Light.parseXML" => {
                    // Text -> [Content] (permissive: Text -> [a])
                    Scheme::poly(vec![a.clone()], Ty::fun(self.builtins.text_ty.clone(), Ty::List(Box::new(Ty::Var(a.clone())))))
                }
                "Text.Pandoc.XML.Light.onlyElems" | "Text.XML.Light.onlyElems" => {
                    // [Content] -> [Element] (permissive: [a] -> [b])
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::List(Box::new(Ty::Var(a.clone()))), Ty::List(Box::new(Ty::Var(b.clone())))))
                }
                "Text.Pandoc.XML.Light.add_attr" | "Text.XML.Light.add_attr" => {
                    // Attr -> Element -> Element (permissive: a -> b -> b)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))))
                }
                "Text.Pandoc.XML.Light.add_attrs" | "Text.XML.Light.add_attrs" => {
                    // [Attr] -> Element -> Element (permissive: [a] -> b -> b)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::List(Box::new(Ty::Var(a.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))))
                }
                "Text.Pandoc.XML.Light.Proc.filterChildrenName"
                | "Text.Pandoc.XML.Light.filterChildrenName" => {
                    // (QName -> Bool) -> Element -> [Element] (permissive: (a -> Bool) -> b -> [b])
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                        Ty::fun(Ty::Var(b.clone()), Ty::List(Box::new(Ty::Var(b.clone())))),
                    ))
                }

                // Data.List operations (with typed stubs)
                "Data.List.isPrefixOf" | "Data.List.isSuffixOf" | "Data.List.isInfixOf" => {
                    // Eq a => [a] -> [a] -> Bool
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), Ty::fun(list_a, self.builtins.bool_ty.clone())))
                }
                "Data.List.stripPrefix" => {
                    // Eq a => [a] -> [a] -> Maybe [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let maybe_list = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(list_a.clone()),
                    );
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), Ty::fun(list_a, maybe_list)))
                }
                "Data.List.find" => {
                    // (a -> Bool) -> [a] -> Maybe a
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                        Ty::fun(list_a, maybe_a),
                    ))
                }
                "Data.List.nub" | "Data.List.sort" | "Data.List.reverse"
                | "Data.List.tails" | "Data.List.inits" => {
                    // [a] -> [a] (nub, sort) or [a] -> [[a]] (tails, inits)
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), list_a))
                }
                "Data.List.intercalate" => {
                    // [a] -> [[a]] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_list_a = Ty::List(Box::new(list_a.clone()));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), Ty::fun(list_list_a, list_a)))
                }
                "Data.List.intersperse" => {
                    // a -> [a] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)))
                }
                "Data.List.partition" | "Data.List.span" | "Data.List.break" => {
                    // (a -> Bool) -> [a] -> ([a], [a])
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let pair = Ty::Tuple(vec![list_a.clone(), list_a.clone()]);
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                        Ty::fun(list_a, pair),
                    ))
                }
                "Data.List.sortBy" | "Data.List.nubBy" => {
                    // (a -> a -> Ordering) -> [a] -> [a]  /  (a -> a -> Bool) -> [a] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
                        Ty::fun(list_a.clone(), list_a),
                    ))
                }
                "Data.List.sortOn" => {
                    // Ord b => (a -> b) -> [a] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                        Ty::fun(list_a.clone(), list_a),
                    ))
                }
                "Data.List.group" => {
                    // Eq a => [a] -> [[a]]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_list_a = Ty::List(Box::new(list_a.clone()));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, list_list_a))
                }
                "Data.List.groupBy" => {
                    // (a -> a -> Bool) -> [a] -> [[a]]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_list_a = Ty::List(Box::new(list_a.clone()));
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone())),
                        Ty::fun(list_a, list_list_a),
                    ))
                }
                "Data.List.delete" => {
                    // Eq a => a -> [a] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::fun(list_a.clone(), list_a)))
                }
                "Data.List.union" | "Data.List.intersect" => {
                    // Eq a => [a] -> [a] -> [a]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a.clone(), Ty::fun(list_a.clone(), list_a)))
                }
                "Data.List.transpose" | "Data.List.subsequences" | "Data.List.permutations" => {
                    // [[a]] -> [[a]] or [a] -> [[a]]
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    let list_list_a = Ty::List(Box::new(list_a.clone()));
                    Scheme::poly(vec![a.clone()], Ty::fun(list_a, list_list_a))
                }
                "Data.List.foldl'" => {
                    // (b -> a -> b) -> b -> [a] -> b
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, Ty::Var(b.clone()))),
                    ))
                }
                "Data.List.unfoldr" => {
                    // (b -> Maybe (a, b)) -> b -> [a]
                    let pair = Ty::Tuple(vec![Ty::Var(a.clone()), Ty::Var(b.clone())]);
                    let maybe_pair = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(pair),
                    );
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        Ty::fun(Ty::Var(b.clone()), maybe_pair),
                        Ty::fun(Ty::Var(b.clone()), list_a),
                    ))
                }
                "Data.List.genericLength" => {
                    // [a] -> b  (Num b)
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(list_a, Ty::Var(b.clone())))
                }

                // System.FilePath operations
                "System.FilePath.takeExtension" | "System.FilePath.dropExtension"
                | "System.FilePath.takeFileName" | "System.FilePath.takeDirectory"
                | "System.FilePath.takeBaseName" | "System.FilePath.dropTrailingPathSeparator" => {
                    // FilePath -> FilePath (String -> String)
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), self.builtins.string_ty.clone()))
                }
                "System.FilePath.Posix.isAbsolute" | "System.FilePath.Windows.isAbsolute"
                | "System.FilePath.isAbsolute" | "System.FilePath.isValid"
                | "System.FilePath.isRelative" | "System.FilePath.hasExtension" => {
                    // FilePath -> Bool
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), self.builtins.bool_ty.clone()))
                }
                "System.FilePath.Posix.combine" | "System.FilePath.Windows.combine"
                | "System.FilePath.combine" | "System.FilePath.</>"
                | "System.FilePath.addExtension" | "System.FilePath.replaceExtension"
                | "System.FilePath.replaceFileName" | "System.FilePath.replaceDirectory" => {
                    // FilePath -> FilePath -> FilePath
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), Ty::fun(self.builtins.string_ty.clone(), self.builtins.string_ty.clone())))
                }
                "System.FilePath.Posix.splitPath" | "System.FilePath.Windows.splitPath"
                | "System.FilePath.splitPath" | "System.FilePath.splitDirectories" => {
                    // FilePath -> [FilePath]
                    let list_str = Ty::List(Box::new(self.builtins.string_ty.clone()));
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), list_str))
                }

                // Text.Printf (permissive)
                "Text.Printf.printf" | "Text.Printf.hPrintf" | "Text.Printf.sprintf" => {
                    // a -> b (very permissive — printf is polyvariadic)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))
                }

                // Network.URI operations
                "Network.URI.escapeURIString" => {
                    // (Char -> Bool) -> String -> String
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(self.builtins.string_ty.clone(), self.builtins.string_ty.clone()),
                    ))
                }
                "Network.URI.unEscapeString" => {
                    // String -> String
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), self.builtins.string_ty.clone()))
                }
                "Network.URI.parseURI" | "Network.URI.parseURIReference"
                | "Network.URI.parseRelativeReference" | "Network.URI.parseAbsoluteURI" => {
                    // String -> Maybe URI  (use a -> Maybe a as stub)
                    let maybe_a = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone()], Ty::fun(self.builtins.string_ty.clone(), maybe_a))
                }
                "Network.URI.isURI" | "Network.URI.isAbsoluteURI"
                | "Network.URI.isRelativeReference" | "Network.URI.isURIReference" => {
                    // String -> Bool
                    Scheme::mono(Ty::fun(self.builtins.string_ty.clone(), self.builtins.bool_ty.clone()))
                }
                "Network.URI.uriScheme" | "Network.URI.uriPath" | "Network.URI.uriQuery"
                | "Network.URI.uriFragment" | "Network.URI.uriRegName"
                | "Network.URI.uriUserInfo" | "Network.URI.uriPort" => {
                    // a -> String (record selector, permissive)
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.builtins.string_ty.clone()))
                }
                "Network.URI.uriAuthority" => {
                    // a -> Maybe b (permissive)
                    let maybe_b = Ty::App(
                        Box::new(Ty::Con(self.builtins.maybe_con.clone())),
                        Box::new(Ty::Var(b.clone())),
                    );
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), maybe_b))
                }

                // Network.HTTP.Types operations
                "Network.HTTP.Types.urlEncode" | "Network.HTTP.Types.urlDecode" => {
                    // Bool -> ByteString -> ByteString
                    Scheme::mono(Ty::fun(
                        self.builtins.bool_ty.clone(),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }

                // Data.ByteString.Base64 operations
                "Data.ByteString.Base64.encode" | "Data.ByteString.Base64.decodeLenient" => {
                    // ByteString -> ByteString
                    Scheme::mono(Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()))
                }
                "Data.ByteString.Base64.decode" => {
                    // ByteString -> Either String ByteString
                    let either_ty = Ty::App(
                        Box::new(Ty::App(
                            Box::new(Ty::Con(self.builtins.either_con.clone())),
                            Box::new(self.builtins.string_ty.clone()),
                        )),
                        Box::new(self.builtins.bytestring_ty.clone()),
                    );
                    Scheme::mono(Ty::fun(self.builtins.bytestring_ty.clone(), either_ty))
                }

                // Data.Attoparsec.Text operations (use permissive a -> b for most)
                "Data.Attoparsec.Text.string" => {
                    // Text -> Parser Text (use Text -> a as stub)
                    Scheme::poly(vec![a.clone()], Ty::fun(self.builtins.text_ty.clone(), Ty::Var(a.clone())))
                }
                "Data.Attoparsec.Text.char" => {
                    // Char -> Parser Char (use Char -> a as stub)
                    Scheme::poly(vec![a.clone()], Ty::fun(self.builtins.char_ty.clone(), Ty::Var(a.clone())))
                }
                "Data.Attoparsec.Text.satisfy" | "Data.Attoparsec.Text.satisfyWith" => {
                    // (Char -> Bool) -> Parser Char (use (Char -> Bool) -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::Var(a.clone()),
                    ))
                }
                "Data.Attoparsec.Text.takeWhile" | "Data.Attoparsec.Text.takeWhile1"
                | "Data.Attoparsec.Text.skipWhile" | "Data.Attoparsec.Text.takeTill" => {
                    // (Char -> Bool) -> Parser Text (use (Char -> Bool) -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::Var(a.clone()),
                    ))
                }
                "Data.Attoparsec.Text.parseOnly" => {
                    // Parser a -> Text -> Either String a
                    let either_ty = Ty::App(
                        Box::new(Ty::App(
                            Box::new(Ty::Con(self.builtins.either_con.clone())),
                            Box::new(self.builtins.string_ty.clone()),
                        )),
                        Box::new(Ty::Var(a.clone())),
                    );
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        Ty::Var(b.clone()),
                        Ty::fun(self.builtins.text_ty.clone(), either_ty),
                    ))
                }
                "Data.Attoparsec.Text.endOfInput" | "Data.Attoparsec.Text.endOfLine" => {
                    // Parser () — use a as stub
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Data.Attoparsec.Text.inClass" | "Data.Attoparsec.Text.notInClass" => {
                    // String -> Char -> Bool
                    Scheme::mono(Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::fun(self.builtins.char_ty.clone(), self.builtins.bool_ty.clone()),
                    ))
                }
                "Data.Attoparsec.Text.option" => {
                    // a -> Parser a -> Parser a (use a -> a -> a as permissive stub)
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                    ))
                }

                // Text.Parsec operations
                "Text.Parsec.Prim.getPosition" | "Text.Parsec.getPosition"
                | "Text.Parsec.Pos.getPosition" => {
                    // m SourcePos (use a as stub)
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Text.Parsec.Prim.getInput" | "Text.Parsec.getInput" => {
                    // m s (use a as stub)
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Text.Parsec.Prim.setInput" | "Text.Parsec.setInput" => {
                    // s -> m () (use a -> b as stub)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))
                }
                "Text.Parsec.Prim.setPosition" | "Text.Parsec.setPosition"
                | "Text.Parsec.Pos.setPosition" => {
                    // SourcePos -> m () (use a -> b as stub)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))
                }
                "Text.Parsec.Pos.newPos" => {
                    // String -> Int -> Int -> SourcePos (use String -> Int -> Int -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        self.builtins.string_ty.clone(),
                        Ty::fun(self.builtins.int_ty.clone(),
                            Ty::fun(self.builtins.int_ty.clone(), Ty::Var(a.clone()))),
                    ))
                }
                "Text.Parsec.Pos.sourceLine" | "Text.Parsec.Pos.sourceColumn" => {
                    // SourcePos -> Int (use a -> Int)
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.builtins.int_ty.clone()))
                }
                "Text.Parsec.Pos.sourceName" => {
                    // SourcePos -> String (use a -> String)
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), self.builtins.string_ty.clone()))
                }
                "Text.Parsec.Pos.incSourceLine" | "Text.Parsec.Pos.incSourceColumn" => {
                    // SourcePos -> Int -> SourcePos (use a -> Int -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(self.builtins.int_ty.clone(), Ty::Var(a.clone())),
                    ))
                }
                "Text.Parsec.Pos.setSourceLine" | "Text.Parsec.Pos.setSourceColumn" => {
                    // SourcePos -> Int -> SourcePos (use a -> Int -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(self.builtins.int_ty.clone(), Ty::Var(a.clone())),
                    ))
                }
                "Text.Parsec.Pos.setSourceName" => {
                    // SourcePos -> String -> SourcePos (use a -> String -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(self.builtins.string_ty.clone(), Ty::Var(a.clone())),
                    ))
                }
                "Text.Parsec.Char.letter" | "Text.Parsec.Char.digit"
                | "Text.Parsec.Char.anyChar" | "Text.Parsec.Char.space"
                | "Text.Parsec.Char.upper" | "Text.Parsec.Char.lower"
                | "Text.Parsec.Char.alphaNum" | "Text.Parsec.Char.newline" => {
                    // Parser Char (use a as stub)
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Text.Parsec.Char.char" => {
                    // Char -> Parser Char (use Char -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(self.builtins.char_ty.clone(), Ty::Var(a.clone())))
                }
                "Text.Parsec.Char.string" => {
                    // String -> Parser String (use String -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(self.builtins.string_ty.clone(), Ty::Var(a.clone())))
                }
                "Text.Parsec.Char.satisfy" | "Text.Parsec.Char.noneOf" | "Text.Parsec.Char.oneOf" => {
                    // (Char -> Bool) -> Parser Char or String -> Parser Char (use a -> b)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))
                }
                "Text.Parsec.Combinator.many1" | "Text.Parsec.Combinator.skipMany"
                | "Text.Parsec.Combinator.skipMany1" | "Text.Parsec.Combinator.optional" => {
                    // Parser a -> Parser [a] (use a -> b as permissive)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))
                }
                "Text.Parsec.Combinator.choice" => {
                    // [Parser a] -> Parser a (use [a] -> a as permissive)
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::List(Box::new(Ty::Var(a.clone()))),
                        Ty::Var(a.clone()),
                    ))
                }
                "Text.Parsec.Combinator.try" | "Text.Parsec.try"
                | "Text.Parsec.Prim.try" | "Text.Parsec.Combinator.lookAhead" => {
                    // Parser a -> Parser a (use a -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))
                }
                "Text.Parsec.Combinator.between" => {
                    // Parser open -> Parser close -> Parser a -> Parser a (use a -> a -> b -> b)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))),
                    ))
                }
                "Text.Parsec.Combinator.sepBy" | "Text.Parsec.Combinator.sepBy1"
                | "Text.Parsec.Combinator.endBy" | "Text.Parsec.Combinator.endBy1"
                | "Text.Parsec.Combinator.sepEndBy" | "Text.Parsec.Combinator.sepEndBy1" => {
                    // Parser a -> Parser sep -> Parser [a] (use a -> b -> c as permissive)
                    Scheme::poly(vec![a.clone(), b.clone(), c.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                    ))
                }
                "Text.Parsec.Combinator.count" => {
                    // Int -> Parser a -> Parser [a] (use Int -> a -> b)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    ))
                }
                "Text.Parsec.Combinator.option" | "Text.Parsec.Combinator.optionMaybe" => {
                    // a -> Parser a -> Parser a (use a -> a -> a)
                    Scheme::poly(vec![a.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                    ))
                }
                "Text.Parsec.Combinator.notFollowedBy" | "Text.Parsec.Combinator.eof" => {
                    // Parser a -> Parser () or Parser () (use a -> b or a)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))
                }
                "Text.Parsec.Combinator.manyTill" | "Text.Parsec.Combinator.chainl1"
                | "Text.Parsec.Combinator.chainr1" => {
                    // a -> b -> c (permissive)
                    Scheme::poly(vec![a.clone(), b.clone(), c.clone()], Ty::fun(
                        Ty::Var(a.clone()),
                        Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone())),
                    ))
                }
                "Text.Parsec.Prim.runParser" | "Text.Parsec.Prim.runParserT"
                | "Text.Parsec.runParser" | "Text.Parsec.runParserT"
                | "Text.Parsec.Prim.parse" | "Text.Parsec.parse" => {
                    // a -> b -> c -> d (permissive)
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone()))),
                    )
                }
                "Text.Parsec.Prim.tokenPrim" | "Text.Parsec.tokenPrim" => {
                    // a -> b -> c -> d (permissive)
                    Scheme::poly(
                        vec![a.clone(), b.clone(), c.clone()],
                        Ty::fun(Ty::Var(a.clone()),
                            Ty::fun(Ty::Var(b.clone()), Ty::Var(c.clone()))),
                    )
                }
                "Text.Parsec.Prim.updateParserState" | "Text.Parsec.updateParserState" => {
                    // (State -> State) -> m () (use (a -> a) -> b as stub)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                        Ty::Var(b.clone()),
                    ))
                }
                "Text.Parsec.Prim.getState" | "Text.Parsec.getState" => {
                    // m s (use a as stub)
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Text.Parsec.Prim.putState" | "Text.Parsec.putState"
                | "Text.Parsec.Prim.modifyState" | "Text.Parsec.modifyState" => {
                    // s -> m () or (s -> s) -> m () (use a -> b)
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))
                }

                // Unknown builtins - skip here, will be handled in second pass
                _ => continue,
            };

            // Register the builtin with its DefId from the lowering pass
            self.env.insert_global(def_info.id, scheme);
        }

        // Second pass: register any remaining definitions (imported items not in builtins)
        // with their interface type schemes (if available) or fresh type variables.
        // We do this in a separate pass to avoid borrow conflicts with the closures above.
        for (_def_id, def_info) in defs.iter() {
            // Skip constructors (handled in first pass)
            if matches!(
                def_info.kind,
                DefKind::Constructor | DefKind::StubConstructor
            ) {
                continue;
            }
            // Skip if already registered
            if self.env.lookup_global(def_info.id).is_some() {
                continue;
            }
            // Use the type scheme from the interface file if available,
            // otherwise fall back to a fresh type variable
            let scheme = if let Some(ref iface_scheme) = def_info.type_scheme {
                iface_scheme.clone()
            } else {
                let fresh = self.fresh_ty();
                Scheme::mono(fresh)
            };
            self.env.insert_global(def_info.id, scheme);
        }
    }

    /// Register a data type definition.
    pub fn register_data_type(&mut self, data: &DataDef) {
        // Register the type constructor
        let kind = Self::compute_type_con_kind(data.params.len());
        let tycon = TyCon::new(data.name, kind);
        self.env.register_type_con(tycon);

        // Track GADT types for case expression refinement
        if data.is_gadt {
            self.gadt_types.insert(data.name);
        }

        // Register data constructors and field accessors
        for con in &data.cons {
            let scheme = self.compute_data_con_scheme(data, con);
            self.env.register_data_con(con.id, con.name, scheme);

            // Track constructor → type mapping for RecordUpdate
            self.type_to_data_cons
                .entry(data.name)
                .or_default()
                .push(con.id);

            // Register field accessor functions for record constructors
            if let ConFields::Named(fields) = &con.fields {
                // Build the data type: T a1 a2 ... an
                let data_ty = Self::build_applied_type(data.name, &data.params);

                // Store field definitions for record type checking
                let field_defs: Vec<(Symbol, Ty)> =
                    fields.iter().map(|f| (f.name, f.ty.clone())).collect();
                self.con_field_defs.insert(con.id, field_defs);

                for field in fields {
                    // Field accessor type: T a1 ... an -> FieldType
                    let accessor_ty = Ty::fun(data_ty.clone(), field.ty.clone());
                    let accessor_scheme = Scheme::poly(data.params.clone(), accessor_ty);
                    // Register the field accessor as a global value
                    self.env.insert_global(field.id, accessor_scheme);

                    // Track field name → (constructor, accessor) for FieldAccess
                    self.field_name_to_con
                        .entry(field.name)
                        .or_default()
                        .push((con.id, field.id));
                }
            }
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

        // Track constructor → type mapping for RecordUpdate
        self.type_to_data_cons
            .entry(newtype.name)
            .or_default()
            .push(newtype.con.id);

        // Register field accessor if this is a record-style newtype
        if let ConFields::Named(fields) = &newtype.con.fields {
            // Store field definitions for record construction type checking
            let field_defs: Vec<(Symbol, Ty)> =
                fields.iter().map(|f| (f.name, f.ty.clone())).collect();
            self.con_field_defs.insert(newtype.con.id, field_defs);

            // Build the newtype: T a1 a2 ... an
            let newtype_ty = Self::build_applied_type(newtype.name, &newtype.params);

            for field in fields {
                // Field accessor type: T a1 ... an -> FieldType
                let accessor_ty = Ty::fun(newtype_ty.clone(), field.ty.clone());
                let accessor_scheme = Scheme::poly(newtype.params.clone(), accessor_ty);
                // Register the field accessor as a global value
                self.env.insert_global(field.id, accessor_scheme);

                // Track field name → (constructor, accessor) for FieldAccess
                self.field_name_to_con
                    .entry(field.name)
                    .or_default()
                    .push((newtype.con.id, field.id));
            }
        }
    }

    /// Look up the named fields for a record constructor.
    ///
    /// Returns the fields as (name, type) pairs if the constructor is a record type,
    /// or None if it's a positional constructor.
    #[must_use]
    pub fn get_con_fields(&self, def_id: DefId) -> Option<&[(Symbol, Ty)]> {
        self.con_field_defs
            .get(&def_id)
            .map(|v: &Vec<(Symbol, Ty)>| v.as_slice())
    }

    /// Register a type class definition.
    pub fn register_class(&mut self, class: &ClassDef) {
        // Build method signatures map
        let methods: FxHashMap<Symbol, Scheme> = class
            .methods
            .iter()
            .map(|m| (m.name, m.ty.clone()))
            .collect();

        // Convert HIR fundeps to typechecker fundeps
        let fundeps: Vec<crate::env::FunDep> = class
            .fundeps
            .iter()
            .map(|fd| crate::env::FunDep {
                from: fd.from.clone(),
                to: fd.to.clone(),
            })
            .collect();

        // Convert associated type declarations
        let assoc_types = class
            .assoc_types
            .iter()
            .map(|at| crate::env::AssocTypeInfo {
                name: at.name,
                params: at.params.clone(),
                kind: at.kind.clone(),
                default: at.default.clone(),
            })
            .collect();

        let info = ClassInfo {
            name: class.name,
            params: class.params.clone(),
            fundeps,
            supers: class.supers.clone(),
            methods: methods.clone(),
            assoc_types,
        };

        self.env.register_class(info);

        // Track this as a user-defined class for dict-passing
        self.user_defined_classes.insert(class.name);

        // Register class methods as globally available functions.
        // Each method gets its declared type scheme with the class constraint
        // added (it's implicit from the class declaration).
        for method in &class.methods {
            let mut method_scheme = method.ty.clone();

            // Add the class constraint to the method's scheme if not already present.
            // For example, `describe :: a -> String` in `class Describable a` becomes
            // `describe :: Describable a => a -> String`.
            if !class.params.is_empty() {
                let class_constraint = Constraint::new_multi(
                    class.name,
                    class.params.iter().map(|p| Ty::Var(p.clone())).collect(),
                    method.span,
                );
                if !method_scheme.constraints.iter().any(|c| c.class == class.name) {
                    method_scheme.constraints.push(class_constraint);
                }
                // Ensure type variables from the class are in the scheme's vars
                for param in &class.params {
                    if !method_scheme.vars.iter().any(|v| v.id == param.id) {
                        method_scheme.vars.push(param.clone());
                    }
                }
            }

            self.env
                .insert_global_by_name(method.name, method_scheme.clone());

            // Also store by DefId so hir-to-core can look it up
            self.def_schemes.insert(method.id, method_scheme);
        }

        // Type-check default method implementations.
        // Default methods must conform to their declared signatures.
        for default in &class.defaults {
            self.check_value_def(default);
        }
    }

    /// Register a type class instance.
    pub fn register_instance(&mut self, instance: &InstanceDef) {
        // Build method implementations map
        let methods = instance.methods.iter().map(|m| (m.name, m.id)).collect();

        // Convert associated type implementations
        let assoc_type_impls = instance
            .assoc_type_impls
            .iter()
            .map(|impl_| crate::env::AssocTypeImpl {
                name: impl_.name,
                args: impl_.args.clone(),
                rhs: impl_.rhs.clone(),
            })
            .collect();

        let info = InstanceInfo {
            class: instance.class,
            types: instance.types.clone(),
            context: instance.constraints.clone(),
            methods,
            assoc_type_impls,
        };

        self.env.register_instance(info);

        // Type check the instance method implementations.
        // If the class is unknown (e.g., Stream from an external package),
        // skip body type-checking to avoid spurious errors.
        let class_known = self.env.lookup_class(instance.class).is_some();
        for method in &instance.methods {
            if class_known {
                self.check_value_def(method);
            } else {
                // Still register the method's scheme so it can be looked up,
                // but use a fresh type variable to avoid constraining other code.
                let fresh = self.fresh_ty();
                let scheme = self.generalize(&fresh);
                self.def_schemes.insert(method.id, scheme.clone());
                self.env.insert_global(method.id, scheme);
            }
        }
    }

    /// Register a standalone type family in the type environment.
    pub fn register_type_family(&mut self, tf: &bhc_hir::TypeFamilyDef) {
        // Register as a type constructor so it can appear in type expressions.
        // Build a kind from arity: e.g., 2 params -> * -> * -> *
        let kind = if tf.params.is_empty() {
            tf.kind.clone()
        } else {
            Self::compute_type_con_kind(tf.params.len())
        };
        let tycon = TyCon { name: tf.name, kind };
        self.env.register_type_con(tycon);

        // Register the family info for reduction
        let equations = tf
            .equations
            .iter()
            .map(|eq| crate::env::TypeFamilyEquation {
                args: eq.args.clone(),
                rhs: eq.rhs.clone(),
            })
            .collect();

        let info = crate::env::TypeFamilyInfo {
            name: tf.name,
            params: tf.params.clone(),
            kind: tf.kind.clone(),
            is_closed: tf.family_kind == bhc_hir::TypeFamilyKind::Closed,
            equations,
        };
        self.env.register_type_family(info);
    }

    /// Register a standalone type family instance (for open families).
    pub fn register_type_family_instance(&mut self, inst: &bhc_hir::TypeFamilyInstance) {
        let eqn = crate::env::TypeFamilyEquation {
            args: inst.args.clone(),
            rhs: inst.rhs.clone(),
        };
        self.env.register_type_family_instance(inst.name, eqn);
    }

    /// Register a standalone data family in the type environment.
    pub fn register_data_family(&mut self, df: &bhc_hir::DataFamilyDef) {
        // Register as a type constructor so it can appear in type expressions
        let kind = if df.params.is_empty() {
            df.kind.clone()
        } else {
            Self::compute_type_con_kind(df.params.len())
        };
        let tycon = TyCon { name: df.name, kind };
        self.env.register_type_con(tycon);
    }

    /// Register a data family instance.
    pub fn register_data_family_instance(&mut self, inst: &bhc_hir::DataFamilyInstance) {
        use bhc_hir::ConFields;

        // Register each constructor as a data constructor
        for con in &inst.cons {
            // Build the constructor type scheme
            // Return type is FamilyName arg1 arg2 ...
            let result_ty = {
                let base_kind = Self::compute_type_con_kind(inst.args.len());
                let base = Ty::Con(TyCon::new(inst.family_name, base_kind));
                inst.args
                    .iter()
                    .fold(base, |acc, arg| Ty::App(Box::new(acc), Box::new(arg.clone())))
            };

            let field_types = match &con.fields {
                ConFields::Positional(tys) => tys.clone(),
                ConFields::Named(fields) => fields.iter().map(|f| f.ty.clone()).collect(),
            };

            // Build: field1 -> field2 -> ... -> FamilyName args
            let con_ty = field_types
                .into_iter()
                .rev()
                .fold(result_ty, |acc, field_ty| Ty::fun(field_ty, acc));

            // Quantify over free type variables
            let free = con_ty.free_vars();
            let scheme = if free.is_empty() {
                Scheme::mono(con_ty)
            } else {
                Scheme::poly(free, con_ty)
            };

            self.env.register_data_con(con.id, con.name, scheme);
        }

        // Track instance for resolution
        let info = crate::env::DataFamilyInstanceInfo {
            args: inst.args.clone(),
            con_ids: inst.cons.iter().map(|c| c.id).collect(),
        };
        self.env
            .register_data_family_instance(inst.family_name, info);
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
    fn compute_data_con_scheme(&self, data: &DataDef, con: &bhc_hir::ConDef) -> Scheme {
        use bhc_hir::ConFields;

        // For GADT constructors, use the explicit return type
        if let Some(ref gadt_ret_ty) = con.gadt_return_ty {
            let field_types = match &con.fields {
                ConFields::Positional(tys) => tys.clone(),
                ConFields::Named(fields) => fields.iter().map(|f| f.ty.clone()).collect(),
            };

            // Build: field1 -> field2 -> ... -> ReturnType
            let con_ty = field_types
                .into_iter()
                .rev()
                .fold(gadt_ret_ty.clone(), |acc, field_ty| {
                    Ty::fun(field_ty, acc)
                });

            // Quantify over all free type variables in the constructor type
            let free = con_ty.free_vars();
            if free.is_empty() {
                Scheme::mono(con_ty)
            } else {
                Scheme::poly(free, con_ty)
            }
        } else {
            // H98 constructor: standard scheme
            // Build the result type: T a1 a2 ... an
            let result_ty = Self::build_applied_type(data.name, &data.params);

            // Build the function type from field types to result type
            let field_types = match &con.fields {
                ConFields::Positional(tys) => tys.clone(),
                ConFields::Named(fields) => fields.iter().map(|f| f.ty.clone()).collect(),
            };

            // Combine data params with existential vars for kind-fixing
            let all_params: Vec<TyVar> = data
                .params
                .iter()
                .chain(con.existential_vars.iter())
                .cloned()
                .collect();

            // Fix the kinds of type variables in field types to match params.
            let fixed_field_types: Vec<Ty> = field_types
                .into_iter()
                .map(|ty| Self::fix_type_var_kinds(&ty, &all_params))
                .collect();

            let con_ty = fixed_field_types
                .into_iter()
                .rev()
                .fold(result_ty, |acc, field_ty| Ty::fun(field_ty, acc));

            // For existential constructors, the scheme includes both data params
            // and existential vars, plus existential constraints.
            // E.g., `data T = forall a. C a => MkT a` gets scheme:
            //   forall a. C a => a -> T
            if con.existential_vars.is_empty() {
                Scheme::poly(data.params.clone(), con_ty)
            } else {
                let mut vars = data.params.clone();
                vars.extend(con.existential_vars.iter().cloned());
                Scheme {
                    vars,
                    constraints: con.existential_context.clone(),
                    ty: con_ty,
                }
            }
        }
    }

    /// Fix the kinds of type variables in a type to match the given params.
    fn fix_type_var_kinds(ty: &Ty, params: &[TyVar]) -> Ty {
        match ty {
            Ty::Var(v) => {
                // Look for a param with the same ID and use its kind
                if let Some(param) = params.iter().find(|p| p.id == v.id) {
                    Ty::Var(param.clone())
                } else {
                    ty.clone()
                }
            }
            Ty::Con(_) | Ty::Prim(_) | Ty::Error => ty.clone(),
            Ty::App(f, a) => Ty::App(
                Box::new(Self::fix_type_var_kinds(f, params)),
                Box::new(Self::fix_type_var_kinds(a, params)),
            ),
            Ty::Fun(from, to) => Ty::Fun(
                Box::new(Self::fix_type_var_kinds(from, params)),
                Box::new(Self::fix_type_var_kinds(to, params)),
            ),
            Ty::Tuple(tys) => Ty::Tuple(
                tys.iter()
                    .map(|t| Self::fix_type_var_kinds(t, params))
                    .collect(),
            ),
            Ty::List(elem) => Ty::List(Box::new(Self::fix_type_var_kinds(elem, params))),
            Ty::Forall(vars, body) => Ty::Forall(
                vars.clone(),
                Box::new(Self::fix_type_var_kinds(body, params)),
            ),
            Ty::Nat(_) | Ty::TyList(_) => ty.clone(),
        }
    }

    /// Compute the type scheme for a newtype constructor.
    fn compute_newtype_con_scheme(&self, newtype: &NewtypeDef) -> Scheme {
        use bhc_hir::ConFields;

        // Build the result type: T a1 a2 ... an
        let result_ty = Self::build_applied_type(newtype.name, &newtype.params);

        // Newtype has exactly one field
        let field_ty = match &newtype.con.fields {
            ConFields::Positional(tys) => tys.first().cloned().unwrap_or_else(Ty::unit),
            ConFields::Named(fields) => fields.first().map_or_else(Ty::unit, |f| f.ty.clone()),
        };

        // Fix the kinds of type variables in field type to match params
        let fixed_field_ty = Self::fix_type_var_kinds(&field_ty, &newtype.params);

        let con_ty = Ty::fun(fixed_field_ty, result_ty);
        Scheme::poly(newtype.params.clone(), con_ty)
    }

    /// Build an applied type: T a1 a2 ... an
    pub fn build_applied_type(name: bhc_intern::Symbol, params: &[TyVar]) -> Ty {
        let base = Ty::Con(TyCon::new(name, Self::compute_type_con_kind(params.len())));
        params.iter().fold(base, |acc, param| {
            Ty::App(Box::new(acc), Box::new(Ty::Var(param.clone())))
        })
    }

    /// Check a binding group (potentially mutually recursive bindings).
    pub fn check_binding_group(&mut self, group: &BindingGroup) {
        match group {
            BindingGroup::NonRecursive(item) => {
                self.check_item(item);
            }
            BindingGroup::Recursive(items) => {
                // Save constraint count for the whole recursive group
                let constraint_start = self.constraints.len();

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

                // Solve constraints accumulated during this group's inference
                let unsolved = self.solve_constraints_partition(constraint_start);

                // Generalize the types with any unsolved constraints
                for (def_id, _) in temp_schemes {
                    if let Some(scheme) = self.def_schemes.get(&def_id) {
                        let generalized = if unsolved.is_empty() {
                            self.generalize(&scheme.ty)
                        } else {
                            self.generalize_with_constraints(
                                &scheme.ty,
                                unsolved.clone(),
                            )
                        };
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
            Item::Class(class_def) => self.register_class(class_def),
            Item::Instance(instance_def) => self.register_instance(instance_def),
            // These are handled separately:
            // - Data/Newtype: registered in register_data_type/register_newtype
            // - TypeAlias: handled during type resolution
            // - Fixity: no type checking needed
            // - Foreign: uses declared type
            Item::Data(_)
            | Item::Newtype(_)
            | Item::TypeAlias(_)
            | Item::Fixity(_)
            | Item::Foreign(_)
            | Item::StandaloneDeriving(_)
            | Item::PatternSynonym(_)
            | Item::TypeFamily(_)
            | Item::TypeFamilyInst(_)
            | Item::DataFamily(_)
            | Item::DataFamilyInst(_) => {}
        }
    }

    /// Check a value definition.
    fn check_value_def(&mut self, value_def: &ValueDef) {
        // Save constraint count for per-binding scoping
        let constraint_start = self.constraints.len();

        // If ScopedTypeVariables is enabled and the sig has forall-bound vars,
        // instantiate the scheme and register scoped type variables so that
        // body annotations can reference the same type variables.
        let scoped_var_ids: Vec<u32>;
        let declared_ty = if self.scoped_type_variables {
            if let Some(sig) = &value_def.sig {
                if !sig.vars.is_empty() {
                    let (instantiated, subst) =
                        crate::instantiate::instantiate_scoped(self, sig);
                    scoped_var_ids = sig.vars.iter().map(|v| v.id).collect();
                    self.push_scoped_type_vars(&subst);
                    Some(instantiated)
                } else {
                    scoped_var_ids = Vec::new();
                    Some(sig.ty.clone())
                }
            } else {
                scoped_var_ids = Vec::new();
                None
            }
        } else {
            scoped_var_ids = Vec::new();
            value_def.sig.as_ref().map(|s| s.ty.clone())
        };

        // Infer the type from equations
        let inferred_ty = self.infer_equations(&value_def.equations, value_def.span);

        // Pop scoped type variables after inference
        if !scoped_var_ids.is_empty() {
            self.pop_scoped_type_vars(&scoped_var_ids);
        }

        // If there's a declared type, unify with inferred type
        if let Some(declared) = &declared_ty {
            self.unify(declared, &inferred_ty, value_def.span);
        }

        // Solve constraints accumulated during this binding's inference.
        // Unsolved constraints (involving type variables) become part of the scheme.
        let unsolved = self.solve_constraints_partition(constraint_start);

        // Generalize and store the scheme
        let final_ty = self.apply_subst(&inferred_ty);
        let scheme = if let Some(sig) = &value_def.sig {
            // Has explicit type signature — use it (it already has constraints).
            // Mark as explicit so finalization doesn't substitute away the
            // universally quantified type variables.
            self.explicit_sig_defs.insert(value_def.id);
            sig.clone()
        } else if unsolved.is_empty() {
            // No unsolved constraints — simple generalization
            self.generalize(&final_ty)
        } else {
            // Unsolved constraints — generalize with them
            self.generalize_with_constraints(&final_ty, unsolved)
        };

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
        // Also clean up quantified variables that have been resolved
        let explicit_sig_defs = &self.explicit_sig_defs;
        let def_schemes = self
            .def_schemes
            .into_iter()
            .map(|(id, scheme)| {
                // For functions with explicit type signatures, preserve the original
                // scheme as-is. The type variables are universally quantified and
                // should not be resolved by unification with call-site types.
                if explicit_sig_defs.contains(&id) {
                    return (id, scheme);
                }

                let applied_ty = self.subst.apply(&scheme.ty);

                // Apply substitution to constraint args too, so type variable IDs
                // stay consistent between the type and constraints
                let applied_constraints: Vec<_> = scheme
                    .constraints
                    .into_iter()
                    .map(|c| {
                        let applied_args: Vec<Ty> = c
                            .args
                            .iter()
                            .map(|arg| self.subst.apply(arg))
                            .collect();
                        Constraint::new_multi(c.class, applied_args, c.span)
                    })
                    .collect();

                // Collect all free vars from both the type and constraints
                let mut all_free_vars = applied_ty.free_vars();
                for c in &applied_constraints {
                    for arg in &c.args {
                        for v in arg.free_vars() {
                            if !all_free_vars.iter().any(|fv| fv.id == v.id) {
                                all_free_vars.push(v);
                            }
                        }
                    }
                }
                let all_free_var_ids: std::collections::HashSet<u32> =
                    all_free_vars.iter().map(|v| v.id).collect();

                // Rebuild vars from free vars in the applied type and constraints.
                // First try to keep original vars that are still free,
                // then add any new free vars from substitution.
                let mut remaining_vars: Vec<_> = scheme
                    .vars
                    .into_iter()
                    .filter(|v| all_free_var_ids.contains(&v.id))
                    .collect();
                let remaining_var_ids: std::collections::HashSet<u32> =
                    remaining_vars.iter().map(|v| v.id).collect();
                for fv in &all_free_vars {
                    if !remaining_var_ids.contains(&fv.id) {
                        remaining_vars.push(fv.clone());
                    }
                }

                // Filter constraints to only those relevant to remaining variables
                let remaining_constraints: Vec<_> = applied_constraints
                    .into_iter()
                    .filter(|c| {
                        c.args.iter().any(|arg| {
                            arg.free_vars()
                                .iter()
                                .any(|v| all_free_var_ids.contains(&v.id))
                        })
                    })
                    .collect();

                (
                    id,
                    Scheme {
                        vars: remaining_vars,
                        constraints: remaining_constraints,
                        ty: applied_ty,
                    },
                )
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

    /// Instantiate a type scheme for an existential pattern match.
    /// Treats constraints as "given" evidence rather than "wanted".
    pub fn instantiate_as_given(&mut self, scheme: &Scheme) -> Ty {
        crate::instantiate::instantiate_as_given(self, scheme)
    }

    /// Generalize a type (implemented in generalize.rs).
    #[must_use]
    pub fn generalize(&self, ty: &Ty) -> Scheme {
        crate::generalize::generalize(self, ty)
    }

    /// Generalize a type with constraints (implemented in generalize.rs).
    #[must_use]
    pub fn generalize_with_constraints(
        &self,
        ty: &Ty,
        constraints: Vec<Constraint>,
    ) -> Scheme {
        crate::generalize::generalize_with_constraints(self, ty, constraints)
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
