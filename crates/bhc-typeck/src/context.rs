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

    /// Maps constructor DefId to named field definitions (name, type) pairs.
    /// Used for record construction type checking with out-of-order fields.
    pub(crate) con_field_defs: FxHashMap<DefId, Vec<(Symbol, Ty)>>,

    /// Collected type class constraints during inference.
    /// These are solved after inference completes or defaulted if ambiguous.
    pub(crate) constraints: Vec<Constraint>,
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
            con_field_defs: FxHashMap::default(),
            constraints: Vec::new(),
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

    /// Emit a type class constraint.
    ///
    /// Constraints are collected during type inference and solved later.
    /// For example, `emit_constraint("Num", ty, span)` records that `ty`
    /// must have a `Num` instance.
    pub fn emit_constraint(&mut self, class: Symbol, ty: Ty, span: bhc_span::Span) {
        self.constraints.push(Constraint::new(class, ty, span));
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
            // Monad IO is always satisfied
            "Monad" if args.len() == 1 => {
                if let Ty::Con(tycon) = &args[0] {
                    if tycon.name.as_str() == "IO" {
                        return true;
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

        match ty {
            Ty::Con(tycon) => {
                let type_name = tycon.name.as_str();
                matches!(
                    (class_name, type_name),
                    // Num instances
                    ("Num", "Int") | ("Num", "Float") | ("Num", "Double") | ("Num", "Integer") |
                    // Eq instances
                    ("Eq", "Int") | ("Eq", "Float") | ("Eq", "Double") | ("Eq", "Bool") |
                    ("Eq", "Char") | ("Eq", "String") |
                    // Ord instances
                    ("Ord", "Int") | ("Ord", "Float") | ("Ord", "Double") | ("Ord", "Char") |
                    // Show instances
                    ("Show", "Int") | ("Show", "Float") | ("Show", "Double") | ("Show", "Bool") |
                    ("Show", "Char") | ("Show", "String") |
                    // Fractional instances
                    ("Fractional", "Float") | ("Fractional", "Double")
                )
            }
            // List instances: Eq [a], Ord [a], Show [a] if element type has the instance
            Ty::List(elem) => {
                matches!(class_name, "Eq" | "Ord" | "Show") && self.is_builtin_instance(class, elem)
            }
            // Tuple instances: Eq (a, b), Ord (a, b), Show (a, b) if all elements have the instance
            Ty::Tuple(elems) => {
                matches!(class_name, "Eq" | "Ord" | "Show")
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
                        // Maybe a has Eq, Ord, Show if a does
                        ("Eq", "Maybe") | ("Ord", "Maybe") | ("Show", "Maybe") => {
                            self.is_builtin_instance(class, arg)
                        }
                        // IO a has Show (for debugging)
                        ("Show", "IO") => true,
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
            // a -> a -> a (for Num types, we simplify to Int for now)
            Scheme::mono(Ty::fun(
                self.builtins.int_ty.clone(),
                Ty::fun(self.builtins.int_ty.clone(), self.builtins.int_ty.clone()),
            ))
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
            let name = def_info.name.as_str();
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
                // foldl :: (b -> a -> b) -> b -> [a] -> b
                "foldl" | "foldl'" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(
                                Ty::Var(b.clone()),
                                Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                            ),
                            Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, Ty::Var(b.clone()))),
                        ),
                    )
                }
                // foldr :: (a -> b -> b) -> b -> [a] -> b
                "foldr" => {
                    let list_a = Ty::List(Box::new(Ty::Var(a.clone())));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(
                            Ty::fun(
                                Ty::Var(a.clone()),
                                Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())),
                            ),
                            Ty::fun(Ty::Var(b.clone()), Ty::fun(list_a, Ty::Var(b.clone()))),
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
                // negate, abs, signum :: Int -> Int
                "negate" | "abs" | "signum" => Scheme::mono(Ty::fun(
                    self.builtins.int_ty.clone(),
                    self.builtins.int_ty.clone(),
                )),
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
                // enumFromTo :: Int -> Int -> [Int]
                "enumFromTo" => {
                    let list_int = Ty::List(Box::new(self.builtins.int_ty.clone()));
                    Scheme::mono(Ty::fun(
                        self.builtins.int_ty.clone(),
                        Ty::fun(self.builtins.int_ty.clone(), list_int),
                    ))
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
                // min, max :: Int -> Int -> Int (Int-specialized for now)
                "min" | "max" => Scheme::mono(Ty::fun(
                    self.builtins.int_ty.clone(),
                    Ty::fun(self.builtins.int_ty.clone(), self.builtins.int_ty.clone()),
                )),
                // fromIntegral, toInteger :: Int -> Int (identity for now)
                "fromIntegral" | "toInteger" => Scheme::mono(Ty::fun(
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
                // Data.Text.Encoding: Text -> ByteString
                "Data.Text.Encoding.encodeUtf8" => {
                    Scheme::mono(Ty::fun(self.builtins.text_ty.clone(), self.builtins.bytestring_ty.clone()))
                }
                // Data.Text.Encoding: ByteString -> Text
                "Data.Text.Encoding.decodeUtf8" => {
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
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.int_ty.clone(), self.builtins.bool_ty.clone()),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                // Data.ByteString: (a -> Int -> a) -> a -> ByteString -> a
                "Data.ByteString.foldl'" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(
                            Ty::fun(Ty::Var(a.clone()), Ty::fun(self.builtins.int_ty.clone(), Ty::Var(a.clone()))),
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
                    Scheme::mono(Ty::fun(
                        Ty::fun(self.builtins.int_ty.clone(), self.builtins.int_ty.clone()),
                        Ty::fun(self.builtins.bytestring_ty.clone(), self.builtins.bytestring_ty.clone()),
                    ))
                }
                // E.20: Data.Map — opaque polymorphic types
                // Map is treated as opaque (Ty::Var) — all ops use poly(a, b)
                "Data.Map.empty" => {
                    Scheme::poly(vec![a.clone(), b.clone()], Ty::Var(a.clone()))
                }
                "Data.Map.singleton" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.Map.null" | "Data.Map.isSubmapOf" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                    )
                }
                "Data.Map.size" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), self.builtins.int_ty.clone()),
                    )
                }
                "Data.Map.member" | "Data.Map.notMember" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), self.builtins.bool_ty.clone())),
                    )
                }
                "Data.Map.lookup" | "Data.Map.!" | "Data.Map.delete" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))),
                    )
                }
                "Data.Map.findWithDefault" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())))),
                    )
                }
                "Data.Map.insert" | "Data.Map.insertWith" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())))),
                    )
                }
                "Data.Map.adjust" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())))),
                    )
                }
                "Data.Map.update" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(b.clone()), self.builtins.maybe_of(Ty::Var(b.clone()))), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())))),
                    )
                }
                "Data.Map.union" | "Data.Map.intersection" | "Data.Map.difference" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.Map.map" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
                    )
                }
                "Data.Map.filter" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(b.clone()), self.builtins.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.Map.keys" | "Data.Map.elems" | "Data.Map.toList"
                | "Data.Map.toAscList" | "Data.Map.toDescList" | "Data.Map.assocs" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone())),
                    )
                }
                "Data.Map.fromList" | "Data.Map.fromListWith" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    )
                }
                "Data.Map.foldr" | "Data.Map.foldl" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
                    )
                }
                "Data.Map.alter" => {
                    let maybe_b = self.builtins.maybe_of(Ty::Var(b.clone()));
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(maybe_b.clone(), maybe_b), Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone())))),
                    )
                }
                "Data.Map.unionWith" | "Data.Map.intersectionWith"
                | "Data.Map.differenceWith" | "Data.Map.unionWithKey" | "Data.Map.mapWithKey"
                | "Data.Map.mapKeys" | "Data.Map.filterWithKey" | "Data.Map.foldrWithKey"
                | "Data.Map.foldlWithKey" | "Data.Map.unions" | "Data.Map.keysSet" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    )
                }
                // Data.Set operations
                "Data.Set.empty" => {
                    Scheme::poly(vec![a.clone()], Ty::Var(a.clone()))
                }
                "Data.Set.singleton" | "Data.Set.toList" | "Data.Set.toAscList"
                | "Data.Set.toDescList" | "Data.Set.fromList" | "Data.Set.elems"
                | "Data.Set.findMin" | "Data.Set.findMax"
                | "Data.Set.lookupMin" | "Data.Set.lookupMax"
                | "Data.Set.unions" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())),
                    )
                }
                "Data.Set.null" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()),
                    )
                }
                "Data.Set.size" => {
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::Var(a.clone()), self.builtins.int_ty.clone()),
                    )
                }
                "Data.Set.member" | "Data.Set.notMember" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), self.builtins.bool_ty.clone())),
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
                    Scheme::poly(
                        vec![a.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(a.clone()))),
                    )
                }
                "Data.Set.partition" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), self.builtins.bool_ty.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone()))),
                    )
                }
                "Data.Set.foldr" | "Data.Set.foldl" => {
                    Scheme::poly(
                        vec![a.clone(), b.clone()],
                        Ty::fun(Ty::fun(Ty::Var(a.clone()), Ty::fun(Ty::Var(b.clone()), Ty::Var(b.clone()))), Ty::fun(Ty::Var(b.clone()), Ty::fun(Ty::Var(a.clone()), Ty::Var(b.clone())))),
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

                // Unknown builtins - skip here, will be handled in second pass
                _ => continue,
            };

            // Register the builtin with its DefId from the lowering pass
            self.env.insert_global(def_info.id, scheme);
        }

        // Second pass: register any remaining definitions (imported items not in builtins)
        // with fresh type variables. We do this in a separate pass to avoid borrow conflicts
        // with the closures above.
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
            // Register with a fresh polymorphic type
            let fresh = self.fresh_ty();
            let scheme = Scheme::mono(fresh);
            self.env.insert_global(def_info.id, scheme);
        }
    }

    /// Register a data type definition.
    pub fn register_data_type(&mut self, data: &DataDef) {
        // Register the type constructor
        let kind = Self::compute_type_con_kind(data.params.len());
        let tycon = TyCon::new(data.name, kind);
        self.env.register_type_con(tycon);

        // Register data constructors and field accessors
        for con in &data.cons {
            let scheme = self.compute_data_con_scheme(data, con);
            self.env.register_data_con(con.id, con.name, scheme);

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

        // Register class methods as globally available functions.
        // Each method gets its declared type scheme, which includes the class
        // constraint implicitly through the type variables.
        for method in &class.methods {
            // The method's type scheme already has the correct form from lowering.
            // We register it globally so expressions can reference the method.
            self.env
                .insert_global_by_name(method.name, method.ty.clone());
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
            context: vec![], // TODO: Parse instance context from HIR
            methods,
            assoc_type_impls,
        };

        self.env.register_instance(info);

        // Type check the instance method implementations
        for method in &instance.methods {
            self.check_value_def(method);
        }
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

        // Build the result type: T a1 a2 ... an
        let result_ty = Self::build_applied_type(data.name, &data.params);

        // Build the function type from field types to result type
        let field_types = match &con.fields {
            ConFields::Positional(tys) => tys.clone(),
            ConFields::Named(fields) => fields.iter().map(|f| f.ty.clone()).collect(),
        };

        // Fix the kinds of type variables in field types to match params.
        // The field types were lowered without kind information, so type
        // variables have kind Star. We need to update them to have the
        // correct kinds from data.params.
        let fixed_field_types: Vec<Ty> = field_types
            .into_iter()
            .map(|ty| Self::fix_type_var_kinds(&ty, &data.params))
            .collect();

        let con_ty = fixed_field_types
            .into_iter()
            .rev()
            .fold(result_ty, |acc, field_ty| Ty::fun(field_ty, acc));

        Scheme::poly(data.params.clone(), con_ty)
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
        // Also clean up quantified variables that have been resolved
        let def_schemes = self
            .def_schemes
            .into_iter()
            .map(|(id, scheme)| {
                let applied_ty = self.subst.apply(&scheme.ty);
                // Only keep quantified variables that are still free in the type
                let remaining_free_vars = applied_ty.free_vars();
                let remaining_free_var_ids: std::collections::HashSet<u32> =
                    remaining_free_vars.iter().map(|v| v.id).collect();
                let remaining_vars: Vec<_> = scheme
                    .vars
                    .into_iter()
                    .filter(|v| remaining_free_var_ids.contains(&v.id))
                    .collect();
                // Filter constraints to only those relevant to remaining variables
                let remaining_constraints: Vec<_> = scheme
                    .constraints
                    .into_iter()
                    .filter(|c| {
                        c.args.iter().any(|arg| {
                            let arg_ty = self.subst.apply(arg);
                            arg_ty
                                .free_vars()
                                .iter()
                                .any(|v| remaining_free_var_ids.contains(&v.id))
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
