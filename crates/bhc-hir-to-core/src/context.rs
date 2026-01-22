//! Lowering context for HIR to Core transformation.
//!
//! The `LowerContext` tracks state during the lowering process, including:
//! - Fresh variable generation
//! - Error collection
//! - Type environment

use bhc_core::{self as core, Bind, CoreModule, Var, VarId};
use bhc_hir::{DefId, Item, Module as HirModule, ValueDef};
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Scheme, Ty};
use rustc_hash::FxHashMap;

use crate::expr::lower_expr;
use crate::{LowerError, LowerResult, TypeSchemeMap};

/// Context for the HIR to Core lowering pass.
pub struct LowerContext {
    /// Counter for generating fresh variable names.
    fresh_counter: u32,

    /// Mapping from HIR DefIds to Core variables.
    var_map: FxHashMap<DefId, Var>,

    /// Type schemes from the type checker (DefId -> Scheme).
    type_schemes: TypeSchemeMap,

    /// Accumulated errors.
    errors: Vec<LowerError>,
}

impl LowerContext {
    /// Create a new lowering context.
    #[must_use]
    pub fn new() -> Self {
        let mut ctx = Self {
            // Start after builtin VarIds (builtins use 9-95, so start at 100 to be safe)
            fresh_counter: 100,
            var_map: FxHashMap::default(),
            type_schemes: FxHashMap::default(),
            errors: Vec::new(),
        };
        ctx.register_builtins();
        ctx
    }

    /// Set the type schemes from the type checker.
    pub fn set_type_schemes(&mut self, schemes: TypeSchemeMap) {
        self.type_schemes = schemes;
    }

    /// Look up the type for a definition from the type checker.
    ///
    /// Returns the monomorphic type from the scheme, or `Ty::Error` if not found.
    pub fn lookup_type(&self, def_id: DefId) -> Ty {
        self.type_schemes
            .get(&def_id)
            .map(|scheme| scheme.ty.clone())
            .unwrap_or(Ty::Error)
    }

    /// Register builtin operators and constructors.
    ///
    /// DefIds must match the allocation order in bhc-lower and bhc-typeck.
    fn register_builtins(&mut self) {
        // DefIds 0-8: Types (not values, skip)
        // DefIds 9-14: Data constructors (True, False, Nothing, Just, Left, Right)
        let constructors = [
            (9, "True"),
            (10, "False"),
            (11, "Nothing"),
            (12, "Just"),
            (13, "Left"),
            (14, "Right"),
            (15, "[]"),
            (16, ":"),
            (17, "()"),
        ];

        for (id, name) in constructors {
            let def_id = DefId::new(id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(id),
                ty: Ty::Error, // Types resolved during evaluation
            };
            self.var_map.insert(def_id, var);
        }

        // DefIds 18+: Operators and functions
        // Order must match bhc-lower/src/context.rs define_builtins
        let operators = [
            // Arithmetic operators (18-26)
            "+", "-", "*", "/", "div", "mod", "^", "^^", "**",
            // Comparison operators (27-32)
            "==", "/=", "<", "<=", ">", ">=",
            // Boolean operators (33-34)
            "&&", "||",
            // List operators (35-37)
            ":", "++", "!!",
            // Function composition (38-39)
            ".", "$",
            // Monadic operators (40-41)
            ">>=", ">>",
            // Applicative operators (42-45)
            "<*>", "<$>", "*>", "<*",
            // Alternative operator (46)
            "<|>",
            // Monadic operations (47-48)
            "return", "pure",
            // List operations (49-62)
            "map", "filter", "foldr", "foldl", "foldl'", "concatMap",
            "head", "tail", "length", "null", "reverse", "take", "drop", "elem",
            // More list operations (63-70)
            "sum", "product", "and", "or", "any", "all", "maximum", "minimum",
            // Zip operations (71-72)
            "zip", "zipWith",
            // Prelude functions (73-79)
            "id", "const", "flip", "error", "undefined", "seq",
            // Numeric operations (80-88)
            "fromInteger", "fromRational", "negate", "abs", "signum",
            "sqrt", "exp", "log", "sin", "cos", "tan",
            // Comparison (89-90)
            "compare", "min", "max",
            // Show (91)
            "show",
            // Boolean (92-93)
            "not", "otherwise",
        ];

        let mut id = 18; // Start after constructors
        for name in operators {
            let def_id = DefId::new(id);
            let var = Var {
                name: Symbol::intern(name),
                id: VarId::new(id),
                ty: Ty::Error, // Types resolved during evaluation
            };
            self.var_map.insert(def_id, var);
            id += 1;
        }
    }

    /// Register builtins using DefIds from the lowering pass.
    ///
    /// This replaces the hardcoded DefIds with the actual DefIds assigned
    /// during AST-to-HIR lowering, ensuring consistency across passes.
    pub fn register_lowered_builtins(&mut self, defs: &crate::DefMap) {
        // Clear the existing hardcoded builtins
        self.var_map.clear();

        // Register all definitions from the lowering pass
        for (_def_id, def_info) in defs.iter() {
            let var = Var {
                name: def_info.name,
                id: VarId::new(def_info.id.index()),
                ty: Ty::Error, // Types resolved during evaluation
            };
            self.var_map.insert(def_info.id, var);
        }
    }

    /// Generate a fresh variable with the given base name.
    ///
    /// The name will be mangled with a counter to ensure uniqueness.
    /// For top-level bindings that need to preserve their original name,
    /// use `named_var` instead.
    pub fn fresh_var(&mut self, base: &str, ty: Ty, _span: Span) -> Var {
        let name = Symbol::intern(&format!("{}_{}", base, self.fresh_counter));
        self.fresh_counter += 1;
        Var {
            name,
            id: VarId::new(self.fresh_counter as usize),
            ty,
        }
    }

    /// Create a variable with a specific name (preserving the original name).
    ///
    /// Use this for top-level bindings where the name must be preserved
    /// for external visibility (e.g., `main`).
    pub fn named_var(&mut self, name: Symbol, ty: Ty) -> Var {
        self.fresh_counter += 1;
        Var {
            name,
            id: VarId::new(self.fresh_counter as usize),
            ty,
        }
    }

    /// Generate a fresh variable ID.
    pub fn fresh_id(&mut self) -> VarId {
        self.fresh_counter += 1;
        VarId::new(self.fresh_counter as usize)
    }

    /// Record an error.
    pub fn error(&mut self, err: LowerError) {
        self.errors.push(err);
    }

    /// Check if any errors have been recorded.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Take all recorded errors.
    pub fn take_errors(&mut self) -> Vec<LowerError> {
        std::mem::take(&mut self.errors)
    }

    /// Register a HIR definition with a Core variable.
    pub fn register_var(&mut self, def_id: DefId, var: Var) {
        self.var_map.insert(def_id, var);
    }

    /// Look up the Core variable for a HIR definition.
    #[must_use]
    pub fn lookup_var(&self, def_id: DefId) -> Option<&Var> {
        self.var_map.get(&def_id)
    }

    /// Lower a HIR module to Core.
    pub fn lower_module(&mut self, module: &HirModule) -> LowerResult<CoreModule> {
        // First pass: collect all top-level definitions and create Core variables
        // We use named_var here to preserve the original names for external visibility
        for item in &module.items {
            if let Item::Value(value_def) = item {
                // Look up the type from the type checker
                let ty = self.lookup_type(value_def.id);
                let var = self.named_var(value_def.name, ty);
                self.register_var(value_def.id, var);
            }
        }

        // Second pass: lower all items
        let mut bindings = Vec::new();

        for item in &module.items {
            match item {
                Item::Value(value_def) => {
                    if let Some(bind) = self.lower_value_def(value_def)? {
                        bindings.push(bind);
                    }
                }
                Item::Data(data_def) => {
                    // Register data constructors so they can be looked up during lowering
                    for con in &data_def.cons {
                        let var = self.named_var(con.name, Ty::Error);
                        self.register_var(con.id, var);
                    }
                }
                Item::Newtype(newtype_def) => {
                    // Register the newtype constructor
                    let var = self.named_var(newtype_def.con.name, Ty::Error);
                    self.register_var(newtype_def.con.id, var);
                }
                Item::TypeAlias(_) => {
                    // Type aliases don't produce bindings
                }
                Item::Class(_) | Item::Instance(_) => {
                    // Type classes are handled by the type checker;
                    // dictionary passing is inserted there
                }
                Item::Fixity(_) => {
                    // Fixity declarations are only used during parsing
                }
                Item::Foreign(foreign) => {
                    // Foreign imports become special Core bindings
                    // Use named_var to preserve the original name
                    let var = self.named_var(
                        foreign.name,
                        Ty::Error, // Type from signature
                    );
                    self.register_var(foreign.id, var.clone());
                    // For now, we create a placeholder binding
                    // TODO: Proper foreign binding representation
                }
            }
        }

        // Check for errors
        if self.has_errors() {
            return Err(LowerError::Multiple(self.take_errors()));
        }

        Ok(CoreModule {
            name: module.name,
            bindings,
            exports: vec![],
        })
    }

    /// Lower a value definition to a Core binding.
    fn lower_value_def(&mut self, value_def: &ValueDef) -> LowerResult<Option<Bind>> {
        let var = self
            .lookup_var(value_def.id)
            .cloned()
            .ok_or_else(|| LowerError::Internal("missing variable for value def".into()))?;

        // Compile equations to a single expression
        let body = self.compile_equations(value_def)?;

        Ok(Some(Bind::NonRec(var, Box::new(body))))
    }

    /// Compile multiple equations into a single Core expression.
    ///
    /// For simple definitions like `f = e`, this just lowers the expression.
    /// For pattern-matching definitions like:
    /// ```haskell
    /// f 0 = 1
    /// f n = n * f (n-1)
    /// ```
    /// This compiles to a lambda with a case expression.
    fn compile_equations(&mut self, value_def: &ValueDef) -> LowerResult<core::Expr> {
        if value_def.equations.is_empty() {
            return Err(LowerError::Internal("value definition with no equations".into()));
        }

        // Simple case: single equation with no patterns
        if value_def.equations.len() == 1 && value_def.equations[0].pats.is_empty() {
            let eq = &value_def.equations[0];
            return lower_expr(self, &eq.rhs);
        }

        // Complex case: multiple equations or patterns
        // Figure out how many arguments the function takes
        let arity = value_def.equations[0].pats.len();

        if arity == 0 {
            // Multiple equations with no arguments - this is an error
            // but we'll just use the first one
            return lower_expr(self, &value_def.equations[0].rhs);
        }

        // Generate fresh variables for each argument
        let args: Vec<Var> = (0..arity)
            .map(|i| self.fresh_var(&format!("arg{}", i), Ty::Error, value_def.span))
            .collect();

        // Find which argument position(s) need constructor matching
        // If only one position has constructors, we can avoid tuple patterns
        let constructor_positions = self.find_constructor_positions(value_def);

        let case_expr = if arity == 1 || constructor_positions.len() != 1 {
            // Single argument or complex case: use original approach
            let scrutinee = if arity == 1 {
                core::Expr::Var(args[0].clone(), value_def.span)
            } else {
                // Multiple arguments: create a tuple scrutinee
                self.make_tuple_expr(&args, value_def.span)
            };

            // Compile pattern matching
            let case_alts = self.compile_pattern_match(value_def, &args)?;

            core::Expr::Case(
                Box::new(scrutinee),
                case_alts,
                Ty::Error, // Result type placeholder
                value_def.span,
            )
        } else {
            // Optimization: only one argument has constructor patterns
            // Case on just that argument, bind others directly
            let match_pos = constructor_positions[0];
            let match_arg = &args[match_pos];

            self.compile_single_position_match(value_def, &args, match_pos)?
        };

        // Wrap in lambdas
        let mut result = case_expr;
        for arg in args.into_iter().rev() {
            result = core::Expr::Lam(arg, Box::new(result), value_def.span);
        }

        Ok(result)
    }

    /// Make a tuple expression from variables.
    fn make_tuple_expr(&mut self, vars: &[Var], span: Span) -> core::Expr {
        // Build tuple constructor application
        let tuple_con_name = Symbol::intern(&format!("({})", ",".repeat(vars.len() - 1)));

        let mut expr = core::Expr::Var(
            Var {
                name: tuple_con_name,
                id: VarId::new(0),
                ty: Ty::Error,
            },
            span,
        );

        for var in vars {
            expr = core::Expr::App(
                Box::new(expr),
                Box::new(core::Expr::Var(var.clone(), span)),
                span,
            );
        }

        expr
    }

    /// Compile pattern matching for multiple equations.
    fn compile_pattern_match(
        &mut self,
        value_def: &ValueDef,
        args: &[Var],
    ) -> LowerResult<Vec<core::Alt>> {
        use crate::pattern::compile_match;
        compile_match(self, value_def, args)
    }

    /// Find which argument positions have constructor patterns.
    fn find_constructor_positions(&self, value_def: &ValueDef) -> Vec<usize> {
        use bhc_hir::Pat;

        let mut positions = Vec::new();
        let arity = value_def.equations.get(0).map(|eq| eq.pats.len()).unwrap_or(0);

        for pos in 0..arity {
            let has_constructor = value_def.equations.iter().any(|eq| {
                eq.pats.get(pos).map(|pat| matches!(pat, Pat::Con(_, _, _) | Pat::Lit(_, _))).unwrap_or(false)
            });
            if has_constructor {
                positions.push(pos);
            }
        }

        positions
    }

    /// Compile pattern matching when only one argument position has constructors.
    fn compile_single_position_match(
        &mut self,
        value_def: &ValueDef,
        args: &[Var],
        match_pos: usize,
    ) -> LowerResult<core::Expr> {
        use bhc_hir::Pat;
        use crate::pattern::{lower_pat_to_alt, bind_pattern_vars};

        let span = value_def.span;
        let mut alts = Vec::new();

        for eq in &value_def.equations {
            // Register all pattern variables before lowering RHS
            for (i, pat) in eq.pats.iter().enumerate() {
                let arg_var = args.get(i).cloned();
                bind_pattern_vars(self, pat, arg_var.as_ref());
            }

            // Lower the RHS
            let rhs = lower_expr(self, &eq.rhs)?;

            // Get the pattern at the match position
            if let Some(pat) = eq.pats.get(match_pos) {
                let alt = lower_pat_to_alt(self, pat, rhs, span)?;
                alts.push(alt);
            }
        }

        // Add default error case
        alts.push(core::Alt {
            con: core::AltCon::Default,
            binders: vec![],
            rhs: crate::pattern::make_pattern_error(span),
        });

        // Case on the matching argument
        Ok(core::Expr::Case(
            Box::new(core::Expr::Var(args[match_pos].clone(), span)),
            alts,
            Ty::Error,
            span,
        ))
    }
}

impl Default for LowerContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_var() {
        let mut ctx = LowerContext::new();
        let v1 = ctx.fresh_var("x", Ty::Error, Span::default());
        let v2 = ctx.fresh_var("x", Ty::Error, Span::default());
        assert_ne!(v1.id, v2.id);
    }
}
