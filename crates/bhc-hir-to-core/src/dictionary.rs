//! Type class dictionary construction and method selection.
//!
//! This module implements the dictionary-passing transformation for type classes.
//! During HIR-to-Core lowering, type class constraints are translated into
//! explicit dictionary parameters, and method calls become field selections
//! from dictionaries.
//!
//! ## Dictionary Representation
//!
//! A dictionary for a type class is represented as a tuple of method implementations.
//! For example, a class like:
//!
//! ```haskell
//! class Eq a where
//!   (==) :: a -> a -> Bool
//!   (/=) :: a -> a -> Bool
//! ```
//!
//! Has dictionaries represented as 2-tuples of functions.
//!
//! ## Instance Resolution
//!
//! When we encounter a constraint like `Eq Int`, we:
//! 1. Look up the `Eq Int` instance
//! 2. Construct a dictionary containing the method implementations
//! 3. Pass this dictionary where needed
//!
//! ## Superclass Dictionaries
//!
//! If a class has superclasses (e.g., `class Eq a => Ord a`), the dictionary
//! includes the superclass dictionaries as additional fields at the beginning.

use bhc_core::{self as core, Bind, Var, VarId};
use bhc_hir::DefId;
use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Constraint, Kind, Scheme, Ty, TyCon};
use rustc_hash::FxHashMap;

/// Information about a type class for dictionary construction.
#[derive(Clone, Debug)]
pub struct ClassInfo {
    /// The class name.
    pub name: Symbol,
    /// Method names in order (defines dictionary field order).
    pub methods: Vec<Symbol>,
    /// Method type signatures.
    pub method_types: FxHashMap<Symbol, Scheme>,
    /// Superclass names.
    pub superclasses: Vec<Symbol>,
    /// Default method implementations (method name -> DefId).
    pub defaults: FxHashMap<Symbol, DefId>,
}

/// Information about a type class instance.
#[derive(Clone, Debug)]
pub struct InstanceInfo {
    /// The class being instantiated.
    pub class: Symbol,
    /// The instance types (e.g., `[Int]` for `instance Eq Int`,
    /// or `[Int, String]` for `instance Convert Int String`).
    pub instance_types: Vec<Ty>,
    /// Method implementations (method name -> DefId).
    pub methods: FxHashMap<Symbol, DefId>,
    /// Superclass instance types for resolving superclass dictionaries.
    pub superclass_instances: Vec<Ty>,
}

impl InstanceInfo {
    /// Create a new single-parameter instance (most common case).
    pub fn new_single(class: Symbol, instance_type: Ty, methods: FxHashMap<Symbol, DefId>) -> Self {
        Self {
            class,
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
        }
    }

    /// Create a new multi-parameter instance.
    pub fn new_multi(class: Symbol, instance_types: Vec<Ty>, methods: FxHashMap<Symbol, DefId>) -> Self {
        Self {
            class,
            instance_types,
            methods,
            superclass_instances: vec![],
        }
    }

    /// Get the first instance type (for backward compatibility with single-param classes).
    pub fn first_type(&self) -> Option<&Ty> {
        self.instance_types.first()
    }
}

/// Registry of classes and instances for dictionary construction.
#[derive(Clone, Debug, Default)]
pub struct ClassRegistry {
    /// Class information by name.
    pub classes: FxHashMap<Symbol, ClassInfo>,
    /// Instances indexed by (class name, instance type hash).
    /// Since we can't use Ty as a key directly, we use a list and linear search.
    pub instances: FxHashMap<Symbol, Vec<InstanceInfo>>,
}

impl ClassRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a type class.
    pub fn register_class(&mut self, info: ClassInfo) {
        self.classes.insert(info.name, info);
    }

    /// Register a type class instance.
    pub fn register_instance(&mut self, info: InstanceInfo) {
        self.instances
            .entry(info.class)
            .or_insert_with(Vec::new)
            .push(info);
    }

    /// Look up a class by name.
    #[must_use]
    pub fn lookup_class(&self, name: Symbol) -> Option<&ClassInfo> {
        self.classes.get(&name)
    }

    /// Resolve an instance for a class and a single concrete type.
    ///
    /// This is a convenience method for single-parameter type classes.
    /// Returns the instance info if found.
    #[must_use]
    pub fn resolve_instance(&self, class: Symbol, ty: &Ty) -> Option<&InstanceInfo> {
        self.resolve_instance_multi(class, &[ty.clone()])
    }

    /// Resolve an instance for a class with multiple type arguments.
    ///
    /// For example, `resolve_instance_multi("Convert", &[Int, String])` would
    /// find `instance Convert Int String`.
    #[must_use]
    pub fn resolve_instance_multi(&self, class: Symbol, types: &[Ty]) -> Option<&InstanceInfo> {
        let instances = self.instances.get(&class)?;

        for inst in instances {
            if types_match_multi(&inst.instance_types, types) {
                return Some(inst);
            }
        }
        None
    }

    /// Get all method names for a class in order.
    #[must_use]
    pub fn class_methods(&self, class: Symbol) -> Vec<Symbol> {
        self.classes
            .get(&class)
            .map(|c| c.methods.clone())
            .unwrap_or_default()
    }

    /// Get superclasses for a class.
    #[must_use]
    pub fn superclasses(&self, class: Symbol) -> Vec<Symbol> {
        self.classes
            .get(&class)
            .map(|c| c.superclasses.clone())
            .unwrap_or_default()
    }
}

/// Check if two types match for instance resolution.
///
/// This is a simplified matcher that checks structural equality.
/// A full implementation would handle type variables and unification.
fn types_match(pattern: &Ty, target: &Ty) -> bool {
    match (pattern, target) {
        (Ty::Con(c1), Ty::Con(c2)) => c1.name == c2.name,
        (Ty::Var(_), _) => true, // Type variable matches anything
        (Ty::App(f1, a1), Ty::App(f2, a2)) => {
            types_match(f1, f2) && types_match(a1, a2)
        }
        (Ty::Fun(a1, r1), Ty::Fun(a2, r2)) => {
            types_match(a1, a2) && types_match(r1, r2)
        }
        (Ty::Tuple(ts1), Ty::Tuple(ts2)) if ts1.len() == ts2.len() => {
            ts1.iter().zip(ts2.iter()).all(|(t1, t2)| types_match(t1, t2))
        }
        (Ty::List(e1), Ty::List(e2)) => types_match(e1, e2),
        _ => false,
    }
}

/// Check if two type lists match for multi-parameter instance resolution.
///
/// Both lists must have the same length, and each corresponding pair must match.
fn types_match_multi(patterns: &[Ty], targets: &[Ty]) -> bool {
    if patterns.len() != targets.len() {
        return false;
    }
    patterns
        .iter()
        .zip(targets.iter())
        .all(|(p, t)| types_match(p, t))
}

/// Context for dictionary construction during lowering.
pub struct DictContext<'a> {
    /// The class registry.
    pub registry: &'a ClassRegistry,
    /// Fresh variable counter.
    fresh_counter: u32,
    /// Generated dictionary bindings to be added to the module.
    pub dict_bindings: Vec<Bind>,
    /// Cache of already-constructed dictionaries (class, type) -> Var.
    dict_cache: FxHashMap<(Symbol, String), Var>,
}

impl<'a> DictContext<'a> {
    /// Create a new dictionary context.
    pub fn new(registry: &'a ClassRegistry) -> Self {
        Self {
            registry,
            fresh_counter: 1000, // Start high to avoid collisions
            dict_bindings: Vec::new(),
            dict_cache: FxHashMap::default(),
        }
    }

    /// Take ownership of the generated dictionary bindings.
    ///
    /// These bindings define intermediate dictionaries that were constructed
    /// during dictionary resolution. They should be added to the enclosing
    /// let expression or module bindings.
    pub fn take_bindings(&mut self) -> Vec<Bind> {
        std::mem::take(&mut self.dict_bindings)
    }

    /// Generate a fresh variable.
    fn fresh_var(&mut self, prefix: &str, ty: Ty, _span: Span) -> Var {
        let name = Symbol::intern(&format!("{}_{}", prefix, self.fresh_counter));
        self.fresh_counter += 1;
        Var {
            name,
            id: VarId::new(self.fresh_counter as usize),
            ty,
        }
    }

    /// Get or create a dictionary for a constraint with concrete type(s).
    ///
    /// For single-parameter classes like `Num Int`, this returns a variable
    /// bound to the `Num Int` dictionary instance.
    ///
    /// For multi-parameter classes like `Convert Int String`, all type
    /// arguments are used to resolve the instance.
    pub fn get_dictionary(
        &mut self,
        constraint: &Constraint,
        span: Span,
    ) -> Option<core::Expr> {
        // Need at least one type argument
        if constraint.args.is_empty() {
            return None;
        }

        // Check cache first - use all type args for the key
        let cache_key = (constraint.class, format!("{:?}", constraint.args));
        if let Some(var) = self.dict_cache.get(&cache_key) {
            return Some(core::Expr::Var(var.clone(), span));
        }

        // Try to resolve the instance using all type arguments
        let instance = self.registry.resolve_instance_multi(constraint.class, &constraint.args)?;
        let class = self.registry.lookup_class(constraint.class)?;

        // Construct the dictionary
        let dict_expr = self.construct_dictionary(class, instance, span)?;

        // Create a variable for this dictionary and cache it
        let type_names: Vec<String> = constraint.args.iter().map(type_name).collect();
        let dict_var = self.fresh_var(
            &format!("$dict{}_{}", constraint.class.as_str(), type_names.join("_")),
            Ty::Error, // Dictionary type
            span,
        );

        // Add binding for the dictionary
        self.dict_bindings.push(Bind::NonRec(
            dict_var.clone(),
            Box::new(dict_expr),
        ));

        self.dict_cache.insert(cache_key, dict_var.clone());
        Some(core::Expr::Var(dict_var, span))
    }

    /// Construct a dictionary expression for an instance.
    ///
    /// The dictionary is a tuple containing:
    /// 1. Superclass dictionaries (if any)
    /// 2. Method implementations
    fn construct_dictionary(
        &mut self,
        class: &ClassInfo,
        instance: &InstanceInfo,
        span: Span,
    ) -> Option<core::Expr> {
        let mut fields: Vec<core::Expr> = Vec::new();

        // First, add superclass dictionaries
        for (i, superclass) in class.superclasses.iter().enumerate() {
            // Get the superclass instance type
            let super_ty = instance.superclass_instances.get(i)?;

            // Recursively get the superclass dictionary
            let super_constraint = Constraint::new(*superclass, super_ty.clone(), span);
            let super_dict = self.get_dictionary(&super_constraint, span)?;
            fields.push(super_dict);
        }

        // Then add method implementations
        for method_name in &class.methods {
            let method_expr = if let Some(&method_def_id) = instance.methods.get(method_name) {
                // Instance provides this method
                self.method_reference(method_def_id, span)
            } else if let Some(&default_def_id) = class.defaults.get(method_name) {
                // Use default implementation
                self.method_reference(default_def_id, span)
            } else {
                // No implementation - this is an error, but we'll use a placeholder
                let error_var = Var {
                    name: Symbol::intern("$missing_method"),
                    id: VarId::new(0),
                    ty: Ty::Error,
                };
                core::Expr::Var(error_var, span)
            };
            fields.push(method_expr);
        }

        // Construct dictionary as a tuple
        Some(make_tuple(fields, span))
    }

    /// Create a reference to a method implementation.
    fn method_reference(&self, def_id: DefId, span: Span) -> core::Expr {
        // Create a variable reference to the method's DefId
        let var = Var {
            name: Symbol::intern(&format!("$method_{}", def_id.index())),
            id: VarId::new(def_id.index()),
            ty: Ty::Error,
        };
        core::Expr::Var(var, span)
    }
}

/// Extract a method from a dictionary.
///
/// Given a dictionary variable and a method name, generates the
/// expression to select that method from the dictionary.
///
/// # Arguments
///
/// * `dict_var` - The dictionary variable
/// * `class` - The class name (to determine field order)
/// * `method_name` - The method to extract
/// * `registry` - The class registry
/// * `span` - Source span
pub fn select_method(
    dict_var: &Var,
    class: Symbol,
    method_name: Symbol,
    registry: &ClassRegistry,
    span: Span,
) -> Option<core::Expr> {
    let class_info = registry.lookup_class(class)?;

    // Calculate the field index
    // First come superclass dictionaries, then methods
    let superclass_count = class_info.superclasses.len();
    let method_index = class_info.methods.iter()
        .position(|m| *m == method_name)?;

    let field_index = superclass_count + method_index;

    // Generate a selector expression
    // For now, use a simple tuple selector pattern
    Some(make_field_selector(dict_var, field_index, span))
}

/// Extract a superclass dictionary from a dictionary.
///
/// # Arguments
///
/// * `dict_var` - The dictionary variable
/// * `class` - The class name
/// * `superclass` - The superclass to extract
/// * `registry` - The class registry
/// * `span` - Source span
pub fn select_superclass(
    dict_var: &Var,
    class: Symbol,
    superclass: Symbol,
    registry: &ClassRegistry,
    span: Span,
) -> Option<core::Expr> {
    let class_info = registry.lookup_class(class)?;

    // Find the superclass index
    let superclass_index = class_info.superclasses.iter()
        .position(|s| *s == superclass)?;

    // Generate a selector expression
    Some(make_field_selector(dict_var, superclass_index, span))
}

/// Create a tuple expression from a list of fields.
fn make_tuple(fields: Vec<core::Expr>, span: Span) -> core::Expr {
    if fields.is_empty() {
        // Unit
        let unit_var = Var {
            name: Symbol::intern("()"),
            id: VarId::new(0),
            ty: Ty::Error,
        };
        return core::Expr::Var(unit_var, span);
    }

    if fields.len() == 1 {
        // Single element - no tuple needed
        return fields.into_iter().next().unwrap();
    }

    // Build tuple constructor application
    let tuple_name = Symbol::intern(&format!("({})", ",".repeat(fields.len() - 1)));
    let tuple_var = Var {
        name: tuple_name,
        id: VarId::new(0),
        ty: Ty::Error,
    };

    let mut result = core::Expr::Var(tuple_var, span);
    for field in fields {
        result = core::Expr::App(Box::new(result), Box::new(field), span);
    }

    result
}

/// Create a field selector expression for a tuple.
///
/// This generates a case expression that extracts the field at the given index.
fn make_field_selector(dict_var: &Var, field_index: usize, span: Span) -> core::Expr {
    // For now, create a simple selector function application
    // A more complete implementation would generate proper case expressions
    let selector_name = Symbol::intern(&format!("$sel_{}", field_index));
    let selector_var = Var {
        name: selector_name,
        id: VarId::new(field_index),
        ty: Ty::Error,
    };

    core::Expr::App(
        Box::new(core::Expr::Var(selector_var, span)),
        Box::new(core::Expr::Var(dict_var.clone(), span)),
        span,
    )
}

/// Get a simple name for a type (for variable naming).
fn type_name(ty: &Ty) -> String {
    match ty {
        Ty::Con(c) => c.name.as_str().to_string(),
        Ty::Var(v) => format!("t{}", v.id),
        Ty::App(f, _) => type_name(f),
        Ty::Fun(_, _) => "Fun".to_string(),
        Ty::Tuple(_) => "Tuple".to_string(),
        Ty::List(_) => "List".to_string(),
        _ => "Unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_types_match_simple() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        assert!(types_match(&int_ty, &int_ty));
        assert!(!types_match(&int_ty, &bool_ty));
    }

    #[test]
    fn test_class_registry() {
        let mut registry = ClassRegistry::new();

        // Register Eq class
        let eq_class = ClassInfo {
            name: Symbol::intern("Eq"),
            methods: vec![Symbol::intern("=="), Symbol::intern("/=")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
        };
        registry.register_class(eq_class);

        // Register Eq Int instance
        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("=="), DefId::new(100));
        methods.insert(Symbol::intern("/="), DefId::new(101));

        let eq_int = InstanceInfo {
            class: Symbol::intern("Eq"),
            instance_types: vec![Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star))],
            methods,
            superclass_instances: vec![],
        };
        registry.register_instance(eq_int);

        // Test lookup
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let instance = registry.resolve_instance(Symbol::intern("Eq"), &int_ty);
        assert!(instance.is_some());

        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));
        let no_instance = registry.resolve_instance(Symbol::intern("Eq"), &bool_ty);
        assert!(no_instance.is_none());
    }

    #[test]
    fn test_multi_param_type_class() {
        let mut registry = ClassRegistry::new();

        // Register Convert class (multi-parameter: class Convert a b where convert :: a -> b)
        let convert_class = ClassInfo {
            name: Symbol::intern("Convert"),
            methods: vec![Symbol::intern("convert")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
        };
        registry.register_class(convert_class);

        // Register instance Convert Int String
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let string_ty = Ty::Con(TyCon::new(Symbol::intern("String"), Kind::Star));

        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("convert"), DefId::new(200));

        let convert_int_string = InstanceInfo::new_multi(
            Symbol::intern("Convert"),
            vec![int_ty.clone(), string_ty.clone()],
            methods.clone(),
        );
        registry.register_instance(convert_int_string);

        // Register instance Convert Int Bool
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));
        methods.insert(Symbol::intern("convert"), DefId::new(201));

        let convert_int_bool = InstanceInfo::new_multi(
            Symbol::intern("Convert"),
            vec![int_ty.clone(), bool_ty.clone()],
            methods,
        );
        registry.register_instance(convert_int_bool);

        // Test multi-param lookup: Convert Int String should resolve
        let instance = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone(), string_ty.clone()],
        );
        assert!(instance.is_some());
        assert_eq!(instance.unwrap().instance_types.len(), 2);

        // Test multi-param lookup: Convert Int Bool should resolve
        let instance2 = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone(), bool_ty.clone()],
        );
        assert!(instance2.is_some());

        // Test multi-param lookup: Convert String Int should NOT resolve
        let no_instance = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[string_ty.clone(), int_ty.clone()],
        );
        assert!(no_instance.is_none());

        // Test that wrong number of args fails
        let wrong_arity = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone()],  // Only one arg instead of two
        );
        assert!(wrong_arity.is_none());
    }

    #[test]
    fn test_types_match_multi() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let string_ty = Ty::Con(TyCon::new(Symbol::intern("String"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        // Same types match
        assert!(types_match_multi(
            &[int_ty.clone(), string_ty.clone()],
            &[int_ty.clone(), string_ty.clone()]
        ));

        // Different types don't match
        assert!(!types_match_multi(
            &[int_ty.clone(), string_ty.clone()],
            &[int_ty.clone(), bool_ty.clone()]
        ));

        // Different lengths don't match
        assert!(!types_match_multi(
            &[int_ty.clone(), string_ty.clone()],
            &[int_ty.clone()]
        ));

        // Empty matches empty
        assert!(types_match_multi(&[], &[]));
    }

    #[test]
    fn test_instance_info_constructors() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let string_ty = Ty::Con(TyCon::new(Symbol::intern("String"), Kind::Star));
        let methods = FxHashMap::default();

        // Test single constructor
        let single = InstanceInfo::new_single(
            Symbol::intern("Eq"),
            int_ty.clone(),
            methods.clone(),
        );
        assert_eq!(single.instance_types.len(), 1);
        assert!(single.first_type().is_some());

        // Test multi constructor
        let multi = InstanceInfo::new_multi(
            Symbol::intern("Convert"),
            vec![int_ty.clone(), string_ty.clone()],
            methods,
        );
        assert_eq!(multi.instance_types.len(), 2);
        assert!(multi.first_type().is_some());
    }
}
