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
use bhc_types::{Constraint, Kind, Scheme, Subst, Ty, TyVar};
use rustc_hash::FxHashMap;

/// Information about an associated type in a class.
#[derive(Clone, Debug)]
pub struct AssocTypeInfo {
    /// The name of the associated type.
    pub name: Symbol,
    /// Additional type parameters beyond the class parameters.
    pub params: Vec<TyVar>,
    /// The result kind (usually `*`).
    pub kind: Kind,
    /// Optional default type definition.
    pub default: Option<Ty>,
}

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
    /// Associated type declarations.
    pub assoc_types: Vec<AssocTypeInfo>,
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
    /// Associated type implementations (assoc type name -> concrete type).
    pub assoc_type_impls: FxHashMap<Symbol, Ty>,
}

impl InstanceInfo {
    /// Create a new single-parameter instance (most common case).
    pub fn new_single(class: Symbol, instance_type: Ty, methods: FxHashMap<Symbol, DefId>) -> Self {
        Self {
            class,
            instance_types: vec![instance_type],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
        }
    }

    /// Create a new multi-parameter instance.
    pub fn new_multi(
        class: Symbol,
        instance_types: Vec<Ty>,
        methods: FxHashMap<Symbol, DefId>,
    ) -> Self {
        Self {
            class,
            instance_types,
            methods,
            superclass_instances: vec![],
            assoc_type_impls: FxHashMap::default(),
        }
    }

    /// Create a new instance with associated types.
    pub fn new_with_assoc_types(
        class: Symbol,
        instance_types: Vec<Ty>,
        methods: FxHashMap<Symbol, DefId>,
        assoc_type_impls: FxHashMap<Symbol, Ty>,
    ) -> Self {
        Self {
            class,
            instance_types,
            methods,
            superclass_instances: vec![],
            assoc_type_impls,
        }
    }

    /// Get the first instance type (for backward compatibility with single-param classes).
    pub fn first_type(&self) -> Option<&Ty> {
        self.instance_types.first()
    }

    /// Look up an associated type implementation.
    pub fn lookup_assoc_type(&self, name: Symbol) -> Option<&Ty> {
        self.assoc_type_impls.get(&name)
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
    /// Returns the instance info and the substitution binding type variables.
    #[must_use]
    pub fn resolve_instance(&self, class: Symbol, ty: &Ty) -> Option<(&InstanceInfo, Subst)> {
        self.resolve_instance_multi(class, &[ty.clone()])
    }

    /// Resolve an instance for a class with multiple type arguments.
    ///
    /// For example, `resolve_instance_multi("Convert", &[Int, String])` would
    /// find `instance Convert Int String`.
    ///
    /// Returns both the instance and a substitution mapping type variables
    /// in the instance head to concrete types. For example, matching
    /// `instance Eq a => Eq [a]` against `Eq [Int]` returns the substitution
    /// `{a -> Int}`.
    #[must_use]
    pub fn resolve_instance_multi(
        &self,
        class: Symbol,
        types: &[Ty],
    ) -> Option<(&InstanceInfo, Subst)> {
        let instances = self.instances.get(&class)?;

        for inst in instances {
            if let Some(subst) = types_match_multi(&inst.instance_types, types) {
                return Some((inst, subst));
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

    /// Get associated types for a class.
    #[must_use]
    pub fn class_assoc_types(&self, class: Symbol) -> Vec<Symbol> {
        self.classes
            .get(&class)
            .map(|c| c.assoc_types.iter().map(|at| at.name).collect())
            .unwrap_or_default()
    }

    /// Look up which class defines an associated type.
    #[must_use]
    pub fn lookup_assoc_type_class(&self, assoc_name: Symbol) -> Option<Symbol> {
        for (class_name, class_info) in &self.classes {
            for assoc in &class_info.assoc_types {
                if assoc.name == assoc_name {
                    return Some(*class_name);
                }
            }
        }
        None
    }

    /// Get information about an associated type.
    #[must_use]
    pub fn lookup_assoc_type(&self, assoc_name: Symbol) -> Option<&AssocTypeInfo> {
        for class_info in self.classes.values() {
            for assoc in &class_info.assoc_types {
                if assoc.name == assoc_name {
                    return Some(assoc);
                }
            }
        }
        None
    }

    /// Resolve an associated type for a given instance.
    ///
    /// Given an associated type name and concrete type arguments, looks up
    /// the matching instance and returns the concrete type that the associated
    /// type resolves to.
    ///
    /// For example, `resolve_assoc_type("Elem", &[List Int])` would return `Int`
    /// if there's an instance `instance Collection [a] where type Elem [a] = a`.
    #[must_use]
    pub fn resolve_assoc_type(&self, assoc_name: Symbol, types: &[Ty]) -> Option<Ty> {
        // Find which class defines this associated type
        let class_name = self.lookup_assoc_type_class(assoc_name)?;

        // Find the matching instance and get the substitution
        let (instance, subst) = self.resolve_instance_multi(class_name, types)?;

        // Look up the associated type implementation
        if let Some(ty) = instance.assoc_type_impls.get(&assoc_name) {
            // Apply the substitution to get the concrete type
            return Some(subst.apply(ty));
        }

        // Check for default (also apply substitution)
        let assoc_info = self.lookup_assoc_type(assoc_name)?;
        assoc_info.default.as_ref().map(|ty| subst.apply(ty))
    }
}

use bhc_types::types_match_multi;

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
    ///
    /// For polymorphic instances like `instance Eq a => Eq [a]`, matching
    /// against `Eq [Int]` will:
    /// 1. Find the instance with substitution `{a -> Int}`
    /// 2. Apply the substitution to get the superclass constraint `Eq Int`
    /// 3. Recursively construct the `Eq Int` dictionary
    pub fn get_dictionary(&mut self, constraint: &Constraint, span: Span) -> Option<core::Expr> {
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
        // This returns both the instance and a substitution mapping type variables
        // to concrete types (e.g., {a -> Int} when matching Eq [a] against Eq [Int])
        let (instance, subst) = self
            .registry
            .resolve_instance_multi(constraint.class, &constraint.args)?;
        let class = self.registry.lookup_class(constraint.class)?;

        // Construct the dictionary, applying the substitution to resolve
        // type variables in superclass constraints
        let dict_expr = self.construct_dictionary(class, instance, &subst, span)?;

        // Create a variable for this dictionary and cache it
        let type_names: Vec<String> = constraint.args.iter().map(type_name).collect();
        let dict_var = self.fresh_var(
            &format!(
                "$dict{}_{}",
                constraint.class.as_str(),
                type_names.join("_")
            ),
            Ty::Error, // Dictionary type
            span,
        );

        // Add binding for the dictionary
        self.dict_bindings
            .push(Bind::NonRec(dict_var.clone(), Box::new(dict_expr)));

        self.dict_cache.insert(cache_key, dict_var.clone());
        Some(core::Expr::Var(dict_var, span))
    }

    /// Construct a dictionary expression for an instance.
    ///
    /// The dictionary is a tuple containing:
    /// 1. Superclass dictionaries (if any)
    /// 2. Method implementations
    ///
    /// The `subst` parameter maps type variables in the instance to concrete types.
    /// This is crucial for polymorphic instances: when we match `instance Eq a => Eq [a]`
    /// against `Eq [Int]`, the substitution is `{a -> Int}`, and we need to
    /// recursively construct the dictionary for `Eq Int` (not `Eq a`).
    fn construct_dictionary(
        &mut self,
        class: &ClassInfo,
        instance: &InstanceInfo,
        subst: &Subst,
        span: Span,
    ) -> Option<core::Expr> {
        let mut fields: Vec<core::Expr> = Vec::new();

        // First, add superclass dictionaries
        for (i, superclass) in class.superclasses.iter().enumerate() {
            // Get the superclass instance type (may contain type variables)
            let super_ty = instance.superclass_instances.get(i)?;

            // Apply the substitution to get the concrete superclass type
            // e.g., for `instance Eq a => Eq [a]` matching `Eq [Int]`,
            // super_ty is `a` and subst maps `a -> Int`, so we get `Int`
            let concrete_super_ty = subst.apply(super_ty);

            // Recursively get the superclass dictionary for the concrete type
            let super_constraint = Constraint::new(*superclass, concrete_super_ty, span);
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
                // No implementation and no default - generate a runtime error
                // This should normally be caught at type checking, but we generate
                // a proper error expression rather than a placeholder variable
                let error_msg = format!(
                    "No implementation for method '{}' in instance {} {}",
                    method_name.as_str(),
                    instance.class.as_str(),
                    instance
                        .instance_types
                        .iter()
                        .map(|t| format!("{:?}", t))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                make_error_expr(&error_msg, span)
            };
            fields.push(method_expr);
        }

        // Construct dictionary as a tuple
        Some(make_tuple(fields, span))
    }

    /// Create a reference to a method implementation.
    fn method_reference(&self, def_id: DefId, span: Span) -> core::Expr {
        // Look up the actual method name from the registry so codegen can resolve it
        let name = self
            .find_method_name(def_id)
            .unwrap_or_else(|| Symbol::intern(&format!("$method_{}", def_id.index())));
        let var = Var {
            name,
            id: VarId::new(def_id.index()),
            ty: Ty::Error,
        };
        core::Expr::Var(var, span)
    }

    /// Find the method name for a given DefId by searching all registered instances.
    fn find_method_name(&self, def_id: DefId) -> Option<Symbol> {
        for instances in self.registry.instances.values() {
            for instance in instances {
                for (name, &id) in &instance.methods {
                    if id == def_id {
                        return Some(*name);
                    }
                }
            }
        }
        // Also check class defaults
        for class in self.registry.classes.values() {
            for (name, &id) in &class.defaults {
                if id == def_id {
                    return Some(*name);
                }
            }
        }
        None
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
    let method_index = class_info.methods.iter().position(|m| *m == method_name)?;

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
    let superclass_index = class_info
        .superclasses
        .iter()
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

/// Create an error expression that will fail at runtime with a message.
///
/// This generates `error "message"` which will throw an exception at runtime.
/// Used for cases that should have been caught at type checking but weren't.
fn make_error_expr(msg: &str, span: Span) -> core::Expr {
    let error_var = Var {
        name: Symbol::intern("error"),
        id: VarId::new(0),
        ty: Ty::Error,
    };
    let msg_lit = core::Expr::Lit(core::Literal::String(Symbol::intern(msg)), Ty::Error, span);
    core::Expr::App(
        Box::new(core::Expr::Var(error_var, span)),
        Box::new(msg_lit),
        span,
    )
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
    use bhc_types::{types_match, TyCon};

    #[test]
    fn test_types_match_simple() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        // Same types match with empty substitution
        assert!(types_match(&int_ty, &int_ty).is_some());
        // Different types don't match
        assert!(types_match(&int_ty, &bool_ty).is_none());
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
            assoc_types: vec![],
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
            assoc_type_impls: FxHashMap::default(),
        };
        registry.register_instance(eq_int);

        // Test lookup
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let result = registry.resolve_instance(Symbol::intern("Eq"), &int_ty);
        assert!(result.is_some());
        let (instance, subst) = result.unwrap();
        assert_eq!(instance.class, Symbol::intern("Eq"));
        // For concrete type matching, substitution should be empty
        assert!(subst.is_empty());

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
            assoc_types: vec![],
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
        let result = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone(), string_ty.clone()],
        );
        assert!(result.is_some());
        let (instance, _subst) = result.unwrap();
        assert_eq!(instance.instance_types.len(), 2);

        // Test multi-param lookup: Convert Int Bool should resolve
        let result2 = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone(), bool_ty.clone()],
        );
        assert!(result2.is_some());

        // Test multi-param lookup: Convert String Int should NOT resolve
        let no_instance = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[string_ty.clone(), int_ty.clone()],
        );
        assert!(no_instance.is_none());

        // Test that wrong number of args fails
        let wrong_arity = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone()], // Only one arg instead of two
        );
        assert!(wrong_arity.is_none());
    }

    #[test]
    fn test_types_match_multi() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let string_ty = Ty::Con(TyCon::new(Symbol::intern("String"), Kind::Star));
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));

        // Same types match (returns Some with empty substitution)
        let result = types_match_multi(
            &[int_ty.clone(), string_ty.clone()],
            &[int_ty.clone(), string_ty.clone()],
        );
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());

        // Different types don't match
        assert!(types_match_multi(
            &[int_ty.clone(), string_ty.clone()],
            &[int_ty.clone(), bool_ty.clone()]
        )
        .is_none());

        // Different lengths don't match
        assert!(
            types_match_multi(&[int_ty.clone(), string_ty.clone()], &[int_ty.clone()]).is_none()
        );

        // Empty matches empty
        assert!(types_match_multi(&[], &[]).is_some());
    }

    #[test]
    fn test_instance_info_constructors() {
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let string_ty = Ty::Con(TyCon::new(Symbol::intern("String"), Kind::Star));
        let methods = FxHashMap::default();

        // Test single constructor
        let single =
            InstanceInfo::new_single(Symbol::intern("Eq"), int_ty.clone(), methods.clone());
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

    #[test]
    fn test_associated_types() {
        let mut registry = ClassRegistry::new();

        // Create a class with an associated type:
        // class Container c where
        //   type Element c
        //   empty :: c
        //   insert :: Element c -> c -> c
        let container_class = ClassInfo {
            name: Symbol::intern("Container"),
            methods: vec![Symbol::intern("empty"), Symbol::intern("insert")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![AssocTypeInfo {
                name: Symbol::intern("Element"),
                params: vec![],
                kind: Kind::Star,
                default: None,
            }],
        };
        registry.register_class(container_class);

        // Register instance Container [a] where type Element [a] = a
        let a_tyvar = bhc_types::TyVar::new_star(0);
        let list_a = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(Ty::Var(a_tyvar.clone())),
        );

        let a_ty = Ty::Var(a_tyvar);

        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("empty"), DefId::new(500));
        methods.insert(Symbol::intern("insert"), DefId::new(501));

        let mut assoc_impls = FxHashMap::default();
        assoc_impls.insert(Symbol::intern("Element"), a_ty.clone());

        let list_instance = InstanceInfo {
            class: Symbol::intern("Container"),
            instance_types: vec![list_a.clone()],
            methods,
            superclass_instances: vec![],
            assoc_type_impls: assoc_impls,
        };
        registry.register_instance(list_instance);

        // Test associated type lookup
        let assoc_types = registry.class_assoc_types(Symbol::intern("Container"));
        assert_eq!(assoc_types.len(), 1);
        assert_eq!(assoc_types[0], Symbol::intern("Element"));

        // Test lookup_assoc_type_class
        let class_name = registry.lookup_assoc_type_class(Symbol::intern("Element"));
        assert_eq!(class_name, Some(Symbol::intern("Container")));

        // Test lookup_assoc_type
        let assoc_info = registry.lookup_assoc_type(Symbol::intern("Element"));
        assert!(assoc_info.is_some());
        assert_eq!(assoc_info.unwrap().name, Symbol::intern("Element"));

        // Test that non-existent associated types return None
        let no_assoc = registry.lookup_assoc_type(Symbol::intern("NotAnAssocType"));
        assert!(no_assoc.is_none());
    }

    #[test]
    fn test_polymorphic_instance_matching() {
        let mut registry = ClassRegistry::new();

        // Register Eq class
        let eq_class = ClassInfo {
            name: Symbol::intern("Eq"),
            methods: vec![Symbol::intern("==")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        registry.register_class(eq_class);

        // Register a polymorphic instance: instance Eq a => Eq [a]
        // This means: if we have Eq a, we can derive Eq [a]
        let a_tyvar = bhc_types::TyVar::new_star(0);
        let a_ty = Ty::Var(a_tyvar.clone());

        // [a] type = List applied to a
        let list_a = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(a_ty.clone()),
        );

        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("=="), DefId::new(100));

        let eq_list_instance = InstanceInfo {
            class: Symbol::intern("Eq"),
            instance_types: vec![list_a.clone()],
            methods,
            // Superclass instance: the Eq a constraint gets stored here as just `a`
            // When we match Eq [Int], we substitute a -> Int to get Eq Int
            superclass_instances: vec![a_ty.clone()],
            assoc_type_impls: FxHashMap::default(),
        };
        registry.register_instance(eq_list_instance);

        // Now try to resolve Eq [Int]
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let list_int = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(int_ty.clone()),
        );

        let result = registry.resolve_instance(Symbol::intern("Eq"), &list_int);
        assert!(result.is_some(), "Should find Eq [Int] via Eq [a] instance");

        let (instance, subst) = result.unwrap();
        assert_eq!(instance.class, Symbol::intern("Eq"));

        // The substitution should map type variable 'a' (id=0) to Int
        assert!(!subst.is_empty(), "Substitution should not be empty");

        // Apply the substitution to the superclass instance type
        // This should give us Int (the type for the Eq a constraint)
        let superclass_ty = &instance.superclass_instances[0];
        let concrete_superclass_ty = subst.apply(superclass_ty);

        // The concrete superclass type should be Int
        match &concrete_superclass_ty {
            Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
            _ => panic!("Expected Int type, got {:?}", concrete_superclass_ty),
        }
    }

    #[test]
    fn test_polymorphic_multi_param_instance() {
        let mut registry = ClassRegistry::new();

        // Register a multi-param class: class Convert a b where convert :: a -> b
        let convert_class = ClassInfo {
            name: Symbol::intern("Convert"),
            methods: vec![Symbol::intern("convert")],
            method_types: FxHashMap::default(),
            superclasses: vec![],
            defaults: FxHashMap::default(),
            assoc_types: vec![],
        };
        registry.register_class(convert_class);

        // Register a polymorphic instance: instance Convert a [a]
        // This converts any type to a singleton list
        let a_tyvar = bhc_types::TyVar::new_star(0);
        let a_ty = Ty::Var(a_tyvar.clone());

        let list_a = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(a_ty.clone()),
        );

        let mut methods = FxHashMap::default();
        methods.insert(Symbol::intern("convert"), DefId::new(200));

        let convert_a_lista = InstanceInfo::new_multi(
            Symbol::intern("Convert"),
            vec![a_ty.clone(), list_a.clone()],
            methods,
        );
        registry.register_instance(convert_a_lista);

        // Try to resolve Convert Int [Int]
        let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        let list_int = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(int_ty.clone()),
        );

        let result = registry.resolve_instance_multi(
            Symbol::intern("Convert"),
            &[int_ty.clone(), list_int.clone()],
        );
        assert!(
            result.is_some(),
            "Should find Convert Int [Int] via Convert a [a]"
        );

        let (instance, subst) = result.unwrap();
        assert_eq!(instance.class, Symbol::intern("Convert"));
        assert!(!subst.is_empty(), "Substitution should map a -> Int");

        // Apply substitution to 'a' should give Int
        let applied = subst.apply(&a_ty);
        match &applied {
            Ty::Con(c) => assert_eq!(c.name, Symbol::intern("Int")),
            _ => panic!("Expected Int, got {:?}", applied),
        }

        // Try to resolve Convert Int [Bool] - should NOT match
        let bool_ty = Ty::Con(TyCon::new(Symbol::intern("Bool"), Kind::Star));
        let list_bool = Ty::App(
            Box::new(Ty::Con(TyCon::new(Symbol::intern("[]"), Kind::Star))),
            Box::new(bool_ty.clone()),
        );

        let no_match = registry
            .resolve_instance_multi(Symbol::intern("Convert"), &[int_ty.clone(), list_bool]);
        assert!(
            no_match.is_none(),
            "Convert Int [Bool] should NOT match Convert a [a]"
        );
    }
}
