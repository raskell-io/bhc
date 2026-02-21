//! Convert interface types to internal compiler types and back.
//!
//! This module bridges [`crate::Type`] (serializable interface types) with
//! [`bhc_types::Ty`] (the compiler's internal type representation). This is
//! needed when loading `.bhi` interface files for cross-module type checking.

use crate::{
    Constraint as IfaceConstraint, DataConstructor, ExportedType, Kind as IfaceKind,
    Type as IfaceType, TypeSignature as IfaceTypeSig,
};
use bhc_intern::Symbol;
use bhc_span::Span;
use bhc_types::{Constraint, Kind, Scheme, Ty, TyCon, TyVar};

/// State for type variable allocation during conversion.
///
/// Interface files store type variables as strings (e.g., `"a"`, `"b"`).
/// This converter maps them to internal [`TyVar`]s with fresh numeric IDs.
pub struct TypeConverter {
    next_var_id: u32,
    var_map: std::collections::HashMap<String, TyVar>,
}

impl TypeConverter {
    /// Create a new converter with a starting variable ID.
    ///
    /// The `start_var_id` should be high enough to avoid collisions with
    /// variables created by the type checker. We use 90000+ to stay out of
    /// the way of the main type inference engine.
    #[must_use]
    pub fn new(start_var_id: u32) -> Self {
        Self {
            next_var_id: start_var_id,
            var_map: std::collections::HashMap::new(),
        }
    }

    /// Reset the variable map for a fresh conversion context.
    pub fn reset_vars(&mut self) {
        self.var_map.clear();
    }

    /// Get or create a type variable for the given name.
    fn get_or_create_var(&mut self, name: &str) -> TyVar {
        if let Some(var) = self.var_map.get(name) {
            return var.clone();
        }
        let id = self.next_var_id;
        self.next_var_id += 1;
        let var = TyVar::new_star(id);
        self.var_map.insert(name.to_string(), var.clone());
        var
    }

    /// Convert an interface type to an internal type.
    #[must_use]
    pub fn convert_type(&mut self, ty: &IfaceType) -> Ty {
        match ty {
            IfaceType::Var(name) => {
                let var = self.get_or_create_var(name);
                Ty::Var(var)
            }
            IfaceType::Con(name) => {
                let kind = Kind::Star; // Default; callers refine if needed
                Ty::Con(TyCon::new(Symbol::intern(name), kind))
            }
            IfaceType::App(f, x) => {
                let f_ty = self.convert_type(f);
                let x_ty = self.convert_type(x);
                Ty::App(Box::new(f_ty), Box::new(x_ty))
            }
            IfaceType::Fun(a, b) => {
                let a_ty = self.convert_type(a);
                let b_ty = self.convert_type(b);
                Ty::Fun(Box::new(a_ty), Box::new(b_ty))
            }
            IfaceType::Tuple(ts) => {
                let tys: Vec<Ty> = ts.iter().map(|t| self.convert_type(t)).collect();
                Ty::Tuple(tys)
            }
            IfaceType::List(t) => {
                let elem = self.convert_type(t);
                Ty::List(Box::new(elem))
            }
        }
    }

    /// Convert an interface constraint to an internal constraint.
    #[must_use]
    pub fn convert_constraint(&mut self, c: &IfaceConstraint) -> Constraint {
        let args: Vec<Ty> = c.args.iter().map(|t| self.convert_type(t)).collect();
        Constraint::new_multi(Symbol::intern(&c.class), args, Span::default())
    }

    /// Convert an interface type signature to a type scheme.
    #[must_use]
    pub fn convert_type_signature(&mut self, sig: &IfaceTypeSig) -> Scheme {
        self.reset_vars();

        // Pre-create type variables for quantified vars
        let vars: Vec<TyVar> = sig
            .type_vars
            .iter()
            .map(|name| self.get_or_create_var(name))
            .collect();

        let constraints: Vec<Constraint> = sig
            .constraints
            .iter()
            .map(|c| self.convert_constraint(c))
            .collect();

        let ty = self.convert_type(&sig.ty);

        if vars.is_empty() && constraints.is_empty() {
            Scheme::mono(ty)
        } else if constraints.is_empty() {
            Scheme::poly(vars, ty)
        } else {
            Scheme::qualified(vars, constraints, ty)
        }
    }

    /// Convert an interface kind to an internal kind.
    #[must_use]
    pub fn convert_kind(kind: &IfaceKind) -> Kind {
        match kind {
            IfaceKind::Type => Kind::Star,
            IfaceKind::Fun(from, to) => Kind::Arrow(
                Box::new(Self::convert_kind(from)),
                Box::new(Self::convert_kind(to)),
            ),
        }
    }

    /// Build a type constructor kind from parameter count.
    ///
    /// A type with n parameters has kind `* -> * -> ... -> *` (n+1 `*`s).
    #[must_use]
    pub fn kind_for_params(n: usize) -> Kind {
        if n == 0 {
            Kind::Star
        } else {
            Kind::Arrow(Box::new(Kind::Star), Box::new(Self::kind_for_params(n - 1)))
        }
    }

    /// Build a constructor type scheme from an exported type and one of its
    /// data constructors.
    ///
    /// For a type `data Maybe a = Nothing | Just a`:
    /// - `Nothing :: Maybe a`
    /// - `Just :: a -> Maybe a`
    #[must_use]
    pub fn build_constructor_scheme(
        &mut self,
        exported_type: &ExportedType,
        constructor: &DataConstructor,
    ) -> Scheme {
        self.reset_vars();

        // Create type variables for type parameters
        let vars: Vec<TyVar> = exported_type
            .params
            .iter()
            .map(|p| self.get_or_create_var(p))
            .collect();

        // Build the result type: TypeName a b c ...
        let mut result_ty = Ty::Con(TyCon::new(
            Symbol::intern(&exported_type.name),
            Self::kind_for_params(vars.len()),
        ));
        for var in &vars {
            result_ty = Ty::App(Box::new(result_ty), Box::new(Ty::Var(var.clone())));
        }

        // Build the constructor type: field1 -> field2 -> ... -> ResultType
        let mut ty = result_ty;
        for field in constructor.fields.iter().rev() {
            let field_ty = self.convert_type(field);
            ty = Ty::Fun(Box::new(field_ty), Box::new(ty));
        }

        if vars.is_empty() {
            Scheme::mono(ty)
        } else {
            Scheme::poly(vars, ty)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Type;

    #[test]
    fn test_convert_simple_type() {
        let mut conv = TypeConverter::new(90000);
        let iface_ty = Type::Con("Int".to_string());
        let ty = conv.convert_type(&iface_ty);
        assert!(matches!(ty, Ty::Con(tc) if tc.name.as_str() == "Int"));
    }

    #[test]
    fn test_convert_function_type() {
        let mut conv = TypeConverter::new(90000);
        let iface_ty = Type::Fun(
            Box::new(Type::Con("Int".to_string())),
            Box::new(Type::Con("Bool".to_string())),
        );
        let ty = conv.convert_type(&iface_ty);
        assert!(matches!(ty, Ty::Fun(_, _)));
    }

    #[test]
    fn test_convert_type_variable() {
        let mut conv = TypeConverter::new(90000);
        let iface_ty = Type::Var("a".to_string());
        let ty = conv.convert_type(&iface_ty);
        assert!(matches!(ty, Ty::Var(v) if v.id == 90000));

        // Same name should return same variable
        let ty2 = conv.convert_type(&iface_ty);
        assert!(matches!(ty2, Ty::Var(v) if v.id == 90000));
    }

    #[test]
    fn test_convert_type_signature() {
        let mut conv = TypeConverter::new(90000);
        let sig = crate::TypeSignature {
            type_vars: vec!["a".to_string()],
            constraints: vec![],
            ty: Type::Fun(
                Box::new(Type::Var("a".to_string())),
                Box::new(Type::Var("a".to_string())),
            ),
        };
        let scheme = conv.convert_type_signature(&sig);
        assert_eq!(scheme.vars.len(), 1);
        assert!(matches!(scheme.ty, Ty::Fun(_, _)));
    }

    #[test]
    fn test_convert_list_type() {
        let mut conv = TypeConverter::new(90000);
        let iface_ty = Type::List(Box::new(Type::Con("Int".to_string())));
        let ty = conv.convert_type(&iface_ty);
        assert!(matches!(ty, Ty::List(_)));
    }

    #[test]
    fn test_convert_tuple_type() {
        let mut conv = TypeConverter::new(90000);
        let iface_ty = Type::Tuple(vec![
            Type::Con("Int".to_string()),
            Type::Con("Bool".to_string()),
        ]);
        let ty = conv.convert_type(&iface_ty);
        match ty {
            Ty::Tuple(tys) => assert_eq!(tys.len(), 2),
            _ => panic!("expected Tuple"),
        }
    }

    #[test]
    fn test_convert_kind() {
        let kind = TypeConverter::convert_kind(&crate::Kind::Type);
        assert_eq!(kind, Kind::Star);

        let fun_kind =
            TypeConverter::convert_kind(&crate::Kind::Fun(Box::new(crate::Kind::Type), Box::new(crate::Kind::Type)));
        assert!(matches!(fun_kind, Kind::Arrow(_, _)));
    }

    #[test]
    fn test_kind_for_params() {
        let k0 = TypeConverter::kind_for_params(0);
        assert_eq!(k0, Kind::Star);

        let k1 = TypeConverter::kind_for_params(1);
        assert!(matches!(k1, Kind::Arrow(_, _)));
    }
}
