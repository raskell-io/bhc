//! Dynamic tensor types for gradual shape adoption.
//!
//! This module provides types for working with tensors whose shapes are not
//! known at compile time, enabling gradual adoption of shape-indexed types.
//!
//! ## Overview
//!
//! Shape-indexed tensors (`Tensor '[m, n] Float`) provide compile-time dimension
//! checking, but sometimes shapes are only known at runtime. `DynTensor` bridges
//! this gap by providing an existentially-quantified wrapper.
//!
//! ## Usage Pattern
//!
//! ```text
//! -- Convert static to dynamic (always succeeds)
//! toDynamic :: Tensor shape a -> DynTensor a
//!
//! -- Convert dynamic to static (may fail at runtime)
//! fromDynamic :: KnownShape shape => DynTensor a -> Maybe (Tensor shape a)
//!
//! -- Example:
//! processData :: DynTensor Float -> IO ()
//! processData dyn = case fromDynamic @'[1024, 768] dyn of
//!     Just tensor -> optimizedPath tensor  -- statically known shape
//!     Nothing     -> fallbackPath dyn      -- dynamic shape handling
//! ```
//!
//! ## Type Signatures
//!
//! ```text
//! DynTensor :: * -> *
//!
//! toDynamic :: forall shape a. Tensor shape a -> DynTensor a
//!
//! fromDynamic :: forall shape a. KnownShape shape
//!             => DynTensor a -> Maybe (Tensor shape a)
//!
//! withDynShape :: DynTensor a -> (forall shape. Tensor shape a -> r) -> r
//!
//! dynShape :: DynTensor a -> [Int]
//! ```

use crate::{Kind, Ty, TyCon, TyList, TyVar};
use bhc_intern::Symbol;

/// Creates the `DynTensor` type constructor.
///
/// `DynTensor :: * -> *`
///
/// This is an existentially-quantified tensor type that hides the shape:
/// ```text
/// data DynTensor a where
///   MkDynTensor :: Tensor shape a -> DynTensor a
/// ```
#[must_use]
pub fn dyn_tensor_tycon() -> TyCon {
    TyCon::new(Symbol::intern("DynTensor"), Kind::star_to_star())
}

/// Creates a `DynTensor a` type.
///
/// # Example
///
/// ```ignore
/// let dyn_float = dyn_tensor_of(float_ty);
/// // Represents: DynTensor Float
/// ```
#[must_use]
pub fn dyn_tensor_of(elem_ty: Ty) -> Ty {
    Ty::App(Box::new(Ty::Con(dyn_tensor_tycon())), Box::new(elem_ty))
}

/// Creates the `ShapeWitness` type constructor.
///
/// `ShapeWitness :: [Nat] -> *`
///
/// A singleton type that reifies a type-level shape to runtime:
/// ```text
/// data ShapeWitness shape = ShapeWitness
/// ```
#[must_use]
pub fn shape_witness_tycon() -> TyCon {
    // ShapeWitness :: [Nat] -> *
    let kind = Kind::Arrow(Box::new(Kind::nat_list()), Box::new(Kind::Star));
    TyCon::new(Symbol::intern("ShapeWitness"), kind)
}

/// Creates a `ShapeWitness shape` type.
#[must_use]
pub fn shape_witness_of(shape: TyList) -> Ty {
    Ty::App(
        Box::new(Ty::Con(shape_witness_tycon())),
        Box::new(Ty::TyList(shape)),
    )
}

/// Creates the type for `toDynamic`.
///
/// ```text
/// toDynamic :: forall shape a. Tensor shape a -> DynTensor a
/// ```
#[must_use]
pub fn to_dynamic_type(tensor_tycon: &TyCon) -> Ty {
    let shape_var = TyVar::new(0, Kind::nat_list());
    let a_var = TyVar::new(1, Kind::Star);

    // Tensor shape a
    let tensor_type = Ty::App(
        Box::new(Ty::App(
            Box::new(Ty::Con(tensor_tycon.clone())),
            Box::new(Ty::TyList(TyList::Var(shape_var.clone()))),
        )),
        Box::new(Ty::Var(a_var.clone())),
    );

    // DynTensor a
    let dyn_tensor_type = dyn_tensor_of(Ty::Var(a_var.clone()));

    // Tensor shape a -> DynTensor a
    let fun_type = Ty::fun(tensor_type, dyn_tensor_type);

    // forall shape a. Tensor shape a -> DynTensor a
    Ty::Forall(vec![shape_var, a_var], Box::new(fun_type))
}

/// Creates the type for `fromDynamic`.
///
/// ```text
/// fromDynamic :: forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)
/// ```
///
/// Note: In a full implementation, the `ShapeWitness` would be provided by
/// a `KnownShape` type class constraint. Here we make it explicit.
#[must_use]
pub fn from_dynamic_type(tensor_tycon: &TyCon, maybe_tycon: &TyCon) -> Ty {
    let shape_var = TyVar::new(0, Kind::nat_list());
    let a_var = TyVar::new(1, Kind::Star);

    // ShapeWitness shape
    let witness_type = shape_witness_of(TyList::Var(shape_var.clone()));

    // DynTensor a
    let dyn_tensor_type = dyn_tensor_of(Ty::Var(a_var.clone()));

    // Tensor shape a
    let tensor_type = Ty::App(
        Box::new(Ty::App(
            Box::new(Ty::Con(tensor_tycon.clone())),
            Box::new(Ty::TyList(TyList::Var(shape_var.clone()))),
        )),
        Box::new(Ty::Var(a_var.clone())),
    );

    // Maybe (Tensor shape a)
    let maybe_tensor = Ty::App(
        Box::new(Ty::Con(maybe_tycon.clone())),
        Box::new(tensor_type),
    );

    // ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)
    let fun_type = Ty::fun(witness_type, Ty::fun(dyn_tensor_type, maybe_tensor));

    // forall shape a. ...
    Ty::Forall(vec![shape_var, a_var], Box::new(fun_type))
}

/// Creates the type for `withDynShape`.
///
/// ```text
/// withDynShape :: forall a r. DynTensor a -> (forall shape. Tensor shape a -> r) -> r
/// ```
///
/// This is a continuation-passing style function for working with dynamic shapes.
#[must_use]
pub fn with_dyn_shape_type(tensor_tycon: &TyCon) -> Ty {
    let a_var = TyVar::new(0, Kind::Star);
    let r_var = TyVar::new(1, Kind::Star);
    let shape_var = TyVar::new(2, Kind::nat_list());

    // DynTensor a
    let dyn_tensor_type = dyn_tensor_of(Ty::Var(a_var.clone()));

    // Tensor shape a
    let tensor_type = Ty::App(
        Box::new(Ty::App(
            Box::new(Ty::Con(tensor_tycon.clone())),
            Box::new(Ty::TyList(TyList::Var(shape_var.clone()))),
        )),
        Box::new(Ty::Var(a_var.clone())),
    );

    // forall shape. Tensor shape a -> r
    let continuation = Ty::Forall(
        vec![shape_var],
        Box::new(Ty::fun(tensor_type, Ty::Var(r_var.clone()))),
    );

    // DynTensor a -> (forall shape. Tensor shape a -> r) -> r
    let fun_type = Ty::fun(
        dyn_tensor_type,
        Ty::fun(continuation, Ty::Var(r_var.clone())),
    );

    // forall a r. ...
    Ty::Forall(vec![a_var, r_var], Box::new(fun_type))
}

/// Creates the type for `dynShape`.
///
/// ```text
/// dynShape :: forall a. DynTensor a -> [Int]
/// ```
///
/// Returns the runtime shape of a dynamic tensor as a list of integers.
#[must_use]
pub fn dyn_shape_type(int_tycon: &TyCon) -> Ty {
    let a_var = TyVar::new(0, Kind::Star);

    // DynTensor a
    let dyn_tensor_type = dyn_tensor_of(Ty::Var(a_var.clone()));

    // [Int]
    let int_list = Ty::List(Box::new(Ty::Con(int_tycon.clone())));

    // DynTensor a -> [Int]
    let fun_type = Ty::fun(dyn_tensor_type, int_list);

    // forall a. DynTensor a -> [Int]
    Ty::Forall(vec![a_var], Box::new(fun_type))
}

/// Creates the type for `dynRank`.
///
/// ```text
/// dynRank :: forall a. DynTensor a -> Int
/// ```
#[must_use]
pub fn dyn_rank_type(int_tycon: &TyCon) -> Ty {
    let a_var = TyVar::new(0, Kind::Star);

    // DynTensor a
    let dyn_tensor_type = dyn_tensor_of(Ty::Var(a_var.clone()));

    // DynTensor a -> Int
    let fun_type = Ty::fun(dyn_tensor_type, Ty::Con(int_tycon.clone()));

    // forall a. DynTensor a -> Int
    Ty::Forall(vec![a_var], Box::new(fun_type))
}

/// Creates a static shape witness type from concrete dimensions.
///
/// # Example
///
/// ```ignore
/// let witness = static_shape_witness(&[1024, 768]);
/// // Represents: ShapeWitness '[1024, 768]
/// ```
#[must_use]
pub fn static_shape_witness(dims: &[u64]) -> Ty {
    let shape = TyList::shape_from_dims(dims);
    shape_witness_of(shape)
}

/// Checks if a type is a `DynTensor` type.
#[must_use]
pub fn is_dyn_tensor(ty: &Ty) -> bool {
    match ty {
        Ty::App(f, _) => matches!(f.as_ref(), Ty::Con(c) if c.name.as_str() == "DynTensor"),
        _ => false,
    }
}

/// Extracts the element type from a `DynTensor a` type.
#[must_use]
pub fn dyn_tensor_elem_type(ty: &Ty) -> Option<&Ty> {
    match ty {
        Ty::App(f, elem) => {
            if matches!(f.as_ref(), Ty::Con(c) if c.name.as_str() == "DynTensor") {
                Some(elem)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyn_tensor_tycon() {
        let tycon = dyn_tensor_tycon();
        assert_eq!(tycon.name.as_str(), "DynTensor");
        assert_eq!(tycon.kind, Kind::star_to_star());
    }

    #[test]
    fn test_shape_witness_tycon() {
        let tycon = shape_witness_tycon();
        assert_eq!(tycon.name.as_str(), "ShapeWitness");
        match &tycon.kind {
            Kind::Arrow(from, to) => {
                assert_eq!(**from, Kind::nat_list());
                assert_eq!(**to, Kind::Star);
            }
            _ => panic!("expected arrow kind"),
        }
    }

    #[test]
    fn test_dyn_tensor_of() {
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
        let dyn_ty = dyn_tensor_of(float_ty.clone());

        assert!(is_dyn_tensor(&dyn_ty));
        assert_eq!(dyn_tensor_elem_type(&dyn_ty), Some(&float_ty));
    }

    #[test]
    fn test_static_shape_witness() {
        let witness = static_shape_witness(&[1024, 768]);
        match &witness {
            Ty::App(f, arg) => {
                assert!(matches!(f.as_ref(), Ty::Con(c) if c.name.as_str() == "ShapeWitness"));
                match arg.as_ref() {
                    Ty::TyList(list) => {
                        let dims = list.to_static_dims().unwrap();
                        assert_eq!(dims, vec![1024, 768]);
                    }
                    _ => panic!("expected TyList"),
                }
            }
            _ => panic!("expected App"),
        }
    }

    #[test]
    fn test_is_dyn_tensor() {
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
        let dyn_ty = dyn_tensor_of(float_ty);
        assert!(is_dyn_tensor(&dyn_ty));

        let non_dyn = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
        assert!(!is_dyn_tensor(&non_dyn));
    }

    #[test]
    fn test_to_dynamic_type() {
        let tensor_tycon = TyCon::new(Symbol::intern("Tensor"), Kind::tensor_kind());
        let ty = to_dynamic_type(&tensor_tycon);

        // Should be forall shape a. Tensor shape a -> DynTensor a
        match ty {
            Ty::Forall(vars, _body) => {
                assert_eq!(vars.len(), 2);
                assert_eq!(vars[0].kind, Kind::nat_list()); // shape
                assert_eq!(vars[1].kind, Kind::Star); // a
            }
            _ => panic!("expected forall type"),
        }
    }

    #[test]
    fn test_from_dynamic_type() {
        let tensor_tycon = TyCon::new(Symbol::intern("Tensor"), Kind::tensor_kind());
        let maybe_tycon = TyCon::new(Symbol::intern("Maybe"), Kind::star_to_star());
        let ty = from_dynamic_type(&tensor_tycon, &maybe_tycon);

        // Should be forall shape a. ShapeWitness shape -> DynTensor a -> Maybe (Tensor shape a)
        match ty {
            Ty::Forall(vars, _body) => {
                assert_eq!(vars.len(), 2);
            }
            _ => panic!("expected forall type"),
        }
    }
}
