//! Integration tests for M9 Dependent Types Preview.
//!
//! These tests verify the complete implementation of shape-indexed tensors,
//! including type-level naturals, type-level lists, type families, and
//! the bridge to Tensor IR.

use bhc_intern::Symbol;
use bhc_types::dyn_tensor::{
    dyn_tensor_of, dyn_tensor_tycon, is_dyn_tensor, shape_witness_of, shape_witness_tycon,
};
use bhc_types::{nat::TyNat, ty_list::TyList, Kind, Scheme, Ty, TyCon, TyVar};

use bhc_typeck::shape_bridge::{
    extract_static_dims, is_static_shape, shape_from_static, ty_list_to_shape,
};
use bhc_typeck::type_families::{
    reduce_broadcast, reduce_concat, reduce_matmul_shape, reduce_transpose, ReductionResult,
    ShapeError,
};

// ============================================================
// Type-Level Natural Tests
// ============================================================

#[test]
fn test_nat_literal_creation() {
    let n = TyNat::lit(1024);
    assert_eq!(n.as_lit(), Some(1024));
    assert!(n.is_ground());
}

#[test]
fn test_nat_variable() {
    let v = TyVar::new(0, Kind::Nat);
    let n = TyNat::Var(v);
    assert!(!n.is_ground());
    assert_eq!(n.as_lit(), None);
}

#[test]
fn test_nat_arithmetic() {
    let a = TyNat::lit(10);
    let b = TyNat::lit(20);
    let sum = TyNat::add(a.clone(), b.clone());
    let product = TyNat::mul(a, b);

    // When both operands are literals, arithmetic is evaluated
    assert_eq!(sum.as_lit(), Some(30));
    assert_eq!(product.as_lit(), Some(200));

    // But with variables, arithmetic is preserved symbolically
    let v = TyVar::new(1, Kind::Nat);
    let var_sum = TyNat::add(TyNat::Var(v), TyNat::lit(5));
    assert!(matches!(var_sum, TyNat::Add(_, _)));
}

#[test]
fn test_nat_display() {
    let n = TyNat::lit(42);
    assert_eq!(format!("{}", n), "42");

    let v = TyVar::new(5, Kind::Nat);
    let var = TyNat::Var(v);
    assert_eq!(format!("{}", var), "n5");

    // When literals are added, they evaluate to a literal
    let sum = TyNat::add(TyNat::lit(1), TyNat::lit(2));
    assert_eq!(format!("{}", sum), "3"); // Evaluated!

    // With variables, the Add is preserved
    let v2 = TyVar::new(6, Kind::Nat);
    let var_sum = TyNat::add(TyNat::Var(v2), TyNat::lit(1));
    assert_eq!(format!("{}", var_sum), "(n6 + 1)");
}

// ============================================================
// Type-Level List Tests
// ============================================================

#[test]
fn test_ty_list_empty() {
    let nil = TyList::nil();
    assert!(nil.is_nil());
    assert!(nil.is_ground());
    assert_eq!(nil.static_len(), Some(0));
}

#[test]
fn test_ty_list_shape_from_dims() {
    let shape = TyList::shape_from_dims(&[1024, 768, 3]);
    assert!(shape.is_ground());
    assert_eq!(shape.static_len(), Some(3));
    assert_eq!(shape.to_static_dims(), Some(vec![1024, 768, 3]));
}

#[test]
fn test_ty_list_polymorphic() {
    let m = TyVar::new(1, Kind::Nat);
    let n = TyVar::new(2, Kind::Nat);

    let shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(m)), Ty::Nat(TyNat::Var(n))]);

    assert!(!shape.is_ground());
    assert_eq!(shape.static_len(), Some(2)); // We know the rank
    assert_eq!(shape.to_static_dims(), None); // But not the dims
}

#[test]
fn test_ty_list_display() {
    let empty = TyList::nil();
    assert_eq!(format!("{}", empty), "'[]");

    let shape = TyList::shape_from_dims(&[2, 3]);
    assert_eq!(format!("{}", shape), "'[2, 3]");
}

// ============================================================
// Kind Tests
// ============================================================

#[test]
fn test_kind_nat() {
    // Ty::Nat has kind Nat
    let nat_ty = Ty::Nat(TyNat::lit(42));
    assert!(nat_ty.is_nat());
}

#[test]
fn test_kind_ty_list() {
    // Ty::TyList has kind [k] where k is the element kind
    let shape = TyList::shape_from_dims(&[1, 2, 3]);
    let shape_ty = Ty::TyList(shape);
    assert!(shape_ty.is_ty_list());
}

#[test]
fn test_tensor_type_kind() {
    // Tensor has kind [Nat] -> * -> *
    let tensor_kind = Kind::Arrow(
        Box::new(Kind::List(Box::new(Kind::Nat))),
        Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
    );

    // Verify the structure
    match tensor_kind {
        Kind::Arrow(arg, result) => {
            assert!(
                matches!(arg.as_ref(), Kind::List(inner) if matches!(inner.as_ref(), Kind::Nat))
            );
            match result.as_ref() {
                Kind::Arrow(elem, final_kind) => {
                    assert!(elem.is_star());
                    assert!(final_kind.is_star());
                }
                _ => panic!("expected arrow kind"),
            }
        }
        _ => panic!("expected arrow kind"),
    }
}

// ============================================================
// Type Family Tests
// ============================================================

#[test]
fn test_matmul_shape_computation() {
    // [1024, 768] x [768, 512] = [1024, 512]
    let a = TyList::shape_from_dims(&[1024, 768]);
    let b = TyList::shape_from_dims(&[768, 512]);

    match reduce_matmul_shape(&a, &b) {
        ReductionResult::Reduced(result) => {
            assert_eq!(result.to_static_dims(), Some(vec![1024, 512]));
        }
        _ => panic!("expected successful matmul"),
    }
}

#[test]
fn test_matmul_dimension_check() {
    // [1024, 768] x [512, 256] - inner dims don't match
    let a = TyList::shape_from_dims(&[1024, 768]);
    let b = TyList::shape_from_dims(&[512, 256]);

    match reduce_matmul_shape(&a, &b) {
        ReductionResult::Error(ShapeError::MatMulDimensionMismatch {
            left_inner,
            right_inner,
        }) => {
            assert_eq!(left_inner, TyNat::lit(768));
            assert_eq!(right_inner, TyNat::lit(512));
        }
        _ => panic!("expected dimension mismatch error"),
    }
}

#[test]
fn test_broadcast_same_shape() {
    let a = TyList::shape_from_dims(&[2, 3, 4]);
    let b = TyList::shape_from_dims(&[2, 3, 4]);

    match reduce_broadcast(&a, &b) {
        ReductionResult::Reduced(result) => {
            assert_eq!(result.to_static_dims(), Some(vec![2, 3, 4]));
        }
        _ => panic!("expected successful broadcast"),
    }
}

#[test]
fn test_broadcast_with_ones() {
    // [1, 3, 4] broadcast [2, 3, 4] = [2, 3, 4]
    let a = TyList::shape_from_dims(&[1, 3, 4]);
    let b = TyList::shape_from_dims(&[2, 3, 4]);

    match reduce_broadcast(&a, &b) {
        ReductionResult::Reduced(result) => {
            assert_eq!(result.to_static_dims(), Some(vec![2, 3, 4]));
        }
        _ => panic!("expected successful broadcast"),
    }
}

#[test]
fn test_broadcast_extend_rank() {
    // [4] broadcast [2, 3, 4] = [2, 3, 4]
    let a = TyList::shape_from_dims(&[4]);
    let b = TyList::shape_from_dims(&[2, 3, 4]);

    match reduce_broadcast(&a, &b) {
        ReductionResult::Reduced(result) => {
            assert_eq!(result.to_static_dims(), Some(vec![2, 3, 4]));
        }
        _ => panic!("expected successful broadcast"),
    }
}

#[test]
fn test_transpose() {
    let shape = TyList::shape_from_dims(&[3, 4]);

    match reduce_transpose(&shape) {
        ReductionResult::Reduced(result) => {
            assert_eq!(result.to_static_dims(), Some(vec![4, 3]));
        }
        _ => panic!("expected successful transpose"),
    }
}

#[test]
fn test_concat() {
    // [2, 3] concat [2, 5] along axis 1 = [2, 8]
    let a = TyList::shape_from_dims(&[2, 3]);
    let b = TyList::shape_from_dims(&[2, 5]);

    match reduce_concat(&a, &b, 1) {
        ReductionResult::Reduced(result) => {
            assert_eq!(result.to_static_dims(), Some(vec![2, 8]));
        }
        _ => panic!("expected successful concat"),
    }
}

// ============================================================
// Shape Bridge Tests
// ============================================================

#[test]
fn test_shape_bridge_static() {
    use bhc_tensor_ir::Dim;

    let ty_shape = TyList::shape_from_dims(&[1024, 768]);
    let ir_shape = ty_list_to_shape(&ty_shape).unwrap();

    assert_eq!(ir_shape.rank(), 2);
    assert_eq!(ir_shape.dims()[0], Dim::Static(1024));
    assert_eq!(ir_shape.dims()[1], Dim::Static(768));
    assert!(ir_shape.is_static());
}

#[test]
fn test_shape_bridge_dynamic() {
    use bhc_tensor_ir::Dim;

    let m = TyVar::new(1, Kind::Nat);
    let ty_shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(m)), Ty::Nat(TyNat::lit(512))]);

    let ir_shape = ty_list_to_shape(&ty_shape).unwrap();

    assert_eq!(ir_shape.rank(), 2);
    assert!(!ir_shape.dims()[0].is_static());
    assert_eq!(ir_shape.dims()[1], Dim::Static(512));
    assert!(!ir_shape.is_static());
}

#[test]
fn test_shape_bridge_scalar() {
    let ty_shape = TyList::nil();
    let ir_shape = ty_list_to_shape(&ty_shape).unwrap();

    assert!(ir_shape.is_scalar());
    assert!(ir_shape.is_static());
}

#[test]
fn test_extract_static_dims() {
    let shape = TyList::shape_from_dims(&[2, 3, 4]);
    let dims = extract_static_dims(&shape);
    assert_eq!(dims, Some(vec![2, 3, 4]));
}

#[test]
fn test_is_static_shape_true() {
    let shape = TyList::shape_from_dims(&[1024, 768]);
    assert!(is_static_shape(&shape));
}

#[test]
fn test_is_static_shape_false() {
    let n = TyVar::new(1, Kind::Nat);
    let shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(n))]);
    assert!(!is_static_shape(&shape));
}

#[test]
fn test_shape_from_static() {
    use bhc_tensor_ir::Dim;

    let shape = shape_from_static(&[1, 2, 3]);
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.dims()[0], Dim::Static(1));
    assert_eq!(shape.dims()[1], Dim::Static(2));
    assert_eq!(shape.dims()[2], Dim::Static(3));
}

// ============================================================
// Tensor Type Construction Tests
// ============================================================

#[test]
fn test_construct_tensor_type() {
    // Tensor '[1024, 768] Float
    let tensor_con = TyCon::new(
        Symbol::intern("Tensor"),
        Kind::Arrow(
            Box::new(Kind::List(Box::new(Kind::Nat))),
            Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
        ),
    );

    let float_con = TyCon::new(Symbol::intern("Float"), Kind::Star);
    let shape = TyList::shape_from_dims(&[1024, 768]);

    // Build: Tensor '[1024, 768] Float
    let tensor_type = Ty::App(
        Box::new(Ty::App(
            Box::new(Ty::Con(tensor_con.clone())),
            Box::new(Ty::TyList(shape)),
        )),
        Box::new(Ty::Con(float_con)),
    );

    // Verify structure
    match &tensor_type {
        Ty::App(f, elem) => {
            assert!(matches!(elem.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("Float")));
            match f.as_ref() {
                Ty::App(tensor, shape) => {
                    assert!(
                        matches!(tensor.as_ref(), Ty::Con(tc) if tc.name == Symbol::intern("Tensor"))
                    );
                    match shape.as_ref() {
                        Ty::TyList(list) => {
                            assert_eq!(list.to_static_dims(), Some(vec![1024, 768]));
                        }
                        _ => panic!("expected TyList"),
                    }
                }
                _ => panic!("expected Tensor applied to shape"),
            }
        }
        _ => panic!("expected application"),
    }
}

#[test]
fn test_polymorphic_tensor_type() {
    // forall m n. Tensor '[m, n] Float
    let tensor_con = TyCon::new(
        Symbol::intern("Tensor"),
        Kind::Arrow(
            Box::new(Kind::List(Box::new(Kind::Nat))),
            Box::new(Kind::Arrow(Box::new(Kind::Star), Box::new(Kind::Star))),
        ),
    );

    let float_con = TyCon::new(Symbol::intern("Float"), Kind::Star);

    let m = TyVar::new(1, Kind::Nat);
    let n = TyVar::new(2, Kind::Nat);

    let shape = TyList::from_vec(vec![
        Ty::Nat(TyNat::Var(m.clone())),
        Ty::Nat(TyNat::Var(n.clone())),
    ]);

    let tensor_type = Ty::App(
        Box::new(Ty::App(
            Box::new(Ty::Con(tensor_con)),
            Box::new(Ty::TyList(shape)),
        )),
        Box::new(Ty::Con(float_con)),
    );

    // Create a polymorphic scheme
    let scheme = Scheme::poly(vec![m, n], tensor_type);

    // Verify we have 2 type variables
    assert_eq!(scheme.vars.len(), 2);
}

// ============================================================
// End-to-End Shape Checking Scenario Tests
// ============================================================

#[test]
fn test_matmul_type_checking_scenario() {
    // Scenario: matmul :: Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a
    //
    // Given:
    //   a :: Tensor '[1024, 768] Float
    //   b :: Tensor '[768, 512] Float
    //
    // Then matmul a b :: Tensor '[1024, 512] Float

    let a_shape = TyList::shape_from_dims(&[1024, 768]);
    let b_shape = TyList::shape_from_dims(&[768, 512]);

    // Type families compute the result shape
    let result = reduce_matmul_shape(&a_shape, &b_shape);

    match result {
        ReductionResult::Reduced(result_shape) => {
            // Verify result shape
            assert_eq!(result_shape.to_static_dims(), Some(vec![1024, 512]));

            // Bridge to Tensor IR
            let ir_shape = ty_list_to_shape(&result_shape).unwrap();
            assert!(ir_shape.is_static());
            assert_eq!(ir_shape.num_elements(), Some(1024 * 512));
        }
        _ => panic!("matmul shape computation should succeed"),
    }
}

#[test]
fn test_shape_mismatch_detection() {
    // Scenario: Try to matmul incompatible shapes
    //
    // Given:
    //   a :: Tensor '[1024, 768] Float
    //   b :: Tensor '[512, 256] Float  -- 512 != 768!
    //
    // This should be caught at compile time

    let a_shape = TyList::shape_from_dims(&[1024, 768]);
    let b_shape = TyList::shape_from_dims(&[512, 256]);

    let result = reduce_matmul_shape(&a_shape, &b_shape);

    assert!(matches!(
        result,
        ReductionResult::Error(ShapeError::MatMulDimensionMismatch { .. })
    ));
}

#[test]
fn test_polymorphic_shape_preservation() {
    // Scenario: matmul with polymorphic inner dimension
    //
    // matmul :: Tensor '[m, k] a -> Tensor '[k, n] a -> Tensor '[m, n] a
    //
    // When k is the same variable in both, result should be [m, n]

    let m = TyVar::new(1, Kind::Nat);
    let k = TyVar::new(2, Kind::Nat);
    let n = TyVar::new(3, Kind::Nat);

    let a_shape = TyList::from_vec(vec![
        Ty::Nat(TyNat::Var(m.clone())),
        Ty::Nat(TyNat::Var(k.clone())),
    ]);
    let b_shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(k)), Ty::Nat(TyNat::Var(n.clone()))]);

    let result = reduce_matmul_shape(&a_shape, &b_shape);

    match result {
        ReductionResult::Reduced(result_shape) => {
            let dims = result_shape.to_vec().unwrap();
            assert_eq!(dims.len(), 2);

            // First dim should be m
            assert!(matches!(&dims[0], Ty::Nat(TyNat::Var(v)) if v.id == m.id));

            // Second dim should be n
            assert!(matches!(&dims[1], Ty::Nat(TyNat::Var(v)) if v.id == n.id));
        }
        _ => panic!("polymorphic matmul should reduce"),
    }
}

// ============================================================
// Dynamic Tensor (DynTensor) Tests
// ============================================================

#[test]
fn test_dyn_tensor_tycon_kind() {
    let tycon = dyn_tensor_tycon();
    assert_eq!(tycon.name.as_str(), "DynTensor");
    assert_eq!(tycon.kind, Kind::star_to_star());
}

#[test]
fn test_shape_witness_tycon_kind() {
    let tycon = shape_witness_tycon();
    assert_eq!(tycon.name.as_str(), "ShapeWitness");

    // ShapeWitness :: [Nat] -> *
    match &tycon.kind {
        Kind::Arrow(from, to) => {
            assert_eq!(**from, Kind::List(Box::new(Kind::Nat)));
            assert_eq!(**to, Kind::Star);
        }
        _ => panic!("expected arrow kind"),
    }
}

#[test]
fn test_dyn_tensor_of_construction() {
    let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
    let dyn_ty = dyn_tensor_of(float_ty.clone());

    // Verify it's a DynTensor type
    assert!(is_dyn_tensor(&dyn_ty));

    // Verify structure: App(DynTensor, Float)
    match &dyn_ty {
        Ty::App(f, arg) => {
            assert!(matches!(f.as_ref(), Ty::Con(tc) if tc.name.as_str() == "DynTensor"));
            assert_eq!(arg.as_ref(), &float_ty);
        }
        _ => panic!("expected App"),
    }
}

#[test]
fn test_shape_witness_of_construction() {
    let shape = TyList::shape_from_dims(&[1024, 768]);
    let witness_ty = shape_witness_of(shape.clone());

    // Verify structure: App(ShapeWitness, TyList)
    match &witness_ty {
        Ty::App(f, arg) => {
            assert!(matches!(f.as_ref(), Ty::Con(tc) if tc.name.as_str() == "ShapeWitness"));
            match arg.as_ref() {
                Ty::TyList(list) => {
                    assert_eq!(list.to_static_dims(), Some(vec![1024, 768]));
                }
                _ => panic!("expected TyList"),
            }
        }
        _ => panic!("expected App"),
    }
}

#[test]
fn test_is_dyn_tensor_true() {
    let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
    let dyn_ty = dyn_tensor_of(float_ty);
    assert!(is_dyn_tensor(&dyn_ty));
}

#[test]
fn test_is_dyn_tensor_false() {
    let int_ty = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
    assert!(!is_dyn_tensor(&int_ty));

    // Regular tensor is not DynTensor
    let tensor_con = TyCon::new(Symbol::intern("Tensor"), Kind::tensor_kind());
    let shape = TyList::shape_from_dims(&[10, 20]);
    let tensor_ty = Ty::App(
        Box::new(Ty::App(
            Box::new(Ty::Con(tensor_con)),
            Box::new(Ty::TyList(shape)),
        )),
        Box::new(Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star))),
    );
    assert!(!is_dyn_tensor(&tensor_ty));
}

#[test]
fn test_dyn_tensor_gradual_adoption_scenario() {
    // Scenario: User has runtime-shaped data and wants to process it
    //
    // 1. Load data with unknown shape -> DynTensor Float
    // 2. Try to cast to known shape -> Maybe (Tensor '[1024, 768] Float)
    // 3. If successful, use optimized path with shape-indexed tensors
    // 4. If not, use dynamic fallback path

    let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
    let tensor_con = TyCon::new(Symbol::intern("Tensor"), Kind::tensor_kind());
    let maybe_con = TyCon::new(Symbol::intern("Maybe"), Kind::star_to_star());

    // Step 1: DynTensor Float (runtime-shaped data)
    let dyn_tensor_ty = dyn_tensor_of(float_ty.clone());
    assert!(is_dyn_tensor(&dyn_tensor_ty));

    // Step 2: Shape witness for target shape
    let target_shape = TyList::shape_from_dims(&[1024, 768]);
    let witness_ty = shape_witness_of(target_shape.clone());

    // Verify witness has correct shape
    match &witness_ty {
        Ty::App(_, arg) => match arg.as_ref() {
            Ty::TyList(list) => {
                assert_eq!(list.to_static_dims(), Some(vec![1024, 768]));
            }
            _ => panic!("expected TyList"),
        },
        _ => panic!("expected App"),
    }

    // Step 3: Target type after successful cast
    let target_tensor_ty = Ty::App(
        Box::new(Ty::App(
            Box::new(Ty::Con(tensor_con)),
            Box::new(Ty::TyList(target_shape)),
        )),
        Box::new(float_ty),
    );

    // Step 4: Result type is Maybe (Tensor '[1024, 768] Float)
    let result_ty = Ty::App(Box::new(Ty::Con(maybe_con)), Box::new(target_tensor_ty));

    // Verify the result type structure
    match &result_ty {
        Ty::App(f, arg) => {
            assert!(matches!(f.as_ref(), Ty::Con(tc) if tc.name.as_str() == "Maybe"));
            match arg.as_ref() {
                Ty::App(tensor_shape, elem) => {
                    assert!(matches!(elem.as_ref(), Ty::Con(tc) if tc.name.as_str() == "Float"));
                    match tensor_shape.as_ref() {
                        Ty::App(tensor, shape) => {
                            assert!(
                                matches!(tensor.as_ref(), Ty::Con(tc) if tc.name.as_str() == "Tensor")
                            );
                            match shape.as_ref() {
                                Ty::TyList(list) => {
                                    assert_eq!(list.to_static_dims(), Some(vec![1024, 768]));
                                }
                                _ => panic!("expected TyList"),
                            }
                        }
                        _ => panic!("expected Tensor applied to shape"),
                    }
                }
                _ => panic!("expected Tensor type"),
            }
        }
        _ => panic!("expected Maybe application"),
    }
}

#[test]
fn test_dyn_tensor_with_polymorphic_element() {
    // DynTensor a - polymorphic element type
    let a = TyVar::new_star(0);
    let dyn_ty = dyn_tensor_of(Ty::Var(a.clone()));

    assert!(is_dyn_tensor(&dyn_ty));

    // Create scheme: forall a. DynTensor a
    let scheme = Scheme::poly(vec![a], dyn_ty);
    assert_eq!(scheme.vars.len(), 1);
    assert!(is_dyn_tensor(&scheme.ty));
}

// ============================================================
// Tensor IR Bridge Tests (Phase 6)
// ============================================================

use bhc_tensor_ir::DType;
use bhc_typeck::shape_bridge::{
    build_tensor_type, extract_tensor_info, extract_tensor_shape, is_tensor_type, ty_to_dtype,
    verify_shape, verify_tensor_type, ShapeVerifyError,
};

#[test]
fn test_tensor_type_to_ir_shape() {
    // Build a tensor type: Tensor '[1024, 768] Float32
    let shape = TyList::shape_from_dims(&[1024, 768]);
    let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float32"), Kind::Star));
    let tensor_ty = build_tensor_type(shape, float_ty);

    // Extract and convert to IR
    let info = extract_tensor_info(&tensor_ty).unwrap();

    // Verify the IR shape
    assert_eq!(info.shape.rank(), 2);
    assert!(info.shape.is_static());
    assert_eq!(info.shape.num_elements(), Some(1024 * 768));
    assert_eq!(info.dtype, DType::Float32);
}

#[test]
fn test_polymorphic_tensor_type_to_ir_shape() {
    // Build a tensor type with polymorphic dimension: Tensor '[m, 768] Float64
    let m = TyVar::new(1, Kind::Nat);
    let shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(m)), Ty::Nat(TyNat::Lit(768))]);
    let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float64"), Kind::Star));
    let tensor_ty = build_tensor_type(shape, float_ty);

    let info = extract_tensor_info(&tensor_ty).unwrap();

    // IR shape should have dynamic first dimension
    assert_eq!(info.shape.rank(), 2);
    assert!(!info.shape.is_static()); // Not fully static
    assert!(!info.shape.dims()[0].is_static()); // First dim is dynamic
    assert!(info.shape.dims()[1].is_static()); // Second dim is static
    assert_eq!(info.dtype, DType::Float64);
}

#[test]
fn test_shape_verification_during_lowering() {
    // Type-level shape
    let ty_shape = TyList::shape_from_dims(&[1024, 768]);

    // Matching runtime shape
    let ir_shape = bhc_tensor_ir::Shape::from_static([1024, 768]);
    assert!(verify_shape(&ty_shape, &ir_shape).is_ok());

    // Mismatched runtime shape
    let wrong_shape = bhc_tensor_ir::Shape::from_static([1024, 512]);
    let result = verify_shape(&ty_shape, &wrong_shape);
    assert!(matches!(
        result,
        Err(ShapeVerifyError::DimensionMismatch { axis: 1, .. })
    ));
}

#[test]
fn test_matmul_type_to_ir_bridge() {
    // Scenario: matmul produces Tensor '[1024, 512] Float
    // We want to verify this converts to IR correctly

    // Result type from matmul
    let result_shape = TyList::shape_from_dims(&[1024, 512]);
    let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
    let result_ty = build_tensor_type(result_shape, float_ty);

    // Extract info for IR
    let info = extract_tensor_info(&result_ty).unwrap();

    // Create TensorMeta (simulating lowering)
    let meta = bhc_tensor_ir::TensorMeta::new_contiguous(info.dtype, info.shape.clone());
    assert!(meta.is_some());

    let meta = meta.unwrap();
    assert_eq!(meta.shape.rank(), 2);
    assert_eq!(meta.shape.dims()[0].static_value(), Some(1024));
    assert_eq!(meta.shape.dims()[1].static_value(), Some(512));
    assert_eq!(meta.dtype, DType::Float32); // Float defaults to Float32
}

#[test]
fn test_dtype_mapping_comprehensive() {
    // Test all supported dtype mappings
    let test_cases = [
        ("Float32", DType::Float32),
        ("Float64", DType::Float64),
        ("Double", DType::Float64),
        ("Int32", DType::Int32),
        ("Int64", DType::Int64),
        ("Int", DType::Int64),
        ("Int16", DType::Int16),
        ("Int8", DType::Int8),
        ("Word32", DType::UInt32),
        ("Word64", DType::UInt64),
        ("Word", DType::UInt64),
        ("Bool", DType::Bool),
    ];

    for (type_name, expected_dtype) in test_cases {
        let ty = Ty::Con(TyCon::new(Symbol::intern(type_name), Kind::Star));
        let dtype = ty_to_dtype(&ty);
        assert_eq!(dtype, Some(expected_dtype), "Failed for type {}", type_name);
    }
}

#[test]
fn test_end_to_end_tensor_lowering_scenario() {
    // Full scenario: Type checking produces a tensor type,
    // we extract it and verify against expected runtime shape

    // 1. Type system produces: Tensor '[batch, 784] Float32
    let batch = TyVar::new(1, Kind::Nat);
    let shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(batch)), Ty::Nat(TyNat::Lit(784))]);
    let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float32"), Kind::Star));
    let tensor_ty = build_tensor_type(shape, float_ty);

    // 2. Verify it's a tensor type
    assert!(is_tensor_type(&tensor_ty));

    // 3. Extract shape from type
    let ty_shape = extract_tensor_shape(&tensor_ty).unwrap();
    assert_eq!(ty_shape.static_len(), Some(2));

    // 4. Convert to IR shape
    let ir_shape = ty_list_to_shape(ty_shape).unwrap();
    assert_eq!(ir_shape.rank(), 2);

    // 5. Verify against various runtime shapes
    // Any batch size is compatible due to polymorphic batch dimension
    let runtime_32 = bhc_tensor_ir::Shape::from_static([32, 784]);
    let runtime_64 = bhc_tensor_ir::Shape::from_static([64, 784]);
    let runtime_128 = bhc_tensor_ir::Shape::from_static([128, 784]);

    assert!(verify_shape(ty_shape, &runtime_32).is_ok());
    assert!(verify_shape(ty_shape, &runtime_64).is_ok());
    assert!(verify_shape(ty_shape, &runtime_128).is_ok());

    // Wrong second dimension should fail
    let wrong_features = bhc_tensor_ir::Shape::from_static([32, 512]);
    assert!(verify_shape(ty_shape, &wrong_features).is_err());
}

#[test]
fn test_verify_tensor_type_integration() {
    // Build a fully static tensor type: Tensor '[256, 128] Float32
    let shape = TyList::shape_from_dims(&[256, 128]);
    let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float32"), Kind::Star));
    let tensor_ty = build_tensor_type(shape, float_ty);

    // Correct runtime shape should pass
    let correct_shape = bhc_tensor_ir::Shape::from_static([256, 128]);
    assert!(verify_tensor_type(&tensor_ty, &correct_shape).is_ok());

    // Wrong shape should fail
    let wrong_shape = bhc_tensor_ir::Shape::from_static([256, 64]);
    let err = verify_tensor_type(&tensor_ty, &wrong_shape).unwrap_err();
    assert!(matches!(
        err,
        ShapeVerifyError::DimensionMismatch { axis: 1, .. }
    ));

    // Non-tensor type should fail
    let non_tensor = Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star));
    let err = verify_tensor_type(&non_tensor, &correct_shape).unwrap_err();
    assert!(matches!(err, ShapeVerifyError::NotATensorType));
}
