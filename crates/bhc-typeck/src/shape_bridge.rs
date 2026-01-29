//! Bridge between type-level shapes and Tensor IR shapes.
//!
//! This module provides conversion between the type system's shape
//! representation (`TyList` of `TyNat`) and the Tensor IR's runtime
//! shape representation (`Shape` of `Dim`).
//!
//! ## Overview
//!
//! During compilation, tensor shapes flow through several representations:
//!
//! ```text
//! Source:      Tensor '[1024, 768] Float
//!                          |
//!                          v
//! Type System: TyList::Cons(TyNat::Lit(1024), Cons(TyNat::Lit(768), Nil))
//!                          |
//!                          v (this module)
//! Tensor IR:   Shape([Dim::Static(1024), Dim::Static(768)])
//! ```
//!
//! ## Conversion Rules
//!
//! | Type-Level | Tensor IR |
//! |------------|-----------|
//! | `TyNat::Lit(n)` | `Dim::Static(n)` |
//! | `TyNat::Var(v)` | `Dim::Dynamic(v.name)` |
//! | `TyNat::Add(...)` | `Dim::Dynamic(symbolic)` |
//! | `TyNat::Mul(...)` | `Dim::Dynamic(symbolic)` |
//!
//! ## Shape Extraction
//!
//! The module provides functions to extract shape information from tensor types:
//!
//! ```text
//! Ty::App(Ty::App(Tensor, shape), elem) -> Some(TyList)
//! ```
//!
//! ## Shape Verification
//!
//! During lowering, shapes from the type system are verified against runtime shapes:
//!
//! - Static dimensions must match exactly
//! - Dynamic dimensions are compatible with any value
//! - Shape ranks must match

use bhc_intern::Symbol;
use bhc_tensor_ir::{DType, Dim, Shape};
use bhc_types::{nat::TyNat, ty_list::TyList, Kind, Ty, TyCon};

/// Error that can occur during shape conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeBridgeError {
    /// The type-level list could not be converted to a vector.
    UnresolvableShape,
    /// A type in the shape list is not a natural number.
    NonNatDimension,
    /// A dimension value is too large for usize.
    DimensionOverflow(u64),
}

impl std::fmt::Display for ShapeBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeBridgeError::UnresolvableShape => {
                write!(f, "cannot resolve shape to concrete dimensions")
            }
            ShapeBridgeError::NonNatDimension => {
                write!(f, "shape dimension is not a natural number type")
            }
            ShapeBridgeError::DimensionOverflow(n) => {
                write!(f, "dimension {} is too large", n)
            }
        }
    }
}

impl std::error::Error for ShapeBridgeError {}

/// Converts a type-level shape (`TyList`) to a Tensor IR `Shape`.
///
/// This is the primary bridge between the type system's compile-time
/// shape checking and the Tensor IR's runtime shape representation.
///
/// # Arguments
///
/// * `ty_list` - The type-level shape to convert
///
/// # Returns
///
/// * `Ok(Shape)` - The converted Tensor IR shape
/// * `Err(ShapeBridgeError)` - If the shape cannot be converted
///
/// # Example
///
/// ```ignore
/// use bhc_types::TyList;
///
/// let shape = TyList::shape_from_dims(&[1024, 768]);
/// let ir_shape = ty_list_to_shape(&shape)?;
/// assert_eq!(ir_shape.dims(), &[Dim::Static(1024), Dim::Static(768)]);
/// ```
pub fn ty_list_to_shape(ty_list: &TyList) -> Result<Shape, ShapeBridgeError> {
    let tys = ty_list
        .to_vec()
        .ok_or(ShapeBridgeError::UnresolvableShape)?;

    let dims: Result<Vec<Dim>, _> = tys.iter().map(ty_to_dim).collect();
    Ok(Shape::new(dims?))
}

/// Converts a single type-level dimension (`TyNat` wrapped in `Ty`) to a `Dim`.
fn ty_to_dim(ty: &Ty) -> Result<Dim, ShapeBridgeError> {
    match ty {
        Ty::Nat(nat) => ty_nat_to_dim(nat),
        _ => Err(ShapeBridgeError::NonNatDimension),
    }
}

/// Converts a type-level natural number to a Tensor IR dimension.
///
/// # Conversion Rules
///
/// - Literal values become static dimensions
/// - Type variables become dynamic (symbolic) dimensions
/// - Arithmetic expressions become dynamic dimensions with generated names
pub fn ty_nat_to_dim(nat: &TyNat) -> Result<Dim, ShapeBridgeError> {
    match nat {
        TyNat::Lit(n) => {
            // Check for overflow when converting u64 to usize
            let n_usize =
                usize::try_from(*n).map_err(|_| ShapeBridgeError::DimensionOverflow(*n))?;
            Ok(Dim::Static(n_usize))
        }
        TyNat::Var(v) => {
            // Generate a symbolic dimension name from the variable id
            let name = format!("dim{}", v.id);
            Ok(Dim::Dynamic(Symbol::intern(&name)))
        }
        TyNat::Add(left, right) => {
            // For now, treat arithmetic as a dynamic dimension
            // In the future, we could try to evaluate constant expressions
            let name = format!("({} + {})", nat_to_symbolic(left), nat_to_symbolic(right));
            Ok(Dim::Dynamic(Symbol::intern(&name)))
        }
        TyNat::Mul(left, right) => {
            let name = format!("({} * {})", nat_to_symbolic(left), nat_to_symbolic(right));
            Ok(Dim::Dynamic(Symbol::intern(&name)))
        }
    }
}

/// Converts a type-level natural to a symbolic string representation.
fn nat_to_symbolic(nat: &TyNat) -> String {
    match nat {
        TyNat::Lit(n) => n.to_string(),
        TyNat::Var(v) => format!("dim{}", v.id),
        TyNat::Add(left, right) => {
            format!("({} + {})", nat_to_symbolic(left), nat_to_symbolic(right))
        }
        TyNat::Mul(left, right) => {
            format!("({} * {})", nat_to_symbolic(left), nat_to_symbolic(right))
        }
    }
}

/// Attempts to extract static dimensions from a type-level shape.
///
/// Returns `None` if any dimension is not a concrete literal.
/// This is useful for operations that require statically known shapes.
///
/// # Example
///
/// ```ignore
/// let shape = TyList::shape_from_dims(&[1024, 768]);
/// let static_dims = extract_static_dims(&shape);
/// assert_eq!(static_dims, Some(vec![1024, 768]));
/// ```
pub fn extract_static_dims(ty_list: &TyList) -> Option<Vec<usize>> {
    let dims = ty_list.to_static_dims()?;
    dims.into_iter().map(|d| usize::try_from(d).ok()).collect()
}

/// Checks if a type-level shape is fully static (all dimensions known).
pub fn is_static_shape(ty_list: &TyList) -> bool {
    ty_list.is_ground()
        && ty_list.to_vec().map_or(false, |tys| {
            tys.iter().all(|ty| matches!(ty, Ty::Nat(TyNat::Lit(_))))
        })
}

/// Creates a Tensor IR shape from static dimension values.
///
/// This is a convenience function for creating shapes from known dimensions.
#[must_use]
pub fn shape_from_static(dims: &[usize]) -> Shape {
    Shape::from_static(dims.iter().copied())
}

// ============================================================
// Shape Extraction from Types
// ============================================================

/// Extracts the shape component from a tensor type.
///
/// Given a type like `Tensor '[1024, 768] Float`, extracts `'[1024, 768]` as a `TyList`.
///
/// # Arguments
///
/// * `ty` - A tensor type of the form `Tensor shape elem`
///
/// # Returns
///
/// * `Some(TyList)` - The shape if the type is a tensor type
/// * `None` - If the type is not a tensor type
///
/// # Example
///
/// ```ignore
/// let tensor_ty = /* Tensor '[1024, 768] Float */;
/// if let Some(shape) = extract_tensor_shape(&tensor_ty) {
///     assert_eq!(shape.to_static_dims(), Some(vec![1024, 768]));
/// }
/// ```
#[must_use]
pub fn extract_tensor_shape(ty: &Ty) -> Option<&TyList> {
    // Tensor type has structure: App(App(Tensor, shape), elem)
    match ty {
        Ty::App(f, _elem) => match f.as_ref() {
            Ty::App(tensor_con, shape) => {
                // Check if this is actually a Tensor constructor
                if is_tensor_tycon(tensor_con) {
                    // Extract the shape TyList
                    match shape.as_ref() {
                        Ty::TyList(list) => Some(list),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        },
        _ => None,
    }
}

/// Extracts the element type from a tensor type.
///
/// Given a type like `Tensor '[1024, 768] Float`, extracts `Float`.
#[must_use]
pub fn extract_tensor_elem(ty: &Ty) -> Option<&Ty> {
    match ty {
        Ty::App(f, elem) => match f.as_ref() {
            Ty::App(tensor_con, _shape) => {
                if is_tensor_tycon(tensor_con) {
                    Some(elem)
                } else {
                    None
                }
            }
            _ => None,
        },
        _ => None,
    }
}

/// Checks if a type is a tensor type constructor.
fn is_tensor_tycon(ty: &Ty) -> bool {
    match ty {
        Ty::Con(tc) => tc.name.as_str() == "Tensor",
        _ => false,
    }
}

/// Checks if a type is a tensor type.
#[must_use]
pub fn is_tensor_type(ty: &Ty) -> bool {
    extract_tensor_shape(ty).is_some()
}

/// Converts a type-system element type to a Tensor IR dtype.
///
/// Maps standard Haskell/BHC numeric types to their unboxed representations.
#[must_use]
pub fn ty_to_dtype(ty: &Ty) -> Option<DType> {
    match ty {
        Ty::Con(tc) => match tc.name.as_str() {
            "Float" | "Float32" => Some(DType::Float32),
            "Double" | "Float64" => Some(DType::Float64),
            "Int" | "Int64" => Some(DType::Int64),
            "Int32" => Some(DType::Int32),
            "Int16" => Some(DType::Int16),
            "Int8" => Some(DType::Int8),
            "Word" | "Word64" => Some(DType::UInt64),
            "Word32" => Some(DType::UInt32),
            "Word16" => Some(DType::UInt16),
            "Word8" => Some(DType::UInt8),
            "Bool" => Some(DType::Bool),
            _ => None,
        },
        _ => None,
    }
}

// ============================================================
// Shape Verification
// ============================================================

/// Error that can occur during shape verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeVerifyError {
    /// Rank (number of dimensions) mismatch.
    RankMismatch {
        /// Expected rank from type.
        expected: usize,
        /// Actual rank at runtime.
        actual: usize,
    },
    /// Dimension mismatch at a specific axis.
    DimensionMismatch {
        /// The axis with the mismatch.
        axis: usize,
        /// Expected dimension from type.
        expected: usize,
        /// Actual dimension at runtime.
        actual: usize,
    },
    /// Could not extract shape from type.
    NotATensorType,
    /// Shape conversion failed.
    ConversionError(ShapeBridgeError),
}

impl std::fmt::Display for ShapeVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeVerifyError::RankMismatch { expected, actual } => {
                write!(
                    f,
                    "tensor rank mismatch: expected {} dimensions, got {}",
                    expected, actual
                )
            }
            ShapeVerifyError::DimensionMismatch {
                axis,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "dimension mismatch at axis {}: expected {}, got {}",
                    axis, expected, actual
                )
            }
            ShapeVerifyError::NotATensorType => {
                write!(f, "type is not a tensor type")
            }
            ShapeVerifyError::ConversionError(e) => {
                write!(f, "shape conversion error: {}", e)
            }
        }
    }
}

impl std::error::Error for ShapeVerifyError {}

/// Verifies that a runtime shape matches a type-level shape.
///
/// This is used during lowering to verify that the runtime shapes are
/// consistent with the types inferred during type checking.
///
/// # Arguments
///
/// * `ty_shape` - The type-level shape from the type system
/// * `runtime_shape` - The runtime shape from Tensor IR
///
/// # Returns
///
/// * `Ok(())` - Shapes are compatible
/// * `Err(ShapeVerifyError)` - Shapes are incompatible
///
/// # Compatibility Rules
///
/// - Ranks must match exactly
/// - Static dimensions (`TyNat::Lit`) must match exactly
/// - Dynamic dimensions (`TyNat::Var`) are compatible with any value
pub fn verify_shape(ty_shape: &TyList, runtime_shape: &Shape) -> Result<(), ShapeVerifyError> {
    let ty_dims = ty_shape.to_vec().ok_or(ShapeVerifyError::ConversionError(
        ShapeBridgeError::UnresolvableShape,
    ))?;

    let runtime_dims = runtime_shape.dims();

    // Check rank
    if ty_dims.len() != runtime_dims.len() {
        return Err(ShapeVerifyError::RankMismatch {
            expected: ty_dims.len(),
            actual: runtime_dims.len(),
        });
    }

    // Check each dimension
    for (axis, (ty_dim, runtime_dim)) in ty_dims.iter().zip(runtime_dims.iter()).enumerate() {
        verify_dimension(axis, ty_dim, runtime_dim)?;
    }

    Ok(())
}

/// Verifies a single dimension matches.
fn verify_dimension(axis: usize, ty_dim: &Ty, runtime_dim: &Dim) -> Result<(), ShapeVerifyError> {
    match ty_dim {
        Ty::Nat(nat) => match nat {
            TyNat::Lit(expected) => {
                // Static dimension - must match exactly
                let expected_usize = usize::try_from(*expected).map_err(|_| {
                    ShapeVerifyError::ConversionError(ShapeBridgeError::DimensionOverflow(
                        *expected,
                    ))
                })?;

                match runtime_dim {
                    Dim::Static(actual) if *actual == expected_usize => Ok(()),
                    Dim::Static(actual) => Err(ShapeVerifyError::DimensionMismatch {
                        axis,
                        expected: expected_usize,
                        actual: *actual,
                    }),
                    Dim::Dynamic(_) => {
                        // Dynamic runtime dim is compatible (will be checked at runtime)
                        Ok(())
                    }
                }
            }
            TyNat::Var(_) | TyNat::Add(_, _) | TyNat::Mul(_, _) => {
                // Type variable or arithmetic - compatible with any runtime dimension
                Ok(())
            }
        },
        _ => Err(ShapeVerifyError::ConversionError(
            ShapeBridgeError::NonNatDimension,
        )),
    }
}

/// Verifies that a tensor type is compatible with a runtime tensor shape.
///
/// This combines shape extraction and verification.
pub fn verify_tensor_type(ty: &Ty, runtime_shape: &Shape) -> Result<(), ShapeVerifyError> {
    let ty_shape = extract_tensor_shape(ty).ok_or(ShapeVerifyError::NotATensorType)?;
    verify_shape(ty_shape, runtime_shape)
}

// ============================================================
// Lowering Helpers
// ============================================================

/// Information extracted from a tensor type for lowering.
#[derive(Debug, Clone)]
pub struct TensorTypeInfo {
    /// The shape as a Tensor IR shape.
    pub shape: Shape,
    /// The element dtype.
    pub dtype: DType,
    /// Whether the shape is fully static.
    pub is_static: bool,
}

/// Extracts tensor information from a type for lowering to Tensor IR.
///
/// This is the main entry point for extracting all information needed
/// to create a tensor in the IR.
///
/// # Arguments
///
/// * `ty` - A tensor type
///
/// # Returns
///
/// * `Ok(TensorTypeInfo)` - Extracted tensor information
/// * `Err(ShapeBridgeError)` - If extraction fails
pub fn extract_tensor_info(ty: &Ty) -> Result<TensorTypeInfo, ShapeBridgeError> {
    let ty_shape = extract_tensor_shape(ty).ok_or(ShapeBridgeError::NonNatDimension)?;

    let shape = ty_list_to_shape(ty_shape)?;
    let is_static = is_static_shape(ty_shape);

    let elem_ty = extract_tensor_elem(ty).ok_or(ShapeBridgeError::NonNatDimension)?;
    let dtype = ty_to_dtype(elem_ty).unwrap_or(DType::Float64);

    Ok(TensorTypeInfo {
        shape,
        dtype,
        is_static,
    })
}

/// Creates a Tensor type constructor with the correct kind.
///
/// `Tensor :: [Nat] -> * -> *`
#[must_use]
pub fn tensor_tycon() -> TyCon {
    TyCon::new(Symbol::intern("Tensor"), Kind::tensor_kind())
}

/// Builds a tensor type from shape and element type.
///
/// Creates `Tensor shape elem`.
#[must_use]
pub fn build_tensor_type(shape: TyList, elem: Ty) -> Ty {
    Ty::App(
        Box::new(Ty::App(
            Box::new(Ty::Con(tensor_tycon())),
            Box::new(Ty::TyList(shape)),
        )),
        Box::new(elem),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_types::Kind;

    #[test]
    fn test_static_shape_conversion() {
        let ty_shape = TyList::shape_from_dims(&[1024, 768]);
        let ir_shape = ty_list_to_shape(&ty_shape).unwrap();

        assert_eq!(ir_shape.rank(), 2);
        assert_eq!(ir_shape.dims()[0], Dim::Static(1024));
        assert_eq!(ir_shape.dims()[1], Dim::Static(768));
    }

    #[test]
    fn test_scalar_shape() {
        let ty_shape = TyList::nil();
        let ir_shape = ty_list_to_shape(&ty_shape).unwrap();

        assert!(ir_shape.is_scalar());
    }

    #[test]
    fn test_dynamic_dimension() {
        use bhc_types::TyVar;

        let m = TyVar::new(1, Kind::Nat);
        let ty_shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(m.clone()))]);

        let ir_shape = ty_list_to_shape(&ty_shape).unwrap();

        assert_eq!(ir_shape.rank(), 1);
        assert!(!ir_shape.is_static());
        match &ir_shape.dims()[0] {
            Dim::Dynamic(_) => {}
            Dim::Static(_) => panic!("expected dynamic dimension"),
        }
    }

    #[test]
    fn test_mixed_shape() {
        use bhc_types::TyVar;

        let n = TyVar::new(1, Kind::Nat);
        let ty_shape = TyList::from_vec(vec![Ty::Nat(TyNat::Lit(1024)), Ty::Nat(TyNat::Var(n))]);

        let ir_shape = ty_list_to_shape(&ty_shape).unwrap();

        assert_eq!(ir_shape.rank(), 2);
        assert_eq!(ir_shape.dims()[0], Dim::Static(1024));
        assert!(!ir_shape.dims()[1].is_static());
    }

    #[test]
    fn test_extract_static_dims() {
        let ty_shape = TyList::shape_from_dims(&[2, 3, 4]);
        let dims = extract_static_dims(&ty_shape).unwrap();
        assert_eq!(dims, vec![2, 3, 4]);
    }

    #[test]
    fn test_is_static_shape() {
        let static_shape = TyList::shape_from_dims(&[1024, 768]);
        assert!(is_static_shape(&static_shape));

        use bhc_types::TyVar;
        let n = TyVar::new(1, Kind::Nat);
        let dynamic_shape = TyList::from_vec(vec![Ty::Nat(TyNat::Var(n))]);
        assert!(!is_static_shape(&dynamic_shape));
    }

    #[test]
    fn test_arithmetic_dimension() {
        let n = TyNat::Var(bhc_types::TyVar::new(1, Kind::Nat));
        let m = TyNat::Lit(2);
        let product = TyNat::Mul(Box::new(n), Box::new(m));

        let dim = ty_nat_to_dim(&product).unwrap();
        assert!(!dim.is_static());
    }

    #[test]
    fn test_non_nat_dimension_error() {
        // Try to use a non-Nat type as a dimension
        let ty_shape =
            TyList::from_vec(vec![Ty::Con(TyCon::new(Symbol::intern("Int"), Kind::Star))]);

        let result = ty_list_to_shape(&ty_shape);
        assert!(matches!(result, Err(ShapeBridgeError::NonNatDimension)));
    }

    // ============================================================
    // Shape Extraction Tests
    // ============================================================

    #[test]
    fn test_extract_tensor_shape() {
        // Build Tensor '[1024, 768] Float
        let shape = TyList::shape_from_dims(&[1024, 768]);
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
        let tensor_ty = build_tensor_type(shape.clone(), float_ty);

        let extracted = extract_tensor_shape(&tensor_ty);
        assert!(extracted.is_some());
        assert_eq!(extracted.unwrap().to_static_dims(), Some(vec![1024, 768]));
    }

    #[test]
    fn test_extract_tensor_elem() {
        let shape = TyList::shape_from_dims(&[100]);
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
        let tensor_ty = build_tensor_type(shape, float_ty);

        let elem = extract_tensor_elem(&tensor_ty);
        assert!(elem.is_some());
        match elem.unwrap() {
            Ty::Con(tc) => assert_eq!(tc.name.as_str(), "Float"),
            _ => panic!("expected Float type"),
        }
    }

    #[test]
    fn test_is_tensor_type() {
        let shape = TyList::shape_from_dims(&[10, 20]);
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
        let tensor_ty = build_tensor_type(shape, float_ty.clone());

        assert!(is_tensor_type(&tensor_ty));
        assert!(!is_tensor_type(&float_ty));
    }

    #[test]
    fn test_ty_to_dtype() {
        let float32 = Ty::Con(TyCon::new(Symbol::intern("Float32"), Kind::Star));
        assert_eq!(ty_to_dtype(&float32), Some(DType::Float32));

        let float64 = Ty::Con(TyCon::new(Symbol::intern("Double"), Kind::Star));
        assert_eq!(ty_to_dtype(&float64), Some(DType::Float64));

        let int32 = Ty::Con(TyCon::new(Symbol::intern("Int32"), Kind::Star));
        assert_eq!(ty_to_dtype(&int32), Some(DType::Int32));

        let unknown = Ty::Con(TyCon::new(Symbol::intern("Unknown"), Kind::Star));
        assert_eq!(ty_to_dtype(&unknown), None);
    }

    // ============================================================
    // Shape Verification Tests
    // ============================================================

    #[test]
    fn test_verify_shape_matching() {
        let ty_shape = TyList::shape_from_dims(&[1024, 768]);
        let ir_shape = Shape::from_static([1024, 768]);

        let result = verify_shape(&ty_shape, &ir_shape);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_shape_rank_mismatch() {
        let ty_shape = TyList::shape_from_dims(&[1024, 768]);
        let ir_shape = Shape::from_static([1024, 768, 3]); // 3D vs 2D

        let result = verify_shape(&ty_shape, &ir_shape);
        assert!(matches!(
            result,
            Err(ShapeVerifyError::RankMismatch {
                expected: 2,
                actual: 3
            })
        ));
    }

    #[test]
    fn test_verify_shape_dimension_mismatch() {
        let ty_shape = TyList::shape_from_dims(&[1024, 768]);
        let ir_shape = Shape::from_static([1024, 512]); // 768 != 512

        let result = verify_shape(&ty_shape, &ir_shape);
        assert!(matches!(
            result,
            Err(ShapeVerifyError::DimensionMismatch {
                axis: 1,
                expected: 768,
                actual: 512
            })
        ));
    }

    #[test]
    fn test_verify_shape_dynamic_compatible() {
        use bhc_types::TyVar;

        // Type-level shape with variable dimension
        let n = TyVar::new(1, Kind::Nat);
        let ty_shape = TyList::from_vec(vec![Ty::Nat(TyNat::Lit(1024)), Ty::Nat(TyNat::Var(n))]);

        // Runtime shape with any second dimension should be compatible
        let ir_shape = Shape::from_static([1024, 999]);

        let result = verify_shape(&ty_shape, &ir_shape);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_tensor_type() {
        let shape = TyList::shape_from_dims(&[100, 200]);
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
        let tensor_ty = build_tensor_type(shape, float_ty);

        let ir_shape = Shape::from_static([100, 200]);
        let result = verify_tensor_type(&tensor_ty, &ir_shape);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_tensor_type_not_tensor() {
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float"), Kind::Star));
        let ir_shape = Shape::from_static([100]);

        let result = verify_tensor_type(&float_ty, &ir_shape);
        assert!(matches!(result, Err(ShapeVerifyError::NotATensorType)));
    }

    // ============================================================
    // Lowering Helper Tests
    // ============================================================

    #[test]
    fn test_extract_tensor_info() {
        let shape = TyList::shape_from_dims(&[1024, 768]);
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float32"), Kind::Star));
        let tensor_ty = build_tensor_type(shape, float_ty);

        let info = extract_tensor_info(&tensor_ty).unwrap();

        assert_eq!(info.shape.rank(), 2);
        assert_eq!(info.shape.dims()[0], Dim::Static(1024));
        assert_eq!(info.shape.dims()[1], Dim::Static(768));
        assert_eq!(info.dtype, DType::Float32);
        assert!(info.is_static);
    }

    #[test]
    fn test_extract_tensor_info_dynamic() {
        use bhc_types::TyVar;

        let n = TyVar::new(1, Kind::Nat);
        let shape = TyList::from_vec(vec![Ty::Nat(TyNat::Lit(1024)), Ty::Nat(TyNat::Var(n))]);
        let float_ty = Ty::Con(TyCon::new(Symbol::intern("Float64"), Kind::Star));
        let tensor_ty = build_tensor_type(shape, float_ty);

        let info = extract_tensor_info(&tensor_ty).unwrap();

        assert_eq!(info.shape.rank(), 2);
        assert_eq!(info.dtype, DType::Float64);
        assert!(!info.is_static);
    }

    #[test]
    fn test_build_tensor_type_roundtrip() {
        let shape = TyList::shape_from_dims(&[32, 64, 128]);
        let elem = Ty::Con(TyCon::new(Symbol::intern("Int32"), Kind::Star));
        let tensor_ty = build_tensor_type(shape.clone(), elem);

        // Extract back
        let extracted_shape = extract_tensor_shape(&tensor_ty).unwrap();
        assert_eq!(extracted_shape.to_static_dims(), shape.to_static_dims());

        let extracted_elem = extract_tensor_elem(&tensor_ty).unwrap();
        match extracted_elem {
            Ty::Con(tc) => assert_eq!(tc.name.as_str(), "Int32"),
            _ => panic!("expected Int32 type"),
        }
    }
}
