//! Type mapping from BHC types to LLVM types.
//!
//! This module handles the conversion of BHC's type representation
//! to LLVM's type system.

use inkwell::context::Context;
use inkwell::types::{
    BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FloatType, FunctionType, IntType, PointerType,
    StructType, VoidType,
};
use inkwell::AddressSpace;

/// Maps BHC types to LLVM types.
///
/// This struct holds references to commonly used LLVM types and provides
/// methods for constructing more complex types.
pub struct TypeMapper<'ctx> {
    context: &'ctx Context,
}

impl<'ctx> TypeMapper<'ctx> {
    /// Create a new type mapper for the given context.
    pub fn new(context: &'ctx Context) -> Self {
        Self { context }
    }

    /// Get the LLVM context.
    #[must_use]
    pub fn context(&self) -> &'ctx Context {
        self.context
    }

    // ========================================================================
    // Primitive Types
    // ========================================================================

    /// The void type.
    #[must_use]
    pub fn void_type(&self) -> VoidType<'ctx> {
        self.context.void_type()
    }

    /// Boolean type (i1).
    #[must_use]
    pub fn bool_type(&self) -> IntType<'ctx> {
        self.context.bool_type()
    }

    /// 8-bit integer type.
    #[must_use]
    pub fn i8_type(&self) -> IntType<'ctx> {
        self.context.i8_type()
    }

    /// 16-bit integer type.
    #[must_use]
    pub fn i16_type(&self) -> IntType<'ctx> {
        self.context.i16_type()
    }

    /// 32-bit integer type.
    #[must_use]
    pub fn i32_type(&self) -> IntType<'ctx> {
        self.context.i32_type()
    }

    /// 64-bit integer type.
    ///
    /// This is the default integer type in BHC (Haskell's `Int`).
    #[must_use]
    pub fn i64_type(&self) -> IntType<'ctx> {
        self.context.i64_type()
    }

    /// Pointer-sized integer type.
    #[must_use]
    pub fn isize_type(&self) -> IntType<'ctx> {
        // TODO: Get actual pointer size from target
        self.context.i64_type()
    }

    /// 32-bit floating point type.
    #[must_use]
    pub fn f32_type(&self) -> FloatType<'ctx> {
        self.context.f32_type()
    }

    /// 64-bit floating point type.
    ///
    /// This is the default floating point type in BHC (Haskell's `Double`).
    #[must_use]
    pub fn f64_type(&self) -> FloatType<'ctx> {
        self.context.f64_type()
    }

    // ========================================================================
    // Pointer Types
    // ========================================================================

    /// Generic pointer type (opaque pointer in LLVM 15+).
    #[must_use]
    pub fn ptr_type(&self) -> PointerType<'ctx> {
        self.context.ptr_type(AddressSpace::default())
    }

    /// Pointer to i8 (commonly used for strings).
    #[must_use]
    pub fn i8_ptr_type(&self) -> PointerType<'ctx> {
        self.context.ptr_type(AddressSpace::default())
    }

    // ========================================================================
    // BHC Runtime Types
    // ========================================================================

    /// The object header type.
    ///
    /// Every heap object in BHC starts with this header:
    /// ```text
    /// struct ObjHeader {
    ///     void* info_ptr;  // Pointer to info table
    /// }
    /// ```
    #[must_use]
    pub fn obj_header_type(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[self.ptr_type().into()], // info_ptr
            false,
        )
    }

    /// A boxed value type (header + payload).
    ///
    /// ```text
    /// struct BoxedValue {
    ///     ObjHeader header;
    ///     i64 value;
    /// }
    /// ```
    #[must_use]
    pub fn boxed_int_type(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[self.obj_header_type().into(), self.i64_type().into()],
            false,
        )
    }

    /// A closure type.
    ///
    /// ```text
    /// struct Closure {
    ///     ObjHeader header;
    ///     void* entry_code;
    ///     i64 arity;
    ///     // Free variables follow...
    /// }
    /// ```
    #[must_use]
    pub fn closure_type(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[
                self.obj_header_type().into(),
                self.ptr_type().into(), // entry_code
                self.i64_type().into(), // arity
            ],
            false,
        )
    }

    /// A thunk type.
    ///
    /// ```text
    /// struct Thunk {
    ///     ObjHeader header;
    ///     void* code;  // Code to evaluate, or indirection
    ///     // Payload for suspended computation...
    /// }
    /// ```
    #[must_use]
    pub fn thunk_type(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[
                self.obj_header_type().into(),
                self.ptr_type().into(), // code/indirection
            ],
            false,
        )
    }

    // ========================================================================
    // Function Types
    // ========================================================================

    /// Create a function type with the given return type and parameter types.
    pub fn fn_type(
        &self,
        ret_type: BasicTypeEnum<'ctx>,
        param_types: &[BasicMetadataTypeEnum<'ctx>],
        is_var_arg: bool,
    ) -> FunctionType<'ctx> {
        ret_type.fn_type(param_types, is_var_arg)
    }

    /// Create a void function type.
    pub fn void_fn_type(
        &self,
        param_types: &[BasicMetadataTypeEnum<'ctx>],
        is_var_arg: bool,
    ) -> FunctionType<'ctx> {
        self.void_type().fn_type(param_types, is_var_arg)
    }

    /// The type of a Haskell IO action that returns unit.
    ///
    /// In the simple case: `void (void)`
    #[must_use]
    pub fn io_unit_fn_type(&self) -> FunctionType<'ctx> {
        self.void_type().fn_type(&[], false)
    }

    /// The type of a Haskell function returning an Int.
    ///
    /// For now: `i64 (void)`
    #[must_use]
    pub fn int_fn_type(&self) -> FunctionType<'ctx> {
        self.i64_type().fn_type(&[], false)
    }
}
