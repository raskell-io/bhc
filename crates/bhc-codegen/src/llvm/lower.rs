//! Lowering from Core IR to LLVM IR.
//!
//! This module implements the translation from BHC's Core IR to LLVM IR.
//! The lowering handles:
//!
//! - Literals: Int, Double, Char, String
//! - Variables: Mapped to LLVM SSA values
//! - Function applications: Compiled to calls
//! - Lambdas: Compiled to closures or direct functions
//! - Let bindings: Compiled to LLVM allocas/phis
//! - Case expressions: Compiled to switch/branch

use crate::{CodegenError, CodegenResult};
use bhc_core::{Alt, AltCon, Bind, CoreModule, DataCon, Expr, Literal, Var, VarId};
use bhc_intern::Symbol;
use rustc_hash::FxHashSet;

/// Primitive operations that compile directly to LLVM instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Rem,
    Quot,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Boolean
    And,
    Or,
    Not,

    // Unary numeric
    Negate,
    Abs,
    Signum,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    ShiftL,
    ShiftR,
    Complement,
}
use bhc_index::Idx;
use bhc_types::Ty;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::types::{BasicType, BasicTypeEnum};
use inkwell::values::{BasicValueEnum, FloatValue, FunctionValue, IntValue, PointerValue};
use rustc_hash::FxHashMap;

use super::context::LlvmContext;
use super::module::LlvmModule;
use super::types::TypeMapper;

/// Metadata about a data constructor, used for code generation.
#[derive(Clone, Debug)]
pub struct ConstructorMeta {
    /// The constructor's tag (0-based index within the data type).
    pub tag: u32,
    /// The number of fields this constructor has.
    pub arity: u32,
}

/// State for lowering Core IR to LLVM IR.
///
/// The struct has two lifetimes:
/// - `'ctx`: The LLVM context lifetime (for types and values)
/// - `'m`: The module reference lifetime (can be shorter than 'ctx)
pub struct Lowering<'ctx, 'm> {
    /// The underlying LLVM context (borrowed from LlvmContext).
    llvm_ctx: &'ctx Context,
    /// The LLVM module being generated.
    module: &'m LlvmModule<'ctx>,
    /// Mapping from Core variables to LLVM values.
    env: FxHashMap<VarId, BasicValueEnum<'ctx>>,
    /// Mapping from Core variables to LLVM functions (for top-level bindings).
    functions: FxHashMap<VarId, FunctionValue<'ctx>>,
    /// Counter for generating unique closure names.
    closure_counter: u32,
    /// Mapping from constructor names to metadata (tag, arity).
    /// This is populated from DataCon entries in case alternatives.
    constructor_metadata: FxHashMap<String, ConstructorMeta>,
    /// Whether we're currently lowering an expression in tail position.
    /// Used for tail call optimization.
    in_tail_position: bool,
}

impl<'ctx, 'm> Lowering<'ctx, 'm> {
    /// Create a new lowering context.
    pub fn new(ctx: &'ctx LlvmContext, module: &'m LlvmModule<'ctx>) -> Self {
        let mut lowering = Self {
            llvm_ctx: ctx.llvm_context(),
            module,
            env: FxHashMap::default(),
            functions: FxHashMap::default(),
            closure_counter: 0,
            constructor_metadata: FxHashMap::default(),
            in_tail_position: false,
        };
        lowering.declare_rts_functions();
        lowering
    }

    /// Register a constructor's metadata for later use.
    pub fn register_constructor(&mut self, name: &str, tag: u32, arity: u32) {
        self.constructor_metadata.insert(
            name.to_string(),
            ConstructorMeta { tag, arity },
        );
    }

    /// Declare external RTS functions.
    fn declare_rts_functions(&mut self) {
        // Get all types upfront to avoid borrow conflicts
        let void_type = self.llvm_ctx.void_type();
        let i64_type = self.type_mapper().i64_type();
        let f64_type = self.type_mapper().f64_type();
        let i32_type = self.type_mapper().i32_type();
        let i8_ptr_type = self.llvm_ctx.ptr_type(inkwell::AddressSpace::default());

        // bhc_print_int_ln(i64) -> void
        let print_int_ln_type = void_type.fn_type(&[i64_type.into()], false);
        let print_int_ln = self.module.llvm_module().add_function("bhc_print_int_ln", print_int_ln_type, None);
        self.functions.insert(VarId::new(1000), print_int_ln); // Use high ID to avoid conflicts

        // bhc_print_double_ln(f64) -> void
        let print_double_ln_type = void_type.fn_type(&[f64_type.into()], false);
        let print_double_ln = self.module.llvm_module().add_function("bhc_print_double_ln", print_double_ln_type, None);
        self.functions.insert(VarId::new(1001), print_double_ln);

        // bhc_print_string_ln(*i8) -> void
        let print_string_ln_type = void_type.fn_type(&[i8_ptr_type.into()], false);
        let print_string_ln = self.module.llvm_module().add_function("bhc_print_string_ln", print_string_ln_type, None);
        self.functions.insert(VarId::new(1002), print_string_ln);

        // bhc_print_int(i64) -> void
        let print_int_type = void_type.fn_type(&[i64_type.into()], false);
        let print_int = self.module.llvm_module().add_function("bhc_print_int", print_int_type, None);
        self.functions.insert(VarId::new(1003), print_int);

        // bhc_print_string(*i8) -> void
        let print_string_type = void_type.fn_type(&[i8_ptr_type.into()], false);
        let print_string = self.module.llvm_module().add_function("bhc_print_string", print_string_type, None);
        self.functions.insert(VarId::new(1004), print_string);

        // bhc_alloc(size: i64) -> ptr - allocate heap memory
        let alloc_type = i8_ptr_type.fn_type(&[i64_type.into()], false);
        let alloc_fn = self.module.llvm_module().add_function("bhc_alloc", alloc_type, None);
        self.functions.insert(VarId::new(1005), alloc_fn);

        // bhc_error(*i8) -> void - runtime error (does not return)
        let error_type = void_type.fn_type(&[i8_ptr_type.into()], false);
        let error_fn = self.module.llvm_module().add_function("bhc_error", error_type, None);
        self.functions.insert(VarId::new(1006), error_fn);

        // bhc_print_bool(i64) -> void - print True/False
        let print_bool_type = void_type.fn_type(&[i64_type.into()], false);
        let print_bool = self.module.llvm_module().add_function("bhc_print_bool", print_bool_type, None);
        self.functions.insert(VarId::new(1007), print_bool);

        // bhc_print_bool_ln(i64) -> void - print True/False with newline
        let print_bool_ln_type = void_type.fn_type(&[i64_type.into()], false);
        let print_bool_ln = self.module.llvm_module().add_function("bhc_print_bool_ln", print_bool_ln_type, None);
        self.functions.insert(VarId::new(1008), print_bool_ln);

        // bhc_print_char(i32) -> void - print a character
        let print_char_type = void_type.fn_type(&[i32_type.into()], false);
        let print_char = self.module.llvm_module().add_function("bhc_print_char", print_char_type, None);
        self.functions.insert(VarId::new(1009), print_char);

        // bhc_print_newline() -> void
        let print_newline_type = void_type.fn_type(&[], false);
        let print_newline = self.module.llvm_module().add_function("bhc_print_newline", print_newline_type, None);
        self.functions.insert(VarId::new(1010), print_newline);
    }

    // ========================================================================
    // ADT (Algebraic Data Type) Value Representation
    // ========================================================================
    //
    // ADT values are represented as heap-allocated structs:
    //
    //   struct ADTValue {
    //       i64 tag;          // Constructor tag (0, 1, 2, ...)
    //       ptr fields[];     // Variable-length array of field pointers
    //   }
    //
    // For example, `Just 42` would be:
    //   { tag: 1, fields: [ptr_to_42] }
    //
    // And `Nothing` would be:
    //   { tag: 0, fields: [] }
    // ========================================================================

    /// Get the LLVM struct type for an ADT value with the given arity.
    fn adt_type(&self, arity: u32) -> inkwell::types::StructType<'ctx> {
        let tm = self.type_mapper();
        let tag_type = tm.i64_type();
        let ptr_type = tm.ptr_type();

        // Create array type for fields
        let fields_type = ptr_type.array_type(arity);

        // Struct: { i64 tag, [arity x ptr] fields }
        self.llvm_ctx.struct_type(&[tag_type.into(), fields_type.into()], false)
    }

    /// Allocate an ADT value with the given tag and arity.
    fn alloc_adt(
        &self,
        tag: u32,
        arity: u32,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let tm = self.type_mapper();
        let adt_ty = self.adt_type(arity);

        // Calculate size: sizeof(i64) + arity * sizeof(ptr)
        let size = 8 + (arity as u64) * 8;

        // Call bhc_alloc
        let alloc_fn = self.functions.get(&VarId::new(1005)).ok_or_else(|| {
            CodegenError::Internal("bhc_alloc not declared".to_string())
        })?;

        let size_val = tm.i64_type().const_int(size, false);
        let raw_ptr = self
            .builder()
            .build_call(*alloc_fn, &[size_val.into()], "adt_alloc")
            .map_err(|e| CodegenError::Internal(format!("failed to call bhc_alloc: {:?}", e)))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CodegenError::Internal("bhc_alloc returned void".to_string()))?;

        let ptr = raw_ptr.into_pointer_value();

        // Store the tag at offset 0
        let tag_ptr = self
            .builder()
            .build_struct_gep(adt_ty, ptr, 0, "tag_ptr")
            .map_err(|e| CodegenError::Internal(format!("failed to get tag ptr: {:?}", e)))?;

        let tag_val = tm.i64_type().const_int(tag as u64, false);
        self.builder()
            .build_store(tag_ptr, tag_val)
            .map_err(|e| CodegenError::Internal(format!("failed to store tag: {:?}", e)))?;

        Ok(ptr)
    }

    /// Store a field value into an ADT at the given index.
    fn store_adt_field(
        &self,
        adt_ptr: PointerValue<'ctx>,
        arity: u32,
        field_index: u32,
        value: BasicValueEnum<'ctx>,
    ) -> CodegenResult<()> {
        let adt_ty = self.adt_type(arity);
        let tm = self.type_mapper();

        // Get pointer to fields array
        let fields_ptr = self
            .builder()
            .build_struct_gep(adt_ty, adt_ptr, 1, "fields_ptr")
            .map_err(|e| CodegenError::Internal(format!("failed to get fields ptr: {:?}", e)))?;

        // Get pointer to specific field
        let field_ptr = unsafe {
            self.builder()
                .build_in_bounds_gep(
                    tm.ptr_type().array_type(arity),
                    fields_ptr,
                    &[
                        tm.i64_type().const_zero(),
                        tm.i64_type().const_int(field_index as u64, false),
                    ],
                    &format!("field_{}", field_index),
                )
                .map_err(|e| CodegenError::Internal(format!("failed to get field ptr: {:?}", e)))?
        };

        // Convert value to pointer if needed
        let ptr_val = self.value_to_ptr(value)?;
        self.builder()
            .build_store(field_ptr, ptr_val)
            .map_err(|e| CodegenError::Internal(format!("failed to store field: {:?}", e)))?;

        Ok(())
    }

    /// Convert a basic value to a pointer (boxing primitives if needed).
    fn value_to_ptr(&self, value: BasicValueEnum<'ctx>) -> CodegenResult<PointerValue<'ctx>> {
        match value {
            BasicValueEnum::PointerValue(p) => Ok(p),
            BasicValueEnum::IntValue(i) => {
                // Box the integer: cast to pointer
                self.builder()
                    .build_int_to_ptr(i, self.type_mapper().ptr_type(), "box_int")
                    .map_err(|e| CodegenError::Internal(format!("failed to box int: {:?}", e)))
            }
            BasicValueEnum::FloatValue(f) => {
                // Box the float: cast bits to int, then to pointer
                let bits = self
                    .builder()
                    .build_bit_cast(f, self.type_mapper().i64_type(), "float_bits")
                    .map_err(|e| CodegenError::Internal(format!("failed to cast float: {:?}", e)))?
                    .into_int_value();
                self.builder()
                    .build_int_to_ptr(bits, self.type_mapper().ptr_type(), "box_float")
                    .map_err(|e| CodegenError::Internal(format!("failed to box float: {:?}", e)))
            }
            _ => Err(CodegenError::Unsupported(
                "cannot box this value type".to_string(),
            )),
        }
    }

    /// Coerce a value to match a target type.
    /// Used to ensure PHI node operands have consistent types.
    fn coerce_to_type(
        &self,
        value: BasicValueEnum<'ctx>,
        target_type: inkwell::types::BasicTypeEnum<'ctx>,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let value_type = value.get_type();

        // If types already match, no coercion needed
        if value_type == target_type {
            return Ok(value);
        }

        let tm = self.type_mapper();

        match (value, target_type) {
            // Pointer to integer: unbox (ptr_to_int)
            (BasicValueEnum::PointerValue(p), inkwell::types::BasicTypeEnum::IntType(int_ty)) => {
                let int_val = self.builder()
                    .build_ptr_to_int(p, int_ty, "coerce_ptr_to_int")
                    .map_err(|e| CodegenError::Internal(format!("failed to coerce ptr to int: {:?}", e)))?;
                Ok(int_val.into())
            }
            // Integer to pointer: box (int_to_ptr)
            (BasicValueEnum::IntValue(i), inkwell::types::BasicTypeEnum::PointerType(_)) => {
                // First extend/truncate to i64 if needed, then convert to pointer
                let i64_val = if i.get_type().get_bit_width() < 64 {
                    self.builder()
                        .build_int_z_extend(i, tm.i64_type(), "extend_to_i64")
                        .map_err(|e| CodegenError::Internal(format!("failed to extend int: {:?}", e)))?
                } else if i.get_type().get_bit_width() > 64 {
                    self.builder()
                        .build_int_truncate(i, tm.i64_type(), "truncate_to_i64")
                        .map_err(|e| CodegenError::Internal(format!("failed to truncate int: {:?}", e)))?
                } else {
                    i
                };
                let ptr_val = self.builder()
                    .build_int_to_ptr(i64_val, tm.ptr_type(), "coerce_int_to_ptr")
                    .map_err(|e| CodegenError::Internal(format!("failed to coerce int to ptr: {:?}", e)))?;
                Ok(ptr_val.into())
            }
            // Pointer to float: unbox (ptr_to_int then bit_cast)
            (BasicValueEnum::PointerValue(p), inkwell::types::BasicTypeEnum::FloatType(float_ty)) => {
                let bits = self.builder()
                    .build_ptr_to_int(p, tm.i64_type(), "coerce_ptr_to_bits")
                    .map_err(|e| CodegenError::Internal(format!("failed to coerce ptr to bits: {:?}", e)))?;
                // For f32, truncate to i32 first, then bit_cast
                let float_val = if float_ty == tm.f32_type() {
                    let bits32 = self.builder()
                        .build_int_truncate(bits, tm.i32_type(), "truncate_bits_f32")
                        .map_err(|e| CodegenError::Internal(format!("failed to truncate bits: {:?}", e)))?;
                    self.builder()
                        .build_bit_cast(bits32, float_ty, "coerce_to_f32")
                        .map_err(|e| CodegenError::Internal(format!("failed to coerce to f32: {:?}", e)))?
                } else {
                    self.builder()
                        .build_bit_cast(bits, float_ty, "coerce_to_f64")
                        .map_err(|e| CodegenError::Internal(format!("failed to coerce to f64: {:?}", e)))?
                };
                Ok(float_val)
            }
            // Float to pointer: box (bit_cast then int_to_ptr)
            (BasicValueEnum::FloatValue(f), inkwell::types::BasicTypeEnum::PointerType(_)) => {
                let bits = self.builder()
                    .build_bit_cast(f, tm.i64_type(), "float_to_bits")
                    .map_err(|e| CodegenError::Internal(format!("failed to cast float to bits: {:?}", e)))?
                    .into_int_value();
                let ptr_val = self.builder()
                    .build_int_to_ptr(bits, tm.ptr_type(), "coerce_float_to_ptr")
                    .map_err(|e| CodegenError::Internal(format!("failed to coerce float to ptr: {:?}", e)))?;
                Ok(ptr_val.into())
            }
            // Integer width conversion
            (BasicValueEnum::IntValue(i), inkwell::types::BasicTypeEnum::IntType(int_ty)) => {
                let src_bits = i.get_type().get_bit_width();
                let dst_bits = int_ty.get_bit_width();
                let result = if src_bits < dst_bits {
                    self.builder()
                        .build_int_s_extend(i, int_ty, "sext")
                        .map_err(|e| CodegenError::Internal(format!("failed to sign extend: {:?}", e)))?
                } else {
                    self.builder()
                        .build_int_truncate(i, int_ty, "trunc")
                        .map_err(|e| CodegenError::Internal(format!("failed to truncate: {:?}", e)))?
                };
                Ok(result.into())
            }
            _ => {
                // Types don't match and we don't know how to coerce
                Err(CodegenError::Internal(format!(
                    "cannot coerce {:?} to {:?}",
                    value_type, target_type
                )))
            }
        }
    }

    /// Extract the tag from an ADT value.
    fn extract_adt_tag(&self, adt_ptr: PointerValue<'ctx>) -> CodegenResult<IntValue<'ctx>> {
        // We need to use a generic adt type for reading - use arity 0 since tag is always at offset 0
        let adt_ty = self.adt_type(0);

        let tag_ptr = self
            .builder()
            .build_struct_gep(adt_ty, adt_ptr, 0, "tag_ptr")
            .map_err(|e| CodegenError::Internal(format!("failed to get tag ptr: {:?}", e)))?;

        let tag = self
            .builder()
            .build_load(self.type_mapper().i64_type(), tag_ptr, "tag")
            .map_err(|e| CodegenError::Internal(format!("failed to load tag: {:?}", e)))?;

        Ok(tag.into_int_value())
    }

    /// Extract a field from an ADT value.
    fn extract_adt_field(
        &self,
        adt_ptr: PointerValue<'ctx>,
        arity: u32,
        field_index: u32,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let adt_ty = self.adt_type(arity);
        let tm = self.type_mapper();

        // Get pointer to fields array
        let fields_ptr = self
            .builder()
            .build_struct_gep(adt_ty, adt_ptr, 1, "fields_ptr")
            .map_err(|e| CodegenError::Internal(format!("failed to get fields ptr: {:?}", e)))?;

        // Get pointer to specific field
        let field_ptr = unsafe {
            self.builder()
                .build_in_bounds_gep(
                    tm.ptr_type().array_type(arity),
                    fields_ptr,
                    &[
                        tm.i64_type().const_zero(),
                        tm.i64_type().const_int(field_index as u64, false),
                    ],
                    &format!("field_ptr_{}", field_index),
                )
                .map_err(|e| CodegenError::Internal(format!("failed to get field ptr: {:?}", e)))?
        };

        // Load the field value (which is a pointer)
        let field_val = self
            .builder()
            .build_load(tm.ptr_type(), field_ptr, &format!("field_{}", field_index))
            .map_err(|e| CodegenError::Internal(format!("failed to load field: {:?}", e)))?;

        Ok(field_val.into_pointer_value())
    }

    /// Get the RTS function ID for a builtin name.
    fn rts_function_id(&self, name: &str) -> Option<VarId> {
        match name {
            "print" => Some(VarId::new(1000)),      // bhc_print_int_ln for Int
            "putStrLn" => Some(VarId::new(1002)),   // bhc_print_string_ln
            "putStr" => Some(VarId::new(1004)),     // bhc_print_string
            _ => None,
        }
    }

    // ========================================================================
    // Builtin Functions
    // ========================================================================
    //
    // These are common Haskell functions that we implement directly in LLVM
    // for performance (avoiding function call overhead) or because they need
    // special handling.
    // ========================================================================

    /// Check if a name is a builtin function and return its arity.
    fn builtin_info(&self, name: &str) -> Option<u32> {
        match name {
            // List operations
            "head" => Some(1),
            "tail" => Some(1),
            "null" => Some(1),
            "length" => Some(1),

            // Tuple operations
            "fst" => Some(1),
            "snd" => Some(1),

            // Maybe operations
            "fromJust" => Some(1),
            "isJust" => Some(1),
            "isNothing" => Some(1),

            // Either operations
            "isLeft" => Some(1),
            "isRight" => Some(1),

            // Error
            "error" => Some(1),
            "undefined" => Some(0),

            // Misc
            "seq" => Some(2),
            "id" => Some(1),
            "const" => Some(2),

            // IO operations
            "putStrLn" => Some(1),
            "putStr" => Some(1),
            "putChar" => Some(1),
            "print" => Some(1),
            "getLine" => Some(0),

            _ => None,
        }
    }

    /// Check if an expression is a saturated builtin function application.
    fn is_saturated_builtin<'a>(&self, expr: &'a Expr) -> Option<(&'a str, Vec<&'a Expr>)> {
        // Collect arguments while unwrapping applications
        let mut args = Vec::new();
        let mut current = expr;

        while let Expr::App(func, arg, _) = current {
            args.push(arg.as_ref());
            current = func.as_ref();
        }

        // Check if the head is a builtin function
        if let Expr::Var(var, _) = current {
            let name = var.name.as_str();
            if let Some(arity) = self.builtin_info(name) {
                args.reverse();
                if args.len() == arity as usize {
                    return Some((name, args));
                }
            }
        }

        None
    }

    /// Lower a builtin function application.
    fn lower_builtin(
        &mut self,
        name: &str,
        args: &[&Expr],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        match name {
            // List operations
            "head" => self.lower_builtin_head(args[0]),
            "tail" => self.lower_builtin_tail(args[0]),
            "null" => self.lower_builtin_null(args[0]),
            "length" => self.lower_builtin_length(args[0]),

            // Tuple operations
            "fst" => self.lower_builtin_fst(args[0]),
            "snd" => self.lower_builtin_snd(args[0]),

            // Maybe operations
            "fromJust" => self.lower_builtin_from_just(args[0]),
            "isJust" => self.lower_builtin_is_just(args[0]),
            "isNothing" => self.lower_builtin_is_nothing(args[0]),

            // Either operations
            "isLeft" => self.lower_builtin_is_left(args[0]),
            "isRight" => self.lower_builtin_is_right(args[0]),

            // Error
            "error" => self.lower_builtin_error(args[0]),
            "undefined" => self.lower_builtin_undefined(),

            // Misc
            "seq" => self.lower_builtin_seq(args[0], args[1]),
            "id" => self.lower_expr(args[0]),
            "const" => self.lower_expr(args[0]),

            // IO operations
            "putStrLn" => self.lower_builtin_put_str_ln(args[0]),
            "putStr" => self.lower_builtin_put_str(args[0]),
            "putChar" => self.lower_builtin_put_char(args[0]),
            "print" => self.lower_builtin_print(args[0]),
            "getLine" => self.lower_builtin_get_line(),

            _ => Err(CodegenError::Internal(format!("unknown builtin: {}", name))),
        }
    }

    /// Lower `head` - extract first element of a list.
    /// head [] = error "empty list"
    /// head (x:_) = x
    fn lower_builtin_head(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("head: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("head expects a list".to_string())),
        };

        // Extract tag to check if empty
        let tag = self.extract_adt_tag(list_ptr)?;
        let tm = self.type_mapper();

        // Create blocks for empty check
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let empty_block = self.llvm_context().append_basic_block(current_fn, "head_empty");
        let cons_block = self.llvm_context().append_basic_block(current_fn, "head_cons");

        // Branch on tag (0 = [], 1 = :)
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder()
            .build_conditional_branch(is_empty, empty_block, cons_block)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Empty case: call error
        self.builder().position_at_end(empty_block);
        let error_msg = self.module.add_global_string("head_empty_error", "head: empty list");
        let error_fn = self.functions.get(&VarId::new(1006)).ok_or_else(|| {
            CodegenError::Internal("bhc_error not declared".to_string())
        })?;
        self.builder()
            .build_call(*error_fn, &[error_msg.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call error: {:?}", e)))?;
        self.builder()
            .build_unreachable()
            .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;

        // Cons case: extract head (field 0)
        self.builder().position_at_end(cons_block);
        let head_ptr = self.extract_adt_field(list_ptr, 2, 0)?; // arity=2 for (:)
        Ok(Some(head_ptr.into()))
    }

    /// Lower `tail` - extract rest of a list.
    /// tail [] = error "empty list"
    /// tail (_:xs) = xs
    fn lower_builtin_tail(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("tail: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("tail expects a list".to_string())),
        };

        // Extract tag to check if empty
        let tag = self.extract_adt_tag(list_ptr)?;
        let tm = self.type_mapper();

        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let empty_block = self.llvm_context().append_basic_block(current_fn, "tail_empty");
        let cons_block = self.llvm_context().append_basic_block(current_fn, "tail_cons");

        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder()
            .build_conditional_branch(is_empty, empty_block, cons_block)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Empty case: call error
        self.builder().position_at_end(empty_block);
        let error_msg = self.module.add_global_string("tail_empty_error", "tail: empty list");
        let error_fn = self.functions.get(&VarId::new(1006)).ok_or_else(|| {
            CodegenError::Internal("bhc_error not declared".to_string())
        })?;
        self.builder()
            .build_call(*error_fn, &[error_msg.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call error: {:?}", e)))?;
        self.builder()
            .build_unreachable()
            .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;

        // Cons case: extract tail (field 1)
        self.builder().position_at_end(cons_block);
        let tail_ptr = self.extract_adt_field(list_ptr, 2, 1)?;
        Ok(Some(tail_ptr.into()))
    }

    /// Lower `null` - check if list is empty.
    /// null [] = True
    /// null (_:_) = False
    fn lower_builtin_null(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("null: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("null expects a list".to_string())),
        };

        // Extract tag: 0 = [], 1 = (:)
        let tag = self.extract_adt_tag(list_ptr)?;
        let tm = self.type_mapper();

        // null = (tag == 0)
        let is_null = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_null")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        // Extend i1 to i64 for Bool representation
        let result = self.builder()
            .build_int_z_extend(is_null, tm.i64_type(), "null_result")
            .map_err(|e| CodegenError::Internal(format!("failed to extend: {:?}", e)))?;

        Ok(Some(result.into()))
    }

    /// Lower `length` - compute list length.
    /// This generates a loop to count elements.
    fn lower_builtin_length(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("length: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("length expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // Create loop blocks
        let loop_header = self.llvm_context().append_basic_block(current_fn, "length_header");
        let loop_body = self.llvm_context().append_basic_block(current_fn, "length_body");
        let loop_exit = self.llvm_context().append_basic_block(current_fn, "length_exit");

        // Initialize count to 0 and branch to header
        let init_count = tm.i64_type().const_zero();
        self.builder()
            .build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header: phi nodes for count and current list pointer
        self.builder().position_at_end(loop_header);

        let count_phi = self.builder()
            .build_phi(tm.i64_type(), "count")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let list_phi = self.builder()
            .build_phi(tm.ptr_type(), "list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Get the entry block for phi incoming
        let entry_block = current_fn.get_first_basic_block()
            .ok_or_else(|| CodegenError::Internal("no entry block".to_string()))?;

        // Check if current list is empty
        let current_list = list_phi.as_basic_value().into_pointer_value();
        let tag = self.extract_adt_tag(current_list)?;

        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        self.builder()
            .build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: increment count, get tail
        self.builder().position_at_end(loop_body);

        let new_count = self.builder()
            .build_int_add(count_phi.as_basic_value().into_int_value(), tm.i64_type().const_int(1, false), "new_count")
            .map_err(|e| CodegenError::Internal(format!("failed to add: {:?}", e)))?;

        let tail_ptr = self.extract_adt_field(current_list, 2, 1)?;

        self.builder()
            .build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming edges
        // Note: We need to find the block that had the initial branch to loop_header
        // This is tricky because we already positioned at loop_body
        count_phi.add_incoming(&[
            (&init_count, entry_block),
            (&new_count, loop_body),
        ]);
        list_phi.add_incoming(&[
            (&list_ptr, entry_block),
            (&tail_ptr, loop_body),
        ]);

        // Loop exit: return final count
        self.builder().position_at_end(loop_exit);
        Ok(Some(count_phi.as_basic_value()))
    }

    /// Lower `fst` - extract first element of a pair.
    fn lower_builtin_fst(&mut self, pair_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let pair_val = self.lower_expr(pair_expr)?.ok_or_else(|| {
            CodegenError::Internal("fst: pair has no value".to_string())
        })?;

        let pair_ptr = match pair_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("fst expects a tuple".to_string())),
        };

        // Extract field 0 (first element)
        let fst_ptr = self.extract_adt_field(pair_ptr, 2, 0)?;
        Ok(Some(fst_ptr.into()))
    }

    /// Lower `snd` - extract second element of a pair.
    fn lower_builtin_snd(&mut self, pair_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let pair_val = self.lower_expr(pair_expr)?.ok_or_else(|| {
            CodegenError::Internal("snd: pair has no value".to_string())
        })?;

        let pair_ptr = match pair_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("snd expects a tuple".to_string())),
        };

        // Extract field 1 (second element)
        let snd_ptr = self.extract_adt_field(pair_ptr, 2, 1)?;
        Ok(Some(snd_ptr.into()))
    }

    /// Lower `fromJust` - extract value from Just, error on Nothing.
    fn lower_builtin_from_just(&mut self, maybe_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let maybe_val = self.lower_expr(maybe_expr)?.ok_or_else(|| {
            CodegenError::Internal("fromJust: maybe has no value".to_string())
        })?;

        let maybe_ptr = match maybe_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("fromJust expects Maybe".to_string())),
        };

        let tag = self.extract_adt_tag(maybe_ptr)?;
        let tm = self.type_mapper();

        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let nothing_block = self.llvm_context().append_basic_block(current_fn, "fromJust_nothing");
        let just_block = self.llvm_context().append_basic_block(current_fn, "fromJust_just");

        // Tag: Nothing=0, Just=1
        let is_nothing = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_nothing")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        self.builder()
            .build_conditional_branch(is_nothing, nothing_block, just_block)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Nothing case: error
        self.builder().position_at_end(nothing_block);
        let error_msg = self.module.add_global_string("fromJust_error", "fromJust: Nothing");
        let error_fn = self.functions.get(&VarId::new(1006)).ok_or_else(|| {
            CodegenError::Internal("bhc_error not declared".to_string())
        })?;
        self.builder()
            .build_call(*error_fn, &[error_msg.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call error: {:?}", e)))?;
        self.builder()
            .build_unreachable()
            .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;

        // Just case: extract value
        self.builder().position_at_end(just_block);
        let val_ptr = self.extract_adt_field(maybe_ptr, 1, 0)?;
        Ok(Some(val_ptr.into()))
    }

    /// Lower `isJust` - check if Maybe is Just.
    fn lower_builtin_is_just(&mut self, maybe_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let maybe_val = self.lower_expr(maybe_expr)?.ok_or_else(|| {
            CodegenError::Internal("isJust: maybe has no value".to_string())
        })?;

        let maybe_ptr = match maybe_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("isJust expects Maybe".to_string())),
        };

        let tag = self.extract_adt_tag(maybe_ptr)?;
        let tm = self.type_mapper();

        // isJust = (tag == 1)
        let is_just = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_int(1, false), "is_just")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        let result = self.builder()
            .build_int_z_extend(is_just, tm.i64_type(), "isJust_result")
            .map_err(|e| CodegenError::Internal(format!("failed to extend: {:?}", e)))?;

        Ok(Some(result.into()))
    }

    /// Lower `isNothing` - check if Maybe is Nothing.
    fn lower_builtin_is_nothing(&mut self, maybe_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let maybe_val = self.lower_expr(maybe_expr)?.ok_or_else(|| {
            CodegenError::Internal("isNothing: maybe has no value".to_string())
        })?;

        let maybe_ptr = match maybe_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("isNothing expects Maybe".to_string())),
        };

        let tag = self.extract_adt_tag(maybe_ptr)?;
        let tm = self.type_mapper();

        // isNothing = (tag == 0)
        let is_nothing = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_nothing")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        let result = self.builder()
            .build_int_z_extend(is_nothing, tm.i64_type(), "isNothing_result")
            .map_err(|e| CodegenError::Internal(format!("failed to extend: {:?}", e)))?;

        Ok(Some(result.into()))
    }

    /// Lower `isLeft` - check if Either is Left.
    fn lower_builtin_is_left(&mut self, either_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let either_val = self.lower_expr(either_expr)?.ok_or_else(|| {
            CodegenError::Internal("isLeft: either has no value".to_string())
        })?;

        let either_ptr = match either_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("isLeft expects Either".to_string())),
        };

        let tag = self.extract_adt_tag(either_ptr)?;
        let tm = self.type_mapper();

        // isLeft = (tag == 0)
        let is_left = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_left")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        let result = self.builder()
            .build_int_z_extend(is_left, tm.i64_type(), "isLeft_result")
            .map_err(|e| CodegenError::Internal(format!("failed to extend: {:?}", e)))?;

        Ok(Some(result.into()))
    }

    /// Lower `isRight` - check if Either is Right.
    fn lower_builtin_is_right(&mut self, either_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let either_val = self.lower_expr(either_expr)?.ok_or_else(|| {
            CodegenError::Internal("isRight: either has no value".to_string())
        })?;

        let either_ptr = match either_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("isRight expects Either".to_string())),
        };

        let tag = self.extract_adt_tag(either_ptr)?;
        let tm = self.type_mapper();

        // isRight = (tag == 1)
        let is_right = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_int(1, false), "is_right")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        let result = self.builder()
            .build_int_z_extend(is_right, tm.i64_type(), "isRight_result")
            .map_err(|e| CodegenError::Internal(format!("failed to extend: {:?}", e)))?;

        Ok(Some(result.into()))
    }

    /// Lower `error` - runtime error.
    fn lower_builtin_error(&mut self, msg_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let msg_val = self.lower_expr(msg_expr)?.ok_or_else(|| {
            CodegenError::Internal("error: message has no value".to_string())
        })?;

        let msg_ptr = match msg_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => {
                // If it's not a pointer, use a default error message
                self.module.add_global_string("error_default", "error called")
            }
        };

        let error_fn = self.functions.get(&VarId::new(1006)).ok_or_else(|| {
            CodegenError::Internal("bhc_error not declared".to_string())
        })?;

        self.builder()
            .build_call(*error_fn, &[msg_ptr.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call error: {:?}", e)))?;

        self.builder()
            .build_unreachable()
            .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;

        // error never returns, but we need to return something for the type system
        Ok(Some(self.type_mapper().ptr_type().const_null().into()))
    }

    /// Lower `undefined` - always errors.
    fn lower_builtin_undefined(&mut self) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let error_msg = self.module.add_global_string("undefined_error", "undefined");
        let error_fn = self.functions.get(&VarId::new(1006)).ok_or_else(|| {
            CodegenError::Internal("bhc_error not declared".to_string())
        })?;

        self.builder()
            .build_call(*error_fn, &[error_msg.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call error: {:?}", e)))?;

        self.builder()
            .build_unreachable()
            .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;

        Ok(Some(self.type_mapper().ptr_type().const_null().into()))
    }

    /// Lower `seq` - force evaluation of first argument, return second.
    fn lower_builtin_seq(&mut self, a_expr: &Expr, b_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Evaluate first argument (for its effect of forcing evaluation)
        let _a = self.lower_expr(a_expr)?;
        // Return second argument
        self.lower_expr(b_expr)
    }

    // ========================================================================
    // IO Builtin Functions
    // ========================================================================

    /// Lower `putStrLn` - print a string followed by newline.
    fn lower_builtin_put_str_ln(&mut self, str_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let str_val = self.lower_expr(str_expr)?.ok_or_else(|| {
            CodegenError::Internal("putStrLn: string has no value".to_string())
        })?;

        let str_ptr = match str_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("putStrLn expects a string".to_string())),
        };

        // Call bhc_print_string_ln
        let print_fn = self.functions.get(&VarId::new(1002)).ok_or_else(|| {
            CodegenError::Internal("bhc_print_string_ln not declared".to_string())
        })?;

        self.builder()
            .build_call(*print_fn, &[str_ptr.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call print: {:?}", e)))?;

        // Return unit (null pointer for IO ())
        Ok(Some(self.type_mapper().ptr_type().const_null().into()))
    }

    /// Lower `putStr` - print a string without newline.
    fn lower_builtin_put_str(&mut self, str_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let str_val = self.lower_expr(str_expr)?.ok_or_else(|| {
            CodegenError::Internal("putStr: string has no value".to_string())
        })?;

        let str_ptr = match str_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("putStr expects a string".to_string())),
        };

        // Call bhc_print_string
        let print_fn = self.functions.get(&VarId::new(1004)).ok_or_else(|| {
            CodegenError::Internal("bhc_print_string not declared".to_string())
        })?;

        self.builder()
            .build_call(*print_fn, &[str_ptr.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call print: {:?}", e)))?;

        // Return unit
        Ok(Some(self.type_mapper().ptr_type().const_null().into()))
    }

    /// Lower `putChar` - print a single character.
    fn lower_builtin_put_char(&mut self, char_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let char_val = self.lower_expr(char_expr)?.ok_or_else(|| {
            CodegenError::Internal("putChar: char has no value".to_string())
        })?;

        let char_int = match char_val {
            BasicValueEnum::IntValue(i) => i,
            BasicValueEnum::PointerValue(p) => {
                // Might be a boxed char - unbox it
                self.builder()
                    .build_ptr_to_int(p, self.type_mapper().i64_type(), "unbox_char")
                    .map_err(|e| CodegenError::Internal(format!("failed to unbox char: {:?}", e)))?
            }
            _ => return Err(CodegenError::TypeError("putChar expects a Char".to_string())),
        };

        // Truncate to i32 for the RTS call
        let char_i32 = self.builder()
            .build_int_truncate(char_int, self.type_mapper().i32_type(), "char_i32")
            .map_err(|e| CodegenError::Internal(format!("failed to truncate char: {:?}", e)))?;

        // Call bhc_print_char
        let print_fn = self.functions.get(&VarId::new(1009)).ok_or_else(|| {
            CodegenError::Internal("bhc_print_char not declared".to_string())
        })?;

        self.builder()
            .build_call(*print_fn, &[char_i32.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call print_char: {:?}", e)))?;

        // Return unit
        Ok(Some(self.type_mapper().ptr_type().const_null().into()))
    }

    /// Lower `print` - print a value with its Show instance (simplified).
    ///
    /// For now, we detect the type at codegen time and call the appropriate
    /// print function. This is a simplification - real Haskell uses type classes.
    fn lower_builtin_print(&mut self, val_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Get the type of the expression to decide how to print
        let expr_ty = val_expr.ty();

        let val = self.lower_expr(val_expr)?.ok_or_else(|| {
            CodegenError::Internal("print: value has no result".to_string())
        })?;

        match val {
            BasicValueEnum::IntValue(i) => {
                // Print as integer
                let print_fn = self.functions.get(&VarId::new(1000)).ok_or_else(|| {
                    CodegenError::Internal("bhc_print_int_ln not declared".to_string())
                })?;

                self.builder()
                    .build_call(*print_fn, &[i.into()], "")
                    .map_err(|e| CodegenError::Internal(format!("failed to call print_int: {:?}", e)))?;
            }
            BasicValueEnum::FloatValue(f) => {
                // Print as double
                let print_fn = self.functions.get(&VarId::new(1001)).ok_or_else(|| {
                    CodegenError::Internal("bhc_print_double_ln not declared".to_string())
                })?;

                // Extend float to double if needed (check if it's f32)
                let f64_type = self.type_mapper().f64_type();
                let f32_type = self.type_mapper().f32_type();
                let f64_val = if f.get_type() == f32_type {
                    self.builder()
                        .build_float_ext(f, f64_type, "to_f64")
                        .map_err(|e| CodegenError::Internal(format!("failed to extend float: {:?}", e)))?
                } else {
                    f
                };

                self.builder()
                    .build_call(*print_fn, &[f64_val.into()], "")
                    .map_err(|e| CodegenError::Internal(format!("failed to call print_double: {:?}", e)))?;
            }
            BasicValueEnum::PointerValue(p) => {
                // Pointer could be a boxed value from closure call, string, or ADT.
                // Use the expression's type to decide how to print.
                if self.is_int_type(&expr_ty) || self.is_type_variable_or_error(&expr_ty) {
                    // Boxed integer (or polymorphic type that might be int) - unbox and print as int.
                    // For type variables from closures, we assume int since that's the most common case.
                    // This is safe because boxed ints use int_to_ptr which stores the value in the
                    // pointer bits, not as an actual memory reference.
                    let int_val = self.builder()
                        .build_ptr_to_int(p, self.type_mapper().i64_type(), "unbox_int")
                        .map_err(|e| CodegenError::Internal(format!("failed to unbox int: {:?}", e)))?;

                    let print_fn = self.functions.get(&VarId::new(1000)).ok_or_else(|| {
                        CodegenError::Internal("bhc_print_int_ln not declared".to_string())
                    })?;

                    self.builder()
                        .build_call(*print_fn, &[int_val.into()], "")
                        .map_err(|e| CodegenError::Internal(format!("failed to call print_int: {:?}", e)))?;
                } else if self.is_float_type(&expr_ty) {
                    // Boxed float - unbox and print as double
                    let bits = self.builder()
                        .build_ptr_to_int(p, self.type_mapper().i64_type(), "unbox_float_bits")
                        .map_err(|e| CodegenError::Internal(format!("failed to unbox float: {:?}", e)))?;
                    let float_val = self.builder()
                        .build_bit_cast(bits, self.type_mapper().f64_type(), "to_double")
                        .map_err(|e| CodegenError::Internal(format!("failed to cast to double: {:?}", e)))?;

                    let print_fn = self.functions.get(&VarId::new(1001)).ok_or_else(|| {
                        CodegenError::Internal("bhc_print_double_ln not declared".to_string())
                    })?;

                    self.builder()
                        .build_call(*print_fn, &[float_val.into()], "")
                        .map_err(|e| CodegenError::Internal(format!("failed to call print_double: {:?}", e)))?;
                } else {
                    // Assume it's a string or other pointer type
                    let print_fn = self.functions.get(&VarId::new(1002)).ok_or_else(|| {
                        CodegenError::Internal("bhc_print_string_ln not declared".to_string())
                    })?;

                    self.builder()
                        .build_call(*print_fn, &[p.into()], "")
                        .map_err(|e| CodegenError::Internal(format!("failed to call print_string: {:?}", e)))?;
                }
            }
            _ => {
                return Err(CodegenError::Unsupported(
                    "print: unsupported value type".to_string(),
                ));
            }
        }

        // Return unit
        Ok(Some(self.type_mapper().ptr_type().const_null().into()))
    }

    /// Check if a type is an integer type.
    fn is_int_type(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Con(con) => {
                let name = con.name.as_str();
                matches!(name, "Int" | "Int#" | "Int64" | "Int32" | "Integer" | "Word" | "Word64" | "Word32")
            }
            Ty::Prim(prim) => {
                use bhc_types::PrimTy;
                matches!(prim, PrimTy::I32 | PrimTy::I64 | PrimTy::U32 | PrimTy::U64)
            }
            Ty::App(f, _) => self.is_int_type(f),
            Ty::Forall(_, body) => self.is_int_type(body),
            _ => false,
        }
    }

    /// Check if a type is a float type.
    fn is_float_type(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Con(con) => {
                let name = con.name.as_str();
                matches!(name, "Float" | "Float#" | "Double" | "Double#")
            }
            Ty::Prim(prim) => {
                use bhc_types::PrimTy;
                matches!(prim, PrimTy::F32 | PrimTy::F64)
            }
            Ty::App(f, _) => self.is_float_type(f),
            Ty::Forall(_, body) => self.is_float_type(body),
            _ => false,
        }
    }

    /// Check if a type is a type variable (polymorphic) or error type.
    /// We treat error types the same as type variables since we don't know
    /// the concrete type and should default to treating it as a potential integer.
    fn is_type_variable_or_error(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Var(_) => true,
            Ty::Error => true,  // Error type might be unresolved - treat as unknown
            Ty::App(f, _) => self.is_type_variable_or_error(f),
            Ty::Forall(_, body) => self.is_type_variable_or_error(body),
            _ => false,
        }
    }

    /// Lower `getLine` - read a line from stdin.
    ///
    /// For now, this is a placeholder that returns an empty string.
    /// Full implementation requires RTS support.
    fn lower_builtin_get_line(&mut self) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Return empty string for now (placeholder)
        let empty_str = self.module.add_global_string("empty_string", "");
        Ok(Some(empty_str.into()))
    }

    /// Check if a name is a data constructor and return (tag, arity).
    ///
    /// This function checks:
    /// 1. Builtin types (Bool, Maybe, Either, List, Tuple, Ordering)
    /// 2. User-defined types registered via `register_constructor`
    fn constructor_info(&self, name: &str) -> Option<(u32, u32)> {
        // First check builtin constructors
        match name {
            // Bool constructors
            "False" => return Some((0, 0)),  // tag=0, arity=0
            "True" => return Some((1, 0)),   // tag=1, arity=0

            // Maybe constructors
            "Nothing" => return Some((0, 0)), // tag=0, arity=0
            "Just" => return Some((1, 1)),    // tag=1, arity=1

            // Either constructors
            "Left" => return Some((0, 1)),   // tag=0, arity=1
            "Right" => return Some((1, 1)),  // tag=1, arity=1

            // List constructors
            "[]" => return Some((0, 0)),     // tag=0, arity=0 (Nil)
            ":" => return Some((1, 2)),      // tag=1, arity=2 (Cons head tail)

            // Unit constructor
            "()" => return Some((0, 0)),     // tag=0, arity=0

            // Ordering constructors
            "LT" => return Some((0, 0)),     // tag=0, arity=0
            "EQ" => return Some((1, 0)),     // tag=1, arity=0
            "GT" => return Some((2, 0)),     // tag=2, arity=0

            _ => {}
        }

        // Check user-defined constructors registered from case alternatives
        if let Some(meta) = self.constructor_metadata.get(name) {
            return Some((meta.tag, meta.arity));
        }

        // If it looks like a constructor (starts with uppercase), return None
        // to indicate we don't have metadata yet (might be registered later)
        if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            None
        } else {
            None
        }
    }

    // ========================================================================
    // Closure Support
    // ========================================================================
    //
    // Closures are represented as heap-allocated structs:
    //
    //   struct Closure {
    //       ptr fn_ptr;        // Pointer to the lifted function
    //       i64 env_size;      // Number of captured variables
    //       ptr env[];         // Array of captured variable pointers
    //   }
    //
    // When a lambda captures free variables, we:
    // 1. Compute the free variables
    // 2. Create a lifted function that takes (env_ptr, params...) as arguments
    // 3. Allocate a closure struct with the function pointer and captured values
    //
    // When calling a closure:
    // 1. Load the function pointer from the closure
    // 2. Call with the closure (as env) and the arguments
    // ========================================================================

    /// Compute the free variables of an expression.
    ///
    /// Returns the set of variable IDs that are referenced but not bound
    /// within the expression.
    fn free_vars(&self, expr: &Expr) -> FxHashSet<VarId> {
        let mut free = FxHashSet::default();
        let mut bound = FxHashSet::default();
        self.collect_free_vars(expr, &mut free, &mut bound);
        free
    }

    /// Helper to collect free variables, tracking bound variables.
    fn collect_free_vars(
        &self,
        expr: &Expr,
        free: &mut FxHashSet<VarId>,
        bound: &mut FxHashSet<VarId>,
    ) {
        match expr {
            Expr::Lit(_, _, _) => {}

            Expr::Var(var, _) => {
                // A variable is free if it's not bound and not a top-level function/constructor
                if !bound.contains(&var.id)
                    && !self.functions.contains_key(&var.id)
                    && self.constructor_info(var.name.as_str()).is_none()
                    && self.primitive_op_info(var.name.as_str()).is_none()
                    && self.rts_function_id(var.name.as_str()).is_none()
                {
                    free.insert(var.id);
                }
            }

            Expr::App(func, arg, _) => {
                self.collect_free_vars(func, free, bound);
                self.collect_free_vars(arg, free, bound);
            }

            Expr::Lam(param, body, _) => {
                bound.insert(param.id);
                self.collect_free_vars(body, free, bound);
                bound.remove(&param.id);
            }

            Expr::Let(bind, body, _) => {
                match &**bind {
                    Bind::NonRec(var, rhs) => {
                        self.collect_free_vars(rhs, free, bound);
                        bound.insert(var.id);
                        self.collect_free_vars(body, free, bound);
                        bound.remove(&var.id);
                    }
                    Bind::Rec(bindings) => {
                        // For recursive bindings, all vars are bound in both RHS and body
                        for (var, _) in bindings {
                            bound.insert(var.id);
                        }
                        for (_, rhs) in bindings {
                            self.collect_free_vars(rhs, free, bound);
                        }
                        self.collect_free_vars(body, free, bound);
                        for (var, _) in bindings {
                            bound.remove(&var.id);
                        }
                    }
                }
            }

            Expr::Case(scrut, alts, _, _) => {
                self.collect_free_vars(scrut, free, bound);
                for alt in alts {
                    for binder in &alt.binders {
                        bound.insert(binder.id);
                    }
                    self.collect_free_vars(&alt.rhs, free, bound);
                    for binder in &alt.binders {
                        bound.remove(&binder.id);
                    }
                }
            }

            Expr::TyApp(inner, _, _) => {
                self.collect_free_vars(inner, free, bound);
            }

            Expr::TyLam(_, body, _) => {
                self.collect_free_vars(body, free, bound);
            }

            Expr::Lazy(inner, _) => {
                self.collect_free_vars(inner, free, bound);
            }

            Expr::Cast(inner, _, _) => {
                self.collect_free_vars(inner, free, bound);
            }

            Expr::Tick(_, inner, _) => {
                self.collect_free_vars(inner, free, bound);
            }

            Expr::Type(_, _) | Expr::Coercion(_, _) => {}
        }
    }

    /// Get the LLVM struct type for a closure with the given environment size.
    ///
    /// Closure layout: { ptr fn_ptr, i64 env_size, [env_size x ptr] env }
    fn closure_type(&self, env_size: u32) -> inkwell::types::StructType<'ctx> {
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let i64_type = tm.i64_type();

        // Create array type for environment
        let env_type = ptr_type.array_type(env_size);

        // Struct: { ptr fn_ptr, i64 env_size, [env_size x ptr] env }
        self.llvm_ctx.struct_type(&[ptr_type.into(), i64_type.into(), env_type.into()], false)
    }

    /// Allocate a closure with the given function pointer and captured variables.
    fn alloc_closure(
        &self,
        fn_ptr: PointerValue<'ctx>,
        captured_vars: &[(VarId, BasicValueEnum<'ctx>)],
    ) -> CodegenResult<PointerValue<'ctx>> {
        let tm = self.type_mapper();
        let env_size = captured_vars.len() as u32;
        let closure_ty = self.closure_type(env_size);

        // Calculate size: sizeof(ptr) + sizeof(i64) + env_size * sizeof(ptr)
        let size = 8 + 8 + (env_size as u64) * 8;

        // Call bhc_alloc
        let alloc_fn = self.functions.get(&VarId::new(1005)).ok_or_else(|| {
            CodegenError::Internal("bhc_alloc not declared".to_string())
        })?;

        let size_val = tm.i64_type().const_int(size, false);
        let raw_ptr = self
            .builder()
            .build_call(*alloc_fn, &[size_val.into()], "closure_alloc")
            .map_err(|e| CodegenError::Internal(format!("failed to call bhc_alloc: {:?}", e)))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CodegenError::Internal("bhc_alloc returned void".to_string()))?;

        let closure_ptr = raw_ptr.into_pointer_value();

        // Store function pointer at offset 0
        let fn_ptr_slot = self
            .builder()
            .build_struct_gep(closure_ty, closure_ptr, 0, "fn_ptr_slot")
            .map_err(|e| CodegenError::Internal(format!("failed to get fn_ptr slot: {:?}", e)))?;

        self.builder()
            .build_store(fn_ptr_slot, fn_ptr)
            .map_err(|e| CodegenError::Internal(format!("failed to store fn_ptr: {:?}", e)))?;

        // Store env_size at offset 1
        let env_size_slot = self
            .builder()
            .build_struct_gep(closure_ty, closure_ptr, 1, "env_size_slot")
            .map_err(|e| CodegenError::Internal(format!("failed to get env_size slot: {:?}", e)))?;

        let env_size_val = tm.i64_type().const_int(env_size as u64, false);
        self.builder()
            .build_store(env_size_slot, env_size_val)
            .map_err(|e| CodegenError::Internal(format!("failed to store env_size: {:?}", e)))?;

        // Store captured variables in environment array
        if env_size > 0 {
            let env_slot = self
                .builder()
                .build_struct_gep(closure_ty, closure_ptr, 2, "env_slot")
                .map_err(|e| CodegenError::Internal(format!("failed to get env slot: {:?}", e)))?;

            for (i, (_var_id, val)) in captured_vars.iter().enumerate() {
                let elem_ptr = unsafe {
                    self.builder()
                        .build_in_bounds_gep(
                            tm.ptr_type().array_type(env_size),
                            env_slot,
                            &[
                                tm.i64_type().const_zero(),
                                tm.i64_type().const_int(i as u64, false),
                            ],
                            &format!("env_{}", i),
                        )
                        .map_err(|e| CodegenError::Internal(format!("failed to get env elem: {:?}", e)))?
                };

                // Convert value to pointer if needed
                let ptr_val = self.value_to_ptr(*val)?;
                self.builder()
                    .build_store(elem_ptr, ptr_val)
                    .map_err(|e| CodegenError::Internal(format!("failed to store env elem: {:?}", e)))?;
            }
        }

        Ok(closure_ptr)
    }

    /// Extract the function pointer from a closure.
    fn extract_closure_fn_ptr(&self, closure_ptr: PointerValue<'ctx>) -> CodegenResult<PointerValue<'ctx>> {
        let closure_ty = self.closure_type(0); // Use 0-size for reading fn_ptr (same offset)

        let fn_ptr_slot = self
            .builder()
            .build_struct_gep(closure_ty, closure_ptr, 0, "fn_ptr_slot")
            .map_err(|e| CodegenError::Internal(format!("failed to get fn_ptr slot: {:?}", e)))?;

        let fn_ptr = self
            .builder()
            .build_load(self.type_mapper().ptr_type(), fn_ptr_slot, "fn_ptr")
            .map_err(|e| CodegenError::Internal(format!("failed to load fn_ptr: {:?}", e)))?;

        Ok(fn_ptr.into_pointer_value())
    }

    /// Extract an element from a closure's environment.
    fn extract_closure_env_elem(
        &self,
        closure_ptr: PointerValue<'ctx>,
        env_size: u32,
        index: u32,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let tm = self.type_mapper();
        let closure_ty = self.closure_type(env_size);

        let env_slot = self
            .builder()
            .build_struct_gep(closure_ty, closure_ptr, 2, "env_slot")
            .map_err(|e| CodegenError::Internal(format!("failed to get env slot: {:?}", e)))?;

        let elem_ptr = unsafe {
            self.builder()
                .build_in_bounds_gep(
                    tm.ptr_type().array_type(env_size),
                    env_slot,
                    &[
                        tm.i64_type().const_zero(),
                        tm.i64_type().const_int(index as u64, false),
                    ],
                    &format!("env_elem_ptr_{}", index),
                )
                .map_err(|e| CodegenError::Internal(format!("failed to get env elem ptr: {:?}", e)))?
        };

        let elem_val = self
            .builder()
            .build_load(tm.ptr_type(), elem_ptr, &format!("env_elem_{}", index))
            .map_err(|e| CodegenError::Internal(format!("failed to load env elem: {:?}", e)))?;

        Ok(elem_val.into_pointer_value())
    }

    /// Generate a unique name for a closure function.
    fn next_closure_name(&mut self) -> String {
        let name = format!("__closure_{}", self.closure_counter);
        self.closure_counter += 1;
        name
    }

    /// Check if an expression is a saturated constructor application.
    ///
    /// Returns Some((tag, collected_args)) if the expression is a fully-applied constructor.
    fn is_saturated_constructor<'a>(
        &self,
        expr: &'a Expr,
    ) -> Option<(u32, u32, Vec<&'a Expr>)> {
        // Collect arguments while unwrapping applications
        let mut args = Vec::new();
        let mut current = expr;

        while let Expr::App(func, arg, _) = current {
            args.push(arg.as_ref());
            current = func.as_ref();
        }

        // Check if the head is a constructor
        if let Expr::Var(var, _) = current {
            if let Some((tag, arity)) = self.constructor_info(var.name.as_str()) {
                args.reverse();
                if args.len() == arity as usize {
                    return Some((tag, arity, args));
                }
            }
        }

        None
    }

    /// Lower a Core module to LLVM IR.
    pub fn lower_module(&mut self, core_module: &CoreModule) -> CodegenResult<()> {
        // Pre-pass: collect all constructor metadata from case alternatives
        // This ensures constructors are known before we try to lower applications
        for bind in &core_module.bindings {
            self.collect_constructors_from_binding(bind);
        }

        // First pass: declare all top-level functions
        for bind in &core_module.bindings {
            self.declare_binding(bind)?;
        }

        // Second pass: define all functions
        for bind in &core_module.bindings {
            self.lower_binding(bind)?;
        }

        Ok(())
    }

    /// Collect constructor metadata from a binding's expression.
    fn collect_constructors_from_binding(&mut self, bind: &Bind) {
        match bind {
            Bind::NonRec(_, expr) => {
                self.collect_constructors_from_expr(expr);
            }
            Bind::Rec(bindings) => {
                for (_, expr) in bindings {
                    self.collect_constructors_from_expr(expr);
                }
            }
        }
    }

    /// Recursively collect constructor metadata from an expression.
    fn collect_constructors_from_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Case(scrut, alts, _, _) => {
                self.collect_constructors_from_expr(scrut);
                for alt in alts {
                    if let AltCon::DataCon(con) = &alt.con {
                        self.register_constructor(con.name.as_str(), con.tag, con.arity);
                    }
                    self.collect_constructors_from_expr(&alt.rhs);
                }
            }
            Expr::App(func, arg, _) => {
                self.collect_constructors_from_expr(func);
                self.collect_constructors_from_expr(arg);
            }
            Expr::TyApp(func, _, _) => {
                self.collect_constructors_from_expr(func);
            }
            Expr::Lam(_, body, _) => {
                self.collect_constructors_from_expr(body);
            }
            Expr::TyLam(_, body, _) => {
                self.collect_constructors_from_expr(body);
            }
            Expr::Let(bind, body, _) => {
                self.collect_constructors_from_binding(bind);
                self.collect_constructors_from_expr(body);
            }
            Expr::Lazy(inner, _) => {
                self.collect_constructors_from_expr(inner);
            }
            Expr::Cast(inner, _, _) => {
                self.collect_constructors_from_expr(inner);
            }
            Expr::Tick(_, inner, _) => {
                self.collect_constructors_from_expr(inner);
            }
            // Leaf nodes that don't contain constructors
            Expr::Var(_, _) | Expr::Lit(_, _, _) | Expr::Type(_, _) | Expr::Coercion(_, _) => {}
        }
    }

    /// Declare a binding (creates function signature without body).
    fn declare_binding(&mut self, bind: &Bind) -> CodegenResult<()> {
        match bind {
            Bind::NonRec(var, expr) => {
                let fn_val = self.declare_function_from_expr(var, expr)?;
                self.functions.insert(var.id, fn_val);
            }
            Bind::Rec(bindings) => {
                for (var, expr) in bindings {
                    let fn_val = self.declare_function_from_expr(var, expr)?;
                    self.functions.insert(var.id, fn_val);
                }
            }
        }
        Ok(())
    }

    /// Count the number of lambda parameters in an expression.
    fn count_lambda_params(&self, expr: &Expr) -> usize {
        let mut count = 0;
        let mut current = expr;
        while let Expr::Lam(_, body, _) = current {
            count += 1;
            current = body.as_ref();
        }
        count
    }

    /// Check if a type contains unresolved type variables or errors.
    fn type_needs_inference(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Error => true,
            Ty::Var(_) => true,
            Ty::Fun(arg, ret) => self.type_needs_inference(arg) || self.type_needs_inference(ret),
            Ty::Forall(_, body) => self.type_needs_inference(body),
            Ty::App(f, arg) => self.type_needs_inference(f) || self.type_needs_inference(arg),
            _ => false,
        }
    }

    /// Declare a function from a Core variable and expression.
    /// Uses the expression to infer the function type when var.ty contains unresolved types.
    fn declare_function_from_expr(&self, var: &Var, expr: &Expr) -> CodegenResult<FunctionValue<'ctx>> {
        let param_count = self.count_lambda_params(expr);

        // Only infer types for functions with parameters
        // For CAFs (0 parameters), use the original type lowering
        let fn_type = if param_count > 0 && self.type_needs_inference(&var.ty) {
            // Infer function type from the expression structure
            let tm = self.type_mapper();

            // Use i64 as default parameter type (most common for numeric code)
            let param_types: Vec<_> = (0..param_count)
                .map(|_| tm.i64_type().into())
                .collect();

            // Default return type is i64
            tm.i64_type().fn_type(&param_types, false)
        } else {
            self.lower_function_type(&var.ty)?
        };
        let name = var.name.as_str();
        Ok(self.module.add_function(name, fn_type))
    }

    /// Declare a function from a Core variable.
    fn declare_function(&self, var: &Var) -> CodegenResult<FunctionValue<'ctx>> {
        let fn_type = self.lower_function_type(&var.ty)?;
        let name = var.name.as_str();
        Ok(self.module.add_function(name, fn_type))
    }

    /// Lower a binding to LLVM IR.
    fn lower_binding(&mut self, bind: &Bind) -> CodegenResult<()> {
        match bind {
            Bind::NonRec(var, expr) => {
                self.lower_function_def(var, expr)?;
            }
            Bind::Rec(bindings) => {
                for (var, expr) in bindings {
                    self.lower_function_def(var, expr)?;
                }
            }
        }
        Ok(())
    }

    /// Lower a function definition.
    fn lower_function_def(&mut self, var: &Var, expr: &Expr) -> CodegenResult<()> {
        eprintln!("[DEBUG] lower_function_def: {}", var.name.as_str());
        let fn_val = self.functions.get(&var.id).copied().ok_or_else(|| {
            CodegenError::Internal(format!("function not declared: {}", var.name.as_str()))
        })?;

        // Create entry block
        let entry = self.llvm_context().append_basic_block(fn_val, "entry");
        self.builder().position_at_end(entry);

        // Lower the function body, handling lambda parameters
        let result = self.lower_function_body(fn_val, expr)?;

        // Build return based on function's declared return type, not the computed result
        // This handles cases like IO () which produces a value but should return void
        let ret_type = fn_val.get_type().get_return_type();
        if ret_type.is_none() {
            // Void return type - don't return a value
            self.builder()
                .build_return(None)
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
        } else if let Some(val) = result {
            self.builder()
                .build_return(Some(&val))
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
        } else {
            // Function expects a return value but body produced none
            // This shouldn't happen with correct type checking
            self.builder()
                .build_return(None)
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
        }

        Ok(())
    }

    /// Lower a Core expression to LLVM IR.
    fn lower_expr(&mut self, expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        match expr {
            Expr::Lit(lit, _ty, _span) => self.lower_literal(lit).map(Some),

            Expr::Var(var, _span) => {
                let name = var.name.as_str();

                // First, check if this is a nullary constructor (like [], True, False, Nothing)
                if let Some((tag, arity)) = self.constructor_info(name) {
                    if arity == 0 {
                        // Nullary constructor - allocate ADT value with just a tag
                        return self.lower_constructor_application(tag, 0, &[]);
                    }
                    // Non-nullary constructor referenced without args - return as closure/function
                    // This case is handled below when the constructor is used in an application
                }

                // Look up the variable in the environment
                if let Some(val) = self.env.get(&var.id) {
                    Ok(Some(*val))
                } else if let Some(fn_val) = self.functions.get(&var.id) {
                    // It's a function reference
                    // Check if it's a CAF (zero-argument function)
                    let fn_type = fn_val.get_type();
                    if fn_type.count_param_types() == 0 {
                        // CAF - call the function to get its value
                        let call_result = self.builder()
                            .build_call(*fn_val, &[], "caf_result")
                            .map_err(|e| CodegenError::Internal(format!("failed to call CAF: {:?}", e)))?;
                        // Get the return value
                        if let Some(ret_val) = call_result.try_as_basic_value().left() {
                            Ok(Some(ret_val))
                        } else {
                            // Void function - shouldn't happen for CAFs
                            Ok(None)
                        }
                    } else {
                        // Function with parameters - return as pointer for potential partial application
                        Ok(Some(fn_val.as_global_value().as_pointer_value().into()))
                    }
                } else {
                    Err(CodegenError::Internal(format!(
                        "unbound variable: {}",
                        name
                    )))
                }
            }

            Expr::App(func, arg, _span) => self.lower_application(func, arg),

            Expr::Lam(param, body, _span) => {
                // Create a closure for this lambda
                self.lower_lambda(param, body)
            }

            Expr::Let(bind, body, _span) => self.lower_let(bind, body),

            Expr::Case(scrut, alts, _ty, _span) => self.lower_case(scrut, alts),

            Expr::TyApp(expr, _ty, _span) => {
                // Type applications are erased at runtime
                self.lower_expr(expr)
            }

            Expr::TyLam(_tyvar, body, _span) => {
                // Type lambdas are erased at runtime
                self.lower_expr(body)
            }

            Expr::Lazy(inner, _span) => {
                // For now, just evaluate eagerly
                // Full thunk support requires RTS integration
                self.lower_expr(inner)
            }

            Expr::Cast(inner, _coercion, _span) => {
                // Coercions are erased at runtime
                self.lower_expr(inner)
            }

            Expr::Tick(_tick, inner, _span) => {
                // Ticks are for profiling, skip for now
                self.lower_expr(inner)
            }

            Expr::Type(_ty, _span) => {
                // Type expressions have no runtime value
                Ok(None)
            }

            Expr::Coercion(_coercion, _span) => {
                // Coercion values have no runtime representation
                Ok(None)
            }
        }
    }

    /// Lower a literal to LLVM IR.
    fn lower_literal(&self, lit: &Literal) -> CodegenResult<BasicValueEnum<'ctx>> {
        let tm = self.type_mapper();
        match lit {
            Literal::Int(n) => Ok(tm.i64_type().const_int(*n as u64, true).into()),

            Literal::Integer(n) => {
                // For large integers, we'd need arbitrary precision
                // For now, truncate to i64
                Ok(tm.i64_type().const_int(*n as u64, true).into())
            }

            Literal::Float(f) => Ok(tm.f32_type().const_float(*f as f64).into()),

            Literal::Double(d) => Ok(tm.f64_type().const_float(*d).into()),

            Literal::Char(c) => Ok(tm.i32_type().const_int(*c as u64, false).into()),

            Literal::String(sym) => {
                let s = sym.as_str();
                let ptr = self.module.add_global_string(&format!("str_{}", sym.as_u32()), s);
                Ok(ptr.into())
            }
        }
    }

    /// Lower a lambda expression to a closure.
    ///
    /// Creates a lifted function and allocates a closure struct containing
    /// the function pointer and captured variables.
    fn lower_lambda(
        &mut self,
        param: &Var,
        body: &Expr,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Collect all parameters (for multi-argument lambdas)
        let mut params = vec![param.clone()];
        let mut current_body = body;

        while let Expr::Lam(next_param, next_body, _) = current_body {
            params.push(next_param.clone());
            current_body = next_body;
        }

        // Compute free variables in the lambda body
        let full_lambda = Expr::Lam(
            param.clone(),
            Box::new(body.clone()),
            body.span(),
        );
        let free = self.free_vars(&full_lambda);

        // Collect the values of free variables from current environment
        let mut captured: Vec<(VarId, BasicValueEnum<'ctx>)> = Vec::new();
        for var_id in &free {
            if let Some(val) = self.env.get(var_id) {
                captured.push((*var_id, *val));
            }
        }

        // Save current insertion point
        let current_block = self.builder().get_insert_block();

        // Generate unique name for the lifted function
        let fn_name = self.next_closure_name();

        // Build the function type:
        // - First param is the closure/env pointer
        // - Remaining params are the lambda parameters
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();

        let mut param_types: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> = Vec::new();
        param_types.push(ptr_type.into()); // env/closure pointer

        for _ in &params {
            // For now, use ptr for all params (polymorphic)
            param_types.push(ptr_type.into());
        }

        // Return type - use ptr for now (polymorphic)
        let fn_type = ptr_type.fn_type(&param_types, false);

        // Create the lifted function
        let lifted_fn = self.module.add_function(&fn_name, fn_type);

        // Create entry block for the lifted function
        let entry = self.llvm_context().append_basic_block(lifted_fn, "entry");
        self.builder().position_at_end(entry);

        // Save old environment and create new scope
        let old_env = std::mem::take(&mut self.env);

        // Bind captured variables from environment
        // The closure pointer is the first argument
        if !captured.is_empty() {
            let closure_ptr = lifted_fn.get_first_param()
                .ok_or_else(|| CodegenError::Internal("missing closure param".to_string()))?
                .into_pointer_value();

            for (i, (var_id, _)) in captured.iter().enumerate() {
                let elem_ptr = self.extract_closure_env_elem(
                    closure_ptr,
                    captured.len() as u32,
                    i as u32,
                )?;
                // Store as pointer - will be unboxed when used
                self.env.insert(*var_id, elem_ptr.into());
            }
        }

        // Bind lambda parameters to function arguments
        // Skip first arg (closure pointer)
        for (i, lam_param) in params.iter().enumerate() {
            if let Some(arg) = lifted_fn.get_nth_param((i + 1) as u32) {
                self.env.insert(lam_param.id, arg);
            }
        }

        // Lower the body
        let result = self.lower_expr(current_body)?;

        // Build return
        if let Some(val) = result {
            // Convert to pointer for return
            let ret_ptr = self.value_to_ptr(val)?;
            self.builder()
                .build_return(Some(&ret_ptr))
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
        } else {
            // Return null pointer for unit/void
            let null = ptr_type.const_null();
            self.builder()
                .build_return(Some(&null))
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
        }

        // Restore old environment
        self.env = old_env;

        // Restore insertion point
        if let Some(block) = current_block {
            self.builder().position_at_end(block);
        }

        // Allocate closure with function pointer and captured values
        let fn_ptr = lifted_fn.as_global_value().as_pointer_value();
        let closure_ptr = self.alloc_closure(fn_ptr, &captured)?;

        Ok(Some(closure_ptr.into()))
    }

    // ========================================================================
    // Primitive Operations
    // ========================================================================

    /// Check if a name is a primitive operation and return its arity.
    fn primitive_op_info(&self, name: &str) -> Option<(PrimOp, u32)> {
        match name {
            // Arithmetic (binary)
            "+" | "GHC.Num.+" => Some((PrimOp::Add, 2)),
            "-" | "GHC.Num.-" => Some((PrimOp::Sub, 2)),
            "*" | "GHC.Num.*" => Some((PrimOp::Mul, 2)),
            "/" | "GHC.Real./" => Some((PrimOp::Div, 2)),
            "div" | "GHC.Real.div" => Some((PrimOp::Div, 2)),
            "mod" | "GHC.Real.mod" => Some((PrimOp::Mod, 2)),
            "rem" | "GHC.Real.rem" => Some((PrimOp::Rem, 2)),
            "quot" | "GHC.Real.quot" => Some((PrimOp::Quot, 2)),

            // Comparison (binary)
            "==" | "GHC.Classes.==" => Some((PrimOp::Eq, 2)),
            "/=" | "GHC.Classes./=" => Some((PrimOp::Ne, 2)),
            "<" | "GHC.Classes.<" => Some((PrimOp::Lt, 2)),
            "<=" | "GHC.Classes.<=" => Some((PrimOp::Le, 2)),
            ">" | "GHC.Classes.>" => Some((PrimOp::Gt, 2)),
            ">=" | "GHC.Classes.>=" => Some((PrimOp::Ge, 2)),

            // Boolean (binary)
            "&&" | "GHC.Classes.&&" => Some((PrimOp::And, 2)),
            "||" | "GHC.Classes.||" => Some((PrimOp::Or, 2)),

            // Unary
            "negate" | "GHC.Num.negate" => Some((PrimOp::Negate, 1)),
            "abs" | "GHC.Num.abs" => Some((PrimOp::Abs, 1)),
            "signum" | "GHC.Num.signum" => Some((PrimOp::Signum, 1)),
            "not" | "GHC.Classes.not" => Some((PrimOp::Not, 1)),

            // Bitwise (binary)
            ".&." => Some((PrimOp::BitAnd, 2)),
            ".|." => Some((PrimOp::BitOr, 2)),
            "xor" => Some((PrimOp::BitXor, 2)),
            "shiftL" => Some((PrimOp::ShiftL, 2)),
            "shiftR" => Some((PrimOp::ShiftR, 2)),

            // Bitwise (unary)
            "complement" => Some((PrimOp::Complement, 1)),

            _ => None,
        }
    }

    /// Check if an expression is a saturated primitive operation.
    fn is_saturated_primop<'a>(&self, expr: &'a Expr) -> Option<(PrimOp, Vec<&'a Expr>)> {
        // Collect arguments while unwrapping applications
        let mut args = Vec::new();
        let mut current = expr;

        while let Expr::App(func, arg, _) = current {
            args.push(arg.as_ref());
            current = func.as_ref();
        }

        // Check if the head is a primitive operation
        if let Expr::Var(var, _) = current {
            if let Some((op, arity)) = self.primitive_op_info(var.name.as_str()) {
                args.reverse();
                if args.len() == arity as usize {
                    return Some((op, args));
                }
            }
        }

        None
    }

    /// Lower a primitive operation to LLVM instructions.
    fn lower_primop(
        &mut self,
        op: PrimOp,
        args: &[&Expr],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        match op {
            // Binary arithmetic operations
            PrimOp::Add | PrimOp::Sub | PrimOp::Mul | PrimOp::Div |
            PrimOp::Mod | PrimOp::Rem | PrimOp::Quot => {
                let lhs = self.lower_expr(args[0])?.ok_or_else(|| {
                    CodegenError::Internal("primop arg has no value".to_string())
                })?;
                let rhs = self.lower_expr(args[1])?.ok_or_else(|| {
                    CodegenError::Internal("primop arg has no value".to_string())
                })?;
                self.lower_binary_arith(op, lhs, rhs)
            }

            // Binary comparison operations
            PrimOp::Eq | PrimOp::Ne | PrimOp::Lt | PrimOp::Le |
            PrimOp::Gt | PrimOp::Ge => {
                let lhs = self.lower_expr(args[0])?.ok_or_else(|| {
                    CodegenError::Internal("primop arg has no value".to_string())
                })?;
                let rhs = self.lower_expr(args[1])?.ok_or_else(|| {
                    CodegenError::Internal("primop arg has no value".to_string())
                })?;
                self.lower_comparison(op, lhs, rhs)
            }

            // Binary boolean operations
            PrimOp::And | PrimOp::Or => {
                let lhs = self.lower_expr(args[0])?.ok_or_else(|| {
                    CodegenError::Internal("primop arg has no value".to_string())
                })?;
                let rhs = self.lower_expr(args[1])?.ok_or_else(|| {
                    CodegenError::Internal("primop arg has no value".to_string())
                })?;
                self.lower_binary_bool(op, lhs, rhs)
            }

            // Binary bitwise operations
            PrimOp::BitAnd | PrimOp::BitOr | PrimOp::BitXor |
            PrimOp::ShiftL | PrimOp::ShiftR => {
                let lhs = self.lower_expr(args[0])?.ok_or_else(|| {
                    CodegenError::Internal("primop arg has no value".to_string())
                })?;
                let rhs = self.lower_expr(args[1])?.ok_or_else(|| {
                    CodegenError::Internal("primop arg has no value".to_string())
                })?;
                self.lower_binary_bitwise(op, lhs, rhs)
            }

            // Unary operations
            PrimOp::Negate | PrimOp::Abs | PrimOp::Signum |
            PrimOp::Not | PrimOp::Complement => {
                let arg = self.lower_expr(args[0])?.ok_or_else(|| {
                    CodegenError::Internal("primop arg has no value".to_string())
                })?;
                self.lower_unary(op, arg)
            }
        }
    }

    /// Lower a binary arithmetic operation.
    fn lower_binary_arith(
        &self,
        op: PrimOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Check if we're dealing with integers or floats
        match (lhs, rhs) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                let result = match op {
                    PrimOp::Add => self.builder().build_int_add(l, r, "add"),
                    PrimOp::Sub => self.builder().build_int_sub(l, r, "sub"),
                    PrimOp::Mul => self.builder().build_int_mul(l, r, "mul"),
                    PrimOp::Div | PrimOp::Quot => self.builder().build_int_signed_div(l, r, "div"),
                    PrimOp::Mod | PrimOp::Rem => self.builder().build_int_signed_rem(l, r, "rem"),
                    _ => return Err(CodegenError::Internal("invalid arith op".to_string())),
                };
                result
                    .map(|v| Some(v.into()))
                    .map_err(|e| CodegenError::Internal(format!("failed to build int op: {:?}", e)))
            }
            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                let result = match op {
                    PrimOp::Add => self.builder().build_float_add(l, r, "fadd"),
                    PrimOp::Sub => self.builder().build_float_sub(l, r, "fsub"),
                    PrimOp::Mul => self.builder().build_float_mul(l, r, "fmul"),
                    PrimOp::Div | PrimOp::Quot => self.builder().build_float_div(l, r, "fdiv"),
                    PrimOp::Mod | PrimOp::Rem => self.builder().build_float_rem(l, r, "frem"),
                    _ => return Err(CodegenError::Internal("invalid arith op".to_string())),
                };
                result
                    .map(|v| Some(v.into()))
                    .map_err(|e| CodegenError::Internal(format!("failed to build float op: {:?}", e)))
            }
            _ => Err(CodegenError::TypeError(
                "arithmetic operations require matching numeric types".to_string(),
            )),
        }
    }

    /// Lower a comparison operation.
    fn lower_comparison(
        &self,
        op: PrimOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        match (lhs, rhs) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                use inkwell::IntPredicate;
                let pred = match op {
                    PrimOp::Eq => IntPredicate::EQ,
                    PrimOp::Ne => IntPredicate::NE,
                    PrimOp::Lt => IntPredicate::SLT,
                    PrimOp::Le => IntPredicate::SLE,
                    PrimOp::Gt => IntPredicate::SGT,
                    PrimOp::Ge => IntPredicate::SGE,
                    _ => return Err(CodegenError::Internal("invalid comparison op".to_string())),
                };
                let cmp = self.builder()
                    .build_int_compare(pred, l, r, "cmp")
                    .map_err(|e| CodegenError::Internal(format!("failed to build int cmp: {:?}", e)))?;

                // Convert i1 to i64 (0 or 1) for consistency with our Bool representation
                let result = self.builder()
                    .build_int_z_extend(cmp, self.type_mapper().i64_type(), "cmp_ext")
                    .map_err(|e| CodegenError::Internal(format!("failed to extend cmp: {:?}", e)))?;

                Ok(Some(result.into()))
            }
            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                use inkwell::FloatPredicate;
                let pred = match op {
                    PrimOp::Eq => FloatPredicate::OEQ,
                    PrimOp::Ne => FloatPredicate::ONE,
                    PrimOp::Lt => FloatPredicate::OLT,
                    PrimOp::Le => FloatPredicate::OLE,
                    PrimOp::Gt => FloatPredicate::OGT,
                    PrimOp::Ge => FloatPredicate::OGE,
                    _ => return Err(CodegenError::Internal("invalid comparison op".to_string())),
                };
                let cmp = self.builder()
                    .build_float_compare(pred, l, r, "fcmp")
                    .map_err(|e| CodegenError::Internal(format!("failed to build float cmp: {:?}", e)))?;

                // Convert i1 to i64
                let result = self.builder()
                    .build_int_z_extend(cmp, self.type_mapper().i64_type(), "fcmp_ext")
                    .map_err(|e| CodegenError::Internal(format!("failed to extend fcmp: {:?}", e)))?;

                Ok(Some(result.into()))
            }
            _ => Err(CodegenError::TypeError(
                "comparison operations require matching types".to_string(),
            )),
        }
    }

    /// Lower a binary boolean operation.
    fn lower_binary_bool(
        &self,
        op: PrimOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Convert to i64 if needed, then perform operation
        let l = self.to_int_value(lhs)?;
        let r = self.to_int_value(rhs)?;

        // Convert to i1 for boolean operations
        let zero = self.type_mapper().i64_type().const_zero();
        let l_bool = self.builder()
            .build_int_compare(inkwell::IntPredicate::NE, l, zero, "l_bool")
            .map_err(|e| CodegenError::Internal(format!("failed to convert to bool: {:?}", e)))?;
        let r_bool = self.builder()
            .build_int_compare(inkwell::IntPredicate::NE, r, zero, "r_bool")
            .map_err(|e| CodegenError::Internal(format!("failed to convert to bool: {:?}", e)))?;

        let result = match op {
            PrimOp::And => self.builder().build_and(l_bool, r_bool, "and"),
            PrimOp::Or => self.builder().build_or(l_bool, r_bool, "or"),
            _ => return Err(CodegenError::Internal("invalid bool op".to_string())),
        };

        let bool_result = result
            .map_err(|e| CodegenError::Internal(format!("failed to build bool op: {:?}", e)))?;

        // Extend back to i64
        let extended = self.builder()
            .build_int_z_extend(bool_result, self.type_mapper().i64_type(), "bool_ext")
            .map_err(|e| CodegenError::Internal(format!("failed to extend bool: {:?}", e)))?;

        Ok(Some(extended.into()))
    }

    /// Lower a binary bitwise operation.
    fn lower_binary_bitwise(
        &self,
        op: PrimOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let l = self.to_int_value(lhs)?;
        let r = self.to_int_value(rhs)?;

        let result = match op {
            PrimOp::BitAnd => self.builder().build_and(l, r, "band"),
            PrimOp::BitOr => self.builder().build_or(l, r, "bor"),
            PrimOp::BitXor => self.builder().build_xor(l, r, "bxor"),
            PrimOp::ShiftL => self.builder().build_left_shift(l, r, "shl"),
            PrimOp::ShiftR => self.builder().build_right_shift(l, r, true, "shr"),
            _ => return Err(CodegenError::Internal("invalid bitwise op".to_string())),
        };

        result
            .map(|v| Some(v.into()))
            .map_err(|e| CodegenError::Internal(format!("failed to build bitwise op: {:?}", e)))
    }

    /// Lower a unary operation.
    fn lower_unary(
        &self,
        op: PrimOp,
        arg: BasicValueEnum<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        match op {
            PrimOp::Negate => {
                match arg {
                    BasicValueEnum::IntValue(i) => {
                        let result = self.builder()
                            .build_int_neg(i, "neg")
                            .map_err(|e| CodegenError::Internal(format!("failed to negate: {:?}", e)))?;
                        Ok(Some(result.into()))
                    }
                    BasicValueEnum::FloatValue(f) => {
                        let result = self.builder()
                            .build_float_neg(f, "fneg")
                            .map_err(|e| CodegenError::Internal(format!("failed to fnegate: {:?}", e)))?;
                        Ok(Some(result.into()))
                    }
                    _ => Err(CodegenError::TypeError("negate requires numeric type".to_string())),
                }
            }
            PrimOp::Abs => {
                match arg {
                    BasicValueEnum::IntValue(i) => {
                        // abs(x) = x < 0 ? -x : x
                        let zero = self.type_mapper().i64_type().const_zero();
                        let is_neg = self.builder()
                            .build_int_compare(inkwell::IntPredicate::SLT, i, zero, "is_neg")
                            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
                        let neg = self.builder()
                            .build_int_neg(i, "neg")
                            .map_err(|e| CodegenError::Internal(format!("failed to negate: {:?}", e)))?;
                        let result = self.builder()
                            .build_select(is_neg, neg, i, "abs")
                            .map_err(|e| CodegenError::Internal(format!("failed to select: {:?}", e)))?;
                        Ok(Some(result))
                    }
                    BasicValueEnum::FloatValue(f) => {
                        // For floats, use llvm.fabs intrinsic or manual comparison
                        let zero = self.type_mapper().f64_type().const_zero();
                        let is_neg = self.builder()
                            .build_float_compare(inkwell::FloatPredicate::OLT, f, zero, "is_neg")
                            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
                        let neg = self.builder()
                            .build_float_neg(f, "fneg")
                            .map_err(|e| CodegenError::Internal(format!("failed to fnegate: {:?}", e)))?;
                        let result = self.builder()
                            .build_select(is_neg, neg, f, "fabs")
                            .map_err(|e| CodegenError::Internal(format!("failed to select: {:?}", e)))?;
                        Ok(Some(result))
                    }
                    _ => Err(CodegenError::TypeError("abs requires numeric type".to_string())),
                }
            }
            PrimOp::Signum => {
                match arg {
                    BasicValueEnum::IntValue(i) => {
                        // signum(x) = x < 0 ? -1 : (x > 0 ? 1 : 0)
                        let zero = self.type_mapper().i64_type().const_zero();
                        let one = self.type_mapper().i64_type().const_int(1, false);
                        let neg_one = self.type_mapper().i64_type().const_int(-1i64 as u64, true);

                        let is_neg = self.builder()
                            .build_int_compare(inkwell::IntPredicate::SLT, i, zero, "is_neg")
                            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
                        let is_pos = self.builder()
                            .build_int_compare(inkwell::IntPredicate::SGT, i, zero, "is_pos")
                            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

                        let pos_or_zero = self.builder()
                            .build_select(is_pos, one, zero, "pos_or_zero")
                            .map_err(|e| CodegenError::Internal(format!("failed to select: {:?}", e)))?;
                        let result = self.builder()
                            .build_select(is_neg, neg_one, pos_or_zero.into_int_value(), "signum")
                            .map_err(|e| CodegenError::Internal(format!("failed to select: {:?}", e)))?;

                        Ok(Some(result))
                    }
                    BasicValueEnum::FloatValue(f) => {
                        let zero = f.get_type().const_zero();
                        let one = f.get_type().const_float(1.0);
                        let neg_one = f.get_type().const_float(-1.0);

                        let is_neg = self.builder()
                            .build_float_compare(inkwell::FloatPredicate::OLT, f, zero, "is_neg")
                            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
                        let is_pos = self.builder()
                            .build_float_compare(inkwell::FloatPredicate::OGT, f, zero, "is_pos")
                            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

                        let pos_or_zero = self.builder()
                            .build_select(is_pos, one, zero, "pos_or_zero")
                            .map_err(|e| CodegenError::Internal(format!("failed to select: {:?}", e)))?;
                        let result = self.builder()
                            .build_select(is_neg, neg_one, pos_or_zero.into_float_value(), "fsignum")
                            .map_err(|e| CodegenError::Internal(format!("failed to select: {:?}", e)))?;

                        Ok(Some(result))
                    }
                    _ => Err(CodegenError::TypeError("signum requires numeric type".to_string())),
                }
            }
            PrimOp::Not => {
                let i = self.to_int_value(arg)?;
                let zero = self.type_mapper().i64_type().const_zero();
                let one = self.type_mapper().i64_type().const_int(1, false);

                // not x = x == 0 ? 1 : 0
                let is_zero = self.builder()
                    .build_int_compare(inkwell::IntPredicate::EQ, i, zero, "is_zero")
                    .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
                let result = self.builder()
                    .build_select(is_zero, one, zero, "not")
                    .map_err(|e| CodegenError::Internal(format!("failed to select: {:?}", e)))?;

                Ok(Some(result))
            }
            PrimOp::Complement => {
                let i = self.to_int_value(arg)?;
                let result = self.builder()
                    .build_not(i, "complement")
                    .map_err(|e| CodegenError::Internal(format!("failed to complement: {:?}", e)))?;
                Ok(Some(result.into()))
            }
            _ => Err(CodegenError::Internal("invalid unary op".to_string())),
        }
    }

    /// Convert a basic value to an int value.
    fn to_int_value(&self, val: BasicValueEnum<'ctx>) -> CodegenResult<IntValue<'ctx>> {
        match val {
            BasicValueEnum::IntValue(i) => Ok(i),
            BasicValueEnum::PointerValue(p) => {
                // Pointer might be a boxed int - convert
                self.builder()
                    .build_ptr_to_int(p, self.type_mapper().i64_type(), "ptr_to_int")
                    .map_err(|e| CodegenError::Internal(format!("failed to convert ptr to int: {:?}", e)))
            }
            _ => Err(CodegenError::TypeError("expected integer value".to_string())),
        }
    }

    /// Lower a function application.
    ///
    /// Handles four cases:
    /// 1. Builtin function (e.g., `head xs`) - generate specialized code
    /// 2. Primitive operation (e.g., `1 + 2`) - generate LLVM instruction
    /// 3. Constructor application (e.g., `Just 42`) - allocate ADT value
    /// 4. Function call - generate call instruction
    fn lower_application(
        &mut self,
        func: &Expr,
        arg: &Expr,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // First, reconstruct the full application to check for special cases
        let full_app = Expr::App(Box::new(func.clone()), Box::new(arg.clone()), func.span());

        // Check if this is a saturated builtin function
        if let Some((name, builtin_args)) = self.is_saturated_builtin(&full_app) {
            return self.lower_builtin(name, &builtin_args);
        }

        // Check if this is a saturated primitive operation
        if let Some((op, prim_args)) = self.is_saturated_primop(&full_app) {
            return self.lower_primop(op, &prim_args);
        }

        // Check if this is a saturated constructor application
        if let Some((tag, arity, con_args)) = self.is_saturated_constructor(&full_app) {
            return self.lower_constructor_application(tag, arity, &con_args);
        }

        // Not a primop or constructor - proceed with function call
        // Collect all arguments (for curried applications)
        let mut args = vec![arg];
        let mut current = func;

        while let Expr::App(inner_func, inner_arg, _) = current {
            args.push(inner_arg);
            current = inner_func;
        }

        args.reverse();

        // Get the function being called
        match current {
            Expr::Var(var, _) => {
                let name = var.name.as_str();

                // Check if this is a nullary constructor (no args, just a value)
                if let Some((tag, arity)) = self.constructor_info(name) {
                    if arity == 0 {
                        // Nullary constructor like True, False, Nothing, ()
                        return self.lower_constructor_application(tag, 0, &[]);
                    }
                }

                // Check if this is an RTS builtin
                if let Some(rts_id) = self.rts_function_id(name) {
                    let fn_val = self.functions.get(&rts_id).copied().ok_or_else(|| {
                        CodegenError::Internal(format!("RTS function not declared: {}", name))
                    })?;
                    return self.lower_direct_call(fn_val, &args);
                }

                // Check if this is a known top-level function
                if let Some(fn_val) = self.functions.get(&var.id).copied() {
                    return self.lower_direct_call(fn_val, &args);
                }

                // Check if this is a closure in the environment
                if let Some(closure_val) = self.env.get(&var.id).copied() {
                    return self.lower_closure_call(closure_val, &args);
                }

                Err(CodegenError::Internal(format!("unknown function: {}", name)))
            }
            _ => {
                // Indirect call - evaluate the function expression
                let func_val = self.lower_expr(current)?.ok_or_else(|| {
                    CodegenError::Internal("function expression has no value".to_string())
                })?;

                // Treat as closure call
                self.lower_closure_call(func_val, &args)
            }
        }
    }

    /// Lower a direct function call (to a known function).
    fn lower_direct_call(
        &mut self,
        fn_val: FunctionValue<'ctx>,
        args: &[&Expr],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Arguments are not in tail position
        let was_tail = self.in_tail_position;
        self.in_tail_position = false;

        // Lower arguments
        let mut llvm_args = Vec::new();
        for arg_expr in args {
            if let Some(val) = self.lower_expr(arg_expr)? {
                llvm_args.push(val.into());
            }
        }

        // Restore tail position flag
        self.in_tail_position = was_tail;

        // Build the call
        let call = self
            .builder()
            .build_call(fn_val, &llvm_args, "call")
            .map_err(|e| CodegenError::Internal(format!("failed to build call: {:?}", e)))?;

        // Mark as tail call if in tail position
        if self.in_tail_position {
            call.set_tail_call(true);
        }

        Ok(call.try_as_basic_value().left())
    }

    /// Lower a closure call (indirect call through closure struct).
    fn lower_closure_call(
        &mut self,
        closure_val: BasicValueEnum<'ctx>,
        args: &[&Expr],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();

        // Get closure pointer
        let closure_ptr = match closure_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => {
                return Err(CodegenError::Internal(
                    "closure value is not a pointer".to_string(),
                ))
            }
        };

        // Extract function pointer from closure
        let fn_ptr = self.extract_closure_fn_ptr(closure_ptr)?;

        // Build function type for the call:
        // - First param is the closure pointer (environment)
        // - Remaining params are the arguments
        let mut param_types: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> = Vec::new();
        param_types.push(ptr_type.into()); // closure/env pointer

        for _ in args {
            param_types.push(ptr_type.into());
        }

        let fn_type = ptr_type.fn_type(&param_types, false);

        // Arguments are not in tail position
        let was_tail = self.in_tail_position;
        self.in_tail_position = false;

        // Lower arguments
        let mut llvm_args: Vec<inkwell::values::BasicMetadataValueEnum<'ctx>> = Vec::new();
        llvm_args.push(closure_ptr.into()); // Pass closure as first argument (environment)

        for arg_expr in args {
            if let Some(val) = self.lower_expr(arg_expr)? {
                // Convert to pointer for uniform calling convention
                let ptr_val = self.value_to_ptr(val)?;
                llvm_args.push(ptr_val.into());
            }
        }

        // Restore tail position flag
        self.in_tail_position = was_tail;

        // Build indirect call through function pointer
        let call = self
            .builder()
            .build_indirect_call(fn_type, fn_ptr, &llvm_args, "closure_call")
            .map_err(|e| CodegenError::Internal(format!("failed to build closure call: {:?}", e)))?;

        // Mark as tail call if in tail position
        if self.in_tail_position {
            call.set_tail_call(true);
        }

        Ok(call.try_as_basic_value().left())
    }

    /// Lower a constructor application to an ADT value.
    ///
    /// Allocates an ADT value with the given tag and stores the arguments as fields.
    fn lower_constructor_application(
        &mut self,
        tag: u32,
        arity: u32,
        args: &[&Expr],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Allocate the ADT value
        let adt_ptr = self.alloc_adt(tag, arity)?;

        // Store each argument as a field
        for (i, arg_expr) in args.iter().enumerate() {
            if let Some(arg_val) = self.lower_expr(arg_expr)? {
                self.store_adt_field(adt_ptr, arity, i as u32, arg_val)?;
            }
        }

        // Return the pointer to the ADT value
        Ok(Some(adt_ptr.into()))
    }

    /// Lower a let binding.
    fn lower_let(&mut self, bind: &Bind, body: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        match bind {
            Bind::NonRec(var, rhs) => {
                // Lower the right-hand side
                if let Some(val) = self.lower_expr(rhs.as_ref())? {
                    // Bind the variable
                    self.env.insert(var.id, val);
                }

                // Lower the body
                let result = self.lower_expr(body)?;

                // Remove the binding (for proper scoping)
                self.env.remove(&var.id);

                Ok(result)
            }

            Bind::Rec(bindings) => {
                // For recursive let bindings, we lift them to top-level functions.
                // This works because:
                // 1. We declare all the functions first (so they can reference each other)
                // 2. Then we define their bodies
                // 3. The body of the let can then call them

                // Save the current insertion point
                let current_block = self.builder().get_insert_block();

                // First pass: declare all recursive functions
                for (var, _expr) in bindings {
                    // Generate a unique name for the lifted function
                    let lifted_name = format!("{}${}", var.name.as_str(), var.id.index());
                    let fn_type = self.lower_function_type(&var.ty)?;
                    let fn_val = self.module.add_function(&lifted_name, fn_type);
                    self.functions.insert(var.id, fn_val);
                }

                // Second pass: define all recursive functions
                for (var, expr) in bindings {
                    self.lower_recursive_function(var, expr)?;
                }

                // Restore insertion point
                if let Some(block) = current_block {
                    self.builder().position_at_end(block);
                }

                // Lower the body (recursive functions are now available)
                let result = self.lower_expr(body)?;

                // Note: We don't remove the functions from self.functions
                // because they're now top-level and may be needed later.
                // This is fine because VarIds are unique.

                Ok(result)
            }
        }
    }

    /// Lower a recursive function that was lifted from a let binding.
    fn lower_recursive_function(&mut self, var: &Var, expr: &Expr) -> CodegenResult<()> {
        let fn_val = self.functions.get(&var.id).copied().ok_or_else(|| {
            CodegenError::Internal(format!("recursive function not declared: {}", var.name.as_str()))
        })?;

        // Create entry block
        let entry = self.llvm_context().append_basic_block(fn_val, "entry");
        self.builder().position_at_end(entry);

        // Handle lambda parameters
        let result = self.lower_function_body(fn_val, expr)?;

        // Build return
        if let Some(val) = result {
            self.builder()
                .build_return(Some(&val))
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
        } else {
            self.builder()
                .build_return(None)
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
        }

        Ok(())
    }

    /// Lower a function body, handling lambda parameters.
    fn lower_function_body(
        &mut self,
        fn_val: FunctionValue<'ctx>,
        expr: &Expr,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // If the expression is a lambda, bind parameters to function arguments
        let mut current = expr;
        let mut param_idx = 0;

        while let Expr::Lam(param, body, _span) = current {
            if let Some(arg) = fn_val.get_nth_param(param_idx) {
                self.env.insert(param.id, arg);
            }
            param_idx += 1;
            current = body.as_ref();
        }

        // Lower the body
        self.lower_expr(current)
    }

    /// Lower a case expression.
    ///
    /// Handles three cases:
    /// 1. Literal patterns (Int, Char) - switch on the primitive value
    /// 2. Constructor patterns (DataCon) - switch on the tag, extract fields
    /// 3. Default pattern - catch-all fallback
    fn lower_case(
        &mut self,
        scrut: &Expr,
        alts: &[Alt],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        eprintln!("[DEBUG] lower_case with {} alternatives", alts.len());
        eprintln!("[DEBUG]   scrut: {:?}", scrut);
        for (i, alt) in alts.iter().enumerate() {
            eprintln!("[DEBUG]   alt[{}] con: {:?}", i, alt.con);
        }
        let scrut_val = self.lower_expr(scrut)?.ok_or_else(|| {
            CodegenError::Internal("scrutinee has no value".to_string())
        })?;

        // Check if all alternatives are Default (no actual pattern matching needed)
        // This happens for simple variable patterns like `f n = 42`
        let all_default = alts.iter().all(|alt| matches!(&alt.con, AltCon::Default));
        if all_default && !alts.is_empty() {
            // Just use the first (and likely only) default alternative
            let alt = &alts[0];

            // Bind any pattern variables to the scrutinee value
            for binder in &alt.binders {
                self.env.insert(binder.id, scrut_val);
            }

            // Lower the RHS
            let result = self.lower_expr(&alt.rhs)?;

            // Clean up bindings
            for binder in &alt.binders {
                self.env.remove(&binder.id);
            }

            return Ok(result);
        }

        // Determine if this is a constructor case or a literal case
        let has_datacon = alts.iter().any(|alt| matches!(&alt.con, AltCon::DataCon(_)));

        // Check if this is a Bool case (True/False patterns) with an integer scrutinee
        // This happens when the condition is a comparison result
        let is_bool_case = alts.iter().any(|alt| {
            if let AltCon::DataCon(con) = &alt.con {
                let name = con.name.as_str();
                name == "True" || name == "False" || name == "GHC.Types.True" || name == "GHC.Types.False"
            } else {
                false
            }
        });

        if is_bool_case && matches!(scrut_val, BasicValueEnum::IntValue(_)) {
            // Bool case with integer scrutinee - use direct integer comparison
            return self.lower_case_bool_as_int(scrut_val, alts);
        }

        if has_datacon {
            self.lower_case_datacon(scrut_val, alts)
        } else {
            self.lower_case_literal(scrut_val, alts)
        }
    }

    /// Lower a case expression with literal patterns.
    fn lower_case_literal(
        &mut self,
        scrut_val: BasicValueEnum<'ctx>,
        alts: &[Alt],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Dispatch based on scrutinee type
        match scrut_val {
            BasicValueEnum::IntValue(i) => self.lower_case_literal_int(i, alts),
            BasicValueEnum::FloatValue(f) => self.lower_case_literal_float(f, alts),
            BasicValueEnum::PointerValue(p) => self.lower_case_literal_string(p, alts),
            _ => Err(CodegenError::Unsupported(
                format!("unsupported scrutinee type for literal case: {:?}", scrut_val.get_type()),
            )),
        }
    }

    /// Lower a case expression with integer literal patterns.
    fn lower_case_literal_int(
        &mut self,
        scrut_int: IntValue<'ctx>,
        alts: &[Alt],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        eprintln!("[DEBUG] lower_case_literal_int called with {} alts", alts.len());
        for (i, alt) in alts.iter().enumerate() {
            eprintln!("[DEBUG]   alt {}: {:?}", i, alt.con);
        }

        let current_fn = self.builder()
            .get_insert_block()
            .ok_or_else(|| CodegenError::Internal("no insert block".to_string()))?
            .get_parent()
            .ok_or_else(|| CodegenError::Internal("no parent function".to_string()))?;

        // Create blocks for each alternative
        let mut blocks = Vec::new();
        let merge_block = self.llvm_context().append_basic_block(current_fn, "case_merge");

        let mut default_block = None;
        let mut cases = Vec::new();

        for alt in alts {
            let block = self.llvm_context().append_basic_block(current_fn, "case_alt");
            blocks.push(block);

            match &alt.con {
                AltCon::Lit(Literal::Int(n)) => {
                    cases.push((self.type_mapper().i64_type().const_int(*n as u64, true), block));
                }
                AltCon::Lit(Literal::Char(c)) => {
                    cases.push((self.type_mapper().i32_type().const_int(*c as u64, false), block));
                }
                AltCon::Default => {
                    default_block = Some(block);
                }
                _ => {
                    return Err(CodegenError::Unsupported(
                        format!("unsupported pattern in literal case: {:?}", alt.con),
                    ))
                }
            }
        }

        // Build switch
        // If there's no explicit default, create an unreachable block
        let default = if let Some(db) = default_block {
            db
        } else {
            let unreachable_block = self.llvm_context().append_basic_block(current_fn, "case_unreachable");
            unreachable_block
        };

        let _switch = self
            .builder()
            .build_switch(scrut_int, default, &cases)
            .map_err(|e| CodegenError::Internal(format!("failed to build switch: {:?}", e)))?;

        // If we created an unreachable block, fill it in
        if default_block.is_none() {
            self.builder().position_at_end(default);
            self.builder().build_unreachable()
                .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;
        }

        // Generate code for each alternative
        self.lower_case_alternatives(alts, &blocks, merge_block)
    }

    /// Lower a case expression with float/double literal patterns.
    /// Uses chained comparisons since LLVM switch doesn't support floats.
    fn lower_case_literal_float(
        &mut self,
        scrut_float: FloatValue<'ctx>,
        alts: &[Alt],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let current_fn = self.builder()
            .get_insert_block()
            .ok_or_else(|| CodegenError::Internal("no insert block".to_string()))?
            .get_parent()
            .ok_or_else(|| CodegenError::Internal("no parent function".to_string()))?;

        let merge_block = self.llvm_context().append_basic_block(current_fn, "float_case_merge");

        // Separate literal alts from default
        let mut literal_alts: Vec<(f64, &Alt)> = Vec::new();
        let mut default_alt: Option<&Alt> = None;

        for alt in alts {
            match &alt.con {
                AltCon::Lit(Literal::Float(f)) => literal_alts.push((*f as f64, alt)),
                AltCon::Lit(Literal::Double(d)) => literal_alts.push((*d, alt)),
                AltCon::Default => default_alt = Some(alt),
                _ => {
                    return Err(CodegenError::Unsupported(
                        format!("unsupported pattern in float case: {:?}", alt.con),
                    ))
                }
            }
        }

        // Create all blocks upfront to avoid creating duplicate blocks
        let mut phi_values: Vec<(BasicValueEnum<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

        // Create the default/else block
        let default_block = self.llvm_context().append_basic_block(current_fn, "float_default");

        // Create comparison blocks (one per literal except the first which uses current block)
        let mut cmp_blocks: Vec<inkwell::basic_block::BasicBlock<'ctx>> = Vec::new();
        for i in 1..literal_alts.len() {
            cmp_blocks.push(self.llvm_context().append_basic_block(current_fn, &format!("float_cmp_{}", i)));
        }

        // Create match blocks (one per literal)
        let mut match_blocks: Vec<inkwell::basic_block::BasicBlock<'ctx>> = Vec::new();
        for i in 0..literal_alts.len() {
            match_blocks.push(self.llvm_context().append_basic_block(current_fn, &format!("float_match_{}", i)));
        }

        // Generate chain of comparisons
        for (i, (val, alt)) in literal_alts.iter().enumerate() {
            // Position at the comparison block
            if i > 0 {
                self.builder().position_at_end(cmp_blocks[i - 1]);
            }
            // i == 0 uses the current block (already positioned there)

            let match_block = match_blocks[i];

            // Determine next block (next comparison or default)
            let next_block = if i + 1 < literal_alts.len() {
                cmp_blocks[i] // cmp_blocks[0] corresponds to float_cmp_1
            } else {
                default_block
            };

            // Build float comparison (ordered equal)
            // Check if it's f32 or f64 by comparing types
            let is_f32 = scrut_float.get_type() == self.type_mapper().f32_type();
            let const_val = if is_f32 {
                self.type_mapper().f32_type().const_float(*val)
            } else {
                self.type_mapper().f64_type().const_float(*val)
            };
            let cmp = self.builder()
                .build_float_compare(inkwell::FloatPredicate::OEQ, scrut_float, const_val, "float_eq")
                .map_err(|e| CodegenError::Internal(format!("failed to build float cmp: {:?}", e)))?;

            self.builder()
                .build_conditional_branch(cmp, match_block, next_block)
                .map_err(|e| CodegenError::Internal(format!("failed to build cond branch: {:?}", e)))?;

            // Generate code for match block
            self.builder().position_at_end(match_block);
            if let Some(result) = self.lower_expr(&alt.rhs)? {
                phi_values.push((result, self.builder().get_insert_block().unwrap()));
            }
            self.builder()
                .build_unconditional_branch(merge_block)
                .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;
        }

        // Generate default block
        self.builder().position_at_end(default_block);
        if let Some(alt) = default_alt {
            // Bind scrutinee to any pattern variables
            for binder in &alt.binders {
                self.env.insert(binder.id, scrut_float.into());
            }
            if let Some(result) = self.lower_expr(&alt.rhs)? {
                phi_values.push((result, self.builder().get_insert_block().unwrap()));
            }
            for binder in &alt.binders {
                self.env.remove(&binder.id);
            }
            self.builder()
                .build_unconditional_branch(merge_block)
                .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;
        } else {
            // No default - generate unreachable
            self.builder()
                .build_unreachable()
                .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;
        }

        // Build phi in merge block
        self.builder().position_at_end(merge_block);
        if phi_values.is_empty() {
            Ok(None)
        } else {
            let target_type = phi_values[0].0.get_type();

            // Coerce all values to the target type
            let mut coerced_values: Vec<(BasicValueEnum<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

            for (val, block) in &phi_values {
                if val.get_type() == target_type {
                    coerced_values.push((*val, *block));
                } else {
                    let terminator = block.get_terminator();
                    if let Some(term) = terminator {
                        self.builder().position_before(&term);
                        let coerced = self.coerce_to_type(*val, target_type)?;
                        coerced_values.push((coerced, *block));
                    } else {
                        coerced_values.push((*val, *block));
                    }
                }
            }

            self.builder().position_at_end(merge_block);

            let phi = self.builder()
                .build_phi(target_type, "float_case_result")
                .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

            for (val, block) in &coerced_values {
                phi.add_incoming(&[(val, *block)]);
            }

            Ok(Some(phi.as_basic_value()))
        }
    }

    /// Lower a case expression with string literal patterns.
    /// Uses strcmp to compare strings.
    fn lower_case_literal_string(
        &mut self,
        scrut_ptr: PointerValue<'ctx>,
        alts: &[Alt],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let current_fn = self.builder()
            .get_insert_block()
            .ok_or_else(|| CodegenError::Internal("no insert block".to_string()))?
            .get_parent()
            .ok_or_else(|| CodegenError::Internal("no parent function".to_string()))?;

        let merge_block = self.llvm_context().append_basic_block(current_fn, "str_case_merge");

        // Get or declare strcmp
        let strcmp_fn = self.get_or_declare_strcmp()?;

        // Separate literal alts from default
        let mut literal_alts: Vec<(&Symbol, &Alt)> = Vec::new();
        let mut default_alt: Option<&Alt> = None;

        for alt in alts {
            match &alt.con {
                AltCon::Lit(Literal::String(s)) => literal_alts.push((s, alt)),
                AltCon::Default => default_alt = Some(alt),
                _ => {
                    return Err(CodegenError::Unsupported(
                        format!("unsupported pattern in string case: {:?}", alt.con),
                    ))
                }
            }
        }

        // Create all blocks upfront to avoid creating duplicate blocks
        let mut phi_values: Vec<(BasicValueEnum<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();
        let default_block = self.llvm_context().append_basic_block(current_fn, "str_default");

        // Create comparison blocks (one per literal except the first which uses current block)
        let mut cmp_blocks: Vec<inkwell::basic_block::BasicBlock<'ctx>> = Vec::new();
        for i in 1..literal_alts.len() {
            cmp_blocks.push(self.llvm_context().append_basic_block(current_fn, &format!("str_cmp_{}", i)));
        }

        // Create match blocks (one per literal)
        let mut match_blocks: Vec<inkwell::basic_block::BasicBlock<'ctx>> = Vec::new();
        for i in 0..literal_alts.len() {
            match_blocks.push(self.llvm_context().append_basic_block(current_fn, &format!("str_match_{}", i)));
        }

        // Generate chain of strcmp comparisons
        for (i, (sym, alt)) in literal_alts.iter().enumerate() {
            // Position at the comparison block
            if i > 0 {
                self.builder().position_at_end(cmp_blocks[i - 1]);
            }
            // i == 0 uses the current block (already positioned there)

            let match_block = match_blocks[i];

            // Determine next block (next comparison or default)
            let next_block = if i + 1 < literal_alts.len() {
                cmp_blocks[i] // cmp_blocks[0] corresponds to str_cmp_1
            } else {
                default_block
            };

            // Create global string constant for the pattern
            let str_const = self.module.add_global_string(&format!("str_pat_{}", i), sym.as_str());

            // Call strcmp(scrut, pattern)
            let cmp_result = self.builder()
                .build_call(
                    strcmp_fn,
                    &[scrut_ptr.into(), str_const.into()],
                    "strcmp_result",
                )
                .map_err(|e| CodegenError::Internal(format!("failed to call strcmp: {:?}", e)))?
                .try_as_basic_value()
                .left()
                .ok_or_else(|| CodegenError::Internal("strcmp returned void".to_string()))?;

            // strcmp returns 0 for equal strings
            let zero = self.type_mapper().i32_type().const_zero();
            let is_equal = self.builder()
                .build_int_compare(
                    inkwell::IntPredicate::EQ,
                    cmp_result.into_int_value(),
                    zero,
                    "str_eq",
                )
                .map_err(|e| CodegenError::Internal(format!("failed to build int cmp: {:?}", e)))?;

            self.builder()
                .build_conditional_branch(is_equal, match_block, next_block)
                .map_err(|e| CodegenError::Internal(format!("failed to build cond branch: {:?}", e)))?;

            // Generate match block
            self.builder().position_at_end(match_block);
            if let Some(result) = self.lower_expr(&alt.rhs)? {
                phi_values.push((result, self.builder().get_insert_block().unwrap()));
            }
            self.builder()
                .build_unconditional_branch(merge_block)
                .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;
        }

        // Generate default block
        self.builder().position_at_end(default_block);
        if let Some(alt) = default_alt {
            for binder in &alt.binders {
                self.env.insert(binder.id, scrut_ptr.into());
            }
            if let Some(result) = self.lower_expr(&alt.rhs)? {
                phi_values.push((result, self.builder().get_insert_block().unwrap()));
            }
            for binder in &alt.binders {
                self.env.remove(&binder.id);
            }
            self.builder()
                .build_unconditional_branch(merge_block)
                .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;
        } else {
            self.builder()
                .build_unreachable()
                .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;
        }

        // Build phi in merge block
        self.builder().position_at_end(merge_block);
        if phi_values.is_empty() {
            Ok(None)
        } else {
            let target_type = phi_values[0].0.get_type();

            // Coerce all values to the target type
            let mut coerced_values: Vec<(BasicValueEnum<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

            for (val, block) in &phi_values {
                if val.get_type() == target_type {
                    coerced_values.push((*val, *block));
                } else {
                    let terminator = block.get_terminator();
                    if let Some(term) = terminator {
                        self.builder().position_before(&term);
                        let coerced = self.coerce_to_type(*val, target_type)?;
                        coerced_values.push((coerced, *block));
                    } else {
                        coerced_values.push((*val, *block));
                    }
                }
            }

            self.builder().position_at_end(merge_block);

            let phi = self.builder()
                .build_phi(target_type, "str_case_result")
                .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

            for (val, block) in &coerced_values {
                phi.add_incoming(&[(val, *block)]);
            }

            Ok(Some(phi.as_basic_value()))
        }
    }

    /// Get or declare the strcmp function.
    fn get_or_declare_strcmp(&self) -> CodegenResult<FunctionValue<'ctx>> {
        let name = "strcmp";
        if let Some(fn_val) = self.module.get_function(name) {
            return Ok(fn_val);
        }

        // int strcmp(const char*, const char*)
        let tm = self.type_mapper();
        let fn_type = tm.i32_type().fn_type(
            &[tm.ptr_type().into(), tm.ptr_type().into()],
            false,
        );
        Ok(self.module.add_function(name, fn_type))
    }

    /// Lower a case expression with constructor patterns.
    fn lower_case_datacon(
        &mut self,
        scrut_val: BasicValueEnum<'ctx>,
        alts: &[Alt],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        eprintln!("[DEBUG] lower_case_datacon with {} alts, scrut_val: {:?}", alts.len(), scrut_val.get_type());
        // For constructor patterns, scrutinee must be a pointer (ADT value)
        let scrut_ptr = match scrut_val {
            BasicValueEnum::PointerValue(p) => p,
            BasicValueEnum::IntValue(i) => {
                // If it's an int, it might be a boxed value - try to interpret as ptr
                self.builder()
                    .build_int_to_ptr(i, self.type_mapper().ptr_type(), "scrut_ptr")
                    .map_err(|e| CodegenError::Internal(format!("failed to cast scrutinee: {:?}", e)))?
            }
            _ => {
                return Err(CodegenError::Unsupported(
                    "case on non-pointer value with constructor patterns".to_string(),
                ))
            }
        };

        // Extract the tag from the ADT value
        let tag = self.extract_adt_tag(scrut_ptr)?;

        let current_fn = self.builder()
            .get_insert_block()
            .ok_or_else(|| CodegenError::Internal("no insert block".to_string()))?
            .get_parent()
            .ok_or_else(|| CodegenError::Internal("no parent function".to_string()))?;

        // Create blocks for each alternative
        let mut blocks = Vec::new();
        let merge_block = self.llvm_context().append_basic_block(current_fn, "case_merge");

        let mut default_block = None;
        let mut cases = Vec::new();

        // Collect DataCon info for field extraction later
        let mut datacon_info: Vec<Option<&DataCon>> = Vec::new();

        for alt in alts {
            let block = self.llvm_context().append_basic_block(current_fn, "case_alt");
            blocks.push(block);

            match &alt.con {
                AltCon::DataCon(con) => {
                    let tag_val = self.type_mapper().i64_type().const_int(con.tag as u64, false);
                    cases.push((tag_val, block));
                    datacon_info.push(Some(con));
                    // Register this constructor for later use in constructor applications
                    self.register_constructor(con.name.as_str(), con.tag, con.arity);
                }
                AltCon::Default => {
                    default_block = Some(block);
                    datacon_info.push(None);
                }
                AltCon::Lit(_) => {
                    return Err(CodegenError::Unsupported(
                        "mixed literal and constructor patterns".to_string(),
                    ))
                }
            }
        }

        // Build switch on tag
        // If there's no explicit default, create an unreachable block to indicate exhaustive matching
        let default = if let Some(db) = default_block {
            db
        } else {
            // Create an unreachable block for the default case
            // This avoids having merge_block as a direct successor of the switch
            let unreachable_block = self.llvm_context().append_basic_block(current_fn, "case_unreachable");
            unreachable_block
        };

        let _switch = self
            .builder()
            .build_switch(tag, default, &cases)
            .map_err(|e| CodegenError::Internal(format!("failed to build switch: {:?}", e)))?;

        // If we created an unreachable block, fill it in
        if default_block.is_none() {
            self.builder().position_at_end(default);
            self.builder().build_unreachable()
                .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;
        }

        // Generate code for each alternative with field extraction
        self.lower_case_datacon_alternatives(alts, &blocks, merge_block, scrut_ptr, &datacon_info)
    }

    /// Lower case alternatives (shared logic for RHS generation).
    ///
    /// Uses a two-pass approach to avoid inserting instructions after terminators:
    /// 1. First pass: lower all RHS expressions and collect (value, block) pairs
    /// 2. Second pass: coerce values if needed, then build branches
    fn lower_case_alternatives(
        &mut self,
        alts: &[Alt],
        blocks: &[inkwell::basic_block::BasicBlock<'ctx>],
        merge_block: inkwell::basic_block::BasicBlock<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        eprintln!("[DEBUG] lower_case_alternatives with {} alts, {} blocks", alts.len(), blocks.len());
        // Pass 1: Lower all expressions and collect values WITHOUT building branches
        let mut collected: Vec<(Option<BasicValueEnum<'ctx>>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

        for (i, alt) in alts.iter().enumerate() {
            eprintln!("[DEBUG]   processing alt {}: {:?}", i, alt.con);
            self.builder().position_at_end(blocks[i]);

            let result = self.lower_expr(&alt.rhs)?;

            // Get the ACTUAL current block (lower_expr may have created nested blocks)
            let current_block = self.builder().get_insert_block()
                .ok_or_else(|| CodegenError::Internal("no current block after lower_expr".to_string()))?;

            collected.push((result, current_block));
        }

        // Determine target type from first non-None value
        let target_type = collected.iter()
            .find_map(|(val, _)| val.map(|v| v.get_type()));

        // Pass 2: Coerce values and build branches
        let mut phi_values: Vec<(BasicValueEnum<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

        for (result, block) in collected {
            // Position at the end of the block where the value was produced
            self.builder().position_at_end(block);

            if let Some(val) = result {
                // Coerce if needed (BEFORE building the branch)
                let final_val = if let Some(target) = target_type {
                    if val.get_type() != target {
                        self.coerce_to_type(val, target)?
                    } else {
                        val
                    }
                } else {
                    val
                };

                // Get the current block
                let final_block = self.builder().get_insert_block()
                    .ok_or_else(|| CodegenError::Internal("no current block after coercion".to_string()))?;
                phi_values.push((final_val, final_block));
            }

            // NOW build the branch (after any coercion)
            self.builder()
                .build_unconditional_branch(merge_block)
                .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;
        }

        // Build phi node in merge block
        self.builder().position_at_end(merge_block);

        if phi_values.is_empty() {
            Ok(None)
        } else {
            let phi_type = phi_values[0].0.get_type();
            let phi = self
                .builder()
                .build_phi(phi_type, "case_result")
                .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

            for (val, block) in &phi_values {
                phi.add_incoming(&[(val, *block)]);
            }

            Ok(Some(phi.as_basic_value()))
        }
    }

    /// Lower a Bool case expression when the scrutinee is an integer (from comparison).
    /// True is represented as non-zero (typically 1), False as 0.
    ///
    /// Uses a two-pass approach to avoid inserting instructions after terminators.
    fn lower_case_bool_as_int(
        &mut self,
        scrut_val: BasicValueEnum<'ctx>,
        alts: &[Alt],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let scrut_int = match scrut_val {
            BasicValueEnum::IntValue(i) => i,
            _ => {
                return Err(CodegenError::Internal(
                    "lower_case_bool_as_int called with non-integer scrutinee".to_string(),
                ))
            }
        };

        let current_fn = self.builder()
            .get_insert_block()
            .ok_or_else(|| CodegenError::Internal("no insert block".to_string()))?
            .get_parent()
            .ok_or_else(|| CodegenError::Internal("no parent function".to_string()))?;

        // Find the True and False alternatives
        let mut true_alt = None;
        let mut false_alt = None;
        let mut default_alt = None;

        for alt in alts {
            match &alt.con {
                AltCon::DataCon(con) => {
                    let name = con.name.as_str();
                    if name == "True" || name == "GHC.Types.True" {
                        true_alt = Some(alt);
                    } else if name == "False" || name == "GHC.Types.False" {
                        false_alt = Some(alt);
                    }
                }
                AltCon::Default => {
                    default_alt = Some(alt);
                }
                _ => {}
            }
        }

        // Create blocks
        let true_block = self.llvm_context().append_basic_block(current_fn, "bool_true");
        let false_block = self.llvm_context().append_basic_block(current_fn, "bool_false");
        let merge_block = self.llvm_context().append_basic_block(current_fn, "bool_merge");

        // Compare scrutinee to zero (False = 0, True = non-zero)
        let zero = self.type_mapper().i64_type().const_zero();
        let is_true = self.builder()
            .build_int_compare(inkwell::IntPredicate::NE, scrut_int, zero, "is_true")
            .map_err(|e| CodegenError::Internal(format!("failed to build bool cmp: {:?}", e)))?;

        // Branch based on the comparison
        self.builder()
            .build_conditional_branch(is_true, true_block, false_block)
            .map_err(|e| CodegenError::Internal(format!("failed to build cond branch: {:?}", e)))?;

        // Pass 1: Lower both branches and collect values WITHOUT building terminators
        // True branch
        self.builder().position_at_end(true_block);
        let true_rhs = if let Some(alt) = true_alt {
            self.lower_expr(&alt.rhs)?
        } else if let Some(alt) = default_alt {
            self.lower_expr(&alt.rhs)?
        } else {
            None
        };
        let true_end_block = self.builder().get_insert_block()
            .ok_or_else(|| CodegenError::Internal("no current block after lower_expr".to_string()))?;

        // False branch
        self.builder().position_at_end(false_block);
        let false_rhs = if let Some(alt) = false_alt {
            self.lower_expr(&alt.rhs)?
        } else if let Some(alt) = default_alt {
            self.lower_expr(&alt.rhs)?
        } else {
            None
        };
        let false_end_block = self.builder().get_insert_block()
            .ok_or_else(|| CodegenError::Internal("no current block after lower_expr".to_string()))?;

        // Determine target type from the first available value
        let target_type = true_rhs.map(|v| v.get_type())
            .or_else(|| false_rhs.map(|v| v.get_type()));

        // Pass 2: Coerce values and build branches
        let mut phi_values: Vec<(BasicValueEnum<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

        // True branch: coerce and terminate
        self.builder().position_at_end(true_end_block);
        if let Some(val) = true_rhs {
            let final_val = if let Some(target) = target_type {
                if val.get_type() != target {
                    self.coerce_to_type(val, target)?
                } else {
                    val
                }
            } else {
                val
            };
            let final_block = self.builder().get_insert_block()
                .ok_or_else(|| CodegenError::Internal("no current block after coercion".to_string()))?;
            phi_values.push((final_val, final_block));
        }
        self.builder()
            .build_unconditional_branch(merge_block)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // False branch: coerce and terminate
        self.builder().position_at_end(false_end_block);
        if let Some(val) = false_rhs {
            let final_val = if let Some(target) = target_type {
                if val.get_type() != target {
                    self.coerce_to_type(val, target)?
                } else {
                    val
                }
            } else {
                val
            };
            let final_block = self.builder().get_insert_block()
                .ok_or_else(|| CodegenError::Internal("no current block after coercion".to_string()))?;
            phi_values.push((final_val, final_block));
        }
        self.builder()
            .build_unconditional_branch(merge_block)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Build phi node in merge block
        self.builder().position_at_end(merge_block);

        if phi_values.is_empty() {
            Ok(None)
        } else {
            let phi_type = phi_values[0].0.get_type();

            let phi = self
                .builder()
                .build_phi(phi_type, "bool_result")
                .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

            for (val, block) in &phi_values {
                phi.add_incoming(&[(val, *block)]);
            }

            Ok(Some(phi.as_basic_value()))
        }
    }

    /// Lower case alternatives with DataCon patterns (extracts fields and binds variables).
    ///
    /// Uses a two-pass approach:
    /// 1. First pass: lower all RHS expressions and collect (value, block) pairs WITHOUT building branches
    /// 2. Second pass: determine target type, coerce values if needed, then build branches
    fn lower_case_datacon_alternatives(
        &mut self,
        alts: &[Alt],
        blocks: &[inkwell::basic_block::BasicBlock<'ctx>],
        merge_block: inkwell::basic_block::BasicBlock<'ctx>,
        scrut_ptr: PointerValue<'ctx>,
        datacon_info: &[Option<&DataCon>],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        eprintln!("[DEBUG] lower_case_datacon_alternatives with {} alts, {} blocks", alts.len(), blocks.len());
        // Pass 1: Lower all expressions and collect values WITHOUT building branches
        // We collect (Option<value>, block) so we know which block each value came from
        let mut collected: Vec<(Option<BasicValueEnum<'ctx>>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

        for (i, alt) in alts.iter().enumerate() {
            eprintln!("[DEBUG]   processing alt {}: {:?}", i, alt.con);
            self.builder().position_at_end(blocks[i]);

            // Extract fields and bind to pattern variables
            if let Some(con) = datacon_info[i] {
                let arity = con.arity;

                // Bind each field to its corresponding pattern variable
                for (field_idx, binder) in alt.binders.iter().enumerate() {
                    if field_idx < arity as usize {
                        let field_ptr = self.extract_adt_field(scrut_ptr, arity, field_idx as u32)?;

                        // Determine if we need to unbox the field
                        let field_val = self.ptr_to_value(field_ptr, &binder.ty)?;
                        self.env.insert(binder.id, field_val);
                    }
                }
            }

            // Lower the RHS with bound variables
            let result = self.lower_expr(&alt.rhs)?;

            // Remove bindings (for proper scoping)
            for binder in &alt.binders {
                self.env.remove(&binder.id);
            }

            // Get the ACTUAL current block (lower_expr may have created nested blocks)
            let current_block = self.builder().get_insert_block()
                .ok_or_else(|| CodegenError::Internal("no current block after lower_expr".to_string()))?;

            collected.push((result, current_block));
        }

        // Determine target type from first non-None value
        let target_type = collected.iter()
            .find_map(|(val, _)| val.map(|v| v.get_type()));

        // Pass 2: Coerce values and build branches
        let mut phi_values: Vec<(BasicValueEnum<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

        for (result, block) in collected {
            // Position at the end of the block where the value was produced
            self.builder().position_at_end(block);

            if let Some(val) = result {
                // Coerce if needed (BEFORE building the branch)
                let final_val = if let Some(target) = target_type {
                    if val.get_type() != target {
                        self.coerce_to_type(val, target)?
                    } else {
                        val
                    }
                } else {
                    val
                };

                // Get the current block (coercion doesn't change it)
                let final_block = self.builder().get_insert_block()
                    .ok_or_else(|| CodegenError::Internal("no current block after coercion".to_string()))?;
                phi_values.push((final_val, final_block));
            }

            // NOW build the branch (after any coercion)
            self.builder()
                .build_unconditional_branch(merge_block)
                .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;
        }

        // Build phi node in merge block
        self.builder().position_at_end(merge_block);

        if phi_values.is_empty() {
            Ok(None)
        } else {
            let phi_type = phi_values[0].0.get_type();
            let phi = self
                .builder()
                .build_phi(phi_type, "case_result")
                .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

            for (val, block) in &phi_values {
                phi.add_incoming(&[(val, *block)]);
            }

            Ok(Some(phi.as_basic_value()))
        }
    }

    /// Convert a pointer to a basic value, unboxing if necessary based on type.
    fn ptr_to_value(&self, ptr: PointerValue<'ctx>, ty: &Ty) -> CodegenResult<BasicValueEnum<'ctx>> {
        let tm = self.type_mapper();

        match ty {
            Ty::Con(con) => {
                let name = con.name.as_str();
                match name {
                    "Int" | "Int#" | "Int64" => {
                        // Unbox: ptr -> int
                        let int_val = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_int")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox int: {:?}", e)))?;
                        Ok(int_val.into())
                    }
                    "Int32" => {
                        let int_val = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_int")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox int: {:?}", e)))?;
                        let truncated = self.builder()
                            .build_int_truncate(int_val, tm.i32_type(), "trunc_i32")
                            .map_err(|e| CodegenError::Internal(format!("failed to truncate: {:?}", e)))?;
                        Ok(truncated.into())
                    }
                    "Bool" => {
                        let int_val = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_bool")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox bool: {:?}", e)))?;
                        let bool_val = self.builder()
                            .build_int_truncate(int_val, tm.bool_type(), "to_bool")
                            .map_err(|e| CodegenError::Internal(format!("failed to truncate bool: {:?}", e)))?;
                        Ok(bool_val.into())
                    }
                    "Char" | "Char#" => {
                        let int_val = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_char")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox char: {:?}", e)))?;
                        let char_val = self.builder()
                            .build_int_truncate(int_val, tm.i32_type(), "to_char")
                            .map_err(|e| CodegenError::Internal(format!("failed to truncate char: {:?}", e)))?;
                        Ok(char_val.into())
                    }
                    "Float" | "Float#" => {
                        // Unbox: ptr -> bits -> float
                        let bits = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_float_bits")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox float: {:?}", e)))?;
                        let truncated = self.builder()
                            .build_int_truncate(bits, tm.i32_type(), "float_bits_32")
                            .map_err(|e| CodegenError::Internal(format!("failed to truncate float bits: {:?}", e)))?;
                        let float_val = self.builder()
                            .build_bit_cast(truncated, tm.f32_type(), "to_float")
                            .map_err(|e| CodegenError::Internal(format!("failed to cast to float: {:?}", e)))?;
                        Ok(float_val)
                    }
                    "Double" | "Double#" => {
                        let bits = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_double_bits")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox double: {:?}", e)))?;
                        let double_val = self.builder()
                            .build_bit_cast(bits, tm.f64_type(), "to_double")
                            .map_err(|e| CodegenError::Internal(format!("failed to cast to double: {:?}", e)))?;
                        Ok(double_val)
                    }
                    _ => {
                        // For other types (ADTs), keep as pointer
                        Ok(ptr.into())
                    }
                }
            }
            Ty::Prim(prim) => {
                use bhc_types::PrimTy;
                match prim {
                    PrimTy::I64 | PrimTy::U64 => {
                        let int_val = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_i64")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox i64: {:?}", e)))?;
                        Ok(int_val.into())
                    }
                    PrimTy::I32 | PrimTy::U32 => {
                        let int_val = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_int")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox int: {:?}", e)))?;
                        let truncated = self.builder()
                            .build_int_truncate(int_val, tm.i32_type(), "trunc_i32")
                            .map_err(|e| CodegenError::Internal(format!("failed to truncate: {:?}", e)))?;
                        Ok(truncated.into())
                    }
                    PrimTy::F32 => {
                        let bits = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_float_bits")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox float: {:?}", e)))?;
                        let truncated = self.builder()
                            .build_int_truncate(bits, tm.i32_type(), "float_bits_32")
                            .map_err(|e| CodegenError::Internal(format!("failed to truncate float bits: {:?}", e)))?;
                        let float_val = self.builder()
                            .build_bit_cast(truncated, tm.f32_type(), "to_float")
                            .map_err(|e| CodegenError::Internal(format!("failed to cast to float: {:?}", e)))?;
                        Ok(float_val)
                    }
                    PrimTy::F64 => {
                        let bits = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_double_bits")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox double: {:?}", e)))?;
                        let double_val = self.builder()
                            .build_bit_cast(bits, tm.f64_type(), "to_double")
                            .map_err(|e| CodegenError::Internal(format!("failed to cast to double: {:?}", e)))?;
                        Ok(double_val)
                    }
                    PrimTy::Char => {
                        let int_val = self.builder()
                            .build_ptr_to_int(ptr, tm.i64_type(), "unbox_char")
                            .map_err(|e| CodegenError::Internal(format!("failed to unbox char: {:?}", e)))?;
                        let char_val = self.builder()
                            .build_int_truncate(int_val, tm.i32_type(), "to_char")
                            .map_err(|e| CodegenError::Internal(format!("failed to truncate char: {:?}", e)))?;
                        Ok(char_val.into())
                    }
                    PrimTy::Addr => Ok(ptr.into()),
                }
            }
            _ => {
                // For function types, type variables, etc., keep as pointer
                Ok(ptr.into())
            }
        }
    }

    /// Convert a Core type to an LLVM function type.
    fn lower_function_type(
        &self,
        ty: &Ty,
    ) -> CodegenResult<inkwell::types::FunctionType<'ctx>> {
        // Collect argument types and return type
        let mut arg_types = Vec::new();
        let mut current = ty;

        while let Ty::Fun(arg, ret) = current {
            let llvm_arg = self.lower_type(arg)?;
            arg_types.push(llvm_arg.into());
            current = ret;
        }

        let ret_type = self.lower_basic_type(current)?;

        if let Some(ret) = ret_type {
            Ok(ret.fn_type(&arg_types, false))
        } else {
            Ok(self.type_mapper().void_type().fn_type(&arg_types, false))
        }
    }

    /// Convert a Core type to an LLVM basic type (for function return).
    fn lower_basic_type(&self, ty: &Ty) -> CodegenResult<Option<BasicTypeEnum<'ctx>>> {
        let tm = self.type_mapper();
        match ty {
            Ty::Con(con) => {
                let name = con.name.as_str();
                match name {
                    "Int" | "Int#" | "Int64" => Ok(Some(tm.i64_type().into())),
                    "Int32" => Ok(Some(tm.i32_type().into())),
                    "Float" | "Float#" => Ok(Some(tm.f32_type().into())),
                    "Double" | "Double#" => Ok(Some(tm.f64_type().into())),
                    "Char" | "Char#" => Ok(Some(tm.i32_type().into())),
                    // Bool uses i64 for consistency with how comparisons return values
                    "Bool" => Ok(Some(tm.i64_type().into())),
                    "()" | "Unit" => Ok(None), // Unit type has no value
                    _ => {
                        // Unknown type - use a pointer for now
                        Ok(Some(tm.ptr_type().into()))
                    }
                }
            }
            Ty::Prim(prim) => {
                use bhc_types::PrimTy;
                match prim {
                    PrimTy::I32 => Ok(Some(tm.i32_type().into())),
                    PrimTy::I64 => Ok(Some(tm.i64_type().into())),
                    PrimTy::U32 => Ok(Some(tm.i32_type().into())),
                    PrimTy::U64 => Ok(Some(tm.i64_type().into())),
                    PrimTy::F32 => Ok(Some(tm.f32_type().into())),
                    PrimTy::F64 => Ok(Some(tm.f64_type().into())),
                    PrimTy::Char => Ok(Some(tm.i32_type().into())),
                    PrimTy::Addr => Ok(Some(tm.ptr_type().into())),
                }
            }
            Ty::App(f, arg) => {
                // Check for IO () which should map to void
                if let Ty::Con(con) = f.as_ref() {
                    if con.name.as_str() == "IO" {
                        // IO a - check if a is ()
                        if let Ty::Tuple(elems) = arg.as_ref() {
                            if elems.is_empty() {
                                // IO () -> void
                                return Ok(None);
                            }
                        }
                        if let Ty::Con(inner_con) = arg.as_ref() {
                            if inner_con.name.as_str() == "()" {
                                // IO () -> void
                                return Ok(None);
                            }
                        }
                    }
                }
                self.lower_basic_type(f)
            }
            Ty::Fun(_, _) => {
                // Function types are pointers
                Ok(Some(tm.ptr_type().into()))
            }
            Ty::Forall(_, body) => self.lower_basic_type(body),
            Ty::Var(_) => {
                // Type variables are erased - use pointer
                Ok(Some(tm.ptr_type().into()))
            }
            Ty::Tuple(elems) => {
                if elems.is_empty() {
                    // Unit type ()
                    Ok(None)
                } else {
                    // Tuples become pointers for now
                    Ok(Some(tm.ptr_type().into()))
                }
            }
            Ty::List(_) => {
                // Lists become pointers
                Ok(Some(tm.ptr_type().into()))
            }
            Ty::Nat(_) => {
                // Type-level naturals are erased at runtime
                Ok(None)
            }
            Ty::TyList(_) => {
                // Type-level lists are erased at runtime
                Ok(None)
            }
            Ty::Error => Ok(None),
        }
    }

    /// Convert a Core type to an LLVM type (for arguments).
    fn lower_type(&self, ty: &Ty) -> CodegenResult<BasicTypeEnum<'ctx>> {
        self.lower_basic_type(ty)?.ok_or_else(|| {
            CodegenError::TypeError("cannot lower void type to basic type".to_string())
        })
    }

    // Helper accessors
    fn llvm_context(&self) -> &'ctx Context {
        self.llvm_ctx
    }

    fn builder(&self) -> &Builder<'ctx> {
        self.module.builder()
    }

    fn type_mapper(&self) -> &TypeMapper<'ctx> {
        self.module.type_mapper()
    }
}

/// Lower a Core module to an LLVM module.
///
/// The module reference `'m` can be shorter than the context lifetime `'ctx`.
pub fn lower_core_module<'ctx, 'm>(
    ctx: &'ctx LlvmContext,
    module: &'m LlvmModule<'ctx>,
    core_module: &CoreModule,
) -> CodegenResult<()> {
    let mut lowering = Lowering::new(ctx, module);
    lowering.lower_module(core_module)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests would go here
}
