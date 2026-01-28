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

        // bhc_force(ptr) -> ptr - Force thunk evaluation to WHNF
        let force_type = i8_ptr_type.fn_type(&[i8_ptr_type.into()], false);
        let force_fn = self.module.llvm_module().add_function("bhc_force", force_type, None);
        self.functions.insert(VarId::new(1011), force_fn);

        // bhc_is_thunk(ptr) -> i32 - Check if object is an unevaluated thunk
        let is_thunk_type = i32_type.fn_type(&[i8_ptr_type.into()], false);
        let is_thunk_fn = self.module.llvm_module().add_function("bhc_is_thunk", is_thunk_type, None);
        self.functions.insert(VarId::new(1012), is_thunk_fn);
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
            .basic()
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
            "take" => Some(2),
            "drop" => Some(2),
            "reverse" => Some(1),
            "append" | "++" => Some(2),
            "enumFromTo" => Some(2),
            "replicate" => Some(2),
            "sum" => Some(1),
            "product" => Some(1),
            "map" => Some(2),
            "filter" => Some(2),
            "foldr" => Some(3),
            "foldl" => Some(3),
            "foldl'" => Some(3),
            "zipWith" => Some(3),
            "zip" => Some(2),
            "last" => Some(1),
            "init" => Some(1),
            "!!" => Some(2),
            "concatMap" => Some(2),
            "concat" => Some(1),

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
            "not" => Some(1),
            "otherwise" => Some(0),

            // IO operations
            "putStrLn" => Some(1),
            "putStr" => Some(1),
            "putChar" => Some(1),
            "print" => Some(1),
            "getLine" => Some(0),

            // Monadic operations
            ">>=" => Some(2),
            ">>" => Some(2),
            "return" => Some(1),
            "pure" => Some(1),

            _ => {
                // Check for field selector pattern: $sel_N where N is a digit
                if name.starts_with("$sel_") {
                    if let Ok(_) = name[5..].parse::<usize>() {
                        return Some(1); // Field selectors take one argument (the tuple/dict)
                    }
                }
                None
            }
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
            "take" => self.lower_builtin_take(args[0], args[1]),
            "drop" => self.lower_builtin_drop(args[0], args[1]),
            "reverse" => self.lower_builtin_reverse(args[0]),
            "append" | "++" => self.lower_builtin_append(args[0], args[1]),
            "enumFromTo" => self.lower_builtin_enum_from_to(args[0], args[1]),
            "replicate" => self.lower_builtin_replicate(args[0], args[1]),
            "sum" => self.lower_builtin_sum(args[0]),
            "product" => self.lower_builtin_product(args[0]),
            "map" => self.lower_builtin_map(args[0], args[1]),
            "filter" => self.lower_builtin_filter(args[0], args[1]),
            "foldr" => self.lower_builtin_foldr(args[0], args[1], args[2]),
            "foldl" => self.lower_builtin_foldl(args[0], args[1], args[2]),
            "foldl'" => self.lower_builtin_foldl_strict(args[0], args[1], args[2]),
            "zipWith" => self.lower_builtin_zipwith(args[0], args[1], args[2]),
            "zip" => self.lower_builtin_zip(args[0], args[1]),
            "last" => self.lower_builtin_last(args[0]),
            "init" => self.lower_builtin_init(args[0]),
            "!!" => self.lower_builtin_index(args[0], args[1]),
            "concatMap" => self.lower_builtin_concat_map(args[0], args[1]),
            "concat" => self.lower_builtin_concat(args[0]),

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
            "not" => self.lower_builtin_not(args[0]),
            "otherwise" => Ok(Some(self.type_mapper().i64_type().const_int(1, false).into())),

            // IO operations
            "putStrLn" => self.lower_builtin_put_str_ln(args[0]),
            "putStr" => self.lower_builtin_put_str(args[0]),
            "putChar" => self.lower_builtin_put_char(args[0]),
            "print" => self.lower_builtin_print(args[0]),
            "getLine" => self.lower_builtin_get_line(),

            // Monadic operations
            ">>=" => self.lower_builtin_bind(args[0], args[1]),
            ">>" => self.lower_builtin_then(args[0], args[1]),
            "return" | "pure" => self.lower_builtin_return(args[0]),

            _ => {
                // Check for field selector pattern: $sel_N
                if name.starts_with("$sel_") {
                    if let Ok(field_index) = name[5..].parse::<u32>() {
                        return self.lower_builtin_field_selector(args[0], field_index);
                    }
                }
                Err(CodegenError::Internal(format!("unknown builtin: {}", name)))
            }
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

        // Capture the current block BEFORE branching (this is the entry point to our loop)
        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;

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

    /// Lower `take` - take first n elements of a list.
    /// Implemented iteratively: collect n elements in reverse, then reverse the result.
    fn lower_builtin_take(&mut self, n_expr: &Expr, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let n_val = self.lower_expr(n_expr)?.ok_or_else(|| {
            CodegenError::Internal("take: n has no value".to_string())
        })?;
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("take: list has no value".to_string())
        })?;

        let n = self.to_int_value(n_val)?;
        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("take expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // Phase 1: Collect n elements into a reversed accumulator
        let collect_header = self.llvm_context().append_basic_block(current_fn, "take_collect_header");
        let collect_body = self.llvm_context().append_basic_block(current_fn, "take_collect_body");
        let collect_exit = self.llvm_context().append_basic_block(current_fn, "take_collect_exit");
        // Phase 2: Reverse the accumulator
        let rev_header = self.llvm_context().append_basic_block(current_fn, "take_rev_header");
        let rev_body = self.llvm_context().append_basic_block(current_fn, "take_rev_body");
        let rev_exit = self.llvm_context().append_basic_block(current_fn, "take_rev_exit");

        // Build nil in entry block before branching
        let nil = self.build_nil()?;
        let entry_block = self.builder().get_insert_block().unwrap();
        self.builder().build_unconditional_branch(collect_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Collect loop header
        self.builder().position_at_end(collect_header);
        let acc_phi = self.builder()
            .build_phi(tm.ptr_type(), "acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(tm.ptr_type(), "list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let count_phi = self.builder()
            .build_phi(tm.i64_type(), "count")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if count <= 0 (taken enough)
        let count = count_phi.as_basic_value().into_int_value();
        let count_le_0 = self.builder()
            .build_int_compare(inkwell::IntPredicate::SLE, count, tm.i64_type().const_zero(), "count_le_0")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        // Check if list is empty
        let current_list = list_phi.as_basic_value().into_pointer_value();
        let tag = self.extract_adt_tag(current_list)?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        // Exit if count <= 0 OR list is empty
        let should_exit = self.builder()
            .build_or(count_le_0, is_empty, "should_exit")
            .map_err(|e| CodegenError::Internal(format!("failed to build or: {:?}", e)))?;
        self.builder().build_conditional_branch(should_exit, collect_exit, collect_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Collect loop body: cons head onto accumulator (in reverse), advance
        self.builder().position_at_end(collect_body);
        let head_ptr = self.extract_adt_field(current_list, 2, 0)?;
        let tail_ptr = self.extract_adt_field(current_list, 2, 1)?;
        let new_acc = self.build_cons(head_ptr.into(), acc_phi.as_basic_value())?;
        let new_count = self.builder()
            .build_int_sub(count, tm.i64_type().const_int(1, false), "new_count")
            .map_err(|e| CodegenError::Internal(format!("failed to sub: {:?}", e)))?;
        self.builder().build_unconditional_branch(collect_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        acc_phi.add_incoming(&[(&nil, entry_block), (&new_acc, collect_body)]);
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, collect_body)]);
        count_phi.add_incoming(&[(&n, entry_block), (&new_count, collect_body)]);

        // collect_exit: build nil for reverse phase and branch to rev_header
        self.builder().position_at_end(collect_exit);
        let nil2 = self.build_nil()?;
        // Save the collected accumulator (will be used as rev_list input)
        let collected_acc = acc_phi.as_basic_value().into_pointer_value();
        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Reverse loop header
        self.builder().position_at_end(rev_header);
        let rev_acc_phi = self.builder()
            .build_phi(tm.ptr_type(), "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list_phi = self.builder()
            .build_phi(tm.ptr_type(), "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_current = rev_list_phi.as_basic_value().into_pointer_value();
        let rev_tag = self.extract_adt_tag(rev_current)?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Reverse loop body
        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_current, 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_current, 2, 1)?;
        let new_rev_acc = self.build_cons(rev_head.into(), rev_acc_phi.as_basic_value())?;
        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming edges
        rev_acc_phi.add_incoming(&[(&nil2, collect_exit), (&new_rev_acc, rev_body)]);
        rev_list_phi.add_incoming(&[(&collected_acc, collect_exit), (&rev_tail, rev_body)]);

        // Return result
        self.builder().position_at_end(rev_exit);
        Ok(Some(rev_acc_phi.as_basic_value()))
    }

    /// Lower `drop` - drop first n elements of a list.
    /// drop n xs | n <= 0 = xs
    /// drop _ [] = []
    /// drop n (_:xs) = drop (n-1) xs
    fn lower_builtin_drop(&mut self, n_expr: &Expr, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let n_val = self.lower_expr(n_expr)?.ok_or_else(|| {
            CodegenError::Internal("drop: n has no value".to_string())
        })?;
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("drop: list has no value".to_string())
        })?;

        let n = self.to_int_value(n_val)?;
        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("drop expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // Create loop blocks for iterative drop
        let loop_header = self.llvm_context().append_basic_block(current_fn, "drop_header");
        let loop_body = self.llvm_context().append_basic_block(current_fn, "drop_body");
        let loop_exit = self.llvm_context().append_basic_block(current_fn, "drop_exit");

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        let entry_block = self.builder().get_insert_block().unwrap();

        // Loop header: phi for remaining count and current list
        self.builder().position_at_end(loop_header);
        let count_phi = self.builder()
            .build_phi(tm.i64_type(), "count")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(tm.ptr_type(), "list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if count <= 0 (done dropping)
        let count_le_0 = self.builder()
            .build_int_compare(inkwell::IntPredicate::SLE, count_phi.as_basic_value().into_int_value(), tm.i64_type().const_zero(), "count_le_0")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        // Check if list is empty
        let current_list = list_phi.as_basic_value().into_pointer_value();
        let tag = self.extract_adt_tag(current_list)?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        // Exit if count <= 0 OR list is empty
        let should_exit = self.builder()
            .build_or(count_le_0, is_empty, "should_exit")
            .map_err(|e| CodegenError::Internal(format!("failed to build or: {:?}", e)))?;
        self.builder().build_conditional_branch(should_exit, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: decrement count, advance to tail
        self.builder().position_at_end(loop_body);
        let new_count = self.builder()
            .build_int_sub(count_phi.as_basic_value().into_int_value(), tm.i64_type().const_int(1, false), "new_count")
            .map_err(|e| CodegenError::Internal(format!("failed to sub: {:?}", e)))?;
        let tail_ptr = self.extract_adt_field(current_list, 2, 1)?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming edges
        count_phi.add_incoming(&[(&n, entry_block), (&new_count, loop_body)]);
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, loop_body)]);

        // Loop exit: return current list
        self.builder().position_at_end(loop_exit);
        Ok(Some(list_phi.as_basic_value()))
    }

    /// Lower `reverse` - reverse a list.
    fn lower_builtin_reverse(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("reverse: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("reverse expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // Create loop blocks for iterative reverse
        let loop_header = self.llvm_context().append_basic_block(current_fn, "rev_header");
        let loop_body = self.llvm_context().append_basic_block(current_fn, "rev_body");
        let loop_exit = self.llvm_context().append_basic_block(current_fn, "rev_exit");

        // Start with empty accumulator
        let nil = self.build_nil()?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        let entry_block = self.builder().get_insert_block().unwrap();

        // Loop header
        self.builder().position_at_end(loop_header);
        let acc_phi = self.builder()
            .build_phi(tm.ptr_type(), "acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(tm.ptr_type(), "list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if list is empty
        let current_list = list_phi.as_basic_value().into_pointer_value();
        let tag = self.extract_adt_tag(current_list)?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
        self.builder().build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: cons head onto accumulator, advance to tail
        self.builder().position_at_end(loop_body);
        let head_ptr = self.extract_adt_field(current_list, 2, 0)?;
        let tail_ptr = self.extract_adt_field(current_list, 2, 1)?;
        let new_acc = self.build_cons(head_ptr.into(), acc_phi.as_basic_value())?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming edges
        acc_phi.add_incoming(&[(&nil, entry_block), (&new_acc, loop_body)]);
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, loop_body)]);

        // Loop exit: return accumulator
        self.builder().position_at_end(loop_exit);
        Ok(Some(acc_phi.as_basic_value()))
    }

    /// Lower `append` / `++` - concatenate two lists.
    fn lower_builtin_append(&mut self, list1_expr: &Expr, list2_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list1_val = self.lower_expr(list1_expr)?.ok_or_else(|| {
            CodegenError::Internal("append: list1 has no value".to_string())
        })?;
        let list2_val = self.lower_expr(list2_expr)?.ok_or_else(|| {
            CodegenError::Internal("append: list2 has no value".to_string())
        })?;

        let list1_ptr = match list1_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("append expects a list".to_string())),
        };
        let list2_ptr = match list2_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("append expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // First reverse list1, then fold onto list2
        // This is: append xs ys = foldl (flip (:)) ys (reverse xs)
        // Or equivalently: go [] xs where go acc [] = acc ++ ys; go acc (x:xs) = go (x:acc) xs

        // Step 1: Reverse list1
        let rev_header = self.llvm_context().append_basic_block(current_fn, "app_rev_header");
        let rev_body = self.llvm_context().append_basic_block(current_fn, "app_rev_body");
        let rev_exit = self.llvm_context().append_basic_block(current_fn, "app_rev_exit");
        let fold_header = self.llvm_context().append_basic_block(current_fn, "app_fold_header");
        let fold_body = self.llvm_context().append_basic_block(current_fn, "app_fold_body");
        let fold_exit = self.llvm_context().append_basic_block(current_fn, "app_fold_exit");

        // Start with empty accumulator for reverse
        let nil = self.build_nil()?;
        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        let entry_block = self.builder().get_insert_block().unwrap();

        // Reverse loop header
        self.builder().position_at_end(rev_header);
        let rev_acc_phi = self.builder()
            .build_phi(tm.ptr_type(), "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list_phi = self.builder()
            .build_phi(tm.ptr_type(), "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_current = rev_list_phi.as_basic_value().into_pointer_value();
        let rev_tag = self.extract_adt_tag(rev_current)?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Reverse loop body
        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_current, 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_current, 2, 1)?;
        let rev_new_acc = self.build_cons(rev_head.into(), rev_acc_phi.as_basic_value())?;
        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        rev_acc_phi.add_incoming(&[(&nil, entry_block), (&rev_new_acc, rev_body)]);
        rev_list_phi.add_incoming(&[(&list1_ptr, entry_block), (&rev_tail, rev_body)]);

        // After reversing, fold reversed list onto list2
        self.builder().position_at_end(rev_exit);
        self.builder().build_unconditional_branch(fold_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Fold loop header
        self.builder().position_at_end(fold_header);
        let fold_acc_phi = self.builder()
            .build_phi(tm.ptr_type(), "fold_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let fold_list_phi = self.builder()
            .build_phi(tm.ptr_type(), "fold_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let fold_current = fold_list_phi.as_basic_value().into_pointer_value();
        let fold_tag = self.extract_adt_tag(fold_current)?;
        let fold_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, fold_tag, tm.i64_type().const_zero(), "fold_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
        self.builder().build_conditional_branch(fold_is_empty, fold_exit, fold_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Fold loop body: cons head of reversed list onto accumulator
        self.builder().position_at_end(fold_body);
        let fold_head = self.extract_adt_field(fold_current, 2, 0)?;
        let fold_tail = self.extract_adt_field(fold_current, 2, 1)?;
        let fold_new_acc = self.build_cons(fold_head.into(), fold_acc_phi.as_basic_value())?;
        self.builder().build_unconditional_branch(fold_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        fold_acc_phi.add_incoming(&[
            (&list2_ptr, rev_exit),
            (&fold_new_acc, fold_body),
        ]);
        fold_list_phi.add_incoming(&[
            (&rev_acc_phi.as_basic_value().into_pointer_value(), rev_exit),
            (&fold_tail, fold_body),
        ]);

        // Return result
        self.builder().position_at_end(fold_exit);
        Ok(Some(fold_acc_phi.as_basic_value()))
    }

    /// Lower `enumFromTo` - generate a list [from..to].
    fn lower_builtin_enum_from_to(&mut self, from_expr: &Expr, to_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let from_val = self.lower_expr(from_expr)?.ok_or_else(|| {
            CodegenError::Internal("enumFromTo: from has no value".to_string())
        })?;
        let to_val = self.lower_expr(to_expr)?.ok_or_else(|| {
            CodegenError::Internal("enumFromTo: to has no value".to_string())
        })?;

        let from = self.to_int_value(from_val)?;
        let to = self.to_int_value(to_val)?;

        let tm = self.type_mapper();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // Build list backwards from to down to from, then we have [from..to]
        let loop_header = self.llvm_context().append_basic_block(current_fn, "enum_header");
        let loop_body = self.llvm_context().append_basic_block(current_fn, "enum_body");
        let loop_exit = self.llvm_context().append_basic_block(current_fn, "enum_exit");

        // Start with empty list, current = to
        let nil = self.build_nil()?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        let entry_block = self.builder().get_insert_block().unwrap();

        // Loop header
        self.builder().position_at_end(loop_header);
        let acc_phi = self.builder()
            .build_phi(tm.ptr_type(), "acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let current_phi = self.builder()
            .build_phi(tm.i64_type(), "current")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if current < from (done)
        let current = current_phi.as_basic_value().into_int_value();
        let done = self.builder()
            .build_int_compare(inkwell::IntPredicate::SLT, current, from, "done")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
        self.builder().build_conditional_branch(done, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: cons current onto accumulator, decrement current
        self.builder().position_at_end(loop_body);
        let boxed_current = self.box_int(current)?;
        let new_acc = self.build_cons(boxed_current.into(), acc_phi.as_basic_value())?;
        let prev_current = self.builder()
            .build_int_sub(current, tm.i64_type().const_int(1, false), "prev")
            .map_err(|e| CodegenError::Internal(format!("failed to sub: {:?}", e)))?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        acc_phi.add_incoming(&[(&nil, entry_block), (&new_acc, loop_body)]);
        current_phi.add_incoming(&[(&to, entry_block), (&prev_current, loop_body)]);

        // Return result
        self.builder().position_at_end(loop_exit);
        Ok(Some(acc_phi.as_basic_value()))
    }

    /// Lower `replicate` - create a list with n copies of an element.
    fn lower_builtin_replicate(&mut self, n_expr: &Expr, elem_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let n_val = self.lower_expr(n_expr)?.ok_or_else(|| {
            CodegenError::Internal("replicate: n has no value".to_string())
        })?;
        let elem_val = self.lower_expr(elem_expr)?.ok_or_else(|| {
            CodegenError::Internal("replicate: elem has no value".to_string())
        })?;

        let n = self.to_int_value(n_val)?;
        let elem_ptr = self.value_to_ptr(elem_val)?;

        let tm = self.type_mapper();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // Build list by consing n times
        let loop_header = self.llvm_context().append_basic_block(current_fn, "rep_header");
        let loop_body = self.llvm_context().append_basic_block(current_fn, "rep_body");
        let loop_exit = self.llvm_context().append_basic_block(current_fn, "rep_exit");

        // Start with empty list
        let nil = self.build_nil()?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        let entry_block = self.builder().get_insert_block().unwrap();

        // Loop header
        self.builder().position_at_end(loop_header);
        let acc_phi = self.builder()
            .build_phi(tm.ptr_type(), "acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let count_phi = self.builder()
            .build_phi(tm.i64_type(), "count")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if count <= 0
        let count = count_phi.as_basic_value().into_int_value();
        let done = self.builder()
            .build_int_compare(inkwell::IntPredicate::SLE, count, tm.i64_type().const_zero(), "done")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;
        self.builder().build_conditional_branch(done, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: cons elem onto accumulator, decrement count
        self.builder().position_at_end(loop_body);
        let new_acc = self.build_cons(elem_ptr.into(), acc_phi.as_basic_value())?;
        let new_count = self.builder()
            .build_int_sub(count, tm.i64_type().const_int(1, false), "new_count")
            .map_err(|e| CodegenError::Internal(format!("failed to sub: {:?}", e)))?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        acc_phi.add_incoming(&[(&nil, entry_block), (&new_acc, loop_body)]);
        count_phi.add_incoming(&[(&n, entry_block), (&new_count, loop_body)]);

        // Return result
        self.builder().position_at_end(loop_exit);
        Ok(Some(acc_phi.as_basic_value()))
    }

    /// Build an empty list (Nil / []).
    fn build_nil(&self) -> CodegenResult<inkwell::values::PointerValue<'ctx>> {
        let tm = self.type_mapper();
        // Nil is represented as ADT with tag 0 and no fields
        // Allocate space: tag (i64) only
        let size = tm.i64_type().const_int(8, false);
        let alloc_fn = self.functions.get(&VarId::new(1005)).ok_or_else(|| {
            CodegenError::Internal("bhc_alloc not declared".to_string())
        })?;
        let alloc_call = self.builder()
            .build_call(*alloc_fn, &[size.into()], "nil_alloc")
            .map_err(|e| CodegenError::Internal(format!("failed to call alloc: {:?}", e)))?;
        let nil_ptr = alloc_call.try_as_basic_value().basic().ok_or_else(|| {
            CodegenError::Internal("alloc returned void".to_string())
        })?.into_pointer_value();

        // Store tag = 0
        self.builder()
            .build_store(nil_ptr, tm.i64_type().const_zero())
            .map_err(|e| CodegenError::Internal(format!("failed to store tag: {:?}", e)))?;

        Ok(nil_ptr)
    }

    /// Build a cons cell (x : xs).
    fn build_cons(&self, head: BasicValueEnum<'ctx>, tail: BasicValueEnum<'ctx>) -> CodegenResult<inkwell::values::PointerValue<'ctx>> {
        let tm = self.type_mapper();
        // Cons is represented as ADT with tag 1 and 2 fields (head, tail)
        // Allocate space: tag (i64) + head (ptr) + tail (ptr) = 24 bytes
        let size = tm.i64_type().const_int(24, false);
        let alloc_fn = self.functions.get(&VarId::new(1005)).ok_or_else(|| {
            CodegenError::Internal("bhc_alloc not declared".to_string())
        })?;
        let alloc_call = self.builder()
            .build_call(*alloc_fn, &[size.into()], "cons_alloc")
            .map_err(|e| CodegenError::Internal(format!("failed to call alloc: {:?}", e)))?;
        let cons_ptr = alloc_call.try_as_basic_value().basic().ok_or_else(|| {
            CodegenError::Internal("alloc returned void".to_string())
        })?.into_pointer_value();

        // Store tag = 1
        self.builder()
            .build_store(cons_ptr, tm.i64_type().const_int(1, false))
            .map_err(|e| CodegenError::Internal(format!("failed to store tag: {:?}", e)))?;

        // Store head at offset 8
        let head_ptr = unsafe {
            self.builder()
                .build_gep(tm.i64_type(), cons_ptr, &[tm.i64_type().const_int(1, false)], "head_ptr")
                .map_err(|e| CodegenError::Internal(format!("failed to build gep: {:?}", e)))?
        };
        let head_as_ptr = self.value_to_ptr(head)?;
        self.builder()
            .build_store(head_ptr, head_as_ptr)
            .map_err(|e| CodegenError::Internal(format!("failed to store head: {:?}", e)))?;

        // Store tail at offset 16
        let tail_slot = unsafe {
            self.builder()
                .build_gep(tm.i64_type(), cons_ptr, &[tm.i64_type().const_int(2, false)], "tail_ptr")
                .map_err(|e| CodegenError::Internal(format!("failed to build gep: {:?}", e)))?
        };
        let tail_as_ptr = self.value_to_ptr(tail)?;
        self.builder()
            .build_store(tail_slot, tail_as_ptr)
            .map_err(|e| CodegenError::Internal(format!("failed to store tail: {:?}", e)))?;

        Ok(cons_ptr)
    }

    /// Try to extract a map application from an expression.
    /// Returns Some((map_fn, inner_list)) if expr is `map f xs`, None otherwise.
    ///
    /// This is used for fusion opportunities like `sum (map f xs)`.
    #[allow(unused_variables)]
    fn try_extract_map_app(_expr: &Expr) -> Option<(&Expr, &Expr)> {
        // TODO: Implement pattern matching for map applications
        // For now, return None to skip fusion and use the non-fused path
        None
    }

    /// Lower a fused sum(map f xs) into a single loop.
    /// This avoids allocating an intermediate list.
    #[allow(unused_variables)]
    fn lower_fused_sum_map(
        &mut self,
        _map_fn: &Expr,
        _inner_list: &Expr,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // TODO: Implement fused sum/map lowering
        // This should generate a single loop that applies map_fn to each element
        // and accumulates the sum, without creating an intermediate list.
        Err(CodegenError::Internal(
            "fused sum/map not yet implemented".to_string(),
        ))
    }

    /// Lower `sum` - sum all elements of a list.
    /// sum [] = 0
    /// sum (x:xs) = x + sum xs
    fn lower_builtin_sum(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Check for fusion opportunity: sum (map f xs)
        // This fuses into a single loop that applies f to each element and sums
        if let Some((map_fn, inner_list)) = Self::try_extract_map_app(list_expr) {
            return self.lower_fused_sum_map(map_fn, inner_list);
        }

        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("sum: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("sum expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "sum_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "sum_body");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "sum_exit");

        // Jump to loop header
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header: phi for accumulator and current list pointer
        self.builder().position_at_end(loop_header);
        let acc_phi = self.builder()
            .build_phi(tm.i64_type(), "sum_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(tm.ptr_type(), "sum_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if list is empty (tag == 0)
        let tag = self.extract_adt_tag(list_phi.as_basic_value().into_pointer_value())?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: extract head, add to accumulator, continue with tail
        self.builder().position_at_end(loop_body);
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let head_val = self.builder()
            .build_ptr_to_int(head_ptr, tm.i64_type(), "head_val")
            .map_err(|e| CodegenError::Internal(format!("failed to ptr_to_int: {:?}", e)))?;
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        let new_acc = self.builder()
            .build_int_add(acc_phi.as_basic_value().into_int_value(), head_val, "new_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to add: {:?}", e)))?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add incoming values to phi nodes
        acc_phi.add_incoming(&[(&tm.i64_type().const_zero(), entry_block), (&new_acc, loop_body)]);
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, loop_body)]);

        // Exit: return accumulator (boxed as pointer)
        self.builder().position_at_end(loop_exit);
        let result = self.builder()
            .build_int_to_ptr(acc_phi.as_basic_value().into_int_value(), tm.ptr_type(), "result")
            .map_err(|e| CodegenError::Internal(format!("failed to int_to_ptr: {:?}", e)))?;
        Ok(Some(result.into()))
    }

    /// Lower `product` - multiply all elements of a list.
    /// product [] = 1
    /// product (x:xs) = x * product xs
    fn lower_builtin_product(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("product: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("product expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "prod_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "prod_body");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "prod_exit");

        // Jump to loop header
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header: phi for accumulator and current list pointer
        self.builder().position_at_end(loop_header);
        let acc_phi = self.builder()
            .build_phi(tm.i64_type(), "prod_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(tm.ptr_type(), "prod_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if list is empty (tag == 0)
        let tag = self.extract_adt_tag(list_phi.as_basic_value().into_pointer_value())?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: extract head, multiply with accumulator, continue with tail
        self.builder().position_at_end(loop_body);
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let head_val = self.builder()
            .build_ptr_to_int(head_ptr, tm.i64_type(), "head_val")
            .map_err(|e| CodegenError::Internal(format!("failed to ptr_to_int: {:?}", e)))?;
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        let new_acc = self.builder()
            .build_int_mul(acc_phi.as_basic_value().into_int_value(), head_val, "new_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to mul: {:?}", e)))?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add incoming values to phi nodes (start with 1 for product)
        acc_phi.add_incoming(&[(&tm.i64_type().const_int(1, false), entry_block), (&new_acc, loop_body)]);
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, loop_body)]);

        // Exit: return accumulator (boxed as pointer)
        self.builder().position_at_end(loop_exit);
        let result = self.builder()
            .build_int_to_ptr(acc_phi.as_basic_value().into_int_value(), tm.ptr_type(), "result")
            .map_err(|e| CodegenError::Internal(format!("failed to int_to_ptr: {:?}", e)))?;
        Ok(Some(result.into()))
    }

    /// Lower `map` - apply a function to each element of a list.
    /// map f [] = []
    /// map f (x:xs) = f x : map f xs
    fn lower_builtin_map(&mut self, fn_expr: &Expr, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Lower the function (should be a closure)
        let fn_val = self.lower_expr(fn_expr)?.ok_or_else(|| {
            CodegenError::Internal("map: function has no value".to_string())
        })?;

        let fn_ptr = match fn_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("map: function must be a closure".to_string())),
        };

        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("map: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("map expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "map_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "map_body");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "map_exit");

        // Build nil in entry block before branching
        let nil = self.build_nil()?;

        // Build result list in reverse (we'll reverse at the end)
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header
        self.builder().position_at_end(loop_header);
        let result_phi = self.builder()
            .build_phi(ptr_type, "map_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(ptr_type, "map_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if list is empty
        let tag = self.extract_adt_tag(list_phi.as_basic_value().into_pointer_value())?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: apply function to head, cons result onto accumulator
        self.builder().position_at_end(loop_body);
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Call the function closure: fn_ptr(fn_ptr, head)
        let closure_fn_ptr = self.extract_closure_fn_ptr(fn_ptr)?;
        let fn_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let mapped_val = self.builder()
            .build_indirect_call(fn_type, closure_fn_ptr, &[fn_ptr.into(), head_ptr.into()], "mapped")
            .map_err(|e| CodegenError::Internal(format!("failed to call map fn: {:?}", e)))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError::Internal("map function returned void".to_string()))?;

        // Build new cons cell
        let new_cons = self.build_cons(mapped_val, result_phi.as_basic_value())?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming values (nil was built in entry block before branching)
        result_phi.add_incoming(&[(&nil, entry_block), (&new_cons, loop_body)]);
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, loop_body)]);

        // Exit: reverse the result (we built it in reverse order)
        self.builder().position_at_end(loop_exit);

        // Call builtin reverse - but we need to do it inline to avoid recursion issues
        // For simplicity, we'll use an iterative reverse
        let rev_header = self.llvm_ctx.append_basic_block(current_fn, "rev_header");
        let rev_body = self.llvm_ctx.append_basic_block(current_fn, "rev_body");
        let rev_exit = self.llvm_ctx.append_basic_block(current_fn, "rev_exit");

        // Build nil2 before branching
        let nil2 = self.build_nil()?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_header);
        let rev_acc = self.builder()
            .build_phi(ptr_type, "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list = self.builder()
            .build_phi(ptr_type, "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_tag = self.extract_adt_tag(rev_list.as_basic_value().into_pointer_value())?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 1)?;
        let rev_new_cons = self.build_cons(rev_head.into(), rev_acc.as_basic_value())?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // nil2 was built in loop_exit before branching
        rev_acc.add_incoming(&[(&nil2, loop_exit), (&rev_new_cons, rev_body)]);
        rev_list.add_incoming(&[(&result_phi.as_basic_value(), loop_exit), (&rev_tail, rev_body)]);

        self.builder().position_at_end(rev_exit);
        Ok(Some(rev_acc.as_basic_value()))
    }

    /// Lower `filter` - keep elements that satisfy a predicate.
    /// filter p [] = []
    /// filter p (x:xs) = if p x then x : filter p xs else filter p xs
    fn lower_builtin_filter(&mut self, pred_expr: &Expr, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Lower the predicate (should be a closure)
        let pred_val = self.lower_expr(pred_expr)?.ok_or_else(|| {
            CodegenError::Internal("filter: predicate has no value".to_string())
        })?;

        let pred_ptr = match pred_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("filter: predicate must be a closure".to_string())),
        };

        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("filter: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("filter expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "filter_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "filter_body");
        let loop_keep = self.llvm_ctx.append_basic_block(current_fn, "filter_keep");
        let loop_skip = self.llvm_ctx.append_basic_block(current_fn, "filter_skip");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "filter_exit");

        // Build nil in entry block before branching
        let nil = self.build_nil()?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header
        self.builder().position_at_end(loop_header);
        let result_phi = self.builder()
            .build_phi(ptr_type, "filter_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(ptr_type, "filter_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if list is empty
        let tag = self.extract_adt_tag(list_phi.as_basic_value().into_pointer_value())?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: check predicate
        self.builder().position_at_end(loop_body);
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Call predicate: pred_ptr(pred_ptr, head)
        let pred_fn_ptr = self.extract_closure_fn_ptr(pred_ptr)?;
        let fn_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let pred_result = self.builder()
            .build_indirect_call(fn_type, pred_fn_ptr, &[pred_ptr.into(), head_ptr.into()], "pred_result")
            .map_err(|e| CodegenError::Internal(format!("failed to call predicate: {:?}", e)))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError::Internal("predicate returned void".to_string()))?;

        // Check if predicate returned True (non-zero value)
        // Bools are boxed as int_to_ptr, so we need to convert back with ptr_to_int
        let pred_bool = self.builder()
            .build_ptr_to_int(pred_result.into_pointer_value(), tm.i64_type(), "pred_bool")
            .map_err(|e| CodegenError::Internal(format!("failed to unbox pred result: {:?}", e)))?;
        let is_true = self.builder()
            .build_int_compare(inkwell::IntPredicate::NE, pred_bool, tm.i64_type().const_zero(), "is_true")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        self.builder().build_conditional_branch(is_true, loop_keep, loop_skip)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Keep: cons head onto result
        self.builder().position_at_end(loop_keep);
        let new_cons = self.build_cons(head_ptr.into(), result_phi.as_basic_value())?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Skip: just continue
        self.builder().position_at_end(loop_skip);
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming values (nil was built in entry block)
        result_phi.add_incoming(&[
            (&nil, entry_block),
            (&new_cons, loop_keep),
            (&result_phi.as_basic_value(), loop_skip),
        ]);
        list_phi.add_incoming(&[
            (&list_ptr, entry_block),
            (&tail_ptr, loop_keep),
            (&tail_ptr, loop_skip),
        ]);

        // Exit: reverse the result
        self.builder().position_at_end(loop_exit);

        // Inline reverse
        let rev_header = self.llvm_ctx.append_basic_block(current_fn, "rev_header");
        let rev_body = self.llvm_ctx.append_basic_block(current_fn, "rev_body");
        let rev_exit = self.llvm_ctx.append_basic_block(current_fn, "rev_exit");

        // Build nil2 before branching
        let nil2 = self.build_nil()?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_header);
        let rev_acc = self.builder()
            .build_phi(ptr_type, "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list = self.builder()
            .build_phi(ptr_type, "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_tag = self.extract_adt_tag(rev_list.as_basic_value().into_pointer_value())?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 1)?;
        let rev_new_cons = self.build_cons(rev_head.into(), rev_acc.as_basic_value())?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // nil2 was built in loop_exit before branching
        rev_acc.add_incoming(&[(&nil2, loop_exit), (&rev_new_cons, rev_body)]);
        rev_list.add_incoming(&[(&result_phi.as_basic_value(), loop_exit), (&rev_tail, rev_body)]);

        self.builder().position_at_end(rev_exit);
        Ok(Some(rev_acc.as_basic_value()))
    }

    /// Lower `foldl` - left fold over a list.
    /// foldl f z [] = z
    /// foldl f z (x:xs) = foldl f (f z x) xs
    ///
    /// Iterative implementation: accumulates from left to right
    fn lower_builtin_foldl(&mut self, func_expr: &Expr, init_expr: &Expr, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Lower the function (should be a closure)
        let func_val = self.lower_expr(func_expr)?.ok_or_else(|| {
            CodegenError::Internal("foldl: function has no value".to_string())
        })?;

        let func_ptr = match func_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("foldl: function must be a closure".to_string())),
        };

        // Lower the initial value
        let init_val = self.lower_expr(init_expr)?.ok_or_else(|| {
            CodegenError::Internal("foldl: initial value has no value".to_string())
        })?;
        let init_ptr = self.value_to_ptr(init_val)?;

        // Lower the list
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("foldl: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("foldl expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "foldl_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "foldl_body");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "foldl_exit");

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header
        self.builder().position_at_end(loop_header);
        let acc_phi = self.builder()
            .build_phi(ptr_type, "foldl_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(ptr_type, "foldl_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if list is empty
        let tag = self.extract_adt_tag(list_phi.as_basic_value().into_pointer_value())?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: acc = f acc head; list = tail
        self.builder().position_at_end(loop_body);
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Call f acc head - function takes (closure_ptr, acc, head) and returns new acc
        // The closure stores the function pointer which expects all args at once
        let fn_ptr = self.extract_closure_fn_ptr(func_ptr)?;
        let fn_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        let new_acc = self.builder()
            .build_indirect_call(fn_type, fn_ptr, &[func_ptr.into(), acc_phi.as_basic_value().into(), head_ptr.into()], "foldl_result")
            .map_err(|e| CodegenError::Internal(format!("failed to call function: {:?}", e)))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError::Internal("foldl: function returned void".to_string()))?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming values
        acc_phi.add_incoming(&[(&init_ptr, entry_block), (&new_acc, loop_body)]);
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, loop_body)]);

        // Exit: return accumulator
        self.builder().position_at_end(loop_exit);
        Ok(Some(acc_phi.as_basic_value()))
    }

    /// Lower `foldr` - right fold over a list.
    /// foldr f z [] = z
    /// foldr f z (x:xs) = f x (foldr f z xs)
    ///
    /// Implementation: reverse the list first, then fold left with swapped args
    fn lower_builtin_foldr(&mut self, func_expr: &Expr, init_expr: &Expr, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Lower the function (should be a closure)
        let func_val = self.lower_expr(func_expr)?.ok_or_else(|| {
            CodegenError::Internal("foldr: function has no value".to_string())
        })?;

        let func_ptr = match func_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("foldr: function must be a closure".to_string())),
        };

        // Lower the initial value
        let init_val = self.lower_expr(init_expr)?.ok_or_else(|| {
            CodegenError::Internal("foldr: initial value has no value".to_string())
        })?;
        let init_ptr = self.value_to_ptr(init_val)?;

        // Lower the list
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("foldr: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("foldr expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // First, reverse the list
        let rev_entry = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let rev_header = self.llvm_ctx.append_basic_block(current_fn, "foldr_rev_header");
        let rev_body = self.llvm_ctx.append_basic_block(current_fn, "foldr_rev_body");
        let rev_exit = self.llvm_ctx.append_basic_block(current_fn, "foldr_rev_exit");

        // Build nil before branching
        let nil = self.build_nil()?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Reverse loop
        self.builder().position_at_end(rev_header);
        let rev_acc = self.builder()
            .build_phi(ptr_type, "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list = self.builder()
            .build_phi(ptr_type, "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_tag = self.extract_adt_tag(rev_list.as_basic_value().into_pointer_value())?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 1)?;
        let rev_new_cons = self.build_cons(rev_head.into(), rev_acc.as_basic_value())?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        rev_acc.add_incoming(&[(&nil, rev_entry), (&rev_new_cons, rev_body)]);
        rev_list.add_incoming(&[(&list_ptr, rev_entry), (&rev_tail, rev_body)]);

        // Now fold the reversed list with f applied as f elem acc
        self.builder().position_at_end(rev_exit);

        let fold_header = self.llvm_ctx.append_basic_block(current_fn, "foldr_fold_header");
        let fold_body = self.llvm_ctx.append_basic_block(current_fn, "foldr_fold_body");
        let fold_exit = self.llvm_ctx.append_basic_block(current_fn, "foldr_fold_exit");

        self.builder().build_unconditional_branch(fold_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Fold loop
        self.builder().position_at_end(fold_header);
        let acc_phi = self.builder()
            .build_phi(ptr_type, "foldr_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(ptr_type, "foldr_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let tag = self.extract_adt_tag(list_phi.as_basic_value().into_pointer_value())?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, fold_exit, fold_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(fold_body);
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Call f head acc - function takes (closure_ptr, elem, acc) and returns new acc
        // Note: for foldr, the function signature is (a -> b -> b), so elem comes first, then acc
        let fn_ptr = self.extract_closure_fn_ptr(func_ptr)?;
        let fn_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        let new_acc = self.builder()
            .build_indirect_call(fn_type, fn_ptr, &[func_ptr.into(), head_ptr.into(), acc_phi.as_basic_value().into()], "foldr_result")
            .map_err(|e| CodegenError::Internal(format!("failed to call function: {:?}", e)))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError::Internal("foldr: function returned void".to_string()))?;

        self.builder().build_unconditional_branch(fold_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming values
        acc_phi.add_incoming(&[(&init_ptr, rev_exit), (&new_acc, fold_body)]);
        list_phi.add_incoming(&[(&rev_acc.as_basic_value(), rev_exit), (&tail_ptr, fold_body)]);

        // Exit: return accumulator
        self.builder().position_at_end(fold_exit);
        Ok(Some(acc_phi.as_basic_value()))
    }

    /// Lower `foldl'` - strict left fold (same as foldl but forces accumulator).
    /// In our implementation, foldl is already strict, so this is the same.
    fn lower_builtin_foldl_strict(
        &mut self,
        func_expr: &Expr,
        init_expr: &Expr,
        list_expr: &Expr,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Our foldl is already strict, so just delegate
        self.lower_builtin_foldl(func_expr, init_expr, list_expr)
    }

    /// Lower `zipWith` - apply a function to pairs of elements from two lists.
    /// zipWith f [] _ = []
    /// zipWith f _ [] = []
    /// zipWith f (x:xs) (y:ys) = f x y : zipWith f xs ys
    fn lower_builtin_zipwith(
        &mut self,
        func_expr: &Expr,
        list1_expr: &Expr,
        list2_expr: &Expr,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Lower the function (should be a closure)
        let func_val = self.lower_expr(func_expr)?.ok_or_else(|| {
            CodegenError::Internal("zipWith: function has no value".to_string())
        })?;

        let func_ptr = match func_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("zipWith: function must be a closure".to_string())),
        };

        let list1_val = self.lower_expr(list1_expr)?.ok_or_else(|| {
            CodegenError::Internal("zipWith: list1 has no value".to_string())
        })?;

        let list1_ptr = match list1_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("zipWith expects lists".to_string())),
        };

        let list2_val = self.lower_expr(list2_expr)?.ok_or_else(|| {
            CodegenError::Internal("zipWith: list2 has no value".to_string())
        })?;

        let list2_ptr = match list2_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("zipWith expects lists".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "zipwith_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "zipwith_body");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "zipwith_exit");

        // Build nil in entry block
        let nil = self.build_nil()?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header
        self.builder().position_at_end(loop_header);
        let result_phi = self.builder()
            .build_phi(ptr_type, "zipwith_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list1_phi = self.builder()
            .build_phi(ptr_type, "zipwith_list1")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list2_phi = self.builder()
            .build_phi(ptr_type, "zipwith_list2")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if either list is empty
        let tag1 = self.extract_adt_tag(list1_phi.as_basic_value().into_pointer_value())?;
        let tag2 = self.extract_adt_tag(list2_phi.as_basic_value().into_pointer_value())?;
        let is_empty1 = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag1, tm.i64_type().const_zero(), "is_empty1")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;
        let is_empty2 = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag2, tm.i64_type().const_zero(), "is_empty2")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;
        let is_empty = self.builder()
            .build_or(is_empty1, is_empty2, "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to build or: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body
        self.builder().position_at_end(loop_body);
        let head1_ptr = self.extract_adt_field(list1_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail1_ptr = self.extract_adt_field(list1_phi.as_basic_value().into_pointer_value(), 2, 1)?;
        let head2_ptr = self.extract_adt_field(list2_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail2_ptr = self.extract_adt_field(list2_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Call the function: f x y
        // For binary functions (most common case with zipWith), we call with both args at once.
        // The closure convention is: closure_fn(closure_ptr, arg1, arg2) -> result
        let closure_fn_ptr = self.extract_closure_fn_ptr(func_ptr)?;
        let fn_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        let mapped_val = self.builder()
            .build_indirect_call(fn_type, closure_fn_ptr, &[func_ptr.into(), head1_ptr.into(), head2_ptr.into()], "mapped")
            .map_err(|e| CodegenError::Internal(format!("failed to call function: {:?}", e)))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError::Internal("zipWith: function returned void".to_string()))?;

        // Build cons cell
        let new_cons = self.build_cons(mapped_val, result_phi.as_basic_value())?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming values
        result_phi.add_incoming(&[(&nil, entry_block), (&new_cons, loop_body)]);
        list1_phi.add_incoming(&[(&list1_ptr, entry_block), (&tail1_ptr, loop_body)]);
        list2_phi.add_incoming(&[(&list2_ptr, entry_block), (&tail2_ptr, loop_body)]);

        // Exit: reverse the result
        self.builder().position_at_end(loop_exit);

        // Inline reverse
        let rev_header = self.llvm_ctx.append_basic_block(current_fn, "rev_header");
        let rev_body = self.llvm_ctx.append_basic_block(current_fn, "rev_body");
        let rev_exit = self.llvm_ctx.append_basic_block(current_fn, "rev_exit");

        let nil2 = self.build_nil()?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_header);
        let rev_acc = self.builder()
            .build_phi(ptr_type, "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list = self.builder()
            .build_phi(ptr_type, "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_tag = self.extract_adt_tag(rev_list.as_basic_value().into_pointer_value())?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 1)?;
        let rev_new_cons = self.build_cons(rev_head.into(), rev_acc.as_basic_value())?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        rev_acc.add_incoming(&[(&nil2, loop_exit), (&rev_new_cons, rev_body)]);
        rev_list.add_incoming(&[(&result_phi.as_basic_value(), loop_exit), (&rev_tail, rev_body)]);

        self.builder().position_at_end(rev_exit);
        Ok(Some(rev_acc.as_basic_value()))
    }

    /// Lower `zip` - combine two lists into a list of pairs.
    /// zip [] _ = []
    /// zip _ [] = []
    /// zip (x:xs) (y:ys) = (x,y) : zip xs ys
    fn lower_builtin_zip(
        &mut self,
        list1_expr: &Expr,
        list2_expr: &Expr,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list1_val = self.lower_expr(list1_expr)?.ok_or_else(|| {
            CodegenError::Internal("zip: list1 has no value".to_string())
        })?;

        let list1_ptr = match list1_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("zip expects lists".to_string())),
        };

        let list2_val = self.lower_expr(list2_expr)?.ok_or_else(|| {
            CodegenError::Internal("zip: list2 has no value".to_string())
        })?;

        let list2_ptr = match list2_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("zip expects lists".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "zip_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "zip_body");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "zip_exit");

        // Build nil in entry block
        let nil = self.build_nil()?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header
        self.builder().position_at_end(loop_header);
        let result_phi = self.builder()
            .build_phi(ptr_type, "zip_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list1_phi = self.builder()
            .build_phi(ptr_type, "zip_list1")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list2_phi = self.builder()
            .build_phi(ptr_type, "zip_list2")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if either list is empty
        let tag1 = self.extract_adt_tag(list1_phi.as_basic_value().into_pointer_value())?;
        let tag2 = self.extract_adt_tag(list2_phi.as_basic_value().into_pointer_value())?;
        let is_empty1 = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag1, tm.i64_type().const_zero(), "is_empty1")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;
        let is_empty2 = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag2, tm.i64_type().const_zero(), "is_empty2")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;
        let is_empty = self.builder()
            .build_or(is_empty1, is_empty2, "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to build or: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body
        self.builder().position_at_end(loop_body);
        let head1_ptr = self.extract_adt_field(list1_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail1_ptr = self.extract_adt_field(list1_phi.as_basic_value().into_pointer_value(), 2, 1)?;
        let head2_ptr = self.extract_adt_field(list2_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail2_ptr = self.extract_adt_field(list2_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Build tuple (x, y) - represented as ADT with tag 0 and 2 fields
        let pair_ptr = self.build_pair(head1_ptr.into(), head2_ptr.into())?;

        // Build cons cell
        let new_cons = self.build_cons(pair_ptr.into(), result_phi.as_basic_value())?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming values
        result_phi.add_incoming(&[(&nil, entry_block), (&new_cons, loop_body)]);
        list1_phi.add_incoming(&[(&list1_ptr, entry_block), (&tail1_ptr, loop_body)]);
        list2_phi.add_incoming(&[(&list2_ptr, entry_block), (&tail2_ptr, loop_body)]);

        // Exit: reverse the result
        self.builder().position_at_end(loop_exit);

        // Inline reverse
        let rev_header = self.llvm_ctx.append_basic_block(current_fn, "rev_header");
        let rev_body = self.llvm_ctx.append_basic_block(current_fn, "rev_body");
        let rev_exit = self.llvm_ctx.append_basic_block(current_fn, "rev_exit");

        let nil2 = self.build_nil()?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_header);
        let rev_acc = self.builder()
            .build_phi(ptr_type, "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list = self.builder()
            .build_phi(ptr_type, "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_tag = self.extract_adt_tag(rev_list.as_basic_value().into_pointer_value())?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 1)?;
        let rev_new_cons = self.build_cons(rev_head.into(), rev_acc.as_basic_value())?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        rev_acc.add_incoming(&[(&nil2, loop_exit), (&rev_new_cons, rev_body)]);
        rev_list.add_incoming(&[(&result_phi.as_basic_value(), loop_exit), (&rev_tail, rev_body)]);

        self.builder().position_at_end(rev_exit);
        Ok(Some(rev_acc.as_basic_value()))
    }

    /// Build a pair (tuple of 2 elements).
    fn build_pair(&self, fst: BasicValueEnum<'ctx>, snd: BasicValueEnum<'ctx>) -> CodegenResult<inkwell::values::PointerValue<'ctx>> {
        let tm = self.type_mapper();
        // Pair is represented as ADT with tag 0 and 2 fields
        // Allocate space: tag (i64) + fst (ptr) + snd (ptr) = 24 bytes
        let size = tm.i64_type().const_int(24, false);
        let alloc_fn = self.functions.get(&VarId::new(1005)).ok_or_else(|| {
            CodegenError::Internal("bhc_alloc not declared".to_string())
        })?;
        let alloc_call = self.builder()
            .build_call(*alloc_fn, &[size.into()], "pair_alloc")
            .map_err(|e| CodegenError::Internal(format!("failed to call alloc: {:?}", e)))?;
        let pair_ptr = alloc_call.try_as_basic_value().basic().ok_or_else(|| {
            CodegenError::Internal("alloc returned void".to_string())
        })?.into_pointer_value();

        // Store tag = 0 (tuple constructor)
        self.builder()
            .build_store(pair_ptr, tm.i64_type().const_zero())
            .map_err(|e| CodegenError::Internal(format!("failed to store tag: {:?}", e)))?;

        // Store fst at offset 8
        let fst_slot = unsafe {
            self.builder()
                .build_gep(tm.i64_type(), pair_ptr, &[tm.i64_type().const_int(1, false)], "fst_ptr")
                .map_err(|e| CodegenError::Internal(format!("failed to build gep: {:?}", e)))?
        };
        let fst_as_ptr = self.value_to_ptr(fst)?;
        self.builder()
            .build_store(fst_slot, fst_as_ptr)
            .map_err(|e| CodegenError::Internal(format!("failed to store fst: {:?}", e)))?;

        // Store snd at offset 16
        let snd_slot = unsafe {
            self.builder()
                .build_gep(tm.i64_type(), pair_ptr, &[tm.i64_type().const_int(2, false)], "snd_ptr")
                .map_err(|e| CodegenError::Internal(format!("failed to build gep: {:?}", e)))?
        };
        let snd_as_ptr = self.value_to_ptr(snd)?;
        self.builder()
            .build_store(snd_slot, snd_as_ptr)
            .map_err(|e| CodegenError::Internal(format!("failed to store snd: {:?}", e)))?;

        Ok(pair_ptr)
    }

    /// Lower `last` - get last element of a list.
    /// last [] = error "empty list"
    /// last [x] = x
    /// last (_:xs) = last xs
    fn lower_builtin_last(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("last: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("last expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;

        // Check if list is empty at start
        let tag = self.extract_adt_tag(list_ptr)?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        let error_block = self.llvm_ctx.append_basic_block(current_fn, "last_error");
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "last_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "last_body");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "last_exit");

        self.builder().build_conditional_branch(is_empty, error_block, loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Error block: empty list
        self.builder().position_at_end(error_block);
        let error_msg = self.module.add_global_string("last_error", "last: empty list");
        let error_fn = self.functions.get(&VarId::new(1006)).ok_or_else(|| {
            CodegenError::Internal("bhc_error not declared".to_string())
        })?;
        self.builder()
            .build_call(*error_fn, &[error_msg.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call error: {:?}", e)))?;
        self.builder()
            .build_unreachable()
            .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;

        // Loop header
        self.builder().position_at_end(loop_header);
        let list_phi = self.builder()
            .build_phi(ptr_type, "last_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Extract head and tail
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Check if tail is empty (this element is the last)
        let tail_tag = self.extract_adt_tag(tail_ptr)?;
        let tail_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tail_tag, tm.i64_type().const_zero(), "tail_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(tail_is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: continue with tail
        self.builder().position_at_end(loop_body);
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming values
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, loop_body)]);

        // Exit: return head
        self.builder().position_at_end(loop_exit);
        Ok(Some(head_ptr.into()))
    }

    /// Lower `init` - get all but last element of a list.
    /// init [] = error "empty list"
    /// init [x] = []
    /// init (x:xs) = x : init xs
    fn lower_builtin_init(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("init: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("init expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;

        // Check if list is empty at start
        let tag = self.extract_adt_tag(list_ptr)?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        let error_block = self.llvm_ctx.append_basic_block(current_fn, "init_error");
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "init_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "init_body");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "init_exit");

        // Build nil in entry block
        let nil = self.build_nil()?;

        self.builder().build_conditional_branch(is_empty, error_block, loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Error block: empty list
        self.builder().position_at_end(error_block);
        let error_msg = self.module.add_global_string("init_error", "init: empty list");
        let error_fn = self.functions.get(&VarId::new(1006)).ok_or_else(|| {
            CodegenError::Internal("bhc_error not declared".to_string())
        })?;
        self.builder()
            .build_call(*error_fn, &[error_msg.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call error: {:?}", e)))?;
        self.builder()
            .build_unreachable()
            .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;

        // Loop header
        self.builder().position_at_end(loop_header);
        let result_phi = self.builder()
            .build_phi(ptr_type, "init_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list_phi = self.builder()
            .build_phi(ptr_type, "init_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Extract head and tail
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Check if tail is empty (this element is the last - don't include it)
        let tail_tag = self.extract_adt_tag(tail_ptr)?;
        let tail_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tail_tag, tm.i64_type().const_zero(), "tail_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(tail_is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: cons head onto result, continue with tail
        self.builder().position_at_end(loop_body);
        let new_cons = self.build_cons(head_ptr.into(), result_phi.as_basic_value())?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming values
        result_phi.add_incoming(&[(&nil, entry_block), (&new_cons, loop_body)]);
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, loop_body)]);

        // Exit: reverse the result
        self.builder().position_at_end(loop_exit);

        // Inline reverse
        let rev_header = self.llvm_ctx.append_basic_block(current_fn, "rev_header");
        let rev_body = self.llvm_ctx.append_basic_block(current_fn, "rev_body");
        let rev_exit = self.llvm_ctx.append_basic_block(current_fn, "rev_exit");

        let nil2 = self.build_nil()?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_header);
        let rev_acc = self.builder()
            .build_phi(ptr_type, "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list = self.builder()
            .build_phi(ptr_type, "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_tag = self.extract_adt_tag(rev_list.as_basic_value().into_pointer_value())?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 1)?;
        let rev_new_cons = self.build_cons(rev_head.into(), rev_acc.as_basic_value())?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        rev_acc.add_incoming(&[(&nil2, loop_exit), (&rev_new_cons, rev_body)]);
        rev_list.add_incoming(&[(&result_phi.as_basic_value(), loop_exit), (&rev_tail, rev_body)]);

        self.builder().position_at_end(rev_exit);
        Ok(Some(rev_acc.as_basic_value()))
    }

    /// Lower `!!` - index into a list.
    /// [] !! _ = error "index too large"
    /// (x:_) !! 0 = x
    /// (_:xs) !! n = xs !! (n-1)
    fn lower_builtin_index(&mut self, list_expr: &Expr, index_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("!!: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("!! expects a list".to_string())),
        };

        let index_val = self.lower_expr(index_expr)?.ok_or_else(|| {
            CodegenError::Internal("!!: index has no value".to_string())
        })?;

        let index = self.to_int_value(index_val)?;

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;

        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "index_header");
        let check_empty = self.llvm_ctx.append_basic_block(current_fn, "index_check_empty");
        let check_zero = self.llvm_ctx.append_basic_block(current_fn, "index_check_zero");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "index_body");
        let error_block = self.llvm_ctx.append_basic_block(current_fn, "index_error");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "index_exit");

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header
        self.builder().position_at_end(loop_header);
        let list_phi = self.builder()
            .build_phi(ptr_type, "index_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let idx_phi = self.builder()
            .build_phi(tm.i64_type(), "index_idx")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        self.builder().build_unconditional_branch(check_empty)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Check if list is empty
        self.builder().position_at_end(check_empty);
        let tag = self.extract_adt_tag(list_phi.as_basic_value().into_pointer_value())?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, error_block, check_zero)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Check if index is zero
        self.builder().position_at_end(check_zero);
        let is_zero = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, idx_phi.as_basic_value().into_int_value(), tm.i64_type().const_zero(), "is_zero")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        self.builder().build_conditional_branch(is_zero, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Error block: index out of bounds
        self.builder().position_at_end(error_block);
        let error_msg = self.module.add_global_string("index_error", "!!: index too large");
        let error_fn = self.functions.get(&VarId::new(1006)).ok_or_else(|| {
            CodegenError::Internal("bhc_error not declared".to_string())
        })?;
        self.builder()
            .build_call(*error_fn, &[error_msg.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to call error: {:?}", e)))?;
        self.builder()
            .build_unreachable()
            .map_err(|e| CodegenError::Internal(format!("failed to build unreachable: {:?}", e)))?;

        // Loop body: decrement index, continue with tail
        self.builder().position_at_end(loop_body);
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;
        let new_idx = self.builder()
            .build_int_sub(idx_phi.as_basic_value().into_int_value(), tm.i64_type().const_int(1, false), "new_idx")
            .map_err(|e| CodegenError::Internal(format!("failed to sub: {:?}", e)))?;
        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Add phi incoming values
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, loop_body)]);
        idx_phi.add_incoming(&[(&index, entry_block), (&new_idx, loop_body)]);

        // Exit: return head
        self.builder().position_at_end(loop_exit);
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        Ok(Some(head_ptr.into()))
    }

    /// Lower `concat` - flatten a list of lists.
    /// concat [] = []
    /// concat (xs:xss) = xs ++ concat xss
    fn lower_builtin_concat(&mut self, list_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("concat: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("concat expects a list of lists".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;

        // Build nil in entry block
        let nil = self.build_nil()?;

        // Outer loop: iterate over list of lists
        let outer_header = self.llvm_ctx.append_basic_block(current_fn, "concat_outer_header");
        let outer_body = self.llvm_ctx.append_basic_block(current_fn, "concat_outer_body");
        let outer_exit = self.llvm_ctx.append_basic_block(current_fn, "concat_outer_exit");

        // Inner loop: iterate over current inner list
        let inner_header = self.llvm_ctx.append_basic_block(current_fn, "concat_inner_header");
        let inner_body = self.llvm_ctx.append_basic_block(current_fn, "concat_inner_body");
        let inner_exit = self.llvm_ctx.append_basic_block(current_fn, "concat_inner_exit");

        self.builder().build_unconditional_branch(outer_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Outer loop header
        self.builder().position_at_end(outer_header);
        let result_phi = self.builder()
            .build_phi(ptr_type, "concat_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let outer_list_phi = self.builder()
            .build_phi(ptr_type, "concat_outer_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if outer list is empty
        let outer_tag = self.extract_adt_tag(outer_list_phi.as_basic_value().into_pointer_value())?;
        let outer_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, outer_tag, tm.i64_type().const_zero(), "outer_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(outer_is_empty, outer_exit, outer_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Outer loop body: get current inner list and process it
        self.builder().position_at_end(outer_body);
        let inner_list_ptr = self.extract_adt_field(outer_list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let outer_tail_ptr = self.extract_adt_field(outer_list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        self.builder().build_unconditional_branch(inner_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Inner loop header
        self.builder().position_at_end(inner_header);
        let inner_result_phi = self.builder()
            .build_phi(ptr_type, "concat_inner_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let inner_list_phi = self.builder()
            .build_phi(ptr_type, "concat_inner_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if inner list is empty
        let inner_tag = self.extract_adt_tag(inner_list_phi.as_basic_value().into_pointer_value())?;
        let inner_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, inner_tag, tm.i64_type().const_zero(), "inner_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(inner_is_empty, inner_exit, inner_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Inner loop body: cons head onto result
        self.builder().position_at_end(inner_body);
        let head_ptr = self.extract_adt_field(inner_list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail_ptr = self.extract_adt_field(inner_list_phi.as_basic_value().into_pointer_value(), 2, 1)?;
        let new_cons = self.build_cons(head_ptr.into(), inner_result_phi.as_basic_value())?;

        self.builder().build_unconditional_branch(inner_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Inner phi incoming values
        inner_result_phi.add_incoming(&[(&result_phi.as_basic_value(), outer_body), (&new_cons, inner_body)]);
        inner_list_phi.add_incoming(&[(&inner_list_ptr, outer_body), (&tail_ptr, inner_body)]);

        // Inner loop exit: continue with outer loop
        self.builder().position_at_end(inner_exit);
        self.builder().build_unconditional_branch(outer_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Outer phi incoming values
        result_phi.add_incoming(&[(&nil, entry_block), (&inner_result_phi.as_basic_value(), inner_exit)]);
        outer_list_phi.add_incoming(&[(&list_ptr, entry_block), (&outer_tail_ptr, inner_exit)]);

        // Outer exit: reverse the result
        self.builder().position_at_end(outer_exit);

        // Inline reverse
        let rev_header = self.llvm_ctx.append_basic_block(current_fn, "rev_header");
        let rev_body = self.llvm_ctx.append_basic_block(current_fn, "rev_body");
        let rev_exit = self.llvm_ctx.append_basic_block(current_fn, "rev_exit");

        let nil2 = self.build_nil()?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_header);
        let rev_acc = self.builder()
            .build_phi(ptr_type, "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list = self.builder()
            .build_phi(ptr_type, "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_tag = self.extract_adt_tag(rev_list.as_basic_value().into_pointer_value())?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 1)?;
        let rev_new_cons = self.build_cons(rev_head.into(), rev_acc.as_basic_value())?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        rev_acc.add_incoming(&[(&nil2, outer_exit), (&rev_new_cons, rev_body)]);
        rev_list.add_incoming(&[(&result_phi.as_basic_value(), outer_exit), (&rev_tail, rev_body)]);

        self.builder().position_at_end(rev_exit);
        Ok(Some(rev_acc.as_basic_value()))
    }

    /// Lower `concatMap` - map and concatenate.
    /// concatMap f xs = concat (map f xs)
    fn lower_builtin_concat_map(
        &mut self,
        func_expr: &Expr,
        list_expr: &Expr,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // For efficiency, we implement this directly rather than via concat (map f xs)
        let func_val = self.lower_expr(func_expr)?.ok_or_else(|| {
            CodegenError::Internal("concatMap: function has no value".to_string())
        })?;

        let func_ptr = match func_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("concatMap: function must be a closure".to_string())),
        };

        let list_val = self.lower_expr(list_expr)?.ok_or_else(|| {
            CodegenError::Internal("concatMap: list has no value".to_string())
        })?;

        let list_ptr = match list_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError("concatMap expects a list".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;

        // Build nil in entry block
        let nil = self.build_nil()?;

        // Outer loop: iterate over input list
        let outer_header = self.llvm_ctx.append_basic_block(current_fn, "concatmap_outer_header");
        let outer_body = self.llvm_ctx.append_basic_block(current_fn, "concatmap_outer_body");
        let outer_exit = self.llvm_ctx.append_basic_block(current_fn, "concatmap_outer_exit");

        // Inner loop: iterate over result of f applied to current element
        let inner_header = self.llvm_ctx.append_basic_block(current_fn, "concatmap_inner_header");
        let inner_body = self.llvm_ctx.append_basic_block(current_fn, "concatmap_inner_body");
        let inner_exit = self.llvm_ctx.append_basic_block(current_fn, "concatmap_inner_exit");

        self.builder().build_unconditional_branch(outer_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Outer loop header
        self.builder().position_at_end(outer_header);
        let result_phi = self.builder()
            .build_phi(ptr_type, "concatmap_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let outer_list_phi = self.builder()
            .build_phi(ptr_type, "concatmap_outer_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if outer list is empty
        let outer_tag = self.extract_adt_tag(outer_list_phi.as_basic_value().into_pointer_value())?;
        let outer_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, outer_tag, tm.i64_type().const_zero(), "outer_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(outer_is_empty, outer_exit, outer_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Outer loop body: apply f to head, process result
        self.builder().position_at_end(outer_body);
        let head_ptr = self.extract_adt_field(outer_list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let outer_tail_ptr = self.extract_adt_field(outer_list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Call f on head
        let closure_fn_ptr = self.extract_closure_fn_ptr(func_ptr)?;
        let fn_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let inner_list_ptr = self.builder()
            .build_indirect_call(fn_type, closure_fn_ptr, &[func_ptr.into(), head_ptr.into()], "f_result")
            .map_err(|e| CodegenError::Internal(format!("failed to call function: {:?}", e)))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError::Internal("concatMap: function returned void".to_string()))?;

        self.builder().build_unconditional_branch(inner_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Inner loop header
        self.builder().position_at_end(inner_header);
        let inner_result_phi = self.builder()
            .build_phi(ptr_type, "concatmap_inner_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let inner_list_phi = self.builder()
            .build_phi(ptr_type, "concatmap_inner_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if inner list is empty
        let inner_tag = self.extract_adt_tag(inner_list_phi.as_basic_value().into_pointer_value())?;
        let inner_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, inner_tag, tm.i64_type().const_zero(), "inner_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(inner_is_empty, inner_exit, inner_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Inner loop body: cons head onto result
        self.builder().position_at_end(inner_body);
        let inner_head_ptr = self.extract_adt_field(inner_list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let inner_tail_ptr = self.extract_adt_field(inner_list_phi.as_basic_value().into_pointer_value(), 2, 1)?;
        let new_cons = self.build_cons(inner_head_ptr.into(), inner_result_phi.as_basic_value())?;

        self.builder().build_unconditional_branch(inner_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Inner phi incoming values
        inner_result_phi.add_incoming(&[(&result_phi.as_basic_value(), outer_body), (&new_cons, inner_body)]);
        inner_list_phi.add_incoming(&[(&inner_list_ptr, outer_body), (&inner_tail_ptr, inner_body)]);

        // Inner loop exit: continue with outer loop
        self.builder().position_at_end(inner_exit);
        self.builder().build_unconditional_branch(outer_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Outer phi incoming values
        result_phi.add_incoming(&[(&nil, entry_block), (&inner_result_phi.as_basic_value(), inner_exit)]);
        outer_list_phi.add_incoming(&[(&list_ptr, entry_block), (&outer_tail_ptr, inner_exit)]);

        // Outer exit: reverse the result
        self.builder().position_at_end(outer_exit);

        // Inline reverse
        let rev_header = self.llvm_ctx.append_basic_block(current_fn, "rev_header");
        let rev_body = self.llvm_ctx.append_basic_block(current_fn, "rev_body");
        let rev_exit = self.llvm_ctx.append_basic_block(current_fn, "rev_exit");

        let nil2 = self.build_nil()?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_header);
        let rev_acc = self.builder()
            .build_phi(ptr_type, "rev_acc")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let rev_list = self.builder()
            .build_phi(ptr_type, "rev_list")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        let rev_tag = self.extract_adt_tag(rev_list.as_basic_value().into_pointer_value())?;
        let rev_is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, rev_tag, tm.i64_type().const_zero(), "rev_is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(rev_is_empty, rev_exit, rev_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        self.builder().position_at_end(rev_body);
        let rev_head = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 0)?;
        let rev_tail = self.extract_adt_field(rev_list.as_basic_value().into_pointer_value(), 2, 1)?;
        let rev_new_cons = self.build_cons(rev_head.into(), rev_acc.as_basic_value())?;

        self.builder().build_unconditional_branch(rev_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        rev_acc.add_incoming(&[(&nil2, outer_exit), (&rev_new_cons, rev_body)]);
        rev_list.add_incoming(&[(&result_phi.as_basic_value(), outer_exit), (&rev_tail, rev_body)]);

        self.builder().position_at_end(rev_exit);
        Ok(Some(rev_acc.as_basic_value()))
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

    /// Lower `$sel_N` - extract Nth field from a tuple/dictionary.
    ///
    /// Field selectors are used for type class dictionary method extraction.
    /// The dictionary is represented as a tuple (ADT with tag 0 and N fields),
    /// and $sel_N extracts the Nth field.
    ///
    /// Memory layout of a tuple: { i64 tag, ptr field_0, ptr field_1, ... }
    /// So field N is at byte offset: 8 + N * 8 = (1 + N) * 8
    fn lower_builtin_field_selector(&mut self, tuple_expr: &Expr, field_index: u32) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let tuple_val = self.lower_expr(tuple_expr)?.ok_or_else(|| {
            CodegenError::Internal(format!("$sel_{}: tuple has no value", field_index))
        })?;

        let tuple_ptr = match tuple_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError(format!("$sel_{} expects a tuple/dictionary", field_index))),
        };

        let tm = self.type_mapper();

        // Use raw pointer arithmetic to access the field at the correct offset.
        // ADT layout: { i64 tag, ptr field_0, ptr field_1, ... }
        // Field N is at index (1 + N) when treating as array of i64-sized elements.
        let field_offset = 1 + field_index; // Skip tag (index 0)

        let field_ptr = unsafe {
            self.builder()
                .build_gep(
                    tm.i64_type(), // Treating memory as array of 8-byte slots
                    tuple_ptr,
                    &[tm.i64_type().const_int(field_offset as u64, false)],
                    &format!("field_ptr_{}", field_index),
                )
                .map_err(|e| CodegenError::Internal(format!("failed to build field gep: {:?}", e)))?
        };

        // Load the field value (which is a pointer)
        let field_val = self
            .builder()
            .build_load(tm.ptr_type(), field_ptr, &format!("field_{}", field_index))
            .map_err(|e| CodegenError::Internal(format!("failed to load field: {:?}", e)))?;

        Ok(Some(field_val))
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

    /// Lower `not` - boolean negation.
    fn lower_builtin_not(&mut self, bool_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let bool_val = self.lower_expr(bool_expr)?.ok_or_else(|| {
            CodegenError::Internal("not: argument has no value".to_string())
        })?;

        let tm = self.type_mapper();

        // Handle different representations of Bool
        let result = match bool_val {
            BasicValueEnum::IntValue(i) => {
                // XOR with 1 to flip the boolean
                let one = i.get_type().const_int(1, false);
                self.builder()
                    .build_xor(i, one, "not_result")
                    .map_err(|e| CodegenError::Internal(format!("failed to build not: {:?}", e)))?
                    .into()
            }
            BasicValueEnum::PointerValue(p) => {
                // Convert pointer to int, XOR with 1, convert back
                let int_val = self.builder()
                    .build_ptr_to_int(p, tm.i64_type(), "bool_to_int")
                    .map_err(|e| CodegenError::Internal(format!("failed to convert bool: {:?}", e)))?;
                let one = tm.i64_type().const_int(1, false);
                let xored = self.builder()
                    .build_xor(int_val, one, "not_result")
                    .map_err(|e| CodegenError::Internal(format!("failed to build not: {:?}", e)))?;
                self.builder()
                    .build_int_to_ptr(xored, tm.ptr_type(), "not_to_ptr")
                    .map_err(|e| CodegenError::Internal(format!("failed to convert not result: {:?}", e)))?
                    .into()
            }
            _ => return Err(CodegenError::TypeError("not expects a boolean".to_string())),
        };

        Ok(Some(result))
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
                // Check if this is a boolean (from comparison or boolean function)
                if self.is_bool_type(&expr_ty) || self.expr_looks_like_bool(val_expr) {
                    // Print as boolean (True/False)
                    let print_fn = self.functions.get(&VarId::new(1008)).ok_or_else(|| {
                        CodegenError::Internal("bhc_print_bool_ln not declared".to_string())
                    })?;

                    self.builder()
                        .build_call(*print_fn, &[i.into()], "")
                        .map_err(|e| CodegenError::Internal(format!("failed to call print_bool: {:?}", e)))?;
                } else {
                    // Print as integer
                    let print_fn = self.functions.get(&VarId::new(1000)).ok_or_else(|| {
                        CodegenError::Internal("bhc_print_int_ln not declared".to_string())
                    })?;

                    self.builder()
                        .build_call(*print_fn, &[i.into()], "")
                        .map_err(|e| CodegenError::Internal(format!("failed to call print_int: {:?}", e)))?;
                }
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
                // Check list first - by type or by expression structure (for when type is Error)
                if self.is_list_type(&expr_ty) || self.expr_looks_like_list(val_expr) {
                    // Print list: [el1, el2, ...]
                    self.print_list(p)?;
                } else if self.is_int_type(&expr_ty) || self.is_type_variable_or_error(&expr_ty) {
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
                } else if self.is_bool_type(&expr_ty) || self.expr_looks_like_bool(val_expr) {
                    // Boxed bool - unbox (ptr_to_int) and print as True/False
                    // Bools are boxed via int_to_ptr (the value is in the pointer bits)
                    let bool_val = self.builder()
                        .build_ptr_to_int(p, self.type_mapper().i64_type(), "unbox_bool")
                        .map_err(|e| CodegenError::Internal(format!("failed to unbox bool: {:?}", e)))?;

                    let print_fn = self.functions.get(&VarId::new(1008)).ok_or_else(|| {
                        CodegenError::Internal("bhc_print_bool_ln not declared".to_string())
                    })?;

                    self.builder()
                        .build_call(*print_fn, &[bool_val.into()], "")
                        .map_err(|e| CodegenError::Internal(format!("failed to call print_bool: {:?}", e)))?;
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

    /// Check if a type is a Bool type.
    fn is_bool_type(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Con(con) => {
                let name = con.name.as_str();
                matches!(name, "Bool" | "Boolean")
            }
            Ty::App(f, _) => self.is_bool_type(f),
            Ty::Forall(_, body) => self.is_bool_type(body),
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

    /// Check if a type is a list type.
    fn is_list_type(&self, ty: &Ty) -> bool {
        match ty {
            // Direct list type
            Ty::List(_) => true,
            // Type application form: [] a
            Ty::App(f, _) => {
                // Check if the type constructor is []
                if let Ty::Con(con) = f.as_ref() {
                    con.name.as_str() == "[]" || con.name.as_str() == "List"
                } else {
                    false
                }
            }
            // Just "[]" without argument (unlikely but handle it)
            Ty::Con(con) => {
                con.name.as_str() == "[]" || con.name.as_str() == "List"
            }
            Ty::Forall(_, body) => self.is_list_type(body),
            _ => false,
        }
    }

    /// Check if an expression looks like a list based on its structure.
    /// This is used when type information is unavailable (Error type).
    fn expr_looks_like_list(&self, expr: &Expr) -> bool {
        match expr {
            // Application of a constructor or function
            Expr::App(f, _, _) => self.expr_looks_like_list(f),
            // Type application
            Expr::TyApp(e, _, _) => self.expr_looks_like_list(e),
            // Let binding - check the body
            Expr::Let(_, body, _) => self.expr_looks_like_list(body),
            // Variable that's a list constructor or function returning a list
            Expr::Var(var, _) => {
                let name = var.name.as_str();
                // List constructors and functions that return lists
                matches!(name, ":" | "[]" | "Nil" | "Cons" |
                         "map" | "filter" | "reverse" | "take" | "drop" |
                         "enumFromTo" | "replicate" | "append" | "tail")
            }
            _ => false,
        }
    }

    /// Check if an expression looks like a boolean based on its structure.
    /// This is used when type information is unavailable (Error type).
    fn expr_looks_like_bool(&self, expr: &Expr) -> bool {
        match expr {
            // Application of comparison or boolean function
            Expr::App(f, _, _) => self.expr_looks_like_bool(f),
            // Type application
            Expr::TyApp(e, _, _) => self.expr_looks_like_bool(e),
            // Let binding - check the body
            Expr::Let(_, body, _) => self.expr_looks_like_bool(body),
            // Variable that's a comparison operator or boolean function
            Expr::Var(var, _) => {
                let name = var.name.as_str();
                // Comparison operators and boolean functions
                matches!(name, ">" | "<" | ">=" | "<=" | "==" | "/=" |
                         "True" | "False" | "not" | "&&" | "||" | "and" | "or" |
                         "isJust" | "isNothing" | "isLeft" | "isRight" |
                         "null" | "elem" | "notElem" | "even" | "odd" |
                         "greaterThan" | "lessThan" | "equals")
            }
            _ => false,
        }
    }

    /// Print a list value: [el1, el2, ...]
    fn print_list(&mut self, list_ptr: inkwell::values::PointerValue<'ctx>) -> CodegenResult<()> {
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();

        let current_fn = self.builder()
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // Get print functions
        let print_char_fn = self.functions.get(&VarId::new(1009)).ok_or_else(|| {
            CodegenError::Internal("bhc_print_char not declared".to_string())
        })?;
        let print_int_fn = self.functions.get(&VarId::new(1003)).ok_or_else(|| {
            CodegenError::Internal("bhc_print_int not declared".to_string())
        })?;

        // Print opening bracket (print_char takes i32)
        self.builder()
            .build_call(*print_char_fn, &[tm.i32_type().const_int('[' as u64, false).into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to print char: {:?}", e)))?;

        // Create loop blocks
        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let loop_header = self.llvm_ctx.append_basic_block(current_fn, "print_list_header");
        let loop_body = self.llvm_ctx.append_basic_block(current_fn, "print_list_body");
        let loop_exit = self.llvm_ctx.append_basic_block(current_fn, "print_list_exit");

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop header: phi for list and first flag
        self.builder().position_at_end(loop_header);
        let list_phi = self.builder()
            .build_phi(ptr_type, "print_list_ptr")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let is_first_phi = self.builder()
            .build_phi(tm.i64_type(), "is_first")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

        // Check if list is empty
        let tag = self.extract_adt_tag(list_phi.as_basic_value().into_pointer_value())?;
        let is_empty = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, tag, tm.i64_type().const_zero(), "is_empty")
            .map_err(|e| CodegenError::Internal(format!("failed to compare tag: {:?}", e)))?;

        self.builder().build_conditional_branch(is_empty, loop_exit, loop_body)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Loop body: print separator if not first, then print element
        self.builder().position_at_end(loop_body);

        // Check if we need to print ", "
        let need_separator = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, is_first_phi.as_basic_value().into_int_value(), tm.i64_type().const_zero(), "need_sep")
            .map_err(|e| CodegenError::Internal(format!("failed to compare: {:?}", e)))?;

        let sep_block = self.llvm_ctx.append_basic_block(current_fn, "print_sep");
        let after_sep = self.llvm_ctx.append_basic_block(current_fn, "after_sep");

        self.builder().build_conditional_branch(need_separator, sep_block, after_sep)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Print ", "
        self.builder().position_at_end(sep_block);
        self.builder()
            .build_call(*print_char_fn, &[tm.i32_type().const_int(',' as u64, false).into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to print char: {:?}", e)))?;
        self.builder()
            .build_call(*print_char_fn, &[tm.i32_type().const_int(' ' as u64, false).into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to print char: {:?}", e)))?;
        self.builder().build_unconditional_branch(after_sep)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Continue printing element
        self.builder().position_at_end(after_sep);
        let head_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 0)?;
        let tail_ptr = self.extract_adt_field(list_phi.as_basic_value().into_pointer_value(), 2, 1)?;

        // Print head as int (for now assume list of ints)
        let head_int = self.builder()
            .build_ptr_to_int(head_ptr, tm.i64_type(), "head_int")
            .map_err(|e| CodegenError::Internal(format!("failed to ptr_to_int: {:?}", e)))?;
        self.builder()
            .build_call(*print_int_fn, &[head_int.into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to print int: {:?}", e)))?;

        self.builder().build_unconditional_branch(loop_header)
            .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;

        // Update phi nodes
        list_phi.add_incoming(&[(&list_ptr, entry_block), (&tail_ptr, after_sep)]);
        is_first_phi.add_incoming(&[
            (&tm.i64_type().const_int(1, false), entry_block),
            (&tm.i64_type().const_zero(), after_sep),
        ]);

        // Loop exit: print closing bracket and newline
        self.builder().position_at_end(loop_exit);
        self.builder()
            .build_call(*print_char_fn, &[tm.i32_type().const_int(']' as u64, false).into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to print char: {:?}", e)))?;
        self.builder()
            .build_call(*print_char_fn, &[tm.i32_type().const_int('\n' as u64, false).into()], "")
            .map_err(|e| CodegenError::Internal(format!("failed to print char: {:?}", e)))?;

        Ok(())
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

    /// Lower `>>=` (bind) for monads.
    /// For IO: execute first action, pass result to function, execute result.
    fn lower_builtin_bind(&mut self, action_expr: &Expr, func_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Execute the first action
        let action_result = self.lower_expr(action_expr)?.ok_or_else(|| {
            CodegenError::Internal(">>=: action has no value".to_string())
        })?;

        // Lower the function
        let func_val = self.lower_expr(func_expr)?.ok_or_else(|| {
            CodegenError::Internal(">>=: function has no value".to_string())
        })?;

        // Apply the function to the action's result
        let func_ptr = match func_val {
            BasicValueEnum::PointerValue(p) => p,
            _ => return Err(CodegenError::TypeError(">>=: function must be a closure".to_string())),
        };

        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();

        // Convert action result to pointer for uniform calling convention
        let action_ptr = self.value_to_ptr(action_result)?;

        // Call the function with the action result
        let fn_ptr = self.extract_closure_fn_ptr(func_ptr)?;
        let fn_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let result = self.builder()
            .build_indirect_call(fn_type, fn_ptr, &[func_ptr.into(), action_ptr.into()], "bind_result")
            .map_err(|e| CodegenError::Internal(format!("failed to call bind function: {:?}", e)))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError::Internal(">>=: function returned void".to_string()))?;

        Ok(Some(result))
    }

    /// Lower `>>` (then) for monads.
    /// For IO: execute first action, ignore result, execute second action.
    fn lower_builtin_then(&mut self, action1_expr: &Expr, action2_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Execute the first action (result is discarded)
        let _action1_result = self.lower_expr(action1_expr)?;

        // Execute the second action and return its result
        self.lower_expr(action2_expr)
    }

    /// Lower `return` / `pure` for monads.
    /// For IO: just return the value wrapped (identity for our simple model).
    fn lower_builtin_return(&mut self, value_expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // For our simple IO model, return is identity
        self.lower_expr(value_expr)
    }

    /// Check if an expression is structurally a list (Cons applications or Nil).
    /// This is used when type information isn't available.
    fn is_list_expr(expr: &Expr) -> bool {
        match expr {
            // Empty list []
            Expr::Var(v, _) if v.name.as_str() == "[]" => true,
            // Cons application: (:) head tail
            Expr::App(func, _, _) => {
                // Check if this is a (:) application
                let mut current = func.as_ref();
                // Unwrap one more App for curried (:) x y
                if let Expr::App(inner_func, _, _) = current {
                    current = inner_func.as_ref();
                }
                // Check if the head is the (:) constructor
                if let Expr::Var(v, _) = current {
                    v.name.as_str() == ":"
                } else {
                    false
                }
            }
            _ => false,
        }
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

            // Tuple constructors - used for type class dictionaries
            "(,)" => return Some((0, 2)),      // 2-tuple
            "(,,)" => return Some((0, 3)),     // 3-tuple
            "(,,,)" => return Some((0, 4)),    // 4-tuple
            "(,,,,)" => return Some((0, 5)),   // 5-tuple
            "(,,,,,)" => return Some((0, 6)),  // 6-tuple
            "(,,,,,,)" => return Some((0, 7)), // 7-tuple
            "(,,,,,,,)" => return Some((0, 8)), // 8-tuple

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
            .basic()
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

    // ========================================================================
    // Thunk Support
    // ========================================================================
    //
    // Thunks are suspended computations for lazy evaluation.
    //
    // Memory Layout:
    //   struct Thunk {
    //       i64   tag;       // -1 = unevaluated, -2 = blackhole, >= 0 = evaluated
    //       ptr   eval_fn;   // Function to evaluate (when unevaluated)
    //       i64   env_size;  // Number of captured variables
    //       ptr[] env;       // Captured environment
    //   }
    //
    // The eval_fn has signature: extern "C" fn(env: *mut u8) -> *mut u8
    // It takes the environment pointer and returns the evaluated value.
    // ========================================================================

    /// Tag constant for unevaluated thunks.
    const THUNK_TAG: i64 = -1;

    /// Get the LLVM struct type for a thunk with the given environment size.
    ///
    /// Thunk layout: { i64 tag, ptr eval_fn, i64 env_size, [env_size x ptr] env }
    fn thunk_type(&self, env_size: u32) -> inkwell::types::StructType<'ctx> {
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let i64_type = tm.i64_type();

        // Create array type for environment
        let env_type = ptr_type.array_type(env_size);

        // Struct: { i64 tag, ptr eval_fn, i64 env_size, [env_size x ptr] env }
        self.llvm_ctx.struct_type(
            &[i64_type.into(), ptr_type.into(), i64_type.into(), env_type.into()],
            false,
        )
    }

    /// Allocate a thunk with the given evaluation function and captured variables.
    fn alloc_thunk(
        &self,
        eval_fn: PointerValue<'ctx>,
        captured_vars: &[(VarId, BasicValueEnum<'ctx>)],
    ) -> CodegenResult<PointerValue<'ctx>> {
        let tm = self.type_mapper();
        let env_size = captured_vars.len() as u32;
        let thunk_ty = self.thunk_type(env_size);

        // Calculate size: sizeof(i64) + sizeof(ptr) + sizeof(i64) + env_size * sizeof(ptr)
        let size = 8 + 8 + 8 + (env_size as u64) * 8;

        // Call bhc_alloc
        let alloc_fn = self.functions.get(&VarId::new(1005)).ok_or_else(|| {
            CodegenError::Internal("bhc_alloc not declared".to_string())
        })?;

        let size_val = tm.i64_type().const_int(size, false);
        let raw_ptr = self
            .builder()
            .build_call(*alloc_fn, &[size_val.into()], "thunk_alloc")
            .map_err(|e| CodegenError::Internal(format!("failed to call bhc_alloc: {:?}", e)))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError::Internal("bhc_alloc returned void".to_string()))?;

        let thunk_ptr = raw_ptr.into_pointer_value();

        // Store tag = -1 (THUNK_TAG) at offset 0
        let tag_slot = self
            .builder()
            .build_struct_gep(thunk_ty, thunk_ptr, 0, "thunk_tag_slot")
            .map_err(|e| CodegenError::Internal(format!("failed to get tag slot: {:?}", e)))?;

        let tag_val = tm.i64_type().const_int(Self::THUNK_TAG as u64, true);
        self.builder()
            .build_store(tag_slot, tag_val)
            .map_err(|e| CodegenError::Internal(format!("failed to store thunk tag: {:?}", e)))?;

        // Store eval function pointer at offset 1
        let eval_fn_slot = self
            .builder()
            .build_struct_gep(thunk_ty, thunk_ptr, 1, "thunk_eval_fn_slot")
            .map_err(|e| CodegenError::Internal(format!("failed to get eval_fn slot: {:?}", e)))?;

        self.builder()
            .build_store(eval_fn_slot, eval_fn)
            .map_err(|e| CodegenError::Internal(format!("failed to store eval_fn: {:?}", e)))?;

        // Store env_size at offset 2
        let env_size_slot = self
            .builder()
            .build_struct_gep(thunk_ty, thunk_ptr, 2, "thunk_env_size_slot")
            .map_err(|e| CodegenError::Internal(format!("failed to get env_size slot: {:?}", e)))?;

        let env_size_val = tm.i64_type().const_int(env_size as u64, false);
        self.builder()
            .build_store(env_size_slot, env_size_val)
            .map_err(|e| CodegenError::Internal(format!("failed to store env_size: {:?}", e)))?;

        // Store captured variables in environment array
        if env_size > 0 {
            let env_slot = self
                .builder()
                .build_struct_gep(thunk_ty, thunk_ptr, 3, "thunk_env_slot")
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
                            &format!("thunk_env_{}", i),
                        )
                        .map_err(|e| CodegenError::Internal(format!("failed to get env elem: {:?}", e)))?
                };

                // Convert value to pointer if needed
                let ptr_val = self.value_to_ptr(*val)?;
                self.builder()
                    .build_store(elem_ptr, ptr_val)
                    .map_err(|e| CodegenError::Internal(format!("failed to store thunk env elem: {:?}", e)))?;
            }
        }

        Ok(thunk_ptr)
    }

    /// Generate a call to bhc_force to evaluate a thunk to WHNF.
    fn build_force(&self, val: BasicValueEnum<'ctx>) -> CodegenResult<BasicValueEnum<'ctx>> {
        let ptr = match val {
            BasicValueEnum::PointerValue(p) => p,
            // Non-pointers don't need forcing (primitives)
            _ => return Ok(val),
        };

        let force_fn = self.functions.get(&VarId::new(1011)).ok_or_else(|| {
            CodegenError::Internal("bhc_force not declared".to_string())
        })?;

        let result = self
            .builder()
            .build_call(*force_fn, &[ptr.into()], "forced")
            .map_err(|e| CodegenError::Internal(format!("failed to call bhc_force: {:?}", e)))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError::Internal("bhc_force returned void".to_string()))?;

        Ok(result)
    }

    /// Generate a unique name for a thunk evaluation function.
    fn next_thunk_name(&mut self) -> String {
        let name = format!("__thunk_eval_{}", self.closure_counter);
        self.closure_counter += 1;
        name
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

        // Only use fallback typing for functions with parameters and Error types
        // For type variables, lower_function_type handles them correctly (as pointers)
        let fn_type = if param_count > 0 && matches!(&var.ty, Ty::Error) {
            // Fallback for Error types only - use pointers for uniform calling convention
            let tm = self.type_mapper();

            // Use pointer type for all parameters (handles closures, polymorphic values)
            let param_types: Vec<_> = (0..param_count)
                .map(|_| tm.ptr_type().into())
                .collect();

            // Default return type is pointer
            tm.ptr_type().fn_type(&param_types, false)
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
        let fn_val = self.functions.get(&var.id).copied().ok_or_else(|| {
            CodegenError::Internal(format!("function not declared: {}", var.name.as_str()))
        })?;

        // Create entry block
        let entry = self.llvm_context().append_basic_block(fn_val, "entry");
        self.builder().position_at_end(entry);

        // Lower the function body, handling lambda parameters
        let result = self.lower_function_body(fn_val, expr)?;

        // Check if the current block already has a terminator (e.g., from `error` or `unreachable`)
        // If so, don't add another terminator
        let current_block = self.builder().get_insert_block();
        let has_terminator = current_block
            .map(|bb| bb.get_terminator().is_some())
            .unwrap_or(false);

        if !has_terminator {
            // Build return based on function's declared return type, not the computed result
            // This handles cases like IO () which produces a value but should return void
            let ret_type = fn_val.get_type().get_return_type();
            if ret_type.is_none() {
                // Void return type - don't return a value
                self.builder()
                    .build_return(None)
                    .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
            } else if let Some(val) = result {
                // Convert to pointer if return type is pointer (uniform calling convention)
                let ret_val: BasicValueEnum<'ctx> = if ret_type == Some(self.type_mapper().ptr_type().into()) {
                    self.value_to_ptr(val)?.into()
                } else {
                    val
                };
                self.builder()
                    .build_return(Some(&ret_val))
                    .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
            } else {
                // Function expects a return value but body produced none
                // This shouldn't happen with correct type checking
                self.builder()
                    .build_return(None)
                    .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
            }
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
                    // Check if it's a CAF (only env pointer, no real arguments)
                    // All functions have at least 1 param (env pointer) due to uniform calling convention
                    let fn_type = fn_val.get_type();
                    if fn_type.count_param_types() <= 1 {
                        // CAF - call the function with null env to get its value
                        let null_env = self.type_mapper().ptr_type().const_null();
                        let call_result = self.builder()
                            .build_call(*fn_val, &[null_env.into()], "caf_result")
                            .map_err(|e| CodegenError::Internal(format!("failed to call CAF: {:?}", e)))?;
                        // Get the return value
                        if let Some(ret_val) = call_result.try_as_basic_value().basic() {
                            Ok(Some(ret_val))
                        } else {
                            // Void function - shouldn't happen for CAFs
                            Ok(None)
                        }
                    } else {
                        // Function with parameters - wrap in closure for uniform calling convention
                        let fn_ptr = fn_val.as_global_value().as_pointer_value();
                        let closure_ptr = self.alloc_closure(fn_ptr, &[])?;
                        Ok(Some(closure_ptr.into()))
                    }
                } else if let Some((primop, arity)) = self.primitive_op_info(name) {
                    // Primop used as a value - create a wrapper closure
                    self.create_primop_closure(primop, arity, name)
                } else if let Some(arity) = self.builtin_info(name) {
                    // Builtin used as a value - create a wrapper closure
                    self.create_builtin_closure(name, arity)
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
                // Create a thunk for lazy evaluation
                self.lower_lazy(inner)
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

        // Lower the body in tail position (enables tail call optimization)
        let was_tail = self.in_tail_position;
        self.in_tail_position = true;
        let result = self.lower_expr(current_body)?;
        self.in_tail_position = was_tail;

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

    /// Lower a lazy expression to a thunk.
    ///
    /// Creates a thunk that suspends the evaluation of the inner expression.
    /// The thunk is evaluated when forced (via bhc_force).
    fn lower_lazy(&mut self, inner: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Compute free variables in the lazy expression
        let free = self.free_vars(inner);

        // Collect the values of free variables from current environment
        let mut captured: Vec<(VarId, BasicValueEnum<'ctx>)> = Vec::new();
        for var_id in &free {
            if let Some(val) = self.env.get(var_id) {
                captured.push((*var_id, *val));
            }
        }

        // Save current insertion point
        let current_block = self.builder().get_insert_block();

        // Generate unique name for the thunk evaluation function
        let fn_name = self.next_thunk_name();

        // Build the function type for the eval function:
        // fn(env: *mut u8) -> *mut u8
        // The env pointer points to the captured variables array
        // Extract types early to avoid borrow conflicts with self.env
        let ptr_type = self.type_mapper().ptr_type();
        let i64_type = self.type_mapper().i64_type();
        let env_array_type = ptr_type.array_type(captured.len() as u32);

        let fn_type = ptr_type.fn_type(&[ptr_type.into()], false);

        // Create the thunk evaluation function
        let eval_fn = self.module.add_function(&fn_name, fn_type);

        // Create entry block for the eval function
        let entry = self.llvm_context().append_basic_block(eval_fn, "entry");
        self.builder().position_at_end(entry);

        // Save old environment and create new scope
        let old_env = std::mem::take(&mut self.env);

        // Bind captured variables from environment parameter
        if !captured.is_empty() {
            let env_ptr = eval_fn.get_first_param()
                .ok_or_else(|| CodegenError::Internal("missing env param".to_string()))?
                .into_pointer_value();

            for (i, (var_id, _)) in captured.iter().enumerate() {
                // Load variable from env array
                let elem_ptr = unsafe {
                    self.builder()
                        .build_in_bounds_gep(
                            env_array_type,
                            env_ptr,
                            &[
                                i64_type.const_zero(),
                                i64_type.const_int(i as u64, false),
                            ],
                            &format!("thunk_env_load_{}", i),
                        )
                        .map_err(|e| CodegenError::Internal(format!("failed to get env elem ptr: {:?}", e)))?
                };

                let elem_val = self
                    .builder()
                    .build_load(ptr_type, elem_ptr, &format!("thunk_env_val_{}", i))
                    .map_err(|e| CodegenError::Internal(format!("failed to load env elem: {:?}", e)))?;

                self.env.insert(*var_id, elem_val);
            }
        }

        // Lower the inner expression (this produces the actual computation)
        let result = self.lower_expr(inner)?;

        // Build return
        if let Some(val) = result {
            let ret_ptr = self.value_to_ptr(val)?;
            self.builder()
                .build_return(Some(&ret_ptr))
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
        } else {
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

        // Allocate thunk with eval function pointer and captured values
        let fn_ptr = eval_fn.as_global_value().as_pointer_value();
        let thunk_ptr = self.alloc_thunk(fn_ptr, &captured)?;

        Ok(Some(thunk_ptr.into()))
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

    /// Create a closure wrapping a primitive operation.
    /// This is used when a primop like (+) is used as a first-class value.
    fn create_primop_closure(
        &mut self,
        primop: PrimOp,
        arity: u32,
        name: &str,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let i64_type = tm.i64_type();

        // Create a unique name for the wrapper function
        let wrapper_name = format!("primop_wrapper_{}", name.replace(|c: char| !c.is_alphanumeric(), "_"));

        // Check if wrapper already exists
        let wrapper_fn = if let Some(existing) = self.module.llvm_module().get_function(&wrapper_name) {
            existing
        } else {
            // Create wrapper function: (ptr env, ptr arg1, [ptr arg2]) -> ptr
            let mut param_types: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> = Vec::new();
            param_types.push(ptr_type.into()); // env/closure pointer
            for _ in 0..arity {
                param_types.push(ptr_type.into());
            }

            let wrapper_fn_type = ptr_type.fn_type(&param_types, false);
            let wrapper_fn = self.module.llvm_module().add_function(&wrapper_name, wrapper_fn_type, None);

            // Build the wrapper function body
            let entry_bb = self.llvm_ctx.append_basic_block(wrapper_fn, "entry");
            let current_bb = self.builder().get_insert_block();

            self.builder().position_at_end(entry_bb);

            // Load arguments (skip env at index 0)
            let args: Vec<BasicValueEnum<'ctx>> = (1..=arity)
                .map(|i| wrapper_fn.get_nth_param(i).unwrap())
                .collect();

            // Perform the primop
            let result = self.lower_primop_direct(primop, &args)?;

            // Box the result and return
            let result_ptr = self.value_to_ptr(result)?;
            self.builder()
                .build_return(Some(&result_ptr))
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;

            // Restore insertion point
            if let Some(bb) = current_bb {
                self.builder().position_at_end(bb);
            }

            wrapper_fn
        };

        // Create a closure wrapping the primop function
        let fn_ptr = wrapper_fn.as_global_value().as_pointer_value();
        let closure_ptr = self.alloc_closure(fn_ptr, &[])?;
        Ok(Some(closure_ptr.into()))
    }

    /// Create a closure wrapping a builtin operation.
    /// This is used when a builtin like (>>) is used as a first-class value.
    fn create_builtin_closure(
        &mut self,
        name: &str,
        arity: u32,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();

        // Create a unique name for the wrapper function
        let wrapper_name = format!("builtin_wrapper_{}", name.replace(|c: char| !c.is_alphanumeric(), "_"));

        // Check if wrapper already exists
        let wrapper_fn = if let Some(existing) = self.module.llvm_module().get_function(&wrapper_name) {
            existing
        } else {
            // Create wrapper function: (ptr env, ptr arg1, ptr arg2, ...) -> ptr
            let mut param_types: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> = Vec::new();
            param_types.push(ptr_type.into()); // env/closure pointer
            for _ in 0..arity {
                param_types.push(ptr_type.into());
            }

            let wrapper_fn_type = ptr_type.fn_type(&param_types, false);
            let wrapper_fn = self.module.llvm_module().add_function(&wrapper_name, wrapper_fn_type, None);

            // Build the wrapper function body
            let entry_bb = self.llvm_ctx.append_basic_block(wrapper_fn, "entry");
            let current_bb = self.builder().get_insert_block();

            self.builder().position_at_end(entry_bb);

            // Load arguments (skip env at index 0)
            let args: Vec<BasicValueEnum<'ctx>> = (1..=arity)
                .map(|i| wrapper_fn.get_nth_param(i).unwrap())
                .collect();

            // Perform the builtin operation
            let result = self.lower_builtin_direct(name, &args)?;

            // Return the result
            let result_ptr = match result {
                Some(v) => v.into_pointer_value(),
                None => ptr_type.const_null(), // Unit/void result
            };
            self.builder()
                .build_return(Some(&result_ptr))
                .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;

            // Restore insertion point
            if let Some(bb) = current_bb {
                self.builder().position_at_end(bb);
            }

            wrapper_fn
        };

        // Create a closure wrapping the builtin function
        let fn_ptr = wrapper_fn.as_global_value().as_pointer_value();
        let closure_ptr = self.alloc_closure(fn_ptr, &[])?;
        Ok(Some(closure_ptr.into()))
    }

    /// Execute a builtin operation directly on LLVM values (already lowered).
    /// This is for when builtins are used as values and receive runtime arguments.
    fn lower_builtin_direct(
        &mut self,
        name: &str,
        args: &[BasicValueEnum<'ctx>],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let ptr_type = self.type_mapper().ptr_type();

        match name {
            ">>" => {
                // (>>) :: m a -> m b -> m b
                // Execute first action (arg1), ignore result, return second (arg2)
                // For our simple model, arg1 and arg2 are thunks/values
                // We just evaluate both and return the second
                // Since args are already evaluated values, we just return arg2
                // Note: If these were actual IO thunks, we'd need to force them
                Ok(Some(args[1]))
            }
            ">>=" => {
                // (>>=) :: m a -> (a -> m b) -> m b
                // Execute first action, pass result to function, return result of function
                let action_result = args[0]; // Result of first action
                let func = args[1].into_pointer_value(); // Function closure

                // Call the function with the action result
                let fn_ptr = self.extract_closure_fn_ptr(func)?;
                let fn_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
                let result = self.builder()
                    .build_indirect_call(fn_type, fn_ptr, &[func.into(), action_result.into()], "bind_result")
                    .map_err(|e| CodegenError::Internal(format!("failed to call bind function: {:?}", e)))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal(">>=: function returned void".to_string()))?;
                Ok(Some(result))
            }
            "=<<" => {
                // (=<<) :: (a -> m b) -> m a -> m b (flipped >>=)
                let func = args[0].into_pointer_value();
                let action_result = args[1];

                let fn_ptr = self.extract_closure_fn_ptr(func)?;
                let fn_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
                let result = self.builder()
                    .build_indirect_call(fn_type, fn_ptr, &[func.into(), action_result.into()], "bind_result")
                    .map_err(|e| CodegenError::Internal(format!("failed to call bind function: {:?}", e)))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError::Internal("=<<: function returned void".to_string()))?;
                Ok(Some(result))
            }
            "return" | "pure" => {
                // return :: a -> m a
                // For our simple IO model, just return the value
                Ok(Some(args[0]))
            }
            _ => Err(CodegenError::Internal(format!(
                "lower_builtin_direct: unhandled builtin '{}'",
                name
            ))),
        }
    }

    /// Execute a primitive operation directly on LLVM values.
    fn lower_primop_direct(
        &mut self,
        op: PrimOp,
        args: &[BasicValueEnum<'ctx>],
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let tm = self.type_mapper();

        match op {
            // Binary arithmetic operations
            PrimOp::Add | PrimOp::Sub | PrimOp::Mul | PrimOp::Div |
            PrimOp::Mod | PrimOp::Rem | PrimOp::Quot => {
                // Unbox arguments
                let lhs = self.ptr_to_int(args[0].into_pointer_value())?;
                let rhs = self.ptr_to_int(args[1].into_pointer_value())?;

                let result = match op {
                    PrimOp::Add => self.builder().build_int_add(lhs, rhs, "add"),
                    PrimOp::Sub => self.builder().build_int_sub(lhs, rhs, "sub"),
                    PrimOp::Mul => self.builder().build_int_mul(lhs, rhs, "mul"),
                    PrimOp::Div | PrimOp::Quot => self.builder().build_int_signed_div(lhs, rhs, "div"),
                    PrimOp::Mod | PrimOp::Rem => self.builder().build_int_signed_rem(lhs, rhs, "rem"),
                    _ => unreachable!(),
                }.map_err(|e| CodegenError::Internal(format!("failed to build arithmetic: {:?}", e)))?;

                Ok(result.into())
            }

            // Comparison operations
            PrimOp::Eq | PrimOp::Ne | PrimOp::Lt | PrimOp::Le |
            PrimOp::Gt | PrimOp::Ge => {
                let lhs = self.ptr_to_int(args[0].into_pointer_value())?;
                let rhs = self.ptr_to_int(args[1].into_pointer_value())?;

                let predicate = match op {
                    PrimOp::Eq => inkwell::IntPredicate::EQ,
                    PrimOp::Ne => inkwell::IntPredicate::NE,
                    PrimOp::Lt => inkwell::IntPredicate::SLT,
                    PrimOp::Le => inkwell::IntPredicate::SLE,
                    PrimOp::Gt => inkwell::IntPredicate::SGT,
                    PrimOp::Ge => inkwell::IntPredicate::SGE,
                    _ => unreachable!(),
                };

                let cmp = self.builder()
                    .build_int_compare(predicate, lhs, rhs, "cmp")
                    .map_err(|e| CodegenError::Internal(format!("failed to build compare: {:?}", e)))?;

                let result = self.builder()
                    .build_int_z_extend(cmp, tm.i64_type(), "cmp_ext")
                    .map_err(|e| CodegenError::Internal(format!("failed to extend: {:?}", e)))?;

                Ok(result.into())
            }

            // Boolean operations
            PrimOp::And => {
                let lhs = self.ptr_to_int(args[0].into_pointer_value())?;
                let rhs = self.ptr_to_int(args[1].into_pointer_value())?;
                let result = self.builder()
                    .build_and(lhs, rhs, "and")
                    .map_err(|e| CodegenError::Internal(format!("failed to build and: {:?}", e)))?;
                Ok(result.into())
            }

            PrimOp::Or => {
                let lhs = self.ptr_to_int(args[0].into_pointer_value())?;
                let rhs = self.ptr_to_int(args[1].into_pointer_value())?;
                let result = self.builder()
                    .build_or(lhs, rhs, "or")
                    .map_err(|e| CodegenError::Internal(format!("failed to build or: {:?}", e)))?;
                Ok(result.into())
            }

            // Unary operations
            PrimOp::Negate => {
                let val = self.ptr_to_int(args[0].into_pointer_value())?;
                let result = self.builder()
                    .build_int_neg(val, "neg")
                    .map_err(|e| CodegenError::Internal(format!("failed to build neg: {:?}", e)))?;
                Ok(result.into())
            }

            PrimOp::Abs => {
                let val = self.ptr_to_int(args[0].into_pointer_value())?;
                let neg = self.builder()
                    .build_int_neg(val, "neg")
                    .map_err(|e| CodegenError::Internal(format!("failed to build neg: {:?}", e)))?;
                let is_neg = self.builder()
                    .build_int_compare(inkwell::IntPredicate::SLT, val, tm.i64_type().const_zero(), "is_neg")
                    .map_err(|e| CodegenError::Internal(format!("failed to build compare: {:?}", e)))?;
                let result = self.builder()
                    .build_select(is_neg, neg, val, "abs")
                    .map_err(|e| CodegenError::Internal(format!("failed to build select: {:?}", e)))?;
                Ok(result)
            }

            PrimOp::Signum => {
                let val = self.ptr_to_int(args[0].into_pointer_value())?;
                let zero = tm.i64_type().const_zero();
                let one = tm.i64_type().const_int(1, false);
                let neg_one = tm.i64_type().const_int(-1i64 as u64, true);

                let is_neg = self.builder()
                    .build_int_compare(inkwell::IntPredicate::SLT, val, zero, "is_neg")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                let is_pos = self.builder()
                    .build_int_compare(inkwell::IntPredicate::SGT, val, zero, "is_pos")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;

                let tmp = self.builder()
                    .build_select(is_neg, neg_one, zero, "tmp")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                let result = self.builder()
                    .build_select(is_pos, one, tmp.into_int_value(), "signum")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                Ok(result)
            }

            PrimOp::Not => {
                let val = self.ptr_to_int(args[0].into_pointer_value())?;
                let is_zero = self.builder()
                    .build_int_compare(inkwell::IntPredicate::EQ, val, tm.i64_type().const_zero(), "is_zero")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                let result = self.builder()
                    .build_int_z_extend(is_zero, tm.i64_type(), "not")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                Ok(result.into())
            }

            // Bitwise operations
            PrimOp::BitAnd => {
                let lhs = self.ptr_to_int(args[0].into_pointer_value())?;
                let rhs = self.ptr_to_int(args[1].into_pointer_value())?;
                let result = self.builder()
                    .build_and(lhs, rhs, "bitand")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                Ok(result.into())
            }

            PrimOp::BitOr => {
                let lhs = self.ptr_to_int(args[0].into_pointer_value())?;
                let rhs = self.ptr_to_int(args[1].into_pointer_value())?;
                let result = self.builder()
                    .build_or(lhs, rhs, "bitor")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                Ok(result.into())
            }

            PrimOp::BitXor => {
                let lhs = self.ptr_to_int(args[0].into_pointer_value())?;
                let rhs = self.ptr_to_int(args[1].into_pointer_value())?;
                let result = self.builder()
                    .build_xor(lhs, rhs, "bitxor")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                Ok(result.into())
            }

            PrimOp::ShiftL => {
                let val = self.ptr_to_int(args[0].into_pointer_value())?;
                let amt = self.ptr_to_int(args[1].into_pointer_value())?;
                let result = self.builder()
                    .build_left_shift(val, amt, "shl")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                Ok(result.into())
            }

            PrimOp::ShiftR => {
                let val = self.ptr_to_int(args[0].into_pointer_value())?;
                let amt = self.ptr_to_int(args[1].into_pointer_value())?;
                let result = self.builder()
                    .build_right_shift(val, amt, true, "shr")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                Ok(result.into())
            }

            PrimOp::Complement => {
                let val = self.ptr_to_int(args[0].into_pointer_value())?;
                let result = self.builder()
                    .build_not(val, "complement")
                    .map_err(|e| CodegenError::Internal(format!("failed: {:?}", e)))?;
                Ok(result.into())
            }
        }
    }

    /// Convert a pointer to an integer (unbox).
    fn ptr_to_int(&self, ptr: PointerValue<'ctx>) -> CodegenResult<IntValue<'ctx>> {
        self.builder()
            .build_ptr_to_int(ptr, self.type_mapper().i64_type(), "unbox")
            .map_err(|e| CodegenError::Internal(format!("failed to unbox: {:?}", e)))
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
                // Check if we're comparing lists - need special handling
                // Since type info may be Error, check structurally if expr is a list
                let is_list = Self::is_list_expr(args[0]);
                if is_list {
                    // List comparison - dispatch to builtin list comparison
                    let lhs = self.lower_expr(args[0])?.ok_or_else(|| {
                        CodegenError::Internal("primop arg has no value".to_string())
                    })?;
                    let rhs = self.lower_expr(args[1])?.ok_or_else(|| {
                        CodegenError::Internal("primop arg has no value".to_string())
                    })?;
                    return self.lower_list_comparison(op, lhs, rhs);
                }

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

    /// Unbox a value to an integer if it's a pointer (boxed int).
    fn unbox_to_int(&self, val: BasicValueEnum<'ctx>) -> CodegenResult<inkwell::values::IntValue<'ctx>> {
        match val {
            BasicValueEnum::IntValue(i) => Ok(i),
            BasicValueEnum::PointerValue(p) => {
                // Unbox: pointer stores int value directly using ptr_to_int
                self.builder()
                    .build_ptr_to_int(p, self.type_mapper().i64_type(), "unbox_int")
                    .map_err(|e| CodegenError::Internal(format!("failed to unbox int: {:?}", e)))
            }
            _ => Err(CodegenError::TypeError("expected int or boxed int".to_string())),
        }
    }

    /// Box an integer to a pointer.
    fn box_int(&self, val: inkwell::values::IntValue<'ctx>) -> CodegenResult<inkwell::values::PointerValue<'ctx>> {
        self.builder()
            .build_int_to_ptr(val, self.type_mapper().ptr_type(), "box_int")
            .map_err(|e| CodegenError::Internal(format!("failed to box int: {:?}", e)))
    }

    /// Lower a binary arithmetic operation.
    fn lower_binary_arith(
        &self,
        op: PrimOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Check if we're dealing with integers or floats
        // Handle boxed values (pointers) by unboxing first
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
            // Handle boxed integers (pointers) - unbox, compute, rebox
            (BasicValueEnum::PointerValue(_), _) | (_, BasicValueEnum::PointerValue(_)) => {
                let l = self.unbox_to_int(lhs)?;
                let r = self.unbox_to_int(rhs)?;
                let result = match op {
                    PrimOp::Add => self.builder().build_int_add(l, r, "add"),
                    PrimOp::Sub => self.builder().build_int_sub(l, r, "sub"),
                    PrimOp::Mul => self.builder().build_int_mul(l, r, "mul"),
                    PrimOp::Div | PrimOp::Quot => self.builder().build_int_signed_div(l, r, "div"),
                    PrimOp::Mod | PrimOp::Rem => self.builder().build_int_signed_rem(l, r, "rem"),
                    _ => return Err(CodegenError::Internal("invalid arith op".to_string())),
                };
                let int_result = result
                    .map_err(|e| CodegenError::Internal(format!("failed to build int op: {:?}", e)))?;
                // Box the result back to pointer for uniform calling convention
                let boxed = self.box_int(int_result)?;
                Ok(Some(boxed.into()))
            }
            _ => Err(CodegenError::TypeError(
                "arithmetic operations require matching numeric types".to_string(),
            )),
        }
    }

    /// Lower a list comparison operation (== or /= for lists).
    /// This generates a loop that compares elements one by one.
    fn lower_list_comparison(
        &mut self,
        op: PrimOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let i64_type = tm.i64_type();
        let current_fn = self.builder().get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::Internal("no current function".to_string()))?;

        // Create basic blocks
        let entry_block = self.builder().get_insert_block().ok_or_else(|| {
            CodegenError::Internal("no current block".to_string())
        })?;
        let loop_block = self.llvm_context().append_basic_block(current_fn, "list_cmp_loop");
        let both_nil_block = self.llvm_context().append_basic_block(current_fn, "list_cmp_both_nil");
        let one_nil_block = self.llvm_context().append_basic_block(current_fn, "list_cmp_one_nil");
        let compare_heads_block = self.llvm_context().append_basic_block(current_fn, "list_cmp_heads");
        let heads_equal_block = self.llvm_context().append_basic_block(current_fn, "list_cmp_heads_eq");
        let done_block = self.llvm_context().append_basic_block(current_fn, "list_cmp_done");

        let lhs_ptr = lhs.into_pointer_value();
        let rhs_ptr = rhs.into_pointer_value();

        // Jump to loop
        self.builder().build_unconditional_branch(loop_block)
            .map_err(|e| CodegenError::Internal(format!("failed to branch: {:?}", e)))?;

        // Loop block: check tags
        self.builder().position_at_end(loop_block);
        let list1_phi = self.builder().build_phi(ptr_type, "list1")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        let list2_phi = self.builder().build_phi(ptr_type, "list2")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        list1_phi.add_incoming(&[(&lhs_ptr, entry_block)]);
        list2_phi.add_incoming(&[(&rhs_ptr, entry_block)]);

        let list1_ptr = list1_phi.as_basic_value().into_pointer_value();
        let list2_ptr = list2_phi.as_basic_value().into_pointer_value();

        // Get tags
        let tag1 = self.builder()
            .build_load(i64_type, list1_ptr, "tag1")
            .map_err(|e| CodegenError::Internal(format!("failed to load tag1: {:?}", e)))?
            .into_int_value();
        let tag2 = self.builder()
            .build_load(i64_type, list2_ptr, "tag2")
            .map_err(|e| CodegenError::Internal(format!("failed to load tag2: {:?}", e)))?
            .into_int_value();

        let zero = i64_type.const_int(0, false);
        let is_nil1 = self.builder().build_int_compare(inkwell::IntPredicate::EQ, tag1, zero, "is_nil1")
            .map_err(|e| CodegenError::Internal(format!("failed to build cmp: {:?}", e)))?;
        let is_nil2 = self.builder().build_int_compare(inkwell::IntPredicate::EQ, tag2, zero, "is_nil2")
            .map_err(|e| CodegenError::Internal(format!("failed to build cmp: {:?}", e)))?;

        // Check if both are nil
        let both_nil = self.builder().build_and(is_nil1, is_nil2, "both_nil")
            .map_err(|e| CodegenError::Internal(format!("failed to build and: {:?}", e)))?;
        self.builder().build_conditional_branch(both_nil, both_nil_block, one_nil_block)
            .map_err(|e| CodegenError::Internal(format!("failed to branch: {:?}", e)))?;

        // Both nil: lists are equal
        self.builder().position_at_end(both_nil_block);
        let true_val = i64_type.const_int(1, false);
        self.builder().build_unconditional_branch(done_block)
            .map_err(|e| CodegenError::Internal(format!("failed to branch: {:?}", e)))?;

        // One nil (but not both): check if either is nil
        self.builder().position_at_end(one_nil_block);
        let either_nil = self.builder().build_or(is_nil1, is_nil2, "either_nil")
            .map_err(|e| CodegenError::Internal(format!("failed to build or: {:?}", e)))?;
        let false_val = i64_type.const_int(0, false);
        // If either is nil (but not both), lists are not equal
        self.builder().build_conditional_branch(either_nil, done_block, compare_heads_block)
            .map_err(|e| CodegenError::Internal(format!("failed to branch: {:?}", e)))?;

        // Compare heads - use extract_adt_field which handles the ADT layout
        self.builder().position_at_end(compare_heads_block);
        let head1 = self.extract_adt_field(list1_ptr, 2, 0)?; // arity=2 (Cons), field=0 (head)
        let head2 = self.extract_adt_field(list2_ptr, 2, 0)?;

        // Compare heads as integers (works for primitive types like Int)
        let head1_int = self.builder()
            .build_ptr_to_int(head1, i64_type, "head1_int")
            .map_err(|e| CodegenError::Internal(format!("failed to ptr_to_int: {:?}", e)))?;
        let head2_int = self.builder()
            .build_ptr_to_int(head2, i64_type, "head2_int")
            .map_err(|e| CodegenError::Internal(format!("failed to ptr_to_int: {:?}", e)))?;
        let heads_equal = self.builder()
            .build_int_compare(inkwell::IntPredicate::EQ, head1_int, head2_int, "heads_eq")
            .map_err(|e| CodegenError::Internal(format!("failed to cmp: {:?}", e)))?;

        // If heads not equal, lists not equal
        self.builder().build_conditional_branch(heads_equal, heads_equal_block, done_block)
            .map_err(|e| CodegenError::Internal(format!("failed to branch: {:?}", e)))?;

        // Heads equal - continue with tails
        self.builder().position_at_end(heads_equal_block);
        let tail1 = self.extract_adt_field(list1_ptr, 2, 1)?; // arity=2 (Cons), field=1 (tail)
        let tail2 = self.extract_adt_field(list2_ptr, 2, 1)?;

        // Update phis and loop back
        list1_phi.add_incoming(&[(&tail1, heads_equal_block)]);
        list2_phi.add_incoming(&[(&tail2, heads_equal_block)]);
        self.builder().build_unconditional_branch(loop_block)
            .map_err(|e| CodegenError::Internal(format!("failed to branch: {:?}", e)))?;

        // Done block: collect result
        self.builder().position_at_end(done_block);
        let result_phi = self.builder().build_phi(i64_type, "list_cmp_result")
            .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;
        result_phi.add_incoming(&[(&true_val, both_nil_block)]);  // Both nil = equal
        result_phi.add_incoming(&[(&false_val, one_nil_block)]); // One nil = not equal
        result_phi.add_incoming(&[(&false_val, compare_heads_block)]); // Heads not equal

        let result = result_phi.as_basic_value().into_int_value();

        // For /= we need to invert the result
        let final_result = if matches!(op, PrimOp::Ne) {
            let one = i64_type.const_int(1, false);
            self.builder().build_xor(result, one, "invert")
                .map_err(|e| CodegenError::Internal(format!("failed to xor: {:?}", e)))?
        } else {
            result
        };

        Ok(Some(final_result.into()))
    }

    /// Lower a comparison operation.
    fn lower_comparison(
        &self,
        op: PrimOp,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Helper to do int comparison and return boxed result
        let do_int_cmp = |this: &Self, l: inkwell::values::IntValue<'ctx>, r: inkwell::values::IntValue<'ctx>| -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
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
            let cmp = this.builder()
                .build_int_compare(pred, l, r, "cmp")
                .map_err(|e| CodegenError::Internal(format!("failed to build int cmp: {:?}", e)))?;

            // Convert i1 to i64 (0 or 1) for consistency with our Bool representation
            let result = this.builder()
                .build_int_z_extend(cmp, this.type_mapper().i64_type(), "cmp_ext")
                .map_err(|e| CodegenError::Internal(format!("failed to extend cmp: {:?}", e)))?;

            Ok(Some(result.into()))
        };

        match (lhs, rhs) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                do_int_cmp(self, l, r)
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
            // Handle boxed integers (pointers)
            (BasicValueEnum::PointerValue(_), _) | (_, BasicValueEnum::PointerValue(_)) => {
                let l = self.unbox_to_int(lhs)?;
                let r = self.unbox_to_int(rhs)?;
                do_int_cmp(self, l, r)
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
        // Get the function's expected parameter count (excluding env pointer)
        let fn_type = fn_val.get_type();
        let expected_params = fn_type.count_param_types() as usize;
        let expected_args = if expected_params > 0 { expected_params - 1 } else { 0 }; // Subtract env pointer

        // Check for over-application: more args than the function expects
        if args.len() > expected_args {
            // Split args: first part goes to this function, rest goes to the returned closure
            let (fn_args, remaining_args) = args.split_at(expected_args);

            // Call the function with its expected args
            let closure_result = self.lower_direct_call_inner(fn_val, fn_args)?;

            // The result should be a closure - call it with remaining args
            if let Some(closure_val) = closure_result {
                return self.lower_closure_call(closure_val, remaining_args);
            } else {
                return Err(CodegenError::Internal(
                    "function returned no value for over-application".to_string(),
                ));
            }
        }

        // Check for under-application: fewer args than the function expects
        if args.len() < expected_args {
            return self.lower_partial_application(fn_val, args, expected_args);
        }

        // Exact application
        self.lower_direct_call_inner(fn_val, args)
    }

    /// Lower a partial application (under-application) to a PAP closure.
    ///
    /// Creates a PAP wrapper function that takes the remaining args and calls
    /// the original function with all args combined.
    fn lower_partial_application(
        &mut self,
        fn_val: FunctionValue<'ctx>,
        applied_args: &[&Expr],
        expected_args: usize,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();
        let i64_type = tm.i64_type();

        let num_applied = applied_args.len();
        let num_remaining = expected_args - num_applied;

        // Lower the applied arguments
        let was_tail = self.in_tail_position;
        self.in_tail_position = false;

        let mut applied_vals: Vec<(VarId, BasicValueEnum<'ctx>)> = Vec::new();
        for (i, arg_expr) in applied_args.iter().enumerate() {
            if let Some(val) = self.lower_expr(arg_expr)? {
                // Use a dummy VarId for PAP captured args
                applied_vals.push((VarId::new(10000 + i), val));
            }
        }

        self.in_tail_position = was_tail;

        // Create the PAP wrapper function
        // Signature: (ptr env, ptr arg1, ptr arg2, ...) -> ptr
        let fn_name = fn_val.get_name().to_str().unwrap_or("fn");
        let wrapper_name = format!("pap_{}_{}", fn_name, num_applied);

        // Check if wrapper already exists
        let wrapper_fn = if let Some(existing) = self.module.llvm_module().get_function(&wrapper_name) {
            existing
        } else {
            // Create param types: env + remaining args
            let mut wrapper_param_types: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> = Vec::new();
            wrapper_param_types.push(ptr_type.into()); // env/closure pointer
            for _ in 0..num_remaining {
                wrapper_param_types.push(ptr_type.into());
            }

            let wrapper_fn_type = ptr_type.fn_type(&wrapper_param_types, false);
            let wrapper_fn = self.module.llvm_module().add_function(&wrapper_name, wrapper_fn_type, None);

            // Build the wrapper function body
            let entry_bb = self.llvm_ctx.append_basic_block(wrapper_fn, "entry");
            let current_bb = self.builder().get_insert_block();

            self.builder().position_at_end(entry_bb);

            // Extract applied args from closure environment
            let closure_ptr = wrapper_fn.get_first_param().unwrap().into_pointer_value();
            let closure_ty = self.closure_type(num_applied as u32);

            let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum<'ctx>> = Vec::new();
            // Original function expects null env for direct calls
            call_args.push(ptr_type.const_null().into());

            // Load applied args from env
            for i in 0..num_applied {
                let env_slot = self
                    .builder()
                    .build_struct_gep(closure_ty, closure_ptr, 2, "env_slot")
                    .map_err(|e| CodegenError::Internal(format!("PAP gep failed: {:?}", e)))?;

                let elem_ptr = unsafe {
                    self.builder()
                        .build_in_bounds_gep(
                            ptr_type.array_type(num_applied as u32),
                            env_slot,
                            &[i64_type.const_zero(), i64_type.const_int(i as u64, false)],
                            &format!("pap_arg_{}", i),
                        )
                        .map_err(|e| CodegenError::Internal(format!("PAP elem gep failed: {:?}", e)))?
                };

                let arg_val = self
                    .builder()
                    .build_load(ptr_type, elem_ptr, &format!("pap_load_{}", i))
                    .map_err(|e| CodegenError::Internal(format!("PAP load failed: {:?}", e)))?;

                call_args.push(arg_val.into());
            }

            // Add remaining args from wrapper params
            for i in 0..num_remaining {
                let param = wrapper_fn.get_nth_param((i + 1) as u32).unwrap(); // +1 to skip env
                call_args.push(param.into());
            }

            // Call the original function
            let result = self
                .builder()
                .build_call(fn_val, &call_args, "pap_call")
                .map_err(|e| CodegenError::Internal(format!("PAP call failed: {:?}", e)))?
                .try_as_basic_value()
                .basic();

            // Return the result
            if let Some(ret_val) = result {
                self.builder()
                    .build_return(Some(&ret_val))
                    .map_err(|e| CodegenError::Internal(format!("PAP return failed: {:?}", e)))?;
            } else {
                self.builder()
                    .build_return(Some(&ptr_type.const_null()))
                    .map_err(|e| CodegenError::Internal(format!("PAP return failed: {:?}", e)))?;
            }

            // Restore insertion point
            if let Some(bb) = current_bb {
                self.builder().position_at_end(bb);
            }

            wrapper_fn
        };

        // Create closure pointing to the wrapper with applied args as env
        let wrapper_ptr = wrapper_fn.as_global_value().as_pointer_value();
        let closure_ptr = self.alloc_closure(wrapper_ptr, &applied_vals)?;

        Ok(Some(closure_ptr.into()))
    }

    /// Inner implementation of direct call (doesn't handle over-application).
    fn lower_direct_call_inner(
        &mut self,
        fn_val: FunctionValue<'ctx>,
        args: &[&Expr],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Arguments are not in tail position
        let was_tail = self.in_tail_position;
        self.in_tail_position = false;

        // All functions take (env_ptr, args...) for uniform calling convention
        let mut llvm_args = Vec::new();
        // First arg is env/closure pointer (null for direct calls)
        let null_env = self.type_mapper().ptr_type().const_null();
        llvm_args.push(null_env.into());

        // Lower remaining arguments and convert to pointers
        for arg_expr in args {
            if let Some(val) = self.lower_expr(arg_expr)? {
                // Box non-pointer values to pointers for uniform calling convention
                let ptr_val = self.value_to_ptr(val)?;
                llvm_args.push(ptr_val.into());
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

        Ok(call.try_as_basic_value().basic())
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

        Ok(call.try_as_basic_value().basic())
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
                // Lower the right-hand side (NOT in tail position)
                let was_tail = self.in_tail_position;
                self.in_tail_position = false;
                let rhs_result = self.lower_expr(rhs.as_ref())?;
                self.in_tail_position = was_tail;

                if let Some(val) = rhs_result {
                    // Bind the variable
                    self.env.insert(var.id, val);
                }

                // Lower the body (preserves tail position from parent)
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

        // Check if the current block already has a terminator (e.g., from `error` or `unreachable`)
        // If so, don't add another terminator
        let current_block = self.builder().get_insert_block();
        let has_terminator = current_block
            .map(|bb| bb.get_terminator().is_some())
            .unwrap_or(false);

        if !has_terminator {
            // Build return - convert to pointer if return type is pointer (uniform calling convention)
            let ret_type = fn_val.get_type().get_return_type();
            if let Some(val) = result {
                let ret_val: BasicValueEnum<'ctx> = if ret_type == Some(self.type_mapper().ptr_type().into()) {
                    self.value_to_ptr(val)?.into()
                } else {
                    val
                };
                self.builder()
                    .build_return(Some(&ret_val))
                    .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
            } else {
                self.builder()
                    .build_return(None)
                    .map_err(|e| CodegenError::Internal(format!("failed to build return: {:?}", e)))?;
            }
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
        // Note: param_idx starts at 1 because index 0 is the env/closure pointer
        let mut current = expr;
        let mut param_idx = 1;

        // Unwrap lambdas, type lambdas, and trivial case expressions.
        // - Lambdas (Lam): bind parameter to function argument
        // - Type lambdas (TyLam): erased at runtime, skip
        // - Trivial Case: pattern matching on a variable we just bound, rebind and continue
        loop {
            match current {
                Expr::Lam(param, body, _span) => {
                    // Term lambda - bind parameter to function argument
                    if let Some(arg) = fn_val.get_nth_param(param_idx) {
                        self.env.insert(param.id, arg);
                    }
                    param_idx += 1;
                    current = body.as_ref();
                }
                Expr::TyLam(_tyvar, body, _span) => {
                    // Type lambda - skip (erased at runtime)
                    current = body.as_ref();
                }
                Expr::Case(scrut, alts, _ty, _span) => {
                    // Check if this is a "trivial" case that just rebinds a variable.
                    // This pattern comes from HIR->Core lowering for pattern matching
                    // on function parameters.
                    //
                    // Pattern: case x of { _ -> body } or case x of { y -> body }
                    // where x is a variable we already have bound AND all alternatives are Default.
                    //
                    // IMPORTANT: Only optimize when ALL alternatives are Default.
                    // If there are any literal or constructor patterns, we must do real pattern matching.
                    let all_default = alts.iter().all(|a| matches!(a.con, AltCon::Default));
                    if all_default {
                        if let Expr::Var(scrut_var, _) = scrut.as_ref() {
                            // Use the first Default alternative
                            if let Some(alt) = alts.first() {
                                // If the alternative has a binder, bind it to the same value as scrut
                                for binder in &alt.binders {
                                    if let Some(val) = self.env.get(&scrut_var.id) {
                                        self.env.insert(binder.id, *val);
                                    }
                                }
                                current = &alt.rhs;
                                continue;
                            }
                        }
                    }
                    // Not a trivial case, stop unwrapping
                    break;
                }
                _ => break,
            }
        }

        // Check if there are remaining LLVM parameters that weren't bound to lambdas.
        // This happens for definitions like `add5 = add 5` where the body is not a lambda
        // but the type implies it takes arguments. We need to eta-expand at codegen.
        let total_params = fn_val.count_params() as u32;
        let remaining_params: Vec<_> = (param_idx..total_params)
            .filter_map(|i| fn_val.get_nth_param(i))
            .collect();

        // Lower the body
        let was_tail = self.in_tail_position;
        self.in_tail_position = remaining_params.is_empty(); // Only tail if no eta-expansion needed
        let result = self.lower_expr(current)?;
        self.in_tail_position = was_tail;

        // If there are remaining parameters, the body should evaluate to a closure.
        // Apply the remaining parameters to it (eta-expansion).
        if !remaining_params.is_empty() {
            if let Some(closure_val) = result {
                // The result should be a closure - call it with remaining params
                let tm = self.type_mapper();
                let ptr_type = tm.ptr_type();

                let closure_ptr = match closure_val {
                    BasicValueEnum::PointerValue(p) => p,
                    _ => {
                        return Err(CodegenError::Internal(
                            "eta-expansion: expected closure but got non-pointer".to_string(),
                        ))
                    }
                };

                // Extract function pointer from closure
                let fn_ptr = self.extract_closure_fn_ptr(closure_ptr)?;

                // Build function type for the call
                let mut param_types: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> = Vec::new();
                param_types.push(ptr_type.into()); // closure/env pointer
                for _ in &remaining_params {
                    param_types.push(ptr_type.into());
                }
                let fn_type = ptr_type.fn_type(&param_types, false);

                // Build args: closure ptr + remaining params
                let mut llvm_args: Vec<inkwell::values::BasicMetadataValueEnum<'ctx>> = Vec::new();
                llvm_args.push(closure_ptr.into());
                for param in &remaining_params {
                    // Convert to pointer if needed
                    let ptr_val = self.value_to_ptr(*param)?;
                    llvm_args.push(ptr_val.into());
                }

                // Build indirect call
                let call = self
                    .builder()
                    .build_indirect_call(fn_type, fn_ptr, &llvm_args, "eta_call")
                    .map_err(|e| {
                        CodegenError::Internal(format!("failed to build eta-expansion call: {:?}", e))
                    })?;

                return Ok(call.try_as_basic_value().basic());
            } else {
                return Err(CodegenError::Internal(
                    "eta-expansion: body has no value".to_string(),
                ));
            }
        }

        Ok(result)
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
        // Scrutinee is NOT in tail position
        let was_tail = self.in_tail_position;
        self.in_tail_position = false;
        let scrut_val = self.lower_expr(scrut)?.ok_or_else(|| {
            CodegenError::Internal("scrutinee has no value".to_string())
        })?;
        self.in_tail_position = was_tail;

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

            // Lower the RHS (inherits tail position from parent case)
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

        if is_bool_case {
            // Bool case - convert scrutinee to integer if needed
            let int_scrut = match scrut_val {
                BasicValueEnum::IntValue(_) => scrut_val,
                BasicValueEnum::PointerValue(p) => {
                    // Boxed boolean - load the tag from the ADT
                    // Bool is arity 0, so ADT structure is { i64 tag, [0 x ptr] }
                    let adt_ty = self.adt_type(0);
                    let tag_ptr = self.builder()
                        .build_struct_gep(adt_ty, p, 0, "bool_tag_ptr")
                        .map_err(|e| CodegenError::Internal(format!("failed to get bool tag ptr: {:?}", e)))?;
                    let tag_val = self.builder()
                        .build_load(self.type_mapper().i64_type(), tag_ptr, "bool_tag")
                        .map_err(|e| CodegenError::Internal(format!("failed to load bool tag: {:?}", e)))?;
                    tag_val
                }
                _ => scrut_val, // Fall through to datacon handling
            };
            if matches!(int_scrut, BasicValueEnum::IntValue(_)) {
                return self.lower_case_bool_as_int(int_scrut, alts);
            }
        }

        // Get the scrutinee's type for determining binder types
        let scrut_ty = scrut.ty();

        if has_datacon {
            self.lower_case_datacon(scrut_val, alts, &scrut_ty)
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
        // Determine the type of literals in the alternatives
        let has_int_lit = alts.iter().any(|alt| matches!(&alt.con, AltCon::Lit(Literal::Int(_))));
        let has_char_lit = alts.iter().any(|alt| matches!(&alt.con, AltCon::Lit(Literal::Char(_))));
        let has_float_lit = alts.iter().any(|alt| matches!(&alt.con, AltCon::Lit(Literal::Double(_))));
        let has_string_lit = alts.iter().any(|alt| matches!(&alt.con, AltCon::Lit(Literal::String(_))));

        // Dispatch based on scrutinee type
        match scrut_val {
            BasicValueEnum::IntValue(i) => self.lower_case_literal_int(i, alts),
            BasicValueEnum::FloatValue(f) => self.lower_case_literal_float(f, alts),
            BasicValueEnum::PointerValue(p) => {
                // Check if the alternatives have integer/char literals - if so, unbox the pointer
                if has_int_lit || has_char_lit {
                    // Pointer contains a boxed integer - unbox it
                    let int_val = self.builder()
                        .build_ptr_to_int(p, self.type_mapper().i64_type(), "unbox_for_case")
                        .map_err(|e| CodegenError::Internal(format!("failed to unbox for case: {:?}", e)))?;
                    self.lower_case_literal_int(int_val, alts)
                } else if has_float_lit {
                    // Pointer contains a boxed float - unbox it
                    let bits = self.builder()
                        .build_ptr_to_int(p, self.type_mapper().i64_type(), "unbox_float_bits")
                        .map_err(|e| CodegenError::Internal(format!("failed to unbox float: {:?}", e)))?;
                    let float_val = self.builder()
                        .build_bit_cast(bits, self.type_mapper().f64_type(), "to_double")
                        .map_err(|e| CodegenError::Internal(format!("failed to cast to double: {:?}", e)))?;
                    if let BasicValueEnum::FloatValue(f) = float_val {
                        self.lower_case_literal_float(f, alts)
                    } else {
                        Err(CodegenError::Internal("expected float value after unboxing".to_string()))
                    }
                } else if has_string_lit {
                    // String pattern matching
                    self.lower_case_literal_string(p, alts)
                } else {
                    // Default case or only Default alternatives - assume integer
                    let int_val = self.builder()
                        .build_ptr_to_int(p, self.type_mapper().i64_type(), "unbox_for_case")
                        .map_err(|e| CodegenError::Internal(format!("failed to unbox for case: {:?}", e)))?;
                    self.lower_case_literal_int(int_val, alts)
                }
            }
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
                    // Use the FIRST default block as the switch default.
                    // The pattern compiler adds error fallbacks as additional Defaults,
                    // but we want non-matching cases to go to the first (user-defined) default.
                    if default_block.is_none() {
                        default_block = Some(block);
                    }
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
                .basic()
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
        scrut_ty: &Ty,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Count existing case_alt blocks before we add more
        let current_fn = self.builder()
            .get_insert_block()
            .ok_or_else(|| CodegenError::Internal("no insert block".to_string()))?
            .get_parent()
            .ok_or_else(|| CodegenError::Internal("no parent function".to_string()))?;
        let existing_blocks: Vec<_> = current_fn.get_basic_block_iter().collect();

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
        let result = self.lower_case_datacon_alternatives(alts, &blocks, merge_block, scrut_ptr, &datacon_info, scrut_ty)?;

        Ok(result)
    }

    /// Lower case alternatives (shared logic for RHS generation).
    ///
    /// Uses a two-pass approach to avoid inserting instructions after terminators:
    /// 1. First pass: lower all RHS expressions and collect (value, block) pairs
    /// 2. Second pass: coerce values if needed, then build branches
    ///
    /// Each alternative's RHS is in tail position if the case expression itself is.
    fn lower_case_alternatives(
        &mut self,
        alts: &[Alt],
        blocks: &[inkwell::basic_block::BasicBlock<'ctx>],
        merge_block: inkwell::basic_block::BasicBlock<'ctx>,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Pass 1: Lower all expressions and collect values WITHOUT building branches
        // Each alternative's RHS is in tail position if the case expression is
        let mut collected: Vec<(Option<BasicValueEnum<'ctx>>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

        for (i, alt) in alts.iter().enumerate() {
            self.builder().position_at_end(blocks[i]);

            // RHS is in tail position (inherits from parent case expression)
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

            // Only add to phi_values if we will branch to merge_block
            // If the block already has a terminator (e.g., error call with unreachable),
            // it doesn't reach merge_block and shouldn't contribute to the PHI
            if block.get_terminator().is_none() {
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

                // Build the branch to merge_block
                self.builder()
                    .build_unconditional_branch(merge_block)
                    .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;
            }
            // If block already has a terminator, skip it entirely for PHI
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
        // Each branch RHS inherits tail position from parent case expression

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

        // True branch: coerce and terminate (only if block reaches merge)
        self.builder().position_at_end(true_end_block);
        if true_end_block.get_terminator().is_none() {
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
        }

        // False branch: coerce and terminate (only if block reaches merge)
        self.builder().position_at_end(false_end_block);
        if false_end_block.get_terminator().is_none() {
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
        }

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
        scrut_ty: &Ty,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Pass 1: Lower all expressions and collect values WITHOUT building branches
        // We collect (Option<value>, block) so we know which block each value came from
        let mut collected: Vec<(Option<BasicValueEnum<'ctx>>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

        for (i, alt) in alts.iter().enumerate() {
            self.builder().position_at_end(blocks[i]);

            // Extract fields and bind to pattern variables
            if let Some(con) = datacon_info[i] {
                let arity = con.arity;

                // Bind each field to its corresponding pattern variable
                for (field_idx, binder) in alt.binders.iter().enumerate() {
                    if field_idx < arity as usize {
                        let field_ptr = self.extract_adt_field(scrut_ptr, arity, field_idx as u32)?;

                        // Determine the field type:
                        // - Use binder.ty if it's not Error
                        // - Otherwise, infer from scrutinee type (e.g., for list elements)
                        let field_ty = if matches!(&binder.ty, Ty::Error) {
                            self.infer_field_type(scrut_ty, con, field_idx)
                        } else {
                            binder.ty.clone()
                        };

                        // Determine if we need to unbox the field
                        let field_val = self.ptr_to_value(field_ptr, &field_ty)?;
                        self.env.insert(binder.id, field_val);
                    }
                }
            }

            // Lower the RHS with bound variables (inherits tail position from parent case)
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

            // Only add to phi_values if we will branch to merge_block
            // If the block already has a terminator (e.g., error call with unreachable),
            // it doesn't reach merge_block and shouldn't contribute to the PHI
            if block.get_terminator().is_none() {
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

                // Build the branch to merge_block
                self.builder()
                    .build_unconditional_branch(merge_block)
                    .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;
            }
            // If block already has a terminator, skip it entirely for PHI
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

    /// Infer the type of a field from a data constructor based on the scrutinee type.
    ///
    /// This is used when pattern binders have `Ty::Error` type (which happens when
    /// HIR-to-Core lowering doesn't preserve types). We can infer the field types
    /// from the scrutinee type.
    ///
    /// For lists:
    /// - Cons `:` (tag 1, arity 2): field 0 is element type, field 1 is list type
    /// - Nil `[]` (tag 0, arity 0): no fields
    fn infer_field_type(&self, scrut_ty: &Ty, con: &DataCon, field_idx: usize) -> Ty {
        match scrut_ty {
            Ty::List(elem_ty) => {
                // List cons `:` has tag 1, arity 2
                // field 0: head element (elem_ty)
                // field 1: tail (list type)
                if con.tag == 1 && con.arity == 2 {
                    match field_idx {
                        0 => (**elem_ty).clone(),  // head: element type
                        1 => scrut_ty.clone(),     // tail: list type
                        _ => Ty::Error,
                    }
                } else {
                    // Nil has no fields
                    Ty::Error
                }
            }
            Ty::Tuple(elem_tys) => {
                // Tuple fields correspond directly to element types
                if field_idx < elem_tys.len() {
                    elem_tys[field_idx].clone()
                } else {
                    Ty::Error
                }
            }
            Ty::App(con_ty, arg_ty) => {
                // For type applications like `Maybe Int`, we need to look up
                // the constructor's field types and substitute type arguments.
                // For now, just propagate the argument type for single-arg constructors.
                if con.arity == 1 && field_idx == 0 {
                    (**arg_ty).clone()
                } else {
                    Ty::Error
                }
            }
            _ => {
                // For other types, we can't infer - return Error and let codegen handle it
                Ty::Error
            }
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
    ///
    /// Uses pointer types for all parameters to enable uniform calling convention
    /// for higher-order functions and closures.
    /// All functions take an env_ptr as the first parameter (even if unused)
    /// to enable uniform closure calling convention.
    fn lower_function_type(
        &self,
        ty: &Ty,
    ) -> CodegenResult<inkwell::types::FunctionType<'ctx>> {
        let tm = self.type_mapper();
        let ptr_type = tm.ptr_type();

        // Count the number of parameters
        let mut param_count = 0;
        let mut current = ty;

        while let Ty::Fun(_, ret) = current {
            param_count += 1;
            current = ret;
        }

        // All functions take (env_ptr, args...) for uniform closure calling convention
        let mut arg_types: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> = Vec::new();
        arg_types.push(ptr_type.into()); // env/closure pointer (may be unused)
        for _ in 0..param_count {
            arg_types.push(ptr_type.into());
        }

        // Return type is also pointer for uniformity
        Ok(ptr_type.fn_type(&arg_types, false))
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
