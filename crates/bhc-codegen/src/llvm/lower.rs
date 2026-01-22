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
use bhc_core::{Alt, AltCon, Bind, CoreModule, Expr, Literal, Var, VarId};
use bhc_index::Idx;
use bhc_types::Ty;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::types::{BasicType, BasicTypeEnum};
use inkwell::values::{BasicValueEnum, FunctionValue};
use rustc_hash::FxHashMap;

use super::context::LlvmContext;
use super::module::LlvmModule;
use super::types::TypeMapper;

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
}

impl<'ctx, 'm> Lowering<'ctx, 'm> {
    /// Create a new lowering context.
    pub fn new(ctx: &'ctx LlvmContext, module: &'m LlvmModule<'ctx>) -> Self {
        let mut lowering = Self {
            llvm_ctx: ctx.llvm_context(),
            module,
            env: FxHashMap::default(),
            functions: FxHashMap::default(),
        };
        lowering.declare_rts_functions();
        lowering
    }

    /// Declare external RTS functions.
    fn declare_rts_functions(&mut self) {
        let tm = self.type_mapper();
        let void_type = self.llvm_ctx.void_type();
        let i64_type = tm.i64_type();
        let f64_type = tm.f64_type();
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

    /// Lower a Core module to LLVM IR.
    pub fn lower_module(&mut self, core_module: &CoreModule) -> CodegenResult<()> {
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

    /// Declare a binding (creates function signature without body).
    fn declare_binding(&mut self, bind: &Bind) -> CodegenResult<()> {
        match bind {
            Bind::NonRec(var, _) => {
                let fn_val = self.declare_function(var)?;
                self.functions.insert(var.id, fn_val);
            }
            Bind::Rec(bindings) => {
                for (var, _) in bindings {
                    let fn_val = self.declare_function(var)?;
                    self.functions.insert(var.id, fn_val);
                }
            }
        }
        Ok(())
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

        // Lower the expression body
        let result = self.lower_expr(expr)?;

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

    /// Lower a Core expression to LLVM IR.
    fn lower_expr(&mut self, expr: &Expr) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        match expr {
            Expr::Lit(lit, _ty, _span) => self.lower_literal(lit).map(Some),

            Expr::Var(var, _span) => {
                // Look up the variable in the environment
                if let Some(val) = self.env.get(&var.id) {
                    Ok(Some(*val))
                } else if let Some(fn_val) = self.functions.get(&var.id) {
                    // It's a function reference - return as pointer
                    Ok(Some(fn_val.as_global_value().as_pointer_value().into()))
                } else {
                    Err(CodegenError::Internal(format!(
                        "unbound variable: {}",
                        var.name.as_str()
                    )))
                }
            }

            Expr::App(func, arg, _span) => self.lower_application(func, arg),

            Expr::Lam(_param, _body, _span) => {
                // For now, lambdas should have been lifted to top-level
                // Full closure support requires heap allocation
                Err(CodegenError::Unsupported(
                    "nested lambdas not yet supported - use let bindings".to_string(),
                ))
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

    /// Lower a function application.
    fn lower_application(
        &mut self,
        func: &Expr,
        arg: &Expr,
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        // Collect all arguments (for curried applications)
        let mut args = vec![arg];
        let mut current = func;

        while let Expr::App(inner_func, inner_arg, _) = current {
            args.push(inner_arg);
            current = inner_func;
        }

        args.reverse();

        // Get the function being called
        let fn_val = match current {
            Expr::Var(var, _) => {
                let name = var.name.as_str();

                // Check if this is an RTS builtin
                if let Some(rts_id) = self.rts_function_id(name) {
                    self.functions.get(&rts_id).copied().ok_or_else(|| {
                        CodegenError::Internal(format!("RTS function not declared: {}", name))
                    })?
                } else {
                    // Look up the user-defined function
                    self.functions.get(&var.id).copied().ok_or_else(|| {
                        CodegenError::Internal(format!("unknown function: {}", name))
                    })?
                }
            }
            _ => {
                return Err(CodegenError::Unsupported(
                    "indirect function calls not yet supported".to_string(),
                ))
            }
        };

        // Lower arguments
        let mut llvm_args = Vec::new();
        for arg_expr in &args {
            if let Some(val) = self.lower_expr(arg_expr)? {
                llvm_args.push(val.into());
            }
        }

        // Build the call
        let call = self
            .builder()
            .build_call(fn_val, &llvm_args, "call")
            .map_err(|e| CodegenError::Internal(format!("failed to build call: {:?}", e)))?;

        Ok(call.try_as_basic_value().left())
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

            Bind::Rec(_bindings) => {
                // For recursive let bindings, we'd need phi nodes or allocas
                // For now, just return an error for non-function recursive bindings
                Err(CodegenError::Unsupported(
                    "recursive let bindings not yet supported".to_string(),
                ))
            }
        }
    }

    /// Lower a case expression.
    fn lower_case(
        &mut self,
        scrut: &Expr,
        alts: &[Alt],
    ) -> CodegenResult<Option<BasicValueEnum<'ctx>>> {
        let scrut_val = self.lower_expr(scrut)?.ok_or_else(|| {
            CodegenError::Internal("scrutinee has no value".to_string())
        })?;

        // For now, only handle integer/char case
        let scrut_int = match scrut_val {
            BasicValueEnum::IntValue(i) => i,
            _ => {
                return Err(CodegenError::Unsupported(
                    "case on non-integer values not yet supported".to_string(),
                ))
            }
        };

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
                        "complex case patterns not yet supported".to_string(),
                    ))
                }
            }
        }

        // Build switch
        let default = default_block.unwrap_or(merge_block);
        let _switch = self
            .builder()
            .build_switch(scrut_int, default, &cases)
            .map_err(|e| CodegenError::Internal(format!("failed to build switch: {:?}", e)))?;

        // Generate code for each alternative
        let mut phi_values = Vec::new();

        for (i, alt) in alts.iter().enumerate() {
            self.builder().position_at_end(blocks[i]);

            if let Some(val) = self.lower_expr(&alt.rhs)? {
                phi_values.push((val, blocks[i]));
            }

            // Jump to merge block
            self.builder()
                .build_unconditional_branch(merge_block)
                .map_err(|e| CodegenError::Internal(format!("failed to build branch: {:?}", e)))?;
        }

        // Build phi node in merge block
        self.builder().position_at_end(merge_block);

        if phi_values.is_empty() {
            Ok(None)
        } else {
            let first_val = phi_values[0].0;
            let phi = self
                .builder()
                .build_phi(first_val.get_type(), "case_result")
                .map_err(|e| CodegenError::Internal(format!("failed to build phi: {:?}", e)))?;

            for (val, block) in &phi_values {
                phi.add_incoming(&[(val, *block)]);
            }

            Ok(Some(phi.as_basic_value()))
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
                    "Bool" => Ok(Some(tm.bool_type().into())),
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
