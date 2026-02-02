//! Core IR to WASM lowering.
//!
//! This module walks Core IR expressions and emits WASM instructions,
//! enabling the Default and Edge profiles to compile Haskell programs
//! to WebAssembly without going through the Tensor/Loop IR pipeline.
//!
//! ## Representation
//!
//! - Integers are represented as unboxed `i32` values
//! - Strings are stored in the WASM data segment as (offset, length) pairs
//! - Functions become WASM functions with `i32` parameters and results
//! - IO actions are executed for their side effects and return `i32(0)`

use bhc_core::{AltCon, Bind, CoreModule, Expr, Literal, Var, VarId};
use bhc_intern::Symbol;
use rustc_hash::FxHashMap;

use crate::codegen::{RuntimeIndices, WasmFunc, WasmFuncType};
use crate::{WasmInstr, WasmModule, WasmResult, WasmType};

/// Starting offset for user string data in linear memory.
///
/// Placed after the WASI scratch area and runtime data segments.
const STRING_DATA_BASE: u32 = 2048;

/// Lower a Core IR module to WASM, adding functions to the given module.
///
/// Returns the function index of `main` so the caller can wire up `_start`.
///
/// # Errors
///
/// Returns `WasmError` if lowering encounters an unsupported construct.
pub fn lower_core_module(
    core: &CoreModule,
    wasm: &mut WasmModule,
    runtime: &RuntimeIndices,
) -> WasmResult<u32> {
    let mut lowering = WasmLowering::new(wasm, runtime);

    // First pass: register all top-level function names so we can resolve calls
    for bind in &core.bindings {
        match bind {
            Bind::NonRec(var, _) => {
                lowering.register_binding(var);
            }
            Bind::Rec(bindings) => {
                for (var, _) in bindings {
                    lowering.register_binding(var);
                }
            }
        }
    }

    // Second pass: lower each binding to a WASM function
    for bind in &core.bindings {
        match bind {
            Bind::NonRec(var, expr) => {
                lowering.lower_binding(var, expr)?;
            }
            Bind::Rec(bindings) => {
                for (var, expr) in bindings {
                    lowering.lower_binding(var, expr)?;
                }
            }
        }
    }

    // Find main's function index
    let main_idx = lowering
        .func_map
        .iter()
        .find(|(name, _)| name.as_str() == "main")
        .map(|(_, idx)| *idx)
        .ok_or_else(|| crate::WasmError::CodegenError("no `main` binding found".to_string()))?;

    Ok(main_idx)
}

/// State for the Core IR to WASM lowering pass.
struct WasmLowering<'a> {
    /// The WASM module being built.
    wasm: &'a mut WasmModule,
    /// Runtime function indices.
    runtime: &'a RuntimeIndices,
    /// Maps Haskell function names to WASM function indices.
    func_map: FxHashMap<Symbol, u32>,
    /// String pool: maps string content to (data_offset, length).
    string_pool: FxHashMap<String, (u32, u32)>,
    /// Next available offset in the data segment for string storage.
    next_data_offset: u32,
    /// Counter for pre-registering function indices.
    next_func_idx: u32,
}

impl<'a> WasmLowering<'a> {
    fn new(wasm: &'a mut WasmModule, runtime: &'a RuntimeIndices) -> Self {
        // Count existing functions (imports + defined) to know where new functions start
        let next_func_idx = wasm.next_function_index();

        Self {
            wasm,
            runtime,
            func_map: FxHashMap::default(),
            string_pool: FxHashMap::default(),
            next_data_offset: STRING_DATA_BASE,
            next_func_idx,
        }
    }

    /// Pre-register a top-level binding so it gets a function index.
    fn register_binding(&mut self, var: &Var) {
        let name = var.name;
        if !self.func_map.contains_key(&name) {
            self.func_map.insert(name, self.next_func_idx);
            self.next_func_idx += 1;
        }
    }

    /// Intern a string into the data segment, returning (offset, length).
    fn intern_string(&mut self, s: &str) -> (u32, u32) {
        if let Some(&entry) = self.string_pool.get(s) {
            return entry;
        }

        let offset = self.next_data_offset;
        let len = s.len() as u32;
        let bytes = s.as_bytes().to_vec();
        self.wasm.add_data_segment(offset, bytes);
        self.next_data_offset += len;
        // Align to 4 bytes
        self.next_data_offset = (self.next_data_offset + 3) & !3;

        self.string_pool.insert(s.to_string(), (offset, len));
        (offset, len)
    }

    /// Lower a single top-level binding to a WASM function.
    fn lower_binding(&mut self, var: &Var, expr: &Expr) -> WasmResult<()> {
        let name = var.name;

        // Peel off lambdas to determine function parameters
        let (params, body) = peel_lambdas(expr);

        let is_main = name.as_str() == "main";

        // Determine function type
        let param_types: Vec<WasmType> = params.iter().map(|_| WasmType::I32).collect();
        let result_types = vec![WasmType::I32]; // All functions return i32

        let mut func = WasmFunc::new(WasmFuncType::new(param_types, result_types));
        func.name = Some(name.as_str().to_string());

        // Build a local variable map: param vars -> local indices
        let mut locals: FxHashMap<VarId, u32> = FxHashMap::default();
        for (i, param) in params.iter().enumerate() {
            locals.insert(param.id, i as u32);
        }

        // Lower the body expression
        let mut instrs = Vec::new();
        let mut local_count = params.len() as u32;
        self.lower_expr(body, &mut instrs, &mut locals, &mut local_count, is_main)?;

        // For main: if the body was an IO action (returns nothing useful),
        // ensure we return 0
        if is_main {
            // The body should have left a value on the stack; for IO programs
            // that emit side effects, we need to ensure there's an i32 on the stack.
            // We always append a return-0 since IO actions may not leave a useful value.
            // The lowering of IO expressions already drops any intermediate results,
            // so we just push 0.
            instrs.push(WasmInstr::Drop);
            instrs.push(WasmInstr::I32Const(0));
        }

        instrs.push(WasmInstr::End);

        // Add locals to function
        for _ in params.len() as u32..local_count {
            func.add_local(WasmType::I32);
        }

        // Add instructions
        for instr in instrs {
            func.emit(instr);
        }

        let actual_idx = self.wasm.add_function(func);

        // Verify the index matches what we pre-registered
        let expected_idx = self.func_map.get(&name).copied();
        if let Some(expected) = expected_idx {
            if actual_idx != expected {
                return Err(crate::WasmError::Internal(format!(
                    "function index mismatch for {}: expected {}, got {}",
                    name, expected, actual_idx
                )));
            }
        }

        Ok(())
    }

    /// Lower a Core IR expression, emitting WASM instructions that leave
    /// one i32 value on the stack.
    fn lower_expr(
        &mut self,
        expr: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        match expr {
            Expr::Lit(lit, _, _) => {
                self.lower_literal(lit, instrs)?;
            }

            Expr::Var(var, _) => {
                let name = var.name.as_str();
                // Check if it's a local variable
                if let Some(&local_idx) = locals.get(&var.id) {
                    instrs.push(WasmInstr::LocalGet(local_idx));
                } else if let Some(&func_idx) = self.func_map.get(&var.name) {
                    // It's a reference to a top-level function (as a value).
                    // For now, just call it with no args (for nullary functions).
                    instrs.push(WasmInstr::Call(func_idx));
                } else {
                    // Unknown variable - push 0 as fallback
                    tracing::warn!(var = name, "unresolved variable, using 0");
                    instrs.push(WasmInstr::I32Const(0));
                }
            }

            Expr::App(_, _, _) => {
                self.lower_app(expr, instrs, locals, local_count, is_main)?;
            }

            Expr::Lam(_, _, _) => {
                // A lambda at expression level (not at the top of a binding)
                // means a closure. For our simple lowering, we don't support
                // higher-order functions yet - just push 0.
                tracing::warn!("lambda expression in non-binding position, using 0");
                instrs.push(WasmInstr::I32Const(0));
            }

            Expr::Let(bind, body, _) => {
                match bind.as_ref() {
                    Bind::NonRec(var, rhs) => {
                        // Evaluate RHS
                        self.lower_expr(rhs, instrs, locals, local_count, false)?;
                        // Allocate a local
                        let local_idx = *local_count;
                        *local_count += 1;
                        instrs.push(WasmInstr::LocalSet(local_idx));
                        locals.insert(var.id, local_idx);
                        // Evaluate body
                        self.lower_expr(body, instrs, locals, local_count, is_main)?;
                    }
                    Bind::Rec(bindings) => {
                        // For recursive let bindings, allocate locals first
                        for (var, _) in bindings {
                            let local_idx = *local_count;
                            *local_count += 1;
                            locals.insert(var.id, local_idx);
                            // Initialize to 0
                            instrs.push(WasmInstr::I32Const(0));
                            instrs.push(WasmInstr::LocalSet(local_idx));
                        }
                        // Then evaluate and store each binding
                        for (var, rhs) in bindings {
                            self.lower_expr(rhs, instrs, locals, local_count, false)?;
                            let local_idx = locals[&var.id];
                            instrs.push(WasmInstr::LocalSet(local_idx));
                        }
                        // Evaluate body
                        self.lower_expr(body, instrs, locals, local_count, is_main)?;
                    }
                }
            }

            Expr::Case(scrut, alts, _, _) => {
                self.lower_case(scrut, alts, instrs, locals, local_count, is_main)?;
            }

            // Type-level constructs: erase and look at inner expression
            Expr::TyApp(inner, _, _) | Expr::TyLam(_, inner, _) => {
                self.lower_expr(inner, instrs, locals, local_count, is_main)?;
            }

            // Transparent wrappers
            Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) | Expr::Lazy(inner, _) => {
                self.lower_expr(inner, instrs, locals, local_count, is_main)?;
            }

            // Type annotations / coercions are not values
            Expr::Type(_, _) | Expr::Coercion(_, _) => {
                instrs.push(WasmInstr::I32Const(0));
            }
        }

        Ok(())
    }

    /// Lower a literal value.
    fn lower_literal(&mut self, lit: &Literal, instrs: &mut Vec<WasmInstr>) -> WasmResult<()> {
        match lit {
            Literal::Int(n) => {
                instrs.push(WasmInstr::I32Const(*n as i32));
            }
            Literal::Integer(n) => {
                instrs.push(WasmInstr::I32Const(*n as i32));
            }
            Literal::Char(c) => {
                instrs.push(WasmInstr::I32Const(*c as i32));
            }
            Literal::String(sym) => {
                // Intern the string and push its offset as the "value"
                let s = sym.as_str();
                let (offset, _len) = self.intern_string(s);
                instrs.push(WasmInstr::I32Const(offset as i32));
            }
            Literal::Float(f) => {
                instrs.push(WasmInstr::I32Const((*f as i32).max(0)));
            }
            Literal::Double(d) => {
                instrs.push(WasmInstr::I32Const((*d as i32).max(0)));
            }
        }
        Ok(())
    }

    /// Lower a function application chain.
    ///
    /// This is the core dispatch logic. We collect the function and all its
    /// arguments, then decide how to emit code based on the function name.
    fn lower_app(
        &mut self,
        expr: &Expr,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        // Collect the spine: f a1 a2 ... an
        let (func_expr, args) = collect_app_spine(expr);

        // Get the function name
        let func_name = match func_expr {
            Expr::Var(var, _) => Some(var.name.as_str()),
            _ => None,
        };

        match func_name {
            // Arithmetic primitives
            Some("+" | "plus#" | "plusInt#" | "GHC.Num.+") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Add);
            }
            Some("-" | "minus#" | "minusInt#" | "GHC.Num.-") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Sub);
            }
            Some("*" | "times#" | "timesInt#" | "GHC.Num.*") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Mul);
            }
            Some("div" | "divInt#" | "GHC.Real.div") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32DivS);
            }
            Some("mod" | "modInt#" | "GHC.Real.mod") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32RemS);
            }
            Some("negate" | "negateInt#" | "GHC.Num.negate") if args.len() == 1 => {
                instrs.push(WasmInstr::I32Const(0));
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Sub);
            }

            // Comparison primitives
            Some("==" | "eqInt#" | "GHC.Classes.==") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Eq);
            }
            Some("/=" | "neInt#" | "GHC.Classes./=") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32Ne);
            }
            Some("<" | "ltInt#" | "GHC.Classes.<") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32LtS);
            }
            Some("<=" | "leInt#" | "GHC.Classes.<=") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32LeS);
            }
            Some(">" | "gtInt#" | "GHC.Classes.>") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32GtS);
            }
            Some(">=" | "geInt#" | "GHC.Classes.>=") if args.len() == 2 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                self.lower_expr(&args[1], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::I32GeS);
            }

            // IO: putStrLn "..." => print_str_ln(offset, len)
            Some("putStrLn" | "System.IO.putStrLn" | "GHC.IO.putStrLn") if args.len() == 1 => {
                if let Some(s) = extract_string_literal(&args[0]) {
                    let (offset, len) = self.intern_string(s);
                    instrs.push(WasmInstr::I32Const(offset as i32));
                    instrs.push(WasmInstr::I32Const(len as i32));
                    instrs.push(WasmInstr::Call(self.runtime.print_str_ln_idx));
                } else {
                    // Dynamic string - we don't handle this yet
                    tracing::warn!("putStrLn with non-literal argument");
                    self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::Drop);
                }
                // IO action returns a dummy value
                instrs.push(WasmInstr::I32Const(0));
            }

            // IO: putStr "..." => print_str(offset, len)
            Some("putStr" | "System.IO.putStr" | "GHC.IO.putStr") if args.len() == 1 => {
                if let Some(s) = extract_string_literal(&args[0]) {
                    let (offset, len) = self.intern_string(s);
                    instrs.push(WasmInstr::I32Const(offset as i32));
                    instrs.push(WasmInstr::I32Const(len as i32));
                    instrs.push(WasmInstr::Call(self.runtime.print_str_idx));
                } else {
                    self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::Drop);
                }
                instrs.push(WasmInstr::I32Const(0));
            }

            // IO: print x => print_i32(x) + newline
            Some("print" | "System.IO.print" | "GHC.Show.print") if args.len() == 1 => {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Call(self.runtime.print_i32_idx));
                instrs.push(WasmInstr::I32Const(0));
            }

            // IO: >> (sequence) - evaluate both sides for effects
            Some(">>" | "GHC.Base.>>") if args.len() == 2 => {
                // Evaluate first action, drop result
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                // Evaluate second action, keep result
                self.lower_expr(&args[1], instrs, locals, local_count, is_main)?;
            }

            // IO: >>= (bind) - evaluate first, pass result to second
            Some(">>=" | "GHC.Base.>>=") if args.len() == 2 => {
                // For simple IO programs, >>= is usually just sequencing
                // Evaluate first action, drop result
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
                instrs.push(WasmInstr::Drop);
                // Evaluate second (it's usually a lambda, so just evaluate it)
                self.lower_expr(&args[1], instrs, locals, local_count, is_main)?;
            }

            // IO: return / pure - just evaluate the argument
            Some("return" | "pure" | "GHC.Base.return" | "GHC.Base.pure")
                if args.len() == 1 =>
            {
                self.lower_expr(&args[0], instrs, locals, local_count, false)?;
            }

            // User-defined function call
            Some(name) => {
                if let Some(&func_idx) = self.func_map.get(&func_expr_symbol(func_expr).unwrap())
                {
                    // Push arguments
                    for arg in &args {
                        self.lower_expr(arg, instrs, locals, local_count, false)?;
                    }
                    instrs.push(WasmInstr::Call(func_idx));
                } else if let Some(var) = func_expr_var(func_expr) {
                    // Check if it's a local variable being called (like a closure)
                    if let Some(&local_idx) = locals.get(&var.id) {
                        // Can't call a local in WASM - just push args and use the value
                        for arg in &args {
                            self.lower_expr(arg, instrs, locals, local_count, false)?;
                            instrs.push(WasmInstr::Drop);
                        }
                        instrs.push(WasmInstr::LocalGet(local_idx));
                    } else {
                        tracing::warn!(func = name, "unknown function, using 0");
                        for arg in &args {
                            self.lower_expr(arg, instrs, locals, local_count, false)?;
                            instrs.push(WasmInstr::Drop);
                        }
                        instrs.push(WasmInstr::I32Const(0));
                    }
                } else {
                    tracing::warn!(func = name, "unknown function, using 0");
                    for arg in &args {
                        self.lower_expr(arg, instrs, locals, local_count, false)?;
                        instrs.push(WasmInstr::Drop);
                    }
                    instrs.push(WasmInstr::I32Const(0));
                }
            }

            // Non-variable function expression (e.g., lambda application)
            None => {
                // Evaluate the function and all args, use the function's result
                self.lower_expr(func_expr, instrs, locals, local_count, false)?;
                for arg in &args {
                    self.lower_expr(arg, instrs, locals, local_count, false)?;
                    instrs.push(WasmInstr::Drop);
                }
            }
        }

        Ok(())
    }

    /// Lower a case expression.
    fn lower_case(
        &mut self,
        scrut: &Expr,
        alts: &[bhc_core::Alt],
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        // Evaluate scrutinee and store in a local
        self.lower_expr(scrut, instrs, locals, local_count, false)?;
        let scrut_local = *local_count;
        *local_count += 1;
        instrs.push(WasmInstr::LocalSet(scrut_local));

        // If there's only a Default alt, just execute it
        if alts.len() == 1 {
            if let AltCon::Default = &alts[0].con {
                // Bind scrutinee to binder if present
                if let Some(binder) = alts[0].binders.first() {
                    locals.insert(binder.id, scrut_local);
                }
                self.lower_expr(&alts[0].rhs, instrs, locals, local_count, is_main)?;
                return Ok(());
            }
        }

        // Find the default alternative (if any)
        let default_alt = alts.iter().find(|a| matches!(a.con, AltCon::Default));
        let lit_alts: Vec<_> = alts
            .iter()
            .filter(|a| matches!(a.con, AltCon::Lit(_)))
            .collect();

        // Generate if-else chain for literal alternatives
        if !lit_alts.is_empty() {
            self.lower_case_lit_chain(
                scrut_local,
                &lit_alts,
                default_alt,
                instrs,
                locals,
                local_count,
                is_main,
            )?;
        } else if let Some(def) = default_alt {
            // Only a default alt
            if let Some(binder) = def.binders.first() {
                locals.insert(binder.id, scrut_local);
            }
            self.lower_expr(&def.rhs, instrs, locals, local_count, is_main)?;
        } else {
            // No alternatives at all - push 0
            instrs.push(WasmInstr::I32Const(0));
        }

        Ok(())
    }

    /// Lower a case expression with literal alternatives using if-else chain.
    #[allow(clippy::too_many_arguments)]
    fn lower_case_lit_chain(
        &mut self,
        scrut_local: u32,
        lit_alts: &[&bhc_core::Alt],
        default_alt: Option<&bhc_core::Alt>,
        instrs: &mut Vec<WasmInstr>,
        locals: &mut FxHashMap<VarId, u32>,
        local_count: &mut u32,
        is_main: bool,
    ) -> WasmResult<()> {
        // We need to emit nested if-else blocks.
        // Since WASM requires a result type for if blocks, we use
        // a block(result i32) + br_table pattern, or nested if/else.

        // Use nested if-else: each level checks one literal
        for (i, alt) in lit_alts.iter().enumerate() {
            let lit_val = match &alt.con {
                AltCon::Lit(Literal::Int(n)) => *n as i32,
                AltCon::Lit(Literal::Integer(n)) => *n as i32,
                AltCon::Lit(Literal::Char(c)) => *c as i32,
                _ => 0,
            };

            // Compare scrutinee with literal
            instrs.push(WasmInstr::LocalGet(scrut_local));
            instrs.push(WasmInstr::I32Const(lit_val));
            instrs.push(WasmInstr::I32Eq);
            instrs.push(WasmInstr::If(Some(WasmType::I32)));

            // Bind scrutinee to binder if present
            if let Some(binder) = alt.binders.first() {
                locals.insert(binder.id, scrut_local);
            }

            // RHS
            self.lower_expr(&alt.rhs, instrs, locals, local_count, is_main)?;

            instrs.push(WasmInstr::Else);

            // If this is the last alternative and there's a default, emit it
            if i == lit_alts.len() - 1 {
                if let Some(def) = default_alt {
                    if let Some(binder) = def.binders.first() {
                        locals.insert(binder.id, scrut_local);
                    }
                    self.lower_expr(&def.rhs, instrs, locals, local_count, is_main)?;
                } else {
                    // No default - unreachable or just push 0
                    instrs.push(WasmInstr::I32Const(0));
                }
            }
        }

        // Close all the if-else blocks
        for _ in lit_alts {
            instrs.push(WasmInstr::End);
        }

        Ok(())
    }
}

// ============================================================
// Helper functions
// ============================================================

/// Peel lambda abstractions off the front of an expression.
fn peel_lambdas(expr: &Expr) -> (Vec<&Var>, &Expr) {
    let mut params = Vec::new();
    let mut current = expr;

    loop {
        match current {
            Expr::Lam(var, body, _) => {
                params.push(var);
                current = body;
            }
            // Skip type abstractions
            Expr::TyLam(_, body, _) => {
                current = body;
            }
            _ => break,
        }
    }

    (params, current)
}

/// Collect the application spine: `f a1 a2 ... an` -> `(f, [a1, a2, ..., an])`.
fn collect_app_spine(expr: &Expr) -> (&Expr, Vec<&Expr>) {
    let mut args = Vec::new();
    let mut current = expr;

    loop {
        match current {
            Expr::App(f, arg, _) => {
                args.push(arg.as_ref());
                current = f;
            }
            // Skip type applications
            Expr::TyApp(f, _, _) => {
                current = f;
            }
            _ => break,
        }
    }

    args.reverse();
    (current, args)
}

/// Extract a string literal from an expression, looking through casts/ticks.
fn extract_string_literal(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Lit(Literal::String(sym), _, _) => Some(sym.as_str()),
        Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) => extract_string_literal(inner),
        Expr::TyApp(inner, _, _) => extract_string_literal(inner),
        Expr::App(f, arg, _) => {
            // Sometimes strings appear as `unpackCString# "literal"`
            if let Expr::Var(var, _) = f.as_ref() {
                if var.name.as_str().contains("unpackCString") {
                    return extract_string_literal(arg);
                }
            }
            None
        }
        _ => None,
    }
}

/// Get the `Symbol` from a function expression if it's a `Var`.
fn func_expr_symbol(expr: &Expr) -> Option<Symbol> {
    match expr {
        Expr::Var(var, _) => Some(var.name),
        _ => None,
    }
}

/// Get the `Var` from a function expression if it's a `Var`.
fn func_expr_var(expr: &Expr) -> Option<&Var> {
    match expr {
        Expr::Var(var, _) => Some(var),
        _ => None,
    }
}
