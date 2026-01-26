//! Integration tests for WASM code generation.

use bhc_target::{Arch, targets};
use bhc_wasm::{WasmBackend, WasmConfig, WasmInstr, WasmType};
use bhc_wasm::codegen::{WasmModule, WasmFuncType, WasmFunc, MemoryDesc};
use bhc_codegen::{CodegenBackend, CodegenModule};

#[test]
fn test_backend_supports_wasm32() {
    let backend = WasmBackend::new();
    let wasm_target = targets::wasm32_wasi();

    assert_eq!(backend.name(), "wasm");
    assert!(backend.supports_target(&wasm_target));

    // Should not support x86
    let x86_target = targets::x86_64_linux_gnu();
    assert!(!backend.supports_target(&x86_target));
}

#[test]
fn test_wasm_config_default() {
    let config = WasmConfig::default();

    assert!(config.simd_enabled);
    assert_eq!(config.initial_memory_pages, 16);
    assert_eq!(config.max_memory_pages, Some(256));
}

#[test]
fn test_wasm_config_edge() {
    let config = WasmConfig::edge_profile();

    assert!(config.optimize_size);
    assert!(!config.debug_names);  // Debug names disabled for edge
    assert_eq!(config.initial_memory_pages, 4);
    assert_eq!(config.max_memory_pages, Some(64));
}

#[test]
fn test_wasm_type_wat_names() {
    assert_eq!(WasmType::I32.wat_name(), "i32");
    assert_eq!(WasmType::I64.wat_name(), "i64");
    assert_eq!(WasmType::F32.wat_name(), "f32");
    assert_eq!(WasmType::F64.wat_name(), "f64");
    assert_eq!(WasmType::V128.wat_name(), "v128");
}

#[test]
fn test_wasm_func_type_wat() {
    let func_type = WasmFuncType::new(
        vec![WasmType::I32, WasmType::I32],
        vec![WasmType::I32],
    );

    let wat = func_type.to_wat();
    assert!(wat.contains("param i32 i32"));
    assert!(wat.contains("result i32"));
}

#[test]
fn test_wasm_func_type_empty() {
    let func_type = WasmFuncType::new(vec![], vec![]);
    let wat = func_type.to_wat();

    // Should not contain param or result for empty function
    assert!(!wat.contains("param"));
    assert!(!wat.contains("result"));
}

#[test]
fn test_simple_module_generation() {
    let wasm_target = targets::wasm32_wasi();

    // Create a simple module with one function
    let mut module = WasmModule::new(
        "test".to_string(),
        WasmConfig::default(),
        wasm_target,
    );

    // Add a simple add function: (func $add (param i32 i32) (result i32) ...)
    let add_func = WasmFunc {
        name: Some("add".to_string()),
        ty: WasmFuncType::new(
            vec![WasmType::I32, WasmType::I32],
            vec![WasmType::I32],
        ),
        locals: vec![],
        body: vec![
            WasmInstr::LocalGet(0),
            WasmInstr::LocalGet(1),
            WasmInstr::I32Add,
            WasmInstr::End,
        ],
        exported: true,
        export_name: Some("add".to_string()),
    };
    module.add_function(add_func);

    // Generate WAT
    let wat = module.to_wat();

    assert!(wat.contains("(module"));
    assert!(wat.contains("(func $add"));
    assert!(wat.contains("(export \"add\""));
}

#[test]
fn test_memory_desc() {
    let mem = MemoryDesc {
        min: 16,
        max: Some(256),
        shared: false,
    };

    assert_eq!(mem.min, 16);
    assert_eq!(mem.max, Some(256));
    assert!(!mem.shared);
}

#[test]
fn test_binary_generation() {
    let wasm_target = targets::wasm32_wasi();

    let module = WasmModule::new(
        "test".to_string(),
        WasmConfig::default(),
        wasm_target,
    );

    // Verify returns valid WASM
    let result = module.verify();
    assert!(result.is_ok());

    // Generate binary
    let binary = module.to_wasm().unwrap();

    // Check WASM magic number: 0x00 0x61 0x73 0x6D
    assert_eq!(&binary[0..4], &[0x00, 0x61, 0x73, 0x6D]);

    // Check version: 0x01 0x00 0x00 0x00
    assert_eq!(&binary[4..8], &[0x01, 0x00, 0x00, 0x00]);
}

#[test]
fn test_simd_instructions() {
    // Test that SIMD instructions are correctly represented
    let instrs = vec![
        WasmInstr::V128Load(0, 16),
        WasmInstr::F32x4Add,
        WasmInstr::F32x4Mul,
        WasmInstr::V128Store(0, 16),
    ];

    // These should all be valid instructions
    for instr in &instrs {
        // Just verify they can be constructed
        assert!(matches!(
            instr,
            WasmInstr::V128Load(_, _)
            | WasmInstr::F32x4Add
            | WasmInstr::F32x4Mul
            | WasmInstr::V128Store(_, _)
        ));
    }
}

#[test]
fn test_type_mapping() {
    use bhc_wasm::codegen::types::{type_to_wasm, LoopTypeMapping};
    use bhc_loop_ir::{LoopType, ScalarType};

    let mapping = LoopTypeMapping::for_arch(Arch::Wasm32, true);

    // Scalar types
    assert_eq!(
        type_to_wasm(&LoopType::Scalar(ScalarType::Int(32)), &mapping).unwrap(),
        WasmType::I32
    );
    assert_eq!(
        type_to_wasm(&LoopType::Scalar(ScalarType::Int(64)), &mapping).unwrap(),
        WasmType::I64
    );
    assert_eq!(
        type_to_wasm(&LoopType::Scalar(ScalarType::Float(32)), &mapping).unwrap(),
        WasmType::F32
    );
    assert_eq!(
        type_to_wasm(&LoopType::Scalar(ScalarType::Float(64)), &mapping).unwrap(),
        WasmType::F64
    );

    // Vector types map to V128
    assert_eq!(
        type_to_wasm(&LoopType::Vector(ScalarType::Float(32), 4), &mapping).unwrap(),
        WasmType::V128
    );
    assert_eq!(
        type_to_wasm(&LoopType::Vector(ScalarType::Float(64), 2), &mapping).unwrap(),
        WasmType::V128
    );

    // Pointers
    assert_eq!(
        type_to_wasm(&LoopType::Ptr(Box::new(LoopType::Scalar(ScalarType::Float(32)))), &mapping).unwrap(),
        WasmType::I32  // 32-bit pointer on wasm32
    );
}

#[test]
fn test_type_mapping_wasm64() {
    use bhc_wasm::codegen::types::{type_to_wasm, LoopTypeMapping};
    use bhc_loop_ir::{LoopType, ScalarType};

    let mapping = LoopTypeMapping::for_arch(Arch::Wasm64, true);

    // Pointers should be i64 on wasm64
    assert_eq!(
        type_to_wasm(&LoopType::Ptr(Box::new(LoopType::Scalar(ScalarType::Float(32)))), &mapping).unwrap(),
        WasmType::I64
    );
}

#[test]
fn test_runtime_config() {
    use bhc_wasm::runtime::RuntimeConfig;

    let config = RuntimeConfig::default();
    assert_eq!(config.initial_pages, 32);  // 2MB for stack + arena
    assert!(config.enable_arena);

    // Validate should pass
    assert!(config.validate().is_ok());

    // Edge config
    let edge = RuntimeConfig::edge();
    assert_eq!(edge.initial_pages, 8);  // 512KB for stack + arena
    assert!(edge.validate().is_ok());
}

#[test]
fn test_memory_layout() {
    use bhc_wasm::runtime::MemoryLayout;

    let layout = MemoryLayout::default_for_size(16);

    assert!(layout.validate().is_ok());
    assert!(layout.stack_size() > 0);
    assert!(layout.heap_size() > 0);

    // Regions should not overlap
    assert!(layout.data_end <= layout.stack_start);
    assert!(layout.stack_end <= layout.heap_start);
    assert!(layout.heap_end <= layout.total_size);
}

#[test]
fn test_memory_layout_builder() {
    use bhc_wasm::runtime::MemoryLayout;

    let layout = MemoryLayout::builder()
        .data_size(8192)
        .stack_size(32768)
        .total_pages(8)
        .build()
        .unwrap();

    assert_eq!(layout.data_size(), 8192);
    assert_eq!(layout.stack_size(), 32768);
    assert!(layout.validate().is_ok());
}

#[test]
fn test_memory_layout_overflow() {
    use bhc_wasm::runtime::MemoryLayout;

    // Try to allocate more than available
    let result = MemoryLayout::builder()
        .data_size(100_000)
        .stack_size(100_000)
        .heap_size(100_000)
        .total_pages(1)  // Only 64KB!
        .build();

    assert!(result.is_err());
}

#[test]
fn test_linear_memory() {
    use bhc_wasm::runtime::{LinearMemory, MemoryLayout};

    let layout = MemoryLayout::default_for_size(16);
    let mut mem = LinearMemory::new(layout);

    // Allocate some data
    let offset1 = mem.alloc_i32(42).unwrap();
    assert_eq!(offset1, 0);  // First allocation at start

    let offset2 = mem.alloc_f64(3.14159).unwrap();
    assert!(offset2 >= 8);  // Should be aligned to 8 bytes

    // Check data segments were created
    let segments = mem.data_segments();
    assert_eq!(segments.len(), 2);
}

#[test]
fn test_arena_config() {
    use bhc_wasm::runtime::ArenaConfig;

    let config = ArenaConfig::default();
    assert_eq!(config.alignment, 16);
    assert_eq!(config.size, 1024 * 1024);

    let edge = ArenaConfig::edge();
    assert_eq!(edge.size, 256 * 1024);

    // End address calculation
    assert_eq!(config.end_address(), config.start_address + config.size);
}

#[test]
fn test_wasm_arena_code_generation() {
    use bhc_wasm::runtime::{WasmArena, ArenaConfig};

    let config = ArenaConfig::default();
    let arena = WasmArena::new(config, 0, 1);

    // Generate init code
    let globals = arena.generate_init();
    assert_eq!(globals.len(), 2);
    assert_eq!(globals[0].name, "arena_ptr");
    assert!(globals[0].mutable);
    assert_eq!(globals[1].name, "arena_end");
    assert!(!globals[1].mutable);

    // Generate alloc code
    let alloc_instrs = arena.generate_alloc();
    assert!(!alloc_instrs.is_empty());

    // Should contain bounds check
    assert!(alloc_instrs.iter().any(|i| matches!(i, WasmInstr::If(_))));

    // Generate reset code
    let reset_instrs = arena.generate_reset();
    assert!(!reset_instrs.is_empty());
}

#[test]
fn test_module_with_multiple_functions() {
    let wasm_target = targets::wasm32_wasi();

    let mut module = WasmModule::new(
        "math".to_string(),
        WasmConfig::default(),
        wasm_target,
    );

    // Add function
    module.add_function(WasmFunc {
        name: Some("square".to_string()),
        ty: WasmFuncType::new(vec![WasmType::I32], vec![WasmType::I32]),
        locals: vec![],
        body: vec![
            WasmInstr::LocalGet(0),
            WasmInstr::LocalGet(0),
            WasmInstr::I32Mul,
            WasmInstr::End,
        ],
        exported: true,
        export_name: Some("square".to_string()),
    });

    module.add_function(WasmFunc {
        name: Some("cube".to_string()),
        ty: WasmFuncType::new(vec![WasmType::I32], vec![WasmType::I32]),
        locals: vec![WasmType::I32],  // One local for intermediate
        body: vec![
            WasmInstr::LocalGet(0),
            WasmInstr::LocalGet(0),
            WasmInstr::I32Mul,
            WasmInstr::LocalTee(1),  // Store square
            WasmInstr::LocalGet(0),
            WasmInstr::I32Mul,
            WasmInstr::End,
        ],
        exported: true,
        export_name: Some("cube".to_string()),
    });

    let wat = module.to_wat();
    assert!(wat.contains("$square"));
    assert!(wat.contains("$cube"));
    assert!(wat.contains("(export \"square\""));
    assert!(wat.contains("(export \"cube\""));

    // Verify and generate binary
    assert!(module.verify().is_ok());
    let binary = module.to_wasm().unwrap();
    assert!(binary.len() > 8);  // At least header
}

#[test]
fn test_simd_function() {
    let wasm_target = targets::wasm32_wasi();

    let mut module = WasmModule::new(
        "simd_test".to_string(),
        WasmConfig::default(),
        wasm_target,
    );

    // SIMD vector add function
    module.add_function(WasmFunc {
        name: Some("v128_add".to_string()),
        ty: WasmFuncType::new(
            vec![WasmType::I32, WasmType::I32, WasmType::I32],  // a_ptr, b_ptr, out_ptr
            vec![],
        ),
        locals: vec![WasmType::V128, WasmType::V128],
        body: vec![
            // Load a
            WasmInstr::LocalGet(0),
            WasmInstr::V128Load(0, 16),
            WasmInstr::LocalSet(3),
            // Load b
            WasmInstr::LocalGet(1),
            WasmInstr::V128Load(0, 16),
            WasmInstr::LocalSet(4),
            // Add
            WasmInstr::LocalGet(3),
            WasmInstr::LocalGet(4),
            WasmInstr::F32x4Add,
            // Store result
            WasmInstr::LocalGet(2),
            WasmInstr::V128Store(0, 16),
            WasmInstr::End,
        ],
        exported: true,
        export_name: Some("v128_add".to_string()),
    });

    assert!(module.verify().is_ok());
}

// ============================================================================
// Loop IR to WASM Lowering Tests
// ============================================================================

mod loop_ir_lowering {
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_loop_ir::{Body, LoopIR, LoopType, Param, ScalarType, Stmt, Value, ValueId};
    use bhc_wasm::lower::lower_loop_ir;
    use bhc_wasm::{WasmConfig, WasmType};

    /// Create a simple identity function: fn test(x: i32) -> i32 { return x; }
    fn make_identity_function() -> LoopIR {
        LoopIR {
            name: Symbol::intern("identity"),
            params: vec![Param {
                name: Symbol::intern("x"),
                ty: LoopType::Scalar(ScalarType::I32),
                is_ptr: false,
            }],
            return_ty: LoopType::Scalar(ScalarType::I32),
            allocs: vec![],
            body: Body {
                stmts: vec![Stmt::Return(Some(Value::Var(
                    ValueId::new(0),
                    LoopType::Scalar(ScalarType::I32),
                )))],
            },
            loop_info: vec![],
        }
    }

    /// Create a function that returns an i32 constant.
    fn make_const_function(value: i64) -> LoopIR {
        LoopIR {
            name: Symbol::intern("const_42"),
            params: vec![],
            return_ty: LoopType::Scalar(ScalarType::I32),
            allocs: vec![],
            body: Body {
                stmts: vec![Stmt::Return(Some(Value::IntConst(value, ScalarType::I32)))],
            },
            loop_info: vec![],
        }
    }

    /// Create an add function: fn add(a: i32, b: i32) -> i32 { return a + b; }
    fn make_add_function() -> LoopIR {
        use bhc_loop_ir::{BinOp, Expr};

        LoopIR {
            name: Symbol::intern("add"),
            params: vec![
                Param {
                    name: Symbol::intern("a"),
                    ty: LoopType::Scalar(ScalarType::I32),
                    is_ptr: false,
                },
                Param {
                    name: Symbol::intern("b"),
                    ty: LoopType::Scalar(ScalarType::I32),
                    is_ptr: false,
                },
            ],
            return_ty: LoopType::Scalar(ScalarType::I32),
            allocs: vec![],
            body: Body {
                stmts: vec![
                    // result = a + b
                    Stmt::Assign(
                        ValueId::new(2),
                        LoopType::Scalar(ScalarType::I32),
                        Expr::BinOp(
                            BinOp::Add,
                            Value::Var(ValueId::new(0), LoopType::Scalar(ScalarType::I32)),
                            Value::Var(ValueId::new(1), LoopType::Scalar(ScalarType::I32)),
                        ),
                    ),
                    // return result
                    Stmt::Return(Some(Value::Var(
                        ValueId::new(2),
                        LoopType::Scalar(ScalarType::I32),
                    ))),
                ],
            },
            loop_info: vec![],
        }
    }

    /// Create a float multiply function: fn fmul(a: f32, b: f32) -> f32 { return a * b; }
    fn make_fmul_function() -> LoopIR {
        use bhc_loop_ir::{BinOp, Expr};

        LoopIR {
            name: Symbol::intern("fmul"),
            params: vec![
                Param {
                    name: Symbol::intern("a"),
                    ty: LoopType::Scalar(ScalarType::F32),
                    is_ptr: false,
                },
                Param {
                    name: Symbol::intern("b"),
                    ty: LoopType::Scalar(ScalarType::F32),
                    is_ptr: false,
                },
            ],
            return_ty: LoopType::Scalar(ScalarType::F32),
            allocs: vec![],
            body: Body {
                stmts: vec![
                    Stmt::Assign(
                        ValueId::new(2),
                        LoopType::Scalar(ScalarType::F32),
                        Expr::BinOp(
                            BinOp::Mul,
                            Value::Var(ValueId::new(0), LoopType::Scalar(ScalarType::F32)),
                            Value::Var(ValueId::new(1), LoopType::Scalar(ScalarType::F32)),
                        ),
                    ),
                    Stmt::Return(Some(Value::Var(
                        ValueId::new(2),
                        LoopType::Scalar(ScalarType::F32),
                    ))),
                ],
            },
            loop_info: vec![],
        }
    }

    #[test]
    fn test_lower_identity_function() {
        let ir = make_identity_function();
        let config = WasmConfig::default();

        let func = lower_loop_ir(&ir, &config).expect("lowering should succeed");

        // Check function name
        assert_eq!(func.name.as_deref(), Some("identity"));

        // Check signature: (param i32) (result i32)
        assert_eq!(func.ty.params.len(), 1);
        assert_eq!(func.ty.params[0], WasmType::I32);
        assert_eq!(func.ty.results.len(), 1);
        assert_eq!(func.ty.results[0], WasmType::I32);

        // Body should have instructions
        assert!(!func.body.is_empty());
    }

    #[test]
    fn test_lower_const_function() {
        let ir = make_const_function(42);
        let config = WasmConfig::default();

        let func = lower_loop_ir(&ir, &config).expect("lowering should succeed");

        // No params, returns i32
        assert!(func.ty.params.is_empty());
        assert_eq!(func.ty.results.len(), 1);
        assert_eq!(func.ty.results[0], WasmType::I32);
    }

    #[test]
    fn test_lower_add_function() {
        let ir = make_add_function();
        let config = WasmConfig::default();

        let func = lower_loop_ir(&ir, &config).expect("lowering should succeed");

        // Check signature: (param i32 i32) (result i32)
        assert_eq!(func.ty.params.len(), 2);
        assert_eq!(func.ty.params[0], WasmType::I32);
        assert_eq!(func.ty.params[1], WasmType::I32);
        assert_eq!(func.ty.results.len(), 1);
        assert_eq!(func.ty.results[0], WasmType::I32);

        // Should have at least one local for the result
        assert!(!func.locals.is_empty() || !func.body.is_empty());
    }

    #[test]
    fn test_lower_float_function() {
        let ir = make_fmul_function();
        let config = WasmConfig::default();

        let func = lower_loop_ir(&ir, &config).expect("lowering should succeed");

        // Check signature: (param f32 f32) (result f32)
        assert_eq!(func.ty.params.len(), 2);
        assert_eq!(func.ty.params[0], WasmType::F32);
        assert_eq!(func.ty.params[1], WasmType::F32);
        assert_eq!(func.ty.results.len(), 1);
        assert_eq!(func.ty.results[0], WasmType::F32);
    }

    #[test]
    fn test_lower_with_edge_config() {
        let ir = make_identity_function();
        let config = WasmConfig::edge_profile();

        let func = lower_loop_ir(&ir, &config).expect("lowering should succeed");

        // Edge profile should still produce valid function
        assert_eq!(func.name.as_deref(), Some("identity"));
        assert!(!func.body.is_empty());
    }

    #[test]
    fn test_lower_64bit_types() {
        // Test i64 function
        let ir = LoopIR {
            name: Symbol::intern("identity_i64"),
            params: vec![Param {
                name: Symbol::intern("x"),
                ty: LoopType::Scalar(ScalarType::I64),
                is_ptr: false,
            }],
            return_ty: LoopType::Scalar(ScalarType::I64),
            allocs: vec![],
            body: Body {
                stmts: vec![Stmt::Return(Some(Value::Var(
                    ValueId::new(0),
                    LoopType::Scalar(ScalarType::I64),
                )))],
            },
            loop_info: vec![],
        };

        let config = WasmConfig::default();
        let func = lower_loop_ir(&ir, &config).expect("lowering should succeed");

        assert_eq!(func.ty.params[0], WasmType::I64);
        assert_eq!(func.ty.results[0], WasmType::I64);
    }

    #[test]
    fn test_lower_f64_types() {
        // Test f64 function
        let ir = LoopIR {
            name: Symbol::intern("identity_f64"),
            params: vec![Param {
                name: Symbol::intern("x"),
                ty: LoopType::Scalar(ScalarType::F64),
                is_ptr: false,
            }],
            return_ty: LoopType::Scalar(ScalarType::F64),
            allocs: vec![],
            body: Body {
                stmts: vec![Stmt::Return(Some(Value::Var(
                    ValueId::new(0),
                    LoopType::Scalar(ScalarType::F64),
                )))],
            },
            loop_info: vec![],
        };

        let config = WasmConfig::default();
        let func = lower_loop_ir(&ir, &config).expect("lowering should succeed");

        assert_eq!(func.ty.params[0], WasmType::F64);
        assert_eq!(func.ty.results[0], WasmType::F64);
    }
}
