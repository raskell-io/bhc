//! Interface file generation from compiled modules.
//!
//! Generates `.bhi` interface files that capture the public API of a module,
//! enabling separate compilation and type checking without source code.

use crate::{
    ClassMethod, Constraint, DataConstructor, ExportedClass, ExportedInstance, ExportedType,
    ExportedValue, Kind, ModuleInterface, Type, TypeDefinition, TypeSignature,
};
use bhc_ast::Module as AstModule;

/// Generate a module interface from a parsed AST and type-checked module.
///
/// Extracts exported values, types, classes, and instances to produce a
/// `ModuleInterface` suitable for serialization to a `.bhi` file.
pub fn generate_interface(
    module_name: &str,
    ast: &AstModule,
    _typed: &bhc_typeck::TypedModule,
) -> ModuleInterface {
    let mut iface = ModuleInterface::new(module_name);

    // Compute a simple hash from the source for consistency checking
    iface.header.module_hash = compute_module_hash(module_name);

    // Extract exports from AST declarations
    for decl in &ast.decls {
        match decl {
            bhc_ast::Decl::TypeSig(sig) => {
                for name in &sig.names {
                    let exported_ty = convert_ast_type(&sig.ty);
                    iface.add_value(ExportedValue {
                        name: name.name.as_str().to_string(),
                        signature: TypeSignature {
                            type_vars: Vec::new(),
                            constraints: Vec::new(),
                            ty: exported_ty,
                        },
                        inline: crate::InlineInfo::None,
                    });
                }
            }
            bhc_ast::Decl::DataDecl(data) => {
                let params: Vec<String> = data
                    .params
                    .iter()
                    .map(|p| p.name.name.as_str().to_string())
                    .collect();
                let constructors: Vec<DataConstructor> = data
                    .constrs
                    .iter()
                    .map(|con| convert_con_decl(con))
                    .collect();
                let kind = params_to_kind(params.len());
                iface.add_type(ExportedType {
                    name: data.name.name.as_str().to_string(),
                    params,
                    kind,
                    definition: Some(TypeDefinition::Data(constructors)),
                });
            }
            bhc_ast::Decl::Newtype(nt) => {
                let params: Vec<String> = nt
                    .params
                    .iter()
                    .map(|p| p.name.name.as_str().to_string())
                    .collect();
                let con = convert_con_decl(&nt.constr);
                let kind = params_to_kind(params.len());
                iface.add_type(ExportedType {
                    name: nt.name.name.as_str().to_string(),
                    params,
                    kind,
                    definition: Some(TypeDefinition::Newtype(con)),
                });
            }
            bhc_ast::Decl::TypeAlias(ta) => {
                let params: Vec<String> = ta
                    .params
                    .iter()
                    .map(|p| p.name.name.as_str().to_string())
                    .collect();
                let kind = params_to_kind(params.len());
                iface.add_type(ExportedType {
                    name: ta.name.name.as_str().to_string(),
                    params,
                    kind,
                    definition: Some(TypeDefinition::TypeSynonym(convert_ast_type(&ta.ty))),
                });
            }
            bhc_ast::Decl::ClassDecl(cls) => {
                let params: Vec<String> = cls
                    .params
                    .iter()
                    .map(|p| p.name.name.as_str().to_string())
                    .collect();
                let supers: Vec<Constraint> = cls
                    .context
                    .iter()
                    .map(convert_ast_constraint)
                    .collect();
                // Extract method signatures from class body declarations
                let methods: Vec<ClassMethod> = cls
                    .methods
                    .iter()
                    .filter_map(|d| {
                        if let bhc_ast::Decl::TypeSig(sig) = d {
                            Some(sig.names.iter().map(move |name| ClassMethod {
                                name: name.name.as_str().to_string(),
                                signature: TypeSignature {
                                    type_vars: Vec::new(),
                                    constraints: Vec::new(),
                                    ty: convert_ast_type(&sig.ty),
                                },
                                has_default: false,
                            }))
                        } else {
                            None
                        }
                    })
                    .flatten()
                    .collect();
                iface.add_class(ExportedClass {
                    name: cls.name.name.as_str().to_string(),
                    params,
                    superclasses: supers,
                    methods,
                });
            }
            bhc_ast::Decl::InstanceDecl(inst) => {
                let constraints: Vec<Constraint> = inst
                    .context
                    .iter()
                    .map(convert_ast_constraint)
                    .collect();
                let types = vec![convert_ast_type(&inst.ty)];
                iface.add_instance(ExportedInstance {
                    class: inst.class.name.as_str().to_string(),
                    types,
                    constraints,
                });
            }
            // Other declarations (FunBind without sig, fixity, foreign, etc.) — skip for MVP
            _ => {}
        }
    }

    // Record import dependencies
    for import in &ast.imports {
        let dep_name = import
            .module
            .parts
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(".");
        iface.dependencies.push(crate::InterfaceDependency {
            module: dep_name,
            hash: 0, // Hash not yet available for dependencies
        });
    }

    iface
}

/// Convert an AST constructor declaration to an interface DataConstructor.
fn convert_con_decl(con: &bhc_ast::ConDecl) -> DataConstructor {
    match &con.fields {
        bhc_ast::ConFields::Positional(types) => DataConstructor {
            name: con.name.name.as_str().to_string(),
            fields: types.iter().map(convert_ast_type).collect(),
            field_names: None,
        },
        bhc_ast::ConFields::Record(fields) => DataConstructor {
            name: con.name.name.as_str().to_string(),
            fields: fields.iter().map(|f| convert_ast_type(&f.ty)).collect(),
            field_names: Some(
                fields
                    .iter()
                    .map(|f| f.name.name.as_str().to_string())
                    .collect(),
            ),
        },
    }
}

/// Convert an AST type expression to an interface Type.
fn convert_ast_type(ty: &bhc_ast::Type) -> Type {
    match ty {
        bhc_ast::Type::Var(tv, _) => Type::Var(tv.name.name.as_str().to_string()),
        bhc_ast::Type::Con(ident, _) => Type::Con(ident.name.as_str().to_string()),
        bhc_ast::Type::QualCon(module, ident, _) => {
            let qual_name = format!(
                "{}.{}",
                module
                    .parts
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("."),
                ident.name.as_str()
            );
            Type::Con(qual_name)
        }
        bhc_ast::Type::App(f, x, _) => {
            Type::App(Box::new(convert_ast_type(f)), Box::new(convert_ast_type(x)))
        }
        bhc_ast::Type::Fun(a, b, _) => {
            Type::Fun(Box::new(convert_ast_type(a)), Box::new(convert_ast_type(b)))
        }
        bhc_ast::Type::Tuple(ts, _) => Type::Tuple(ts.iter().map(convert_ast_type).collect()),
        bhc_ast::Type::List(t, _) => Type::List(Box::new(convert_ast_type(t))),
        bhc_ast::Type::Paren(t, _) => convert_ast_type(t),
        bhc_ast::Type::Forall(_, inner, _) => convert_ast_type(inner),
        bhc_ast::Type::Constrained(_, inner, _) => convert_ast_type(inner),
        bhc_ast::Type::Bang(inner, _) | bhc_ast::Type::Lazy(inner, _) => convert_ast_type(inner),
        bhc_ast::Type::InfixOp(lhs, op, rhs, _) => {
            // Desugar `a `Op` b` to `Op a b`
            let op_con = Type::Con(op.name.as_str().to_string());
            let app_l = Type::App(Box::new(op_con), Box::new(convert_ast_type(lhs)));
            Type::App(Box::new(app_l), Box::new(convert_ast_type(rhs)))
        }
        bhc_ast::Type::PromotedList(_, _) | bhc_ast::Type::NatLit(_, _) => {
            Type::Con("Unknown".to_string())
        }
    }
}

/// Convert an AST constraint to an interface Constraint.
fn convert_ast_constraint(constraint: &bhc_ast::Constraint) -> Constraint {
    Constraint {
        class: constraint.class.name.as_str().to_string(),
        args: constraint.args.iter().map(convert_ast_type).collect(),
    }
}

/// Compute a kind for a type constructor with the given number of parameters.
fn params_to_kind(n: usize) -> Kind {
    if n == 0 {
        Kind::Type
    } else {
        Kind::fun(Kind::Type, params_to_kind(n - 1))
    }
}

/// Compute a simple hash for a module name (placeholder for content-based hashing).
fn compute_module_hash(module_name: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    module_name.hash(&mut hasher);
    hasher.finish()
}
