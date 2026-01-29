//! Documentation extraction from AST.
//!
//! This module parses Haskell source files and extracts documentation
//! from the AST into the internal documentation model.

use anyhow::{Context, Result};
use bhc_ast::{
    ClassDecl, ConDecl, ConFields, DataDecl, Decl, DocComment, FunBind, InstanceDecl, Module,
    NewtypeDecl, TypeAlias, TypeSig,
};
use bhc_parser::Parser;
use bhc_span::FileId;
use std::path::Path;

use crate::haddock;
use crate::model::{
    Annotations, ClassDoc, ConstructorDoc, DocContent, DocItem, FieldDoc, FieldsDoc, FunctionDoc,
    InstanceDoc, ModuleDoc, NewtypeDoc, SourceLocation, TypeAliasDoc, TypeDoc,
};

/// Extract documentation from a source file.
pub fn extract_file(path: &Path) -> Result<ModuleDoc> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {:?}", path))?;
    extract_source(&source, path)
}

/// Extract documentation from source code.
pub fn extract_source(source: &str, path: &Path) -> Result<ModuleDoc> {
    // Parse the module
    let mut parser = Parser::new(source, FileId::new(0));
    let module = parser
        .parse_module()
        .map_err(|e| anyhow::anyhow!("Parse error: {:?}", e))?;

    // Extract documentation from the AST
    extract_module(&module, path, source)
}

/// Extract documentation from a parsed module.
pub fn extract_module(module: &Module, path: &Path, source: &str) -> Result<ModuleDoc> {
    let name = module
        .name
        .as_ref()
        .map(|n| {
            n.parts
                .iter()
                .map(|p| p.as_str())
                .collect::<Vec<_>>()
                .join(".")
        })
        .unwrap_or_else(|| {
            path.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        });

    let doc = module.doc.as_ref().map(|d| extract_doc_content(d));

    let mut items = Vec::new();
    let mut type_sigs: std::collections::HashMap<String, TypeSig> =
        std::collections::HashMap::new();

    // First pass: collect type signatures
    for decl in &module.decls {
        if let Decl::TypeSig(sig) = decl {
            for name in &sig.names {
                type_sigs.insert(name.name.as_str().to_string(), sig.clone());
            }
        }
    }

    // Second pass: extract documentation for each declaration
    for decl in &module.decls {
        match decl {
            Decl::TypeSig(sig) => {
                // Type signatures are handled with their bindings
                // But if there's no binding, document the signature
                let has_binding = module.decls.iter().any(|d| {
                    matches!(d, Decl::FunBind(f) if sig.names.iter().any(|n| n.name == f.name.name))
                });
                if !has_binding {
                    for name in &sig.names {
                        items.push(DocItem::Function(extract_function_from_sig(
                            sig,
                            name.name.as_str(),
                            path,
                            source,
                        )));
                    }
                }
            }
            Decl::FunBind(bind) => {
                // Skip internal pattern bindings
                if bind.name.name.as_str() == "$patbind" {
                    continue;
                }
                let sig = type_sigs.get(bind.name.name.as_str());
                items.push(DocItem::Function(extract_function(bind, sig, path, source)));
            }
            Decl::DataDecl(data) => {
                items.push(DocItem::Type(extract_data_type(data, path, source)));
            }
            Decl::TypeAlias(alias) => {
                items.push(DocItem::TypeAlias(extract_type_alias(alias, path, source)));
            }
            Decl::Newtype(newtype) => {
                items.push(DocItem::Newtype(extract_newtype(newtype, path, source)));
            }
            Decl::ClassDecl(class) => {
                items.push(DocItem::Class(extract_class(class, path, source)));
            }
            Decl::InstanceDecl(inst) => {
                items.push(DocItem::Instance(extract_instance(inst, path, source)));
            }
            _ => {}
        }
    }

    Ok(ModuleDoc {
        name,
        doc,
        items,
        reexports: vec![],
        submodules: vec![],
    })
}

fn extract_doc_content(doc: &DocComment) -> DocContent {
    haddock::parse(&doc.text)
}

/// Compute line and column from byte offset
fn byte_offset_to_line_col(source: &str, offset: u32) -> (usize, usize) {
    let offset = offset as usize;
    let before = &source[..offset.min(source.len())];
    let line = before.matches('\n').count() + 1;
    let col = before.rfind('\n').map(|i| offset - i).unwrap_or(offset + 1);
    (line, col)
}

fn make_source_location(path: &Path, source: &str, span: bhc_span::Span) -> SourceLocation {
    let (line, column) = byte_offset_to_line_col(source, span.lo.as_u32());
    SourceLocation {
        file: path.to_string_lossy().into_owned(),
        line: line as u32,
        column: column as u32,
    }
}

fn extract_function(
    bind: &FunBind,
    sig: Option<&TypeSig>,
    path: &Path,
    source: &str,
) -> FunctionDoc {
    let name = bind.name.name.as_str().to_string();

    // Prefer doc from type signature, then from binding
    let doc = sig
        .and_then(|s| s.doc.as_ref())
        .or(bind.doc.as_ref())
        .map(extract_doc_content);

    let signature = sig.map(|s| format_type(&s.ty)).unwrap_or_default();

    FunctionDoc {
        name,
        signature,
        signature_parsed: None, // TODO: Parse type for search
        doc,
        annotations: Annotations::default(),
        source: Some(make_source_location(path, source, bind.span)),
    }
}

fn extract_function_from_sig(sig: &TypeSig, name: &str, path: &Path, source: &str) -> FunctionDoc {
    let doc = sig.doc.as_ref().map(extract_doc_content);
    let signature = format_type(&sig.ty);

    FunctionDoc {
        name: name.to_string(),
        signature,
        signature_parsed: None,
        doc,
        annotations: Annotations::default(),
        source: Some(make_source_location(path, source, sig.span)),
    }
}

fn extract_data_type(data: &DataDecl, path: &Path, source: &str) -> TypeDoc {
    let name = data.name.name.as_str().to_string();
    let params = data
        .params
        .iter()
        .map(|p| p.name.name.as_str().to_string())
        .collect();
    let doc = data.doc.as_ref().map(extract_doc_content);
    let constructors = data.constrs.iter().map(extract_constructor).collect();
    let deriving = data
        .deriving
        .iter()
        .map(|d| d.name.as_str().to_string())
        .collect();

    TypeDoc {
        name,
        params,
        doc,
        constructors,
        deriving,
        source: Some(make_source_location(path, source, data.span)),
    }
}

fn extract_constructor(con: &ConDecl) -> ConstructorDoc {
    let name = con.name.name.as_str().to_string();
    let doc = con.doc.as_ref().map(extract_doc_content);
    let fields = match &con.fields {
        ConFields::Positional(types) => FieldsDoc::Positional {
            types: types.iter().map(format_type).collect(),
        },
        ConFields::Record(fields) => FieldsDoc::Record {
            fields: fields
                .iter()
                .map(|f| FieldDoc {
                    name: f.name.name.as_str().to_string(),
                    ty: format_type(&f.ty),
                    doc: f.doc.as_ref().map(extract_doc_content),
                })
                .collect(),
        },
    };

    ConstructorDoc { name, fields, doc }
}

fn extract_type_alias(alias: &TypeAlias, path: &Path, source: &str) -> TypeAliasDoc {
    TypeAliasDoc {
        name: alias.name.name.as_str().to_string(),
        params: alias
            .params
            .iter()
            .map(|p| p.name.name.as_str().to_string())
            .collect(),
        rhs: format_type(&alias.ty),
        doc: alias.doc.as_ref().map(extract_doc_content),
        source: Some(make_source_location(path, source, alias.span)),
    }
}

fn extract_newtype(newtype: &NewtypeDecl, path: &Path, source: &str) -> NewtypeDoc {
    NewtypeDoc {
        name: newtype.name.name.as_str().to_string(),
        params: newtype
            .params
            .iter()
            .map(|p| p.name.name.as_str().to_string())
            .collect(),
        constructor: extract_constructor(&newtype.constr),
        doc: newtype.doc.as_ref().map(extract_doc_content),
        deriving: newtype
            .deriving
            .iter()
            .map(|d| d.name.as_str().to_string())
            .collect(),
        source: Some(make_source_location(path, source, newtype.span)),
    }
}

fn extract_class(class: &ClassDecl, path: &Path, source: &str) -> ClassDoc {
    let methods = class
        .methods
        .iter()
        .filter_map(|d| match d {
            Decl::TypeSig(sig) => Some(
                sig.names
                    .iter()
                    .map(|n| extract_function_from_sig(sig, n.name.as_str(), path, source))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect();

    ClassDoc {
        name: class.name.name.as_str().to_string(),
        params: class
            .params
            .iter()
            .map(|p| p.name.name.as_str().to_string())
            .collect(),
        superclasses: class.context.iter().map(|c| format_constraint(c)).collect(),
        fundeps: class
            .fundeps
            .iter()
            .map(|f| {
                let from: Vec<_> = f.from.iter().map(|i| i.name.as_str()).collect();
                let to: Vec<_> = f.to.iter().map(|i| i.name.as_str()).collect();
                format!("{} -> {}", from.join(" "), to.join(" "))
            })
            .collect(),
        doc: class.doc.as_ref().map(extract_doc_content),
        methods,
        assoc_types: vec![], // TODO: Extract associated types
        source: Some(make_source_location(path, source, class.span)),
    }
}

fn extract_instance(inst: &InstanceDecl, path: &Path, source: &str) -> InstanceDoc {
    InstanceDoc {
        class: inst.class.name.as_str().to_string(),
        ty: format_type(&inst.ty),
        context: inst.context.iter().map(|c| format_constraint(c)).collect(),
        doc: inst.doc.as_ref().map(extract_doc_content),
        source: Some(make_source_location(path, source, inst.span)),
    }
}

fn format_type(ty: &bhc_ast::Type) -> String {
    match ty {
        bhc_ast::Type::Var(name, _) => name.name.name.as_str().to_string(),
        bhc_ast::Type::Con(name, _) => name.name.as_str().to_string(),
        bhc_ast::Type::QualCon(mod_name, name, _) => {
            let parts: Vec<_> = mod_name.parts.iter().map(|p| p.as_str()).collect();
            format!("{}.{}", parts.join("."), name.name.as_str())
        }
        bhc_ast::Type::App(f, arg, _) => format!("{} {}", format_type(f), format_type_atom(arg)),
        bhc_ast::Type::Fun(from, to, _) => {
            format!("{} -> {}", format_type_atom(from), format_type(to))
        }
        bhc_ast::Type::Tuple(elems, _) => {
            if elems.is_empty() {
                "()".to_string()
            } else {
                let parts: Vec<_> = elems.iter().map(format_type).collect();
                format!("({})", parts.join(", "))
            }
        }
        bhc_ast::Type::List(elem, _) => format!("[{}]", format_type(elem)),
        bhc_ast::Type::Paren(inner, _) => format!("({})", format_type(inner)),
        bhc_ast::Type::Forall(vars, ty, _) => {
            let var_names: Vec<_> = vars.iter().map(|v| v.name.name.as_str()).collect();
            format!("forall {}. {}", var_names.join(" "), format_type(ty))
        }
        bhc_ast::Type::Constrained(ctx, ty, _) => {
            let constraints: Vec<_> = ctx.iter().map(format_constraint).collect();
            format!("({}) => {}", constraints.join(", "), format_type(ty))
        }
        bhc_ast::Type::PromotedList(elems, _) => {
            let parts: Vec<_> = elems.iter().map(format_type).collect();
            format!("'[{}]", parts.join(", "))
        }
        bhc_ast::Type::NatLit(n, _) => n.to_string(),
        bhc_ast::Type::Bang(inner, _) => format!("!{}", format_type_atom(inner)),
        bhc_ast::Type::Lazy(inner, _) => format!("~{}", format_type_atom(inner)),
    }
}

fn format_type_atom(ty: &bhc_ast::Type) -> String {
    match ty {
        bhc_ast::Type::Var(_, _)
        | bhc_ast::Type::Con(_, _)
        | bhc_ast::Type::QualCon(_, _, _)
        | bhc_ast::Type::Tuple(_, _)
        | bhc_ast::Type::List(_, _)
        | bhc_ast::Type::Paren(_, _)
        | bhc_ast::Type::NatLit(_, _) => format_type(ty),
        _ => format!("({})", format_type(ty)),
    }
}

fn format_constraint(c: &bhc_ast::Constraint) -> String {
    let args: Vec<_> = c.args.iter().map(format_type).collect();
    if args.is_empty() {
        c.class.name.as_str().to_string()
    } else {
        format!("{} {}", c.class.name.as_str(), args.join(" "))
    }
}
