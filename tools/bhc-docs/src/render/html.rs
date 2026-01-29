//! HTML documentation renderer.
//!
//! Generates beautiful HTML documentation with:
//! - Responsive design with sidebar navigation
//! - Dark mode with CSS variables
//! - Syntax highlighting via syntect
//! - Keyboard navigation (/, j/k)
//! - BHC badges (fusion, SIMD, profiles)

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use syntect::highlighting::ThemeSet;
use syntect::html::highlighted_html_for_string;
use syntect::parsing::SyntaxSet;
use tera::{Context, Tera};

use super::RenderConfig;
use crate::model::{ClassDoc, DocItem, FieldsDoc, FunctionDoc, ModuleDoc, TypeDoc};

/// Render documentation to HTML.
pub fn render(docs: &[ModuleDoc], output: &Path, config: &RenderConfig) -> Result<()> {
    let tera = create_templates()?;
    let ss = SyntaxSet::load_defaults_newlines();
    let ts = ThemeSet::load_defaults();
    let theme = ts
        .themes
        .get(&config.theme)
        .unwrap_or_else(|| ts.themes.values().next().unwrap());

    // Build all_modules list for sidebar
    let all_modules: Vec<HashMap<String, String>> = docs
        .iter()
        .map(|m| {
            let mut map = HashMap::new();
            map.insert("name".to_string(), m.name.clone());
            map.insert("path".to_string(), m.name.replace('.', "-"));
            map
        })
        .collect();

    // Write CSS
    write_assets(output)?;

    // Write index
    write_index(docs, output, &tera, config)?;

    // Write each module
    for module in docs {
        write_module(module, &all_modules, output, &tera, &ss, theme, config)?;
    }

    // Write search index
    write_search_index(docs, output)?;

    Ok(())
}

fn create_templates() -> Result<Tera> {
    let mut tera = Tera::default();

    // Base template
    tera.add_raw_template("base.html", include_str!("../../templates/base.html"))?;
    tera.add_raw_template("index.html", include_str!("../../templates/index.html"))?;
    tera.add_raw_template("module.html", include_str!("../../templates/module.html"))?;
    tera.add_raw_template("search.html", include_str!("../../templates/search.html"))?;

    Ok(tera)
}

fn write_assets(output: &Path) -> Result<()> {
    let css_dir = output.join("css");
    std::fs::create_dir_all(&css_dir)?;

    std::fs::write(
        css_dir.join("style.css"),
        include_str!("../../assets/css/style.css"),
    )?;

    let js_dir = output.join("js");
    std::fs::create_dir_all(&js_dir)?;

    std::fs::write(
        js_dir.join("main.js"),
        include_str!("../../assets/js/main.js"),
    )?;

    Ok(())
}

fn write_index(
    docs: &[ModuleDoc],
    output: &Path,
    tera: &Tera,
    config: &RenderConfig,
) -> Result<()> {
    let mut context = Context::new();

    // Transform modules for template
    let modules: Vec<HashMap<String, String>> = docs
        .iter()
        .map(|m| {
            let mut map = HashMap::new();
            map.insert("name".to_string(), m.name.clone());
            map.insert("path".to_string(), m.name.replace('.', "-"));
            if let Some(doc) = &m.doc {
                map.insert("brief".to_string(), doc.brief.clone());
            }
            map
        })
        .collect();

    // Count totals
    let total_functions: usize = docs
        .iter()
        .map(|m| {
            m.items
                .iter()
                .filter(|i| matches!(i, DocItem::Function(_)))
                .count()
        })
        .sum();
    let total_types: usize = docs
        .iter()
        .map(|m| {
            m.items
                .iter()
                .filter(|i| {
                    matches!(
                        i,
                        DocItem::Type(_) | DocItem::TypeAlias(_) | DocItem::Newtype(_)
                    )
                })
                .count()
        })
        .sum();

    context.insert("title", "Documentation");
    context.insert("modules", &modules);
    context.insert("total_functions", &total_functions);
    context.insert("total_types", &total_types);
    context.insert("base_url", &config.base_url.as_deref().unwrap_or(""));
    context.insert("playground", &config.playground);
    context.insert("version", &config.version);
    context.insert("versions", &config.versions);

    let html = tera.render("index.html", &context)?;
    std::fs::write(output.join("index.html"), html)?;

    Ok(())
}

fn write_module(
    module: &ModuleDoc,
    all_modules: &[HashMap<String, String>],
    output: &Path,
    tera: &Tera,
    ss: &SyntaxSet,
    theme: &syntect::highlighting::Theme,
    config: &RenderConfig,
) -> Result<()> {
    let syntax = ss.find_syntax_by_extension("hs").unwrap();

    // Group items by category
    let mut functions = Vec::new();
    let mut types = Vec::new();
    let mut classes = Vec::new();
    let mut instances = Vec::new();

    for item in &module.items {
        match item {
            DocItem::Function(f) => functions.push(render_function(f, ss, syntax, theme)?),
            DocItem::Type(t) => types.push(render_type(t, ss, syntax, theme)?),
            DocItem::TypeAlias(t) => types.push(render_type_alias(t)?),
            DocItem::Newtype(n) => types.push(render_newtype(n)?),
            DocItem::Class(c) => classes.push(render_class(c, ss, syntax, theme)?),
            DocItem::Instance(i) => instances.push(render_instance(i)?),
        }
    }

    // Get source file info from first item if available
    let source_file = module
        .items
        .iter()
        .find_map(|item| match item {
            DocItem::Function(f) => f.source.as_ref(),
            DocItem::Type(t) => t.source.as_ref(),
            DocItem::TypeAlias(t) => t.source.as_ref(),
            DocItem::Newtype(n) => n.source.as_ref(),
            DocItem::Class(c) => c.source.as_ref(),
            DocItem::Instance(i) => i.source.as_ref(),
        })
        .map(|s| s.file.clone());

    // Build source URL from config
    let source_url = source_file.as_ref().and_then(|file| {
        config
            .source_base_url
            .as_ref()
            .map(|base| format!("{}/{}", base, file))
    });

    // Build the module object the template expects
    let module_data = serde_json::json!({
        "name": module.name,
        "doc": module.doc.as_ref().map(|d| &d.description),
        "types": types,
        "classes": classes,
        "functions": functions,
        "instances": instances,
        "badges": [],  // Empty for now
        "source_file": source_file,
        "source_url": source_url,
    });

    let mut context = Context::new();
    context.insert("module", &module_data);
    context.insert("all_modules", all_modules);
    context.insert("base_url", &config.base_url.as_deref().unwrap_or(""));
    context.insert("playground", &config.playground);
    context.insert("version", &config.version);
    context.insert("versions", &config.versions);

    let html = tera.render("module.html", &context)?;

    // Create module directory path
    let module_path = module.name.replace('.', "/");
    let module_dir = output.join(&module_path);
    std::fs::create_dir_all(&module_dir)?;
    std::fs::write(module_dir.join("index.html"), html)?;

    // Also write to flat name for convenience
    let flat_name = format!("{}.html", module.name.replace('.', "-"));
    std::fs::write(
        output.join(&flat_name),
        &std::fs::read(module_dir.join("index.html"))?,
    )?;

    Ok(())
}

fn render_function(
    func: &FunctionDoc,
    _ss: &SyntaxSet,
    _syntax: &syntect::parsing::SyntaxReference,
    _theme: &syntect::highlighting::Theme,
) -> Result<serde_json::Value> {
    let mut badges = Vec::new();

    // Annotations badges
    if func
        .annotations
        .fusion
        .as_ref()
        .map(|f| f.fusible)
        .unwrap_or(false)
    {
        badges.push(serde_json::json!({"kind": "fusion", "label": "Fusion", "title": "This function participates in fusion"}));
    }
    if func
        .annotations
        .simd
        .as_ref()
        .map(|s| s.accelerated)
        .unwrap_or(false)
    {
        badges.push(
            serde_json::json!({"kind": "simd", "label": "SIMD", "title": "SIMD accelerated"}),
        );
    }
    if let Some(complexity) = &func.annotations.complexity {
        badges.push(serde_json::json!({"kind": "complexity", "label": complexity, "title": "Time complexity"}));
    }

    Ok(serde_json::json!({
        "name": func.name,
        "signature": func.signature,
        "doc": func.doc.as_ref().map(|d| &d.description),
        "badges": badges,
        "examples": [],  // TODO: extract examples
        "source_line": func.source.as_ref().map(|s| s.line),
    }))
}

fn render_type(
    ty: &TypeDoc,
    _ss: &SyntaxSet,
    _syntax: &syntect::parsing::SyntaxReference,
    _theme: &syntect::highlighting::Theme,
) -> Result<serde_json::Value> {
    let params = ty.params.join(" ");

    // Build constructors
    let constructors: Vec<serde_json::Value> = ty
        .constructors
        .iter()
        .map(|con| {
            let fields_str = match &con.fields {
                FieldsDoc::Positional { types } => types.join(" "),
                FieldsDoc::Record { fields } => {
                    let parts: Vec<_> = fields
                        .iter()
                        .map(|f| format!("{} :: {}", f.name, f.ty))
                        .collect();
                    format!("{{ {} }}", parts.join(", "))
                }
            };
            serde_json::json!({
                "name": con.name,
                "fields": fields_str,
                "doc": con.doc.as_ref().map(|d| &d.brief)
            })
        })
        .collect();

    Ok(serde_json::json!({
        "name": ty.name,
        "keyword": "data",
        "params": params,
        "constructors": constructors,
        "doc": ty.doc.as_ref().map(|d| &d.description),
        "deriving": ty.deriving.join(", "),
        "badges": [],
        "source_line": ty.source.as_ref().map(|s| s.line),
    }))
}

fn render_type_alias(ty: &crate::model::TypeAliasDoc) -> Result<serde_json::Value> {
    let params = ty.params.join(" ");

    Ok(serde_json::json!({
        "name": ty.name,
        "keyword": "type",
        "params": if params.is_empty() { format!("= {}", ty.rhs) } else { format!("{} = {}", params, ty.rhs) },
        "constructors": [],
        "doc": ty.doc.as_ref().map(|d| &d.description),
        "badges": [],
        "source_line": ty.source.as_ref().map(|s| s.line),
    }))
}

fn render_newtype(ty: &crate::model::NewtypeDoc) -> Result<serde_json::Value> {
    let params = ty.params.join(" ");
    let con = &ty.constructor;
    let fields_str = match &con.fields {
        FieldsDoc::Positional { types } => types.join(" "),
        FieldsDoc::Record { fields } => {
            let parts: Vec<_> = fields
                .iter()
                .map(|f| format!("{} :: {}", f.name, f.ty))
                .collect();
            format!("{{ {} }}", parts.join(", "))
        }
    };

    Ok(serde_json::json!({
        "name": ty.name,
        "keyword": "newtype",
        "params": params,
        "constructors": [{
            "name": con.name,
            "fields": fields_str,
            "doc": con.doc.as_ref().map(|d| &d.brief)
        }],
        "doc": ty.doc.as_ref().map(|d| &d.description),
        "badges": [],
        "source_line": ty.source.as_ref().map(|s| s.line),
    }))
}

fn render_class(
    class: &ClassDoc,
    _ss: &SyntaxSet,
    _syntax: &syntect::parsing::SyntaxReference,
    _theme: &syntect::highlighting::Theme,
) -> Result<serde_json::Value> {
    let params = class.params.join(" ");
    let context = if class.superclasses.is_empty() {
        None
    } else {
        Some(class.superclasses.join(", "))
    };

    // Methods
    let methods: Vec<serde_json::Value> = class
        .methods
        .iter()
        .map(|method| {
            serde_json::json!({
                "name": method.name,
                "signature": method.signature,
                "doc": method.doc.as_ref().map(|d| &d.brief)
            })
        })
        .collect();

    Ok(serde_json::json!({
        "name": class.name,
        "params": params,
        "context": context,
        "methods": methods,
        "doc": class.doc.as_ref().map(|d| &d.description),
        "source_line": class.source.as_ref().map(|s| s.line),
    }))
}

fn render_instance(inst: &crate::model::InstanceDoc) -> Result<serde_json::Value> {
    let context = if inst.context.is_empty() {
        None
    } else {
        Some(inst.context.join(", "))
    };
    let head = format!("{} {}", inst.class, inst.ty);

    Ok(serde_json::json!({
        "head": head,
        "context": context,
        "doc": inst.doc.as_ref().map(|d| &d.brief),
        "source_line": inst.source.as_ref().map(|s| s.line),
    }))
}

fn write_search_index(docs: &[ModuleDoc], output: &Path) -> Result<()> {
    let mut index = Vec::new();

    for module in docs {
        for item in &module.items {
            let entry = serde_json::json!({
                "module": module.name,
                "name": item.name(),
                "kind": match item {
                    DocItem::Function(_) => "function",
                    DocItem::Type(_) => "type",
                    DocItem::TypeAlias(_) => "type",
                    DocItem::Newtype(_) => "newtype",
                    DocItem::Class(_) => "class",
                    DocItem::Instance(_) => "instance",
                },
                "signature": match item {
                    DocItem::Function(f) => Some(&f.signature),
                    _ => None,
                },
                "doc": item.doc().map(|d| &d.brief),
            });
            index.push(entry);
        }
    }

    std::fs::write(
        output.join("search-index.json"),
        serde_json::to_string(&index)?,
    )?;

    Ok(())
}
