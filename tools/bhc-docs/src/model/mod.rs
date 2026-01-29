//! Documentation model types.
//!
//! This module defines the internal representation of documentation,
//! independent of the source format (AST) or output format (HTML/Markdown).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A documented module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDoc {
    /// Module name (e.g., "Data.List").
    pub name: String,

    /// Module-level documentation.
    pub doc: Option<DocContent>,

    /// Exported items grouped by category.
    pub items: Vec<DocItem>,

    /// Re-exports from other modules.
    pub reexports: Vec<ReExport>,

    /// Submodules (for hierarchical display).
    pub submodules: Vec<String>,
}

/// A documented item (function, type, class, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum DocItem {
    /// Function or value.
    Function(FunctionDoc),
    /// Data type.
    Type(TypeDoc),
    /// Type alias.
    TypeAlias(TypeAliasDoc),
    /// Newtype.
    Newtype(NewtypeDoc),
    /// Type class.
    Class(ClassDoc),
    /// Type class instance.
    Instance(InstanceDoc),
}

impl DocItem {
    /// Get the name of this item.
    pub fn name(&self) -> &str {
        match self {
            Self::Function(f) => &f.name,
            Self::Type(t) => &t.name,
            Self::TypeAlias(t) => &t.name,
            Self::Newtype(n) => &n.name,
            Self::Class(c) => &c.name,
            Self::Instance(i) => &i.class,
        }
    }

    /// Get the documentation for this item.
    pub fn doc(&self) -> Option<&DocContent> {
        match self {
            Self::Function(f) => f.doc.as_ref(),
            Self::Type(t) => t.doc.as_ref(),
            Self::TypeAlias(t) => t.doc.as_ref(),
            Self::Newtype(n) => n.doc.as_ref(),
            Self::Class(c) => c.doc.as_ref(),
            Self::Instance(i) => i.doc.as_ref(),
        }
    }

    /// Check if this item has documentation.
    pub fn is_documented(&self) -> bool {
        self.doc().is_some()
    }
}

/// Documentation for a function or value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDoc {
    /// Function name.
    pub name: String,

    /// Type signature (rendered as string).
    pub signature: String,

    /// Parsed type for search indexing.
    pub signature_parsed: Option<TypeSignature>,

    /// Documentation content.
    pub doc: Option<DocContent>,

    /// BHC-specific annotations.
    pub annotations: Annotations,

    /// Source location.
    pub source: Option<SourceLocation>,
}

/// Documentation for a data type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDoc {
    /// Type name.
    pub name: String,

    /// Type parameters.
    pub params: Vec<String>,

    /// Documentation content.
    pub doc: Option<DocContent>,

    /// Constructors.
    pub constructors: Vec<ConstructorDoc>,

    /// Derived instances.
    pub deriving: Vec<String>,

    /// Source location.
    pub source: Option<SourceLocation>,
}

/// Documentation for a type alias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeAliasDoc {
    /// Alias name.
    pub name: String,

    /// Type parameters.
    pub params: Vec<String>,

    /// The aliased type (rendered).
    pub rhs: String,

    /// Documentation content.
    pub doc: Option<DocContent>,

    /// Source location.
    pub source: Option<SourceLocation>,
}

/// Documentation for a newtype.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewtypeDoc {
    /// Newtype name.
    pub name: String,

    /// Type parameters.
    pub params: Vec<String>,

    /// Constructor.
    pub constructor: ConstructorDoc,

    /// Documentation content.
    pub doc: Option<DocContent>,

    /// Derived instances.
    pub deriving: Vec<String>,

    /// Source location.
    pub source: Option<SourceLocation>,
}

/// Documentation for a data constructor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructorDoc {
    /// Constructor name.
    pub name: String,

    /// Fields (either positional or named).
    pub fields: FieldsDoc,

    /// Documentation content.
    pub doc: Option<DocContent>,
}

/// Constructor fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum FieldsDoc {
    /// Positional fields.
    Positional { types: Vec<String> },
    /// Record fields.
    Record { fields: Vec<FieldDoc> },
}

/// A record field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDoc {
    /// Field name.
    pub name: String,
    /// Field type (rendered).
    pub ty: String,
    /// Documentation.
    pub doc: Option<DocContent>,
}

/// Documentation for a type class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassDoc {
    /// Class name.
    pub name: String,

    /// Type parameters.
    pub params: Vec<String>,

    /// Superclass constraints.
    pub superclasses: Vec<String>,

    /// Functional dependencies.
    pub fundeps: Vec<String>,

    /// Documentation content.
    pub doc: Option<DocContent>,

    /// Method signatures.
    pub methods: Vec<FunctionDoc>,

    /// Associated types.
    pub assoc_types: Vec<TypeAliasDoc>,

    /// Source location.
    pub source: Option<SourceLocation>,
}

/// Documentation for a type class instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceDoc {
    /// Class name.
    pub class: String,

    /// Instance type (rendered).
    pub ty: String,

    /// Instance constraints.
    pub context: Vec<String>,

    /// Documentation content.
    pub doc: Option<DocContent>,

    /// Source location.
    pub source: Option<SourceLocation>,
}

/// A re-export from another module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReExport {
    /// The item name.
    pub name: String,
    /// Original module.
    pub original_module: String,
}

/// Structured documentation content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocContent {
    /// Brief description (first paragraph).
    pub brief: String,

    /// Full description (all paragraphs).
    pub description: String,

    /// Named sections (e.g., "Examples", "Complexity").
    pub sections: HashMap<String, String>,

    /// Examples with code.
    pub examples: Vec<Example>,

    /// See also links.
    pub see_also: Vec<String>,

    /// Since version.
    pub since: Option<String>,

    /// Deprecated message.
    pub deprecated: Option<String>,
}

/// A code example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Example code.
    pub code: String,

    /// Expected output (for `>>>` style examples).
    pub output: Option<String>,

    /// Is this example runnable in the playground?
    pub runnable: bool,
}

/// BHC-specific annotations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Annotations {
    /// Complexity annotation (e.g., "O(n log n)").
    pub complexity: Option<String>,

    /// Fusion behavior.
    pub fusion: Option<FusionInfo>,

    /// SIMD support.
    pub simd: Option<SimdInfo>,

    /// Profile-specific behavior.
    pub profiles: HashMap<String, String>,
}

/// Fusion information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionInfo {
    /// Does this function participate in fusion?
    pub fusible: bool,
    /// Fusion rules.
    pub rules: Vec<String>,
}

/// SIMD information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdInfo {
    /// Is this function SIMD-accelerated?
    pub accelerated: bool,
    /// Vector width.
    pub width: Option<u32>,
}

/// A parsed type signature for search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSignature {
    /// Type variables in the signature.
    pub vars: Vec<String>,
    /// Constraints.
    pub constraints: Vec<String>,
    /// The type structure.
    pub ty: TypeExpr,
}

/// A type expression for search matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum TypeExpr {
    /// Type variable.
    Var { name: String },
    /// Type constructor.
    Con { name: String },
    /// Type application.
    App {
        func: Box<TypeExpr>,
        arg: Box<TypeExpr>,
    },
    /// Function type.
    Arrow {
        from: Box<TypeExpr>,
        to: Box<TypeExpr>,
    },
    /// Tuple type.
    Tuple { elements: Vec<TypeExpr> },
    /// List type.
    List { elem: Box<TypeExpr> },
}

/// Source location for linking to source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File path (relative to source root).
    pub file: String,
    /// Line number.
    pub line: u32,
    /// Column number.
    pub column: u32,
}
