//! Interface files (.bhi) for BHC.
//!
//! This crate handles reading and writing interface files, which contain
//! the public API of compiled modules. Interface files enable separate
//! compilation and are used for type checking without requiring source code.
//!
//! # File Format
//!
//! Interface files (`.bhi`) contain:
//!
//! - Module metadata (name, version, hash)
//! - Exported type signatures
//! - Exported function signatures
//! - Instance declarations
//! - Re-exports
//!
//! # Usage
//!
//! Interface files are:
//!
//! - Generated during compilation for each module
//! - Used during type checking to get types of imported modules
//! - Distributed with compiled packages
//!
//! # Format Versions
//!
//! The interface format is versioned to allow evolution while maintaining
//! backward compatibility where possible.

#![warn(missing_docs)]

pub mod convert;
pub mod generate;

use camino::{Utf8Path, Utf8PathBuf};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use thiserror::Error;

/// Current interface file format version.
pub const INTERFACE_VERSION: u32 = 1;

/// Magic bytes for interface files.
pub const INTERFACE_MAGIC: &[u8; 4] = b"BHCI";

/// Errors that can occur during interface operations.
#[derive(Debug, Error)]
pub enum InterfaceError {
    /// Interface file not found.
    #[error("interface file not found: {0}")]
    NotFound(Utf8PathBuf),

    /// Invalid interface file format.
    #[error("invalid interface file: {0}")]
    InvalidFormat(String),

    /// Version mismatch.
    #[error("interface version mismatch: expected {expected}, found {found}")]
    VersionMismatch {
        /// Expected version.
        expected: u32,
        /// Found version.
        found: u32,
    },

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for interface operations.
pub type InterfaceResult<T> = Result<T, InterfaceError>;

/// A type signature in the interface.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeSignature {
    /// The type variables (quantified).
    pub type_vars: Vec<String>,
    /// Type class constraints.
    pub constraints: Vec<Constraint>,
    /// The actual type.
    pub ty: Type,
}

/// A type class constraint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraint {
    /// Class name.
    pub class: String,
    /// Type arguments.
    pub args: Vec<Type>,
}

/// A type representation in the interface.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    /// Type variable.
    Var(String),
    /// Type constructor.
    Con(String),
    /// Type application.
    App(Box<Type>, Box<Type>),
    /// Function type.
    Fun(Box<Type>, Box<Type>),
    /// Tuple type.
    Tuple(Vec<Type>),
    /// List type.
    List(Box<Type>),
}

impl Type {
    /// Create a type variable.
    #[must_use]
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// Create a type constructor.
    #[must_use]
    pub fn con(name: impl Into<String>) -> Self {
        Self::Con(name.into())
    }

    /// Create a function type.
    #[must_use]
    pub fn fun(from: Type, to: Type) -> Self {
        Self::Fun(Box::new(from), Box::new(to))
    }

    /// Create a type application.
    #[must_use]
    pub fn app(f: Type, arg: Type) -> Self {
        Self::App(Box::new(f), Box::new(arg))
    }
}

/// An exported value declaration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportedValue {
    /// Value name.
    pub name: String,
    /// Type signature.
    pub signature: TypeSignature,
    /// Inline pragma information.
    #[serde(default)]
    pub inline: InlineInfo,
}

/// Inline pragma information.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum InlineInfo {
    /// No inline annotation.
    #[default]
    None,
    /// INLINE pragma.
    Inline,
    /// INLINABLE pragma.
    Inlinable,
    /// NOINLINE pragma.
    NoInline,
}

/// An exported type declaration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportedType {
    /// Type name.
    pub name: String,
    /// Type parameters.
    pub params: Vec<String>,
    /// Kind signature.
    pub kind: Kind,
    /// Type definition (None for abstract types).
    pub definition: Option<TypeDefinition>,
}

/// A kind (type of types).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    /// Type kind (*).
    Type,
    /// Function kind.
    Fun(Box<Kind>, Box<Kind>),
}

impl Kind {
    /// The Type kind.
    pub const TYPE: Self = Self::Type;

    /// Create a function kind.
    #[must_use]
    pub fn fun(from: Kind, to: Kind) -> Self {
        Self::Fun(Box::new(from), Box::new(to))
    }
}

/// A type definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TypeDefinition {
    /// Algebraic data type.
    Data(Vec<DataConstructor>),
    /// Newtype.
    Newtype(DataConstructor),
    /// Type synonym.
    TypeSynonym(Type),
}

/// A data constructor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataConstructor {
    /// Constructor name.
    pub name: String,
    /// Field types.
    pub fields: Vec<Type>,
    /// Field names (if record).
    pub field_names: Option<Vec<String>>,
}

/// An exported type class.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportedClass {
    /// Class name.
    pub name: String,
    /// Type parameters.
    pub params: Vec<String>,
    /// Superclass constraints.
    pub superclasses: Vec<Constraint>,
    /// Method signatures.
    pub methods: Vec<ClassMethod>,
}

/// A type class method.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassMethod {
    /// Method name.
    pub name: String,
    /// Method type signature.
    pub signature: TypeSignature,
    /// Whether a default implementation exists.
    pub has_default: bool,
}

/// An exported instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportedInstance {
    /// Class name.
    pub class: String,
    /// Instance types.
    pub types: Vec<Type>,
    /// Instance constraints.
    pub constraints: Vec<Constraint>,
}

/// Module interface header.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterfaceHeader {
    /// Interface format version.
    pub version: u32,
    /// Module name.
    pub module_name: String,
    /// Module hash for consistency checking.
    pub module_hash: u64,
    /// Compiler version that generated this interface.
    pub compiler_version: String,
}

/// A complete module interface.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleInterface {
    /// Interface header.
    pub header: InterfaceHeader,
    /// Exported values (functions, constants).
    pub values: Vec<ExportedValue>,
    /// Exported types.
    pub types: Vec<ExportedType>,
    /// Exported type classes.
    pub classes: Vec<ExportedClass>,
    /// Exported instances.
    pub instances: Vec<ExportedInstance>,
    /// Module dependencies (imported interfaces).
    pub dependencies: Vec<InterfaceDependency>,
    /// Re-exports from other modules.
    pub reexports: HashMap<String, String>,
}

/// A dependency on another interface.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterfaceDependency {
    /// Module name.
    pub module: String,
    /// Expected hash.
    pub hash: u64,
}

impl ModuleInterface {
    /// Create a new empty interface.
    #[must_use]
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            header: InterfaceHeader {
                version: INTERFACE_VERSION,
                module_name: module_name.into(),
                module_hash: 0,
                compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            },
            values: Vec::new(),
            types: Vec::new(),
            classes: Vec::new(),
            instances: Vec::new(),
            dependencies: Vec::new(),
            reexports: HashMap::new(),
        }
    }

    /// Add an exported value.
    pub fn add_value(&mut self, value: ExportedValue) {
        self.values.push(value);
    }

    /// Add an exported type.
    pub fn add_type(&mut self, ty: ExportedType) {
        self.types.push(ty);
    }

    /// Add an exported class.
    pub fn add_class(&mut self, class: ExportedClass) {
        self.classes.push(class);
    }

    /// Add an exported instance.
    pub fn add_instance(&mut self, instance: ExportedInstance) {
        self.instances.push(instance);
    }

    /// Look up a value by name.
    #[must_use]
    pub fn lookup_value(&self, name: &str) -> Option<&ExportedValue> {
        self.values.iter().find(|v| v.name == name)
    }

    /// Look up a type by name.
    #[must_use]
    pub fn lookup_type(&self, name: &str) -> Option<&ExportedType> {
        self.types.iter().find(|t| t.name == name)
    }

    /// Look up a class by name.
    #[must_use]
    pub fn lookup_class(&self, name: &str) -> Option<&ExportedClass> {
        self.classes.iter().find(|c| c.name == name)
    }

    /// Write the interface to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_to_file(&self, path: impl AsRef<Utf8Path>) -> InterfaceResult<()> {
        let path = path.as_ref();
        let mut file = std::fs::File::create(path)?;
        self.write(&mut file)
    }

    /// Write the interface to a writer.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn write(&self, writer: &mut impl Write) -> InterfaceResult<()> {
        // Write magic bytes
        writer.write_all(INTERFACE_MAGIC)?;

        // Serialize with bincode
        let encoded =
            bincode::serialize(self).map_err(|e| InterfaceError::Serialization(e.to_string()))?;
        writer.write_all(&encoded)?;

        Ok(())
    }

    /// Read an interface from a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is invalid.
    pub fn read_from_file(path: impl AsRef<Utf8Path>) -> InterfaceResult<Self> {
        let path = path.as_ref();
        let mut file =
            std::fs::File::open(path).map_err(|_| InterfaceError::NotFound(path.to_path_buf()))?;
        Self::read(&mut file)
    }

    /// Read an interface from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn read(reader: &mut impl Read) -> InterfaceResult<Self> {
        // Check magic bytes
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != INTERFACE_MAGIC {
            return Err(InterfaceError::InvalidFormat(
                "invalid magic bytes".to_string(),
            ));
        }

        // Read remaining data
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        // Deserialize
        let interface: Self = bincode::deserialize(&data)
            .map_err(|e| InterfaceError::Serialization(e.to_string()))?;

        // Check version
        if interface.header.version != INTERFACE_VERSION {
            return Err(InterfaceError::VersionMismatch {
                expected: INTERFACE_VERSION,
                found: interface.header.version,
            });
        }

        Ok(interface)
    }
}

/// Get the interface file path for a module.
#[must_use]
pub fn interface_path(output_dir: &Utf8Path, module_name: &str) -> Utf8PathBuf {
    let file_name = module_name.replace('.', "/") + ".bhi";
    output_dir.join(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_interface() {
        let mut interface = ModuleInterface::new("Test.Module");

        interface.add_value(ExportedValue {
            name: "foo".to_string(),
            signature: TypeSignature {
                type_vars: vec!["a".to_string()],
                constraints: vec![],
                ty: Type::fun(Type::var("a"), Type::var("a")),
            },
            inline: InlineInfo::None,
        });

        assert_eq!(interface.header.module_name, "Test.Module");
        assert_eq!(interface.values.len(), 1);
        assert!(interface.lookup_value("foo").is_some());
    }

    #[test]
    fn test_type_construction() {
        let int_type = Type::con("Int");
        let list_int = Type::app(Type::con("List"), int_type.clone());
        let fun_type = Type::fun(int_type, Type::con("Bool"));

        assert!(matches!(fun_type, Type::Fun(_, _)));
        assert!(matches!(list_int, Type::App(_, _)));
    }

    #[test]
    fn test_interface_roundtrip() {
        let mut interface = ModuleInterface::new("Test");
        interface.add_type(ExportedType {
            name: "MyType".to_string(),
            params: vec!["a".to_string()],
            kind: Kind::fun(Kind::TYPE, Kind::TYPE),
            definition: None,
        });

        let mut buffer = Vec::new();
        interface.write(&mut buffer).unwrap();

        let mut cursor = std::io::Cursor::new(buffer);
        let loaded = ModuleInterface::read(&mut cursor).unwrap();

        assert_eq!(loaded.header.module_name, "Test");
        assert_eq!(loaded.types.len(), 1);
        assert_eq!(loaded.types[0].name, "MyType");
    }
}
