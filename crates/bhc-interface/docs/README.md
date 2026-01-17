# bhc-interface

Interface files for the Basel Haskell Compiler.

## Overview

`bhc-interface` handles reading and writing interface files (`.bhi`), which contain the public API of compiled modules. Features:

- **Separate compilation**: Type check without source code
- **Type signatures**: Exported functions and types
- **Instance declarations**: Type class instances
- **Binary format**: Efficient storage with versioning

## Interface File Contents

Interface files contain:

- Module metadata (name, version, hash)
- Exported type signatures
- Exported function signatures
- Instance declarations
- Re-exports from other modules

## Core Types

| Type | Description |
|------|-------------|
| `ModuleInterface` | Complete module interface |
| `InterfaceHeader` | Metadata and version info |
| `ExportedValue` | Function/constant export |
| `ExportedType` | Type export |
| `ExportedClass` | Type class export |
| `ExportedInstance` | Instance export |

## File Format

```
+------------------------+
| Magic: "BHCI" (4 bytes)|
+------------------------+
| Version (u32)          |
+------------------------+
| Bincode-encoded data   |
|   - Header             |
|   - Values             |
|   - Types              |
|   - Classes            |
|   - Instances          |
|   - Dependencies       |
|   - Re-exports         |
+------------------------+
```

## ModuleInterface

```rust
pub struct ModuleInterface {
    pub header: InterfaceHeader,
    pub values: Vec<ExportedValue>,
    pub types: Vec<ExportedType>,
    pub classes: Vec<ExportedClass>,
    pub instances: Vec<ExportedInstance>,
    pub dependencies: Vec<InterfaceDependency>,
    pub reexports: HashMap<String, String>,
}

// Create new interface
let mut interface = ModuleInterface::new("Data.List");

// Add exports
interface.add_value(value);
interface.add_type(ty);
interface.add_class(class);
interface.add_instance(instance);

// Look up exports
let value = interface.lookup_value("map");
let ty = interface.lookup_type("Maybe");
let class = interface.lookup_class("Functor");
```

## Type Signatures

```rust
pub struct TypeSignature {
    /// Quantified type variables
    pub type_vars: Vec<String>,
    /// Type class constraints
    pub constraints: Vec<Constraint>,
    /// The type
    pub ty: Type,
}

pub struct Constraint {
    pub class: String,
    pub args: Vec<Type>,
}

// Example: Eq a => a -> a -> Bool
let sig = TypeSignature {
    type_vars: vec!["a".into()],
    constraints: vec![Constraint {
        class: "Eq".into(),
        args: vec![Type::var("a")],
    }],
    ty: Type::fun(
        Type::var("a"),
        Type::fun(Type::var("a"), Type::con("Bool"))
    ),
};
```

## Type Representation

```rust
pub enum Type {
    /// Type variable: a
    Var(String),
    /// Type constructor: Int, Maybe
    Con(String),
    /// Type application: Maybe Int
    App(Box<Type>, Box<Type>),
    /// Function type: a -> b
    Fun(Box<Type>, Box<Type>),
    /// Tuple: (a, b)
    Tuple(Vec<Type>),
    /// List: [a]
    List(Box<Type>),
}

// Construction helpers
let int = Type::con("Int");
let list_int = Type::app(Type::con("List"), int.clone());
let fun = Type::fun(int, Type::con("Bool"));
```

## Exported Values

```rust
pub struct ExportedValue {
    pub name: String,
    pub signature: TypeSignature,
    pub inline: InlineInfo,
}

pub enum InlineInfo {
    None,      // No annotation
    Inline,    // INLINE pragma
    Inlinable, // INLINABLE pragma
    NoInline,  // NOINLINE pragma
}
```

## Exported Types

```rust
pub struct ExportedType {
    pub name: String,
    pub params: Vec<String>,
    pub kind: Kind,
    pub definition: Option<TypeDefinition>,
}

pub enum Kind {
    Type,                         // *
    Fun(Box<Kind>, Box<Kind>),   // * -> *
}

pub enum TypeDefinition {
    Data(Vec<DataConstructor>),
    Newtype(DataConstructor),
    TypeSynonym(Type),
}

pub struct DataConstructor {
    pub name: String,
    pub fields: Vec<Type>,
    pub field_names: Option<Vec<String>>,
}
```

## Type Classes

```rust
pub struct ExportedClass {
    pub name: String,
    pub params: Vec<String>,
    pub superclasses: Vec<Constraint>,
    pub methods: Vec<ClassMethod>,
}

pub struct ClassMethod {
    pub name: String,
    pub signature: TypeSignature,
    pub has_default: bool,
}

pub struct ExportedInstance {
    pub class: String,
    pub types: Vec<Type>,
    pub constraints: Vec<Constraint>,
}
```

## Reading and Writing

```rust
use bhc_interface::ModuleInterface;

// Write to file
interface.write_to_file("Data/List.bhi")?;

// Read from file
let interface = ModuleInterface::read_from_file("Data/List.bhi")?;

// Write to arbitrary writer
let mut buffer = Vec::new();
interface.write(&mut buffer)?;

// Read from arbitrary reader
let interface = ModuleInterface::read(&mut cursor)?;
```

## Interface Paths

```rust
use bhc_interface::interface_path;

// Get path for module
let path = interface_path("build", "Data.List");
// Returns: build/Data/List.bhi
```

## Dependencies

```rust
pub struct InterfaceDependency {
    pub module: String,
    pub hash: u64,
}

// Track dependencies for consistency checking
for dep in &interface.dependencies {
    println!("Depends on {} (hash: {:x})", dep.module, dep.hash);
}
```

## Error Handling

```rust
pub enum InterfaceError {
    /// File not found
    NotFound(Utf8PathBuf),
    /// Invalid format
    InvalidFormat(String),
    /// Version mismatch
    VersionMismatch { expected: u32, found: u32 },
    /// Serialization error
    Serialization(String),
    /// IO error
    Io(std::io::Error),
}
```

## Version Information

```rust
/// Current interface format version
pub const INTERFACE_VERSION: u32 = 1;

/// Magic bytes
pub const INTERFACE_MAGIC: &[u8; 4] = b"BHCI";

pub struct InterfaceHeader {
    pub version: u32,
    pub module_name: String,
    pub module_hash: u64,
    pub compiler_version: String,
}
```

## See Also

- `bhc-typeck`: Uses interfaces for type checking imports
- `bhc-driver`: Generates interfaces during compilation
- `bhc-package`: Package-level interface management
- GHC interface file documentation (inspiration)
