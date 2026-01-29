//! File path manipulation utilities
//!
//! This module provides cross-platform path manipulation functions.
//! All functions work with path strings and handle both Unix and Windows
//! path separators appropriately.
//!
//! # Example
//!
//! ```
//! use bhc_system::filepath::{join, file_name, extension, parent};
//!
//! let path = join(&["home", "user", "documents", "file.txt"]);
//! assert_eq!(file_name(&path), Some("file.txt"));
//! assert_eq!(extension(&path), Some("txt"));
//! ```

use std::path::{Path, PathBuf, MAIN_SEPARATOR};

/// Join path components into a single path
///
/// # Example
///
/// ```
/// use bhc_system::filepath::join;
///
/// let path = join(&["home", "user", "file.txt"]);
/// // On Unix: "home/user/file.txt"
/// // On Windows: "home\\user\\file.txt"
/// ```
pub fn join(components: &[&str]) -> String {
    let path: PathBuf = components.iter().collect();
    path.to_string_lossy().to_string()
}

/// Join two paths
///
/// # Example
///
/// ```
/// use bhc_system::filepath::join2;
///
/// let path = join2("/home/user", "file.txt");
/// assert!(path.ends_with("file.txt"));
/// ```
pub fn join2(base: &str, path: &str) -> String {
    let base_path = Path::new(base);
    base_path.join(path).to_string_lossy().to_string()
}

/// Get the file name component of a path
///
/// # Example
///
/// ```
/// use bhc_system::filepath::file_name;
///
/// assert_eq!(file_name("/home/user/file.txt"), Some("file.txt"));
/// assert_eq!(file_name("/"), None);
/// ```
pub fn file_name(path: &str) -> Option<&str> {
    Path::new(path).file_name().and_then(|s| s.to_str())
}

/// Get the file stem (name without extension)
///
/// # Example
///
/// ```
/// use bhc_system::filepath::stem;
///
/// assert_eq!(stem("/home/user/file.txt"), Some("file"));
/// assert_eq!(stem("/home/user/archive.tar.gz"), Some("archive.tar"));
/// ```
pub fn stem(path: &str) -> Option<&str> {
    Path::new(path).file_stem().and_then(|s| s.to_str())
}

/// Get the file extension
///
/// # Example
///
/// ```
/// use bhc_system::filepath::extension;
///
/// assert_eq!(extension("file.txt"), Some("txt"));
/// assert_eq!(extension("archive.tar.gz"), Some("gz"));
/// assert_eq!(extension("no_extension"), None);
/// ```
pub fn extension(path: &str) -> Option<&str> {
    Path::new(path).extension().and_then(|s| s.to_str())
}

/// Get the parent directory
///
/// # Example
///
/// ```
/// use bhc_system::filepath::parent;
///
/// assert_eq!(parent("/home/user/file.txt"), Some("/home/user".to_string()));
/// assert_eq!(parent("/"), None);
/// ```
pub fn parent(path: &str) -> Option<String> {
    Path::new(path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_string_lossy().to_string())
}

/// Set or replace the file extension
///
/// # Example
///
/// ```
/// use bhc_system::filepath::set_extension;
///
/// assert_eq!(set_extension("file.txt", "md"), "file.md");
/// assert_eq!(set_extension("file", "txt"), "file.txt");
/// ```
pub fn set_extension(path: &str, ext: &str) -> String {
    let mut path_buf = PathBuf::from(path);
    path_buf.set_extension(ext);
    path_buf.to_string_lossy().to_string()
}

/// Split a path into stem and extension
///
/// # Example
///
/// ```
/// use bhc_system::filepath::split_extension;
///
/// assert_eq!(split_extension("file.txt"), ("file", Some("txt")));
/// assert_eq!(split_extension("file"), ("file", None));
/// ```
pub fn split_extension(path: &str) -> (&str, Option<&str>) {
    let p = Path::new(path);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or(path);
    let ext = p.extension().and_then(|s| s.to_str());
    (stem, ext)
}

/// Check if a path is absolute
///
/// # Example
///
/// ```
/// use bhc_system::filepath::is_absolute;
///
/// assert!(is_absolute("/home/user"));
/// assert!(!is_absolute("relative/path"));
/// ```
pub fn is_absolute(path: &str) -> bool {
    Path::new(path).is_absolute()
}

/// Check if a path is relative
///
/// # Example
///
/// ```
/// use bhc_system::filepath::is_relative;
///
/// assert!(is_relative("relative/path"));
/// assert!(!is_relative("/absolute/path"));
/// ```
pub fn is_relative(path: &str) -> bool {
    Path::new(path).is_relative()
}

/// Normalize a path by resolving `.` and `..` components
///
/// Note: This performs lexical normalization only; it does not
/// resolve symlinks or check if the path exists.
///
/// # Example
///
/// ```
/// use bhc_system::filepath::normalize;
///
/// let normalized = normalize("/home/user/../user/./documents");
/// // Result: "/home/user/documents" (on Unix)
/// ```
pub fn normalize(path: &str) -> String {
    let path = Path::new(path);
    let mut components = Vec::new();

    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Prefix(p) => components.push(p.as_os_str().to_string_lossy().to_string()),
            Component::RootDir => components.push(MAIN_SEPARATOR.to_string()),
            Component::CurDir => {} // Skip "."
            Component::ParentDir => {
                // Pop the last component if possible (and not root)
                if !components.is_empty() && components.last() != Some(&MAIN_SEPARATOR.to_string())
                {
                    components.pop();
                }
            }
            Component::Normal(s) => components.push(s.to_string_lossy().to_string()),
        }
    }

    if components.is_empty() {
        ".".to_string()
    } else if components.len() == 1 && components[0] == MAIN_SEPARATOR.to_string() {
        MAIN_SEPARATOR.to_string()
    } else {
        // Join components, handling root specially
        let mut result = String::new();
        for (i, comp) in components.iter().enumerate() {
            if comp == &MAIN_SEPARATOR.to_string() {
                result.push(MAIN_SEPARATOR);
            } else {
                if i > 0 && !result.ends_with(MAIN_SEPARATOR) {
                    result.push(MAIN_SEPARATOR);
                }
                result.push_str(comp);
            }
        }
        result
    }
}

/// Get all components of a path
///
/// # Example
///
/// ```
/// use bhc_system::filepath::components;
///
/// let parts = components("/home/user/file.txt");
/// // Returns something like ["/", "home", "user", "file.txt"]
/// ```
pub fn components(path: &str) -> Vec<String> {
    Path::new(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect()
}

/// Check if a path has a specific extension
///
/// # Example
///
/// ```
/// use bhc_system::filepath::has_extension;
///
/// assert!(has_extension("file.txt", "txt"));
/// assert!(!has_extension("file.txt", "md"));
/// ```
pub fn has_extension(path: &str, ext: &str) -> bool {
    extension(path) == Some(ext)
}

/// Add an extension to a path
///
/// Unlike `set_extension`, this always adds a new extension.
///
/// # Example
///
/// ```
/// use bhc_system::filepath::add_extension;
///
/// assert_eq!(add_extension("file.tar", "gz"), "file.tar.gz");
/// ```
pub fn add_extension(path: &str, ext: &str) -> String {
    format!("{}.{}", path, ext)
}

/// Strip an extension from a path
///
/// # Example
///
/// ```
/// use bhc_system::filepath::strip_extension;
///
/// assert_eq!(strip_extension("file.txt"), "file");
/// assert_eq!(strip_extension("archive.tar.gz"), "archive.tar");
/// ```
pub fn strip_extension(path: &str) -> String {
    let path_buf = PathBuf::from(path);
    match (path_buf.parent(), path_buf.file_stem()) {
        (Some(parent), Some(stem)) if !parent.as_os_str().is_empty() => {
            parent.join(stem).to_string_lossy().to_string()
        }
        (_, Some(stem)) => stem.to_string_lossy().to_string(),
        _ => path.to_string(),
    }
}

/// Check if one path starts with another
///
/// # Example
///
/// ```
/// use bhc_system::filepath::starts_with;
///
/// assert!(starts_with("/home/user/file.txt", "/home/user"));
/// assert!(!starts_with("/home/user/file.txt", "/other"));
/// ```
pub fn starts_with(path: &str, prefix: &str) -> bool {
    Path::new(path).starts_with(prefix)
}

/// Check if one path ends with another
///
/// # Example
///
/// ```
/// use bhc_system::filepath::ends_with;
///
/// assert!(ends_with("/home/user/file.txt", "file.txt"));
/// assert!(ends_with("/home/user/file.txt", "user/file.txt"));
/// ```
pub fn ends_with(path: &str, suffix: &str) -> bool {
    Path::new(path).ends_with(suffix)
}

/// Get the path separator for the current platform
pub fn separator() -> char {
    MAIN_SEPARATOR
}

/// Convert path separators to the current platform's separator
pub fn to_native_separators(path: &str) -> String {
    if MAIN_SEPARATOR == '/' {
        path.replace('\\', "/")
    } else {
        path.replace('/', "\\")
    }
}

/// Make a path relative to a base path
///
/// # Example
///
/// ```
/// use bhc_system::filepath::relative_to;
///
/// let rel = relative_to("/home/user/docs/file.txt", "/home/user");
/// // Returns Some("docs/file.txt") or similar
/// ```
pub fn relative_to(path: &str, base: &str) -> Option<String> {
    let path = Path::new(path);
    let base = Path::new(base);

    path.strip_prefix(base)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

// FFI exports

/// Join paths (FFI)
#[no_mangle]
pub extern "C" fn bhc_filepath_join(
    base: *const i8,
    path: *const i8,
    out_len: *mut usize,
) -> *mut u8 {
    use std::ffi::CStr;

    if base.is_null() || path.is_null() {
        return std::ptr::null_mut();
    }

    let base = unsafe { CStr::from_ptr(base) };
    let path = unsafe { CStr::from_ptr(path) };

    let base = match base.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let result = join2(base, path);
    let bytes = result.into_bytes();
    let len = bytes.len();
    let ptr = bytes.leak().as_mut_ptr();

    if !out_len.is_null() {
        unsafe { *out_len = len };
    }
    ptr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join() {
        let path = join(&["home", "user", "file.txt"]);
        assert!(path.contains("home"));
        assert!(path.contains("user"));
        assert!(path.contains("file.txt"));
    }

    #[test]
    fn test_join2() {
        let path = join2("/home/user", "documents/file.txt");
        assert!(path.ends_with("file.txt"));
    }

    #[test]
    fn test_file_name() {
        assert_eq!(file_name("/home/user/file.txt"), Some("file.txt"));
        assert_eq!(file_name("file.txt"), Some("file.txt"));
        assert_eq!(file_name("/"), None);
    }

    #[test]
    fn test_stem() {
        assert_eq!(stem("file.txt"), Some("file"));
        assert_eq!(stem("archive.tar.gz"), Some("archive.tar"));
        assert_eq!(stem("no_ext"), Some("no_ext"));
    }

    #[test]
    fn test_extension() {
        assert_eq!(extension("file.txt"), Some("txt"));
        assert_eq!(extension("archive.tar.gz"), Some("gz"));
        assert_eq!(extension("no_extension"), None);
        assert_eq!(extension(".hidden"), None);
    }

    #[test]
    fn test_parent() {
        assert_eq!(
            parent("/home/user/file.txt"),
            Some("/home/user".to_string())
        );
        assert_eq!(parent("file.txt"), None);
    }

    #[test]
    fn test_set_extension() {
        assert_eq!(set_extension("file.txt", "md"), "file.md");
        assert_eq!(set_extension("file", "txt"), "file.txt");
        assert_eq!(set_extension("file.tar.gz", "bz2"), "file.tar.bz2");
    }

    #[test]
    fn test_is_absolute() {
        assert!(is_absolute("/home/user"));
        assert!(!is_absolute("relative/path"));
        assert!(!is_absolute("./local"));
    }

    #[test]
    fn test_is_relative() {
        assert!(is_relative("relative/path"));
        assert!(is_relative("./local"));
        assert!(!is_relative("/absolute/path"));
    }

    #[test]
    fn test_normalize() {
        // Basic normalization
        let n = normalize("/home/user/../user/./docs");
        assert!(n.contains("user"));
        assert!(n.contains("docs"));
        assert!(!n.contains(".."));
        assert!(!n.contains("/./")); // No redundant current dir
    }

    #[test]
    fn test_has_extension() {
        assert!(has_extension("file.txt", "txt"));
        assert!(!has_extension("file.txt", "md"));
        assert!(!has_extension("file", "txt"));
    }

    #[test]
    fn test_add_extension() {
        assert_eq!(add_extension("file.tar", "gz"), "file.tar.gz");
        assert_eq!(add_extension("file", "txt"), "file.txt");
    }

    #[test]
    fn test_strip_extension() {
        assert_eq!(strip_extension("file.txt"), "file");
        assert_eq!(strip_extension("archive.tar.gz"), "archive.tar");
    }

    #[test]
    fn test_starts_with() {
        assert!(starts_with("/home/user/file.txt", "/home"));
        assert!(starts_with("/home/user/file.txt", "/home/user"));
        assert!(!starts_with("/home/user/file.txt", "/other"));
    }

    #[test]
    fn test_ends_with() {
        assert!(ends_with("/home/user/file.txt", "file.txt"));
        assert!(ends_with("/home/user/file.txt", "user/file.txt"));
        assert!(!ends_with("/home/user/file.txt", "other.txt"));
    }

    #[test]
    fn test_components() {
        let parts = components("/home/user/file.txt");
        assert!(parts.len() >= 3);
    }

    #[test]
    fn test_relative_to() {
        let rel = relative_to("/home/user/docs/file.txt", "/home/user");
        assert!(rel.is_some());
        let rel = rel.unwrap();
        assert!(rel.contains("docs"));
        assert!(rel.contains("file.txt"));
    }

    #[test]
    fn test_to_native_separators() {
        let path = to_native_separators("home/user/file.txt");
        assert!(path.contains(separator()));
    }
}
