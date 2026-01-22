//! JSON parsing and serialization
//!
//! This module provides JSON parsing and serialization capabilities.
//!
//! # Overview
//!
//! The main type is [`Json`], representing a JSON value. It supports:
//!
//! - Parsing JSON strings
//! - Serializing to JSON strings
//! - Querying and manipulating JSON values
//!
//! # Example
//!
//! ```
//! use bhc_utils::json::Json;
//!
//! // Parse JSON
//! let json = Json::parse(r#"{"name": "Alice", "age": 30}"#).unwrap();
//!
//! // Access values
//! assert_eq!(json.get("name").and_then(|v| v.as_str()), Some("Alice"));
//! assert_eq!(json.get("age").and_then(|v| v.as_i64()), Some(30));
//!
//! // Build JSON
//! let json = Json::object([
//!     ("name", Json::string("Bob")),
//!     ("active", Json::bool(true)),
//! ]);
//! ```

use std::collections::HashMap;
use std::fmt;

/// A JSON value
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// Null value
    Null,
    /// Boolean value
    Bool(bool),
    /// Number value (stored as f64)
    Number(f64),
    /// String value
    String(String),
    /// Array of JSON values
    Array(Vec<Json>),
    /// Object (map of string keys to JSON values)
    Object(HashMap<String, Json>),
}

/// Error type for JSON operations
#[derive(Debug, Clone)]
pub struct JsonError {
    /// Error message
    pub message: String,
    /// Position in input where error occurred
    pub position: usize,
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON error at position {}: {}", self.position, self.message)
    }
}

impl std::error::Error for JsonError {}

/// Result type for JSON operations
pub type JsonResult<T> = Result<T, JsonError>;

impl Json {
    // Constructors

    /// Create a null value
    pub fn null() -> Self {
        Json::Null
    }

    /// Create a boolean value
    pub fn bool(b: bool) -> Self {
        Json::Bool(b)
    }

    /// Create a number value
    pub fn number(n: f64) -> Self {
        Json::Number(n)
    }

    /// Create an integer number value
    pub fn int(n: i64) -> Self {
        Json::Number(n as f64)
    }

    /// Create a string value
    pub fn string<S: Into<String>>(s: S) -> Self {
        Json::String(s.into())
    }

    /// Create an array value
    pub fn array<I: IntoIterator<Item = Json>>(items: I) -> Self {
        Json::Array(items.into_iter().collect())
    }

    /// Create an object value
    pub fn object<K, I>(pairs: I) -> Self
    where
        K: Into<String>,
        I: IntoIterator<Item = (K, Json)>,
    {
        Json::Object(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    // Type checks

    /// Check if this is null
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// Check if this is a boolean
    pub fn is_bool(&self) -> bool {
        matches!(self, Json::Bool(_))
    }

    /// Check if this is a number
    pub fn is_number(&self) -> bool {
        matches!(self, Json::Number(_))
    }

    /// Check if this is a string
    pub fn is_string(&self) -> bool {
        matches!(self, Json::String(_))
    }

    /// Check if this is an array
    pub fn is_array(&self) -> bool {
        matches!(self, Json::Array(_))
    }

    /// Check if this is an object
    pub fn is_object(&self) -> bool {
        matches!(self, Json::Object(_))
    }

    // Value extraction

    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as f64
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as i64 (truncated)
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Number(n) => Some(*n as i64),
            _ => None,
        }
    }

    /// Get as u64 (truncated, only if non-negative)
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Number(n) if *n >= 0.0 => Some(*n as u64),
            _ => None,
        }
    }

    /// Get as string reference
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as array reference
    pub fn as_array(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Get as mutable array reference
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Json>> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Get as object reference
    pub fn as_object(&self) -> Option<&HashMap<String, Json>> {
        match self {
            Json::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Get as mutable object reference
    pub fn as_object_mut(&mut self) -> Option<&mut HashMap<String, Json>> {
        match self {
            Json::Object(o) => Some(o),
            _ => None,
        }
    }

    // Object/Array access

    /// Get a value by key (for objects) or index (for arrays)
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(o) => o.get(key),
            Json::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
            _ => None,
        }
    }

    /// Get a value by index (for arrays)
    pub fn get_index(&self, index: usize) -> Option<&Json> {
        match self {
            Json::Array(a) => a.get(index),
            _ => None,
        }
    }

    /// Get array length or object key count
    pub fn len(&self) -> Option<usize> {
        match self {
            Json::Array(a) => Some(a.len()),
            Json::Object(o) => Some(o.len()),
            _ => None,
        }
    }

    /// Check if array or object is empty
    pub fn is_empty(&self) -> Option<bool> {
        self.len().map(|n| n == 0)
    }

    /// Get object keys
    pub fn keys(&self) -> Option<Vec<&str>> {
        match self {
            Json::Object(o) => Some(o.keys().map(|s| s.as_str()).collect()),
            _ => None,
        }
    }

    // Parsing

    /// Parse a JSON string
    ///
    /// # Example
    ///
    /// ```
    /// use bhc_utils::json::Json;
    ///
    /// let json = Json::parse(r#"[1, 2, 3]"#).unwrap();
    /// assert_eq!(json.as_array().map(|a| a.len()), Some(3));
    /// ```
    pub fn parse(input: &str) -> JsonResult<Self> {
        let mut parser = Parser::new(input);
        parser.parse_value()
    }

    // Serialization

    /// Serialize to a JSON string
    ///
    /// # Example
    ///
    /// ```
    /// use bhc_utils::json::Json;
    ///
    /// let json = Json::array([Json::int(1), Json::int(2), Json::int(3)]);
    /// assert_eq!(json.to_string(), "[1,2,3]");
    /// ```
    pub fn to_json_string(&self) -> String {
        match self {
            Json::Null => "null".to_string(),
            Json::Bool(true) => "true".to_string(),
            Json::Bool(false) => "false".to_string(),
            Json::Number(n) => {
                if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                    (*n as i64).to_string()
                } else {
                    n.to_string()
                }
            }
            Json::String(s) => format!("\"{}\"", escape_string(s)),
            Json::Array(a) => {
                let items: Vec<String> = a.iter().map(|v| v.to_json_string()).collect();
                format!("[{}]", items.join(","))
            }
            Json::Object(o) => {
                let pairs: Vec<String> = o
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", escape_string(k), v.to_json_string()))
                    .collect();
                format!("{{{}}}", pairs.join(","))
            }
        }
    }

    /// Serialize to a pretty-printed JSON string
    pub fn to_json_string_pretty(&self) -> String {
        self.pretty_print(0)
    }

    fn pretty_print(&self, indent: usize) -> String {
        let indent_str = "  ".repeat(indent);
        let next_indent = "  ".repeat(indent + 1);

        match self {
            Json::Null => "null".to_string(),
            Json::Bool(true) => "true".to_string(),
            Json::Bool(false) => "false".to_string(),
            Json::Number(n) => {
                if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                    (*n as i64).to_string()
                } else {
                    n.to_string()
                }
            }
            Json::String(s) => format!("\"{}\"", escape_string(s)),
            Json::Array(a) if a.is_empty() => "[]".to_string(),
            Json::Array(a) => {
                let items: Vec<String> = a
                    .iter()
                    .map(|v| format!("{}{}", next_indent, v.pretty_print(indent + 1)))
                    .collect();
                format!("[\n{}\n{}]", items.join(",\n"), indent_str)
            }
            Json::Object(o) if o.is_empty() => "{}".to_string(),
            Json::Object(o) => {
                let pairs: Vec<String> = o
                    .iter()
                    .map(|(k, v)| {
                        format!(
                            "{}\"{}\": {}",
                            next_indent,
                            escape_string(k),
                            v.pretty_print(indent + 1)
                        )
                    })
                    .collect();
                format!("{{\n{}\n{}}}", pairs.join(",\n"), indent_str)
            }
        }
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_json_string())
    }
}

// Helper functions

fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result
}

// Parser

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser { input, pos: 0 }
    }

    fn error(&self, message: &str) -> JsonError {
        JsonError {
            message: message.to_string(),
            position: self.pos,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek() {
            self.pos += c.len_utf8();
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> JsonResult<Json> {
        self.skip_whitespace();

        match self.peek() {
            None => Err(self.error("Unexpected end of input")),
            Some('n') => self.parse_null(),
            Some('t') | Some('f') => self.parse_bool(),
            Some('"') => self.parse_string(),
            Some('[') => self.parse_array(),
            Some('{') => self.parse_object(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(self.error(&format!("Unexpected character: {}", c))),
        }
    }

    fn parse_null(&mut self) -> JsonResult<Json> {
        if self.input[self.pos..].starts_with("null") {
            self.pos += 4;
            Ok(Json::Null)
        } else {
            Err(self.error("Expected 'null'"))
        }
    }

    fn parse_bool(&mut self) -> JsonResult<Json> {
        if self.input[self.pos..].starts_with("true") {
            self.pos += 4;
            Ok(Json::Bool(true))
        } else if self.input[self.pos..].starts_with("false") {
            self.pos += 5;
            Ok(Json::Bool(false))
        } else {
            Err(self.error("Expected 'true' or 'false'"))
        }
    }

    fn parse_number(&mut self) -> JsonResult<Json> {
        let start = self.pos;

        // Optional negative sign
        if self.peek() == Some('-') {
            self.advance();
        }

        // Integer part
        match self.peek() {
            Some('0') => self.advance(),
            Some(c) if c.is_ascii_digit() && c != '0' => {
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            _ => return Err(self.error("Invalid number")),
        }

        // Fractional part
        if self.peek() == Some('.') {
            self.advance();
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.error("Expected digit after decimal point"));
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Exponent part
        if matches!(self.peek(), Some('e') | Some('E')) {
            self.advance();
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.advance();
            }
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.error("Expected digit in exponent"));
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        let number_str = &self.input[start..self.pos];
        match number_str.parse::<f64>() {
            Ok(n) => Ok(Json::Number(n)),
            Err(_) => Err(self.error("Invalid number")),
        }
    }

    fn parse_string(&mut self) -> JsonResult<Json> {
        if self.peek() != Some('"') {
            return Err(self.error("Expected '\"'"));
        }
        self.advance();

        let mut result = String::new();

        loop {
            match self.peek() {
                None => return Err(self.error("Unterminated string")),
                Some('"') => {
                    self.advance();
                    return Ok(Json::String(result));
                }
                Some('\\') => {
                    self.advance();
                    match self.peek() {
                        Some('"') => {
                            result.push('"');
                            self.advance();
                        }
                        Some('\\') => {
                            result.push('\\');
                            self.advance();
                        }
                        Some('/') => {
                            result.push('/');
                            self.advance();
                        }
                        Some('b') => {
                            result.push('\x08');
                            self.advance();
                        }
                        Some('f') => {
                            result.push('\x0c');
                            self.advance();
                        }
                        Some('n') => {
                            result.push('\n');
                            self.advance();
                        }
                        Some('r') => {
                            result.push('\r');
                            self.advance();
                        }
                        Some('t') => {
                            result.push('\t');
                            self.advance();
                        }
                        Some('u') => {
                            self.advance();
                            let mut code = 0u32;
                            for _ in 0..4 {
                                match self.peek() {
                                    Some(c) if c.is_ascii_hexdigit() => {
                                        code = code * 16 + c.to_digit(16).unwrap();
                                        self.advance();
                                    }
                                    _ => return Err(self.error("Invalid unicode escape")),
                                }
                            }
                            if let Some(c) = char::from_u32(code) {
                                result.push(c);
                            } else {
                                return Err(self.error("Invalid unicode code point"));
                            }
                        }
                        _ => return Err(self.error("Invalid escape sequence")),
                    }
                }
                Some(c) if c.is_control() => {
                    return Err(self.error("Control character in string"));
                }
                Some(c) => {
                    result.push(c);
                    self.advance();
                }
            }
        }
    }

    fn parse_array(&mut self) -> JsonResult<Json> {
        if self.peek() != Some('[') {
            return Err(self.error("Expected '['"));
        }
        self.advance();
        self.skip_whitespace();

        let mut items = Vec::new();

        if self.peek() == Some(']') {
            self.advance();
            return Ok(Json::Array(items));
        }

        loop {
            items.push(self.parse_value()?);
            self.skip_whitespace();

            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                }
                Some(']') => {
                    self.advance();
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.error("Expected ',' or ']'")),
            }
        }
    }

    fn parse_object(&mut self) -> JsonResult<Json> {
        if self.peek() != Some('{') {
            return Err(self.error("Expected '{'"));
        }
        self.advance();
        self.skip_whitespace();

        let mut pairs = HashMap::new();

        if self.peek() == Some('}') {
            self.advance();
            return Ok(Json::Object(pairs));
        }

        loop {
            // Parse key
            let key = match self.parse_string()? {
                Json::String(s) => s,
                _ => return Err(self.error("Expected string key")),
            };

            self.skip_whitespace();

            // Expect colon
            if self.peek() != Some(':') {
                return Err(self.error("Expected ':'"));
            }
            self.advance();

            // Parse value
            let value = self.parse_value()?;
            pairs.insert(key, value);

            self.skip_whitespace();

            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                }
                Some('}') => {
                    self.advance();
                    return Ok(Json::Object(pairs));
                }
                _ => return Err(self.error("Expected ',' or '}'")),
            }
        }
    }
}

// FFI exports

/// Parse JSON string (FFI)
#[no_mangle]
pub extern "C" fn bhc_json_parse(input: *const i8, out_error: *mut i32) -> *mut Json {
    use std::ffi::CStr;

    if input.is_null() {
        if !out_error.is_null() {
            unsafe { *out_error = 1 };
        }
        return std::ptr::null_mut();
    }

    let input = unsafe { CStr::from_ptr(input) };
    let input = match input.to_str() {
        Ok(s) => s,
        Err(_) => {
            if !out_error.is_null() {
                unsafe { *out_error = 1 };
            }
            return std::ptr::null_mut();
        }
    };

    match Json::parse(input) {
        Ok(json) => {
            if !out_error.is_null() {
                unsafe { *out_error = 0 };
            }
            Box::into_raw(Box::new(json))
        }
        Err(_) => {
            if !out_error.is_null() {
                unsafe { *out_error = 1 };
            }
            std::ptr::null_mut()
        }
    }
}

/// Free JSON value (FFI)
#[no_mangle]
pub extern "C" fn bhc_json_free(json: *mut Json) {
    if !json.is_null() {
        unsafe {
            drop(Box::from_raw(json));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_null() {
        let json = Json::parse("null").unwrap();
        assert!(json.is_null());
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(Json::parse("true").unwrap(), Json::Bool(true));
        assert_eq!(Json::parse("false").unwrap(), Json::Bool(false));
    }

    #[test]
    fn test_parse_number() {
        assert_eq!(Json::parse("42").unwrap().as_i64(), Some(42));
        assert_eq!(Json::parse("-42").unwrap().as_i64(), Some(-42));
        assert_eq!(Json::parse("3.14").unwrap().as_f64(), Some(3.14));
        assert_eq!(Json::parse("1e10").unwrap().as_f64(), Some(1e10));
        assert_eq!(Json::parse("1.5e-2").unwrap().as_f64(), Some(0.015));
    }

    #[test]
    fn test_parse_string() {
        assert_eq!(
            Json::parse(r#""hello""#).unwrap().as_str(),
            Some("hello")
        );
        assert_eq!(
            Json::parse(r#""hello\nworld""#).unwrap().as_str(),
            Some("hello\nworld")
        );
        assert_eq!(
            Json::parse(r#""hello\u0041""#).unwrap().as_str(),
            Some("helloA")
        );
    }

    #[test]
    fn test_parse_array() {
        let json = Json::parse("[1, 2, 3]").unwrap();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_i64(), Some(1));
    }

    #[test]
    fn test_parse_object() {
        let json = Json::parse(r#"{"name": "Alice", "age": 30}"#).unwrap();
        assert_eq!(json.get("name").and_then(|v| v.as_str()), Some("Alice"));
        assert_eq!(json.get("age").and_then(|v| v.as_i64()), Some(30));
    }

    #[test]
    fn test_parse_nested() {
        let json = Json::parse(r#"{"users": [{"name": "Alice"}, {"name": "Bob"}]}"#).unwrap();
        let users = json.get("users").unwrap().as_array().unwrap();
        assert_eq!(users.len(), 2);
        assert_eq!(users[0].get("name").and_then(|v| v.as_str()), Some("Alice"));
    }

    #[test]
    fn test_serialize() {
        let json = Json::object([
            ("name", Json::string("Alice")),
            ("age", Json::int(30)),
            ("active", Json::bool(true)),
        ]);
        let s = json.to_json_string();
        assert!(s.contains("\"name\":\"Alice\""));
        assert!(s.contains("\"age\":30"));
    }

    #[test]
    fn test_serialize_array() {
        let json = Json::array([Json::int(1), Json::int(2), Json::int(3)]);
        assert_eq!(json.to_json_string(), "[1,2,3]");
    }

    #[test]
    fn test_serialize_escape() {
        let json = Json::string("hello\nworld");
        assert_eq!(json.to_json_string(), "\"hello\\nworld\"");
    }

    #[test]
    fn test_pretty_print() {
        let json = Json::object([("name", Json::string("Alice"))]);
        let pretty = json.to_json_string_pretty();
        assert!(pretty.contains('\n'));
    }

    #[test]
    fn test_constructors() {
        assert!(Json::null().is_null());
        assert!(Json::bool(true).is_bool());
        assert!(Json::number(3.14).is_number());
        assert!(Json::string("test").is_string());
        assert!(Json::array([]).is_array());
        assert!(Json::object::<&str, _>([]).is_object());
    }

    #[test]
    fn test_get_index() {
        let json = Json::array([Json::int(1), Json::int(2), Json::int(3)]);
        assert_eq!(json.get_index(1).and_then(|v| v.as_i64()), Some(2));
        assert!(json.get_index(10).is_none());
    }

    #[test]
    fn test_len() {
        let arr = Json::array([Json::int(1), Json::int(2)]);
        assert_eq!(arr.len(), Some(2));

        let obj = Json::object([("a", Json::int(1))]);
        assert_eq!(obj.len(), Some(1));

        let num = Json::int(42);
        assert_eq!(num.len(), None);
    }

    #[test]
    fn test_parse_whitespace() {
        let json = Json::parse("  { \"a\" : 1 }  ").unwrap();
        assert_eq!(json.get("a").and_then(|v| v.as_i64()), Some(1));
    }

    #[test]
    fn test_parse_error() {
        assert!(Json::parse("{invalid}").is_err());
        assert!(Json::parse("[1, 2,]").is_err());
        assert!(Json::parse("").is_err());
    }

    #[test]
    fn test_roundtrip() {
        let original = r#"{"name":"Alice","scores":[100,95,87],"active":true}"#;
        let json = Json::parse(original).unwrap();
        let serialized = json.to_json_string();
        let reparsed = Json::parse(&serialized).unwrap();
        assert_eq!(json, reparsed);
    }

    #[test]
    fn test_display() {
        let json = Json::int(42);
        assert_eq!(format!("{}", json), "42");
    }
}
