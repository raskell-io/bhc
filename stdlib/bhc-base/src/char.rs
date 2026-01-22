//! Character operations
//!
//! Unicode character classification and conversion functions.

use bhc_prelude::bool::Bool;

/// Check if character is a letter
#[no_mangle]
pub extern "C" fn bhc_char_is_alpha(c: u32) -> Bool {
    Bool::from_bool(char::from_u32(c).map_or(false, |c| c.is_alphabetic()))
}

/// Check if character is a digit
#[no_mangle]
pub extern "C" fn bhc_char_is_digit(c: u32) -> Bool {
    Bool::from_bool(char::from_u32(c).map_or(false, |c| c.is_ascii_digit()))
}

/// Check if character is alphanumeric
#[no_mangle]
pub extern "C" fn bhc_char_is_alphanumeric(c: u32) -> Bool {
    Bool::from_bool(char::from_u32(c).map_or(false, |c| c.is_alphanumeric()))
}

/// Check if character is whitespace
#[no_mangle]
pub extern "C" fn bhc_char_is_space(c: u32) -> Bool {
    Bool::from_bool(char::from_u32(c).map_or(false, |c| c.is_whitespace()))
}

/// Check if character is uppercase
#[no_mangle]
pub extern "C" fn bhc_char_is_upper(c: u32) -> Bool {
    Bool::from_bool(char::from_u32(c).map_or(false, |c| c.is_uppercase()))
}

/// Check if character is lowercase
#[no_mangle]
pub extern "C" fn bhc_char_is_lower(c: u32) -> Bool {
    Bool::from_bool(char::from_u32(c).map_or(false, |c| c.is_lowercase()))
}

/// Convert to uppercase
#[no_mangle]
pub extern "C" fn bhc_char_to_upper(c: u32) -> u32 {
    char::from_u32(c)
        .map(|c| c.to_uppercase().next().unwrap_or(c))
        .map(|c| c as u32)
        .unwrap_or(c)
}

/// Convert to lowercase
#[no_mangle]
pub extern "C" fn bhc_char_to_lower(c: u32) -> u32 {
    char::from_u32(c)
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .map(|c| c as u32)
        .unwrap_or(c)
}

/// Get numeric value of digit character
#[no_mangle]
pub extern "C" fn bhc_char_digit_to_int(c: u32) -> i32 {
    char::from_u32(c)
        .and_then(|c| c.to_digit(10))
        .map(|d| d as i32)
        .unwrap_or(-1)
}

/// Get character for digit
#[no_mangle]
pub extern "C" fn bhc_char_int_to_digit(n: i32) -> u32 {
    if (0..=9).contains(&n) {
        ('0' as u32) + (n as u32)
    } else {
        0
    }
}

/// Get character's Unicode general category
#[no_mangle]
pub extern "C" fn bhc_char_is_print(c: u32) -> Bool {
    Bool::from_bool(char::from_u32(c).map_or(false, |c| !c.is_control()))
}

/// Check if ASCII
#[no_mangle]
pub extern "C" fn bhc_char_is_ascii(c: u32) -> Bool {
    Bool::from_bool(c <= 127)
}

/// Get character's code point
#[no_mangle]
pub extern "C" fn bhc_char_ord(c: u32) -> u32 {
    c
}

/// Convert code point to character
#[no_mangle]
pub extern "C" fn bhc_char_chr(n: u32) -> u32 {
    if char::from_u32(n).is_some() {
        n
    } else {
        0xFFFD // Replacement character
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_alpha() {
        assert_eq!(bhc_char_is_alpha('a' as u32), Bool::True);
        assert_eq!(bhc_char_is_alpha('Z' as u32), Bool::True);
        assert_eq!(bhc_char_is_alpha('1' as u32), Bool::False);
    }

    #[test]
    fn test_case_conversion() {
        assert_eq!(bhc_char_to_upper('a' as u32), 'A' as u32);
        assert_eq!(bhc_char_to_lower('A' as u32), 'a' as u32);
    }

    #[test]
    fn test_digit_conversion() {
        assert_eq!(bhc_char_digit_to_int('5' as u32), 5);
        assert_eq!(bhc_char_int_to_digit(5), '5' as u32);
    }
}
