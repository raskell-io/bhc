//! Numeric operations
//!
//! Core numeric operations for BHC. These provide the foundation for
//! the Num, Fractional, and other numeric type classes.

use crate::ordering::Ordering;

/// Integer division
#[no_mangle]
pub extern "C" fn bhc_div_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("division by zero");
    }
    // Haskell-style div: rounds towards negative infinity
    let q = a / b;
    let r = a % b;
    if (r != 0) && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Integer modulo
#[no_mangle]
pub extern "C" fn bhc_mod_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("modulo by zero");
    }
    // Haskell-style mod: result has same sign as divisor
    let r = a % b;
    if (r != 0) && ((r < 0) != (b < 0)) {
        r + b
    } else {
        r
    }
}

/// Integer quotient (rounds towards zero)
#[no_mangle]
pub extern "C" fn bhc_quot_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("quotient by zero");
    }
    a / b
}

/// Integer remainder (same sign as dividend)
#[no_mangle]
pub extern "C" fn bhc_rem_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("remainder by zero");
    }
    a % b
}

/// Absolute value
#[no_mangle]
pub extern "C" fn bhc_abs_i64(x: i64) -> i64 {
    x.abs()
}

/// Absolute value for f64
#[no_mangle]
pub extern "C" fn bhc_abs_f64(x: f64) -> f64 {
    x.abs()
}

/// Sign of a number
///
/// Returns -1, 0, or 1
#[no_mangle]
pub extern "C" fn bhc_signum_i64(x: i64) -> i64 {
    x.signum()
}

/// Sign of a number for f64
///
/// Returns -1.0, 0.0, or 1.0 (or NaN for NaN input)
#[no_mangle]
pub extern "C" fn bhc_signum_f64(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Negation
#[no_mangle]
pub extern "C" fn bhc_negate_i64(x: i64) -> i64 {
    -x
}

/// Negation for f64
#[no_mangle]
pub extern "C" fn bhc_negate_f64(x: f64) -> f64 {
    -x
}

/// Addition
#[no_mangle]
pub extern "C" fn bhc_add_i64(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

/// Addition for f64
#[no_mangle]
pub extern "C" fn bhc_add_f64(a: f64, b: f64) -> f64 {
    a + b
}

/// Subtraction
#[no_mangle]
pub extern "C" fn bhc_sub_i64(a: i64, b: i64) -> i64 {
    a.wrapping_sub(b)
}

/// Subtraction for f64
#[no_mangle]
pub extern "C" fn bhc_sub_f64(a: f64, b: f64) -> f64 {
    a - b
}

/// Multiplication
#[no_mangle]
pub extern "C" fn bhc_mul_i64(a: i64, b: i64) -> i64 {
    a.wrapping_mul(b)
}

/// Multiplication for f64
#[no_mangle]
pub extern "C" fn bhc_mul_f64(a: f64, b: f64) -> f64 {
    a * b
}

/// Division for f64
#[no_mangle]
pub extern "C" fn bhc_div_f64(a: f64, b: f64) -> f64 {
    a / b
}

/// Reciprocal for f64
#[no_mangle]
pub extern "C" fn bhc_recip_f64(x: f64) -> f64 {
    1.0 / x
}

/// Integer power
#[no_mangle]
pub extern "C" fn bhc_pow_i64(base: i64, exp: u64) -> i64 {
    let mut result: i64 = 1;
    let mut base = base;
    let mut exp = exp;

    while exp > 0 {
        if exp & 1 == 1 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exp >>= 1;
    }

    result
}

/// Floating point power
#[no_mangle]
pub extern "C" fn bhc_pow_f64(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Square root
#[no_mangle]
pub extern "C" fn bhc_sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}

/// Natural logarithm
#[no_mangle]
pub extern "C" fn bhc_log_f64(x: f64) -> f64 {
    x.ln()
}

/// Exponential
#[no_mangle]
pub extern "C" fn bhc_exp_f64(x: f64) -> f64 {
    x.exp()
}

/// Sine
#[no_mangle]
pub extern "C" fn bhc_sin_f64(x: f64) -> f64 {
    x.sin()
}

/// Cosine
#[no_mangle]
pub extern "C" fn bhc_cos_f64(x: f64) -> f64 {
    x.cos()
}

/// Tangent
#[no_mangle]
pub extern "C" fn bhc_tan_f64(x: f64) -> f64 {
    x.tan()
}

/// Arc sine
#[no_mangle]
pub extern "C" fn bhc_asin_f64(x: f64) -> f64 {
    x.asin()
}

/// Arc cosine
#[no_mangle]
pub extern "C" fn bhc_acos_f64(x: f64) -> f64 {
    x.acos()
}

/// Arc tangent
#[no_mangle]
pub extern "C" fn bhc_atan_f64(x: f64) -> f64 {
    x.atan()
}

/// Two-argument arc tangent
#[no_mangle]
pub extern "C" fn bhc_atan2_f64(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Hyperbolic sine
#[no_mangle]
pub extern "C" fn bhc_sinh_f64(x: f64) -> f64 {
    x.sinh()
}

/// Hyperbolic cosine
#[no_mangle]
pub extern "C" fn bhc_cosh_f64(x: f64) -> f64 {
    x.cosh()
}

/// Hyperbolic tangent
#[no_mangle]
pub extern "C" fn bhc_tanh_f64(x: f64) -> f64 {
    x.tanh()
}

/// Floor
#[no_mangle]
pub extern "C" fn bhc_floor_f64(x: f64) -> f64 {
    x.floor()
}

/// Ceiling
#[no_mangle]
pub extern "C" fn bhc_ceiling_f64(x: f64) -> f64 {
    x.ceil()
}

/// Round to nearest integer
#[no_mangle]
pub extern "C" fn bhc_round_f64(x: f64) -> f64 {
    x.round()
}

/// Truncate towards zero
#[no_mangle]
pub extern "C" fn bhc_truncate_f64(x: f64) -> f64 {
    x.trunc()
}

/// GCD using Euclidean algorithm
#[no_mangle]
pub extern "C" fn bhc_gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();

    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }

    a
}

/// LCM
#[no_mangle]
pub extern "C" fn bhc_lcm_i64(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / bhc_gcd_i64(a, b)).abs() * b.abs()
    }
}

/// Check if even
#[no_mangle]
pub extern "C" fn bhc_even_i64(x: i64) -> crate::bool::Bool {
    crate::bool::Bool::from_bool(x % 2 == 0)
}

/// Check if odd
#[no_mangle]
pub extern "C" fn bhc_odd_i64(x: i64) -> crate::bool::Bool {
    crate::bool::Bool::from_bool(x % 2 != 0)
}

/// Minimum of two values
#[no_mangle]
pub extern "C" fn bhc_min_i64(a: i64, b: i64) -> i64 {
    a.min(b)
}

/// Maximum of two values
#[no_mangle]
pub extern "C" fn bhc_max_i64(a: i64, b: i64) -> i64 {
    a.max(b)
}

/// Minimum of two f64 values
#[no_mangle]
pub extern "C" fn bhc_min_f64(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// Maximum of two f64 values
#[no_mangle]
pub extern "C" fn bhc_max_f64(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Convert i64 to f64
#[no_mangle]
pub extern "C" fn bhc_from_integral_i64_f64(x: i64) -> f64 {
    x as f64
}

/// Convert f64 to i64 (truncates)
#[no_mangle]
pub extern "C" fn bhc_truncate_f64_i64(x: f64) -> i64 {
    x as i64
}

/// Convert f64 to i64 (rounds)
#[no_mangle]
pub extern "C" fn bhc_round_f64_i64(x: f64) -> i64 {
    x.round() as i64
}

/// Convert f64 to i64 (floor)
#[no_mangle]
pub extern "C" fn bhc_floor_f64_i64(x: f64) -> i64 {
    x.floor() as i64
}

/// Convert f64 to i64 (ceiling)
#[no_mangle]
pub extern "C" fn bhc_ceiling_f64_i64(x: f64) -> i64 {
    x.ceil() as i64
}

/// Check if NaN
#[no_mangle]
pub extern "C" fn bhc_is_nan_f64(x: f64) -> crate::bool::Bool {
    crate::bool::Bool::from_bool(x.is_nan())
}

/// Check if infinite
#[no_mangle]
pub extern "C" fn bhc_is_infinite_f64(x: f64) -> crate::bool::Bool {
    crate::bool::Bool::from_bool(x.is_infinite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_mod() {
        // Haskell-style div and mod
        assert_eq!(bhc_div_i64(7, 3), 2);
        assert_eq!(bhc_mod_i64(7, 3), 1);

        // Negative numbers
        assert_eq!(bhc_div_i64(-7, 3), -3);  // Rounds towards negative infinity
        assert_eq!(bhc_mod_i64(-7, 3), 2);   // Same sign as divisor

        assert_eq!(bhc_div_i64(7, -3), -3);
        assert_eq!(bhc_mod_i64(7, -3), -2);

        assert_eq!(bhc_div_i64(-7, -3), 2);
        assert_eq!(bhc_mod_i64(-7, -3), -1);
    }

    #[test]
    fn test_quot_rem() {
        // C-style quot and rem
        assert_eq!(bhc_quot_i64(7, 3), 2);
        assert_eq!(bhc_rem_i64(7, 3), 1);

        assert_eq!(bhc_quot_i64(-7, 3), -2);  // Rounds towards zero
        assert_eq!(bhc_rem_i64(-7, 3), -1);   // Same sign as dividend
    }

    #[test]
    fn test_gcd_lcm() {
        assert_eq!(bhc_gcd_i64(12, 8), 4);
        assert_eq!(bhc_gcd_i64(-12, 8), 4);
        assert_eq!(bhc_gcd_i64(0, 5), 5);
        assert_eq!(bhc_gcd_i64(5, 0), 5);

        assert_eq!(bhc_lcm_i64(4, 6), 12);
        assert_eq!(bhc_lcm_i64(0, 5), 0);
    }

    #[test]
    fn test_pow() {
        assert_eq!(bhc_pow_i64(2, 10), 1024);
        assert_eq!(bhc_pow_i64(3, 0), 1);
        assert_eq!(bhc_pow_i64(-2, 3), -8);
    }

    #[test]
    fn test_trig() {
        let pi = std::f64::consts::PI;
        assert!((bhc_sin_f64(0.0)).abs() < 1e-10);
        assert!((bhc_cos_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((bhc_sin_f64(pi / 2.0) - 1.0).abs() < 1e-10);
    }
}
