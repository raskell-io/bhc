//! Random number generation
//!
//! This module provides random number generation capabilities.
//!
//! # Overview
//!
//! The main type is [`Rng`], a pseudo-random number generator using
//! the xorshift128+ algorithm. It provides:
//!
//! - Integer generation in ranges
//! - Floating point generation
//! - Boolean generation
//! - Shuffling and sampling
//!
//! # Example
//!
//! ```
//! use bhc_utils::random::Rng;
//!
//! let mut rng = Rng::new();
//!
//! // Generate random numbers
//! let n: u32 = rng.next_u32();
//! let f: f64 = rng.next_f64();
//! let b: bool = rng.next_bool();
//!
//! // Generate in range
//! let dice = rng.range(1, 7); // 1-6 inclusive
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

/// A pseudo-random number generator
///
/// Uses the xorshift128+ algorithm for fast, high-quality randomness.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 2],
}

impl Rng {
    /// Create a new RNG seeded from system time
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self::from_seed(seed)
    }

    /// Create a new RNG with a specific seed
    ///
    /// # Example
    ///
    /// ```
    /// use bhc_utils::random::Rng;
    ///
    /// let mut rng = Rng::from_seed(12345);
    /// let n = rng.next_u64();
    /// ```
    pub fn from_seed(seed: u64) -> Self {
        // Use splitmix64 to initialize state from seed
        let mut state = [0u64; 2];
        let mut x = seed;

        x = x.wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        state[0] = x ^ (x >> 31);

        x = x.wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        state[1] = x ^ (x >> 31);

        // Ensure state is not all zeros
        if state[0] == 0 && state[1] == 0 {
            state[0] = 1;
        }

        Rng { state }
    }

    /// Generate the next u64 value
    pub fn next_u64(&mut self) -> u64 {
        // xorshift128+
        let mut s1 = self.state[0];
        let s0 = self.state[1];
        let result = s0.wrapping_add(s1);

        self.state[0] = s0;
        s1 ^= s1 << 23;
        self.state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);

        result
    }

    /// Generate a random u32
    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// Generate a random i32
    pub fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }

    /// Generate a random i64
    pub fn next_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    /// Generate a random usize
    pub fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    /// Generate a random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Generate a random f32 in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / (1u32 << 24) as f32;
        (self.next_u32() >> 8) as f32 * SCALE
    }

    /// Generate a random boolean
    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// Generate a random boolean with given probability of being true
    pub fn next_bool_weighted(&mut self, probability: f64) -> bool {
        self.next_f64() < probability
    }

    /// Generate a random integer in [min, max] (inclusive)
    ///
    /// # Example
    ///
    /// ```
    /// use bhc_utils::random::Rng;
    ///
    /// let mut rng = Rng::new();
    /// let dice = rng.range(1, 6); // 1, 2, 3, 4, 5, or 6
    /// assert!(dice >= 1 && dice <= 6);
    /// ```
    pub fn range(&mut self, min: i64, max: i64) -> i64 {
        if min >= max {
            return min;
        }
        let range = (max - min + 1) as u64;
        min + (self.next_u64() % range) as i64
    }

    /// Generate a random usize in [0, max) (exclusive)
    pub fn range_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        self.next_usize() % max
    }

    /// Generate a random f64 in [min, max)
    pub fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    /// Choose a random element from a slice
    ///
    /// # Example
    ///
    /// ```
    /// use bhc_utils::random::Rng;
    ///
    /// let mut rng = Rng::new();
    /// let items = [1, 2, 3, 4, 5];
    /// let choice = rng.choose(&items);
    /// assert!(choice.is_some());
    /// ```
    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        if slice.is_empty() {
            None
        } else {
            Some(&slice[self.range_usize(slice.len())])
        }
    }

    /// Shuffle a slice in place (Fisher-Yates shuffle)
    ///
    /// # Example
    ///
    /// ```
    /// use bhc_utils::random::Rng;
    ///
    /// let mut rng = Rng::new();
    /// let mut items = [1, 2, 3, 4, 5];
    /// rng.shuffle(&mut items);
    /// // items is now in random order
    /// ```
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.range_usize(i + 1);
            slice.swap(i, j);
        }
    }

    /// Sample n elements from a slice without replacement
    pub fn sample<T: Clone>(&mut self, slice: &[T], n: usize) -> Vec<T> {
        let mut indices: Vec<usize> = (0..slice.len()).collect();
        self.shuffle(&mut indices);
        indices
            .into_iter()
            .take(n)
            .map(|i| slice[i].clone())
            .collect()
    }

    /// Generate a random byte array
    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(len);
        let mut remaining = len;

        while remaining >= 8 {
            let val = self.next_u64();
            result.extend_from_slice(&val.to_le_bytes());
            remaining -= 8;
        }

        if remaining > 0 {
            let val = self.next_u64();
            result.extend_from_slice(&val.to_le_bytes()[..remaining]);
        }

        result
    }

    /// Generate a random alphanumeric string
    pub fn alphanumeric(&mut self, len: usize) -> String {
        const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        (0..len)
            .map(|_| {
                let idx = self.range_usize(CHARS.len());
                CHARS[idx] as char
            })
            .collect()
    }

    /// Generate a random UUID v4
    pub fn uuid(&mut self) -> String {
        let mut bytes = [0u8; 16];
        let r1 = self.next_u64();
        let r2 = self.next_u64();
        bytes[..8].copy_from_slice(&r1.to_le_bytes());
        bytes[8..].copy_from_slice(&r2.to_le_bytes());

        // Set version (4) and variant bits
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;

        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5],
            bytes[6], bytes[7],
            bytes[8], bytes[9],
            bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        )
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a random number using a thread-local RNG
///
/// This is convenient for quick random generation without managing state.
pub fn random_u64() -> u64 {
    Rng::new().next_u64()
}

/// Generate a random f64 in [0, 1)
pub fn random_f64() -> f64 {
    Rng::new().next_f64()
}

/// Generate a random bool
pub fn random_bool() -> bool {
    Rng::new().next_bool()
}

/// Weighted random selection
///
/// Given items and their weights, randomly select one item.
///
/// # Example
///
/// ```
/// use bhc_utils::random::weighted_choice;
///
/// let items = vec![("common", 70), ("rare", 25), ("epic", 5)];
/// let mut rng = bhc_utils::random::Rng::new();
/// let choice = weighted_choice(&items, &mut rng);
/// ```
pub fn weighted_choice<'a, T>(items: &'a [(T, u32)], rng: &mut Rng) -> Option<&'a T> {
    if items.is_empty() {
        return None;
    }

    let total: u32 = items.iter().map(|(_, w)| w).sum();
    if total == 0 {
        return None;
    }

    let mut threshold = rng.range(0, total as i64 - 1) as u32;

    for (item, weight) in items {
        if threshold < *weight {
            return Some(item);
        }
        threshold -= weight;
    }

    Some(&items.last()?.0)
}

/// Normal distribution using Box-Muller transform
pub struct Normal {
    mean: f64,
    std_dev: f64,
    spare: Option<f64>,
}

impl Normal {
    /// Create a normal distribution
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Normal {
            mean,
            std_dev,
            spare: None,
        }
    }

    /// Sample from the distribution
    pub fn sample(&mut self, rng: &mut Rng) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare * self.std_dev + self.mean;
        }

        loop {
            let u = rng.next_f64() * 2.0 - 1.0;
            let v = rng.next_f64() * 2.0 - 1.0;
            let s = u * u + v * v;

            if s > 0.0 && s < 1.0 {
                let mul = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * mul);
                return u * mul * self.std_dev + self.mean;
            }
        }
    }
}

// FFI exports

/// Create new RNG (FFI)
#[no_mangle]
pub extern "C" fn bhc_rng_new() -> *mut Rng {
    Box::into_raw(Box::new(Rng::new()))
}

/// Create seeded RNG (FFI)
#[no_mangle]
pub extern "C" fn bhc_rng_from_seed(seed: u64) -> *mut Rng {
    Box::into_raw(Box::new(Rng::from_seed(seed)))
}

/// Free RNG (FFI)
#[no_mangle]
pub extern "C" fn bhc_rng_free(rng: *mut Rng) {
    if !rng.is_null() {
        unsafe {
            drop(Box::from_raw(rng));
        }
    }
}

/// Get random u64 (FFI)
#[no_mangle]
pub extern "C" fn bhc_rng_next_u64(rng: *mut Rng) -> u64 {
    if rng.is_null() {
        return 0;
    }
    unsafe { (*rng).next_u64() }
}

/// Get random in range (FFI)
#[no_mangle]
pub extern "C" fn bhc_rng_range(rng: *mut Rng, min: i64, max: i64) -> i64 {
    if rng.is_null() {
        return min;
    }
    unsafe { (*rng).range(min, max) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_new() {
        let mut rng = Rng::new();
        let _ = rng.next_u64();
    }

    #[test]
    fn test_rng_from_seed_deterministic() {
        let mut rng1 = Rng::from_seed(12345);
        let mut rng2 = Rng::from_seed(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_range() {
        let mut rng = Rng::from_seed(42);

        for _ in 0..1000 {
            let n = rng.range(1, 6);
            assert!(n >= 1 && n <= 6);
        }
    }

    #[test]
    fn test_rng_f64() {
        let mut rng = Rng::from_seed(42);

        for _ in 0..1000 {
            let f = rng.next_f64();
            assert!(f >= 0.0 && f < 1.0);
        }
    }

    #[test]
    fn test_rng_bool() {
        let mut rng = Rng::from_seed(42);
        let mut trues = 0;
        let mut falses = 0;

        for _ in 0..1000 {
            if rng.next_bool() {
                trues += 1;
            } else {
                falses += 1;
            }
        }

        // Both should occur with reasonable frequency
        assert!(trues > 400 && falses > 400);
    }

    #[test]
    fn test_rng_choose() {
        let mut rng = Rng::from_seed(42);
        let items = [1, 2, 3, 4, 5];

        for _ in 0..100 {
            let choice = rng.choose(&items);
            assert!(choice.is_some());
            assert!(items.contains(choice.unwrap()));
        }
    }

    #[test]
    fn test_rng_choose_empty() {
        let mut rng = Rng::new();
        let items: [i32; 0] = [];
        assert!(rng.choose(&items).is_none());
    }

    #[test]
    fn test_rng_shuffle() {
        let mut rng = Rng::from_seed(42);
        let mut items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let original = items;

        rng.shuffle(&mut items);

        // Should still contain same elements
        let mut sorted = items;
        sorted.sort();
        let mut orig_sorted = original;
        orig_sorted.sort();
        assert_eq!(sorted, orig_sorted);

        // Should be different order (extremely unlikely to be same with 10 items)
        assert_ne!(items, original);
    }

    #[test]
    fn test_rng_sample() {
        let mut rng = Rng::from_seed(42);
        let items = vec![1, 2, 3, 4, 5];

        let sample = rng.sample(&items, 3);
        assert_eq!(sample.len(), 3);

        // All sampled items should be from original
        for item in &sample {
            assert!(items.contains(item));
        }
    }

    #[test]
    fn test_rng_bytes() {
        let mut rng = Rng::from_seed(42);

        let bytes = rng.bytes(16);
        assert_eq!(bytes.len(), 16);

        let bytes = rng.bytes(5);
        assert_eq!(bytes.len(), 5);
    }

    #[test]
    fn test_rng_alphanumeric() {
        let mut rng = Rng::from_seed(42);
        let s = rng.alphanumeric(20);

        assert_eq!(s.len(), 20);
        assert!(s.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn test_rng_uuid() {
        let mut rng = Rng::from_seed(42);
        let uuid = rng.uuid();

        assert_eq!(uuid.len(), 36);
        assert_eq!(&uuid[8..9], "-");
        assert_eq!(&uuid[13..14], "-");
        assert_eq!(&uuid[14..15], "4"); // Version 4
        assert_eq!(&uuid[18..19], "-");
        assert_eq!(&uuid[23..24], "-");
    }

    #[test]
    fn test_weighted_choice() {
        let mut rng = Rng::from_seed(42);
        let items = vec![("a", 90), ("b", 10)];

        let mut a_count = 0;
        let mut b_count = 0;

        for _ in 0..1000 {
            match weighted_choice(&items, &mut rng) {
                Some(&"a") => a_count += 1,
                Some(&"b") => b_count += 1,
                _ => {}
            }
        }

        // "a" should be chosen roughly 9x more often than "b"
        assert!(a_count > b_count * 5);
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = Rng::from_seed(42);
        let mut normal = Normal::new(0.0, 1.0);

        let mut sum = 0.0;
        let n = 10000;

        for _ in 0..n {
            sum += normal.sample(&mut rng);
        }

        let mean = sum / n as f64;
        // Mean should be close to 0
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_range_f64() {
        let mut rng = Rng::from_seed(42);

        for _ in 0..1000 {
            let f = rng.range_f64(10.0, 20.0);
            assert!(f >= 10.0 && f < 20.0);
        }
    }
}
