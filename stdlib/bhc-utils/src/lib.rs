//! BHC Utility Modules
//!
//! This crate provides common utility modules for the BHC standard library:
//!
//! - [`time`] - Date, time, and duration operations
//! - [`random`] - Random number generation
//! - [`json`] - JSON parsing and serialization
//!
//! # Example
//!
//! ```
//! use bhc_utils::time::{Duration, Instant};
//! use bhc_utils::random::Rng;
//!
//! // Measure elapsed time
//! let start = Instant::now();
//! // ... do work ...
//! let elapsed = start.elapsed();
//!
//! // Generate random numbers
//! let mut rng = Rng::new();
//! let n = rng.next_u32();
//! ```

#![warn(missing_docs)]
#![warn(unsafe_code)]

pub mod json;
pub mod random;
pub mod time;

// Re-export main types
pub use json::{Json, JsonError, JsonResult};
pub use random::Rng;
pub use time::{Date, DateTime, Duration, Instant, Time};
