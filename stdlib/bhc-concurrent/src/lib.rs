//! BHC Concurrent Library
//!
//! Structured concurrency primitives for BHC.
//!
//! # Features
//!
//! - **Structured Concurrency**: All tasks complete within their scope
//! - **Cancellation**: Cooperative cancellation with propagation
//! - **STM**: Software transactional memory
//!
//! # Modules
//!
//! - `scope` - Scoped task execution
//! - `task` - Task creation and management
//! - `channel` - Communication channels
//! - `stm` - Software transactional memory

#![warn(missing_docs)]
#![warn(unsafe_code)]

pub mod channel;
pub mod scope;
pub mod stm;
pub mod task;
