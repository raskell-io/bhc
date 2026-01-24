//! BHC Concurrent Library - Rust support
//!
//! Runtime primitives for BHC structured concurrency and STM.
//!
//! # Architecture
//!
//! The high-level concurrency API is **defined in Haskell** (see
//! `hs/BHC/Control/Concurrent/*.hs`). This Rust crate provides the
//! low-level runtime primitives that cannot be expressed in Haskell.
//!
//! # What belongs here
//!
//! - STM runtime (TVar, atomically, retry, orElse)
//! - Structured concurrency scopes (Scope, Task)
//! - Low-level channels (MPSC, broadcast)
//! - Thread pool and scheduler primitives
//!
//! # What does NOT belong here
//!
//! - High-level concurrency abstractions (those are Haskell)
//! - Monad instances for concurrent types (those are Haskell)
//! - Lifted IO operations (those are Haskell)
//!
//! # FFI Exports
//!
//! This crate exports C-ABI functions for BHC to call:
//! - STM: `bhc_tvar_new`, `bhc_tvar_read`, `bhc_atomically`
//! - Scope: `bhc_scope_new`, `bhc_spawn`, `bhc_await`
//! - Channel: `bhc_chan_new`, `bhc_chan_send`, `bhc_chan_recv`
//!
//! # Features
//!
//! - **Structured Concurrency**: All tasks complete within their scope
//! - **Cancellation**: Cooperative cancellation with propagation
//! - **STM**: Software transactional memory with retry/orElse

#![warn(missing_docs)]
#![warn(unsafe_code)]

pub mod channel;
pub mod ffi;
pub mod scope;
pub mod stm;
pub mod task;
