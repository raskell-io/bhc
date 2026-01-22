//! BHC Transformers Library
//!
//! Monad transformers for composing effects in pure functional style.
//!
//! # Overview
//!
//! Monad transformers allow you to combine multiple effects (state, errors,
//! logging, environment) in a single computation. Each transformer adds
//! one effect to an underlying monad.
//!
//! # Available Transformers
//!
//! - [`identity`] - Identity monad and IdentityT (base case for stacking)
//! - [`reader`] - ReaderT for read-only environment access
//! - [`writer`] - WriterT for logging/accumulation
//! - [`state`] - StateT for mutable state
//! - [`except`] - ExceptT for error handling
//! - [`maybe`] - MaybeT for optional/failure semantics
//! - [`rws`] - RWS combined Reader/Writer/State
//!
//! # Example
//!
//! ```ignore
//! use bhc_transformers::{Reader, State, Writer};
//!
//! // Reader for configuration
//! let config_reader = Reader::asks(|cfg: &Config| cfg.timeout);
//!
//! // State for counters
//! let tick = State::get().and_then(|n| State::put(n + 1));
//!
//! // Writer for logging
//! let logged = Writer::tell(vec!["started"]).and_then(|_| Writer::pure(42));
//! ```

#![warn(missing_docs)]
#![warn(unsafe_code)]

pub mod except;
pub mod identity;
pub mod maybe;
pub mod reader;
pub mod rws;
pub mod state;
pub mod writer;

// Re-export main types at crate level
pub use except::{Except, ExceptT};
pub use identity::{Identity, IdentityT};
pub use maybe::MaybeT;
pub use reader::{Reader, ReaderT};
pub use state::{State, StateT};
pub use writer::{Monoid, Product, Sum, Writer, WriterT};
pub use rws::{RWS, RWST};
