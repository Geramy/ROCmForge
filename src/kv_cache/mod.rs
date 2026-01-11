//! Paged KV cache module for efficient memory management
//!
//! This module exports the production-grade paged KV cache implementation
//! with PagedAttention support, LRU eviction, and block sharing.
//!
//! # Which KV Cache Should I Use?
//!
//! - **Use `KvCache` from this module** for production inference
//! - See `crate::model::kv_cache::KVCache` for the legacy simple implementation

pub mod kv_cache;

pub use kv_cache::*;
