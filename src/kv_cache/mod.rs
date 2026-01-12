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
pub mod page_table;
pub mod block_allocator;

// Re-export from kv_cache
pub use kv_cache::{
    KvCache, KvCacheError, KvCacheResult, CacheConfig, CachePage, CacheStats,
    SequenceCache, PagedCacheStats, BlockTable, PhysicalBlockPool
};

// Re-export from page_table
pub use page_table::{PageTable};

// Re-export from block_allocator (use fully qualified names to avoid conflicts)
pub use block_allocator::{BlockAllocator as BlockAllocator};

// Note: BlockId and PhysicalBlock are defined in multiple modules
// Use fully qualified paths: kv_cache::BlockId, block_allocator::BlockId, etc.
pub use kv_cache::BlockId;
pub use block_allocator::BlockId as BlockAllocatorBlockId;
pub use block_allocator::PhysicalBlock as AllocatorPhysicalBlock;
