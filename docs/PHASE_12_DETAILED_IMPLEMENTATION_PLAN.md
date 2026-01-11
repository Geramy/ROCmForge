# Phase 12: PagedAttention & Continuous Batching - Detailed Implementation Plan

**Date**: 2026-01-08
**Status**: READY FOR IMPLEMENTATION
**Follows**: CLAUDE.md Rules (MANDATORY)
**Tools**: CodeMCP (find_symbols, refactor_rename, refactor_delete)
**Test Health Target**: 116/116 passing → 150/150 passing

---

## Table of Contents

1. [Pre-Implementation Checklist](#1-pre-implementation-checklist)
2. [Architecture Overview](#2-architecture-overview)
3. [Task 12.1: PagedAttention Implementation](#3-task-121-pagedattention-implementation)
4. [Task 12.2: Continuous Batching Enhancement](#4-task-122-continuous-batching-enhancement)
5. [Task 12.3: Attention Backend Registry](#5-task-123-attention-backend-registry)
6. [Task 12.4: Integration & Testing](#6-task-124-integration--testing)
7. [CodeMCP Tool Usage Guide](#7-codemcp-tool-usage-guide)
8. [Success Criteria](#8-success-criteria)

---

## 1. Pre-Implementation Checklist

**BEFORE STARTING ANY CODE**:

```
[ ] 1. Read CLAUDE.md (COMPLETED)
[ ] 2. Read this entire plan document
[ ] 3. Run: cargo test --lib (baseline: 116/116 passing)
[ ] 4. Run: cargo check (no errors)
[ ] 5. Initialize magellan: magellan_init(workspace_root="/home/feanor/Projects/ROCmForge")
[ ] 6. Start watcher: magellan_watch(action="start", workspace_root="/home/feanor/Projects/ROCmForge")
[ ] 7. Store architectural decision in operations.db
```

---

## 2. Architecture Overview

### Current State

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| KvCache | `src/kv_cache/kv_cache.rs` | 448 | Basic paged, no shared blocks |
| Scheduler | `src/scheduler/scheduler.rs` | 641 | Static batching, no continuous |
| Attention | `src/attention/mod.rs` | 218 | Simple CPU/GPU enum |
| Attention Backend | `src/attention/backend.rs` | 15 | No registry system |

### Target State

| Component | Target | Gain |
|-----------|--------|------|
| KvCache | PagedAttention with block sharing | 3x memory efficiency |
| Scheduler | True continuous batching | 10-23x throughput |
| Attention | Pluggable backend registry | Flexibility |

---

## 3. Task 12.1: PagedAttention Implementation

### Overview

**Goal**: Implement PagedAttention algorithm from vLLM paper (Kwon et al., 2023)
**Effort**: 2 weeks
**Impact**: 3x memory efficiency, zero fragmentation

### 3.1 Files to Modify

#### File: `src/kv_cache/kv_cache.rs` (MODIFY)

**Current LOC**: 448
**Target LOC**: ~600
**Changes**:

1. **Add Block Table Structure** (after line 53, after `CacheConfig`)

```rust
/// Block table entry for PagedAttention
/// Maps logical block IDs to physical GPU memory blocks
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical block ID
    pub block_id: BlockId,
    /// Physical GPU memory block
    pub physical_block: PhysicalBlock,
    /// Reference count for sharing
    pub ref_count: Arc<AtomicUsize>,
    /// Sequence IDs using this block (for sharing)
    pub sequences: HashSet<u32>,
}

pub type BlockId = u32;
```

2. **Add Physical Block Pool** (after line 53, after `CacheConfig`)

```rust
/// Pool of pre-allocated GPU memory blocks
pub struct PhysicalBlockPool {
    /// Pre-allocated GPU blocks
    blocks: Vec<PhysicalBlock>,
    /// Free list for O(1) allocation
    free_list: VecDeque<BlockId>,
    /// Block size in tokens
    block_size: usize,
    /// Number of KV heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
}

impl PhysicalBlockPool {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        backend: &HipBackend,
    ) -> KvCacheResult<Self> {
        // Allocate GPU memory for each block
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_list = VecDeque::with_capacity(num_blocks);

        for block_id in 0..num_blocks {
            let key_size = block_size * num_heads * head_dim * std::mem::size_of::<f32>();
            let value_size = key_size;

            let key_buffer = backend.allocate_buffer(key_size)?;
            let value_buffer = backend.allocate_buffer(value_size)?;

            blocks.push(PhysicalBlock {
                block_id,
                key_buffer,
                value_buffer,
            });
            free_list.push_back(block_id);
        }

        Ok(PhysicalBlockPool {
            blocks,
            free_list,
            block_size,
            num_heads,
            head_dim,
        })
    }

    pub fn allocate(&mut self) -> Option<BlockId> {
        self.free_list.pop_front()
    }

    pub fn deallocate(&mut self, block_id: BlockId) {
        self.free_list.push_back(block_id);
    }
}
```

3. **Modify `KvCache` Struct** (line 139-146)

```rust
// BEFORE:
pub struct KvCache {
    config: CacheConfig,
    backend: Arc<HipBackend>,
    pages: HashMap<u32, CachePage>,
    sequences: HashMap<u32, SequenceCache>,
    free_pages: Vec<u32>,
    next_page_id: u32,
}

// AFTER:
pub struct KvCache {
    config: CacheConfig,
    backend: Arc<HipBackend>,
    /// Block pool for physical GPU memory
    block_pool: PhysicalBlockPool,
    /// Block table: logical ID -> physical block mapping
    block_table: HashMap<BlockId, BlockTable>,
    /// Sequence to blocks mapping
    sequences: HashMap<u32, SequenceCache>,
    /// Free logical block IDs
    free_blocks: Vec<BlockId>,
    next_block_id: BlockId,
}
```

4. **Add New Methods** (before `#[cfg(test)]` at line 292)

```rust
impl KvCache {
    /// Allocate a block for a sequence (PagedAttention style)
    pub fn allocate_block(&mut self, sequence_id: u32) -> KvCacheResult<BlockId> {
        let logical_id = if let Some(free_id) = self.free_blocks.pop() {
            free_id
        } else {
            let id = self.next_block_id;
            self.next_block_id += 1;
            id
        };

        let physical_id = self.block_pool.allocate()
            .ok_or(KvCacheError::CapacityExceeded)?;

        let block_table = BlockTable {
            block_id: logical_id,
            physical_block: self.block_pool.blocks[physical_id as usize].clone(),
            ref_count: Arc::new(AtomicUsize::new(1)),
            sequences: HashSet::from([sequence_id]),
        };

        self.block_table.insert(logical_id, block_table);

        // Update sequence cache
        let sequence = self.sequences
            .entry(sequence_id)
            .or_insert_with(|| SequenceCache::new(sequence_id));
        sequence.add_block(logical_id);

        Ok(logical_id)
    }

    /// Get block for reading (PagedAttention)
    pub fn get_block(&self, block_id: BlockId) -> KvCacheResult<&BlockTable> {
        self.block_table.get(&block_id)
            .ok_or(KvCacheError::PageNotFound(block_id))
    }

    /// Increment reference count for block sharing
    pub fn ref_block(&mut self, block_id: BlockId) -> KvCacheResult<()> {
        if let Some(block) = self.block_table.get_mut(&block_id) {
            block.ref_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        } else {
            Err(KvCacheError::PageNotFound(block_id))
        }
    }

    /// Decrement reference count and free if zero
    pub fn unref_block(&mut self, block_id: BlockId) -> KvCacheResult<bool> {
        if let Some(block) = self.block_table.get_mut(&block_id) {
            let old_count = block.ref_count.fetch_sub(1, Ordering::SeqCst);
            if old_count == 1 {
                // Last reference, free the block
                self.block_pool.deallocate(block.physical_block.block_id);
                self.block_table.remove(&block_id);
                self.free_blocks.push(block_id);
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Err(KvCacheError::PageNotFound(block_id))
        }
    }

    /// Copy block for COW (Copy-on-Write) optimization
    pub fn copy_block(&mut self, block_id: BlockId, new_sequence_id: u32) -> KvCacheResult<BlockId> {
        let new_block_id = self.allocate_block(new_sequence_id)?;

        if let Some(src_block) = self.get_block(block_id) {
            // Copy GPU memory from src to dst
            let dst_block = self.get_block(new_block_id)?;

            // TODO: Use HIP memcpy kernels for GPU-to-GPU copy
            // For now, this is a placeholder

            Ok(new_block_id)
        } else {
            Err(KvCacheError::PageNotFound(block_id))
        }
    }
}
```

### 3.2 New Files to Create

#### File: `src/kv_cache/physical_block.rs` (NEW)

**Purpose**: Physical GPU block representation
**LOC**: ~150

```rust
//! Physical GPU block representation for PagedAttention

use crate::backend::{HipBuffer, HipBackend};
use std::clone::Clone;

/// Physical GPU memory block containing KV cache data
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    /// Block identifier
    pub block_id: u32,
    /// Key cache stored in GPU memory
    pub key_buffer: HipBuffer,
    /// Value cache stored in GPU memory
    pub value_buffer: HipBuffer,
}

impl PhysicalBlock {
    /// Create a new physical block
    pub fn new(
        block_id: u32,
        key_buffer: HipBuffer,
        value_buffer: HipBuffer,
    ) -> Self {
        PhysicalBlock {
            block_id,
            key_buffer,
            value_buffer,
        }
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.key_buffer.size()
    }

    /// Get the capacity in tokens
    pub fn capacity_tokens(&self) -> usize {
        self.key_buffer.size() / std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_block_creation() {
        // TODO: Add test
    }
}
```

#### File: `src/kv_cache/block_table.rs` (NEW)

**Purpose**: Block table management for PagedAttention
**LOC**: ~200

```rust
//! Block table management for PagedAttention algorithm

use super::physical_block::PhysicalBlock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashSet;

/// Block table entry mapping logical to physical blocks
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical block identifier
    pub block_id: u32,
    /// Physical GPU memory block
    pub physical_block: PhysicalBlock,
    /// Reference count for block sharing across sequences
    pub ref_count: Arc<AtomicUsize>,
    /// Set of sequence IDs using this block
    pub sequences: HashSet<u32>,
}

impl BlockTable {
    pub fn new(
        block_id: u32,
        physical_block: PhysicalBlock,
    ) -> Self {
        BlockTable {
            block_id,
            physical_block,
            ref_count: Arc::new(AtomicUsize::new(1)),
            sequences: HashSet::new(),
        }
    }

    /// Add a sequence to this block
    pub fn add_sequence(&mut self, sequence_id: u32) {
        self.sequences.insert(sequence_id);
    }

    /// Remove a sequence from this block
    pub fn remove_sequence(&mut self, sequence_id: u32) -> bool {
        self.sequences.remove(&sequence_id)
    }

    /// Get reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// Increment reference count
    pub fn incr_ref(&self) -> usize {
        self.ref_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement reference count, returns previous count
    pub fn decr_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) - 1
    }
}

/// Manager for block table operations
pub struct BlockTableManager {
    tables: HashMap<u32, BlockTable>,
    next_id: u32,
}

impl BlockTableManager {
    pub fn new() -> Self {
        BlockTableManager {
            tables: HashMap::new(),
            next_id: 0,
        }
    }

    /// Insert a new block table entry
    pub fn insert(&mut self, block: BlockTable) {
        let id = block.block_id;
        self.tables.insert(id, block);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
    }

    /// Get a block table entry
    pub fn get(&self, block_id: u32) -> Option<&BlockTable> {
        self.tables.get(&block_id)
    }

    /// Get mutable reference
    pub fn get_mut(&mut self, block_id: u32) -> Option<&mut BlockTable> {
        self.tables.get_mut(&block_id)
    }

    /// Remove a block table entry
    pub fn remove(&mut self, block_id: u32) -> Option<BlockTable> {
        self.tables.remove(&block_id)
    }
}

impl Default for BlockTableManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_table_ref_counting() {
        let block = BlockTable::new(0, unimplemented!());

        assert_eq!(block.ref_count(), 1);
        assert_eq!(block.incr_ref(), 2);
        assert_eq!(block.decr_ref(), 1);
    }

    #[test]
    fn test_block_table_sequences() {
        let mut block = BlockTable::new(0, unimplemented!());

        block.add_sequence(1);
        block.add_sequence(2);

        assert!(block.sequences.contains(&1));
        assert!(block.sequences.contains(&2));

        assert!(block.remove_sequence(1));
        assert!(!block.sequences.contains(&1));
        assert!(block.sequences.contains(&2));
    }
}
```

### 3.3 Tests to Add

#### File: `src/kv_cache/paged_attention_tests.rs` (NEW)

**Purpose**: Tests for PagedAttention functionality
**LOC**: ~300

```rust
//! Tests for PagedAttention KV cache functionality

use super::super::*;
use crate::backend::HipBackend;

#[test]
fn test_physical_block_pool_allocation() {
    let backend = HipBackend::new().unwrap();
    let pool = PhysicalBlockPool::new(100, 16, 32, 128, &backend);

    assert!(pool.is_ok());
    let pool = pool.unwrap();

    assert_eq!(pool.blocks.len(), 100);
    assert_eq!(pool.free_list.len(), 100);
}

#[test]
fn test_block_allocation() {
    let backend = HipBackend::new().unwrap();
    let mut pool = PhysicalBlockPool::new(10, 16, 32, 128, &backend).unwrap();

    // Allocate first block
    let block1 = pool.allocate();
    assert!(block1.is_some());
    assert_eq!(block1.unwrap(), 0);

    // Free list decreased
    assert_eq!(pool.free_list.len(), 9);
}

#[test]
fn test_block_deallocation() {
    let backend = HipBackend::new().unwrap();
    let mut pool = PhysicalBlockPool::new(10, 16, 32, 128, &backend).unwrap();

    pool.allocate().unwrap();
    let block2 = pool.allocate().unwrap();

    pool.deallocate(block2);

    assert_eq!(pool.free_list.len(), 9);
}

#[test]
fn test_block_table_ref_counting() {
    let table = BlockTable::new(0, unimplemented!());

    assert_eq!(table.ref_count(), 1);
    table.incr_ref();
    assert_eq!(table.ref_count(), 2);
    table.decr_ref();
    assert_eq!(table.ref_count(), 1);
}

#[test]
fn test_kv_cache_allocate_block() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(16, 100, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    let block_id = cache.allocate_block(1);
    assert!(block_id.is_ok());
    assert_eq!(block_id.unwrap(), 0);
}

#[test]
fn test_kv_cache_block_ref_unref() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(16, 100, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    let block_id = cache.allocate_block(1).unwrap();

    // Ref twice
    cache.ref_block(block_id).unwrap();
    cache.ref_block(block_id).unwrap();

    // Unref once - still alive
    assert!(!cache.unref_block(block_id).unwrap());

    // Unref again - should free
    assert!(cache.unref_block(block_id).unwrap());
}

// Property tests
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_block_pool_allocation_deallocation(
        num_blocks in 1..1000usize,
        alloc_count in 1..100usize
    ) {
        let backend = HipBackend::new().unwrap();
        let mut pool = PhysicalBlockPool::new(num_blocks, 16, 32, 128, &backend).unwrap();

        let mut allocated = Vec::new();
        for _ in 0..alloc_count {
            if let Some(block_id) = pool.allocate() {
                allocated.push(block_id);
            }
        }

        // All allocated blocks should be unique
        let unique: HashSet<_> = allocated.iter().collect();
        prop_assert_eq!(unique.len(), allocated.len());

        // Deallocate all
        for block_id in allocated {
            pool.deallocate(block_id);
        }

        // Free list should be restored
        prop_assert_eq!(pool.free_list.len(), num_blocks);
    }
}
```

### 3.4 Module Updates

#### File: `src/kv_cache/mod.rs` (MODIFY)

**Current**:
```rust
pub mod kv_cache;
```

**After**:
```rust
pub mod block_table;
pub mod kv_cache;
pub mod physical_block;

#[cfg(test)]
mod paged_attention_tests;
```

---

## 4. Task 12.2: Continuous Batching Enhancement

### Overview

**Goal**: Enhance scheduler to support true continuous batching (vLLM style)
**Effort**: 1 week
**Impact**: 10-23x throughput improvement

### 4.1 Files to Modify

#### File: `src/scheduler/scheduler.rs` (MODIFY)

**Current LOC**: 641
**Target LOC**: ~850

**Key Changes**:

1. **Add Iteration Batch State** (after line 227, after `Batch` struct)

```rust
/// Represents a single iteration's batch for continuous batching
#[derive(Debug)]
pub struct IterationBatch {
    pub requests: Vec<GenerationRequest>,
    pub sequence_positions: Vec<usize>,
    pub completed_indices: Vec<usize>,
}

impl IterationBatch {
    pub fn new() -> Self {
        IterationBatch {
            requests: Vec::new(),
            sequence_positions: Vec::new(),
            completed_indices: Vec::new(),
        }
    }

    /// Remove completed requests and re-sort remaining by position
    pub fn compact(&mut self) {
        let mut active_requests = Vec::new();
        let mut active_positions = Vec::new();

        for (i, req) in self.requests.iter().enumerate() {
            if !req.is_complete() && req.state != RequestState::Failed {
                active_requests.push(req.clone());
                active_positions.push(self.sequence_positions[i]);
            } else {
                self.completed_indices.push(i);
            }
        }

        self.requests = active_requests;
        self.sequence_positions = active_positions;
    }

    pub fn size(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}
```

2. **Add Continuous Batching Logic** (after line 297, modify `create_batch`)

```rust
/// Create a batch with continuous batching support
/// Unlike static batching, this allows requests to enter/exit dynamically
pub fn create_batch(&mut self) -> SchedulerResult<Batch> {
    // First, update processing requests from previous iteration
    self.update_processing_state();

    let batch_id = self.next_batch_id;
    self.next_batch_id += 1;

    let mut batch = Batch::new(batch_id);

    // Strategy 1: Add back any still-processing requests (continuous batching)
    for (req_id, request) in &self.processing_requests {
        if request.state == RequestState::Processing && batch.size() < self.config.max_batch_size {
            batch.add_request(request.clone())?;
        }
    }

    // Strategy 2: Fill remaining slots with new pending requests
    while batch.size() < self.config.max_batch_size && !self.pending_queue.is_empty() {
        // Sort pending by priority (e.g., arrival time)
        if let Some(mut request) = self.pending_queue.pop_front() {
            request.start_processing()?;
            self.processing_requests.insert(request.request_id, request.clone());
            batch.add_request(request)?;
        }
    }

    Ok(batch)
}

/// Update processing state after an iteration
fn update_processing_state(&mut self) {
    let mut to_complete = Vec::new();

    for (req_id, request) in &self.processing_requests {
        if request.is_complete() || request.state == RequestState::Failed {
            to_complete.push(*req_id);
        }
    }

    for req_id in to_complete {
        if let Some(mut request) = self.processing_requests.remove(&req_id) {
            if request.state == RequestState::Processing {
                let _ = request.complete(None);
            }
            self.completed_requests.insert(req_id, request);
        }
    }
}
```

3. **Add `get_next_iteration_batch` Method** (after `can_create_batch` at line 444)

```rust
/// Get the next iteration's batch for continuous batching
/// This is the main entry point for the inference loop
pub fn get_next_iteration_batch(&mut self) -> SchedulerResult<IterationBatch> {
    // Move completed requests out of processing
    self.update_processing_state();

    let mut iteration_batch = IterationBatch::new();

    // Add all currently processing requests
    for (req_id, request) in &self.processing_requests {
        if request.state == RequestState::Processing {
            iteration_batch.requests.push(request.clone());
            iteration_batch.sequence_positions.push(request.total_tokens());
        }
    }

    // Fill empty slots with new requests from pending queue
    while iteration_batch.size() < self.config.max_batch_size && !self.pending_queue.is_empty() {
        if let Some(mut request) = self.pending_queue.pop_front() {
            request.start_processing()?;
            let req_id = request.request_id;
            self.processing_requests.insert(req_id, request.clone());
            iteration_batch.requests.push(request);
            iteration_batch.sequence_positions.push(request.total_tokens());
        }
    }

    Ok(iteration_batch)
}

/// Update batch after one iteration of token generation
pub fn update_iteration_batch(&mut self, batch: IterationBatch) -> SchedulerResult<Vec<GenerationRequest>> {
    let mut completed = Vec::new();

    for request in batch.requests {
        let req_id = request.request_id;
        self.processing_requests.insert(req_id, request);

        if request.is_complete() || request.state == RequestState::Failed {
            if let Some(mut req) = self.processing_requests.remove(&req_id) {
                if req.state == RequestState::Processing {
                    let _ = req.complete(None);
                }
                self.completed_requests.insert(req_id, req.clone());
                completed.push(req);
            }
        }
    }

    Ok(completed)
}
```

### 4.2 New Files to Create

#### File: `src/scheduler/continuous_batch.rs` (NEW)

**Purpose**: Continuous batching logic
**LOC**: ~250

```rust
//! Continuous batching implementation for vLLM-style scheduling

use super::{GenerationRequest, RequestState, SchedulerConfig, SchedulerError};
use std::collections::{HashMap, VecDeque};

pub type SchedulerResult<T> = Result<T, SchedulerError>;

/// Configuration for continuous batching
#[derive(Debug, Clone)]
pub struct ContinuousBatchConfig {
    /// Maximum number of concurrent requests
    pub max_concurrent: usize,
    /// Maximum wait time before forced batch creation
    pub max_wait_ms: u64,
    /// Target batch size for optimal throughput
    pub target_batch_size: usize,
}

impl Default for ContinuousBatchConfig {
    fn default() -> Self {
        ContinuousBatchConfig {
            max_concurrent: 32,
            max_wait_ms: 50,
            target_batch_size: 16,
        }
    }
}

/// Manages continuous batching state across iterations
pub struct ContinuousBatchManager {
    config: ContinuousBatchConfig,
    active_requests: HashMap<u32, GenerationRequest>,
    pending_queue: VecDeque<GenerationRequest>,
}

impl ContinuousBatchManager {
    pub fn new(config: ContinuousBatchConfig) -> Self {
        ContinuousBatchManager {
            config,
            active_requests: HashMap::new(),
            pending_queue: VecDeque::new(),
        }
    }

    /// Add a new request to the pending queue
    pub fn add_request(&mut self, request: GenerationRequest) {
        self.pending_queue.push_back(request);
    }

    /// Get the next iteration's batch
    /// This implements the core continuous batching logic:
    /// 1. Keep processing requests from previous iteration
    /// 2. Add new requests to fill empty slots
    /// 3. Remove completed requests
    pub fn get_iteration_batch(&mut self) -> Vec<GenerationRequest> {
        let mut batch = Vec::new();

        // First: add all still-processing requests (continuous batching)
        let mut to_remove = Vec::new();
        for (req_id, request) in &self.active_requests {
            if request.is_complete() || request.state == RequestState::Failed {
                to_remove.push(*req_id);
            } else if batch.len() < self.config.max_concurrent {
                batch.push(request.clone());
            }
        }

        // Remove completed requests
        for req_id in to_remove {
            self.active_requests.remove(&req_id);
        }

        // Second: fill remaining slots with new requests
        while batch.len() < self.config.max_concurrent && !self.pending_queue.is_empty() {
            if let Some(request) = self.pending_queue.pop_front() {
                batch.push(request);
            } else {
                break;
            }
        }

        batch
    }

    /// Update batch with new generated tokens
    pub fn update_batch(&mut self, requests: Vec<GenerationRequest>) -> Vec<GenerationRequest> {
        let mut completed = Vec::new();

        for request in requests {
            let req_id = request.request_id;
            self.active_requests.insert(req_id, request.clone());

            if request.is_complete() || request.state == RequestState::Failed {
                if let Some(req) = self.active_requests.remove(&req_id) {
                    completed.push(req);
                }
            }
        }

        completed
    }

    /// Get statistics about the continuous batching state
    pub fn get_stats(&self) -> ContinuousBatchStats {
        ContinuousBatchStats {
            active_requests: self.active_requests.len(),
            pending_requests: self.pending_queue.len(),
            total_capacity: self.config.max_concurrent,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContinuousBatchStats {
    pub active_requests: usize,
    pub pending_requests: usize,
    pub total_capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_batch_manager_creation() {
        let config = ContinuousBatchConfig::default();
        let manager = ContinuousBatchManager::new(config);

        assert_eq!(manager.active_requests.len(), 0);
        assert_eq!(manager.pending_queue.len(), 0);
    }

    #[test]
    fn test_add_request() {
        let config = ContinuousBatchConfig::default();
        let mut manager = ContinuousBatchManager::new(config);

        let request = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);
        manager.add_request(request);

        assert_eq!(manager.pending_queue.len(), 1);
    }

    #[test]
    fn test_get_iteration_batch() {
        let config = ContinuousBatchConfig {
            max_concurrent: 4,
            ..Default::default()
        };
        let mut manager = ContinuousBatchManager::new(config);

        // Add some requests
        for i in 0..3 {
            manager.add_request(GenerationRequest::new(
                i, vec![1, 2, 3], 10, 0.8, 50, 0.9
            ));
        }

        let batch = manager.get_iteration_batch();
        assert_eq!(batch.len(), 3);
    }
}
```

### 4.3 Tests to Add

#### File: `src/scheduler/continuous_batch_tests.rs` (NEW)

**Purpose**: Tests for continuous batching
**LOC**: ~200

```rust
//! Tests for continuous batching functionality

use super::super::*;
use super::continuous_batch::*;

#[test]
fn test_iteration_batch_compaction() {
    let mut batch = IterationBatch::new();

    // Add some requests
    batch.requests.push(mock_request(1, 10));
    batch.sequence_positions.push(10);
    batch.requests.push(mock_request(2, 20));
    batch.sequence_positions.push(20);

    // Mark one as complete
    batch.requests[1].state = RequestState::Completed;

    batch.compact();

    assert_eq!(batch.size(), 1);
    assert_eq!(batch.completed_indices.len(), 1);
}

fn mock_request(id: u32, tokens: usize) -> GenerationRequest {
    GenerationRequest::new(id, vec![1; tokens], 100, 0.8, 50, 0.9)
}
```

---

## 5. Task 12.3: Attention Backend Registry

### Overview

**Goal**: Create pluggable attention backend system like vLLM
**Effort**: 1 week
**Impact**: Flexibility, extensibility

### 5.1 Files to Modify

#### File: `src/attention/backend.rs` (REPLACE ENTIRE FILE)

**Current LOC**: 15
**Target LOC**: ~200

**New Implementation**:

```rust
//! Backend abstraction for attention computation
//!
//! Implements pluggable attention backend system inspired by vLLM.
//! Supports multiple attention implementations with automatic selection.

use std::sync::Arc;
use thiserror::Error;

/// Error type for attention backend operations
#[derive(Error, Debug)]
pub enum AttentionBackendError {
    #[error("Backend not found: {0}")]
    NotFound(String),
    #[error("Backend initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Feature not supported by backend: {0}")]
    NotSupported(String),
}

pub type AttentionBackendResult<T> = Result<T, AttentionBackendError>;

/// Trait for attention backend implementations
pub trait AttentionBackend: Send + Sync {
    /// Get the name of this backend
    fn name(&self) -> &str;

    /// Check if this backend supports the given configuration
    fn supports(&self, config: &AttentionConfig) -> bool;

    /// Get the required KV cache layout (if any)
    fn required_kv_layout(&self) -> Option<KvCacheLayout>;

    /// Execute attention computation
    fn forward(
        &self,
        config: &AttentionConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
    ) -> AttentionBackendResult<Vec<f32>>;
}

/// Configuration for attention operations
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Model dimension
    pub dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Whether to use causal masking
    pub is_causal: bool,
    /// Dropout probability
    pub dropout: Option<f32>,
}

impl AttentionConfig {
    pub fn new(dim: usize, num_heads: usize, head_dim: usize) -> Self {
        AttentionConfig {
            dim,
            num_heads,
            head_dim,
            max_sequence_length: 4096,
            is_causal: false,
            dropout: None,
        }
    }

    pub fn with_causal(mut self, is_causal: bool) -> Self {
        self.is_causal = is_causal;
        self
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = Some(dropout);
        self
    }
}

/// KV cache layout options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheLayout {
    /// Standard contiguous layout
    Contiguous,
    /// Block-sparse layout (for PagedAttention)
    BlockSparse,
    /// FlashAttention-optimized layout
    FlashAttention,
}

/// Simple CPU/GPU backend selection (legacy, for backward compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum LegacyBackend {
    #[default]
    Cpu,
    #[cfg(feature = "rocm")]
    Gpu,
}

/// Attention backend registry with pluggable implementations
pub struct AttentionBackendRegistry {
    backends: Vec<Box<dyn AttentionBackend>>,
    default_backend: Option<String>,
}

impl AttentionBackendRegistry {
    pub fn new() -> Self {
        let backends: Vec<Box<dyn AttentionBackend>> = vec![
            Box::new(cpu_backend::CpuAttentionBackend::new()),
            #[cfg(feature = "rocm")]
            Box::new(gpu_backend::GpuAttentionBackend::new()),
        ];

        AttentionBackendRegistry {
            backends,
            default_backend: None,
        }
    }

    /// Register a new backend
    pub fn register(&mut self, backend: Box<dyn AttentionBackend>) {
        self.backends.push(backend);
    }

    /// Select the best backend for the given configuration
    pub fn select_backend(
        &self,
        config: &AttentionConfig,
    ) -> AttentionBackendResult<&dyn AttentionBackend> {
        // First try default if set
        if let Some(ref default_name) = self.default_backend {
            if let Some(backend) = self.backends.iter().find(|b| b.name() == default_name) {
                return Ok(backend.as_ref());
            }
        }

        // Auto-select based on configuration
        for backend in &self.backends {
            if backend.supports(config) {
                return Ok(backend.as_ref());
            }
        }

        Err(AttentionBackendError::NotFound(
            "No suitable backend found for configuration".to_string()
        ))
    }

    /// Set the default backend by name
    pub fn set_default(&mut self, name: String) -> AttentionBackendResult<()> {
        let exists = self.backends.iter().any(|b| b.name() == name);
        if exists {
            self.default_backend = Some(name);
            Ok(())
        } else {
            Err(AttentionBackendError::NotFound(name))
        }
    }

    /// Get all registered backend names
    pub fn list_backends(&self) -> Vec<String> {
        self.backends.iter().map(|b| b.name().to_string()).collect()
    }
}

impl Default for AttentionBackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Module for CPU backend implementation
pub mod cpu_backend {
    use super::*;
    use crate::attention::cpu as cpu_impl;

    pub struct CpuAttentionBackend;

    impl CpuAttentionBackend {
        pub fn new() -> Self {
            CpuAttentionBackend
        }
    }

    impl AttentionBackend for CpuAttentionBackend {
        fn name(&self) -> &str {
            "cpu"
        }

        fn supports(&self, _config: &AttentionConfig) -> bool {
            true // CPU always supports everything
        }

        fn required_kv_layout(&self) -> Option<KvCacheLayout> {
            Some(KvCacheLayout::Contiguous)
        }

        fn forward(
            &self,
            config: &AttentionConfig,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            mask: Option<&[f32]>,
        ) -> AttentionBackendResult<Vec<f32>> {
            cpu_impl::CpuBackend::forward(
                config.dim,
                q,
                k,
                v,
                mask,
                None, // dropout not implemented in CPU backend
            ).map_err(|e| AttentionBackendError::InitializationFailed(e.to_string()))
        }
    }
}

// Module for GPU backend implementation
#[cfg(feature = "rocm")]
pub mod gpu_backend {
    use super::*;
    use crate::attention::gpu as gpu_impl;

    pub struct GpuAttentionBackend {
        #[allow(dead_code)] // Will be used when FlashAttention is implemented
        use_flash_attention: bool,
    }

    impl GpuAttentionBackend {
        pub fn new() -> Self {
            GpuAttentionBackend {
                use_flash_attention: false, // TODO: detect from system
            }
        }
    }

    impl AttentionBackend for GpuAttentionBackend {
        fn name(&self) -> &str {
            "gpu"
        }

        fn supports(&self, config: &AttentionConfig) -> bool {
            // GPU supports most configurations
            config.dim % config.num_heads == 0 // dim must be divisible by num_heads
        }

        fn required_kv_layout(&self) -> Option<KvCacheLayout> {
            if self.use_flash_attention {
                Some(KvCacheLayout::FlashAttention)
            } else {
                Some(KvCacheLayout::BlockSparse)
            }
        }

        fn forward(
            &self,
            config: &AttentionConfig,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            mask: Option<&[f32]>,
        ) -> AttentionBackendResult<Vec<f32>> {
            // Use GPU implementation
            gpu_impl::GpuBackend::forward(
                config.dim,
                q,
                k,
                v,
                mask,
                None,
            ).map_err(|e| AttentionBackendError::InitializationFailed(e.to_string()))
        }
    }
}
```

### 5.2 Module Updates

#### File: `src/attention/mod.rs` (MODIFY)

**Add after line 12**:

```rust
pub mod backend_registry;

use backend_registry::{
    AttentionBackend as AttentionBackendTrait,
    AttentionBackendRegistry,
    AttentionConfig,
    AttentionBackendError,
    AttentionBackendResult,
    KvCacheLayout,
};
```

---

## 6. Task 12.4: Integration & Testing

### 6.1 Integration Points

#### File: `src/engine.rs` (MODIFY)

**Location**: Update InferenceEngine to use new scheduler

**Changes at line ~120** (scheduler initialization):

```rust
// BEFORE:
let scheduler = Arc::new(RwLock::new(Scheduler::new(SchedulerConfig {
    max_batch_size: config.max_batch_size,
    max_queue_size: 1000,
    batch_timeout: config.batch_timeout,
    max_sequence_length: config.max_sequence_length,
})));

// AFTER:
use crate::scheduler::continuous_batch::ContinuousBatchConfig;

let scheduler = Arc::new(RwLock::new(Scheduler::new_with_continuous(
    SchedulerConfig {
        max_batch_size: config.max_batch_size,
        max_queue_size: 1000,
        batch_timeout: config.batch_timeout,
        max_sequence_length: config.max_sequence_length,
    },
    ContinuousBatchConfig::default(),
)));
```

### 6.2 Tests to Verify

**Run after each task**:

```bash
# After Task 12.1 (PagedAttention)
cargo test --lib kv_cache::paged_attention_tests

# After Task 12.2 (Continuous Batching)
cargo test --lib scheduler::continuous_batch_tests

# After Task 12.3 (Backend Registry)
cargo test --lib attention::backend_registry

# Full test suite
cargo test --lib
```

---

## 7. CodeMCP Tool Usage Guide

### 7.1 Finding Symbols

**ALWAYS use** `find_symbols` **instead of grep**:

```bash
# Find all occurrences of a symbol
find_symbols(query="allocate_page")

# Find references only
find_symbols(query="references:KvCache")

# List all symbols in a file
find_symbols(query="file:src/kv_cache/kv_cache.rs")
```

### 7.2 Refactoring Operations

**For symbol renaming**:

```bash
# Rename a symbol across all files
refactor_rename(
    symbol_name="KvCache",
    new_name="PagedKvCache",
    kind="struct",
    workspace_root="/home/feanor/Projects/ROCmForge"
)
```

**For symbol deletion**:

```bash
# Delete a symbol and all references
refactor_delete(
    symbol_name="old_function",
    file="src/kv_cache/kv_cache.rs",
    kind="fn",
    workspace_root="/home/feanor/Projects/ROCmForge"
)
```

### 7.3 Reading Code

**ALWAYS use** `Read` **tool**:

```bash
# Read a file to understand its current state
Read /path/to/file.rs

# Get code chunks for a specific symbol
get_code_chunks(file_path="src/kv_cache/kv_cache.rs", symbol_name="allocate_page")
```

---

## 8. Success Criteria

### 8.1 Test Health

| Metric | Before | Target |
|--------|--------|--------|
| Unit Tests Passing | 116/116 | 150/150 |
| Integration Tests | Compiling | Compiling |
| New PagedAttention Tests | 0 | 34 |
| New Continuous Batch Tests | 0 | 20 |

### 8.2 Performance Benchmarks

| Metric | Before | Target |
|--------|--------|--------|
| Memory Efficiency (vs static) | 1x | 3x |
| Throughput Improvement | 1x | 10x |
| Cache Fragmentation | High | Near Zero |
| Batch Formation Time | N/A | <5ms |

### 8.3 Code Quality

| Metric | Target |
|--------|--------|
| No unwrap() in prod paths | ✅ Pass |
| Max 300 LOC per file | ✅ Pass |
| All changes documented | ✅ Pass |
| No dead_code attributes | ✅ Pass |

---

## 9. Execution Order

**CRITICAL**: Follow this order EXACTLY:

```
Week 1, Day 1-2: Task 12.1.1-12.1.3 (Physical Block & Block Table)
Week 1, Day 3-4: Task 12.1.4 (Modify KvCache)
Week 1, Day 5:   Task 12.1.5 (Tests)

Week 2, Day 1-2: Task 12.2.1 (Modify Scheduler)
Week 2, Day 3-4: Task 12.2.2 (ContinuousBatchManager)
Week 2, Day 5:   Task 12.2.3 (Tests)

Week 3, Day 1-3: Task 12.3.1-12.3.2 (Backend Registry)
Week 3, Day 4-5: Task 12.4 (Integration & Testing)
```

---

## 10. Risk Mitigation

### Risk 1: Breaking Changes

**Mitigation**:
- Run `cargo test --lib` after EACH file modification
- Use `refactor_rename` for any symbol changes
- Store architectural decision before coding

### Risk 2: Performance Regression

**Mitigation**:
- Benchmark before and after each task
- Profile with `rocm-smi` during tests
- Compare against baseline metrics

### Risk 3: Memory Leaks

**Mitigation**:
- Valgrind/GPU memory profiling after Task 12.1
- Check reference counting in block table
- Test with high request load

---

## 11. Next Steps After Phase 12

Once Phase 12 is COMPLETE:

1. **Phase 13.1**: Model Auto-Discovery (Shimmy pattern)
2. **Phase 13.2**: Universal Backend Adapter
3. **Phase 13.3**: Response Caching (LRU)

---

**REMEMBER**:
- NEVER GUESS - ALWAYS VERIFY
- Store architectural decisions before coding
- TDD: Write failing tests first
- Use CodeMCP tools for all refactoring
- Cite your sources

**DO NOT DEVIATE FROM THIS PLAN** without architectural decision record.
