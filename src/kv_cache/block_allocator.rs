//! BlockAllocator for KV cache memory management
//!
//! The BlockAllocator provides O(1) allocation/deallocation of KV cache blocks
//! using a free list approach. It works alongside PhysicalBlockPool and PageTable.

use std::collections::VecDeque;

/// Manages allocation/deallocation of KV cache blocks
///
/// # Purpose
/// Provides fast O(1) block allocation using a free list. This is a logical
/// allocator that tracks which block IDs are available, separate from the
/// PhysicalBlockPool which manages actual GPU memory.
///
/// # Block Size
/// Blocks are pre-allocated during initialization. The allocator maintains
/// a free list of available block IDs for constant-time allocation.
///
/// # Example
/// ```ignore
/// let mut alloc = BlockAllocator::new(100, 16, 32, 128);
/// assert_eq!(alloc.total_blocks(), 100);
/// assert_eq!(alloc.free_blocks(), 100);
///
/// let block_id = alloc.allocate().unwrap();
/// assert_eq!(alloc.free_blocks(), 99);
///
/// alloc.deallocate(block_id);
/// assert_eq!(alloc.free_blocks(), 100);
/// ```
#[derive(Debug)]
pub struct BlockAllocator {
    /// Pre-allocated GPU blocks
    blocks: Vec<PhysicalBlock>,

    /// Free block IDs (O(1) allocation)
    free_list: VecDeque<BlockId>,

    /// Block configuration
    block_size: usize,
    num_heads: usize,
    head_dim: usize,
}

/// A physical KV cache block stored in GPU memory
#[derive(Debug)]
pub struct PhysicalBlock {
    pub block_id: BlockId,
    pub key_buffer: Option<crate::backend::HipBuffer>,
    pub value_buffer: Option<crate::backend::HipBuffer>,
    pub ref_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

pub type BlockId = u32;

impl BlockAllocator {
    /// Create a new BlockAllocator
    ///
    /// # Arguments
    /// * `num_blocks` - Total number of blocks to allocate
    /// * `block_size` - Number of tokens per block
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each attention head
    ///
    /// # Returns
    /// A new BlockAllocator with all blocks initially free
    ///
    /// # Example
    /// ```ignore
    /// let alloc = BlockAllocator::new(100, 16, 32, 128);
    /// assert_eq!(alloc.total_blocks(), 100);
    /// assert_eq!(alloc.free_blocks(), 100);
    /// ```
    pub fn new(num_blocks: usize, block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        // Create empty allocator - blocks will be allocated later
        // Actual GPU allocation happens in initialize()
        let mut free_list = VecDeque::with_capacity(num_blocks);
        for i in 0..num_blocks {
            free_list.push_back(i as BlockId);
        }

        Self {
            blocks: Vec::with_capacity(num_blocks),
            free_list,
            block_size,
            num_heads,
            head_dim,
        }
    }

    /// Allocate a single block
    ///
    /// # Returns
    /// * `Some(block_id)` - Next available block ID
    /// * `None` - No blocks available
    ///
    /// # Example
    /// ```ignore
    /// let mut alloc = BlockAllocator::new(10, 16, 32, 128);
    /// let block_id = alloc.allocate().unwrap();
    /// assert_eq!(block_id, 0); // First block
    /// assert_eq!(alloc.free_blocks(), 9);
    /// ```
    pub fn allocate(&mut self) -> Option<BlockId> {
        self.free_list.pop_front()
    }

    /// Allocate multiple contiguous blocks for a sequence
    ///
    /// # Arguments
    /// * `count` - Number of blocks to allocate
    ///
    /// # Returns
    /// * `Some(Vec<BlockId>)` - Vector of allocated block IDs
    /// * `None` - Not enough blocks available
    ///
    /// # Example
    /// ```ignore
    /// let mut alloc = BlockAllocator::new(10, 16, 32, 128);
    /// let blocks = alloc.allocate_sequence(3).unwrap();
    /// assert_eq!(blocks.len(), 3);
    /// assert_eq!(blocks, vec![0, 1, 2]);
    /// assert_eq!(alloc.free_blocks(), 7);
    /// ```
    pub fn allocate_sequence(&mut self, count: usize) -> Option<Vec<BlockId>> {
        if self.free_list.len() < count {
            return None;
        }
        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            blocks.push(self.free_list.pop_front()?);
        }
        Some(blocks)
    }

    /// Deallocate a block (returns to free list)
    ///
    /// # Arguments
    /// * `block_id` - Block ID to deallocate
    ///
    /// # Example
    /// ```ignore
    /// let mut alloc = BlockAllocator::new(10, 16, 32, 128);
    /// let block_id = alloc.allocate().unwrap();
    /// alloc.deallocate(block_id);
    /// assert_eq!(alloc.free_blocks(), 10);
    /// ```
    pub fn deallocate(&mut self, block_id: BlockId) {
        self.free_list.push_back(block_id);
    }

    /// Get total number of blocks
    pub fn total_blocks(&self) -> usize {
        self.blocks.capacity()
    }

    /// Get number of free blocks
    pub fn free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Get block size (tokens per block)
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get number of attention heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator_new() {
        let alloc = BlockAllocator::new(100, 16, 32, 128);
        assert_eq!(alloc.total_blocks(), 100);
        assert_eq!(alloc.free_blocks(), 100);
    }

    #[test]
    fn test_block_allocator_allocate() {
        let mut alloc = BlockAllocator::new(10, 16, 32, 128);
        let block_id = alloc.allocate().unwrap();
        assert_eq!(block_id, 0); // First block
        assert_eq!(alloc.free_blocks(), 9);
    }

    #[test]
    fn test_block_allocator_allocate_sequence() {
        let mut alloc = BlockAllocator::new(10, 16, 32, 128);
        let blocks = alloc.allocate_sequence(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks, vec![0, 1, 2]);
        assert_eq!(alloc.free_blocks(), 7);
    }

    #[test]
    fn test_block_allocator_deallocate() {
        let mut alloc = BlockAllocator::new(10, 16, 32, 128);
        let block_id = alloc.allocate().unwrap();
        alloc.deallocate(block_id);
        assert_eq!(alloc.free_blocks(), 10);
    }

    #[test]
    fn test_block_allocator_exhausted() {
        let mut alloc = BlockAllocator::new(1, 16, 32, 128);
        alloc.allocate().unwrap();
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn test_block_allocator_allocate_sequence_too_many() {
        let mut alloc = BlockAllocator::new(5, 16, 32, 128);
        assert!(alloc.allocate_sequence(10).is_none());
    }

    #[test]
    fn test_block_allocator_deallocate_reuse() {
        let mut alloc = BlockAllocator::new(10, 16, 32, 128);

        // Allocate all blocks
        let block_ids: Vec<BlockId> = (0..10).map(|_| alloc.allocate().unwrap()).collect();
        assert_eq!(alloc.free_blocks(), 0);

        // Deallocate all blocks
        for block_id in block_ids {
            alloc.deallocate(block_id);
        }
        assert_eq!(alloc.free_blocks(), 10);

        // Allocate again - should reuse IDs
        let _new_block = alloc.allocate().unwrap();
        assert_eq!(alloc.free_blocks(), 9);
        // The exact ID depends on deallocation order, but some ID should be available
    }

    #[test]
    fn test_block_allocator_config() {
        let alloc = BlockAllocator::new(100, 16, 32, 128);
        assert_eq!(alloc.block_size(), 16);
        assert_eq!(alloc.num_heads(), 32);
        assert_eq!(alloc.head_dim(), 128);
    }

    #[test]
    fn test_block_allocator_empty() {
        let mut alloc = BlockAllocator::new(0, 16, 32, 128);
        assert_eq!(alloc.total_blocks(), 0);
        assert_eq!(alloc.free_blocks(), 0);
        assert!(alloc.allocate().is_none());
    }
}
