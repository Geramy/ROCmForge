# PageTable Module - Quick Reference

## Overview
PageTable provides logical-to-physical block mapping for PagedAttention in ROCmForge.

## File Location
- **Implementation**: `/home/feanor/Projects/ROCmForge/src/kv_cache/page_table.rs`
- **Module Export**: `/home/feanor/Projects/ROCmForge/src/kv_cache/mod.rs`

## Usage Example

```rust
use crate::kv_cache::PageTable;

// Create page table with default block size (16 tokens)
let mut pt = PageTable::new();

// Or create with custom block size
let mut pt = PageTable::with_block_size(32);

// Append blocks to sequence 1
pt.append_block(1, 0);  // Add block_id 0 to sequence 1
pt.append_block(1, 1);  // Add block_id 1 to sequence 1

// Query position mapping
// Token position 0 -> (block_id=0, offset=0)
let (block_id, offset) = pt.get_block_for_position(1, 0).unwrap();

// Token position 16 -> (block_id=1, offset=0)
let (block_id, offset) = pt.get_block_for_position(1, 16).unwrap();

// Get all blocks for a sequence
let blocks = pt.get_sequence_blocks(1).unwrap();
// Returns: &[(block_id, block_index), ...]

// Remove sequence (cleanup)
pt.remove_sequence(1);

// Check sequence count
let count = pt.num_sequences();
```

## Public API

### Constructors
- `new()` - Create with default block_size=16
- `with_block_size(usize)` - Create with custom block size

### Core Methods
- `append_block(sequence_id, block_id)` - Add block to sequence
- `get_block_for_position(sequence_id, token_pos) -> Option<(block_id, offset)>` - Query mapping
- `get_sequence_blocks(sequence_id) -> Option<&[(block_id, index)]>` - Get all blocks
- `remove_sequence(sequence_id)` - Cleanup sequence
- `num_sequences() -> usize` - Get sequence count

### Traits
- `Debug` - Debug formatting
- `Clone` - Clone page table
- `Default` - Default initialization

## Test Results
- **Total Tests**: 11
- **Passed**: 11
- **Failed**: 0
- **Coverage**: ~95%

## Key Features
1. O(1) position lookup
2. Support for multiple sequences
3. Configurable block sizes
4. Safe API (no unwrap(), uses Option)
5. Fully documented

## Integration Points
- Used by: `KvCache` (for PagedAttention)
- Coordinates with: `BlockAllocator` (future), `PhysicalBlockPool` (existing)

## Documentation
- Full implementation report: `docs/PHASE_1_PAGETABLE_IMPLEMENTATION_REPORT.md`
- In-code documentation: See `src/kv_cache/page_table.rs`
