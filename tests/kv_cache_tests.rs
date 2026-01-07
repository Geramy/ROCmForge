//! Comprehensive TDD tests for KV cache module

use rocmforge::backend::HipBackend;
use rocmforge::kv_cache::{CacheConfig, CachePage, KvCache, KvCacheError, SequenceCache};
use std::collections::HashMap;

#[test]
fn test_cache_config_validation() {
    // Valid configuration
    let config = CacheConfig::new(1024, 100, 32, 128, 24);
    assert!(config.is_ok());

    let config = config.unwrap();
    assert_eq!(config.page_size, 1024);
    assert_eq!(config.max_pages, 100);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.num_layers, 24);

    // Invalid configurations
    let invalid_configs = vec![
        CacheConfig::new(0, 100, 32, 128, 24),   // page_size = 0
        CacheConfig::new(1024, 0, 32, 128, 24),  // max_pages = 0
        CacheConfig::new(1024, 100, 0, 128, 24), // num_heads = 0
        CacheConfig::new(1024, 100, 32, 0, 24),  // head_dim = 0
        CacheConfig::new(1024, 100, 32, 128, 0), // num_layers = 0
    ];

    for invalid_config in invalid_configs {
        assert!(invalid_config.is_err());
        assert!(matches!(
            invalid_config,
            Err(KvCacheError::InvalidConfiguration)
        ));
    }
}

#[test]
fn test_kv_cache_initialization() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(1024, 100, 32, 128, 24).unwrap();
    let cache = KvCache::new(config, backend);

    assert!(cache.is_ok());

    let cache = cache.unwrap();
    let stats = cache.get_cache_stats();
    assert_eq!(stats.total_pages, 0);
    assert_eq!(stats.free_pages, 0);
    assert_eq!(stats.active_sequences, 0);
    assert_eq!(stats.total_tokens, 0);
}

#[test]
fn test_page_allocation() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    // Allocate first page
    let page_id1 = cache.allocate_page(1);
    assert!(page_id1.is_ok());
    assert_eq!(page_id1.unwrap(), 0);

    let stats = cache.get_cache_stats();
    assert_eq!(stats.total_pages, 1);
    assert_eq!(stats.free_pages, 0);
    assert_eq!(stats.active_sequences, 1);

    // Allocate second page for same sequence
    let page_id2 = cache.allocate_page(1);
    assert!(page_id2.is_ok());
    assert_eq!(page_id2.unwrap(), 1);

    let stats = cache.get_cache_stats();
    assert_eq!(stats.total_pages, 2);
    assert_eq!(stats.free_pages, 0);
    assert_eq!(stats.active_sequences, 1);

    // Allocate page for different sequence
    let page_id3 = cache.allocate_page(2);
    assert!(page_id3.is_ok());
    assert_eq!(page_id3.unwrap(), 2);

    let stats = cache.get_cache_stats();
    assert_eq!(stats.total_pages, 3);
    assert_eq!(stats.free_pages, 0);
    assert_eq!(stats.active_sequences, 2);
}

#[test]
fn test_capacity_limit() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(4, 2, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    // Allocate up to capacity
    let page_id1 = cache.allocate_page(1);
    assert!(page_id1.is_ok());

    let page_id2 = cache.allocate_page(2);
    assert!(page_id2.is_ok());

    // Should fail when exceeding capacity
    let page_id3 = cache.allocate_page(3);
    assert!(page_id3.is_err());
    assert!(matches!(page_id3, Err(KvCacheError::CapacityExceeded)));
}

#[test]
fn test_token_appending() {
    let backend = HipBackend::new().unwrap();
    // Set max_pages=1 to prevent auto-allocation when page is full
    let config = CacheConfig::new(4, 1, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    // Allocate a page first
    cache.allocate_page(1).unwrap();

    // Append tokens within page capacity
    for i in 0..4 {
        let result = cache.append_token(1, i);
        assert!(result.is_ok());
    }

    // Should fail when page is full (max_pages=1 prevents auto-allocation)
    let result = cache.append_token(1, 5);
    assert!(result.is_err());
    assert!(matches!(result, Err(KvCacheError::CapacityExceeded)));
}

#[test]
fn test_token_appending_with_new_page() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(2, 10, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    cache.allocate_page(1).unwrap();

    // Fill first page
    cache.append_token(1, 1).unwrap();
    cache.append_token(1, 2).unwrap();

    // Should automatically allocate new page
    let result = cache.append_token(1, 3);
    assert!(result.is_ok());

    let stats = cache.get_cache_stats();
    assert_eq!(stats.total_pages, 2);
    assert_eq!(stats.active_sequences, 1);
}

#[test]
fn test_sequence_retrieval() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    cache.allocate_page(1).unwrap();

    // Append tokens
    let expected_tokens = vec![10, 20, 30];
    for &token in &expected_tokens {
        cache.append_token(1, token).unwrap();
    }

    // Retrieve tokens
    let retrieved_tokens = cache.get_sequence_tokens(1).unwrap();
    assert_eq!(retrieved_tokens, expected_tokens);

    // Check sequence length
    let length = cache.get_sequence_length(1).unwrap();
    assert_eq!(length, 3);
}

#[test]
fn test_sequence_removal() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    cache.allocate_page(1).unwrap();
    cache.append_token(1, 42).unwrap();

    let stats_before = cache.get_cache_stats();
    assert_eq!(stats_before.active_sequences, 1);
    assert_eq!(stats_before.free_pages, 0);

    // Remove sequence
    cache.remove_sequence(1).unwrap();

    let stats_after = cache.get_cache_stats();
    assert_eq!(stats_after.active_sequences, 0);
    assert_eq!(stats_after.free_pages, 1);

    // Should not be able to retrieve removed sequence
    let result = cache.get_sequence_tokens(1);
    assert!(result.is_err());
    assert!(matches!(result, Err(KvCacheError::InvalidSequenceId(1))));
}

#[test]
fn test_multiple_sequences() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(4, 20, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    // Create multiple sequences
    cache.allocate_page(1).unwrap();
    cache.allocate_page(2).unwrap();
    cache.allocate_page(3).unwrap();

    // Add tokens to each sequence
    cache.append_token(1, 100).unwrap();
    cache.append_token(1, 101).unwrap();

    cache.append_token(2, 200).unwrap();

    cache.append_token(3, 300).unwrap();
    cache.append_token(3, 301).unwrap();
    cache.append_token(3, 302).unwrap();

    // Verify each sequence
    let seq1_tokens = cache.get_sequence_tokens(1).unwrap();
    assert_eq!(seq1_tokens, vec![100, 101]);

    let seq2_tokens = cache.get_sequence_tokens(2).unwrap();
    assert_eq!(seq2_tokens, vec![200]);

    let seq3_tokens = cache.get_sequence_tokens(3).unwrap();
    assert_eq!(seq3_tokens, vec![300, 301, 302]);

    let stats = cache.get_cache_stats();
    assert_eq!(stats.active_sequences, 3);
    assert_eq!(stats.total_tokens, 6);
}

#[test]
fn test_page_reuse() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    // Create and remove sequence
    cache.allocate_page(1).unwrap();
    cache.append_token(1, 42).unwrap();
    cache.remove_sequence(1).unwrap();

    let stats_after_removal = cache.get_cache_stats();
    assert_eq!(stats_after_removal.free_pages, 1);

    // Create new sequence - should reuse the freed page
    cache.allocate_page(2).unwrap();

    let stats_after_reuse = cache.get_cache_stats();
    assert_eq!(stats_after_reuse.total_pages, 1); // Still only 1 page
    assert_eq!(stats_after_reuse.free_pages, 0); // Page is now used
    assert_eq!(stats_after_reuse.active_sequences, 1);
}

#[test]
fn test_invalid_operations() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    // Try to append token to non-existent sequence
    let result = cache.append_token(999, 42);
    assert!(result.is_err());
    assert!(matches!(result, Err(KvCacheError::InvalidSequenceId(999))));

    // Try to get tokens from non-existent sequence
    let result = cache.get_sequence_tokens(999);
    assert!(result.is_err());
    assert!(matches!(result, Err(KvCacheError::InvalidSequenceId(999))));

    // Try to get length of non-existent sequence
    let result = cache.get_sequence_length(999);
    assert!(result.is_err());
    assert!(matches!(result, Err(KvCacheError::InvalidSequenceId(999))));

    // Try to remove non-existent sequence
    let result = cache.remove_sequence(999);
    assert!(result.is_err());
    assert!(matches!(result, Err(KvCacheError::InvalidSequenceId(999))));
}

// Property-based tests
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_token_appending_properties(
        tokens in prop::collection::vec(0u32..1000, 1..20),
        page_size in 1usize..10
    ) {
        let backend = HipBackend::new().unwrap();
        // Set max_pages=1 to prevent auto-allocation
        let config = CacheConfig::new(page_size, 1, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        cache.allocate_page(1).unwrap();

        let mut success_count = 0;
        for &token in &tokens {
            if cache.append_token(1, token).is_ok() {
                success_count += 1;
            }
        }

        let retrieved = cache.get_sequence_tokens(1).unwrap();
        assert_eq!(retrieved.len(), success_count);

        // Check that retrieved tokens match the first success_count tokens
        assert_eq!(&retrieved[..], &tokens[..success_count]);

        // Verify that we can't exceed page capacity (max_pages=1 prevents auto-allocation)
        assert!(success_count <= page_size);
    }

    #[test]
    fn test_multiple_sequences_properties(
        seq1_tokens in prop::collection::vec(1u32..100, 1..10),
        seq2_tokens in prop::collection::vec(101u32..200, 1..10),
        page_size in 5usize..15
    ) {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(page_size, 20, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Create two sequences
        cache.allocate_page(1).unwrap();
        cache.allocate_page(2).unwrap();

        // Add tokens to sequence 1
        let mut seq1_success = 0;
        for &token in &seq1_tokens {
            if cache.append_token(1, token).is_ok() {
                seq1_success += 1;
            }
        }

        // Add tokens to sequence 2
        let mut seq2_success = 0;
        for &token in &seq2_tokens {
            if cache.append_token(2, token).is_ok() {
                seq2_success += 1;
            }
        }

        // Verify sequences
        let retrieved1 = cache.get_sequence_tokens(1).unwrap();
        assert_eq!(retrieved1.len(), seq1_success);
        assert_eq!(&retrieved1[..], &seq1_tokens[..seq1_success]);

        let retrieved2 = cache.get_sequence_tokens(2).unwrap();
        assert_eq!(retrieved2.len(), seq2_success);
        assert_eq!(&retrieved2[..], &seq2_tokens[..seq2_success]);

        // Verify stats
        let stats = cache.get_cache_stats();
        assert_eq!(stats.active_sequences, 2);
        assert_eq!(stats.total_tokens, seq1_success + seq2_success);
    }

    #[test]
    fn test_sequence_lifecycle_properties(
        operations in prop::collection::vec(
            prop_oneof![
                any::<u32>().prop_map(|x| ('a', x)), // add token
                any::<u32>().prop_map(|x| ('r', x)), // remove sequence
                any::<u32>().prop_map(|x| ('c', x)), // create sequence
            ],
            1..20
        ),
        page_size in 3usize..8
    ) {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(page_size, 20, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        let mut active_sequences = std::collections::HashSet::new();

        for (op_type, value) in operations {
            match op_type {
                'a' => { // add token
                    if active_sequences.contains(&value) {
                        let _ = cache.append_token(value, value);
                    }
                }
                'r' => { // remove sequence
                    if active_sequences.contains(&value) {
                        let _ = cache.remove_sequence(value);
                        active_sequences.remove(&value);
                    }
                }
                'c' => { // create sequence
                    if !active_sequences.contains(&value) {
                        let _ = cache.allocate_page(value);
                        active_sequences.insert(value);
                    }
                }
                _ => unreachable!(),
            }
        }

        // Final verification
        let stats = cache.get_cache_stats();
        assert_eq!(stats.active_sequences, active_sequences.len());

        // Verify all active sequences can be retrieved
        for &seq_id in &active_sequences {
            let _ = cache.get_sequence_tokens(seq_id).unwrap();
        }
    }
}
