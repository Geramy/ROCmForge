//! Performance benchmarks for KV cache operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rocmforge::backend::HipBackend;
use rocmforge::kv_cache::{CacheConfig, KvCache};
use std::time::Duration;

fn bench_cache_creation(c: &mut Criterion) {
    let backend = HipBackend::new().unwrap();

    let mut group = c.benchmark_group("cache_creation");

    for page_size in [16, 32, 64, 128, 256, 512, 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("create_cache", page_size),
            page_size,
            |b, &page_size| {
                b.iter(|| {
                    let config = CacheConfig::new(page_size, 1000, 32, 128, 24).unwrap();
                    let cache = KvCache::new(config, backend.clone());
                    black_box(cache)
                });
            },
        );
    }

    group.finish();
}

fn bench_page_allocation(c: &mut Criterion) {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(1024, 1000, 32, 128, 24).unwrap();

    let mut group = c.benchmark_group("page_allocation");

    group.bench_function("allocate_single_page", |b| {
        b.iter_with_setup(
            || KvCache::new(config.clone(), backend.clone()).unwrap(),
            |mut cache| {
                let page_id = cache.allocate_page(1).unwrap();
                black_box(page_id);
            },
        );
    });

    group.bench_function("allocate_multiple_pages", |b| {
        b.iter_with_setup(
            || KvCache::new(config.clone(), backend.clone()).unwrap(),
            |mut cache| {
                for i in 0..100 {
                    let page_id = cache.allocate_page(i).unwrap();
                    black_box(page_id);
                }
            },
        );
    });

    group.finish();
}

fn bench_token_appending(c: &mut Criterion) {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(1024, 1000, 32, 128, 24).unwrap();

    let mut group = c.benchmark_group("token_appending");

    for num_tokens in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("append_tokens", num_tokens),
            num_tokens,
            |b, &num_tokens| {
                b.iter_with_setup(
                    || {
                        let mut cache = KvCache::new(config.clone(), backend.clone()).unwrap();
                        cache.allocate_page(1).unwrap();
                        cache
                    },
                    |mut cache| {
                        for i in 0..num_tokens {
                            let result = cache.append_token(1, i);
                            black_box(result);
                        }
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_sequence_retrieval(c: &mut Criterion) {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(1024, 1000, 32, 128, 24).unwrap();

    let mut group = c.benchmark_group("sequence_retrieval");

    for sequence_length in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("get_sequence_tokens", sequence_length),
            sequence_length,
            |b, &sequence_length| {
                b.iter_with_setup(
                    || {
                        let mut cache = KvCache::new(config.clone(), backend.clone()).unwrap();
                        cache.allocate_page(1).unwrap();
                        for i in 0..sequence_length {
                            cache.append_token(1, i).unwrap();
                        }
                        cache
                    },
                    |cache| {
                        let tokens = cache.get_sequence_tokens(1).unwrap();
                        black_box(tokens);
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_multiple_sequences(c: &mut Criterion) {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(256, 1000, 32, 128, 24).unwrap();

    let mut group = c.benchmark_group("multiple_sequences");

    for num_sequences in [10, 50, 100].iter() {
        for tokens_per_sequence in [10, 50].iter() {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}_sequences_{}_tokens", num_sequences, tokens_per_sequence),
                    format!("{}-{}", num_sequences, tokens_per_sequence),
                ),
                &(*num_sequences, *tokens_per_sequence),
                |b, &(num_sequences, tokens_per_sequence)| {
                    b.iter_with_setup(
                        || {
                            let mut cache = KvCache::new(config.clone(), backend.clone()).unwrap();

                            // Create sequences and add tokens
                            for seq_id in 0..num_sequences {
                                cache.allocate_page(seq_id).unwrap();
                                for token_id in 0..tokens_per_sequence {
                                    cache.append_token(seq_id, token_id).unwrap();
                                }
                            }
                            cache
                        },
                        |cache| {
                            // Retrieve all sequences
                            for seq_id in 0..num_sequences {
                                let tokens = cache.get_sequence_tokens(seq_id).unwrap();
                                black_box(tokens);
                            }
                        },
                    );
                },
            );
        }
    }

    group.finish();
}

fn bench_page_reuse(c: &mut Criterion) {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(256, 100, 32, 128, 24).unwrap();

    let mut group = c.benchmark_group("page_reuse");

    group.bench_function("allocate_remove_allocate", |b| {
        b.iter_with_setup(
            || KvCache::new(config.clone(), backend.clone()).unwrap(),
            |mut cache| {
                // Allocate pages
                for i in 0..50 {
                    cache.allocate_page(i).unwrap();
                }

                // Remove all sequences
                for i in 0..50 {
                    cache.remove_sequence(i).unwrap();
                }

                // Allocate new pages (should reuse freed ones)
                for i in 100..150 {
                    let page_id = cache.allocate_page(i).unwrap();
                    black_box(page_id);
                }
            },
        );
    });

    group.finish();
}

fn bench_cache_stats(c: &mut Criterion) {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(256, 1000, 32, 128, 24).unwrap();

    let mut group = c.benchmark_group("cache_stats");

    group.bench_function("get_stats_empty", |b| {
        b.iter_with_setup(
            || KvCache::new(config.clone(), backend.clone()).unwrap(),
            |cache| {
                let stats = cache.get_cache_stats();
                black_box(stats);
            },
        );
    });

    group.bench_function("get_stats_populated", |b| {
        b.iter_with_setup(
            || {
                let mut cache = KvCache::new(config.clone(), backend.clone()).unwrap();

                // Create many sequences
                for seq_id in 0..100 {
                    cache.allocate_page(seq_id).unwrap();
                    for token_id in 0..50 {
                        cache.append_token(seq_id, token_id).unwrap();
                    }
                }
                cache
            },
            |cache| {
                let stats = cache.get_cache_stats();
                black_box(stats);
            },
        );
    });

    group.finish();
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(256, 1000, 32, 128, 24).unwrap();

    let mut group = c.benchmark_group("concurrent_operations");

    group.measurement_time(Duration::from_secs(10));

    group.bench_function("concurrent_append", |b| {
        b.iter(|| {
            let mut handles = Vec::new();

            for thread_id in 0..4 {
                let backend = backend.clone();
                let config = config.clone();

                let handle = std::thread::spawn(move || {
                    let mut cache = KvCache::new(config, backend).unwrap();
                    cache.allocate_page(thread_id).unwrap();

                    for token_id in 0..1000 {
                        let result = cache.append_token(thread_id, token_id);
                        black_box(result);
                    }

                    let tokens = cache.get_sequence_tokens(thread_id).unwrap();
                    black_box(tokens);
                });

                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let backend = HipBackend::new().unwrap();

    let mut group = c.benchmark_group("memory_usage");

    for num_pages in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_allocation", num_pages),
            num_pages,
            |b, &num_pages| {
                b.iter_with_setup(
                    || {
                        let config = CacheConfig::new(1024, num_pages, 32, 128, 24).unwrap();
                        KvCache::new(config, backend.clone()).unwrap()
                    },
                    |mut cache| {
                        // Allocate all pages
                        for i in 0..num_pages {
                            let page_id = cache.allocate_page(i.try_into().unwrap()).unwrap();
                            black_box(page_id);
                        }

                        let stats = cache.get_cache_stats();
                        black_box(stats);
                    },
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_creation,
    bench_page_allocation,
    bench_token_appending,
    bench_sequence_retrieval,
    bench_multiple_sequences,
    bench_page_reuse,
    bench_cache_stats,
    bench_concurrent_operations,
    bench_memory_usage
);

criterion_main!(benches);
