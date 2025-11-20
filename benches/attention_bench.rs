//! Attention performance benchmarks comparing CPU vs GPU implementations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rocmforge::attention::{Attention, AttentionBackend};
use rocmforge::tensor::Tensor;

fn bench_cpu_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_cpu");

    // Small case: batch=4, seq=128, dim=64
    group.bench_with_input(
        BenchmarkId::new("small", "small"),
        &(4, 128, 64),
        |b, &(batch_size, seq_len, dim)| {
            b.iter(|| {
                let q = Tensor::random(batch_size * seq_len * dim);
                let k = Tensor::random(batch_size * seq_len * dim);
                let v = Tensor::random(batch_size * seq_len * dim);

                let attention = Attention::with_backend(dim, AttentionBackend::Cpu);
                black_box(
                    attention
                        .forward(&q.data, &k.data, &v.data, None, None)
                        .unwrap(),
                );
            });
        },
    );

    // Medium case: batch=2, seq=256, dim=128
    group.bench_with_input(
        BenchmarkId::new("medium", "medium"),
        &(2, 256, 128),
        |b, &(batch_size, seq_len, dim)| {
            b.iter(|| {
                let q = Tensor::random(batch_size * seq_len * dim);
                let k = Tensor::random(batch_size * seq_len * dim);
                let v = Tensor::random(batch_size * seq_len * dim);

                let attention = Attention::with_backend(dim, AttentionBackend::Cpu);
                black_box(
                    attention
                        .forward(&q.data, &k.data, &v.data, None, None)
                        .unwrap(),
                );
            });
        },
    );

    // Large case: batch=1, seq=512, dim=256
    group.bench_with_input(
        BenchmarkId::new("large", "large"),
        &(1, 512, 256),
        |b, &(batch_size, seq_len, dim)| {
            b.iter(|| {
                let q = Tensor::random(batch_size * seq_len * dim);
                let k = Tensor::random(batch_size * seq_len * dim);
                let v = Tensor::random(batch_size * seq_len * dim);

                let attention = Attention::with_backend(dim, AttentionBackend::Cpu);
                black_box(
                    attention
                        .forward(&q.data, &k.data, &v.data, None, None)
                        .unwrap(),
                );
            });
        },
    );

    group.finish();
}

#[cfg(feature = "rocm")]
fn bench_gpu_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_gpu");

    // Small case: batch=4, seq=128, dim=64
    group.bench_with_input(
        BenchmarkId::new("small", "small"),
        &(4, 128, 64),
        |b, &(batch_size, seq_len, dim)| {
            b.iter(|| {
                let q = Tensor::random(batch_size * seq_len * dim);
                let k = Tensor::random(batch_size * seq_len * dim);
                let v = Tensor::random(batch_size * seq_len * dim);

                let attention = Attention::with_backend(dim, AttentionBackend::Gpu);
                black_box(
                    attention
                        .forward(&q.data, &k.data, &v.data, None, None)
                        .unwrap(),
                );
            });
        },
    );

    // Medium case: batch=2, seq=256, dim=128
    group.bench_with_input(
        BenchmarkId::new("medium", "medium"),
        &(2, 256, 128),
        |b, &(batch_size, seq_len, dim)| {
            b.iter(|| {
                let q = Tensor::random(batch_size * seq_len * dim);
                let k = Tensor::random(batch_size * seq_len * dim);
                let v = Tensor::random(batch_size * seq_len * dim);

                let attention = Attention::with_backend(dim, AttentionBackend::Gpu);
                black_box(
                    attention
                        .forward(&q.data, &k.data, &v.data, None, None)
                        .unwrap(),
                );
            });
        },
    );

    // Large case: batch=1, seq=512, dim=256
    group.bench_with_input(
        BenchmarkId::new("large", "large"),
        &(1, 512, 256),
        |b, &(batch_size, seq_len, dim)| {
            b.iter(|| {
                let q = Tensor::random(batch_size * seq_len * dim);
                let k = Tensor::random(batch_size * seq_len * dim);
                let v = Tensor::random(batch_size * seq_len * dim);

                let attention = Attention::with_backend(dim, AttentionBackend::Gpu);
                black_box(
                    attention
                        .forward(&q.data, &k.data, &v.data, None, None)
                        .unwrap(),
                );
            });
        },
    );

    group.finish();
}

#[cfg(feature = "rocm")]
fn bench_qk_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk_matmul");

    // QK^T matmul micro-benchmark: batch=1, seq=128, dim=64
    group.bench_with_input(
        BenchmarkId::new("cpu", "cpu"),
        &(1, 128, 64),
        |b, &(batch_size, seq_len, dim)| {
            b.iter(|| {
                let q = Tensor::random(batch_size * seq_len * dim);
                let k = Tensor::random(batch_size * seq_len * dim);

                // Use CPU matmul directly
                let attention = Attention::with_backend(dim, AttentionBackend::Cpu);
                black_box(
                    attention
                        .forward(&q.data, &k.data, &Tensor::zeros(1), None, None)
                        .unwrap(),
                );
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("gpu", "gpu"),
        &(1, 128, 64),
        |b, &(batch_size, seq_len, dim)| {
            b.iter(|| {
                let q = Tensor::random(batch_size * seq_len * dim);
                let k = Tensor::random(batch_size * seq_len * dim);

                // Use GPU matmul directly
                let attention = Attention::with_backend(dim, AttentionBackend::Gpu);
                black_box(
                    attention
                        .forward(&q.data, &k.data, &Tensor::zeros(1), None, None)
                        .unwrap(),
                );
            });
        },
    );

    group.finish();
}

#[cfg(feature = "rocm")]
criterion_group!(
    benches,
    bench_cpu_attention,
    bench_gpu_attention,
    bench_qk_matmul
);

#[cfg(not(feature = "rocm"))]
criterion_group!(benches, bench_cpu_attention);

criterion_main!(benches);
