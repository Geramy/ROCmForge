# Option B: Async GPU Loading - Architecture Diagrams

## Current Architecture (Synchronous)

```
┌─────────────────────────────────────────────────────────────┐
│                     GgufLoader::load_to_gpu()               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  For each tensor (sequential loop):                         │
│                                                             │
│  1. Read from mmap (zero-copy)                              │
│  2. Dequantize on CPU (single-threaded)                     │
│  3. Upload to GPU (synchronous, 1 stream)                   │
│  4. Wait for upload to complete                             │
│  5. Cache in gpu_cache                                      │
│                                                             │
│  Total: 50s for 7B model (35s CPU + 12s GPU + 3s overhead)  │
└─────────────────────────────────────────────────────────────┘

  Timeline:
  Tensor 0:  [████████████████████████]  dequant  [████████████████]  upload
  Tensor 1:                                   [████████████████████████]  dequant  [████████████████]  upload
  Tensor 2:                                                                              [████████████████████████]  dequant  [████████████████]  upload
  ...
  Total:  50s  ────────────────────────────────────────────────────────────────────────────────────────────────>

  Problem: No overlap between CPU work and GPU work!
```

## Target Architecture (Async)

```
┌─────────────────────────────────────────────────────────────┐
│              AsyncModelLoader::load_to_gpu_async()          │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌────────────────────────────┐
│  Phase 1: CPU (Parallel) │    │  Phase 2: GPU (Concurrent) │
│                          │    │                            │
│  Rayon Thread Pool (8T)  │    │  HIP Streams (4 streams)   │
│                          │    │                            │
│  ┌────────────────────┐  │    │  Stream 0: Tensor 0,4,8.. │
│  │  Tensor 0: Q4_0    │  │    │  Stream 1: Tensor 1,5,9.. │
│  │  Tensor 1: Q8_0    │──┼───→│  Stream 2: Tensor 2,6,10.│
│  │  Tensor 2: F16     │  │    │  Stream 3: Tensor 3,7,11.│
│  │  Tensor 3: Q4_0    │  │    │                            │
│  │  ...               │  │    │  Event Tracking:           │
│  └────────────────────┘  │    │  - Record event per upload│
│                          │    │  - Synchronize all events  │
│  Time: ~5s (parallel)    │    │  Time: ~4s (concurrent)   │
└──────────────────────────┘    └────────────────────────────┘

  Timeline:
  CPU Thread 0:  [████████] T0 dequant  [████████] T8 dequant
  CPU Thread 1:  [████████] T1 dequant  [████████] T9 dequant
  CPU Thread 2:  [████████] T2 dequant  [████████] T10 dequant
  CPU Thread 3:  [████████] T3 dequant  [████████] T11 dequant
  ...
  GPU Stream 0:        [████████████] T0 upload        [████████████] T8 upload
  GPU Stream 1:        [████████████] T1 upload        [████████████] T9 upload
  GPU Stream 2:        [████████████] T2 upload        [████████████] T10 upload
  GPU Stream 3:        [████████████] T3 upload        [████████████] T11 upload
  Total:  10s  ──────────────────────────────────────────────────────────────────>

  Benefit: 5x speedup from parallelization + concurrency!
```

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                            │
│                    (load 7B model to GPU)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AsyncModelLoader                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Fields:                                                    │ │
│  │ - backend: Arc<HipBackend>                                 │ │
│  │ - upload_streams: Vec<HipStream>  (4 streams)              │ │
│  │ - completion_events: Mutex<HashMap<String, HipEvent>>      │ │
│  │ - num_cpu_threads: usize (8)                               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────┬───────────────────────────────┬───────────────────┘
              │                               │
      ┌───────┴───────┐               ┌──────┴────────┐
      ▼               ▼               ▼               ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│HipBackend│   │HipStream │   │HipEvent  │   │Rayon     │
│          │   │× 4       │   │per tensor│   │Thread    │
│          │   │          │   │          │   │Pool × 8  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
      │               │               │               │
      └───────────────┴───────────────┴───────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GPU Memory (16GB VRAM)                     │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐              │
│  │Stream 0│  │Stream 1│  │Stream 2│  │Stream 3│              │
│  │Tensor 0│  │Tensor 1│  │Tensor 2│  │Tensor 3│  ...        │
│  │Tensor 4│  │Tensor 5│  │Tensor 6│  │Tensor 7│              │
│  └────────┘  └────────┘  └────────┘  └────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
Step 1: Metadata Parsing
┌────────────┐     ┌────────────┐     ┌────────────┐
│  GGUF File │────→│  MmapGguf  │────→│ LazyTensor │
│            │     │ (zero-copy)│     │  Handles   │
└────────────┘     └────────────┘     └────────────┘
                                            │
                                            ▼
Step 2: Parallel CPU Dequantization (8 threads)
┌──────────────────────────────────────────────────────────────┐
│  Thread 0: [T0: Q4_0] ──→ f32 data (500MB)                  │
│  Thread 1: [T1: Q8_0] ──→ f32 data (500MB)                  │
│  Thread 2: [T2: F16]  ──→ f32 data (500MB)                  │
│  Thread 3: [T3: Q4_0] ──→ f32 data (500MB)                  │
│  ...                                                       │
│  Total: 4GB f32 data in ~5 seconds                          │
└──────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
Step 3: Concurrent GPU Upload (4 streams)
┌──────────────────────────────────────────────────────────────┐
│  Stream 0: [T0] ──hipMemcpyAsync→ GPU Event0                 │
│  Stream 1: [T1] ──hipMemcpyAsync→ GPU Event1                 │
│  Stream 2: [T2] ──hipMemcpyAsync→ GPU Event2                 │
│  Stream 3: [T3] ──hipMemcpyAsync→ GPU Event3                 │
│  ...                                                       │
│  Total: 4GB in ~4 seconds (concurrent)                      │
└──────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
Step 4: Synchronization
┌──────────────────────────────────────────────────────────────┐
│  Wait for all events:                                        │
│  Event0::synchronize() ✓                                     │
│  Event1::synchronize() ✓                                     │
│  Event2::synchronize() ✓                                     │
│  ...                                                       │
│  All tensors ready in GPU memory!                           │
└──────────────────────────────────────────────────────────────┘
```

## Memory Layout Comparison

### Current (Synchronous)
```
Host RAM (32GB):
  [GGUF mmap: 4GB] ───► [CPU dequant buffer: 4GB] ───► [upload buffer] ───► GPU
       │                        │                         │
       │ zero-copy              │ temporary              │ freed after upload
       └────────────────────────┴─────────────────────────┘

GPU VRAM (16GB):
  [Tensor 0: 500MB]
  [Tensor 1: 500MB]
  [Tensor 2: 500MB]
  ... (uploaded sequentially)

Problem: 4GB temporary buffer allocation, no reuse
```

### Target (Async)
```
Host RAM (32GB):
  [GGUF mmap: 4GB] ───► [Thread-local buffers: 8 × 64MB = 512MB]
       │                      │
       │ zero-copy            │ per-thread, reused
       └──────────────────────┘

GPU VRAM (16GB):
  [Stream 0: Tensor 0,4,8...]
  [Stream 1: Tensor 1,5,9...]
  [Stream 2: Tensor 2,6,10...]
  [Stream 3: Tensor 3,7,11...]
  (uploaded concurrently, events track completion)

Improvement: 8x less host memory (512MB vs 4GB)
```

## Performance Comparison Table

| Metric                     | Current (Sync) | Target (Async) | Improvement |
|----------------------------|----------------|----------------|-------------|
| **Total Loading Time**     | 50s            | 10s            | **5.0x**    |
| **CPU Dequantization**     | 35s (1 thread) | 5s (8 threads) | **7.0x**    |
| **GPU Uploads**            | 12s (1 stream) | 4s (4 streams) | **3.0x**    |
| **Host Memory Usage**      | 4GB            | 512MB          | **8.0x**    |
| **CPU Utilization**        | 12.5%          | 100%           | **8.0x**    |
| **GPU Utilization**        | 33%            | 100%           | **3.0x**    |
| **PCIe Bandwidth**         | 2.8 GB/s       | 8.4 GB/s       | **3.0x**    |

## Timeline Visualization

### Current Timeline (50s total)
```
0s     10s    20s    30s    40s    50s
│      │      │      │      │      │
└──────┴──────┴──────┴──────┴──────┴
  T0 deq  T0 upl  T1 deq  T1 upl  T2...

Key observations:
- CPU and GPU never work simultaneously
- GPU idle while CPU works (35s)
- CPU idle while GPU works (12s)
- Total waste: 47s idle time
```

### Target Timeline (10s total)
```
0s     2s     4s     6s     8s     10s
│      │      │      │      │      │
└──────┴──────┴──────┴──────┴──────┴
  T0-7 deq (parallel)  T0-7 upl (concurrent)

Key improvements:
- All CPU threads work simultaneously (5s)
- All GPU streams work simultaneously (4s)
- CPU and GPU overlap (pipelined)
- Total waste: <1s idle time
```

## Risk Mitigation Strategies

### Risk 1: Race Conditions in GPU Cache
```
Problem: Multiple threads writing to gpu_cache simultaneously
Solution: Arc<DeviceTensor> is immutable, write protected by Mutex

┌─────────────────────────────────────────────────────────┐
│  Thread 0: cache.write_lock() → insert(T0) → unlock()  │
│  Thread 1: cache.write_lock() → insert(T1) → unlock()  │
│  Thread 2: cache.write_lock() → insert(T2) → unlock()  │
│  ...                                                   │
│  Mutex ensures sequential access, no data races        │
└─────────────────────────────────────────────────────────┘
```

### Risk 2: GPU Memory Exhaustion
```
Problem: Uploading too many tensors simultaneously
Solution: Pre-calculate memory, batch uploads

┌─────────────────────────────────────────────────────────┐
│  1. Calculate total GPU memory needed: Σ tensor_sizes   │
│  2. Check available GPU memory: HipBackend::mem_info() │
│  3. If insufficient, batch uploads (e.g., 16 tensors)  │
│  4. Use HipEvent to track batch completion              │
└─────────────────────────────────────────────────────────┘
```

### Risk 3: Stream Starvation
```
Problem: Too many streams competing for PCIe bandwidth
Solution: Limit to 4 streams (optimal for PCIe 3.0 x16)

┌─────────────────────────────────────────────────────────┐
│  Streams: 1    2    4    8    16                       │
│  Bandwidth: 2.8  5.6  8.4  8.4  8.4 GB/s               │
│                 ↑                                        │
│           Diminishing returns beyond 4 streams          │
└─────────────────────────────────────────────────────────┘
```

## Test Coverage Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Test Pyramid                                  │
│                                                                  │
│                      ┌───────────┐                               │
│                      │  E2E Test │  1 test                       │
│                      │  (smoke)  │  async_model_loading          │
│                      └─────┬─────┘                               │
│                            │                                     │
│          ┌─────────────────┼─────────────────┐                   │
│          ▼                 ▼                 ▼                   │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐          │
│  │ Integration   │ │  Integration  │ │ Integration   │          │
│  │ (memory)      │ │ (correctness) │ │ (stress)      │          │
│  │ 1 test        │ │ 3 tests       │ │ 2 tests       │          │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘          │
│          │                 │                 │                   │
│  ┌───────┼─────────────────┼─────────────────┼───────┐          │
│  ▼       ▼                 ▼                 ▼       ▼          │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│ │Unit: HIP│ │Unit:    │ │Unit:    │ │Unit:    │ │Bench:   │     │
│ │Event    │ │Rayon    │ │Async    │ │Correct- │ │Perf     │     │
│ │2 tests  │ │dequant  │ │Upload   │ │ness     │ │3 benches│     │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
│                                                                  │
│  Total: ~20 tests + 3 benchmarks                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-11
**Status:** DESIGN DOCUMENT (pending implementation)
