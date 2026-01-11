# Strategic Comparison: Shimmy, vLLM, and ATOM
## Architecture Analysis and Learnings for ROCmForge

**Date**: 2026-01-08
**Purpose**: Strategic research to inform ROCmForge development roadmap
**Scope**: Architecture, features, and implementation patterns from competing inference engines

---

## Executive Summary

This document analyzes three leading LLM inference platforms to identify strategic opportunities for ROCmForge:

| Project | Type | Key Differentiator | Primary Language | Target Hardware |
|---------|------|-------------------|------------------|------------------|
| **Shimmy** | Inference API Server | Universal backend adapter, auto-discovery | Rust | Multi-vendor |
| **vLLM** | Inference Engine | PagedAttention, continuous batching | Python | NVIDIA/AMD/Intel |
| **Rebellions ATOM** | NPU Hardware | Inference-optimized silicon | Hardware + Software | Rebellions NPU |

**Key Finding**: vLLM's PagedAttention and continuous batching deliver **3-23x throughput improvements** over static batching. Shimmy's universal backend pattern enables **single-binary deployment** across multiple inference engines. ATOM demonstrates **hardware-software co-design** for inference optimization.

**Strategic Priority**: ROCmForge should implement PagedAttention and continuous batching (Phase 12) while adopting Shimmy's universal backend pattern for ecosystem compatibility.

---

## 1. Shimmy Analysis

### 1.1 Architecture Overview

**Location**: `/home/feanor/Projects/shimmy`

**Core Philosophy**: Universal backend adapter with zero-configuration auto-discovery

```
┌─────────────────────────────────────────────────────────────┐
│                    Shimmy CLI (Single Binary)               │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │   Model    │  │   Backend    │  │   Response          │ │
│  │  Registry  │  │   Adapter    │  │   Cache             │ │
│  └────────────┘  └──────────────┘  └─────────────────────┘ │
│         │                 │                     │            │
│  ┌──────▼─────────────────▼─────────────────────▼───────┐  │
│  │              Auto-Discovery System                    │  │
│  │  (GGUF, SafeTensors, HuggingFace, Ollama, MLX)       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Key Features

#### Model Registry (`src/model_registry.rs:16-123`)
```rust
pub struct Registry {
    inner: HashMap<String, ModelEntry>,           // User-registered models
    discovered_models: HashMap<String, DiscoveredModel>, // Auto-discovered
}
```

**Strategic Value**: Dual-layer model management enables both explicit configuration and automatic model discovery.

#### Auto-Discovery System (`src/auto_discovery.rs:54-168`)

**Search Paths** (comprehensive multi-ecosystem):
- `./models` (local working directory)
- `$SHIMMY_BASE_GGUF` parent directory
- `$SHIMMY_MODEL_PATHS` (custom paths)
- `$OLLAMA_MODELS`
- `~/.cache/huggingface/hub`
- `~/.ollama/models`
- `~/.lmstudio/models`
- Platform-specific paths (Windows/macOS)

**Intelligent Filtering**:
- Excludes non-LLM models (flux, whisper, clip, etc.)
- Groups sharded files (`model-00001-of-00004.safetensors`)
- Distinguishes GGUF, SafeTensors, Ollama formats

#### Backend Adapter (`src/engine/adapter.rs:61-181`)

```rust
fn select_backend(&self, spec: &ModelSpec) -> BackendChoice {
    // Priority-based selection:
    // 1. File extension (.safetensors > .gguf > .hf-model-id)
    // 2. Platform optimization (MLX on Apple Silicon)
    // 3. Pattern matching (Ollama blobs, model names)
    // 4. Fallback to HuggingFace
}
```

**Supported Backends**:
- **LlamaGGUF**: llama.cpp via `shimmy-llama-cpp-2`
- **HuggingFace**: Python-free HF model loading
- **SafeTensors**: Native Rust implementation
- **MLX**: Apple Silicon Metal GPU

#### Response Caching (`src/cache/response_cache.rs:140-282`)

**Features**:
- LRU eviction with size-based limits
- TTL support (default 1 hour)
- Statistics tracking (hit rate, time saved)
- Background cleanup task
- Configurable limits (1000 entries, 512MB max)

### 1.3 Strategic Learnings for ROCmForge

| Feature | Shimmy Approach | ROCmForge Opportunity |
|---------|-----------------|----------------------|
| **Backend Abstraction** | Universal adapter pattern | Create ROCm/HIP/CUDA abstraction layer |
| **Model Discovery** | Multi-ecosystem search | Add HuggingFace/Ollama/LMStudio paths |
| **Caching** | LRU with TTL | Implement similar for GPU tensors |
| **Error Handling** | Standardized error types | Adopt consistent error patterns |
| **Binary Size** | Sub-5MB with selective compilation | Use feature flags for minimal builds |

**Priority Recommendations**:

1. **P0 (High Value, Low Effort)**: Implement Shimmy-style auto-discovery
2. **P1 (High Value, Medium Effort)**: Universal backend adapter pattern
3. **P2 (Medium Value, Medium Effort)**: Response caching with LRU eviction

---

## 2. vLLM Analysis

### 2.1 Architecture Overview

**Location**: [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Core Philosophy**: PagedAttention + Continuous Batching for maximum throughput

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Engine (Python)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   Scheduler  │  │ PagedAttention│  │   Block Manager  │ │
│  │              │  │   (KV Cache)  │  │                  │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
│         │                 │                    │            │
│  ┌──────▼─────────────────▼────────────────────▼───────┐  │
│  │              Attention Backend Selector              │  │
│  │  (FlashAttention, FlashInfer, Torch, xFormers, ...) │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                   │
│  ┌──────▼───────────────────────────────────────────────┐  │
│  │          Model Executor (CUDA/ROCm/TPU/CPU)          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Features

#### PagedAttention (Core Innovation)

**Source**: [vLLM Paper](https://arxiv.org/abs/2309.06180)

**Concept**: Treat KV cache like virtual memory - allocate in fixed-size blocks

```
Traditional KV Cache:         PagedAttention:
┌────────────────────┐        ┌────────────────────┐
│ [████████████████] │        │ [████] [████] [██] │
│ Fixed per sequence │        │ Shared block pool  │
│ Fragmented memory  │        │ Zero fragmentation  │
└────────────────────┘        └────────────────────┘
```

**Benefits**:
- **3x memory efficiency** vs traditional caching
- Dynamic sharing of KV blocks across requests
- Near-zero memory fragmentation
- O(1) allocation/deallocation

#### Continuous Batching

**Source**: vLLM scheduler implementation

**Before (Static Batching)**:
```
Batch 1: [====================]  ████████████████████  (wait)
Batch 2:                     [====================]  ████████████████████
         └─ All requests must finish together
```

**After (Continuous Batching)**:
```
Time: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
      ┌─┐ ┌┐ ┌─────┐ ┌──┐ ┌─┐ ┌────┐ ┌──┐
Req1: │█│ ││ │     │ │  │ │█  │ │   │
      └─┘ └┘ └─────┘ └──┘ └─┘ └────┘ └──┘
      ┌───┐ ┌───────┐ ┌──┐
Req2: │   │ │       │ │  │
      └───┘ └───────┘ └──┘
      ┌────┐ ┌─────┐ ┌──────┐
Req3: │    │ │     │ │      │
      └────┘ └─────┘ └──────┘
             └─ Requests enter/exit continuously
```

**Throughput Gains**: **23x improvement** over static batching (Anyscale benchmark)

#### Attention Backend Selector (`vllm/attention/selector.py`)

**Platform-Specific Priority**:

```python
@cache
def _cached_get_attn_backend(backend, attn_selector_config):
    # Platform-aware backend selection
    attention_cls = current_platform.get_attn_backend_cls(
        backend, attn_selector_config
    )
    # Adjust KV cache layout if backend requires
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        set_kv_cache_layout(required_layout)
    return backend
```

**Available Backends**:
- **FlashAttention**: Optimized CUDA kernels
- **FlashInfer**: High-performance with advanced features
- **Torch**: PyTorch native attention
- **xFormers**: Memory-efficient attention
- **Aiter**: AMD-optimized (ROCm support)

#### V1 Engine Architecture (`vllm/v1/engine/llm_engine.py`)

**Key Components**:
- **InputProcessor**: Tokenization and input validation
- **EngineCoreClient**: Decoupled engine core interface
- **OutputProcessor**: Response formatting and streaming
- **StatLoggerManager**: Performance metrics and logging

**Multi-Process Support**:
```python
if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
    logger.debug("Enabling multiprocessing for LLMEngine.")
    enable_multiprocessing = True
```

### 2.3 ROCm Support

**Docker Images**:
```bash
docker pull rocm/vllm-dev:nightly
```

**Requirements Files**:
- `requirements/rocm.txt` - ROCm runtime dependencies
- `requirements/rocm-build.txt` - ROCm build dependencies
- `requirements/rocm-test.txt` - ROCm testing dependencies

**Key Contributors**:
- **Andreas Karatzas** (AMD): ROCm support and CI infrastructure

### 2.4 Strategic Learnings for ROCmForge

| Feature | vLLM Approach | ROCmForge Opportunity |
|---------|---------------|----------------------|
| **PagedAttention** | Block-based KV cache | Implement for AMD GPUs (Phase 12) |
| **Continuous Batching** | Dynamic request scheduling | Already have scheduler, enhance it |
| **Backend Selection** | Platform-aware registry | Create similar attention backend registry |
| **Speculative Decoding** | Draft model acceleration | Add for 2x speedup |
| **Quantization** | GPTQ, AWQ, FP8 | MXFP4/MXFP6 already implemented |
| **Prefix Caching** | Shared prompt caching | Implement for common prompts |

**Performance Comparison**:

| Metric | Traditional | vLLM | ROCmForge Current | Target |
|--------|-------------|------|------------------|--------|
| Memory Efficiency | 1x | 3x | 1x | 3x |
| Throughput (req/s) | 1x | 23x | ~2x | 10x+ |
| KV Cache Fragmentation | High | None | Medium | Low |
| Support for Long Context | Limited | 100K+ | 4K | 32K+ |

**Priority Recommendations**:

1. **P0 (Maximum Impact)**: Implement PagedAttention for AMD GPUs
2. **P0 (Maximum Impact)**: Enable continuous batching in scheduler
3. **P1 (High Impact)**: Create attention backend registry
4. **P2 (Medium Impact)**: Add prefix caching for shared prompts
5. **P3 (Nice to Have)**: Speculative decoding support

---

## 3. Rebellions ATOM Analysis

### 3.1 Architecture Overview

**Clarification**: ATOM is **not** an AMD project. It is a Korean NPU (Neural Processing Unit) by Rebellions.

**White Paper**: [ATOM™ Architecture White Paper](https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf)

**Core Philosophy**: Hardware-software co-design for inference optimization

```
┌─────────────────────────────────────────────────────────────┐
│                   ATOM NPU (Samsung 5nm)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │              8 Neural Engines (2 Clusters)           │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │  │
│  │  │ Task   │ │ Compute│ │Scratch ││ Task   │      │  │
│  │  │Manager │ │  Units │ │  Pad   ││Manager │ ... │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘      │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                        │                          │
│  ┌──────▼────────────────────────▼──────────────────────┐  │
│  │           Network-on-Chip (NoC) Fabric                │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                   │
│  ┌──────▼───────────────────────────────────────────────┐  │
│  │     64MB SRAM + GDDR6 Memory Interface                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **Process** | Samsung 5nm |
| **Neural Engines** | 8 engines in 2 clusters |
| **Compute** | 32 TFLOPS FP16 |
| **On-Chip Memory** | 64MB SRAM |
| **Memory Interface** | GDDR6 + PCIe 5.0 |
| **Interconnect** | Network-on-Chip (NoC) |

### 3.3 Software Stack (SqueezeBits)

**Components**:
- **Runtime**: PyTorch-native execution environment
- **Compiler**: Bridge between models and NPU
- **Profiler**: Performance analysis tools
- **Optimization**: Support for continuous/dynamic batching

**Key Features**:
- Dynamic batching support
- Variable-length sequence handling
- Optimized for LLM inference patterns
- 35-70% power reduction vs GPUs

### 3.4 Strategic Learnings for ROCmForge

| Aspect | ATOM Approach | ROCmForge Relevance |
|--------|---------------|---------------------|
| **Hardware-Software Co-Design** | NPU optimized for inference | Focus on ROCm-specific optimizations |
| **On-Chip Memory** | 64MB SRAM for KV cache | Use HIP shared memory for hot data |
| **Network-on-Chip** | Low-latency interconnect | Optimize kernel launch overhead |
| **Power Efficiency** | 35-70% vs GPU | Profile and optimize power usage |

**Relevance Level**: **LOW** - ATOM is a competing hardware platform, not directly applicable to AMD GPU development.

**Key Takeaway**: Hardware-software co-design principles are valuable, but ROCmForge should focus on AMD GPU-specific optimizations (wave32, LDS optimization, etc.).

---

## 4. Comparative Analysis

### 4.1 Feature Matrix

| Feature | Shimmy | vLLM | ATOM | ROCmForge | Priority |
|---------|--------|------|------|-----------|----------|
| **Auto-Discovery** | ✅ Comprehensive | ❌ Manual | ✅ Tooling | ⚠️ Basic | P1 |
| **PagedAttention** | ❌ | ✅ Core | ✅ Hardware | ❌ | **P0** |
| **Continuous Batching** | ❌ | ✅ Core | ✅ Software | ⚠️ Basic | **P0** |
| **Multi-Backend** | ✅ Universal | ❌ PyTorch | N/A | ❌ | P1 |
| **Response Caching** | ✅ LRU+TTL | ✅ Prefix | ❌ | ❌ | P2 |
| **ROCm Support** | ⚠️ Via llama.cpp | ✅ Official | ❌ | ✅ Native | — |
| **Quantization** | Via backend | ✅ GPTQ/AWQ/FP8 | ✅ Hardware | ✅ MXFP4/MXFP6 | — |
| **OpenAI API** | ✅ Compatible | ✅ Compatible | Via SDK | ✅ Compatible | — |
| **Language** | Rust | Python | C++/Python | Rust | — |

### 4.2 Performance Comparison

| Metric | Shimmy | vLLM | ROCmForge | Gap |
|--------|--------|------|-----------|-----|
| **Throughput** | 2-3x (via llama.cpp) | 23x | ~2x | 10x vs vLLM |
| **Memory Efficiency** | Standard | 3x | Standard | 3x vs vLLM |
| **Startup Time** | <1s | 2-5s | <1s | ✅ Competitive |
| **Model Loading** | 2x faster | Standard | Standard | ✅ Competitive |
| **Cold Start** | Fast | Slow | Fast | ✅ Competitive |

### 4.3 Architecture Patterns

#### Shimmy: Universal Backend Pattern
```rust
pub enum Backend {
    LlamaGGUF,
    HuggingFace,
    SafeTensors,
    MLX,
}
```

**Adopt for**: ROCmForge backend abstraction (ROCm/HIP/CPU)

#### vLLM: Pluggable Attention Backend
```python
class AttentionBackend(ABC):
    @abstractmethod
    def get_name(self) -> str: ...
    @abstractmethod
    def get_required_kv_cache_layout(self) -> Optional[str]: ...
```

**Adopt for**: ROCmForge attention selection (Flash/MQ/GQA)

#### ATOM: Hardware-Software Co-Design
```c
// Neural Engine with local scratchpad
struct NeuralEngine {
    TaskManager* task_mgr;
    ComputeUnit* compute_units;
    ScratchPadMemory* local_mem;
};
```

**Adopt for**: ROCmForge HIP kernel optimization strategies

---

## 5. Strategic Recommendations for ROCmForge

### 5.1 Immediate Priorities (Phase 12)

#### P0-1: Implement PagedAttention
**Effort**: 2-3 weeks | **Impact**: 3x memory efficiency

**Implementation**:
```rust
// src/kv_cache/paged_attention.rs
pub struct PagedKvCache {
    block_size: usize,        // Fixed block size (e.g., 16 tokens)
    blocks: Vec<KvBlock>,     // Pre-allocated block pool
    block_table: HashMap<u64, Vec<BlockId>>,  // Seq -> blocks
    free_blocks: Vec<BlockId>, // Free list
}

pub struct KvBlock {
    keys: DeviceTensor<f32>,
    values: DeviceTensor<f32>,
    ref_count: Arc<AtomicUsize>,
}
```

**Benefits**:
- Zero fragmentation
- O(1) allocation
- Block sharing across sequences
- Support for 100K+ context

#### P0-2: Enable True Continuous Batching
**Effort**: 1-2 weeks | **Impact**: 10-23x throughput

**Current State**: `src/scheduler/scheduler.rs` has basic batching

**Enhancement**:
```rust
// src/scheduler/continuous_batch.rs
pub struct ContinuousBatchScheduler {
    active_requests: HashMap<RequestId, RequestState>,
    pending_queue: VecDeque<RequestId>,

    fn schedule_iteration(&mut self) -> Batch {
        // 1. Remove completed requests
        // 2. Add new requests from pending
        // 3. Re-sort by position
        // 4. Create batch for next iteration
    }
}
```

### 5.2 Short-Term Priorities (Phase 13)

#### P1-1: Attention Backend Registry
**Effort**: 1 week | **Impact**: Flexibility

```rust
// src/attention/registry.rs
pub trait AttentionBackend: Send + Sync {
    fn name(&self) -> &str;
    fn supports(&self, config: &AttentionConfig) -> bool;
    fn execute(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor>;
}

pub struct BackendRegistry {
    backends: Vec<Box<dyn AttentionBackend>>,
}

impl BackendRegistry {
    pub fn select_best(&self, config: &AttentionConfig) -> &dyn AttentionBackend {
        // Platform-aware selection
    }
}
```

#### P1-2: Universal Backend Adapter (Shimmy Pattern)
**Effort**: 2 weeks | **Impact**: Ecosystem compatibility

```rust
// src/backend/adapter.rs
pub enum InferenceBackend {
    ROCm-native,
    HIP-compatibility,
    CPU-fallback,
    ExternalLlamaCpp,
}

pub struct BackendAdapter {
    backend: InferenceBackend,
}

impl BackendAdapter {
    pub fn auto_select(model_path: &Path) -> InferenceBackend {
        // 1. Check file extension
        // 2. Check platform capabilities
        // 3. Select best backend
    }
}
```

#### P1-3: Model Auto-Discovery
**Effort**: 1 week | **Impact**: UX improvement

```rust
// src/loader/discovery.rs
pub fn discover_models() -> Vec<ModelPath> {
    let search_paths = vec![
        "./models",
        "~/.cache/huggingface/hub",
        "~/.ollama/models",
        "~/.lmstudio/models",
        std::env::var("ROCMFORGE_MODEL_PATH").ok(),
    ];

    search_paths.iter()
        .flat_map(|p| walkdir(p))
        .filter(|f| is_llm_model(f))
        .collect()
}
```

### 5.3 Medium-Term Priorities (Phase 14+)

#### P2-1: Prefix Caching
**Effort**: 2 weeks | **Impact**: 2-5x for repeated prompts

#### P2-2: Response Caching (Shimmy LRU)
**Effort**: 1 week | **Impact**: Reduced API latency

#### P2-3: Speculative Decoding
**Effort**: 3 weeks | **Impact**: 2x generation speed

#### P2-4: Multi-GPU Tensor Parallelism
**Effort**: 4 weeks | **Impact**: Scale to larger models

### 5.4 AMD-Specific Optimizations

Learn from ATOM's hardware-software co-design:

| Optimization | Description | Effort |
|--------------|-------------|--------|
| **Wave32 Optimization** | Align workgroups to AMD wave size | Low |
| **LDS Maximization** | Use shared memory for KV cache reuse | Medium |
| **RDNA3 Tuning** | Optimize for gfx1100+ architecture | Medium |
| **CDNA Support** | Add MI300/MI350 optimizations | High |

---

## 6. Implementation Roadmap

### Phase 12: PagedAttention + Continuous Batching (4-5 weeks)
- Week 1-2: Implement PagedKvCache
- Week 3-4: Enhance scheduler for continuous batching
- Week 5: Testing and benchmarking

**Expected Gains**: 10-23x throughput, 3x memory efficiency

### Phase 13: Ecosystem Compatibility (3-4 weeks)
- Week 1: Attention backend registry
- Week 2: Universal backend adapter
- Week 3: Model auto-discovery
- Week 4: Testing and documentation

**Expected Gains**: Ecosystem parity with Shimmy/vLLM

### Phase 14: Advanced Features (6-8 weeks)
- Week 1-2: Prefix caching
- Week 3: Response caching
- Week 4-6: Speculative decoding
- Week 7-8: Multi-GPU support

**Expected Gains**: Production-ready feature parity

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **PagedAttention Complexity** | Medium | High | Incremental implementation, testing |
| **ROCm Limitations** | Low | High | Workarounds, fallback paths |
| **Performance Gap** | Low | Medium | Benchmark-driven optimization |
| **Ecosystem Fragmentation** | High | Medium | Universal backend adapter |

---

## 8. Sources

### Shimmy
- Source code: `/home/feanor/Projects/shimmy`
- Key files:
  - `src/model_registry.rs:16-123`
  - `src/auto_discovery.rs:54-168`
  - `src/engine/adapter.rs:61-181`
  - `src/cache/response_cache.rs:140-282`

### vLLM
- Repository: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- Paper: [Efficient Memory Management with PagedAttention](https://arxiv.org/abs/2309.06180)
- Documentation:
  - [Attention Backend Selection](https://zread.ai/vllm-project/vllm/14-attention-backend-selection)
  - [Attention Kernels and Optimization](https://zread.ai/vllm-project/vllm/16-attention-kernels-and-optimization)
  - [ROCm Support](https://zread.ai/vllm-project/vllm/2-quick-start)

### Rebellions ATOM
- White Paper: [ATOM™ Architecture White Paper](https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf)
- Architecture Overview: [Finding the Sweet Spot for GenAI](https://rebellions.ai/atom-architecture-finding-the-sweet-spot-for-genai/)
- ATOM-MAX: [Boosted Performance for Large-Scale Inference](https://rebellions.ai/atom-max-boosted-performance-for-large-scale-inference/)
- Documentation: [RBLN NPU Architecture](https://docs.rbln.ai/v0.8.3/software/profiler/architecture.html)

### Additional References
- [Anyscale: Continuous Batching LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) (23x throughput gains)
- [AMD CDNA 4 Architecture White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [ROCm 7.0 Release Notes](https://rocm.docs.amd.com/en/docs-7.0.0/about/release-notes.html)

---

**Next Steps**:
1. Review and prioritize recommendations
2. Update `docs/PLAN.md` with Phase 12-14 tasks
3. Create architectural decision records for PagedAttention and continuous batching
4. Begin P0-1 implementation

**Document Owner**: ROCmForge Project
**Last Updated**: 2026-01-08
**Review Cycle**: Quarterly
