# ROCmForge API Documentation

> **Version**: 0.1.0
> **Last Updated**: 2026-01-08
> **Source**: Analysis of src/ directory

This document provides comprehensive API reference for ROCmForge, an AMD GPU inference engine for Large Language Models.

---

## Table of Contents

1. [Core Engine API](#core-engine-api)
2. [Backend API](#backend-api)
3. [Model Loading API](#model-loading-api)
4. [Attention API](#attention-api)
5. [KV Cache API](#kv-cache-api)
6. [Scheduler API](#scheduler-api)
7. [Sampler API](#sampler-api)
8. [HTTP Server API](#http-server-api)
9. [Tensor API](#tensor-api)
10. [Model Execution API](#model-execution-api)
11. [Error Types](#error-types)

---

## Core Engine API

**Module**: `src/engine.rs`

### `InferenceEngine`

Main inference engine orchestrating model loading, request scheduling, and token generation.

**Source**: `src/engine.rs:67`

```rust
pub struct InferenceEngine {
    /* private fields */
}
```

#### Methods

##### `new`

```rust
pub fn new(config: EngineConfig) -> EngineResult<Self>
```

Creates a new inference engine instance.

**Parameters**:
- `config`: Engine configuration specifying batch sizes, cache parameters, etc.

**Returns**:
- `EngineResult<InferenceEngine>`: Engine instance or error

**Errors**:
- `EngineError::BackendFailed`: HIP backend initialization failed
- `EngineError::CacheFailed`: KV cache initialization failed

**Example**:
```rust
use rocmforge::InferenceEngine;
use rocmforge::engine::EngineConfig;

let config = EngineConfig::default();
let engine = InferenceEngine::new(config)?;
```

---

##### `load_gguf_model`

```rust
pub async fn load_gguf_model<P: AsRef<std::path::Path>>(
    &mut self,
    path: P,
) -> EngineResult<()>
```

Loads a GGUF model from disk.

**Parameters**:
- `path`: Path to GGUF model file

**Returns**:
- `EngineResult<()>`: Success or error

**Errors**:
- `EngineError::ModelLoadFailed`: Model file could not be loaded

**Source**: `src/engine.rs:146`

---

##### `load_onnx_model`

```rust
pub async fn load_onnx_model<P: AsRef<std::path::Path>>(
    &self,
    path: P
) -> EngineResult<()>
```

Loads an ONNX model from disk.

**Source**: `src/engine.rs:176`

---

##### `start`

```rust
pub async fn start(&self) -> EngineResult<()>
```

Starts the inference engine and begins processing requests.

**Source**: `src/engine.rs:188`

---

##### `submit_request`

```rust
pub async fn submit_request(
    &self,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> EngineResult<u32>
```

Submits a generation request to the engine.

**Parameters**:
- `prompt_tokens`: Token IDs forming the prompt
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (0.0 to 2.0)
- `top_k`: Top-k sampling parameter
- `top_p`: Nucleus sampling parameter (0.0 to 1.0)

**Returns**:
- `EngineResult<u32>`: Request ID for tracking

**Source**: `src/engine.rs:256`

---

##### `get_request_status`

```rust
pub async fn get_request_status(
    &self,
    request_id: u32,
) -> EngineResult<Option<GenerationRequest>>
```

Queries the status of a submitted request.

**Source**: `src/engine.rs:285`

---

##### `cancel_request`

```rust
pub async fn cancel_request(&self, request_id: u32) -> EngineResult<()>
```

Cancels a pending or processing request.

**Source**: `src/engine.rs:297`

---

##### `get_engine_stats`

```rust
pub async fn get_engine_stats(&self) -> EngineStats
```

Returns current engine statistics.

**Returns**:
- `EngineStats`: Statistics including queue depth, cache usage, etc.

**Source**: `src/engine.rs:628`

---

### `EngineConfig`

Configuration for the inference engine.

**Source**: `src/engine.rs:35`

```rust
pub struct EngineConfig {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub cache_page_size: usize,
    pub max_cache_pages: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub batch_timeout: Duration,
}
```

**Default Values**:
```rust
EngineConfig {
    max_batch_size: 32,
    max_sequence_length: 4096,
    cache_page_size: 16,
    max_cache_pages: 1000,
    num_heads: 32,
    head_dim: 128,
    num_layers: 24,
    batch_timeout: Duration::from_millis(50),
}
```

---

### `EngineError`

Errors that can occur during engine operation.

**Source**: `src/engine.rs:17`

```rust
pub enum EngineError {
    BackendFailed(String),
    ModelLoadFailed(String),
    CacheFailed(String),
    SchedulerError(String),
    InferenceFailed(String),
    InvalidConfig(String),
}
```

---

### `EngineStats`

Engine runtime statistics.

**Source**: `src/engine.rs:651`

```rust
pub struct EngineStats {
    pub is_running: bool,
    pub scheduler_stats: crate::scheduler::QueueStats,
    pub cache_stats: crate::kv_cache::CacheStats,
    pub model_loaded: bool,
}
```

---

## Backend API

**Module**: `src/backend/hip_backend.rs`

### `HipBackend`

Main HIP/ROCm backend for GPU operations.

**Source**: `src/backend/hip_backend.rs:524` (approximate location)

```rust
pub struct HipBackend {
    /* private fields */
}
```

#### Methods

##### `new`

```rust
pub fn new() -> HipResult<Arc<HipBackend>>
```

Creates a new HIP backend instance, initializes ROCm driver.

**Returns**:
- `HipResult<Arc<HipBackend>>`: Arc-wrapped backend or error

**Errors**:
- `HipError::InitializationFailed`: ROCm driver not found or initialization failed
- `HipError::DeviceNotFound`: No AMD GPU detected

---

##### `allocate_buffer`

```rust
pub fn allocate_buffer(&self, size: usize) -> HipResult<HipBuffer>
```

Allocates GPU memory buffer.

**Parameters**:
- `size`: Size in bytes

**Returns**:
- `HipResult<HipBuffer>`: GPU buffer handle

---

##### `get_device_count`

```rust
pub fn get_device_count(&self) -> i32
```

Returns number of available AMD GPUs.

---

##### `get_device_properties`

```rust
pub fn get_device_properties(&self, device_id: i32) -> HipResult<HipDeviceProp>
```

Gets properties for a specific GPU.

---

### `HipBuffer`

GPU memory buffer with Arc-based reference counting.

**Source**: `src/backend/hip_backend.rs:221`

```rust
#[derive(Debug, Clone)]
pub struct HipBuffer {
    /* private fields */
}
```

#### Methods

##### `new`

```rust
pub fn new(size: usize) -> HipResult<Self>
```

Allocates new GPU memory buffer.

**Parameters**:
- `size`: Buffer size in bytes

---

##### `size`

```rust
pub fn size(&self) -> usize
```

Returns buffer size in bytes.

---

##### `copy_from_host`

```rust
pub fn copy_from_host<T>(&self, data: &[T]) -> HipResult<()>
```

Copies data from host memory to GPU.

**Type Parameters**:
- `T`: Element type (must be POD)

**Parameters**:
- `data`: Host data slice

**Errors**:
- `HipError::MemoryCopyFailed`: Copy operation failed
- `HipError::MemoryAllocationFailed`: Source data too large

---

##### `copy_to_host`

```rust
pub fn copy_to_host<T>(&self, data: &mut [T]) -> HipResult<()>
```

Copies data from GPU to host memory.

---

##### `copy_from_buffer`

```rust
pub fn copy_from_buffer(&self, src: &HipBuffer) -> HipResult<()>
```

Copies from another GPU buffer (device-to-device).

---

##### `sub_buffer_view`

```rust
pub fn sub_buffer_view(&self, offset: usize, size: usize) -> HipResult<Self>
```

Creates a view into a sub-region of the buffer without allocating new memory.

**Parameters**:
- `offset`: Byte offset from buffer start
- `size`: Size of sub-region in bytes

**Returns**:
- New HipBuffer sharing the same underlying GPU memory

**Example**:
```rust
let large_buffer = HipBuffer::new(1024 * 1024)?;
let view = large_buffer.sub_buffer_view(4096, 1024)?;
// view now points to bytes 4096-5120 of large_buffer
```

---

##### `as_ptr`

```rust
pub fn as_ptr(&self) -> *mut c_void
```

Returns raw pointer to GPU memory (for FFI).

---

### `HipDeviceProp`

GPU device properties.

**Source**: `src/backend/hip_backend.rs:66`

```rust
#[repr(C)]
#[derive(Debug, Clone)]
pub struct HipDeviceProp {
    /* private buffer of 1472 bytes */
}
```

#### Methods

##### `name`

```rust
pub fn name(&self) -> String
```

Returns GPU device name (e.g., "AMD Radeon RX 7900 XT").

**Source**: `src/backend/hip_backend.rs:85`

---

##### `total_global_mem`

```rust
pub fn total_global_mem(&self) -> u64
```

Returns total GPU memory in bytes.

**Source**: `src/backend/hip_backend.rs:92`

---

##### `multi_processor_count`

```rust
pub fn multi_processor_count(&self) -> i32
```

Returns number of compute units (CUs).

**Source**: `src/backend/hip_backend.rs:99`

---

### `HipError`

HIP/ROCm error types.

**Source**: `src/backend/hip_backend.rs:118`

```rust
pub enum HipError {
    InitializationFailed(String),
    KernelLoadFailed(String),
    MemoryAllocationFailed(String),
    MemoryCopyFailed(String),
    MemoryQueryFailed(String),
    KernelLaunchFailed(String),
    DeviceNotFound,
    DeviceError(String),
    GenericError(String),
}
```

---

### `DeviceTensor`

Tensor stored in GPU memory with shape information.

**Module**: `src/backend/hip_backend.rs`

```rust
pub struct DeviceTensor {
    /* private fields */
}
```

#### Methods

##### `empty`

```rust
pub fn empty(backend: &HipBackend, shape: TensorShape) -> HipResult<Self>
```

Creates uninitialized GPU tensor.

---

##### `from_host_vec`

```rust
pub fn from_host_vec(
    backend: &HipBackend,
    data: Vec<f32>,
    shape: TensorShape,
) -> HipResult<Self>
```

Creates GPU tensor from host data.

---

##### `to_host_vec`

```rust
pub fn to_host_vec(&self) -> HipResult<Vec<f32>>
```

Copies tensor data to host.

---

##### `shape`

```rust
pub fn shape(&self) -> &TensorShape
```

Returns tensor shape.

---

##### `buffer`

```rust
pub fn buffer(&self) -> &HipBuffer
```

Returns underlying GPU buffer.

---

## Model Loading API

**Module**: `src/loader/gguf.rs`

### `GgufLoader`

GGUF (GPT-Generated Unified Format) model loader.

**Source**: `src/loader/gguf.rs:547`

```rust
pub struct GgufLoader {
    /* private fields */
}
```

#### Methods

##### `load`

```rust
pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error>
```

Loads a GGUF model file.

**Parameters**:
- `path`: Path to .gguf file

**Returns**:
- Loaded model with metadata and tensors

**Supported Features**:
- Q4_0, Q4_1, Q5_0, Q5_1 quantization
- FP16, FP32 tensors
- Architecture auto-detection (Qwen2, LLaMA, Mistral)
- MXFP quantization (MXFP4, MXFP6 per OCP MX Spec v1.0)

---

##### `get_tensor`

```rust
pub fn get_tensor(&self, name: &str) -> Option<&GgufTensor>
```

Retrieves a tensor by name.

---

##### `metadata`

```rust
pub fn metadata(&self) -> &GgufMetadata
```

Returns model metadata.

---

### `GgufTensor`

Single tensor in GGUF file.

**Source**: `src/loader/gguf.rs:486`

```rust
pub struct GgufTensor {
    pub name: String,
    pub shape: TensorShape,
    pub tensor_type: GgufTensorType,
    pub offset: u64,
    /* private fields */
}
```

---

### `GgufMetadata`

GGUF file metadata.

**Source**: `src/loader/gguf.rs:449`

```rust
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u32,
    pub kv_pairs: HashMap<String, GgufMetadataValue>,
    /* private fields */
}
```

---

### `GgufTensorType`

Supported GGUF tensor types.

**Source**: `src/loader/gguf.rs:367`

```rust
pub enum GgufTensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    /* ... other types */
}
```

---

### `TensorShape`

Multi-dimensional tensor shape.

**Module**: `src/loader/mmap_loader.rs`

```rust
pub struct TensorShape {
    pub dims: Vec<usize>,
}
```

#### Methods

##### `from_dims`

```rust
pub fn from_dims(dims: &[usize]) -> Self
```

Creates shape from dimension array.

---

##### `num_elements`

```rust
pub fn num_elements(&self) -> usize
```

Returns total number of elements.

---

##### `size_bytes`

```rust
pub fn size_bytes(&self, type_size: usize) -> usize
```

Returns size in bytes.

---

## Attention API

**Module**: `src/attention/mod.rs`

### `Attention`

Scaled dot-product attention mechanism.

**Source**: `src/attention/mod.rs:85`

```rust
pub struct Attention {
    pub dim: usize,
    pub backend: AttentionBackend,
}
```

#### Methods

##### `new`

```rust
pub fn new(dim: usize) -> Self
```

Creates attention layer with CPU backend.

---

##### `with_backend`

```rust
pub fn with_backend(dim: usize, backend: AttentionBackend) -> Self
```

Creates attention layer with specified backend.

---

##### `forward`

```rust
pub fn forward(
    &self,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    dropout: Option<f32>,
) -> AttentionResult<Vec<f32>>
```

Computes attention with host memory inputs.

**Parameters**:
- `q`: Query tensor [seq_len, dim]
- `k`: Key tensor [seq_len, dim]
- `v`: Value tensor [seq_len, dim]
- `mask`: Optional attention mask
- `dropout`: Optional dropout rate

**Returns**:
- Attention output [seq_len, dim]

---

##### `forward_device`

```rust
#[cfg(feature = "rocm")]
pub fn forward_device(
    &self,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    mask: Option<&DeviceTensor>,
    dropout: Option<f32>,
) -> AttentionResult<DeviceTensor>
```

Zero-copy GPU attention computation.

**Features**:
- FlashAttention algorithm (divide-and-conquer)
- Causal masking support
- 2-5x speedup vs CPU

**Source**: `src/attention/mod.rs:135`

---

### `AttentionBackend`

Execution backend for attention operations.

**Source**: `src/attention/backend.rs`

```rust
pub enum AttentionBackend {
    Cpu,
    #[cfg(feature = "rocm")]
    Gpu,
}
```

---

### `AttentionError`

Attention operation errors.

**Source**: `src/attention/mod.rs:66`

```rust
pub enum AttentionError {
    ShapeMismatch(String),
    DimensionError(String),
    MemoryAllocation(String),
    MemoryCopy(String),
    GpuOperation(String),
    HandleCreation(String),
    Synchronization(String),
}
```

---

## KV Cache API

**Module**: `src/kv_cache/kv_cache.rs`

### `KvCache`

Paged key-value cache for efficient autoregressive generation.

**Source**: `src/kv_cache/kv_cache.rs:139`

```rust
pub struct KvCache {
    /* private fields */
}
```

#### Methods

##### `new`

```rust
pub fn new(
    config: CacheConfig,
    backend: Arc<HipBackend>,
) -> KvCacheResult<Self>
```

Creates new paged KV cache.

---

##### `append_token`

```rust
pub fn append_token(
    &mut self,
    sequence_id: u32,
    token: u32,
) -> KvCacheResult<()>
```

Appends a token to the cache.

---

##### `remove_sequence`

```rust
pub fn remove_sequence(&mut self, sequence_id: u32) -> KvCacheResult<()>
```

Removes a sequence and frees its pages.

---

##### `get_cache_stats`

```rust
pub fn get_cache_stats(&self) -> CacheStats
```

Returns cache statistics.

---

### `CacheConfig`

KV cache configuration.

**Source**: `src/kv_cache/kv_cache.rs:25`

```rust
pub struct CacheConfig {
    pub page_size: usize,
    pub max_pages: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
}
```

#### Methods

##### `new`

```rust
pub fn new(
    page_size: usize,
    max_pages: usize,
    num_heads: usize,
    head_dim: usize,
    num_layers: usize,
) -> KvCacheResult<Self>
```

Validates and creates cache configuration.

**Validation**:
- All parameters must be non-zero

---

### `CachePage`

Single cache page storing K/V tensors.

**Source**: `src/kv_cache/kv_cache.rs:56`

```rust
pub struct CachePage {
    pub page_id: u32,
    pub sequence_id: u32,
    pub tokens: Vec<u32>,
    pub key_buffer: HipBuffer,
    pub value_buffer: HipBuffer,
    pub is_free: bool,
}
```

---

### `CacheStats`

Cache usage statistics.

```rust
pub struct CacheStats {
    pub total_pages: usize,
    pub free_pages: usize,
    pub active_sequences: usize,
    pub total_tokens: usize,
}
```

---

### `KvCacheError`

KV cache operation errors.

**Source**: `src/kv_cache/kv_cache.rs:9`

```rust
pub enum KvCacheError {
    CapacityExceeded,
    InvalidSequenceId(u32),
    PageNotFound(u32),
    GpuError(HipError),
    InvalidConfiguration,
}
```

---

## Scheduler API

**Module**: `src/scheduler/scheduler.rs`

### `Scheduler`

Continuous batching scheduler for request management.

```rust
pub struct Scheduler {
    /* private fields */
}
```

#### Methods

##### `new`

```rust
pub fn new(config: SchedulerConfig) -> Self
```

Creates new scheduler.

---

##### `submit_request`

```rust
pub fn submit_request(
    &mut self,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<u32, String>
```

Submits generation request and returns request ID.

---

##### `create_batch`

```rust
pub fn create_batch(&mut self) -> Result<RequestBatch, String>
```

Creates a batch from pending requests.

---

##### `get_request`

```rust
pub fn get_request(&self, request_id: u32) -> Result<&GenerationRequest, String>
```

Gets request by ID.

---

##### `cancel_request`

```rust
pub fn cancel_request(&mut self, request_id: u32) -> Result<(), String>
```

Cancels a request.

---

### `SchedulerConfig`

Scheduler configuration.

```rust
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_queue_size: usize,
    pub batch_timeout: Duration,
    pub max_sequence_length: usize,
}
```

---

### `GenerationRequest`

Single generation request tracked by scheduler.

```rust
pub struct GenerationRequest {
    pub request_id: u32,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub state: RequestState,
}
```

---

### `RequestState`

Request processing state.

```rust
pub enum RequestState {
    Pending,
    Processing,
    Completed,
    Cancelled,
    Failed,
}
```

---

### `RequestBatch`

Batch of requests for parallel processing.

```rust
pub struct RequestBatch {
    pub batch_id: u32,
    pub requests: Vec<GenerationRequest>,
}
```

---

## Sampler API

**Module**: `src/sampler/sampler.rs`

### `Sampler`

Token sampling with various strategies.

```rust
pub struct Sampler {
    /* private fields */
}
```

#### Methods

##### `new`

```rust
pub fn new(config: SamplingConfig) -> Self
```

Creates sampler with configuration.

---

##### `sample`

```rust
pub fn sample(&self, logits: &[f32]) -> Result<u32, String>
```

Samples next token from logits.

---

##### `sample_with_history`

```rust
pub fn sample_with_history(
    &self,
    logits: &[f32],
    history: &[u32],
) -> Result<u32, String>
```

Samples with repetition penalty based on history.

---

### `SamplingConfig`

Sampling parameters.

```rust
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}
```

**Default**:
```rust
SamplingConfig {
    temperature: 1.0,
    top_k: 50,
    top_p: 0.9,
    repetition_penalty: 1.0,
}
```

---

## HTTP Server API

**Module**: `src/http/server.rs`

### `InferenceServer`

HTTP/SSE server providing REST API for inference.

**Source**: `src/http/server.rs:128`

```rust
pub struct InferenceServer {
    /* private fields */
}
```

#### Methods

##### `new`

```rust
pub fn new(
    engine: Option<Arc<InferenceEngine>>,
    tokenizer: TokenizerAdapter,
) -> Self
```

Creates new HTTP server.

---

##### `run`

```rust
pub async fn run(self, addr: SocketAddr) -> Result<(), anyhow::Error>
```

Starts HTTP server on specified address.

---

### HTTP Endpoints

#### POST /v1/generate

Generate text (non-streaming).

**Request**:
```json
{
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9,
  "stream": false
}
```

**Response**:
```json
{
  "request_id": 1,
  "text": "Hello, world! This is...",
  "tokens": [1234, 5678, ...],
  "finished": true,
  "finish_reason": "length"
}
```

**Request Type**: `src/http/server.rs:57`
```rust
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}
```

**Response Type**: `src/http/server.rs:67`
```rust
pub struct GenerateResponse {
    pub request_id: u32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub finished: bool,
    pub finish_reason: Option<String>,
}
```

---

#### POST /v1/generate/stream

Generate text with Server-Sent Events streaming.

**Request**: Same as `/v1/generate` with `"stream": true`

**Response**: SSE stream of `TokenStream` events

**Stream Event Type**: `src/http/server.rs:76`
```rust
pub struct TokenStream {
    pub request_id: u32,
    pub token: u32,
    pub text: String,
    pub finished: bool,
    pub finish_reason: Option<String>,
}
```

---

#### GET /v1/status

Get server status and statistics.

**Response**:
```json
{
  "status": "running",
  "model_loaded": true,
  "pending_requests": 5,
  "cache_utilization": 0.75
}
```

---

### `ServerError`

HTTP server errors.

**Source**: `src/http/server.rs:25`

```rust
pub enum ServerError {
    InvalidRequest(String),
    GenerationFailed(String),
    RequestNotFound(u32),
    InternalError(String),
}
```

---

## Tensor API

**Module**: `src/tensor/mod.rs`

### `Tensor`

Basic CPU-side tensor structure.

**Source**: `src/tensor/mod.rs:11`

```rust
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
}
```

#### Methods

##### `new`

```rust
pub fn new(data: Vec<f32>) -> Self
```

Creates tensor from data.

---

##### `random`

```rust
pub fn random(size: usize) -> Self
```

Creates tensor with random values in [0, 1).

---

##### `random_seeded`

```rust
pub fn random_seeded(size: usize, seed: u64) -> Self
```

Creates reproducible random tensor.

---

##### `zeros`

```rust
pub fn zeros(size: usize) -> Self
```

Creates zero-filled tensor.

---

##### `len`

```rust
pub fn len(&self) -> usize
```

Returns number of elements.

---

##### `as_slice`

```rust
pub fn as_slice(&self) -> &[f32]
```

Returns immutable slice of data.

---

##### `as_mut_slice`

```rust
pub fn as_mut_slice(&mut self) -> &mut [f32]
```

Returns mutable slice of data.

---

## Model Execution API

**Module**: `src/model/execution_plan.rs`

### `ExecutionPlan`

Static execution plan for transformer models.

**Source**: `src/model/execution_plan.rs:49`

```rust
pub struct ExecutionPlan {
    /* private fields */
}
```

#### Methods

##### `new`

```rust
pub fn new(
    backend: &HipBackend,
    config: &ModelConfig,
) -> HipResult<Self>
```

Creates execution plan from configuration.

---

##### `layers`

```rust
pub fn layers(&self) -> &[LayerPlan]
```

Returns layer execution plans.

---

##### `embedding_weights`

```rust
pub fn embedding_weights(&self) -> &DeviceTensor
```

Returns token embedding weights.

---

##### `lm_head`

```rust
pub fn lm_head(&self) -> &DeviceTensor
```

Returns language model head weights.

---

### `LayerPlan`

Execution plan for single transformer layer.

**Source**: `src/model/execution_plan.rs:64`

```rust
pub struct LayerPlan {
    pub qkv_weight: DeviceTensor,
    pub qkv_bias: Option<DeviceTensor>,
    pub o_proj: DeviceTensor,
    pub mlp_gate_proj: DeviceTensor,
    pub mlp_up_proj: DeviceTensor,
    pub mlp_down_proj: DeviceTensor,
    pub mlp_fc1: DeviceTensor,
    pub mlp_fc1_bias: Option<DeviceTensor>,
    pub mlp_fc2: DeviceTensor,
    pub mlp_fc2_bias: Option<DeviceTensor>,
    pub norm1_weight: DeviceTensor,
    pub norm1_bias: Option<DeviceTensor>,
    pub norm2_weight: DeviceTensor,
    pub norm2_bias: Option<DeviceTensor>,
}
```

Contains all weight tensors for:
- QKV projection
- Output projection
- MLP layers (GLU-style)
- Layer normalization

---

### `ModelRuntime`

Runtime for model execution with loaded weights.

**Module**: `src/backend/hip_backend.rs`

```rust
pub struct ModelRuntime {
    /* private fields */
}
```

#### Methods

##### `load_from_gguf`

```rust
pub fn load_from_gguf(path: &str) -> Result<Self, anyhow::Error>
```

Loads model from GGUF file.

---

##### `decode_step`

```rust
pub fn decode_step(
    &mut self,
    embeddings: &DeviceTensor,
) -> Result<DeviceTensor, String>
```

Runs single decode step (returns logits).

---

##### `execution_plan`

```rust
pub fn execution_plan(&self) -> Option<&ExecutionPlan>
```

Returns execution plan if loaded.

---

### `Architecture`

Detected model architecture type.

**Source**: `src/model/execution_plan.rs:15`

```rust
pub enum Architecture {
    Qwen2,
    LLaMA,
    Mistral,
}
```

#### Methods

##### `layer_prefix`

```rust
pub fn layer_prefix(&self, layer_idx: usize) -> String
```

Returns tensor naming prefix for layer.

---

## Error Types Summary

### Type Aliases

Each module provides a `Result` type alias:

```rust
// Engine
pub type EngineResult<T> = Result<T, EngineError>;

// Backend
pub type HipResult<T> = Result<T, HipError>;

// Attention
pub type AttentionResult<T> = Result<T, AttentionError>;

// KV Cache
pub type KvCacheResult<T> = Result<T, KvCacheError>;
```

### Common Error Patterns

All error types implement:
- `std::fmt::Display`
- `std::fmt::Debug`
- `std::error::Error` (via `thiserror::Error`)

Most errors are wrapped from lower-level components:
- `EngineError` wraps `HipError` for GPU failures
- `KvCacheError` wraps `HipError` for memory operations
- `AttentionError` wraps both `HipError` and dimension errors

---

## Usage Examples

### Basic Generation

```rust
use rocmforge::{InferenceEngine, engine::EngineConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine
    let config = EngineConfig::default();
    let mut engine = InferenceEngine::new(config)?;

    // Load model
    engine.load_gguf_model("/path/to/model.gguf").await?;

    // Start engine
    engine.start().await?;

    // Submit request
    let prompt_tokens = vec![1234, 5678, 9012]; // Token IDs
    let request_id = engine.submit_request(
        prompt_tokens,
        100,    // max_tokens
        0.8,    // temperature
        50,     // top_k
        0.9,    // top_p
    ).await?;

    // Poll for completion
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if let Some(request) = engine.get_request_status(request_id).await? {
            if request.is_complete() {
                println!("Generated: {:?}", request.generated_tokens);
                break;
            }
        }
    }

    Ok(())
}
```

### HTTP Server

```rust
use rocmforge::{InferenceEngine, http::{InferenceServer, TokenizerAdapter}};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup
    let config = EngineConfig::default();
    let mut engine = InferenceEngine::new(config)?;
    engine.load_gguf_model("/path/to/model.gguf").await?;

    // Create server
    let tokenizer = TokenizerAdapter::from_gguf("/path/to/model.gguf")?;
    let server = InferenceServer::new(
        Some(Arc::new(engine)),
        tokenizer,
    );

    // Run server
    let addr: SocketAddr = "[::1]:8080".parse()?;
    server.run(addr).await?;

    Ok(())
}
```

### Direct Attention Usage

```rust
use rocmforge::Attention;

// CPU attention
let attention = Attention::new(128); // dim=128
let output = attention.forward(
    &query_tensor,
    &key_tensor,
    &value_tensor,
    None,  // no mask
    None,  // no dropout
)?;

// GPU attention (zero-copy)
#[cfg(feature = "rocm")]
{
    let gpu_attention = Attention::with_backend(128, AttentionBackend::Gpu);
    let output = gpu_attention.forward_device(
        &query_gpu,
        &key_gpu,
        &value_gpu,
        None,
        None,
    )?;
}
```

---

## Performance Notes

### GPU Memory Management

- **Sub-buffer allocation**: Use `HipBuffer::sub_buffer_view()` for zero-copy views
- **Selective memory pooling**: Large buffers (>100MB) use dedicated allocation
- **Paged KV cache**: Automatic page management for efficient cache usage

### Attention Performance

- **FlashAttention**: 2-5x speedup on GPU vs CPU
- **Causal masking**: Optimized for autoregressive generation
- **Zero-copy**: `forward_device()` avoids host-GPU copies

### Batch Processing

- **Continuous batching**: Dynamic request batching
- **Configurable timeout**: Balance latency vs throughput
- **Default**: 50ms batch timeout, 32 max batch size

---

## Thread Safety

The following types are thread-safe (`Send + Sync`):

- `InferenceEngine` (via `Arc<RwLock<T>>` internals)
- `HipBackend` (via `Arc`)
- `HipBuffer` (via `Arc<HipBufferInner>`)
- `KvCache` (when wrapped in `Arc<RwLock<T>>`)

**Note**: `HipStream` is `Send + Sync` but does NOT implement `Clone` to prevent double-free.

---

## Feature Flags

- `rocm`: Enable ROCm/HIP support (requires AMD GPU)
- `default`: CPU-only execution

Conditional compilation:
```rust
#[cfg(feature = "rocm")]
// GPU-specific code
```

---

## Version History

### 0.1.0 (Current)
- Initial public API
- GGUF model loading (Q4_0, Q4_1, Q5_0, Q5_1, FP16, FP32)
- FlashAttention GPU implementation
- Paged KV cache
- Continuous batching scheduler
- HTTP/SSE server
- Multi-architecture support (Qwen2, LLaMA, Mistral)
- MXFP quantization (MXFP4, MXFP6)

---

## Further Reading

- [README.md](../README.md) - Project overview
- [TODO.md](TODO.md) - Development roadmap
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) - Internal database schema
- [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) - Contribution guide

---

**Document Version**: 1.0
**Generated**: 2026-01-08
**Source Analysis**: Complete audit of src/ directory (59 source files)
