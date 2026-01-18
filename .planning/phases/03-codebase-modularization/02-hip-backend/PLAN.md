# Plan 03-02: Split hip_backend.rs into Focused Modules

**Phase**: 03 - Codebase Modularization
**Status**: Pending
**Complexity**: High
**Estimated Time**: 5-6 hours

---

## Problem Statement

`src/backend/hip_backend.rs` is 3684 lines - far exceeding the 300 LOC convention. This monolithic file contains multiple distinct concerns:
- HIP FFI bindings (extern "C" functions)
- Error types and conversions
- Device properties and opaque FFI structs
- Stream and event management
- Buffer management
- Module and kernel loading
- Backend implementation
- Tensor operations
- Async loading utilities

**Current State**:
- 3684 lines in a single file
- Mix of FFI bindings, error handling, and GPU operations
- Difficult to locate specific functionality

---

## Analysis

### File Structure Breakdown

Based on code analysis, the file contains these distinct components:

1. **HIP FFI Bindings** (~100 lines)
   - `extern "C"` block with HIP API function declarations
   - Constants (HIP_MEMCPY_*, HIP_SUCCESS)

2. **FFI Structs** (~150 lines)
   - `HipDeviceProp` (opaque 1472-byte buffer)
   - `hipUUID`
   - Field offset constants and accessor methods

3. **Error Types** (~100 lines)
   - `HipError` enum
   - `HipResult<T>` type alias
   - Error conversions (From implementations)

4. **HipStream** (~150 lines)
   - Stream creation, synchronization, destruction
   - Drop implementation

5. **HipEvent** (~200 lines)
   - Event creation, recording, synchronization
   - Elapsed time measurement
   - Drop implementation

6. **HipBuffer** (~400 lines)
   - Buffer allocation, copying (H2D, D2D, D2H)
   - Stream-aware copy methods (Phase 1 fix)
   - Drop implementation

7. **HipModule** (~200 lines)
   - Module loading from file/data
   - Function lookup
   - Drop implementation

8. **HipKernel** (~150 lines)
   - Kernel launching
   - Grid/block dimension configuration

9. **HipDevice** (~200 lines)
   - Device properties, memory info
   - Device selection

10. **HipBackend** (~800 lines)
    - Main backend struct
    - Device management
    - Stream operations
    - Memory allocation

11. **DeviceTensor** (~400 lines)
    - Tensor wrapper around HipBuffer
    - Shape and dtype information
    - Copy operations

12. **ModelRuntime** (~200 lines)
    - Runtime state management
    - Cleanup handling

13. **AsyncLoader** (~400 lines)
    - Async GPU loading (Phase 2)
    - Rayon integration

---

## Implementation Plan

### Target Structure

```
src/backend/
├── mod.rs              # Public exports
├── ffi.rs              # HIP FFI bindings (extern "C", constants)
├── error.rs            # HipError, HipResult, error conversions
├── device.rs           # HipDevice, HipDeviceProp, hipUUID
├── stream.rs           # HipStream
├── event.rs            # HipEvent
├── buffer.rs           # HipBuffer
├── module.rs           # HipModule, HipKernel
├── backend.rs          # HipBackend (main implementation)
├── tensor.rs           # DeviceTensor
├── runtime.rs          # ModelRuntime
└── async_loader.rs     # AsyncLoader (Phase 2)
```

### Task 1: Create Module Files

```bash
# Create new backend module files
touch src/backend/{ffi.rs,error.rs,device.rs,stream.rs,event.rs,buffer.rs,module.rs,tensor.rs,runtime.rs,async_loader.rs}
```

### Task 2: Extract FFI Bindings

**File**: `src/backend/ffi.rs`

```rust
//! HIP FFI bindings
//!
//! Raw foreign function interface bindings to the HIP runtime.
//! These are re-exported from the hip_backend module.

use std::ffi::{c_void, CString};
use std::ptr;

#[link(name = "amdhip64")]
#[allow(dead_code)]
extern "C" {
    pub fn hipInit(flags: u32) -> i32;
    pub fn hipGetDeviceCount(count: *mut i32) -> i32;
    pub fn hipGetDeviceProperties(props: *mut super::HipDeviceProp, deviceId: i32) -> i32;
    pub fn hipSetDevice(deviceId: i32) -> i32;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn hipFree(ptr: *mut c_void) -> i32;
    // ... all other HIP functions
}

// HIP constants
pub const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
pub const HIP_SUCCESS: i32 = 0;
```

### Task 3: Extract Error Types

**File**: `src/backend/error.rs`

```rust
//! Error types for HIP operations

use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum HipError {
    #[error("HIP initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Kernel loading failed: {0}")]
    KernelLoadFailed(String),
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
    #[error("Memory copy failed: {0}")]
    MemoryCopyFailed(String),
    #[error("Memory query failed: {0}")]
    MemoryQueryFailed(String),
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    #[error("Device not found")]
    DeviceNotFound,
    #[error("Device error: {0}")]
    DeviceError(String),
    #[error("Generic error: {0}")]
    GenericError(String),
    #[error("Internal lock poisoned - this indicates a bug: {0}")]
    LockPoisoned(String),
}

impl<T> From<std::sync::PoisonError<T>> for HipError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        HipError::LockPoisoned(format!("Lock poisoned: {}", err))
    }
}

// Error conversions for other error types
impl From<crate::model::kv_cache::KVCacheError> for HipError { ... }
impl From<crate::ggml::GgmlError> for HipError { ... }

pub type HipResult<T> = Result<T, HipError>;
```

### Task 4: Extract Device Types

**File**: `src/backend/device.rs`

```rust
//! HIP device properties and management

use super::ffi;
use std::fmt;

/// Opaque buffer for hipDeviceProp_t - MUST be exactly 1472 bytes
#[repr(C)]
#[derive(Debug, Clone)]
pub struct HipDeviceProp {
    _buffer: [u8; 1472],
}

impl HipDeviceProp {
    const NAME_OFFSET: usize = 0;
    const TOTAL_GLOBAL_MEM_OFFSET: usize = 284;
    const MULTI_PROCESSOR_COUNT_OFFSET: usize = 508;

    pub fn name(&self) -> String { ... }
    pub fn total_global_mem(&self) -> u64 { ... }
    pub fn multi_processor_count(&self) -> i32 { ... }
}

impl Default for HipDeviceProp {
    fn default() -> Self { ... }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct hipUUID {
    pub bytes: [u8; 16],
}

/// HIP device handle
#[derive(Debug, Clone)]
pub struct HipDevice {
    pub id: i32,
    pub name: String,
    pub props: HipDeviceProp,
}
```

### Task 5: Extract HipStream

**File**: `src/backend/stream.rs`

```rust
//! HIP stream management

use super::ffi;
use super::{HipError, HipResult};
use std::ptr;

#[derive(Debug)]
pub struct HipStream {
    pub(crate) ptr: *mut std::ffi::c_void,
}

impl HipStream {
    pub fn new() -> HipResult<Self> { ... }
    pub fn ptr(&self) -> *mut std::ffi::c_void { ... }
    pub fn synchronize(&self) -> HipResult<()> { ... }
}

impl Drop for HipStream {
    fn drop(&mut self) { ... }
}
```

### Task 6: Extract HipEvent

**File**: `src/backend/event.rs`

```rust
//! HIP event management

use super::ffi;
use super::{HipError, HipResult};

#[derive(Debug)]
pub struct HipEvent {
    pub(crate) ptr: *mut std::ffi::c_void,
}

impl HipEvent {
    pub fn new() -> HipResult<Self> { ... }
    pub fn record(&self, stream: &super::HipStream) -> HipResult<()> { ... }
    pub fn synchronize(&self) -> HipResult<()> { ... }
    pub fn elapsed_ms(&self, start: &Self) -> HipResult<f32> { ... }
}

impl Drop for HipEvent {
    fn drop(&mut self) { ... }
}
```

### Task 7: Extract HipBuffer

**File**: `src/backend/buffer.rs`

```rust
//! HIP buffer management

use super::ffi;
use super::{HipError, HipResult, HipStream};

#[derive(Debug, Clone)]
pub struct HipBuffer {
    pub(crate) ptr: *mut std::ffi::c_void,
    pub size: usize,
}

impl HipBuffer {
    pub fn allocate(size: usize) -> HipResult<Self> { ... }
    pub fn from_host<T: Clone>(data: &[T]) -> HipResult<Self> { ... }
    pub fn copy_to_host<T: Clone>(&self) -> HipResult<Vec<T>> { ... }
    pub fn copy_from_buffer(&self, src: &Self, stream: &HipStream) -> HipResult<()> { ... }
    pub fn copy_from_buffer_with_stream(&self, src: &Self, stream: &HipStream) -> HipResult<()> { ... }
}

// HipBufferInner for mutable state
pub struct HipBufferInner { ... }
impl Drop for HipBufferInner { ... }
```

### Task 8: Extract HipModule and HipKernel

**File**: `src/backend/module.rs`

```rust
//! HIP module and kernel management

use super::ffi;
use super::{HipError, HipResult, HipStream};

#[derive(Debug)]
pub struct HipModule {
    pub(crate) ptr: *mut std::ffi::c_void,
}

impl HipModule {
    pub fn load_from_file(path: &str) -> HipResult<Self> { ... }
    pub fn load_from_data(image: &[u8]) -> HipResult<Self> { ... }
    pub fn get_function(&self, name: &str) -> HipResult<HipKernel> { ... }
}

impl Drop for HipModule {
    fn drop(&mut self) { ... }
}

#[derive(Debug)]
pub struct HipKernel {
    pub(crate) ptr: *mut std::ffi::c_void,
}

impl HipKernel {
    pub fn launch(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: &HipStream,
    ) -> HipResult<()> { ... }
}
```

### Task 9: Extract DeviceTensor

**File**: `src/backend/tensor.rs`

```rust
//! GPU tensor wrapper

use super::{HipBuffer, HipBackend, HipError, HipResult};
use crate::loader::mmap_loader::TensorShape;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub buffer: HipBuffer,
    pub shape: TensorShape,
    pub dtype: DType,
}

impl DeviceTensor {
    pub fn allocate(shape: TensorShape, dtype: DType) -> HipResult<Self> { ... }
    pub fn from_host<T: Clone>(data: &[T], shape: TensorShape, dtype: DType) -> HipResult<Self> { ... }
    pub fn copy_to_host(&self) -> HipResult<Vec<u8>> { ... }
    pub fn copy_from_device_safe(&self, other: &Self, backend: &HipBackend) -> HipResult<()> { ... }
}
```

### Task 10: Extract AsyncLoader

**File**: `src/backend/async_loader.rs`

```rust
//! Async GPU loading utilities (Phase 2)

use super::HipBackend;
use super::DeviceTensor;
use rayon::prelude::*;

pub struct AsyncLoader {
    backend: Arc<HipBackend>,
    chunk_size: usize,
}

impl AsyncLoader {
    pub fn new(backend: Arc<HipBackend>) -> Self { ... }
    pub fn load_tensor_parallel(&self, data: &[u8], shape: TensorShape) -> HipResult<DeviceTensor> { ... }
}
```

### Task 11: Extract ModelRuntime

**File**: `src/backend/runtime.rs`

```rust
//! Model runtime state management

use super::HipBackend;
use std::sync::Arc;

#[derive(Debug)]
pub struct ModelRuntime {
    backend: Arc<HipBackend>,
    cleanup_timeout_ms: u64,
}

impl ModelRuntime {
    pub fn new(backend: Arc<HipBackend>) -> Self { ... }
    pub fn set_cleanup_timeout(&mut self, timeout_ms: u64) { ... }
    pub fn cleanup(&self) -> HipResult<()> { ... }
}
```

### Task 12: Update hip_backend.rs to Main Backend

**File**: `src/backend/hip_backend.rs` (now focused on HipBackend)

```rust
//! ROCm/HIP backend for GPU kernel execution

mod ffi;
mod error;
mod device;
mod stream;
mod event;
mod buffer;
mod module;
mod tensor;
mod runtime;
mod async_loader;

// Public exports
pub use error::{HipError, HipResult};
pub use device::{HipDevice, HipDeviceProp, hipUUID};
pub use stream::HipStream;
pub use event::HipEvent;
pub use buffer::{HipBuffer, HipBufferInner};
pub use module::{HipModule, HipKernel};
pub use tensor::DeviceTensor;
pub use runtime::ModelRuntime;
pub use async_loader::AsyncLoader;

use std::sync::Arc;

/// Main HIP backend
#[derive(Debug)]
pub struct HipBackend {
    device: HipDevice,
    stream: HipStream,
    // ...
}

impl HipBackend {
    pub fn new(device_id: usize) -> HipResult<Self> { ... }
    pub fn default_device() -> HipResult<Self> { ... }
    // ... backend methods
}
```

### Task 13: Update mod.rs

**File**: `src/backend/mod.rs`

```rust
//! HIP/ROCm GPU backend abstraction

pub mod hip_backend;
pub use hip_backend::*;
```

---

## Strategy

1. **Start with FFI**: Extract ffi.rs first (no dependencies)
2. **Extract errors**: error.rs next (minimal dependencies)
3. **Extract primitives**: device, stream, event, buffer, module (depend on ffi/error)
4. **Extract higher-level**: tensor, async_loader, runtime (depend on primitives)
5. **Keep main backend**: hip_backend.rs becomes the orchestrator
6. **Test incrementally**: Run `cargo check` after each extraction

---

## Dependencies

**No Dependencies**: Can run in parallel with 03-01, 03-03, 03-04

**Affects**:
- `src/backend/mod.rs` (module declarations)
- All files importing from `crate::backend::hip_backend`

---

## Definition of Done

- [ ] 12 new module files created
- [ ] Each module <500 LOC
- [ ] All public APIs preserved
- [ ] `cargo check` passes
- [ ] `cargo test --lib` passes
- [ ] FFI bindings properly isolated
- [ ] Phase 1 stream-safety fix preserved (copy_from_buffer_with_stream)

---

## Notes

- FFI bindings must use `#[allow(dead_code)]` - not all HIP functions are used
- Keep HipDeviceProp as opaque buffer (1472 bytes) - this is correct
- Preserve all Phase 1 fixes (hipMemcpyAsync with stream)
- Preserve all Phase 2 async loading functionality

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Max file LOC | 3684 | <500 per file |
| Module count | 1 | 12 |
| Public API | Same | Same |
| Test pass rate | 100% | 100% |

---

*Plan: 03-02*
*Created: 2026-01-18*
