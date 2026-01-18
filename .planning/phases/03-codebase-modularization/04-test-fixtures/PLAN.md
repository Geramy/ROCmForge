# Plan 03-04: Consolidate Duplicate Test Fixtures

**Phase**: 03 - Codebase Modularization
**Status**: Pending
**Complexity**: Medium
**Estimated Time**: 2-3 hours

---

## Problem Statement

The test suite contains duplicate fixture code across multiple files. Common patterns like `create_test_gguf()`, `tempfile::NamedTempFile::new()`, and `create_test_backend()` are repeated in many test files.

**Current State**:
- `create_test_gguf()` duplicated in `tests/loader_tests.rs` and potentially elsewhere
- `create_test_backend()` duplicated in multiple files
- tempfile patterns repeated everywhere
- No central test utilities module

---

## Analysis

### Duplicate Fixture Locations

Based on code analysis, these fixtures appear in multiple files:

1. **GGUF Creation Fixtures**
   - `tests/loader_tests.rs`: `create_test_gguf()`, `create_test_gguf_with_f32()`
   - `tests/gguf_loader_tests.rs`: May have similar functions
   - `tests/embedding_to_lmhead_tests.rs`: `create_embedding_gguf()` (different format)

2. **Backend Fixtures**
   - `tests/execution_plan_weight_mapping_tests.rs`: `create_test_backend()`
   - Likely duplicated in other GPU tests

3. **Tempfile Patterns**
   - `tests/attention_device_tensor_tests.rs`: `tempfile::NamedTempFile::new()`
   - `tests/device_tensor_mmap_tests.rs`: `tempfile::NamedTempFile::new()`
   - `tests/mmap_loader_tests.rs`: `tempfile::NamedTempFile::new()`
   - `tests/typed_view_tests.rs`: `tempfile::NamedTempFile::new()`

4. **Tempdir Patterns**
   - `tests/gguf_loader_tests.rs`: `tempdir()`
   - `tests/glm_model_tests.rs`: `tempdir()`
   - `tests/decode_step_integration_tests.rs`: `tempdir()`

5. **Tensor Creation**
   - `tests/q_dequant_tests.rs`: `create_test_tensor()`

6. **Existing Utilities**
   - `tests/common/mod.rs`: Already has `test_model_path()`, `has_test_model()`, `serial` export

---

## Implementation Plan

### Target Structure

```
tests/
├── common/
│   ├── mod.rs              # Existing - export all utilities
│   ├── fixtures.rs         # NEW - GGUF, backend, tensor fixtures
│   └── tempfile_helpers.rs # NEW - tempfile/tempdir helpers
```

### Task 1: Create fixtures.rs

**File**: `tests/common/fixtures.rs`

```rust
//! Common test fixtures for GGUF, backend, and tensor creation

use crate::backend::hip_backend::{HipBackend, DeviceTensor};
use crate::loader::gguf::{GgufTensor, GgufTensorType, TensorShape};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

/// Create a minimal valid GGUF file for testing
///
/// Creates a GGUF file with:
/// - Valid GGUF magic number
/// - Basic metadata (architecture: "test", vocab_size: 1000)
/// - One tensor with given data
pub fn create_test_gguf(path: &Path) -> anyhow::Result<()> {
    let mut file = File::create(path)?;

    // GGUF magic
    file.write_all(b"GGUF")?;

    // Version (3 = latest)
    file.write_all(&3u32.to_le_bytes())?;

    // Tensor count
    file.write_all(&1u64.to_le_bytes())?;

    // Metadata KV count
    file.write_all(&2u32.to_le_bytes())?;

    // Write metadata key-value pairs
    write_gguf_string(&mut file, "general.architecture")?;
    write_gguf_string(&mut file, "test")?;
    write_gguf_string(&mut file, "general.vocab_size")?;
    file.write_all(&1000u32.to_le_bytes())?;

    // Tensor info
    write_gguf_string(&mut file, "test.weight")?;
    file.write_all(&[2u32, 2u32])?; // n_dims
    file.write_all(&[4u64, 4u64])?; // shape
    file.write_all(&(0u8).to_le_bytes())?; // type (F32)
    file.write_all(&0u64)?; // offset

    // Alignment padding
    file.write_all(&[0u8; 32])?;

    Ok(())
}

/// Create a GGUF file with F32 tensor data
pub fn create_test_gguf_with_f32(path: &Path) -> anyhow::Result<()> {
    create_test_gguf(path)?;
    // Append F32 data
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    for &val in &data {
        file.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

/// Create a GGUF file with embedding weights for testing
pub fn create_embedding_gguf(
    path: &Path,
    vocab_size: usize,
    hidden_size: usize,
) -> anyhow::Result<()> {
    let mut file = File::create(path)?;

    // GGUF magic and version
    file.write_all(b"GGUF")?;
    file.write_all(&3u32.to_le_bytes())?;

    // Tensor count (2: token_embd, output)
    file.write_all(&2u64.to_le_bytes())?;

    // Metadata
    file.write_all(&4u32.to_le_bytes())?;
    write_gguf_string(&mut file, "general.architecture")?;
    write_gguf_string(&mut file, "test")?;
    write_gguf_string(&mut file, "general.vocab_size")?;
    file.write_all(&(vocab_size as u32).to_le_bytes())?;
    write_gguf_string(&mut file, "embedding_length")?;
    file.write_all(&(hidden_size as u32).to_le_bytes())?;

    // token_embd.weight tensor
    write_gguf_string(&mut file, "token_embd.weight")?;
    file.write_all(&[2u32, 2u32])?;
    file.write_all(&[vocab_size as u64, hidden_size as u64])?;
    file.write_all(&(0u8).to_le_bytes())?; // F32
    file.write_all(&0u64)?;

    // output.weight tensor
    write_gguf_string(&mut file, "output.weight")?;
    file.write_all(&[2u32, 2u32])?;
    file.write_all(&[vocab_size as u64, hidden_size as u64])?;
    file.write_all(&(0u8).to_le_bytes())?; // F32
    file.write_all(&0u64)?;

    // Padding
    file.write_all(&[0u8; 32])?;

    Ok(())
}

/// Create a test GgufTensor struct
pub fn create_test_tensor(
    tensor_type: GgufTensorType,
    data: Vec<u8>,
    shape: Vec<usize>,
) -> GgufTensor {
    GgufTensor {
        name: "test_tensor".to_string(),
        shape: TensorShape::new(shape),
        tensor_type,
        offset: 0,
        data: Some(data),
    }
}

/// Create a HIP backend for testing
///
/// Uses device 0 by default. Skips test if no GPU available.
pub fn create_test_backend() -> Arc<HipBackend> {
    match HipBackend::new(0) {
        Ok(backend) => Arc::new(backend),
        Err(e) => {
            eprintln!("Skipping test: HIP backend not available: {}", e);
            panic!("No GPU available for test");
        }
    }
}

/// Create a test backend, returning None if unavailable
pub fn try_create_test_backend() -> Option<Arc<HipBackend>> {
    HipBackend::new(0).ok().map(Arc::new)
}

/// Create a DeviceTensor with test data
pub fn create_test_device_tensor(
    data: &[f32],
    shape: Vec<usize>,
    backend: &HipBackend,
) -> anyhow::Result<DeviceTensor> {
    DeviceTensor::from_host(
        data,
        TensorShape::new(shape),
        crate::backend::hip_backend::DType::F32,
    )
}

fn write_gguf_string(file: &mut File, s: &str) -> std::io::Result<()> {
    let bytes = s.as_bytes();
    file.write_all(&(bytes.len() as u64).to_le_bytes())?;
    file.write_all(bytes)?;
    Ok(())
}
```

### Task 2: Create tempfile_helpers.rs

**File**: `tests/common/tempfile_helpers.rs`

```rust
//! Helper functions for tempfile/tempdir usage in tests

use std::path::PathBuf;

/// Create a named temp file with a helpful error message
///
/// Wrapper around tempfile::NamedTempFile with consistent error context.
pub fn create_temp_file() -> anyhow::Result<tempfile::NamedTempFile> {
    tempfile::NamedTempFile::new()
        .context("Failed to create temporary file for test")
}

/// Create a temp directory with a helpful error message
///
/// Wrapper around tempfile::tempdir with consistent error context.
pub fn create_temp_dir() -> anyhow::Result<tempfile::TempDir> {
    tempfile::tempdir()
        .context("Failed to create temporary directory for test")
}

/// Create a temp file with a specific suffix
///
/// Useful for files that need specific extensions (e.g., ".gguf").
pub fn create_temp_file_with_suffix(suffix: &str) -> anyhow::Result<tempfile::NamedTempFile> {
    tempfile::NamedTempFile::with_suffix(suffix)
        .context("Failed to create temporary file with suffix")
}

/// Get a temp path that doesn't exist yet (for path testing)
pub fn temp_path() -> PathBuf {
    std::env::temp_dir().join(format!("rocmforge_test_{}", std::process::id()))
}

// Re-export commonly used tempfile types
pub use tempfile::{NamedTempFile, TempDir};
```

### Task 3: Update common/mod.rs

**File**: `tests/common/mod.rs`

```rust
//! Common test utilities and fixtures

mod fixtures;
mod tempfile_helpers;

// Public exports
pub use fixtures::{
    create_test_gguf,
    create_test_gguf_with_f32,
    create_embedding_gguf,
    create_test_tensor,
    create_test_backend,
    try_create_test_backend,
    create_test_device_tensor,
};

pub use tempfile_helpers::{
    create_temp_file,
    create_temp_dir,
    create_temp_file_with_suffix,
    temp_path,
    NamedTempFile,
    TempDir,
};

// Existing exports
use std::path::PathBuf;
use serial_test::serial;

/// Path to a small test model
pub fn test_model_path() -> PathBuf {
    env::var("ROCFORGE_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/models/tiny-llama.gguf"))
}

/// Check if test model is available
pub fn has_test_model() -> bool {
    test_model_path().exists()
}

// Re-export serial for GPU tests
pub use serial_test::serial;
```

### Task 4: Refactor loader_tests.rs

**Before**:
```rust
use tempfile::NamedTempFile;

fn create_test_gguf(path: &Path) -> anyhow::Result<()> {
    // ... 50 lines of GGUF creation
}

fn create_test_gguf_with_f32(path: &Path) -> anyhow::Result<()> {
    // ... 60 lines
}
```

**After**:
```rust
use crate::common::{
    create_test_gguf,
    create_test_gguf_with_f32,
    create_temp_file,
};

// Remove duplicate definitions
```

### Task 5: Refactor embedding_to_lmhead_tests.rs

**Before**:
```rust
use tempfile::NamedTempFile;

fn create_embedding_gguf(...) -> anyhow::Result<()> {
    // ... duplicate code
}
```

**After**:
```rust
use crate::common::{create_embedding_gguf, create_temp_file};

// Remove duplicate definition
```

### Task 6: Refactor GPU Tests

Files to update:
- `tests/execution_plan_weight_mapping_tests.rs`
- Any other files with `create_test_backend()`

**Before**:
```rust
fn create_test_backend() -> Arc<HipBackend> {
    match HipBackend::new(0) {
        Ok(backend) => Arc::new(backend),
        Err(e) => { ... }
    }
}
```

**After**:
```rust
use crate::common::{create_test_backend, try_create_test_backend};

// Remove duplicate definition
```

### Task 7: Verify All Tests Pass

```bash
cargo test
```

---

## Strategy

1. **Add to common**: Extend existing `tests/common/` module
2. **Create first**: Write fixtures.rs and tempfile_helpers.rs
3. **Update exports**: Modify common/mod.rs to re-export
4. **Refactor incrementally**: Update one test file at a time
5. **Test after each**: Run `cargo test` after each file update

---

## Dependencies

**No Dependencies**: Can run in parallel with 03-01, 03-02, 03-03

**Affects**:
- `tests/common/mod.rs` (add exports)
- `tests/loader_tests.rs`
- `tests/embedding_to_lmhead_tests.rs`
- `tests/execution_plan_weight_mapping_tests.rs`
- Any other test files with duplicate fixtures

---

## Definition of Done

- [ ] New files: `tests/common/fixtures.rs`, `tests/common/tempfile_helpers.rs`
- [ ] `tests/common/mod.rs` updated with new exports
- [ ] `loader_tests.rs` refactored to use common fixtures
- [ ] `embedding_to_lmhead_tests.rs` refactored
- [ ] GPU test files refactored to use common backend fixture
- [ ] All tempfile patterns use common helpers
- [ ] No duplicate fixture code remaining
- [ ] `cargo test` passes (all 238 tests still passing)

---

## Notes

- Keep existing utilities (`test_model_path()`, `has_test_model()`, `serial`)
- Add `.context()` to all tempfile errors for consistency
- Some tests may need specific fixture variants - keep those in the test file

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| create_test_gguf locations | 2+ | 1 (common) |
| create_test_backend locations | 2+ | 1 (common) |
| tempfile::NamedTempFile::new patterns | 5+ | use common helper |
| Duplicate fixture LOC | ~200 | 0 |

---

*Plan: 03-04*
*Created: 2026-01-18*
