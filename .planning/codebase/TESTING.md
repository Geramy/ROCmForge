# Testing Patterns

**Analysis Date:** 2026-01-18

## Test Framework

**Runner:**
- Cargo built-in test harness
- Config: Built into Cargo (no separate config file)

**Assertion Library:**
- Built-in `assert!`, `assert_eq!`, `assert_ne!` macros
- Custom GPU comparison helpers

**Run Commands:**
```bash
cargo test                              # Run all tests
cargo test --test {test_name}          # Run specific test
cargo test -- --skip gpu               # Skip GPU tests
cargo bench                             # Run benchmarks
```

## Test File Organization

**Location:**
- Unit tests: Embedded in source files using `#[cfg(test)]` modules
- Integration tests: Separate files in `tests/` directory
- Benchmarks: Separate files in `benches/` directory

**Naming:**
- Integration tests: `{feature}_tests.rs` (e.g., `hip_blas_matmul_tests.rs`)
- Unit tests: `mod tests` within source files
- Benchmarks: `{phase}_benchmark.rs` (e.g., `phase12_benchmark.rs`)

**Structure:**
```
src/
  backend/
    hip_backend.rs      # Contains embedded unit tests
  lib.rs                # Contains embedded unit tests
tests/
  hip_blas_matmul_tests.rs   # Integration tests
  attention_tests.rs         # Integration tests
benches/
  phase12_benchmark.rs       # Criterion benchmarks
```

## Test Structure

**Suite Organization:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_name() {
        // arrange
        let input = create_test_input();

        // act
        let result = function_to_test(input);

        // assert
        assert_eq!(result, expected);
    }

    // GPU test pattern
    #[test]
    fn gpu_operation_test() {
        let device = initialize_gpu();
        let result = device.execute();
        assert!(result.is_ok());
    }
}
```

**Patterns:**
- Use `#[cfg(test)]` for test modules
- GPU tests use device initialization pattern
- Deterministic RNG implementations for reproducible tests
- Test helpers like `GpuTensor` for test data management

## Mocking

**Framework:**
- Mockall 0.12 for mocking
- Location: Available in dev-dependencies

**Patterns:**
- Not extensively used (GPU code doesn't mock well)
- Mocking more common in CPU fallback tests

**What to Mock:**
- File I/O for test isolation
- External dependencies (rare - this project has few)

**What NOT to Mock:**
- GPU operations (test real hardware or skip)
- Core tensor operations (test real implementation)

## Fixtures and Factories

**Test Data:**
```rust
// Helper structs for GPU test data
struct GpuTensor {
    // GPU tensor wrapper for tests
}

// Factory pattern for test inputs
fn create_test_tensor(shape: Vec<usize>) -> Tensor {
    // Create test tensor
}
```

**Location:**
- Inline in test files (most common)
- Test helper modules in test files
- Example: `tests/hip_blas_matmul_tests.rs`

## Coverage

**Requirements:**
- No enforced coverage target
- Coverage tracked for awareness only

**Configuration:**
- No explicit coverage tool configured
- Coverage analysis not automated

**View Coverage:**
- Not available (needs setup)

## Test Types

**Unit Tests:**
- Test single function in isolation
- Embedded in source files
- Example: `src/backend/hip_backend.rs:950-1000`

**Integration Tests:**
- Test multiple modules together
- Located in `tests/` directory
- GPU/CPU comparison tests
- Example: `tests/attention_tests.rs`

**E2E Tests:**
- Not currently implemented
- Inference end-to-end flows incomplete

## Common Patterns

**Async Testing:**
```rust
#[tokio::test]
async fn test_async_operation() {
    let result = async_function().await;
    assert!(result.is_ok());
}
```

**Error Testing:**
```rust
#[test]
fn test_error_case() {
    let result = function_that_fails();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), ExpectedError);
}
```

**GPU Testing:**
```rust
#[test]
fn gpu_matmul_test() {
    let device = HipDevice::new();
    let a = device.create_tensor(...);
    let b = device.create_tensor(...);
    let c = a.matmul(&b);
    assert_eq!(c.to_vec(), expected_result);
}
```

**Skipping GPU Tests:**
```rust
#[test]
#[ignore] // or panic!("GPU_SKIP") pattern
fn requires_gpu() {
    // Test that requires actual GPU hardware
}
```

---

*Testing analysis: 2026-01-18*
*Update when test patterns change*
