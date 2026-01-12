# E2E Integration Tests - Quick Reference

## Overview

Comprehensive end-to-end integration tests for ROCmForge that validate the complete inference pipeline from model loading through token generation.

## Test Results Summary

✅ **5/6 tests passing** (1 test ignored by design - slow full pipeline test)

## Quick Start

### Run All E2E Tests
```bash
cargo test --test e2e_integration_tests --features rocm -- --test-threads=1
```

**Expected Output**:
```
running 6 tests
test test_error_recovery_e2e ... ok
test test_full_pipeline_e2e ... ignored
test test_inference_execution_e2e ... ok
test test_kv_cache_e2e ... ok
test test_model_loading_e2e ... ok
test test_scheduler_e2e ... ok

test result: ok. 5 passed; 0 failed; 1 ignored
```

### Run Individual Tests

```bash
# Model loading test
cargo test --test e2e_integration_tests test_model_loading_e2e --features rocm -- --test-threads=1

# Inference execution test
cargo test --test e2e_integration_tests test_inference_execution_e2e --features rocm -- --test-threads=1

# KV cache test
cargo test --test e2e_integration_tests test_kv_cache_e2e --features rocm -- --test-threads=1

# Scheduler test
cargo test --test e2e_integration_tests test_scheduler_e2e --features rocm -- --test-threads=1

# Error recovery test
cargo test --test e2e_integration_tests test_error_recovery_e2e --features rocm -- --test-threads=1
```

### Run Slow Full Pipeline Test
```bash
cargo test --test e2e_integration_tests --features rocm -- --ignored --test-threads=1
```

### Run with Verbose Output
```bash
cargo test --test e2e_integration_tests --features rocm -- --test-threads=1 --nocapture
```

## Test Coverage

| Test | Scenario | Validates |
|------|----------|-----------|
| test_model_loading_e2e | Model Loading | Engine initialization, model loading, stats verification |
| test_inference_execution_e2e | Inference Execution | Token generation, finish reasons, prompt processing |
| test_kv_cache_e2e | KV Cache | Cache population, active sequences, token tracking |
| test_scheduler_e2e | Scheduler | Request queuing, batching, completion tracking |
| test_error_recovery_e2e | Error Recovery | Invalid inputs, parameter validation, cancellation |
| test_full_pipeline_e2e | Full Pipeline | Performance, throughput, multi-request handling |

## Model Requirements

Tests look for models in this order:
1. `/home/feanor/Projects/ROCmForge/models/qwen2.5-0.5b.gguf`
2. `/home/feanor/Projects/ROCmForge/models/bge-small-en-v1.5.Q8_0.gguf`
3. `tests/data/tiny_model.gguf`

If no models are found, tests skip gracefully.

## GPU Requirements

- AMD GPU with ROCm support
- ROCm runtime installed
- `amdhip64` library in `LD_LIBRARY_PATH`

If GPU is unavailable, tests skip gracefully.

## Troubleshooting

### Tests Skip with "No test model found"
**Solution**: Download a GGUF model to one of the expected paths:
```bash
mkdir -p models
cd models
# Download your preferred GGUF model
wget https://example.com/qwen2.5-0.5b.gguf
```

### Tests Skip with "GPU not available"
**Solution**: Ensure ROCm is properly installed:
```bash
# Check ROCm installation
rocm-smi

# Check amdhip64 library
echo $LD_LIBRARY_PATH | grep amdhip64
```

### Test Fails with "No embedding tensor found"
**Cause**: Model uses different tensor naming convention
**Impact**: Test skips gracefully
**Status**: Known issue - needs Qwen2 tensor name support

## Test Duration

- Fast tests (5): ~2 seconds total
- Full pipeline test: ~30-60 seconds (ignored by default)

## CI/CD Integration

Add to your CI pipeline:
```yaml
- name: Run E2E Tests
  run: cargo test --test e2e_integration_tests --features rocm -- --test-threads=1
  env:
    ROCMFORGE_MODELS: /path/to/models
```

## Related Documentation

- **Implementation Report**: `E2E_INTEGRATION_TESTS_IMPLEMENTATION_REPORT.md`
- **Test Source**: `tests/e2e_integration_tests.rs`
- **Engine Docs**: `src/engine.rs`
- **Scheduler Docs**: `src/scheduler/scheduler.rs`
