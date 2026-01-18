# Codebase Concerns

**Analysis Date:** 2026-01-18

## Tech Debt

**Large File: `src/kv_cache/kv_cache.rs` (1,439 lines):**
- Issue: Far exceeds 300 LOC limit; needs modularization
- Why: KV cache functionality grew organically
- Impact: Difficult to navigate, test, and maintain
- Fix approach: Split into smaller modules by concern (cache state, block management, sequence tracking)

**Placeholder Dependencies:**
- Issue: HIP/ROCm dependencies commented out in `Cargo.toml`
- Files: `Cargo.toml` lines with `# hip = { version = "0.1", features = ["v5"] }`
- Why: Incomplete ROCm integration
- Impact: Project may not build without manual dependency setup
- Fix approach: Complete HIP bindings or document setup requirements

**Extensive Clippy Allowances:**
- Issue: 15+ clippy allowances in `src/lib.rs` mask potential issues
- Why: GPU-specific patterns trigger many lints
- Impact: Real code quality issues may be hidden
- Fix approach: Refactor to reduce allowances, document each exception

## Known Bugs

**Inference Race Conditions:**
- Symptoms: Documented race condition in engine inference loop
- Files: `src/engine.rs` (around line 578), `tests/inference_loop_spawn_race_condition_test.rs`
- Trigger: Concurrent inference requests
- Workaround: None documented
- Root cause: `unwrap()` at line 578 with known race condition

**Test Compilation Errors:**
- Symptoms: 6 test files have compilation errors (as of Jan 2026)
- Files: Various test files in `tests/`
- Trigger: Running `cargo test`
- Workaround: Skip failing tests
- Root cause: Ongoing Phase fixes not propagated to tests

**GPU Test Skipping:**
- Symptoms: Tests use `panic!("GPU_SKIP")` pattern
- Files: Multiple test files in `tests/`
- Trigger: Tests run on systems without GPU
- Workaround: Run with `--skip gpu` flag
- Fix approach: Use proper `#[ignore]` attribute

## Security Considerations

**Excessive `unwrap()` Usage (651+ instances):**
- Risk: Panics in production from unexpected failures
- Files: `src/models.rs:376-391`, `src/attention/backend_registry.rs:380-480`, `src/tensor/matmul.rs:316-318`
- Current mitigation: None
- Recommendations: Replace with proper error handling using `?` operator or `Result` returns

**Unsafe Code Without Documentation:**
- Risk: Undefined behavior if invariants are violated
- Files: 34 files containing `unsafe`, including `src/backend/hip_backend.rs`, `src/loader/mmap_loader.rs:45,85`, `src/sampler/gpu.rs`
- Current mitigation: Some unsafe blocks have comments, many don't
- Recommendations: Document safety invariants for all `unsafe` blocks and `unsafe impl Send/Sync`

**File Operation Validation:**
- Risk: Potential path traversal in model loading
- Files: `src/models.rs:376-391`
- Current mitigation: None detected
- Recommendations: Add path validation to prevent directory traversal

## Performance Bottlenecks

**Memory Allocation Warnings:**
- Problem: Large allocations (1GB+) in GPU backend
- Files: `src/backend/hip_backend.rs`
- Measurement: Allocation size warnings during HIP buffer creation
- Cause: Full tensor allocation without streaming
- Improvement path: Implement staged allocation or memory pooling

**Excessive Cloning:**
- Problem: Found in codebase scan
- Files: Throughout `src/`
- Measurement: Not quantified
- Cause: Rust ownership patterns
- Improvement path: Use references, `Cow`, or `Arc` where appropriate

**No Visible N+1 Issues:**
- Status: Good - no N+1 query patterns detected

## Fragile Areas

**HIP FFI Layer (`src/backend/hip_backend.rs`):**
- Why fragile: Direct FFI calls to HIP API with minimal error handling
- Common failures: GPU initialization failures, kernel launch errors
- Safe modification: Wrap all HIP calls in checked Result-returning functions
- Test coverage: Partial - some GPU tests, but not comprehensive

**KV Cache Concurrency (`src/kv_cache/kv_cache.rs`):**
- Why fragile: Complex RwLock patterns for concurrent access
- Common failures: Deadlocks, race conditions
- Safe modification: Use higher-level concurrency primitives
- Test coverage: Limited - needs more concurrent stress tests

**Backend Selection (`src/attention/backend_registry.rs`):**
- Why fragile: Dynamic backend selection with panic on failure
- Common failures: Missing backend causes panic
- Safe modification: Return Result instead of panicking
- Test coverage: Needs backend availability tests

## Scaling Limits

**GPU Memory:**
- Current capacity: Limited by AMD GPU VRAM
- Limit: OOM on models larger than available VRAM
- Symptoms at limit: Allocation failures, crashes
- Scaling path: Implement CPU offloading or model sharding

**Single-threaded HTTP:**
- Current capacity: One inference request at a time per GPU
- Limit: No concurrent request processing
- Symptoms at limit: Requests queue up
- Scaling path: Implement proper continuous batching (scheduler exists)

## Dependencies at Risk

**ROCm/HIP Bindings:**
- Risk: Placeholder dependencies, incomplete integration
- Impact: Project may not build
- Migration plan: Complete HIP bindings or use rocblas crate

**No External API Dependencies:**
- Risk: None - self-contained
- Impact: None
- Migration plan: Not applicable

## Missing Critical Features

**End-to-End Testing:**
- Problem: Inference end-to-end flows incomplete
- Current workaround: Manual testing
- Blocks: Production deployment confidence
- Implementation complexity: Medium

**Proper Timeout Handling:**
- Problem: Inference uses timeout panics (`src/tiny_inference_smoke.rs:56`)
- Current workaround: None
- Blocks: Reliable long-running inference
- Implementation complexity: Low

**Model Download:**
- Problem: No HuggingFace Hub integration or model downloading
- Current workaround: Manual model file acquisition
- Blocks: Easy onboarding for new users
- Implementation complexity: Medium

## Test Coverage Gaps

**Error Paths:**
- What's not tested: Most `unwrap()` code paths assume success
- Risk: Panics in production from untested error conditions
- Priority: High
- Difficulty to test: Requires mocking GPU failures

**Concurrent Access:**
- What's not tested: KV cache under concurrent load
- Risk: Deadlocks, data races
- Priority: High
- Difficulty to test: Requires multi-threaded test setup

**End-to-End Flows:**
- What's not tested: Complete inference from HTTP request to response
- Risk: Integration failures between components
- Priority: Medium
- Difficulty to test: Requires full stack including GPU

---

*Concerns audit: 2026-01-18*
*Update as issues are fixed or new ones discovered*
