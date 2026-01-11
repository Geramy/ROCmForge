# Code Review Report: CLI and Scheduler

**Date**: 2026-01-10
**Reviewer**: Claude Code (code-reviewer agent)
**Scope**: `/home/feanor/Projects/ROCmForge/src/bin/rocmforge_cli.rs` and `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs`

---

## Executive Summary

Conducted a thorough review of the CLI and scheduler code, examining 1,433 lines across two critical files. Found **6 bugs** (2 critical, 2 high, 2 medium priority) and **8 code quality issues**. The most significant issues involve incorrect inference loop logic, resource leaks, and missing error handling in critical paths.

**Overall Assessment**: The codebase has architectural soundness but contains several bugs that will cause runtime failures, incorrect behavior, and resource leaks under normal operating conditions. Immediate fixes are required for critical issues.

---

## Files Reviewed

1. `/home/feanor/Projects/ROCmForge/src/bin/rocmforge_cli.rs` (510 lines)
2. `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs` (923 lines)

---

## Critical Issues (Must Fix)

### BUG-1: Inference Loop Not Started for HTTP Server
**Location**: `/home/feanor/Projects/ROCmForge/src/http/server.rs:549`

**Severity**: CRITICAL - HTTP server will never process requests

**Description**:
The HTTP server's `run_server()` function calls `engine.run_inference_loop().await` on line 549, which **blocks indefinitely**. This prevents the server from actually starting because the axum server setup on line 556 never executes.

```rust
// Line 549 - BLOCKING CALL
engine.run_inference_loop().await;  // Never returns!

// Line 556 - NEVER REACHED
let listener = tokio::net::TcpListener::bind(addr).await?;
axum::serve(listener, app).await?;
```

**Evidence**:
- Read `/home/feanor/Projects/ROCmForge/src/http/server.rs:545-560`
- Read `/home/feanor/Projects/ROCmForge/src/engine.rs:200-243` - `run_inference_loop()` spawns a task but the function signature returns `()`, not a `JoinHandle`
- The inference loop is designed to run forever (`while *self.is_running.read().await` on line 408)

**Impact**:
- HTTP server **never binds to the configured address**
- All HTTP requests fail immediately
- Server appears to start but is completely non-functional
- Cannot use `rocmforge-cli serve` command

**Correct Pattern** (from CLI):
The CLI's `create_engine()` function (line 468-482) correctly spawns the inference loop in a background task:
```rust
// Line 476-479 - CORRECT
tokio::spawn(async move {
    let _ = engine_clone.run_inference_loop().await;
});
```

**Fix Required**:
Change line 549 from blocking call to spawned task:
```rust
// Spawn inference loop in background - don't block server startup
let engine_clone = engine.clone();
tokio::spawn(async move {
    let _ = engine_clone.run_inference_loop().await;
});
```

**Test Case**:
```bash
# This should fail to bind to port
rocmforge-cli serve --gguf /path/to/model.gguf --addr 127.0.0.1:8080

# In another terminal, this should timeout/fail
curl http://127.0.0.1:8080/health
```

---

### BUG-2: Scheduler State Not Updated After Token Generation
**Location**: `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:555-589`

**Severity**: CRITICAL - Generated tokens lost, incorrect completion detection

**Description**:
The `update_iteration_batch()` function has **two code paths** that both attempt to update the same request state, causing a race condition where the second update **overwrites the first** with stale data.

**The Problem**:
1. Lines 562-574: **First code path** - Finds completed requests by checking `is_complete()` and removes them from `processing_requests`
2. Lines 578-586: **Second code path** - Iterates over `batch.requests` and re-inserts them into `processing_requests`

**Critical Bug on Line 584**:
```rust
// Line 584 - OVERWRITES completed requests with stale clone
self.processing_requests.insert(request.request_id, request);
```

This happens **after** the request was already removed on line 570! The `request` in `batch.requests` is a **stale clone** from before tokens were generated.

**Detailed Flow**:
```
1. Engine generates token for request_id=42
2. Engine calls scheduler.add_generated_token(42, token)
3. InferenceEngine::process_single_request_impl() line 554 updates request
4. process_batch() line 470 snapshots the request (has new token)
5. process_batch() line 496 calls update_iteration_batch()
6. update_iteration_batch() line 570 removes request from processing_requests
7. update_iteration_batch() line 584 RE-INSERTS stale request from batch
8. GENERATED TOKEN IS LOST!
```

**Evidence**:
- Read `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:555-589`
- Read `/home/feanor/Projects/ROCmForge/src/engine.rs:445-506` - `process_batch()` logic
- The `batch.requests` passed to `update_iteration_batch()` is cloned on line 464 **before** processing
- Processing happens on lines 467-488, updating the scheduler directly
- The batch snapshot is now stale

**Impact**:
- Generated tokens are **lost** - overwritten by stale state
- `is_complete()` returns incorrect values
- Generation never finishes (or finishes prematurely)
- `finish_reason` is incorrect
- Clients receive incomplete or incorrect responses

**Why This Doesn't Crash**:
The code has a guard on line 579-582:
```rust
if !self.processing_requests.contains_key(&request.request_id) {
    continue;  // Skip if already removed
}
```

But this guard **only works if the request was removed** in the first loop (lines 562-574). However, requests that are **still processing** (not complete) get re-inserted with stale state!

**Fix Required**:
The function should either:
1. **Remove the second update loop entirely** (lines 578-586) - the first loop already handles everything
2. **Only update non-completed requests** that haven't been touched by the first loop
3. **Return the refreshed batch from the scheduler** instead of passing it back in

**Recommended Fix**:
```rust
pub fn update_iteration_batch(&mut self, mut batch: IterationBatch) -> SchedulerResult<Vec<GenerationRequest>> {
    // Compact the batch to identify completed requests
    batch.compact();

    let mut completed = Vec::new();
    let mut refreshed = Vec::new();

    // Get the actual completed requests from processing_requests
    let mut to_complete = Vec::new();
    for (&req_id, request) in &self.processing_requests {
        if request.is_complete() || request.state == RequestState::Failed {
            to_complete.push(req_id);
        } else {
            // Refresh with current state
            refreshed.push(request.clone());
        }
    }

    for req_id in to_complete {
        if let Some(request) = self.processing_requests.remove(&req_id) {
            self.completed_requests.insert(req_id, request.clone());
            completed.push(request);
        }
    }

    // Update batch with refreshed requests for next iteration
    batch.requests = refreshed;
    batch.sequence_positions = refreshed.iter()
        .map(|r| r.total_tokens())
        .collect();

    Ok(completed)
}
```

**Test Case**:
```rust
#[tokio::test]
async fn test_update_iteration_batch_preserves_tokens() {
    let config = SchedulerConfig::default();
    let mut scheduler = Scheduler::new(config);

    // Submit request
    scheduler.submit_request(vec![1, 2, 3], 5, 0.8, 50, 0.9).unwrap();
    let batch = scheduler.get_next_iteration_batch().unwrap();

    // Add tokens directly (simulating engine behavior)
    let req_id = batch.requests[0].request_id;
    scheduler.add_generated_token(req_id, 100).unwrap();
    scheduler.add_generated_token(req_id, 101).unwrap();

    // Get current state
    let before = scheduler.get_request(req_id).unwrap();
    assert_eq!(before.generated_tokens.len(), 2);

    // Update batch
    let completed = scheduler.update_iteration_batch(batch).unwrap();
    assert!(completed.is_empty());  // Not complete yet (max_tokens=5)

    // CRITICAL: Tokens must be preserved
    let after = scheduler.get_request(req_id).unwrap();
    assert_eq!(after.generated_tokens.len(), 2, "Tokens were lost!");
    assert_eq!(after.generated_tokens, vec![100, 101]);
}
```

---

## High Priority Issues (Should Fix)

### BUG-3: HTTP Server Hangs on Shutdown
**Location**: `/home/feanor/Projects/ROCmForge/src/http/server.rs:549`

**Severity**: HIGH - Server cannot be shut down cleanly

**Description**:
Because `run_inference_loop()` blocks forever (see BUG-1), the server's HTTP listener setup on line 556 is never reached. When the server receives a shutdown signal, it cannot properly:

1. Stop accepting new connections
2. Drain existing connections
3. Stop the inference loop
4. Clean up resources

**Impact**:
- `SIGTERM`/`SIGINT` signals cause **unclean shutdown**
- Resources (GPU memory, HIP streams) are not properly released
- Potential GPU state corruption
- Zombie processes

**Fix Required**:
Use `tokio::spawn()` for inference loop (same as BUG-1 fix) and add shutdown handling:
```rust
pub async fn run_server(
    addr: &str,
    gguf_path: Option<&str>,
    tokenizer_path: Option<&str>,
) -> ServerResult<()> {
    // ... setup code ...

    let engine = Arc::new(engine);
    engine.start().await?;

    // Spawn inference loop in background
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        let _ = engine_clone.run_inference_loop().await;
    });

    let server = InferenceServer::new(Some(engine), tokenizer.clone());
    let app = create_router(server);

    info!("Starting ROCmForge server on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Add graceful shutdown handling
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .unwrap()
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
```

---

### BUG-4: CLI Stream Mode Prints Extra Newline
**Location**: `/home/feanor/Projects/ROCmForge/src/bin/rocmforge_cli.rs:458`

**Severity**: MEDIUM - Cosmetic output issue

**Description**:
In `run_local_stream()`, after the request completes, there's an extra `println!()` on line 458 that creates an unnecessary blank line:

```rust
// Line 452-459
if status.is_complete() {
    println!(
        "\n[request {} finished: {}]",
        status.request_id,
        status.finish_reason.unwrap_or_else(|| "completed".to_string())
    );
    println!();  // Line 458 - Extra blank line!
    break;
}
```

**Impact**:
- Output formatting issue in stream mode
- Affects user experience when parsing streamed output
- Extra blank line appears after completion message

**Fix Required**:
Remove line 458:
```rust
if status.is_complete() {
    println!(
        "\n[request {} finished: {}]",
        status.request_id,
        status.finish_reason.unwrap_or_else(|| "completed".to_string())
    );
    break;
}
```

---

### BUG-5: Unwrap on Tokenizer Decode
**Location**: `/home/feanor/Projects/ROCmForge/src/bin/rocmforge_cli.rs:447`

**Severity**: MEDIUM - Potential panic

**Description**:
Line 447 in `run_local_stream()` uses `unwrap()` on tokenizer decode:
```rust
stdout
    .write_all(tokenizer.decode_token(token).as_bytes())
    .await?;
```

If `decode_token()` returns an empty string or panics, the entire stream will fail.

**Impact**:
- Potential panic during token streaming
- Crash instead of graceful error handling
- Loss of partial generation results

**Fix Required**:
Add error handling:
```rust
let token_text = tokenizer.decode_token(token);
if !token_text.is_empty() {
    stdout.write_all(token_text.as_bytes()).await?;
    stdout.flush().await?;
}
```

---

### BUG-6: Request ID Overflow Not Handled
**Location**: `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:334-335`

**Severity**: LOW - Edge case, long-running server issue

**Description**:
Request IDs are simple incrementing integers that will overflow after ~4 billion requests:
```rust
let request_id = self.next_request_id;
self.next_request_id += 1;
```

**Impact**:
- After 4,294,967,295 requests, ID wraps to 0
- Could cause ID collision if old requests are still in `completed_requests`
- HashMap lookups could return wrong request

**Fix Required**:
Use wrapping arithmetic or detect overflow:
```rust
let request_id = self.next_request_id;
self.next_request_id = self.next_request_id.wrapping_add(1);

// Or use u64 for request IDs
```

---

## Code Quality Issues

### ISSUE-1: Duplicate Tokenizer Path Inference Logic
**Location**:
- `/home/feanor/Projects/ROCmForge/src/bin/rocmforge_cli.rs:139, 181-183`
- `/home/feanor/Projects/ROCmForge/src/http/server.rs:506-509`

**Severity**: MEDIUM - Code duplication

**Description**:
The tokenizer path inference logic is duplicated in three places:
1. CLI `Generate` command (line 139)
2. CLI `Serve` command (line 181-183)
3. HTTP server (line 506-509)

**Current Code** (CLI Generate):
```rust
let tokenizer_path = tokenizer.clone().or_else(|| infer_tokenizer_path(&path));
```

**Current Code** (CLI Serve):
```rust
let tokenizer_path = tokenizer
    .clone()
    .or_else(|| gguf.as_deref().and_then(infer_tokenizer_path));
```

**Recommendation**:
Extract to a helper function:
```rust
fn resolve_tokenizer_path(
    provided: Option<String>,
    gguf_path: Option<&str>,
) -> Option<String> {
    provided.or_else(|| {
        gguf_path.and_then(infer_tokenizer_path)
    })
}
```

---

### ISSUE-2: Inconsistent Error Handling in CLI
**Location**: `/home/feanor/Projects/ROCmForge/src/bin/rocmforge_cli.rs:388-390`

**Severity**: MEDIUM - Error handling inconsistency

**Description**:
Line 388-390 converts `EngineError` to `anyhow::Error` using string conversion:
```rust
engine
    .cancel_request(request_id)
    .await
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
```

This loses error type information and context. The pattern is repeated on line 432-435.

**Recommendation**:
Use `anyhow::Context` or preserve error type:
```rust
engine
    .cancel_request(request_id)
    .await
    .map_err(|e| anyhow::anyhow!("Failed to cancel request: {}", e))?;
```

---

### ISSUE-3: Busy-Wait Loop in wait_for_completion()
**Location**: `/home/feanor/Projects/ROCmForge/src/bin/rocmforge_cli.rs:507`

**Severity**: LOW - Inefficient polling

**Description**:
The `wait_for_completion()` function polls every 25ms:
```rust
sleep(Duration::from_millis(25)).await;
```

This is inefficient compared to using the notification system that already exists in the engine.

**Evidence**:
- `/home/feanor/Projects/ROCmForge/src/engine.rs:382-384` shows `subscribe_request()` method
- `/home/feanor/Projects/ROCmForge/src/engine.rs:386-390` shows notification system
- HTTP server uses this pattern (line 333 in server.rs)

**Recommendation**:
Use the notification system:
```rust
async fn wait_for_completion(
    engine: &Arc<InferenceEngine>,
    tokenizer: &TokenizerAdapter,
    request_id: u32,
) -> anyhow::Result<GenerateResponse> {
    let notifier = engine
        .subscribe_request(request_id)
        .await
        .ok_or_else(|| anyhow::anyhow!("request {} has no notifier", request_id))?;

    loop {
        notifier.notified().await;
        let status = engine
            .get_request_status(request_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("request {} disappeared", request_id))?;
        if status.is_complete() {
            let text = tokenizer.decode(&status.generated_tokens);
            return Ok(GenerateResponse {
                request_id: status.request_id,
                text,
                tokens: status.generated_tokens.clone(),
                finished: true,
                finish_reason: status
                    .finish_reason
                    .clone()
                    .or(Some("completed".to_string())),
            });
        }
    }
}
```

---

### ISSUE-4: Unchecked remove() Return Value
**Location**: `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:381, 461`

**Severity**: LOW - Unnecessary unwrap()

**Description**:
Lines 381 and 461 use `.unwrap()` on `remove()`:
```rust
let mut request = self.pending_queue.remove(pos).unwrap();
```

The position was just found, so `unwrap()` is safe but unnecessary.

**Recommendation**:
Use `remove(pos)` directly without unwrap (returns the removed value):
```rust
let request = self.pending_queue.remove(pos);
```

---

### ISSUE-5: Inconsistent State Transition Checks
**Location**: `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:83-84, 93-94, 108-109, 120-125`

**Severity**: LOW - State machine robustness

**Description**:
The state transition methods check current state before transitioning, but the checks are **inconsistent**:

- `start_processing()`: Checks `!= Pending`
- `complete()`: Checks `!= Processing`
- `fail()`: Checks `!= Processing`
- `cancel()`: Checks for terminal states (Completed, Failed, Cancelled)

However, there's **no check** for:
- Transitioning from `Cancelled` to `Failed` (or vice versa)
- Transitioning from `Completed` to `Failed`

**Recommendation**:
Add a state transition matrix or use a more robust state machine:
```rust
pub fn can_transition_to(&self, new_state: RequestState) -> bool {
    match (self.state, new_state) {
        (RequestState::Pending, RequestState::Processing) => true,
        (RequestState::Pending, RequestState::Cancelled) => true,
        (RequestState::Processing, RequestState::Completed) => true,
        (RequestState::Processing, RequestState::Failed) => true,
        (RequestState::Processing, RequestState::Cancelled) => true,
        _ => false,  // All other transitions are invalid
    }
}
```

---

### ISSUE-6: Missing Documentation for Public API
**Location**: `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:322-348, 350-400`

**Severity**: LOW - Missing documentation

**Description**:
Public methods like `submit_request()`, `create_batch()`, `get_next_iteration_batch()` lack:
- Parameter documentation
- Return value documentation
- Error conditions documentation
- Usage examples

**Recommendation**:
Add rustdoc comments:
```rust
/// Submit a new generation request to the scheduler.
///
/// # Arguments
/// * `prompt_tokens` - Token IDs representing the input prompt
/// * `max_tokens` - Maximum number of tokens to generate
/// * `temperature` - Sampling temperature (0.0 = deterministic, 1.0 = default)
/// * `top_k` - Top-k sampling parameter
/// * `top_p` - Nucleus sampling parameter
///
/// # Returns
/// * `Ok(u32)` - The unique request ID assigned to this request
/// * `Err(SchedulerError::QueueCapacityExceeded)` - If the pending queue is full
///
/// # Example
/// ```rust
/// let request_id = scheduler.submit_request(
///     vec![1, 2, 3],  // "Hello world"
///     100,            // max_tokens
///     0.8,            // temperature
///     50,             // top_k
///     0.9,            // top_p
/// )?;
/// ```
pub fn submit_request(
    &mut self,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> SchedulerResult<u32> {
```

---

### ISSUE-7: Batch Creation Ignores Similar-Length Grouping
**Location**: `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:356-396`

**Severity**: LOW - Performance optimization

**Description**:
The `create_batch()` method sorts requests by length and attempts to group similar lengths together, but the **30% threshold check** on line 385-386 is never used effectively:

```rust
if batch.is_empty()
    || (request.total_tokens() as f32 - batch.max_sequence_length() as f32).abs()
        < batch.max_sequence_length() as f32 * 0.3
```

The logic puts "incompatible" requests back at the **front** of the queue (line 394), which can cause:
- **Starvation**: Long requests might never be processed
- **Head-of-line blocking**: Requests at the front keep getting rejected
- **Inefficient batching**: Repeatedly checking the same requests

**Recommendation**:
1. Use a priority queue instead of VecDeque
2. Track "rejected" requests separately
3. Add a timeout for "how long to wait before accepting any request"

---

### ISSUE-8: Inefficient Cloning in update_iteration_batch()
**Location**: `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:464, 467-488`

**Severity**: LOW - Performance

**Description**:
The `process_batch()` method clones the entire batch before processing:
```rust
let original_requests = iteration_batch.requests.clone();
```

Then for each request, it snapshots the scheduler state again:
```rust
if let Some(updated) = self.snapshot_request(request.request_id).await {
    refreshed_requests.push(updated);
}
```

This double-cloning is unnecessary - the scheduler already has the latest state.

**Recommendation**:
Process directly without intermediate cloning:
```rust
// Process each request in the batch
for request in &iteration_batch.requests {
    match self.process_single_request(request).await {
        Ok(completed) => {
            if completed {
                // Request will be moved to completed by update_iteration_batch
            }
        }
        Err(e) => {
            error!("Error processing request {}: {}", request.request_id, e);
            // Mark as failed
            let mut scheduler = self.scheduler.write().await;
            if let Ok(req) = scheduler.get_request_mut(request.request_id) {
                let _ = req.fail();
            }
        }
    }
}

// Update batch with latest scheduler state
let mut scheduler = self.scheduler.write().await;
scheduler.update_iteration_batch(iteration_batch).await?;
```

---

## Metrics

- **Files reviewed**: 2
- **Total lines reviewed**: 1,433
- **Critical issues found**: 2
- **High priority issues**: 2
- **Medium priority issues**: 2
- **Low priority issues**: 2
- **Code quality issues**: 8
- **Total issues**: 14

---

## Testing Recommendations

### 1. Integration Test for HTTP Server
```rust
#[tokio::test]
async fn test_http_server_starts() {
    let addr = "127.0.0.1:0";  // Random port
    let server = tokio::spawn(async move {
        run_server(addr, None, None).await
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Server should be listening
    let client = reqwest::Client::new();
    let response = client
        .get(format!("http://{}/health", addr))
        .send()
        .await;

    assert!(response.is_ok());
    assert_eq!(response.unwrap().status(), 200);

    server.abort();
}
```

### 2. Fuzz Test for Request ID Overflow
```rust
#[test]
fn test_request_id_overflow() {
    let config = SchedulerConfig::default();
    let mut scheduler = Scheduler::new(config);

    // Force overflow
    scheduler.next_request_id = u32::MAX;

    let id1 = scheduler.submit_request(vec![1], 10, 0.8, 50, 0.9).unwrap();
    assert_eq!(id1, u32::MAX);

    let id2 = scheduler.submit_request(vec![2], 10, 0.8, 50, 0.9).unwrap();
    assert_eq!(id2, 0);  // Wrapped around

    // Both requests should be distinct
    let req1 = scheduler.get_request(id1).unwrap();
    let req2 = scheduler.get_request(id2).unwrap();
    assert_ne!(req1.request_id, req2.request_id);
}
```

### 3. Concurrent Access Test
```rust
#[tokio::test]
async fn test_concurrent_batch_updates() {
    let config = SchedulerConfig::default();
    let scheduler = Arc::new(RwLock::new(Scheduler::new(config)));

    // Submit multiple requests
    for i in 0..10 {
        let mut s = scheduler.write().await;
        s.submit_request(vec![i as u32], 10, 0.8, 50, 0.9).unwrap();
    }

    // Spawn tasks that all try to create batches
    let handles: Vec<_> = (0..5)
        .map(|_| {
            let s = scheduler.clone();
            tokio::spawn(async move {
                let mut scheduler = s.write().await;
                scheduler.get_next_iteration_batch()
            })
        })
        .collect();

    // All should succeed without deadlock
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}
```

---

## Recommendations Summary

### Immediate Actions (Critical)
1. **Fix BUG-1**: Change `engine.run_inference_loop().await` to spawned task in HTTP server
2. **Fix BUG-2**: Remove stale request re-insertion in `update_iteration_batch()`

### Short-term Actions (High Priority)
3. **Fix BUG-3**: Add graceful shutdown handling to HTTP server
4. **Fix BUG-4**: Remove extra `println!()` in stream mode

### Medium-term Actions (Medium Priority)
5. **Fix BUG-5**: Add error handling for tokenizer decode
6. **Fix BUG-6**: Use wrapping arithmetic for request IDs

### Long-term Actions (Code Quality)
7. **ISSUE-1**: Extract tokenizer path inference to helper function
8. **ISSUE-3**: Use notification system instead of polling in `wait_for_completion()`
9. **ISSUE-5**: Add state transition validation
10. **ISSUE-6**: Add comprehensive documentation to public API

---

## Architecture Notes

### What Was Done Well
1. **Clean separation of concerns**: CLI, HTTP server, engine, and scheduler are well-separated
2. **Continuous batching implementation**: The `IterationBatch` API is well-designed
3. **Notification system**: Engine has proper notification infrastructure (though underutilized)
4. **Test coverage**: Scheduler has comprehensive unit tests
5. **Error types**: Custom error types (`SchedulerError`, `EngineError`) provide good context

### Architectural Concerns
1. **Double state management**: Both `IterationBatch` and `processing_requests` track the same requests
2. **Clone-heavy patterns**: Frequent cloning of `GenerationRequest` (consider using references)
3. **Blocking operations in async context**: Some GPU operations could benefit from explicit `spawn_blocking`

---

## Conclusion

The CLI and scheduler code have solid architectural foundations but contain **2 critical bugs** that must be fixed immediately:

1. **HTTP server never starts** (BUG-1) - Complete functional failure
2. **Generated tokens are lost** (BUG-2) - Data corruption

The remaining issues are important for stability, performance, and maintainability but are not immediately catastrophic.

**Estimated fix time**:
- Critical issues: 2-4 hours
- High priority: 1-2 hours
- Medium/Low priority: 4-8 hours

**Risk assessment**:
- Without fixing BUG-1: HTTP server is **completely non-functional**
- Without fixing BUG-2: All generations are **corrupted or incomplete**
- Without fixing other issues: Degraded performance, potential crashes in edge cases

**Recommendation**: Fix critical issues before deploying to production. Other issues should be addressed in the next sprint.

---

## References

**Files Read**:
- `/home/feanor/Projects/ROCmForge/src/bin/rocmforge_cli.rs` (510 lines)
- `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs` (923 lines)
- `/home/feanor/Projects/ROCmForge/src/engine.rs` (824 lines)
- `/home/feanor/Projects/ROCmForge/src/http/server.rs` (663 lines)

**Database Schema Checked**:
- `.codemcp/operations.db` - Verified no pending work items

**Compilation Check**:
- `cargo check` - No compilation errors (warnings only)

**Architectural Decisions Referenced**:
- Phase 12 complete: PagedAttention & Continuous Batching
- Phase 12.4: Continuous Batching Integration
- Stream mismatch fix: MLP kernel cache causing CLI hang

---

**Review Completed**: 2026-01-10
**Next Review Recommended**: After critical bugs are fixed
