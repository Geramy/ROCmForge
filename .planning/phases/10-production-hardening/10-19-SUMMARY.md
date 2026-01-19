# Task 10-19: Graceful Degradation for HTTP Error Handling

**Phase:** 10 (Production Hardening - Gap Closure)
**Task:** 10-19
**Date:** 2026-01-19
**Status:** Complete

## Overview

Implemented graceful degradation for recoverable errors in the HTTP server by mapping `RocmForgeError` categories to appropriate HTTP status codes. This provides better client-side error handling and retry behavior.

## Changes Made

### File: `src/http/server.rs`

#### 1. Added `HttpError` struct

```rust
pub struct HttpError {
    pub error: RocmForgeError,
    pub retry_after: Option<u32>,
}
```

- Wraps `RocmForgeError` with HTTP-specific handling
- Includes `retry_after` header value for recoverable errors
- Implements `From<RocmForgeError>` for easy conversion

#### 2. HTTP Status Code Mapping

| Error Category | HTTP Status | Retry-After |
|----------------|-------------|-------------|
| `User` | 400 Bad Request | No |
| `Model` | 400 Bad Request | No |
| `Recoverable` | 503 Service Unavailable | Yes (60s) |
| `Backend` | 503 Service Unavailable | Yes (60s) |
| `Internal` | 500 Internal Server Error | No |

#### 3. Enhanced Error Response Format

Error responses now include:
- `error`: Human-readable error message
- `category`: Error category (User/Model/Recoverable/Backend/Internal)
- `recoverable`: Boolean indicating if client should retry
- `status`: Always "error"
- `Retry-After`: Header set to 60 seconds for recoverable errors

#### 4. Legacy `ServerError` Compatibility

The existing `ServerError` enum was updated to delegate to `HttpError`, maintaining backward compatibility:

```rust
impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let http_error: HttpError = match self {
            ServerError::InvalidRequest(msg) => RocmForgeError::InvalidRequest(msg).into(),
            // ...
        };
        http_error.into_response()
    }
}
```

#### 5. Comprehensive Test Coverage

Added 20 new tests verifying:
- User errors return 400
- Model errors return 400
- Recoverable errors return 503 with Retry-After
- Backend errors return 503 with Retry-After
- Internal errors return 500
- Legacy `ServerError` mapping
- Response structure includes all required fields

## Acceptance Criteria

- [x] Recoverable errors return 503 with Retry-After
- [x] User errors return 400
- [x] Backend errors return 503
- [x] Tests verify status codes (20 tests passing)

## Test Results

```
test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 538 filtered out
```

## Example Error Responses

### 400 Bad Request (User Error)
```json
{
  "error": "Invalid temperature: 0. Must be > 0",
  "category": "User",
  "recoverable": false,
  "status": "error"
}
```

### 503 Service Unavailable (Capacity Limit)
```json
{
  "error": "KV cache capacity exceeded",
  "category": "Recoverable",
  "recoverable": true,
  "status": "error"
}
```
Headers: `Retry-After: 60`

### 500 Internal Server Error (Bug)
```json
{
  "error": "Internal error: unexpected state",
  "category": "Internal",
  "recoverable": false,
  "status": "error"
}
```

## Related Files

- `/home/feanor/Projects/ROCmForge/src/http/server.rs` - Main implementation
- `/home/feanor/Projects/ROCmForge/src/error.rs` - Error type definitions and categorization

## Next Steps

- Monitor error rates in production via metrics
- Consider adding rate limiting for 503 responses
- Document error handling for API consumers
