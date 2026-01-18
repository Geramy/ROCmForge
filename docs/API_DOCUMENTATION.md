# ROCmForge HTTP API Documentation

**Version:** 0.1.0
**Last Updated:** 2026-01-19

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Response Formats](#common-response-formats)
4. [Error Responses](#error-responses)
5. [Endpoints](#endpoints)
   - [POST /generate](#post-generate)
   - [POST /generate/stream](#post-generatestream)
   - [GET /status/:request_id](#get-statusrequest_id)
   - [POST /cancel/:request_id](#post-cancelrequest_id)
   - [GET /models](#get-models)
   - [GET /health](#get-health)
   - [GET /ready](#get-ready)
   - [GET /metrics](#get-metrics)
   - [GET /traces](#get-traces)

---

## Overview

ROCmForge provides a RESTful HTTP API for text generation using large language models running on AMD GPUs via ROCm. The server supports both synchronous and streaming generation, request management, and monitoring endpoints.

### Base URL

```
http://localhost:8080
```

The default port is `8080` but can be configured when starting the server.

### CORS

The server enables CORS (Cross-Origin Resource Sharing) for all origins and headers by default.

---

## Authentication

**Status:** Not Implemented

Currently, ROCmForge does not implement authentication. The server should be run behind a reverse proxy (e.g., nginx) with authentication in production environments.

See [Deployment Guide](DEPLOYMENT.md) for security recommendations.

---

## Common Response Formats

### Success Response

Most endpoints return JSON responses with a `200` status code on success:

```json
{
  "field": "value"
}
```

### SSE Streaming

Streaming endpoints use Server-Sent Events (SSE) with `Content-Type: text/event-stream`.

---

## Error Responses

### Error Response Schema

All errors return a JSON response with an `error` field:

```json
{
  "error": "Error message description",
  "status": "error"
}
```

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| `200` | Success |
| `400` | Bad Request - Invalid request parameters |
| `404` | Not Found - Request ID not found |
| `500` | Internal Server Error - Generation or server error |
| `503` | Service Unavailable - Engine not ready |

### Error Types

| Error | Status Code | Description |
|-------|-------------|-------------|
| `InvalidRequest` | 400 | Request parameters are invalid or missing |
| `RequestNotFound` | 404 | The specified request_id does not exist |
| `GenerationFailed` | 500 | Text generation failed |
| `InternalError` | 500 | Server internal error (e.g., engine not initialized) |

---

## Endpoints

### POST /generate

Generate text synchronously. The request will block until generation is complete or the maximum token limit is reached.

**Endpoint:** `POST /generate`

**Request Body:**

```json
{
  "prompt": "string (required)",
  "max_tokens": "number (optional, default: 100)",
  "temperature": "number (optional, default: 1.0, range: 0.0-2.0)",
  "top_k": "number (optional, default: 50)",
  "top_p": "number (optional, default: 0.9, range: 0.0-1.0)",
  "stream": "boolean (optional, ignored for non-streaming endpoint)"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | The input text prompt for generation |
| `max_tokens` | number | No | Maximum tokens to generate (default: 100) |
| `temperature` | number | No | Sampling temperature (default: 1.0) |
| `top_k` | number | No | Top-k sampling parameter (default: 50) |
| `top_p` | number | No | Nucleus sampling parameter (default: 0.9) |
| `stream` | boolean | No | Ignored for this endpoint (use `/generate/stream`) |

**Response:** `200 OK`

```json
{
  "request_id": 123,
  "text": "generated text here",
  "tokens": [15043, 862, 338, 273],
  "finished": true,
  "finish_reason": "length"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | number | Unique identifier for this generation request |
| `text` | string | The complete generated text |
| `tokens` | array | Array of generated token IDs |
| `finished` | boolean | Whether generation is complete |
| `finish_reason` | string/null | Reason for completion: `length`, `stop`, `eos`, `cancelled`, `failed` |

**cURL Example:**

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Response:**

```json
{
  "request_id": 1,
  "text": " Paris, which is also the largest city in the country.",
  "tokens": [3849, 15, 487, 315, 322, 287, 3771, 315, 296, 1176, 13],
  "finished": true,
  "finish_reason": "length"
}
```

---

### POST /generate/stream

Generate text using Server-Sent Events (SSE). Tokens are streamed as they are generated.

**Endpoint:** `POST /generate/stream`

**Request Body:** Same as `POST /generate`

**Response:** `200 OK` with `Content-Type: text/event-stream`

**SSE Event Format:**

Each SSE event contains a JSON object with the following fields:

```json
{
  "request_id": 123,
  "token": 15043,
  "text": " Paris",
  "finished": false,
  "finish_reason": null
}
```

**SSE Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | number | Unique identifier for this generation request |
| `token` | number | The token ID generated in this event |
| `text` | string | The decoded text for this token |
| `finished` | boolean | Whether generation is complete |
| `finish_reason` | string/null | Reason for completion (present only when finished) |

**cURL Example:**

```bash
curl -N -X POST http://localhost:8080/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about programming:",
    "max_tokens": 30,
    "temperature": 0.8
  }'
```

**Response (SSE stream):**

```
data: {"request_id":2,"token":478,"text":" Code","finished":false,"finish_reason":null}

data: {"request_id":2,"token":338,"text":" flows","finished":false,"finish_reason":null}

data: {"request_id":2,"token":273,"text":" like","finished":false,"finish_reason":null}

data: {"request_id":2,"token":1305,"text":" water","finished":false,"finish_reason":null}

data: {"request_id":2,"token":13,"text":".","finished":true,"finish_reason":"length"}

```

**Python Example:**

```python
import requests
import json

response = requests.post(
    "http://localhost:8080/generate/stream",
    json={
        "prompt": "Explain quantum computing:",
        "max_tokens": 100,
        "temperature": 0.7
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        data = json.loads(line[6:])
        print(data["text"], end="", flush=True)
        if data["finished"]:
            print(f"\n\nFinish reason: {data['finish_reason']}")
```

---

### GET /status/:request_id

Get the current status of a generation request.

**Endpoint:** `GET /status/:request_id`

**URL Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `request_id` | number | Yes | The request ID to check |

**Response:** `200 OK`

```json
{
  "request_id": 123,
  "text": "partially generated text",
  "tokens": [15043, 862, 338],
  "finished": false,
  "finish_reason": null
}
```

**cURL Example:**

```bash
curl http://localhost:8080/status/123
```

**Response:**

```json
{
  "request_id": 123,
  "text": "The quick brown fox jumps over the lazy",
  "tokens": [464, 3686, 8156, 21831, 18045, 346, 30184, 1704, 263],
  "finished": false,
  "finish_reason": null
}
```

**Error Responses:**

- `404 Not Found` - Request ID does not exist
- `500 Internal Server Error` - Engine not initialized

---

### POST /cancel/:request_id

Cancel an in-progress generation request.

**Endpoint:** `POST /cancel/:request_id`

**URL Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `request_id` | number | Yes | The request ID to cancel |

**Response:** `200 OK`

```json
{
  "request_id": 123,
  "text": "partially generated text",
  "tokens": [15043, 862, 338],
  "finished": true,
  "finish_reason": "cancelled"
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8080/cancel/123
```

**Response:**

```json
{
  "request_id": 123,
  "text": "The quick brown fox",
  "tokens": [464, 3686, 8156, 21831],
  "finished": true,
  "finish_reason": "cancelled"
}
```

**Error Responses:**

- `404 Not Found` - Request ID does not exist

---

### GET /models

Get information about available models and tokenizer cache statistics.

**Endpoint:** `GET /models`

**Response:** `200 OK`

```json
{
  "models": [
    {
      "path": "/path/to/model.gguf",
      "size": 4294967296,
      "metadata": {
        "general.architecture": "llama",
        "general.file_type": 3,
        "llama.context_length": 2048,
        "llama.embedding_length": 4096,
        "llama.block_count": 32,
        "llama.attention.head_count": 32
      }
    }
  ],
  "tokenizer_cache": {
    "hits": 42,
    "misses": 1,
    "bytes": 1234567
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `models` | array | List of discovered GGUF models |
| `models[].path` | string | Absolute path to the model file |
| `models[].size` | number | File size in bytes |
| `models[].metadata` | object | GGUF metadata key-value pairs |
| `tokenizer_cache.hits` | number | Tokenizer cache hits |
| `tokenizer_cache.misses` | number | Tokenizer cache misses |
| `tokenizer_cache.bytes` | number | Tokenizer cache size in bytes |

**cURL Example:**

```bash
curl http://localhost:8080/models
```

**Response:**

```json
{
  "models": [
    {
      "path": "/models/llama-2-7b.gguf",
      "size": 4100000000,
      "metadata": {
        "general.architecture": "llama",
        "general.file_type": 3,
        "llama.context_length": 2048,
        "llama.embedding_length": 4096,
        "llama.block_count": 32,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 32
      }
    }
  ],
  "tokenizer_cache": {
    "hits": 15,
    "misses": 0,
    "bytes": 2048000
  }
}
```

---

### GET /health

Get detailed health status of the server and engine.

**Endpoint:** `GET /health`

**Response:** `200 OK`

```json
{
  "status": "healthy",
  "service": "rocmforge",
  "version": "0.1.0",
  "checks": {
    "engine": {
      "running": true,
      "model_loaded": true
    },
    "gpu": {
      "available": true,
      "memory": {
        "free_bytes": 17179869184,
        "total_bytes": 17179869184,
        "free_mb": 16384,
        "total_mb": 16384,
        "used_mb": 0,
        "utilization_percent": 0
      }
    },
    "requests": {
      "active": 0,
      "queued": 0
    },
    "cache": {
      "pages_used": 0,
      "pages_total": 1000,
      "pages_free": 1000,
      "active_sequences": 0
    }
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Overall status: `healthy`, `unhealthy` |
| `service` | string | Service name (always "rocmforge") |
| `version` | string | Server version |
| `checks.engine` | object | Engine status |
| `checks.engine.running` | boolean | Whether engine is running |
| `checks.engine.model_loaded` | boolean | Whether model is loaded |
| `checks.gpu` | object | GPU status |
| `checks.gpu.available` | boolean | Whether GPU is available |
| `checks.gpu.memory` | object | GPU memory information |
| `checks.requests` | object | Request queue information |
| `checks.cache` | object | KV cache information |

**cURL Example:**

```bash
curl http://localhost:8080/health
```

**Response (healthy):**

```json
{
  "status": "healthy",
  "service": "rocmforge",
  "version": "0.1.0",
  "checks": {
    "engine": {
      "running": true,
      "model_loaded": true
    },
    "gpu": {
      "available": true,
      "memory": {
        "free_bytes": 12079595520,
        "total_bytes": 17179869184,
        "free_mb": 11520,
        "total_mb": 16384,
        "used_mb": 4864,
        "utilization_percent": 29
      }
    },
    "requests": {
      "active": 2,
      "queued": 1
    },
    "cache": {
      "pages_used": 128,
      "pages_total": 1000,
      "pages_free": 872,
      "active_sequences": 2
    }
  }
}
```

**Response (unhealthy - no engine):**

```json
{
  "status": "unhealthy",
  "service": "rocmforge",
  "version": "0.1.0",
  "checks": {
    "engine": {
      "running": false,
      "model_loaded": false
    },
    "gpu": {
      "available": false
    }
  }
}
```

---

### GET /ready

Readiness probe for container orchestration (Kubernetes). Returns 200 when the engine is ready to accept requests, 503 otherwise.

**Endpoint:** `GET /ready`

**Response:** `200 OK` when ready

```json
{
  "ready": true,
  "service": "rocmforge"
}
```

**Response:** `503 Service Unavailable` when not ready

```json
{
  "error": "Service Unavailable",
  "status": "error"
}
```

**cURL Example:**

```bash
curl -w "\nHTTP Status: %{http_code}\n" http://localhost:8080/ready
```

**Response (ready):**

```json
{
  "ready": true,
  "service": "rocmforge"
}
```

**Response (not ready):**

```
HTTP Status: 503
{"error":"Service Unavailable","status":"error"}
```

**Kubernetes Usage:**

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

---

### GET /metrics

Get Prometheus-compatible metrics for monitoring and alerting.

**Endpoint:** `GET /metrics`

**Response:** `200 OK` with `Content-Type: text/plain`

**Format:** Prometheus exposition format

**cURL Example:**

```bash
curl http://localhost:8080/metrics
```

**Response:**

```
# HELP rocmforge_requests_started_total Total number of inference requests started
# TYPE rocmforge_requests_started_total counter
rocmforge_requests_started_total 42

# HELP rocmforge_requests_completed_total Total number of inference requests completed
# TYPE rocmforge_requests_completed_total counter
rocmforge_requests_completed_total 38

# HELP rocmforge_requests_failed_total Total number of inference requests failed
# TYPE rocmforge_requests_failed_total counter
rocmforge_requests_failed_total 2

# HELP rocmforge_requests_cancelled_total Total number of inference requests cancelled
# TYPE rocmforge_requests_cancelled_total counter
rocmforge_requests_cancelled_total 2

# HELP rocmforge_tokens_generated_total Total number of tokens generated
# TYPE rocmforge_tokens_generated_total counter
rocmforge_tokens_generated_total 15234

# HELP rocmforge_prefill_duration_seconds Prefill phase duration in seconds
# TYPE rocmforge_prefill_duration_seconds histogram
rocmforge_prefill_duration_seconds_bucket{le="0.001"} 0
rocmforge_prefill_duration_seconds_bucket{le="0.01"} 5
rocmforge_prefill_duration_seconds_bucket{le="0.1"} 38
rocmforge_prefill_duration_seconds_bucket{le="1"} 42
rocmforge_prefill_duration_seconds_bucket{le="10"} 42
rocmforge_prefill_duration_seconds_bucket{le="100"} 42
rocmforge_prefill_duration_seconds_bucket{le="+Inf"} 42
rocmforge_prefill_duration_seconds_sum 12.345
rocmforge_prefill_duration_seconds_count 42

# HELP rocmforge_decode_duration_seconds Decode phase duration in seconds
# TYPE rocmforge_decode_duration_seconds histogram
rocmforge_decode_duration_seconds_bucket{le="0.001"} 0
rocmforge_decode_duration_seconds_bucket{le="0.01"} 12
rocmforge_decode_duration_seconds_bucket{le="0.1"} 38
rocmforge_decode_duration_seconds_bucket{le="1"} 42
rocmforge_decode_duration_seconds_bucket{le="10"} 42
rocmforge_decode_duration_seconds_bucket{le="100"} 42
rocmforge_decode_duration_seconds_bucket{le="+Inf"} 42
rocmforge_decode_duration_seconds_sum 8.765
rocmforge_decode_duration_seconds_count 42

# HELP rocmforge_total_duration_seconds Total inference duration in seconds
# TYPE rocmforge_total_duration_seconds histogram
rocmforge_total_duration_seconds_bucket{le="0.001"} 0
rocmforge_total_duration_seconds_bucket{le="0.01"} 0
rocmforge_total_duration_seconds_bucket{le="0.1"} 8
rocmforge_total_duration_seconds_bucket{le="1"} 35
rocmforge_total_duration_seconds_bucket{le="10"} 42
rocmforge_total_duration_seconds_bucket{le="100"} 42
rocmforge_total_duration_seconds_bucket{le="+Inf"} 42
rocmforge_total_duration_seconds_sum 45.678
rocmforge_total_duration_seconds_count 42

# HELP rocmforge_queue_length Current number of requests in queue
# TYPE rocmforge_queue_length gauge
rocmforge_queue_length 3

# HELP rocmforge_active_requests Current number of active requests
# TYPE rocmforge_active_requests gauge
rocmforge_active_requests 2

# HELP rocmforge_ttft_seconds Time to first token in seconds
# TYPE rocmforge_ttft_seconds histogram
rocmforge_ttft_seconds_bucket{le="0.001"} 0
rocmforge_ttft_seconds_bucket{le="0.01"} 0
rocmforge_ttft_seconds_bucket{le="0.1"} 28
rocmforge_ttft_seconds_bucket{le="1"} 42
rocmforge_ttft_seconds_bucket{le="10"} 42
rocmforge_ttft_seconds_bucket{le="100"} 42
rocmforge_ttft_seconds_bucket{le="+Inf"} 42
rocmforge_ttft_seconds_sum 6.543
rocmforge_ttft_seconds_count 42

# HELP rocmforge_tokens_per_second Tokens generated per second
# TYPE rocmforge_tokens_per_second gauge
rocmforge_tokens_per_second 32.5
```

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `rocmforge_requests_started_total` | Counter | Total requests started |
| `rocmforge_requests_completed_total` | Counter | Total requests completed successfully |
| `rocmforge_requests_failed_total` | Counter | Total requests that failed |
| `rocmforge_requests_cancelled_total` | Counter | Total requests cancelled |
| `rocmforge_tokens_generated_total` | Counter | Total tokens generated across all requests |
| `rocmforge_prefill_duration_seconds` | Histogram | Prefill phase duration |
| `rocmforge_decode_duration_seconds` | Histogram | Decode phase duration |
| `rocmforge_total_duration_seconds` | Histogram | Total inference duration |
| `rocmforge_ttft_seconds` | Histogram | Time to first token |
| `rocmforge_queue_length` | Gauge | Current queue length |
| `rocmforge_active_requests` | Gauge | Current active requests |
| `rocmforge_tokens_per_second` | Gauge | Current throughput |

**Grafana Dashboard Example:**

```promql
# Request success rate
sum(rate(rocmforge_requests_completed_total[5m])) / sum(rate(rocmforge_requests_started_total[5m]))

# Average tokens per second
rate(rocmforge_tokens_generated_total[5m]) / sum(rate(rocmforge_requests_completed_total[5m]))

# P95 time to first token
histogram_quantile(0.95, sum(rate(rocmforge_ttft_seconds_bucket[5m])) by (le))

# Current utilization
rocmforge_active_requests / rocmforge_queue_length
```

---

### GET /traces

Get OpenTelemetry traces in OTLP JSON format for distributed tracing.

**Endpoint:** `GET /traces`

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | number | No | Maximum number of traces to return (default: all) |
| `clear` | boolean | No | If true, clears traces after returning them |

**Response:** `200 OK`

```json
{
  "resource_spans": [
    {
      "resource": {
        "attributes": [
          {
            "key": "service.name",
            "value": { "stringValue": "rocmforge" }
          }
        ]
      },
      "scope_spans": [
        {
          "scope": {
            "name": "rocmforge.engine"
          },
          "spans": [
            {
              "trace_id": "AAEBZLx5hK7j9L8qwNQBgA==",
              "span_id": "ZHkKAGKJZiE=",
              "parent_span_id": "ZHkKAGKJZiA=",
              "name": "inference_request",
              "kind": 2,
              "start_time_unix_nano": 1705689600000000000,
              "end_time_unix_nano": 1705689601000000000,
              "attributes": [
                {
                  "key": "request_id",
                  "value": { "intValue": "123" }
                },
                {
                  "key": "token_count",
                  "value": { "intValue": "50" }
                }
              ],
              "status": {
                "status": 1
              }
            }
          ]
        }
      ]
    }
  ]
}
```

**cURL Examples:**

Get all traces:

```bash
curl http://localhost:8080/traces
```

Get last 10 traces:

```bash
curl "http://localhost:8080/traces?limit=10"
```

Get traces and clear them:

```bash
curl "http://localhost:8080/traces?clear=true"
```

---

## Sampling Parameters

### Temperature

Controls randomness in generation. Lower values make output more deterministic.

- **Range:** 0.0 to 2.0
- **Default:** 1.0
- **Typical values:**
  - `0.1-0.3`: Very focused, deterministic output
  - `0.7-0.9`: Balanced creativity and coherence (recommended)
  - `1.0-1.5`: More creative, diverse output
  - `1.5-2.0`: Very creative, may be less coherent

### Top-K (top_k)

Limits sampling to the K most likely tokens.

- **Range:** 1 to vocabulary size
- **Default:** 50
- **Recommendation:** 40-50 for most use cases

### Top-P (top_p / Nucleus Sampling)

Limits sampling to the smallest set of tokens whose cumulative probability exceeds P.

- **Range:** 0.0 to 1.0
- **Default:** 0.9
- **Recommendation:** 0.9-0.95 for most use cases

---

## Complete Example

### Python Client

```python
import requests
import json

class ROCmForgeClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url

    def generate(self, prompt, max_tokens=100, temperature=1.0, top_k=50, top_p=0.9):
        """Generate text synchronously."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            }
        )
        response.raise_for_status()
        return response.json()

    def generate_stream(self, prompt, max_tokens=100, temperature=1.0, top_k=50, top_p=0.9):
        """Generate text with streaming."""
        response = requests.post(
            f"{self.base_url}/generate/stream",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            },
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line.startswith(b"data: "):
                yield json.loads(line[6:])

    def get_status(self, request_id):
        """Get the status of a request."""
        response = requests.get(f"{self.base_url}/status/{request_id}")
        response.raise_for_status()
        return response.json()

    def cancel(self, request_id):
        """Cancel a request."""
        response = requests.post(f"{self.base_url}/cancel/{request_id}")
        response.raise_for_status()
        return response.json()

    def health(self):
        """Get server health status."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def ready(self):
        """Check if server is ready."""
        response = requests.get(f"{self.base_url}/ready")
        return response.ok

    def models(self):
        """Get available models."""
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

# Usage example
client = ROCmForgeClient()

# Synchronous generation
result = client.generate("The future of AI is", max_tokens=50, temperature=0.8)
print(result["text"])

# Streaming generation
for event in client.generate_stream("Write a poem:", max_tokens=100):
    print(event["text"], end="", flush=True)
    if event["finished"]:
        print()
```

### JavaScript Client

```javascript
class ROCmForgeClient {
  constructor(baseUrl = 'http://localhost:8080') {
    this.baseUrl = baseUrl;
  }

  async generate(prompt, options = {}) {
    const response = await fetch(`${this.baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: options.maxTokens || 100,
        temperature: options.temperature || 1.0,
        top_k: options.topK || 50,
        top_p: options.topP || 0.9
      })
    });
    return await response.json();
  }

  async *generateStream(prompt, options = {}) {
    const response = await fetch(`${this.baseUrl}/generate/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: options.maxTokens || 100,
        temperature: options.temperature || 1.0,
        top_k: options.topK || 50,
        top_p: options.topP || 0.9
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          yield JSON.parse(line.slice(6));
        }
      }
    }
  }

  async getStatus(requestId) {
    const response = await fetch(`${this.baseUrl}/status/${requestId}`);
    return await response.json();
  }

  async cancel(requestId) {
    const response = await fetch(`${this.baseUrl}/cancel/${requestId}`, {
      method: 'POST'
    });
    return await response.json();
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return await response.json();
  }

  async ready() {
    const response = await fetch(`${this.baseUrl}/ready`);
    return response.ok;
  }

  async models() {
    const response = await fetch(`${this.baseUrl}/models`);
    return await response.json();
  }
}

// Usage example
const client = new ROCmForgeClient();

// Streaming generation
async function main() {
  for await (const event of client.generateStream("Tell me a joke:", { maxTokens: 50 })) {
    process.stdout.write(event.text);
    if (event.finished) {
      console.log('\n');
      console.log('Finish reason:', event.finish_reason);
    }
  }
}

main();
```

---

## Rate Limiting

**Status:** Not Implemented

Currently, ROCmForge does not enforce rate limiting. Rate limiting should be implemented at the reverse proxy level in production deployments.

---

## WebSocket Support

**Status:** Not Implemented

ROCmForge uses Server-Sent Events (SSE) for streaming instead of WebSockets. SSE is simpler, works with HTTP/1.1, and is sufficient for unidirectional streaming from server to client.

---

## Versioning

The API is currently at version **0.1.0** and may change. Future versions will implement API versioning via URL prefixes (e.g., `/v1/generate`).

---

## Support

For issues, questions, or contributions, please see the main project repository.
