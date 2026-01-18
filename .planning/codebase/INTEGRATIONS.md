# External Integrations

**Analysis Date:** 2026-01-18

## APIs & External Services

**Payment Processing:**
- None detected

**Email/SMS:**
- None detected

**External APIs:**
- None detected (no OpenAI, Anthropic, Google, Azure integrations)

## Data Storage

**Databases:**
- None (uses in-memory state only)

**File Storage:**
- Local model files only (GGUF format)
- No cloud storage integration (S3, Azure Blob, etc.)

**Caching:**
- In-memory KV cache for inference (`src/kv_cache/`)
- No external cache (Redis, Memcached)

## Authentication & Identity

**Auth Provider:**
- None (HTTP server is open, no authentication)

**OAuth Integrations:**
- None

## Monitoring & Observability

**Error Tracking:**
- None (basic structured logging via tracing crate only)

**Analytics:**
- None

**Logs:**
- Structured logging via tracing crate
- stdout/stderr output only

## CI/CD & Deployment

**Hosting:**
- Not applicable (native binary distribution)

**CI Pipeline:**
- No GitHub Actions workflows detected for main project
- Some workflow files in `docs/examples/`

## Environment Configuration

**Development:**
- Required env vars: ROCMFORGE_GGUF, ROCMFORGE_TOKENIZER, ROCMFORGE_MODELS
- Secrets location: Not applicable (no secrets)
- Mock/stub services: GPU tests use panic-based skipping

**Staging:**
- Not applicable

**Production:**
- Secrets management: Not applicable (no secrets)
- Failover/redundancy: Not implemented

## Hardware Integration

**GPU Platform:**
- AMD ROCm/HIP - Primary GPU computing platform
- Location: `src/backend/hip_backend.rs`
- Status: Placeholder dependencies in `Cargo.toml` (hip-sys commented out)

**Model Loading:**
- GGUF format support - `src/loader/gguf.rs`
- Local files only - No HuggingFace Hub integration
- Tokenizer support - Embedded or local tokenizer.json

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- Server-Sent Events (SSE) for streaming inference responses

---

*Integration audit: 2026-01-18*
*Update when adding/removing external services*
