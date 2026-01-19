# ROCmForge Manual

Complete manual for ROCmForge - An AMD GPU LLM inference engine.

**Version:** 0.1.0
**Last Updated:** 2026-01-19

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [CLI Reference](#cli-reference)
5. [HTTP API Reference](#http-api-reference)
6. [Model Support](#model-support)
7. [Troubleshooting](#troubleshooting)
8. [Development](#development)

---

## Overview

ROCmForge is an LLM inference engine for AMD GPUs using ROCm and HIP. It loads GGUF-format models and provides an OpenAI-compatible HTTP API.

### Key Features

- **GGUF Support**: 15 quantization formats
- **Hybrid Execution**: Automatic CPU/GPU backend selection
- **CPU SIMD**: AVX2/NEON/AVX-512 support
- **HTTP API**: OpenAI-compatible completions endpoint
- **CLI Tools**: serve, generate, context management

### What ROCmForge Does

1. Loads GGUF-format models (llama.cpp ecosystem)
2. Executes inference on AMD GPU or CPU (with SIMD)
3. Serves completions via HTTP API
4. Provides CLI for text generation

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| ROCm | 5.0+ | Linux only |
| Rust | 1.82+ | For SIMD support |
| AMD GPU | RDNA2/3 or CDNA2/3 | With ROCm support |

### Build from Source

```bash
# Clone repository
git clone https://github.com/oldnordic/ROCmForge.git
cd ROCmForge

# Build release
cargo build --release

# Binary location
./target/release/rocmforge_cli
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `rocm` | ROCm/HIP support | Enabled |
| `simd` | CPU SIMD (requires nightly) | Optional |
| `avx512` | AVX-512 code paths | Optional |
| `context` | SQLiteGraph context engine | Optional |

```bash
# Build with all features
cargo build --release --features "rocm,simd,avx512,context"

# Build with context support
cargo build --release --features context
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCMFORGE_GGUF` | Path to GGUF model | - |
| `ROCMFORGE_TOKENIZER` | Path to tokenizer JSON | - |
| `ROCMFORGE_ADDR` | Server bind address | `127.0.0.1:8080` |
| `ROCMFORGE_LOG` | Log level (error, warn, info, debug, trace) | `info` |
| `ROCMFORGE_MAX_CONTEXT` | Maximum context length | `2048` |
| `ROCMFORGE_THREADS` | CPU threads for inference | CPU count |

### Example .env File

```bash
# Model
ROCMFORGE_GGUF=/path/to/model.gguf
ROCMFORGE_TOKENIZER=/path/to/tokenizer.json

# Server
ROCMFORGE_ADDR=0.0.0.0:8080

# Logging
ROCMFORGE_LOG=debug

# Inference
ROCMFORGE_MAX_CONTEXT=4096
ROCMFORGE_THREADS=8
```

---

## CLI Reference

### Global Options

```
rocmforge_cli [OPTIONS] <COMMAND>

OPTIONS:
    -h, --help       Print help
    -V, --version    Print version
```

### Commands

#### serve

Start the HTTP server.

```bash
rocmforge_cli serve [OPTIONS]

OPTIONS:
    --addr <ADDR>         Bind address [default: 127.0.0.1:8080]
    --gguf <GGUF>         Path to GGUF model
    --tokenizer <TOKENIZER> Path to tokenizer JSON
```

**Example:**
```bash
rocmforge_cli serve --addr 0.0.0.0:8080 --gguf model.gguf
```

#### generate

Generate text from a prompt.

```bash
rocmforge_cli generate [OPTIONS] --prompt <PROMPT>

OPTIONS:
    --gguf <GGUF>              Path to GGUF model
    --max-tokens <N>           Max tokens to generate [default: 50]
    --temperature <N>          Sampling temperature [default: 0.8]
    --top-p <N>                Top-p sampling [default: 0.9]
    --top-k <N>                Top-k sampling [default: 40]
    --prompt <PROMPT>          Input prompt
```

**Example:**
```bash
rocmforge_cli generate --gguf model.gguf --prompt "Hello, world" --max-tokens 100
```

#### context

Manage semantic context (requires `--features context`).

```bash
rocmforge_cli context <COMMAND>

COMMANDS:
    add <TEXT>      Add message to context store
    search <QUERY>   Search context by similarity
    list             List all messages
    clear            Clear all messages
```

**Example:**
```bash
rocmforge_cli context add "Your important context here"
rocmforge_cli context search "relevant keywords"
```

---

## HTTP API Reference

### Base URL

```
http://localhost:8080
```

### Endpoints

#### POST /v1/completions

Generate text completions.

**Request:**
```json
{
  "model": "model-name",
  "prompt": "Hello, world",
  "max_tokens": 50,
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40
}
```

**Response:**
```json
{
  "id": "cmpl-123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "text": "Generated text here",
    "index": 0,
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 50,
    "total_tokens": 53
  }
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

#### GET /ready

Readiness probe endpoint.

**Response:**
```json
{
  "ready": true
}
```

#### GET /metrics

Prometheus-style metrics endpoint.

---

## Model Support

### Supported Architectures

- LLaMA / LLaMA 2 / LLaMA 3
- Qwen / Qwen 2
- Mistral
- Yi
- Mixtral (MoE)
- GLM

### Supported Quantization Formats

| Format | Bits | Description |
|--------|------|-------------|
| F32 | 32 | Full precision |
| F16 | 16 | Half precision |
| Q4_0 | 4 | Block-scale (32 elements) |
| Q4_1 | 4 | Block-scale with min |
| Q5_0 | 5 | Block-scale |
| Q5_1 | 5 | Block-scale with min |
| Q8_0 | 8 | Block-scale |
| Q2_K | 2 | K-quantization |
| Q3_K | 3 | K-quantization |
| Q4_K | 4 | K-quantization (super-block) |
| Q5_K | 5 | K-quantization (super-block) |
| Q6_K | 6 | K-quantization (super-block) |
| MXFP4 | 4 | OCP MX Spec v1.0 |
| MXFP6 | 6 | OCP MX Spec v1.0 |

---

## Troubleshooting

### Common Issues

#### "ROCm device not found"

**Cause:** ROCm not properly installed or GPU not visible.

**Solution:**
```bash
# Check ROCm installation
rocminfo | head -20

# Check GPU visibility
rocm-smi

# Verify device nodes
ls /dev/kfd
ls /dev/dri/
```

#### "Model loading failed"

**Cause:** Unsupported quantization format or corrupted GGUF file.

**Solution:**
```bash
# Verify GGUF file
python3 -c "
import gguf
reader = gguf.GGUFReader('model.gguf')
print('Tensor count:', len(reader.tensors))
print('Metadata:', reader.fields.keys())
"
```

#### "Out of memory"

**Cause:** Model too large for available VRAM.

**Solution:**
- Use a smaller model
- Use more aggressive quantization (Q4_K instead of Q8_0)
- Reduce `ROCMFORGE_MAX_CONTEXT`
- Close other GPU applications

### Debug Logging

Enable trace logging:

```bash
RUST_LOG=trace ./target/release/rocmforge_cli serve --gguf model.gguf
```

---

## Development

### Running Tests

```bash
# Run all lib tests
cargo test --lib

# Run with ROCm feature
cargo test --lib --features rocm

# Run specific module
cargo test --lib attention
cargo test --lib backend::cpu

# Run SIMD tests (requires nightly)
cargo test --lib --features simd
```

### Code Organization

```
src/
├── attention/       # Attention implementations
├── backend/         # CPU and GPU backends
│   ├── cpu/         # CPU backend with SIMD
│   └── hip_backend/ # HIP/ROCm GPU backend
├── context/         # Context engine (feature-gated)
├── ggml/            # GGML IR execution
├── http/            # HTTP server
├── loader/          # GGUF loader
├── model/           # Model configs
├── scheduler/       # Request scheduler
└── tensor/          # Tensor types
```

### Adding Features

1. **New quantization format:**
   - Add dequantization in `src/loader/dequant.rs`
   - Add GPU kernel in `kernels/`
   - Add to build.rs
   - Add tests

2. **New architecture support:**
   - Add to `src/model/config.rs`
   - Add tensor mapping in `src/loader/gguf.rs`

3. **New HTTP endpoint:**
   - Add handler in `src/http/server.rs`
   - Update documentation

---

## License

GPL-3.0

---

**Manual Version:** 0.1.0
**Last Updated:** 2026-01-19
