# ROCmForge User Manual

**AMD GPU Inference Engine for Large Language Models**

Version 0.1.0 | Phase 11 Complete | Last Updated: January 2026

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Model Loading](#model-loading)
4. [CLI Usage](#cli-usage)
5. [HTTP Server](#http-server)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)

---

## Installation

### Prerequisites

| Requirement | Minimum Version | Recommended |
|-------------|-----------------|-------------|
| Rust | 1.70+ | Latest stable |
| ROCm | 5.x | 7.1+ |
| AMD GPU | RDNA2+ | RX 7900 XT or better |
| RAM | 16GB | 32GB+ |
| OS | Linux | Ubuntu 22.04, Arch |

### Step 1: Install ROCm

```bash
# Ubuntu/Debian
wget https://repo.radeon.com/rocm/rocm.gpg.key
sudo apt-key add rocm.gpg.key
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.1 ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-hip-sdk rocm-dev

# Arch Linux
sudo pacman -S hip-sdk rocm-hip-runtime rocm-core
```

### Step 2: Verify ROCm Installation

```bash
# Check ROCm version
rocminfo | grep "ROCm Version"

# Check GPU detection
rocm-smi

# Verify hipcc
hipcc --version
```

### Step 3: Build ROCmForge

```bash
# Clone repository
git clone https://github.com/your-repo/ROCmForge.git
cd ROCmForge

# Build release binary
cargo build --release

# Run tests (requires AMD GPU)
cargo test --features rocm --lib -- --test-threads=1
```

### Step 4: Add to PATH (Optional)

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export PATH="$PATH:/path/to/ROCmForge/target/release"' >> ~/.bashrc
source ~/.bashrc
```

---

## Quick Start

### 1. Download a GGUF Model

```bash
# Create models directory
mkdir -p ~/.config/rocmforge/models

# Download a model (example: Qwen2.5 0.5B)
huggingface-cli download TheBloke/Qwen2.5-0.5B-GGUF \
  qwen2.5-0.5b.Q4_K_M.gguf \
  --local-dir ~/.config/rocmforge/models
```

### 2. Generate Text (CLI)

```bash
rocmforge_cli generate \
  --gguf ~/.config/rocmforge/models/qwen2.5-0.5b.Q4_K_M.gguf \
  --prompt "The future of AI is" \
  --max-tokens 50 \
  --temperature 0.7
```

### 3. Start HTTP Server

```bash
rocmforge_cli serve --port 8080

# In another terminal, test it
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "prompt": "Explain quantum computing",
    "max_tokens": 100
  }'
```

---

## Model Loading

### Supported Formats

ROCmForge supports GGUF format with the following tensor types:

| Category | Tensor Types | Notes |
|----------|--------------|-------|
| Floating Point | F32, F16 | Full precision |
| Quantized (4-bit) | Q4_0, Q4_1, Q4_K | 4-bit quantization |
| Quantized (5-bit) | Q5_0, Q5_1, Q5_K | 5-bit quantization |
| Quantized (8-bit) | Q8_0 | 8-bit quantization |
| MXFP | MXFP4, MXFP6 | AMD-optimized (OCP MX Spec v1.0) |

### Supported Architectures

- Qwen2 / Qwen2.5
- LLaMA / LLaMA 2 / LLaMA 3
- Mistral
- GLM
- And more (auto-detected from GGUF metadata)

### Model Inspection

```bash
# View model metadata
rocmforge_cli inspect --model /path/to/model.gguf

# Output includes:
# - Architecture type
# - Tensor count and types
# - Vocabulary size
# - Layer count
# - Memory requirements
```

---

## CLI Usage

### Commands

| Command | Description |
|---------|-------------|
| `generate` | Generate text from a prompt |
| `serve` | Start HTTP API server |
| `inspect` | View GGUF model metadata |

### Generate Options

```bash
rocmforge_cli generate [OPTIONS]

OPTIONS:
    --gguf <PATH>              Path to GGUF model file
    --prompt <TEXT>            Input prompt
    --max-tokens <N>           Maximum tokens to generate [default: 50]
    --temperature <0.0-2.0>    Sampling temperature [default: 0.7]
    --top-p <0.0-1.0>          Nucleus sampling threshold [default: 0.9]
    --top-k <N>                Top-k sampling [default: 40]
    --seed <N>                 Random seed for reproducibility
```

### Examples

```bash
# Creative writing (high temperature)
rocmforge_cli generate \
  --gguf model.gguf \
  --prompt "Write a short story about" \
  --temperature 1.2 \
  --max-tokens 200

# Factual responses (low temperature)
rocmforge_cli generate \
  --gguf model.gguf \
  --prompt "What is the capital of France?" \
  --temperature 0.1 \
  --max-tokens 50

# Reproducible output
rocmforge_cli generate \
  --gguf model.gguf \
  --prompt "Hello world" \
  --seed 42 \
  --max-tokens 20
```

---

## HTTP Server

### Starting the Server

```bash
rocmforge_cli serve [OPTIONS]

OPTIONS:
    --port <N>           HTTP port [default: 8080]
    --host <IP>          Bind address [default: 0.0.0.0]
    --gguf <PATH>        Path to model file (optional, can load dynamically)
```

### API Endpoints

#### Health Check

```bash
GET /health

Response:
{"status": "ok", "model": "loaded"}
```

#### Completions (OpenAI-Compatible)

```bash
POST /v1/completions

{
  "model": "model-name",
  "prompt": "Your prompt here",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40
}

Response:
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
    "prompt_tokens": 5,
    "completion_tokens": 50,
    "total_tokens": 55
  }
}
```

#### Streaming Completions (SSE)

```bash
POST /v1/completions
Content-Type: application/json

{
  "model": "model-name",
  "prompt": "Write a poem",
  "stream": true
}

# Response: Server-Sent Events (text/event-stream)
data: {"id": "cmpl-123", "choices": [{"text": "Once"}]}
data: {"id": "cmpl-123", "choices": [{"text": " upon"}]}
data: {"id": "cmpl-123", "choices": [{"text": " a"}]}
...
data: [DONE]
```

### Python Client Example

```python
import requests

url = "http://localhost:8080/v1/completions"
data = {
    "model": "qwen2.5-0.5b",
    "prompt": "Explain recursion",
    "max_tokens": 100
}

response = requests.post(url, json=data)
print(response.json()["choices"][0]["text"])
```

### cURL Examples

```bash
# Simple completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "prompt": "Count to 10",
    "max_tokens": 50
  }'

# Streaming
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "prompt": "Write code",
    "stream": true
  }'
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCMFORGE_MODEL_PATH` | Default model directory | `~/.config/rocmforge/models` |
| `ROCMFORGE_GPU_MEMORY` | GPU memory limit (GB) | Auto-detect |
| `ROCMFORCE_LOG_LEVEL` | Logging level | `info` |

### Example Configuration

```bash
# ~/.bashrc or ~/.zshrc
export ROCFORGE_MODEL_PATH="/mnt/models"
export ROCFORGE_GPU_MEMORY=16
export ROCFORGE_LOG_LEVEL=debug
```

---

## Troubleshooting

### Common Issues

#### 1. "No HIP GPU found"

```bash
# Check GPU detection
rocm-smi

# Verify ROCm installation
rocminfo

# Check HIP SDK
ls -la /opt/rocm/bin/hipcc
```

#### 2. Model Loading Hangs at 180 Seconds

This is a known ROCm MES firmware bug. ROCmForge includes a workaround (memory pooling).

**Solution**: The issue is automatically mitigated. If it persists:

```bash
# Try disabling MES (temporary workaround)
sudo modprobe -r amdgpu
sudo modprobe amdgpu amdgpu.mes=0
```

#### 3. Out of Memory Errors

```bash
# Monitor GPU memory
watch -n 1 rocm-smi

# Try a smaller model
# - Use Q4_K quantization instead of Q8_0
# - Reduce context length
# - Close other GPU applications
```

#### 4. Slow Generation

- **First run is slow**: Model loading takes time
- **Subsequent runs**: Use in-memory caching
- **CPU bottleneck**: Check that GPU is actually being used (`rocm-smi`)

#### 5. CLI Crashes (Known Issue)

The CLI is experimental. For stable operation, use the HTTP server.

```bash
# Use HTTP server instead
rocmforge_cli serve --port 8080
```

### Debug Mode

```bash
# Enable verbose logging
RUST_LOG=debug rocmforge_cli generate --prompt "test"

# Run with backtrace
RUST_BACKTRACE=1 rocmforge_cli generate --prompt "test"
```

---

## Performance Tuning

### GPU Memory Optimization

| Setting | Impact | Recommendation |
|---------|--------|----------------|
| Quantization | 4x memory reduction | Use Q4_K for best quality/size |
| Context Length | Linear memory increase | Keep < 4096 for 7B models |
| Batch Size | Linear memory increase | Use HTTP server for batching |

### Sampling Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| Temperature | 0.0 - 2.0 | Higher = more random |
| Top-P | 0.0 - 1.0 | Nucleus sampling threshold |
| Top-K | 1 - 100 | Limit to top K tokens |

### Recommended Settings

**Creative Writing**:
```bash
--temperature 1.2 --top-p 0.9 --top-k 50
```

**Factual Responses**:
```bash
--temperature 0.1 --top-p 0.95 --top-k 40
```

**Balanced**:
```bash
--temperature 0.7 --top-p 0.9 --top-k 40
```

---

## Known Limitations

| Issue | Status | Workaround |
|-------|--------|------------|
| CLI crashes | ⚠️ Known | Use HTTP server |
| Phase 6 (GPU Sampler) | ⚠️ Pending | CPU fallback works |
| FP16 Compute | ⚠️ Planned | FP32 currently used |

---

## Getting Help

- **Documentation**: See `docs/` directory
- **API Reference**: `docs/API.md`
- **Changelog**: `docs/CHANGELOG.md`
- **Issues**: Report bugs on GitHub

---

## License

MIT License - See LICENSE file for details

---

**Next**: See [API.md](API.md) for complete API reference
