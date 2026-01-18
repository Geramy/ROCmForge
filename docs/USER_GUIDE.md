# ROCmForge User Guide

This guide provides comprehensive information for installing and using ROCmForge.

## Prerequisites

- **ROCm**: AMD GPU driver and toolkit (ROCm 5.0 or later recommended)
- **Rust**: Version 1.82 or later
- **GGUF Model**: A compatible LLaMA-based model in GGUF format

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/ROCmForge.git
cd ROCmForge

# Build the project
cargo build --release

# The binary will be at target/release/rocmforge_cli
```

## Quick Start

### Using the CLI

```bash
# Generate text using a local model
rocmforge-cli generate \
  --gguf /path/to/model.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 100

# Start the HTTP server
rocmforge-cli serve \
  --gguf /path/to/model.gguf \
  --addr 127.0.0.1:8080
```

### Environment Variables

ROCmForge can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCMFORGE_GGUF` | Path to the GGUF model file | (required) |
| `ROCMFORGE_TOKENIZER` | Path to tokenizer.json | Auto-detected |
| `ROCMFORGE_MODELS` | Directory containing GGUF models | `./models` |
| `ROCMFORGE_GPU_DEVICE` | GPU device number | 0 |
| `RUST_LOG` | Standard tracing filter | `info` |
| `ROCFORGE_LOG_LEVEL` | Simple log level | `info` |
| `ROCFORGE_LOG_FORMAT` | Output format (`human` or `json`) | `human` |
| `ROCFORGE_LOG_FILE` | Optional file path for log output | (none) |

## Logging Configuration

ROCmForge uses the `tracing` ecosystem for structured logging. Logging can be configured via environment variables or programmatically.

### Environment Variables

#### RUST_LOG (Standard Tracing Filter)

The `RUST_LOG` variable follows the standard tracing filter syntax:

```bash
# Set default level to info
export RUST_LOG=info

# Set module-specific levels
export RUST_LOG=rocmforge=debug,hyper=info

# Enable trace for everything
export RUST_LOG=trace

# Common pattern: debug for ROCmForge, info for dependencies
export RUST_LOG=rocmforge=debug,warn
```

#### ROCFORGE_LOG_LEVEL (Simple Level)

For simpler configuration, use `ROCFORGE_LOG_LEVEL`:

```bash
# Available levels: error, warn, info, debug, trace
export ROCFORGE_LOG_LEVEL=debug
```

#### ROCFORGE_LOG_FORMAT (Output Format)

Control how logs are displayed:

```bash
# Human-readable colored output (default)
export ROCFORGE_LOG_FORMAT=human

# JSON format for log aggregation
export ROCFORGE_LOG_FORMAT=json
```

#### ROCFORGE_LOG_FILE (File Output)

Write logs to a file (always in JSON format):

```bash
# Log to a file
export ROCFORGE_LOG_FILE=/var/log/rocmforge/app.log

# Combined with console output
export ROCFORGE_LOG_LEVEL=info
export ROCFORGE_LOG_FORMAT=human
export ROCFORGE_LOG_FILE=/tmp/rocmforge.log
```

### Log Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `error` | Only errors | Production (minimal output) |
| `warn` | Warnings and errors | Production (standard) |
| `info` | Normal operations | Development/testing |
| `debug` | Detailed diagnostics | Troubleshooting |
| `trace` | Everything including internal details | Deep debugging |

### Examples

#### Development Setup

```bash
# Enable debug output with colored console
export ROCFORGE_LOG_LEVEL=debug
export ROCFORGE_LOG_FORMAT=human
rocmforge-cli serve --gguf model.gguf
```

#### Production Setup

```bash
# Minimal output to console, detailed logs to file
export ROCFORGE_LOG_LEVEL=warn
export ROCFORGE_LOG_FILE=/var/log/rocmforge/production.log
rocmforge-cli serve --gguf model.gguf
```

#### Troubleshooting

```bash
# Maximum verbosity with module-specific filtering
export RUST_LOG=rocmforge=trace,hip=debug
rocmforge-cli generate --gguf model.gguf --prompt "test"
```

### Programmatic Configuration

You can also configure logging programmatically:

```rust
use rocmforge::logging::{LoggingConfig, LogLevel, LogFormat};

// Simple configuration
let config = LoggingConfig::new()
    .with_level(LogLevel::Debug)
    .with_format(LogFormat::Human);

rocmforge::init_with_config(&config);

// With file output
use std::path::PathBuf;

let config = LoggingConfig::new()
    .with_level(LogLevel::Info)
    .with_format(LogFormat::Json)
    .with_log_file(PathBuf::from("/var/log/rocmforge/app.log"))
    .with_file_info(true)
    .with_span_events(true);

rocmforge::init_with_config(&config);
```

## Common Use Cases

### Text Generation

```bash
rocmforge-cli generate \
  --gguf /models/llama-2-7b.gguf \
  --prompt "Explain quantum computing" \
  --max-tokens 500 \
  --temperature 0.8 \
  --top-k 50
```

### HTTP Server

```bash
# Start the server
rocmforge-cli serve --gguf model.gguf --addr 0.0.0.0:8080

# Generate text via HTTP
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

### Streaming Generation

```bash
rocmforge-cli generate \
  --gguf model.gguf \
  --prompt "Tell me a story" \
  --stream \
  --max-tokens 200
```

## Troubleshooting

### Common Issues

#### "GPU not found"

- Ensure ROCm is installed: `rocm-smi`
- Check GPU visibility: `ls /dev/kfd`
- Set the correct device: `export ROCFORGE_GPU_DEVICE=0`

#### "Failed to load model"

- Verify the GGUF file is valid: `rocmforge-cli models --dir /path/to/models`
- Check file permissions
- Ensure sufficient disk space

#### Out of Memory

- Reduce context length in the model configuration
- Use a quantized model (Q4_K, Q5_K, etc.)
- Close other GPU applications

## Getting Help

- Check logs: Set `ROCFORGE_LOG_LEVEL=debug` and re-run the command
- File issues: [GitHub Issues](https://github.com/your-org/ROCmForge/issues)
- Documentation: See `docs/` directory for more detailed guides
