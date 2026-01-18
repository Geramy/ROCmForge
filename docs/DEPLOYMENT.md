# ROCmForge Deployment Guide

> **Version:** 0.1.0
> **Last Updated:** 2026-01-19
> **Status:** Development / Testing

This guide provides comprehensive information for deploying ROCmForge in production-like environments.

**IMPORTANT:** ROCmForge is currently in development. This guide is for testing and development environments. Do not use for critical production systems without thorough testing.

---

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Binary Deployment](#binary-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Configuration Management](#configuration-management)
5. [Service Setup (systemd)](#service-setup-systemd)
6. [Reverse Proxy Configuration](#reverse-proxy-configuration)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Security Considerations](#security-considerations)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

---

## Deployment Options

ROCmForge supports the following deployment methods:

| Method | Complexity | Portability | Recommended For |
|--------|-----------|-------------|-----------------|
| Binary | Low | Low | Single-server deployments |
| Docker | Medium | High | Containerized environments |
| From Source | High | Low | Development and custom builds |

---

## Binary Deployment

### Building the Release Binary

```bash
# Clone the repository
git clone https://github.com/your-org/ROCmForge.git
cd ROCmForge

# Build optimized release binary
cargo build --release

# The binary will be at target/release/rocmforge_cli
```

### System Installation

```bash
# Install to system location
sudo install -m 755 target/release/rocmforge_cli /usr/local/bin/rocmforge-cli

# Verify installation
rocmforge-cli --help
```

### Directory Structure

```
/opt/rocmforge/
├── bin/
│   └── rocmforge-cli          # Main binary
├── models/
│   └── llama-2-7b.gguf        # Model files
├── tokenizers/
│   └── tokenizer.json         # Tokenizer files
└── logs/
    └── rocmforge.log          # Application logs
```

### Quick Start Command

```bash
# Set environment variables
export ROCMFORGE_GGUF=/opt/rocmforge/models/llama-2-7b.gguf
export ROCMFORGE_TOKENIZER=/opt/rocmforge/tokenizers/tokenizer.json
export ROCFORGE_LOG_LEVEL=info
export ROCFORGE_LOG_FORMAT=json

# Start the server
rocmforge-cli serve --addr 0.0.0.0:8080
```

---

## Docker Deployment

### Dockerfile

```dockerfile
# Build stage
FROM rust:1.82-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

# Build release binary
RUN cargo build --release

# Runtime stage
FROM ubuntu:22.04

# Install ROCm runtime (adjust for your GPU)
# See https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
RUN apt-get update && apt-get install -y \
    rocm-hip-sdk \
    rocm-dev \
    libnuma1 \
    && rm -rf /var/lib/apt/lists/*

ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/rocmforge_cli /app/rocmforge-cli

# Create directories
RUN mkdir -p /app/models /app/tokenizers /app/logs

# Set permissions
RUN chmod +x /app/rocmforge-cli

# Expose default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set default environment variables
ENV ROCFORGE_LOG_LEVEL=info
ENV ROCFORGE_LOG_FORMAT=json

# Run as non-root user (create first)
RUN useradd -m -u 1000 rocmforge && \
    chown -R rocmforge:rocmforge /app
USER rocmforge

CMD ["./rocmforge-cli", "serve", "--addr", "0.0.0.0:8080"]
```

### Building and Running

```bash
# Build the image
docker build -t rocmforge:0.1.0 .

# Run with model mounted
docker run -d \
    --name rocmforge \
    --gpus all \
    -p 8080:8080 \
    -v /path/to/models:/app/models:ro \
    -v /path/to/tokenizers:/app/tokenizers:ro \
    -e ROCMFORGE_GGUF=/app/models/llama-2-7b.gguf \
    -e ROCMFORGE_TOKENIZER=/app/tokenizers/tokenizer.json \
    -e ROCFORGE_LOG_LEVEL=info \
    --restart unless-stopped \
    rocmforge:0.1.0

# View logs
docker logs -f rocmforge

# Check health
docker exec rocmforge curl http://localhost:8080/health
```

### Docker Compose

```yaml
version: '3.8'

services:
  rocmforge:
    image: rocmforge:0.1.0
    container_name: rocmforge
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia  # or 'amd' for AMD GPU support
              count: all
              capabilities: [gpu]
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models:ro
      - ./tokenizers:/app/tokenizers:ro
      - ./logs:/app/logs:rw
    environment:
      - ROCMFORGE_GGUF=/app/models/llama-2-7b.gguf
      - ROCMFORGE_TOKENIZER=/app/tokenizers/tokenizer.json
      - ROCFORGE_LOG_LEVEL=info
      - ROCFORGE_LOG_FORMAT=json
      - ROCFORGE_LOG_FILE=/app/logs/rocmforge.log
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - rocmforge-network

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - rocmforge-network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped
    networks:
      - rocmforge-network

networks:
  rocmforge-network:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

---

## Configuration Management

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ROCMFORGE_GGUF` | Path to GGUF model file | - | Yes |
| `ROCMFORGE_TOKENIZER` | Path to tokenizer.json | Auto-detected | No |
| `ROCMFORGE_MODELS` | Directory containing GGUF models | `./models` | No |
| `ROCMFORGE_GPU_DEVICE` | GPU device number | `0` | No |
| `ROCFORGE_LOG_LEVEL` | Log level (error/warn/info/debug/trace) | `info` | No |
| `ROCFORGE_LOG_FORMAT` | Log format (human/json) | `human` | No |
| `ROCFORGE_LOG_FILE` | Path to log file | - | No |
| `RUST_LOG` | Standard tracing filter | `info` | No |

### Configuration File

Create `/etc/rocmforge/config.env`:

```bash
# ROCmForge Configuration

# Model paths
export ROCMFORGE_GGUF=/opt/rocmforge/models/llama-2-7b.gguf
export ROCMFORGE_TOKENIZER=/opt/rocmforge/tokenizers/tokenizer.json
export ROCMFORGE_MODELS=/opt/rocmforge/models

# GPU configuration
export ROCMFORGE_GPU_DEVICE=0

# Logging
export ROCFORGE_LOG_LEVEL=info
export ROCFORGE_LOG_FORMAT=json
export ROCFORGE_LOG_FILE=/var/log/rocmforge/app.log

# Server configuration
export ROCFORGE_BIND_ADDRESS=0.0.0.0:8080
```

Load configuration in systemd service file:
```ini
EnvironmentFile=/etc/rocmforge/config.env
```

---

## Service Setup (systemd)

### Create Systemd Service

Create `/etc/systemd/system/rocmforge.service`:

```ini
[Unit]
Description=ROCmForge Inference Server
Documentation=https://github.com/your-org/ROCmForge
After=network.target multi-user.target
Wants=network-online.target

[Service]
Type=simple
User=rocmforge
Group=rocmforge
WorkingDirectory=/opt/rocmforge

# Load environment variables
EnvironmentFile=/etc/rocmforge/config.env

# ExecStart
ExecStart=/usr/local/bin/rocmforge-cli serve \
    --gguf ${ROCMFORGE_GGUF} \
    --tokenizer ${ROCMFORGE_TOKENIZER} \
    --addr ${ROCMFORGE_BIND_ADDRESS}

# Restart configuration
Restart=always
RestartSec=10
TimeoutStartSec=120
TimeoutStopSec=30

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=32G

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/rocmforge /var/log/rocmforge
CapabilityBoundingSet=CAP_SYS_ADMIN

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=rocmforge

[Install]
WantedBy=multi-user.target
```

### Create User and Directories

```bash
# Create dedicated user
sudo useradd -r -s /bin/false -d /opt/rocmforge rocmforge

# Create directories
sudo mkdir -p /opt/rocmforge/{models,tokenizers,bin,logs}
sudo mkdir -p /etc/rocmforge
sudo mkdir -p /var/log/rocmforge

# Set permissions
sudo chown -R rocmforge:rocmforge /opt/rocmforge
sudo chown -R rocmforge:rocmforge /var/log/rocmforge
sudo chmod 755 /opt/rocmforge
sudo chmod 750 /var/log/rocmforge
```

### Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable rocmforge

# Start service
sudo systemctl start rocmforge

# Check status
sudo systemctl status rocmforge

# View logs
sudo journalctl -u rocmforge -f

# View last 100 lines
sudo journalctl -u rocmforge -n 100
```

### Service Management Commands

```bash
# Start service
sudo systemctl start rocmforge

# Stop service
sudo systemctl stop rocmforge

# Restart service
sudo systemctl restart rocmforge

# Reload configuration (if supported)
sudo systemctl reload rocmforge

# Check if service is active
sudo systemctl is-active rocmforge

# Check if service is enabled
sudo systemctl is-enabled rocmforge

# View service status
sudo systemctl status rocmforge --no-pager

# View logs since boot
sudo journalctl -u rocmforge -b

# Follow logs in real-time
sudo journalctl -u rocmforge -f
```

---

## Reverse Proxy Configuration

### Nginx Configuration

Create `/etc/nginx/sites-available/rocmforge`:

```nginx
upstream rocmforge_backend {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    server_name rocmforge.example.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name rocmforge.example.com;

    # SSL configuration (use Let's Encrypt or your certificates)
    ssl_certificate /etc/ssl/certs/rocmforge.crt;
    ssl_certificate_key /etc/ssl/private/rocmforge.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Logging
    access_log /var/log/nginx/rocmforge_access.log;
    error_log /var/log/nginx/rocmforge_error.log;

    # Client upload size (for large prompts)
    client_max_body_size 10M;

    # Timeouts
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;

    # Generate endpoint
    location /v1/generate {
        proxy_pass http://rocmforge_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
    }

    # Streaming endpoint
    location /v1/generate/stream {
        proxy_pass http://rocmforge_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
    }

    # Status endpoint
    location /v1/status {
        proxy_pass http://rocmforge_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Health check (no auth required)
    location /health {
        proxy_pass http://rocmforge_backend;
        access_log off;
    }

    # Metrics (restrict access)
    location /metrics {
        proxy_pass http://rocmforge_backend;
        auth_basic "Prometheus Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd;
        allow 127.0.0.1;
        allow 10.0.0.0/8;  # Adjust for your monitoring network
        deny all;
    }

    # Ready endpoint (for load balancers)
    location /ready {
        proxy_pass http://rocmforge_backend;
        access_log off;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/rocmforge /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Apache Configuration

```apache
<VirtualHost *:80>
    ServerName rocmforge.example.com
    Redirect permanent / https://rocmforge.example.com/
</VirtualHost>

<VirtualHost *:443>
    ServerName rocmforge.example.com

    SSLEngine on
    SSLCertificateFile /etc/ssl/certs/rocmforge.crt
    SSLCertificateKeyFile /etc/ssl/private/rocmforge.key
    SSLProtocol all -SSLv3 -TLSv1 -TLSv1.1
    SSLCipherSuite HIGH:!aNULL:!MD5

    ProxyPreserveHost On
    ProxyPass /health http://127.0.0.1:8080/health retry=0
    ProxyPassReverse /health http://127.0.0.1:8080/health

    ProxyPass /ready http://127.0.0.1:8080/ready retry=0
    ProxyPassReverse /ready http://127.0.0.1:8080/ready

    ProxyPass /metrics http://127.0.0.1:8080/metrics retry=0
    ProxyPassReverse /metrics http://127.0.0.1:8080/metrics

    ProxyPass /v1/ http://127.0.0.1:8080/ retry=0 timeout=300
    ProxyPassReverse /v1/ http://127.0.0.1:8080/

    # For SSE streaming
    <Location /v1/generate/stream>
        ProxyPass http://127.0.0.1:8080/generate/stream retry=0 timeout=300
        ProxyPassReverse http://127.0.0.1:8080/generate/stream
        RequestHeader set Connection ""
    </Location>

    ErrorLog ${APACHE_LOG_DIR}/rocmforge_error.log
    CustomLog ${APACHE_LOG_DIR}/rocmforge_access.log combined
</VirtualHost>
```

### HAProxy Configuration

```
backend rocmforge
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    server rocmforge1 127.0.0.1:8080 check inter 5s rise 2 fall 3

frontend rocmforge_http
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/rocmforge.pem
    http-request set-header X-Forwarded-Proto https if { ssl_fc }
    default_backend rocmforge

    # Health check endpoint
    acl is_health_check path /health
    use_backend rocmforge if is_health_check

    # Metrics endpoint (restrict access)
    acl is_metrics path /metrics
    acl allowed_metrics src 10.0.0.0/8
    http-request deny unless is_metrics allowed_metrics
    use_backend rocmforge if is_metrics
```

---

## Monitoring and Observability

### Monitoring Endpoints

ROCmForge provides several endpoints for monitoring:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check status |
| `/ready` | GET | Readiness probe (returns 503 if not ready) |
| `/metrics` | GET | Prometheus metrics |
| `/traces` | GET | OpenTelemetry traces (OTLP JSON format) |

### Health Endpoint Response

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
        "utilization_percent": 45
      }
    },
    "requests": {
      "active": 2,
      "queued": 1
    },
    "cache": {
      "pages_used": 450,
      "pages_total": 1000,
      "pages_free": 550,
      "active_sequences": 2
    }
  }
}
```

### Prometheus Metrics

```
# Request counters
rocmforge_requests_started_total{status="started"} 1245
rocmforge_requests_completed_total{status="completed"} 1198
rocmforge_requests_failed_total{status="failed"} 23
rocmforge_requests_cancelled_total{status="cancelled"} 24

# Token generation
rocmforge_tokens_generated_total 45234

# Duration histograms
rocmforge_prefill_duration_seconds_bucket{le="0.1"} 850
rocmforge_prefill_duration_seconds_bucket{le="0.5"} 1200
rocmforge_decode_duration_seconds_bucket{le="0.01"} 500
rocmforge_ttft_seconds_bucket{le="0.2"} 780

# Queue metrics
rocmforge_queue_length 3
rocmforge_active_requests 2

# Performance
rocmforge_tokens_per_second 45.2
```

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rocmforge'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

### Grafana Dashboard

Key metrics to visualize:

1. **Request Rate**: `rate(rocmforge_requests_started_total[5m])`
2. **Request Duration**: `histogram_quantile(0.95, rate(rocmforge_decode_duration_seconds_bucket[5m]))`
3. **Tokens/sec**: `rocmforge_tokens_per_second`
4. **GPU Memory**: `rocmforge_gpu_memory_used_bytes`
5. **Cache Utilization**: `rocmforge_cache_pages_used / rocmforge_cache_pages_total`
6. **Queue Depth**: `rocmforge_queue_length`

### Log Aggregation

#### Loki + Promtail

Configure Promtail to scrape journal logs:

```yaml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://localhost:3100/loki/api/v1/push

scrape_configs:
  - job_name: systemd-journal
    journal:
      max_age: 12h
      labels:
        job: systemd-journal
        unit: rocmforge
    relabel_configs:
      - source_labels: ['__journal__systemd_unit']
        target_label: 'unit'
      - source_labels: ['__journal_priority']
        target_label: 'level'
```

#### File-based Logging

For direct file scraping:

```yaml
scrape_configs:
  - job_name: rocmforge-file
    static_configs:
      - targets:
          - localhost
        labels:
          job: rocmforge
          __path__: /var/log/rocmforge/*.log
```

---

## Security Considerations

### Network Security

1. **Bind Address**: Use `127.0.0.1` for local-only access, `0.0.0.0` only with firewall
2. **Reverse Proxy**: Always use Nginx/Apache for public-facing deployments
3. **TLS/SSL**: Enable HTTPS for all external access
4. **Firewall Rules**: Restrict access to monitoring endpoints

```bash
# Example UFW rules
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8080/tcp  # Block direct access to ROCmForge
```

### Authentication

#### Basic Auth (Nginx)

```bash
# Create password file
sudo htpasswd -c /etc/nginx/.htpasswd admin
```

#### API Keys (Future)

API key authentication is planned for future releases. Currently, implement at the reverse proxy level.

### Resource Limits

```ini
# In systemd service
MemoryMax=32G
CPUQuota=400%
TasksMax=1000
```

### File Permissions

```bash
# Secure model files
chmod 640 /opt/rocmforge/models/*.gguf
chown root:rocmforge /opt/rocmforge/models/*.gguf

# Secure logs
chmod 640 /var/log/rocmforge/*
chown rocmforge:adm /var/log/rocmforge/*
```

### Running as Non-Root

Always run ROCmForge as a non-privileged user:

```ini
# In systemd
User=rocmforge
Group=rocmforge
NoNewPrivileges=true
```

---

## Performance Tuning

### GPU Memory Optimization

1. **Model Quantization**: Use Q4_K or Q5_K quantized models
2. **Context Length**: Reduce if memory constrained
3. **Batch Size**: Adjust based on GPU memory

```
# Estimated memory requirements (7B model)
Q4_K:  ~4-5 GB
Q5_K:  ~5-6 GB
Q6_K:  ~6-7 GB
Q8_0:  ~8-9 GB
F16:   ~14 GB
```

### CPU Resource Allocation

```ini
# In systemd, pin to specific CPUs
AllowedCPUs=4-15

# Set CPU affinity (alternative)
TaskCPUAffinity=4-15
```

### I/O Performance

- Use SSD/NVMe for model storage
- Pre-load models at startup
- Use memory-mapped file loading (default)

### Network Optimization

```
# sysctl settings
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
```

### ROCm Configuration

```bash
# Set GPU performance mode
sudo rocm-smi --setsperf 170

# Disable GPU power capping
sudo rocm-smi --setpoweroverdrive 0

# Check GPU status
rocm-smi
```

### Kernel Parameters

```
# /etc/sysctl.d/99-rocmforge.conf
vm.swappiness=10
vm.dirty_ratio=15
vm.dirty_background_ratio=5
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check service status
sudo systemctl status rocmforge

# Check logs
sudo journalctl -u rocmforge -n 100 --no-pager

# Check GPU availability
rocm-smi
ls /dev/kfd
```

### GPU Not Detected

```bash
# Verify ROCm installation
hipconfig --version

# Check GPU devices
rocm-smi --showallinfo

# Verify permissions
ls -la /dev/kfd /dev/dri/*
```

### Out of Memory Errors

```bash
# Check GPU memory usage
rocm-smi --showmemuse

# Reduce context length
export ROCMFORGE_MAX_CONTEXT=2048

# Use more quantized model
```

### High Latency

```bash
# Check queue depth
curl http://localhost:8080/metrics | grep queue_length

# Check GPU utilization
rocm-smi --showuse

# Verify batch size
curl http://localhost:8080/metrics | grep active_requests
```

### Monitoring Health

```bash
# Quick health check
watch -n 5 'curl -s http://localhost:8080/health | jq .'

# Continuous metrics
curl -s http://localhost:8080/metrics | grep rocmforge
```

---

## Known Limitations

1. **Single Model**: Currently loads one model per instance
2. **No Model Hot-Swap**: Requires restart to change models
3. **CPU Fallback**: Falls back to CPU if GPU unavailable (slow)
4. **Experimental**: Not yet production-ready

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-org/ROCmForge/issues)
- **Documentation**: See `docs/` directory
- **Logs**: Check with `journalctl -u rocmforge`

---

**Document Version:** 1.0
**Last Updated:** 2026-01-19
**ROCmForge Version:** 0.1.0
