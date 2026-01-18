# GPU Kernel Tuning Guide

**Last Updated:** 2026-01-18
**Status:** Ready for GPU validation

---

## Overview

ROCmForge now supports configurable kernel tuning parameters optimized for different AMD GPU architectures. This guide explains how to tune kernels for optimal performance.

---

## Architecture Detection

### Supported Architectures

| Architecture | GFX IP | Wave Size | LDS per CU | Typical Products |
|--------------|--------|-----------|------------|------------------|
| RDNA3 | gfx1100-1102 | 32 | 256 KB | RX 7900 series |
| RDNA2 | gfx1030-1035 | 32 | 128 KB | RX 6000 series |
| CDNA3 | gfx940-942 | 64 | Large | MI300 series |
| CDNA2 | gfx90a | 64 | 128 KB | MI200 series |

### Detecting Your Architecture

```bash
# Use rocminfo to find your GFX IP
rocminfo | grep "Name:"

# Or use clinfo
clinfo | grep " gfx"
```

---

## Tuning Parameters

### 1. Block Size (BLOCK_SIZE)

Number of threads per kernel block. Affects occupancy and resource usage.

| Value | Use Case | Pros | Cons |
|-------|----------|------|------|
| 128 | Low register pressure kernels | High occupancy, fewer resources | May not fully utilize device |
| 256 | Default for RDNA | Balanced | - |
| 384 | Medium register pressure | Good balance | May reduce occupancy |
| 512 | CDNA/default for high LDS | Max utilization for datacenter | Lower occupancy on RDNA |

**Guidelines:**
- RDNA: 256 is usually optimal
- CDNA: 512 is usually optimal
- Reduce if seeing "too many resources" compiler errors

### 2. Wave Size (WARP_SIZE)

Number of threads per wavefront. This is architecture-dependent.

| Architecture | Wave Size |
|--------------|-----------|
| RDNA2, RDNA3 | 32 |
| CDNA2, CDNA3 | 64 |

**Important:** Using the wrong wave size will cause incorrect results!

### 3. Tile Sizes (TILE_SIZE_K, TILE_SIZE_N)

Dimensions of matrix tiles loaded into shared memory.

| Value | Use Case |
|-------|----------|
| 16 | Small matrices, limited LDS |
| 32 | Default (balanced) |
| 64 | Large matrices, abundant LDS |
| 128 | Very large matrices (datacenter) |

**Guidelines:**
- Start with 32x32
- Increase if LDS is underutilized
- Decrease if seeing occupancy issues

### 4. LDS Usage (USE_LDS)

Enable/disable Local Data Share optimization.

| Setting | Use Case |
|---------|----------|
| 0 (disabled) | Small matrices, very limited LDS |
| 1 (enabled) | Default for most cases |

**When to disable LDS:**
- Matrix dimensions < 128
- Very high concurrent block count
- Seeing LDS spill warnings

---

## Compile-Time Tuning

### Method 1: Preprocessor Defines

```bash
# Set defines at compile time
hipcc \
    -DBLOCK_SIZE=256 \
    -DWARP_SIZE=32 \
    -DTILE_SIZE_K=32 \
    -DTILE_SIZE_N=32 \
    -DUSE_LDS=1 \
    -c q4_0_matmul.hip
```

### Method 2: Environment Variables

The Rust code can read tuning from environment variables:

```bash
# Set before running the application
export ROCFORGE_BLOCK_SIZE=256
export ROCFORGE_WARP_SIZE=32
export ROCFORGE_USE_LDS=1
export ROCFORGE_TILE_K=32
export ROCFORGE_TILE_N=64

# Run your application
cargo run --release --features rocm
```

---

## Architecture-Specific Tuning

### RDNA3 Tuning

**Target:** Radeon RX 7900 series (gfx1100)

```bash
# Recommended settings
BLOCK_SIZE=256
WARP_SIZE=32
TILE_SIZE_K=32
TILE_SIZE_N=32
USE_LDS=1
```

**Characteristics:**
- Wave32 execution
- 256 KB LDS per CU
- High clock speeds
- Good for: consumer workloads, gaming, inference

### RDNA2 Tuning

**Target:** Radeon RX 6000 series (gfx1030)

```bash
# Recommended settings
BLOCK_SIZE=256
WARP_SIZE=32
TILE_SIZE_K=32
TILE_SIZE_N=32
USE_LDS=1
LDS_SIZE_PER_BLOCK=16384  # Conservative due to 128 KB LDS
```

**Characteristics:**
- Wave32 execution
- 128 KB LDS per CU (less than RDNA3)
- Good for: similar workloads to RDNA3 but with less LDS

### CDNA2 Tuning

**Target:** Instinct MI200 series (gfx90a)

```bash
# Recommended settings
BLOCK_SIZE=512
WARP_SIZE=64
TILE_SIZE_K=64
TILE_SIZE_N=64
USE_LDS=1
```

**Characteristics:**
- Wave64 execution
- 128 KB LDS per CU
- Lower clock speeds, more cores
- Good for: datacenter, training, large batch inference

### CDNA3 Tuning

**Target:** Instinct MI300 series (gfx940+)

```bash
# Recommended settings
BLOCK_SIZE=512
WARP_SIZE=64
TILE_SIZE_K=64
TILE_SIZE_N=64
USE_LDS=1
```

**Characteristics:**
- Wave64 execution
- Large LDS (size varies by SKU)
- APU + GPU architecture
- Good for: AI workloads, large models

---

## Profiling and Validation

### Using rocprof

```bash
# Basic kernel profiling
rocprof --hip-trace -o output ./your_app

# Memory bandwidth analysis
rocprof -m mem_bw ./your_app

# Occupancy analysis
rocprof -m occupancy ./your_app
```

### Key Metrics to Watch

1. **Memory Bandwidth Utilization**
   - Target: >80% of peak bandwidth
   - If lower: increase tile sizes, enable LDS

2. **Occupancy**
   - Target: >50% occupancy
   - If lower: reduce block size, reduce LDS usage

3. **LDS Bank Conflicts**
   - Check for high LDS stall cycles
   - Fix: adjust tile sizes or access patterns

4. **Wave Utilization**
   - Target: close to 100% (all lanes active)
   - If low: restructure loops for better divergence

---

## Troubleshooting

### Compilation Errors

**Error: "Too many resources requested"**
- Solution: Reduce BLOCK_SIZE
- Solution: Reduce TILE_SIZE_K/TILE_SIZE_N
- Solution: Set USE_LDS=0

### Runtime Errors

**Error: "Out of memory"**
- Solution: Reduce LDS_SIZE_PER_BLOCK
- Solution: Reduce BLOCK_SIZE

**Error: Incorrect results**
- Solution: Verify WARP_SIZE matches your architecture
- Solution: Check for race conditions in LDS code

### Performance Issues

**Problem: Slower than expected**
- Verify LDS is enabled (USE_LDS=1)
- Check block size matches architecture
- Profile with rocprof to find bottleneck

**Problem: No improvement from tuning**
- Some workloads are memory-bound (kernel won't help)
- Check if you're hitting peak memory bandwidth
- Try different matrix sizes (some sizes tune better)

---

## Example Configurations

### Small Batch Inference (1-4 tokens)

```bash
# For small workloads, prioritize latency
BLOCK_SIZE=128
WARP_SIZE=32
TILE_SIZE_K=16
TILE_SIZE_N=16
USE_LDS=0  # Skip LDS for small matrices
```

### Large Batch Inference (32+ tokens)

```bash
# For large workloads, prioritize throughput
BLOCK_SIZE=256
WARP_SIZE=32
TILE_SIZE_K=32
TILE_SIZE_N=32
USE_LDS=1  # Enable LDS
```

### Training Workloads

```bash
# For training, maximize throughput
BLOCK_SIZE=512
WARP_SIZE=64
TILE_SIZE_K=64
TILE_SIZE_N=64
USE_LDS=1
```

---

## API Reference

### KernelTuning Struct

```rust
pub struct KernelTuning {
    pub block_size: u32,
    pub warp_size: u32,
    pub use_lds: bool,
    pub lds_size_per_block: u32,
    pub tile_size_k: u32,
    pub tile_size_n: u32,
    pub accumulators_per_thread: u32,
}
```

### GpuArchitecture Enum

```rust
pub enum GpuArchitecture {
    Rdna3,    // gfx1100+
    Rdna2,    // gfx1030+
    Cdna2,    // gfx90a
    Cdna3,    // gfx940+
    Unknown,
}
```

### Key Methods

```rust
// Auto-detect tuning from GFX IP
let tuning = KernelTuning::for_architecture("gfx1100");

// Read from environment variables
let tuning = KernelTuning::from_env();

// Customize manually
let tuning = KernelTuning::default()
    .with_override(|c| c.block_size = 384);

// Validate configuration
tuning.validate()?;

// Get kernel defines
let defines = tuning.kernel_defines();
```

---

*Guide last updated: 2026-01-18*
*For issues or questions, please file a GitHub issue.*
