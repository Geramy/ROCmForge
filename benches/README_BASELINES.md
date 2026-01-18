# Performance Baseline Management

This document describes how to work with performance baselines in ROCmForge benchmarks.

## Overview

Performance baselines are used to:
- Track performance over time
- Detect regressions in critical metrics
- Compare performance across hardware configurations
- Validate optimization impact

## Baseline File Structure

Baselines are stored as JSON files in `benchmarks/baselines/`:

```
benchmarks/baselines/
├── rdna3-baseline.json       # CPU baselines (current)
├── rdna3-gpu-baseline.json   # GPU baselines (to be collected on RDNA3)
└── baseline-template.json     # Template for new baselines
```

### Baseline JSON Schema

Each baseline file contains:

```json
{
  "baselines": {
    "benchmark_name": {
      "name": "benchmark_name",
      "timestamp": 1737220800,
      "hardware": {
        "gpu_name": "Optional GPU name",
        "gpu_architecture": "RDNA3",
        "rocm_version": "6.0.0",
        "cpu_arch": "x86_64",
        "os": "linux",
        "rustc_version": "1.82.0"
      },
      "metrics": {
        "avg_ms": 10.5,
        "min_ms": 9.8,
        "max_ms": 11.2,
        "p50_ms": 10.3,
        "p95_ms": 10.9,
        "p99_ms": 11.1,
        "iterations": 100
      },
      "metadata": {
        "backend": "cpu",
        "format": "Q4_0"
      }
    }
  },
  "metadata": {
    "description": "Baseline description",
    "hardware_type": "cpu",
    "baseline_date": "2025-01-18"
  },
  "hardware": { ... },
  "timestamp": 1737220800
}
```

## Using Baselines in Benchmarks

### Creating a Baseline Helper

```rust
use rocmforge::profiling::BenchmarkBaseline;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a baseline helper
    let mut baseline_helper = BenchmarkBaseline::new();

    // Add benchmark results
    let durations = vec![10.0, 11.0, 9.0, 10.5, 9.5];
    baseline_helper.add_benchmark("my_benchmark", &durations);

    // Save to file
    baseline_helper.save("benchmarks/baselines/my-baseline.json")?;

    Ok(())
}
```

### Comparing Against Baselines

```rust
use rocmforge::profiling::{BaselineCollection, BaselineMetrics};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load baseline collection
    let collection = BaselineCollection::load("benchmarks/baselines/rdna3-baseline.json")?;

    // Create current metrics
    let current_metrics = BaselineMetrics {
        avg_ms: 10.8,
        min_ms: 10.0,
        max_ms: 11.5,
        p50_ms: 10.7,
        p95_ms: 11.3,
        p99_ms: 11.4,
        iterations: 100,
    };

    // Compare against baseline
    let baseline = collection.get("my_benchmark").unwrap();
    let result = baseline.compare_metrics("my_benchmark", &current_metrics, 0.10);

    if result.is_regression() {
        eprintln!("Regression detected: {}", result.description());
    }

    Ok(())
}
```

### Full Comparison Report

```rust
use rocmforge::profiling::BaselineCollection;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let collection = BaselineCollection::load("benchmarks/baselines/rdna3-baseline.json")?;

    // Current run metrics
    let mut current = HashMap::new();
    current.insert("bench1".to_string(), /* metrics */);
    current.insert("bench2".to_string(), /* metrics */);

    // Generate report
    let report = collection.compare_and_report(&current, 0.10);
    report.print();

    // Check for failures
    if report.has_failures() {
        eprintln!("Failed benchmarks: {:?}", report.failed_benchmarks());
        std::process::exit(1);
    }

    Ok(())
}
```

## Command-Line Usage

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench --bench matmul_bench
cargo bench --bench attention_bench
cargo bench --bench dequant_bench
cargo bench --bench inference_bench
cargo bench --bench memory_bench

# Run with ROCm feature for GPU benchmarks
cargo bench --bench matmul_bench --features rocm
```

### Saving Baselines

```bash
# Save baselines (in benchmark code)
cargo bench --bench matmul_bench -- --save-baseline baselines/my-baseline.json

# Compare against existing baseline
cargo bench --bench matmul_bench -- --compare-baseline baselines/rdna3-baseline.json
```

## Regression Detection

### Threshold Configuration

The default regression threshold is 10% for average execution time:

```rust
use rocmforge::profiling::RegressionThreshold;

let threshold = RegressionThreshold {
    avg_threshold_pct: 0.10,  // 10%
    p95_threshold_pct: 0.15,  // 15%
    p99_threshold_pct: 0.20,  // 20%
};
```

### Comparison Result Types

- **Ok**: Performance within acceptable range
- **Improved**: Performance is faster than baseline
- **Regression**: Performance is slower than baseline (exceeds threshold)
- **HardwareMismatch**: Hardware architectures differ

## GPU-Specific Baselines

For GPU-specific baselines (RDNA3):

```rust
use rocmforge::profiling::{HardwareInfo, BenchmarkBaseline};

let gpu_hardware = HardwareInfo::with_gpu("RX 7900 XTX", "RDNA3", "6.0.0");
let mut baseline_helper = BenchmarkBaseline::with_hardware(gpu_hardware);

// Add GPU benchmarks
baseline_helper.add_benchmark("gpu_matmul_4096", &durations);

// Save with hardware info
baseline_helper.save("benchmarks/baselines/rdna3-gpu-baseline.json")?;
```

## Updating Baselines

When to update baselines:

1. **After optimizations**: When performance has legitimately improved
2. **Hardware changes**: When running on different hardware
3. **ROCm version changes**: When upgrading ROCm
4. **Model changes**: When benchmark inputs change significantly

To update a baseline:

1. Run benchmarks to verify new performance
2. Document the reason for the update in commit message
3. Update the baseline JSON file
4. Commit the changes with rationale

## Best Practices

1. **Always run multiple iterations**: Use at least 10 iterations for stable baselines
2. **Use consistent hardware**: Don't compare CPU baselines with GPU baselines
3. **Document metadata**: Include format, size, and backend in metadata
4. **Version control baselines**: Commit baseline files to track changes
5. **Review regressions**: Don't blindly update baselines; investigate regressions first

## Example Workflow

```bash
# 1. Run benchmarks with baseline comparison
cargo bench --bench matmul_bench

# 2. If regression detected, investigate
# Check recent code changes, profile with rocprof

# 3. If legitimate improvement, update baseline
# Edit benchmarks/baselines/rdna3-baseline.json with new values

# 4. Verify new baseline passes
cargo bench --bench matmul_bench

# 5. Commit with rationale
git commit -m "perf(matmul): update baseline after optimization

Optimized matmul kernel for better cache locality.
Improvement: 15% faster on RDNA3 (450ms -> 380ms for 4096x4096)"
```

## Troubleshooting

### Hardware Mismatch Errors

If you see "Hardware mismatch" errors:

```rust
// Check hardware compatibility
let collection = BaselineCollection::load("path/to/baseline.json")?;
match collection.check_hardware_compatibility() {
    Ok(_) => println!("Hardware compatible"),
    Err(e) => eprintln!("Hardware mismatch: {:?}", e),
}
```

### Missing Baselines

If a benchmark is missing from the baseline:

```rust
// Check which baselines exist
let collection = BaselineCollection::load("path/to/baseline.json")?;
println!("Available baselines: {:?}", collection.names());
```

### Flaky Measurements

If measurements are inconsistent:

1. Increase iteration count
2. Check for background processes
3. Verify CPU frequency scaling is disabled
4. Use warmed-up runs (discard first few iterations)

## See Also

- [Profiling Guide](../../docs/PROFILING_GUIDE.md) - ROCm profiling tools
- [Performance Notes](../../docs/PERFORMANCE.md) - Performance tuning guidelines
- [Baseline API](../src/profiling/baseline.rs) - Rust API documentation
