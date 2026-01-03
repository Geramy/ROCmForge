# ROCm/HIP Developer Setup Guide

> Complete guide for setting up ROCm development environment for ROCmForge
>
> **Your system:** ROCm 7.1 detected - this guide applies. Key differences from 6.0 noted throughout.

---

## Table of Contents

1. [ROCm Installation](#1-rocm-installation)
2. [HIP Development Tools](#2-hip-development-tools)
3. [Compiling HIP Kernels](#3-compiling-hip-kernels)
4. [Rust + HIP Integration](#4-rust--hip-integration)
5. [Common Commands](#5-common-commands)
6. [Troubleshooting](#6-troubleshooting)
7. [Code Examples](#7-code-examples)

---

## Quick Check: What's Installed?

```bash
# Check ROCm version
cat /opt/rocm/.info/version
# Or
rocminfo | head -20

# Check hipcc version
hipcc --version

# List all ROCm components
dpkg -l | grep rocm
```

**Your system (ROCm 7.1):**
- Newer LLVM/Clang backend
- Improved CDNA3 (gfx942) support
- Better FP16 support
- Updated hipBLAS/rocBLAS

---

## 1. ROCm Installation

### Supported GPUs

ROCm supports AMD GPUs from the following series:
- **CDNA2**: MI200 series (gfx90a)
- **CDNA3**: MI300 series (gfx942)
- **RDNA3**: RX 7000 series (gfx1100+)
- Older: Vega (gfx900), Radeon VII (gfx906)

Check your GPU:
```bash
lspci | grep -i vga
```

### Ubuntu 22.04/24.04 Installation

#### Step 1: Add AMD Repository

```bash
# Download amdgpu-install script
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb

# Install the package
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
```

#### Step 2: Install ROCm

```bash
# Update package list
sudo apt update

# Install ROCm (core packages)
sudo amdgpu-install --usecase=rocm --no-dkms

# This installs:
# - rocm-core
# - hip-runtime
# - rocblas
# - hipcub
# - amdhip64 (runtime library)
```

#### Step 3: Set Environment Variables

Add to `~/.bashrc`:
```bash
# ROCm environment
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# For development
export ROCM_PATH=/opt/rocm
export HIPCC=/opt/rocm/bin/hipcc
```

Then reload:
```bash
source ~/.bashrc
```

#### Step 4: Verify Installation

```bash
# Check ROCm version
rocminfo

# Check hipcc
hipcc --version

# Test GPU access
/opt/rocm/bin/rocminfo | grep "Device*"
```

### Windows Installation

Windows support is newer and more limited:

1. Download from [ROCm Installation for Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.1.2/)
2. Install HIP SDK for Windows
3. Visual Studio 2019/2022 required

---

## 2. HIP Development Tools

### Installed Components

| Component | Path | Purpose |
|-----------|------|---------|
| hipcc | `/opt/rocm/bin/hipcc` | HIP compiler |
| amdhip64.so | `/opt/rocm/lib/libamdhip64.so` | HIP runtime |
| rocblas | `/opt/rocm/lib/librocblas.so` | BLAS library |
| hipify | `/opt/rocm/bin/hipify-clang` | CUDA → HIP converter |

### Verification

```bash
# Check all components
ls -la /opt/rocm/bin/ | grep hip
ls -la /opt/rocm/lib/ | grep hip
ls -la /opt/rocm/lib/ | grep rocblas
```

---

## 3. Compiling HIP Kernels

### Basic Compilation

```bash
# Compile HIP file to executable
hipcc my_kernel.hip -o my_program

# Compile with optimizations
hipcc -O3 my_kernel.hip -o my_program

# Compile for specific architecture
hipcc --offload-arch=gfx942 my_kernel.hip -o my_program
```

### Architecture Targets

| Architecture | GPUs | Target Flag | Wave Size | Notes |
|-------------|------|-------------|-----------|-------|
| **CDNA2** | | | **64** | Datacenter |
| gfx906 | Vega 20 | `--offload-arch=gfx906` | 64 | Stable |
| gfx908 | MI100 | `--offload-arch=gfx908` | 64 | Stable |
| gfx90a | MI200 (Aldebaran) | `--offload-arch=gfx90a` | 64 | Improved perf |
| | | | | |
| **CDNA3** | | | **64** | Newer datacenter |
| gfx940 | MI200 variant | `--offload-arch=gfx940` | 64 | New in 7.x |
| **gfx942** | **MI300** | `--offload-arch=gfx942` | 64 | Better tuning in 7.1 |
| gfx950 | MI300X | `--offload-arch=gfx950` | 64 | New in ROCm 7.1 |
| | | | | |
| **RDNA2/3** | | | **32** | Consumer GPUs |
| gfx1100 | RX 7900 XT/XTX | `--offload-arch=gfx1100` | **32** | Your GPU |
| gfx1101 | RX 7000XT | `--offload-arch=gfx1101` | 32 | New in ROCm 7.1 |
| gfx1102 | RX 9000 | `--offload-arch=gfx1102` | 32 | New in ROCm 7.1 |
| gfx1030 | RX 6800/6900 | `--offload-arch=gfx1030` | 32 | RDNA2 |

**⚠️ RDNA3 vs CDNA3 Differences:**

| Feature | RDNA3 (your RX 7900 XT) | CDNA3 (MI300) |
|---------|-------------------------|---------------|
| Wavefront size | **32** | 64 |
| MFMA instructions | ❌ No | ✅ Yes |
| FP16 performance | ✅ Fast | ✅ Very Fast |
| Double precision | Limited | Optimized |
| Use case | Gaming/Consumer AI | Datacenter training |
| Block size tuning | Multiples of 32 | Multiples of 64 |

**For your RX 7900 XT (gfx1100):**
- Use block sizes that are multiples of 32: 32, 64, 96, 128, 256, 512...
- Wave reductions should use power-of-2 strides up to 32
- No MFMA - rely on regular FP16/FP32 operations
- FP16 is still fast and useful for LLM inference

### Generate HSACO (Kernel Object)

For loading kernels at runtime:

```bash
# Method 1: Compile to HSACO directly
hipcc -c --genco my_kernel.hip -o my_kernel.hsaco

# Method 2: Extract from object file
hipcc -c my_kernel.hip -o my_kernel.o
clang-offload-bundler --inputs=my_kernel.o --outputs=my_kernel.hsaco --type=o --targets=hip-amdgcn-amd-amdhsa-gfx942

# Method 3: With specific architecture
hipcc -c --offload-arch=gfx942 my_kernel.hip -o my_kernel.o
```

### Common hipcc Flags

```
-O3                  # Optimize
--offload-arch=XXX   # Target specific GPU
-fgpu-rdc           # Relocatable Device Code (for separate compilation)
-g                  # Include debug info
-I/path/to/include  # Add include path
-L/path/to/lib      # Add library path
-lamdhip64          # Link HIP runtime
```

---

## 4. Rust + HIP Integration

### build.rs Pattern

Create `build.rs` in your project root:

```rust
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=kernels");

    // Only build if rocm feature is enabled
    #[cfg(feature = "rocm")]
    {
        // Find hipcc
        let hipcc = env::var("HIPCC")
            .unwrap_or_else(|_| "/opt/rocm/bin/hipcc".to_string());

        // Kernels to compile
        let kernels = vec![
            ("kernels/scale.hip", "SCALE_HSACO", "scale_kernel"),
            ("kernels/mask.hip", "MASK_HSACO", "mask_kernel"),
            ("kernels/softmax.hip", "SOFTMAX_HSACO", "softmax_kernel"),
        ];

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        for (src_path, env_name, kernel_name) in &kernels {
            let src = PathBuf::from(src_path);
            if !src.exists() {
                println!("cargo:warning=Kernel not found: {}", src_path);
                continue;
            }

            let hsaco_path = out_dir.join(format!("{}.hsaco", kernel_name));

            // Compile hip to hsaco
            let status = std::process::Command::new(&hipcc)
                .arg("-c")
                .arg("--genco")
                .arg("--offload-arch=gfx942")  // Or detect from rocminfo
                .arg("-O3")
                .arg(src_path)
                .arg("-o")
                .arg(&hsaco_path)
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!("cargo:rustc-env={}={}", env_name, hsaco_path.display());
                }
                Ok(s) => {
                    println!("cargo:warning=Failed to compile {}: exit code {:?}", src_path, s.code());
                }
                Err(e) => {
                    println!("cargo:warning=Failed to execute hipcc: {}", e);
                }
            }
        }

        // Link against amdhip64
        println!("cargo:rustc-link-lib=amdhip64");

        // Add ROCm library path
        if let Ok(rocm_path) = env::var("ROCM_PATH") {
            println!("cargo:rustc-link-search={}/lib", rocm_path);
        }
    }
}
```

### FFI Declarations

In your Rust code:

```rust
use std::ffi::{c_void, CString};
use std::ptr;

// HIP FFI bindings
#[link(name = "amdhip64")]
extern "C" {
    fn hipInit(flags: u32) -> i32;
    fn hipGetDeviceCount(count: *mut i32) -> i32;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn hipFree(ptr: *mut c_void) -> i32;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn hipModuleLoad(module: *mut *mut c_void, path: *const i8) -> i32;
    fn hipModuleGetFunction(func: *mut *mut c_void, module: *mut c_void, name: *const i8) -> i32;
    fn hipModuleLaunchKernel(
        func: *mut c_void,
        gridDimX: u32, gridDimY: u32, gridDimZ: u32,
        blockDimX: u32, blockDimY: u32, blockDimZ: u32,
        sharedMemBytes: u32,
        stream: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

const hipMemcpyHostToDevice: i32 = 1;
const hipMemcpyDeviceToHost: i32 = 2;
const hipSuccess: i32 = 0;

pub struct HipBackend {
    // ... fields
}

impl HipBackend {
    pub fn load_kernel(&self, hsaco_path: &str, kernel_name: &str) -> Result<(*mut c_void, *mut c_void), String> {
        unsafe {
            let mut module: *mut c_void = ptr::null_mut();
            let path_cstr = CString::new(hsaco_path).unwrap();

            let result = hipModuleLoad(&mut module, path_cstr.as_ptr());
            if result != hipSuccess {
                return Err(format!("Failed to load module: {}", result));
            }

            let mut func: *mut c_void = ptr::null_mut();
            let name_cstr = CString::new(kernel_name).unwrap();

            let result = hipModuleGetFunction(&mut func, module, name_cstr.as_ptr());
            if result != hipSuccess {
                return Err(format!("Failed to get function: {}", result));
            }

            Ok((module, func))
        }
    }

    pub fn launch_kernel(
        &self,
        func: *mut c_void,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &[*mut c_void],
    ) -> Result<(), String> {
        unsafe {
            let result = hipModuleLaunchKernel(
                func,
                grid_dim.0, grid_dim.1, grid_dim.2,
                block_dim.0, block_dim.1, block_dim.2,
                0,  // sharedMemBytes
                ptr::null_mut(),  // stream
                args.as_ptr() as *mut _,
                ptr::null_mut(),  // extra
            );

            if result != hipSuccess {
                return Err(format!("Kernel launch failed: {}", result));
            }

            Ok(())
        }
    }
}
```

### Cargo.toml Configuration

```toml
[package]
name = "rocmforge"
version = "0.1.0"
edition = "2021"

[features]
default = []
rocm = []  # Enables ROCm/HIP support

[build-dependencies]
cc = "1.0"

[dependencies]
# ... other dependencies
```

---

## 5. Common Commands

### Development Workflow

```bash
# 1. Check GPU status
rocminfo
rocm-smi

# 2. Compile a single kernel
hipcc kernels/scale.hip -o /tmp/test_scale -O3

# 3. Run and test
/tmp/test_scale

# 4. Generate HSACO
hipcc -c --genco kernels/scale.hip -o kernels/scale.hsaco

# 5. Check assembly output
hipcc -S -O3 kernels/scale.hip -o kernels/scale.s

# 6. Build Rust project
cargo build --features rocm

# 7. Run tests
cargo test --features rocm

# 8. Watch GPU usage
watch -n 1 rocm-smi
```

### GPU Monitoring

```bash
# ROCm System Management Interface
rocm-smi

# Show all metrics
rocm-smi --showuse --showtemp --showpower

# Continuous monitoring
watch -n 1 rocm-smi

# Check specific GPU
rocm-smi -d 0 --showuse
```

---

## 6. Troubleshooting

### "hipcc: command not found"

```bash
# Add to PATH
export PATH=/opt/rocm/bin:$PATH

# Or create symlink
sudo ln -s /opt/rocm/bin/hipcc /usr/local/bin/hipcc
```

### "libamdhip64.so: not found"

```bash
# Add to library path
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Or add to ldconfig
echo "/opt/rocm/lib" | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig
```

### "No AMD GPU found"

```bash
# Check if GPU is visible
lspci | grep -i vga
ls /dev/kfd
dmesg | grep -i amdgpu

# Load amdgpu driver
sudo modprobe amdgpu
```

### Kernel Launch Fails

```bash
# Check GPU is accessible
rocminfo

# Try running with simple test first
echo '
#include <hip/hip_runtime.h>
__global__ void test() {}
int main() {
    hipLaunchKernelGGL(test, 1, 1, 0, 0);
    hipDeviceSynchronize();
}
' > /tmp/test.hip
hipcc /tmp/test.hip -o /tmp/test
/tmp/test
```

---

## 7. Code Examples

### Minimal HIP Kernel

```cpp
// hello_hip.hip
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    const int n = 1024;
    float *d_data;

    // Allocate device memory
    hipMalloc(&d_data, n * sizeof(float));

    // Copy data to device
    float h_data[n];
    for (int i = 0; i < n; i++) h_data[i] = i * 1.0f;
    hipMemcpy(d_data, h_data, n * sizeof(float), hipMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(hello_kernel,
                       dim3(numBlocks),
                       dim3(blockSize),
                       0, 0,
                       d_data, n);

    // Copy back
    hipMemcpy(h_data, d_data, n * sizeof(float), hipMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < n; i++) {
        if (h_data[i] != i * 2.0f) {
            printf("Mismatch at %d: got %f, expected %f\n", i, h_data[i], i * 2.0f);
            return 1;
        }
    }

    printf("Success!\n");

    hipFree(d_data);
    return 0;
}
```

### Compile and Run:
```bash
hipcc hello_hip.hip -o hello_hip -O3
./hello_hip
```

### Load Kernel at Runtime

```cpp
// runtime_load.hip
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Note: For runtime loading, compile separately:
// hipcc -c --genco vector_add.hip -o vector_add.hsaco
```

---

## References

### Official Documentation

- [HIP 6.0 Installation Guide](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/how_to_guides/install.html)
- [ROCm Installation via amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.0.0/how-to/amdgpu-install.html)
- [HIP Compilers Documentation](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.4/understand/compilers.html)
- [HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [AMD HIP Programming Guide (PDF)](https://raw.githubusercontent.com/RadeonOpenCompute/ROCm/rocm-4.5.2/AMD_HIP_Programming_Guide.pdf)

### Community Resources

- [ROCm on GitHub](https://github.com/RustNSparks/rocm-rs)
- [ROCm Discussions](https://github.com/ROCm/ROCm/discussions)
- [r/ROCm Subreddit](https://www.reddit.com/r/ROCm/)
- [AMD Lab Notes - ROCm Installation](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-rocm-installation-readme/)
- [ROCm Development Exercises](https://enccs.github.io/amd-rocm-development/exercises-1/)

### Specific Issues & Solutions

- [Compile device-only HIP to HSACO](https://github.com/ROCm/HIP/issues/1098)
- [hipcc compilation discussion](https://github.com/ROCm/Developer-Tools/HIP/issues/190)
- [Linking hipcc object with g++](https://github.com/ROCm/HIP/issues/988)

---

## Quick Reference Card

```bash
# Installation
sudo amdgpu-install --usecase=rocm

# Environment
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/rocm

# Compile
hipcc kernel.hip -o program                # Executable
hipcc -c --genco kernel.hip -o kernel.hsaco # HSACO for runtime
hipcc -S kernel.hip -o kernel.s            # Assembly

# Target architectures
--offload-arch=gfx90a  # MI200
--offload-arch=gfx942  # MI300
--offload-arch=gfx1100 # RX 7000

# Monitoring
rocminfo        # GPU info
rocm-smi        # GPU stats
```

---

> Next: See `implementation_roadmap.md` for specific kernel implementations.
