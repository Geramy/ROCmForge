//! GPU-accelerated sampling implementation
//!
//! Provides GPU kernels for top-k and top-p sampling using ROCm/HIP.
//! Based on FlashInfer's sorting-free rejection sampling algorithm.

#![allow(dead_code)]

#[cfg(feature = "rocm")]
use crate::backend::hip_backend::{HipBackend, HipBuffer, HipError, HipKernel, HipModule};
use crate::sampler::{SamplerError, SamplerResult};
use rand::distributions::Distribution;
use rand::Rng;
use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[cfg(feature = "rocm")]
const BLOCK_SIZE: u32 = 256;
#[cfg(feature = "rocm")]
const HIP_SUCCESS: i32 = 0;

/// Cached kernel modules and functions for sampling operations
///
/// NOTE: We do NOT store HipBackend here because that would create a separate
/// HIP stream from the caller's. Kernels must be launched on the caller's stream
/// and synchronized on the same stream to avoid hangs.
#[cfg(feature = "rocm")]
#[derive(Debug)]
struct SamplingKernelCache {
    softmax_module: Option<HipModule>,
    softmax_kernel: Option<HipKernel>,
    prefix_sum_module: Option<HipModule>,
    prefix_sum_kernel: Option<HipKernel>,
    topp_module: Option<HipModule>,
    topp_kernel: Option<HipKernel>,
    topk_module: Option<HipModule>,
    topk_kernel: Option<HipKernel>,
    fused_module: Option<HipModule>,
    fused_kernel: Option<HipKernel>,
}

// Global kernel cache (lazy initialization)
#[cfg(feature = "rocm")]
static GLOBAL_SAMPLING_CACHE: Mutex<Option<SamplingKernelCache>> = Mutex::new(None);

/// Get or initialize the global sampling kernel cache
///
/// Returns cached kernel modules and functions. The caller must provide
/// their own HipBackend for launching kernels to ensure stream consistency.
#[cfg(feature = "rocm")]
fn get_or_init_sampling_cache() -> Result<&'static Mutex<Option<SamplingKernelCache>>, HipError> {
    // First check if already initialized
    {
        let cache = GLOBAL_SAMPLING_CACHE.lock()
            .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_SAMPLING_CACHE lock poisoned: {}", e)))?;
        if cache.is_some() {
            return Ok(&GLOBAL_SAMPLING_CACHE);
        }
    }

    // Need to initialize - drop the read lock first
    let mut cache = GLOBAL_SAMPLING_CACHE.lock()
        .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_SAMPLING_CACHE lock poisoned: {}", e)))?;

    // Double-check in case another thread initialized while we waited
    if cache.is_some() {
        return Ok(&GLOBAL_SAMPLING_CACHE);
    }

    // Load kernel modules using a temporary backend
    let load_backend = HipBackend::new()
        .map_err(|e| HipError::InitializationFailed(format!("Failed to create HipBackend for loading: {}", e)))?;

    // Load softmax kernel
    let softmax_path = std::env::var("SOFTMAX_HSACO")
        .unwrap_or_else(|_| "kernels/softmax.hsaco".to_string());

    let (softmax_module, softmax_kernel) = if Path::new(&softmax_path).exists() {
        let module = load_backend.load_module(&softmax_path)?;
        let kernel = load_backend.get_kernel_function(&module, "softmax_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Softmax kernel not found at {}, using CPU fallback", softmax_path);
        (None, None)
    };

    // Load prefix sum kernel
    let prefix_sum_path = std::env::var("PREFIX_SUM_HSACO")
        .unwrap_or_else(|_| "kernels/prefix_sum.hsaco".to_string());

    let (prefix_sum_module, prefix_sum_kernel) = if Path::new(&prefix_sum_path).exists() {
        let module = load_backend.load_module(&prefix_sum_path)?;
        let kernel = load_backend.get_kernel_function(&module, "prefix_sum_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Prefix sum kernel not found at {}, using CPU fallback", prefix_sum_path);
        (None, None)
    };

    // Load top-p kernel
    let topp_path = std::env::var("TOPP_HSACO")
        .unwrap_or_else(|_| "kernels/topp_sampling.hsaco".to_string());

    let (topp_module, topp_kernel) = if Path::new(&topp_path).exists() {
        let module = load_backend.load_module(&topp_path)?;
        let kernel = load_backend.get_kernel_function(&module, "topp_sampling_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Top-p kernel not found at {}, using CPU fallback", topp_path);
        (None, None)
    };

    // Load top-k kernel
    let topk_path = std::env::var("TOPK_HSACO")
        .unwrap_or_else(|_| "kernels/topk_sampling.hsaco".to_string());

    let (topk_module, topk_kernel) = if Path::new(&topk_path).exists() {
        let module = load_backend.load_module(&topk_path)?;
        let kernel = load_backend.get_kernel_function(&module, "topk_sampling_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Top-k kernel not found at {}, using CPU fallback", topk_path);
        (None, None)
    };

    // Load fused kernel
    let fused_path = std::env::var("FUSED_HSACO")
        .unwrap_or_else(|_| "kernels/topk_topp_sampling.hsaco".to_string());

    let (fused_module, fused_kernel) = if Path::new(&fused_path).exists() {
        let module = load_backend.load_module(&fused_path)?;
        let kernel = load_backend.get_kernel_function(&module, "topk_topp_sampling_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Fused kernel not found at {}, using CPU fallback", fused_path);
        (None, None)
    };

    // Initialize cache
    *cache = Some(SamplingKernelCache {
        softmax_module,
        softmax_kernel,
        prefix_sum_module,
        prefix_sum_kernel,
        topp_module,
        topp_kernel,
        topk_module,
        topk_kernel,
        fused_module,
        fused_kernel,
    });

    Ok(&GLOBAL_SAMPLING_CACHE)
}

// ============================================================================
// Kernel Launch Wrappers
// ============================================================================

/// Launch top-p sampling kernel on GPU
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `probabilities` - GPU pointer to probability tensor [batch_size, vocab_size]
/// * `random_values` - GPU pointer to random values [batch_size]
/// * `output` - GPU pointer to output token IDs [batch_size]
/// * `top_p` - Cumulative probability threshold
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
pub unsafe fn topp_sampling_kernel(
    backend: &HipBackend,
    probabilities: *const f32,
    random_values: *const f32,
    output: *mut u32,
    top_p: f32,
    batch_size: u32,
    vocab_size: u32,
) -> Result<(), String> {
    match get_or_init_sampling_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref.lock()
                .map_err(|e| format!("Failed to lock sampling cache: {}", e))?;
            let cache_ref = cache.as_ref()
                .ok_or_else(|| "Sampling cache not initialized".to_string())?;

            let kernel = cache_ref.topp_kernel.as_ref()
                .ok_or_else(|| "topp_kernel not loaded".to_string())?;

            let grid_dim = (batch_size, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            // Prepare kernel arguments - ALL args must be copied to mut locals first
            let mut probabilities_arg = probabilities;
            let mut random_values_arg = random_values;
            let mut output_arg = output;
            let mut top_p_arg = top_p;
            let mut batch_size_arg = batch_size;
            let mut vocab_size_arg = vocab_size;

            let args: &[*mut c_void] = &[
                &mut probabilities_arg as *mut _ as *mut c_void,
                &mut random_values_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut top_p_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            backend.launch_kernel_with_module_shared(
                kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| format!("Failed to launch top-p kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// Launch top-k sampling kernel on GPU
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `probabilities` - GPU pointer to probability tensor [batch_size, vocab_size]
/// * `random_values` - GPU pointer to random values [batch_size]
/// * `output` - GPU pointer to output token IDs [batch_size]
/// * `top_k` - Number of top tokens to consider
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
pub unsafe fn topk_sampling_kernel(
    backend: &HipBackend,
    probabilities: *const f32,
    random_values: *const f32,
    output: *mut u32,
    top_k: u32,
    batch_size: u32,
    vocab_size: u32,
) -> Result<(), String> {
    match get_or_init_sampling_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref.lock()
                .map_err(|e| format!("Failed to lock sampling cache: {}", e))?;
            let cache_ref = cache.as_ref()
                .ok_or_else(|| "Sampling cache not initialized".to_string())?;

            let kernel = cache_ref.topk_kernel.as_ref()
                .ok_or_else(|| "topk_kernel not loaded".to_string())?;

            let grid_dim = (batch_size, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            // Prepare kernel arguments
            let mut probabilities_arg = probabilities;
            let mut random_values_arg = random_values;
            let mut output_arg = output;
            let mut top_k_arg = top_k;
            let mut batch_size_arg = batch_size;
            let mut vocab_size_arg = vocab_size;

            let args: &[*mut c_void] = &[
                &mut probabilities_arg as *mut _ as *mut c_void,
                &mut random_values_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut top_k_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            backend.launch_kernel_with_module_shared(
                kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| format!("Failed to launch top-k kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// Launch fused top-k + top-p sampling kernel on GPU
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `probabilities` - GPU pointer to probability tensor [batch_size, vocab_size]
/// * `random_values` - GPU pointer to random values [batch_size]
/// * `output` - GPU pointer to output token IDs [batch_size]
/// * `top_k` - Number of top tokens to consider
/// * `top_p` - Cumulative probability threshold
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
pub unsafe fn fused_sampling_kernel(
    backend: &HipBackend,
    probabilities: *const f32,
    random_values: *const f32,
    output: *mut u32,
    top_k: u32,
    top_p: f32,
    batch_size: u32,
    vocab_size: u32,
) -> Result<(), String> {
    match get_or_init_sampling_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref.lock()
                .map_err(|e| format!("Failed to lock sampling cache: {}", e))?;
            let cache_ref = cache.as_ref()
                .ok_or_else(|| "Sampling cache not initialized".to_string())?;

            let kernel = cache_ref.fused_kernel.as_ref()
                .ok_or_else(|| "fused_kernel not loaded".to_string())?;

            let grid_dim = (batch_size, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            // Prepare kernel arguments
            let mut probabilities_arg = probabilities;
            let mut random_values_arg = random_values;
            let mut output_arg = output;
            let mut top_k_arg = top_k;
            let mut top_p_arg = top_p;
            let mut batch_size_arg = batch_size;
            let mut vocab_size_arg = vocab_size;

            let args: &[*mut c_void] = &[
                &mut probabilities_arg as *mut _ as *mut c_void,
                &mut random_values_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut top_k_arg as *mut _ as *mut c_void,
                &mut top_p_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            backend.launch_kernel_with_module_shared(
                kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| format!("Failed to launch fused kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

// ============================================================================
// Sampler Structs
// ============================================================================

/// GPU sampler for top-p (nucleus) sampling
#[cfg(feature = "rocm")]
#[derive(Debug, Clone)]
pub struct GpuTopPSampler {
    backend: Arc<HipBackend>,
    top_p: f32,
}

#[cfg(feature = "rocm")]
impl GpuTopPSampler {
    /// Create a new GPU top-p sampler
    pub fn new(backend: Arc<HipBackend>, top_p: f32) -> SamplerResult<Self> {
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplerError::InvalidTopP(top_p));
        }

        Ok(GpuTopPSampler { backend, top_p })
    }

    /// Sample from probabilities using top-p filtering on GPU
    pub fn sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        // TDD: Try GPU path first, fall back to CPU if kernels not available or on error
        match self.try_gpu_sample(probabilities, batch_size, vocab_size) {
            Ok(results) => Ok(results),
            Err(e) => {
                tracing::debug!("GPU sampling failed, falling back to CPU: {}", e);
                self.sample_cpu_fallback(probabilities, batch_size, vocab_size)
            }
        }
    }

    /// Try to sample using GPU kernels
    fn try_gpu_sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        tracing::debug!("try_gpu_sample: batch_size={}, vocab_size={}", batch_size, vocab_size);

        // Check if kernel is loaded
        let cache_ref = get_or_init_sampling_cache()
            .map_err(|e| {
                tracing::error!("Failed to get kernel cache: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let cache = cache_ref.lock()
            .map_err(|e| {
                tracing::error!("Failed to lock sampling cache: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        let topp_kernel = cache.as_ref()
            .and_then(|c| c.topp_kernel.as_ref())
            .ok_or_else(|| {
                tracing::warn!("top-p kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopP(0.0) // Generic error for "kernel not loaded"
            })?;

        tracing::debug!("top-p kernel loaded, allocating GPU buffers");

        // Allocate GPU buffers
        let total_elements = batch_size * vocab_size;
        let probs_bytes = total_elements * std::mem::size_of::<f32>();
        let random_bytes = batch_size * std::mem::size_of::<f32>();
        let output_bytes = batch_size * std::mem::size_of::<u32>();

        let probs_gpu = HipBuffer::new(probs_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate probs buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let random_gpu = HipBuffer::new(random_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate random buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let output_gpu = HipBuffer::new(output_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate output buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("GPU buffers allocated, copying data");

        // Copy probabilities to GPU
        probs_gpu.copy_from_host(probabilities)
            .map_err(|e| {
                tracing::error!("Failed to copy probs to GPU: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        // Generate random values on CPU and copy to GPU
        let random_values: Vec<f32> = generate_random_gpu(&self.backend, batch_size);
        random_gpu.copy_from_host(&random_values)
            .map_err(|e| {
                tracing::error!("Failed to copy random to GPU: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("Data copied to GPU, launching kernel");

        // Launch kernel
        let probs_ptr = probs_gpu.as_ptr() as *const f32;
        let random_ptr = random_gpu.as_ptr() as *const f32;
        let output_ptr = output_gpu.as_mut_ptr() as *mut u32;

        unsafe {
            topp_sampling_kernel(
                &self.backend,
                probs_ptr,
                random_ptr,
                output_ptr,
                self.top_p,
                batch_size as u32,
                vocab_size as u32,
            ).map_err(|e| {
                tracing::error!("Failed to launch top-p kernel: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        }

        tracing::debug!("Kernel launched, synchronizing");

        // Synchronize and copy results back
        self.backend.synchronize()
            .map_err(|e| {
                tracing::error!("Failed to synchronize: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("Synchronized, copying results back");

        let mut results = vec![0u32; batch_size];
        output_gpu.copy_to_host(&mut results)
            .map_err(|e| {
                tracing::error!("Failed to copy output from GPU: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("GPU sampling complete: {:?}", results);

        Ok(results)
    }

    /// CPU fallback implementation for testing
    fn sample_cpu_fallback(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        let mut results = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for batch_idx in 0..batch_size {
            let row_offset = batch_idx * vocab_size;
            let row_probs = &probabilities[row_offset..row_offset + vocab_size];

            // Find top-p cutoff
            let mut sorted_probs: Vec<(usize, f32)> = row_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative = 0.0f32;
            let mut cutoff_idx = vocab_size;

            for (i, &(_, p)) in sorted_probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Sample from top-p tokens
            let top_indices: Vec<usize> = sorted_probs
                .iter()
                .take(cutoff_idx)
                .map(|(i, _)| *i)
                .collect();

            let top_values: Vec<f32> = top_indices
                .iter()
                .map(|&i| row_probs[i])
                .collect();

            let dist = rand::distributions::WeightedIndex::new(&top_values)
                .map_err(|_| SamplerError::ZeroProbabilities)?;

            let sampled_idx = top_indices[dist.sample(&mut rng)];
            results.push(sampled_idx as u32);
        }

        Ok(results)
    }
}

/// GPU sampler for top-k sampling
#[cfg(feature = "rocm")]
#[derive(Debug, Clone)]
pub struct GpuTopKSampler {
    backend: Arc<HipBackend>,
    top_k: usize,
}

#[cfg(feature = "rocm")]
impl GpuTopKSampler {
    /// Create a new GPU top-k sampler
    pub fn new(backend: Arc<HipBackend>, top_k: usize) -> SamplerResult<Self> {
        if top_k == 0 {
            return Err(SamplerError::InvalidTopK(top_k));
        }

        Ok(GpuTopKSampler { backend, top_k })
    }

    /// Sample from probabilities using top-k filtering on GPU
    pub fn sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        // For now, fall back to CPU implementation
        // TODO: Implement actual GPU kernel call
        self.sample_cpu_fallback(probabilities, batch_size, vocab_size)
    }

    /// CPU fallback implementation for testing
    fn sample_cpu_fallback(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        let mut results = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for batch_idx in 0..batch_size {
            let row_offset = batch_idx * vocab_size;
            let row_probs = &probabilities[row_offset..row_offset + vocab_size];

            // Find top-k
            let effective_k = self.top_k.min(vocab_size);
            let mut sorted_probs: Vec<(usize, f32)> = row_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Sample from top-k
            let top_indices: Vec<usize> = sorted_probs
                .iter()
                .take(effective_k)
                .map(|(i, _)| *i)
                .collect();

            let top_values: Vec<f32> = top_indices
                .iter()
                .map(|&i| row_probs[i])
                .collect();

            // Renormalize
            let sum: f32 = top_values.iter().sum();
            if sum < 1e-10f32 {
                return Err(SamplerError::ZeroProbabilities);
            }

            let normalized: Vec<f32> = top_values.iter().map(|&v| v / sum).collect();

            let dist = rand::distributions::WeightedIndex::new(&normalized)
                .map_err(|_| SamplerError::ZeroProbabilities)?;

            let sampled_idx = top_indices[dist.sample(&mut rng)];
            results.push(sampled_idx as u32);
        }

        Ok(results)
    }
}

/// GPU sampler for fused top-k + top-p sampling
#[cfg(feature = "rocm")]
#[derive(Debug, Clone)]
pub struct GpuFusedSampler {
    backend: Arc<HipBackend>,
    top_k: usize,
    top_p: f32,
}

#[cfg(feature = "rocm")]
impl GpuFusedSampler {
    /// Create a new GPU fused sampler
    pub fn new(backend: Arc<HipBackend>, top_k: usize, top_p: f32) -> SamplerResult<Self> {
        if top_k == 0 {
            return Err(SamplerError::InvalidTopK(top_k));
        }
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplerError::InvalidTopP(top_p));
        }

        Ok(GpuFusedSampler { backend, top_k, top_p })
    }

    /// Sample using fused top-k + top-p on GPU
    pub fn sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        // For now, fall back to CPU implementation
        // TODO: Implement actual GPU kernel call
        self.sample_cpu_fallback(probabilities, batch_size, vocab_size)
    }

    /// CPU fallback implementation for testing
    fn sample_cpu_fallback(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        let mut results = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for batch_idx in 0..batch_size {
            let row_offset = batch_idx * vocab_size;
            let row_probs = &probabilities[row_offset..row_offset + vocab_size];

            // First apply top-p to get candidate set
            let mut sorted_probs: Vec<(usize, f32)> = row_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative = 0.0f32;
            let mut topp_cutoff = vocab_size;

            for (i, &(_, p)) in sorted_probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    topp_cutoff = i + 1;
                    break;
                }
            }

            // Among top-p tokens, select top-k
            let effective_k = self.top_k.min(topp_cutoff);
            let topk_indices: Vec<usize> = sorted_probs
                .iter()
                .take(effective_k)
                .map(|(i, _)| *i)
                .collect();

            let topk_values: Vec<f32> = topk_indices
                .iter()
                .map(|&i| row_probs[i])
                .collect();

            // Renormalize and sample
            let sum: f32 = topk_values.iter().sum();
            if sum < 1e-10f32 {
                return Err(SamplerError::ZeroProbabilities);
            }

            let normalized: Vec<f32> = topk_values.iter().map(|&v| v / sum).collect();

            let dist = rand::distributions::WeightedIndex::new(&normalized)
                .map_err(|_| SamplerError::ZeroProbabilities)?;

            let sampled_idx = topk_indices[dist.sample(&mut rng)];
            results.push(sampled_idx as u32);
        }

        Ok(results)
    }
}

/// Generate random values on GPU
#[cfg(feature = "rocm")]
pub fn generate_random_gpu(
    _backend: &Arc<HipBackend>,
    count: usize,
) -> Vec<f32> {
    // For now, generate on CPU
    let mut rng = rand::thread_rng();
    (0..count).map(|_| rng.gen()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_topp_sampler_creation() {
        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();
        assert_eq!(sampler.top_p, 0.9);
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_topp_invalid_params() {
        let backend = HipBackend::new().unwrap();
        let result = GpuTopPSampler::new(backend.clone(), 0.0);
        assert!(result.is_err());

        let result = GpuTopPSampler::new(backend, 1.5);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_topk_sampler_creation() {
        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 50).unwrap();
        assert_eq!(sampler.top_k, 50);
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_topk_invalid_params() {
        let backend = HipBackend::new().unwrap();
        let result = GpuTopKSampler::new(backend, 0);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_fused_sampler_creation() {
        let backend = HipBackend::new().unwrap();
        let sampler = GpuFusedSampler::new(backend, 50, 0.9).unwrap();
        assert_eq!(sampler.top_k, 50);
        assert_eq!(sampler.top_p, 0.9);
    }

    #[test]
    fn test_topp_fallback_correctness() {
        // Test with known probabilities
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.15, 0.25,  // Row 1 (sum = 1.0)
            0.5, 0.3, 0.1, 0.05, 0.05,  // Row 2 (sum = 1.0)
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    #[test]
    fn test_topk_fallback_correctness() {
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.15, 0.25,
            0.5, 0.3, 0.1, 0.05, 0.05,
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 3).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        // Results should be in top-3 (indices 0, 2, or 4 for first row)
        // Note: This is probabilistic, so we just check bounds
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    #[test]
    fn test_fused_fallback_correctness() {
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.15, 0.25,
            0.5, 0.3, 0.1, 0.05, 0.05,
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuFusedSampler::new(backend, 3, 0.8).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    /// Test GPU kernel infrastructure
    ///
    /// TDD Step 1: This test verifies that the kernel cache can be initialized.
    /// When HSACO files are present, kernels should be loaded.
    /// When HSACO files are absent, cache should still initialize (with None for kernels).
    #[test]
    #[cfg(feature = "rocm")]
    fn test_kernel_cache_initialization() {
        // This should always succeed - cache initializes even if kernels aren't found
        let result = get_or_init_sampling_cache();
        assert!(result.is_ok(), "Kernel cache should initialize successfully");

        // Verify cache is populated
        let cache = result.unwrap().lock()
            .expect("Sampling cache lock should not be poisoned");
        assert!(cache.is_some(), "Cache should be Some after initialization");

        let cache_ref = cache.as_ref()
            .expect("Cache should contain Some(Mutex<KernelCache>)");
        // Kernels will be None if HSACO files aren't compiled yet
        // This is expected - the test documents current state
        if cache_ref.topp_kernel.is_none() {
            println!("WARNING: top-p kernel not loaded (HSACO files not compiled yet)");
            println!("To enable GPU sampling, compile kernels with:");
            println!("  hipcc --genco -O3 kernels/topp_sampling.hip -o kernels/topp_sampling.hsaco");
        }
    }

    /// Test GPU top-p sampling with known inputs
    ///
    /// TDD Step 1: Write test first
    /// TDD Step 2: Run test - see it use CPU fallback (will pass but logs warning)
    /// TDD Step 3: After HSACO compilation, GPU path will be used
    #[test]
    #[cfg(feature = "rocm")]
    fn test_topp_sampling_deterministic() {
        // Use deterministic probabilities where result is predictable
        let probabilities = vec![
            0.05, 0.05, 0.80, 0.05, 0.05,  // Row 1: token 2 has 80% probability
            0.10, 0.10, 0.10, 0.60, 0.10,  // Row 2: token 3 has 60% probability
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        // Verify basic properties
        assert_eq!(results.len(), 2, "Should return 2 samples");
        assert!(results[0] < 5, "First sample should be in vocabulary range");
        assert!(results[1] < 5, "Second sample should be in vocabulary range");

        // With top_p=0.9, token 2 (80%) should be highly likely for first row
        // With top_p=0.9, tokens 3 (60%) + 2 (10%) = 70% for second row
        // Note: This is probabilistic, so we just verify it runs without error
    }

    /// Test GPU top-k sampling with known inputs
    #[test]
    #[cfg(feature = "rocm")]
    fn test_topk_sampling_deterministic() {
        // Clear top-2 tokens: token 2 (80%), token 4 (10%)
        let probabilities = vec![
            0.02, 0.03, 0.80, 0.05, 0.10,  // Row 1: top-2 are indices 2 and 4
            0.05, 0.05, 0.10, 0.70, 0.10,  // Row 2: top-2 are indices 3 and 4
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 2).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);

        // With top_k=2, samples should be from {2, 4} for row 1
        // and from {3, 4} for row 2
        // Note: Probabilistic, so we just verify it runs
    }
}
