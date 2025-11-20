//! hipBLAS bindings for ROCmForge
//! Provides safe Rust wrappers around hipBLAS library for GPU matrix operations

use std::ffi::c_void;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HipBlasError {
    #[error("hipBLAS initialization failed: {0}")]
    InitializationFailed(String),
    #[error("hipBLAS operation failed: {0}")]
    OperationFailed(String),
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
}

pub type HipBlasResult<T> = Result<T, HipBlasError>;

// FFI bindings to hipBLAS library
#[link(name = "hipblas")]
extern "C" {
    fn hipblasCreate(handle: *mut *mut c_void) -> i32;
    fn hipblasDestroy(handle: *mut c_void) -> i32;
    fn hipblasSaxpy(
        handle: *mut c_void,
        n: i32,
        alpha: *const f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> i32;
    fn hipblasSscal(handle: *mut c_void, n: i32, alpha: *const f32, x: *mut f32, incx: i32) -> i32;
    fn hipblasSgemm(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        beta: *const f32,
        C: *mut f32,
        ldc: i32,
    ) -> i32;
}

// hipBLAS constants
pub const HIPBLAS_SUCCESS: i32 = 0;
pub const HIPBLAS_OP_N: i32 = 111; // No transpose
pub const HIPBLAS_OP_T: i32 = 112; // Transpose

/// Safe wrapper around hipBLAS handle
#[derive(Debug)]
pub struct HipBlasHandle {
    raw: *mut c_void,
}

impl HipBlasHandle {
    /// Create a new hipBLAS handle
    pub fn new() -> HipBlasResult<Self> {
        let mut handle: *mut c_void = ptr::null_mut();

        let result = unsafe { hipblasCreate(&mut handle) };

        if result != HIPBLAS_SUCCESS {
            return Err(HipBlasError::InitializationFailed(format!(
                "hipblasCreate failed with code {}",
                result
            )));
        }

        if handle.is_null() {
            return Err(HipBlasError::InitializationFailed(
                "hipblasCreate returned null handle".to_string(),
            ));
        }

        Ok(HipBlasHandle { raw: handle })
    }

    /// Get raw handle pointer (for FFI calls)
    pub fn as_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for HipBlasHandle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let result = unsafe { hipblasDestroy(self.raw) };
            if result != HIPBLAS_SUCCESS {
                eprintln!("Warning: hipblasDestroy failed with code {}", result);
            }
        }
    }
}

// SAFETY: HipBlasHandle is Send+Sync because hipBLAS operations are thread-safe
// when using separate handles per thread
unsafe impl Send for HipBlasHandle {}
unsafe impl Sync for HipBlasHandle {}

/// Perform single-precision matrix multiplication: C = alpha * A * B + beta * C
///
/// Arguments:
/// - handle: hipBLAS handle
/// - transa, transb: transpose flags (use HIPBLAS_OP_N for no transpose)
/// - m, n, k: matrix dimensions (A is m×k, B is k×n, C is m×n)
/// - alpha, beta: scalar multipliers
/// - A, B: input matrices
/// - lda, ldb, ldc: leading dimensions
/// - C: output matrix
pub fn sgemm(
    handle: &HipBlasHandle,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    A: *const f32,
    lda: i32,
    B: *const f32,
    ldb: i32,
    beta: f32,
    C: *mut f32,
    ldc: i32,
) -> HipBlasResult<()> {
    let result = unsafe {
        hipblasSgemm(
            handle.as_ptr(),
            transa,
            transb,
            m,
            n,
            k,
            &alpha,
            A,
            lda,
            B,
            ldb,
            &beta,
            C,
            ldc,
        )
    };

    if result != HIPBLAS_SUCCESS {
        return Err(HipBlasError::OperationFailed(format!(
            "hipblasSgemm failed with code {}",
            result
        )));
    }

    Ok(())
}

/// Perform single-precision vector addition: y = alpha * x + y
pub fn saxpy(
    handle: &HipBlasHandle,
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
) -> HipBlasResult<()> {
    let result = unsafe { hipblasSaxpy(handle.as_ptr(), n, &alpha, x, incx, y, incy) };

    if result != HIPBLAS_SUCCESS {
        return Err(HipBlasError::OperationFailed(format!(
            "hipblasSaxpy failed with code {}",
            result
        )));
    }

    Ok(())
}

/// Scale a vector in-place: x = alpha * x
pub fn sscal(
    handle: &HipBlasHandle,
    n: i32,
    alpha: f32,
    x: *mut f32,
    incx: i32,
) -> HipBlasResult<()> {
    let result = unsafe { hipblasSscal(handle.as_ptr(), n, &alpha, x, incx) };

    if result != HIPBLAS_SUCCESS {
        return Err(HipBlasError::OperationFailed(format!(
            "hipblasSscal failed with code {}",
            result
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hipblas_handle_creation() {
        let handle = HipBlasHandle::new();
        assert!(handle.is_ok(), "hipBLAS handle creation should succeed");

        let handle = handle.unwrap();
        assert!(!handle.as_ptr().is_null(), "Handle should not be null");

        // Handle should be destroyed when dropped
    }

    #[test]
    fn test_sgemm_invalid_handle() {
        // Test with null handle - should fail gracefully
        let result = sgemm(
            &HipBlasHandle {
                raw: ptr::null_mut(),
            },
            HIPBLAS_OP_N,
            HIPBLAS_OP_N,
            2,
            2,
            2,
            1.0,
            ptr::null(),
            2,
            ptr::null(),
            2,
            0.0,
            ptr::null_mut(),
            2,
        );

        assert!(result.is_err(), "sgemm with null handle should fail");
    }
}
