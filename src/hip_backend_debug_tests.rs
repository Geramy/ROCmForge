#[cfg(test)]
mod hip_backend_debug_tests {
    use crate::backend::hip_backend::*;

    #[test]
    fn test_detect_amd_gpu_step_by_step() {
        println!("Testing detect_amd_gpu step by step...");

        // First initialize HIP
        HipBackend::new().expect("HIP initialization should succeed");

        // Step 1: Get device count
        let mut count: i32 = 0;
        let result = unsafe { super::super::backend::hip_backend::hipGetDeviceCount(&mut count) };
        println!("Device count result: {}, count: {}", result, count);
        assert_eq!(result, 0, "hipGetDeviceCount should succeed");
        assert!(count > 0, "Should have at least one device");

        // Step 2: Get properties using proper alignment
        let mut props = std::mem::MaybeUninit::<HipDeviceProp>::uninit();
        let result = unsafe {
            super::super::backend::hip_backend::hipGetDeviceProperties(props.as_mut_ptr(), 0)
        };
        println!("Get properties result: {}", result);
        assert_eq!(result, 0, "hipGetDeviceProperties should succeed");

        // Step 3: Read the properties safely
        let props = unsafe { props.assume_init() };
        println!("Device name: {}", props.name());
        println!("Total memory: {} bytes", props.total_global_mem());
        println!("Compute units: {}", props.multi_processor_count());
    }
}
