#[link(name = "amdhip64")]
extern "C" {
    fn hipInit(flags: u32) -> i32;
    fn hipGetDeviceCount(count: *mut i32) -> i32;
    fn hipMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn hipFree(ptr: *mut std::ffi::c_void) -> i32;
}

fn main() {
    println!("Testing minimal HIP functionality...");
    
    unsafe {
        let result = hipInit(0);
        println!("hipInit result: {}", result);
        
        if result != 0 {
            return;
        }
        
        let mut count: i32 = 0;
        let result = hipGetDeviceCount(&mut count);
        println!("hipGetDeviceCount result: {}, devices: {}", result, count);
        
        if result != 0 || count == 0 {
            return;
        }
        
        println!("Attempting to allocate 1024 bytes...");
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = hipMalloc(&mut ptr, 1024);
        println!("hipMalloc result: {}, ptr: {:p}", result, ptr);
        
        if result == 0 && !ptr.is_null() {
            println!("Allocation successful, freeing...");
            hipFree(ptr);
            println!("Free successful");
        } else {
            println!("Allocation failed");
        }
    }
}