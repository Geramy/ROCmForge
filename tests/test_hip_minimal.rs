use std::ffi::c_void;
use std::ptr;

#[link(name = "amdhip64")]
extern "C" {
    fn hipInit(flags: u32) -> i32;
    fn hipGetDeviceCount(count: *mut i32) -> i32;
}

const hipSuccess: i32 = 0;

fn main() {
    println!("Testing HIP initialization...");
    
    unsafe {
        println!("Calling hipInit(0)...");
        let result = hipInit(0);
        println!("hipInit result: {}", result);
        
        if result == hipSuccess {
            println!("HIP initialized successfully!");
            
            let mut count: i32 = 0;
            println!("Calling hipGetDeviceCount...");
            let device_result = hipGetDeviceCount(&mut count);
            println!("hipGetDeviceCount result: {}, count: {}", device_result, count);
        } else {
            println!("HIP initialization failed with code: {}", result);
        }
    }
}