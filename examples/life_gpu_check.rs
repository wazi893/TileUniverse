//! Oracle-verify the packed GPU Game-of-Life kernel against the CPU reference,
//! then measure real cell-update throughput on the device.
//!
//!     cargo run --release --features cuda --example life_gpu_check

use engine::cuda_life::{measure_life_throughput_gpu, verify_gpu_matches_cpu};
use logic_fabric_core::cuda::CudaRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = CudaRuntime::new()?;
    println!("device: {}", rt.device_name().unwrap_or_else(|_| "?".into()));

    println!("\n== correctness (GPU must equal CPU reference) ==");
    for &(w, h, gens, seed) in &[(256usize, 256usize, 50u32, 1u64), (320, 256, 77, 2), (512, 256, 120, 3)] {
        let ok = verify_gpu_matches_cpu(&rt, w, h, gens, seed)?;
        println!("  {w}x{h}  {gens} gens  seed {seed}: {}", if ok { "MATCH" } else { "MISMATCH" });
        assert!(ok, "GPU kernel diverged from CPU reference");
    }

    println!("\n== throughput (random soup, kernel-only timing) ==");
    for &(w, h, gens) in &[(1024usize, 1024usize, 200u32), (2048, 2048, 200), (4096, 4096, 100), (8192, 8192, 100)] {
        let t = measure_life_throughput_gpu(&rt, w, h, gens, 0x1234_5678)?;
        println!(
            "  {w}x{h}  x{gens} gens:  {:.3} ms  ->  {:.2} G cell-updates/s",
            t.elapsed_secs * 1e3,
            t.cell_updates_per_sec / 1e9
        );
    }
    Ok(())
}
