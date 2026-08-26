//! Quick check: per-tick latency on large sparse grid
//! If GPU prefix sum works, should be <1ms per tick
//! If CPU-bound, would be >10ms per tick for 1B tiles

#[cfg(feature = "cuda")]
use engine::cuda_tiles::{GpuSparseContext, run_sparse_tick};
#[cfg(feature = "cuda")]
use logic_fabric_core::cuda::CudaRuntime;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        println!("Requires --features cuda");
    }

    #[cfg(feature = "cuda")]
    {
        let rt = CudaRuntime::new().expect("CUDA init failed");
        println!("GPU: {}", rt.device_name().unwrap_or("Unknown".into()));

        // Test different grid sizes
        for (width, height, label) in [
            (1024, 1024, "1M tiles"),
            (4096, 4096, "16M tiles"),
            (8192, 8192, "67M tiles"),
            (16384, 16384, "268M tiles"),
            (32768, 32768, "1B tiles"),
        ] {
            let tile_count = width * height;
            let num_words = (tile_count + 63) / 64;

            // Create sparse context (all tiles initially dirty)
            let mut ctx = match GpuSparseContext::new_all_dirty(&rt, width, height) {
                Ok(c) => c,
                Err(e) => {
                    println!("{}: SKIP ({})", label, e);
                    continue;
                }
            };

            // Warmup
            for _ in 0..5 {
                let _ = run_sparse_tick(&rt, &mut ctx);
            }

            // Measure 10 ticks
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = run_sparse_tick(&rt, &mut ctx);
            }
            let elapsed = start.elapsed();
            let per_tick_us = elapsed.as_micros() as f64 / 10.0;
            let per_tick_ms = per_tick_us / 1000.0;

            // Analysis
            let verdict = if per_tick_ms < 1.0 {
                "✓ GPU prefix sum working"
            } else if per_tick_ms < 5.0 {
                "~ borderline"
            } else {
                "✗ likely CPU-bound"
            };

            println!(
                "{:>12}: {:>8.2} ms/tick  ({} words) {}",
                label, per_tick_ms, num_words, verdict
            );
        }
    }
}
