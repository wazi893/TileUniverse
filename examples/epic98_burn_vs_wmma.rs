//! EPIC 98: Benchmark Burn NN vs hand-tuned WMMA NN
//!
//! Compares:
//! - Burn's Linear layers with CubeCL/WGPU backend
//! - Reference: WMMA 16x16 NN from EPIC 96 (1.048B organisms/sec)
//!
//! Expected: WMMA faster for simple NN, but Burn enables LSTM/Transformer
//!
//! Run with:
//! ```
//! cargo run --release --example epic98_burn_vs_wmma --features burn-compute
//! ```

#[cfg(feature = "burn-compute")]
mod bench {
    use burn::nn::{Linear, LinearConfig};
    use burn::prelude::*;
    use burn::tensor::backend::Backend;
    use burn_cubecl::CubeBackend;
    use cubecl_wgpu::WgpuRuntime;
    use std::time::Instant;

    // Backend type: CubeCL with WGPU runtime, f32 floats, i32 ints, u8 bools
    type B = CubeBackend<WgpuRuntime, f32, i32, u8>;

    /// Simple feedforward brain matching WMMA architecture: 16 -> 16 -> 5
    #[derive(Module, Debug, Clone)]
    pub struct BurnSimpleNN<Backend: burn::tensor::backend::Backend> {
        layer1: Linear<Backend>,
        layer2: Linear<Backend>,
    }

    impl<Backend: burn::tensor::backend::Backend> BurnSimpleNN<Backend> {
        pub fn new(device: &Backend::Device) -> Self {
            Self {
                layer1: LinearConfig::new(16, 16).init(device),
                layer2: LinearConfig::new(16, 5).init(device),
            }
        }

        pub fn forward(&self, x: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
            let x = self.layer1.forward(x);
            let x = burn::tensor::activation::tanh(x);
            let x = self.layer2.forward(x);
            burn::tensor::activation::tanh(x)
        }
    }

    /// Benchmark a single configuration
    fn benchmark_burn(n_organisms: usize, n_ticks: usize) -> (std::time::Duration, f64) {
        let device = <B as Backend>::Device::default();
        let brain = BurnSimpleNN::<B>::new(&device);

        // Create batched sensor input [n_organisms, 16]
        let sensors = Tensor::<B, 2>::random(
            [n_organisms, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // Warmup - run 10 iterations to ensure GPU is ready
        for _ in 0..10 {
            let output = brain.forward(sensors.clone());
            // Force sync by reading a single value
            let _ = output.slice([0..1, 0..1]).into_data();
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..n_ticks {
            let output = brain.forward(sensors.clone());
            // We need to sync on the last iteration to ensure all work is complete
            if false {
                // Don't sync every iteration - too slow
                let _ = output.into_data();
            }
        }
        // Final sync - force all GPU work to complete
        let final_output = brain.forward(sensors.clone());
        let _ = final_output.into_data();

        let elapsed = start.elapsed();
        let total_evals = (n_organisms * n_ticks) as f64;
        let throughput = total_evals / elapsed.as_secs_f64();

        (elapsed, throughput)
    }

    /// Run the full benchmark suite
    pub fn run() {
        println!("╔═══════════════════════════════════════════════════════════════╗");
        println!("║  EPIC 98: Burn vs WMMA Neural Network Benchmark               ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");
        println!();

        // Print device info
        let device = <B as Backend>::Device::default();
        println!("  Backend: CubeCL + WGPU");
        println!("  Device:  {:?}", device);
        println!();

        // Test configurations: (organisms, ticks)
        let configs = vec![
            (1_000, 1000, "1K org × 1K ticks"),
            (10_000, 1000, "10K org × 1K ticks"),
            (100_000, 100, "100K org × 100 ticks"),
            (1_000_000, 10, "1M org × 10 ticks"),
        ];

        println!(
            "  {:>22} | {:>12} | {:>15} | {:>12}",
            "Configuration", "Time", "Throughput", "vs WMMA"
        );
        println!("  {:-<22}-+-{:-<12}-+-{:-<15}-+-{:-<12}", "", "", "", "");

        // WMMA reference: 1.048B organisms/sec from EPIC 96
        let wmma_throughput = 1.048e9;

        for (n_org, n_ticks, desc) in configs {
            let (elapsed, throughput) = benchmark_burn(n_org, n_ticks);
            let ratio = wmma_throughput / throughput;

            println!(
                "  {:>22} | {:>10.2?} | {:>12.3e}/s | {:>10.1}x",
                desc, elapsed, throughput, ratio
            );
        }

        println!();
        println!("  ┌─────────────────────────────────────────────────────────────┐");
        println!("  │ Reference: WMMA 16×16 NN = 1.048B organisms/sec (EPIC 96)   │");
        println!("  ├─────────────────────────────────────────────────────────────┤");
        println!("  │ Decision Thresholds:                                        │");
        println!("  │   • Within 2×  → Use Burn for flexibility (LSTM, etc.)      │");
        println!("  │   • 2-10× slower → Hybrid: Burn for complex, WMMA for simple│");
        println!("  │   • >10× slower → Keep WMMA for all simple NN workloads     │");
        println!("  └─────────────────────────────────────────────────────────────┘");
        println!();

        // Additional: test larger batch to find peak throughput
        println!("  Peak throughput search (larger batches):");
        println!();

        let peak_configs = vec![
            (2_000_000, 5, "2M org × 5 ticks"),
            (4_000_000, 2, "4M org × 2 ticks"),
        ];

        for (n_org, n_ticks, desc) in peak_configs {
            match std::panic::catch_unwind(|| benchmark_burn(n_org, n_ticks)) {
                Ok((elapsed, throughput)) => {
                    let ratio = wmma_throughput / throughput;
                    println!(
                        "  {:>22} | {:>10.2?} | {:>12.3e}/s | {:>10.1}x",
                        desc, elapsed, throughput, ratio
                    );
                }
                Err(_) => {
                    println!("  {:>22} | OUT OF MEMORY", desc);
                }
            }
        }

        println!();
        println!("  Benchmark complete.");
    }
}

fn main() {
    #[cfg(feature = "burn-compute")]
    bench::run();

    #[cfg(not(feature = "burn-compute"))]
    {
        println!("EPIC 98: Burn vs WMMA Benchmark");
        println!();
        println!("This benchmark requires the burn-compute feature.");
        println!(
            "Run with: cargo run --release --example epic98_burn_vs_wmma --features burn-compute"
        );
    }
}
