//! EPIC 99: LSTM Brain Throughput Benchmark
//!
//! Measures LSTM throughput to validate estimated 100-200M organisms/sec
//! and compare against SimpleBrain (464M/sec) and WMMA (1.048B/sec).
//!
//! Run with:
//! ```
//! cargo run --release --example epic99_lstm_benchmark --features burn-compute
//! ```

#[cfg(feature = "burn-compute")]
mod bench {
    use burn::prelude::*;
    use burn::tensor::backend::Backend;
    use burn_cubecl::CubeBackend;
    use cubecl_wgpu::WgpuRuntime;
    use engine::burn_brain::{BurnLstmState, LstmBrain, SimpleBrain};
    use std::time::Instant;

    type B = CubeBackend<WgpuRuntime, f32, i32, u8>;

    fn benchmark_simple(n_organisms: usize, n_ticks: usize) -> (std::time::Duration, f64) {
        let device = <B as Backend>::Device::default();
        let brain = SimpleBrain::<B>::new(16, 16, 5, &device);

        let sensors = Tensor::<B, 2>::random(
            [n_organisms, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // Warmup
        for _ in 0..10 {
            let output = brain.forward(sensors.clone());
            let _ = output.slice([0..1, 0..1]).into_data();
        }

        let start = Instant::now();
        for _ in 0..n_ticks {
            let _ = brain.forward(sensors.clone());
        }
        let final_output = brain.forward(sensors.clone());
        let _ = final_output.into_data();

        let elapsed = start.elapsed();
        let throughput = (n_organisms * n_ticks) as f64 / elapsed.as_secs_f64();
        (elapsed, throughput)
    }

    fn benchmark_lstm_stateless(n_organisms: usize, n_ticks: usize) -> (std::time::Duration, f64) {
        let device = <B as Backend>::Device::default();
        let brain = LstmBrain::<B>::new(16, 16, 5, &device);

        let sensors = Tensor::<B, 2>::random(
            [n_organisms, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // Warmup (stateless - no state persistence)
        for _ in 0..10 {
            let (output, _) = brain.forward(sensors.clone(), None);
            let _ = output.slice([0..1, 0..1]).into_data();
        }

        let start = Instant::now();
        for _ in 0..n_ticks {
            let _ = brain.forward(sensors.clone(), None);
        }
        let (final_output, _) = brain.forward(sensors.clone(), None);
        let _ = final_output.into_data();

        let elapsed = start.elapsed();
        let throughput = (n_organisms * n_ticks) as f64 / elapsed.as_secs_f64();
        (elapsed, throughput)
    }

    fn benchmark_lstm_stateful(n_organisms: usize, n_ticks: usize) -> (std::time::Duration, f64) {
        let device = <B as Backend>::Device::default();
        let brain = LstmBrain::<B>::new(16, 16, 5, &device);

        let sensors = Tensor::<B, 2>::random(
            [n_organisms, 16],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // Warmup with state persistence
        let mut state: Option<BurnLstmState<B>> = None;
        for _ in 0..10 {
            let (output, new_state) = brain.forward(sensors.clone(), state);
            state = Some(new_state);
            let _ = output.slice([0..1, 0..1]).into_data();
        }

        // Reset state for benchmark
        let mut state: Option<BurnLstmState<B>> = None;

        let start = Instant::now();
        for _ in 0..n_ticks {
            let (_, new_state) = brain.forward(sensors.clone(), state);
            state = Some(new_state);
        }
        let (final_output, _) = brain.forward(sensors.clone(), state);
        let _ = final_output.into_data();

        let elapsed = start.elapsed();
        let throughput = (n_organisms * n_ticks) as f64 / elapsed.as_secs_f64();
        (elapsed, throughput)
    }

    pub fn run() {
        println!("╔═══════════════════════════════════════════════════════════════╗");
        println!("║  EPIC 99: LSTM Brain Throughput Benchmark                     ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");
        println!();

        let device = <B as Backend>::Device::default();
        println!("  Backend: CubeCL + WGPU");
        println!("  Device:  {:?}", device);
        println!();

        // Reference values
        let wmma_throughput = 1.048e9;

        // Test configurations
        let configs = vec![
            (10_000, 1000, "10K org × 1K ticks"),
            (100_000, 100, "100K org × 100 ticks"),
            (1_000_000, 10, "1M org × 10 ticks"),
        ];

        // Header
        println!(
            "  ┌────────────────────────────────────────────────────────────────────────────────┐"
        );
        println!(
            "  │ SimpleBrain vs LstmBrain Comparison                                           │"
        );
        println!(
            "  └────────────────────────────────────────────────────────────────────────────────┘"
        );
        println!();

        println!(
            "  {:>22} | {:>12} | {:>12} | {:>12}",
            "Configuration", "Simple", "LSTM (no st)", "LSTM (state)"
        );
        println!("  {:-<22}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "");

        for (n_org, n_ticks, desc) in &configs {
            let (_, simple_tp) = benchmark_simple(*n_org, *n_ticks);
            let (_, lstm_stateless_tp) = benchmark_lstm_stateless(*n_org, *n_ticks);
            let (_, lstm_stateful_tp) = benchmark_lstm_stateful(*n_org, *n_ticks);

            println!(
                "  {:>22} | {:>9.2e}/s | {:>9.2e}/s | {:>9.2e}/s",
                desc, simple_tp, lstm_stateless_tp, lstm_stateful_tp
            );
        }

        println!();
        println!(
            "  ┌────────────────────────────────────────────────────────────────────────────────┐"
        );
        println!(
            "  │ Performance Summary                                                           │"
        );
        println!(
            "  └────────────────────────────────────────────────────────────────────────────────┘"
        );
        println!();

        // Run at optimal batch size (100K) for summary
        let (_, simple_tp) = benchmark_simple(100_000, 100);
        let (_, lstm_stateless_tp) = benchmark_lstm_stateless(100_000, 100);
        let (_, lstm_stateful_tp) = benchmark_lstm_stateful(100_000, 100);

        println!("  At 100K organisms (optimal batch):");
        println!();
        println!(
            "    WMMA (EPIC 96):      {:>12.3e}/s  (reference)",
            wmma_throughput
        );
        println!(
            "    SimpleBrain (Burn):  {:>12.3e}/s  ({:.1}× vs WMMA)",
            simple_tp,
            wmma_throughput / simple_tp
        );
        println!(
            "    LSTM (stateless):    {:>12.3e}/s  ({:.1}× vs WMMA, {:.1}× vs Simple)",
            lstm_stateless_tp,
            wmma_throughput / lstm_stateless_tp,
            simple_tp / lstm_stateless_tp
        );
        println!(
            "    LSTM (stateful):     {:>12.3e}/s  ({:.1}× vs WMMA, {:.1}× vs Simple)",
            lstm_stateful_tp,
            wmma_throughput / lstm_stateful_tp,
            simple_tp / lstm_stateful_tp
        );

        println!();
        println!(
            "  ┌────────────────────────────────────────────────────────────────────────────────┐"
        );
        println!(
            "  │ Conclusions                                                                   │"
        );
        println!(
            "  └────────────────────────────────────────────────────────────────────────────────┘"
        );
        println!();

        let lstm_vs_simple = simple_tp / lstm_stateful_tp;

        if lstm_vs_simple < 2.0 {
            println!("    LSTM overhead is acceptable (<2× vs Simple)");
            println!("    → Proceed to EPIC 101: Test if LSTM provides evolutionary advantage");
        } else if lstm_vs_simple < 5.0 {
            println!(
                "    LSTM has moderate overhead ({:.1}× vs Simple)",
                lstm_vs_simple
            );
            println!("    → LSTM viable for cognitive tasks, but not default");
        } else {
            println!(
                "    LSTM has high overhead ({:.1}× vs Simple)",
                lstm_vs_simple
            );
            println!("    → Only use LSTM for tasks that absolutely require memory");
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
        println!("EPIC 99: LSTM Benchmark");
        println!();
        println!("This benchmark requires the burn-compute feature.");
        println!(
            "Run with: cargo run --release --example epic99_lstm_benchmark --features burn-compute"
        );
    }
}
