// EPIC 115: Benchmark GPU vs CPU FP32↔FP16 conversion
//
// Run with: cargo run --example benchmark_fp16_conversion --features cuda --release

#[cfg(feature = "cuda")]
fn main() {
    use logic_fabric_core::cuda::{CudaRuntime, GpuQState, GpuQStateF16, is_cuda_available};
    use logic_fabric_core::quantum::QState;
    use std::time::Instant;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 115: GPU vs CPU FP32↔FP16 Conversion Benchmark           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    if !is_cuda_available() {
        println!("CUDA not available!");
        return;
    }

    let rt = CudaRuntime::new().expect("CUDA runtime");

    // Test different sizes
    let sizes = vec![
        (8, "256 amps (256 B)"),
        (12, "4K amps (4 KB)"),
        (16, "64K amps (64 KB)"),
        (20, "1M amps (1 MB)"),
        (24, "16M amps (16 MB)"),
    ];

    println!("┌──────────┬────────────────────┬────────────────┬────────────────┬──────────┐");
    println!("│  Qubits  │     Description    │  CPU Time (ms) │  GPU Time (ms) │  Speedup │");
    println!("├──────────┼────────────────────┼────────────────┼────────────────┼──────────┤");

    for (n_qubits, desc) in sizes {
        // Create CPU state
        let mut qstate = QState::new_zero_multitile(n_qubits, 1);
        for i in 0..qstate.len.min(1000) {
            qstate.real.as_mut_slice()[i] = (i as f32) * 0.001;
            qstate.imag.as_mut_slice()[i] = (i as f32) * -0.0005;
        }

        // Upload to GPU as FP32
        let gpu_state = GpuQState::from_qstate(&rt, &qstate).expect("FP32 upload");

        // Warmup
        let _ = GpuQStateF16::from_fp32_gpu(&rt, &gpu_state);

        // Benchmark CPU conversion (if we can call it directly)
        let iterations = if n_qubits <= 16 { 10 } else { 3 };

        // GPU conversion timing
        let start_gpu = Instant::now();
        for _ in 0..iterations {
            let fp16 = GpuQStateF16::from_fp32_gpu(&rt, &gpu_state).expect("GPU conv");
            let _ = fp16.to_fp32_gpu(&rt).expect("GPU conv back");
        }
        let gpu_time = start_gpu.elapsed().as_secs_f64() * 1000.0 / (iterations as f64 * 2.0);

        // CPU conversion timing (force CPU path)
        let start_cpu = Instant::now();
        for _ in 0..iterations {
            let fp16 = GpuQStateF16::from_fp32_cpu(&rt, &gpu_state).expect("CPU conv");
            let _ = fp16.to_fp32_cpu(&rt).expect("CPU conv back");
        }
        let cpu_time = start_cpu.elapsed().as_secs_f64() * 1000.0 / (iterations as f64 * 2.0);

        let speedup = cpu_time / gpu_time;

        println!(
            "│    {:2}    │ {:18} │ {:14.3} │ {:14.3} │ {:7.1}x │",
            n_qubits, desc, cpu_time, gpu_time, speedup
        );
    }

    println!("└──────────┴────────────────────┴────────────────┴────────────────┴──────────┘");
    println!("\nNote: Times are per conversion (FP32→FP16 or FP16→FP32)");
    println!("GPU conversion eliminates PCIe round-trip (~32 GB/s → ~1.8 TB/s bandwidth)");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the 'cuda' feature.");
    println!("Run with: cargo run --example benchmark_fp16_conversion --features cuda --release");
}
