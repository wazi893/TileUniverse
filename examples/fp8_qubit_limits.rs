// EPIC 115.5: FP8 Qubit Limits Calculator
//
// Calculate the maximum qubit count achievable on RTX 5090 with FP8.
//
// Run with: cargo run --example fp8_qubit_limits --features cuda --release

#[cfg(feature = "cuda")]
fn main() {
    use logic_fabric_core::cuda::{CudaRuntime, is_cuda_available};

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 115.5: FP8 Quantum Simulation - Maximum Qubit Limits     ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    if !is_cuda_available() {
        println!("CUDA not available!");
        return;
    }

    let rt = CudaRuntime::new().expect("CUDA runtime");

    // Get GPU memory info
    let (free_mem, total_mem) = rt.get_memory_info().expect("Memory info");

    println!("GPU Memory:");
    println!(
        "  Total: {:.2} GB",
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  Free:  {:.2} GB",
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!();

    println!("┌─────────┬───────────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ Qubits  │    Amplitudes     │  FP8 (2B)   │  FP16 (4B)  │  FP32 (8B)  │");
    println!("├─────────┼───────────────────┼─────────────┼─────────────┼─────────────┤");

    for n in 20..=40 {
        let amplitudes: u128 = 1u128 << n;
        let fp8_bytes = amplitudes * 2; // 1 byte real + 1 byte imag
        let fp16_bytes = amplitudes * 4; // 2 bytes real + 2 bytes imag
        let fp32_bytes = amplitudes * 8; // 4 bytes real + 4 bytes imag

        let fp8_gb = fp8_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let fp16_gb = fp16_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let fp32_gb = fp32_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        let fp8_fit = if fp8_bytes <= free_mem as u128 {
            "✓"
        } else {
            "✗"
        };
        let fp16_fit = if fp16_bytes <= free_mem as u128 {
            "✓"
        } else {
            "✗"
        };
        let fp32_fit = if fp32_bytes <= free_mem as u128 {
            "✓"
        } else {
            "✗"
        };

        let amp_str = if amplitudes >= 1_000_000_000_000 {
            format!("{:.1}T", amplitudes as f64 / 1e12)
        } else if amplitudes >= 1_000_000_000 {
            format!("{:.1}B", amplitudes as f64 / 1e9)
        } else if amplitudes >= 1_000_000 {
            format!("{:.1}M", amplitudes as f64 / 1e6)
        } else {
            format!("{:.1}K", amplitudes as f64 / 1e3)
        };

        println!(
            "│   {:2}    │ {:>17} │ {:>7.1} GB {} │ {:>7.1} GB {} │ {:>7.1} GB {} │",
            n, amp_str, fp8_gb, fp8_fit, fp16_gb, fp16_fit, fp32_gb, fp32_fit
        );
    }

    println!("└─────────┴───────────────────┴─────────────┴─────────────┴─────────────┘");

    // Calculate exact limits
    let fp8_max_amps = free_mem / 2; // 2 bytes per complex
    let fp16_max_amps = free_mem / 4;
    let fp32_max_amps = free_mem / 8;

    let fp8_max_qubits = (fp8_max_amps as f64).log2().floor() as u32;
    let fp16_max_qubits = (fp16_max_amps as f64).log2().floor() as u32;
    let fp32_max_qubits = (fp32_max_amps as f64).log2().floor() as u32;

    println!("\n📊 Maximum Qubit Counts (with current free memory):");
    println!("   FP8:  {} qubits (2 bytes/amplitude)", fp8_max_qubits);
    println!("   FP16: {} qubits (4 bytes/amplitude)", fp16_max_qubits);
    println!("   FP32: {} qubits (8 bytes/amplitude)", fp32_max_qubits);

    // With full memory
    let fp8_max_full = (total_mem as f64 / 2.0).log2().floor() as u32;
    let fp16_max_full = (total_mem as f64 / 4.0).log2().floor() as u32;
    let fp32_max_full = (total_mem as f64 / 8.0).log2().floor() as u32;

    println!(
        "\n📊 Maximum Qubit Counts (with full {} GB):",
        total_mem / (1024 * 1024 * 1024)
    );
    println!("   FP8:  {} qubits", fp8_max_full);
    println!("   FP16: {} qubits", fp16_max_full);
    println!("   FP32: {} qubits", fp32_max_full);

    // Practical limits (leaving ~20% for gates and overhead)
    let practical_mem = (total_mem as f64 * 0.8) as usize;
    let fp8_practical = (practical_mem as f64 / 2.0).log2().floor() as u32;
    let fp16_practical = (practical_mem as f64 / 4.0).log2().floor() as u32;

    println!("\n📊 Practical Limits (80% memory, 20% for overhead):");
    println!("   FP8:  {} qubits", fp8_practical);
    println!("   FP16: {} qubits", fp16_practical);

    println!("\n🎯 FP8 Advantage:");
    println!(
        "   FP8 enables {} extra qubits vs FP16",
        fp8_max_full - fp16_max_full
    );
    println!(
        "   FP8 enables {} extra qubits vs FP32",
        fp8_max_full - fp32_max_full
    );
    println!("   Each extra qubit = 2x more quantum states!");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires 'cuda' feature.");
    println!("Run with: cargo run --example fp8_qubit_limits --features cuda --release");
}
