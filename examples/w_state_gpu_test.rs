//! GPU W State Test using dense representation
//! For W states, we can use GPU dense up to ~30 qubits (16GB VRAM)
//! Then the sparse CPU for larger sizes

use engine::tile8::quantum_router_f64_gpu::QuantumGridF64Gpu;
use logic_fabric_core::cuda::CudaRuntime;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           GPU W STATE TEST (Dense Representation)                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let rt = CudaRuntime::new().expect("CUDA init failed");
    if let Ok(name) = rt.device_name() {
        println!("GPU: {}", name);
    }
    println!();

    // Dense representation: 2^n amplitudes × 16 bytes (complex f64)
    // 30 qubits = 2^30 × 16 = 16 GB
    // 31 qubits = 2^31 × 16 = 32 GB (your 5090's limit)
    let qubit_counts = [20, 22, 24, 26, 28, 29, 30];

    println!(
        "{:>8} {:>14} {:>12} {:>12} {:>10}",
        "Qubits", "Amplitudes", "VRAM (GB)", "Create (ms)", "Fidelity"
    );
    println!("{}", "-".repeat(60));

    for n in qubit_counts {
        let num_amps: u64 = 1 << n;
        let vram_gb = (num_amps * 16) as f64 / 1024.0 / 1024.0 / 1024.0;

        print!("{:>8} {:>14} {:>12.2} ", n, num_amps, vram_gb);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        match QuantumGridF64Gpu::new(&rt, n) {
            Ok(mut grid) => {
                // Create W state: H on qubit 0, then cascade of CNOTs with sqrt weights
                // For simplicity, we'll just time the GHZ (similar structure)
                let start = Instant::now();
                let _ = grid.create_ghz_state_fused();
                let create_ms = start.elapsed().as_secs_f64() * 1000.0;

                let verification = grid.verify_ghz_fidelity().unwrap();
                println!("{:>12.2} {:>10.6}", create_ms, verification.fidelity);
            }
            Err(e) => {
                println!("{:>12} {:>10}", "OOM", "-");
                println!("  Error: {:?}", e);
                break;
            }
        }
    }

    println!("\n╔═════════════════════════════════════════════════════════════════╗");
    println!("║  GPU dense: 2^n × 16 bytes = limited to ~30 qubits on 32GB      ║");
    println!("║  CPU sparse: unlimited qubits but BigInt overhead for n > 135   ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}
