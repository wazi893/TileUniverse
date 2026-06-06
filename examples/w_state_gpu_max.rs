//! GPU Max Qubit Test - Push to 31 qubits (32GB VRAM)
use engine::tile8::quantum_router_f64_gpu::QuantumGridF64Gpu;
use logic_fabric_core::cuda::CudaRuntime;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           GPU 31-QUBIT TEST (32GB VRAM)                          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let rt = CudaRuntime::new().expect("CUDA init failed");
    if let Ok(name) = rt.device_name() {
        println!("GPU: {}\n", name);
    }

    let n = 31;
    let num_amps: u64 = 1 << n;
    let vram_gb = (num_amps * 16) as f64 / 1024.0 / 1024.0 / 1024.0;

    println!("Attempting {} qubits:", n);
    println!("  Amplitudes: {} (2^{})", num_amps, n);
    println!("  VRAM required: {:.1} GB\n", vram_gb);

    println!("Allocating GPU memory...");
    let start = Instant::now();

    match QuantumGridF64Gpu::new(&rt, n) {
        Ok(mut grid) => {
            println!(
                "Allocated in {:.1} ms",
                start.elapsed().as_secs_f64() * 1000.0
            );

            println!("\nCreating GHZ state...");
            let start = Instant::now();
            let _ = grid.create_ghz_state_fused();
            let create_ms = start.elapsed().as_secs_f64() * 1000.0;
            println!("Created in {:.1} ms", create_ms);

            println!("\nVerifying...");
            let start = Instant::now();
            let verification = grid.verify_ghz_fidelity().unwrap();
            let verify_ms = start.elapsed().as_secs_f64() * 1000.0;

            println!("\n╔═════════════════════════════════════════════════════════════════╗");
            println!("║  SUCCESS: 31-QUBIT STATE ON GPU                                 ║");
            println!("╠═════════════════════════════════════════════════════════════════╣");
            println!("║  Qubits:     31                                                 ║");
            println!("║  Amplitudes: 2,147,483,648                                      ║");
            println!("║  VRAM:       32 GB                                              ║");
            println!(
                "║  Create:     {:.1} ms                                           ║",
                create_ms
            );
            println!(
                "║  Verify:     {:.1} ms                                           ║",
                verify_ms
            );
            println!(
                "║  Fidelity:   {:.10}                                      ║",
                verification.fidelity
            );
            println!("╚═════════════════════════════════════════════════════════════════╝");
        }
        Err(e) => {
            println!("\n╔═════════════════════════════════════════════════════════════════╗");
            println!("║  FAILED: Out of VRAM                                            ║");
            println!("╚═════════════════════════════════════════════════════════════════╝");
            println!("Error: {:?}", e);
        }
    }
}
