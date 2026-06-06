//! GPU vs CPU Stabilizer Performance Benchmark
//!
//! Compares the performance of GPU-accelerated stabilizer operations
//! against the CPU implementation.
//!
//! Run with: cargo run --release --features cuda --example gpu_stabilizer_bench

use engine::qec::StabilizerTableau;
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(70));
    println!("  GPU vs CPU Stabilizer Performance Benchmark");
    println!("{}", "=".repeat(70));
    println!();

    #[cfg(feature = "cuda")]
    {
        use engine::cuda::CudaRuntime;
        use engine::qec::GpuStabilizerTableau;
        use std::sync::Arc;

        // Try to initialize CUDA
        let rt = match CudaRuntime::new() {
            Ok(rt) => Arc::new(rt),
            Err(e) => {
                println!("CUDA not available: {:?}", e);
                println!("Running CPU-only benchmarks...");
                run_cpu_only_bench();
                return;
            }
        };

        println!("CUDA initialized successfully");
        println!();

        // Benchmark: Column-Major CSR approach (GPU-optimal)
        bench_column_major_csr(&rt);

        // Skip block sparse for now - column-major is the new approach
        // bench_block_sparse(&rt);
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Running CPU-only benchmarks.");
        run_cpu_only_bench();
    }

    println!("{}", "=".repeat(70));
    println!("  Benchmark Complete!");
    println!("{}", "=".repeat(70));
}

#[cfg(feature = "cuda")]
fn bench_creation(rt: &std::sync::Arc<engine::cuda::CudaRuntime>) {
    use engine::qec::GpuStabilizerTableau;

    println!("{}", "-".repeat(70));
    println!("Benchmark 1: Tableau Creation");
    println!("{}", "-".repeat(70));

    let sizes = [64, 128, 256, 512, 1024, 2048];

    println!(
        "{:>10} {:>15} {:>15} {:>12}",
        "Qubits", "CPU (ms)", "GPU (ms)", "Speedup"
    );
    println!("{}", "-".repeat(55));

    for &n in &sizes {
        // CPU creation (average of 5 runs)
        let mut cpu_total = std::time::Duration::ZERO;
        for _ in 0..5 {
            let cpu_start = Instant::now();
            let _cpu_tableau = StabilizerTableau::new(n);
            cpu_total += cpu_start.elapsed();
        }
        let cpu_time = cpu_total / 5;

        // GPU creation (average of 5 runs)
        let mut gpu_total = std::time::Duration::ZERO;
        for _ in 0..5 {
            let gpu_start = Instant::now();
            let _gpu_tableau = GpuStabilizerTableau::new(rt.clone(), n);
            gpu_total += gpu_start.elapsed();
        }
        let gpu_time = gpu_total / 5;

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:>10} {:>15.3} {:>15.3} {:>12.2}x",
            n,
            cpu_time.as_secs_f64() * 1000.0,
            gpu_time.as_secs_f64() * 1000.0,
            speedup
        );
    }
    println!();
}

#[cfg(feature = "cuda")]
fn bench_find_anticommuting(rt: &std::sync::Arc<engine::cuda::CudaRuntime>) {
    use engine::qec::GpuStabilizerTableau;

    println!("{}", "-".repeat(70));
    println!("Benchmark 2: Find Anticommuting Rows (Parallel Scan)");
    println!("{}", "-".repeat(70));

    let sizes = [128, 256, 512, 1024, 2048];
    let iterations = 100;

    println!(
        "{:>10} {:>15} {:>15} {:>12}",
        "Qubits", "CPU (us/op)", "GPU (us/op)", "Speedup"
    );
    println!("{}", "-".repeat(55));

    for &n in &sizes {
        // Create tableaux
        let mut cpu_tableau = StabilizerTableau::new(n);
        let gpu_tableau = match GpuStabilizerTableau::new(rt.clone(), n) {
            Ok(t) => t,
            Err(e) => {
                println!("{:>10} GPU error: {}", n, e);
                continue;
            }
        };

        // Apply some gates to make state non-trivial
        for i in 0..n.min(32) {
            cpu_tableau.hadamard(i);
            if i + 1 < n {
                cpu_tableau.cnot(i, i + 1);
            }
        }

        // CPU benchmark: manually scan for anticommuting rows
        let cpu_start = Instant::now();
        for iter in 0..iterations {
            let qubit = iter % n;
            let mut _count = 0usize;
            // Scan all stabilizers for X on qubit
            for i in 0..n {
                if cpu_tableau.stabilizer(i).get_x(qubit) {
                    _count += 1;
                }
            }
            // Scan all destabilizers
            for i in 0..n {
                if cpu_tableau.destabilizer(i).get_x(qubit) {
                    _count += 1;
                }
            }
        }
        let cpu_time = cpu_start.elapsed();

        // GPU benchmark: use find_anticommuting kernel
        let gpu_start = Instant::now();
        for iter in 0..iterations {
            let qubit = iter % n;
            let _result = gpu_tableau.find_anticommuting(qubit);
        }
        let gpu_time = gpu_start.elapsed();

        let cpu_us = cpu_time.as_secs_f64() * 1_000_000.0 / iterations as f64;
        let gpu_us = gpu_time.as_secs_f64() * 1_000_000.0 / iterations as f64;
        let speedup = cpu_us / gpu_us;

        println!(
            "{:>10} {:>15.2} {:>15.2} {:>12.2}x",
            n, cpu_us, gpu_us, speedup
        );
    }
    println!();
}

#[cfg(feature = "cuda")]
fn bench_batched_measurement(rt: &std::sync::Arc<engine::cuda::CudaRuntime>) {
    use engine::qec::GpuStabilizerTableau;

    println!("{}", "-".repeat(70));
    println!("Benchmark 4: Batched Measurement (GPU vs CPU)");
    println!("{}", "-".repeat(70));

    let n_qubits = 1024;
    let batch_sizes = [1, 10, 50, 100, 200, 500];
    let iterations = 20;

    println!(
        "{:>12} {:>15} {:>15} {:>12}",
        "Batch Size", "CPU (us)", "GPU (us)", "Speedup"
    );
    println!("{}", "-".repeat(57));

    for &batch_size in &batch_sizes {
        // Create fresh tableaux for each test
        let mut cpu_times = Vec::new();
        let mut gpu_times = Vec::new();

        for _ in 0..iterations {
            // CPU: measure batch_size qubits sequentially
            let mut cpu_tableau = StabilizerTableau::new_plus_state(n_qubits);
            let cpu_start = Instant::now();
            for i in 0..batch_size {
                let qubit = i % n_qubits;
                cpu_tableau.measure_z(qubit, i % 2 == 0);
            }
            cpu_times.push(cpu_start.elapsed());

            // GPU: measure batch_size qubits in one kernel
            let mut gpu_tableau = GpuStabilizerTableau::new(rt.clone(), n_qubits).unwrap();
            let qubits: Vec<usize> = (0..batch_size).map(|i| i % n_qubits).collect();
            let random_bits: Vec<bool> = (0..batch_size).map(|i| i % 2 == 0).collect();

            let gpu_start = Instant::now();
            let _ = gpu_tableau.batch_measure(&qubits, &random_bits);
            gpu_times.push(gpu_start.elapsed());
        }

        let cpu_avg_us = cpu_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / iterations as f64
            * 1_000_000.0;
        let gpu_avg_us = gpu_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / iterations as f64
            * 1_000_000.0;
        let speedup = cpu_avg_us / gpu_avg_us;

        println!(
            "{:>12} {:>15.1} {:>15.1} {:>12.2}x",
            batch_size, cpu_avg_us, gpu_avg_us, speedup
        );
    }
    println!();

    // Now test with larger tableau
    println!("Batched measurement scaling with tableau size (batch=100):");
    println!(
        "{:>10} {:>15} {:>15} {:>12} {:>10}",
        "Qubits", "CPU (ms)", "GPU (ms)", "Speedup", "GPU Mem"
    );
    println!("{}", "-".repeat(65));

    let batch_size = 100;
    let tableau_sizes = [1024, 4096, 16384, 65536, 100000];

    for &n in &tableau_sizes {
        let mut cpu_times = Vec::new();
        let mut gpu_times = Vec::new();
        let iters = if n >= 65536 { 5 } else { iterations };

        for _ in 0..iters {
            // CPU
            let mut cpu_tableau = StabilizerTableau::new_plus_state(n);
            let cpu_start = Instant::now();
            for i in 0..batch_size {
                cpu_tableau.measure_z(i % n, i % 2 == 0);
            }
            cpu_times.push(cpu_start.elapsed());

            // GPU
            if let Ok(mut gpu_tableau) = GpuStabilizerTableau::new(rt.clone(), n) {
                let qubits: Vec<usize> = (0..batch_size).map(|i| i % n).collect();
                let random_bits: Vec<bool> = (0..batch_size).map(|i| i % 2 == 0).collect();

                let gpu_start = Instant::now();
                let _ = gpu_tableau.batch_measure(&qubits, &random_bits);
                gpu_times.push(gpu_start.elapsed());
            }
        }

        if gpu_times.is_empty() {
            println!("{:>10} - GPU failed", n);
            continue;
        }

        let cpu_avg_ms =
            cpu_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / iters as f64 * 1000.0;
        let gpu_avg_ms =
            gpu_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / iters as f64 * 1000.0;
        let speedup = cpu_avg_ms / gpu_avg_ms;
        let mem_mb = (n * n / 8 * 2 + n * 2) as f64 / (1024.0 * 1024.0);

        println!(
            "{:>10} {:>15.2} {:>15.2} {:>12.2}x {:>8.1} MB",
            n, cpu_avg_ms, gpu_avg_ms, speedup, mem_mb
        );
    }
    println!();
}

fn bench_measurement() {
    println!("{}", "-".repeat(70));
    println!("Benchmark 3: Measurement Performance (CPU baseline)");
    println!("{}", "-".repeat(70));

    let sizes = [64, 128, 256, 512, 1024];
    let iterations = 1000;

    println!(
        "{:>10} {:>15} {:>15} {:>15}",
        "Qubits", "Random (us)", "Determ (us)", "Meas/sec"
    );
    println!("{}", "-".repeat(58));

    for &n in &sizes {
        // Create GHZ-like state (deterministic measurements)
        let mut tableau_det = StabilizerTableau::new(n);
        tableau_det.hadamard(0);
        for i in 1..n {
            tableau_det.cnot(0, i);
        }

        // Create + state (random measurements)
        let mut tableau_rand = StabilizerTableau::new_plus_state(n);

        // Measure deterministic case
        let det_start = Instant::now();
        for _ in 0..iterations {
            let mut t = tableau_det.clone();
            t.measure_z(0, false);
        }
        let det_time = det_start.elapsed();

        // Measure random case
        let rand_start = Instant::now();
        for iter in 0..iterations {
            let mut t = tableau_rand.clone();
            t.measure_z(iter % n, iter % 2 == 0);
        }
        let rand_time = rand_start.elapsed();

        let det_us = det_time.as_secs_f64() * 1_000_000.0 / iterations as f64;
        let rand_us = rand_time.as_secs_f64() * 1_000_000.0 / iterations as f64;
        let meas_per_sec = iterations as f64 / rand_time.as_secs_f64();

        println!(
            "{:>10} {:>15.2} {:>15.2} {:>15.0}",
            n, rand_us, det_us, meas_per_sec
        );
    }
    println!();
}

#[cfg(feature = "cuda")]
fn bench_scaling(rt: &std::sync::Arc<engine::cuda::CudaRuntime>) {
    use engine::qec::GpuStabilizerTableau;

    println!("{}", "-".repeat(70));
    println!("Benchmark 5: Scaling Analysis (Find Anticommuting)");
    println!("{}", "-".repeat(70));

    let sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536];
    let iterations = 20;

    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>12}",
        "Qubits", "CPU (ms)", "GPU (ms)", "Speedup", "GPU Mem"
    );
    println!("{}", "-".repeat(62));

    for &n in &sizes {
        // Create tableaux
        let cpu_tableau = StabilizerTableau::new(n);
        let gpu_result = GpuStabilizerTableau::new(rt.clone(), n);

        let gpu_tableau = match gpu_result {
            Ok(t) => t,
            Err(e) => {
                println!("{:>10} GPU error: {}", n, e);
                continue;
            }
        };

        // CPU benchmark: scan for anticommuting
        let cpu_start = Instant::now();
        for iter in 0..iterations {
            let qubit = iter % n;
            let mut _count = 0usize;
            for i in 0..n {
                if cpu_tableau.stabilizer(i).get_x(qubit) {
                    _count += 1;
                }
            }
            for i in 0..n {
                if cpu_tableau.destabilizer(i).get_x(qubit) {
                    _count += 1;
                }
            }
        }
        let cpu_time = cpu_start.elapsed();

        // GPU benchmark
        let gpu_start = Instant::now();
        for iter in 0..iterations {
            let qubit = iter % n;
            let _result = gpu_tableau.find_anticommuting(qubit);
        }
        let gpu_time = gpu_start.elapsed();

        let cpu_ms = cpu_time.as_secs_f64() * 1000.0;
        let gpu_ms = gpu_time.as_secs_f64() * 1000.0;
        let speedup = cpu_ms / gpu_ms;
        let mem_mb = gpu_tableau.memory_bytes() as f64 / (1024.0 * 1024.0);

        println!(
            "{:>10} {:>12.2} {:>12.2} {:>12.2}x {:>10.2} MB",
            n, cpu_ms, gpu_ms, speedup, mem_mb
        );
    }
    println!();

    // Memory scaling note
    println!("Memory scaling: O(n^2) bits = n^2 / 8 bytes");
    println!("  1K qubits  -> 0.25 MB");
    println!("  4K qubits  -> 4 MB");
    println!("  10K qubits -> 25 MB");
    println!("  100K qubits -> 2.5 GB");
    println!();
}

#[cfg(feature = "cuda")]
fn bench_gpu_gates(rt: &std::sync::Arc<engine::cuda::CudaRuntime>) {
    use engine::qec::GpuStabilizerTableau;

    println!("{}", "-".repeat(70));
    println!("Benchmark 6: GPU Gate Operations (H, S, CZ)");
    println!("{}", "-".repeat(70));

    let sizes = [1024, 4096, 16384, 65536];
    let iterations = 100;

    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>12}",
        "Qubits", "H (us)", "S (us)", "CZ (us)", "Batch CZ"
    );
    println!("{}", "-".repeat(62));

    for &n in &sizes {
        let mut gpu_tableau = match GpuStabilizerTableau::new(rt.clone(), n) {
            Ok(t) => t,
            Err(e) => {
                println!("{:>10} GPU error: {}", n, e);
                continue;
            }
        };

        // Single Hadamard
        let h_start = Instant::now();
        for i in 0..iterations {
            let _ = gpu_tableau.hadamard(i % n);
        }
        let h_time = h_start.elapsed();

        // Single Phase
        let s_start = Instant::now();
        for i in 0..iterations {
            let _ = gpu_tableau.phase_gate(i % n);
        }
        let s_time = s_start.elapsed();

        // Single CZ
        let cz_start = Instant::now();
        for i in 0..iterations {
            let _ = gpu_tableau.cz(i % n, (i + 1) % n);
        }
        let cz_time = cz_start.elapsed();

        // Batch CZ (100 pairs at once)
        let pairs: Vec<(usize, usize)> = (0..100).map(|i| (i % n, (i + 1) % n)).collect();
        let batch_start = Instant::now();
        for _ in 0..iterations {
            let _ = gpu_tableau.batch_cz(&pairs);
        }
        let batch_time = batch_start.elapsed();

        println!(
            "{:>10} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
            n,
            h_time.as_secs_f64() * 1_000_000.0 / iterations as f64,
            s_time.as_secs_f64() * 1_000_000.0 / iterations as f64,
            cz_time.as_secs_f64() * 1_000_000.0 / iterations as f64,
            batch_time.as_secs_f64() * 1_000_000.0 / iterations as f64,
        );
    }
    println!();

    // Batch operation scaling
    println!("Batch CZ scaling (16K qubits, varying batch sizes):");
    println!(
        "{:>12} {:>15} {:>15}",
        "Batch Size", "Time (us)", "Gates/us"
    );
    println!("{}", "-".repeat(45));

    let n = 16384;
    let mut gpu_tableau = GpuStabilizerTableau::new(rt.clone(), n).unwrap();
    let batch_sizes = [10, 100, 500, 1000, 2000, 5000];

    for &batch in &batch_sizes {
        let pairs: Vec<(usize, usize)> = (0..batch).map(|i| (i % n, (i + 1) % n)).collect();
        let start = Instant::now();
        for _ in 0..50 {
            let _ = gpu_tableau.batch_cz(&pairs);
        }
        let elapsed = start.elapsed();
        let us_per_op = elapsed.as_secs_f64() * 1_000_000.0 / 50.0;
        let gates_per_us = batch as f64 / us_per_op;

        println!("{:>12} {:>15.2} {:>15.2}", batch, us_per_op, gates_per_us);
    }
    println!();
}

#[cfg(feature = "cuda")]
fn bench_network_step(rt: &std::sync::Arc<engine::cuda::CudaRuntime>) {
    use engine::snn::gpu_stabilizer_network::{GpuStabilizerNetwork, StabilizerBackend};
    use engine::snn::stabilizer_network::{StabilizerNetworkConfig, StabilizerNeuronNetwork};

    println!("{}", "-".repeat(70));
    println!("Benchmark 7: Full Network Step (GPU vs CPU)");
    println!("{}", "-".repeat(70));

    // Note: Dense batch_cz has O(n_rows * n_pairs) complexity, slowing GPU path
    // for networks with many synapses. Block sparse approach (Benchmark 8) is preferred.
    let configs = [
        (100, vec![500], 10, "Small (610 neurons)"),
        (100, vec![1000, 500], 10, "Medium (1610 neurons)"),
        (100, vec![2000, 1000], 10, "Large (3110 neurons)"),
    ];

    println!(
        "{:>20} {:>12} {:>12} {:>12}",
        "Config", "CPU (ms)", "GPU (ms)", "Speedup"
    );
    println!("{}", "-".repeat(60));

    for (n_inputs, hidden, n_outputs, name) in configs {
        let config = StabilizerNetworkConfig {
            n_inputs,
            hidden_layers: hidden.clone(),
            n_outputs,
            connectivity: 0.1,
            ..Default::default()
        };

        let inputs: Vec<u8> = (0..n_inputs).map(|i| ((i * 37) % 256) as u8).collect();
        let iterations = 20;

        // CPU network
        let mut cpu_network = StabilizerNeuronNetwork::new(config.clone(), 42);
        let cpu_start = Instant::now();
        for _ in 0..iterations {
            let _ = cpu_network.step(&inputs);
        }
        let cpu_time = cpu_start.elapsed();

        // GPU network
        let gpu_result =
            GpuStabilizerNetwork::with_backend(config.clone(), 42, StabilizerBackend::Gpu);

        let gpu_time = match gpu_result {
            Ok(mut gpu_network) => {
                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = gpu_network.step(&inputs);
                }
                Some(start.elapsed())
            }
            Err(e) => {
                println!("{:>20} GPU error: {}", name, e);
                None
            }
        };

        let cpu_ms = cpu_time.as_secs_f64() * 1000.0 / iterations as f64;

        if let Some(gt) = gpu_time {
            let gpu_ms = gt.as_secs_f64() * 1000.0 / iterations as f64;
            let speedup = cpu_ms / gpu_ms;
            println!(
                "{:>20} {:>12.2} {:>12.2} {:>12.2}x",
                name, cpu_ms, gpu_ms, speedup
            );
        } else {
            println!("{:>20} {:>12.2} {:>12} {:>12}", name, cpu_ms, "N/A", "N/A");
        }
    }
    println!();

    // Scaling test with GPU-only
    println!("GPU network scaling (connectivity=0.05):");
    println!(
        "{:>12} {:>12} {:>12} {:>12}",
        "Neurons", "Step (ms)", "Synapses", "Memory"
    );
    println!("{}", "-".repeat(52));

    let scales = [
        (100, vec![1000], 100),
        (200, vec![2000, 1000], 200),
        (500, vec![5000, 2000], 500),
    ];

    for (n_in, hidden, n_out) in scales {
        let config = StabilizerNetworkConfig {
            n_inputs: n_in,
            hidden_layers: hidden.clone(),
            n_outputs: n_out,
            connectivity: 0.05,
            ..Default::default()
        };

        let total_neurons: usize = n_in + hidden.iter().sum::<usize>() + n_out;
        let inputs: Vec<u8> = (0..n_in).map(|i| ((i * 37) % 256) as u8).collect();

        let gpu_result =
            GpuStabilizerNetwork::with_backend(config.clone(), 42, StabilizerBackend::Gpu);

        match gpu_result {
            Ok(mut gpu_network) => {
                let iterations = 10;
                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = gpu_network.step(&inputs);
                }
                let elapsed = start.elapsed();
                let ms_per_step = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
                let n_synapses = gpu_network.num_synapses();
                let mem_mb = (total_neurons * total_neurons / 8 * 2) as f64 / (1024.0 * 1024.0);

                println!(
                    "{:>12} {:>12.2} {:>12} {:>10.1} MB",
                    total_neurons, ms_per_step, n_synapses, mem_mb
                );
            }
            Err(e) => {
                println!("{:>12} GPU error: {}", total_neurons, e);
            }
        }
    }
    println!();
}

#[cfg(feature = "cuda")]
fn bench_block_sparse(rt: &std::sync::Arc<engine::cuda::CudaRuntime>) {
    use engine::qec::GpuStabilizerTableau;
    use engine::snn::block_sparse_synapses::{BlockSparseSynapseMap, SparsityPattern};

    println!("{}", "-".repeat(70));
    println!("Benchmark 8: Block Sparse CZ Performance");
    println!("{}", "-".repeat(70));

    // Test scaling with block sparse approach
    // Reduced sizes to avoid memory issues during benchmarking
    // Scale up to larger networks
    let configs = [
        (1000, vec![(0, 100), (100, 600), (600, 1000)], "1K neurons"),
        (
            5000,
            vec![(0, 200), (200, 2500), (2500, 5000)],
            "5K neurons",
        ),
        (
            10000,
            vec![(0, 200), (200, 5000), (5000, 10000)],
            "10K neurons",
        ),
        (
            20000,
            vec![(0, 500), (500, 10000), (10000, 20000)],
            "20K neurons",
        ),
        (
            50000,
            vec![(0, 1000), (1000, 25000), (25000, 50000)],
            "50K neurons",
        ),
    ];

    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "Config", "Time (ms)", "Block Pairs", "Synapses", "Syn/ms", "Memory"
    );
    println!("{}", "-".repeat(75));

    for (n_neurons, layers, name) in configs {
        let connectivity = 0.05; // Lower connectivity for larger networks

        // Create block sparse synapse map with 4:8 pattern
        let sparse_map = BlockSparseSynapseMap::from_layers(
            n_neurons,
            &layers,
            connectivity,
            SparsityPattern::FourEight,
            42,
        );

        let stats = sparse_map.stats();
        let gpu_data = sparse_map.gpu_data();

        // Create GPU tableau
        let mut gpu_tableau = match GpuStabilizerTableau::new(rt.clone(), n_neurons) {
            Ok(t) => t,
            Err(e) => {
                println!("{:>12} GPU error: {}", name, e);
                continue;
            }
        };

        // Warm up
        let _ = gpu_tableau.block_sparse_cz(
            &gpu_data.src_blocks,
            &gpu_data.dst_blocks,
            &gpu_data.synapse_masks,
            gpu_data.block_size as usize,
        );

        // Benchmark block sparse approach
        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = gpu_tableau.block_sparse_cz(
                &gpu_data.src_blocks,
                &gpu_data.dst_blocks,
                &gpu_data.synapse_masks,
                gpu_data.block_size as usize,
            );
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let syn_per_ms = stats.n_synapses as f64 / ms;

        println!(
            "{:>12} {:>12.3} {:>12} {:>12} {:>12.0} {:>8.1} KB",
            name,
            ms,
            stats.n_block_pairs,
            stats.n_synapses,
            syn_per_ms,
            stats.memory_bytes as f64 / 1024.0
        );
    }
    println!();

    // Sparsity pattern comparison (10K neurons for production-scale testing)
    println!("Sparsity pattern comparison (10K neurons, 5% connectivity):");
    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>12}",
        "Pattern", "Time (ms)", "Synapses", "Density", "Syn/ms"
    );
    println!("{}", "-".repeat(62));

    let n = 10000;
    let layers = vec![(0, 200), (200, 5000), (5000, 10000)];
    let patterns = [
        (SparsityPattern::TwoFour, "2:4"),
        (SparsityPattern::FourEight, "4:8"),
        (SparsityPattern::Custom(128), "Custom 50%"),
        (SparsityPattern::Custom(64), "Custom 25%"),
    ];

    for (pattern, name) in patterns {
        let sparse_map = BlockSparseSynapseMap::from_layers(n, &layers, 0.05, pattern, 42);
        let gpu_data = sparse_map.gpu_data();
        let stats = sparse_map.stats();

        let mut gpu_tableau = GpuStabilizerTableau::new(rt.clone(), n).unwrap();

        // Warm up
        let _ = gpu_tableau.block_sparse_cz(
            &gpu_data.src_blocks,
            &gpu_data.dst_blocks,
            &gpu_data.synapse_masks,
            gpu_data.block_size as usize,
        );

        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = gpu_tableau.block_sparse_cz(
                &gpu_data.src_blocks,
                &gpu_data.dst_blocks,
                &gpu_data.synapse_masks,
                gpu_data.block_size as usize,
            );
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let syn_per_ms = stats.n_synapses as f64 / ms;

        println!(
            "{:>12} {:>12.3} {:>12} {:>12.2}% {:>12.0}",
            name,
            ms,
            stats.n_synapses,
            stats.actual_density * 100.0,
            syn_per_ms
        );
    }
    println!();
}

// =============================================================================
// COLUMN-MAJOR CSR BENCHMARK
// GPU-optimal memory layout with CSR synapse representation
// =============================================================================
#[cfg(feature = "cuda")]
fn bench_column_major_csr(rt: &std::sync::Arc<engine::cuda::CudaRuntime>) {
    use engine::snn::column_major_stabilizer::{GpuColumnMajorTableau, SynapseCSR};

    println!("{}", "-".repeat(70));
    println!("Benchmark: Column-Major CSR (GPU-Optimal Layout)");
    println!("{}", "-".repeat(70));
    println!();
    println!("IMPORTANT: Using REALISTIC synapse counts (~1000 per layer pair)");
    println!("  - This matches the CPU version's adaptive connectivity");
    println!("  - 5% connectivity on 50K neurons = 31M synapses (UNREALISTIC)");
    println!("  - Real neural networks use sparse connectivity");
    println!();

    // REALISTIC configurations with capped synapse counts
    // ~1000 synapses per layer pair, matching CPU adaptive behavior
    let configs = [
        (1_000, 1000, "1K neurons"),
        (5_000, 2000, "5K neurons"),
        (10_000, 3000, "10K neurons"),
        (50_000, 5000, "50K neurons"),
        (100_000, 6000, "100K neurons"),
        (200_000, 8000, "200K neurons"),
        (500_000, 10000, "500K neurons"),
    ];

    println!(
        "{:>14} {:>10} {:>10} {:>12} {:>10}",
        "Config", "Synapses", "Time (ms)", "Syn/ms", "Memory"
    );
    println!("{}", "-".repeat(60));

    for (n_qubits, n_synapses, name) in configs {
        // Generate random synapse pairs (simulating sparse connectivity)
        use engine::qec::noise::SimpleRng;
        let mut rng = SimpleRng::new(42);
        let mut pairs = Vec::with_capacity(n_synapses);
        for _ in 0..n_synapses {
            let src = (rng.next_u64() as usize) % n_qubits;
            let dst = (rng.next_u64() as usize) % n_qubits;
            if src != dst {
                pairs.push((src, dst));
            }
        }

        let csr = SynapseCSR::from_pairs(n_qubits, &pairs);
        let stats = csr.stats();

        // Create column-major tableau
        let mut tableau = match GpuColumnMajorTableau::new(rt.clone(), n_qubits, &csr) {
            Ok(t) => t,
            Err(e) => {
                println!("{:>14} GPU error: {}", name, e);
                continue;
            }
        };

        // Warm up
        for _ in 0..3 {
            let _ = tableau.apply_all_cz();
        }
        let _ = tableau.sync();

        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tableau.apply_all_cz();
        }
        let _ = tableau.sync();
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let syn_per_ms = stats.n_synapses as f64 / ms;
        let mem_mb = tableau.memory_bytes() as f64 / (1024.0 * 1024.0);

        println!(
            "{:>14} {:>10} {:>10.3} {:>12.0} {:>8.1} MB",
            name, stats.n_synapses, ms, syn_per_ms, mem_mb
        );
    }

    println!();

    // Now compare with dense connectivity to show the scaling difference
    println!("COMPARISON: Dense vs Sparse Connectivity (10K neurons)");
    println!("{}", "-".repeat(60));
    println!(
        "{:>20} {:>12} {:>12} {:>12}",
        "Connectivity", "Synapses", "Time (ms)", "Syn/ms"
    );
    println!("{}", "-".repeat(60));

    let n = 10_000;
    let conn_tests = [
        ("Sparse (1K)", 1000),
        ("Sparse (5K)", 5000),
        ("Sparse (10K)", 10_000),
        ("Medium (50K)", 50_000),
        ("Dense (1%)", (n * n / 100 / 2).min(500_000)), // Cap at 500K for sanity
    ];

    for (label, n_syn) in conn_tests {
        use engine::qec::noise::SimpleRng;
        let mut rng = SimpleRng::new(42);
        let mut pairs = Vec::with_capacity(n_syn);
        for _ in 0..n_syn {
            let src = (rng.next_u64() as usize) % n;
            let dst = (rng.next_u64() as usize) % n;
            if src != dst {
                pairs.push((src, dst));
            }
        }

        let csr = SynapseCSR::from_pairs(n, &pairs);
        let stats = csr.stats();

        let mut tableau = match GpuColumnMajorTableau::new(rt.clone(), n, &csr) {
            Ok(t) => t,
            Err(e) => {
                println!("{:>20} Error: {}", label, e);
                continue;
            }
        };

        for _ in 0..3 {
            let _ = tableau.apply_all_cz();
        }
        let _ = tableau.sync();

        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tableau.apply_all_cz();
        }
        let _ = tableau.sync();
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let syn_per_ms = stats.n_synapses as f64 / ms;

        println!(
            "{:>20} {:>12} {:>12.3} {:>12.0}",
            label, stats.n_synapses, ms, syn_per_ms
        );
    }

    println!();

    // Test UNIFIED MEMORY for large sizes
    println!("UNIFIED MEMORY MODE (can use system RAM):");
    println!("{}", "-".repeat(60));
    println!(
        "{:>14} {:>10} {:>10} {:>12} {:>10}",
        "Config", "Synapses", "Time (ms)", "Syn/ms", "Memory"
    );
    println!("{}", "-".repeat(60));

    // Test larger configurations with unified memory
    let unified_configs = [
        (100_000, 6000, "100K neurons"),
        (200_000, 8000, "200K neurons"),
        (500_000, 10000, "500K neurons"),
        (1_000_000, 15000, "1M neurons"),
    ];

    for (n_qubits, n_synapses, name) in unified_configs {
        use engine::qec::noise::SimpleRng;
        let mut rng = SimpleRng::new(42);
        let mut pairs = Vec::with_capacity(n_synapses);
        for _ in 0..n_synapses {
            let src = (rng.next_u64() as usize) % n_qubits;
            let dst = (rng.next_u64() as usize) % n_qubits;
            if src != dst {
                pairs.push((src, dst));
            }
        }

        let csr = SynapseCSR::from_pairs(n_qubits, &pairs);
        let stats = csr.stats();

        // Use unified memory mode
        let mut tableau = match GpuColumnMajorTableau::new_unified(rt.clone(), n_qubits, &csr) {
            Ok(t) => t,
            Err(e) => {
                println!("{:>14} Error: {}", name, e);
                continue;
            }
        };

        // Warm up (important for unified memory - pages data to GPU)
        for _ in 0..5 {
            let _ = tableau.apply_all_cz();
        }
        let _ = tableau.sync();

        // Benchmark
        let iterations = 20; // Fewer iterations for large sizes
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tableau.apply_all_cz();
        }
        let _ = tableau.sync();
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let syn_per_ms = stats.n_synapses as f64 / ms;
        let mem_mb = tableau.memory_bytes() as f64 / (1024.0 * 1024.0);

        println!(
            "{:>14} {:>10} {:>10.1} {:>12.0} {:>8.1} MB",
            name, stats.n_synapses, ms, syn_per_ms, mem_mb
        );
    }

    println!();
}

fn run_cpu_only_bench() {
    println!("{}", "-".repeat(70));
    println!("CPU-Only Stabilizer Benchmark");
    println!("{}", "-".repeat(70));

    let sizes = [64, 128, 256, 512, 1024];
    let iterations = 100;

    println!(
        "{:>10} {:>15} {:>15} {:>15}",
        "Qubits", "Create (ms)", "Gates (us)", "Measure (us)"
    );
    println!("{}", "-".repeat(58));

    for &n in &sizes {
        // Creation
        let create_start = Instant::now();
        let mut tableau = StabilizerTableau::new(n);
        let create_time = create_start.elapsed();

        // Apply gates
        let gate_start = Instant::now();
        for i in 0..n.min(64) {
            tableau.hadamard(i);
            if i + 1 < n {
                tableau.cnot(i, i + 1);
            }
        }
        let gate_time = gate_start.elapsed();

        // Measurement
        let meas_start = Instant::now();
        for i in 0..iterations {
            let qubit = i % n;
            let mut t = tableau.clone();
            t.measure_z(qubit, i % 2 == 0);
        }
        let meas_time = meas_start.elapsed();

        let meas_us = meas_time.as_secs_f64() * 1_000_000.0 / iterations as f64;

        println!(
            "{:>10} {:>15.3} {:>15.2} {:>15.2}",
            n,
            create_time.as_secs_f64() * 1000.0,
            gate_time.as_secs_f64() * 1_000_000.0,
            meas_us
        );
    }
    println!();
}
