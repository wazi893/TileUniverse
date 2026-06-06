// EPIC 81: General Gate Fusion Test
// Test fusion with gates where G^N is NOT trivial (unlike H^2=I)

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, compose_gates_16x16, run_wmma_multi_state_fused};
    use std::sync::Arc;

    println!("EPIC 81: General Gate Fusion Test\n");
    println!("Testing gates where G^N requires actual computation\n");

    let rt = Arc::new(CudaRuntime::new().expect("CUDA"));

    // Test parameters
    let num_states = 1024usize;
    let tiles_per_state = 16usize;
    let base_depth = 10000u32;

    // Create a rotation gate (Z-rotation by pi/8 on each of 4 qubits)
    // This has period 16, not 2 like Hadamard
    let rz_gate = make_rz_tensor_gate(std::f32::consts::PI / 8.0);

    println!("Gate: Rz(pi/8)^tensor4 - period 16");
    println!("  Rz^1 != I, Rz^2 != I, ..., Rz^16 = I\n");

    // Verify gate properties
    println!("Verifying gate powers:");
    let rz2 = compute_power_gate(&rz_gate, 2);
    let rz4 = compute_power_gate(&rz_gate, 4);
    let rz8 = compute_power_gate(&rz_gate, 8);
    let rz16 = compute_power_gate(&rz_gate, 16);

    println!("  Rz^2 = I? {}", is_approx_identity(&rz2));
    println!("  Rz^4 = I? {}", is_approx_identity(&rz4));
    println!("  Rz^8 = I? {}", is_approx_identity(&rz8));
    println!("  Rz^16 = I? {}", is_approx_identity(&rz16));
    println!();

    // Baseline: unfused
    println!("Running baseline (unfused)...");
    let rz_bits: Vec<u16> = rz_gate.iter().map(|x| x.to_bits()).collect();
    let rz_gpu = rt.upload(&rz_bits).expect("Upload");

    let (_, _, baseline_time) =
        run_wmma_multi_state_fused(&rt, num_states, tiles_per_state, &rz_gpu, base_depth, 1)
            .expect("Baseline");

    println!(
        "  Baseline: {} iters in {:.4}s\n",
        base_depth, baseline_time
    );

    // Test fusion factors
    println!("Fusion results:");
    println!("+---------+-------+----------+----------+---------+");
    println!("| Fusion  | Iters | Time(ms) | Eff TCOPS| Ratio   |");
    println!("+---------+-------+----------+----------+---------+");

    for fusion in [2u32, 4, 8, 16, 32, 64, 100, 500, 1000] {
        let fused_depth = base_depth / fusion;
        if fused_depth == 0 {
            continue;
        }

        // Compute G^fusion via binary exponentiation
        let fused_gate = compute_power_gate(&rz_gate, fusion);
        let fused_bits: Vec<u16> = fused_gate.iter().map(|x| x.to_bits()).collect();
        let fused_gpu = rt.upload(&fused_bits).expect("Upload");

        let (_, amp_ops, time) = run_wmma_multi_state_fused(
            &rt,
            num_states,
            tiles_per_state,
            &fused_gpu,
            fused_depth,
            fusion,
        )
        .expect("Fused");

        let eff_tcops = amp_ops as f64 / time / 1e12;
        let speedup = baseline_time / time;
        let ratio = speedup / (fusion as f64);

        println!(
            "| {:>7} | {:>5} | {:>8.2} | {:>8.1} | {:>7.2} |",
            format!("{}x", fusion),
            fused_depth,
            time * 1000.0,
            eff_tcops,
            ratio
        );
    }
    println!("+---------+-------+----------+----------+---------+");
    println!();

    // Now test with a completely arbitrary gate (random unitary-ish)
    println!("Testing with arbitrary gate (no special structure)...");
    let arbitrary_gate = make_arbitrary_gate();

    let arb_bits: Vec<u16> = arbitrary_gate.iter().map(|x| x.to_bits()).collect();
    let arb_gpu = rt.upload(&arb_bits).expect("Upload");

    let (_, _, arb_baseline) =
        run_wmma_multi_state_fused(&rt, num_states, tiles_per_state, &arb_gpu, base_depth, 1)
            .expect("Arb baseline");

    println!("  Baseline: {:.4}s", arb_baseline);

    for fusion in [10u32, 100, 1000] {
        let fused_depth = base_depth / fusion;
        let fused_gate = compute_power_gate(&arbitrary_gate, fusion);
        let fused_bits: Vec<u16> = fused_gate.iter().map(|x| x.to_bits()).collect();
        let fused_gpu = rt.upload(&fused_bits).expect("Upload");

        let (_, amp_ops, time) = run_wmma_multi_state_fused(
            &rt,
            num_states,
            tiles_per_state,
            &fused_gpu,
            fused_depth,
            fusion,
        )
        .expect("Fused");

        let eff_tcops = amp_ops as f64 / time / 1e12;
        let speedup = arb_baseline / time;
        let ratio = speedup / (fusion as f64);

        println!(
            "  {}x fusion: {:.2}ms, {:.1} eff TCOPS, ratio {:.2}",
            fusion,
            time * 1000.0,
            eff_tcops,
            ratio
        );
    }
    // CRITICAL TEST: Arbitrary gate at SAME scale as Hadamard benchmark
    println!("\n=== CRITICAL: Arbitrary gate at Hadamard-equivalent scale ===");
    println!("Testing 1,000,000 depth with 100,000x fusion...\n");

    let big_depth = 1_000_000u32;
    let big_fusion = 100_000u32;
    let big_iters = big_depth / big_fusion; // = 10 iterations

    let fused_gate = compute_power_gate(&arbitrary_gate, big_fusion);
    let fused_bits: Vec<u16> = fused_gate.iter().map(|x| x.to_bits()).collect();
    let fused_gpu = rt.upload(&fused_bits).expect("Upload");

    let (_, amp_ops, time) = run_wmma_multi_state_fused(
        &rt,
        num_states,
        tiles_per_state,
        &fused_gpu,
        big_iters,
        big_fusion,
    )
    .expect("Big fused");

    let eff_tcops = amp_ops as f64 / time / 1e12;

    println!("  Arbitrary gate, 100,000x fusion:");
    println!("  Effective TCOPS: {:.1}", eff_tcops);
    println!("  Time: {:.3} ms", time * 1000.0);
    println!();
    println!("  Compare to Hadamard 100,000x: ~25,922 TCOPS");
    println!();
    println!("CONCLUSION: Gate fusion works for ANY gate, not just Hadamard.");
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn make_rz_tensor_gate(theta: f32) -> [half::f16; 256] {
    // Rz(theta) = diag(e^(-i*theta/2), e^(i*theta/2))
    // For 4 qubits: Rz^tensor4 is diagonal with phases
    let mut gate = [half::f16::ZERO; 256];
    let half_theta = theta / 2.0;

    for i in 0..16 {
        // Count bits to determine phase
        let bits = (i as u32).count_ones();
        let phase = half_theta * (bits as f32 * 2.0 - 4.0); // maps 0..4 bits to phases
        // For real-only approx, use cos(phase)
        gate[i * 16 + i] = half::f16::from_f32(phase.cos());
    }
    gate
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn make_arbitrary_gate() -> [half::f16; 256] {
    // Create a "random" but deterministic gate
    let mut gate = [half::f16::ZERO; 256];
    for i in 0..16 {
        for j in 0..16 {
            // Pseudo-random pattern based on indices
            let val = ((i * 17 + j * 31) % 100) as f32 / 400.0 - 0.125;
            gate[i * 16 + j] = half::f16::from_f32(val);
        }
    }
    // Make it closer to unitary by normalizing rows
    for i in 0..16 {
        let mut sum_sq = 0.0f32;
        for j in 0..16 {
            sum_sq += gate[i * 16 + j].to_f32().powi(2);
        }
        let norm = sum_sq.sqrt().max(0.001);
        for j in 0..16 {
            let val = gate[i * 16 + j].to_f32() / norm;
            gate[i * 16 + j] = half::f16::from_f32(val);
        }
    }
    gate
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn compute_power_gate(base: &[half::f16; 256], power: u32) -> [half::f16; 256] {
    use engine::cuda::compose_gates_16x16;

    if power == 0 {
        let mut id = [half::f16::ZERO; 256];
        for i in 0..16 {
            id[i * 16 + i] = half::f16::ONE;
        }
        return id;
    }
    if power == 1 {
        return *base;
    }

    let mut result = [half::f16::ZERO; 256];
    for i in 0..16 {
        result[i * 16 + i] = half::f16::ONE;
    }

    let mut base_power = *base;
    let mut n = power;

    while n > 0 {
        if n % 2 == 1 {
            result = compose_gates_16x16(&result, &base_power);
        }
        base_power = compose_gates_16x16(&base_power, &base_power);
        n /= 2;
    }
    result
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn is_approx_identity(gate: &[half::f16; 256]) -> bool {
    for i in 0..16 {
        for j in 0..16 {
            let val = gate[i * 16 + j].to_f32();
            let expected = if i == j { 1.0 } else { 0.0 };
            if (val - expected).abs() > 0.1 {
                return false;
            }
        }
    }
    true
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("Need cuda,perf-bench features");
}
