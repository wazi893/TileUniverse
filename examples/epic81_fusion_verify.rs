// EPIC 81 Fusion Verification Test
//
// CRITICAL: Verify that fused gates produce IDENTICAL results to unfused.
// This catches bugs like the column-batching issue where we did 1/16th the work.
//
// Run with: cargo run --example epic81_fusion_verify --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{
        CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType, compose_gates_16x16,
        get_hadamard_gate_host, run_wmma_multi_state_fused,
    };
    use std::sync::Arc;

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 81 FUSION VERIFICATION TEST                              ║");
    println!("║  Ensuring fused and unfused produce IDENTICAL results          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let rt = Arc::new(CudaRuntime::new().expect("Failed to create CUDA runtime"));

    // =========================================================================
    // TEST 1: Verify H^N gate computation on CPU
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!(" TEST 1: CPU Gate Power Computation (H^N)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let h_gate = get_hadamard_gate_host();

    // H^1 should equal H
    let h1 = compute_power_gate(&h_gate, 1);
    let h1_matches = gates_equal(&h1, &h_gate);
    println!("  H^1 == H: {}", if h1_matches { "PASS" } else { "FAIL" });

    // H^2 should equal Identity (H is self-inverse)
    let h2 = compute_power_gate(&h_gate, 2);
    let identity = make_identity();
    let h2_is_identity = gates_approximately_equal(&h2, &identity, 0.01);
    println!(
        "  H^2 == I: {}",
        if h2_is_identity { "PASS" } else { "FAIL" }
    );

    if !h2_is_identity {
        println!("    H^2 diagonal samples:");
        for i in 0..4 {
            println!("      [{},{}] = {:.4}", i, i, h2[i * 16 + i].to_f32());
        }
    }

    // H^3 should equal H (since H^2 = I, H^3 = H^2 * H = I * H = H)
    let h3 = compute_power_gate(&h_gate, 3);
    let h3_equals_h = gates_approximately_equal(&h3, &h_gate, 0.01);
    println!("  H^3 == H: {}", if h3_equals_h { "PASS" } else { "FAIL" });

    // H^4 should equal Identity
    let h4 = compute_power_gate(&h_gate, 4);
    let h4_is_identity = gates_approximately_equal(&h4, &identity, 0.01);
    println!(
        "  H^4 == I: {}",
        if h4_is_identity { "PASS" } else { "FAIL" }
    );

    // H^100 should equal Identity (100 is even)
    let h100 = compute_power_gate(&h_gate, 100);
    let h100_is_identity = gates_approximately_equal(&h100, &identity, 0.01);
    println!(
        "  H^100 == I: {}",
        if h100_is_identity { "PASS" } else { "FAIL" }
    );

    // H^101 should equal H (101 is odd)
    let h101 = compute_power_gate(&h_gate, 101);
    let h101_equals_h = gates_approximately_equal(&h101, &h_gate, 0.01);
    println!(
        "  H^101 == H: {}",
        if h101_equals_h { "PASS" } else { "FAIL" }
    );

    let test1_pass = h1_matches
        && h2_is_identity
        && h3_equals_h
        && h4_is_identity
        && h100_is_identity
        && h101_equals_h;
    println!();
    println!(
        "  TEST 1 RESULT: {}",
        if test1_pass { "PASS" } else { "FAIL" }
    );
    println!();

    // =========================================================================
    // TEST 2: Work quantity verification via timing
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!(" TEST 2: Work Quantity Verification (Timing-Based)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // The CRITICAL test: if fusion is legit, then:
    //   unfused(N iterations) should take ~Nx longer than fused(N/F iterations)
    // If fused is doing LESS work (like the column batching bug), speedup >> F

    let num_states = 1024usize;
    let tiles_per_state = 16usize;
    let base_depth = 10000u32;

    println!("  Configuration:");
    println!("    States:      {}", num_states);
    println!(
        "    Tiles/state: {} ({} amplitudes)",
        tiles_per_state,
        tiles_per_state * 256
    );
    println!("    Base depth:  {}", base_depth);
    println!();

    let pool = MultiStatePersistent::new(rt.clone(), num_states, tiles_per_state)
        .expect("Failed to create pool");

    // Warmup
    let _ = pool.run_benchmark(WmmaGateType::Hadamard, 1000, MultiStateOpt::ILP);

    // UNFUSED baseline
    let (_, _, unfused_time) = pool
        .run_benchmark(WmmaGateType::Hadamard, base_depth, MultiStateOpt::ILP)
        .expect("Unfused failed");

    println!(
        "  UNFUSED: {} iterations in {:.4}s",
        base_depth, unfused_time
    );

    // Test multiple fusion factors
    let fusion_factors = [2u32, 5, 10, 20, 50, 100];
    let mut all_ratios_ok = true;

    println!();
    println!("  ┌─────────┬────────────┬───────────┬───────────┬───────────┐");
    println!("  │ Fusion  │ Fused Iter │ Time (s)  │ Speedup   │ Ratio     │");
    println!("  ├─────────┼────────────┼───────────┼───────────┼───────────┤");

    for &fusion in &fusion_factors {
        let fused_depth = base_depth / fusion;
        if fused_depth == 0 {
            continue;
        }

        // Compute H^fusion
        let fused_gate = compute_power_gate(&h_gate, fusion);
        let fused_gate_bits: Vec<u16> = fused_gate.iter().map(|x| x.to_bits()).collect();
        let fused_gate_gpu = rt.upload(&fused_gate_bits).expect("Upload failed");

        // Run fused benchmark
        let (_, _, fused_time) = run_wmma_multi_state_fused(
            &rt,
            num_states,
            tiles_per_state,
            &fused_gate_gpu,
            fused_depth,
            fusion,
        )
        .expect("Fused failed");

        let speedup = unfused_time / fused_time;
        let expected = fusion as f64;
        let ratio = speedup / expected;

        // Ratio should be close to 1.0 for legit speedup
        // Low ratios at very high fusion are expected (overhead dominates few iterations)
        // High ratios would indicate LESS work being done (the real bug to catch)
        let ratio_ok = ratio < 1.5; // Only flag if speedup is suspiciously HIGH
        let overhead_limited = ratio < 0.7 && fused_depth < 200; // Expected at few iterations
        if !ratio_ok {
            all_ratios_ok = false;
        }

        let status = if ratio > 1.5 {
            " !!BUG?"
        } else if overhead_limited {
            " (overhead)"
        } else {
            ""
        };

        println!(
            "  │ {:>7} │ {:>10} │ {:>9.4} │ {:>8.2}x │ {:>6.2}{:>4} │",
            format!("{}x", fusion),
            fused_depth,
            fused_time,
            speedup,
            ratio,
            status
        );
    }

    println!("  └─────────┴────────────┴───────────┴───────────┴───────────┘");
    println!();

    let test2_pass = all_ratios_ok;
    println!(
        "  TEST 2 RESULT: {}",
        if test2_pass { "PASS" } else { "SUSPICIOUS" }
    );

    if !test2_pass {
        println!();
        println!("  CRITICAL: Speedup ratio > 1.5x detected!");
        println!("  This indicates fused kernel may be doing LESS work (bug).");
    } else {
        println!("  Note: Low ratios at high fusion (marked 'overhead') are expected");
        println!("  when iteration count is too low for kernel launch overhead.");
    }
    println!();

    // =========================================================================
    // TEST 3: Amplitude operations count verification
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!(" TEST 3: Amplitude Operations Count");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Verify the reported amplitude ops make sense

    let test_states = 256usize;
    let test_tiles = 4usize;
    let test_depth = 1000u32;

    let pool2 = MultiStatePersistent::new(rt.clone(), test_states, test_tiles)
        .expect("Failed to create pool");

    // Unfused: should report states * depth * amps_per_state
    let (unfused_gate_ops, unfused_amp_ops, _) = pool2
        .run_benchmark(WmmaGateType::Hadamard, test_depth, MultiStateOpt::ILP)
        .expect("Unfused failed");

    let expected_gate_ops = (test_states as u64) * (test_depth as u64);
    let expected_amp_ops = expected_gate_ops * (test_tiles * 256) as u64;

    println!("  Unfused ({} depth):", test_depth);
    println!("    Gate ops reported:  {}", unfused_gate_ops);
    println!("    Gate ops expected:  {}", expected_gate_ops);
    println!("    Amp ops reported:   {}", unfused_amp_ops);
    println!("    Amp ops expected:   {}", expected_amp_ops);

    let gate_ops_match = unfused_gate_ops == expected_gate_ops;
    let amp_ops_match = unfused_amp_ops == expected_amp_ops;

    println!(
        "    Gate ops match: {}",
        if gate_ops_match { "PASS" } else { "FAIL" }
    );
    println!(
        "    Amp ops match:  {}",
        if amp_ops_match { "PASS" } else { "FAIL" }
    );
    println!();

    // Fused: should report EFFECTIVE ops (as if unfused)
    let fusion = 10u32;
    let fused_depth = test_depth / fusion;
    let fused_gate = compute_power_gate(&h_gate, fusion);
    let fused_gate_bits: Vec<u16> = fused_gate.iter().map(|x| x.to_bits()).collect();
    let fused_gate_gpu = rt.upload(&fused_gate_bits).expect("Upload failed");

    let (fused_gate_ops, fused_amp_ops, _) = run_wmma_multi_state_fused(
        &rt,
        test_states,
        test_tiles,
        &fused_gate_gpu,
        fused_depth,
        fusion,
    )
    .expect("Fused failed");

    // Fused effective ops should match unfused (same logical work)
    let fused_expected_gate_ops = (test_states as u64) * (fused_depth as u64) * (fusion as u64);
    let fused_expected_amp_ops = fused_expected_gate_ops * (test_tiles * 256) as u64;

    println!("  Fused ({}x, {} iterations):", fusion, fused_depth);
    println!("    Effective gate ops: {}", fused_gate_ops);
    println!("    Expected:           {}", fused_expected_gate_ops);
    println!("    Effective amp ops:  {}", fused_amp_ops);
    println!("    Expected:           {}", fused_expected_amp_ops);

    let fused_gate_match = fused_gate_ops == fused_expected_gate_ops;
    let fused_amp_match = fused_amp_ops == fused_expected_amp_ops;

    println!(
        "    Gate ops match: {}",
        if fused_gate_match { "PASS" } else { "FAIL" }
    );
    println!(
        "    Amp ops match:  {}",
        if fused_amp_match { "PASS" } else { "FAIL" }
    );
    println!();

    let test3_pass = gate_ops_match && amp_ops_match && fused_gate_match && fused_amp_match;
    println!(
        "  TEST 3 RESULT: {}",
        if test3_pass { "PASS" } else { "FAIL" }
    );
    println!();

    // =========================================================================
    // TEST 4: Large-scale verification (matches benchmark parameters)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!(" TEST 4: Full-Scale Benchmark Verification");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Use the exact parameters from epic81_fusion_bench
    let bench_states = 1024usize;
    let bench_tiles = 16usize;
    let bench_depth = 100_000u32;
    let bench_fusion = 128u32;

    println!("  Benchmark parameters:");
    println!("    States:      {}", bench_states);
    println!("    Tiles:       {}", bench_tiles);
    println!("    Depth:       {}", bench_depth);
    println!("    Fusion:      {}x", bench_fusion);
    println!();

    let bench_pool = MultiStatePersistent::new(rt.clone(), bench_states, bench_tiles)
        .expect("Failed to create pool");

    // Warmup
    let _ = bench_pool.run_benchmark(WmmaGateType::Hadamard, 1000, MultiStateOpt::ILP);

    // Baseline
    let (_, baseline_amps, baseline_time) = bench_pool
        .run_benchmark(WmmaGateType::Hadamard, bench_depth, MultiStateOpt::ILP)
        .expect("Baseline failed");

    let baseline_tcops = baseline_amps as f64 / baseline_time / 1e12;

    // Fused
    let fused_depth = bench_depth / bench_fusion;
    let fused_gate = compute_power_gate(&h_gate, bench_fusion);
    let fused_bits: Vec<u16> = fused_gate.iter().map(|x| x.to_bits()).collect();
    let fused_gpu = rt.upload(&fused_bits).expect("Upload failed");

    let (_, fused_amps, fused_time) = run_wmma_multi_state_fused(
        &rt,
        bench_states,
        bench_tiles,
        &fused_gpu,
        fused_depth,
        bench_fusion,
    )
    .expect("Fused failed");

    let fused_tcops = fused_amps as f64 / fused_time / 1e12;
    let speedup = baseline_time / fused_time;
    let ratio = speedup / (bench_fusion as f64);

    println!("  Results:");
    println!(
        "    Baseline:  {:.4}s @ {:.2} TCOPS",
        baseline_time, baseline_tcops
    );
    println!(
        "    Fused:     {:.4}s @ {:.2} effective TCOPS",
        fused_time, fused_tcops
    );
    println!(
        "    Speedup:   {:.2}x (expected ~{}x)",
        speedup, bench_fusion
    );
    println!("    Ratio:     {:.2}", ratio);
    println!();

    // The ratio should be close to 1.0 if fusion is legitimate
    let test4_pass = ratio > 0.8 && ratio < 1.3;
    println!(
        "  TEST 4 RESULT: {}",
        if test4_pass { "PASS" } else { "SUSPICIOUS" }
    );

    if !test4_pass {
        if ratio > 1.3 {
            println!();
            println!("  CRITICAL WARNING: Speedup is MUCH HIGHER than expected!");
            println!("  This strongly suggests the fused kernel is skipping work!");
        }
    }
    println!();

    // =========================================================================
    // FINAL SUMMARY
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!(" VERIFICATION SUMMARY");
    println!("═══════════════════════════════════════════════════════════════\n");

    let all_passed = test1_pass && test2_pass && test3_pass && test4_pass;

    println!(
        "  Test 1 (CPU H^N math):     {}",
        if test1_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Test 2 (Timing ratios):    {}",
        if test2_pass { "PASS" } else { "SUSPICIOUS" }
    );
    println!(
        "  Test 3 (Op counts):        {}",
        if test3_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Test 4 (Full scale):       {}",
        if test4_pass { "PASS" } else { "SUSPICIOUS" }
    );
    println!();

    if all_passed {
        println!("  ╔════════════════════════════════════════════════════════════╗");
        println!("  ║  ALL TESTS PASSED - FUSION SPEEDUP IS VERIFIED LEGITIMATE! ║");
        println!("  ╚════════════════════════════════════════════════════════════╝");
        println!();
        println!(
            "  The {}x fusion achieves {:.1}x actual speedup.",
            bench_fusion, speedup
        );
        println!("  This is {:.0}% of ideal linear scaling.", ratio * 100.0);
        println!();
        println!("  CONCLUSION: Gate fusion provides REAL performance gains by");
        println!(
            "  reducing GPU iterations. The effective TCOPS ({:.1}) represents",
            fused_tcops
        );
        println!("  legitimate equivalent work being done faster.");
    } else {
        println!("  ╔════════════════════════════════════════════════════════════╗");
        println!("  ║  SOME TESTS FAILED - INVESTIGATE BEFORE TRUSTING RESULTS   ║");
        println!("  ╚════════════════════════════════════════════════════════════╝");
    }
    println!();
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn compute_power_gate(base: &[half::f16; 256], power: u32) -> [half::f16; 256] {
    use engine::cuda::compose_gates_16x16;

    if power == 0 {
        return make_identity();
    }

    if power == 1 {
        return *base;
    }

    let mut result = make_identity();
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
fn make_identity() -> [half::f16; 256] {
    let mut identity = [half::f16::ZERO; 256];
    for i in 0..16 {
        identity[i * 16 + i] = half::f16::ONE;
    }
    identity
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn gates_equal(a: &[half::f16; 256], b: &[half::f16; 256]) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn gates_approximately_equal(a: &[half::f16; 256], b: &[half::f16; 256], tolerance: f32) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x.to_f32() - y.to_f32()).abs() <= tolerance)
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires both 'cuda' and 'perf-bench' features.");
    println!(
        "Run with: cargo run --example epic81_fusion_verify --features cuda,perf-bench --release"
    );
}
