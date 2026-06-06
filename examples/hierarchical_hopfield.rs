//! Hierarchical Hopfield Network Benchmark (Sprint 81)
//!
//! Tests the two-level Hopfield architecture:
//!   L0: Local packed blocks with Hebbian weights
//!   L1: Boundary consistency checker (soft stitching)
//!
//! Pushes into the regime where L1 should matter:
//!   - Rich templates (8 types with internal structure)
//!   - 64x64 and 128x128 grids (64-256 blocks)
//!   - Confusable patterns (locally identical, globally different)
//!   - Manual recall loop (no adaptive L1 shutoff)
//!   - Boundary weight survival diagnostics
//!
//! Run with: cargo run --release --features cuda --example hierarchical_hopfield

#[cfg(feature = "cuda")]
fn main() {
    use engine::cuda_tiles::{HierarchicalConfig, HierarchicalHopfield};
    use logic_fabric_core::cuda::CudaRuntime;
    use std::time::Instant;

    println!("=== Hierarchical Hopfield Network Benchmark (v2) ===");
    println!("Sprint 81: Pushing L1 into the regime where it matters\n");

    let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");
    println!("GPU: {}\n", rt.device_name().unwrap_or("unknown".into()));

    let block_size = 8;

    // ========================================================================
    // Phase 0: Quick Sanity Check (condensed from old parameter sweeps)
    // ========================================================================
    println!("=== Phase 0: Sanity Check (32x32, block pattern, 20% corruption) ===\n");
    {
        let w = 32;
        let h = 32;
        let pattern = generate_block_pattern(w, h, block_size, 42);
        let corrupted = corrupt_pattern(&pattern, 0.2, 42);

        println!(
            "  {:>8}  {:>8}  {:>10}  {:>10}",
            "Beta", "Strength", "Overlap", "Boundary"
        );
        println!("  {:->8}  {:->8}  {:->10}  {:->10}", "", "", "", "");

        for &(beta, strength) in &[(3.0, 0.3), (5.0, 0.3), (5.0, 0.5)] {
            let config = HierarchicalConfig::new(block_size)
                .with_l0_steps(4)
                .with_correction_strength(strength)
                .with_beta(beta, beta);

            let mut hh = HierarchicalHopfield::new(&rt, w, h, config).expect("create");
            hh.store_pattern(&pattern).expect("store");
            hh.finalize_weights(&rt).expect("finalize");

            let (recovered, stats) = manual_recall(&mut hh, &rt, &corrupted, 50);
            let overlap = pattern_overlap(&pattern, &recovered);
            let ba = stats.last().map(|s| s.boundary_agreement).unwrap_or(0.0);

            println!(
                "  {:>8.1}  {:>8.1}  {:>10.4}  {:>10.4}",
                beta, strength, overlap, ba
            );
        }
    }

    // ========================================================================
    // Phase 1: Multi-Pattern Capacity at 64x64 (the key test)
    // ========================================================================
    println!("\n=== Phase 1: Multi-Pattern Capacity at 64x64 ===");
    println!("  64x64 grid, 8x8 blocks = 64 blocks. Rich templates (8 types).");
    println!("  L0 capacity for 64 neurons ~ 9 patterns. Pushing past that.\n");
    {
        let w = 64;
        let h = 64;
        let templates = generate_rich_templates(block_size);

        println!(
            "  {:>3} {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}  {:>6}",
            "P", "flat20%", "flat30%", "hier20%", "hier30%", "d20%", "d30%", "surv"
        );
        println!(
            "  {:->3} {:->8} {:->8}  {:->8} {:->8}  {:->8} {:->8}  {:->6}",
            "", "", "", "", "", "", "", ""
        );

        for &num_patterns in &[3, 5, 9, 12, 15, 20, 25] {
            let patterns: Vec<Vec<i8>> = (0..num_patterns)
                .map(|s| generate_rich_compositional(w, h, block_size, &templates, 100 + s as u64))
                .collect();

            // Count surviving boundary weights at this pattern count
            let surviving = count_surviving_boundary_weights(&patterns, w, h, block_size);

            let mut flat_ov = [0.0f64; 2];
            let mut hier_ov = [0.0f64; 2];

            for (ci, &corruption) in [0.2, 0.3].iter().enumerate() {
                let mut f_sum = 0.0;
                let mut h_sum = 0.0;

                for p_idx in 0..num_patterns {
                    let corrupted =
                        corrupt_pattern(&patterns[p_idx], corruption, 42 + p_idx as u64);

                    // Flat (no L1)
                    let mut hh = HierarchicalHopfield::new(
                        &rt,
                        w,
                        h,
                        HierarchicalConfig::new(block_size)
                            .with_l0_steps(4)
                            .with_correction_strength(0.0)
                            .with_beta(5.0, 5.0),
                    )
                    .expect("create");
                    for pat in &patterns {
                        hh.store_pattern(pat).expect("store");
                    }
                    hh.finalize_weights(&rt).expect("finalize");
                    let (rec, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
                    f_sum += pattern_overlap(&patterns[p_idx], &rec);

                    // Hierarchical (L1 active)
                    let mut hh = HierarchicalHopfield::new(
                        &rt,
                        w,
                        h,
                        HierarchicalConfig::new(block_size)
                            .with_l0_steps(4)
                            .with_correction_strength(0.5)
                            .with_beta(5.0, 5.0),
                    )
                    .expect("create");
                    for pat in &patterns {
                        hh.store_pattern(pat).expect("store");
                    }
                    hh.finalize_weights(&rt).expect("finalize");
                    let (rec, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
                    h_sum += pattern_overlap(&patterns[p_idx], &rec);
                }

                flat_ov[ci] = f_sum / num_patterns as f64;
                hier_ov[ci] = h_sum / num_patterns as f64;
            }

            let d20 = hier_ov[0] - flat_ov[0];
            let d30 = hier_ov[1] - flat_ov[1];
            let marker = if d20 > 0.01 {
                "  <-- L1 helps"
            } else if d20 < -0.01 {
                "  <-- L1 hurts"
            } else {
                ""
            };

            println!(
                "  {:>3} {:>8.4} {:>8.4}  {:>8.4} {:>8.4}  {:>+8.4} {:>+8.4}  {:>6}{}",
                num_patterns,
                flat_ov[0],
                flat_ov[1],
                hier_ov[0],
                hier_ov[1],
                d20,
                d30,
                surviving,
                marker
            );
        }
    }

    // ========================================================================
    // Phase 2: Scale Comparison (64x64 vs 128x128)
    // ========================================================================
    println!("\n=== Phase 2: Scale Comparison (10 rich patterns, flat vs hierarchical) ===\n");
    {
        let templates = generate_rich_templates(block_size);
        let num_patterns = 10;

        println!(
            "  {:>8} {:>6} {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}  {:>5}",
            "Grid", "Blks", "flat20%", "flat30%", "hier20%", "hier30%", "d20%", "d30%", "surv"
        );
        println!(
            "  {:->8} {:->6} {:->8} {:->8}  {:->8} {:->8}  {:->8} {:->8}  {:->5}",
            "", "", "", "", "", "", "", "", ""
        );

        for &(w, h) in &[(64, 64), (128, 128)] {
            let patterns: Vec<Vec<i8>> = (0..num_patterns)
                .map(|s| generate_rich_compositional(w, h, block_size, &templates, 300 + s as u64))
                .collect();

            let surviving = count_surviving_boundary_weights(&patterns, w, h, block_size);

            let mut flat_ov = [0.0f64; 2];
            let mut hier_ov = [0.0f64; 2];

            for (ci, &corruption) in [0.2, 0.3].iter().enumerate() {
                let mut f_sum = 0.0;
                let mut h_sum = 0.0;

                for p_idx in 0..num_patterns {
                    let corrupted =
                        corrupt_pattern(&patterns[p_idx], corruption, 42 + p_idx as u64);

                    let mut hh = HierarchicalHopfield::new(
                        &rt,
                        w,
                        h,
                        HierarchicalConfig::new(block_size)
                            .with_l0_steps(4)
                            .with_correction_strength(0.0)
                            .with_beta(5.0, 5.0),
                    )
                    .expect("create");
                    for pat in &patterns {
                        hh.store_pattern(pat).expect("store");
                    }
                    hh.finalize_weights(&rt).expect("finalize");
                    let (rec, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
                    f_sum += pattern_overlap(&patterns[p_idx], &rec);

                    let mut hh = HierarchicalHopfield::new(
                        &rt,
                        w,
                        h,
                        HierarchicalConfig::new(block_size)
                            .with_l0_steps(4)
                            .with_correction_strength(0.5)
                            .with_beta(5.0, 5.0),
                    )
                    .expect("create");
                    for pat in &patterns {
                        hh.store_pattern(pat).expect("store");
                    }
                    hh.finalize_weights(&rt).expect("finalize");
                    let (rec, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
                    h_sum += pattern_overlap(&patterns[p_idx], &rec);
                }

                flat_ov[ci] = f_sum / num_patterns as f64;
                hier_ov[ci] = h_sum / num_patterns as f64;
            }

            let d20 = hier_ov[0] - flat_ov[0];
            let d30 = hier_ov[1] - flat_ov[1];
            let marker = if d20 > 0.01 {
                "  <-- L1 helps"
            } else if d20 < -0.01 {
                "  <-- L1 hurts"
            } else {
                ""
            };

            println!(
                "  {:>4}x{:<3} {:>6} {:>8.4} {:>8.4}  {:>8.4} {:>8.4}  {:>+8.4} {:>+8.4}  {:>5}{}",
                w,
                h,
                (w / block_size) * (h / block_size),
                flat_ov[0],
                flat_ov[1],
                hier_ov[0],
                hier_ov[1],
                d20,
                d30,
                surviving,
                marker
            );
        }
    }

    // ========================================================================
    // Phase 3: Confusable Patterns at 64x64
    // ========================================================================
    println!("\n=== Phase 3: Confusable Patterns at 64x64 ===");
    println!("  Patterns share 70-80% of block assignments, differ in 20-30%.");
    println!("  L0 can't distinguish. L1 boundary weights carry the signal.\n");
    {
        let w = 64;
        let h = 64;
        let templates = generate_rich_templates(block_size);

        println!(
            "  {:>3} {:>5} {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}  {:>5}",
            "P", "flip%", "flat20%", "flat30%", "hier20%", "hier30%", "d20%", "d30%", "surv"
        );
        println!(
            "  {:->3} {:->5} {:->8} {:->8}  {:->8} {:->8}  {:->8} {:->8}  {:->5}",
            "", "", "", "", "", "", "", "", ""
        );

        for &(num_patterns, flip_frac) in
            &[(4, 0.25), (8, 0.25), (12, 0.30), (16, 0.25), (16, 0.15)]
        {
            let patterns = generate_confusable_patterns(
                w,
                h,
                block_size,
                &templates,
                num_patterns,
                flip_frac,
                500,
            );

            let surviving = count_surviving_boundary_weights(&patterns, w, h, block_size);

            let mut flat_ov = [0.0f64; 2];
            let mut hier_ov = [0.0f64; 2];

            for (ci, &corruption) in [0.2, 0.3].iter().enumerate() {
                let mut f_sum = 0.0;
                let mut h_sum = 0.0;

                for p_idx in 0..num_patterns {
                    let corrupted =
                        corrupt_pattern(&patterns[p_idx], corruption, 42 + p_idx as u64);

                    let mut hh = HierarchicalHopfield::new(
                        &rt,
                        w,
                        h,
                        HierarchicalConfig::new(block_size)
                            .with_l0_steps(4)
                            .with_correction_strength(0.0)
                            .with_beta(5.0, 5.0),
                    )
                    .expect("create");
                    for pat in &patterns {
                        hh.store_pattern(pat).expect("store");
                    }
                    hh.finalize_weights(&rt).expect("finalize");
                    let (rec, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
                    f_sum += pattern_overlap(&patterns[p_idx], &rec);

                    let mut hh = HierarchicalHopfield::new(
                        &rt,
                        w,
                        h,
                        HierarchicalConfig::new(block_size)
                            .with_l0_steps(4)
                            .with_correction_strength(0.5)
                            .with_beta(5.0, 5.0),
                    )
                    .expect("create");
                    for pat in &patterns {
                        hh.store_pattern(pat).expect("store");
                    }
                    hh.finalize_weights(&rt).expect("finalize");
                    let (rec, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
                    h_sum += pattern_overlap(&patterns[p_idx], &rec);
                }

                flat_ov[ci] = f_sum / num_patterns as f64;
                hier_ov[ci] = h_sum / num_patterns as f64;
            }

            let d20 = hier_ov[0] - flat_ov[0];
            let d30 = hier_ov[1] - flat_ov[1];
            let marker = if d20 > 0.01 {
                "  <-- L1 helps"
            } else if d20 < -0.01 {
                "  <-- L1 hurts"
            } else {
                ""
            };

            println!(
                "  {:>3} {:>4}% {:>8.4} {:>8.4}  {:>8.4} {:>8.4}  {:>+8.4} {:>+8.4}  {:>5}{}",
                num_patterns,
                (flip_frac * 100.0) as u32,
                flat_ov[0],
                flat_ov[1],
                hier_ov[0],
                hier_ov[1],
                d20,
                d30,
                surviving,
                marker
            );
        }
    }

    // ========================================================================
    // Phase 4: Boundary Diagnostic Trace
    // ========================================================================
    println!("\n=== Phase 4: Boundary Diagnostic (64x64, 15 rich patterns, 25% corruption) ===");
    println!("  Tracking per-cycle: overlap, raw boundary agreement, weighted agreement,");
    println!("  and surviving boundary weight count after noise thresholding.\n");
    {
        let w = 64;
        let h = 64;
        let templates = generate_rich_templates(block_size);
        let num_patterns = 15;

        let patterns: Vec<Vec<i8>> = (0..num_patterns)
            .map(|s| generate_rich_compositional(w, h, block_size, &templates, 700 + s as u64))
            .collect();

        let surviving = count_surviving_boundary_weights(&patterns, w, h, block_size);
        let boundary_weights = compute_boundary_hebbian(&patterns, w, h, block_size);

        println!(
            "  Boundary weights surviving 2sigma threshold: {} / {} total edges",
            surviving,
            count_total_boundary_edges(w, h, block_size)
        );

        // Hierarchical run with diagnostic
        let config = HierarchicalConfig::new(block_size)
            .with_l0_steps(4)
            .with_correction_strength(0.5)
            .with_beta(5.0, 5.0);

        let mut hh = HierarchicalHopfield::new(&rt, w, h, config).expect("create");
        for pat in &patterns {
            hh.store_pattern(pat).expect("store");
        }
        hh.finalize_weights(&rt).expect("finalize");

        let target_pattern = &patterns[0];
        let corrupted = corrupt_pattern(target_pattern, 0.25, 42);
        hh.load_pattern(&rt, &corrupted).expect("load");

        let initial_overlap = pattern_overlap(target_pattern, &corrupted);
        println!("  Initial overlap: {:.4}\n", initial_overlap);

        println!(
            "  {:>6}  {:>10}  {:>8}  {:>10}",
            "Cycle", "Overlap", "Raw BA", "Wt'd BA"
        );
        println!("  {:->6}  {:->10}  {:->8}  {:->10}", "", "", "", "");

        for cycle in 0..80 {
            let stats = hh.step(&rt).expect("step");

            if cycle < 5 || cycle % 5 == 4 {
                let state = hh.read_state(&rt).expect("read");
                let overlap = pattern_overlap(target_pattern, &state);
                let weighted_ba = compute_weighted_boundary_agreement(
                    &state,
                    &boundary_weights,
                    w,
                    h,
                    block_size,
                    num_patterns,
                );

                println!(
                    "  {:>6}  {:>10.4}  {:>8.4}  {:>10.4}",
                    stats.cycle, overlap, stats.boundary_agreement, weighted_ba
                );
            }
        }

        // Also show flat comparison for same target
        let mut hh_flat = HierarchicalHopfield::new(
            &rt,
            w,
            h,
            HierarchicalConfig::new(block_size)
                .with_l0_steps(4)
                .with_correction_strength(0.0)
                .with_beta(5.0, 5.0),
        )
        .expect("create");
        for pat in &patterns {
            hh_flat.store_pattern(pat).expect("store");
        }
        hh_flat.finalize_weights(&rt).expect("finalize");
        let (flat_rec, _) = manual_recall(&mut hh_flat, &rt, &corrupted, 80);
        let flat_overlap = pattern_overlap(target_pattern, &flat_rec);

        let hier_state = hh.read_state(&rt).expect("read");
        let hier_overlap = pattern_overlap(target_pattern, &hier_state);

        println!(
            "\n  Final: flat={:.4}  hier={:.4}  delta={:+.4}",
            flat_overlap,
            hier_overlap,
            hier_overlap - flat_overlap
        );
    }

    // ========================================================================
    // Phase 5: Throughput at Scale
    // ========================================================================
    println!("\n=== Phase 5: Throughput (10 rich patterns, timed) ===\n");
    {
        let templates = generate_rich_templates(block_size);

        println!(
            "  {:>8} {:>8} {:>8} {:>10} {:>12} {:>10}",
            "Grid", "Neurons", "Blocks", "Cycles", "Time (ms)", "Cycles/s"
        );
        println!(
            "  {:->8} {:->8} {:->8} {:->10} {:->12} {:->10}",
            "", "", "", "", "", ""
        );

        for &(w, h) in &[(64, 64), (128, 128), (256, 256)] {
            let config = HierarchicalConfig::new(block_size)
                .with_l0_steps(4)
                .with_correction_strength(0.5)
                .with_beta(5.0, 5.0);

            let mut hh = HierarchicalHopfield::new(&rt, w, h, config).expect("create");

            let patterns: Vec<Vec<i8>> = (0..10)
                .map(|s| generate_rich_compositional(w, h, block_size, &templates, 900 + s as u64))
                .collect();
            for pat in &patterns {
                hh.store_pattern(pat).expect("store");
            }
            hh.finalize_weights(&rt).expect("finalize");

            let corrupted = corrupt_pattern(&patterns[0], 0.3, 42);
            hh.load_pattern(&rt, &corrupted).expect("load");

            // Warmup
            for _ in 0..5 {
                let _ = hh.step(&rt);
            }

            // Benchmark
            let num_cycles = 100;
            hh.load_pattern(&rt, &corrupted).expect("load");

            let start = Instant::now();
            for _ in 0..num_cycles {
                let _ = hh.step(&rt);
            }
            rt.synchronize().expect("sync");
            let elapsed = start.elapsed();

            let cycles_per_sec = num_cycles as f64 / elapsed.as_secs_f64();

            println!(
                "  {:>4}x{:<4} {:>8} {:>8} {:>10} {:>12.2} {:>10.1}",
                w,
                h,
                w * h,
                (w / block_size) * (h / block_size),
                num_cycles,
                elapsed.as_secs_f64() * 1000.0,
                cycles_per_sec
            );
        }
    }

    // ========================================================================
    // Phase 6: Summary
    // ========================================================================
    println!("\n=== Phase 6: Summary — Flat vs Hierarchical Across All Regimes ===\n");
    {
        let templates = generate_rich_templates(block_size);

        println!(
            "  {:<28} {:>8} {:>8}  {:>8}",
            "Regime", "Flat", "Hier", "Delta"
        );
        println!("  {:<28} {:->8} {:->8}  {:->8}", "", "", "", "");

        // A) Single pattern, 32x32 (should be equal — L0 handles it)
        {
            let w = 32;
            let h = 32;
            let pat = generate_block_pattern(w, h, block_size, 42);
            let corrupted = corrupt_pattern(&pat, 0.3, 42);

            let mut hf = HierarchicalHopfield::new(
                &rt,
                w,
                h,
                HierarchicalConfig::new(block_size)
                    .with_l0_steps(4)
                    .with_correction_strength(0.0)
                    .with_beta(5.0, 5.0),
            )
            .expect("c");
            hf.store_pattern(&pat).expect("s");
            hf.finalize_weights(&rt).expect("f");
            let (rf, _) = manual_recall(&mut hf, &rt, &corrupted, 50);
            let of = pattern_overlap(&pat, &rf);

            let mut hh = HierarchicalHopfield::new(
                &rt,
                w,
                h,
                HierarchicalConfig::new(block_size)
                    .with_l0_steps(4)
                    .with_correction_strength(0.5)
                    .with_beta(5.0, 5.0),
            )
            .expect("c");
            hh.store_pattern(&pat).expect("s");
            hh.finalize_weights(&rt).expect("f");
            let (rh, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
            let oh = pattern_overlap(&pat, &rh);

            println!(
                "  {:<28} {:>8.4} {:>8.4}  {:>+8.4}",
                "1 pat, 32x32, 30%",
                of,
                oh,
                oh - of
            );
        }

        // B) 15 rich patterns, 64x64
        {
            let w = 64;
            let h = 64;
            let np = 15;
            let pats: Vec<Vec<i8>> = (0..np)
                .map(|s| generate_rich_compositional(w, h, block_size, &templates, 100 + s as u64))
                .collect();
            let corrupted = corrupt_pattern(&pats[0], 0.2, 42);

            let mut hf = HierarchicalHopfield::new(
                &rt,
                w,
                h,
                HierarchicalConfig::new(block_size)
                    .with_l0_steps(4)
                    .with_correction_strength(0.0)
                    .with_beta(5.0, 5.0),
            )
            .expect("c");
            for p in &pats {
                hf.store_pattern(p).expect("s");
            }
            hf.finalize_weights(&rt).expect("f");
            let (rf, _) = manual_recall(&mut hf, &rt, &corrupted, 50);
            let of = pattern_overlap(&pats[0], &rf);

            let mut hh = HierarchicalHopfield::new(
                &rt,
                w,
                h,
                HierarchicalConfig::new(block_size)
                    .with_l0_steps(4)
                    .with_correction_strength(0.5)
                    .with_beta(5.0, 5.0),
            )
            .expect("c");
            for p in &pats {
                hh.store_pattern(p).expect("s");
            }
            hh.finalize_weights(&rt).expect("f");
            let (rh, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
            let oh = pattern_overlap(&pats[0], &rh);

            println!(
                "  {:<28} {:>8.4} {:>8.4}  {:>+8.4}",
                "15 rich, 64x64, 20%",
                of,
                oh,
                oh - of
            );
        }

        // C) 8 confusable patterns, 64x64
        {
            let w = 64;
            let h = 64;
            let pats = generate_confusable_patterns(w, h, block_size, &templates, 8, 0.25, 500);
            let corrupted = corrupt_pattern(&pats[0], 0.2, 42);

            let mut hf = HierarchicalHopfield::new(
                &rt,
                w,
                h,
                HierarchicalConfig::new(block_size)
                    .with_l0_steps(4)
                    .with_correction_strength(0.0)
                    .with_beta(5.0, 5.0),
            )
            .expect("c");
            for p in &pats {
                hf.store_pattern(p).expect("s");
            }
            hf.finalize_weights(&rt).expect("f");
            let (rf, _) = manual_recall(&mut hf, &rt, &corrupted, 50);
            let of = pattern_overlap(&pats[0], &rf);

            let mut hh = HierarchicalHopfield::new(
                &rt,
                w,
                h,
                HierarchicalConfig::new(block_size)
                    .with_l0_steps(4)
                    .with_correction_strength(0.5)
                    .with_beta(5.0, 5.0),
            )
            .expect("c");
            for p in &pats {
                hh.store_pattern(p).expect("s");
            }
            hh.finalize_weights(&rt).expect("f");
            let (rh, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
            let oh = pattern_overlap(&pats[0], &rh);

            println!(
                "  {:<28} {:>8.4} {:>8.4}  {:>+8.4}",
                "8 confusable, 64x64, 20%",
                of,
                oh,
                oh - of
            );
        }

        // D) 10 rich patterns, 128x128
        {
            let w = 128;
            let h = 128;
            let np = 10;
            let pats: Vec<Vec<i8>> = (0..np)
                .map(|s| generate_rich_compositional(w, h, block_size, &templates, 300 + s as u64))
                .collect();
            let corrupted = corrupt_pattern(&pats[0], 0.2, 42);

            let mut hf = HierarchicalHopfield::new(
                &rt,
                w,
                h,
                HierarchicalConfig::new(block_size)
                    .with_l0_steps(4)
                    .with_correction_strength(0.0)
                    .with_beta(5.0, 5.0),
            )
            .expect("c");
            for p in &pats {
                hf.store_pattern(p).expect("s");
            }
            hf.finalize_weights(&rt).expect("f");
            let (rf, _) = manual_recall(&mut hf, &rt, &corrupted, 50);
            let of = pattern_overlap(&pats[0], &rf);

            let mut hh = HierarchicalHopfield::new(
                &rt,
                w,
                h,
                HierarchicalConfig::new(block_size)
                    .with_l0_steps(4)
                    .with_correction_strength(0.5)
                    .with_beta(5.0, 5.0),
            )
            .expect("c");
            for p in &pats {
                hh.store_pattern(p).expect("s");
            }
            hh.finalize_weights(&rt).expect("f");
            let (rh, _) = manual_recall(&mut hh, &rt, &corrupted, 50);
            let oh = pattern_overlap(&pats[0], &rh);

            println!(
                "  {:<28} {:>8.4} {:>8.4}  {:>+8.4}",
                "10 rich, 128x128, 20%",
                of,
                oh,
                oh - of
            );
        }
    }

    println!("\n=== Benchmark Complete ===");
}

// ============================================================================
// Manual Recall (bypasses adaptive L1 shutoff in recall())
// ============================================================================

#[cfg(feature = "cuda")]
fn manual_recall(
    hh: &mut engine::cuda_tiles::HierarchicalHopfield,
    rt: &logic_fabric_core::cuda::CudaRuntime,
    corrupted: &[i8],
    max_cycles: u32,
) -> (Vec<i8>, Vec<engine::cuda_tiles::HierarchicalStepStats>) {
    hh.load_pattern(rt, corrupted).expect("load");
    let mut stats = Vec::new();
    for _ in 0..max_cycles {
        stats.push(hh.step(rt).expect("step"));
    }
    let result = hh.read_state(rt).expect("read");
    (result, stats)
}

// ============================================================================
// Rich Templates (8 structurally distinct block types)
// ============================================================================

fn generate_rich_templates(block_size: usize) -> Vec<Vec<i8>> {
    let bs = block_size;
    let n = bs * bs;
    vec![
        // 0: Solid +1
        vec![1i8; n],
        // 1: Solid -1
        vec![-1i8; n],
        // 2: Top half +1, bottom half -1
        (0..n)
            .map(|i| if i / bs < bs / 2 { 1 } else { -1 })
            .collect(),
        // 3: Left half +1, right half -1
        (0..n)
            .map(|i| if i % bs < bs / 2 { 1 } else { -1 })
            .collect(),
        // 4: Diagonal: upper-left triangle +1
        (0..n)
            .map(|i| if (i % bs) + (i / bs) < bs { 1 } else { -1 })
            .collect(),
        // 5: Anti-diagonal: upper-right triangle +1
        (0..n)
            .map(|i| {
                if (bs - 1 - i % bs) + (i / bs) < bs {
                    1
                } else {
                    -1
                }
            })
            .collect(),
        // 6: 2x2 checkerboard within block
        (0..n)
            .map(|i| {
                if ((i % bs) / 2 + (i / bs) / 2) % 2 == 0 {
                    1
                } else {
                    -1
                }
            })
            .collect(),
        // 7: Corners +1, center -1
        (0..n)
            .map(|i| {
                let x = i % bs;
                let y = i / bs;
                if (x < bs / 4 || x >= bs - bs / 4) && (y < bs / 4 || y >= bs - bs / 4) {
                    1
                } else {
                    -1
                }
            })
            .collect(),
    ]
}

// ============================================================================
// Pattern Generators
// ============================================================================

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next() & 0xFFFFFFFF) as f64 / 4294967295.0
    }
}

/// Generate a pattern by assigning random rich templates to each block
fn generate_rich_compositional(
    width: usize,
    height: usize,
    block_size: usize,
    templates: &[Vec<i8>],
    seed: u64,
) -> Vec<i8> {
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;
    let mut rng = Rng::new(seed);

    let mut p = vec![0i8; width * height];
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let t = &templates[(rng.next() as usize) % templates.len()];
            for dy in 0..block_size {
                for dx in 0..block_size {
                    p[(by * block_size + dy) * width + bx * block_size + dx] =
                        t[dy * block_size + dx];
                }
            }
        }
    }
    p
}

/// Generate confusable patterns: start from a base, flip a fraction of block assignments
fn generate_confusable_patterns(
    width: usize,
    height: usize,
    block_size: usize,
    templates: &[Vec<i8>],
    num_patterns: usize,
    flip_fraction: f64,
    seed: u64,
) -> Vec<Vec<i8>> {
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;
    let num_blocks = blocks_x * blocks_y;
    let mut rng = Rng::new(seed);

    // Base assignment: random template per block
    let base_assignment: Vec<usize> = (0..num_blocks)
        .map(|_| (rng.next() as usize) % templates.len())
        .collect();

    let mut patterns = Vec::with_capacity(num_patterns);

    // First pattern is the base
    patterns.push(assignment_to_pattern(
        &base_assignment,
        templates,
        width,
        height,
        block_size,
    ));

    // Subsequent patterns: flip flip_fraction of blocks to a different template
    for _ in 1..num_patterns {
        let mut assignment = base_assignment.clone();
        for b in 0..num_blocks {
            if rng.next_f64() < flip_fraction {
                // Pick a different template
                let old = assignment[b];
                loop {
                    let new = (rng.next() as usize) % templates.len();
                    if new != old {
                        assignment[b] = new;
                        break;
                    }
                }
            }
        }
        patterns.push(assignment_to_pattern(
            &assignment,
            templates,
            width,
            height,
            block_size,
        ));
    }

    patterns
}

fn assignment_to_pattern(
    assignment: &[usize],
    templates: &[Vec<i8>],
    width: usize,
    height: usize,
    block_size: usize,
) -> Vec<i8> {
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;
    let mut p = vec![0i8; width * height];
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let t = &templates[assignment[by * blocks_x + bx]];
            for dy in 0..block_size {
                for dx in 0..block_size {
                    p[(by * block_size + dy) * width + bx * block_size + dx] =
                        t[dy * block_size + dx];
                }
            }
        }
    }
    p
}

fn generate_block_pattern(width: usize, height: usize, block_size: usize, seed: u64) -> Vec<i8> {
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;
    let mut rng = Rng::new(seed);
    let signs: Vec<i8> = (0..blocks_x * blocks_y)
        .map(|_| if rng.next() & 1 == 0 { 1i8 } else { -1i8 })
        .collect();

    let mut p = vec![0i8; width * height];
    for y in 0..height {
        for x in 0..width {
            p[y * width + x] = signs[(y / block_size) * blocks_x + x / block_size];
        }
    }
    p
}

fn corrupt_pattern(pattern: &[i8], fraction: f64, seed: u64) -> Vec<i8> {
    let mut rng = Rng::new(seed);
    pattern
        .iter()
        .map(|&s| if rng.next_f64() < fraction { -s } else { s })
        .collect()
}

fn pattern_overlap(a: &[i8], b: &[i8]) -> f64 {
    let dot: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai as i64 * bi as i64)
        .sum();
    dot as f64 / a.len() as f64
}

// ============================================================================
// Boundary Weight Diagnostics
// ============================================================================

/// Compute boundary Hebbian weights from stored patterns (CPU-side)
/// Returns (boundary_h, boundary_v) as Vec<f64> over width*height
fn compute_boundary_hebbian(
    patterns: &[Vec<i8>],
    width: usize,
    height: usize,
    block_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = width * height;
    let mut bh = vec![0.0f64; n];
    let mut bv = vec![0.0f64; n];

    for pat in patterns {
        for y in 0..height {
            for x in 0..width {
                let si = pat[y * width + x] as f64;
                // Horizontal cross-block edge
                if x % block_size == block_size - 1 && x + 1 < width {
                    let sj = pat[y * width + x + 1] as f64;
                    bh[y * width + x] += si * sj;
                }
                // Vertical cross-block edge
                if y % block_size == block_size - 1 && y + 1 < height {
                    let sj = pat[(y + 1) * width + x] as f64;
                    bv[y * width + x] += si * sj;
                }
            }
        }
    }
    (bh, bv)
}

/// Count how many boundary weights survive the 2sigma noise threshold
fn count_surviving_boundary_weights(
    patterns: &[Vec<i8>],
    width: usize,
    height: usize,
    block_size: usize,
) -> usize {
    let (bh, bv) = compute_boundary_hebbian(patterns, width, height, block_size);
    let p = patterns.len() as f64;
    let noise_floor = if p > 1.0 { 2.0 / p.sqrt() } else { 0.0 };

    let mut count = 0;
    for y in 0..height {
        for x in 0..width {
            if x % block_size == block_size - 1 && x + 1 < width {
                if (bh[y * width + x] / p).abs() > noise_floor {
                    count += 1;
                }
            }
            if y % block_size == block_size - 1 && y + 1 < height {
                if (bv[y * width + x] / p).abs() > noise_floor {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Count total boundary edges in the grid
fn count_total_boundary_edges(width: usize, height: usize, block_size: usize) -> usize {
    let mut count = 0;
    for y in 0..height {
        for x in 0..width {
            if x % block_size == block_size - 1 && x + 1 < width {
                count += 1;
            }
            if y % block_size == block_size - 1 && y + 1 < height {
                count += 1;
            }
        }
    }
    count
}

/// Compute weighted boundary agreement (CPU-side)
/// Uses Hebbian weights to determine whether each boundary pair SHOULD agree or disagree.
/// Agreement = fraction of boundary edges where sign(J * si * sj) > 0, weighted by |J|.
fn compute_weighted_boundary_agreement(
    state: &[i8],
    boundary_weights: &(Vec<f64>, Vec<f64>),
    width: usize,
    height: usize,
    block_size: usize,
    num_patterns: usize,
) -> f64 {
    let p = num_patterns as f64;
    let (ref bh, ref bv) = *boundary_weights;

    let mut weighted_correct = 0.0;
    let mut weight_sum = 0.0;

    for y in 0..height {
        for x in 0..width {
            // Horizontal cross-block edge
            if x % block_size == block_size - 1 && x + 1 < width {
                let si = state[y * width + x] as f64;
                let sj = state[y * width + x + 1] as f64;
                let j = bh[y * width + x] / p;
                let w = j.abs();
                if w > 1e-10 {
                    if j * si * sj > 0.0 {
                        weighted_correct += w;
                    }
                    weight_sum += w;
                }
            }
            // Vertical cross-block edge
            if y % block_size == block_size - 1 && y + 1 < height {
                let si = state[y * width + x] as f64;
                let sj = state[(y + 1) * width + x] as f64;
                let j = bv[y * width + x] / p;
                let w = j.abs();
                if w > 1e-10 {
                    if j * si * sj > 0.0 {
                        weighted_correct += w;
                    }
                    weight_sum += w;
                }
            }
        }
    }

    if weight_sum > 0.0 {
        weighted_correct / weight_sum
    } else {
        1.0
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the 'cuda' feature.");
    println!("Run with: cargo run --release --features cuda --example hierarchical_hopfield");
}
