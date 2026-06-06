//! Test for Week 2.5 adaptive annealing with Explore → Freeze → Polish
//!
//! Run with: cargo run --release --features cuda,perf-bench --example adaptive_annealing_test

#[cfg(feature = "cuda")]
fn main() {
    use engine::cuda::CudaRuntime;
    use engine::cuda_tiles::{
        AdaptiveMode, AdaptiveSchedule, AnnealingSchedule, GpuIsingGrid, IsingConfig,
        run_adaptive_annealing, run_simulated_annealing_best,
    };

    println!("=== TileAnneal Phase 2 Week 2.5: Adaptive Annealing with Polish ===\n");

    let rt = CudaRuntime::new().expect("CUDA runtime init failed");
    let width = 128;
    let height = 128;
    let j_coupling = -1.0; // Antiferromagnetic for MaxCut

    // Calculate theoretical optimal cut for reference (checkerboard = all edges cut)
    let optimal_cut = 2 * width * height - width - height;
    println!(
        "Grid: {}×{}, Optimal MaxCut: {} edges\n",
        width, height, optimal_cut
    );

    let total_sweeps = 50_000u32;

    // =========================================================================
    // Test 1: ThreeStage (Explore → Freeze → Polish) - THE RECOMMENDED MODE
    // =========================================================================
    println!("Test 1: ThreeStage Mode (Explore → Freeze → Polish)");
    println!("----------------------------------------------------");
    let threestage_result = {
        let config = IsingConfig::new(width, height).with_seed(42);
        let mut grid = GpuIsingGrid::new(&rt, &config).expect("grid");

        // Use the recommended robust preset (validated 98.6% quality)
        let mut schedule = AdaptiveSchedule::new(total_sweeps)
            .with_mode(AdaptiveMode::robust())
            .with_initial_beta(0.01);

        let result = run_adaptive_annealing(&rt, &mut grid, &mut schedule, j_coupling)
            .expect("adaptive annealing");

        let quality = result.best_cut as f64 / optimal_cut as f64 * 100.0;
        println!(
            "  Best cut: {} ({:.1}% of optimal)",
            result.best_cut, quality
        );
        println!("  Found at sweep: {}", result.best_sweep);
        println!("  Final β: {:.4}", schedule.beta());

        // Show phase transitions in history
        println!("  Phase progression:");
        let n = result.history.len();
        let sample_points = [0, n / 5, 2 * n / 5, 3 * n / 5, 4 * n / 5, n - 1];
        for i in sample_points {
            if i < n {
                let (sw, beta, rate, phase_ind) = result.history[i];
                let phase = if phase_ind > 0.0 {
                    "EXPLORE"
                } else if phase_ind == 0.0 {
                    "FREEZE"
                } else {
                    "POLISH"
                };
                println!(
                    "    sweep={:5}, β={:8.4}, up_rate={:.4} [{}]",
                    sw, beta, rate, phase
                );
            }
        }

        quality
    };
    println!();

    // =========================================================================
    // Test 2: Hold-acceptance (baseline thermostat, no commit)
    // =========================================================================
    println!("Test 2: Hold-Acceptance Mode (baseline thermostat)");
    println!("---------------------------------------------------");
    let hold_result = {
        let config = IsingConfig::new(width, height).with_seed(42);
        let mut grid = GpuIsingGrid::new(&rt, &config).expect("grid");

        let mut schedule = AdaptiveSchedule::new(total_sweeps)
            .with_mode(AdaptiveMode::HoldAcceptance { target: 0.25 })
            .with_initial_beta(0.01);

        let result = run_adaptive_annealing(&rt, &mut grid, &mut schedule, j_coupling)
            .expect("adaptive annealing");

        let quality = result.best_cut as f64 / optimal_cut as f64 * 100.0;
        println!(
            "  Best cut: {} ({:.1}% of optimal)",
            result.best_cut, quality
        );
        println!(
            "  Final β: {:.4} (stays warm - this is why it underperforms)",
            schedule.beta()
        );

        quality
    };
    println!();

    // =========================================================================
    // Test 3: TwoStage (explore + refine, no polish)
    // =========================================================================
    println!("Test 3: TwoStage Mode (no T=0 polish)");
    println!("-------------------------------------");
    let twostage_result = {
        let config = IsingConfig::new(width, height).with_seed(42);
        let mut grid = GpuIsingGrid::new(&rt, &config).expect("grid");

        let mut schedule = AdaptiveSchedule::new(total_sweeps)
            .with_mode(AdaptiveMode::TwoStage {
                explore_target: 0.35,
                refine_target: 0.05,
                explore_fraction: 0.5,
            })
            .with_initial_beta(0.01);

        let result = run_adaptive_annealing(&rt, &mut grid, &mut schedule, j_coupling)
            .expect("adaptive annealing");

        let quality = result.best_cut as f64 / optimal_cut as f64 * 100.0;
        println!(
            "  Best cut: {} ({:.1}% of optimal)",
            result.best_cut, quality
        );
        println!("  Final β: {:.4}", schedule.beta());

        quality
    };
    println!();

    // =========================================================================
    // Test 4: Static exponential (hand-tuned baseline)
    // =========================================================================
    println!("Test 4: Static Exponential Schedule (hand-tuned baseline)");
    println!("----------------------------------------------------------");
    let static_result = {
        let config = IsingConfig::new(width, height).with_seed(42);
        let mut grid = GpuIsingGrid::new(&rt, &config).expect("grid");
        let schedule = AnnealingSchedule::exponential(0.01, 50.0, total_sweeps);
        let result = run_simulated_annealing_best(&rt, &mut grid, &schedule, j_coupling)
            .expect("static annealing");

        let quality = result.best_cut as f64 / optimal_cut as f64 * 100.0;
        println!(
            "  Best cut: {} ({:.1}% of optimal)",
            result.best_cut, quality
        );
        println!("  (β ramps from 0.01 → 50.0 over {} sweeps)", total_sweeps);

        quality
    };
    println!();

    // =========================================================================
    // Summary comparison
    // =========================================================================
    println!("========================================");
    println!("SUMMARY: Quality vs Robustness Tradeoff");
    println!("========================================");
    println!();
    println!("| Mode              | Quality | Notes                    |");
    println!("|-------------------|---------|--------------------------|");
    println!(
        "| Static exponential| {:5.1}%  | Requires schedule tuning |",
        static_result
    );
    println!(
        "| ThreeStage (NEW)  | {:5.1}%  | Self-tuning, recommended |",
        threestage_result
    );
    println!(
        "| TwoStage          | {:5.1}%  | No T=0 polish            |",
        twostage_result
    );
    println!(
        "| Hold-acceptance   | {:5.1}%  | Thermostat only          |",
        hold_result
    );
    println!();

    // Check if ThreeStage closes the gap
    let gap_closed = threestage_result >= static_result - 10.0;
    if gap_closed {
        println!("✓ ThreeStage closes gap with static (within 10%)");
        println!(
            "  → \"Adaptive self-tunes exploration, then deterministic polish yields strong solutions.\""
        );
    } else {
        println!(
            "⚠ ThreeStage still behind static by {:.1}%",
            static_result - threestage_result
        );
        println!("  → Consider tuning explore_fraction or freeze_multiplier");
    }

    // Check improvement over baseline
    let improvement = threestage_result - hold_result;
    println!();
    println!(
        "Polish improvement: +{:.1}% over hold-acceptance baseline",
        improvement
    );

    println!("\n=== Week 2.5 Validation Complete ===");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature.");
    std::process::exit(1);
}
