//! Magic State Budget Benchmarks
//!
//! SPRINT 51.0 Phase 5: Comprehensive Resource Estimation
//!
//! Compares magic state costs across:
//! 1. Different QRAM sizes (16 to 64K cells)
//! 2. Different target error rates (10⁻⁶ to 10⁻¹⁵)
//! 3. Different distillation protocols
//! 4. Comparison with literature estimates
//!
//! Run: cargo run --release --example magic_state_budget

use engine::qram::{
    DistillationConfig, DistillationFactory, DistillationProtocol, MagicStateBudget,
    StabilizerQRAM, ToffoliDecomposition, ToffoliTracker,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║      SPRINT 51.0: Magic State Budget Benchmarks                      ║");
    println!("║      Fault-Tolerant Resource Estimation for QRAM                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Part 1: QRAM Size Scaling
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Part 1: QRAM Size Scaling (target error: 10⁻¹²)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    qram_size_scaling();

    // Part 2: Error Rate Comparison
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 2: Error Rate Comparison (depth=8, 256 cells)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    error_rate_comparison();

    // Part 3: Distillation Protocol Comparison
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 3: Distillation Protocol Comparison");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    distillation_protocol_comparison();

    // Part 4: Toffoli Decomposition Analysis
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 4: Toffoli Decomposition Analysis");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    toffoli_decomposition_analysis();

    // Part 5: StabilizerQRAM Integration Demo
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 5: StabilizerQRAM Integration Demo");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    stabilizer_qram_demo();

    // Part 6: Literature Comparison
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 6: Comparison with Literature Estimates");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    literature_comparison();

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("SPRINT 51.0 Magic State Budgeting Complete!");
    println!("═══════════════════════════════════════════════════════════════════════");
}

/// Compare magic state costs across different QRAM sizes
fn qram_size_scaling() {
    let depths = [4, 8, 10, 12, 14, 16]; // 16 to 64K cells
    let target_error = 1e-12;
    let config = DistillationConfig::for_error_rate(target_error);

    println!("Target error rate: {:.0e}", target_error);
    println!("Distillation levels: {}", config.levels);
    println!("Overhead ratio: {}:1\n", config.overhead_ratio());

    println!(
        "{:>6} {:>8} {:>10} {:>12} {:>15} {:>12} {:>15}",
        "Depth", "Cells", "Toffolis", "Bare T", "Magic States", "Factory(q)", "Est. Time"
    );
    println!("{}", "-".repeat(85));

    for depth in depths {
        let cells = 1usize << depth;
        let budget = MagicStateBudget::for_qram_routing(depth, 1).with_distillation(&config);

        let time_str = format_time(budget.estimated_seconds);

        println!(
            "{:>6} {:>8} {:>10} {:>12} {:>15} {:>12} {:>15}",
            depth,
            format_number(cells),
            budget.toffoli_count,
            budget.total_t_count_bare,
            format_number(budget.total_magic_states),
            format_number(budget.factory_qubits),
            time_str
        );
    }

    // Multi-query scaling
    println!("\n--- Multi-Query Scaling (depth=8, 256 cells) ---\n");
    println!(
        "{:>10} {:>12} {:>15} {:>15} {:>15}",
        "Queries", "Bare T", "Magic States", "Qubit-Cycles", "Est. Time"
    );
    println!("{}", "-".repeat(70));

    for queries in [1, 10, 100, 1000, 10000] {
        let budget = MagicStateBudget::for_qram_routing(8, queries).with_distillation(&config);

        println!(
            "{:>10} {:>12} {:>15} {:>15} {:>15}",
            format_number(queries),
            format_number(budget.total_t_count_bare),
            format_number(budget.total_magic_states),
            format_large_number(budget.qubit_cycles),
            format_time(budget.estimated_seconds)
        );
    }
}

/// Compare costs at different target error rates
fn error_rate_comparison() {
    let targets = [1e-6, 1e-9, 1e-12, 1e-15];
    let depth = 8;

    println!(
        "{:>12} {:>8} {:>10} {:>12} {:>15} {:>15}",
        "Target", "Levels", "Distance", "Overhead", "Magic States", "Factory(q)"
    );
    println!("{}", "-".repeat(80));

    for target in targets {
        let config = DistillationConfig::for_error_rate(target);
        let budget = MagicStateBudget::for_qram_routing(depth, 1).with_distillation(&config);

        println!(
            "{:>12.0e} {:>8} {:>10} {:>12}:1 {:>15} {:>15}",
            target,
            config.levels,
            config.code_distance,
            config.overhead_ratio(),
            format_number(budget.total_magic_states),
            format_number(budget.factory_qubits)
        );
    }

    // Show achieved error rates
    println!("\n--- Achieved Error Rates (physical: 10⁻³) ---\n");
    println!("{:>12} {:>8} {:>20}", "Target", "Levels", "Achieved");
    println!("{}", "-".repeat(45));

    for target in targets {
        let config = DistillationConfig::for_error_rate(target);
        let achieved = config.achieved_error_rate();

        println!("{:>12.0e} {:>8} {:>20.2e}", target, config.levels, achieved);
    }
}

/// Compare different distillation protocols
fn distillation_protocol_comparison() {
    let protocols = [
        ("15-to-1 (Standard)", DistillationProtocol::fifteen_to_one()),
        ("10-to-2 (Improved)", DistillationProtocol::ten_to_two()),
        ("Reed-Muller", DistillationProtocol::reed_muller()),
        (
            "7-to-1 (Low overhead)",
            DistillationProtocol::seven_to_one(),
        ),
    ];

    println!("Single-level protocol comparison:\n");
    println!(
        "{:>20} {:>8} {:>8} {:>12} {:>15} {:>12}",
        "Protocol", "In", "Out", "Qubits", "Cycles", "Efficiency"
    );
    println!("{}", "-".repeat(80));

    for (name, protocol) in &protocols {
        let efficiency = protocol.output_states as f64
            / (protocol.input_states as f64 * protocol.cycle_overhead as f64);

        println!(
            "{:>20} {:>8} {:>8} {:>12} {:>15} {:>12.4}",
            name,
            protocol.input_states,
            protocol.output_states,
            protocol.qubit_overhead,
            protocol.cycle_overhead,
            efficiency
        );
    }

    // Multi-level factory comparison
    println!("\n--- Multi-Level Factory Comparison (target: 10⁻¹²) ---\n");

    let n_states = 100; // Need 100 clean magic states
    let target = 1e-12;

    println!(
        "{:>20} {:>10} {:>15} {:>15} {:>15}",
        "Protocol", "Levels", "Total Qubits", "Total Cycles", "Noisy States"
    );
    println!("{}", "-".repeat(80));

    // 15-to-1 factory (standard)
    let factory_15 = DistillationFactory::for_target_error(1e-3, target);
    let resources_15 = factory_15.resources_for(n_states);

    println!(
        "{:>20} {:>10} {:>15} {:>15} {:>15}",
        "15-to-1 (standard)",
        factory_15.num_levels(),
        format_number(resources_15.total_qubits),
        format_number(resources_15.total_cycles),
        format_number(resources_15.noisy_states_consumed)
    );

    // Optimized factory (may use 10-to-2 where efficient)
    let factory_opt = DistillationFactory::optimized(1e-3, target);
    let resources_opt = factory_opt.resources_for(n_states);

    println!(
        "{:>20} {:>10} {:>15} {:>15} {:>15}",
        "Optimized",
        factory_opt.num_levels(),
        format_number(resources_opt.total_qubits),
        format_number(resources_opt.total_cycles),
        format_number(resources_opt.noisy_states_consumed)
    );
}

/// Analyze different Toffoli decomposition methods
fn toffoli_decomposition_analysis() {
    let decompositions = [
        ("Standard (Nielsen-Chuang)", ToffoliDecomposition::Standard),
        ("CCZ-Based", ToffoliDecomposition::CCZBased),
        ("Relative-Phase", ToffoliDecomposition::RelativePhase),
        (
            "Magic State Injection",
            ToffoliDecomposition::MagicStateInjection,
        ),
        ("Measurement-Based", ToffoliDecomposition::MeasurementBased),
    ];

    println!(
        "{:>30} {:>10} {:>10} {:>10}",
        "Decomposition", "T-gates", "CNOTs", "T-depth"
    );
    println!("{}", "-".repeat(65));

    for (name, decomp) in &decompositions {
        println!(
            "{:>30} {:>10} {:>10} {:>10}",
            name,
            decomp.t_count(),
            decomp.cnot_count(),
            decomp.t_depth()
        );
    }

    // Track Toffolis in different contexts
    println!("\n--- Context-Aware Tracking Example ---\n");

    let mut tracker = ToffoliTracker::with_decomposition(ToffoliDecomposition::Standard);

    // Add routing Toffolis (simulating depth=4 QRAM routing)
    for level in 0..4 {
        for router in 0..(1 << level) {
            tracker.track_routing(0, 1, 2, level, router);
        }
    }

    // Add some data operation Toffolis
    tracker.track_data_operation(0, 1, 2, "multiply");
    tracker.track_data_operation(0, 1, 2, "add");

    println!("{}", tracker);
    println!("  Routing Toffolis: {}", tracker.routing_count());
    println!("  Total T-gates: {}", tracker.t_count());
    println!("  Total T-depth: {}", tracker.t_depth());

    // Convert to budget
    let budget = tracker.to_budget();
    println!("\n  As MagicStateBudget:");
    println!("    Routing T-gates: {}", budget.routing_t_gates);
    println!("    Data T-gates: {}", budget.data_t_gates);
}

/// Demo StabilizerQRAM integration with magic state budgeting
fn stabilizer_qram_demo() {
    // Create and use stabilizer QRAM
    let mut qram = StabilizerQRAM::new(8); // 256 cells
    let data: Vec<u64> = (0..256).map(|i| i * 100).collect();
    qram.load(&data);

    println!("Created StabilizerQRAM with 256 cells (depth=8)\n");

    // Perform some queries
    for i in 0..5 {
        let _ = qram.query_with_frame(i * 50, i as u64);
    }

    println!("Performed 5 queries with Pauli frame tracking\n");

    // Get magic state budget
    let budget = qram.magic_state_budget();
    println!("--- Current Magic State Budget ---");
    println!("  Toffoli gates tracked: {}", budget.toffoli_count);
    println!("  Bare T-count: {}", budget.total_t_count_bare);
    println!("  Routing T-gates: {}", budget.routing_t_gates);

    // Estimate for more queries
    println!("\n--- Budget Estimate for 1000 Queries ---");
    let estimate = qram.estimate_budget(1000, 1e-12);
    println!("  Total T-count: {}", estimate.total_t_count_bare);
    println!(
        "  Magic states (with distillation): {}",
        estimate.total_magic_states
    );
    println!("  Factory qubits: {}", estimate.factory_qubits);
    println!(
        "  Estimated time: {}",
        format_time(estimate.estimated_seconds)
    );

    // Full resource comparison
    println!("\n--- Full Resource Comparison ---");
    let comparison = qram.full_resource_comparison(1e-12);
    println!("{}", comparison);
}

/// Compare with literature estimates
fn literature_comparison() {
    println!("Comparing with estimates from key papers:\n");

    // Gidney & Ekera (RSA 2048)
    println!("1. Gidney & Ekera (2021) - RSA 2048 factoring");
    println!("   Paper: 20 million noisy qubits, 8 hours");
    println!("   Their T-count: ~10^9");

    let rsa_toffolis = 140_000_000; // ~140M Toffolis estimated for RSA
    let rsa_budget = MagicStateBudget::from_gate_counts(rsa_toffolis, 0, 0)
        .with_distillation(&DistillationConfig::for_error_rate(1e-15));

    println!("   Our estimate:");
    println!(
        "     T-gates: {}",
        format_number(rsa_budget.total_t_count_bare)
    );
    println!(
        "     Magic states: {}",
        format_large_number(rsa_budget.total_magic_states as u64)
    );
    println!(
        "     Factory qubits: {}",
        format_number(rsa_budget.factory_qubits)
    );
    println!();

    // QRAM estimates from Giovannetti et al.
    println!("2. Giovannetti et al. (2008) - Bucket-Brigade QRAM");
    println!("   For N cells: O(log N) Toffoli gates per query");
    println!("   Our implementation matches this:");

    for depth in [8, 12, 16] {
        let budget = MagicStateBudget::for_qram_routing(depth, 1);
        println!(
            "     Depth {}: {} cells, {} Toffolis",
            depth,
            1 << depth,
            budget.toffoli_count
        );
    }
    println!();

    // Surface code factory estimates
    println!("3. Surface Code Factory Estimates (Fowler et al.)");
    println!("   15-to-1 protocol: ~20 qubits per level");
    println!("   Our model:");

    for levels in 1..=3 {
        let config = DistillationConfig::with_levels(1e-3, 1e-15, levels);
        let factory = config.factory_estimate(1);
        println!(
            "     {} level(s): {} qubits, {}:1 overhead",
            levels,
            factory.qubits,
            config.overhead_ratio()
        );
    }
}

// Helper formatting functions

fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

fn format_large_number(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.2}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

fn format_time(seconds: f64) -> String {
    if seconds >= 86400.0 {
        format!("{:.1} days", seconds / 86400.0)
    } else if seconds >= 3600.0 {
        format!("{:.1} hours", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1} min", seconds / 60.0)
    } else if seconds >= 1.0 {
        format!("{:.1} sec", seconds)
    } else if seconds >= 0.001 {
        format!("{:.1} ms", seconds * 1000.0)
    } else {
        format!("{:.1} µs", seconds * 1_000_000.0)
    }
}
