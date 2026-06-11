//! Hardware Mapping Benchmarks
//!
//! SPRINT 52.0: Hardware-Aware Resource Estimation
//!
//! Compares magic state costs across different hardware platforms:
//! 1. IBM Falcon/Heron (superconducting, heavy-hex)
//! 2. IonQ Aria (trapped ion, all-to-all)
//! 3. Google Sycamore (superconducting, grid)
//! 4. QuEra Aquila (neutral atom, 2D array)
//!
//! Run: cargo run --release --example hardware_mapping_bench

use engine::hardware::{
    DeviceTopology, FactoryFootprint, FactoryLayoutOptimizer, HardwareAwareBudget, SwapRouter,
    format_hardware_comparison, profiles,
};
use engine::qram::{DistillationConfig, MagicStateBudget};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║      SPRINT 52.0: Hardware Mapping Benchmarks                        ║");
    println!("║      Fault-Tolerant QRAM on Real Hardware Topologies                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Part 1: Device Topology Comparison
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Part 1: Device Topology Comparison");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    topology_comparison();

    // Part 2: Hardware Profile Comparison
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 2: Hardware Profile Comparison");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    profile_comparison();

    // Part 3: Factory Layout on Different Devices
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 3: Factory Layout on Different Devices");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    factory_layout_comparison();

    // Part 4: SWAP Routing Analysis
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 4: SWAP Routing Analysis");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    swap_routing_analysis();

    // Part 5: Full Hardware-Aware Budget Comparison
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 5: Hardware-Aware Budget Comparison (QRAM 256 cells, 100 queries)");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    hardware_aware_comparison();

    // Part 6: Scaling Analysis
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Part 6: Scaling Analysis (vary query count)");
    println!("═══════════════════════════════════════════════════════════════════════\n");
    scaling_analysis();

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("SPRINT 52.0 Hardware Mapping Complete!");
    println!("═══════════════════════════════════════════════════════════════════════");
}

fn topology_comparison() {
    let topologies = [
        ("Grid 5x5", DeviceTopology::grid_2d(5, 5)),
        ("Grid 8x8", DeviceTopology::grid_2d(8, 8)),
        ("Linear 25", DeviceTopology::linear_chain(25)),
        ("Ring 25", DeviceTopology::ring(25)),
        ("Star 24", DeviceTopology::star(24)),
        ("Full 25", DeviceTopology::fully_connected(25)),
    ];

    println!(
        "{:>15} {:>8} {:>8} {:>10} {:>10} {:>12}",
        "Topology", "Qubits", "Edges", "MaxDegree", "Diameter", "AvgDistance"
    );
    println!("{}", "-".repeat(70));

    for (name, topo) in &topologies {
        let coupling = topo.to_coupling_map();
        println!(
            "{:>15} {:>8} {:>8} {:>10} {:>10} {:>12.2}",
            name,
            topo.qubit_count(),
            topo.edge_count(),
            topo.max_degree(),
            coupling.diameter(),
            coupling.average_distance()
        );
    }
}

fn profile_comparison() {
    let profiles = vec![
        profiles::ibm_falcon(),
        profiles::ibm_heron(),
        profiles::google_sycamore(),
        profiles::ionq_aria(),
        profiles::quantinuum_h2(),
        profiles::quera_aquila(),
    ];

    println!(
        "{:>15} {:>8} {:>12} {:>10} {:>10} {:>12} {:>10}",
        "Device", "Qubits", "Technology", "2Q Fid", "2Q Time", "T2 (µs)", "SWAP Cost"
    );
    println!("{}", "-".repeat(90));

    for p in &profiles {
        let tech = format!("{:?}", p.technology);
        println!(
            "{:>15} {:>8} {:>12} {:>10.4} {:>10.0} {:>12.0} {:>10}",
            p.name,
            p.qubit_count,
            &tech[..tech.len().min(12)],
            p.t2q_fidelity,
            p.t2q_time_ns,
            p.t2_us,
            p.swap_cost
        );
    }

    println!("\n--- Routing Metrics ---\n");
    println!(
        "{:>15} {:>12} {:>15} {:>15}",
        "Device", "MaxDistance", "AvgSWAPOverhead", "FullConnect"
    );
    println!("{}", "-".repeat(60));

    for p in &profiles {
        println!(
            "{:>15} {:>12} {:>15.2} {:>15}",
            p.name,
            p.max_routing_distance(),
            p.average_swap_overhead(),
            if p.has_full_connectivity() {
                "Yes"
            } else {
                "No"
            }
        );
    }
}

fn factory_layout_comparison() {
    let footprint = FactoryFootprint::fifteen_to_one();
    println!(
        "Factory: {} ({}x{} = {} qubits)\n",
        footprint.name,
        footprint.width,
        footprint.height,
        footprint.qubit_count()
    );

    let profiles = vec![
        ("IBM Falcon", profiles::ibm_falcon()),
        ("IBM Heron", profiles::ibm_heron()),
        ("IonQ Aria", profiles::ionq_aria()),
        ("QuEra Aquila", profiles::quera_aquila()),
    ];

    println!(
        "{:>15} {:>10} {:>12} {:>12} {:>15} {:>12}",
        "Device", "MaxFact", "Placed", "SWAPOvhd", "RoutingCyc", "Utilization"
    );
    println!("{}", "-".repeat(80));

    for (name, profile) in &profiles {
        let optimizer = FactoryLayoutOptimizer::new(profile);
        let max_factories = optimizer.max_factories(&footprint);
        let placements = optimizer.place_factories(&footprint, 2.min(max_factories));
        let resources = optimizer.total_resources(&placements);

        println!(
            "{:>15} {:>10} {:>12} {:>12} {:>15} {:>12.1}%",
            name,
            max_factories,
            resources.factory_count,
            resources.total_swaps,
            resources.total_routing_cycles,
            resources.device_utilization * 100.0
        );
    }
}

fn swap_routing_analysis() {
    let profiles = vec![
        (
            "Grid 4x4",
            profiles::custom_superconducting("Grid4x4", 4, 4, 0.99),
        ),
        (
            "Grid 8x8",
            profiles::custom_superconducting("Grid8x8", 8, 8, 0.99),
        ),
        ("IonQ 25", profiles::ionq_aria()),
    ];

    println!("Routing random gate pairs (1000 pairs):\n");
    println!(
        "{:>15} {:>12} {:>12} {:>12} {:>15}",
        "Device", "TotalSWAPs", "AvgDist", "Adjacent", "OverheadRatio"
    );
    println!("{}", "-".repeat(70));

    for (name, profile) in &profiles {
        let router = SwapRouter::new(&profile);
        let n = profile.qubit_count;

        // Generate random pairs
        let pairs: Vec<_> = (0..1000)
            .map(|i| {
                let q1 = (i * 7) % n;
                let q2 = (i * 13 + 5) % n;
                (q1, q2)
            })
            .filter(|(a, b)| a != b)
            .collect();

        let stats = router.calculate_overhead(&pairs);
        println!(
            "{:>15} {:>12} {:>12.2} {:>12} {:>15.2}",
            name,
            stats.total_swaps,
            stats.average_distance,
            stats.adjacent_count,
            stats.overhead_ratio
        );
    }
}

fn hardware_aware_comparison() {
    // Create abstract budget
    let budget = MagicStateBudget::for_qram_routing(8, 100)
        .with_distillation(&DistillationConfig::for_error_rate(1e-12));

    println!("Abstract Budget:");
    println!("  QRAM depth: 8 (256 cells)");
    println!("  Queries: 100");
    println!("  Target error: 10⁻¹²");
    println!("  T-gates: {}", budget.total_t_count_bare);
    println!("  Magic states: {}", budget.total_magic_states);
    println!(
        "  Abstract factory cycles: {}\n",
        budget.total_factory_cycles
    );

    let all_profiles = [
        profiles::ibm_falcon(),
        profiles::ibm_heron(),
        profiles::ionq_aria(),
        profiles::google_sycamore(),
    ];

    let hw_budgets: Vec<_> = all_profiles
        .iter()
        .map(|p| HardwareAwareBudget::from_budget(&budget, p, 1))
        .collect();

    println!("{}", format_hardware_comparison(&hw_budgets));

    // Detailed breakdown for IBM Heron
    println!("\n--- Detailed: IBM Heron ---");
    let heron_budget = &hw_budgets[1];
    println!("{}", heron_budget);
}

fn scaling_analysis() {
    let queries = [1, 10, 100, 1000];
    let profile = profiles::ibm_heron();

    println!("Device: IBM Heron (133 qubits)\n");
    println!(
        "{:>10} {:>12} {:>15} {:>12} {:>12} {:>12}",
        "Queries", "T-gates", "MagicStates", "PhysQubits", "SWAPs", "Time(s)"
    );
    println!("{}", "-".repeat(80));

    for &q in &queries {
        let budget = MagicStateBudget::for_qram_routing(8, q)
            .with_distillation(&DistillationConfig::for_error_rate(1e-12));

        let hw_budget = HardwareAwareBudget::from_budget(&budget, &profile, 2);

        println!(
            "{:>10} {:>12} {:>15} {:>12} {:>12} {:>12.4}",
            q,
            format_number(budget.total_t_count_bare),
            format_number(budget.total_magic_states),
            hw_budget.physical_qubits,
            hw_budget.swap_gates,
            hw_budget.adjusted_time_seconds
        );
    }

    // Compare superconducting vs ion trap for large queries
    println!("\n--- Superconducting vs Ion Trap (1000 queries) ---\n");

    let budget_1000 = MagicStateBudget::for_qram_routing(8, 1000)
        .with_distillation(&DistillationConfig::for_error_rate(1e-12));

    let sc_budget = HardwareAwareBudget::from_budget(&budget_1000, &profiles::ibm_heron(), 2);
    let ion_budget = HardwareAwareBudget::from_budget(&budget_1000, &profiles::ionq_aria(), 1);

    println!("IBM Heron (superconducting):");
    println!("  Physical qubits: {}", sc_budget.physical_qubits);
    println!("  SWAP gates: {}", sc_budget.swap_gates);
    println!("  Adjusted time: {:.3}s", sc_budget.adjusted_time_seconds);
    println!("  Estimated fidelity: {:.4}", sc_budget.estimated_fidelity);
    println!("  Bottleneck: {}", sc_budget.bottleneck);

    println!("\nIonQ Aria (trapped ion):");
    println!("  Physical qubits: {}", ion_budget.physical_qubits);
    println!("  SWAP gates: {}", ion_budget.swap_gates);
    println!("  Adjusted time: {:.3}s", ion_budget.adjusted_time_seconds);
    println!("  Estimated fidelity: {:.4}", ion_budget.estimated_fidelity);
    println!("  Bottleneck: {}", ion_budget.bottleneck);

    let comparison = sc_budget.compare_to(&ion_budget);
    println!("\nComparison: {}", comparison);
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}
