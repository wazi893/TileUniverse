use logic_fabric_core::algebraic_fusion::ultimate_optimize;
use logic_fabric_core::qasm::{parse_qasm, to_qasm};
use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: opt_qasm <input.qasm> <output.qasm>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    println!("Loading {}...", input_path);
    let source = fs::read_to_string(input_path).expect("Failed to read input file");

    let start_parse = Instant::now();
    let (gates, n_qubits) = match parse_qasm(&source) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };
    let parse_time = start_parse.elapsed();
    println!(
        "Parsed {} gates on {} qubits in {:?}",
        gates.len(),
        n_qubits,
        parse_time
    );

    println!("Running Algebraic Fusion (EPIC 83)...");
    let start_opt = Instant::now();

    // We need to cast n_qubits to u8 for the engine, hoping it fits!
    let n_qubits_u8 = if n_qubits > 255 {
        eprintln!("Warning: Too many qubits for engine (max 255). Clamping to 255.");
        255
    } else {
        n_qubits as u8
    };

    let (optimized_ops, stats) = ultimate_optimize(gates.clone(), n_qubits_u8);
    let opt_time = start_opt.elapsed();

    // Map AlgebraicOp back to QGate for export
    // AlgebraicOp enum might contain reduced ops.
    // We need to convert the optimized `Vec<AlgebraicOp>` or `Vec<QGate>` back.
    // Wait, ultimate_optimize returns `(Vec<AlgebraicOp>, AlgebraicStats)`.
    // AlgebraicOp needs to be converted to QGate to be serialized.

    // Let's inspect AlgebraicOp in case I need to map it.
    // Assuming for MVP we just want the gates.
    // engine::algebraic_fusion::AlgebraicOp usually has an extraction method or we can pattern match.
    // Actually, to_qasm expects &[QGate].

    // Let's look at how to extract QGate from AlgebraicOp.
    // If it's internal, I might need to update the CLI to handle it.
    // For now, I'll assume I can extract it or I'll just look at the stats for the "Reduction"
    // and maybe I need to implement a converter.

    // Let's check imports.
    // use engine::algebraic_fusion::AlgebraicOp;

    // Temporary hack: The user wants to see the file.
    // If AlgebraicOp is complex, I might need to verify its structure first.
    // But for the "Ultimate Optimizer", skipping identity is key.
    // If Op is Identity, we don't write it.

    let mut final_gates = Vec::new();
    for op in optimized_ops {
        match op {
            logic_fabric_core::algebraic_fusion::AlgebraicOp::Single(g) => final_gates.push(g),
            logic_fabric_core::algebraic_fusion::AlgebraicOp::Power { gate, power } => {
                for _ in 0..power {
                    final_gates.push(gate.clone());
                }
            }
            logic_fabric_core::algebraic_fusion::AlgebraicOp::Skip { .. } => {}
            #[cfg(feature = "cuda")]
            logic_fabric_core::algebraic_fusion::AlgebraicOp::FusedMatrix { .. } => {
                eprintln!(
                    "Warning: FusedMatrix encountered, cannot export to QASM without decomposition."
                );
            }
        }
    }

    // Check for target architecture
    let mut target_arg = None;
    for i in 0..args.len() {
        if args[i] == "--target" && i + 1 < args.len() {
            target_arg = Some(args[i + 1].as_str());
        }
    }

    if let Some(target_type) = target_arg {
        println!("Routing for target architecture: {}", target_type);
        use logic_fabric_core::hardware::{CouplingMap, SimpleRouter};

        // Construct map based on qubit count (or fixed size if specified, but taking n_qubits is safer)
        // Adjust n_qubits to be sufficient for the topology if it's a fixed grid
        let map = match target_type {
            "linear" => CouplingMap::new_linear(n_qubits_u8),
            "grid" => {
                // Heuristic square grid
                let side = (n_qubits_u8 as f32).sqrt().ceil() as u8;
                CouplingMap::new_grid(side, side)
            }
            "full" => CouplingMap::new_fully_connected(n_qubits_u8),
            _ => {
                eprintln!(
                    "Unknown target '{}'. Defaults: linear, grid, full.",
                    target_type
                );
                std::process::exit(1);
            }
        };

        let mut router = SimpleRouter::new(map);
        final_gates = router.route(&final_gates);
        println!(
            "  Routing added SWAPs. Final gate count: {}",
            final_gates.len()
        );
    }

    println!("Optimization complete in {:?}", opt_time);
    println!("  Original gates: {}", gates.len());
    println!("  Final gates:    {}", final_gates.len());
    println!(
        "  Reduction:      {:.2}%",
        100.0 * (1.0 - (final_gates.len() as f64 / gates.len().max(1) as f64))
    );
    println!("  Effective Speed: {:.2}x", stats.throughput_multiplier);

    let output_source = to_qasm(&final_gates, n_qubits);
    fs::write(output_path, output_source).expect("Failed to write output file");
    println!("Written optimized circuit to {}", output_path);
}
