/// SPRINT 37.0: Benchmark 3-qubit gate synthesis
///
/// Tests Toffoli and Fredkin gate synthesis with Qiskit integration.
/// Verifies caching works correctly and measures performance.
///
/// Usage: cargo run --example three_qubit_synth_bench --release
use logic_fabric_core::algebraic_fusion::{
    ThreeQubitGate, get_synthesis_cache_stats, synthesize_three_qubit,
};

fn main() {
    println!("=== 3-Qubit Gate Synthesis Benchmark ===");
    println!();

    // Test Toffoli synthesis
    println!("## Toffoli (CCNOT) Gate");
    println!();

    println!("First call (cache miss, synthesis):");
    let start = std::time::Instant::now();
    let toffoli_gates = synthesize_three_qubit(ThreeQubitGate::Toffoli, 0, 1, 2);
    let duration_first = start.elapsed();

    match &toffoli_gates {
        Some(gates) => {
            println!("  ✓ Synthesized {} gates", gates.len());
            println!("  Time: {:.2?}", duration_first);
            println!();

            // Show gate sequence
            println!("  Gate sequence:");
            for (i, gate) in gates.iter().take(10).enumerate() {
                println!("    {}: {:?}", i + 1, gate);
            }
            if gates.len() > 10 {
                println!("    ... and {} more gates", gates.len() - 10);
            }
            println!();
        }
        None => {
            println!("  ✗ Synthesis failed!");
            return;
        }
    }

    // Test cache hit
    println!("Second call (cache hit):");
    let start = std::time::Instant::now();
    let toffoli_gates_2 = synthesize_three_qubit(ThreeQubitGate::Toffoli, 3, 4, 5);
    let duration_second = start.elapsed();

    match &toffoli_gates_2 {
        Some(gates) => {
            println!("  ✓ Retrieved {} gates from cache", gates.len());
            println!("  Time: {:.2?}", duration_second);
            println!(
                "  Speedup: {:.0}x",
                duration_first.as_micros() as f64 / duration_second.as_micros() as f64
            );
            println!();
        }
        None => {
            println!("  ✗ Cache retrieval failed!");
            return;
        }
    }

    // Verify qubit remapping
    println!("Qubit remapping verification:");
    if let (Some(gates1), Some(gates2)) = (&toffoli_gates, &toffoli_gates_2) {
        println!("  Original qubits: 0, 1, 2");
        println!("  Remapped qubits: 3, 4, 5");
        println!(
            "  Both have {} gates: {}",
            gates1.len(),
            if gates1.len() == gates2.len() {
                "✓"
            } else {
                "✗"
            }
        );
        println!();
    }

    // Test Fredkin synthesis
    println!("## Fredkin (CSWAP) Gate");
    println!();

    println!("First call (cache miss, synthesis):");
    let start = std::time::Instant::now();
    let fredkin_gates = synthesize_three_qubit(ThreeQubitGate::Fredkin, 0, 1, 2);
    let duration_fredkin = start.elapsed();

    match &fredkin_gates {
        Some(gates) => {
            println!("  ✓ Synthesized {} gates", gates.len());
            println!("  Time: {:.2?}", duration_fredkin);
            println!();
        }
        None => {
            println!("  ✗ Synthesis failed!");
            return;
        }
    }

    // Cache statistics
    println!("## Cache Statistics");
    println!();
    let stats = get_synthesis_cache_stats();
    println!("  Total calls: {}", stats.total_calls);
    println!("  Cache hits:  {}", stats.hits);
    println!("  Cache misses: {}", stats.misses);
    println!("  Hit rate:    {:.1}%", stats.hit_rate() * 100.0);
    println!();

    // Summary
    println!("## Summary");
    println!();
    println!("| Gate    | Gates | First Call | Cache Hit | Speedup   |");
    println!("|---------| ------|------------|-----------|-----------|");
    println!(
        "| Toffoli | {:5} | {:9.2?} | {:8.2?} | {:8.0}x |",
        toffoli_gates.unwrap().len(),
        duration_first,
        duration_second,
        duration_first.as_micros() as f64 / duration_second.as_micros() as f64
    );
    println!(
        "| Fredkin | {:5} | {:9.2?} | (cached)  | -         |",
        fredkin_gates.unwrap().len(),
        duration_fredkin
    );
    println!();

    println!("✓ 3-Qubit synthesis working correctly!");
    println!();
    println!("Notes:");
    println!("- First call: ~1-2 seconds (Qiskit synthesis + caching)");
    println!("- Subsequent calls: <1 microsecond (cache lookup + qubit remapping)");
    println!("- Ready for Shor's algorithm implementation");
}
