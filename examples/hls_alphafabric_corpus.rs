//! HLS → AlphaFabric: a labeled placement corpus from source code (Sprint 383, Phase 4).
//!
//! The HLS middle layer makes the front-end a *generator of placement problems*:
//! every (source shape × bit width) point becomes a fresh netlist, which the
//! AlphaFabric environment places, the SA baseline optimizes, and the reward
//! function scores. Each printed row is one corpus record:
//!
//! ```text
//! (source, width) → netlist (gates) → placement (hpwl, wire) → reward (cost) [+ phys gate]
//! ```
//!
//! Because the netlist's causal structure is known by construction from the
//! source, this doubles as a ground-truth dataset for interpretability work.
//!
//! Run: `cargo run --release --example hls_alphafabric_corpus`

use engine::synth::alphafabric::{AnnealConfig, PlacementEnv, anneal, hls_instances};
use rayon::prelude::*;
use std::time::Instant;

struct CorpusRow {
    line: String,
    verified: usize,
    unroutable: usize,
    violations: usize,
}

fn main() {
    let instances = hls_instances();
    let config = AnnealConfig {
        iterations: 2000,
        ..AnnealConfig::default()
    };

    println!(
        "HLS-generated placement corpus ({} instances, SA {} iters each)\n",
        instances.len(),
        config.iterations
    );
    println!(
        "{:<12} {:>5} {:>6} {:>13} {:>13} {:>13} {:>8}",
        "instance", "width", "gates", "hpwl(b>best)", "wire(b>best)", "cost(b>best)", "phys_ok"
    );
    println!("{}", "-".repeat(80));

    let t0 = Instant::now();
    let rows: Vec<_> = instances
        .par_iter()
        .map(|inst| {
            let mut env = match PlacementEnv::new(&inst.circuit) {
                Ok(env) => env,
                Err(e) => {
                    return CorpusRow {
                        line: format!(
                            "{:<12} {:>5} skipped (env: {e:?})",
                            inst.circuit.name, inst.width
                        ),
                        verified: 0,
                        unroutable: 0,
                        violations: 0,
                    };
                }
            };
            let r = anneal(&mut env, &config);

            // Three record labels: verified (routable + correct on tiles),
            // unroutable (an honest negative example for a learned placer), and
            // INTEGRITY VIOLATION (routable but physically wrong — a router defect,
            // see the Sprint 383 capacity-boundary finding in synth::hls_seq).
            let (label, verified, unroutable, violations) = if !r.best.metrics.routable {
                ("n/r", 0, 1, 0)
            } else if env.verify_physical() {
                ("ok", 1, 0, 0)
            } else {
                ("VIOLATION", 0, 0, 1)
            };

            let wire = |m: &engine::synth::alphafabric::PlacementMetrics| {
                if m.routable {
                    m.wire_tiles.to_string()
                } else {
                    "n/r".to_string()
                }
            };
            let cost = |c: f64| {
                if c >= 1e9 {
                    "n/r".to_string()
                } else {
                    format!("{c:.0}")
                }
            };
            CorpusRow {
                line: format!(
                    "{:<12} {:>5} {:>6} {:>6}>{:<6} {:>6}>{:<6} {:>6}>{:<6} {:>9}",
                    inst.circuit.name,
                    inst.width,
                    inst.circuit.num_gates(),
                    r.baseline_hpwl,
                    r.best_hpwl,
                    wire(&r.baseline.metrics),
                    wire(&r.best.metrics),
                    cost(r.baseline.cost),
                    cost(r.best.cost),
                    label,
                ),
                verified,
                unroutable,
                violations,
            }
        })
        .collect();

    for row in &rows {
        println!("{}", row.line);
    }
    let verified: usize = rows.iter().map(|row| row.verified).sum();
    let unroutable: usize = rows.iter().map(|row| row.unroutable).sum();
    let violations: usize = rows.iter().map(|row| row.violations).sum();

    println!("{}", "-".repeat(80));
    println!(
        "{} records in {:?}: {verified} physically verified, {unroutable} unroutable \
         (labeled negatives), {violations} integrity violations",
        instances.len(),
        t0.elapsed(),
    );
    if violations > 0 {
        println!(
            "NOTE: an integrity violation = routable layout that fails physical \
             verification — the place&route correctness defect documented in \
             synth::hls_seq (HLS-generated instances reproduce it at ~50 gates)."
        );
    }
    println!(
        "\nProvenance per record: C-like source → HLS AIG → mapped netlist → placement → reward."
    );
}
