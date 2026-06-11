//! HLS-generated placement instances (Sprint 383, Phase 4).
//!
//! The HLS middle layer turns any C-like function into a datapath netlist — which
//! makes it an **endless, parameterized generator of placement problems** for the
//! AlphaFabric environment: sweep the source shape and the bit width and every
//! point is a fresh, labeled instance. The full provenance chain is preserved:
//!
//! ```text
//! source (C-like) → AIG (synth::hls) → mapped netlist → placement → reward
//! ```
//!
//! That chain is also a clean ground-truth corpus for the interpretability-testbed
//! framing: each instance's causal structure is known *by construction* from the
//! source that generated it.
//!
//! Correctness gate: placement never changes logic; the honest invariant is
//! **routable ⟹ correct**, checked with [`verify_physical`](super::env::verify_physical)
//! on real tiles.

use super::circuit::Circuit;
use crate::synth::hls::{HlsConfig, synthesize_func};
use crate::tile_cpu::v2_parser::parse_program;

/// One HLS-generated placement instance with its provenance label.
pub struct HlsInstance {
    /// The C-like source that generated this instance.
    pub source: &'static str,
    /// Function name within the source (also the instance family).
    pub func_name: &'static str,
    /// Datapath bit width this instance was synthesized at.
    pub width: u32,
    /// The placement problem: AIG + mapped netlist + library.
    pub circuit: Circuit,
}

/// The source shapes swept by [`hls_instances`]. Straight-line datapath functions
/// (the combinational HLS subset), chosen to vary structure: carry chains, an
/// array multiplier, bitwise mixing, and a shared-subexpression polynomial.
const SHAPES: &[(&str, &str, &[u32])] = &[
    (
        "int acc(int a, int b) { return a + b; }",
        "acc",
        &[2, 4, 6, 8],
    ),
    (
        "int mac(int a, int b, int c) { return a * b + c; }",
        "mac",
        &[2, 3, 4],
    ),
    (
        "int mix(int a, int b, int c) { int t = a ^ b; return (t & c) | (a & b); }",
        "mix",
        &[2, 4, 6, 8],
    ),
    (
        "int horner(int x, int c0, int c1) { int t = c1 * x + c0; return t * x; }",
        "horner",
        &[2, 3],
    ),
];

/// Generate the HLS placement-instance sweep: every shape in [`SHAPES`] at every
/// listed width. Deterministic; each instance's `Circuit` is mapped and ready for
/// [`PlacementEnv`](super::env::PlacementEnv).
pub fn hls_instances() -> Vec<HlsInstance> {
    let mut out = Vec::new();
    for &(source, func_name, widths) in SHAPES {
        let program = parse_program(source).expect("corpus shape parses");
        let func = program
            .funcs
            .iter()
            .find(|f| f.name == func_name)
            .expect("corpus shape contains its function");
        for &width in widths {
            let aig = synthesize_func(func, &HlsConfig { width })
                .expect("combinational corpus shape synthesizes");
            let name = format!("hls_{func_name}{width}");
            out.push(HlsInstance {
                source,
                func_name,
                width,
                circuit: Circuit::from_aig(name, aig),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::alphafabric::{AnnealConfig, PlacementEnv, anneal};

    #[test]
    fn hls_instances_build_and_map_correctly() {
        let instances = hls_instances();
        assert!(instances.len() >= 10, "sweep yields a real corpus");
        for inst in &instances {
            assert!(
                inst.circuit.num_gates() > 0,
                "{}: no gates",
                inst.circuit.name
            );
            // Exhaustive netlist-vs-AIG equivalence where cheap.
            if inst.circuit.num_inputs() <= 12 {
                assert!(
                    inst.circuit.mapping_is_correct(),
                    "{}: mapped incorrectly",
                    inst.circuit.name
                );
            }
        }
    }

    /// The Phase 4 gate: an HLS-generated instance goes through the AlphaFabric
    /// loop (baseline → SA), the result is no worse than the baseline, and the
    /// optimized placement still computes the source function on real tiles.
    #[test]
    fn hls_instance_anneals_and_stays_physically_correct() {
        let instances = hls_instances();
        // Two small, structurally different instances keep the test fast.
        let picks: Vec<_> = instances
            .iter()
            .filter(|i| i.circuit.name == "hls_acc4" || i.circuit.name == "hls_mix2")
            .collect();
        assert_eq!(picks.len(), 2, "expected corpus members present");

        for inst in picks {
            let mut env = PlacementEnv::new(&inst.circuit)
                .unwrap_or_else(|e| panic!("{}: env: {e:?}", inst.circuit.name));
            let config = AnnealConfig {
                iterations: 800,
                ..AnnealConfig::default()
            };
            let r = anneal(&mut env, &config);
            assert!(
                r.best.cost <= r.baseline.cost,
                "{}: SA worsened cost ({} -> {})",
                inst.circuit.name,
                r.baseline.cost,
                r.best.cost
            );
            assert!(
                r.best.metrics.routable,
                "{}: best layout unroutable",
                inst.circuit.name
            );
            // Physical authority: the re-placed datapath still computes correctly.
            assert!(
                env.verify_physical(),
                "{}: annealed placement is not physically correct",
                inst.circuit.name
            );
        }
    }
}
